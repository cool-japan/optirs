// Progressive Neural Architecture Search
//
// Implements a progressive search strategy that starts with simple architectures
// and gradually increases complexity through phases. Each phase builds upon
// the best architectures found in previous phases.
//
// Reference: Liu et al., "Progressive Neural Architecture Search" (ECCV 2018)

use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::architecture::{ComponentPosition, ComponentType};
#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::config::{ComponentType as ConfigComponentType, ParameterRange};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::{SearchStrategy, SearchStrategyStatistics};

/// Progressive NAS search strategy
///
/// Searches architectures in phases of increasing complexity.
/// Phase 0 starts with the simplest architectures (fewest components),
/// and each subsequent phase allows more complex architectures.
/// Only top-performing architectures from each phase are expanded
/// into the next phase.
pub struct ProgressiveNAS<T: Float + Debug + Send + Sync + 'static> {
    /// Current search phase (0-indexed)
    current_phase: usize,
    /// Maximum number of phases
    max_phases: usize,
    /// Architectures found in each phase, indexed by phase
    phase_architectures: Vec<Vec<OptimizerArchitecture<T>>>,
    /// Maximum complexity (number of components) allowed per phase
    complexity_schedule: Vec<usize>,
    /// Search statistics
    statistics: SearchStrategyStatistics<T>,
    /// Number of top architectures to expand per phase
    top_k_per_phase: usize,
    /// Phase performance tracking: (architecture_index, performance)
    phase_performances: Vec<Vec<(usize, T)>>,
    /// Number of architectures generated in current phase
    current_phase_generated: usize,
    /// Budget per phase (architectures to evaluate before advancing)
    phase_budget: usize,
    /// Whether the search has completed all phases
    search_complete: bool,
    /// RNG for randomized generation
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > ProgressiveNAS<T>
{
    /// Create a new ProgressiveNAS instance
    ///
    /// # Arguments
    /// * `max_phases` - Number of phases of increasing complexity
    /// * `phase_budget` - Number of architectures to evaluate per phase
    /// * `top_k_per_phase` - Number of top architectures to carry forward
    pub fn new(max_phases: usize, phase_budget: usize, top_k_per_phase: usize) -> Self {
        let max_phases = max_phases.max(1);
        // Generate complexity schedule: phase i allows (i+1) components
        let complexity_schedule: Vec<usize> = (1..=max_phases).collect();

        Self {
            current_phase: 0,
            max_phases,
            phase_architectures: vec![Vec::new(); max_phases],
            complexity_schedule,
            statistics: SearchStrategyStatistics::default(),
            top_k_per_phase: top_k_per_phase.max(1),
            phase_performances: vec![Vec::new(); max_phases],
            current_phase_generated: 0,
            phase_budget: phase_budget.max(1),
            search_complete: false,
            rng: Random::seed(42),
        }
    }

    /// Create with a custom complexity schedule
    ///
    /// # Arguments
    /// * `complexity_schedule` - Custom max components per phase
    /// * `phase_budget` - Number of architectures to evaluate per phase
    /// * `top_k_per_phase` - Number of top architectures to carry forward
    pub fn with_schedule(
        complexity_schedule: Vec<usize>,
        phase_budget: usize,
        top_k_per_phase: usize,
    ) -> Self {
        let max_phases = complexity_schedule.len().max(1);
        Self {
            current_phase: 0,
            max_phases,
            phase_architectures: vec![Vec::new(); max_phases],
            complexity_schedule,
            statistics: SearchStrategyStatistics::default(),
            top_k_per_phase: top_k_per_phase.max(1),
            phase_performances: vec![Vec::new(); max_phases],
            current_phase_generated: 0,
            phase_budget: phase_budget.max(1),
            search_complete: false,
            rng: Random::seed(42),
        }
    }

    /// Get the current phase
    pub fn current_phase(&self) -> usize {
        self.current_phase
    }

    /// Check if the search has completed all phases
    pub fn is_complete(&self) -> bool {
        self.search_complete
    }

    /// Get the maximum complexity for the current phase
    fn current_max_complexity(&self) -> usize {
        if self.current_phase < self.complexity_schedule.len() {
            self.complexity_schedule[self.current_phase]
        } else {
            // Fallback: use the last schedule entry
            self.complexity_schedule.last().copied().unwrap_or(1)
        }
    }

    /// Try to advance to the next phase
    ///
    /// Selects the top-k architectures from the current phase and
    /// uses them as seeds for the next phase.
    fn try_advance_phase(&mut self) -> bool {
        if self.current_phase >= self.max_phases - 1 {
            self.search_complete = true;
            return false;
        }

        // Get top-k architectures from current phase by performance
        let current_performances = &self.phase_performances[self.current_phase];
        if current_performances.is_empty() {
            return false;
        }

        let mut sorted_performances = current_performances.clone();
        sorted_performances
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-k and seed next phase
        let top_k = sorted_performances
            .iter()
            .take(self.top_k_per_phase)
            .collect::<Vec<_>>();

        let next_phase = self.current_phase + 1;

        // Clone architectures to carry forward to avoid borrow conflict
        let carry_forward: Vec<OptimizerArchitecture<T>> = top_k
            .iter()
            .filter_map(|(arch_idx, _perf)| {
                self.phase_architectures[self.current_phase]
                    .get(*arch_idx)
                    .cloned()
            })
            .collect();

        for arch in carry_forward {
            self.phase_architectures[next_phase].push(arch);
        }

        self.current_phase = next_phase;
        self.current_phase_generated = 0;
        true
    }

    /// Generate a new architecture within the current phase's complexity budget
    fn generate_phase_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        let max_components = self.current_max_complexity();
        let seed_architectures = &self.phase_architectures[self.current_phase];

        // If we have seed architectures from a previous phase, expand one
        if !seed_architectures.is_empty() && self.current_phase > 0 {
            let seed_idx = self.rng.gen_range(0..seed_architectures.len());
            let seed = seed_architectures[seed_idx].clone();
            return self.expand_architecture(&seed, max_components, searchspace);
        }

        // Otherwise generate from scratch within complexity budget
        self.generate_random_architecture(max_components, searchspace)
    }

    /// Expand a seed architecture by adding components up to max complexity
    fn expand_architecture(
        &mut self,
        seed: &OptimizerArchitecture<T>,
        max_components: usize,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        let current_size = seed.components.len();
        let additional = if current_size < max_components {
            self.rng.gen_range(0..=(max_components - current_size))
        } else {
            0
        };

        let mut components = seed.components.clone();
        let mut parameters = seed.parameters.clone();

        for idx in 0..additional {
            let component_config =
                &searchspace.components[self.rng.gen_range(0..searchspace.components.len())];

            let component_type = match component_config.component_type {
                ConfigComponentType::SGD => ComponentType::SGD,
                ConfigComponentType::Adam => ComponentType::Adam,
                ConfigComponentType::AdamW => ComponentType::AdamW,
                ConfigComponentType::RMSprop => ComponentType::RMSprop,
                ConfigComponentType::AdaGrad => ComponentType::AdaGrad,
                ConfigComponentType::AdaDelta => ComponentType::AdaDelta,
                ConfigComponentType::LBFGS => ComponentType::LBFGS,
                ConfigComponentType::Momentum => ComponentType::Momentum,
                ConfigComponentType::Nesterov => ComponentType::Nesterov,
                ConfigComponentType::Custom(_) => ComponentType::Custom,
                _ => ComponentType::Adam,
            };

            // Sample hyperparameters for the new component
            for (param_name, param_range) in &component_config.hyperparameter_ranges {
                let value = self.sample_parameter(param_range);
                let key = format!("{}_{}", current_size + idx, param_name);
                parameters.insert(
                    key,
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero()),
                );
            }

            components.push(format!("{:?}", component_type));
        }

        Ok(OptimizerArchitecture {
            components,
            parameters,
            connections: Vec::new(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("phase".to_string(), format!("{}", self.current_phase));
                m
            },
            hyperparameters: HashMap::new(),
            architecture_id: format!(
                "prog_arch_p{}_{}",
                self.current_phase,
                self.rng.random::<u32>()
            ),
        })
    }

    /// Generate a random architecture within complexity budget
    fn generate_random_architecture(
        &mut self,
        max_components: usize,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        let num_components = self.rng.gen_range(1..=max_components);
        let mut components = Vec::new();
        let mut parameters = HashMap::new();

        for idx in 0..num_components {
            let component_config =
                &searchspace.components[self.rng.gen_range(0..searchspace.components.len())];

            let component_type = match component_config.component_type {
                ConfigComponentType::SGD => ComponentType::SGD,
                ConfigComponentType::Adam => ComponentType::Adam,
                ConfigComponentType::AdamW => ComponentType::AdamW,
                ConfigComponentType::RMSprop => ComponentType::RMSprop,
                ConfigComponentType::AdaGrad => ComponentType::AdaGrad,
                ConfigComponentType::AdaDelta => ComponentType::AdaDelta,
                ConfigComponentType::LBFGS => ComponentType::LBFGS,
                ConfigComponentType::Momentum => ComponentType::Momentum,
                ConfigComponentType::Nesterov => ComponentType::Nesterov,
                ConfigComponentType::Custom(_) => ComponentType::Custom,
                _ => ComponentType::Adam,
            };

            for (param_name, param_range) in &component_config.hyperparameter_ranges {
                let value = self.sample_parameter(param_range);
                let key = format!("{}_{}", idx, param_name);
                parameters.insert(
                    key,
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero()),
                );
            }

            components.push(format!("{:?}", component_type));
        }

        Ok(OptimizerArchitecture {
            components,
            parameters,
            connections: Vec::new(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("phase".to_string(), format!("{}", self.current_phase));
                m
            },
            hyperparameters: HashMap::new(),
            architecture_id: format!(
                "prog_arch_p{}_{}",
                self.current_phase,
                self.rng.random::<u32>()
            ),
        })
    }

    /// Sample a parameter value from the given range
    fn sample_parameter(&mut self, param_range: &ParameterRange) -> f64 {
        match param_range {
            ParameterRange::Continuous(min, max) => self.rng.gen_range(*min..*max),
            ParameterRange::LogUniform(min, max) => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_val = self.rng.gen_range(log_min..log_max);
                log_val.exp()
            }
            ParameterRange::Integer(min, max) => self.rng.gen_range(*min..*max) as f64,
            ParameterRange::Boolean => {
                if self.rng.random::<f64>() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ParameterRange::Discrete(values) => {
                let idx = self.rng.gen_range(0..values.len());
                values[idx]
            }
            ParameterRange::Categorical(_) => 0.0,
        }
    }
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > SearchStrategy<T> for ProgressiveNAS<T>
{
    fn initialize(&mut self, _searchspace: &SearchSpaceConfig) -> Result<()> {
        self.current_phase = 0;
        self.phase_architectures = vec![Vec::new(); self.max_phases];
        self.phase_performances = vec![Vec::new(); self.max_phases];
        self.current_phase_generated = 0;
        self.search_complete = false;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        // Check if current phase budget is exhausted
        if self.current_phase_generated >= self.phase_budget {
            self.try_advance_phase();
        }

        // If search is complete, still generate from the last phase
        let architecture = self.generate_phase_architecture(searchspace)?;

        // Store the architecture in current phase
        let arch_idx = self.phase_architectures[self.current_phase].len();
        self.phase_architectures[self.current_phase].push(architecture.clone());
        self.current_phase_generated += 1;
        self.statistics.total_architectures_generated += 1;

        Ok(architecture)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        for result in results {
            let performance = result
                .evaluation_results
                .metric_scores
                .get(&EvaluationMetric::FinalPerformance)
                .cloned()
                .unwrap_or(T::zero());

            // Find the architecture index in current phase
            let arch_idx = self.phase_architectures[self.current_phase]
                .iter()
                .position(|a| a.architecture_id == result.architecture.architecture_id)
                .unwrap_or(
                    self.phase_architectures[self.current_phase]
                        .len()
                        .saturating_sub(1),
                );

            self.phase_performances[self.current_phase].push((arch_idx, performance));
        }

        // Update global statistics
        let all_performances: Vec<T> = self
            .phase_performances
            .iter()
            .flat_map(|phase| phase.iter().map(|(_, p)| *p))
            .collect();

        if !all_performances.is_empty() {
            self.statistics.best_performance = all_performances
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            let sum: T = all_performances.iter().cloned().sum();
            self.statistics.average_performance =
                sum / T::from(all_performances.len()).expect("conversion from usize to T failed");

            // Convergence rate: fraction of phases completed
            self.statistics.convergence_rate = scirs2_core::numeric::NumCast::from(
                self.current_phase as f64 / self.max_phases as f64,
            )
            .unwrap_or_else(|| T::zero());
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "ProgressiveNAS"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        // Early phases are more exploratory, later phases more exploitative
        let exploration = 1.0 - (self.current_phase as f64 / self.max_phases as f64);
        stats.exploration_rate =
            scirs2_core::numeric::NumCast::from(exploration).unwrap_or_else(|| T::zero());
        stats.exploitation_rate = T::one() - stats.exploration_rate;
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_nas_creation() {
        let nas = ProgressiveNAS::<f64>::new(5, 20, 3);
        assert_eq!(nas.current_phase(), 0);
        assert!(!nas.is_complete());
        assert_eq!(nas.name(), "ProgressiveNAS");
    }

    #[test]
    fn test_progressive_nas_with_schedule() {
        let schedule = vec![1, 2, 4, 8];
        let nas = ProgressiveNAS::<f64>::with_schedule(schedule.clone(), 10, 5);
        assert_eq!(nas.max_phases, 4);
        assert_eq!(nas.complexity_schedule, schedule);
    }

    #[test]
    fn test_progressive_nas_statistics() {
        let nas = ProgressiveNAS::<f64>::new(5, 20, 3);
        let stats = nas.get_statistics();
        // Phase 0 of 5 should be fully exploratory
        assert!(stats.exploration_rate > 0.9);
    }
}
