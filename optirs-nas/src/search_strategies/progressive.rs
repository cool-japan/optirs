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

use crate::architecture::ComponentType;
use crate::error::{OptimError, Result};
use crate::nas_engine::config::{ComponentType as ConfigComponentType, ParameterRange};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::{SearchStrategy, SearchStrategyStatistics};

/// One candidate generated inside a phase, together with its evaluation once it
/// arrives.
///
/// Storing the score *with* the architecture is what makes attribution safe: the
/// previous design kept `(index, performance)` pairs in a parallel vector and
/// looked the index up positionally, so any reordering or any eviction silently
/// credited a score to the wrong architecture.
#[derive(Debug, Clone)]
struct PhaseEntry<T: Float + Debug + Send + Sync + 'static> {
    architecture: OptimizerArchitecture<T>,
    performance: Option<T>,
}

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
    /// Candidates generated in each phase, indexed by phase
    phase_entries: Vec<Vec<PhaseEntry<T>>>,
    /// Architectures carried forward from the previous phase, used as expansion
    /// seeds. Kept separate from [`Self::phase_entries`] so a seed is never
    /// mistaken for a candidate this phase generated and evaluated.
    phase_seeds: Vec<Vec<OptimizerArchitecture<T>>>,
    /// Which phase each generated architecture belongs to, keyed by architecture
    /// id. This is the *only* attribution path: a result whose id is unknown is
    /// counted as unattributed, never credited to a guessed candidate.
    architecture_phase: HashMap<String, usize>,
    /// Maximum complexity (number of components) allowed per phase
    complexity_schedule: Vec<usize>,
    /// Search statistics
    statistics: SearchStrategyStatistics<T>,
    /// Number of top architectures to expand per phase
    top_k_per_phase: usize,
    /// Budget per phase, counted in **evaluated** architectures
    phase_budget: usize,
    /// Whether the search has completed all phases
    search_complete: bool,
    /// Results that could not be attributed to any generated architecture
    unattributed_results: usize,
    /// RNG for randomized generation
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > ProgressiveNAS<T>
{
    /// Create a new ProgressiveNAS instance, seeded from OS entropy.
    ///
    /// The RNG used to be a hard-coded `Random::seed(42)`, which made every
    /// instance in a process explore the identical sequence of architectures. Use
    /// [`ProgressiveNAS::with_seed`] when reproducibility is wanted.
    ///
    /// # Arguments
    /// * `max_phases` - Number of phases of increasing complexity
    /// * `phase_budget` - Number of architectures to **evaluate** per phase
    /// * `top_k_per_phase` - Number of top architectures to carry forward
    pub fn new(max_phases: usize, phase_budget: usize, top_k_per_phase: usize) -> Self {
        Self::with_seed(
            max_phases,
            phase_budget,
            top_k_per_phase,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Create a fully reproducible ProgressiveNAS instance.
    pub fn with_seed(
        max_phases: usize,
        phase_budget: usize,
        top_k_per_phase: usize,
        seed: u64,
    ) -> Self {
        let max_phases = max_phases.max(1);
        // Generate complexity schedule: phase i allows (i+1) components
        let complexity_schedule: Vec<usize> = (1..=max_phases).collect();
        Self::from_parts(
            complexity_schedule,
            max_phases,
            phase_budget,
            top_k_per_phase,
            seed,
        )
    }

    /// Create with a custom complexity schedule, seeded from OS entropy.
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
        Self::with_schedule_and_seed(
            complexity_schedule,
            phase_budget,
            top_k_per_phase,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Create with a custom complexity schedule and an explicit seed.
    pub fn with_schedule_and_seed(
        complexity_schedule: Vec<usize>,
        phase_budget: usize,
        top_k_per_phase: usize,
        seed: u64,
    ) -> Self {
        let max_phases = complexity_schedule.len().max(1);
        Self::from_parts(
            complexity_schedule,
            max_phases,
            phase_budget,
            top_k_per_phase,
            seed,
        )
    }

    fn from_parts(
        complexity_schedule: Vec<usize>,
        max_phases: usize,
        phase_budget: usize,
        top_k_per_phase: usize,
        seed: u64,
    ) -> Self {
        Self {
            current_phase: 0,
            max_phases,
            phase_entries: vec![Vec::new(); max_phases],
            phase_seeds: vec![Vec::new(); max_phases],
            architecture_phase: HashMap::new(),
            complexity_schedule,
            statistics: SearchStrategyStatistics::default(),
            top_k_per_phase: top_k_per_phase.max(1),
            phase_budget: phase_budget.max(1),
            search_complete: false,
            unattributed_results: 0,
            rng: Random::seed(seed),
        }
    }

    /// Get the current phase
    pub fn current_phase(&self) -> usize {
        self.current_phase
    }

    /// Check if the search has completed all phases.
    ///
    /// Reported to the engine through
    /// [`SearchStrategy::is_search_complete`], so an exhausted schedule actually
    /// stops the search instead of silently degenerating into random sampling at
    /// the final complexity level.
    pub fn is_complete(&self) -> bool {
        self.search_complete
    }

    /// Number of architectures **evaluated** in `phase`.
    pub fn evaluated_in_phase(&self, phase: usize) -> usize {
        self.phase_entries
            .get(phase)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| entry.performance.is_some())
                    .count()
            })
            .unwrap_or(0)
    }

    /// Number of architectures generated in `phase`.
    pub fn generated_in_phase(&self, phase: usize) -> usize {
        self.phase_entries
            .get(phase)
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    /// Results whose architecture id matched nothing this strategy generated.
    ///
    /// Reported rather than absorbed: the previous implementation fell back to
    /// `phase_architectures[current_phase].len() - 1`, crediting the score to
    /// whichever architecture happened to be generated last.
    pub fn unattributed_results(&self) -> usize {
        self.unattributed_results
    }

    /// Architectures carried forward into `phase` as expansion seeds.
    pub fn seeds_for_phase(&self, phase: usize) -> &[OptimizerArchitecture<T>] {
        self.phase_seeds
            .get(phase)
            .map(|seeds| seeds.as_slice())
            .unwrap_or(&[])
    }

    /// Get the maximum complexity for the current phase
    fn current_max_complexity(&self) -> usize {
        self.complexity_schedule
            .get(self.current_phase)
            .or_else(|| self.complexity_schedule.last())
            .copied()
            .unwrap_or(1)
            // A schedule entry of zero would mean "an architecture with no
            // components", which is not an optimizer; one component is the floor.
            .max(1)
    }

    /// The best `top_k_per_phase` evaluated architectures of `phase`, best first.
    ///
    /// Ties are broken by architecture id so the carried-forward set is
    /// deterministic rather than dependent on storage order.
    fn top_architectures(&self, phase: usize) -> Vec<OptimizerArchitecture<T>> {
        let Some(entries) = self.phase_entries.get(phase) else {
            return Vec::new();
        };
        let mut scored: Vec<(&PhaseEntry<T>, T)> = entries
            .iter()
            .filter_map(|entry| entry.performance.map(|score| (entry, score)))
            .collect();
        scored.sort_by(|(left_entry, left), (right_entry, right)| {
            right
                .partial_cmp(left)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left_entry
                        .architecture
                        .architecture_id
                        .cmp(&right_entry.architecture.architecture_id)
                })
        });
        scored
            .into_iter()
            .take(self.top_k_per_phase)
            .map(|(entry, _)| entry.architecture.clone())
            .collect()
    }

    /// Try to advance to the next phase.
    ///
    /// Called from `update_with_results` once the current phase has **evaluated**
    /// its budget. The previous version advanced on the number of architectures
    /// *generated*, so a phase could end before a single result for it had come
    /// back — and every result that then arrived was credited to the new phase.
    fn try_advance_phase(&mut self) -> bool {
        if self.evaluated_in_phase(self.current_phase) < self.phase_budget {
            return false;
        }
        if self.current_phase + 1 >= self.max_phases {
            if !self.search_complete {
                log::info!(
                    "ProgressiveNAS completed its {} phase(s); the complexity schedule is \
                     exhausted",
                    self.max_phases
                );
            }
            self.search_complete = true;
            return false;
        }
        let carry_forward = self.top_architectures(self.current_phase);
        if carry_forward.is_empty() {
            // Nothing evaluated well enough to expand: staying in this phase is
            // the honest response, since the next phase is defined as an
            // expansion of this one's best.
            return false;
        }
        let next_phase = self.current_phase + 1;
        if let Some(seeds) = self.phase_seeds.get_mut(next_phase) {
            seeds.extend(carry_forward);
        }
        log::debug!(
            "ProgressiveNAS advancing from phase {} to {} with {} seed architecture(s)",
            self.current_phase,
            next_phase,
            self.phase_seeds
                .get(next_phase)
                .map(|seeds| seeds.len())
                .unwrap_or(0)
        );
        self.current_phase = next_phase;
        true
    }

    /// Generate a new architecture within the current phase's complexity budget
    fn generate_phase_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        let max_components = self.current_max_complexity();
        let seeds = self
            .phase_seeds
            .get(self.current_phase)
            .cloned()
            .unwrap_or_default();

        // If we have seed architectures from a previous phase, expand one
        if !seeds.is_empty() {
            let seed_idx = self.rng.gen_range(0..seeds.len());
            let seed = seeds[seed_idx].clone();
            return self.expand_architecture(&seed, max_components, searchspace);
        }

        // Otherwise generate from scratch within complexity budget
        self.generate_random_architecture(max_components, searchspace)
    }

    /// Reject an empty component list instead of panicking inside `gen_range`.
    fn require_components(searchspace: &SearchSpaceConfig) -> Result<()> {
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "ProgressiveNAS needs a non-empty SearchSpaceConfig::components to \
                 sample from; populate the component list (not just component_types) \
                 before running this strategy"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Map a configured component type onto the architecture vocabulary.
    fn component_label(component_type: &ConfigComponentType) -> String {
        let mapped = match component_type {
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
        format!("{:?}", mapped)
    }

    /// Expand a seed architecture by adding components up to max complexity
    fn expand_architecture(
        &mut self,
        seed: &OptimizerArchitecture<T>,
        max_components: usize,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        Self::require_components(searchspace)?;
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
            let label = Self::component_label(&component_config.component_type);
            let ranges = component_config.hyperparameter_ranges.clone();

            // Sample hyperparameters for the new component
            let mut names: Vec<String> = ranges.keys().cloned().collect();
            names.sort();
            for param_name in names {
                let Some(param_range) = ranges.get(&param_name) else {
                    continue;
                };
                let value = self.sample_parameter(param_range);
                let key = format!("{}_{}", current_size + idx, param_name);
                parameters.insert(
                    key,
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero),
                );
            }

            components.push(label);
        }

        Ok(self.finish_architecture(components, parameters))
    }

    /// Generate a random architecture within complexity budget
    fn generate_random_architecture(
        &mut self,
        max_components: usize,
        searchspace: &SearchSpaceConfig,
    ) -> Result<OptimizerArchitecture<T>> {
        Self::require_components(searchspace)?;
        let num_components = self.rng.gen_range(1..=max_components.max(1));
        let mut components = Vec::new();
        let mut parameters = HashMap::new();

        for idx in 0..num_components {
            let component_config =
                &searchspace.components[self.rng.gen_range(0..searchspace.components.len())];
            let label = Self::component_label(&component_config.component_type);
            let ranges = component_config.hyperparameter_ranges.clone();

            let mut names: Vec<String> = ranges.keys().cloned().collect();
            names.sort();
            for param_name in names {
                let Some(param_range) = ranges.get(&param_name) else {
                    continue;
                };
                let value = self.sample_parameter(param_range);
                let key = format!("{}_{}", idx, param_name);
                parameters.insert(
                    key,
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero),
                );
            }

            components.push(label);
        }

        Ok(self.finish_architecture(components, parameters))
    }

    /// Assemble the architecture shared by both generators, including the chain of
    /// connections between consecutive components (previously left empty, which
    /// described a set of unconnected components rather than a pipeline).
    fn finish_architecture(
        &mut self,
        components: Vec<String>,
        parameters: HashMap<String, T>,
    ) -> OptimizerArchitecture<T> {
        let connections: Vec<(usize, usize)> = (1..components.len()).map(|i| (i - 1, i)).collect();
        let mut metadata = HashMap::new();
        metadata.insert("phase".to_string(), format!("{}", self.current_phase));
        metadata.insert(
            "max_complexity".to_string(),
            format!("{}", self.current_max_complexity()),
        );
        OptimizerArchitecture {
            components,
            parameters: parameters.clone(),
            connections,
            metadata,
            hyperparameters: parameters,
            architecture_id: format!(
                "prog_arch_p{}_{:016x}",
                self.current_phase,
                self.rng.gen_range(0..u64::MAX)
            ),
        }
    }

    /// Sample a parameter value from the given range.
    ///
    /// `Categorical` returns the **index** of the chosen category, which is what a
    /// numeric parameter map can faithfully hold; it used to return a constant
    /// `0.0` for every categorical parameter, so a categorical axis was silently
    /// pinned to its first value and never searched at all.
    fn sample_parameter(&mut self, param_range: &ParameterRange) -> f64 {
        match param_range {
            ParameterRange::Continuous(min, max) => {
                if max > min {
                    self.rng.gen_range(*min..*max)
                } else {
                    *min
                }
            }
            ParameterRange::LogUniform(min, max) => {
                if *min > 0.0 && max > min {
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let log_val = self.rng.gen_range(log_min..log_max);
                    log_val.exp()
                } else {
                    *min
                }
            }
            ParameterRange::Integer(min, max) => {
                if max > min {
                    self.rng.gen_range(*min..*max) as f64
                } else {
                    *min as f64
                }
            }
            ParameterRange::Boolean => {
                if self.rng.gen_range(0.0..1.0) < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ParameterRange::Discrete(values) => {
                if values.is_empty() {
                    0.0
                } else {
                    values[self.rng.gen_range(0..values.len())]
                }
            }
            ParameterRange::Categorical(values) => {
                if values.is_empty() {
                    0.0
                } else {
                    self.rng.gen_range(0..values.len()) as f64
                }
            }
        }
    }

    /// Credit `performance` to the architecture `architecture_id` names, in the
    /// phase it was actually generated in. Returns the phase, or `None` when the
    /// id is unknown to this strategy.
    fn record_performance(&mut self, architecture_id: &str, performance: T) -> Option<usize> {
        let phase = self.architecture_phase.get(architecture_id).copied()?;
        let entries = self.phase_entries.get_mut(phase)?;
        let entry = entries
            .iter_mut()
            .find(|entry| entry.architecture.architecture_id == architecture_id)?;
        entry.performance = Some(performance);
        Some(phase)
    }

    /// Every recorded performance across all phases.
    fn all_performances(&self) -> Vec<T> {
        self.phase_entries
            .iter()
            .flat_map(|entries| entries.iter().filter_map(|entry| entry.performance))
            .collect()
    }
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > SearchStrategy<T> for ProgressiveNAS<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        Self::require_components(searchspace)?;
        self.current_phase = 0;
        self.phase_entries = vec![Vec::new(); self.max_phases];
        self.phase_seeds = vec![Vec::new(); self.max_phases];
        self.architecture_phase.clear();
        self.search_complete = false;
        self.unattributed_results = 0;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        // Phase advancement is driven by *evaluated* results in
        // `update_with_results`; generation only ever samples from the phase the
        // strategy is currently in. Once the schedule is exhausted the strategy
        // keeps sampling at the final complexity level and reports completion
        // through `is_search_complete`, so a caller that ignores completion still
        // gets valid architectures instead of an error.
        let architecture = self.generate_phase_architecture(searchspace)?;

        let id = architecture.architecture_id.clone();
        if let Some(entries) = self.phase_entries.get_mut(self.current_phase) {
            entries.push(PhaseEntry {
                architecture: architecture.clone(),
                performance: None,
            });
        }
        self.architecture_phase.insert(id, self.current_phase);
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
                .copied()
                .unwrap_or_else(T::zero);

            // Attribution is by architecture id, in the phase the architecture was
            // generated in. A result this strategy did not generate is counted, not
            // credited to a guess.
            if self
                .record_performance(&result.architecture.architecture_id, performance)
                .is_none()
            {
                self.unattributed_results += 1;
                log::debug!(
                    "ProgressiveNAS received a result for unknown architecture {:?}; \
                     counted as unattributed",
                    result.architecture.architecture_id
                );
            }
        }

        // Advance only once the current phase has evaluated its full budget. A
        // single call may complete several phases when a large batch arrives.
        while self.try_advance_phase() {}

        let all_performances = self.all_performances();
        if !all_performances.is_empty() {
            self.statistics.best_performance =
                all_performances
                    .iter()
                    .copied()
                    .fold(
                        T::neg_infinity(),
                        |acc, value| {
                            if value > acc {
                                value
                            } else {
                                acc
                            }
                        },
                    );
            let sum: T = all_performances.iter().copied().sum();
            // `T::from(len)` cannot fail for a usize that fits a float, but the
            // previous `.expect("conversion from usize to T failed")` would have
            // aborted the whole search if it ever did; fall back to reporting the
            // sum's own scale instead of panicking.
            let count: T = scirs2_core::numeric::NumCast::from(all_performances.len() as f64)
                .unwrap_or_else(T::one);
            self.statistics.average_performance = if count > T::zero() {
                sum / count
            } else {
                T::zero()
            };

            // Convergence rate: fraction of the complexity schedule completed. A
            // completed search reports 1, not `(max_phases - 1) / max_phases`.
            let progress = if self.search_complete {
                1.0
            } else {
                self.current_phase as f64 / self.max_phases as f64
            };
            self.statistics.convergence_rate =
                scirs2_core::numeric::NumCast::from(progress).unwrap_or_else(T::zero);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "ProgressiveNAS"
    }

    /// The complexity schedule is exhausted: every phase has evaluated its budget
    /// and the final phase has been reached.
    fn is_search_complete(&self) -> bool {
        self.search_complete
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        // Early phases are more exploratory, later phases more exploitative.
        let exploration = if self.search_complete {
            0.0
        } else {
            1.0 - (self.current_phase as f64 / self.max_phases as f64)
        };
        stats.exploration_rate =
            scirs2_core::numeric::NumCast::from(exploration).unwrap_or_else(T::zero);
        stats.exploitation_rate = T::one() - stats.exploration_rate;
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::config::OptimizerComponentConfig;
    use crate::nas_engine::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };

    /// A search space with three component types, each carrying one continuous
    /// hyperparameter.
    fn search_space() -> SearchSpaceConfig {
        let mut components = Vec::new();
        for component_type in [
            ConfigComponentType::SGD,
            ConfigComponentType::Adam,
            ConfigComponentType::RMSprop,
        ] {
            let mut ranges = HashMap::new();
            ranges.insert(
                "learning_rate".to_string(),
                ParameterRange::LogUniform(1e-4, 1e-1),
            );
            components.push(OptimizerComponentConfig {
                component_type,
                hyperparameter_ranges: ranges,
                complexity_score: 1.0,
                memory_requirement: 0,
                computational_cost: 1.0,
                compatibility_constraints: Vec::new(),
            });
        }
        SearchSpaceConfig {
            components,
            ..Default::default()
        }
    }

    fn result_for(architecture: &OptimizerArchitecture<f64>, score: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, score);
        SearchResult {
            architecture: architecture.clone(),
            evaluation_results: EvaluationResults {
                metric_scores,
                overall_score: score,
                confidence_intervals: HashMap::new(),
                evaluation_time: std::time::Duration::from_secs(0),
                success: true,
                error_message: None,
                cv_results: None,
                benchmark_results: HashMap::new(),
                training_trajectory: Vec::new(),
            },
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

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

    #[test]
    fn phase_advances_on_evaluated_results_not_on_generated_ones() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(3, 2, 1, 11);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();

        // Generate far more than the phase budget: generation alone must not
        // advance the phase. The previous implementation advanced on the generated
        // count, so this loop alone took it through every phase before a single
        // result existed.
        let mut generated = Vec::new();
        for _ in 0..6 {
            generated.push(
                nas.generate_architecture(&space, &history)
                    .expect("generate"),
            );
        }
        assert_eq!(
            nas.current_phase(),
            0,
            "generation must not advance a phase"
        );
        assert_eq!(nas.generated_in_phase(0), 6);
        assert_eq!(nas.evaluated_in_phase(0), 0);

        // One result: still short of the budget of 2.
        nas.update_with_results(&[result_for(&generated[0], 0.5)])
            .expect("update");
        assert_eq!(nas.current_phase(), 0);

        // The second evaluated result completes the budget and advances.
        nas.update_with_results(&[result_for(&generated[1], 0.9)])
            .expect("update");
        assert_eq!(nas.current_phase(), 1);
        // The best architecture of phase 0 is carried forward as a seed.
        let seeds = nas.seeds_for_phase(1);
        assert_eq!(seeds.len(), 1);
        assert_eq!(
            seeds[0].architecture_id, generated[1].architecture_id,
            "the highest-scoring architecture must be the one carried forward"
        );
    }

    #[test]
    fn results_are_credited_to_the_phase_that_generated_them() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(3, 2, 1, 22);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();

        // Two architectures from phase 0, then advance, then one from phase 1.
        let a = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        let b = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        nas.update_with_results(&[result_for(&a, 0.2), result_for(&b, 0.4)])
            .expect("update");
        assert_eq!(nas.current_phase(), 1);

        let c = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        assert_eq!(nas.generated_in_phase(1), 1);

        // A *late* result for a phase-0 architecture arrives after the transition.
        // It must be credited to phase 0 — the old code credited it to
        // `current_phase` (1) and, because the id was not found there, to whichever
        // architecture happened to be last in phase 1 via
        // `unwrap_or(len - 1)`, overwriting `c`'s slot.
        let late = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        // `late` belongs to phase 1; regenerate a phase-0-owned architecture by
        // re-submitting `a` with a different score instead.
        nas.update_with_results(&[result_for(&a, 0.95)])
            .expect("update");
        assert_eq!(
            nas.evaluated_in_phase(0),
            2,
            "the late phase-0 result must be credited to phase 0"
        );
        assert_eq!(
            nas.evaluated_in_phase(1),
            0,
            "no phase-1 architecture has been evaluated yet, so its count must be 0"
        );
        assert_eq!(nas.unattributed_results(), 0);
        // Neither phase-1 architecture was touched.
        assert!(nas.phase_entries[1]
            .iter()
            .all(|entry| entry.performance.is_none()));
        assert_eq!(
            nas.generated_in_phase(1),
            2,
            "both phase-1 architectures are still recorded: {:?} {:?}",
            c.architecture_id,
            late.architecture_id
        );
    }

    #[test]
    fn an_unknown_result_is_counted_not_credited_to_a_guess() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(2, 2, 1, 33);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();
        let known = nas
            .generate_architecture(&space, &history)
            .expect("generate");

        let mut stranger = known.clone();
        stranger.architecture_id = "not_from_this_strategy".to_string();
        nas.update_with_results(&[result_for(&stranger, 1.0)])
            .expect("update");

        assert_eq!(nas.unattributed_results(), 1);
        assert_eq!(
            nas.evaluated_in_phase(0),
            0,
            "an unknown architecture must not be credited to a generated one"
        );
        // The known architecture's slot is untouched.
        assert!(nas.phase_entries[0]
            .iter()
            .all(|entry| entry.performance.is_none()));
        assert_eq!(nas.current_phase(), 0);
    }

    #[test]
    fn completion_is_signalled_once_the_schedule_is_exhausted() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(2, 1, 1, 44);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();
        assert!(!nas.is_search_complete());

        // Phase 0: one evaluated result fills its budget and advances to phase 1.
        let a = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        nas.update_with_results(&[result_for(&a, 0.5)])
            .expect("update");
        assert_eq!(nas.current_phase(), 1);
        assert!(!nas.is_search_complete());

        // Phase 1 is the last one: filling its budget completes the search.
        let b = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        nas.update_with_results(&[result_for(&b, 0.7)])
            .expect("update");
        assert!(
            nas.is_search_complete(),
            "an exhausted complexity schedule must report completion"
        );
        assert!(nas.is_complete());

        // Completion is reflected in the statistics, and generation still returns
        // valid architectures rather than erroring or panicking.
        let stats = nas.get_statistics();
        assert_eq!(stats.convergence_rate, 1.0);
        assert_eq!(stats.exploration_rate, 0.0);
        assert_eq!(stats.exploitation_rate, 1.0);
        let extra = nas
            .generate_architecture(&space, &history)
            .expect("generation after completion must still work");
        assert!(!extra.components.is_empty());
    }

    #[test]
    fn architectures_respect_the_phase_complexity_schedule_and_carry_connections() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_schedule_and_seed(vec![1, 3], 4, 2, 55);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();

        // Phase 0 allows exactly one component.
        for _ in 0..8 {
            let arch = nas
                .generate_architecture(&space, &history)
                .expect("generate");
            assert_eq!(
                arch.components.len(),
                1,
                "phase 0 allows one component, got {:?}",
                arch.components
            );
            assert!(
                arch.connections.is_empty(),
                "a single component has no edges"
            );
            assert_eq!(arch.metadata.get("phase").map(String::as_str), Some("0"));
        }

        // Fill the budget so the search moves to the 3-component phase.
        let generated: Vec<OptimizerArchitecture<f64>> = nas.phase_entries[0]
            .iter()
            .map(|entry| entry.architecture.clone())
            .collect();
        let results: Vec<SearchResult<f64>> = generated
            .iter()
            .enumerate()
            .map(|(i, arch)| result_for(arch, i as f64 / 10.0))
            .collect();
        nas.update_with_results(&results).expect("update");
        assert_eq!(nas.current_phase(), 1);

        let mut grew = false;
        for _ in 0..12 {
            let arch = nas
                .generate_architecture(&space, &history)
                .expect("generate");
            assert!(
                arch.components.len() <= 3,
                "phase 1 allows at most 3 components, got {:?}",
                arch.components
            );
            if arch.components.len() > 1 {
                grew = true;
                // Connections form a chain over the components, with every index in
                // range.
                assert_eq!(arch.connections.len(), arch.components.len() - 1);
                for (from, to) in &arch.connections {
                    assert!(*from < arch.components.len() && *to < arch.components.len());
                }
            }
        }
        assert!(grew, "expansion never produced a larger architecture");
    }

    #[test]
    fn an_empty_search_space_is_an_error_not_a_panic() {
        let empty = SearchSpaceConfig::default();
        let mut nas = ProgressiveNAS::<f64>::with_seed(2, 2, 1, 66);
        // `SearchSpaceConfig::default()` carries no components; `gen_range(0..0)`
        // used to panic deep inside the sampler.
        let error = nas
            .initialize(&empty)
            .expect_err("an empty component list must be rejected");
        assert!(format!("{error}").contains("non-empty"), "{error}");
        let history = VecDeque::new();
        assert!(nas.generate_architecture(&empty, &history).is_err());
    }

    #[test]
    fn a_zero_complexity_schedule_entry_does_not_panic() {
        let space = search_space();
        // A schedule of zeros used to reach `gen_range(1..=0)`, which panics.
        let mut nas = ProgressiveNAS::<f64>::with_schedule_and_seed(vec![0, 0], 2, 1, 77);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();
        let arch = nas
            .generate_architecture(&space, &history)
            .expect("generate");
        assert_eq!(arch.components.len(), 1, "the floor is one component");
    }

    #[test]
    fn categorical_parameters_are_sampled_instead_of_pinned_to_zero() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(2, 2, 1, 88);
        nas.initialize(&space).expect("initialize");
        let categorical = ParameterRange::Categorical(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ]);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..80 {
            let value = nas.sample_parameter(&categorical);
            assert!(
                (0.0..4.0).contains(&value),
                "a categorical index must be in range, got {value}"
            );
            seen.insert(value as usize);
        }
        assert!(
            seen.len() > 1,
            "every categorical draw returned the same value ({seen:?}); the axis is \
             not being searched"
        );
        // An empty category list has no index to return.
        assert_eq!(
            nas.sample_parameter(&ParameterRange::Categorical(Vec::new())),
            0.0
        );
    }

    #[test]
    fn seeding_is_reproducible_and_unseeded_instances_differ() {
        let space = search_space();
        let signature = |seed: Option<u64>| -> Vec<String> {
            let mut nas = match seed {
                Some(seed) => ProgressiveNAS::<f64>::with_seed(3, 4, 2, seed),
                None => ProgressiveNAS::<f64>::new(3, 4, 2),
            };
            nas.initialize(&space).expect("initialize");
            let history = VecDeque::new();
            (0..6)
                .map(|_| {
                    nas.generate_architecture(&space, &history)
                        .expect("generate")
                        .components
                        .join("+")
                })
                .collect()
        };
        assert_eq!(signature(Some(4242)), signature(Some(4242)));

        let mut differ = false;
        for _ in 0..5 {
            if signature(None) != signature(None) {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "two unseeded ProgressiveNAS instances must not always explore identically"
        );
    }

    #[test]
    fn statistics_report_measured_performance() {
        let space = search_space();
        let mut nas = ProgressiveNAS::<f64>::with_seed(2, 3, 1, 99);
        nas.initialize(&space).expect("initialize");
        let history = VecDeque::new();
        let architectures: Vec<OptimizerArchitecture<f64>> = (0..3)
            .map(|_| {
                nas.generate_architecture(&space, &history)
                    .expect("generate")
            })
            .collect();
        let scores = [0.25, 0.75, 0.5];
        let results: Vec<SearchResult<f64>> = architectures
            .iter()
            .zip(scores.iter())
            .map(|(arch, score)| result_for(arch, *score))
            .collect();
        nas.update_with_results(&results).expect("update");

        let stats = nas.get_statistics();
        assert_eq!(stats.total_architectures_generated, 3);
        assert!((stats.best_performance - 0.75).abs() < 1e-12);
        assert!((stats.average_performance - 0.5).abs() < 1e-12);
    }
}
