// Evolutionary search strategy using genetic algorithms

use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::config::ParameterRange;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// Evolutionary search strategy using genetic algorithms
pub struct EvolutionarySearch<T: Float + Debug + Send + Sync + 'static> {
    pub(crate) population: Vec<OptimizerArchitecture<T>>,
    pub population_size: usize,
    pub(crate) mutation_rate: f64,
    pub(crate) crossover_rate: f64,
    pub(crate) tournament_size: usize,
    pub(crate) generation_count: usize,
    pub(crate) statistics: SearchStrategyStatistics<T>,
    pub(crate) elite_preservation: bool,
    pub(crate) elitism_ratio: f64,
    pub(crate) adaptive_rates: bool,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    EvolutionarySearch<T>
{
    pub fn new(
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
    ) -> Self {
        Self {
            population: Vec::new(),
            population_size,
            mutation_rate,
            crossover_rate,
            tournament_size,
            generation_count: 0,
            statistics: SearchStrategyStatistics::default(),
            elite_preservation: true,
            elitism_ratio: 0.1,
            adaptive_rates: true,
        }
    }

    fn initialize_population(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        self.population.clear();

        // Use random search to generate initial population
        let mut random_search = RandomSearch::<T>::new(Some(42));
        random_search.initialize(searchspace)?;

        for _ in 0..self.population_size {
            let architecture =
                random_search.generate_architecture(searchspace, &VecDeque::new())?;
            self.population.push(architecture);
        }

        Ok(())
    }

    fn selection(&self, fitnessscores: &[T]) -> Result<usize> {
        // Tournament selection
        let mut best_idx = 0;
        let mut best_fitness = T::neg_infinity();

        let mut rng = scirs2_core::random::thread_rng();
        for _ in 0..self.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            if fitnessscores[idx] > best_fitness {
                best_fitness = fitnessscores[idx];
                best_idx = idx;
            }
        }

        Ok(best_idx)
    }

    fn crossover(
        &self,
        parent1: &OptimizerArchitecture<T>,
        parent2: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        let mut child_components = Vec::new();
        let max_len = parent1.components.len().max(parent2.components.len());

        for i in 0..max_len {
            let component = if i < parent1.components.len() && i < parent2.components.len() {
                // Crossover between components
                if scirs2_core::random::Random::default().random::<f64>() < 0.5 {
                    parent1.components[i].clone()
                } else {
                    parent2.components[i].clone()
                }
            } else if i < parent1.components.len() {
                parent1.components[i].clone()
            } else {
                parent2.components[i].clone()
            };

            child_components.push(component);
        }

        // Crossover parameters
        let mut child_parameters = HashMap::new();
        for (key, value) in &parent1.parameters {
            if scirs2_core::random::Random::default().random::<f64>() < 0.5 {
                child_parameters.insert(key.clone(), *value);
            } else if let Some(parent2_value) = parent2.parameters.get(key) {
                child_parameters.insert(key.clone(), *parent2_value);
            } else {
                child_parameters.insert(key.clone(), *value);
            }
        }

        Ok(OptimizerArchitecture {
            components: child_components,
            parameters: child_parameters,
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: format!("arch_{}", Random::default().random::<u64>()),
        })
    }

    fn mutate(
        &self,
        architecture: &mut OptimizerArchitecture<T>,
        searchspace: &SearchSpaceConfig,
    ) -> Result<()> {
        // Mutate parameters directly
        for (param_name, current_value) in architecture.parameters.iter_mut() {
            if scirs2_core::random::Random::default().random::<f64>() < self.mutation_rate {
                // Find the parameter range from search space config
                let param_range = searchspace
                    .components
                    .iter()
                    .flat_map(|c| c.hyperparameter_ranges.iter())
                    .find(|(name, _)| name == &param_name)
                    .map(|(_, range)| range);

                if let Some(param_range) = param_range {
                    match param_range {
                        ParameterRange::Continuous(min, max) => {
                            let noise =
                                scirs2_core::random::Random::default().random::<f64>() * 0.1 - 0.05;
                            let current_f64 = current_value.to_f64().unwrap_or(0.0);
                            let new_val = current_f64 + noise;
                            let clamped = new_val.max(*min).min(*max);
                            *current_value = scirs2_core::numeric::NumCast::from(clamped)
                                .unwrap_or_else(|| T::zero());
                        }
                        ParameterRange::LogUniform(min, max) => {
                            let current_f64 = current_value.to_f64().unwrap_or(0.001);
                            let log_val = current_f64.ln();
                            let noise =
                                scirs2_core::random::Random::default().random::<f64>() * 0.2 - 0.1;
                            let new_log = log_val + noise;
                            let new_val = new_log.exp().max(*min).min(*max);
                            *current_value = scirs2_core::numeric::NumCast::from(new_val)
                                .unwrap_or_else(|| T::zero());
                        }
                        _ => {
                            // For other types, regenerate randomly
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn calculate_recent_improvement(&self, performances: &[T]) -> T {
        if performances.len() < 10 {
            return T::zero();
        }

        let recent_avg = performances.iter().rev().take(5).cloned().sum::<T>()
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());
        let earlier_avg = performances
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .cloned()
            .sum::<T>()
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());

        recent_avg - earlier_avg
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    SearchStrategy<T> for EvolutionarySearch<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        self.initialize_population(searchspace)?;
        self.generation_count = 0;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        if self.population.is_empty() {
            self.initialize_population(searchspace)?;
        }

        // Extract fitness scores from recent history
        let recent_results: Vec<_> = history.iter().rev().take(self.population_size).collect();

        if recent_results.len() >= self.population_size {
            // Perform evolutionary step
            let fitnessscores: Vec<T> = recent_results
                .iter()
                .filter_map(|r| {
                    r.evaluation_results
                        .metric_scores
                        .get(&EvaluationMetric::FinalPerformance)
                })
                .cloned()
                .collect();

            if fitnessscores.len() == self.population_size {
                // Selection and reproduction
                let parent1_idx = self.selection(&fitnessscores)?;
                let parent2_idx = self.selection(&fitnessscores)?;

                let mut child = if scirs2_core::random::Random::default().random::<f64>()
                    < self.crossover_rate
                {
                    self.crossover(&self.population[parent1_idx], &self.population[parent2_idx])?
                } else {
                    self.population[parent1_idx].clone()
                };

                self.mutate(&mut child, searchspace)?;

                // Replace worst individual with child
                let worst_idx = fitnessscores
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                self.population[worst_idx] = child.clone();
                self.generation_count += 1;

                self.statistics.total_architectures_generated += 1;
                return Ok(child);
            }
        }

        // Fallback to random generation
        let idx = scirs2_core::random::thread_rng().gen_range(0..self.population.len());
        self.statistics.total_architectures_generated += 1;
        Ok(self.population[idx].clone())
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if !results.is_empty() {
            let performances: Vec<T> = results
                .iter()
                .filter_map(|r| {
                    r.evaluation_results
                        .metric_scores
                        .get(&EvaluationMetric::FinalPerformance)
                })
                .cloned()
                .collect();

            if !performances.is_empty() {
                self.statistics.best_performance = performances
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .cloned()
                    .unwrap_or(T::zero());

                let sum: T = performances.iter().cloned().sum();
                self.statistics.average_performance =
                    sum / T::from(performances.len()).expect("conversion from usize to T failed");

                // Adaptive rate adjustment
                if self.adaptive_rates && self.generation_count > 10 {
                    let recent_improvement = self.calculate_recent_improvement(&performances);
                    if recent_improvement
                        < scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero())
                    {
                        self.mutation_rate = (self.mutation_rate * 1.1).min(0.5);
                    } else {
                        self.mutation_rate = (self.mutation_rate * 0.95).max(0.01);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "EvolutionarySearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate =
            scirs2_core::numeric::NumCast::from(self.mutation_rate).unwrap_or_else(|| T::zero());
        stats.exploitation_rate = scirs2_core::numeric::NumCast::from(1.0 - self.mutation_rate)
            .unwrap_or_else(|| T::zero());
        stats
    }
}
