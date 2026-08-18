// Random search baseline strategy

use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::architecture::{ComponentPosition, ComponentType};
use crate::error::{OptimError, Result};
use crate::nas_engine::config::{ComponentType as ConfigComponentType, ParameterRange};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::{SearchStrategy, SearchStrategyStatistics};

/// Random search baseline strategy
pub struct RandomSearch<T: Float + Debug + Send + Sync + 'static + std::iter::Sum> {
    pub(crate) rng: Random<scirs2_core::random::rngs::StdRng>,
    pub(crate) statistics: SearchStrategyStatistics<T>,
    pub(crate) searchspace: Option<SearchSpaceConfig>,
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > RandomSearch<T>
{
    /// Create a random-search strategy.
    ///
    /// `Some(seed)` gives a fully reproducible sequence. `None` draws a seed
    /// from OS entropy, so two independently created searches explore
    /// different regions — the previous behaviour of falling back to a
    /// hard-coded seed made every unseeded search identical.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = Random::seed(seed.unwrap_or_else(scirs2_core::random::random::<u64>));

        Self {
            rng,
            statistics: SearchStrategyStatistics::default(),
            searchspace: None,
        }
    }

    /// Sample a single value from a parameter range.
    ///
    /// Every branch guards against a degenerate range: `gen_range` panics on an
    /// empty range, so `min >= max` collapses to the lower bound and an empty
    /// `Discrete` set is reported as a search-space error rather than
    /// crashing the search.
    fn sample_parameter(&mut self, name: &str, range: &ParameterRange) -> Result<f64> {
        let value = match range {
            ParameterRange::Continuous(min, max) => {
                if min < max {
                    self.rng.gen_range(*min..*max)
                } else {
                    *min
                }
            }
            ParameterRange::LogUniform(min, max) => {
                if *min <= 0.0 || *max <= 0.0 {
                    return Err(OptimError::SearchSpaceError(format!(
                        "log-uniform range for '{}' must be strictly positive, got ({}, {})",
                        name, min, max
                    )));
                }
                let log_min = min.ln();
                let log_max = max.ln();
                if log_min < log_max {
                    self.rng.gen_range(log_min..log_max).exp()
                } else {
                    *min
                }
            }
            ParameterRange::Integer(min, max) => {
                if min < max {
                    self.rng.gen_range(*min..*max) as f64
                } else {
                    *min as f64
                }
            }
            ParameterRange::Boolean => {
                if self.rng.random::<f64>() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ParameterRange::Discrete(values) => {
                if values.is_empty() {
                    return Err(OptimError::SearchSpaceError(format!(
                        "discrete parameter range for '{}' is empty",
                        name
                    )));
                }
                let value_idx = self.rng.gen_range(0..values.len());
                values[value_idx]
            }
            ParameterRange::Categorical(values) => {
                if values.is_empty() {
                    return Err(OptimError::SearchSpaceError(format!(
                        "categorical parameter range for '{}' is empty",
                        name
                    )));
                }
                // Categorical values are encoded as their index in the declared
                // vocabulary, which keeps the numeric representation stable and
                // lets the choice actually vary across samples.
                self.rng.gen_range(0..values.len()) as f64
            }
        };

        Ok(value)
    }
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + 'static + std::fmt::Debug + std::iter::Sum,
    > SearchStrategy<T> for RandomSearch<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        self.searchspace = Some(searchspace.clone());
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        use crate::architecture::OptimizerComponent;

        // An empty component vocabulary cannot be sampled from. Report it
        // instead of indexing into an empty slice (which used to panic).
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "search space declares no components; \
                 SearchSpaceConfig::components must contain at least one entry"
                    .to_string(),
            ));
        }

        // Randomly select the number of components, respecting the configured
        // bounds so a sampled architecture is valid by construction.
        let min_components = searchspace.min_components.max(1);
        let max_components = searchspace.max_components.max(min_components);
        let num_components = if min_components < max_components {
            self.rng.gen_range(min_components..=max_components)
        } else {
            min_components
        };

        let mut components = Vec::new();
        // Architecture-level hyperparameters: the union of every selected
        // component's sampled values, prefixed by the component index when a
        // name repeats. Populating this map is what lets downstream consumers
        // (evaluation, predictors, encoders) see the sampled hyperparameters at
        // all — they used to be dropped on the floor.
        let mut architecture_hyperparameters: HashMap<String, T> = HashMap::new();

        for idx in 0..num_components {
            // Randomly select component type
            let component_index = self.rng.gen_range(0..searchspace.components.len());
            let component_config = searchspace.components[component_index].clone();

            let mut hyperparameters = HashMap::new();

            // Randomly sample hyperparameters
            let mut parameter_names: Vec<&String> =
                component_config.hyperparameter_ranges.keys().collect();
            // Sort so the sequence of RNG draws is deterministic for a given
            // seed regardless of HashMap iteration order.
            parameter_names.sort();

            for param_name in parameter_names {
                let param_range = match component_config.hyperparameter_ranges.get(param_name) {
                    Some(range) => range,
                    None => continue,
                };
                let value = self.sample_parameter(param_name, param_range)?;

                hyperparameters.insert(param_name.clone(), value);

                let scoped_name = if idx == 0 {
                    param_name.clone()
                } else {
                    format!("{}_{}", param_name, idx)
                };
                architecture_hyperparameters.insert(
                    scoped_name,
                    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(|| T::zero()),
                );
            }

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
                // Any component type not covered above still maps onto the
                // architecture-level vocabulary; Adam is the documented
                // fallback rather than a silent drop.
                _ => ComponentType::Adam,
            };

            components.push(OptimizerComponent {
                id: format!("comp_{}", idx),
                component_type,
                hyperparameters,
                enabled: true,
                position: ComponentPosition {
                    layer: 0,
                    index: idx as u32,
                    x: 0.0,
                    y: 0.0,
                },
            });
        }

        self.statistics.total_architectures_generated += 1;

        // Sequential connections between consecutive components, capped by the
        // configured `max_connections`.
        let connection_budget = searchspace.max_connections;
        let connections: Vec<(usize, usize)> = (0..components.len().saturating_sub(1))
            .take(connection_budget)
            .map(|i| (i, i + 1))
            .collect();

        Ok(OptimizerArchitecture {
            components: components
                .into_iter()
                .map(|c| format!("{:?}", c.component_type))
                .collect(),
            parameters: architecture_hyperparameters.clone(),
            connections,
            metadata: HashMap::new(),
            hyperparameters: architecture_hyperparameters,
            architecture_id: format!("arch_{:08x}", self.rng.random::<u32>()),
        })
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
                let count: T = scirs2_core::numeric::NumCast::from(performances.len())
                    .unwrap_or_else(|| T::one());
                self.statistics.average_performance = sum / count;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "RandomSearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        self.statistics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::config::OptimizerComponentConfig;

    /// Two-component search space with a continuous and a log-uniform range.
    fn search_space() -> SearchSpaceConfig {
        SearchSpaceConfig {
            components: vec![
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::Adam,
                    hyperparameter_ranges: {
                        let mut ranges = HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::LogUniform(1e-4, 1e-1),
                        );
                        ranges.insert("beta1".to_string(), ParameterRange::Continuous(0.8, 0.99));
                        ranges
                    },
                    complexity_score: 1.0,
                    memory_requirement: 1024,
                    computational_cost: 1.0,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::SGD,
                    hyperparameter_ranges: {
                        let mut ranges = HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::Continuous(1e-3, 1e-1),
                        );
                        ranges
                    },
                    complexity_score: 0.5,
                    memory_requirement: 512,
                    computational_cost: 0.5,
                    compatibility_constraints: Vec::new(),
                },
            ],
            min_components: 1,
            max_components: 4,
            max_connections: 8,
            ..SearchSpaceConfig::default()
        }
    }

    fn generate(
        search: &mut RandomSearch<f64>,
        space: &SearchSpaceConfig,
        n: usize,
    ) -> Vec<String> {
        (0..n)
            .filter_map(|_| {
                search
                    .generate_architecture(space, &VecDeque::new())
                    .ok()
                    .map(|a| format!("{:?}|{:?}", a.components, sorted_hyper(&a)))
            })
            .collect()
    }

    fn sorted_hyper(arch: &OptimizerArchitecture<f64>) -> Vec<(String, String)> {
        let mut entries: Vec<(String, String)> = arch
            .hyperparameters
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:.12}", v)))
            .collect();
        entries.sort();
        entries
    }

    #[test]
    fn seeded_searches_are_reproducible() {
        let space = search_space();
        let mut a = RandomSearch::<f64>::new(Some(7));
        let mut b = RandomSearch::<f64>::new(Some(7));

        assert_eq!(generate(&mut a, &space, 8), generate(&mut b, &space, 8));
    }

    #[test]
    fn different_seeds_diverge() {
        let space = search_space();
        let mut a = RandomSearch::<f64>::new(Some(1));
        let mut b = RandomSearch::<f64>::new(Some(2));

        assert_ne!(generate(&mut a, &space, 8), generate(&mut b, &space, 8));
    }

    #[test]
    fn unseeded_searches_differ() {
        // Regression test for the old `None => Random::seed(42)` behaviour,
        // which made every unseeded search produce the identical sequence.
        let space = search_space();
        let mut differed = false;
        for _ in 0..8 {
            let mut a = RandomSearch::<f64>::new(None);
            let mut b = RandomSearch::<f64>::new(None);
            if generate(&mut a, &space, 8) != generate(&mut b, &space, 8) {
                differed = true;
                break;
            }
        }
        assert!(
            differed,
            "two unseeded RandomSearch instances must not produce identical sequences"
        );
    }

    #[test]
    fn sampled_hyperparameters_are_recorded_and_vary() {
        let space = search_space();
        let mut search = RandomSearch::<f64>::new(Some(11));

        let mut seen_learning_rates = std::collections::HashSet::new();
        for _ in 0..25 {
            let arch = search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            assert!(
                !arch.hyperparameters.is_empty(),
                "sampled hyperparameters must reach the architecture"
            );
            assert_eq!(
                arch.hyperparameters, arch.parameters,
                "parameters mirror hyperparameters"
            );
            if let Some(lr) = arch.hyperparameters.get("learning_rate") {
                assert!(*lr > 0.0 && *lr <= 0.1);
                seen_learning_rates.insert(format!("{:.9}", lr));
            }
        }
        assert!(
            seen_learning_rates.len() > 1,
            "learning rates must actually vary across samples"
        );
    }

    #[test]
    fn respects_component_count_bounds() {
        let mut space = search_space();
        space.min_components = 2;
        space.max_components = 3;
        let mut search = RandomSearch::<f64>::new(Some(3));

        for _ in 0..30 {
            let arch = search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            assert!(arch.components.len() >= 2 && arch.components.len() <= 3);
            assert!(arch.connections.len() <= space.max_connections);
        }
    }

    #[test]
    fn empty_search_space_is_an_error_not_a_panic() {
        let mut space = search_space();
        space.components.clear();
        let mut search = RandomSearch::<f64>::new(Some(1));

        let err = search.generate_architecture(&space, &VecDeque::new());
        assert!(matches!(err, Err(OptimError::SearchSpaceError(_))));
    }

    #[test]
    fn degenerate_ranges_do_not_panic() {
        let mut space = search_space();
        space.components[0].hyperparameter_ranges = {
            let mut ranges = HashMap::new();
            // min == max: `gen_range` would panic on an empty range.
            ranges.insert("fixed".to_string(), ParameterRange::Continuous(0.5, 0.5));
            ranges.insert("int_fixed".to_string(), ParameterRange::Integer(3, 3));
            ranges.insert("flag".to_string(), ParameterRange::Boolean);
            ranges
        };
        space.components.truncate(1);
        let mut search = RandomSearch::<f64>::new(Some(5));

        let arch = search
            .generate_architecture(&space, &VecDeque::new())
            .expect("degenerate ranges must be handled, not panic");
        assert_eq!(arch.hyperparameters.get("fixed"), Some(&0.5));
    }

    #[test]
    fn empty_discrete_range_is_an_error() {
        let mut space = search_space();
        space.components.truncate(1);
        space.components[0].hyperparameter_ranges = {
            let mut ranges = HashMap::new();
            ranges.insert("choice".to_string(), ParameterRange::Discrete(Vec::new()));
            ranges
        };
        let mut search = RandomSearch::<f64>::new(Some(5));

        let err = search.generate_architecture(&space, &VecDeque::new());
        assert!(matches!(err, Err(OptimError::SearchSpaceError(_))));
    }
}
