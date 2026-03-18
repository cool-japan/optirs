// Random search baseline strategy

use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};
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
    pub fn new(seed: Option<u64>) -> Self {
        let rng = if let Some(seed) = seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        Self {
            rng,
            statistics: SearchStrategyStatistics::default(),
            searchspace: None,
        }
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

        // Randomly select number of components
        let num_components = self.rng.gen_range(1..5);
        let mut components = Vec::new();

        for idx in 0..num_components {
            // Randomly select component type
            let component_config =
                &searchspace.components[self.rng.gen_range(0..searchspace.components.len())];

            let mut hyperparameters = HashMap::new();

            // Randomly sample hyperparameters
            for (param_name, param_range) in &component_config.hyperparameter_ranges {
                let value = match param_range {
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
                        let value_idx = self.rng.gen_range(0..values.len());
                        values[value_idx]
                    }
                    ParameterRange::Categorical(_values) => {
                        // For categorical, we'll use index as value
                        0.0
                        // Simplified
                    }
                };

                hyperparameters.insert(param_name.clone(), value);
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
                _ => ComponentType::Adam, // Default fallback
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

        Ok(OptimizerArchitecture {
            components: components
                .into_iter()
                .map(|c| format!("{:?}", c.component_type))
                .collect(),
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: format!("arch_{}", self.rng.random::<u32>()),
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
                self.statistics.average_performance =
                    sum / T::from(performances.len()).expect("conversion from usize to T failed");
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
