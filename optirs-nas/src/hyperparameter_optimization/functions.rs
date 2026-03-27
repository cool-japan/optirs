//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, Instant};
#[allow(unused_imports)]
use crate::error::Result;

use super::types::{HPOConfig, HPOResult, HPOStrategyStatistics, HyperparameterOptimizationPipeline, ParameterConfiguration, ParameterSearchSpace};

/// Base trait for HPO strategies
pub trait HPOStrategy<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Initialize the strategy
    fn initialize(&mut self, config: &HPOConfig<T>) -> Result<()>;
    /// Suggest next parameter configuration
    fn suggest(&mut self, history: &[HPOResult<T>]) -> Result<ParameterConfiguration<T>>;
    /// Update strategy with new results
    fn update(&mut self, result: &HPOResult<T>) -> Result<()>;
    /// Check if strategy should stop
    fn should_stop(&self) -> bool;
    /// Get strategy statistics
    fn get_statistics(&self) -> HPOStrategyStatistics<T>;
    /// Get strategy name
    fn name(&self) -> &str;
}
impl<T: Float + Debug + Send + Sync + 'static> Default for EvaluationBudget {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            max_time_seconds: 86400,
            max_flops: 1_000_000_000_000,
            early_stopping_patience: 50,
            min_evaluation_time: Duration::from_secs(60),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hpo_config_creation() {
        let config = HPOConfig::<f64>::default();
        assert!(! config.strategies.is_empty());
        assert!(config.early_stopping.enabled);
    }
    #[test]
    fn test_parameter_space_creation() {
        let space = ParameterSearchSpace::default();
        assert!(space.continuous_params.is_empty());
        assert!(space.discrete_params.is_empty());
    }
    #[test]
    fn test_hpo_pipeline_creation() {
        let config = HPOConfig::<f64>::default();
        let pipeline = HyperparameterOptimizationPipeline::new(config);
        assert!(pipeline.is_ok());
    }
}
