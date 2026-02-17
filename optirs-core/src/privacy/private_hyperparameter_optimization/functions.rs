//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::{
    BudgetAllocationStrategy, EarlyStoppingConfig, HPOBudgetManager, HPOEvaluation, HPOResult,
    HyperparameterNoiseMechanism, ParameterBounds, ParameterConfiguration, ParameterDefinition,
    ParameterPrior, ParameterSpace, ParameterTransformation, ParameterType, PrivateHPOConfig,
    PrivateRandomSearch, SearchAlgorithm, SensitivityBounds, SmoothSensitivityParams,
    StatisticalTestResult, ValidationStrategy,
};

pub type ObjectiveFn<T> = Box<dyn Fn(&ParameterConfiguration<T>) -> Result<f64> + Send + Sync>;
pub type RuleFn<T> = Box<dyn Fn(&HPOResult<T>) -> bool + Send + Sync>;
pub type TestFn<T> = Box<dyn Fn(&[HPOResult<T>]) -> StatisticalTestResult + Send + Sync>;
/// Trait for noisy optimization algorithms
pub trait NoisyOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Suggest next hyperparameter configuration with privacy
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>>;
    /// Update optimizer with new evaluation result
    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()>;
    /// Get optimizer name
    fn name(&self) -> &str;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_private_hpoconfig() {
        let config = PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 100,
            cv_folds: 5,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 10,
                min_improvement: 0.01,
                max_evaluations: 100,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Gaussian,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::<String, f64>::new(),
                local_sensitivity: HashMap::<String, (f64, f64)>::new(),
                smooth_sensitivity: HashMap::<String, SmoothSensitivityParams<f64>>::new(),
            },
            private_model_selection: true,
            validation_strategy: ValidationStrategy::KFoldCV,
        };
        assert_eq!(config.num_evaluations, 100);
        assert_eq!(config.cv_folds, 5);
        assert!(config.early_stopping.enabled);
    }
    #[test]
    fn test_parameter_space() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "learning_rate".to_string(),
            ParameterDefinition {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous,
                bounds: ParameterBounds {
                    min: Some(0.001),
                    max: Some(0.1),
                    step: None,
                    valid_values: None,
                },
                prior: Some(ParameterPrior::LogNormal(-3.0, 1.0)),
                transformation: Some(ParameterTransformation::Log),
            },
        );
        let parameterspace = ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        };
        assert!(parameterspace.parameters.contains_key("learning_rate"));
    }
    #[test]
    fn test_budget_manager() {
        let baseconfig = DifferentialPrivacyConfig::default();
        let budget_manager = HPOBudgetManager::new(baseconfig, BudgetAllocationStrategy::Equal, 10)
            .expect("unwrap failed");
        assert!(budget_manager
            .has_budget_remaining()
            .expect("unwrap failed"));
    }
    #[test]
    fn test_private_random_search() {
        let config = PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 10,
            cv_folds: 3,
            early_stopping: EarlyStoppingConfig {
                enabled: false,
                patience: 5,
                min_improvement: 0.01,
                max_evaluations: 10,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Gaussian,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::<String, f64>::new(),
                local_sensitivity: HashMap::<String, (f64, f64)>::new(),
                smooth_sensitivity: HashMap::<String, SmoothSensitivityParams<f64>>::new(),
            },
            private_model_selection: false,
            validation_strategy: ValidationStrategy::HoldOut,
        };
        let optimizer = PrivateRandomSearch::new(config).expect("unwrap failed");
        assert_eq!(optimizer.name(), "PrivateRandomSearch");
    }
}
