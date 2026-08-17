//! `NoisyOptimizer` for private random search.
//!
//! # The defect this replaces
//!
//! `update` was `Ok(())`, so `history` never grew and every proposal was
//! therefore named `config_0`; the results of each trial were discarded.

use crate::error::{OptimError, Result};
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::NoisyOptimizer;
use super::types::{
    unix_timestamp, HPOEvaluation, HPOResult, ParameterConfiguration, ParameterSpace,
    ParameterType, ParameterValue, PrivateRandomSearch,
};

impl<T: Float + Debug + Send + Sync + 'static> NoisyOptimizer<T> for PrivateRandomSearch<T> {
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        _evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>> {
        let mut values = HashMap::new();
        for (param_name, param_def) in &parameterspace.parameters {
            let value = match &param_def.param_type {
                ParameterType::Continuous => {
                    let min = param_def.bounds.min.unwrap_or(T::zero());
                    let max = param_def.bounds.max.unwrap_or(T::one());
                    let raw: f64 = self.rng.gen_range(0.0..1.0);
                    let random_val = T::from(raw).ok_or_else(|| {
                        OptimError::InvalidConfig(format!(
                            "failed to convert the sampled value {raw} into the parameter type"
                        ))
                    })?;
                    ParameterValue::Continuous(min + random_val * (max - min))
                }
                ParameterType::Integer => {
                    let min = param_def
                        .bounds
                        .min
                        .unwrap_or(T::zero())
                        .to_i64()
                        .unwrap_or(0);
                    let max = param_def
                        .bounds
                        .max
                        .unwrap_or(T::from(100).unwrap_or_else(|| T::zero()))
                        .to_i64()
                        .unwrap_or(100);
                    ParameterValue::Integer(self.rng.gen_range(min..max + 1))
                }
                ParameterType::Boolean => ParameterValue::Boolean(self.rng.gen_range(0..2) == 1),
                ParameterType::Categorical(categories) => {
                    let idx = self.rng.gen_range(0..categories.len());
                    ParameterValue::Categorical(categories[idx].clone())
                }
                ParameterType::Ordinal(values) => {
                    let idx = self.rng.gen_range(0..values.len());
                    ParameterValue::Ordinal(idx)
                }
            };
            values.insert(param_name.clone(), value);
        }
        Ok(ParameterConfiguration {
            values,
            id: format!("config_{}", self.history.len()),
            metadata: HashMap::new(),
        })
    }
    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        result: &HPOResult<T>,
        privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        // `Ok(())` here is what kept `history` permanently empty, which in turn
        // made every generated configuration id `config_0`.
        let index = self.history.len();
        self.history.push(HPOEvaluation {
            id: format!("random_search_{index}"),
            configuration: config.clone(),
            result: result.clone(),
            privacy_cost: privacy_budget.clone(),
            timestamp: unix_timestamp()?,
            metadata: HashMap::new(),
        });
        Ok(())
    }
    fn name(&self) -> &str {
        "PrivateRandomSearch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::{
        BudgetAllocationStrategy, EarlyStoppingConfig, HyperparameterNoiseMechanism,
        ParameterBounds, ParameterDefinition, PrivateBayesianOptimization, PrivateHPOConfig,
        SearchAlgorithm, SensitivityBounds, ValidationStrategy,
    };

    fn hpo_config() -> PrivateHPOConfig<f64> {
        PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 8,
            cv_folds: 3,
            early_stopping: EarlyStoppingConfig {
                enabled: false,
                patience: 2,
                min_improvement: 1e-3,
                max_evaluations: 8,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Gaussian,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::new(),
                local_sensitivity: HashMap::new(),
                smooth_sensitivity: HashMap::new(),
            },
            private_model_selection: true,
            validation_strategy: ValidationStrategy::HoldOut,
        }
    }

    fn one_continuous_parameter() -> ParameterSpace<f64> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "learning_rate".to_string(),
            ParameterDefinition {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous,
                bounds: ParameterBounds {
                    min: Some(0.0),
                    max: Some(1.0),
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        }
    }

    fn first_value(config: &ParameterConfiguration<f64>) -> f64 {
        match config.values.get("learning_rate") {
            Some(ParameterValue::Continuous(value)) => *value,
            other => panic!("expected a continuous learning_rate, got {other:?}"),
        }
    }

    fn propose<O: NoisyOptimizer<f64>>(optimizer: &mut O, space: &ParameterSpace<f64>) -> f64 {
        let budget = PrivacyBudget::default();
        match optimizer.suggest_next(space, &[], &budget) {
            Ok(config) => first_value(&config),
            Err(err) => panic!("suggest_next failed: {err}"),
        }
    }

    #[test]
    fn random_search_proposals_are_not_seeded_from_a_constant() {
        // Regression for the hardcoded `Random::seed(42)`: two independently
        // constructed searches used to walk exactly the same trajectory, so
        // the set of configurations evaluated against the private dataset was
        // public knowledge.
        let space = one_continuous_parameter();
        let mut first = match PrivateRandomSearch::<f64>::new(hpo_config()) {
            Ok(search) => search,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut second = match PrivateRandomSearch::<f64>::new(hpo_config()) {
            Ok(search) => search,
            Err(err) => panic!("construction failed: {err}"),
        };

        let left: Vec<f64> = (0..8).map(|_| propose(&mut first, &space)).collect();
        let right: Vec<f64> = (0..8).map(|_| propose(&mut second, &space)).collect();
        assert_ne!(
            left, right,
            "two searches must not propose an identical trajectory"
        );
        assert!(left.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn random_search_with_an_explicit_seed_is_reproducible() {
        let space = one_continuous_parameter();
        let mut first = match PrivateRandomSearch::<f64>::new_with_seed(hpo_config(), 7) {
            Ok(search) => search,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut second = match PrivateRandomSearch::<f64>::new_with_seed(hpo_config(), 7) {
            Ok(search) => search,
            Err(err) => panic!("construction failed: {err}"),
        };

        let left: Vec<f64> = (0..8).map(|_| propose(&mut first, &space)).collect();
        let right: Vec<f64> = (0..8).map(|_| propose(&mut second, &space)).collect();
        assert_eq!(left, right, "an explicit seed must be reproducible");
    }

    #[test]
    fn bayesian_optimizer_initial_proposal_is_not_a_compile_time_constant() {
        // The initial (history-free) configuration used to be drawn from a
        // freshly constructed `Random::seed(42)` *inside* `suggest_next`, so
        // every process proposed exactly the same starting point.
        let space = one_continuous_parameter();
        let mut first = match PrivateBayesianOptimization::<f64>::new(hpo_config()) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        let mut second = match PrivateBayesianOptimization::<f64>::new(hpo_config()) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };

        let left: Vec<f64> = (0..8).map(|_| propose(&mut first, &space)).collect();
        let right: Vec<f64> = (0..8).map(|_| propose(&mut second, &space)).collect();
        assert_ne!(
            left, right,
            "the initial Bayesian proposal must not be a constant"
        );
    }
}
