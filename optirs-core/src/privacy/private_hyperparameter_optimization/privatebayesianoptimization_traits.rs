//! # PrivateBayesianOptimization - Trait Implementations
//!
//! This module contains trait implementations for `PrivateBayesianOptimization`.
//!
//! ## Implemented Traits
//!
//! - `NoisyOptimizer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::NoisyOptimizer;
use super::types::{
    HPOEvaluation, HPOResult, ParameterConfiguration, ParameterSpace, ParameterType,
    ParameterValue, PrivateBayesianOptimization,
};

impl<T: Float + Debug + Send + Sync + 'static> NoisyOptimizer<T>
    for PrivateBayesianOptimization<T>
{
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>> {
        if evaluation_history.is_empty() {
            let mut rng = scirs2_core::random::Random::seed(42);
            let mut values = HashMap::new();
            for (param_name, param_def) in &parameterspace.parameters {
                let value = match &param_def.param_type {
                    ParameterType::Continuous => {
                        let min = param_def.bounds.min.unwrap_or(T::zero());
                        let max = param_def.bounds.max.unwrap_or(T::one());
                        let random_val = T::from(rng.gen_range(0.0..1.0)).expect("unwrap failed");
                        ParameterValue::Continuous(min + random_val * (max - min))
                    }
                    _ => ParameterValue::Continuous(T::from(0.5).unwrap_or_else(|| T::zero())),
                };
                values.insert(param_name.clone(), value);
            }
            return Ok(ParameterConfiguration {
                values,
                id: "initialconfig".to_string(),
                metadata: HashMap::new(),
            });
        }
        Ok(ParameterConfiguration {
            values: HashMap::new(),
            id: format!("bayesianconfig_{}", evaluation_history.len()),
            metadata: HashMap::new(),
        })
    }
    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        _result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str {
        "PrivateBayesianOptimization"
    }
}
