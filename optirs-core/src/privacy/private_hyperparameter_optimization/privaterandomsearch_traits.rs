//! # PrivateRandomSearch - Trait Implementations
//!
//! This module contains trait implementations for `PrivateRandomSearch`.
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
    ParameterValue, PrivateRandomSearch,
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
                    let random_val = T::from(self.rng.gen_range(0.0..1.0)).expect("unwrap failed");
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
        _result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str {
        "PrivateRandomSearch"
    }
}
