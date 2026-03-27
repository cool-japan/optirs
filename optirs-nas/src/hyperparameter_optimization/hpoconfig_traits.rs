//! # HPOConfig - Trait Implementations
//!
//! This module contains trait implementations for `HPOConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{EarlyStoppingCriteria, EnsembleSettings, HPOConfig, HPOStrategyType, MultiObjectiveSettings, ObjectiveFunctionType, OptimizationDirection, OptimizationObjective, ParallelizationSettings, ParameterSearchSpace, ResourceConstraints};

impl<T: Float + Debug + Send + Sync + 'static> Default for HPOConfig<T> {
    fn default() -> Self {
        Self {
            strategies: vec![
                HPOStrategyType::BayesianOptimization, HPOStrategyType::RandomSearch
            ],
            parameter_space: ParameterSearchSpace::default(),
            evaluation_budget: EvaluationBudget::default(),
            multi_objective: MultiObjectiveSettings::default(),
            early_stopping: EarlyStoppingCriteria::default(),
            multi_fidelity: None,
            ensemble_settings: EnsembleSettings::default(),
            parallelization: ParallelizationSettings::default(),
            resource_constraints: ResourceConstraints::default(),
            objectives: vec![
                OptimizationObjective { name : "performance".to_string(), function :
                ObjectiveFunctionType::Performance("accuracy".to_string()), direction :
                OptimizationDirection::Maximize, weight : T::one(), bounds : None }
            ],
            constraints: vec![],
        }
    }
}

