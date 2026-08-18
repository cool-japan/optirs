//! # `WeightedSumPriorityCalculator` - Trait Implementations
//!
//! This module contains trait implementations for `WeightedSumPriorityCalculator`.
//!
//! ## Implemented Traits
//!
//! - `PriorityCalculationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::PriorityCalculationAlgorithm;
use super::types::{
    AlgorithmComplexity, PriorityLevel, PriorityWeights, TaskContext, WeightedSumPriorityCalculator,
};

impl<T: Float + Debug + Send + Sync + 'static> PriorityCalculationAlgorithm<T>
    for WeightedSumPriorityCalculator
{
    fn calculate_priority(
        &self,
        task_context: &TaskContext<T>,
        weights: &PriorityWeights<T>,
    ) -> Result<PriorityLevel<T>> {
        let params = &task_context.parameters;
        let base_priority = Self::dimension(params, "base_priority");
        let urgency = Self::dimension(params, "urgency");
        let importance = Self::dimension(params, "importance");
        let efficiency = Self::dimension(params, "efficiency");
        let cost = Self::dimension(params, "cost");
        let quality = Self::dimension(params, "quality");
        let deadline_factor = Self::dimension(params, "deadline_factor");

        let mut dynamic_factors = HashMap::new();
        let mut dynamic_contribution = T::zero();
        for (key, weight) in &weights.dynamic_weights {
            let value = params.get(key).copied().unwrap_or_else(T::zero);
            dynamic_factors.insert(key.clone(), value);
            dynamic_contribution = dynamic_contribution + *weight * value;
        }

        // `cost` is lower-is-better, so it contributes as `(1 - cost)` --
        // consistent with every other dimension being higher-is-better.
        let composite_score = weights.base_weight * base_priority
            + weights.urgency_weight * urgency
            + weights.importance_weight * importance
            + weights.efficiency_weight * efficiency
            + weights.cost_weight * (T::one() - cost)
            + weights.quality_weight * quality
            + weights.deadline_weight * deadline_factor
            + dynamic_contribution;

        Ok(PriorityLevel {
            base_priority,
            urgency,
            importance,
            efficiency,
            cost,
            quality,
            deadline_factor,
            dynamic_factors,
            composite_score,
            weights: weights.clone(),
        })
    }

    fn name(&self) -> &str {
        "WeightedSum"
    }

    fn complexity(&self) -> AlgorithmComplexity {
        AlgorithmComplexity::Linear
    }
}
