//! # `PriorityWeights` - Trait Implementations
//!
//! This module contains trait implementations for `PriorityWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::PriorityWeights;

impl<T: Float + Debug + Default + Send + Sync> Default for PriorityWeights<T> {
    fn default() -> Self {
        Self {
            base_weight: T::from(0.2).unwrap_or_else(|| T::zero()),
            urgency_weight: T::from(0.25).unwrap_or_else(|| T::zero()),
            importance_weight: T::from(0.2).unwrap_or_else(|| T::zero()),
            efficiency_weight: T::from(0.15).unwrap_or_else(|| T::zero()),
            cost_weight: T::from(0.1).unwrap_or_else(|| T::zero()),
            quality_weight: T::from(0.05).unwrap_or_else(|| T::zero()),
            deadline_weight: T::from(0.05).unwrap_or_else(|| T::zero()),
            dynamic_weights: HashMap::new(),
        }
    }
}
