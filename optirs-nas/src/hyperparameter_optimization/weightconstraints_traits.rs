//! # WeightConstraints - Trait Implementations
//!
//! This module contains trait implementations for `WeightConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::WeightConstraints;

impl<T: Float + Debug + Send + Sync + 'static> Default for WeightConstraints<T> {
    fn default() -> Self {
        Self {
            min_weight: scirs2_core::numeric::NumCast::from(0.01)
                .unwrap_or_else(|| T::zero()),
            max_weight: T::one(),
            sum_constraint: Some(T::one()),
            smoothness_constraint: None,
        }
    }
}

