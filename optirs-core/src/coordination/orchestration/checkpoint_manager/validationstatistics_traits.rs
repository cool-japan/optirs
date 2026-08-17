//! # `ValidationStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `ValidationStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::ValidationStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for ValidationStatistics<T> {
    fn default() -> Self {
        Self {
            total_validations: 0,
            total_passed: 0,
            total_failed: 0,
            average_validation_time: Duration::from_secs(0),
            success_rate: T::zero(),
        }
    }
}
