//! # MetricsQuality - Trait Implementations
//!
//! This module contains trait implementations for `MetricsQuality`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::MetricsQuality;

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for MetricsQuality<T> {
    fn default() -> Self {
        Self {
            completeness: T::one(),
            accuracy: T::one(),
            timeliness: T::one(),
            consistency: T::one(),
            overall_quality: T::one(),
        }
    }
}
