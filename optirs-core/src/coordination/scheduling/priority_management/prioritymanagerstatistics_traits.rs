//! # `PriorityManagerStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `PriorityManagerStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::PriorityManagerStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for PriorityManagerStatistics<T> {
    fn default() -> Self {
        Self {
            total_updates: 0,
            average_update_latency: Duration::from_millis(5),
            accuracy: T::from(0.8).unwrap_or_else(|| T::zero()),
            effectiveness: T::from(0.75).unwrap_or_else(|| T::zero()),
            learning_performance: T::from(0.6).unwrap_or_else(|| T::zero()),
        }
    }
}
