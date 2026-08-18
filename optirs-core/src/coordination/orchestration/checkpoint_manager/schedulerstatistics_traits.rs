//! # `SchedulerStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `SchedulerStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types_15::SchedulerStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for SchedulerStatistics<T> {
    fn default() -> Self {
        Self {
            total_scheduled: 0,
            total_completed: 0,
            total_failed: 0,
            average_latency: Duration::from_secs(0),
            efficiency: T::zero(),
        }
    }
}
