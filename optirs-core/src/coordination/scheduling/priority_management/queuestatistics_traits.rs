//! # `QueueStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `QueueStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::QueueStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for QueueStatistics<T> {
    fn default() -> Self {
        Self {
            total_processed: 0,
            average_length: T::zero(),
            average_wait_time: Duration::from_secs(0),
            throughput: T::zero(),
            efficiency: T::from(0.5).unwrap_or_else(|| T::zero()),
        }
    }
}
