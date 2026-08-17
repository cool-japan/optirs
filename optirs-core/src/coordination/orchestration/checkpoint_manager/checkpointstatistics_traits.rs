//! # `CheckpointStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `CheckpointStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types_15::CheckpointStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for CheckpointStatistics<T> {
    fn default() -> Self {
        Self {
            total_created: 0,
            total_restored: 0,
            total_deleted: 0,
            average_size_bytes: 0,
            average_creation_time: Duration::from_secs(0),
            average_restoration_time: Duration::from_secs(0),
            storage_utilization: T::zero(),
            success_rate: T::zero(),
        }
    }
}
