//! # PipelineStatistics - Trait Implementations
//!
//! This module contains trait implementations for `PipelineStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::PipelineStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for PipelineStatistics<T> {
    fn default() -> Self {
        Self {
            total_executed: 0,
            total_completed: 0,
            total_failed: 0,
            average_execution_time: Duration::from_secs(0),
            success_rate: T::zero(),
        }
    }
}
