//! # ExperimentManagerStatistics - Trait Implementations
//!
//! This module contains trait implementations for `ExperimentManagerStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::ExperimentManagerStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for ExperimentManagerStatistics<T> {
    fn default() -> Self {
        Self {
            total_experiments: 0,
            completed_experiments: 0,
            failed_experiments: 0,
            average_duration: Duration::from_secs(0),
            success_rate: T::zero(),
        }
    }
}
