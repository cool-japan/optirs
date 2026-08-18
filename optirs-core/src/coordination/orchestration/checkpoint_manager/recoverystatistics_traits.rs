//! # `RecoveryStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `RecoveryStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types_15::RecoveryStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for RecoveryStatistics<T> {
    fn default() -> Self {
        Self {
            total_attempts: 0,
            successful_recoveries: 0,
            failed_recoveries: 0,
            average_recovery_time: Duration::from_secs(0),
            success_rate: T::zero(),
        }
    }
}
