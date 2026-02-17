//! # StorageStatistics - Trait Implementations
//!
//! This module contains trait implementations for `StorageStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::StorageStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for StorageStatistics<T> {
    fn default() -> Self {
        Self {
            storage_rate: T::zero(),
            query_rate: T::zero(),
            average_storage_latency: Duration::from_secs(0),
            average_query_latency: Duration::from_secs(0),
            storage_errors: 0,
            query_errors: 0,
        }
    }
}
