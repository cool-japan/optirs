//! # `IndexingStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `IndexingStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types_15::IndexingStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for IndexingStatistics<T> {
    fn default() -> Self {
        Self {
            total_entries: 0,
            index_size_bytes: 0,
            average_lookup_time: Duration::from_secs(0),
            efficiency: T::zero(),
            last_rebuild_time: Duration::from_secs(0),
        }
    }
}
