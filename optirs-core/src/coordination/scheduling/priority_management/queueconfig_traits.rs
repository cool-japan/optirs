//! # `QueueConfig` - Trait Implementations
//!
//! This module contains trait implementations for `QueueConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::QueueConfig;

impl<T: Float + Debug + Default + Send + Sync> Default for QueueConfig<T> {
    fn default() -> Self {
        Self {
            max_size: Some(10000),
            update_frequency: Duration::from_secs(10),
            maintenance_interval: Duration::from_secs(60),
            performance_threshold: T::from(0.8).unwrap_or_else(|| T::zero()),
        }
    }
}
