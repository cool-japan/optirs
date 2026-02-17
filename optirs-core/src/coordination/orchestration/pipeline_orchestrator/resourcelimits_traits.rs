//! # ResourceLimits - Trait Implementations
//!
//! This module contains trait implementations for `ResourceLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::ResourceLimits;

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for ResourceLimits<T> {
    fn default() -> Self {
        Self {
            max_cpu: T::from(1.0).unwrap_or_else(T::one),
            max_memory_mb: 1024,
            max_gpu: T::from(1.0).unwrap_or_else(T::one),
            max_storage_gb: 10,
            max_network_mbps: 100.0,
            custom_limits: HashMap::new(),
        }
    }
}
