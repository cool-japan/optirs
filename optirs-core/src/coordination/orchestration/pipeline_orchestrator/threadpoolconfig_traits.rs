//! # ThreadPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `ThreadPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::ThreadPoolConfig;

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            core_pool_size: 4,
            max_pool_size: 8,
            keep_alive_time: Duration::from_secs(60),
            queue_capacity: 1000,
        }
    }
}
