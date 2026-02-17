//! # ParallelismSettings - Trait Implementations
//!
//! This module contains trait implementations for `ParallelismSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{LoadBalancingStrategy, ParallelismSettings, ThreadPoolConfig};

impl Default for ParallelismSettings {
    fn default() -> Self {
        Self {
            max_parallel_stages: 4,
            stage_parallelism: HashMap::new(),
            thread_pool_config: ThreadPoolConfig::default(),
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}
