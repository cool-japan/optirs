//! # ParallelizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `ParallelizationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LoadBalancingStrategy, ParallelizationSettings, SynchronizationStrategy};

impl Default for ParallelizationSettings {
    fn default() -> Self {
        Self {
            num_workers: 4,
            synchronization: SynchronizationStrategy::Asynchronous,
            load_balancing: LoadBalancingStrategy::Dynamic,
            communication_overhead_limit: 0.1,
        }
    }
}

