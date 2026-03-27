//! # RealTimeConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{InterruptStrategy, RealTimeConfig};

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            scheduling_priority: 50,
            cpu_affinity: None,
            memory_preallocation_mb: 64,
            numa_optimization: true,
            deadline_us: 10000,
            lock_free_structures: true,
            interrupt_strategy: InterruptStrategy::Deferred,
        }
    }
}
