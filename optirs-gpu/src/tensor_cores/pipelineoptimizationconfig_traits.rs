//! # PipelineOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `PipelineOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LoadBalancingStrategy, PipelineOptimizationConfig};

impl Default for PipelineOptimizationConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            dependency_tracking: true,
            prefetch_distance: 2,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            priority_scheduling: true,
        }
    }
}
