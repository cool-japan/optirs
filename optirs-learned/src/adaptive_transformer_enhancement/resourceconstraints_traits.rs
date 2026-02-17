//! # ResourceConstraints - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceConstraints;

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory: 8192,
            max_computation_time: 5000,
            max_parameters: 1_000_000,
            energy_budget: None,
        }
    }
}

