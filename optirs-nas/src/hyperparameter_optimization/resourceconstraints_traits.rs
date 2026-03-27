//! # ResourceConstraints - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{ComputeResources, ResourceConstraints};

impl<T: Float + Debug + Send + Sync + 'static> Default for ResourceConstraints<T> {
    fn default() -> Self {
        Self {
            max_memory_gb: scirs2_core::numeric::NumCast::from(32.0)
                .unwrap_or_else(|| T::zero()),
            max_time_hours: scirs2_core::numeric::NumCast::from(24.0)
                .unwrap_or_else(|| T::zero()),
            max_cost: scirs2_core::numeric::NumCast::from(1000.0)
                .unwrap_or_else(|| T::zero()),
            compute_resources: ComputeResources::default(),
        }
    }
}

