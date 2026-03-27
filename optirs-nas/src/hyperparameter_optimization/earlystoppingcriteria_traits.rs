//! # EarlyStoppingCriteria - Trait Implementations
//!
//! This module contains trait implementations for `EarlyStoppingCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::types::EarlyStoppingCriteria;

impl<T: Float + Debug + Send + Sync + 'static> Default for EarlyStoppingCriteria<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 50,
            min_improvement: scirs2_core::numeric::NumCast::from(0.001)
                .unwrap_or_else(|| T::zero()),
            relative_improvement: scirs2_core::numeric::NumCast::from(0.01)
                .unwrap_or_else(|| T::zero()),
            target_performance: None,
            max_evaluations: Some(1000),
            max_time: Some(Duration::from_hours(24)),
        }
    }
}

