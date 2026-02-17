//! # CompressionParams - Trait Implementations
//!
//! This module contains trait implementations for `CompressionParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::CompressionParams;

impl<T: Float + Debug + Send + Sync + 'static> Default for CompressionParams<T> {
    fn default() -> Self {
        Self {
            target_ratio: scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
            quality_threshold: scirs2_core::numeric::NumCast::from(0.95)
                .unwrap_or_else(|| T::zero()),
            max_time: 1000,
            strength: scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero()),
        }
    }
}
