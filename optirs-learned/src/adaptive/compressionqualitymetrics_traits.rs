//! # CompressionQualityMetrics - Trait Implementations
//!
//! This module contains trait implementations for `CompressionQualityMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::CompressionQualityMetrics;

impl<T: Float + Debug + Send + Sync + 'static> Default for CompressionQualityMetrics<T> {
    fn default() -> Self {
        Self {
            reconstruction_error: scirs2_core::numeric::NumCast::from(0.05)
                .unwrap_or_else(|| T::zero()),
            information_loss: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            compression_ratio: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            compression_time: 100,
        }
    }
}
