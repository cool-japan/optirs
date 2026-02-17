//! # GradientCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `GradientCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::GradientCharacteristics;

impl<T: Float + Debug + Send + Sync + 'static> Default for GradientCharacteristics<T> {
    fn default() -> Self {
        Self {
            gradient_norm: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            consistency: scirs2_core::numeric::NumCast::from(0.8).unwrap_or_else(|| T::zero()),
            noise_ratio: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            correlation: scirs2_core::numeric::NumCast::from(0.7).unwrap_or_else(|| T::zero()),
        }
    }
}
