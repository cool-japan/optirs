//! # LandscapeFeatures - Trait Implementations
//!
//! This module contains trait implementations for `LandscapeFeatures`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{CurvatureInfo, GradientCharacteristics, LandscapeFeatures};

impl<T: Float + Debug + Send + Sync + 'static> Default for LandscapeFeatures<T> {
    fn default() -> Self {
        Self {
            smoothness: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            multimodality: scirs2_core::numeric::NumCast::from(0.3)
                .unwrap_or_else(|| T::zero()),
            noise_level: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            curvature: CurvatureInfo::default(),
            gradient_characteristics: GradientCharacteristics::default(),
        }
    }
}

