//! # CurvatureInfo - Trait Implementations
//!
//! This module contains trait implementations for `CurvatureInfo`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::CurvatureInfo;

impl<T: Float + Debug + Send + Sync + 'static> Default for CurvatureInfo<T> {
    fn default() -> Self {
        Self {
            mean_curvature: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
            gaussian_curvature: scirs2_core::numeric::NumCast::from(0.05)
                .unwrap_or_else(|| T::zero()),
            principal_curvatures: vec![
                scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
                scirs2_core::numeric::NumCast::from(-0.05).unwrap_or_else(|| T::zero()),
            ],
            condition_number: scirs2_core::numeric::NumCast::from(10.0)
                .unwrap_or_else(|| T::zero()),
        }
    }
}
