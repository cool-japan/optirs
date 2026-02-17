//! # BootstrapEstimator - Trait Implementations
//!
//! This module contains trait implementations for `BootstrapEstimator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::BootstrapEstimator;

impl<T: Float + Debug + Send + Sync + 'static> Default for BootstrapEstimator<T> {
    fn default() -> Self {
        Self::new()
    }
}
