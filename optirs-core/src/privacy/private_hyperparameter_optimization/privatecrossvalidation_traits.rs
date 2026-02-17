//! # PrivateCrossValidation - Trait Implementations
//!
//! This module contains trait implementations for `PrivateCrossValidation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::PrivateCrossValidation;

impl<T: Float + Debug + Send + Sync + 'static> Default for PrivateCrossValidation<T> {
    fn default() -> Self {
        Self::new()
    }
}
