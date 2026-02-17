//! # SelectionMechanism - Trait Implementations
//!
//! This module contains trait implementations for `SelectionMechanism`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::SelectionMechanism;

impl<T: Float + Debug + Send + Sync + 'static> Default for SelectionMechanism<T> {
    fn default() -> Self {
        Self::new()
    }
}
