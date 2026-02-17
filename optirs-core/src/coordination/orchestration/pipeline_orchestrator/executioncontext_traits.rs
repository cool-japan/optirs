//! # ExecutionContext - Trait Implementations
//!
//! This module contains trait implementations for `ExecutionContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::ExecutionContext;

impl<T: Float + Debug + Send + Sync + 'static> Default for ExecutionContext<T> {
    fn default() -> Self {
        Self::new()
    }
}
