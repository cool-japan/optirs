//! # `PriorityQueue` - Trait Implementations
//!
//! This module contains trait implementations for `PriorityQueue`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::PriorityQueue;

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
