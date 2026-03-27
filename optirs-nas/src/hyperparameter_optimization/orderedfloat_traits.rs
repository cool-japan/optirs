//! # OrderedFloat - Trait Implementations
//!
//! This module contains trait implementations for `OrderedFloat`.
//!
//! ## Implemented Traits
//!
//! - `Eq`
//! - `Ord`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::OrderedFloat;

impl<T: Float + Debug + Send + Sync + 'static> Eq for OrderedFloat<T> {}

impl<T: Float + Debug + Send + Sync + 'static> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

