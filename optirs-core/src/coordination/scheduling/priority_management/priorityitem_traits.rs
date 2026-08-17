//! # `PriorityItem` - Trait Implementations
//!
//! This module contains trait implementations for `PriorityItem`.
//!
//! ## Implemented Traits
//!
//! - `PartialEq`
//! - `Eq`
//! - `PartialOrd`
//! - `Ord`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::fmt::Debug;

use super::types::PriorityItem;

impl<T: Float + Debug + Send + Sync + 'static> PartialEq for PriorityItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority.composite_score == other.priority.composite_score
    }
}
impl<T: Float + Debug + Send + Sync + 'static> Eq for PriorityItem<T> {}
impl<T: Float + Debug + Send + Sync + 'static> PartialOrd for PriorityItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Float + Debug + Send + Sync + 'static> Ord for PriorityItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
