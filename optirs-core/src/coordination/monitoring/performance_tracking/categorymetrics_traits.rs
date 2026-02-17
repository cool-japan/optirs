//! # CategoryMetrics - Trait Implementations
//!
//! This module contains trait implementations for `CategoryMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::{CategoryMetrics, CategoryStatus, CategoryTrends};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for CategoryMetrics<T> {
    fn default() -> Self {
        Self {
            metrics: HashMap::new(),
            weight: T::one(),
            status: CategoryStatus::Normal,
            trends: CategoryTrends::default(),
        }
    }
}
