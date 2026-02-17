//! # CategoryTrends - Trait Implementations
//!
//! This module contains trait implementations for `CategoryTrends`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{CategoryTrends, TrendDirection};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for CategoryTrends<T> {
    fn default() -> Self {
        Self {
            short_term: TrendDirection::Unknown,
            long_term: TrendDirection::Unknown,
            trend_strength: T::zero(),
            trend_confidence: T::zero(),
            predictions: Vec::new(),
        }
    }
}
