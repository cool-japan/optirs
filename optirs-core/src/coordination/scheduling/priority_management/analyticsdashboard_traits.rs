//! # `AnalyticsDashboard` - Trait Implementations
//!
//! This module contains trait implementations for `AnalyticsDashboard`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::AnalyticsDashboard;

impl<T: Float + Debug + Send + Sync + 'static> Default for AnalyticsDashboard<T> {
    fn default() -> Self {
        Self::new()
    }
}
