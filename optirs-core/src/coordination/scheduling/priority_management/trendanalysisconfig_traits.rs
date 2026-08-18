//! # `TrendAnalysisConfig` - Trait Implementations
//!
//! This module contains trait implementations for `TrendAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::TrendAnalysisConfig;

impl<T: Float + Debug + Default + Send + Sync> Default for TrendAnalysisConfig<T> {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(3600), // 1 hour
            sensitivity: T::from(0.1).unwrap_or_else(|| T::zero()),
            prediction_horizon: Duration::from_secs(1800), // 30 minutes
            confidence_threshold: T::from(0.7).unwrap_or_else(|| T::zero()),
        }
    }
}
