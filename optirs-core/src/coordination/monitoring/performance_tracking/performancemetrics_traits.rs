//! # PerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{MetricsMetadata, MetricsQuality, PerformanceMetrics};

impl<T: Float + Debug + Send + Sync + 'static + Default> Default for PerformanceMetrics<T> {
    fn default() -> Self {
        Self {
            categories: HashMap::new(),
            timestamp: SystemTime::now(),
            interval: Duration::from_secs(60),
            metadata: MetricsMetadata::default(),
            quality: MetricsQuality::default(),
        }
    }
}
