//! # TrackerStatistics - Trait Implementations
//!
//! This module contains trait implementations for `TrackerStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::TrackerStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for TrackerStatistics<T> {
    fn default() -> Self {
        Self {
            total_metrics_collected: 0,
            collection_rate: T::zero(),
            total_alerts_generated: 0,
            alert_rate: T::zero(),
            average_processing_latency: Duration::from_secs(0),
            system_utilization: T::zero(),
        }
    }
}
