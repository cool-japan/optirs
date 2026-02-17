//! # AlertStatistics - Trait Implementations
//!
//! This module contains trait implementations for `AlertStatistics`.
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

use super::types::AlertStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for AlertStatistics<T> {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            alerts_by_severity: HashMap::new(),
            alert_rate: T::zero(),
            average_resolution_time: Duration::from_secs(0),
            false_positive_rate: T::zero(),
            true_positive_rate: T::zero(),
        }
    }
}
