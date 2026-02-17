//! # StreamMetrics - Trait Implementations
//!
//! This module contains trait implementations for `StreamMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::*;
use super::state::*;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::StreamMetrics;

impl<T: Float + Debug + Send + Sync + 'static> Default for StreamMetrics<T> {
    fn default() -> Self {
        Self {
            events_per_second: T::zero(),
            total_events: 0,
            error_rate: T::zero(),
            average_latency: Duration::from_secs(0),
            buffer_utilization: T::zero(),
        }
    }
}

