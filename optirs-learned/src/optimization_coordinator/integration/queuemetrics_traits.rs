//! # QueueMetrics - Trait Implementations
//!
//! This module contains trait implementations for `QueueMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::*;
use super::state::*;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::QueueMetrics;

impl<T: Float + Debug + Send + Sync + 'static> Default for QueueMetrics<T> {
    fn default() -> Self {
        Self {
            messages_per_second: T::zero(),
            queue_depth: HashMap::new(),
            average_message_size: T::zero(),
            processing_latency: Duration::from_secs(0),
            error_rate: T::zero(),
        }
    }
}

