//! # IntegrationMetrics - Trait Implementations
//!
//! This module contains trait implementations for `IntegrationMetrics`.
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

use super::types::IntegrationMetrics;

impl<T: Float + Debug + Send + Sync + 'static> Default for IntegrationMetrics<T> {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time: Duration::from_secs(0),
            request_rate: T::zero(),
            error_rate: T::zero(),
            uptime: Duration::from_secs(0),
            connector_metrics: HashMap::new(),
        }
    }
}

