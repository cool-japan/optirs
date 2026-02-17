//! # WebhookMetrics - Trait Implementations
//!
//! This module contains trait implementations for `WebhookMetrics`.
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

use super::types::WebhookMetrics;

impl<T: Float + Debug + Send + Sync + 'static> Default for WebhookMetrics<T> {
    fn default() -> Self {
        Self {
            total_deliveries: 0,
            successful_deliveries: 0,
            failed_deliveries: 0,
            average_delivery_time: Duration::from_secs(0),
            success_rate: T::zero(),
            error_rates: HashMap::new(),
        }
    }
}

