//! # AlertAggregation - Trait Implementations
//!
//! This module contains trait implementations for `AlertAggregation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{AggregationStrategy, AlertAggregation, DeduplicationSettings};

impl Default for AlertAggregation {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(300),
            strategy: AggregationStrategy::default(),
            deduplication: DeduplicationSettings::default(),
        }
    }
}
