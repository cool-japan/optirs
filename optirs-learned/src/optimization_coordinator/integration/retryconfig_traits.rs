//! # RetryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RetryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::*;
use super::state::*;
use std::time::{Duration, SystemTime};

use super::types::{RetryConfig, RetryStrategy};

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            strategy: RetryStrategy::ExponentialBackoff,
            retriable_errors: vec![500, 502, 503, 504],
        }
    }
}

