//! # `HistoryConfig` - Trait Implementations
//!
//! This module contains trait implementations for `HistoryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{HistoryConfig, SamplingStrategy};

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_records: 10000,
            retention_period: Duration::from_secs(86400 * 7), // 1 week
            sampling_strategy: SamplingStrategy::Uniform,
        }
    }
}
