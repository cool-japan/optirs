//! # MetricsMetadata - Trait Implementations
//!
//! This module contains trait implementations for `MetricsMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::MetricsMetadata;

impl Default for MetricsMetadata {
    fn default() -> Self {
        Self {
            source: "default".to_string(),
            collection_method: "automatic".to_string(),
            sampling_rate: 1.0,
            quality_score: 1.0,
            custom: HashMap::new(),
        }
    }
}
