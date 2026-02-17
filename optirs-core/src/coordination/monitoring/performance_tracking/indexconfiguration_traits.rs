//! # IndexConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `IndexConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{IndexConfiguration, IndexType};

impl Default for IndexConfiguration {
    fn default() -> Self {
        Self {
            indexed_fields: vec!["timestamp".to_string(), "metric_name".to_string()],
            index_type: IndexType::default(),
            refresh_interval: Duration::from_secs(1),
        }
    }
}
