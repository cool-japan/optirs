//! # DeduplicationSettings - Trait Implementations
//!
//! This module contains trait implementations for `DeduplicationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::DeduplicationSettings;

impl Default for DeduplicationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(300),
            key_fields: vec!["id".to_string(), "source".to_string()],
        }
    }
}
