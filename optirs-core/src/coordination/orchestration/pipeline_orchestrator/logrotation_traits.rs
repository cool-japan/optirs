//! # LogRotation - Trait Implementations
//!
//! This module contains trait implementations for `LogRotation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{LogRotation, RotationStrategy};

impl Default for LogRotation {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: RotationStrategy::SizeBased,
            max_size: 100 * 1024 * 1024,
            max_age: Duration::from_secs(86400 * 7),
            max_files: 10,
        }
    }
}
