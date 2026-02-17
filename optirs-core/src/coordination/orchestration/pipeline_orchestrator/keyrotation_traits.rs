//! # KeyRotation - Trait Implementations
//!
//! This module contains trait implementations for `KeyRotation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{KeyRotation, RotationStrategy};

impl Default for KeyRotation {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(86400 * 30),
            strategy: RotationStrategy::TimeBased,
        }
    }
}
