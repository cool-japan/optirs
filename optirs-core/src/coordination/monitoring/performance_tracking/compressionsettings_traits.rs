//! # CompressionSettings - Trait Implementations
//!
//! This module contains trait implementations for `CompressionSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CompressionSettings;

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "zstd".to_string(),
            level: 3,
            threshold_bytes: 1024,
        }
    }
}
