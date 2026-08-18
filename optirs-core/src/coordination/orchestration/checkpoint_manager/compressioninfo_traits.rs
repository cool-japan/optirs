//! # `CompressionInfo` - Trait Implementations
//!
//! This module contains trait implementations for `CompressionInfo`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::CompressionAlgorithm;
use super::types_15::CompressionInfo;

impl Default for CompressionInfo {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::None,
            level: 0,
            original_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
            compression_time: Duration::from_secs(0),
        }
    }
}
