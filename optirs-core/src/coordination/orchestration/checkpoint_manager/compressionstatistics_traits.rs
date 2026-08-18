//! # `CompressionStatistics` - Trait Implementations
//!
//! This module contains trait implementations for `CompressionStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::time::Duration;

use super::types::CompressionStatistics;

impl<T: Float + Debug + Default + Send + Sync> Default for CompressionStatistics<T> {
    fn default() -> Self {
        Self {
            total_bytes_compressed: 0,
            total_bytes_decompressed: 0,
            overall_compression_ratio: T::zero(),
            total_compression_time: Duration::from_secs(0),
            total_decompression_time: Duration::from_secs(0),
        }
    }
}
