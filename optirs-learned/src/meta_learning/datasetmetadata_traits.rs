//! # DatasetMetadata - Trait Implementations
//!
//! This module contains trait implementations for `DatasetMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DatasetMetadata;

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            num_samples: 0,
            feature_dim: 0,
            distribution_type: "unknown".to_string(),
            noise_level: 0.0,
        }
    }
}
