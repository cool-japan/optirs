//! # CompressionSettings - Trait Implementations
//!
//! This module contains trait implementations for `CompressionSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::types::{CompressionAlgorithm, CompressionSettings};

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::None,
            level: 6,
            options: HashMap::new(),
        }
    }
}
