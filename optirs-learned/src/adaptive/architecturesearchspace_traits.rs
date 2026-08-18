//! # ArchitectureSearchSpace - Trait Implementations
//!
//! This module contains trait implementations for `ArchitectureSearchSpace`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ArchitectureSearchSpace;

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            layer_count_range: (2, 12),
            attention_head_options: vec![4, 8, 12, 16],
        }
    }
}
