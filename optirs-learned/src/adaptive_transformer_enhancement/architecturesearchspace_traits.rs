//! # ArchitectureSearchSpace - Trait Implementations
//!
//! This module contains trait implementations for `ArchitectureSearchSpace`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ActivationType, ArchitectureSearchSpace};

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            layer_count_range: (2, 12),
            hidden_size_options: vec![128, 256, 512, 768, 1024],
            attention_head_options: vec![4, 8, 12, 16],
            ff_dim_options: vec![512, 1024, 2048, 4096],
            activation_options: vec![
                ActivationType::ReLU, ActivationType::GELU, ActivationType::Swish,
            ],
        }
    }
}

