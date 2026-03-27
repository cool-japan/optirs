//! # TensorCoreConfig - Trait Implementations
//!
//! This module contains trait implementations for `TensorCoreConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TensorCoreConfig;

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            use_volta_cores: true,
            use_turing_cores: true,
            use_ampere_cores: true,
            use_hopper_cores: false,
            wmma_tile_m: 16,
            wmma_tile_n: 16,
            wmma_tile_k: 16,
            auto_layout_optimization: true,
            use_tf32: true,
            sparsity_ratio: 0.0,
            async_execution: true,
        }
    }
}
