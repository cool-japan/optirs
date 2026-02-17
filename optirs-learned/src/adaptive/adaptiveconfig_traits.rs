//! # AdaptiveConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::AdaptiveConfig;

impl<T: Float + Debug + Send + Sync + 'static> Default for AdaptiveConfig<T> {
    fn default() -> Self {
        Self {
            adaptive_sequence_length: true,
            max_sequence_length: 1024,
            min_sequence_length: 64,
            attention_sparsity_threshold: scirs2_core::numeric::NumCast::from(0.1)
                .unwrap_or_else(|| T::zero()),
            memory_budget: 8192,
            dynamic_head_pruning: true,
            layer_adaptation: true,
            landscape_analysis_frequency: 100,
            prediction_horizon: 50,
            adaptation_lr: scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero()),
        }
    }
}
