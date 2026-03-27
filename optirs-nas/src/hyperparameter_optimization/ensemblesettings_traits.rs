//! # EnsembleSettings - Trait Implementations
//!
//! This module contains trait implementations for `EnsembleSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::types::{EnsembleAdaptationStrategy, EnsembleCombinationMethod, EnsembleSettings};

impl<T: Float + Debug + Send + Sync + 'static> Default for EnsembleSettings<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            strategies: vec![],
            strategy_weights: vec![],
            combination_method: EnsembleCombinationMethod::WeightedAverage,
            adaptation_strategy: EnsembleAdaptationStrategy::PerformanceBased,
            tracking_window: 100,
        }
    }
}

