//! # `TrendInfo` - Trait Implementations
//!
//! This module contains trait implementations for `TrendInfo`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TrendInfo;

impl Default for TrendInfo {
    fn default() -> Self {
        Self {
            direction: 0.0,
            strength: 0.0,
            acceleration: 0.0,
            stability: 0.0,
            change_points: Vec::new(),
        }
    }
}
