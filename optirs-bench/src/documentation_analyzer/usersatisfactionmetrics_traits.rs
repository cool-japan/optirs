//! # `UserSatisfactionMetrics` - Trait Implementations
//!
//! This module contains trait implementations for `UserSatisfactionMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::UserSatisfactionMetrics;

impl Default for UserSatisfactionMetrics {
    fn default() -> Self {
        Self {
            clarity_score: 0.0,
            completeness_score: 0.0,
            helpfulness_score: 0.0,
            overall_satisfaction: 0.0,
        }
    }
}
