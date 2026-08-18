//! # `ExampleQualityMetrics` - Trait Implementations
//!
//! This module contains trait implementations for `ExampleQualityMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ExampleQualityMetrics;

impl Default for ExampleQualityMetrics {
    fn default() -> Self {
        Self {
            average_length: 0.0,
            error_handling_coverage: 0.0,
            comment_coverage: 0.0,
            best_practices_score: 0.0,
        }
    }
}
