//! # `DocumentationMetrics` - Trait Implementations
//!
//! This module contains trait implementations for `DocumentationMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DocumentationMetrics, UserSatisfactionMetrics};

impl Default for DocumentationMetrics {
    fn default() -> Self {
        Self {
            total_doc_lines: 0,
            code_to_doc_ratio: 0.0,
            average_quality_score: 0.0,
            maintenance_burden: 0.0,
            user_satisfaction: UserSatisfactionMetrics::default(),
        }
    }
}
