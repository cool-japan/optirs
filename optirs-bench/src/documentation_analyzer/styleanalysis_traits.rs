//! # `StyleAnalysis` - Trait Implementations
//!
//! This module contains trait implementations for `StyleAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{FormatAnalysis, StyleAnalysis};

impl Default for StyleAnalysis {
    fn default() -> Self {
        Self {
            consistency_score: 1.0,
            violations: HashMap::new(),
            recommendations: Vec::new(),
            format_analysis: FormatAnalysis::default(),
        }
    }
}
