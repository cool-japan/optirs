//! # `CoverageAnalysis` - Trait Implementations
//!
//! This module contains trait implementations for `CoverageAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::CoverageAnalysis;

impl Default for CoverageAnalysis {
    fn default() -> Self {
        Self {
            total_public_items: 0,
            documented_items: 0,
            coverage_percentage: 0.0,
            undocumented_by_category: HashMap::new(),
            quality_by_module: HashMap::new(),
        }
    }
}
