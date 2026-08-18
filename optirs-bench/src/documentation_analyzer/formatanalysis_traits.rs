//! # `FormatAnalysis` - Trait Implementations
//!
//! This module contains trait implementations for `FormatAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::FormatAnalysis;

impl Default for FormatAnalysis {
    fn default() -> Self {
        Self {
            markdown_compliance: 1.0,
            rustdoc_compliance: 1.0,
            cross_reference_completeness: 1.0,
            toc_quality: 1.0,
        }
    }
}
