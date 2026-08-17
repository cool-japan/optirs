//! # `AnalysisResults` - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisResults`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AnalysisResults, ApiCompletenessAnalysis, CoverageAnalysis, ExampleVerificationResults,
    LinkCheckingResults, StyleAnalysis,
};

impl Default for AnalysisResults {
    fn default() -> Self {
        Self {
            coverage: CoverageAnalysis::default(),
            example_verification: ExampleVerificationResults::default(),
            link_checking: LinkCheckingResults::default(),
            style_analysis: StyleAnalysis::default(),
            api_completeness: ApiCompletenessAnalysis::default(),
            overall_quality_score: 0.0,
        }
    }
}
