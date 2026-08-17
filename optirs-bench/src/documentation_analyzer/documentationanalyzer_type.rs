//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AnalysisResults, AnalyzerConfig, DocumentationMetrics};

/// Documentation analyzer for API completeness and quality
#[derive(Debug)]
#[allow(dead_code)]
pub struct DocumentationAnalyzer {
    /// Configuration for documentation analysis
    pub(super) config: AnalyzerConfig,
    /// Analysis results
    pub(super) analysis_results: AnalysisResults,
    /// Documentation metrics
    pub(super) metrics: DocumentationMetrics,
}
