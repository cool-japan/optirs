//! # `StaticAnalysisResult` - Trait Implementations
//!
//! This module contains trait implementations for `StaticAnalysisResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::StaticAnalysisResult;

impl Default for StaticAnalysisResult {
    fn default() -> Self {
        Self {
            security_issues: Vec::new(),
            quality_issues: Vec::new(),
            files_scanned: 0,
            lines_analyzed: 0,
            analysis_duration: Duration::from_secs(0),
        }
    }
}
