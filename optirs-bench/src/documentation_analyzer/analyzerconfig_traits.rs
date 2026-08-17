//! # `AnalyzerConfig` - Trait Implementations
//!
//! This module contains trait implementations for `AnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;

use super::types::AnalyzerConfig;

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            source_directories: vec![PathBuf::from("src")],
            docs_output_dir: PathBuf::from("target/doc"),
            min_coverage_threshold: 0.8, // 80% coverage required
            verify_examples: true,
            check_links: true,
            check_style_consistency: true,
            required_sections: vec![
                "Examples".to_string(),
                "Arguments".to_string(),
                "Returns".to_string(),
                "Errors".to_string(),
            ],
            language_preferences: vec!["en".to_string()],
        }
    }
}
