// API Documentation Analysis and Verification
//
// This module provides tools for analyzing API documentation completeness,
// verifying examples, and ensuring documentation quality standards.
//
// Split into submodules by splitrs; see individual files for details.

pub mod accessibilityanalysis_traits;
pub mod analysisresults_traits;
pub mod analyzerconfig_traits;
pub mod coverageanalysis_traits;
pub mod documentationanalyzer_parsing;
pub mod documentationanalyzer_queries;
pub mod documentationanalyzer_type;
pub mod documentationdebt_traits;
pub mod documentationmetrics_traits;
pub mod examplequalitymetrics_traits;
pub mod formatanalysis_traits;
pub mod functions;
pub mod linkcheckingresults_traits;
pub mod styleanalysis_traits;
pub mod types;
pub mod usersatisfactionmetrics_traits;

// Re-export all types
pub use documentationanalyzer_type::*;
pub use types::*;
