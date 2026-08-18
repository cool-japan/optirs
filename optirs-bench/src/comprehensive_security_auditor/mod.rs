// Comprehensive Security Audit Engine
//
// This module provides advanced security auditing capabilities including dependency
// scanning, vulnerability detection, supply chain security analysis, and automated
// security monitoring for the optimization library and its plugins.
//
// Split into submodules by splitrs; see individual files for details.

pub mod comprehensivesecurityauditor_impl;
pub mod comprehensivesecurityauditor_queries;
pub mod comprehensivesecurityauditor_type;
pub mod constants;
pub mod dependencyscanconfig_traits;
pub mod dependencyscanresult_traits;
pub mod functions;
pub mod riskassessment_traits;
pub mod securityauditconfig_traits;
pub mod staticanalysisresult_traits;
pub mod types;
pub mod vulnerabilitydatabaseconfig_traits;

// Re-export all types
pub use comprehensivesecurityauditor_type::*;
pub use types::*;
