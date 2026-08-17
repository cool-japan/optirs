//! # `SecurityScanResult` - Trait Implementations
//!
//! This module contains trait implementations for `SecurityScanResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SecurityScanResult;

impl Default for SecurityScanResult {
    fn default() -> Self {
        Self {
            scan_successful: false,
            threats: Vec::new(),
            permission_violations: Vec::new(),
            security_score: 0.0,
            signature_verification: None,
            plugin_hash: String::new(),
            integrity_valid: false,
        }
    }
}
