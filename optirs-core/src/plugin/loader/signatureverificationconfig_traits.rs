//! # `SignatureVerificationConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SignatureVerificationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SignatureVerificationConfig;
use super::types_7::SignatureAlgorithm;

impl Default for SignatureVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            required_algorithm: SignatureAlgorithm::Rsa2048Sha256,
            min_key_size: 2048,
            allow_self_signed: false,
            max_chain_depth: 5,
            check_revocation: false,
            validation_timeout: std::time::Duration::from_secs(30),
        }
    }
}
