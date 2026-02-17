//! # EncryptionSettings - Trait Implementations
//!
//! This module contains trait implementations for `EncryptionSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EncryptionAlgorithm, EncryptionScope, EncryptionSettings, KeyManagement};

impl Default for EncryptionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: EncryptionAlgorithm::AES256,
            key_management: KeyManagement::default(),
            scope: EncryptionScope::SensitiveOnly,
        }
    }
}
