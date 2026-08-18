//! # `SecurityPolicy` - Trait Implementations
//!
//! This module contains trait implementations for `SecurityPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{Permission, SecurityPolicy, SignatureVerificationConfig};
use super::types_7::SandboxConfig;

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allow_unsigned: true,
            required_permissions: Vec::new(),
            forbidden_permissions: vec![Permission::ProcessExecution, Permission::SystemInfo],
            max_plugin_size: 100 * 1024 * 1024, // 100MB
            enable_code_scanning: false,
            sandbox_config: SandboxConfig::default(),
            signature_verification: SignatureVerificationConfig::default(),
            _trustedcas: Vec::new(),
            plugin_allowlist: Vec::new(),
            integrity_monitoring: false,
        }
    }
}
