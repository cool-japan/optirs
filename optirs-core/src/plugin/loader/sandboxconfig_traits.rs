//! # `SandboxConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SandboxConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;

use super::types_7::SandboxConfig;

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            process_isolation: false,
            memory_limit: 512 * 1024 * 1024, // 512MB
            cpu_time_limit: 60.0,            // 60 seconds
            network_access: false,
            filesystem_access: vec![PathBuf::from("./tmp")],
        }
    }
}
