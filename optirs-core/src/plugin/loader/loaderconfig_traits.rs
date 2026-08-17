//! # `LoaderConfig` - Trait Implementations
//!
//! This module contains trait implementations for `LoaderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use crate::plugin::core::*;
use crate::plugin::registry::*;
use std::path::PathBuf;

use super::types::{LoaderConfig, SecurityPolicy};

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_loading: true,
            plugin_directories: vec![PathBuf::from("./plugins")],
            max_plugins: 100,
            load_timeout: std::time::Duration::from_secs(30),
            enable_sandboxing: false,
            allowed_sources: vec![PluginSource::Local(PathBuf::from("./plugins"))],
            security_policy: SecurityPolicy::default(),
        }
    }
}
