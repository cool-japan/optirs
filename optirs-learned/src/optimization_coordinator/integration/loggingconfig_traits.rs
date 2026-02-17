//! # LoggingConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoggingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::*;
use super::state::*;

use super::types::{LogLevel, LoggingConfig};

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            log_requests: true,
            log_responses: false,
            log_errors: true,
            log_file: None,
            max_file_size: 10 * 1024 * 1024,
            rotation_enabled: true,
        }
    }
}

