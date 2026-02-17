//! # LoggingConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `LoggingConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LogDestination, LogFormat, LogLevel, LogRotation, LoggingConfiguration};

impl Default for LoggingConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Info,
            destinations: vec![LogDestination::Console],
            format: LogFormat::JSON,
            rotation: LogRotation::default(),
        }
    }
}
