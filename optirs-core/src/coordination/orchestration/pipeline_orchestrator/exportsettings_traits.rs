//! # ExportSettings - Trait Implementations
//!
//! This module contains trait implementations for `ExportSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{ExportDestination, ExportFormat, ExportSettings};

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            destinations: vec![ExportDestination::InMemory],
            format: ExportFormat::JSON,
            frequency: Duration::from_secs(60),
            filters: Vec::new(),
        }
    }
}
