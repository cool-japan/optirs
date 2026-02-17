//! # AuditSettings - Trait Implementations
//!
//! This module contains trait implementations for `AuditSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::{AuditDestination, AuditSettings};

impl Default for AuditSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            events: Vec::new(),
            destination: AuditDestination::File(String::from("/tmp/audit.log")),
            retention: Duration::from_secs(86400 * 30),
        }
    }
}
