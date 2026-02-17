//! # ComplianceRule - Trait Implementations
//!
//! This module contains trait implementations for `ComplianceRule`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use crate::error::{OptimError, Result};
use std::fmt::Debug;

use super::types::ComplianceRule;

impl std::fmt::Debug for ComplianceRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplianceRule")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("evaluation_fn", &"<function>")
            .field("severity", &self.severity)
            .field("frameworks", &self.frameworks)
            .finish()
    }
}
