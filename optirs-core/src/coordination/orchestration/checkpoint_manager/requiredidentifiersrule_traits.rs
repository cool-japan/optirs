//! # `RequiredIdentifiersRule` - Trait Implementations
//!
//! This module contains trait implementations for `RequiredIdentifiersRule`.
//!
//! ## Implemented Traits
//!
//! - `ValidationRule`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::ValidationRule;
use super::types::{Checkpoint, ErrorSeverity, RequiredIdentifiersRule};
use super::types_15::{ValidationError, ValidationResult};

impl<T: Float + Debug + Send + Sync + 'static> ValidationRule<T> for RequiredIdentifiersRule {
    fn validate(&self, checkpoint: &Checkpoint<T>) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        if checkpoint.checkpoint_id.trim().is_empty() {
            errors.push(ValidationError {
                code: "EMPTY_CHECKPOINT_ID".to_string(),
                message: "checkpoint_id must not be empty".to_string(),
                severity: ErrorSeverity::Critical,
                context: HashMap::new(),
            });
        }
        if checkpoint.workflow_id.trim().is_empty() {
            errors.push(ValidationError {
                code: "EMPTY_WORKFLOW_ID".to_string(),
                message: "workflow_id must not be empty".to_string(),
                severity: ErrorSeverity::Critical,
                context: HashMap::new(),
            });
        }
        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "required_identifiers"
    }

    fn description(&self) -> &str {
        "checkpoint_id and workflow_id must be non-empty"
    }
}
