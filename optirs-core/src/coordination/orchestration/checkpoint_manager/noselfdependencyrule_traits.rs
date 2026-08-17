//! # `NoSelfDependencyRule` - Trait Implementations
//!
//! This module contains trait implementations for `NoSelfDependencyRule`.
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
use super::types::{Checkpoint, ErrorSeverity, NoSelfDependencyRule};
use super::types_15::{ValidationError, ValidationResult};

impl<T: Float + Debug + Send + Sync + 'static> ValidationRule<T> for NoSelfDependencyRule {
    fn validate(&self, checkpoint: &Checkpoint<T>) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        if checkpoint
            .dependencies
            .iter()
            .any(|dep| dep == &checkpoint.checkpoint_id)
        {
            errors.push(ValidationError {
                code: "SELF_DEPENDENCY".to_string(),
                message: format!(
                    "checkpoint '{}' lists itself as a dependency",
                    checkpoint.checkpoint_id
                ),
                severity: ErrorSeverity::Error,
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
        "no_self_dependency"
    }

    fn description(&self) -> &str {
        "a checkpoint must not declare itself as one of its own dependencies"
    }
}
