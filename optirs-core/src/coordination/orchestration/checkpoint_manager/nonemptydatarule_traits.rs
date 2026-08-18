//! # `NonEmptyDataRule` - Trait Implementations
//!
//! This module contains trait implementations for `NonEmptyDataRule`.
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
use super::types::{Checkpoint, NonEmptyDataRule, ValidationWarning};
use super::types_15::ValidationResult;

impl<T: Float + Debug + Send + Sync + 'static> ValidationRule<T> for NonEmptyDataRule {
    fn validate(&self, checkpoint: &Checkpoint<T>) -> Result<ValidationResult> {
        let data = &checkpoint.data;
        let has_content = data.model_state.is_some()
            || data.optimizer_state.is_some()
            || data.training_state.is_some()
            || data.data_loader_state.is_some()
            || data.rng_state.is_some()
            || data.environment_state.is_some()
            || !data.custom_state.is_empty()
            || !data.attachments.is_empty();

        let warnings = if has_content {
            Vec::new()
        } else {
            vec![ValidationWarning {
                code: "EMPTY_CHECKPOINT_DATA".to_string(),
                message: "checkpoint carries no model/optimizer/training/custom state".to_string(),
                context: HashMap::new(),
            }]
        };

        Ok(ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "non_empty_data"
    }

    fn description(&self) -> &str {
        "warns when a checkpoint carries no state of any kind"
    }
}
