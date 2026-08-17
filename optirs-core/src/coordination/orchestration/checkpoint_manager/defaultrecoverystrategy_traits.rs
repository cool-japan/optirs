//! # `DefaultRecoveryStrategy` - Trait Implementations
//!
//! This module contains trait implementations for `DefaultRecoveryStrategy`.
//!
//! ## Implemented Traits
//!
//! - `RecoveryStrategy`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::functions::{empty_checkpoint_data, RecoveryStrategy};
use super::types::{
    Checkpoint, CheckpointData, CheckpointType, RecoveryResult, RecoveryTarget, StateType,
};
use super::types_15::{DefaultRecoveryStrategy, RecoveryCapabilities, RecoveryMetrics};

impl<T: Float + Debug + Send + Sync + 'static + Clone> RecoveryStrategy<T>
    for DefaultRecoveryStrategy
{
    fn recover(
        &self,
        checkpoint: &Checkpoint<T>,
        target_state: &RecoveryTarget,
    ) -> Result<RecoveryResult<T>> {
        let start = std::time::Instant::now();
        let data = &checkpoint.data;

        // How many state components the requested `StateType` cares
        // about, and how many of those are actually present in `data`.
        let (wanted, present) = match target_state.state_type {
            StateType::Full => {
                let present_fields = [
                    data.model_state.is_some(),
                    data.optimizer_state.is_some(),
                    data.training_state.is_some(),
                    data.data_loader_state.is_some(),
                    data.rng_state.is_some(),
                    data.environment_state.is_some(),
                ];
                (
                    present_fields.len(),
                    present_fields.iter().filter(|present| **present).count(),
                )
            }
            StateType::ModelOnly => (1, usize::from(data.model_state.is_some())),
            StateType::OptimizerOnly => (1, usize::from(data.optimizer_state.is_some())),
            StateType::TrainingOnly => (1, usize::from(data.training_state.is_some())),
            StateType::Custom => (1, usize::from(!data.custom_state.is_empty())),
        };

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if present == 0 {
            errors.push(format!(
                "checkpoint '{}' has no data for requested state type {:?}",
                checkpoint.checkpoint_id, target_state.state_type
            ));
        } else if present < wanted && !target_state.options.allow_partial {
            errors.push(format!(
                "checkpoint '{}' only has {present}/{wanted} components available for state \
                 type {:?}; set RecoveryOptions::allow_partial to accept a partial recovery",
                checkpoint.checkpoint_id, target_state.state_type
            ));
        } else if present < wanted {
            warnings.push(format!(
                "partial recovery: {present}/{wanted} components present for state type {:?}",
                target_state.state_type
            ));
        }

        let success = errors.is_empty();

        let recovered_state = if !success {
            None
        } else {
            Some(match target_state.state_type {
                StateType::Full => data.clone(),
                StateType::ModelOnly => CheckpointData {
                    model_state: data.model_state.clone(),
                    ..empty_checkpoint_data()
                },
                StateType::OptimizerOnly => CheckpointData {
                    optimizer_state: data.optimizer_state.clone(),
                    ..empty_checkpoint_data()
                },
                StateType::TrainingOnly => CheckpointData {
                    training_state: data.training_state.clone(),
                    ..empty_checkpoint_data()
                },
                StateType::Custom => CheckpointData {
                    custom_state: data.custom_state.clone(),
                    ..empty_checkpoint_data()
                },
            })
        };

        let completeness_score = if wanted > 0 {
            T::from(present as f64 / wanted as f64).unwrap_or_else(T::zero)
        } else {
            T::one()
        };

        Ok(RecoveryResult {
            success,
            recovered_state,
            metrics: RecoveryMetrics {
                recovery_time: start.elapsed(),
                integrity_score: if success { T::one() } else { T::zero() },
                completeness_score,
                efficiency: T::one(),
            },
            errors,
            warnings,
        })
    }

    fn name(&self) -> &str {
        "default"
    }

    fn capabilities(&self) -> RecoveryCapabilities {
        RecoveryCapabilities {
            supported_types: vec![
                CheckpointType::Full,
                CheckpointType::Incremental,
                CheckpointType::Differential,
                CheckpointType::ModelParameters,
                CheckpointType::OptimizerState,
                CheckpointType::DataState,
                CheckpointType::Configuration,
                CheckpointType::Emergency,
            ],
            partial_recovery: true,
            cross_platform: true,
            version_compatibility: vec!["1.0".to_string()],
        }
    }
}
