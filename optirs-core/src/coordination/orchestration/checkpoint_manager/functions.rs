//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::{
    Checkpoint, CheckpointData, CheckpointManager, CheckpointType, CompressionAlgorithmStats,
    CompressorConfig, CreatorInfo, InMemoryCheckpointStorage, IndexerConfig, RecoveryConfig,
    RecoveryManager, RecoveryResult, RecoveryTarget, StateType, StorageConfig, StorageStatistics,
};
use super::types_15::{
    AccessPermissions, CheckpointConfiguration, CheckpointMetadata, CheckpointValidator,
    CompressionInfo, RecoveryCapabilities, RecoveryOptions, RngState, SchedulerConfig,
    ValidationResult, ValidationStatus, ValidatorConfig,
};

/// Checkpoint storage trait
pub trait CheckpointStorage<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Store a checkpoint
    fn store(&mut self, checkpoint: &Checkpoint<T>) -> Result<String>;

    /// Retrieve a checkpoint
    fn retrieve(&self, checkpoint_id: &str) -> Result<Checkpoint<T>>;

    /// Delete a checkpoint
    fn delete(&mut self, checkpoint_id: &str) -> Result<()>;

    /// List available checkpoints
    fn list(&self, workflow_id: Option<&str>) -> Result<Vec<String>>;

    /// Check if checkpoint exists
    fn exists(&self, checkpoint_id: &str) -> Result<bool>;

    /// Get checkpoint metadata
    fn get_metadata(&self, checkpoint_id: &str) -> Result<CheckpointMetadata<T>>;

    /// Get storage statistics
    fn get_statistics(&self) -> Result<StorageStatistics>;
}

/// Validation rule trait
pub trait ValidationRule<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Validate a checkpoint
    fn validate(&self, checkpoint: &Checkpoint<T>) -> Result<ValidationResult>;

    /// Get rule name
    fn name(&self) -> &str;

    /// Get rule description
    fn description(&self) -> &str;
}

/// Compression algorithm implementation trait
pub trait CompressionAlgorithmImpl<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Compress data
    fn compress(&self, data: &[u8], level: u8) -> Result<Vec<u8>>;

    /// Decompress data
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get compression statistics
    fn get_statistics(&self) -> CompressionAlgorithmStats;
}

/// Recovery strategy trait
pub trait RecoveryStrategy<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Recover from checkpoint
    fn recover(
        &self,
        checkpoint: &Checkpoint<T>,
        target_state: &RecoveryTarget,
    ) -> Result<RecoveryResult<T>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy capabilities
    fn capabilities(&self) -> RecoveryCapabilities;
}

/// `CheckpointData<T>` with every field empty/`None` -- `CheckpointData`
/// does not derive `Default` (it would force a spurious `T: Default`
/// bound onto every caller), so [`DefaultRecoveryStrategy`] builds a
/// selective copy from this base instead of a struct-update off an actual
/// default instance.
pub(super) fn empty_checkpoint_data<T: Float + Debug + Send + Sync + 'static>() -> CheckpointData<T>
{
    CheckpointData {
        model_state: None,
        optimizer_state: None,
        training_state: None,
        data_loader_state: None,
        rng_state: None,
        environment_state: None,
        custom_state: HashMap::new(),
        attachments: HashMap::new(),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    fn empty_checkpoint(id: &str, workflow_id: &str) -> Checkpoint<f64> {
        Checkpoint {
            checkpoint_id: id.to_string(),
            workflow_id: workflow_id.to_string(),
            checkpoint_type: CheckpointType::Full,
            data: empty_checkpoint_data(),
            metadata: CheckpointMetadata {
                description: "test checkpoint".to_string(),
                tags: Vec::new(),
                version: "1.0".to_string(),
                creator: CreatorInfo {
                    name: "test".to_string(),
                    email: None,
                    tool: "test".to_string(),
                    tool_version: "1.0".to_string(),
                },
                metrics: HashMap::new(),
                validation_status: ValidationStatus {
                    valid: false,
                    errors: Vec::new(),
                    warnings: Vec::new(),
                    validated_at: SystemTime::now(),
                    validator_version: "1.0".to_string(),
                },
                storage_location: String::new(),
                backup_locations: Vec::new(),
                permissions: AccessPermissions::default(),
            },
            created_at: SystemTime::now(),
            size_bytes: 128,
            hash: "test-hash".to_string(),
            compression: CompressionInfo::default(),
            dependencies: Vec::new(),
        }
    }

    fn validator_config() -> ValidatorConfig<f64> {
        ValidatorConfig {
            strict_validation: false,
            validation_timeout: Duration::from_secs(5),
            custom_params: HashMap::new(),
        }
    }

    fn recovery_config() -> RecoveryConfig<f64> {
        RecoveryConfig {
            default_timeout: Duration::from_secs(30),
            max_attempts: 3,
            retry_delay: Duration::from_secs(1),
            custom_params: HashMap::new(),
        }
    }

    fn recovery_target(state_type: StateType, allow_partial: bool) -> RecoveryTarget {
        RecoveryTarget {
            workflow_id: "wf".to_string(),
            state_type,
            options: RecoveryOptions {
                allow_partial,
                skip_validation: false,
                force_recovery: false,
                timeout: None,
                custom_options: HashMap::new(),
            },
            environment: None,
        }
    }

    // Regression tests for F9's validator half: `CheckpointValidator::validate`
    // used to unconditionally return `valid: true` regardless of input.

    #[test]
    fn validate_accepts_a_well_formed_checkpoint() {
        let validator = CheckpointValidator::<f64>::new(validator_config()).expect("new");
        let mut checkpoint = empty_checkpoint("ckpt-1", "wf-1");
        checkpoint
            .data
            .custom_state
            .insert("k".to_string(), vec![1, 2, 3]);

        let result = validator.validate(&checkpoint).expect("validate");
        assert!(result.valid);
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn validate_rejects_empty_checkpoint_id() {
        let validator = CheckpointValidator::<f64>::new(validator_config()).expect("new");
        let checkpoint = empty_checkpoint("", "wf-1");

        let result = validator.validate(&checkpoint).expect("validate");
        assert!(!result.valid, "an empty checkpoint_id must fail validation");
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == "EMPTY_CHECKPOINT_ID"));
    }

    #[test]
    fn validate_rejects_self_dependency() {
        let validator = CheckpointValidator::<f64>::new(validator_config()).expect("new");
        let mut checkpoint = empty_checkpoint("ckpt-1", "wf-1");
        checkpoint.dependencies.push("ckpt-1".to_string());

        let result = validator.validate(&checkpoint).expect("validate");
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "SELF_DEPENDENCY"));
    }

    #[test]
    fn validate_warns_but_does_not_fail_on_empty_data() {
        let validator = CheckpointValidator::<f64>::new(validator_config()).expect("new");
        let checkpoint = empty_checkpoint("ckpt-1", "wf-1");

        let result = validator.validate(&checkpoint).expect("validate");
        assert!(
            result.valid,
            "an empty-but-otherwise-well-formed checkpoint should only warn, not fail"
        );
        assert!(result
            .warnings
            .iter()
            .any(|w| w.code == "EMPTY_CHECKPOINT_DATA"));
    }

    #[test]
    fn validate_strict_mode_fails_on_warnings() {
        let mut config = validator_config();
        config.strict_validation = true;
        let validator = CheckpointValidator::<f64>::new(config).expect("new");
        let checkpoint = empty_checkpoint("ckpt-1", "wf-1");

        let result = validator.validate(&checkpoint).expect("validate");
        assert!(
            !result.valid,
            "strict_validation must fail a checkpoint that only produced warnings"
        );
    }

    // Regression tests for F9's recovery half: `RecoveryManager::recover`
    // used to ignore its arguments and always return
    // `success: true, recovered_state: None`.

    #[test]
    fn recover_populates_state_for_the_requested_component() {
        let manager = RecoveryManager::<f64>::new(recovery_config()).expect("new");
        let mut checkpoint = empty_checkpoint("ckpt-1", "wf-1");
        checkpoint
            .data
            .custom_state
            .insert("weights".to_string(), vec![9, 9, 9]);
        // Also populate a field outside the requested `Custom` scope, to
        // confirm the recovered state is filtered, not the full checkpoint.
        checkpoint.data.rng_state = Some(RngState {
            rng_type: "pcg".to_string(),
            seed: 42,
            state_data: vec![0, 1, 2],
            version: "1.0".to_string(),
        });

        let target = recovery_target(StateType::Custom, false);
        let result = manager.recover(&checkpoint, &target).expect("recover");

        assert!(result.success);
        let recovered = result
            .recovered_state
            .expect("successful recovery must populate recovered_state");
        assert_eq!(recovered.custom_state.get("weights"), Some(&vec![9, 9, 9]));
        assert!(
            recovered.rng_state.is_none(),
            "recovering `Custom` state must not leak unrelated components"
        );
    }

    #[test]
    fn recover_fails_honestly_when_requested_state_is_absent() {
        let manager = RecoveryManager::<f64>::new(recovery_config()).expect("new");
        let checkpoint = empty_checkpoint("ckpt-1", "wf-1"); // model_state: None

        let target = recovery_target(StateType::ModelOnly, false);
        let result = manager.recover(&checkpoint, &target).expect("recover");

        assert!(
            !result.success,
            "recovering ModelOnly from a checkpoint with no model_state must not report success"
        );
        assert!(result.recovered_state.is_none());
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn recover_partial_full_state_requires_allow_partial() {
        let manager = RecoveryManager::<f64>::new(recovery_config()).expect("new");
        let mut checkpoint = empty_checkpoint("ckpt-1", "wf-1");
        checkpoint.data.rng_state = Some(RngState {
            rng_type: "pcg".to_string(),
            seed: 7,
            state_data: vec![],
            version: "1.0".to_string(),
        });

        // Without allow_partial: only 1 of 6 Full-state components present
        // must be treated as a failure, not silently "succeed".
        let strict_target = recovery_target(StateType::Full, false);
        let strict_result = manager
            .recover(&checkpoint, &strict_target)
            .expect("recover");
        assert!(!strict_result.success);

        // With allow_partial: succeeds, but reports the real completeness
        // fraction (1/6) rather than a fabricated 1.0.
        let lenient_target = recovery_target(StateType::Full, true);
        let lenient_result = manager
            .recover(&checkpoint, &lenient_target)
            .expect("recover");
        assert!(lenient_result.success);
        assert!(!lenient_result.warnings.is_empty());
        let completeness = lenient_result.metrics.completeness_score;
        assert!(
            (completeness - 1.0 / 6.0).abs() < 1e-9,
            "expected completeness 1/6, got {completeness}"
        );
    }

    // Regression tests for F9's storage half: `CheckpointStorage` had zero
    // implementors anywhere in the crate.

    #[test]
    fn in_memory_storage_round_trips_a_checkpoint() {
        let mut storage = InMemoryCheckpointStorage::<f64>::new();
        let mut checkpoint = empty_checkpoint("ckpt-1", "wf-1");
        checkpoint
            .data
            .custom_state
            .insert("payload".to_string(), vec![1, 2, 3, 4]);

        assert!(!storage.exists("ckpt-1").expect("exists"));
        let location = storage.store(&checkpoint).expect("store");
        assert!(location.contains("ckpt-1"));
        assert!(storage.exists("ckpt-1").expect("exists"));

        let retrieved = storage.retrieve("ckpt-1").expect("retrieve");
        assert_eq!(retrieved.checkpoint_id, "ckpt-1");
        assert_eq!(
            retrieved.data.custom_state.get("payload"),
            Some(&vec![1, 2, 3, 4])
        );

        let stats = storage.get_statistics().expect("get_statistics");
        assert_eq!(stats.total_checkpoints, 1);
        assert_eq!(stats.total_storage_bytes, checkpoint.size_bytes);

        storage.delete("ckpt-1").expect("delete");
        assert!(!storage.exists("ckpt-1").expect("exists"));
        assert!(storage.retrieve("ckpt-1").is_err());
    }

    #[test]
    fn in_memory_storage_list_filters_by_workflow() {
        let mut storage = InMemoryCheckpointStorage::<f64>::new();
        storage
            .store(&empty_checkpoint("a", "wf-1"))
            .expect("store");
        storage
            .store(&empty_checkpoint("b", "wf-2"))
            .expect("store");

        let mut wf1: Vec<String> = storage.list(Some("wf-1")).expect("list");
        wf1.sort();
        assert_eq!(wf1, vec!["a".to_string()]);

        let mut all: Vec<String> = storage.list(None).expect("list");
        all.sort();
        assert_eq!(all, vec!["a".to_string(), "b".to_string()]);
    }

    // End-to-end: `CheckpointManager` is now actually constructible and
    // usable with zero caller-supplied code, driving validator, storage,
    // and recovery manager together.
    #[test]
    fn checkpoint_manager_create_and_restore_round_trip() {
        let config = CheckpointConfiguration {
            default_checkpoint_type: CheckpointType::Full,
            storage_config: StorageConfig {
                backend_type: "memory".to_string(),
                location: String::new(),
                credentials: None,
                options: HashMap::new(),
            },
            compression_config: CompressorConfig {
                enable_compression: false,
                default_compression_level: 0,
                size_threshold: usize::MAX,
                custom_params: HashMap::new(),
            },
            validation_config: validator_config(),
            recovery_config: recovery_config(),
            scheduling_config: SchedulerConfig {
                default_interval: Duration::from_secs(60),
                max_frequency: 1.0,
                min_interval: Duration::from_secs(1),
                adaptive_params: HashMap::new(),
            },
            indexing_config: IndexerConfig {
                rebuild_interval: Duration::from_secs(300),
                compaction_threshold: 0.5,
                enable_caching: true,
                custom_params: HashMap::new(),
            },
        };

        let mut manager = CheckpointManager::<f64>::new(
            config,
            Box::new(InMemoryCheckpointStorage::<f64>::new()),
        )
        .expect("CheckpointManager::new");

        let mut data = empty_checkpoint_data::<f64>();
        data.custom_state
            .insert("round_trip".to_string(), vec![7, 7, 7]);

        let checkpoint_id = manager
            .create_checkpoint("wf-1".to_string(), CheckpointType::Full, data)
            .expect("create_checkpoint");

        let target = recovery_target(StateType::Custom, false);
        let result = manager
            .restore_checkpoint(&checkpoint_id, target)
            .expect("restore_checkpoint");

        assert!(result.success);
        let recovered = result.recovered_state.expect("recovered_state");
        assert_eq!(
            recovered.custom_state.get("round_trip"),
            Some(&vec![7, 7, 7])
        );
    }
}
