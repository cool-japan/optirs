//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::functions::ValidationRule;
use super::types::{
    AttachmentType, BatchNormSettings, Checkpoint, CheckpointType, CompressionAlgorithm,
    CompressorConfig, CreatorInfo, ErrorSeverity, HardwareConfig, IndexMetadata, IndexerConfig,
    NoSelfDependencyRule, NonEmptyDataRule, PermissionSet, RecoveryConfig, RequiredIdentifiersRule,
    ScheduledCheckpoint, StorageConfig, SystemInfo, ValidationStatistics, ValidationWarning,
};

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Base learning rate
    pub base_lr: T,
    /// Momentum parameters
    pub momentum: HashMap<String, T>,
    /// Adaptive parameters
    pub adaptive_params: HashMap<String, T>,
    /// Regularization parameters
    pub regularization_params: HashMap<String, T>,
}
/// Metrics history
#[derive(Debug, Clone)]
pub struct MetricsHistory<T: Float + Debug + Send + Sync + 'static> {
    /// Metric values over time
    pub metrics: HashMap<String, Vec<T>>,
    /// Timestamps
    pub timestamps: Vec<SystemTime>,
    /// Epoch numbers
    pub epochs: Vec<usize>,
    /// Step numbers
    pub steps: Vec<usize>,
}
/// Environment state information
#[derive(Debug, Clone)]
pub struct EnvironmentState {
    /// Environment variables
    pub environment_vars: HashMap<String, String>,
    /// System information
    pub system_info: SystemInfo,
    /// Software versions
    pub software_versions: HashMap<String, String>,
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
}
/// Regularization settings
#[derive(Debug, Clone)]
pub struct RegularizationSettings<T: Float + Debug + Send + Sync + 'static> {
    /// L1 regularization strength
    pub l1_strength: T,
    /// L2 regularization strength
    pub l2_strength: T,
    /// Dropout rate
    pub dropout_rate: T,
    /// Batch normalization settings
    pub batch_norm: BatchNormSettings<T>,
    /// Custom regularization
    pub custom_regularization: HashMap<String, T>,
}
/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Default checkpoint interval
    pub default_interval: Duration,
    /// Maximum checkpoint frequency
    pub max_frequency: T,
    /// Minimum checkpoint interval
    pub min_interval: Duration,
    /// Adaptive scheduling parameters
    pub adaptive_params: HashMap<String, T>,
}
/// Compression information
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
    /// Original size (bytes)
    pub original_size: usize,
    /// Compressed size (bytes)
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Compression time
    pub compression_time: Duration,
}
/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total checkpoints scheduled
    pub total_scheduled: usize,
    /// Total checkpoints completed
    pub total_completed: usize,
    /// Total checkpoints failed
    pub total_failed: usize,
    /// Average scheduling latency
    pub average_latency: Duration,
    /// Scheduling efficiency
    pub efficiency: T,
}
/// Warmup methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupMethod {
    Linear,
    Exponential,
    Constant,
}
/// Shuffle state information
#[derive(Debug, Clone)]
pub struct ShuffleState {
    /// Random seed used
    pub seed: u64,
    /// Shuffle indices
    pub indices: Vec<usize>,
    /// Current position
    pub position: usize,
}
/// Checkpoint scheduler
#[derive(Debug)]
pub struct CheckpointScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Scheduled checkpoints
    pub(super) scheduled_checkpoints: VecDeque<ScheduledCheckpoint>,
    /// Scheduling strategy
    pub(super) strategy: SchedulingStrategy,
    /// Scheduler configuration
    pub(super) config: SchedulerConfig<T>,
    /// Scheduler statistics
    pub(super) stats: SchedulerStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CheckpointScheduler<T> {
    pub fn new(config: SchedulerConfig<T>) -> Result<Self> {
        Ok(Self {
            scheduled_checkpoints: VecDeque::new(),
            strategy: SchedulingStrategy::TimeBased,
            config,
            stats: SchedulerStatistics::default(),
        })
    }
}
/// Validator configuration
#[derive(Debug, Clone)]
pub struct ValidatorConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable strict validation
    pub strict_validation: bool,
    /// Validation timeout
    pub validation_timeout: Duration,
    /// Custom validation parameters
    pub custom_params: HashMap<String, T>,
}
/// Recovery capabilities
#[derive(Debug, Clone)]
pub struct RecoveryCapabilities {
    /// Supported checkpoint types
    pub supported_types: Vec<CheckpointType>,
    /// Partial recovery support
    pub partial_recovery: bool,
    /// Cross-platform recovery
    pub cross_platform: bool,
    /// Version compatibility
    pub version_compatibility: Vec<String>,
}
/// Attachment for additional data
#[derive(Debug, Clone)]
pub struct Attachment {
    /// Attachment identifier
    pub attachment_id: String,
    /// Attachment type
    pub attachment_type: AttachmentType,
    /// Attachment data
    pub data: Vec<u8>,
    /// Content type
    pub content_type: String,
    /// Attachment metadata
    pub metadata: HashMap<String, String>,
}
/// Data loader configuration
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size
    pub batch_size: usize,
    /// Shuffle enabled
    pub shuffle: bool,
    /// Number of workers
    pub num_workers: usize,
    /// Pin memory
    pub pin_memory: bool,
    /// Drop last batch
    pub drop_last: bool,
}
/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation metadata
    pub metadata: HashMap<String, String>,
}
/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total recovery attempts
    pub total_attempts: usize,
    /// Successful recoveries
    pub successful_recoveries: usize,
    /// Failed recoveries
    pub failed_recoveries: usize,
    /// Average recovery time
    pub average_recovery_time: Duration,
    /// Recovery success rate
    pub success_rate: T,
}
/// Checkpoint index
#[derive(Debug, Clone)]
pub struct CheckpointIndex<T: Float + Debug + Send + Sync + 'static> {
    /// Index entries
    pub entries: HashMap<String, IndexEntry<T>>,
    /// Index metadata
    pub metadata: IndexMetadata,
    /// Index version
    pub version: String,
    /// Last updated
    pub last_updated: SystemTime,
}
/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Default checkpoint type
    pub default_checkpoint_type: CheckpointType,
    /// Storage backend configuration
    pub storage_config: StorageConfig,
    /// Compression configuration
    pub compression_config: CompressorConfig<T>,
    /// Validation configuration
    pub validation_config: ValidatorConfig<T>,
    /// Recovery configuration
    pub recovery_config: RecoveryConfig<T>,
    /// Scheduling configuration
    pub scheduling_config: SchedulerConfig<T>,
    /// Indexing configuration
    pub indexing_config: IndexerConfig<T>,
}
/// Principal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrincipalType {
    User,
    Group,
    Role,
    Service,
}
/// Access permissions
#[derive(Debug, Clone)]
pub struct AccessPermissions {
    /// Owner permissions
    pub owner: PermissionSet,
    /// Group permissions
    pub group: PermissionSet,
    /// Public permissions
    pub public: PermissionSet,
    /// Access control list
    pub acl: Vec<AclEntry>,
}
/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total checkpoints created
    pub total_created: usize,
    /// Total checkpoints restored
    pub total_restored: usize,
    /// Total checkpoints deleted
    pub total_deleted: usize,
    /// Average checkpoint size
    pub average_size_bytes: usize,
    /// Average creation time
    pub average_creation_time: Duration,
    /// Average restoration time
    pub average_restoration_time: Duration,
    /// Storage utilization
    pub storage_utilization: T,
    /// Checkpoint success rate
    pub success_rate: T,
}
/// Index entry
#[derive(Debug, Clone)]
pub struct IndexEntry<T: Float + Debug + Send + Sync + 'static> {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// Workflow identifier
    pub workflow_id: String,
    /// Checkpoint type
    pub checkpoint_type: CheckpointType,
    /// Storage location
    pub storage_location: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Checkpoint size
    pub size_bytes: usize,
    /// Checkpoint hash
    pub hash: String,
    /// Index metadata
    pub metadata: HashMap<String, T>,
}
/// Checkpoint metadata
#[derive(Debug, Clone)]
pub struct CheckpointMetadata<T: Float + Debug + Send + Sync + 'static> {
    /// Checkpoint description
    pub description: String,
    /// Checkpoint tags
    pub tags: Vec<String>,
    /// Checkpoint version
    pub version: String,
    /// Creator information
    pub creator: CreatorInfo,
    /// Checkpoint metrics
    pub metrics: HashMap<String, T>,
    /// Validation status
    pub validation_status: ValidationStatus,
    /// Storage location
    pub storage_location: String,
    /// Backup locations
    pub backup_locations: Vec<String>,
    /// Access permissions
    pub permissions: AccessPermissions,
}
/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory (bytes)
    pub total_memory: usize,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// Memory type
    pub memory_type: String,
    /// Memory speed
    pub memory_speed: f64,
}
/// Recovery metrics
#[derive(Debug, Clone)]
pub struct RecoveryMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Recovery time
    pub recovery_time: Duration,
    /// Data integrity score
    pub integrity_score: T,
    /// Completeness score
    pub completeness_score: T,
    /// Recovery efficiency
    pub efficiency: T,
}
/// Validation status
#[derive(Debug, Clone)]
pub struct ValidationStatus {
    /// Validation result
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation timestamp
    pub validated_at: SystemTime,
    /// Validator version
    pub validator_version: String,
}
/// Scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Time-based scheduling
    TimeBased,
    /// Progress-based scheduling
    ProgressBased,
    /// Adaptive scheduling
    Adaptive,
    /// Custom scheduling
    Custom,
}
/// Checkpoint validator
#[derive(Debug)]
pub struct CheckpointValidator<T: Float + Debug + Send + Sync + 'static> {
    /// Validation rules
    pub(super) validation_rules: Vec<Box<dyn ValidationRule<T>>>,
    /// Validator configuration
    pub(super) config: ValidatorConfig<T>,
    /// Validation statistics
    pub(super) stats: ValidationStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CheckpointValidator<T> {
    /// Creates a validator pre-registered with the built-in rules
    /// ([`RequiredIdentifiersRule`], [`NoSelfDependencyRule`],
    /// [`NonEmptyDataRule`]).
    ///
    /// Regression fix (F9): [`Self::validate`] used to unconditionally
    /// return `valid: true` regardless of `validation_rules` (which
    /// nothing ever populated -- there was no `register_rule` method, so
    /// even a caller who read the source could not add one), so
    /// `create_checkpoint`'s `ValidationStatus::valid` was a fabricated
    /// constant.
    pub fn new(config: ValidatorConfig<T>) -> Result<Self> {
        let validation_rules: Vec<Box<dyn ValidationRule<T>>> = vec![
            Box::new(RequiredIdentifiersRule),
            Box::new(NoSelfDependencyRule),
            Box::new(NonEmptyDataRule),
        ];
        Ok(Self {
            validation_rules,
            config,
            stats: ValidationStatistics::default(),
        })
    }
    /// Register an additional validation rule, run (in registration order,
    /// after the built-ins) by every subsequent call to [`Self::validate`].
    pub fn register_rule(&mut self, rule: Box<dyn ValidationRule<T>>) {
        self.validation_rules.push(rule);
    }
    /// Runs every registered rule against `checkpoint` and aggregates the
    /// results: `valid` is `false` when any rule reports an `Error`- or
    /// `Critical`-severity error (an `Info`/`Warning` finding is surfaced
    /// but does not by itself invalidate the checkpoint), or when
    /// `config.strict_validation` is set and any rule reported so much as
    /// a warning.
    pub fn validate(&self, checkpoint: &Checkpoint<T>) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut metadata = HashMap::new();
        for rule in &self.validation_rules {
            let result = rule.validate(checkpoint)?;
            errors.extend(result.errors);
            warnings.extend(result.warnings);
            metadata.extend(result.metadata);
        }
        let has_blocking_error = errors.iter().any(|e| e.severity >= ErrorSeverity::Error);
        let valid = if self.config.strict_validation {
            !has_blocking_error && warnings.is_empty()
        } else {
            !has_blocking_error
        };
        Ok(ValidationResult {
            valid,
            errors,
            warnings,
            metadata,
        })
    }
}
/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error context
    pub context: HashMap<String, String>,
}
/// Built-in recovery strategy: reconstructs `recovered_state` directly
/// from the checkpoint's own (already in-memory, once retrieved from
/// storage) `data`, restricted to the components relevant to
/// `target_state.state_type`. No I/O of any kind -- "recovery" here means
/// selecting and cloning the requested subset of already-available state,
/// which is exactly what every caller of [`RecoveryManager::recover`]
/// needs back.
#[derive(Debug)]
pub(super) struct DefaultRecoveryStrategy;
/// Recovery options
#[derive(Debug, Clone)]
pub struct RecoveryOptions {
    /// Allow partial recovery
    pub allow_partial: bool,
    /// Skip validation
    pub skip_validation: bool,
    /// Force recovery
    pub force_recovery: bool,
    /// Recovery timeout
    pub timeout: Option<Duration>,
    /// Custom options
    pub custom_options: HashMap<String, String>,
}
/// Random number generator state
#[derive(Debug, Clone)]
pub struct RngState {
    /// RNG type
    pub rng_type: String,
    /// RNG seed
    pub seed: u64,
    /// RNG state data
    pub state_data: Vec<u8>,
    /// State version
    pub version: String,
}
/// Access control list entry
#[derive(Debug, Clone)]
pub struct AclEntry {
    /// Principal (user/group)
    pub principal: String,
    /// Principal type
    pub principal_type: PrincipalType,
    /// Permissions
    pub permissions: PermissionSet,
}
/// Indexing statistics
#[derive(Debug, Clone)]
pub struct IndexingStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total index entries
    pub total_entries: usize,
    /// Index size (bytes)
    pub index_size_bytes: usize,
    /// Average lookup time
    pub average_lookup_time: Duration,
    /// Index efficiency
    pub efficiency: T,
    /// Last rebuild time
    pub last_rebuild_time: Duration,
}
