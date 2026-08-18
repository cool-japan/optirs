//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
#[allow(dead_code)]
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::functions::{CheckpointStorage, CompressionAlgorithmImpl, RecoveryStrategy};
use super::types_15::{
    AccessPermissions, Attachment, CheckpointConfiguration, CheckpointIndex, CheckpointMetadata,
    CheckpointScheduler, CheckpointStatistics, CheckpointValidator, CompressionInfo,
    DataLoaderConfig, DefaultRecoveryStrategy, EnvironmentState, IndexEntry, IndexingStatistics,
    MemoryInfo, MetricsHistory, OptimizerConfiguration, RecoveryMetrics, RecoveryOptions,
    RecoveryStatistics, RegularizationSettings, RngState, ShuffleState, ValidationStatus,
    WarmupMethod,
};

/// Optimizer state information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizerState<T: Float + Debug + Send + Sync + 'static> {
    /// Optimizer type
    pub optimizer_type: String,
    /// Optimizer parameters
    pub parameters: HashMap<String, Array1<T>>,
    /// Optimizer buffers
    pub buffers: HashMap<String, Array1<T>>,
    /// Optimizer configuration
    pub configuration: OptimizerConfiguration<T>,
    /// Step counter
    pub step_counter: usize,
    /// Learning rate history
    pub lr_history: Vec<T>,
}
/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Storage backend type
    pub backend_type: String,
    /// Storage location
    pub location: String,
    /// Storage credentials
    pub credentials: Option<String>,
    /// Storage options
    pub options: HashMap<String, String>,
}
/// Permission set
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PermissionSet {
    /// Read permission
    pub read: bool,
    /// Write permission
    pub write: bool,
    /// Execute permission
    pub execute: bool,
    /// Delete permission
    pub delete: bool,
}
/// Gradient clipping methods
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClippingMethod {
    Norm,
    Value,
    GlobalNorm,
}
/// Monitor modes for early stopping
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorMode {
    Min,
    Max,
    Auto,
}
/// Checkpoint representation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Checkpoint<T: Float + Debug + Send + Sync + 'static> {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// Associated workflow/experiment ID
    pub workflow_id: String,
    /// Checkpoint type
    pub checkpoint_type: CheckpointType,
    /// Checkpoint data
    pub data: CheckpointData<T>,
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata<T>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Checkpoint size (bytes)
    pub size_bytes: usize,
    /// Checkpoint hash for integrity
    pub hash: String,
    /// Compression information
    pub compression: CompressionInfo,
    /// Dependencies on other checkpoints
    pub dependencies: Vec<String>,
}
/// Compression algorithms
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Bzip2,
    Lz4,
    Zstd,
    Snappy,
    Custom(u8),
}
/// Model architecture specification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelArchitecture {
    /// Architecture type
    pub architecture_type: String,
    /// Layer specifications
    pub layers: Vec<LayerSpec>,
    /// Connection specifications
    pub connections: Vec<ConnectionSpec>,
    /// Architecture parameters
    pub parameters: HashMap<String, String>,
}
/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total checkpoints stored
    pub total_checkpoints: usize,
    /// Total storage used (bytes)
    pub total_storage_bytes: usize,
    /// Average checkpoint size (bytes)
    pub average_checkpoint_size: usize,
    /// Storage utilization
    pub utilization_percentage: f64,
    /// Available storage (bytes)
    pub available_storage_bytes: usize,
}
/// Checkpoint compressor
#[derive(Debug)]
pub struct CheckpointCompressor<T: Float + Debug + Send + Sync + 'static> {
    /// Compression algorithms
    pub(super) algorithms: HashMap<CompressionAlgorithm, Box<dyn CompressionAlgorithmImpl<T>>>,
    /// Default algorithm
    pub(super) default_algorithm: CompressionAlgorithm,
    /// Compressor configuration
    pub(super) config: CompressorConfig<T>,
    /// Compression statistics
    pub(super) stats: CompressionStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CheckpointCompressor<T> {
    pub fn new(config: CompressorConfig<T>) -> Result<Self> {
        Ok(Self {
            algorithms: HashMap::new(),
            default_algorithm: CompressionAlgorithm::None,
            config,
            stats: CompressionStatistics::default(),
        })
    }
}
/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Warning context
    pub context: HashMap<String, String>,
}
/// Checkpoint indexer
#[derive(Debug)]
pub struct CheckpointIndexer<T: Float + Debug + Send + Sync + 'static> {
    /// Checkpoint index
    pub(super) index: CheckpointIndex<T>,
    /// Indexing strategy
    pub(super) strategy: IndexingStrategy,
    /// Indexer configuration
    pub(super) config: IndexerConfig<T>,
    /// Indexing statistics
    pub(super) stats: IndexingStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CheckpointIndexer<T> {
    pub fn new(config: IndexerConfig<T>) -> Result<Self> {
        Ok(Self {
            index: CheckpointIndex {
                entries: HashMap::new(),
                metadata: IndexMetadata {
                    format_version: "1.0".to_string(),
                    created_at: SystemTime::now(),
                    last_rebuild: SystemTime::now(),
                    statistics: HashMap::new(),
                },
                version: "1.0".to_string(),
                last_updated: SystemTime::now(),
            },
            strategy: IndexingStrategy::InMemory,
            config,
            stats: IndexingStatistics::default(),
        })
    }
    pub fn add_entry(&mut self, checkpoint: &Checkpoint<T>) -> Result<()> {
        let entry = IndexEntry {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            workflow_id: checkpoint.workflow_id.clone(),
            checkpoint_type: checkpoint.checkpoint_type.clone(),
            storage_location: checkpoint.metadata.storage_location.clone(),
            created_at: checkpoint.created_at,
            size_bytes: checkpoint.size_bytes,
            hash: checkpoint.hash.clone(),
            metadata: HashMap::new(),
        };
        self.index
            .entries
            .insert(checkpoint.checkpoint_id.clone(), entry);
        self.index.last_updated = SystemTime::now();
        self.stats.total_entries += 1;
        Ok(())
    }
    pub fn remove_entry(&mut self, checkpoint_id: &str) -> Result<()> {
        self.index.entries.remove(checkpoint_id);
        self.index.last_updated = SystemTime::now();
        if self.stats.total_entries > 0 {
            self.stats.total_entries -= 1;
        }
        Ok(())
    }
}
/// Recovery manager
#[derive(Debug)]
pub struct RecoveryManager<T: Float + Debug + Send + Sync + 'static> {
    /// Recovery strategies
    pub(super) recovery_strategies: HashMap<String, Box<dyn RecoveryStrategy<T>>>,
    /// Default recovery strategy
    pub(super) default_strategy: String,
    /// Recovery configuration
    pub(super) config: RecoveryConfig<T>,
    /// Recovery statistics
    pub(super) stats: RecoveryStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> RecoveryManager<T> {
    /// Creates a recovery manager pre-registered with
    /// `DefaultRecoveryStrategy` under the `"default"` name (also its
    /// `default_strategy`).
    ///
    /// Regression fix (F9): [`Self::recover`] used to ignore both of its
    /// arguments and unconditionally return `success: true` with
    /// `recovered_state: None` -- i.e. every recovery "succeeded" at
    /// restoring nothing, and `recovery_strategies` (which nothing ever
    /// populated) was dead weight.
    pub fn new(config: RecoveryConfig<T>) -> Result<Self> {
        let mut recovery_strategies: HashMap<String, Box<dyn RecoveryStrategy<T>>> = HashMap::new();
        recovery_strategies.insert("default".to_string(), Box::new(DefaultRecoveryStrategy));
        Ok(Self {
            recovery_strategies,
            default_strategy: "default".to_string(),
            config,
            stats: RecoveryStatistics::default(),
        })
    }
    /// Register an additional named recovery strategy. Does not change
    /// `default_strategy`; call [`Self::set_default_strategy`] to select
    /// it.
    pub fn register_strategy(
        &mut self,
        name: impl Into<String>,
        strategy: Box<dyn RecoveryStrategy<T>>,
    ) {
        self.recovery_strategies.insert(name.into(), strategy);
    }
    /// Select which registered strategy [`Self::recover`] dispatches to.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if `name` is not registered.
    pub fn set_default_strategy(&mut self, name: &str) -> Result<()> {
        if !self.recovery_strategies.contains_key(name) {
            return Err(OptimError::InvalidConfig(format!(
                "cannot set default recovery strategy to unregistered name '{name}'"
            )));
        }
        self.default_strategy = name.to_string();
        Ok(())
    }
    /// Dispatches to the strategy registered under `default_strategy`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if `default_strategy` names a
    /// strategy that is not (or is no longer) registered.
    pub fn recover(
        &self,
        checkpoint: &Checkpoint<T>,
        target: &RecoveryTarget,
    ) -> Result<RecoveryResult<T>> {
        let strategy = self
            .recovery_strategies
            .get(&self.default_strategy)
            .ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "no recovery strategy registered under default_strategy '{}'",
                    self.default_strategy
                ))
            })?;
        strategy.recover(checkpoint, target)
    }
}
/// CPU information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CpuInfo {
    /// CPU model
    pub model: String,
    /// Number of cores
    pub cores: usize,
    /// Number of threads
    pub threads: usize,
    /// Base frequency
    pub base_frequency: f64,
    /// Max frequency
    pub max_frequency: f64,
}
/// Types of checkpoints
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum CheckpointType {
    /// Full state checkpoint
    Full,
    /// Incremental checkpoint
    Incremental,
    /// Differential checkpoint
    Differential,
    /// Model parameter checkpoint
    ModelParameters,
    /// Optimizer state checkpoint
    OptimizerState,
    /// Data state checkpoint
    DataState,
    /// Configuration checkpoint
    Configuration,
    /// Emergency checkpoint
    Emergency,
    /// Custom checkpoint type
    Custom(String),
}
/// Types of attachments
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentType {
    /// Configuration file
    Configuration,
    /// Log file
    Log,
    /// Visualization
    Visualization,
    /// Report
    Report,
    /// Custom attachment
    Custom,
}
/// State types for recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateType {
    /// Full state recovery
    Full,
    /// Model parameters only
    ModelOnly,
    /// Optimizer state only
    OptimizerOnly,
    /// Training state only
    TrainingOnly,
    /// Custom state recovery
    Custom,
}
/// Model state information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelState<T: Float + Debug + Send + Sync + 'static> {
    /// Model parameters
    pub parameters: HashMap<String, Array1<T>>,
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Model configuration
    pub configuration: ModelConfiguration<T>,
    /// Model version
    pub version: String,
    /// Model hash
    pub hash: String,
}
/// Warmup configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WarmupConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Warmup steps
    pub warmup_steps: usize,
    /// Warmup method
    pub warmup_method: WarmupMethod,
    /// Initial learning rate
    pub initial_lr: T,
}
/// System information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// OS version
    pub os_version: String,
    /// Architecture
    pub architecture: String,
    /// Hostname
    pub hostname: String,
    /// Username
    pub username: String,
}
/// Indexing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingStrategy {
    /// In-memory indexing
    InMemory,
    /// Persistent indexing
    Persistent,
    /// Distributed indexing
    Distributed,
    /// Custom indexing
    Custom,
}
/// Batch normalization settings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchNormSettings<T: Float + Debug + Send + Sync + 'static> {
    /// Enable batch normalization
    pub enabled: bool,
    /// Momentum
    pub momentum: T,
    /// Epsilon
    pub epsilon: T,
    /// Track running statistics
    pub track_running_stats: bool,
}
/// Index metadata
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// Index format version
    pub format_version: String,
    /// Index creation time
    pub created_at: SystemTime,
    /// Last rebuild time
    pub last_rebuild: SystemTime,
    /// Index statistics
    pub statistics: HashMap<String, usize>,
}
/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total validations performed
    pub total_validations: usize,
    /// Total validations passed
    pub total_passed: usize,
    /// Total validations failed
    pub total_failed: usize,
    /// Average validation time
    pub average_validation_time: Duration,
    /// Validation success rate
    pub success_rate: T,
}
/// Early stopping configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable early stopping
    pub enabled: bool,
    /// Patience (epochs without improvement)
    pub patience: usize,
    /// Minimum improvement delta
    pub min_delta: T,
    /// Metric to monitor
    pub monitor_metric: String,
    /// Monitor mode (min/max)
    pub monitor_mode: MonitorMode,
}
/// A fully real, in-process implementor of [`CheckpointStorage`]: every
/// checkpoint handed to [`Self::store`] is retained (via `Clone`, which
/// `Checkpoint<T>` already derives) and read back with identical field
/// values by [`Self::retrieve`]. There is no filesystem or network I/O --
/// this is [`CheckpointManager`]'s batteries-included default backend,
/// intended for same-process pause/resume/rollback. For persistence across
/// process restarts, implement [`CheckpointStorage`] against real storage
/// (filesystem, object store, database) instead; that is what the trait
/// boundary exists for.
///
/// Regression fix (F9): `CheckpointStorage` had zero implementations
/// anywhere in this crate, so `CheckpointManager::new` (which requires a
/// `Box<dyn CheckpointStorage<T>>`) could not be constructed by any caller
/// without first writing a full backend from scratch.
#[derive(Debug)]
pub struct InMemoryCheckpointStorage<T: Float + Debug + Send + Sync + 'static> {
    pub(super) checkpoints: HashMap<String, Checkpoint<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static> InMemoryCheckpointStorage<T> {
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
        }
    }
    pub(super) fn not_found(checkpoint_id: &str) -> OptimError {
        OptimError::InvalidConfig(format!(
            "checkpoint '{checkpoint_id}' not found in in-memory storage"
        ))
    }
}
/// Checkpoint manager for optimization workflows
#[derive(Debug)]
pub struct CheckpointManager<T: Float + Debug + Send + Sync + 'static> {
    /// Active checkpoints
    pub(super) active_checkpoints: HashMap<String, Checkpoint<T>>,
    /// Checkpoint storage backend
    pub(super) storage_backend: Box<dyn CheckpointStorage<T>>,
    /// Checkpoint scheduler
    pub(super) scheduler: CheckpointScheduler<T>,
    /// Checkpoint validator
    pub(super) validator: CheckpointValidator<T>,
    /// Checkpoint compressor
    pub(super) compressor: CheckpointCompressor<T>,
    /// Recovery manager
    pub(super) recovery_manager: RecoveryManager<T>,
    /// Checkpoint indexer
    pub(super) indexer: CheckpointIndexer<T>,
    /// Manager configuration
    pub(super) config: CheckpointConfiguration<T>,
    /// Manager statistics
    pub(super) stats: CheckpointStatistics<T>,
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> CheckpointManager<T> {
    /// Create new checkpoint manager
    pub fn new(
        config: CheckpointConfiguration<T>,
        storage_backend: Box<dyn CheckpointStorage<T>>,
    ) -> Result<Self> {
        Ok(Self {
            active_checkpoints: HashMap::new(),
            storage_backend,
            scheduler: CheckpointScheduler::new(config.scheduling_config.clone())?,
            validator: CheckpointValidator::new(config.validation_config.clone())?,
            compressor: CheckpointCompressor::new(config.compression_config.clone())?,
            recovery_manager: RecoveryManager::new(config.recovery_config.clone())?,
            indexer: CheckpointIndexer::new(config.indexing_config.clone())?,
            config,
            stats: CheckpointStatistics::default(),
        })
    }
    /// Create a checkpoint
    pub fn create_checkpoint(
        &mut self,
        workflow_id: String,
        checkpoint_type: CheckpointType,
        data: CheckpointData<T>,
    ) -> Result<String> {
        let checkpoint_id = format!(
            "ckpt_{}_{}",
            workflow_id,
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        let mut checkpoint = Checkpoint {
            checkpoint_id: checkpoint_id.clone(),
            workflow_id,
            checkpoint_type,
            data,
            metadata: CheckpointMetadata {
                description: "Auto-generated checkpoint".to_string(),
                tags: Vec::new(),
                version: "1.0".to_string(),
                creator: CreatorInfo {
                    name: "CheckpointManager".to_string(),
                    email: None,
                    tool: "scirs2-optim".to_string(),
                    tool_version: "0.1.0".to_string(),
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
            size_bytes: 0,
            hash: String::new(),
            compression: CompressionInfo::default(),
            dependencies: Vec::new(),
        };
        let validation_result = self.validator.validate(&checkpoint)?;
        checkpoint.metadata.validation_status = ValidationStatus {
            valid: validation_result.valid,
            errors: validation_result
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect(),
            warnings: validation_result
                .warnings
                .iter()
                .map(|w| w.message.clone())
                .collect(),
            validated_at: SystemTime::now(),
            validator_version: "1.0".to_string(),
        };
        let storage_location = self.storage_backend.store(&checkpoint)?;
        checkpoint.metadata.storage_location = storage_location;
        self.indexer.add_entry(&checkpoint)?;
        self.active_checkpoints
            .insert(checkpoint_id.clone(), checkpoint);
        self.stats.total_created += 1;
        Ok(checkpoint_id)
    }
    /// Restore from checkpoint
    pub fn restore_checkpoint(
        &mut self,
        checkpoint_id: &str,
        target: RecoveryTarget,
    ) -> Result<RecoveryResult<T>> {
        let checkpoint = self.storage_backend.retrieve(checkpoint_id)?;
        let result = self.recovery_manager.recover(&checkpoint, &target)?;
        if result.success {
            self.stats.total_restored += 1;
        }
        Ok(result)
    }
    /// Delete checkpoint
    pub fn delete_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        self.storage_backend.delete(checkpoint_id)?;
        self.indexer.remove_entry(checkpoint_id)?;
        self.active_checkpoints.remove(checkpoint_id);
        self.stats.total_deleted += 1;
        Ok(())
    }
    /// List available checkpoints
    pub fn list_checkpoints(&self, workflow_id: Option<&str>) -> Result<Vec<String>> {
        self.storage_backend.list(workflow_id)
    }
    /// Get checkpoint metadata
    pub fn get_checkpoint_metadata(&self, checkpoint_id: &str) -> Result<CheckpointMetadata<T>> {
        self.storage_backend.get_metadata(checkpoint_id)
    }
    /// Get manager statistics
    pub fn get_statistics(&self) -> &CheckpointStatistics<T> {
        &self.stats
    }
}
/// Storage information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StorageInfo {
    /// Storage device
    pub device: String,
    /// Storage type
    pub storage_type: String,
    /// Total capacity (bytes)
    pub total_capacity: usize,
    /// Available capacity (bytes)
    pub available_capacity: usize,
}
/// Learning rate schedule types
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    ReduceOnPlateau,
    Custom,
}
/// Data loader state
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataLoaderState {
    /// Current batch index
    pub current_batch: usize,
    /// Current epoch
    pub current_epoch: usize,
    /// Shuffle state
    pub shuffle_state: Option<ShuffleState>,
    /// Sampler state
    pub sampler_state: Option<SamplerState>,
    /// Data loader configuration
    pub config: DataLoaderConfig,
}
/// Hardware configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareConfig {
    /// CPU information
    pub cpu_info: CpuInfo,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// GPU information
    pub gpu_info: Vec<GpuInfo>,
    /// Storage information
    pub storage_info: Vec<StorageInfo>,
}
/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult<T: Float + Debug + Send + Sync + 'static> {
    /// Recovery success
    pub success: bool,
    /// Recovered state
    pub recovered_state: Option<CheckpointData<T>>,
    /// Recovery metrics
    pub metrics: RecoveryMetrics<T>,
    /// Recovery errors
    pub errors: Vec<String>,
    /// Recovery warnings
    pub warnings: Vec<String>,
}
/// Recovery configuration
#[derive(Debug, Clone)]
pub struct RecoveryConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Default recovery timeout
    pub default_timeout: Duration,
    /// Maximum recovery attempts
    pub max_attempts: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Custom recovery parameters
    pub custom_params: HashMap<String, T>,
}
/// Built-in validation rule: `dependencies` must not list the checkpoint's
/// own id (a direct self-dependency cycle would make
/// [`RecoveryManager::recover`] of a dependency-aware strategy loop
/// forever).
#[derive(Debug)]
pub(super) struct NoSelfDependencyRule;
/// Gradient clipping settings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GradientClippingSettings<T: Float + Debug + Send + Sync + 'static> {
    /// Enable gradient clipping
    pub enabled: bool,
    /// Clipping method
    pub method: ClippingMethod,
    /// Clipping value
    pub value: T,
}
/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}
/// Compressor configuration
#[derive(Debug, Clone)]
pub struct CompressorConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable compression
    pub enable_compression: bool,
    /// Default compression level
    pub default_compression_level: u8,
    /// Size threshold for compression
    pub size_threshold: usize,
    /// Custom compression parameters
    pub custom_params: HashMap<String, T>,
}
/// Indexer configuration
#[derive(Debug, Clone)]
pub struct IndexerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Index rebuild interval
    pub rebuild_interval: Duration,
    /// Index compaction threshold
    pub compaction_threshold: T,
    /// Enable index caching
    pub enable_caching: bool,
    /// Custom indexer parameters
    pub custom_params: HashMap<String, T>,
}
/// Checkpoint priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}
/// Built-in validation rule: `checkpoint_id` and `workflow_id` must be
/// non-empty, since an empty identifier can never be looked back up via
/// [`CheckpointStorage::retrieve`]/[`CheckpointManager::list_checkpoints`].
#[derive(Debug)]
pub(super) struct RequiredIdentifiersRule;
/// Scheduled checkpoint
#[derive(Debug, Clone)]
pub struct ScheduledCheckpoint {
    /// Workflow identifier
    pub workflow_id: String,
    /// Checkpoint type
    pub checkpoint_type: CheckpointType,
    /// Scheduled time
    pub scheduled_time: SystemTime,
    /// Priority
    pub priority: CheckpointPriority,
    /// Scheduling reason
    pub reason: SchedulingReason,
}
/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total bytes compressed
    pub total_bytes_compressed: usize,
    /// Total bytes decompressed
    pub total_bytes_decompressed: usize,
    /// Overall compression ratio
    pub overall_compression_ratio: T,
    /// Total compression time
    pub total_compression_time: Duration,
    /// Total decompression time
    pub total_decompression_time: Duration,
}
/// Training state information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingState<T: Float + Debug + Send + Sync + 'static> {
    /// Current epoch
    pub current_epoch: usize,
    /// Current step
    pub current_step: usize,
    /// Training metrics history
    pub training_metrics: MetricsHistory<T>,
    /// Validation metrics history
    pub validation_metrics: MetricsHistory<T>,
    /// Loss history
    pub loss_history: Vec<T>,
    /// Best metric values
    pub best_metrics: HashMap<String, T>,
    /// Training configuration
    pub training_config: TrainingConfiguration<T>,
}
/// Recovery target specification
#[derive(Debug, Clone)]
pub struct RecoveryTarget {
    /// Target workflow ID
    pub workflow_id: String,
    /// Target state type
    pub state_type: StateType,
    /// Recovery options
    pub options: RecoveryOptions,
    /// Target environment
    pub environment: Option<EnvironmentState>,
}
/// Creator information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreatorInfo {
    /// Creator name
    pub name: String,
    /// Creator email
    pub email: Option<String>,
    /// Creation tool
    pub tool: String,
    /// Tool version
    pub tool_version: String,
}
/// Optimization settings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptimizationSettings<T: Float + Debug + Send + Sync + 'static> {
    /// Optimizer type
    pub optimizer_type: String,
    /// Optimizer parameters
    pub parameters: HashMap<String, T>,
    /// Gradient clipping
    pub gradient_clipping: GradientClippingSettings<T>,
    /// Weight decay
    pub weight_decay: T,
}
/// Model configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Hyperparameters
    pub hyperparameters: HashMap<String, T>,
    /// Training configuration
    pub training_config: TrainingConfiguration<T>,
    /// Regularization settings
    pub regularization: RegularizationSettings<T>,
    /// Optimization settings
    pub optimization: OptimizationSettings<T>,
}
/// Sampler state information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplerState {
    /// Sampler type
    pub sampler_type: String,
    /// Sampler parameters
    pub parameters: HashMap<String, String>,
    /// Current state
    pub state: Vec<u8>,
}
/// Learning rate schedule
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LearningRateSchedule<T: Float + Debug + Send + Sync + 'static> {
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Schedule parameters
    pub parameters: HashMap<String, T>,
    /// Warmup configuration
    pub warmup: Option<WarmupConfig<T>>,
}
/// GPU information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GpuInfo {
    /// GPU model
    pub model: String,
    /// GPU memory (bytes)
    pub memory: usize,
    /// Compute capability
    pub compute_capability: String,
    /// Driver version
    pub driver_version: String,
}
/// Scheduling reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingReason {
    /// Periodic checkpoint
    Periodic,
    /// Progress-based checkpoint
    Progress,
    /// Performance-based checkpoint
    Performance,
    /// Error recovery checkpoint
    ErrorRecovery,
    /// Manual request
    Manual,
    /// System shutdown
    Shutdown,
}
/// Compression algorithm statistics
#[derive(Debug, Clone)]
pub struct CompressionAlgorithmStats {
    /// Total compressions performed
    pub total_compressions: usize,
    /// Total decompressions performed
    pub total_decompressions: usize,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Average compression time
    pub average_compression_time: Duration,
    /// Average decompression time
    pub average_decompression_time: Duration,
}
/// Built-in validation rule: warns (does not fail) when a checkpoint's
/// `data` carries no model/optimizer/training/loader/rng/environment
/// state and no custom state or attachments either -- almost always a
/// caller bug (nothing was populated on `CheckpointData` before
/// `create_checkpoint`), surfaced as a warning rather than an error since
/// an intentionally empty marker checkpoint is a legitimate, if unusual,
/// use of the API.
#[derive(Debug)]
pub(super) struct NonEmptyDataRule;
/// Connection specification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectionSpec {
    /// Connection identifier
    pub connection_id: String,
    /// Source layer
    pub source_layer: String,
    /// Target layer
    pub target_layer: String,
    /// Connection type
    pub connection_type: String,
    /// Connection parameters
    pub parameters: HashMap<String, String>,
}
/// Layer specification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerSpec {
    /// Layer identifier
    pub layer_id: String,
    /// Layer type
    pub layer_type: String,
    /// Layer parameters
    pub parameters: HashMap<String, String>,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shapes
    pub output_shapes: Vec<Vec<usize>>,
}
/// Training configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Learning rate
    pub learning_rate: T,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Validation frequency
    pub validation_frequency: usize,
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig<T>,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule<T>,
}
/// Checkpoint data container
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckpointData<T: Float + Debug + Send + Sync + 'static> {
    /// Model state
    pub model_state: Option<ModelState<T>>,
    /// Optimizer state
    pub optimizer_state: Option<OptimizerState<T>>,
    /// Training state
    pub training_state: Option<TrainingState<T>>,
    /// Data loader state
    pub data_loader_state: Option<DataLoaderState>,
    /// Random number generator state
    pub rng_state: Option<RngState>,
    /// Environment state
    pub environment_state: Option<EnvironmentState>,
    /// Custom state data
    pub custom_state: HashMap<String, Vec<u8>>,
    /// Metadata attachments
    pub attachments: HashMap<String, Attachment>,
}
