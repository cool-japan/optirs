// Compression Pipelines and Stage Management
//
// This module handles compression pipeline configuration and execution
// for TPU event synchronization systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Compression pipelines configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPipelines {
    /// Available pipelines
    pub pipelines: Vec<Pipeline>,
    /// Default pipeline
    pub default_pipeline: String,
    /// Pipeline execution settings
    pub execution: PipelineExecution,
    /// Pipeline optimization
    pub optimization: PipelineOptimization,
    /// Pipeline monitoring
    pub monitoring: PipelineMonitoring,
}

impl Default for CompressionPipelines {
    fn default() -> Self {
        Self {
            pipelines: vec![
                Pipeline::standard(),
                Pipeline::fast(),
                Pipeline::high_compression(),
            ],
            default_pipeline: "standard".to_string(),
            execution: PipelineExecution::default(),
            optimization: PipelineOptimization::default(),
            monitoring: PipelineMonitoring::default(),
        }
    }
}

/// Compression pipeline definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Pipeline name
    pub name: String,
    /// Pipeline description
    pub description: String,
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Pipeline configuration
    pub config: PipelineConfig,
}

impl Pipeline {
    /// Standard compression pipeline
    pub fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            description: "Standard compression pipeline with balanced performance".to_string(),
            stages: vec![
                PipelineStage {
                    name: "preprocessing".to_string(),
                    stage_type: StageType::Preprocessing,
                    config: StageConfig::default(),
                    enabled: true,
                },
                PipelineStage {
                    name: "compression".to_string(),
                    stage_type: StageType::Compression,
                    config: StageConfig::compression(),
                    enabled: true,
                },
                PipelineStage {
                    name: "validation".to_string(),
                    stage_type: StageType::Validation,
                    config: StageConfig::validation(),
                    enabled: true,
                },
            ],
            config: PipelineConfig::default(),
        }
    }

    /// Fast compression pipeline
    pub fn fast() -> Self {
        Self {
            name: "fast".to_string(),
            description: "Fast compression pipeline optimized for speed".to_string(),
            stages: vec![
                PipelineStage {
                    name: "compression".to_string(),
                    stage_type: StageType::Compression,
                    config: StageConfig::fast_compression(),
                    enabled: true,
                },
            ],
            config: PipelineConfig::fast(),
        }
    }

    /// High compression pipeline
    pub fn high_compression() -> Self {
        Self {
            name: "high_compression".to_string(),
            description: "High compression pipeline optimized for ratio".to_string(),
            stages: vec![
                PipelineStage {
                    name: "preprocessing".to_string(),
                    stage_type: StageType::Preprocessing,
                    config: StageConfig::advanced_preprocessing(),
                    enabled: true,
                },
                PipelineStage {
                    name: "analysis".to_string(),
                    stage_type: StageType::Analysis,
                    config: StageConfig::analysis(),
                    enabled: true,
                },
                PipelineStage {
                    name: "compression".to_string(),
                    stage_type: StageType::Compression,
                    config: StageConfig::high_compression(),
                    enabled: true,
                },
                PipelineStage {
                    name: "optimization".to_string(),
                    stage_type: StageType::Optimization,
                    config: StageConfig::optimization(),
                    enabled: true,
                },
                PipelineStage {
                    name: "validation".to_string(),
                    stage_type: StageType::Validation,
                    config: StageConfig::thorough_validation(),
                    enabled: true,
                },
            ],
            config: PipelineConfig::high_compression(),
        }
    }
}

/// Pipeline stage definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name
    pub name: String,
    /// Stage type
    pub stage_type: StageType,
    /// Stage configuration
    pub config: StageConfig,
    /// Stage enabled
    pub enabled: bool,
}

/// Pipeline stage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageType {
    /// Data preprocessing
    Preprocessing,
    /// Data analysis
    Analysis,
    /// Compression
    Compression,
    /// Optimization
    Optimization,
    /// Validation
    Validation,
    /// Post-processing
    PostProcessing,
    /// Custom stage
    Custom(String),
}

/// Stage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Stage parameters
    pub parameters: HashMap<String, String>,
    /// Stage timeout
    pub timeout: Option<Duration>,
    /// Stage retry configuration
    pub retry: Option<RetryConfig>,
    /// Stage dependencies
    pub dependencies: Vec<String>,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            timeout: Some(Duration::from_secs(60)),
            retry: Some(RetryConfig::default()),
            dependencies: Vec::new(),
        }
    }
}

impl StageConfig {
    /// Compression stage configuration
    pub fn compression() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("algorithm".to_string(), "zstd".to_string());
        parameters.insert("level".to_string(), "3".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(30)),
            retry: Some(RetryConfig::default()),
            dependencies: vec!["preprocessing".to_string()],
        }
    }

    /// Fast compression configuration
    pub fn fast_compression() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("algorithm".to_string(), "lz4".to_string());
        parameters.insert("level".to_string(), "1".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(5)),
            retry: Some(RetryConfig::minimal()),
            dependencies: Vec::new(),
        }
    }

    /// High compression configuration
    pub fn high_compression() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("algorithm".to_string(), "zstd".to_string());
        parameters.insert("level".to_string(), "15".to_string());
        parameters.insert("dictionary".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(300)),
            retry: Some(RetryConfig::robust()),
            dependencies: vec!["preprocessing".to_string(), "analysis".to_string()],
        }
    }

    /// Advanced preprocessing configuration
    pub fn advanced_preprocessing() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("deduplication".to_string(), "enabled".to_string());
        parameters.insert("sorting".to_string(), "enabled".to_string());
        parameters.insert("filtering".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(120)),
            retry: Some(RetryConfig::default()),
            dependencies: Vec::new(),
        }
    }

    /// Analysis stage configuration
    pub fn analysis() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("entropy_analysis".to_string(), "enabled".to_string());
        parameters.insert("pattern_detection".to_string(), "enabled".to_string());
        parameters.insert("compression_prediction".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(60)),
            retry: Some(RetryConfig::default()),
            dependencies: vec!["preprocessing".to_string()],
        }
    }

    /// Optimization stage configuration
    pub fn optimization() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("parameter_tuning".to_string(), "enabled".to_string());
        parameters.insert("dictionary_optimization".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(180)),
            retry: Some(RetryConfig::default()),
            dependencies: vec!["compression".to_string()],
        }
    }

    /// Validation stage configuration
    pub fn validation() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("integrity_check".to_string(), "enabled".to_string());
        parameters.insert("ratio_validation".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(30)),
            retry: Some(RetryConfig::validation()),
            dependencies: vec!["compression".to_string()],
        }
    }

    /// Thorough validation configuration
    pub fn thorough_validation() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("integrity_check".to_string(), "enabled".to_string());
        parameters.insert("ratio_validation".to_string(), "enabled".to_string());
        parameters.insert("decompression_test".to_string(), "enabled".to_string());
        parameters.insert("performance_validation".to_string(), "enabled".to_string());

        Self {
            parameters,
            timeout: Some(Duration::from_secs(120)),
            retry: Some(RetryConfig::thorough()),
            dependencies: vec!["compression".to_string(), "optimization".to_string()],
        }
    }
}

/// Retry configuration for pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Initial delay
    pub initial_delay: Duration,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
    /// Maximum delay
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            backoff: BackoffStrategy::Exponential,
            max_delay: Duration::from_secs(5),
        }
    }
}

impl RetryConfig {
    /// Minimal retry configuration
    pub fn minimal() -> Self {
        Self {
            max_attempts: 1,
            initial_delay: Duration::from_millis(10),
            backoff: BackoffStrategy::None,
            max_delay: Duration::from_millis(100),
        }
    }

    /// Robust retry configuration
    pub fn robust() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            backoff: BackoffStrategy::ExponentialWithJitter,
            max_delay: Duration::from_secs(30),
        }
    }

    /// Validation retry configuration
    pub fn validation() -> Self {
        Self {
            max_attempts: 2,
            initial_delay: Duration::from_millis(50),
            backoff: BackoffStrategy::Linear,
            max_delay: Duration::from_secs(2),
        }
    }

    /// Thorough retry configuration
    pub fn thorough() -> Self {
        Self {
            max_attempts: 10,
            initial_delay: Duration::from_millis(100),
            backoff: BackoffStrategy::ExponentialWithJitter,
            max_delay: Duration::from_secs(60),
        }
    }
}

/// Backoff strategies for retries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    None,
    Linear,
    Exponential,
    ExponentialWithJitter,
    Fixed(Duration),
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline timeout
    pub timeout: Duration,
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Parallel execution settings
    pub parallel_execution: ParallelExecution,
    /// Error handling
    pub error_handling: PipelineErrorHandling,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            execution_mode: ExecutionMode::Sequential,
            parallel_execution: ParallelExecution::default(),
            error_handling: PipelineErrorHandling::default(),
        }
    }
}

impl PipelineConfig {
    /// Fast pipeline configuration
    pub fn fast() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            execution_mode: ExecutionMode::Parallel,
            parallel_execution: ParallelExecution::aggressive(),
            error_handling: PipelineErrorHandling::fail_fast(),
        }
    }

    /// High compression pipeline configuration
    pub fn high_compression() -> Self {
        Self {
            timeout: Duration::from_secs(1800), // 30 minutes
            execution_mode: ExecutionMode::Sequential,
            parallel_execution: ParallelExecution::conservative(),
            error_handling: PipelineErrorHandling::robust(),
        }
    }
}

/// Pipeline execution modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Hybrid execution
    Hybrid,
    /// Adaptive execution
    Adaptive,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecution {
    /// Enable parallel execution
    pub enabled: bool,
    /// Maximum parallel stages
    pub max_parallel_stages: usize,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Work stealing enabled
    pub work_stealing: bool,
}

impl Default for ParallelExecution {
    fn default() -> Self {
        Self {
            enabled: false,
            max_parallel_stages: 2,
            thread_pool_size: 4,
            work_stealing: true,
        }
    }
}

impl ParallelExecution {
    /// Aggressive parallel execution
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            max_parallel_stages: 8,
            thread_pool_size: 16,
            work_stealing: true,
        }
    }

    /// Conservative parallel execution
    pub fn conservative() -> Self {
        Self {
            enabled: false,
            max_parallel_stages: 1,
            thread_pool_size: 2,
            work_stealing: false,
        }
    }
}

/// Pipeline error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineErrorHandling {
    /// Error handling strategy
    pub strategy: ErrorHandlingStrategy,
    /// Continue on error
    pub continue_on_error: bool,
    /// Rollback on failure
    pub rollback_on_failure: bool,
    /// Error notification
    pub error_notification: bool,
}

impl Default for PipelineErrorHandling {
    fn default() -> Self {
        Self {
            strategy: ErrorHandlingStrategy::StopOnError,
            continue_on_error: false,
            rollback_on_failure: true,
            error_notification: true,
        }
    }
}

impl PipelineErrorHandling {
    /// Fail-fast error handling
    pub fn fail_fast() -> Self {
        Self {
            strategy: ErrorHandlingStrategy::FailFast,
            continue_on_error: false,
            rollback_on_failure: false,
            error_notification: false,
        }
    }

    /// Robust error handling
    pub fn robust() -> Self {
        Self {
            strategy: ErrorHandlingStrategy::RetryAndContinue,
            continue_on_error: true,
            rollback_on_failure: true,
            error_notification: true,
        }
    }
}

/// Error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingStrategy {
    StopOnError,
    FailFast,
    RetryAndContinue,
    SkipAndContinue,
    FallbackToDefault,
}

/// Pipeline execution settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecution {
    /// Execution scheduler
    pub scheduler: ExecutionScheduler,
    /// Resource management
    pub resource_management: ResourceManagement,
    /// State management
    pub state_management: StateManagement,
}

impl Default for PipelineExecution {
    fn default() -> Self {
        Self {
            scheduler: ExecutionScheduler::default(),
            resource_management: ResourceManagement::default(),
            state_management: StateManagement::default(),
        }
    }
}

/// Execution scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionScheduler {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Priority handling
    pub priority_handling: bool,
    /// Load balancing
    pub load_balancing: bool,
    /// Queue management
    pub queue_management: QueueManagement,
}

impl Default for ExecutionScheduler {
    fn default() -> Self {
        Self {
            algorithm: SchedulingAlgorithm::FIFO,
            priority_handling: false,
            load_balancing: false,
            queue_management: QueueManagement::default(),
        }
    }
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    FIFO,
    LIFO,
    Priority,
    ShortestJobFirst,
    RoundRobin,
    FairShare,
}

/// Queue management for pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueManagement {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Queue overflow strategy
    pub overflow_strategy: QueueOverflowStrategy,
    /// Queue priorities
    pub priorities: usize,
}

impl Default for QueueManagement {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            overflow_strategy: QueueOverflowStrategy::Block,
            priorities: 3,
        }
    }
}

/// Queue overflow strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueOverflowStrategy {
    Block,
    Drop,
    DropOldest,
    Expand,
}

/// Resource management for pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagement {
    /// Memory limits
    pub memory_limits: MemoryLimits,
    /// CPU limits
    pub cpu_limits: CpuLimits,
    /// Resource isolation
    pub isolation: ResourceIsolation,
}

impl Default for ResourceManagement {
    fn default() -> Self {
        Self {
            memory_limits: MemoryLimits::default(),
            cpu_limits: CpuLimits::default(),
            isolation: ResourceIsolation::default(),
        }
    }
}

/// Memory limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum memory per pipeline
    pub max_memory_per_pipeline: usize,
    /// Maximum memory per stage
    pub max_memory_per_stage: usize,
    /// Memory monitoring enabled
    pub monitoring_enabled: bool,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_memory_per_pipeline: 1024 * 1024 * 1024, // 1GB
            max_memory_per_stage: 256 * 1024 * 1024, // 256MB
            monitoring_enabled: true,
        }
    }
}

/// CPU limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU usage per pipeline
    pub max_cpu_per_pipeline: f64,
    /// Maximum CPU usage per stage
    pub max_cpu_per_stage: f64,
    /// CPU monitoring enabled
    pub monitoring_enabled: bool,
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_cpu_per_pipeline: 0.8, // 80%
            max_cpu_per_stage: 0.5, // 50%
            monitoring_enabled: true,
        }
    }
}

/// Resource isolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceIsolation {
    /// Enable isolation
    pub enabled: bool,
    /// Isolation method
    pub method: IsolationMethod,
    /// Resource containers
    pub containers: bool,
}

impl Default for ResourceIsolation {
    fn default() -> Self {
        Self {
            enabled: false,
            method: IsolationMethod::ProcessBased,
            containers: false,
        }
    }
}

/// Resource isolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationMethod {
    ThreadBased,
    ProcessBased,
    ContainerBased,
    VirtualMachine,
}

/// State management for pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateManagement {
    /// State persistence
    pub persistence: StatePersistence,
    /// State recovery
    pub recovery: StateRecovery,
    /// State monitoring
    pub monitoring: StateMonitoring,
}

impl Default for StateManagement {
    fn default() -> Self {
        Self {
            persistence: StatePersistence::default(),
            recovery: StateRecovery::default(),
            monitoring: StateMonitoring::default(),
        }
    }
}

/// State persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatePersistence {
    /// Enable state persistence
    pub enabled: bool,
    /// Persistence method
    pub method: PersistenceMethod,
    /// Persistence frequency
    pub frequency: PersistenceFrequency,
}

impl Default for StatePersistence {
    fn default() -> Self {
        Self {
            enabled: false,
            method: PersistenceMethod::Memory,
            frequency: PersistenceFrequency::OnStageCompletion,
        }
    }
}

/// State persistence methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceMethod {
    Memory,
    File,
    Database,
    DistributedCache,
}

/// State persistence frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceFrequency {
    OnStageCompletion,
    Periodic(Duration),
    OnCheckpoint,
    Continuous,
}

/// State recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateRecovery {
    /// Enable state recovery
    pub enabled: bool,
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// Recovery timeout
    pub timeout: Duration,
}

impl Default for StateRecovery {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: RecoveryStrategy::RestartFromBeginning,
            timeout: Duration::from_secs(300),
        }
    }
}

/// State recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    RestartFromBeginning,
    RestartFromLastCheckpoint,
    RestartFromFailedStage,
    ContinueFromState,
}

/// State monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMonitoring {
    /// Enable state monitoring
    pub enabled: bool,
    /// Monitoring frequency
    pub frequency: Duration,
    /// State metrics
    pub metrics: Vec<StateMetric>,
}

impl Default for StateMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(30),
            metrics: vec![
                StateMetric::PipelineStatus,
                StateMetric::StageProgress,
                StateMetric::ResourceUsage,
            ],
        }
    }
}

/// State metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateMetric {
    PipelineStatus,
    StageProgress,
    ResourceUsage,
    ErrorCount,
    ExecutionTime,
    ThroughputRate,
}

/// Pipeline optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOptimization {
    /// Optimization enabled
    pub enabled: bool,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Optimization frequency
    pub frequency: Duration,
    /// Performance targets
    pub targets: OptimizationTargets,
}

impl Default for PipelineOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                OptimizationStrategy::StageReordering,
                OptimizationStrategy::ParameterTuning,
            ],
            frequency: Duration::from_secs(3600), // 1 hour
            targets: OptimizationTargets::default(),
        }
    }
}

/// Pipeline optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    StageReordering,
    ParameterTuning,
    ResourceAllocation,
    ParallelizationOptimization,
    CacheOptimization,
    PipelineSelection,
}

/// Optimization targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTargets {
    /// Target execution time
    pub execution_time: Option<Duration>,
    /// Target throughput
    pub throughput: Option<f64>,
    /// Target compression ratio
    pub compression_ratio: Option<f64>,
    /// Target resource usage
    pub resource_usage: Option<f64>,
}

impl Default for OptimizationTargets {
    fn default() -> Self {
        Self {
            execution_time: Some(Duration::from_secs(60)),
            throughput: Some(1_000_000.0), // 1 MB/s
            compression_ratio: Some(2.0), // 2:1
            resource_usage: Some(0.5), // 50%
        }
    }
}

/// Pipeline monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMonitoring {
    /// Monitoring enabled
    pub enabled: bool,
    /// Monitoring configuration
    pub config: MonitoringConfig,
    /// Performance tracking
    pub performance_tracking: PerformanceTracking,
    /// Health monitoring
    pub health_monitoring: HealthMonitoring,
}

impl Default for PipelineMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            config: MonitoringConfig::default(),
            performance_tracking: PerformanceTracking::default(),
            health_monitoring: HealthMonitoring::default(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Monitoring frequency
    pub frequency: Duration,
    /// Monitored metrics
    pub metrics: Vec<MonitoringMetric>,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
    /// Monitoring storage
    pub storage: MonitoringStorage,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(30),
            metrics: vec![
                MonitoringMetric::ExecutionTime,
                MonitoringMetric::StageProgress,
                MonitoringMetric::ResourceUsage,
                MonitoringMetric::ErrorRate,
            ],
            thresholds: HashMap::new(),
            storage: MonitoringStorage::default(),
        }
    }
}

/// Monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringMetric {
    ExecutionTime,
    StageProgress,
    ResourceUsage,
    ErrorRate,
    ThroughputRate,
    QueueLength,
    MemoryUsage,
    CpuUsage,
}

/// Monitoring storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStorage {
    /// Storage backend
    pub backend: StorageBackend,
    /// Retention period
    pub retention: Duration,
    /// Compression enabled
    pub compression: bool,
}

impl Default for MonitoringStorage {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Memory,
            retention: Duration::from_secs(3600), // 1 hour
            compression: true,
        }
    }
}

/// Storage backends for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Memory,
    File,
    Database,
    TimeSeries,
}

/// Performance tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTracking {
    /// Enable performance tracking
    pub enabled: bool,
    /// Tracking granularity
    pub granularity: TrackingGranularity,
    /// Performance baselines
    pub baselines: PerformanceBaselines,
    /// Trend analysis
    pub trend_analysis: bool,
}

impl Default for PerformanceTracking {
    fn default() -> Self {
        Self {
            enabled: true,
            granularity: TrackingGranularity::Stage,
            baselines: PerformanceBaselines::default(),
            trend_analysis: true,
        }
    }
}

/// Performance tracking granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackingGranularity {
    Pipeline,
    Stage,
    Operation,
    Fine,
}

/// Performance baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaselines {
    /// Baseline metrics
    pub metrics: HashMap<String, f64>,
    /// Baseline update frequency
    pub update_frequency: Duration,
    /// Deviation thresholds
    pub deviation_thresholds: HashMap<String, f64>,
}

impl Default for PerformanceBaselines {
    fn default() -> Self {
        Self {
            metrics: HashMap::new(),
            update_frequency: Duration::from_secs(86400), // 24 hours
            deviation_thresholds: HashMap::new(),
        }
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoring {
    /// Enable health monitoring
    pub enabled: bool,
    /// Health check frequency
    pub frequency: Duration,
    /// Health indicators
    pub indicators: Vec<HealthIndicator>,
    /// Health actions
    pub actions: Vec<HealthAction>,
}

impl Default for HealthMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(60),
            indicators: vec![
                HealthIndicator::PipelineStatus,
                HealthIndicator::ResourceHealth,
                HealthIndicator::ErrorRate,
            ],
            actions: vec![
                HealthAction::Alert,
                HealthAction::Log,
            ],
        }
    }
}

/// Health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthIndicator {
    PipelineStatus,
    StageHealth,
    ResourceHealth,
    ErrorRate,
    PerformanceDeviations,
    QueueHealth,
}

/// Health actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthAction {
    Alert,
    Log,
    AutoRestart,
    Fallback,
    Notification,
    Escalation,
}