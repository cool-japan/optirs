//! Data-only types for the TPU backend: device/topology descriptors,
//! execution-task and buffer metadata, program/compile artifacts, and
//! statistics structs.
//!
//! No private-field-reaching logic lives here; see the sibling modules
//! (`backend`, `device_manager`, `execution`, `memory`, `profiling`,
//! `buffer`) for the types that pair private data with behavior.

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use scirs2_core::numeric::Float;

use super::buffer::TPUBuffer;
// `ComputationId` is the real XLA frontend's computation identifier
// (`xla::frontend::graph_capture::ComputationId`); re-exported here (not just
// `use`d) so `tpu_backend::ComputationId` keeps working through this module's
// `pub use types::*;` in `tpu_backend/mod.rs`, same as before the now-deleted
// `xla_compilation` duplicate module was folded into `xla`.
pub use crate::xla::ComputationId;
use crate::{PodTopology, TPUConfig, TPUVersion, XLAOptimizationLevel};

/// TPU backend configuration
#[derive(Debug, Clone)]
pub struct TPUBackendConfig {
    /// Target TPU configuration
    pub tpu_config: TPUConfig,

    /// Enable runtime optimization
    pub runtime_optimization: bool,

    /// Enable automatic memory management
    pub auto_memory_management: bool,

    /// Execution timeout (milliseconds)
    pub execution_timeout_ms: u64,

    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,

    /// Buffer size for async execution
    pub async_buffer_size: usize,

    /// Enable error recovery
    pub enable_error_recovery: bool,

    /// Maximum retry attempts
    pub max_retry_attempts: usize,

    /// Prefetch strategy
    pub prefetch_strategy: PrefetchStrategy,

    /// Memory allocation strategy
    pub memory_allocation_strategy: MemoryAllocationStrategy,
}

impl Default for TPUBackendConfig {
    fn default() -> Self {
        Self {
            tpu_config: TPUConfig::default(),
            runtime_optimization: true,
            auto_memory_management: true,
            execution_timeout_ms: 30000,
            enable_performance_monitoring: true,
            async_buffer_size: 32,
            enable_error_recovery: true,
            max_retry_attempts: 3,
            prefetch_strategy: PrefetchStrategy::Adaptive,
            memory_allocation_strategy: MemoryAllocationStrategy::BestFit,
        }
    }
}

/// TPU device representation
#[derive(Debug, Clone)]
pub struct TPUDevice {
    /// Device ID
    pub id: DeviceId,

    /// Device type
    pub device_type: TPUVersion,

    /// Memory capacity (bytes)
    pub memory_capacity: usize,

    /// Compute capability
    pub compute_capability: ComputeCapability,

    /// Device status
    pub status: DeviceStatus,

    /// Interconnect links
    pub interconnect_links: Vec<InterconnectLink>,

    /// Device coordinates in pod
    pub coordinates: Option<(usize, usize)>,

    /// Performance characteristics
    pub performance_characteristics: DevicePerformanceCharacteristics,
}

/// Unique device identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

/// Device status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceStatus {
    Available,
    Busy,
    Error,
    Maintenance,
    Offline,
}

/// Device health status
#[derive(Debug, Clone)]
pub struct DeviceHealthStatus {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Temperature (Celsius)
    pub temperature: f64,

    /// Power consumption (Watts)
    pub power_consumption: f64,

    /// Memory health
    pub memory_health: MemoryHealthStatus,

    /// Compute unit health
    pub compute_health: ComputeHealthStatus,

    /// Last health check
    pub last_check: Instant,
}

/// Memory health status
#[derive(Debug, Clone)]
pub struct MemoryHealthStatus {
    /// Memory errors detected
    pub error_count: usize,

    /// Memory bandwidth efficiency
    pub bandwidth_efficiency: f64,

    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Compute unit health status
#[derive(Debug, Clone)]
pub struct ComputeHealthStatus {
    /// Matrix unit efficiency
    pub matrix_unit_efficiency: f64,

    /// Vector unit efficiency
    pub vector_unit_efficiency: f64,

    /// Scalar unit efficiency
    pub scalar_unit_efficiency: f64,

    /// Instruction cache hit rate
    pub instruction_cache_hit_rate: f64,
}

/// Compute capability description
#[derive(Debug, Clone)]
pub struct ComputeCapability {
    /// Peak FLOPS (operations per second)
    pub peak_flops: u64,

    /// Matrix multiplication FLOPS
    pub matrix_flops: u64,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gb_s: f64,

    /// Supported data types
    pub supported_dtypes: Vec<DataType>,

    /// Maximum dimensions
    pub max_dimensions: usize,

    /// Special features
    pub features: Vec<TPUFeature>,
}

/// Supported data types
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    F16,
    F32,
    BF16,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
    Bool,
}

/// TPU-specific features
#[derive(Debug, Clone)]
pub enum TPUFeature {
    MatrixUnits,
    VectorUnits,
    HighBandwidthMemory,
    MixedPrecision,
    SparsitySupport,
    TransformerOptimizations,
    ConvolutionOptimizations,
}

/// Device performance characteristics
#[derive(Debug, Clone)]
pub struct DevicePerformanceCharacteristics {
    /// Effective memory bandwidth
    pub effective_memory_bandwidth: f64,

    /// Compute utilization efficiency
    pub compute_efficiency: f64,

    /// Communication latency (microseconds)
    pub communication_latency_us: f64,

    /// Thermal throttling threshold
    pub thermal_threshold: f64,
}

/// Interconnect link between devices
#[derive(Debug, Clone)]
pub struct InterconnectLink {
    /// Target device
    pub target_device: DeviceId,

    /// Link bandwidth (GB/s)
    pub bandwidth_gb_s: f64,

    /// Link latency (microseconds)
    pub latency_us: f64,

    /// Link type
    pub link_type: InterconnectType,

    /// Link status
    pub status: LinkStatus,
}

/// Types of interconnect
#[derive(Debug, Clone, Copy)]
pub enum InterconnectType {
    IntraChip,
    InterChip,
    InterNode,
    HighSpeed,
    LowLatency,
}

/// Link status
#[derive(Debug, Clone, Copy)]
pub enum LinkStatus {
    Active,
    Inactive,
    Error,
    Degraded,
}

/// Topology types
#[derive(Debug, Clone, Copy)]
pub enum TopologyType {
    Linear,
    Ring,
    Mesh2D,
    Mesh3D,
    Torus,
    Tree,
    HyperCube,
    Custom,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastLoaded,
    PowerAware,
    LocalityAware,
    Adaptive,
    WorkStealing,
}

/// Load sample for monitoring
#[derive(Debug, Clone)]
pub struct LoadSample {
    /// Timestamp
    pub timestamp: Instant,

    /// Utilization (0.0 to 1.0)
    pub utilization: f64,

    /// Memory usage (0.0 to 1.0)
    pub memory_usage: f64,

    /// Temperature
    pub temperature: f64,
}

/// Assignment statistics
#[derive(Debug, Clone)]
pub struct AssignmentStatistics {
    /// Total assignments
    pub total_assignments: usize,

    /// Average assignment time
    pub avg_assignment_time: Duration,

    /// Load balance efficiency
    pub load_balance_efficiency: f64,

    /// Device utilization variance
    pub utilization_variance: f64,
}

/// Execution task
#[derive(Debug)]
pub struct ExecutionTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task ID
    pub id: TaskId,

    /// Computation to execute
    pub computation: ComputationId,

    /// Input data
    pub inputs: Vec<TPUBuffer<T>>,

    /// Expected outputs
    pub expected_outputs: Vec<OutputSpec<T>>,

    /// Execution priority
    pub priority: TaskPriority,

    /// Task dependencies
    pub dependencies: Vec<TaskId>,

    /// Execution constraints
    pub constraints: ExecutionConstraints,

    /// Timeout
    pub timeout: Duration,
}

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
    Realtime,
}

/// Execution constraints
#[derive(Debug, Clone)]
pub struct ExecutionConstraints {
    /// Required device features
    pub required_features: Vec<TPUFeature>,

    /// Memory constraints
    pub memory_constraints: MemoryConstraints,

    /// Performance constraints
    pub performance_constraints: PerformanceConstraints,

    /// Locality constraints
    pub locality_constraints: LocalityConstraints,
}

impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self {
            required_features: Vec::new(),
            memory_constraints: MemoryConstraints {
                max_memory_usage: usize::MAX,
                min_bandwidth_gb_s: 0.0,
                layout_preferences: vec![MemoryLayout::RowMajor],
            },
            performance_constraints: PerformanceConstraints {
                max_execution_time: Duration::from_secs(300),
                min_throughput: 0.0,
                max_latency: Duration::from_millis(100),
                power_budget: None,
            },
            locality_constraints: LocalityConstraints {
                preferred_devices: Vec::new(),
                avoid_devices: Vec::new(),
                locality_scope: LocalityScope::Global,
            },
        }
    }
}

/// Memory constraints
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum memory usage
    pub max_memory_usage: usize,

    /// Memory bandwidth requirement
    pub min_bandwidth_gb_s: f64,

    /// Memory layout preferences
    pub layout_preferences: Vec<MemoryLayout>,
}

/// Performance constraints
#[derive(Debug, Clone)]
pub struct PerformanceConstraints {
    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Minimum throughput
    pub min_throughput: f64,

    /// Maximum latency
    pub max_latency: Duration,

    /// Power constraints
    pub power_budget: Option<f64>,
}

/// Locality constraints
#[derive(Debug, Clone)]
pub struct LocalityConstraints {
    /// Preferred devices
    pub preferred_devices: Vec<DeviceId>,

    /// Avoid devices
    pub avoid_devices: Vec<DeviceId>,

    /// Locality scope
    pub locality_scope: LocalityScope,
}

/// Locality scope
#[derive(Debug, Clone, Copy)]
pub enum LocalityScope {
    Device,
    Chip,
    Node,
    Pod,
    Global,
}

/// Memory layout types
#[derive(Debug, Clone, Copy)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked,
    Tiled,
    Sparse,
    Custom,
}

/// Scheduling policies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingPolicy {
    FIFO,
    Priority,
    ShortestJobFirst,
    RoundRobin,
    FairShare,
    Adaptive,
}

/// Buffer metadata
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Creation timestamp
    pub created_at: Instant,

    /// Last access timestamp
    pub last_accessed: Instant,

    /// Access count
    pub access_count: usize,

    /// Data type
    pub data_type: DataType,

    /// Buffer flags
    pub flags: BufferFlags,
}

/// Buffer flags
#[derive(Debug, Clone)]
pub struct BufferFlags {
    /// Read-only buffer
    pub read_only: bool,

    /// Persistent buffer
    pub persistent: bool,

    /// Prefetch hint
    pub prefetch: bool,

    /// Memory pinned
    pub pinned: bool,
}

/// Output specification
#[derive(Debug, Clone)]
pub struct OutputSpec<T: Float + Debug + Send + Sync + 'static> {
    /// Expected shape
    pub shape: Vec<usize>,

    /// Data type
    pub data_type: DataType,

    /// Memory layout
    pub layout: MemoryLayout,

    /// Phantom data
    _phantom: std::marker::PhantomData<T>,
}

/// Compiled program for TPU execution
#[derive(Debug, Clone)]
pub struct CompiledProgram {
    /// Program binary
    pub binary: Vec<u8>,

    /// Program metadata
    pub metadata: ProgramMetadata,

    /// Memory requirements
    pub memory_requirements: ProgramMemoryRequirements,

    /// Performance characteristics
    pub performance_characteristics: ProgramPerformanceCharacteristics,
}

/// Program metadata
#[derive(Debug, Clone)]
pub struct ProgramMetadata {
    /// Compilation timestamp
    pub compiled_at: Instant,

    /// Compiler version
    pub compiler_version: String,

    /// Optimization level
    pub optimization_level: XLAOptimizationLevel,

    /// Target architecture
    pub target_architecture: TPUVersion,

    /// Program size
    pub program_size: usize,

    /// Output specifications
    pub output_specs: Vec<String>,
}

/// Program memory requirements
#[derive(Debug, Clone)]
pub struct ProgramMemoryRequirements {
    /// Code memory
    pub code_memory: usize,

    /// Data memory
    pub data_memory: usize,

    /// Stack memory
    pub stack_memory: usize,

    /// Scratch memory
    pub scratch_memory: usize,

    /// Total memory
    pub total_memory: usize,
}

/// Program performance characteristics
#[derive(Debug, Clone)]
pub struct ProgramPerformanceCharacteristics {
    /// Estimated execution time. `Duration::ZERO` when produced by
    /// [`super::backend::TPUBackend`]'s compiler: that path only ever sees a
    /// bare [`ComputationId`], with no op list or tensor shapes to time, so
    /// zero honestly means "not measured" rather than a fabricated guess.
    pub estimated_execution_time: Duration,

    /// Estimated FLOPS. `0` for the same reason as
    /// [`Self::estimated_execution_time`] -- honestly absent, not a
    /// placeholder round number.
    pub estimated_flops: u64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,

    /// Compute utilization
    pub compute_utilization: f64,
}

/// Prefetch strategies
#[derive(Debug, Clone, Copy)]
pub enum PrefetchStrategy {
    None,
    Sequential,
    Adaptive,
    Predictive,
    UserHint,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryAllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    BuddySystem,
    PoolBased,
    Adaptive,
}

/// Memory allocation tracking
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Per-device allocations
    pub device_allocations: HashMap<DeviceId, usize>,

    /// Total allocated memory
    pub total_allocated: usize,
}

/// Computation task for execution
#[derive(Debug, Clone)]
pub struct ComputationTask {
    /// Task identifier
    pub task_id: TaskId,

    /// Computation to execute
    pub computation_id: ComputationId,

    /// Input data
    pub input_data: Vec<u8>,

    /// Expected output specifications
    pub expected_outputs: Vec<String>,
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskExecutionResult {
    /// Task identifier
    pub task_id: TaskId,

    /// Execution time
    pub execution_time: std::time::Duration,

    /// Memory used during execution
    pub memory_used: usize,

    /// Energy consumed
    pub energy_consumed: f64,

    /// Output data
    pub output_data: Vec<u8>,
}

/// Device topology information
#[derive(Debug, Clone, Default)]
pub struct DeviceTopology {
    /// Device connections
    pub connections: HashMap<DeviceId, Vec<DeviceId>>,

    /// Bandwidth matrix
    pub bandwidth_matrix: HashMap<(DeviceId, DeviceId), f64>,
}

/// Load balancer for distributing tasks
#[derive(Debug, Clone, Default)]
pub struct LoadBalancer {
    /// Current load per device
    pub device_loads: HashMap<DeviceId, f64>,

    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
}

/// Task executor
#[derive(Debug, Clone, Default)]
pub struct TaskExecutor {
    /// Execution threads
    pub thread_count: usize,

    /// Task queue capacity
    pub queue_capacity: usize,
}

/// Backend performance statistics
#[derive(Debug, Clone)]
pub struct BackendPerformanceStatistics {
    pub total_executions: usize,
    pub average_execution_time: Duration,
    pub device_utilization: HashMap<DeviceId, f64>,
    pub memory_utilization: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
}

/// Memory block descriptor
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block start address
    pub start_address: usize,

    /// Block size
    pub size: usize,

    /// Allocation timestamp
    pub allocated_at: Instant,

    /// Last access timestamp
    pub last_accessed: Instant,

    /// Access count
    pub access_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStatistics {
    /// Total allocated memory
    pub total_allocated: usize,

    /// Peak memory usage
    pub peak_usage: usize,

    /// Average allocation size
    pub average_allocation_size: usize,

    /// Fragmentation ratio
    pub fragmentation_ratio: f64,

    /// Allocation success rate
    pub allocation_success_rate: f64,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Copy)]
pub enum GCStrategy {
    MarkAndSweep,
    Generational,
    Reference,
    LeastRecentlyUsed,
    Adaptive,
}

/// Garbage collection statistics
#[derive(Debug, Clone)]
pub struct GCStatistics {
    /// Total collections
    pub total_collections: usize,

    /// Total memory reclaimed
    pub total_memory_reclaimed: usize,

    /// Average collection time
    pub average_collection_time: Duration,

    /// Collection efficiency
    pub collection_efficiency: f64,
}

/// Profile sample
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// Timestamp
    pub timestamp: Instant,

    /// CPU utilization
    pub cpu_utilization: f64,

    /// Memory utilization
    pub memory_utilization: f64,

    /// Device utilization
    pub device_utilization: HashMap<DeviceId, f64>,

    /// Active tasks
    pub active_tasks: usize,

    /// Queue length
    pub queue_length: usize,
}

/// Error statistics
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Total errors
    pub total_errors: usize,

    /// Error rate
    pub error_rate: f64,

    /// Errors by type
    pub errors_by_type: HashMap<ErrorType, usize>,

    /// Recovery success rate
    pub recovery_success_rate: f64,
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate: 0.0,
            errors_by_type: HashMap::new(),
            recovery_success_rate: 0.0,
        }
    }
}

/// Error types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
    DeviceError,
    MemoryError,
    ComputationError,
    CommunicationError,
    TimeoutError,
    ResourceError,
}

/// Recovery strategies
#[derive(Debug, Clone, Copy)]
pub enum RecoveryStrategy {
    Retry,
    Fallback,
    Restart,
    Migrate,
    Ignore,
}

/// Performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Timestamp
    pub timestamp: Instant,

    /// Execution time
    pub execution_time: Duration,

    /// Throughput
    pub throughput: f64,

    /// Device utilization
    pub device_utilization: f64,

    /// Memory utilization
    pub memory_utilization: f64,
}
