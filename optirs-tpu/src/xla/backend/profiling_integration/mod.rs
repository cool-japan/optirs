// Performance profiling integration for XLA executables
//
// This module provides comprehensive profiling capabilities for XLA executables
// running on TPU hardware, including performance counters, trace collection,
// memory profiling, and power consumption tracking.
//
// Timeline profiling (`TimelineProfiler` and its supporting types) lives in
// the `timeline` submodule, split out purely to keep this file under the
// 2000-line convention -- its struct definitions and `impl` were moved
// together as a self-contained unit, so no field visibility changed.

mod timeline;
pub use timeline::*;

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use super::super::frontend::XLAComputation;
use super::BackendConfig;
use crate::error::{OptimError, Result};

/// Profiling integration manager
pub struct ProfilingIntegration<T> {
    /// Profiling configuration
    config: ProfilingConfig,

    /// Performance counter manager
    counter_manager: PerformanceCounterManager,

    /// Trace collector
    trace_collector: TraceCollector,

    /// Memory profiler
    memory_profiler: MemoryProfiler,

    /// Timeline profiler
    timeline_profiler: TimelineProfiler<T>,

    /// Export manager
    export_manager: ProfileExportManager,

    /// Profiling statistics
    profiling_stats: ProfilingStatistics,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable performance counter collection
    pub enable_perf_counters: bool,

    /// Enable trace collection
    pub enable_trace_collection: bool,

    /// Enable memory profiling
    pub enable_memory_profiling: bool,

    /// Enable power profiling
    pub enable_power_profiling: bool,

    /// Enable timeline profiling
    pub enable_timeline_profiling: bool,

    /// Sampling rate (Hz)
    pub sampling_rate: u64,

    /// Maximum trace buffer size (MB)
    pub max_trace_buffer_mb: usize,

    /// Profile output directory
    pub output_directory: String,

    /// Export format
    pub export_format: ExportFormat,

    /// Detailed profiling mode
    pub detailed_mode: bool,
}

/// Export formats for profiling data
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// JSON format
    JSON,

    /// Protocol buffers
    ProtoBuf,

    /// Chrome trace format
    ChromeTrace,

    /// CSV format
    CSV,

    /// Binary format
    Binary,
}

/// Profiling statistics
#[derive(Debug, Default)]
pub struct ProfilingStatistics {
    /// Total samples collected
    pub samples_collected: u64,

    /// Trace events captured
    pub trace_events: u64,

    /// Memory snapshots taken
    pub memory_snapshots: u64,

    /// Power samples collected
    pub power_samples: u64,

    /// Profiling overhead (microseconds)
    pub profiling_overhead_us: u64,

    /// Data export time (microseconds)
    pub export_time_us: u64,
}

/// Performance counter manager
pub struct PerformanceCounterManager {
    /// Available counters
    available_counters: HashMap<String, CounterInfo>,

    /// Active counter sessions
    active_sessions: HashMap<String, CounterSession>,

    /// Counter data storage
    counter_data: Arc<RwLock<HashMap<String, CounterTimeSeries>>>,
}

/// Performance counter information
#[derive(Debug, Clone)]
pub struct CounterInfo {
    /// Counter name
    pub name: String,

    /// Counter description
    pub description: String,

    /// Counter type
    pub counter_type: CounterType,

    /// Units of measurement
    pub units: String,

    /// Sampling granularity
    pub granularity: CounterGranularity,

    /// Hardware dependency
    pub hardware_dependency: Option<String>,
}

/// Types of performance counters
#[derive(Debug, Clone)]
pub enum CounterType {
    /// Cumulative counter (always increasing)
    Cumulative,

    /// Gauge counter (point-in-time value)
    Gauge,

    /// Rate counter (per-second rate)
    Rate,

    /// Histogram counter
    Histogram,
}

/// Counter granularity levels
#[derive(Debug, Clone)]
pub enum CounterGranularity {
    /// Per-instruction granularity
    Instruction,

    /// Per-operation granularity
    Operation,

    /// Per-kernel granularity
    Kernel,

    /// Per-execution granularity
    Execution,

    /// System-wide granularity
    System,
}

/// Counter session for tracking active profiling
#[derive(Debug)]
pub struct CounterSession {
    /// Session ID
    pub id: String,

    /// Session start time
    pub start_time: Instant,

    /// Enabled counters
    pub enabled_counters: Vec<String>,

    /// Sample buffer
    pub sample_buffer: VecDeque<CounterSample>,

    /// Session configuration
    pub config: SessionConfig,
}

/// Counter sample
#[derive(Debug, Clone)]
pub struct CounterSample {
    /// Sample timestamp
    pub timestamp: Instant,

    /// Counter name
    pub counter_name: String,

    /// Sample value
    pub value: CounterValue,

    /// Associated context
    pub context: Option<String>,
}

/// Counter value types
#[derive(Debug, Clone)]
pub enum CounterValue {
    /// Integer value
    Integer(i64),

    /// Floating point value
    Float(f64),

    /// Boolean value
    Boolean(bool),

    /// String value
    String(String),

    /// Histogram value
    Histogram(Vec<(f64, u64)>),
}

/// Time series data for counters
#[derive(Debug)]
pub struct CounterTimeSeries {
    /// Counter name
    pub counter_name: String,

    /// Time series samples
    pub samples: Vec<(Instant, CounterValue)>,

    /// Aggregate statistics
    pub statistics: TimeSeriesStats,
}

/// Time series statistics
#[derive(Debug, Default)]
pub struct TimeSeriesStats {
    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Average value
    pub average: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Sample count
    pub sample_count: usize,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Sampling interval (microseconds)
    pub sampling_interval_us: u64,

    /// Buffer size (samples)
    pub buffer_size: usize,

    /// Auto-flush threshold
    pub auto_flush_threshold: usize,

    /// Include context information
    pub include_context: bool,
}

/// Counter configuration
#[derive(Debug)]
pub struct CounterConfig {
    /// Default sampling rate
    pub default_sampling_rate: u64,

    /// Counter groups
    pub counter_groups: HashMap<String, Vec<String>>,

    /// Counter aliases
    pub aliases: HashMap<String, String>,
}

/// Trace collector for execution traces
pub struct TraceCollector {
    /// Trace buffer
    trace_buffer: Arc<Mutex<TraceBuffer>>,

    /// Trace sessions
    trace_sessions: HashMap<String, TraceSession>,
}

/// Trace buffer for storing events
#[derive(Debug)]
pub struct TraceBuffer {
    /// Events in the buffer
    pub events: VecDeque<TraceEvent>,

    /// Maximum buffer size
    pub max_size: usize,

    /// Current buffer size (bytes)
    pub current_size: usize,

    /// Buffer statistics
    pub stats: BufferStats,
}

/// Trace event
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Event ID
    pub id: u64,

    /// Event timestamp
    pub timestamp: Instant,

    /// Event type
    pub event_type: EventType,

    /// Event phase
    pub phase: EventPhase,

    /// Associated thread ID
    pub thread_id: Option<u64>,

    /// Associated process ID
    pub process_id: Option<u64>,

    /// Event name
    pub name: String,

    /// Event category
    pub category: String,

    /// Event duration (for duration events)
    pub duration: Option<Duration>,

    /// Event arguments
    pub args: HashMap<String, String>,

    /// Stack trace
    pub stack_trace: Option<Vec<String>>,
}

/// Types of trace events
#[derive(Debug, Clone)]
pub enum EventType {
    /// Function call
    FunctionCall,

    /// Kernel execution
    KernelExecution,

    /// Memory operation
    MemoryOperation,

    /// Communication operation
    Communication,

    /// Synchronization
    Synchronization,

    /// Resource allocation
    ResourceAllocation,

    /// Custom event
    Custom(String),
}

/// Event phases
#[derive(Debug, Clone)]
pub enum EventPhase {
    /// Begin phase
    Begin,

    /// End phase
    End,

    /// Instant event
    Instant,

    /// Complete event (begin + end)
    Complete,

    /// Async begin
    AsyncBegin,

    /// Async end
    AsyncEnd,
}

/// Trace session
#[derive(Debug)]
pub struct TraceSession {
    /// Session ID
    pub id: String,

    /// Session start time
    pub start_time: Instant,

    /// Enabled event types
    pub enabled_events: Vec<EventType>,

    /// Session buffer
    pub session_buffer: Vec<TraceEvent>,

    /// Session metadata
    pub metadata: TraceMetadata,
}

/// Trace metadata
#[derive(Debug, Default)]
pub struct TraceMetadata {
    /// Session name
    pub session_name: String,

    /// Target executable
    pub target_executable: Option<String>,

    /// Hardware information
    pub hardware_info: HashMap<String, String>,

    /// Software information
    pub software_info: HashMap<String, String>,
}

/// Event filter for trace collection
#[derive(Debug)]
pub struct EventFilter {
    /// Filter name
    pub name: String,

    /// Event type filter
    pub event_type_filter: Option<EventType>,

    /// Category filter
    pub category_filter: Option<String>,

    /// Duration threshold (minimum)
    pub duration_threshold: Option<Duration>,

    /// Include/exclude flag
    pub include: bool,
}

/// Buffer statistics
#[derive(Debug, Default)]
pub struct BufferStats {
    /// Events written
    pub events_written: u64,

    /// Events dropped
    pub events_dropped: u64,

    /// Buffer overruns
    pub overruns: u64,

    /// Peak buffer usage
    pub peak_usage: usize,
}

/// Trace configuration
#[derive(Debug)]
pub struct TraceConfig {
    /// Buffer size (events)
    pub buffer_size: usize,

    /// Include stack traces
    pub include_stack_traces: bool,

    /// Maximum stack trace depth
    pub max_stack_depth: usize,

    /// Event compression
    pub enable_compression: bool,
}

/// Memory profiler
pub struct MemoryProfiler {
    /// Memory tracking sessions
    tracking_sessions: HashMap<String, MemoryTrackingSession>,

    /// Allocation tracker
    allocation_tracker: AllocationTracker,

    /// Memory usage snapshots
    usage_snapshots: Vec<MemorySnapshot>,
}

/// Memory tracking session
#[derive(Debug)]
pub struct MemoryTrackingSession {
    /// Session ID
    pub id: String,

    /// Session start time
    pub start_time: Instant,

    /// Tracked allocations
    pub allocations: HashMap<usize, AllocationInfo>,

    /// Memory statistics
    pub stats: MemoryTrackingStats,
}

/// Allocation information
#[derive(Debug)]
pub struct AllocationInfo {
    /// Allocation address
    pub address: usize,

    /// Allocation size
    pub size: usize,

    /// Allocation timestamp
    pub timestamp: Instant,

    /// Allocation source
    pub source: AllocationSource,

    /// Stack trace at allocation
    pub stack_trace: Option<Vec<String>>,

    /// Allocation tags
    pub tags: Vec<String>,
}

/// Allocation sources
#[derive(Debug)]
pub enum AllocationSource {
    /// Kernel execution
    Kernel(String),

    /// Runtime system
    Runtime,

    /// User code
    User,

    /// Unknown source
    Unknown,
}

/// Memory tracking statistics
#[derive(Debug, Default)]
pub struct MemoryTrackingStats {
    /// Total allocations
    pub total_allocations: usize,

    /// Total deallocations
    pub total_deallocations: usize,

    /// Current allocation count
    pub current_allocations: usize,

    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,

    /// Current memory usage (bytes)
    pub current_memory_usage: usize,

    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Allocation tracker
pub struct AllocationTracker {
    /// Active allocations
    active_allocations: HashMap<usize, AllocationInfo>,

    /// Allocation history
    allocation_history: Vec<AllocationEvent>,
}

/// Allocation event
#[derive(Debug)]
pub struct AllocationEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Event type
    pub event_type: AllocationEventType,

    /// Allocation address
    pub address: usize,

    /// Allocation size
    pub size: usize,

    /// Associated context
    pub context: Option<String>,
}

/// Types of allocation events
#[derive(Debug)]
pub enum AllocationEventType {
    /// Memory allocation
    Allocate,

    /// Memory deallocation
    Deallocate,

    /// Memory reallocation
    Reallocate,
}

/// Tracker configuration
#[derive(Debug)]
pub struct TrackerConfig {
    /// Track stack traces
    pub track_stack_traces: bool,

    /// Maximum history size
    pub max_history_size: usize,

    /// Enable leak detection
    pub enable_leak_detection: bool,
}

/// Memory snapshot
#[derive(Debug)]
pub struct MemorySnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,

    /// Memory regions
    pub regions: Vec<MemoryRegion>,

    /// Total memory usage
    pub total_usage: usize,

    /// Fragmentation information
    pub fragmentation: FragmentationInfo,
}

/// Memory region information
#[derive(Debug)]
pub struct MemoryRegion {
    /// Region start address
    pub start_address: usize,

    /// Region size
    pub size: usize,

    /// Region type
    pub region_type: MemoryRegionType,

    /// Usage information
    pub usage: RegionUsage,
}

/// Memory region types
#[derive(Debug)]
pub enum MemoryRegionType {
    /// Code region
    Code,

    /// Data region
    Data,

    /// Stack region
    Stack,

    /// Heap region
    Heap,

    /// Device memory region
    Device,
}

/// Region usage information
#[derive(Debug)]
pub struct RegionUsage {
    /// Used bytes
    pub used_bytes: usize,

    /// Free bytes
    pub free_bytes: usize,

    /// Fragmentation level
    pub fragmentation: f64,
}

/// Fragmentation information
#[derive(Debug, Default)]
pub struct FragmentationInfo {
    /// External fragmentation
    pub external_fragmentation: f64,

    /// Internal fragmentation
    pub internal_fragmentation: f64,

    /// Largest free block
    pub largest_free_block: usize,

    /// Free block count
    pub free_block_count: usize,
}

/// Memory profiling configuration
#[derive(Debug)]
pub struct MemoryProfilingConfig {
    /// Snapshot interval (milliseconds)
    pub snapshot_interval_ms: u64,

    /// Track individual allocations
    pub track_allocations: bool,

    /// Maximum snapshots to keep
    pub max_snapshots: usize,

    /// Enable heap profiling
    pub enable_heap_profiling: bool,
}

/// Power monitoring session
#[derive(Debug)]
pub struct PowerMonitoringSession {
    /// Session ID
    pub id: String,

    /// Session start time
    pub start_time: Instant,

    /// Monitored components
    pub components: Vec<PowerComponent>,

    /// Session samples
    pub samples: Vec<PowerSample>,
}

/// Power component
#[derive(Debug, Clone)]
pub enum PowerComponent {
    /// CPU power
    CPU,

    /// TPU power
    TPU,

    /// Memory power
    Memory,

    /// Interconnect power
    Interconnect,

    /// Total system power
    System,
}

/// Power sample
#[derive(Debug, Clone)]
pub struct PowerSample {
    /// Sample timestamp
    pub timestamp: Instant,

    /// Component
    pub component: PowerComponent,

    /// Power consumption (watts)
    pub power_watts: f64,

    /// Voltage (volts)
    pub voltage: Option<f64>,

    /// Current (amperes)
    pub current: Option<f64>,

    /// Temperature (celsius)
    pub temperature: Option<f64>,
}

/// Power profiling configuration
#[derive(Debug)]
pub struct PowerProfilingConfig {
    /// Sampling rate (Hz)
    pub sampling_rate: u64,

    /// Enable component-level monitoring
    pub component_level_monitoring: bool,

    /// Include thermal information
    pub include_thermal: bool,

    /// Power model accuracy
    pub model_accuracy: PowerModelAccuracy,
}

/// Power model accuracy levels
#[derive(Debug)]
pub enum PowerModelAccuracy {
    /// Low accuracy (fast)
    Low,

    /// Medium accuracy
    Medium,

    /// High accuracy (detailed)
    High,
}

/// Component power model
#[derive(Debug)]
pub struct ComponentPowerModel {
    /// Base power consumption
    pub base_power: f64,

    /// Dynamic power factors
    pub dynamic_factors: HashMap<String, f64>,

    /// Thermal coefficients
    pub thermal_coefficients: Vec<f64>,
}

/// Aggregated metrics
#[derive(Debug, Default)]
pub struct AggregatedMetrics {
    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Memory metrics
    pub memory: MemoryMetrics,

    /// Power metrics
    pub power: PowerMetrics,

    /// Timeline metrics
    pub timeline: TimelineMetrics,
}

/// Performance metrics
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// Average execution time
    pub avg_execution_time_us: f64,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f64,
}

/// Memory metrics
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_usage_bytes: usize,

    /// Average memory usage
    pub avg_usage_bytes: f64,

    /// Memory efficiency
    pub efficiency: f64,

    /// Allocation rate
    pub allocation_rate: f64,
}

/// Power metrics
#[derive(Debug, Default)]
pub struct PowerMetrics {
    /// Average power consumption
    pub avg_power_watts: f64,

    /// Peak power consumption
    pub peak_power_watts: f64,

    /// Energy consumed
    pub energy_joules: f64,

    /// Power efficiency
    pub efficiency: f64,
}

/// Timeline metrics
#[derive(Debug, Default)]
pub struct TimelineMetrics {
    /// Total operations
    pub total_operations: usize,

    /// Average operation duration
    pub avg_operation_duration_us: f64,

    /// Critical path length
    pub critical_path_length_us: u64,

    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
}

/// Aggregation configuration
#[derive(Debug)]
pub struct AggregationConfig {
    /// Aggregation interval (seconds)
    pub interval_seconds: u64,

    /// Enable real-time aggregation
    pub real_time: bool,

    /// Retention period (hours)
    pub retention_hours: u32,
}

/// Profile export manager
pub struct ProfileExportManager {
    /// Export configuration
    export_config: ExportConfig,

    /// Export statistics
    export_stats: ExportStatistics,
}

/// Export configuration
#[derive(Debug)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,

    /// Output directory
    pub output_dir: String,

    /// Include raw data
    pub include_raw_data: bool,

    /// Compression enabled
    pub compression: bool,

    /// Export metadata
    pub include_metadata: bool,
}

/// Export statistics
#[derive(Debug, Default)]
pub struct ExportStatistics {
    /// Files exported
    pub files_exported: usize,

    /// Total export size (bytes)
    pub total_size_bytes: usize,

    /// Export time (microseconds)
    pub export_time_us: u64,

    /// Compression ratio
    pub compression_ratio: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> ProfilingIntegration<T> {
    /// Create new profiling integration
    pub fn new(config: &BackendConfig) -> Self {
        let profiling_config = ProfilingConfig {
            enable_perf_counters: config.enable_profiling,
            enable_trace_collection: config.enable_profiling,
            enable_memory_profiling: config.enable_profiling,
            enable_power_profiling: false,
            enable_timeline_profiling: config.enable_profiling,
            sampling_rate: 1000, // 1kHz
            max_trace_buffer_mb: 100,
            output_directory: "/tmp/scirs_profiles".to_string(),
            export_format: ExportFormat::JSON,
            detailed_mode: config.debug_mode,
        };

        Self {
            counter_manager: PerformanceCounterManager::new(),
            trace_collector: TraceCollector::new(&profiling_config),
            memory_profiler: MemoryProfiler::new(&profiling_config),
            timeline_profiler: TimelineProfiler::new(&profiling_config),
            export_manager: ProfileExportManager::new(&profiling_config),
            config: profiling_config,
            profiling_stats: ProfilingStatistics::default(),
        }
    }

    /// Setup profiling for computation
    pub fn setup_profiling(
        &mut self,
        _computation: &XLAComputation<T>,
        _binary: &[u8],
    ) -> Result<()> {
        if self.config.enable_perf_counters {
            self.counter_manager.start_session("main_session")?;
        }

        if self.config.enable_trace_collection {
            self.trace_collector.start_tracing("main_trace")?;
        }

        if self.config.enable_memory_profiling {
            self.memory_profiler.start_tracking("main_memory")?;
        }

        if self.config.enable_timeline_profiling {
            self.timeline_profiler.start_timeline("main_timeline")?;
        }

        Ok(())
    }

    /// Record the real wall-clock timings measured for one compile step
    /// (code generation and runtime integration, both real
    /// [`Instant::elapsed`] measurements taken by
    /// [`super::XLABackend::compile_and_integrate`]) into the active
    /// profiling session.
    ///
    /// This is the measurement side that was entirely missing before:
    /// [`Self::setup_profiling`] only ever created empty sessions, and
    /// nothing fed samples into them, so [`Self::export_data`] always wrote
    /// hardcoded empty files regardless of how much real compilation work
    /// had happened. A no-op when the corresponding profiling category is
    /// disabled, matching every other method here.
    pub fn record_compile_timings(&mut self, codegen: Duration, runtime_integration: Duration) {
        if self.config.enable_perf_counters {
            self.counter_manager.record_sample(
                "main_session",
                "codegen_time_us",
                CounterValue::Integer(codegen.as_micros() as i64),
            );
            self.counter_manager.record_sample(
                "main_session",
                "runtime_integration_time_us",
                CounterValue::Integer(runtime_integration.as_micros() as i64),
            );
            self.profiling_stats.samples_collected += 2;
        }

        if self.config.enable_trace_collection {
            self.trace_collector.record_event(TraceEvent {
                id: 0, // overwritten by `record_event`
                timestamp: Instant::now(),
                event_type: EventType::FunctionCall,
                phase: EventPhase::Complete,
                thread_id: None,
                process_id: None,
                name: "codegen".to_string(),
                category: "compile".to_string(),
                duration: Some(codegen),
                args: HashMap::new(),
                stack_trace: None,
            });
            self.trace_collector.record_event(TraceEvent {
                id: 0,
                timestamp: Instant::now(),
                event_type: EventType::FunctionCall,
                phase: EventPhase::Complete,
                thread_id: None,
                process_id: None,
                name: "runtime_integration".to_string(),
                category: "compile".to_string(),
                duration: Some(runtime_integration),
                args: HashMap::new(),
                stack_trace: None,
            });
            self.profiling_stats.trace_events += 2;
        }
    }

    /// Export profiling data
    pub fn export_data(&mut self) -> Result<Vec<String>> {
        let mut exported_files = Vec::new();

        // Export performance counter data
        if self.config.enable_perf_counters {
            let file_path = self
                .export_manager
                .export_counter_data(&self.counter_manager)?;
            exported_files.push(file_path);
        }

        // Export trace data
        if self.config.enable_trace_collection {
            let file_path = self
                .export_manager
                .export_trace_data(&self.trace_collector)?;
            exported_files.push(file_path);
        }

        // Export memory data
        if self.config.enable_memory_profiling {
            let file_path = self
                .export_manager
                .export_memory_data(&self.memory_profiler)?;
            exported_files.push(file_path);
        }

        Ok(exported_files)
    }

    /// Record a device-memory reservation made while running a compiled
    /// program, so the exported memory profile reflects real allocator
    /// activity. A no-op when memory profiling is disabled.
    pub fn record_memory_allocation(
        &mut self,
        session_id: &str,
        address: usize,
        size: usize,
        context: Option<String>,
    ) {
        if !self.config.enable_memory_profiling {
            return;
        }
        self.memory_profiler.record_allocation(
            session_id,
            address,
            size,
            AllocationSource::Runtime,
            context,
        );
    }

    /// Capture a usage snapshot of the reservations currently live, carrying
    /// the allocator's real fragmentation. A no-op when memory profiling is
    /// disabled.
    ///
    /// Call this *while* the memory is held: a snapshot taken after release
    /// records an empty live set, so a series built only from post-release
    /// snapshots is structurally empty regardless of real activity.
    pub fn capture_memory_snapshot(&mut self, fragmentation: FragmentationInfo) {
        if !self.config.enable_memory_profiling {
            return;
        }
        self.memory_profiler.capture_snapshot(fragmentation);
    }

    /// Companion to [`Self::record_memory_allocation`] for the release side,
    /// followed by a usage snapshot carrying the allocator's real
    /// fragmentation. A no-op when memory profiling is disabled.
    pub fn record_memory_release(
        &mut self,
        session_id: &str,
        addresses: &[usize],
        fragmentation: FragmentationInfo,
        context: Option<String>,
    ) {
        if !self.config.enable_memory_profiling {
            return;
        }
        for &address in addresses {
            self.memory_profiler
                .record_deallocation(session_id, address, context.clone());
        }
        self.memory_profiler.capture_snapshot(fragmentation);
    }

    /// The memory profiler backing [`Self::record_memory_allocation`].
    pub fn memory_profiler(&self) -> &MemoryProfiler {
        &self.memory_profiler
    }

    /// Reset profiling state
    pub fn reset(&mut self) {
        self.profiling_stats = ProfilingStatistics::default();
        self.counter_manager.reset();
        self.trace_collector.reset();
        self.memory_profiler.reset();
        self.timeline_profiler.reset();
    }
}

impl Default for PerformanceCounterManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceCounterManager {
    /// Create new performance counter manager
    pub fn new() -> Self {
        let mut available_counters = HashMap::new();

        // Add common TPU performance counters
        available_counters.insert(
            "matrix_ops".to_string(),
            CounterInfo {
                name: "matrix_ops".to_string(),
                description: "Matrix operations executed".to_string(),
                counter_type: CounterType::Cumulative,
                units: "operations".to_string(),
                granularity: CounterGranularity::Operation,
                hardware_dependency: Some("matrix_unit".to_string()),
            },
        );

        available_counters.insert(
            "memory_bandwidth".to_string(),
            CounterInfo {
                name: "memory_bandwidth".to_string(),
                description: "Memory bandwidth utilization".to_string(),
                counter_type: CounterType::Gauge,
                units: "GB/s".to_string(),
                granularity: CounterGranularity::System,
                hardware_dependency: Some("memory_controller".to_string()),
            },
        );

        Self {
            available_counters,
            active_sessions: HashMap::new(),
            counter_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start counter session
    pub fn start_session(&mut self, session_id: &str) -> Result<()> {
        let session = CounterSession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            enabled_counters: self.available_counters.keys().cloned().collect(),
            sample_buffer: VecDeque::new(),
            config: SessionConfig {
                sampling_interval_us: 1000, // 1ms
                buffer_size: 10000,
                auto_flush_threshold: 8000,
                include_context: true,
            },
        };

        self.active_sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Record one real counter sample into `session_id`'s sample buffer and
    /// into the shared, exportable `counter_data` time series.
    ///
    /// Creates the session on first use if it does not already exist,
    /// rather than erroring: a caller recording a real measurement should
    /// not have to separately remember to call [`Self::start_session`]
    /// first, and this keeps the manager honest either way -- it only ever
    /// reports what was actually recorded through this method, never a
    /// canned response.
    pub fn record_sample(&mut self, session_id: &str, counter_name: &str, value: CounterValue) {
        let now = Instant::now();

        self.active_sessions
            .entry(session_id.to_string())
            .or_insert_with(|| CounterSession {
                id: session_id.to_string(),
                start_time: now,
                enabled_counters: Vec::new(),
                sample_buffer: VecDeque::new(),
                config: SessionConfig {
                    sampling_interval_us: 1000,
                    buffer_size: 10000,
                    auto_flush_threshold: 8000,
                    include_context: true,
                },
            })
            .sample_buffer
            .push_back(CounterSample {
                timestamp: now,
                counter_name: counter_name.to_string(),
                value: value.clone(),
                context: None,
            });

        // A poisoned lock only means some thread panicked while holding it;
        // the counter data is still safe to update, so recover the guard
        // instead of panicking (matches `reset`'s existing recovery policy).
        let mut data = self
            .counter_data
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let series = data
            .entry(counter_name.to_string())
            .or_insert_with(|| CounterTimeSeries {
                counter_name: counter_name.to_string(),
                samples: Vec::new(),
                statistics: TimeSeriesStats::default(),
            });
        series.samples.push((now, value));
        recompute_time_series_statistics(series);
    }

    /// Reset counter manager
    pub fn reset(&mut self) {
        self.active_sessions.clear();
        // A poisoned lock only means some thread panicked while holding it; the
        // counter data is still safe to clear, so recover the guard instead of
        // panicking.
        let mut data = self
            .counter_data
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        data.clear();
    }
}

/// Recompute `series.statistics` from `series.samples`, over exactly the
/// numeric samples (`Integer`/`Float`); non-numeric samples
/// (`Boolean`/`String`/`Histogram`) contribute to `sample_count` but are not
/// folded into min/max/average/std_dev, since there is no single real number
/// to average them into.
///
/// This is genuine descriptive statistics computed from whatever was
/// actually recorded -- never a fabricated round number -- so it is empty
/// (all zero, matching [`TimeSeriesStats::default`]) exactly when nothing
/// numeric has been recorded yet.
fn recompute_time_series_statistics(series: &mut CounterTimeSeries) {
    let values: Vec<f64> = series
        .samples
        .iter()
        .filter_map(|(_, v)| match v {
            CounterValue::Integer(i) => Some(*i as f64),
            CounterValue::Float(f) => Some(*f),
            CounterValue::Boolean(_) | CounterValue::String(_) | CounterValue::Histogram(_) => None,
        })
        .collect();

    series.statistics.sample_count = series.samples.len();

    if values.is_empty() {
        series.statistics.min = 0.0;
        series.statistics.max = 0.0;
        series.statistics.average = 0.0;
        series.statistics.std_dev = 0.0;
        return;
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().sum();
    let average = sum / values.len() as f64;
    let variance = values.iter().map(|v| (v - average).powi(2)).sum::<f64>() / values.len() as f64;

    series.statistics.min = min;
    series.statistics.max = max;
    series.statistics.average = average;
    series.statistics.std_dev = variance.sqrt();
}

impl TraceCollector {
    /// Create new trace collector
    pub fn new(config: &ProfilingConfig) -> Self {
        Self {
            trace_buffer: Arc::new(Mutex::new(TraceBuffer {
                events: VecDeque::new(),
                max_size: config.max_trace_buffer_mb * 1024 * 1024,
                current_size: 0,
                stats: BufferStats::default(),
            })),
            trace_sessions: HashMap::new(),
        }
    }

    /// Start tracing session
    pub fn start_tracing(&mut self, session_id: &str) -> Result<()> {
        let session = TraceSession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            enabled_events: vec![EventType::KernelExecution, EventType::MemoryOperation],
            session_buffer: Vec::new(),
            metadata: TraceMetadata::default(),
        };

        self.trace_sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Record one real trace event into the shared, exportable trace
    /// buffer, honestly respecting `max_size` (dropping the oldest event
    /// and counting a real overrun rather than growing the buffer without
    /// bound) and updating [`BufferStats`] to match.
    ///
    /// `event.id` is overwritten with the buffer's own running
    /// `events_written` counter, so callers do not need to allocate unique
    /// IDs themselves.
    pub fn record_event(&mut self, mut event: TraceEvent) {
        let mut buffer = self
            .trace_buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        event.id = buffer.stats.events_written;

        // A real (if rough) per-event byte estimate, used only to respect
        // `max_size` honestly rather than growing the in-memory buffer
        // without bound.
        let event_size =
            std::mem::size_of::<TraceEvent>() + event.name.len() + event.category.len();

        if buffer.current_size + event_size > buffer.max_size && !buffer.events.is_empty() {
            if let Some(dropped) = buffer.events.pop_front() {
                let dropped_size =
                    std::mem::size_of::<TraceEvent>() + dropped.name.len() + dropped.category.len();
                buffer.current_size = buffer.current_size.saturating_sub(dropped_size);
            }
            buffer.stats.events_dropped += 1;
            buffer.stats.overruns += 1;
        }

        buffer.current_size += event_size;
        buffer.stats.events_written += 1;
        buffer.stats.peak_usage = buffer.stats.peak_usage.max(buffer.current_size);
        buffer.events.push_back(event);
    }

    /// Reset trace collector
    pub fn reset(&mut self) {
        self.trace_sessions.clear();
        // Recover the guard on poisoning: clearing the trace buffer is safe
        // regardless of a prior panic in another thread.
        let mut buffer = self
            .trace_buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        buffer.events.clear();
        buffer.current_size = 0;
        buffer.stats = BufferStats::default();
    }
}

impl MemoryProfiler {
    /// Create new memory profiler
    pub fn new(_config: &ProfilingConfig) -> Self {
        Self {
            tracking_sessions: HashMap::new(),
            allocation_tracker: AllocationTracker::new(),
            usage_snapshots: Vec::new(),
        }
    }

    /// Start memory tracking
    pub fn start_tracking(&mut self, session_id: &str) -> Result<()> {
        let session = MemoryTrackingSession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            allocations: HashMap::new(),
            stats: MemoryTrackingStats::default(),
        };

        self.tracking_sessions
            .insert(session_id.to_string(), session);
        Ok(())
    }

    /// Record a real device-memory reservation.
    ///
    /// `address` and `size` are the allocator's own figures (see
    /// [`crate::tpu_backend::TPUMemoryManager`], which calls this as it
    /// reserves and releases blocks). The session is created on demand, so a
    /// caller that never called [`Self::start_tracking`] still gets tracked
    /// rather than silently dropped.
    pub fn record_allocation(
        &mut self,
        session_id: &str,
        address: usize,
        size: usize,
        source: AllocationSource,
        context: Option<String>,
    ) {
        let session = self.session_mut(session_id);
        session.allocations.insert(
            address,
            AllocationInfo {
                address,
                size,
                timestamp: Instant::now(),
                source,
                stack_trace: None,
                tags: Vec::new(),
            },
        );
        session.stats.total_allocations += 1;
        session.stats.current_allocations = session.allocations.len();
        session.stats.current_memory_usage =
            session.stats.current_memory_usage.saturating_add(size);
        session.stats.peak_memory_usage = session
            .stats
            .peak_memory_usage
            .max(session.stats.current_memory_usage);

        self.allocation_tracker.record(AllocationEvent {
            timestamp: Instant::now(),
            event_type: AllocationEventType::Allocate,
            address,
            size,
            context,
        });
    }

    /// Record the release of a block previously passed to
    /// [`Self::record_allocation`]. Releasing an address the profiler never saw
    /// is ignored rather than counted, so the statistics cannot go negative.
    pub fn record_deallocation(
        &mut self,
        session_id: &str,
        address: usize,
        context: Option<String>,
    ) {
        let session = self.session_mut(session_id);
        let Some(info) = session.allocations.remove(&address) else {
            return;
        };
        session.stats.total_deallocations += 1;
        session.stats.current_allocations = session.allocations.len();
        session.stats.current_memory_usage =
            session.stats.current_memory_usage.saturating_sub(info.size);

        self.allocation_tracker.record(AllocationEvent {
            timestamp: Instant::now(),
            event_type: AllocationEventType::Deallocate,
            address,
            size: info.size,
            context,
        });
    }

    /// Capture a usage snapshot of everything currently live.
    ///
    /// `fragmentation` is supplied by the allocator rather than inferred here:
    /// this profiler observes allocation events and has no view of the free
    /// list, so computing external fragmentation from the live set would be a
    /// guess dressed up as a measurement.
    pub fn capture_snapshot(&mut self, fragmentation: FragmentationInfo) {
        let regions: Vec<MemoryRegion> = self
            .allocation_tracker
            .active_allocations
            .values()
            .map(|info| MemoryRegion {
                start_address: info.address,
                size: info.size,
                region_type: MemoryRegionType::Data,
                usage: RegionUsage {
                    // A live reservation is fully used by definition: the
                    // region *is* the allocated block, so there is no free
                    // remainder inside it and nothing to fragment.
                    used_bytes: info.size,
                    free_bytes: 0,
                    fragmentation: 0.0,
                },
            })
            .collect();
        let total_usage = regions.iter().map(|region| region.size).sum();

        self.usage_snapshots.push(MemorySnapshot {
            timestamp: Instant::now(),
            regions,
            total_usage,
            fragmentation,
        });
    }

    /// Tracking statistics for `session_id`, if that session exists.
    pub fn tracking_stats(&self, session_id: &str) -> Option<&MemoryTrackingStats> {
        self.tracking_sessions
            .get(session_id)
            .map(|session| &session.stats)
    }

    /// Number of allocation events recorded so far.
    pub fn recorded_events(&self) -> usize {
        self.allocation_tracker.allocation_history.len()
    }

    /// Usage snapshots captured so far, oldest first.
    pub fn usage_snapshots(&self) -> &[MemorySnapshot] {
        &self.usage_snapshots
    }

    fn session_mut(&mut self, session_id: &str) -> &mut MemoryTrackingSession {
        self.tracking_sessions
            .entry(session_id.to_string())
            .or_insert_with(|| MemoryTrackingSession {
                id: session_id.to_string(),
                start_time: Instant::now(),
                allocations: HashMap::new(),
                stats: MemoryTrackingStats::default(),
            })
    }

    /// Reset memory profiler
    pub fn reset(&mut self) {
        self.tracking_sessions.clear();
        self.usage_snapshots.clear();
        self.allocation_tracker.reset();
    }
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationTracker {
    /// Create new allocation tracker
    pub fn new() -> Self {
        Self {
            active_allocations: HashMap::new(),
            allocation_history: Vec::new(),
        }
    }

    /// Append `event` to the history and keep the live set in step with it.
    fn record(&mut self, event: AllocationEvent) {
        match event.event_type {
            AllocationEventType::Allocate | AllocationEventType::Reallocate => {
                self.active_allocations.insert(
                    event.address,
                    AllocationInfo {
                        address: event.address,
                        size: event.size,
                        timestamp: event.timestamp,
                        source: AllocationSource::Runtime,
                        stack_trace: None,
                        tags: Vec::new(),
                    },
                );
            }
            AllocationEventType::Deallocate => {
                self.active_allocations.remove(&event.address);
            }
        }
        self.allocation_history.push(event);
    }

    /// Reset allocation tracker
    pub fn reset(&mut self) {
        self.active_allocations.clear();
        self.allocation_history.clear();
    }
}

impl ProfileExportManager {
    /// Create new export manager
    pub fn new(config: &ProfilingConfig) -> Self {
        Self {
            export_config: ExportConfig {
                format: config.export_format.clone(),
                output_dir: config.output_directory.clone(),
                include_raw_data: config.detailed_mode,
                compression: true,
                include_metadata: true,
            },
            export_stats: ExportStatistics::default(),
        }
    }

    /// Export counter data.
    ///
    /// Serializes whatever is *actually* in `counter_manager` -- real
    /// recorded samples plus real, computed-from-those-samples statistics
    /// -- rather than a hardcoded empty body. Genuinely reflects "nothing
    /// recorded yet" as an honest empty array when that is the real state,
    /// and reflects real data the moment
    /// [`PerformanceCounterManager::record_sample`] has been called.
    pub fn export_counter_data(
        &mut self,
        counter_manager: &PerformanceCounterManager,
    ) -> Result<String> {
        let export_start = Instant::now();
        let filename = format!(
            "{}/counters_{}_{}.json",
            self.export_config.output_dir,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| OptimError::from(e.to_string()))?
                .as_secs(),
            self.export_stats.files_exported
        );

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.export_config.output_dir)?;

        // A poisoned lock only means some thread panicked while holding it;
        // the counter data is still safe to read, so recover the guard
        // instead of panicking.
        let data = counter_manager
            .counter_data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let counters: Vec<serde_json::Value> = data
            .values()
            .map(|series| {
                let samples: Vec<serde_json::Value> = series
                    .samples
                    .iter()
                    .map(|(timestamp, value)| {
                        serde_json::json!({
                            "age_us": timestamp.elapsed().as_micros() as u64,
                            "value": counter_value_to_json(value),
                        })
                    })
                    .collect();

                serde_json::json!({
                    "name": series.counter_name,
                    "sample_count": series.statistics.sample_count,
                    "min": series.statistics.min,
                    "max": series.statistics.max,
                    "average": series.statistics.average,
                    "std_dev": series.statistics.std_dev,
                    "samples": samples,
                })
            })
            .collect();
        drop(data);

        let payload = serde_json::json!({
            "counters": counters,
            "metadata": {
                "available_counters": counter_manager.available_counters.len(),
                "active_sessions": counter_manager.active_sessions.len(),
            },
        });

        let mut file = File::create(&filename)?;
        writeln!(
            file,
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| OptimError::from(e.to_string()))?
        )?;

        self.export_stats.files_exported += 1;
        self.export_stats.export_time_us = export_start.elapsed().as_micros() as u64;
        Ok(filename)
    }

    /// Export trace data.
    ///
    /// Serializes whatever is *actually* in `trace_collector`'s buffer
    /// (real recorded events, in Chrome-trace-flavored JSON) instead of a
    /// hardcoded empty `traceEvents` array.
    pub fn export_trace_data(&mut self, trace_collector: &TraceCollector) -> Result<String> {
        let export_start = Instant::now();
        let filename = format!(
            "{}/trace_{}_{}.json",
            self.export_config.output_dir,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| OptimError::from(e.to_string()))?
                .as_secs(),
            self.export_stats.files_exported
        );

        std::fs::create_dir_all(&self.export_config.output_dir)?;

        // A poisoned lock only means some thread panicked while holding it;
        // the trace buffer is still safe to read, so recover the guard
        // instead of panicking.
        let buffer = trace_collector
            .trace_buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let trace_events: Vec<serde_json::Value> = buffer
            .events
            .iter()
            .map(|event| {
                serde_json::json!({
                    "id": event.id,
                    "name": event.name,
                    "cat": event.category,
                    "ph": event_phase_code(&event.phase),
                    "age_us": event.timestamp.elapsed().as_micros() as u64,
                    "dur_us": event.duration.map(|d| d.as_micros() as u64),
                    "tid": event.thread_id,
                    "pid": event.process_id,
                })
            })
            .collect();
        let events_written = buffer.stats.events_written;
        let events_dropped = buffer.stats.events_dropped;
        drop(buffer);

        let payload = serde_json::json!({
            "traceEvents": trace_events,
            "displayTimeUnit": "us",
            "metadata": {
                "events_written": events_written,
                "events_dropped": events_dropped,
            },
        });

        let mut file = File::create(&filename)?;
        writeln!(
            file,
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| OptimError::from(e.to_string()))?
        )?;

        self.export_stats.files_exported += 1;
        self.export_stats.export_time_us = export_start.elapsed().as_micros() as u64;
        Ok(filename)
    }

    /// Export memory data.
    ///
    /// Serializes whatever is *actually* in `memory_profiler`: real usage
    /// snapshots and real allocation-history events, never a hardcoded body.
    ///
    /// Those entries come from [`crate::tpu_backend::TPUBackend`], which
    /// reports every device-memory reservation and release it makes through
    /// [`ProfilingIntegration::record_memory_allocation`]/
    /// [`ProfilingIntegration::record_memory_release`] while running a
    /// compiled program. Compiling without executing therefore still exports
    /// honestly empty arrays -- genuinely absent data, not a fabrication.
    pub fn export_memory_data(&mut self, memory_profiler: &MemoryProfiler) -> Result<String> {
        let export_start = Instant::now();
        let filename = format!(
            "{}/memory_{}_{}.json",
            self.export_config.output_dir,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| OptimError::from(e.to_string()))?
                .as_secs(),
            self.export_stats.files_exported
        );

        std::fs::create_dir_all(&self.export_config.output_dir)?;

        let snapshots: Vec<serde_json::Value> = memory_profiler
            .usage_snapshots
            .iter()
            .map(|snapshot| {
                serde_json::json!({
                    "age_us": snapshot.timestamp.elapsed().as_micros() as u64,
                    "total_usage": snapshot.total_usage,
                    "region_count": snapshot.regions.len(),
                    "external_fragmentation": snapshot.fragmentation.external_fragmentation,
                    "internal_fragmentation": snapshot.fragmentation.internal_fragmentation,
                })
            })
            .collect();

        let allocations: Vec<serde_json::Value> = memory_profiler
            .allocation_tracker
            .allocation_history
            .iter()
            .map(|event| {
                serde_json::json!({
                    "age_us": event.timestamp.elapsed().as_micros() as u64,
                    "event_type": allocation_event_type_str(&event.event_type),
                    "address": event.address,
                    "size": event.size,
                    "context": event.context,
                })
            })
            .collect();

        let payload = serde_json::json!({
            "snapshots": snapshots,
            "allocations": allocations,
        });

        let mut file = File::create(&filename)?;
        writeln!(
            file,
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| OptimError::from(e.to_string()))?
        )?;

        self.export_stats.files_exported += 1;
        self.export_stats.export_time_us = export_start.elapsed().as_micros() as u64;
        Ok(filename)
    }
}

/// Render one [`CounterValue`] as a real JSON value (never a fabricated
/// placeholder).
fn counter_value_to_json(value: &CounterValue) -> serde_json::Value {
    match value {
        CounterValue::Integer(i) => serde_json::json!(i),
        CounterValue::Float(f) => serde_json::json!(f),
        CounterValue::Boolean(b) => serde_json::json!(b),
        CounterValue::String(s) => serde_json::json!(s),
        CounterValue::Histogram(bins) => serde_json::json!(bins
            .iter()
            .map(|(edge, count)| serde_json::json!([edge, count]))
            .collect::<Vec<_>>()),
    }
}

/// Chrome-trace-style single-letter phase code for one [`EventPhase`].
fn event_phase_code(phase: &EventPhase) -> &'static str {
    match phase {
        EventPhase::Begin => "B",
        EventPhase::End => "E",
        EventPhase::Instant => "I",
        EventPhase::Complete => "X",
        EventPhase::AsyncBegin => "b",
        EventPhase::AsyncEnd => "e",
    }
}

/// Human-readable label for one [`AllocationEventType`].
fn allocation_event_type_str(event_type: &AllocationEventType) -> &'static str {
    match event_type {
        AllocationEventType::Allocate => "allocate",
        AllocationEventType::Deallocate => "deallocate",
        AllocationEventType::Reallocate => "reallocate",
    }
}

#[cfg(test)]
mod tests;
