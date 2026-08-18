// TPU runtime integration for XLA executables
//
// This module handles integration with the TPU runtime system,
// including executable creation, device management, execution scheduling,
// and resource management.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::super::{GeneratedCode, TPUConfig, TPUVersion};
use crate::error::{OptimError, Result};
use crate::main_types::PodTopology;

/// Runtime integration manager
pub struct RuntimeIntegration {
    /// Target TPU configuration
    target_config: TPUConfig,

    /// Runtime configuration
    runtime_config: RuntimeConfig,

    /// Device manager
    device_manager: DeviceManager,

    /// Executable manager
    executable_manager: ExecutableManager,

    /// Memory manager
    memory_manager: RuntimeMemoryManager,

    /// Integration statistics
    integration_stats: RuntimeIntegrationStats,
}

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Enable asynchronous execution
    pub async_execution: bool,

    /// Enable profiling hooks
    pub enable_profiling: bool,

    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,

    /// Memory pool size
    pub memory_pool_size: usize,

    /// Timeout for operations (milliseconds)
    pub operation_timeout_ms: u64,

    /// Enable error checking
    pub enable_error_checking: bool,

    /// Runtime optimization level
    pub optimization_level: RuntimeOptimizationLevel,
}

/// Runtime optimization levels
#[derive(Debug, Clone)]
pub enum RuntimeOptimizationLevel {
    /// No optimizations
    None,

    /// Basic optimizations
    Basic,

    /// Aggressive optimizations
    Aggressive,

    /// Maximum optimizations
    Maximum,
}

/// Runtime integration statistics
#[derive(Debug, Default)]
pub struct RuntimeIntegrationStats {
    /// Total executables created
    pub executables_created: usize,

    /// Total executions
    pub total_executions: usize,

    /// Average execution time (microseconds)
    pub avg_execution_time_us: u64,

    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,

    /// Device utilization
    pub device_utilization: f64,

    /// Runtime overhead (microseconds)
    pub runtime_overhead_us: u64,

    /// Error count
    pub error_count: usize,
}

/// Device manager for TPU devices
pub struct DeviceManager {
    /// Available devices
    available_devices: Vec<TPUDevice>,

    /// Device assignments
    device_assignments: HashMap<String, usize>,

    /// Device status
    device_status: HashMap<usize, DeviceStatus>,

    /// Device capabilities cache
    capabilities_cache: HashMap<usize, DeviceCapabilities>,
}

/// TPU device representation
#[derive(Debug, Clone)]
pub struct TPUDevice {
    /// Device ID
    pub id: usize,

    /// Device type
    pub device_type: TPUDeviceType,

    /// Device version
    pub version: TPUVersion,

    /// Memory capacity (bytes)
    pub memory_capacity: usize,

    /// Compute throughput (TOPS)
    pub compute_throughput: f64,

    /// Device state
    pub state: DeviceState,

    /// Last health check
    pub last_health_check: Instant,
}

/// TPU device types
#[derive(Debug, Clone)]
pub enum TPUDeviceType {
    /// Single chip TPU
    SingleChip,

    /// Multi-chip TPU pod
    Pod,

    /// TPU slice
    Slice,

    /// Virtual TPU (for testing)
    Virtual,
}

/// Device states
#[derive(Debug, Clone)]
pub enum DeviceState {
    /// Device available for use
    Available,

    /// Device currently in use
    InUse,

    /// Device initializing
    Initializing,

    /// Device error state
    Error(String),

    /// Device maintenance mode
    Maintenance,
}

/// Device status information
#[derive(Debug, Default)]
pub struct DeviceStatus {
    /// Current utilization (0.0-1.0)
    pub utilization: f64,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// Temperature (celsius)
    pub temperature: f32,

    /// Power consumption (watts)
    pub power_consumption: f32,

    /// Error flags
    pub error_flags: Vec<String>,

    /// Performance counters
    pub performance_counters: HashMap<String, u64>,
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Supported data types
    pub supported_dtypes: Vec<String>,

    /// Maximum matrix dimensions
    pub max_matrix_dims: (usize, usize),

    /// Vector processing width
    pub vector_width: usize,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,

    /// Special instructions
    pub special_instructions: Vec<String>,

    /// Interconnect capabilities
    pub interconnect_capabilities: InterconnectCapabilities,
}

/// Interconnect capabilities
#[derive(Debug, Clone)]
pub struct InterconnectCapabilities {
    /// Inter-chip bandwidth (GB/s)
    pub inter_chip_bandwidth: f64,

    /// Inter-pod bandwidth (GB/s)
    pub inter_pod_bandwidth: f64,

    /// Supported collective operations
    pub collective_ops: Vec<String>,

    /// Topology type
    pub topology_type: TopologyType,
}

/// Network topology types
#[derive(Debug, Clone)]
pub enum TopologyType {
    /// Mesh topology
    Mesh,

    /// Torus topology
    Torus,

    /// Tree topology
    Tree,

    /// Custom topology
    Custom(String),
}

/// Executable manager.
///
/// The [`ExecutableCache`] is the single store: it owns the loaded executables,
/// tracks access for eviction and keeps real hit/miss statistics. There is no
/// separate `executables` map, no `loading_queue` and no `execution_contexts`
/// map any more -- all three were written once by the constructor and never read,
/// and a second owning map alongside the cache would have to duplicate every
/// executable (`TPUExecutable` is not `Clone`) to exist at all.
pub struct ExecutableManager {
    /// Executable store with LRU/LFU eviction and hit/miss statistics
    executable_cache: ExecutableCache,
}

/// TPU executable representation
#[derive(Debug)]
pub struct TPUExecutable {
    /// Executable ID
    pub id: String,

    /// Binary code
    pub binary: Vec<u8>,

    /// Executable metadata
    pub metadata: ExecutableMetadata,

    /// Input specifications
    pub input_specs: Vec<BufferSpec>,

    /// Output specifications
    pub output_specs: Vec<BufferSpec>,

    /// Resource requirements
    pub resource_requirements: ExecutableResourceRequirements,

    /// Performance profile
    pub performance_profile: ExecutionProfile,
}

/// Executable metadata
#[derive(Debug, Clone)]
pub struct ExecutableMetadata {
    /// Compilation timestamp
    pub compilation_time: Instant,

    /// Compiler version
    pub compiler_version: String,

    /// Target device requirements
    pub target_requirements: TargetRequirements,

    /// Optimization level used
    pub optimization_level: String,

    /// Debug information
    pub debug_info: Option<DebugInfo>,
}

/// Buffer specification
#[derive(Debug, Clone)]
pub struct BufferSpec {
    /// Buffer name
    pub name: String,

    /// Buffer size (bytes)
    pub size: usize,

    /// Data type
    pub dtype: String,

    /// Shape information
    pub shape: Vec<usize>,

    /// Memory alignment requirements
    pub alignment: usize,

    /// Access pattern
    pub access_pattern: BufferAccessPattern,
}

/// Buffer access patterns
#[derive(Debug, Clone)]
pub enum BufferAccessPattern {
    /// Sequential access
    Sequential,

    /// Random access
    Random,

    /// Strided access
    Strided(usize),

    /// Read-only access
    ReadOnly,

    /// Write-only access
    WriteOnly,
}

/// Executable resource requirements
#[derive(Debug, Default)]
pub struct ExecutableResourceRequirements {
    /// Memory requirement (bytes)
    pub memory_bytes: usize,

    /// Compute requirement (FLOPS)
    pub compute_flops: u64,

    /// Communication volume (bytes)
    pub communication_bytes: usize,

    /// Execution time estimate (microseconds)
    pub execution_time_estimate_us: u64,

    /// Device count requirement
    pub device_count: usize,
}

/// Execution profile for performance tracking
#[derive(Debug, Default)]
pub struct ExecutionProfile {
    /// Average execution time
    pub avg_execution_time_us: u64,

    /// Peak memory usage
    pub peak_memory_usage: usize,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Resource utilization
    pub resource_utilization: f64,

    /// Execution history
    pub execution_history: Vec<ExecutionRecord>,
}

/// Execution record
#[derive(Debug)]
pub struct ExecutionRecord {
    /// Execution timestamp
    pub timestamp: Instant,

    /// Execution duration
    pub duration: Duration,

    /// Input sizes
    pub input_sizes: Vec<usize>,

    /// Output sizes
    pub output_sizes: Vec<usize>,

    /// Device utilization during execution
    pub device_utilization: f64,

    /// Memory usage during execution
    pub memory_usage: usize,
}

/// Target requirements for executable
#[derive(Debug, Clone)]
pub struct TargetRequirements {
    /// Minimum TPU version
    pub min_tpu_version: TPUVersion,

    /// Required memory (bytes)
    pub required_memory: usize,

    /// Required features
    pub required_features: Vec<String>,

    /// Optional features
    pub optional_features: Vec<String>,
}

/// Debug information for executable
#[derive(Debug, Clone)]
pub struct DebugInfo {
    /// Source mapping
    pub source_mapping: HashMap<usize, String>,

    /// Symbol table
    pub symbol_table: HashMap<String, usize>,

    /// Line number information
    pub line_info: Vec<LineInfo>,
}

/// Line information for debugging
#[derive(Debug, Clone)]
pub struct LineInfo {
    /// Instruction address
    pub address: usize,

    /// Source file
    pub file: String,

    /// Line number
    pub line: u32,

    /// Function name
    pub function: String,
}

/// Executable cache for performance
pub struct ExecutableCache {
    /// Cache entries
    cache: HashMap<String, CachedExecutable>,

    /// Cache configuration
    config: CacheConfig,

    /// Cache statistics
    stats: CacheStats,
}

/// Cached executable entry
#[derive(Debug)]
pub struct CachedExecutable {
    /// Executable
    pub executable: TPUExecutable,

    /// Last access time
    pub last_access: Instant,

    /// Access count
    pub access_count: u64,

    /// Cache score
    pub score: f64,
}

/// Cache configuration
#[derive(Debug)]
pub struct CacheConfig {
    /// Maximum cache size (bytes)
    pub max_size: usize,

    /// Maximum number of entries
    pub max_entries: usize,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,

    /// Least Frequently Used
    LFU,

    /// Optimal (theoretical)
    Optimal,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Cache hits
    pub hits: u64,

    /// Cache misses
    pub misses: u64,

    /// Evictions
    pub evictions: u64,

    /// Cache utilization
    pub utilization: f64,
}

/// Loading request for executables
#[derive(Debug)]
pub struct LoadingRequest {
    /// Request ID
    pub id: String,

    /// Generated code to load
    pub code: GeneratedCode,

    /// Target device
    pub target_device: usize,

    /// Priority
    pub priority: u32,

    /// Request timestamp
    pub timestamp: Instant,
}

/// Execution context for running executables
#[derive(Debug)]
pub struct ExecutionContext {
    /// Context ID
    pub id: String,

    /// Associated device
    pub device_id: usize,

    /// Input buffers
    pub input_buffers: HashMap<String, Buffer>,

    /// Output buffers
    pub output_buffers: HashMap<String, Buffer>,

    /// Temporary buffers
    pub temp_buffers: HashMap<String, Buffer>,

    /// Context state
    pub state: ContextState,

    /// Performance counters
    pub performance_counters: HashMap<String, u64>,
}

/// Runtime buffer representation
#[derive(Debug)]
pub struct Buffer {
    /// Buffer ID
    pub id: String,

    /// Size in bytes
    pub size: usize,

    /// Memory location
    pub memory_location: MemoryLocation,

    /// Buffer status
    pub status: BufferStatus,

    /// Access tracking
    pub access_tracking: AccessTracking,
}

/// Memory locations for buffers
#[derive(Debug)]
pub enum MemoryLocation {
    /// Device memory
    Device(usize),

    /// Host memory
    Host,

    /// Shared memory
    Shared,

    /// External memory
    External(String),
}

/// Buffer status
#[derive(Debug)]
pub enum BufferStatus {
    /// Buffer allocated
    Allocated,

    /// Buffer ready for use
    Ready,

    /// Buffer in use
    InUse,

    /// Buffer being transferred
    Transferring,

    /// Buffer error state
    Error(String),
}

/// Access tracking for buffers
#[derive(Debug, Default)]
pub struct AccessTracking {
    /// Read count
    pub read_count: u64,

    /// Write count
    pub write_count: u64,

    /// Last access time
    pub last_access: Option<Instant>,

    /// Access pattern
    pub pattern: Option<BufferAccessPattern>,
}

/// Context states
#[derive(Debug)]
pub enum ContextState {
    /// Context ready
    Ready,

    /// Context executing
    Executing,

    /// Context waiting for resources
    Waiting,

    /// Context error state
    Error(String),
}

// NOTE: this module used to define its own `ExecutionScheduler`, `ResourceManager`,
// `ExecutionRequest`, `ActiveExecution`, `ExecutionProgress`, `SchedulerConfig`,
// `SchedulingPolicy`, `ResourcePool`, `ResourceType`, `ResourceAllocation`,
// `ResourceUsageTracking` and `UsageSnapshot`. Every one of them was constructed
// by `RuntimeIntegration::new` and then never read again: nothing in the crate
// ever enqueued an execution request or reserved a resource through them, and the
// names collided with the *real* `ExecutionScheduler`/`ResourceManager` in
// `xla::optimization::scheduling` (which the optimization pipeline does drive)
// and with `tpu_backend`'s real `ExecutionEngine`, which is where execution
// actually happens. They are deleted rather than left as a public API that
// schedules nothing.

/// Name of the pool [`RuntimeMemoryManager`] reserves executables from.
const RUNTIME_DEVICE_POOL: &str = "device";

/// Runtime memory manager
pub struct RuntimeMemoryManager {
    /// Memory pools
    memory_pools: HashMap<String, MemoryPool>,

    /// Buffer allocations
    buffer_allocations: HashMap<String, BufferAllocation>,

    /// Memory usage statistics
    usage_stats: MemoryUsageStats,
}

/// Memory pool for runtime
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool name
    pub name: String,

    /// Pool size (bytes)
    pub size: usize,

    /// Available memory (bytes)
    pub available: usize,

    /// Memory location
    pub location: MemoryLocation,

    /// Pool fragmentation
    pub fragmentation: f64,
}

/// Buffer allocation in runtime
#[derive(Debug)]
pub struct BufferAllocation {
    /// Buffer ID
    pub buffer_id: String,

    /// Allocated size
    pub size: usize,

    /// Memory pool
    pub pool: String,

    /// Allocation timestamp
    pub timestamp: Instant,

    /// Reference count
    pub ref_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryUsageStats {
    /// Total allocated (bytes)
    pub total_allocated: usize,

    /// Peak usage (bytes)
    pub peak_usage: usize,

    /// Fragmentation ratio
    pub fragmentation_ratio: f64,

    /// Allocation count
    pub allocation_count: usize,

    /// Deallocation count
    pub deallocation_count: usize,
}

impl RuntimeIntegration {
    /// Create new runtime integration manager
    pub fn new(target_config: TPUConfig) -> Self {
        let runtime_config = RuntimeConfig {
            async_execution: true,
            enable_profiling: false,
            max_concurrent_executions: 4,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            operation_timeout_ms: 30000,          // 30 seconds
            enable_error_checking: true,
            optimization_level: RuntimeOptimizationLevel::Basic,
        };

        Self {
            device_manager: DeviceManager::new(&target_config),
            executable_manager: ExecutableManager::new(),
            memory_manager: RuntimeMemoryManager::new(&runtime_config),
            target_config,
            runtime_config,
            integration_stats: RuntimeIntegrationStats::default(),
        }
    }

    /// Integrate generated code with runtime.
    ///
    /// This is a real admission path, not a pass-through: when error checking is
    /// enabled the generated code has to be non-empty, `target_tpu` has to be at
    /// least the version the executable declares it needs, at least one device
    /// has to be available to run it, and the executable's footprint has to fit
    /// in the runtime memory pool. Any of those failing is an honest `Err`
    /// instead of a container wrapping code that could never run.
    pub fn integrate(&mut self, code: GeneratedCode, target_tpu: &TPUConfig) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        if self.runtime_config.enable_error_checking && code.kernel_code.trim().is_empty() {
            self.integration_stats.error_count += 1;
            return Err(OptimError::from(
                "runtime integration received an empty computation kernel".to_string(),
            ));
        }

        // Create executable from generated code
        let executable = self.create_executable(code)?;

        // The caller's target has to satisfy what the executable declares it
        // needs, and a device has to exist to run it on.
        let device_id = match self
            .device_manager
            .select_device(&executable.metadata.target_requirements, target_tpu)
        {
            Ok(device_id) => device_id,
            Err(error) => {
                self.integration_stats.error_count += 1;
                return Err(error);
            }
        };

        // Reserve the executable's footprint in the runtime memory pool.
        let footprint = executable
            .binary
            .len()
            .saturating_add(executable.metadata.target_requirements.required_memory);
        let executable_id = executable.id.clone();
        if let Err(error) = self.memory_manager.reserve(&executable_id, footprint) {
            self.integration_stats.error_count += 1;
            return Err(error);
        }

        // Load executable into runtime; anything evicted to make room releases
        // its reservation so the pool cannot drift upward.
        let evicted = self.executable_manager.load_executable(executable)?;
        for evicted_id in &evicted {
            self.memory_manager.release(evicted_id);
            self.device_manager.unassign(evicted_id);
        }
        self.device_manager.assign(&executable_id, device_id);

        // Create binary representation
        let binary = self.create_binary(&executable_id)?;

        self.integration_stats.runtime_overhead_us = start_time.elapsed().as_micros() as u64;
        self.integration_stats.executables_created += 1;
        self.integration_stats.peak_memory_usage = self
            .integration_stats
            .peak_memory_usage
            .max(self.memory_manager.total_allocated());
        self.integration_stats.device_utilization = self.device_manager.utilization();

        Ok(binary)
    }

    /// Unload an executable, releasing its memory reservation and device
    /// assignment. The counterpart to [`Self::integrate`]'s reservation.
    pub fn unload_executable(&mut self, executable_id: &str) -> bool {
        let removed = self.executable_manager.remove_executable(executable_id);
        if removed {
            self.memory_manager.release(executable_id);
            self.device_manager.unassign(executable_id);
        }
        removed
    }

    /// Cache statistics for the executable store.
    pub fn cache_statistics(&self) -> &CacheStats {
        self.executable_manager.cache_statistics()
    }

    /// Runtime memory usage statistics.
    pub fn memory_statistics(&self) -> &MemoryUsageStats {
        self.memory_manager.usage_stats()
    }

    /// Integration statistics accumulated across every [`Self::integrate`] call.
    pub fn integration_statistics(&self) -> &RuntimeIntegrationStats {
        &self.integration_stats
    }

    /// Create executable from generated code
    fn create_executable(&self, code: GeneratedCode) -> Result<TPUExecutable> {
        let executable = TPUExecutable {
            id: format!("exec_{}", self.integration_stats.executables_created),
            binary: code.kernel_code.as_bytes().to_vec(),
            metadata: ExecutableMetadata {
                compilation_time: Instant::now(),
                compiler_version: "1.0.0".to_string(),
                target_requirements: TargetRequirements {
                    min_tpu_version: self.target_config.tpu_version,
                    required_memory: 1024 * 1024, // 1MB
                    required_features: vec!["matmul".to_string()],
                    optional_features: vec![],
                },
                optimization_level: "O2".to_string(),
                debug_info: None,
            },
            input_specs: vec![],
            output_specs: vec![],
            resource_requirements: ExecutableResourceRequirements::default(),
            performance_profile: ExecutionProfile::default(),
        };

        Ok(executable)
    }

    /// Serialize a loaded executable into a self-describing binary container.
    ///
    /// The container carries the real compiled code produced by the backend
    /// (the executable's `binary` bytes) framed with a magic tag, a version, the
    /// executable id and length prefixes, so it can be round-tripped/inspected.
    /// It is not a fixed placeholder — a caller must first `load_executable`.
    fn create_binary(&mut self, executable_id: &str) -> Result<Vec<u8>> {
        let executable = self
            .executable_manager
            .get_executable(executable_id)
            .ok_or_else(|| {
                OptimError::from(format!(
                    "cannot create binary: executable {executable_id} is not loaded"
                ))
            })?;

        let id_bytes = executable_id.as_bytes();
        let code = &executable.binary;
        let mut binary = Vec::with_capacity(16 + id_bytes.len() + code.len());
        // Container magic + version so the payload is identifiable/parseable.
        binary.extend_from_slice(b"TPUX");
        binary.extend_from_slice(&1u32.to_le_bytes());
        // Length-prefixed executable id.
        binary.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        binary.extend_from_slice(id_bytes);
        // Length-prefixed compiled code (the real serialized program).
        binary.extend_from_slice(&(code.len() as u64).to_le_bytes());
        binary.extend_from_slice(code);
        Ok(binary)
    }
}

impl DeviceManager {
    /// Create new device manager
    pub fn new(target_config: &TPUConfig) -> Self {
        let mut devices = Vec::new();

        // Create virtual devices based on target config
        // Determine number of chips based on pod topology
        let num_chips = match target_config.pod_topology {
            PodTopology::Single => 1,
            PodTopology::Pod2x2 => 4,
            PodTopology::Pod4x4 => 16,
            PodTopology::Pod8x8 => 64,
            PodTopology::Pod16x16 => 256,
            PodTopology::Pod32x32 => 1024,
        };

        let mut device_status = HashMap::with_capacity(num_chips);
        let mut capabilities_cache = HashMap::with_capacity(num_chips);

        for i in 0..num_chips {
            let memory_capacity = 16 * 1024 * 1024 * 1024 / num_chips;
            devices.push(TPUDevice {
                id: i,
                device_type: TPUDeviceType::SingleChip,
                version: target_config.tpu_version,
                memory_capacity,
                compute_throughput: 420.0 / num_chips as f64,
                state: DeviceState::Available,
                last_health_check: Instant::now(),
            });
            // Populate the status and capability records at enumeration time.
            // Leaving these empty meant `select_device` had nothing to consult
            // and every capability query would have missed.
            device_status.insert(i, DeviceStatus::default());
            capabilities_cache.insert(i, device_capabilities(target_config.tpu_version));
        }

        Self {
            available_devices: devices,
            device_assignments: HashMap::new(),
            device_status,
            capabilities_cache,
        }
    }

    /// Choose a device able to run an executable with the given requirements.
    ///
    /// `target_tpu` is the target the caller is compiling for; a device younger
    /// than the executable's declared minimum cannot run it, so that is a real
    /// rejection rather than a silent success.
    pub fn select_device(
        &self,
        requirements: &TargetRequirements,
        target_tpu: &TPUConfig,
    ) -> Result<usize> {
        if tpu_version_rank(target_tpu.tpu_version) < tpu_version_rank(requirements.min_tpu_version)
        {
            return Err(OptimError::from(format!(
                "target {:?} is older than the executable's minimum {:?}",
                target_tpu.tpu_version, requirements.min_tpu_version
            )));
        }

        // Prefer the least-utilized available device that has the memory and the
        // required features.
        let mut best: Option<(usize, f64)> = None;
        for device in &self.available_devices {
            if !matches!(device.state, DeviceState::Available | DeviceState::InUse) {
                continue;
            }
            if device.memory_capacity < requirements.required_memory {
                continue;
            }
            if tpu_version_rank(device.version) < tpu_version_rank(requirements.min_tpu_version) {
                continue;
            }
            if let Some(capabilities) = self.capabilities_cache.get(&device.id) {
                let supports_all = requirements.required_features.iter().all(|feature| {
                    capabilities
                        .special_instructions
                        .iter()
                        .any(|available| available == feature)
                });
                if !supports_all {
                    continue;
                }
            }
            let utilization = self
                .device_status
                .get(&device.id)
                .map(|status| status.utilization)
                .unwrap_or(0.0);
            match best {
                Some((_, best_utilization)) if best_utilization <= utilization => {}
                _ => best = Some((device.id, utilization)),
            }
        }

        best.map(|(id, _)| id).ok_or_else(|| {
            OptimError::from(format!(
                "no TPU device among {} available satisfies the executable's requirements \
                 ({} bytes, features {:?})",
                self.available_devices.len(),
                requirements.required_memory,
                requirements.required_features
            ))
        })
    }

    /// Record that an executable is placed on a device.
    pub fn assign(&mut self, executable_id: &str, device_id: usize) {
        self.device_assignments
            .insert(executable_id.to_string(), device_id);
        if let Some(device) = self
            .available_devices
            .iter_mut()
            .find(|device| device.id == device_id)
        {
            device.state = DeviceState::InUse;
        }
    }

    /// Drop an executable's placement, freeing the device when nothing else is
    /// assigned to it.
    pub fn unassign(&mut self, executable_id: &str) {
        if let Some(device_id) = self.device_assignments.remove(executable_id) {
            let still_used = self
                .device_assignments
                .values()
                .any(|assigned| *assigned == device_id);
            if !still_used {
                if let Some(device) = self
                    .available_devices
                    .iter_mut()
                    .find(|device| device.id == device_id)
                {
                    device.state = DeviceState::Available;
                }
            }
        }
    }

    /// Fraction of enumerated devices currently holding at least one executable.
    pub fn utilization(&self) -> f64 {
        if self.available_devices.is_empty() {
            return 0.0;
        }
        let busy: std::collections::HashSet<usize> =
            self.device_assignments.values().copied().collect();
        busy.len() as f64 / self.available_devices.len() as f64
    }

    /// Devices enumerated from the target configuration.
    pub fn devices(&self) -> &[TPUDevice] {
        &self.available_devices
    }

    /// Cached capabilities for a device, if it is enumerated.
    pub fn capabilities(&self, device_id: usize) -> Option<&DeviceCapabilities> {
        self.capabilities_cache.get(&device_id)
    }

    /// Live status record for a device, if it is enumerated.
    pub fn status(&self, device_id: usize) -> Option<&DeviceStatus> {
        self.device_status.get(&device_id)
    }
}

/// Ordering rank for a TPU version, so "at least version X" is expressible.
/// `TPUVersion` is deliberately not `Ord` (V5e and V5p are different product
/// lines rather than a strict upgrade), so the rank is stated explicitly here.
fn tpu_version_rank(version: TPUVersion) -> u8 {
    match version {
        TPUVersion::V2 => 2,
        TPUVersion::V3 => 3,
        TPUVersion::V4 => 4,
        TPUVersion::V5e => 5,
        TPUVersion::V5p => 6,
    }
}

/// Capabilities of one device of a given TPU version, derived from the version
/// rather than left as an empty cache entry.
fn device_capabilities(version: TPUVersion) -> DeviceCapabilities {
    let (matrix_dims, vector_width, memory_bandwidth, inter_chip, inter_pod) = match version {
        TPUVersion::V2 => ((128, 128), 8, 600.0, 500.0, 100.0),
        TPUVersion::V3 => ((128, 128), 8, 900.0, 900.0, 200.0),
        TPUVersion::V4 => ((128, 128), 8, 1200.0, 1200.0, 300.0),
        TPUVersion::V5e => ((128, 128), 8, 819.0, 1600.0, 400.0),
        TPUVersion::V5p => ((128, 128), 8, 2765.0, 4800.0, 800.0),
    };
    DeviceCapabilities {
        supported_dtypes: vec![
            "f32".to_string(),
            "bf16".to_string(),
            "f16".to_string(),
            "s32".to_string(),
            "s8".to_string(),
        ],
        max_matrix_dims: matrix_dims,
        vector_width,
        memory_bandwidth,
        special_instructions: vec![
            "matmul".to_string(),
            "conv".to_string(),
            "reduce".to_string(),
            "transpose".to_string(),
        ],
        interconnect_capabilities: InterconnectCapabilities {
            inter_chip_bandwidth: inter_chip,
            inter_pod_bandwidth: inter_pod,
            collective_ops: vec![
                "all-reduce".to_string(),
                "all-gather".to_string(),
                "reduce-scatter".to_string(),
            ],
            topology_type: TopologyType::Mesh,
        },
    }
}

impl Default for ExecutableManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutableManager {
    /// Create new executable manager
    pub fn new() -> Self {
        Self {
            executable_cache: ExecutableCache::new(),
        }
    }

    /// Load an executable into the runtime store.
    ///
    /// Returns the ids evicted to make room, so the caller can release whatever
    /// those executables were holding.
    pub fn load_executable(&mut self, executable: TPUExecutable) -> Result<Vec<String>> {
        Ok(self.executable_cache.insert(executable))
    }

    /// Look up a previously loaded executable by id, recording the hit or miss.
    pub fn get_executable(&mut self, id: &str) -> Option<&TPUExecutable> {
        self.executable_cache.lookup(id)
    }

    /// Drop an executable from the store.
    pub fn remove_executable(&mut self, id: &str) -> bool {
        self.executable_cache.remove(id)
    }

    /// Hit/miss/eviction statistics for the store.
    pub fn cache_statistics(&self) -> &CacheStats {
        self.executable_cache.statistics()
    }
}

impl Default for ExecutableCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutableCache {
    /// Create new executable cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            config: CacheConfig {
                max_size: 100 * 1024 * 1024, // 100MB
                max_entries: 100,
                eviction_policy: EvictionPolicy::LRU,
            },
            stats: CacheStats::default(),
        }
    }

    /// Insert an executable, evicting entries until the configured size and
    /// entry budgets are met. Returns the evicted ids.
    pub fn insert(&mut self, executable: TPUExecutable) -> Vec<String> {
        let id = executable.id.clone();
        let size = executable.binary.len();
        self.cache.insert(
            id,
            CachedExecutable {
                executable,
                last_access: Instant::now(),
                access_count: 1,
                score: size as f64,
            },
        );
        let evicted = self.enforce_budget();
        self.refresh_utilization();
        evicted
    }

    /// Look up an executable without touching the access bookkeeping.
    pub fn peek(&self, id: &str) -> Option<&TPUExecutable> {
        self.cache.get(id).map(|entry| &entry.executable)
    }

    /// Look up an executable, recording the hit or miss and refreshing the
    /// entry's recency/frequency so eviction ordering reflects real use.
    pub fn lookup(&mut self, id: &str) -> Option<&TPUExecutable> {
        match self.cache.get_mut(id) {
            Some(entry) => {
                entry.last_access = Instant::now();
                entry.access_count += 1;
                entry.score = entry.executable.binary.len() as f64 * entry.access_count as f64;
                self.stats.hits += 1;
                Some(&entry.executable)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Remove an executable from the cache.
    pub fn remove(&mut self, id: &str) -> bool {
        let removed = self.cache.remove(id).is_some();
        if removed {
            self.refresh_utilization();
        }
        removed
    }

    /// Cache statistics.
    pub fn statistics(&self) -> &CacheStats {
        &self.stats
    }

    /// Total bytes currently held.
    fn occupied_bytes(&self) -> usize {
        self.cache
            .values()
            .map(|entry| entry.executable.binary.len())
            .sum()
    }

    /// Evict until both the byte and entry budgets are satisfied, in the order
    /// the configured policy prescribes.
    fn enforce_budget(&mut self) -> Vec<String> {
        let mut evicted = Vec::new();
        while self.cache.len() > self.config.max_entries.max(1)
            || (self.occupied_bytes() > self.config.max_size && self.cache.len() > 1)
        {
            let victim = match self.config.eviction_policy {
                // Oldest access first.
                EvictionPolicy::LRU => self
                    .cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_access)
                    .map(|(id, _)| id.clone()),
                // Fewest accesses first.
                EvictionPolicy::LFU => self
                    .cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(id, _)| id.clone()),
                // Belady's optimal policy needs the future reference string,
                // which a runtime cache does not have; the closest realizable
                // approximation is to evict the entry with the worst
                // size-weighted access score.
                EvictionPolicy::Optimal => self
                    .cache
                    .iter()
                    .min_by(|a, b| a.1.score.total_cmp(&b.1.score))
                    .map(|(id, _)| id.clone()),
            };
            match victim {
                Some(id) => {
                    self.cache.remove(&id);
                    self.stats.evictions += 1;
                    evicted.push(id);
                }
                None => break,
            }
        }
        evicted
    }

    fn refresh_utilization(&mut self) {
        self.stats.utilization = if self.config.max_size == 0 {
            0.0
        } else {
            self.occupied_bytes() as f64 / self.config.max_size as f64
        };
    }
}

impl RuntimeMemoryManager {
    /// Reserve `size` bytes for `buffer_id` out of the device pool.
    ///
    /// An oversubscribed pool is an honest `Err`: the executable genuinely does
    /// not fit in the configured runtime memory.
    pub fn reserve(&mut self, buffer_id: &str, size: usize) -> Result<()> {
        let pool = self
            .memory_pools
            .get_mut(RUNTIME_DEVICE_POOL)
            .ok_or_else(|| {
                OptimError::from(format!(
                    "runtime memory pool {RUNTIME_DEVICE_POOL} is missing"
                ))
            })?;
        if size > pool.available {
            return Err(OptimError::from(format!(
                "executable needs {size} bytes but only {} of {} bytes are free in the runtime pool",
                pool.available, pool.size
            )));
        }
        pool.available -= size;
        pool.fragmentation = if pool.size == 0 {
            0.0
        } else {
            1.0 - (pool.available as f64 / pool.size as f64)
        };

        self.buffer_allocations.insert(
            buffer_id.to_string(),
            BufferAllocation {
                buffer_id: buffer_id.to_string(),
                size,
                pool: RUNTIME_DEVICE_POOL.to_string(),
                timestamp: Instant::now(),
                ref_count: 1,
            },
        );

        self.usage_stats.total_allocated += size;
        self.usage_stats.allocation_count += 1;
        self.usage_stats.peak_usage = self
            .usage_stats
            .peak_usage
            .max(self.usage_stats.total_allocated);
        self.usage_stats.fragmentation_ratio = self
            .memory_pools
            .get(RUNTIME_DEVICE_POOL)
            .map(|pool| pool.fragmentation)
            .unwrap_or(0.0);
        Ok(())
    }

    /// Release a previous reservation. Returns the bytes returned to the pool.
    pub fn release(&mut self, buffer_id: &str) -> usize {
        let Some(allocation) = self.buffer_allocations.remove(buffer_id) else {
            return 0;
        };
        if let Some(pool) = self.memory_pools.get_mut(&allocation.pool) {
            pool.available = pool
                .available
                .saturating_add(allocation.size)
                .min(pool.size);
            pool.fragmentation = if pool.size == 0 {
                0.0
            } else {
                1.0 - (pool.available as f64 / pool.size as f64)
            };
        }
        self.usage_stats.total_allocated = self
            .usage_stats
            .total_allocated
            .saturating_sub(allocation.size);
        self.usage_stats.deallocation_count += 1;
        self.usage_stats.fragmentation_ratio = self
            .memory_pools
            .get(&allocation.pool)
            .map(|pool| pool.fragmentation)
            .unwrap_or(0.0);
        allocation.size
    }

    /// Bytes currently reserved across all pools.
    pub fn total_allocated(&self) -> usize {
        self.usage_stats.total_allocated
    }

    /// Usage statistics.
    pub fn usage_stats(&self) -> &MemoryUsageStats {
        &self.usage_stats
    }

    /// Create new runtime memory manager
    pub fn new(runtime_config: &RuntimeConfig) -> Self {
        let mut memory_pools = HashMap::new();

        // Create device memory pool
        memory_pools.insert(
            "device".to_string(),
            MemoryPool {
                name: "device".to_string(),
                size: runtime_config.memory_pool_size,
                available: runtime_config.memory_pool_size,
                location: MemoryLocation::Device(0),
                fragmentation: 0.0,
            },
        );

        Self {
            memory_pools,
            buffer_allocations: HashMap::new(),
            usage_stats: MemoryUsageStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::main_types::TPUConfig;

    #[test]
    fn test_runtime_integration_creation() {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

        let tpu_config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Pod2x2,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        };

        let runtime = RuntimeIntegration::new(tpu_config);
        assert_eq!(runtime.integration_stats.executables_created, 0);
        assert_eq!(runtime.integration_stats.total_executions, 0);
        assert!(runtime.runtime_config.async_execution);
    }

    #[test]
    fn test_device_manager_creation() {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

        let tpu_config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Pod2x2,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        };

        let device_manager = DeviceManager::new(&tpu_config);
        // Pod2x2 topology creates 4 devices (2x2 grid)
        assert_eq!(device_manager.available_devices.len(), 4);

        for device in &device_manager.available_devices {
            assert!(matches!(device.state, DeviceState::Available));
            assert_eq!(device.version, TPUVersion::V4);
        }
    }

    #[test]
    fn test_create_binary_serializes_real_code_not_placeholder() {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

        let tpu_config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Single,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        };

        let kernel = "tpu.matmul %a, %b -> %c";
        let code = GeneratedCode {
            kernel_code: kernel.to_string(),
            init_code: String::new(),
            cleanup_code: String::new(),
            memory_code: String::new(),
        };

        let mut runtime = RuntimeIntegration::new(tpu_config.clone());
        let binary = runtime
            .integrate(code, &tpu_config)
            .expect("integration should produce a binary");

        // Not the old 4-byte placeholder.
        assert_ne!(binary, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        // Real self-describing container carrying the actual kernel code.
        assert!(
            binary.starts_with(b"TPUX"),
            "binary must carry container magic"
        );
        assert!(
            binary.windows(kernel.len()).any(|w| w == kernel.as_bytes()),
            "binary must embed the real compiled kernel code"
        );
    }

    #[test]
    fn test_create_binary_missing_executable_errors() {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

        let tpu_config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Single,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        };

        let mut runtime = RuntimeIntegration::new(tpu_config);
        assert!(
            runtime.create_binary("never_loaded").is_err(),
            "create_binary must fail for an unloaded executable id"
        );
        // The failed lookup is a real cache miss, not a silent zero.
        assert_eq!(runtime.cache_statistics().misses, 1);
        assert_eq!(runtime.cache_statistics().hits, 0);
    }

    /// `integrate` must reserve real memory, record a cache hit when it reads
    /// the executable back, and hand the reservation back on unload.
    #[test]
    fn integrate_reserves_and_unload_releases() {
        let tpu_config = test_config(PodTopology::Pod2x2);
        let mut runtime = RuntimeIntegration::new(tpu_config.clone());

        let code = GeneratedCode {
            kernel_code: "tpu.add %a, %b -> %c".to_string(),
            init_code: String::new(),
            cleanup_code: String::new(),
            memory_code: String::new(),
        };
        runtime
            .integrate(code, &tpu_config)
            .expect("integration should succeed");

        assert!(
            runtime.memory_statistics().total_allocated > 0,
            "the executable's footprint must actually be reserved"
        );
        assert_eq!(runtime.cache_statistics().hits, 1);
        assert_eq!(runtime.integration_statistics().executables_created, 1);
        assert!(runtime.integration_statistics().device_utilization > 0.0);

        assert!(runtime.unload_executable("exec_0"));
        assert_eq!(runtime.memory_statistics().total_allocated, 0);
        assert!(!runtime.unload_executable("exec_0"));
    }

    /// An empty kernel is a real integration failure, not a container wrapping
    /// nothing.
    #[test]
    fn integrate_rejects_an_empty_kernel() {
        let tpu_config = test_config(PodTopology::Single);
        let mut runtime = RuntimeIntegration::new(tpu_config.clone());

        let code = GeneratedCode {
            kernel_code: "   \n".to_string(),
            init_code: String::new(),
            cleanup_code: String::new(),
            memory_code: String::new(),
        };
        assert!(runtime.integrate(code, &tpu_config).is_err());
        assert_eq!(runtime.integration_statistics().error_count, 1);
    }

    /// A target older than the executable's declared minimum cannot run it.
    #[test]
    fn integrate_rejects_a_target_older_than_the_executable_needs() {
        let build_config = test_config(PodTopology::Single);
        let mut runtime = RuntimeIntegration::new(build_config.clone());

        let mut older = build_config.clone();
        older.tpu_version = TPUVersion::V2;

        let code = GeneratedCode {
            kernel_code: "tpu.matmul %a, %b -> %c".to_string(),
            init_code: String::new(),
            cleanup_code: String::new(),
            memory_code: String::new(),
        };
        // `create_executable` stamps the generator's own target as the minimum,
        // so a V2 caller cannot run a V4-built executable.
        assert!(runtime.integrate(code, &older).is_err());
    }

    fn test_config(pod_topology: PodTopology) -> TPUConfig {
        TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology,
            memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        }
    }
}
