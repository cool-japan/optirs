use std::fmt::Debug;
// Resource allocation and management for optimization coordination
//
// This module provides comprehensive resource allocation capabilities including
// dynamic resource pools, intelligent allocation strategies, and optimization
// of resource utilization across optimization tasks.

#[allow(dead_code)]
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use crate::error::{OptimError, Result};

/// Group a sorted, deduplicated list of free resource IDs into contiguous
/// runs (e.g. `[0,1,2,5,6,9]` -> `[[0,1,2],[5,6],[9]]`).
fn contiguous_blocks(sorted_ids: &[usize]) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &id in sorted_ids {
        let extends_last = matches!(
            blocks.last().and_then(|b: &Vec<usize>| b.last()),
            Some(&last) if id == last + 1
        );
        if extends_last {
            if let Some(block) = blocks.last_mut() {
                block.push(id);
            }
        } else {
            blocks.push(vec![id]);
        }
    }
    blocks
}

/// Pick `count` IDs from `free_ids` (already sorted ascending) by taking the
/// lowest-numbered free IDs, without regard to contiguity. Returns `None`
/// (exhaustion) if fewer than `count` IDs are free.
fn pick_first_fit(free_ids: &[usize], count: usize) -> Option<Vec<usize>> {
    if free_ids.len() < count {
        None
    } else {
        Some(free_ids[..count].to_vec())
    }
}

/// Pick `count` IDs from `free_ids` (already sorted ascending) preferring the
/// smallest contiguous block that can satisfy the request (minimizing
/// fragmentation waste). Falls back to draining the largest blocks first when
/// no single block is big enough. Returns `None` (exhaustion) if fewer than
/// `count` IDs are free in total.
fn pick_best_fit(free_ids: &[usize], count: usize) -> Option<Vec<usize>> {
    if free_ids.len() < count {
        return None;
    }
    if count == 0 {
        return Some(Vec::new());
    }

    let blocks = contiguous_blocks(free_ids);

    if let Some(best_block) = blocks
        .iter()
        .filter(|b| b.len() >= count)
        .min_by_key(|b| b.len())
    {
        return Some(best_block[..count].to_vec());
    }

    // No single contiguous block is big enough: drain the largest blocks
    // first, which still minimizes the number of fragments created.
    let mut sorted_blocks = blocks;
    sorted_blocks.sort_by_key(|block| std::cmp::Reverse(block.len()));
    let mut chosen = Vec::with_capacity(count);
    for block in sorted_blocks {
        for id in block {
            if chosen.len() == count {
                break;
            }
            chosen.push(id);
        }
        if chosen.len() == count {
            break;
        }
    }
    Some(chosen)
}

/// Resource manager for optimization processes
#[derive(Debug)]
pub struct ResourceManager<T: Float + Debug + Send + Sync + 'static> {
    /// Available resource pool
    resource_pool: ResourcePool,

    /// Resource allocation tracker
    allocation_tracker: ResourceAllocationTracker<T>,

    /// Allocation strategy
    allocation_strategy: ResourceAllocationStrategy,

    /// Resource constraints
    constraints: ResourceConstraints,

    /// Manager configuration
    config: ResourceManagerConfig<T>,

    /// Resource statistics
    stats: ResourceStatistics<T>,
}

/// Resource pool representing available system resources
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// CPU cores available
    pub cpu_cores: usize,

    /// CPU core specifications
    pub cpu_specs: Vec<CpuCoreSpec>,

    /// Memory available (MB)
    pub memory_mb: usize,

    /// Memory specifications
    pub memory_specs: MemorySpec,

    /// GPU devices available
    pub gpu_devices: usize,

    /// GPU specifications
    pub gpu_specs: Vec<GpuSpec>,

    /// Storage available (GB)
    pub storage_gb: usize,

    /// Storage specifications
    pub storage_specs: Vec<StorageSpec>,

    /// Network bandwidth (Mbps)
    pub network_bandwidth: f64,

    /// Network specifications
    pub network_specs: NetworkSpec,

    /// Special hardware resources
    pub special_hardware: HashMap<String, usize>,

    /// Resource availability timestamps
    pub availability_times: HashMap<String, SystemTime>,
}

/// CPU core specification
#[derive(Debug, Clone)]
pub struct CpuCoreSpec {
    /// Core identifier
    pub core_id: usize,

    /// Core frequency (GHz)
    pub frequency_ghz: f64,

    /// Core architecture
    pub architecture: String,

    /// Cache size (MB)
    pub cache_mb: usize,

    /// Performance rating
    pub performance_rating: f64,

    /// Power consumption (watts)
    pub power_consumption: f64,
}

/// Memory specification
#[derive(Debug, Clone)]
pub struct MemorySpec {
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: String,

    /// Memory speed (MHz)
    pub speed_mhz: usize,

    /// Memory channels
    pub channels: usize,

    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f64,

    /// ECC support
    pub ecc_support: bool,
}

/// GPU specification
#[derive(Debug, Clone)]
pub struct GpuSpec {
    /// GPU identifier
    pub gpu_id: usize,

    /// GPU model
    pub model: String,

    /// Memory size (GB)
    pub memory_gb: usize,

    /// Compute units
    pub compute_units: usize,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f64,

    /// Compute capability
    pub compute_capability: String,

    /// Power consumption (watts)
    pub power_consumption: f64,
}

/// Storage specification
#[derive(Debug, Clone)]
pub struct StorageSpec {
    /// Storage identifier
    pub storage_id: usize,

    /// Storage type (SSD, HDD, NVMe)
    pub storage_type: String,

    /// Capacity (GB)
    pub capacity_gb: usize,

    /// Read speed (MB/s)
    pub read_speed_mbps: f64,

    /// Write speed (MB/s)
    pub write_speed_mbps: f64,

    /// Latency (microseconds)
    pub latency_us: f64,
}

/// Network specification
#[derive(Debug, Clone)]
pub struct NetworkSpec {
    /// Network type (Ethernet, InfiniBand, etc.)
    pub network_type: String,

    /// Maximum bandwidth (Mbps)
    pub max_bandwidth_mbps: f64,

    /// Latency (microseconds)
    pub latency_us: f64,

    /// Protocol support
    pub protocols: Vec<String>,
}

/// Resource allocation tracker
#[derive(Debug)]
pub struct ResourceAllocationTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Current allocations
    current_allocations: HashMap<String, ResourceAllocation>,

    /// Allocation history
    allocation_history: VecDeque<AllocationEvent>,

    /// Resource utilization tracking
    utilization_tracker: UtilizationTracker<T>,
}

/// Resource allocation for a specific task
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Task identifier
    pub task_id: String,

    /// Allocated CPU cores
    pub cpu_cores: Vec<usize>,

    /// Allocated memory (MB)
    pub memory_mb: usize,

    /// Allocated GPU devices
    pub gpu_devices: Vec<usize>,

    /// Allocated storage (GB)
    pub storage_gb: usize,

    /// Allocated network bandwidth (Mbps)
    pub network_bandwidth: f64,

    /// Special hardware allocations
    pub special_hardware: HashMap<String, usize>,

    /// Allocation timestamp
    pub allocated_at: SystemTime,

    /// Expected release time
    pub expected_release: SystemTime,

    /// Allocation priority
    pub priority: u8,
}

/// Allocation event for tracking
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Event type
    pub event_type: AllocationEventType,

    /// Task identifier
    pub task_id: String,

    /// Resource allocation details
    pub allocation: ResourceAllocation,

    /// Event timestamp
    pub timestamp: SystemTime,

    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Types of allocation events
#[derive(Debug, Clone, Copy)]
pub enum AllocationEventType {
    /// Resource allocated
    Allocated,

    /// Resource deallocated
    Deallocated,

    /// Allocation modified
    Modified,

    /// Allocation failed
    Failed,

    /// Allocation expired
    Expired,
}

/// Resource utilization tracker
#[derive(Debug)]
pub struct UtilizationTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Current CPU utilization per core
    cpu_utilization: Vec<T>,

    /// Current memory utilization
    memory_utilization: T,

    /// Current GPU utilization per device
    gpu_utilization: Vec<T>,

    /// Current storage utilization
    storage_utilization: T,

    /// Current network utilization
    network_utilization: T,
}

/// Utilization snapshot
#[derive(Debug, Clone)]
pub struct UtilizationSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    pub timestamp: SystemTime,

    /// CPU utilization
    pub cpu_utilization: Vec<T>,

    /// Memory utilization
    pub memory_utilization: T,

    /// GPU utilization
    pub gpu_utilization: Vec<T>,

    /// Storage utilization
    pub storage_utilization: T,

    /// Network utilization
    pub network_utilization: T,

    /// Overall system utilization
    pub overall_utilization: T,
}

/// Trend direction
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}

/// Conflict check definition
#[derive(Debug)]
pub struct ConflictCheck {
    /// Check identifier
    pub check_id: String,

    /// Resource types to check
    pub resource_types: Vec<String>,

    /// Conflict detection algorithm
    pub detection_algorithm: ConflictDetectionAlgorithm,

    /// Check frequency
    pub check_frequency: Duration,
}

/// Conflict detection algorithms
#[derive(Debug, Clone, Copy)]
pub enum ConflictDetectionAlgorithm {
    /// Simple overlap detection
    SimpleOverlap,

    /// Resource capacity checking
    CapacityBased,

    /// Time-based conflict detection
    TimeBased,

    /// Dependency-based detection
    DependencyBased,

    /// Machine learning based
    MLBased,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy)]
pub enum ConflictResolutionStrategy {
    /// First-come-first-served
    FirstComeFirstServed,

    /// Priority-based resolution
    PriorityBased,

    /// Resource sharing
    ResourceSharing,

    /// Time-slicing
    TimeSlicing,

    /// Alternative resource allocation
    AlternativeAllocation,

    /// Preemption
    Preemption,
}

/// Allocation conflict representation
#[derive(Debug, Clone)]
pub struct AllocationConflict {
    /// Conflict identifier
    pub conflict_id: String,

    /// Conflicting task identifiers
    pub conflicting_tasks: Vec<String>,

    /// Conflicting resources
    pub conflicting_resources: Vec<String>,

    /// Conflict type
    pub conflict_type: ConflictType,

    /// Conflict timestamp
    pub timestamp: SystemTime,

    /// Resolution applied
    pub resolution: Option<ConflictResolutionStrategy>,

    /// Resolution success
    pub resolved: bool,
}

/// Types of resource conflicts
#[derive(Debug, Clone, Copy)]
pub enum ConflictType {
    /// Resource over-allocation
    OverAllocation,

    /// Resource dependency conflict
    DependencyConflict,

    /// Time overlap conflict
    TimeOverlap,

    /// Exclusive access conflict
    ExclusiveAccess,

    /// Performance interference
    PerformanceInterference,
}

/// Resource optimization objectives
#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    /// Maximize throughput
    MaximizeThroughput,

    /// Minimize latency
    MinimizeLatency,

    /// Maximize utilization
    MaximizeUtilization,

    /// Minimize power consumption
    MinimizePower,

    /// Maximize fairness
    MaximizeFairness,

    /// Minimize cost
    MinimizeCost,
}

/// Resource optimization algorithm trait
pub trait ResourceOptimizationAlgorithm<T: Float + Debug + Send + Sync + 'static>:
    Send + Sync + std::fmt::Debug
{
    /// Optimize resource allocation
    fn optimize(
        &mut self,
        current_state: &ResourceState<T>,
        objectives: &[OptimizationObjective],
    ) -> Result<OptimizationResult<T>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm performance metrics
    fn get_metrics(&self) -> HashMap<String, T>;
}

/// Current resource state
#[derive(Debug, Clone)]
pub struct ResourceState<T: Float + Debug + Send + Sync + 'static> {
    /// Available resources
    pub available_resources: ResourcePool,

    /// Current allocations
    pub current_allocations: HashMap<String, ResourceAllocation>,

    /// Resource utilization
    pub utilization: UtilizationSnapshot<T>,

    /// Pending requests
    pub pending_requests: Vec<ResourceRequest>,

    /// Performance metrics
    pub performance_metrics: HashMap<String, T>,
}

/// Resource request from tasks
#[derive(Debug, Clone)]
pub struct ResourceRequest {
    /// Requesting task identifier
    pub task_id: String,

    /// Required CPU cores
    pub cpu_cores: usize,

    /// Required memory (MB)
    pub memory_mb: usize,

    /// Required GPU devices
    pub gpu_devices: usize,

    /// Required storage (GB)
    pub storage_gb: usize,

    /// Required network bandwidth (Mbps)
    pub network_bandwidth: f64,

    /// Special hardware requirements
    pub special_hardware: HashMap<String, usize>,

    /// Request priority
    pub priority: u8,

    /// Request deadline
    pub deadline: Option<SystemTime>,

    /// Request timestamp
    pub requested_at: SystemTime,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Proposed allocations
    pub proposed_allocations: HashMap<String, ResourceAllocation>,

    /// Expected performance improvement
    pub performance_improvement: T,

    /// Optimization objectives achieved
    pub objectives_achieved: HashMap<OptimizationObjective, T>,

    /// Optimization cost
    pub optimization_cost: T,

    /// Confidence in result
    pub confidence: T,

    /// Optimization algorithm used
    pub algorithm_used: String,
}

/// Performance data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint<T: Float + Debug + Send + Sync + 'static> {
    /// Resource allocation
    pub allocation: ResourceAllocation,

    /// Actual performance achieved
    pub performance: T,

    /// Task characteristics
    pub task_characteristics: HashMap<String, T>,

    /// Environmental factors
    pub environmental_factors: HashMap<String, T>,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round robin balancing
    RoundRobin,

    /// Least loaded first
    LeastLoaded,

    /// Weighted round robin
    WeightedRoundRobin,

    /// Resource-aware balancing
    ResourceAware,

    /// Performance-based balancing
    PerformanceBased,

    /// Predictive balancing
    Predictive,
}

/// Load balancing event
#[derive(Debug, Clone)]
pub struct LoadBalancingEvent<T: Float + Debug + Send + Sync + 'static> {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Balancing strategy used
    pub strategy: LoadBalancingStrategy,

    /// Load before balancing
    pub load_before: HashMap<String, T>,

    /// Load after balancing
    pub load_after: HashMap<String, T>,

    /// Balancing effectiveness
    pub effectiveness: T,
}

/// Load prediction
#[derive(Debug, Clone)]
pub struct LoadPrediction<T: Float + Debug + Send + Sync + 'static> {
    /// Prediction timestamp
    pub timestamp: SystemTime,

    /// Target time
    pub target_time: SystemTime,

    /// Predicted loads
    pub predicted_loads: HashMap<String, T>,

    /// Prediction confidence
    pub confidence: T,

    /// Actual loads (filled in later)
    pub actual_loads: Option<HashMap<String, T>>,
}

/// Resource allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum ResourceAllocationStrategy {
    /// Best fit allocation
    BestFit,

    /// First fit allocation
    FirstFit,

    /// Worst fit allocation
    WorstFit,

    /// Performance-optimized allocation
    PerformanceOptimized,

    /// Energy-efficient allocation
    EnergyEfficient,

    /// Fair share allocation
    FairShare,

    /// Priority-based allocation
    PriorityBased,
}

/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum CPU utilization allowed
    pub max_cpu_utilization: f64,

    /// Maximum memory utilization allowed
    pub max_memory_utilization: f64,

    /// Maximum GPU utilization allowed
    pub max_gpu_utilization: f64,

    /// Minimum available resources to maintain
    pub min_available_resources: ResourcePool,

    /// Resource isolation requirements
    pub isolation_requirements: HashMap<String, IsolationLevel>,

    /// Performance guarantees
    pub performance_guarantees: Vec<PerformanceGuarantee>,
}

/// Resource isolation levels
#[derive(Debug, Clone, Copy)]
pub enum IsolationLevel {
    /// No isolation
    None,

    /// Process-level isolation
    Process,

    /// Container-level isolation
    Container,

    /// Virtual machine isolation
    VirtualMachine,

    /// Hardware-level isolation
    Hardware,
}

/// Performance guarantee
#[derive(Debug, Clone)]
pub struct PerformanceGuarantee {
    /// Guarantee identifier
    pub guarantee_id: String,

    /// Resource type
    pub resource_type: String,

    /// Minimum performance level
    pub min_performance: f64,

    /// Performance metric
    pub metric: String,

    /// Guarantee priority
    pub priority: u8,
}

/// Resource manager configuration
#[derive(Debug, Clone)]
pub struct ResourceManagerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Allocation timeout
    pub allocation_timeout: Duration,

    /// Resource monitoring interval
    pub monitoring_interval: Duration,

    /// Optimization interval
    pub optimization_interval: Duration,

    /// Maximum allocation retries
    pub max_allocation_retries: usize,

    /// Enable predictive allocation
    pub enable_predictive_allocation: bool,

    /// Enable load balancing
    pub enable_load_balancing: bool,

    /// Resource over-provisioning factor
    pub over_provisioning_factor: T,

    /// Enable conflict detection
    pub enable_conflict_detection: bool,
}

/// Resource statistics
#[derive(Debug, Clone)]
pub struct ResourceStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total allocations made
    pub total_allocations: usize,

    /// Total deallocations
    pub total_deallocations: usize,

    /// Failed allocations
    pub failed_allocations: usize,

    /// Average allocation time
    pub average_allocation_time: Duration,

    /// Average utilization efficiency
    pub average_utilization_efficiency: T,

    /// Resource fragmentation
    pub fragmentation: T,

    /// Load balancing effectiveness
    pub load_balance_effectiveness: T,

    /// Conflict resolution success rate
    pub conflict_resolution_rate: T,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    ResourceManager<T>
{
    /// Create new resource manager
    pub fn new(resource_pool: ResourcePool, config: ResourceManagerConfig<T>) -> Result<Self> {
        Self::with_constraints(resource_pool, config, ResourceConstraints::default())
    }

    /// Create a manager with explicit resource constraints.
    ///
    /// Both `config` and `constraints` are validated, and both then *drive*
    /// [`Self::allocate_resources`]. Until 0.3.2 they were stored and never
    /// read: the utilization ceilings, the reserved floor and the
    /// over-provisioning factor had no effect at all, and the three
    /// `enable_*` switches advertised features with no implementation behind
    /// them.
    ///
    /// # Errors
    ///
    /// * [`OptimError::InvalidConfig`] -- a utilization ceiling outside
    ///   `(0, 1]`, a non-positive or non-finite `over_provisioning_factor`, or a
    ///   reserved floor larger than the pool itself.
    /// * [`OptimError::UnsupportedOperation`] -- `enable_predictive_allocation`,
    ///   `enable_load_balancing` or `enable_conflict_detection` is set. No code
    ///   path implements them, and silently ignoring a switch is how a caller
    ///   ends up believing it got a guarantee it did not buy.
    pub fn with_constraints(
        resource_pool: ResourcePool,
        config: ResourceManagerConfig<T>,
        constraints: ResourceConstraints,
    ) -> Result<Self> {
        for (name, enabled) in [
            (
                "enable_predictive_allocation",
                config.enable_predictive_allocation,
            ),
            ("enable_load_balancing", config.enable_load_balancing),
            (
                "enable_conflict_detection",
                config.enable_conflict_detection,
            ),
        ] {
            if enabled {
                return Err(OptimError::UnsupportedOperation(format!(
                    "ResourceManagerConfig::{name} was requested, but no code path implements it"
                )));
            }
        }

        let factor = config.over_provisioning_factor.to_f64().unwrap_or(f64::NAN);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "over_provisioning_factor must be positive and finite, got {factor}"
            )));
        }

        for (name, ceiling) in [
            ("max_cpu_utilization", constraints.max_cpu_utilization),
            ("max_memory_utilization", constraints.max_memory_utilization),
            ("max_gpu_utilization", constraints.max_gpu_utilization),
        ] {
            if !ceiling.is_finite() || ceiling <= 0.0 || ceiling > 1.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "{name} must lie in (0, 1], got {ceiling}"
                )));
            }
        }

        let reserved = &constraints.min_available_resources;
        if reserved.cpu_cores > resource_pool.cpu_cores
            || reserved.memory_mb > resource_pool.memory_mb
            || reserved.gpu_devices > resource_pool.gpu_devices
            || reserved.storage_gb > resource_pool.storage_gb
        {
            return Err(OptimError::InvalidConfig(
                "min_available_resources reserves more than the pool contains, so no allocation \
                 could ever succeed"
                    .to_string(),
            ));
        }

        Ok(Self {
            resource_pool,
            allocation_tracker: ResourceAllocationTracker::new()?,
            allocation_strategy: ResourceAllocationStrategy::BestFit,
            constraints,
            config,
            stats: ResourceStatistics::default(),
        })
    }

    /// The constraints this manager enforces.
    pub fn constraints(&self) -> &ResourceConstraints {
        &self.constraints
    }

    /// The configuration this manager was built with.
    pub fn config(&self) -> &ResourceManagerConfig<T> {
        &self.config
    }

    /// Effective capacity of a countable resource: the pool scaled by
    /// `over_provisioning_factor` and by the resource's utilization ceiling,
    /// less the reserved floor.
    fn effective_capacity(&self, pool: usize, ceiling: f64, reserved: usize) -> usize {
        let factor = self.config.over_provisioning_factor.to_f64().unwrap_or(1.0);
        let scaled = (pool as f64 * factor * ceiling).floor();
        let usable = if scaled.is_finite() && scaled > 0.0 {
            scaled as usize
        } else {
            0
        };
        usable.saturating_sub(reserved)
    }

    /// Allocate resources for a task
    pub fn allocate_resources(&mut self, request: ResourceRequest) -> Result<ResourceAllocation> {
        // Check resource availability against the true free pool (derived from
        // currently-tracked allocations, not a stale utilization snapshot).
        self.check_resource_availability(&request)?;

        // Apply allocation strategy (performs its own free-pool search and can
        // still fail here if a concurrent caller raced us, since this method
        // takes `&mut self` there is no such race within a single manager).
        let allocation = self.apply_allocation_strategy(&request)?;

        // Track the allocation and refresh the utilization snapshot so the
        // free pool used by the next call reflects this allocation.
        self.allocation_tracker
            .track_allocation(&allocation, &self.resource_pool)?;

        // Update statistics
        self.stats.total_allocations += 1;

        Ok(allocation)
    }

    /// Deallocate resources for a task
    pub fn deallocate_resources(&mut self, task_id: &str) -> Result<()> {
        self.allocation_tracker
            .deallocate(task_id, &self.resource_pool)?;
        self.stats.total_deallocations += 1;
        Ok(())
    }

    /// CPU core IDs currently held by any active allocation.
    fn used_cpu_core_ids(&self) -> std::collections::HashSet<usize> {
        self.allocation_tracker
            .current_allocations
            .values()
            .flat_map(|a| a.cpu_cores.iter().copied())
            .collect()
    }

    /// GPU device IDs currently held by any active allocation.
    fn used_gpu_device_ids(&self) -> std::collections::HashSet<usize> {
        self.allocation_tracker
            .current_allocations
            .values()
            .flat_map(|a| a.gpu_devices.iter().copied())
            .collect()
    }

    fn used_memory_mb(&self) -> usize {
        self.allocation_tracker
            .current_allocations
            .values()
            .map(|a| a.memory_mb)
            .sum()
    }

    fn used_storage_gb(&self) -> usize {
        self.allocation_tracker
            .current_allocations
            .values()
            .map(|a| a.storage_gb)
            .sum()
    }

    fn used_network_bandwidth(&self) -> f64 {
        self.allocation_tracker
            .current_allocations
            .values()
            .map(|a| a.network_bandwidth)
            .sum()
    }

    fn used_special_hardware(&self, name: &str) -> usize {
        self.allocation_tracker
            .current_allocations
            .values()
            .filter_map(|a| a.special_hardware.get(name).copied())
            .sum()
    }

    /// Free (unallocated) CPU core IDs, in ascending order.
    fn free_cpu_core_ids(&self) -> Vec<usize> {
        let used = self.used_cpu_core_ids();
        (0..self.resource_pool.cpu_cores)
            .filter(|c| !used.contains(c))
            .collect()
    }

    /// Free (unallocated) GPU device IDs, in ascending order.
    fn free_gpu_device_ids(&self) -> Vec<usize> {
        let used = self.used_gpu_device_ids();
        (0..self.resource_pool.gpu_devices)
            .filter(|g| !used.contains(g))
            .collect()
    }

    /// Get current resource utilization
    pub fn get_utilization(&self) -> UtilizationSnapshot<T> {
        self.allocation_tracker.get_current_utilization()
    }

    /// Re-optimize the current allocation against an objective.
    ///
    /// **Not implemented.** Until 0.3.2 this method ran a
    /// `ResourceOptimizationEngine` whose `optimize` returned the *current*
    /// allocations unchanged with hardcoded `performance_improvement: 0.1`,
    /// `optimization_cost: 0.05` and `confidence: 0.8`, discarded the result
    /// (`let _result = ...`), and returned `Ok(())`. A caller could not tell
    /// that nothing had been optimized. The engine and its `LoadBalancer` /
    /// `LoadPredictor` / `PredictionModel` scaffolding stored fields that no
    /// code path read, so there was nothing to salvage; they were deleted and
    /// this method now says so.
    ///
    /// # Errors
    ///
    /// Always returns [`OptimError::UnsupportedOperation`].
    pub fn optimize_allocation(&mut self, objective: OptimizationObjective) -> Result<()> {
        Err(OptimError::UnsupportedOperation(format!(
            "resource re-optimization for {objective:?} is not implemented: no code path computes \
             or applies a revised allocation. Allocate explicitly with \
             ResourceManager::allocate_resources, or release and re-request."
        )))
    }

    /// Get resource statistics
    pub fn get_statistics(&self) -> &ResourceStatistics<T> {
        &self.stats
    }

    /// Check if resources are available for request, against the real free
    /// pool derived from currently-tracked allocations (not a stale
    /// utilization snapshot). Returns `Err(OptimError::ResourceUnavailable)`
    /// naming the first exhausted resource type.
    fn check_resource_availability(&self, request: &ResourceRequest) -> Result<()> {
        // Constraint ceilings and the reserved floor bound what may be handed
        // out, regardless of what is physically free.
        let cpu_budget = self.effective_capacity(
            self.resource_pool.cpu_cores,
            self.constraints.max_cpu_utilization,
            self.constraints.min_available_resources.cpu_cores,
        );
        let cpu_in_use = self.resource_pool.cpu_cores - self.free_cpu_core_ids().len();
        if cpu_in_use + request.cpu_cores > cpu_budget {
            return Err(OptimError::ResourceUnavailable(format!(
                "CPU constraint exceeded: {} in use plus {} requested exceeds the {} cores allowed \
                 by max_cpu_utilization {} and min_available_resources",
                cpu_in_use,
                request.cpu_cores,
                cpu_budget,
                self.constraints.max_cpu_utilization
            )));
        }

        let memory_budget = self.effective_capacity(
            self.resource_pool.memory_mb,
            self.constraints.max_memory_utilization,
            self.constraints.min_available_resources.memory_mb,
        );
        let memory_in_use = self.used_memory_mb();
        if memory_in_use + request.memory_mb > memory_budget {
            return Err(OptimError::ResourceUnavailable(format!(
                "memory constraint exceeded: {}MB in use plus {}MB requested exceeds the {}MB \
                 allowed by max_memory_utilization {} and min_available_resources",
                memory_in_use,
                request.memory_mb,
                memory_budget,
                self.constraints.max_memory_utilization
            )));
        }

        let gpu_budget = self.effective_capacity(
            self.resource_pool.gpu_devices,
            self.constraints.max_gpu_utilization,
            self.constraints.min_available_resources.gpu_devices,
        );
        let gpu_in_use = self.resource_pool.gpu_devices - self.free_gpu_device_ids().len();
        if gpu_in_use + request.gpu_devices > gpu_budget {
            return Err(OptimError::ResourceUnavailable(format!(
                "GPU constraint exceeded: {} in use plus {} requested exceeds the {} devices \
                 allowed by max_gpu_utilization {} and min_available_resources",
                gpu_in_use, request.gpu_devices, gpu_budget, self.constraints.max_gpu_utilization
            )));
        }

        let free_cpu = self.free_cpu_core_ids().len();
        if free_cpu < request.cpu_cores {
            return Err(OptimError::ResourceUnavailable(format!(
                "insufficient CPU cores: requested {}, free {}",
                request.cpu_cores, free_cpu
            )));
        }

        let free_memory = self
            .resource_pool
            .memory_mb
            .saturating_sub(self.used_memory_mb());
        if free_memory < request.memory_mb {
            return Err(OptimError::ResourceUnavailable(format!(
                "insufficient memory: requested {}MB, free {}MB",
                request.memory_mb, free_memory
            )));
        }

        let free_gpu = self.free_gpu_device_ids().len();
        if free_gpu < request.gpu_devices {
            return Err(OptimError::ResourceUnavailable(format!(
                "insufficient GPU devices: requested {}, free {}",
                request.gpu_devices, free_gpu
            )));
        }

        let free_storage = self
            .resource_pool
            .storage_gb
            .saturating_sub(self.used_storage_gb());
        if free_storage < request.storage_gb {
            return Err(OptimError::ResourceUnavailable(format!(
                "insufficient storage: requested {}GB, free {}GB",
                request.storage_gb, free_storage
            )));
        }

        let free_bandwidth =
            (self.resource_pool.network_bandwidth - self.used_network_bandwidth()).max(0.0);
        if free_bandwidth < request.network_bandwidth {
            return Err(OptimError::ResourceUnavailable(format!(
                "insufficient network bandwidth: requested {}Mbps, free {}Mbps",
                request.network_bandwidth, free_bandwidth
            )));
        }

        for (hw_name, &requested_units) in &request.special_hardware {
            let capacity = self
                .resource_pool
                .special_hardware
                .get(hw_name)
                .copied()
                .unwrap_or(0);
            let free = capacity.saturating_sub(self.used_special_hardware(hw_name));
            if free < requested_units {
                return Err(OptimError::ResourceUnavailable(format!(
                    "insufficient special hardware '{hw_name}': requested {requested_units}, free {free}"
                )));
            }
        }

        Ok(())
    }

    /// Apply allocation strategy. `WorstFit`, `PerformanceOptimized`,
    /// `EnergyEfficient`, `FairShare`, and `PriorityBased` are not yet
    /// implemented as independently-optimized strategies; they fall back to
    /// the honest first-fit search rather than silently echoing the request.
    fn apply_allocation_strategy(&self, request: &ResourceRequest) -> Result<ResourceAllocation> {
        match self.allocation_strategy {
            ResourceAllocationStrategy::BestFit => self.best_fit_allocation(request),
            ResourceAllocationStrategy::FirstFit => self.first_fit_allocation(request),
            _ => self.default_allocation(request),
        }
    }

    /// Best-fit allocation: among free CPU/GPU IDs, prefer the smallest
    /// contiguous block that still satisfies the request (minimizing
    /// fragmentation waste); falls back to draining the largest blocks first
    /// when no single block is big enough. Fails with
    /// `ResourceUnavailable` when the free pool cannot satisfy the request.
    fn best_fit_allocation(&self, request: &ResourceRequest) -> Result<ResourceAllocation> {
        self.check_resource_availability(request)?;

        let free_cpu = self.free_cpu_core_ids();
        let cpu_cores = pick_best_fit(&free_cpu, request.cpu_cores).ok_or_else(|| {
            OptimError::ResourceUnavailable(format!(
                "best-fit: cannot satisfy {} CPU cores from {} free",
                request.cpu_cores,
                free_cpu.len()
            ))
        })?;

        let free_gpu = self.free_gpu_device_ids();
        let gpu_devices = pick_best_fit(&free_gpu, request.gpu_devices).ok_or_else(|| {
            OptimError::ResourceUnavailable(format!(
                "best-fit: cannot satisfy {} GPU devices from {} free",
                request.gpu_devices,
                free_gpu.len()
            ))
        })?;

        Ok(ResourceAllocation {
            task_id: request.task_id.clone(),
            cpu_cores,
            memory_mb: request.memory_mb,
            gpu_devices,
            storage_gb: request.storage_gb,
            network_bandwidth: request.network_bandwidth,
            special_hardware: request.special_hardware.clone(),
            allocated_at: SystemTime::now(),
            expected_release: SystemTime::now() + Duration::from_secs(3600),
            priority: request.priority,
        })
    }

    /// First-fit allocation: takes the lowest-numbered free CPU/GPU IDs in
    /// ascending order, without regard to contiguity. Fails with
    /// `ResourceUnavailable` when the free pool cannot satisfy the request.
    fn first_fit_allocation(&self, request: &ResourceRequest) -> Result<ResourceAllocation> {
        self.check_resource_availability(request)?;

        let free_cpu = self.free_cpu_core_ids();
        let cpu_cores = pick_first_fit(&free_cpu, request.cpu_cores).ok_or_else(|| {
            OptimError::ResourceUnavailable(format!(
                "first-fit: cannot satisfy {} CPU cores from {} free",
                request.cpu_cores,
                free_cpu.len()
            ))
        })?;

        let free_gpu = self.free_gpu_device_ids();
        let gpu_devices = pick_first_fit(&free_gpu, request.gpu_devices).ok_or_else(|| {
            OptimError::ResourceUnavailable(format!(
                "first-fit: cannot satisfy {} GPU devices from {} free",
                request.gpu_devices,
                free_gpu.len()
            ))
        })?;

        Ok(ResourceAllocation {
            task_id: request.task_id.clone(),
            cpu_cores,
            memory_mb: request.memory_mb,
            gpu_devices,
            storage_gb: request.storage_gb,
            network_bandwidth: request.network_bandwidth,
            special_hardware: request.special_hardware.clone(),
            allocated_at: SystemTime::now(),
            expected_release: SystemTime::now() + Duration::from_secs(3600),
            priority: request.priority,
        })
    }

    /// Default allocation strategy for strategies without a dedicated
    /// implementation: behaves exactly like first-fit (real free-pool
    /// search, honest failure on exhaustion) rather than echoing the
    /// request unchecked.
    fn default_allocation(&self, request: &ResourceRequest) -> Result<ResourceAllocation> {
        self.first_fit_allocation(request)
    }
}

// Helper implementations

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> ResourceAllocationTracker<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_allocations: HashMap::new(),
            allocation_history: VecDeque::new(),
            utilization_tracker: UtilizationTracker::new()?,
        })
    }

    /// Record a new allocation and refresh the utilization snapshot so the
    /// free pool used by subsequent availability checks reflects it.
    pub fn track_allocation(
        &mut self,
        allocation: &ResourceAllocation,
        pool: &ResourcePool,
    ) -> Result<()> {
        self.current_allocations
            .insert(allocation.task_id.clone(), allocation.clone());

        let event = AllocationEvent {
            event_type: AllocationEventType::Allocated,
            task_id: allocation.task_id.clone(),
            allocation: allocation.clone(),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        };

        self.allocation_history.push_back(event);
        self.refresh_utilization(pool);
        Ok(())
    }

    /// Release a task's allocation, restoring its resources to the free pool
    /// and refreshing the utilization snapshot. Errors if the task has no
    /// active allocation (nothing to deallocate).
    pub fn deallocate(&mut self, task_id: &str, pool: &ResourcePool) -> Result<()> {
        match self.current_allocations.remove(task_id) {
            Some(allocation) => {
                let event = AllocationEvent {
                    event_type: AllocationEventType::Deallocated,
                    task_id: task_id.to_string(),
                    allocation,
                    timestamp: SystemTime::now(),
                    metadata: HashMap::new(),
                };

                self.allocation_history.push_back(event);
                self.refresh_utilization(pool);
                Ok(())
            }
            None => Err(OptimError::InvalidState(format!(
                "cannot deallocate unknown task '{task_id}': no active allocation found"
            ))),
        }
    }

    pub fn get_current_utilization(&self) -> UtilizationSnapshot<T> {
        self.utilization_tracker.get_current_snapshot()
    }

    pub fn get_current_allocations(&self) -> HashMap<String, ResourceAllocation> {
        self.current_allocations.clone()
    }

    /// Recompute the utilization snapshot fields from the ground-truth set
    /// of current allocations, so `get_current_utilization()` reflects
    /// reality instead of staying pinned at its initial (zero) value.
    fn refresh_utilization(&mut self, pool: &ResourcePool) {
        let mut cpu_used = vec![false; pool.cpu_cores];
        let mut gpu_used = vec![false; pool.gpu_devices];
        let mut memory_used = 0usize;
        let mut storage_used = 0usize;
        let mut bandwidth_used = 0.0f64;

        for alloc in self.current_allocations.values() {
            for &core in &alloc.cpu_cores {
                if let Some(slot) = cpu_used.get_mut(core) {
                    *slot = true;
                }
            }
            for &gpu in &alloc.gpu_devices {
                if let Some(slot) = gpu_used.get_mut(gpu) {
                    *slot = true;
                }
            }
            memory_used += alloc.memory_mb;
            storage_used += alloc.storage_gb;
            bandwidth_used += alloc.network_bandwidth;
        }

        self.utilization_tracker.cpu_utilization = cpu_used
            .iter()
            .map(|&used| if used { T::one() } else { T::zero() })
            .collect();
        self.utilization_tracker.gpu_utilization = gpu_used
            .iter()
            .map(|&used| if used { T::one() } else { T::zero() })
            .collect();
        self.utilization_tracker.memory_utilization = if pool.memory_mb > 0 {
            T::from(memory_used).unwrap_or_else(|| T::zero())
                / T::from(pool.memory_mb).unwrap_or_else(|| T::one())
        } else {
            T::zero()
        };
        self.utilization_tracker.storage_utilization = if pool.storage_gb > 0 {
            T::from(storage_used).unwrap_or_else(|| T::zero())
                / T::from(pool.storage_gb).unwrap_or_else(|| T::one())
        } else {
            T::zero()
        };
        self.utilization_tracker.network_utilization = if pool.network_bandwidth > 0.0 {
            T::from(bandwidth_used / pool.network_bandwidth).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        };
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> UtilizationTracker<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            cpu_utilization: Vec::new(),
            memory_utilization: T::zero(),
            gpu_utilization: Vec::new(),
            storage_utilization: T::zero(),
            network_utilization: T::zero(),
        })
    }

    pub fn get_current_snapshot(&self) -> UtilizationSnapshot<T> {
        UtilizationSnapshot {
            timestamp: SystemTime::now(),
            cpu_utilization: self.cpu_utilization.clone(),
            memory_utilization: self.memory_utilization,
            gpu_utilization: self.gpu_utilization.clone(),
            storage_utilization: self.storage_utilization,
            network_utilization: self.network_utilization,
            overall_utilization: self.calculate_overall_utilization(),
        }
    }

    fn calculate_overall_utilization(&self) -> T {
        // Simplified overall utilization calculation
        (self.memory_utilization + self.storage_utilization + self.network_utilization)
            / T::from(3.0).unwrap_or_else(|| T::zero())
    }
}

// Default implementations

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_utilization: 0.9,
            max_memory_utilization: 0.85,
            max_gpu_utilization: 0.9,
            // Reserve nothing by default. This was `ResourcePool::default()`,
            // which reserves an entire 8-core / 16 GB / 1-GPU machine as the
            // *floor* that must stay free -- so on any pool that size or
            // smaller no allocation could ever succeed. Nothing read the field
            // before 0.3.2, which is why an obviously wrong default went
            // unnoticed.
            min_available_resources: ResourcePool {
                cpu_cores: 0,
                cpu_specs: Vec::new(),
                memory_mb: 0,
                memory_specs: MemorySpec::default(),
                gpu_devices: 0,
                gpu_specs: Vec::new(),
                storage_gb: 0,
                storage_specs: Vec::new(),
                network_bandwidth: 0.0,
                network_specs: NetworkSpec::default(),
                special_hardware: HashMap::new(),
                availability_times: HashMap::new(),
            },
            isolation_requirements: HashMap::new(),
            performance_guarantees: Vec::new(),
        }
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            cpu_specs: Vec::new(),
            memory_mb: 16384,
            memory_specs: MemorySpec::default(),
            gpu_devices: 1,
            gpu_specs: Vec::new(),
            storage_gb: 1000,
            storage_specs: Vec::new(),
            network_bandwidth: 1000.0,
            network_specs: NetworkSpec::default(),
            special_hardware: HashMap::new(),
            availability_times: HashMap::new(),
        }
    }
}

impl Default for MemorySpec {
    fn default() -> Self {
        Self {
            memory_type: "DDR4".to_string(),
            speed_mhz: 3200,
            channels: 2,
            bandwidth_gbps: 51.2,
            ecc_support: false,
        }
    }
}

impl Default for NetworkSpec {
    fn default() -> Self {
        Self {
            network_type: "Ethernet".to_string(),
            max_bandwidth_mbps: 1000.0,
            latency_us: 100.0,
            protocols: vec!["TCP".to_string(), "UDP".to_string()],
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for ResourceStatistics<T> {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            failed_allocations: 0,
            average_allocation_time: Duration::from_millis(5),
            average_utilization_efficiency: T::from(0.7).unwrap_or_else(|| T::zero()),
            fragmentation: T::from(0.1).unwrap_or_else(|| T::zero()),
            load_balance_effectiveness: T::from(0.8).unwrap_or_else(|| T::zero()),
            conflict_resolution_rate: T::from(0.9).unwrap_or_else(|| T::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_pool(cpu_cores: usize, memory_mb: usize, gpu_devices: usize) -> ResourcePool {
        ResourcePool {
            cpu_cores,
            cpu_specs: Vec::new(),
            memory_mb,
            memory_specs: MemorySpec::default(),
            gpu_devices,
            gpu_specs: Vec::new(),
            storage_gb: 1000,
            storage_specs: Vec::new(),
            network_bandwidth: 1000.0,
            network_specs: NetworkSpec::default(),
            special_hardware: HashMap::new(),
            availability_times: HashMap::new(),
        }
    }

    fn test_config() -> ResourceManagerConfig<f64> {
        ResourceManagerConfig {
            allocation_timeout: Duration::from_secs(1),
            monitoring_interval: Duration::from_secs(1),
            optimization_interval: Duration::from_secs(1),
            max_allocation_retries: 3,
            enable_predictive_allocation: false,
            enable_load_balancing: false,
            over_provisioning_factor: 1.0,
            enable_conflict_detection: false,
        }
    }

    fn request(
        task_id: &str,
        cpu_cores: usize,
        memory_mb: usize,
        gpu_devices: usize,
    ) -> ResourceRequest {
        ResourceRequest {
            task_id: task_id.to_string(),
            cpu_cores,
            memory_mb,
            gpu_devices,
            storage_gb: 0,
            network_bandwidth: 0.0,
            special_hardware: HashMap::new(),
            priority: 5,
            deadline: None,
            requested_at: SystemTime::now(),
        }
    }

    /// Constraints that reserve nothing and cap nothing, so a test exercising
    /// *free-pool* exhaustion is not confounded by the utilization ceilings.
    fn unconstrained() -> ResourceConstraints {
        ResourceConstraints {
            max_cpu_utilization: 1.0,
            max_memory_utilization: 1.0,
            max_gpu_utilization: 1.0,
            ..ResourceConstraints::default()
        }
    }

    fn manager(pool: ResourcePool) -> ResourceManager<f64> {
        ResourceManager::<f64>::with_constraints(pool, test_config(), unconstrained())
            .expect("manager construction failed")
    }

    /// The utilization ceilings and the reserved floor must actually bound what
    /// is handed out. Until 0.3.2 `ResourceConstraints` was stored by
    /// `ResourceManager::new` and never read, so every ceiling was inert.
    #[test]
    fn utilization_ceilings_and_reserved_floor_bound_allocation() {
        let constraints = ResourceConstraints {
            max_cpu_utilization: 0.5,
            ..unconstrained()
        };
        let mut mgr = ResourceManager::<f64>::with_constraints(
            small_pool(8, 8192, 0),
            test_config(),
            constraints,
        )
        .expect("manager construction");

        // 4 of 8 cores is exactly the 50% ceiling.
        mgr.allocate_resources(request("task-a", 4, 512, 0))
            .expect("an allocation at the ceiling must succeed");
        // One more core would exceed it, even though 4 cores are physically free.
        let over = mgr.allocate_resources(request("task-b", 1, 512, 0));
        assert!(
            matches!(over, Err(OptimError::ResourceUnavailable(_))),
            "the CPU ceiling must be enforced, got {over:?}"
        );

        // The reserved floor bites the same way.
        let mut reserved = unconstrained();
        reserved.min_available_resources.cpu_cores = 6;
        let mut mgr = ResourceManager::<f64>::with_constraints(
            small_pool(8, 8192, 0),
            test_config(),
            reserved,
        )
        .expect("manager construction");
        mgr.allocate_resources(request("task-a", 2, 512, 0))
            .expect("2 of the 2 unreserved cores must be allocatable");
        assert!(mgr
            .allocate_resources(request("task-b", 1, 512, 0))
            .is_err());
    }

    /// A configuration reserving more than the pool holds can never satisfy any
    /// request, so it is refused at construction rather than failing every call.
    #[test]
    fn an_impossible_reserved_floor_is_refused() {
        let mut impossible = unconstrained();
        impossible.min_available_resources.cpu_cores = 99;
        assert!(matches!(
            ResourceManager::<f64>::with_constraints(
                small_pool(8, 8192, 0),
                test_config(),
                impossible
            ),
            Err(OptimError::InvalidConfig(_))
        ));
    }

    /// Every `enable_*` switch names an unimplemented feature, so setting one
    /// must be refused rather than silently ignored.
    #[test]
    fn unimplemented_manager_switches_are_refused() {
        for apply in [
            (|cfg: &mut ResourceManagerConfig<f64>| cfg.enable_predictive_allocation = true)
                as fn(&mut ResourceManagerConfig<f64>),
            |cfg: &mut ResourceManagerConfig<f64>| cfg.enable_load_balancing = true,
            |cfg: &mut ResourceManagerConfig<f64>| cfg.enable_conflict_detection = true,
        ] {
            let mut config = test_config();
            apply(&mut config);
            assert!(matches!(
                ResourceManager::<f64>::new(small_pool(8, 8192, 0), config),
                Err(OptimError::UnsupportedOperation(_))
            ));
        }

        let mut config = test_config();
        config.over_provisioning_factor = 0.0;
        assert!(matches!(
            ResourceManager::<f64>::new(small_pool(8, 8192, 0), config),
            Err(OptimError::InvalidConfig(_))
        ));
    }

    /// Re-optimization is not implemented and must say so rather than returning
    /// `Ok(())` after discarding an engine's fabricated result.
    #[test]
    fn optimize_allocation_reports_that_it_is_unimplemented() {
        let mut mgr = manager(small_pool(4, 4096, 0));
        assert!(matches!(
            mgr.optimize_allocation(OptimizationObjective::MaximizeUtilization),
            Err(OptimError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn oversubscription_now_fails() {
        // Regression test for F10: the pool has 4 cores total; the first
        // request takes all 4, so a second request for even 1 more core
        // must fail instead of silently succeeding with duplicate IDs.
        let mut mgr = manager(small_pool(4, 8192, 1));

        let first = mgr
            .allocate_resources(request("task-a", 4, 1024, 0))
            .expect("first allocation should succeed");
        assert_eq!(first.cpu_cores.len(), 4);

        let second = mgr.allocate_resources(request("task-b", 1, 1024, 0));
        assert!(
            second.is_err(),
            "allocating beyond the free pool must fail, not silently succeed"
        );
    }

    #[test]
    fn memory_exhaustion_fails() {
        let mut mgr = manager(small_pool(8, 2048, 0));
        mgr.allocate_resources(request("task-a", 1, 2000, 0))
            .expect("first allocation should succeed");

        let result = mgr.allocate_resources(request("task-b", 1, 100, 0));
        assert!(result.is_err(), "memory exhaustion must fail allocation");
    }

    #[test]
    fn decrement_and_restore_round_trip() {
        let mut mgr = manager(small_pool(4, 4096, 0));

        assert_eq!(mgr.free_cpu_core_ids().len(), 4);
        mgr.allocate_resources(request("task-a", 2, 512, 0))
            .expect("allocation should succeed");
        assert_eq!(
            mgr.free_cpu_core_ids().len(),
            2,
            "free pool must decrement by exactly what was allocated"
        );

        mgr.deallocate_resources("task-a")
            .expect("deallocation should succeed");
        assert_eq!(
            mgr.free_cpu_core_ids().len(),
            4,
            "free pool must be fully restored after deallocation"
        );
    }

    #[test]
    fn deallocate_unknown_task_errors() {
        let mut mgr = manager(small_pool(4, 4096, 0));
        let result = mgr.deallocate_resources("never-allocated");
        assert!(result.is_err());
    }

    #[test]
    fn sequential_allocations_do_not_collide() {
        let mut mgr = manager(small_pool(8, 8192, 0));

        let a = mgr
            .allocate_resources(request("task-a", 3, 512, 0))
            .expect("task-a allocation should succeed");
        let b = mgr
            .allocate_resources(request("task-b", 3, 512, 0))
            .expect("task-b allocation should succeed");

        let a_ids: std::collections::HashSet<_> = a.cpu_cores.iter().copied().collect();
        let b_ids: std::collections::HashSet<_> = b.cpu_cores.iter().copied().collect();
        assert!(
            a_ids.is_disjoint(&b_ids),
            "concurrently active allocations must not share CPU core IDs: {a_ids:?} vs {b_ids:?}"
        );
    }

    #[test]
    fn best_fit_prefers_smallest_sufficient_contiguous_block() {
        // Pool of 5 cores (0..5). Hold core 0, then release it after taking
        // core 1 too, leaving free = {0,2,3,4}: a singleton block [0] and a
        // contiguous block [2,3,4] (size 3). A best-fit request for 1 core
        // must prefer the singleton [0] over splitting the larger block.
        let mut mgr = manager(small_pool(5, 8192, 0));
        mgr.allocation_strategy = ResourceAllocationStrategy::FirstFit;

        let hold_a = mgr
            .allocate_resources(request("hold-a", 1, 0, 0))
            .expect("hold-a should succeed"); // takes core 0
        assert_eq!(hold_a.cpu_cores, vec![0]);
        let hold_b = mgr
            .allocate_resources(request("hold-b", 1, 0, 0))
            .expect("hold-b should succeed"); // takes core 1
        assert_eq!(hold_b.cpu_cores, vec![1]);
        mgr.deallocate_resources(&hold_a.task_id)
            .expect("release hold-a");
        // Free set is now {0, 2, 3, 4}.
        assert_eq!(mgr.free_cpu_core_ids(), vec![0, 2, 3, 4]);

        mgr.allocation_strategy = ResourceAllocationStrategy::BestFit;
        let picked = mgr
            .allocate_resources(request("best-fit-pick", 1, 0, 0))
            .expect("best-fit allocation should succeed");
        assert_eq!(
            picked.cpu_cores,
            vec![0],
            "best-fit should prefer the exact-size singleton block over fragmenting the larger block"
        );
    }

    #[test]
    fn pick_first_fit_and_best_fit_behave_differently_on_fragmented_pool() {
        // Free IDs with a fragmented layout: [0] and [2,3,4,5] (two blocks).
        let free_ids = vec![0usize, 2, 3, 4, 5];

        let first = pick_first_fit(&free_ids, 1).expect("first-fit should succeed");
        assert_eq!(first, vec![0]);

        // Best-fit for size 1 should also prefer the smallest sufficient
        // block, which is the singleton [0] (size 1) over the size-4 block.
        let best = pick_best_fit(&free_ids, 1).expect("best-fit should succeed");
        assert_eq!(best, vec![0]);

        // For size 4, only the [2,3,4,5] block suffices.
        let best4 = pick_best_fit(&free_ids, 4).expect("best-fit size 4 should succeed");
        assert_eq!(best4, vec![2, 3, 4, 5]);
    }

    #[test]
    fn pick_functions_return_none_on_exhaustion() {
        let free_ids = vec![0usize, 1, 2];
        assert!(pick_first_fit(&free_ids, 4).is_none());
        assert!(pick_best_fit(&free_ids, 4).is_none());
    }

    #[test]
    fn contiguous_blocks_groups_runs_correctly() {
        let ids = vec![0usize, 1, 2, 5, 6, 9];
        let blocks = contiguous_blocks(&ids);
        assert_eq!(blocks, vec![vec![0, 1, 2], vec![5, 6], vec![9]]);
    }
}
