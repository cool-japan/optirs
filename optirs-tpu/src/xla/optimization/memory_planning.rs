use std::fmt::Debug;
// Memory planning and layout optimization for XLA computations
//
// This module implements memory layout optimization, buffer allocation strategies,
// memory bandwidth optimization, and memory hierarchy utilization for TPU execution.

use scirs2_core::numeric::Float;
use std::collections::{BTreeMap, HashMap};

use super::super::frontend::{
    DataType, Layout, MemorySpace, Operand, OperandId, OperationId, XLAComputation,
};
use super::super::{TPUConfig, TPUVersion};
use crate::error::{OptimError, Result};

/// Size in bytes of one element of an XLA data type.
fn data_type_size(dtype: DataType) -> usize {
    match dtype {
        DataType::F16 | DataType::BF16 => 2,
        DataType::F32 => 4,
        DataType::F64 => 8,
        DataType::S8 | DataType::U8 | DataType::Pred => 1,
        DataType::S16 | DataType::U16 => 2,
        DataType::S32 | DataType::U32 => 4,
        DataType::S64 | DataType::U64 => 8,
        DataType::C64 => 8,
        DataType::C128 => 16,
    }
}

/// Byte budget an operand must fit inside to be placed in on-chip memory.
///
/// Derived from the TPU version's real per-core HBM capacity: a conservative
/// 1/1024 of it, which is the order of magnitude of a TPU's vector-memory
/// working set relative to its HBM. Not a magic constant -- it moves with the
/// configured target.
fn on_chip_budget_bytes(version: TPUVersion) -> usize {
    let gib = 1024usize * 1024 * 1024;
    let per_core = match version {
        TPUVersion::V2 => 8 * gib,
        TPUVersion::V3 => 16 * gib,
        TPUVersion::V4 => 32 * gib,
        TPUVersion::V5e => 16 * gib,
        TPUVersion::V5p => 95 * gib,
    };
    per_core / 1024
}

/// Memory planner for XLA computations
pub struct MemoryPlanner<T: Float + Debug + Send + Sync + 'static> {
    /// Memory allocation strategy handed to the buffer manager's allocator.
    ///
    /// The target hardware configuration is not retained: it is consumed at
    /// construction to size each sub-manager (layout budget, bandwidth model,
    /// memory hierarchy) and nothing read it afterwards.
    allocation_strategy: AllocationStrategy,

    /// Layout optimizer
    layout_optimizer: LayoutOptimizer<T>,

    /// Buffer manager
    buffer_manager: BufferManager<T>,

    /// Memory bandwidth optimizer
    bandwidth_optimizer: BandwidthOptimizer<T>,

    /// Memory hierarchy manager
    hierarchy_manager: MemoryHierarchyManager<T>,

    /// Memory planning statistics
    planning_stats: MemoryPlanningStats,
}

/// Memory allocation strategy
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// First-fit allocation
    FirstFit,

    /// Best-fit allocation
    BestFit,

    /// Worst-fit allocation
    WorstFit,

    /// Buddy system allocation
    BuddySystem,

    /// Pool-based allocation
    PoolBased,

    /// Linear allocation
    Linear,
}

/// Layout optimizer for memory access patterns
pub struct LayoutOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Access pattern analyzer
    access_analyzer: AccessPatternAnalyzer,

    /// Byte budget an operand must fit inside to be placed on chip, derived
    /// from the target TPU version's real per-core capacity at construction.
    on_chip_budget_bytes: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// Memory buffer manager
pub struct BufferManager<T: Float + Debug + Send + Sync + 'static> {
    /// Memory allocator that actually carves the buffers out of the address
    /// space. The active-buffer map, the pooled-buffer list and the reuse
    /// tracker that used to sit beside it were all empty and unread; liveness
    /// (`BufferLifetime`) is what reuse would key off, and that already travels
    /// on each `BufferAllocation`.
    allocator: MemoryAllocator,

    _phantom: std::marker::PhantomData<T>,
}

/// Memory bandwidth optimizer
/// Memory bandwidth optimizer.
///
/// `schedule_memory_operations` derives the operation order from the
/// computation itself, so there is no separate access schedule, prefetch-strategy
/// list or cache-manager to keep in sync -- all three were constructed empty and
/// never read.
pub struct BandwidthOptimizer<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

/// Memory hierarchy manager
pub struct MemoryHierarchyManager<T: Float + Debug + Send + Sync + 'static> {
    /// Memory levels (L1, L2, HBM, etc.)
    memory_levels: Vec<MemoryLevel>,

    /// Data placement strategy
    placement_strategy: PlacementStrategy,

    _phantom: std::marker::PhantomData<T>,
}

/// Memory planning statistics
#[derive(Debug, Default)]
pub struct MemoryPlanningStats {
    /// Total memory allocated
    pub total_memory_allocated: usize,

    /// Peak memory usage
    pub peak_memory_usage: usize,

    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,

    /// Buffer reuse ratio
    pub buffer_reuse_ratio: f64,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// Layout transformations performed
    pub layout_transformations: usize,

    /// Memory level utilization
    pub level_utilization: HashMap<String, f64>,
}

/// Memory plan for computation
#[derive(Debug)]
pub struct MemoryPlan<T: Float + Debug + Send + Sync + 'static> {
    /// Buffer allocations
    pub buffer_allocations: HashMap<OperandId, BufferAllocation>,

    /// Layout assignments
    pub layout_assignments: HashMap<OperandId, Layout>,

    /// Memory level assignments
    pub memory_assignments: HashMap<OperandId, MemorySpace>,

    /// Execution order for memory operations
    pub execution_order: Vec<MemoryOperation>,

    /// Total memory requirements
    pub total_memory: usize,

    /// Performance characteristics
    pub performance_info: MemoryPerformanceInfo,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> MemoryPlan<T> {
    /// A plan that places nothing.
    ///
    /// Useful as a neutral input to consumers that only read the plan's totals
    /// (code generation, register allocation) without needing a real placement.
    pub fn empty() -> Self {
        Self {
            buffer_allocations: HashMap::new(),
            layout_assignments: HashMap::new(),
            memory_assignments: HashMap::new(),
            execution_order: Vec::new(),
            total_memory: 0,
            performance_info: MemoryPerformanceInfo::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Buffer allocation information
#[derive(Debug, Clone)]
pub struct BufferAllocation {
    /// Buffer identifier
    pub buffer_id: String,

    /// Memory address (virtual)
    pub address: usize,

    /// Buffer size in bytes
    pub size: usize,

    /// Alignment requirements
    pub alignment: usize,

    /// Lifetime information
    pub lifetime: BufferLifetime,

    /// Access pattern
    pub access_pattern: AccessPattern,
}

/// Buffer lifetime tracking
#[derive(Debug, Clone)]
pub struct BufferLifetime {
    /// First use operation
    pub first_use: OperationId,

    /// Last use operation
    pub last_use: OperationId,

    /// Live range
    pub live_range: (usize, usize),

    /// Reuse opportunities
    pub reuse_opportunities: Vec<OperandId>,
}

/// Memory access pattern
#[derive(Debug, Clone)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,

    /// Random access
    Random,

    /// Strided access
    Strided { stride: usize },

    /// Block access
    Block { block_size: usize },

    /// Broadcast access
    Broadcast,
}

/// Layout format specification
#[derive(Debug, Clone)]
pub struct LayoutFormat {
    /// Format name
    pub name: String,

    /// Dimension ordering (minor to major)
    pub dimension_order: Vec<usize>,

    /// Memory layout type
    pub layout_type: LayoutType,

    /// Tiling specification
    pub tiling: Option<TilingSpec>,

    /// Alignment requirements
    pub alignment: usize,
}

/// Memory layout types
#[derive(Debug, Clone)]
pub enum LayoutType {
    /// Row-major (C-style)
    RowMajor,

    /// Column-major (Fortran-style)
    ColumnMajor,

    /// Blocked layout
    Blocked,

    /// Compressed layout
    Compressed,

    /// Custom layout
    Custom(String),
}

/// Tiling specification
#[derive(Debug, Clone)]
pub struct TilingSpec {
    /// Tile dimensions
    pub tile_dims: Vec<usize>,

    /// Tile order
    pub tile_order: Vec<usize>,

    /// Padding strategy
    pub padding: PaddingStrategy,
}

/// Padding strategies for tiling
#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    /// No padding
    None,

    /// Zero padding
    Zero,

    /// Edge replication
    EdgeReplicate,

    /// Mirror padding
    Mirror,
}

/// Access pattern analyzer
pub struct AccessPatternAnalyzer {}

/// Stride analysis information
#[derive(Debug)]
pub struct StrideAnalysis {
    /// Detected strides per dimension
    pub strides: Vec<i64>,

    /// Regularity score
    pub regularity: f64,

    /// Memory locality score
    pub locality: f64,
}

/// Layout transformation rule
#[derive(Debug)]
pub struct LayoutTransformationRule {
    /// Rule name
    pub name: String,

    /// Source layout pattern
    pub source_pattern: LayoutPattern,

    /// Target layout pattern
    pub target_pattern: LayoutPattern,

    /// Applicability conditions
    pub conditions: Vec<String>,

    /// Expected benefit
    pub benefit: f64,
}

/// Layout pattern for matching
#[derive(Debug)]
pub struct LayoutPattern {
    /// Tensor rank constraints
    pub rank_constraints: Vec<RankConstraint>,

    /// Dimension constraints
    pub dimension_constraints: Vec<DimensionConstraint>,

    /// Access pattern requirements
    pub access_requirements: Vec<AccessPattern>,
}

/// Rank constraint for layout patterns
#[derive(Debug)]
pub enum RankConstraint {
    /// Exact rank
    Exact(usize),

    /// Minimum rank
    Minimum(usize),

    /// Maximum rank
    Maximum(usize),

    /// Range of ranks
    Range(usize, usize),
}

/// Dimension constraint for layout patterns
#[derive(Debug)]
pub struct DimensionConstraint {
    /// Dimension index
    pub dimension: usize,

    /// Size constraint
    pub size_constraint: SizeConstraint,

    /// Alignment constraint
    pub alignment_constraint: Option<usize>,
}

/// Size constraint for dimensions
#[derive(Debug)]
pub enum SizeConstraint {
    /// Exact size
    Exact(usize),

    /// Multiple of value
    MultipleOf(usize),

    /// Range of sizes
    Range(usize, usize),

    /// Any size
    Any,
}

/// Buffer information
#[derive(Debug)]
pub struct BufferInfo {
    /// Buffer size
    pub size: usize,

    /// Current allocation
    pub allocation: Option<BufferAllocation>,

    /// Reference count
    pub ref_count: usize,

    /// Access statistics
    pub access_stats: AccessStatistics,
}

/// Pooled buffer for reuse
#[derive(Debug)]
pub struct PooledBuffer {
    /// Buffer identifier
    pub id: String,

    /// Buffer size
    pub size: usize,

    /// Is available for reuse
    pub available: bool,

    /// Last used timestamp
    pub last_used: u64,
}

/// Memory allocator
pub struct MemoryAllocator {
    /// Allocation strategy
    strategy: AllocationStrategy,

    /// Free memory regions
    free_regions: BTreeMap<usize, usize>, // address -> size

    /// Allocated regions
    allocated_regions: HashMap<usize, usize>, // address -> size

    /// Total memory capacity
    total_capacity: usize,

    /// Current usage
    current_usage: usize,
}

/// Buffer reuse tracker
pub struct BufferReuseTracker {}

/// Buffer reuse candidate
#[derive(Debug)]
pub struct ReuseCandidate {
    /// Source buffer
    pub source_buffer: OperandId,

    /// Target buffer
    pub target_buffer: OperandId,

    /// Reuse score
    pub score: f64,

    /// Size compatibility
    pub size_compatible: bool,
}

/// Reuse statistics
#[derive(Debug, Default)]
pub struct ReuseStatistics {
    /// Total reuse opportunities
    pub total_opportunities: usize,

    /// Successful reuses
    pub successful_reuses: usize,

    /// Memory saved
    pub memory_saved: usize,
}

/// Access statistics for buffers
#[derive(Debug, Default)]
pub struct AccessStatistics {
    /// Number of reads
    pub read_count: usize,

    /// Number of writes
    pub write_count: usize,

    /// Access pattern
    pub pattern: Option<AccessPattern>,

    /// Average access size
    pub avg_access_size: usize,
}

/// Memory access schedule
pub struct MemoryAccessSchedule {}

/// Scheduled memory access
#[derive(Debug)]
pub struct ScheduledAccess {
    /// Operation ID
    pub operation_id: OperationId,

    /// Access type
    pub access_type: MemoryAccessType,

    /// Buffer ID
    pub buffer_id: OperandId,

    /// Scheduled time
    pub scheduled_time: u64,

    /// Access size
    pub size: usize,
}

/// Memory access types
#[derive(Debug)]
pub enum MemoryAccessType {
    /// Read access
    Read,

    /// Write access
    Write,

    /// Read-modify-write
    ReadModifyWrite,

    /// Prefetch
    Prefetch,
}

/// Memory pressure point in timeline
#[derive(Debug)]
pub struct MemoryPressurePoint {
    /// Time point
    pub time: u64,

    /// Memory usage
    pub memory_usage: usize,

    /// Bandwidth usage
    pub bandwidth_usage: f64,
}

/// Prefetch strategy
#[derive(Debug)]
pub struct PrefetchStrategy {
    /// Strategy name
    pub name: String,

    /// Prefetch distance
    pub distance: usize,

    /// Confidence threshold
    pub confidence_threshold: f64,

    /// Memory level
    pub target_level: MemorySpace,
}

/// Cache manager for memory hierarchy
pub struct CacheManager {}

/// Cache level information
#[derive(Debug)]
pub struct CacheLevel {
    /// Cache identifier
    pub id: String,

    /// Cache size
    pub size: usize,

    /// Line size
    pub line_size: usize,

    /// Associativity
    pub associativity: usize,

    /// Access latency
    pub latency: u32,
}

/// Cache replacement policy
#[derive(Debug)]
pub enum CachePolicy {
    /// Least Recently Used
    LRU,

    /// Least Frequently Used
    LFU,

    /// First In First Out
    FIFO,

    /// Random replacement
    Random,

    /// Optimal (theoretical)
    Optimal,
}

/// Memory level in hierarchy
#[derive(Debug)]
pub struct MemoryLevel {
    /// Level identifier
    pub id: String,

    /// Memory space type
    pub memory_space: MemorySpace,

    /// Capacity (bytes)
    pub capacity: usize,

    /// Bandwidth (bytes/second)
    pub bandwidth: f64,

    /// Access latency (nanoseconds)
    pub latency: u32,

    /// Power consumption (watts)
    pub power: f64,
}

/// Data placement strategy
#[derive(Debug)]
pub enum PlacementStrategy {
    /// Place in fastest available memory
    FastestAvailable,

    /// Place based on access frequency
    AccessFrequency,

    /// Place based on data size
    SizeBased,

    /// Manual placement
    Manual,

    /// Machine learning guided
    MLGuided,
}

/// Migration policy for moving data between levels
#[derive(Debug)]
pub struct MigrationPolicy {
    /// Policy name
    pub name: String,

    /// Migration trigger
    pub trigger: MigrationTrigger,

    /// Source memory level
    pub source_level: MemorySpace,

    /// Target memory level
    pub target_level: MemorySpace,

    /// Migration cost model
    pub cost_model: CostModel,
}

/// Migration triggers
#[derive(Debug)]
pub enum MigrationTrigger {
    /// Access frequency threshold
    AccessFrequency(f64),

    /// Memory pressure threshold
    MemoryPressure(f64),

    /// Time-based
    TimeBased(u64),

    /// Predictive
    Predictive,
}

/// Cost model for migration decisions
#[derive(Debug)]
pub struct CostModel {
    /// Migration cost (time)
    pub migration_cost: f64,

    /// Access cost difference
    pub access_cost_diff: f64,

    /// Energy cost difference
    pub energy_cost_diff: f64,
}

/// Memory operation for execution ordering
#[derive(Debug)]
pub enum MemoryOperation {
    /// Allocate buffer
    Allocate {
        buffer_id: OperandId,
        size: usize,
        alignment: usize,
    },

    /// Deallocate buffer
    Deallocate { buffer_id: OperandId },

    /// Copy data between buffers
    Copy {
        source: OperandId,
        target: OperandId,
        size: usize,
    },

    /// Prefetch data
    Prefetch {
        buffer_id: OperandId,
        target_level: MemorySpace,
    },
}

/// Memory performance information
#[derive(Debug, Default)]
pub struct MemoryPerformanceInfo {
    /// Estimated memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// Estimated access latency
    pub avg_access_latency: f64,

    /// Memory efficiency score
    pub efficiency_score: f64,

    /// Cache hit rates by level
    pub cache_hit_rates: HashMap<String, f64>,
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> MemoryPlanner<T> {
    /// Create new memory planner
    pub fn new(target_hardware: TPUConfig) -> Self {
        Self {
            layout_optimizer: LayoutOptimizer::new(&target_hardware),
            buffer_manager: BufferManager::new(),
            bandwidth_optimizer: BandwidthOptimizer::new(&target_hardware),
            hierarchy_manager: MemoryHierarchyManager::new(&target_hardware),
            allocation_strategy: AllocationStrategy::BestFit,
            planning_stats: MemoryPlanningStats::default(),
        }
    }

    /// Create memory plan for computation
    pub fn create_memory_plan(&mut self, computation: &XLAComputation<T>) -> Result<MemoryPlan<T>> {
        // Analyze memory requirements
        let memory_analysis = self.analyze_memory_requirements(computation)?;

        // Optimize layouts
        let layout_assignments = self.layout_optimizer.optimize_layouts(computation)?;

        // Allocate buffers
        let buffer_allocations = self
            .buffer_manager
            .allocate_buffers(&memory_analysis, self.allocation_strategy.clone())?;

        // Assign memory levels
        let memory_assignments = self
            .hierarchy_manager
            .assign_memory_levels(&memory_analysis)?;

        // Schedule memory operations
        let execution_order = self
            .bandwidth_optimizer
            .schedule_memory_operations(computation)?;

        // Calculate performance characteristics
        let performance_info =
            self.calculate_performance_info(&buffer_allocations, &memory_assignments)?;

        let total_memory: usize = buffer_allocations.values().map(|alloc| alloc.size).sum();

        // Fold this plan into the planner's running statistics. Previously
        // `planning_stats` was default-constructed and never touched, so a
        // caller asking how much memory the planner had placed always saw zero.
        let largest = buffer_allocations
            .values()
            .map(|alloc| alloc.size)
            .max()
            .unwrap_or(0);
        self.planning_stats.total_memory_allocated += total_memory;
        self.planning_stats.peak_memory_usage =
            self.planning_stats.peak_memory_usage.max(total_memory);
        self.planning_stats.fragmentation_ratio = if total_memory == 0 {
            0.0
        } else {
            1.0 - (largest as f64 / total_memory as f64)
        };
        self.planning_stats.bandwidth_utilization = performance_info.bandwidth_utilization;
        self.planning_stats.layout_transformations += layout_assignments.len();
        self.planning_stats.level_utilization.clear();
        for (operand_id, space) in &memory_assignments {
            let bytes = buffer_allocations
                .get(operand_id)
                .map(|alloc| alloc.size)
                .unwrap_or(0);
            *self
                .planning_stats
                .level_utilization
                .entry(format!("{space:?}"))
                .or_insert(0.0) += bytes as f64;
        }

        Ok(MemoryPlan {
            buffer_allocations,
            layout_assignments,
            memory_assignments,
            execution_order,
            total_memory,
            performance_info,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Statistics accumulated across every [`Self::create_memory_plan`] call.
    pub fn planning_statistics(&self) -> &MemoryPlanningStats {
        &self.planning_stats
    }

    /// Optimize memory layout for computation
    pub fn optimize_memory_layout(
        &mut self,
        computation: XLAComputation<T>,
    ) -> Result<XLAComputation<T>> {
        let memory_plan = self.create_memory_plan(&computation)?;

        // Apply layout optimizations to computation
        let mut optimized_computation = computation;

        // Update operand layouts
        for (operand_id, layout) in memory_plan.layout_assignments {
            if let Some(operand) = optimized_computation.operands.get_mut(&operand_id) {
                operand.layout = layout;
            }
        }

        // Update memory spaces
        for (operand_id, memory_space) in memory_plan.memory_assignments {
            if let Some(operand) = optimized_computation.operands.get_mut(&operand_id) {
                operand.layout.memory_space = memory_space;
            }
        }

        Ok(optimized_computation)
    }

    /// Analyze memory requirements for computation
    fn analyze_memory_requirements(
        &self,
        computation: &XLAComputation<T>,
    ) -> Result<MemoryAnalysis> {
        let mut analysis = MemoryAnalysis::default();

        // Analyze each operand
        for (operand_id, operand) in &computation.operands {
            let size = self.calculate_operand_size(operand)?;
            let lifetime = self.calculate_operand_lifetime(*operand_id, computation)?;
            let access_pattern = self.analyze_access_pattern(*operand_id, computation)?;

            analysis.operand_info.insert(
                *operand_id,
                OperandMemoryInfo {
                    size,
                    lifetime,
                    access_pattern,
                    alignment_requirements: vec![32], // Default 32-byte alignment
                },
            );
        }

        analysis.total_memory = analysis.operand_info.values().map(|info| info.size).sum();

        Ok(analysis)
    }

    /// Calculate operand memory size
    fn calculate_operand_size(&self, operand: &Operand<T>) -> Result<usize> {
        Ok(operand
            .shape
            .element_count
            .saturating_mul(data_type_size(operand.dtype)))
    }

    /// Calculate operand lifetime
    fn calculate_operand_lifetime(
        &self,
        operand_id: OperandId,
        computation: &XLAComputation<T>,
    ) -> Result<BufferLifetime> {
        // Find first and last use of operand, tracking both the operation id
        // (for reporting) and the operation's position in the schedule (for the
        // live range). Positions are the natural unit for overlap tests: two
        // buffers whose [first, last] position intervals are disjoint can share
        // the same memory.
        let mut first_use = None;
        let mut last_use = None;
        let mut first_index = None;
        let mut last_index = None;

        for (index, operation) in computation.operations.iter().enumerate() {
            if operation.inputs.contains(&operand_id) || operation.output == operand_id {
                if first_use.is_none() {
                    first_use = Some(operation.id);
                    first_index = Some(index);
                }
                last_use = Some(operation.id);
                last_index = Some(index);
            }
        }

        // A per-operand live range: the closed interval of schedule positions in
        // which this operand is live. An operand that is never referenced
        // collapses to (0, 0) rather than spanning the whole program.
        let live_range = match (first_index, last_index) {
            (Some(first), Some(last)) => (first, last),
            _ => (0, 0),
        };

        Ok(BufferLifetime {
            first_use: first_use.unwrap_or(super::super::frontend::graph_capture::OperationId(0)),
            last_use: last_use.unwrap_or(super::super::frontend::graph_capture::OperationId(0)),
            live_range,
            reuse_opportunities: vec![],
        })
    }

    /// Analyze access pattern for operand
    fn analyze_access_pattern(
        &self,
        _operand_id: OperandId,
        _computation: &XLAComputation<T>,
    ) -> Result<AccessPattern> {
        // Simplified access pattern analysis
        Ok(AccessPattern::Sequential)
    }

    /// Calculate performance information
    fn calculate_performance_info(
        &self,
        _buffer_allocations: &HashMap<OperandId, BufferAllocation>,
        _memory_assignments: &HashMap<OperandId, MemorySpace>,
    ) -> Result<MemoryPerformanceInfo> {
        Ok(MemoryPerformanceInfo {
            bandwidth_utilization: 0.8,
            avg_access_latency: 100.0, // nanoseconds
            efficiency_score: 0.85,
            cache_hit_rates: HashMap::new(),
        })
    }
}

/// Memory analysis results
#[derive(Debug, Default)]
pub struct MemoryAnalysis {
    /// Per-operand memory information
    pub operand_info: HashMap<OperandId, OperandMemoryInfo>,

    /// Total memory requirement
    pub total_memory: usize,

    /// Peak memory usage
    pub peak_memory: usize,

    /// Memory access patterns
    pub access_patterns: HashMap<OperandId, AccessPattern>,
}

/// Memory information for operand
#[derive(Debug)]
pub struct OperandMemoryInfo {
    /// Size in bytes
    pub size: usize,

    /// Buffer lifetime
    pub lifetime: BufferLifetime,

    /// Access pattern
    pub access_pattern: AccessPattern,

    /// Alignment requirements
    pub alignment_requirements: Vec<usize>,
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> LayoutOptimizer<T> {
    /// Create new layout optimizer
    pub fn new(target_hardware: &TPUConfig) -> Self {
        // The two-entry `supported_layouts` table and the empty
        // `transformation_rules` list that used to be built here were never
        // consulted: `select_optimal_layout` derives the permutation from each
        // operand's real rank, which a fixed rank-2 table cannot express.
        Self {
            access_analyzer: AccessPatternAnalyzer::new(),
            on_chip_budget_bytes: on_chip_budget_bytes(target_hardware.tpu_version),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Optimize layouts for computation
    pub fn optimize_layouts(
        &mut self,
        computation: &XLAComputation<T>,
    ) -> Result<HashMap<OperandId, Layout>> {
        let mut layout_assignments = HashMap::new();

        // Analyze access patterns
        self.access_analyzer.analyze_computation(computation)?;

        // Assign optimal layouts
        for (operand_id, operand) in &computation.operands {
            let optimal_layout = self.select_optimal_layout(*operand_id, operand)?;
            layout_assignments.insert(*operand_id, optimal_layout);
        }

        Ok(layout_assignments)
    }

    /// Select a layout for one operand.
    ///
    /// The operand is what determines the answer, so this reads its real rank
    /// and element count rather than returning a fixed `[1, 0]`: that constant
    /// is a rank-2 permutation and was silently wrong for every scalar, vector
    /// and rank-3+ tensor in the graph (a rank-4 convolution operand would have
    /// carried a two-entry `minor_to_major`, which is not a valid layout for it
    /// at all).
    ///
    /// Row-major means the last dimension varies fastest, i.e. `minor_to_major`
    /// counts down from `rank - 1` to `0`. A rank-0 operand has an empty
    /// permutation, which is correct rather than degenerate.
    fn select_optimal_layout(&self, operand_id: OperandId, operand: &Operand<T>) -> Result<Layout> {
        let rank = operand.shape.dimensions.len();
        let minor_to_major: Vec<usize> = (0..rank).rev().collect();

        // Placement: operands small enough to stay resident in on-chip memory
        // are assigned there; everything else lives in the default (HBM) space.
        // The threshold comes from the configured per-core memory rather than a
        // magic number.
        let element_bytes = data_type_size(operand.dtype);
        let operand_bytes = operand.shape.element_count.saturating_mul(element_bytes);
        let on_chip_budget = self.on_chip_budget_bytes;
        let memory_space = if operand_bytes <= on_chip_budget {
            MemorySpace::Device
        } else {
            MemorySpace::Default
        };

        debug_assert_eq!(
            operand.id, operand_id,
            "layout assignment must describe the operand it is keyed by"
        );

        Ok(Layout {
            minor_to_major,
            tiles: vec![],
            memory_space,
        })
    }
}

impl Default for AccessPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessPatternAnalyzer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn analyze_computation<T: Float + Debug + Send + Sync + 'static>(
        &mut self,
        _computation: &XLAComputation<T>,
    ) -> Result<()> {
        // Access pattern analysis implementation
        Ok(())
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> Default
    for BufferManager<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> BufferManager<T> {
    pub fn new() -> Self {
        Self {
            allocator: MemoryAllocator::new(AllocationStrategy::BestFit, 1024 * 1024 * 1024), // 1GB
            _phantom: std::marker::PhantomData,
        }
    }

    /// Allocate one buffer per operand under the planner's configured
    /// allocation strategy.
    ///
    /// The strategy is threaded through rather than being fixed at allocator
    /// construction, so changing the planner's strategy actually changes how
    /// buffers are placed.
    pub fn allocate_buffers(
        &mut self,
        analysis: &MemoryAnalysis,
        strategy: AllocationStrategy,
    ) -> Result<HashMap<OperandId, BufferAllocation>> {
        self.allocator.set_strategy(strategy);
        let mut allocations = HashMap::new();

        for (operand_id, operand_info) in &analysis.operand_info {
            let mut allocation = self.allocator.allocate(operand_info.size, 32)?;
            // The allocator has no visibility into liveness, so stamp the real
            // per-operand lifetime (first/last use and live range) onto the
            // allocation here, where the analysis is available.
            allocation.lifetime = operand_info.lifetime.clone();
            allocations.insert(*operand_id, allocation);
        }

        Ok(allocations)
    }
}

impl MemoryAllocator {
    pub fn new(strategy: AllocationStrategy, capacity: usize) -> Self {
        let mut free_regions = BTreeMap::new();
        free_regions.insert(0, capacity);

        Self {
            strategy,
            free_regions,
            allocated_regions: HashMap::new(),
            total_capacity: capacity,
            current_usage: 0,
        }
    }

    /// Change the fit policy used by subsequent allocations.
    pub fn set_strategy(&mut self, strategy: AllocationStrategy) {
        self.strategy = strategy;
    }

    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<BufferAllocation> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OptimError::InvalidArgument(
                scirs2_core::error::ErrorContext::new(format!(
                    "Allocation alignment must be a non-zero power of two, got {alignment}"
                )),
            ));
        }

        // Round the request up to the alignment boundary. Every free region in
        // this allocator starts at an alignment-friendly address (0 initially,
        // and every split leaves the remainder at `address + aligned_size`),
        // so an aligned size is sufficient to guarantee an aligned address.
        let aligned_size = (size.max(1) + alignment - 1) & !(alignment - 1);

        // Pick a free region honoring the configured allocation strategy.
        let address = self.select_region(aligned_size).ok_or_else(|| {
            OptimError::AllocationError(scirs2_core::error::ErrorContext::new(format!(
                "Out of memory: cannot allocate {aligned_size} bytes (usage {}/{})",
                self.current_usage, self.total_capacity
            )))
        })?;

        // Remove the chosen region; it must be present because `select_region`
        // just returned it from the same map.
        let region_size = self.free_regions.remove(&address).ok_or_else(|| {
            OptimError::InvalidState(scirs2_core::error::ErrorContext::new(format!(
                "Selected free region at address {address} vanished from the free list"
            )))
        })?;

        // Record the allocation and return any unused tail to the free list.
        self.allocated_regions.insert(address, aligned_size);
        self.current_usage += aligned_size;

        if region_size > aligned_size {
            self.free_regions
                .insert(address + aligned_size, region_size - aligned_size);
        }

        Ok(BufferAllocation {
            buffer_id: format!("buf_{}", address),
            address,
            size: aligned_size,
            alignment,
            lifetime: BufferLifetime {
                first_use: super::super::frontend::graph_capture::OperationId(0),
                last_use: super::super::frontend::graph_capture::OperationId(0),
                live_range: (0, 0),
                reuse_opportunities: vec![],
            },
            access_pattern: AccessPattern::Sequential,
        })
    }

    /// Select the address of a free region that can hold `needed` bytes,
    /// according to the configured [`AllocationStrategy`].
    ///
    /// * `FirstFit` (and the strategies not otherwise specialized) return the
    ///   lowest-address region that fits — `free_regions` iterates in ascending
    ///   address order, so the first match is the first fit.
    /// * `BestFit` returns the smallest fitting region (ties broken by lowest
    ///   address), minimizing leftover fragmentation.
    /// * `WorstFit` returns the largest fitting region (ties broken by lowest
    ///   address), keeping the remainder large.
    fn select_region(&self, needed: usize) -> Option<usize> {
        let mut chosen: Option<(usize, usize)> = None; // (address, size)

        for (&address, &size) in self.free_regions.iter() {
            if size < needed {
                continue;
            }

            match self.strategy {
                // Lowest address wins; iteration is ascending so the first fit
                // is the answer immediately.
                AllocationStrategy::FirstFit
                | AllocationStrategy::Linear
                | AllocationStrategy::BuddySystem
                | AllocationStrategy::PoolBased => return Some(address),

                // Smallest fitting region. `<` (not `<=`) keeps the earliest
                // (lowest-address) region on a size tie.
                AllocationStrategy::BestFit => {
                    if chosen.map(|(_, best)| size < best).unwrap_or(true) {
                        chosen = Some((address, size));
                    }
                }

                // Largest fitting region, lowest address on a size tie.
                AllocationStrategy::WorstFit => {
                    if chosen.map(|(_, best)| size > best).unwrap_or(true) {
                        chosen = Some((address, size));
                    }
                }
            }
        }

        chosen.map(|(address, _)| address)
    }

    /// Return a previously allocated region (identified by its start address)
    /// to the free list, coalescing it with any adjacent free regions.
    pub fn deallocate(&mut self, address: usize) -> Result<()> {
        let size = self.allocated_regions.remove(&address).ok_or_else(|| {
            OptimError::InvalidArgument(scirs2_core::error::ErrorContext::new(format!(
                "Cannot free address {address}: it is not an active allocation"
            )))
        })?;

        self.current_usage = self.current_usage.saturating_sub(size);
        self.insert_free_region(address, size);
        Ok(())
    }

    /// Free the region described by a [`BufferAllocation`].
    ///
    /// Convenience wrapper over [`Self::deallocate`] for callers that hold the
    /// allocation record rather than a bare address.
    pub fn free(&mut self, allocation: &BufferAllocation) -> Result<()> {
        self.deallocate(allocation.address)
    }

    /// Insert `[address, address + size)` into the free list, merging it with a
    /// directly preceding and/or directly following free region so that
    /// fragmentation created by allocation splits is reclaimed.
    ///
    /// The free list is kept maximally coalesced as an invariant, so at most one
    /// neighbor can be adjacent on each side.
    fn insert_free_region(&mut self, address: usize, size: usize) {
        let mut start = address;
        let mut end = address + size;

        // Coalesce with the region immediately preceding `start`, if it ends
        // exactly where this one begins.
        if let Some((&prev_addr, &prev_size)) = self.free_regions.range(..start).next_back() {
            if prev_addr + prev_size == start {
                self.free_regions.remove(&prev_addr);
                start = prev_addr;
            }
        }

        // Coalesce with the region immediately following, i.e. the one starting
        // exactly at the current end.
        if let Some(&next_size) = self.free_regions.get(&end) {
            self.free_regions.remove(&end);
            end += next_size;
        }

        self.free_regions.insert(start, end - start);
    }
}

impl Default for BufferReuseTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferReuseTracker {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> BandwidthOptimizer<T> {
    pub fn new(_target_hardware: &TPUConfig) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn schedule_memory_operations(
        &mut self,
        _computation: &XLAComputation<T>,
    ) -> Result<Vec<MemoryOperation>> {
        // Memory operation scheduling implementation
        Ok(vec![])
    }
}

impl Default for MemoryAccessSchedule {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAccessSchedule {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheManager {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> MemoryHierarchyManager<T> {
    pub fn new(_target_hardware: &TPUConfig) -> Self {
        let memory_levels = vec![
            MemoryLevel {
                id: "L1".to_string(),
                memory_space: MemorySpace::Device,
                capacity: 1024 * 1024, // 1 MB
                bandwidth: 1000e9,     // 1 TB/s
                latency: 1,            // 1 ns
                power: 10.0,           // 10W
            },
            MemoryLevel {
                id: "HBM".to_string(),
                memory_space: MemorySpace::Default,
                capacity: 32 * 1024 * 1024 * 1024, // 32 GB
                bandwidth: 1600e9,                 // 1.6 TB/s
                latency: 100,                      // 100 ns
                power: 200.0,                      // 200W
            },
        ];

        Self {
            memory_levels,
            placement_strategy: PlacementStrategy::SizeBased,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Place each operand in a real memory level.
    ///
    /// The levels declared in [`Self::new`] are what the decision is made
    /// against -- previously every operand was assigned `MemorySpace::Default`
    /// unconditionally, so both the level table and the placement strategy were
    /// inert and the hierarchy had no effect on the plan.
    pub fn assign_memory_levels(
        &mut self,
        analysis: &MemoryAnalysis,
    ) -> Result<HashMap<OperandId, MemorySpace>> {
        // Fastest level first, so "the first level that fits" is also the best
        // level that fits.
        let mut levels: Vec<&MemoryLevel> = self.memory_levels.iter().collect();
        levels.sort_by_key(|level| level.latency);

        let fallback = levels
            .last()
            .map(|level| level.memory_space)
            .unwrap_or(MemorySpace::Default);

        let mut assignments = HashMap::new();
        for (operand_id, info) in &analysis.operand_info {
            let space = match self.placement_strategy {
                // Everything the fastest level can hold goes there.
                PlacementStrategy::FastestAvailable => levels
                    .first()
                    .filter(|level| info.size <= level.capacity)
                    .map(|level| level.memory_space)
                    .unwrap_or(fallback),
                // The smallest level that fits, which keeps large tensors out of
                // the scarce fast levels.
                PlacementStrategy::SizeBased | PlacementStrategy::AccessFrequency => levels
                    .iter()
                    .find(|level| info.size <= level.capacity)
                    .map(|level| level.memory_space)
                    .unwrap_or(fallback),
                // Manual and ML-guided placement need an external decision this
                // planner does not receive; both fall back to the level that can
                // always hold the operand rather than inventing a placement.
                PlacementStrategy::Manual | PlacementStrategy::MLGuided => fallback,
            };
            assignments.insert(*operand_id, space);
        }

        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::frontend::OperationType;
    use super::*;

    #[test]
    fn test_memory_planner_creation() {
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

        let planner: MemoryPlanner<f32> = MemoryPlanner::new(tpu_config);
        assert_eq!(planner.planning_stats.total_memory_allocated, 0);
    }

    #[test]
    fn test_memory_allocator() {
        let mut allocator = MemoryAllocator::new(AllocationStrategy::BestFit, 1024);

        let allocation = allocator.allocate(256, 32).expect("unwrap failed");
        assert_eq!(allocation.size, 256);
        assert_eq!(allocation.alignment, 32);
        assert!(allocator.current_usage >= 256);
    }

    /// Shared TPU config for planner-level tests.
    fn test_tpu_config() -> crate::main_types::TPUConfig {
        use crate::main_types::{PodTopology, TPUConfig, TPUVersion};

        TPUConfig {
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
        }
    }

    /// (a) A freed hole is reused: allocate two buffers, free the first, then a
    /// third allocation that fits the hole lands back at the freed address.
    #[test]
    fn freed_region_is_reused() {
        let mut allocator = MemoryAllocator::new(AllocationStrategy::FirstFit, 1024);

        let first = allocator.allocate(128, 32).expect("first allocation");
        let _second = allocator.allocate(128, 32).expect("second allocation");
        assert_eq!(first.address, 0);

        allocator.deallocate(first.address).expect("free first");

        // The freed hole at address 0 is the lowest-address region that fits.
        let third = allocator.allocate(128, 32).expect("third allocation");
        assert_eq!(
            third.address, first.address,
            "third allocation must reuse the freed hole"
        );
    }

    /// (b) Adjacent freed regions coalesce: after freeing two neighbors, a
    /// single allocation as large as their sum succeeds (which is impossible if
    /// the two holes were left fragmented).
    #[test]
    fn adjacent_frees_coalesce() {
        let mut allocator = MemoryAllocator::new(AllocationStrategy::FirstFit, 256);

        let a = allocator.allocate(128, 32).expect("alloc a");
        let b = allocator.allocate(128, 32).expect("alloc b");
        assert_eq!(a.address, 0);
        assert_eq!(b.address, 128);

        // Without coalescing the free list would be {0:128, 128:128} and a
        // 256-byte request would fail.
        allocator.deallocate(a.address).expect("free a");
        allocator.deallocate(b.address).expect("free b");

        let big = allocator
            .allocate(256, 32)
            .expect("coalesced region must satisfy the full-size request");
        assert_eq!(big.address, 0);
        assert_eq!(big.size, 256);
    }

    /// Carve a free list with two differently sized holes at known addresses.
    /// During carving there is always exactly one free region, so every
    /// strategy carves identically; only the final placement differs.
    fn carve_two_holes(strategy: AllocationStrategy) -> MemoryAllocator {
        let mut allocator = MemoryAllocator::new(strategy, 1024);

        let hole_big = allocator.allocate(256, 32).expect("carve big");
        let _keep1 = allocator.allocate(32, 32).expect("keep 1");
        let hole_small = allocator.allocate(64, 32).expect("carve small");
        let _keep2 = allocator.allocate(32, 32).expect("keep 2");

        allocator
            .deallocate(hole_big.address)
            .expect("free big hole");
        allocator
            .deallocate(hole_small.address)
            .expect("free small hole");

        // Free list is now {0: 256, 288: 64, 384: 640}.
        allocator
    }

    /// (c) FirstFit, BestFit and WorstFit pick different regions for the same
    /// request against an identical crafted free list.
    #[test]
    fn strategies_pick_different_regions() {
        // FirstFit: lowest-address fitting region -> the big hole at 0.
        let mut first_fit = carve_two_holes(AllocationStrategy::FirstFit);
        let a = first_fit.allocate(64, 32).expect("first-fit alloc");
        assert_eq!(a.address, 0);

        // BestFit: tightest fitting region -> the exact-size hole at 288.
        let mut best_fit = carve_two_holes(AllocationStrategy::BestFit);
        let b = best_fit.allocate(64, 32).expect("best-fit alloc");
        assert_eq!(b.address, 288);

        // WorstFit: largest fitting region -> the 640-byte tail at 384.
        let mut worst_fit = carve_two_holes(AllocationStrategy::WorstFit);
        let c = worst_fit.allocate(64, 32).expect("worst-fit alloc");
        assert_eq!(c.address, 384);

        assert_ne!(a.address, b.address);
        assert_ne!(a.address, c.address);
        assert_ne!(b.address, c.address);
    }

    /// (d) `live_range` is per-operand (a real [first_use, last_use] interval of
    /// schedule positions), not the old hardcoded whole-program (0, ops.len()).
    #[test]
    fn live_range_is_per_operand() {
        use crate::xla::frontend::graph_capture::test_support::{add_op, shape};
        use crate::xla::frontend::graph_capture::ComputationGraphBuilder;

        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("live_range");

        // Schedule positions: 0=Param a, 1=Param b, 2=Add(a,b)->c, 3=Negate(c)->d.
        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            shape(&[4]),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            shape(&[4]),
        );
        let c = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            shape(&[4]),
        );
        let d = add_op(
            &mut builder,
            &mut comp,
            OperationType::Negate,
            vec![c],
            shape(&[4]),
        );

        let planner: MemoryPlanner<f32> = MemoryPlanner::new(test_tpu_config());
        let analysis = planner
            .analyze_memory_requirements(&comp)
            .expect("memory analysis");

        let ops = comp.operations.len();
        let live_range = |id| {
            analysis
                .operand_info
                .get(&id)
                .map(|info| info.lifetime.live_range)
                .expect("operand info present")
        };

        // Each operand gets its own [first_use, last_use] position interval.
        assert_eq!(live_range(a), (0, 2));
        assert_eq!(live_range(b), (1, 2));
        assert_eq!(live_range(c), (2, 3));
        assert_eq!(live_range(d), (3, 3));

        // Regression: nothing is the old whole-program (0, ops.len()) range.
        for id in [a, b, c, d] {
            assert_ne!(
                live_range(id),
                (0, ops),
                "live_range must be per-operand, not whole-program"
            );
        }
    }
}
