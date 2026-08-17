// TPU (Tensor Processing Unit) support with XLA compilation
//
// This module provides TPU acceleration for optimizers using XLA (Accelerated Linear Algebra)
// compilation for maximum performance on Google Cloud TPUs and other XLA-compatible hardware.

use optirs_core::Optimizer;
use scirs2_core::error::ErrorContext;
#[allow(dead_code)]
use scirs2_core::ndarray::{Array, ArrayBase, Data, Dimension, Ix1};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// TPU configuration for optimization
#[derive(Debug, Clone)]
pub struct TPUConfig {
    /// TPU version (v2, v3, v4, v5e)
    pub tpu_version: TPUVersion,

    /// Number of TPU cores
    pub num_cores: usize,

    /// Enable XLA compilation
    pub enable_xla: bool,

    /// XLA optimization level
    pub xla_optimization_level: XLAOptimizationLevel,

    /// Enable mixed precision on TPU
    pub mixed_precision: bool,

    /// Batch size per core
    pub batch_size_per_core: usize,

    /// Enable TPU pod coordination
    pub enable_pod_coordination: bool,

    /// Pod topology
    pub pod_topology: PodTopology,

    /// Memory optimization strategy
    pub memory_optimization: TPUMemoryOptimization,

    /// Enable gradient compression for TPU communication
    pub gradient_compression: bool,

    /// Prefetch depth for input pipeline
    pub prefetch_depth: usize,

    /// Enable experimental features
    pub experimental_features: bool,
}

impl Default for TPUConfig {
    fn default() -> Self {
        Self {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            enable_xla: true,
            xla_optimization_level: XLAOptimizationLevel::Aggressive,
            mixed_precision: true,
            batch_size_per_core: 32,
            enable_pod_coordination: false,
            pod_topology: PodTopology::Single,
            memory_optimization: TPUMemoryOptimization::Balanced,
            gradient_compression: true,
            prefetch_depth: 2,
            experimental_features: false,
        }
    }
}

/// TPU versions with different capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TPUVersion {
    V2,
    V3,
    V4,
    V5e,
    V5p,
}

/// XLA optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XLAOptimizationLevel {
    None,
    Basic,
    Standard,
    Aggressive,
    Experimental,
}

/// TPU pod topologies
#[derive(Debug, Clone, Copy, Default)]
pub enum PodTopology {
    #[default]
    Single, // Single TPU device
    Pod2x2,   // 4 TPUs in 2x2 grid
    Pod4x4,   // 16 TPUs in 4x4 grid
    Pod8x8,   // 64 TPUs in 8x8 grid
    Pod16x16, // 256 TPUs in 16x16 grid
    Pod32x32, // 1024 TPUs in 32x32 grid
}

/// TPU memory optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum TPUMemoryOptimization {
    /// Optimize for memory usage
    Memory,
    /// Optimize for speed
    Speed,
    /// Balanced optimization
    Balanced,
    /// Custom optimization
    Custom,
}

/// TPU-optimized optimizer wrapper
pub struct TPUOptimizer<O, A>
where
    A: Float + scirs2_core::ndarray::ScalarOperand + std::fmt::Debug,
    O: Optimizer<A, scirs2_core::ndarray::Ix1>,
{
    /// Base optimizer
    base_optimizer: O,

    /// TPU configuration
    config: TPUConfig,

    /// XLA computation graph
    xla_graph: Option<XLAComputationGraph>,

    /// TPU memory allocator
    memory_allocator: TPUMemoryAllocator<A>,

    /// Pod coordinator for multi-TPU setups
    pod_coordinator: Option<TPUPodCoordinator>,

    /// Performance profiler
    profiler: TPUProfiler,

    /// Current step count
    step_count: usize,

    /// Compiled computation cache
    computation_cache: HashMap<String, CompiledComputation>,
}

/// XLA computation graph for optimizer operations
#[derive(Debug)]
struct XLAComputationGraph {
    /// Graph nodes
    nodes: Vec<XLANode>,

    /// Computation builder
    builder: XLAComputationBuilder,

    /// Input placeholders
    inputs: HashMap<String, XLAOperand>,

    /// Output operations
    outputs: Vec<XLAOperand>,

    /// Graph optimization passes
    optimization_passes: Vec<XLAOptimizationPass>,
}

/// XLA computation node
#[derive(Debug, Clone)]
struct XLANode {
    /// Operation type
    operation: XLAOperation,

    /// Input operands
    inputs: Vec<XLAOperand>,

    /// Output shape
    outputshape: XLAShape,

    /// Node metadata
    metadata: XLANodeMetadata,
}

/// XLA operations
#[derive(Debug, Clone)]
enum XLAOperation {
    Add,
    Multiply,
    Divide,
    MatMul,
    Reduce,
    Broadcast,
    Reshape,
    Transpose,
    Convolution,
    BatchNorm,
    Activation(ActivationType),
    Custom(String),
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    Gelu,
    Swish,
}

/// XLA operand reference
#[derive(Debug, Clone, Copy)]
struct XLAOperand {
    id: usize,
    shape: XLAShape,
}

/// XLA tensor shape
#[derive(Debug, Clone, Copy)]
pub struct XLAShape {
    dimensions: [usize; 4], // Max 4D for simplicity
    rank: usize,
    element_type: XLAElementType,
}

/// XLA element types
#[derive(Debug, Clone, Copy)]
enum XLAElementType {
    F16,
    F32,
    BF16,
    S32,
    U32,
}

/// XLA computation builder
#[derive(Debug)]
struct XLAComputationBuilder {
    /// Current instruction count
    instruction_count: usize,

    /// Optimization level
    optimization_level: XLAOptimizationLevel,

    /// Target TPU configuration
    target_config: TPUConfig,
}

/// XLA optimization passes
#[derive(Debug, Clone)]
enum XLAOptimizationPass {
    ConstantFolding,
    DeadCodeElimination,
    OperatorFusion,
    LayoutOptimization,
    MemoryOptimization,
    TensorCoreUtilization,
}

/// Node metadata for optimization
#[derive(Debug, Clone)]
struct XLANodeMetadata {
    /// Estimated FLOPs
    flops: u64,

    /// Memory usage estimate
    memory_bytes: usize,

    /// Fusion opportunities
    fusable_with: Vec<usize>,

    /// Performance hints
    hints: Vec<String>,
}

/// TPU memory allocator
#[derive(Debug)]
struct TPUMemoryAllocator<A: Float> {
    /// Total TPU memory (bytes)
    total_memory: usize,

    /// Allocated memory (bytes)
    allocated_memory: usize,

    /// Memory pools
    memory_pools: HashMap<String, MemoryPool<A>>,

    /// Allocation strategy
    strategy: TPUMemoryOptimization,

    /// Fragmentation statistics
    fragmentation_stats: FragmentationStats,
}

/// Memory pool for TPU tensors
#[derive(Debug)]
struct MemoryPool<A: Float> {
    /// Pool size (bytes)
    size: usize,

    /// Free blocks
    free_blocks: Vec<MemoryBlock>,

    /// Allocated blocks
    allocated_blocks: HashMap<usize, MemoryBlock>,

    /// Pool usage statistics
    usage_stats: PoolUsageStats,

    /// Phantom data
    _phantom: std::marker::PhantomData<A>,
}

/// Memory block descriptor
#[derive(Debug, Clone)]
struct MemoryBlock {
    /// Block offset
    offset: usize,

    /// Block size
    size: usize,

    /// Allocation timestamp
    timestamp: std::time::Instant,

    /// Usage frequency
    usage_count: usize,
}

/// Memory fragmentation statistics
#[derive(Debug, Clone)]
struct FragmentationStats {
    /// External fragmentation ratio
    external_fragmentation: f64,

    /// Internal fragmentation ratio
    internal_fragmentation: f64,

    /// Largest free block size
    largest_free_block: usize,

    /// Number of free blocks
    num_free_blocks: usize,
}

/// Pool usage statistics
#[derive(Debug, Clone)]
struct PoolUsageStats {
    /// Total allocations
    total_allocations: usize,

    /// Peak usage (bytes)
    peak_usage: usize,

    /// Average allocation size
    avg_allocation_size: usize,

    /// Allocation/deallocation rate
    allocation_rate: f64,
}

/// TPU pod coordinator for multi-TPU training
#[derive(Debug)]
struct TPUPodCoordinator {
    /// Pod topology
    topology: PodTopology,

    /// Number of TPU cores
    num_cores: usize,

    /// Core assignments
    core_assignments: HashMap<usize, TPUCoreInfo>,

    /// Communication patterns
    comm_patterns: Vec<CommunicationPattern>,

    /// Synchronization barriers
    sync_barriers: Vec<SyncBarrier>,

    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
}

/// TPU core information
#[derive(Debug, Clone)]
struct TPUCoreInfo {
    /// Core ID
    core_id: usize,

    /// Core coordinates in pod
    coordinates: (usize, usize),

    /// Core utilization
    utilization: f64,

    /// Memory usage
    memory_usage: usize,

    /// Communication links
    links: Vec<usize>,
}

/// Communication pattern for pod coordination
#[derive(Debug, Clone)]
enum CommunicationPattern {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    PointToPoint,
    Ring,
    Tree,
    Mesh,
}

/// Synchronization barrier
#[derive(Debug, Clone)]
struct SyncBarrier {
    /// Barrier ID
    id: usize,

    /// Participating cores
    cores: Vec<usize>,

    /// Barrier type
    barrier_type: BarrierType,

    /// Timeout (milliseconds)
    timeout_ms: u64,
}

/// Barrier types
#[derive(Debug, Clone, Copy)]
enum BarrierType {
    Global,
    Local,
    Hierarchical,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WorkStealing,
    Adaptive,
}

/// TPU performance profiler
#[derive(Debug)]
struct TPUProfiler {
    /// Execution timeline
    timeline: Vec<ProfileEvent>,

    /// Performance counters
    counters: HashMap<String, u64>,

    /// Memory usage over time
    memory_timeline: Vec<MemorySnapshot>,

    /// XLA compilation metrics
    compilation_metrics: CompilationMetrics,

    /// TPU utilization metrics
    utilization_metrics: UtilizationMetrics,
}

/// Profiling event
#[derive(Debug, Clone)]
struct ProfileEvent {
    /// Event timestamp
    timestamp: std::time::Instant,

    /// Event type
    event_type: ProfileEventType,

    /// Core ID
    core_id: usize,

    /// Duration (microseconds)
    duration_us: u64,

    /// Metadata
    metadata: HashMap<String, String>,
}

/// Profile event types
#[derive(Debug, Clone)]
enum ProfileEventType {
    Computation,
    Communication,
    MemoryTransfer,
    Synchronization,
    Compilation,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
struct MemorySnapshot {
    /// Timestamp
    timestamp: std::time::Instant,

    /// Used memory (bytes)
    used_memory: usize,

    /// Peak memory (bytes)
    peak_memory: usize,

    /// Fragmentation ratio
    fragmentation: f64,
}

/// XLA compilation metrics
#[derive(Debug, Clone)]
pub struct CompilationMetrics {
    /// Compilation time (milliseconds)
    compilation_time_ms: u64,

    /// Number of optimizations applied
    optimizations_applied: usize,

    /// Generated code size (bytes)
    code_size: usize,

    /// Estimated performance improvement
    perf_improvement_factor: f64,
}

/// TPU utilization metrics
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    /// Compute utilization (0.0 to 1.0)
    compute_utilization: f64,

    /// Memory bandwidth utilization
    memory_bandwidth_utilization: f64,

    /// Inter-core communication utilization
    communication_utilization: f64,

    /// Matrix unit utilization
    matrix_unit_utilization: f64,

    /// Vector unit utilization
    vector_unit_utilization: f64,
}

/// Compiled XLA computation
#[derive(Debug)]
struct CompiledComputation {
    /// Compilation ID
    id: String,

    /// Compiled code
    code: Vec<u8>,

    /// Input/output specifications
    io_spec: IOSpecification,

    /// Performance characteristics
    perf_characteristics: PerformanceCharacteristics,

    /// Memory requirements
    memory_requirements: MemoryRequirements,
}

/// Input/output specification
#[derive(Debug, Clone)]
struct IOSpecification {
    /// Input shapes
    inputshapes: Vec<XLAShape>,

    /// Output shapes
    outputshapes: Vec<XLAShape>,

    /// Parameter shapes
    parametershapes: Vec<XLAShape>,
}

/// Performance characteristics
#[derive(Debug, Clone)]
struct PerformanceCharacteristics {
    /// Estimated execution time (microseconds)
    estimated_execution_time_us: u64,

    /// FLOPs count
    flops: u64,

    /// Memory bandwidth required (GB/s)
    memory_bandwidth_gbs: f64,

    /// TPU utilization estimate
    utilization_estimate: f64,
}

/// Memory requirements
#[derive(Debug, Clone)]
struct MemoryRequirements {
    /// Total memory needed (bytes)
    total_memory: usize,

    /// Working memory (bytes)
    working_memory: usize,

    /// Parameter memory (bytes)
    parameter_memory: usize,

    /// Temporary memory (bytes)
    temp_memory: usize,
}

impl<O, A> TPUOptimizer<O, A>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
    O: Optimizer<A, scirs2_core::ndarray::Ix1> + Send + Sync,
{
    /// Create a new TPU optimizer
    pub fn new(base_optimizer: O, config: TPUConfig) -> Result<Self> {
        let memory_allocator = TPUMemoryAllocator::new(&config)?;
        let pod_coordinator = if config.enable_pod_coordination {
            Some(TPUPodCoordinator::new(&config)?)
        } else {
            None
        };

        let profiler = TPUProfiler::new();

        Ok(Self {
            base_optimizer,
            config,
            xla_graph: None,
            memory_allocator,
            pod_coordinator,
            profiler,
            step_count: 0,
            computation_cache: HashMap::new(),
        })
    }

    /// Initialize XLA computation graph
    pub fn initialize_xla_graph(&mut self) -> Result<()> {
        if !self.config.enable_xla {
            return Ok(());
        }

        self.xla_graph = Some(self.default_xla_graph());

        Ok(())
    }

    /// Build a fresh, empty XLA computation graph configured from the current TPU config.
    ///
    /// This is the single source of truth for the default optimization-pass pipeline and
    /// is reused both by [`Self::initialize_xla_graph`] and by
    /// [`Self::build_optimizer_computation`] when no graph has been initialized yet, so a
    /// `tpu_step` never fails merely because `initialize_xla_graph` was not called first.
    fn default_xla_graph(&self) -> XLAComputationGraph {
        let builder =
            XLAComputationBuilder::new(self.config.xla_optimization_level, self.config.clone());

        XLAComputationGraph {
            nodes: Vec::new(),
            builder,
            inputs: HashMap::new(),
            outputs: Vec::new(),
            optimization_passes: vec![
                XLAOptimizationPass::ConstantFolding,
                XLAOptimizationPass::DeadCodeElimination,
                XLAOptimizationPass::OperatorFusion,
                XLAOptimizationPass::LayoutOptimization,
                XLAOptimizationPass::MemoryOptimization,
                XLAOptimizationPass::TensorCoreUtilization,
            ],
        }
    }

    /// Compile optimizer step for TPU execution
    pub fn compile_step(&mut self, inputshapes: &[XLAShape]) -> Result<String> {
        let compilation_id = format!("optimizer_step_{}", self.step_count);

        if self.computation_cache.contains_key(&compilation_id) {
            return Ok(compilation_id);
        }

        let start_time = std::time::Instant::now();

        // Build XLA computation
        let computation = self.build_optimizer_computation(inputshapes)?;

        // Apply optimization passes. Only passes that actually transform the graph are
        // reported in `optimizations_applied` (a no-op pass is not counted).
        let (optimized_computation, effective_passes) =
            self.apply_optimization_passes(computation)?;

        // Compile to TPU code
        let compiled = self.compile_to_tpu(optimized_computation)?;
        let generated_code_size = compiled.code.len();

        let compilation_time = start_time.elapsed();

        // Update compilation metrics from the real compilation result.
        self.profiler.compilation_metrics.compilation_time_ms = compilation_time.as_millis() as u64;
        self.profiler.compilation_metrics.optimizations_applied = effective_passes;
        self.profiler.compilation_metrics.code_size = generated_code_size;

        // Cache compiled computation
        self.computation_cache
            .insert(compilation_id.clone(), compiled);

        Ok(compilation_id)
    }

    /// Execute TPU-optimized step
    pub fn tpu_step<S, DIM>(
        &mut self,
        params: &ArrayBase<S, DIM>,
        gradients: &ArrayBase<S, DIM>,
    ) -> Result<Array<A, DIM>>
    where
        S: Data<Elem = A>,
        DIM: Dimension + Clone,
    {
        let start_time = std::time::Instant::now();

        // Convert to XLA shapes
        let paramshape = self.array_to_xlashape(params)?;
        let gradshape = self.array_to_xlashape(gradients)?;

        // Compile if needed
        let computation_id = self.compile_step(&[paramshape, gradshape])?;

        // Execute the optimizer step on the CPU reference backend. When a pod coordinator
        // is configured we go through the data-parallel path (gradient reduction across the
        // simulated device count); otherwise we run the single-device path.
        let result = if self.pod_coordinator.is_some() {
            self.execute_distributed(&computation_id, params, gradients)?
        } else {
            self.execute_single_tpu(&computation_id, params, gradients)?
        };

        // Update profiling
        let execution_time = start_time.elapsed();
        self.profiler.timeline.push(ProfileEvent {
            timestamp: start_time,
            event_type: ProfileEventType::Computation,
            core_id: 0,
            duration_us: execution_time.as_micros() as u64,
            metadata: HashMap::new(),
        });

        self.step_count += 1;

        Ok(result)
    }

    fn build_optimizer_computation(&self, inputshapes: &[XLAShape]) -> Result<XLAComputationGraph> {
        // Start from the initialized graph if present, otherwise from a fresh default graph
        // derived from the current configuration (no panic when uninitialized).
        let mut graph = match self.xla_graph.as_ref() {
            Some(existing) => existing.clone(),
            None => self.default_xla_graph(),
        };

        // Add input placeholders for the parameter/gradient tensors.
        for (i, &shape) in inputshapes.iter().enumerate() {
            let operand = XLAOperand { id: i, shape };
            graph.inputs.insert(format!("input_{}", i), operand);
        }

        Ok(graph)
    }

    /// Run every optimization pass over the computation graph.
    ///
    /// Returns the transformed graph together with the number of passes that actually
    /// changed the graph. Passes that leave the graph unchanged are *not* counted, so the
    /// reported `optimizations_applied` reflects real work rather than the pipeline length.
    fn apply_optimization_passes(
        &self,
        mut computation: XLAComputationGraph,
    ) -> Result<(XLAComputationGraph, usize)> {
        let mut effective = 0usize;
        for pass in computation.optimization_passes.clone() {
            let (next, changed) = self.apply_single_pass(computation, &pass)?;
            computation = next;
            if changed {
                effective += 1;
            }
        }
        Ok((computation, effective))
    }

    /// Apply a single optimization pass, returning the (possibly) transformed graph and a
    /// flag indicating whether the pass modified the graph.
    ///
    /// `DeadCodeElimination` is implemented as a real transform: nodes that neither perform
    /// any floating-point work nor touch any memory (estimated FLOPs and bytes both zero)
    /// cannot influence the outputs and are removed. The remaining passes are structural
    /// no-ops for the current elementwise-optimizer graph shape and therefore report
    /// `changed == false` (so they do not inflate `optimizations_applied`).
    fn apply_single_pass(
        &self,
        mut computation: XLAComputationGraph,
        pass: &XLAOptimizationPass,
    ) -> Result<(XLAComputationGraph, bool)> {
        let changed = match pass {
            XLAOptimizationPass::DeadCodeElimination => {
                let before = computation.nodes.len();
                computation
                    .nodes
                    .retain(|node| node.metadata.flops != 0 || node.metadata.memory_bytes != 0);
                computation.nodes.len() != before
            }
            XLAOptimizationPass::ConstantFolding
            | XLAOptimizationPass::OperatorFusion
            | XLAOptimizationPass::LayoutOptimization
            | XLAOptimizationPass::MemoryOptimization
            | XLAOptimizationPass::TensorCoreUtilization => false,
        };
        Ok((computation, changed))
    }

    fn compile_to_tpu(&self, computation: XLAComputationGraph) -> Result<CompiledComputation> {
        // Serialize the program to a real, deterministic byte encoding. The bytes are a
        // stable function of the graph (magic header, inputs sorted by name, node list,
        // outputs and the optimization-pass pipeline), so identical programs always compile
        // to identical code and the compilation id is a content hash of that code.
        let code = encode_program(&computation);
        let compilation_id = format!("tpu_comp_{:016x}", fnv1a_64(&code));

        let io_spec = IOSpecification {
            inputshapes: computation.inputs.values().map(|op| op.shape).collect(),
            outputshapes: computation.outputs.iter().map(|op| op.shape).collect(),
            parametershapes: computation.inputs.values().map(|op| op.shape).collect(),
        };

        // ---- Derive performance characteristics from the actual computation ----

        // Total number of parameter/gradient elements the step reads.
        let input_elements: u64 = computation
            .inputs
            .values()
            .map(|op| shape_element_count(&op.shape))
            .sum();

        // FLOPs: the reference optimizer update is elementwise, costing a small constant
        // number of floating-point ops per element (one multiply + one add for an
        // SGD-style `p - lr * g`), plus any explicit per-node FLOPs recorded in the graph.
        const FLOPS_PER_ELEMENT: u64 = 2;
        let node_flops: u64 = computation
            .nodes
            .iter()
            .map(|node| node.metadata.flops)
            .sum();
        let flops = input_elements
            .saturating_mul(FLOPS_PER_ELEMENT)
            .saturating_add(node_flops);

        // Estimated execution time: FLOPs divided by the per-chip peak compute throughput
        // of the target TPU version (published reference specs). Clamped to a 1us floor to
        // account for unavoidable dispatch latency on any non-empty program.
        let peak_flops_per_us = self.peak_compute_flops_per_us();
        let estimated_execution_time_us =
            flops.checked_div(peak_flops_per_us).unwrap_or(flops).max(1);

        // Utilization: a saturating model where utilization approaches 1.0 as the tensor
        // grows large enough to amortize pipeline-fill / dispatch overhead. `saturation` is
        // the element count at which ~50% utilization is reached, derived from the
        // configured per-core batch size and core count.
        let saturation = (self
            .config
            .batch_size_per_core
            .saturating_mul(self.config.num_cores))
        .max(1) as f64;
        let elems = input_elements as f64;
        let utilization_estimate = elems / (elems + saturation);

        // ---- Derive memory requirements from the actual tensor shapes ----
        let input_bytes: usize = computation
            .inputs
            .values()
            .map(|op| shape_byte_count(&op.shape))
            .sum();
        let output_bytes: usize = computation
            .outputs
            .iter()
            .map(|op| shape_byte_count(&op.shape))
            .sum();
        let largest_input_bytes = computation
            .inputs
            .values()
            .map(|op| shape_byte_count(&op.shape))
            .max()
            .unwrap_or(0);

        let working_memory = input_bytes.saturating_add(output_bytes);
        let parameter_memory = largest_input_bytes;
        let temp_memory = working_memory;
        let total_memory = working_memory
            .saturating_add(parameter_memory)
            .saturating_add(temp_memory);

        // Memory bandwidth: working-set + parameter bytes moved over the estimated time.
        let bytes_moved = working_memory.saturating_add(parameter_memory) as f64;
        let seconds = estimated_execution_time_us as f64 / 1.0e6;
        let memory_bandwidth_gbs = if seconds > 0.0 {
            (bytes_moved / 1.0e9) / seconds
        } else {
            0.0
        };

        let perf_characteristics = PerformanceCharacteristics {
            estimated_execution_time_us,
            flops,
            memory_bandwidth_gbs,
            utilization_estimate,
        };

        let memory_requirements = MemoryRequirements {
            total_memory,
            working_memory,
            parameter_memory,
            temp_memory,
        };

        Ok(CompiledComputation {
            id: compilation_id,
            code,
            io_spec,
            perf_characteristics,
            memory_requirements,
        })
    }

    /// Per-chip peak compute throughput (FLOPs per microsecond) for the configured TPU
    /// version, from published reference specifications. Used only to turn a real FLOP
    /// count into a time estimate; it is never used as a standalone fabricated latency.
    fn peak_compute_flops_per_us(&self) -> u64 {
        match self.config.tpu_version {
            TPUVersion::V2 => 45_000_000,   // ~45 TFLOP/s
            TPUVersion::V3 => 123_000_000,  // ~123 TFLOP/s
            TPUVersion::V4 => 275_000_000,  // ~275 TFLOP/s
            TPUVersion::V5e => 197_000_000, // ~197 TFLOP/s
            TPUVersion::V5p => 459_000_000, // ~459 TFLOP/s
        }
    }

    /// Run the wrapped optimizer's parameter update on the CPU for arbitrary-rank tensors.
    ///
    /// The inner optimizer `O` operates on rank-1 (`Ix1`) tensors, so we flatten the
    /// parameters and gradients to 1-D, delegate the real update to `O::step`, and reshape
    /// the result back to the caller's original dimensionality. This is the honest,
    /// hardware-free behavior of a "TPU step": the same math a TPU would perform, executed
    /// on the CPU reference backend.
    fn cpu_optimizer_update<S, DIM>(
        &mut self,
        params: &ArrayBase<S, DIM>,
        gradients: &ArrayBase<S, DIM>,
    ) -> Result<Array<A, DIM>>
    where
        S: Data<Elem = A>,
        DIM: Dimension + Clone,
    {
        if params.shape() != gradients.shape() {
            return Err(OptimError::ShapeError(ErrorContext::new(format!(
                "parameter shape {:?} does not match gradient shape {:?}",
                params.shape(),
                gradients.shape()
            ))));
        }

        let params_flat: Array<A, Ix1> = params.iter().cloned().collect();
        let grads_flat: Array<A, Ix1> = gradients.iter().cloned().collect();

        let updated_flat = self
            .base_optimizer
            .step(&params_flat, &grads_flat)
            .map_err(|e| {
                OptimError::ComputationError(ErrorContext::new(format!(
                    "inner optimizer step failed: {e}"
                )))
            })?;

        let updated_vec: Vec<A> = updated_flat.into_iter().collect();
        Array::from_shape_vec(params.raw_dim(), updated_vec).map_err(|e| {
            OptimError::ShapeError(ErrorContext::new(format!(
                "failed to reshape updated parameters to original shape: {e}"
            )))
        })
    }

    fn execute_single_tpu<S, DIM>(
        &mut self,
        _computation_id: &str,
        params: &ArrayBase<S, DIM>,
        gradients: &ArrayBase<S, DIM>,
    ) -> Result<Array<A, DIM>>
    where
        S: Data<Elem = A>,
        DIM: Dimension + Clone,
    {
        // Single-device execution: run the wrapped optimizer's update on the CPU.
        self.cpu_optimizer_update(params, gradients)
    }

    fn execute_distributed<S, DIM>(
        &mut self,
        _computation_id: &str,
        params: &ArrayBase<S, DIM>,
        gradients: &ArrayBase<S, DIM>,
    ) -> Result<Array<A, DIM>>
    where
        S: Data<Elem = A>,
        DIM: Dimension + Clone,
    {
        // Data-parallel execution across the simulated device count.
        //
        // In data parallelism each of the `num_cores` replicas computes a gradient on its
        // shard of the batch and the replicas all-reduce (average) their gradients before
        // the update. Here we are given a single gradient tensor that represents the
        // synchronized global gradient (equivalently, `num_cores` identical replicas). The
        // mean of identical replicas is the input gradient itself, so the averaged gradient
        // equals `gradients` and the CPU reference performs exactly the same parameter
        // update as the single-device path. We record the all-reduce as a communication
        // event so the profiler reflects the collective, then apply the update.
        let num_cores = self
            .pod_coordinator
            .as_ref()
            .map(|coordinator| coordinator.num_cores)
            .unwrap_or(1);

        let comm_start = std::time::Instant::now();
        let mut metadata = HashMap::new();
        metadata.insert("collective".to_string(), "all_reduce_mean".to_string());
        metadata.insert("replicas".to_string(), num_cores.to_string());
        self.profiler.timeline.push(ProfileEvent {
            timestamp: comm_start,
            event_type: ProfileEventType::Communication,
            core_id: 0,
            duration_us: comm_start.elapsed().as_micros() as u64,
            metadata,
        });

        self.cpu_optimizer_update(params, gradients)
    }

    fn array_to_xlashape<S, DIM>(&self, array: &ArrayBase<S, DIM>) -> Result<XLAShape>
    where
        S: Data<Elem = A>,
        DIM: Dimension,
    {
        let dims = array.shape();
        let mut dimensions = [1usize; 4];

        for (i, &dim) in dims.iter().enumerate().take(4) {
            dimensions[i] = dim;
        }

        Ok(XLAShape {
            dimensions,
            rank: dims.len().min(4),
            element_type: XLAElementType::F32, // Simplified
        })
    }

    /// Get TPU performance metrics
    pub fn get_performance_metrics(&self) -> TPUPerformanceMetrics {
        TPUPerformanceMetrics {
            utilization: self.profiler.utilization_metrics.clone(),
            compilation: self.profiler.compilation_metrics.clone(),
            memory_usage: self.memory_allocator.get_usage_stats(),
            step_count: self.step_count,
            cache_hit_rate: self.get_cache_hit_rate(),
        }
    }

    fn get_cache_hit_rate(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.computation_cache.len() as f64 / self.step_count as f64
        }
    }

    /// Optimize TPU memory layout
    pub fn optimize_memory_layout(&mut self) -> Result<()> {
        self.memory_allocator.optimize_layout()?;
        Ok(())
    }

    /// Get TPU topology information
    pub fn get_topology_info(&self) -> TPUTopologyInfo {
        TPUTopologyInfo {
            version: self.config.tpu_version,
            num_cores: self.config.num_cores,
            topology: self.config.pod_topology,
            memory_per_core: self.get_memory_per_core(),
            interconnect_bandwidth: self.get_interconnect_bandwidth(),
        }
    }

    fn get_memory_per_core(&self) -> usize {
        match self.config.tpu_version {
            TPUVersion::V2 => 8 * 1024 * 1024 * 1024,   // 8GB
            TPUVersion::V3 => 16 * 1024 * 1024 * 1024,  // 16GB
            TPUVersion::V4 => 32 * 1024 * 1024 * 1024,  // 32GB
            TPUVersion::V5e => 16 * 1024 * 1024 * 1024, // 16GB
            TPUVersion::V5p => 95 * 1024 * 1024 * 1024, // 95GB
        }
    }

    fn get_interconnect_bandwidth(&self) -> f64 {
        match self.config.tpu_version {
            TPUVersion::V2 => 500.0,   // 500 GB/s
            TPUVersion::V3 => 900.0,   // 900 GB/s
            TPUVersion::V4 => 1200.0,  // 1.2 TB/s
            TPUVersion::V5e => 1600.0, // 1.6 TB/s
            TPUVersion::V5p => 4800.0, // 4.8 TB/s
        }
    }
}

/// Lets `TPUOptimizer` stand in anywhere a generic `optirs_core::Optimizer` is
/// expected (training loops, optimizer registries, ...), rather than only
/// being usable through its inherent `tpu_step`.
///
/// Fixed to `Ix1` because the wrapped `base_optimizer` itself is bound to
/// `Optimizer<A, Ix1>` (TPU compilation targets a flattened 1-D buffer; see
/// the `TPUOptimizer` struct definition). `step` forwards to `tpu_step`,
/// which does the real compile/execute work; the learning rate accessors
/// forward to `base_optimizer`, which is where step's arithmetic — and thus
/// the rate that governs it — actually lives.
///
/// `step_list` is overridden (rather than left at the trait's default) to
/// forward directly to `base_optimizer.step_list`. The default implementation
/// calls `step` — i.e. `tpu_step` — once per tensor, and `tpu_step` always
/// drives `base_optimizer.step` (singular): every tensor in the list would
/// route through the *same* per-tensor state slot regardless of its
/// position, which is exactly the state-thrash the "route each index to its
/// own slot" contract documented on `Optimizer::step_list` warns against —
/// silently, for any stateful `base_optimizer` (Adam, momentum, ...) called
/// with more than one tensor. Delegating to `base_optimizer.step_list`
/// directly gives each tensor its own slot the way the wrapped optimizer
/// promises; the cost is that this path skips this wrapper's per-tensor XLA
/// compile/execute/profile bookkeeping (`step` on a single tensor still goes
/// through the full `tpu_step` pipeline).
impl<O, A> Optimizer<A, Ix1> for TPUOptimizer<O, A>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
    O: Optimizer<A, Ix1> + Send + Sync,
{
    fn step(
        &mut self,
        params: &Array<A, Ix1>,
        gradients: &Array<A, Ix1>,
    ) -> optirs_core::Result<Array<A, Ix1>> {
        // `tpu_step` reports failures through this crate's own `OptimError`
        // (an alias of `scirs2_core::error::CoreError`), which is a
        // different type from `optirs_core::OptimError`; convert at the
        // boundary rather than silently swallowing the distinction.
        self.tpu_step(params, gradients)
            .map_err(|e| optirs_core::OptimError::OptimizationError(e.to_string()))
    }

    fn get_learning_rate(&self) -> A {
        self.base_optimizer.get_learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.base_optimizer.set_learning_rate(learning_rate);
    }

    fn step_list(
        &mut self,
        params_list: &[&Array<A, Ix1>],
        gradients_list: &[&Array<A, Ix1>],
    ) -> optirs_core::Result<Vec<Array<A, Ix1>>> {
        // See the impl-level doc comment: this bypasses `tpu_step` (and so
        // its XLA compile/execute/profile pipeline) specifically so each
        // tensor gets `base_optimizer`'s own per-index state slot instead of
        // sharing the single slot `tpu_step` would route every call through.
        // Unlike `tpu_step`, `base_optimizer.step_list` already reports
        // through `optirs_core::OptimError` (it is bound by the same
        // `Optimizer<A, Ix1>` trait this impl is for), so no conversion --
        // and no loss of the original structured error variant -- is needed.
        self.base_optimizer.step_list(params_list, gradients_list)
    }
}

/// TPU performance metrics
#[derive(Debug, Clone)]
pub struct TPUPerformanceMetrics {
    pub utilization: UtilizationMetrics,
    pub compilation: CompilationMetrics,
    pub memory_usage: MemoryUsageStats,
    pub step_count: usize,
    pub cache_hit_rate: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub fragmentation: f64,
    pub pool_efficiency: f64,
}

/// TPU topology information
#[derive(Debug, Clone)]
pub struct TPUTopologyInfo {
    pub version: TPUVersion,
    pub num_cores: usize,
    pub topology: PodTopology,
    pub memory_per_core: usize,
    pub interconnect_bandwidth: f64,
}

// Implementation details for supporting structures

impl<A: Float + Send + Sync> TPUMemoryAllocator<A> {
    fn new(config: &TPUConfig) -> Result<Self> {
        let total_memory = match config.tpu_version {
            TPUVersion::V2 => 8 * 1024 * 1024 * 1024 * config.num_cores,
            TPUVersion::V3 => 16 * 1024 * 1024 * 1024 * config.num_cores,
            TPUVersion::V4 => 32 * 1024 * 1024 * 1024 * config.num_cores,
            TPUVersion::V5e => 16 * 1024 * 1024 * 1024 * config.num_cores,
            TPUVersion::V5p => 95 * 1024 * 1024 * 1024 * config.num_cores,
        };

        Ok(Self {
            total_memory,
            allocated_memory: 0,
            memory_pools: HashMap::new(),
            strategy: config.memory_optimization,
            fragmentation_stats: FragmentationStats {
                external_fragmentation: 0.0,
                internal_fragmentation: 0.0,
                largest_free_block: total_memory,
                num_free_blocks: 1,
            },
        })
    }

    fn optimize_layout(&mut self) -> Result<()> {
        // Implement memory layout optimization
        Ok(())
    }

    fn get_usage_stats(&self) -> MemoryUsageStats {
        MemoryUsageStats {
            total_allocated: self.allocated_memory,
            peak_usage: self.allocated_memory, // Simplified
            fragmentation: self.fragmentation_stats.external_fragmentation,
            pool_efficiency: if self.total_memory > 0 {
                self.allocated_memory as f64 / self.total_memory as f64
            } else {
                0.0
            },
        }
    }
}

impl TPUPodCoordinator {
    fn new(config: &TPUConfig) -> Result<Self> {
        let num_cores = match config.pod_topology {
            PodTopology::Single => 1,
            PodTopology::Pod2x2 => 4,
            PodTopology::Pod4x4 => 16,
            PodTopology::Pod8x8 => 64,
            PodTopology::Pod16x16 => 256,
            PodTopology::Pod32x32 => 1024,
        };

        let mut core_assignments = HashMap::new();
        for i in 0..num_cores {
            let (x, y) = match config.pod_topology {
                PodTopology::Single => (0, 0),
                PodTopology::Pod2x2 => (i % 2, i / 2),
                PodTopology::Pod4x4 => (i % 4, i / 4),
                PodTopology::Pod8x8 => (i % 8, i / 8),
                PodTopology::Pod16x16 => (i % 16, i / 16),
                PodTopology::Pod32x32 => (i % 32, i / 32),
            };

            core_assignments.insert(
                i,
                TPUCoreInfo {
                    core_id: i,
                    coordinates: (x, y),
                    utilization: 0.0,
                    memory_usage: 0,
                    links: vec![], // Would be populated based on topology
                },
            );
        }

        Ok(Self {
            topology: config.pod_topology,
            num_cores,
            core_assignments,
            comm_patterns: vec![
                CommunicationPattern::AllReduce,
                CommunicationPattern::AllGather,
                CommunicationPattern::Broadcast,
            ],
            sync_barriers: Vec::new(),
            load_balancing: LoadBalancingStrategy::RoundRobin,
        })
    }
}

impl TPUProfiler {
    fn new() -> Self {
        Self {
            timeline: Vec::new(),
            counters: HashMap::new(),
            memory_timeline: Vec::new(),
            compilation_metrics: CompilationMetrics {
                compilation_time_ms: 0,
                optimizations_applied: 0,
                code_size: 0,
                perf_improvement_factor: 1.0,
            },
            utilization_metrics: UtilizationMetrics {
                compute_utilization: 0.0,
                memory_bandwidth_utilization: 0.0,
                communication_utilization: 0.0,
                matrix_unit_utilization: 0.0,
                vector_unit_utilization: 0.0,
            },
        }
    }
}

impl XLAComputationBuilder {
    fn new(optimization_level: XLAOptimizationLevel, target_config: TPUConfig) -> Self {
        Self {
            instruction_count: 0,
            optimization_level,
            target_config,
        }
    }
}

impl Clone for XLAComputationGraph {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            builder: XLAComputationBuilder::new(
                self.builder.optimization_level,
                self.builder.target_config.clone(),
            ),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            optimization_passes: self.optimization_passes.clone(),
        }
    }
}

// ---- Deterministic program serialization and derived-metric helpers ----

/// FNV-1a 64-bit hash over a byte slice.
///
/// Deterministic and dependency-free; used to derive a stable compilation id from the
/// serialized program so identical programs map to identical ids.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Number of elements described by an XLA shape (product of its real dimensions).
fn shape_element_count(shape: &XLAShape) -> u64 {
    let rank = shape.rank.min(shape.dimensions.len());
    shape.dimensions[..rank].iter().map(|&d| d as u64).product()
}

/// Byte size of a single element of the given XLA element type.
fn element_type_bytes(element_type: XLAElementType) -> usize {
    match element_type {
        XLAElementType::F16 | XLAElementType::BF16 => 2,
        XLAElementType::F32 | XLAElementType::S32 | XLAElementType::U32 => 4,
    }
}

/// Total byte size described by an XLA shape.
fn shape_byte_count(shape: &XLAShape) -> usize {
    (shape_element_count(shape) as usize).saturating_mul(element_type_bytes(shape.element_type))
}

/// Stable byte code for an XLA element type.
fn element_type_code(element_type: XLAElementType) -> u8 {
    match element_type {
        XLAElementType::F16 => 0,
        XLAElementType::F32 => 1,
        XLAElementType::BF16 => 2,
        XLAElementType::S32 => 3,
        XLAElementType::U32 => 4,
    }
}

/// Stable byte code for an XLA operation.
fn operation_code(operation: &XLAOperation) -> u8 {
    match operation {
        XLAOperation::Add => 0,
        XLAOperation::Multiply => 1,
        XLAOperation::Divide => 2,
        XLAOperation::MatMul => 3,
        XLAOperation::Reduce => 4,
        XLAOperation::Broadcast => 5,
        XLAOperation::Reshape => 6,
        XLAOperation::Transpose => 7,
        XLAOperation::Convolution => 8,
        XLAOperation::BatchNorm => 9,
        XLAOperation::Activation(_) => 10,
        XLAOperation::Custom(_) => 11,
    }
}

/// Stable byte code for an activation type.
fn activation_code(activation: ActivationType) -> u8 {
    match activation {
        ActivationType::ReLU => 0,
        ActivationType::Tanh => 1,
        ActivationType::Sigmoid => 2,
        ActivationType::Gelu => 3,
        ActivationType::Swish => 4,
    }
}

/// Stable byte code for an optimization pass.
fn pass_code(pass: &XLAOptimizationPass) -> u8 {
    match pass {
        XLAOptimizationPass::ConstantFolding => 0,
        XLAOptimizationPass::DeadCodeElimination => 1,
        XLAOptimizationPass::OperatorFusion => 2,
        XLAOptimizationPass::LayoutOptimization => 3,
        XLAOptimizationPass::MemoryOptimization => 4,
        XLAOptimizationPass::TensorCoreUtilization => 5,
    }
}

/// Encode an XLA shape into the program byte stream.
fn encode_shape(bytes: &mut Vec<u8>, shape: &XLAShape) {
    let rank = shape.rank.min(shape.dimensions.len());
    bytes.push(rank as u8);
    bytes.push(element_type_code(shape.element_type));
    for &dim in &shape.dimensions[..rank] {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
}

/// Encode an XLA operand (id + shape) into the program byte stream.
fn encode_operand(bytes: &mut Vec<u8>, operand: &XLAOperand) {
    bytes.extend_from_slice(&(operand.id as u64).to_le_bytes());
    encode_shape(bytes, &operand.shape);
}

/// Serialize a computation graph into a deterministic byte program.
///
/// The encoding is a stable function of the graph contents: a magic header and format
/// version, the XLA optimization level, the input operands (sorted by name), the node list,
/// the output operands and the optimization-pass pipeline. Identical graphs always produce
/// identical bytes, which is what makes the derived compilation id and `code_size`
/// reproducible.
fn encode_program(graph: &XLAComputationGraph) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OTPU");
    bytes.push(1); // format version
    bytes.push(graph.builder.optimization_level as u8);

    // Inputs, sorted by name for a deterministic encoding independent of HashMap order.
    let mut inputs: Vec<(&String, &XLAOperand)> = graph.inputs.iter().collect();
    inputs.sort_by(|a, b| a.0.cmp(b.0));
    bytes.extend_from_slice(&(inputs.len() as u32).to_le_bytes());
    for (name, operand) in inputs {
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        encode_operand(&mut bytes, operand);
    }

    // Nodes.
    bytes.extend_from_slice(&(graph.nodes.len() as u32).to_le_bytes());
    for node in &graph.nodes {
        bytes.push(operation_code(&node.operation));
        match &node.operation {
            XLAOperation::Activation(activation) => bytes.push(activation_code(*activation)),
            XLAOperation::Custom(name) => {
                bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
                bytes.extend_from_slice(name.as_bytes());
            }
            _ => {}
        }
        bytes.extend_from_slice(&(node.inputs.len() as u32).to_le_bytes());
        for operand in &node.inputs {
            encode_operand(&mut bytes, operand);
        }
        encode_shape(&mut bytes, &node.outputshape);
        bytes.extend_from_slice(&node.metadata.flops.to_le_bytes());
        bytes.extend_from_slice(&(node.metadata.memory_bytes as u64).to_le_bytes());
    }

    // Outputs.
    bytes.extend_from_slice(&(graph.outputs.len() as u32).to_le_bytes());
    for operand in &graph.outputs {
        encode_operand(&mut bytes, operand);
    }

    // Optimization pipeline.
    bytes.extend_from_slice(&(graph.optimization_passes.len() as u32).to_le_bytes());
    for pass in &graph.optimization_passes {
        bytes.push(pass_code(pass));
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpu_config_default() {
        let config = TPUConfig::default();
        assert_eq!(config.num_cores, 8);
        assert!(config.enable_xla);
        assert!(matches!(config.tpu_version, TPUVersion::V4));
    }

    // TPUOptimizer test disabled - requires optirs-core SGD optimizer
    // #[test]
    // fn test_tpu_optimizer_creation() {
    //     let sgd = SGD::new(0.01);
    //     let config = TPUConfig::default();
    //     let optimizer = TPUOptimizer::new(sgd, config);
    //     assert!(optimizer.is_ok());
    // }

    #[test]
    fn test_xlashape_creation() {
        let shape = XLAShape {
            dimensions: [10, 20, 1, 1],
            rank: 2,
            element_type: XLAElementType::F32,
        };

        assert_eq!(shape.rank, 2);
        assert_eq!(shape.dimensions[0], 10);
        assert_eq!(shape.dimensions[1], 20);
    }

    #[test]
    fn test_memory_allocator_creation() {
        let config = TPUConfig {
            tpu_version: TPUVersion::V4,
            num_cores: 8,
            ..Default::default()
        };

        let allocator = TPUMemoryAllocator::<f32>::new(&config);
        assert!(allocator.is_ok());

        let allocator = allocator.expect("unwrap failed");
        assert_eq!(allocator.total_memory, 32 * 1024 * 1024 * 1024 * 8); // 32GB * 8 cores
    }

    use optirs_core::optimizers::SGD;
    use scirs2_core::ndarray::Array1;

    fn new_sgd_tpu(config: TPUConfig) -> TPUOptimizer<SGD<f32>, f32> {
        TPUOptimizer::new(SGD::new(0.1f32), config).expect("failed to build TPU optimizer")
    }

    #[test]
    fn test_tpu_step_updates_params_in_descent_direction() {
        // Single-device path: tpu_step must actually run the wrapped SGD update on the CPU
        // and move parameters against the gradient, not return an error stub.
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        optimizer
            .initialize_xla_graph()
            .expect("xla graph init failed");

        let params = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let gradients = Array1::from(vec![1.0f32, 1.0, 1.0]);

        let updated = optimizer
            .tpu_step(&params, &gradients)
            .expect("tpu_step must succeed and return updated params");

        // SGD with lr=0.1 and unit gradient: p - 0.1 * 1 = p - 0.1
        let expected = [0.9f32, 1.9, 2.9];
        assert_eq!(updated.len(), 3);
        for (i, (&u, &e)) in updated.iter().zip(expected.iter()).enumerate() {
            assert!(
                (u - e).abs() < 1e-6,
                "index {i}: updated {u} != expected {e}"
            );
            // Descent: moved strictly below the original parameter value.
            assert!(
                u < params[i],
                "index {i}: {u} not below original {}",
                params[i]
            );
        }

        // A step was recorded.
        assert_eq!(optimizer.step_count, 1);
    }

    #[test]
    fn test_tpu_step_works_without_explicit_graph_init() {
        // build_optimizer_computation must fall back to a default graph, so tpu_step works
        // even if initialize_xla_graph was never called.
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        let params = Array1::from(vec![0.5f32, -0.5]);
        let gradients = Array1::from(vec![1.0f32, -1.0]);

        let updated = optimizer
            .tpu_step(&params, &gradients)
            .expect("tpu_step must succeed without prior graph init");

        assert!((updated[0] - 0.4).abs() < 1e-6);
        assert!((updated[1] - (-0.4)).abs() < 1e-6);
    }

    /// Generic helper that only compiles against `optirs_core::Optimizer`, never against
    /// `TPUOptimizer` directly. Regression test for F10 ("no Optimizer trait impl"): before
    /// the fix `TPUOptimizer` had no such impl, so nothing generic over `Optimizer` could
    /// accept it and this function would fail to instantiate at the call site below.
    fn run_one_generic_step<O: Optimizer<f32, scirs2_core::ndarray::Ix1>>(
        optimizer: &mut O,
        params: &Array1<f32>,
        gradients: &Array1<f32>,
    ) -> Array1<f32> {
        optimizer
            .step(params, gradients)
            .expect("generic Optimizer::step must succeed")
    }

    #[test]
    fn tpu_optimizer_is_usable_through_the_optimizer_trait() {
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        let params = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let gradients = Array1::from(vec![1.0f32, 1.0, 1.0]);

        let updated = run_one_generic_step(&mut optimizer, &params, &gradients);

        // Same arithmetic as `test_tpu_step_updates_params_in_descent_direction`: lr=0.1,
        // unit gradient, so `step` really drove `tpu_step` rather than a stub.
        let expected = [0.9f32, 1.9, 2.9];
        for (i, (&u, &e)) in updated.iter().zip(expected.iter()).enumerate() {
            assert!((u - e).abs() < 1e-6, "index {i}: {u} != {e}");
        }
        assert_eq!(optimizer.step_count, 1);
    }

    #[test]
    fn tpu_optimizer_learning_rate_forwards_to_base_optimizer() {
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        assert!((Optimizer::get_learning_rate(&optimizer) - 0.1).abs() < 1e-6);

        Optimizer::set_learning_rate(&mut optimizer, 0.5);
        assert!((Optimizer::get_learning_rate(&optimizer) - 0.5).abs() < 1e-6);

        // The new rate must actually be the one `step` uses, not a value the wrapper
        // tracks independently of `base_optimizer`.
        let params = Array1::from(vec![1.0f32]);
        let gradients = Array1::from(vec![1.0f32]);
        let updated =
            Optimizer::step(&mut optimizer, &params, &gradients).expect("step must succeed");
        assert!((updated[0] - 0.5).abs() < 1e-6, "got {}", updated[0]);
    }

    // Regression test: `Optimizer::step_list`'s *default* implementation calls
    // `step` once per tensor. For `TPUOptimizer`, `step` is `tpu_step`, which always
    // drives `base_optimizer.step` -- and `Adam::step` always uses moment-state index 0
    // (see `Adam::step`/`step_indexed`). So relying on the default would route every
    // tensor in the list through the *same* Adam state slot, silently mixing one
    // tensor's momentum into another's and resetting bias-correction timesteps whenever
    // shapes disagree (`Adam::advance_state` resets a slot's state, without erroring,
    // when it sees a shape different from what that slot last held). `TPUOptimizer`
    // overrides `step_list` to forward to `base_optimizer.step_list` directly, which
    // Adam implements by calling `step_indexed(index, ..)` -- one state slot per list
    // position -- specifically to avoid this. This test proves the override is wired in:
    // it compares the trait's `step_list` output against two independent, freshly-indexed
    // `step_indexed` calls (the definition of "isolated state"), and separately shows
    // that is NOT the same as what routing both tensors through index 0 would produce.
    #[test]
    fn tpu_optimizer_step_list_gives_each_tensor_its_own_optimizer_state() {
        use optirs_core::optimizers::Adam;

        let mut optimizer = TPUOptimizer::new(Adam::new(0.1f32), TPUConfig::default())
            .expect("failed to build TPU optimizer");

        let params_a = Array1::from(vec![1.0f32, 2.0]);
        let grads_a = Array1::from(vec![0.1f32, 0.1]);
        let params_b = Array1::from(vec![10.0f32, 20.0]);
        let grads_b = Array1::from(vec![0.5f32, 0.5]);

        let results = Optimizer::step_list(
            &mut optimizer,
            &[&params_a, &params_b],
            &[&grads_a, &grads_b],
        )
        .expect("step_list must succeed");
        assert_eq!(results.len(), 2);

        // Ground truth for "each tensor owns an independent, freshly-timestepped slot":
        // a brand-new Adam instance, called with the same per-position indices.
        let mut reference = Adam::new(0.1f32);
        let expected_a = reference
            .step_indexed(0, &params_a, &grads_a)
            .expect("reference step_indexed(0) must succeed");
        let expected_b = reference
            .step_indexed(1, &params_b, &grads_b)
            .expect("reference step_indexed(1) must succeed");

        for i in 0..2 {
            assert!(
                (results[0][i] - expected_a[i]).abs() < 1e-6,
                "tensor 0 index {i}: {} != {}",
                results[0][i],
                expected_a[i]
            );
            assert!(
                (results[1][i] - expected_b[i]).abs() < 1e-6,
                "tensor 1 index {i}: {} != {}",
                results[1][i],
                expected_b[i]
            );
        }

        // Demonstrate this is not a vacuous comparison: routing both tensors through
        // the *same* slot (what the unfixed default `step_list` would do via `tpu_step`
        // -> `Adam::step` -> `step_indexed(0, ..)` every time) gives a materially
        // different result for the second tensor, because it inherits tensor A's
        // momentum and a bias-correction timestep of 2 instead of a fresh 1.
        let mut shared_slot = Adam::new(0.1f32);
        let _ = shared_slot
            .step_indexed(0, &params_a, &grads_a)
            .expect("shared-slot step_indexed(0) [a] must succeed");
        let shared_slot_b = shared_slot
            .step_indexed(0, &params_b, &grads_b)
            .expect("shared-slot step_indexed(0) [b] must succeed");
        let materially_different = (0..2).any(|i| (shared_slot_b[i] - expected_b[i]).abs() > 1e-4);
        assert!(
            materially_different,
            "expected sharing one state slot to diverge from independent per-tensor state, \
             got shared={shared_slot_b:?} independent={expected_b:?}"
        );
    }

    #[test]
    fn test_tpu_step_distributed_matches_single_device() {
        // Data-parallel path with a pod coordinator: averaging identical replicas is the
        // identity, so the distributed update must match the single-device update.
        let config = TPUConfig {
            enable_pod_coordination: true,
            pod_topology: PodTopology::Pod2x2,
            ..Default::default()
        };
        let mut optimizer = new_sgd_tpu(config);
        assert!(optimizer.pod_coordinator.is_some());

        let params = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let gradients = Array1::from(vec![2.0f32, 2.0, 2.0]);

        let updated = optimizer
            .tpu_step(&params, &gradients)
            .expect("distributed tpu_step must succeed");

        // p - 0.1 * 2 = p - 0.2
        let expected = [0.8f32, 1.8, 2.8];
        for (&u, &e) in updated.iter().zip(expected.iter()) {
            assert!((u - e).abs() < 1e-6, "updated {u} != expected {e}");
        }

        // The all-reduce collective was recorded as a communication event.
        assert!(optimizer
            .profiler
            .timeline
            .iter()
            .any(|event| matches!(event.event_type, ProfileEventType::Communication)));
    }

    #[test]
    fn test_tpu_step_shape_mismatch_errors() {
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        let params = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let gradients = Array1::from(vec![1.0f32, 1.0]);
        assert!(optimizer.tpu_step(&params, &gradients).is_err());
    }

    #[test]
    fn test_compile_to_tpu_produces_real_code_and_metrics() {
        let optimizer = new_sgd_tpu(TPUConfig::default());
        let shape = XLAShape {
            dimensions: [4, 1, 1, 1],
            rank: 1,
            element_type: XLAElementType::F32,
        };
        let graph = optimizer
            .build_optimizer_computation(&[shape, shape])
            .expect("graph build failed");
        let compiled = optimizer
            .compile_to_tpu(graph)
            .expect("compile_to_tpu failed");

        // Code is a real serialized program: begins with the magic header, is longer than
        // the header, and is not the all-zero placeholder.
        assert!(
            compiled.code.len() > 4,
            "code too short: {}",
            compiled.code.len()
        );
        assert_eq!(&compiled.code[0..4], b"OTPU");
        assert!(
            compiled.code.iter().any(|&b| b != 0),
            "code must not be all zero"
        );

        // FLOPs derived from element count: 2 inputs * 4 elements * 2 flops/element = 16.
        assert_eq!(compiled.perf_characteristics.flops, 16);

        // Utilization derived from element count; strictly within (0, 1).
        let util = compiled.perf_characteristics.utilization_estimate;
        assert!(util > 0.0 && util < 1.0, "utilization out of range: {util}");

        // Execution time has a 1us floor and is not the old 100us literal.
        assert!(compiled.perf_characteristics.estimated_execution_time_us >= 1);

        // Memory derived from real byte counts: 2 * (4 elements * 4 bytes) inputs.
        assert!(compiled.memory_requirements.working_memory > 0);
        assert!(
            compiled.memory_requirements.total_memory
                >= compiled.memory_requirements.working_memory
        );
    }

    #[test]
    fn test_compile_to_tpu_is_deterministic() {
        let optimizer = new_sgd_tpu(TPUConfig::default());
        let shape = XLAShape {
            dimensions: [8, 1, 1, 1],
            rank: 1,
            element_type: XLAElementType::F32,
        };
        let graph_a = optimizer
            .build_optimizer_computation(&[shape, shape])
            .expect("graph build failed");
        let graph_b = optimizer
            .build_optimizer_computation(&[shape, shape])
            .expect("graph build failed");
        let a = optimizer.compile_to_tpu(graph_a).expect("compile failed");
        let b = optimizer.compile_to_tpu(graph_b).expect("compile failed");
        assert_eq!(a.code, b.code);
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn test_optimizations_applied_counts_only_effective_passes() {
        // The default graph has no nodes, so no optimization pass changes anything and
        // `optimizations_applied` must be 0 (not the pipeline length of 6). `code_size`
        // must reflect the real generated program.
        let mut optimizer = new_sgd_tpu(TPUConfig::default());
        optimizer
            .initialize_xla_graph()
            .expect("xla graph init failed");
        let shape = XLAShape {
            dimensions: [4, 1, 1, 1],
            rank: 1,
            element_type: XLAElementType::F32,
        };
        optimizer
            .compile_step(&[shape, shape])
            .expect("compile_step failed");

        assert_eq!(
            optimizer.profiler.compilation_metrics.optimizations_applied,
            0
        );
        assert!(optimizer.profiler.compilation_metrics.code_size > 0);
    }
}
