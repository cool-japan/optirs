use std::fmt::Debug;
// XLA (Accelerated Linear Algebra) Compilation for TPU Optimization
//
// This module implements comprehensive XLA compilation capabilities for TPU-optimized
// optimization algorithms. It provides high-level abstractions for building, optimizing,
// and executing XLA computations on TPU hardware.

pub mod backend;
pub mod execution;
pub mod frontend;
pub mod optimization;

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::{TPUConfig, TPUVersion, XLAOptimizationLevel};
use crate::error::{OptimError, Result};

// Note: Submodules have conflicting type names (ExecutionScheduler, ResourceManager, MemorySpace)
// Users should access types via full module paths (e.g., backend::ExecutionScheduler)

// Selective re-exports of non-conflicting types used in this module
use backend::XLABackend;
use frontend::{OperationLowering, ShapeInference};
use optimization::{MemoryPlanner, OptimizationPipeline, PerformanceAnalyzer};

// Re-export for public API
pub use execution::{ReferenceExecutor, ValueMap};
pub use frontend::{ComputationId, XLAComputation};

/// XLA Compiler for TPU optimization
pub struct XLACompiler<T: Float + Debug + Send + Sync + 'static> {
    /// Compiler configuration
    config: XLACompilerConfig,

    /// Optimization pipeline
    optimization_pipeline: OptimizationPipeline<T>,

    /// Code-generation and runtime-integration backend.
    ///
    /// This compiler used to own a bare [`TPUCodeGenerator`] and mint a throwaway
    /// [`RuntimeIntegration`] per compile, duplicating exactly what
    /// [`backend::XLABackend`] already does -- two parallel implementations of
    /// "graph plus memory plan to binary", only one of which fed the profiling
    /// integration. `XLABackend` is now the single source of truth for that
    /// step: it owns the code generator, the runtime manager and the profiler,
    /// and this compiler drives it.
    backend: XLABackend<T>,

    /// Compilation cache
    compilation_cache: Arc<RwLock<CompilationCache>>,

    /// Performance analyzer
    performance_analyzer: PerformanceAnalyzer<T>,

    /// Memory planner
    memory_planner: MemoryPlanner<T>,

    /// Work queue backing [`Self::compile_batch`].
    parallel_compiler: ParallelCompilationManager<T>,

    /// Profiling data
    profiling_data: ProfilingData,
}

/// XLA compiler configuration
#[derive(Debug, Clone)]
pub struct XLACompilerConfig {
    /// Target TPU configuration
    pub target_tpu: TPUConfig,

    /// Optimization level
    pub optimization_level: XLAOptimizationLevel,

    /// Enable auto-tuning
    pub enable_auto_tuning: bool,

    /// Compilation timeout (seconds)
    pub compilation_timeout: u64,

    /// Maximum cache size (MB)
    pub max_cache_size_mb: usize,

    /// Enable parallel compilation
    pub parallel_compilation: bool,

    /// Number of compilation threads
    pub compilation_threads: usize,

    /// Enable fusion optimization
    pub enable_fusion: bool,

    /// Enable layout optimization
    pub enable_layout_optimization: bool,

    /// Enable memory optimization
    pub enable_memory_optimization: bool,

    /// Enable pipeline optimization
    pub enable_pipeline_optimization: bool,

    /// Debug mode
    pub debug_mode: bool,

    /// Profile compilation
    pub profile_compilation: bool,

    /// Custom optimization passes
    pub custom_passes: Vec<String>,

    /// Enable advanced tensor core optimizations
    pub enable_tensor_core_optimization: bool,

    /// Enable sparsity-aware optimizations
    pub enable_sparsity_optimization: bool,
}

/// Compilation cache for reusing compiled computations
#[derive(Debug)]
pub struct CompilationCache {
    /// Cached compiled computations
    pub cache: HashMap<String, CachedComputation>,

    /// Cache statistics
    pub stats: CacheStatistics,

    /// Maximum cache size
    pub max_size: usize,

    /// Current cache size
    pub current_size: usize,
}

/// Cached compilation result
#[derive(Debug, Clone)]
pub struct CachedComputation {
    /// Computation identifier
    pub id: String,

    /// Compiled binary
    pub binary: Vec<u8>,

    /// Compilation metadata
    pub metadata: CompilationMetadata,

    /// Last access time
    pub last_accessed: Instant,

    /// Access count
    pub access_count: u64,

    /// Size in bytes
    pub size: usize,
}

/// Compilation metadata
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    /// Source computation hash
    pub computation_hash: String,

    /// Compiler version
    pub compiler_version: String,

    /// Target TPU configuration
    pub target_config: TPUConfig,

    /// Compilation time
    pub compilation_time: Duration,

    /// Optimization passes applied
    pub optimization_passes: Vec<String>,

    /// Performance characteristics
    pub performance_info: PerformanceInfo,
}

/// Performance information
#[derive(Debug, Clone, Default)]
pub struct PerformanceInfo {
    /// Estimated execution time (microseconds)
    pub estimated_execution_time: u64,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// Flop count
    pub flop_count: u64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f64,

    /// Compute utilization
    pub compute_utilization: f64,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,

    /// Total cache misses
    pub misses: u64,

    /// Total evictions
    pub evictions: u64,

    /// Hit rate
    pub hit_rate: f64,
}

/// Profiling data for compilation analysis
#[derive(Debug, Default)]
pub struct ProfilingData {
    /// Pass execution times
    pub pass_times: HashMap<String, Duration>,

    /// Memory usage per pass
    pub pass_memory_usage: HashMap<String, usize>,

    /// Total compilation time
    pub total_compilation_time: Duration,

    /// Peak memory usage
    pub peak_memory_usage: usize,

    /// Number of operations processed
    pub operations_processed: usize,
}

/// Work queue for batched compilation.
///
/// There is deliberately no thread-pool handle here: see
/// [`XLACompiler::compile_batch`] for why the drain is serial. A `JoinHandle`
/// field that is never populated would claim a capability this type does not
/// have.
#[derive(Debug)]
pub struct ParallelCompilationManager<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum number of tasks admitted per drain
    pub num_threads: usize,

    /// Compilation queue
    pub compilation_queue: VecDeque<CompilationTask<T>>,

    /// Active compilations
    pub active_compilations: HashMap<String, CompilationProgress>,
}

/// Compilation task
#[derive(Debug)]
pub struct CompilationTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub id: String,

    /// Computation to compile
    pub computation: XLAComputation<T>,

    /// Compilation configuration
    pub config: XLACompilerConfig,

    /// Priority level
    pub priority: u8,

    /// Creation time
    pub created_at: Instant,
}

/// Compilation progress tracking
#[derive(Debug)]
pub struct CompilationProgress {
    /// Current phase
    pub current_phase: CompilationPhase,

    /// Progress percentage (0.0-1.0)
    pub progress: f64,

    /// Start time
    pub started_at: Instant,

    /// Estimated completion time
    pub estimated_completion: Option<Instant>,
}

/// Compilation phases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompilationPhase {
    GraphCapture,
    ShapeInference,
    OperationLowering,
    GraphOptimization,
    KernelFusion,
    MemoryPlanning,
    Scheduling,
    CodeGeneration,
    RuntimeIntegration,
    Finalization,
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> XLACompiler<T> {
    /// Create new XLA compiler
    pub fn new(config: XLACompilerConfig) -> Result<Self> {
        let optimization_pipeline = OptimizationPipeline::new(&config);
        let backend = XLABackend::new(backend::BackendConfig {
            target_tpu: config.target_tpu.clone(),
            enable_optimized_codegen: config.optimization_level != XLAOptimizationLevel::None,
            enable_profiling: config.profile_compilation,
            debug_mode: config.debug_mode,
            verification_mode: config.debug_mode,
            custom_options: HashMap::new(),
        });
        let compilation_cache =
            Arc::new(RwLock::new(CompilationCache::new(config.max_cache_size_mb)));
        let performance_analyzer = PerformanceAnalyzer::new();
        let memory_planner = MemoryPlanner::new(config.target_tpu.clone());
        let parallel_compiler = ParallelCompilationManager::new(config.compilation_threads);

        Ok(Self {
            config,
            optimization_pipeline,
            backend,
            compilation_cache,
            performance_analyzer,
            memory_planner,
            parallel_compiler,
            profiling_data: ProfilingData::default(),
        })
    }

    /// Compile XLA computation
    pub fn compile(&mut self, computation: XLAComputation<T>) -> Result<CompiledComputation> {
        let start_time = Instant::now();

        // Check cache first
        let computation_hash = self.compute_hash(&computation);
        if let Some(cached) = self.get_cached_computation(&computation_hash)? {
            let execution_info =
                ExecutionInfo::from_performance_info(&cached.metadata.performance_info);
            return Ok(CompiledComputation {
                binary: cached.binary,
                metadata: cached.metadata,
                execution_info,
            });
        }

        // Run compilation pipeline, timing every phase into `profiling_data` so
        // the recorded pass timings are the real ones rather than an empty map.
        let phase_start = Instant::now();
        let shaped_computation = self.run_shape_inference(computation)?;
        self.record_phase("shape_inference", phase_start.elapsed());

        let phase_start = Instant::now();
        let lowered_computation = self.run_operation_lowering(shaped_computation)?;
        self.record_phase("operation_lowering", phase_start.elapsed());

        let phase_start = Instant::now();
        let optimized_computation = self.optimization_pipeline.optimize(lowered_computation)?;
        self.record_phase("graph_optimization", phase_start.elapsed());

        let phase_start = Instant::now();
        let memory_plan = self
            .memory_planner
            .create_memory_plan(&optimized_computation)?;
        self.record_phase("memory_planning", phase_start.elapsed());

        // Code generation and runtime integration, both owned by the single
        // backend rather than duplicated here.
        let phase_start = Instant::now();
        let binary = self
            .backend
            .compile_and_integrate(&optimized_computation, &memory_plan)?;
        self.record_phase("code_generation", phase_start.elapsed());

        // Real performance information derived from the graph and memory plan
        // that were actually produced.
        let performance_info = self.performance_analyzer.analyze(
            &optimized_computation,
            &memory_plan,
            &self.config.target_tpu,
        );

        self.profiling_data.operations_processed += optimized_computation.operations.len();
        self.profiling_data.peak_memory_usage = self
            .profiling_data
            .peak_memory_usage
            .max(memory_plan.total_memory);
        self.profiling_data.total_compilation_time += start_time.elapsed();

        // Cache the result
        let metadata = CompilationMetadata {
            computation_hash: computation_hash.clone(),
            compiler_version: "1.0.0".to_string(),
            target_config: self.config.target_tpu.clone(),
            compilation_time: start_time.elapsed(),
            optimization_passes: self.optimization_pipeline.get_applied_passes(),
            performance_info,
        };

        self.cache_computation(computation_hash, binary.clone(), metadata.clone())?;

        let execution_info = ExecutionInfo::from_performance_info(&metadata.performance_info);
        Ok(CompiledComputation {
            binary,
            metadata,
            execution_info,
        })
    }

    /// Compile a batch of computations through the work queue.
    ///
    /// Tasks are enqueued with their priority, then drained highest-priority
    /// first (ties broken by submission order, so the schedule is deterministic)
    /// and each is compiled through [`Self::compile`]. Progress for the task
    /// currently in flight is visible via [`Self::compilation_status`].
    ///
    /// The drain is **serial**: the compilation pipeline runs behind `&mut self`
    /// and its sub-managers (graph optimizer, memory planner, code generator)
    /// are not shareable across threads, so dispatching phases onto
    /// `compilation_threads` OS threads would require restructuring the whole
    /// pipeline around owned per-task state. `compilation_threads` therefore
    /// bounds how many tasks the queue admits per drain rather than spawning
    /// threads, and `parallel_compilation == false` disables batching entirely
    /// so each computation is compiled as it arrives. That is stated here
    /// instead of being implied by a thread count that spawns nothing.
    pub fn compile_batch(
        &mut self,
        computations: Vec<XLAComputation<T>>,
    ) -> Result<Vec<CompiledComputation>> {
        if !self.config.parallel_compilation {
            return computations
                .into_iter()
                .map(|computation| self.compile(computation))
                .collect();
        }

        for computation in computations {
            let task = CompilationTask {
                id: format!("comp_{}", computation.id.0),
                priority: compilation_priority(&computation),
                computation,
                config: self.config.clone(),
                created_at: Instant::now(),
            };
            self.parallel_compiler.submit_task(task);
        }

        let batch = self.parallel_compiler.take_batch();
        let mut results = Vec::with_capacity(batch.len());
        for task in batch {
            self.parallel_compiler
                .begin_task(&task.id, CompilationPhase::GraphCapture);
            let compiled = self.compile(task.computation);
            self.parallel_compiler.finish_task(&task.id);
            results.push(compiled?);
        }
        Ok(results)
    }

    /// Progress of a task currently being compiled by [`Self::compile_batch`].
    pub fn compilation_status(&self, task_id: &str) -> Option<&CompilationProgress> {
        self.parallel_compiler.get_status(task_id)
    }

    /// Statistics gathered across every [`Self::compile`] call so far.
    pub fn profiling_data(&self) -> &ProfilingData {
        &self.profiling_data
    }

    /// Backend statistics for the most recent code-generation step.
    pub fn backend_statistics(&self) -> &backend::BackendStatistics {
        self.backend.get_statistics()
    }

    /// The profiling integration the backend maintains for compiled programs.
    pub fn profiling(&self) -> &backend::ProfilingIntegration<T> {
        self.backend.profiling()
    }

    /// Mutable access to that profiling integration, so a runtime built on this
    /// compiler (see [`crate::tpu_backend::TPUBackend`]) can record the
    /// execution-side events -- device memory reservations above all -- into
    /// the same profile the compile step opened.
    pub fn profiling_mut(&mut self) -> &mut backend::ProfilingIntegration<T> {
        self.backend.profiling_mut()
    }

    /// Record a compilation phase's real duration and the memory the pipeline
    /// was holding when it finished.
    fn record_phase(&mut self, phase: &str, elapsed: Duration) {
        *self
            .profiling_data
            .pass_times
            .entry(phase.to_string())
            .or_insert(Duration::ZERO) += elapsed;
        self.profiling_data
            .pass_memory_usage
            .insert(phase.to_string(), self.profiling_data.peak_memory_usage);
    }

    /// Run shape inference on computation
    fn run_shape_inference(&self, computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        // Delegate to frontend shape inference
        ShapeInference::infer_shapes(computation)
    }

    /// Run operation lowering
    fn run_operation_lowering(&self, computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        // Delegate to frontend operation lowering
        OperationLowering::lower_operations(computation)
    }

    /// Content hash of `computation`, used as the compilation-cache key.
    ///
    /// This must be a function of the graph's *contents*, not of
    /// [`ComputationId`]: ids are minted per builder and a caller that rebuilds
    /// a different graph under a reused id would otherwise be served the
    /// previous graph's binary. The digest therefore covers every operation
    /// (its type and attributes, via `Debug`, plus its operand wiring) and the
    /// declared input/output interface, in the graph's own deterministic order.
    fn compute_hash(&self, computation: &XLAComputation<T>) -> String {
        use std::fmt::Write as _;

        let mut canonical = String::new();
        // Writing into a String is infallible; `write!`'s Result is discarded
        // deliberately rather than propagated as an impossible error.
        let _ = write!(canonical, "id={};", computation.id.0);
        for operation in &computation.operations {
            let _ = write!(
                canonical,
                "op{}:{:?}<{:?}>->{:?}:{:?};",
                operation.id.0,
                operation.op_type,
                operation.inputs,
                operation.output,
                operation.attributes,
            );
        }
        for input in &computation.inputs {
            let _ = write!(
                canonical,
                "in{}:{:?}:{:?}:{:?};",
                input.index, input.operand, input.dtype, input.shape.dimensions,
            );
        }
        for output in &computation.outputs {
            let _ = write!(
                canonical,
                "out{}:{:?}:{:?}:{:?};",
                output.index, output.operand, output.dtype, output.shape.dimensions,
            );
        }

        format!("comp_{:016x}", fnv1a_64(canonical.as_bytes()))
    }

    /// Look `hash` up in the compilation cache.
    ///
    /// Takes the write lock because a lookup genuinely mutates: it records the
    /// hit or miss in [`CacheStatistics`] and refreshes the entry's LRU
    /// bookkeeping. While this was a read-only borrow, `stats` and
    /// `last_accessed`/`access_count` stayed at their constructed defaults
    /// forever, so `hit_rate` always read zero and `evict_lru` always saw every
    /// entry as equally old.
    fn get_cached_computation(&self, hash: &str) -> Result<Option<CachedComputation>> {
        let mut cache = self
            .compilation_cache
            .write()
            .map_err(|_| OptimError::from("compilation cache lock poisoned".to_string()))?;

        let hit = match cache.cache.get_mut(hash) {
            Some(entry) => {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
                Some(entry.clone())
            }
            None => None,
        };
        cache.record_lookup(hit.is_some());
        Ok(hit)
    }

    /// Cache compiled computation
    fn cache_computation(
        &self,
        hash: String,
        binary: Vec<u8>,
        metadata: CompilationMetadata,
    ) -> Result<()> {
        let mut cache = self
            .compilation_cache
            .write()
            .map_err(|_| OptimError::from("compilation cache lock poisoned".to_string()))?;

        let binary_size = binary.len();
        let cached_comp = CachedComputation {
            id: hash,
            binary,
            metadata,
            last_accessed: Instant::now(),
            access_count: 0,
            size: binary_size,
        };

        cache.insert(cached_comp);
        Ok(())
    }

    /// Snapshot of the compilation cache's hit/miss/eviction counters.
    ///
    /// Recovers from a poisoned lock rather than failing: a panic in another
    /// thread cannot invalidate the counters, and statistics are never worth
    /// propagating an error for.
    pub fn cache_statistics(&self) -> CacheStatistics {
        self.compilation_cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .stats
            .clone()
    }
}

/// FNV-1a 64-bit hash: small, dependency-free and deterministic across runs,
/// which is what a cache key derived from graph contents needs.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Compiled computation result
#[derive(Debug)]
pub struct CompiledComputation {
    /// Compiled binary
    pub binary: Vec<u8>,

    /// Compilation metadata
    pub metadata: CompilationMetadata,

    /// Execution information
    pub execution_info: ExecutionInfo,
}

/// Execution information
#[derive(Debug, Default)]
pub struct ExecutionInfo {
    /// Estimated execution time
    pub estimated_time: Duration,

    /// Memory requirements
    pub memory_requirements: usize,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

impl ExecutionInfo {
    /// Project the compiler's derived [`PerformanceInfo`] onto the
    /// execution-side view, so a caller of [`XLACompiler::compile`] sees the
    /// real analysis instead of an all-zero default.
    ///
    /// `interconnect` is left at zero: this compiler produces a single-device
    /// binary and has no cross-device transfer to measure, so any non-zero
    /// figure here would be invented.
    fn from_performance_info(info: &PerformanceInfo) -> Self {
        Self {
            estimated_time: Duration::from_micros(info.estimated_execution_time),
            memory_requirements: info.memory_usage,
            resource_utilization: ResourceUtilization {
                compute: info.compute_utilization,
                memory_bandwidth: info.memory_bandwidth_util,
                interconnect: 0.0,
            },
        }
    }
}

/// Resource utilization information
#[derive(Debug, Default)]
pub struct ResourceUtilization {
    /// Compute utilization
    pub compute: f64,

    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,

    /// Interconnect utilization
    pub interconnect: f64,
}

/// Generated code structure
#[derive(Debug)]
pub struct GeneratedCode {
    /// Main computation kernel
    pub kernel_code: String,

    /// Initialization code
    pub init_code: String,

    /// Cleanup code
    pub cleanup_code: String,

    /// Memory management code
    pub memory_code: String,
}

impl CompilationCache {
    /// Create new compilation cache
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: HashMap::new(),
            stats: CacheStatistics::default(),
            max_size: max_size_mb * 1024 * 1024, // Convert to bytes
            current_size: 0,
        }
    }

    /// Check if cache needs eviction
    pub fn needs_eviction(&self, new_size: usize) -> bool {
        self.current_size.saturating_add(new_size) > self.max_size
    }

    /// Record a lookup outcome and refresh the derived hit rate.
    pub fn record_lookup(&mut self, hit: bool) {
        if hit {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        let total = self.stats.hits + self.stats.misses;
        self.stats.hit_rate = if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        };
    }

    /// Insert a compiled entry, evicting least-recently-used entries first so
    /// the cache actually honours `max_size`.
    ///
    /// `current_size` used to never be incremented, which made
    /// [`Self::needs_eviction`] permanently false and left `max_size` and
    /// [`Self::evict_lru`] as decoration on an unbounded map.
    ///
    /// An entry larger than the entire budget is not cached at all: evicting
    /// every other entry for it would still leave the cache over budget, so
    /// refusing is the only outcome that keeps the bound true.
    pub fn insert(&mut self, cached: CachedComputation) {
        let size = cached.size;
        if size > self.max_size {
            return;
        }

        if let Some(previous) = self.cache.remove(&cached.id) {
            self.current_size = self.current_size.saturating_sub(previous.size);
        }

        if self.needs_eviction(size) {
            let over_budget = self
                .current_size
                .saturating_add(size)
                .saturating_sub(self.max_size);
            self.evict_lru(over_budget);
        }

        self.current_size = self.current_size.saturating_add(size);
        self.cache.insert(cached.id.clone(), cached);
    }

    /// Evict least recently used items
    pub fn evict_lru(&mut self, target_size: usize) {
        let mut items: Vec<_> = self.cache.iter().collect();
        items.sort_by_key(|(_, cached)| cached.last_accessed);

        let mut freed_size = 0;
        let mut to_remove = Vec::new();

        for (key, cached) in items {
            if freed_size >= target_size {
                break;
            }
            freed_size += cached.size;
            to_remove.push(key.clone());
        }

        for key in to_remove {
            if let Some(cached) = self.cache.remove(&key) {
                self.current_size = self.current_size.saturating_sub(cached.size);
                self.stats.evictions += 1;
            }
        }
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Send + Sync> ParallelCompilationManager<T> {
    /// Create new parallel compilation manager
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            compilation_queue: VecDeque::new(),
            active_compilations: HashMap::new(),
        }
    }

    /// Submit compilation task
    pub fn submit_task(&mut self, task: CompilationTask<T>) {
        self.compilation_queue.push_back(task);
    }

    /// Take up to `num_threads` queued tasks, highest priority first.
    ///
    /// The sort is stable, so equal-priority tasks keep submission order and the
    /// resulting schedule is deterministic.
    pub fn take_batch(&mut self) -> Vec<CompilationTask<T>> {
        let mut queued: Vec<CompilationTask<T>> = self.compilation_queue.drain(..).collect();
        queued.sort_by_key(|task| std::cmp::Reverse(task.priority));
        let overflow = queued.split_off(queued.len().min(self.num_threads));
        // Anything beyond this drain stays queued for the next one.
        for task in overflow {
            self.compilation_queue.push_back(task);
        }
        queued
    }

    /// Mark a task as in flight at `phase`.
    pub fn begin_task(&mut self, task_id: &str, phase: CompilationPhase) {
        self.active_compilations.insert(
            task_id.to_string(),
            CompilationProgress {
                current_phase: phase,
                progress: 0.0,
                started_at: Instant::now(),
                estimated_completion: None,
            },
        );
    }

    /// Remove a finished task from the in-flight set.
    pub fn finish_task(&mut self, task_id: &str) {
        self.active_compilations.remove(task_id);
    }

    /// Get compilation status
    pub fn get_status(&self, task_id: &str) -> Option<&CompilationProgress> {
        self.active_compilations.get(task_id)
    }

    /// Number of tasks still waiting.
    pub fn queued(&self) -> usize {
        self.compilation_queue.len()
    }
}

/// Priority for a queued compilation: larger graphs first, so the longest job
/// starts earliest in a batch. Saturates at the priority field's width.
fn compilation_priority<T: Float + Debug + Send + Sync + 'static>(
    computation: &XLAComputation<T>,
) -> u8 {
    u8::try_from(computation.operations.len().min(u8::MAX as usize)).unwrap_or(u8::MAX)
}

impl Default for XLACompilerConfig {
    fn default() -> Self {
        Self {
            target_tpu: TPUConfig::default(),
            optimization_level: XLAOptimizationLevel::Standard,
            enable_auto_tuning: true,
            compilation_timeout: 300, // 5 minutes
            max_cache_size_mb: 1024,  // 1 GB
            parallel_compilation: true,
            compilation_threads: num_cpus::get(),
            enable_fusion: true,
            enable_layout_optimization: true,
            enable_memory_optimization: true,
            enable_pipeline_optimization: true,
            debug_mode: false,
            profile_compilation: false,
            custom_passes: Vec::new(),
            enable_tensor_core_optimization: true,
            enable_sparsity_optimization: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_compiler_creation() {
        let config = XLACompilerConfig::default();
        let compiler: Result<XLACompiler<f32>> = XLACompiler::new(config);
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_compilation_cache() {
        let cache = CompilationCache::new(10); // 10 MB
        assert_eq!(cache.current_size, 0);
        assert!(!cache.needs_eviction(100));
    }

    fn cached(id: &str, size: usize) -> CachedComputation {
        CachedComputation {
            id: id.to_string(),
            binary: vec![0u8; size],
            metadata: CompilationMetadata {
                computation_hash: id.to_string(),
                compiler_version: "test".to_string(),
                target_config: TPUConfig::default(),
                compilation_time: Duration::ZERO,
                optimization_passes: Vec::new(),
                performance_info: PerformanceInfo::default(),
            },
            last_accessed: Instant::now(),
            access_count: 0,
            size,
        }
    }

    /// `current_size` tracks what is actually stored, so `max_size` bounds the
    /// cache and the least-recently-used entry is the one that goes.
    #[test]
    fn cache_evicts_to_stay_within_its_budget() {
        // `CompilationCache::new` takes megabytes; build the budget directly so
        // the test can work in bytes.
        let mut cache = CompilationCache::new(0);
        cache.max_size = 300;

        cache.insert(cached("a", 100));
        cache.insert(cached("b", 100));
        assert_eq!(cache.current_size, 200);
        assert_eq!(cache.stats.evictions, 0);

        // Touch "a" so "b" becomes the least recently used entry.
        if let Some(entry) = cache.cache.get_mut("a") {
            entry.last_accessed = Instant::now();
        }

        cache.insert(cached("c", 150));
        assert!(
            cache.current_size <= cache.max_size,
            "current_size {} must respect the {} byte budget",
            cache.current_size,
            cache.max_size
        );
        assert_eq!(cache.stats.evictions, 1);
        assert!(cache.cache.contains_key("c"));
        assert!(!cache.cache.contains_key("b"), "the LRU entry is evicted");
    }

    /// An entry that cannot fit even in an empty cache is refused rather than
    /// evicting everything and still blowing the budget.
    #[test]
    fn an_oversized_entry_is_not_cached() {
        let mut cache = CompilationCache::new(0);
        cache.max_size = 100;
        cache.insert(cached("huge", 1_000));
        assert!(cache.cache.is_empty());
        assert_eq!(cache.current_size, 0);
    }

    /// Lookups move the hit/miss counters and the derived hit rate.
    #[test]
    fn lookups_are_counted() {
        let mut cache = CompilationCache::new(1);
        cache.record_lookup(false);
        assert_eq!(cache.stats.misses, 1);
        assert_eq!(cache.stats.hit_rate, 0.0);
        cache.record_lookup(true);
        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.hit_rate, 0.5);
    }

    #[test]
    fn test_parallel_compilation_manager() {
        let manager: ParallelCompilationManager<f32> = ParallelCompilationManager::new(4);
        assert_eq!(manager.num_threads, 4);
        assert!(manager.compilation_queue.is_empty());
    }
}
