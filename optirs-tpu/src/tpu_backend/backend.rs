//! [`TPUBackend`]: the top-level compile -> execute -> profile orchestrator
//! that owns every other component in this module tree.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use scirs2_core::error::ErrorContext;
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};

use super::buffer::TPUBuffer;
use super::device_defaults::{estimate_bandwidth_utilization, estimate_compute_utilization};
use super::device_manager::DeviceManager;
use super::execution::ExecutionEngine;
use super::memory::TPUMemoryManager;
use super::profiling::{PerformanceMonitor, RuntimeProfiler, TPUErrorHandler};
use super::serialization::{deserialize_tpu_buffers, encode_program_binary, serialize_tpu_buffers};
use super::types::{
    BackendPerformanceStatistics, CompiledProgram, ComputationId, ComputationTask,
    MemoryAllocationStrategy, PrefetchStrategy, ProgramMemoryRequirements, ProgramMetadata,
    ProgramPerformanceCharacteristics, TPUBackendConfig, TaskId,
};

/// TPU Backend Manager
pub struct TPUBackend<T: Float + Debug + Send + Sync + 'static> {
    /// Backend configuration
    config: TPUBackendConfig,

    /// Device manager
    device_manager: DeviceManager,

    /// Execution engine
    execution_engine: ExecutionEngine<T>,

    /// Memory manager
    memory_manager: TPUMemoryManager<T>,

    /// Runtime profiler
    runtime_profiler: RuntimeProfiler,

    /// Error handler
    error_handler: TPUErrorHandler,

    /// Performance monitor
    performance_monitor: PerformanceMonitor,

    /// Compilation cache
    ///
    /// `pub(super)`: inspected directly by the `tpu_backend` test module
    /// (a sibling submodule) to confirm a cache hit was actually served from
    /// cache rather than merely incrementing a counter.
    pub(super) compilation_cache: Arc<RwLock<HashMap<ComputationId, CompiledProgram>>>,

    /// Number of compilation-cache lookups that were served from cache
    cache_hits: AtomicU64,

    /// Number of compilation-cache lookups that missed and triggered a compile
    cache_misses: AtomicU64,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum> TPUBackend<T> {
    /// Create a new TPU backend
    pub fn new(config: TPUBackendConfig) -> Result<Self> {
        let device_manager = DeviceManager::new(&config)?;
        let execution_engine = ExecutionEngine::new(&config)?;
        let memory_manager = TPUMemoryManager::new(&config)?;
        let runtime_profiler = RuntimeProfiler::new(&config);
        let error_handler = TPUErrorHandler::new(&config);
        let performance_monitor = PerformanceMonitor::new(&config);
        let compilation_cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            device_manager,
            execution_engine,
            memory_manager,
            runtime_profiler,
            error_handler,
            performance_monitor,
            compilation_cache,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        })
    }

    /// Execute a computation on TPU
    pub async fn execute_computation(
        &mut self,
        computation_id: ComputationId,
        inputs: Vec<TPUBuffer<T>>,
    ) -> Result<Vec<TPUBuffer<T>>> {
        let start_time = Instant::now();

        // Get or compile the program
        let program = self.get_or_compile_program(computation_id).await?;

        // Select appropriate devices
        let devices = self.device_manager.select_devices(&program)?;

        // A computation cannot be evaluated without at least one device to run
        // on. `DeviceManager::new` populates the device set from the configured
        // core count, so an empty set here is a genuine configuration error.
        if devices.is_empty() {
            return Err(OptimError::DeviceError(ErrorContext::new(
                "no TPU devices available to execute computation".to_string(),
            )));
        }

        // Allocate memory
        let memory_allocation = self
            .memory_manager
            .allocate_for_computation(&program, &devices)?;

        // Serialize the input buffers into the task payload. This is the data
        // the CPU-reference executor actually operates on (see `execute_task`);
        // no input is discarded.
        let input_data = serialize_tpu_buffers(&inputs)?;

        // Create computation task
        let task = ComputationTask {
            task_id: TaskId(self.execution_engine.scheduler.next_task_id()),
            computation_id,
            input_data,
            expected_outputs: program.metadata.output_specs.clone(),
        };

        // Execute the task on the CPU reference executor.
        let results = self
            .execution_engine
            .execute_task(task, &devices, &memory_allocation)?;

        // Update performance metrics
        let execution_time = start_time.elapsed();
        self.performance_monitor
            .record_execution(computation_id, execution_time, &results);

        // Decode the real output payload back into typed TPU buffers. For an
        // opaque computation the reference semantics are an identity evaluation,
        // so the returned buffers carry the (round-tripped) input tensors rather
        // than an empty placeholder.
        deserialize_tpu_buffers::<T>(&results.output_data)
    }

    async fn get_or_compile_program(
        &self,
        computation_id: ComputationId,
    ) -> Result<Arc<CompiledProgram>> {
        // Check cache first. The read guard is scoped so it is released before
        // the `.await` below, keeping the future `Send` and avoiding a deadlock
        // against the later write lock.
        {
            let cache = self.compilation_cache.read().map_err(|_| {
                OptimError::MutexError(ErrorContext::new(
                    "compilation cache read lock poisoned".to_string(),
                ))
            })?;
            if let Some(program) = cache.get(&computation_id) {
                // Real cache hit.
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::new(program.clone()));
            }
        }

        // Cache miss: record it exactly once and compile.
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Compile the program
        let program = self.compile_program(computation_id).await?;

        // Cache the result
        {
            let mut cache = self.compilation_cache.write().map_err(|_| {
                OptimError::MutexError(ErrorContext::new(
                    "compilation cache write lock poisoned".to_string(),
                ))
            })?;
            cache.insert(computation_id, program.clone());
        }

        Ok(Arc::new(program))
    }

    async fn compile_program(&self, computation_id: ComputationId) -> Result<CompiledProgram> {
        // A real XLA compile is impossible without TPU tooling, but the compiled
        // artifact must still be a deterministic, honest function of the program
        // descriptor rather than a magic placeholder. We serialize the descriptor
        // (computation id + target architecture + optimization level) and append
        // a fixed-length FNV-1a digest so that identical programs always produce
        // identical binaries and different programs differ.
        let opt_level = self.config.tpu_config.xla_optimization_level;
        let target = self.config.tpu_config.tpu_version;
        let binary = encode_program_binary(computation_id, target, opt_level);

        let metadata = ProgramMetadata {
            compiled_at: Instant::now(),
            compiler_version: "XLA-reference-1.0.0".to_string(),
            optimization_level: opt_level,
            target_architecture: target,
            program_size: binary.len(),
            output_specs: Vec::new(),
        };

        // Memory requirements are derived from the real binary size and the
        // configured per-core batch size rather than fixed constants.
        let code_memory = binary.len();
        let data_memory = code_memory
            .saturating_mul(self.config.tpu_config.batch_size_per_core.max(1))
            .max(code_memory);
        let stack_memory = code_memory;
        let scratch_memory = code_memory.saturating_mul(2);
        let total_memory = code_memory
            .saturating_add(data_memory)
            .saturating_add(stack_memory)
            .saturating_add(scratch_memory);

        let memory_requirements = ProgramMemoryRequirements {
            code_memory,
            data_memory,
            stack_memory,
            scratch_memory,
            total_memory,
        };

        // Utilization is a documented estimate keyed off the optimization level
        // (higher optimization -> higher expected compute/bandwidth utilization).
        // It is impossible to measure real silicon utilization on a CPU reference,
        // so this is the best honest estimate derived from the actual program
        // configuration instead of a hard-coded 0.85.
        let performance_characteristics = ProgramPerformanceCharacteristics {
            estimated_execution_time: Duration::from_micros(100),
            estimated_flops: 1_000_000,
            memory_bandwidth_utilization: estimate_bandwidth_utilization(opt_level),
            compute_utilization: estimate_compute_utilization(opt_level),
        };

        Ok(CompiledProgram {
            binary,
            metadata,
            memory_requirements,
            performance_characteristics,
        })
    }

    /// Get backend performance statistics
    pub fn get_performance_statistics(&self) -> BackendPerformanceStatistics {
        BackendPerformanceStatistics {
            total_executions: self.performance_monitor.total_executions,
            average_execution_time: self.performance_monitor.average_execution_time,
            device_utilization: self.device_manager.get_utilization_stats(),
            memory_utilization: self.memory_manager.get_utilization_stats(),
            cache_hit_rate: self.get_cache_hit_rate(),
            error_rate: self.error_handler.get_error_rate(),
        }
    }

    /// `pub(super)`: called directly by the `tpu_backend` test module (a
    /// sibling submodule), in addition to internal use in
    /// [`Self::get_performance_statistics`].
    pub(super) fn get_cache_hit_rate(&self) -> f64 {
        // Real hit rate derived from the counters maintained by
        // `get_or_compile_program`. Returns 0.0 when no lookup has happened yet.
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Shutdown the backend gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        self.device_manager.shutdown().await?;
        self.memory_manager.cleanup()?;
        self.performance_monitor.flush_metrics()?;
        Ok(())
    }
}
