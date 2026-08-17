//! [`TPUBackend`]: the top-level compile -> execute -> profile orchestrator
//! that owns every other component in this module tree.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

use scirs2_core::error::ErrorContext;
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};
use crate::xla::backend::{FragmentationInfo, ProfilingIntegration};
use crate::xla::frontend::XLAComputation;
use crate::xla::{
    CacheStatistics, CompiledComputation as XlaCompiledComputation, XLACompiler, XLACompilerConfig,
};

use super::buffer::TPUBuffer;
use super::device_manager::DeviceManager;
use super::execution::ExecutionEngine;
use super::memory::TPUMemoryManager;
use super::profiling::{PerformanceMonitor, TPUErrorHandler};
use super::serialization::{deserialize_tpu_buffers, serialize_tpu_buffers};
use super::types::{
    BackendPerformanceStatistics, CompiledProgram, ComputationId, ComputationTask,
    ExecutionUtilization, ProgramMemoryRequirements, ProgramMetadata,
    ProgramPerformanceCharacteristics, TPUBackendConfig, TaskExecutionResult, TaskId,
};
use super::DeviceId;

/// TPU Backend Manager
///
/// # Relationship to the XLA compiler
///
/// This backend is the **run** side: devices, memory pools, scheduling, retry
/// policy and profiling. It deliberately contains no compiler of its own --
/// [`crate::xla::XLACompiler`] (which in turn drives
/// [`crate::xla::backend::XLABackend`] for code generation and runtime
/// integration) is the single graph-to-binary path, and this type drives it.
///
/// Consequently a computation must be handed to [`Self::register_computation`]
/// before [`Self::execute_computation`] names it. A bare [`ComputationId`] is
/// just a key; this backend will not invent a program for an id it has never
/// been given the graph for.
pub struct TPUBackend<T: Float + Debug + Send + Sync + 'static> {
    /// Backend configuration
    config: TPUBackendConfig,

    /// Device manager
    device_manager: DeviceManager,

    /// Execution engine
    execution_engine: ExecutionEngine<T>,

    /// Memory manager
    memory_manager: TPUMemoryManager<T>,

    /// Error handler
    error_handler: TPUErrorHandler,

    /// Performance monitor
    performance_monitor: PerformanceMonitor,

    /// The one and only compilation path this backend has.
    ///
    /// It owns the compilation cache too, so there is a single answer to "was
    /// this program served from cache" (see [`Self::get_cache_hit_rate`])
    /// rather than a second `ComputationId -> binary` map maintained here.
    xla_compiler: XLACompiler<T>,

    /// Graphs registered for execution, keyed by their own id.
    ///
    /// Held behind [`Arc`] so a run can hold the graph while the backend is
    /// mutably borrowed for allocation and scheduling, without cloning the
    /// whole operation list on every execution.
    programs: HashMap<ComputationId, Arc<XLAComputation<T>>>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::iter::Sum> TPUBackend<T> {
    /// Create a new TPU backend
    pub fn new(config: TPUBackendConfig) -> Result<Self> {
        let device_manager = DeviceManager::new(&config)?;
        let execution_engine = ExecutionEngine::new(&config)?;
        let memory_manager = TPUMemoryManager::new(&config)?;
        let error_handler = TPUErrorHandler::new(&config);
        let performance_monitor = PerformanceMonitor::new(&config);
        let xla_compiler = XLACompiler::new(XLACompilerConfig {
            target_tpu: config.tpu_config.clone(),
            optimization_level: config.tpu_config.xla_optimization_level,
            profile_compilation: config.enable_performance_monitoring,
            ..XLACompilerConfig::default()
        })?;

        Ok(Self {
            config,
            device_manager,
            execution_engine,
            memory_manager,
            error_handler,
            performance_monitor,
            xla_compiler,
            programs: HashMap::new(),
        })
    }

    /// Register an XLA computation with this backend and return its id.
    ///
    /// Registering the same id again replaces the graph; the compiler's cache
    /// is keyed by graph *contents* (see [`XLACompiler`]), so a replaced graph
    /// is recompiled rather than served from the previous one's entry.
    pub fn register_computation(&mut self, computation: XLAComputation<T>) -> ComputationId {
        let id = computation.id;
        self.programs.insert(id, Arc::new(computation));
        id
    }

    /// The graph registered under `computation_id`, if any.
    pub fn registered_computation(
        &self,
        computation_id: ComputationId,
    ) -> Option<&XLAComputation<T>> {
        self.programs.get(&computation_id).map(Arc::as_ref)
    }

    /// Execute a computation on TPU
    pub async fn execute_computation(
        &mut self,
        computation_id: ComputationId,
        inputs: Vec<TPUBuffer<T>>,
    ) -> Result<Vec<TPUBuffer<T>>> {
        let start_time = Instant::now();

        // The graph is the program. An id with no registered graph is a caller
        // error, not an invitation to synthesize a binary.
        let computation = self.programs.get(&computation_id).cloned().ok_or_else(|| {
            OptimError::InvalidInput(ErrorContext::new(format!(
                "computation {} was never registered with this backend; call \
                 register_computation before executing it",
                computation_id.0
            )))
        })?;

        // Compile (or hit the compiler's cache) through the single XLA path.
        let program = self.compile_registered(&computation)?;

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

        // Record where this computation was placed, so a later execution of the
        // same id can be recognised as already assigned.
        self.device_manager
            .assign_devices(computation_id, devices.clone());

        // Serialize the input buffers into the task payload. This is the data
        // the CPU-reference executor binds to the graph's parameters (see
        // `execute_task`); no input is discarded.
        let input_data = serialize_tpu_buffers(&inputs)?;

        // Attempt the reserve-execute-release cycle under the error handler's
        // recovery policy: a retryable failure (timeout, resource contention)
        // is retried up to the configured attempt budget, while anything the
        // policy does not classify as retryable is surfaced immediately rather
        // than looped over.
        self.error_handler.record_operation();
        let max_attempts = self.error_handler.max_attempts();
        let mut attempt = 0usize;
        let (results, utilization) = loop {
            match self
                .run_one_attempt(&computation, &program, &devices, &input_data)
                .await
            {
                Ok(outcome) => {
                    if attempt > 0 {
                        self.error_handler.record_recovery_success();
                    }
                    break outcome;
                }
                Err(error) => {
                    let retry = self.error_handler.record_error(&error, attempt);
                    if !retry {
                        return Err(error);
                    }
                    attempt += 1;
                    debug_assert!(attempt < max_attempts);
                }
            }
        };

        // Update performance metrics
        let execution_time = start_time.elapsed();
        self.performance_monitor.record_execution(
            computation_id,
            execution_time,
            &results,
            utilization,
        );

        // Decode the real output payload back into typed TPU buffers: one per
        // output the graph declares, carrying the values the reference executor
        // actually computed.
        deserialize_tpu_buffers::<T>(&results.output_data)
    }

    /// One reserve -> execute -> release cycle.
    ///
    /// The memory reserved up front is always returned to the pools, including
    /// on the failure path, so a retry starts from the same capacity the first
    /// attempt saw instead of a pool that the failed attempt leaked into.
    async fn run_one_attempt(
        &mut self,
        computation: &XLAComputation<T>,
        program: &CompiledProgram,
        devices: &[DeviceId],
        input_data: &[u8],
    ) -> Result<(TaskExecutionResult, ExecutionUtilization)> {
        // Reserve the program's real memory footprint on the selected devices.
        // This is admission control: an oversized program fails here rather than
        // being handed a zero-byte allocation.
        let memory_allocation = self
            .memory_manager
            .allocate_for_computation(program, devices)?;

        let task = ComputationTask {
            task_id: TaskId(self.execution_engine.scheduler.next_task_id()),
            computation_id: computation.id,
            input_data: input_data.to_vec(),
            expected_outputs: program.metadata.output_specs.clone(),
        };

        // Report the reservation into the profile the compile step opened, so
        // an exported memory profile carries the allocator's real events
        // rather than an honestly-empty file.
        let session = format!("computation_{}", computation.id.0);
        for reservation in &memory_allocation.reservations {
            self.xla_compiler.profiling_mut().record_memory_allocation(
                &session,
                reservation.address,
                reservation.size,
                Some(format!("device_{}", reservation.device.0)),
            );
        }

        let execution =
            self.execution_engine
                .execute_task(task, computation, devices, &memory_allocation);

        // Report the observed per-device load (the fraction of each device's
        // capacity this computation held) before releasing it, so the load
        // balancer's next placement decision sees real occupancy.
        let mut device_share_total = 0.0f64;
        let mut device_share_count = 0usize;
        for (device, bytes) in &memory_allocation.device_allocations {
            if let Some(capacity) = self
                .device_manager
                .device_capacity(*device)
                .filter(|capacity| *capacity > 0)
            {
                let share = *bytes as f64 / capacity as f64;
                device_share_total += share;
                device_share_count += 1;
                self.device_manager.record_device_load(*device, share);
            }
        }
        let utilization = ExecutionUtilization {
            device: if device_share_count == 0 {
                0.0
            } else {
                device_share_total / device_share_count as f64
            },
            memory: self.memory_manager.get_utilization_stats(),
        };

        // Release before propagating any error: the reservation must not outlive
        // the attempt that made it.
        self.memory_manager.release_allocation(&memory_allocation);

        // Mirror the release into the profile, together with the free-list
        // shape the allocator reports once the blocks are back and coalesced.
        let (largest_free_block, free_block_count) = self.memory_manager.free_block_summary();
        let fragmentation = FragmentationInfo {
            external_fragmentation: self.memory_manager.usage_statistics().fragmentation_ratio,
            // The pools hand out exactly the bytes requested (only the buddy
            // strategy rounds, and it rounds the *request*), so no allocated
            // block carries unused slack: internal fragmentation is genuinely
            // zero here rather than unmeasured.
            internal_fragmentation: 0.0,
            largest_free_block,
            free_block_count,
        };
        let released: Vec<usize> = memory_allocation
            .reservations
            .iter()
            .map(|reservation| reservation.address)
            .collect();
        self.xla_compiler.profiling_mut().record_memory_release(
            &session,
            &released,
            fragmentation,
            Some("release".to_string()),
        );

        Ok((execution?, utilization))
    }

    /// Compile a registered computation without executing it.
    ///
    /// Useful for inspecting a program's real footprint and cost estimate
    /// ahead of a run; [`Self::execute_computation`] takes exactly this path.
    pub fn compile(&mut self, computation_id: ComputationId) -> Result<CompiledProgram> {
        let computation = self.programs.get(&computation_id).cloned().ok_or_else(|| {
            OptimError::InvalidInput(ErrorContext::new(format!(
                "computation {} was never registered with this backend; call \
                 register_computation before compiling it",
                computation_id.0
            )))
        })?;
        self.compile_registered(&computation)
    }

    /// Compile `computation` through the XLA pipeline and project the result
    /// onto this module's [`CompiledProgram`] view.
    ///
    /// Every number below comes from the compile that just ran: the binary is
    /// the code generator's output, the FLOP count and time estimate come from
    /// [`crate::xla::optimization::PerformanceAnalyzer`] walking the real
    /// operation list, and the data footprint is the memory planner's placed
    /// total. This is what replaced the descriptor-encoding stand-in that used
    /// to run here, which could only ever report zero FLOPs because it never
    /// saw an operation list.
    fn compile_registered(
        &mut self,
        computation: &Arc<XLAComputation<T>>,
    ) -> Result<CompiledProgram> {
        let output_specs = computation
            .outputs
            .iter()
            .map(|spec| {
                format!(
                    "output_{}:{:?}:{:?}",
                    spec.index, spec.dtype, spec.shape.dimensions
                )
            })
            .collect();

        let compiled = self.xla_compiler.compile((**computation).clone())?;
        Ok(self.project_compiled(compiled, output_specs))
    }

    /// Map the XLA compiler's result onto [`CompiledProgram`].
    fn project_compiled(
        &self,
        compiled: XlaCompiledComputation,
        output_specs: Vec<String>,
    ) -> CompiledProgram {
        let info = &compiled.metadata.performance_info;

        let performance_characteristics = ProgramPerformanceCharacteristics {
            estimated_execution_time: Duration::from_micros(info.estimated_execution_time),
            estimated_flops: info.flop_count,
            memory_bandwidth_utilization: info.memory_bandwidth_util,
            compute_utilization: info.compute_utilization,
        };

        // `code_memory` is the real generated binary; `data_memory` is the
        // memory planner's total placed buffer bytes.
        //
        // `stack_memory` and `scratch_memory` stay at zero and that is the
        // honest value, not a placeholder: the XLA memory planner places every
        // operand buffer explicitly and models neither a call stack nor a
        // separate scratch arena, so there is no measured quantity to report.
        // Splitting `data_memory` into persistent and transient halves would
        // require the plan's per-buffer lifetimes, which `CompiledComputation`
        // does not carry today.
        let code_memory = compiled.binary.len();
        let data_memory = info.memory_usage;
        let memory_requirements = ProgramMemoryRequirements {
            code_memory,
            data_memory,
            stack_memory: 0,
            scratch_memory: 0,
            total_memory: code_memory.saturating_add(data_memory),
        };

        let metadata = ProgramMetadata {
            compiled_at: Instant::now(),
            compiler_version: compiled.metadata.compiler_version,
            optimization_level: self.config.tpu_config.xla_optimization_level,
            target_architecture: self.config.tpu_config.tpu_version,
            program_size: code_memory,
            output_specs,
        };

        CompiledProgram {
            binary: compiled.binary,
            metadata,
            memory_requirements,
            performance_characteristics,
        }
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

    /// Fraction of compilation lookups served from cache.
    ///
    /// Read straight off the compiler's own cache statistics: this backend
    /// keeps no second cache, so "hit" means "the compiler returned a cached
    /// binary" and cannot drift from what actually happened.
    ///
    /// `pub(super)`: called directly by the `tpu_backend` test module (a
    /// sibling submodule), in addition to internal use in
    /// [`Self::get_performance_statistics`].
    pub(super) fn get_cache_hit_rate(&self) -> f64 {
        self.compilation_cache_statistics().hit_rate
    }

    /// Hit/miss/eviction counters for the compilation cache backing this
    /// backend -- that is, the XLA compiler's, since there is only one.
    pub fn compilation_cache_statistics(&self) -> CacheStatistics {
        self.xla_compiler.cache_statistics()
    }

    /// The profile this backend records into.
    ///
    /// Compilation events are recorded by the XLA backend; the device-memory
    /// reservations made while executing are recorded here (see
    /// [`Self::execute_computation`]). Both land in the same profile, which is
    /// what [`ProfilingIntegration::export_data`] writes out.
    pub fn profiling(&self) -> &ProfilingIntegration<T> {
        self.xla_compiler.profiling()
    }

    /// Mutable access to that profile, for callers that want to export or
    /// reset it.
    pub fn profiling_mut(&mut self) -> &mut ProfilingIntegration<T> {
        self.xla_compiler.profiling_mut()
    }

    /// Shutdown the backend gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        self.device_manager.shutdown().await?;
        self.memory_manager.cleanup()?;
        self.performance_monitor.flush_metrics()?;
        Ok(())
    }
}
