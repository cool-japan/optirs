//! Execution engine: the scheduler and CPU-reference task executor that
//! [`super::backend::TPUBackend`] drives to run a compiled program.

use std::fmt::Debug;
use std::time::{Duration, Instant};

use scirs2_core::error::ErrorContext;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};
use crate::xla::execution::{ReferenceExecutor, ValueMap};
use crate::xla::frontend::XLAComputation;

use super::serialization::{
    decode_ref_tensors, encode_ref_tensors, RefTensor, ENERGY_PER_BYTE_NANOJOULE,
};
use super::types::{ComputationTask, MemoryAllocation, TPUBackendConfig, TaskExecutionResult};
use super::DeviceId;

/// Execution engine for TPU computations
#[derive(Debug)]
pub struct ExecutionEngine<T: Float + Debug + Send + Sync + 'static> {
    /// Execution scheduler
    ///
    /// `pub(super)`: [`super::backend::TPUBackend::execute_computation`] and the
    /// `tpu_backend` test module both mint task ids via `.scheduler.next_task_id()`
    /// from a sibling submodule, so this needs module-subtree visibility rather
    /// than file-private access.
    pub(super) scheduler: ExecutionScheduler<T>,

    /// Wall-clock budget for a single task, from
    /// [`TPUBackendConfig::execution_timeout_ms`]. `Duration::ZERO` disables the
    /// check.
    execution_timeout: Duration,
}

/// Execution scheduler
#[derive(Debug)]
pub struct ExecutionScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Monotonic counter backing `next_task_id`
    next_task_id_counter: u64,

    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ExecutionScheduler<T> {
    pub fn next_task_id(&mut self) -> u64 {
        // Return-then-increment so ids are unique and strictly monotonic.
        let id = self.next_task_id_counter;
        self.next_task_id_counter = self.next_task_id_counter.wrapping_add(1);
        id
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ExecutionEngine<T> {
    pub fn new(config: &TPUBackendConfig) -> Result<Self> {
        Ok(Self {
            scheduler: ExecutionScheduler {
                next_task_id_counter: 0,
                _phantom: std::marker::PhantomData,
            },
            execution_timeout: Duration::from_millis(config.execution_timeout_ms),
        })
    }

    /// The configured per-task wall-clock budget.
    pub fn execution_timeout(&self) -> Duration {
        self.execution_timeout
    }

    /// Evaluate one task's `computation` against its serialized arguments.
    ///
    /// `computation` is the graph the caller registered with
    /// [`super::backend::TPUBackend::register_computation`]; the arguments are
    /// bound to the parameters that graph declares and every operation is then
    /// evaluated by [`ReferenceExecutor`]. This used to be an identity
    /// evaluation over the input tensors, because a bare `ComputationId`
    /// reached this far with no operation list attached -- it now runs the real
    /// program.
    ///
    /// # Devices are not consulted
    ///
    /// `devices` is deliberately unused: evaluation happens on the CPU, so the
    /// selected devices affect admission control, placement and accounting --
    /// not the arithmetic. Sharding a computation across the device set would
    /// mean partitioning the graph and reducing across the partitions, which
    /// this reference executor does not do and does not pretend to. The
    /// parameter is kept because the memory reserved on those devices is what
    /// `memory_allocation` describes, and a real backend would need it here.
    pub fn execute_task(
        &self,
        task: ComputationTask,
        computation: &XLAComputation<T>,
        _devices: &[DeviceId],
        memory_allocation: &MemoryAllocation,
    ) -> Result<TaskExecutionResult>
    where
        T: Default + Clone,
    {
        let start = Instant::now();

        // CPU-reference execution of the real graph: decode the argument
        // tensors, bind them to the declared parameters, evaluate, and
        // re-encode the declared outputs. Fully defined without TPU silicon and
        // deterministic.
        let input_tensors = decode_ref_tensors(&task.input_data)?;
        if input_tensors.len() != computation.inputs.len() {
            return Err(OptimError::InvalidInput(ErrorContext::new(format!(
                "computation '{}' declares {} parameter(s) but {} argument tensor(s) were supplied",
                computation.metadata.name,
                computation.inputs.len(),
                input_tensors.len()
            ))));
        }

        let mut values = ValueMap::with_capacity(input_tensors.len());
        for (spec, tensor) in computation.inputs.iter().zip(&input_tensors) {
            let array = ArrayD::from_shape_vec(IxDyn(&tensor.shape), tensor.data.clone()).map_err(
                |error| {
                    OptimError::InvalidInput(ErrorContext::new(format!(
                        "argument {} of computation '{}' declares shape {:?}, which does not \
                         describe its {} element(s): {error}",
                        spec.index,
                        computation.metadata.name,
                        tensor.shape,
                        tensor.data.len()
                    )))
                },
            )?;
            values.insert(spec.operand, array);
        }

        let outputs = ReferenceExecutor::new().execute(computation, values)?;
        let output_tensors: Vec<RefTensor> = outputs
            .iter()
            .map(|array| RefTensor {
                shape: array.shape().to_vec(),
                data: array.iter().copied().collect(),
            })
            .collect();
        let output_data = encode_ref_tensors(&output_tensors);

        // Enforce the configured budget. The reference executor is synchronous
        // and cannot be pre-empted mid-evaluation, so the budget is checked once
        // the work has finished: an over-budget task is reported as a real
        // timeout instead of quietly returning a result the caller has already
        // stopped waiting for. `execution_timeout_ms == 0` means "no budget".
        let execution_time = start.elapsed();
        if !self.execution_timeout.is_zero() && execution_time > self.execution_timeout {
            // `TimeoutError`, not `ComputationError`: this is the class
            // `TPUErrorHandler`'s recovery policy marks retryable, so the
            // backend's retry path has a real producer.
            return Err(OptimError::TimeoutError(ErrorContext::new(format!(
                "task {} exceeded the configured execution budget ({} ms) after {} ms",
                task.task_id.0,
                self.execution_timeout.as_millis(),
                execution_time.as_millis()
            ))));
        }

        // Real, derived accounting rather than magic constants.
        let bytes_touched = task.input_data.len() + output_data.len();
        let memory_used = bytes_touched + memory_allocation.total_allocated;
        // Deterministic energy estimate: a fixed nanojoule cost per byte moved
        // through the reference executor.
        let energy_consumed = bytes_touched as f64 * ENERGY_PER_BYTE_NANOJOULE;

        Ok(TaskExecutionResult {
            task_id: task.task_id,
            execution_time,
            memory_used,
            energy_consumed,
            output_data,
        })
    }
}
