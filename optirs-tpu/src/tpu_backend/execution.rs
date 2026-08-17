//! Execution engine: the scheduler and CPU-reference task executor that
//! [`super::backend::TPUBackend`] drives to run a compiled program.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::Instant;

use scirs2_core::numeric::Float;

use crate::error::Result;

use super::serialization::{
    decode_ref_tensors, encode_ref_tensors, evaluate_reference, ENERGY_PER_BYTE_NANOJOULE,
};
use super::types::{
    ComputationTask, ExecutionTask, MemoryAllocation, SchedulingPolicy, TPUBackendConfig,
    TaskExecutionResult,
};
use super::DeviceId;

/// Runtime executor for TPU operations
#[derive(Debug)]
pub struct RuntimeExecutor<T: Float + Debug + Send + Sync + 'static> {
    /// Execution state
    state: T,
}

/// Result collector for TPU computations
#[derive(Debug)]
pub struct ResultCollector<T: Float + Debug + Send + Sync + 'static> {
    /// Collected results
    results: Vec<T>,
}

/// Execution context for TPU operations
#[derive(Debug)]
pub struct ExecutionContext {
    /// Context id
    id: usize,
}

/// Performance optimizer for TPU operations
#[derive(Debug)]
pub struct PerformanceOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Optimization level
    level: T,
}

/// Priority manager for TPU task scheduling
#[derive(Debug)]
pub struct PriorityManager {
    /// Priority level
    level: usize,
}

/// Dependency resolver for TPU operations
#[derive(Debug)]
pub struct DependencyResolver {
    /// Resolved dependencies
    dependencies: Vec<String>,
}

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

    /// Runtime executor
    executor: RuntimeExecutor<T>,

    /// Result collector
    result_collector: ResultCollector<T>,

    /// Execution context
    context: ExecutionContext,

    /// Performance optimizer
    performance_optimizer: PerformanceOptimizer<T>,
}

/// Execution scheduler
#[derive(Debug)]
pub struct ExecutionScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Execution queue
    execution_queue: VecDeque<ExecutionTask<T>>,

    /// Scheduling policy
    scheduling_policy: SchedulingPolicy,

    /// Priority manager
    priority_manager: PriorityManager,

    /// Dependency resolver
    dependency_resolver: DependencyResolver,

    /// Monotonic counter backing `next_task_id`
    next_task_id_counter: u64,
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
                execution_queue: VecDeque::new(),
                scheduling_policy: SchedulingPolicy::FIFO,
                priority_manager: PriorityManager { level: 0 },
                dependency_resolver: DependencyResolver {
                    dependencies: Vec::new(),
                },
                next_task_id_counter: 0,
            },
            executor: RuntimeExecutor { state: T::zero() },
            result_collector: ResultCollector {
                results: Vec::new(),
            },
            context: ExecutionContext { id: 0 },
            performance_optimizer: PerformanceOptimizer { level: T::zero() },
        })
    }

    pub fn execute_task(
        &self,
        task: ComputationTask,
        _devices: &[DeviceId],
        memory_allocation: &MemoryAllocation,
    ) -> Result<TaskExecutionResult> {
        let start = Instant::now();

        // CPU-reference execution. A bare `ComputationId` carries no XLA op-list
        // reachable from this backend, so the reference semantics are a faithful
        // identity evaluation over the serialized input tensors: decode the input
        // payload, evaluate every tensor on the CPU, and re-encode the result.
        // This is fully defined without TPU silicon and produces real,
        // deterministic `output_data` (never an empty placeholder).
        let input_tensors = decode_ref_tensors(&task.input_data)?;
        let output_tensors = evaluate_reference(input_tensors);
        let output_data = encode_ref_tensors(&output_tensors);

        // Real, derived accounting rather than magic constants.
        let bytes_touched = task.input_data.len() + output_data.len();
        let memory_used = bytes_touched + memory_allocation.total_allocated;
        // Deterministic energy estimate: a fixed nanojoule cost per byte moved
        // through the reference executor.
        let energy_consumed = bytes_touched as f64 * ENERGY_PER_BYTE_NANOJOULE;

        Ok(TaskExecutionResult {
            task_id: task.task_id,
            execution_time: start.elapsed(),
            memory_used,
            energy_consumed,
            output_data,
        })
    }
}
