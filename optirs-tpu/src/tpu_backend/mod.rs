// TPU Backend Implementation
//
// This module provides the core TPU backend implementation for executing
// optimized computations on Google Cloud TPUs and compatible hardware.
//
// Split into focused submodules (originally a single ~2200-line file):
// - `types`: data-only descriptors, task/program/statistics structs.
// - `device_manager`: TPU device enumeration and health/utilization tracking.
// - `buffer`: the typed `TPUBuffer` data container.
// - `execution`: the execution engine/scheduler and CPU-reference task executor.
// - `memory`: per-device memory pools and garbage-collection bookkeeping.
// - `profiling`: runtime profiler, error handler, and performance monitor.
// - `backend`: `TPUBackend`, the top-level compile/execute/profile orchestrator.
// - `serialization`: the reference tensor codec and program-binary codec.
// - `device_defaults`: derived (non-hardcoded) per-`TPUVersion` defaults.
//
// A handful of fields/functions that were file-private in the original single
// file are `pub(super)` here solely because they are now reached from a
// sibling submodule (mostly the consolidated `tests` module); see each such
// item's doc comment. None of this widens the crate's public API: every type
// that was `pub` at the top of the original file is re-exported below at the
// same `tpu_backend::` path.

mod backend;
mod buffer;
mod device_defaults;
mod device_manager;
mod execution;
mod memory;
mod profiling;
mod serialization;
#[cfg(test)]
mod tests;
mod types;

pub use backend::TPUBackend;
pub use buffer::TPUBuffer;
pub use device_manager::DeviceManager;
pub use execution::{ExecutionEngine, ExecutionScheduler};
pub use memory::{MemoryGarbageCollector, MemoryPool, TPUMemoryManager};
pub use profiling::{PerformanceMonitor, TPUErrorHandler};
pub use types::*;
