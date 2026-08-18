//! # OptiRS TPU - TPU Coordination and Pod Management
//!
//! **Version:** 0.3.2
//! **Status:** Working CPU-reference implementation; no vendor TPU runtime
//!
//! `optirs-tpu` provides TPU-style coordination, pod management, and an XLA-shaped
//! compilation pipeline for OptiRS, built on
//! [SciRS2](https://github.com/cool-japan/scirs)'s abstractions.
//!
//! ⚠️ **Hardware note:** there is no Google TPU (or other XLA vendor) runtime linked
//! into this crate — that is proprietary and not distributable as pure Rust. Every
//! algorithm below (graph optimization, shape inference, checkpointing, ...) is a
//! real implementation that runs and is tested on the CPU reference executor;
//! [`tpu_backend`] and [`fault_tolerance`] document, function by function, exactly
//! where a real vendor runtime would be required and return an explicit
//! [`error::TpuError`]/[`error::OptimError`] there instead of a fabricated result.
//!
//! ## What is real today
//!
//! - **[`TPUOptimizer`]** wraps any [`optirs_core::Optimizer`] (also implementing
//!   that trait itself) and drives it through a compile → execute → profile
//!   pipeline; `tpu_step` performs the real update, not a stub.
//! - **XLA-shaped compiler pipeline** ([`xla`]): a genuine computation-graph
//!   builder with producer/consumer dependency tracking, dead-code elimination
//!   (with a fail-safe against deleting undeclared-output graphs), constant
//!   folding, common-subexpression elimination, kernel-fusion legality checks,
//!   a real (non-bump) memory allocator with free/coalescing, and shape
//!   inference for reshape/convolution/dot/broadcast.
//! - **[`tpu_backend::TPUBackend`]**: the run side -- device selection, memory
//!   pools, retry policy, profiling. It owns no compiler of its own:
//!   [`xla::XLACompiler`] is the single graph-to-binary path and this backend
//!   drives it, so a program's FLOP count, time estimate and memory footprint
//!   all come from the graph that was actually compiled. Register a graph with
//!   `register_computation` before executing it; an id with no registered graph
//!   is an error rather than a synthesized binary. Execution evaluates that
//!   graph through [`xla::execution::ReferenceExecutor`], and every device
//!   memory reservation it makes is recorded into the same profile the compile
//!   step opened.
//! - **[`coordination::PodCoordinator`]**: device/channel topology, barrier
//!   synchronization, load balancing, and fault detection over real in-process
//!   state (no `sleep`-and-report-success placeholders).
//! - **[`fault_tolerance`]**: checkpoints are serialized to disk with a SHA-256
//!   integrity hash and verified on restore; rollback and replication go through
//!   that same verified path.
//! - **[`synchronization`]**: barriers with a correctly-signaled condvar predicate,
//!   plus ring all-reduce/broadcast/reduce-scatter collectives.
//!
//! ## What is not implemented
//!
//! - Executing on real TPU silicon (needs a vendor runtime — see the hardware
//!   note above).
//! - Cross-device workload migration (`fault_tolerance::migrate_workload`
//!   returns `Err` rather than fabricate a live migration).
//! - Most of [`pod_coordination`]'s `communication`, `topology` and
//!   `resource_scheduling`/`load_balancing`/`gradient_aggregation`/
//!   `batch_coordination`/`performance` submodules are still data-only
//!   scaffolding: real types with derives, no methods, nothing wired to a
//!   caller yet. A few pieces of `pod_coordination` are real today —
//!   [`pod_coordination::coordination::coordinator::TPUPodCoordinator`]
//!   delegates to the real [`coordination::PodCoordinator`], and
//!   `synchronization::clocks::protocols::NtpSynchronizer` implements genuine
//!   RFC 5905 clock-offset estimation — consult each module's own doc
//!   comments for its individual status.
//!
//! ## Example
//!
//! ```rust,ignore
//! use optirs_core::optimizers::SGD;
//! use optirs_tpu::{TPUConfig, TPUOptimizer};
//!
//! let base_optimizer = SGD::new(0.01f32);
//! let mut tpu_opt = TPUOptimizer::new(base_optimizer, TPUConfig::default())?;
//!
//! // Runs the real compile/execute pipeline on the CPU reference backend.
//! let updated = tpu_opt.tpu_step(&params, &grads)?;
//! # Ok::<(), optirs_tpu::error::OptimError>(())
//! ```
//!
//! ## Architecture
//!
//! Built on SciRS2 abstractions:
//! - **Numeric**: `scirs2_core::ndarray`, `scirs2_core::numeric::Float`
//! - **Errors**: `scirs2_core::error::CoreError` (re-exported here as [`error::OptimError`])
//!
//! ## Contributing
//!
//! Match the existing standard: no fabricated success values or hardcoded
//! placeholder outputs. Where a capability genuinely requires hardware or a
//! vendor runtime this crate does not have, return a descriptive `Err` rather
//! than simulate one — see `FaultToleranceManager::migrate_workload` in
//! [`fault_tolerance`] for the pattern.

pub mod coordination;
pub mod error;
pub mod fault_tolerance;
pub mod monitoring;
pub mod pod_coordination;
pub mod synchronization;
pub mod tpu_backend;
pub mod xla;

// Re-export main types from mod.rs
mod main_types;
pub use main_types::*;

pub use coordination::PodCoordinator;
pub use tpu_backend::DeviceId;
