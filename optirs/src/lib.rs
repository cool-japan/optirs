//! # OptiRS - Advanced ML Optimization Built on SciRS2
//!
//! **Version:** 0.3.2
//!
//! [![Crates.io](https://img.shields.io/crates/v/optirs.svg)](https://crates.io/crates/optirs)
//! [![Documentation](https://docs.rs/optirs/badge.svg)](https://docs.rs/optirs)
//! [![License](https://img.shields.io/crates/l/optirs.svg)](https://github.com/cool-japan/optirs)
//!
//! OptiRS is a comprehensive optimization library for machine learning, built exclusively on
//! the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem. It provides
//! state-of-the-art optimization algorithms with advanced hardware acceleration.
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.6.5 - Required foundation
//!
//! ## Sub-Crate Status (v0.3.2)
//!
//! - ✅ `optirs-core` - Stable, production-ready (optimizers, schedulers, regularizers,
//!   SIMD and parallel paths, metrics)
//! - ✅ `optirs-bench` - Available (benchmarking, profiling, regression detection)
//! - 🚧 `optirs-gpu` - Real GPU compute path (Metal backend live end-to-end; WebGPU
//!   kernels implemented but blocked on an upstream `scirs2-core` adapter-probe bug;
//!   OpenCL is context-only; CUDA/ROCm have no backend) plus a fully-tested CPU
//!   library of GPU-aware algorithms
//! - 🔬 `optirs-learned` - Research-grade learned optimizers and meta-learning (real,
//!   tested implementations; APIs may still change)
//! - 🔬 `optirs-nas` - Research-grade neural architecture search (real, tested
//!   implementations; APIs may still change)
//! - 📝 `optirs-tpu` - Working CPU-reference implementation of TPU-style coordination
//!   and an XLA-shaped compiler; no vendor TPU runtime is linked (proprietary hardware)
//!
//! ## Quick Start
//!
//! Add OptiRS to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! optirs-core = "0.3.2"
//! ```
//!
//! Basic usage:
//!
//! ```rust
//! use optirs::prelude::*;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! // Create Adam optimizer
//! let mut optimizer = Adam::new(0.001);
//!
//! // Prepare parameters and gradients
//! let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let gradients = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.08]);
//!
//! // Perform optimization step
//! let updated_params = optimizer.step(&params, &gradients)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! ### Core Optimizers (`optirs-core`)
//!
//! - **First-Order**: SGD, Adam, AdamW, AdaDelta, AdaBound, Adagrad, RMSprop, LAMB,
//!   LARS, Lion, RAdam, Ranger, SAM
//! - **SIMD-Accelerated**: `SimdSGD`
//! - **Sparse / grouped**: `SparseAdam`, `GroupedAdam`
//! - **Meta-learning**: `MAML`, `MetaSGD`, `ReptileOptimizer`
//! - **Wrapper**: `Lookahead`
//! - **Second-Order** (`optirs_core::second_order`): L-BFGS, Newton, Newton-CG, K-FAC
//! - **Distributed** (`optirs_core::distributed`): FedProx
//!
//! #### Performance Features
//!
//! - **SIMD** - vectorized optimizer steps through `scirs2_core::simd_ops`
//! - **Parallel** - parameter groups distributed across cores through
//!   `scirs2_core::parallel_ops`
//! - **Memory-Efficient** - gradient accumulation and chunked processing
//! - **GPU** - see the `optirs-gpu` status below for which backends are real
//! - **Production Metrics** - per-step monitoring with gradient and parameter statistics
//!
//! Speedups are workload- and hardware-dependent; measure them with the Criterion
//! benchmarks in `optirs-core/benches/` rather than assuming a headline figure.
//!
//! ### GPU Acceleration (`optirs-gpu`)
//!
//! ```toml
//! [dependencies]
//! optirs-gpu = { version = "0.3.2", features = ["metal"] }
//! ```
//!
//! - **Metal**: real compute shaders (MSL pipelines, buffers, dispatch, readback) run
//!   Adam, AdamW, SGD, RMSprop, Adagrad and LAMB end-to-end today
//! - **WebGPU**: WGSL kernels are implemented, but blocked on an upstream `scirs2-core`
//!   adapter-probe bug; **OpenCL**: context creation only, no kernels shipped yet;
//!   **CUDA / ROCm**: no backend (`scirs2-core` 0.6.x dropped its CUDA backend)
//! - **Tensor Cores**: real mixed-precision tiled GEMM on the wgpu path
//! - **Memory Management**: CPU-side GPU memory pool models (arena/buddy/slab allocators)
//! - **Multi-GPU**: single-device reduction kernels; true cross-device collectives
//!   return an explicit `UnsupportedOperation` error rather than a fabricated result
//!
//! ### TPU Coordination (`optirs-tpu`)
//!
//! ```toml
//! [dependencies]
//! optirs-tpu = "0.3.2"
//! ```
//!
//! A working CPU-reference implementation - no vendor TPU runtime is linked (that is
//! proprietary and not distributable as pure Rust); every path below runs and is tested
//! on the CPU executor, and returns an explicit error where real TPU silicon would be
//! required instead of a fabricated result.
//!
//! - **Pod Management**: device/channel topology, barrier sync, load balancing, fault detection
//! - **XLA-shaped Compiler**: graph builder, dead-code elimination, constant folding,
//!   common-subexpression elimination, kernel-fusion legality checks, a real allocator,
//!   shape inference
//! - **Fault Tolerance**: checkpoints serialized with a SHA-256 integrity hash, verified on restore
//! - **Collectives**: ring all-reduce / broadcast / reduce-scatter
//!
//! ### Learned Optimizers (`optirs-learned`) [Research-Grade]
//!
//! - **Transformer-based**: self-attention optimizer with a real backward pass
//! - **LSTM**: recurrent optimizer networks trained by truncated BPTT, with seeded,
//!   reproducible initialization
//! - **Meta-Learning**: MAML, Reptile, Meta-SGD and online meta-learning across tasks
//! - **Few-Shot**: prototypical networks, fast adaptation, episodic memory
//! - **Continual Learning**: elastic weight consolidation, progressive networks
//!
//! ### Neural Architecture Search (`optirs-nas`) [Research-Grade]
//!
//! - **Search Strategies**: random, evolutionary, reinforcement-learning, Bayesian and
//!   differentiable (DARTS, PC-DARTS, RobustDARTS)
//! - **Multi-Objective**: NSGA-II and MOEA/D with exact hypervolume
//! - **Hyperparameter Search**: grid, TPE and a kernel-regression surrogate
//! - **Progressive**: search with a gradually increasing complexity budget
//! - **Hardware-Aware**: latency, memory and energy cost modelling
//!
//! ## Module Organization
//!
//! OptiRS is organized into feature-gated modules:
//!
//! - [`core`] - Core optimizers and utilities (always available)
//! - `gpu` - GPU acceleration (feature: `gpu`)
//! - `tpu` - TPU coordination (feature: `tpu`)
//! - `learned` - Learned optimizers (feature: `learned`)
//! - `nas` - Neural architecture search (feature: `nas`)
//! - `bench` - Benchmarking tools (feature: `bench`)
//!
//! ## Examples
//!
//! ### SIMD Acceleration
//!
//! ```rust
//! use optirs::prelude::*;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! // Large parameter array (SIMD shines with 10k+ elements)
//! let params = Array1::from_elem(100_000, 1.0f32);
//! let grads = Array1::from_elem(100_000, 0.001f32);
//!
//! let mut optimizer = SimdSGD::new(0.01f32);
//! let updated = optimizer.step(&params, &grads)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Parallel Processing
//!
//! ```rust
//! use optirs::prelude::*;
//! use optirs::core::parallel_optimizer::parallel_step_array1;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let params_list = vec![
//!     Array1::from_elem(10_000, 1.0),
//!     Array1::from_elem(20_000, 1.0),
//! ];
//! let grads_list = vec![
//!     Array1::from_elem(10_000, 0.01),
//!     Array1::from_elem(20_000, 0.01),
//! ];
//!
//! let mut optimizer = Adam::new(0.001);
//! let results = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Production Monitoring
//!
//! ```rust
//! use optirs::core::optimizer_metrics::MetricsCollector;
//! use optirs::prelude::*;
//! use scirs2_core::ndarray::Array1;
//! use std::time::Instant;
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let mut collector = MetricsCollector::new();
//! collector.register_optimizer("adam");
//!
//! let mut optimizer = Adam::new(0.001);
//! let params = Array1::from_elem(1000, 1.0);
//! let grads = Array1::from_elem(1000, 0.01);
//!
//! let params_before = params.clone();
//! let start = Instant::now();
//! let params = optimizer.step(&params, &grads)?;
//! let duration = start.elapsed();
//!
//! collector.update(
//!     "adam",
//!     duration,
//!     0.001,
//!     &grads.view(),
//!     &params_before.view(),
//!     &params.view(),
//! )?;
//!
//! println!("{}", collector.summary_report());
//! # Ok(())
//! # }
//! ```
//!
//! ## SciRS2 Integration
//!
//! OptiRS is built **exclusively** on SciRS2:
//!
//! - ✅ **Arrays**: `scirs2_core::ndarray` (NOT direct ndarray)
//! - ✅ **Random**: `scirs2_core::random` (NOT direct rand)
//! - ✅ **SIMD**: `scirs2_core::simd_ops`
//! - ✅ **Parallel**: `scirs2_core::parallel_ops`
//! - ✅ **GPU**: `scirs2_core::gpu`
//! - ✅ **Metrics**: `scirs2_core::metrics`
//!
//! This ensures type safety, performance, and consistency across the ecosystem.
//!
//! ## Project health
//!
//! Measured on the 0.3.2 release candidate with `--all-features`:
//!
//! - more than 4,200 unit and integration tests passing workspace-wide, plus the
//!   doc tests
//! - `cargo check` and `cargo clippy --workspace --all-targets` both at zero warnings,
//!   with no blanket `allow` attributes anywhere
//! - `cargo deny check bans` passes
//!
//! Reproduce with `cargo nextest run --workspace --all-features` and
//! `cargo clippy --workspace --all-features --all-targets`.
//!
//! ## Documentation
//!
//! - **API Documentation**: [docs.rs/optirs](https://docs.rs/optirs)
//! - **User Guide**: `USAGE_GUIDE.md` in the repository
//! - **Examples**: the `examples/` directory of this crate
//! - **Release notes**: `CHANGELOG.md` in the repository
//!
//! ## Contributing
//!
//! Contributions are welcome! Ensure:
//!
//! - **100% SciRS2 usage** - No direct external dependencies
//! - **All tests pass** - Run `cargo test`
//! - **Zero warnings** - Run `cargo clippy`
//! - **Documentation** - Add examples to public APIs
//!
//! ## License
//!
//! licensed under Apache-2.0

pub use optirs_core as core;

#[cfg(feature = "gpu")]
pub use optirs_gpu as gpu;

#[cfg(feature = "tpu")]
pub use optirs_tpu as tpu;

#[cfg(feature = "learned")]
pub use optirs_learned as learned;

#[cfg(feature = "nas")]
pub use optirs_nas as nas;

#[cfg(feature = "bench")]
pub use optirs_bench as bench;

/// Common imports for ease of use.
///
/// This intentionally covers only `optirs-core` (optimizers, regularizers,
/// schedulers), which is always available and whose names are verified not
/// to collide with one another. The `gpu`/`tpu`/`learned`/`nas` extension
/// crates are deliberately **not** globbed in here: they are independently
/// versioned and, with more than one enabled at once, their public names do
/// collide with `core` and with each other (for example, both
/// `optirs-core::optimizers` and `optirs-gpu` export a `SparseAdam`, and both
/// `optirs-learned` and `optirs-nas` export their own `OptimError`/`Result`).
/// A glob re-export of colliding names is ambiguous and unusable through the
/// path that introduced the ambiguity (`ambiguous_glob_reexports`), so
/// pulling them in here would silently break `optirs::prelude::SparseAdam`
/// (etc.) the moment two of those features are enabled together.
///
/// Reach extension-crate types through their own namespace instead, e.g.
/// `optirs::gpu::GpuAdam`, `optirs::learned::LSTMOptimizer`,
/// `optirs::nas::ArchitectureSpace`.
pub mod prelude {
    pub use crate::core::optimizers::*;
    pub use crate::core::regularizers::*;
    pub use crate::core::schedulers::*;
}

// Re-export core functionality at the top level
pub use crate::core::error::{OptimError, Result};
pub use crate::core::optimizers;
pub use crate::core::regularizers;
pub use crate::core::schedulers;
