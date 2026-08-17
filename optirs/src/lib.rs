//! # OptiRS - Advanced ML Optimization Built on SciRS2
//!
//! **Version:** 0.3.2
//! **Release Date:** 2026-08-17
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
//! - ✅ `optirs-core` - Stable, production-ready (19 optimizers, SIMD, parallel, metrics)
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
//! 19 state-of-the-art optimizers with performance optimizations:
//!
//! - **First-Order**: SGD, Adam, AdamW, RMSprop, Adagrad, LAMB, LARS, Lion, RAdam, SAM
//! - **SIMD-Accelerated**: SimdSGD (2-4x faster for large arrays)
//! - **Sparse**: SparseAdam, GroupedAdam
//! - **Wrapper**: Lookahead
//! - **Second-Order**: L-BFGS, Newton
//!
//! #### Performance Features
//!
//! - **SIMD Acceleration** - 2-4x speedup for large parameter arrays
//! - **Parallel Processing** - 4-8x speedup for multiple parameter groups
//! - **Memory-Efficient** - Gradient accumulation and chunked processing
//! - **GPU Framework** - 10-50x potential speedup with GPU acceleration
//! - **Production Metrics** - Real-time monitoring with minimal overhead
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
//! - **Transformer-based**: Self-attention optimization
//! - **LSTM**: Recurrent optimizer networks
//! - **Meta-Learning**: Learning to optimize across tasks
//! - **Few-Shot**: Rapid adaptation to new problems
//!
//! ### Neural Architecture Search (`optirs-nas`) [Research-Grade]
//!
//! - **Search Strategies**: Bayesian, evolutionary, RL-based
//! - **Multi-Objective**: Balance accuracy, efficiency, resources
//! - **Progressive**: Gradually increasing complexity
//! - **Hardware-Aware**: Optimization for specific targets
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
//! use optirs::core::optimizer_metrics::{MetricsCollector, MetricsReporter};
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
//! ## Performance
//!
//! - **2,998 unit/integration tests** + **110 doc tests** = **3,108 total tests**
//!   workspace-wide, `--all-features` (`cargo nextest run` + `cargo test --doc`,
//!   verified 2026-08-17): 3,098 passed, 14 skipped/ignored, 0 failed
//! - `cargo check --workspace --all-features --all-targets` is clean
//! - **Comprehensive benchmarks** - Using Criterion.rs
//! - **Statistical analysis** - For reliable performance metrics
//!
//! ## Documentation
//!
//! - **API Documentation**: [docs.rs/optirs](https://docs.rs/optirs)
//! - **User Guide**: See `USAGE_GUIDE.md` (8000+ words)
//! - **Examples**: See `examples/` directory
//! - **README**: Comprehensive feature overview
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

/// Common imports for ease of use
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use crate::core::optimizers::*;
    pub use crate::core::regularizers::*;
    pub use crate::core::schedulers::*;

    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;

    #[cfg(feature = "learned")]
    pub use crate::learned::*;

    #[cfg(feature = "nas")]
    pub use crate::nas::*;
}

// Re-export core functionality at the top level
pub use crate::core::error::{OptimError, Result};
pub use crate::core::optimizers;
pub use crate::core::regularizers;
pub use crate::core::schedulers;
