//! # OptiRS GPU — GPU acceleration and GPU-aware optimizer tooling
//!
//! **Version:** 0.3.3
//!
//! `optirs-gpu` has two halves, and it is worth being precise about which is
//! which:
//!
//! 1. **A real GPU optimizer path.** [`optimizers`] runs Adam, AdamW, SGD,
//!    RMSprop, Adagrad and LAMB as compute shaders through
//!    [`scirs2_core::gpu`]. Parameters and gradients are uploaded to device
//!    buffers, a compiled pipeline is dispatched, and the result is read back;
//!    the per-parameter optimizer state stays resident in device memory between
//!    steps. The kernels ship in both WGSL and MSL ([`shaders`]) so the same
//!    optimizer runs on whichever backend the machine can reach.
//! 2. **A CPU library of GPU-*aware* algorithms.** [`occupancy`],
//!    [`kernel_fusion`], [`quantization`], [`sparse_optimizer`] and
//!    [`memory::allocation`] / [`memory::management`] are pure-CPU models,
//!    planners and numerical routines that reason *about* GPU execution. They
//!    have no device dependency and are fully covered by unit tests.
//!
//! ## Backend support matrix
//!
//! | Backend | Status |
//! |---------|--------|
//! | Metal (`metal`, automatic on macOS) | ✅ real compute: MSL pipelines, buffers, dispatch, readback |
//! | WebGPU (`wgpu`, default) | ✅ WGSL kernels are implemented, but `scirs2-core` 0.6.5's runtime device probe never enumerates wgpu adapters, so `GpuContext::new(Wgpu)` currently fails everywhere. The path goes live when that probe is fixed |
//! | OpenCL (`opencl`) | 🚧 context creation only — no OpenCL C kernel sources are shipped |
//! | CUDA (`cuda`) | ❌ not available — `scirs2-core` removed its CUDA backend in 0.6.x; the feature gates reporting code only |
//! | ROCm | ❌ not available |
//!
//! [`optimizers::GpuOptimizerConfig`] defaults to probing
//! [`optimizers::SUPPORTED_BACKENDS`] in order and using the first that opens.
//!
//! ## Not implemented (and not faked)
//!
//! * Cross-**device** collectives. [`multi_gpu`] can drive a real reduction
//!   kernel on a single device; anything that would require moving data
//!   between two physical GPUs returns
//!   [`GpuOptimError::UnsupportedOperation`].
//! * Literal NVIDIA tensor cores / `wmma`. [`tensor_cores`] provides real
//!   CPU-side matrix-layout optimization, precision selection and AMP loss
//!   scaling; the device GEMM entry points (`tensor_core_gemm`,
//!   `fused_adam_tensor_core`, ...) report an honest
//!   [`GpuOptimError::UnsupportedOperation`] because no backend this crate can
//!   reach exposes NVIDIA tensor cores.
//!
//! ## Example
//!
//! ```no_run
//! use optirs_gpu::optimizers::{AdamParams, GpuAdam};
//! use optirs_gpu::GpuOptimizer;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut optimizer = GpuAdam::new(AdamParams::default())?;
//! optimizer.move_to_gpu()?;
//!
//! let mut params = Array1::from_elem(1_024, 1.0f32);
//! let grads = Array1::from_elem(1_024, 0.01f32);
//! optimizer.step_gpu(&mut params, &grads)?;
//!
//! // Bring the moment estimates back to host memory when done.
//! optimizer.move_to_cpu()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! Every device access goes through SciRS2:
//! - **GPU context**: `scirs2_core::gpu::GpuContext`
//! - **GPU memory**: `scirs2_core::gpu::GpuBuffer`
//! - **Kernel compilation**: `scirs2_core::gpu::GpuCompiler`
//!
//! `wgpu`, `pollster`, `metal` and friends are *not* direct dependencies of
//! this crate; they arrive through the `scirs2-core/<backend>` features that
//! this crate's features forward.

use scirs2_core::gpu::GpuError;
use scirs2_core::ndarray::{Array, Dimension};
use scirs2_core::numeric::Float;

pub mod backends;
pub mod kernel_fusion;
pub mod memory;
pub mod mixed_precision;
pub mod multi_gpu;
pub mod occupancy;
pub mod optimizers;
pub mod quantization;
pub mod shaders;
pub mod sparse_optimizer;
pub mod tensor_cores;
pub mod utils;

pub use backends::GpuBackend;
pub use kernel_fusion::{FusionGraph, FusionGroup, FusionOp, FusionPlan, FusionPlanner, OpKind};
pub use memory::MemoryPool;
pub use mixed_precision::{
    f16_bits_to_f32, f32_to_f16_bits, DynamicLossScaler, MixedPrecisionConfig, OverflowStats,
};
pub use occupancy::{
    calculate_occupancy, optimal_block_size, KernelResourceUsage, OccupancyLimiter,
    OccupancyResult, SmResourceLimits,
};
pub use optimizers::{
    AdagradParams, AdamParams, GpuAdagrad, GpuAdam, GpuAdamW, GpuLamb, GpuOptimizerConfig,
    GpuRmsprop, GpuSgd, RmspropParams, SgdParams,
};
pub use quantization::{
    fake_quant_backward, fake_quant_fp8, fake_quant_int, fake_quant_int_per_channel,
    per_channel_params, Fp8Format, IntDtype, QatConfig, QatOptimizer, QuantParams, QuantScheme,
    QuantTarget, RoundingMode,
};
pub use sparse_optimizer::{
    CooGradient, CsrGradient, LazyAdamMode, SparseAdam, SparseAdamConfig, SparseAdamTable,
    SparseSgd, SparseSgdConfig, SparseSgdTable,
};

/// Error type for GPU optimizer operations
#[derive(Debug, thiserror::Error)]
pub enum GpuOptimError {
    /// GPU backend error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),

    /// Unsupported operation
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    /// Invalid state
    #[error("Invalid optimizer state: {0}")]
    InvalidState(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Not initialized
    #[error("GPU optimizer not initialized")]
    NotInitialized,

    /// CUDA not available
    #[error("CUDA is not available on this system")]
    CudaNotAvailable,
}

/// Trait for GPU-accelerated optimizers
pub trait GpuOptimizer<A: Float, D: Dimension> {
    /// Check if GPU acceleration is available
    fn is_gpu_available(&self) -> bool;

    /// Move optimizer state to GPU
    fn move_to_gpu(&mut self) -> Result<(), GpuOptimError>;

    /// Move optimizer state back to CPU
    fn move_to_cpu(&mut self) -> Result<(), GpuOptimError>;

    /// Deprecated alias for [`Self::move_to_gpu`].
    ///
    /// `to_gpu` on a `&mut self` method triggers
    /// `clippy::wrong_self_convention` (`to_*` names are conventionally
    /// reserved for cheap `&self` -> owned conversions); `move_to_gpu`
    /// names what this actually does. This shim delegates to
    /// [`Self::move_to_gpu`] and exists only so 0.3.1-era callers keep
    /// compiling.
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_gpu`")]
    fn to_gpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_gpu()
    }

    /// Deprecated alias for [`Self::move_to_cpu`].
    ///
    /// See [`Self::to_gpu`] for why this was renamed. This shim delegates
    /// to [`Self::move_to_cpu`] and exists only so 0.3.1-era callers keep
    /// compiling.
    #[deprecated(since = "0.3.2", note = "renamed to `move_to_cpu`")]
    fn to_cpu(&mut self) -> Result<(), GpuOptimError> {
        self.move_to_cpu()
    }

    /// Perform optimization step on GPU
    fn step_gpu(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<(), GpuOptimError>;
}
