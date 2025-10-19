//! # OptiRS GPU - GPU Acceleration for ML Optimization
//!
//! **Version:** 0.1.0-beta.3
//! **Status:** Framework Ready (GPU Kernels Coming Soon)
//!
//! `optirs-gpu` provides GPU acceleration for OptiRS optimizers, built on
//! [SciRS2](https://github.com/cool-japan/scirs)'s GPU abstractions.
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.1.0-rc.2 - Required foundation
//!
//! ## Implementation Status (Beta.3)
//!
//! - ✅ GPU context management
//! - ✅ Multi-backend support framework (CUDA, Metal, OpenCL, WebGPU)
//! - ✅ Memory transfer utilities
//! - ✅ Configuration and initialization
//! - 🚧 GPU kernels (in development)
//! - 🚧 Tensor cores support (in development)
//! - 📝 Multi-GPU coordination (planned)
//!
//! ## Status: Coming Soon
//!
//! This crate is under active development. GPU acceleration will leverage:
//! - `scirs2_core::gpu` for GPU context and memory management
//! - `scirs2_core::tensor_cores` for mixed-precision training
//! - `scirs2_core::array_protocol::GPUArray` for zero-copy operations
//!
//! ## Planned Features
//!
//! ### Multi-Backend Support
//! - **CUDA** - NVIDIA GPUs with full tensor core support
//! - **Metal** - Apple Silicon M1/M2/M3 with unified memory
//! - **OpenCL** - Cross-platform GPU compute
//! - **WebGPU** - Browser and cross-platform support
//!
//! ### Performance Optimizations
//! - **Tensor Cores** - FP16/BF16 mixed-precision training
//! - **Memory Pools** - Advanced GPU memory management
//! - **Kernel Fusion** - Optimized kernel execution
//! - **Multi-GPU** - Distributed optimization across GPUs
//!
//! ### Expected Speedup
//! - **10-50x** for large models (1M+ parameters)
//! - **100x+** for very large models (100M+ parameters)
//! - **Near-linear scaling** with multiple GPUs
//!
//! ## Example Usage (Future)
//!
//! ```rust,ignore
//! use optirs_gpu::GpuOptimizer;
//! use optirs::prelude::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create GPU-accelerated optimizer
//! let optimizer = Adam::new(0.001);
//! let mut gpu_opt = GpuOptimizer::new(optimizer)?;
//!
//! // Use like any optimizer - GPU acceleration is automatic
//! let params = Array1::from_elem(1_000_000, 1.0);
//! let grads = Array1::from_elem(1_000_000, 0.01);
//! let updated = gpu_opt.step(&params, &grads)?;
//! ```
//!
//! ## Architecture
//!
//! Built exclusively on SciRS2:
//! - **GPU Context**: `scirs2_core::gpu::GpuContext`
//! - **GPU Memory**: `scirs2_core::gpu::GpuBuffer`
//! - **GPU Kernels**: `scirs2_core::gpu::GpuKernel`
//! - **Tensor Cores**: `scirs2_core::tensor_cores`
//! - **Zero-Copy**: `scirs2_core::array_protocol::GPUArray`
//!
//! ## Contributing
//!
//! GPU acceleration development follows SciRS2 integration guidelines.
//! All GPU operations must use `scirs2_core::gpu` abstractions.

use scirs2_core::gpu::GpuError;
use scirs2_core::ndarray::{Array, Dimension};
use scirs2_core::numeric::Float;

pub mod backends;
pub mod kernels;
pub mod memory;
pub mod multi_gpu;
pub mod tensor_cores;
pub mod utils;

pub use backends::GpuBackend;
pub use memory::MemoryPool;

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
    fn to_gpu(&mut self) -> Result<(), GpuOptimError>;

    /// Move optimizer state back to CPU
    fn to_cpu(&mut self) -> Result<(), GpuOptimError>;

    /// Perform optimization step on GPU
    fn step_gpu(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<(), GpuOptimError>;
}
