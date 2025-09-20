//! # OptiRS GPU
//!
//! GPU acceleration for OptiRS optimizers.
//!
//! This crate provides:
//! - Multi-GPU optimization support
//! - CUDA, Metal, OpenCL, and WebGPU backends
//! - Memory pool management
//! - Tensor core optimizations
//! - Cross-platform GPU abstraction

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension};

use scirs2_core::gpu::GpuError;

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
