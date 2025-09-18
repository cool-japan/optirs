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

pub mod backends;
pub mod kernels;
pub mod memory;
pub mod multi_gpu;
pub mod tensor_cores;
pub mod utils;

pub use backends::GpuBackend;
pub use memory::MemoryPool;
