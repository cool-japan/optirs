//! GPU backend identifiers and capability data.
//!
//! This module used to also define a `Backend` trait with five
//! implementations (`CudaBackend`, `RocmBackend`, `MetalBackend`,
//! `WgpuBackend`, `CpuBackend`) that were pure fabrication: every
//! `allocate`/`copy_to_device`/`copy_to_host`/`launch_kernel` was a bare
//! `Ok(())` regardless of backend availability, `DeviceCapabilities` numbers
//! were hardcoded fiction (`"CUDA Device"`, 8 GB, compute capability `(8,
//! 6)`, ...) on every machine, and `copy_to_host` left its destination
//! untouched while reporting success. Nothing outside this file ever called
//! any of it — it was dead, misleading public API surface.
//!
//! Real GPU access in this crate goes through [`scirs2_core::gpu`]
//! exclusively (see [`crate::optimizers`], [`crate::multi_gpu`]). What
//! remains here is the parts of the old surface that are still load-bearing:
//! the [`GpuBackend`] identifier enum and the [`DeviceCapabilities`] data
//! struct [`crate::occupancy`] uses to model launch-configuration limits —
//! neither of those fabricates anything by existing.

use thiserror::Error;

/// GPU backend types supported by the optimizer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm backend
    Rocm,
    /// Apple Metal backend
    Metal,
    /// WebGPU backend (cross-platform)
    Wgpu,
    /// CPU fallback (no GPU acceleration)
    Cpu,
}

impl Default for GpuBackend {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return Self::Metal;

        #[cfg(not(target_os = "macos"))]
        return Self::Cuda;
    }
}

/// Errors that can occur with GPU backends
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Backend not available: {backend:?}")]
    NotAvailable { backend: GpuBackend },

    #[error("Backend initialization failed: {reason}")]
    InitializationFailed { reason: String },

    #[error("Operation not supported by backend: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Backend error: {message}")]
    BackendSpecific { message: String },

    #[error("Device error: {device_id}")]
    DeviceError { device_id: u32 },
}

/// GPU device capabilities.
///
/// A plain data holder — nothing in this module manufactures values for it.
/// [`crate::occupancy`] derives real [`crate::occupancy::SmResourceLimits`]
/// from whatever is put here; populating it with honest numbers (e.g. from
/// `scirs2_core::gpu::GpuContext::backend()` plus documented per-vendor
/// specs) is the caller's responsibility.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name
    pub name: String,

    /// Total memory in bytes
    pub total_memory: usize,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Supports half precision (f16)
    pub supports_f16: bool,

    /// Supports bfloat16
    pub supports_bf16: bool,

    /// Supports tensor cores
    pub supports_tensor_cores: bool,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Maximum shared memory per block
    pub max_shared_memory_per_block: usize,

    /// Number of streaming multiprocessors
    pub multiprocessor_count: u32,

    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),
}

/// Kernel launch geometry: grid/block dimensions, shared-memory footprint
/// and (optionally) the stream to launch on.
///
/// A plain data holder, populated by the caller (e.g. via
/// [`crate::utils::calculate_block_size`] or
/// [`crate::occupancy::optimal_block_size`]) — this type does not compute or
/// fabricate a launch configuration by itself.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_size: (u32, u32, u32),

    /// Block dimensions (x, y, z)
    pub block_size: (u32, u32, u32),

    /// Shared memory size in bytes
    pub shared_memory_size: usize,

    /// Backend-specific stream identifier, if the launch targets one
    pub stream: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_matches_the_build_platform() {
        let backend = GpuBackend::default();
        #[cfg(target_os = "macos")]
        assert_eq!(backend, GpuBackend::Metal);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(backend, GpuBackend::Cuda);
    }

    #[test]
    fn device_capabilities_is_a_plain_data_struct() {
        // No factory manufactures this — constructing it directly with
        // caller-supplied numbers is the only way to get one, which is the
        // point of deleting the fabricating `Backend` impls.
        let caps = DeviceCapabilities {
            name: "test device".to_string(),
            total_memory: 1024,
            available_memory: 512,
            supports_f16: false,
            supports_bf16: false,
            supports_tensor_cores: false,
            max_threads_per_block: 256,
            max_shared_memory_per_block: 0,
            multiprocessor_count: 1,
            compute_capability: (0, 0),
        };
        assert_eq!(caps.name, "test device");
        assert_eq!(caps.total_memory, 1024);
    }
}
