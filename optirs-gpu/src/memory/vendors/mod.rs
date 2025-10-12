// Vendor-specific GPU memory backends
//
// This module provides vendor-specific GPU memory management implementations
// for different GPU architectures and platforms.

pub mod cuda_backend;
pub mod metal_backend;
pub mod oneapi_backend;
pub mod rocm_backend;

use std::ffi::c_void;
use std::time::Duration;

pub use cuda_backend::{
    CudaConfig, CudaError, CudaMemoryBackend, CudaMemoryType, ThreadSafeCudaBackend,
};
pub use metal_backend::{
    MetalConfig, MetalError, MetalMemoryBackend, MetalMemoryType, ThreadSafeMetalBackend,
};
pub use oneapi_backend::{
    OneApiConfig, OneApiError, OneApiMemoryBackend, OneApiMemoryType, ThreadSafeOneApiBackend,
};
pub use rocm_backend::{
    RocmConfig, RocmError, RocmMemoryBackend, RocmMemoryType, ThreadSafeRocmBackend,
};

/// Unified GPU vendor types
#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

/// Unified memory backend trait for all GPU vendors
pub trait GpuMemoryBackend {
    type Error: std::error::Error + Send + Sync + 'static;
    type MemoryType: Clone + PartialEq;
    type Stats: Clone;

    /// Allocate GPU memory
    fn allocate(
        &mut self,
        size: usize,
        memory_type: Self::MemoryType,
    ) -> Result<*mut c_void, Self::Error>;

    /// Free GPU memory
    fn free(&mut self, ptr: *mut c_void, memory_type: Self::MemoryType) -> Result<(), Self::Error>;

    /// Get memory statistics
    fn get_stats(&self) -> Self::Stats;

    /// Synchronize all operations
    fn synchronize(&mut self) -> Result<(), Self::Error>;

    /// Get GPU vendor
    fn get_vendor(&self) -> GpuVendor;

    /// Get device name
    fn get_device_name(&self) -> &str;

    /// Get total memory size
    fn get_total_memory(&self) -> usize;
}

/// Vendor detection and backend creation
pub struct GpuBackendFactory;

impl GpuBackendFactory {
    /// Detect available GPU vendors
    pub fn detect_available_vendors() -> Vec<GpuVendor> {
        let mut vendors = Vec::new();

        // Simulate vendor detection
        #[cfg(target_os = "linux")]
        {
            vendors.push(GpuVendor::Nvidia);
            vendors.push(GpuVendor::Amd);
            vendors.push(GpuVendor::Intel);
        }

        #[cfg(target_os = "windows")]
        {
            vendors.push(GpuVendor::Nvidia);
            vendors.push(GpuVendor::Amd);
            vendors.push(GpuVendor::Intel);
        }

        #[cfg(target_os = "macos")]
        {
            vendors.push(GpuVendor::Apple);
            vendors.push(GpuVendor::Intel); // Intel Macs
        }

        vendors
    }

    /// Get preferred vendor based on platform
    pub fn get_preferred_vendor() -> GpuVendor {
        #[cfg(target_os = "macos")]
        {
            GpuVendor::Apple
        }

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // Prefer NVIDIA for CUDA support, then AMD, then Intel
            let vendors = Self::detect_available_vendors();
            if vendors.contains(&GpuVendor::Nvidia) {
                return GpuVendor::Nvidia;
            } else if vendors.contains(&GpuVendor::Amd) {
                return GpuVendor::Amd;
            } else if vendors.contains(&GpuVendor::Intel) {
                return GpuVendor::Intel;
            }
            return GpuVendor::Unknown;
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            GpuVendor::Unknown
        }
    }

    /// Create backend configuration for vendor
    pub fn create_default_config(vendor: GpuVendor) -> VendorConfig {
        match vendor {
            GpuVendor::Nvidia => VendorConfig::Cuda(CudaConfig::default()),
            GpuVendor::Amd => VendorConfig::Rocm(RocmConfig::default()),
            GpuVendor::Intel => VendorConfig::OneApi(OneApiConfig::default()),
            GpuVendor::Apple => VendorConfig::Metal(MetalConfig::default()),
            GpuVendor::Unknown => VendorConfig::Cuda(CudaConfig::default()), // Fallback
        }
    }
}

/// Unified configuration for all vendors
#[derive(Debug, Clone)]
pub enum VendorConfig {
    Cuda(CudaConfig),
    Rocm(RocmConfig),
    OneApi(OneApiConfig),
    Metal(MetalConfig),
}

/// Unified backend wrapper
pub enum UnifiedGpuBackend {
    Cuda(CudaMemoryBackend),
    Rocm(RocmMemoryBackend),
    OneApi(OneApiMemoryBackend),
    Metal(MetalMemoryBackend),
}

impl UnifiedGpuBackend {
    /// Create backend from configuration
    pub fn new(config: VendorConfig) -> Result<Self, UnifiedGpuError> {
        match config {
            VendorConfig::Cuda(config) => {
                let backend = CudaMemoryBackend::new(config)?;
                Ok(UnifiedGpuBackend::Cuda(backend))
            }
            VendorConfig::Rocm(config) => {
                let backend = RocmMemoryBackend::new(config)?;
                Ok(UnifiedGpuBackend::Rocm(backend))
            }
            VendorConfig::OneApi(config) => {
                let backend = OneApiMemoryBackend::new(config)?;
                Ok(UnifiedGpuBackend::OneApi(backend))
            }
            VendorConfig::Metal(config) => {
                let backend = MetalMemoryBackend::new(config)?;
                Ok(UnifiedGpuBackend::Metal(backend))
            }
        }
    }

    /// Auto-detect and create best backend
    pub fn auto_create() -> Result<Self, UnifiedGpuError> {
        let vendor = GpuBackendFactory::get_preferred_vendor();
        let config = GpuBackendFactory::create_default_config(vendor);
        Self::new(config)
    }

    /// Get vendor type
    pub fn get_vendor(&self) -> GpuVendor {
        match self {
            UnifiedGpuBackend::Cuda(_) => GpuVendor::Nvidia,
            UnifiedGpuBackend::Rocm(_) => GpuVendor::Amd,
            UnifiedGpuBackend::OneApi(_) => GpuVendor::Intel,
            UnifiedGpuBackend::Metal(_) => GpuVendor::Apple,
        }
    }

    /// Allocate memory with unified interface
    pub fn allocate(&mut self, size: usize) -> Result<*mut c_void, UnifiedGpuError> {
        match self {
            UnifiedGpuBackend::Cuda(backend) => backend
                .allocate(size, CudaMemoryType::Device)
                .map_err(UnifiedGpuError::Cuda),
            UnifiedGpuBackend::Rocm(backend) => backend
                .allocate(size, RocmMemoryType::Device)
                .map_err(UnifiedGpuError::Rocm),
            UnifiedGpuBackend::OneApi(backend) => backend
                .allocate(size, OneApiMemoryType::Device)
                .map_err(UnifiedGpuError::OneApi),
            UnifiedGpuBackend::Metal(backend) => backend
                .allocate(size, MetalMemoryType::Private)
                .map_err(UnifiedGpuError::Metal),
        }
    }

    /// Free memory with unified interface
    pub fn free(&mut self, ptr: *mut c_void) -> Result<(), UnifiedGpuError> {
        match self {
            UnifiedGpuBackend::Cuda(backend) => backend
                .free(ptr, CudaMemoryType::Device)
                .map_err(UnifiedGpuError::Cuda),
            UnifiedGpuBackend::Rocm(backend) => backend
                .free(ptr, RocmMemoryType::Device)
                .map_err(UnifiedGpuError::Rocm),
            UnifiedGpuBackend::OneApi(backend) => backend
                .free(ptr, OneApiMemoryType::Device)
                .map_err(UnifiedGpuError::OneApi),
            UnifiedGpuBackend::Metal(backend) => backend
                .free(ptr, MetalMemoryType::Private)
                .map_err(UnifiedGpuError::Metal),
        }
    }

    /// Get unified memory statistics
    /// Get total available GPU memory
    pub fn get_total_memory(&self) -> usize {
        // Default to 8GB if backend doesn't provide memory info
        // Individual backends should implement proper memory querying
        match self {
            UnifiedGpuBackend::Cuda(_) => 8 * 1024 * 1024 * 1024, // 8GB default for CUDA
            UnifiedGpuBackend::Rocm(_) => 8 * 1024 * 1024 * 1024, // 8GB default for ROCm
            UnifiedGpuBackend::OneApi(_) => 8 * 1024 * 1024 * 1024, // 8GB default for OneAPI
            UnifiedGpuBackend::Metal(_) => 8 * 1024 * 1024 * 1024, // 8GB default for Metal
        }
    }

    pub fn get_memory_stats(&self) -> UnifiedMemoryStats {
        match self {
            UnifiedGpuBackend::Cuda(backend) => {
                let stats = backend.get_stats();
                UnifiedMemoryStats {
                    total_allocations: stats.total_allocations,
                    bytes_allocated: stats.bytes_allocated,
                    peak_memory_usage: stats.peak_memory_usage,
                    average_allocation_time: stats.average_allocation_time,
                }
            }
            UnifiedGpuBackend::Rocm(backend) => {
                let stats = backend.get_stats();
                UnifiedMemoryStats {
                    total_allocations: stats.total_allocations,
                    bytes_allocated: stats.bytes_allocated,
                    peak_memory_usage: stats.peak_memory_usage,
                    average_allocation_time: stats.average_allocation_time,
                }
            }
            UnifiedGpuBackend::OneApi(backend) => {
                let stats = backend.get_stats();
                UnifiedMemoryStats {
                    total_allocations: stats.total_allocations,
                    bytes_allocated: stats.bytes_allocated,
                    peak_memory_usage: stats.peak_memory_usage,
                    average_allocation_time: stats.average_allocation_time,
                }
            }
            UnifiedGpuBackend::Metal(backend) => {
                let stats = backend.get_stats();
                UnifiedMemoryStats {
                    total_allocations: stats.total_allocations,
                    bytes_allocated: stats.bytes_allocated,
                    peak_memory_usage: stats.peak_memory_usage,
                    average_allocation_time: stats.average_allocation_time,
                }
            }
        }
    }
}

/// Unified memory statistics across all vendors
#[derive(Debug, Clone, Default)]
pub struct UnifiedMemoryStats {
    pub total_allocations: u64,
    pub bytes_allocated: u64,
    pub peak_memory_usage: usize,
    pub average_allocation_time: Duration,
}

/// Unified error type for all GPU backends
#[derive(Debug)]
pub enum UnifiedGpuError {
    Cuda(CudaError),
    Rocm(RocmError),
    OneApi(OneApiError),
    Metal(MetalError),
    VendorNotSupported(String),
    InitializationFailed(String),
}

impl std::fmt::Display for UnifiedGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnifiedGpuError::Cuda(err) => write!(f, "CUDA Error: {}", err),
            UnifiedGpuError::Rocm(err) => write!(f, "ROCm Error: {}", err),
            UnifiedGpuError::OneApi(err) => write!(f, "OneAPI Error: {}", err),
            UnifiedGpuError::Metal(err) => write!(f, "Metal Error: {}", err),
            UnifiedGpuError::VendorNotSupported(msg) => write!(f, "Vendor not supported: {}", msg),
            UnifiedGpuError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for UnifiedGpuError {}

impl From<CudaError> for UnifiedGpuError {
    fn from(err: CudaError) -> Self {
        UnifiedGpuError::Cuda(err)
    }
}

impl From<RocmError> for UnifiedGpuError {
    fn from(err: RocmError) -> Self {
        UnifiedGpuError::Rocm(err)
    }
}

impl From<OneApiError> for UnifiedGpuError {
    fn from(err: OneApiError) -> Self {
        UnifiedGpuError::OneApi(err)
    }
}

impl From<MetalError> for UnifiedGpuError {
    fn from(err: MetalError) -> Self {
        UnifiedGpuError::Metal(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_detection() {
        let vendors = GpuBackendFactory::detect_available_vendors();
        assert!(!vendors.is_empty());
    }

    #[test]
    fn test_preferred_vendor() {
        let vendor = GpuBackendFactory::get_preferred_vendor();
        assert_ne!(vendor, GpuVendor::Unknown);
    }

    #[test]
    fn test_unified_backend_creation() {
        let vendor = GpuBackendFactory::get_preferred_vendor();
        let config = GpuBackendFactory::create_default_config(vendor);
        let backend = UnifiedGpuBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_auto_create() {
        let backend = UnifiedGpuBackend::auto_create();
        assert!(backend.is_ok());
    }
}
