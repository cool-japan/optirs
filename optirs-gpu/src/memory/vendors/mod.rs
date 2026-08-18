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
    /// Detect available GPU vendors.
    ///
    /// This used to unconditionally claim NVIDIA *and* AMD *and* Intel were
    /// all present on every Linux/Windows machine (and Intel on every Mac,
    /// which is simply false on Apple Silicon). It now either asks something
    /// real or reports honestly empty instead of guessing:
    ///
    /// * **macOS**: every system has at least one Metal-capable device, so
    ///   this opens a real [`scirs2_core::gpu::GpuContext`] on the `Metal`
    ///   backend and reports `Apple` only if that actually succeeds. It never
    ///   claims `Intel`: most Macs sold since 2020 have none.
    /// * **Linux**: reads real PCI vendor IDs from `/sys/bus/pci/devices`
    ///   (no FFI, no root required) via the private `detect_pci_display_vendors`
    ///   helper below.
    /// * **Everywhere else** (including Windows): this crate has no
    ///   dependency-free way to query real vendor hardware, so it reports an
    ///   empty list rather than a fabricated one.
    pub fn detect_available_vendors() -> Vec<GpuVendor> {
        #[cfg(target_os = "macos")]
        {
            match scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::Metal) {
                Ok(_) => vec![GpuVendor::Apple],
                Err(_) => Vec::new(),
            }
        }

        #[cfg(target_os = "linux")]
        {
            detect_pci_display_vendors(std::path::Path::new("/sys/bus/pci/devices"))
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Vec::new()
        }
    }

    /// The first vendor from [`Self::detect_available_vendors`] this crate
    /// has a backend for, preferring the one most likely to work on the
    /// current platform. `Unknown` when detection found nothing — never a
    /// fabricated guess.
    pub fn get_preferred_vendor() -> GpuVendor {
        let vendors = Self::detect_available_vendors();
        for candidate in [
            GpuVendor::Apple,
            GpuVendor::Nvidia,
            GpuVendor::Amd,
            GpuVendor::Intel,
        ] {
            if vendors.contains(&candidate) {
                return candidate;
            }
        }
        GpuVendor::Unknown
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

/// Parse a `/sys/bus/pci/devices`-shaped directory tree for real GPU vendor
/// IDs: for every device directory whose `class` file starts with `0x03`
/// (the PCI "display controller" class), read `vendor` and map the standard
/// PCI vendor ID to a [`GpuVendor`]. Takes the root as a parameter so the
/// parsing logic is unit-testable against a fake tree without touching the
/// real `/sys` (which is kernel-owned and cannot be written to by a test).
///
/// A missing or unreadable root (non-Linux, sandboxed, containerized without
/// `/sys` mounted, ...) returns an empty list — the honest "detection is not
/// possible here" answer, not a guess.
///
/// Only the Linux arm of [`GpuBackendFactory::detect_available_vendors`]
/// calls this in production, so non-Linux builds see it as unused from the
/// compiler's point of view. It is not dead: `detect_pci_display_vendors_*`
/// below deliberately exercise this pure parsing logic against a fake sysfs
/// tree on every platform the test suite runs on (see their doc comments),
/// not just Linux, so the tests stay platform-independent rather than being
/// narrowed to `cfg(target_os = "linux")`. The allow below silences that
/// cross-platform false positive instead of deleting real, tested detection
/// logic or reducing test coverage.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn detect_pci_display_vendors(pci_root: &std::path::Path) -> Vec<GpuVendor> {
    const DISPLAY_CLASS_PREFIX: &str = "0x03";
    const NVIDIA_VENDOR_ID: &str = "0x10de";
    const AMD_VENDOR_ID: &str = "0x1002";
    const INTEL_VENDOR_ID: &str = "0x8086";

    let Ok(entries) = std::fs::read_dir(pci_root) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let device_dir = entry.path();
        let class = std::fs::read_to_string(device_dir.join("class")).unwrap_or_default();
        if !class.trim().starts_with(DISPLAY_CLASS_PREFIX) {
            continue;
        }
        let vendor_id = std::fs::read_to_string(device_dir.join("vendor")).unwrap_or_default();
        let vendor = match vendor_id.trim() {
            NVIDIA_VENDOR_ID => GpuVendor::Nvidia,
            AMD_VENDOR_ID => GpuVendor::Amd,
            INTEL_VENDOR_ID => GpuVendor::Intel,
            _ => continue,
        };
        if !found.contains(&vendor) {
            found.push(vendor);
        }
    }
    found
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

    /// Regression test for F27: `detect_available_vendors` must not claim
    /// vendors it has no evidence for. On macOS specifically, it must never
    /// claim `Intel` unconditionally — most Macs sold since 2020 (Apple
    /// Silicon) have no Intel GPU at all.
    #[test]
    fn test_vendor_detection() {
        let vendors = GpuBackendFactory::detect_available_vendors();
        // No duplicates, and every entry must be a vendor this platform's
        // detector can actually justify.
        let mut seen = Vec::new();
        for vendor in &vendors {
            assert!(
                !seen.contains(vendor),
                "duplicate vendor reported: {vendor:?}"
            );
            seen.push(vendor.clone());
        }
        #[cfg(target_os = "macos")]
        {
            assert!(
                !vendors.contains(&GpuVendor::Intel),
                "macOS detection must never assume Intel — most Macs have none"
            );
            assert!(
                !vendors.contains(&GpuVendor::Nvidia) && !vendors.contains(&GpuVendor::Amd),
                "macOS PCI vendors are not detected by this code path"
            );
        }
    }

    #[test]
    fn test_preferred_vendor() {
        // Must be self-consistent with detection: `Unknown` is legitimate
        // when nothing was detected (e.g. a sandboxed Linux CI runner with no
        // `/sys/bus/pci/devices`), so this only asserts internal consistency,
        // not "always finds something" (that was the fabrication).
        let vendors = GpuBackendFactory::detect_available_vendors();
        let preferred = GpuBackendFactory::get_preferred_vendor();
        if preferred != GpuVendor::Unknown {
            assert!(
                vendors.contains(&preferred),
                "preferred vendor {preferred:?} was not among the detected vendors {vendors:?}"
            );
        }
    }

    /// The PCI-parsing logic itself, exercised against a fake sysfs tree
    /// (never the real `/sys`, which is kernel-owned) so it is verified on
    /// every platform this test suite runs on, not just Linux.
    #[test]
    fn detect_pci_display_vendors_reads_real_vendor_ids() {
        let root = std::env::temp_dir().join(format!(
            "optirs_gpu_pci_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).expect("create fake pci root");

        let make_device = |name: &str, class: &str, vendor: &str| {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("create fake device dir");
            std::fs::write(dir.join("class"), class).expect("write class");
            std::fs::write(dir.join("vendor"), vendor).expect("write vendor");
        };

        // A real NVIDIA display controller.
        make_device("0000:01:00.0", "0x030000\n", "0x10de\n");
        // A real AMD display controller.
        make_device("0000:02:00.0", "0x030000\n", "0x1002\n");
        // A non-display NVIDIA device (e.g. an audio codec on the same
        // card) must NOT count as a display vendor.
        make_device("0000:01:00.1", "0x040300\n", "0x10de\n");
        // A display controller from an unrecognised vendor must be ignored,
        // not misattributed.
        make_device("0000:03:00.0", "0x030000\n", "0x1234\n");

        let found = detect_pci_display_vendors(&root);
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(found.len(), 2, "expected exactly NVIDIA and AMD: {found:?}");
        assert!(found.contains(&GpuVendor::Nvidia));
        assert!(found.contains(&GpuVendor::Amd));
        assert!(!found.contains(&GpuVendor::Intel));
    }

    #[test]
    fn detect_pci_display_vendors_missing_root_is_empty_not_an_error() {
        let missing = std::env::temp_dir().join("optirs_gpu_pci_test_does_not_exist_at_all");
        assert!(detect_pci_display_vendors(&missing).is_empty());
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
