// Utility functions for GPU memory pool operations

// Utility functions for GPU operations

use crate::GpuOptimError;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "opencl",
    feature = "wgpu"
))]
use scirs2_core::gpu::{GpuBackend, GpuContext};

/// Round size up to next alignment boundary
pub fn align_size(size: usize, alignment: usize) -> usize {
    if alignment == 0 || !alignment.is_power_of_two() {
        return size;
    }
    (size + alignment - 1) & !(alignment - 1)
}

/// Check if address is aligned to boundary
pub fn is_aligned(addr: usize, alignment: usize) -> bool {
    if !alignment.is_power_of_two() {
        return false;
    }
    addr & (alignment - 1) == 0
}

/// Calculate fragmentation ratio
pub fn calculate_fragmentation(free_blocks: &[(usize, usize)]) -> f32 {
    if free_blocks.is_empty() {
        return 0.0;
    }

    let total_free: usize = free_blocks.iter().map(|(size, count)| size * count).sum();
    let largest_block = free_blocks.iter().map(|(size, _)| *size).max().unwrap_or(0);

    if total_free == 0 {
        0.0
    } else {
        1.0 - (largest_block as f32 / total_free as f32)
    }
}

/// Format bytes in human readable format
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Calculate next power of 2
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    1 << (64 - (n - 1).leading_zeros())
}

/// Validate pointer and size parameters
pub fn validate_ptr_and_size(ptr: *mut u8, size: usize) -> Result<(), GpuOptimError> {
    if ptr.is_null() {
        return Err(GpuOptimError::InvalidState("Null pointer".to_string()));
    }

    if size == 0 {
        return Err(GpuOptimError::InvalidState("Zero size".to_string()));
    }

    Ok(())
}

/// Calculate optimal block size for GPU kernels
pub fn calculate_block_size(n: usize, max_threads: usize) -> (usize, usize) {
    let block_size = 256.min(max_threads);
    let grid_size = (n + block_size - 1) / block_size;
    (grid_size, block_size)
}

/// Get the optimal GPU backend for the current system
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "opencl",
    feature = "wgpu"
))]
pub fn get_optimal_backend() -> GpuBackend {
    // Try backends in order of preference
    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::Rocm,
        GpuBackend::Wgpu,
    ];

    for backend in &backends {
        if GpuContext::new(*backend).is_ok() {
            return *backend;
        }
    }

    // Fallback to CPU if no GPU backend available
    GpuBackend::Cpu
}

/// Get the optimal GPU backend for the current system (fallback for when no GPU features are enabled)
#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "opencl",
    feature = "wgpu"
)))]
pub fn get_optimal_backend() -> GpuBackend {
    GpuBackend::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(100, 256), 256);
        assert_eq!(align_size(256, 256), 256);
        assert_eq!(align_size(300, 256), 512);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0x1000, 256));
        assert!(!is_aligned(0x1001, 256));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(128), 128);
        assert_eq!(next_power_of_two(0), 1);
    }
}
