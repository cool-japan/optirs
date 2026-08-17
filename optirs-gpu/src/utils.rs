//! Small utilities shared by the GPU memory and kernel-launch paths.
//!
//! Everything here is pure CPU arithmetic; nothing touches a device except
//! [`get_optimal_backend`], which probes `scirs2_core::gpu` for a usable
//! backend.

use scirs2_core::gpu::{GpuBackend, GpuContext};

/// Round `size` up to the next multiple of `alignment`.
///
/// Returns `size` unchanged when `alignment` is zero or not a power of two,
/// since no meaningful rounding is defined in that case.
pub fn align_size(size: usize, alignment: usize) -> usize {
    if alignment == 0 || !alignment.is_power_of_two() {
        return size;
    }
    (size + alignment - 1) & !(alignment - 1)
}

/// Whether `addr` sits on an `alignment` boundary.
pub fn is_aligned(addr: usize, alignment: usize) -> bool {
    if !alignment.is_power_of_two() {
        return false;
    }
    addr & (alignment - 1) == 0
}

/// External fragmentation of a free list given as `(block_size, count)` pairs.
///
/// `0.0` means all free memory is in one block; values approaching `1.0` mean
/// the free memory is split into many small blocks.
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

/// Format a byte count with a binary unit suffix.
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

/// Smallest power of two greater than or equal to `n`, or `None` on overflow.
///
/// `n == 0` maps to `1`. Inputs above `usize::MAX / 2 + 1` have no representable
/// answer and yield `None` instead of silently wrapping to zero.
pub fn checked_next_power_of_two(n: usize) -> Option<usize> {
    if n == 0 {
        return Some(1);
    }
    if n.is_power_of_two() {
        return Some(n);
    }
    let shift = usize::BITS - (n - 1).leading_zeros();
    if shift >= usize::BITS {
        None
    } else {
        Some(1usize << shift)
    }
}

/// Launch geometry for a 1-D kernel over `n` elements.
///
/// Returns `(grid_size, block_size)` where `block_size` never exceeds
/// `max_threads` (nor the 256-wide workgroup the shipped kernels declare), and
/// `grid_size * block_size >= n` so the tail is always covered.
///
/// A `max_threads` of `0` is treated as `1`: a launch geometry of zero threads
/// would silently drop the whole workload.
pub fn calculate_block_size(n: usize, max_threads: usize) -> (usize, usize) {
    let block_size = crate::shaders::WORKGROUP_SIZE.min(max_threads.max(1));
    let grid_size = n.div_ceil(block_size);
    (grid_size, block_size)
}

/// First backend from `SUPPORTED_BACKENDS`-style probing that opens a context.
///
/// Backends are tried in the order WebGPU → Metal → OpenCL, then CPU as the
/// always-available fallback. CUDA and ROCm are not probed: `scirs2-core` 0.6.x
/// has no working context for either, so probing them would only cost a failed
/// `nvidia-smi`/`rocm-smi` invocation per call.
pub fn get_optimal_backend() -> GpuBackend {
    for backend in [GpuBackend::Wgpu, GpuBackend::Metal, GpuBackend::OpenCL] {
        if GpuContext::new(backend).is_ok() {
            return backend;
        }
    }
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
        // Non power-of-two and zero alignments are pass-through.
        assert_eq!(align_size(300, 3), 300);
        assert_eq!(align_size(300, 0), 300);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0x1000, 256));
        assert!(!is_aligned(0x1001, 256));
        assert!(!is_aligned(0x1000, 3));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn checked_next_power_of_two_handles_edges() {
        assert_eq!(checked_next_power_of_two(0), Some(1));
        assert_eq!(checked_next_power_of_two(1), Some(1));
        assert_eq!(checked_next_power_of_two(100), Some(128));
        assert_eq!(checked_next_power_of_two(128), Some(128));
        let highest = 1usize << (usize::BITS - 1);
        assert_eq!(checked_next_power_of_two(highest), Some(highest));
        // Anything above the highest power of two has no representable answer.
        assert_eq!(checked_next_power_of_two(highest + 1), None);
        assert_eq!(checked_next_power_of_two(usize::MAX), None);
    }

    #[test]
    fn calculate_block_size_honours_max_threads_and_covers_the_tail() {
        // Block size is clamped by max_threads...
        assert_eq!(calculate_block_size(1000, 64), (16, 64));
        // ...and by the shipped kernels' workgroup width.
        assert_eq!(calculate_block_size(1000, 1024), (4, 256));
        // The tail element is never dropped.
        let (grid, block) = calculate_block_size(257, 256);
        assert_eq!((grid, block), (2, 256));
        assert!(grid * block >= 257);
        // A zero thread budget must not produce a zero-thread launch.
        let (grid, block) = calculate_block_size(10, 0);
        assert_eq!(block, 1);
        assert_eq!(grid, 10);
    }

    #[test]
    fn calculate_fragmentation_bounds() {
        assert_eq!(calculate_fragmentation(&[]), 0.0);
        assert_eq!(calculate_fragmentation(&[(1024, 1)]), 0.0);
        let frag = calculate_fragmentation(&[(256, 4)]);
        assert!(frag > 0.7 && frag < 0.8, "unexpected fragmentation {frag}");
    }

    #[test]
    fn get_optimal_backend_returns_something_usable() {
        let backend = get_optimal_backend();
        // Whatever comes back must be a backend a context can actually open.
        assert!(
            GpuContext::new(backend).is_ok(),
            "get_optimal_backend returned unusable backend {backend}"
        );
    }
}
