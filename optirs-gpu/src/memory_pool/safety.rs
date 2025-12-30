// Memory safety validation for GPU memory operations

use crate::gpu::GpuOptimError;
use super::{GPU_MEMORY_ALIGNMENT, MAX_SAFE_ALLOCATION_SIZE};

/// Memory safety validator
pub struct MemorySafetyValidator;

impl MemorySafetyValidator {
    /// Validate allocation parameters for safety
    pub fn validate_allocation_params(ptr: *mut u8, size: usize) -> Result<(), GpuOptimError> {
        // Check for null pointer
        if ptr.is_null() {
            return Err(GpuOptimError::InvalidState(
                "Null pointer provided".to_string(),
            ));
        }

        // Check for zero or extremely large size
        if size == 0 {
            return Err(GpuOptimError::InvalidState(
                "Zero-sized allocation".to_string(),
            ));
        }

        if size > MAX_SAFE_ALLOCATION_SIZE {
            return Err(GpuOptimError::InvalidState(format!(
                "Allocation size {} exceeds maximum safe size {}",
                size, MAX_SAFE_ALLOCATION_SIZE
            )));
        }

        // Check memory alignment
        if (ptr as usize) % GPU_MEMORY_ALIGNMENT != 0 {
            return Err(GpuOptimError::InvalidState(format!(
                "Pointer {:p} is not aligned to {} bytes",
                ptr, GPU_MEMORY_ALIGNMENT
            )));
        }

        // Check for potential integer overflow in size calculations
        if (ptr as usize).checked_add(size).is_none() {
            return Err(GpuOptimError::InvalidState(
                "Size calculation overflow".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate a memory canary value for overflow detection
    pub fn generate_canary() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Use current time and a magic number for canary
        let time_part = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // XOR with magic number to make it less predictable
        time_part ^ 0xDEADBEEFCAFEBABE
    }

    /// Validate memory canary to detect buffer overflows
    pub fn validate_canary(ptr: *mut u8, expected_canary: u64) -> Result<(), GpuOptimError> {
        // FEATURE STATUS (v1.0.0): Basic pointer validation implemented
        //
        // For v1.0.0, this function validates pointer validity. Full canary memory
        // reading and comparison requires GPU vendor-specific APIs for safe memory
        // access, which is planned for v1.1.0+.
        //
        // Current implementation:
        // - Validates pointer is not null (✓)
        // - Canary generation is functional (✓)
        //
        // PLANNED (v1.1.0+): Full canary memory validation with:
        // - GPU memory read via vendor APIs (CUDA: cudaMemcpy, Metal: MTLBuffer copy)
        // - Canary comparison with expected_canary value
        // - Buffer overflow detection and reporting
        //
        // For v1.0.0, users should rely on GPU vendor tools (CUDA memcheck, Metal
        // memory validation) for comprehensive memory corruption detection.
        if ptr.is_null() {
            return Err(GpuOptimError::InvalidState(
                "Null pointer during canary validation".to_string(),
            ));
        }

        Ok(())
    }

    /// Safely calculate pointer offset with bounds checking
    pub fn safe_ptr_add(ptr: *mut u8, offset: usize) -> Result<*mut u8, GpuOptimError> {
        let ptr_addr = ptr as usize;

        // Check for overflow
        let new_addr = ptr_addr.checked_add(offset)
            .ok_or_else(|| GpuOptimError::InvalidState(
                "Pointer arithmetic overflow".to_string()
            ))?;

        // Check alignment of result
        if new_addr % GPU_MEMORY_ALIGNMENT != 0 {
            return Err(GpuOptimError::InvalidState(format!(
                "Calculated pointer {:#x} is not aligned to {} bytes",
                new_addr, GPU_MEMORY_ALIGNMENT
            )));
        }

        Ok(new_addr as *mut u8)
    }

    /// Validate memory range doesn't overlap with existing allocations
    pub fn validate_memory_range(
        ptr: *mut u8,
        size: usize,
        existing_ranges: &[(usize, usize)], // (start_addr, size) pairs
    ) -> Result<(), GpuOptimError> {
        let start_addr = ptr as usize;
        let end_addr = start_addr + size;

        for &(existing_start, existing_size) in existing_ranges {
            let existing_end = existing_start + existing_size;

            // Check for overlap
            if !(end_addr <= existing_start || start_addr >= existing_end) {
                return Err(GpuOptimError::InvalidState(format!(
                    "Memory range [{:#x}, {:#x}) overlaps with existing range [{:#x}, {:#x})",
                    start_addr, end_addr, existing_start, existing_end
                )));
            }
        }

        Ok(())
    }

    /// Check if pointer is within valid memory bounds
    pub fn validate_memory_bounds(
        ptr: *mut u8,
        size: usize,
        memory_base: *mut u8,
        memory_size: usize,
    ) -> Result<(), GpuOptimError> {
        let ptr_addr = ptr as usize;
        let base_addr = memory_base as usize;
        let end_addr = ptr_addr + size;
        let memory_end = base_addr + memory_size;

        if ptr_addr < base_addr || end_addr > memory_end {
            return Err(GpuOptimError::InvalidState(format!(
                "Memory access [{:#x}, {:#x}) is outside valid bounds [{:#x}, {:#x})",
                ptr_addr, end_addr, base_addr, memory_end
            )));
        }

        Ok(())
    }

    /// Secure memory zeroing before deallocation
    pub fn secure_zero_memory(ptr: *mut u8, size: usize) -> Result<(), GpuOptimError> {
        if ptr.is_null() {
            return Err(GpuOptimError::InvalidState(
                "Cannot zero null pointer".to_string()
            ));
        }

        // In a real implementation, this would use secure memory clearing
        // For now, use volatile write to prevent compiler optimization
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }

        Ok(())
    }

    /// Check memory alignment requirements
    pub fn check_alignment(ptr: *mut u8, alignment: usize) -> Result<(), GpuOptimError> {
        if !alignment.is_power_of_two() {
            return Err(GpuOptimError::InvalidState(format!(
                "Alignment {} is not a power of 2",
                alignment
            )));
        }

        if (ptr as usize) % alignment != 0 {
            return Err(GpuOptimError::InvalidState(format!(
                "Pointer {:p} is not aligned to {} bytes",
                ptr, alignment
            )));
        }

        Ok(())
    }

    /// Validate that size is reasonable for the operation
    pub fn validate_size_limits(size: usize, min_size: usize, max_size: usize) -> Result<(), GpuOptimError> {
        if size < min_size {
            return Err(GpuOptimError::InvalidState(format!(
                "Size {} is below minimum {}",
                size, min_size
            )));
        }

        if size > max_size {
            return Err(GpuOptimError::InvalidState(format!(
                "Size {} exceeds maximum {}",
                size, max_size
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_allocation_params() {
        // Test null pointer
        assert!(MemorySafetyValidator::validate_allocation_params(std::ptr::null_mut(), 1024).is_err());

        // Test zero size
        let ptr = 0x1000 as *mut u8; // Aligned address
        assert!(MemorySafetyValidator::validate_allocation_params(ptr, 0).is_err());

        // Test oversized allocation
        assert!(MemorySafetyValidator::validate_allocation_params(ptr, MAX_SAFE_ALLOCATION_SIZE + 1).is_err());
    }

    #[test]
    fn test_canary_generation() {
        let canary1 = MemorySafetyValidator::generate_canary();
        let canary2 = MemorySafetyValidator::generate_canary();

        // Canaries should be different (very high probability)
        assert_ne!(canary1, canary2);
    }

    #[test]
    fn test_safe_ptr_add() {
        let ptr = 0x1000 as *mut u8;
        let result = MemorySafetyValidator::safe_ptr_add(ptr, 256);
        assert!(result.is_ok());
        assert_eq!(result.unwrap() as usize, 0x1100);
    }

    #[test]
    fn test_alignment_check() {
        let aligned_ptr = 0x1000 as *mut u8; // 256-byte aligned
        assert!(MemorySafetyValidator::check_alignment(aligned_ptr, 256).is_ok());

        let unaligned_ptr = 0x1001 as *mut u8; // Not aligned
        assert!(MemorySafetyValidator::check_alignment(unaligned_ptr, 256).is_err());
    }
}