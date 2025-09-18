// CUDA memory pool management for efficient GPU memory allocation
//
// This module provides memory pooling to reduce allocation overhead
// and improve performance for repeated GPU operations.

pub mod core;
pub mod safety;
pub mod allocation_strategies;
pub mod adaptive;
pub mod pressure_monitor;
pub mod batch_buffers;
pub mod stats;
pub mod config;
pub mod utils;

// Re-export main types to maintain backward compatibility
pub use core::CudaMemoryPool;
pub use safety::MemorySafetyValidator;
pub use allocation_strategies::{AllocationStrategy, AllocationImplementations};
pub use adaptive::{AdaptiveSizing, AllocationEvent};
pub use pressure_monitor::{MemoryPressureMonitor, PressureReading};
pub use batch_buffers::{BatchBuffer, BatchBufferType};
pub use stats::{MemoryStats, DetailedMemoryStats};
pub use config::LargeBatchConfig;

use crate::gpu::GpuOptimError;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuContext;

/// Memory alignment for GPU operations (must be power of 2)
pub const GPU_MEMORY_ALIGNMENT: usize = 256;

/// Maximum safe allocation size (1GB)
pub const MAX_SAFE_ALLOCATION_SIZE: usize = 1024 * 1024 * 1024;

/// Memory block representation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Memory pointer
    pub ptr: std::ptr::NonNull<u8>,
    /// Block size
    pub size: usize,
    /// Whether block is in use
    pub in_use: bool,
    /// Creation timestamp
    pub created_at: std::time::Instant,
    /// Last access timestamp
    pub last_used: std::time::Instant,
    /// Memory canary for overflow detection
    pub memory_canary: u64,
}

impl MemoryBlock {
    /// Create new memory block
    pub fn new(ptr: *mut u8, size: usize) -> Result<Self, GpuOptimError> {
        let ptr = std::ptr::NonNull::new(ptr)
            .ok_or_else(|| GpuOptimError::InvalidState("Null pointer".to_string()))?;

        let canary = safety::MemorySafetyValidator::generate_canary();

        Ok(Self {
            ptr,
            size,
            in_use: false,
            created_at: std::time::Instant::now(),
            last_used: std::time::Instant::now(),
            memory_canary: canary,
        })
    }

    /// Mark block as used
    pub fn mark_used(&mut self) {
        self.in_use = true;
        self.last_used = std::time::Instant::now();
    }

    /// Mark block as free
    pub fn mark_free(&mut self) {
        self.in_use = false;
    }

    /// Validate memory integrity using canary
    pub fn validate_integrity(&self) -> Result<(), GpuOptimError> {
        safety::MemorySafetyValidator::validate_canary(self.ptr.as_ptr(), self.memory_canary)
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}