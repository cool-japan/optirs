// Batch buffer management for large GPU memory operations

use crate::gpu::GpuOptimError;
use super::config::LargeBatchConfig;
use std::time::{Duration, Instant};

#[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
use scirs2_core::gpu::GpuContext;

/// Pre-allocated batch buffer for large operations
#[derive(Debug)]
pub struct BatchBuffer {
    /// Buffer pointer
    pub ptr: *mut u8,
    /// Buffer size
    pub size: usize,
    /// Whether buffer is currently in use
    pub in_use: bool,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last used timestamp
    pub last_used: Instant,
    /// Usage count
    pub usage_count: usize,
    /// Buffer type/category
    pub buffer_type: BatchBufferType,
}

/// Types of batch buffers for different use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchBufferType {
    /// General purpose batch buffer
    General,
    /// Gradient accumulation buffer
    GradientAccumulation,
    /// Parameter update buffer
    ParameterUpdate,
    /// Multi-GPU communication buffer
    MultiGpuComm,
    /// Optimizer workspace buffer
    OptimizerWorkspace,
}

impl BatchBuffer {
    /// Create new batch buffer
    pub fn new(
        ptr: *mut u8,
        size: usize,
        buffer_type: BatchBufferType,
    ) -> Self {
        Self {
            ptr,
            size,
            in_use: false,
            created_at: Instant::now(),
            last_used: Instant::now(),
            usage_count: 0,
            buffer_type,
        }
    }

    /// Check if buffer has expired based on configured lifetime
    pub fn is_expired(&self, lifetime_seconds: u64) -> bool {
        self.created_at.elapsed().as_secs() > lifetime_seconds
    }

    /// Get efficiency score based on usage patterns
    pub fn get_efficiency_score(&self) -> f32 {
        let age_seconds = self.created_at.elapsed().as_secs() as f32;
        if age_seconds == 0.0 {
            return 0.0;
        }

        let usage_frequency = self.usage_count as f32 / age_seconds;
        let recency_factor = {
            let seconds_since_last_use = self.last_used.elapsed().as_secs() as f32;
            (-seconds_since_last_use / 3600.0).exp() // Exponential decay over 1 hour
        };

        usage_frequency * recency_factor
    }

    /// Mark buffer as used
    pub fn mark_used(&mut self) {
        self.in_use = true;
        self.last_used = Instant::now();
        self.usage_count += 1;
    }

    /// Mark buffer as free
    pub fn mark_free(&mut self) {
        self.in_use = false;
    }

    /// Get buffer utilization ratio
    pub fn get_utilization(&self, actual_used_size: usize) -> f32 {
        if self.size == 0 {
            return 0.0;
        }
        (actual_used_size as f32 / self.size as f32).min(1.0)
    }

    /// Check if buffer is suitable for the requested size
    pub fn is_suitable_for(&self, requested_size: usize, max_waste_ratio: f32) -> bool {
        if self.in_use || self.size < requested_size {
            return false;
        }

        let waste_ratio = (self.size - requested_size) as f32 / self.size as f32;
        waste_ratio <= max_waste_ratio
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Get time since last use in seconds
    pub fn idle_time_seconds(&self) -> u64 {
        self.last_used.elapsed().as_secs()
    }
}

unsafe impl Send for BatchBuffer {}
unsafe impl Sync for BatchBuffer {}

/// Batch buffer manager
pub struct BatchBufferManager {
    /// Pre-allocated buffers
    buffers: Vec<BatchBuffer>,
    /// Configuration
    config: LargeBatchConfig,
    /// GPU context for memory operations
    #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
    gpu_context: Option<std::sync::Arc<GpuContext>>,
    /// Statistics
    stats: BatchBufferStats,
}

/// Batch buffer statistics
#[derive(Debug, Clone, Default)]
pub struct BatchBufferStats {
    /// Total allocations from batch buffers
    pub batch_allocations: usize,
    /// Cache hits (reused existing buffers)
    pub cache_hits: usize,
    /// Cache misses (had to allocate new buffers)
    pub cache_misses: usize,
    /// Total bytes allocated through batch buffers
    pub total_bytes_allocated: usize,
    /// Total bytes wasted due to size mismatches
    pub total_bytes_wasted: usize,
    /// Number of buffer cleanups performed
    pub cleanup_count: usize,
    /// Number of buffers freed during cleanup
    pub buffers_freed: usize,
}

impl BatchBufferManager {
    /// Create new batch buffer manager
    pub fn new(config: LargeBatchConfig) -> Self {
        Self {
            buffers: Vec::new(),
            config,
            #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
            gpu_context: None,
            stats: BatchBufferStats::default(),
        }
    }

    /// Create manager with GPU context
    #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
    pub fn new_with_context(
        config: LargeBatchConfig,
        gpu_context: std::sync::Arc<GpuContext>,
    ) -> Self {
        Self {
            buffers: Vec::new(),
            config,
            gpu_context: Some(gpu_context),
            stats: BatchBufferStats::default(),
        }
    }

    /// Try to allocate from existing batch buffers
    pub fn try_allocate(&mut self, size: usize) -> Option<*mut u8> {
        if !self.config.qualifies_for_batch(size) {
            return None;
        }

        // Look for suitable buffer
        for buffer in &mut self.buffers {
            if buffer.is_suitable_for(size, 0.3) { // Allow up to 30% waste
                buffer.mark_used();
                self.stats.batch_allocations += 1;
                self.stats.cache_hits += 1;
                self.stats.total_bytes_allocated += size;
                self.stats.total_bytes_wasted += buffer.size - size;
                return Some(buffer.ptr);
            }
        }

        self.stats.cache_misses += 1;
        None
    }

    /// Allocate new batch buffer
    pub fn allocate_new_buffer(
        &mut self,
        size: usize,
        buffer_type: BatchBufferType,
    ) -> Result<*mut u8, GpuOptimError> {
        if self.buffers.len() >= self.config.max_batch_buffers {
            return Err(GpuOptimError::OutOfMemory(
                "Maximum batch buffers reached".to_string()
            ));
        }

        let buffer_size = self.config.calculate_buffer_size(size);
        let ptr = self.allocate_raw_memory(buffer_size)?;

        let mut buffer = BatchBuffer::new(ptr, buffer_size, buffer_type);
        buffer.mark_used();

        self.buffers.push(buffer);
        self.stats.batch_allocations += 1;
        self.stats.total_bytes_allocated += size;
        self.stats.total_bytes_wasted += buffer_size - size;

        Ok(ptr)
    }

    /// Free buffer by pointer
    pub fn free_buffer(&mut self, ptr: *mut u8) -> Result<(), GpuOptimError> {
        for buffer in &mut self.buffers {
            if buffer.ptr == ptr {
                buffer.mark_free();
                return Ok(());
            }
        }

        Err(GpuOptimError::InvalidState(
            "Buffer not found for deallocation".to_string()
        ))
    }

    /// Cleanup expired buffers
    pub fn cleanup_expired_buffers(&mut self) -> Result<usize, GpuOptimError> {
        let mut freed_count = 0;
        let lifetime = self.config.buffer_lifetime;

        self.buffers.retain(|buffer| {
            if !buffer.in_use && buffer.is_expired(lifetime) {
                #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
                {
                    if let Some(ref context) = self.gpu_context {
                        let _ = context.free_memory(buffer.ptr as *mut std::ffi::c_void);
                    }
                }
                freed_count += 1;
                false
            } else {
                true
            }
        });

        self.stats.cleanup_count += 1;
        self.stats.buffers_freed += freed_count;

        Ok(freed_count)
    }

    /// Preallocate buffers for anticipated operations
    pub fn preallocate_buffers(&mut self, sizes: &[usize]) -> Result<(), GpuOptimError> {
        for &size in sizes {
            if self.buffers.len() >= self.config.max_batch_buffers {
                break;
            }

            let _ = self.allocate_new_buffer(size, BatchBufferType::General)?;
        }

        Ok(())
    }

    /// Get buffer utilization statistics
    pub fn get_utilization_stats(&self) -> (f32, f32, usize) {
        let total_buffers = self.buffers.len();
        if total_buffers == 0 {
            return (0.0, 0.0, 0);
        }

        let used_buffers = self.buffers.iter().filter(|b| b.in_use).count();
        let usage_ratio = used_buffers as f32 / total_buffers as f32;

        let avg_efficiency = self.buffers.iter()
            .map(|b| b.get_efficiency_score())
            .sum::<f32>() / total_buffers as f32;

        (usage_ratio, avg_efficiency, total_buffers)
    }

    /// Force cleanup of all unused buffers
    pub fn force_cleanup(&mut self) -> Result<usize, GpuOptimError> {
        let mut freed_count = 0;

        self.buffers.retain(|buffer| {
            if !buffer.in_use {
                #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
                {
                    if let Some(ref context) = self.gpu_context {
                        let _ = context.free_memory(buffer.ptr as *mut std::ffi::c_void);
                    }
                }
                freed_count += 1;
                false
            } else {
                true
            }
        });

        self.stats.cleanup_count += 1;
        self.stats.buffers_freed += freed_count;

        Ok(freed_count)
    }

    /// Get statistics
    pub fn get_stats(&self) -> &BatchBufferStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BatchBufferStats::default();
    }

    /// Get number of active buffers
    pub fn active_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.in_use).count()
    }

    /// Get total buffer count
    pub fn total_buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Allocate raw memory (placeholder implementation)
    fn allocate_raw_memory(&self, size: usize) -> Result<*mut u8, GpuOptimError> {
        #[cfg(any(feature = "cuda", feature = "metal", feature = "opencl", feature = "wgpu"))]
        {
            if let Some(ref context) = self.gpu_context {
                return context.allocate_memory(size)
                    .map(|ptr| ptr as *mut u8)
                    .map_err(|e| GpuOptimError::AllocationFailed(e.to_string()));
            }
        }

        // Fallback for testing/non-GPU builds
        let layout = std::alloc::Layout::from_size_align(size, 256)
            .map_err(|_| GpuOptimError::InvalidState("Invalid layout".to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(GpuOptimError::OutOfMemory("System allocation failed".to_string()))
        } else {
            Ok(ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_buffer_creation() {
        let ptr = 0x1000 as *mut u8;
        let buffer = BatchBuffer::new(ptr, 1024, BatchBufferType::General);

        assert_eq!(buffer.ptr, ptr);
        assert_eq!(buffer.size, 1024);
        assert!(!buffer.in_use);
        assert_eq!(buffer.usage_count, 0);
    }

    #[test]
    fn test_buffer_efficiency_score() {
        let mut buffer = BatchBuffer::new(0x1000 as *mut u8, 1024, BatchBufferType::General);

        // New buffer should have low efficiency
        let initial_score = buffer.get_efficiency_score();
        assert!(initial_score >= 0.0);

        // After use, efficiency should potentially increase
        buffer.mark_used();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let used_score = buffer.get_efficiency_score();
        assert!(used_score >= 0.0);
    }

    #[test]
    fn test_batch_buffer_manager() {
        let config = LargeBatchConfig::default();
        let mut manager = BatchBufferManager::new(config);

        // Try to allocate from empty manager
        let result = manager.try_allocate(2 * 1024 * 1024);
        assert!(result.is_none());

        // Check stats
        let stats = manager.get_stats();
        assert_eq!(stats.cache_misses, 1);
    }
}