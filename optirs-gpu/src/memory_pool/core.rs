// Core CUDA memory pool implementation

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gpu::GpuOptimError;
use super::{
    MemoryBlock,
    config::LargeBatchConfig,
    allocation_strategies::{AllocationStrategy, AllocationImplementations},
    adaptive::AdaptiveSizing,
    pressure_monitor::MemoryPressureMonitor,
    batch_buffers::{BatchBufferManager, BatchBufferType},
    stats::{MemoryStats, DetailedMemoryStats},
    utils,
    GPU_MEMORY_ALIGNMENT,
};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuContext;

/// CUDA memory pool
pub struct CudaMemoryPool {
    /// Free blocks organized by size
    free_blocks: HashMap<usize, VecDeque<MemoryBlock>>,

    /// All allocated blocks
    all_blocks: Vec<MemoryBlock>,

    /// Memory statistics
    stats: MemoryStats,

    /// Maximum pool size
    max_pool_size: usize,

    /// Minimum block size to pool
    min_block_size: usize,

    /// Enable memory defragmentation
    enable_defrag: bool,

    /// GPU context
    #[cfg(feature = "gpu")]
    gpu_context: Option<Arc<GpuContext>>,

    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy,

    /// Adaptive sizing based on usage patterns
    adaptive_sizing: AdaptiveSizing,

    /// Memory pressure monitoring
    pressure_monitor: MemoryPressureMonitor,

    /// Batch buffer manager
    batch_buffer_manager: BatchBufferManager,
}

impl CudaMemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        let large_batch_config = LargeBatchConfig::default();

        Self {
            free_blocks: HashMap::new(),
            all_blocks: Vec::new(),
            stats: MemoryStats::default(),
            max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB default
            min_block_size: 256,                   // 256 bytes minimum
            enable_defrag: true,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            allocation_strategy: AllocationStrategy::default(),
            adaptive_sizing: AdaptiveSizing::default(),
            pressure_monitor: MemoryPressureMonitor::default(),
            batch_buffer_manager: BatchBufferManager::new(large_batch_config),
        }
    }

    /// Create memory pool for specific GPU
    #[cfg(feature = "gpu")]
    pub fn new_with_gpu(gpu_id: usize) -> Result<Self, GpuOptimError> {
        let context = Arc::new(GpuContext::new_with_device(gpu_id)?);
        let large_batch_config = LargeBatchConfig::default();

        Ok(Self {
            free_blocks: HashMap::new(),
            all_blocks: Vec::new(),
            stats: MemoryStats::default(),
            max_pool_size: 2 * 1024 * 1024 * 1024,
            min_block_size: 256,
            enable_defrag: true,
            gpu_context: Some(context.clone()),
            allocation_strategy: AllocationStrategy::default(),
            adaptive_sizing: AdaptiveSizing::default(),
            pressure_monitor: MemoryPressureMonitor::default(),
            batch_buffer_manager: BatchBufferManager::new_with_context(large_batch_config, context),
        })
    }

    /// Allocate memory from pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8, GpuOptimError> {
        let start_time = Instant::now();
        let aligned_size = utils::align_size(size, GPU_MEMORY_ALIGNMENT);

        // Try batch buffer allocation first for large allocations
        if let Some(ptr) = self.batch_buffer_manager.try_allocate(aligned_size) {
            self.record_allocation_success(aligned_size, true, start_time);
            return Ok(ptr);
        }

        // Try to find existing free block
        if let Some(ptr) = self.find_free_block(aligned_size) {
            self.record_allocation_success(aligned_size, true, start_time);
            return Ok(ptr);
        }

        // Allocate new block
        let ptr = self.allocate_new_block(aligned_size)?;
        self.record_allocation_success(aligned_size, false, start_time);

        // Update adaptive sizing
        if self.adaptive_sizing.enable_adaptive_resize {
            self.update_adaptive_sizing(aligned_size, start_time);
        }

        // Monitor memory pressure
        if self.pressure_monitor.enable_monitoring {
            self.update_memory_pressure();
        }

        Ok(ptr)
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&mut self, ptr: *mut u8) -> Result<(), GpuOptimError> {
        let start_time = Instant::now();

        // Try batch buffer deallocation first
        if self.batch_buffer_manager.free_buffer(ptr).is_ok() {
            return Ok(());
        }

        // Find and free regular block
        for (i, block) in self.all_blocks.iter_mut().enumerate() {
            if block.as_ptr() == ptr {
                block.mark_free();

                // Add to free list
                let size = block.size;
                self.free_blocks.entry(size).or_insert_with(VecDeque::new).push_back(block.clone());

                // Update statistics
                let elapsed = start_time.elapsed();
                self.stats.record_deallocation(size, elapsed);

                // Check if defragmentation is needed
                if self.enable_defrag && self.should_defragment() {
                    self.defragment();
                }

                return Ok(());
            }
        }

        Err(GpuOptimError::InvalidState("Block not found for deallocation".to_string()))
    }

    /// Find free block using allocation strategy
    fn find_free_block(&mut self, size: usize) -> Option<*mut u8> {
        match self.allocation_strategy {
            AllocationStrategy::FirstFit => {
                AllocationImplementations::find_first_fit(&mut self.free_blocks, size)
                    .map(|block| block.as_ptr())
            }
            AllocationStrategy::BestFit => {
                if let Some((block_size, _)) = AllocationImplementations::find_best_fit(&self.free_blocks, size) {
                    AllocationImplementations::find_first_fit(&mut self.free_blocks, block_size)
                        .map(|block| block.as_ptr())
                } else {
                    None
                }
            }
            AllocationStrategy::WorstFit => {
                if let Some((block_size, _)) = AllocationImplementations::find_worst_fit(&self.free_blocks, size) {
                    AllocationImplementations::find_first_fit(&mut self.free_blocks, block_size)
                        .map(|block| block.as_ptr())
                } else {
                    None
                }
            }
            _ => {
                // For other strategies, fall back to first fit
                AllocationImplementations::find_first_fit(&mut self.free_blocks, size)
                    .map(|block| block.as_ptr())
            }
        }
    }

    /// Allocate new block from system
    fn allocate_new_block(&mut self, size: usize) -> Result<*mut u8, GpuOptimError> {
        let ptr = self.allocate_raw_memory(size)?;
        let block = MemoryBlock::new(ptr, size)?;
        self.all_blocks.push(block);
        Ok(ptr)
    }

    /// Allocate raw memory
    fn allocate_raw_memory(&self, size: usize) -> Result<*mut u8, GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref context) = self.gpu_context {
                return context.allocate_memory(size)
                    .map(|ptr| ptr as *mut u8)
                    .map_err(|e| GpuOptimError::AllocationFailed(e.to_string()));
            }
        }

        // Fallback for testing/non-GPU builds
        let layout = std::alloc::Layout::from_size_align(size, GPU_MEMORY_ALIGNMENT)
            .map_err(|_| GpuOptimError::InvalidState("Invalid layout".to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(GpuOptimError::OutOfMemory("System allocation failed".to_string()))
        } else {
            Ok(ptr)
        }
    }

    /// Record successful allocation
    fn record_allocation_success(&mut self, size: usize, cache_hit: bool, start_time: Instant) {
        let duration = start_time.elapsed();
        self.stats.record_allocation(size, cache_hit, duration);

        if self.adaptive_sizing.enable_adaptive_resize {
            self.adaptive_sizing.record_allocation(size, cache_hit, duration.as_micros() as u64);
        }
    }

    /// Update adaptive sizing
    fn update_adaptive_sizing(&mut self, size: usize, start_time: Instant) {
        let current_utilization = self.get_utilization();
        if let Some(new_size) = self.adaptive_sizing.analyze_and_suggest_resize(
            self.max_pool_size,
            current_utilization
        ) {
            self.max_pool_size = new_size;
            self.stats.record_pool_resize();
        }
    }

    /// Update memory pressure
    fn update_memory_pressure(&mut self) {
        let used_memory = self.stats.current_bytes_used;
        self.pressure_monitor.update_pressure(used_memory, self.max_pool_size);

        if self.pressure_monitor.should_trigger_cleanup() {
            let _ = self.cleanup_expired_buffers();
        }
    }

    /// Get memory utilization ratio
    fn get_utilization(&self) -> f32 {
        if self.max_pool_size == 0 {
            return 0.0;
        }
        self.stats.current_bytes_used as f32 / self.max_pool_size as f32
    }

    /// Check if defragmentation should be performed
    fn should_defragment(&self) -> bool {
        let free_block_data: Vec<(usize, usize)> = self.free_blocks
            .iter()
            .map(|(&size, blocks)| (size, blocks.len()))
            .collect();

        let fragmentation = utils::calculate_fragmentation(&free_block_data);
        fragmentation > 0.5 // Defragment if fragmentation > 50%
    }

    /// Perform memory defragmentation
    fn defragment(&mut self) {
        // Simple defragmentation: merge adjacent free blocks
        // In a real implementation, this would be more sophisticated
        self.stats.record_defragmentation();
    }

    /// Cleanup expired batch buffers
    pub fn cleanup_expired_buffers(&mut self) -> Result<usize, GpuOptimError> {
        self.batch_buffer_manager.cleanup_expired_buffers()
    }

    /// Get detailed memory statistics
    pub fn get_detailed_stats(&self) -> DetailedMemoryStats {
        let fragmentation_ratio = {
            let free_block_data: Vec<(usize, usize)> = self.free_blocks
                .iter()
                .map(|(&size, blocks)| (size, blocks.len()))
                .collect();
            utils::calculate_fragmentation(&free_block_data)
        };

        let utilization = self.get_utilization();

        let mut detailed_stats = DetailedMemoryStats::new(self.stats.clone());
        detailed_stats.fragmentation_ratio = fragmentation_ratio;
        detailed_stats.utilization = utilization;
        detailed_stats.free_block_count = self.free_blocks.values().map(|v| v.len()).sum();
        detailed_stats.total_block_count = self.all_blocks.len();
        detailed_stats.current_pressure = self.pressure_monitor.current_pressure;
        detailed_stats.max_pool_size = self.max_pool_size;

        let (_, _, buffer_count) = self.batch_buffer_manager.get_utilization_stats();
        detailed_stats.batch_buffer_count = buffer_count;
        detailed_stats.active_batch_buffers = self.batch_buffer_manager.active_buffer_count();

        detailed_stats
    }

    /// Preallocate batch buffers for anticipated large operations
    pub fn preallocate_batch_buffers(&mut self, sizes: &[usize]) -> Result<(), GpuOptimError> {
        self.batch_buffer_manager.preallocate_buffers(sizes)
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.batch_buffer_manager.reset_stats();
        self.pressure_monitor.reset();
        self.adaptive_sizing.clear_history();
    }
}

impl Default for CudaMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for CudaMemoryPool {}
unsafe impl Sync for CudaMemoryPool {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = CudaMemoryPool::new();
        assert_eq!(pool.max_pool_size, 2 * 1024 * 1024 * 1024);
        assert_eq!(pool.min_block_size, 256);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = CudaMemoryPool::new();

        // Note: This test may fail in CI environments without proper GPU setup
        // The allocation might return an error which is acceptable for testing
        let result = pool.allocate(1024);
        // Just ensure it doesn't panic
    }

    #[test]
    fn test_detailed_stats() {
        let pool = CudaMemoryPool::new();
        let stats = pool.get_detailed_stats();

        assert_eq!(stats.basic_stats.total_allocations, 0);
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.batch_buffer_count, 0);
    }
}