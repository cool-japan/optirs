// Configuration structures for GPU memory pool management

/// Large batch optimization configuration
#[derive(Debug, Clone)]
pub struct LargeBatchConfig {
    /// Minimum batch size to consider for optimization
    pub min_batch_size: usize,
    /// Maximum number of pre-allocated batch buffers
    pub max_batch_buffers: usize,
    /// Buffer size growth factor
    pub growth_factor: f32,
    /// Enable batch buffer coalescing
    pub enable_coalescing: bool,
    /// Pre-allocation threshold (percentage of max pool size)
    pub preallocation_threshold: f32,
    /// Batch buffer lifetime (seconds)
    pub buffer_lifetime: u64,
}

impl Default for LargeBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1024 * 1024, // 1MB
            max_batch_buffers: 16,
            growth_factor: 1.5,
            enable_coalescing: true,
            preallocation_threshold: 0.8,
            buffer_lifetime: 300, // 5 minutes
        }
    }
}

impl LargeBatchConfig {
    /// Create a new configuration with custom parameters
    pub fn new(
        min_batch_size: usize,
        max_batch_buffers: usize,
        growth_factor: f32,
        buffer_lifetime: u64,
    ) -> Self {
        Self {
            min_batch_size,
            max_batch_buffers,
            growth_factor,
            enable_coalescing: true,
            preallocation_threshold: 0.8,
            buffer_lifetime,
        }
    }

    /// Create configuration optimized for training workloads
    pub fn for_training() -> Self {
        Self {
            min_batch_size: 512 * 1024, // 512KB
            max_batch_buffers: 32,
            growth_factor: 2.0,
            enable_coalescing: true,
            preallocation_threshold: 0.7,
            buffer_lifetime: 600, // 10 minutes
        }
    }

    /// Create configuration optimized for inference workloads
    pub fn for_inference() -> Self {
        Self {
            min_batch_size: 2 * 1024 * 1024, // 2MB
            max_batch_buffers: 8,
            growth_factor: 1.2,
            enable_coalescing: false, // Lower overhead for inference
            preallocation_threshold: 0.9,
            buffer_lifetime: 120, // 2 minutes
        }
    }

    /// Create configuration for memory-constrained environments
    pub fn for_low_memory() -> Self {
        Self {
            min_batch_size: 2 * 1024 * 1024, // 2MB
            max_batch_buffers: 4,
            growth_factor: 1.1,
            enable_coalescing: true,
            preallocation_threshold: 0.95,
            buffer_lifetime: 60, // 1 minute
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.min_batch_size == 0 {
            return Err("min_batch_size must be greater than 0".to_string());
        }

        if self.max_batch_buffers == 0 {
            return Err("max_batch_buffers must be greater than 0".to_string());
        }

        if self.growth_factor <= 1.0 {
            return Err("growth_factor must be greater than 1.0".to_string());
        }

        if self.preallocation_threshold < 0.0 || self.preallocation_threshold > 1.0 {
            return Err("preallocation_threshold must be between 0.0 and 1.0".to_string());
        }

        if self.buffer_lifetime == 0 {
            return Err("buffer_lifetime must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Calculate optimal buffer size for given allocation size
    pub fn calculate_buffer_size(&self, requested_size: usize) -> usize {
        if requested_size < self.min_batch_size {
            return requested_size;
        }

        let optimal_size = (requested_size as f32 * self.growth_factor) as usize;

        // Round up to next power of 2 for better alignment
        optimal_size.next_power_of_two()
    }

    /// Check if size qualifies for batch optimization
    pub fn qualifies_for_batch(&self, size: usize) -> bool {
        size >= self.min_batch_size
    }

    /// Get recommended number of buffers based on workload
    pub fn recommended_buffer_count(&self, expected_concurrent_ops: usize) -> usize {
        let recommended = (expected_concurrent_ops as f32 * 1.5) as usize;
        recommended.min(self.max_batch_buffers).max(1)
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum pool size in bytes
    pub max_pool_size: usize,
    /// Minimum block size to pool
    pub min_block_size: usize,
    /// Enable memory defragmentation
    pub enable_defrag: bool,
    /// Large batch configuration
    pub large_batch_config: LargeBatchConfig,
    /// Pool growth strategy
    pub growth_strategy: PoolGrowthStrategy,
    /// Memory alignment requirement
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            min_block_size: 256,
            enable_defrag: true,
            large_batch_config: LargeBatchConfig::default(),
            growth_strategy: PoolGrowthStrategy::Exponential,
            alignment: 256,
        }
    }
}

/// Pool growth strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolGrowthStrategy {
    /// Fixed size increments
    Fixed(usize),
    /// Linear growth
    Linear(f32),
    /// Exponential growth
    Exponential,
    /// Adaptive growth based on usage patterns
    Adaptive,
}

impl Default for PoolGrowthStrategy {
    fn default() -> Self {
        PoolGrowthStrategy::Exponential
    }
}

impl MemoryPoolConfig {
    /// Create new configuration
    pub fn new(max_pool_size: usize, min_block_size: usize) -> Self {
        Self {
            max_pool_size,
            min_block_size,
            enable_defrag: true,
            large_batch_config: LargeBatchConfig::default(),
            growth_strategy: PoolGrowthStrategy::Exponential,
            alignment: 256,
        }
    }

    /// Create configuration for high-performance workloads
    pub fn for_high_performance() -> Self {
        Self {
            max_pool_size: 8 * 1024 * 1024 * 1024, // 8GB
            min_block_size: 1024,
            enable_defrag: false, // Disable for performance
            large_batch_config: LargeBatchConfig::for_training(),
            growth_strategy: PoolGrowthStrategy::Exponential,
            alignment: 512,
        }
    }

    /// Create configuration for memory-efficient workloads
    pub fn for_memory_efficiency() -> Self {
        Self {
            max_pool_size: 512 * 1024 * 1024, // 512MB
            min_block_size: 128,
            enable_defrag: true,
            large_batch_config: LargeBatchConfig::for_low_memory(),
            growth_strategy: PoolGrowthStrategy::Linear(1.2),
            alignment: 256,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_pool_size == 0 {
            return Err("max_pool_size must be greater than 0".to_string());
        }

        if self.min_block_size == 0 {
            return Err("min_block_size must be greater than 0".to_string());
        }

        if !self.alignment.is_power_of_two() {
            return Err("alignment must be a power of 2".to_string());
        }

        if self.min_block_size < self.alignment {
            return Err("min_block_size must be at least as large as alignment".to_string());
        }

        self.large_batch_config.validate()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_batch_config_defaults() {
        let config = LargeBatchConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.min_batch_size, 1024 * 1024);
        assert_eq!(config.max_batch_buffers, 16);
    }

    #[test]
    fn test_large_batch_config_validation() {
        let mut config = LargeBatchConfig::default();

        // Test invalid min_batch_size
        config.min_batch_size = 0;
        assert!(config.validate().is_err());

        // Test invalid growth_factor
        config.min_batch_size = 1024;
        config.growth_factor = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_buffer_size_calculation() {
        let config = LargeBatchConfig::default();

        let size = config.calculate_buffer_size(1024 * 1024);
        assert!(size >= 1024 * 1024);

        // Should be power of 2
        assert_eq!(size & (size - 1), 0);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default();
        assert!(config.validate().is_ok());

        let high_perf = MemoryPoolConfig::for_high_performance();
        assert!(high_perf.validate().is_ok());
        assert!(!high_perf.enable_defrag); // Should be disabled for performance
    }
}