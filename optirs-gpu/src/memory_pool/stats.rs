// Memory statistics and metrics for GPU memory pool

use std::time::{Duration, Instant};

/// Basic memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocations requested
    pub total_allocations: usize,
    /// Total deallocations performed
    pub total_deallocations: usize,
    /// Current number of active allocations
    pub active_allocations: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Total bytes deallocated
    pub total_bytes_deallocated: usize,
    /// Current bytes in use
    pub current_bytes_used: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Cache hits (reused existing blocks)
    pub cache_hits: usize,
    /// Cache misses (had to allocate new blocks)
    pub cache_misses: usize,
    /// Number of defragmentation operations
    pub defragmentation_count: usize,
    /// Number of pool resize operations
    pub pool_resize_count: usize,
    /// Total allocation time (microseconds)
    pub total_allocation_time_us: u64,
    /// Total deallocation time (microseconds)
    pub total_deallocation_time_us: u64,
}

impl MemoryStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record successful allocation
    pub fn record_allocation(&mut self, size: usize, cache_hit: bool, duration: Duration) {
        self.total_allocations += 1;
        self.active_allocations += 1;
        self.total_bytes_allocated += size;
        self.current_bytes_used += size;
        self.total_allocation_time_us += duration.as_micros() as u64;

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        if self.current_bytes_used > self.peak_memory_usage {
            self.peak_memory_usage = self.current_bytes_used;
        }
    }

    /// Record deallocation
    pub fn record_deallocation(&mut self, size: usize, duration: Duration) {
        self.total_deallocations += 1;
        if self.active_allocations > 0 {
            self.active_allocations -= 1;
        }
        self.total_bytes_deallocated += size;
        if self.current_bytes_used >= size {
            self.current_bytes_used -= size;
        }
        self.total_deallocation_time_us += duration.as_micros() as u64;
    }

    /// Record defragmentation operation
    pub fn record_defragmentation(&mut self) {
        self.defragmentation_count += 1;
    }

    /// Record pool resize operation
    pub fn record_pool_resize(&mut self) {
        self.pool_resize_count += 1;
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f32 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total_requests as f32
    }

    /// Get average allocation time in microseconds
    pub fn average_allocation_time_us(&self) -> f64 {
        if self.total_allocations == 0 {
            return 0.0;
        }
        self.total_allocation_time_us as f64 / self.total_allocations as f64
    }

    /// Get average deallocation time in microseconds
    pub fn average_deallocation_time_us(&self) -> f64 {
        if self.total_deallocations == 0 {
            return 0.0;
        }
        self.total_deallocation_time_us as f64 / self.total_deallocations as f64
    }

    /// Get memory utilization ratio
    pub fn memory_utilization(&self) -> f32 {
        if self.peak_memory_usage == 0 {
            return 0.0;
        }
        self.current_bytes_used as f32 / self.peak_memory_usage as f32
    }

    /// Get allocation efficiency (successful allocations / total requests)
    pub fn allocation_efficiency(&self) -> f32 {
        let total_requests = self.total_allocations;
        if total_requests == 0 {
            return 1.0;
        }
        // Assuming all recorded allocations were successful
        1.0
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get total allocation throughput (allocations per second)
    pub fn allocation_throughput(&self, elapsed_time: Duration) -> f64 {
        let elapsed_secs = elapsed_time.as_secs_f64();
        if elapsed_secs == 0.0 {
            return 0.0;
        }
        self.total_allocations as f64 / elapsed_secs
    }

    /// Get total bandwidth (bytes per second)
    pub fn bandwidth(&self, elapsed_time: Duration) -> f64 {
        let elapsed_secs = elapsed_time.as_secs_f64();
        if elapsed_secs == 0.0 {
            return 0.0;
        }
        self.total_bytes_allocated as f64 / elapsed_secs
    }
}

/// Detailed memory statistics including fragmentation and utilization
#[derive(Debug, Clone)]
pub struct DetailedMemoryStats {
    /// Basic memory statistics
    pub basic_stats: MemoryStats,
    /// Memory fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,
    /// Memory utilization (0.0-1.0)
    pub utilization: f32,
    /// Number of free blocks
    pub free_block_count: usize,
    /// Total number of blocks
    pub total_block_count: usize,
    /// Number of batch buffers
    pub batch_buffer_count: usize,
    /// Number of active batch buffers
    pub active_batch_buffers: usize,
    /// Current memory pressure level
    pub current_pressure: f32,
    /// Maximum pool size
    pub max_pool_size: usize,
    /// Largest free block size
    pub largest_free_block: usize,
    /// Average free block size
    pub average_free_block_size: f32,
    /// Memory overhead (internal structures)
    pub memory_overhead: usize,
}

impl DetailedMemoryStats {
    /// Create new detailed stats from basic stats
    pub fn new(basic_stats: MemoryStats) -> Self {
        Self {
            basic_stats,
            fragmentation_ratio: 0.0,
            utilization: 0.0,
            free_block_count: 0,
            total_block_count: 0,
            batch_buffer_count: 0,
            active_batch_buffers: 0,
            current_pressure: 0.0,
            max_pool_size: 0,
            largest_free_block: 0,
            average_free_block_size: 0.0,
            memory_overhead: 0,
        }
    }

    /// Update fragmentation statistics
    pub fn update_fragmentation(&mut self, free_blocks: &[(usize, usize)]) {
        if free_blocks.is_empty() {
            self.fragmentation_ratio = 0.0;
            self.largest_free_block = 0;
            self.average_free_block_size = 0.0;
            return;
        }

        self.free_block_count = free_blocks.len();
        self.largest_free_block = free_blocks.iter().map(|(size, _)| *size).max().unwrap_or(0);

        let total_free: usize = free_blocks.iter().map(|(size, count)| size * count).sum();
        self.average_free_block_size = total_free as f32 / self.free_block_count as f32;

        // Calculate fragmentation as 1 - (largest_block / total_free)
        if total_free > 0 {
            self.fragmentation_ratio = 1.0 - (self.largest_free_block as f32 / total_free as f32);
        } else {
            self.fragmentation_ratio = 0.0;
        }
    }

    /// Update utilization statistics
    pub fn update_utilization(&mut self, used_memory: usize, total_memory: usize) {
        if total_memory > 0 {
            self.utilization = used_memory as f32 / total_memory as f32;
        } else {
            self.utilization = 0.0;
        }
    }

    /// Get memory efficiency score (combination of utilization and low fragmentation)
    pub fn efficiency_score(&self) -> f32 {
        let fragmentation_penalty = self.fragmentation_ratio * 0.5;
        let efficiency = self.utilization - fragmentation_penalty;
        efficiency.max(0.0).min(1.0)
    }

    /// Check if memory is under pressure
    pub fn is_under_pressure(&self, threshold: f32) -> bool {
        self.current_pressure > threshold
    }

    /// Get memory health score (0.0 = poor, 1.0 = excellent)
    pub fn health_score(&self) -> f32 {
        let utilization_score = if self.utilization > 0.8 {
            // High utilization is good, but too high (>95%) might cause issues
            if self.utilization > 0.95 {
                1.0 - (self.utilization - 0.95) * 10.0
            } else {
                self.utilization
            }
        } else {
            self.utilization * 0.8 // Lower utilization is less optimal
        };

        let fragmentation_score = 1.0 - self.fragmentation_ratio;
        let pressure_score = 1.0 - self.current_pressure;

        // Weighted average
        (utilization_score * 0.4 + fragmentation_score * 0.4 + pressure_score * 0.2)
            .max(0.0)
            .min(1.0)
    }

    /// Format detailed report
    pub fn format_report(&self) -> String {
        format!(
            "Memory Pool Statistics Report\n\
             ===============================\n\
             Basic Stats:\n\
             - Total Allocations: {}\n\
             - Active Allocations: {}\n\
             - Current Memory Used: {} bytes\n\
             - Peak Memory Usage: {} bytes\n\
             - Cache Hit Ratio: {:.2}%\n\
             \n\
             Performance:\n\
             - Avg Allocation Time: {:.2} μs\n\
             - Avg Deallocation Time: {:.2} μs\n\
             \n\
             Memory Health:\n\
             - Utilization: {:.2}%\n\
             - Fragmentation: {:.2}%\n\
             - Memory Pressure: {:.2}%\n\
             - Health Score: {:.2}/1.0\n\
             \n\
             Block Management:\n\
             - Free Blocks: {}\n\
             - Total Blocks: {}\n\
             - Largest Free Block: {} bytes\n\
             - Avg Free Block Size: {:.1} bytes\n\
             \n\
             Batch Buffers:\n\
             - Total Batch Buffers: {}\n\
             - Active Batch Buffers: {}\n",
            self.basic_stats.total_allocations,
            self.basic_stats.active_allocations,
            self.basic_stats.current_bytes_used,
            self.basic_stats.peak_memory_usage,
            self.basic_stats.cache_hit_ratio() * 100.0,
            self.basic_stats.average_allocation_time_us(),
            self.basic_stats.average_deallocation_time_us(),
            self.utilization * 100.0,
            self.fragmentation_ratio * 100.0,
            self.current_pressure * 100.0,
            self.health_score(),
            self.free_block_count,
            self.total_block_count,
            self.largest_free_block,
            self.average_free_block_size,
            self.batch_buffer_count,
            self.active_batch_buffers
        )
    }
}

/// Performance metrics tracker
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Start time for tracking session
    start_time: Instant,
    /// Allocation latency samples
    allocation_latencies: Vec<Duration>,
    /// Deallocation latency samples
    deallocation_latencies: Vec<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new(max_samples: usize) -> Self {
        Self {
            start_time: Instant::now(),
            allocation_latencies: Vec::with_capacity(max_samples),
            deallocation_latencies: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record allocation latency
    pub fn record_allocation_latency(&mut self, latency: Duration) {
        if self.allocation_latencies.len() >= self.max_samples {
            self.allocation_latencies.remove(0);
        }
        self.allocation_latencies.push(latency);
    }

    /// Record deallocation latency
    pub fn record_deallocation_latency(&mut self, latency: Duration) {
        if self.deallocation_latencies.len() >= self.max_samples {
            self.deallocation_latencies.remove(0);
        }
        self.deallocation_latencies.push(latency);
    }

    /// Get percentile latency for allocations
    pub fn allocation_percentile(&self, percentile: f32) -> Duration {
        self.calculate_percentile(&self.allocation_latencies, percentile)
    }

    /// Get percentile latency for deallocations
    pub fn deallocation_percentile(&self, percentile: f32) -> Duration {
        self.calculate_percentile(&self.deallocation_latencies, percentile)
    }

    /// Calculate percentile from sorted samples
    fn calculate_percentile(&self, samples: &[Duration], percentile: f32) -> Duration {
        if samples.is_empty() {
            return Duration::from_nanos(0);
        }

        let mut sorted = samples.to_vec();
        sorted.sort();

        let index = ((samples.len() - 1) as f32 * percentile / 100.0) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Get total tracking duration
    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.allocation_latencies.clear();
        self.deallocation_latencies.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_basic() {
        let mut stats = MemoryStats::new();

        // Record allocation
        stats.record_allocation(1024, false, Duration::from_micros(100));

        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.current_bytes_used, 1024);
        assert_eq!(stats.cache_misses, 1);

        // Record deallocation
        stats.record_deallocation(1024, Duration::from_micros(50));

        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.current_bytes_used, 0);
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut stats = MemoryStats::new();

        stats.record_allocation(1024, true, Duration::from_micros(50));
        stats.record_allocation(2048, false, Duration::from_micros(100));

        assert_eq!(stats.cache_hit_ratio(), 0.5);
    }

    #[test]
    fn test_detailed_stats() {
        let basic_stats = MemoryStats::new();
        let mut detailed = DetailedMemoryStats::new(basic_stats);

        detailed.update_utilization(1024, 2048);
        assert_eq!(detailed.utilization, 0.5);

        detailed.update_fragmentation(&[(512, 2), (256, 1)]);
        assert_eq!(detailed.free_block_count, 2);
        assert!(detailed.fragmentation_ratio > 0.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(100);

        tracker.record_allocation_latency(Duration::from_micros(100));
        tracker.record_allocation_latency(Duration::from_micros(200));

        let p50 = tracker.allocation_percentile(50.0);
        assert!(p50.as_micros() >= 100);
    }
}