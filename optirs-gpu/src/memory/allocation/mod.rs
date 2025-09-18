// GPU memory allocation strategies and algorithms
//
// This module provides various memory allocation strategies optimized for
// different GPU workload patterns and memory usage scenarios.

#[allow(dead_code)]
pub mod arena_allocator;
pub mod buddy_allocator;
pub mod slab_allocator;
pub mod strategies;

// Re-export main types for convenience
pub use strategies::{
    AdaptiveConfig, AllocationEvent, AllocationPattern, AllocationStats, AllocationStrategy,
    AllocationStrategyManager, HybridConfig, MLConfig, MLFeatures, MLPrediction, MemoryBlock,
};

pub use buddy_allocator::{
    AllocationInfo, BuddyAllocator, BuddyBlock, BuddyConfig, BuddyError, BuddyStats,
    FreeBlockStats, MemoryUsage, ThreadSafeBuddyAllocator,
};

pub use slab_allocator::{
    CacheConfig, CacheInfo, MemoryPool, MemoryPoolUsage, Slab, SlabAllocator, SlabAllocatorStats,
    SlabCache, SlabConfig, SlabError, ThreadSafeSlabAllocator,
};

pub use arena_allocator::{
    ArenaAllocator, ArenaConfig, ArenaError, ArenaStats, ArenaUsage, CheckpointHandle,
    ExternalAllocator, GrowingArena, MemoryLayout, MemoryRegion, RingArena, RingConfig, RingUsage,
    ThreadSafeArena,
};

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Unified allocator interface that can use different allocation strategies
pub struct UnifiedAllocator {
    /// Strategy manager for general allocations
    strategy_manager: AllocationStrategyManager,
    /// Buddy allocator for power-of-2 allocations
    buddy_allocator: Option<BuddyAllocator>,
    /// Slab allocator for fixed-size objects
    slab_allocator: Option<SlabAllocator>,
    /// Arena allocator for temporary allocations
    arena_allocator: Option<ArenaAllocator>,
    /// Configuration
    config: UnifiedConfig,
    /// Statistics
    stats: UnifiedStats,
    /// Allocation routing table
    routing_table: AllocationRouter,
}

/// Configuration for unified allocator
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// Default allocation strategy
    pub default_strategy: AllocationStrategy,
    /// Enable buddy allocator
    pub enable_buddy: bool,
    /// Enable slab allocator
    pub enable_slab: bool,
    /// Enable arena allocator
    pub enable_arena: bool,
    /// Size threshold for buddy allocator
    pub buddy_threshold: usize,
    /// Size threshold for slab allocator
    pub slab_threshold: usize,
    /// Size threshold for arena allocator
    pub arena_threshold: usize,
    /// Enable automatic routing optimization
    pub enable_auto_routing: bool,
    /// Statistics collection interval
    pub stats_interval: std::time::Duration,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            default_strategy: AllocationStrategy::Adaptive,
            enable_buddy: true,
            enable_slab: true,
            enable_arena: true,
            buddy_threshold: 1024,
            slab_threshold: 4096,
            arena_threshold: 64 * 1024,
            enable_auto_routing: true,
            stats_interval: std::time::Duration::from_secs(1),
        }
    }
}

/// Unified allocator statistics
#[derive(Debug, Clone, Default)]
pub struct UnifiedStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub strategy_allocations: HashMap<AllocationStrategy, u64>,
    pub buddy_allocations: u64,
    pub slab_allocations: u64,
    pub arena_allocations: u64,
    pub routing_decisions: u64,
    pub routing_cache_hits: u64,
    pub average_allocation_time_ns: f64,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
}

/// Allocation routing logic
pub struct AllocationRouter {
    /// Size-based routing rules
    size_routes: Vec<SizeRoute>,
    /// Pattern-based routing cache
    pattern_cache: HashMap<AllocationPattern, AllocatorType>,
    /// Performance history for routing decisions
    performance_history: HashMap<AllocatorType, PerformanceMetrics>,
    /// Configuration
    config: RouterConfig,
}

/// Size-based routing rule
#[derive(Debug, Clone)]
pub struct SizeRoute {
    pub min_size: usize,
    pub max_size: Option<usize>,
    pub preferred_allocator: AllocatorType,
    pub fallback_allocator: Option<AllocatorType>,
}

/// Allocator type identification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocatorType {
    Strategy(AllocationStrategy),
    Buddy,
    Slab,
    Arena,
}

/// Performance metrics for routing decisions
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub average_latency_ns: f64,
    pub success_rate: f64,
    pub fragmentation_ratio: f64,
    pub cache_hit_rate: f64,
    pub memory_efficiency: f64,
}

/// Router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub enable_performance_tracking: bool,
    pub cache_size: usize,
    pub adaptation_threshold: f64,
    pub performance_window: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enable_performance_tracking: true,
            cache_size: 1000,
            adaptation_threshold: 0.1,
            performance_window: 100,
        }
    }
}

impl AllocationRouter {
    pub fn new(config: RouterConfig) -> Self {
        let size_routes = vec![
            SizeRoute {
                min_size: 0,
                max_size: Some(256),
                preferred_allocator: AllocatorType::Slab,
                fallback_allocator: Some(AllocatorType::Strategy(AllocationStrategy::FirstFit)),
            },
            SizeRoute {
                min_size: 257,
                max_size: Some(4096),
                preferred_allocator: AllocatorType::Strategy(AllocationStrategy::BestFit),
                fallback_allocator: Some(AllocatorType::Buddy),
            },
            SizeRoute {
                min_size: 4097,
                max_size: Some(64 * 1024),
                preferred_allocator: AllocatorType::Buddy,
                fallback_allocator: Some(AllocatorType::Strategy(AllocationStrategy::BestFit)),
            },
            SizeRoute {
                min_size: 64 * 1024 + 1,
                max_size: None,
                preferred_allocator: AllocatorType::Arena,
                fallback_allocator: Some(AllocatorType::Strategy(AllocationStrategy::WorstFit)),
            },
        ];

        Self {
            size_routes,
            pattern_cache: HashMap::new(),
            performance_history: HashMap::new(),
            config,
        }
    }

    /// Route allocation request to appropriate allocator
    pub fn route_allocation(
        &mut self,
        size: usize,
        pattern: Option<AllocationPattern>,
    ) -> AllocatorType {
        // Check pattern cache first
        if let Some(pattern) = pattern {
            if let Some(&allocator_type) = self.pattern_cache.get(&pattern) {
                return allocator_type;
            }
        }

        // Use size-based routing
        for route in &self.size_routes {
            if size >= route.min_size && route.max_size.map_or(true, |max| size <= max) {
                // Check performance if tracking is enabled
                if self.config.enable_performance_tracking {
                    let preferred_perf = self
                        .performance_history
                        .get(&route.preferred_allocator)
                        .cloned()
                        .unwrap_or_default();

                    if let Some(fallback) = &route.fallback_allocator {
                        let fallback_perf = self
                            .performance_history
                            .get(fallback)
                            .cloned()
                            .unwrap_or_default();

                        // Choose based on performance
                        if fallback_perf.average_latency_ns > 0.0
                            && preferred_perf.average_latency_ns > 0.0
                        {
                            let perf_ratio = fallback_perf.average_latency_ns
                                / preferred_perf.average_latency_ns;
                            if perf_ratio < 1.0 - self.config.adaptation_threshold {
                                return fallback.clone();
                            }
                        }
                    }
                }

                return route.preferred_allocator.clone();
            }
        }

        // Default fallback
        AllocatorType::Strategy(AllocationStrategy::BestFit)
    }

    /// Update performance metrics for an allocator
    pub fn update_performance(
        &mut self,
        allocator_type: AllocatorType,
        metrics: PerformanceMetrics,
    ) {
        self.performance_history.insert(allocator_type, metrics);
    }

    /// Cache pattern-based routing decision
    pub fn cache_pattern_route(
        &mut self,
        pattern: AllocationPattern,
        allocator_type: AllocatorType,
    ) {
        if self.pattern_cache.len() >= self.config.cache_size {
            // Remove oldest entry (simplified - could use LRU)
            if let Some(key) = self.pattern_cache.keys().next().cloned() {
                self.pattern_cache.remove(&key);
            }
        }
        self.pattern_cache.insert(pattern, allocator_type);
    }
}

impl UnifiedAllocator {
    /// Create a new unified allocator
    pub fn new(
        base_ptr: NonNull<u8>,
        total_size: usize,
        config: UnifiedConfig,
    ) -> Result<Self, AllocationError> {
        let strategy_manager = AllocationStrategyManager::new(config.default_strategy.clone());

        let buddy_allocator = if config.enable_buddy {
            let buddy_config = BuddyConfig::default();
            let buddy_size = total_size / 4; // Allocate 1/4 of memory to buddy allocator
            let buddy_ptr = base_ptr;
            Some(BuddyAllocator::new(buddy_ptr, buddy_size, buddy_config)?)
        } else {
            None
        };

        let slab_allocator = if config.enable_slab {
            let slab_config = SlabConfig::default();
            let slab_size = total_size / 4; // Allocate 1/4 of memory to slab allocator
            let slab_ptr = unsafe { NonNull::new_unchecked(base_ptr.as_ptr().add(total_size / 4)) };
            Some(SlabAllocator::new(slab_ptr, slab_size, slab_config))
        } else {
            None
        };

        let arena_allocator = if config.enable_arena {
            let arena_config = ArenaConfig::default();
            let arena_size = total_size / 4; // Allocate 1/4 of memory to arena allocator
            let arena_ptr =
                unsafe { NonNull::new_unchecked(base_ptr.as_ptr().add(total_size / 2)) };
            Some(ArenaAllocator::new(arena_ptr, arena_size, arena_config)?)
        } else {
            None
        };

        let routing_table = AllocationRouter::new(RouterConfig::default());

        Ok(Self {
            strategy_manager,
            buddy_allocator,
            slab_allocator,
            arena_allocator,
            config,
            stats: UnifiedStats::default(),
            routing_table,
        })
    }

    /// Allocate memory using the unified interface
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, AllocationError> {
        let start_time = Instant::now();
        self.stats.total_allocations += 1;

        // Analyze allocation pattern
        let pattern = self.strategy_manager.analyze_allocation_patterns();

        // Route to appropriate allocator
        let allocator_type = self.routing_table.route_allocation(size, Some(pattern));

        let result = match allocator_type {
            AllocatorType::Strategy(strategy) => {
                self.strategy_manager.set_strategy(strategy.clone());
                self.strategy_manager.find_free_block(size).ok_or_else(|| {
                    AllocationError::OutOfMemory("Strategy allocator failed".to_string())
                })
            }
            AllocatorType::Buddy => {
                if let Some(ref mut buddy) = self.buddy_allocator {
                    buddy
                        .allocate(size)
                        .map_err(|e| AllocationError::BuddyError(e))
                } else {
                    Err(AllocationError::AllocatorNotAvailable(
                        "Buddy allocator not enabled".to_string(),
                    ))
                }
            }
            AllocatorType::Slab => {
                if let Some(ref mut slab) = self.slab_allocator {
                    slab.allocate(size)
                        .map_err(|e| AllocationError::SlabError(e))
                } else {
                    Err(AllocationError::AllocatorNotAvailable(
                        "Slab allocator not enabled".to_string(),
                    ))
                }
            }
            AllocatorType::Arena => {
                if let Some(ref mut arena) = self.arena_allocator {
                    arena
                        .allocate(size)
                        .map_err(|e| AllocationError::ArenaError(e))
                } else {
                    Err(AllocationError::AllocatorNotAvailable(
                        "Arena allocator not enabled".to_string(),
                    ))
                }
            }
        };

        let allocation_time = start_time.elapsed().as_nanos() as f64;

        match &result {
            Ok(_) => {
                self.stats.bytes_allocated += size as u64;
                self.stats.current_memory_usage += size;
                if self.stats.current_memory_usage > self.stats.peak_memory_usage {
                    self.stats.peak_memory_usage = self.stats.current_memory_usage;
                }

                // Update strategy-specific stats
                match allocator_type {
                    AllocatorType::Strategy(strategy) => {
                        *self.stats.strategy_allocations.entry(strategy).or_insert(0) += 1;
                    }
                    AllocatorType::Buddy => self.stats.buddy_allocations += 1,
                    AllocatorType::Slab => self.stats.slab_allocations += 1,
                    AllocatorType::Arena => self.stats.arena_allocations += 1,
                }

                // Update performance metrics
                let metrics = PerformanceMetrics {
                    average_latency_ns: allocation_time,
                    success_rate: 1.0,
                    fragmentation_ratio: 0.0, // Would need to calculate from allocator
                    cache_hit_rate: 0.0,      // Would need to get from allocator
                    memory_efficiency: 1.0,   // Would need to calculate
                };
                self.routing_table
                    .update_performance(allocator_type, metrics);
            }
            Err(_) => {
                // Update failure metrics
                let metrics = PerformanceMetrics {
                    average_latency_ns: allocation_time,
                    success_rate: 0.0,
                    ..Default::default()
                };
                self.routing_table
                    .update_performance(allocator_type, metrics);
            }
        }

        // Update average allocation time
        let total_time = self.stats.average_allocation_time_ns
            * (self.stats.total_allocations - 1) as f64
            + allocation_time;
        self.stats.average_allocation_time_ns = total_time / self.stats.total_allocations as f64;

        result
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) -> Result<(), AllocationError> {
        self.stats.total_deallocations += 1;
        self.stats.bytes_deallocated += size as u64;
        self.stats.current_memory_usage = self.stats.current_memory_usage.saturating_sub(size);

        // Try each allocator to find which one owns this pointer
        if let Some(ref mut buddy) = self.buddy_allocator {
            if let Ok(()) = buddy.deallocate(ptr) {
                return Ok(());
            }
        }

        if let Some(ref mut slab) = self.slab_allocator {
            if let Ok(()) = slab.deallocate(ptr, size) {
                return Ok(());
            }
        }

        if let Some(ref mut arena) = self.arena_allocator {
            if arena.contains_pointer(ptr) {
                // Arena allocator typically doesn't support individual deallocation
                return Ok(());
            }
        }

        Err(AllocationError::InvalidPointer(
            "Pointer not found in any allocator".to_string(),
        ))
    }

    /// Get unified statistics
    pub fn get_stats(&self) -> &UnifiedStats {
        &self.stats
    }

    /// Get detailed allocator information
    pub fn get_detailed_info(&self) -> DetailedAllocatorInfo {
        let mut info = DetailedAllocatorInfo {
            strategy_info: Some(self.strategy_manager.get_stats().clone()),
            buddy_info: None,
            slab_info: None,
            arena_info: None,
            unified_stats: self.stats.clone(),
        };

        if let Some(ref buddy) = self.buddy_allocator {
            info.buddy_info = Some(buddy.get_stats().clone());
        }

        if let Some(ref slab) = self.slab_allocator {
            info.slab_info = Some(slab.get_stats());
        }

        if let Some(ref arena) = self.arena_allocator {
            info.arena_info = Some(arena.get_stats().clone());
        }

        info
    }

    /// Reset specific allocator
    pub fn reset_allocator(
        &mut self,
        allocator_type: AllocatorType,
    ) -> Result<(), AllocationError> {
        match allocator_type {
            AllocatorType::Strategy(_) => {
                self.strategy_manager.clear_history();
            }
            AllocatorType::Buddy => {
                if let Some(ref mut buddy) = self.buddy_allocator {
                    buddy.reset();
                } else {
                    return Err(AllocationError::AllocatorNotAvailable(
                        "Buddy allocator not enabled".to_string(),
                    ));
                }
            }
            AllocatorType::Slab => {
                return Err(AllocationError::UnsupportedOperation(
                    "Slab allocator reset not supported".to_string(),
                ));
            }
            AllocatorType::Arena => {
                if let Some(ref mut arena) = self.arena_allocator {
                    arena.reset();
                } else {
                    return Err(AllocationError::AllocatorNotAvailable(
                        "Arena allocator not enabled".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Force garbage collection on applicable allocators
    pub fn garbage_collect(&mut self) -> GarbageCollectionResult {
        let mut result = GarbageCollectionResult::default();

        if let Some(ref mut slab) = self.slab_allocator {
            result.slab_reclaimed = slab.reclaim_memory();
        }

        if let Some(ref mut buddy) = self.buddy_allocator {
            result.buddy_defragmented = buddy.defragment();
        }

        result
    }
}

/// Detailed information about all allocators
#[derive(Debug, Clone)]
pub struct DetailedAllocatorInfo {
    pub strategy_info: Option<AllocationStats>,
    pub buddy_info: Option<BuddyStats>,
    pub slab_info: Option<SlabAllocatorStats>,
    pub arena_info: Option<ArenaStats>,
    pub unified_stats: UnifiedStats,
}

/// Result of garbage collection operations
#[derive(Debug, Clone, Default)]
pub struct GarbageCollectionResult {
    pub slab_reclaimed: usize,
    pub buddy_defragmented: usize,
    pub arena_reset: bool,
    pub total_bytes_freed: usize,
}

/// Unified allocation errors
#[derive(Debug, Clone)]
pub enum AllocationError {
    OutOfMemory(String),
    InvalidPointer(String),
    AllocatorNotAvailable(String),
    UnsupportedOperation(String),
    BuddyError(BuddyError),
    SlabError(SlabError),
    ArenaError(ArenaError),
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocationError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            AllocationError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            AllocationError::AllocatorNotAvailable(msg) => {
                write!(f, "Allocator not available: {}", msg)
            }
            AllocationError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            AllocationError::BuddyError(e) => write!(f, "Buddy allocator error: {}", e),
            AllocationError::SlabError(e) => write!(f, "Slab allocator error: {}", e),
            AllocationError::ArenaError(e) => write!(f, "Arena allocator error: {}", e),
        }
    }
}

impl std::error::Error for AllocationError {}

impl From<BuddyError> for AllocationError {
    fn from(error: BuddyError) -> Self {
        AllocationError::BuddyError(error)
    }
}

impl From<SlabError> for AllocationError {
    fn from(error: SlabError) -> Self {
        AllocationError::SlabError(error)
    }
}

impl From<ArenaError> for AllocationError {
    fn from(error: ArenaError) -> Self {
        AllocationError::ArenaError(error)
    }
}

/// Thread-safe unified allocator wrapper
pub struct ThreadSafeUnifiedAllocator {
    allocator: Arc<Mutex<UnifiedAllocator>>,
}

impl ThreadSafeUnifiedAllocator {
    pub fn new(
        base_ptr: NonNull<u8>,
        total_size: usize,
        config: UnifiedConfig,
    ) -> Result<Self, AllocationError> {
        let allocator = UnifiedAllocator::new(base_ptr, total_size, config)?;
        Ok(Self {
            allocator: Arc::new(Mutex::new(allocator)),
        })
    }

    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>, AllocationError> {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.allocate(size)
    }

    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<(), AllocationError> {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.deallocate(ptr, size)
    }

    pub fn get_stats(&self) -> UnifiedStats {
        let allocator = self.allocator.lock().unwrap();
        allocator.get_stats().clone()
    }

    pub fn garbage_collect(&self) -> GarbageCollectionResult {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.garbage_collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_allocator_creation() {
        let size = 1024 * 1024; // 1MB
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = UnifiedConfig::default();
        let allocator = UnifiedAllocator::new(ptr, size, config);
        assert!(allocator.is_ok());
    }

    #[test]
    fn test_unified_allocation() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = UnifiedConfig::default();
        let mut allocator = UnifiedAllocator::new(ptr, size, config).unwrap();

        // Test different sizes to trigger different allocators
        let small_alloc = allocator.allocate(100); // Should use slab
        assert!(small_alloc.is_ok());

        let medium_alloc = allocator.allocate(2048); // Should use buddy
        assert!(medium_alloc.is_ok());

        let large_alloc = allocator.allocate(128 * 1024); // Should use arena
        assert!(large_alloc.is_ok());

        let stats = allocator.get_stats();
        assert_eq!(stats.total_allocations, 3);
    }

    #[test]
    fn test_allocation_routing() {
        let config = RouterConfig::default();
        let mut router = AllocationRouter::new(config);

        let small_route = router.route_allocation(100, None);
        assert_eq!(small_route, AllocatorType::Slab);

        let medium_route = router.route_allocation(2048, None);
        assert_eq!(
            medium_route,
            AllocatorType::Strategy(AllocationStrategy::BestFit)
        );

        let large_route = router.route_allocation(128 * 1024, None);
        assert_eq!(large_route, AllocatorType::Arena);
    }

    #[test]
    fn test_thread_safe_unified_allocator() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = UnifiedConfig::default();
        let allocator = ThreadSafeUnifiedAllocator::new(ptr, size, config).unwrap();

        let alloc_result = allocator.allocate(1024);
        assert!(alloc_result.is_ok());

        let stats = allocator.get_stats();
        assert!(stats.total_allocations > 0);
    }
}
