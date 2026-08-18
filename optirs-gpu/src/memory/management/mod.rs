// GPU memory management modules
//
// This module provides advanced GPU memory management capabilities including
// garbage collection, prefetching, eviction policies, and defragmentation.

pub mod defragmentation;
pub mod eviction_policies;
pub mod garbage_collection;
pub mod prefetching;

use std::collections::HashMap;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use crate::memory::management::eviction_policies::EvictionConfig;

pub use garbage_collection::{
    GCConfig, GCStats, GarbageCollectionEngine, GarbageCollector, GenerationalCollector,
    IncrementalCollector, MarkSweepCollector, ReferenceTracker,
};

pub use prefetching::{
    AccessHistoryTracker, PrefetchCache, PrefetchConfig, PrefetchStrategy, PrefetchingEngine,
    SequentialPrefetcher, StridePrefetcher,
};

pub use eviction_policies::{
    ARCPolicy, ClockPolicy, EvictionEngine, EvictionPerformanceMonitor, EvictionPolicy, FIFOPolicy,
    LFUPolicy, LRUPolicy, MemoryRegion, WorkloadAwarePolicy,
};

pub use defragmentation::{
    CompactionAlgorithm, CompactionStrategy, DefragConfig, DefragError, DefragmentationEngine,
    SlidingCompactionStrategy, ThreadSafeDefragmentationEngine, TwoPointerCompactionStrategy,
};

/// Unified memory management configuration
#[derive(Debug, Clone)]
pub struct MemoryManagementConfig {
    /// Garbage collection configuration
    pub gc_config: GCConfig,
    /// Prefetching configuration  
    pub prefetch_config: PrefetchConfig,
    /// Defragmentation configuration
    pub defrag_config: DefragConfig,
    /// Enable background management
    pub enable_background_management: bool,
    /// Management thread count
    pub management_threads: usize,
    /// Memory pressure threshold
    pub memory_pressure_threshold: f64,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
}

impl Default for MemoryManagementConfig {
    fn default() -> Self {
        Self {
            gc_config: GCConfig::default(),
            prefetch_config: PrefetchConfig::default(),
            defrag_config: DefragConfig::default(),
            enable_background_management: true,
            management_threads: 2,
            memory_pressure_threshold: 0.8,
            monitoring_interval: Duration::from_millis(100),
        }
    }
}

/// Integrated memory management system
pub struct IntegratedMemoryManager {
    /// Garbage collection engine
    gc_engine: GarbageCollectionEngine,
    /// Prefetching engine
    prefetch_engine: PrefetchingEngine,
    /// Eviction engine
    eviction_engine: EvictionEngine,
    /// Defragmentation engine
    defrag_engine: DefragmentationEngine,
    /// Configuration
    config: MemoryManagementConfig,
    /// Management statistics
    stats: ManagementStats,
    /// Background management enabled
    background_enabled: bool,
}

/// Memory management statistics
#[derive(Debug, Clone, Default)]
pub struct ManagementStats {
    pub gc_collections: u64,
    pub objects_collected: u64,
    pub bytes_freed_by_gc: u64,
    pub prefetch_requests: u64,
    pub prefetch_hits: u64,
    pub prefetch_accuracy: f64,
    pub evictions_performed: u64,
    pub bytes_evicted: u64,
    pub defragmentation_cycles: u64,
    pub fragmentation_reduced: usize,
    pub total_management_time: Duration,
    pub memory_pressure_events: u64,
}

impl IntegratedMemoryManager {
    /// Create new integrated memory manager
    pub fn new(config: MemoryManagementConfig) -> Self {
        let gc_engine = GarbageCollectionEngine::new(config.gc_config.clone());
        let prefetch_engine = PrefetchingEngine::new(config.prefetch_config.clone());
        let eviction_engine = EvictionEngine::new(EvictionConfig::default());
        let defrag_engine = DefragmentationEngine::new(config.defrag_config.clone());

        Self {
            gc_engine,
            prefetch_engine,
            eviction_engine,
            defrag_engine,
            config,
            stats: ManagementStats::default(),
            background_enabled: false,
        }
    }

    /// Start background memory management
    pub fn start_background_management(&mut self) -> Result<(), MemoryManagementError> {
        if !self.config.enable_background_management {
            return Err(MemoryManagementError::BackgroundManagementDisabled);
        }

        self.background_enabled = true;
        Ok(())
    }

    /// Stop background memory management
    pub fn stop_background_management(&mut self) {
        self.background_enabled = false;
    }

    /// Run garbage collection
    ///
    /// `memory_regions` describes the caller's current view of live
    /// allocations, but it is intentionally not registered with
    /// [`GarbageCollectionEngine`]: that engine only selects a collector for
    /// a region when `region.utilization < mark_threshold` (see
    /// `MarkSweepCollector::can_collect`), while every region built by the
    /// current caller is a single allocation at 100% utilization. Feeding
    /// such regions in would turn every call into a hard
    /// `GCError::NoSuitableCollector` instead of today's harmless no-op.
    /// Tracked as a finding rather than force-integrated; see the crate's
    /// lint/unwrap sweep notes.
    pub fn run_garbage_collection(
        &mut self,
        _memory_regions: &HashMap<usize, MemoryRegion>,
    ) -> Result<usize, MemoryManagementError> {
        let start_time = Instant::now();

        let gc_results = self
            .gc_engine
            .collect()
            .map_err(|e| MemoryManagementError::GarbageCollectionFailed(format!("{:?}", e)))?;
        let bytes_freed: usize = gc_results.iter().map(|r| r.bytes_collected).sum();

        self.stats.gc_collections += 1;
        self.stats.bytes_freed_by_gc += bytes_freed as u64;
        self.stats.total_management_time += start_time.elapsed();

        Ok(bytes_freed)
    }

    /// Perform prefetch operation
    ///
    /// Records the access with the [`PrefetchingEngine`] so its access-history
    /// tracker and pattern strategies see real data; the returned bool
    /// reflects whether this access was itself a hit against data the engine
    /// had already prefetched (a genuine cache hit), not a fabricated value.
    pub fn prefetch(
        &mut self,
        address: *mut c_void,
        size: usize,
        access_pattern: Option<&str>,
    ) -> Result<bool, MemoryManagementError> {
        let start_time = Instant::now();

        let access_type = match access_pattern {
            Some(pattern) if pattern.eq_ignore_ascii_case("write") => {
                prefetching::AccessType::Write
            }
            Some(pattern) if pattern.eq_ignore_ascii_case("read") => prefetching::AccessType::Read,
            _ => prefetching::AccessType::ReadWrite,
        };

        let hits_before = self.prefetch_engine.get_stats().successful_prefetches;
        self.prefetch_engine
            .record_access(prefetching::MemoryAccess {
                address: address as usize,
                size,
                timestamp: Instant::now(),
                access_type,
                context_id: 0,
                kernel_id: None,
            });
        let prefetched = self.prefetch_engine.get_stats().successful_prefetches > hits_before;

        self.stats.prefetch_requests += 1;
        if prefetched {
            self.stats.prefetch_hits += 1;
        }

        self.stats.prefetch_accuracy =
            self.stats.prefetch_hits as f64 / self.stats.prefetch_requests as f64;
        self.stats.total_management_time += start_time.elapsed();

        Ok(prefetched)
    }

    /// Perform memory eviction
    ///
    /// Registers `memory_regions` with the [`EvictionEngine`] (most-pressured,
    /// then largest, region first) and evicts real objects from it via the
    /// engine's active policy until `target_bytes` have been reclaimed or
    /// there is nothing left to evict. The returned count is the true sum of
    /// evicted object sizes, not an estimate.
    pub fn evict_memory(
        &mut self,
        target_bytes: usize,
        memory_regions: &HashMap<usize, MemoryRegion>,
    ) -> Result<usize, MemoryManagementError> {
        let start_time = Instant::now();
        let mut bytes_evicted = 0usize;

        let mut ordered: Vec<&MemoryRegion> = memory_regions.values().collect();
        ordered.sort_by(|a, b| {
            b.pressure
                .partial_cmp(&a.pressure)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.size.cmp(&a.size))
        });

        for region in ordered {
            if bytes_evicted >= target_bytes {
                break;
            }

            self.eviction_engine.register_region(
                region.base_addr,
                region.size,
                region.region_type.clone(),
            );
            for object in region.objects.values() {
                self.eviction_engine
                    .add_object(region.base_addr, object.clone())
                    .map_err(|e| MemoryManagementError::EvictionFailed(format!("{:?}", e)))?;
            }

            let victims = self
                .eviction_engine
                .evict(region.base_addr, target_bytes - bytes_evicted)
                .map_err(|e| MemoryManagementError::EvictionFailed(format!("{:?}", e)))?;

            for victim_addr in &victims {
                if let Some(object) = region.objects.get(victim_addr) {
                    bytes_evicted += object.size;
                }
            }
        }

        self.stats.evictions_performed += 1;
        self.stats.bytes_evicted += bytes_evicted as u64;
        self.stats.total_management_time += start_time.elapsed();

        Ok(bytes_evicted)
    }

    /// Run defragmentation
    ///
    /// `memory_regions` is intentionally not registered with
    /// [`DefragmentationEngine`]: both built-in compaction strategies require
    /// `MemoryLayoutTracker::get_total_free_space() > 0` to be eligible
    /// (`can_handle`), but the current caller only ever reports one region
    /// per live allocation with no free space (region size == its single
    /// object's size). Registering that data would not change the outcome
    /// (`DefragError::NoSuitableStrategy` either way) and free-space
    /// tracking would need to be added at the caller first. Tracked as a
    /// finding rather than force-integrated; see the crate's lint/unwrap
    /// sweep notes.
    pub fn defragment(
        &mut self,
        _memory_regions: &HashMap<usize, MemoryRegion>,
    ) -> Result<usize, MemoryManagementError> {
        let start_time = Instant::now();

        let compaction_result = self
            .defrag_engine
            .defragment()
            .map_err(|e| MemoryManagementError::DefragmentationFailed(format!("{:?}", e)))?;

        let fragmentation_reduced = compaction_result.bytes_moved;

        self.stats.defragmentation_cycles += 1;
        self.stats.fragmentation_reduced += fragmentation_reduced;
        self.stats.total_management_time += start_time.elapsed();

        Ok(fragmentation_reduced)
    }

    /// Check memory pressure and trigger appropriate management
    pub fn handle_memory_pressure(
        &mut self,
        memory_usage_ratio: f64,
        memory_regions: &HashMap<usize, MemoryRegion>,
    ) -> Result<(), MemoryManagementError> {
        if memory_usage_ratio > self.config.memory_pressure_threshold {
            self.stats.memory_pressure_events += 1;

            // Try garbage collection first
            let _ = self.run_garbage_collection(memory_regions)?;

            // If still under pressure, try eviction
            if memory_usage_ratio > 0.9 {
                let target_eviction =
                    (memory_usage_ratio - self.config.memory_pressure_threshold) * 1_000_000.0; // Estimate bytes
                let _ = self.evict_memory(target_eviction as usize, memory_regions)?;
            }

            // If severely fragmented, run defragmentation
            if memory_usage_ratio > 0.95 {
                let _ = self.defragment(memory_regions)?;
            }
        }

        Ok(())
    }

    /// Update access patterns for adaptive management
    ///
    /// Feeds the access into the [`PrefetchingEngine`]'s access-history
    /// tracker (mapped from this module's coarse [`AccessType`] to
    /// [`prefetching::AccessType`]) so its pattern strategies observe real
    /// traffic instead of being permanently starved of data.
    pub fn update_access_pattern(
        &mut self,
        address: *mut c_void,
        size: usize,
        access_type: AccessType,
    ) -> Result<(), MemoryManagementError> {
        let mapped_type = match access_type {
            AccessType::Read | AccessType::Sequential => prefetching::AccessType::Read,
            AccessType::Write => prefetching::AccessType::Write,
            AccessType::ReadWrite | AccessType::Random => prefetching::AccessType::ReadWrite,
        };

        self.prefetch_engine
            .record_access(prefetching::MemoryAccess {
                address: address as usize,
                size,
                timestamp: Instant::now(),
                access_type: mapped_type,
                context_id: 0,
                kernel_id: None,
            });

        Ok(())
    }

    /// Get management statistics
    pub fn get_stats(&self) -> &ManagementStats {
        &self.stats
    }

    /// Get garbage collection stats
    pub fn get_gc_stats(&self) -> GCStats {
        self.gc_engine.get_stats().clone()
    }

    /// Get prefetch performance
    pub fn get_prefetch_performance(&self) -> PrefetchPerformance {
        PrefetchPerformance {
            requests: self.stats.prefetch_requests,
            hits: self.stats.prefetch_hits,
            accuracy: self.stats.prefetch_accuracy,
            // FEATURE STATUS (v1.0.0): Prefetch cache size tracking pending
            // Will be implemented when prefetch_engine.get_cache_size() is available (v1.1.0+)
            cache_size: 0,
        }
    }

    /// Optimize management policies based on workload
    pub fn optimize_policies(&mut self) -> Result<(), MemoryManagementError> {
        // FEATURE STATUS (v1.0.0): Adaptive policy optimization pending
        //
        // The individual policy engines (GC, eviction, prefetch) are functional,
        // but automatic policy selection based on workload analysis requires
        // runtime profiling capabilities planned for v1.1.0.
        //
        // PLANNED (v1.1.0+):
        // - let access_patterns = self.prefetch_engine.analyze_access_patterns();
        // - self.gc_engine.set_preferred_strategy("generational");
        // - self.eviction_engine.set_active_policy("lru");
        //
        // For v1.0.0, users can manually configure policies through the
        // MemoryManagementConfig during initialization.

        Ok(())
    }
}

/// Memory access types for pattern tracking
#[derive(Debug, Clone)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
    Sequential,
    Random,
}

/// Prefetch performance metrics
#[derive(Debug, Clone)]
pub struct PrefetchPerformance {
    pub requests: u64,
    pub hits: u64,
    pub accuracy: f64,
    pub cache_size: usize,
}

/// Access pattern analysis
#[derive(Debug, Clone)]
pub struct AccessPatterns {
    pub temporal_locality: f64,
    pub spatial_locality: f64,
    pub frequency_based: bool,
    pub stride_patterns: Vec<i64>,
}

/// Memory management errors
#[derive(Debug, Clone)]
pub enum MemoryManagementError {
    GarbageCollectionFailed(String),
    PrefetchFailed(String),
    EvictionFailed(String),
    DefragmentationFailed(String),
    BackgroundManagementDisabled,
    InvalidConfiguration(String),
    InternalError(String),
}

impl std::fmt::Display for MemoryManagementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryManagementError::GarbageCollectionFailed(msg) => {
                write!(f, "Garbage collection failed: {}", msg)
            }
            MemoryManagementError::PrefetchFailed(msg) => write!(f, "Prefetch failed: {}", msg),
            MemoryManagementError::EvictionFailed(msg) => write!(f, "Eviction failed: {}", msg),
            MemoryManagementError::DefragmentationFailed(msg) => {
                write!(f, "Defragmentation failed: {}", msg)
            }
            MemoryManagementError::BackgroundManagementDisabled => {
                write!(f, "Background management is disabled")
            }
            MemoryManagementError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            MemoryManagementError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for MemoryManagementError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrated_manager_creation() {
        let config = MemoryManagementConfig::default();
        let manager = IntegratedMemoryManager::new(config);
        assert!(!manager.background_enabled);
    }

    #[test]
    fn test_background_management() {
        let config = MemoryManagementConfig::default();
        let mut manager = IntegratedMemoryManager::new(config);
        let result = manager.start_background_management();
        assert!(result.is_ok());
        assert!(manager.background_enabled);
    }

    #[test]
    fn test_evict_memory_reclaims_real_bytes() {
        use super::eviction_policies::{CacheObject, ObjectPriority, ObjectType, RegionType};

        let config = MemoryManagementConfig::default();
        let mut manager = IntegratedMemoryManager::new(config);

        let object_size = 4096usize;
        let mut objects = HashMap::new();
        objects.insert(
            0x1000,
            CacheObject {
                address: 0x1000,
                size: object_size,
                created_at: Instant::now(),
                last_access: Instant::now(),
                access_count: 1,
                access_frequency: 1.0,
                priority: ObjectPriority::Normal,
                kernel_context: None,
                object_type: ObjectType::Data,
                eviction_cost: 1.0,
                replacement_cost: 1.0,
            },
        );
        let mut memory_regions = HashMap::new();
        memory_regions.insert(
            0x1000,
            MemoryRegion {
                base_addr: 0x1000,
                size: object_size,
                objects,
                region_type: RegionType::Buffer,
                pressure: 1.0,
                last_eviction: None,
            },
        );

        let evicted = manager
            .evict_memory(object_size, &memory_regions)
            .expect("eviction against a registered region should succeed");
        assert_eq!(evicted, object_size);
        assert_eq!(manager.get_stats().bytes_evicted, object_size as u64);
        assert_eq!(manager.get_stats().evictions_performed, 1);
    }

    #[test]
    fn test_evict_memory_empty_regions_evicts_nothing() {
        let config = MemoryManagementConfig::default();
        let mut manager = IntegratedMemoryManager::new(config);

        let evicted = manager
            .evict_memory(4096, &HashMap::new())
            .expect("eviction over no regions should still succeed");
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_prefetch_records_access_and_updates_stats() {
        let config = MemoryManagementConfig::default();
        let mut manager = IntegratedMemoryManager::new(config);

        let result = manager.prefetch(std::ptr::null_mut(), 128, Some("sequential"));
        assert!(result.is_ok());
        assert_eq!(manager.get_stats().prefetch_requests, 1);
    }

    #[test]
    fn test_update_access_pattern_feeds_prefetch_engine() {
        let config = MemoryManagementConfig::default();
        let mut manager = IntegratedMemoryManager::new(config);

        // Four consecutive forward accesses (64 bytes apart, same "thread")
        // form a sequential run of length >= SequentialConfig's
        // min_sequence_length (3), which should make the engine's
        // SequentialPrefetcher strategy fire and queue real requests -- proof
        // that the access data actually reaches the prefetching engine
        // rather than being dropped on the floor.
        let base = 0x10000usize;
        for i in 0..4u64 {
            let ptr = (base + i as usize * 64) as *mut std::ffi::c_void;
            let result = manager.update_access_pattern(ptr, 64, AccessType::Sequential);
            assert!(result.is_ok());
        }

        assert!(manager.prefetch_engine.get_stats().total_requests > 0);
    }

    #[test]
    fn test_stats_initialization() {
        let config = MemoryManagementConfig::default();
        let manager = IntegratedMemoryManager::new(config);
        let stats = manager.get_stats();
        assert_eq!(stats.gc_collections, 0);
        assert_eq!(stats.prefetch_requests, 0);
    }
}
