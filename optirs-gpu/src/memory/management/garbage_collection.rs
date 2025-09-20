// Garbage collection for GPU memory management
//
// This module provides advanced garbage collection algorithms specifically
// optimized for GPU memory patterns, including mark-and-sweep, generational,
// incremental, and real-time garbage collection strategies.

#[allow(dead_code)]
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Main garbage collection engine
pub struct GarbageCollectionEngine {
    /// GC configuration
    config: GCConfig,
    /// GC statistics
    stats: GCStats,
    /// Active GC algorithms
    collectors: Vec<Box<dyn GarbageCollector>>,
    /// Memory regions under management
    memory_regions: HashMap<usize, MemoryRegion>,
    /// Object reference tracking
    reference_tracker: ReferenceTracker,
    /// GC scheduling state
    scheduler: GCScheduler,
    /// Performance history
    performance_history: VecDeque<GCPerformance>,
}

/// Garbage collection configuration
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Enable automatic garbage collection
    pub auto_gc: bool,
    /// GC trigger threshold (memory usage ratio)
    pub gc_threshold: f64,
    /// Maximum pause time for real-time GC (milliseconds)
    pub max_pause_time: Duration,
    /// Enable generational collection
    pub enable_generational: bool,
    /// Enable incremental collection
    pub enable_incremental: bool,
    /// Enable concurrent collection
    pub enable_concurrent: bool,
    /// Young generation size ratio
    pub young_gen_ratio: f64,
    /// Survivor space ratio
    pub survivor_ratio: f64,
    /// Tenuring threshold for promotion
    pub tenuring_threshold: u32,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// GC algorithm preference
    pub preferred_algorithm: GCAlgorithm,
    /// Enable parallel collection
    pub parallel_gc: bool,
    /// Number of GC worker threads
    pub gc_threads: usize,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            auto_gc: true,
            gc_threshold: 0.8,
            max_pause_time: Duration::from_millis(10),
            enable_generational: true,
            enable_incremental: true,
            enable_concurrent: false,
            young_gen_ratio: 0.3,
            survivor_ratio: 0.1,
            tenuring_threshold: 15,
            enable_stats: true,
            preferred_algorithm: GCAlgorithm::Generational,
            parallel_gc: true,
            gc_threads: 2,
        }
    }
}

/// Available garbage collection algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum GCAlgorithm {
    /// Mark and sweep collection
    MarkSweep,
    /// Copying collection
    Copying,
    /// Generational collection
    Generational,
    /// Incremental collection
    Incremental,
    /// Concurrent collection
    Concurrent,
    /// Reference counting
    ReferenceCounting,
    /// Adaptive algorithm selection
    Adaptive,
}

/// GC statistics
#[derive(Debug, Clone, Default)]
pub struct GCStats {
    /// Total GC cycles
    pub total_cycles: u64,
    /// Total time spent in GC
    pub total_gc_time: Duration,
    /// Total bytes collected
    pub total_bytes_collected: u64,
    /// Total objects collected
    pub total_objects_collected: u64,
    /// Average GC pause time
    pub average_pause_time: Duration,
    /// Maximum GC pause time
    pub max_pause_time: Duration,
    /// GC efficiency (bytes collected per millisecond)
    pub gc_efficiency: f64,
    /// Young generation collections
    pub young_gen_collections: u64,
    /// Old generation collections
    pub old_gen_collections: u64,
    /// Promotion rate (objects/sec)
    pub promotion_rate: f64,
    /// Memory reclaim rate
    pub reclaim_rate: f64,
    /// GC overhead percentage
    pub gc_overhead: f64,
    /// Last GC timestamp
    pub last_gc_time: Option<Instant>,
}

/// Memory region managed by GC
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Base address
    pub base_addr: usize,
    /// Region size
    pub size: usize,
    /// Generation (0 = young, 1+ = old)
    pub generation: u32,
    /// Objects in this region
    pub objects: HashMap<usize, ObjectMetadata>,
    /// Free space bitmap
    pub free_bitmap: Vec<u64>,
    /// Last collection time
    pub last_collection: Option<Instant>,
    /// Collection count
    pub collection_count: u32,
    /// Utilization ratio
    pub utilization: f64,
}

/// Object metadata for GC tracking
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object address
    pub address: usize,
    /// Object size
    pub size: usize,
    /// Object type identifier
    pub type_id: u32,
    /// Reference count
    pub ref_count: u32,
    /// Mark state for mark-and-sweep
    pub marked: bool,
    /// Age in collection cycles
    pub age: u32,
    /// Last access time
    pub last_access: Option<Instant>,
    /// Reference list (for precise GC)
    pub references: Vec<usize>,
}

/// Reference tracking system
pub struct ReferenceTracker {
    /// Object reference graph
    reference_graph: HashMap<usize, HashSet<usize>>,
    /// Reverse reference mapping
    reverse_references: HashMap<usize, HashSet<usize>>,
    /// Root references (stack, globals, etc.)
    root_references: HashSet<usize>,
    /// Write barrier log for concurrent GC
    write_barrier_log: VecDeque<WriteBarrierEntry>,
}

/// Write barrier entry for concurrent GC
#[derive(Debug, Clone)]
pub struct WriteBarrierEntry {
    pub source: usize,
    pub target: usize,
    pub timestamp: Instant,
}

impl ReferenceTracker {
    pub fn new() -> Self {
        Self {
            reference_graph: HashMap::new(),
            reverse_references: HashMap::new(),
            root_references: HashSet::new(),
            write_barrier_log: VecDeque::new(),
        }
    }

    /// Add reference between objects
    pub fn add_reference(&mut self, from: usize, to: usize) {
        self.reference_graph
            .entry(from)
            .or_insert_with(HashSet::new)
            .insert(to);
        self.reverse_references
            .entry(to)
            .or_insert_with(HashSet::new)
            .insert(from);
    }

    /// Remove reference between objects
    pub fn remove_reference(&mut self, from: usize, to: usize) {
        if let Some(refs) = self.reference_graph.get_mut(&from) {
            refs.remove(&to);
        }
        if let Some(refs) = self.reverse_references.get_mut(&to) {
            refs.remove(&from);
        }
    }

    /// Add root reference
    pub fn add_root(&mut self, obj: usize) {
        self.root_references.insert(obj);
    }

    /// Remove root reference
    pub fn remove_root(&mut self, obj: usize) {
        self.root_references.remove(&obj);
    }

    /// Get all objects reachable from roots
    pub fn get_reachable_objects(&self) -> HashSet<usize> {
        let mut reachable = HashSet::new();
        let mut work_list = VecDeque::new();

        // Start with roots
        for &root in &self.root_references {
            reachable.insert(root);
            work_list.push_back(root);
        }

        // Breadth-first traversal
        while let Some(obj) = work_list.pop_front() {
            if let Some(refs) = self.reference_graph.get(&obj) {
                for &target in refs {
                    if reachable.insert(target) {
                        work_list.push_back(target);
                    }
                }
            }
        }

        reachable
    }

    /// Record write barrier for concurrent GC
    pub fn write_barrier(&mut self, source: usize, target: usize) {
        let entry = WriteBarrierEntry {
            source,
            target,
            timestamp: Instant::now(),
        };
        self.write_barrier_log.push_back(entry);
    }
}

/// GC scheduling and coordination
pub struct GCScheduler {
    /// Scheduled GC tasks
    scheduled_tasks: VecDeque<GCTask>,
    /// Current executing task
    current_task: Option<GCTask>,
    /// GC timing state
    timing_state: GCTimingState,
    /// Trigger conditions
    trigger_conditions: Vec<GCTrigger>,
}

/// GC task representation
#[derive(Debug, Clone)]
pub struct GCTask {
    pub id: u64,
    pub algorithm: GCAlgorithm,
    pub priority: GCPriority,
    pub target_region: Option<usize>,
    pub estimated_duration: Duration,
    pub created_at: Instant,
    pub deadline: Option<Instant>,
}

/// GC task priority
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum GCPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// GC timing state
#[derive(Debug, Clone)]
pub struct GCTimingState {
    pub last_young_gc: Option<Instant>,
    pub last_old_gc: Option<Instant>,
    pub gc_frequency: f64,
    pub allocation_rate: f64,
    pub memory_pressure: f64,
}

/// GC trigger conditions
#[derive(Debug, Clone)]
pub enum GCTrigger {
    MemoryThreshold(f64),
    TimeInterval(Duration),
    AllocationCount(u64),
    ExplicitRequest,
    MemoryPressure,
}

/// Garbage collector trait
pub trait GarbageCollector: Send + Sync {
    fn name(&self) -> &str;
    fn can_collect(&self, region: &MemoryRegion) -> bool;
    fn estimate_collection_time(&self, region: &MemoryRegion) -> Duration;
    fn collect(
        &mut self,
        region: &mut MemoryRegion,
        tracker: &mut ReferenceTracker,
    ) -> Result<GCResult, GCError>;
    fn get_statistics(&self) -> GCCollectorStats;
    fn configure(&mut self, config: &GCConfig);
}

/// Result of a garbage collection cycle
#[derive(Debug, Clone)]
pub struct GCResult {
    pub bytes_collected: usize,
    pub objects_collected: u32,
    pub collection_time: Duration,
    pub algorithm_used: GCAlgorithm,
    pub regions_collected: Vec<usize>,
    pub promotion_count: u32,
    pub compaction_performed: bool,
    pub efficiency_score: f64,
}

/// GC collector statistics
#[derive(Debug, Clone, Default)]
pub struct GCCollectorStats {
    pub collections: u64,
    pub total_time: Duration,
    pub total_bytes_collected: u64,
    pub total_objects_collected: u64,
    pub average_efficiency: f64,
    pub success_rate: f64,
}

/// Mark and sweep garbage collector
pub struct MarkSweepCollector {
    stats: GCCollectorStats,
    config: MarkSweepConfig,
}

/// Mark and sweep configuration
#[derive(Debug, Clone)]
pub struct MarkSweepConfig {
    pub enable_compaction: bool,
    pub mark_threshold: f64,
    pub sweep_threshold: f64,
    pub enable_parallel_marking: bool,
    pub enable_parallel_sweeping: bool,
}

impl Default for MarkSweepConfig {
    fn default() -> Self {
        Self {
            enable_compaction: true,
            mark_threshold: 0.7,
            sweep_threshold: 0.5,
            enable_parallel_marking: true,
            enable_parallel_sweeping: true,
        }
    }
}

impl MarkSweepCollector {
    pub fn new(config: MarkSweepConfig) -> Self {
        Self {
            stats: GCCollectorStats::default(),
            config,
        }
    }

    fn mark_phase(&self, region: &mut MemoryRegion, tracker: &ReferenceTracker) -> HashSet<usize> {
        let reachable = tracker.get_reachable_objects();

        // Mark all reachable objects in this region
        for (addr, obj) in region.objects.iter_mut() {
            obj.marked = reachable.contains(addr);
        }

        reachable
    }

    fn sweep_phase(&self, region: &mut MemoryRegion) -> (usize, u32) {
        let mut bytes_collected = 0;
        let mut objects_collected = 0;
        let mut objects_to_remove = Vec::new();

        for (addr, obj) in &region.objects {
            if !obj.marked {
                bytes_collected += obj.size;
                objects_collected += 1;
                objects_to_remove.push(*addr);
            }
        }

        // Remove unmarked objects
        for addr in objects_to_remove {
            region.objects.remove(&addr);
        }

        // Reset marks for next collection
        for obj in region.objects.values_mut() {
            obj.marked = false;
        }

        (bytes_collected, objects_collected)
    }
}

impl GarbageCollector for MarkSweepCollector {
    fn name(&self) -> &str {
        "MarkSweep"
    }

    fn can_collect(&self, region: &MemoryRegion) -> bool {
        !region.objects.is_empty() && region.utilization < self.config.mark_threshold
    }

    fn estimate_collection_time(&self, region: &MemoryRegion) -> Duration {
        let object_count = region.objects.len();
        let base_time = Duration::from_micros((object_count * 10) as u64);

        if self.config.enable_compaction {
            base_time + Duration::from_micros((object_count * 5) as u64)
        } else {
            base_time
        }
    }

    fn collect(
        &mut self,
        region: &mut MemoryRegion,
        tracker: &mut ReferenceTracker,
    ) -> Result<GCResult, GCError> {
        let start_time = Instant::now();

        // Mark phase
        let reachable = self.mark_phase(region, tracker);

        // Sweep phase
        let (bytes_collected, objects_collected) = self.sweep_phase(region);

        let collection_time = start_time.elapsed();

        // Update statistics
        self.stats.collections += 1;
        self.stats.total_time += collection_time;
        self.stats.total_bytes_collected += bytes_collected as u64;
        self.stats.total_objects_collected += objects_collected as u64;

        let efficiency = if collection_time.as_millis() > 0 {
            bytes_collected as f64 / collection_time.as_millis() as f64
        } else {
            0.0
        };

        self.stats.average_efficiency =
            (self.stats.average_efficiency * (self.stats.collections - 1) as f64 + efficiency)
                / self.stats.collections as f64;
        self.stats.success_rate = 1.0; // Mark-sweep always succeeds

        Ok(GCResult {
            bytes_collected,
            objects_collected,
            collection_time,
            algorithm_used: GCAlgorithm::MarkSweep,
            regions_collected: vec![region.base_addr],
            promotion_count: 0,
            compaction_performed: self.config.enable_compaction,
            efficiency_score: efficiency,
        })
    }

    fn get_statistics(&self) -> GCCollectorStats {
        self.stats.clone()
    }

    fn configure(&mut self, config: &GCConfig) {
        // Update configuration based on global GC config
        self.config.enable_parallel_marking = config.parallel_gc;
        self.config.enable_parallel_sweeping = config.parallel_gc;
    }
}

/// Generational garbage collector
pub struct GenerationalCollector {
    stats: GCCollectorStats,
    config: GenerationalConfig,
    young_gen_collector: Box<dyn GarbageCollector>,
    old_gen_collector: Box<dyn GarbageCollector>,
}

/// Generational GC configuration
#[derive(Debug, Clone)]
pub struct GenerationalConfig {
    pub young_gen_threshold: usize,
    pub promotion_age: u32,
    pub minor_gc_frequency: u32,
    pub major_gc_threshold: f64,
    pub enable_remembered_set: bool,
}

impl Default for GenerationalConfig {
    fn default() -> Self {
        Self {
            young_gen_threshold: 1024 * 1024, // 1MB
            promotion_age: 3,
            minor_gc_frequency: 10,
            major_gc_threshold: 0.8,
            enable_remembered_set: true,
        }
    }
}

impl GenerationalCollector {
    pub fn new(config: GenerationalConfig) -> Self {
        let young_collector = Box::new(MarkSweepCollector::new(MarkSweepConfig::default()));
        let old_collector = Box::new(MarkSweepCollector::new(MarkSweepConfig {
            enable_compaction: true,
            ..MarkSweepConfig::default()
        }));

        Self {
            stats: GCCollectorStats::default(),
            config,
            young_gen_collector: young_collector,
            old_gen_collector: old_collector,
        }
    }

    fn should_promote(&self, obj: &ObjectMetadata) -> bool {
        obj.age >= self.config.promotion_age
    }

    fn promote_objects(&self, region: &mut MemoryRegion) -> u32 {
        let mut promoted = 0;

        for obj in region.objects.values_mut() {
            if self.should_promote(obj) && region.generation == 0 {
                promoted += 1;
                // In a real implementation, this would move the object to old generation
            }
        }

        promoted
    }
}

impl GarbageCollector for GenerationalCollector {
    fn name(&self) -> &str {
        "Generational"
    }

    fn can_collect(&self, region: &MemoryRegion) -> bool {
        !region.objects.is_empty()
    }

    fn estimate_collection_time(&self, region: &MemoryRegion) -> Duration {
        if region.generation == 0 {
            self.young_gen_collector.estimate_collection_time(region)
        } else {
            self.old_gen_collector.estimate_collection_time(region)
        }
    }

    fn collect(
        &mut self,
        region: &mut MemoryRegion,
        tracker: &mut ReferenceTracker,
    ) -> Result<GCResult, GCError> {
        let start_time = Instant::now();

        let result = if region.generation == 0 {
            // Minor GC
            self.young_gen_collector.collect(region, tracker)?
        } else {
            // Major GC
            self.old_gen_collector.collect(region, tracker)?
        };

        // Handle promotion for young generation
        let promotion_count = if region.generation == 0 {
            self.promote_objects(region)
        } else {
            0
        };

        // Update ages
        for obj in region.objects.values_mut() {
            obj.age += 1;
        }

        let collection_time = start_time.elapsed();

        // Update statistics
        self.stats.collections += 1;
        self.stats.total_time += collection_time;
        self.stats.total_bytes_collected += result.bytes_collected as u64;
        self.stats.total_objects_collected += result.objects_collected as u64;

        Ok(GCResult {
            promotion_count,
            ..result
        })
    }

    fn get_statistics(&self) -> GCCollectorStats {
        self.stats.clone()
    }

    fn configure(&mut self, config: &GCConfig) {
        self.young_gen_collector.configure(config);
        self.old_gen_collector.configure(config);
    }
}

/// Incremental garbage collector
pub struct IncrementalCollector {
    stats: GCCollectorStats,
    config: IncrementalConfig,
    current_phase: IncrementalPhase,
    work_queue: VecDeque<IncrementalWork>,
}

/// Incremental GC configuration
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    pub time_slice: Duration,
    pub work_unit_size: usize,
    pub pause_threshold: Duration,
    pub enable_write_barriers: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            time_slice: Duration::from_millis(2),
            work_unit_size: 100,
            pause_threshold: Duration::from_millis(5),
            enable_write_barriers: true,
        }
    }
}

/// Incremental GC phases
#[derive(Debug, Clone, PartialEq)]
pub enum IncrementalPhase {
    Idle,
    Marking,
    Sweeping,
    Compacting,
    Finalizing,
}

/// Incremental work unit
#[derive(Debug, Clone)]
pub struct IncrementalWork {
    pub phase: IncrementalPhase,
    pub region_addr: usize,
    pub object_range: (usize, usize),
    pub estimated_time: Duration,
}

impl IncrementalCollector {
    pub fn new(config: IncrementalConfig) -> Self {
        Self {
            stats: GCCollectorStats::default(),
            config,
            current_phase: IncrementalPhase::Idle,
            work_queue: VecDeque::new(),
        }
    }

    fn schedule_incremental_work(&mut self, region: &MemoryRegion) {
        let object_addrs: Vec<usize> = region.objects.keys().copied().collect();
        let chunk_size = self.config.work_unit_size;

        // Schedule marking work
        for chunk in object_addrs.chunks(chunk_size) {
            if !chunk.is_empty() {
                let work = IncrementalWork {
                    phase: IncrementalPhase::Marking,
                    region_addr: region.base_addr,
                    object_range: (chunk[0], chunk[chunk.len() - 1]),
                    estimated_time: Duration::from_micros(chunk.len() as u64 * 10),
                };
                self.work_queue.push_back(work);
            }
        }
    }

    fn perform_incremental_work(
        &mut self,
        region: &mut MemoryRegion,
        tracker: &mut ReferenceTracker,
    ) -> Option<GCResult> {
        let time_budget = self.config.time_slice;
        let start_time = Instant::now();
        let mut work_done = false;

        while start_time.elapsed() < time_budget {
            if let Some(work) = self.work_queue.pop_front() {
                match work.phase {
                    IncrementalPhase::Marking => {
                        // Perform incremental marking
                        let reachable = tracker.get_reachable_objects();
                        for addr in work.object_range.0..=work.object_range.1 {
                            if let Some(obj) = region.objects.get_mut(&addr) {
                                obj.marked = reachable.contains(&addr);
                            }
                        }
                        work_done = true;
                    }
                    IncrementalPhase::Sweeping => {
                        // Perform incremental sweeping
                        for addr in work.object_range.0..=work.object_range.1 {
                            if let Some(obj) = region.objects.get(&addr) {
                                if !obj.marked {
                                    region.objects.remove(&addr);
                                }
                            }
                        }
                        work_done = true;
                    }
                    _ => {}
                }
            } else {
                break;
            }
        }

        if work_done && self.work_queue.is_empty() {
            // Collection complete
            Some(GCResult {
                bytes_collected: 0,   // Would track actual bytes
                objects_collected: 0, // Would track actual objects
                collection_time: start_time.elapsed(),
                algorithm_used: GCAlgorithm::Incremental,
                regions_collected: vec![region.base_addr],
                promotion_count: 0,
                compaction_performed: false,
                efficiency_score: 0.0,
            })
        } else {
            None
        }
    }
}

impl GarbageCollector for IncrementalCollector {
    fn name(&self) -> &str {
        "Incremental"
    }

    fn can_collect(&self, region: &MemoryRegion) -> bool {
        !region.objects.is_empty()
    }

    fn estimate_collection_time(&self, region: &MemoryRegion) -> Duration {
        let object_count = region.objects.len();
        Duration::from_millis(
            (object_count / self.config.work_unit_size) as u64
                * self.config.time_slice.as_millis() as u64,
        )
    }

    fn collect(
        &mut self,
        region: &mut MemoryRegion,
        tracker: &mut ReferenceTracker,
    ) -> Result<GCResult, GCError> {
        if self.work_queue.is_empty() {
            self.schedule_incremental_work(region);
        }

        if let Some(result) = self.perform_incremental_work(region, tracker) {
            self.stats.collections += 1;
            self.stats.total_time += result.collection_time;
            Ok(result)
        } else {
            Err(GCError::CollectionIncomplete(
                "Incremental collection in progress".to_string(),
            ))
        }
    }

    fn get_statistics(&self) -> GCCollectorStats {
        self.stats.clone()
    }

    fn configure(&mut self, config: &GCConfig) {
        self.config.time_slice = config.max_pause_time;
    }
}

/// GC performance metrics
#[derive(Debug, Clone)]
pub struct GCPerformance {
    pub timestamp: Instant,
    pub algorithm: GCAlgorithm,
    pub collection_time: Duration,
    pub bytes_collected: usize,
    pub objects_collected: u32,
    pub regions_affected: usize,
    pub efficiency_score: f64,
    pub memory_before: usize,
    pub memory_after: usize,
}

impl GarbageCollectionEngine {
    pub fn new(config: GCConfig) -> Self {
        let mut collectors: Vec<Box<dyn GarbageCollector>> = Vec::new();

        // Add default collectors
        collectors.push(Box::new(
            MarkSweepCollector::new(MarkSweepConfig::default()),
        ));

        if config.enable_generational {
            collectors.push(Box::new(GenerationalCollector::new(
                GenerationalConfig::default(),
            )));
        }

        if config.enable_incremental {
            collectors.push(Box::new(IncrementalCollector::new(
                IncrementalConfig::default(),
            )));
        }

        let gc_threshold = config.gc_threshold;
        Self {
            config,
            stats: GCStats::default(),
            collectors,
            memory_regions: HashMap::new(),
            reference_tracker: ReferenceTracker::new(),
            scheduler: GCScheduler {
                scheduled_tasks: VecDeque::new(),
                current_task: None,
                timing_state: GCTimingState {
                    last_young_gc: None,
                    last_old_gc: None,
                    gc_frequency: 0.0,
                    allocation_rate: 0.0,
                    memory_pressure: 0.0,
                },
                trigger_conditions: vec![
                    GCTrigger::MemoryThreshold(gc_threshold),
                    GCTrigger::TimeInterval(Duration::from_secs(30)),
                ],
            },
            performance_history: VecDeque::with_capacity(1000),
        }
    }

    /// Register a memory region for GC management
    pub fn register_region(&mut self, base_addr: usize, size: usize, generation: u32) {
        let region = MemoryRegion {
            base_addr,
            size,
            generation,
            objects: HashMap::new(),
            free_bitmap: vec![0; (size / 64) + 1],
            last_collection: None,
            collection_count: 0,
            utilization: 0.0,
        };

        self.memory_regions.insert(base_addr, region);
    }

    /// Add object to GC tracking
    pub fn track_object(
        &mut self,
        region_addr: usize,
        obj_addr: usize,
        size: usize,
        type_id: u32,
    ) -> Result<(), GCError> {
        let region = self
            .memory_regions
            .get_mut(&region_addr)
            .ok_or_else(|| GCError::RegionNotFound("Region not registered".to_string()))?;

        let metadata = ObjectMetadata {
            address: obj_addr,
            size,
            type_id,
            ref_count: 0,
            marked: false,
            age: 0,
            last_access: Some(Instant::now()),
            references: Vec::new(),
        };

        region.objects.insert(obj_addr, metadata);
        Ok(())
    }

    /// Check if GC should be triggered
    pub fn should_collect(&mut self) -> bool {
        if !self.config.auto_gc {
            return false;
        }

        for trigger in &self.scheduler.trigger_conditions {
            match trigger {
                GCTrigger::MemoryThreshold(threshold) => {
                    let total_used = self.calculate_total_memory_usage();
                    let total_size = self.calculate_total_memory_size();
                    if total_size > 0 && (total_used as f64 / total_size as f64) > *threshold {
                        return true;
                    }
                }
                GCTrigger::TimeInterval(interval) => {
                    if let Some(last_gc) = self.stats.last_gc_time {
                        if last_gc.elapsed() > *interval {
                            return true;
                        }
                    } else {
                        return true; // First GC
                    }
                }
                GCTrigger::MemoryPressure => {
                    if self.scheduler.timing_state.memory_pressure > 0.8 {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }

    /// Trigger garbage collection
    pub fn collect(&mut self) -> Result<Vec<GCResult>, GCError> {
        let mut results = Vec::new();

        // First, collect collector indices for each region
        let collector_indices: Result<Vec<(usize, usize)>, GCError> = self
            .memory_regions
            .iter()
            .map(|(addr, region)| {
                let collector_index = self.select_collector(region)?;
                Ok((*addr, collector_index))
            })
            .collect();

        let collector_indices = collector_indices?;

        // Now perform collections
        for (region_addr, collector_index) in collector_indices {
            // Calculate utilization before getting mutable reference
            let utilization = {
                let region = self
                    .memory_regions
                    .get(&region_addr)
                    .ok_or_else(|| GCError::InvalidRegion("Region not found".to_string()))?;
                self.calculate_region_utilization(region)
            };

            let region = self
                .memory_regions
                .get_mut(&region_addr)
                .ok_or_else(|| GCError::InvalidRegion("Region not found".to_string()))?;

            let collector = &mut self.collectors[collector_index];

            // Perform collection
            let result = collector.collect(region, &mut self.reference_tracker)?;

            // Update region state
            region.last_collection = Some(Instant::now());
            region.collection_count += 1;
            region.utilization = utilization;

            // Update global statistics
            self.stats.total_cycles += 1;
            self.stats.total_gc_time += result.collection_time;
            self.stats.total_bytes_collected += result.bytes_collected as u64;
            self.stats.total_objects_collected += result.objects_collected as u64;

            // Update average pause time
            let pause_time = result.collection_time;
            if pause_time > self.stats.max_pause_time {
                self.stats.max_pause_time = pause_time;
            }

            let total_time = self.stats.average_pause_time.as_nanos() as u64
                * (self.stats.total_cycles - 1)
                + pause_time.as_nanos() as u64;
            self.stats.average_pause_time =
                Duration::from_nanos(total_time / self.stats.total_cycles);

            // Record performance
            let performance = GCPerformance {
                timestamp: Instant::now(),
                algorithm: result.algorithm_used.clone(),
                collection_time: result.collection_time,
                bytes_collected: result.bytes_collected,
                objects_collected: result.objects_collected,
                regions_affected: 1,
                efficiency_score: result.efficiency_score,
                memory_before: region.size, // Simplified
                memory_after: region.size - result.bytes_collected,
            };

            self.performance_history.push_back(performance);
            if self.performance_history.len() > 1000 {
                self.performance_history.pop_front();
            }

            results.push(result);
        }

        self.stats.last_gc_time = Some(Instant::now());
        Ok(results)
    }

    fn select_collector(&self, region: &MemoryRegion) -> Result<usize, GCError> {
        for (i, collector) in self.collectors.iter().enumerate() {
            if collector.can_collect(region) {
                return Ok(i);
            }
        }

        Err(GCError::NoSuitableCollector(
            "No collector available for region".to_string(),
        ))
    }

    fn calculate_total_memory_usage(&self) -> usize {
        self.memory_regions
            .values()
            .map(|region| region.objects.values().map(|obj| obj.size).sum::<usize>())
            .sum()
    }

    fn calculate_total_memory_size(&self) -> usize {
        self.memory_regions.values().map(|region| region.size).sum()
    }

    fn calculate_region_utilization(&self, region: &MemoryRegion) -> f64 {
        let used_size: usize = region.objects.values().map(|obj| obj.size).sum();
        used_size as f64 / region.size as f64
    }

    /// Get GC statistics
    pub fn get_stats(&self) -> &GCStats {
        &self.stats
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<GCPerformance> {
        &self.performance_history
    }

    /// Get collector information
    pub fn get_collector_info(&self) -> Vec<(String, GCCollectorStats)> {
        self.collectors
            .iter()
            .map(|collector| (collector.name().to_string(), collector.get_statistics()))
            .collect()
    }

    /// Force collection on specific region
    pub fn force_collect_region(&mut self, region_addr: usize) -> Result<GCResult, GCError> {
        // First get collector index with immutable borrow
        let collector_index = {
            let region = self
                .memory_regions
                .get(&region_addr)
                .ok_or_else(|| GCError::RegionNotFound("Region not found".to_string()))?;
            self.select_collector(region)?
        };

        // Now get mutable reference and perform collection
        let region = self
            .memory_regions
            .get_mut(&region_addr)
            .ok_or_else(|| GCError::RegionNotFound("Region not found".to_string()))?;

        let collector = &mut self.collectors[collector_index];
        collector.collect(region, &mut self.reference_tracker)
    }

    /// Add reference between objects
    pub fn add_reference(&mut self, from: usize, to: usize) {
        self.reference_tracker.add_reference(from, to);
    }

    /// Remove reference between objects  
    pub fn remove_reference(&mut self, from: usize, to: usize) {
        self.reference_tracker.remove_reference(from, to);
    }

    /// Add root reference
    pub fn add_root_reference(&mut self, obj: usize) {
        self.reference_tracker.add_root(obj);
    }

    /// Remove root reference
    pub fn remove_root_reference(&mut self, obj: usize) {
        self.reference_tracker.remove_root(obj);
    }
}

/// GC errors
#[derive(Debug, Clone)]
pub enum GCError {
    RegionNotFound(String),
    ObjectNotFound(String),
    CollectionFailed(String),
    CollectionIncomplete(String),
    NoSuitableCollector(String),
    ConfigurationError(String),
    InternalError(String),
    InvalidRegion(String),
}

impl std::fmt::Display for GCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GCError::RegionNotFound(msg) => write!(f, "Region not found: {}", msg),
            GCError::ObjectNotFound(msg) => write!(f, "Object not found: {}", msg),
            GCError::CollectionFailed(msg) => write!(f, "Collection failed: {}", msg),
            GCError::CollectionIncomplete(msg) => write!(f, "Collection incomplete: {}", msg),
            GCError::NoSuitableCollector(msg) => write!(f, "No suitable collector: {}", msg),
            GCError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            GCError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            GCError::InvalidRegion(msg) => write!(f, "Invalid region: {}", msg),
        }
    }
}

impl std::error::Error for GCError {}

/// Thread-safe garbage collection engine
pub struct ThreadSafeGCEngine {
    engine: Arc<RwLock<GarbageCollectionEngine>>,
}

impl ThreadSafeGCEngine {
    pub fn new(config: GCConfig) -> Self {
        Self {
            engine: Arc::new(RwLock::new(GarbageCollectionEngine::new(config))),
        }
    }

    pub fn should_collect(&self) -> bool {
        let mut engine = self.engine.write().unwrap();
        engine.should_collect()
    }

    pub fn collect(&self) -> Result<Vec<GCResult>, GCError> {
        let mut engine = self.engine.write().unwrap();
        engine.collect()
    }

    pub fn get_stats(&self) -> GCStats {
        let engine = self.engine.read().unwrap();
        engine.get_stats().clone()
    }

    pub fn track_object(
        &self,
        region_addr: usize,
        obj_addr: usize,
        size: usize,
        type_id: u32,
    ) -> Result<(), GCError> {
        let mut engine = self.engine.write().unwrap();
        engine.track_object(region_addr, obj_addr, size, type_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_engine_creation() {
        let config = GCConfig::default();
        let engine = GarbageCollectionEngine::new(config);
        assert!(engine.collectors.len() > 0);
    }

    #[test]
    fn test_region_registration() {
        let config = GCConfig::default();
        let mut engine = GarbageCollectionEngine::new(config);

        engine.register_region(0x1000, 4096, 0);
        assert!(engine.memory_regions.contains_key(&0x1000));
    }

    #[test]
    fn test_object_tracking() {
        let config = GCConfig::default();
        let mut engine = GarbageCollectionEngine::new(config);

        engine.register_region(0x1000, 4096, 0);
        let result = engine.track_object(0x1000, 0x1100, 64, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reference_tracking() {
        let mut tracker = ReferenceTracker::new();

        tracker.add_root(100);
        tracker.add_reference(100, 200);
        tracker.add_reference(200, 300);

        let reachable = tracker.get_reachable_objects();
        assert!(reachable.contains(&100));
        assert!(reachable.contains(&200));
        assert!(reachable.contains(&300));
    }

    #[test]
    fn test_mark_sweep_collector() {
        let config = MarkSweepConfig::default();
        let collector = MarkSweepCollector::new(config);

        assert_eq!(collector.name(), "MarkSweep");
    }

    #[test]
    fn test_generational_collector() {
        let config = GenerationalConfig::default();
        let collector = GenerationalCollector::new(config);

        assert_eq!(collector.name(), "Generational");
    }

    #[test]
    fn test_incremental_collector() {
        let config = IncrementalConfig::default();
        let collector = IncrementalCollector::new(config);

        assert_eq!(collector.name(), "Incremental");
    }

    #[test]
    fn test_thread_safe_gc_engine() {
        let config = GCConfig::default();
        let engine = ThreadSafeGCEngine::new(config);

        let stats = engine.get_stats();
        assert_eq!(stats.total_cycles, 0);
    }
}
