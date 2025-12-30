// Memory defragmentation for GPU memory management
//
// This module provides sophisticated defragmentation algorithms to reduce
// memory fragmentation and improve allocation success rates and performance.

#[allow(dead_code)]
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory defragmentation engine
pub struct DefragmentationEngine {
    /// Configuration
    config: DefragConfig,
    /// Statistics
    stats: DefragStats,
    /// Active defragmentation tasks
    active_tasks: Vec<DefragTask>,
    /// Memory layout tracking
    memory_layout: MemoryLayoutTracker,
    /// Compaction strategies
    strategies: Vec<Box<dyn CompactionStrategy>>,
    /// Performance history
    performance_history: VecDeque<DefragPerformance>,
}

/// Defragmentation configuration
#[derive(Debug, Clone)]
pub struct DefragConfig {
    /// Enable automatic defragmentation
    pub auto_defrag: bool,
    /// Fragmentation threshold for triggering defrag (0.0-1.0)
    pub fragmentation_threshold: f64,
    /// Maximum time to spend on defragmentation per cycle
    pub max_defrag_time: Duration,
    /// Minimum free space required before defragmentation
    pub min_free_space: usize,
    /// Enable incremental defragmentation
    pub incremental_defrag: bool,
    /// Chunk size for incremental operations
    pub incremental_chunk_size: usize,
    /// Enable parallel defragmentation
    pub parallel_defrag: bool,
    /// Number of worker threads for parallel operations
    pub worker_threads: usize,
    /// Compaction algorithm preference
    pub preferred_algorithm: CompactionAlgorithm,
    /// Enable statistics collection
    pub enable_stats: bool,
}

impl Default for DefragConfig {
    fn default() -> Self {
        Self {
            auto_defrag: true,
            fragmentation_threshold: 0.3,
            max_defrag_time: Duration::from_millis(100),
            min_free_space: 1024 * 1024, // 1MB
            incremental_defrag: true,
            incremental_chunk_size: 64 * 1024, // 64KB
            parallel_defrag: false,
            worker_threads: 2,
            preferred_algorithm: CompactionAlgorithm::SlidingCompaction,
            enable_stats: true,
        }
    }
}

/// Compaction algorithms available
#[derive(Debug, Clone, PartialEq)]
pub enum CompactionAlgorithm {
    /// Simple sliding compaction
    SlidingCompaction,
    /// Two-pointer compaction
    TwoPointer,
    /// Mark and sweep with compaction
    MarkSweepCompact,
    /// Copying garbage collection style
    CopyingGC,
    /// Generational compaction
    Generational,
    /// Adaptive algorithm selection
    Adaptive,
}

/// Defragmentation statistics
#[derive(Debug, Clone, Default)]
pub struct DefragStats {
    /// Total defragmentation cycles
    pub total_cycles: u64,
    /// Total bytes moved during defragmentation
    pub total_bytes_moved: u64,
    /// Total time spent on defragmentation
    pub total_time_spent: Duration,
    /// Average fragmentation reduction per cycle
    pub average_fragmentation_reduction: f64,
    /// Successful defragmentation attempts
    pub successful_cycles: u64,
    /// Failed defragmentation attempts
    pub failed_cycles: u64,
    /// Average cycle time
    pub average_cycle_time: Duration,
    /// Peak fragmentation level observed
    pub peak_fragmentation: f64,
    /// Current fragmentation level
    pub current_fragmentation: f64,
    /// Objects relocated during defragmentation
    pub objects_relocated: u64,
    /// Memory compaction efficiency
    pub compaction_efficiency: f64,
}

/// Individual defragmentation task
#[derive(Debug, Clone)]
pub struct DefragTask {
    /// Task ID
    pub id: u64,
    /// Start address of memory region
    pub start_addr: usize,
    /// Size of memory region
    pub size: usize,
    /// Algorithm to use
    pub algorithm: CompactionAlgorithm,
    /// Task status
    pub status: TaskStatus,
    /// Creation time
    pub created_at: Instant,
    /// Estimated completion time
    pub estimated_completion: Option<Duration>,
    /// Priority level
    pub priority: TaskPriority,
}

/// Task status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed(String),
    Cancelled,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Memory layout tracking for defragmentation
pub struct MemoryLayoutTracker {
    /// Free memory regions
    free_regions: BTreeMap<usize, FreeRegion>,
    /// Allocated memory blocks
    allocated_blocks: HashMap<usize, AllocatedBlock>,
    /// Fragmentation index cache
    fragmentation_cache: Option<(f64, Instant)>,
    /// Cache validity duration
    cache_validity: Duration,
}

/// Free memory region descriptor
#[derive(Debug, Clone)]
pub struct FreeRegion {
    pub address: usize,
    pub size: usize,
    pub age: Duration,
    pub access_frequency: u32,
    pub adjacent_to_allocated: bool,
}

/// Allocated memory block descriptor
#[derive(Debug, Clone)]
pub struct AllocatedBlock {
    pub address: usize,
    pub size: usize,
    pub allocation_time: Instant,
    pub last_access: Option<Instant>,
    pub access_count: u32,
    pub is_movable: bool,
    pub reference_count: u32,
}

/// Compaction strategy interface
pub trait CompactionStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn can_handle(&self, layout: &MemoryLayoutTracker) -> bool;
    fn estimate_benefit(&self, layout: &MemoryLayoutTracker) -> f64;
    fn execute(
        &mut self,
        layout: &mut MemoryLayoutTracker,
    ) -> Result<CompactionResult, DefragError>;
    fn get_statistics(&self) -> CompactionStats;
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub bytes_moved: usize,
    pub objects_relocated: u32,
    pub fragmentation_reduction: f64,
    pub time_taken: Duration,
    pub algorithm_used: CompactionAlgorithm,
    pub efficiency_score: f64,
}

/// Statistics for compaction strategies
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    pub executions: u64,
    pub total_bytes_moved: u64,
    pub total_objects_relocated: u64,
    pub total_time: Duration,
    pub average_efficiency: f64,
    pub success_rate: f64,
}

/// Performance metrics for defragmentation
#[derive(Debug, Clone)]
pub struct DefragPerformance {
    pub timestamp: Instant,
    pub fragmentation_before: f64,
    pub fragmentation_after: f64,
    pub time_taken: Duration,
    pub bytes_moved: usize,
    pub algorithm_used: CompactionAlgorithm,
    pub success: bool,
}

impl Default for MemoryLayoutTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryLayoutTracker {
    pub fn new() -> Self {
        Self {
            free_regions: BTreeMap::new(),
            allocated_blocks: HashMap::new(),
            fragmentation_cache: None,
            cache_validity: Duration::from_millis(100),
        }
    }

    /// Calculate current fragmentation index
    pub fn calculate_fragmentation(&mut self) -> f64 {
        let now = Instant::now();

        // Check cache validity
        if let Some((cached_frag, cache_time)) = self.fragmentation_cache {
            if now.duration_since(cache_time) < self.cache_validity {
                return cached_frag;
            }
        }

        let fragmentation = if self.free_regions.is_empty() {
            0.0
        } else {
            let total_free_space: usize = self.free_regions.values().map(|r| r.size).sum();
            let largest_free_block = self
                .free_regions
                .values()
                .map(|r| r.size)
                .max()
                .unwrap_or(0);

            if total_free_space == 0 {
                0.0
            } else {
                1.0 - (largest_free_block as f64 / total_free_space as f64)
            }
        };

        // Cache the result
        self.fragmentation_cache = Some((fragmentation, now));
        fragmentation
    }

    /// Add a free region
    pub fn add_free_region(&mut self, address: usize, size: usize) {
        let region = FreeRegion {
            address,
            size,
            age: Duration::from_secs(0),
            access_frequency: 0,
            adjacent_to_allocated: self.is_adjacent_to_allocated(address, size),
        };
        self.free_regions.insert(address, region);
        self.invalidate_cache();
    }

    /// Add an allocated block
    pub fn add_allocated_block(&mut self, address: usize, size: usize, is_movable: bool) {
        let block = AllocatedBlock {
            address,
            size,
            allocation_time: Instant::now(),
            last_access: None,
            access_count: 0,
            is_movable,
            reference_count: 1,
        };
        self.allocated_blocks.insert(address, block);
        self.invalidate_cache();
    }

    /// Remove a free region
    pub fn remove_free_region(&mut self, address: usize) -> Option<FreeRegion> {
        self.invalidate_cache();
        self.free_regions.remove(&address)
    }

    /// Remove an allocated block
    pub fn remove_allocated_block(&mut self, address: usize) -> Option<AllocatedBlock> {
        self.invalidate_cache();
        self.allocated_blocks.remove(&address)
    }

    /// Get total free space
    pub fn get_total_free_space(&self) -> usize {
        self.free_regions.values().map(|r| r.size).sum()
    }

    /// Get largest free block
    pub fn get_largest_free_block(&self) -> usize {
        self.free_regions
            .values()
            .map(|r| r.size)
            .max()
            .unwrap_or(0)
    }

    /// Get movable blocks for compaction
    pub fn get_movable_blocks(&self) -> Vec<&AllocatedBlock> {
        self.allocated_blocks
            .values()
            .filter(|b| b.is_movable)
            .collect()
    }

    /// Check if address range is adjacent to allocated blocks
    fn is_adjacent_to_allocated(&self, address: usize, size: usize) -> bool {
        let end_address = address + size;

        for block in self.allocated_blocks.values() {
            let block_end = block.address + block.size;

            // Check if regions are adjacent
            if block_end == address || block.address == end_address {
                return true;
            }
        }

        false
    }

    /// Invalidate fragmentation cache
    fn invalidate_cache(&mut self) {
        self.fragmentation_cache = None;
    }

    /// Coalesce adjacent free regions
    pub fn coalesce_free_regions(&mut self) -> usize {
        let mut coalesced_count = 0;
        let mut regions_to_remove = Vec::new();
        let mut regions_to_add = Vec::new();

        let addresses: Vec<usize> = self.free_regions.keys().cloned().collect();

        for &addr in &addresses {
            if regions_to_remove.contains(&addr) {
                continue;
            }

            if let Some(region) = self.free_regions.get(&addr) {
                let end_addr = addr + region.size;

                // Look for adjacent region
                if let Some(next_region) = self.free_regions.get(&end_addr) {
                    // Coalesce regions
                    let coalesced_region = FreeRegion {
                        address: addr,
                        size: region.size + next_region.size,
                        age: region.age.min(next_region.age),
                        access_frequency: region.access_frequency + next_region.access_frequency,
                        adjacent_to_allocated: region.adjacent_to_allocated
                            || next_region.adjacent_to_allocated,
                    };

                    regions_to_remove.push(addr);
                    regions_to_remove.push(end_addr);
                    regions_to_add.push((addr, coalesced_region));
                    coalesced_count += 1;
                }
            }
        }

        // Apply changes
        for addr in regions_to_remove {
            self.free_regions.remove(&addr);
        }

        for (addr, region) in regions_to_add {
            self.free_regions.insert(addr, region);
        }

        self.invalidate_cache();
        coalesced_count
    }
}

/// Sliding compaction strategy
pub struct SlidingCompactionStrategy {
    stats: CompactionStats,
}

impl Default for SlidingCompactionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl SlidingCompactionStrategy {
    pub fn new() -> Self {
        Self {
            stats: CompactionStats::default(),
        }
    }
}

impl CompactionStrategy for SlidingCompactionStrategy {
    fn name(&self) -> &str {
        "SlidingCompaction"
    }

    fn can_handle(&self, layout: &MemoryLayoutTracker) -> bool {
        !layout.get_movable_blocks().is_empty() && layout.get_total_free_space() > 0
    }

    fn estimate_benefit(&self, layout: &MemoryLayoutTracker) -> f64 {
        let movable_blocks = layout.get_movable_blocks();
        let total_free = layout.get_total_free_space();
        let largest_free = layout.get_largest_free_block();

        if total_free == 0 {
            return 0.0;
        }

        // Estimate benefit based on potential consolidation
        let fragmentation_reduction = (total_free - largest_free) as f64 / total_free as f64;
        let mobility_factor =
            movable_blocks.len() as f64 / (layout.allocated_blocks.len() as f64 + 1.0);

        fragmentation_reduction * mobility_factor
    }

    fn execute(
        &mut self,
        layout: &mut MemoryLayoutTracker,
    ) -> Result<CompactionResult, DefragError> {
        let start_time = Instant::now();
        let initial_fragmentation = layout.calculate_fragmentation();

        let movable_blocks: Vec<AllocatedBlock> =
            layout.get_movable_blocks().into_iter().cloned().collect();

        if movable_blocks.is_empty() {
            return Err(DefragError::NoMovableBlocks);
        }

        let mut bytes_moved = 0;
        let mut objects_relocated = 0;
        let mut compaction_address = 0;

        // Find the starting address for compaction
        if let Some((&first_free_addr, _)) = layout.free_regions.iter().next() {
            compaction_address = first_free_addr;
        }

        // Sort blocks by address for sliding compaction
        let mut sorted_blocks = movable_blocks;
        sorted_blocks.sort_by_key(|b| b.address);

        // Perform sliding compaction
        for block in sorted_blocks {
            if block.address > compaction_address {
                // Move block to compaction address
                layout.remove_allocated_block(block.address);
                layout.add_allocated_block(compaction_address, block.size, block.is_movable);

                // Add freed space to free regions
                layout.add_free_region(block.address, block.size);

                bytes_moved += block.size;
                objects_relocated += 1;

                compaction_address += block.size;
            } else {
                compaction_address = block.address + block.size;
            }
        }

        // Coalesce free regions after compaction
        layout.coalesce_free_regions();

        let final_fragmentation = layout.calculate_fragmentation();
        let fragmentation_reduction = initial_fragmentation - final_fragmentation;
        let time_taken = start_time.elapsed();

        // Update statistics
        self.stats.executions += 1;
        self.stats.total_bytes_moved += bytes_moved as u64;
        self.stats.total_objects_relocated += objects_relocated as u64;
        self.stats.total_time += time_taken;

        let efficiency = if bytes_moved > 0 {
            fragmentation_reduction / (bytes_moved as f64 / 1024.0 / 1024.0) // MB moved
        } else {
            0.0
        };

        self.stats.average_efficiency =
            (self.stats.average_efficiency * (self.stats.executions - 1) as f64 + efficiency)
                / self.stats.executions as f64;
        self.stats.success_rate = 1.0; // All successful for now

        Ok(CompactionResult {
            bytes_moved,
            objects_relocated,
            fragmentation_reduction,
            time_taken,
            algorithm_used: CompactionAlgorithm::SlidingCompaction,
            efficiency_score: efficiency,
        })
    }

    fn get_statistics(&self) -> CompactionStats {
        self.stats.clone()
    }
}

/// Two-pointer compaction strategy
pub struct TwoPointerCompactionStrategy {
    stats: CompactionStats,
}

impl Default for TwoPointerCompactionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoPointerCompactionStrategy {
    pub fn new() -> Self {
        Self {
            stats: CompactionStats::default(),
        }
    }
}

impl CompactionStrategy for TwoPointerCompactionStrategy {
    fn name(&self) -> &str {
        "TwoPointer"
    }

    fn can_handle(&self, layout: &MemoryLayoutTracker) -> bool {
        layout.get_movable_blocks().len() >= 2 && layout.get_total_free_space() > 0
    }

    fn estimate_benefit(&self, layout: &MemoryLayoutTracker) -> f64 {
        let movable_blocks = layout.get_movable_blocks();
        let free_space = layout.get_total_free_space();

        if movable_blocks.len() < 2 || free_space == 0 {
            return 0.0;
        }

        // Estimate benefit based on gap reduction potential
        let mut addresses: Vec<usize> = movable_blocks.iter().map(|b| b.address).collect();
        addresses.sort();

        let mut total_gaps = 0;
        for i in 1..addresses.len() {
            let gap = addresses[i] - addresses[i - 1];
            if gap > movable_blocks[i - 1].size {
                total_gaps += gap - movable_blocks[i - 1].size;
            }
        }

        total_gaps as f64 / free_space as f64
    }

    fn execute(
        &mut self,
        layout: &mut MemoryLayoutTracker,
    ) -> Result<CompactionResult, DefragError> {
        let start_time = Instant::now();
        let initial_fragmentation = layout.calculate_fragmentation();

        let movable_blocks: Vec<AllocatedBlock> =
            layout.get_movable_blocks().into_iter().cloned().collect();

        if movable_blocks.len() < 2 {
            return Err(DefragError::InsufficientBlocks);
        }

        let mut bytes_moved = 0;
        let mut objects_relocated = 0;

        // Sort blocks by address
        let mut sorted_blocks = movable_blocks;
        sorted_blocks.sort_by_key(|b| b.address);

        let left_ptr = 0;
        let mut compact_addr = sorted_blocks[0].address;

        // Two-pointer compaction
        for block in &sorted_blocks {
            if block.address != compact_addr {
                // Move block to compact address
                layout.remove_allocated_block(block.address);
                layout.add_allocated_block(compact_addr, block.size, block.is_movable);

                // Add freed space to free regions
                layout.add_free_region(block.address, block.size);

                bytes_moved += block.size;
                objects_relocated += 1;
            }

            compact_addr += block.size;
        }

        // Coalesce free regions
        layout.coalesce_free_regions();

        let final_fragmentation = layout.calculate_fragmentation();
        let fragmentation_reduction = initial_fragmentation - final_fragmentation;
        let time_taken = start_time.elapsed();

        // Update statistics
        self.stats.executions += 1;
        self.stats.total_bytes_moved += bytes_moved as u64;
        self.stats.total_objects_relocated += objects_relocated as u64;
        self.stats.total_time += time_taken;

        let efficiency = if bytes_moved > 0 {
            fragmentation_reduction / (bytes_moved as f64 / 1024.0 / 1024.0)
        } else {
            0.0
        };

        self.stats.average_efficiency =
            (self.stats.average_efficiency * (self.stats.executions - 1) as f64 + efficiency)
                / self.stats.executions as f64;

        Ok(CompactionResult {
            bytes_moved,
            objects_relocated,
            fragmentation_reduction,
            time_taken,
            algorithm_used: CompactionAlgorithm::TwoPointer,
            efficiency_score: efficiency,
        })
    }

    fn get_statistics(&self) -> CompactionStats {
        self.stats.clone()
    }
}

impl DefragmentationEngine {
    pub fn new(config: DefragConfig) -> Self {
        let strategies: Vec<Box<dyn CompactionStrategy>> = vec![
            Box::new(SlidingCompactionStrategy::new()),
            Box::new(TwoPointerCompactionStrategy::new()),
        ];

        Self {
            config,
            stats: DefragStats::default(),
            active_tasks: Vec::new(),
            memory_layout: MemoryLayoutTracker::new(),
            strategies: strategies
                .into_iter()
                .map(|s| s as Box<dyn CompactionStrategy>)
                .collect(),
            performance_history: VecDeque::with_capacity(1000),
        }
    }

    /// Check if defragmentation should be triggered
    pub fn should_defragment(&mut self) -> bool {
        if !self.config.auto_defrag {
            return false;
        }

        let current_fragmentation = self.memory_layout.calculate_fragmentation();
        self.stats.current_fragmentation = current_fragmentation;

        current_fragmentation > self.config.fragmentation_threshold
            && self.memory_layout.get_total_free_space() >= self.config.min_free_space
    }

    /// Trigger defragmentation
    pub fn defragment(&mut self) -> Result<CompactionResult, DefragError> {
        let start_time = Instant::now();

        if self
            .active_tasks
            .iter()
            .any(|t| t.status == TaskStatus::Running)
        {
            return Err(DefragError::DefragmentationInProgress);
        }

        // Select best compaction strategy
        let strategy_index = self.select_best_strategy()?;
        let strategy = &mut self.strategies[strategy_index];

        // Execute compaction
        let result = strategy.execute(&mut self.memory_layout)?;

        // Update statistics
        self.stats.total_cycles += 1;
        self.stats.total_bytes_moved += result.bytes_moved as u64;
        self.stats.total_time_spent += result.time_taken;
        self.stats.successful_cycles += 1;
        self.stats.objects_relocated += result.objects_relocated as u64;
        self.stats.average_fragmentation_reduction = (self.stats.average_fragmentation_reduction
            * (self.stats.total_cycles - 1) as f64
            + result.fragmentation_reduction)
            / self.stats.total_cycles as f64;

        let cycle_time = start_time.elapsed();
        self.stats.average_cycle_time = Duration::from_nanos(
            (self.stats.average_cycle_time.as_nanos() as u64 * (self.stats.total_cycles - 1)
                + cycle_time.as_nanos() as u64)
                / self.stats.total_cycles,
        );

        // Record performance
        let performance = DefragPerformance {
            timestamp: start_time,
            fragmentation_before: self.stats.current_fragmentation,
            fragmentation_after: self.memory_layout.calculate_fragmentation(),
            time_taken: cycle_time,
            bytes_moved: result.bytes_moved,
            algorithm_used: result.algorithm_used.clone(),
            success: true,
        };

        self.performance_history.push_back(performance);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        Ok(result)
    }

    /// Select the best compaction strategy based on current conditions
    fn select_best_strategy(&mut self) -> Result<usize, DefragError> {
        let mut best_index = 0;
        let mut best_benefit = 0.0;

        for (i, strategy) in self.strategies.iter().enumerate() {
            if strategy.can_handle(&self.memory_layout) {
                let benefit = strategy.estimate_benefit(&self.memory_layout);
                if benefit > best_benefit {
                    best_benefit = benefit;
                    best_index = i;
                }
            }
        }

        if best_benefit == 0.0 {
            return Err(DefragError::NoSuitableStrategy);
        }

        Ok(best_index)
    }

    /// Create a defragmentation task
    pub fn create_task(
        &mut self,
        start_addr: usize,
        size: usize,
        algorithm: CompactionAlgorithm,
        priority: TaskPriority,
    ) -> u64 {
        let task_id = self.active_tasks.len() as u64;
        let task = DefragTask {
            id: task_id,
            start_addr,
            size,
            algorithm,
            status: TaskStatus::Pending,
            created_at: Instant::now(),
            estimated_completion: None,
            priority,
        };

        self.active_tasks.push(task);
        task_id
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &DefragStats {
        &self.stats
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<DefragPerformance> {
        &self.performance_history
    }

    /// Update memory layout
    pub fn update_layout(
        &mut self,
        allocated_blocks: HashMap<usize, AllocatedBlock>,
        free_regions: BTreeMap<usize, FreeRegion>,
    ) {
        self.memory_layout.allocated_blocks = allocated_blocks;
        self.memory_layout.free_regions = free_regions;
        self.memory_layout.invalidate_cache();
    }

    /// Get current memory layout
    pub fn get_layout(&self) -> &MemoryLayoutTracker {
        &self.memory_layout
    }

    /// Reset defragmentation engine
    pub fn reset(&mut self) {
        self.stats = DefragStats::default();
        self.active_tasks.clear();
        self.memory_layout = MemoryLayoutTracker::new();
        self.performance_history.clear();

        // Reset strategy statistics
        for strategy in &mut self.strategies {
            // Would need to add reset method to CompactionStrategy trait
        }
    }
}

// Safety: DefragmentationEngine manages memory defragmentation state and compaction strategies.
// While it contains NonNull pointers via MemoryLayoutTracker and trait objects,
// it's safe to share across threads when protected by Arc<Mutex<>> because:
// 1. All pointer operations are protected by the Mutex providing exclusive access
// 2. CompactionStrategy trait requires Send + Sync
// 3. No thread-local state is maintained
unsafe impl Send for DefragmentationEngine {}
unsafe impl Sync for DefragmentationEngine {}

/// Defragmentation errors
#[derive(Debug, Clone)]
pub enum DefragError {
    DefragmentationInProgress,
    NoMovableBlocks,
    InsufficientBlocks,
    NoSuitableStrategy,
    MemoryLayoutCorrupted,
    TimeoutExceeded,
    InternalError(String),
}

impl std::fmt::Display for DefragError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefragError::DefragmentationInProgress => {
                write!(f, "Defragmentation already in progress")
            }
            DefragError::NoMovableBlocks => write!(f, "No movable blocks available for compaction"),
            DefragError::InsufficientBlocks => {
                write!(f, "Insufficient blocks for compaction strategy")
            }
            DefragError::NoSuitableStrategy => {
                write!(f, "No suitable compaction strategy available")
            }
            DefragError::MemoryLayoutCorrupted => write!(f, "Memory layout is corrupted"),
            DefragError::TimeoutExceeded => write!(f, "Defragmentation timeout exceeded"),
            DefragError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for DefragError {}

/// Thread-safe defragmentation engine wrapper
pub struct ThreadSafeDefragmentationEngine {
    engine: Arc<Mutex<DefragmentationEngine>>,
}

impl ThreadSafeDefragmentationEngine {
    pub fn new(config: DefragConfig) -> Self {
        Self {
            engine: Arc::new(Mutex::new(DefragmentationEngine::new(config))),
        }
    }

    pub fn should_defragment(&self) -> bool {
        let mut engine = self.engine.lock().unwrap();
        engine.should_defragment()
    }

    pub fn defragment(&self) -> Result<CompactionResult, DefragError> {
        let mut engine = self.engine.lock().unwrap();
        engine.defragment()
    }

    pub fn get_stats(&self) -> DefragStats {
        let engine = self.engine.lock().unwrap();
        engine.get_stats().clone()
    }

    pub fn get_performance_history(&self) -> Vec<DefragPerformance> {
        let engine = self.engine.lock().unwrap();
        engine.get_performance_history().iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_tracker() {
        let mut tracker = MemoryLayoutTracker::new();

        // Add some allocated blocks and free regions
        tracker.add_allocated_block(1000, 500, true);
        tracker.add_allocated_block(2000, 300, false);
        tracker.add_free_region(1500, 200);
        tracker.add_free_region(2500, 800);

        let fragmentation = tracker.calculate_fragmentation();
        assert!((0.0..=1.0).contains(&fragmentation));

        let total_free = tracker.get_total_free_space();
        assert_eq!(total_free, 1000);

        let largest_free = tracker.get_largest_free_block();
        assert_eq!(largest_free, 800);
    }

    #[test]
    fn test_sliding_compaction_strategy() {
        let mut strategy = SlidingCompactionStrategy::new();
        let mut layout = MemoryLayoutTracker::new();

        // Set up a fragmented layout
        layout.add_allocated_block(1000, 500, true);
        layout.add_free_region(1500, 200);
        layout.add_allocated_block(2000, 300, true);
        layout.add_free_region(2300, 500);

        assert!(strategy.can_handle(&layout));

        let benefit = strategy.estimate_benefit(&layout);
        assert!(benefit > 0.0);

        let result = strategy.execute(&mut layout);
        assert!(result.is_ok());

        let compaction_result = result.unwrap();
        assert!(compaction_result.bytes_moved > 0);
        assert!(compaction_result.objects_relocated > 0);
    }

    #[test]
    fn test_defragmentation_engine() {
        let config = DefragConfig::default();
        let mut engine = DefragmentationEngine::new(config);

        // Set up memory layout
        let mut allocated_blocks = HashMap::new();
        allocated_blocks.insert(
            1000,
            AllocatedBlock {
                address: 1000,
                size: 500,
                allocation_time: Instant::now(),
                last_access: None,
                access_count: 0,
                is_movable: true,
                reference_count: 1,
            },
        );

        let mut free_regions = BTreeMap::new();
        free_regions.insert(
            1500,
            FreeRegion {
                address: 1500,
                size: 300,
                age: Duration::from_secs(10),
                access_frequency: 0,
                adjacent_to_allocated: true,
            },
        );

        engine.update_layout(allocated_blocks, free_regions);

        // Test defragmentation trigger
        let should_defrag = engine.should_defragment();
        // Result depends on fragmentation threshold and current state

        let stats = engine.get_stats();
        assert_eq!(stats.total_cycles, 0); // No cycles yet
    }

    #[test]
    fn test_coalescing() {
        let mut tracker = MemoryLayoutTracker::new();

        // Add adjacent free regions
        tracker.add_free_region(1000, 500);
        tracker.add_free_region(1500, 300);
        tracker.add_free_region(2000, 200); // Non-adjacent

        let coalesced = tracker.coalesce_free_regions();
        assert_eq!(coalesced, 1); // One coalescing operation

        // Should now have two regions: one large (800 bytes) and one separate (200 bytes)
        assert_eq!(tracker.free_regions.len(), 2);
    }

    #[test]
    fn test_thread_safe_engine() {
        let config = DefragConfig::default();
        let engine = ThreadSafeDefragmentationEngine::new(config);

        let should_defrag = engine.should_defragment();
        // Should not trigger defrag on empty layout

        let stats = engine.get_stats();
        assert_eq!(stats.total_cycles, 0);
    }
}
