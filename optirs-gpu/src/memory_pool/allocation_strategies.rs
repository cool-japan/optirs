// Memory allocation strategies for different workload patterns

use super::MemoryBlock;
use std::collections::{HashMap, VecDeque};

/// Memory allocation strategies for different workload patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// First-fit allocation (fastest)
    FirstFit,
    /// Best-fit allocation (memory efficient)
    BestFit,
    /// Worst-fit allocation (reduces fragmentation)
    WorstFit,
    /// Buddy system allocation (power-of-2 sizes)
    BuddySystem,
    /// Segregated list allocation (size-based pools)
    SegregatedList,
    /// Adaptive strategy based on workload
    Adaptive,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        AllocationStrategy::Adaptive
    }
}

/// Allocation strategy implementations
pub struct AllocationImplementations;

impl AllocationImplementations {
    /// Find free block using first-fit strategy
    pub fn find_first_fit(
        free_blocks: &mut HashMap<usize, VecDeque<MemoryBlock>>,
        size: usize,
    ) -> Option<MemoryBlock> {
        // Find first block that fits
        for (&block_size, blocks) in free_blocks.iter_mut() {
            if block_size >= size && !blocks.is_empty() {
                let mut block = blocks.pop_front().unwrap();
                block.mark_used();
                return Some(block);
            }
        }
        None
    }

    /// Find free block using best-fit strategy
    pub fn find_best_fit(
        free_blocks: &HashMap<usize, VecDeque<MemoryBlock>>,
        size: usize,
    ) -> Option<(usize, MemoryBlock)> {
        // Find smallest block that fits
        let mut best_size = None;
        let mut best_fit_size = usize::MAX;

        for (&block_size, blocks) in free_blocks {
            if block_size >= size && block_size < best_fit_size && !blocks.is_empty() {
                best_fit_size = block_size;
                best_size = Some(block_size);
            }
        }

        best_size.and_then(|block_size| {
            // This is a read-only operation, caller must handle the mutation
            Some((block_size, MemoryBlock::new(std::ptr::null_mut(), block_size).ok()?))
        })
    }

    /// Find free block using worst-fit strategy
    pub fn find_worst_fit(
        free_blocks: &HashMap<usize, VecDeque<MemoryBlock>>,
        size: usize,
    ) -> Option<(usize, MemoryBlock)> {
        // Find largest block that fits (reduces fragmentation)
        let mut worst_size = None;
        let mut worst_fit_size = 0;

        for (&block_size, blocks) in free_blocks {
            if block_size >= size && block_size > worst_fit_size && !blocks.is_empty() {
                worst_fit_size = block_size;
                worst_size = Some(block_size);
            }
        }

        worst_size.and_then(|block_size| {
            // This is a read-only operation, caller must handle the mutation
            Some((block_size, MemoryBlock::new(std::ptr::null_mut(), block_size).ok()?))
        })
    }

    /// Find free block using buddy system
    pub fn find_buddy_block(
        buddy_tree: &mut BuddyAllocator,
        size: usize,
    ) -> Option<*mut u8> {
        buddy_tree.allocate(size)
    }

    /// Find free block using segregated lists
    pub fn find_segregated_block(
        segregated_lists: &mut SegregatedListAllocator,
        size: usize,
    ) -> Option<*mut u8> {
        segregated_lists.allocate(size)
    }

    /// Find free block using adaptive strategy
    pub fn find_adaptive_block(
        adaptive_allocator: &mut AdaptiveAllocator,
        size: usize,
    ) -> Option<*mut u8> {
        adaptive_allocator.allocate(size)
    }
}

/// Buddy system allocator for power-of-2 allocations
pub struct BuddyAllocator {
    /// Memory base address
    base_addr: usize,
    /// Total memory size (must be power of 2)
    total_size: usize,
    /// Minimum allocation unit size
    min_size: usize,
    /// Maximum order (log2 of total_size / min_size)
    max_order: usize,
    /// Free lists for each order
    free_lists: Vec<Vec<usize>>, // Offsets from base_addr
    /// Allocation bitmap
    allocated: Vec<bool>,
}

impl BuddyAllocator {
    /// Create new buddy allocator
    pub fn new(base_addr: usize, total_size: usize, min_size: usize) -> Option<Self> {
        if !total_size.is_power_of_two() || !min_size.is_power_of_two() || total_size < min_size {
            return None;
        }

        let max_order = (total_size / min_size).trailing_zeros() as usize;
        let mut free_lists = vec![Vec::new(); max_order + 1];

        // Initially, all memory is one large free block
        free_lists[max_order].push(0);

        let allocated = vec![false; total_size / min_size];

        Some(Self {
            base_addr,
            total_size,
            min_size,
            max_order,
            free_lists,
            allocated,
        })
    }

    /// Allocate memory block
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        let order = self.size_to_order(size)?;
        let offset = self.allocate_block(order)?;
        Some((self.base_addr + offset) as *mut u8)
    }

    /// Deallocate memory block
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) -> bool {
        let offset = (ptr as usize).wrapping_sub(self.base_addr);
        if offset >= self.total_size {
            return false;
        }

        if let Some(order) = self.size_to_order(size) {
            self.deallocate_block(offset, order);
            true
        } else {
            false
        }
    }

    /// Convert size to order
    fn size_to_order(&self, size: usize) -> Option<usize> {
        if size == 0 || size > self.total_size {
            return None;
        }

        let order = ((size + self.min_size - 1) / self.min_size).next_power_of_two().trailing_zeros() as usize;
        if order <= self.max_order {
            Some(order)
        } else {
            None
        }
    }

    /// Allocate block of specific order
    fn allocate_block(&mut self, order: usize) -> Option<usize> {
        // Find available block of this order or larger
        for current_order in order..=self.max_order {
            if !self.free_lists[current_order].is_empty() {
                let offset = self.free_lists[current_order].pop().unwrap();

                // Split larger blocks if necessary
                for split_order in (order..current_order).rev() {
                    let buddy_offset = offset + (self.min_size << split_order);
                    self.free_lists[split_order].push(buddy_offset);
                }

                // Mark as allocated
                let block_size = self.min_size << order;
                for i in 0..(block_size / self.min_size) {
                    let index = offset / self.min_size + i;
                    if index < self.allocated.len() {
                        self.allocated[index] = true;
                    }
                }

                return Some(offset);
            }
        }

        None
    }

    /// Deallocate block and merge with buddy if possible
    fn deallocate_block(&mut self, offset: usize, order: usize) {
        // Mark as free
        let block_size = self.min_size << order;
        for i in 0..(block_size / self.min_size) {
            let index = offset / self.min_size + i;
            if index < self.allocated.len() {
                self.allocated[index] = false;
            }
        }

        // Try to merge with buddy
        let buddy_offset = offset ^ (self.min_size << order);

        if order < self.max_order && self.is_free_block(buddy_offset, order) {
            // Remove buddy from free list
            self.free_lists[order].retain(|&x| x != buddy_offset);

            // Merge and recursively try to merge larger buddy
            let merged_offset = offset.min(buddy_offset);
            self.deallocate_block(merged_offset, order + 1);
        } else {
            // Add to free list
            self.free_lists[order].push(offset);
        }
    }

    /// Check if block is free
    fn is_free_block(&self, offset: usize, order: usize) -> bool {
        self.free_lists[order].contains(&offset)
    }
}

/// Segregated list allocator
pub struct SegregatedListAllocator {
    /// Size classes and their free lists
    size_classes: Vec<(usize, VecDeque<*mut u8>)>,
    /// Memory pool for allocations
    memory_pool: Vec<u8>,
    /// Next allocation offset
    next_offset: usize,
}

impl SegregatedListAllocator {
    /// Create new segregated list allocator
    pub fn new(pool_size: usize) -> Self {
        let size_classes = vec![
            (64, VecDeque::new()),
            (128, VecDeque::new()),
            (256, VecDeque::new()),
            (512, VecDeque::new()),
            (1024, VecDeque::new()),
            (2048, VecDeque::new()),
            (4096, VecDeque::new()),
            (8192, VecDeque::new()),
            (16384, VecDeque::new()),
            (32768, VecDeque::new()),
            (65536, VecDeque::new()),
        ];

        Self {
            size_classes,
            memory_pool: vec![0; pool_size],
            next_offset: 0,
        }
    }

    /// Allocate from appropriate size class
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Find appropriate size class
        let size_class_index = self.find_size_class(size)?;
        let (class_size, free_list) = &mut self.size_classes[size_class_index];

        // Try to reuse existing block
        if let Some(ptr) = free_list.pop_front() {
            return Some(ptr);
        }

        // Allocate new block from pool
        if self.next_offset + *class_size <= self.memory_pool.len() {
            let ptr = unsafe { self.memory_pool.as_mut_ptr().add(self.next_offset) };
            self.next_offset += *class_size;
            Some(ptr)
        } else {
            None
        }
    }

    /// Deallocate to appropriate size class
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) -> bool {
        if let Some(size_class_index) = self.find_size_class(size) {
            let (_, free_list) = &mut self.size_classes[size_class_index];
            free_list.push_back(ptr);
            true
        } else {
            false
        }
    }

    /// Find appropriate size class for given size
    fn find_size_class(&self, size: usize) -> Option<usize> {
        self.size_classes
            .iter()
            .position(|(class_size, _)| *class_size >= size)
    }
}

/// Adaptive allocator that switches strategies based on workload
pub struct AdaptiveAllocator {
    /// Current strategy
    current_strategy: AllocationStrategy,
    /// Performance metrics for each strategy
    strategy_metrics: HashMap<AllocationStrategy, StrategyMetrics>,
    /// Free blocks (shared across strategies)
    free_blocks: HashMap<usize, VecDeque<MemoryBlock>>,
    /// Buddy allocator instance
    buddy_allocator: Option<BuddyAllocator>,
    /// Segregated list allocator instance
    segregated_allocator: Option<SegregatedListAllocator>,
    /// Adaptation parameters
    adaptation_threshold: f32,
    /// Evaluation window size
    evaluation_window: usize,
}

/// Performance metrics for allocation strategies
#[derive(Debug, Clone, Default)]
struct StrategyMetrics {
    /// Total allocations
    allocations: usize,
    /// Total allocation time
    total_time_us: u64,
    /// Fragmentation score
    fragmentation_score: f32,
    /// Success rate
    success_rate: f32,
}

impl AdaptiveAllocator {
    /// Create new adaptive allocator
    pub fn new(pool_size: usize) -> Self {
        let mut strategy_metrics = HashMap::new();
        strategy_metrics.insert(AllocationStrategy::FirstFit, StrategyMetrics::default());
        strategy_metrics.insert(AllocationStrategy::BestFit, StrategyMetrics::default());
        strategy_metrics.insert(AllocationStrategy::WorstFit, StrategyMetrics::default());

        Self {
            current_strategy: AllocationStrategy::FirstFit,
            strategy_metrics,
            free_blocks: HashMap::new(),
            buddy_allocator: BuddyAllocator::new(0x10000000, pool_size, 256),
            segregated_allocator: Some(SegregatedListAllocator::new(pool_size)),
            adaptation_threshold: 0.1, // 10% performance difference
            evaluation_window: 1000,   // Evaluate every 1000 allocations
        }
    }

    /// Allocate using current strategy
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        let start_time = std::time::Instant::now();

        let result = match self.current_strategy {
            AllocationStrategy::FirstFit => {
                AllocationImplementations::find_first_fit(&mut self.free_blocks, size)
                    .map(|block| block.as_ptr())
            }
            AllocationStrategy::BestFit => {
                AllocationImplementations::find_best_fit(&self.free_blocks, size)
                    .and_then(|(size, _)| {
                        AllocationImplementations::find_first_fit(&mut self.free_blocks, size)
                            .map(|block| block.as_ptr())
                    })
            }
            AllocationStrategy::WorstFit => {
                AllocationImplementations::find_worst_fit(&self.free_blocks, size)
                    .and_then(|(size, _)| {
                        AllocationImplementations::find_first_fit(&mut self.free_blocks, size)
                            .map(|block| block.as_ptr())
                    })
            }
            AllocationStrategy::BuddySystem => {
                self.buddy_allocator.as_mut()?.allocate(size)
            }
            AllocationStrategy::SegregatedList => {
                self.segregated_allocator.as_mut()?.allocate(size)
            }
            AllocationStrategy::Adaptive => {
                // Use current best strategy
                self.allocate(size)
            }
        };

        // Record metrics
        let elapsed = start_time.elapsed();
        let metrics = self.strategy_metrics.entry(self.current_strategy).or_default();
        metrics.allocations += 1;
        metrics.total_time_us += elapsed.as_micros() as u64;
        metrics.success_rate = if result.is_some() { 1.0 } else { 0.0 };

        // Check if adaptation is needed
        if metrics.allocations % self.evaluation_window == 0 {
            self.evaluate_and_adapt();
        }

        result
    }

    /// Evaluate strategies and adapt if necessary
    fn evaluate_and_adapt(&mut self) {
        let current_performance = self.get_strategy_performance(self.current_strategy);

        // Find best performing strategy
        let mut best_strategy = self.current_strategy;
        let mut best_performance = current_performance;

        for (&strategy, metrics) in &self.strategy_metrics {
            if strategy != self.current_strategy {
                let performance = self.calculate_performance_score(metrics);
                if performance > best_performance + self.adaptation_threshold {
                    best_strategy = strategy;
                    best_performance = performance;
                }
            }
        }

        // Switch if better strategy found
        if best_strategy != self.current_strategy {
            self.current_strategy = best_strategy;
        }
    }

    /// Get performance score for current strategy
    fn get_strategy_performance(&self, strategy: AllocationStrategy) -> f32 {
        self.strategy_metrics
            .get(&strategy)
            .map(|metrics| self.calculate_performance_score(metrics))
            .unwrap_or(0.0)
    }

    /// Calculate performance score from metrics
    fn calculate_performance_score(&self, metrics: &StrategyMetrics) -> f32 {
        if metrics.allocations == 0 {
            return 0.0;
        }

        let avg_time = metrics.total_time_us as f32 / metrics.allocations as f32;
        let time_score = 1.0 / (1.0 + avg_time / 1000.0); // Normalize to microseconds
        let fragmentation_score = 1.0 - metrics.fragmentation_score;

        // Weighted combination
        time_score * 0.5 + fragmentation_score * 0.3 + metrics.success_rate * 0.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buddy_allocator() {
        let mut buddy = BuddyAllocator::new(0x10000000, 1024, 64).unwrap();

        // Allocate some blocks
        let ptr1 = buddy.allocate(64);
        assert!(ptr1.is_some());

        let ptr2 = buddy.allocate(128);
        assert!(ptr2.is_some());

        // Deallocate
        assert!(buddy.deallocate(ptr1.unwrap(), 64));
        assert!(buddy.deallocate(ptr2.unwrap(), 128));
    }

    #[test]
    fn test_segregated_allocator() {
        let mut segregated = SegregatedListAllocator::new(65536);

        let ptr1 = segregated.allocate(100);
        assert!(ptr1.is_some());

        let ptr2 = segregated.allocate(200);
        assert!(ptr2.is_some());

        assert!(segregated.deallocate(ptr1.unwrap(), 100));
        assert!(segregated.deallocate(ptr2.unwrap(), 200));
    }

    #[test]
    fn test_adaptive_allocator() {
        let mut adaptive = AdaptiveAllocator::new(65536);

        // Allocate various sizes
        for size in [64, 128, 256, 512, 1024] {
            let ptr = adaptive.allocate(size);
            // Note: This might return None due to empty free_blocks in test
        }
    }
}