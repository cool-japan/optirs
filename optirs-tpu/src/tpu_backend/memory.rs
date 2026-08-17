//! TPU memory manager: per-device memory pools, usage accounting, and
//! garbage collection bookkeeping.
//!
//! The pools here are real bookkeeping over an address space rather than a
//! handle to physical TPU HBM (which needs a vendor runtime): every allocation
//! carves a block out of a device's free list, every release returns it, and
//! the garbage collector coalesces the resulting fragmentation. That makes
//! `allocate_for_computation` a genuine admission-control decision -- an
//! oversized program is rejected with an honest `Err` instead of being told
//! that zero bytes were reserved for it.

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use scirs2_core::error::ErrorContext;
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};

use super::device_defaults::device_memory_capacity;
use super::types::{
    CompiledProgram, DeviceReservation, GCStatistics, GCStrategy, MemoryAllocation,
    MemoryAllocationStrategy, MemoryBlock, MemoryUsageStatistics, TPUBackendConfig,
};
use super::DeviceId;

/// TPU memory manager
#[derive(Debug)]
pub struct TPUMemoryManager<T: Float + Debug + Send + Sync + 'static> {
    /// Memory pools
    ///
    /// `pub(super)`: inspected by the `tpu_backend` test module (a sibling
    /// submodule) to confirm allocations really land in a device's pool.
    pub(super) memory_pools: HashMap<DeviceId, MemoryPool<T>>,

    /// Allocation strategy
    allocation_strategy: MemoryAllocationStrategy,

    /// Memory usage statistics
    usage_statistics: MemoryUsageStatistics,

    /// Garbage collector
    garbage_collector: MemoryGarbageCollector<T>,

    /// Number of [`Self::allocate_for_computation`] calls seen so far, and how
    /// many of them succeeded. Kept as exact counters so
    /// `usage_statistics.allocation_success_rate` is a real ratio rather than a
    /// float accumulated in place.
    allocation_attempts: usize,

    /// Successful allocation count backing the success rate and the running
    /// mean allocation size.
    allocation_successes: usize,
}

/// Memory pool for a device
#[derive(Debug)]
pub struct MemoryPool<T: Float + Debug + Send + Sync + 'static> {
    /// Total pool size
    total_size: usize,

    /// Available memory
    available_memory: usize,

    /// Free blocks
    free_blocks: Vec<MemoryBlock>,

    /// Allocated blocks
    allocated_blocks: HashMap<usize, MemoryBlock>,

    /// Allocation counter
    allocation_counter: usize,

    /// Phantom data
    _phantom: std::marker::PhantomData<T>,
}

/// Memory garbage collector
#[derive(Debug)]
pub struct MemoryGarbageCollector<T: Float + Debug + Send + Sync + 'static> {
    /// Collection strategy
    strategy: GCStrategy,

    /// Collection threshold
    threshold: f64,

    /// Last collection time
    last_collection: Instant,

    /// Collection statistics
    statistics: GCStatistics,

    /// Phantom data
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> MemoryPool<T> {
    /// A pool covering `total_size` bytes as a single free block.
    fn new(total_size: usize) -> Self {
        let now = Instant::now();
        Self {
            total_size,
            available_memory: total_size,
            free_blocks: vec![MemoryBlock {
                start_address: 0,
                size: total_size,
                allocated_at: now,
                last_accessed: now,
                access_count: 0,
            }],
            allocated_blocks: HashMap::new(),
            allocation_counter: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Bytes currently handed out.
    fn allocated_bytes(&self) -> usize {
        self.total_size.saturating_sub(self.available_memory)
    }

    /// Fraction of the pool currently handed out (0.0 for an empty pool).
    fn utilization(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            self.allocated_bytes() as f64 / self.total_size as f64
        }
    }

    /// External fragmentation: how much of the free space is *not* in the
    /// largest contiguous run. Zero when the free space is one block.
    fn fragmentation_ratio(&self) -> f64 {
        if self.available_memory == 0 {
            return 0.0;
        }
        let largest = self
            .free_blocks
            .iter()
            .map(|block| block.size)
            .max()
            .unwrap_or(0);
        1.0 - (largest as f64 / self.available_memory as f64)
    }

    /// Index of the free block that `strategy` would carve `size` bytes from.
    fn choose_free_block(&self, size: usize, strategy: MemoryAllocationStrategy) -> Option<usize> {
        let fits = |block: &MemoryBlock| block.size >= size;
        match strategy {
            // First block large enough, in address order.
            MemoryAllocationStrategy::FirstFit | MemoryAllocationStrategy::PoolBased => {
                self.free_blocks.iter().position(fits)
            }
            // Tightest fit: minimises leftover, at the cost of small remnants.
            MemoryAllocationStrategy::BestFit | MemoryAllocationStrategy::BuddySystem => self
                .free_blocks
                .iter()
                .enumerate()
                .filter(|(_, block)| fits(block))
                .min_by_key(|(_, block)| block.size)
                .map(|(index, _)| index),
            // Loosest fit: leaves the largest usable remnant behind.
            MemoryAllocationStrategy::WorstFit => self
                .free_blocks
                .iter()
                .enumerate()
                .filter(|(_, block)| fits(block))
                .max_by_key(|(_, block)| block.size)
                .map(|(index, _)| index),
            // Switch behaviour on the pool's own state: tighten up (best fit)
            // while the free space is still contiguous, fall back to first fit
            // once fragmentation makes the search cost outweigh the benefit.
            MemoryAllocationStrategy::Adaptive => {
                if self.fragmentation_ratio() > 0.5 {
                    self.free_blocks.iter().position(fits)
                } else {
                    self.free_blocks
                        .iter()
                        .enumerate()
                        .filter(|(_, block)| fits(block))
                        .min_by_key(|(_, block)| block.size)
                        .map(|(index, _)| index)
                }
            }
        }
    }

    /// Round a request up the way `strategy` requires. Only the buddy system
    /// quantises; every other strategy allocates the exact request.
    fn round_request(size: usize, strategy: MemoryAllocationStrategy) -> usize {
        match strategy {
            MemoryAllocationStrategy::BuddySystem => size.max(1).next_power_of_two(),
            _ => size,
        }
    }

    /// Carve `size` bytes out of the pool, returning the allocation handle.
    fn allocate(&mut self, size: usize, strategy: MemoryAllocationStrategy) -> Option<usize> {
        let request = Self::round_request(size, strategy);
        if request == 0 || request > self.available_memory {
            return None;
        }
        let index = self.choose_free_block(request, strategy)?;
        let start_address = self.free_blocks[index].start_address;

        if self.free_blocks[index].size == request {
            self.free_blocks.remove(index);
        } else {
            self.free_blocks[index].start_address += request;
            self.free_blocks[index].size -= request;
        }

        let now = Instant::now();
        let handle = self.allocation_counter;
        self.allocation_counter += 1;
        self.allocated_blocks.insert(
            handle,
            MemoryBlock {
                start_address,
                size: request,
                allocated_at: now,
                last_accessed: now,
                access_count: 1,
            },
        );
        self.available_memory -= request;
        Some(handle)
    }

    /// Base address of a live block, or `None` if the handle is not allocated.
    fn block_address(&self, handle: usize) -> Option<usize> {
        self.allocated_blocks
            .get(&handle)
            .map(|block| block.start_address)
    }

    /// Largest contiguous free run and the number of free runs, as the
    /// allocator actually sees them.
    fn free_block_summary(&self) -> (usize, usize) {
        let largest = self
            .free_blocks
            .iter()
            .map(|block| block.size)
            .max()
            .unwrap_or(0);
        (largest, self.free_blocks.len())
    }

    /// Return a previously allocated block to the free list. Returns the number
    /// of bytes released (zero for an unknown handle).
    fn release(&mut self, handle: usize) -> usize {
        match self.allocated_blocks.remove(&handle) {
            Some(block) => {
                let size = block.size;
                self.available_memory += size;
                self.free_blocks.push(block);
                self.free_blocks.sort_by_key(|block| block.start_address);
                size
            }
            None => 0,
        }
    }

    /// Merge adjacent free blocks. Returns `(merges_performed,
    /// bytes_folded_into_a_neighbour)`.
    fn coalesce(&mut self) -> (usize, usize) {
        if self.free_blocks.len() < 2 {
            return (0, 0);
        }
        self.free_blocks.sort_by_key(|block| block.start_address);
        let mut merged: Vec<MemoryBlock> = Vec::with_capacity(self.free_blocks.len());
        let mut merges = 0usize;
        let mut bytes_folded = 0usize;
        for block in std::mem::take(&mut self.free_blocks) {
            match merged.last_mut() {
                Some(previous) if previous.start_address + previous.size == block.start_address => {
                    previous.size += block.size;
                    previous.last_accessed = block.last_accessed.max(previous.last_accessed);
                    previous.access_count += block.access_count;
                    merges += 1;
                    bytes_folded += block.size;
                }
                _ => merged.push(block),
            }
        }
        self.free_blocks = merged;
        (merges, bytes_folded)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MemoryGarbageCollector<T> {
    /// Whether the collector's policy says it is time to run on a pool with the
    /// given utilization.
    fn should_collect(&self, utilization: f64, fragmentation: f64) -> bool {
        match self.strategy {
            // Sweep on pressure alone.
            GCStrategy::MarkAndSweep | GCStrategy::Reference => utilization >= self.threshold,
            // Age-sensitive strategies additionally rate-limit themselves.
            GCStrategy::Generational | GCStrategy::LeastRecentlyUsed => {
                utilization >= self.threshold
                    && self.last_collection.elapsed() >= Duration::from_millis(1)
            }
            // React to fragmentation as well as pressure: coalescing is exactly
            // the remedy for a fragmented free list, even below the threshold.
            GCStrategy::Adaptive => utilization >= self.threshold || fragmentation > 0.5,
        }
    }

    /// Coalesce a pool's free list and fold the outcome into the collector's
    /// statistics.
    ///
    /// `total_memory_reclaimed` counts the bytes that stopped being a separate
    /// free fragment because they were folded into a neighbouring block -- the
    /// real effect of coalescing. It is not a claim that live memory was freed;
    /// no live allocation is ever reclaimed behind the caller's back.
    fn collect(&mut self, pool: &mut MemoryPool<T>) -> usize {
        let started = Instant::now();
        let blocks_before = pool.free_blocks.len();
        let (merges, bytes_folded) = pool.coalesce();
        let elapsed = started.elapsed();

        self.statistics.total_collections += 1;
        self.statistics.total_memory_reclaimed += bytes_folded;
        // Running mean over every collection so far.
        let collections = self.statistics.total_collections as u32;
        let previous_total = self
            .statistics
            .average_collection_time
            .saturating_mul(collections.saturating_sub(1));
        self.statistics.average_collection_time = (previous_total + elapsed) / collections.max(1);
        self.statistics.collection_efficiency = if blocks_before == 0 {
            0.0
        } else {
            merges as f64 / blocks_before as f64
        };
        self.last_collection = Instant::now();
        bytes_folded
    }
}

impl<T: Float + Debug + Send + Sync + 'static> TPUMemoryManager<T> {
    /// Create a new TPU memory manager
    pub fn new(config: &TPUBackendConfig) -> Result<Self> {
        let usage_statistics = MemoryUsageStatistics {
            total_allocated: 0,
            peak_usage: 0,
            average_allocation_size: 0,
            fragmentation_ratio: 0.0,
            allocation_success_rate: 1.0,
        };

        let garbage_collector = MemoryGarbageCollector {
            strategy: GCStrategy::Adaptive,
            threshold: 0.8,
            last_collection: Instant::now(),
            statistics: GCStatistics {
                total_collections: 0,
                total_memory_reclaimed: 0,
                average_collection_time: Duration::ZERO,
                collection_efficiency: 0.0,
            },
            _phantom: std::marker::PhantomData,
        };

        // One pool per configured core, sized from the configured TPU version's
        // real per-core HBM capacity, mirroring `DeviceManager::new`'s device
        // enumeration so pool ids and device ids line up.
        let device_count = config.tpu_config.num_cores.max(1);
        let per_device_capacity = device_memory_capacity(config.tpu_config.tpu_version);
        let memory_pools = (0..device_count)
            .map(|index| (DeviceId(index), MemoryPool::new(per_device_capacity)))
            .collect();

        Ok(Self {
            memory_pools,
            allocation_strategy: config.memory_allocation_strategy,
            usage_statistics,
            garbage_collector,
            allocation_attempts: 0,
            allocation_successes: 0,
        })
    }

    /// Get memory utilization statistics: the fraction of the managed pools'
    /// combined capacity that is currently allocated.
    pub fn get_utilization_stats(&self) -> f64 {
        let total: usize = self.memory_pools.values().map(|pool| pool.total_size).sum();
        if total == 0 {
            return 0.0;
        }
        let allocated: usize = self
            .memory_pools
            .values()
            .map(|pool| pool.allocated_bytes())
            .sum();
        allocated as f64 / total as f64
    }

    /// Garbage-collection statistics accumulated so far.
    pub fn gc_statistics(&self) -> &GCStatistics {
        &self.garbage_collector.statistics
    }

    /// Aggregate usage statistics accumulated so far.
    pub fn usage_statistics(&self) -> &MemoryUsageStatistics {
        &self.usage_statistics
    }

    /// The allocator's real free-list shape across every device pool, as
    /// `(largest contiguous free run, number of free runs)`.
    ///
    /// Reported straight from the free lists rather than inferred from
    /// allocation totals, so a consumer (such as the memory profiler's
    /// snapshots) records fragmentation the allocator actually observes.
    pub fn free_block_summary(&self) -> (usize, usize) {
        self.memory_pools
            .values()
            .fold((0, 0), |(largest, count), pool| {
                let (pool_largest, pool_count) = pool.free_block_summary();
                (largest.max(pool_largest), count + pool_count)
            })
    }

    /// Cleanup memory resources
    pub fn cleanup(&mut self) -> Result<()> {
        self.memory_pools.clear();
        self.usage_statistics.total_allocated = 0;
        Ok(())
    }

    /// Reserve the program's memory footprint across `devices`.
    ///
    /// The requirement is split evenly across the given devices (the last
    /// device absorbs the rounding remainder) and each share is carved out of
    /// that device's pool with the configured
    /// [`MemoryAllocationStrategy`]. A share that does not fit triggers a
    /// garbage-collection pass on that pool before a single retry; if it still
    /// does not fit, every block reserved for this call is rolled back and the
    /// caller gets an honest `Err` rather than a zero-byte "allocation".
    pub fn allocate_for_computation(
        &mut self,
        program: &CompiledProgram,
        devices: &[DeviceId],
    ) -> Result<MemoryAllocation> {
        if devices.is_empty() {
            return Err(OptimError::InvalidInput(ErrorContext::new(
                "cannot allocate computation memory without at least one device".to_string(),
            )));
        }

        let required = program.memory_requirements.total_memory;
        let share = required / devices.len();
        let remainder = required % devices.len();

        let mut device_allocations: HashMap<DeviceId, usize> = HashMap::new();
        let mut reservations: Vec<DeviceReservation> = Vec::with_capacity(devices.len());
        let mut total_allocated = 0usize;

        for (index, device) in devices.iter().enumerate() {
            let mut bytes = share;
            if index + 1 == devices.len() {
                bytes += remainder;
            }
            if bytes == 0 {
                continue;
            }

            match self.allocate_on_device(*device, bytes) {
                Ok(handle) => {
                    let address = self
                        .memory_pools
                        .get(device)
                        .and_then(|pool| pool.block_address(handle))
                        .unwrap_or(0);
                    reservations.push(DeviceReservation {
                        device: *device,
                        handle,
                        address,
                        size: bytes,
                    });
                    *device_allocations.entry(*device).or_insert(0) += bytes;
                    total_allocated += bytes;
                }
                Err(error) => {
                    // Roll back anything already reserved for this call so a
                    // failed admission leaves no orphaned blocks behind.
                    for reservation in reservations {
                        if let Some(pool) = self.memory_pools.get_mut(&reservation.device) {
                            pool.release(reservation.handle);
                        }
                    }
                    self.record_allocation_outcome(false, 0);
                    return Err(error);
                }
            }
        }

        self.record_allocation_outcome(true, total_allocated);

        Ok(MemoryAllocation {
            reservations,
            device_allocations,
            total_allocated,
        })
    }

    /// Release every block reserved by a previous
    /// [`Self::allocate_for_computation`] for the same allocation, and coalesce
    /// the freed space back into contiguous runs.
    ///
    /// Returns the number of bytes returned to the pools. Without this the
    /// pools would monotonically fill up across executions and the second
    /// large computation on a device would be (correctly, but pointlessly)
    /// rejected.
    pub fn release_allocation(&mut self, allocation: &MemoryAllocation) -> usize {
        let mut released = 0usize;
        let mut touched: Vec<DeviceId> = Vec::new();
        for reservation in &allocation.reservations {
            if let Some(pool) = self.memory_pools.get_mut(&reservation.device) {
                // Release exactly the block this allocation reserved.
                released += pool.release(reservation.handle);
                if !touched.contains(&reservation.device) {
                    touched.push(reservation.device);
                }
            }
        }
        for device in touched {
            if let Some(pool) = self.memory_pools.get_mut(&device) {
                pool.coalesce();
            }
        }
        self.usage_statistics.total_allocated = self
            .usage_statistics
            .total_allocated
            .saturating_sub(released);
        self.refresh_fragmentation();
        released
    }

    /// Reserve `bytes` on one device, garbage-collecting once before giving up.
    fn allocate_on_device(&mut self, device: DeviceId, bytes: usize) -> Result<usize> {
        let strategy = self.allocation_strategy;
        let pool = self.memory_pools.get_mut(&device).ok_or_else(|| {
            OptimError::DeviceError(ErrorContext::new(format!(
                "no memory pool exists for device {}",
                device.0
            )))
        })?;

        if let Some(handle) = pool.allocate(bytes, strategy) {
            return Ok(handle);
        }

        // Under pressure or fragmentation the collector's remedy is to merge
        // the free list; retry exactly once behind it.
        let utilization = pool.utilization();
        let fragmentation = pool.fragmentation_ratio();
        if self
            .garbage_collector
            .should_collect(utilization, fragmentation)
        {
            // Re-borrow: `collect` needs the pool mutably too.
            if let Some(pool) = self.memory_pools.get_mut(&device) {
                self.garbage_collector.collect(pool);
                if let Some(handle) = pool.allocate(bytes, strategy) {
                    return Ok(handle);
                }
            }
        }

        let pool = self.memory_pools.get(&device).ok_or_else(|| {
            OptimError::DeviceError(ErrorContext::new(format!(
                "no memory pool exists for device {}",
                device.0
            )))
        })?;
        Err(OptimError::MemoryError(ErrorContext::new(format!(
            "device {} cannot reserve {bytes} bytes: {} of {} bytes free, largest contiguous run {} bytes",
            device.0,
            pool.available_memory,
            pool.total_size,
            pool.free_blocks
                .iter()
                .map(|block| block.size)
                .max()
                .unwrap_or(0)
        ))))
    }

    /// Fold one allocation attempt into the aggregate statistics.
    fn record_allocation_outcome(&mut self, succeeded: bool, bytes: usize) {
        self.allocation_attempts += 1;
        if succeeded {
            self.allocation_successes += 1;
        }
        self.usage_statistics.allocation_success_rate =
            self.allocation_successes as f64 / self.allocation_attempts as f64;

        if succeeded {
            self.usage_statistics.total_allocated += bytes;
            self.usage_statistics.peak_usage = self
                .usage_statistics
                .peak_usage
                .max(self.usage_statistics.total_allocated);
            // Running mean of successful allocation sizes.
            let successes = self.allocation_successes;
            self.usage_statistics.average_allocation_size =
                (self.usage_statistics.average_allocation_size * (successes - 1) + bytes)
                    / successes;
        }
        self.refresh_fragmentation();
    }

    /// Capacity-weighted mean external fragmentation across the pools.
    fn refresh_fragmentation(&mut self) {
        let total: usize = self.memory_pools.values().map(|pool| pool.total_size).sum();
        self.usage_statistics.fragmentation_ratio = if total == 0 {
            0.0
        } else {
            self.memory_pools
                .values()
                .map(|pool| pool.fragmentation_ratio() * pool.total_size as f64)
                .sum::<f64>()
                / total as f64
        };
    }
}
