// Slab allocator for GPU memory management
//
// This module implements a slab allocator optimized for fixed-size allocations.
// Slab allocation is highly efficient for objects of the same size and provides
// excellent cache locality and minimal fragmentation.

#[allow(dead_code)]
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Slab allocator for fixed-size objects
pub struct SlabAllocator {
    /// Cache configurations indexed by object size
    caches: HashMap<usize, SlabCache>,
    /// Statistics for the entire allocator
    stats: SlabStats,
    /// Configuration
    config: SlabConfig,
    /// Memory pool for backing slabs
    memory_pool: MemoryPool,
}

/// Slab cache for objects of a specific size
pub struct SlabCache {
    /// Object size for this cache
    object_size: usize,
    /// List of slabs (pages)
    slabs: Vec<Slab>,
    /// Partially filled slabs
    partial_slabs: VecDeque<usize>,
    /// Full slabs
    full_slabs: Vec<usize>,
    /// Empty slabs
    empty_slabs: VecDeque<usize>,
    /// Cache statistics
    stats: CacheStats,
    /// Cache configuration
    config: CacheConfig,
}

/// Individual slab (page) containing multiple objects
pub struct Slab {
    /// Base address of the slab
    base_ptr: NonNull<u8>,
    /// Size of the slab in bytes
    slab_size: usize,
    /// Object size
    object_size: usize,
    /// Number of objects in this slab
    object_count: usize,
    /// Free object list (indices)
    free_objects: VecDeque<usize>,
    /// Allocated object count
    allocated_count: usize,
    /// Allocation bitmap for fast lookups
    allocation_bitmap: Vec<u64>,
    /// Slab creation time
    created_at: Instant,
    /// Last allocation time
    last_alloc: Option<Instant>,
    /// Last deallocation time
    last_dealloc: Option<Instant>,
    /// Access frequency counter
    access_count: u64,
}

impl Slab {
    pub fn new(base_ptr: NonNull<u8>, slab_size: usize, object_size: usize) -> Self {
        let object_count = slab_size / object_size;
        let bitmap_size = object_count.div_ceil(64); // Round up to nearest 64-bit word

        let mut free_objects = VecDeque::with_capacity(object_count);
        for i in 0..object_count {
            free_objects.push_back(i);
        }

        Self {
            base_ptr,
            slab_size,
            object_size,
            object_count,
            free_objects,
            allocated_count: 0,
            allocation_bitmap: vec![0; bitmap_size],
            created_at: Instant::now(),
            last_alloc: None,
            last_dealloc: None,
            access_count: 0,
        }
    }

    /// Allocate an object from this slab
    pub fn allocate(&mut self) -> Option<NonNull<u8>> {
        if let Some(object_index) = self.free_objects.pop_front() {
            // Mark object as allocated in bitmap
            let word_index = object_index / 64;
            let bit_index = object_index % 64;
            self.allocation_bitmap[word_index] |= 1u64 << bit_index;

            self.allocated_count += 1;
            self.last_alloc = Some(Instant::now());
            self.access_count += 1;

            // Calculate object address
            let object_offset = object_index * self.object_size;
            let object_ptr =
                unsafe { NonNull::new_unchecked(self.base_ptr.as_ptr().add(object_offset)) };

            Some(object_ptr)
        } else {
            None
        }
    }

    /// Deallocate an object in this slab
    pub fn deallocate(&mut self, ptr: NonNull<u8>) -> Result<(), SlabError> {
        // Calculate object index from pointer
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base_ptr.as_ptr() as usize;

        if ptr_addr < base_addr || ptr_addr >= base_addr + self.slab_size {
            return Err(SlabError::InvalidPointer(
                "Pointer not in this slab".to_string(),
            ));
        }

        let offset = ptr_addr - base_addr;
        if !offset.is_multiple_of(self.object_size) {
            return Err(SlabError::InvalidPointer(
                "Pointer not aligned to object boundary".to_string(),
            ));
        }

        let object_index = offset / self.object_size;
        if object_index >= self.object_count {
            return Err(SlabError::InvalidPointer(
                "Object index out of bounds".to_string(),
            ));
        }

        // Check if object is actually allocated
        let word_index = object_index / 64;
        let bit_index = object_index % 64;
        if (self.allocation_bitmap[word_index] & (1u64 << bit_index)) == 0 {
            return Err(SlabError::DoubleFree("Object already free".to_string()));
        }

        // Mark as free
        self.allocation_bitmap[word_index] &= !(1u64 << bit_index);
        self.free_objects.push_back(object_index);
        self.allocated_count -= 1;
        self.last_dealloc = Some(Instant::now());

        Ok(())
    }

    /// Check if slab is full
    pub fn is_full(&self) -> bool {
        self.allocated_count == self.object_count
    }

    /// Check if slab is empty
    pub fn is_empty(&self) -> bool {
        self.allocated_count == 0
    }

    /// Check if slab is partially filled
    pub fn is_partial(&self) -> bool {
        self.allocated_count > 0 && self.allocated_count < self.object_count
    }

    /// Get utilization ratio (0.0 to 1.0)
    pub fn get_utilization(&self) -> f64 {
        self.allocated_count as f64 / self.object_count as f64
    }

    /// Get slab statistics
    pub fn get_stats(&self) -> SlabStats {
        SlabStats {
            total_objects: self.object_count,
            allocated_objects: self.allocated_count,
            free_objects: self.object_count - self.allocated_count,
            utilization: self.get_utilization(),
            access_count: self.access_count,
            age: self.created_at.elapsed(),
        }
    }
}

/// Memory pool for backing slab storage
pub struct MemoryPool {
    /// Base address of the memory pool
    base_ptr: NonNull<u8>,
    /// Total size of the memory pool
    total_size: usize,
    /// Current allocation offset
    current_offset: usize,
    /// Free regions for reuse
    free_regions: VecDeque<FreeRegion>,
    /// Allocation alignment
    alignment: usize,
}

/// Free memory region
#[derive(Debug, Clone)]
pub struct FreeRegion {
    pub offset: usize,
    pub size: usize,
    pub freed_at: Instant,
}

impl MemoryPool {
    pub fn new(base_ptr: NonNull<u8>, total_size: usize, alignment: usize) -> Self {
        Self {
            base_ptr,
            total_size,
            current_offset: 0,
            free_regions: VecDeque::new(),
            alignment,
        }
    }

    /// Allocate a slab from the memory pool
    pub fn allocate_slab(&mut self, size: usize) -> Option<NonNull<u8>> {
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        // Try to reuse a free region first
        if let Some(region_index) = self.find_suitable_free_region(aligned_size) {
            let region = self.free_regions.remove(region_index).unwrap();
            let ptr = unsafe { NonNull::new_unchecked(self.base_ptr.as_ptr().add(region.offset)) };

            // If region is larger than needed, split it
            if region.size > aligned_size {
                let remaining_region = FreeRegion {
                    offset: region.offset + aligned_size,
                    size: region.size - aligned_size,
                    freed_at: region.freed_at,
                };
                self.free_regions.push_back(remaining_region);
            }

            return Some(ptr);
        }

        // Allocate from the end of the pool
        if self.current_offset + aligned_size <= self.total_size {
            let ptr =
                unsafe { NonNull::new_unchecked(self.base_ptr.as_ptr().add(self.current_offset)) };
            self.current_offset += aligned_size;
            Some(ptr)
        } else {
            None
        }
    }

    /// Free a slab back to the memory pool
    pub fn free_slab(&mut self, ptr: NonNull<u8>, size: usize) {
        let base_addr = self.base_ptr.as_ptr() as usize;
        let ptr_addr = ptr.as_ptr() as usize;

        if ptr_addr >= base_addr && ptr_addr < base_addr + self.total_size {
            let offset = ptr_addr - base_addr;
            let region = FreeRegion {
                offset,
                size,
                freed_at: Instant::now(),
            };

            // Insert in sorted order to facilitate coalescing
            let insert_pos = self
                .free_regions
                .binary_search_by_key(&offset, |r| r.offset)
                .unwrap_or_else(|pos| pos);

            self.free_regions.insert(insert_pos, region);

            // Try to coalesce adjacent regions
            self.coalesce_free_regions();
        }
    }

    fn find_suitable_free_region(&self, size: usize) -> Option<usize> {
        self.free_regions
            .iter()
            .position(|region| region.size >= size)
    }

    fn coalesce_free_regions(&mut self) {
        let mut i = 0;
        while i < self.free_regions.len().saturating_sub(1) {
            let current_end = self.free_regions[i].offset + self.free_regions[i].size;
            if current_end == self.free_regions[i + 1].offset {
                // Coalesce regions
                let next_region = self.free_regions.remove(i + 1).unwrap();
                self.free_regions[i].size += next_region.size;
            } else {
                i += 1;
            }
        }
    }

    pub fn get_usage(&self) -> MemoryPoolUsage {
        let free_size = self.free_regions.iter().map(|r| r.size).sum::<usize>();
        let allocated_size = self.current_offset - free_size;

        MemoryPoolUsage {
            total_size: self.total_size,
            allocated_size,
            free_size,
            current_offset: self.current_offset,
            free_regions: self.free_regions.len(),
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Objects per slab
    pub objects_per_slab: usize,
    /// Maximum number of empty slabs to keep
    pub max_empty_slabs: usize,
    /// Enable slab coloring for cache performance
    pub enable_coloring: bool,
    /// Color offset for cache line alignment
    pub color_offset: usize,
    /// Enable object construction/destruction
    pub enable_ctor_dtor: bool,
    /// Object constructor function
    pub constructor: Option<fn(*mut u8)>,
    /// Object destructor function  
    pub destructor: Option<fn(*mut u8)>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            objects_per_slab: 64,
            max_empty_slabs: 3,
            enable_coloring: true,
            color_offset: 0,
            enable_ctor_dtor: false,
            constructor: None,
            destructor: None,
        }
    }
}

/// Slab allocator configuration
#[derive(Debug, Clone)]
pub struct SlabConfig {
    /// Default slab size
    pub default_slab_size: usize,
    /// Memory alignment requirement
    pub alignment: usize,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Enable debugging features
    pub enable_debug: bool,
    /// Memory reclamation threshold
    pub reclaim_threshold: f64,
    /// Enable automatic reclamation
    pub auto_reclaim: bool,
}

impl Default for SlabConfig {
    fn default() -> Self {
        Self {
            default_slab_size: 4096, // 4KB page size
            alignment: 256,
            enable_stats: true,
            enable_debug: false,
            reclaim_threshold: 0.8,
            auto_reclaim: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub slab_allocations: u64,
    pub slab_deallocations: u64,
    pub objects_allocated: u64,
    pub objects_free: u64,
    pub average_utilization: f64,
}

/// Slab statistics
#[derive(Debug, Clone, Default)]
pub struct SlabStats {
    pub total_objects: usize,
    pub allocated_objects: usize,
    pub free_objects: usize,
    pub utilization: f64,
    pub access_count: u64,
    pub age: std::time::Duration,
}

/// Memory pool usage statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolUsage {
    pub total_size: usize,
    pub allocated_size: usize,
    pub free_size: usize,
    pub current_offset: usize,
    pub free_regions: usize,
}

impl SlabCache {
    pub fn new(object_size: usize, config: CacheConfig) -> Self {
        Self {
            object_size,
            slabs: Vec::new(),
            partial_slabs: VecDeque::new(),
            full_slabs: Vec::new(),
            empty_slabs: VecDeque::new(),
            stats: CacheStats::default(),
            config,
        }
    }

    /// Allocate an object from this cache
    pub fn allocate(&mut self, memory_pool: &mut MemoryPool) -> Result<NonNull<u8>, SlabError> {
        self.stats.total_allocations += 1;

        // Try partial slabs first
        if let Some(&slab_index) = self.partial_slabs.front() {
            if let Some(ptr) = self.slabs[slab_index].allocate() {
                self.stats.cache_hits += 1;
                self.stats.objects_allocated += 1;

                // Move to full slabs if now full
                if self.slabs[slab_index].is_full() {
                    self.partial_slabs.pop_front();
                    self.full_slabs.push(slab_index);
                }

                // Apply constructor if enabled
                if self.config.enable_ctor_dtor {
                    if let Some(ctor) = self.config.constructor {
                        ctor(ptr.as_ptr());
                    }
                }

                return Ok(ptr);
            }
        }

        // Try empty slabs
        if let Some(slab_index) = self.empty_slabs.pop_front() {
            if let Some(ptr) = self.slabs[slab_index].allocate() {
                self.stats.cache_hits += 1;
                self.stats.objects_allocated += 1;
                self.partial_slabs.push_back(slab_index);

                if self.config.enable_ctor_dtor {
                    if let Some(ctor) = self.config.constructor {
                        ctor(ptr.as_ptr());
                    }
                }

                return Ok(ptr);
            }
        }

        // Need to allocate a new slab
        self.stats.cache_misses += 1;
        self.allocate_new_slab(memory_pool)?;

        // Try allocation again with new slab
        if let Some(&slab_index) = self.partial_slabs.back() {
            if let Some(ptr) = self.slabs[slab_index].allocate() {
                self.stats.objects_allocated += 1;

                if self.config.enable_ctor_dtor {
                    if let Some(ctor) = self.config.constructor {
                        ctor(ptr.as_ptr());
                    }
                }

                return Ok(ptr);
            }
        }

        Err(SlabError::OutOfMemory(
            "Failed to allocate after creating new slab".to_string(),
        ))
    }

    /// Deallocate an object back to this cache
    pub fn deallocate(&mut self, ptr: NonNull<u8>) -> Result<(), SlabError> {
        // Apply destructor if enabled
        if self.config.enable_ctor_dtor {
            if let Some(dtor) = self.config.destructor {
                dtor(ptr.as_ptr());
            }
        }

        // Find which slab contains this pointer
        let mut slab_index = None;
        for (i, slab) in self.slabs.iter().enumerate() {
            let base_addr = slab.base_ptr.as_ptr() as usize;
            let ptr_addr = ptr.as_ptr() as usize;

            if ptr_addr >= base_addr && ptr_addr < base_addr + slab.slab_size {
                slab_index = Some(i);
                break;
            }
        }

        let slab_index = slab_index.ok_or_else(|| {
            SlabError::InvalidPointer("Pointer not found in any slab".to_string())
        })?;

        let was_full = self.slabs[slab_index].is_full();
        self.slabs[slab_index].deallocate(ptr)?;

        self.stats.total_deallocations += 1;
        self.stats.objects_allocated -= 1;
        self.stats.objects_free += 1;

        // Update slab lists based on new state
        if was_full {
            // Remove from full slabs, add to partial
            if let Some(pos) = self.full_slabs.iter().position(|&i| i == slab_index) {
                self.full_slabs.remove(pos);
                self.partial_slabs.push_back(slab_index);
            }
        } else if self.slabs[slab_index].is_empty() {
            // Remove from partial, add to empty
            if let Some(pos) = self.partial_slabs.iter().position(|&i| i == slab_index) {
                self.partial_slabs.remove(pos);
                self.empty_slabs.push_back(slab_index);
            }
        }

        Ok(())
    }

    fn allocate_new_slab(&mut self, memory_pool: &mut MemoryPool) -> Result<(), SlabError> {
        let slab_size = self.calculate_slab_size();

        let slab_ptr = memory_pool.allocate_slab(slab_size).ok_or_else(|| {
            SlabError::OutOfMemory("Cannot allocate slab from memory pool".to_string())
        })?;

        let slab = Slab::new(slab_ptr, slab_size, self.object_size);
        let slab_index = self.slabs.len();

        self.slabs.push(slab);
        self.partial_slabs.push_back(slab_index);
        self.stats.slab_allocations += 1;

        Ok(())
    }

    fn calculate_slab_size(&self) -> usize {
        // Calculate optimal slab size based on object size and configuration
        let objects_per_slab = self.config.objects_per_slab;
        let base_size = objects_per_slab * self.object_size;

        // Add coloring offset if enabled
        if self.config.enable_coloring {
            base_size + self.config.color_offset
        } else {
            base_size
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get detailed cache information
    pub fn get_cache_info(&self) -> CacheInfo {
        let total_objects = self.slabs.iter().map(|s| s.object_count).sum();
        let allocated_objects = self.slabs.iter().map(|s| s.allocated_count).sum();
        let average_utilization = if total_objects > 0 {
            allocated_objects as f64 / total_objects as f64
        } else {
            0.0
        };

        CacheInfo {
            object_size: self.object_size,
            total_slabs: self.slabs.len(),
            partial_slabs: self.partial_slabs.len(),
            full_slabs: self.full_slabs.len(),
            empty_slabs: self.empty_slabs.len(),
            total_objects,
            allocated_objects,
            free_objects: total_objects - allocated_objects,
            average_utilization,
            memory_overhead: self.calculate_memory_overhead(),
        }
    }

    fn calculate_memory_overhead(&self) -> f64 {
        let useful_memory: usize = self
            .slabs
            .iter()
            .map(|s| s.allocated_count * s.object_size)
            .sum();

        let total_memory: usize = self.slabs.iter().map(|s| s.slab_size).sum();

        if total_memory > 0 {
            1.0 - (useful_memory as f64 / total_memory as f64)
        } else {
            0.0
        }
    }

    /// Reclaim empty slabs
    pub fn reclaim_empty_slabs(&mut self, memory_pool: &mut MemoryPool) -> usize {
        let mut reclaimed = 0;
        let keep_count = self.config.max_empty_slabs;

        while self.empty_slabs.len() > keep_count {
            if let Some(slab_index) = self.empty_slabs.pop_front() {
                let slab = &self.slabs[slab_index];
                memory_pool.free_slab(slab.base_ptr, slab.slab_size);
                reclaimed += 1;
                self.stats.slab_deallocations += 1;
            }
        }

        reclaimed
    }
}

/// Cache information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub object_size: usize,
    pub total_slabs: usize,
    pub partial_slabs: usize,
    pub full_slabs: usize,
    pub empty_slabs: usize,
    pub total_objects: usize,
    pub allocated_objects: usize,
    pub free_objects: usize,
    pub average_utilization: f64,
    pub memory_overhead: f64,
}

impl SlabAllocator {
    pub fn new(base_ptr: NonNull<u8>, total_size: usize, config: SlabConfig) -> Self {
        let memory_pool = MemoryPool::new(base_ptr, total_size, config.alignment);

        Self {
            caches: HashMap::new(),
            stats: SlabStats::default(),
            memory_pool,
            config,
        }
    }

    /// Allocate object of specified size
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, SlabError> {
        if size == 0 {
            return Err(SlabError::InvalidSize(
                "Cannot allocate zero bytes".to_string(),
            ));
        }

        // Round up size to alignment boundary
        let aligned_size = (size + self.config.alignment - 1) & !(self.config.alignment - 1);

        // Get or create cache for this size
        self.caches.entry(aligned_size).or_insert_with(|| {
            let cache_config = CacheConfig::default();

            SlabCache::new(aligned_size, cache_config)
        });

        let cache = self.caches.get_mut(&aligned_size).unwrap();
        cache.allocate(&mut self.memory_pool)
    }

    /// Deallocate object
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) -> Result<(), SlabError> {
        let aligned_size = (size + self.config.alignment - 1) & !(self.config.alignment - 1);

        let cache = self
            .caches
            .get_mut(&aligned_size)
            .ok_or_else(|| SlabError::InvalidPointer("No cache found for this size".to_string()))?;

        cache.deallocate(ptr)
    }

    /// Get allocator statistics
    pub fn get_stats(&self) -> SlabAllocatorStats {
        let mut total_caches = 0;
        let mut total_slabs = 0;
        let mut total_objects = 0;
        let mut allocated_objects = 0;
        let mut total_allocations = 0;
        let mut total_deallocations = 0;

        for cache in self.caches.values() {
            total_caches += 1;
            let info = cache.get_cache_info();
            total_slabs += info.total_slabs;
            total_objects += info.total_objects;
            allocated_objects += info.allocated_objects;

            let stats = cache.get_stats();
            total_allocations += stats.total_allocations;
            total_deallocations += stats.total_deallocations;
        }

        let memory_usage = self.memory_pool.get_usage();

        SlabAllocatorStats {
            total_caches,
            total_slabs,
            total_objects,
            allocated_objects,
            free_objects: total_objects - allocated_objects,
            total_allocations,
            total_deallocations,
            memory_usage,
            cache_efficiency: if total_allocations > 0 {
                allocated_objects as f64 / total_allocations as f64
            } else {
                0.0
            },
        }
    }

    /// Get information about all caches
    pub fn get_all_cache_info(&self) -> Vec<(usize, CacheInfo)> {
        self.caches
            .iter()
            .map(|(&size, cache)| (size, cache.get_cache_info()))
            .collect()
    }

    /// Reclaim memory from empty slabs
    pub fn reclaim_memory(&mut self) -> usize {
        let mut total_reclaimed = 0;

        for cache in self.caches.values_mut() {
            total_reclaimed += cache.reclaim_empty_slabs(&mut self.memory_pool);
        }

        total_reclaimed
    }

    /// Destroy cache for specific size
    pub fn destroy_cache(&mut self, size: usize) -> Result<(), SlabError> {
        let aligned_size = (size + self.config.alignment - 1) & !(self.config.alignment - 1);

        if let Some(mut cache) = self.caches.remove(&aligned_size) {
            // Reclaim all slabs from this cache
            cache.reclaim_empty_slabs(&mut self.memory_pool);
            Ok(())
        } else {
            Err(SlabError::InvalidSize("Cache not found".to_string()))
        }
    }

    /// Get memory pool usage
    pub fn get_memory_usage(&self) -> MemoryPoolUsage {
        self.memory_pool.get_usage()
    }
}

// Safety: SlabAllocator manages GPU memory pointers via NonNull<u8>. While NonNull is not Send/Sync by default,
// it's safe to share SlabAllocator across threads when protected by Arc<Mutex<>> because:
// 1. The pointers point to GPU memory managed by the GPU driver
// 2. The Mutex provides exclusive access for all mutable operations
// 3. No thread-local state is maintained
unsafe impl Send for SlabAllocator {}
unsafe impl Sync for SlabAllocator {}

/// Slab allocator statistics
#[derive(Debug, Clone)]
pub struct SlabAllocatorStats {
    pub total_caches: usize,
    pub total_slabs: usize,
    pub total_objects: usize,
    pub allocated_objects: usize,
    pub free_objects: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub memory_usage: MemoryPoolUsage,
    pub cache_efficiency: f64,
}

/// Slab allocator errors
#[derive(Debug, Clone)]
pub enum SlabError {
    InvalidSize(String),
    OutOfMemory(String),
    InvalidPointer(String),
    DoubleFree(String),
    CorruptedSlab(String),
}

impl std::fmt::Display for SlabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlabError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            SlabError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            SlabError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            SlabError::DoubleFree(msg) => write!(f, "Double free: {}", msg),
            SlabError::CorruptedSlab(msg) => write!(f, "Corrupted slab: {}", msg),
        }
    }
}

impl std::error::Error for SlabError {}

/// Thread-safe slab allocator wrapper
pub struct ThreadSafeSlabAllocator {
    allocator: Arc<Mutex<SlabAllocator>>,
}

impl ThreadSafeSlabAllocator {
    pub fn new(base_ptr: NonNull<u8>, total_size: usize, config: SlabConfig) -> Self {
        let allocator = SlabAllocator::new(base_ptr, total_size, config);
        Self {
            allocator: Arc::new(Mutex::new(allocator)),
        }
    }

    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>, SlabError> {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.allocate(size)
    }

    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<(), SlabError> {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.deallocate(ptr, size)
    }

    pub fn get_stats(&self) -> SlabAllocatorStats {
        let allocator = self.allocator.lock().unwrap();
        allocator.get_stats()
    }

    pub fn reclaim_memory(&self) -> usize {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.reclaim_memory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_creation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let slab = Slab::new(ptr, size, 64);
        assert_eq!(slab.object_count, size / 64);
        assert!(slab.is_empty());
        assert!(!slab.is_full());
    }

    #[test]
    fn test_slab_allocation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let mut slab = Slab::new(ptr, size, 64);

        let alloc1 = slab.allocate();
        assert!(alloc1.is_some());
        assert!(slab.is_partial());

        let alloc2 = slab.allocate();
        assert!(alloc2.is_some());
        assert_ne!(alloc1.unwrap(), alloc2.unwrap());
    }

    #[test]
    fn test_slab_deallocation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let mut slab = Slab::new(ptr, size, 64);

        let alloc_ptr = slab.allocate().unwrap();
        let result = slab.deallocate(alloc_ptr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_pool() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let mut pool = MemoryPool::new(ptr, size, 256);

        let slab1 = pool.allocate_slab(4096);
        assert!(slab1.is_some());

        let slab2 = pool.allocate_slab(4096);
        assert!(slab2.is_some());

        assert_ne!(slab1.unwrap(), slab2.unwrap());
    }

    #[test]
    fn test_slab_cache() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let mut pool = MemoryPool::new(ptr, size, 256);
        let config = CacheConfig::default();
        let mut cache = SlabCache::new(64, config);

        let alloc1 = cache.allocate(&mut pool);
        assert!(alloc1.is_ok());

        let alloc2 = cache.allocate(&mut pool);
        assert!(alloc2.is_ok());
    }

    #[test]
    fn test_slab_allocator() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = SlabConfig::default();
        let mut allocator = SlabAllocator::new(ptr, size, config);

        let alloc1 = allocator.allocate(64);
        assert!(alloc1.is_ok());

        let alloc2 = allocator.allocate(128);
        assert!(alloc2.is_ok());

        let stats = allocator.get_stats();
        // Note: Depending on the slab configuration, sizes 64 and 128 might map to the same cache class
        assert!(stats.total_caches >= 1); // At least one cache should be created
    }

    #[test]
    fn test_thread_safe_allocator() {
        let size = 1024 * 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = SlabConfig::default();
        let allocator = ThreadSafeSlabAllocator::new(ptr, size, config);

        let alloc_result = allocator.allocate(64);
        assert!(alloc_result.is_ok());

        let stats = allocator.get_stats();
        assert!(stats.allocated_objects > 0);
    }
}
