// Arena allocator for GPU memory management
//
// This module implements arena (linear) allocators that allocate objects
// sequentially from a contiguous block of memory. Arena allocators are
// extremely fast for allocation and are ideal for temporary allocations
// that can be freed all at once.

#[allow(dead_code)]
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Main arena allocator implementation
pub struct ArenaAllocator {
    /// Base pointer of the arena
    base_ptr: NonNull<u8>,
    /// Total size of the arena
    total_size: usize,
    /// Current allocation offset
    current_offset: usize,
    /// High water mark (maximum offset reached)
    high_water_mark: usize,
    /// Memory alignment requirement
    alignment: usize,
    /// Arena configuration
    config: ArenaConfig,
    /// Allocation tracking (if enabled)
    allocations: Vec<AllocationRecord>,
    /// Statistics
    stats: ArenaStats,
    /// Checkpoints for nested scopes
    checkpoints: Vec<ArenaCheckpoint>,
}

/// Arena allocation record
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Pointer to the allocation
    pub ptr: NonNull<u8>,
    /// Size of the allocation
    pub size: usize,
    /// Offset from base
    pub offset: usize,
    /// Timestamp of allocation
    pub allocated_at: Instant,
    /// Allocation ID for debugging
    pub id: u64,
    /// Optional debug tag
    pub tag: Option<String>,
}

/// Arena checkpoint for nested scopes
#[derive(Debug, Clone)]
pub struct ArenaCheckpoint {
    /// Offset at checkpoint creation
    pub offset: usize,
    /// Number of allocations at checkpoint
    pub allocation_count: usize,
    /// Timestamp of checkpoint creation
    pub created_at: Instant,
    /// Optional checkpoint name
    pub name: Option<String>,
}

/// Arena allocator configuration
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Memory alignment (must be power of 2)
    pub alignment: usize,
    /// Enable allocation tracking
    pub enable_tracking: bool,
    /// Enable debug mode with extra checks
    pub enable_debug: bool,
    /// Enable checkpoint support
    pub enable_checkpoints: bool,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Growth strategy for resizable arenas
    pub growth_strategy: GrowthStrategy,
    /// Initial capacity for allocation tracking
    pub initial_tracking_capacity: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            alignment: 8,
            enable_tracking: false,
            enable_debug: false,
            enable_checkpoints: true,
            enable_stats: true,
            growth_strategy: GrowthStrategy::Fixed,
            initial_tracking_capacity: 1024,
        }
    }
}

/// Growth strategy for resizable arenas
#[derive(Debug, Clone, PartialEq)]
pub enum GrowthStrategy {
    /// Fixed size arena (no growth)
    Fixed,
    /// Double the size when full
    Double,
    /// Linear growth by fixed amount
    Linear(usize),
    /// Custom growth function
    Custom(fn(usize) -> usize),
}

/// Arena allocator statistics
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total number of allocations
    pub total_allocations: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Current bytes allocated
    pub current_bytes_allocated: usize,
    /// Peak bytes allocated
    pub peak_bytes_allocated: usize,
    /// Number of resets
    pub reset_count: u64,
    /// Number of checkpoint operations
    pub checkpoint_count: u64,
    /// Number of rollback operations
    pub rollback_count: u64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Memory utilization ratio
    pub utilization_ratio: f64,
    /// Time of first allocation
    pub first_allocation_time: Option<Instant>,
    /// Time of last allocation
    pub last_allocation_time: Option<Instant>,
}

impl ArenaStats {
    pub fn record_allocation(&mut self, size: usize) {
        let now = Instant::now();

        self.total_allocations += 1;
        self.total_bytes_allocated += size as u64;
        self.current_bytes_allocated += size;

        if self.current_bytes_allocated > self.peak_bytes_allocated {
            self.peak_bytes_allocated = self.current_bytes_allocated;
        }

        // Update average allocation size
        self.average_allocation_size =
            self.total_bytes_allocated as f64 / self.total_allocations as f64;

        // Update allocation rate
        if let Some(first_time) = self.first_allocation_time {
            let elapsed = now.duration_since(first_time).as_secs_f64();
            if elapsed > 0.0 {
                self.allocation_rate = self.total_allocations as f64 / elapsed;
            }
        } else {
            self.first_allocation_time = Some(now);
        }

        self.last_allocation_time = Some(now);
    }

    pub fn record_reset(&mut self) {
        self.reset_count += 1;
        self.current_bytes_allocated = 0;
    }

    pub fn record_checkpoint(&mut self) {
        self.checkpoint_count += 1;
    }

    pub fn record_rollback(&mut self, bytes_freed: usize) {
        self.rollback_count += 1;
        self.current_bytes_allocated = self.current_bytes_allocated.saturating_sub(bytes_freed);
    }

    pub fn update_utilization(&mut self, total_size: usize) {
        if total_size > 0 {
            self.utilization_ratio = self.current_bytes_allocated as f64 / total_size as f64;
        }
    }
}

impl ArenaAllocator {
    /// Create a new arena allocator
    pub fn new(
        base_ptr: NonNull<u8>,
        size: usize,
        config: ArenaConfig,
    ) -> Result<Self, ArenaError> {
        if size == 0 {
            return Err(ArenaError::InvalidSize(
                "Arena size cannot be zero".to_string(),
            ));
        }

        if !config.alignment.is_power_of_two() {
            return Err(ArenaError::InvalidAlignment(format!(
                "Alignment {} is not a power of two",
                config.alignment
            )));
        }

        let allocations = if config.enable_tracking {
            Vec::with_capacity(config.initial_tracking_capacity)
        } else {
            Vec::new()
        };

        Ok(Self {
            base_ptr,
            total_size: size,
            current_offset: 0,
            high_water_mark: 0,
            alignment: config.alignment,
            allocations,
            stats: ArenaStats::default(),
            checkpoints: Vec::new(),
            config,
        })
    }

    /// Allocate memory from the arena
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, ArenaError> {
        if size == 0 {
            return Err(ArenaError::InvalidSize(
                "Cannot allocate zero bytes".to_string(),
            ));
        }

        // Align the size
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        // Check if we have enough space
        if self.current_offset + aligned_size > self.total_size {
            return Err(ArenaError::OutOfMemory(format!(
                "Not enough space: need {}, have {}",
                aligned_size,
                self.total_size - self.current_offset
            )));
        }

        // Calculate the pointer
        let ptr =
            unsafe { NonNull::new_unchecked(self.base_ptr.as_ptr().add(self.current_offset)) };

        // Update state
        self.current_offset += aligned_size;
        if self.current_offset > self.high_water_mark {
            self.high_water_mark = self.current_offset;
        }

        // Record allocation
        if self.config.enable_tracking {
            let record = AllocationRecord {
                ptr,
                size: aligned_size,
                offset: self.current_offset - aligned_size,
                allocated_at: Instant::now(),
                id: self.stats.total_allocations,
                tag: None,
            };
            self.allocations.push(record);
        }

        // Update statistics
        if self.config.enable_stats {
            self.stats.record_allocation(aligned_size);
            self.stats.update_utilization(self.total_size);
        }

        Ok(ptr)
    }

    /// Allocate memory with a debug tag
    pub fn allocate_tagged(&mut self, size: usize, tag: String) -> Result<NonNull<u8>, ArenaError> {
        let ptr = self.allocate(size)?;

        if self.config.enable_tracking && !self.allocations.is_empty() {
            let last_idx = self.allocations.len() - 1;
            self.allocations[last_idx].tag = Some(tag);
        }

        Ok(ptr)
    }

    /// Allocate aligned memory
    pub fn allocate_aligned(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<NonNull<u8>, ArenaError> {
        if !alignment.is_power_of_two() {
            return Err(ArenaError::InvalidAlignment(format!(
                "Alignment {} is not a power of two",
                alignment
            )));
        }

        // Calculate aligned offset
        let aligned_offset = (self.current_offset + alignment - 1) & !(alignment - 1);
        let padding = aligned_offset - self.current_offset;

        // Check if we have enough space including padding
        if aligned_offset + size > self.total_size {
            return Err(ArenaError::OutOfMemory(format!(
                "Not enough space for aligned allocation: need {}, have {}",
                aligned_offset + size - self.current_offset,
                self.total_size - self.current_offset
            )));
        }

        // Update offset to aligned position
        self.current_offset = aligned_offset;

        // Now allocate normally
        self.allocate(size)
    }

    /// Reset the arena to empty state
    pub fn reset(&mut self) {
        self.current_offset = 0;

        if self.config.enable_tracking {
            self.allocations.clear();
        }

        if self.config.enable_stats {
            self.stats.record_reset();
            self.stats.update_utilization(self.total_size);
        }

        self.checkpoints.clear();
    }

    /// Create a checkpoint for later rollback
    pub fn checkpoint(&mut self) -> Result<CheckpointHandle, ArenaError> {
        if !self.config.enable_checkpoints {
            return Err(ArenaError::CheckpointsDisabled);
        }

        let checkpoint = ArenaCheckpoint {
            offset: self.current_offset,
            allocation_count: self.allocations.len(),
            created_at: Instant::now(),
            name: None,
        };

        self.checkpoints.push(checkpoint);

        if self.config.enable_stats {
            self.stats.record_checkpoint();
        }

        Ok(CheckpointHandle {
            index: self.checkpoints.len() - 1,
            offset: self.current_offset,
        })
    }

    /// Create a named checkpoint
    pub fn checkpoint_named(&mut self, name: String) -> Result<CheckpointHandle, ArenaError> {
        if !self.config.enable_checkpoints {
            return Err(ArenaError::CheckpointsDisabled);
        }

        let checkpoint = ArenaCheckpoint {
            offset: self.current_offset,
            allocation_count: self.allocations.len(),
            created_at: Instant::now(),
            name: Some(name),
        };

        self.checkpoints.push(checkpoint);

        if self.config.enable_stats {
            self.stats.record_checkpoint();
        }

        Ok(CheckpointHandle {
            index: self.checkpoints.len() - 1,
            offset: self.current_offset,
        })
    }

    /// Rollback to a checkpoint
    pub fn rollback(&mut self, handle: CheckpointHandle) -> Result<(), ArenaError> {
        if !self.config.enable_checkpoints {
            return Err(ArenaError::CheckpointsDisabled);
        }

        if handle.index >= self.checkpoints.len() {
            return Err(ArenaError::InvalidCheckpoint(
                "Checkpoint index out of range".to_string(),
            ));
        }

        let checkpoint = &self.checkpoints[handle.index];
        let bytes_freed = self.current_offset - checkpoint.offset;

        // Rollback state
        self.current_offset = checkpoint.offset;

        if self.config.enable_tracking {
            self.allocations.truncate(checkpoint.allocation_count);
        }

        // Remove checkpoints created after this one
        self.checkpoints.truncate(handle.index);

        if self.config.enable_stats {
            self.stats.record_rollback(bytes_freed);
            self.stats.update_utilization(self.total_size);
        }

        Ok(())
    }

    /// Get current usage information
    pub fn get_usage(&self) -> ArenaUsage {
        ArenaUsage {
            total_size: self.total_size,
            used_size: self.current_offset,
            free_size: self.total_size - self.current_offset,
            high_water_mark: self.high_water_mark,
            allocation_count: self.allocations.len(),
            checkpoint_count: self.checkpoints.len(),
            utilization_ratio: self.current_offset as f64 / self.total_size as f64,
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> &ArenaStats {
        &self.stats
    }

    /// Get allocation records (if tracking enabled)
    pub fn get_allocations(&self) -> &[AllocationRecord] {
        &self.allocations
    }

    /// Get checkpoints
    pub fn get_checkpoints(&self) -> &[ArenaCheckpoint] {
        &self.checkpoints
    }

    /// Check if a pointer belongs to this arena
    pub fn contains_pointer(&self, ptr: NonNull<u8>) -> bool {
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base_ptr.as_ptr() as usize;

        ptr_addr >= base_addr && ptr_addr < base_addr + self.current_offset
    }

    /// Get allocation info for a pointer (if tracking enabled)
    pub fn get_allocation_info(&self, ptr: NonNull<u8>) -> Option<&AllocationRecord> {
        if !self.config.enable_tracking {
            return None;
        }

        self.allocations.iter().find(|record| record.ptr == ptr)
    }

    /// Validate arena consistency
    pub fn validate(&self) -> Result<(), ArenaError> {
        if self.current_offset > self.total_size {
            return Err(ArenaError::CorruptedArena(format!(
                "Current offset {} exceeds total size {}",
                self.current_offset, self.total_size
            )));
        }

        if self.high_water_mark > self.total_size {
            return Err(ArenaError::CorruptedArena(format!(
                "High water mark {} exceeds total size {}",
                self.high_water_mark, self.total_size
            )));
        }

        if self.high_water_mark < self.current_offset {
            return Err(ArenaError::CorruptedArena(format!(
                "High water mark {} is less than current offset {}",
                self.high_water_mark, self.current_offset
            )));
        }

        // Validate tracking records if enabled
        if self.config.enable_tracking {
            let mut total_tracked_size = 0;

            for (i, record) in self.allocations.iter().enumerate() {
                // Check pointer is within arena bounds
                if !self.contains_pointer(record.ptr) {
                    return Err(ArenaError::CorruptedArena(format!(
                        "Allocation {} has pointer outside arena bounds",
                        i
                    )));
                }

                total_tracked_size += record.size;
            }

            // Note: total_tracked_size might be less than current_offset due to alignment padding
            if total_tracked_size > self.current_offset {
                return Err(ArenaError::CorruptedArena(format!(
                    "Tracked size {} exceeds current offset {}",
                    total_tracked_size, self.current_offset
                )));
            }
        }

        Ok(())
    }

    /// Get memory layout information
    pub fn get_memory_layout(&self) -> MemoryLayout {
        let mut layout = MemoryLayout {
            base_address: self.base_ptr.as_ptr() as usize,
            total_size: self.total_size,
            used_size: self.current_offset,
            regions: Vec::new(),
        };

        if self.config.enable_tracking {
            for record in &self.allocations {
                layout.regions.push(MemoryRegion {
                    offset: record.offset,
                    size: record.size,
                    allocated_at: record.allocated_at,
                    tag: record.tag.clone(),
                });
            }
        }

        layout
    }
}

/// Checkpoint handle for rollback operations
#[derive(Debug, Clone)]
pub struct CheckpointHandle {
    index: usize,
    offset: usize,
}

/// Arena usage information
#[derive(Debug, Clone)]
pub struct ArenaUsage {
    pub total_size: usize,
    pub used_size: usize,
    pub free_size: usize,
    pub high_water_mark: usize,
    pub allocation_count: usize,
    pub checkpoint_count: usize,
    pub utilization_ratio: f64,
}

/// Memory layout information
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub base_address: usize,
    pub total_size: usize,
    pub used_size: usize,
    pub regions: Vec<MemoryRegion>,
}

/// Memory region within arena
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub offset: usize,
    pub size: usize,
    pub allocated_at: Instant,
    pub tag: Option<String>,
}

/// Ring buffer arena allocator for circular allocation patterns
pub struct RingArena {
    arena: ArenaAllocator,
    /// Read pointer for ring buffer
    read_offset: usize,
    /// Number of live allocations
    live_allocations: usize,
    /// Ring configuration
    ring_config: RingConfig,
}

/// Ring arena configuration
#[derive(Debug, Clone)]
pub struct RingConfig {
    /// Enable overwrite protection
    pub overwrite_protection: bool,
    /// Callback when data is overwritten
    pub overwrite_callback: Option<fn(*mut u8, usize)>,
    /// Enable statistics
    pub enable_stats: bool,
}

impl Default for RingConfig {
    fn default() -> Self {
        Self {
            overwrite_protection: true,
            overwrite_callback: None,
            enable_stats: true,
        }
    }
}

impl RingArena {
    pub fn new(
        base_ptr: NonNull<u8>,
        size: usize,
        ring_config: RingConfig,
    ) -> Result<Self, ArenaError> {
        let arena_config = ArenaConfig {
            enable_tracking: ring_config.enable_stats,
            enable_checkpoints: false,
            ..ArenaConfig::default()
        };

        let arena = ArenaAllocator::new(base_ptr, size, arena_config)?;

        Ok(Self {
            arena,
            read_offset: 0,
            live_allocations: 0,
            ring_config,
        })
    }

    /// Allocate from ring buffer
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, ArenaError> {
        // Check if allocation would wrap around and collide with live data
        if self.ring_config.overwrite_protection {
            let aligned_size = (size + self.arena.alignment - 1) & !(self.arena.alignment - 1);

            if self.arena.current_offset + aligned_size > self.arena.total_size {
                // Would wrap around
                if self.read_offset > 0 && aligned_size > self.read_offset {
                    return Err(ArenaError::RingBufferFull(
                        "Ring buffer full, would overwrite live data".to_string(),
                    ));
                }

                // Safe to wrap
                self.arena.current_offset = 0;
            } else if self.read_offset > self.arena.current_offset {
                // Normal case, check collision
                if self.arena.current_offset + aligned_size > self.read_offset {
                    return Err(ArenaError::RingBufferFull(
                        "Ring buffer full, would overwrite live data".to_string(),
                    ));
                }
            }
        }

        let ptr = self.arena.allocate(size)?;
        self.live_allocations += 1;

        Ok(ptr)
    }

    /// Mark data as consumed (advance read pointer)
    pub fn consume(&mut self, size: usize) -> Result<(), ArenaError> {
        let aligned_size = (size + self.arena.alignment - 1) & !(self.arena.alignment - 1);

        if self.read_offset + aligned_size > self.arena.total_size {
            // Wrap around
            self.read_offset = aligned_size - (self.arena.total_size - self.read_offset);
        } else {
            self.read_offset += aligned_size;
        }

        self.live_allocations = self.live_allocations.saturating_sub(1);

        Ok(())
    }

    /// Reset ring buffer
    pub fn reset(&mut self) {
        self.arena.reset();
        self.read_offset = 0;
        self.live_allocations = 0;
    }

    /// Get ring buffer usage
    pub fn get_ring_usage(&self) -> RingUsage {
        let total_size = self.arena.total_size;
        let write_offset = self.arena.current_offset;

        let used_size = if write_offset >= self.read_offset {
            write_offset - self.read_offset
        } else {
            total_size - self.read_offset + write_offset
        };

        RingUsage {
            total_size,
            used_size,
            free_size: total_size - used_size,
            read_offset: self.read_offset,
            write_offset,
            live_allocations: self.live_allocations,
        }
    }
}

/// Ring buffer usage information
#[derive(Debug, Clone)]
pub struct RingUsage {
    pub total_size: usize,
    pub used_size: usize,
    pub free_size: usize,
    pub read_offset: usize,
    pub write_offset: usize,
    pub live_allocations: usize,
}

/// Growing arena that can expand its capacity
pub struct GrowingArena {
    /// Current arena
    current_arena: ArenaAllocator,
    /// Previous arenas (for lookups)
    previous_arenas: Vec<ArenaAllocator>,
    /// Growth strategy
    growth_strategy: GrowthStrategy,
    /// External memory allocator for growth
    external_allocator: Option<Box<dyn ExternalAllocator>>,
}

/// External allocator trait for growing arenas
pub trait ExternalAllocator {
    fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, ArenaError>;
    fn deallocate(&mut self, ptr: NonNull<u8>, size: usize);
}

impl GrowingArena {
    pub fn new(
        base_ptr: NonNull<u8>,
        initial_size: usize,
        growth_strategy: GrowthStrategy,
    ) -> Result<Self, ArenaError> {
        let config = ArenaConfig::default();
        let arena = ArenaAllocator::new(base_ptr, initial_size, config)?;

        Ok(Self {
            current_arena: arena,
            previous_arenas: Vec::new(),
            growth_strategy,
            external_allocator: None,
        })
    }

    pub fn with_external_allocator(mut self, allocator: Box<dyn ExternalAllocator>) -> Self {
        self.external_allocator = Some(allocator);
        self
    }

    /// Allocate with automatic growth
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, ArenaError> {
        // Try current arena first
        match self.current_arena.allocate(size) {
            Ok(ptr) => Ok(ptr),
            Err(ArenaError::OutOfMemory(_)) => {
                // Need to grow
                self.grow(size)?;
                self.current_arena.allocate(size)
            }
            Err(e) => Err(e),
        }
    }

    fn grow(&mut self, min_additional_size: usize) -> Result<(), ArenaError> {
        if self.external_allocator.is_none() {
            return Err(ArenaError::CannotGrow(
                "No external allocator configured".to_string(),
            ));
        }

        let current_size = self.current_arena.total_size;
        let new_size = match &self.growth_strategy {
            GrowthStrategy::Fixed => {
                return Err(ArenaError::CannotGrow("Fixed size arena".to_string()))
            }
            GrowthStrategy::Double => current_size * 2,
            GrowthStrategy::Linear(increment) => current_size + increment,
            GrowthStrategy::Custom(func) => func(current_size),
        };

        let actual_new_size = new_size.max(min_additional_size);

        let new_ptr = self
            .external_allocator
            .as_mut()
            .unwrap()
            .allocate(actual_new_size)?;

        // Move current arena to previous arenas
        let old_arena = std::mem::replace(
            &mut self.current_arena,
            ArenaAllocator::new(new_ptr, actual_new_size, ArenaConfig::default())?,
        );

        self.previous_arenas.push(old_arena);

        Ok(())
    }

    /// Check if pointer belongs to any arena
    pub fn contains_pointer(&self, ptr: NonNull<u8>) -> bool {
        if self.current_arena.contains_pointer(ptr) {
            return true;
        }

        self.previous_arenas
            .iter()
            .any(|arena| arena.contains_pointer(ptr))
    }

    /// Get total usage across all arenas
    pub fn get_total_usage(&self) -> GrowingArenaUsage {
        let mut total_size = self.current_arena.total_size;
        let mut used_size = self.current_arena.current_offset;
        let mut allocation_count = self.current_arena.allocations.len();

        for arena in &self.previous_arenas {
            total_size += arena.total_size;
            used_size += arena.current_offset;
            allocation_count += arena.allocations.len();
        }

        GrowingArenaUsage {
            total_size,
            used_size,
            free_size: total_size - used_size,
            arena_count: 1 + self.previous_arenas.len(),
            allocation_count,
            current_arena_size: self.current_arena.total_size,
            utilization_ratio: used_size as f64 / total_size as f64,
        }
    }
}

/// Growing arena usage information
#[derive(Debug, Clone)]
pub struct GrowingArenaUsage {
    pub total_size: usize,
    pub used_size: usize,
    pub free_size: usize,
    pub arena_count: usize,
    pub allocation_count: usize,
    pub current_arena_size: usize,
    pub utilization_ratio: f64,
}

/// Arena allocator errors
#[derive(Debug, Clone)]
pub enum ArenaError {
    InvalidSize(String),
    InvalidAlignment(String),
    OutOfMemory(String),
    CheckpointsDisabled,
    InvalidCheckpoint(String),
    CorruptedArena(String),
    RingBufferFull(String),
    CannotGrow(String),
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArenaError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            ArenaError::InvalidAlignment(msg) => write!(f, "Invalid alignment: {}", msg),
            ArenaError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            ArenaError::CheckpointsDisabled => write!(f, "Checkpoints are disabled"),
            ArenaError::InvalidCheckpoint(msg) => write!(f, "Invalid checkpoint: {}", msg),
            ArenaError::CorruptedArena(msg) => write!(f, "Corrupted arena: {}", msg),
            ArenaError::RingBufferFull(msg) => write!(f, "Ring buffer full: {}", msg),
            ArenaError::CannotGrow(msg) => write!(f, "Cannot grow: {}", msg),
        }
    }
}

impl std::error::Error for ArenaError {}

/// Thread-safe arena allocator wrapper
pub struct ThreadSafeArena {
    arena: Arc<Mutex<ArenaAllocator>>,
}

impl ThreadSafeArena {
    pub fn new(
        base_ptr: NonNull<u8>,
        size: usize,
        config: ArenaConfig,
    ) -> Result<Self, ArenaError> {
        let arena = ArenaAllocator::new(base_ptr, size, config)?;
        Ok(Self {
            arena: Arc::new(Mutex::new(arena)),
        })
    }

    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>, ArenaError> {
        let mut arena = self.arena.lock().unwrap();
        arena.allocate(size)
    }

    pub fn reset(&self) {
        let mut arena = self.arena.lock().unwrap();
        arena.reset();
    }

    pub fn checkpoint(&self) -> Result<CheckpointHandle, ArenaError> {
        let mut arena = self.arena.lock().unwrap();
        arena.checkpoint()
    }

    pub fn rollback(&self, handle: CheckpointHandle) -> Result<(), ArenaError> {
        let mut arena = self.arena.lock().unwrap();
        arena.rollback(handle)
    }

    pub fn get_usage(&self) -> ArenaUsage {
        let arena = self.arena.lock().unwrap();
        arena.get_usage()
    }

    pub fn get_stats(&self) -> ArenaStats {
        let arena = self.arena.lock().unwrap();
        arena.get_stats().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig::default();
        let arena = ArenaAllocator::new(ptr, size, config);
        assert!(arena.is_ok());
    }

    #[test]
    fn test_basic_allocation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig::default();
        let mut arena = ArenaAllocator::new(ptr, size, config).unwrap();

        let alloc1 = arena.allocate(100);
        assert!(alloc1.is_ok());

        let alloc2 = arena.allocate(200);
        assert!(alloc2.is_ok());

        let usage = arena.get_usage();
        assert!(usage.used_size > 0);
        assert!(usage.allocation_count == 2 || !arena.config.enable_tracking);
    }

    #[test]
    fn test_alignment() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig {
            alignment: 16,
            ..ArenaConfig::default()
        };
        let mut arena = ArenaAllocator::new(ptr, size, config).unwrap();

        let alloc_ptr = arena.allocate(10).unwrap();
        assert_eq!(alloc_ptr.as_ptr() as usize % 16, 0);
    }

    #[test]
    fn test_checkpoints() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig {
            enable_checkpoints: true,
            enable_tracking: true,
            ..ArenaConfig::default()
        };
        let mut arena = ArenaAllocator::new(ptr, size, config).unwrap();

        arena.allocate(100).unwrap();
        let checkpoint = arena.checkpoint().unwrap();
        arena.allocate(200).unwrap();

        let usage_before = arena.get_usage();
        arena.rollback(checkpoint).unwrap();
        let usage_after = arena.get_usage();

        assert!(usage_after.used_size < usage_before.used_size);
    }

    #[test]
    fn test_reset() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig::default();
        let mut arena = ArenaAllocator::new(ptr, size, config).unwrap();

        arena.allocate(100).unwrap();
        arena.allocate(200).unwrap();

        let usage_before = arena.get_usage();
        assert!(usage_before.used_size > 0);

        arena.reset();
        let usage_after = arena.get_usage();
        assert_eq!(usage_after.used_size, 0);
    }

    #[test]
    fn test_ring_arena() {
        let size = 1024;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = RingConfig::default();
        let mut ring = RingArena::new(ptr, size, config).unwrap();

        let alloc1 = ring.allocate(100);
        assert!(alloc1.is_ok());

        ring.consume(100).unwrap();

        let alloc2 = ring.allocate(100);
        assert!(alloc2.is_ok());
    }

    #[test]
    fn test_thread_safe_arena() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig::default();
        let arena = ThreadSafeArena::new(ptr, size, config).unwrap();

        let alloc_result = arena.allocate(100);
        assert!(alloc_result.is_ok());

        let usage = arena.get_usage();
        assert!(usage.used_size > 0);
    }

    #[test]
    fn test_arena_validation() {
        let size = 4096;
        let memory = vec![0u8; size];
        let ptr = NonNull::new(memory.as_ptr() as *mut u8).unwrap();

        let config = ArenaConfig {
            enable_tracking: true,
            ..ArenaConfig::default()
        };
        let mut arena = ArenaAllocator::new(ptr, size, config).unwrap();

        arena.allocate(100).unwrap();
        arena.allocate(200).unwrap();

        let validation_result = arena.validate();
        assert!(validation_result.is_ok());
    }
}
