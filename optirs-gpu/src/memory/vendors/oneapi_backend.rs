// OneAPI backend for GPU memory management
//
// This module provides Intel OneAPI/SYCL-specific memory management functionality,
// including device memory allocation, SYCL queues, and performance optimization
// features specific to Intel GPUs and accelerators.
//
// # This is a host-memory simulation, not real OneAPI
//
// `optirs-gpu` is Pure Rust with no FFI dependencies by default, and this
// crate ships no SYCL/Level Zero runtime bindings. There is therefore no
// real `malloc_device`/`malloc_shared` underneath this module: "device",
// "host", "shared" and "system" USM allocations are all the *same*
// system-heap allocation (see `sim_alloc`/`sim_dealloc` below), and
// `SyclDeviceProperties`/`OneApiStats` are example numbers, not a query of
// real hardware. This module models the SYCL USM memory-management *API
// shape* for testing that shape in isolation; treat every allocation as host
// memory and every device number as illustrative.
//
// This extends to data movement: `usm_memcpy` copies **zero bytes**. It
// builds a `SyclOperation` record, hands it to
// `SyclQueueManager::execute_operation` (which returns immediately and never
// dereferences `src_ptr`/`dst_ptr`), and increments
// `OneApiStats::memory_transfers` — that counter says "this many
// `usm_memcpy` calls were made," not "this many bytes moved." An earlier
// revision of this module also injected a `std::thread::sleep` here to
// imitate transfer/kernel-submission latency by operation kind; that fake
// timing has been removed, so the distinction between
// `MemcpyHostToDevice`/`MemcpyDeviceToHost`/`MemcpyDeviceToDevice`/
// `UsmMemcpy`/`KernelSubmit` no longer affects anything observable.
// `OneApiStats::queue_operations` and `OneApiStats::kernel_submissions` are
// declared for API-shape completeness but nothing in this module ever
// increments them — read a `0` there as "not tracked," not "none occurred."

#[allow(dead_code)]
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Upper bound this module's simulated allocator supports for a caller-
/// requested alignment (both `sycl_malloc`'s `usm_properties.alignment` and
/// `direct_allocate`'s local `64` are well under this in practice). Also the
/// fixed header size reserved ahead of every non-empty allocation, so a
/// request with alignment `<= SIM_ALLOC_ALIGN` needs no per-call alignment
/// bookkeeping: over-aligning to `SIM_ALLOC_ALIGN` always satisfies a
/// smaller request too.
const SIM_ALLOC_ALIGN: usize = 256;

/// Allocate `size` bytes at (at least) `align`-byte alignment through the
/// system allocator, without the two ways the naive
/// `std::alloc::alloc(Layout::from_size_align_unchecked(size, align))` this
/// module used to call was undefined behaviour: a zero-size layout is
/// unsound to pass to `GlobalAlloc::alloc`, and a real allocation failure
/// returns null, which must never be treated as valid memory. See
/// `cuda_backend::sim_alloc` for the full rationale (this mirrors it, with
/// an added alignment check since this module's alignment is caller-chosen
/// rather than a fixed constant).
///
/// The payload is prefixed with a `SIM_ALLOC_ALIGN`-byte header recording
/// the requested size, so [`sim_dealloc`] can reconstruct the exact `Layout`
/// this function used — the `Layout::from_size_align_unchecked(1, 1)` this
/// module used at free time was a mismatched-layout deallocation, itself
/// unconditionally undefined behaviour.
fn sim_alloc(size: usize, align: usize) -> Result<*mut c_void, OneApiError> {
    if !align.is_power_of_two() || align > SIM_ALLOC_ALIGN {
        return Err(OneApiError::OutOfMemory(format!(
            "unsupported allocation alignment {align}: this simulated backend supports \
             power-of-two alignments up to {SIM_ALLOC_ALIGN} bytes"
        )));
    }
    if size == 0 {
        return Ok(SIM_ALLOC_ALIGN as *mut c_void);
    }
    let total = SIM_ALLOC_ALIGN.checked_add(size).ok_or_else(|| {
        OneApiError::OutOfMemory(format!(
            "{size}-byte request overflows the allocator's size limit"
        ))
    })?;
    // Always allocate at `SIM_ALLOC_ALIGN`: since `align <= SIM_ALLOC_ALIGN`
    // and both are powers of two, this over-aligned block also satisfies the
    // caller's smaller request.
    let layout = std::alloc::Layout::from_size_align(total, SIM_ALLOC_ALIGN)
        .map_err(|e| OneApiError::OutOfMemory(format!("invalid allocation layout: {e}")))?;
    // SAFETY: `layout` has non-zero size (checked above) and a valid
    // (power-of-two) alignment constructed by `Layout::from_size_align`.
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return Err(OneApiError::OutOfMemory(format!(
            "allocator returned null for a {size}-byte request"
        )));
    }
    // SAFETY: `base` is non-null and `layout`'s size is at least
    // `SIM_ALLOC_ALIGN + size >= SIM_ALLOC_ALIGN >= size_of::<usize>()`, so
    // writing one `usize` at the start of the block is in-bounds.
    unsafe { (base as *mut usize).write(size) };
    // SAFETY: `base` was allocated with `total = SIM_ALLOC_ALIGN + size`
    // bytes, so offsetting by `SIM_ALLOC_ALIGN` stays within (or one past)
    // the allocation.
    Ok(unsafe { base.add(SIM_ALLOC_ALIGN) } as *mut c_void)
}

/// Free a pointer returned by [`sim_alloc`]. A no-op for a null pointer or
/// the zero-size sentinel — neither was ever allocated.
///
/// Only ever called from this module (it is not `pub`) with a pointer
/// `sim_alloc` returned that has not already been freed — the unsafety of
/// the pointer arithmetic below is contained to that invariant, matching
/// this module's existing style of confining `unsafe` to the raw
/// `std::alloc` calls rather than marking `free()`'s public wrapper unsafe.
fn sim_dealloc(ptr: *mut c_void) {
    if ptr.is_null() || (ptr as usize) == SIM_ALLOC_ALIGN {
        return;
    }
    // SAFETY: by this function's contract `ptr` came from `sim_alloc`.
    let base = unsafe { (ptr as *mut u8).sub(SIM_ALLOC_ALIGN) };
    // SAFETY: `sim_alloc` wrote a `usize` at `base` before returning.
    let size = unsafe { (base as *const usize).read() };
    if let Ok(layout) = std::alloc::Layout::from_size_align(SIM_ALLOC_ALIGN + size, SIM_ALLOC_ALIGN)
    {
        // SAFETY: `layout` is exactly the layout `sim_alloc` allocated
        // `base` with.
        unsafe { std::alloc::dealloc(base, layout) };
    }
}

/// OneAPI memory backend implementation
pub struct OneApiMemoryBackend {
    /// Backend configuration
    config: OneApiConfig,
    /// Device properties
    device_properties: SyclDeviceProperties,
    /// Active SYCL contexts
    contexts: HashMap<u32, SyclContext>,
    /// Memory pools
    memory_pools: HashMap<OneApiMemoryType, OneApiMemoryPool>,
    /// Statistics
    stats: OneApiStats,
    /// Queue management
    queue_manager: SyclQueueManager,
}

/// OneAPI backend configuration
#[derive(Debug, Clone)]
pub struct OneApiConfig {
    /// Device ID to use
    pub device_id: u32,
    /// Enable unified shared memory (USM)
    pub enable_usm: bool,
    /// Enable device-specific USM
    pub enable_device_usm: bool,
    /// Enable host USM
    pub enable_host_usm: bool,
    /// Enable shared USM
    pub enable_shared_usm: bool,
    /// Enable memory pools
    pub enable_memory_pools: bool,
    /// Enable async memory operations
    pub enable_async_ops: bool,
    /// Memory pool growth size
    pub pool_growth_size: usize,
    /// Maximum number of queues
    pub max_queues: u32,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable sub-groups
    pub enable_sub_groups: bool,
}

impl Default for OneApiConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_usm: true,
            enable_device_usm: true,
            enable_host_usm: true,
            enable_shared_usm: true,
            enable_memory_pools: true,
            enable_async_ops: true,
            pool_growth_size: 64 * 1024 * 1024, // 64MB
            max_queues: 16,
            enable_profiling: false,
            enable_sub_groups: true,
        }
    }
}

/// SYCL device properties
#[derive(Debug, Clone)]
pub struct SyclDeviceProperties {
    pub device_id: u32,
    pub name: String,
    pub vendor: String,
    pub device_type: SyclDeviceType,
    pub driver_version: String,
    pub global_memory_size: usize,
    pub local_memory_size: usize,
    pub max_work_group_size: u32,
    pub max_work_item_dimensions: u32,
    pub max_work_item_sizes: [u32; 3],
    pub compute_units: u32,
    pub max_compute_units: u32,
    pub sub_group_sizes: Vec<u32>,
    pub preferred_sub_group_size: u32,
    pub max_constant_buffer_size: usize,
    pub has_fp64: bool,
    pub has_fp16: bool,
    pub has_atomic64: bool,
    pub usm_device_allocations: bool,
    pub usm_host_allocations: bool,
    pub usm_shared_allocations: bool,
    pub usm_system_allocations: bool,
}

/// SYCL device types
#[derive(Debug, Clone, PartialEq)]
pub enum SyclDeviceType {
    GPU,
    CPU,
    Accelerator,
    Host,
    Custom,
}

/// OneAPI memory types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OneApiMemoryType {
    Device,
    Host,
    Shared,
    System,
    Buffer,
}

/// SYCL context for managing device state
pub struct SyclContext {
    /// Context handle (simulated)
    pub handle: *mut c_void,
    /// Device ID
    pub device_id: u32,
    /// Associated device
    pub device_properties: SyclDeviceProperties,
    /// Creation time
    pub created_at: Instant,
    /// Active queues
    pub queues: Vec<SyclQueue>,
    /// USM allocations
    pub usm_allocations: HashMap<*mut c_void, UsmAllocation>,
}

/// USM (Unified Shared Memory) allocation info
#[derive(Debug, Clone)]
pub struct UsmAllocation {
    pub ptr: *mut c_void,
    pub size: usize,
    pub usm_kind: UsmKind,
    pub allocated_at: Instant,
    pub device_id: u32,
    pub alignment: usize,
}

/// USM allocation kinds
#[derive(Debug, Clone, PartialEq)]
pub enum UsmKind {
    Device, // Device-only memory
    Host,   // Host-accessible memory
    Shared, // Automatically migrating memory
    System, // System allocator memory
}

/// SYCL queue for asynchronous operations
pub struct SyclQueue {
    /// Queue handle (simulated)
    pub handle: *mut c_void,
    /// Queue ID
    pub id: u32,
    /// Queue properties
    pub properties: SyclQueueProperties,
    /// Creation time
    pub created_at: Instant,
    /// Operations queue
    pub operations: std::collections::VecDeque<SyclOperation>,
    /// Associated context
    pub context_id: Option<u32>,
}

/// SYCL queue properties
#[derive(Debug, Clone)]
pub struct SyclQueueProperties {
    pub in_order: bool,
    pub enable_profiling: bool,
    pub priority: SyclQueuePriority,
}

impl Default for SyclQueueProperties {
    fn default() -> Self {
        Self {
            in_order: false,
            enable_profiling: false,
            priority: SyclQueuePriority::Normal,
        }
    }
}

/// SYCL queue priorities
#[derive(Debug, Clone, PartialEq)]
pub enum SyclQueuePriority {
    Low,
    Normal,
    High,
}

/// SYCL asynchronous operation
#[derive(Debug, Clone)]
pub struct SyclOperation {
    pub op_type: SyclOperationType,
    pub src_ptr: Option<*mut c_void>,
    pub dst_ptr: Option<*mut c_void>,
    pub size: usize,
    pub timestamp: Instant,
    pub event_handle: Option<*mut c_void>,
}

/// Types of SYCL operations
#[derive(Debug, Clone)]
pub enum SyclOperationType {
    MemcpyHostToDevice,
    MemcpyDeviceToHost,
    MemcpyDeviceToDevice,
    UsmMemcpy,
    UsmMemset,
    KernelSubmit,
    BarrierWait,
    Fill,
}

/// OneAPI memory pool
pub struct OneApiMemoryPool {
    /// Memory type
    memory_type: OneApiMemoryType,
    /// Pool handle (simulated)
    handle: *mut c_void,
    /// Current size
    current_size: usize,
    /// Maximum size
    max_size: usize,
    /// Used size
    used_size: usize,
    /// Free blocks
    free_blocks: std::collections::VecDeque<OneApiMemoryBlock>,
    /// Allocated blocks
    allocated_blocks: HashMap<*mut c_void, OneApiMemoryBlock>,
    /// USM properties
    usm_properties: UsmProperties,
}

/// OneAPI memory block
#[derive(Debug, Clone)]
pub struct OneApiMemoryBlock {
    pub ptr: *mut c_void,
    pub size: usize,
    pub memory_type: OneApiMemoryType,
    pub allocated_at: Instant,
    pub last_access: Option<Instant>,
    pub ref_count: u32,
    pub usm_kind: Option<UsmKind>,
    pub device_accessible: bool,
    pub host_accessible: bool,
}

/// USM memory properties
#[derive(Debug, Clone)]
pub struct UsmProperties {
    pub alignment: usize,
    pub device_read_only: bool,
    pub device_access: bool,
    pub host_access: bool,
    pub supports_atomics: bool,
}

impl Default for UsmProperties {
    fn default() -> Self {
        Self {
            alignment: 64, // Common alignment for Intel GPUs
            device_read_only: false,
            device_access: true,
            host_access: false,
            supports_atomics: true,
        }
    }
}

impl OneApiMemoryPool {
    pub fn new(memory_type: OneApiMemoryType, max_size: usize) -> Self {
        let usm_properties = match memory_type {
            OneApiMemoryType::Device => UsmProperties {
                alignment: 64,
                device_read_only: false,
                device_access: true,
                host_access: false,
                supports_atomics: true,
            },
            OneApiMemoryType::Host => UsmProperties {
                alignment: 64,
                device_read_only: false,
                device_access: true,
                host_access: true,
                supports_atomics: false,
            },
            OneApiMemoryType::Shared => UsmProperties {
                alignment: 64,
                device_read_only: false,
                device_access: true,
                host_access: true,
                supports_atomics: true,
            },
            _ => UsmProperties::default(),
        };

        Self {
            memory_type,
            handle: std::ptr::null_mut(),
            current_size: 0,
            max_size,
            used_size: 0,
            free_blocks: std::collections::VecDeque::new(),
            allocated_blocks: HashMap::new(),
            usm_properties,
        }
    }

    /// Allocate from pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut c_void, OneApiError> {
        // Try to find suitable free block
        for i in 0..self.free_blocks.len() {
            if self.free_blocks[i].size >= size {
                let Some(mut block) = self.free_blocks.remove(i) else {
                    continue;
                };

                // Split block if much larger
                if block.size > size * 2 {
                    let remaining_block = OneApiMemoryBlock {
                        ptr: unsafe { block.ptr.add(size) },
                        size: block.size - size,
                        memory_type: block.memory_type.clone(),
                        allocated_at: block.allocated_at,
                        last_access: None,
                        ref_count: 0,
                        usm_kind: block.usm_kind.clone(),
                        device_accessible: block.device_accessible,
                        host_accessible: block.host_accessible,
                    };
                    self.free_blocks.push_back(remaining_block);
                    block.size = size;
                }

                block.last_access = Some(Instant::now());
                block.ref_count = 1;

                let ptr = block.ptr;
                self.allocated_blocks.insert(ptr, block);
                self.used_size += size;

                return Ok(ptr);
            }
        }

        // Need to allocate new memory
        if self.current_size + size > self.max_size {
            return Err(OneApiError::OutOfMemory(
                "Pool size limit exceeded".to_string(),
            ));
        }

        let ptr = self.sycl_malloc(size)?;
        let usm_kind = match self.memory_type {
            OneApiMemoryType::Device => Some(UsmKind::Device),
            OneApiMemoryType::Host => Some(UsmKind::Host),
            OneApiMemoryType::Shared => Some(UsmKind::Shared),
            OneApiMemoryType::System => Some(UsmKind::System),
            _ => None,
        };

        let block = OneApiMemoryBlock {
            ptr,
            size,
            memory_type: self.memory_type.clone(),
            allocated_at: Instant::now(),
            last_access: Some(Instant::now()),
            ref_count: 1,
            usm_kind,
            device_accessible: self.usm_properties.device_access,
            host_accessible: self.usm_properties.host_access,
        };

        self.allocated_blocks.insert(ptr, block);
        self.current_size += size;
        self.used_size += size;

        Ok(ptr)
    }

    /// Free back to pool
    pub fn free(&mut self, ptr: *mut c_void) -> Result<(), OneApiError> {
        if let Some(block) = self.allocated_blocks.remove(&ptr) {
            self.used_size -= block.size;

            // Add to free blocks
            self.free_blocks.push_back(OneApiMemoryBlock {
                ptr: block.ptr,
                size: block.size,
                memory_type: block.memory_type,
                allocated_at: block.allocated_at,
                last_access: None,
                ref_count: 0,
                usm_kind: block.usm_kind,
                device_accessible: block.device_accessible,
                host_accessible: block.host_accessible,
            });

            // Try to coalesce adjacent blocks
            self.coalesce_free_blocks();

            Ok(())
        } else {
            Err(OneApiError::InvalidPointer(
                "Pointer not found in pool".to_string(),
            ))
        }
    }

    fn coalesce_free_blocks(&mut self) {
        // Sort free blocks by address
        let mut blocks: Vec<OneApiMemoryBlock> = self.free_blocks.drain(..).collect();
        blocks.sort_by_key(|block| block.ptr as usize);

        let mut coalesced = Vec::new();
        let mut current_block: Option<OneApiMemoryBlock> = None;

        for block in blocks {
            match current_block.take() {
                None => current_block = Some(block),
                Some(mut prev_block) => {
                    let prev_end = prev_block.ptr as usize + prev_block.size;
                    let block_start = block.ptr as usize;

                    if prev_end == block_start && prev_block.memory_type == block.memory_type {
                        // Coalesce blocks
                        prev_block.size += block.size;
                        current_block = Some(prev_block);
                    } else {
                        coalesced.push(prev_block);
                        current_block = Some(block);
                    }
                }
            }
        }

        if let Some(block) = current_block {
            coalesced.push(block);
        }

        self.free_blocks = coalesced.into();
    }

    fn sycl_malloc(&self, size: usize) -> Result<*mut c_void, OneApiError> {
        // Simulate SYCL USM allocation
        match self.memory_type {
            OneApiMemoryType::Device => sim_alloc(size, self.usm_properties.alignment), // malloc_device
            OneApiMemoryType::Host => sim_alloc(size, self.usm_properties.alignment), // malloc_host
            OneApiMemoryType::Shared => sim_alloc(size, self.usm_properties.alignment), // malloc_shared
            OneApiMemoryType::System => sim_alloc(size, self.usm_properties.alignment), // system malloc
            _ => Err(OneApiError::UnsupportedOperation(
                "Unsupported memory type for allocation".to_string(),
            )),
        }
    }
}

/// SYCL queue manager
pub struct SyclQueueManager {
    /// Available queues
    queues: Vec<SyclQueue>,
    /// Queue pool for reuse
    queue_pool: std::collections::VecDeque<SyclQueue>,
    /// Next queue ID
    next_queue_id: u32,
    /// Configuration
    config: SyclQueueConfig,
}

/// Queue manager configuration
#[derive(Debug, Clone)]
pub struct SyclQueueConfig {
    pub default_priority: SyclQueuePriority,
    pub enable_priorities: bool,
    pub max_operations_per_queue: usize,
    pub default_in_order: bool,
}

impl Default for SyclQueueConfig {
    fn default() -> Self {
        Self {
            default_priority: SyclQueuePriority::Normal,
            enable_priorities: true,
            max_operations_per_queue: 1000,
            default_in_order: false,
        }
    }
}

impl SyclQueueManager {
    pub fn new(config: SyclQueueConfig) -> Self {
        Self {
            queues: Vec::new(),
            queue_pool: std::collections::VecDeque::new(),
            next_queue_id: 0,
            config,
        }
    }

    /// Create new queue
    pub fn create_queue(
        &mut self,
        properties: Option<SyclQueueProperties>,
    ) -> Result<u32, OneApiError> {
        let queue_id = self.next_queue_id;
        self.next_queue_id += 1;

        let queue_properties = properties.unwrap_or_else(|| SyclQueueProperties {
            in_order: self.config.default_in_order,
            enable_profiling: false,
            priority: self.config.default_priority.clone(),
        });

        let queue = SyclQueue {
            handle: std::ptr::null_mut(), // Would be actual SYCL queue
            id: queue_id,
            properties: queue_properties,
            created_at: Instant::now(),
            operations: std::collections::VecDeque::new(),
            context_id: None,
        };

        self.queues.push(queue);
        Ok(queue_id)
    }

    /// Destroy queue
    pub fn destroy_queue(&mut self, queue_id: u32) -> Result<(), OneApiError> {
        if let Some(pos) = self.queues.iter().position(|q| q.id == queue_id) {
            let queue = self.queues.remove(pos);
            // Clean up queue resources
            Ok(())
        } else {
            Err(OneApiError::InvalidQueue("Queue not found".to_string()))
        }
    }

    /// Submit operation to queue
    pub fn submit_operation(
        &mut self,
        queue_id: u32,
        operation: SyclOperation,
    ) -> Result<(), OneApiError> {
        if let Some(queue) = self.queues.iter_mut().find(|q| q.id == queue_id) {
            if queue.operations.len() >= self.config.max_operations_per_queue {
                return Err(OneApiError::QueueFull(
                    "Queue operation limit reached".to_string(),
                ));
            }

            queue.operations.push_back(operation);
            Ok(())
        } else {
            Err(OneApiError::InvalidQueue("Queue not found".to_string()))
        }
    }

    /// Wait for queue completion
    pub fn wait_for_queue(&mut self, queue_id: u32) -> Result<(), OneApiError> {
        // First, collect all operations from the queue
        let mut operations = Vec::new();
        if let Some(queue) = self.queues.iter_mut().find(|q| q.id == queue_id) {
            while let Some(operation) = queue.operations.pop_front() {
                operations.push(operation);
            }
        } else {
            return Err(OneApiError::InvalidQueue("Queue not found".to_string()));
        }

        // Now execute all operations
        for operation in operations {
            self.execute_operation(operation)?;
        }

        Ok(())
    }

    fn execute_operation(&self, _operation: SyclOperation) -> Result<(), OneApiError> {
        // Host-memory simulation (see module docs): there is no real
        // SYCL/Level Zero device to transfer to or from, so no data movement
        // happens here. This used to also inject an artificial
        // `std::thread::sleep` per operation type to mimic device-transfer
        // and kernel-submission latency; that fake timing has been removed
        // rather than left as an undisclosed simulated number, so callers
        // now see the true (near-zero) cost of this simulation instead of a
        // fabricated one.
        Ok(())
    }
}

/// OneAPI statistics
#[derive(Debug, Clone, Default)]
pub struct OneApiStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub device_memory_used: usize,
    pub host_memory_used: usize,
    pub shared_memory_used: usize,
    pub usm_allocations: u64,
    pub queue_operations: u64,
    pub kernel_submissions: u64,
    pub memory_transfers: u64,
    pub average_allocation_time: Duration,
    pub peak_memory_usage: usize,
}

impl OneApiMemoryBackend {
    /// Create new OneAPI backend
    pub fn new(config: OneApiConfig) -> Result<Self, OneApiError> {
        // Initialize OneAPI device
        let device_properties = Self::query_device_properties(config.device_id)?;

        // Create memory pools
        let mut memory_pools = HashMap::new();
        if config.enable_memory_pools {
            let pool_size = device_properties.global_memory_size / 4; // Use 1/4 of total memory

            if config.enable_device_usm {
                memory_pools.insert(
                    OneApiMemoryType::Device,
                    OneApiMemoryPool::new(OneApiMemoryType::Device, pool_size),
                );
            }

            if config.enable_host_usm {
                memory_pools.insert(
                    OneApiMemoryType::Host,
                    OneApiMemoryPool::new(OneApiMemoryType::Host, pool_size),
                );
            }

            if config.enable_shared_usm {
                memory_pools.insert(
                    OneApiMemoryType::Shared,
                    OneApiMemoryPool::new(OneApiMemoryType::Shared, pool_size / 2),
                );
            }

            memory_pools.insert(
                OneApiMemoryType::System,
                OneApiMemoryPool::new(OneApiMemoryType::System, pool_size / 4),
            );
        }

        let queue_manager = SyclQueueManager::new(SyclQueueConfig::default());

        Ok(Self {
            config,
            device_properties,
            contexts: HashMap::new(),
            memory_pools,
            stats: OneApiStats::default(),
            queue_manager,
        })
    }

    /// Query device properties
    fn query_device_properties(device_id: u32) -> Result<SyclDeviceProperties, OneApiError> {
        // Simulate querying OneAPI/SYCL device properties
        Ok(SyclDeviceProperties {
            device_id,
            name: format!("Intel GPU {}", device_id),
            vendor: "Intel Corporation".to_string(),
            device_type: SyclDeviceType::GPU,
            driver_version: "1.3.0".to_string(),
            global_memory_size: 12 * 1024 * 1024 * 1024, // 12GB
            local_memory_size: 64 * 1024,                // 64KB
            max_work_group_size: 1024,
            max_work_item_dimensions: 3,
            max_work_item_sizes: [1024, 1024, 1024],
            compute_units: 96,
            max_compute_units: 96,
            sub_group_sizes: vec![8, 16, 32],
            preferred_sub_group_size: 16,
            max_constant_buffer_size: 64 * 1024,
            has_fp64: true,
            has_fp16: true,
            has_atomic64: true,
            usm_device_allocations: true,
            usm_host_allocations: true,
            usm_shared_allocations: true,
            usm_system_allocations: true,
        })
    }

    /// Allocate memory
    pub fn allocate(
        &mut self,
        size: usize,
        memory_type: OneApiMemoryType,
    ) -> Result<*mut c_void, OneApiError> {
        let start_time = Instant::now();

        let ptr = if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.allocate(size)?
            } else {
                return Err(OneApiError::UnsupportedMemoryType(
                    "Memory type not supported".to_string(),
                ));
            }
        } else {
            // Direct allocation
            self.direct_allocate(size, memory_type.clone())?
        };

        // Update statistics
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += size as u64;

        match memory_type {
            OneApiMemoryType::Device => self.stats.device_memory_used += size,
            OneApiMemoryType::Host => self.stats.host_memory_used += size,
            OneApiMemoryType::Shared => self.stats.shared_memory_used += size,
            _ => {}
        }

        if matches!(
            memory_type,
            OneApiMemoryType::Device | OneApiMemoryType::Host | OneApiMemoryType::Shared
        ) {
            self.stats.usm_allocations += 1;
        }

        let allocation_time = start_time.elapsed();
        let total_time = self.stats.average_allocation_time.as_nanos() as u64
            * (self.stats.total_allocations - 1)
            + allocation_time.as_nanos() as u64;
        self.stats.average_allocation_time =
            Duration::from_nanos(total_time / self.stats.total_allocations);

        let current_usage = self.stats.device_memory_used
            + self.stats.host_memory_used
            + self.stats.shared_memory_used;
        if current_usage > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = current_usage;
        }

        Ok(ptr)
    }

    fn direct_allocate(
        &self,
        size: usize,
        memory_type: OneApiMemoryType,
    ) -> Result<*mut c_void, OneApiError> {
        // Simulate direct SYCL allocation
        let alignment = 64; // Common alignment for Intel GPUs

        match memory_type {
            OneApiMemoryType::Device => sim_alloc(size, alignment), // malloc_device
            OneApiMemoryType::Host => sim_alloc(size, alignment),   // malloc_host
            OneApiMemoryType::Shared => sim_alloc(size, alignment), // malloc_shared
            OneApiMemoryType::System => sim_alloc(size, alignment), // system malloc
            _ => Err(OneApiError::UnsupportedMemoryType(
                "Unsupported memory type".to_string(),
            )),
        }
    }

    /// Free memory
    pub fn free(
        &mut self,
        ptr: *mut c_void,
        memory_type: OneApiMemoryType,
    ) -> Result<(), OneApiError> {
        if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.free(ptr)?;
            } else {
                return Err(OneApiError::UnsupportedMemoryType(
                    "Memory type not supported".to_string(),
                ));
            }
        } else {
            // Direct deallocation. `ptr` was returned by `sim_alloc` via
            // `sycl_malloc`/`direct_allocate` above, and this is the first
            // time it is freed.
            sim_dealloc(ptr);
        }

        self.stats.total_deallocations += 1;
        Ok(())
    }

    /// USM memory copy
    pub fn usm_memcpy(
        &mut self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        queue_id: u32,
    ) -> Result<(), OneApiError> {
        let operation = SyclOperation {
            op_type: SyclOperationType::UsmMemcpy,
            src_ptr: Some(src as *mut c_void),
            dst_ptr: Some(dst),
            size,
            timestamp: Instant::now(),
            event_handle: None,
        };

        self.queue_manager.submit_operation(queue_id, operation)?;
        self.stats.memory_transfers += 1;
        Ok(())
    }

    /// Create SYCL context
    pub fn create_context(&mut self) -> Result<u32, OneApiError> {
        let context_id = self.contexts.len() as u32;

        let context = SyclContext {
            handle: std::ptr::null_mut(), // Would be actual SYCL context
            device_id: self.config.device_id,
            device_properties: self.device_properties.clone(),
            created_at: Instant::now(),
            queues: Vec::new(),
            usm_allocations: HashMap::new(),
        };

        self.contexts.insert(context_id, context);
        Ok(context_id)
    }

    /// Create queue
    pub fn create_queue(
        &mut self,
        properties: Option<SyclQueueProperties>,
    ) -> Result<u32, OneApiError> {
        self.queue_manager.create_queue(properties)
    }

    /// Destroy queue
    pub fn destroy_queue(&mut self, queue_id: u32) -> Result<(), OneApiError> {
        self.queue_manager.destroy_queue(queue_id)
    }

    /// Wait for all queues
    pub fn wait_all(&mut self) -> Result<(), OneApiError> {
        let queue_ids: Vec<u32> = self.queue_manager.queues.iter().map(|q| q.id).collect();
        for queue_id in queue_ids {
            self.queue_manager.wait_for_queue(queue_id)?;
        }
        Ok(())
    }

    /// Get device properties
    pub fn get_device_properties(&self) -> &SyclDeviceProperties {
        &self.device_properties
    }

    /// Get statistics
    pub fn get_stats(&self) -> &OneApiStats {
        &self.stats
    }

    /// Query USM pointer information
    pub fn query_usm_ptr(&self, ptr: *mut c_void) -> Result<UsmAllocation, OneApiError> {
        // In a real implementation, this would query the actual USM pointer
        // For now, return default information
        Ok(UsmAllocation {
            ptr,
            size: 0, // Would need to track actual size
            usm_kind: UsmKind::Device,
            allocated_at: Instant::now(),
            device_id: self.config.device_id,
            alignment: 64,
        })
    }
}

// Safety: OneApiMemoryBackend manages OneAPI/SYCL GPU memory pointers via *mut c_void.
// While raw pointers are not Send/Sync by default, it's safe to share across threads
// when protected by Arc<Mutex<>> because:
// 1. All pointers point to SYCL USM memory managed by the OneAPI runtime
// 2. The Mutex provides exclusive access for all mutable operations
// 3. No thread-local state is maintained
unsafe impl Send for OneApiMemoryBackend {}
unsafe impl Sync for OneApiMemoryBackend {}

/// OneAPI errors
#[derive(Debug, Clone)]
pub enum OneApiError {
    DeviceNotFound(String),
    OutOfMemory(String),
    InvalidPointer(String),
    InvalidQueue(String),
    QueueFull(String),
    UnsupportedOperation(String),
    UnsupportedMemoryType(String),
    ContextCreationFailed(String),
    KernelSubmissionFailed(String),
    SynchronizationFailed(String),
    InternalError(String),
}

impl std::fmt::Display for OneApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OneApiError::DeviceNotFound(msg) => write!(f, "Device not found: {}", msg),
            OneApiError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            OneApiError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            OneApiError::InvalidQueue(msg) => write!(f, "Invalid queue: {}", msg),
            OneApiError::QueueFull(msg) => write!(f, "Queue full: {}", msg),
            OneApiError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            OneApiError::UnsupportedMemoryType(msg) => {
                write!(f, "Unsupported memory type: {}", msg)
            }
            OneApiError::ContextCreationFailed(msg) => {
                write!(f, "Context creation failed: {}", msg)
            }
            OneApiError::KernelSubmissionFailed(msg) => {
                write!(f, "Kernel submission failed: {}", msg)
            }
            OneApiError::SynchronizationFailed(msg) => write!(f, "Synchronization failed: {}", msg),
            OneApiError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for OneApiError {}

/// Thread-safe OneAPI backend wrapper
pub struct ThreadSafeOneApiBackend {
    backend: Arc<Mutex<OneApiMemoryBackend>>,
}

impl ThreadSafeOneApiBackend {
    pub fn new(config: OneApiConfig) -> Result<Self, OneApiError> {
        let backend = OneApiMemoryBackend::new(config)?;
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    pub fn allocate(
        &self,
        size: usize,
        memory_type: OneApiMemoryType,
    ) -> Result<*mut c_void, OneApiError> {
        let mut backend = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        backend.allocate(size, memory_type)
    }

    pub fn free(&self, ptr: *mut c_void, memory_type: OneApiMemoryType) -> Result<(), OneApiError> {
        let mut backend = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        backend.free(ptr, memory_type)
    }

    pub fn get_stats(&self) -> OneApiStats {
        let backend = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        backend.get_stats().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for F26: a zero-size request must not reach
    /// `std::alloc::alloc` (unsound for a zero-size layout), and an
    /// out-of-range alignment must be an honest error, not silent UB.
    #[test]
    fn sim_alloc_zero_size_is_a_safe_sentinel_not_a_ub_call() {
        let ptr = sim_alloc(0, 64).expect("zero-size request must succeed");
        assert!(!ptr.is_null());
        sim_dealloc(ptr);
    }

    #[test]
    fn sim_alloc_rejects_unsupported_alignment() {
        assert!(sim_alloc(16, 3).is_err(), "3 is not a power of two");
        assert!(
            sim_alloc(16, 512).is_err(),
            "512 exceeds this simulated backend's supported alignment"
        );
    }

    /// A real allocation must be readable/writable across its full size and
    /// must free through the same layout it was allocated with.
    #[test]
    fn sim_alloc_real_allocation_round_trips_and_frees_cleanly() {
        for (size, align) in [(1usize, 8usize), (7, 16), (256, 64), (4096, 128)] {
            let ptr = sim_alloc(size, align).expect("allocation must succeed") as *mut u8;
            assert!(!ptr.is_null());
            assert_eq!(
                (ptr as usize) % align,
                0,
                "returned pointer does not honour the requested alignment"
            );
            unsafe {
                for i in 0..size {
                    ptr.add(i).write(0xAB);
                }
                for i in 0..size {
                    assert_eq!(ptr.add(i).read(), 0xAB);
                }
                sim_dealloc(ptr as *mut c_void);
            }
        }
    }

    #[test]
    fn sim_dealloc_null_is_a_no_op() {
        sim_dealloc(std::ptr::null_mut());
    }

    #[test]
    fn test_oneapi_backend_creation() {
        let config = OneApiConfig::default();
        let backend = OneApiMemoryBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = OneApiMemoryPool::new(OneApiMemoryType::Device, 1024 * 1024);
        let ptr = pool.allocate(1024);
        assert!(ptr.is_ok());

        let ptr = ptr.expect("unwrap failed");
        let result = pool.free(ptr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sycl_queue_manager() {
        let mut manager = SyclQueueManager::new(SyclQueueConfig::default());
        let queue_id = manager.create_queue(None);
        assert!(queue_id.is_ok());

        let queue_id = queue_id.expect("unwrap failed");
        let result = manager.destroy_queue(queue_id);
        assert!(result.is_ok());
    }

    #[test]
    fn test_thread_safe_backend() {
        let config = OneApiConfig::default();
        let backend = ThreadSafeOneApiBackend::new(config);
        assert!(backend.is_ok());

        let backend = backend.expect("unwrap failed");
        let stats = backend.get_stats();
        assert_eq!(stats.total_allocations, 0);
    }
}
