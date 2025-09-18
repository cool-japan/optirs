// Metal backend for GPU memory management
//
// This module provides Apple Metal-specific memory management functionality,
// including device memory allocation, Metal command buffers, and performance
// optimization features specific to Apple Silicon GPUs.

#[allow(dead_code)]
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Metal memory backend implementation
pub struct MetalMemoryBackend {
    /// Backend configuration
    config: MetalConfig,
    /// Device properties
    device_properties: MetalDeviceProperties,
    /// Active Metal devices
    devices: HashMap<u32, MetalDevice>,
    /// Memory pools
    memory_pools: HashMap<MetalMemoryType, MetalMemoryPool>,
    /// Statistics
    stats: MetalStats,
    /// Command queue management
    command_manager: MetalCommandManager,
}

/// Metal backend configuration
#[derive(Debug, Clone)]
pub struct MetalConfig {
    /// Device ID to use
    pub device_id: u32,
    /// Enable private memory
    pub enable_private_memory: bool,
    /// Enable shared memory
    pub enable_shared_memory: bool,
    /// Enable managed memory
    pub enable_managed_memory: bool,
    /// Enable memory pools
    pub enable_memory_pools: bool,
    /// Enable async memory operations
    pub enable_async_ops: bool,
    /// Memory pool growth size
    pub pool_growth_size: usize,
    /// Enable memoryless render targets
    pub enable_memoryless_targets: bool,
    /// Maximum number of command queues
    pub max_command_queues: u32,
    /// Enable Metal Performance Shaders
    pub enable_mps: bool,
    /// Enable heap-based allocation
    pub enable_heap_allocation: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_private_memory: true,
            enable_shared_memory: true,
            enable_managed_memory: true,
            enable_memory_pools: true,
            enable_async_ops: true,
            pool_growth_size: 64 * 1024 * 1024, // 64MB
            enable_memoryless_targets: false,
            max_command_queues: 8,
            enable_mps: true,
            enable_heap_allocation: true,
        }
    }
}

/// Metal device properties
#[derive(Debug, Clone)]
pub struct MetalDeviceProperties {
    pub device_id: u32,
    pub name: String,
    pub device_type: MetalDeviceType,
    pub family: MetalGPUFamily,
    pub max_threads_per_threadgroup: u32,
    pub threadgroup_memory_length: u32,
    pub max_buffer_length: usize,
    pub max_texture_size_2d: u32,
    pub max_texture_size_3d: u32,
    pub unified_memory: bool,
    pub discrete_memory: bool,
    pub low_power: bool,
    pub headless: bool,
    pub supports_shader_debugging: bool,
    pub supports_function_pointers: bool,
    pub supports_dynamic_libraries: bool,
    pub supports_render_dynamic_libraries: bool,
    pub recommended_max_working_set_size: usize,
    pub max_transfer_rate: u64,
    pub has_unified_memory: bool,
}

/// Metal device types
#[derive(Debug, Clone, PartialEq)]
pub enum MetalDeviceType {
    Integrated,
    Discrete,
    External,
    Virtual,
}

/// Metal GPU families (Apple Silicon generations)
#[derive(Debug, Clone, PartialEq)]
pub enum MetalGPUFamily {
    Apple1, // A7
    Apple2, // A8
    Apple3, // A9, A10
    Apple4, // A11
    Apple5, // A12, A13
    Apple6, // A14, M1
    Apple7, // A15, M1 Pro, M1 Max
    Apple8, // A16, M2
    Apple9, // M2 Pro, M2 Max, M3
    Mac1,   // Intel Iris Pro
    Mac2,   // Intel Iris Pro, AMD
}

/// Metal memory types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalMemoryType {
    Private,    // GPU-only memory
    Shared,     // CPU-GPU shared memory
    Managed,    // Automatically managed memory
    Memoryless, // Tile memory (iOS only)
}

/// Metal device abstraction
pub struct MetalDevice {
    /// Device handle (simulated)
    pub handle: *mut c_void,
    /// Device ID
    pub device_id: u32,
    /// Device properties
    pub properties: MetalDeviceProperties,
    /// Creation time
    pub created_at: Instant,
    /// Command queues
    pub command_queues: Vec<MetalCommandQueue>,
    /// Memory heaps
    pub heaps: HashMap<usize, MetalHeap>,
    /// Active resources
    pub resources: HashMap<*mut c_void, MetalResource>,
}

/// Metal command queue for GPU operations
pub struct MetalCommandQueue {
    /// Queue handle (simulated)
    pub handle: *mut c_void,
    /// Queue ID
    pub id: u32,
    /// Queue label
    pub label: Option<String>,
    /// Creation time
    pub created_at: Instant,
    /// Command buffers
    pub command_buffers: std::collections::VecDeque<MetalCommandBuffer>,
    /// Queue priority
    pub priority: MetalQueuePriority,
}

/// Metal queue priorities
#[derive(Debug, Clone, PartialEq)]
pub enum MetalQueuePriority {
    High,
    Normal,
    Low,
    Background,
}

/// Metal command buffer
#[derive(Debug, Clone)]
pub struct MetalCommandBuffer {
    pub buffer_id: u32,
    pub commands: Vec<MetalCommand>,
    pub timestamp: Instant,
    pub committed: bool,
    pub completed: bool,
}

/// Metal GPU commands
#[derive(Debug, Clone)]
pub enum MetalCommand {
    BlitCommand {
        src_buffer: *mut c_void,
        dst_buffer: *mut c_void,
        size: usize,
    },
    ComputeCommand {
        kernel_id: u32,
        threadgroup_size: (u32, u32, u32),
        threadgroups: (u32, u32, u32),
    },
    RenderCommand {
        render_pass: u32,
    },
    MemoryBarrier,
}

/// Metal memory pool
pub struct MetalMemoryPool {
    /// Memory type
    memory_type: MetalMemoryType,
    /// Pool handle (simulated)
    handle: *mut c_void,
    /// Current size
    current_size: usize,
    /// Maximum size
    max_size: usize,
    /// Used size
    used_size: usize,
    /// Free blocks
    free_blocks: std::collections::VecDeque<MetalMemoryBlock>,
    /// Allocated blocks
    allocated_blocks: HashMap<*mut c_void, MetalMemoryBlock>,
    /// Storage mode
    storage_mode: MetalStorageMode,
    /// Cache mode
    cache_mode: MetalCacheMode,
}

/// Metal memory block
#[derive(Debug, Clone)]
pub struct MetalMemoryBlock {
    pub ptr: *mut c_void,
    pub size: usize,
    pub memory_type: MetalMemoryType,
    pub allocated_at: Instant,
    pub last_access: Option<Instant>,
    pub ref_count: u32,
    pub storage_mode: MetalStorageMode,
    pub cache_mode: MetalCacheMode,
    pub gpu_address: Option<u64>,
}

/// Metal storage modes
#[derive(Debug, Clone, PartialEq)]
pub enum MetalStorageMode {
    Shared,     // CPU and GPU accessible
    Managed,    // Managed by Metal
    Private,    // GPU-only
    Memoryless, // Tile memory
}

/// Metal cache modes
#[derive(Debug, Clone, PartialEq)]
pub enum MetalCacheMode {
    DefaultCache,
    WriteCombined,
}

/// Metal heap for resource allocation
pub struct MetalHeap {
    /// Heap handle (simulated)
    pub handle: *mut c_void,
    /// Heap ID
    pub id: usize,
    /// Size
    pub size: usize,
    /// Used size
    pub used_size: usize,
    /// Storage mode
    pub storage_mode: MetalStorageMode,
    /// CPU cache mode
    pub cpu_cache_mode: MetalCacheMode,
    /// Allocated resources
    pub resources: HashMap<*mut c_void, MetalResource>,
}

/// Metal resource (buffer, texture, etc.)
#[derive(Debug, Clone)]
pub struct MetalResource {
    pub ptr: *mut c_void,
    pub size: usize,
    pub resource_type: MetalResourceType,
    pub storage_mode: MetalStorageMode,
    pub allocated_at: Instant,
    pub heap_offset: Option<usize>,
}

/// Metal resource types
#[derive(Debug, Clone, PartialEq)]
pub enum MetalResourceType {
    Buffer,
    Texture1D,
    Texture2D,
    Texture3D,
    TextureCube,
}

impl MetalMemoryPool {
    pub fn new(memory_type: MetalMemoryType, max_size: usize) -> Self {
        let (storage_mode, cache_mode) = match memory_type {
            MetalMemoryType::Private => (MetalStorageMode::Private, MetalCacheMode::DefaultCache),
            MetalMemoryType::Shared => (MetalStorageMode::Shared, MetalCacheMode::DefaultCache),
            MetalMemoryType::Managed => (MetalStorageMode::Managed, MetalCacheMode::DefaultCache),
            MetalMemoryType::Memoryless => {
                (MetalStorageMode::Memoryless, MetalCacheMode::DefaultCache)
            }
        };

        Self {
            memory_type,
            handle: std::ptr::null_mut(),
            current_size: 0,
            max_size,
            used_size: 0,
            free_blocks: std::collections::VecDeque::new(),
            allocated_blocks: HashMap::new(),
            storage_mode,
            cache_mode,
        }
    }

    /// Allocate from pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut c_void, MetalError> {
        // Try to find suitable free block
        for i in 0..self.free_blocks.len() {
            if self.free_blocks[i].size >= size {
                let mut block = self.free_blocks.remove(i).unwrap();

                // Split block if much larger
                if block.size > size * 2 {
                    let remaining_block = MetalMemoryBlock {
                        ptr: unsafe { block.ptr.add(size) },
                        size: block.size - size,
                        memory_type: block.memory_type.clone(),
                        allocated_at: block.allocated_at,
                        last_access: None,
                        ref_count: 0,
                        storage_mode: block.storage_mode.clone(),
                        cache_mode: block.cache_mode.clone(),
                        gpu_address: None,
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
            return Err(MetalError::OutOfMemory(
                "Pool size limit exceeded".to_string(),
            ));
        }

        let ptr = self.metal_allocate(size)?;
        let block = MetalMemoryBlock {
            ptr,
            size,
            memory_type: self.memory_type.clone(),
            allocated_at: Instant::now(),
            last_access: Some(Instant::now()),
            ref_count: 1,
            storage_mode: self.storage_mode.clone(),
            cache_mode: self.cache_mode.clone(),
            gpu_address: Some(ptr as u64), // Simulate GPU address
        };

        self.allocated_blocks.insert(ptr, block);
        self.current_size += size;
        self.used_size += size;

        Ok(ptr)
    }

    /// Free back to pool
    pub fn free(&mut self, ptr: *mut c_void) -> Result<(), MetalError> {
        if let Some(block) = self.allocated_blocks.remove(&ptr) {
            self.used_size -= block.size;

            // Add to free blocks
            self.free_blocks.push_back(MetalMemoryBlock {
                ptr: block.ptr,
                size: block.size,
                memory_type: block.memory_type,
                allocated_at: block.allocated_at,
                last_access: None,
                ref_count: 0,
                storage_mode: block.storage_mode,
                cache_mode: block.cache_mode,
                gpu_address: block.gpu_address,
            });

            // Try to coalesce adjacent blocks
            self.coalesce_free_blocks();

            Ok(())
        } else {
            Err(MetalError::InvalidPointer(
                "Pointer not found in pool".to_string(),
            ))
        }
    }

    fn coalesce_free_blocks(&mut self) {
        // Sort free blocks by address
        let mut blocks: Vec<MetalMemoryBlock> = self.free_blocks.drain(..).collect();
        blocks.sort_by_key(|block| block.ptr as usize);

        let mut coalesced = Vec::new();
        let mut current_block: Option<MetalMemoryBlock> = None;

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

    fn metal_allocate(&self, size: usize) -> Result<*mut c_void, MetalError> {
        // Simulate Metal buffer allocation
        let alignment = match self.memory_type {
            MetalMemoryType::Private => 64,    // GPU alignment
            MetalMemoryType::Shared => 16,     // CPU-GPU shared
            MetalMemoryType::Managed => 16,    // Managed memory
            MetalMemoryType::Memoryless => 64, // Tile memory
        };

        match self.memory_type {
            MetalMemoryType::Private => {
                // MTLBuffer with private storage
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                        size, alignment,
                    )) as *mut c_void
                })
            }
            MetalMemoryType::Shared => {
                // MTLBuffer with shared storage
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                        size, alignment,
                    )) as *mut c_void
                })
            }
            MetalMemoryType::Managed => {
                // MTLBuffer with managed storage
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                        size, alignment,
                    )) as *mut c_void
                })
            }
            MetalMemoryType::Memoryless => {
                // Memoryless render target (tile memory)
                if size > 8 * 1024 * 1024 {
                    // 8MB tile memory limit
                    return Err(MetalError::UnsupportedOperation(
                        "Memoryless allocation too large".to_string(),
                    ));
                }
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                        size, alignment,
                    )) as *mut c_void
                })
            }
        }
    }
}

/// Metal command manager
pub struct MetalCommandManager {
    /// Command queues
    queues: Vec<MetalCommandQueue>,
    /// Next queue ID
    next_queue_id: u32,
    /// Next command buffer ID
    next_buffer_id: u32,
    /// Configuration
    config: MetalCommandConfig,
}

/// Command manager configuration
#[derive(Debug, Clone)]
pub struct MetalCommandConfig {
    pub max_command_buffers_per_queue: usize,
    pub enable_command_buffer_reuse: bool,
    pub enable_parallel_encoding: bool,
}

impl Default for MetalCommandConfig {
    fn default() -> Self {
        Self {
            max_command_buffers_per_queue: 64,
            enable_command_buffer_reuse: true,
            enable_parallel_encoding: true,
        }
    }
}

impl MetalCommandManager {
    pub fn new(config: MetalCommandConfig) -> Self {
        Self {
            queues: Vec::new(),
            next_queue_id: 0,
            next_buffer_id: 0,
            config,
        }
    }

    /// Create command queue
    pub fn create_command_queue(
        &mut self,
        label: Option<String>,
        priority: MetalQueuePriority,
    ) -> Result<u32, MetalError> {
        let queue_id = self.next_queue_id;
        self.next_queue_id += 1;

        let queue = MetalCommandQueue {
            handle: std::ptr::null_mut(),
            id: queue_id,
            label,
            created_at: Instant::now(),
            command_buffers: std::collections::VecDeque::new(),
            priority,
        };

        self.queues.push(queue);
        Ok(queue_id)
    }

    /// Create command buffer
    pub fn create_command_buffer(&mut self, queue_id: u32) -> Result<u32, MetalError> {
        if let Some(queue) = self.queues.iter_mut().find(|q| q.id == queue_id) {
            if queue.command_buffers.len() >= self.config.max_command_buffers_per_queue {
                return Err(MetalError::QueueFull("Command queue is full".to_string()));
            }

            let buffer_id = self.next_buffer_id;
            self.next_buffer_id += 1;

            let command_buffer = MetalCommandBuffer {
                buffer_id,
                commands: Vec::new(),
                timestamp: Instant::now(),
                committed: false,
                completed: false,
            };

            queue.command_buffers.push_back(command_buffer);
            Ok(buffer_id)
        } else {
            Err(MetalError::InvalidQueue("Queue not found".to_string()))
        }
    }

    /// Add command to buffer
    pub fn add_command(
        &mut self,
        queue_id: u32,
        buffer_id: u32,
        command: MetalCommand,
    ) -> Result<(), MetalError> {
        if let Some(queue) = self.queues.iter_mut().find(|q| q.id == queue_id) {
            if let Some(buffer) = queue
                .command_buffers
                .iter_mut()
                .find(|b| b.buffer_id == buffer_id)
            {
                if buffer.committed {
                    return Err(MetalError::InvalidOperation(
                        "Command buffer already committed".to_string(),
                    ));
                }
                buffer.commands.push(command);
                Ok(())
            } else {
                Err(MetalError::InvalidCommandBuffer(
                    "Command buffer not found".to_string(),
                ))
            }
        } else {
            Err(MetalError::InvalidQueue("Queue not found".to_string()))
        }
    }

    /// Commit command buffer
    pub fn commit_command_buffer(
        &mut self,
        queue_id: u32,
        buffer_id: u32,
    ) -> Result<(), MetalError> {
        if let Some(queue) = self.queues.iter_mut().find(|q| q.id == queue_id) {
            if let Some(buffer) = queue
                .command_buffers
                .iter_mut()
                .find(|b| b.buffer_id == buffer_id)
            {
                buffer.committed = true;
                // Simulate command execution
                std::thread::sleep(Duration::from_micros(50));
                buffer.completed = true;
                Ok(())
            } else {
                Err(MetalError::InvalidCommandBuffer(
                    "Command buffer not found".to_string(),
                ))
            }
        } else {
            Err(MetalError::InvalidQueue("Queue not found".to_string()))
        }
    }

    /// Wait for completion
    pub fn wait_until_completed(
        &mut self,
        queue_id: u32,
        buffer_id: u32,
    ) -> Result<(), MetalError> {
        if let Some(queue) = self.queues.iter().find(|q| q.id == queue_id) {
            if let Some(buffer) = queue
                .command_buffers
                .iter()
                .find(|b| b.buffer_id == buffer_id)
            {
                while !buffer.completed {
                    std::thread::sleep(Duration::from_micros(10));
                }
                Ok(())
            } else {
                Err(MetalError::InvalidCommandBuffer(
                    "Command buffer not found".to_string(),
                ))
            }
        } else {
            Err(MetalError::InvalidQueue("Queue not found".to_string()))
        }
    }
}

/// Metal statistics
#[derive(Debug, Clone, Default)]
pub struct MetalStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub private_memory_used: usize,
    pub shared_memory_used: usize,
    pub managed_memory_used: usize,
    pub command_buffers_created: u64,
    pub command_buffers_completed: u64,
    pub compute_commands: u64,
    pub blit_commands: u64,
    pub render_commands: u64,
    pub average_allocation_time: Duration,
    pub peak_memory_usage: usize,
}

impl MetalMemoryBackend {
    /// Create new Metal backend
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        // Query Metal device
        let device_properties = Self::query_device_properties(config.device_id)?;

        // Create memory pools
        let mut memory_pools = HashMap::new();
        if config.enable_memory_pools {
            let pool_size = device_properties.recommended_max_working_set_size / 4;

            if config.enable_private_memory {
                memory_pools.insert(
                    MetalMemoryType::Private,
                    MetalMemoryPool::new(MetalMemoryType::Private, pool_size),
                );
            }

            if config.enable_shared_memory {
                memory_pools.insert(
                    MetalMemoryType::Shared,
                    MetalMemoryPool::new(MetalMemoryType::Shared, pool_size),
                );
            }

            if config.enable_managed_memory {
                memory_pools.insert(
                    MetalMemoryType::Managed,
                    MetalMemoryPool::new(MetalMemoryType::Managed, pool_size),
                );
            }
        }

        let command_manager = MetalCommandManager::new(MetalCommandConfig::default());

        Ok(Self {
            config,
            device_properties,
            devices: HashMap::new(),
            memory_pools,
            stats: MetalStats::default(),
            command_manager,
        })
    }

    /// Query device properties
    fn query_device_properties(device_id: u32) -> Result<MetalDeviceProperties, MetalError> {
        // Simulate querying Metal device properties
        Ok(MetalDeviceProperties {
            device_id,
            name: "Apple M1 Pro".to_string(),
            device_type: MetalDeviceType::Integrated,
            family: MetalGPUFamily::Apple7,
            max_threads_per_threadgroup: 1024,
            threadgroup_memory_length: 32768,
            max_buffer_length: 2 * 1024 * 1024 * 1024, // 2GB
            max_texture_size_2d: 16384,
            max_texture_size_3d: 2048,
            unified_memory: true,
            discrete_memory: false,
            low_power: false,
            headless: false,
            supports_shader_debugging: true,
            supports_function_pointers: true,
            supports_dynamic_libraries: true,
            supports_render_dynamic_libraries: true,
            recommended_max_working_set_size: 32 * 1024 * 1024 * 1024, // 32GB
            max_transfer_rate: 400_000_000_000,                        // 400 GB/s
            has_unified_memory: true,
        })
    }

    /// Allocate memory
    pub fn allocate(
        &mut self,
        size: usize,
        memory_type: MetalMemoryType,
    ) -> Result<*mut c_void, MetalError> {
        let start_time = Instant::now();

        let ptr = if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.allocate(size)?
            } else {
                return Err(MetalError::UnsupportedMemoryType(
                    "Memory type not supported".to_string(),
                ));
            }
        } else {
            // Direct allocation
            self.direct_allocate(size, memory_type)?
        };

        // Update statistics
        self.stats.total_allocations += 1;
        self.stats.bytes_allocated += size as u64;

        match memory_type {
            MetalMemoryType::Private => self.stats.private_memory_used += size,
            MetalMemoryType::Shared => self.stats.shared_memory_used += size,
            MetalMemoryType::Managed => self.stats.managed_memory_used += size,
            _ => {}
        }

        let allocation_time = start_time.elapsed();
        let total_time = self.stats.average_allocation_time.as_nanos() as u64
            * (self.stats.total_allocations - 1)
            + allocation_time.as_nanos() as u64;
        self.stats.average_allocation_time =
            Duration::from_nanos(total_time / self.stats.total_allocations);

        let current_usage = self.stats.private_memory_used
            + self.stats.shared_memory_used
            + self.stats.managed_memory_used;
        if current_usage > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = current_usage;
        }

        Ok(ptr)
    }

    fn direct_allocate(
        &self,
        size: usize,
        memory_type: MetalMemoryType,
    ) -> Result<*mut c_void, MetalError> {
        let alignment = match memory_type {
            MetalMemoryType::Private => 64,
            MetalMemoryType::Shared => 16,
            MetalMemoryType::Managed => 16,
            MetalMemoryType::Memoryless => 64,
        };

        // Simulate Metal buffer allocation
        match memory_type {
            MetalMemoryType::Private => Ok(unsafe {
                std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                    size, alignment,
                )) as *mut c_void
            }),
            MetalMemoryType::Shared => Ok(unsafe {
                std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                    size, alignment,
                )) as *mut c_void
            }),
            MetalMemoryType::Managed => Ok(unsafe {
                std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                    size, alignment,
                )) as *mut c_void
            }),
            MetalMemoryType::Memoryless => {
                if size > 8 * 1024 * 1024 {
                    return Err(MetalError::UnsupportedOperation(
                        "Memoryless allocation too large".to_string(),
                    ));
                }
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                        size, alignment,
                    )) as *mut c_void
                })
            }
        }
    }

    /// Free memory
    pub fn free(
        &mut self,
        ptr: *mut c_void,
        memory_type: MetalMemoryType,
    ) -> Result<(), MetalError> {
        if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.free(ptr)?;
            } else {
                return Err(MetalError::UnsupportedMemoryType(
                    "Memory type not supported".to_string(),
                ));
            }
        } else {
            // Direct deallocation
            unsafe {
                std::alloc::dealloc(
                    ptr as *mut u8,
                    std::alloc::Layout::from_size_align_unchecked(1, 1),
                );
            }
        }

        self.stats.total_deallocations += 1;
        Ok(())
    }

    /// Copy memory using Metal blit encoder
    pub fn blit_copy(
        &mut self,
        src: *const c_void,
        dst: *mut c_void,
        size: usize,
        queue_id: u32,
    ) -> Result<(), MetalError> {
        let buffer_id = self.command_manager.create_command_buffer(queue_id)?;
        let command = MetalCommand::BlitCommand {
            src_buffer: src as *mut c_void,
            dst_buffer: dst,
            size,
        };

        self.command_manager
            .add_command(queue_id, buffer_id, command)?;
        self.command_manager
            .commit_command_buffer(queue_id, buffer_id)?;
        self.command_manager
            .wait_until_completed(queue_id, buffer_id)?;

        self.stats.blit_commands += 1;
        Ok(())
    }

    /// Create command queue
    pub fn create_command_queue(
        &mut self,
        label: Option<String>,
        priority: MetalQueuePriority,
    ) -> Result<u32, MetalError> {
        self.command_manager.create_command_queue(label, priority)
    }

    /// Get device properties
    pub fn get_device_properties(&self) -> &MetalDeviceProperties {
        &self.device_properties
    }

    /// Get statistics
    pub fn get_stats(&self) -> &MetalStats {
        &self.stats
    }

    /// Wait for all operations to complete
    pub fn wait_until_idle(&mut self) -> Result<(), MetalError> {
        // Wait for all command buffers to complete
        for queue in &self.command_manager.queues {
            for buffer in &queue.command_buffers {
                if buffer.committed && !buffer.completed {
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
        }
        Ok(())
    }
}

/// Metal errors
#[derive(Debug, Clone)]
pub enum MetalError {
    DeviceNotFound(String),
    OutOfMemory(String),
    InvalidPointer(String),
    InvalidQueue(String),
    InvalidCommandBuffer(String),
    QueueFull(String),
    InvalidOperation(String),
    UnsupportedOperation(String),
    UnsupportedMemoryType(String),
    AllocationFailed(String),
    InternalError(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::DeviceNotFound(msg) => write!(f, "Device not found: {}", msg),
            MetalError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            MetalError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            MetalError::InvalidQueue(msg) => write!(f, "Invalid queue: {}", msg),
            MetalError::InvalidCommandBuffer(msg) => write!(f, "Invalid command buffer: {}", msg),
            MetalError::QueueFull(msg) => write!(f, "Queue full: {}", msg),
            MetalError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            MetalError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            MetalError::UnsupportedMemoryType(msg) => write!(f, "Unsupported memory type: {}", msg),
            MetalError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            MetalError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for MetalError {}

/// Thread-safe Metal backend wrapper
pub struct ThreadSafeMetalBackend {
    backend: Arc<Mutex<MetalMemoryBackend>>,
}

impl ThreadSafeMetalBackend {
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        let backend = MetalMemoryBackend::new(config)?;
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    pub fn allocate(
        &self,
        size: usize,
        memory_type: MetalMemoryType,
    ) -> Result<*mut c_void, MetalError> {
        let mut backend = self.backend.lock().unwrap();
        backend.allocate(size, memory_type)
    }

    pub fn free(&self, ptr: *mut c_void, memory_type: MetalMemoryType) -> Result<(), MetalError> {
        let mut backend = self.backend.lock().unwrap();
        backend.free(ptr, memory_type)
    }

    pub fn get_stats(&self) -> MetalStats {
        let backend = self.backend.lock().unwrap();
        backend.get_stats().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let config = MetalConfig::default();
        let backend = MetalMemoryBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MetalMemoryPool::new(MetalMemoryType::Private, 1024 * 1024);
        let ptr = pool.allocate(1024);
        assert!(ptr.is_ok());

        let ptr = ptr.unwrap();
        let result = pool.free(ptr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_manager() {
        let mut manager = MetalCommandManager::new(MetalCommandConfig::default());
        let queue_id =
            manager.create_command_queue(Some("test".to_string()), MetalQueuePriority::Normal);
        assert!(queue_id.is_ok());

        let queue_id = queue_id.unwrap();
        let buffer_id = manager.create_command_buffer(queue_id);
        assert!(buffer_id.is_ok());
    }

    #[test]
    fn test_thread_safe_backend() {
        let config = MetalConfig::default();
        let backend = ThreadSafeMetalBackend::new(config);
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        let stats = backend.get_stats();
        assert_eq!(stats.total_allocations, 0);
    }
}
