// CUDA backend for GPU memory management
//
// This module provides NVIDIA CUDA-specific memory management functionality,
// including device memory allocation, unified memory, streams, and performance
// optimization features specific to CUDA GPUs.

#[allow(dead_code)]
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// CUDA memory backend implementation
pub struct CudaMemoryBackend {
    /// Backend configuration
    config: CudaConfig,
    /// Device properties
    device_properties: CudaDeviceProperties,
    /// Active memory contexts
    contexts: HashMap<u32, CudaContext>,
    /// Memory pools
    memory_pools: HashMap<CudaMemoryType, CudaMemoryPool>,
    /// Statistics
    stats: CudaStats,
    /// Stream management
    stream_manager: CudaStreamManager,
}

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// Device ID to use
    pub device_id: u32,
    /// Enable unified memory
    pub enable_unified_memory: bool,
    /// Enable memory pools
    pub enable_memory_pools: bool,
    /// Enable async memory operations
    pub enable_async_ops: bool,
    /// Memory pool growth size
    pub pool_growth_size: usize,
    /// Enable memory mapped host memory
    pub enable_mapped_memory: bool,
    /// Enable CUDA graphs for memory ops
    pub enable_cuda_graphs: bool,
    /// Enable cooperative groups
    pub enable_cooperative_groups: bool,
    /// Maximum number of streams
    pub max_streams: u32,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_unified_memory: true,
            enable_memory_pools: true,
            enable_async_ops: true,
            pool_growth_size: 64 * 1024 * 1024, // 64MB
            enable_mapped_memory: true,
            enable_cuda_graphs: false, // Experimental
            enable_cooperative_groups: false,
            max_streams: 16,
        }
    }
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_global_memory: usize,
    pub shared_memory_per_block: usize,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_blocks_per_multiprocessor: u32,
    pub multiprocessor_count: u32,
    pub memory_clock_rate: u32,
    pub memory_bus_width: u32,
    pub l2_cache_size: usize,
    pub unified_addressing: bool,
    pub managed_memory: bool,
    pub concurrent_kernels: bool,
    pub async_engine_count: u32,
}

/// CUDA memory types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CudaMemoryType {
    Device,
    Host,
    Unified,
    Mapped,
    Array,
    Texture,
}

/// CUDA context for managing device state
pub struct CudaContext {
    /// Context handle (simulated)
    pub handle: *mut c_void,
    /// Device ID
    pub device_id: u32,
    /// Context flags
    pub flags: CudaContextFlags,
    /// Creation time
    pub created_at: Instant,
    /// Active streams
    pub streams: Vec<CudaStream>,
}

/// CUDA context creation flags
#[derive(Debug, Clone)]
pub struct CudaContextFlags {
    pub sched_auto: bool,
    pub sched_spin: bool,
    pub sched_yield: bool,
    pub sched_blocking_sync: bool,
    pub map_host: bool,
    pub lmem_resize_to_max: bool,
}

impl Default for CudaContextFlags {
    fn default() -> Self {
        Self {
            sched_auto: true,
            sched_spin: false,
            sched_yield: false,
            sched_blocking_sync: false,
            map_host: false,
            lmem_resize_to_max: false,
        }
    }
}

/// CUDA stream for asynchronous operations
pub struct CudaStream {
    /// Stream handle (simulated)
    pub handle: *mut c_void,
    /// Stream ID
    pub id: u32,
    /// Stream priority
    pub priority: i32,
    /// Stream flags
    pub flags: CudaStreamFlags,
    /// Creation time
    pub created_at: Instant,
    /// Operations queue
    pub operations: std::collections::VecDeque<CudaOperation>,
}

/// CUDA stream flags
#[derive(Debug, Clone)]
pub struct CudaStreamFlags {
    pub default: bool,
    pub non_blocking: bool,
    pub per_thread: bool,
}

impl Default for CudaStreamFlags {
    fn default() -> Self {
        Self {
            default: true,
            non_blocking: false,
            per_thread: false,
        }
    }
}

/// CUDA asynchronous operation
#[derive(Debug, Clone)]
pub struct CudaOperation {
    pub op_type: CudaOperationType,
    pub src_ptr: Option<*mut c_void>,
    pub dst_ptr: Option<*mut c_void>,
    pub size: usize,
    pub timestamp: Instant,
}

/// Types of CUDA operations
#[derive(Debug, Clone)]
pub enum CudaOperationType {
    MemcpyHostToDevice,
    MemcpyDeviceToHost,
    MemcpyDeviceToDevice,
    MemcpyAsync,
    MemsetAsync,
    KernelLaunch,
    EventRecord,
    EventSynchronize,
}

/// CUDA memory pool
pub struct CudaMemoryPool {
    /// Memory type
    memory_type: CudaMemoryType,
    /// Pool handle (simulated)
    handle: *mut c_void,
    /// Current size
    current_size: usize,
    /// Maximum size
    max_size: usize,
    /// Used size
    used_size: usize,
    /// Free blocks
    free_blocks: std::collections::VecDeque<CudaMemoryBlock>,
    /// Allocated blocks
    allocated_blocks: HashMap<*mut c_void, CudaMemoryBlock>,
}

/// CUDA memory block
#[derive(Debug, Clone)]
pub struct CudaMemoryBlock {
    pub ptr: *mut c_void,
    pub size: usize,
    pub memory_type: CudaMemoryType,
    pub allocated_at: Instant,
    pub last_access: Option<Instant>,
    pub ref_count: u32,
}

impl CudaMemoryPool {
    pub fn new(memory_type: CudaMemoryType, max_size: usize) -> Self {
        Self {
            memory_type,
            handle: std::ptr::null_mut(),
            current_size: 0,
            max_size,
            used_size: 0,
            free_blocks: std::collections::VecDeque::new(),
            allocated_blocks: HashMap::new(),
        }
    }

    /// Allocate from pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut c_void, CudaError> {
        // Try to find suitable free block
        for i in 0..self.free_blocks.len() {
            if self.free_blocks[i].size >= size {
                let mut block = self.free_blocks.remove(i).unwrap();

                // Split block if much larger
                if block.size > size * 2 {
                    let remaining_block = CudaMemoryBlock {
                        ptr: unsafe { block.ptr.add(size) },
                        size: block.size - size,
                        memory_type: block.memory_type.clone(),
                        allocated_at: block.allocated_at,
                        last_access: None,
                        ref_count: 0,
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
            return Err(CudaError::OutOfMemory(
                "Pool size limit exceeded".to_string(),
            ));
        }

        let ptr = self.cuda_malloc(size)?;
        let block = CudaMemoryBlock {
            ptr,
            size,
            memory_type: self.memory_type.clone(),
            allocated_at: Instant::now(),
            last_access: Some(Instant::now()),
            ref_count: 1,
        };

        self.allocated_blocks.insert(ptr, block);
        self.current_size += size;
        self.used_size += size;

        Ok(ptr)
    }

    /// Free back to pool
    pub fn free(&mut self, ptr: *mut c_void) -> Result<(), CudaError> {
        if let Some(block) = self.allocated_blocks.remove(&ptr) {
            self.used_size -= block.size;

            // Add to free blocks
            self.free_blocks.push_back(CudaMemoryBlock {
                ptr: block.ptr,
                size: block.size,
                memory_type: block.memory_type,
                allocated_at: block.allocated_at,
                last_access: None,
                ref_count: 0,
            });

            // Try to coalesce adjacent blocks
            self.coalesce_free_blocks();

            Ok(())
        } else {
            Err(CudaError::InvalidPointer(
                "Pointer not found in pool".to_string(),
            ))
        }
    }

    fn coalesce_free_blocks(&mut self) {
        // Sort free blocks by address
        let mut blocks: Vec<CudaMemoryBlock> = self.free_blocks.drain(..).collect();
        blocks.sort_by_key(|block| block.ptr as usize);

        let mut coalesced = Vec::new();
        let mut current_block: Option<CudaMemoryBlock> = None;

        for block in blocks {
            match current_block.take() {
                None => current_block = Some(block),
                Some(mut prev_block) => {
                    let prev_end = prev_block.ptr as usize + prev_block.size;
                    let block_start = block.ptr as usize;

                    if prev_end == block_start {
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

    fn cuda_malloc(&self, size: usize) -> Result<*mut c_void, CudaError> {
        // Simulate CUDA memory allocation
        match self.memory_type {
            CudaMemoryType::Device => {
                // cudaMalloc equivalent
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            CudaMemoryType::Host => {
                // cudaMallocHost equivalent
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            CudaMemoryType::Unified => {
                // cudaMallocManaged equivalent
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            CudaMemoryType::Mapped => {
                // cudaHostAlloc with mapping flags
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            _ => Err(CudaError::UnsupportedOperation(
                "Unsupported memory type for allocation".to_string(),
            )),
        }
    }
}

/// CUDA stream manager
pub struct CudaStreamManager {
    /// Available streams
    streams: Vec<CudaStream>,
    /// Stream pool for reuse
    stream_pool: std::collections::VecDeque<CudaStream>,
    /// Next stream ID
    next_stream_id: u32,
    /// Configuration
    config: CudaStreamConfig,
}

/// Stream manager configuration
#[derive(Debug, Clone)]
pub struct CudaStreamConfig {
    pub default_priority: i32,
    pub enable_priorities: bool,
    pub max_operations_per_stream: usize,
}

impl Default for CudaStreamConfig {
    fn default() -> Self {
        Self {
            default_priority: 0,
            enable_priorities: true,
            max_operations_per_stream: 1000,
        }
    }
}

impl CudaStreamManager {
    pub fn new(config: CudaStreamConfig) -> Self {
        Self {
            streams: Vec::new(),
            stream_pool: std::collections::VecDeque::new(),
            next_stream_id: 0,
            config,
        }
    }

    /// Create new stream
    pub fn create_stream(&mut self, priority: Option<i32>) -> Result<u32, CudaError> {
        let stream_id = self.next_stream_id;
        self.next_stream_id += 1;

        let stream = CudaStream {
            handle: std::ptr::null_mut(), // Would be actual CUDA stream
            id: stream_id,
            priority: priority.unwrap_or(self.config.default_priority),
            flags: CudaStreamFlags::default(),
            created_at: Instant::now(),
            operations: std::collections::VecDeque::new(),
        };

        self.streams.push(stream);
        Ok(stream_id)
    }

    /// Destroy stream
    pub fn destroy_stream(&mut self, stream_id: u32) -> Result<(), CudaError> {
        if let Some(pos) = self.streams.iter().position(|s| s.id == stream_id) {
            let stream = self.streams.remove(pos);
            // Clean up stream resources
            Ok(())
        } else {
            Err(CudaError::InvalidStream("Stream not found".to_string()))
        }
    }

    /// Add operation to stream
    pub fn add_operation(
        &mut self,
        stream_id: u32,
        operation: CudaOperation,
    ) -> Result<(), CudaError> {
        if let Some(stream) = self.streams.iter_mut().find(|s| s.id == stream_id) {
            if stream.operations.len() >= self.config.max_operations_per_stream {
                return Err(CudaError::StreamFull(
                    "Stream operation queue is full".to_string(),
                ));
            }

            stream.operations.push_back(operation);
            Ok(())
        } else {
            Err(CudaError::InvalidStream("Stream not found".to_string()))
        }
    }

    /// Synchronize stream
    pub fn synchronize_stream(&mut self, stream_id: u32) -> Result<(), CudaError> {
        // First, collect all operations from the stream
        let mut operations = Vec::new();
        if let Some(stream) = self.streams.iter_mut().find(|s| s.id == stream_id) {
            while let Some(operation) = stream.operations.pop_front() {
                operations.push(operation);
            }
        } else {
            return Err(CudaError::InvalidStream("Stream not found".to_string()));
        }

        // Now execute all operations
        for operation in operations {
            self.execute_operation(operation)?;
        }

        Ok(())
    }

    fn execute_operation(&self, operation: CudaOperation) -> Result<(), CudaError> {
        // Simulate operation execution
        match operation.op_type {
            CudaOperationType::MemcpyHostToDevice => {
                // Simulate cudaMemcpy
                std::thread::sleep(Duration::from_micros(100));
            }
            CudaOperationType::MemcpyDeviceToHost => {
                // Simulate cudaMemcpy
                std::thread::sleep(Duration::from_micros(100));
            }
            CudaOperationType::MemcpyDeviceToDevice => {
                // Simulate cudaMemcpy
                std::thread::sleep(Duration::from_micros(50));
            }
            CudaOperationType::MemcpyAsync => {
                // Simulate cudaMemcpyAsync
                std::thread::sleep(Duration::from_micros(10));
            }
            _ => {
                // Other operations
            }
        }
        Ok(())
    }
}

/// CUDA statistics
#[derive(Debug, Clone, Default)]
pub struct CudaStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub device_memory_used: usize,
    pub host_memory_used: usize,
    pub unified_memory_used: usize,
    pub stream_operations: u64,
    pub kernel_launches: u64,
    pub memory_transfers: u64,
    pub average_allocation_time: Duration,
    pub peak_memory_usage: usize,
}

impl CudaMemoryBackend {
    /// Create new CUDA backend
    pub fn new(config: CudaConfig) -> Result<Self, CudaError> {
        // Initialize CUDA device
        let device_properties = Self::query_device_properties(config.device_id)?;

        // Create memory pools
        let mut memory_pools = HashMap::new();
        if config.enable_memory_pools {
            let pool_size = device_properties.total_global_memory / 4; // Use 1/4 of total memory
            memory_pools.insert(
                CudaMemoryType::Device,
                CudaMemoryPool::new(CudaMemoryType::Device, pool_size),
            );
            memory_pools.insert(
                CudaMemoryType::Host,
                CudaMemoryPool::new(CudaMemoryType::Host, pool_size),
            );

            if config.enable_unified_memory && device_properties.managed_memory {
                memory_pools.insert(
                    CudaMemoryType::Unified,
                    CudaMemoryPool::new(CudaMemoryType::Unified, pool_size),
                );
            }
        }

        let stream_manager = CudaStreamManager::new(CudaStreamConfig::default());

        Ok(Self {
            config,
            device_properties,
            contexts: HashMap::new(),
            memory_pools,
            stats: CudaStats::default(),
            stream_manager,
        })
    }

    /// Query device properties
    fn query_device_properties(device_id: u32) -> Result<CudaDeviceProperties, CudaError> {
        // Simulate querying CUDA device properties
        Ok(CudaDeviceProperties {
            device_id,
            name: format!("CUDA Device {}", device_id),
            compute_capability: (7, 5), // Simulate Turing architecture
            total_global_memory: 8 * 1024 * 1024 * 1024, // 8GB
            shared_memory_per_block: 48 * 1024, // 48KB
            warp_size: 32,
            max_threads_per_block: 1024,
            max_blocks_per_multiprocessor: 16,
            multiprocessor_count: 68,
            memory_clock_rate: 7001000, // 7 GHz
            memory_bus_width: 256,
            l2_cache_size: 4 * 1024 * 1024, // 4MB
            unified_addressing: true,
            managed_memory: true,
            concurrent_kernels: true,
            async_engine_count: 2,
        })
    }

    /// Allocate device memory
    pub fn allocate(
        &mut self,
        size: usize,
        memory_type: CudaMemoryType,
    ) -> Result<*mut c_void, CudaError> {
        let start_time = Instant::now();

        let ptr = if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.allocate(size)?
            } else {
                return Err(CudaError::UnsupportedMemoryType(
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
            CudaMemoryType::Device => self.stats.device_memory_used += size,
            CudaMemoryType::Host => self.stats.host_memory_used += size,
            CudaMemoryType::Unified => self.stats.unified_memory_used += size,
            _ => {}
        }

        let allocation_time = start_time.elapsed();
        let total_time = self.stats.average_allocation_time.as_nanos() as u64
            * (self.stats.total_allocations - 1)
            + allocation_time.as_nanos() as u64;
        self.stats.average_allocation_time =
            Duration::from_nanos(total_time / self.stats.total_allocations);

        let current_usage = self.stats.device_memory_used
            + self.stats.host_memory_used
            + self.stats.unified_memory_used;
        if current_usage > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = current_usage;
        }

        Ok(ptr)
    }

    fn direct_allocate(
        &self,
        size: usize,
        memory_type: CudaMemoryType,
    ) -> Result<*mut c_void, CudaError> {
        // Simulate direct CUDA allocation
        match memory_type {
            CudaMemoryType::Device => {
                // cudaMalloc
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            CudaMemoryType::Host => {
                // cudaMallocHost
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            CudaMemoryType::Unified => {
                // cudaMallocManaged
                if !self.device_properties.managed_memory {
                    return Err(CudaError::UnsupportedOperation(
                        "Unified memory not supported".to_string(),
                    ));
                }
                Ok(unsafe {
                    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 256))
                        as *mut c_void
                })
            }
            _ => Err(CudaError::UnsupportedMemoryType(
                "Unsupported memory type".to_string(),
            )),
        }
    }

    /// Free device memory
    pub fn free(&mut self, ptr: *mut c_void, memory_type: CudaMemoryType) -> Result<(), CudaError> {
        if self.config.enable_memory_pools {
            if let Some(pool) = self.memory_pools.get_mut(&memory_type) {
                pool.free(ptr)?;
            } else {
                return Err(CudaError::UnsupportedMemoryType(
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

    /// Copy memory
    pub fn memcpy(
        &mut self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: CudaMemcpyKind,
    ) -> Result<(), CudaError> {
        let operation = CudaOperation {
            op_type: match kind {
                CudaMemcpyKind::HostToDevice => CudaOperationType::MemcpyHostToDevice,
                CudaMemcpyKind::DeviceToHost => CudaOperationType::MemcpyDeviceToHost,
                CudaMemcpyKind::DeviceToDevice => CudaOperationType::MemcpyDeviceToDevice,
                CudaMemcpyKind::HostToHost => CudaOperationType::MemcpyAsync,
            },
            src_ptr: Some(src as *mut c_void),
            dst_ptr: Some(dst),
            size,
            timestamp: Instant::now(),
        };

        // Execute synchronously for now
        self.stream_manager.execute_operation(operation)?;
        self.stats.memory_transfers += 1;

        Ok(())
    }

    /// Asynchronous memory copy
    pub fn memcpy_async(
        &mut self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: CudaMemcpyKind,
        stream_id: u32,
    ) -> Result<(), CudaError> {
        let operation = CudaOperation {
            op_type: CudaOperationType::MemcpyAsync,
            src_ptr: Some(src as *mut c_void),
            dst_ptr: Some(dst),
            size,
            timestamp: Instant::now(),
        };

        self.stream_manager.add_operation(stream_id, operation)?;
        Ok(())
    }

    /// Create CUDA context
    pub fn create_context(&mut self, flags: CudaContextFlags) -> Result<u32, CudaError> {
        let context_id = self.contexts.len() as u32;

        let context = CudaContext {
            handle: std::ptr::null_mut(), // Would be actual CUDA context
            device_id: self.config.device_id,
            flags,
            created_at: Instant::now(),
            streams: Vec::new(),
        };

        self.contexts.insert(context_id, context);
        Ok(context_id)
    }

    /// Get device properties
    pub fn get_device_properties(&self) -> &CudaDeviceProperties {
        &self.device_properties
    }

    /// Get statistics
    pub fn get_stats(&self) -> &CudaStats {
        &self.stats
    }

    /// Synchronize device
    pub fn device_synchronize(&mut self) -> Result<(), CudaError> {
        // Synchronize all streams
        let stream_ids: Vec<u32> = self.stream_manager.streams.iter().map(|s| s.id).collect();
        for stream_id in stream_ids {
            self.stream_manager.synchronize_stream(stream_id)?;
        }
        Ok(())
    }

    /// Create stream
    pub fn create_stream(&mut self, priority: Option<i32>) -> Result<u32, CudaError> {
        self.stream_manager.create_stream(priority)
    }

    /// Destroy stream
    pub fn destroy_stream(&mut self, stream_id: u32) -> Result<(), CudaError> {
        self.stream_manager.destroy_stream(stream_id)
    }
}

/// CUDA memory copy kinds
#[derive(Debug, Clone)]
pub enum CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
}

/// CUDA errors
#[derive(Debug, Clone)]
pub enum CudaError {
    DeviceNotFound(String),
    OutOfMemory(String),
    InvalidPointer(String),
    InvalidStream(String),
    StreamFull(String),
    UnsupportedOperation(String),
    UnsupportedMemoryType(String),
    ContextCreationFailed(String),
    KernelLaunchFailed(String),
    SynchronizationFailed(String),
    InternalError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound(msg) => write!(f, "Device not found: {}", msg),
            CudaError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            CudaError::InvalidPointer(msg) => write!(f, "Invalid pointer: {}", msg),
            CudaError::InvalidStream(msg) => write!(f, "Invalid stream: {}", msg),
            CudaError::StreamFull(msg) => write!(f, "Stream full: {}", msg),
            CudaError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            CudaError::UnsupportedMemoryType(msg) => write!(f, "Unsupported memory type: {}", msg),
            CudaError::ContextCreationFailed(msg) => write!(f, "Context creation failed: {}", msg),
            CudaError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            CudaError::SynchronizationFailed(msg) => write!(f, "Synchronization failed: {}", msg),
            CudaError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

/// Thread-safe CUDA backend wrapper
pub struct ThreadSafeCudaBackend {
    backend: Arc<Mutex<CudaMemoryBackend>>,
}

impl ThreadSafeCudaBackend {
    pub fn new(config: CudaConfig) -> Result<Self, CudaError> {
        let backend = CudaMemoryBackend::new(config)?;
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    pub fn allocate(
        &self,
        size: usize,
        memory_type: CudaMemoryType,
    ) -> Result<*mut c_void, CudaError> {
        let mut backend = self.backend.lock().unwrap();
        backend.allocate(size, memory_type)
    }

    pub fn free(&self, ptr: *mut c_void, memory_type: CudaMemoryType) -> Result<(), CudaError> {
        let mut backend = self.backend.lock().unwrap();
        backend.free(ptr, memory_type)
    }

    pub fn get_stats(&self) -> CudaStats {
        let backend = self.backend.lock().unwrap();
        backend.get_stats().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_creation() {
        let config = CudaConfig::default();
        let backend = CudaMemoryBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = CudaMemoryPool::new(CudaMemoryType::Device, 1024 * 1024);
        let ptr = pool.allocate(1024);
        assert!(ptr.is_ok());

        let ptr = ptr.unwrap();
        let result = pool.free(ptr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_manager() {
        let mut manager = CudaStreamManager::new(CudaStreamConfig::default());
        let stream_id = manager.create_stream(Some(1));
        assert!(stream_id.is_ok());

        let stream_id = stream_id.unwrap();
        let result = manager.destroy_stream(stream_id);
        assert!(result.is_ok());
    }

    #[test]
    fn test_thread_safe_backend() {
        let config = CudaConfig::default();
        let backend = ThreadSafeCudaBackend::new(config);
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        let stats = backend.get_stats();
        assert_eq!(stats.total_allocations, 0);
    }
}
