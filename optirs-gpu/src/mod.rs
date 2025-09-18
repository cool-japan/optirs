// GPU acceleration support for optimizers
//
// This module provides GPU-accelerated implementations of optimizers using CUDA kernels.

use scirs2_core::ndarray_ext::{Array, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuError};

pub mod adagrad_gpu;
pub mod adam_gpu;
pub mod adamw_gpu;
pub mod cross_platform_abstraction;
pub mod cuda_kernels;
pub mod lamb_gpu;
pub mod memory_pool;
pub mod mixed_precision;
pub mod multi_gpu;
pub mod multi_gpu_example;
pub mod multi_gpu_sync;
pub mod performance_benchmarks;
pub mod rmsprop_gpu;
pub mod rocm_backend;
pub mod sgd_gpu;
pub mod tensor_core_optimization;

/// Trait for GPU-accelerated optimizers
pub trait GpuOptimizer<A: Float, D: Dimension> {
    /// Check if GPU acceleration is available
    fn is_gpu_available(&self) -> bool;

    /// Move optimizer state to GPU
    fn to_gpu(&mut self) -> Result<(), GpuOptimError>;

    /// Move optimizer state back to CPU
    fn to_cpu(&mut self) -> Result<(), GpuOptimError>;

    /// Perform optimization step on GPU
    fn step_gpu(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<(), GpuOptimError>;
}

/// Error type for GPU optimizer operations
#[derive(Debug, thiserror::Error)]
pub enum GpuOptimError {
    /// GPU backend error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),

    /// Unsupported operation
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    /// Invalid state
    #[error("Invalid optimizer state: {0}")]
    InvalidState(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Not initialized
    #[error("GPU optimizer not initialized")]
    NotInitialized,
}

/// GPU optimizer configuration
#[derive(Debug, Clone)]
pub struct GpuOptimizerConfig {
    /// GPU backend to use
    pub backend: GpuBackend,

    /// Enable mixed precision training
    pub mixed_precision: bool,

    /// Loss scaling factor for mixed precision
    pub loss_scale: f32,

    /// Dynamic loss scaling
    pub dynamic_loss_scaling: bool,

    /// Memory pool size in bytes
    pub memory_pool_size: usize,

    /// Number of GPUs for multi-GPU training
    pub num_gpus: usize,

    /// Enable gradient clipping
    pub gradient_clipping: bool,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for GpuOptimizerConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::default(),
            mixed_precision: false,
            loss_scale: 1024.0,
            dynamic_loss_scaling: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            num_gpus: 1,
            gradient_clipping: false,
            max_grad_norm: 1.0,
        }
    }
}

/// GPU memory manager for optimizer states
#[allow(dead_code)]
pub struct GpuOptimizerMemory<A: Float> {
    /// GPU context
    context: Arc<GpuContext>,

    /// Parameter buffer on GPU
    params_gpu: Option<GpuBuffer<A>>,

    /// Gradient buffer on GPU
    grads_gpu: Option<GpuBuffer<A>>,

    /// First moment buffer (for Adam-like optimizers)
    m_gpu: Option<GpuBuffer<A>>,

    /// Second moment buffer (for Adam-like optimizers)
    v_gpu: Option<GpuBuffer<A>>,

    /// Size of buffers
    size: usize,

    /// Configuration
    config: GpuOptimizerConfig,
}

impl<A: Float + Send + Sync> GpuOptimizerMemory<A> {
    /// Create new GPU memory manager
    pub fn new(size: usize, config: GpuOptimizerConfig) -> Result<Self, GpuOptimError> {
        let context = Arc::new(GpuContext::new(config.backend)?);

        Ok(Self {
            context,
            params_gpu: None,
            grads_gpu: None,
            m_gpu: None,
            v_gpu: None,
            size,
            config,
        })
    }

    /// Allocate GPU buffers
    pub fn allocate(&mut self) -> Result<(), GpuOptimError> {
        self.params_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.grads_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.m_gpu = Some(self.context.create_buffer::<A>(self.size));
        self.v_gpu = Some(self.context.create_buffer::<A>(self.size));
        Ok(())
    }

    /// Copy parameters to GPU
    #[allow(clippy::too_many_arguments)]
    pub fn copy_params_to_gpu<S, D>(
        &mut self,
        params: &ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        if let Some(ref params_gpu) = self.params_gpu {
            let params_slice = params.as_slice().ok_or_else(|| {
                GpuOptimError::InvalidState("Parameters must be contiguous".to_string())
            })?;
            params_gpu.copy_from_host(params_slice);
            Ok(())
        } else {
            Err(GpuOptimError::NotInitialized)
        }
    }

    /// Copy parameters from GPU
    #[allow(clippy::too_many_arguments)]
    pub fn copy_params_from_gpu<S, D>(
        &self,
        params: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        if let Some(ref params_gpu) = self.params_gpu {
            let params_slice = params.as_slice_mut().ok_or_else(|| {
                GpuOptimError::InvalidState("Parameters must be contiguous".to_string())
            })?;
            params_gpu.copy_to_host(params_slice);
            Ok(())
        } else {
            Err(GpuOptimError::NotInitialized)
        }
    }

    /// Get GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Get configuration
    pub fn config(&self) -> &GpuOptimizerConfig {
        &self.config
    }
}

/// GPU performance monitoring
#[derive(Debug, Clone)]
pub struct GpuPerformanceMetrics {
    /// Total GPU kernel launches
    pub kernel_launches: usize,

    /// Total GPU memory allocations
    pub memory_allocations: usize,

    /// Peak GPU memory usage (bytes)
    pub peak_memory_usage: usize,

    /// Current GPU memory usage (bytes)
    pub current_memory_usage: usize,

    /// Total GPU computation time (microseconds)
    pub total_gpu_time_us: u64,

    /// Average kernel execution time (microseconds)
    pub avg_kernel_time_us: f64,

    /// Memory transfer time CPU->GPU (microseconds)
    pub cpu_to_gpu_transfer_time_us: u64,

    /// Memory transfer time GPU->CPU (microseconds)
    pub gpu_to_cpu_transfer_time_us: u64,

    /// GPU utilization percentage
    pub gpu_utilization: f32,

    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_utilization: f32,

    /// Number of synchronization events
    pub synchronization_events: usize,
}

impl Default for GpuPerformanceMetrics {
    fn default() -> Self {
        Self {
            kernel_launches: 0,
            memory_allocations: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            total_gpu_time_us: 0,
            avg_kernel_time_us: 0.0,
            cpu_to_gpu_transfer_time_us: 0,
            gpu_to_cpu_transfer_time_us: 0,
            gpu_utilization: 0.0,
            memory_bandwidth_utilization: 0.0,
            synchronization_events: 0,
        }
    }
}

impl GpuPerformanceMetrics {
    /// Record a kernel launch
    pub fn record_kernel_launch(&mut self, execution_timeus: u64) {
        self.kernel_launches += 1;
        self.total_gpu_time_us += execution_time_us;

        // Update average kernel time
        self.avg_kernel_time_us = self.total_gpu_time_us as f64 / self.kernel_launches as f64;
    }

    /// Record memory allocation
    pub fn record_memory_allocation(&mut self, bytes: usize) {
        self.memory_allocations += 1;
        self.current_memory_usage += bytes;

        if self.current_memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = self.current_memory_usage;
        }
    }

    /// Record memory deallocation
    pub fn record_memory_deallocation(&mut self, bytes: usize) {
        self.current_memory_usage = self.current_memory_usage.saturating_sub(bytes);
    }

    /// Record memory transfer time
    pub fn record_transfer_time(&mut self, cpu_to_gpu: bool, timeus: u64) {
        if cpu_to_gpu {
            self.cpu_to_gpu_transfer_time_us += time_us;
        } else {
            self.gpu_to_cpu_transfer_time_us += time_us;
        }
    }

    /// Record synchronization event
    pub fn record_synchronization(&mut self) {
        self.synchronization_events += 1;
    }

    /// Get total operation time
    pub fn total_time_us(&self) -> u64 {
        self.total_gpu_time_us + self.cpu_to_gpu_transfer_time_us + self.gpu_to_cpu_transfer_time_us
    }

    /// Get operations per second
    pub fn operations_per_second(&self) -> f64 {
        if self.total_time_us() == 0 {
            0.0
        } else {
            (self.kernel_launches as f64) / (self.total_time_us() as f64 / 1_000_000.0)
        }
    }

    /// Get memory efficiency (operations per MB)
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_memory_usage == 0 {
            0.0
        } else {
            (self.kernel_launches as f64) / (self.peak_memory_usage as f64 / (1024.0 * 1024.0))
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,

    /// Total GPU memory (bytes)
    pub total_memory: usize,

    /// Available GPU memory (bytes)
    pub available_memory: usize,

    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),

    /// Number of streaming multiprocessors
    pub multiprocessor_count: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Maximum shared memory per block
    pub max_shared_memory_per_block: usize,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gb_s: f32,

    /// Base clock frequency (MHz)
    pub base_clock_mhz: u32,

    /// Memory clock frequency (MHz)
    pub memory_clock_mhz: u32,

    /// Supports tensor cores
    pub supports_tensor_cores: bool,

    /// Supports mixed precision
    pub supports_mixed_precision: bool,
}

/// Advanced GPU optimizer with performance monitoring
#[allow(dead_code)]
pub struct AdvancedGpuOptimizer<A: Float, D: Dimension> {
    /// GPU memory manager
    memory: GpuOptimizerMemory<A>,

    /// Performance metrics
    performance_metrics: GpuPerformanceMetrics,

    /// Device information
    device_info: Option<GpuDeviceInfo>,

    /// Automatic memory management
    auto_memory_management: bool,

    /// Memory usage threshold for cleanup (percentage)
    memory_cleanup_threshold: f32,

    /// Kernel cache for reusing compiled kernels
    kernel_cache: std::collections::HashMap<String, Arc<dyn std::any::Any + Send + Sync>>,

    /// Error recovery strategies
    error_recovery_enabled: bool,

    /// Maximum retry attempts for failed operations
    max_retry_attempts: usize,
}

impl<A: Float, D: Dimension + Send + Sync> AdvancedGpuOptimizer<A, D> {
    /// Create new advanced GPU optimizer
    pub fn new(config: GpuOptimizerConfig) -> Result<Self, GpuOptimError> {
        let memory = GpuOptimizerMemory::new(0_config)?;

        Ok(Self {
            memory,
            performance_metrics: GpuPerformanceMetrics::default(),
            device_info: None,
            auto_memory_management: true,
            memory_cleanup_threshold: 0.8,
            kernel_cache: std::collections::HashMap::new(),
            error_recovery_enabled: true,
            max_retry_attempts: 3,
        })
    }

    /// Initialize device information
    pub fn initialize_device_info(&mut self) -> Result<(), GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            let context = self.memory.context();

            self.device_info = Some(GpuDeviceInfo {
                name: context.device_name()?,
                total_memory: context.total_memory()?,
                available_memory: context.available_memory()?,
                compute_capability: context.compute_capability()?,
                multiprocessor_count: context.multiprocessor_count()?,
                max_threads_per_block: context.max_threads_per_block()?,
                max_shared_memory_per_block: context.max_shared_memory_per_block()?,
                memory_bandwidth_gb_s: context.memory_bandwidth_gb_s()?,
                base_clock_mhz: context.base_clock_mhz()?,
                memory_clock_mhz: context.memory_clock_mhz()?,
                supports_tensor_cores: context.supports_tensor_cores()?,
                supports_mixed_precision: context.supports_mixed_precision()?,
            });
        }

        Ok(())
    }

    /// Enable/disable automatic memory management
    pub fn set_auto_memory_management(&mut self, enabled: bool) {
        self.auto_memory_management = enabled;
    }

    /// Set memory cleanup threshold
    pub fn set_memory_cleanup_threshold(&mut self, threshold: f32) {
        self.memory_cleanup_threshold = threshold.max(0.0).min(1.0);
    }

    /// Perform memory cleanup if needed
    pub fn maybe_cleanup_memory(&mut self) -> Result<(), GpuOptimError> {
        if !self.auto_memory_management {
            return Ok(());
        }

        if let Some(ref device_info) = self.device_info {
            let usage_ratio = self.performance_metrics.current_memory_usage as f32
                / device_info.total_memory as f32;

            if usage_ratio > self.memory_cleanup_threshold {
                self.cleanup_memory()?;
            }
        }

        Ok(())
    }

    /// Force memory cleanup
    pub fn cleanup_memory(&mut self) -> Result<(), GpuOptimError> {
        // Clear kernel cache
        self.kernel_cache.clear();

        // Force garbage collection on GPU
        #[cfg(feature = "gpu")]
        {
            self.memory.context().synchronize()?;
            self.memory.context().garbage_collect()?;
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &GpuPerformanceMetrics {
        &self.performance_metrics
    }

    /// Get device information
    pub fn device_info(&self) -> Option<&GpuDeviceInfo> {
        self.device_info.as_ref()
    }

    /// Execute operation with error recovery
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_recovery<F, R>(&mut self, operation: F) -> Result<R, GpuOptimError>
    where
        F: Fn() -> Result<R, GpuOptimError>,
    {
        if !self.error_recovery_enabled {
            return operation();
        }

        let mut attempts = 0;
        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    attempts += 1;

                    if attempts >= self.max_retry_attempts {
                        return Err(error);
                    }

                    // Try to recover from common errors
                    match &error {
                        GpuOptimError::GpuError(_) => {
                            // Reset GPU context and retry
                            #[cfg(feature = "gpu")]
                            {
                                if let Err(_) = self.memory.context().reset() {
                                    return Err(error);
                                }
                            }
                        }
                        _ => return Err(error),
                    }
                }
            }
        }
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        let metrics = &self.performance_metrics;

        format!(
            "GPU Performance Report\n\
             =====================\n\
             Kernel Launches: {}\n\
             Total GPU Time: {:.2} ms\n\
             Average Kernel Time: {:.2} μs\n\
             Memory Allocations: {}\n\
             Peak Memory Usage: {:.2} MB\n\
             Current Memory Usage: {:.2} MB\n\
             CPU->GPU Transfer Time: {:.2} ms\n\
             GPU->CPU Transfer Time: {:.2} ms\n\
             Operations per Second: {:.2}\n\
             Memory Efficiency: {:.2} ops/MB\n\
             Synchronization Events: {}\n",
            metrics.kernel_launches,
            metrics.total_gpu_time_us as f64 / 1000.0,
            metrics.avg_kernel_time_us,
            metrics.memory_allocations,
            metrics.peak_memory_usage as f64 / (1024.0 * 1024.0),
            metrics.current_memory_usage as f64 / (1024.0 * 1024.0),
            metrics.cpu_to_gpu_transfer_time_us as f64 / 1000.0,
            metrics.gpu_to_cpu_transfer_time_us as f64 / 1000.0,
            metrics.operations_per_second(),
            metrics.memory_efficiency(),
            metrics.synchronization_events,
        )
    }
}

/// Helper functions for GPU operations
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is available for the given backend
    #[allow(dead_code)]
    pub fn is_gpu_available(backend: GpuBackend) -> bool {
        match GpuContext::new(_backend) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Get the optimal GPU backend for the current system
    #[allow(dead_code)]
    pub fn get_optimal_backend() -> GpuBackend {
        // Try backends in order of preference
        let backends = [
            GpuBackend::Cuda,
            GpuBackend::Metal,
            GpuBackend::Rocm,
            GpuBackend::Wgpu,
        ];

        for backend in &backends {
            if is_gpu_available(*backend) {
                return *backend;
            }
        }

        GpuBackend::Cpu
    }

    /// Calculate optimal block size for GPU kernels
    #[allow(dead_code)]
    pub fn calculate_block_size(n: usize, maxthreads: usize) -> (usize, usize) {
        let block_size = 256.min(max_threads);
        let grid_size = (n + block_size - 1) / block_size;
        (grid_size, block_size)
    }

    /// Calculate optimal block size for 2D operations
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_block_size_2d(
        width: usize,
        height: usize,
        max_threads: usize,
    ) -> ((usize, usize), (usize, usize)) {
        let _total_threads = width * height;
        let block_dim_x = 16.min((max_threads as f64).sqrt() as usize);
        let block_dim_y = (max_threads / block_dim_x).min(16);

        let grid_dim_x = (width + block_dim_x - 1) / block_dim_x;
        let grid_dim_y = (height + block_dim_y - 1) / block_dim_y;

        ((grid_dim_x, grid_dim_y), (block_dim_x, block_dim_y))
    }

    /// Estimate memory bandwidth utilization
    pub fn estimate_memory_bandwidth_utilization(
        bytes_transferred: usize,
        time_us: u64,
        peak_bandwidth_gb_s: f32,
    ) -> f32 {
        if time_us == 0 {
            return 0.0;
        }

        let actual_bandwidth_gb_s = (bytes_transferred as f64)
            / (time_us as f64 / 1_000_000.0)
            / (1024.0 * 1024.0 * 1024.0);
        (actual_bandwidth_gb_s as f32 / peak_bandwidth_gb_s).min(1.0)
    }

    /// Calculate occupancy for GPU kernels
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_occupancy(
        threads_per_block: usize,
        shared_memory_per_block: usize,
        registers_per_thread: usize,
        max_threads_per_sm: usize,
        max_blocks_per_sm: usize,
        max_shared_memory_per_sm: usize,
        max_registers_per_sm: usize,
    ) -> f32 {
        if threads_per_block == 0 {
            return 0.0;
        }

        // Limit by threads per SM
        let blocks_by_threads = max_threads_per_sm / threads_per_block;

        // Limit by shared memory per SM
        let blocks_by_shared_memory = if shared_memory_per_block > 0 {
            max_shared_memory_per_sm / shared_memory_per_block
        } else {
            max_blocks_per_sm
        };

        // Limit by registers per SM
        let blocks_by_registers = if registers_per_thread > 0 {
            max_registers_per_sm / (registers_per_thread * threads_per_block)
        } else {
            max_blocks_per_sm
        };

        let active_blocks = blocks_by_threads
            .min(blocks_by_shared_memory)
            .min(blocks_by_registers)
            .min(max_blocks_per_sm);

        let active_threads = active_blocks * threads_per_block;
        active_threads as f32 / max_threads_per_sm as f32
    }

    /// Choose optimal data type for mixed precision
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn choose_optimal_dtype(
        supports_fp16: bool,
        supports_bf16: bool,
        supports_tf32: bool,
        prefer_accuracy: bool,
    ) -> String {
        if prefer_accuracy {
            if supports_tf32 {
                "_tf32".to_string()
            } else {
                "fp32".to_string()
            }
        } else {
            if supports_bf16 {
                "_bf16".to_string()
            } else if supports_fp16 {
                "_fp16".to_string()
            } else if supports_tf32 {
                "_tf32".to_string()
            } else {
                "fp32".to_string()
            }
        }
    }

    /// Benchmark GPU operation
    pub fn benchmark_operation<F>(operation: F, iterations: usize) -> Result<f64, GpuOptimError>
    where
        F: Fn() -> Result<(), GpuOptimError>,
    {
        let start = std::time::Instant::now();

        for _ in 0..iterations {
            _operation()?;
        }

        // Synchronize to ensure all operations complete
        #[cfg(feature = "gpu")]
        {
            // GPU synchronization would go here
        }

        let elapsed = start.elapsed();
        Ok(elapsed.as_secs_f64() / iterations as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array1;

    #[test]
    fn test_gpu_optimizer_config_default() {
        let config = GpuOptimizerConfig::default();
        assert!(!config.mixed_precision);
        assert_eq!(config.loss_scale, 1024.0);
        assert!(config.dynamic_loss_scaling);
        assert_eq!(config.num_gpus, 1);
    }

    #[test]
    fn test_gpu_memory_creation() {
        let config = GpuOptimizerConfig {
            backend: GpuBackend::Cpu, // Use CPU backend for testing
            ..Default::default()
        };

        let memory = GpuOptimizerMemory::<f32>::new(1000, config);
        assert!(memory.is_ok());
    }

    #[test]
    fn test_optimal_backend_selection() {
        let backend = utils::get_optimal_backend();
        // Should at least return CPU backend
        assert!(matches!(
            backend,
            GpuBackend::Cuda
                | GpuBackend::Metal
                | GpuBackend::Rocm
                | GpuBackend::Wgpu
                | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_block_size_calculation() {
        let (grid, block) = utils::calculate_block_size(1000, 1024);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // (1000 + 255) / 256 = 4

        let (grid, block) = utils::calculate_block_size(100, 128);
        assert_eq!(block, 128);
        assert_eq!(grid, 1); // (100 + 127) / 128 = 1
    }
}
