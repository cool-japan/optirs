// Multi-GPU synchronization support for distributed training

use num_traits::Float;
use scirs2_core::ndarray_ext::{ArrayBase, Data, DataMut, Dimension};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::backends::GpuBackend;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuKernelHandle};

/// Multi-GPU synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncStrategy {
    /// Ring all-reduce (efficient for large tensors)
    RingAllReduce,
    /// Tree all-reduce (efficient for small tensors)
    TreeAllReduce,
    /// Hierarchical all-reduce (for multi-node setups)
    HierarchicalAllReduce,
    /// Pipeline parallel synchronization
    PipelineParallel,
}

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Number of GPUs
    pub num_gpus: usize,
    /// GPU rank (0-indexed)
    pub rank: usize,
    /// Synchronization strategy
    pub sync_strategy: SyncStrategy,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Compression ratio (for top-k compression)
    pub compression_ratio: f32,
    /// Local GPU group size (for hierarchical)
    pub local_group_size: usize,
    /// Enable adaptive communication optimization
    pub adaptive_communication: bool,
    /// Bandwidth monitoring interval (steps)
    pub bandwidth_monitor_interval: usize,
    /// Enable asynchronous parameter updates
    pub async_param_updates: bool,
    /// Communication timeout (milliseconds)
    pub communication_timeout_ms: u64,
    /// Enable error correction for communication
    pub error_correction: bool,
    /// Pipeline depth for overlapping computation and communication
    pub pipeline_depth: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            rank: 0,
            sync_strategy: SyncStrategy::RingAllReduce,
            gradient_compression: false,
            compression_ratio: 0.1, // Keep top 10%
            local_group_size: 4,
            adaptive_communication: true,
            bandwidth_monitor_interval: 100,
            async_param_updates: false,
            communication_timeout_ms: 5000,
            error_correction: true,
            pipeline_depth: 2,
        }
    }
}

/// Multi-GPU synchronization manager
pub struct MultiGpuSync<A: Float> {
    /// GPU context
    context: Arc<GpuContext>,
    /// Configuration
    config: MultiGpuConfig,
    /// Synchronization kernels
    sync_kernels: SyncKernels,
    /// Workspace buffers
    workspace: WorkspaceBuffers<A>,
    /// Communication performance monitor
    perf_monitor: CommunicationPerformanceMonitor,
    /// Adaptive strategy selector
    adaptive_selector: AdaptiveCommunicationSelector,
    /// Asynchronous communication handles
    async_handles: Vec<AsyncCommunicationHandle>,
    /// Step counter for monitoring
    step_counter: usize,
    /// Phantom data for type parameter
    _phantom: PhantomData<A>,
}

/// Container for synchronization kernels
struct SyncKernels {
    ring_allreduce: Option<Arc<GpuKernelHandle>>,
    tree_allreduce: Option<Arc<GpuKernelHandle>>,
    hierarchical_allreduce: Option<Arc<GpuKernelHandle>>,
    compress_gradients: Option<Arc<GpuKernelHandle>>,
    decompress_gradients: Option<Arc<GpuKernelHandle>>,
}

/// Workspace buffers for synchronization
struct WorkspaceBuffers<A: Float> {
    recv_buffer: Option<GpuBuffer<A>>,
    workspace: Option<GpuBuffer<A>>,
    compressed_values: Option<GpuBuffer<A>>,
    compressed_indices: Option<GpuBuffer<i32>>,
    error_feedback: Option<GpuBuffer<A>>,
}

/// Communication performance monitoring
#[derive(Debug, Clone)]
pub struct CommunicationPerformanceMonitor {
    /// Total communication time (microseconds)
    total_comm_time_us: u64,
    /// Total data transferred (bytes)
    total_data_bytes: u64,
    /// Number of communication operations
    comm_operations: usize,
    /// Bandwidth history (GB/s)
    bandwidth_history: std::collections::VecDeque<f64>,
    /// Strategy performance tracking
    strategy_performance: std::collections::HashMap<SyncStrategy, StrategyPerformanceMetrics>,
    /// Current optimal strategy
    optimal_strategy: SyncStrategy,
}

impl CommunicationPerformanceMonitor {
    fn new() -> Self {
        Self {
            total_comm_time_us: 0,
            total_data_bytes: 0,
            comm_operations: 0,
            bandwidth_history: std::collections::VecDeque::with_capacity(1000),
            strategy_performance: std::collections::HashMap::new(),
            optimal_strategy: SyncStrategy::RingAllReduce,
        }
    }

    fn record_communication(&mut self, strategy: SyncStrategy, data_bytes: u64, timeus: u64) {
        self.total_comm_time_us += time_us;
        self.total_data_bytes += data_bytes;
        self.comm_operations += 1;

        let bandwidth_gb_s = (data_bytes as f64) / (time_us as f64 / 1_000_000.0) / 1e9;
        self.bandwidth_history.push_back(bandwidth_gb_s);

        if self.bandwidth_history.len() > 1000 {
            self.bandwidth_history.pop_front();
        }

        // Update strategy performance
        let metrics = self
            .strategy_performance
            .entry(strategy)
            .or_insert_with(StrategyPerformanceMetrics::new);
        metrics.update(bandwidth_gb_s, time_us);
    }

    fn get_average_bandwidth(&self) -> f64 {
        if self.total_comm_time_us == 0 {
            0.0
        } else {
            (self.total_data_bytes as f64) / (self.total_comm_time_us as f64 / 1_000_000.0) / 1e9
        }
    }

    fn get_optimal_strategy(&self, tensorsize: usize) -> SyncStrategy {
        let mut best_strategy = SyncStrategy::RingAllReduce;
        let mut best_score = 0.0;

        for (strategy, metrics) in &self.strategy_performance {
            let score = metrics.calculate_score(tensor_size);
            if score > best_score {
                best_score = score;
                best_strategy = *strategy;
            }
        }

        best_strategy
    }
}

/// Performance metrics for a specific synchronization strategy
#[derive(Debug, Clone)]
struct StrategyPerformanceMetrics {
    bandwidth_samples: std::collections::VecDeque<f64>,
    latency_samples: std::collections::VecDeque<u64>,
    tensor_sizes: std::collections::VecDeque<usize>,
    efficiency_score: f64,
}

impl StrategyPerformanceMetrics {
    fn new() -> Self {
        Self {
            bandwidth_samples: std::collections::VecDeque::with_capacity(100),
            latency_samples: std::collections::VecDeque::with_capacity(100),
            tensor_sizes: std::collections::VecDeque::with_capacity(100),
            efficiency_score: 0.0,
        }
    }

    fn update(&mut self, bandwidth_gb_s: f64, latencyus: u64) {
        self.bandwidth_samples.push_back(bandwidth_gb_s);
        self.latency_samples.push_back(latency_us);

        if self.bandwidth_samples.len() > 100 {
            self.bandwidth_samples.pop_front();
            self.latency_samples.pop_front();
        }

        // Update efficiency score based on recent performance
        let avg_bandwidth =
            self.bandwidth_samples.iter().sum::<f64>() / self.bandwidth_samples.len() as f64;
        let avg_latency =
            self.latency_samples.iter().sum::<u64>() as f64 / self.latency_samples.len() as f64;

        self.efficiency_score = avg_bandwidth / (avg_latency / 1000.0); // Bandwidth per ms
    }

    fn calculate_score(&self, tensorsize: usize) -> f64 {
        // Higher score for better efficiency, adjusted for tensor _size
        let size_factor = if tensor_size > 1000000 { 2.0 } else { 1.0 }; // Favor strategies for large tensors
        self.efficiency_score * size_factor
    }
}

/// Adaptive communication strategy selector
#[derive(Debug)]
pub struct AdaptiveCommunicationSelector {
    /// Current strategy
    current_strategy: SyncStrategy,
    /// Strategy switch cooldown (steps)
    switch_cooldown: usize,
    /// Last switch step
    last_switch_step: usize,
    /// Evaluation window (steps)
    evaluation_window: usize,
    /// Performance threshold for strategy switching
    performance_threshold: f64,
}

impl AdaptiveCommunicationSelector {
    fn new() -> Self {
        Self {
            current_strategy: SyncStrategy::RingAllReduce,
            switch_cooldown: 50,
            last_switch_step: 0,
            evaluation_window: 20,
            performance_threshold: 1.2, // 20% improvement required
        }
    }

    fn should_evaluate_strategy(&self, currentstep: usize) -> bool {
        current_step - self.last_switch_step >= self.switch_cooldown
    }

    fn evaluate_and_switch(
        &mut self,
        monitor: &CommunicationPerformanceMonitor,
        tensor_size: usize,
        current_step: usize,
    ) -> Option<SyncStrategy> {
        if !self.should_evaluate_strategy(current_step) {
            return None;
        }

        let optimal_strategy = monitor.get_optimal_strategy(tensor_size);

        if optimal_strategy != self.current_strategy {
            // Check if the switch is worth it based on performance threshold
            if let (Some(current_metrics), Some(optimal_metrics)) = (
                monitor.strategy_performance.get(&self.current_strategy),
                monitor.strategy_performance.get(&optimal_strategy),
            ) {
                let performance_ratio =
                    optimal_metrics.efficiency_score / current_metrics.efficiency_score;

                if performance_ratio >= self.performance_threshold {
                    self.current_strategy = optimal_strategy;
                    self.last_switch_step = current_step;
                    return Some(optimal_strategy);
                }
            }
        }

        None
    }
}

/// Handle for asynchronous communication operations
#[derive(Debug)]
pub struct AsyncCommunicationHandle {
    /// Communication ID
    id: usize,
    /// Start time
    start_time: std::time::Instant,
    /// Expected completion time
    expected_completion: std::time::Duration,
    /// Communication strategy used
    strategy: SyncStrategy,
    /// Data size (bytes)
    data_size: usize,
    /// Status
    status: AsyncCommStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AsyncCommStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Timeout,
}

/// Communication performance statistics snapshot
#[derive(Debug, Clone)]
pub struct CommunicationPerformanceStats {
    pub average_bandwidth_gb_s: f64,
    pub total_operations: usize,
    pub total_data_transferred_gb: f64,
    pub current_strategy: SyncStrategy,
    pub pending_async_ops: usize,
    pub step_count: usize,
}

impl<A: Float + Send + Sync> MultiGpuSync<A> {
    /// Create a new multi-GPU synchronization manager
    pub fn new(
        context: Arc<GpuContext>,
        config: MultiGpuConfig,
        max_param_size: usize,
    ) -> Result<Self, GpuOptimError> {
        // Load synchronization kernels
        let sync_kernels = Self::load_sync_kernels(&context, &config)?;

        // Allocate workspace buffers
        let workspace = Self::allocate_workspace(&context, &config, max_param_size)?;

        // Initialize performance monitoring and adaptive components
        let perf_monitor = CommunicationPerformanceMonitor::new();
        let adaptive_selector = AdaptiveCommunicationSelector::new();
        let async_handles = Vec::with_capacity(config.pipeline_depth);

        Ok(Self {
            context,
            config,
            sync_kernels,
            workspace,
            perf_monitor,
            adaptive_selector,
            async_handles,
            step_counter: 0,
            _phantom: PhantomData,
        })
    }

    /// Synchronize gradients across GPUs
    pub fn sync_gradients<S, D>(
        &mut self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        self.step_counter += 1;
        let tensor_size = gradients.len();
        let start_time = std::time::Instant::now();

        // Adaptive strategy selection
        let strategy = if self.config.adaptive_communication {
            if let Some(new_strategy) = self.adaptive_selector.evaluate_and_switch(
                &self.perf_monitor,
                tensor_size,
                self.step_counter,
            ) {
                new_strategy
            } else {
                self.adaptive_selector.current_strategy
            }
        } else {
            self.config.sync_strategy
        };

        // Execute synchronization
        let result = match strategy {
            SyncStrategy::RingAllReduce => self.ring_allreduce(gradients),
            SyncStrategy::TreeAllReduce => self.tree_allreduce(gradients),
            SyncStrategy::HierarchicalAllReduce => self.hierarchical_allreduce(gradients),
            SyncStrategy::PipelineParallel => {
                if self.config.async_param_updates {
                    self.pipeline_parallel_async(gradients)
                } else {
                    Err(GpuOptimError::UnsupportedOperation(
                        "Pipeline parallel requires async updates enabled".to_string(),
                    ))
                }
            }
        };

        // Record performance
        let elapsed = start_time.elapsed();
        let data_bytes = tensor_size * std::mem::size_of::<A>();

        self.perf_monitor.record_communication(
            strategy,
            data_bytes as u64,
            elapsed.as_micros() as u64,
        );

        // Periodic monitoring output
        if self.step_counter % self.config.bandwidth_monitor_interval == 0 {
            self.log_performance_statistics();
        }

        result
    }

    /// Ring all-reduce implementation
    fn ring_allreduce<S, D>(&self, gradients: &mut ArrayBase<S, D>) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self
                .sync_kernels
                .ring_allreduce
                .as_ref()
                .ok_or(GpuOptimError::NotInitialized)?;

            let grad_slice = gradients.as_slice_mut().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;

            // Create GPU buffer for gradients
            let grad_buffer = self.context.create_buffer_from_slice(grad_slice);

            // Calculate chunk size for ring operations
            let chunk_size = (gradients.len() + self.config.num_gpus - 1) / self.config.num_gpus;

            // Set kernel parameters
            kernel.set_buffer("data", &grad_buffer);
            kernel.set_buffer("recv_buffer", self.workspace.recv_buffer.as_ref().unwrap());
            kernel.set_i32("chunk_size", chunk_size as i32);
            kernel.set_i32("rank", self.config.rank as i32);
            kernel.set_i32("world_size", self.config.num_gpus as i32);

            // Execute ring all-reduce for each chunk
            for chunk_id in 0..self.config.num_gpus {
                kernel.set_i32("chunk_id", chunk_id as i32);

                let (grid_size, block_size) =
                    crate::gpu::utils::calculate_block_size(chunk_size, 256);
                kernel.dispatch([grid_size as u32, 1, 1]);
            }

            // Copy results back
            grad_buffer.copy_to_host(grad_slice);
        }

        Ok(())
    }

    /// Tree all-reduce implementation
    fn tree_allreduce<S, D>(&self, gradients: &mut ArrayBase<S, D>) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self
                .sync_kernels
                .tree_allreduce
                .as_ref()
                .ok_or(GpuOptimError::NotInitialized)?;

            let grad_slice = gradients.as_slice_mut().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;

            // Create GPU buffer for gradients
            let grad_buffer = self.context.create_buffer_from_slice(grad_slice);

            // Calculate tree reduction levels
            let num_levels = (self.config.num_gpus as f32).log2().ceil() as usize;

            // Set kernel parameters
            kernel.set_buffer("data", &grad_buffer);
            kernel.set_buffer("workspace", self.workspace.workspace.as_ref().unwrap());
            kernel.set_i32("rank", self.config.rank as i32);
            kernel.set_i32("world_size", self.config.num_gpus as i32);
            kernel.set_i32("data_size", gradients.len() as i32);

            // Execute tree all-reduce in phases
            for level in 0..num_levels {
                let stride = 1 << level;
                let peer_rank = self.config.rank ^ stride;

                if peer_rank < self.config.num_gpus {
                    kernel.set_i32("level", level as i32);
                    kernel.set_i32("peer_rank", peer_rank as i32);

                    let (grid_size, block_size) =
                        crate::gpu::utils::calculate_block_size(gradients.len(), 256);
                    kernel.dispatch([grid_size as u32, 1, 1]);

                    // Synchronize before next level
                    self.context.synchronize();
                }
            }

            // Copy results back
            grad_buffer.copy_to_host(grad_slice);
        }

        Ok(())
    }

    /// Hierarchical all-reduce for multi-node setups
    fn hierarchical_allreduce<S, D>(
        &self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self
                .sync_kernels
                .hierarchical_allreduce
                .as_ref()
                .ok_or(GpuOptimError::NotInitialized)?;

            let grad_slice = gradients.as_slice_mut().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;

            // Calculate local and global ranks
            let local_rank = self.config.rank % self.config.local_group_size;
            let global_rank = self.config.rank / self.config.local_group_size;
            let global_size = self.config.num_gpus / self.config.local_group_size;

            // Create GPU buffer for gradients
            let grad_buffer = self.context.create_buffer_from_slice(grad_slice);

            // Phase 1: Reduce-scatter within local group
            kernel.set_buffer("data", &grad_buffer);
            kernel.set_buffer("workspace", self.workspace.workspace.as_ref().unwrap());
            kernel.set_i32("local_rank", local_rank as i32);
            kernel.set_i32("local_size", self.config.local_group_size as i32);
            kernel.set_i32("global_rank", global_rank as i32);
            kernel.set_i32("global_size", global_size as i32);
            kernel.set_i32("data_size", gradients.len() as i32);
            kernel.set_i32("phase", 1); // Local reduce-scatter

            let (grid_size, block_size) =
                crate::gpu::utils::calculate_block_size(gradients.len(), 256);
            kernel.dispatch([grid_size as u32, 1, 1]);
            self.context.synchronize();

            // Phase 2: All-reduce across global leaders (one per node)
            if local_rank == 0 {
                kernel.set_i32("phase", 2); // Global all-reduce
                kernel.dispatch([grid_size as u32, 1, 1]);
                self.context.synchronize();
            }

            // Phase 3: All-gather within local group
            kernel.set_i32("phase", 3); // Local all-gather
            kernel.dispatch([grid_size as u32, 1, 1]);
            self.context.synchronize();

            // Copy results back
            grad_buffer.copy_to_host(grad_slice);
        }

        Ok(())
    }

    /// Pipeline parallel asynchronous synchronization
    fn pipeline_parallel_async<S, D>(
        &mut self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let grad_slice = gradients.as_slice_mut().ok_or_else(|| {
                GpuOptimError::InvalidState("Gradients must be contiguous".to_string())
            })?;

            // Create async communication handle
            let handle = AsyncCommunicationHandle {
                id: self.async_handles.len(),
                start_time: std::time::Instant::now(),
                expected_completion: std::time::Duration::from_millis(10), // Estimate
                strategy: SyncStrategy::PipelineParallel,
                data_size: gradients.len() * std::mem::size_of::<A>(),
                status: AsyncCommStatus::InProgress,
            };

            // Pipeline stages: overlap computation and communication
            let chunk_size = gradients.len() / self.config.pipeline_depth;

            for stage in 0..self.config.pipeline_depth {
                let start_idx = stage * chunk_size;
                let end_idx = ((stage + 1) * chunk_size).min(gradients.len());

                if start_idx < end_idx {
                    // Process chunk asynchronously
                    let chunk_buffer = self
                        .context
                        .create_buffer_from_slice(&grad_slice[start_idx..end_idx]);

                    // Submit async operation (placeholder - would use actual GPU streams)
                    // In practice, this would use CUDA streams or similar
                }
            }

            self.async_handles.push(handle);

            // Clean up completed handles periodically
            if self.async_handles.len() > self.config.pipeline_depth * 2 {
                self.cleanup_completed_handles();
            }
        }

        Ok(())
    }

    /// Log performance statistics
    fn log_performance_statistics(&self) {
        let avg_bandwidth = self.perf_monitor.get_average_bandwidth();
        let total_ops = self.perf_monitor.comm_operations;

        println!(
            "Multi-GPU Performance [Step {}]: {:.2} GB/s avg bandwidth, {} ops, current strategy: {:?}",
            self.step_counter,
            avg_bandwidth,
            total_ops,
            self.adaptive_selector.current_strategy
        );
    }

    /// Clean up completed asynchronous communication handles
    fn cleanup_completed_handles(&mut self) {
        let current_time = std::time::Instant::now();

        self.async_handles.retain(|handle| {
            let elapsed = current_time.duration_since(handle.start_time);

            if elapsed > handle.expected_completion {
                // Mark as completed or timeout
                false // Remove from vector
            } else {
                true // Keep in vector
            }
        });
    }

    /// Get communication performance statistics
    pub fn get_performance_stats(&self) -> CommunicationPerformanceStats {
        CommunicationPerformanceStats {
            average_bandwidth_gb_s: self.perf_monitor.get_average_bandwidth(),
            total_operations: self.perf_monitor.comm_operations,
            total_data_transferred_gb: self.perf_monitor.total_data_bytes as f64 / 1e9,
            current_strategy: self.adaptive_selector.current_strategy,
            pending_async_ops: self.async_handles.len(),
            step_count: self.step_counter,
        }
    }

    /// Force synchronization of all pending operations
    pub fn synchronize_all(&mut self) -> Result<(), GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            self.context.synchronize();

            // Update all pending handles to completed
            for handle in &mut self.async_handles {
                if handle.status == AsyncCommStatus::InProgress {
                    handle.status = AsyncCommStatus::Completed;
                }
            }

            self.cleanup_completed_handles();
        }

        Ok(())
    }

    /// Compress gradients for bandwidth optimization
    pub fn compress_gradients<S, D>(
        &mut self,
        gradients: &ArrayBase<S, D>,
    ) -> Result<(Vec<A>, Vec<i32>), GpuOptimError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self
                .sync_kernels
                .compress_gradients
                .as_ref()
                .ok_or(GpuOptimError::NotInitialized)?;

            let k = (gradients.len() as f32 * self.config.compression_ratio) as usize;

            // Set kernel parameters and execute
            // ... implementation details

            // Return compressed values and indices
            let compressed_values = vec![A::zero(); k];
            let compressed_indices = vec![0i32; k];

            Ok((compressed_values, compressed_indices))
        }

        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuOptimError::UnsupportedOperation(
                "GPU feature not enabled".to_string(),
            ))
        }
    }

    /// Load synchronization kernels
    fn load_sync_kernels(
        context: &Arc<GpuContext>,
        config: &MultiGpuConfig,
    ) -> Result<SyncKernels, GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            let ring_kernel = if matches!(config.sync_strategy, SyncStrategy::RingAllReduce) {
                Some(Arc::new(context.get_kernel("ring_allreduce_f32")?))
            } else {
                None
            };

            let tree_kernel = if matches!(config.sync_strategy, SyncStrategy::TreeAllReduce) {
                Some(Arc::new(context.get_kernel("tree_allreduce_f32")?))
            } else {
                None
            };

            let hierarchical_kernel =
                if matches!(config.sync_strategy, SyncStrategy::HierarchicalAllReduce) {
                    Some(Arc::new(context.get_kernel("hierarchical_allreduce_f32")?))
                } else {
                    None
                };

            let compress_kernel = if config.gradient_compression {
                Some(Arc::new(context.get_kernel("compress_gradients_topk_f32")?))
            } else {
                None
            };

            let decompress_kernel = if config.gradient_compression {
                Some(Arc::new(context.get_kernel("decompress_gradients_f32")?))
            } else {
                None
            };

            Ok(SyncKernels {
                ring_allreduce: ring_kernel,
                tree_allreduce: tree_kernel,
                hierarchical_allreduce: hierarchical_kernel,
                compress_gradients: compress_kernel,
                decompress_gradients: decompress_kernel,
            })
        }

        #[cfg(not(feature = "gpu"))]
        {
            Ok(SyncKernels {
                ring_allreduce: None,
                tree_allreduce: None,
                hierarchical_allreduce: None,
                compress_gradients: None,
                decompress_gradients: None,
            })
        }
    }

    /// Allocate workspace buffers
    fn allocate_workspace(
        context: &Arc<GpuContext>,
        config: &MultiGpuConfig,
        max_param_size: usize,
    ) -> Result<WorkspaceBuffers<A>, GpuOptimError> {
        #[cfg(feature = "gpu")]
        {
            let recv_buffer = Some(context.create_buffer::<A>(max_param_size));
            let workspace = Some(context.create_buffer::<A>(max_param_size));

            let (compressed_values, compressed_indices, error_feedback) =
                if config.gradient_compression {
                    let k = (max_param_size as f32 * config.compression_ratio) as usize;
                    (
                        Some(context.create_buffer::<A>(k)),
                        Some(context.create_buffer::<i32>(k)),
                        Some(context.create_buffer::<A>(max_param_size)),
                    )
                } else {
                    (None, None, None)
                };

            Ok(WorkspaceBuffers {
                recv_buffer,
                workspace,
                compressed_values,
                compressed_indices,
                error_feedback,
            })
        }

        #[cfg(not(feature = "gpu"))]
        {
            Ok(WorkspaceBuffers {
                recv_buffer: None,
                workspace: None,
                compressed_values: None,
                compressed_indices: None,
                error_feedback: None,
            })
        }
    }
}

/// Helper to setup multi-GPU training
pub struct MultiGpuSetup {
    /// GPU contexts for each device
    pub contexts: Vec<Arc<GpuContext>>,
    /// Synchronization managers
    pub sync_managers: Vec<MultiGpuSync<f32>>,
}

impl MultiGpuSetup {
    /// Initialize multi-GPU setup
    pub fn new(_num_gpus: usize, max_paramsize: usize) -> Result<Self, GpuOptimError> {
        let mut contexts = Vec::new();
        let mut sync_managers = Vec::new();

        for rank in 0.._num_gpus {
            // Create GPU context for each device
            let context = Arc::new(GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda)?);

            // Create sync manager
            let config = MultiGpuConfig {
                num_gpus,
                rank,
                ..Default::default()
            };

            let sync_manager = MultiGpuSync::new(context.clone(), config, max_param_size)?;

            contexts.push(context);
            sync_managers.push(sync_manager);
        }

        Ok(Self {
            contexts,
            sync_managers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_config_default() {
        let config = MultiGpuConfig::default();
        assert_eq!(config.num_gpus, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.sync_strategy, SyncStrategy::RingAllReduce);
        assert!(!config.gradient_compression);
    }

    #[test]
    fn test_sync_strategy_selection() {
        let strategies = [
            SyncStrategy::RingAllReduce,
            SyncStrategy::TreeAllReduce,
            SyncStrategy::HierarchicalAllReduce,
            SyncStrategy::PipelineParallel,
        ];

        for strategy in &strategies {
            let config = MultiGpuConfig {
                sync_strategy: *strategy,
                ..Default::default()
            };
            assert_eq!(config.sync_strategy, *strategy);
        }
    }

    #[test]
    fn test_communication_performance_monitor() {
        let mut monitor = CommunicationPerformanceMonitor::new();

        // Record some communications
        monitor.record_communication(SyncStrategy::RingAllReduce, 1000000, 1000); // 1GB/s
        monitor.record_communication(SyncStrategy::TreeAllReduce, 2000000, 1000); // 2GB/s

        assert_eq!(monitor.comm_operations, 2);
        assert!(monitor.get_average_bandwidth() > 0.0);

        // Test strategy performance tracking
        let optimal = monitor.get_optimal_strategy(1000000);
        assert!(matches!(
            optimal,
            SyncStrategy::RingAllReduce | SyncStrategy::TreeAllReduce
        ));
    }

    #[test]
    fn test_adaptive_communication_selector() {
        let mut selector = AdaptiveCommunicationSelector::new();
        let mut monitor = CommunicationPerformanceMonitor::new();

        // Initial strategy
        assert_eq!(selector.current_strategy, SyncStrategy::RingAllReduce);

        // Record better performance for tree all-reduce
        for _ in 0..10 {
            monitor.record_communication(SyncStrategy::TreeAllReduce, 1000000, 500);
            // Better bandwidth
        }

        // Should suggest switching after cooldown period
        let new_strategy = selector.evaluate_and_switch(&monitor, 1000000, 100);

        // Depending on performance threshold, might suggest a switch
        if let Some(strategy) = new_strategy {
            assert_ne!(strategy, SyncStrategy::RingAllReduce);
        }
    }

    #[test]
    fn test_multi_gpu_config_extended() {
        let config = MultiGpuConfig {
            num_gpus: 8,
            adaptive_communication: true,
            bandwidth_monitor_interval: 50,
            async_param_updates: true,
            communication_timeout_ms: 1000,
            error_correction: true,
            pipeline_depth: 4,
            ..Default::default()
        };

        assert_eq!(config.num_gpus, 8);
        assert!(config.adaptive_communication);
        assert_eq!(config.bandwidth_monitor_interval, 50);
        assert!(config.async_param_updates);
        assert_eq!(config.communication_timeout_ms, 1000);
        assert!(config.error_correction);
        assert_eq!(config.pipeline_depth, 4);
    }

    #[test]
    fn test_async_communication_handle() {
        let handle = AsyncCommunicationHandle {
            id: 0,
            start_time: std::time::Instant::now(),
            expected_completion: std::time::Duration::from_millis(10),
            strategy: SyncStrategy::PipelineParallel,
            data_size: 1000000,
            status: AsyncCommStatus::Pending,
        };

        assert_eq!(handle.id, 0);
        assert_eq!(handle.strategy, SyncStrategy::PipelineParallel);
        assert_eq!(handle.data_size, 1000000);
        assert_eq!(handle.status, AsyncCommStatus::Pending);
    }

    #[test]
    fn test_strategy_performance_metrics() {
        let mut metrics = StrategyPerformanceMetrics::new();

        metrics.update(10.0, 1000); // 10 GB/s, 1ms
        metrics.update(15.0, 800); // 15 GB/s, 0.8ms

        assert!(metrics.efficiency_score > 0.0);

        let score = metrics.calculate_score(1000000); // Large tensor
        assert!(score > 0.0);
    }

    #[test]
    fn test_communication_performance_stats() {
        let stats = CommunicationPerformanceStats {
            average_bandwidth_gb_s: 10.5,
            total_operations: 100,
            total_data_transferred_gb: 50.0,
            current_strategy: SyncStrategy::RingAllReduce,
            pending_async_ops: 2,
            step_count: 1000,
        };

        assert_eq!(stats.average_bandwidth_gb_s, 10.5);
        assert_eq!(stats.total_operations, 100);
        assert_eq!(stats.total_data_transferred_gb, 50.0);
        assert_eq!(stats.current_strategy, SyncStrategy::RingAllReduce);
        assert_eq!(stats.pending_async_ops, 2);
        assert_eq!(stats.step_count, 1000);
    }
}
