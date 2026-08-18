// Multi-GPU synchronization support for distributed training
//
// # What is real here
//
// [`MultiGpuSync`] drives a real, compiled compute shader
// ([`crate::shaders::CollectiveKernel::AllReduceMean`]) through
// [`scirs2_core::gpu`] on whatever single device the caller's [`GpuContext`]
// opened. That is honestly the extent of it: `scirs2-core` 0.6.x exposes one
// device per context and no cross-device transport (no NCCL/MPI-equivalent),
// so there is no way for this crate to fetch another physical GPU's data.
// Every method here is therefore real for `num_gpus == 1` (the only case
// where "all-reduce" is answerable from local data alone — the answer is the
// local data itself) and an honest [`GpuOptimError::UnsupportedOperation`]
// for `num_gpus > 1`, rather than a kernel dispatch to a name nothing
// registers, or a bare `Ok(())` that quietly did nothing.

use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuDataType, GpuKernelHandle};
use scirs2_core::ndarray::{ArrayBase, Data, DataMut, Dimension};
use scirs2_core::numeric::Float;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::shaders::{CollectiveKernel, WORKGROUP_SIZE};
use crate::GpuOptimError;

/// Multi-GPU synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl MultiGpuConfig {
    /// Check every field that is later used as a divisor or an index bound,
    /// so a bad config fails here with a clear message instead of panicking
    /// (division by zero) deep inside a sync call.
    pub fn validate(&self) -> Result<(), GpuOptimError> {
        let invalid =
            |what: &str| GpuOptimError::InvalidState(format!("invalid multi-GPU config: {what}"));
        if self.num_gpus == 0 {
            return Err(invalid("num_gpus must be >= 1"));
        }
        if self.rank >= self.num_gpus {
            return Err(invalid("rank must be < num_gpus"));
        }
        if self.local_group_size == 0 {
            return Err(invalid("local_group_size must be >= 1"));
        }
        if self.pipeline_depth == 0 {
            return Err(invalid("pipeline_depth must be >= 1"));
        }
        if self.gradient_compression
            && !(self.compression_ratio.is_finite()
                && self.compression_ratio > 0.0
                && self.compression_ratio <= 1.0)
        {
            return Err(invalid("compression_ratio must be finite and in (0, 1]"));
        }
        Ok(())
    }
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
}

impl CommunicationPerformanceMonitor {
    fn new() -> Self {
        Self {
            total_comm_time_us: 0,
            total_data_bytes: 0,
            comm_operations: 0,
            bandwidth_history: std::collections::VecDeque::with_capacity(1000),
            strategy_performance: std::collections::HashMap::new(),
        }
    }

    fn record_communication(
        &mut self,
        strategy: SyncStrategy,
        data_bytes: u64,
        timeus: u64,
        tensor_size: usize,
    ) {
        // A sub-microsecond elapsed time is real (small local ops legitimately
        // take under 1us), but dividing by it is not: clamp to 1us so the
        // bandwidth estimate is merely optimistic instead of `inf`/`NaN`.
        let timeus = timeus.max(1);
        self.total_comm_time_us += timeus;
        self.total_data_bytes += data_bytes;
        self.comm_operations += 1;

        let bandwidth_gb_s = (data_bytes as f64) / (timeus as f64 / 1_000_000.0) / 1e9;
        self.bandwidth_history.push_back(bandwidth_gb_s);

        if self.bandwidth_history.len() > 1000 {
            self.bandwidth_history.pop_front();
        }

        // Update strategy performance
        let metrics = self
            .strategy_performance
            .entry(strategy)
            .or_insert_with(StrategyPerformanceMetrics::new);
        metrics.update(bandwidth_gb_s, timeus, tensor_size);
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
            let score = metrics.calculate_score(tensorsize);
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

    fn update(&mut self, bandwidth_gb_s: f64, latencyus: u64, tensor_size: usize) {
        self.bandwidth_samples.push_back(bandwidth_gb_s);
        self.latency_samples.push_back(latencyus);
        self.tensor_sizes.push_back(tensor_size);

        if self.bandwidth_samples.len() > 100 {
            self.bandwidth_samples.pop_front();
            self.latency_samples.pop_front();
            self.tensor_sizes.pop_front();
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
        let size_factor = if tensorsize > 1000000 { 2.0 } else { 1.0 }; // Favor strategies for large tensors

        // Trust this strategy's efficiency score less when it has no track
        // record at a comparable tensor size (within 10x): bandwidth and
        // latency measured on very differently-sized transfers may not
        // generalize to this one. A strategy with no history at all is not
        // penalized further here -- its `efficiency_score` already starts
        // at 0.0 until `update` has run at least once.
        let has_comparable_history = self.tensor_sizes.is_empty()
            || self.tensor_sizes.iter().any(|&recorded| {
                let (small, large) = if recorded <= tensorsize {
                    (recorded.max(1), tensorsize.max(1))
                } else {
                    (tensorsize.max(1), recorded)
                };
                large <= small * 10
            });
        let relevance = if has_comparable_history { 1.0 } else { 0.5 };

        self.efficiency_score * size_factor * relevance
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
    /// Evaluation window (steps).
    ///
    /// Not currently consulted by [`Self::should_evaluate_strategy`] or
    /// [`Self::evaluate_and_switch`]: `switch_cooldown` (steps since the
    /// last switch) is the only gate implemented today. Whether
    /// `evaluation_window` should instead gate how often a potential
    /// switch is *checked for* (independent of `switch_cooldown`, which
    /// gates when a switch may actually happen), or how much recent
    /// history `evaluate_and_switch` compares (as opposed to each
    /// strategy's full rolling `efficiency_score`), is a scheduling-policy
    /// decision this lint pass is not making unilaterally -- especially
    /// since the two fields' default values (50 and 20) are not a clean
    /// multiple of each other, so guessing the intended relationship risks
    /// getting it wrong. Recorded as a finding rather than force-wired.
    #[allow(dead_code)]
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
        currentstep - self.last_switch_step >= self.switch_cooldown
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

/// Communication performance statistics snapshot
#[derive(Debug, Clone)]
pub struct CommunicationPerformanceStats {
    pub average_bandwidth_gb_s: f64,
    pub total_operations: usize,
    pub total_data_transferred_gb: f64,
    pub current_strategy: SyncStrategy,
    /// Always `0`: this build has no asynchronous collective machinery (see
    /// the module docs). The field is kept so callers that already match on
    /// this struct do not need to change; it is not a rounded-down real count.
    pub pending_async_ops: usize,
    pub step_count: usize,
}

/// Encode a `usize` element count into an `f32` slot bit-for-bit, recovered on
/// the device with `bitcast<u32>` / `as_type<uint>`.
fn encode_u32(value: usize) -> Result<f32, GpuOptimError> {
    let raw = u32::try_from(value).map_err(|_| {
        GpuOptimError::UnsupportedOperation(format!("{value} does not fit in a u32 kernel operand"))
    })?;
    Ok(f32::from_bits(raw))
}

/// Number of `WORKGROUP_SIZE`-wide workgroups needed to cover `n` elements.
fn workgroup_count(n: usize) -> Result<u32, GpuOptimError> {
    let groups = n.div_ceil(WORKGROUP_SIZE);
    u32::try_from(groups).map_err(|_| {
        GpuOptimError::UnsupportedOperation(format!(
            "{n} elements need {groups} workgroups, which exceeds the u32 dispatch limit"
        ))
    })
}

/// Dispatch the real all-reduce-mean kernel over `data[range]`, in place.
///
/// Works for any `A: Float`, not just `f32`: the host round-trips through an
/// `f32` buffer (the shaders are `f32`-only, matching every other kernel this
/// crate ships) via the same numeric conversion used throughout this crate
/// rather than a byte reinterpret, so it is correct for `A = f64` too, just
/// rounded through `f32` precision.
fn dispatch_local_reduce(
    context: &GpuContext,
    kernel: &GpuKernelHandle,
    host: &[f32],
    num_gpus: usize,
) -> Result<Vec<f32>, GpuOptimError> {
    let n = host.len();
    let hyper = [encode_u32(n)?, encode_u32(num_gpus)?];
    let groups = workgroup_count(n)?;

    let x_buf = context.create_buffer::<f32>(n);
    x_buf.copy_from_host(host)?;
    let y_buf = context.create_buffer::<f32>(hyper.len());
    y_buf.copy_from_host(&hyper)?;

    kernel.set_buffer("x", &x_buf);
    kernel.set_buffer("y", &y_buf);
    kernel.dispatch([groups, 1, 1]);

    let mut out = vec![0.0f32; n];
    x_buf.copy_to_host(&mut out)?;
    Ok(out)
}

/// Multi-GPU synchronization manager
pub struct MultiGpuSync<A: Float + GpuDataType> {
    /// GPU context
    context: Arc<GpuContext>,
    /// Configuration
    config: MultiGpuConfig,
    /// Upper bound on the number of elements a single sync call will accept.
    max_param_size: usize,
    /// The real local-reduction kernel, compiled once for the context's
    /// backend. `None` when that backend has no shader source for it (every
    /// backend this crate ships kernels for is `Wgpu`/`Metal`); every method
    /// that would need it then returns an honest `UnsupportedOperation`
    /// instead of dereferencing a handle that was never created.
    reduce_kernel: Option<GpuKernelHandle>,
    /// Communication performance monitor
    perf_monitor: CommunicationPerformanceMonitor,
    /// Adaptive strategy selector
    adaptive_selector: AdaptiveCommunicationSelector,
    /// Step counter for monitoring
    step_counter: usize,
    /// Phantom data for type parameter
    _phantom: PhantomData<A>,
}

impl<A: Float + GpuDataType + Send + Sync> MultiGpuSync<A> {
    /// Create a new multi-GPU synchronization manager.
    ///
    /// `max_param_size` bounds how large a single tensor `sync_gradients` (and
    /// friends) will accept; it is a resource cap the caller opts into, not a
    /// buffer that gets preallocated.
    pub fn new(
        context: Arc<GpuContext>,
        config: MultiGpuConfig,
        max_param_size: usize,
    ) -> Result<Self, GpuOptimError> {
        config.validate()?;

        let reduce_kernel = match CollectiveKernel::AllReduceMean.source_for(context.backend()) {
            Some(source) => Some(context.execute(|compiler| compiler.compile(source))?),
            None => None,
        };

        Ok(Self {
            context,
            config,
            max_param_size,
            reduce_kernel,
            perf_monitor: CommunicationPerformanceMonitor::new(),
            adaptive_selector: AdaptiveCommunicationSelector::new(),
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

        // Execute synchronization. Without a cross-device transport every
        // topology (ring / tree / hierarchical) answers the single-device
        // case identically, and every topology is equally unable to serve
        // `num_gpus > 1` — see the module docs.
        let result = match strategy {
            SyncStrategy::RingAllReduce
            | SyncStrategy::TreeAllReduce
            | SyncStrategy::HierarchicalAllReduce => self.local_reduce(gradients),
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
            tensor_size,
        );

        // Periodic monitoring output
        if self
            .step_counter
            .is_multiple_of(self.config.bandwidth_monitor_interval)
        {
            self.log_performance_statistics();
        }

        result
    }

    /// The one real operation every collective strategy reduces to on a
    /// single device: divide the local buffer by the replica count. For
    /// `num_gpus == 1` this is the exact all-reduce-mean answer — there is
    /// nothing else to sum — computed for real via a compiled compute
    /// shader. For `num_gpus > 1` this honestly refuses: there is no
    /// transport in this build to fetch the other replicas' data.
    fn local_reduce<S, D>(&self, gradients: &mut ArrayBase<S, D>) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        if self.config.num_gpus > 1 {
            return Err(GpuOptimError::UnsupportedOperation(format!(
                "all-reduce across {} GPUs needs a cross-device transport (an NCCL/MPI \
                 equivalent); this build has a single scirs2_core::gpu::GpuContext and no such \
                 transport, so peer devices' data can never be fetched",
                self.config.num_gpus
            )));
        }
        let n = gradients.len();
        if n == 0 {
            return Ok(());
        }
        if n > self.max_param_size {
            return Err(GpuOptimError::InvalidState(format!(
                "gradient tensor has {n} elements, above the {}-element bound this \
                 MultiGpuSync was constructed with",
                self.max_param_size
            )));
        }
        let kernel = self.reduce_kernel.as_ref().ok_or_else(|| {
            GpuOptimError::UnsupportedOperation(format!(
                "no all-reduce kernel source for backend {}",
                self.context.backend()
            ))
        })?;

        let host: Vec<f32> = gradients
            .iter()
            .map(|v| v.to_f32().unwrap_or(0.0))
            .collect();
        let out = dispatch_local_reduce(&self.context, kernel, &host, 1)?;
        write_back(gradients, &out)
    }

    /// Pipeline-parallel synchronization.
    ///
    /// Splits the tensor into [`MultiGpuConfig::pipeline_depth`] chunks and
    /// submits one real dispatch per chunk with
    /// [`GpuKernelHandle::dispatch_no_wait`], then waits for the whole batch
    /// with one [`GpuContext::gpu_sync`]. That is genuine command-queue
    /// overlap — what "pipelining" means at the hardware level — and it does
    /// not require a second physical device to be real. `num_gpus > 1` is
    /// still an honest error for the same reason as [`Self::local_reduce`].
    fn pipeline_parallel_async<S, D>(
        &mut self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        if self.config.num_gpus > 1 {
            return Err(GpuOptimError::UnsupportedOperation(format!(
                "pipeline-parallel sync across {} GPUs needs a cross-device transport this \
                 build does not have",
                self.config.num_gpus
            )));
        }
        let n = gradients.len();
        if n == 0 {
            return Ok(());
        }
        if n > self.max_param_size {
            return Err(GpuOptimError::InvalidState(format!(
                "gradient tensor has {n} elements, above the {}-element bound this \
                 MultiGpuSync was constructed with",
                self.max_param_size
            )));
        }
        let kernel = self.reduce_kernel.as_ref().ok_or_else(|| {
            GpuOptimError::UnsupportedOperation(format!(
                "no all-reduce kernel source for backend {}",
                self.context.backend()
            ))
        })?;

        let host: Vec<f32> = gradients
            .iter()
            .map(|v| v.to_f32().unwrap_or(0.0))
            .collect();
        let depth = self.config.pipeline_depth.max(1);
        // `div_ceil` so the tail is never dropped: the last chunk absorbs
        // whatever remainder `n` does not divide evenly by `depth`.
        let chunk_size = n.div_ceil(depth).max(1);

        let mut chunks: Vec<(usize, usize, GpuBuffer<f32>)> = Vec::with_capacity(depth);
        for stage in 0..depth {
            let start = stage * chunk_size;
            if start >= n {
                break;
            }
            let end = (start + chunk_size).min(n);
            let hyper = [encode_u32(end - start)?, encode_u32(1)?];

            let x_buf = self.context.create_buffer::<f32>(end - start);
            x_buf.copy_from_host(&host[start..end])?;
            let y_buf = self.context.create_buffer::<f32>(hyper.len());
            y_buf.copy_from_host(&hyper)?;

            kernel.set_buffer("x", &x_buf);
            kernel.set_buffer("y", &y_buf);
            kernel.dispatch_no_wait([workgroup_count(end - start)?, 1, 1]);
            chunks.push((start, end, x_buf));
        }

        // One fence for the whole batch: Metal command queues are FIFO, so
        // waiting on a buffer submitted after every chunk's guarantees every
        // chunk has completed (see `GpuContext::gpu_sync` docs).
        self.context.gpu_sync()?;

        let mut out = vec![0.0f32; n];
        for (start, end, buf) in &chunks {
            buf.copy_to_host(&mut out[*start..*end])?;
        }

        write_back(gradients, &out)
    }

    /// Log performance statistics
    fn log_performance_statistics(&self) {
        let avg_bandwidth = self.perf_monitor.get_average_bandwidth();
        let total_ops = self.perf_monitor.comm_operations;

        log::info!(
            "Multi-GPU Performance [Step {}]: {:.2} GB/s avg bandwidth, {} ops, current strategy: {:?}",
            self.step_counter,
            avg_bandwidth,
            total_ops,
            self.adaptive_selector.current_strategy
        );
    }

    /// Get communication performance statistics
    pub fn get_performance_stats(&self) -> CommunicationPerformanceStats {
        CommunicationPerformanceStats {
            average_bandwidth_gb_s: self.perf_monitor.get_average_bandwidth(),
            total_operations: self.perf_monitor.comm_operations,
            total_data_transferred_gb: self.perf_monitor.total_data_bytes as f64 / 1e9,
            current_strategy: self.adaptive_selector.current_strategy,
            pending_async_ops: 0,
            step_count: self.step_counter,
        }
    }

    /// Wait for every dispatch issued so far to complete.
    pub fn synchronize_all(&mut self) -> Result<(), GpuOptimError> {
        self.context.gpu_sync().map_err(GpuOptimError::from)
    }

    /// Compress gradients for bandwidth optimization with real top-*k*
    /// (largest-magnitude) selection.
    ///
    /// This is host-side selection, not a GPU kernel: choosing the *k* largest
    /// magnitudes is a sort/partition, not a per-element map, and gains
    /// nothing from a compute shader at the sizes this crate targets. The
    /// returned `indices` are into the flattened (`.iter()`-order) tensor.
    pub fn compress_gradients<S, D>(
        &mut self,
        gradients: &ArrayBase<S, D>,
    ) -> Result<(Vec<A>, Vec<i32>), GpuOptimError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        let len = gradients.len();
        if len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        // `k = 0` would silently compress every tensor to nothing; a ratio in
        // (0, 1] (enforced by `MultiGpuConfig::validate`) always keeps at
        // least the single largest element.
        let k = (((len as f64) * (self.config.compression_ratio as f64)).round() as usize)
            .clamp(1, len);

        let mut indexed: Vec<(usize, A)> = gradients.iter().copied().enumerate().collect();
        // `Float` gives no `Ord`/`total_cmp`; NaNs sort as equal instead of
        // panicking the comparator.
        indexed.sort_by(|(_, a), (_, b)| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);

        let mut values = Vec::with_capacity(k);
        let mut indices = Vec::with_capacity(k);
        for (idx, value) in indexed {
            values.push(value);
            indices.push(idx as i32);
        }
        Ok((values, indices))
    }
}

/// Write a flat `f32` slice back into an array of any layout, converting each
/// element back to `A` through the same numeric path [`local_reduce`] read it
/// with (never a byte reinterpret).
fn write_back<A, S, D>(array: &mut ArrayBase<S, D>, values: &[f32]) -> Result<(), GpuOptimError>
where
    A: Float,
    S: DataMut<Elem = A>,
    D: Dimension,
{
    for (dst, &src) in array.iter_mut().zip(values.iter()) {
        *dst = A::from(src).ok_or_else(|| {
            GpuOptimError::InvalidState(format!(
                "{src} is not representable in the target float type"
            ))
        })?;
    }
    Ok(())
}

/// Helper to setup multi-GPU training
pub struct MultiGpuSetup {
    /// GPU contexts for each device
    pub contexts: Vec<Arc<GpuContext>>,
    /// Synchronization managers
    pub sync_managers: Vec<MultiGpuSync<f32>>,
}

impl MultiGpuSetup {
    /// Initialize multi-GPU setup.
    ///
    /// Every logical rank shares the *same* physical device: `scirs2-core`
    /// 0.6.x has no API to enumerate or address more than one, so there is
    /// nothing else this constructor could honestly open. Opens the context
    /// via [`crate::optimizers::SUPPORTED_BACKENDS`] (the backends this
    /// crate actually ships kernels for), never the removed `Cuda` backend
    /// that always errors.
    pub fn new(num_gpus: usize, max_param_size: usize) -> Result<Self, GpuOptimError> {
        let mut reasons = Vec::new();
        let mut opened = None;
        for backend in crate::optimizers::SUPPORTED_BACKENDS {
            match GpuContext::new(backend) {
                Ok(context) => {
                    opened = Some(context);
                    break;
                }
                Err(e) => reasons.push(format!("{backend}: {e}")),
            }
        }
        let Some(shared_context) = opened else {
            return Err(GpuOptimError::UnsupportedOperation(format!(
                "no GPU backend available for multi-GPU setup ({})",
                reasons.join("; ")
            )));
        };

        let mut contexts = Vec::with_capacity(num_gpus);
        let mut sync_managers = Vec::with_capacity(num_gpus);
        let context = Arc::new(shared_context);

        for rank in 0..num_gpus {
            let config = MultiGpuConfig {
                num_gpus,
                rank,
                ..Default::default()
            };

            let sync_manager = MultiGpuSync::new(context.clone(), config, max_param_size)?;

            contexts.push(context.clone());
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
    use crate::optimizers::SUPPORTED_BACKENDS;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_multi_gpu_config_default() {
        let config = MultiGpuConfig::default();
        assert_eq!(config.num_gpus, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.sync_strategy, SyncStrategy::RingAllReduce);
        assert!(!config.gradient_compression);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_rejects_divide_by_zero_fields() {
        let base = MultiGpuConfig::default();
        assert!(MultiGpuConfig {
            num_gpus: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(MultiGpuConfig {
            local_group_size: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(MultiGpuConfig {
            pipeline_depth: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(MultiGpuConfig {
            rank: 5,
            num_gpus: 2,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(MultiGpuConfig {
            gradient_compression: true,
            compression_ratio: 0.0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(MultiGpuConfig {
            gradient_compression: true,
            compression_ratio: f32::NAN,
            ..base
        }
        .validate()
        .is_err());
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
        monitor.record_communication(SyncStrategy::RingAllReduce, 1000000, 1000, 1000000); // 1GB/s
        monitor.record_communication(SyncStrategy::TreeAllReduce, 2000000, 1000, 1000000); // 2GB/s

        assert_eq!(monitor.comm_operations, 2);
        assert!(monitor.get_average_bandwidth() > 0.0);

        // Test strategy performance tracking
        let optimal = monitor.get_optimal_strategy(1000000);
        assert!(matches!(
            optimal,
            SyncStrategy::RingAllReduce | SyncStrategy::TreeAllReduce
        ));
    }

    /// A zero-microsecond sample must not poison the running bandwidth with
    /// `inf`/`NaN` (regression test for F17).
    #[test]
    fn record_communication_clamps_zero_elapsed_time() {
        let mut monitor = CommunicationPerformanceMonitor::new();
        monitor.record_communication(SyncStrategy::RingAllReduce, 1_000_000, 0, 1_000_000);
        let avg = monitor.get_average_bandwidth();
        assert!(avg.is_finite(), "average bandwidth was not finite: {avg}");
        assert!(avg > 0.0);
        assert!(monitor
            .bandwidth_history
            .back()
            .copied()
            .unwrap_or(f64::NAN)
            .is_finite());
    }

    #[test]
    fn test_adaptive_communication_selector() {
        let mut selector = AdaptiveCommunicationSelector::new();
        let mut monitor = CommunicationPerformanceMonitor::new();

        // Initial strategy
        assert_eq!(selector.current_strategy, SyncStrategy::RingAllReduce);

        // Record better performance for tree all-reduce
        for _ in 0..10 {
            monitor.record_communication(SyncStrategy::TreeAllReduce, 1000000, 500, 1000000);
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
    fn test_strategy_performance_metrics() {
        let mut metrics = StrategyPerformanceMetrics::new();

        metrics.update(10.0, 1000, 1000000); // 10 GB/s, 1ms
        metrics.update(15.0, 800, 1000000); // 15 GB/s, 0.8ms

        assert!(metrics.efficiency_score > 0.0);

        let score = metrics.calculate_score(1000000); // Large tensor
        assert!(score > 0.0);
    }

    /// `calculate_score` must genuinely use recorded tensor sizes (not just
    /// accept and discard them): a strategy whose entire track record is at
    /// one scale should be trusted less when scored against a tensor size
    /// three orders of magnitude away, versus a size close to what it has
    /// actually proven itself on.
    #[test]
    fn test_calculate_score_discounts_unfamiliar_tensor_sizes() {
        let mut metrics = StrategyPerformanceMetrics::new();
        metrics.update(10.0, 1000, 1_000_000);
        metrics.update(10.0, 1000, 1_000_000);

        let familiar = metrics.calculate_score(1_000_000);
        let unfamiliar = metrics.calculate_score(1_000);

        assert!(
            unfamiliar < familiar,
            "score for an unfamiliar tensor size ({unfamiliar}) should be lower than for a \
             size this strategy has a track record at ({familiar})"
        );
    }

    #[test]
    fn test_communication_performance_stats() {
        let stats = CommunicationPerformanceStats {
            average_bandwidth_gb_s: 10.5,
            total_operations: 100,
            total_data_transferred_gb: 50.0,
            current_strategy: SyncStrategy::RingAllReduce,
            pending_async_ops: 0,
            step_count: 1000,
        };

        assert_eq!(stats.average_bandwidth_gb_s, 10.5);
        assert_eq!(stats.total_operations, 100);
        assert_eq!(stats.total_data_transferred_gb, 50.0);
        assert_eq!(stats.current_strategy, SyncStrategy::RingAllReduce);
        assert_eq!(stats.pending_async_ops, 0);
        assert_eq!(stats.step_count, 1000);
    }

    /// Real top-*k* selection: the returned values must be exactly the *k*
    /// largest-magnitude elements (regression test for F12 — this used to
    /// unconditionally return zeros).
    #[test]
    fn compress_gradients_selects_real_top_k() {
        let context = match probe_backend() {
            Some(backend) => Arc::new(GpuContext::new(backend).expect("backend just probed")),
            None => {
                eprintln!("SKIP: compress_gradients_selects_real_top_k — no usable GPU backend");
                return;
            }
        };
        let config = MultiGpuConfig {
            gradient_compression: true,
            compression_ratio: 0.25,
            ..Default::default()
        };
        let mut sync = MultiGpuSync::<f32>::new(context, config, 1024).expect("construction");

        let data = Array1::from(vec![0.1f32, -5.0, 2.0, 0.3, -4.0, 1.0, 0.05, -0.2]);
        let (values, indices) = sync.compress_gradients(&data).expect("compression");

        // ratio 0.25 of 8 elements -> k = 2; the two largest magnitudes are
        // -5.0 (index 1) and -4.0 (index 4).
        assert_eq!(values.len(), 2);
        assert_eq!(indices.len(), 2);
        let mut got: Vec<(i32, f32)> = indices.into_iter().zip(values).collect();
        got.sort_by_key(|(idx, _)| *idx);
        assert_eq!(got, vec![(1, -5.0), (4, -4.0)]);
    }

    #[test]
    fn compress_gradients_ratio_never_selects_zero_elements() {
        let context = match probe_backend() {
            Some(backend) => Arc::new(GpuContext::new(backend).expect("backend just probed")),
            None => {
                eprintln!(
                    "SKIP: compress_gradients_ratio_never_selects_zero_elements — no usable GPU backend"
                );
                return;
            }
        };
        let config = MultiGpuConfig {
            gradient_compression: true,
            compression_ratio: 0.01, // rounds to 0 of 4 elements without the clamp
            ..Default::default()
        };
        let mut sync = MultiGpuSync::<f32>::new(context, config, 1024).expect("construction");
        let data = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0]);
        let (values, _) = sync.compress_gradients(&data).expect("compression");
        assert_eq!(
            values.len(),
            1,
            "a nonzero ratio must keep at least one element"
        );
    }

    fn probe_backend() -> Option<scirs2_core::gpu::GpuBackend> {
        SUPPORTED_BACKENDS
            .into_iter()
            .find(|&backend| GpuContext::new(backend).is_ok())
    }

    /// `MultiGpuSync::new` used to unconditionally fail (F2: it asked the
    /// registry for kernel names nothing registers). It must now construct,
    /// and a single-device sync must actually run the kernel and leave the
    /// data numerically unchanged (dividing one replica by one).
    #[test]
    fn single_device_sync_runs_a_real_kernel_and_is_the_identity() {
        let backend = match probe_backend() {
            Some(b) => b,
            None => {
                eprintln!(
                    "SKIP: single_device_sync_runs_a_real_kernel_and_is_the_identity — no usable GPU backend"
                );
                return;
            }
        };
        let context = Arc::new(GpuContext::new(backend).expect("backend just probed"));
        let config = MultiGpuConfig::default(); // num_gpus: 1
        let mut sync = MultiGpuSync::<f32>::new(context, config, 4096).expect("construction");

        for strategy in [
            SyncStrategy::RingAllReduce,
            SyncStrategy::TreeAllReduce,
            SyncStrategy::HierarchicalAllReduce,
        ] {
            sync.config.sync_strategy = strategy;
            let original: Array1<f32> =
                Array1::from((0..777).map(|i| i as f32 * 0.5 - 10.0).collect::<Vec<_>>());
            let mut grads = original.clone();
            sync.sync_gradients(&mut grads).unwrap_or_else(|e| {
                panic!("{strategy:?}: single-device sync must succeed, got {e}")
            });
            for (a, b) in original.iter().zip(grads.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{strategy:?}: single-device all-reduce changed the data: {a} -> {b}"
                );
            }
        }
    }

    /// `num_gpus > 1` must be an explicit, honest error — never a silent
    /// `Ok(())` that did nothing (F15) and never a panic (F2/F18).
    #[test]
    fn multi_device_sync_is_an_honest_unsupported_error() {
        let backend = match probe_backend() {
            Some(b) => b,
            None => {
                eprintln!("SKIP: multi_device_sync_is_an_honest_unsupported_error — no usable GPU backend");
                return;
            }
        };
        let context = Arc::new(GpuContext::new(backend).expect("backend just probed"));
        let config = MultiGpuConfig {
            num_gpus: 2,
            ..Default::default()
        };
        let mut sync = MultiGpuSync::<f32>::new(context, config, 4096).expect("construction");
        let mut grads = Array1::from_elem(16, 1.0f32);
        let err = sync
            .sync_gradients(&mut grads)
            .expect_err("num_gpus > 1 must fail, not silently succeed");
        assert!(matches!(err, GpuOptimError::UnsupportedOperation(_)));
    }

    /// Pipeline-parallel sync with a chunk count that does not evenly divide
    /// the tensor length must not drop the tail (regression test for F13).
    #[test]
    fn pipeline_parallel_covers_every_element_including_the_tail() {
        let backend = match probe_backend() {
            Some(b) => b,
            None => {
                eprintln!(
                    "SKIP: pipeline_parallel_covers_every_element_including_the_tail — no usable GPU backend"
                );
                return;
            }
        };
        let context = Arc::new(GpuContext::new(backend).expect("backend just probed"));
        let config = MultiGpuConfig {
            sync_strategy: SyncStrategy::PipelineParallel,
            async_param_updates: true,
            pipeline_depth: 4,
            adaptive_communication: false,
            ..Default::default()
        };
        let mut sync = MultiGpuSync::<f32>::new(context, config, 4096).expect("construction");

        // 777 does not divide evenly by 4.
        let original: Array1<f32> = Array1::from((0..777).map(|i| i as f32).collect::<Vec<_>>());
        let mut grads = original.clone();
        sync.sync_gradients(&mut grads).expect("pipeline sync");
        for (i, (a, b)) in original.iter().zip(grads.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "element {i} was dropped or corrupted: {a} -> {b}"
            );
        }
    }

    #[test]
    fn synchronize_all_waits_on_a_real_fence() {
        let backend = match probe_backend() {
            Some(b) => b,
            None => {
                eprintln!("SKIP: synchronize_all_waits_on_a_real_fence — no usable GPU backend");
                return;
            }
        };
        let context = Arc::new(GpuContext::new(backend).expect("backend just probed"));
        let mut sync = MultiGpuSync::<f32>::new(context, MultiGpuConfig::default(), 1024)
            .expect("construction");
        assert!(sync.synchronize_all().is_ok());
    }

    #[test]
    fn multi_gpu_setup_opens_a_real_backend_not_the_removed_cuda_one() {
        match MultiGpuSetup::new(2, 1024) {
            Ok(setup) => {
                assert_eq!(setup.contexts.len(), 2);
                assert_eq!(setup.sync_managers.len(), 2);
                for context in &setup.contexts {
                    assert_ne!(
                        context.backend(),
                        scirs2_core::gpu::GpuBackend::Cuda,
                        "must never request the CUDA backend scirs2-core 0.6.x always errors on"
                    );
                }
            }
            Err(e) => {
                // Legitimate on a headless machine with no GPU adapter at all.
                eprintln!(
                    "SKIP: multi_gpu_setup_opens_a_real_backend_not_the_removed_cuda_one — {e}"
                );
            }
        }
    }
}
