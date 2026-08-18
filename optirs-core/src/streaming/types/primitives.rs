//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, ArrayBase};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::functions::to_a_or;
use super::prediction::{PredictionModel, DEFAULT_MODEL_ORDER};
use super::types_2::StreamingDataPoint;
use super::types_3::{AdvancedQoSConfig, RealTimeConfig, SyncBarrier};

/// Timing tracker for performance monitoring
#[derive(Debug)]
pub(super) struct TimingTracker {
    /// Latency samples
    pub(super) latency_samples: VecDeque<Duration>,
    /// Processing start time for current batch
    pub(super) batch_start: Option<Instant>,
    /// Maximum samples to keep
    pub(super) max_samples: usize,
}
/// Resource reservation strategies
#[derive(Debug, Clone, Copy)]
pub enum ResourceReservationStrategy {
    Static,
    Dynamic,
    Adaptive,
    PredictiveBased,
}
/// Coordination strategies for pipeline stages
#[derive(Debug, Clone, Copy)]
pub enum CoordinationStrategy {
    DataParallel,
    TaskParallel,
    PipelineParallel,
    Hybrid,
}
/// Load balancing strategies for multi-stream processing
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    PriorityBased,
    AdaptiveLoadAware,
}
/// Streaming optimization configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for mini-batches
    pub buffer_size: usize,
    /// Maximum latency budget (milliseconds)
    pub latency_budget_ms: u64,
    /// Enable adaptive learning rates
    pub adaptive_learning_rate: bool,
    /// Concept drift detection threshold
    pub drift_threshold: f64,
    /// Window size for drift detection
    pub drift_window_size: usize,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Enable asynchronous updates
    pub async_updates: bool,
    /// Maximum staleness for asynchronous updates
    pub max_staleness: usize,
    /// Enable memory-efficient processing
    pub memory_efficient: bool,
    /// Target memory usage (MB)
    pub memory_budget_mb: usize,
    /// Learning rate adaptation strategy
    pub lr_adaptation: LearningRateAdaptation,
    /// Enable adaptive batching
    pub adaptive_batching: bool,
    /// Dynamic buffer sizing
    pub dynamic_buffer_sizing: bool,
    /// Real-time priority levels
    pub enable_priority_scheduling: bool,
    /// Advanced drift detection
    pub advanced_drift_detection: bool,
    /// Predictive processing
    pub enable_prediction: bool,
    /// Quality of service guarantees
    pub qos_enabled: bool,
    /// Enable multi-stream coordination
    pub multi_stream_coordination: bool,
    /// Enable predictive streaming algorithms
    pub predictive_streaming: bool,
    /// Enable stream fusion optimization
    pub stream_fusion: bool,
    /// Advanced QoS configuration
    pub advanced_qos_config: AdvancedQoSConfig,
    /// Real-time optimization parameters
    pub real_time_config: RealTimeConfig,
    /// Pipeline parallelism degree
    pub pipeline_parallelism_degree: usize,
    /// Enable adaptive resource allocation
    pub adaptive_resource_allocation: bool,
    /// Enable distributed streaming
    pub distributed_streaming: bool,
    /// Stream processing priority
    pub processingpriority: StreamPriority,
}
/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage<A: Float + Send + Sync> {
    pub stage_id: String,
    pub processing_function: String,
    pub input_buffer: VecDeque<StreamingDataPoint<A>>,
    pub output_buffer: VecDeque<StreamingDataPoint<A>>,
    pub stage_metrics: StageMetrics,
}
/// Resource constraints
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory_mb: usize,
    pub max_cpu_cores: usize,
    pub max_latency_ms: u64,
}
/// Learning rate adaptation strategies for streaming
#[derive(Debug, Clone, Copy)]
pub enum LearningRateAdaptation {
    /// Fixed learning rate
    Fixed,
    /// AdaGrad-style adaptation
    Adagrad,
    /// RMSprop-style adaptation
    RMSprop,
    /// Performance-based adaptation
    PerformanceBased,
    /// Concept drift aware adaptation
    DriftAware,
    /// Adaptive momentum-based
    AdaptiveMomentum,
    /// Gradient variance-based
    GradientVariance,
    /// Predictive adaptation
    PredictiveLR,
}
/// Asynchronous update state.
///
/// Previously also carried a `pending_gradients` vector and an
/// `update_thread` join handle, neither of which was ever written to or read:
/// asynchronous updates are queued in `update_queue` and applied inline by
/// `process_async_updates`, with no background thread involved.
#[derive(Debug)]
pub(super) struct AsyncUpdateState<A: Float, D: scirs2_core::ndarray::Dimension> {
    /// Update queue
    pub(super) update_queue: VecDeque<AsyncUpdate<A, D>>,
    /// Staleness counter
    pub(super) staleness_counter: HashMap<usize, usize>,
}
/// Stream processing priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Background,
    Low,
    Normal,
    High,
    Critical,
    RealTime,
}
/// Resource allocation result
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub memory_allocation_mb: usize,
    pub cpu_allocation: usize,
    pub priority_adjustment: i32,
    pub timestamp: Instant,
}
/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total samples processed
    pub samples_processed: usize,
    /// Current processing rate (samples/second)
    pub processing_rate: f64,
    /// Average latency per sample (milliseconds)
    pub avg_latency_ms: f64,
    /// 95th percentile latency (milliseconds)
    pub p95_latency_ms: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Concept drifts detected
    pub drift_count: usize,
    /// Current loss
    pub current_loss: f64,
    /// Learning rate
    pub current_learning_rate: f64,
    /// Throughput violations (exceeded latency budget)
    pub throughput_violations: usize,
}
/// Real-time optimization result
#[derive(Debug, Clone)]
pub struct RTOptimizationResult {
    pub optimization_applied: bool,
    pub performance_gain: f64,
    pub latency_reduction_ms: f64,
}
/// Stage metrics
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    pub processing_time_ms: f64,
    pub throughput_samples_per_sec: f64,
    pub buffer_utilization: f64,
    pub error_count: usize,
}
/// Service level objective
#[derive(Debug, Clone)]
pub struct ServiceLevelObjective {
    pub metric: QoSMetric,
    pub target_value: f64,
    pub tolerance: f64,
}
/// Quality of Service metrics
#[derive(Debug, Clone, Copy)]
pub enum QoSMetric {
    Latency,
    Throughput,
    MemoryUsage,
    CpuUtilization,
    PredictionAccuracy,
    StreamSynchronization,
}
/// Real-time optimization state
#[derive(Debug, Clone, Default)]
pub struct RTOptimizationState {
    pub current_priority: i32,
    pub cpu_affinity_mask: u64,
    pub memory_pools: Vec<usize>,
    pub optimization_level: u8,
}
/// Fused optimization step
#[derive(Debug, Clone)]
pub struct FusedOptimizationStep<A: Float + Send + Sync> {
    pub step: Array1<A>,
    pub confidence: A,
    pub contributing_streams: Vec<String>,
    pub timestamp: Instant,
}
/// Predictive streaming engine for anticipating data patterns
pub struct PredictiveStreamingEngine<A: Float + Send + Sync> {
    /// Prediction model state
    pub(super) prediction_model: PredictionModel<A>,
    /// Historical data for pattern learning
    pub(super) historical_buffer: VecDeque<StreamingDataPoint<A>>,
    /// Maximum number of samples retained in `historical_buffer`. Tracked
    /// explicitly because `VecDeque::capacity` is only a lower bound on the
    /// allocation and grows on its own, so using it as the retention limit
    /// (as this type previously did) never actually bounded the buffer.
    pub(super) history_capacity: usize,
    /// Prediction horizon (time steps)
    pub(super) prediction_horizon: usize,
    /// Confidence threshold for predictions
    pub(super) confidence_threshold: A,
    /// Adaptation rate for model updates
    pub(super) adaptation_rate: A,
}
impl<A: Float + Send + Sync + Send + Sync> PredictiveStreamingEngine<A> {
    pub fn new(config: &StreamingConfig) -> Result<Self> {
        let adaptation_rate = to_a_or(0.1, A::one());
        // The adaptation rate is the engine's single "how fast do we chase a
        // changing stream" knob; the model's exponential forgetting factor is
        // derived from it so the knob actually controls something. A rate of
        // 0.1 gives lambda = 0.99, i.e. an effective memory of ~100 samples.
        let forgetting_factor =
            (A::one() - adaptation_rate / to_a_or(10.0, A::one())).max(to_a_or(0.5, A::one()));
        let history_capacity = (config.buffer_size * 2).max(DEFAULT_MODEL_ORDER + 1);
        Ok(Self {
            prediction_model: PredictionModel::with_forgetting_factor(
                DEFAULT_MODEL_ORDER,
                forgetting_factor,
            )?,
            historical_buffer: VecDeque::with_capacity(history_capacity),
            history_capacity,
            prediction_horizon: 10,
            confidence_threshold: to_a_or(0.8, A::one()),
            adaptation_rate,
        })
    }

    /// Train on the newly arrived samples, then forecast the continuation of
    /// the stream.
    ///
    /// T8: every arriving sample is now actually *used to fit* the model
    /// (one exact recursive-least-squares update per observed coordinate)
    /// before any prediction is made, and predictions are only emitted once
    /// the model's measured out-of-sample accuracy clears
    /// `confidence_threshold`. Previously the model was never trained and
    /// `predict_next` returned copies of the most recent observation, which
    /// carried no information about the future at all.
    pub fn predict_next(
        &mut self,
        current_data: &[StreamingDataPoint<A>],
    ) -> Result<Vec<StreamingDataPoint<A>>> {
        for data_point in current_data {
            self.prediction_model.observe(data_point)?;
            self.historical_buffer.push_back(data_point.clone());
            while self.historical_buffer.len() > self.history_capacity {
                self.historical_buffer.pop_front();
            }
        }
        if self.prediction_model.confidence() < self.confidence_threshold {
            return Ok(Vec::new());
        }
        self.prediction_model
            .predict(&self.historical_buffer, self.prediction_horizon)
    }

    /// The model's current measured confidence in `[0, 1]`.
    pub fn prediction_confidence(&self) -> A {
        self.prediction_model.confidence()
    }

    /// Measured out-of-sample mean absolute one-step prediction error.
    pub fn prediction_error(&self) -> A {
        self.prediction_model.mean_absolute_error()
    }

    /// Measured prediction error relative to the scale of the observed data.
    pub fn normalized_prediction_error(&self) -> A {
        self.prediction_model.normalized_error()
    }

    /// The engine's adaptation rate (how aggressively the fit chases a
    /// changing stream).
    pub fn adaptation_rate(&self) -> A {
        self.adaptation_rate
    }
}
/// Real-time metrics
#[derive(Debug, Clone, Default)]
pub struct RealTimeMetrics {
    pub avg_processing_time_us: f64,
    pub worst_case_latency_us: f64,
    pub deadline_misses: usize,
    pub cpu_utilization: f64,
    pub memory_pressure: f64,
}
/// Stream configuration for individual streams
#[derive(Debug, Clone)]
pub struct StreamConfig<A: Float + Send + Sync> {
    pub buffer_size: usize,
    pub latency_tolerance_ms: u64,
    pub throughput_target: f64,
    pub quality_threshold: A,
}
/// Asynchronous update entry.
///
/// Previously also carried a `timestamp` and an `UpdatePriority`, both of
/// which were written on every enqueue and never read by anything: staleness
/// is tracked by the `staleness` counter (which `max_staleness` is expressed
/// in), and the queue is drained strictly FIFO, so a per-entry priority had
/// no consumer.
#[derive(Debug, Clone)]
pub(super) struct AsyncUpdate<A: Float, D: scirs2_core::ndarray::Dimension> {
    /// Parameter update
    pub(super) update: ArrayBase<scirs2_core::ndarray::OwnedRepr<A>, D>,
    /// Staleness
    pub(super) staleness: usize,
}
/// Stage coordination
#[derive(Debug, Clone)]
pub struct StageCoordinator {
    pub coordination_strategy: CoordinationStrategy,
    pub synchronization_barriers: Vec<SyncBarrier>,
    pub parallelismdegree: usize,
}
impl StageCoordinator {
    pub fn new(parallelismdegree: usize) -> Self {
        Self {
            coordination_strategy: CoordinationStrategy::DataParallel,
            synchronization_barriers: Vec::new(),
            parallelismdegree,
        }
    }
}
/// Current resource usage
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub memory_usage_mb: usize,
    pub cpu_usage_percent: f64,
    pub bandwidth_usage_mbps: f64,
    pub storage_usage_mb: usize,
}
/// Metadata key a data point uses to declare which logical stream it came
/// from. Points without it are attributed to [`DEFAULT_STREAM_ID`].
pub const STREAM_ID_METADATA_KEY: &str = "stream_id";

/// Stream id used for points that do not declare one.
pub const DEFAULT_STREAM_ID: &str = "default";

/// Multi-stream coordinator for synchronizing multiple data streams
pub struct MultiStreamCoordinator<A: Float + Send + Sync> {
    /// Stream configurations
    pub(super) stream_configs: HashMap<String, StreamConfig<A>>,
    /// Synchronization buffer
    pub(super) sync_buffer: HashMap<String, VecDeque<StreamingDataPoint<A>>>,
    /// Global clock for synchronization
    pub(super) global_clock: Instant,
    /// Maximum synchronization window
    pub(super) max_sync_window_ms: u64,
    /// Stream priorities
    pub(super) stream_priorities: HashMap<String, StreamPriority>,
    /// Load balancing strategy
    pub(super) load_balancer: LoadBalancingStrategy,
    /// Arrival instant of the newest sample seen on each stream. Kept
    /// separately from `sync_buffer` so the measured skew survives the
    /// window-based draining that `coordinate_streams` performs.
    pub(super) last_arrival: HashMap<String, Instant>,
    /// Number of samples ever observed per stream.
    pub(super) arrival_counts: HashMap<String, usize>,
}
impl<A: Float + Send + Sync + Send + Sync> MultiStreamCoordinator<A> {
    pub fn new(config: &StreamingConfig) -> Result<Self> {
        Ok(Self {
            stream_configs: HashMap::new(),
            sync_buffer: HashMap::new(),
            global_clock: Instant::now(),
            max_sync_window_ms: config.latency_budget_ms * 2,
            stream_priorities: HashMap::new(),
            load_balancer: LoadBalancingStrategy::RoundRobin,
            last_arrival: HashMap::new(),
            arrival_counts: HashMap::new(),
        })
    }
    /// Add a new stream
    pub fn add_stream(
        &mut self,
        stream_id: String,
        config: StreamConfig<A>,
        priority: StreamPriority,
    ) {
        self.stream_configs.insert(stream_id.clone(), config);
        self.sync_buffer.insert(stream_id.clone(), VecDeque::new());
        self.stream_priorities.insert(stream_id, priority);
    }

    /// The logical stream a data point belongs to.
    pub fn stream_id_of(point: &StreamingDataPoint<A>) -> &str {
        point
            .metadata
            .get(STREAM_ID_METADATA_KEY)
            .map(String::as_str)
            .unwrap_or(DEFAULT_STREAM_ID)
    }

    /// Record the arrival of one sample, registering its stream on first
    /// sight (T13: nothing ever fed `sync_buffer` before, so every
    /// coordination decision was made over permanently empty state).
    pub fn record_arrival(&mut self, point: &StreamingDataPoint<A>) {
        let stream_id = Self::stream_id_of(point).to_string();
        if !self.stream_configs.contains_key(&stream_id) {
            let config = StreamConfig {
                buffer_size: 0,
                latency_tolerance_ms: self.max_sync_window_ms,
                throughput_target: 0.0,
                quality_threshold: A::zero(),
            };
            self.add_stream(stream_id.clone(), config, StreamPriority::Normal);
        }
        if let Some(buffer) = self.sync_buffer.get_mut(&stream_id) {
            buffer.push_back(point.clone());
        }
        self.last_arrival.insert(stream_id.clone(), point.timestamp);
        *self.arrival_counts.entry(stream_id).or_insert(0) += 1;
    }

    /// Number of streams seen so far.
    pub fn stream_count(&self) -> usize {
        self.stream_configs.len()
    }

    /// Samples observed per stream.
    pub fn arrival_counts(&self) -> &HashMap<String, usize> {
        &self.arrival_counts
    }

    /// Measured spread between the newest arrival of the earliest stream and
    /// that of the latest one, in milliseconds. `None` while fewer than two
    /// streams have been seen (there is nothing to be out of sync with).
    pub fn synchronization_skew_ms(&self) -> Option<f64> {
        if self.last_arrival.len() < 2 {
            return None;
        }
        let newest = self.last_arrival.values().max()?;
        let oldest = self.last_arrival.values().min()?;
        Some(newest.saturating_duration_since(*oldest).as_secs_f64() * 1000.0)
    }

    /// The configured synchronization window in milliseconds.
    pub fn max_sync_window_ms(&self) -> u64 {
        self.max_sync_window_ms
    }

    /// Per-stream freshness weights in `(0, 1]`, computed from how stale each
    /// stream's newest sample is relative to the newest sample overall:
    /// `1 / (1 + lag / window)`. A stream that has stopped producing loses
    /// influence smoothly instead of continuing to count as an equal.
    pub fn stream_freshness_weights(&self) -> HashMap<String, A> {
        let mut weights = HashMap::new();
        let newest = match self.last_arrival.values().max() {
            Some(instant) => *instant,
            None => return weights,
        };
        let window_ms = self.max_sync_window_ms.max(1) as f64;
        for (stream_id, arrival) in &self.last_arrival {
            let lag_ms = newest.saturating_duration_since(*arrival).as_secs_f64() * 1000.0;
            let weight = 1.0 / (1.0 + lag_ms / window_ms);
            weights.insert(stream_id.clone(), to_a_or(weight, A::one()));
        }
        weights
    }

    /// The load balancing strategy in force.
    pub fn load_balancing_strategy(&self) -> LoadBalancingStrategy {
        self.load_balancer
    }

    /// Select the load balancing strategy.
    pub fn set_load_balancing_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.load_balancer = strategy;
    }

    /// How long the coordinator has been running.
    pub fn uptime(&self) -> Duration {
        self.global_clock.elapsed()
    }
    /// Coordinate data from multiple streams
    pub fn coordinate_streams(&mut self) -> Result<Vec<StreamingDataPoint<A>>> {
        let mut coordinated_data = Vec::new();
        let current_time = Instant::now();
        for (stream_id, buffer) in &mut self.sync_buffer {
            let window_start = current_time - Duration::from_millis(self.max_sync_window_ms);
            buffer.retain(|point| point.timestamp >= window_start);
            if let Some(priority) = self.stream_priorities.get(stream_id) {
                match priority {
                    StreamPriority::RealTime | StreamPriority::Critical => {
                        coordinated_data.extend(buffer.drain(..));
                    }
                    _ => {
                        if buffer.len() >= 10 {
                            coordinated_data.extend(buffer.drain(..buffer.len() / 2));
                        }
                    }
                }
            }
        }
        Ok(coordinated_data)
    }
}
/// Fusion strategies for combining optimization streams
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    WeightedAverage,
    MedianFusion,
    ConsensusBased,
    AdaptiveFusion,
}
