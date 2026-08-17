//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::constants::PERFORMANCE_HISTORY_CAPACITY;
use super::functions::to_a_or;
use super::pipeline::PipelineExecutionManager;
use super::primitives::{
    AsyncUpdate, AsyncUpdateState, LearningRateAdaptation, MultiStreamCoordinator,
    PredictiveStreamingEngine, RTOptimizationResult, ResourceAllocation, StreamingConfig,
    StreamingMetrics, TimingTracker,
};
use super::realtime::RealTimeOptimizer;
use super::types_3::{
    AdaptiveResourceManager, AdvancedQoSManager, LearningRateAdaptationState, QoSStatus,
    StreamFusionOptimizer, StreamingHealthStatus,
};
use super::wiring::PredictionSignal;

/// Streaming gradient descent optimizer
pub struct StreamingOptimizer<O, A, D>
where
    A: Float + Send + Sync + ScalarOperand + Debug,
    D: scirs2_core::ndarray::Dimension,
    O: Optimizer<A, D>,
{
    /// Base optimizer
    pub(super) baseoptimizer: O,
    /// Configuration
    pub(super) config: StreamingConfig,
    /// Data buffer for mini-batches
    pub data_buffer: VecDeque<StreamingDataPoint<A>>,
    /// Gradient buffer
    pub(super) gradient_buffer: Option<Array1<A>>,
    /// Live model parameters, updated in place by every sync/async step.
    /// `None` until the feature dimensionality is known (first sample seen).
    pub(super) current_parameters: Option<Array1<A>>,
    /// Learning rate adaptation state
    pub(super) lr_adaptation_state: LearningRateAdaptationState<A>,
    /// Concept drift detector
    pub(super) drift_detector: StreamingDriftDetector<A>,
    /// Performance metrics
    pub(super) metrics: StreamingMetrics,
    /// Timing information
    pub(super) timing: TimingTracker,
    /// Memory usage tracker
    pub(super) memory_tracker: MemoryTracker,
    /// Asynchronous update state
    pub(super) async_state: Option<AsyncUpdateState<A, D>>,
    /// Current step count
    pub step_count: usize,
    /// Multi-stream coordinator
    pub(super) multi_stream_coordinator: Option<MultiStreamCoordinator<A>>,
    /// Predictive streaming engine
    pub(super) predictive_engine: Option<PredictiveStreamingEngine<A>>,
    /// Stream fusion optimizer
    pub(super) fusion_optimizer: Option<StreamFusionOptimizer<A>>,
    /// Advanced QoS manager
    pub(super) qos_manager: AdvancedQoSManager,
    /// Real-time performance optimizer
    pub(super) rt_optimizer: RealTimeOptimizer,
    /// Resource allocation manager
    pub(super) resource_manager: Option<AdaptiveResourceManager>,
    /// Pipeline execution manager
    pub(super) pipeline_manager: PipelineExecutionManager<A>,
    /// Most recent QoS evaluation (T10: the QoS manager is now actually
    /// consulted once per processed batch).
    pub(super) qos_status: QoSStatus,
    /// Most recent resource allocation decision, when adaptive resource
    /// allocation is enabled.
    pub(super) last_allocation: Option<ResourceAllocation>,
    /// Most recent real-time optimization outcome.
    pub(super) last_rt_result: Option<RTOptimizationResult>,
    /// State of the predictive engine's live signal.
    pub(super) prediction_signal: PredictionSignal<A>,
    /// Cumulative time actually spent processing batches, and the instant the
    /// optimizer was created: together these give a real CPU duty cycle for
    /// the streaming path rather than an invented utilization figure.
    pub(super) busy_time: Duration,
    pub(super) started_at: Instant,
    /// Measured inter-stream synchronization skew (ms), when multi-stream
    /// coordination is enabled and more than one stream has been seen.
    pub(super) sync_skew_ms: Option<f64>,
}
impl<O, A, D> StreamingOptimizer<O, A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + std::fmt::Debug
        + ScalarOperand
        + std::iter::Sum
        + std::ops::DivAssign,
    D: scirs2_core::ndarray::Dimension,
    O: Optimizer<A, D> + Send + Sync,
{
    /// Create a new streaming optimizer
    pub fn new(baseoptimizer: O, config: StreamingConfig) -> Result<Self> {
        let lr_adaptation_state = LearningRateAdaptationState {
            current_lr: to_a_or(0.01, A::one()),
            accumulated_gradients: None,
            ema_squared_gradients: None,
            performance_history: VecDeque::with_capacity(PERFORMANCE_HISTORY_CAPACITY),
            last_adaptation: Instant::now(),
            adaptation_frequency: Duration::from_millis(1000),
        };
        let drift_detector = StreamingDriftDetector {
            loss_window: VecDeque::with_capacity(config.drift_window_size),
            historical_mean: A::zero(),
            historical_std: A::one(),
            threshold: to_a_or(config.drift_threshold, A::one()),
            last_drift: None,
            drift_count: 0,
            baseline_ready: false,
            update_norm_ema: A::zero(),
            update_norm_var_ema: A::zero(),
            lr_boost_applied_for_current_drift: false,
        };
        let timing = TimingTracker {
            latency_samples: VecDeque::with_capacity(1000),
            batch_start: None,
            max_samples: 1000,
        };
        let memory_tracker = MemoryTracker {
            current_usage: 0,
            peak_usage: 0,
            budget: config.memory_budget_mb * 1024 * 1024,
            usage_history: VecDeque::with_capacity(100),
        };
        let async_state = if config.async_updates {
            Some(AsyncUpdateState {
                update_queue: VecDeque::new(),
                staleness_counter: HashMap::new(),
            })
        } else {
            None
        };
        let multi_stream_coordinator = if config.multi_stream_coordination {
            Some(MultiStreamCoordinator::new(&config)?)
        } else {
            None
        };
        let predictive_engine = if config.predictive_streaming {
            Some(PredictiveStreamingEngine::new(&config)?)
        } else {
            None
        };
        let fusion_optimizer = if config.stream_fusion {
            Some(StreamFusionOptimizer::new(&config)?)
        } else {
            None
        };
        let qos_manager = AdvancedQoSManager::new(config.advanced_qos_config.clone());
        let rt_optimizer = RealTimeOptimizer::new(config.real_time_config.clone())?;
        let resource_manager = if config.adaptive_resource_allocation {
            Some(AdaptiveResourceManager::new(&config)?)
        } else {
            None
        };
        let pipeline_manager = PipelineExecutionManager::new(
            config.pipeline_parallelism_degree,
            config.processingpriority,
        );
        let buffer_size = config.buffer_size;
        Ok(Self {
            baseoptimizer,
            config,
            data_buffer: VecDeque::with_capacity(buffer_size),
            gradient_buffer: None,
            current_parameters: None,
            lr_adaptation_state,
            drift_detector,
            metrics: StreamingMetrics::default(),
            timing,
            memory_tracker,
            async_state,
            step_count: 0,
            multi_stream_coordinator,
            predictive_engine,
            fusion_optimizer,
            qos_manager,
            rt_optimizer,
            resource_manager,
            pipeline_manager,
            qos_status: QoSStatus {
                is_compliant: true,
                violations: Vec::new(),
                timestamp: Instant::now(),
            },
            last_allocation: None,
            last_rt_result: None,
            prediction_signal: PredictionSignal::default(),
            busy_time: Duration::ZERO,
            started_at: Instant::now(),
            sync_skew_ms: None,
        })
    }
    /// Process a single streaming data point
    #[allow(clippy::too_many_arguments)]
    pub fn process_sample(
        &mut self,
        data_point: StreamingDataPoint<A>,
    ) -> Result<Option<Array1<A>>> {
        let starttime = Instant::now();
        if self.timing.batch_start.is_none() {
            self.timing.batch_start = Some(starttime);
        }
        // T13: the multi-stream coordinator sees every arrival, so its
        // synchronization state is derived from real traffic.
        self.record_stream_arrival(&data_point);
        self.data_buffer.push_back(data_point);
        self.update_memory_usage();
        let should_update = self.data_buffer.len() >= self.effective_buffer_size()
            || self.should_force_update(starttime);
        if should_update {
            let result = self.process_buffer()?;
            let latency = starttime.elapsed();
            self.update_timing_metrics(latency);
            if let Some(ref update) = result {
                self.check_concept_drift(update)?;
            }
            // T10: QoS, resource allocation and the real-time controller all
            // run against the metrics this batch just produced, and their
            // decisions feed back into the next batch's behaviour.
            self.run_subsystem_controllers()?;
            Ok(result)
        } else {
            Ok(None)
        }
    }
    pub(super) fn should_force_update(&self, starttime: Instant) -> bool {
        if let Some(batch_start) = self.timing.batch_start {
            let elapsed = starttime.duration_since(batch_start);
            elapsed.as_millis() as u64 >= self.force_update_threshold_ms()
        } else {
            false
        }
    }
    /// Elapsed time, in milliseconds, after which an unfilled batch is flushed
    /// anyway.
    ///
    /// Half the configured latency budget, shortened as the real-time
    /// controller escalates (trading batch efficiency for lower per-sample
    /// latency). The escalated value is floored at 1ms: integer division would
    /// otherwise collapse a moderate budget to a threshold of zero at a high
    /// level, which silently forces a one-sample batch on *every* sample and
    /// destroys throughput. Level 0 is left exactly as it was, budget included.
    pub(super) fn force_update_threshold_ms(&self) -> u64 {
        let base = self.config.latency_budget_ms / 2;
        let level = self.rt_optimizer.optimization_state().optimization_level as u64;
        if level == 0 {
            base
        } else {
            (base / (1 + level)).max(1)
        }
    }
    pub(super) fn process_buffer(&mut self) -> Result<Option<Array1<A>>> {
        if self.data_buffer.is_empty() {
            return Ok(None);
        }
        let processing_start = Instant::now();
        let ingested = self.data_buffer.len();
        let batch_window = self.timing.batch_start.map(|start| start.elapsed());
        // T12: the configured pipeline stages actually run over the ingested
        // batch before anything is learned from it.
        let ingested_batch: Vec<StreamingDataPoint<A>> = self.data_buffer.drain(..).collect();
        let batch = self.pipeline_manager.execute_pipeline(ingested_batch)?;
        if batch.is_empty() {
            // Every sample was filtered out by the pipeline: there is nothing
            // to learn from, but the samples were still ingested.
            self.busy_time += processing_start.elapsed();
            self.update_metrics(ingested, batch_window);
            self.timing.batch_start = None;
            return Ok(None);
        }
        let featuredim = batch[0].features.len();
        self.ensure_parameters_initialized(featuredim);
        let current_params = self.get_current_parameters()?;
        let (gradient, batch_loss) = self.compute_batch_gradient(&batch, &current_params)?;
        self.metrics.current_loss = batch_loss.to_f64().unwrap_or(0.0);
        self.lr_adaptation_state
            .performance_history
            .push_back(batch_loss);
        if self.lr_adaptation_state.performance_history.len() > PERFORMANCE_HISTORY_CAPACITY {
            self.lr_adaptation_state.performance_history.pop_front();
        }
        // T8: the predictive engine is trained on the batch and its measured
        // accuracy becomes a live drift signal.
        self.update_prediction_signal(&batch)?;
        // T10: with stream fusion enabled the applied gradient is the robust
        // fusion of each contributing stream's own gradient.
        let gradient = self.fuse_stream_gradients(&batch, &current_params, gradient)?;
        let compressed_gradient = if self.config.gradient_compression {
            self.compress_gradient(&gradient)?
        } else {
            gradient
        };
        self.adapt_learning_rate(&compressed_gradient)?;
        if self.config.adaptive_learning_rate {
            self.baseoptimizer
                .set_learning_rate(self.lr_adaptation_state.current_lr);
        }
        let updated_params = if self.config.async_updates {
            self.async_update(&current_params, &compressed_gradient)?
        } else {
            self.sync_update(&current_params, &compressed_gradient)?
        };
        self.current_parameters = Some(updated_params.clone());
        // Retain the gradient that was actually applied, so the memory
        // accounting below reflects it (the buffer used to be permanently
        // `None`, making the gradient's contribution silently zero) and
        // callers can inspect the last real update direction.
        self.gradient_buffer = Some(compressed_gradient);
        self.step_count += 1;
        self.busy_time += processing_start.elapsed();
        self.update_metrics(ingested, batch_window);
        self.timing.batch_start = None;
        Ok(Some(updated_params))
    }
    /// Compute the mini-batch gradient of squared-error loss for a simple
    /// linear model `prediction = dot(params, features)`, along with the
    /// mean squared error of the batch (used to report `current_loss` and
    /// to drive concept-drift detection).
    ///
    /// Using the *actual current parameters* here (rather than a hardcoded
    /// zero prediction) is what lets the streaming optimizer converge: the
    /// residual `prediction - target` now reflects the model's real error.
    ///
    /// Takes the batch explicitly rather than reading `data_buffer`, because
    /// the batch that is learned from is the *pipeline output*, which is not
    /// necessarily what was ingested.
    pub(super) fn compute_batch_gradient(
        &self,
        batch: &[StreamingDataPoint<A>],
        params: &Array1<A>,
    ) -> Result<(Array1<A>, A)> {
        if batch.is_empty() {
            return Err(OptimError::InvalidConfig("Empty data buffer".to_string()));
        }
        let batch_size = batch.len();
        let featuredim = batch[0].features.len();
        let mut gradient = Array1::zeros(featuredim);
        let mut loss_sum = A::zero();
        let mut loss_count = 0usize;
        for data_point in batch {
            if let Some(target) = data_point.target {
                let prediction = if params.len() == data_point.features.len() {
                    data_point
                        .features
                        .iter()
                        .zip(params.iter())
                        .map(|(&feature, &weight)| feature * weight)
                        .sum::<A>()
                } else {
                    A::zero()
                };
                let error = prediction - target;
                loss_sum = loss_sum + error * error;
                loss_count += 1;
                for (i, &feature) in data_point.features.iter().enumerate() {
                    if i < featuredim {
                        gradient[i] = gradient[i] + error * feature * data_point.weight;
                    }
                }
            }
        }
        let batch_size_a = to_a_or(batch_size as f64, A::one());
        gradient.mapv_inplace(|g| g / batch_size_a);
        let mean_loss = if loss_count > 0 {
            loss_sum / to_a_or(loss_count as f64, A::one())
        } else {
            A::zero()
        };
        Ok((gradient, mean_loss))
    }
    pub(super) fn compress_gradient(&self, gradient: &Array1<A>) -> Result<Array1<A>> {
        let k = ((gradient.len() as f64 * self.effective_compression_ratio()) as usize)
            .max(1)
            .min(gradient.len());
        let mut compressed = gradient.clone();
        let mut abs_values: Vec<(usize, A)> = gradient
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect();
        abs_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, _) in abs_values.iter().skip(k) {
            compressed[*i] = A::zero();
        }
        Ok(compressed)
    }
    pub(super) fn adapt_learning_rate(&mut self, gradient: &Array1<A>) -> Result<()> {
        if !self.config.adaptive_learning_rate {
            return Ok(());
        }
        match self.config.lr_adaptation {
            LearningRateAdaptation::Fixed => {}
            LearningRateAdaptation::Adagrad => {
                self.adapt_adagrad(gradient)?;
            }
            LearningRateAdaptation::RMSprop => {
                self.adapt_rmsprop(gradient)?;
            }
            LearningRateAdaptation::PerformanceBased => {
                self.adapt_performance_based()?;
            }
            LearningRateAdaptation::DriftAware => {
                self.adapt_drift_aware()?;
            }
            LearningRateAdaptation::AdaptiveMomentum => {
                self.adapt_momentum_based(gradient)?;
            }
            LearningRateAdaptation::GradientVariance => {
                self.adapt_gradient_variance(gradient)?;
            }
            LearningRateAdaptation::PredictiveLR => {
                self.adapt_predictive()?;
            }
        }
        Ok(())
    }
    pub(super) fn adapt_adagrad(&mut self, gradient: &Array1<A>) -> Result<()> {
        let acc_grads = self
            .lr_adaptation_state
            .accumulated_gradients
            .get_or_insert_with(|| Array1::zeros(gradient.len()));
        for i in 0..gradient.len() {
            acc_grads[i] = acc_grads[i] + gradient[i] * gradient[i];
        }
        let base_lr = to_a_or(0.01, A::one());
        let eps = to_a_or(1e-8, A::one());
        let norm_sum = acc_grads.iter().copied().sum::<A>();
        let adaptive_factor = (norm_sum + eps).sqrt();
        self.lr_adaptation_state.current_lr = base_lr / adaptive_factor;
        Ok(())
    }
    pub(super) fn adapt_rmsprop(&mut self, gradient: &Array1<A>) -> Result<()> {
        let ema_grads = self
            .lr_adaptation_state
            .ema_squared_gradients
            .get_or_insert_with(|| Array1::zeros(gradient.len()));
        let decay = to_a_or(0.9, A::one());
        let one_minus_decay = A::one() - decay;
        for i in 0..gradient.len() {
            ema_grads[i] = decay * ema_grads[i] + one_minus_decay * gradient[i] * gradient[i];
        }
        let base_lr = to_a_or(0.01, A::one());
        let eps = to_a_or(1e-8, A::one());
        let rms = ema_grads.iter().copied().sum::<A>().sqrt();
        self.lr_adaptation_state.current_lr = base_lr / (rms + eps);
        Ok(())
    }
    pub(super) fn adapt_performance_based(&mut self) -> Result<()> {
        let n = self.lr_adaptation_state.performance_history.len();
        if n < 2 {
            return Ok(());
        }
        let recent_perf = self.lr_adaptation_state.performance_history[n - 1];
        let prev_perf = self.lr_adaptation_state.performance_history[n - 2];
        let improvement = prev_perf - recent_perf;
        if improvement > A::zero() {
            self.lr_adaptation_state.current_lr =
                self.lr_adaptation_state.current_lr * to_a_or(1.01, A::one());
        } else {
            self.lr_adaptation_state.current_lr =
                self.lr_adaptation_state.current_lr * to_a_or(0.99, A::one());
        }
        Ok(())
    }
    pub(super) fn adapt_drift_aware(&mut self) -> Result<()> {
        if let Some(last_drift) = self.drift_detector.last_drift {
            let stale = last_drift.elapsed() >= Duration::from_secs(60);
            if !stale && !self.drift_detector.lr_boost_applied_for_current_drift {
                self.lr_adaptation_state.current_lr =
                    self.lr_adaptation_state.current_lr * to_a_or(1.5, A::one());
                self.drift_detector.lr_boost_applied_for_current_drift = true;
            }
        }
        Ok(())
    }
    pub(super) fn check_concept_drift(&mut self, update: &Array1<A>) -> Result<()> {
        let current_loss = to_a_or(self.metrics.current_loss, A::one());
        self.drift_detector.loss_window.push_back(current_loss);
        if self.drift_detector.loss_window.len() > self.config.drift_window_size {
            self.drift_detector.loss_window.pop_front();
        }
        let update_norm = update
            .iter()
            .map(|&v| v * v)
            .fold(A::zero(), |acc, v| acc + v)
            .sqrt();
        let eps = to_a_or(1e-8, A::one());
        let ewma_alpha = to_a_or(0.1, A::one());
        if self.drift_detector.loss_window.len() >= 10 {
            let mean = self.drift_detector.loss_window.iter().cloned().sum::<A>()
                / to_a_or(self.drift_detector.loss_window.len() as f64, A::one());
            let variance = self
                .drift_detector
                .loss_window
                .iter()
                .map(|&loss| {
                    let diff = loss - mean;
                    diff * diff
                })
                .sum::<A>()
                / to_a_or(self.drift_detector.loss_window.len() as f64, A::one());
            let std = variance.sqrt();
            if !self.drift_detector.baseline_ready {
                self.drift_detector.historical_mean = mean;
                self.drift_detector.historical_std = std;
                self.drift_detector.update_norm_ema = update_norm;
                self.drift_detector.update_norm_var_ema = A::zero();
                self.drift_detector.baseline_ready = true;
                return Ok(());
            }
            let loss_z_score = (current_loss - self.drift_detector.historical_mean).abs()
                / (self.drift_detector.historical_std + eps);
            let update_variance_established = self.drift_detector.update_norm_var_ema > A::zero();
            let update_z_score = if update_variance_established {
                let update_std = self.drift_detector.update_norm_var_ema.sqrt();
                (update_norm - self.drift_detector.update_norm_ema).abs() / (update_std + eps)
            } else {
                A::zero()
            };
            // T8/T10: a predictive model that suddenly stops being able to
            // predict the stream is direct evidence of non-stationarity, so
            // its measured error participates in the drift decision.
            let prediction_drifted = self.predictive_drift_signal();
            let drifted = loss_z_score > self.drift_detector.threshold
                || (update_variance_established && update_z_score > self.drift_detector.threshold)
                || prediction_drifted;
            if drifted {
                self.drift_detector.last_drift = Some(Instant::now());
                self.drift_detector.lr_boost_applied_for_current_drift = false;
                self.drift_detector.drift_count += 1;
                self.metrics.drift_count = self.drift_detector.drift_count;
                self.drift_detector.historical_mean = mean;
                self.drift_detector.historical_std = std;
                self.drift_detector.update_norm_ema = update_norm;
                self.drift_detector.update_norm_var_ema = A::zero();
                self.reset_prediction_baseline();
                if matches!(
                    self.config.lr_adaptation,
                    LearningRateAdaptation::DriftAware
                ) {
                    self.adapt_drift_aware()?;
                }
            } else {
                self.update_prediction_baseline();
                self.drift_detector.historical_mean = self.drift_detector.historical_mean
                    * (A::one() - ewma_alpha)
                    + mean * ewma_alpha;
                self.drift_detector.historical_std =
                    self.drift_detector.historical_std * (A::one() - ewma_alpha) + std * ewma_alpha;
                let diff = update_norm - self.drift_detector.update_norm_ema;
                self.drift_detector.update_norm_ema =
                    self.drift_detector.update_norm_ema + ewma_alpha * diff;
                self.drift_detector.update_norm_var_ema = (A::one() - ewma_alpha)
                    * (self.drift_detector.update_norm_var_ema + ewma_alpha * diff * diff);
            }
        }
        Ok(())
    }
    /// Return the live model parameters. Callers must call
    /// [`Self::ensure_parameters_initialized`] first once the feature
    /// dimensionality is known; before that this returns an empty vector.
    pub(super) fn get_current_parameters(&self) -> Result<Array1<A>> {
        match &self.current_parameters {
            Some(params) => Ok(params.clone()),
            None => Ok(Array1::zeros(0)),
        }
    }
    /// Lazily allocate the live parameter vector once the feature
    /// dimensionality is known. A no-op once parameters already exist.
    pub(super) fn ensure_parameters_initialized(&mut self, featuredim: usize) {
        if self.current_parameters.is_none() {
            self.current_parameters = Some(Array1::zeros(featuredim));
        }
    }
    pub(super) fn sync_update(
        &mut self,
        params: &Array1<A>,
        gradient: &Array1<A>,
    ) -> Result<Array1<A>> {
        let params_owned = params.clone();
        let gradient_owned = gradient.clone();
        let params_generic = params_owned.into_dimensionality::<D>()?;
        let gradient_generic = gradient_owned.into_dimensionality::<D>()?;
        let result = self
            .baseoptimizer
            .step(&params_generic, &gradient_generic)?;
        Ok(result.into_dimensionality::<scirs2_core::ndarray::Ix1>()?)
    }
    /// Enqueue a gradient update for asynchronous application. Existing
    /// queued updates age by one tick of staleness; once the queue is full
    /// or any entry has become too stale, the whole queue is drained and
    /// applied in order via [`Self::process_async_updates`], which is what
    /// actually advances `current_parameters` for async mode.
    pub(super) fn async_update(
        &mut self,
        params: &Array1<A>,
        gradient: &Array1<A>,
    ) -> Result<Array1<A>> {
        let should_process = if let Some(async_state) = self.async_state.as_mut() {
            let gradient_generic = gradient.clone().into_dimensionality::<D>()?;
            for pending in async_state.update_queue.iter_mut() {
                pending.staleness += 1;
            }
            async_state.update_queue.push_back(AsyncUpdate {
                update: gradient_generic,
                staleness: 0,
            });
            async_state.update_queue.len() >= self.config.buffer_size
        } else {
            return Ok(params.clone());
        };
        if should_process || self.max_staleness_reached() {
            return self.process_async_updates();
        }
        Ok(params.clone())
    }
    pub(super) fn max_staleness_reached(&self) -> bool {
        if let Some(ref async_state) = self.async_state {
            async_state
                .update_queue
                .iter()
                .any(|update| update.staleness >= self.config.max_staleness)
        } else {
            false
        }
    }
    /// Drain the queue of pending asynchronous updates and actually apply
    /// each one, in FIFO order, to `current_parameters` via the base
    /// optimizer. Staleness of each applied update is recorded for
    /// diagnostics. Returns the resulting (now up to date) parameters.
    pub(super) fn process_async_updates(&mut self) -> Result<Array1<A>> {
        let mut current_params = self.get_current_parameters()?;
        let pending: Vec<AsyncUpdate<A, D>> = match self.async_state.as_mut() {
            Some(async_state) => async_state.update_queue.drain(..).collect(),
            None => Vec::new(),
        };
        for update in pending {
            if let Some(async_state) = self.async_state.as_mut() {
                *async_state
                    .staleness_counter
                    .entry(update.staleness)
                    .or_insert(0) += 1;
            }
            let params_generic = current_params.clone().into_dimensionality::<D>()?;
            let new_params = self.baseoptimizer.step(&params_generic, &update.update)?;
            current_params = new_params.into_dimensionality::<scirs2_core::ndarray::Ix1>()?;
        }
        self.current_parameters = Some(current_params.clone());
        Ok(current_params)
    }
    pub(super) fn update_timing_metrics(&mut self, latency: Duration) {
        self.timing.latency_samples.push_back(latency);
        if self.timing.latency_samples.len() > self.timing.max_samples {
            self.timing.latency_samples.pop_front();
        }
        if latency.as_millis() as u64 > self.config.latency_budget_ms {
            self.metrics.throughput_violations += 1;
        }
        // T11: the real-time controller only has something to optimize
        // because every real batch latency is handed to it.
        self.rt_optimizer.observe_latency(latency);
    }
    pub(super) fn update_memory_usage(&mut self) {
        let buffer_size = self.data_buffer.len() * std::mem::size_of::<StreamingDataPoint<A>>();
        let gradient_size = self
            .gradient_buffer
            .as_ref()
            .map(|g| g.len() * std::mem::size_of::<A>())
            .unwrap_or(0);
        self.memory_tracker.current_usage = buffer_size + gradient_size;
        self.memory_tracker.peak_usage = self
            .memory_tracker
            .peak_usage
            .max(self.memory_tracker.current_usage);
        self.memory_tracker
            .usage_history
            .push_back(self.memory_tracker.current_usage);
        if self.memory_tracker.usage_history.len() > 100 {
            self.memory_tracker.usage_history.pop_front();
        }
    }
    /// Refresh the reported metrics for a batch of `ingested` samples that
    /// spent `batch_window` accumulating.
    ///
    /// Both are passed explicitly: the batch is drained from `data_buffer`
    /// before the pipeline runs, so reading `data_buffer.len()` here would
    /// (as it once did) measure an already-empty buffer.
    pub(super) fn update_metrics(&mut self, ingested: usize, batch_window: Option<Duration>) {
        self.metrics.samples_processed += ingested;
        if let Some(window) = batch_window {
            let elapsed = window.as_secs_f64();
            if elapsed > 0.0 {
                self.metrics.processing_rate = ingested as f64 / elapsed;
            }
        }
        if !self.timing.latency_samples.is_empty() {
            let sum: Duration = self.timing.latency_samples.iter().sum();
            self.metrics.avg_latency_ms =
                sum.as_millis() as f64 / self.timing.latency_samples.len() as f64;
            let mut sorted_latencies: Vec<_> = self.timing.latency_samples.iter().collect();
            sorted_latencies.sort();
            let p95_index = (0.95 * sorted_latencies.len() as f64) as usize;
            if p95_index < sorted_latencies.len() {
                self.metrics.p95_latency_ms = sorted_latencies[p95_index].as_millis() as f64;
            }
        }
        self.metrics.memory_usage_mb = self.memory_tracker.current_usage as f64 / (1024.0 * 1024.0);
        self.metrics.current_learning_rate =
            self.lr_adaptation_state.current_lr.to_f64().unwrap_or(0.0);
    }
    /// Get current streaming metrics
    pub fn get_metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }
    /// The gradient most recently applied to the parameters, if any.
    pub fn last_gradient(&self) -> Option<&Array1<A>> {
        self.gradient_buffer.as_ref()
    }
    /// Get the live model parameters currently maintained by the streaming
    /// optimizer, if any samples have been processed yet (F27: parameters
    /// are tracked on the optimizer itself, not recomputed from nothing).
    pub fn current_parameters(&self) -> Option<&Array1<A>> {
        self.current_parameters.as_ref()
    }
    /// Check if streaming optimizer is healthy (within budgets)
    pub fn is_healthy(&self) -> StreamingHealthStatus {
        let mut warnings = Vec::new();
        let mut is_healthy = true;
        if self.metrics.avg_latency_ms > self.config.latency_budget_ms as f64 {
            warnings.push("Average latency exceeds budget".to_string());
            is_healthy = false;
        }
        if self.memory_tracker.current_usage > self.memory_tracker.budget {
            warnings.push("Memory usage exceeds budget".to_string());
            is_healthy = false;
        }
        if self.metrics.drift_count > 10 && self.step_count > 0 {
            let drift_rate = self.metrics.drift_count as f64 / self.step_count as f64;
            if drift_rate > 0.1 {
                warnings.push("High concept drift rate detected".to_string());
            }
        }
        StreamingHealthStatus {
            is_healthy,
            warnings,
            metrics: self.metrics.clone(),
        }
    }
    /// Force processing of current buffer
    pub fn flush(&mut self) -> Result<Option<Array1<A>>> {
        if !self.data_buffer.is_empty() {
            self.process_buffer()
        } else {
            Ok(None)
        }
    }
    /// Adaptive momentum-based learning rate adaptation
    pub(super) fn adapt_momentum_based(&mut self, gradient: &Array1<A>) -> Result<()> {
        let momentum = self
            .lr_adaptation_state
            .ema_squared_gradients
            .get_or_insert_with(|| Array1::zeros(gradient.len()));
        let beta = to_a_or(0.9, A::one());
        let one_minus_beta = A::one() - beta;
        for i in 0..gradient.len() {
            momentum[i] = beta * momentum[i] + one_minus_beta * gradient[i];
        }
        let momentum_norm = momentum.iter().map(|&m| m * m).sum::<A>().sqrt();
        let base_lr = to_a_or(0.01, A::one());
        let adaptation_factor = A::one() + momentum_norm * to_a_or(0.1, A::one());
        self.lr_adaptation_state.current_lr = base_lr / adaptation_factor;
        Ok(())
    }
    /// Gradient variance-based learning rate adaptation
    pub(super) fn adapt_gradient_variance(&mut self, gradient: &Array1<A>) -> Result<()> {
        let mean_grad = self
            .lr_adaptation_state
            .accumulated_gradients
            .get_or_insert_with(|| Array1::zeros(gradient.len()));
        let mean_squared_grad = self
            .lr_adaptation_state
            .ema_squared_gradients
            .get_or_insert_with(|| Array1::zeros(gradient.len()));
        let alpha = to_a_or(0.99, A::one());
        let one_minus_alpha = A::one() - alpha;
        for i in 0..gradient.len() {
            mean_grad[i] = alpha * mean_grad[i] + one_minus_alpha * gradient[i];
            mean_squared_grad[i] =
                alpha * mean_squared_grad[i] + one_minus_alpha * gradient[i] * gradient[i];
        }
        let variance = mean_squared_grad
            .iter()
            .zip(mean_grad.iter())
            .map(|(&sq, &m)| sq - m * m)
            .sum::<A>()
            / to_a_or(gradient.len() as f64, A::one());
        let base_lr = to_a_or(0.01, A::one());
        let var_factor = A::one() + variance.sqrt() * to_a_or(10.0, A::one());
        self.lr_adaptation_state.current_lr = base_lr / var_factor;
        Ok(())
    }
    /// Predictive learning rate adaptation
    pub(super) fn adapt_predictive(&mut self) -> Result<()> {
        if self.lr_adaptation_state.performance_history.len() < 3 {
            return Ok(());
        }
        let history = &self.lr_adaptation_state.performance_history;
        let n = history.len();
        let recent_trend = if n >= 3 {
            let last = history[n - 1];
            let second_last = history[n - 2];
            let third_last = history[n - 3];
            let first_diff = last - second_last;
            let second_diff = second_last - third_last;
            first_diff - second_diff
        } else {
            A::zero()
        };
        let adjustment = if recent_trend > A::zero() {
            to_a_or(0.95, A::one())
        } else {
            to_a_or(1.02, A::one())
        };
        self.lr_adaptation_state.current_lr = self.lr_adaptation_state.current_lr * adjustment;
        let min_lr = to_a_or(1e-6, A::one());
        let max_lr = to_a_or(1.0, A::one());
        self.lr_adaptation_state.current_lr =
            self.lr_adaptation_state.current_lr.max(min_lr).min(max_lr);
        Ok(())
    }
}
/// Memory usage tracker
#[derive(Debug)]
pub(super) struct MemoryTracker {
    /// Current estimated usage (bytes)
    pub(super) current_usage: usize,
    /// Peak usage
    pub(super) peak_usage: usize,
    /// Memory budget (bytes)
    pub(super) budget: usize,
    /// Usage history
    pub(super) usage_history: VecDeque<usize>,
}
/// Streaming data point
#[derive(Debug, Clone)]
pub struct StreamingDataPoint<A: Float + Send + Sync> {
    /// Feature vector
    pub features: Array1<A>,
    /// Target value (for supervised learning)
    pub target: Option<A>,
    /// Timestamp
    pub timestamp: Instant,
    /// Sample weight
    pub weight: A,
    /// Metadata
    pub metadata: HashMap<String, String>,
}
/// Resource allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum ResourceAllocationStrategy {
    Static,
    Adaptive,
    PredictiveBased,
    LoadAware,
}
/// Streaming concept drift detection
#[derive(Debug, Clone)]
pub(super) struct StreamingDriftDetector<A: Float + Send + Sync> {
    /// Window of recent losses
    pub(super) loss_window: VecDeque<A>,
    /// Historical loss statistics
    pub(super) historical_mean: A,
    pub(super) historical_std: A,
    /// Drift detection threshold
    pub(super) threshold: A,
    /// Last drift detection time
    pub(super) last_drift: Option<Instant>,
    /// Drift count
    pub(super) drift_count: usize,
    /// T7: whether `historical_mean`/`historical_std` have been bootstrapped
    /// from real data yet. Before this is true they are the constructor's
    /// placeholder defaults (mean 0, std 1), which are not a meaningful
    /// baseline to compare real losses against.
    pub(super) baseline_ready: bool,
    /// T7: EWMA of the parameter-update L2 norm, tracked alongside the loss
    /// baseline so a sudden jump in update magnitude (not just in loss) can
    /// also be recognised as drift — previously the `update` argument to
    /// `check_concept_drift` was accepted but never read.
    pub(super) update_norm_ema: A,
    /// EWMA estimate of the variance of the update norm.
    pub(super) update_norm_var_ema: A,
    /// Whether `adapt_drift_aware`'s learning-rate boost has already been
    /// applied for the drift event currently recorded in `last_drift`. Reset
    /// to `false` every time a *new* drift is detected. Without this,
    /// `adapt_drift_aware` (called up to twice per batch — once from
    /// `adapt_learning_rate`, once from `check_concept_drift` — for as long
    /// as `Instant::now() - last_drift < 60s`) re-multiplies `current_lr` by
    /// 1.5 on every single call, compounding to `inf`/NaN within a few dozen
    /// batches instead of applying a one-shot post-drift boost.
    pub(super) lr_boost_applied_for_current_drift: bool,
}
