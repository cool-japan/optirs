//! Subsystem wiring for [`StreamingOptimizer`] (T10 - T13).
//!
//! Seven subsystems — stream fusion, the predictive engine, the resource
//! manager, the pipeline manager, the multi-stream coordinator, the QoS
//! manager and the real-time optimizer — were built in
//! `StreamingOptimizer::new` and then never referenced again, so their state
//! could not influence anything and `optimize_realtime` fabricated its result
//! outright. This module is where each of them is actually driven from the
//! live optimize path, and where the loop closes: their measurements feed the
//! next batch's batching, compression, drift decision and gradient.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

use crate::optimizers::Optimizer;

use super::functions::to_a_or;
use super::pipeline::PipelineExecutionManager;
use super::primitives::{
    MultiStreamCoordinator, PredictiveStreamingEngine, RTOptimizationResult, ResourceAllocation,
    ResourceUsage,
};
use super::realtime::RealTimeOptimizer;
use super::types_2::{StreamingDataPoint, StreamingOptimizer};
use super::types_3::{QoSObservation, QoSStatus, StreamFusionOptimizer};

/// Smoothing factor for the predictive-engine error baseline.
const PREDICTION_BASELINE_ALPHA: f64 = 0.1;

/// The predictive engine's live signal.
///
/// `normalized_error` is the model's measured out-of-sample error relative to
/// the scale of the data it is predicting, and `error_ema` is its slow
/// baseline: a sudden jump of the former above the latter is what makes the
/// forecast usable as a drift detector.
#[derive(Debug, Clone)]
pub(super) struct PredictionSignal<A: Float + Send + Sync> {
    /// Measured confidence of the model in `[0, 1]`.
    pub(super) confidence: A,
    /// Measured normalized one-step prediction error.
    pub(super) normalized_error: A,
    /// Slow baseline of `normalized_error` during stable periods.
    pub(super) error_ema: A,
    /// Whether `error_ema` has been seeded from real data.
    pub(super) baseline_ready: bool,
    /// Number of points in the most recent forecast.
    pub(super) forecast_len: usize,
}

impl<A: Float + Send + Sync> Default for PredictionSignal<A> {
    fn default() -> Self {
        Self {
            confidence: A::zero(),
            normalized_error: A::zero(),
            error_ema: A::zero(),
            baseline_ready: false,
            forecast_len: 0,
        }
    }
}

impl<O, A, D> StreamingOptimizer<O, A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + Debug
        + ScalarOperand
        + std::iter::Sum
        + std::ops::DivAssign,
    D: scirs2_core::ndarray::Dimension,
    O: Optimizer<A, D> + Send + Sync,
{
    // ----------------------------------------------------------------- //
    // Multi-stream coordination (T13)
    // ----------------------------------------------------------------- //

    /// Register one arrival with the multi-stream coordinator.
    pub(super) fn record_stream_arrival(&mut self, point: &StreamingDataPoint<A>) {
        if let Some(coordinator) = self.multi_stream_coordinator.as_mut() {
            coordinator.record_arrival(point);
        }
    }

    /// Drive the coordinator: drain its synchronization window and refresh the
    /// measured skew. Returns the measured skew in milliseconds, if any.
    pub(super) fn coordinate_streams(&mut self) -> Result<Option<f64>> {
        let skew = match self.multi_stream_coordinator.as_mut() {
            Some(coordinator) => {
                // Measured before draining, since draining empties the window.
                let skew = coordinator.synchronization_skew_ms();
                coordinator.coordinate_streams()?;
                skew
            }
            None => None,
        };
        self.sync_skew_ms = skew;
        Ok(skew)
    }

    /// Measured inter-stream synchronization skew (ms), if multi-stream
    /// coordination is enabled and at least two streams have been seen.
    pub fn stream_synchronization_skew_ms(&self) -> Option<f64> {
        self.sync_skew_ms
    }

    /// Number of distinct logical streams observed so far.
    pub fn registered_stream_count(&self) -> usize {
        self.multi_stream_coordinator
            .as_ref()
            .map(|coordinator| coordinator.stream_count())
            .unwrap_or(0)
    }

    /// Read-only access to the multi-stream coordinator.
    pub fn multi_stream_coordinator(&self) -> Option<&MultiStreamCoordinator<A>> {
        self.multi_stream_coordinator.as_ref()
    }

    // ----------------------------------------------------------------- //
    // Pipeline (T12)
    // ----------------------------------------------------------------- //

    /// Append a processing stage to the pipeline every batch flows through.
    /// The stage function name is validated immediately.
    pub fn add_pipeline_stage(
        &mut self,
        stage_id: impl Into<String>,
        processing_function: &str,
    ) -> Result<()> {
        self.pipeline_manager
            .add_stage(stage_id, processing_function)
    }

    /// Read-only access to the pipeline manager and its measured stage
    /// metrics.
    pub fn pipeline(&self) -> &PipelineExecutionManager<A> {
        &self.pipeline_manager
    }

    // ----------------------------------------------------------------- //
    // Stream fusion (T10)
    // ----------------------------------------------------------------- //

    /// Mutable access to the stream fusion optimizer, for selecting a fusion
    /// strategy or consensus algorithm.
    pub fn fusion_optimizer_mut(&mut self) -> Option<&mut StreamFusionOptimizer<A>> {
        self.fusion_optimizer.as_mut()
    }

    /// Read-only access to the stream fusion optimizer.
    pub fn fusion_optimizer(&self) -> Option<&StreamFusionOptimizer<A>> {
        self.fusion_optimizer.as_ref()
    }

    /// Per-stream gradients over `batch`, keyed by the stream each sample
    /// declared in its metadata (see
    /// [`MultiStreamCoordinator::stream_id_of`]). Streams are returned in a
    /// deterministic (sorted) order so fusion is reproducible.
    pub(super) fn per_stream_gradients(
        &self,
        batch: &[StreamingDataPoint<A>],
        params: &Array1<A>,
    ) -> Result<Vec<(String, Array1<A>)>> {
        let dim = params.len();
        if dim == 0 {
            return Ok(Vec::new());
        }
        let mut accumulators: HashMap<String, (Array1<A>, usize)> = HashMap::new();
        for point in batch {
            if point.features.len() != dim {
                return Err(OptimError::InvalidConfig(format!(
                    "sample has {} features but the model has {dim} parameters",
                    point.features.len()
                )));
            }
            let stream_id = MultiStreamCoordinator::<A>::stream_id_of(point).to_string();
            let entry = accumulators
                .entry(stream_id)
                .or_insert_with(|| (Array1::zeros(dim), 0));
            entry.1 += 1;
            let Some(target) = point.target else {
                continue;
            };
            let prediction = point
                .features
                .iter()
                .zip(params.iter())
                .fold(A::zero(), |acc, (&feature, &weight)| acc + feature * weight);
            let error = prediction - target;
            for (index, &feature) in point.features.iter().enumerate() {
                entry.0[index] = entry.0[index] + error * feature * point.weight;
            }
        }
        let mut per_stream: Vec<(String, Array1<A>)> = accumulators
            .into_iter()
            .map(|(stream_id, (mut gradient, count))| {
                if count > 0 {
                    let divisor = to_a_or(count as f64, A::one());
                    gradient.mapv_inplace(|value| value / divisor);
                }
                (stream_id, gradient)
            })
            .collect();
        per_stream.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(per_stream)
    }

    /// Replace the batch gradient with the fusion of its per-stream gradients
    /// when stream fusion is enabled.
    ///
    /// Stream weights come from the coordinator's measured freshness, so a
    /// stream that has fallen behind loses influence. With a single
    /// contributing stream every strategy reduces to that stream's own
    /// gradient, which is exactly the unfused result — fusion changes the
    /// outcome only when there is genuinely more than one proposal.
    pub(super) fn fuse_stream_gradients(
        &mut self,
        batch: &[StreamingDataPoint<A>],
        params: &Array1<A>,
        gradient: Array1<A>,
    ) -> Result<Array1<A>> {
        if self.fusion_optimizer.is_none() {
            return Ok(gradient);
        }
        let per_stream = self.per_stream_gradients(batch, params)?;
        if per_stream.is_empty() {
            return Ok(gradient);
        }
        let freshness = self
            .multi_stream_coordinator
            .as_ref()
            .map(|coordinator| coordinator.stream_freshness_weights())
            .unwrap_or_default();
        let Some(fusion) = self.fusion_optimizer.as_mut() else {
            return Ok(gradient);
        };
        if !freshness.is_empty() {
            fusion.set_stream_weights(freshness);
        }
        fusion.fuse_optimization_steps(&per_stream)
    }

    // ----------------------------------------------------------------- //
    // Predictive engine (T8)
    // ----------------------------------------------------------------- //

    /// Train the predictive engine on the batch just ingested and refresh the
    /// live prediction signal.
    pub(super) fn update_prediction_signal(
        &mut self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<()> {
        let Some(engine) = self.predictive_engine.as_mut() else {
            return Ok(());
        };
        let forecast = engine.predict_next(batch)?;
        let confidence = engine.prediction_confidence();
        let normalized_error = engine.normalized_prediction_error();
        self.prediction_signal.confidence = confidence;
        self.prediction_signal.normalized_error = normalized_error;
        self.prediction_signal.forecast_len = forecast.len();
        Ok(())
    }

    /// Read-only access to the predictive engine.
    pub fn predictive_engine(&self) -> Option<&PredictiveStreamingEngine<A>> {
        self.predictive_engine.as_ref()
    }

    /// Measured confidence of the predictive engine in `[0, 1]`; zero while
    /// the engine is disabled or untrained.
    pub fn prediction_confidence(&self) -> A {
        self.prediction_signal.confidence
    }

    /// Measured normalized one-step prediction error of the predictive engine.
    pub fn normalized_prediction_error(&self) -> A {
        self.prediction_signal.normalized_error
    }

    /// Number of points in the predictive engine's most recent forecast.
    pub fn forecast_len(&self) -> usize {
        self.prediction_signal.forecast_len
    }

    /// Whether the predictive engine's error has jumped far enough above its
    /// own stable baseline to count as evidence of concept drift.
    ///
    /// Requires a trained model (`confidence > 0`) and an established
    /// baseline, so a cold start cannot manufacture a drift event.
    pub(super) fn predictive_drift_signal(&self) -> bool {
        if !self.prediction_signal.baseline_ready || self.prediction_signal.confidence <= A::zero()
        {
            return false;
        }
        let threshold = self.drift_detector.threshold.max(A::one());
        let bound = self.prediction_signal.error_ema * (A::one() + threshold);
        // A baseline of exactly zero would make any error at all a "drift";
        // require an absolute floor as well.
        let floor = to_a_or(1e-6, A::zero());
        self.prediction_signal.normalized_error > bound.max(floor)
    }

    /// Fold the current prediction error into its stable baseline. Called on
    /// every non-drifting drift check.
    pub(super) fn update_prediction_baseline(&mut self) {
        if self.prediction_signal.confidence <= A::zero() {
            return;
        }
        let alpha = to_a_or(PREDICTION_BASELINE_ALPHA, A::one());
        if self.prediction_signal.baseline_ready {
            self.prediction_signal.error_ema = self.prediction_signal.error_ema
                * (A::one() - alpha)
                + self.prediction_signal.normalized_error * alpha;
        } else {
            self.prediction_signal.error_ema = self.prediction_signal.normalized_error;
            self.prediction_signal.baseline_ready = true;
        }
    }

    /// Re-seed the prediction baseline after a drift event: the old baseline
    /// described a regime that no longer exists.
    pub(super) fn reset_prediction_baseline(&mut self) {
        self.prediction_signal.error_ema = self.prediction_signal.normalized_error;
        self.prediction_signal.baseline_ready = self.prediction_signal.confidence > A::zero();
    }

    // ----------------------------------------------------------------- //
    // QoS, resources and real-time control (T10, T11)
    // ----------------------------------------------------------------- //

    /// Run the control subsystems for the batch that has just been processed.
    ///
    /// Order matters: the QoS evaluation produces the pressure signal the
    /// resource manager reacts to, and the resource manager's decision is
    /// applied to the pipeline before the next batch arrives.
    pub(super) fn run_subsystem_controllers(&mut self) -> Result<()> {
        let skew_ms = self.coordinate_streams()?;

        let cpu_utilization = self.cpu_duty_cycle_percent();
        let prediction_accuracy =
            if self.predictive_engine.is_some() && self.prediction_signal.confidence > A::zero() {
                self.prediction_signal.confidence.to_f64()
            } else {
                None
            };
        let observation = QoSObservation {
            latency_ms: self.metrics.avg_latency_ms,
            throughput_samples_per_sec: self.metrics.processing_rate,
            memory_usage_mb: self.metrics.memory_usage_mb,
            cpu_utilization_percent: Some(cpu_utilization),
            prediction_accuracy,
            stream_synchronization_delay_ms: skew_ms,
        };
        self.qos_status = self.qos_manager.monitor_qos(&observation);
        let qos_pressure = self.qos_manager.pressure();

        let memory_pressure = {
            let budget_mb = (self.memory_tracker.budget as f64) / (1024.0 * 1024.0);
            if budget_mb > 0.0 {
                self.metrics.memory_usage_mb / budget_mb
            } else {
                0.0
            }
        };
        self.rt_optimizer
            .observe_utilization(cpu_utilization, memory_pressure);
        let rt_result = self
            .rt_optimizer
            .optimize_realtime(Duration::from_millis(self.config.latency_budget_ms))?;
        self.last_rt_result = Some(rt_result);

        let priority = self.pipeline_manager.processing_priority();
        let usage = ResourceUsage {
            memory_usage_mb: self.metrics.memory_usage_mb.max(0.0) as usize,
            cpu_usage_percent: cpu_utilization,
            // Bandwidth and storage are not measured by this optimizer; they
            // stay at zero rather than being invented, and a host that does
            // measure them can supply them via `observe_usage`.
            bandwidth_usage_mbps: 0.0,
            storage_usage_mb: 0,
        };
        let metrics_snapshot = self.metrics.clone();
        if let Some(manager) = self.resource_manager.as_mut() {
            manager.observe_usage(usage);
            let allocation = manager.adapt_allocation(&metrics_snapshot, qos_pressure, priority)?;
            self.pipeline_manager
                .set_parallelism_degree(allocation.cpu_allocation);
            self.last_allocation = Some(allocation);
        }
        Ok(())
    }

    /// The most recent QoS evaluation.
    pub fn qos_status(&self) -> &QoSStatus {
        &self.qos_status
    }

    /// How far outside its service level objectives the system currently is.
    pub fn qos_pressure(&self) -> f64 {
        self.qos_manager.pressure()
    }

    /// The most recent resource allocation decision, if adaptive resource
    /// allocation is enabled.
    pub fn resource_allocation(&self) -> Option<&ResourceAllocation> {
        self.last_allocation.as_ref()
    }

    /// The most recent real-time optimization outcome.
    pub fn realtime_result(&self) -> Option<&RTOptimizationResult> {
        self.last_rt_result.as_ref()
    }

    /// Read-only access to the real-time optimizer and its measurements.
    pub fn realtime_optimizer(&self) -> &RealTimeOptimizer {
        &self.rt_optimizer
    }

    /// Measured fraction of wall-clock time this optimizer has spent actually
    /// processing batches, as a percentage. A real duty-cycle measurement,
    /// not a guess at system-wide CPU load.
    pub fn cpu_duty_cycle_percent(&self) -> f64 {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        ((self.busy_time.as_secs_f64() / elapsed) * 100.0).clamp(0.0, 100.0)
    }

    /// Effective compression ratio: tightened as the real-time controller
    /// escalates, so a system missing its deadlines transmits fewer gradient
    /// coordinates per step.
    pub(super) fn effective_compression_ratio(&self) -> f64 {
        let level = self.rt_optimizer.optimization_state().optimization_level as f64;
        (self.config.compression_ratio / (1.0 + level)).clamp(0.0, 1.0)
    }

    /// Effective flush threshold for the ingest buffer.
    ///
    /// Starts from the configured `buffer_size` and is reduced by the
    /// real-time controller's level (flush sooner under deadline pressure)
    /// and, when `dynamic_buffer_sizing` is enabled, by however many samples
    /// actually fit in the memory the resource manager has allocated.
    pub(super) fn effective_buffer_size(&self) -> usize {
        let mut size = self.config.buffer_size.max(1);
        let level = self.rt_optimizer.optimization_state().optimization_level as usize;
        if level > 0 {
            size = (size / (level + 1)).max(1);
        }
        if self.config.dynamic_buffer_sizing {
            if let Some(allocation) = self.last_allocation.as_ref() {
                let bytes = allocation.memory_allocation_mb.saturating_mul(1024 * 1024);
                let per_sample = std::mem::size_of::<StreamingDataPoint<A>>().max(1);
                size = size.min((bytes / per_sample).max(1));
            }
        }
        size
    }
}

#[cfg(test)]
mod subsystem_wiring_tests {
    use super::*;
    use crate::optimizers::SGD;
    use scirs2_core::ndarray::Ix1;
    use std::time::Instant;

    use super::super::primitives::{
        FusionStrategy, LearningRateAdaptation, QoSMetric, ServiceLevelObjective, StreamPriority,
        StreamingConfig, StreamingMetrics,
    };
    use super::super::types_3::{AdaptiveResourceManager, AdvancedQoSConfig};

    type TestOptimizer = StreamingOptimizer<SGD<f64>, f64, Ix1>;

    fn base_config() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 4,
            adaptive_learning_rate: false,
            lr_adaptation: LearningRateAdaptation::Fixed,
            async_updates: false,
            gradient_compression: false,
            predictive_streaming: false,
            stream_fusion: false,
            adaptive_resource_allocation: false,
            multi_stream_coordination: false,
            dynamic_buffer_sizing: false,
            ..Default::default()
        }
    }

    fn point(x0: f64, x1: f64) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(vec![x0, x1]),
            target: Some(x0 - 2.0 * x1),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn stream_point(stream_id: &str, x0: f64, x1: f64, target: f64) -> StreamingDataPoint<f64> {
        let mut metadata = HashMap::new();
        metadata.insert("stream_id".to_string(), stream_id.to_string());
        StreamingDataPoint {
            features: Array1::from_vec(vec![x0, x1]),
            target: Some(target),
            timestamp: Instant::now(),
            weight: 1.0,
            metadata,
        }
    }

    fn feed(optimizer: &mut TestOptimizer, batches: usize) {
        for _ in 0..batches {
            for &(x0, x1) in &[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)] {
                optimizer
                    .process_sample(point(x0, x1))
                    .expect("process_sample");
            }
        }
    }

    /// T10: the QoS manager must actually be consulted, against the
    /// objectives that were configured — not the three hardcoded thresholds
    /// its old implementation compared everything to.
    #[test]
    fn qos_manager_is_consulted_and_honours_configured_objectives() {
        let mut config = base_config();
        config.advanced_qos_config = AdvancedQoSConfig {
            // An unattainable throughput objective: the real measured rate
            // must be reported as a violation.
            service_level_objectives: vec![ServiceLevelObjective {
                metric: QoSMetric::Throughput,
                target_value: 1.0e12,
                tolerance: 0.0,
            }],
            adaptive_adjustment: false,
            ..Default::default()
        };
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");
        feed(&mut optimizer, 3);

        assert!(
            !optimizer.qos_status().is_compliant,
            "T10 regression: the QoS manager was never consulted (status is \
             still the untouched compliant default)"
        );
        assert!(
            !optimizer.qos_manager.violation_history().is_empty(),
            "violations must be recorded in the manager's history"
        );
        assert!(
            optimizer.qos_pressure() > 0.0,
            "a breached objective must produce nonzero pressure"
        );

        // The same run against an attainable objective must be compliant,
        // proving the verdict comes from the configuration.
        let mut config = base_config();
        config.advanced_qos_config = AdvancedQoSConfig {
            service_level_objectives: vec![ServiceLevelObjective {
                metric: QoSMetric::Throughput,
                target_value: 0.0,
                tolerance: 0.0,
            }],
            adaptive_adjustment: false,
            ..Default::default()
        };
        let mut relaxed: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");
        feed(&mut relaxed, 3);
        assert!(
            relaxed.qos_status().is_compliant,
            "an attainable objective must not be reported as violated"
        );
    }

    /// T11: the real-time optimizer must see real latencies and report only
    /// measured outcomes. The old implementation was never called at all, and
    /// would have claimed a 1.2x gain and 5ms saved unconditionally.
    #[test]
    fn realtime_optimizer_observes_real_latencies_and_reports_measurements() {
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), base_config()).expect("construct");
        feed(&mut optimizer, 5);

        assert!(
            optimizer.realtime_optimizer().observation_count() >= 5,
            "T11 regression: no batch latency ever reached the real-time \
             optimizer (observations: {})",
            optimizer.realtime_optimizer().observation_count()
        );
        let result = optimizer
            .realtime_result()
            .expect("T11 regression: optimize_realtime was never called");
        assert_eq!(
            result.performance_gain, 1.0,
            "T11 regression: reported a performance gain that was never measured"
        );
        assert_eq!(
            result.latency_reduction_ms, 0.0,
            "T11 regression: reported a latency reduction that was never measured"
        );
        assert!(
            !result.optimization_applied,
            "nothing needed optimizing on a healthy stream, so nothing may be \
             reported as applied"
        );
        assert!(
            optimizer.cpu_duty_cycle_percent() > 0.0,
            "the measured CPU duty cycle must be a real nonzero fraction"
        );
    }

    /// T10: the resource manager must run and its decision must be derived
    /// from the live load, not from constants.
    #[test]
    fn resource_manager_runs_and_derives_its_decision_from_load() {
        let mut config = base_config();
        config.adaptive_resource_allocation = true;
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");
        feed(&mut optimizer, 3);

        let allocation = optimizer
            .resource_allocation()
            .expect("T10 regression: the resource manager was never invoked");
        assert!(
            allocation.memory_allocation_mb >= 1,
            "an allocation of zero megabytes is not a usable reservation"
        );
        assert!(allocation.cpu_allocation >= 1);
    }

    /// The allocation figures that used to be hardcoded (`cpu_allocation: 2`,
    /// `priority_adjustment: 0`) must now respond to their inputs.
    #[test]
    fn allocation_figures_respond_to_load_and_priority() {
        let config = base_config();
        let mut manager = AdaptiveResourceManager::new(&config).expect("construct");
        let cores = manager.constraints().max_cpu_cores;

        let idle = StreamingMetrics {
            avg_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            ..Default::default()
        };
        let idle_allocation = manager
            .adapt_allocation(&idle, 0.0, StreamPriority::Normal)
            .expect("allocate");
        assert_eq!(
            idle_allocation.cpu_allocation, 1,
            "idle stream needs one core"
        );
        assert_eq!(
            idle_allocation.priority_adjustment, 0,
            "a compliant Normal-priority stream needs no priority change"
        );

        let loaded = StreamingMetrics {
            // Saturating the configured latency budget.
            avg_latency_ms: config.latency_budget_ms as f64,
            memory_usage_mb: 8.0,
            ..Default::default()
        };
        let loaded_allocation = manager
            .adapt_allocation(&loaded, 2.0, StreamPriority::High)
            .expect("allocate");
        assert_eq!(
            loaded_allocation.cpu_allocation, cores,
            "a fully loaded stream must request every available core"
        );
        assert!(
            loaded_allocation.priority_adjustment > idle_allocation.priority_adjustment,
            "a high-priority stream breaching its objectives must ask for more \
             priority than an idle normal one"
        );
        assert!(
            loaded_allocation.memory_allocation_mb > idle_allocation.memory_allocation_mb,
            "memory reservation must follow the real footprint"
        );
        assert_eq!(manager.allocation_history().len(), 2);
    }

    /// T12: a configured pipeline stage must change what the optimizer learns.
    #[test]
    fn pipeline_stages_change_what_the_optimizer_learns() {
        let mut plain: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), base_config()).expect("construct");
        let mut staged: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), base_config()).expect("construct");
        staged
            .add_pipeline_stage("halve", "scale:0.5")
            .expect("add stage");

        feed(&mut plain, 4);
        feed(&mut staged, 4);

        let plain_params = plain.current_parameters().expect("params").clone();
        let staged_params = staged.current_parameters().expect("params").clone();
        let difference: f64 = plain_params
            .iter()
            .zip(staged_params.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            difference > 1e-6,
            "T12 regression: the pipeline stage had no effect at all \
             (execute_pipeline returned the batch unchanged)"
        );
        let metrics = staged.pipeline().stage_metrics();
        assert_eq!(metrics.len(), 1);
        assert!(
            !staged.pipeline().synchronization_barriers().is_empty(),
            "pipeline executions must be recorded by the stage coordinator"
        );
    }

    /// A pipeline that filters everything out means there is nothing to learn
    /// from — the samples are still counted as ingested, and no update is
    /// produced (rather than an update computed from an empty batch).
    #[test]
    fn a_fully_filtering_pipeline_produces_no_update() {
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), base_config()).expect("construct");
        optimizer
            .add_pipeline_stage("drop-all", "filter_low_weight:1000.0")
            .expect("add stage");
        feed(&mut optimizer, 2);
        assert!(optimizer.current_parameters().is_none());
        assert_eq!(optimizer.get_metrics().samples_processed, 8);
    }

    /// T10: with stream fusion enabled the applied gradient is the robust
    /// fusion of each stream's own gradient, so one corrupted stream cannot
    /// drag the model with it.
    #[test]
    fn stream_fusion_rejects_a_corrupted_stream_in_the_live_path() {
        // Six samples, three streams, one of which reports wildly wrong
        // targets. All six land in a single batch.
        let batch = || {
            vec![
                stream_point("a", 1.0, 0.0, 1.0),
                stream_point("a", 0.0, 1.0, -2.0),
                stream_point("b", 1.0, 0.0, 1.0),
                stream_point("b", 0.0, 1.0, -2.0),
                stream_point("bad", 1.0, 0.0, 500.0),
                stream_point("bad", 0.0, 1.0, -500.0),
            ]
        };

        let mut fused_config = base_config();
        fused_config.buffer_size = 6;
        fused_config.stream_fusion = true;
        let mut fused: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), fused_config).expect("construct");
        if let Some(fusion) = fused.fusion_optimizer_mut() {
            fusion.set_fusion_strategy(FusionStrategy::MedianFusion);
        }

        let mut unfused_config = base_config();
        unfused_config.buffer_size = 6;
        let mut unfused: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), unfused_config).expect("construct");

        for sample in batch() {
            fused.process_sample(sample).expect("fused sample");
        }
        for sample in batch() {
            unfused.process_sample(sample).expect("unfused sample");
        }

        let fused_params = fused.current_parameters().expect("fused params").clone();
        let unfused_params = unfused.current_parameters().expect("params").clone();
        let fused_magnitude: f64 = fused_params.iter().map(|value| value.abs()).sum();
        let unfused_magnitude: f64 = unfused_params.iter().map(|value| value.abs()).sum();

        assert!(
            fused_magnitude < unfused_magnitude * 0.5,
            "T10 regression: fusion did not suppress the corrupted stream \
             (fused magnitude {fused_magnitude}, unfused {unfused_magnitude})"
        );
        let fusion = fused
            .fusion_optimizer()
            .expect("fusion optimizer must exist when stream_fusion is enabled");
        assert!(
            !fusion.fusion_history().is_empty(),
            "every fusion must be recorded with its measured agreement"
        );
        assert_eq!(
            fusion.fusion_history()[0].contributing_streams.len(),
            3,
            "all three streams must be recorded as contributors"
        );
    }

    /// T13: the coordinator must see real arrivals, register the streams it
    /// observes, and measure a real skew between them.
    #[test]
    fn coordinator_sees_real_arrivals_and_measures_skew() {
        let mut config = base_config();
        config.buffer_size = 4;
        config.multi_stream_coordination = true;
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");

        // Two streams, the second lagging by a measurable amount.
        let mut stale = stream_point("stale", 1.0, 0.0, 1.0);
        stale.timestamp = Instant::now() - Duration::from_millis(50);
        optimizer.process_sample(stale).expect("stale sample");
        for _ in 0..3 {
            optimizer
                .process_sample(stream_point("fresh", 0.0, 1.0, -2.0))
                .expect("fresh sample");
        }

        assert_eq!(
            optimizer.registered_stream_count(),
            2,
            "T13 regression: no stream was ever registered from live traffic"
        );
        let skew = optimizer
            .stream_synchronization_skew_ms()
            .expect("T13 regression: no synchronization skew was measured");
        assert!(
            skew >= 40.0,
            "the measured skew must reflect the real 50ms lag, got {skew}ms"
        );
        let counts = optimizer
            .multi_stream_coordinator()
            .expect("coordinator")
            .arrival_counts();
        assert_eq!(counts.get("fresh").copied(), Some(3));
        assert_eq!(counts.get("stale").copied(), Some(1));
    }

    /// T13 + T10: the coordinator's measured freshness becomes the fusion
    /// weight, so a stalled stream really does lose influence.
    #[test]
    fn stale_streams_lose_fusion_weight() {
        let mut config = base_config();
        config.buffer_size = 4;
        config.multi_stream_coordination = true;
        config.stream_fusion = true;
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");

        let mut stale = stream_point("stale", 1.0, 0.0, 1.0);
        stale.timestamp = Instant::now() - Duration::from_millis(200);
        optimizer.process_sample(stale).expect("stale sample");
        for _ in 0..3 {
            optimizer
                .process_sample(stream_point("fresh", 0.0, 1.0, -2.0))
                .expect("fresh sample");
        }

        let fusion = optimizer.fusion_optimizer().expect("fusion");
        let stale_weight = fusion
            .stream_weight("stale")
            .expect("stale stream must have been weighted");
        let fresh_weight = fusion
            .stream_weight("fresh")
            .expect("fresh stream must have been weighted");
        assert!(
            stale_weight < fresh_weight,
            "a stalled stream must weigh less than a live one \
             (stale={stale_weight}, fresh={fresh_weight})"
        );
    }

    /// T8: the predictive engine's measured error is a live drift signal —
    /// silent while the stream is stationary, and raised when the model
    /// genuinely stops being able to predict it.
    #[test]
    fn predictive_drift_signal_only_fires_on_a_real_breakdown() {
        let mut config = base_config();
        config.predictive_streaming = true;
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");

        let stationary = |t: usize| -> Vec<StreamingDataPoint<f64>> {
            (0..4)
                .map(|i| {
                    let step = (t * 4 + i) as f64;
                    let value = (2.0 * step).sin();
                    StreamingDataPoint {
                        features: Array1::from_vec(vec![value, 0.6 * (0.7 * step + 0.4).sin()]),
                        target: Some(value),
                        timestamp: Instant::now(),
                        weight: 1.0,
                        metadata: HashMap::new(),
                    }
                })
                .collect()
        };

        for t in 0..120 {
            let batch = stationary(t);
            optimizer
                .update_prediction_signal(&batch)
                .expect("prediction signal");
            optimizer.update_prediction_baseline();
        }
        assert!(
            optimizer.prediction_confidence() > 0.9,
            "the engine should have learned this stream (confidence {})",
            optimizer.prediction_confidence()
        );
        assert!(
            !optimizer.predictive_drift_signal(),
            "a stationary stream must not raise a predictive drift signal \
             (normalized error {}, baseline {})",
            optimizer.normalized_prediction_error(),
            optimizer.prediction_signal.error_ema
        );

        // An abrupt regime change the fitted recurrence cannot follow.
        let shocked: Vec<StreamingDataPoint<f64>> = (0..4)
            .map(|i| StreamingDataPoint {
                features: Array1::from_vec(vec![500.0 + i as f64 * 137.0, -900.0]),
                target: Some(1.0),
                timestamp: Instant::now(),
                weight: 1.0,
                metadata: HashMap::new(),
            })
            .collect();
        optimizer
            .update_prediction_signal(&shocked)
            .expect("prediction signal");
        assert!(
            optimizer.predictive_drift_signal(),
            "a total prediction breakdown must raise the drift signal \
             (normalized error {}, baseline {})",
            optimizer.normalized_prediction_error(),
            optimizer.prediction_signal.error_ema
        );
    }

    /// Real-time escalation shortens the force-flush threshold, but must never
    /// collapse it to zero: a zero threshold force-flushes a one-sample batch
    /// on every single sample, destroying throughput. Integer division makes
    /// that reachable for any budget below `2 * (1 + level)`.
    #[test]
    fn realtime_escalation_never_collapses_the_flush_threshold() {
        let mut config = base_config();
        config.latency_budget_ms = 4;
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.05_f64), config).expect("construct");

        // Level 0 keeps exactly the historical behaviour: budget / 2.
        assert_eq!(optimizer.force_update_threshold_ms(), 2);

        // 4 / (1 + 3) = 1 by division; 4 / 5 would be 0 without the floor.
        optimizer.rt_optimizer.optimization_state.optimization_level = 3;
        let threshold = optimizer.force_update_threshold_ms();
        assert!(
            threshold >= 1,
            "escalated force-flush threshold collapsed to {threshold}ms, which \
             force-flushes every single sample"
        );

        // And a freshly started batch must therefore not be force-flushed.
        optimizer.timing.batch_start = Some(Instant::now());
        assert!(
            !optimizer.should_force_update(Instant::now()),
            "a batch that has just started must not be force-flushed"
        );
    }

    /// T10: with every subsystem enabled, none of them may be left with
    /// untouched constructed state after a real run.
    #[test]
    fn every_subsystem_has_live_state_after_a_real_run() {
        let config = StreamingConfig {
            buffer_size: 4,
            multi_stream_coordination: true,
            predictive_streaming: true,
            stream_fusion: true,
            adaptive_resource_allocation: true,
            adaptive_learning_rate: false,
            lr_adaptation: LearningRateAdaptation::Fixed,
            async_updates: false,
            gradient_compression: false,
            ..Default::default()
        };
        let mut optimizer: TestOptimizer =
            StreamingOptimizer::new(SGD::new(0.01_f64), config).expect("construct");
        optimizer
            .add_pipeline_stage("clip", "clip:10.0")
            .expect("add stage");

        for t in 0..40 {
            for i in 0..4 {
                let step = (t * 4 + i) as f64;
                let stream = if i % 2 == 0 { "even" } else { "odd" };
                let value = (0.3 * step).sin();
                let mut sample = stream_point(stream, value, 0.5 * value, value * 0.25);
                sample.timestamp = Instant::now();
                optimizer.process_sample(sample).expect("process_sample");
            }
        }

        // 1. Pipeline really ran.
        assert!(optimizer.pipeline().stage_count() == 1);
        assert!(!optimizer.pipeline().synchronization_barriers().is_empty());
        // 2. Multi-stream coordinator really tracked both streams.
        assert_eq!(optimizer.registered_stream_count(), 2);
        // 3. Fusion really fused.
        assert!(!optimizer
            .fusion_optimizer()
            .expect("fusion")
            .fusion_history()
            .is_empty());
        // 4. Predictive engine really trained.
        assert!(optimizer.prediction_confidence() > 0.0);
        // 5. Resource manager really allocated.
        assert!(optimizer.resource_allocation().is_some());
        // 6. Real-time optimizer really observed.
        assert!(optimizer.realtime_optimizer().observation_count() > 0);
        // 7. QoS manager really evaluated.
        assert!(!optimizer.qos_manager.smoothed_metrics().is_empty());
    }
}
