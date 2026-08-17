// Core adaptive streaming optimizer implementation
//
// This module contains the main AdaptiveStreamingOptimizer that orchestrates
// all streaming optimization components including drift detection, performance
// tracking, resource management, and adaptive learning rate control.

use super::anomaly_detection::{
    AnomalyDetector, AnomalyDiagnostics, EnsembleAnomalyDetector, MLAnomalyDetector,
    StatisticalAnomalyDetector,
};
use super::buffering::{AdaptiveBuffer, BufferDiagnostics};
use super::config::*;
use super::drift_detection::{DriftDiagnostics, EnhancedDriftDetector};
use super::meta_learning::{
    ExperienceReplay, MetaAction, MetaLearner, MetaLearningDiagnostics, MetaState, StrategySelector,
};
use super::performance::{
    DataStatistics, PerformanceDiagnostics, PerformanceSnapshot, PerformanceTracker,
};
use super::resource_management::{ResourceDiagnostics, ResourceManager, ResourceUsage};

use crate::adaptive_selection::OptimizerType;
// Removed dependency on learned_optimizers - using stub implementation
use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, IxDyn};
use scirs2_core::numeric::Float;
use scirs2_core::ScientificNumber;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Adaptive learning-rate controller.
///
/// O1: this used to be a stub whose every method ignored its arguments — the
/// rate never moved, `compute_adaptation` echoed the base rate back and
/// `last_change` was hard-coded to `None`, so the whole `AdaptationType::
/// LearningRate` pipeline was a no-op that "applied" the same value forever.
///
/// The real controller combines two established online signals, both computed
/// from data the caller already supplies:
///
/// - **Gradient-norm normalisation** (an AdaGrad-style trust region): the rate
///   is scaled by `1 / (1 + sqrt(accumulated squared gradient norm))`, so a
///   burst of large gradients shrinks the step and a quiet stretch restores it.
/// - **Performance feedback**: the sign of the recent loss trend, estimated by
///   ordinary least squares over the supplied metric window, nudges the rate
///   up while the loss is falling and down while it is rising.
///
/// Every update is clamped to `[min_rate, max_rate]` from the configuration and
/// recorded, so `last_change` reports the real delta that was applied.
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateController<A: Float> {
    /// Current learning rate.
    current_lr: A,
    /// Rate the controller was constructed with.
    initial_lr: A,
    /// Lower bound on the rate.
    min_lr: A,
    /// Upper bound on the rate.
    max_lr: A,
    /// AdaGrad-style accumulator of squared gradient norms.
    squared_gradient_norm_sum: A,
    /// Multiplicative step used to act on the performance trend.
    trend_step: A,
    /// Change applied by the most recent update, if any.
    last_change: Option<A>,
    /// Number of updates applied.
    updates: usize,
}

impl<A: Float> AdaptiveLearningRateController<A> {
    /// Builds a controller from the streaming learning-rate configuration.
    pub fn new(config: &StreamingConfig) -> Result<Self, crate::error::OptimError> {
        let lr_config = &config.learning_rate_config;
        let convert = |value: f64, name: &str| -> Result<A, crate::error::OptimError> {
            A::from(value).ok_or_else(|| {
                crate::error::OptimError::InvalidConfig(format!(
                    "learning rate {name} ({value}) is not representable in the element type"
                ))
            })
        };

        let initial_lr = convert(lr_config.initial_rate, "initial_rate")?;
        let min_lr = convert(lr_config.min_rate, "min_rate")?;
        let max_lr = convert(lr_config.max_rate, "max_rate")?;
        if min_lr > max_lr {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "learning rate min_rate ({}) exceeds max_rate ({})",
                lr_config.min_rate, lr_config.max_rate
            )));
        }
        let trend_step = convert(
            lr_config.performance_sensitivity.clamp(1e-6, 0.5),
            "performance_sensitivity",
        )?;

        Ok(Self {
            current_lr: initial_lr.max(min_lr).min(max_lr),
            initial_lr,
            min_lr,
            max_lr,
            squared_gradient_norm_sum: A::zero(),
            trend_step,
            last_change: None,
            updates: 0,
        })
    }

    /// Folds a real gradient into the controller and returns the resulting rate.
    ///
    /// The gradient's squared L2 norm feeds an AdaGrad accumulator, so the
    /// effective rate is `initial / (1 + sqrt(sum of squared norms))` — large or
    /// repeated gradients genuinely shrink the step.
    pub fn update_learning_rate(&mut self, gradient: &Array1<A>) -> A {
        let squared_norm = gradient.iter().fold(A::zero(), |acc, &g| acc + g * g);
        if squared_norm.is_finite() {
            self.squared_gradient_norm_sum = self.squared_gradient_norm_sum + squared_norm;
        }

        let scale = A::one() / (A::one() + self.squared_gradient_norm_sum.sqrt());
        let proposed = (self.initial_lr * scale).max(self.min_lr).min(self.max_lr);
        self.set_rate(proposed);
        self.current_lr
    }

    /// Current learning rate.
    pub fn current_rate(&self) -> A {
        self.current_lr
    }

    /// Accumulated squared gradient norm (the AdaGrad state).
    pub fn accumulated_squared_gradient_norm(&self) -> A {
        self.squared_gradient_norm_sum
    }

    /// Number of updates the controller has applied.
    pub fn update_count(&self) -> usize {
        self.updates
    }

    /// Proposes the next learning rate from a window of recent performance
    /// metrics, most-recent-last.
    ///
    /// The trend is the ordinary-least-squares slope of the metric against its
    /// index. A falling metric (negative slope) means the current rate is
    /// working, so the rate is grown by `1 + performance_sensitivity`; a rising
    /// metric shrinks it by `1 - performance_sensitivity`. With fewer than two
    /// samples there is no trend to read and the current rate is returned
    /// unchanged.
    pub fn compute_adaptation(&self, performance_metrics: &[A]) -> A {
        if performance_metrics.len() < 2 {
            return self.current_lr;
        }

        let n = match A::from(performance_metrics.len()) {
            Some(value) => value,
            None => return self.current_lr,
        };
        let (Some(one_half), Some(six), Some(two)) = (A::from(0.5), A::from(6.0), A::from(2.0))
        else {
            return self.current_lr;
        };

        // Closed forms for sum(x), sum(x^2) over x = 1..=n.
        let sum_x = n * (n + A::one()) * one_half;
        let sum_x_squared = n * (n + A::one()) * (two * n + A::one()) / six;
        let mut sum_y = A::zero();
        let mut sum_xy = A::zero();
        for (index, &value) in performance_metrics.iter().enumerate() {
            let x = match A::from(index + 1) {
                Some(x) => x,
                None => return self.current_lr,
            };
            sum_y = sum_y + value;
            sum_xy = sum_xy + x * value;
        }

        let denominator = n * sum_x_squared - sum_x * sum_x;
        if denominator == A::zero() {
            return self.current_lr;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        let factor = if slope < A::zero() {
            A::one() + self.trend_step
        } else if slope > A::zero() {
            A::one() - self.trend_step
        } else {
            A::one()
        };

        (self.current_lr * factor).max(self.min_lr).min(self.max_lr)
    }

    /// Applies a proposed rate, recording the real delta.
    pub fn apply_adaptation(&mut self, adaptation: A) {
        if !adaptation.is_finite() || adaptation <= A::zero() {
            // A non-positive or non-finite rate would silently destroy the
            // optimizer; ignore it rather than adopting it.
            return;
        }
        self.set_rate(adaptation.max(self.min_lr).min(self.max_lr));
    }

    fn set_rate(&mut self, new_rate: A) {
        let delta = new_rate - self.current_lr;
        if delta != A::zero() {
            self.last_change = Some(delta);
            self.updates += 1;
        }
        self.current_lr = new_rate;
    }

    /// Delta applied by the most recent rate change, or `None` if the rate has
    /// never moved.
    pub fn last_change(&self) -> Option<A> {
        self.last_change
    }

    /// Resets the controller to its configured initial rate.
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr.max(self.min_lr).min(self.max_lr);
        self.squared_gradient_norm_sum = A::zero();
        self.last_change = None;
        self.updates = 0;
    }
}

/// Streaming data point for optimization
#[derive(Debug, Clone)]
pub struct StreamingDataPoint<A: Float + Send + Sync> {
    /// Input features
    pub features: Array1<A>,
    /// Target values (optional for unsupervised learning)
    pub target: Option<Array1<A>>,
    /// Timestamp when data was received
    pub timestamp: Instant,
    /// Data source identifier
    pub source_id: Option<String>,
    /// Data quality score (0.0 to 1.0)
    pub quality_score: A,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Adaptation instruction for optimizer components
#[derive(Debug, Clone)]
pub struct Adaptation<A: Float + Send + Sync> {
    /// Type of adaptation
    pub adaptation_type: AdaptationType,
    /// Magnitude of adaptation
    pub magnitude: A,
    /// Target component for adaptation
    pub target_component: String,
    /// Adaptation parameters
    pub parameters: HashMap<String, A>,
    /// Priority of this adaptation
    pub priority: AdaptationPriority,
    /// Timestamp when adaptation was computed
    pub timestamp: Instant,
}

/// Types of adaptations that can be applied
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptationType {
    /// Adjust learning rate
    LearningRate,
    /// Modify buffer size
    BufferSize,
    /// Change drift sensitivity
    DriftSensitivity,
    /// Update resource allocation
    ResourceAllocation,
    /// Adjust performance thresholds
    PerformanceThreshold,
    /// Modify anomaly detection parameters
    AnomalyDetection,
    /// Update meta-learning parameters
    MetaLearning,
    /// Custom adaptation type
    Custom(String),
}

/// Priority levels for adaptations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdaptationPriority {
    /// Low priority adaptation
    Low = 0,
    /// Normal priority adaptation
    Normal = 1,
    /// High priority adaptation
    High = 2,
    /// Critical adaptation that must be applied immediately
    Critical = 3,
}

/// Statistics for adaptive streaming optimization
#[derive(Debug, Clone, Serialize)]
pub struct AdaptiveStreamingStats {
    /// Total number of data points processed
    pub total_data_points: usize,
    /// Total number of optimization steps performed
    pub optimization_steps: usize,
    /// Number of drift events detected
    pub drift_events: usize,
    /// Number of anomalies detected
    pub anomalies_detected: usize,
    /// Number of adaptations applied
    pub adaptations_applied: usize,
    /// Current buffer size
    pub current_buffer_size: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Average processing time per batch
    pub avg_processing_time_ms: f64,
    /// Resource utilization statistics
    pub resource_utilization: ResourceUsage,
    /// Performance trend (improvement/degradation)
    pub performance_trend: f64,
    /// Meta-learning effectiveness score
    pub meta_learning_score: f64,
}

/// Main adaptive streaming optimizer
pub struct AdaptiveStreamingOptimizer<O, A, D>
where
    A: Float + Default + Clone + Send + Sync + std::iter::Sum,
    D: Dimension,
{
    /// Base optimizer instance
    base_optimizer: O,
    /// Streaming configuration
    config: StreamingConfig,
    /// Adaptive buffer for incoming data
    buffer: AdaptiveBuffer<A>,
    /// Drift detection system
    drift_detector: EnhancedDriftDetector<A>,
    /// Performance tracking system
    performance_tracker: PerformanceTracker<A>,
    /// Resource management system
    resource_manager: ResourceManager,
    /// Meta-learning system
    meta_learner: MetaLearner<A>,
    /// Anomaly detection system
    anomaly_detector: AnomalyDetector<A>,
    /// Learning rate controller
    learning_rate_controller: AdaptiveLearningRateController<A>,
    /// Current model parameters
    parameters: Option<Array<A, D>>,
    /// Optimization statistics
    stats: AdaptiveStreamingStats,
    /// Last adaptation timestamp
    last_adaptation: Instant,
    /// Adaptation history
    adaptation_history: VecDeque<Adaptation<A>>,
    /// Performance baseline for comparison
    performance_baseline: Option<A>,
    /// Rolling window of recently observed feature vectors, used to compute the
    /// real feature-wise median that `adapt_for_anomaly` clips against.
    recent_feature_window: VecDeque<Vec<A>>,
    /// L2 norm of the gradient used by the most recent optimization step.
    last_gradient_norm: Option<A>,
    /// L2 norm of the parameter delta applied by the most recent step.
    last_update_magnitude: Option<A>,
    /// Wall-clock time the most recent optimization step took.
    last_step_duration: Duration,
    /// Phantom data for dimension type
    _phantom: PhantomData<D>,
}

/// Number of data points retained for the rolling feature median.
const FEATURE_WINDOW_CAPACITY: usize = 256;

/// Number of recent performance snapshots the learning-rate controller fits its
/// loss trend over.
const LR_TREND_WINDOW: usize = 10;

/// Relative tolerance within which a regression prediction counts as correct
/// for the reported accuracy metric.
const ACCURACY_RELATIVE_TOLERANCE: f64 = 0.1;

impl<O, A, D> AdaptiveStreamingOptimizer<O, A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + std::iter::Sum
        + std::fmt::Debug
        + std::ops::DivAssign
        + scirs2_core::ndarray::ScalarOperand
        + 'static,
    D: Dimension,
    O: Clone,
{
    /// Creates a new adaptive streaming optimizer
    pub fn new(base_optimizer: O, config: StreamingConfig) -> Result<Self, String> {
        // Validate configuration
        config.validate()?;

        let buffer = AdaptiveBuffer::new(&config)?;
        let drift_detector = EnhancedDriftDetector::new(&config)?;
        let performance_tracker = PerformanceTracker::new(&config)?;
        let resource_manager = ResourceManager::new(&config)?;
        let meta_learner = MetaLearner::new(&config)?;
        let anomaly_detector = AnomalyDetector::new(&config)?;
        let learning_rate_controller =
            AdaptiveLearningRateController::new(&config).map_err(|e| e.to_string())?;

        let stats = AdaptiveStreamingStats {
            total_data_points: 0,
            optimization_steps: 0,
            drift_events: 0,
            anomalies_detected: 0,
            adaptations_applied: 0,
            current_buffer_size: config.buffer_config.initial_size,
            current_learning_rate: config.learning_rate_config.initial_rate,
            avg_processing_time_ms: 0.0,
            resource_utilization: ResourceUsage::default(),
            performance_trend: 0.0,
            meta_learning_score: 0.0,
        };

        Ok(Self {
            base_optimizer,
            config,
            buffer,
            drift_detector,
            performance_tracker,
            resource_manager,
            meta_learner,
            anomaly_detector,
            learning_rate_controller,
            parameters: None,
            stats,
            last_adaptation: Instant::now(),
            adaptation_history: VecDeque::with_capacity(1000),
            performance_baseline: None,
            recent_feature_window: VecDeque::with_capacity(FEATURE_WINDOW_CAPACITY),
            last_gradient_norm: None,
            last_update_magnitude: None,
            last_step_duration: Duration::ZERO,
            _phantom: PhantomData,
        })
    }

    /// Performs an adaptive optimization step with streaming data
    pub fn adaptive_step(
        &mut self,
        data_batch: Vec<StreamingDataPoint<A>>,
    ) -> Result<Array<A, D>, String> {
        let start_time = Instant::now();

        // Update resource utilization tracking
        self.resource_manager.update_utilization()?;

        // Add data to buffer and check for anomalies
        let filtered_batch = self.filter_anomalies(data_batch)?;
        self.buffer.add_batch(filtered_batch)?;

        // Check if buffer should be processed
        if !self.should_process_buffer()? {
            return self
                .parameters
                .clone()
                .ok_or("No parameters available".to_string());
        }

        // Get batch from buffer for processing
        let processing_batch = self.buffer.get_batch_for_processing()?;
        self.stats.total_data_points += processing_batch.len();

        // Detect drift in the data
        let drift_detected = self.drift_detector.detect_drift(&processing_batch)?;
        if drift_detected {
            self.stats.drift_events += 1;
        }

        // Compute necessary adaptations
        let adaptations = self.compute_adaptations(&processing_batch, drift_detected)?;

        // Apply adaptations to system components
        self.apply_adaptations(&adaptations)?;

        // Perform actual optimization step
        let updated_parameters = self.perform_optimization_step(&processing_batch)?;

        // Evaluate performance of the optimization step
        let performance = self.evaluate_performance(&processing_batch, &updated_parameters)?;

        // Feed the buffer the real per-batch processing cost so its latency
        // statistics (and the batch-size decisions that read them) are based on
        // measurements rather than the zero they were stuck at.
        self.buffer
            .record_processing_duration(self.last_step_duration);

        // Update performance tracking
        self.performance_tracker
            .add_performance(performance.clone())?;

        // Update meta-learner with experience
        self.update_meta_learner(&processing_batch, &adaptations, &performance)?;

        // Update statistics
        self.stats.optimization_steps += 1;
        self.stats.adaptations_applied += adaptations.len();
        self.stats.current_buffer_size = self.buffer.current_size();
        self.stats.current_learning_rate = self
            .learning_rate_controller
            .current_rate()
            .to_f64()
            .unwrap_or(0.0);
        self.stats.performance_trend = self.compute_performance_trend();
        self.stats.meta_learning_score = self
            .meta_learner
            .get_effectiveness_score()
            .to_f64()
            .unwrap_or(0.0);

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.stats.avg_processing_time_ms = (self.stats.avg_processing_time_ms
            * (self.stats.optimization_steps - 1) as f64
            + processing_time)
            / self.stats.optimization_steps as f64;

        // Store updated parameters
        self.parameters = Some(updated_parameters.clone());

        Ok(updated_parameters)
    }

    /// Filters out anomalous data points
    fn filter_anomalies(
        &mut self,
        data_batch: Vec<StreamingDataPoint<A>>,
    ) -> Result<Vec<StreamingDataPoint<A>>, String> {
        if !self.config.anomaly_config.enable_detection {
            return Ok(data_batch);
        }

        // Feed the detector the context it cannot observe for itself, from real
        // current state. A3: `AnomalyContext` used to be built from hard-coded
        // 0.8/0.7, 0.6/0.5 and 0.1 placeholders.
        self.publish_anomaly_context_signals()?;

        let mut filtered_batch = Vec::new();

        for data_point in data_batch {
            // Retain the point in the rolling median window *before* it is
            // classified, so `compute_feature_median` is computed against real
            // history rather than the point itself.
            self.remember_features(&data_point.features);
            let is_anomaly = self.anomaly_detector.detect_anomaly(&data_point)?;

            if is_anomaly {
                self.stats.anomalies_detected += 1;

                // Apply anomaly response strategy
                match &self.config.anomaly_config.response_strategy {
                    AnomalyResponseStrategy::Ignore => {
                        // Include the data point anyway
                        filtered_batch.push(data_point);
                    }
                    AnomalyResponseStrategy::Filter => {
                        // Skip this data point
                        continue;
                    }
                    AnomalyResponseStrategy::Adaptive => {
                        // Adapt the data point or model
                        let adapted_point = self.adapt_for_anomaly(data_point)?;
                        filtered_batch.push(adapted_point);
                    }
                    AnomalyResponseStrategy::Reset => {
                        // Reset relevant components (implemented in apply_adaptations)
                        filtered_batch.push(data_point);
                    }
                    AnomalyResponseStrategy::Custom(_) => {
                        // Custom handling (simplified)
                        filtered_batch.push(data_point);
                    }
                }
            } else {
                filtered_batch.push(data_point);
            }
        }

        Ok(filtered_batch)
    }

    /// Publishes the current performance, resource and drift state into the
    /// anomaly detector, which has no direct handle on any of them.
    fn publish_anomaly_context_signals(&mut self) -> Result<(), String> {
        let performance_metrics: Vec<A> =
            match self.performance_tracker.get_recent_performance(1).first() {
                Some(snapshot) => vec![
                    snapshot.loss,
                    snapshot.accuracy.unwrap_or_else(A::zero),
                    snapshot.convergence_rate.unwrap_or_else(A::zero),
                ],
                None => Vec::new(),
            };

        let usage = self.resource_manager.current_usage()?;
        let mut resource_usage = Vec::new();
        if let Some(memory_mb) = A::from(usage.memory_usage_mb as f64) {
            resource_usage.push(memory_mb);
        }
        if let Some(cpu) = A::from(usage.cpu_usage_percent) {
            resource_usage.push(cpu);
        }

        // The drift detector's live state and observed false-positive rate are
        // the two drift signals this module genuinely knows.
        let diagnostics = self.drift_detector.get_diagnostics();
        let mut drift_indicators = Vec::new();
        if let Some(state) = A::from(match diagnostics.current_state {
            crate::streaming::adaptive_streaming::drift_detection::DriftState::Stable => 0.0,
            crate::streaming::adaptive_streaming::drift_detection::DriftState::Warning => 1.0,
            crate::streaming::adaptive_streaming::drift_detection::DriftState::Drift => 2.0,
            crate::streaming::adaptive_streaming::drift_detection::DriftState::Recovery => 3.0,
        }) {
            drift_indicators.push(state);
        }
        if let Some(fp_rate) = A::from(diagnostics.false_positive_rate) {
            drift_indicators.push(fp_rate);
        }

        // The meta-learner's bandit context needs the same real signals.
        self.meta_learner
            .update_context_signals(resource_usage.clone(), drift_indicators.clone());

        self.anomaly_detector.update_context_signals(
            performance_metrics,
            resource_usage,
            drift_indicators,
        );
        Ok(())
    }

    /// Adapts a data point that was detected as anomalous
    fn adapt_for_anomaly(
        &self,
        mut data_point: StreamingDataPoint<A>,
    ) -> Result<StreamingDataPoint<A>, String> {
        // Simple adaptation: reduce the influence of extreme values
        let median = self.compute_feature_median(&data_point.features)?;

        for (i, value) in data_point.features.iter_mut().enumerate() {
            let diff = (*value - median[i]).abs();
            let threshold =
                median[i] * A::from(self.config.anomaly_config.threshold).expect("unwrap failed");

            if diff > threshold {
                // Clip the value to be within the threshold
                let sign = if *value > median[i] {
                    A::one()
                } else {
                    -A::one()
                };
                *value = median[i] + sign * threshold;
            }
        }

        // Reduce quality score for adapted anomalous data
        data_point.quality_score = data_point.quality_score * A::from(0.5).expect("unwrap failed");

        Ok(data_point)
    }

    /// Computes the feature-wise median over the rolling window of recently
    /// observed data points.
    ///
    /// O2: this used to `return Ok(features.clone())`, which made
    /// `adapt_for_anomaly` a guaranteed no-op — every `diff` was
    /// `|value - value| == 0`, so nothing was ever clipped and the
    /// `AnomalyResponseStrategy::Adaptive` branch silently did nothing beyond
    /// halving the quality score.
    ///
    /// The median is now a genuine per-coordinate order statistic over the
    /// retained window, selected in expected linear time with
    /// `select_nth_unstable_by`. Coordinates the window has never seen fall back
    /// to the incoming value, which is the only defensible estimate available
    /// for them.
    fn compute_feature_median(&self, features: &Array1<A>) -> Result<Array1<A>, String> {
        let window = &self.recent_feature_window;
        if window.is_empty() {
            // No history yet: the point is its own best estimate of the centre.
            return Ok(features.clone());
        }

        let mut medians = Array1::zeros(features.len());
        for index in 0..features.len() {
            let mut column: Vec<A> = window
                .iter()
                .filter_map(|point| point.get(index).copied())
                .filter(|value| !value.is_nan())
                .collect();
            medians[index] = match super::statistics::median_in_place(&mut column) {
                Some(median) => median,
                None => features[index],
            };
        }
        Ok(medians)
    }

    /// Records a data point's features in the rolling window backing
    /// [`Self::compute_feature_median`].
    fn remember_features(&mut self, features: &Array1<A>) {
        if self.recent_feature_window.len() >= FEATURE_WINDOW_CAPACITY {
            self.recent_feature_window.pop_front();
        }
        self.recent_feature_window.push_back(features.to_vec());
    }

    /// Number of data points retained in the median window.
    pub fn feature_window_len(&self) -> usize {
        self.recent_feature_window.len()
    }

    /// Checks if the buffer should be processed
    fn should_process_buffer(&self) -> Result<bool, String> {
        let buffer_quality = self.buffer.get_quality_metrics();
        let buffer_size = self.buffer.current_size();

        // Check size threshold
        let size_threshold = self.config.buffer_config.initial_size;
        let size_ready = buffer_size >= size_threshold;

        // Check quality threshold
        let quality_ready = buffer_quality.average_quality
            >= A::from(self.config.buffer_config.quality_threshold).expect("unwrap failed");

        // Check timeout
        let timeout_ready = self.buffer.time_since_last_processing()
            >= self.config.buffer_config.processing_timeout;

        // Check resource availability
        let resources_available = self
            .resource_manager
            .has_sufficient_resources_for_processing()?;

        Ok((size_ready && quality_ready) || timeout_ready && resources_available)
    }

    /// Computes necessary adaptations based on current state
    fn compute_adaptations(
        &mut self,
        batch: &[StreamingDataPoint<A>],
        drift_detected: bool,
    ) -> Result<Vec<Adaptation<A>>, String> {
        let mut adaptations = Vec::new();

        // Learning rate adaptation, driven by the real recent loss history
        // (oldest first, which is the order `compute_adaptation` fits its
        // trend over). Previously an empty slice was passed, so the controller
        // could never see anything and always echoed its own rate back.
        let mut recent_losses: Vec<A> = self
            .performance_tracker
            .get_recent_performance(LR_TREND_WINDOW)
            .iter()
            .map(|snapshot| snapshot.loss)
            .collect();
        recent_losses.reverse();
        let lr_value = self
            .learning_rate_controller
            .compute_adaptation(&recent_losses);
        let lr_adaptation = Adaptation {
            adaptation_type: AdaptationType::LearningRate,
            magnitude: lr_value,
            target_component: String::from("learning_rate"),
            parameters: HashMap::new(),
            priority: AdaptationPriority::Normal,
            timestamp: Instant::now(),
        };
        adaptations.push(lr_adaptation);

        // Drift-based adaptations
        if drift_detected {
            if let Some(drift_adaptation) = self.drift_detector.compute_sensitivity_adaptation()? {
                adaptations.push(drift_adaptation);
            }
        }

        // Buffer size adaptation
        if let Some(buffer_adaptation) = self
            .buffer
            .compute_size_adaptation(&self.performance_tracker)?
        {
            adaptations.push(buffer_adaptation);
        }

        // Resource allocation adaptation.
        //
        // O3: this was commented out with a "type mismatch (f32 vs A)" note,
        // which silently disabled every memory- and CPU-pressure response the
        // resource manager computes. `ResourceManager` works in `f32`, so the
        // adaptation is converted across the boundary here — the target
        // component string is preserved verbatim because
        // `apply_allocation_adaptation` dispatches on it.
        if let Some(resource_adaptation) = self.resource_manager.compute_allocation_adaptation()? {
            let magnitude = A::from(resource_adaptation.magnitude).ok_or_else(|| {
                format!(
                    "resource adaptation magnitude {} is not representable in the element type",
                    resource_adaptation.magnitude
                )
            })?;
            let mut parameters = HashMap::new();
            for (key, value) in &resource_adaptation.parameters {
                let converted = A::from(*value).ok_or_else(|| {
                    format!("resource adaptation parameter '{key}' ({value}) is not representable")
                })?;
                parameters.insert(key.clone(), converted);
            }
            adaptations.push(Adaptation {
                adaptation_type: resource_adaptation.adaptation_type.clone(),
                magnitude,
                target_component: resource_adaptation.target_component.clone(),
                parameters,
                priority: resource_adaptation.priority.clone(),
                timestamp: resource_adaptation.timestamp,
            });
        }

        // Meta-learning based adaptations
        let meta_adaptations = self
            .meta_learner
            .recommend_adaptations(batch, &self.performance_tracker)?;
        adaptations.extend(meta_adaptations);

        // Sort adaptations by priority
        adaptations.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(adaptations)
    }

    /// Applies computed adaptations to system components
    fn apply_adaptations(&mut self, adaptations: &[Adaptation<A>]) -> Result<(), String> {
        for adaptation in adaptations {
            match &adaptation.adaptation_type {
                AdaptationType::LearningRate => {
                    self.learning_rate_controller
                        .apply_adaptation(adaptation.magnitude);
                }
                AdaptationType::BufferSize => {
                    self.buffer.apply_size_adaptation(adaptation)?;
                }
                AdaptationType::DriftSensitivity => {
                    self.drift_detector
                        .apply_sensitivity_adaptation(adaptation)?;
                }
                AdaptationType::ResourceAllocation => {
                    // Convert back into the `f32` domain the resource manager
                    // works in and apply it for real.
                    let magnitude = adaptation.magnitude.to_f32().ok_or_else(|| {
                        "resource adaptation magnitude is not representable as f32".to_string()
                    })?;
                    let mut parameters = HashMap::new();
                    for (key, value) in &adaptation.parameters {
                        let converted = value.to_f32().ok_or_else(|| {
                            format!(
                                "resource adaptation parameter '{key}' is not representable as f32"
                            )
                        })?;
                        parameters.insert(key.clone(), converted);
                    }
                    let converted = Adaptation::<f32> {
                        adaptation_type: adaptation.adaptation_type.clone(),
                        magnitude,
                        target_component: adaptation.target_component.clone(),
                        parameters,
                        priority: adaptation.priority.clone(),
                        timestamp: adaptation.timestamp,
                    };
                    self.resource_manager
                        .apply_allocation_adaptation(&converted)?;
                }
                AdaptationType::PerformanceThreshold => {
                    self.performance_tracker
                        .apply_threshold_adaptation(adaptation)?;
                }
                AdaptationType::AnomalyDetection => {
                    self.anomaly_detector.apply_adaptation(adaptation)?;
                }
                AdaptationType::MetaLearning => {
                    self.meta_learner.apply_adaptation(adaptation)?;
                }
                AdaptationType::Custom(name) => {
                    // There is no registry of custom adaptation handlers, so
                    // accepting one silently (or merely printing it to stdout
                    // from library code) would let it look applied when nothing
                    // happened.
                    return Err(format!(
                        "no handler is registered for custom adaptation '{name}'"
                    ));
                }
            }

            // Store adaptation in history
            if self.adaptation_history.len() >= 1000 {
                self.adaptation_history.pop_front();
            }
            self.adaptation_history.push_back(adaptation.clone());
        }

        self.last_adaptation = Instant::now();
        Ok(())
    }

    /// Performs the actual optimization step
    fn perform_optimization_step(
        &mut self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<Array<A, D>, String> {
        let started = Instant::now();

        // Compute gradients from the batch
        let gradients = self.compute_batch_gradients(batch)?;

        // Fold the real gradient into the learning-rate controller before the
        // step, so the AdaGrad-style trust region actually sees it. The
        // controller previously never received a gradient at all.
        let learning_rate = self
            .learning_rate_controller
            .update_learning_rate(&gradients);

        // Apply optimization step (simplified implementation)
        let mut updated_parameters = if let Some(params) = self.parameters.clone() {
            params
        } else {
            // Cannot initialize parameters without proper dimension info
            return Err("Parameters not initialized".to_string());
        };

        // Simple gradient descent update (in practice would use the base optimizer)
        let mut squared_update = A::zero();
        for (param, &grad) in updated_parameters.iter_mut().zip(gradients.iter()) {
            let delta = learning_rate * grad;
            squared_update = squared_update + delta * delta;
            *param = *param - delta;
        }

        // Record the real magnitudes so `evaluate_performance` reports
        // measurements instead of the fixed 1.0 / 0.1 placeholders.
        let squared_gradient = gradients.iter().fold(A::zero(), |acc, &g| acc + g * g);
        self.last_gradient_norm = Some(squared_gradient.sqrt());
        self.last_update_magnitude = Some(squared_update.sqrt());
        self.last_step_duration = started.elapsed();

        Ok(updated_parameters)
    }

    /// Computes batch gradients from streaming data
    fn compute_batch_gradients(
        &self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<Array1<A>, String> {
        if batch.is_empty() {
            return Err("Cannot compute gradients from empty batch".to_string());
        }

        let feature_dim = batch[0].features.len();
        let mut gradients = Array1::zeros(feature_dim);

        // Simplified gradient computation (in practice would depend on loss function)
        for data_point in batch {
            for (i, &feature) in data_point.features.iter().enumerate() {
                gradients[i] = gradients[i] + feature * data_point.quality_score;
            }
        }

        // Normalize by batch size
        let batch_size = A::from(batch.len()).expect("unwrap failed");
        gradients /= batch_size;

        Ok(gradients)
    }

    /// Evaluates performance of the optimization step
    fn evaluate_performance(
        &self,
        batch: &[StreamingDataPoint<A>],
        parameters: &Array<A, D>,
    ) -> Result<PerformanceSnapshot<A>, String> {
        // Compute various performance metrics
        let loss = self.compute_loss(batch, parameters)?;
        let accuracy = self.compute_accuracy(batch, parameters)?;
        let convergence_rate = self.compute_convergence_rate(parameters)?;

        // Compute data statistics
        let data_stats = self.compute_data_statistics(batch)?;

        // Get resource usage
        let resource_usage = self.resource_manager.current_usage()?;

        let performance = PerformanceSnapshot {
            timestamp: Instant::now(),
            // Real wall-clock cost of the step that produced this snapshot.
            // B2: `compute_size_adaptation` used to read `timestamp.elapsed()`
            // (the snapshot's *age*) as if it were the processing time.
            processing_duration: self.last_step_duration,
            loss,
            accuracy: Some(accuracy),
            convergence_rate: Some(convergence_rate),
            // Measured in `perform_optimization_step`; `None` before the first
            // step rather than a fabricated 1.0 / 0.1.
            gradient_norm: self.last_gradient_norm,
            parameter_update_magnitude: self.last_update_magnitude,
            data_statistics: data_stats,
            resource_usage,
            custom_metrics: HashMap::new(),
        };

        Ok(performance)
    }

    /// Linear prediction of the model for one data point.
    ///
    /// `perform_optimization_step` updates `parameters` coordinate-wise against
    /// the per-feature gradient, so the parameter array is aligned with the
    /// feature vector in row-major order and the model this optimizer is
    /// actually fitting is the linear one `y_hat = <w, x>`. Computing the
    /// prediction that way makes the loss a genuine function of the parameters
    /// rather than of the input alone.
    fn linear_prediction(&self, features: &Array1<A>, parameters: &Array<A, D>) -> A {
        parameters
            .iter()
            .zip(features.iter())
            .fold(A::zero(), |acc, (&weight, &feature)| acc + weight * feature)
    }

    /// Computes mean squared error for the current batch and parameters.
    fn compute_loss(
        &self,
        batch: &[StreamingDataPoint<A>],
        parameters: &Array<A, D>,
    ) -> Result<A, String> {
        // Mean squared error of the model's own prediction. This used to take
        // `prediction = &data_point.features`, i.e. it scored the *input*
        // against the target and ignored `parameters` entirely — so the reported
        // loss never moved when the model improved.
        let mut total_loss = A::zero();
        let mut count = 0usize;

        for data_point in batch {
            let Some(target) = data_point.target.as_ref() else {
                continue;
            };
            let Some(&target_value) = target.iter().next() else {
                continue;
            };
            let prediction = self.linear_prediction(&data_point.features, parameters);
            let residual = prediction - target_value;
            total_loss = total_loss + residual * residual;
            count += 1;
        }

        if count == 0 {
            // No labelled point in the batch: there is no loss to report, which
            // is honestly zero contribution rather than a made-up figure.
            return Ok(A::zero());
        }
        let divisor =
            A::from(count).ok_or_else(|| format!("batch size {count} is not representable"))?;
        Ok(total_loss / divisor)
    }

    /// Computes accuracy for the current batch and parameters.
    ///
    /// For a regression model "accuracy" is the fraction of predictions that
    /// land within a tolerance of the target. The tolerance is the configured
    /// convergence threshold scaled by the target magnitude, so it is
    /// scale-free. This used to count a point as "correct" whenever its
    /// `quality_score > 0.5`, which measured the *input data quality* and had
    /// nothing to do with the model's predictions.
    fn compute_accuracy(
        &self,
        batch: &[StreamingDataPoint<A>],
        parameters: &Array<A, D>,
    ) -> Result<A, String> {
        let relative_tolerance = A::from(ACCURACY_RELATIVE_TOLERANCE).ok_or_else(|| {
            format!("accuracy tolerance {ACCURACY_RELATIVE_TOLERANCE} is not representable")
        })?;
        let epsilon = A::from(1e-8).ok_or_else(|| "1e-8 is not representable".to_string())?;

        let mut correct = 0usize;
        let mut total = 0usize;

        for data_point in batch {
            let Some(target) = data_point.target.as_ref() else {
                continue;
            };
            let Some(&target_value) = target.iter().next() else {
                continue;
            };
            let prediction = self.linear_prediction(&data_point.features, parameters);
            let tolerance = relative_tolerance * target_value.abs().max(epsilon);
            if (prediction - target_value).abs() <= tolerance {
                correct += 1;
            }
            total += 1;
        }

        if total == 0 {
            // Nothing labelled to score against: report zero rather than the
            // perfect `1.0` this used to claim for an unlabelled batch.
            return Ok(A::zero());
        }
        let numerator =
            A::from(correct).ok_or_else(|| format!("{correct} is not representable"))?;
        let denominator = A::from(total).ok_or_else(|| format!("{total} is not representable"))?;
        Ok(numerator / denominator)
    }

    /// Computes convergence rate
    fn compute_convergence_rate(&self, _parameters: &Array<A, D>) -> Result<A, String> {
        // `get_recent_losses` returns most-recent-first (it reverses the
        // history buffer), so index 0 is the newest loss and the last
        // index is the oldest loss in the window (O5 fix). "Convergence
        // rate" should be positive when loss is decreasing: that requires
        // `oldest - newest`, not `newest - oldest` (which the previous code
        // computed, inverting the sign — a genuinely converging model
        // reported a *negative* rate and a diverging one a *positive* rate).
        let recent_losses = self.performance_tracker.get_recent_losses(10);
        if recent_losses.len() >= 2 {
            let newest = recent_losses[0];
            let oldest = recent_losses[recent_losses.len() - 1];
            let improvement = oldest - newest;
            if oldest != A::zero() {
                Ok(improvement / oldest)
            } else {
                Ok(A::zero())
            }
        } else {
            Ok(A::zero())
        }
    }

    /// Computes comprehensive data statistics
    fn compute_data_statistics(
        &self,
        batch: &[StreamingDataPoint<A>],
    ) -> Result<DataStatistics<A>, String> {
        if batch.is_empty() {
            return Ok(DataStatistics::default());
        }

        let feature_dim = batch[0].features.len();
        let mut feature_means = Array1::zeros(feature_dim);
        let mut feature_stds = Array1::zeros(feature_dim);
        let mut quality_scores = Vec::new();

        // Compute means
        for data_point in batch {
            feature_means = feature_means + &data_point.features;
            quality_scores.push(data_point.quality_score);
        }
        feature_means /= A::from(batch.len()).expect("unwrap failed");

        // Compute standard deviations
        for data_point in batch {
            let diff = &data_point.features - &feature_means;
            feature_stds = feature_stds + &diff.mapv(|x| x * x);
        }
        feature_stds /= A::from(batch.len()).expect("unwrap failed");
        feature_stds = feature_stds.mapv(|x| x.sqrt());

        let avg_quality = quality_scores.iter().copied().sum::<A>()
            / A::from(quality_scores.len()).expect("unwrap failed");

        Ok(DataStatistics {
            sample_count: batch.len(),
            feature_means,
            feature_stds,
            average_quality: avg_quality,
            timestamp: Instant::now(),
        })
    }

    /// Updates meta-learner with experience from this optimization step
    fn update_meta_learner(
        &mut self,
        batch: &[StreamingDataPoint<A>],
        adaptations: &[Adaptation<A>],
        performance: &PerformanceSnapshot<A>,
    ) -> Result<(), String> {
        if !self.config.meta_learning_config.enable_meta_learning {
            return Ok(());
        }

        // Extract meta-state from current situation
        let meta_state = self.extract_meta_state(performance)?;

        // Extract meta-action from applied adaptations
        let meta_action = self.extract_meta_action(adaptations)?;

        // Compute reward based on performance improvement
        let reward = self.compute_meta_reward(performance)?;

        // Update meta-learner
        self.meta_learner
            .update_experience(meta_state, meta_action, reward)?;

        Ok(())
    }

    /// Extracts meta-state representation from performance data
    fn extract_meta_state(
        &self,
        performance: &PerformanceSnapshot<A>,
    ) -> Result<MetaState<A>, String> {
        let state = MetaState {
            performance_metrics: vec![
                performance.loss,
                performance.accuracy.unwrap_or(A::zero()),
                performance.convergence_rate.unwrap_or(A::zero()),
            ],
            resource_state: vec![
                A::from(performance.resource_usage.memory_usage_mb as f64).expect("unwrap failed"),
                A::from(performance.resource_usage.cpu_usage_percent).expect("unwrap failed"),
            ],
            drift_indicators: vec![A::from(if self.drift_detector.is_drift_detected() {
                1.0
            } else {
                0.0
            })
            .expect("unwrap failed")],
            adaptation_history: self.adaptation_history.len(),
            timestamp: Instant::now(),
        };

        Ok(state)
    }

    /// Extracts meta-action representation from adaptations
    fn extract_meta_action(&self, adaptations: &[Adaptation<A>]) -> Result<MetaAction<A>, String> {
        let mut adaptation_vector = Vec::new();
        let mut adaptation_types = Vec::new();

        for adaptation in adaptations {
            adaptation_vector.push(adaptation.magnitude);
            adaptation_types.push(adaptation.adaptation_type.clone());
        }

        let action = MetaAction {
            adaptation_magnitudes: adaptation_vector,
            adaptation_types,
            learning_rate_change: self
                .learning_rate_controller
                .last_change()
                .unwrap_or(A::zero()),
            buffer_size_change: A::from(self.buffer.last_size_change()).unwrap_or(A::zero()),
            timestamp: Instant::now(),
        };

        Ok(action)
    }

    /// Computes reward for meta-learning based on performance improvement
    fn compute_meta_reward(&self, performance: &PerformanceSnapshot<A>) -> Result<A, String> {
        // Compare with baseline or previous performance
        let reward = if let Some(baseline) = self.performance_baseline {
            performance.loss - baseline // Negative reward for higher loss
        } else {
            A::zero()
        };

        Ok(reward)
    }

    /// Gets current adaptive streaming statistics
    pub fn get_adaptive_stats(&self) -> AdaptiveStreamingStats {
        let mut stats = self.stats.clone();
        stats.resource_utilization = self.resource_manager.current_usage().unwrap_or_default();
        stats
    }

    /// Counts the number of adaptations applied in recent history
    fn count_adaptations_applied(&self) -> usize {
        let recent_threshold = Instant::now() - Duration::from_secs(300); // Last 5 minutes
        self.adaptation_history
            .iter()
            .filter(|adaptation| adaptation.timestamp > recent_threshold)
            .count()
    }

    /// Computes performance trend over recent optimization steps
    fn compute_performance_trend(&self) -> f64 {
        let recent_performance = self.performance_tracker.get_recent_performance(20);
        if recent_performance.len() >= 2 {
            let recent_avg = recent_performance
                .iter()
                .rev()
                .take(5)
                .map(|p| p.loss.to_f64().unwrap_or(0.0))
                .sum::<f64>()
                / 5.0;

            let older_avg = recent_performance
                .iter()
                .take(5)
                .map(|p| p.loss.to_f64().unwrap_or(0.0))
                .sum::<f64>()
                / 5.0;

            // Negative trend means improvement (lower loss)
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        }
    }

    /// Forces an adaptation cycle even if normal triggers haven't fired
    pub fn force_adaptation(&mut self) -> Result<(), String> {
        let empty_batch = Vec::new();
        let adaptations = self.compute_adaptations(&empty_batch, false)?;
        self.apply_adaptations(&adaptations)?;
        Ok(())
    }

    /// Resets the optimizer to initial state while preserving learned knowledge
    pub fn soft_reset(&mut self) -> Result<(), String> {
        // Reset components while preserving meta-learning knowledge
        self.buffer.reset()?;
        self.drift_detector.reset()?;
        self.performance_tracker.reset()?;

        // Don't reset meta-learner to preserve learned adaptations
        // self.meta_learner.reset()?;

        self.stats = AdaptiveStreamingStats {
            total_data_points: 0,
            optimization_steps: 0,
            drift_events: 0,
            anomalies_detected: 0,
            adaptations_applied: 0,
            current_buffer_size: self.config.buffer_config.initial_size,
            current_learning_rate: self.config.learning_rate_config.initial_rate,
            avg_processing_time_ms: 0.0,
            resource_utilization: ResourceUsage::default(),
            performance_trend: 0.0,
            meta_learning_score: self.meta_learner.get_effectiveness_score() as f64,
        };

        self.adaptation_history.clear();
        self.performance_baseline = None;

        Ok(())
    }

    /// Gets detailed diagnostic information
    pub fn get_diagnostics(&self) -> StreamingDiagnostics {
        StreamingDiagnostics {
            buffer_diagnostics: self.buffer.get_diagnostics(),
            drift_diagnostics: self.drift_detector.get_diagnostics(),
            performance_diagnostics: self.performance_tracker.get_diagnostics(),
            resource_diagnostics: self.resource_manager.get_diagnostics(),
            meta_learning_diagnostics: self.meta_learner.get_diagnostics(),
            anomaly_diagnostics: self.anomaly_detector.get_diagnostics(),
        }
    }
}

/// Comprehensive diagnostic information for streaming optimizer
#[derive(Debug, Clone)]
pub struct StreamingDiagnostics {
    pub buffer_diagnostics: BufferDiagnostics,
    pub drift_diagnostics: DriftDiagnostics,
    pub performance_diagnostics: PerformanceDiagnostics,
    pub resource_diagnostics: ResourceDiagnostics,
    pub meta_learning_diagnostics: MetaLearningDiagnostics,
    pub anomaly_diagnostics: AnomalyDiagnostics,
}

#[cfg(test)]
mod o5_convergence_rate_tests {
    use super::*;
    use scirs2_core::ndarray::Ix1;

    fn make_snapshot(loss: f64) -> PerformanceSnapshot<f64> {
        PerformanceSnapshot {
            timestamp: Instant::now(),
            processing_duration: Duration::from_millis(5),
            loss,
            accuracy: None,
            convergence_rate: None,
            gradient_norm: None,
            parameter_update_magnitude: None,
            data_statistics: DataStatistics::default(),
            resource_usage: ResourceUsage::default(),
            custom_metrics: HashMap::new(),
        }
    }

    fn make_point(features: Vec<f64>) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(features),
            target: None,
            timestamp: Instant::now(),
            source_id: None,
            quality_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn make_optimizer() -> AdaptiveStreamingOptimizer<(), f64, Ix1> {
        AdaptiveStreamingOptimizer::new((), StreamingConfig::default()).expect("construct")
    }

    /// O5: `compute_convergence_rate` must report a *positive* rate when
    /// loss is genuinely decreasing over the tracked window, and negative
    /// when it is increasing. `get_recent_losses` returns most-recent-first,
    /// so the previous `recent_losses[0] - recent_losses[len-1]` computed
    /// `newest - oldest`, which is negative for a converging (loss
    /// decreasing) run — exactly inverted.
    #[test]
    fn positive_rate_when_loss_is_decreasing() {
        let mut opt = make_optimizer();
        // Oldest -> newest: loss falls from 10.0 to 1.0.
        for loss in [10.0, 8.0, 6.0, 4.0, 2.0, 1.0] {
            opt.performance_tracker
                .add_performance(make_snapshot(loss))
                .expect("add_performance");
        }

        let params = Array::<f64, Ix1>::zeros(3);
        let rate = opt
            .compute_convergence_rate(&params)
            .expect("compute_convergence_rate");

        assert!(
            rate > 0.0,
            "O5 regression: convergence rate was not positive for a \
             genuinely decreasing loss sequence (rate={rate})"
        );
    }

    /// O5: the mirror case — loss increasing (diverging) must report a
    /// negative rate.
    #[test]
    fn negative_rate_when_loss_is_increasing() {
        let mut opt = make_optimizer();
        for loss in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0] {
            opt.performance_tracker
                .add_performance(make_snapshot(loss))
                .expect("add_performance");
        }

        let params = Array::<f64, Ix1>::zeros(3);
        let rate = opt
            .compute_convergence_rate(&params)
            .expect("compute_convergence_rate");

        assert!(
            rate < 0.0,
            "O5 regression: convergence rate was not negative for a \
             genuinely increasing (diverging) loss sequence (rate={rate})"
        );
    }
    /// O2: `compute_feature_median` used to `return Ok(features.clone())`, so
    /// `adapt_for_anomaly` compared every value against itself, `diff` was
    /// always exactly zero, and no value was ever clipped. This asserts the
    /// median is a real order statistic over the rolling window and that an
    /// extreme value genuinely gets pulled towards it.
    #[test]
    fn feature_median_is_a_real_order_statistic_over_the_window() {
        let mut opt = make_optimizer();

        // A window whose per-coordinate medians are known exactly:
        // coordinate 0 -> {1,2,3,4,5} median 3; coordinate 1 -> {10,20,30,40,50} median 30.
        for (a, b) in [
            (1.0, 10.0),
            (2.0, 20.0),
            (3.0, 30.0),
            (4.0, 40.0),
            (5.0, 50.0),
        ] {
            opt.remember_features(&Array1::from_vec(vec![a, b]));
        }
        assert_eq!(opt.feature_window_len(), 5);

        let probe = Array1::from_vec(vec![1000.0, -1000.0]);
        let median = opt
            .compute_feature_median(&probe)
            .expect("compute_feature_median");

        assert_eq!(
            median.to_vec(),
            vec![3.0, 30.0],
            "O2 regression: the median echoed its input instead of summarising \
             the rolling window"
        );
        assert_ne!(
            median.to_vec(),
            probe.to_vec(),
            "O2 regression: compute_feature_median returned its argument"
        );
    }

    /// O2: the consequence — `adapt_for_anomaly` must actually clip an extreme
    /// coordinate towards the window median. Against the old code the value came
    /// back untouched.
    #[test]
    fn adapt_for_anomaly_clips_extreme_values_towards_the_median() {
        let mut opt = make_optimizer();
        for value in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9] {
            opt.remember_features(&Array1::from_vec(vec![value]));
        }

        let outlier = make_point(vec![10_000.0]);
        let original = outlier.features[0];
        let adapted = opt.adapt_for_anomaly(outlier).expect("adapt_for_anomaly");

        assert!(
            adapted.features[0] < original,
            "O2 regression: an extreme value was not clipped at all \
             (before={original}, after={})",
            adapted.features[0]
        );
        assert!(
            adapted.features[0] < 1_000.0,
            "the clipped value {} is still nowhere near the window median ~10",
            adapted.features[0]
        );
    }

    /// O1: the learning-rate controller was a stub whose rate never moved. A
    /// real AdaGrad-style controller must shrink the rate as squared gradient
    /// norm accumulates, and must report the delta it applied.
    #[test]
    fn learning_rate_controller_responds_to_real_gradients() {
        let config = StreamingConfig::default();
        let mut controller =
            AdaptiveLearningRateController::<f64>::new(&config).expect("controller");

        let initial = controller.current_rate();
        assert!(controller.last_change().is_none());
        assert_eq!(controller.accumulated_squared_gradient_norm(), 0.0);

        let large_gradient = Array1::from_vec(vec![5.0, -5.0, 5.0]);
        let after_first = controller.update_learning_rate(&large_gradient);
        assert!(
            after_first < initial,
            "O1 regression: a large gradient did not shrink the rate \
             ({initial} -> {after_first})"
        );
        assert!(
            controller.accumulated_squared_gradient_norm() > 0.0,
            "O1 regression: the gradient was ignored entirely"
        );
        assert!(
            controller.last_change().is_some(),
            "O1 regression: last_change is still hard-coded to None"
        );

        let after_second = controller.update_learning_rate(&large_gradient);
        assert!(
            after_second < after_first,
            "the accumulator must keep shrinking the rate ({after_first} -> {after_second})"
        );
        assert!(
            after_second >= config.learning_rate_config.min_rate,
            "the rate must respect the configured floor"
        );
    }

    /// O1: `compute_adaptation` used to ignore its argument and echo the base
    /// rate. It must now move the rate in opposite directions for a falling and
    /// a rising loss trend.
    #[test]
    fn learning_rate_adaptation_follows_the_loss_trend() {
        let config = StreamingConfig::default();
        let controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
        let current = controller.current_rate();

        // Oldest -> newest, loss falling: the rate should grow.
        let improving = [10.0, 8.0, 6.0, 4.0, 2.0];
        let grown = controller.compute_adaptation(&improving);
        assert!(
            grown > current,
            "O1 regression: a falling loss did not grow the rate \
             ({current} -> {grown})"
        );

        // Oldest -> newest, loss rising: the rate should shrink.
        let worsening = [2.0, 4.0, 6.0, 8.0, 10.0];
        let shrunk = controller.compute_adaptation(&worsening);
        assert!(
            shrunk < current,
            "O1 regression: a rising loss did not shrink the rate \
             ({current} -> {shrunk})"
        );
        assert!(
            (grown - shrunk).abs() > 1e-12,
            "the two trends must not produce the same rate"
        );

        // Too little data to fit a trend: unchanged, not fabricated.
        assert_eq!(controller.compute_adaptation(&[]), current);
        assert_eq!(controller.compute_adaptation(&[3.0]), current);
    }

    /// O1: a non-finite or non-positive proposal must be rejected rather than
    /// silently destroying the optimizer.
    #[test]
    fn learning_rate_controller_rejects_invalid_rates() {
        let config = StreamingConfig::default();
        let mut controller =
            AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
        let before = controller.current_rate();

        controller.apply_adaptation(f64::NAN);
        controller.apply_adaptation(-1.0);
        controller.apply_adaptation(0.0);
        assert_eq!(
            controller.current_rate(),
            before,
            "an invalid proposal must leave the rate untouched"
        );

        controller.apply_adaptation(0.01);
        assert!((controller.current_rate() - 0.01).abs() < 1e-12);
    }

    /// O3: a custom adaptation with no registered handler must be an honest
    /// error rather than a `println!` that makes it look applied.
    #[test]
    fn custom_adaptation_without_a_handler_is_an_error() {
        let mut opt = make_optimizer();
        let adaptation = Adaptation {
            adaptation_type: AdaptationType::Custom("nonexistent".to_string()),
            magnitude: 0.1,
            target_component: "somewhere".to_string(),
            parameters: HashMap::new(),
            priority: AdaptationPriority::Normal,
            timestamp: Instant::now(),
        };
        let result = opt.apply_adaptations(std::slice::from_ref(&adaptation));
        assert!(
            result.is_err(),
            "an unhandled custom adaptation must not report success"
        );
    }

    /// A3: the anomaly detector's context signals must be real values published
    /// by the optimizer, not the old hard-coded 0.8/0.7, 0.6/0.5, 0.1.
    #[test]
    fn anomaly_context_signals_come_from_real_state() {
        let mut opt = make_optimizer();
        opt.performance_tracker
            .add_performance(make_snapshot(7.25))
            .expect("add_performance");

        opt.publish_anomaly_context_signals()
            .expect("publish_anomaly_context_signals");

        // Drive one detection so a context is actually built.
        let point = make_point(vec![1.0, 2.0, 3.0]);
        opt.anomaly_detector
            .detect_anomaly(&point)
            .expect("detect_anomaly");

        let context = opt
            .anomaly_detector
            .build_context_for_test(&point)
            .expect("context");
        assert_eq!(
            context.performance_metrics.first().copied(),
            Some(7.25),
            "the published loss must reach the anomaly context verbatim"
        );
        assert!(
            !context.drift_indicators.is_empty(),
            "drift indicators must be published"
        );
        assert_ne!(
            context.performance_metrics,
            vec![0.8, 0.7],
            "A3 regression: the old hard-coded performance placeholders are back"
        );
    }
    /// `compute_loss` used to score `data_point.features` against the target and
    /// ignore `parameters` entirely, so the reported loss never moved when the
    /// model improved. It must now be a genuine function of the parameters.
    #[test]
    fn loss_is_a_function_of_the_parameters() {
        let opt = make_optimizer();
        let batch = vec![StreamingDataPoint {
            features: Array1::from_vec(vec![1.0, 2.0]),
            target: Some(Array1::from_vec(vec![5.0])),
            timestamp: Instant::now(),
            source_id: None,
            quality_score: 1.0,
            metadata: HashMap::new(),
        }];

        // w = (1, 2) gives <w, x> = 1 + 4 = 5, an exact fit.
        let exact = Array::<f64, Ix1>::from_vec(vec![1.0, 2.0]);
        let exact_loss = opt.compute_loss(&batch, &exact).expect("loss");
        assert!(
            exact_loss.abs() < 1e-12,
            "an exact fit must have zero loss, got {exact_loss}"
        );

        // w = (0, 0) predicts 0 against a target of 5, so the loss is 25.
        let wrong = Array::<f64, Ix1>::zeros(2);
        let wrong_loss = opt.compute_loss(&batch, &wrong).expect("loss");
        assert!(
            (wrong_loss - 25.0).abs() < 1e-12,
            "expected loss 25 for a zero model, got {wrong_loss}"
        );
        assert!(
            wrong_loss > exact_loss,
            "the loss must respond to the parameters ({exact_loss} vs {wrong_loss})"
        );
    }

    /// `compute_accuracy` used to count a point as correct whenever its
    /// `quality_score > 0.5` — it measured input data quality, not the model.
    /// A high-quality point that the model predicts badly must now count as
    /// wrong.
    #[test]
    fn accuracy_scores_predictions_not_input_quality() {
        let opt = make_optimizer();
        // Perfect input quality, but the target is unreachable for w = 0.
        let batch = vec![StreamingDataPoint {
            features: Array1::from_vec(vec![1.0, 1.0]),
            target: Some(Array1::from_vec(vec![100.0])),
            timestamp: Instant::now(),
            source_id: None,
            quality_score: 1.0,
            metadata: HashMap::new(),
        }];

        let zero_model = Array::<f64, Ix1>::zeros(2);
        let bad = opt.compute_accuracy(&batch, &zero_model).expect("accuracy");
        assert_eq!(
            bad, 0.0,
            "a quality-1.0 point the model gets completely wrong must not count \
             as correct"
        );

        // w = (50, 50) predicts exactly 100.
        let good_model = Array::<f64, Ix1>::from_vec(vec![50.0, 50.0]);
        let good = opt.compute_accuracy(&batch, &good_model).expect("accuracy");
        assert_eq!(good, 1.0, "an exact prediction must count as correct");
    }

    /// An unlabelled batch cannot be scored, so accuracy must be `0`, not the
    /// perfect `1.0` the previous implementation returned.
    #[test]
    fn unlabelled_batches_do_not_report_perfect_accuracy() {
        let opt = make_optimizer();
        let batch = vec![make_point(vec![1.0, 2.0])];
        let parameters = Array::<f64, Ix1>::zeros(2);
        assert_eq!(
            opt.compute_accuracy(&batch, &parameters).expect("accuracy"),
            0.0
        );
    }
}
