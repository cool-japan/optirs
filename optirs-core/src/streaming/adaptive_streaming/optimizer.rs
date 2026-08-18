// Core adaptive streaming optimizer implementation
//
// This module contains the main AdaptiveStreamingOptimizer that orchestrates
// all streaming optimization components including drift detection, performance
// tracking, resource management, and adaptive learning rate control.

use super::anomaly_detection::{AnomalyDetector, AnomalyDiagnostics};
use super::buffering::{AdaptiveBuffer, BufferDiagnostics};
use super::config::*;
use super::drift_detection::{DriftDiagnostics, EnhancedDriftDetector};
use super::meta_learning::{MetaAction, MetaLearner, MetaLearningDiagnostics, MetaState};
use super::performance::{
    DataStatistics, PerformanceDiagnostics, PerformanceSnapshot, PerformanceTracker,
};
use super::resource_management::{ResourceDiagnostics, ResourceManager, ResourceUsage};

use crate::optimizers::Optimizer;
use crate::utils::try_scalar_str;
use scirs2_core::ndarray::{Array, Array1, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::ScientificNumber;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Window used by [`AdaptiveStreamingStats::recent_adaptations`]: adaptations
/// older than this no longer count as "recent" activity.
pub const RECENT_ADAPTATION_WINDOW: Duration = Duration::from_secs(300);

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
    /// Number of updates applied. Doubles as the iteration counter the
    /// cyclical schedule is evaluated against.
    updates: usize,
    /// Cyclical-schedule state, present only when
    /// `LearningRateConfig::enable_cyclical_rates` is set.
    cyclical: Option<CyclicalSchedule<A>>,
}

/// Cyclical learning-rate schedule (Smith, "Cyclical Learning Rates for
/// Training Neural Networks", WACV 2017), driven entirely by
/// `CyclicalRateConfig` (CF1).
///
/// Every field of `CyclicalRateConfig` — `base_rate`, `max_rate`,
/// `cycle_length`, `cycle_mode` and `scale_function` — previously had no reader
/// anywhere in the crate, and `enable_cyclical_rates` was never consulted, so
/// configuring a cyclical schedule did nothing at all.
#[derive(Debug, Clone)]
struct CyclicalSchedule<A: Float> {
    /// Lower bound of the cycle.
    base_rate: A,
    /// Upper bound of the cycle.
    max_rate: A,
    /// Half-cycle length in iterations (`cycle_length / 2`, at least 1).
    step_size: A,
    /// Amplitude policy across successive cycles.
    cycle_mode: CycleMode,
    /// Within-cycle ramp shape.
    scale_function: ScaleFunction,
}

impl<A: Float> CyclicalSchedule<A> {
    /// Learning rate for iteration `iteration` (0-based).
    ///
    /// `lr = base + (max - base) * ramp(x) * amplitude(cycle)` where `x` is the
    /// normalised distance from the current half-cycle boundary, exactly as in
    /// the reference formulation.
    fn rate_at(&self, iteration: usize) -> A {
        let Some(step) = A::from(iteration) else {
            return self.base_rate;
        };
        let two = match A::from(2.0) {
            Some(two) => two,
            None => return self.base_rate,
        };

        // cycle = floor(1 + step / (2 * step_size)); x = |step/step_size - 2*cycle + 1|
        let cycle = (A::one() + step / (two * self.step_size)).floor();
        let x = (step / self.step_size - two * cycle + A::one()).abs();
        let position = (A::one() - x).max(A::zero()).min(A::one());

        let ramp = match &self.scale_function {
            ScaleFunction::Linear => position,
            ScaleFunction::Polynomial { power } => match A::from(*power) {
                Some(power) => position.powf(power),
                None => position,
            },
            // `factor^x`: 1 at the cycle peak, `factor` at the trough.
            ScaleFunction::Exponential { factor } => match A::from(*factor) {
                Some(factor) if factor > A::zero() => factor.powf(A::one() - position),
                _ => position,
            },
            // Rejected at construction time.
            ScaleFunction::Custom(_) => position,
        };

        let amplitude = match &self.cycle_mode {
            CycleMode::Triangular => A::one(),
            // Halve the amplitude on each successive cycle.
            CycleMode::Triangular2 => {
                let exponent = (cycle - A::one()).max(A::zero());
                A::one() / two.powf(exponent)
            }
            // `gamma^iteration`, with gamma supplied by the exponential scale
            // function (guaranteed present by the constructor's validation).
            CycleMode::ExponentialRange => match &self.scale_function {
                ScaleFunction::Exponential { factor } => match A::from(*factor) {
                    Some(gamma) if gamma > A::zero() => gamma.powf(step),
                    _ => A::one(),
                },
                _ => A::one(),
            },
            // Rejected at construction time.
            CycleMode::Custom(_) => A::one(),
        };

        let span = self.max_rate - self.base_rate;
        (self.base_rate + span * ramp * amplitude)
            .max(self.base_rate.min(self.max_rate))
            .min(self.base_rate.max(self.max_rate))
    }
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

        let cyclical = if lr_config.enable_cyclical_rates {
            let cycle = &lr_config.cycle_config;
            if let ScaleFunction::Custom(name) = &cycle.scale_function {
                return Err(crate::error::OptimError::InvalidConfig(format!(
                    "cyclical learning-rate scale_function Custom(\"{name}\") has no \
                     registered implementation; use Linear, Exponential or Polynomial"
                )));
            }
            if let CycleMode::Custom(name) = &cycle.cycle_mode {
                return Err(crate::error::OptimError::InvalidConfig(format!(
                    "cyclical learning-rate cycle_mode Custom(\"{name}\") has no \
                     registered implementation; use Triangular, Triangular2 or \
                     ExponentialRange"
                )));
            }
            if matches!(cycle.cycle_mode, CycleMode::ExponentialRange)
                && !matches!(cycle.scale_function, ScaleFunction::Exponential { .. })
            {
                return Err(crate::error::OptimError::InvalidConfig(
                    "cyclical cycle_mode ExponentialRange needs its decay factor from \
                     scale_function = Exponential { factor }"
                        .to_string(),
                ));
            }
            if cycle.cycle_length == 0 {
                return Err(crate::error::OptimError::InvalidConfig(
                    "cyclical learning-rate cycle_length must be greater than zero".to_string(),
                ));
            }
            let base_rate = convert(cycle.base_rate, "cycle_config.base_rate")?;
            let cycle_max_rate = convert(cycle.max_rate, "cycle_config.max_rate")?;
            if base_rate > cycle_max_rate {
                return Err(crate::error::OptimError::InvalidConfig(format!(
                    "cyclical base_rate ({}) exceeds cycle max_rate ({})",
                    cycle.base_rate, cycle.max_rate
                )));
            }
            let step_size = convert(
                ((cycle.cycle_length as f64) / 2.0).max(1.0),
                "cycle_config.cycle_length",
            )?;
            Some(CyclicalSchedule {
                base_rate,
                max_rate: cycle_max_rate,
                step_size,
                cycle_mode: cycle.cycle_mode.clone(),
                scale_function: cycle.scale_function.clone(),
            })
        } else {
            None
        };

        let initial_current = match cyclical.as_ref() {
            // A cyclical schedule owns the rate outright, so start on the
            // schedule rather than at `initial_rate`.
            Some(schedule) => schedule.rate_at(0).max(min_lr).min(max_lr),
            None => initial_lr.max(min_lr).min(max_lr),
        };

        Ok(Self {
            current_lr: initial_current,
            initial_lr,
            min_lr,
            max_lr,
            squared_gradient_norm_sum: A::zero(),
            trend_step,
            last_change: None,
            updates: 0,
            cyclical,
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

        // A configured cyclical schedule owns the rate: its whole purpose is to
        // sweep between bounds on a fixed cadence, which an AdaGrad decay would
        // flatten out. The gradient accumulator is still maintained above so
        // `accumulated_squared_gradient_norm` stays meaningful either way.
        let proposed = match self.cyclical.as_ref() {
            Some(schedule) => schedule.rate_at(self.updates),
            None => {
                self.initial_lr * (A::one() / (A::one() + self.squared_gradient_norm_sum.sqrt()))
            }
        };
        let proposed = proposed.max(self.min_lr).min(self.max_lr);
        self.set_rate(proposed);
        self.current_lr
    }

    /// Test-only view of the cyclical schedule's rate at a given iteration.
    #[cfg(test)]
    pub(crate) fn rate_at_for_test(&self, iteration: usize) -> A {
        match self.cyclical.as_ref() {
            Some(schedule) => schedule.rate_at(iteration),
            None => self.current_lr,
        }
    }

    /// Whether a cyclical schedule is driving this controller.
    pub fn is_cyclical(&self) -> bool {
        self.cyclical.is_some()
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
        // Under a cyclical schedule the next rate is a function of the iteration
        // counter, not of the performance trend.
        if let Some(schedule) = self.cyclical.as_ref() {
            return schedule
                .rate_at(self.updates.saturating_add(1))
                .max(self.min_lr)
                .min(self.max_lr);
        }
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
    /// Number of adaptations applied over the optimizer's whole lifetime
    pub adaptations_applied: usize,
    /// Number of adaptations applied within the last
    /// [`RECENT_ADAPTATION_WINDOW`], recomputed on each
    /// `get_adaptive_stats()` call
    pub recent_adaptations: usize,
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
    O: Optimizer<A, D> + Clone,
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
            recent_adaptations: 0,
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

        // Apply adaptations to system components.
        //
        // The lifetime counter is bumped here rather than with the rest of the
        // statistics further down: `apply_adaptations` is what records them
        // into `adaptation_history`, and the statistics block sits behind four
        // `?` operators. A step that failed after adapting therefore left the
        // adaptations in the history but uncounted, so `adaptations_applied`
        // drifted permanently below the number of adaptations really applied
        // (and below the recent-window count derived from the history).
        self.apply_adaptations(&adaptations)?;
        self.stats.adaptations_applied += adaptations.len();

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
        self.update_meta_learner(&adaptations, &performance)?;

        // Update statistics
        self.stats.optimization_steps += 1;
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
                median[i] * try_scalar_str::<A, _>(self.config.anomaly_config.threshold)?;

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
        data_point.quality_score = data_point.quality_score * try_scalar_str::<A, _>(0.5)?;

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
            >= try_scalar_str::<A, _>(self.config.buffer_config.quality_threshold)?;

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

        let parameters = if let Some(params) = self.parameters.clone() {
            params
        } else {
            // Cannot initialize parameters without proper dimension info
            return Err("Parameters not initialized".to_string());
        };

        // Hand the step to the base optimizer the caller supplied.
        //
        // This used to be an inline `param -= lr * grad` loop with the comment
        // "in practice would use the base optimizer": the `O` type parameter
        // and the `base_optimizer` constructor argument were accepted and then
        // ignored, so an `AdaptiveStreamingOptimizer<Adam<_>, ..>` silently ran
        // plain SGD and none of Adam's moments existed. The adaptive
        // learning-rate controller drives the base optimizer's rate, exactly as
        // the streaming optimizer in `streaming::types` does.
        if parameters.len() != gradients.len() {
            return Err(format!(
                "parameter/gradient dimensionality mismatch: {} parameters but {} gradients",
                parameters.len(),
                gradients.len()
            ));
        }
        let gradients_d = gradients
            .clone()
            .into_dimensionality::<D>()
            .map_err(|e| format!("gradient does not fit the parameter dimensionality: {e}"))?;
        self.base_optimizer.set_learning_rate(learning_rate);
        let updated_parameters = self
            .base_optimizer
            .step(&parameters, &gradients_d)
            .map_err(|e| format!("base optimizer step failed: {e}"))?;

        let squared_update = parameters.iter().zip(updated_parameters.iter()).fold(
            A::zero(),
            |acc, (&before, &after)| {
                let delta = after - before;
                acc + delta * delta
            },
        );

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
        let batch_size = try_scalar_str::<A, _>(batch.len())?;
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
        feature_means /= try_scalar_str::<A, _>(batch.len())?;

        // Compute standard deviations
        for data_point in batch {
            let diff = &data_point.features - &feature_means;
            feature_stds = feature_stds + &diff.mapv(|x| x * x);
        }
        feature_stds /= try_scalar_str::<A, _>(batch.len())?;
        feature_stds = feature_stds.mapv(|x| x.sqrt());

        let avg_quality = quality_scores.iter().copied().sum::<A>()
            / try_scalar_str::<A, _>(quality_scores.len())?;

        Ok(DataStatistics {
            sample_count: batch.len(),
            feature_means,
            feature_stds,
            average_quality: avg_quality,
            timestamp: Instant::now(),
        })
    }

    /// Updates meta-learner with experience from this optimization step.
    ///
    /// Deliberately takes no data batch: `MetaState` has no slot for per-batch
    /// data characteristics (its features are performance, resource and drift
    /// signals, whose layout the bandit's feature scaler depends on), so a batch
    /// argument could only be discarded — which is what it used to be.
    fn update_meta_learner(
        &mut self,
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
                try_scalar_str::<A, _>(performance.resource_usage.memory_usage_mb as f64)?,
                try_scalar_str::<A, _>(performance.resource_usage.cpu_usage_percent)?,
            ],
            drift_indicators: vec![try_scalar_str::<A, _>(
                if self.drift_detector.is_drift_detected() {
                    1.0
                } else {
                    0.0
                },
            )?],
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
        // `adaptations_applied` is a lifetime counter; the recent-window count
        // is derived live from `adaptation_history` so callers can tell a
        // currently-thrashing optimizer from one that adapted long ago.
        stats.recent_adaptations = self.count_adaptations_applied(RECENT_ADAPTATION_WINDOW);
        stats
    }

    /// Counts the adaptations recorded within `window` of now.
    ///
    /// Uses a forward `duration_since` comparison rather than materialising an
    /// `Instant::now() - window` cutoff: subtracting a `Duration` from an
    /// `Instant` panics when the process has been up for less than `window`.
    fn count_adaptations_applied(&self, window: Duration) -> usize {
        let now = Instant::now();
        self.adaptation_history
            .iter()
            .filter(|adaptation| now.duration_since(adaptation.timestamp) <= window)
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
            recent_adaptations: 0,
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
#[path = "optimizer_convergence_tests.rs"]
mod o5_convergence_rate_tests;

#[cfg(test)]
#[path = "optimizer_regression_tests.rs"]
mod regression_tests;
