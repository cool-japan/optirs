// Performance tracking and prediction for streaming optimization
//
// This module provides comprehensive performance monitoring, trend analysis,
// and prediction capabilities for streaming optimization scenarios, including
// real-time metrics collection, statistical analysis, and predictive modeling.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationType};
use super::resource_management::ResourceUsage;

use crate::utils::{scalar_or, try_scalar_str};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::iter::Sum;
use std::time::{Duration, Instant};

/// Performance snapshot representing metrics at a specific point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot<A: Float + Send + Sync> {
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Wall-clock time the optimization step that produced this snapshot took.
    ///
    /// This is a measured duration, distinct from `timestamp.elapsed()` (which
    /// is the snapshot's *age*). Consumers reasoning about processing cost must
    /// read this field.
    pub processing_duration: Duration,
    /// Primary loss metric
    pub loss: A,
    /// Accuracy metric (if applicable)
    pub accuracy: Option<A>,
    /// Convergence rate
    pub convergence_rate: Option<A>,
    /// Gradient norm
    pub gradient_norm: Option<A>,
    /// Parameter update magnitude
    pub parameter_update_magnitude: Option<A>,
    /// Data quality statistics
    pub data_statistics: DataStatistics<A>,
    /// Resource usage at snapshot time
    pub resource_usage: ResourceUsage,
    /// Custom performance metrics
    pub custom_metrics: HashMap<String, A>,
}

/// Data quality and distribution statistics
#[derive(Debug, Clone)]
pub struct DataStatistics<A: Float + Send + Sync> {
    /// Number of samples in this batch
    pub sample_count: usize,
    /// Feature-wise means
    pub feature_means: scirs2_core::ndarray::Array1<A>,
    /// Feature-wise standard deviations
    pub feature_stds: scirs2_core::ndarray::Array1<A>,
    /// Average data quality score
    pub average_quality: A,
    /// Timestamp of statistics computation
    pub timestamp: Instant,
}

impl<A: Float + Send + Sync> Default for DataStatistics<A> {
    fn default() -> Self {
        Self {
            sample_count: 0,
            feature_means: scirs2_core::ndarray::Array1::zeros(0),
            feature_stds: scirs2_core::ndarray::Array1::zeros(0),
            average_quality: A::zero(),
            timestamp: Instant::now(),
        }
    }
}

/// Performance metrics for tracking and analysis
#[derive(Debug, Clone)]
pub enum PerformanceMetric<A: Float + Send + Sync> {
    /// Loss function value
    Loss(A),
    /// Classification/regression accuracy
    Accuracy(A),
    /// Rate of convergence
    ConvergenceRate(A),
    /// Gradient magnitude
    GradientNorm(A),
    /// Learning rate effectiveness
    LearningRateEffectiveness(A),
    /// Resource utilization efficiency
    ResourceEfficiency(A),
    /// Data quality score
    DataQuality(A),
    /// Custom metric with name and value
    Custom(String, A),
}

/// Context information for performance evaluation
#[derive(Debug, Clone)]
pub struct PerformanceContext<A: Float + Send + Sync> {
    /// Current learning rate
    pub learning_rate: A,
    /// Current batch size
    pub batch_size: usize,
    /// Current buffer size
    pub buffer_size: usize,
    /// Recent drift detection status
    pub drift_detected: bool,
    /// Resource constraints
    pub resource_constraints: ResourceUsage,
    /// Time since last adaptation
    pub time_since_adaptation: Duration,
}

/// Performance tracker for streaming optimization
pub struct PerformanceTracker<A: Float + Send + Sync + std::iter::Sum> {
    /// Configuration for performance tracking
    config: PerformanceConfig,
    /// Performance history
    performance_history: VecDeque<PerformanceSnapshot<A>>,
    /// Trend analyzer
    trend_analyzer: PerformanceTrendAnalyzer<A>,
    /// Performance predictor
    predictor: PerformancePredictor<A>,
    /// Performance baseline
    baseline: Option<PerformanceSnapshot<A>>,
    /// Current performance context
    current_context: Option<PerformanceContext<A>>,
    /// Performance improvement tracker
    improvement_tracker: PerformanceImprovementTracker<A>,
    /// Anomaly detector for performance
    performance_anomaly_detector: PerformanceAnomalyDetector<A>,
    /// Number of snapshots accepted since the baseline was last refreshed,
    /// driving `PerformanceConfig::baseline_update_frequency`.
    snapshots_since_baseline: usize,
}

/// Trend analysis for performance metrics
pub struct PerformanceTrendAnalyzer<A: Float + Send + Sync> {
    /// Window size for trend analysis
    window_size: usize,
    /// Current trends for different metrics
    trends: HashMap<String, TrendData<A>>,
}

/// Trend data for a specific metric
#[derive(Debug, Clone)]
pub struct TrendData<A: Float + Send + Sync> {
    /// Linear trend slope
    pub slope: A,
    /// Trend correlation coefficient
    pub correlation: A,
    /// Trend volatility
    pub volatility: A,
    /// Trend confidence
    pub confidence: A,
    /// Recent values used for trend calculation
    pub recent_values: VecDeque<A>,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Methods for trend calculation
#[derive(Debug, Clone)]
pub enum TrendMethod {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage { window: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
    /// Seasonal decomposition
    SeasonalDecomposition,
}

/// Performance predictor using various forecasting methods
pub struct PerformancePredictor<A: Float + Send + Sync> {
    /// Prediction methods to use
    prediction_methods: Vec<PredictionMethod>,
    /// Historical predictions for accuracy tracking
    prediction_history: VecDeque<PredictionResult<A>>,
    /// Model accuracy scores, keyed by the per-method label used in
    /// `ensemble_weights`.
    model_accuracies: HashMap<String, A>,
    /// Ensemble weights for combining predictions
    ensemble_weights: HashMap<String, A>,
    /// Per-method forecasts still awaiting their ground-truth observation.
    ///
    /// `PredictionResult` carries only the combined ensemble label, so scoring
    /// individual methods (which is what the ensemble weights need) requires
    /// keeping each method's own forecast until the actual value arrives.
    pending_method_forecasts: VecDeque<MethodForecast<A>>,
    /// Number of snapshots observed, used as the clock a forecast horizon is
    /// measured against (`steps_ahead` counts snapshots, not seconds).
    snapshots_since_start: usize,
}

/// A single method's forecast, held until its horizon elapses.
#[derive(Debug, Clone)]
struct MethodForecast<A: Float + Send + Sync> {
    /// Per-method label matching the `model_accuracies` key.
    label: String,
    /// Forecast value.
    predicted_value: A,
    /// Number of steps ahead the forecast was made for.
    steps_ahead: usize,
    /// Snapshot index at which the forecast was issued.
    issued_at_index: usize,
}

/// Prediction methods for performance forecasting
#[derive(Debug, Clone)]
pub enum PredictionMethod {
    /// Linear extrapolation
    Linear,
    /// Exponential smoothing
    Exponential { alpha: f64, beta: f64 },
    /// ARIMA model
    ARIMA { p: usize, d: usize, q: usize },
    /// Neural network
    NeuralNetwork { hidden_layers: Vec<usize> },
    /// Ensemble of multiple methods
    Ensemble,
}

/// Result of performance prediction
#[derive(Debug, Clone)]
pub struct PredictionResult<A: Float + Send + Sync> {
    /// Predicted metric value
    pub predicted_value: A,
    /// Prediction confidence interval
    pub confidence_interval: (A, A),
    /// Prediction method used
    pub method: String,
    /// Steps ahead predicted
    pub steps_ahead: usize,
    /// Snapshot index at which the prediction was issued. `steps_ahead` counts
    /// snapshots, so this is the clock its horizon is measured against.
    pub issued_at_index: usize,
    /// Prediction timestamp
    pub timestamp: Instant,
    /// Actual value (filled in later for accuracy assessment)
    pub actual_value: Option<A>,
}

/// Performance improvement tracking
pub struct PerformanceImprovementTracker<A: Float + Send + Sync> {
    /// Baseline performance metrics
    baseline_metrics: HashMap<String, A>,
    /// Current improvement rates
    improvement_rates: HashMap<String, A>,
    /// Improvement history
    improvement_history: VecDeque<ImprovementEvent<A>>,
    /// Plateau detection
    plateau_detector: PlateauDetector<A>,
}

/// Performance improvement event
#[derive(Debug, Clone)]
pub struct ImprovementEvent<A: Float + Send + Sync> {
    /// Event timestamp
    pub timestamp: Instant,
    /// Metric that improved
    pub metric_name: String,
    /// Improvement magnitude
    pub improvement: A,
    /// Improvement rate (per unit time)
    pub improvement_rate: A,
    /// Context when improvement occurred
    pub context: String,
}

/// Plateau detection for performance metrics
pub struct PlateauDetector<A: Float + Send + Sync> {
    /// Window size for plateau detection
    window_size: usize,
    /// Plateau threshold (minimum change for non-plateau)
    plateau_threshold: A,
    /// Recent performance values
    recent_values: VecDeque<A>,
    /// Current plateau status
    is_plateau: bool,
    /// Plateau duration, measured from `plateau_started`.
    plateau_duration: Duration,
    /// Instant the current plateau began, if any.
    plateau_started: Option<Instant>,
    /// Last significant change timestamp
    last_significant_change: Option<Instant>,
}

/// Anomaly detection for performance metrics
pub struct PerformanceAnomalyDetector<A: Float + Send + Sync> {
    /// Anomaly detection threshold (standard deviations)
    threshold: A,
    /// Historical statistics for anomaly detection
    historical_stats: HashMap<String, MetricStatistics<A>>,
    /// Recent anomalies detected
    recent_anomalies: VecDeque<PerformanceAnomaly<A>>,
    /// Adaptive threshold adjustment
    adaptive_threshold: bool,
}

/// Statistics for a performance metric
#[derive(Debug, Clone)]
pub struct MetricStatistics<A: Float + Send + Sync> {
    /// Running mean
    pub mean: A,
    /// Running variance
    pub variance: A,
    /// Minimum observed value
    pub min_value: A,
    /// Maximum observed value
    pub max_value: A,
    /// Number of observations
    pub count: usize,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Performance anomaly event
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly<A: Float + Send + Sync> {
    /// Anomaly timestamp
    pub timestamp: Instant,
    /// Affected metric
    pub metric_name: String,
    /// Observed value
    pub observed_value: A,
    /// Expected value range
    pub expected_range: (A, A),
    /// Anomaly severity
    pub severity: AnomalySeverity,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
}

/// Severity levels for performance anomalies
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Minor anomaly
    Minor,
    /// Moderate anomaly requiring attention
    Moderate,
    /// Major anomaly requiring intervention
    Major,
    /// Critical anomaly requiring immediate action
    Critical,
}

/// Types of performance anomalies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    /// Value significantly higher than expected
    High,
    /// Value significantly lower than expected
    Low,
    /// Sudden change in trend
    TrendChange,
    /// Unexpected oscillation
    Oscillation,
    /// Performance degradation
    Degradation,
    /// Performance plateau
    Plateau,
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync + std::fmt::Debug>
    PerformanceTracker<A>
{
    /// Creates a new performance tracker
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let performance_config = config.performance_config.clone();

        let trend_analyzer = PerformanceTrendAnalyzer::new(performance_config.trend_window_size);
        let predictor = PerformancePredictor::new();
        let improvement_tracker = PerformanceImprovementTracker::new();
        let performance_anomaly_detector = PerformanceAnomalyDetector::new(2.0); // 2 sigma threshold

        Ok(Self {
            config: performance_config,
            performance_history: VecDeque::with_capacity(1000),
            trend_analyzer,
            predictor,
            baseline: None,
            snapshots_since_baseline: 0,
            current_context: None,
            improvement_tracker,
            performance_anomaly_detector,
        })
    }

    /// Test-only view of the current baseline's loss.
    #[cfg(test)]
    pub(crate) fn baseline_loss_for_test(&self) -> Option<A> {
        self.baseline.as_ref().map(|snapshot| snapshot.loss)
    }

    /// Adds a new performance snapshot.
    ///
    /// Two `PerformanceConfig` fields that previously had no reader at all now
    /// govern this (CF1):
    ///
    /// * `enable_tracking == false` makes this a no-op, so turning tracking off
    ///   actually stops history, trend, prediction and anomaly work instead of
    ///   silently doing all of it anyway.
    /// * `baseline_update_frequency` re-bases the comparison baseline every N
    ///   accepted snapshots. Before, the baseline was pinned to the very first
    ///   measurement forever, so every "improvement over baseline" figure was
    ///   measured against the start of the run no matter how long it had been
    ///   running.
    pub fn add_performance(&mut self, snapshot: PerformanceSnapshot<A>) -> Result<(), String> {
        if !self.config.enable_tracking {
            return Ok(());
        }

        // Store in history
        if self.performance_history.len() >= self.config.history_size {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(snapshot.clone());

        // Set baseline if this is the first measurement, then refresh it on the
        // configured cadence.
        self.snapshots_since_baseline = self.snapshots_since_baseline.saturating_add(1);
        let refresh_due = self.config.baseline_update_frequency > 0
            && self.snapshots_since_baseline >= self.config.baseline_update_frequency;
        if self.baseline.is_none() || refresh_due {
            self.baseline = Some(snapshot.clone());
            // Only a *refresh* starts a new window. Establishing the very first
            // baseline must not also consume a window slot: the sample that
            // set it is the first sample of the window, so zeroing here made
            // every cadence one sample too long (a frequency of 3 re-based on
            // the 4th sample, then the 7th).
            if refresh_due {
                self.snapshots_since_baseline = 0;
            }
        }

        // Update trend analysis
        if self.config.enable_trend_analysis {
            self.trend_analyzer.update(&snapshot)?;
        }

        // Update improvement tracking
        self.improvement_tracker.update(&snapshot)?;

        // Check for performance anomalies
        let anomalies = self
            .performance_anomaly_detector
            .check_for_anomalies(&snapshot)?;
        if !anomalies.is_empty() {
            // Handle detected anomalies
            self.handle_performance_anomalies(&anomalies)?;
        }

        // Update predictions if enabled
        if self.config.enable_prediction {
            self.predictor.update_with_actual(&snapshot)?;
        }

        Ok(())
    }

    /// Gets recent performance snapshots
    pub fn get_recent_performance(&self, count: usize) -> Vec<PerformanceSnapshot<A>> {
        self.performance_history
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Gets recent loss values for trend analysis
    pub fn get_recent_losses(&self, count: usize) -> Vec<A> {
        self.performance_history
            .iter()
            .rev()
            .take(count)
            .map(|snapshot| snapshot.loss)
            .collect()
    }

    /// Predicts future performance
    pub fn predict_performance(
        &mut self,
        steps_ahead: usize,
    ) -> Result<PredictionResult<A>, String> {
        if !self.config.enable_prediction {
            return Err("Performance prediction is disabled".to_string());
        }

        self.predictor
            .predict(steps_ahead, &self.performance_history)
    }

    /// Gets current performance trends
    pub fn get_performance_trends(&self) -> HashMap<String, TrendData<A>> {
        self.trend_analyzer.get_current_trends()
    }

    /// Computes adaptation for performance thresholds
    pub fn apply_threshold_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::PerformanceThreshold {
            // Adjust anomaly detection thresholds
            let new_threshold = self.performance_anomaly_detector.threshold + adaptation.magnitude;
            self.performance_anomaly_detector
                .update_threshold(new_threshold);
        }
        Ok(())
    }

    /// Handles detected performance anomalies
    fn handle_performance_anomalies(
        &mut self,
        anomalies: &[PerformanceAnomaly<A>],
    ) -> Result<(), String> {
        for anomaly in anomalies {
            match anomaly.severity {
                AnomalySeverity::Critical | AnomalySeverity::Major => {
                    // Log critical anomalies for immediate attention
                    println!("Critical performance anomaly detected: {:?}", anomaly);
                }
                _ => {
                    // Store for analysis
                    self.performance_anomaly_detector
                        .recent_anomalies
                        .push_back(anomaly.clone());
                }
            }
        }
        Ok(())
    }

    /// Resets performance tracking
    pub fn reset(&mut self) -> Result<(), String> {
        self.performance_history.clear();
        self.baseline = None;
        self.snapshots_since_baseline = 0;
        self.current_context = None;
        self.trend_analyzer.reset();
        self.predictor.reset();
        self.improvement_tracker.reset();
        self.performance_anomaly_detector.reset();
        Ok(())
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> PerformanceDiagnostics {
        PerformanceDiagnostics {
            history_size: self.performance_history.len(),
            baseline_set: self.baseline.is_some(),
            trends_available: !self.trend_analyzer.trends.is_empty(),
            anomalies_detected: self.performance_anomaly_detector.recent_anomalies.len(),
            plateau_detected: self.improvement_tracker.plateau_detector.is_plateau,
            prediction_accuracy: self.predictor.get_average_accuracy(),
        }
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> PerformanceTrendAnalyzer<A> {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            trends: HashMap::new(),
        }
    }

    fn update(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        // Update trends for different metrics
        self.update_metric_trend("loss", snapshot.loss)?;

        if let Some(accuracy) = snapshot.accuracy {
            self.update_metric_trend("accuracy", accuracy)?;
        }

        if let Some(convergence) = snapshot.convergence_rate {
            self.update_metric_trend("convergence", convergence)?;
        }

        self.compute_trends()?;
        Ok(())
    }

    fn update_metric_trend(&mut self, metric_name: &str, value: A) -> Result<(), String> {
        let trend_data = self
            .trends
            .entry(metric_name.to_string())
            .or_insert_with(|| TrendData {
                slope: A::zero(),
                correlation: A::zero(),
                volatility: A::zero(),
                confidence: A::zero(),
                recent_values: VecDeque::with_capacity(self.window_size),
                last_update: Instant::now(),
            });

        if trend_data.recent_values.len() >= self.window_size {
            trend_data.recent_values.pop_front();
        }
        trend_data.recent_values.push_back(value);
        trend_data.last_update = Instant::now();

        Ok(())
    }

    fn compute_trends(&mut self) -> Result<(), String> {
        let keys: Vec<_> = self.trends.keys().cloned().collect();

        // Collect all computed values first
        let mut computed_values = Vec::new();
        for metric_name in &keys {
            if let Some(trend_data) = self.trends.get(metric_name) {
                if trend_data.recent_values.len() >= 3 {
                    let values = trend_data.recent_values.clone();
                    let slope = self.compute_slope(&values)?;
                    let correlation = self.compute_correlation(&values)?;
                    let volatility = self.compute_volatility(&values)?;
                    let confidence = self.compute_confidence(&values)?;
                    computed_values.push((
                        metric_name.clone(),
                        slope,
                        correlation,
                        volatility,
                        confidence,
                    ));
                }
            }
        }

        // Now update the trend data
        for (metric_name, slope, correlation, volatility, confidence) in computed_values {
            if let Some(trend_data) = self.trends.get_mut(&metric_name) {
                trend_data.slope = slope;
                trend_data.correlation = correlation;
                trend_data.volatility = volatility;
                trend_data.confidence = confidence;
            }
        }

        Ok(())
    }

    fn compute_slope(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let n = try_scalar_str::<A, _>(values.len())?;
        // Compute sum_x = 1 + 2 + ... + n = n*(n+1)/2
        let sum_x = n * (n + A::one()) / try_scalar_str::<A, _>(2.0)?;
        let sum_y = values.iter().cloned().sum::<A>();
        let sum_xy = values
            .iter()
            .enumerate()
            .map(|(i, &y)| try_scalar_str::<A, _>(i + 1).map(|x| x * y))
            .collect::<Result<Vec<A>, String>>()?
            .into_iter()
            .sum::<A>();
        // Compute sum_x_squared = 1^2 + 2^2 + ... + n^2 = n*(n+1)*(2n+1)/6
        let two = try_scalar_str::<A, _>(2.0)?;
        let six = try_scalar_str::<A, _>(6.0)?;
        let sum_x_squared = n * (n + A::one()) * (two * n + A::one()) / six;

        let denominator = n * sum_x_squared - sum_x * sum_x;
        if denominator == A::zero() {
            return Ok(A::zero());
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Ok(slope)
    }

    fn compute_correlation(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        // Simplified correlation with time index
        let n = values.len();
        let time_values: Vec<A> = (1..=n)
            .map(try_scalar_str::<A, _>)
            .collect::<Result<Vec<A>, String>>()?;
        let value_vec: Vec<A> = values.iter().cloned().collect();

        let mean_time = time_values.iter().cloned().sum::<A>() / try_scalar_str::<A, _>(n)?;
        let mean_value = value_vec.iter().cloned().sum::<A>() / try_scalar_str::<A, _>(n)?;

        let numerator = time_values
            .iter()
            .zip(value_vec.iter())
            .map(|(&t, &v)| (t - mean_time) * (v - mean_value))
            .sum::<A>();

        let time_variance = time_values
            .iter()
            .map(|&t| (t - mean_time) * (t - mean_time))
            .sum::<A>();

        let value_variance = value_vec
            .iter()
            .map(|&v| (v - mean_value) * (v - mean_value))
            .sum::<A>();

        let denominator = (time_variance * value_variance).sqrt();
        if denominator == A::zero() {
            return Ok(A::zero());
        }

        Ok(numerator / denominator)
    }

    fn compute_volatility(&self, values: &VecDeque<A>) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let mean = values.iter().cloned().sum::<A>() / try_scalar_str::<A, _>(values.len())?;
        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<A>()
            / try_scalar_str::<A, _>(values.len())?;

        Ok(variance.sqrt())
    }

    fn compute_confidence(&self, values: &VecDeque<A>) -> Result<A, String> {
        // Simple confidence based on trend consistency
        if values.len() < 3 {
            return Ok(A::zero());
        }

        let slope = self.compute_slope(values)?;
        let correlation = self.compute_correlation(values)?;

        // Confidence increases with stronger correlation and consistent slope direction
        let confidence = correlation.abs() * (A::one() - (slope.abs() / (slope.abs() + A::one())));
        Ok(confidence)
    }

    fn get_current_trends(&self) -> HashMap<String, TrendData<A>> {
        self.trends.clone()
    }

    fn reset(&mut self) {
        self.trends.clear();
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> PerformancePredictor<A> {
    fn new() -> Self {
        Self {
            prediction_methods: vec![
                PredictionMethod::Linear,
                PredictionMethod::Exponential {
                    alpha: 0.3,
                    beta: 0.1,
                },
            ],
            prediction_history: VecDeque::with_capacity(1000),
            model_accuracies: HashMap::new(),
            ensemble_weights: HashMap::new(),
            pending_method_forecasts: VecDeque::with_capacity(1000),
            snapshots_since_start: 0,
        }
    }

    /// Forecasts the loss `steps_ahead` batches into the future.
    ///
    /// P3f/P4f: this used to run only `linear_prediction`, leaving
    /// `exponential_prediction`, the `prediction_methods` list and the
    /// `ensemble_weights` map as permanently-unread dead state. Every
    /// configured method now runs, and their forecasts are combined by weights
    /// derived from each method's *measured* accuracy (see
    /// `update_accuracy_metrics`), so a method that has been predicting badly
    /// loses influence. Methods with no accuracy history yet are weighted
    /// equally.
    fn predict(
        &mut self,
        steps_ahead: usize,
        history: &VecDeque<PerformanceSnapshot<A>>,
    ) -> Result<PredictionResult<A>, String> {
        if history.len() < 2 {
            return Err("Insufficient history for prediction".to_string());
        }

        // Extract loss values for prediction
        let loss_values: Vec<A> = history.iter().map(|s| s.loss).collect();

        // Run every configured method.
        let methods = self.prediction_methods.clone();
        let mut forecasts: Vec<(String, A)> = Vec::with_capacity(methods.len());
        for method in &methods {
            let (label, value) = match method {
                PredictionMethod::Linear => (
                    "linear".to_string(),
                    self.linear_prediction(&loss_values, steps_ahead)?,
                ),
                PredictionMethod::Exponential { alpha, beta } => (
                    format!("exponential(alpha={alpha},beta={beta})"),
                    self.exponential_prediction(&loss_values, steps_ahead, *alpha, *beta)?,
                ),
                other => {
                    // An unimplemented forecaster must not silently contribute a
                    // made-up number to the ensemble.
                    return Err(format!(
                        "prediction method {other:?} has no implementation; remove it from \
                         `prediction_methods` or implement it"
                    ));
                }
            };
            if value.is_finite() {
                forecasts.push((label, value));
            }
        }

        if forecasts.is_empty() {
            return Err("no prediction method produced a finite forecast".to_string());
        }

        // Weight each method by its measured accuracy, refreshing the stored
        // ensemble weights so they are real, inspectable state.
        let mut total_weight = A::zero();
        let mut weighted_sum = A::zero();
        for (label, value) in &forecasts {
            // An unseen method starts at the neutral weight of 1; a measured
            // accuracy in [0, 1] is floored so a method is never fully muted.
            let accuracy = self
                .model_accuracies
                .get(label)
                .copied()
                .unwrap_or_else(A::one);
            let floor = A::from(0.05).ok_or_else(|| "0.05 is not representable".to_string())?;
            let weight = accuracy.max(floor);
            self.ensemble_weights.insert(label.clone(), weight);
            total_weight = total_weight + weight;
            weighted_sum = weighted_sum + weight * *value;
        }
        if total_weight <= A::zero() {
            return Err("ensemble weights sum to zero".to_string());
        }
        let predicted_value = weighted_sum / total_weight;

        // Confidence interval from the real recent volatility of the series,
        // widened by the disagreement between the methods (a genuine measure of
        // model uncertainty).
        let recent_volatility = self.compute_recent_volatility(&loss_values)?;
        let spread = if forecasts.len() > 1 {
            let values: Vec<A> = forecasts.iter().map(|(_, value)| *value).collect();
            let count = A::from(values.len())
                .ok_or_else(|| "sample count not representable".to_string())?;
            let mean = values.iter().fold(A::zero(), |acc, &v| acc + v) / count;
            (values
                .iter()
                .fold(A::zero(), |acc, &v| acc + (v - mean) * (v - mean))
                / count)
                .sqrt()
        } else {
            A::zero()
        };
        let half_width = recent_volatility + spread;
        let confidence_interval = (predicted_value - half_width, predicted_value + half_width);

        let method = if forecasts.len() == 1 {
            forecasts[0].0.clone()
        } else {
            format!(
                "ensemble[{}]",
                forecasts
                    .iter()
                    .map(|(label, _)| label.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };

        let prediction = PredictionResult {
            predicted_value,
            confidence_interval,
            method,
            steps_ahead,
            issued_at_index: self.snapshots_since_start,
            timestamp: Instant::now(),
            actual_value: None,
        };

        // Store prediction for later accuracy assessment
        if self.prediction_history.len() >= 1000 {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back(prediction.clone());

        // Keep each method's own forecast so its accuracy — and therefore its
        // ensemble weight — can be scored against the real outcome.
        let issued_at_index = self.snapshots_since_start;
        for (label, value) in forecasts {
            if self.pending_method_forecasts.len() >= 1000 {
                self.pending_method_forecasts.pop_front();
            }
            self.pending_method_forecasts.push_back(MethodForecast {
                label,
                predicted_value: value,
                steps_ahead,
                issued_at_index,
            });
        }

        Ok(prediction)
    }

    /// Ordinary-least-squares extrapolation over the whole observed series.
    ///
    /// The previous version bound four locals (`x1`, `y1`, `x2`, `y2`) it never
    /// read and then estimated the slope from a two-point finite difference over
    /// `values[n-1] - values[n-3]`, which is dominated by noise on a jittery
    /// stream. A least-squares fit uses every observation.
    fn linear_prediction(&self, values: &[A], steps_ahead: usize) -> Result<A, String> {
        let n = values.len();
        if n == 0 {
            return Err("linear prediction requires at least one observation".to_string());
        }
        if n < 2 {
            return Ok(values[0]);
        }

        let count = A::from(n).ok_or_else(|| "sample count not representable".to_string())?;
        let two = A::from(2.0).ok_or_else(|| "2.0 not representable".to_string())?;
        let six = A::from(6.0).ok_or_else(|| "6.0 not representable".to_string())?;

        // x = 1..=n
        let sum_x = count * (count + A::one()) / two;
        let sum_x_squared = count * (count + A::one()) * (two * count + A::one()) / six;
        let mut sum_y = A::zero();
        let mut sum_xy = A::zero();
        for (index, &value) in values.iter().enumerate() {
            let x = A::from(index + 1).ok_or_else(|| format!("index {index} not representable"))?;
            sum_y = sum_y + value;
            sum_xy = sum_xy + x * value;
        }

        let denominator = count * sum_x_squared - sum_x * sum_x;
        if denominator == A::zero() {
            return Ok(values[n - 1]);
        }
        let slope = (count * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / count;

        let horizon = A::from(n + steps_ahead)
            .ok_or_else(|| "forecast horizon not representable".to_string())?;
        Ok(intercept + slope * horizon)
    }

    /// Holt's linear (double exponential) smoothing.
    ///
    /// `alpha` smooths the level and `beta` the trend; the forecast is
    /// `level + steps_ahead * trend`. The previous implementation used single
    /// exponential smoothing and then multiplied the forecast by `0.99` once per
    /// step ahead — a fabricated "assume slight improvement" factor unrelated to
    /// the data — and ignored `beta` entirely.
    fn exponential_prediction(
        &self,
        values: &[A],
        steps_ahead: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<A, String> {
        if values.is_empty() {
            return Err("exponential prediction requires at least one observation".to_string());
        }
        if values.len() < 2 {
            return Ok(values[0]);
        }

        let alpha = A::from(alpha.clamp(f64::MIN_POSITIVE, 1.0))
            .ok_or_else(|| "alpha not representable".to_string())?;
        let beta =
            A::from(beta.clamp(0.0, 1.0)).ok_or_else(|| "beta not representable".to_string())?;

        let mut level = values[0];
        let mut trend = values[1] - values[0];
        for &value in values.iter().skip(1) {
            let previous_level = level;
            level = alpha * value + (A::one() - alpha) * (previous_level + trend);
            trend = beta * (level - previous_level) + (A::one() - beta) * trend;
        }

        let horizon =
            A::from(steps_ahead).ok_or_else(|| "forecast horizon not representable".to_string())?;
        Ok(level + horizon * trend)
    }

    fn compute_recent_volatility(&self, values: &[A]) -> Result<A, String> {
        if values.len() < 2 {
            return Ok(A::zero());
        }

        let recent_count = values.len().min(10);
        let recent_values = &values[values.len() - recent_count..];

        let mean = recent_values.iter().cloned().sum::<A>() / try_scalar_str::<A, _>(recent_count)?;
        let variance = recent_values
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<A>()
            / try_scalar_str::<A, _>(recent_count)?;

        Ok(variance.sqrt())
    }

    /// Matches issued forecasts against the observed value and updates each
    /// method's measured accuracy.
    ///
    /// P4f: the previous version gated on a fabricated
    /// `Duration::from_secs(steps_ahead * 10)` — "assume 10s per step" — which
    /// bears no relation to the real batch cadence, so on a fast stream no
    /// prediction was ever scored and `model_accuracies` stayed empty forever.
    /// The horizon is now counted in *snapshots*, which is the unit
    /// `steps_ahead` is actually expressed in.
    fn update_with_actual(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        self.snapshots_since_start = self.snapshots_since_start.saturating_add(1);
        let now = self.snapshots_since_start;

        // Score every per-method forecast whose horizon has elapsed, counted in
        // snapshots rather than wall-clock seconds.
        let mut matured: Vec<MethodForecast<A>> = Vec::new();
        self.pending_method_forecasts.retain(|forecast| {
            let due_at = forecast.issued_at_index + forecast.steps_ahead.max(1);
            if now >= due_at {
                matured.push(forecast.clone());
                false
            } else {
                true
            }
        });

        for forecast in &matured {
            let accuracy = Self::accuracy_of(forecast.predicted_value, snapshot.loss)?;
            // Exponentially-weighted so a method's score reflects its recent
            // behaviour rather than only its latest observation.
            let smoothing = A::from(0.3).ok_or_else(|| "0.3 is not representable".to_string())?;
            let updated = match self.model_accuracies.get(&forecast.label) {
                Some(&previous) => smoothing * accuracy + (A::one() - smoothing) * previous,
                None => accuracy,
            };
            self.model_accuracies
                .insert(forecast.label.clone(), updated);
        }

        // Fill in the actual value on the combined predictions whose horizon has
        // also elapsed, so the history is a complete record.
        let mut index = 0usize;
        while index < self.prediction_history.len() {
            if let Some(prediction) = self.prediction_history.get_mut(index) {
                if prediction.actual_value.is_none()
                    && now >= prediction.issued_at_index + prediction.steps_ahead.max(1)
                {
                    prediction.actual_value = Some(snapshot.loss);
                }
            }
            index += 1;
        }

        Ok(())
    }

    /// Accuracy of a single forecast: `1 - min(1, |error| / scale)`, where the
    /// scale is the magnitude of the observed value (floored so a near-zero
    /// observation cannot make the relative error explode).
    fn accuracy_of(predicted: A, actual: A) -> Result<A, String> {
        let epsilon = A::from(1e-8).ok_or_else(|| "1e-8 is not representable".to_string())?;
        let error = (predicted - actual).abs();
        let scale = actual.abs().max(epsilon);
        Ok((A::one() - (error / scale).min(A::one())).max(A::zero()))
    }

    /// Measured accuracy of a single named method, if it has been scored.
    pub fn method_accuracy(&self, label: &str) -> Option<A> {
        self.model_accuracies.get(label).copied()
    }

    /// Current ensemble weights, keyed by method label.
    pub fn ensemble_weights(&self) -> &HashMap<String, A> {
        &self.ensemble_weights
    }

    fn get_average_accuracy(&self) -> f64 {
        if self.model_accuracies.is_empty() {
            return 0.0;
        }

        let sum: A = self.model_accuracies.values().cloned().sum();
        let avg = sum / scalar_or(self.model_accuracies.len(), A::one());
        avg.to_f64().unwrap_or(0.0)
    }

    fn reset(&mut self) {
        self.prediction_history.clear();
        self.model_accuracies.clear();
        self.ensemble_weights.clear();
        self.pending_method_forecasts.clear();
        self.snapshots_since_start = 0;
    }
}

impl<A: Float + Default + Clone + Sum + Send + Sync + Send + Sync>
    PerformanceImprovementTracker<A>
{
    fn new() -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            improvement_rates: HashMap::new(),
            improvement_history: VecDeque::with_capacity(1000),
            plateau_detector: PlateauDetector::new(50, scalar_or(0.01, A::zero())),
        }
    }

    fn update(&mut self, snapshot: &PerformanceSnapshot<A>) -> Result<(), String> {
        // Update baseline if not set
        if self.baseline_metrics.is_empty() {
            self.baseline_metrics
                .insert("loss".to_string(), snapshot.loss);
            if let Some(accuracy) = snapshot.accuracy {
                self.baseline_metrics
                    .insert("accuracy".to_string(), accuracy);
            }
        }

        // Check for improvements
        if let Some(&baseline_loss) = self.baseline_metrics.get("loss") {
            if snapshot.loss < baseline_loss {
                let improvement = baseline_loss - snapshot.loss;

                // Real rate: improvement per second since the previous
                // improvement (or since this snapshot's own step, for the first
                // one). Dividing by a literal `1.0` made `improvement_rate` an
                // exact duplicate of `improvement`, so the field carried no
                // information about how *fast* the optimizer was improving.
                let elapsed = match self.improvement_history.back() {
                    Some(previous) => snapshot
                        .timestamp
                        .saturating_duration_since(previous.timestamp),
                    None => snapshot.processing_duration,
                };
                let seconds = elapsed.as_secs_f64();
                let improvement_rate = if seconds > 0.0 {
                    let divisor = A::from(seconds)
                        .ok_or_else(|| format!("elapsed {seconds}s is not representable"))?;
                    improvement / divisor
                } else {
                    // No measurable interval yet: report the raw improvement
                    // rather than dividing by zero.
                    improvement
                };

                let improvement_event = ImprovementEvent {
                    timestamp: snapshot.timestamp,
                    metric_name: "loss".to_string(),
                    improvement,
                    improvement_rate,
                    context: "optimization_step".to_string(),
                };

                if self.improvement_history.len() >= 1000 {
                    self.improvement_history.pop_front();
                }
                self.improvement_history.push_back(improvement_event);
                self.improvement_rates
                    .insert("loss".to_string(), improvement_rate);

                // Update baseline
                self.baseline_metrics
                    .insert("loss".to_string(), snapshot.loss);
            }
        }

        // Update plateau detector
        self.plateau_detector.update(snapshot.loss);

        Ok(())
    }

    fn reset(&mut self) {
        self.baseline_metrics.clear();
        self.improvement_rates.clear();
        self.improvement_history.clear();
        self.plateau_detector.reset();
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> PlateauDetector<A> {
    fn new(window_size: usize, threshold: A) -> Self {
        Self {
            window_size,
            plateau_threshold: threshold,
            recent_values: VecDeque::with_capacity(window_size),
            is_plateau: false,
            plateau_duration: Duration::ZERO,
            plateau_started: None,
            last_significant_change: None,
        }
    }

    fn update(&mut self, value: A) {
        if self.recent_values.len() >= self.window_size {
            self.recent_values.pop_front();
        }
        self.recent_values.push_back(value);

        if self.recent_values.len() >= self.window_size {
            self.detect_plateau();
        }
    }

    /// Detects whether the tracked metric has flattened out.
    ///
    /// P1f: the range used to be computed with `fold(A::zero(), A::max)` and
    /// `fold(A::zero(), A::min)`, i.e. seeded at zero. For any all-positive
    /// metric (loss, latency, error rate — essentially all of them) the seeded
    /// minimum stayed at `0`, so `range == max_val` and the detector only ever
    /// fired when the *largest observed value* fell below the plateau
    /// threshold. The seed is now the first observation, which is the only
    /// correct identity for a min/max reduction.
    ///
    /// P2f: `plateau_duration` was advanced by a hard-coded
    /// `Duration::from_secs(1)` per update, so it reported "one second per
    /// sample" regardless of how fast or slow samples actually arrived. It is
    /// now measured from the real `Instant` at which the plateau began.
    fn detect_plateau(&mut self) {
        let mut values = self.recent_values.iter().copied();
        let Some(first) = values.next() else {
            return;
        };
        if self.recent_values.len() < 2 {
            return;
        }

        let mut min_val = first;
        let mut max_val = first;
        for value in values {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
        }
        let range = max_val - min_val;

        let was_plateau = self.is_plateau;
        self.is_plateau = range < self.plateau_threshold;

        if self.is_plateau {
            if !was_plateau {
                // Plateau just began: stamp the real clock.
                self.plateau_started = Some(Instant::now());
                self.plateau_duration = Duration::ZERO;
            } else if let Some(started) = self.plateau_started {
                self.plateau_duration = started.elapsed();
            }
        } else {
            self.last_significant_change = Some(Instant::now());
            self.plateau_started = None;
            self.plateau_duration = Duration::ZERO;
        }
    }

    /// Whether the metric is currently plateaued.
    pub fn is_plateau(&self) -> bool {
        self.is_plateau
    }

    /// Real elapsed duration of the current plateau, measured from the instant
    /// it was first detected. `Duration::ZERO` when not plateaued.
    pub fn plateau_duration(&self) -> Duration {
        self.plateau_duration
    }

    /// Instant of the most recent significant (non-plateau) change.
    pub fn last_significant_change(&self) -> Option<Instant> {
        self.last_significant_change
    }

    fn reset(&mut self) {
        self.recent_values.clear();
        self.is_plateau = false;
        self.plateau_duration = Duration::ZERO;
        self.last_significant_change = None;
    }
}

impl<A: Float + Default + Clone + Sum + Send + Sync + Send + Sync> PerformanceAnomalyDetector<A> {
    fn new(threshold: f64) -> Self {
        Self {
            threshold: scalar_or(threshold, A::zero()),
            historical_stats: HashMap::new(),
            recent_anomalies: VecDeque::with_capacity(100),
            adaptive_threshold: true,
        }
    }

    fn check_for_anomalies(
        &mut self,
        snapshot: &PerformanceSnapshot<A>,
    ) -> Result<Vec<PerformanceAnomaly<A>>, String> {
        let mut anomalies = Vec::new();

        // Check loss anomaly
        let loss_anomaly = self.check_metric_anomaly("loss", snapshot.loss, snapshot.timestamp)?;
        if let Some(anomaly) = loss_anomaly {
            anomalies.push(anomaly);
        }

        // Check accuracy anomaly if available
        if let Some(accuracy) = snapshot.accuracy {
            let accuracy_anomaly =
                self.check_metric_anomaly("accuracy", accuracy, snapshot.timestamp)?;
            if let Some(anomaly) = accuracy_anomaly {
                anomalies.push(anomaly);
            }
        }

        Ok(anomalies)
    }

    fn check_metric_anomaly(
        &mut self,
        metric_name: &str,
        value: A,
        timestamp: Instant,
    ) -> Result<Option<PerformanceAnomaly<A>>, String> {
        // Update statistics for this metric
        let stats = self
            .historical_stats
            .entry(metric_name.to_string())
            .or_insert_with(|| MetricStatistics {
                mean: value,
                variance: A::zero(),
                min_value: value,
                max_value: value,
                count: 0,
                last_update: timestamp,
            });

        // Update running statistics
        stats.count += 1;
        let delta = value - stats.mean;
        stats.mean = stats.mean + delta / try_scalar_str::<A, _>(stats.count)?;
        let delta2 = value - stats.mean;
        stats.variance = stats.variance + delta * delta2;
        stats.min_value = stats.min_value.min(value);
        stats.max_value = stats.max_value.max(value);
        stats.last_update = timestamp;

        // Check for anomaly after sufficient samples
        if stats.count >= 10 {
            let std_dev = (stats.variance / try_scalar_str::<A, _>(stats.count - 1)?).sqrt();
            let z_score = (value - stats.mean) / std_dev.max(try_scalar_str::<A, _>(1e-8)?);

            if z_score.abs() > self.threshold {
                let severity = if z_score.abs() > try_scalar_str::<A, _>(3.0)? {
                    AnomalySeverity::Critical
                } else if z_score.abs() > try_scalar_str::<A, _>(2.5)? {
                    AnomalySeverity::Major
                } else {
                    AnomalySeverity::Moderate
                };

                let anomaly_type = if z_score > A::zero() {
                    AnomalyType::High
                } else {
                    AnomalyType::Low
                };

                let expected_range = (
                    stats.mean - self.threshold * std_dev,
                    stats.mean + self.threshold * std_dev,
                );

                let anomaly = PerformanceAnomaly {
                    timestamp,
                    metric_name: metric_name.to_string(),
                    observed_value: value,
                    expected_range,
                    severity,
                    anomaly_type,
                };

                return Ok(Some(anomaly));
            }
        }

        Ok(None)
    }

    /// Move the anomaly-detection threshold, if this detector is configured to
    /// adapt it.
    ///
    /// `adaptive_threshold` was set at construction and never consulted, so a
    /// detector configured with a fixed threshold still had it moved by every
    /// `AdaptationType::PerformanceThreshold` adaptation. Returns whether the
    /// threshold actually moved.
    fn update_threshold(&mut self, new_threshold: A) -> bool {
        if !self.adaptive_threshold {
            return false;
        }
        self.threshold = new_threshold;
        true
    }

    /// Whether this detector adapts its threshold.
    pub fn is_threshold_adaptive(&self) -> bool {
        self.adaptive_threshold
    }

    fn reset(&mut self) {
        self.historical_stats.clear();
        self.recent_anomalies.clear();
    }
}

/// Diagnostic information for performance tracking
#[derive(Debug, Clone)]
pub struct PerformanceDiagnostics {
    pub history_size: usize,
    pub baseline_set: bool,
    pub trends_available: bool,
    pub anomalies_detected: usize,
    pub plateau_detected: bool,
    pub prediction_accuracy: f64,
}

#[cfg(test)]
mod plateau_and_prediction_regression_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn snapshot(loss: f64) -> PerformanceSnapshot<f64> {
        PerformanceSnapshot {
            timestamp: Instant::now(),
            processing_duration: Duration::from_millis(3),
            loss,
            accuracy: Some(1.0 - loss.min(1.0)),
            convergence_rate: None,
            gradient_norm: Some(loss.sqrt()),
            parameter_update_magnitude: Some(loss / 10.0),
            data_statistics: DataStatistics {
                sample_count: 1,
                feature_means: Array1::from_vec(vec![loss]),
                feature_stds: Array1::from_vec(vec![0.0]),
                average_quality: 1.0,
                timestamp: Instant::now(),
            },
            resource_usage: ResourceUsage::default(),
            custom_metrics: HashMap::new(),
        }
    }

    /// P1f: `detect_plateau` reduced with `fold(A::zero(), A::min)`, seeding the
    /// minimum at `0`. For an all-positive metric the observed minimum could
    /// never be anything but `0`, so `range == max_val` and the detector only
    /// fired when the *largest* value fell under the threshold. A genuinely flat
    /// series at level 5.0 with a 0.01 threshold must be reported as a plateau —
    /// under the old code `range` would have been `5.0` and it never would be.
    #[test]
    fn plateau_is_detected_for_a_flat_series_away_from_zero() {
        let mut detector = PlateauDetector::<f64>::new(10, 0.01);
        for i in 0..10 {
            // Flat at 5.0, jitter of 0.001 — total range 0.001 < 0.01.
            detector.update(5.0 + 0.001 * ((i % 2) as f64));
        }
        assert!(
            detector.is_plateau(),
            "P1f regression: a flat series at level 5.0 was not detected as a \
             plateau (the min seed was still 0)"
        );
    }

    /// P1f: the mirror case — a series with genuine variation must not be
    /// reported as a plateau.
    #[test]
    fn plateau_is_not_detected_for_a_varying_series() {
        let mut detector = PlateauDetector::<f64>::new(10, 0.01);
        for i in 0..10 {
            detector.update(5.0 + i as f64);
        }
        assert!(
            !detector.is_plateau(),
            "a series spanning 9.0 must not be a plateau under a 0.01 threshold"
        );
        assert_eq!(detector.plateau_duration(), Duration::ZERO);
        assert!(detector.last_significant_change().is_some());
    }

    /// P2f: `plateau_duration` was advanced by a hard-coded
    /// `Duration::from_secs(1)` per update, so after N updates it always claimed
    /// exactly N-1 seconds regardless of how fast samples arrived. It must now
    /// reflect real elapsed time — far less than a second for a tight loop.
    #[test]
    fn plateau_duration_is_real_elapsed_time_not_one_second_per_sample() {
        let mut detector = PlateauDetector::<f64>::new(5, 1.0);
        // 5 samples to trigger, then 20 more updates inside the plateau.
        for _ in 0..25 {
            detector.update(3.0);
        }
        assert!(detector.is_plateau());

        let duration = detector.plateau_duration();
        assert!(
            duration < Duration::from_secs(1),
            "P2f regression: plateau_duration is {duration:?} — the fabricated \
             one-second-per-sample clock is still in use (20 in-plateau updates \
             would have claimed ~20s)"
        );
    }

    /// P2f: the duration must genuinely grow with wall-clock time.
    #[test]
    fn plateau_duration_grows_with_wall_clock_time() {
        let mut detector = PlateauDetector::<f64>::new(3, 1.0);
        for _ in 0..3 {
            detector.update(2.0);
        }
        assert!(detector.is_plateau());
        let first = detector.plateau_duration();

        std::thread::sleep(Duration::from_millis(25));
        detector.update(2.0);
        let second = detector.plateau_duration();

        assert!(
            second > first,
            "the plateau duration must advance with real time ({first:?} -> {second:?})"
        );
        assert!(
            second >= Duration::from_millis(20),
            "expected at least the ~25ms that actually elapsed, got {second:?}"
        );
    }

    /// P3f: `predict` only ever ran `linear_prediction`;
    /// `exponential_prediction`, `prediction_methods` and `ensemble_weights`
    /// were dead. Every configured method must now run and be combined, which
    /// shows up as a populated `ensemble_weights` map and an ensemble label.
    #[test]
    fn prediction_runs_every_configured_method_and_records_weights() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");

        for i in 0..12 {
            tracker
                .add_performance(snapshot(10.0 - 0.5 * i as f64))
                .expect("add_performance");
        }

        let prediction = tracker.predict_performance(3).expect("prediction");
        assert!(
            prediction.method.starts_with("ensemble["),
            "P3f regression: only one method ran (method={})",
            prediction.method
        );
        assert!(
            prediction.method.contains("linear") && prediction.method.contains("exponential"),
            "both configured methods must appear in the ensemble label (got {})",
            prediction.method
        );
        assert!(
            prediction.confidence_interval.0 < prediction.predicted_value
                && prediction.predicted_value < prediction.confidence_interval.1,
            "the interval must bracket the point forecast"
        );
    }

    /// P3f: the linear forecaster must extrapolate a real least-squares trend.
    /// A perfectly linear series must be predicted almost exactly.
    #[test]
    fn linear_prediction_extrapolates_a_known_trend_exactly() {
        let predictor = PerformancePredictor::<f64>::new();
        // y = 2x for x = 1..=10, so the value at x = 13 is 26.
        let values: Vec<f64> = (1..=10).map(|x| 2.0 * x as f64).collect();
        let forecast = predictor
            .linear_prediction(&values, 3)
            .expect("linear_prediction");
        assert!(
            (forecast - 26.0).abs() < 1e-9,
            "expected 26.0 for a perfect y = 2x fit, got {forecast}"
        );
    }

    /// P3f: Holt's linear smoothing must follow the trend, not multiply the
    /// forecast by a fabricated 0.99 "assume slight improvement" factor.
    #[test]
    fn exponential_prediction_follows_the_trend_not_a_fixed_decay() {
        let predictor = PerformancePredictor::<f64>::new();
        // Steadily rising series: the forecast must be above the last value.
        let rising: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let up = predictor
            .exponential_prediction(&rising, 5, 0.5, 0.5)
            .expect("exponential_prediction");
        assert!(
            up > 20.0,
            "a rising series must forecast above its last value, got {up}"
        );

        // Steadily falling series: the forecast must be below the last value.
        let falling: Vec<f64> = (1..=20).map(|x| 21.0 - x as f64).collect();
        let down = predictor
            .exponential_prediction(&falling, 5, 0.5, 0.5)
            .expect("exponential_prediction");
        assert!(
            down < 1.0,
            "a falling series must forecast below its last value, got {down}"
        );
    }

    /// P4f: accuracy scoring was gated on a fabricated
    /// `Duration::from_secs(steps_ahead * 10)`, so on any stream faster than
    /// "10 seconds per batch" no prediction was ever scored and
    /// `model_accuracies` stayed permanently empty — which in turn meant the
    /// ensemble weights had nothing to be derived from. The horizon is now
    /// counted in snapshots, so accuracy is measured within a handful of
    /// batches with no sleeping at all.
    #[test]
    fn prediction_accuracy_is_scored_without_waiting_ten_seconds_per_step() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");

        for i in 0..12 {
            tracker
                .add_performance(snapshot(10.0 - 0.5 * i as f64))
                .expect("add_performance");
        }
        tracker.predict_performance(1).expect("prediction");

        // Two more snapshots is more than the 1-step horizon.
        for i in 0..2 {
            tracker
                .add_performance(snapshot(4.0 - 0.5 * i as f64))
                .expect("add_performance");
        }

        let diagnostics = tracker.get_diagnostics();
        assert!(
            diagnostics.prediction_accuracy > 0.0,
            "P4f regression: no prediction was ever scored, so accuracy is still \
             {} after the horizon elapsed",
            diagnostics.prediction_accuracy
        );
    }

    /// P4f: a forecaster with no implementation must be an honest error rather
    /// than contributing a fabricated number to the ensemble.
    #[test]
    fn unimplemented_prediction_methods_are_an_error() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");
        tracker.predictor.prediction_methods = vec![PredictionMethod::ARIMA { p: 1, d: 1, q: 1 }];

        for i in 0..12 {
            tracker
                .add_performance(snapshot(10.0 - 0.5 * i as f64))
                .expect("add_performance");
        }

        assert!(
            tracker.predict_performance(2).is_err(),
            "an unimplemented forecaster must not silently produce a value"
        );
    }
    /// `improvement_rate` was `improvement / 1.0`, an exact duplicate of
    /// `improvement`, so it said nothing about how *fast* the optimizer was
    /// improving. It must now be a real per-second rate, which for two
    /// improvements separated by a measurable interval differs from the raw
    /// improvement.
    #[test]
    fn improvement_rate_is_a_real_per_second_rate() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");

        tracker.add_performance(snapshot(10.0)).expect("first");
        // Second improvement, well after the first.
        std::thread::sleep(Duration::from_millis(40));
        tracker.add_performance(snapshot(9.0)).expect("second");
        std::thread::sleep(Duration::from_millis(40));
        tracker.add_performance(snapshot(8.0)).expect("third");

        let events = &tracker.improvement_tracker.improvement_history;
        assert!(
            events.len() >= 2,
            "at least two improvements should have been recorded, got {}",
            events.len()
        );
        let last = events
            .back()
            .expect("an improvement event must have been recorded");
        assert!(
            (last.improvement - 1.0).abs() < 1e-12,
            "the raw improvement should be 1.0, got {}",
            last.improvement
        );
        assert!(
            (last.improvement_rate - last.improvement).abs() > 1e-9,
            "the rate must differ from the raw improvement once a real interval \
             has elapsed (improvement={}, rate={})",
            last.improvement,
            last.improvement_rate
        );
        // 1.0 of improvement over ~40ms is roughly 25 per second, and certainly
        // more than 1 per second.
        assert!(
            last.improvement_rate > 1.0,
            "expected a rate well above 1/s for a 40ms interval, got {}",
            last.improvement_rate
        );
    }
}
