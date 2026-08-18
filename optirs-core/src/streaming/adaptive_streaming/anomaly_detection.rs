// Anomaly detection for streaming optimization data
//
// This module provides comprehensive anomaly detection capabilities for streaming
// data including statistical outlier detection, machine learning-based methods,
// ensemble approaches, and adaptive threshold management.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationType, StreamingDataPoint};

use crate::utils::scalar_or;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Comprehensive anomaly detector for streaming data
pub struct AnomalyDetector<A: Float + Send + Sync> {
    /// Anomaly detection configuration
    config: AnomalyConfig,
    /// Statistical anomaly detectors
    statistical_detectors: HashMap<String, Box<dyn StatisticalAnomalyDetector<A>>>,
    /// Machine learning-based detectors
    ml_detectors: HashMap<String, Box<dyn MLAnomalyDetector<A>>>,
    /// Ensemble detector
    ensemble_detector: EnsembleAnomalyDetector<A>,
    /// Adaptive threshold manager
    threshold_manager: AdaptiveThresholdManager<A>,
    /// Anomaly history and statistics
    anomaly_history: VecDeque<AnomalyEvent<A>>,
    /// False positive tracker
    false_positive_tracker: FalsePositiveTracker<A>,
    /// Anomaly response system
    response_system: AnomalyResponseSystem<A>,
    /// Bounded window of the most recently scored data points.
    ///
    /// `create_anomaly_context`/`calculate_recent_statistics` compute from this
    /// window; before it existed they returned a hard-coded two-feature summary
    /// that was wrong for every stream and silently assumed a feature width of
    /// exactly two.
    recent_points: VecDeque<StreamingDataPoint<A>>,
    /// Capacity of `recent_points`.
    recent_capacity: usize,
    /// Externally supplied context signals, set by the owning optimizer via
    /// [`AnomalyDetector::update_context_signals`]. Empty means "the caller has
    /// not reported any", which is honest — the detector has no direct access
    /// to the performance tracker, resource manager or drift detector.
    context_performance_metrics: Vec<A>,
    context_resource_usage: Vec<A>,
    context_drift_indicators: Vec<A>,
    /// Bounded window of recent ensemble anomaly scores, used to recalibrate
    /// detector thresholds against `AnomalyConfig::contamination_rate`.
    recent_scores: VecDeque<A>,
    /// Points scored since the last threshold recalibration.
    points_since_recalibration: usize,
}

/// Anomaly event record
#[derive(Debug, Clone)]
pub struct AnomalyEvent<A: Float + Send + Sync> {
    /// Event ID
    pub id: u64,
    /// Timestamp of detection
    pub timestamp: Instant,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Anomaly severity
    pub severity: AnomalySeverity,
    /// Confidence score
    pub confidence: A,
    /// Anomalous data point
    pub data_point: StreamingDataPoint<A>,
    /// Detector that found the anomaly
    pub detector_name: String,
    /// Anomaly score
    pub anomaly_score: A,
    /// Context information
    pub context: AnomalyContext<A>,
    /// Response actions taken
    pub response_actions: Vec<String>,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    /// Statistical outlier
    StatisticalOutlier,
    /// Sudden change in pattern
    PatternChange,
    /// Temporal anomaly
    TemporalAnomaly,
    /// Spatial anomaly in feature space
    SpatialAnomaly,
    /// Contextual anomaly
    ContextualAnomaly,
    /// Collective anomaly
    CollectiveAnomaly,
    /// Point anomaly
    PointAnomaly,
    /// Data quality anomaly
    DataQualityAnomaly,
    /// Performance anomaly
    PerformanceAnomaly,
    /// Custom anomaly type
    Custom(String),
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity requiring immediate action
    Critical,
}

/// Context information for anomaly
#[derive(Debug, Clone)]
pub struct AnomalyContext<A: Float + Send + Sync> {
    /// Recent data statistics
    pub recent_statistics: DataStatistics<A>,
    /// Performance metrics at detection time
    pub performance_metrics: Vec<A>,
    /// Resource usage at detection time
    pub resource_usage: Vec<A>,
    /// Recent drift indicators
    pub drift_indicators: Vec<A>,
    /// Time since last anomaly
    pub time_since_last_anomaly: Duration,
}

/// Data statistics for anomaly context
#[derive(Debug, Clone)]
pub struct DataStatistics<A: Float + Send + Sync> {
    /// Mean values for features
    pub means: Vec<A>,
    /// Standard deviations for features
    pub std_devs: Vec<A>,
    /// Minimum values
    pub min_values: Vec<A>,
    /// Maximum values
    pub max_values: Vec<A>,
    /// Median values
    pub medians: Vec<A>,
    /// Skewness values
    pub skewness: Vec<A>,
    /// Kurtosis values
    pub kurtosis: Vec<A>,
}

/// Trait for statistical anomaly detectors
pub trait StatisticalAnomalyDetector<A: Float + Send + Sync>: Send + Sync {
    /// Detects anomalies in the given data point
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String>;

    /// Updates the detector with new data
    fn update(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String>;

    /// Resets the detector state
    fn reset(&mut self);

    /// Gets the detector name
    fn name(&self) -> String;

    /// Gets current detection threshold
    fn get_threshold(&self) -> A;

    /// Sets detection threshold
    fn set_threshold(&mut self, threshold: A);
}

/// Trait for machine learning-based anomaly detectors
pub trait MLAnomalyDetector<A: Float + Send + Sync>: Send + Sync {
    /// Detects anomalies using ML model
    fn detect_anomaly(
        &mut self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyDetectionResult<A>, String>;

    /// Trains the ML model with new data
    fn train(&mut self, training_data: &[StreamingDataPoint<A>]) -> Result<(), String>;

    /// Updates the model incrementally
    fn update_incremental(&mut self, data_point: &StreamingDataPoint<A>) -> Result<(), String>;

    /// Records the ground truth for one of this detector's predictions.
    ///
    /// Quality metrics for an unsupervised novelty detector are unknowable
    /// without labels, so they are only defined once outcomes are fed back in
    /// through this method (see [`AnomalyDetector::record_detection_outcome`]).
    fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool);

    /// Gets model performance metrics.
    ///
    /// Returns an error while no labelled outcome has been recorded: an
    /// accuracy figure invented before any ground truth exists would be a
    /// fabrication, not a measurement.
    fn get_performance_metrics(&self) -> Result<MLModelMetrics<A>, String>;

    /// Gets detector name
    fn name(&self) -> String;
}

// The confusion-matrix counters, the bounded score-retention buffer and the
// full-curve ROC computation live in `anomaly_scoring`; `DetectionCounters` is
// re-exported here so its existing path keeps resolving.
pub use super::anomaly_scoring::DetectionCounters;

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult<A: Float + Send + Sync> {
    /// Whether an anomaly was detected
    pub is_anomaly: bool,
    /// Anomaly score (higher = more anomalous)
    pub anomaly_score: A,
    /// Confidence in the detection
    pub confidence: A,
    /// Anomaly type if detected
    pub anomaly_type: Option<AnomalyType>,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Additional metadata
    pub metadata: HashMap<String, A>,
}

/// Performance metrics for ML models
#[derive(Debug, Clone)]
pub struct MLModelMetrics<A: Float + Send + Sync> {
    /// Accuracy of anomaly detection
    pub accuracy: A,
    /// Precision (true positives / (true positives + false positives))
    pub precision: A,
    /// Recall (true positives / (true positives + false negatives))
    pub recall: A,
    /// F1 score
    pub f1_score: A,
    /// Area under the ROC curve, traced over every threshold the retained
    /// labelled scores admit, or `None` while fewer than two labelled scores of
    /// either class have been fed back.
    ///
    /// This is deliberately optional. A detector's confusion matrix alone fixes
    /// exactly one point on the ROC curve, so no area can be derived from it;
    /// the single-operating-point substitute `(TPR + TNR) / 2` that used to be
    /// reported here was a different statistic wearing the AUC's name. Call
    /// [`DetectionCounters::auc_roc`] for the reason it is unavailable.
    pub auc_roc: Option<A>,
    /// False positive rate
    pub false_positive_rate: A,
    /// Training time
    pub training_time: Duration,
    /// Inference time per sample
    pub inference_time: Duration,
}

// The ensemble voting detector lives in `anomaly_ensemble`; its types are
// re-exported here so their existing paths keep resolving.
pub use super::anomaly_ensemble::{
    EnsembleAnomalyDetector, EnsembleConfig, EnsembleVotingStrategy,
};

/// Performance tracking for individual detectors
#[derive(Debug, Clone)]
pub struct DetectorPerformance<A: Float + Send + Sync> {
    /// Recent accuracy
    pub recent_accuracy: A,
    /// Historical accuracy
    pub historical_accuracy: A,
    /// False positive rate
    pub false_positive_rate: A,
    /// False negative rate
    pub false_negative_rate: A,
    /// Detection latency
    pub detection_latency: Duration,
    /// Reliability score
    pub reliability_score: A,
}

/// Adaptive threshold management system
pub struct AdaptiveThresholdManager<A: Float + Send + Sync> {
    /// Current thresholds for different detectors
    thresholds: HashMap<String, A>,
    /// Threshold bounds
    threshold_bounds: HashMap<String, (A, A)>,
}

/// Threshold adaptation strategies
#[derive(Debug, Clone)]
pub enum ThresholdAdaptationStrategy {
    /// Fixed thresholds
    Fixed,
    /// Performance-based adaptation
    PerformanceBased,
    /// Quantile-based adaptation
    QuantileBased { quantile: f64 },
    /// ROC-optimized thresholds
    ROCOptimized,
    /// Precision-recall optimized
    PROptimized,
    /// False positive rate controlled
    FPRControlled { target_fpr: f64 },
    /// Adaptive based on data distribution
    DistributionAdaptive,
}

/// Performance feedback for threshold adaptation
#[derive(Debug, Clone)]
pub struct ThresholdPerformanceFeedback<A: Float + Send + Sync> {
    /// Detector name
    pub detector_name: String,
    /// Threshold value
    pub threshold: A,
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// True negatives
    pub true_negatives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// Timestamp
    pub timestamp: Instant,
}

/// Threshold adaptation parameters
#[derive(Debug, Clone)]
pub struct ThresholdAdaptationParams<A: Float + Send + Sync> {
    /// Learning rate for threshold updates
    pub learning_rate: A,
    /// Momentum for threshold changes
    pub momentum: A,
    /// Minimum threshold change
    pub min_change: A,
    /// Maximum threshold change per update
    pub max_change: A,
    /// Adaptation frequency
    pub adaptation_frequency: usize,
}

/// False positive tracking system
pub struct FalsePositiveTracker<A: Float + Send + Sync> {
    /// Recent false positive events
    false_positives: VecDeque<FalsePositiveEvent<A>>,
    /// False positive rate calculation
    fp_rate_calculator: FPRateCalculator<A>,
}

/// False positive event
#[derive(Debug, Clone)]
pub struct FalsePositiveEvent<A: Float + Send + Sync> {
    /// Event timestamp
    pub timestamp: Instant,
    /// Data point incorrectly flagged
    pub data_point: StreamingDataPoint<A>,
    /// Detector that generated false positive
    pub detector_name: String,
    /// Anomaly score given
    pub anomaly_score: A,
    /// Context at time of false positive
    pub context: AnomalyContext<A>,
}

/// False positive rate calculator
pub struct FPRateCalculator<A: Float + Send + Sync> {
    /// Recent detection results
    recent_results: VecDeque<DetectionResult>,
    /// Calculation window size
    window_size: usize,
    /// Current false positive rate
    current_fp_rate: A,
}

/// Detection result for FP rate calculation
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Timestamp
    pub timestamp: Instant,
    /// Was anomaly detected
    pub anomaly_detected: bool,
    /// Was it actually an anomaly (ground truth)
    pub ground_truth: Option<bool>,
    /// Detector name
    pub detector_name: String,
}

/// Patterns in false positives
#[derive(Debug, Clone)]
pub struct FalsePositivePatterns<A: Float + Send + Sync> {
    /// Temporal patterns
    pub temporal_patterns: Vec<TemporalPattern>,
    /// Feature-based patterns
    pub feature_patterns: HashMap<String, A>,
    /// Context patterns
    pub context_patterns: Vec<ContextPattern<A>>,
    /// Detector-specific patterns
    pub detector_patterns: HashMap<String, Vec<A>>,
}

/// Temporal pattern in false positives
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Pattern type
    pub pattern_type: TemporalPatternType,
    /// Pattern strength
    pub strength: f64,
    /// Pattern period (if periodic)
    pub period: Option<Duration>,
    /// Pattern confidence
    pub confidence: f64,
}

/// Types of temporal patterns
#[derive(Debug, Clone)]
pub enum TemporalPatternType {
    /// Periodic false positives
    Periodic,
    /// False positives at specific times
    TimeSpecific,
    /// Burst of false positives
    Burst,
    /// Gradual increase in false positives
    Trend,
}

/// Context pattern for false positives
#[derive(Debug, Clone)]
pub struct ContextPattern<A: Float + Send + Sync> {
    /// Context features associated with false positives
    pub context_features: Vec<A>,
    /// Pattern frequency
    pub frequency: usize,
    /// Pattern reliability
    pub reliability: A,
}

/// False positive mitigation strategies
#[derive(Debug, Clone)]
pub enum FPMitigationStrategy {
    /// Adjust detection thresholds
    ThresholdAdjustment,
    /// Feature selection/weighting
    FeatureAdjustment,
    /// Ensemble reweighting
    EnsembleReweighting,
    /// Context-aware filtering
    ContextFiltering,
    /// Temporal filtering
    TemporalFiltering,
    /// Model retraining
    ModelRetraining,
}

/// Anomaly response system
pub struct AnomalyResponseSystem<A: Float + Send + Sync> {
    /// Response strategies
    response_strategies: HashMap<AnomalyType, Vec<ResponseAction>>,
    /// Response execution engine
    response_executor: ResponseExecutor<A>,
    /// Monotonically increasing response identifier.
    next_response_id: u64,
    /// Messages written by executed `Log` actions.
    log_entries: VecDeque<String>,
    /// Messages written by executed `Alert` actions.
    alert_entries: VecDeque<String>,
    /// Data points held by executed `Quarantine` actions.
    quarantined_points: VecDeque<StreamingDataPoint<A>>,
    /// Threshold adjustment requested by `ModelAdjustment` actions, awaiting
    /// application by the owning detector.
    pending_threshold_adjustment: Option<f64>,
    /// Monitoring level, raised by `IncreaseMonitoring` actions.
    monitoring_level: u32,
}

/// Number of scored data points retained for context statistics.
const RECENT_POINT_WINDOW: usize = 512;

/// Bound on the response log / alert / execution histories.
const RESPONSE_HISTORY_CAPACITY: usize = 1000;

/// Bound on the quarantine buffer.
const QUARANTINE_CAPACITY: usize = 256;

/// Response actions for anomalies
#[derive(Debug, Clone)]
pub enum ResponseAction {
    /// Log the anomaly
    Log,
    /// Alert operators
    Alert,
    /// Quarantine the data
    Quarantine,
    /// Adjust model parameters
    ModelAdjustment,
    /// Increase monitoring
    IncreaseMonitoring,
    /// Trigger recovery procedure
    TriggerRecovery,
    /// Custom action
    Custom(String),
}

/// Response execution engine
pub struct ResponseExecutor<A: Float + Send + Sync> {
    /// Pending responses
    pending_responses: VecDeque<PendingResponse<A>>,
    /// Response execution history
    execution_history: VecDeque<ResponseExecution<A>>,
    /// Resource limits for responses
    resource_limits: ResponseResourceLimits,
}

/// Pending response action
#[derive(Debug, Clone)]
pub struct PendingResponse<A: Float + Send + Sync> {
    /// Response ID
    pub id: u64,
    /// Associated anomaly event
    pub anomaly_event: AnomalyEvent<A>,
    /// Response action to execute
    pub action: ResponseAction,
    /// Priority level
    pub priority: ResponsePriority,
    /// Scheduled execution time
    pub scheduled_time: Instant,
    /// Timeout for execution
    pub timeout: Duration,
}

/// Response execution record
#[derive(Debug, Clone)]
pub struct ResponseExecution<A: Float + Send + Sync> {
    /// Execution ID
    pub id: u64,
    /// Response that was executed
    pub response: PendingResponse<A>,
    /// Execution start time
    pub start_time: Instant,
    /// Execution duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Resources consumed
    pub resources_consumed: HashMap<String, A>,
}

/// Response priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResponsePriority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Resource limits for response execution
#[derive(Debug, Clone)]
pub struct ResponseResourceLimits {
    /// Maximum concurrent responses
    pub max_concurrent_responses: usize,
    /// Maximum CPU usage for responses
    pub max_cpu_usage: f64,
    /// Maximum memory usage for responses
    pub max_memory_usage: usize,
    /// Maximum response execution time
    pub max_execution_time: Duration,
}

/// Effectiveness metrics for responses
#[derive(Debug, Clone)]
pub struct EffectivenessMetrics<A: Float + Send + Sync> {
    /// Success rate
    pub success_rate: A,
    /// Average response time
    pub avg_response_time: Duration,
    /// Problem resolution rate
    pub resolution_rate: A,
    /// False alarm reduction
    pub false_alarm_reduction: A,
    /// Cost-benefit ratio
    pub cost_benefit_ratio: A,
}

/// Response outcome record
#[derive(Debug, Clone)]
pub struct ResponseOutcome<A: Float + Send + Sync> {
    /// Response execution
    pub execution: ResponseExecution<A>,
    /// Outcome measurement
    pub outcome: OutcomeMeasurement<A>,
    /// Follow-up required
    pub follow_up_required: bool,
    /// Lessons learned
    pub lessons_learned: Vec<String>,
}

/// Measurement of response outcome
#[derive(Debug, Clone)]
pub struct OutcomeMeasurement<A: Float + Send + Sync> {
    /// Did response resolve the issue
    pub issue_resolved: bool,
    /// Time to resolution
    pub time_to_resolution: Duration,
    /// Performance impact
    pub performance_impact: A,
    /// Side effects observed
    pub side_effects: Vec<String>,
    /// Overall effectiveness score
    pub effectiveness_score: A,
}

/// Trend analysis for response effectiveness
#[derive(Debug, Clone)]
pub struct TrendAnalysis<A: Float + Send + Sync> {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend magnitude
    pub trend_magnitude: A,
    /// Trend confidence
    pub trend_confidence: A,
    /// Trend stability
    pub trend_stability: A,
}

/// Trend directions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving effectiveness
    Improving,
    /// Declining effectiveness
    Declining,
    /// Stable effectiveness
    Stable,
    /// Oscillating effectiveness
    Oscillating,
}

/// Escalation rules for severe anomalies
#[derive(Debug, Clone)]
pub struct EscalationRule<A: Float + Send + Sync> {
    /// Rule name
    pub name: String,
    /// Conditions for escalation
    pub conditions: Vec<EscalationCondition<A>>,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
    /// Priority level
    pub priority: EscalationPriority,
}

/// Escalation conditions
#[derive(Debug, Clone)]
pub struct EscalationCondition<A: Float + Send + Sync> {
    /// Condition type
    pub condition_type: EscalationConditionType,
    /// Threshold value
    pub threshold: A,
    /// Time window for condition
    pub time_window: Duration,
}

/// Types of escalation conditions
#[derive(Debug, Clone)]
pub enum EscalationConditionType {
    /// Multiple anomalies in time window
    MultipleAnomalies,
    /// High severity anomaly
    HighSeverity,
    /// Response failure
    ResponseFailure,
    /// Performance degradation
    PerformanceDegradation,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// Escalation actions
#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Notify administrators
    NotifyAdmin,
    /// Trigger emergency protocols
    EmergencyProtocol,
    /// Shutdown affected systems
    SystemShutdown,
    /// Activate backup systems
    ActivateBackup,
    /// Increase resource allocation
    IncreaseResources,
}

/// Escalation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EscalationPriority {
    /// Normal escalation
    Normal = 0,
    /// Urgent escalation
    Urgent = 1,
    /// Emergency escalation
    Emergency = 2,
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync + 'static> AnomalyDetector<A> {
    /// Creates a new anomaly detector
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let anomaly_config = config.anomaly_config.clone();

        let mut statistical_detectors: HashMap<String, Box<dyn StatisticalAnomalyDetector<A>>> =
            HashMap::new();
        let mut ml_detectors: HashMap<String, Box<dyn MLAnomalyDetector<A>>> = HashMap::new();

        // Initialize statistical detectors
        statistical_detectors.insert(
            "zscore".to_string(),
            Box::new(super::anomaly_statistical::ZScoreDetector::new(
                anomaly_config.threshold,
            )?),
        );
        statistical_detectors.insert(
            "iqr".to_string(),
            Box::new(super::anomaly_statistical::IQRDetector::new(
                anomaly_config.threshold,
            )?),
        );

        // Initialize ML detectors based on method
        match anomaly_config.detection_method {
            AnomalyDetectionMethod::IsolationForest => {
                ml_detectors.insert(
                    "isolation_forest".to_string(),
                    Box::new(super::anomaly_ml::IsolationForestDetector::new()?),
                );
            }
            AnomalyDetectionMethod::OneClassSVM => {
                ml_detectors.insert(
                    "one_class_svm".to_string(),
                    Box::new(super::anomaly_ml::OneClassSvmDetector::new()?),
                );
            }
            AnomalyDetectionMethod::LocalOutlierFactor => {
                ml_detectors.insert(
                    "lof".to_string(),
                    Box::new(super::anomaly_ml::LofDetector::new()?),
                );
            }
            _ => {
                // Use statistical methods for other cases
            }
        }

        let ensemble_detector = EnsembleAnomalyDetector::new(EnsembleVotingStrategy::Weighted)?;
        // `AnomalyConfig::enable_adaptive_threshold` and `contamination_rate`
        // are honoured by `recalibrate_thresholds` below, which is the code
        // that actually moves thresholds. A `ThresholdAdaptationStrategy` was
        // also derived here and handed to the manager, but the manager never
        // read it -- two parallel spellings of the same setting, one of them
        // inert. Only the working one remains.
        let threshold_manager = AdaptiveThresholdManager::new()?;
        let false_positive_tracker = FalsePositiveTracker::new();
        let response_system = AnomalyResponseSystem::new(&anomaly_config.response_strategy)?;

        let recent_capacity = RECENT_POINT_WINDOW;
        Ok(Self {
            config: anomaly_config,
            statistical_detectors,
            ml_detectors,
            ensemble_detector,
            threshold_manager,
            anomaly_history: VecDeque::with_capacity(10000),
            false_positive_tracker,
            response_system,
            recent_points: VecDeque::with_capacity(recent_capacity),
            recent_capacity,
            context_performance_metrics: Vec::new(),
            context_resource_usage: Vec::new(),
            context_drift_indicators: Vec::new(),
            recent_scores: VecDeque::with_capacity(recent_capacity),
            points_since_recalibration: 0,
        })
    }

    /// Recalibrates every statistical detector's threshold to the empirical
    /// `1 - contamination_rate` quantile of the recent ensemble anomaly scores
    /// (CF1).
    ///
    /// `AnomalyConfig::contamination_rate` — the assumed fraction of the stream
    /// that is anomalous — previously had no reader at all, so the "assumption"
    /// influenced nothing: thresholds stayed wherever they were initialised no
    /// matter how the stream was actually distributed. Calibrating to that
    /// quantile is the standard use of a contamination parameter: it makes the
    /// detector flag approximately that fraction of points.
    ///
    /// Runs only when `enable_adaptive_threshold` is set, and only once every
    /// `window_size` scored points. A `contamination_rate` outside `(0, 1)` is
    /// an honest error rather than a silently clamped guess.
    fn recalibrate_thresholds(&mut self) -> Result<(), String> {
        if !self.config.enable_adaptive_threshold {
            return Ok(());
        }
        let window = self.config.window_size.max(1);
        if self.points_since_recalibration < window || self.recent_scores.len() < window {
            return Ok(());
        }
        self.points_since_recalibration = 0;

        let contamination = self.config.contamination_rate;
        if !(contamination > 0.0 && contamination < 1.0) {
            return Err(format!(
                "AnomalyConfig::contamination_rate must be in (0, 1), got {contamination}"
            ));
        }

        let mut scores: Vec<A> = self.recent_scores.iter().copied().collect();
        scores.sort_by(crate::utils::total_order);
        // Nearest-rank quantile: index of the first score at or above the
        // (1 - contamination) quantile of the window.
        let rank = ((1.0 - contamination) * scores.len() as f64).floor() as usize;
        let index = rank.min(scores.len().saturating_sub(1));
        let Some(&target) = scores.get(index) else {
            return Ok(());
        };

        for (name, detector) in &mut self.statistical_detectors {
            let bounded = match self.threshold_manager.threshold_bounds.get(name) {
                Some(&(low, high)) => target.max(low).min(high),
                None => target,
            };
            detector.set_threshold(bounded);
            self.threshold_manager
                .thresholds
                .insert(name.clone(), bounded);
        }
        Ok(())
    }

    /// Test-only list of thresholds the contamination calibration has applied.
    #[cfg(test)]
    pub(crate) fn calibrated_threshold_names_for_test(&self) -> Vec<A> {
        self.threshold_manager
            .thresholds
            .values()
            .copied()
            .collect()
    }

    /// Threshold currently applied to `detector_name` by the contamination-rate
    /// calibration, if it has run.
    pub fn calibrated_threshold(&self, detector_name: &str) -> Option<A> {
        self.threshold_manager
            .thresholds
            .get(detector_name)
            .copied()
    }

    /// Supplies the context signals that the detector cannot observe itself.
    ///
    /// The owning optimizer knows the current performance, resource and drift
    /// state; feeding them in here makes [`AnomalyContext`] carry real values
    /// instead of the placeholders it used to be built from.
    pub fn update_context_signals(
        &mut self,
        performance_metrics: Vec<A>,
        resource_usage: Vec<A>,
        drift_indicators: Vec<A>,
    ) {
        self.context_performance_metrics = performance_metrics;
        self.context_resource_usage = resource_usage;
        self.context_drift_indicators = drift_indicators;
    }

    /// Feeds ground truth for the most recent detection back into the
    /// detectors and the false-positive tracker.
    ///
    /// This is the only path by which the ML detectors' quality metrics become
    /// defined; without it `get_performance_metrics` honestly reports that it
    /// has nothing to measure.
    pub fn record_detection_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        for detector in self.ml_detectors.values_mut() {
            detector.record_outcome(predicted_anomaly, was_true_anomaly);
        }
        // Every ensemble member is scored against its *own* verdict on the most
        // recent point, which is what `EnsembleVotingStrategy::Adaptive` needs
        // to weight them by measured skill.
        self.ensemble_detector.record_outcome(was_true_anomaly);
        self.false_positive_tracker
            .record_outcome(predicted_anomaly, was_true_anomaly);

        // A confirmed false positive is attributed to the most recent recorded
        // anomaly, which is the one the caller is giving feedback on.
        if predicted_anomaly && !was_true_anomaly {
            if let Some(event) = self.anomaly_history.back().cloned() {
                self.false_positive_tracker.record_false_positive(&event);
            }
        }
    }

    /// Number of confirmed false positives retained by the tracker.
    pub fn confirmed_false_positive_count(&self) -> usize {
        self.false_positive_tracker.confirmed_false_positive_count()
    }

    /// Selects how the per-detector verdicts are combined.
    ///
    /// The constructor builds a `Weighted` ensemble; `Adaptive` is the strategy
    /// that derives its weights from the ground-truth feedback recorded by
    /// [`Self::record_detection_outcome`], and was unreachable while no setter
    /// existed.
    pub fn set_ensemble_voting_strategy(&mut self, strategy: EnsembleVotingStrategy) {
        self.ensemble_detector.set_voting_strategy(strategy);
    }

    /// Sets the weight [`EnsembleVotingStrategy::Weighted`] gives one detector.
    ///
    /// Without this the configured-weight strategy had no way to be configured,
    /// so it always fell back to the uniform weight of one.
    pub fn set_detector_weight(&mut self, detector_name: &str, weight: A) {
        self.ensemble_detector
            .set_detector_weight(detector_name, weight);
    }

    /// Balanced accuracy measured for one ensemble member, or `None` before any
    /// ground truth has been recorded for it.
    pub fn detector_balanced_accuracy(&self, detector_name: &str) -> Option<f64> {
        self.ensemble_detector
            .detector_balanced_accuracy(detector_name)
    }

    /// Number of response executions the response system has recorded.
    pub fn response_execution_count(&self) -> usize {
        self.response_system.execution_count()
    }

    /// Entries written by executed `Log` response actions.
    pub fn response_log_entry_count(&self) -> usize {
        self.response_system.log_entry_count()
    }

    /// Entries written by executed `Alert` response actions.
    pub fn response_alert_entry_count(&self) -> usize {
        self.response_system.alert_entry_count()
    }

    /// Data points held by executed `Quarantine` response actions.
    pub fn quarantined_point_count(&self) -> usize {
        self.response_system.quarantined_count()
    }

    /// Monitoring level, raised by executed `IncreaseMonitoring` actions.
    pub fn monitoring_level(&self) -> u32 {
        self.response_system.monitoring_level()
    }

    /// Quality metrics for every registered ML detector that has labelled
    /// outcomes recorded. Detectors without ground truth are reported with the
    /// error explaining why, rather than a fabricated score.
    pub fn ml_performance_metrics(&self) -> HashMap<String, Result<MLModelMetrics<A>, String>> {
        self.ml_detectors
            .iter()
            .map(|(name, detector)| (name.clone(), detector.get_performance_metrics()))
            .collect()
    }

    /// Number of data points currently retained for context statistics.
    pub fn recent_window_len(&self) -> usize {
        self.recent_points.len()
    }

    /// Builds the anomaly context for a data point without recording an event.
    ///
    /// Exposed so callers (and tests) can inspect the real statistics the
    /// detector derives from its retained window.
    pub fn build_context_for_test(
        &self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyContext<A>, String> {
        self.create_anomaly_context(data_point)
    }

    /// Adds a scored point to the bounded context window.
    fn remember_point(&mut self, data_point: &StreamingDataPoint<A>) {
        if self.recent_points.len() >= self.recent_capacity {
            self.recent_points.pop_front();
        }
        self.recent_points.push_back(data_point.clone());
    }

    /// Detects anomalies in a data point
    pub fn detect_anomaly(&mut self, data_point: &StreamingDataPoint<A>) -> Result<bool, String> {
        let mut detection_results = HashMap::new();

        // Run statistical detectors
        for (name, detector) in &mut self.statistical_detectors {
            let result = detector.detect_anomaly(data_point)?;
            detection_results.insert(name.clone(), result);
        }

        // Run ML detectors
        for (name, detector) in &mut self.ml_detectors {
            let result = detector.detect_anomaly(data_point)?;
            detection_results.insert(name.clone(), result);
        }

        // Combine results using ensemble
        let ensemble_result = self.ensemble_detector.combine_results(detection_results)?;

        // Retain the point for the context statistics window regardless of the
        // verdict — the summary describes the recent stream, not just the
        // anomalies in it.
        self.remember_point(data_point);

        // Feed the contamination-rate calibration window (CF1).
        if self.recent_scores.len() >= self.recent_capacity {
            self.recent_scores.pop_front();
        }
        self.recent_scores.push_back(ensemble_result.anomaly_score);
        self.points_since_recalibration = self.points_since_recalibration.saturating_add(1);
        self.recalibrate_thresholds()?;

        // Check if anomaly was detected
        if ensemble_result.is_anomaly {
            // Create anomaly event
            let mut anomaly_event = AnomalyEvent {
                id: self.generate_event_id(),
                timestamp: Instant::now(),
                anomaly_type: ensemble_result
                    .anomaly_type
                    .as_ref()
                    .cloned()
                    .unwrap_or(AnomalyType::StatisticalOutlier),
                severity: ensemble_result.severity.clone(),
                confidence: ensemble_result.confidence,
                data_point: data_point.clone(),
                detector_name: "ensemble".to_string(),
                anomaly_score: ensemble_result.anomaly_score,
                context: self.create_anomaly_context(data_point)?,
                response_actions: Vec::new(),
            };

            // Trigger the configured responses and record what actually ran,
            // so `response_actions` reflects real executions instead of the
            // empty vector a no-op `trigger_response` left behind.
            anomaly_event.response_actions =
                self.response_system.trigger_response(&anomaly_event)?;

            // Any threshold adjustment the responses asked for is applied for
            // real, not merely logged.
            let pending_adjustment = self
                .response_system
                .take_pending_threshold_adjustment()
                // CF1: with adaptive thresholding switched off, a response may
                // still be recorded but must not move any threshold.
                .filter(|_| self.config.enable_adaptive_threshold);
            if let Some(adjustment) = pending_adjustment {
                let magnitude = A::from(adjustment).ok_or_else(|| {
                    format!("threshold adjustment {adjustment} is not representable")
                })?;
                for detector in self.statistical_detectors.values_mut() {
                    let updated = detector.get_threshold() + magnitude;
                    detector.set_threshold(updated);
                }
                self.ensemble_detector.adjust_sensitivity(magnitude)?;
            }

            // Record anomaly
            self.record_anomaly(anomaly_event)?;

            return Ok(true);
        }

        // Update detectors with normal data
        for detector in self.statistical_detectors.values_mut() {
            detector.update(data_point)?;
        }

        for detector in self.ml_detectors.values_mut() {
            detector.update_incremental(data_point)?;
        }

        Ok(false)
    }

    /// Generates unique event ID
    fn generate_event_id(&self) -> u64 {
        self.anomaly_history.len() as u64 + 1
    }

    /// Creates anomaly context
    fn create_anomaly_context(
        &self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<AnomalyContext<A>, String> {
        // Calculate recent statistics from the retained window.
        let recent_statistics = self.calculate_recent_statistics(data_point)?;

        // Calculate time since last anomaly. With no prior anomaly there is no
        // interval to report, so use the age of the oldest retained
        // observation — a real measurement — rather than a fixed hour.
        let time_since_last_anomaly = match self.anomaly_history.back() {
            Some(last_anomaly) => last_anomaly.timestamp.elapsed(),
            None => self
                .recent_points
                .front()
                .map(|point| point.timestamp.elapsed())
                .unwrap_or(Duration::ZERO),
        };

        Ok(AnomalyContext {
            recent_statistics,
            // Reported by the owning optimizer through
            // `update_context_signals`; empty means "not reported", which is
            // honest, unlike the 0.8/0.7, 0.6/0.5, 0.1 placeholders this used
            // to invent.
            performance_metrics: self.context_performance_metrics.clone(),
            resource_usage: self.context_resource_usage.clone(),
            drift_indicators: self.context_drift_indicators.clone(),
            time_since_last_anomaly,
        })
    }

    /// Computes per-feature statistics over the retained window of recent data
    /// points.
    ///
    /// The summary is dimension-general: it is sized from the widest feature
    /// vector actually observed (including the point being classified), so a
    /// stream with any number of features is described correctly. Every
    /// quantity — mean, standard deviation, min, max, median, skewness and
    /// kurtosis — is computed from the real observations.
    fn calculate_recent_statistics(
        &self,
        data_point: &StreamingDataPoint<A>,
    ) -> Result<DataStatistics<A>, String> {
        let feature_count = self
            .recent_points
            .iter()
            .map(|point| point.features.len())
            .chain(std::iter::once(data_point.features.len()))
            .max()
            .unwrap_or(0);

        let mut means = Vec::with_capacity(feature_count);
        let mut std_devs = Vec::with_capacity(feature_count);
        let mut min_values = Vec::with_capacity(feature_count);
        let mut max_values = Vec::with_capacity(feature_count);
        let mut medians = Vec::with_capacity(feature_count);
        let mut skewness = Vec::with_capacity(feature_count);
        let mut kurtosis = Vec::with_capacity(feature_count);

        for index in 0..feature_count {
            let mut column: Vec<A> = self
                .recent_points
                .iter()
                .filter_map(|point| point.features.get(index).copied())
                .collect();
            if let Some(value) = data_point.features.get(index) {
                column.push(*value);
            }

            if column.is_empty() {
                means.push(A::zero());
                std_devs.push(A::zero());
                min_values.push(A::zero());
                max_values.push(A::zero());
                medians.push(A::zero());
                skewness.push(A::zero());
                kurtosis.push(A::zero());
                continue;
            }

            let count = A::from(column.len())
                .ok_or_else(|| format!("sample count {} is not representable", column.len()))?;
            let mean = column.iter().fold(A::zero(), |acc, &v| acc + v) / count;
            let variance = column
                .iter()
                .fold(A::zero(), |acc, &v| acc + (v - mean) * (v - mean))
                / count;
            let std_dev = variance.sqrt();

            let minimum = column
                .iter()
                .copied()
                .reduce(|a, b| {
                    if super::statistics::total_order(&b, &a) == std::cmp::Ordering::Less {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or_else(A::zero);
            let maximum = column
                .iter()
                .copied()
                .reduce(|a, b| {
                    if super::statistics::total_order(&b, &a) == std::cmp::Ordering::Greater {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or_else(A::zero);
            let median = super::statistics::median_in_place(&mut column).unwrap_or(mean);

            // Standardised third and fourth central moments. Both are
            // undefined for a constant column, which is reported as zero
            // (a symmetric, mesokurtic degenerate distribution) rather than a
            // division by zero.
            let (skew, kurt) = if std_dev > A::zero() {
                let mut third = A::zero();
                let mut fourth = A::zero();
                for &value in column.iter() {
                    let z = (value - mean) / std_dev;
                    let z2 = z * z;
                    third = third + z2 * z;
                    fourth = fourth + z2 * z2;
                }
                let three = A::from(3.0).ok_or_else(|| "3.0 is not representable".to_string())?;
                // Excess kurtosis, so a Gaussian column reports ~0.
                (third / count, fourth / count - three)
            } else {
                (A::zero(), A::zero())
            };

            means.push(mean);
            std_devs.push(std_dev);
            min_values.push(minimum);
            max_values.push(maximum);
            medians.push(median);
            skewness.push(skew);
            kurtosis.push(kurt);
        }

        Ok(DataStatistics {
            means,
            std_devs,
            min_values,
            max_values,
            medians,
            skewness,
            kurtosis,
        })
    }

    /// Records anomaly in history
    fn record_anomaly(&mut self, anomaly_event: AnomalyEvent<A>) -> Result<(), String> {
        if self.anomaly_history.len() >= 10000 {
            self.anomaly_history.pop_front();
        }
        self.anomaly_history.push_back(anomaly_event);
        Ok(())
    }

    /// Applies adaptation to anomaly detection parameters
    pub fn apply_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::AnomalyDetection {
            // Adjust detection thresholds
            let threshold_adjustment = adaptation.magnitude;

            for detector in self.statistical_detectors.values_mut() {
                let current_threshold = detector.get_threshold();
                let new_threshold = current_threshold + threshold_adjustment;
                detector.set_threshold(new_threshold);
            }

            // Update ensemble configuration
            self.ensemble_detector
                .adjust_sensitivity(threshold_adjustment)?;
        }

        Ok(())
    }

    /// Gets recent anomaly events
    pub fn get_recent_anomalies(&self, count: usize) -> Vec<&AnomalyEvent<A>> {
        self.anomaly_history.iter().rev().take(count).collect()
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> AnomalyDiagnostics {
        AnomalyDiagnostics {
            total_anomalies: self.anomaly_history.len(),
            recent_anomaly_rate: self.calculate_recent_anomaly_rate(),
            false_positive_rate: self.false_positive_tracker.get_current_fp_rate(),
            detector_count: self.statistical_detectors.len() + self.ml_detectors.len(),
            response_success_rate: self.response_system.get_success_rate(),
            response_executions: self.response_system.execution_count(),
            recent_window_len: self.recent_points.len(),
        }
    }

    /// Calculates recent anomaly rate
    fn calculate_recent_anomaly_rate(&self) -> f64 {
        let recent_window = Duration::from_secs(3600); // 1 hour
                                                       // `Instant::now() - Duration` panics if the process has been up for
                                                       // less than the window, so subtract with a checked operation.
        let now = Instant::now();

        let recent_count = self
            .anomaly_history
            .iter()
            .filter(|event| now.duration_since(event.timestamp) <= recent_window)
            .count();

        recent_count as f64 / recent_window.as_secs_f64() // Anomalies per second
    }
}

// The classic statistical detectors (z-score and IQR) live in
// `super::anomaly_statistical`; the real machine-learning detectors (isolation
// forest, online one-class SVM and local outlier factor) live in
// `super::anomaly_ml`, where the latter three were previously stubs here that
// returned the constants 0.3 / 0.2 / 0.1 regardless of their input.

// The real machine-learning detectors (isolation forest, online one-class SVM
// and local outlier factor) live in `super::anomaly_ml`; they were previously
// three stubs here that returned the constants 0.3 / 0.2 / 0.1 regardless of
// their input.

// Simplified implementations for supporting structures

impl<A: Float + Default + Clone + Send + Sync + Send + Sync> AdaptiveThresholdManager<A> {
    fn new() -> Result<Self, String> {
        Ok(Self {
            thresholds: HashMap::new(),
            threshold_bounds: HashMap::new(),
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + Send + Sync> FalsePositiveTracker<A> {
    fn new() -> Self {
        Self {
            false_positives: VecDeque::with_capacity(1000),
            fp_rate_calculator: FPRateCalculator {
                recent_results: VecDeque::with_capacity(1000),
                window_size: 1000,
                current_fp_rate: scalar_or(0.05, A::zero()),
            },
        }
    }

    /// Observed false-positive rate, or `None` before any labelled outcome has
    /// been recorded.
    ///
    /// The constructor seeds `current_fp_rate` with the *target* rate, so
    /// reporting it unconditionally would present a configuration value as a
    /// measurement.
    fn get_current_fp_rate(&self) -> Option<f64> {
        if self.fp_rate_calculator.recent_results.is_empty() {
            return None;
        }
        self.fp_rate_calculator.current_fp_rate.to_f64()
    }

    /// Records ground truth for one prediction and recomputes the observed
    /// false-positive rate over the sliding evaluation window.
    fn record_outcome(&mut self, predicted_anomaly: bool, was_true_anomaly: bool) {
        let calculator = &mut self.fp_rate_calculator;
        if calculator.recent_results.len() >= calculator.window_size {
            calculator.recent_results.pop_front();
        }
        calculator.recent_results.push_back(DetectionResult {
            timestamp: Instant::now(),
            anomaly_detected: predicted_anomaly,
            ground_truth: Some(was_true_anomaly),
            detector_name: "ensemble".to_string(),
        });

        // The false-positive rate is `FP / (FP + TN)`: the fraction of
        // genuinely normal points that were incorrectly flagged.
        let negatives = calculator
            .recent_results
            .iter()
            .filter(|result| result.ground_truth == Some(false))
            .count();
        if negatives > 0 {
            let false_positives = calculator
                .recent_results
                .iter()
                .filter(|result| result.anomaly_detected && result.ground_truth == Some(false))
                .count();
            if let Some(rate) = A::from(false_positives as f64 / negatives as f64) {
                calculator.current_fp_rate = rate;
            }
        }
    }

    /// Records the full detail of a confirmed false positive.
    fn record_false_positive(&mut self, event: &AnomalyEvent<A>) {
        if self.false_positives.len() >= RESPONSE_HISTORY_CAPACITY {
            self.false_positives.pop_front();
        }
        self.false_positives.push_back(FalsePositiveEvent {
            timestamp: event.timestamp,
            data_point: event.data_point.clone(),
            detector_name: event.detector_name.clone(),
            anomaly_score: event.anomaly_score,
            context: event.context.clone(),
        });
    }

    /// Number of confirmed false positives retained.
    fn confirmed_false_positive_count(&self) -> usize {
        self.false_positives.len()
    }
}

impl<A: Float + Default + Clone + Send + Sync + Send + Sync> AnomalyResponseSystem<A> {
    fn new(response_strategy: &AnomalyResponseStrategy) -> Result<Self, String> {
        let mut response_strategies = HashMap::new();

        // Set up default response strategies
        match response_strategy {
            AnomalyResponseStrategy::Ignore => {
                response_strategies
                    .insert(AnomalyType::StatisticalOutlier, vec![ResponseAction::Log]);
            }
            AnomalyResponseStrategy::Filter => {
                response_strategies.insert(
                    AnomalyType::StatisticalOutlier,
                    vec![ResponseAction::Quarantine],
                );
            }
            AnomalyResponseStrategy::Adaptive => {
                response_strategies.insert(
                    AnomalyType::StatisticalOutlier,
                    vec![ResponseAction::Log, ResponseAction::ModelAdjustment],
                );
            }
            _ => {
                response_strategies
                    .insert(AnomalyType::StatisticalOutlier, vec![ResponseAction::Alert]);
            }
        }

        Ok(Self {
            response_strategies,
            response_executor: ResponseExecutor {
                pending_responses: VecDeque::new(),
                execution_history: VecDeque::with_capacity(1000),
                resource_limits: ResponseResourceLimits {
                    max_concurrent_responses: 10,
                    max_cpu_usage: 0.2,
                    max_memory_usage: 100 * 1024 * 1024, // 100MB
                    max_execution_time: Duration::from_secs(60),
                },
            },
            next_response_id: 0,
            log_entries: VecDeque::with_capacity(RESPONSE_HISTORY_CAPACITY),
            alert_entries: VecDeque::with_capacity(RESPONSE_HISTORY_CAPACITY),
            quarantined_points: VecDeque::with_capacity(QUARANTINE_CAPACITY),
            pending_threshold_adjustment: None,
            monitoring_level: 0,
        })
    }

    /// Queues and executes the responses configured for this anomaly type,
    /// returning the names of the actions that actually ran.
    ///
    /// Every action either performs a real, observable state change (a log or
    /// alert entry, a quarantined data point, a queued threshold adjustment, a
    /// raised monitoring level) or is recorded as an honest failure with the
    /// reason — there is no subsystem behind `TriggerRecovery` or a custom
    /// action, so claiming success for them would be a fabrication.
    fn trigger_response(&mut self, event: &AnomalyEvent<A>) -> Result<Vec<String>, String> {
        let actions = self
            .response_strategies
            .get(&event.anomaly_type)
            .or_else(|| {
                self.response_strategies
                    .get(&AnomalyType::StatisticalOutlier)
            })
            .cloned()
            .unwrap_or_default();

        if actions.is_empty() {
            return Ok(Vec::new());
        }

        let priority = match event.severity {
            AnomalySeverity::Critical => ResponsePriority::Critical,
            AnomalySeverity::High => ResponsePriority::High,
            AnomalySeverity::Medium => ResponsePriority::Normal,
            AnomalySeverity::Low => ResponsePriority::Low,
        };
        let timeout = self.response_executor.resource_limits.max_execution_time;

        for action in actions {
            if self.response_executor.pending_responses.len()
                >= self
                    .response_executor
                    .resource_limits
                    .max_concurrent_responses
            {
                // Respect the configured concurrency limit instead of growing
                // an unbounded queue.
                break;
            }
            self.next_response_id += 1;
            self.response_executor
                .pending_responses
                .push_back(PendingResponse {
                    id: self.next_response_id,
                    anomaly_event: event.clone(),
                    action,
                    priority: priority.clone(),
                    scheduled_time: Instant::now(),
                    timeout,
                });
        }

        self.execute_pending_responses()
    }

    /// Drains the pending queue, highest priority first, executing each action.
    fn execute_pending_responses(&mut self) -> Result<Vec<String>, String> {
        // Highest priority first; the queue is small (bounded by
        // `max_concurrent_responses`) so a sort is cheap.
        self.response_executor
            .pending_responses
            .make_contiguous()
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut executed = Vec::new();
        while let Some(pending) = self.response_executor.pending_responses.pop_front() {
            let started = Instant::now();
            let outcome = self.perform_action(&pending);
            let duration = started.elapsed();

            let (success, error_message) = match &outcome {
                Ok(()) => (true, None),
                Err(reason) => (false, Some(reason.clone())),
            };
            if success {
                executed.push(format!("{:?}", pending.action));
            }

            let mut resources_consumed = HashMap::new();
            if let Some(millis) = A::from(duration.as_secs_f64() * 1000.0) {
                resources_consumed.insert("execution_time_ms".to_string(), millis);
            }

            let execution = ResponseExecution {
                id: pending.id,
                response: pending,
                start_time: started,
                duration,
                success,
                error_message,
                resources_consumed,
            };

            if self.response_executor.execution_history.len() >= RESPONSE_HISTORY_CAPACITY {
                self.response_executor.execution_history.pop_front();
            }
            self.response_executor
                .execution_history
                .push_back(execution);
        }

        Ok(executed)
    }

    /// Carries out a single response action.
    fn perform_action(&mut self, pending: &PendingResponse<A>) -> Result<(), String> {
        match &pending.action {
            ResponseAction::Log => {
                self.push_bounded(
                    ResponseChannel::Log,
                    format!(
                        "anomaly {} type={:?} severity={:?} score={:?}",
                        pending.anomaly_event.id,
                        pending.anomaly_event.anomaly_type,
                        pending.anomaly_event.severity,
                        pending.anomaly_event.anomaly_score.to_f64()
                    ),
                );
                Ok(())
            }
            ResponseAction::Alert => {
                self.push_bounded(
                    ResponseChannel::Alert,
                    format!(
                        "ALERT: anomaly {} severity={:?}",
                        pending.anomaly_event.id, pending.anomaly_event.severity
                    ),
                );
                Ok(())
            }
            ResponseAction::Quarantine => {
                if self.quarantined_points.len() >= QUARANTINE_CAPACITY {
                    self.quarantined_points.pop_front();
                }
                self.quarantined_points
                    .push_back(pending.anomaly_event.data_point.clone());
                Ok(())
            }
            ResponseAction::ModelAdjustment => {
                // Ask the owning detector to raise its thresholds in
                // proportion to the severity of what got through. The caller
                // applies this via `take_pending_threshold_adjustment`, so the
                // adjustment is a real state change rather than a log line.
                let step = match pending.anomaly_event.severity {
                    AnomalySeverity::Critical => 0.20,
                    AnomalySeverity::High => 0.10,
                    AnomalySeverity::Medium => 0.05,
                    AnomalySeverity::Low => 0.01,
                };
                self.pending_threshold_adjustment =
                    Some(self.pending_threshold_adjustment.unwrap_or(0.0) + step);
                Ok(())
            }
            ResponseAction::IncreaseMonitoring => {
                self.monitoring_level = self.monitoring_level.saturating_add(1);
                Ok(())
            }
            ResponseAction::TriggerRecovery => Err(
                "no recovery procedure is registered with this response system; \
                 a recovery handler must be installed before this action can run"
                    .to_string(),
            ),
            ResponseAction::Custom(name) => Err(format!(
                "no handler is registered for custom response action '{name}'"
            )),
        }
    }

    fn push_bounded(&mut self, channel: ResponseChannel, message: String) {
        let sink = match channel {
            ResponseChannel::Log => &mut self.log_entries,
            ResponseChannel::Alert => &mut self.alert_entries,
        };
        if sink.len() >= RESPONSE_HISTORY_CAPACITY {
            sink.pop_front();
        }
        sink.push_back(message);
    }

    /// Takes any threshold adjustment the responses requested, clearing it.
    fn take_pending_threshold_adjustment(&mut self) -> Option<f64> {
        self.pending_threshold_adjustment.take()
    }

    /// Fraction of executed responses that succeeded, or `None` when nothing
    /// has been executed yet.
    ///
    /// Returning `None` is the honest answer for an empty history; the previous
    /// implementation reported a hard-coded `0.85` from the moment the system
    /// was constructed.
    fn get_success_rate(&self) -> Option<f64> {
        let history = &self.response_executor.execution_history;
        if history.is_empty() {
            return None;
        }
        let successes = history.iter().filter(|execution| execution.success).count();
        Some(successes as f64 / history.len() as f64)
    }

    /// Entries written by executed `Log` actions.
    fn log_entry_count(&self) -> usize {
        self.log_entries.len()
    }

    /// Entries written by executed `Alert` actions.
    fn alert_entry_count(&self) -> usize {
        self.alert_entries.len()
    }

    /// Data points held by executed `Quarantine` actions.
    fn quarantined_count(&self) -> usize {
        self.quarantined_points.len()
    }

    /// Current monitoring level, raised by `IncreaseMonitoring` actions.
    fn monitoring_level(&self) -> u32 {
        self.monitoring_level
    }

    /// Number of response executions recorded.
    fn execution_count(&self) -> usize {
        self.response_executor.execution_history.len()
    }
}

/// Sinks that a response action can write to.
#[derive(Debug, Clone, Copy)]
enum ResponseChannel {
    Log,
    Alert,
}

/// Diagnostic information for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDiagnostics {
    /// Anomalies retained in the history buffer.
    pub total_anomalies: usize,
    /// Anomalies per second over the last hour.
    pub recent_anomaly_rate: f64,
    /// Observed false-positive rate, or `None` before any labelled outcome has
    /// been recorded (it is genuinely unmeasurable until then).
    pub false_positive_rate: Option<f64>,
    /// Number of registered detectors.
    pub detector_count: usize,
    /// Fraction of executed responses that succeeded, or `None` before any
    /// response has run.
    pub response_success_rate: Option<f64>,
    /// Number of response executions recorded.
    pub response_executions: usize,
    /// Data points currently retained for context statistics.
    pub recent_window_len: usize,
}

#[cfg(test)]
#[path = "anomaly_detection_regression_tests.rs"]
mod regression_tests;
