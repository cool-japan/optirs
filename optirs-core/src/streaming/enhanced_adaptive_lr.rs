// Enhanced adaptive learning rate mechanisms for streaming optimization
//
// This module provides advanced adaptive learning rate controllers that can
// dynamically adjust learning rates based on multiple signals including
// gradient statistics, performance metrics, concept drift, and resource constraints.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::utils::scalar_or;

/// Performance metric types for adaptation
#[derive(Debug, Clone)]
pub enum PerformanceMetric<A: Float + Send + Sync> {
    Loss(A),
    Accuracy(A),
    F1Score(A),
    AUC(A),
    Custom { name: String, value: A },
}

/// Enhanced adaptive learning rate controller with multiple adaptation mechanisms
#[derive(Debug, Clone)]
pub struct EnhancedAdaptiveLRController<A: Float + Send + Sync> {
    /// Current learning rate
    current_lr: A,

    /// Base learning rate
    base_lr: A,

    /// Multi-signal adaptation strategy
    adaptation_strategy: MultiSignalAdaptationStrategy<A>,

    /// Gradient-based adaptation state
    gradient_adapter: GradientBasedAdapter<A>,

    /// Performance-based adaptation state
    performance_adapter: PerformanceBasedAdapter<A>,

    /// Drift-aware adaptation
    drift_adapter: DriftAwareAdapter<A>,

    /// Resource-aware adaptation
    resource_adapter: ResourceAwareAdapter<A>,

    /// Meta-learning for hyperparameter optimization
    meta_optimizer: MetaOptimizer<A>,

    /// Adaptation history for analysis
    adaptation_history: VecDeque<AdaptationEvent<A>>,

    /// Configuration
    config: AdaptiveLRConfig<A>,
}

/// Configuration for adaptive learning rate controller
#[derive(Debug, Clone)]
pub struct AdaptiveLRConfig<A: Float + Send + Sync> {
    /// Base learning rate
    pub base_lr: A,

    /// Minimum allowed learning rate
    pub min_lr: A,

    /// Maximum allowed learning rate  
    pub max_lr: A,

    /// Enable gradient-based adaptation
    pub enable_gradient_adaptation: bool,

    /// Enable performance-based adaptation
    pub enable_performance_adaptation: bool,

    /// Enable drift-aware adaptation
    pub enable_drift_adaptation: bool,

    /// Enable resource-aware adaptation
    pub enable_resource_adaptation: bool,

    /// Enable meta-learning optimization
    pub enable_meta_learning: bool,

    /// History window size
    pub history_window_size: usize,

    /// Adaptation frequency (steps)
    pub adaptation_frequency: usize,

    /// Sensitivity to changes
    pub adaptation_sensitivity: A,

    /// Use ensemble voting for conflicting signals
    pub use_ensemble_voting: bool,

    /// Wall-clock budget for a single optimizer step, when the deployment has
    /// one. Without it there is nothing to measure time pressure against, so
    /// the resource signal reports no time term rather than inventing one.
    pub step_time_budget: Option<Duration>,

    /// Memory budget in MB, when the deployment has one.
    pub memory_budget_mb: Option<f64>,
}

/// Multi-signal adaptation strategy
#[derive(Debug, Clone)]
pub struct MultiSignalAdaptationStrategy<A: Float + Send + Sync> {
    /// Weighted voting system for adaptation signals
    pub(crate) signal_weights: HashMap<AdaptationSignalType, A>,

    /// Signal voting history
    pub(crate) voting_history: VecDeque<SignalVote<A>>,

    /// Conflict resolution method
    pub(crate) conflict_resolution: ConflictResolution,

    /// Signal reliability scores
    pub(crate) signal_reliability: HashMap<AdaptationSignalType, A>,

    /// Last adaptation decision
    pub(crate) last_decision: Option<AdaptationDecision<A>>,
}

/// Types of adaptation signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptationSignalType {
    GradientMagnitude,
    GradientVariance,
    LossProgression,
    AccuracyTrend,
    ConceptDrift,
    ResourceUtilization,
    ModelComplexity,
    DataQuality,
}

/// Signal vote for learning rate adaptation
#[derive(Debug, Clone)]
pub struct SignalVote<A: Float + Send + Sync> {
    signal_type: AdaptationSignalType,
    recommended_lr_change: A, // Multiplier (1.0 = no change)
    confidence: A,
    reasoning: String,
    timestamp: Instant,
}

impl<A: Float + Send + Sync> SignalVote<A> {
    /// Which signal cast this vote.
    pub fn signal_type(&self) -> AdaptationSignalType {
        self.signal_type
    }

    /// Multiplier this signal recommends applying to the learning rate.
    pub fn recommended_lr_change(&self) -> A {
        self.recommended_lr_change
    }

    /// Confidence the signal attaches to its recommendation.
    pub fn confidence(&self) -> A {
        self.confidence
    }

    /// Human-readable justification the signal produced for this vote.
    ///
    /// Each adapter builds this string from the statistics it actually
    /// measured; it had no reader outside the tests, so callers had no way to
    /// see *why* a learning-rate change was proposed.
    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    /// When the vote was cast.
    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }
}

/// Conflict resolution methods for contradictory signals
#[derive(Debug, Clone, Copy)]
pub enum ConflictResolution {
    /// Use weighted average of all signals
    WeightedAverage,
    /// Use signal with highest confidence
    HighestConfidence,
    /// Use majority vote (requires threshold)
    MajorityVote { threshold: f64 },
    /// Use conservative approach (smallest change)
    Conservative,
    /// Use meta-learning to resolve conflicts
    MetaLearned,
}

/// Adaptation decision with rationale
#[derive(Debug, Clone)]
pub struct AdaptationDecision<A: Float + Send + Sync> {
    new_lr: A,
    lr_multiplier: A,
    contributing_signals: Vec<AdaptationSignalType>,
    confidence: A,
    rationale: String,
    timestamp: Instant,
}

impl<A: Float + Send + Sync> AdaptationDecision<A> {
    /// Learning rate this decision settled on.
    pub fn new_lr(&self) -> A {
        self.new_lr
    }

    /// Multiplier applied to the previous learning rate.
    pub fn lr_multiplier(&self) -> A {
        self.lr_multiplier
    }

    /// Signals that contributed to the decision.
    pub fn contributing_signals(&self) -> &[AdaptationSignalType] {
        &self.contributing_signals
    }

    /// Confidence in the decision, aggregated across contributing signals.
    pub fn confidence(&self) -> A {
        self.confidence
    }

    /// Human-readable explanation of how the contributing signals were
    /// reconciled, built by the configured conflict-resolution rule.
    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    /// When the decision was taken.
    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }
}

/// Gradient-based adaptation using statistical analysis
#[derive(Debug, Clone)]
pub struct GradientBasedAdapter<A: Float + Send + Sync> {
    /// Gradient magnitude history
    magnitude_history: VecDeque<A>,

    /// Gradient direction variance
    direction_variance_history: VecDeque<A>,

    /// Gradient norm statistics
    norm_statistics: GradientNormStatistics<A>,

    /// Signal-to-noise ratio estimation
    snr_estimator: SignalToNoiseEstimator<A>,

    /// Gradient staleness detection
    staleness_detector: GradientStalenessDetector,
}

/// Performance-based adaptation using multiple metrics
#[derive(Debug, Clone)]
pub struct PerformanceBasedAdapter<A: Float + Send + Sync> {
    /// Performance metric history
    metric_history: HashMap<String, VecDeque<A>>,

    /// Performance trend analysis
    trend_analyzer: PerformanceTrendAnalyzer<A>,

    /// Plateau detection
    plateau_detector: PlateauDetector<A>,

    /// Overfitting detection
    overfitting_detector: OverfittingDetector<A>,

    /// Learning efficiency tracker
    efficiency_tracker: LearningEfficiencyTracker<A>,
}

/// Drift-aware adaptation for non-stationary data
#[derive(Debug, Clone)]
pub struct DriftAwareAdapter<A: Float + Send + Sync> {
    /// Concept drift detection methods
    drift_detectors: Vec<ConceptDriftDetector<A>>,

    /// Data distribution shift detection
    distribution_tracker: DistributionTracker<A>,

    /// Adaptation speed controller
    adaptation_speed: AdaptationSpeedController<A>,

    /// Drift severity assessment
    drift_severity: DriftSeverityAssessor<A>,
}

/// Resource-aware adaptation based on computational constraints
#[derive(Debug, Clone)]
pub struct ResourceAwareAdapter<A: Float + Send + Sync> {
    /// Memory usage tracker
    memory_tracker: MemoryUsageTracker,

    /// Computation time tracker
    compute_tracker: ComputationTimeTracker,

    /// Energy consumption tracker
    energy_tracker: EnergyConsumptionTracker,

    /// Throughput requirements
    throughput_requirements: ThroughputRequirements<A>,

    /// Resource budget manager
    budget_manager: ResourceBudgetManager<A>,
}

/// Meta-learning optimizer for hyperparameter adaptation
#[derive(Debug, Clone)]
pub struct MetaOptimizer<A: Float + Send + Sync> {
    /// Hyperparameter optimization history
    optimization_history: VecDeque<HyperparameterUpdate<A>>,

    /// Multi-armed bandit for exploration
    exploration_strategy: ExplorationStrategy<A>,

    /// Transfer learning from similar tasks
    transfer_learner: TransferLearner<A>,
}

/// Adaptation event for tracking and analysis
#[derive(Debug, Clone)]
pub struct AdaptationEvent<A: Float + Send + Sync> {
    timestamp: Instant,
    old_lr: A,
    new_lr: A,
    trigger_signals: Vec<AdaptationSignalType>,
    effectiveness_score: Option<A>, // Measured retrospectively
}

/// Gradient norm statistics for adaptation
#[derive(Debug, Clone)]
pub struct GradientNormStatistics<A: Float + Send + Sync> {
    mean: A,
    variance: A,
    skewness: A,
    kurtosis: A,
    percentiles: Vec<A>, // 5th, 25th, 50th, 75th, 95th
    autocorrelation: A,
}

/// Signal-to-noise ratio estimation for gradients
#[derive(Debug, Clone)]
pub struct SignalToNoiseEstimator<A: Float + Send + Sync> {
    signal_estimate: A,
    noise_estimate: A,
    snr_history: VecDeque<A>,
}

#[derive(Debug, Clone, Copy)]
pub enum SNREstimationMethod {
    MovingAverage,
    ExponentialSmoothing,
    RobustEstimation,
    WaveletDenoising,
}

/// Gradient staleness detection for distributed settings
#[derive(Debug, Clone, Default)]
pub struct GradientStalenessDetector {
    gradient_timestamps: VecDeque<Instant>,
}

/// Performance trend analysis for learning rate adaptation
#[derive(Debug, Clone)]
pub struct PerformanceTrendAnalyzer<A: Float + Send + Sync> {
    trend_detection_window: usize,
    trend_types: Vec<TrendType>,
    trend_strength: A,
}

#[derive(Debug, Clone, Copy)]
pub enum TrendType {
    Improving,
    Degrading,
    Oscillating,
    Plateau,
    Volatile,
}

/// Plateau detection in learning curves
#[derive(Debug, Clone)]
pub struct PlateauDetector<A: Float + Send + Sync> {
    plateau_threshold: A,
    min_plateau_duration: usize,
    current_plateau_length: usize,
    plateau_confidence: A,
}

/// Overfitting detection mechanism
#[derive(Debug, Clone)]
pub struct OverfittingDetector<A: Float + Send + Sync> {
    train_loss_history: VecDeque<A>,
    val_loss_history: VecDeque<A>,
}

/// Learning efficiency tracking
#[derive(Debug, Clone)]
pub struct LearningEfficiencyTracker<A: Float + Send + Sync> {
    loss_reduction_per_step: VecDeque<A>,
    efficiency_score: A,
    efficiency_trend: TrendType,
}

/// Concept drift detection over the loss/gradient stream.
///
/// E3: this struct had no methods at all and `DriftAwareAdapter::drift_detectors`
/// was `vec![]`, so `generate_signal` returned a hardcoded "No drift detected"
/// vote on every call. It now delegates to the real detectors implemented in
/// [`crate::streaming::concept_drift`] rather than reimplementing them.
#[derive(Debug, Clone)]
pub struct ConceptDriftDetector<A: Float + Send + Sync> {
    pub(crate) detection_method: DriftDetectionMethod,
    pub(crate) drift_threshold: A,
    pub(crate) window_size: usize,
    pub(crate) drift_confidence: A,
    pub(crate) last_drift_time: Option<Instant>,
    pub(crate) inner: LossDriftDetector<A>,
}

#[derive(Debug, Clone, Copy)]
pub enum DriftDetectionMethod {
    ADWIN,
    DDM,
    EDDM,
    PageHinkley,
    KSWIN,
    Statistical,
}

/// Data distribution tracking
#[derive(Debug, Clone)]
pub struct DistributionTracker<A: Float + Send + Sync> {
    feature_distributions: HashMap<usize, FeatureDistribution<A>>,
    distribution_drift_score: A,
}

#[derive(Debug, Clone)]
pub struct FeatureDistribution<A: Float + Send + Sync> {
    mean: A,
    variance: A,
    histogram: Vec<A>,
    last_update: Instant,
}

/// Adaptation speed controller for drift response
#[derive(Debug, Clone)]
pub struct AdaptationSpeedController<A: Float + Send + Sync> {
    base_adaptation_rate: A,
    current_adaptation_rate: A,
    acceleration_factor: A,
    deceleration_factor: A,
    momentum: A,
}

/// Drift severity assessment
#[derive(Debug, Clone)]
pub struct DriftSeverityAssessor<A: Float + Send + Sync> {
    severity_levels: Vec<DriftSeverityLevel<A>>,
    current_severity: DriftSeverityLevel<A>,
    severity_history: VecDeque<DriftSeverityLevel<A>>,
}

#[derive(Debug, Clone)]
pub struct DriftSeverityLevel<A: Float + Send + Sync> {
    level: DriftSeverity,
    recommended_lr_adjustment: A,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftSeverity {
    None,
    Mild,
    Moderate,
    Severe,
    Critical,
}

/// Resource usage tracking components
#[derive(Debug, Clone, Default)]
pub struct MemoryUsageTracker {
    pub(crate) current_usage_mb: f64,
    pub(crate) peak_usage_mb: f64,
    pub(crate) usage_history: VecDeque<f64>,
    /// E4: `memory_pressure` was never written to, so the resource signal
    /// always read a 0.0 that it then interpreted as "plenty of head-room" and
    /// pushed the learning rate *up*. It is now `None` until a real usage
    /// figure and a budget are both available.
    pub(crate) memory_pressure: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ComputationTimeTracker {
    pub(crate) step_times: VecDeque<Duration>,
    pub(crate) average_step_time: Duration,
    pub(crate) time_budget: Option<Duration>,
    /// `None` until both a measured step time and a configured budget exist.
    pub(crate) time_pressure: Option<f64>,
}

/// Energy consumption tracker.
///
/// There is no portable, pure-Rust way to read energy draw, so every field here
/// stays empty unless a caller feeds real measurements through
/// [`EnhancedAdaptiveLRController::record_energy_sample`].
#[derive(Debug, Clone, Default)]
pub struct EnergyConsumptionTracker {
    pub(crate) energy_per_step: VecDeque<f64>,
    pub(crate) cumulative_energy: f64,
    pub(crate) energy_efficiency: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ThroughputRequirements<A: Float + Send + Sync> {
    min_samples_per_second: A,
    current_throughput: A,
    throughput_deficit: A,
}

#[derive(Debug, Clone)]
pub struct ResourceBudgetManager<A: Float + Send + Sync> {
    memory_budget_mb: f64,
    compute_budget_seconds: f64,
    budget_utilization: A,
    budget_violations: usize,
}

/// Hyperparameter update record
#[derive(Debug, Clone)]
pub struct HyperparameterUpdate<A: Float + Send + Sync> {
    features: Array1<A>,
    reward: A, // Performance improvement
}

/// Exploration strategy for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct ExplorationStrategy<A: Float + Send + Sync> {
    exploration_rate: A,
    arm_rewards: HashMap<usize, A>,
    arm_counts: HashMap<usize, usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum ExplorationStrategyType {
    EpsilonGreedy,
    UCB1,
    ThompsonSampling,
    LinUCB,
    ContextualBandit,
}

/// Transfer learning for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct TransferLearner<A: Float + Send + Sync> {
    source_task_data: Vec<TaskData<A>>,
    transfer_confidence: A,
}

#[derive(Debug, Clone)]
pub struct TaskData<A: Float + Send + Sync> {
    optimal_lr_sequence: Vec<A>,
}

/// Adaptation statistics for monitoring and analysis
#[derive(Debug, Clone, Default)]
pub struct AdaptationStatistics<A: Float + Send + Sync> {
    /// Total number of adaptations
    pub total_adaptations: usize,

    /// Successful adaptations (led to improvement)
    pub successful_adaptations: usize,

    /// Average adaptation frequency
    pub avg_adaptation_frequency: A,

    /// Learning rate volatility
    pub lr_volatility: A,

    /// Signal reliability scores
    pub signal_reliability_scores: HashMap<AdaptationSignalType, A>,

    /// Adaptation effectiveness by signal type
    pub signal_effectiveness: HashMap<AdaptationSignalType, A>,

    /// Resource efficiency improvements
    pub resource_efficiency_gains: A,

    /// Convergence speed improvement
    pub convergence_speed_improvement: A,
}

impl<A: Float + Default + Clone + std::iter::Sum + Send + Sync> EnhancedAdaptiveLRController<A> {
    /// Create a new enhanced adaptive learning rate controller
    pub fn new(config: AdaptiveLRConfig<A>) -> Result<Self> {
        let adaptation_strategy = MultiSignalAdaptationStrategy::new(&config)?;
        let gradient_adapter = GradientBasedAdapter::new(&config)?;
        let performance_adapter = PerformanceBasedAdapter::new(&config)?;
        let drift_adapter = DriftAwareAdapter::new(&config)?;
        let resource_adapter = ResourceAwareAdapter::new(&config)?;
        let meta_optimizer = MetaOptimizer::new(&config)?;

        Ok(Self {
            current_lr: config.base_lr,
            base_lr: config.base_lr,
            adaptation_strategy,
            gradient_adapter,
            performance_adapter,
            drift_adapter,
            resource_adapter,
            meta_optimizer,
            adaptation_history: VecDeque::with_capacity(config.history_window_size),
            config,
        })
    }

    /// Report the memory this optimizer's workload is currently using, so the
    /// resource signal has a real figure to work from (E4).
    pub fn record_memory_usage_mb(&mut self, usage_mb: f64) {
        self.resource_adapter
            .record_memory_usage(usage_mb, self.config.memory_budget_mb);
    }

    /// Report measured energy consumption for the most recent step.
    pub fn record_energy_sample(&mut self, joules: f64) {
        self.resource_adapter.record_energy(joules);
    }

    /// Report the observed sample throughput so the resource signal can compare
    /// it against the configured requirement.
    pub fn record_throughput(&mut self, samples_per_second: A) {
        self.resource_adapter.record_throughput(samples_per_second);
    }

    /// Register a previously solved task so the meta-optimizer can transfer its
    /// learning-rate schedule (it contributes nothing while none are known).
    pub fn add_source_task(&mut self, task: TaskData<A>) {
        self.meta_optimizer.add_source_task(task);
    }

    /// Update learning rate based on multiple adaptation signals
    pub fn update_learning_rate(
        &mut self,
        gradients: &Array1<A>,
        loss: A,
        metrics: &HashMap<String, A>,
        step: usize,
    ) -> Result<A> {
        let step_started = Instant::now();

        // Honour the configured adaptation cadence: outside an adaptation step
        // the learning rate is left exactly as it is (`adaptation_frequency`
        // used to be ignored entirely).
        let frequency = self.config.adaptation_frequency.max(1);
        if !step.is_multiple_of(frequency) {
            self.gradient_adapter.observe_only(gradients);
            self.performance_adapter.observe_only(loss);
            self.resource_adapter
                .record_step_time(step_started.elapsed(), self.config.step_time_budget);
            return Ok(self.current_lr);
        }

        // Collect adaptation signals from all components
        let mut signals = Vec::new();

        if self.config.enable_gradient_adaptation {
            if let Ok(signal) = self.gradient_adapter.generate_signal(gradients, step) {
                signals.push(signal);
            }
        }

        if self.config.enable_performance_adaptation {
            if let Ok(signal) = self
                .performance_adapter
                .generate_signal(loss, metrics, step)
            {
                signals.push(signal);
            }
        }

        if self.config.enable_drift_adaptation {
            if let Ok(signal) = self.drift_adapter.generate_signal(gradients, step) {
                signals.push(signal);
            }
        }

        if self.config.enable_resource_adaptation {
            if let Ok(signal) = self.resource_adapter.generate_signal(step) {
                signals.push(signal);
            }
        }

        // Resolve conflicts and make adaptation decision. E1: the resolver used
        // to ignore the live learning rate entirely and multiply the hardcoded
        // literal 0.001 by the vote, so every adaptation snapped the learning
        // rate back to ~0.001 no matter where it actually was.
        let previous_lr = self.current_lr;
        let decision = self.adaptation_strategy.resolve_signals(
            signals,
            previous_lr,
            self.config.adaptation_sensitivity,
            self.config.use_ensemble_voting,
            step,
        )?;

        // Apply meta-learning if enabled
        if self.config.enable_meta_learning {
            let meta_adjustment = self.meta_optimizer.meta_optimize(&decision, step)?;
            self.current_lr = self.apply_meta_adjustment(decision.new_lr, meta_adjustment);
        } else {
            self.current_lr = decision.new_lr;
        }

        // Ensure learning rate is within bounds
        self.current_lr = self
            .current_lr
            .clamp(self.config.min_lr, self.config.max_lr);

        // Record adaptation event
        let event = AdaptationEvent {
            timestamp: Instant::now(),
            // The learning rate *before* this adaptation; storing the decision's
            // own output here made every recorded event look like a no-op.
            old_lr: previous_lr,
            new_lr: self.current_lr,
            trigger_signals: decision.contributing_signals,
            effectiveness_score: None, // Will be updated later
        };

        self.adaptation_history.push_back(event);
        if self.adaptation_history.len() > self.config.history_window_size {
            self.adaptation_history.pop_front();
        }

        self.resource_adapter
            .record_step_time(step_started.elapsed(), self.config.step_time_budget);

        Ok(self.current_lr)
    }

    /// Learning rate before the most recent adaptation, if there was one.
    pub fn previous_lr(&self) -> Option<A> {
        self.adaptation_history.back().map(|event| event.old_lr)
    }

    /// The most recent adaptation decision, including its rationale.
    pub fn last_decision(&self) -> Option<&AdaptationDecision<A>> {
        self.adaptation_strategy.last_decision.as_ref()
    }

    /// Recorded signal votes, newest last.
    pub fn voting_history(&self) -> &VecDeque<SignalVote<A>> {
        &self.adaptation_strategy.voting_history
    }

    /// Get current learning rate
    pub fn get_current_lr(&self) -> A {
        self.current_lr
    }

    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics<A> {
        let total_adaptations = self.adaptation_history.len();
        let successful_adaptations = self
            .adaptation_history
            .iter()
            .filter(|event| {
                event
                    .effectiveness_score
                    .is_some_and(|score| score > A::zero())
            })
            .count();

        let lr_volatility = if !self.adaptation_history.is_empty() {
            let lr_values: Vec<A> = self
                .adaptation_history
                .iter()
                .map(|event| event.new_lr)
                .collect();

            let mean_lr = lr_values.iter().fold(A::zero(), |acc, &lr| acc + lr)
                / scalar_or(lr_values.len(), A::one());

            let variance = lr_values
                .iter()
                .map(|&lr| {
                    let diff = lr - mean_lr;
                    diff * diff
                })
                .fold(A::zero(), |acc, var| acc + var)
                / scalar_or(lr_values.len(), A::one());

            variance.sqrt()
        } else {
            A::zero()
        };

        // Real per-signal reliability and effectiveness rather than the empty
        // maps `..Default::default()` used to leave behind.
        let signal_reliability_scores = self.adaptation_strategy.signal_reliability.clone();
        let mut signal_effectiveness: HashMap<AdaptationSignalType, A> = HashMap::new();
        let mut signal_counts: HashMap<AdaptationSignalType, usize> = HashMap::new();
        for event in &self.adaptation_history {
            let Some(score) = event.effectiveness_score else {
                continue;
            };
            for signal_type in &event.trigger_signals {
                let count = signal_counts.entry(*signal_type).or_insert(0);
                *count += 1;
                let steps = A::from(*count).unwrap_or_else(A::one);
                let entry = signal_effectiveness
                    .entry(*signal_type)
                    .or_insert_with(A::zero);
                *entry = *entry + (score - *entry) / steps;
            }
        }

        let avg_adaptation_frequency = if let (Some(first), Some(last)) = (
            self.adaptation_history.front(),
            self.adaptation_history.back(),
        ) {
            let span = last.timestamp.saturating_duration_since(first.timestamp);
            if span.as_secs_f64() > 0.0 {
                A::from(total_adaptations as f64 / span.as_secs_f64()).unwrap_or_else(A::zero)
            } else {
                A::zero()
            }
        } else {
            A::zero()
        };

        let convergence_speed_improvement = {
            let scores: Vec<A> = self
                .adaptation_history
                .iter()
                .filter_map(|event| event.effectiveness_score)
                .collect();
            if scores.is_empty() {
                A::zero()
            } else {
                let count = A::from(scores.len()).unwrap_or_else(A::one);
                scores.iter().fold(A::zero(), |acc, score| acc + *score) / count
            }
        };

        AdaptationStatistics {
            total_adaptations,
            successful_adaptations,
            avg_adaptation_frequency,
            lr_volatility,
            signal_reliability_scores,
            signal_effectiveness,
            resource_efficiency_gains: A::from(self.resource_adapter.budget_violations() as f64)
                .map(|violations| A::zero() - violations)
                .unwrap_or_else(A::zero),
            convergence_speed_improvement,
        }
    }

    /// Apply meta-learning adjustment to base decision
    fn apply_meta_adjustment(&self, base_lr: A, meta_adjustment: A) -> A {
        // Combine base decision with meta-learning recommendation
        let alpha = scalar_or(0.7, A::zero()); // Weight for base decision
        let beta = scalar_or(0.3, A::zero()); // Weight for meta-learning

        alpha * base_lr + beta * meta_adjustment
    }

    /// Evaluate adaptation effectiveness retrospectively
    pub fn evaluate_adaptation_effectiveness(&mut self, performance_improvement: A) {
        let mut signals = Vec::new();
        if let Some(last_event) = self.adaptation_history.back_mut() {
            last_event.effectiveness_score = Some(performance_improvement);
            signals = last_event.trigger_signals.clone();
        }
        // Update signal reliability based on effectiveness
        for signal_type in signals {
            self.adaptation_strategy
                .update_signal_reliability(signal_type, performance_improvement);
        }
        // E2: the meta-optimizer's bandit learns from the same measurement, so
        // its arm values come from observed effectiveness rather than a constant.
        self.meta_optimizer.record_reward(performance_improvement);
    }

    /// Measured mean reward per meta-learning arm.
    pub fn meta_arm_rewards(&self) -> &HashMap<usize, A> {
        self.meta_optimizer.arm_rewards()
    }

    /// Times each meta-learning arm has been played.
    pub fn meta_arm_counts(&self) -> &HashMap<usize, usize> {
        self.meta_optimizer.arm_counts()
    }

    /// Measured reliability of each adaptation signal.
    pub fn signal_reliability(&self) -> &HashMap<AdaptationSignalType, A> {
        &self.adaptation_strategy.signal_reliability
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.adaptation_history.clear();
        self.gradient_adapter.reset();
        self.performance_adapter.reset();
        self.drift_adapter.reset();
        self.resource_adapter.reset();
        self.meta_optimizer.reset();
    }
}

mod meta;
mod signals;

#[cfg(test)]
mod adaptive_lr_tests;

pub(crate) use signals::LossDriftDetector;

// Default implementations for various structures
impl<A: Float + Default + Send + Sync + Send + Sync> Default for GradientNormStatistics<A> {
    fn default() -> Self {
        Self {
            mean: A::default(),
            variance: A::default(),
            skewness: A::default(),
            kurtosis: A::default(),
            percentiles: vec![A::default(); 5],
            autocorrelation: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for SignalToNoiseEstimator<A> {
    fn default() -> Self {
        Self {
            signal_estimate: A::default(),
            noise_estimate: A::default(),
            snr_history: VecDeque::new(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for PerformanceTrendAnalyzer<A> {
    fn default() -> Self {
        Self {
            trend_detection_window: 10,
            trend_types: vec![],
            trend_strength: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for PlateauDetector<A> {
    fn default() -> Self {
        Self {
            plateau_threshold: A::default(),
            min_plateau_duration: 5,
            current_plateau_length: 0,
            plateau_confidence: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for OverfittingDetector<A> {
    fn default() -> Self {
        Self {
            train_loss_history: VecDeque::new(),
            val_loss_history: VecDeque::new(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for LearningEfficiencyTracker<A> {
    fn default() -> Self {
        Self {
            loss_reduction_per_step: VecDeque::new(),
            efficiency_score: A::default(),
            efficiency_trend: TrendType::Improving,
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for DistributionTracker<A> {
    fn default() -> Self {
        Self {
            feature_distributions: HashMap::new(),
            distribution_drift_score: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for AdaptationSpeedController<A> {
    fn default() -> Self {
        Self {
            base_adaptation_rate: A::from(0.1).unwrap_or_default(),
            current_adaptation_rate: A::from(0.1).unwrap_or_default(),
            acceleration_factor: A::from(1.1).unwrap_or_default(),
            deceleration_factor: A::from(0.9).unwrap_or_default(),
            momentum: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for DriftSeverityAssessor<A> {
    fn default() -> Self {
        Self {
            severity_levels: vec![],
            current_severity: DriftSeverityLevel::default(),
            severity_history: VecDeque::new(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for DriftSeverityLevel<A> {
    fn default() -> Self {
        Self {
            level: DriftSeverity::None,
            recommended_lr_adjustment: A::one(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for ExplorationStrategy<A> {
    fn default() -> Self {
        Self {
            exploration_rate: A::from(0.1).unwrap_or_default(),
            arm_rewards: HashMap::new(),
            arm_counts: HashMap::new(),
        }
    }
}

impl<A: Float + Default + Send + Sync + Send + Sync> Default for TransferLearner<A> {
    fn default() -> Self {
        Self {
            source_task_data: vec![],
            transfer_confidence: A::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_enhanced_adaptive_lr_controller_creation() {
        let config = AdaptiveLRConfig {
            base_lr: 0.01,
            min_lr: 1e-6,
            max_lr: 1.0,
            enable_gradient_adaptation: true,
            enable_performance_adaptation: true,
            enable_drift_adaptation: false,
            enable_resource_adaptation: false,
            enable_meta_learning: false,
            history_window_size: 100,
            adaptation_frequency: 10,
            adaptation_sensitivity: 0.1,
            use_ensemble_voting: true,
            step_time_budget: None,
            memory_budget_mb: None,
        };

        let controller = EnhancedAdaptiveLRController::<f32>::new(config);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_learning_rate_update() {
        let config = AdaptiveLRConfig {
            base_lr: 0.01,
            min_lr: 1e-6,
            max_lr: 1.0,
            enable_gradient_adaptation: true,
            enable_performance_adaptation: true,
            enable_drift_adaptation: false,
            enable_resource_adaptation: false,
            enable_meta_learning: false,
            history_window_size: 100,
            adaptation_frequency: 10,
            adaptation_sensitivity: 0.1,
            use_ensemble_voting: true,
            step_time_budget: None,
            memory_budget_mb: None,
        };

        let mut controller =
            EnhancedAdaptiveLRController::<f32>::new(config).expect("unwrap failed");
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        let loss = 0.5;
        let metrics = HashMap::new();

        let new_lr = controller.update_learning_rate(&gradients, loss, &metrics, 1);
        assert!(new_lr.is_ok());
        assert!(new_lr.expect("unwrap failed") > 0.0);
    }

    #[test]
    fn test_adaptation_statistics() {
        let config = AdaptiveLRConfig {
            base_lr: 0.01,
            min_lr: 1e-6,
            max_lr: 1.0,
            enable_gradient_adaptation: true,
            enable_performance_adaptation: true,
            enable_drift_adaptation: false,
            enable_resource_adaptation: false,
            enable_meta_learning: false,
            history_window_size: 100,
            adaptation_frequency: 10,
            adaptation_sensitivity: 0.1,
            use_ensemble_voting: true,
            step_time_budget: None,
            memory_budget_mb: None,
        };

        let controller = EnhancedAdaptiveLRController::<f32>::new(config).expect("unwrap failed");
        let stats = controller.get_adaptation_statistics();

        assert_eq!(stats.total_adaptations, 0);
        assert_eq!(stats.successful_adaptations, 0);
    }
}
