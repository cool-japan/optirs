// Component implementations for federated privacy algorithms
//
// # 0.3.2 changes
//
// * Public field names were corrected to snake_case: `clientid` -> `client_id`,
//   `epsilonconsumed` -> `epsilon_consumed`, `amplificationfactor` ->
//   `amplification_factor`, `compressionratio` -> `compression_ratio`,
//   `significancelevel` -> `significance_level`. These are breaking renames on a
//   public API; the project's naming policy mandates snake_case and the 0.3.x
//   series is the place to fix them.
// * `RoundComposition` gained `total_clients` and `noise_multiplier`, without
//   which `FederatedCompositionAnalyzer` cannot compose through the moments
//   accountant.
// * `FederatedMetaLearner::new` honours its argument (it was `_parametersize`).
// * The real method bodies for `PrivacyAmplificationAnalyzer`,
//   `FederatedCompositionAnalyzer`, `FederatedMetaLearner` and `TaskDetector`
//   live in the `composition` and `adaptation` modules; this file keeps the type
//   definitions and constructors.

use super::super::PrivacyBudget;
use super::config::*;
use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;

// Advanced federated learning implementation structures

/// Byzantine-robust aggregation engine
pub struct ByzantineRobustAggregator<T: Float + Debug + Send + Sync + 'static> {
    config: ByzantineRobustConfig,
    client_reputations: HashMap<String, f64>,
    outlier_history: VecDeque<OutlierDetectionResult>,
    statistical_analyzer: StatisticalAnalyzer<T>,
    robust_estimators: RobustEstimators<T>,
}

/// Personalized federated learning manager
pub struct PersonalizationManager<T: Float + Debug + Send + Sync + 'static> {
    config: PersonalizationConfig,
    client_models: HashMap<String, PersonalizedModel<T>>,
    global_model: Option<Array1<T>>,
    clustering_engine: ClusteringEngine<T>,
    meta_learner: FederatedMetaLearner<T>,
    adaptation_tracker: AdaptationTracker<T>,
}

/// Adaptive privacy budget manager
pub struct AdaptiveBudgetManager<T: Float + Debug + Send + Sync + 'static> {
    config: AdaptiveBudgetConfig,
    client_budgets: HashMap<String, AdaptiveBudget>,
    global_budget_tracker: GlobalBudgetTracker,
    utility_estimator: UtilityEstimator,
    fairness_monitor: FairnessMonitor,
    contextual_analyzer: ContextualAnalyzer,
    _phantom: std::marker::PhantomData<T>,
}

/// Communication efficiency optimizer
pub struct CommunicationOptimizer<T: Float + Debug + Send + Sync + 'static> {
    config: CommunicationConfig,
    compression_engine: CompressionEngine<T>,
    bandwidth_monitor: BandwidthMonitor,
    transmission_scheduler: TransmissionScheduler,
    gradient_buffers: HashMap<String, GradientBuffer<T>>,
    quality_controller: QualityController,
}

/// Continual learning coordinator
pub struct ContinualLearningCoordinator<T: Float + Debug + Send + Sync + 'static> {
    config: ContinualLearningConfig,
    task_detector: TaskDetector<T>,
    memory_manager: MemoryManager<T>,
    knowledge_transfer_engine: KnowledgeTransferEngine<T>,
    forgetting_prevention: ForgettingPreventionEngine<T>,
    task_history: VecDeque<TaskInfo>,
}

// Supporting implementation structures

/// Statistical analyzer for Byzantine detection
pub struct StatisticalAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    window_size: usize,
    significance_level: f64,
    test_statistics: VecDeque<TestStatistic<T>>,
}

/// Robust estimators for aggregation
pub struct RobustEstimators<T: Float + Debug + Send + Sync + 'static> {
    trimmed_mean_cache: HashMap<String, T>,
    median_cache: HashMap<String, T>,
    krum_scores: HashMap<String, f64>,
}

/// Outlier detection result
#[derive(Debug, Clone)]
pub struct OutlierDetectionResult {
    pub client_id: String,
    pub round: usize,
    pub is_outlier: bool,
    pub outlier_score: f64,
    pub detection_method: String,
}

/// Personalized model for each client
#[derive(Debug, Clone)]
pub struct PersonalizedModel<T: Float + Debug + Send + Sync + 'static> {
    pub model_parameters: Array1<T>,
    pub personal_layers: HashMap<usize, Array1<T>>,
    pub adaptation_state: AdaptationState<T>,
    pub performance_history: Vec<f64>,
    pub last_update_round: usize,
}

/// Adaptation state for personalized models
#[derive(Debug, Clone)]
pub struct AdaptationState<T: Float + Debug + Send + Sync + 'static> {
    pub learning_rate: f64,
    pub momentum: Array1<T>,
    pub adaptation_count: usize,
    pub gradient_history: VecDeque<Array1<T>>,
}

/// Clustering engine for federated learning
pub struct ClusteringEngine<T: Float + Debug + Send + Sync + 'static> {
    method: ClusteringMethod,
    cluster_centers: HashMap<usize, Array1<T>>,
    client_clusters: HashMap<String, usize>,
    cluster_update_counter: usize,
}

/// Federated meta-learner
pub struct FederatedMetaLearner<T: Float + Debug + Send + Sync + 'static> {
    pub(super) meta_parameters: Array1<T>,
    pub(super) client_adaptations: HashMap<String, Array1<T>>,
    pub(super) meta_gradient_buffer: Array1<T>,
    pub(super) task_distributions: HashMap<String, TaskDistribution<T>>,
}

/// Task distribution for meta-learning
#[derive(Debug, Clone)]
pub struct TaskDistribution<T: Float + Debug + Send + Sync + 'static> {
    pub support_gradient: Array1<T>,
    pub query_gradient: Array1<T>,
    pub task_similarity: f64,
    pub adaptation_steps: usize,
}

/// Adaptation tracker
pub struct AdaptationTracker<T: Float + Debug + Send + Sync + 'static> {
    adaptation_history: HashMap<String, Vec<AdaptationEvent<T>>>,
    convergence_metrics: HashMap<String, ConvergenceMetrics>,
}

/// Adaptation event
#[derive(Debug, Clone)]
pub struct AdaptationEvent<T: Float + Debug + Send + Sync + 'static> {
    pub round: usize,
    pub parameter_change: Array1<T>,
    pub loss_improvement: f64,
    pub adaptation_method: String,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub convergence_rate: f64,
    pub stability_measure: f64,
    pub adaptation_efficiency: f64,
}

/// Adaptive budget for each client
#[derive(Debug, Clone)]
pub struct AdaptiveBudget {
    pub current_epsilon: f64,
    pub current_delta: f64,
    pub allocated_epsilon: f64,
    pub allocated_delta: f64,
    pub consumption_rate: f64,
    pub importance_weight: f64,
    pub context_factors: HashMap<String, f64>,
}

/// Global budget tracker
pub struct GlobalBudgetTracker {
    total_allocated: f64,
    consumption_history: VecDeque<BudgetConsumption>,
    allocation_strategy: BudgetAllocationStrategy,
}

/// Budget consumption record
#[derive(Debug, Clone)]
pub struct BudgetConsumption {
    pub round: usize,
    pub client_id: String,
    pub epsilon_consumed: f64,
    pub delta_consumed: f64,
    pub utility_achieved: f64,
}

/// Utility estimator
pub struct UtilityEstimator {
    utility_history: VecDeque<UtilityMeasurement>,
    prediction_model: UtilityPredictionModel,
}

/// Utility measurement
#[derive(Debug, Clone)]
pub struct UtilityMeasurement {
    pub round: usize,
    pub accuracy: f64,
    pub loss: f64,
    pub convergence_rate: f64,
    pub noise_level: f64,
}

/// Utility prediction model
pub struct UtilityPredictionModel {
    model_type: String,
    parameters: HashMap<String, f64>,
}

/// Fairness monitor
pub struct FairnessMonitor {
    fairness_metrics: FairnessMetrics,
    client_fairness_scores: HashMap<String, f64>,
    fairness_constraints: Vec<FairnessConstraint>,
}

/// Fairness metrics
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    pub demographic_parity: f64,
    pub equalized_opportunity: f64,
    pub individual_fairness: f64,
    pub group_fairness: f64,
}

/// Fairness constraint
#[derive(Debug, Clone)]
pub struct FairnessConstraint {
    pub constraint_type: String,
    pub threshold: f64,
    pub affected_groups: Vec<String>,
}

/// Selection diversity metrics for client sampling
#[derive(Debug, Clone)]
pub struct SelectionDiversityMetrics {
    pub geographic_diversity: f64,
    pub demographic_diversity: f64,
    pub resource_diversity: f64,
    pub temporal_diversity: f64,
}

/// Contextual analyzer
pub struct ContextualAnalyzer {
    context_history: VecDeque<ContextSnapshot>,
    context_model: ContextModel,
}

/// Context snapshot
#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    pub timestamp: u64,
    pub context_factors: HashMap<String, f64>,
    pub privacy_requirement: f64,
    pub utility_requirement: f64,
}

/// Context model for privacy adaptation
pub struct ContextModel {
    model_parameters: HashMap<String, f64>,
    adaptation_learning_rate: f64,
}

/// Compression engine
pub struct CompressionEngine<T: Float + Debug + Send + Sync + 'static> {
    strategy: CompressionStrategy,
    compression_history: VecDeque<CompressionResult<T>>,
    error_feedback_memory: HashMap<String, Array1<T>>,
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult<T: Float + Debug + Send + Sync + 'static> {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub reconstruction_error: T,
    pub compression_time: u64,
}

/// Bandwidth monitor
pub struct BandwidthMonitor {
    bandwidth_history: VecDeque<BandwidthMeasurement>,
    current_conditions: NetworkConditions,
}

/// Bandwidth measurement
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    pub timestamp: u64,
    pub upload_bandwidth: f64,
    pub download_bandwidth: f64,
    pub latency: f64,
    pub packet_loss: f64,
}

/// Network conditions
#[derive(Debug, Clone)]
pub struct NetworkConditions {
    pub available_bandwidth: f64,
    pub network_quality: NetworkQuality,
    pub congestion_level: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Transmission scheduler
pub struct TransmissionScheduler {
    schedule_queue: VecDeque<TransmissionTask>,
    priority_weights: HashMap<String, f64>,
}

/// Transmission task
#[derive(Debug, Clone)]
pub struct TransmissionTask {
    pub client_id: String,
    pub data_size: usize,
    pub priority: f64,
    pub deadline: u64,
    pub compression_required: bool,
}

/// Gradient buffer for communication optimization
pub struct GradientBuffer<T: Float + Debug + Send + Sync + 'static> {
    buffered_gradients: VecDeque<Array1<T>>,
    staleness_tolerance: usize,
    buffer_capacity: usize,
}

/// Quality controller for communication
pub struct QualityController {
    qos_requirements: QoSConfig,
    performance_monitor: PerformanceMonitor,
}

/// Performance monitor
pub struct PerformanceMonitor {
    latency_measurements: VecDeque<f64>,
    throughput_measurements: VecDeque<f64>,
    quality_violations: usize,
}

/// Task detector for continual learning
pub struct TaskDetector<T: Float + Debug + Send + Sync + 'static> {
    pub(super) detection_method: TaskDetectionMethod,
    pub(super) gradient_buffer: VecDeque<Array1<T>>,
    pub(super) change_points: Vec<ChangePoint>,
    pub(super) detection_threshold: f64,
}

/// Change point for task detection
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub round: usize,
    pub confidence: f64,
    pub change_magnitude: f64,
}

/// Memory manager for continual learning
pub struct MemoryManager<T: Float + Debug + Send + Sync + 'static> {
    memory_budget: usize,
    stored_examples: VecDeque<MemoryExample<T>>,
    eviction_strategy: EvictionStrategy,
    compression_enabled: bool,
}

/// Memory example for continual learning
#[derive(Debug, Clone)]
pub struct MemoryExample<T: Float + Debug + Send + Sync + 'static> {
    pub features: Array1<T>,
    pub target: Array1<T>,
    pub importance: f64,
    pub timestamp: u64,
    pub task_id: usize,
}

/// Knowledge transfer engine
pub struct KnowledgeTransferEngine<T: Float + Debug + Send + Sync + 'static> {
    transfer_method: KnowledgeTransferMethod,
    transfer_matrices: HashMap<String, Array1<T>>,
    similarity_cache: HashMap<String, f64>,
}

/// Forgetting prevention engine
pub struct ForgettingPreventionEngine<T: Float + Debug + Send + Sync + 'static> {
    method: ForgettingPreventionMethod,
    importance_weights: HashMap<String, Array1<T>>,
    regularization_strength: f64,
    memory_replay_buffer: VecDeque<Array1<T>>,
}

/// Task information
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: usize,
    pub start_round: usize,
    pub end_round: Option<usize>,
    pub task_description: String,
    pub performance_metrics: HashMap<String, f64>,
}

/// Test statistic for outlier detection
#[derive(Debug, Clone)]
pub struct TestStatistic<T: Float + Debug + Send + Sync + 'static> {
    pub round: usize,
    pub statistic_value: T,
    pub p_value: f64,
    pub test_type: StatisticalTestType,
    pub client_id: String,
}

/// Secure aggregation protocol implementation
pub struct SecureAggregator<T: Float + Debug + Send + Sync + 'static> {
    config: SecureAggregationConfig,
    client_masks: HashMap<String, Array1<T>>,
    shared_randomness: Arc<std::sync::Mutex<u64>>,
    aggregation_threshold: usize,
    round_keys: Vec<u64>,
}

/// Privacy amplification analyzer
pub struct PrivacyAmplificationAnalyzer {
    pub(super) config: AmplificationConfig,
    pub(super) subsampling_history: VecDeque<SubsamplingEvent>,
    pub(super) amplification_factors: HashMap<String, f64>,
}

/// Cross-device privacy manager
pub struct CrossDevicePrivacyManager<T: Float + Debug + Send + Sync + 'static> {
    config: CrossDeviceConfig,
    user_clusters: HashMap<String, Vec<String>>,
    device_profiles: HashMap<String, DeviceProfile<T>>,
    temporal_correlations: HashMap<String, Vec<TemporalEvent>>,
}

/// Federated composition analyzer
pub struct FederatedCompositionAnalyzer {
    pub(super) method: FederatedCompositionMethod,
    pub(super) round_compositions: Vec<RoundComposition>,
    pub(super) client_compositions: HashMap<String, Vec<ClientComposition>>,
}

/// Client participation in a round
#[derive(Debug, Clone)]
pub struct ParticipationRound {
    pub round: usize,
    pub participating_clients: Vec<String>,
    pub sampling_probability: f64,
    pub privacy_cost: PrivacyCost,
    pub aggregation_noise: f64,
}

/// Privacy cost breakdown
#[derive(Debug, Clone)]
pub struct PrivacyCost {
    pub epsilon: f64,
    pub delta: f64,
    pub client_contribution: f64,
    pub amplification_factor: f64,
    pub composition_cost: f64,
}

/// Subsampling event for amplification analysis
#[derive(Debug, Clone)]
pub struct SubsamplingEvent {
    pub round: usize,
    pub sampling_rate: f64,
    pub clients_sampled: usize,
    pub total_clients: usize,
    pub amplification_factor: f64,
}

/// Device profile for cross-device privacy
#[derive(Debug, Clone)]
pub struct DeviceProfile<T: Float + Debug + Send + Sync + 'static> {
    pub device_id: String,
    pub user_id: String,
    pub device_type: DeviceType,
    pub location_cluster: String,
    pub participation_frequency: f64,
    pub local_privacy_budget: PrivacyBudget,
    pub sensitivity_estimate: T,
}

/// Device types for privacy analysis
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum DeviceType {
    Mobile,
    Desktop,
    IoT,
    Edge,
    Server,
}

/// Temporal event for privacy tracking
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    pub timestamp: u64,
    pub event_type: TemporalEventType,
    pub privacy_impact: f64,
}

#[derive(Debug, Clone)]
pub enum TemporalEventType {
    ClientParticipation,
    ModelUpdate,
    PrivacyBudgetConsumption,
    AggregationEvent,
}

/// Round composition for privacy accounting
#[derive(Debug, Clone)]
pub struct RoundComposition {
    /// Round number.
    pub round: usize,
    /// Clients that participated in this round.
    pub participating_clients: usize,
    /// Size of the federation the round sampled from.
    ///
    /// Added in 0.3.2: the moments-accountant composition needs the sampling
    /// rate, and the previous shape could only report the numerator.
    pub total_clients: usize,
    /// Epsilon consumed by the round.
    pub epsilon_consumed: f64,
    /// Delta the round's epsilon is reported at.
    pub delta_consumed: f64,
    /// Whether the subsampling amplification bound was applied.
    pub amplification_applied: bool,
    /// Composition method the round was accounted under.
    pub composition_method: FederatedCompositionMethod,
    /// Noise multiplier used by the round, when the round was a Gaussian
    /// mechanism. Required by
    /// `FederatedCompositionMethod::FederatedMomentsAccountant`.
    pub noise_multiplier: Option<f64>,
}

/// Client-specific composition tracking
#[derive(Debug, Clone)]
pub struct ClientComposition {
    pub client_id: String,
    pub round: usize,
    pub local_epsilon: f64,
    pub local_delta: f64,
    pub contribution_weight: f64,
}

// Implementation blocks for components

impl ContextualAnalyzer {
    /// Create a new contextual analyzer
    pub fn new() -> Self {
        Self {
            context_history: VecDeque::with_capacity(100),
            context_model: ContextModel::new(),
        }
    }
}

impl Default for ContextualAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextModel {
    /// Create a new context model
    pub fn new() -> Self {
        Self {
            model_parameters: HashMap::new(),
            adaptation_learning_rate: 0.01,
        }
    }
}

impl Default for ContextModel {
    fn default() -> Self {
        Self::new()
    }
}

impl FairnessMonitor {
    /// Create a new fairness monitor
    pub fn new() -> Self {
        Self {
            fairness_metrics: FairnessMetrics {
                demographic_parity: 0.0,
                equalized_opportunity: 0.0,
                individual_fairness: 0.0,
                group_fairness: 0.0,
            },
            client_fairness_scores: HashMap::new(),
            fairness_constraints: Vec::new(),
        }
    }

    /// Get current fairness metrics
    pub fn get_metrics(&self) -> &FairnessMetrics {
        &self.fairness_metrics
    }

    /// Compute fairness weights for clients
    pub fn compute_fairness_weights(&self, client_ids: &[String]) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        for client_id in client_ids {
            // Use existing fairness score or default to 1.0
            let weight = self
                .client_fairness_scores
                .get(client_id)
                .copied()
                .unwrap_or(1.0);
            weights.insert(client_id.clone(), weight);
        }
        weights
    }
}

impl Default for FairnessMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand,
    > FederatedMetaLearner<T>
{
    /// Create a new federated meta-learner sized for `parameter_size`
    /// parameters.
    ///
    /// The argument used to be `_parametersize` and was discarded, so both
    /// buffers were allocated as `Array1::default(0)`: a caller constructing for
    /// a one-million-parameter model got empty buffers and every later
    /// elementwise operation was a length mismatch. `0` remains legal and means
    /// "not yet sized"; `compute_client_meta_gradients` (in
    /// [`super::adaptation`]) reports that rather than returning a length-0
    /// array.
    pub fn new(parameter_size: usize) -> Self {
        Self {
            meta_parameters: Array1::zeros(parameter_size),
            client_adaptations: HashMap::new(),
            meta_gradient_buffer: Array1::zeros(parameter_size),
            task_distributions: HashMap::new(),
        }
    }

    /// Number of parameters this learner is sized for.
    pub fn parameter_size(&self) -> usize {
        self.meta_parameters.len()
    }

    /// The current meta-parameters.
    pub fn meta_parameters(&self) -> &Array1<T> {
        &self.meta_parameters
    }

    /// The most recently computed meta-gradient.
    pub fn meta_gradient_buffer(&self) -> &Array1<T> {
        &self.meta_gradient_buffer
    }

    /// The per-client adaptation recorded for `client_id`, if any.
    pub fn client_adaptation(&self, client_id: &str) -> Option<&Array1<T>> {
        self.client_adaptations.get(client_id)
    }

    /// The task distribution recorded for `client_id`, if any.
    pub fn task_distribution(&self, client_id: &str) -> Option<&TaskDistribution<T>> {
        self.task_distributions.get(client_id)
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> ClusteringEngine<T> {
    /// Create a new clustering engine
    pub fn new() -> Self {
        Self {
            method: ClusteringMethod::KMeans,
            cluster_centers: HashMap::new(),
            client_clusters: HashMap::new(),
            cluster_update_counter: 0,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> Default for ClusteringEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdaptationTracker<T> {
    /// Create a new adaptation tracker
    pub fn new() -> Self {
        Self {
            adaptation_history: HashMap::new(),
            convergence_metrics: HashMap::new(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for AdaptationTracker<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalBudgetTracker {
    /// Create a new global budget tracker
    pub fn new() -> Self {
        Self {
            total_allocated: 0.0,
            consumption_history: VecDeque::with_capacity(1000),
            allocation_strategy: BudgetAllocationStrategy::Uniform,
        }
    }
}

impl Default for GlobalBudgetTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> TaskDetector<T> {
    /// Create a new task detector
    pub fn new() -> Self {
        Self {
            detection_method: TaskDetectionMethod::GradientBased,
            gradient_buffer: VecDeque::with_capacity(100),
            change_points: Vec::new(),
            detection_threshold: 0.1,
        }
    }

    /// Detection threshold on the normalised gradient shift.
    pub fn detection_threshold(&self) -> f64 {
        self.detection_threshold
    }

    /// Replace the detection threshold.
    pub fn set_detection_threshold(&mut self, threshold: f64) -> Result<()> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(crate::error::OptimError::InvalidParameter(format!(
                "the task-detection threshold must be positive and finite, got {threshold}"
            )));
        }
        self.detection_threshold = threshold;
        Ok(())
    }

    /// The configured detection method.
    pub fn detection_method(&self) -> TaskDetectionMethod {
        self.detection_method
    }

    /// Change points detected so far.
    pub fn change_points(&self) -> &[ChangePoint] {
        &self.change_points
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for TaskDetector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl TransmissionScheduler {
    /// Create a new transmission scheduler
    pub fn new() -> Self {
        Self {
            schedule_queue: VecDeque::new(),
            priority_weights: HashMap::new(),
        }
    }
}

impl Default for TransmissionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityController {
    /// Create a new quality controller
    pub fn new() -> Self {
        Self {
            qos_requirements: QoSConfig::default(),
            performance_monitor: PerformanceMonitor::new(),
        }
    }
}

impl Default for QualityController {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            latency_measurements: VecDeque::with_capacity(1000),
            throughput_measurements: VecDeque::with_capacity(1000),
            quality_violations: 0,
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> MemoryManager<T> {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            memory_budget: 1000,
            stored_examples: VecDeque::new(),
            eviction_strategy: EvictionStrategy::LRU,
            compression_enabled: false,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for MemoryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> KnowledgeTransferEngine<T> {
    /// Create a new knowledge transfer engine
    pub fn new() -> Self {
        Self {
            transfer_method: KnowledgeTransferMethod::ParameterTransfer,
            transfer_matrices: HashMap::new(),
            similarity_cache: HashMap::new(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for KnowledgeTransferEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ForgettingPreventionEngine<T> {
    /// Create a new forgetting prevention engine
    pub fn new() -> Self {
        Self {
            method: ForgettingPreventionMethod::EWC,
            importance_weights: HashMap::new(),
            regularization_strength: 0.1,
            memory_replay_buffer: VecDeque::new(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ForgettingPreventionEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for component creation

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone> ByzantineRobustAggregator<T> {
    /// Create an aggregator with the historical hardcoded defaults.
    ///
    /// Retained for compatibility. Prefer
    /// [`ByzantineRobustAggregator::with_config`]: this constructor takes no
    /// argument, so whatever the user configured could never reach the
    /// aggregator.
    pub fn new() -> Result<Self> {
        Self::with_config(ByzantineRobustConfig {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio: 0.2 },
            expected_byzantine_ratio: 0.2,
            dynamic_detection: true,
            reputation_system: ReputationSystemConfig::default(),
            statistical_tests: StatisticalTestConfig::default(),
        })
    }

    /// Create an aggregator from a configuration.
    pub fn with_config(config: ByzantineRobustConfig) -> Result<Self> {
        if !(0.0..0.5).contains(&config.expected_byzantine_ratio) {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "expected_byzantine_ratio must lie in [0, 0.5), got {}; no robust aggregation rule                  tolerates half or more of the clients being adversarial",
                config.expected_byzantine_ratio
            )));
        }
        if let ByzantineRobustMethod::TrimmedMean { trim_ratio } = config.method {
            if !(0.0..0.5).contains(&trim_ratio) {
                return Err(crate::error::OptimError::InvalidConfig(format!(
                    "TrimmedMean trim_ratio must lie in [0, 0.5), got {trim_ratio}"
                )));
            }
        }
        let significance_level = config.statistical_tests.significance_level;
        if !(0.0..1.0).contains(&significance_level) || significance_level <= 0.0 {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "statistical_tests.significance_level must lie in (0, 1), got {significance_level}"
            )));
        }
        Ok(Self {
            config,
            client_reputations: HashMap::new(),
            outlier_history: VecDeque::new(),
            statistical_analyzer: StatisticalAnalyzer {
                window_size: 10,
                significance_level,
                test_statistics: VecDeque::new(),
            },
            robust_estimators: RobustEstimators {
                trimmed_mean_cache: HashMap::new(),
                median_cache: HashMap::new(),
                krum_scores: HashMap::new(),
            },
        })
    }

    /// The configuration this aggregator is running under.
    pub fn config(&self) -> &ByzantineRobustConfig {
        &self.config
    }

    /// The significance level the outlier tests use.
    pub fn significance_level(&self) -> f64 {
        self.statistical_analyzer.significance_level
    }

    /// Recorded outlier detections, oldest first.
    pub fn outlier_history(&self) -> impl Iterator<Item = &OutlierDetectionResult> {
        self.outlier_history.iter()
    }
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + Default
            + Clone
            + scirs2_core::ndarray::ScalarOperand,
    > PersonalizationManager<T>
{
    /// Create a manager with personalization disabled.
    ///
    /// Retained for compatibility; prefer
    /// [`PersonalizationManager::with_config`], which is the only way a
    /// configured strategy can reach runtime.
    pub fn new() -> Result<Self> {
        Self::with_config(
            PersonalizationConfig {
                strategy: PersonalizationStrategy::None,
                local_adaptation: LocalAdaptationConfig::default(),
                clustering: ClusteringConfig::default(),
                meta_learning: MetaLearningConfig::default(),
                privacy_preserving: false,
            },
            0,
        )
    }

    /// Create a manager from a configuration, sized for `parameter_size`
    /// parameters.
    pub fn with_config(config: PersonalizationConfig, parameter_size: usize) -> Result<Self> {
        Ok(Self {
            config,
            client_models: HashMap::new(),
            global_model: None,
            clustering_engine: ClusteringEngine::new(),
            meta_learner: FederatedMetaLearner::new(parameter_size),
            adaptation_tracker: AdaptationTracker::new(),
        })
    }

    /// The configuration this manager is running under.
    pub fn config(&self) -> &PersonalizationConfig {
        &self.config
    }

    /// The meta-learner backing the personalization strategy.
    pub fn meta_learner(&self) -> &FederatedMetaLearner<T> {
        &self.meta_learner
    }

    /// Mutable access to the meta-learner.
    pub fn meta_learner_mut(&mut self) -> &mut FederatedMetaLearner<T> {
        &mut self.meta_learner
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdaptiveBudgetManager<T> {
    /// Create a manager with the default (disabled) adaptive budget config.
    ///
    /// Retained for compatibility; prefer
    /// [`AdaptiveBudgetManager::with_config`].
    pub fn new() -> Result<Self> {
        Self::with_config(AdaptiveBudgetConfig::default())
    }

    /// Create a manager from a configuration.
    pub fn with_config(config: AdaptiveBudgetConfig) -> Result<Self> {
        Ok(Self {
            config,
            client_budgets: HashMap::new(),
            global_budget_tracker: GlobalBudgetTracker::new(),
            utility_estimator: UtilityEstimator {
                utility_history: VecDeque::new(),
                prediction_model: UtilityPredictionModel {
                    model_type: "linear".to_string(),
                    parameters: HashMap::new(),
                },
            },
            fairness_monitor: FairnessMonitor::new(),
            contextual_analyzer: ContextualAnalyzer::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// The configuration this manager is running under.
    pub fn config(&self) -> &AdaptiveBudgetConfig {
        &self.config
    }

    /// The fairness monitor.
    pub fn fairness_monitor(&self) -> &FairnessMonitor {
        &self.fairness_monitor
    }

    /// The adaptive budget recorded for a client, if any.
    pub fn client_budget(&self, client_id: &str) -> Option<&AdaptiveBudget> {
        self.client_budgets.get(client_id)
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default> CommunicationOptimizer<T> {
    /// Create an optimizer with compression disabled.
    ///
    /// Retained for compatibility; prefer
    /// [`CommunicationOptimizer::with_config`], which is the only way a
    /// configured compression strategy can reach the compression engine (the
    /// strategy used to be hardcoded to `None` in two places).
    pub fn new() -> Result<Self> {
        Self::with_config(CommunicationConfig {
            compression: CompressionStrategy::None,
            lazy_aggregation: LazyAggregationConfig::default(),
            federated_dropout: FederatedDropoutConfig::default(),
            async_updates: AsyncUpdateConfig::default(),
            bandwidth_adaptation: BandwidthAdaptationConfig::default(),
        })
    }

    /// Create an optimizer from a configuration.
    pub fn with_config(config: CommunicationConfig) -> Result<Self> {
        let strategy = config.compression;
        Ok(Self {
            config,
            compression_engine: CompressionEngine {
                strategy,
                compression_history: VecDeque::new(),
                error_feedback_memory: HashMap::new(),
            },
            bandwidth_monitor: BandwidthMonitor {
                bandwidth_history: VecDeque::new(),
                current_conditions: NetworkConditions {
                    available_bandwidth: 100.0,
                    network_quality: NetworkQuality::Good,
                    congestion_level: 0.5,
                },
            },
            transmission_scheduler: TransmissionScheduler::new(),
            gradient_buffers: HashMap::new(),
            quality_controller: QualityController::new(),
        })
    }

    /// The configuration this optimizer is running under.
    pub fn config(&self) -> &CommunicationConfig {
        &self.config
    }

    /// The compression strategy actually installed in the engine.
    pub fn compression_strategy(&self) -> CompressionStrategy {
        self.compression_engine.strategy
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default> ContinualLearningCoordinator<T> {
    /// Create a coordinator with the task-agnostic strategy.
    ///
    /// Retained for compatibility; prefer
    /// [`ContinualLearningCoordinator::with_config`].
    pub fn new() -> Result<Self> {
        Self::with_config(ContinualLearningConfig {
            strategy: ContinualLearningStrategy::TaskAgnostic,
            memory_management: MemoryManagementConfig::default(),
            task_detection: TaskDetectionConfig::default(),
            knowledge_transfer: KnowledgeTransferConfig::default(),
            forgetting_prevention: ForgettingPreventionConfig::default(),
        })
    }

    /// Create a coordinator from a configuration.
    ///
    /// The task detector is configured from `config.task_detection`, so the
    /// configured detection method and threshold reach runtime instead of the
    /// hardcoded `GradientBased` / `0.1` pair.
    pub fn with_config(config: ContinualLearningConfig) -> Result<Self> {
        let mut task_detector = TaskDetector::new();
        task_detector.detection_method = config.task_detection.detection_method;
        task_detector.set_detection_threshold(config.task_detection.sensitivity_threshold)?;
        Ok(Self {
            config,
            task_detector,
            memory_manager: MemoryManager::new(),
            knowledge_transfer_engine: KnowledgeTransferEngine::new(),
            forgetting_prevention: ForgettingPreventionEngine::new(),
            task_history: VecDeque::new(),
        })
    }

    /// The configuration this coordinator is running under.
    pub fn config(&self) -> &ContinualLearningConfig {
        &self.config
    }

    /// The task detector.
    pub fn task_detector(&self) -> &TaskDetector<T> {
        &self.task_detector
    }

    /// Mutable access to the task detector.
    pub fn task_detector_mut(&mut self) -> &mut TaskDetector<T> {
        &mut self.task_detector
    }

    /// Recorded task history, oldest first.
    pub fn task_history(&self) -> impl Iterator<Item = &TaskInfo> {
        self.task_history.iter()
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default> SecureAggregator<T> {
    /// Create a new secure aggregator.
    ///
    /// The aggregation threshold is taken from `config.min_clients`. It was
    /// previously hardcoded to `10` while the configuration was stored and
    /// ignored, so a federation configured with a different threshold silently
    /// got 10.
    pub fn new(config: SecureAggregationConfig) -> Result<Self> {
        if config.min_clients < 2 {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "secure_aggregation.min_clients must be at least 2, got {}",
                config.min_clients
            )));
        }
        if config.max_dropouts >= config.min_clients {
            return Err(crate::error::OptimError::InvalidConfig(format!(
                "secure_aggregation.max_dropouts ({}) must be below min_clients ({})",
                config.max_dropouts, config.min_clients
            )));
        }
        let aggregation_threshold = config.min_clients;
        Ok(Self {
            config,
            client_masks: HashMap::new(),
            shared_randomness: Arc::new(std::sync::Mutex::new(0u64)),
            aggregation_threshold,
            round_keys: Vec::new(),
        })
    }

    /// The configuration this aggregator is running under.
    pub fn config(&self) -> &SecureAggregationConfig {
        &self.config
    }

    /// Minimum number of clients required before an aggregate is released.
    pub fn aggregation_threshold(&self) -> usize {
        self.aggregation_threshold
    }
}

impl PrivacyAmplificationAnalyzer {
    /// Create a new privacy amplification analyzer.
    ///
    /// The analysis itself lives in [`super::composition`]:
    /// `compute_amplification_factor` now evaluates the published subsampling
    /// bound instead of the discarded `(1/q).sqrt()` placeholder, and records the
    /// real client counts instead of a fabricated `total_clients: 1000`.
    pub fn new(config: AmplificationConfig) -> Self {
        Self {
            config,
            subsampling_history: VecDeque::new(),
            amplification_factors: HashMap::new(),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> CrossDevicePrivacyManager<T> {
    /// Create a new cross-device privacy manager
    pub fn new(config: CrossDeviceConfig) -> Self {
        Self {
            config,
            user_clusters: HashMap::new(),
            device_profiles: HashMap::new(),
            temporal_correlations: HashMap::new(),
        }
    }

    /// The configuration this manager is running under.
    ///
    /// Every field of it is currently unimplemented, which is why
    /// `FederatedPrivacyConfig::validate` refuses a configuration that sets any
    /// of them; see `privacy::federated::cross_device_manager` for the working
    /// implementation of the same idea.
    pub fn config(&self) -> &CrossDeviceConfig {
        &self.config
    }

    /// Devices recorded for a user, if any.
    pub fn user_devices(&self, user_id: &str) -> &[String] {
        self.user_clusters
            .get(user_id)
            .map(|devices| devices.as_slice())
            .unwrap_or(&[])
    }
}

impl FederatedCompositionAnalyzer {
    /// Create a new federated composition analyzer
    pub fn new(method: FederatedCompositionMethod) -> Self {
        Self {
            method,
            round_compositions: Vec::new(),
            client_compositions: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::federated_privacy::config::{
        AdaptiveBudgetConfig, BandwidthAdaptationConfig, ByzantineRobustConfig,
        ByzantineRobustMethod, ClusteringConfig, CommunicationConfig, CompressionStrategy,
        ContinualLearningConfig, ContinualLearningStrategy, CrossDeviceConfig,
        FederatedDropoutConfig, ForgettingPreventionConfig, KnowledgeTransferConfig,
        LazyAggregationConfig, LocalAdaptationConfig, MemoryManagementConfig, MetaLearningConfig,
        PersonalizationConfig, PersonalizationStrategy, ReputationSystemConfig,
        SecureAggregationConfig, StatisticalTestConfig, TaskDetectionConfig, TaskDetectionMethod,
    };

    fn byzantine_config(trim_ratio: f64, byzantine_ratio: f64) -> ByzantineRobustConfig {
        ByzantineRobustConfig {
            method: ByzantineRobustMethod::TrimmedMean { trim_ratio },
            expected_byzantine_ratio: byzantine_ratio,
            dynamic_detection: true,
            reputation_system: ReputationSystemConfig::default(),
            statistical_tests: StatisticalTestConfig::default(),
        }
    }

    #[test]
    fn the_byzantine_aggregator_takes_its_configuration() {
        // Regression for F104: `new()` took no argument and hardcoded
        // TrimmedMean{0.2} with expected_byzantine_ratio 0.2, so whatever the
        // user configured could never reach the aggregator.
        let aggregator =
            match ByzantineRobustAggregator::<f64>::with_config(byzantine_config(0.35, 0.4)) {
                Ok(aggregator) => aggregator,
                Err(err) => panic!("construction failed: {err}"),
            };
        assert_eq!(aggregator.config().expected_byzantine_ratio, 0.4);
        match aggregator.config().method {
            ByzantineRobustMethod::TrimmedMean { trim_ratio } => {
                assert!((trim_ratio - 0.35).abs() < 1e-12)
            }
            other => panic!("unexpected method {other:?}"),
        }
        assert!((aggregator.significance_level() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn an_unusable_byzantine_configuration_is_refused() {
        for (trim, byzantine) in [
            (0.5f64, 0.2f64),
            (0.9, 0.2),
            (-0.1, 0.2),
            (0.2, 0.5),
            (0.2, 1.0),
        ] {
            assert!(
                ByzantineRobustAggregator::<f64>::with_config(byzantine_config(trim, byzantine))
                    .is_err(),
                "trim={trim}, byzantine={byzantine} must be refused"
            );
        }
        let mut config = byzantine_config(0.2, 0.2);
        config.statistical_tests.significance_level = 1.0;
        assert!(ByzantineRobustAggregator::<f64>::with_config(config).is_err());
    }

    #[test]
    fn the_secure_aggregator_honours_the_configured_threshold() {
        // The threshold was hardcoded to 10 while the config was stored and
        // ignored.
        let config = SecureAggregationConfig {
            min_clients: 25,
            max_dropouts: 4,
            ..SecureAggregationConfig::default()
        };
        let aggregator = match SecureAggregator::<f64>::new(config) {
            Ok(aggregator) => aggregator,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert_eq!(aggregator.aggregation_threshold(), 25);
        assert_eq!(aggregator.config().min_clients, 25);

        for (min_clients, max_dropouts) in [(1usize, 0usize), (0, 0), (5, 5), (5, 9)] {
            let config = SecureAggregationConfig {
                min_clients,
                max_dropouts,
                ..SecureAggregationConfig::default()
            };
            assert!(
                SecureAggregator::<f64>::new(config).is_err(),
                "min_clients={min_clients}, max_dropouts={max_dropouts} must be refused"
            );
        }
    }

    #[test]
    fn the_personalization_manager_takes_its_strategy_and_size() {
        let config = PersonalizationConfig {
            strategy: PersonalizationStrategy::MetaLearning {
                inner_lr: 0.01,
                outer_lr: 0.001,
            },
            local_adaptation: LocalAdaptationConfig::default(),
            clustering: ClusteringConfig::default(),
            meta_learning: MetaLearningConfig::default(),
            privacy_preserving: true,
        };
        let manager = match PersonalizationManager::<f64>::with_config(config, 128) {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!(matches!(
            manager.config().strategy,
            PersonalizationStrategy::MetaLearning { .. }
        ));
        assert!(manager.config().privacy_preserving);
        assert_eq!(
            manager.meta_learner().parameter_size(),
            128,
            "the meta-learner must be sized for the model, not left at 0"
        );

        // The compatibility constructor keeps the historical behaviour.
        let default_manager = match PersonalizationManager::<f64>::new() {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!(matches!(
            default_manager.config().strategy,
            PersonalizationStrategy::None
        ));
    }

    #[test]
    fn the_communication_optimizer_installs_the_configured_compression() {
        // `CompressionStrategy::None` was hardcoded twice, so the configured
        // strategy never reached the engine.
        let config = CommunicationConfig {
            compression: CompressionStrategy::TopK { k: 128 },
            lazy_aggregation: LazyAggregationConfig::default(),
            federated_dropout: FederatedDropoutConfig::default(),
            async_updates: AsyncUpdateConfig::default(),
            bandwidth_adaptation: BandwidthAdaptationConfig::default(),
        };
        let optimizer = match CommunicationOptimizer::<f64>::with_config(config) {
            Ok(optimizer) => optimizer,
            Err(err) => panic!("construction failed: {err}"),
        };
        match optimizer.compression_strategy() {
            CompressionStrategy::TopK { k } => assert_eq!(k, 128),
            other => panic!("the engine got {other:?} instead of the configured strategy"),
        }
        match optimizer.config().compression {
            CompressionStrategy::TopK { .. } => {}
            other => panic!("the config records {other:?}"),
        }
    }

    #[test]
    fn the_continual_learning_coordinator_configures_its_detector() {
        let config = ContinualLearningConfig {
            strategy: ContinualLearningStrategy::EWC { lambda: 0.5 },
            memory_management: MemoryManagementConfig::default(),
            task_detection: TaskDetectionConfig {
                enabled: true,
                detection_method: TaskDetectionMethod::GradientBased,
                sensitivity_threshold: 0.42,
                adaptation_delay: 3,
            },
            knowledge_transfer: KnowledgeTransferConfig::default(),
            forgetting_prevention: ForgettingPreventionConfig::default(),
        };
        let coordinator = match ContinualLearningCoordinator::<f64>::with_config(config) {
            Ok(coordinator) => coordinator,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!(matches!(
            coordinator.config().strategy,
            ContinualLearningStrategy::EWC { .. }
        ));
        assert!(
            (coordinator.task_detector().detection_threshold() - 0.42).abs() < 1e-12,
            "the configured sensitivity must reach the detector, got {}",
            coordinator.task_detector().detection_threshold()
        );
        assert!(coordinator.task_history().next().is_none());
    }

    #[test]
    fn an_unusable_task_detection_threshold_is_refused() {
        let config = ContinualLearningConfig {
            strategy: ContinualLearningStrategy::TaskAgnostic,
            memory_management: MemoryManagementConfig::default(),
            task_detection: TaskDetectionConfig {
                enabled: true,
                detection_method: TaskDetectionMethod::GradientBased,
                sensitivity_threshold: 0.0,
                adaptation_delay: 1,
            },
            knowledge_transfer: KnowledgeTransferConfig::default(),
            forgetting_prevention: ForgettingPreventionConfig::default(),
        };
        assert!(ContinualLearningCoordinator::<f64>::with_config(config).is_err());
    }

    #[test]
    fn the_adaptive_budget_manager_takes_its_configuration() {
        let config = AdaptiveBudgetConfig {
            enabled: true,
            ..AdaptiveBudgetConfig::default()
        };
        let manager = match AdaptiveBudgetManager::<f64>::with_config(config) {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!(
            manager.config().enabled,
            "the configured `enabled` flag must reach the manager"
        );
        assert!(manager.client_budget("nobody").is_none());

        let default_manager = match AdaptiveBudgetManager::<f64>::new() {
            Ok(manager) => manager,
            Err(err) => panic!("construction failed: {err}"),
        };
        assert!(!default_manager.config().enabled);
    }

    #[test]
    fn the_cross_device_manager_exposes_its_configuration() {
        let manager = CrossDevicePrivacyManager::<f64>::new(CrossDeviceConfig::default());
        assert!(!manager.config().user_level_privacy);
        assert!(manager.user_devices("nobody").is_empty());
    }

    #[test]
    fn fairness_weights_default_to_one_for_unknown_clients() {
        let monitor = FairnessMonitor::new();
        let weights = monitor.compute_fairness_weights(&["a".to_string(), "b".to_string()]);
        assert_eq!(weights.len(), 2);
        assert!(weights.values().all(|weight| (*weight - 1.0).abs() < 1e-12));
        assert_eq!(monitor.get_metrics().demographic_parity, 0.0);
    }
}
