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

use super::config::*;
use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Byzantine-robust aggregation, re-exported from the audited implementation in
/// [`crate::privacy::federated::byzantine_aggregation`].
///
/// Until 0.3.2 this module declared a second, field-identical
/// `ByzantineRobustAggregator` whose `client_reputations` and
/// `robust_estimators` were written once and never read, with its methods
/// supplied by an `impl` block in `coordinator.rs` that returned `Ok(0.9)` for
/// the robustness factor and an empty map for the reputations. The real engine
/// implements trimmed mean, coordinate-wise median, Krum, Multi-Krum, Bulyan,
/// centered clipping, reputation weighting and the outlier tests, and returns an
/// error rather than a plain mean when a method cannot be applied.
pub use super::super::federated::byzantine_aggregation::{
    AdaptivePrivacyAllocation, ByzantineRobustAggregator, OutlierDetectionResult, RobustEstimators,
    StatisticalAnalyzer, TestStatistic,
};

/// Cross-device privacy management, re-exported from
/// [`crate::privacy::federated::cross_device_manager`].
///
/// The copy this replaces stored `device_profiles` and `temporal_correlations`
/// that nothing read, so no user-level or temporal-correlation accounting
/// happened at all. The real manager tracks per-subject epsilon budgets across a
/// participation window.
pub use super::super::federated::cross_device_manager::{
    CrossDevicePrivacyManager, DeviceProfile, DeviceType, TemporalEvent, TemporalEventType,
};

// Advanced federated learning implementation structures

/// Personalized federated learning manager
pub struct PersonalizationManager<T: Float + Debug + Send + Sync + 'static> {
    config: PersonalizationConfig,
    client_models: HashMap<String, PersonalizedModel<T>>,
    global_model: Option<Array1<T>>,
    meta_learner: FederatedMetaLearner<T>,
}

/// Adaptive privacy budget manager
pub struct AdaptiveBudgetManager<T: Float + Debug + Send + Sync + 'static> {
    config: AdaptiveBudgetConfig,
    client_budgets: HashMap<String, AdaptiveBudget>,
    fairness_monitor: FairnessMonitor,
    _phantom: std::marker::PhantomData<T>,
}

/// Continual learning coordinator
pub struct ContinualLearningCoordinator<T: Float + Debug + Send + Sync + 'static> {
    config: ContinualLearningConfig,
    task_detector: TaskDetector<T>,
    task_history: VecDeque<TaskInfo>,
}

// Supporting implementation structures

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

/// Fairness monitor
pub struct FairnessMonitor {
    fairness_metrics: FairnessMetrics,
    client_fairness_scores: HashMap<String, f64>,
}

/// Fairness metrics
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    pub demographic_parity: f64,
    pub equalized_opportunity: f64,
    pub individual_fairness: f64,
    pub group_fairness: f64,
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

/// Task information
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: usize,
    pub start_round: usize,
    pub end_round: Option<usize>,
    pub task_description: String,
    pub performance_metrics: HashMap<String, f64>,
}

/// Secure aggregation protocol implementation.
///
/// Re-exported from [`crate::privacy::federated::secure_aggregation`]. Until
/// 0.3.2 this module declared a second `SecureAggregator<T>` whose entire state
/// -- `client_masks`, a mutex-guarded `shared_randomness` counter and
/// `round_keys` -- was written once at construction and never read, and whose
/// `aggregate_with_masks` was a plaintext mean. Two types with the same name,
/// one of which only pretended to mask, is exactly how a caller ends up
/// believing an unmasked mean is confidential. The pretender is gone.
pub use super::super::federated::secure_aggregation::SecureAggregator;

/// Privacy amplification analyzer
pub struct PrivacyAmplificationAnalyzer {
    pub(super) config: AmplificationConfig,
    pub(super) subsampling_history: VecDeque<SubsamplingEvent>,
    pub(super) amplification_factors: HashMap<String, f64>,
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
        }
    }

    /// Record a client's fairness score, which
    /// [`Self::compute_fairness_weights`] turns into a selection weight.
    ///
    /// Without this the score map could never be populated and every client's
    /// weight was unconditionally `1.0`.
    pub fn set_client_score(&mut self, client_id: String, score: f64) -> Result<()> {
        if !score.is_finite() || score < 0.0 {
            return Err(crate::error::OptimError::InvalidParameter(format!(
                "a fairness score must be non-negative and finite, got {score}"
            )));
        }
        self.client_fairness_scores.insert(client_id, score);
        Ok(())
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

// Default implementations for component creation

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
            meta_learner: FederatedMetaLearner::new(parameter_size),
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

    /// Apply an aggregated client update to the global model and return it.
    ///
    /// # Semantics
    ///
    /// `aggregated_update` is the cohort's *delta* (the FedAvg convention: each
    /// client uploads `local_params - global_params`, the server averages them).
    /// The global model therefore advances by `global += aggregated_update`. On
    /// the first call the manager holds no global model yet, so the update *is*
    /// the model.
    ///
    /// Before 0.3.2 this function was `Ok(aggregate.clone())` -- it returned the
    /// caller's own input, never touched `global_model`, and so the field was
    /// written once at construction and never read. A federation driving its
    /// global model through this function stayed at round one forever while the
    /// return value made it look as though every round had been applied.
    ///
    /// # Errors
    ///
    /// [`crate::error::OptimError::DimensionMismatch`] if the update's length differs from the
    /// stored global model's, since silently zero-extending or truncating would
    /// corrupt the model. [`crate::error::OptimError::InvalidParameter`] for an empty update.
    pub fn update_global_model(&mut self, aggregated_update: &Array1<T>) -> Result<Array1<T>> {
        if aggregated_update.is_empty() {
            return Err(crate::error::OptimError::InvalidParameter(
                "the aggregated update is empty; there is nothing to apply".to_string(),
            ));
        }
        match self.global_model.as_mut() {
            Some(model) => {
                if model.len() != aggregated_update.len() {
                    return Err(crate::error::OptimError::DimensionMismatch(format!(
                        "the aggregated update has {} coordinates but the global model has {}",
                        aggregated_update.len(),
                        model.len()
                    )));
                }
                for (slot, &delta) in model.iter_mut().zip(aggregated_update.iter()) {
                    *slot = *slot + delta;
                }
                Ok(model.clone())
            }
            None => {
                self.global_model = Some(aggregated_update.clone());
                Ok(aggregated_update.clone())
            }
        }
    }

    /// The current global model, or `None` before the first
    /// [`Self::update_global_model`].
    pub fn global_model(&self) -> Option<&Array1<T>> {
        self.global_model.as_ref()
    }

    /// Record a client's personalized model.
    ///
    /// Personalization strategies keep a per-client model alongside the global
    /// one; without this the `client_models` map could never be populated.
    pub fn set_client_model(&mut self, client_id: String, model: PersonalizedModel<T>) {
        self.client_models.insert(client_id, model);
    }

    /// The personalized model recorded for `client_id`, if any.
    pub fn client_model(&self, client_id: &str) -> Option<&PersonalizedModel<T>> {
        self.client_models.get(client_id)
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
            fairness_monitor: FairnessMonitor::new(),
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
        AdaptiveBudgetConfig, ByzantineRobustConfig, ByzantineRobustMethod, ClusteringConfig,
        ContinualLearningConfig, ContinualLearningStrategy, CrossDeviceConfig,
        ForgettingPreventionConfig, KnowledgeTransferConfig, LocalAdaptationConfig,
        MemoryManagementConfig, MetaLearningConfig, PersonalizationConfig, PersonalizationStrategy,
        ReputationSystemConfig, SecureAggregationConfig, StatisticalTestConfig,
        TaskDetectionConfig, TaskDetectionMethod,
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
        assert!((aggregator.statistical_analyzer().significance_level() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn an_unusable_byzantine_configuration_is_refused() {
        // `trim_ratio` is the *total* fraction removed, split evenly between the
        // two tails, so 0.5 (25% per tail) is legal; 1.0 and above would remove
        // everything.
        for (trim, byzantine) in [
            (1.0f64, 0.2f64),
            (1.5, 0.2),
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
        config.statistical_tests.significancelevel = 1.0;
        assert!(ByzantineRobustAggregator::<f64>::with_config(config).is_err());
    }

    /// The `SecureAggregator` name reachable from this module must resolve to
    /// the audited Bonawitz implementation, not to a re-introduced shell: the
    /// shell stored a `client_masks` map on the *server*, which is the opposite
    /// of secure aggregation.
    #[test]
    fn the_secure_aggregator_is_the_audited_implementation() {
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
        // Only the real implementation exposes the round/key-registration state
        // machine; the shell had no notion of a plan or of client keys.
        assert_eq!(aggregator.rounds_prepared(), 0);
        assert!(aggregator.current_plan().is_none());
        assert!(aggregator.modulus() > 0);

        for min_clients in [0usize, 1] {
            let config = SecureAggregationConfig {
                min_clients,
                max_dropouts: 0,
                ..SecureAggregationConfig::default()
            };
            assert!(
                SecureAggregator::<f64>::new(config).is_err(),
                "min_clients={min_clients} must be refused"
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
        assert!(manager.get_user_cluster("nobody").is_none());
        assert_eq!(manager.device_count(), 0);
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
