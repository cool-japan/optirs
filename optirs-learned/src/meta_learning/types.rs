//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
use crate::error::{OptimError, Result};
use optirs_core::optimizers::Optimizer;
#[allow(dead_code)]
use scirs2_core::ndarray::{Array1, Array2, Dimension};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

use super::functions::MetaLearner;

/// Adaptation step
#[derive(Debug, Clone)]
pub struct AdaptationStep<T: Float + Debug + Send + Sync + 'static> {
    /// Step number
    pub step: usize,
    /// Loss at this step
    pub loss: T,
    /// Gradient norm
    pub gradient_norm: T,
    /// Parameter change norm
    pub parameter_change_norm: T,
    /// Learning rate used
    pub learning_rate: T,
}
/// Meta-training metrics
#[derive(Debug, Clone)]
pub struct MetaTrainingMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Average adaptation speed
    pub avg_adaptation_speed: T,
    /// Generalization performance
    pub generalization_performance: T,
    /// Task diversity handled
    pub task_diversity: T,
    /// Gradient alignment score
    pub gradient_alignment: T,
}
/// Task identification methods
#[derive(Debug, Clone, Copy)]
pub enum TaskIdentificationMethod {
    Oracle,
    Learned,
    Clustering,
    EntropyBased,
    GradientBased,
}
/// Query evaluation metrics
#[derive(Debug, Clone)]
pub struct QueryEvaluationMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Mean squared error (for regression)
    pub mse: Option<T>,
    /// Classification accuracy (for classification)
    pub classification_accuracy: Option<T>,
    /// AUC score
    pub auc: Option<T>,
    /// Uncertainty estimation quality
    pub uncertainty_quality: T,
}
/// Meta-parameters for meta-learning
#[derive(Debug, Clone)]
pub struct MetaParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter values
    pub parameters: HashMap<String, Array1<T>>,
    /// Parameter metadata
    pub metadata: HashMap<String, String>,
}
/// MAML implementation over the shared linear task model
/// ([`super::linear_model`]).
///
/// The inner loop descends the *support-set* loss of that model with analytic
/// gradients; the outer loop applies FOMAML by default and the exact
/// second-order MAML meta-gradient when `config.second_order` is set (exact
/// because the model is linear in its parameters, so its MSE Hessian is
/// constant and the Hessian-vector product is closed-form).
pub struct MAMLLearner<T: Float + Debug + Send + Sync + 'static, D: Dimension> {
    /// MAML configuration
    pub(super) config: MAMLConfig<T>,
    /// Task adaptation history (most recent last, bounded)
    pub(super) adaptation_history: VecDeque<TaskAdaptationResult<T>>,
    /// Number of meta-training steps performed
    pub(super) meta_steps: usize,
    /// Dimension marker retained for API compatibility
    _phantom: std::marker::PhantomData<D>,
}

/// Maximum number of task adaptations retained for reporting.
pub(super) const MAML_HISTORY_CAPACITY: usize = 1000;

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
        D: Dimension,
    > MAMLLearner<T, D>
{
    pub fn new(config: MAMLConfig<T>) -> Result<Self> {
        if config.inner_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "MAML requires at least one inner step".to_string(),
            ));
        }
        Ok(Self {
            config,
            adaptation_history: VecDeque::with_capacity(MAML_HISTORY_CAPACITY),
            meta_steps: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Number of task adaptations recorded so far.
    pub fn adaptations_recorded(&self) -> usize {
        self.adaptation_history.len()
    }
}
impl<T: Float + Debug + Send + Sync + 'static + Default + Clone + std::iter::Sum, D: Dimension>
    MAMLLearner<T, D>
{
    /// Support-set loss of the shared linear model at `parameters`.
    pub(super) fn compute_support_loss(
        &self,
        task: &MetaTask<T>,
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<T> {
        super::linear_model::mse_loss(
            &task.support_set.features,
            &task.support_set.targets,
            parameters,
        )
    }

    /// Analytic gradient of the support-set loss at `parameters`.
    pub(super) fn compute_support_gradients(
        &self,
        task: &MetaTask<T>,
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut grads = super::linear_model::mse_gradients(
            &task.support_set.features,
            &task.support_set.targets,
            parameters,
        )?;
        if let Some(clip) = self.config.gradient_clip {
            if let Some(max_norm) = scirs2_core::numeric::NumCast::from(clip) {
                super::linear_model::clip_gradients(&mut grads, max_norm);
            }
        }
        Ok(grads)
    }

    /// Analytic gradient of the query-set loss at `parameters`.
    pub(super) fn compute_query_gradients(
        &self,
        task: &MetaTask<T>,
        parameters: &HashMap<String, Array1<T>>,
    ) -> Result<HashMap<String, Array1<T>>> {
        super::linear_model::mse_gradients(
            &task.query_set.features,
            &task.query_set.targets,
            parameters,
        )
    }
}
/// Memory selection criteria
#[derive(Debug, Clone, Copy)]
pub enum MemorySelectionCriteria {
    Random,
    GradientMagnitude,
    LossBased,
    Uncertainty,
    Diversity,
    TemporalProximity,
}
/// Task metadata
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Task name
    pub name: String,
    /// Task description
    pub description: String,
    /// Task properties
    pub properties: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Task source
    pub source: String,
}
/// Meta-training epoch
#[derive(Debug, Clone)]
pub struct MetaTrainingEpoch<T: Float + Debug + Send + Sync + 'static> {
    pub epoch: usize,
    pub training_result: MetaTrainingResult<T>,
    pub validation_result: MetaValidationResult<T>,
    pub meta_parameters: HashMap<String, Array1<T>>,
}
/// Task dataset
#[derive(Debug, Clone)]
pub struct TaskDataset<T: Float + Debug + Send + Sync + 'static> {
    /// Input features
    pub features: Vec<Array1<T>>,
    /// Target values
    pub targets: Vec<T>,
    /// Sample weights
    pub weights: Vec<T>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}
/// MAML configuration
#[derive(Debug, Clone)]
pub struct MAMLConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable second-order gradients
    pub second_order: bool,
    /// Inner learning rate
    pub inner_lr: T,
    /// Outer learning rate
    pub outer_lr: T,
    /// Number of inner steps
    pub inner_steps: usize,
    /// Allow unused parameters
    pub allow_unused: bool,
    /// Gradient clipping
    pub gradient_clip: Option<f64>,
}
/// Meta-training result
#[derive(Debug, Clone)]
pub struct MetaTrainingResult<T: Float + Debug + Send + Sync + 'static> {
    /// Meta-loss
    pub meta_loss: T,
    /// Per-task losses
    pub task_losses: Vec<T>,
    /// Meta-gradients
    pub meta_gradients: HashMap<String, Array1<T>>,
    /// Training metrics
    pub metrics: MetaTrainingMetrics<T>,
    /// Adaptation statistics
    pub adaptation_stats: AdaptationStatistics<T>,
}
/// Metric learning settings
#[derive(Debug, Clone)]
pub struct MetricLearningSettings {
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Learned metric parameters
    pub learned_metric: bool,
}
/// Multi-task learning result
#[derive(Debug, Clone)]
pub struct MultiTaskResult<T: Float + Debug + Send + Sync + 'static> {
    pub task_results: Vec<TaskResult<T>>,
    pub coordination_overhead: T,
    pub convergence_status: String,
}
/// Memory replay settings
#[derive(Debug, Clone)]
pub struct MemoryReplaySettings {
    /// Memory buffer size
    pub buffer_size: usize,
    /// Replay strategy
    pub replay_strategy: ReplayStrategy,
    /// Replay frequency
    pub replay_frequency: usize,
    /// Memory selection criteria
    pub selection_criteria: MemorySelectionCriteria,
}
/// Task types
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    Regression,
    Classification,
    Optimization,
    ReinforcementLearning,
    StructuredPrediction,
    Generative,
}
/// Interference mitigation strategies
#[derive(Debug, Clone, Copy)]
pub enum InterferenceMitigationStrategy {
    OrthogonalGradients,
    TaskSpecificLayers,
    AttentionMechanisms,
    MetaGradients,
}
/// Replay strategies
#[derive(Debug, Clone, Copy)]
pub enum ReplayStrategy {
    Random,
    GradientBased,
    UncertaintyBased,
    DiversityBased,
    Temporal,
}
/// Transfer learning settings
#[derive(Debug, Clone)]
pub struct TransferLearningSettings {
    /// Enable domain adaptation
    pub domain_adaptation: bool,
    /// Source domain weights
    pub source_domain_weights: Vec<f64>,
    /// Transfer learning strategies
    pub strategies: Vec<TransferStrategy>,
    /// Domain similarity measures
    pub similarity_measures: Vec<SimilarityMeasure>,
    /// Enable progressive transfer
    pub progressive_transfer: bool,
}
/// Validation result for meta-learning
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation loss
    pub validation_loss: f64,
    /// Additional validation metrics
    pub metrics: HashMap<String, f64>,
}
/// Transfer learning result
#[derive(Debug, Clone)]
pub struct TransferLearningResult<T: Float + Debug + Send + Sync + 'static> {
    pub transfer_efficiency: T,
    pub domain_adaptation_score: T,
    pub source_task_retention: T,
    pub target_task_performance: T,
}
/// Adaptation strategies
#[derive(Debug, Clone, Copy)]
pub enum AdaptationStrategy {
    /// Fine-tuning all parameters
    FullFineTuning,
    /// Fine-tuning only specific layers
    LayerWiseFineTuning,
    /// Parameter-efficient adaptation
    ParameterEfficient,
    /// Adaptation via learned learning rates
    LearnedLearningRates,
    /// Gradient-based adaptation
    GradientBased,
    /// Memory-based adaptation
    MemoryBased,
    /// Attention-based adaptation
    AttentionBased,
    /// Modular adaptation
    ModularAdaptation,
}
/// Meta-task representation
#[derive(Debug, Clone)]
pub struct MetaTask<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub id: String,
    /// Support set (training data for adaptation)
    pub support_set: TaskDataset<T>,
    /// Query set (test data for evaluation)
    pub query_set: TaskDataset<T>,
    /// Task metadata
    pub metadata: TaskMetadata,
    /// Task difficulty
    pub difficulty: T,
    /// Task domain
    pub domain: String,
    /// Task type
    pub task_type: TaskType,
}
/// Continual learning result
#[derive(Debug, Clone)]
pub struct ContinualLearningResult<T: Float + Debug + Send + Sync + 'static> {
    pub sequence_results: Vec<TaskResult<T>>,
    pub forgetting_measure: T,
    pub adaptation_efficiency: T,
}
/// Anti-forgetting strategies
#[derive(Debug, Clone, Copy)]
pub enum AntiForgettingStrategy {
    ElasticWeightConsolidation,
    SynapticIntelligence,
    MemoryReplay,
    ProgressiveNetworks,
    PackNet,
    Piggyback,
    HAT,
}
/// Training result for meta-learning
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Training loss
    pub training_loss: f64,
    /// Training metrics
    pub metrics: HashMap<String, f64>,
    /// Number of training steps
    pub steps: usize,
}
/// Meta-training results
#[derive(Debug, Clone)]
pub struct MetaTrainingResults<T: Float + Debug + Send + Sync + 'static> {
    pub final_parameters: HashMap<String, Array1<T>>,
    pub training_history: Vec<MetaTrainingEpoch<T>>,
    pub best_performance: T,
    pub total_epochs: usize,
}
/// Dataset metadata
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Number of samples
    pub num_samples: usize,
    /// Feature dimension
    pub feature_dim: usize,
    /// Data distribution type
    pub distribution_type: String,
    /// Noise level
    pub noise_level: f64,
}
/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStatistics<T: Float + Debug + Send + Sync + 'static> {
    pub algorithm: MetaLearningAlgorithm,
    pub total_tasks_seen: usize,
    pub adaptation_efficiency: T,
    pub transfer_success_rate: T,
    pub forgetting_measure: T,
    pub multitask_interference: T,
    pub few_shot_performance: T,
}
/// Task adaptation result
#[derive(Debug, Clone)]
pub struct TaskAdaptationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Adapted parameters
    pub adapted_parameters: HashMap<String, Array1<T>>,
    /// Adaptation trajectory
    pub adaptation_trajectory: Vec<AdaptationStep<T>>,
    /// Final adaptation loss
    pub final_loss: T,
    /// Adaptation metrics
    pub metrics: TaskAdaptationMetrics<T>,
}
/// Transfer strategies
#[derive(Debug, Clone, Copy)]
pub enum TransferStrategy {
    FeatureExtraction,
    FineTuning,
    DomainAdaptation,
    MultiTask,
    MetaTransfer,
    Progressive,
}
/// Augmentation strategies
#[derive(Debug, Clone, Copy)]
pub enum AugmentationStrategy {
    Geometric,
    Color,
    Noise,
    Mixup,
    CutMix,
    Learned,
}
/// Stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter stability
    pub parameter_stability: T,
    /// Performance stability
    pub performance_stability: T,
    /// Gradient stability
    pub gradient_stability: T,
    /// Catastrophic forgetting measure
    pub forgetting_measure: T,
}
/// Task result for meta-learning
#[derive(Debug, Clone)]
pub struct TaskResult<T: Float + Debug + Send + Sync + 'static> {
    pub task_id: String,
    pub loss: T,
    pub metrics: HashMap<String, T>,
}
/// Meta-learning algorithms
#[derive(Debug, Clone, Copy)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// First-Order MAML (FOMAML)
    FOMAML,
    /// Reptile algorithm
    Reptile,
    /// Meta-SGD
    MetaSGD,
    /// Learning to Learn by Gradient Descent
    L2L,
    /// Gradient-Based Meta-Learning
    GBML,
    /// Meta-Learning with Implicit Gradients
    IMaml,
    /// Prototypical Networks
    ProtoNet,
    /// Matching Networks
    MatchingNet,
    /// Relation Networks
    RelationNet,
    /// Memory-Augmented Neural Networks
    MANN,
    /// Meta-Learning with Warped Gradient Descent
    WarpGrad,
    /// Learned Gradient Descent
    LearnedGD,
}
/// Gradient balancing methods
#[derive(Debug, Clone, Copy)]
pub enum GradientBalancingMethod {
    Uniform,
    GradNorm,
    PCGrad,
    CAGrad,
    NashMTL,
}
/// Query evaluation result
#[derive(Debug, Clone)]
pub struct QueryEvaluationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Query set loss
    pub query_loss: T,
    /// Prediction accuracy
    pub accuracy: T,
    /// Per-sample predictions
    pub predictions: Vec<T>,
    /// Confidence scores
    pub confidence_scores: Vec<T>,
    /// Evaluation metrics
    pub metrics: QueryEvaluationMetrics<T>,
}
/// Few-shot learning algorithms
#[derive(Debug, Clone, Copy)]
pub enum FewShotAlgorithm {
    Prototypical,
    Matching,
    Relation,
    MAML,
    Reptile,
    MetaOptNet,
}
/// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
}
/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Convergence steps per task
    pub convergence_steps: Vec<usize>,
    /// Final losses per task
    pub final_losses: Vec<T>,
    /// Adaptation efficiency
    pub adaptation_efficiency: T,
    /// Stability metrics
    pub stability_metrics: StabilityMetrics<T>,
}
/// Meta-validation result
#[derive(Debug, Clone)]
pub struct MetaValidationResult<T: Float + Debug + Send + Sync + 'static> {
    pub performance: T,
    pub adaptation_speed: T,
    pub generalization_gap: T,
    pub task_specific_metrics: HashMap<String, T>,
}
/// Loss functions
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    Hinge,
    Huber,
}
/// Continual learning settings
#[derive(Debug, Clone)]
pub struct ContinualLearningSettings {
    /// Catastrophic forgetting mitigation
    pub anti_forgetting_strategies: Vec<AntiForgettingStrategy>,
    /// Memory replay settings
    pub memory_replay: MemoryReplaySettings,
    /// Task identification method
    pub task_identification: TaskIdentificationMethod,
    /// Plasticity-stability trade-off
    pub plasticity_stability_balance: f64,
}
/// Task weighting strategies
#[derive(Debug, Clone, Copy)]
pub enum TaskWeightingStrategy {
    Uniform,
    UncertaintyBased,
    GradientMagnitude,
    PerformanceBased,
    Adaptive,
    Learned,
}
/// Multi-task settings
#[derive(Debug, Clone)]
pub struct MultiTaskSettings {
    /// Task weighting strategy
    pub task_weighting: TaskWeightingStrategy,
    /// Gradient balancing method
    pub gradient_balancing: GradientBalancingMethod,
    /// Task interference mitigation
    pub interference_mitigation: InterferenceMitigationStrategy,
    /// Shared representation learning
    pub shared_representation: SharedRepresentationStrategy,
}
/// Shared representation strategies
#[derive(Debug, Clone, Copy)]
pub enum SharedRepresentationStrategy {
    HardSharing,
    SoftSharing,
    HierarchicalSharing,
    AttentionBased,
    Modular,
}
/// Distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Mahalanobis,
    Learned,
}
/// Task adaptation metrics
#[derive(Debug, Clone)]
pub struct TaskAdaptationMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Convergence speed
    pub convergence_speed: T,
    /// Final performance
    pub final_performance: T,
    /// Adaptation efficiency
    pub efficiency: T,
    /// Robustness to noise
    pub robustness: T,
}
/// Few-shot learning settings
#[derive(Debug, Clone)]
pub struct FewShotSettings {
    /// Number of shots (examples per class)
    pub num_shots: usize,
    /// Number of ways (classes)
    pub num_ways: usize,
    /// Few-shot algorithm
    pub algorithm: FewShotAlgorithm,
    /// Metric learning settings
    pub metric_learning: MetricLearningSettings,
    /// Augmentation strategies
    pub augmentation_strategies: Vec<AugmentationStrategy>,
}
/// Domain similarity measures
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMeasure {
    CosineDistance,
    KLDivergence,
    WassersteinDistance,
    CentralMomentDiscrepancy,
    MaximumMeanDiscrepancy,
}
/// Meta-learning configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Meta-learning algorithm
    pub algorithm: MetaLearningAlgorithm,
    /// Number of inner loop steps
    pub inner_steps: usize,
    /// Number of outer loop steps
    pub outer_steps: usize,
    /// Meta-learning rate
    pub meta_learning_rate: f64,
    /// Inner learning rate
    pub inner_learning_rate: f64,
    /// Task batch size
    pub task_batch_size: usize,
    /// Support set size per task
    pub support_set_size: usize,
    /// Query set size per task
    pub query_set_size: usize,
    /// Enable second-order gradients
    pub second_order: bool,
    /// Gradient clipping threshold
    pub gradient_clip: f64,
    /// Adaptation strategies
    pub adaptation_strategies: Vec<AdaptationStrategy>,
    /// Transfer learning settings
    pub transfer_settings: TransferLearningSettings,
    /// Continual learning settings
    pub continual_settings: ContinualLearningSettings,
    /// Multi-task settings
    pub multitask_settings: MultiTaskSettings,
    /// Few-shot learning settings
    pub few_shot_settings: FewShotSettings,
    /// Enable meta-regularization
    pub enable_meta_regularization: bool,
    /// Meta-regularization strength
    pub meta_regularization_strength: f64,
    /// Task sampling strategy
    pub task_sampling_strategy: TaskSamplingStrategy,
}
/// Task sampling strategies
#[derive(Debug, Clone, Copy)]
pub enum TaskSamplingStrategy {
    Uniform,
    Curriculum,
    DifficultyBased,
    DiversityBased,
    ActiveLearning,
    Adversarial,
}
/// Few-shot learning result
#[derive(Debug, Clone)]
pub struct FewShotResult<T: Float + Debug + Send + Sync + 'static> {
    pub accuracy: T,
    pub confidence: T,
    pub adaptation_steps: usize,
    pub uncertainty_estimates: Vec<T>,
}

impl Default for MetaLearningConfig {
    /// A small, valid configuration: first-order MAML over the shared linear
    /// task model with uniform task sampling.
    fn default() -> Self {
        Self {
            algorithm: MetaLearningAlgorithm::MAML,
            inner_steps: 5,
            outer_steps: 100,
            meta_learning_rate: 0.3,
            inner_learning_rate: 0.4,
            task_batch_size: 8,
            support_set_size: 6,
            query_set_size: 6,
            second_order: false,
            gradient_clip: 10.0,
            adaptation_strategies: vec![AdaptationStrategy::FullFineTuning],
            transfer_settings: TransferLearningSettings {
                domain_adaptation: true,
                source_domain_weights: vec![1.0],
                strategies: vec![TransferStrategy::FineTuning],
                similarity_measures: vec![SimilarityMeasure::CosineDistance],
                progressive_transfer: false,
            },
            continual_settings: ContinualLearningSettings {
                anti_forgetting_strategies: vec![
                    AntiForgettingStrategy::ElasticWeightConsolidation,
                ],
                memory_replay: MemoryReplaySettings {
                    buffer_size: 256,
                    replay_strategy: ReplayStrategy::Random,
                    replay_frequency: 8,
                    selection_criteria: MemorySelectionCriteria::Random,
                },
                task_identification: TaskIdentificationMethod::Oracle,
                plasticity_stability_balance: 0.5,
            },
            multitask_settings: MultiTaskSettings {
                task_weighting: TaskWeightingStrategy::Uniform,
                gradient_balancing: GradientBalancingMethod::Uniform,
                interference_mitigation: InterferenceMitigationStrategy::OrthogonalGradients,
                shared_representation: SharedRepresentationStrategy::HardSharing,
            },
            few_shot_settings: FewShotSettings {
                num_shots: 5,
                num_ways: 2,
                algorithm: FewShotAlgorithm::MAML,
                metric_learning: MetricLearningSettings {
                    distance_metric: DistanceMetric::Euclidean,
                    embedding_dim: 16,
                    learned_metric: false,
                },
                augmentation_strategies: Vec::new(),
            },
            enable_meta_regularization: false,
            meta_regularization_strength: 0.0,
            task_sampling_strategy: TaskSamplingStrategy::Uniform,
        }
    }
}

impl MetaLearningConfig {
    /// Reject configurations that cannot produce a meaningful meta-training run.
    pub fn validate(&self) -> Result<()> {
        if self.inner_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "inner_steps must be greater than zero".to_string(),
            ));
        }
        if self.task_batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "task_batch_size must be greater than zero".to_string(),
            ));
        }
        if !(self.meta_learning_rate.is_finite() && self.meta_learning_rate > 0.0) {
            return Err(OptimError::InvalidConfig(
                "meta_learning_rate must be finite and positive".to_string(),
            ));
        }
        if !(self.inner_learning_rate.is_finite() && self.inner_learning_rate > 0.0) {
            return Err(OptimError::InvalidConfig(
                "inner_learning_rate must be finite and positive".to_string(),
            ));
        }
        if !(self.gradient_clip.is_finite() && self.gradient_clip > 0.0) {
            return Err(OptimError::InvalidConfig(
                "gradient_clip must be finite and positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.continual_settings.plasticity_stability_balance) {
            return Err(OptimError::InvalidConfig(
                "plasticity_stability_balance must lie in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}
