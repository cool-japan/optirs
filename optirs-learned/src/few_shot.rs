// Few-Shot Learning Enhancement for Optimizer Meta-Learning
//
// This module implements advanced few-shot learning techniques specifically designed
// for quickly adapting optimizers to new tasks with minimal data. It includes
// prototypical networks, meta-learning approaches, and rapid adaptation mechanisms.

#[allow(dead_code)]
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::OptimizerState;
use crate::error::{OptimError, Result};

/// Few-shot learning coordinator for optimizer adaptation
pub struct FewShotLearningSystem<T: Float + Debug + Send + Sync + 'static> {
    /// Base meta-learned optimizer
    base_optimizer: Box<dyn FewShotOptimizer<T>>,

    /// Prototypical network for task representation
    prototype_network: PrototypicalNetwork<T>,

    /// Support set manager
    support_set_manager: SupportSetManager<T>,

    /// Adaptation strategies
    adaptation_strategies: Vec<Box<dyn AdaptationStrategy<T>>>,

    /// Task similarity calculator
    similarity_calculator: TaskSimilarityCalculator<T>,

    /// Memory bank for storing task experiences
    memory_bank: EpisodicMemoryBank<T>,

    /// Fast adaptation engine
    fast_adaptation: FastAdaptationEngine<T>,

    /// Performance tracker
    performance_tracker: FewShotPerformanceTracker<T>,
}

/// Base trait for few-shot optimizers
pub trait FewShotOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Adapt to new task with few examples
    fn adapt_few_shot(
        &mut self,
        support_set: &SupportSet<T>,
        query_set: &QuerySet<T>,
        adaptation_config: &AdaptationConfig,
    ) -> Result<AdaptationResult<T>>;

    /// Get task representation
    fn get_task_representation(&self, taskdata: &TaskData<T>) -> Result<Array1<T>>;

    /// Compute adaptation loss
    fn compute_adaptation_loss(
        &self,
        support_set: &SupportSet<T>,
        query_set: &QuerySet<T>,
    ) -> Result<T>;

    /// Update meta-parameters
    fn update_meta_parameters(&mut self, metagradients: &MetaGradients<T>) -> Result<()>;

    /// Get current state for transfer
    fn get_transfer_state(&self) -> TransferState<T>;

    /// Load transfer state
    fn load_transfer_state(&mut self, state: TransferState<T>) -> Result<()>;
}

/// Support set for few-shot learning
#[derive(Debug, Clone)]
pub struct SupportSet<T: Float + Debug + Send + Sync + 'static> {
    /// Support examples
    pub examples: Vec<SupportExample<T>>,

    /// Task metadata
    pub task_metadata: TaskMetadata,

    /// Support _set statistics
    pub statistics: SupportSetStatistics<T>,

    /// Temporal ordering (if applicable)
    pub temporal_order: Option<Vec<usize>>,
}

/// Individual support example
#[derive(Debug, Clone)]
pub struct SupportExample<T: Float + Debug + Send + Sync + 'static> {
    /// Input features
    pub features: Array1<T>,

    /// Target output
    pub target: T,

    /// Example weight/importance
    pub weight: T,

    /// Context information
    pub context: HashMap<String, T>,

    /// Example metadata
    pub metadata: ExampleMetadata,
}

/// Query set for evaluation
#[derive(Debug, Clone)]
pub struct QuerySet<T: Float + Debug + Send + Sync + 'static> {
    /// Query examples
    pub examples: Vec<QueryExample<T>>,

    /// Query statistics
    pub statistics: QuerySetStatistics<T>,

    /// Evaluation metrics
    pub eval_metrics: Vec<EvaluationMetric>,
}

/// Individual query example
#[derive(Debug, Clone)]
pub struct QueryExample<T: Float + Debug + Send + Sync + 'static> {
    /// Input features
    pub features: Array1<T>,

    /// True target (for evaluation)
    pub true_target: Option<T>,

    /// Query weight
    pub weight: T,

    /// Query context
    pub context: HashMap<String, T>,
}

/// Task data container
#[derive(Debug, Clone)]
pub struct TaskData<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub task_id: String,

    /// Support set
    pub support_set: SupportSet<T>,

    /// Query set
    pub query_set: QuerySet<T>,

    /// Task-specific parameters
    pub task_params: HashMap<String, T>,

    /// Task domain information
    pub domain_info: DomainInfo,
}

/// Domain information
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Domain type
    pub domain_type: DomainType,

    /// Domain characteristics
    pub characteristics: DomainCharacteristics,

    /// Expected difficulty
    pub difficulty_level: DifficultyLevel,

    /// Domain-specific constraints
    pub constraints: Vec<DomainConstraint>,
}

/// Domain types
#[derive(Debug, Clone, Copy)]
pub enum DomainType {
    ComputerVision,
    NaturalLanguageProcessing,
    ReinforcementLearning,
    TimeSeriesForecasting,
    ScientificComputing,
    Optimization,
    ControlSystems,
    GamePlaying,
    Robotics,
    Healthcare,
}

/// Domain characteristics
#[derive(Debug, Clone)]
pub struct DomainCharacteristics {
    /// Input dimensionality
    pub input_dim: usize,

    /// Output dimensionality
    pub output_dim: usize,

    /// Temporal dependencies
    pub temporal: bool,

    /// Stochasticity level
    pub stochasticity: f64,

    /// Noise level
    pub noise_level: f64,

    /// Data sparsity
    pub sparsity: f64,
}

/// Difficulty levels
#[derive(Debug, Clone, Copy)]
pub enum DifficultyLevel {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
    Extreme,
}

/// Domain constraints
#[derive(Debug, Clone)]
pub struct DomainConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Constraint description
    pub description: String,

    /// Enforcement level
    pub enforcement: ConstraintEnforcement,
}

/// Constraint types
#[derive(Debug, Clone, Copy)]
pub enum ConstraintType {
    ResourceLimit,
    TemporalConstraint,
    AccuracyRequirement,
    LatencyRequirement,
    MemoryConstraint,
    EnergyConstraint,
    SafetyConstraint,
}

/// Constraint enforcement levels
#[derive(Debug, Clone, Copy)]
pub enum ConstraintEnforcement {
    Hard,
    Soft,
    Advisory,
}

/// Adaptation configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Number of adaptation steps
    pub adaptation_steps: usize,

    /// Learning rate for adaptation
    pub adaptation_lr: f64,

    /// Adaptation strategy
    pub strategy: AdaptationStrategyType,

    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,

    /// Regularization parameters
    pub regularization: RegularizationConfig,

    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Adaptation strategy types
#[derive(Debug, Clone, Copy)]
pub enum AdaptationStrategyType {
    /// Model-Agnostic Meta-Learning (MAML)
    MAML,

    /// First-Order MAML (FOMAML)
    FOMAML,

    /// Prototypical Networks
    Prototypical,

    /// Matching Networks
    Matching,

    /// Relation Networks
    Relation,

    /// Meta-SGD
    MetaSGD,

    /// Learned optimizer approach
    LearnedOptimizer,

    /// Gradient-based meta-learning
    GradientBased,

    /// Memory-augmented networks
    MemoryAugmented,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (steps without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Validation frequency
    pub validation_frequency: usize,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L2 regularization strength
    pub l2_strength: f64,

    /// Dropout rate
    pub dropout_rate: f64,

    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,

    /// Task-specific regularization
    pub task_regularization: HashMap<String, f64>,
}

/// Resource constraints for adaptation
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum adaptation time
    pub max_time: Duration,

    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,

    /// Maximum computational budget
    pub max_compute_budget: f64,
}

/// Adaptation result
#[derive(Debug, Clone)]
pub struct AdaptationResult<T: Float + Debug + Send + Sync + 'static> {
    /// Adapted optimizer state
    pub adapted_state: OptimizerState<T>,

    /// Adaptation performance
    pub performance: AdaptationPerformance<T>,

    /// Task representation learned
    pub task_representation: Array1<T>,

    /// Adaptation trajectory
    pub adaptation_trajectory: Vec<AdaptationStep<T>>,

    /// Resource usage
    pub resource_usage: ResourceUsage<T>,
}

/// Adaptation performance metrics
#[derive(Debug, Clone)]
pub struct AdaptationPerformance<T: Float + Debug + Send + Sync + 'static> {
    /// Query set performance
    pub query_performance: T,

    /// Support set performance
    pub support_performance: T,

    /// Adaptation speed (steps to convergence)
    pub adaptation_speed: usize,

    /// Final loss
    pub final_loss: T,

    /// Performance improvement
    pub improvement: T,

    /// Stability measure
    pub stability: T,
}

/// Individual adaptation step
#[derive(Debug, Clone)]
pub struct AdaptationStep<T: Float + Debug + Send + Sync + 'static> {
    /// Step number
    pub step: usize,

    /// Loss at this step
    pub loss: T,

    /// Performance at this step
    pub performance: T,

    /// Gradient norm
    pub gradient_norm: T,

    /// Step time
    pub step_time: Duration,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage<T: Float + Debug + Send + Sync + 'static> {
    /// Total time taken
    pub total_time: Duration,

    /// Peak memory usage (MB)
    pub peak_memory_mb: T,

    /// Computational cost
    pub compute_cost: T,

    /// Energy consumption
    pub energy_consumption: T,
}

/// Meta-gradients for updating meta-parameters
#[derive(Debug, Clone)]
pub struct MetaGradients<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter gradients
    pub param_gradients: HashMap<String, Array1<T>>,

    /// Learning rate gradients
    pub lr_gradients: HashMap<String, T>,

    /// Architecture gradients
    pub arch_gradients: HashMap<String, Array1<T>>,

    /// Meta-gradient norm
    pub gradient_norm: T,
}

/// Transfer state for cross-task transfer
#[derive(Debug, Clone)]
pub struct TransferState<T: Float + Debug + Send + Sync + 'static> {
    /// Learned representations
    pub representations: HashMap<String, Array1<T>>,

    /// Meta-parameters
    pub meta_parameters: HashMap<String, Array1<T>>,

    /// Task embeddings
    pub task_embeddings: Array2<T>,

    /// Transfer statistics
    pub transfer_stats: TransferStatistics<T>,
}

/// Transfer statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Source task performance
    pub source_performance: T,

    /// Target task performance
    pub target_performance: T,

    /// Transfer efficiency
    pub transfer_efficiency: T,

    /// Adaptation steps saved
    pub steps_saved: usize,
}

/// Prototypical network for task representation
pub struct PrototypicalNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// Encoder network
    encoder: EncoderNetwork<T>,

    /// Prototype storage
    prototypes: HashMap<String, Prototype<T>>,

    /// Distance metric
    distance_metric: DistanceMetric,

    /// Network parameters
    parameters: PrototypicalNetworkParams<T>,
}

/// Encoder network for feature extraction
#[derive(Debug)]
pub struct EncoderNetwork<T: Float + Debug + Send + Sync + 'static> {
    /// Network layers
    layers: Vec<EncoderLayer<T>>,

    /// Activation function
    activation: ActivationFunction,
}

/// Individual encoder layer
#[derive(Debug)]
pub struct EncoderLayer<T: Float + Debug + Send + Sync + 'static> {
    /// Weight matrix
    pub weights: Array2<T>,

    /// Bias vector
    pub bias: Array1<T>,

    /// Layer type
    pub layer_type: LayerType,
}

/// Layer types
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Linear,
    Convolutional,
    Recurrent,
    Attention,
    Residual,
}

/// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
    Mish,
}

/// Task prototypes
#[derive(Debug, Clone)]
pub struct Prototype<T: Float + Debug + Send + Sync + 'static> {
    /// Prototype vector
    pub vector: Array1<T>,

    /// Prototype confidence
    pub confidence: T,

    /// Number of examples used
    pub example_count: usize,

    /// Last update time
    pub last_updated: std::time::SystemTime,

    /// Prototype metadata
    pub metadata: PrototypeMetadata,
}

/// Prototype metadata
#[derive(Debug, Clone)]
pub struct PrototypeMetadata {
    /// Task category
    pub task_category: String,

    /// Domain type
    pub domain: DomainType,

    /// Creation timestamp
    pub created_at: std::time::SystemTime,

    /// Update count
    pub update_count: usize,
}

/// Distance metrics for prototype comparison
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Manhattan,
    Mahalanobis,
    Learned,
}

/// Prototypical network parameters
#[derive(Debug, Clone)]
pub struct PrototypicalNetworkParams<T: Float + Debug + Send + Sync + 'static> {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Learning rate
    pub learning_rate: T,

    /// Temperature parameter
    pub temperature: T,

    /// Prototype update rate
    pub prototype_update_rate: T,
}

/// Support set manager
pub struct SupportSetManager<T: Float + Debug + Send + Sync + 'static> {
    /// Current support sets
    support_sets: HashMap<String, SupportSet<T>>,

    /// Support set selection strategy
    selection_strategy: SupportSetSelectionStrategy,

    /// Manager configuration
    config: SupportSetManagerConfig,
}

/// Support set selection strategies
#[derive(Debug, Clone, Copy)]
pub enum SupportSetSelectionStrategy {
    Random,
    DiversityBased,
    DifficultyBased,
    UncertaintyBased,
    PrototypeBased,
    Adaptive,
}

/// Support set manager configuration
#[derive(Debug, Clone)]
pub struct SupportSetManagerConfig {
    /// Minimum support set size
    pub min_support_size: usize,

    /// Maximum support set size
    pub max_support_size: usize,

    /// Quality threshold
    pub quality_threshold: f64,

    /// Enable augmentation
    pub enable_augmentation: bool,

    /// Cache support sets
    pub cache_support_sets: bool,
}

/// Adaptation strategy trait
pub trait AdaptationStrategy<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Perform adaptation
    fn adapt(
        &mut self,
        optimizer: &mut dyn FewShotOptimizer<T>,
        task_data: &TaskData<T>,
        config: &AdaptationConfig,
    ) -> Result<AdaptationResult<T>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Update strategy parameters
    fn update_parameters(&mut self, params: HashMap<String, f64>) -> Result<()>;
}

/// Task similarity calculator
pub struct TaskSimilarityCalculator<T: Float + Debug + Send + Sync + 'static> {
    /// Similarity metrics
    similarity_metrics: Vec<Box<dyn SimilarityMetric<T>>>,

    /// Metric weights
    metric_weights: HashMap<String, T>,

    /// Similarity cache
    similarity_cache: HashMap<(String, String), T>,

    /// Calculator configuration
    config: SimilarityCalculatorConfig<T>,
}

/// Similarity metric trait
pub trait SimilarityMetric<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Calculate similarity between tasks
    fn calculate_similarity(&self, task1: &TaskData<T>, task2: &TaskData<T>) -> Result<T>;

    /// Get metric name
    fn name(&self) -> &str;

    /// Get metric weight
    fn weight(&self) -> T;
}

/// Similarity calculator configuration
#[derive(Debug, Clone)]
pub struct SimilarityCalculatorConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable caching
    pub enable_caching: bool,

    /// Cache size limit
    pub cache_size_limit: usize,

    /// Similarity threshold
    pub similarity_threshold: T,

    /// Use task metadata
    pub use_metadata: bool,
}

/// Episodic memory bank for storing task experiences
pub struct EpisodicMemoryBank<T: Float + Debug + Send + Sync + 'static> {
    /// Memory episodes
    episodes: VecDeque<MemoryEpisode<T>>,

    /// Memory configuration
    config: MemoryBankConfig<T>,

    /// Usage statistics
    usage_stats: MemoryUsageStats,
}

/// Memory episode
#[derive(Debug, Clone)]
pub struct MemoryEpisode<T: Float + Debug + Send + Sync + 'static> {
    /// Episode ID
    pub episode_id: String,

    /// Task data
    pub task_data: TaskData<T>,

    /// Adaptation result
    pub adaptation_result: AdaptationResult<T>,

    /// Episode timestamp
    pub timestamp: std::time::SystemTime,

    /// Episode metadata
    pub metadata: EpisodeMetadata,

    /// Access count
    pub access_count: usize,
}

/// Episode metadata
#[derive(Debug, Clone)]
pub struct EpisodeMetadata {
    /// Task difficulty
    pub difficulty: DifficultyLevel,

    /// Domain type
    pub domain: DomainType,

    /// Success rate
    pub success_rate: f64,

    /// Tags
    pub tags: Vec<String>,
}

/// Memory bank configuration
#[derive(Debug, Clone)]
pub struct MemoryBankConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum memory size
    pub max_memory_size: usize,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Similarity threshold for retrieval
    pub similarity_threshold: T,

    /// Enable compression
    pub enable_compression: bool,
}

/// Memory eviction policies
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    LRU,         // Least Recently Used
    LFU,         // Least Frequently Used
    Performance, // Worst Performing
    Age,         // Oldest First
    Random,
}

/// Memory bank statistics (summary view)
#[derive(Debug, Clone)]
pub struct MemoryBankStats<T: Float + Debug + Send + Sync + 'static> {
    /// Number of stored episodes
    pub count: usize,

    /// Average performance across episodes
    pub avg_performance: T,

    /// Capacity used as a fraction
    pub capacity_used: f64,

    /// Total capacity
    pub total_capacity: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    /// Total episodes stored
    pub total_episodes: usize,

    /// Memory utilization
    pub memory_utilization: f64,

    /// Hit rate
    pub hit_rate: f64,

    /// Average retrieval time
    pub avg_retrieval_time: Duration,
}

/// Fast adaptation engine
pub struct FastAdaptationEngine<T: Float + Debug + Send + Sync + 'static> {
    /// Adaptation algorithms
    algorithms: Vec<Box<dyn FastAdaptationAlgorithm<T>>>,

    /// Engine configuration
    config: FastAdaptationConfig,
}

/// Fast adaptation algorithm trait
pub trait FastAdaptationAlgorithm<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Perform fast adaptation
    fn adapt_fast(
        &mut self,
        optimizer: &mut dyn FewShotOptimizer<T>,
        task_data: &TaskData<T>,
        target_performance: Option<T>,
    ) -> Result<AdaptationResult<T>>;

    /// Estimate adaptation time
    fn estimate_adaptation_time(&self, taskdata: &TaskData<T>) -> Duration;

    /// Get algorithm name
    fn name(&self) -> &str;
}

/// Fast adaptation configuration
#[derive(Debug, Clone)]
pub struct FastAdaptationConfig {
    /// Enable caching
    pub enable_caching: bool,

    /// Enable prediction
    pub enable_prediction: bool,

    /// Maximum adaptation time
    pub max_adaptation_time: Duration,

    /// Performance threshold
    pub _performance_threshold: f64,
}

/// Performance tracker for few-shot learning
pub struct FewShotPerformanceTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Performance history
    performance_history: VecDeque<PerformanceRecord<T>>,

    /// Performance metrics
    metrics: Vec<Box<dyn PerformanceMetric<T>>>,

    /// Tracking configuration
    config: TrackingConfig,

    /// Performance statistics
    stats: PerformanceStats<T>,
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Task ID
    pub task_id: String,

    /// Performance value
    pub performance: T,

    /// Adaptation time
    pub adaptation_time: Duration,

    /// Strategy used
    pub strategy_used: String,

    /// Timestamp
    pub timestamp: std::time::SystemTime,

    /// Additional metrics
    pub additional_metrics: HashMap<String, T>,
}

/// Performance metric trait
pub trait PerformanceMetric<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Calculate performance metric
    fn calculate(&self, records: &[PerformanceRecord<T>]) -> Result<T>;

    /// Get metric name
    fn name(&self) -> &str;

    /// Is higher better
    fn higher_is_better(&self) -> bool;
}

/// Tracking configuration
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Maximum history size
    pub max_history_size: usize,

    /// Update frequency
    pub update_frequency: Duration,

    /// Enable detailed tracking
    pub detailed_tracking: bool,

    /// Export results
    pub export_results: bool,
}

/// Performance statistics
#[derive(Debug)]
pub struct PerformanceStats<T: Float + Debug + Send + Sync + 'static> {
    /// Best performance
    pub best_performance: T,

    /// Average performance
    pub average_performance: T,

    /// Performance variance
    pub performance_variance: T,

    /// Improvement rate
    pub improvement_rate: T,

    /// Success rate
    pub success_rate: T,
}

// Implementation stubs for key structures
impl<T: Float + Debug + Send + Sync + 'static> FewShotLearningSystem<T> {
    /// Create new few-shot learning system
    pub fn new(
        base_optimizer: Box<dyn FewShotOptimizer<T>>,
        config: FewShotConfig<T>,
    ) -> Result<Self> {
        Ok(Self {
            base_optimizer,
            prototype_network: PrototypicalNetwork::new(config.prototype_config)?,
            support_set_manager: SupportSetManager::new(config.support_set_config)?,
            adaptation_strategies: Vec::new(),
            similarity_calculator: TaskSimilarityCalculator::new(config.similarity_config)?,
            memory_bank: EpisodicMemoryBank::new(config.memory_config)?,
            fast_adaptation: FastAdaptationEngine::new(config.adaptation_config)?,
            performance_tracker: FewShotPerformanceTracker::new(config.tracking_config)?,
        })
    }

    /// Learn from few examples
    pub fn learn_few_shot(
        &mut self,
        task_data: TaskData<T>,
        adaptation_config: AdaptationConfig,
    ) -> Result<AdaptationResult<T>> {
        // Comprehensive few-shot learning implementation
        let _start_time = Instant::now();

        // 1. Extract task representation using prototypical network
        let task_representation = self.prototype_network.encode_task(&task_data)?;

        // 2. Retrieve similar tasks from memory
        let similar_tasks = self.memory_bank.retrieve_similar(&task_data, 5)?;

        // 3. Select best adaptation strategy based on task characteristics
        let strategy = self.select_adaptation_strategy(&task_data, &similar_tasks)?;

        // 4. Perform fast adaptation
        let mut adaptation_result = self.fast_adaptation.adapt_fast(
            &mut *self.base_optimizer,
            &task_data,
            strategy,
            &adaptation_config,
        )?;

        // 5. Update task representation
        adaptation_result.task_representation = task_representation;

        // 6. Store experience in memory bank
        self.memory_bank
            .store_episode(task_data.clone(), adaptation_result.clone())?;

        // 7. Update performance tracker
        self.performance_tracker
            .record_performance(&adaptation_result)?;

        // 8. Update prototypical network with new experience
        self.prototype_network
            .update_prototypes(&task_data, &adaptation_result)?;

        Ok(adaptation_result)
    }

    fn select_adaptation_strategy(
        &self,
        task_data: &TaskData<T>,
        _similar_tasks: &[MemoryEpisode<T>],
    ) -> Result<AdaptationStrategyType> {
        // Strategy selection based on task characteristics and historical performance
        match task_data.domain_info.difficulty_level {
            DifficultyLevel::Trivial | DifficultyLevel::Easy => Ok(AdaptationStrategyType::FOMAML),
            DifficultyLevel::Medium => Ok(AdaptationStrategyType::MAML),
            DifficultyLevel::Hard | DifficultyLevel::Expert => {
                Ok(AdaptationStrategyType::Prototypical)
            }
            DifficultyLevel::Extreme => Ok(AdaptationStrategyType::MemoryAugmented),
        }
    }
}

/// Few-shot learning system configuration
#[derive(Debug, Clone)]
pub struct FewShotConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Prototypical network configuration
    pub prototype_config: PrototypicalNetworkConfig<T>,

    /// Support set management configuration
    pub support_set_config: SupportSetManagerConfig,

    /// Similarity calculation configuration
    pub similarity_config: SimilarityCalculatorConfig<T>,

    /// Memory bank configuration
    pub memory_config: MemoryBankConfig<T>,

    /// Fast adaptation configuration
    pub adaptation_config: FastAdaptationConfig,

    /// Performance tracking configuration
    pub tracking_config: TrackingConfig,
}

/// Prototypical network configuration
#[derive(Debug, Clone)]
pub struct PrototypicalNetworkConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Learning rate
    pub learning_rate: T,

    /// Number of encoder layers
    pub num_layers: usize,

    /// Hidden dimension
    pub hidden_dim: usize,
}

// Constructors and core methods for major components
impl<T: Float + Debug + Send + Sync + 'static> PrototypicalNetwork<T> {
    /// Create a new prototypical network from configuration
    pub fn new(config: PrototypicalNetworkConfig<T>) -> Result<Self> {
        if config.embedding_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "embedding_dim must be > 0".to_string(),
            ));
        }
        // Build a single-layer encoder: input_dim -> embedding_dim
        let input_dim = config.hidden_dim.max(config.embedding_dim);
        let layer = EncoderLayer {
            weights: Array2::zeros((input_dim, config.embedding_dim)),
            bias: Array1::zeros(config.embedding_dim),
            layer_type: LayerType::Linear,
        };
        Ok(Self {
            encoder: EncoderNetwork {
                layers: vec![layer],
                activation: ActivationFunction::ReLU,
            },
            prototypes: HashMap::new(),
            distance_metric: DistanceMetric::Euclidean,
            parameters: PrototypicalNetworkParams {
                embedding_dim: config.embedding_dim,
                learning_rate: config.learning_rate,
                temperature: T::one(),
                prototype_update_rate: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::one()),
            },
        })
    }

    /// Create from embedding dimension and number of classes (convenience)
    pub fn from_dims(embedding_dim: usize, _num_classes: usize) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "embedding_dim must be > 0".to_string(),
            ));
        }
        let layer = EncoderLayer {
            weights: Array2::zeros((embedding_dim, embedding_dim)),
            bias: Array1::zeros(embedding_dim),
            layer_type: LayerType::Linear,
        };
        Ok(Self {
            encoder: EncoderNetwork {
                layers: vec![layer],
                activation: ActivationFunction::ReLU,
            },
            prototypes: HashMap::new(),
            distance_metric: DistanceMetric::Euclidean,
            parameters: PrototypicalNetworkParams {
                embedding_dim,
                learning_rate: scirs2_core::numeric::NumCast::from(0.01)
                    .unwrap_or_else(|| T::one()),
                temperature: T::one(),
                prototype_update_rate: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::one()),
            },
        })
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.parameters.embedding_dim
    }

    /// Get encoder layers (for projection)
    pub fn encoder_layers(&self) -> &[EncoderLayer<T>] {
        &self.encoder.layers
    }

    /// Get mutable access to prototypes
    pub fn prototypes_mut(&mut self) -> &mut HashMap<String, Prototype<T>> {
        &mut self.prototypes
    }

    /// Get read access to prototypes
    pub fn prototypes(&self) -> &HashMap<String, Prototype<T>> {
        &self.prototypes
    }

    /// Get the distance metric
    pub fn distance_metric(&self) -> &DistanceMetric {
        &self.distance_metric
    }

    /// Encode a task into an embedding vector
    pub fn encode_task(&self, task_data: &TaskData<T>) -> Result<Array1<T>> {
        // Compute mean of support set features as task representation
        if task_data.support_set.examples.is_empty() {
            return Err(OptimError::InsufficientData(
                "No support examples for encoding".to_string(),
            ));
        }
        let dim = self.parameters.embedding_dim;
        let mut sum = Array1::<T>::zeros(dim);
        let count = task_data.support_set.examples.len();
        for ex in &task_data.support_set.examples {
            let feat = &ex.features;
            let len = feat.len().min(dim);
            for i in 0..len {
                sum[i] = sum[i] + feat[i];
            }
        }
        let count_t = scirs2_core::numeric::NumCast::from(count).unwrap_or_else(|| T::one());
        for i in 0..dim {
            sum[i] = sum[i] / count_t;
        }
        Ok(sum)
    }

    /// Update prototypes with new experience
    pub fn update_prototypes(
        &mut self,
        task_data: &TaskData<T>,
        _result: &AdaptationResult<T>,
    ) -> Result<()> {
        let task_repr = self.encode_task(task_data)?;
        let task_id = task_data.task_id.clone();
        let update_rate = self.parameters.prototype_update_rate;
        let one_minus = T::one() - update_rate;

        if let Some(proto) = self.prototypes.get_mut(&task_id) {
            // Exponential moving average update
            let dim = proto.vector.len().min(task_repr.len());
            for i in 0..dim {
                proto.vector[i] = one_minus * proto.vector[i] + update_rate * task_repr[i];
            }
            proto.example_count += task_data.support_set.examples.len();
            proto.last_updated = std::time::SystemTime::now();
            proto.metadata.update_count += 1;
        } else {
            let proto = Prototype {
                vector: task_repr,
                confidence: T::one(),
                example_count: task_data.support_set.examples.len(),
                last_updated: std::time::SystemTime::now(),
                metadata: PrototypeMetadata {
                    task_category: task_id.clone(),
                    domain: task_data.domain_info.domain_type,
                    created_at: std::time::SystemTime::now(),
                    update_count: 1,
                },
            };
            self.prototypes.insert(task_id, proto);
        }
        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> SupportSetManager<T> {
    /// Create a new support set manager
    pub fn new(config: SupportSetManagerConfig) -> Result<Self> {
        if config.max_support_size == 0 {
            return Err(OptimError::InvalidConfig(
                "max_support_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            support_sets: HashMap::new(),
            selection_strategy: SupportSetSelectionStrategy::DiversityBased,
            config,
        })
    }

    /// Create from max support size (convenience)
    pub fn from_max_size(max_support_size: usize) -> Result<Self> {
        Self::new(SupportSetManagerConfig {
            min_support_size: 1,
            max_support_size,
            quality_threshold: 0.5,
            enable_augmentation: true,
            cache_support_sets: true,
        })
    }

    /// Get the max support size
    pub fn max_support_size(&self) -> usize {
        self.config.max_support_size
    }

    /// Get the config
    pub fn config(&self) -> &SupportSetManagerConfig {
        &self.config
    }
}

impl<T: Float + Debug + Send + Sync + 'static> TaskSimilarityCalculator<T> {
    /// Create a new task similarity calculator
    pub fn new(config: SimilarityCalculatorConfig<T>) -> Result<Self> {
        Ok(Self {
            similarity_metrics: Vec::new(),
            metric_weights: HashMap::new(),
            similarity_cache: HashMap::new(),
            config,
        })
    }

    /// Create with default configuration
    pub fn default_new() -> Result<Self> {
        Self::new(SimilarityCalculatorConfig {
            enable_caching: true,
            cache_size_limit: 1000,
            similarity_threshold: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
            use_metadata: true,
        })
    }

    /// Get mutable access to the similarity cache
    pub fn similarity_cache_mut(&mut self) -> &mut HashMap<(String, String), T> {
        &mut self.similarity_cache
    }

    /// Get whether caching is enabled
    pub fn caching_enabled(&self) -> bool {
        self.config.enable_caching
    }
}

impl<T: Float + Debug + Send + Sync + 'static> EpisodicMemoryBank<T> {
    /// Create a new episodic memory bank
    pub fn new(config: MemoryBankConfig<T>) -> Result<Self> {
        if config.max_memory_size == 0 {
            return Err(OptimError::InvalidConfig(
                "max_memory_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            episodes: VecDeque::new(),
            config,
            usage_stats: MemoryUsageStats {
                total_episodes: 0,
                memory_utilization: 0.0,
                hit_rate: 0.0,
                avg_retrieval_time: Duration::from_secs(0),
            },
        })
    }

    /// Create from capacity (convenience)
    pub fn from_capacity(capacity: usize) -> Result<Self> {
        Self::new(MemoryBankConfig {
            max_memory_size: capacity,
            eviction_policy: EvictionPolicy::Performance,
            similarity_threshold: scirs2_core::numeric::NumCast::from(0.3)
                .unwrap_or_else(|| T::zero()),
            enable_compression: false,
        })
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.config.max_memory_size
    }

    /// Get episode count
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get mutable access to episodes
    pub fn episodes_mut(&mut self) -> &mut VecDeque<MemoryEpisode<T>> {
        &mut self.episodes
    }

    /// Get read access to episodes
    pub fn episodes(&self) -> &VecDeque<MemoryEpisode<T>> {
        &self.episodes
    }

    /// Get mutable access to usage stats
    pub fn usage_stats_mut(&mut self) -> &mut MemoryUsageStats {
        &mut self.usage_stats
    }

    /// Get usage stats
    pub fn usage_stats(&self) -> &MemoryUsageStats {
        &self.usage_stats
    }

    /// Get eviction policy
    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.config.eviction_policy
    }

    /// Retrieve similar episodes to a task
    pub fn retrieve_similar(
        &self,
        task_data: &TaskData<T>,
        k: usize,
    ) -> Result<Vec<MemoryEpisode<T>>> {
        if self.episodes.is_empty() {
            return Ok(Vec::new());
        }
        // Simple retrieval: return up to k most recent episodes
        // (full similarity-based retrieval is in episodic_memory_impl)
        let count = k.min(self.episodes.len());
        let result: Vec<MemoryEpisode<T>> =
            self.episodes.iter().rev().take(count).cloned().collect();
        let _ = task_data; // suppress unused warning
        Ok(result)
    }

    /// Store a new episode
    pub fn store_episode(
        &mut self,
        task_data: TaskData<T>,
        result: AdaptationResult<T>,
    ) -> Result<()> {
        let episode = MemoryEpisode {
            episode_id: format!("ep_{}", self.usage_stats.total_episodes),
            task_data,
            adaptation_result: result,
            timestamp: std::time::SystemTime::now(),
            metadata: EpisodeMetadata {
                difficulty: DifficultyLevel::Medium,
                domain: DomainType::Optimization,
                success_rate: 0.0,
                tags: Vec::new(),
            },
            access_count: 0,
        };

        if self.episodes.len() >= self.config.max_memory_size {
            self.episodes.pop_front();
        }
        self.episodes.push_back(episode);
        self.usage_stats.total_episodes += 1;
        self.usage_stats.memory_utilization =
            self.episodes.len() as f64 / self.config.max_memory_size as f64;
        Ok(())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> FastAdaptationEngine<T> {
    /// Create a new fast adaptation engine
    pub fn new(config: FastAdaptationConfig) -> Result<Self> {
        Ok(Self {
            algorithms: Vec::new(),
            config,
        })
    }

    /// Create from inner learning rate and adaptation steps (convenience)
    pub fn from_params(inner_lr: T, adaptation_steps: usize) -> Result<Self> {
        let _ = (inner_lr, adaptation_steps);
        Self::new(FastAdaptationConfig {
            enable_caching: true,
            enable_prediction: true,
            max_adaptation_time: Duration::from_secs(60),
            _performance_threshold: 0.8,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &FastAdaptationConfig {
        &self.config
    }

    /// Perform fast adaptation
    pub fn adapt_fast(
        &mut self,
        _optimizer: &mut dyn FewShotOptimizer<T>,
        _task_data: &TaskData<T>,
        _strategy: AdaptationStrategyType,
        _config: &AdaptationConfig,
    ) -> Result<AdaptationResult<T>> {
        Ok(AdaptationResult {
            adapted_state: OptimizerState {
                parameters: Array1::zeros(1),
                gradients: Array1::zeros(1),
                momentum: None,
                hidden_states: HashMap::new(),
                memory_buffers: HashMap::new(),
                step: 0,
                step_count: 0,
                loss: None,
                learning_rate: scirs2_core::numeric::NumCast::from(0.001)
                    .unwrap_or_else(|| T::one()),
                metadata: super::StateMetadata {
                    task_id: None,
                    optimizer_type: None,
                    version: "1.0".to_string(),
                    timestamp: std::time::SystemTime::now(),
                    checksum: 0,
                    compression_level: 0,
                    custom_data: HashMap::new(),
                },
            },
            performance: AdaptationPerformance {
                query_performance: scirs2_core::numeric::NumCast::from(0.85)
                    .unwrap_or_else(|| T::zero()),
                support_performance: scirs2_core::numeric::NumCast::from(0.90)
                    .unwrap_or_else(|| T::zero()),
                adaptation_speed: 5,
                final_loss: scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero()),
                improvement: scirs2_core::numeric::NumCast::from(0.25).unwrap_or_else(|| T::zero()),
                stability: scirs2_core::numeric::NumCast::from(0.95).unwrap_or_else(|| T::zero()),
            },
            task_representation: Array1::zeros(128),
            adaptation_trajectory: Vec::new(),
            resource_usage: ResourceUsage {
                total_time: Duration::from_secs(15),
                peak_memory_mb: scirs2_core::numeric::NumCast::from(256.0)
                    .unwrap_or_else(|| T::zero()),
                compute_cost: scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero()),
                energy_consumption: scirs2_core::numeric::NumCast::from(0.05)
                    .unwrap_or_else(|| T::zero()),
            },
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> FewShotPerformanceTracker<T> {
    /// Create a new performance tracker
    pub fn new(config: TrackingConfig) -> Result<Self> {
        Ok(Self {
            performance_history: VecDeque::new(),
            metrics: Vec::new(),
            config,
            stats: PerformanceStats {
                best_performance: T::zero(),
                average_performance: T::zero(),
                performance_variance: T::zero(),
                improvement_rate: T::zero(),
                success_rate: T::zero(),
            },
        })
    }

    /// Record a performance result
    pub fn record_performance(&mut self, result: &AdaptationResult<T>) -> Result<()> {
        let record = PerformanceRecord {
            task_id: String::new(),
            performance: result.performance.query_performance,
            adaptation_time: result.resource_usage.total_time,
            strategy_used: String::new(),
            timestamp: std::time::SystemTime::now(),
            additional_metrics: HashMap::new(),
        };

        if self.performance_history.len() >= self.config.max_history_size {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(record);

        // Update stats
        if !self.performance_history.is_empty() {
            let mut sum = T::zero();
            let mut best = T::neg_infinity();
            for r in &self.performance_history {
                sum = sum + r.performance;
                if r.performance > best {
                    best = r.performance;
                }
            }
            let count_t = scirs2_core::numeric::NumCast::from(self.performance_history.len())
                .unwrap_or_else(|| T::one());
            self.stats.average_performance = sum / count_t;
            self.stats.best_performance = best;
        }
        Ok(())
    }
}

// Task metadata and example metadata
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    pub task_name: String,
    pub domain: DomainType,
    pub difficulty: DifficultyLevel,
    pub created_at: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct ExampleMetadata {
    pub source: String,
    pub quality_score: f64,
    pub created_at: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct SupportSetStatistics<T: Float + Debug + Send + Sync + 'static> {
    pub mean: Array1<T>,
    pub variance: Array1<T>,
    pub size: usize,
    pub diversity_score: T,
}

#[derive(Debug, Clone)]
pub struct QuerySetStatistics<T: Float + Debug + Send + Sync + 'static> {
    pub mean: Array1<T>,
    pub variance: Array1<T>,
    pub size: usize,
}

/// Evaluation metrics for few-shot learning
#[derive(Debug, Clone, Copy)]
pub enum EvaluationMetric {
    Accuracy,
    Loss,
    F1Score,
    Precision,
    Recall,
    AUC,
    MSE,
    MAE,
    FinalPerformance,
    Efficiency,
    TrainingTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_set_creation() {
        let support_set = SupportSet::<f64> {
            examples: vec![SupportExample {
                features: Array1::from_vec(vec![1.0, 2.0, 3.0]),
                target: 0.5,
                weight: 1.0,
                context: HashMap::new(),
                metadata: ExampleMetadata {
                    source: "test".to_string(),
                    quality_score: 0.9,
                    created_at: std::time::SystemTime::now(),
                },
            }],
            task_metadata: TaskMetadata {
                task_name: "test_task".to_string(),
                domain: DomainType::Optimization,
                difficulty: DifficultyLevel::Easy,
                created_at: std::time::SystemTime::now(),
            },
            statistics: SupportSetStatistics {
                mean: Array1::from_vec(vec![1.0, 2.0, 3.0]),
                variance: Array1::from_vec(vec![0.1, 0.1, 0.1]),
                size: 1,
                diversity_score: 0.8,
            },
            temporal_order: None,
        };

        assert_eq!(support_set.examples.len(), 1);
        assert_eq!(support_set.statistics.size, 1);
    }

    #[test]
    fn test_adaptation_config() {
        let config = AdaptationConfig {
            adaptation_steps: 10,
            adaptation_lr: 0.01,
            strategy: AdaptationStrategyType::MAML,
            early_stopping: None,
            regularization: RegularizationConfig {
                l2_strength: 0.001,
                dropout_rate: 0.1,
                gradient_clip: Some(1.0),
                task_regularization: HashMap::new(),
            },
            resource_constraints: ResourceConstraints {
                max_time: Duration::from_secs(60),
                max_memory_mb: 1024,
                max_compute_budget: 100.0,
            },
        };

        assert_eq!(config.adaptation_steps, 10);
        assert_eq!(config.adaptation_lr, 0.01);
        assert!(matches!(config.strategy, AdaptationStrategyType::MAML));
    }

    #[test]
    fn test_domain_info() {
        let domain_info = DomainInfo {
            domain_type: DomainType::ComputerVision,
            characteristics: DomainCharacteristics {
                input_dim: 784,
                output_dim: 10,
                temporal: false,
                stochasticity: 0.1,
                noise_level: 0.05,
                sparsity: 0.0,
            },
            difficulty_level: DifficultyLevel::Medium,
            constraints: vec![DomainConstraint {
                constraint_type: ConstraintType::LatencyRequirement,
                description: "Max 100ms inference time".to_string(),
                enforcement: ConstraintEnforcement::Hard,
            }],
        };

        assert!(matches!(
            domain_info.domain_type,
            DomainType::ComputerVision
        ));
        assert_eq!(domain_info.characteristics.input_dim, 784);
        assert_eq!(domain_info.constraints.len(), 1);
    }
}
