//! Common types for evaluation system
//!
//! Contains shared enums and small structs used across multiple modules.

use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

/// Benchmark types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkType {
    /// Convergence speed test
    ConvergenceSpeed,

    /// Final performance test
    FinalPerformance,

    /// Robustness test
    Robustness,

    /// Generalization test
    Generalization,

    /// Efficiency test
    Efficiency,

    /// Scalability test
    Scalability,

    /// Transfer learning test
    TransferLearning,

    /// Multi-task test
    MultiTask,

    /// Noisy optimization test
    NoisyOptimization,

    /// Non-convex optimization test
    NonConvexOptimization,
}

/// Types of test functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestFunctionType {
    /// Quadratic bowl
    Quadratic,

    /// Rosenbrock function
    Rosenbrock,

    /// Rastrigin function
    Rastrigin,

    /// Ackley function
    Ackley,

    /// Sphere function
    Sphere,

    /// Beale function
    Beale,

    /// Neural network training
    NeuralNetworkTraining,

    /// Linear regression
    LinearRegression,

    /// Logistic regression
    LogisticRegression,

    /// Custom function
    Custom(String),
}

/// Difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
    Extreme,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Memory requirement (MB)
    pub memory_mb: usize,

    /// CPU cores required
    pub cpu_cores: usize,

    /// GPU memory (MB, if needed)
    pub gpu_memory_mb: Option<usize>,

    /// Maximum runtime (seconds)
    pub max_runtime_seconds: u64,

    /// Storage requirement (MB)
    pub storage_mb: usize,
}

/// Problem types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemType {
    Regression,
    Classification,
    Clustering,
    DimensionalityReduction,
    ReinforcementLearning,
    GenerativeModeling,
    FeatureSelection,
    HyperparameterOptimization,
}

/// Correlation structures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrelationStructure {
    Independent,
    Linear,
    Nonlinear,
    Hierarchical,
    Spatial,
    Temporal,
}

/// Distribution types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    Gaussian,
    Uniform,
    Exponential,
    PowerLaw,
    Multimodal,
    HeavyTailed,
}

/// Metric types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricType {
    Accuracy,
    Loss,
    F1Score,
    AUC,
    Precision,
    Recall,
    RMSE,
    MAE,
    R2,
    LogLikelihood,
    Perplexity,
    Custom(u32),
}

/// Evaluator types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluatorType {
    MLModel,
    OptimizationFunction,
    Simulator,
    RealWorldAPI,
    Custom,
}

/// Data formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFormat {
    Dense,
    Sparse,
    Sequential,
    Graph,
    Image,
    Text,
    Audio,
    Custom,
}

/// Activation functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    ELU,
    LeakyReLU,
}

/// Schedule types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleType {
    Constant,
    Exponential,
    StepDecay,
    CosineAnnealing,
    ReduceOnPlateau,
    OneCycle,
}

/// Feature extraction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureExtractionMethod {
    ArchitectureEmbedding,
    HyperparameterEncoding,
    ResourceUsageFeatures,
    PerformanceHistory,
    DatasetCharacteristics,
    OptimizationLandscape,
}

/// Normalization methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Robust,
    Quantile,
    PowerTransform,
}

/// Scaling methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingMethod {
    Standard,
    MinMax,
    Robust,
    MaxAbs,
    Quantile,
}

/// Feature selection methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureSelectionMethod {
    VarianceThreshold,
    UnivariateSelection,
    RecursiveFeatureElimination,
    SelectFromModel,
    SequentialFeatureSelection,
    MutualInformation,
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheEvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
}

/// Predictor model types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictorModelType {
    LinearRegression,
    RandomForest,
    GradientBoosting,
    NeuralNetwork,
    GaussianProcess,
    SupportVectorMachine,
    Ensemble,
}

/// Uncertainty estimation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UncertaintyEstimationMethod {
    MonteCarloDropout,
    DeepEnsemble,
    BayesianNeuralNetwork,
    QuantileRegression,
    ConformalPrediction,
    GaussianProcessUncertainty,
}

/// Temporal pattern types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalPatternType {
    Burst,
    Steady,
    Periodic,
    Random,
    Declining,
}

/// Statistical test types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalTestType {
    TTest,
    WilcoxonSignedRank,
    MannWhitneyU,
    KruskalWallis,
    FriedmanTest,
    ChiSquare,
    FisherExact,
    ANOVA,
}

/// Analysis methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisMethod {
    DescriptiveStatistics,
    CorrelationAnalysis,
    RegressionAnalysis,
    ClusterAnalysis,
    FactorAnalysis,
    PrincipalComponentAnalysis,
    SurvivalAnalysis,
}

/// Multiple comparison correction methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultipleComparisonCorrection {
    None,
    Bonferroni,
    HolmBonferroni,
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Sidak,
}

/// Data characteristics
#[derive(Debug, Clone)]
pub struct DataCharacteristics<T: Float + Debug + Send + Sync + 'static> {
    /// Noise level
    pub noise_level: T,

    /// Data sparsity
    pub sparsity: T,

    /// Correlation structure
    pub correlation: CorrelationStructure,

    /// Distribution type
    pub distribution: DistributionType,

    /// Outlier percentage
    pub outlier_percentage: T,
}

/// Success metrics
#[derive(Debug, Clone)]
pub struct SuccessMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum performance threshold
    pub min_performance: T,

    /// Maximum convergence time
    pub max_convergence_time: Duration,

    /// Required stability
    pub stability_threshold: T,

    /// Resource efficiency requirement
    pub efficiency_threshold: T,
}

/// Termination conditions
#[derive(Debug, Clone)]
pub struct TerminationConditions<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum iterations
    pub max_iterations: usize,

    /// Maximum time
    pub max_time: Duration,

    /// Convergence tolerance
    pub convergence_tolerance: T,

    /// Stagnation threshold
    pub stagnation_threshold: usize,

    /// Early stopping criteria
    pub early_stopping: EarlyStoppingCriteria<T>,
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria<T: Float + Debug + Send + Sync + 'static> {
    /// Patience (iterations without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: T,

    /// Validation metric
    pub validation_metric: MetricType,

    /// Relative improvement flag
    pub relative_improvement: bool,
}

/// Evaluation criterion
#[derive(Debug, Clone)]
pub struct EvaluationCriterion<T: Float + Debug + Send + Sync + 'static> {
    /// Criterion name
    pub name: String,

    /// Metric type
    pub metric_type: MetricType,

    /// Target value
    pub target_value: T,

    /// Tolerance
    pub tolerance: T,

    /// Weight in overall score
    pub weight: T,
}

/// Input/output specification
#[derive(Debug, Clone)]
pub struct IOSpecification {
    /// Input format
    pub input_format: DataFormat,

    /// Output format
    pub output_format: DataFormat,

    /// Batch processing support
    pub supports_batching: bool,

    /// Parallelization support
    pub supports_parallel: bool,
}

/// Validation criteria
#[derive(Debug, Clone)]
pub struct ValidationCriteria<T: Float + Debug + Send + Sync + 'static> {
    /// Cross-validation folds
    pub cv_folds: usize,

    /// Validation split ratio
    pub validation_split: T,

    /// Statistical significance level
    pub significance_level: T,

    /// Confidence intervals
    pub confidence_level: T,

    /// Bootstrap samples
    pub bootstrap_samples: usize,
}

/// Performance ranking
#[derive(Debug, Clone)]
pub struct PerformanceRanking {
    /// Overall rank
    pub overall_rank: usize,

    /// Category ranks
    pub category_ranks: HashMap<BenchmarkType, usize>,

    /// Percentile scores
    pub percentile_scores: HashMap<BenchmarkType, f64>,

    /// Relative performance
    pub relative_performance: f64,
}

/// Statistical summary
#[derive(Debug, Clone)]
pub struct StatisticalSummary<T: Float + Debug + Send + Sync + 'static> {
    /// Mean score
    pub mean: T,

    /// Median score
    pub median: T,

    /// Standard deviation
    pub std_dev: T,

    /// Minimum score
    pub min: T,

    /// Maximum score
    pub max: T,

    /// Quartiles
    pub quartiles: (T, T, T),

    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (T, T)>,
}

/// Resource usage summary
#[derive(Debug, Clone)]
pub struct ResourceSummary<T: Float + Debug + Send + Sync + 'static> {
    /// Total memory usage
    pub total_memory_mb: T,

    /// Peak memory usage
    pub peak_memory_mb: T,

    /// Total CPU time
    pub total_cpu_seconds: T,

    /// Total GPU time
    pub total_gpu_seconds: T,

    /// Energy consumption
    pub energy_consumption_kwh: T,

    /// Cost estimate
    pub cost_estimate_usd: T,
}
