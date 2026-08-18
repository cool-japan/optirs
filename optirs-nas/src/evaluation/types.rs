//! Common types for evaluation system
//!
//! Contains shared enums and small structs used across multiple modules.
//!
//! Two families of types were removed from this module rather than left as a
//! public vocabulary for subsystems that do not exist:
//!
//! * `ScheduleType`, `FeatureExtractionMethod`, `NormalizationMethod`,
//!   `ScalingMethod`, `FeatureSelectionMethod` and `UncertaintyEstimationMethod`
//!   described a neural-network performance predictor with a feature-engineering
//!   pipeline that was never built. Their only occurrences were one initialiser
//!   each inside [`super::predictor`], whose value nothing then read; the
//!   predictor is a linear ridge model and now says so.
//!   ([`crate::nas_engine::config::ScheduleType`], which *is* consulted, is a
//!   different type and is unaffected.)
//! * `ProblemType`, `CorrelationStructure`, `DistributionType`, `MetricType`,
//!   `EvaluatorType`, `DataFormat`, `DataCharacteristics`, `SuccessMetrics`,
//!   `TerminationConditions`, `EarlyStoppingCriteria`, `EvaluationCriterion`,
//!   `IOSpecification` and `ValidationCriteria` described caller-supplied custom
//!   benchmarks. [`super::benchmark::BenchmarkSuite`] could neither register nor
//!   run one — see that module's notes.

use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
