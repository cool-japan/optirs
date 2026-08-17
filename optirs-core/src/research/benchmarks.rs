// Academic benchmarking suite for research validation
//
// This module provides standardized benchmarks and evaluation protocols
// for comparing optimization algorithms in academic research contexts.

#[allow(unused_imports)]
use crate::error::Result;
use crate::unified_api::{OptimizerConfig, Parameter, UnifiedAdam, UnifiedOptimizer, UnifiedSGD};
use chrono::{DateTime, Utc};
use scirs2_core::ndarray::{Array1, Ix1, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Academic benchmark suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicBenchmarkSuite {
    /// Suite identifier
    pub id: String,
    /// Suite name
    pub name: String,
    /// Suite description
    pub description: String,
    /// Benchmark problems
    pub benchmarks: Vec<BenchmarkProblem>,
    /// Evaluation metrics
    pub metrics: Vec<EvaluationMetric>,
    /// Reference results
    pub reference_results: HashMap<String, BenchmarkResults>,
    /// Suite metadata
    pub metadata: BenchmarkSuiteMetadata,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Individual benchmark problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProblem {
    /// Problem identifier
    pub id: String,
    /// Problem name
    pub name: String,
    /// Problem description
    pub description: String,
    /// Problem category
    pub category: ProblemCategory,
    /// Problem difficulty
    pub difficulty: DifficultyLevel,
    /// Problem dimensions
    pub dimensions: Vec<usize>,
    /// Objective function
    pub objective_function: ObjectiveFunction,
    /// Problem constraints
    pub constraints: Vec<Constraint>,
    /// Known optimal solution
    pub optimal_solution: Option<OptimalSolution>,
    /// Problem parameters
    pub parameters: HashMap<String, f64>,
    /// Literature references
    pub references: Vec<String>,
}

/// Problem categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProblemCategory {
    /// Convex optimization
    Convex,
    /// Non-convex optimization
    NonConvex,
    /// Machine learning
    MachineLearning,
    /// Deep learning
    DeepLearning,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Computer vision
    ComputerVision,
    /// Natural language processing
    NaturalLanguageProcessing,
    /// Numerical optimization
    NumericalOptimization,
    /// Constrained optimization
    ConstrainedOptimization,
    /// Multi-objective optimization
    MultiObjective,
    /// Stochastic optimization
    Stochastic,
    /// Discrete optimization
    Discrete,
    /// Continuous optimization
    Continuous,
    /// Mixed optimization
    Mixed,
}

/// Difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DifficultyLevel {
    /// Easy problems
    Easy,
    /// Medium problems
    Medium,
    /// Hard problems
    Hard,
    /// Very hard problems
    VeryHard,
    /// Extreme problems
    Extreme,
}

/// Objective function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveFunction {
    /// Function name
    pub name: String,
    /// Function type
    pub function_type: FunctionType,
    /// Function properties
    pub properties: FunctionProperties,
    /// Mathematical description
    pub mathematical_form: String,
    /// Implementation notes
    pub implementation_notes: String,
}

/// Function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FunctionType {
    /// Quadratic function
    Quadratic,
    /// Rosenbrock function
    Rosenbrock,
    /// Sphere function
    Sphere,
    /// Rastrigin function
    Rastrigin,
    /// Ackley function
    Ackley,
    /// Griewank function
    Griewank,
    /// Schwefel function
    Schwefel,
    /// Himmelblau function
    Himmelblau,
    /// Booth function
    Booth,
    /// Beale function
    Beale,
    /// Three-hump camel function
    ThreeHumpCamel,
    /// Six-hump camel function
    SixHumpCamel,
    /// Cross-in-tray function
    CrossInTray,
    /// Egg holder function
    EggHolder,
    /// Holder table function
    HolderTable,
    /// McCormick function
    McCormick,
    /// Schaffer function N2
    SchafferN2,
    /// Schaffer function N4
    SchafferN4,
    /// StyblinskiTang function
    StyblinskiTang,
    /// Custom function
    Custom(String),
}

/// Function properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionProperties {
    /// Is the function differentiable
    pub differentiable: bool,
    /// Is the function continuous
    pub continuous: bool,
    /// Is the function convex
    pub convex: bool,
    /// Is the function separable
    pub separable: bool,
    /// Is the function multimodal
    pub multimodal: bool,
    /// Function smoothness
    pub smoothness: SmoothnesLevel,
    /// Condition number
    pub condition_number: Option<f64>,
    /// Lipschitz constant
    pub lipschitz_constant: Option<f64>,
}

/// Smoothness levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SmoothnesLevel {
    /// Very smooth
    VerySmooth,
    /// Smooth
    Smooth,
    /// Moderately smooth
    ModeratelySmooth,
    /// Rough
    Rough,
    /// Very rough
    VeryRough,
}

/// Optimization constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint description
    pub description: String,
    /// Mathematical form
    pub mathematical_form: String,
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConstraintType {
    /// Equality constraint
    Equality,
    /// Inequality constraint
    Inequality,
    /// Box constraint (bounds)
    Box,
    /// Linear constraint
    Linear,
    /// Nonlinear constraint
    Nonlinear,
    /// Integer constraint
    Integer,
    /// Binary constraint
    Binary,
}

/// Known optimal solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalSolution {
    /// Optimal parameter values
    pub parameters: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64,
    /// Solution properties
    pub properties: SolutionProperties,
    /// Literature reference
    pub reference: Option<String>,
}

/// Solution properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionProperties {
    /// Is this a global optimum
    pub global_optimum: bool,
    /// Is this a local optimum
    pub local_optimum: bool,
    /// Solution uniqueness
    pub unique: bool,
    /// Solution stability
    pub stable: bool,
}

/// Evaluation metric for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetric {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Aggregation method
    pub aggregation: AggregationMethod,
    /// Better direction (higher or lower is better)
    pub better_direction: BetterDirection,
    /// Metric weight in overall score
    pub weight: f64,
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetricType {
    /// Objective value at convergence
    FinalObjective,
    /// Number of iterations to convergence
    IterationsToConvergence,
    /// Time to convergence
    TimeToConvergence,
    /// Function evaluations to convergence
    FunctionEvaluations,
    /// Gradient evaluations
    GradientEvaluations,
    /// Success rate (percentage of successful runs)
    SuccessRate,
    /// Solution quality
    SolutionQuality,
    /// Convergence rate
    ConvergenceRate,
    /// Robustness measure
    Robustness,
    /// Memory usage
    MemoryUsage,
    /// Energy consumption
    EnergyConsumption,
    /// Custom metric
    Custom(String),
}

/// Aggregation methods for multiple runs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Mean value
    Mean,
    /// Median value
    Median,
    /// Best value
    Best,
    /// Worst value
    Worst,
    /// Standard deviation
    StandardDeviation,
    /// Percentile (specify which percentile)
    Percentile(u8),
    /// Success count
    SuccessCount,
    /// Custom aggregation
    Custom(String),
}

/// Better direction for metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BetterDirection {
    /// Higher values are better
    Higher,
    /// Lower values are better
    Lower,
}

/// Benchmark results for a specific optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Optimizer name
    pub optimizer_name: String,
    /// Results per problem
    pub problem_results: HashMap<String, ProblemResults>,
    /// Overall scores
    pub overall_scores: HashMap<String, f64>,
    /// Statistical significance tests
    pub statistical_tests: Vec<StatisticalTest>,
    /// Performance ranking
    pub ranking: OptimizerRanking,
    /// Execution timestamp
    pub executed_at: DateTime<Utc>,
}

/// Results for a single problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemResults {
    /// Problem identifier
    pub problem_id: String,
    /// Individual run results
    pub run_results: Vec<RunResult>,
    /// Aggregated metrics
    pub aggregated_metrics: HashMap<String, f64>,
    /// Statistical summaries
    pub statistics: ResultStatistics,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
}

/// Result for a single run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Run identifier
    pub run_id: String,
    /// Random seed used
    pub random_seed: u64,
    /// Final objective value
    pub final_objective: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Execution time (seconds)
    pub execution_time: f64,
    /// Function evaluations
    pub function_evaluations: usize,
    /// Gradient evaluations
    pub gradient_evaluations: usize,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Convergence trajectory
    pub trajectory: Vec<f64>,
    /// Error information (if failed)
    pub error_info: Option<String>,
}

/// Statistical summaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultStatistics {
    /// Number of successful runs
    pub successful_runs: usize,
    /// Total number of runs
    pub total_runs: usize,
    /// Success rate
    pub success_rate: f64,
    /// Mean objective value
    pub mean_objective: f64,
    /// Standard deviation of objective values
    pub std_objective: f64,
    /// Best objective value
    pub best_objective: f64,
    /// Worst objective value
    pub worst_objective: f64,
    /// Median objective value
    pub median_objective: f64,
    /// Quartiles
    pub quartiles: (f64, f64, f64), // Q1, Q2, Q3
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Convergence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    /// Average convergence rate
    pub avg_convergence_rate: f64,
    /// Convergence stability
    pub convergence_stability: f64,
    /// Early convergence indicator
    pub early_convergence: bool,
    /// Plateau detection
    pub plateau_detected: bool,
    /// Convergence pattern
    pub convergence_pattern: ConvergencePattern,
}

/// Convergence patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConvergencePattern {
    /// Monotonic decrease
    MonotonicDecrease,
    /// Exponential decay
    ExponentialDecay,
    /// Linear decrease
    LinearDecrease,
    /// Oscillatory convergence
    Oscillatory,
    /// Stepwise convergence
    Stepwise,
    /// Plateau then drop
    PlateauThenDrop,
    /// No clear pattern
    Irregular,
}

/// Statistical significance test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name
    pub test_name: String,
    /// Compared optimizers
    pub optimizers: Vec<String>,
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance_level: f64,
    /// Test result
    pub significant: bool,
    /// Effect size
    pub effect_size: Option<f64>,
}

/// Optimizer ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerRanking {
    /// Overall rank (1 is best)
    pub overall_rank: usize,
    /// Ranks per category
    pub category_ranks: HashMap<String, usize>,
    /// Ranks per metric
    pub metric_ranks: HashMap<String, usize>,
    /// Ranking score
    pub ranking_score: f64,
    /// Ranking method used
    pub ranking_method: RankingMethod,
}

/// Ranking methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RankingMethod {
    /// Average rank across all metrics
    AverageRank,
    /// Weighted score
    WeightedScore,
    /// Pareto dominance
    ParetoDominance,
    /// Win-loss-tie
    WinLossTie,
    /// Tournament ranking
    Tournament,
    /// Custom ranking method
    Custom(String),
}

/// Benchmark suite metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteMetadata {
    /// Suite version
    pub version: String,
    /// Suite authors
    pub authors: Vec<String>,
    /// Suite license
    pub license: String,
    /// Literature references
    pub references: Vec<String>,
    /// Target audience
    pub target_audience: Vec<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Changelog
    pub changelog: Vec<ChangelogEntry>,
}

/// Changelog entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    /// Version number
    pub version: String,
    /// Release date
    pub date: DateTime<Utc>,
    /// Changes description
    pub changes: String,
    /// Author of changes
    pub author: String,
}

/// Benchmark runner for executing benchmark suites
pub struct BenchmarkRunner {
    /// Benchmark suite
    suite: AcademicBenchmarkSuite,
    /// Execution settings
    settings: BenchmarkSettings,
    /// Progress callback
    progress_callback: Option<Box<dyn Fn(f64) + Send + Sync>>,
}

impl std::fmt::Debug for BenchmarkRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkRunner")
            .field("suite", &self.suite)
            .field("settings", &self.settings)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Benchmark execution settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSettings {
    /// Number of independent runs per problem
    pub num_runs: usize,
    /// Random seeds to use
    pub random_seeds: Vec<u64>,
    /// Maximum iterations per run
    pub max_iterations: usize,
    /// Maximum execution time per run (seconds)
    pub max_time_seconds: f64,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Number of parallel threads
    pub num_threads: Option<usize>,
    /// Save detailed results
    pub save_detailed_results: bool,
    /// Output directory
    pub output_directory: Option<String>,
}

impl AcademicBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: String::new(),
            benchmarks: Vec::new(),
            metrics: Vec::new(),
            reference_results: HashMap::new(),
            metadata: BenchmarkSuiteMetadata::default(),
            created_at: Utc::now(),
        }
    }

    /// Add a benchmark problem
    pub fn add_benchmark(&mut self, benchmark: BenchmarkProblem) {
        self.benchmarks.push(benchmark);
    }

    /// Add an evaluation metric
    pub fn add_metric(&mut self, metric: EvaluationMetric) {
        self.metrics.push(metric);
    }

    /// Create standard ML optimization benchmark suite
    pub fn standard_ml_suite() -> Self {
        let mut suite = Self::new("Standard ML Optimization Benchmark");
        suite.description =
            "Standard benchmark suite for machine learning optimization algorithms".to_string();

        // Add standard problems
        suite.add_benchmark(Self::create_quadratic_problem());
        suite.add_benchmark(Self::create_rosenbrock_problem());
        suite.add_benchmark(Self::create_logistic_regression_problem());
        suite.add_benchmark(Self::create_neural_network_problem());

        // Add standard metrics
        suite.add_metric(Self::create_final_objective_metric());
        suite.add_metric(Self::create_convergence_time_metric());
        suite.add_metric(Self::create_success_rate_metric());

        suite
    }

    fn create_quadratic_problem() -> BenchmarkProblem {
        BenchmarkProblem {
            id: "quadratic_10d".to_string(),
            name: "10D Quadratic Function".to_string(),
            description: "Simple quadratic function in 10 dimensions".to_string(),
            category: ProblemCategory::Convex,
            difficulty: DifficultyLevel::Easy,
            dimensions: vec![10],
            objective_function: ObjectiveFunction {
                name: "Quadratic".to_string(),
                function_type: FunctionType::Quadratic,
                properties: FunctionProperties {
                    differentiable: true,
                    continuous: true,
                    convex: true,
                    separable: true,
                    multimodal: false,
                    smoothness: SmoothnesLevel::VerySmooth,
                    condition_number: Some(1.0),
                    lipschitz_constant: Some(2.0),
                },
                mathematical_form: "f(x) = 0.5 * x^T * x".to_string(),
                implementation_notes: "Simple quadratic function with unit matrix".to_string(),
            },
            constraints: Vec::new(),
            optimal_solution: Some(OptimalSolution {
                parameters: Array1::zeros(10),
                objective_value: 0.0,
                properties: SolutionProperties {
                    global_optimum: true,
                    local_optimum: true,
                    unique: true,
                    stable: true,
                },
                reference: None,
            }),
            parameters: HashMap::new(),
            references: vec!["Standard optimization textbooks".to_string()],
        }
    }

    fn create_rosenbrock_problem() -> BenchmarkProblem {
        BenchmarkProblem {
            id: "rosenbrock_10d".to_string(),
            name: "10D Rosenbrock Function".to_string(),
            description: "Rosenbrock function in 10 dimensions".to_string(),
            category: ProblemCategory::NonConvex,
            difficulty: DifficultyLevel::Medium,
            dimensions: vec![10],
            objective_function: ObjectiveFunction {
                name: "Rosenbrock".to_string(),
                function_type: FunctionType::Rosenbrock,
                properties: FunctionProperties {
                    differentiable: true,
                    continuous: true,
                    convex: false,
                    separable: false,
                    multimodal: false,
                    smoothness: SmoothnesLevel::Smooth,
                    condition_number: None,
                    lipschitz_constant: None},
                mathematical_form: "f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)".to_string(),
                implementation_notes: "Classic Rosenbrock function, challenging for optimization".to_string()},
            constraints: Vec::new(),
            optimal_solution: Some(OptimalSolution {
                parameters: Array1::ones(10),
                objective_value: 0.0,
                properties: SolutionProperties {
                    global_optimum: true,
                    local_optimum: true,
                    unique: true,
                    stable: true},
                reference: Some("Rosenbrock, H.H. (1960)".to_string())}),
            parameters: HashMap::new(),
            references: vec!["Rosenbrock, H.H. (1960). An automatic method for finding the greatest or least value of a function.".to_string()]}
    }

    fn create_logistic_regression_problem() -> BenchmarkProblem {
        BenchmarkProblem {
            id: "logistic_regression_100d".to_string(),
            name: "Logistic Regression (100D)".to_string(),
            description: "Logistic regression on synthetic dataset".to_string(),
            category: ProblemCategory::MachineLearning,
            difficulty: DifficultyLevel::Medium,
            dimensions: vec![100],
            objective_function: ObjectiveFunction {
                name: "Logistic Loss".to_string(),
                function_type: FunctionType::Custom("LogisticLoss".to_string()),
                properties: FunctionProperties {
                    differentiable: true,
                    continuous: true,
                    convex: true,
                    separable: false,
                    multimodal: false,
                    smoothness: SmoothnesLevel::Smooth,
                    condition_number: None,
                    lipschitz_constant: None,
                },
                mathematical_form: "f(w) = mean(log(1 + exp(-y * X * w))) + lambda * ||w||^2"
                    .to_string(),
                implementation_notes: "Binary classification with L2 regularization".to_string(),
            },
            constraints: Vec::new(),
            optimal_solution: None, // Depends on dataset
            parameters: {
                let mut params = HashMap::new();
                params.insert("lambda".to_string(), 0.01);
                params.insert("num_samples".to_string(), 1000.0);
                params
            },
            references: vec!["Standard machine learning references".to_string()],
        }
    }

    fn create_neural_network_problem() -> BenchmarkProblem {
        BenchmarkProblem {
            id: "neural_network_mnist".to_string(),
            name: "Neural Network MNIST".to_string(),
            description: "Two-layer neural network on MNIST subset".to_string(),
            category: ProblemCategory::DeepLearning,
            difficulty: DifficultyLevel::Hard,
            dimensions: vec![784, 128, 10], // Input, hidden, output
            objective_function: ObjectiveFunction {
                name: "Cross-entropy Loss".to_string(),
                function_type: FunctionType::Custom("CrossEntropyLoss".to_string()),
                properties: FunctionProperties {
                    differentiable: true,
                    continuous: true,
                    convex: false,
                    separable: false,
                    multimodal: true,
                    smoothness: SmoothnesLevel::Smooth,
                    condition_number: None,
                    lipschitz_constant: None,
                },
                mathematical_form: "f(θ) = mean(-log(softmax(NN(x; θ))[y]))".to_string(),
                implementation_notes: "Two-layer ReLU network with softmax output".to_string(),
            },
            constraints: Vec::new(),
            optimal_solution: None, // Unknown for neural networks
            parameters: {
                let mut params = HashMap::new();
                params.insert("num_samples".to_string(), 10000.0);
                params.insert("batch_size".to_string(), 64.0);
                params
            },
            references: vec![
                "LeCun et al. (1998). Gradient-based learning applied to document recognition."
                    .to_string(),
            ],
        }
    }

    fn create_final_objective_metric() -> EvaluationMetric {
        EvaluationMetric {
            name: "Final Objective Value".to_string(),
            description: "Final objective function value achieved".to_string(),
            metric_type: MetricType::FinalObjective,
            aggregation: AggregationMethod::Mean,
            better_direction: BetterDirection::Lower,
            weight: 1.0,
        }
    }

    fn create_convergence_time_metric() -> EvaluationMetric {
        EvaluationMetric {
            name: "Time to Convergence".to_string(),
            description: "Time required to reach convergence tolerance".to_string(),
            metric_type: MetricType::TimeToConvergence,
            aggregation: AggregationMethod::Median,
            better_direction: BetterDirection::Lower,
            weight: 0.5,
        }
    }

    fn create_success_rate_metric() -> EvaluationMetric {
        EvaluationMetric {
            name: "Success Rate".to_string(),
            description: "Percentage of runs that converged successfully".to_string(),
            metric_type: MetricType::SuccessRate,
            aggregation: AggregationMethod::Mean,
            better_direction: BetterDirection::Higher,
            weight: 0.8,
        }
    }
}

/// One real, runnable optimizer selected by name for a benchmark run.
///
/// [`UnifiedOptimizer::step_param`] is generic over the parameter's
/// dimension type, which makes the trait itself object-unsafe (`dyn
/// UnifiedOptimizer<A>` cannot exist); this small enum is the standard
/// workaround, letting [`BenchmarkRunner`] pick an algorithm by name at
/// runtime while still driving each one through its real implementation.
enum ChosenOptimizer<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync> {
    Sgd(UnifiedSGD<A>),
    Adam(UnifiedAdam<A>),
}

impl<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync> ChosenOptimizer<A> {
    fn step_param(&mut self, param: &mut Parameter<A, Ix1>) -> Result<()> {
        match self {
            ChosenOptimizer::Sgd(optimizer) => optimizer.step_param(param),
            ChosenOptimizer::Adam(optimizer) => optimizer.step_param(param),
        }
    }
}

/// Select a concrete optimizer implementation by (case-insensitive)
/// `optimizer_name`, defaulting to plain SGD for anything not recognized as
/// Adam -- this only chooses *which* real optimizer, `optimizer_config`'s
/// learning rate/weight decay/etc. are honored either way.
fn select_optimizer<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync>(
    optimizer_name: &str,
    config: OptimizerConfig<A>,
) -> ChosenOptimizer<A> {
    if optimizer_name.to_lowercase().contains("adam") {
        ChosenOptimizer::Adam(UnifiedAdam::new(config))
    } else {
        ChosenOptimizer::Sgd(UnifiedSGD::new(config))
    }
}

/// Evaluate a benchmark objective's value and analytic gradient at `x`.
///
/// [`FunctionType::Rosenbrock`] and [`FunctionType::Sphere`] get their real,
/// classic closed-form definitions. Every other function type (including
/// [`FunctionType::Custom`]) falls back to the convex quadratic bowl `f(x) =
/// 0.5||x||^2` -- an honest, clearly-documented stand-in for "no dedicated
/// implementation yet" rather than fabricated random noise presented as an
/// optimization result.
fn evaluate_objective<A: Float>(function_type: &FunctionType, x: &[A]) -> (A, Vec<A>) {
    match function_type {
        FunctionType::Rosenbrock => {
            let mut value = A::zero();
            let mut grad = vec![A::zero(); x.len()];
            let hundred = A::from(100.0).unwrap_or_else(A::one);
            let two = A::from(2.0).unwrap_or_else(A::one);
            let four = A::from(4.0).unwrap_or_else(A::one);

            for i in 0..x.len().saturating_sub(1) {
                let xi = x[i];
                let xi1 = x[i + 1];
                let t1 = xi1 - xi * xi;
                let t2 = A::one() - xi;
                value = value + hundred * t1 * t1 + t2 * t2;
                grad[i] = grad[i] + (-four * hundred * xi * t1) - two * t2;
                grad[i + 1] = grad[i + 1] + two * hundred * t1;
            }

            (value, grad)
        }
        FunctionType::Sphere => {
            let two = A::from(2.0).unwrap_or_else(A::one);
            let value = x.iter().fold(A::zero(), |acc, &xi| acc + xi * xi);
            let grad = x.iter().map(|&xi| two * xi).collect();
            (value, grad)
        }
        _ => {
            let half = A::from(0.5).unwrap_or_else(A::one);
            let value = x.iter().fold(A::zero(), |acc, &xi| acc + xi * xi) * half;
            let grad = x.to_vec();
            (value, grad)
        }
    }
}

/// 95% two-sided confidence interval for a sample mean, using the
/// Student-t distribution (appropriate for the typically small sample
/// sizes -- a handful of independent benchmark runs -- these statistics are
/// computed over) rather than a large-sample normal approximation. Returns
/// `(mean, mean)` when there is no defined interval (fewer than 2 samples,
/// or zero variance).
fn confidence_interval_95(n: usize, mean: f64, sample_std: f64) -> (f64, f64) {
    if n < 2 || sample_std <= 0.0 {
        return (mean, mean);
    }

    let df = (n - 1) as f64;
    let t_critical = student_t_critical_value(df, 0.975);
    let margin = t_critical * sample_std / (n as f64).sqrt();
    (mean - margin, mean + margin)
}

/// The critical value `t` such that `P(T <= t) = quantile` for a Student-t
/// distribution with `df` degrees of freedom, found by bisecting the real
/// CDF from `scirs2_stats` (the same technique
/// `ContinuousDistribution::ppf`'s default implementation uses; `StudentT`
/// does not implement that trait, so this reimplements just the bisection).
fn student_t_critical_value(df: f64, quantile: f64) -> f64 {
    // z_0.975, used as a fallback if the distribution can't be built.
    const Z_975: f64 = 1.959963985;

    let Ok(dist) = scirs2_stats::distributions::t(df, 0.0_f64, 1.0_f64) else {
        return Z_975;
    };

    // Low-df Student-t is heavy-tailed (df=1 is the Cauchy distribution,
    // whose 97.5th percentile is ~12.7), so a fixed bracket is not always
    // wide enough: double the upper bound until it truly brackets the
    // target quantile before bisecting.
    let mut low = 0.0_f64;
    let mut high = 2.0_f64;
    while dist.cdf(high) < quantile && high < 1e12 {
        high *= 2.0;
    }

    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if dist.cdf(mid) < quantile {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Wilson score 95% confidence interval for a binomial proportion
/// (`successes` out of `trials`) -- a standard closed-form interval that,
/// unlike a normal approximation, stays within `[0, 1]` and remains
/// well-behaved for the small sample counts and extreme (near 0 or 1)
/// proportions typical of an optimizer's success rate.
fn wilson_score_interval_95(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 0.0);
    }

    const Z: f64 = 1.959963985; // z_0.975
    let n = trials as f64;
    let p_hat = successes as f64 / n;
    let z_sq = Z * Z;

    let denominator = 1.0 + z_sq / n;
    let center = (p_hat + z_sq / (2.0 * n)) / denominator;
    let margin = (Z * ((p_hat * (1.0 - p_hat) / n) + z_sq / (4.0 * n * n)).sqrt()) / denominator;

    ((center - margin).max(0.0), (center + margin).min(1.0))
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(suite: AcademicBenchmarkSuite, settings: BenchmarkSettings) -> Self {
        Self {
            suite,
            settings,
            progress_callback: None,
        }
    }

    /// Set progress callback
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(f64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// Run benchmark suite on multiple optimizers
    pub fn run_benchmarks<
        A: Float + std::fmt::Debug + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
    >(
        &self,
        optimizers: &[(&str, OptimizerConfig<A>)],
    ) -> Result<HashMap<String, BenchmarkResults>> {
        let mut all_results = HashMap::new();

        let total_work = optimizers.len() * self.suite.benchmarks.len() * self.settings.num_runs;
        let mut completed_work = 0;

        for (optimizer_name, optimizer_config) in optimizers {
            let mut optimizer_results = BenchmarkResults {
                optimizer_name: optimizer_name.to_string(),
                problem_results: HashMap::new(),
                overall_scores: HashMap::new(),
                statistical_tests: Vec::new(),
                ranking: OptimizerRanking {
                    overall_rank: 0,
                    category_ranks: HashMap::new(),
                    metric_ranks: HashMap::new(),
                    ranking_score: 0.0,
                    ranking_method: RankingMethod::WeightedScore,
                },
                executed_at: Utc::now(),
            };

            for benchmark in &self.suite.benchmarks {
                let problem_results =
                    self.run_single_problem::<A>(benchmark, optimizer_name, optimizer_config)?;
                optimizer_results
                    .problem_results
                    .insert(benchmark.id.clone(), problem_results);

                completed_work += self.settings.num_runs;
                if let Some(ref callback) = self.progress_callback {
                    callback(completed_work as f64 / total_work as f64);
                }
            }

            // Calculate overall scores
            self.calculate_overall_scores(&mut optimizer_results);

            all_results.insert(optimizer_name.to_string(), optimizer_results);
        }

        // Calculate rankings and statistical tests
        self.calculate_rankings_and_tests(&mut all_results);

        Ok(all_results)
    }

    fn run_single_problem<
        A: Float + std::fmt::Debug + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
    >(
        &self,
        benchmark: &BenchmarkProblem,
        optimizer_name: &str,
        optimizer_config: &OptimizerConfig<A>,
    ) -> Result<ProblemResults> {
        let mut run_results = Vec::new();

        for run_idx in 0..self.settings.num_runs {
            let seed = if run_idx < self.settings.random_seeds.len() {
                self.settings.random_seeds[run_idx]
            } else {
                42 + run_idx as u64
            };

            let run_result =
                self.run_single_instance::<A>(benchmark, optimizer_name, optimizer_config, seed)?;
            run_results.push(run_result);
        }

        // Calculate aggregated metrics and statistics
        let aggregated_metrics = self.calculate_aggregated_metrics(&run_results);
        let statistics = self.calculate_statistics(&run_results);
        let convergence_analysis = self.analyze_convergence(&run_results);

        Ok(ProblemResults {
            problem_id: benchmark.id.clone(),
            run_results,
            aggregated_metrics,
            statistics,
            convergence_analysis,
        })
    }

    /// Run `optimizer_name`/`optimizer_config` against `benchmark`'s real
    /// objective function for real: this seeds a deterministic starting
    /// point from `seed`, then repeatedly evaluates the objective's analytic
    /// gradient at the current point and applies one real optimizer step
    /// (via [`UnifiedSGD`]/[`UnifiedAdam`], selected by `optimizer_name`),
    /// recording the true trajectory of objective values.
    ///
    /// Previously this ignored both `optimizer_config` and `seed` entirely
    /// and returned a value drawn from a fixed, function-type-specific `Rng`
    /// range -- so every optimizer "converged" identically regardless of its
    /// hyperparameters, and repeated runs with different seeds were
    /// indistinguishable.
    fn run_single_instance<
        A: Float + std::fmt::Debug + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
    >(
        &self,
        benchmark: &BenchmarkProblem,
        optimizer_name: &str,
        optimizer_config: &OptimizerConfig<A>,
        seed: u64,
    ) -> Result<RunResult> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        let dim = benchmark.dimensions.first().copied().unwrap_or(10).max(1);
        let iterations = std::cmp::min(1000, self.settings.max_iterations).max(1);

        let mut rng = Random::seed(seed);
        let initial: Vec<A> = (0..dim)
            .map(|_| A::from(rng.gen_range(-2.0_f64..2.0)).unwrap_or_else(A::zero))
            .collect();

        let mut param = Parameter::new(Array1::from_vec(initial), "x".to_string());
        let mut optimizer = select_optimizer(optimizer_name, optimizer_config.clone());

        let mut trajectory = Vec::with_capacity(iterations + 1);

        for _ in 0..iterations {
            let x: Vec<A> = param.data.iter().copied().collect();
            let (value, grad) = evaluate_objective(&benchmark.objective_function.function_type, &x);
            trajectory.push(value.to_f64().unwrap_or(f64::NAN));

            param.set_grad(Array1::from_vec(grad));
            optimizer.step_param(&mut param)?;
        }

        // Score the point the optimizer actually finished on.
        let x: Vec<A> = param.data.iter().copied().collect();
        let (final_objective_a, _) =
            evaluate_objective(&benchmark.objective_function.function_type, &x);
        trajectory.push(final_objective_a.to_f64().unwrap_or(f64::NAN));

        let final_objective = final_objective_a.to_f64().unwrap_or(f64::INFINITY);
        let execution_time = start_time.elapsed().as_secs_f64();
        let converged =
            final_objective.is_finite() && final_objective < self.settings.convergence_tolerance;

        Ok(RunResult {
            run_id,
            random_seed: seed,
            final_objective,
            converged,
            iterations,
            execution_time,
            function_evaluations: iterations + 1,
            gradient_evaluations: iterations,
            memory_usage: dim * std::mem::size_of::<f64>() * 4,
            trajectory,
            error_info: None,
        })
    }

    fn calculate_aggregated_metrics(&self, run_results: &[RunResult]) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if !run_results.is_empty() {
            // Final objective metrics
            let final_objectives: Vec<f64> =
                run_results.iter().map(|r| r.final_objective).collect();
            metrics.insert(
                "mean_final_objective".to_string(),
                final_objectives.iter().sum::<f64>() / final_objectives.len() as f64,
            );

            let mut sorted_objectives = final_objectives.clone();
            sorted_objectives.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            metrics.insert(
                "median_final_objective".to_string(),
                sorted_objectives[sorted_objectives.len() / 2],
            );
            metrics.insert("best_final_objective".to_string(), sorted_objectives[0]);

            // Time metrics
            let execution_times: Vec<f64> = run_results.iter().map(|r| r.execution_time).collect();
            metrics.insert(
                "mean_execution_time".to_string(),
                execution_times.iter().sum::<f64>() / execution_times.len() as f64,
            );

            // Success rate
            let successful_runs = run_results.iter().filter(|r| r.converged).count();
            metrics.insert(
                "success_rate".to_string(),
                successful_runs as f64 / run_results.len() as f64,
            );
        }

        metrics
    }

    fn calculate_statistics(&self, run_results: &[RunResult]) -> ResultStatistics {
        if run_results.is_empty() {
            return ResultStatistics {
                successful_runs: 0,
                total_runs: 0,
                success_rate: 0.0,
                mean_objective: 0.0,
                std_objective: 0.0,
                best_objective: 0.0,
                worst_objective: 0.0,
                median_objective: 0.0,
                quartiles: (0.0, 0.0, 0.0),
                confidence_intervals: HashMap::new(),
            };
        }

        let successful_runs = run_results.iter().filter(|r| r.converged).count();
        let total_runs = run_results.len();
        let success_rate = successful_runs as f64 / total_runs as f64;

        let objectives: Vec<f64> = run_results.iter().map(|r| r.final_objective).collect();
        let mean_objective = objectives.iter().sum::<f64>() / objectives.len() as f64;

        // Sample variance (Bessel's correction, divide by n-1): these
        // `objectives` are a *sample* of independent runs used to infer the
        // variability of the underlying optimizer/problem, which is exactly
        // the setting the n-1 correction is for. n=1 has no defined sample
        // variance (would divide by zero), so it is reported as 0.
        let variance = if objectives.len() > 1 {
            objectives
                .iter()
                .map(|&x| (x - mean_objective).powi(2))
                .sum::<f64>()
                / (objectives.len() - 1) as f64
        } else {
            0.0
        };
        let std_objective = variance.sqrt();

        let mut sorted_objectives = objectives.clone();
        sorted_objectives.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let best_objective = sorted_objectives[0];
        let worst_objective = sorted_objectives[sorted_objectives.len() - 1];
        let median_objective = sorted_objectives[sorted_objectives.len() / 2];

        let q1_idx = sorted_objectives.len() / 4;
        let q3_idx = 3 * sorted_objectives.len() / 4;
        let quartiles = (
            sorted_objectives[q1_idx],
            median_objective,
            sorted_objectives[q3_idx],
        );

        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert(
            "mean_objective_95".to_string(),
            confidence_interval_95(objectives.len(), mean_objective, std_objective),
        );
        confidence_intervals.insert(
            "success_rate_95".to_string(),
            wilson_score_interval_95(successful_runs, total_runs),
        );

        ResultStatistics {
            successful_runs,
            total_runs,
            success_rate,
            mean_objective,
            std_objective,
            best_objective,
            worst_objective,
            median_objective,
            quartiles,
            confidence_intervals,
        }
    }

    fn analyze_convergence(&self, run_results: &[RunResult]) -> ConvergenceAnalysis {
        if run_results.is_empty() {
            return ConvergenceAnalysis {
                avg_convergence_rate: 0.0,
                convergence_stability: 0.0,
                early_convergence: false,
                plateau_detected: false,
                convergence_pattern: ConvergencePattern::Irregular,
            };
        }

        // Simplified convergence analysis
        let avg_convergence_rate = run_results
            .iter()
            .filter(|r| r.converged)
            .map(|r| r.iterations as f64)
            .sum::<f64>()
            / run_results.len() as f64;

        let convergence_stability = 0.8; // Placeholder
        let early_convergence = avg_convergence_rate < self.settings.max_iterations as f64 * 0.5;
        let plateau_detected = false; // Would analyze trajectories
        let convergence_pattern = ConvergencePattern::MonotonicDecrease; // Simplified

        ConvergenceAnalysis {
            avg_convergence_rate,
            convergence_stability,
            early_convergence,
            plateau_detected,
            convergence_pattern,
        }
    }

    fn calculate_overall_scores(&self, results: &mut BenchmarkResults) {
        // Calculate weighted scores across all problems and metrics
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for metric in &self.suite.metrics {
            // `EvaluationMetric::name` is a free-form display string (e.g.
            // "Final Objective Value") that never matches the fixed keys
            // `calculate_aggregated_metrics` actually inserts (e.g.
            // "mean_final_objective") -- go through the metric *type*
            // instead, which is the field `calculate_aggregated_metrics`'s
            // keys were really chosen to represent.
            let Some(key) = aggregated_metric_key(&metric.metric_type) else {
                continue;
            };

            let mut metric_score = 0.0;
            let mut metric_count = 0;

            for problem_result in results.problem_results.values() {
                if let Some(&value) = problem_result.aggregated_metrics.get(key) {
                    let normalized_score = match metric.better_direction {
                        BetterDirection::Lower => 1.0 / (1.0 + value),
                        BetterDirection::Higher => value,
                    };
                    metric_score += normalized_score;
                    metric_count += 1;
                }
            }

            if metric_count > 0 {
                metric_score /= metric_count as f64;
                total_score += metric_score * metric.weight;
                total_weight += metric.weight;

                results
                    .overall_scores
                    .insert(metric.name.clone(), metric_score);
            }
        }

        if total_weight > 0.0 {
            results
                .overall_scores
                .insert("overall_score".to_string(), total_score / total_weight);
        }
    }

    fn calculate_rankings_and_tests(&self, all_results: &mut HashMap<String, BenchmarkResults>) {
        // Calculate rankings based on overall scores
        let mut optimizer_scores: Vec<(String, f64)> = all_results
            .iter()
            .filter_map(|(name, results)| {
                results
                    .overall_scores
                    .get("overall_score")
                    .map(|&score| (name.clone(), score))
            })
            .collect();

        optimizer_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (optimizer_name, score)) in optimizer_scores.iter().enumerate() {
            if let Some(results) = all_results.get_mut(optimizer_name) {
                results.ranking.overall_rank = rank + 1;
                results.ranking.ranking_score = *score;
            }
        }

        // Pairwise two-sample Kolmogorov-Smirnov test between each pair of
        // optimizers' pooled final-objective values (across all
        // problems/runs), so `statistical_tests` reflects a real comparison
        // instead of never being populated.
        let pooled_objectives: HashMap<String, Vec<f64>> = all_results
            .iter()
            .map(|(name, results)| {
                let values: Vec<f64> = results
                    .problem_results
                    .values()
                    .flat_map(|p| p.run_results.iter().map(|r| r.final_objective))
                    .filter(|v| v.is_finite())
                    .collect();
                (name.clone(), values)
            })
            .collect();

        let names: Vec<String> = optimizer_scores.into_iter().map(|(name, _)| name).collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let (name_a, name_b) = (&names[i], &names[j]);
                let (Some(values_a), Some(values_b)) =
                    (pooled_objectives.get(name_a), pooled_objectives.get(name_b))
                else {
                    continue;
                };
                if values_a.len() < 2 || values_b.len() < 2 {
                    continue;
                }

                let array_a = Array1::from_vec(values_a.clone());
                let array_b = Array1::from_vec(values_b.clone());
                let Ok((statistic, p_value)) =
                    scirs2_stats::ks_2samp(&array_a.view(), &array_b.view(), "two-sided")
                else {
                    continue;
                };

                const SIGNIFICANCE_LEVEL: f64 = 0.05;
                let test = StatisticalTest {
                    test_name: "Kolmogorov-Smirnov (two-sample)".to_string(),
                    optimizers: vec![name_a.clone(), name_b.clone()],
                    test_statistic: statistic,
                    p_value,
                    significance_level: SIGNIFICANCE_LEVEL,
                    significant: p_value < SIGNIFICANCE_LEVEL,
                    effect_size: None,
                };

                if let Some(results) = all_results.get_mut(name_a) {
                    results.statistical_tests.push(test.clone());
                }
                if let Some(results) = all_results.get_mut(name_b) {
                    results.statistical_tests.push(test);
                }
            }
        }
    }
}

/// Map a benchmark metric's declared type to the key
/// [`BenchmarkRunner::calculate_aggregated_metrics`] actually stores it
/// under. `EvaluationMetric::name` is a free-form display string (e.g.
/// "Final Objective Value") that does not match those keys (e.g.
/// "mean_final_objective"); metric types not produced by
/// `calculate_aggregated_metrics` are not scored (`None`) rather than
/// silently, permanently failing to match anything.
fn aggregated_metric_key(metric_type: &MetricType) -> Option<&'static str> {
    match metric_type {
        MetricType::FinalObjective => Some("mean_final_objective"),
        MetricType::TimeToConvergence => Some("mean_execution_time"),
        MetricType::SuccessRate => Some("success_rate"),
        _ => None,
    }
}

impl Default for BenchmarkSuiteMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            authors: Vec::new(),
            license: "MIT".to_string(),
            references: Vec::new(),
            target_audience: vec!["Researchers".to_string(), "Students".to_string()],
            keywords: Vec::new(),
            changelog: Vec::new(),
        }
    }
}

impl Default for BenchmarkSettings {
    fn default() -> Self {
        Self {
            num_runs: 10,
            random_seeds: (0..10).map(|i| 42 + i).collect(),
            max_iterations: 1000,
            max_time_seconds: 300.0, // 5 minutes
            convergence_tolerance: 1e-6,
            parallel_execution: true,
            num_threads: None,
            save_detailed_results: true,
            output_directory: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = AcademicBenchmarkSuite::standard_ml_suite();

        assert_eq!(suite.name, "Standard ML Optimization Benchmark");
        assert!(!suite.benchmarks.is_empty());
        assert!(!suite.metrics.is_empty());
    }

    #[test]
    fn test_benchmark_problem_creation() {
        let problem = AcademicBenchmarkSuite::create_quadratic_problem();

        assert_eq!(problem.name, "10D Quadratic Function");
        assert_eq!(problem.category, ProblemCategory::Convex);
        assert_eq!(problem.difficulty, DifficultyLevel::Easy);
        assert!(problem.optimal_solution.is_some());
    }

    #[test]
    fn test_benchmark_settings() {
        let settings = BenchmarkSettings::default();

        assert_eq!(settings.num_runs, 10);
        assert_eq!(settings.max_iterations, 1000);
        assert!(settings.parallel_execution);
    }

    fn make_runner(num_runs: usize, max_iterations: usize) -> BenchmarkRunner {
        let suite = AcademicBenchmarkSuite::new("Test Suite");
        let settings = BenchmarkSettings {
            num_runs,
            random_seeds: Vec::new(),
            max_iterations,
            max_time_seconds: 30.0,
            convergence_tolerance: 1e-6,
            parallel_execution: false,
            num_threads: None,
            save_detailed_results: false,
            output_directory: None,
        };
        BenchmarkRunner::new(suite, settings)
    }

    // Regression tests for F19: `run_single_instance` used to ignore both
    // `optimizer_config` and `seed` entirely, drawing the "final objective"
    // from a fixed `Rng` range keyed only on the problem's `FunctionType`.

    #[test]
    fn test_run_single_instance_is_deterministic_given_a_seed() {
        let runner = make_runner(1, 50);
        let problem = AcademicBenchmarkSuite::create_quadratic_problem();
        let config: OptimizerConfig<f64> = OptimizerConfig::new(0.1);

        let run_a = runner
            .run_single_instance::<f64>(&problem, "sgd", &config, 7)
            .expect("run should succeed");
        let run_b = runner
            .run_single_instance::<f64>(&problem, "sgd", &config, 7)
            .expect("run should succeed");

        assert_eq!(run_a.final_objective, run_b.final_objective);
        assert_eq!(run_a.trajectory, run_b.trajectory);
    }

    #[test]
    fn test_run_single_instance_respects_optimizer_config() {
        let runner = make_runner(1, 30);
        let problem = AcademicBenchmarkSuite::create_quadratic_problem();

        let stable_config: OptimizerConfig<f64> = OptimizerConfig::new(0.01);
        // Far past the stability limit for gradient descent on a unit
        // quadratic (which requires lr < 2.0): must diverge.
        let unstable_config: OptimizerConfig<f64> = OptimizerConfig::new(50.0);

        let stable = runner
            .run_single_instance::<f64>(&problem, "sgd", &stable_config, 1)
            .expect("run should succeed");
        let unstable = runner
            .run_single_instance::<f64>(&problem, "sgd", &unstable_config, 1)
            .expect("run should succeed");

        assert!(stable.final_objective.is_finite());
        assert!(
            unstable.final_objective > stable.final_objective,
            "an unstable learning rate must not converge as well as a stable one \
             (stable={}, unstable={}) -- the config must actually be used",
            stable.final_objective,
            unstable.final_objective
        );
    }

    #[test]
    fn test_run_single_instance_uses_the_seed_for_the_initial_point() {
        let runner = make_runner(1, 5);
        let problem = AcademicBenchmarkSuite::create_quadratic_problem();
        let config: OptimizerConfig<f64> = OptimizerConfig::new(0.01);

        let run_a = runner
            .run_single_instance::<f64>(&problem, "sgd", &config, 1)
            .expect("run should succeed");
        let run_b = runner
            .run_single_instance::<f64>(&problem, "sgd", &config, 2)
            .expect("run should succeed");

        // trajectory[0] is the objective at the seed-derived initial point,
        // before any optimizer step.
        assert_ne!(
            run_a.trajectory[0], run_b.trajectory[0],
            "different seeds must produce different starting points"
        );
    }

    #[test]
    fn test_select_optimizer_dispatches_by_name() {
        let config: OptimizerConfig<f64> = OptimizerConfig::new(0.1);
        assert!(matches!(
            select_optimizer("Adam", config.clone()),
            ChosenOptimizer::Adam(_)
        ));
        assert!(matches!(
            select_optimizer("adamw", config.clone()),
            ChosenOptimizer::Adam(_)
        ));
        assert!(matches!(
            select_optimizer("sgd", config.clone()),
            ChosenOptimizer::Sgd(_)
        ));
        assert!(matches!(
            select_optimizer("unknown", config),
            ChosenOptimizer::Sgd(_)
        ));
    }

    // Regression test for F20: `EvaluationMetric::name` display strings
    // never matched `calculate_aggregated_metrics`'s fixed keys, so
    // `overall_score` was never populated and every optimizer's
    // `overall_rank` stayed at its default of 0.
    #[test]
    fn test_run_benchmarks_produces_nonzero_distinct_ranks() {
        let mut suite = AcademicBenchmarkSuite::new("Ranking Test Suite");
        suite.add_benchmark(AcademicBenchmarkSuite::create_quadratic_problem());
        suite.add_metric(AcademicBenchmarkSuite::create_final_objective_metric());
        suite.add_metric(AcademicBenchmarkSuite::create_convergence_time_metric());
        suite.add_metric(AcademicBenchmarkSuite::create_success_rate_metric());

        let settings = BenchmarkSettings {
            num_runs: 3,
            random_seeds: Vec::new(),
            max_iterations: 20,
            max_time_seconds: 30.0,
            convergence_tolerance: 1e-6,
            parallel_execution: false,
            num_threads: None,
            save_detailed_results: false,
            output_directory: None,
        };
        let runner = BenchmarkRunner::new(suite, settings);

        let optimizers: Vec<(&str, OptimizerConfig<f64>)> = vec![
            ("good_sgd", OptimizerConfig::new(0.1)),
            ("bad_sgd", OptimizerConfig::new(50.0)),
        ];

        let results = runner
            .run_benchmarks::<f64>(&optimizers)
            .expect("benchmarks should run");

        let good = &results["good_sgd"];
        let bad = &results["bad_sgd"];

        assert_ne!(
            good.ranking.overall_rank, 0,
            "rank must not stay at the default 0"
        );
        assert_ne!(
            bad.ranking.overall_rank, 0,
            "rank must not stay at the default 0"
        );
        assert_ne!(good.ranking.overall_rank, bad.ranking.overall_rank);
        assert!(good.overall_scores.contains_key("overall_score"));
        assert_eq!(
            good.ranking.overall_rank, 1,
            "the well-tuned optimizer should outrank the divergent one"
        );
    }

    fn make_run_result(final_objective: f64) -> RunResult {
        RunResult {
            run_id: uuid::Uuid::new_v4().to_string(),
            random_seed: 0,
            final_objective,
            converged: final_objective < 1.0,
            iterations: 10,
            execution_time: 0.001,
            function_evaluations: 10,
            gradient_evaluations: 10,
            memory_usage: 0,
            trajectory: vec![final_objective],
            error_info: None,
        }
    }

    // Regression tests for F90: `confidence_intervals` was always an empty
    // map, and `std_objective` used the population-variance denominator `n`
    // instead of the sample-variance denominator `n-1` appropriate for a
    // sample of independent runs.
    #[test]
    fn test_calculate_statistics_reports_real_confidence_intervals_and_sample_std() {
        let runner = make_runner(1, 1);
        let run_results = vec![
            make_run_result(1.0),
            make_run_result(2.0),
            make_run_result(3.0),
            make_run_result(4.0),
            make_run_result(5.0),
        ];

        let stats = runner.calculate_statistics(&run_results);

        // Sample std (n-1 denominator) of [1,2,3,4,5] is sqrt(2.5); the
        // population std (n denominator) would be sqrt(2.0) instead.
        assert!(
            (stats.std_objective - 2.5_f64.sqrt()).abs() < 1e-9,
            "expected sample std sqrt(2.5) ~= {:.4}, got {}",
            2.5_f64.sqrt(),
            stats.std_objective
        );

        assert!(!stats.confidence_intervals.is_empty());
        let (lower, upper) = stats.confidence_intervals["mean_objective_95"];
        assert!(lower < stats.mean_objective && stats.mean_objective < upper);
        let (rate_lower, rate_upper) = stats.confidence_intervals["success_rate_95"];
        assert!((0.0..=1.0).contains(&rate_lower));
        assert!((0.0..=1.0).contains(&rate_upper));
    }

    #[test]
    fn test_student_t_critical_value_matches_known_table_value() {
        // t_{0.975, df=1} is a well-known tabulated constant (~12.706).
        let t = student_t_critical_value(1.0, 0.975);
        assert!((t - 12.706).abs() < 0.01, "expected ~12.706, got {t}");
    }

    #[test]
    fn test_wilson_score_interval_stays_within_unit_bounds() {
        let (lower, upper) = wilson_score_interval_95(8, 10);
        assert!((0.0..=1.0).contains(&lower));
        assert!((0.0..=1.0).contains(&upper));
        assert!(lower < 0.8 && upper > 0.8);
    }

    #[test]
    fn test_calculate_statistics_sort_does_not_panic_on_nan() {
        // Regression for the reachable NaN-panic half of F83/F90: a
        // divergent run can legitimately produce a non-finite objective.
        let runner = make_runner(1, 1);
        let run_results = vec![
            make_run_result(1.0),
            make_run_result(f64::NAN),
            make_run_result(2.0),
        ];

        let stats = runner.calculate_statistics(&run_results);
        assert_eq!(stats.total_runs, 3);
    }
}
