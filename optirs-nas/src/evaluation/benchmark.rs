//! Benchmark suite for evaluating optimizer architectures
//!
//! Provides comprehensive benchmark tests and test functions.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant, SystemTime};

use super::types::*;
use crate::error::{OptimError, Result};
use crate::{EvaluationConfig, OptimizerArchitecture, ResourceUsage};

/// Test function for benchmarks
#[derive(Debug, Clone)]
pub struct TestFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function type
    pub function_type: TestFunctionType,

    /// Function parameters
    pub parameters: HashMap<String, T>,

    /// Dimensionality
    pub dimensions: usize,

    /// Evaluation budget
    pub max_evaluations: usize,

    /// Target performance
    pub target_performance: Option<T>,
}

/// Standard benchmark test
#[derive(Debug, Clone)]
pub struct StandardBenchmark<T: Float + Debug + Send + Sync + 'static> {
    /// Benchmark name
    pub name: String,

    /// Benchmark type
    pub benchmark_type: BenchmarkType,

    /// Test function
    pub test_function: TestFunction<T>,

    /// Expected performance range
    pub expected_range: (T, T),

    /// Difficulty level
    pub difficulty: DifficultyLevel,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Custom benchmark test
#[derive(Debug, Clone)]
pub struct CustomBenchmark<T: Float + Debug + Send + Sync + 'static> {
    /// Benchmark name
    pub name: String,

    /// Custom test configuration
    pub config: CustomBenchmarkConfig<T>,

    /// Evaluation function
    pub evaluator: CustomEvaluator<T>,

    /// Validation criteria
    pub validation: ValidationCriteria<T>,
}

/// Custom benchmark configuration
#[derive(Debug, Clone)]
pub struct CustomBenchmarkConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Problem definition
    pub problem_definition: ProblemDefinition<T>,

    /// Evaluation criteria
    pub evaluation_criteria: Vec<EvaluationCriterion<T>>,

    /// Success metrics
    pub success_metrics: SuccessMetrics<T>,

    /// Termination conditions
    pub termination_conditions: TerminationConditions<T>,
}

/// Problem definition for custom benchmarks
#[derive(Debug, Clone)]
pub struct ProblemDefinition<T: Float + Debug + Send + Sync + 'static> {
    /// Problem type
    pub problem_type: ProblemType,

    /// Input dimensionality
    pub input_dim: usize,

    /// Output dimensionality
    pub output_dim: usize,

    /// Dataset size
    pub dataset_size: usize,

    /// Problem-specific parameters
    pub parameters: HashMap<String, T>,

    /// Data characteristics
    pub data_characteristics: DataCharacteristics<T>,
}

/// Custom evaluator function
#[derive(Debug, Clone)]
pub struct CustomEvaluator<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluator type
    pub evaluator_type: EvaluatorType,

    /// Evaluation function parameters
    pub parameters: HashMap<String, T>,

    /// Input/output specifications
    pub io_spec: IOSpecification,
}

/// Benchmark metadata
#[derive(Debug, Clone)]
pub struct BenchmarkMetadata {
    /// Suite name
    pub name: String,

    /// Version
    pub version: String,

    /// Description
    pub description: String,

    /// Creation date
    pub created_at: SystemTime,

    /// Last updated
    pub updated_at: SystemTime,

    /// Author information
    pub author: String,

    /// License
    pub license: String,
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults<T: Float + Debug + Send + Sync + 'static> {
    /// Individual test results
    pub test_results: Vec<TestResult<T>>,

    /// Overall score
    pub overall_score: T,

    /// Performance ranking
    pub ranking: PerformanceRanking,

    /// Statistical summary
    pub statistical_summary: StatisticalSummary<T>,

    /// Resource usage summary
    pub resource_summary: ResourceSummary<T>,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult<T: Float + Debug + Send + Sync + 'static> {
    /// Test name
    pub test_name: String,

    /// Score
    pub score: T,

    /// Normalized score
    pub normalized_score: T,

    /// Percentile rank
    pub percentile_rank: T,

    /// Execution time
    pub execution_time: Duration,

    /// Resource usage
    pub resource_usage: ResourceUsage<T>,

    /// Additional metrics
    pub metrics: HashMap<String, T>,
}

/// Comprehensive benchmark suite
#[derive(Debug)]
pub struct BenchmarkSuite<T: Float + Debug + Send + Sync + 'static> {
    /// Standard benchmarks
    standard_benchmarks: Vec<StandardBenchmark<T>>,

    /// Custom benchmarks
    custom_benchmarks: Vec<CustomBenchmark<T>>,

    /// Benchmark metadata
    metadata: BenchmarkMetadata,

    /// Benchmark results cache
    results_cache: HashMap<String, BenchmarkResults<T>>,

    /// Empirical distribution of normalized scores observed per benchmark name.
    ///
    /// Populated as architectures are benchmarked and used to compute a *real*
    /// percentile rank for each new result (the fraction of recorded runs the
    /// new run is at least as good as, the new run included).
    score_history: HashMap<String, Vec<f64>>,

    /// Number of optimizer steps executed per benchmark run. Derived from the
    /// evaluation configuration's epoch budget in [`BenchmarkSuite::initialize`].
    steps_per_benchmark: usize,
}

/// Default number of optimizer steps executed per benchmark when the
/// evaluation configuration does not specify a usable epoch budget.
const DEFAULT_STEPS_PER_BENCHMARK: usize = 60;

/// Upper bound on the number of optimizer steps executed per benchmark.
///
/// Architecture search evaluates hundreds of candidates, each running every
/// registered benchmark; capping the inner loop keeps a full search bounded
/// while still being long enough for the optimizers to separate.
const MAX_STEPS_PER_BENCHMARK: usize = 400;

impl<T: Float + Debug + Default + Send + Sync> BenchmarkSuite<T> {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            standard_benchmarks: Vec::new(),
            custom_benchmarks: Vec::new(),
            metadata: BenchmarkMetadata {
                name: "Standard Benchmark Suite".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: "Optimizer evaluation suite over classical \
                              continuous-optimization test functions"
                    .to_string(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                author: "COOLJAPAN OU (Team KitaSan)".to_string(),
                license: "Apache-2.0".to_string(),
            },
            results_cache: HashMap::new(),
            score_history: HashMap::new(),
            steps_per_benchmark: DEFAULT_STEPS_PER_BENCHMARK,
        })
    }

    pub(crate) fn initialize(&mut self, config: &EvaluationConfig) -> Result<()> {
        // Drive the inner optimization loop from the configured epoch budget so
        // callers can trade evaluation fidelity against search wall time.
        self.steps_per_benchmark = (config.epochs as usize).max(1).min(MAX_STEPS_PER_BENCHMARK);

        // Initialize standard benchmarks
        self.standard_benchmarks.clear();
        self.add_standard_benchmarks()?;
        Ok(())
    }

    /// Number of optimizer steps executed per benchmark run.
    pub fn steps_per_benchmark(&self) -> usize {
        self.steps_per_benchmark
    }

    /// Registered standard benchmarks.
    pub fn standard_benchmarks(&self) -> &[StandardBenchmark<T>] {
        &self.standard_benchmarks
    }

    fn add_standard_benchmarks(&mut self) -> Result<()> {
        // Add Rosenbrock function benchmark
        self.standard_benchmarks.push(StandardBenchmark {
            name: "Rosenbrock".to_string(),
            benchmark_type: BenchmarkType::NonConvexOptimization,
            test_function: TestFunction {
                function_type: TestFunctionType::Rosenbrock,
                parameters: HashMap::new(),
                dimensions: 10,
                max_evaluations: 1000,
                target_performance: Some(
                    scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()),
                ),
            },
            expected_range: (
                scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero()),
                scirs2_core::numeric::NumCast::from(1e-2).unwrap_or_else(|| T::zero()),
            ),
            difficulty: DifficultyLevel::Medium,
            resource_requirements: ResourceRequirements {
                memory_mb: 100,
                cpu_cores: 1,
                gpu_memory_mb: None,
                max_runtime_seconds: 300,
                storage_mb: 10,
            },
        });

        // Add Quadratic benchmark
        self.standard_benchmarks.push(StandardBenchmark {
            name: "Quadratic".to_string(),
            benchmark_type: BenchmarkType::ConvergenceSpeed,
            test_function: TestFunction {
                function_type: TestFunctionType::Quadratic,
                parameters: HashMap::new(),
                dimensions: 20,
                max_evaluations: 500,
                target_performance: Some(
                    scirs2_core::numeric::NumCast::from(1e-8).unwrap_or_else(|| T::zero()),
                ),
            },
            expected_range: (
                scirs2_core::numeric::NumCast::from(1e-10).unwrap_or_else(|| T::zero()),
                scirs2_core::numeric::NumCast::from(1e-4).unwrap_or_else(|| T::zero()),
            ),
            difficulty: DifficultyLevel::Easy,
            resource_requirements: ResourceRequirements {
                memory_mb: 50,
                cpu_cores: 1,
                gpu_memory_mb: None,
                max_runtime_seconds: 120,
                storage_mb: 5,
            },
        });

        // Add Rastrigin benchmark (highly multimodal)
        self.standard_benchmarks.push(StandardBenchmark {
            name: "Rastrigin".to_string(),
            benchmark_type: BenchmarkType::NonConvexOptimization,
            test_function: TestFunction {
                function_type: TestFunctionType::Rastrigin,
                parameters: HashMap::new(),
                dimensions: 10,
                max_evaluations: 1000,
                target_performance: Some(
                    scirs2_core::numeric::NumCast::from(1e-4).unwrap_or_else(|| T::zero()),
                ),
            },
            expected_range: (
                scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()),
                scirs2_core::numeric::NumCast::from(1e1).unwrap_or_else(|| T::zero()),
            ),
            difficulty: DifficultyLevel::Hard,
            resource_requirements: ResourceRequirements {
                memory_mb: 100,
                cpu_cores: 1,
                gpu_memory_mb: None,
                max_runtime_seconds: 300,
                storage_mb: 10,
            },
        });

        // Add Sphere benchmark (smooth convex baseline)
        self.standard_benchmarks.push(StandardBenchmark {
            name: "Sphere".to_string(),
            benchmark_type: BenchmarkType::ConvergenceSpeed,
            test_function: TestFunction {
                function_type: TestFunctionType::Sphere,
                parameters: HashMap::new(),
                dimensions: 20,
                max_evaluations: 500,
                target_performance: Some(
                    scirs2_core::numeric::NumCast::from(1e-10).unwrap_or_else(|| T::zero()),
                ),
            },
            expected_range: (
                scirs2_core::numeric::NumCast::from(1e-12).unwrap_or_else(|| T::zero()),
                scirs2_core::numeric::NumCast::from(1e-2).unwrap_or_else(|| T::zero()),
            ),
            difficulty: DifficultyLevel::Easy,
            resource_requirements: ResourceRequirements {
                memory_mb: 50,
                cpu_cores: 1,
                gpu_memory_mb: None,
                max_runtime_seconds: 120,
                storage_mb: 5,
            },
        });

        Ok(())
    }

    /// Run every registered standard benchmark against `architecture`.
    ///
    /// Each benchmark instantiates the concrete optimizer described by the
    /// architecture and actually runs it for [`Self::steps_per_benchmark`]
    /// gradient steps on the benchmark's test function, reporting the objective
    /// value it achieved.
    pub(crate) fn run_benchmarks(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<Vec<TestResult<T>>> {
        let mut results = Vec::with_capacity(self.standard_benchmarks.len());

        // Index-based iteration: `run_single_benchmark` records the observed
        // score into `self.score_history`, so it needs `&mut self`.
        for idx in 0..self.standard_benchmarks.len() {
            let benchmark = self.standard_benchmarks[idx].clone();
            let result = self.run_single_benchmark(&benchmark, architecture)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute a single benchmark: build the candidate optimizer from the
    /// architecture, minimize the test function for a fixed step budget, and
    /// report the achieved objective together with a normalized score, an
    /// empirical percentile rank and measured execution time.
    fn run_single_benchmark(
        &mut self,
        benchmark: &StandardBenchmark<T>,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<TestResult<T>> {
        let function_type = &benchmark.test_function.function_type;
        let dimensions = benchmark.test_function.dimensions.max(1);

        // Deterministic starting point: depends only on the test function and
        // dimensionality, so every candidate architecture is compared on
        // exactly the same problem instance and repeated runs reproduce.
        let mut params = initial_point(function_type, dimensions);
        let initial_objective = objective_value(function_type, &params)?;

        let spec = OptimizerSpec::from_architecture(architecture);
        let mut optimizer = CandidateOptimizer::new(spec.clone(), params.len());

        let start_time = Instant::now();
        let mut best_objective = initial_objective;

        for _ in 0..self.steps_per_benchmark {
            let gradient = objective_gradient(function_type, &params)?;
            optimizer.step(&mut params, &gradient);

            // A diverging optimizer can produce non-finite iterates; stop there
            // and keep the best objective observed so far rather than
            // propagating NaN into the score.
            if params.iter().any(|v| !v.is_finite()) {
                break;
            }

            let value = objective_value(function_type, &params)?;
            if value < best_objective {
                best_objective = value;
            }
        }

        let execution_time = start_time.elapsed();

        // Normalized score: fraction of the initial objective that was removed.
        // 1.0 means the optimizer drove the objective to zero, 0.0 means it made
        // no progress (or diverged). Bounded and comparable across functions.
        let normalized = if initial_objective > 0.0 && initial_objective.is_finite() {
            ((initial_objective - best_objective) / initial_objective).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Empirical percentile rank against all runs recorded for this
        // benchmark so far, this run included.
        let history = self
            .score_history
            .entry(benchmark.name.clone())
            .or_default();
        history.push(normalized);
        let total = history.len();
        let at_or_below = history.iter().filter(|&&s| s <= normalized).count();
        let percentile = at_or_below as f64 / total as f64;

        // Honest resource accounting. Only wall-clock time and the analytic
        // working-set size of the optimizer are actually known here; GPU,
        // network, disk, energy and cost are not measured and are reported as
        // zero rather than fabricated.
        let working_set_bytes =
            (dimensions * spec.state_slots_per_parameter() * std::mem::size_of::<f64>()) as f64;
        let memory_gb = working_set_bytes / 1.0e9;
        let elapsed_secs = execution_time.as_secs_f64();

        let mut metrics = HashMap::new();
        metrics.insert(
            "initial_objective".to_string(),
            to_t::<T>(initial_objective),
        );
        metrics.insert("final_objective".to_string(), to_t::<T>(best_objective));
        metrics.insert(
            "steps".to_string(),
            to_t::<T>(self.steps_per_benchmark as f64),
        );
        metrics.insert("learning_rate".to_string(), to_t::<T>(spec.learning_rate));

        Ok(TestResult {
            test_name: benchmark.name.clone(),
            score: to_t::<T>(best_objective),
            normalized_score: to_t::<T>(normalized),
            percentile_rank: to_t::<T>(percentile),
            execution_time,
            resource_usage: ResourceUsage {
                memory_gb: to_t::<T>(memory_gb),
                cpu_time_seconds: to_t::<T>(elapsed_secs),
                gpu_time_seconds: T::zero(),
                energy_kwh: T::zero(),
                network_io_gb: T::zero(),
                disk_io_gb: T::zero(),
                peak_memory_gb: to_t::<T>(memory_gb),
                efficiency_score: to_t::<T>(normalized / (1.0 + elapsed_secs)),
                cost_usd: T::zero(),
                network_gb: T::zero(),
            },
            metrics,
        })
    }
}

/// Convert a finite `f64` into `T`, falling back to zero.
fn to_t<T: Float>(value: f64) -> T {
    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero)
}

// ---------------------------------------------------------------------------
// Test functions
// ---------------------------------------------------------------------------

/// Deterministic starting point for a test function of the given dimension.
///
/// The classical literature start points are used where they exist
/// (Rosenbrock's `(-1.2, 1, -1.2, 1, ...)`), otherwise a fixed, reproducible
/// offset from the optimum that leaves a non-trivial amount of work to do.
fn initial_point(function_type: &TestFunctionType, dimensions: usize) -> Vec<f64> {
    match function_type {
        TestFunctionType::Rosenbrock => (0..dimensions)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect(),
        TestFunctionType::Rastrigin => (0..dimensions)
            .map(|i| 2.5 - 0.1 * ((i % 7) as f64))
            .collect(),
        TestFunctionType::Ackley => (0..dimensions)
            .map(|i| 3.0 - 0.2 * ((i % 5) as f64))
            .collect(),
        TestFunctionType::Beale => vec![1.0, 1.0],
        // Quadratic / Sphere and anything else: a fixed non-zero start.
        _ => (0..dimensions)
            .map(|i| 1.0 + 0.5 * ((i % 4) as f64))
            .collect(),
    }
}

/// Objective value of a supported continuous test function.
///
/// Returns [`OptimError::NotImplemented`] for function types that have no
/// analytic implementation, so a caller never silently benchmarks against a
/// fabricated score.
fn objective_value(function_type: &TestFunctionType, x: &[f64]) -> Result<f64> {
    if x.is_empty() {
        return Err(OptimError::EvaluationError(
            "test function evaluated on an empty parameter vector".to_string(),
        ));
    }

    let value = match function_type {
        // f(x) = sum_i i-weighted x_i^2 — an ill-conditioned quadratic bowl.
        TestFunctionType::Quadratic => x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (1.0 + i as f64) * xi * xi)
            .sum(),

        // f(x) = sum_i x_i^2 — the isotropic sphere.
        TestFunctionType::Sphere => x.iter().map(|&xi| xi * xi).sum(),

        // f(x) = sum_{i<n-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        TestFunctionType::Rosenbrock => {
            let mut acc = 0.0;
            for i in 0..x.len().saturating_sub(1) {
                let t = x[i + 1] - x[i] * x[i];
                let u = 1.0 - x[i];
                acc += 100.0 * t * t + u * u;
            }
            acc
        }

        // f(x) = 10 n + sum_i [x_i^2 - 10 cos(2 pi x_i)]
        TestFunctionType::Rastrigin => {
            let two_pi = 2.0 * std::f64::consts::PI;
            10.0 * x.len() as f64
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (two_pi * xi).cos())
                    .sum::<f64>()
        }

        // Standard Ackley function (a = 20, b = 0.2, c = 2 pi).
        TestFunctionType::Ackley => {
            let n = x.len() as f64;
            let two_pi = 2.0 * std::f64::consts::PI;
            let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
            let sum_cos = x.iter().map(|&xi| (two_pi * xi).cos()).sum::<f64>() / n;
            -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
        }

        // Beale function (2-D only).
        TestFunctionType::Beale => {
            if x.len() < 2 {
                return Err(OptimError::EvaluationError(
                    "Beale test function requires two dimensions".to_string(),
                ));
            }
            let (a, b) = (x[0], x[1]);
            let t1 = 1.5 - a + a * b;
            let t2 = 2.25 - a + a * b * b;
            let t3 = 2.625 - a + a * b * b * b;
            t1 * t1 + t2 * t2 + t3 * t3
        }

        other => {
            return Err(OptimError::NotImplemented(format!(
                "test function {:?} has no analytic implementation",
                other
            )))
        }
    };

    Ok(value)
}

/// Analytic gradient of a supported continuous test function.
fn objective_gradient(function_type: &TestFunctionType, x: &[f64]) -> Result<Vec<f64>> {
    if x.is_empty() {
        return Err(OptimError::EvaluationError(
            "test function differentiated on an empty parameter vector".to_string(),
        ));
    }

    let gradient = match function_type {
        TestFunctionType::Quadratic => x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * (1.0 + i as f64) * xi)
            .collect(),

        TestFunctionType::Sphere => x.iter().map(|&xi| 2.0 * xi).collect(),

        TestFunctionType::Rosenbrock => {
            let n = x.len();
            let mut g = vec![0.0; n];
            for i in 0..n.saturating_sub(1) {
                let t = x[i + 1] - x[i] * x[i];
                g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
                g[i + 1] += 200.0 * t;
            }
            g
        }

        TestFunctionType::Rastrigin => {
            let two_pi = 2.0 * std::f64::consts::PI;
            x.iter()
                .map(|&xi| 2.0 * xi + 10.0 * two_pi * (two_pi * xi).sin())
                .collect()
        }

        TestFunctionType::Ackley => {
            let n = x.len() as f64;
            let two_pi = 2.0 * std::f64::consts::PI;
            let mean_sq = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
            let root = mean_sq.sqrt();
            let mean_cos = x.iter().map(|&xi| (two_pi * xi).cos()).sum::<f64>() / n;
            let exp1 = (-0.2 * root).exp();
            let exp2 = mean_cos.exp();

            x.iter()
                .map(|&xi| {
                    // d/dxi of -20 exp(-0.2 sqrt(mean(x^2)))
                    let first = if root > 1e-12 {
                        4.0 * exp1 * xi / (n * root)
                    } else {
                        0.0
                    };
                    // d/dxi of -exp(mean(cos(2 pi x)))
                    let second = exp2 * two_pi * (two_pi * xi).sin() / n;
                    first + second
                })
                .collect()
        }

        TestFunctionType::Beale => {
            if x.len() < 2 {
                return Err(OptimError::EvaluationError(
                    "Beale test function requires two dimensions".to_string(),
                ));
            }
            let (a, b) = (x[0], x[1]);
            let t1 = 1.5 - a + a * b;
            let t2 = 2.25 - a + a * b * b;
            let t3 = 2.625 - a + a * b * b * b;
            let da = 2.0 * t1 * (b - 1.0) + 2.0 * t2 * (b * b - 1.0) + 2.0 * t3 * (b * b * b - 1.0);
            let db = 2.0 * t1 * a + 2.0 * t2 * (2.0 * a * b) + 2.0 * t3 * (3.0 * a * b * b);
            vec![da, db]
        }

        other => {
            return Err(OptimError::NotImplemented(format!(
                "test function {:?} has no analytic gradient",
                other
            )))
        }
    };

    Ok(gradient)
}

// ---------------------------------------------------------------------------
// Candidate optimizer construction
// ---------------------------------------------------------------------------

/// Optimizer family a candidate architecture resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimizerFamily {
    Sgd,
    Adam,
    AdamW,
    RmsProp,
    AdaGrad,
    Lion,
    RAdam,
}

/// Concrete, runnable description of the optimizer encoded by an architecture.
#[derive(Debug, Clone)]
struct OptimizerSpec {
    family: OptimizerFamily,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    momentum: f64,
}

impl OptimizerSpec {
    /// Resolve the architecture's leading recognised component into an
    /// optimizer family and read its hyperparameters.
    ///
    /// Hyperparameters are looked up in the architecture's `hyperparameters`
    /// map first and then in `parameters`, so architectures produced by either
    /// convention are honoured. Values outside a numerically sane band are
    /// clamped so a pathological candidate cannot make the benchmark diverge
    /// into non-finite arithmetic before the loop's own guard fires.
    fn from_architecture<T: Float + Debug + Send + Sync + 'static>(
        architecture: &OptimizerArchitecture<T>,
    ) -> Self {
        let family = architecture
            .components
            .iter()
            .find_map(|c| family_from_name(c))
            .unwrap_or(OptimizerFamily::Sgd);

        let lookup = |keys: &[&str], default: f64| -> f64 {
            for key in keys {
                if let Some(value) = architecture
                    .hyperparameters
                    .get(*key)
                    .or_else(|| architecture.parameters.get(*key))
                {
                    if let Some(v) = value.to_f64() {
                        if v.is_finite() {
                            return v;
                        }
                    }
                }
            }
            default
        };

        let default_lr = match family {
            OptimizerFamily::Sgd => 1e-2,
            OptimizerFamily::Lion => 1e-4,
            _ => 1e-3,
        };

        Self {
            family,
            learning_rate: lookup(&["learning_rate", "lr"], default_lr).clamp(1e-8, 1.0),
            beta1: lookup(&["beta1", "momentum"], 0.9).clamp(0.0, 0.9999),
            beta2: lookup(&["beta2", "decay", "rho"], 0.999).clamp(0.0, 0.999999),
            epsilon: lookup(&["epsilon", "eps"], 1e-8).clamp(1e-16, 1e-1),
            weight_decay: lookup(&["weight_decay", "l2"], 0.0).clamp(0.0, 1.0),
            momentum: lookup(&["momentum", "beta1"], 0.9).clamp(0.0, 0.9999),
        }
    }

    /// Number of `f64` slots the optimizer keeps per model parameter
    /// (the parameter vector itself plus its persistent optimizer state).
    /// Used for the analytic working-set estimate reported as memory usage.
    fn state_slots_per_parameter(&self) -> usize {
        match self.family {
            OptimizerFamily::AdaGrad => 2, // params + accumulated squares
            OptimizerFamily::RmsProp => 2, // params + squared-gradient EMA
            OptimizerFamily::Lion => 2,    // params + momentum
            OptimizerFamily::Sgd => {
                if self.momentum > 0.0 {
                    2
                } else {
                    1
                }
            }
            // params + first and second moments
            OptimizerFamily::Adam | OptimizerFamily::AdamW | OptimizerFamily::RAdam => 3,
        }
    }
}

/// Map a component name (the `Debug` rendering of a component type) onto an
/// optimizer family. Matching is case-insensitive and order-sensitive so
/// `AdamW` is not swallowed by the `Adam` prefix.
fn family_from_name(name: &str) -> Option<OptimizerFamily> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("adamw") {
        Some(OptimizerFamily::AdamW)
    } else if lower.contains("radam") {
        Some(OptimizerFamily::RAdam)
    } else if lower.contains("adagrad") {
        Some(OptimizerFamily::AdaGrad)
    } else if lower.contains("rmsprop") {
        Some(OptimizerFamily::RmsProp)
    } else if lower.contains("lion") {
        Some(OptimizerFamily::Lion)
    } else if lower.contains("adam") {
        Some(OptimizerFamily::Adam)
    } else if lower.contains("sgd") || lower.contains("momentum") || lower.contains("nesterov") {
        Some(OptimizerFamily::Sgd)
    } else {
        None
    }
}

/// A runnable optimizer instance carrying its persistent state.
///
/// The update rules are implemented here rather than delegated so that the
/// benchmark harness is hermetic: a NAS score depends only on the candidate
/// architecture and the test function, never on a sibling crate's internals,
/// and a given architecture reproduces bit-for-bit across releases.
///
/// All rules follow the published formulations:
/// * SGD with (heavy-ball) momentum and L2 weight decay,
/// * Adam (Kingma & Ba, 2015) with bias correction,
/// * AdamW (Loshchilov & Hutter, 2019) with decoupled weight decay,
/// * RMSprop (Tieleman & Hinton, 2012),
/// * AdaGrad (Duchi et al., 2011),
/// * Lion (Chen et al., 2023),
/// * RAdam (Liu et al., 2020) with the variance-rectification term.
#[derive(Debug, Clone)]
struct CandidateOptimizer {
    spec: OptimizerSpec,
    /// First moment / momentum buffer.
    m: Vec<f64>,
    /// Second moment / squared-gradient accumulator.
    v: Vec<f64>,
    /// Step counter (1-based once the first step has been taken).
    t: u32,
}

impl CandidateOptimizer {
    fn new(spec: OptimizerSpec, dimensions: usize) -> Self {
        Self {
            spec,
            m: vec![0.0; dimensions],
            v: vec![0.0; dimensions],
            t: 0,
        }
    }

    /// Apply one in-place optimizer update to `params`.
    fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t = self.t.saturating_add(1);
        let t = self.t as f64;

        let lr = self.spec.learning_rate;
        let b1 = self.spec.beta1;
        let b2 = self.spec.beta2;
        let eps = self.spec.epsilon;
        let wd = self.spec.weight_decay;

        match self.spec.family {
            OptimizerFamily::Sgd => {
                let mu = self.spec.momentum;
                for i in 0..params.len() {
                    let g = gradients[i] + wd * params[i];
                    self.m[i] = mu * self.m[i] + g;
                    params[i] -= lr * self.m[i];
                }
            }

            OptimizerFamily::Adam => {
                let bc1 = 1.0 - b1.powf(t);
                let bc2 = 1.0 - b2.powf(t);
                for i in 0..params.len() {
                    let g = gradients[i] + wd * params[i];
                    self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
                    self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
                    let m_hat = self.m[i] / bc1.max(f64::MIN_POSITIVE);
                    let v_hat = self.v[i] / bc2.max(f64::MIN_POSITIVE);
                    params[i] -= lr * m_hat / (v_hat.max(0.0).sqrt() + eps);
                }
            }

            OptimizerFamily::AdamW => {
                let bc1 = 1.0 - b1.powf(t);
                let bc2 = 1.0 - b2.powf(t);
                for i in 0..params.len() {
                    // Decoupled weight decay: applied to the parameter, not the
                    // gradient, so it does not enter the moment estimates.
                    params[i] -= lr * wd * params[i];

                    let g = gradients[i];
                    self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
                    self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
                    let m_hat = self.m[i] / bc1.max(f64::MIN_POSITIVE);
                    let v_hat = self.v[i] / bc2.max(f64::MIN_POSITIVE);
                    params[i] -= lr * m_hat / (v_hat.max(0.0).sqrt() + eps);
                }
            }

            OptimizerFamily::RmsProp => {
                // `beta2` doubles as RMSprop's decay factor `rho`.
                let rho = b2.clamp(0.5, 0.9999);
                for i in 0..params.len() {
                    let g = gradients[i] + wd * params[i];
                    self.v[i] = rho * self.v[i] + (1.0 - rho) * g * g;
                    params[i] -= lr * g / (self.v[i].max(0.0).sqrt() + eps);
                }
            }

            OptimizerFamily::AdaGrad => {
                for i in 0..params.len() {
                    let g = gradients[i] + wd * params[i];
                    self.v[i] += g * g;
                    params[i] -= lr * g / (self.v[i].max(0.0).sqrt() + eps);
                }
            }

            OptimizerFamily::Lion => {
                for i in 0..params.len() {
                    let g = gradients[i];
                    // Update direction is the sign of an interpolated momentum.
                    let interpolated = b1 * self.m[i] + (1.0 - b1) * g;
                    let direction = if interpolated > 0.0 {
                        1.0
                    } else if interpolated < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    params[i] -= lr * (direction + wd * params[i]);
                    // Momentum is updated with the second interpolation factor.
                    self.m[i] = b2 * self.m[i] + (1.0 - b2) * g;
                }
            }

            OptimizerFamily::RAdam => {
                let bc1 = 1.0 - b1.powf(t);
                let b2_t = b2.powf(t);
                let bc2 = 1.0 - b2_t;
                // Maximum length of the approximated SMA and its value at step t.
                let rho_inf = 2.0 / (1.0 - b2).max(f64::MIN_POSITIVE) - 1.0;
                let rho_t = rho_inf - 2.0 * t * b2_t / bc2.max(f64::MIN_POSITIVE);

                // Rectification term; only defined once the variance estimate is
                // tractable (rho_t > 4), otherwise fall back to plain SGD on the
                // bias-corrected first moment.
                let rectification = if rho_t > 4.0 {
                    let numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
                    let denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
                    if denominator > 0.0 {
                        Some((numerator / denominator).max(0.0).sqrt())
                    } else {
                        None
                    }
                } else {
                    None
                };

                for i in 0..params.len() {
                    let g = gradients[i] + wd * params[i];
                    self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
                    self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
                    let m_hat = self.m[i] / bc1.max(f64::MIN_POSITIVE);

                    match rectification {
                        Some(r) => {
                            let v_hat = (self.v[i] / bc2.max(f64::MIN_POSITIVE)).max(0.0).sqrt();
                            params[i] -= lr * r * m_hat / (v_hat + eps);
                        }
                        None => params[i] -= lr * m_hat,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn architecture(component: &str, hyper: &[(&str, f64)]) -> OptimizerArchitecture<f64> {
        let mut hyperparameters = HashMap::new();
        for (k, v) in hyper {
            hyperparameters.insert((*k).to_string(), *v);
        }
        OptimizerArchitecture {
            components: vec![component.to_string()],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters,
            architecture_id: format!("{}_test", component),
        }
    }

    fn suite() -> BenchmarkSuite<f64> {
        let mut suite = BenchmarkSuite::<f64>::new().expect("suite");
        let mut config = EvaluationConfig::default();
        config.epochs = 40;
        suite.initialize(&config).expect("initialize");
        suite
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::<f64>::new();
        assert!(suite.is_ok());
    }

    #[test]
    fn test_standard_benchmarks_registered() {
        let suite = suite();
        let names: Vec<&str> = suite
            .standard_benchmarks()
            .iter()
            .map(|b| b.name.as_str())
            .collect();
        assert!(names.contains(&"Rosenbrock"));
        assert!(names.contains(&"Quadratic"));
        assert!(names.contains(&"Rastrigin"));
        assert!(names.contains(&"Sphere"));
        assert_eq!(suite.steps_per_benchmark(), 40);
    }

    #[test]
    fn test_objective_values_at_known_optima() {
        // Sphere and Quadratic have their minimum at the origin.
        assert!(
            objective_value(&TestFunctionType::Sphere, &[0.0; 5])
                .expect("sphere")
                .abs()
                < 1e-12
        );
        assert!(
            objective_value(&TestFunctionType::Quadratic, &[0.0; 5])
                .expect("quadratic")
                .abs()
                < 1e-12
        );
        // Rosenbrock's minimum is at the all-ones vector.
        assert!(
            objective_value(&TestFunctionType::Rosenbrock, &[1.0; 6])
                .expect("rosenbrock")
                .abs()
                < 1e-12
        );
        // Rastrigin's minimum is at the origin.
        assert!(
            objective_value(&TestFunctionType::Rastrigin, &[0.0; 4])
                .expect("rastrigin")
                .abs()
                < 1e-12
        );
        // Beale's minimum is at (3, 0.5).
        assert!(
            objective_value(&TestFunctionType::Beale, &[3.0, 0.5])
                .expect("beale")
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn test_gradients_match_finite_differences() {
        let point = [0.7_f64, -0.4, 1.3, 0.2];
        for function_type in [
            TestFunctionType::Sphere,
            TestFunctionType::Quadratic,
            TestFunctionType::Rosenbrock,
            TestFunctionType::Rastrigin,
            TestFunctionType::Ackley,
        ] {
            let analytic = objective_gradient(&function_type, &point).expect("gradient");
            let h = 1e-6;
            for i in 0..point.len() {
                let mut plus = point;
                let mut minus = point;
                plus[i] += h;
                minus[i] -= h;
                let numeric = (objective_value(&function_type, &plus).expect("f+")
                    - objective_value(&function_type, &minus).expect("f-"))
                    / (2.0 * h);
                assert!(
                    (analytic[i] - numeric).abs() < 1e-3,
                    "{:?} gradient[{}]: analytic {} vs numeric {}",
                    function_type,
                    i,
                    analytic[i],
                    numeric
                );
            }
        }
    }

    #[test]
    fn test_unimplemented_test_function_is_reported() {
        let err = objective_value(&TestFunctionType::NeuralNetworkTraining, &[1.0]);
        assert!(matches!(err, Err(OptimError::NotImplemented(_))));
    }

    #[test]
    fn test_benchmarks_actually_optimize() {
        let mut suite = suite();
        let results = suite
            .run_benchmarks(&architecture("Adam", &[("learning_rate", 0.1)]))
            .expect("run benchmarks");

        assert_eq!(results.len(), 4);
        for result in &results {
            // A working optimizer reduces the objective from the start point.
            assert!(
                result.normalized_score > 0.0,
                "{} made no progress",
                result.test_name
            );
            assert!(result.normalized_score <= 1.0);
            assert!(result.score.is_finite());
            assert!(result.percentile_rank > 0.0 && result.percentile_rank <= 1.0);
        }
    }

    #[test]
    fn test_different_architectures_produce_different_scores() {
        let mut suite = suite();
        let fast = suite
            .run_benchmarks(&architecture("Adam", &[("learning_rate", 0.2)]))
            .expect("fast");
        let slow = suite
            .run_benchmarks(&architecture("SGD", &[("learning_rate", 1e-6)]))
            .expect("slow");

        // The well-tuned candidate must beat the crippled one on the smooth
        // convex benchmark; a benchmark that ignored the architecture could not
        // distinguish them.
        let fast_sphere = fast
            .iter()
            .find(|r| r.test_name == "Sphere")
            .expect("sphere")
            .normalized_score;
        let slow_sphere = slow
            .iter()
            .find(|r| r.test_name == "Sphere")
            .expect("sphere")
            .normalized_score;
        assert!(
            fast_sphere > slow_sphere,
            "expected {} > {}",
            fast_sphere,
            slow_sphere
        );
    }

    #[test]
    fn test_component_type_alone_changes_the_score() {
        // The architectures produced by the live `RandomSearch` currently carry
        // an empty hyperparameter map, so the *only* signal available is the
        // component type. Scores must still differ: each family has its own
        // default learning rate and its own update rule.
        let mut suite = suite();
        let adam = suite
            .run_benchmarks(&architecture("Adam", &[]))
            .expect("adam");
        let sgd = suite
            .run_benchmarks(&architecture("SGD", &[]))
            .expect("sgd");
        let rmsprop = suite
            .run_benchmarks(&architecture("RMSprop", &[]))
            .expect("rmsprop");

        let score_of = |results: &[TestResult<f64>], name: &str| -> f64 {
            results
                .iter()
                .find(|r| r.test_name == name)
                .map(|r| r.normalized_score)
                .unwrap_or(f64::NAN)
        };

        for benchmark in ["Sphere", "Rosenbrock"] {
            let a = score_of(&adam, benchmark);
            let s = score_of(&sgd, benchmark);
            let r = score_of(&rmsprop, benchmark);
            assert!(
                (a - s).abs() > 1e-9,
                "{}: Adam and SGD must differ ({} vs {})",
                benchmark,
                a,
                s
            );
            assert!(
                (a - r).abs() > 1e-9,
                "{}: Adam and RMSprop must differ ({} vs {})",
                benchmark,
                a,
                r
            );
        }
    }

    #[test]
    fn test_unrecognised_component_falls_back_to_sgd() {
        // `ComponentType::Custom` renders as "Custom" through `format!("{:?}")`
        // and matches no family; the deliberate fallback is plain SGD so an
        // exotic candidate is still benchmarked rather than skipped.
        assert_eq!(family_from_name("Custom"), None);
        let spec = OptimizerSpec::from_architecture(&architecture("Custom", &[]));
        assert_eq!(spec.family, OptimizerFamily::Sgd);
    }

    #[test]
    fn test_identical_architecture_reproduces() {
        let mut suite_a = suite();
        let mut suite_b = suite();
        let arch = architecture("AdamW", &[("learning_rate", 0.05), ("beta1", 0.85)]);

        let a = suite_a.run_benchmarks(&arch).expect("a");
        let b = suite_b.run_benchmarks(&arch).expect("b");

        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.test_name, rb.test_name);
            assert!(
                (ra.score - rb.score).abs() < 1e-12,
                "{} not reproducible: {} vs {}",
                ra.test_name,
                ra.score,
                rb.score
            );
        }
    }

    #[test]
    fn test_optimizer_family_resolution() {
        assert_eq!(family_from_name("AdamW"), Some(OptimizerFamily::AdamW));
        assert_eq!(family_from_name("Adam"), Some(OptimizerFamily::Adam));
        assert_eq!(family_from_name("RAdam"), Some(OptimizerFamily::RAdam));
        assert_eq!(family_from_name("SGD"), Some(OptimizerFamily::Sgd));
        assert_eq!(family_from_name("RMSprop"), Some(OptimizerFamily::RmsProp));
        assert_eq!(family_from_name("Unrelated"), None);
    }
}
