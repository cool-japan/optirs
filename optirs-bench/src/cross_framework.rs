// Cross-framework benchmarking against PyTorch and TensorFlow optimizers
//
// This module provides comprehensive benchmarking capabilities to compare
// SciRS2 optimizers against their PyTorch and TensorFlow counterparts.

use crate::error::{OptimError, Result};
use crate::regression_tester::distributions::{
    f_distribution_sf, sample_variance, t_quantile, welch_t_test,
};
use crate::TestFunction;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::process::Command;
use std::time::{Duration, Instant};

/// Type alias for optimizer functions taking parameters and gradients
type OptimizerFn<A> = Box<dyn Fn(&Array1<A>, &Array1<A>) -> Array1<A>>;

/// Type alias for optimizer function references (trait objects)
type OptimizerFnRef<'a, A> = &'a dyn Fn(&Array1<A>, &Array1<A>) -> Array1<A>;

/// Cross-framework benchmark configuration
#[derive(Debug, Clone)]
pub struct CrossFrameworkConfig {
    /// Enable PyTorch comparison
    pub enable_pytorch: bool,
    /// Enable TensorFlow comparison
    pub enable_tensorflow: bool,
    /// Python executable path
    pub python_path: String,
    /// Temporary directory for Python scripts
    pub temp_dir: String,
    /// Benchmark precision (f32 or f64)
    pub precision: Precision,
    /// Maximum iterations per test
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Problem dimensions to test
    pub problem_dimensions: Vec<usize>,
    /// Number of runs per test for statistical significance
    pub num_runs: usize,
    /// Confidence level used for reported intervals (e.g. `0.95`)
    pub confidence_level: f64,
    /// Learning rate handed to the external framework optimizers
    pub learning_rate: f64,
    /// PyTorch optimizers to benchmark (`torch.optim` names, case-insensitive)
    pub pytorch_optimizers: Vec<String>,
    /// TensorFlow optimizers to benchmark (`tf.keras.optimizers` names)
    pub tensorflow_optimizers: Vec<String>,
}

impl CrossFrameworkConfig {
    /// Confidence level clamped into the open interval `(0, 1)`.
    ///
    /// A misconfigured level (0, 1, negative, NaN) falls back to `0.95` rather
    /// than producing an undefined quantile.
    pub fn confidence_level(&self) -> f64 {
        if self.confidence_level.is_finite()
            && self.confidence_level > 0.0
            && self.confidence_level < 1.0
        {
            self.confidence_level
        } else {
            0.95
        }
    }
}

impl Default for CrossFrameworkConfig {
    fn default() -> Self {
        Self {
            enable_pytorch: true,
            enable_tensorflow: true,
            python_path: "python3".to_string(),
            temp_dir: "/tmp/scirs2_benchmark".to_string(),
            precision: Precision::F64,
            max_iterations: 1000,
            tolerance: 1e-6,
            random_seed: 42,
            batch_sizes: vec![1, 32, 128, 512],
            problem_dimensions: vec![10, 100, 1000],
            num_runs: 5,
            confidence_level: 0.95,
            learning_rate: 0.01,
            pytorch_optimizers: vec!["Adam".to_string(), "SGD".to_string(), "RMSprop".to_string()],
            tensorflow_optimizers: vec![
                "Adam".to_string(),
                "SGD".to_string(),
                "RMSprop".to_string(),
            ],
        }
    }
}

/// Precision options for benchmarking
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F64,
}

/// Framework identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Framework {
    SciRS2,
    PyTorch,
    TensorFlow,
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Framework::SciRS2 => write!(f, "SciRS2"),
            Framework::PyTorch => write!(f, "PyTorch"),
            Framework::TensorFlow => write!(f, "TensorFlow"),
        }
    }
}

/// Optimizer identifier for cross-framework comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizerIdentifier {
    pub framework: Framework,
    pub name: String,
    pub version: Option<String>,
}

impl std::fmt::Display for OptimizerIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref version) = self.version {
            write!(f, "{}-{}-v{}", self.framework, self.name, version)
        } else {
            write!(f, "{}-{}", self.framework, self.name)
        }
    }
}

/// Comprehensive benchmark result with framework comparison
#[derive(Debug, Clone)]
pub struct CrossFrameworkBenchmarkResult<A: Float> {
    /// Test configuration
    pub config: CrossFrameworkConfig,
    /// Test function name
    pub function_name: String,
    /// Problem dimension
    pub problem_dim: usize,
    /// Batch size
    pub batch_size: usize,
    /// Results per optimizer
    pub optimizer_results: HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    /// Statistical comparison
    pub statistical_comparison: StatisticalComparison<A>,
    /// Performance ranking
    pub performance_ranking: Vec<(OptimizerIdentifier, f64)>,
    /// Resource usage comparison
    pub resource_usage: ResourceUsageComparison,
    /// External frameworks that could not be benchmarked, with the reason.
    ///
    /// A framework appears here when its Python interpreter or package is not
    /// installed. Its results are omitted entirely - never substituted with
    /// placeholder numbers.
    pub skipped_frameworks: Vec<SkippedFramework>,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// A framework that was requested but could not be benchmarked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFramework {
    /// Framework that was skipped
    pub framework: Framework,
    /// Human-readable explanation (missing interpreter, missing package, ...)
    pub reason: String,
}

/// Outcome of attempting to benchmark an external (Python) framework.
#[derive(Debug, Clone)]
pub enum ExternalFrameworkOutcome<A: Float> {
    /// The benchmark ran and produced results.
    Completed(HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>),
    /// The benchmark could not run; no results were produced.
    Skipped(SkippedFramework),
}

/// Summary statistics for an optimizer across multiple runs
#[derive(Debug, Clone)]
pub struct OptimizerBenchmarkSummary<A: Float> {
    /// Optimizer identifier
    pub optimizer: OptimizerIdentifier,
    /// Number of successful runs
    pub successful_runs: usize,
    /// Total runs attempted
    pub total_runs: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Mean convergence time
    pub mean_convergence_time: Duration,
    /// Standard deviation of convergence time
    pub std_convergence_time: Duration,
    /// Mean final function value
    pub mean_final_value: A,
    /// Standard deviation of final function value
    pub std_final_value: A,
    /// Mean iterations to convergence
    pub mean_iterations: f64,
    /// Standard deviation of iterations
    pub std_iterations: f64,
    /// Mean final gradient norm
    pub mean_gradient_norm: A,
    /// Standard deviation of gradient norm
    pub std_gradient_norm: A,
    /// Convergence curves (one per run)
    pub convergence_curves: Vec<Vec<A>>,
    /// Wall-clock time to convergence for each run, in seconds.
    ///
    /// These are the samples the pairwise convergence-time t-tests consume.
    /// Deriving them from the length of a convergence curve (as an earlier
    /// version did) compares iteration counts, not time, and for parsed
    /// external results the curves were synthetic constants - which made every
    /// t-test degenerate.
    pub run_convergence_times_secs: Vec<f64>,
    /// Final objective value achieved by each run.
    pub run_final_values: Vec<f64>,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,
}

/// Statistical comparison between optimizers
#[derive(Debug, Clone)]
pub struct StatisticalComparison<A: Float> {
    /// Pairwise t-test results for convergence time
    pub convergence_time_tests: HashMap<(OptimizerIdentifier, OptimizerIdentifier), TTestResult>,
    /// Pairwise t-test results for final function value
    pub final_value_tests: HashMap<(OptimizerIdentifier, OptimizerIdentifier), TTestResult>,
    /// ANOVA results
    pub anova_results: AnovaResult<A>,
    /// Effect sizes (Cohen's d)
    pub effect_sizes: HashMap<(OptimizerIdentifier, OptimizerIdentifier), f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<OptimizerIdentifier, ConfidenceInterval<A>>,
}

/// T-test result for pairwise comparison
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// T-statistic
    pub t_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: f64,
    /// Is statistically significant (p < 0.05)
    pub is_significant: bool,
}

/// ANOVA result for multiple group comparison
#[derive(Debug, Clone)]
pub struct AnovaResult<A: Float> {
    /// F-statistic
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Between-group sum of squares
    pub between_ss: A,
    /// Within-group sum of squares
    pub within_ss: A,
    /// Total sum of squares
    pub total_ss: A,
    /// Degrees of freedom between groups
    pub df_between: usize,
    /// Degrees of freedom within groups
    pub df_within: usize,
}

/// Confidence interval
#[derive(Debug, Clone)]
pub struct ConfidenceInterval<A: Float> {
    /// Lower bound
    pub lower: A,
    /// Upper bound
    pub upper: A,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

/// Resource usage comparison
#[derive(Debug, Clone)]
pub struct ResourceUsageComparison {
    /// Memory usage per optimizer
    pub memory_usage: HashMap<OptimizerIdentifier, MemoryStats>,
    /// CPU usage per optimizer
    pub cpu_usage: HashMap<OptimizerIdentifier, CpuStats>,
    /// GPU usage per optimizer (if applicable)
    pub gpu_usage: HashMap<OptimizerIdentifier, Option<GpuStats>>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Average memory usage (bytes)
    pub avg_memory_bytes: usize,
    /// Memory allocations count
    pub allocation_count: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// CPU usage statistics
///
/// All-zero values mean "not measured": this harness does not install CPU
/// performance counters, so it reports zeros rather than plausible-looking
/// numbers. See [`CpuStats::unmeasured`].
#[derive(Debug, Clone)]
pub struct CpuStats {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// Number of CPU cores used
    pub cores_used: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Context switches
    pub context_switches: usize,
}

impl CpuStats {
    /// A CPU statistics record explicitly marked as unmeasured.
    pub fn unmeasured() -> Self {
        Self {
            cpu_percent: 0.0,
            cores_used: 0,
            cache_misses: 0,
            context_switches: 0,
        }
    }
}

/// GPU usage statistics
#[derive(Debug, Clone)]
pub struct GpuStats {
    /// GPU utilization percentage
    pub gpu_percent: f64,
    /// GPU memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Kernel launches
    pub kernel_launches: usize,
    /// Average kernel execution time (microseconds)
    pub avg_kernel_time_us: f64,
}

/// Cross-framework benchmark suite
pub struct CrossFrameworkBenchmark<A: Float> {
    /// Configuration
    config: CrossFrameworkConfig,
    /// Test functions
    test_functions: Vec<TestFunction<A>>,
    /// Python script templates
    python_scripts: PythonScriptTemplates,
    /// Results storage
    results: Vec<CrossFrameworkBenchmarkResult<A>>,
}

/// Python script templates for external framework benchmarking
struct PythonScriptTemplates {
    /// PyTorch optimizer script template
    pytorch_template: String,
    /// TensorFlow optimizer script template
    tensorflow_template: String,
}

impl<A: Float + Debug + Send + Sync> CrossFrameworkBenchmark<A> {
    /// Create a new cross-framework benchmark suite
    pub fn new(config: CrossFrameworkConfig) -> Result<Self> {
        let python_scripts = PythonScriptTemplates::new();

        // Create temporary directory
        std::fs::create_dir_all(&config.temp_dir).map_err(|e| {
            OptimError::InvalidConfig(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            config,
            test_functions: Vec::new(),
            python_scripts,
            results: Vec::new(),
        })
    }

    /// Add a test function to the benchmark suite
    pub fn add_test_function(&mut self, test_function: TestFunction<A>) {
        self.test_functions.push(test_function);
    }

    /// Add standard optimization test functions
    pub fn add_standard_test_functions(&mut self) {
        // Quadratic function
        self.add_test_function(TestFunction {
            name: "Quadratic".to_string(),
            dimension: 10,
            function: Box::new(|x: &Array1<A>| x.mapv(|val| val * val).sum()),
            gradient: Box::new(|x: &Array1<A>| {
                x.mapv(|val| A::from(2.0).expect("unwrap failed") * val)
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::zeros(10)),
        });

        // Rosenbrock function
        self.add_test_function(TestFunction {
            name: "Rosenbrock".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let b = A::from(100.0).expect("unwrap failed");
                let term1 = (a - x[0]) * (a - x[0]);
                let term2 = b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
                term1 + term2
            }),
            gradient: Box::new(|x: &Array1<A>| {
                let a = A::one();
                let b = A::from(100.0).expect("unwrap failed");
                let grad_x = A::from(-2.0).expect("unwrap failed") * (a - x[0])
                    - A::from(4.0).expect("unwrap failed") * b * x[0] * (x[1] - x[0] * x[0]);
                let grad_y = A::from(2.0).expect("unwrap failed") * b * (x[1] - x[0] * x[0]);
                Array1::from_vec(vec![grad_x, grad_y])
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::from_vec(vec![A::one(), A::one()])),
        });

        // Beale function
        self.add_test_function(TestFunction {
            name: "Beale".to_string(),
            dimension: 2,
            function: Box::new(|x: &Array1<A>| {
                let x1 = x[0];
                let x2 = x[1];
                let term1 = (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2);
                let term2 = (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                    * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2);
                let term3 = (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                    * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2);
                term1 + term2 + term3
            }),
            gradient: Box::new(|x: &Array1<A>| {
                let x1 = x[0];
                let x2 = x[1];
                let dx1 = A::from(2.0).expect("unwrap failed")
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * (x2 - A::one())
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                        * (x2 * x2 - A::one())
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                        * (x2 * x2 * x2 - A::one());
                let dx2 = A::from(2.0).expect("unwrap failed")
                    * (A::from(1.5).expect("unwrap failed") - x1 + x1 * x2)
                    * x1
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.25).expect("unwrap failed") - x1 + x1 * x2 * x2)
                        * (A::from(2.0).expect("unwrap failed") * x1 * x2)
                    + A::from(2.0).expect("unwrap failed")
                        * (A::from(2.625).expect("unwrap failed") - x1 + x1 * x2 * x2 * x2)
                        * (A::from(3.0).expect("unwrap failed") * x1 * x2 * x2);
                Array1::from_vec(vec![dx1, dx2])
            }),
            optimal_value: Some(A::zero()),
            optimal_point: Some(Array1::from_vec(vec![
                A::from(3.0).expect("unwrap failed"),
                A::from(0.5).expect("unwrap failed"),
            ])),
        });
    }

    /// Run comprehensive cross-framework benchmark
    pub fn run_comprehensive_benchmark(
        &mut self,
        scirs2_optimizers: Vec<(String, OptimizerFn<A>)>,
    ) -> Result<Vec<CrossFrameworkBenchmarkResult<A>>> {
        let mut all_results = Vec::new();

        for test_function in &self.test_functions {
            for &problem_dim in &self.config.problem_dimensions {
                for &batch_size in &self.config.batch_sizes {
                    let result = self.run_single_benchmark(
                        test_function,
                        problem_dim,
                        batch_size,
                        &scirs2_optimizers,
                    )?;
                    all_results.push(result);
                }
            }
        }

        self.results.extend(all_results.clone());
        Ok(all_results)
    }

    /// Run benchmark for a single configuration
    fn run_single_benchmark(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
        scirs2_optimizers: &[(String, OptimizerFn<A>)],
    ) -> Result<CrossFrameworkBenchmarkResult<A>> {
        let mut optimizer_results = HashMap::new();
        let mut skipped_frameworks = Vec::new();

        // Run SciRS2 _optimizers
        for (name, optimizer) in scirs2_optimizers {
            let identifier = OptimizerIdentifier {
                framework: Framework::SciRS2,
                name: name.clone(),
                version: Some("0.1.0".to_string()),
            };

            let summary =
                self.benchmark_scirs2_optimizer(test_function, problem_dim, batch_size, optimizer)?;
            optimizer_results.insert(identifier, summary);
        }

        // Run PyTorch _optimizers
        if self.config.enable_pytorch {
            match self.benchmark_pytorch_optimizers(test_function, problem_dim, batch_size)? {
                ExternalFrameworkOutcome::Completed(results) => {
                    optimizer_results.extend(results);
                }
                ExternalFrameworkOutcome::Skipped(skip) => skipped_frameworks.push(skip),
            }
        }

        // Run TensorFlow _optimizers
        if self.config.enable_tensorflow {
            match self.benchmark_tensorflow_optimizers(test_function, problem_dim, batch_size)? {
                ExternalFrameworkOutcome::Completed(results) => {
                    optimizer_results.extend(results);
                }
                ExternalFrameworkOutcome::Skipped(skip) => skipped_frameworks.push(skip),
            }
        }

        // Perform statistical analysis
        let statistical_comparison = self.perform_statistical_analysis(&optimizer_results)?;

        // Rank _optimizers by performance
        let performance_ranking = self.rank_optimizers(&optimizer_results);

        // Analyze resource usage
        let resource_usage = self.analyze_resource_usage(&optimizer_results);

        Ok(CrossFrameworkBenchmarkResult {
            config: self.config.clone(),
            function_name: test_function.name.clone(),
            problem_dim,
            batch_size,
            optimizer_results,
            statistical_comparison,
            performance_ranking,
            resource_usage,
            skipped_frameworks,
            timestamp: std::time::Instant::now(),
        })
    }

    /// Benchmark SciRS2 optimizer
    fn benchmark_scirs2_optimizer(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        _batch_size: usize,
        optimizer: &OptimizerFn<A>,
    ) -> Result<OptimizerBenchmarkSummary<A>> {
        let mut convergence_times = Vec::new();
        let mut final_values = Vec::new();
        let mut iterations_counts = Vec::new();
        let mut gradient_norms = Vec::new();
        let mut convergence_curves = Vec::new();
        let mut successful_runs = 0;

        for run in 0..self.config.num_runs {
            // Set random seed for reproducibility
            let mut rng_seed = self.config.random_seed + run as u64;

            // Initialize parameters
            let mut x = Array1::from_vec(
                (0..problem_dim)
                    .map(|_| {
                        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                        A::from((rng_seed % 1000) as f64 / 1000.0 - 0.5).unwrap_or_else(A::zero)
                    })
                    .collect(),
            );

            let start_time = Instant::now();
            let mut convergence_curve = Vec::new();
            let mut converged = false;

            for iteration in 0..self.config.max_iterations {
                let f_val = (test_function.function)(&x);
                let grad = (test_function.gradient)(&x);
                let grad_norm = grad.mapv(|g| g * g).sum().sqrt();

                convergence_curve.push(f_val);

                // Check convergence
                if grad_norm.to_f64().unwrap_or(f64::INFINITY) < self.config.tolerance {
                    let elapsed = start_time.elapsed();
                    convergence_times.push(elapsed);
                    final_values.push(f_val);
                    iterations_counts.push(iteration as f64);
                    gradient_norms.push(grad_norm);
                    convergence_curves.push(convergence_curve.clone());
                    successful_runs += 1;
                    converged = true;
                    break;
                }

                // Perform optimization step
                x = optimizer(&x, &grad);
            }

            // If didn't converge, record final state
            if !converged {
                let elapsed = start_time.elapsed();
                let f_val = (test_function.function)(&x);
                let grad = (test_function.gradient)(&x);
                let grad_norm = grad.mapv(|g| g * g).sum().sqrt();

                convergence_times.push(elapsed);
                final_values.push(f_val);
                iterations_counts.push(self.config.max_iterations as f64);
                gradient_norms.push(grad_norm);
                convergence_curves.push(convergence_curve);
            }
        }

        // Calculate statistics
        let success_rate = successful_runs as f64 / self.config.num_runs as f64;

        let mean_convergence_time = if !convergence_times.is_empty() {
            convergence_times.iter().sum::<Duration>() / convergence_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let mean_final_value = match A::from(final_values.len()) {
            Some(count) if !final_values.is_empty() => {
                final_values.iter().fold(A::zero(), |acc, &x| acc + x) / count
            }
            _ => A::zero(),
        };

        let mean_iterations = if !iterations_counts.is_empty() {
            iterations_counts.iter().sum::<f64>() / iterations_counts.len() as f64
        } else {
            0.0
        };

        let mean_gradient_norm = match A::from(gradient_norms.len()) {
            Some(count) if !gradient_norms.is_empty() => {
                gradient_norms.iter().fold(A::zero(), |acc, &x| acc + x) / count
            }
            _ => A::zero(),
        };

        // Calculate standard deviations
        let std_convergence_time =
            self.calculate_duration_std(&convergence_times, mean_convergence_time);
        let std_final_value = self.calculate_std(&final_values, mean_final_value);
        let std_iterations = self.calculate_f64std(&iterations_counts, mean_iterations);
        let std_gradient_norm = self.calculate_std(&gradient_norms, mean_gradient_norm);

        Ok(OptimizerBenchmarkSummary {
            optimizer: OptimizerIdentifier {
                framework: Framework::SciRS2,
                name: "SciRS2".to_string(),
                version: Some("0.1.0".to_string()),
            },
            successful_runs,
            total_runs: self.config.num_runs,
            success_rate,
            mean_convergence_time,
            std_convergence_time,
            mean_final_value,
            std_final_value,
            mean_iterations,
            std_iterations,
            mean_gradient_norm,
            std_gradient_norm,
            convergence_curves,
            run_convergence_times_secs: convergence_times.iter().map(|d| d.as_secs_f64()).collect(),
            run_final_values: final_values
                .iter()
                .filter_map(|value| value.to_f64())
                .collect(),
            memory_stats: MemoryStats {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                allocation_count: 0,
                fragmentation_ratio: 0.0,
            },
            gpu_utilization: None,
        })
    }

    /// Benchmark PyTorch optimizers.
    ///
    /// Requires a Python interpreter with `torch` installed. When either is
    /// missing the run is reported as
    /// [`ExternalFrameworkOutcome::Skipped`] - never as fabricated results.
    fn benchmark_pytorch_optimizers(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        let script_path = format!("{}/pytorch_benchmark.py", self.config.temp_dir);
        let script_content = self.python_scripts.generate_pytorch_script(
            &test_function.name,
            problem_dim,
            batch_size,
            &self.config,
        );

        self.run_python_benchmark(Framework::PyTorch, &script_path, &script_content)
    }

    /// Benchmark TensorFlow optimizers.
    ///
    /// Requires a Python interpreter with `tensorflow` installed. When either
    /// is missing the run is reported as
    /// [`ExternalFrameworkOutcome::Skipped`] - never as fabricated results.
    fn benchmark_tensorflow_optimizers(
        &self,
        test_function: &TestFunction<A>,
        problem_dim: usize,
        batch_size: usize,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        let script_path = format!("{}/tensorflow_benchmark.py", self.config.temp_dir);
        let script_content = self.python_scripts.generate_tensorflow_script(
            &test_function.name,
            problem_dim,
            batch_size,
            &self.config,
        );

        self.run_python_benchmark(Framework::TensorFlow, &script_path, &script_content)
    }

    /// Write a generated script, execute it, and parse the JSON it prints.
    fn run_python_benchmark(
        &self,
        framework: Framework,
        script_path: &str,
        script_content: &str,
    ) -> Result<ExternalFrameworkOutcome<A>> {
        std::fs::write(script_path, script_content).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to write {} benchmark script to {}: {}",
                framework, script_path, e
            ))
        })?;

        let output = match Command::new(&self.config.python_path)
            .arg(script_path)
            .output()
        {
            Ok(output) => output,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    framework: framework.clone(),
                    reason: format!(
                        "Python interpreter '{}' was not found; install Python to enable this comparison",
                        self.config.python_path
                    ),
                }));
            }
            Err(e) => {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    framework: framework.clone(),
                    reason: format!("Failed to launch '{}': {}", self.config.python_path, e),
                }));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            // The generated scripts exit with code 2 and a machine-readable
            // marker when the framework package itself is unavailable.
            if stderr.contains(MISSING_DEPENDENCY_MARKER)
                || stdout.contains(MISSING_DEPENDENCY_MARKER)
            {
                return Ok(ExternalFrameworkOutcome::Skipped(SkippedFramework {
                    reason: format!(
                        "{} is not installed for interpreter '{}'",
                        framework, self.config.python_path
                    ),
                    framework,
                }));
            }
            return Err(OptimError::InvalidConfig(format!(
                "{} benchmark failed (exit status {:?}): {}",
                framework,
                output.status.code(),
                stderr.trim()
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
            OptimError::InvalidConfig(format!(
                "Failed to parse {} results as JSON: {} (stdout was: {})",
                framework,
                e,
                stdout.trim()
            ))
        })?;

        let framework_version = parsed
            .get("framework_version")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());

        let results = parsed
            .get("optimizers")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "{} results are missing the required 'optimizers' object",
                    framework
                ))
            })?;

        let mut optimizer_results = HashMap::new();
        for (optimizer_name, result_data) in results {
            let identifier = OptimizerIdentifier {
                framework: framework.clone(),
                name: optimizer_name.clone(),
                version: framework_version.clone(),
            };
            let summary = self.parse_python_results(identifier.clone(), result_data)?;
            optimizer_results.insert(identifier, summary);
        }

        Ok(ExternalFrameworkOutcome::Completed(optimizer_results))
    }

    /// Parse one optimizer's results out of the JSON produced by a generated
    /// Python benchmark script.
    ///
    /// Parsing is strict: every field the summary needs must be present and of
    /// the right type. A missing field is an error, because silently
    /// substituting "100 ms" or "0.1" for an absent measurement produces a
    /// comparison table that looks authoritative and is entirely fictional.
    fn parse_python_results(
        &self,
        identifier: OptimizerIdentifier,
        result_data: &serde_json::Value,
    ) -> Result<OptimizerBenchmarkSummary<A>> {
        let context = identifier.to_string();

        let required_f64 = |field: &str| -> Result<f64> {
            result_data
                .get(field)
                .and_then(|value| value.as_f64())
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "{}: required numeric field '{}' is missing from the Python results",
                        context, field
                    ))
                })
        };
        let required_array = |field: &str| -> Result<Vec<f64>> {
            let array = result_data
                .get(field)
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "{}: required array field '{}' is missing from the Python results",
                        context, field
                    ))
                })?;
            array
                .iter()
                .map(|value| {
                    value.as_f64().ok_or_else(|| {
                        OptimError::InvalidConfig(format!(
                            "{}: field '{}' contains a non-numeric entry",
                            context, field
                        ))
                    })
                })
                .collect()
        };

        let total_runs = required_f64("total_runs")? as usize;
        let successful_runs = required_f64("successful_runs")? as usize;
        if total_runs == 0 {
            return Err(OptimError::InvalidConfig(format!(
                "{}: 'total_runs' must be greater than zero",
                context
            )));
        }
        let success_rate = successful_runs as f64 / total_runs as f64;

        // Wall-clock convergence time is recorded per run by the script.
        let run_convergence_times_secs = required_array("convergence_times_secs")?;
        let run_final_values = required_array("final_values")?;
        let run_iterations = required_array("iterations")?;
        let run_gradient_norms = required_array("gradient_norms")?;

        if run_convergence_times_secs.len() != total_runs
            || run_final_values.len() != total_runs
            || run_iterations.len() != total_runs
            || run_gradient_norms.len() != total_runs
        {
            return Err(OptimError::InvalidConfig(format!(
                "{}: per-run arrays must all have exactly 'total_runs' = {} entries",
                context, total_runs
            )));
        }

        let convergence_curves: Vec<Vec<A>> = result_data
            .get("convergence_curves")
            .and_then(|value| value.as_array())
            .map(|curves| {
                curves
                    .iter()
                    .filter_map(|curve| curve.as_array())
                    .map(|curve| {
                        curve
                            .iter()
                            .filter_map(|value| value.as_f64())
                            .filter_map(A::from)
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mean_of = |values: &[f64]| -> f64 {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        };
        let std_of =
            |values: &[f64]| -> f64 { sample_variance(values).map(f64::sqrt).unwrap_or(0.0) };

        let mean_convergence_secs = mean_of(&run_convergence_times_secs);
        let std_convergence_secs = std_of(&run_convergence_times_secs);
        let mean_final = mean_of(&run_final_values);
        let std_final = std_of(&run_final_values);
        let mean_gradient = mean_of(&run_gradient_norms);
        let std_gradient = std_of(&run_gradient_norms);

        let memory_stats = MemoryStats {
            // Memory is optional: `0` means the script did not measure it.
            peak_memory_bytes: result_data
                .get("peak_memory_bytes")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            avg_memory_bytes: result_data
                .get("avg_memory_bytes")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            allocation_count: result_data
                .get("allocation_count")
                .and_then(|value| value.as_f64())
                .map(|value| value.max(0.0) as usize)
                .unwrap_or(0),
            fragmentation_ratio: result_data
                .get("fragmentation_ratio")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
        };

        Ok(OptimizerBenchmarkSummary {
            optimizer: identifier,
            successful_runs,
            total_runs,
            success_rate,
            mean_convergence_time: Duration::from_secs_f64(mean_convergence_secs.max(0.0)),
            std_convergence_time: Duration::from_secs_f64(std_convergence_secs.max(0.0)),
            mean_final_value: A::from(mean_final).unwrap_or_else(A::zero),
            std_final_value: A::from(std_final).unwrap_or_else(A::zero),
            mean_iterations: mean_of(&run_iterations),
            std_iterations: std_of(&run_iterations),
            mean_gradient_norm: A::from(mean_gradient).unwrap_or_else(A::zero),
            std_gradient_norm: A::from(std_gradient).unwrap_or_else(A::zero),
            convergence_curves,
            run_convergence_times_secs,
            run_final_values,
            memory_stats,
            gpu_utilization: result_data
                .get("gpu_utilization")
                .and_then(|value| value.as_f64()),
        })
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> Result<StatisticalComparison<A>> {
        let mut convergence_time_tests = HashMap::new();
        let mut final_value_tests = HashMap::new();
        let mut effect_sizes = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        // Pairwise comparisons
        let optimizers: Vec<_> = results.keys().collect();
        for i in 0..optimizers.len() {
            for j in (i + 1)..optimizers.len() {
                let opt1 = optimizers[i];
                let opt2 = optimizers[j];

                let result1 = &results[opt1];
                let result2 = &results[opt2];

                // Welch t-test on the recorded per-run wall-clock times.
                let time_test = self.perform_t_test(
                    &result1.run_convergence_times_secs,
                    &result2.run_convergence_times_secs,
                );
                convergence_time_tests.insert((opt1.clone(), opt2.clone()), time_test);

                // Welch t-test on the recorded per-run final objective values.
                let value_test =
                    self.perform_t_test(&result1.run_final_values, &result2.run_final_values);
                final_value_tests.insert((opt1.clone(), opt2.clone()), value_test);

                // Effect size (Cohen's d)
                let effect_size =
                    self.calculate_cohens_d(&result1.run_final_values, &result2.run_final_values);
                effect_sizes.insert((opt1.clone(), opt2.clone()), effect_size);
            }

            // Confidence interval for the mean final objective value.
            let result = &results[optimizers[i]];
            let ci = self.calculate_confidence_interval(
                &result.run_final_values,
                self.config.confidence_level(),
            );
            confidence_intervals.insert(optimizers[i].clone(), ci);
        }

        // ANOVA across the per-run final objective values of every optimizer.
        let all_final_values: Vec<Vec<f64>> = results
            .values()
            .map(|result| result.run_final_values.clone())
            .collect();
        let anova_results = self.perform_anova(&all_final_values);

        Ok(StatisticalComparison {
            convergence_time_tests,
            final_value_tests,
            anova_results,
            effect_sizes,
            confidence_intervals,
        })
    }

    /// Rank optimizers by performance
    fn rank_optimizers(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> Vec<(OptimizerIdentifier, f64)> {
        let mut rankings: Vec<_> = results
            .iter()
            .map(|(identifier, summary)| {
                // Composite score: success_rate * (1 / mean_final_value) * (1 / mean_convergence_time)
                let time_factor = 1.0 / (summary.mean_convergence_time.as_millis() as f64 + 1.0);
                let value_factor = 1.0 / (summary.mean_final_value.to_f64().unwrap_or(1.0) + 1e-10);
                let score = summary.success_rate * time_factor * value_factor;
                (identifier.clone(), score)
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Analyze resource usage
    fn analyze_resource_usage(
        &self,
        results: &HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>,
    ) -> ResourceUsageComparison {
        let memory_usage = results
            .iter()
            .map(|(id, summary)| (id.clone(), summary.memory_stats.clone()))
            .collect();

        // CPU counters are not instrumented by this harness; zeros here mean
        // "not measured" and are documented as such on `CpuStats`.
        let cpu_usage = results
            .iter()
            .map(|(id, _summary)| (id.clone(), CpuStats::unmeasured()))
            .collect();

        let gpu_usage = results
            .iter()
            .map(|(id, summary)| {
                let gpu_stats = summary.gpu_utilization.map(|gpu_percent| GpuStats {
                    gpu_percent,
                    memory_usage_bytes: 0,
                    kernel_launches: 0,
                    avg_kernel_time_us: 0.0,
                });
                (id.clone(), gpu_stats)
            })
            .collect();

        ResourceUsageComparison {
            memory_usage,
            cpu_usage,
            gpu_usage,
        }
    }

    /// Generate comprehensive benchmark report
    pub fn generate_comprehensive_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Cross-Framework Optimizer Benchmark Report\n\n");
        report.push_str(&format!(
            "Generated: {:?}\n\n",
            std::time::SystemTime::now()
        ));

        if self.results.is_empty() {
            report.push_str("No benchmark results available.\n");
            return report;
        }

        // Executive summary
        report.push_str("## Executive Summary\n\n");
        report.push_str(&format!(
            "Total test configurations: {}\n",
            self.results.len()
        ));

        // Framework coverage
        let frameworks: std::collections::HashSet<_> = self
            .results
            .iter()
            .flat_map(|result| result.optimizer_results.keys())
            .map(|id| &id.framework)
            .collect();
        report.push_str(&format!("Frameworks tested: {:?}\n\n", frameworks));

        // Performance rankings
        report.push_str("## Overall Performance Rankings\n\n");
        for result in &self.results {
            report.push_str(&format!(
                "### {} ({}D, batch={})\n\n",
                result.function_name, result.problem_dim, result.batch_size
            ));

            for (rank, (optimizer, score)) in result.performance_ranking.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} - Score: {:.6}\n",
                    rank + 1,
                    optimizer,
                    score
                ));
            }
            report.push('\n');
        }

        // Statistical significance
        report.push_str("## Statistical Analysis\n\n");
        for result in &self.results {
            report.push_str(&format!("### {} Results\n\n", result.function_name));

            // ANOVA results
            let anova = &result.statistical_comparison.anova_results;
            report.push_str(&format!(
                "ANOVA F-statistic: {:.4}, p-value: {:.6}\n",
                anova.f_statistic, anova.p_value
            ));

            if anova.p_value < 0.05 {
                report.push_str(
                    "**Statistically significant differences found between optimizers.**\n\n",
                );
            } else {
                report.push_str("No statistically significant differences found.\n\n");
            }
        }

        report
    }

    // Utility functions for statistical calculations

    /// Calculate standard deviation for Duration values
    fn calculate_duration_std(&self, values: &[Duration], mean: Duration) -> Duration {
        if values.len() <= 1 {
            return Duration::from_millis(0);
        }

        let variance = values
            .iter()
            .map(|&v| {
                let diff = v.as_millis() as i64 - mean.as_millis() as i64;
                (diff * diff) as f64
            })
            .sum::<f64>()
            / (values.len() - 1) as f64;

        Duration::from_millis(variance.sqrt() as u64)
    }

    /// Calculate standard deviation for Float values
    fn calculate_std(&self, values: &[A], mean: A) -> A {
        if values.len() <= 1 {
            return A::zero();
        }

        let variance = values
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(values.len() - 1).expect("unwrap failed");

        variance.sqrt()
    }

    /// Calculate standard deviation for f64 values
    fn calculate_f64std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
            / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Welch's unequal-variance t-test between two samples.
    ///
    /// Both the statistic and the p-value are exact: the p-value comes from the
    /// regularized incomplete beta representation of the Student's t
    /// distribution, so it spans the full `(0, 1]` range instead of saturating
    /// around `0.25` as the previous closed-form approximation did.
    ///
    /// Degenerate inputs (fewer than two observations in a group, or a zero
    /// pooled standard error because both groups are constant) yield
    /// `p = 1` and `is_significant = false` rather than a division by zero.
    fn perform_t_test(&self, sample1: &[f64], sample2: &[f64]) -> TTestResult {
        match welch_t_test(sample1, sample2) {
            Some(test) => TTestResult {
                t_statistic: test.t_statistic,
                p_value: test.p_value,
                degrees_of_freedom: test.degrees_of_freedom,
                is_significant: test.p_value < 1.0 - self.config.confidence_level(),
            },
            None => TTestResult {
                t_statistic: 0.0,
                p_value: 1.0,
                degrees_of_freedom: 0.0,
                is_significant: false,
            },
        }
    }

    /// Student's t cumulative distribution function.
    ///
    /// Delegates to the shared, table-checked implementation.
    fn t_distribution_cdf(&self, t: f64, df: f64) -> f64 {
        crate::regression_tester::distributions::student_t_cdf(t, df)
    }

    /// Cohen's d effect size using the pooled standard deviation.
    ///
    /// Returns `0.0` when either sample has fewer than two observations or the
    /// pooled standard deviation is zero (no scale against which to measure an
    /// effect).
    fn calculate_cohens_d(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        let (Some(var1), Some(var2)) = (sample_variance(sample1), sample_variance(sample2)) else {
            return 0.0;
        };
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        if n1 < 2.0 || n2 < 2.0 {
            return 0.0;
        }

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        // Pooled standard deviation weighted by degrees of freedom.
        let pooled_variance = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let pooled_std = pooled_variance.sqrt();
        if !pooled_std.is_finite() || pooled_std <= 0.0 {
            return 0.0;
        }

        (mean1 - mean2) / pooled_std
    }

    /// Confidence interval for the mean, honouring the requested level.
    ///
    /// The critical value is the Student's t quantile for `n - 1` degrees of
    /// freedom at the requested level - not a hardcoded `1.96`, which is only
    /// correct for a 95% interval with infinitely many samples.
    fn calculate_confidence_interval(
        &self,
        values: &[f64],
        confidence_level: f64,
    ) -> ConfidenceInterval<A> {
        let degenerate = |value: f64| ConfidenceInterval {
            lower: A::from(value).unwrap_or_else(A::zero),
            upper: A::from(value).unwrap_or_else(A::zero),
            confidence_level,
        };

        if values.is_empty() {
            return degenerate(0.0);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let Some(std_dev) = sample_variance(values).map(f64::sqrt) else {
            // A single observation supports no interval.
            return degenerate(mean);
        };
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return degenerate(mean);
        }

        let n = values.len() as f64;
        let standard_error = std_dev / n.sqrt();
        let alpha = 1.0 - confidence_level;
        let Some(critical_value) = t_quantile(1.0 - alpha / 2.0, n - 1.0) else {
            return degenerate(mean);
        };
        let margin_of_error = critical_value * standard_error;

        ConfidenceInterval {
            lower: A::from(mean - margin_of_error).unwrap_or_else(A::zero),
            upper: A::from(mean + margin_of_error).unwrap_or_else(A::zero),
            confidence_level,
        }
    }

    /// One-way ANOVA across the supplied groups.
    ///
    /// The p-value is the exact upper-tail probability of the F distribution,
    /// `P(F_{df_between, df_within} > F)`, evaluated through the regularized
    /// incomplete beta function. The previous two-valued lookup ("0.01 if
    /// F > 3 else 0.1") reported significance that had nothing to do with the
    /// degrees of freedom of the design.
    ///
    /// Groups with fewer than one observation are dropped; the analysis is
    /// undefined (and returns `p = 1`) when fewer than two groups survive or
    /// when there is no residual degrees of freedom.
    fn perform_anova(&self, groups: &[Vec<f64>]) -> AnovaResult<A> {
        let undefined = AnovaResult {
            f_statistic: 0.0,
            p_value: 1.0,
            between_ss: A::zero(),
            within_ss: A::zero(),
            total_ss: A::zero(),
            df_between: 0,
            df_within: 0,
        };

        let usable: Vec<Vec<f64>> = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect::<Vec<f64>>()
            })
            .filter(|group| !group.is_empty())
            .collect();

        if usable.len() < 2 {
            return undefined;
        }

        let total_n: usize = usable.iter().map(|g| g.len()).sum();
        if total_n <= usable.len() {
            // No within-group degrees of freedom.
            return undefined;
        }
        let grand_mean = usable.iter().flat_map(|g| g.iter()).sum::<f64>() / total_n as f64;

        let between_ss = usable
            .iter()
            .map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.len() as f64 * (group_mean - grand_mean).powi(2)
            })
            .sum::<f64>();

        let within_ss = usable
            .iter()
            .flat_map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.iter().map(move |&x| (x - group_mean).powi(2))
            })
            .sum::<f64>();

        let total_ss = between_ss + within_ss;
        let df_between = usable.len() - 1;
        let df_within = total_n - usable.len();

        let ms_between = between_ss / df_between as f64;
        let ms_within = within_ss / df_within as f64;

        // Zero within-group variance: either the groups are identical (no
        // effect) or they differ deterministically (infinite F). Both are
        // reported without dividing by zero.
        let (f_statistic, p_value) = if ms_within > 0.0 && ms_within.is_finite() {
            let f = ms_between / ms_within;
            (f, f_distribution_sf(f, df_between as f64, df_within as f64))
        } else if between_ss > 0.0 {
            (f64::INFINITY, 0.0)
        } else {
            (0.0, 1.0)
        };

        AnovaResult {
            f_statistic,
            p_value,
            between_ss: A::from(between_ss).unwrap_or_else(A::zero),
            within_ss: A::from(within_ss).unwrap_or_else(A::zero),
            total_ss: A::from(total_ss).unwrap_or_else(A::zero),
            df_between,
            df_within,
        }
    }
}

/// Marker printed by the generated Python scripts when the framework package
/// itself is missing, so a missing dependency can be distinguished from a real
/// benchmark failure.
const MISSING_DEPENDENCY_MARKER: &str = "OPTIRS_MISSING_DEPENDENCY";

impl PythonScriptTemplates {
    fn new() -> Self {
        Self {
            pytorch_template: PYTORCH_BENCHMARK_TEMPLATE.to_string(),
            tensorflow_template: TENSORFLOW_BENCHMARK_TEMPLATE.to_string(),
        }
    }

    /// Render a template into a runnable script.
    fn render(
        template: &str,
        function_name: &str,
        problem_dim: usize,
        batch_size: usize,
        config: &CrossFrameworkConfig,
        optimizers: &[String],
    ) -> String {
        // Optimizer names are rendered as a Python list literal. Quotes and
        // backslashes are escaped so a hostile name cannot break out of the
        // literal.
        let optimizer_list = format!(
            "[{}]",
            optimizers
                .iter()
                .map(|name| format!("\"{}\"", name.replace('\\', "\\\\").replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(", ")
        );

        template
            .replace(
                "{{FUNCTION_NAME}}",
                &function_name.replace('\\', "\\\\").replace('"', "\\\""),
            )
            .replace("{{PROBLEM_DIM}}", &problem_dim.to_string())
            .replace("{{BATCH_SIZE}}", &batch_size.to_string())
            .replace("{{MAX_ITERATIONS}}", &config.max_iterations.to_string())
            .replace("{{TOLERANCE}}", &format!("{:e}", config.tolerance))
            .replace("{{NUM_RUNS}}", &config.num_runs.to_string())
            .replace("{{RANDOM_SEED}}", &config.random_seed.to_string())
            .replace("{{LEARNING_RATE}}", &format!("{:e}", config.learning_rate))
            .replace("{{OPTIMIZERS}}", &optimizer_list)
    }

    fn generate_pytorch_script(
        &self,
        function_name: &str,
        problem_dim: usize,
        batch_size: usize,
        config: &CrossFrameworkConfig,
    ) -> String {
        Self::render(
            &self.pytorch_template,
            function_name,
            problem_dim,
            batch_size,
            config,
            &config.pytorch_optimizers,
        )
    }

    fn generate_tensorflow_script(
        &self,
        function_name: &str,
        problem_dim: usize,
        batch_size: usize,
        config: &CrossFrameworkConfig,
    ) -> String {
        Self::render(
            &self.tensorflow_template,
            function_name,
            problem_dim,
            batch_size,
            config,
            &config.tensorflow_optimizers,
        )
    }
}

/// Real PyTorch benchmark script (see [`PythonScriptTemplates`]).
const PYTORCH_BENCHMARK_TEMPLATE: &str = r##""""Cross-framework optimizer benchmark (PyTorch), generated by optirs-bench.

Runs each requested PyTorch optimizer on a standard optimization test function
for up to MAX_ITERATIONS steps, NUM_RUNS times, and prints one JSON document to
stdout. Nothing is fabricated: if PyTorch is unavailable the script exits with
status 2 and the marker below so the Rust side can skip the comparison instead
of inventing numbers.
"""

import json
import sys
import time

try:
    import torch
except Exception as exc:  # ImportError, or a broken installation
    sys.stderr.write("OPTIRS_MISSING_DEPENDENCY: torch: %s\n" % exc)
    sys.exit(2)

FUNCTION_NAME = "{{FUNCTION_NAME}}"
PROBLEM_DIM = {{PROBLEM_DIM}}
BATCH_SIZE = {{BATCH_SIZE}}
MAX_ITERATIONS = {{MAX_ITERATIONS}}
TOLERANCE = {{TOLERANCE}}
NUM_RUNS = {{NUM_RUNS}}
RANDOM_SEED = {{RANDOM_SEED}}
LEARNING_RATE = {{LEARNING_RATE}}
OPTIMIZERS = {{OPTIMIZERS}}


def dimension():
    if FUNCTION_NAME in ("Rosenbrock", "Beale"):
        return 2
    return PROBLEM_DIM


def objective(x):
    if FUNCTION_NAME == "Rosenbrock":
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    if FUNCTION_NAME == "Beale":
        x1 = x[0]
        x2 = x[1]
        return (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2 ** 2) ** 2
            + (2.625 - x1 + x1 * x2 ** 3) ** 2
        )
    return (x ** 2).sum()


def build_optimizer(name, params):
    lower = name.lower()
    if lower == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE)
    if lower == "adamw":
        return torch.optim.AdamW(params, lr=LEARNING_RATE)
    if lower == "sgd":
        return torch.optim.SGD(params, lr=LEARNING_RATE)
    if lower == "momentum":
        return torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
    if lower == "rmsprop":
        return torch.optim.RMSprop(params, lr=LEARNING_RATE)
    if lower == "adagrad":
        return torch.optim.Adagrad(params, lr=LEARNING_RATE)
    if lower == "adadelta":
        return torch.optim.Adadelta(params, lr=LEARNING_RATE)
    if lower == "radam":
        return torch.optim.RAdam(params, lr=LEARNING_RATE)
    if lower == "nadam":
        return torch.optim.NAdam(params, lr=LEARNING_RATE)
    raise ValueError("unsupported PyTorch optimizer: %s" % name)


def run_once(name, seed):
    torch.manual_seed(seed)
    x = torch.nn.Parameter(torch.rand(dimension(), dtype=torch.float64) - 0.5)
    optimizer = build_optimizer(name, [x])

    curve = []
    converged = False
    iterations = MAX_ITERATIONS
    gradient_norm = float("inf")

    start = time.perf_counter()
    for iteration in range(MAX_ITERATIONS):
        optimizer.zero_grad(set_to_none=False)
        value = objective(x)
        value.backward()
        gradient_norm = float(x.grad.norm().item())
        curve.append(float(value.item()))
        if gradient_norm < TOLERANCE:
            converged = True
            iterations = iteration
            break
        optimizer.step()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        final_value = float(objective(x).item())

    return {
        "elapsed": elapsed,
        "final_value": final_value,
        "iterations": float(iterations),
        "gradient_norm": gradient_norm,
        "converged": converged,
        "curve": curve,
    }


def main():
    results = {}
    for name in OPTIMIZERS:
        runs = [run_once(name, RANDOM_SEED + index) for index in range(NUM_RUNS)]
        results[name] = {
            "total_runs": len(runs),
            "successful_runs": sum(1 for run in runs if run["converged"]),
            "convergence_times_secs": [run["elapsed"] for run in runs],
            "final_values": [run["final_value"] for run in runs],
            "iterations": [run["iterations"] for run in runs],
            "gradient_norms": [run["gradient_norm"] for run in runs],
            "convergence_curves": [run["curve"] for run in runs],
        }

    json.dump(
        {
            "framework": "PyTorch",
            "framework_version": torch.__version__,
            "function": FUNCTION_NAME,
            "problem_dim": dimension(),
            "batch_size": BATCH_SIZE,
            "optimizers": results,
        },
        sys.stdout,
    )


main()
"##;

/// Real TensorFlow benchmark script (see [`PythonScriptTemplates`]).
const TENSORFLOW_BENCHMARK_TEMPLATE: &str = r##""""Cross-framework optimizer benchmark (TensorFlow), generated by optirs-bench.

Mirrors the PyTorch script: each requested Keras optimizer is run on the same
test function for up to MAX_ITERATIONS steps, NUM_RUNS times, and the results
are printed as a single JSON document. A missing TensorFlow installation exits
with status 2 and the marker below so the comparison is skipped, not faked.
"""

import json
import sys
import time

try:
    import tensorflow as tf
except Exception as exc:  # ImportError, or a broken installation
    sys.stderr.write("OPTIRS_MISSING_DEPENDENCY: tensorflow: %s\n" % exc)
    sys.exit(2)

FUNCTION_NAME = "{{FUNCTION_NAME}}"
PROBLEM_DIM = {{PROBLEM_DIM}}
BATCH_SIZE = {{BATCH_SIZE}}
MAX_ITERATIONS = {{MAX_ITERATIONS}}
TOLERANCE = {{TOLERANCE}}
NUM_RUNS = {{NUM_RUNS}}
RANDOM_SEED = {{RANDOM_SEED}}
LEARNING_RATE = {{LEARNING_RATE}}
OPTIMIZERS = {{OPTIMIZERS}}


def dimension():
    if FUNCTION_NAME in ("Rosenbrock", "Beale"):
        return 2
    return PROBLEM_DIM


def objective(x):
    if FUNCTION_NAME == "Rosenbrock":
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    if FUNCTION_NAME == "Beale":
        x1 = x[0]
        x2 = x[1]
        return (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2 ** 2) ** 2
            + (2.625 - x1 + x1 * x2 ** 3) ** 2
        )
    return tf.reduce_sum(x * x)


def build_optimizer(name):
    lower = name.lower()
    optimizers = tf.keras.optimizers
    if lower == "adam":
        return optimizers.Adam(learning_rate=LEARNING_RATE)
    if lower == "adamw":
        return optimizers.AdamW(learning_rate=LEARNING_RATE)
    if lower == "sgd":
        return optimizers.SGD(learning_rate=LEARNING_RATE)
    if lower == "momentum":
        return optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    if lower == "rmsprop":
        return optimizers.RMSprop(learning_rate=LEARNING_RATE)
    if lower == "adagrad":
        return optimizers.Adagrad(learning_rate=LEARNING_RATE)
    if lower == "adadelta":
        return optimizers.Adadelta(learning_rate=LEARNING_RATE)
    if lower == "nadam":
        return optimizers.Nadam(learning_rate=LEARNING_RATE)
    raise ValueError("unsupported TensorFlow optimizer: %s" % name)


def run_once(name, seed):
    tf.random.set_seed(seed)
    initial = tf.random.uniform([dimension()], dtype=tf.float64) - 0.5
    x = tf.Variable(initial, dtype=tf.float64)
    optimizer = build_optimizer(name)

    curve = []
    converged = False
    iterations = MAX_ITERATIONS
    gradient_norm = float("inf")

    start = time.perf_counter()
    for iteration in range(MAX_ITERATIONS):
        with tf.GradientTape() as tape:
            value = objective(x)
        gradient = tape.gradient(value, x)
        gradient_norm = float(tf.norm(gradient).numpy())
        curve.append(float(value.numpy()))
        if gradient_norm < TOLERANCE:
            converged = True
            iterations = iteration
            break
        optimizer.apply_gradients([(gradient, x)])
    elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "final_value": float(objective(x).numpy()),
        "iterations": float(iterations),
        "gradient_norm": gradient_norm,
        "converged": converged,
        "curve": curve,
    }


def main():
    results = {}
    for name in OPTIMIZERS:
        runs = [run_once(name, RANDOM_SEED + index) for index in range(NUM_RUNS)]
        results[name] = {
            "total_runs": len(runs),
            "successful_runs": sum(1 for run in runs if run["converged"]),
            "convergence_times_secs": [run["elapsed"] for run in runs],
            "final_values": [run["final_value"] for run in runs],
            "iterations": [run["iterations"] for run in runs],
            "gradient_norms": [run["gradient_norm"] for run in runs],
            "convergence_curves": [run["curve"] for run in runs],
        }

    json.dump(
        {
            "framework": "TensorFlow",
            "framework_version": tf.__version__,
            "function": FUNCTION_NAME,
            "problem_dim": dimension(),
            "batch_size": BATCH_SIZE,
            "optimizers": results,
        },
        sys.stdout,
    )


main()
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_framework_config() {
        let config = CrossFrameworkConfig::default();
        assert!(config.enable_pytorch);
        assert!(config.enable_tensorflow);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_optimizer_identifier() {
        let id = OptimizerIdentifier {
            framework: Framework::SciRS2,
            name: "Adam".to_string(),
            version: Some("0.1.0".to_string()),
        };
        assert_eq!(id.to_string(), "SciRS2-Adam-v0.1.0");
    }

    #[test]
    fn test_precision_enum() {
        let precision = Precision::F64;
        assert!(matches!(precision, Precision::F64));
    }

    #[test]
    fn test_framework_display() {
        assert_eq!(Framework::SciRS2.to_string(), "SciRS2");
        assert_eq!(Framework::PyTorch.to_string(), "PyTorch");
        assert_eq!(Framework::TensorFlow.to_string(), "TensorFlow");
    }

    #[test]
    fn test_python_script_generation() {
        let templates = PythonScriptTemplates::new();
        let config = CrossFrameworkConfig::default();

        let script = templates.generate_pytorch_script("Quadratic", 10, 32, &config);
        assert!(script.contains("PROBLEM_DIM = 10"));
        assert!(script.contains("BATCH_SIZE = 32"));
        assert!(script.contains("MAX_ITERATIONS = 1000"));
        // The script must actually construct optimizers and emit JSON, not
        // just print a banner.
        assert!(script.contains("torch.optim.Adam"));
        assert!(script.contains("value.backward()"));
        assert!(script.contains("optimizer.step()"));
        assert!(script.contains("json.dump"));
        assert!(script.contains("convergence_times_secs"));
        assert!(script.contains(MISSING_DEPENDENCY_MARKER));
        assert!(script.contains("OPTIMIZERS = [\"Adam\", \"SGD\", \"RMSprop\"]"));

        let tf_script = templates.generate_tensorflow_script("Rosenbrock", 2, 1, &config);
        assert!(tf_script.contains("tf.GradientTape"));
        assert!(tf_script.contains("apply_gradients"));
        assert!(tf_script.contains("optimizers.Adam"));
        assert!(tf_script.contains("json.dump"));
    }

    fn test_benchmark() -> CrossFrameworkBenchmark<f64> {
        let mut config = CrossFrameworkConfig::default();
        config.temp_dir = std::env::temp_dir()
            .join("optirs_cross_framework_tests")
            .to_string_lossy()
            .into_owned();
        CrossFrameworkBenchmark::<f64>::new(config).expect("temp dir is creatable")
    }

    /// F47: the previous t-distribution CDF saturated, so no comparison could
    /// ever be significant. Two clearly separated samples must now be.
    #[test]
    fn t_test_reports_real_p_values() {
        let benchmark = test_benchmark();

        let fast = [0.100, 0.102, 0.098, 0.101, 0.099, 0.100];
        let slow = [0.200, 0.202, 0.198, 0.201, 0.199, 0.200];
        let separated = benchmark.perform_t_test(&fast, &slow);
        assert!(separated.p_value < 1e-6, "p = {}", separated.p_value);
        assert!(separated.is_significant);
        assert!(separated.degrees_of_freedom > 0.0);

        let overlapping = benchmark.perform_t_test(&fast, &fast);
        assert!(overlapping.p_value > 0.5);
        assert!(!overlapping.is_significant);

        // Degenerate inputs must not divide by a zero pooled standard error.
        let degenerate = benchmark.perform_t_test(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_eq!(degenerate.p_value, 1.0);
        assert!(!degenerate.is_significant);
        assert!(!benchmark.perform_t_test(&[], &[1.0, 2.0]).is_significant);
    }

    #[test]
    fn t_distribution_cdf_matches_tables() {
        let benchmark = test_benchmark();
        // df = 10: P(T <= 3) = 0.993328
        assert!((benchmark.t_distribution_cdf(3.0, 10.0) - 0.993_328).abs() < 1e-5);
        assert!((benchmark.t_distribution_cdf(0.0, 10.0) - 0.5).abs() < 1e-12);
    }

    /// F48: the ANOVA p-value must come from the F distribution.
    #[test]
    fn anova_uses_the_f_distribution() {
        let benchmark = test_benchmark();

        let separated = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![5.0, 5.1, 4.9, 5.05, 4.95],
            vec![9.0, 9.1, 8.9, 9.05, 8.95],
        ];
        let result = benchmark.perform_anova(&separated);
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 1e-9, "p = {}", result.p_value);
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);

        let identical = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
        ];
        let flat = benchmark.perform_anova(&identical);
        assert!(flat.p_value > 0.9, "p = {}", flat.p_value);

        // Guards.
        assert_eq!(benchmark.perform_anova(&[]).p_value, 1.0);
        assert_eq!(benchmark.perform_anova(&[vec![1.0, 2.0]]).p_value, 1.0);
        assert_eq!(
            benchmark.perform_anova(&[vec![1.0], vec![2.0]]).p_value,
            1.0
        );
    }

    /// F49: the critical value must follow the requested confidence level.
    #[test]
    fn confidence_intervals_honour_the_confidence_level() {
        let benchmark = test_benchmark();
        let values = [10.0, 12.0, 9.0, 11.0, 13.0, 8.0, 10.5, 11.5];

        let ci95 = benchmark.calculate_confidence_interval(&values, 0.95);
        let ci99 = benchmark.calculate_confidence_interval(&values, 0.99);
        assert_eq!(ci95.confidence_level, 0.95);
        assert!(ci99.upper - ci99.lower > ci95.upper - ci95.lower);
        assert!(ci95.lower < 10.6 && ci95.upper > 10.6);

        // With n = 8 the 95% t critical value is 2.365, clearly wider than the
        // hardcoded 1.96 the previous implementation always used.
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let half_width = ci95.upper - mean;
        let normal_half_width = 1.96
            * (1.767_766_952_966_369 / (values.len() as f64).sqrt() * 0.0 + {
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                variance.sqrt() / (values.len() as f64).sqrt()
            });
        assert!(half_width > normal_half_width);

        // Degenerate inputs.
        let empty = benchmark.calculate_confidence_interval(&[], 0.95);
        assert_eq!(empty.lower, 0.0);
        assert_eq!(empty.upper, 0.0);
        let single = benchmark.calculate_confidence_interval(&[4.0], 0.95);
        assert_eq!(single.lower, 4.0);
        assert_eq!(single.upper, 4.0);
    }

    #[test]
    fn cohens_d_guards_degenerate_samples() {
        let benchmark = test_benchmark();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert!((benchmark.calculate_cohens_d(&a, &b) + 3.098_386_676_965_933).abs() < 1e-9);

        assert_eq!(benchmark.calculate_cohens_d(&[1.0], &b), 0.0);
        assert_eq!(benchmark.calculate_cohens_d(&[], &b), 0.0);
        assert_eq!(
            benchmark.calculate_cohens_d(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]),
            0.0
        );
    }

    /// F45: parsing must be strict - a missing field is an error, never a
    /// silently substituted default.
    #[test]
    fn python_result_parsing_is_strict() {
        let benchmark = test_benchmark();
        let identifier = OptimizerIdentifier {
            framework: Framework::PyTorch,
            name: "Adam".to_string(),
            version: Some("2.3.0".to_string()),
        };

        let complete = serde_json::json!({
            "total_runs": 3,
            "successful_runs": 2,
            "convergence_times_secs": [0.10, 0.12, 0.11],
            "final_values": [1e-8, 2e-8, 1.5e-8],
            "iterations": [120.0, 140.0, 130.0],
            "gradient_norms": [1e-7, 2e-7, 1.5e-7],
        });
        let summary = benchmark
            .parse_python_results(identifier.clone(), &complete)
            .expect("complete results parse");
        assert_eq!(summary.total_runs, 3);
        assert_eq!(summary.successful_runs, 2);
        assert!((summary.success_rate - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(summary.run_convergence_times_secs.len(), 3);
        assert!((summary.mean_iterations - 130.0).abs() < 1e-9);
        // Memory is optional and defaults to "not measured" (zero), never 1 MB.
        assert_eq!(summary.memory_stats.peak_memory_bytes, 0);
        assert!(summary.gpu_utilization.is_none());

        // Every required field must be present.
        for field in [
            "total_runs",
            "successful_runs",
            "convergence_times_secs",
            "final_values",
            "iterations",
            "gradient_norms",
        ] {
            let mut partial = complete.clone();
            if let Some(object) = partial.as_object_mut() {
                object.remove(field);
            }
            assert!(
                benchmark
                    .parse_python_results(identifier.clone(), &partial)
                    .is_err(),
                "missing '{}' must be an error",
                field
            );
        }

        // Length mismatches are rejected too.
        let mut mismatched = complete.clone();
        if let Some(object) = mismatched.as_object_mut() {
            object.insert("iterations".to_string(), serde_json::json!([1.0]));
        }
        assert!(benchmark
            .parse_python_results(identifier, &mismatched)
            .is_err());
    }

    /// F43/F44: a missing interpreter must skip the comparison explicitly.
    #[test]
    fn missing_python_interpreter_is_skipped_not_faked() {
        let mut config = CrossFrameworkConfig::default();
        config.temp_dir = std::env::temp_dir()
            .join("optirs_cross_framework_missing_python")
            .to_string_lossy()
            .into_owned();
        config.python_path = "optirs-definitely-not-a-real-python-interpreter".to_string();
        let benchmark = CrossFrameworkBenchmark::<f64>::new(config).expect("temp dir is creatable");

        let mut functions = CrossFrameworkBenchmark::<f64>::new(CrossFrameworkConfig {
            temp_dir: std::env::temp_dir()
                .join("optirs_cross_framework_missing_python_fn")
                .to_string_lossy()
                .into_owned(),
            ..Default::default()
        })
        .expect("temp dir is creatable");
        functions.add_standard_test_functions();
        let test_function = functions
            .test_functions
            .first()
            .expect("standard functions were added");

        let outcome = benchmark
            .benchmark_pytorch_optimizers(test_function, 10, 1)
            .expect("a missing interpreter is not a hard error");
        match outcome {
            ExternalFrameworkOutcome::Skipped(skip) => {
                assert_eq!(skip.framework, Framework::PyTorch);
                assert!(skip.reason.contains("not found"), "reason: {}", skip.reason);
            }
            ExternalFrameworkOutcome::Completed(results) => {
                panic!("expected a skip, got {} results", results.len())
            }
        }
    }
}
