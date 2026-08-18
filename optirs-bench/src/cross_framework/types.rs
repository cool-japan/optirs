//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

use super::constants::{PYTORCH_BENCHMARK_TEMPLATE, TENSORFLOW_BENCHMARK_TEMPLATE};

/// Optimizer identifier for cross-framework comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizerIdentifier {
    pub framework: Framework,
    pub name: String,
    pub version: Option<String>,
}
/// Python script templates for external framework benchmarking
pub(super) struct PythonScriptTemplates {
    /// PyTorch optimizer script template
    pub(super) pytorch_template: String,
    /// TensorFlow optimizer script template
    pub(super) tensorflow_template: String,
}
impl PythonScriptTemplates {
    pub(super) fn new() -> Self {
        Self {
            pytorch_template: PYTORCH_BENCHMARK_TEMPLATE.to_string(),
            tensorflow_template: TENSORFLOW_BENCHMARK_TEMPLATE.to_string(),
        }
    }
    /// Render a template into a runnable script.
    pub(super) fn render(
        template: &str,
        function_name: &str,
        problem_dim: usize,
        batch_size: usize,
        config: &CrossFrameworkConfig,
        optimizers: &[String],
    ) -> String {
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
    pub(super) fn generate_pytorch_script(
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
    pub(super) fn generate_tensorflow_script(
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
/// Precision options for benchmarking
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F64,
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
/// Framework identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Framework {
    SciRS2,
    PyTorch,
    TensorFlow,
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
/// Outcome of attempting to benchmark an external (Python) framework.
#[derive(Debug, Clone)]
pub enum ExternalFrameworkOutcome<A: Float> {
    /// The benchmark ran and produced results.
    Completed(HashMap<OptimizerIdentifier, OptimizerBenchmarkSummary<A>>),
    /// The benchmark could not run; no results were produced.
    Skipped(SkippedFramework),
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
/// A framework that was requested but could not be benchmarked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFramework {
    /// Framework that was skipped
    pub framework: Framework,
    /// Human-readable explanation (missing interpreter, missing package, ...)
    pub reason: String,
}
