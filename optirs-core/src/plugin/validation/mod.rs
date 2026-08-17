// Plugin validation and testing framework
//
// This module provides comprehensive validation and testing capabilities for optimizer plugins,
// including functionality tests, performance tests, convergence validation, and compliance checks.

#[allow(dead_code)]
use super::core::*;
use super::sdk::*;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Comprehensive plugin validation framework
#[derive(Debug)]
pub struct PluginValidationFramework<A: Float> {
    /// Validation configuration
    config: ValidationConfig,
    /// Test suites
    test_suites: Vec<Box<dyn ValidationTestSuite<A>>>,
    /// Compliance checkers
    compliance_checkers: Vec<Box<dyn ComplianceChecker>>,
    /// Performance benchmarker
    benchmarker: PerformanceBenchmarker<A>,
    /// Results storage
    results: ValidationResults<A>,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    /// Numerical tolerance
    pub numerical_tolerance: f64,
    /// Performance tolerance (percentage)
    pub performance_tolerance: f64,
    /// Maximum test duration
    pub max_test_duration: Duration,
    /// Enable memory leak detection
    pub check_memory_leaks: bool,
    /// Enable thread safety testing
    pub check_thread_safety: bool,
    /// Enable convergence testing
    pub check_convergence: bool,
    /// Random seed for reproducible tests
    pub random_seed: u64,
    /// Test data sizes
    pub test_data_sizes: Vec<usize>,
}

/// Validation test suite trait
pub trait ValidationTestSuite<A: Float>: Debug {
    /// Run all tests in the suite
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult;

    /// Get suite name
    fn name(&self) -> &str;

    /// Get suite description
    fn description(&self) -> &str;

    /// Get test count
    fn test_count(&self) -> usize;
}

/// Individual test suite result
#[derive(Debug, Clone)]
pub struct SuiteResult {
    /// Suite name
    pub suite_name: String,
    /// Test results
    pub test_results: Vec<TestResult>,
    /// Overall suite passed
    pub suite_passed: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Summary statistics
    pub summary: TestSummary,
    /// Whether this suite actually ran real checks against the plugin.
    /// `false` means the suite could not be executed (e.g. a required
    /// capability is unavailable) -- such results are excluded from the
    /// overall score and the pass gate rather than counted as a pass.
    pub verified: bool,
}

/// Test execution summary
#[derive(Debug, Clone)]
pub struct TestSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Passed tests
    pub passed_tests: usize,
    /// Failed tests
    pub failed_tests: usize,
    /// Skipped tests
    pub skipped_tests: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Compliance checker trait
pub trait ComplianceChecker: Debug {
    /// Check plugin compliance
    fn check_compliance(&self, plugininfo: &PluginInfo) -> ComplianceResult;

    /// Get checker name
    fn name(&self) -> &str;

    /// Get compliance requirements
    fn requirements(&self) -> Vec<ComplianceRequirement>;
}

/// Compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    /// Compliance check passed
    pub compliant: bool,
    /// Violations found
    pub violations: Vec<ComplianceViolation>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Compliance score (0.0 to 1.0)
    pub compliance_score: f64,
    /// Whether this checker actually inspected the plugin's declared
    /// metadata. `false` means the category is not decidable from the data
    /// this checker receives (e.g. performance conformance needs benchmark
    /// measurements, not just `PluginInfo`) -- such results are excluded
    /// from the overall score and the pass gate rather than counted as a
    /// pass.
    pub verified: bool,
}

/// Compliance violation
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Violation description
    pub description: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Types of compliance violations
#[derive(Debug, Clone)]
pub enum ViolationType {
    /// Missing required metadata
    MissingMetadata,
    /// Invalid configuration
    InvalidConfiguration,
    /// Security violation
    SecurityViolation,
    /// Performance violation
    PerformanceViolation,
    /// API violation
    ApiViolation,
    /// Documentation violation
    DocumentationViolation,
}

/// Violation severity levels
#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance requirement
#[derive(Debug, Clone)]
pub struct ComplianceRequirement {
    /// Requirement ID
    pub id: String,
    /// Requirement description
    pub description: String,
    /// Required/optional
    pub mandatory: bool,
    /// Category
    pub category: ComplianceCategory,
}

/// Compliance categories
#[derive(Debug, Clone)]
pub enum ComplianceCategory {
    Security,
    Performance,
    API,
    Documentation,
    Metadata,
    Testing,
}

/// Performance benchmarker
#[derive(Debug)]
pub struct PerformanceBenchmarker<A: Float> {
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Standard benchmarks
    benchmarks: Vec<Box<dyn PerformanceBenchmark<A>>>,
    /// Baseline results
    baselines: HashMap<String, BenchmarkBaseline>,
}

/// Performance benchmark trait
pub trait PerformanceBenchmark<A: Float>: Debug {
    /// Run benchmark
    fn run(&self, plugin: &mut dyn OptimizerPlugin<A>) -> BenchmarkResult<A>;

    /// Get benchmark name
    fn name(&self) -> &str;

    /// Get benchmark type
    fn benchmark_type(&self) -> BenchmarkType;

    /// Get expected baseline
    fn expected_baseline(&self) -> Option<BenchmarkBaseline>;
}

/// Benchmark types
#[derive(Debug, Clone)]
pub enum BenchmarkType {
    /// Throughput benchmark
    Throughput,
    /// Latency benchmark
    Latency,
    /// Memory usage benchmark
    Memory,
    /// Convergence speed benchmark
    Convergence,
    /// Scalability benchmark
    Scalability,
}

/// Benchmark baseline
#[derive(Debug, Clone)]
pub struct BenchmarkBaseline {
    /// Expected value
    pub expected_value: f64,
    /// Tolerance (percentage)
    pub tolerance: f64,
    /// Units
    pub units: String,
}

/// Complete validation results
#[derive(Debug, Clone)]
pub struct ValidationResults<A: Float> {
    /// Overall validation passed
    pub validation_passed: bool,
    /// Test suite results
    pub suite_results: Vec<SuiteResult>,
    /// Compliance results
    pub compliance_results: Vec<ComplianceResult>,
    /// Performance benchmark results
    pub benchmark_results: Vec<BenchmarkResult<A>>,
    /// Overall score (0.0 to 1.0), excluding any unverified category.
    /// `None` when every category was unverified -- there is no evidence
    /// to score, so this must never be reported as a passing number.
    pub overall_score: Option<f64>,
    /// Validation timestamp
    pub timestamp: std::time::SystemTime,
    /// Total validation time
    pub total_time: Duration,
}

// Built-in test suites

/// Functionality test suite
#[derive(Debug)]
pub struct FunctionalityTestSuite<A: Float> {
    config: ValidationConfig,
    _phantom: std::marker::PhantomData<A>,
}

/// Numerical accuracy test suite
#[derive(Debug)]
pub struct NumericalAccuracyTestSuite<A: Float> {
    config: ValidationConfig,
    _phantom: std::marker::PhantomData<A>,
}

/// Thread safety test suite
#[derive(Debug)]
pub struct ThreadSafetyTestSuite<A: Float + std::fmt::Debug> {
    config: ValidationConfig,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + std::fmt::Debug + Send + Sync> ThreadSafetyTestSuite<A> {
    /// Create a new thread safety test suite
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> ThreadSafetyTestSuite<A> {
    /// Concurrent step smoke test: clone the plugin, share it behind
    /// `Arc<Mutex<_>>` across several threads, and drive many `step()`
    /// calls concurrently. `OptimizerPlugin<A>: Send + Sync` is already a
    /// supertrait bound, so this is always runnable -- there is no
    /// "capability unavailable" case to fall back to `Unverified` for here.
    fn test_concurrent_steps(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        let start_time = Instant::now();

        const NUM_THREADS: usize = 4;
        const STEPS_PER_THREAD: usize = 25;
        // Dimension comes from the configured test data sizes rather than a
        // literal, so a caller that asked for a different workload gets it.
        let dim = self
            .config
            .test_data_sizes
            .iter()
            .copied()
            .find(|size| *size > 0)
            .unwrap_or(8);
        let dim = dim.min(4096);

        let shared: Arc<Mutex<Box<dyn OptimizerPlugin<A>>>> =
            Arc::new(Mutex::new(plugin.clone_plugin()));
        {
            let mut guard = shared.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = guard.initialize(&[dim]) {
                return TestResult {
                    passed: false,
                    message: format!("initialize failed before concurrency test: {e}"),
                    execution_time: start_time.elapsed(),
                    data: HashMap::new(),
                };
            }
        }

        let params: Array1<A> =
            Array1::from_iter((0..dim).map(|i| A::from(1.0 + i as f64).unwrap_or_else(A::one)));
        let gradients: Array1<A> =
            Array1::from_iter((0..dim).map(|_| A::from(0.01).unwrap_or_else(A::zero)));
        let saw_error = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::with_capacity(NUM_THREADS);
        for _ in 0..NUM_THREADS {
            let shared = Arc::clone(&shared);
            let saw_error = Arc::clone(&saw_error);
            let params = params.clone();
            let gradients = gradients.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..STEPS_PER_THREAD {
                    let mut guard = shared.lock().unwrap_or_else(|e| e.into_inner());
                    match guard.step(&params, &gradients) {
                        Ok(result) => {
                            if result.iter().any(|v| !v.is_finite()) {
                                saw_error.store(true, Ordering::SeqCst);
                            }
                        }
                        Err(_) => saw_error.store(true, Ordering::SeqCst),
                    }
                }
            }));
        }

        let mut any_panicked = false;
        for handle in handles {
            if handle.join().is_err() {
                any_panicked = true;
            }
        }

        let passed = !any_panicked && !saw_error.load(Ordering::SeqCst);
        let message = if any_panicked {
            "A worker thread panicked while calling step() concurrently through Arc<Mutex<_>>"
                .to_string()
        } else if passed {
            format!(
                "{NUM_THREADS} threads completed {STEPS_PER_THREAD} concurrent step() calls \
                 each through a shared Arc<Mutex<_>> instance with no panics and finite output"
            )
        } else {
            "Concurrent step() calls produced an error or a non-finite result".to_string()
        };

        TestResult {
            passed,
            message,
            execution_time: start_time.elapsed(),
            data: HashMap::new(),
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync + 'static> ValidationTestSuite<A>
    for ThreadSafetyTestSuite<A>
{
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult {
        let start_time = Instant::now();
        let result = self.test_concurrent_steps(plugin);
        let passed = result.passed;

        SuiteResult {
            suite_name: "Thread Safety".to_string(),
            test_results: vec![result],
            suite_passed: passed,
            execution_time: start_time.elapsed(),
            summary: TestSummary {
                total_tests: 1,
                passed_tests: passed as usize,
                failed_tests: (!passed) as usize,
                skipped_tests: 0,
                success_rate: if passed { 1.0 } else { 0.0 },
            },
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Thread Safety Tests"
    }

    fn description(&self) -> &str {
        "Tests for thread safety and concurrent access"
    }

    fn test_count(&self) -> usize {
        1
    }
}

/// Memory management test suite
#[derive(Debug)]
pub struct MemoryTestSuite<A: Float + std::fmt::Debug> {
    config: ValidationConfig,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + std::fmt::Debug + Send + Sync> MemoryTestSuite<A> {
    /// Create a new memory test suite
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> MemoryTestSuite<A> {
    /// Sample the plugin's self-reported `memory_usage()` across many steps
    /// and flag runs where the peak reported usage keeps climbing in the
    /// second half relative to the first -- a crude but real growth
    /// heuristic. Plugins that never override `memory_usage()` report a
    /// constant `0`, which is a real (if uninformative) measurement -- not
    /// a fabricated pass -- and the flat sequence correctly reads as "no
    /// growth observed".
    fn test_memory_growth(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();
        const SAMPLES: usize = 50;
        // Dimension from the configured workload rather than a literal.
        let dim = self
            .config
            .test_data_sizes
            .iter()
            .copied()
            .find(|size| *size > 0)
            .unwrap_or(16)
            .min(4096);

        if !self.config.check_memory_leaks {
            return TestResult {
                passed: true,
                message: "memory leak detection disabled by ValidationConfig::check_memory_leaks"
                    .to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }

        if let Err(e) = plugin.initialize(&[dim]) {
            return TestResult {
                passed: false,
                message: format!("initialize failed before memory growth probe: {e}"),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }

        let mut params: Array1<A> = Array1::from_elem(dim, A::one());
        let gradients: Array1<A> = Array1::from_elem(dim, A::from(0.01).unwrap_or_else(A::zero));
        let baseline = plugin.memory_usage().current_usage;
        let mut samples = Vec::with_capacity(SAMPLES);

        for _ in 0..SAMPLES {
            match plugin.step(&params, &gradients) {
                Ok(next) => params = next,
                Err(e) => {
                    return TestResult {
                        passed: false,
                        message: format!("step failed during memory growth probe: {e}"),
                        execution_time: start_time.elapsed(),
                        data: HashMap::new(),
                    };
                }
            }
            samples.push(plugin.memory_usage().current_usage);
        }

        let half = SAMPLES / 2;
        let first_half_peak = samples[..half].iter().copied().max().unwrap_or(0);
        let second_half_peak = samples[half..].iter().copied().max().unwrap_or(0);
        let growth_factor = if first_half_peak == 0 {
            if second_half_peak == 0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            second_half_peak as f64 / first_half_peak as f64
        };
        // Allow modest growth (e.g. lazily-allocated optimizer state
        // settling in) but flag sustained, unbounded growth across the run.
        let passed = growth_factor <= 1.5;

        TestResult {
            passed,
            message: format!(
                "self-reported current_usage over {SAMPLES} steps: baseline={baseline}B \
                 first_half_peak={first_half_peak}B second_half_peak={second_half_peak}B \
                 growth_factor={growth_factor:.2}"
            ),
            execution_time: start_time.elapsed(),
            data: HashMap::new(),
        }
    }
}

impl<A: Float + std::fmt::Debug + Send + Sync> ValidationTestSuite<A> for MemoryTestSuite<A> {
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult {
        let start_time = Instant::now();
        let result = self.test_memory_growth(plugin);
        let passed = result.passed;

        SuiteResult {
            suite_name: "Memory Management".to_string(),
            test_results: vec![result],
            suite_passed: passed,
            execution_time: start_time.elapsed(),
            summary: TestSummary {
                total_tests: 1,
                passed_tests: passed as usize,
                failed_tests: (!passed) as usize,
                skipped_tests: 0,
                success_rate: if passed { 1.0 } else { 0.0 },
            },
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Memory Management Tests"
    }

    fn description(&self) -> &str {
        "Tests for memory allocation and management"
    }

    fn test_count(&self) -> usize {
        1
    }
}

pub mod convergence;
pub use convergence::{ConvergenceTestSuite, TestProblem};

// Built-in compliance checkers

/// API compliance checker
#[derive(Debug)]
pub struct ApiComplianceChecker;

/// Security compliance checker
#[derive(Debug)]
pub struct SecurityComplianceChecker;

/// Performance compliance checker
#[derive(Debug)]
pub struct PerformanceComplianceChecker;

/// Documentation compliance checker
#[derive(Debug)]
pub struct DocumentationComplianceChecker;

// Built-in performance benchmarks

/// Throughput benchmark
#[derive(Debug)]
pub struct ThroughputBenchmark<A: Float> {
    problemsize: usize,
    iterations: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> ThroughputBenchmark<A> {
    /// Create a new throughput benchmark
    pub fn new(problemsize: usize, iterations: usize) -> Self {
        Self {
            problemsize,
            iterations,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + Debug + Send + Sync> PerformanceBenchmark<A> for ThroughputBenchmark<A> {
    fn run(&self, plugin: &mut dyn OptimizerPlugin<A>) -> BenchmarkResult<A> {
        let start_time = Instant::now();
        let dim = self.problemsize.max(1);

        if let Err(e) = plugin.initialize(&[dim]) {
            let mut metrics = HashMap::new();
            metrics.insert("error".to_string(), 0.0);
            return BenchmarkResult {
                name: format!("Throughput (initialize failed: {e})"),
                score: 0.0,
                metrics,
                execution_time: start_time.elapsed(),
                memory_usage: 0,
                data: HashMap::new(),
                verified: false,
            };
        }

        let params: Array1<A> = Array1::from_iter(
            (0..dim).map(|i| A::from(1.0 + (i % 7) as f64 * 0.1).unwrap_or_else(A::one)),
        );
        let gradients: Array1<A> = Array1::from_iter(
            (0..dim).map(|i| A::from(0.01 + (i % 5) as f64 * 0.001).unwrap_or_else(A::zero)),
        );

        let run_start = Instant::now();
        let mut current = params;
        let mut completed = 0usize;
        for _ in 0..self.iterations {
            match plugin.step(&current, &gradients) {
                Ok(next) => {
                    current = next;
                    completed += 1;
                }
                Err(_) => break,
            }
        }
        let elapsed_secs = run_start.elapsed().as_secs_f64();
        let ops_per_sec = if elapsed_secs > 0.0 {
            completed as f64 / elapsed_secs
        } else {
            completed as f64
        };

        // Normalize against the baseline into [0, 1] rather than surfacing
        // the raw ops/sec magnitude -- a magnitude in the hundreds
        // previously dominated the [0,1] overall score regardless of what
        // the functional tests found.
        let score = self
            .expected_baseline()
            .map(|baseline| {
                (ops_per_sec / baseline.expected_value.max(f64::EPSILON)).clamp(0.0, 1.0)
            })
            .unwrap_or(0.0);

        let mut metrics = HashMap::new();
        metrics.insert("ops_per_sec".to_string(), ops_per_sec);
        metrics.insert("completed_iterations".to_string(), completed as f64);

        BenchmarkResult {
            name: "Throughput".to_string(),
            score,
            metrics,
            execution_time: start_time.elapsed(),
            memory_usage: plugin.memory_usage().current_usage,
            data: HashMap::new(),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Throughput Benchmark"
    }

    fn benchmark_type(&self) -> BenchmarkType {
        BenchmarkType::Throughput
    }

    fn expected_baseline(&self) -> Option<BenchmarkBaseline> {
        Some(BenchmarkBaseline {
            expected_value: 50.0,
            tolerance: 10.0,
            units: "ops/sec".to_string(),
        })
    }
}

/// Latency benchmark
#[derive(Debug)]
pub struct LatencyBenchmark<A: Float> {
    problemsize: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> LatencyBenchmark<A> {
    /// Create a new latency benchmark
    pub fn new(problemsize: usize) -> Self {
        Self {
            problemsize,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + Debug + Send + Sync> PerformanceBenchmark<A> for LatencyBenchmark<A> {
    fn run(&self, plugin: &mut dyn OptimizerPlugin<A>) -> BenchmarkResult<A> {
        let start_time = Instant::now();
        let dim = self.problemsize.max(1);

        if let Err(e) = plugin.initialize(&[dim]) {
            return BenchmarkResult {
                name: format!("Latency (initialize failed: {e})"),
                score: 0.0,
                metrics: HashMap::new(),
                execution_time: start_time.elapsed(),
                memory_usage: 0,
                data: HashMap::new(),
                verified: false,
            };
        }

        let params: Array1<A> = Array1::from_iter(
            (0..dim).map(|i| A::from(1.0 + (i % 7) as f64 * 0.1).unwrap_or_else(A::one)),
        );
        let gradients: Array1<A> =
            Array1::from_iter((0..dim).map(|_| A::from(0.01).unwrap_or_else(A::zero)));

        const SAMPLES: usize = 50;
        let run_start = Instant::now();
        let mut current = params;
        let mut completed = 0usize;
        for _ in 0..SAMPLES {
            match plugin.step(&current, &gradients) {
                Ok(next) => {
                    current = next;
                    completed += 1;
                }
                Err(_) => break,
            }
        }
        let elapsed = run_start.elapsed();
        let avg_latency_ms = if completed > 0 {
            elapsed.as_secs_f64() * 1000.0 / completed as f64
        } else {
            f64::INFINITY
        };

        // Lower latency is better: normalize as baseline/actual, clamped to
        // [0, 1] so a plugin faster than the baseline scores 1.0 rather than
        // an unbounded value that would dominate the overall score.
        let score = self
            .expected_baseline()
            .map(|baseline| {
                if avg_latency_ms.is_finite() && avg_latency_ms > 0.0 {
                    (baseline.expected_value / avg_latency_ms).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);

        let mut metrics = HashMap::new();
        metrics.insert("avg_latency_ms".to_string(), avg_latency_ms);

        BenchmarkResult {
            name: "Latency".to_string(),
            score,
            metrics,
            execution_time: start_time.elapsed(),
            memory_usage: plugin.memory_usage().current_usage,
            data: HashMap::new(),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Latency Benchmark"
    }

    fn benchmark_type(&self) -> BenchmarkType {
        BenchmarkType::Latency
    }

    fn expected_baseline(&self) -> Option<BenchmarkBaseline> {
        Some(BenchmarkBaseline {
            expected_value: 20.0,
            tolerance: 5.0,
            units: "ms".to_string(),
        })
    }
}

/// Memory efficiency benchmark
#[derive(Debug)]
pub struct MemoryBenchmark<A: Float> {
    problemsize: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> MemoryBenchmark<A> {
    /// Create a new memory benchmark
    pub fn new(problemsize: usize) -> Self {
        Self {
            problemsize,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + Debug + Send + Sync> PerformanceBenchmark<A> for MemoryBenchmark<A> {
    fn run(&self, plugin: &mut dyn OptimizerPlugin<A>) -> BenchmarkResult<A> {
        let start_time = Instant::now();
        let dim = self.problemsize.max(1);

        if let Err(e) = plugin.initialize(&[dim]) {
            return BenchmarkResult {
                name: format!("Memory (initialize failed: {e})"),
                score: 0.0,
                metrics: HashMap::new(),
                execution_time: start_time.elapsed(),
                memory_usage: 0,
                data: HashMap::new(),
                verified: false,
            };
        }

        let params: Array1<A> = Array1::from_iter(
            (0..dim).map(|i| A::from(1.0 + (i % 7) as f64 * 0.1).unwrap_or_else(A::one)),
        );
        let gradients: Array1<A> =
            Array1::from_iter((0..dim).map(|_| A::from(0.01).unwrap_or_else(A::zero)));

        let mut current = params;
        for _ in 0..20 {
            match plugin.step(&current, &gradients) {
                Ok(next) => current = next,
                Err(_) => break,
            }
        }

        let usage = plugin.memory_usage();
        let usage_mb = usage.current_usage as f64 / (1024.0 * 1024.0);

        // Lower memory is better: normalize as baseline/actual. A plugin
        // that never overrides `memory_usage()` reports `0` -- that is a
        // real measurement of "nothing self-reported", not evidence of
        // either compliance or violation, so it scores neutrally (1.0)
        // rather than being penalized for a metric it never populated.
        let score = self
            .expected_baseline()
            .map(|baseline| {
                if usage_mb > 0.0 {
                    (baseline.expected_value / usage_mb).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .unwrap_or(0.0);

        let mut metrics = HashMap::new();
        metrics.insert("memory_usage_mb".to_string(), usage_mb);

        BenchmarkResult {
            name: "Memory".to_string(),
            score,
            metrics,
            execution_time: start_time.elapsed(),
            memory_usage: usage.current_usage,
            data: HashMap::new(),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Memory Benchmark"
    }

    fn benchmark_type(&self) -> BenchmarkType {
        BenchmarkType::Memory
    }

    fn expected_baseline(&self) -> Option<BenchmarkBaseline> {
        Some(BenchmarkBaseline {
            expected_value: 100.0,
            tolerance: 20.0,
            units: "MB".to_string(),
        })
    }
}

impl<A: Float + Debug + Send + Sync + 'static> PluginValidationFramework<A> {
    /// Create a new validation framework
    pub fn new(config: ValidationConfig) -> Self {
        let mut framework = Self {
            config: config.clone(),
            test_suites: Vec::new(),
            compliance_checkers: Vec::new(),
            benchmarker: PerformanceBenchmarker::new(BenchmarkConfig::default()),
            results: ValidationResults::new(),
        };

        // Add default test suites
        framework.add_default_test_suites();
        framework.add_default_compliance_checkers();
        framework.add_default_benchmarks();

        framework
    }

    /// Run complete validation on a plugin
    pub fn validate_plugin(&mut self, plugin: &mut dyn OptimizerPlugin<A>) -> ValidationResults<A> {
        let start_time = Instant::now();
        let mut suite_results = Vec::new();
        let mut compliance_results = Vec::new();
        let mut benchmark_results = Vec::new();

        // Run test suites
        for testsuite in &self.test_suites {
            let result = testsuite.run_tests(plugin);
            suite_results.push(result);
        }

        // Run compliance checks
        let plugininfo = plugin.plugin_info();
        for checker in &self.compliance_checkers {
            let result = checker.check_compliance(&plugininfo);
            compliance_results.push(result);
        }

        // Run performance benchmarks
        let bench_results = self.benchmarker.run_all_benchmarks(plugin);
        benchmark_results.extend(bench_results);

        // Calculate overall score, excluding any unverified category from
        // both the numerator and the weight sum -- an unrun check must never
        // read as a pass.
        let overall_score =
            self.calculate_overall_score(&suite_results, &compliance_results, &benchmark_results);

        // Determine if validation passed. `None` means nothing could be
        // verified at all -- there is no evidence to certify against, so
        // the gate cannot pass on zero evidence. Otherwise only the
        // *verified* suites/checkers must agree, mirroring the score.
        let validation_passed = match overall_score {
            Some(score) => {
                score >= 0.8 // 80% threshold
                    && suite_results
                        .iter()
                        .filter(|r| r.verified)
                        .all(|r| r.suite_passed)
                    && compliance_results
                        .iter()
                        .filter(|r| r.verified)
                        .all(|r| r.compliant)
            }
            None => false,
        };

        let results = ValidationResults {
            validation_passed,
            suite_results,
            compliance_results,
            benchmark_results,
            overall_score,
            timestamp: std::time::SystemTime::now(),
            total_time: start_time.elapsed(),
        };
        // Retain the run so a caller can re-read it without re-validating.
        // Until 0.3.2 `results` was initialised in `new` and never written or
        // read, so the framework's own record of what it had certified did not
        // exist.
        self.results = results.clone();
        results
    }

    /// The most recent [`Self::validate_plugin`] run, or a
    /// never-validated placeholder (`validation_passed: false`,
    /// `overall_score: None`) if none has happened.
    pub fn last_results(&self) -> &ValidationResults<A> {
        &self.results
    }

    /// Add custom test suite
    pub fn add_test_suite(&mut self, testsuite: Box<dyn ValidationTestSuite<A>>) {
        self.test_suites.push(testsuite);
    }

    /// Add custom compliance checker
    pub fn add_compliance_checker(&mut self, checker: Box<dyn ComplianceChecker>) {
        self.compliance_checkers.push(checker);
    }

    /// Add custom benchmark
    pub fn add_benchmark(&mut self, benchmark: Box<dyn PerformanceBenchmark<A>>) {
        self.benchmarker.add_benchmark(benchmark);
    }

    fn add_default_test_suites(&mut self) {
        self.test_suites
            .push(Box::new(FunctionalityTestSuite::new(self.config.clone())));
        self.test_suites
            .push(Box::new(NumericalAccuracyTestSuite::new(
                self.config.clone(),
            )));

        if self.config.check_thread_safety {
            self.test_suites
                .push(Box::new(ThreadSafetyTestSuite::new(self.config.clone())));
        }

        if self.config.check_memory_leaks {
            self.test_suites
                .push(Box::new(MemoryTestSuite::new(self.config.clone())));
        }

        if self.config.check_convergence {
            self.test_suites
                .push(Box::new(ConvergenceTestSuite::new(self.config.clone())));
        }
    }

    fn add_default_compliance_checkers(&mut self) {
        self.compliance_checkers
            .push(Box::new(ApiComplianceChecker));
        self.compliance_checkers
            .push(Box::new(SecurityComplianceChecker));
        self.compliance_checkers
            .push(Box::new(PerformanceComplianceChecker));
        self.compliance_checkers
            .push(Box::new(DocumentationComplianceChecker));
    }

    fn add_default_benchmarks(&mut self) {
        for &size in &self.config.test_data_sizes {
            self.benchmarker
                .add_benchmark(Box::new(ThroughputBenchmark::new(size, 100)));
            self.benchmarker
                .add_benchmark(Box::new(LatencyBenchmark::new(size)));
            self.benchmarker
                .add_benchmark(Box::new(MemoryBenchmark::new(size)));
        }
    }

    /// Aggregate suite/compliance/benchmark results into a single [0, 1]
    /// score, excluding any category that was not actually verified from
    /// both the numerator and the weight sum. An unverified category
    /// (a suite that could not run, a checker that could not decide) must
    /// never contribute as though it had passed -- and if *nothing* could
    /// be verified there is no evidence to score at all, hence `None`
    /// rather than a fabricated `0.0` that a caller might read as "checked
    /// and failed" instead of "not checked".
    fn calculate_overall_score(
        &self,
        suite_results: &[SuiteResult],
        compliance_results: &[ComplianceResult],
        benchmark_results: &[BenchmarkResult<A>],
    ) -> Option<f64> {
        let mut total_score = 0.0;
        let mut weight_sum = 0.0;

        // Test suite scores (50% weight)
        let verified_suites: Vec<&SuiteResult> =
            suite_results.iter().filter(|r| r.verified).collect();
        if !verified_suites.is_empty() {
            let suite_score = verified_suites
                .iter()
                .map(|r| r.summary.success_rate)
                .sum::<f64>()
                / verified_suites.len() as f64;
            total_score += suite_score * 0.5;
            weight_sum += 0.5;
        }

        // Compliance scores (30% weight)
        let verified_compliance: Vec<&ComplianceResult> =
            compliance_results.iter().filter(|r| r.verified).collect();
        if !verified_compliance.is_empty() {
            let compliance_score = verified_compliance
                .iter()
                .map(|r| r.compliance_score)
                .sum::<f64>()
                / verified_compliance.len() as f64;
            total_score += compliance_score * 0.3;
            weight_sum += 0.3;
        }

        // Performance scores (20% weight). Individual benchmark scores are
        // already normalized into [0, 1] against their baseline (see
        // ThroughputBenchmark/LatencyBenchmark/MemoryBenchmark::run), so
        // this average stays commensurable with the other two categories
        // instead of a raw ops/sec or MB magnitude dominating the mean.
        let verified_benchmarks: Vec<&BenchmarkResult<A>> =
            benchmark_results.iter().filter(|r| r.verified).collect();
        if !verified_benchmarks.is_empty() {
            let perf_score = verified_benchmarks.iter().map(|r| r.score).sum::<f64>()
                / verified_benchmarks.len() as f64;
            total_score += perf_score * 0.2;
            weight_sum += 0.2;
        }

        if weight_sum > 0.0 {
            Some((total_score / weight_sum).clamp(0.0, 1.0))
        } else {
            None
        }
    }
}

// Implementation of test suites

impl<A: Float + Debug + Send + Sync + 'static> FunctionalityTestSuite<A> {
    fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + Debug + Send + Sync + 'static> ValidationTestSuite<A>
    for FunctionalityTestSuite<A>
{
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult {
        let start_time = Instant::now();
        let mut test_results = Vec::new();

        // Test 1: Basic step functionality
        let result1 = self.test_basic_step(plugin);
        test_results.push(result1);

        // Test 2: Parameter initialization
        let result2 = self.test_initialization(plugin);
        test_results.push(result2);

        // Test 3: State management
        let result3 = self.test_state_management(plugin);
        test_results.push(result3);

        // Test 4: Configuration handling
        let result4 = self.test_configuration(plugin);
        test_results.push(result4);

        let passed_tests = test_results.iter().filter(|r| r.passed).count();
        let total_tests = test_results.len();

        SuiteResult {
            suite_name: self.name().to_string(),
            test_results,
            suite_passed: passed_tests == total_tests,
            execution_time: start_time.elapsed(),
            summary: TestSummary {
                total_tests,
                passed_tests,
                failed_tests: total_tests - passed_tests,
                skipped_tests: 0,
                success_rate: passed_tests as f64 / total_tests as f64,
            },
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Functionality Tests"
    }

    fn description(&self) -> &str {
        "Tests basic optimizer functionality and API compliance"
    }

    fn test_count(&self) -> usize {
        4
    }
}

impl<A: Float + Debug + Send + Sync + 'static> FunctionalityTestSuite<A> {
    fn test_basic_step(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();

        // Create test data. `A::from` cannot fail for these literals in any
        // real float type, but a plugin generic over an exotic `A` must not
        // abort the whole validation run, so a failed conversion reports a
        // failed test instead of panicking.
        let literal = |value: f64| A::from(value);
        let (Some(p0), Some(p1), Some(g0), Some(g1)) =
            (literal(1.0), literal(2.0), literal(0.1), literal(0.2))
        else {
            return TestResult {
                passed: false,
                message: "the element type cannot represent the test literals".to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        };
        let params = Array1::from_vec(vec![p0, p1]);
        let gradients = Array1::from_vec(vec![g0, g1]);

        match plugin.step(&params, &gradients) {
            Ok(result) => {
                if result.len() == params.len() {
                    // A step must actually move the parameters. "Moved" is
                    // judged against the configured `numerical_tolerance` rather
                    // than a hardcoded epsilon, which is what that setting is
                    // for.
                    let moved = result.iter().zip(params.iter()).any(|(&after, &before)| {
                        (after - before)
                            .abs()
                            .to_f64()
                            .is_some_and(|delta| delta > self.config.numerical_tolerance)
                    });
                    TestResult {
                        passed: moved,
                        message: if moved {
                            "Basic step test passed".to_string()
                        } else {
                            format!(
                                "step left every parameter within numerical_tolerance {:.3e}, so \
                                 no optimization happened",
                                self.config.numerical_tolerance
                            )
                        },
                        execution_time: start_time.elapsed(),
                        data: HashMap::new(),
                    }
                } else {
                    TestResult {
                        passed: false,
                        message: "Step result has incorrect dimensions".to_string(),
                        execution_time: start_time.elapsed(),
                        data: HashMap::new(),
                    }
                }
            }
            Err(e) => TestResult {
                passed: false,
                message: format!("Step function failed: {}", e),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
        }
    }

    fn test_initialization(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();

        match plugin.initialize(&[10, 20]) {
            Ok(()) => TestResult {
                passed: true,
                message: "Initialization test passed".to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
            Err(e) => TestResult {
                passed: false,
                message: format!("Initialization failed: {}", e),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
        }
    }

    fn test_state_management(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();

        // Test getting and setting state
        match (plugin.get_state(), plugin.reset()) {
            (Ok(_), Ok(())) => TestResult {
                passed: true,
                message: "State management test passed".to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
            (Err(e), _) => TestResult {
                passed: false,
                message: format!("Failed to get state: {}", e),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
            (_, Err(e)) => TestResult {
                passed: false,
                message: format!("Failed to reset: {}", e),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
        }
    }

    fn test_configuration(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();

        let config = plugin.get_config();
        match plugin.set_config(config) {
            Ok(()) => TestResult {
                passed: true,
                message: "Configuration test passed".to_string(),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
            Err(e) => TestResult {
                passed: false,
                message: format!("Configuration test failed: {}", e),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
        }
    }
}

// Similar implementations for other test suites would follow...

impl<A: Float + Debug + Send + Sync + 'static> NumericalAccuracyTestSuite<A> {
    fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// A config roundtrip should preserve the learning rate within
    /// `numerical_tolerance` -- catches plugins that silently lose
    /// precision (or drop fields) between `get_config`/`set_config`.
    fn test_config_roundtrip(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();
        let original = plugin.get_config();
        if let Err(e) = plugin.set_config(original.clone()) {
            return TestResult {
                passed: false,
                message: format!("set_config failed during roundtrip: {e}"),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }
        let after = plugin.get_config();
        let diff = (after.learning_rate - original.learning_rate).abs();
        let passed = diff <= self.config.numerical_tolerance;
        TestResult {
            passed,
            message: format!(
                "learning_rate roundtrip |diff|={diff:.3e} tolerance={:.3e}",
                self.config.numerical_tolerance
            ),
            execution_time: start_time.elapsed(),
            data: HashMap::new(),
        }
    }

    /// A well-conditioned finite step must not silently produce NaN/inf.
    fn test_step_output_finite(&self, plugin: &mut dyn OptimizerPlugin<A>) -> TestResult {
        let start_time = Instant::now();
        const DIM: usize = 6;

        if let Err(e) = plugin.initialize(&[DIM]) {
            return TestResult {
                passed: false,
                message: format!("initialize failed before numerical accuracy probe: {e}"),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            };
        }

        let params: Array1<A> = Array1::from_iter(
            (0..DIM).map(|i| A::from(1.0 + i as f64 * 0.1).unwrap_or_else(A::one)),
        );
        let gradients: Array1<A> = Array1::from_iter(
            (0..DIM).map(|i| A::from(0.05 - i as f64 * 0.005).unwrap_or_else(A::zero)),
        );

        match plugin.step(&params, &gradients) {
            Ok(result) => {
                let all_finite = result.iter().all(|v| v.is_finite());
                TestResult {
                    passed: all_finite,
                    message: if all_finite {
                        "step() output is finite for well-conditioned input".to_string()
                    } else {
                        "step() produced a non-finite value for finite, well-conditioned input"
                            .to_string()
                    },
                    execution_time: start_time.elapsed(),
                    data: HashMap::new(),
                }
            }
            Err(e) => TestResult {
                passed: false,
                message: format!("step() failed: {e}"),
                execution_time: start_time.elapsed(),
                data: HashMap::new(),
            },
        }
    }
}

impl<A: Float + Debug + Send + Sync + 'static> ValidationTestSuite<A>
    for NumericalAccuracyTestSuite<A>
{
    fn run_tests(&self, plugin: &mut dyn OptimizerPlugin<A>) -> SuiteResult {
        let start_time = Instant::now();
        let test_results = vec![
            self.test_config_roundtrip(plugin),
            self.test_step_output_finite(plugin),
        ];

        let passed_tests = test_results.iter().filter(|r| r.passed).count();
        let total_tests = test_results.len();

        SuiteResult {
            suite_name: self.name().to_string(),
            test_results,
            suite_passed: passed_tests == total_tests,
            execution_time: start_time.elapsed(),
            summary: TestSummary {
                total_tests,
                passed_tests,
                failed_tests: total_tests - passed_tests,
                skipped_tests: 0,
                success_rate: passed_tests as f64 / total_tests as f64,
            },
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Numerical Accuracy Tests"
    }

    fn description(&self) -> &str {
        "Tests numerical precision and accuracy of optimization steps"
    }

    fn test_count(&self) -> usize {
        2
    }
}

// Implementation placeholders for other components...

impl<A: Float + Send + Sync> PerformanceBenchmarker<A> {
    fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            benchmarks: Vec::new(),
            baselines: HashMap::new(),
        }
    }

    fn add_benchmark(&mut self, benchmark: Box<dyn PerformanceBenchmark<A>>) {
        self.benchmarks.push(benchmark);
    }

    /// Register a baseline a benchmark's `execution_time` must stay within.
    pub fn set_baseline(&mut self, benchmark_name: String, baseline: BenchmarkBaseline) {
        self.baselines.insert(benchmark_name, baseline);
    }

    /// Registered baselines, keyed by benchmark name.
    pub fn baselines(&self) -> &HashMap<String, BenchmarkBaseline> {
        &self.baselines
    }

    /// The benchmark configuration in force.
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Run every registered benchmark `config.runs` times after
    /// `config.warmup_iterations` discarded warmup runs, keeping the best score
    /// per benchmark, and check each against its registered baseline.
    ///
    /// Until 0.3.2 this ran each benchmark exactly once and ignored both
    /// `config` and `baselines` entirely: `runs`, `warmup_iterations` and every
    /// registered baseline were stored and never read, so a benchmark that blew
    /// past its declared budget still reported whatever score it computed.
    fn run_all_benchmarks(
        &mut self,
        plugin: &mut dyn OptimizerPlugin<A>,
    ) -> Vec<BenchmarkResult<A>> {
        let runs = self.config.runs.max(1);
        let warmup = self.config.warmup_iterations;
        let mut results = Vec::with_capacity(self.benchmarks.len());

        for bench in &self.benchmarks {
            for _ in 0..warmup {
                let _ = bench.run(plugin);
            }
            let mut best: Option<BenchmarkResult<A>> = None;
            for _ in 0..runs {
                let candidate = bench.run(plugin);
                best = match best {
                    Some(current) if current.score >= candidate.score => Some(current),
                    _ => Some(candidate),
                };
            }
            let Some(mut result) = best else { continue };

            if let Some(baseline) = self.baselines.get(&result.name) {
                let measured = result.execution_time.as_secs_f64();
                let ceiling = baseline.expected_value * (1.0 + baseline.tolerance / 100.0);
                let within = measured <= ceiling;
                result
                    .metrics
                    .insert("baseline_expected".to_string(), baseline.expected_value);
                result
                    .metrics
                    .insert("baseline_ceiling".to_string(), ceiling);
                result.metrics.insert(
                    "baseline_within_tolerance".to_string(),
                    if within { 1.0 } else { 0.0 },
                );
                if !within {
                    // A benchmark that misses its declared budget must not keep
                    // a passing score.
                    result.score = 0.0;
                }
            }
            results.push(result);
        }

        results
    }
}

impl<A: Float + Send + Sync> ValidationResults<A> {
    fn new() -> Self {
        Self {
            validation_passed: false,
            suite_results: Vec::new(),
            compliance_results: Vec::new(),
            benchmark_results: Vec::new(),
            overall_score: None,
            timestamp: std::time::SystemTime::now(),
            total_time: Duration::from_secs(0),
        }
    }
}

// Default implementations

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            numerical_tolerance: 1e-10,
            performance_tolerance: 20.0,
            max_test_duration: Duration::from_secs(300),
            check_memory_leaks: true,
            check_thread_safety: false,
            check_convergence: true,
            random_seed: 42,
            test_data_sizes: vec![10, 100, 1000],
        }
    }
}

// Placeholder implementations for compliance checkers

impl ComplianceChecker for ApiComplianceChecker {
    fn check_compliance(&self, plugininfo: &PluginInfo) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut score = 1.0;

        if plugininfo.name.trim().is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::ApiViolation,
                description: "Plugin name is empty".to_string(),
                severity: ViolationSeverity::Critical,
                suggested_fix: Some("Provide a non-empty plugin name".to_string()),
            });
            score -= 0.4;
        }

        if plugininfo.version.trim().is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::ApiViolation,
                description: "Plugin version is empty".to_string(),
                severity: ViolationSeverity::High,
                suggested_fix: Some("Provide a semantic version string".to_string()),
            });
            score -= 0.3;
        }

        if plugininfo.supported_types.is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::ApiViolation,
                description: "Plugin declares no supported data types".to_string(),
                severity: ViolationSeverity::Medium,
                suggested_fix: Some(
                    "Declare at least one entry in `supported_types` (e.g. DataType::F64)"
                        .to_string(),
                ),
            });
            score -= 0.2;
        }

        if plugininfo.min_sdk_version.trim().is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::ApiViolation,
                description: "Plugin declares no minimum SDK version".to_string(),
                severity: ViolationSeverity::Low,
                suggested_fix: Some(
                    "Set `min_sdk_version` to the SDK version targeted".to_string(),
                ),
            });
            score -= 0.1;
        }

        ComplianceResult {
            compliant: violations.is_empty(),
            violations,
            warnings: Vec::new(),
            compliance_score: score.max(0.0),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "API Compliance"
    }

    fn requirements(&self) -> Vec<ComplianceRequirement> {
        vec![
            ComplianceRequirement {
                id: "api-1".to_string(),
                description: "Plugin must declare a non-empty name".to_string(),
                mandatory: true,
                category: ComplianceCategory::API,
            },
            ComplianceRequirement {
                id: "api-2".to_string(),
                description: "Plugin must declare a non-empty version".to_string(),
                mandatory: true,
                category: ComplianceCategory::API,
            },
            ComplianceRequirement {
                id: "api-3".to_string(),
                description: "Plugin must declare at least one supported data type".to_string(),
                mandatory: true,
                category: ComplianceCategory::API,
            },
        ]
    }
}

impl ComplianceChecker for SecurityComplianceChecker {
    /// Inspects the declarative `PluginInfo` metadata this checker is given
    /// (license presence, dependency version bounds). Static/dynamic code
    /// inspection (unsafe blocks, filesystem/network access, signature
    /// verification) is a different trust boundary handled by
    /// `plugin::loader::SecurityManager`/`CodeScanner` at load time, which
    /// this checker does not have access to -- it must not claim to have
    /// verified what it cannot see.
    fn check_compliance(&self, plugininfo: &PluginInfo) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut score: f64 = 1.0;

        if plugininfo.license.trim().is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::SecurityViolation,
                description: "Plugin declares no license; provenance cannot be assessed"
                    .to_string(),
                severity: ViolationSeverity::Medium,
                suggested_fix: Some("Declare an SPDX license identifier".to_string()),
            });
            score -= 0.3;
        }

        for dep in &plugininfo.dependencies {
            let version_req = dep.version.trim();
            if version_req.is_empty() || version_req == "*" {
                violations.push(ComplianceViolation {
                    violation_type: ViolationType::SecurityViolation,
                    description: format!(
                        "Dependency '{}' has an unbounded version requirement ('{}'); this \
                         lets any future release -- including a compromised one -- be pulled \
                         in transparently",
                        dep.name, dep.version
                    ),
                    severity: ViolationSeverity::High,
                    suggested_fix: Some("Pin dependencies to a bounded version range".to_string()),
                });
                score -= 0.2;
            }
        }

        let compliant = !violations.iter().any(|v| {
            matches!(
                v.severity,
                ViolationSeverity::Critical | ViolationSeverity::High
            )
        });

        ComplianceResult {
            compliant,
            violations,
            warnings: vec![
                "Security compliance here covers declared metadata only; code-level scanning \
                 and signature verification happen separately in PluginLoader::SecurityManager"
                    .to_string(),
            ],
            compliance_score: score.max(0.0),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Security Compliance"
    }

    fn requirements(&self) -> Vec<ComplianceRequirement> {
        vec![
            ComplianceRequirement {
                id: "sec-1".to_string(),
                description: "Plugin should declare a license".to_string(),
                mandatory: false,
                category: ComplianceCategory::Security,
            },
            ComplianceRequirement {
                id: "sec-2".to_string(),
                description: "Dependencies must not use unbounded version requirements".to_string(),
                mandatory: true,
                category: ComplianceCategory::Security,
            },
        ]
    }
}

impl ComplianceChecker for PerformanceComplianceChecker {
    /// `check_compliance` only receives `PluginInfo` metadata -- it has no
    /// access to benchmark measurements, so performance conformance is not
    /// decidable here. `PerformanceBenchmarker` (see `ThroughputBenchmark`,
    /// `LatencyBenchmark`, `MemoryBenchmark`) already contributes real,
    /// measured performance to the overall score under its own weight, so
    /// this checker reports `Unverified` rather than a second, fabricated
    /// opinion.
    fn check_compliance(&self, _plugininfo: &PluginInfo) -> ComplianceResult {
        ComplianceResult {
            compliant: false,
            violations: Vec::new(),
            warnings: vec![
                "Performance compliance is not decidable from PluginInfo alone; see the \
                 benchmark suite (ThroughputBenchmark/LatencyBenchmark/MemoryBenchmark) instead"
                    .to_string(),
            ],
            compliance_score: 0.0,
            verified: false,
        }
    }

    fn name(&self) -> &str {
        "Performance Compliance"
    }

    fn requirements(&self) -> Vec<ComplianceRequirement> {
        vec![ComplianceRequirement {
            id: "perf-1".to_string(),
            description: "Performance must meet the declared benchmark baseline (see \
                           PerformanceBenchmarker)"
                .to_string(),
            mandatory: false,
            category: ComplianceCategory::Performance,
        }]
    }
}

impl ComplianceChecker for DocumentationComplianceChecker {
    fn check_compliance(&self, plugininfo: &PluginInfo) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut score = 1.0;

        if plugininfo.description.len() < 10 {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::DocumentationViolation,
                description: "Plugin description is too short".to_string(),
                severity: ViolationSeverity::Medium,
                suggested_fix: Some("Provide a more detailed description".to_string()),
            });
            score -= 0.2;
        }

        if plugininfo.author.is_empty() {
            violations.push(ComplianceViolation {
                violation_type: ViolationType::MissingMetadata,
                description: "Author information is missing".to_string(),
                severity: ViolationSeverity::Low,
                suggested_fix: Some("Add author information".to_string()),
            });
            score -= 0.1;
        }

        ComplianceResult {
            compliant: violations.is_empty(),
            violations,
            warnings: Vec::new(),
            compliance_score: score.max(0.0),
            verified: true,
        }
    }

    fn name(&self) -> &str {
        "Documentation Compliance"
    }

    fn requirements(&self) -> Vec<ComplianceRequirement> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(!config.strict_mode);
        assert!(config.check_memory_leaks);
        assert!(config.check_convergence);
    }

    #[test]
    fn test_validation_framework_creation() {
        let config = ValidationConfig::default();
        let framework = PluginValidationFramework::<f64>::new(config);
        assert!(!framework.test_suites.is_empty());
        assert!(!framework.compliance_checkers.is_empty());
    }

    #[test]
    fn test_documentation_compliance_checker() {
        let checker = DocumentationComplianceChecker;

        let info = PluginInfo {
            description: "Short".to_string(),
            author: "".to_string(),
            ..Default::default()
        };

        let result = checker.check_compliance(&info);
        assert!(!result.compliant);
        assert_eq!(result.violations.len(), 2);
    }
}
