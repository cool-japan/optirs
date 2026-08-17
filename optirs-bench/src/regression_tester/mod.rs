// Performance regression testing framework
//
// This module provides comprehensive performance regression detection capabilities
// including baseline establishment, historical tracking, statistical analysis,
// and automated CI/CD integration for continuous performance monitoring.

// Import all submodules
pub mod alerts;
pub mod config;
pub mod database;
pub mod detectors;
pub mod distributions;
pub mod statistics;
pub mod types;

// Re-export commonly used types and functions
pub use alerts::{AlertStatistics, AlertSystem, SeverityCounts};
pub use config::{
    Alert, AlertConfig, AlertSeverity, AlertStatus, CiReportFormat, EmailAlertConfig,
    GitHubAlertConfig, RegressionConfig, SlackAlertConfig, TestEnvironment,
};
pub use database::PerformanceDatabase;
pub use detectors::{ChangePointDetector, SlidingWindowDetector, StatisticalTestDetector};
pub use statistics::{OutlierAnalyzer, TrendAnalyzer};
pub use types::{
    BaselineStatistics, ChangePointAnalysis, ConfidenceIntervals, ConvergenceMetrics,
    ConvergenceStatistics, DatabaseMetadata, EfficiencyMetrics, EfficiencyStatistics,
    FragmentationStatistics, MemoryMetrics, MemoryStatistics, OutlierAnalysis, OutlierType,
    PerformanceBaseline, PerformanceMetrics, PerformanceRecord, RegressionAnalysis,
    RegressionDetector, RegressionResult, StatisticalAnalysisResult, StatisticalAnalyzer,
    StatisticalTestResult, TimingMetrics, TimingStatistics, TrendAnalysis, TrendDirection,
};

// For now, we'll include a simplified main framework here
// In a full refactoring, this would be split into core.rs and ci_integration.rs

use crate::error::Result;
use crate::regression_tester::distributions::t_quantile;
use crate::BenchmarkResult;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::time::Duration;

/// Resource measurements captured while a benchmark ran.
///
/// This is a plain data carrier so that a system sampler (or any other
/// instrumentation) can populate it without the regression tester depending on
/// the sampler: fill the fields you measured and leave the struct out entirely
/// when nothing was measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceMeasurements {
    /// Peak resident memory observed, in bytes.
    pub peak_memory_bytes: usize,
    /// Mean resident memory observed, in bytes.
    pub avg_memory_bytes: usize,
    /// Number of allocations observed, when an allocator hook was installed.
    pub allocation_count: usize,
    /// Fragmentation ratio in `0.0..=1.0`, when the allocator reports it.
    pub fragmentation_ratio: f64,
    /// Memory efficiency score in `0.0..=1.0`, when the allocator reports it.
    pub efficiency_score: f64,
}

impl ResourceMeasurements {
    /// Construct a measurement carrying only peak and average memory.
    pub fn from_memory(peak_memory_bytes: usize, avg_memory_bytes: usize) -> Self {
        Self {
            peak_memory_bytes,
            avg_memory_bytes,
            allocation_count: 0,
            fragmentation_ratio: 0.0,
            efficiency_score: 0.0,
        }
    }
}

/// Optional measurements supplied alongside a [`BenchmarkResult`].
///
/// `BenchmarkResult` records a single wall-clock duration and no memory usage.
/// Anything richer has to be measured by the caller and handed over here;
/// omitting a field means "not measured", and the regression tester reports it
/// as such instead of inventing a value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Measurements {
    /// Per-iteration wall-clock timings, in benchmark order.
    pub iteration_times: Vec<Duration>,
    /// Resource usage measured during the run, when instrumentation was active.
    pub resources: Option<ResourceMeasurements>,
}

impl Measurements {
    /// Build measurements from per-iteration timings only.
    pub fn from_iteration_times(iteration_times: Vec<Duration>) -> Self {
        Self {
            iteration_times,
            resources: None,
        }
    }
}

/// Derive full timing statistics from per-iteration samples (in nanoseconds).
///
/// Returns `None` when no finite sample is present.
fn timing_metrics_from_samples(samples_ns: &[f64]) -> Option<TimingMetrics> {
    let sorted = distributions::sorted_finite(samples_ns);
    if sorted.is_empty() {
        return None;
    }

    let mean = distributions::mean(&sorted)?;
    // A single sample has no spread; `sample_std_dev` correctly returns `None`.
    let std_dev = distributions::sample_std_dev(&sorted).unwrap_or(0.0);
    let median = distributions::median_sorted(&sorted)?;
    let p95 = distributions::percentile_sorted(&sorted, 95.0)?;
    let p99 = distributions::percentile_sorted(&sorted, 99.0)?;
    let minimum = *sorted.first()?;
    let maximum = *sorted.last()?;

    let to_ns = |value: f64| -> u64 {
        if value.is_finite() && value >= 0.0 {
            value.round() as u64
        } else {
            0
        }
    };

    Some(TimingMetrics {
        mean_time_ns: to_ns(mean),
        std_time_ns: to_ns(std_dev),
        median_time_ns: to_ns(median),
        p95_time_ns: to_ns(p95),
        p99_time_ns: to_ns(p99),
        min_time_ns: to_ns(minimum),
        max_time_ns: to_ns(maximum),
    })
}

/// Estimate a linear convergence rate from an objective-value history.
///
/// The rate is the geometric mean per-iteration contraction factor of the
/// objective, `(f_last / f_first)^(1/(n-1))`, defined only for a strictly
/// positive, shrinking objective. Anything else yields `0.0` (unknown) rather
/// than a fabricated constant.
fn convergence_rate_from_history<A: Float>(history: &[A]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let (Some(first), Some(last)) = (
        history.first().and_then(|v| v.to_f64()),
        history.last().and_then(|v| v.to_f64()),
    ) else {
        return 0.0;
    };
    if !(first.is_finite() && last.is_finite()) || first <= 0.0 || last <= 0.0 {
        return 0.0;
    }
    let exponent = 1.0 / (history.len() - 1) as f64;
    let rate = (last / first).powf(exponent);
    if rate.is_finite() {
        rate.clamp(0.0, f64::MAX)
    } else {
        0.0
    }
}

/// Two-sided confidence interval for a mean, using the Student's t quantile
/// for `n - 1` degrees of freedom.
///
/// Falls back to a degenerate `(mean, mean)` interval when there is no usable
/// spread (fewer than two samples, or a zero/non-finite standard deviation),
/// which is the honest answer: a single observation supports no interval.
fn confidence_interval_of_mean(
    mean: f64,
    std_dev: f64,
    sample_count: usize,
    confidence_level: f64,
) -> (f64, f64) {
    if sample_count < 2 || !std_dev.is_finite() || std_dev <= 0.0 || !mean.is_finite() {
        return (mean, mean);
    }
    let n = sample_count as f64;
    let standard_error = std_dev / n.sqrt();
    let alpha = 1.0 - confidence_level;
    let Some(critical) = t_quantile(1.0 - alpha / 2.0, n - 1.0) else {
        return (mean, mean);
    };
    let margin = critical * standard_error;
    (mean - margin, mean + margin)
}

/// Comprehensive performance regression testing framework
#[derive(Debug)]
pub struct RegressionTester<A: Float> {
    /// Configuration for regression testing
    config: RegressionConfig,
    /// Historical performance database
    performance_db: PerformanceDatabase<A>,
    /// Baseline performance metrics
    baselines: HashMap<String, PerformanceBaseline<A>>,
    /// Regression detection algorithms
    detectors: Vec<Box<dyn RegressionDetector<A>>>,
    /// Statistical analyzers
    analyzers: Vec<Box<dyn StatisticalAnalyzer<A>>>,
    /// Alert system
    alert_system: AlertSystem,
}

/// Regression test result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult<A: Float> {
    /// Test identifier
    pub test_id: String,
    /// Test execution status
    pub status: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of regressions detected
    pub regression_count: usize,
    /// Detected regressions
    pub regressions: Vec<RegressionResult<A>>,
    /// Performance metrics
    pub metrics: PerformanceMetrics<A>,
    /// Baseline comparison results
    pub baseline_comparison: Option<BaselineComparison>,
}

/// Baseline comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Performance change percentage
    pub performance_change_percent: f64,
    /// Memory change percentage
    pub memory_change_percent: f64,
    /// Efficiency change percentage
    pub efficiency_change_percent: f64,
    /// Comparison status
    pub status: String,
}

/// CI report structure
#[derive(Debug, Serialize)]
pub struct CiReport {
    /// Report timestamp
    pub timestamp: u64,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Individual test results
    pub test_results: Vec<CiTestResult>,
}

/// Individual test result for CI
#[derive(Debug, Serialize)]
pub struct CiTestResult {
    /// Test name
    pub name: String,
    /// Test status
    pub status: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of regressions detected
    pub regression_count: usize,
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de> + Send + Sync> RegressionTester<A> {
    /// Create a new regression tester
    pub fn new(config: RegressionConfig) -> Result<Self> {
        // Ensure baseline directory exists
        fs::create_dir_all(&config.baseline_dir)?;

        let performance_db = PerformanceDatabase::load(&config.baseline_dir)
            .unwrap_or_else(|_| PerformanceDatabase::new());

        let mut tester = Self {
            config: config.clone(),
            performance_db,
            baselines: HashMap::new(),
            detectors: Vec::new(),
            analyzers: Vec::new(),
            alert_system: AlertSystem::new(),
        };

        // Initialize default detectors and analyzers
        tester.initialize_default_components()?;

        // Load existing baselines
        tester.load_baselines()?;

        Ok(tester)
    }

    /// Initialize default regression detectors and analyzers
    fn initialize_default_components(&mut self) -> Result<()> {
        // Add statistical test detector
        if self
            .config
            .detection_algorithms
            .contains(&"statistical_test".to_string())
        {
            self.detectors
                .push(Box::new(StatisticalTestDetector::new()));
        }

        // Add sliding window detector
        if self
            .config
            .detection_algorithms
            .contains(&"sliding_window".to_string())
        {
            self.detectors.push(Box::new(SlidingWindowDetector::new()));
        }

        // Add change point detector
        if self
            .config
            .detection_algorithms
            .contains(&"change_point".to_string())
        {
            self.detectors.push(Box::new(ChangePointDetector::new()));
        }

        // Add default statistical analyzers
        self.analyzers.push(Box::new(TrendAnalyzer::new()));
        self.analyzers.push(Box::new(OutlierAnalyzer::new()));

        Ok(())
    }

    /// Load existing baselines from disk
    fn load_baselines(&mut self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        if baseline_path.exists() {
            let data = fs::read_to_string(&baseline_path)?;
            self.baselines = serde_json::from_str(&data)?;
        }
        Ok(())
    }

    /// Save baselines to disk
    fn save_baselines(&self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        let data = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(&baseline_path, data)?;
        Ok(())
    }

    /// Run regression test on a benchmark result.
    ///
    /// `BenchmarkResult` only carries a single wall-clock duration, so the
    /// derived timing distribution collapses to that one observation. Use
    /// [`RegressionTester::run_regression_test_with_measurements`] to supply
    /// per-iteration timings (and measured memory) and obtain real percentiles.
    pub fn run_regression_test(
        &mut self,
        key: &str,
        result: &BenchmarkResult<A>,
    ) -> Result<RegressionTestResult<A>> {
        self.run_regression_test_with_measurements(key, result, &Measurements::default())
    }

    /// Run regression test on a benchmark result using externally collected
    /// per-iteration timings and resource measurements.
    pub fn run_regression_test_with_measurements(
        &mut self,
        key: &str,
        result: &BenchmarkResult<A>,
        measurements: &Measurements,
    ) -> Result<RegressionTestResult<A>> {
        let start_time = std::time::Instant::now();

        // Extract performance metrics from benchmark result
        let metrics = self.extract_performance_metrics(result, measurements)?;

        // Create performance record
        let record = PerformanceRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            commit_hash: None,
            branch: None,
            environment: TestEnvironment::default(),
            metrics: metrics.clone(),
            metadata: HashMap::new(),
        };

        // Add to database
        self.performance_db.add_record(key.to_string(), record);

        // Detect regressions
        let mut regressions = Vec::new();
        if let (Some(baseline), Some(history)) = (
            self.baselines.get(key),
            self.performance_db.get_history(key),
        ) {
            for detector in &self.detectors {
                match detector.detect_regression(baseline, &metrics, history) {
                    Ok(regression) => {
                        if regression.regression_detected {
                            regressions.push(regression);
                        }
                    }
                    Err(e) => eprintln!("Regression detection error: {}", e),
                }
            }
        }

        // Send alerts for regressions
        for regression in &regressions {
            if let Err(e) = self.alert_system.send_alert(regression) {
                eprintln!("Alert sending error: {}", e);
            }
        }

        // Update baselines if needed
        self.update_baselines(key)?;

        // Save database
        self.performance_db.save(&self.config.baseline_dir)?;
        self.save_baselines()?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(RegressionTestResult {
            test_id: key.to_string(),
            status: if regressions.is_empty() {
                "passed".to_string()
            } else {
                "failed".to_string()
            },
            execution_time_ms: execution_time,
            regression_count: regressions.len(),
            regressions,
            metrics: metrics.clone(),
            baseline_comparison: self
                .baselines
                .get(key)
                .map(|baseline| self.compare_with_baseline(&metrics, baseline)),
        })
    }

    /// Extract performance metrics from a benchmark result.
    ///
    /// Every field is either measured or explicitly marked as unmeasured; no
    /// value is invented. In particular:
    ///
    /// * Timing percentiles come from `measurements.iteration_times` when
    ///   supplied. With a single wall-clock observation the distribution is a
    ///   point mass, so `min == median == max == mean` and `std_dev == 0`,
    ///   rather than the old `+20% / +50% / -20%` fabrications.
    /// * Memory is reported as `0` when it was not measured. Zero peak bytes is
    ///   physically impossible for a running benchmark, so downstream detectors
    ///   treat `0` as "not measured" and skip the memory comparison instead of
    ///   comparing against a constant 1 MiB.
    /// * Efficiency counters are `0` unless a profiler supplied them.
    fn extract_performance_metrics(
        &self,
        result: &BenchmarkResult<A>,
        measurements: &Measurements,
    ) -> Result<PerformanceMetrics<A>> {
        let elapsed_nanos = result.elapsed_time.as_nanos() as u64;

        let timing = if measurements.iteration_times.is_empty() {
            // Single observation: a point mass, honestly reported as such.
            TimingMetrics {
                mean_time_ns: elapsed_nanos,
                std_time_ns: 0,
                median_time_ns: elapsed_nanos,
                p95_time_ns: elapsed_nanos,
                p99_time_ns: elapsed_nanos,
                min_time_ns: elapsed_nanos,
                max_time_ns: elapsed_nanos,
            }
        } else {
            let samples: Vec<f64> = measurements
                .iteration_times
                .iter()
                .map(|d| d.as_nanos() as f64)
                .collect();
            timing_metrics_from_samples(&samples).unwrap_or(TimingMetrics {
                mean_time_ns: elapsed_nanos,
                std_time_ns: 0,
                median_time_ns: elapsed_nanos,
                p95_time_ns: elapsed_nanos,
                p99_time_ns: elapsed_nanos,
                min_time_ns: elapsed_nanos,
                max_time_ns: elapsed_nanos,
            })
        };

        let mut custom = HashMap::new();
        custom.insert(
            "timing_sample_count".to_string(),
            measurements.iteration_times.len().max(1) as f64,
        );
        custom.insert(
            "memory_measured".to_string(),
            if measurements.resources.is_some() {
                1.0
            } else {
                0.0
            },
        );

        let memory = match &measurements.resources {
            Some(resources) => MemoryMetrics {
                peak_memory_bytes: resources.peak_memory_bytes,
                avg_memory_bytes: resources.avg_memory_bytes,
                allocation_count: resources.allocation_count,
                fragmentation_ratio: resources.fragmentation_ratio,
                efficiency_score: resources.efficiency_score,
            },
            // `0` means "not measured" - see the doc comment above.
            None => MemoryMetrics::default(),
        };

        // Convergence rate is derived from the recorded objective history when
        // it is long enough; otherwise it stays at zero (unknown).
        let convergence_rate = convergence_rate_from_history(&result.function_value_history);

        Ok(PerformanceMetrics {
            timing,
            memory,
            efficiency: EfficiencyMetrics::default(),
            convergence: ConvergenceMetrics {
                final_objective: result.final_function_value,
                convergence_rate,
                iterations_to_convergence: result.convergence_step,
                quality_score: 0.0,
                stability_score: 0.0,
            },
            custom,
        })
    }

    /// Compare metrics with baseline.
    ///
    /// Every ratio is guarded: a zero (i.e. unmeasured) baseline yields a `0.0`
    /// change rather than an infinity or a NaN.
    fn compare_with_baseline(
        &self,
        metrics: &PerformanceMetrics<A>,
        baseline: &PerformanceBaseline<A>,
    ) -> BaselineComparison {
        fn relative_change(current: f64, reference: f64) -> f64 {
            if !current.is_finite() || !reference.is_finite() || reference == 0.0 {
                return 0.0;
            }
            ((current - reference) / reference.abs()) * 100.0
        }

        let timing_change = relative_change(
            metrics.timing.mean_time_ns as f64,
            baseline.baseline_stats.timing.mean,
        );

        let memory_change = relative_change(
            metrics.memory.peak_memory_bytes as f64,
            baseline.baseline_stats.memory.mean_memory,
        );

        let efficiency_change = relative_change(
            metrics.efficiency.efficiency_score,
            baseline.baseline_stats.efficiency.mean_efficiency,
        );

        let status = if timing_change > self.config.degradation_threshold
            || memory_change > self.config.memory_threshold
        {
            "degraded".to_string()
        } else if timing_change < -self.config.degradation_threshold {
            "improved".to_string()
        } else {
            "stable".to_string()
        };

        BaselineComparison {
            performance_change_percent: timing_change,
            memory_change_percent: memory_change,
            efficiency_change_percent: efficiency_change,
            status,
        }
    }

    /// Update baselines based on new performance data
    fn update_baselines(&mut self, key: &str) -> Result<bool> {
        if let Some(history) = self.performance_db.get_history(key) {
            if history.len() >= self.config.min_baseline_samples {
                let new_baseline = self.calculate_baseline(history)?;
                self.baselines.insert(key.to_string(), new_baseline);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Calculate baseline statistics from performance history.
    ///
    /// Every statistic is computed from the recorded samples:
    ///
    /// * median and IQR come from the sorted timing series (the median is no
    ///   longer aliased to the mean, and the IQR is no longer `1.35 * sigma`);
    /// * memory statistics are computed only over records that actually
    ///   measured memory (`peak_memory_bytes > 0`), and collapse to zero when
    ///   nothing was measured instead of `mean * 0.1`;
    /// * efficiency and convergence aggregates come from the records;
    /// * confidence intervals use the Student's t quantile for `n - 1` degrees
    ///   of freedom on the standard error of the mean, so a five-sample
    ///   baseline is not treated as if it had infinitely many samples.
    ///
    /// A history with fewer than two records has no variance, so the sample
    /// standard deviation is reported as `0` rather than dividing by `n - 1 = 0`.
    fn calculate_baseline(
        &self,
        history: &std::collections::VecDeque<PerformanceRecord<A>>,
    ) -> Result<PerformanceBaseline<A>> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let timing_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .filter(|value| value.is_finite())
            .collect();

        if timing_values.is_empty() {
            return Err(crate::error::OptimError::InvalidConfig(
                "Cannot build a baseline: history contains no finite timing measurements"
                    .to_string(),
            ));
        }

        let sorted_timing = distributions::sorted_finite(&timing_values);
        let timing_mean = distributions::mean(&sorted_timing).unwrap_or(0.0);
        let timing_std = distributions::sample_std_dev(&sorted_timing).unwrap_or(0.0);
        let timing_median = distributions::median_sorted(&sorted_timing).unwrap_or(timing_mean);
        let timing_iqr = distributions::iqr_sorted(&sorted_timing).unwrap_or(0.0);
        let coefficient_of_variation = if timing_mean != 0.0 {
            timing_std / timing_mean
        } else {
            0.0
        };

        // Memory is only aggregated over records that measured it.
        let memory_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.memory.peak_memory_bytes as f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .collect();
        let sorted_memory = distributions::sorted_finite(&memory_values);
        let memory_mean = distributions::mean(&sorted_memory).unwrap_or(0.0);
        let memory_std = distributions::sample_std_dev(&sorted_memory).unwrap_or(0.0);

        let mut peak_memory_percentiles = HashMap::new();
        for (label, percentile) in [("p50", 50.0), ("p90", 90.0), ("p95", 95.0), ("p99", 99.0)] {
            if let Some(value) = distributions::percentile_sorted(&sorted_memory, percentile) {
                peak_memory_percentiles.insert(label.to_string(), value);
            }
        }

        let fragmentation_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.memory.fragmentation_ratio)
            .filter(|value| value.is_finite())
            .collect();
        let fragmentation_trend = distributions::linear_regression(&fragmentation_values)
            .map(|trend| trend.slope)
            .unwrap_or(0.0);

        let flops_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.efficiency.flops)
            .filter(|value| value.is_finite())
            .collect();
        let mean_flops = distributions::mean(&flops_values).unwrap_or(0.0);
        let flops_cv = match (
            distributions::sample_std_dev(&flops_values),
            mean_flops != 0.0,
        ) {
            (Some(std_dev), true) => std_dev / mean_flops,
            _ => 0.0,
        };

        let efficiency_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.efficiency.efficiency_score)
            .filter(|value| value.is_finite())
            .collect();
        let mean_efficiency = distributions::mean(&efficiency_values).unwrap_or(0.0);

        let objective_values: Vec<f64> = history
            .iter()
            .filter_map(|r| r.metrics.convergence.final_objective.to_f64())
            .filter(|value| value.is_finite())
            .collect();
        let mean_objective = distributions::mean(&objective_values).unwrap_or(0.0);
        let std_objective = distributions::sample_std_dev(&objective_values).unwrap_or(0.0);

        let convergence_rates: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.convergence.convergence_rate)
            .filter(|value| value.is_finite())
            .collect();
        let mean_convergence_rate = distributions::mean(&convergence_rates).unwrap_or(0.0);
        // Consistency: 1 - coefficient of variation, clamped into 0..=1.
        let convergence_consistency = match (
            distributions::sample_std_dev(&convergence_rates),
            mean_convergence_rate != 0.0,
        ) {
            (Some(std_dev), true) => {
                (1.0 - (std_dev / mean_convergence_rate).abs()).clamp(0.0, 1.0)
            }
            _ => 0.0,
        };

        let sample_count = timing_values.len();
        let timing_ci = |confidence: f64| -> (f64, f64) {
            confidence_interval_of_mean(timing_mean, timing_std, sample_count, confidence)
        };
        let memory_ci = |confidence: f64| -> (f64, f64) {
            confidence_interval_of_mean(memory_mean, memory_std, sorted_memory.len(), confidence)
        };

        Ok(PerformanceBaseline {
            name: "auto_baseline".to_string(),
            baseline_stats: BaselineStatistics {
                timing: TimingStatistics {
                    mean: timing_mean,
                    std_dev: timing_std,
                    median: timing_median,
                    iqr: timing_iqr,
                    coefficient_of_variation,
                },
                memory: MemoryStatistics {
                    mean_memory: memory_mean,
                    std_dev_memory: memory_std,
                    peak_memory_percentiles,
                    fragmentation_stats: FragmentationStatistics {
                        mean_ratio: distributions::mean(&fragmentation_values).unwrap_or(0.0),
                        std_dev_ratio: distributions::sample_std_dev(&fragmentation_values)
                            .unwrap_or(0.0),
                        trend: fragmentation_trend,
                    },
                },
                efficiency: EfficiencyStatistics {
                    mean_flops,
                    flops_cv,
                    mean_efficiency,
                    custom_efficiency: HashMap::new(),
                },
                convergence: ConvergenceStatistics {
                    mean_objective: A::from(mean_objective).unwrap_or_else(A::zero),
                    std_objective: A::from(std_objective).unwrap_or_else(A::zero),
                    mean_convergence_rate,
                    convergence_consistency,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                timing_ci_95: timing_ci(0.95),
                memory_ci_95: memory_ci(0.95),
                timing_ci_99: timing_ci(0.99),
                memory_ci_99: memory_ci(0.99),
            },
            sample_count,
            created_at: now,
            updated_at: now,
        })
    }

    /// Generate CI report
    pub fn generate_ci_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        match self.config.ci_report_format {
            CiReportFormat::Json => self.generate_json_report(results),
            CiReportFormat::JunitXml => self.generate_junit_xml_report(results),
            CiReportFormat::Markdown => self.generate_markdown_report(results),
            CiReportFormat::GitHubActions => self.generate_github_actions_report(results),
        }
    }

    /// Generate JSON CI report
    fn generate_json_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let passed_tests = results.iter().filter(|r| r.status == "passed").count();
        let failed_tests = results.len() - passed_tests;

        let report = CiReport {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            total_tests: results.len(),
            passed_tests,
            failed_tests,
            test_results: results.iter().map(|r| r.to_ci_test_result()).collect(),
        };

        Ok(serde_json::to_string_pretty(&report)?)
    }

    /// Generate JUnit XML report (simplified)
    fn generate_junit_xml_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let total_time: f64 = results.iter().map(|r| r.execution_time_ms as f64).sum();
        let failed_tests = results.iter().filter(|r| r.status == "failed").count();

        let mut xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="performance_regression" tests="{}" failures="{}" time="{:.3}">
"#,
            results.len(),
            failed_tests,
            total_time / 1000.0
        );

        for result in results {
            xml.push_str(&format!(
                r#"  <testcase name="{}" time="{:.3}""#,
                result.test_id,
                result.execution_time_ms as f64 / 1000.0
            ));

            if result.status == "failed" {
                xml.push_str(&format!(
                    r#">
    <failure message="Performance regression detected">{} regression(s) detected</failure>
  </testcase>
"#,
                    result.regression_count
                ));
            } else {
                xml.push_str(" />\n");
            }
        }

        xml.push_str("</testsuite>\n");
        Ok(xml)
    }

    /// Generate Markdown report (simplified)
    fn generate_markdown_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let mut md = String::from("# Performance Regression Test Report\n\n");

        let passed = results.iter().filter(|r| r.status == "passed").count();
        let failed = results.len() - passed;

        md.push_str(&format!(
            "- **Total Tests**: {}\n- **Passed**: {}\n- **Failed**: {}\n\n",
            results.len(),
            passed,
            failed
        ));

        md.push_str("## Test Results\n\n| Test | Status | Regressions | Time (ms) |\n|------|--------|-------------|----------|\n");

        for result in results {
            md.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                result.test_id, result.status, result.regression_count, result.execution_time_ms
            ));
        }

        Ok(md)
    }

    /// Generate GitHub Actions report (simplified)
    fn generate_github_actions_report(
        &self,
        results: &[RegressionTestResult<A>],
    ) -> Result<String> {
        let failed_results: Vec<_> = results.iter().filter(|r| r.status == "failed").collect();

        if failed_results.is_empty() {
            Ok("::notice title=Performance Test::All performance tests passed".to_string())
        } else {
            let mut output = String::new();
            for result in failed_results {
                output.push_str(&format!(
                    "::error title=Performance Regression::{} detected {} regression(s)\n",
                    result.test_id, result.regression_count
                ));
            }
            Ok(output)
        }
    }
}

impl<A: Float + Send + Sync> RegressionTestResult<A> {
    /// Convert to CI test result
    pub fn to_ci_test_result(&self) -> CiTestResult {
        CiTestResult {
            name: self.test_id.clone(),
            status: self.status.clone(),
            execution_time_ms: self.execution_time_ms,
            regression_count: self.regression_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn timing_metrics_come_from_real_samples() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let metrics = timing_metrics_from_samples(&samples).expect("samples are usable");

        assert_eq!(metrics.min_time_ns, 1);
        assert_eq!(metrics.max_time_ns, 100);
        assert_eq!(metrics.mean_time_ns, 51); // 50.5 rounds to 51
        assert_eq!(metrics.median_time_ns, 51); // interpolated 50.5
        assert_eq!(metrics.p95_time_ns, 95);
        assert_eq!(metrics.p99_time_ns, 99);
        // Uniform 1..100 has sample std dev sqrt(101*100/12) ~= 29.01
        assert_eq!(metrics.std_time_ns, 29);
    }

    #[test]
    fn timing_metrics_reject_empty_and_nan_samples() {
        assert!(timing_metrics_from_samples(&[]).is_none());
        assert!(timing_metrics_from_samples(&[f64::NAN, f64::INFINITY]).is_none());

        // A single sample is a point mass with zero spread.
        let single = timing_metrics_from_samples(&[42.0]).expect("one sample is usable");
        assert_eq!(single.mean_time_ns, 42);
        assert_eq!(single.std_time_ns, 0);
        assert_eq!(single.min_time_ns, 42);
        assert_eq!(single.max_time_ns, 42);
        assert_eq!(single.p99_time_ns, 42);
    }

    #[test]
    fn convergence_rate_is_derived_or_unknown() {
        // Halving every step: contraction factor 0.5.
        let history: Vec<f64> = (0..5).map(|i| 1.0 / (2.0f64).powi(i)).collect();
        let rate = convergence_rate_from_history(&history);
        assert!((rate - 0.5).abs() < 1e-12, "rate = {}", rate);

        // Unknown cases return 0.0 rather than a fabricated 0.95.
        assert_eq!(convergence_rate_from_history::<f64>(&[]), 0.0);
        assert_eq!(convergence_rate_from_history(&[1.0]), 0.0);
        assert_eq!(convergence_rate_from_history(&[0.0, 1.0]), 0.0);
        assert_eq!(convergence_rate_from_history(&[-1.0, -2.0]), 0.0);
    }

    #[test]
    fn confidence_intervals_use_the_t_quantile() {
        // n = 10, mean 100, sd 10: SE = 3.1623, t_{0.975,9} = 2.262.
        let (low, high) = confidence_interval_of_mean(100.0, 10.0, 10, 0.95);
        assert!((high - low - 2.0 * 2.262_157 * 3.162_278).abs() < 1e-3);
        assert!(low < 100.0 && high > 100.0);

        // A larger confidence level yields a wider interval.
        let (low99, high99) = confidence_interval_of_mean(100.0, 10.0, 10, 0.99);
        assert!(high99 - low99 > high - low);

        // Degenerate inputs collapse instead of producing NaN/infinity.
        assert_eq!(confidence_interval_of_mean(5.0, 0.0, 10, 0.95), (5.0, 5.0));
        assert_eq!(confidence_interval_of_mean(5.0, 1.0, 1, 0.95), (5.0, 5.0));
        assert_eq!(confidence_interval_of_mean(5.0, 1.0, 0, 0.95), (5.0, 5.0));
    }

    #[test]
    fn measurements_default_to_unmeasured() {
        let measurements = Measurements::default();
        assert!(measurements.iteration_times.is_empty());
        assert!(measurements.resources.is_none());

        let timed = Measurements::from_iteration_times(vec![Duration::from_millis(1)]);
        assert_eq!(timed.iteration_times.len(), 1);
        assert!(timed.resources.is_none());

        let resources = ResourceMeasurements::from_memory(2048, 1024);
        assert_eq!(resources.peak_memory_bytes, 2048);
        assert_eq!(resources.avg_memory_bytes, 1024);
        assert_eq!(resources.allocation_count, 0);
    }
}
