// Regression detection algorithms for performance testing
//
// This module provides various algorithms for detecting performance regressions,
// including statistical hypothesis testing, sliding window analysis, and change
// point detection.
//
// All statistical machinery lives in [`crate::regression_tester::distributions`]
// and is exercised against published reference values, so the p-values reported
// here are real p-values rather than fixed constants.

use crate::error::Result;
use crate::regression_tester::distributions::{
    binary_segmentation, mean as finite_mean, sample_std_dev, single_observation_t_test,
};
use crate::regression_tester::types::{
    ChangePointAnalysis, OutlierAnalysis, PerformanceBaseline, PerformanceMetrics,
    PerformanceRecord, RegressionAnalysis, RegressionDetector, RegressionResult,
    StatisticalTestResult, TrendAnalysis, TrendDirection,
};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Default performance degradation threshold in percent.
///
/// Kept in sync with [`crate::regression_tester::config::RegressionConfig`].
pub const DEFAULT_DEGRADATION_THRESHOLD_PERCENT: f64 = 5.0;

/// Default memory regression threshold in percent.
pub const DEFAULT_MEMORY_THRESHOLD_PERCENT: f64 = 10.0;

/// Convert an observed relative change into a `0.0..=1.0` severity score,
/// calibrated against the threshold that made the change interesting.
///
/// # Contract
///
/// Severity is expressed **relative to the configured threshold**, not as a raw
/// fraction of the change. Previously a 6% regression against a 5% threshold
/// produced `0.06`, which every downstream consumer rounded away to "noise".
///
/// With `ratio = change_percent / threshold_percent` the mapping is:
///
/// | `ratio`   | severity | [`crate::regression_tester::config::AlertSeverity`] |
/// |-----------|----------|-----------------------------------------------------|
/// | `<= 0`    | `0.00`   | not alertable                                        |
/// | `1.0`     | `0.25`   | `Low`, and above the default `0.05` alert floor       |
/// | `~1.28`   | `0.30`   | `Medium`                                             |
/// | `~2.91`   | `0.60`   | `High`                                               |
/// | `>= 4.0`  | `>= 0.80`| `Critical`                                           |
///
/// The curve is continuous and strictly increasing, and saturates towards
/// `1.0` for extreme regressions. Alert consumers should therefore keep their
/// severity cut-offs on the `0.25 / 0.30 / 0.60 / 0.80` grid above.
pub fn severity_from_threshold_ratio(change_percent: f64, threshold_percent: f64) -> f64 {
    if !change_percent.is_finite() || !threshold_percent.is_finite() {
        return 0.0;
    }
    if change_percent <= 0.0 {
        return 0.0;
    }
    let threshold = if threshold_percent > 0.0 {
        threshold_percent
    } else {
        DEFAULT_DEGRADATION_THRESHOLD_PERCENT
    };
    let ratio = change_percent / threshold;

    let severity = if ratio < 1.0 {
        0.25 * ratio
    } else if ratio <= 4.0 {
        0.25 + 0.55 * (ratio - 1.0) / 3.0
    } else {
        0.80 + 0.20 * (1.0 - (-(ratio - 4.0) / 4.0).exp())
    };

    severity.clamp(0.0, 1.0)
}

/// Build the "not enough information" result shared by every detector.
fn inconclusive_result<A: Float>(
    test_id: &str,
    reason: &str,
    recommendation: &str,
) -> RegressionResult<A> {
    RegressionResult {
        test_id: test_id.to_string(),
        regression_detected: false,
        severity: 0.0,
        confidence: 0.0,
        performance_change_percent: 0.0,
        memory_change_percent: 0.0,
        affected_metrics: vec![],
        statistical_tests: vec![],
        analysis: RegressionAnalysis {
            trend_analysis: TrendAnalysis {
                direction: TrendDirection::Stable,
                magnitude: 0.0,
                significance: 0.0,
                start_point: None,
            },
            change_point_analysis: ChangePointAnalysis {
                change_points: vec![],
                magnitudes: vec![],
                confidences: vec![],
            },
            outlier_analysis: OutlierAnalysis {
                outlier_indices: vec![],
                outlier_scores: vec![],
                outlier_types: vec![],
            },
            root_cause_hints: vec![reason.to_string()],
        },
        recommendations: vec![recommendation.to_string()],
    }
}

/// Percentage change from `baseline` to `current`, or `None` when the baseline
/// carries no usable scale.
fn percent_change(current: f64, baseline: f64) -> Option<f64> {
    if !current.is_finite() || !baseline.is_finite() || baseline == 0.0 {
        return None;
    }
    Some(((current - baseline) / baseline.abs()) * 100.0)
}

/// Statistical test-based regression detector
///
/// Compares a single new measurement against a baseline distribution with a
/// one-sample Student's t-test built on the *prediction* standard error
/// `sigma * sqrt(1 + 1/n)`. A single observation is not a sample mean, so the
/// standard error of the mean (`sigma / sqrt(n)`) would grossly overstate the
/// evidence and flag every run once the baseline grew large.
#[derive(Debug)]
pub struct StatisticalTestDetector {
    /// Statistical significance threshold (alpha level)
    alpha: f64,
    /// Timing degradation threshold in percent, used to scale severity
    degradation_threshold_percent: f64,
    /// Memory regression threshold in percent
    memory_threshold_percent: f64,
}

impl StatisticalTestDetector {
    /// Create a new statistical test detector with default significance level
    pub fn new() -> Self {
        Self {
            alpha: 0.05,
            degradation_threshold_percent: DEFAULT_DEGRADATION_THRESHOLD_PERCENT,
            memory_threshold_percent: DEFAULT_MEMORY_THRESHOLD_PERCENT,
        }
    }

    /// Create a new statistical test detector with custom significance level
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::new()
        }
    }

    /// Create a detector with custom alpha and severity-calibration thresholds
    pub fn with_thresholds(
        alpha: f64,
        degradation_threshold_percent: f64,
        memory_threshold_percent: f64,
    ) -> Self {
        Self {
            alpha,
            degradation_threshold_percent,
            memory_threshold_percent,
        }
    }
}

impl Default for StatisticalTestDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug + Send + Sync> RegressionDetector<A> for StatisticalTestDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        _history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        let current_time = current_metrics.timing.mean_time_ns as f64;
        let baseline_mean = baseline.baseline_stats.timing.mean;
        let baseline_std = baseline.baseline_stats.timing.std_dev;

        let Some(change_percent) = percent_change(current_time, baseline_mean) else {
            return Ok(inconclusive_result(
                "statistical_test",
                "Baseline timing mean is zero or non-finite; cannot compute a relative change",
                "Rebuild the baseline from measurements with a non-zero mean execution time",
            ));
        };

        // Memory change is only meaningful when the baseline actually measured
        // memory. `None` here means "not measured", never "no change".
        let current_memory = current_metrics.memory.peak_memory_bytes as f64;
        let memory_change_percent =
            percent_change(current_memory, baseline.baseline_stats.memory.mean_memory);

        let Some(test) = single_observation_t_test(
            current_time,
            baseline_mean,
            baseline_std,
            baseline.sample_count,
        ) else {
            return Ok(inconclusive_result(
                "statistical_test",
                "Baseline has zero variance or fewer than two samples; a t-test is undefined",
                "Collect at least two baseline runs with non-degenerate timing before testing",
            ));
        };

        let timing_regression = test.p_value < self.alpha && change_percent > 0.0;
        let memory_regression = memory_change_percent
            .map(|change| change > self.memory_threshold_percent)
            .unwrap_or(false);
        let regression_detected = timing_regression || memory_regression;

        let mut affected_metrics = Vec::new();
        if timing_regression {
            affected_metrics.push("timing".to_string());
        }
        if memory_regression {
            affected_metrics.push("memory".to_string());
        }

        let timing_severity =
            severity_from_threshold_ratio(change_percent, self.degradation_threshold_percent);
        let memory_severity = memory_change_percent
            .map(|change| severity_from_threshold_ratio(change, self.memory_threshold_percent))
            .unwrap_or(0.0);
        let severity = if regression_detected {
            timing_severity.max(memory_severity)
        } else {
            0.0
        };

        let mut root_cause_hints = Vec::new();
        if memory_change_percent.is_none() {
            root_cause_hints.push(
                "Memory was not measured for this baseline; memory delta unavailable".to_string(),
            );
        }

        Ok(RegressionResult {
            test_id: "statistical_test".to_string(),
            regression_detected,
            severity,
            confidence: (1.0 - test.p_value).clamp(0.0, 1.0),
            performance_change_percent: change_percent,
            memory_change_percent: memory_change_percent.unwrap_or(0.0),
            affected_metrics,
            statistical_tests: vec![StatisticalTestResult {
                test_name: "one_sample_t_test_prediction_interval".to_string(),
                test_statistic: test.t_statistic,
                p_value: test.p_value,
                degrees_of_freedom: Some(baseline.sample_count.saturating_sub(1)),
                conclusion: if regression_detected {
                    format!(
                        "Significant regression detected (p = {:.6} < alpha = {:.4})",
                        test.p_value, self.alpha
                    )
                } else {
                    format!(
                        "No significant change (p = {:.6} >= alpha = {:.4})",
                        test.p_value, self.alpha
                    )
                },
            }],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    // Execution time is a lower-is-better metric.
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else if change_percent < 0.0 {
                        TrendDirection::Improving
                    } else {
                        TrendDirection::Stable
                    },
                    magnitude: change_percent.abs(),
                    significance: (1.0 - test.p_value).clamp(0.0, 1.0),
                    start_point: None,
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints,
            },
            recommendations: if regression_detected {
                vec![
                    "Check for recent code changes that might affect performance".to_string(),
                    "Review system load and resource availability".to_string(),
                    "Consider running additional test iterations for confirmation".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "statistical_test"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("alpha".to_string(), self.alpha.to_string());
        config.insert(
            "degradation_threshold_percent".to_string(),
            self.degradation_threshold_percent.to_string(),
        );
        config.insert(
            "memory_threshold_percent".to_string(),
            self.memory_threshold_percent.to_string(),
        );
        config
    }
}

/// Sliding window regression detector
///
/// Compares the current measurement against a sliding window of recent
/// measurements. Both timing and memory deltas are computed from the window,
/// and the reported confidence comes from a t-test of the current observation
/// against the window distribution rather than from a fixed constant.
#[derive(Debug)]
pub struct SlidingWindowDetector {
    /// Size of the sliding window for comparison
    window_size: usize,
    /// Performance degradation threshold (percentage)
    threshold: f64,
}

impl SlidingWindowDetector {
    /// Create a new sliding window detector with default parameters
    pub fn new() -> Self {
        Self {
            window_size: 10,
            threshold: DEFAULT_DEGRADATION_THRESHOLD_PERCENT,
        }
    }

    /// Create a new sliding window detector with custom parameters
    pub fn with_params(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
        }
    }
}

impl Default for SlidingWindowDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug + Send + Sync> RegressionDetector<A> for SlidingWindowDetector {
    fn detect_regression(
        &self,
        _baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        if history.len() < self.window_size {
            return Ok(inconclusive_result(
                "sliding_window",
                "Insufficient data for sliding window analysis",
                "Collect more performance data for accurate analysis",
            ));
        }

        let recent_times: Vec<f64> = history
            .iter()
            .rev()
            .take(self.window_size)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        // Memory samples are only usable when the window actually recorded
        // memory; a window of zeroes means "unmeasured".
        let recent_memory: Vec<f64> = history
            .iter()
            .rev()
            .take(self.window_size)
            .map(|r| r.metrics.memory.peak_memory_bytes as f64)
            .filter(|value| *value > 0.0)
            .collect();

        let Some(recent_avg) = finite_mean(&recent_times) else {
            return Ok(inconclusive_result(
                "sliding_window",
                "Sliding window contains no finite timing measurements",
                "Verify that the benchmark harness records execution time",
            ));
        };

        let current_time = current_metrics.timing.mean_time_ns as f64;
        let Some(change_percent) = percent_change(current_time, recent_avg) else {
            return Ok(inconclusive_result(
                "sliding_window",
                "Sliding window mean is zero; cannot compute a relative change",
                "Verify that the benchmark harness records execution time",
            ));
        };

        // F65: real memory delta over the window instead of a hardcoded zero.
        let current_memory = current_metrics.memory.peak_memory_bytes as f64;
        let memory_change_percent = if recent_memory.is_empty() || current_memory <= 0.0 {
            None
        } else {
            finite_mean(&recent_memory)
                .and_then(|window_memory| percent_change(current_memory, window_memory))
        };

        let regression_detected = change_percent > self.threshold;

        // Confidence is derived from the window distribution: how surprising is
        // the current observation given the recent spread?
        let window_std = sample_std_dev(&recent_times);
        let test = window_std.and_then(|std_dev| {
            single_observation_t_test(current_time, recent_avg, std_dev, recent_times.len())
        });
        let confidence = test
            .map(|t| (1.0 - t.p_value).clamp(0.0, 1.0))
            .unwrap_or(0.0);

        let mut affected_metrics = Vec::new();
        if regression_detected {
            affected_metrics.push("timing".to_string());
        }
        if memory_change_percent
            .map(|change| change > DEFAULT_MEMORY_THRESHOLD_PERCENT)
            .unwrap_or(false)
        {
            affected_metrics.push("memory".to_string());
        }

        let statistical_tests = test
            .map(|t| {
                vec![StatisticalTestResult {
                    test_name: "sliding_window_t_test".to_string(),
                    test_statistic: t.t_statistic,
                    p_value: t.p_value,
                    degrees_of_freedom: Some(recent_times.len().saturating_sub(1)),
                    conclusion: format!(
                        "Current run is {:.2}% from the {}-sample window mean (p = {:.6})",
                        change_percent,
                        recent_times.len(),
                        t.p_value
                    ),
                }]
            })
            .unwrap_or_default();

        let mut root_cause_hints = Vec::new();
        if memory_change_percent.is_none() {
            root_cause_hints.push(
                "Memory was not measured over the sliding window; memory delta unavailable"
                    .to_string(),
            );
        }
        if test.is_none() {
            root_cause_hints.push(
                "Sliding window has zero variance; confidence cannot be estimated".to_string(),
            );
        }

        Ok(RegressionResult {
            test_id: "sliding_window".to_string(),
            regression_detected,
            severity: if regression_detected {
                severity_from_threshold_ratio(change_percent, self.threshold)
            } else {
                0.0
            },
            confidence,
            performance_change_percent: change_percent,
            memory_change_percent: memory_change_percent.unwrap_or(0.0),
            affected_metrics,
            statistical_tests,
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else if change_percent < 0.0 {
                        TrendDirection::Improving
                    } else {
                        TrendDirection::Stable
                    },
                    magnitude: change_percent.abs(),
                    significance: confidence,
                    start_point: Some(history.len().saturating_sub(self.window_size)),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints,
            },
            recommendations: if regression_detected {
                vec![
                    "Performance degradation detected in recent window".to_string(),
                    "Compare current run with recent baseline".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "sliding_window"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("window_size".to_string(), self.window_size.to_string());
        config.insert("threshold".to_string(), self.threshold.to_string());
        config
    }
}

/// Change point detection regression detector
///
/// Locates level shifts in the historical series with binary segmentation: at
/// each step the split that maximises the between-segment sum of squares is
/// chosen and then accepted only when a Welch t-test across the split is
/// significant. The reported change points are those argmax locations - not a
/// fixed midpoint.
#[derive(Debug)]
pub struct ChangePointDetector {
    /// Minimum segment size for change point analysis
    min_segment_size: usize,
    /// Statistical significance threshold for change detection
    significance_threshold: f64,
    /// Maximum number of change points to report
    max_change_points: usize,
}

impl ChangePointDetector {
    /// Create a new change point detector with default parameters
    pub fn new() -> Self {
        Self {
            min_segment_size: 5,
            significance_threshold: 0.05,
            max_change_points: 8,
        }
    }

    /// Create a new change point detector with custom parameters
    pub fn with_params(min_segment_size: usize, significance_threshold: f64) -> Self {
        Self {
            min_segment_size,
            significance_threshold,
            ..Self::new()
        }
    }
}

impl Default for ChangePointDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug + Send + Sync> RegressionDetector<A> for ChangePointDetector {
    fn detect_regression(
        &self,
        _baseline: &PerformanceBaseline<A>,
        _current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        if history.len() < 2 * self.min_segment_size {
            return Ok(inconclusive_result(
                "change_point",
                "Insufficient data for change point detection",
                "Collect more performance data for change point analysis",
            ));
        }

        let series: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let change_points = binary_segmentation(
            &series,
            self.min_segment_size,
            self.significance_threshold,
            self.max_change_points,
        );

        if change_points.is_empty() {
            return Ok(inconclusive_result(
                "change_point",
                "No statistically significant level shift detected in the history",
                "Continue monitoring; the series is consistent with a single regime",
            ));
        }

        // `binary_segmentation` returns points ordered by index, so the last
        // entry is the most recent regime change. That - not the largest shift
        // in the whole history - is what decides whether the code is currently
        // regressed: a series that improved 30% long ago and then regressed 5%
        // yesterday is regressed today, even though the improvement has the
        // bigger magnitude.
        let Some(latest) = change_points.last() else {
            return Ok(inconclusive_result(
                "change_point",
                "No statistically significant level shift detected in the history",
                "Continue monitoring; the series is consistent with a single regime",
            ));
        };

        let change_percent = latest.relative_magnitude * 100.0;
        // Align with the other detectors: a statistically significant but
        // negligible shift is not a regression.
        let regression_detected = change_percent > DEFAULT_DEGRADATION_THRESHOLD_PERCENT;
        let confidence = (1.0 - latest.p_value).clamp(0.0, 1.0);

        Ok(RegressionResult {
            test_id: "change_point".to_string(),
            regression_detected,
            severity: if regression_detected {
                severity_from_threshold_ratio(change_percent, DEFAULT_DEGRADATION_THRESHOLD_PERCENT)
            } else {
                0.0
            },
            confidence,
            performance_change_percent: change_percent,
            memory_change_percent: 0.0,
            affected_metrics: if regression_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: change_points
                .iter()
                .map(|cp| StatisticalTestResult {
                    test_name: format!("binary_segmentation_welch_t@{}", cp.index),
                    test_statistic: cp.t_statistic,
                    p_value: cp.p_value,
                    degrees_of_freedom: None,
                    conclusion: format!(
                        "Level shift of {:.2}% at index {}",
                        cp.relative_magnitude * 100.0,
                        cp.index
                    ),
                })
                .collect(),
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else if change_percent < 0.0 {
                        TrendDirection::Improving
                    } else {
                        TrendDirection::Stable
                    },
                    magnitude: change_percent.abs(),
                    significance: confidence,
                    start_point: Some(latest.index),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: change_points.iter().map(|cp| cp.index).collect(),
                    magnitudes: change_points
                        .iter()
                        .map(|cp| cp.relative_magnitude * 100.0)
                        .collect(),
                    confidences: change_points
                        .iter()
                        .map(|cp| (1.0 - cp.p_value).clamp(0.0, 1.0))
                        .collect(),
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![format!(
                    "Most recent significant performance change detected at index {} ({:+.2}%); \
                     {} change point(s) found in total",
                    latest.index,
                    change_percent,
                    change_points.len()
                )],
            },
            recommendations: if regression_detected {
                vec![
                    "Investigate changes that occurred around the detected change point"
                        .to_string(),
                    "Review commits and deployments near the change point".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "change_point"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "min_segment_size".to_string(),
            self.min_segment_size.to_string(),
        );
        config.insert(
            "significance_threshold".to_string(),
            self.significance_threshold.to_string(),
        );
        config.insert(
            "max_change_points".to_string(),
            self.max_change_points.to_string(),
        );
        config
    }
}

/// Cumulative distribution function of the standard normal distribution.
///
/// Retained for backwards compatibility; delegates to the shared, table-checked
/// implementation in [`crate::regression_tester::distributions`].
pub fn normal_cdf(x: f64) -> f64 {
    crate::regression_tester::distributions::normal_cdf(x)
}

/// Error function.
///
/// Retained for backwards compatibility; delegates to the shared, table-checked
/// implementation in [`crate::regression_tester::distributions`].
pub fn erf(x: f64) -> f64 {
    crate::regression_tester::distributions::erf(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression_tester::config::TestEnvironment;
    use crate::regression_tester::types::{
        BaselineStatistics, ConfidenceIntervals, ConvergenceMetrics, ConvergenceStatistics,
        EfficiencyMetrics, EfficiencyStatistics, FragmentationStatistics, MemoryMetrics,
        MemoryStatistics, PerformanceMetrics, TimingMetrics, TimingStatistics,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default()
    }

    fn create_test_baseline() -> PerformanceBaseline<f64> {
        PerformanceBaseline {
            name: "test_baseline".to_string(),
            baseline_stats: BaselineStatistics {
                timing: TimingStatistics {
                    mean: 1000.0,
                    std_dev: 100.0,
                    median: 1000.0,
                    iqr: 200.0,
                    coefficient_of_variation: 0.1,
                },
                memory: MemoryStatistics {
                    mean_memory: 1000000.0,
                    std_dev_memory: 100000.0,
                    peak_memory_percentiles: HashMap::new(),
                    fragmentation_stats: FragmentationStatistics {
                        mean_ratio: 0.1,
                        std_dev_ratio: 0.05,
                        trend: 0.0,
                    },
                },
                efficiency: EfficiencyStatistics {
                    mean_flops: 1000.0,
                    flops_cv: 0.1,
                    mean_efficiency: 0.8,
                    custom_efficiency: HashMap::new(),
                },
                convergence: ConvergenceStatistics {
                    mean_objective: 0.01,
                    std_objective: 0.001,
                    mean_convergence_rate: 0.95,
                    convergence_consistency: 0.9,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                timing_ci_95: (900.0, 1100.0),
                memory_ci_95: (900000.0, 1100000.0),
                timing_ci_99: (850.0, 1150.0),
                memory_ci_99: (850000.0, 1150000.0),
            },
            sample_count: 100,
            created_at: timestamp(),
            updated_at: timestamp(),
        }
    }

    fn create_test_metrics(mean_time_ns: u64) -> PerformanceMetrics<f64> {
        PerformanceMetrics {
            timing: TimingMetrics {
                mean_time_ns,
                std_time_ns: 10,
                median_time_ns: mean_time_ns,
                p95_time_ns: mean_time_ns + 50,
                p99_time_ns: mean_time_ns + 100,
                min_time_ns: mean_time_ns.saturating_sub(20),
                max_time_ns: mean_time_ns + 200,
            },
            memory: MemoryMetrics {
                peak_memory_bytes: 1000000,
                avg_memory_bytes: 900000,
                allocation_count: 100,
                fragmentation_ratio: 0.1,
                efficiency_score: 0.9,
            },
            efficiency: EfficiencyMetrics {
                flops: 1000.0,
                arithmetic_intensity: 2.0,
                cache_hit_ratio: 0.95,
                cpu_utilization: 0.8,
                efficiency_score: 0.85,
                custom_metrics: HashMap::new(),
            },
            convergence: ConvergenceMetrics {
                final_objective: 0.01,
                convergence_rate: 0.95,
                iterations_to_convergence: Some(100),
                quality_score: 0.9,
                stability_score: 0.85,
            },
            custom: HashMap::new(),
        }
    }

    fn record(time_ns: u64) -> PerformanceRecord<f64> {
        PerformanceRecord {
            timestamp: timestamp(),
            commit_hash: None,
            branch: None,
            environment: TestEnvironment::default(),
            metrics: create_test_metrics(time_ns),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn statistical_detector_reports_a_real_p_value() {
        let detector = StatisticalTestDetector::new();
        let baseline = create_test_baseline();
        let history = VecDeque::new();

        // Baseline is 1000 +/- 100 over 100 samples. 1200 is two prediction
        // standard errors away, which the t-test must call significant.
        let regressed = detector
            .detect_regression(&baseline, &create_test_metrics(1200), &history)
            .expect("detector runs");

        assert_eq!(regressed.test_id, "statistical_test");
        assert!(regressed.performance_change_percent > 15.0);
        let p = regressed
            .statistical_tests
            .first()
            .map(|t| t.p_value)
            .expect("a test is reported");
        assert!(p < 0.05, "expected a significant p-value, got {}", p);
        assert!(regressed.regression_detected);
        // Severity must be alertable, not the raw 0.20 fraction of the change.
        assert!(
            regressed.severity >= 0.8,
            "20% against a 5% threshold is critical, got {}",
            regressed.severity
        );
    }

    #[test]
    fn statistical_detector_stays_quiet_on_noise() {
        let detector = StatisticalTestDetector::new();
        let baseline = create_test_baseline();
        let history = VecDeque::new();

        // Well inside one prediction standard error.
        let stable = detector
            .detect_regression(&baseline, &create_test_metrics(1030), &history)
            .expect("detector runs");
        assert!(!stable.regression_detected);
        assert_eq!(stable.severity, 0.0);
        let p = stable
            .statistical_tests
            .first()
            .map(|t| t.p_value)
            .expect("a test is reported");
        assert!(p > 0.05, "expected a non-significant p-value, got {}", p);
    }

    #[test]
    fn statistical_detector_guards_degenerate_baselines() {
        let detector = StatisticalTestDetector::new();
        let history = VecDeque::new();

        let mut zero_std = create_test_baseline();
        zero_std.baseline_stats.timing.std_dev = 0.0;
        let result = detector
            .detect_regression(&zero_std, &create_test_metrics(1200), &history)
            .expect("detector runs");
        assert!(!result.regression_detected);
        assert!(result
            .analysis
            .root_cause_hints
            .iter()
            .any(|hint| hint.contains("zero variance")));

        let mut zero_mean = create_test_baseline();
        zero_mean.baseline_stats.timing.mean = 0.0;
        let result = detector
            .detect_regression(&zero_mean, &create_test_metrics(1200), &history)
            .expect("detector runs");
        assert!(!result.regression_detected);
        assert!(result.performance_change_percent == 0.0);

        let mut single_sample = create_test_baseline();
        single_sample.sample_count = 1;
        let result = detector
            .detect_regression(&single_sample, &create_test_metrics(1200), &history)
            .expect("detector runs");
        assert!(!result.regression_detected);
    }

    #[test]
    fn severity_is_calibrated_against_the_threshold() {
        // Just over the threshold: low, but above the 0.05 alert floor.
        let just_over = severity_from_threshold_ratio(5.1, 5.0);
        assert!(just_over > 0.05, "must be alertable, got {}", just_over);
        assert!(just_over < 0.3, "must be Low, got {}", just_over);

        // 4x the threshold is critical.
        assert!(severity_from_threshold_ratio(20.0, 5.0) >= 0.8);
        // Monotone and bounded.
        assert!(severity_from_threshold_ratio(10.0, 5.0) > severity_from_threshold_ratio(6.0, 5.0));
        assert!(severity_from_threshold_ratio(1.0e9, 5.0) <= 1.0);
        // Improvements and garbage inputs carry no severity.
        assert_eq!(severity_from_threshold_ratio(-10.0, 5.0), 0.0);
        assert_eq!(severity_from_threshold_ratio(f64::NAN, 5.0), 0.0);
    }

    #[test]
    fn sliding_window_detects_degradation_and_memory_change() {
        let detector = SlidingWindowDetector::new();
        let baseline = create_test_baseline();

        let mut history = VecDeque::new();
        for i in 0..15 {
            // Slight jitter so the window has a usable variance.
            history.push_back(record(1000 + (i % 3)));
        }

        let result = detector
            .detect_regression(&baseline, &create_test_metrics(1200), &history)
            .expect("detector runs");

        assert_eq!(result.test_id, "sliding_window");
        assert!(result.performance_change_percent > 15.0);
        assert!(result.regression_detected);
        // Confidence must come from the data, not the old 0.8 constant.
        assert!(result.confidence > 0.9, "confidence {}", result.confidence);
        assert!(!result.statistical_tests.is_empty());
        // Memory is identical across the window, so the delta is exactly zero
        // (and reported as measured, not as "unavailable").
        assert_eq!(result.memory_change_percent, 0.0);
        assert!(!result
            .analysis
            .root_cause_hints
            .iter()
            .any(|hint| hint.contains("Memory was not measured")));
    }

    #[test]
    fn sliding_window_reports_memory_growth() {
        let detector = SlidingWindowDetector::new();
        let baseline = create_test_baseline();

        let mut history = VecDeque::new();
        for i in 0..12 {
            let mut rec = record(1000 + (i % 3));
            rec.metrics.memory.peak_memory_bytes = 1_000_000;
            history.push_back(rec);
        }

        let mut current = create_test_metrics(1001);
        current.memory.peak_memory_bytes = 1_500_000;

        let result = detector
            .detect_regression(&baseline, &current, &history)
            .expect("detector runs");
        assert!((result.memory_change_percent - 50.0).abs() < 1e-9);
        assert!(result
            .affected_metrics
            .iter()
            .any(|metric| metric == "memory"));
    }

    #[test]
    fn sliding_window_needs_a_full_window() {
        let detector = SlidingWindowDetector::new();
        let baseline = create_test_baseline();
        let history = VecDeque::new();

        let result = detector
            .detect_regression(&baseline, &create_test_metrics(1000), &history)
            .expect("detector runs");

        assert!(!result.regression_detected);
        assert!(result
            .analysis
            .root_cause_hints
            .iter()
            .any(|hint| hint.contains("Insufficient data")));
    }

    #[test]
    fn change_point_detector_finds_the_actual_shift() {
        let detector = ChangePointDetector::new();
        let baseline = create_test_baseline();

        // 20 stable runs then 10 degraded ones: the change point is at 20,
        // not at the midpoint 15.
        let mut history = VecDeque::new();
        for i in 0..20 {
            history.push_back(record(1000 + (i % 3)));
        }
        for i in 0..10 {
            history.push_back(record(1200 + (i % 3)));
        }

        let result = detector
            .detect_regression(&baseline, &create_test_metrics(1200), &history)
            .expect("detector runs");

        assert_eq!(result.test_id, "change_point");
        let points = &result.analysis.change_point_analysis.change_points;
        assert!(!points.is_empty());
        assert!(
            points.iter().any(|&p| p.abs_diff(20) <= 1),
            "expected a change point near index 20, got {:?}",
            points
        );
        assert!(result.regression_detected);
        assert!(result.performance_change_percent > 15.0);
        assert!(result.confidence > 0.95);
    }

    /// A history that improved a lot and then regressed must report the
    /// *recent* regression, not the older (larger) improvement.
    #[test]
    fn change_point_detector_uses_the_most_recent_shift() {
        let detector = ChangePointDetector::new();
        let baseline = create_test_baseline();

        let mut history = VecDeque::new();
        for i in 0..20 {
            history.push_back(record(1000 + (i % 3)));
        }
        // Big improvement.
        for i in 0..20 {
            history.push_back(record(700 + (i % 3)));
        }
        // Smaller, but real, regression on top of the improved level.
        for i in 0..20 {
            history.push_back(record(770 + (i % 3)));
        }

        let result = detector
            .detect_regression(&baseline, &create_test_metrics(770), &history)
            .expect("detector runs");

        assert!(
            result.regression_detected,
            "a recent regression must not be masked by an older improvement"
        );
        assert!(result.performance_change_percent > 5.0);
        let points = &result.analysis.change_point_analysis.change_points;
        assert!(points.len() >= 2, "expected both shifts, got {:?}", points);
        assert!(
            result
                .analysis
                .trend_analysis
                .start_point
                .map(|p| p.abs_diff(40) <= 2)
                .unwrap_or(false),
            "expected the latest change point near index 40, got {:?}",
            result.analysis.trend_analysis.start_point
        );
    }

    #[test]
    fn change_point_detector_ignores_stable_history() {
        let detector = ChangePointDetector::new();
        let baseline = create_test_baseline();

        let mut history = VecDeque::new();
        for i in 0..40 {
            history.push_back(record(1000 + (i % 5)));
        }

        let result = detector
            .detect_regression(&baseline, &create_test_metrics(1000), &history)
            .expect("detector runs");

        assert!(!result.regression_detected);
        assert!(result
            .analysis
            .change_point_analysis
            .change_points
            .is_empty());
    }

    #[test]
    fn normal_cdf_delegates_to_the_shared_implementation() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-12);
        assert!(normal_cdf(-2.0) < 0.05);
        assert!(normal_cdf(2.0) > 0.95);
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 1e-12);
    }

    #[test]
    fn detector_configuration_is_reported() {
        let detector = StatisticalTestDetector::with_alpha(0.01);
        let config = <StatisticalTestDetector as crate::regression_tester::types::RegressionDetector<f64>>::config(&detector);
        assert_eq!(config.get("alpha"), Some(&"0.01".to_string()));

        let detector = SlidingWindowDetector::with_params(20, 10.0);
        let config = <SlidingWindowDetector as crate::regression_tester::types::RegressionDetector<f64>>::config(&detector);
        assert_eq!(config.get("window_size"), Some(&"20".to_string()));
        assert_eq!(config.get("threshold"), Some(&"10".to_string()));
    }
}
