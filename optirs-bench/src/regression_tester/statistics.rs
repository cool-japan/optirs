// Statistical analysis implementations for regression testing
//
// This module provides statistical analyzers for identifying trends, outliers,
// and patterns in performance data to support regression detection.
//
// Both analyzers are NaN-safe: non-finite measurements are filtered out before
// any ordering or arithmetic happens, so a single corrupted sample can never
// panic a comparison or poison a percentile.

use crate::error::Result;
use crate::regression_tester::distributions::{
    self, linear_regression, sorted_finite, LinearTrend,
};
use crate::regression_tester::types::{
    PerformanceRecord, StatisticalAnalysisResult, StatisticalAnalyzer, StatisticalTestResult,
};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Relative total change (as a fraction of the mean) above which a trend is
/// called "strong", provided the correlation also supports it.
const STRONG_RELATIVE_CHANGE: f64 = 0.20;
/// Relative total change above which a trend is called "moderate".
const MODERATE_RELATIVE_CHANGE: f64 = 0.05;
/// Relative total change above which a trend is called "weak".
const WEAK_RELATIVE_CHANGE: f64 = 0.01;

/// Trend analysis implementation
///
/// Analyzes performance data to detect linear trends that may indicate
/// gradual performance improvements or degradations over time.
///
/// Trend strength is classified from the *relative* total change implied by the
/// fitted line (`slope * (n - 1) / mean`) rather than from the raw slope. The
/// raw slope carries the unit of the metric - a slope of "1" is enormous for a
/// ratio and invisible for a nanosecond timing - so thresholds expressed in raw
/// units silently mean different things for different metrics.
#[derive(Debug)]
pub struct TrendAnalyzer {
    /// Minimum number of data points required for analysis
    min_data_points: usize,
}

impl TrendAnalyzer {
    /// Create a new trend analyzer with default parameters
    pub fn new() -> Self {
        Self { min_data_points: 5 }
    }

    /// Create a new trend analyzer with custom minimum data points
    pub fn with_min_data_points(min_data_points: usize) -> Self {
        Self { min_data_points }
    }

    /// Classify a fitted trend into a (direction, strength) pair.
    fn classify(trend: &LinearTrend) -> (&'static str, &'static str) {
        let Some(relative_change) = trend.relative_total_change() else {
            return ("stable", "none");
        };
        let correlation = trend.correlation.abs();
        let magnitude = relative_change.abs();

        let strength = if magnitude > STRONG_RELATIVE_CHANGE && correlation > 0.7 {
            "strong"
        } else if magnitude > MODERATE_RELATIVE_CHANGE && correlation > 0.5 {
            "moderate"
        } else if magnitude > WEAK_RELATIVE_CHANGE && correlation > 0.3 {
            "weak"
        } else {
            return ("stable", "none");
        };

        let direction = match (relative_change > 0.0, strength) {
            (true, "strong") => "strongly increasing",
            (true, "moderate") => "increasing",
            (true, _) => "slightly increasing",
            (false, "strong") => "strongly decreasing",
            (false, "moderate") => "decreasing",
            (false, _) => "slightly decreasing",
        };

        (direction, strength)
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug + Send + Sync> StatisticalAnalyzer<A> for TrendAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.len() < self.min_data_points {
            return Ok(StatisticalAnalysisResult {
                summary: "Insufficient data for trend analysis".to_string(),
                tests: vec![],
                patterns: vec!["Insufficient data".to_string()],
                anomalies: vec![],
            });
        }

        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .filter(|value| value.is_finite())
            .collect();

        let Some(trend) = linear_regression(&times) else {
            return Ok(StatisticalAnalysisResult {
                summary: "Trend analysis undefined: degenerate or constant series".to_string(),
                tests: vec![],
                patterns: vec!["No usable variation in performance data".to_string()],
                anomalies: vec![],
            });
        };

        let (trend_direction, trend_strength) = Self::classify(&trend);
        let relative_change_percent = trend
            .relative_total_change()
            .map(|change| change * 100.0)
            .unwrap_or(0.0);
        let confidence_interval = trend.slope_confidence_interval(0.95);

        let mut patterns = vec![
            format!("Linear trend: {}", trend_direction),
            format!("Trend strength: {}", trend_strength),
            format!("Correlation coefficient: {:.3}", trend.correlation),
            format!("Relative change: {:.2}%", relative_change_percent),
            format!("Slope p-value: {:.6}", trend.p_value),
        ];
        if let Some((lower, upper)) = confidence_interval {
            patterns.push(format!("Slope 95% CI: [{:.4}, {:.4}]", lower, upper));
        }

        Ok(StatisticalAnalysisResult {
            summary: format!(
                "Trend analysis: {} trend with slope {:.2} (correlation: {:.3}, change: {:.2}%, p = {:.6})",
                trend_direction, trend.slope, trend.correlation, relative_change_percent, trend.p_value
            ),
            tests: vec![StatisticalTestResult {
                test_name: "ols_slope_t_test".to_string(),
                test_statistic: trend.t_statistic,
                p_value: trend.p_value,
                degrees_of_freedom: Some(trend.sample_count.saturating_sub(2)),
                conclusion: if trend.p_value < 0.05 {
                    format!("Slope differs significantly from zero ({})", trend_direction)
                } else {
                    "Slope is not significantly different from zero".to_string()
                },
            }],
            patterns,
            anomalies: vec![],
        })
    }

    fn name(&self) -> &str {
        "trend_analyzer"
    }
}

/// Outlier detection analyzer
///
/// Detects anomalous performance measurements that deviate significantly
/// from the expected distribution using statistical methods.
#[derive(Debug)]
pub struct OutlierAnalyzer {
    /// Z-score threshold for outlier detection (standard deviations)
    z_threshold: f64,
}

impl OutlierAnalyzer {
    /// Create a new outlier analyzer with default threshold
    pub fn new() -> Self {
        Self { z_threshold: 2.0 }
    }

    /// Create a new outlier analyzer with custom Z-score threshold
    pub fn with_threshold(z_threshold: f64) -> Self {
        Self { z_threshold }
    }
}

impl Default for OutlierAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug + Send + Sync> StatisticalAnalyzer<A> for OutlierAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.is_empty() {
            return Ok(StatisticalAnalysisResult {
                summary: "No data for outlier analysis".to_string(),
                tests: vec![],
                patterns: vec!["No data available".to_string()],
                anomalies: vec![],
            });
        }

        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .filter(|value| value.is_finite())
            .collect();

        if times.len() < 3 {
            return Ok(StatisticalAnalysisResult {
                summary: "Insufficient data for reliable outlier detection".to_string(),
                tests: vec![],
                patterns: vec!["Insufficient data for statistical analysis".to_string()],
                anomalies: vec![],
            });
        }

        let (Some(mean), Some(std_dev)) = (
            distributions::mean(&times),
            distributions::sample_std_dev(&times),
        ) else {
            return Ok(StatisticalAnalysisResult {
                summary: "Outlier analysis undefined for the supplied data".to_string(),
                tests: vec![],
                patterns: vec!["No usable variation in performance data".to_string()],
                anomalies: vec![],
            });
        };

        if std_dev == 0.0 {
            return Ok(StatisticalAnalysisResult {
                summary: "No variation in data - all values identical".to_string(),
                tests: vec![],
                patterns: vec!["Zero variance in performance data".to_string()],
                anomalies: vec![],
            });
        }

        let mut outliers = Vec::new();
        let mut severe_outliers = 0usize;
        let mut moderate_outliers = 0usize;

        for &time in &times {
            let z_score = (time - mean) / std_dev;
            let abs_z_score = z_score.abs();

            if abs_z_score > self.z_threshold {
                // `A::from` fails for values a narrower float cannot hold; skip
                // such scores rather than panicking on `expect`.
                let Some(score) = A::from(z_score) else {
                    continue;
                };
                if !score.is_finite() {
                    continue;
                }
                outliers.push(score);

                if abs_z_score > 3.0 {
                    severe_outliers += 1;
                } else {
                    moderate_outliers += 1;
                }
            }
        }

        let outlier_percentage = (outliers.len() as f64 / times.len() as f64) * 100.0;

        // NaN-safe ordering, then the shared interpolating percentile helper.
        let sorted_times = sorted_finite(&times);
        let q1 = distributions::percentile_sorted(&sorted_times, 25.0).unwrap_or(mean);
        let q3 = distributions::percentile_sorted(&sorted_times, 75.0).unwrap_or(mean);
        let iqr = q3 - q1;

        let severity = if severe_outliers > 0 {
            "severe"
        } else if moderate_outliers > 2 {
            "moderate"
        } else if outliers.len() > 1 {
            "mild"
        } else {
            "minimal"
        };

        let patterns = if outliers.is_empty() {
            vec!["No significant outliers detected".to_string()]
        } else {
            let mut patterns = vec![
                format!(
                    "{} potential outliers found ({:.1}% of data)",
                    outliers.len(),
                    outlier_percentage
                ),
                format!("Outlier severity: {}", severity),
                format!("Mean: {:.2}, Std Dev: {:.2}", mean, std_dev),
                format!("IQR: {:.2} (Q1: {:.2}, Q3: {:.2})", iqr, q1, q3),
            ];

            if severe_outliers > 0 {
                patterns.push(format!("{} severe outliers (|z| > 3.0)", severe_outliers));
            }
            if moderate_outliers > 0 {
                patterns.push(format!(
                    "{} moderate outliers ({:.1} < |z| <= 3.0)",
                    moderate_outliers, self.z_threshold
                ));
            }

            patterns
        };

        Ok(StatisticalAnalysisResult {
            summary: format!(
                "Outlier analysis: {} outliers detected ({:.1}% of data, {} severity)",
                outliers.len(),
                outlier_percentage,
                severity
            ),
            tests: vec![],
            patterns,
            anomalies: outliers,
        })
    }

    fn name(&self) -> &str {
        "outlier_analyzer"
    }
}

/// Helper functions for statistical calculations
///
/// These are thin, NaN-safe wrappers over
/// [`crate::regression_tester::distributions`].
pub mod stats_utils {
    use crate::regression_tester::distributions;

    /// Calculate the median of a sorted slice.
    ///
    /// Returns `0.0` for an empty slice to preserve the historical contract.
    pub fn median(sorted_data: &[f64]) -> f64 {
        distributions::median_sorted(sorted_data).unwrap_or(0.0)
    }

    /// Calculate the interpolated percentile (`p` in 0..=100) of sorted data.
    pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        distributions::percentile_sorted(sorted_data, p).unwrap_or(0.0)
    }

    /// Calculate the median absolute deviation of unsorted data.
    ///
    /// Non-finite values are dropped instead of panicking a `partial_cmp`.
    pub fn mad(data: &[f64]) -> f64 {
        let sorted_data = distributions::sorted_finite(data);
        if sorted_data.is_empty() {
            return 0.0;
        }
        let median_value = median(&sorted_data);

        let deviations: Vec<f64> = sorted_data
            .iter()
            .map(|&x| (x - median_value).abs())
            .collect();
        let sorted_deviations = distributions::sorted_finite(&deviations);

        median(&sorted_deviations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression_tester::config::TestEnvironment;
    use crate::regression_tester::types::PerformanceMetrics;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_record(time_ns: u64) -> PerformanceRecord<f64> {
        PerformanceRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default(),
            commit_hash: None,
            branch: None,
            environment: TestEnvironment::default(),
            metrics: PerformanceMetrics {
                timing: crate::regression_tester::types::TimingMetrics {
                    mean_time_ns: time_ns,
                    std_time_ns: 10,
                    median_time_ns: time_ns,
                    p95_time_ns: time_ns + 50,
                    p99_time_ns: time_ns + 100,
                    min_time_ns: time_ns.saturating_sub(20),
                    max_time_ns: time_ns + 200,
                },
                memory: Default::default(),
                efficiency: Default::default(),
                convergence: Default::default(),
                custom: HashMap::new(),
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn trend_analyzer_reports_increasing_trend_with_p_value() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        for i in 0..10 {
            data.push_back(create_test_record(1000 + i * 100));
        }

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert!(result.summary.contains("increasing"));
        assert!(result.patterns.iter().any(|p| p.contains("increasing")));
        let test = result.tests.first().expect("a slope test is reported");
        assert_eq!(test.test_name, "ols_slope_t_test");
        assert!(test.p_value < 1e-6, "p = {}", test.p_value);
        assert!(result.patterns.iter().any(|p| p.contains("Slope 95% CI")));
    }

    #[test]
    fn trend_analyzer_reports_decreasing_trend() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        for i in 0..10 {
            data.push_back(create_test_record(2000 - i * 100));
        }

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert!(result.summary.contains("decreasing"));
        assert!(result.patterns.iter().any(|p| p.contains("decreasing")));
    }

    /// F66: thresholds must not be expressed in raw nanoseconds. Two series
    /// with identical *relative* shape must be classified identically even when
    /// their absolute scale differs by nine orders of magnitude.
    #[test]
    fn trend_classification_is_scale_invariant() {
        let analyzer = TrendAnalyzer::new();

        let mut small = VecDeque::new();
        let mut large = VecDeque::new();
        for i in 0..10u64 {
            small.push_back(create_test_record(1000 + i * 100));
            large.push_back(create_test_record(1_000_000_000 + i * 100_000_000));
        }

        let small_result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&small).expect("analysis runs");
        let large_result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&large).expect("analysis runs");

        let strength_of = |result: &StatisticalAnalysisResult<f64>| {
            result
                .patterns
                .iter()
                .find(|p| p.starts_with("Trend strength:"))
                .cloned()
                .unwrap_or_default()
        };
        assert_eq!(strength_of(&small_result), strength_of(&large_result));
        assert!(strength_of(&small_result).contains("strong"));
    }

    /// A tiny absolute drift on a huge baseline is noise, not a trend.
    #[test]
    fn trend_analyzer_ignores_negligible_relative_drift() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();
        for i in 0..10u64 {
            data.push_back(create_test_record(1_000_000_000 + i));
        }

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");
        assert!(
            result.summary.contains("stable"),
            "summary was {}",
            result.summary
        );
    }

    #[test]
    fn trend_analyzer_requires_minimum_data() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        for i in 0..3 {
            data.push_back(create_test_record(1000 + i * 10));
        }

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert!(result.summary.contains("Insufficient data"));
        assert!(result.patterns.iter().any(|p| p.contains("Insufficient")));
    }

    #[test]
    fn outlier_analyzer_flags_extremes() {
        let analyzer = OutlierAnalyzer::new();
        let mut data = VecDeque::new();

        for _ in 0..10 {
            data.push_back(create_test_record(1000));
        }
        data.push_back(create_test_record(2000));
        data.push_back(create_test_record(500));

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert!(!result.anomalies.is_empty());
        assert!(result.summary.contains("outliers detected"));
        assert!(result.patterns.iter().any(|p| p.contains("IQR")));
    }

    #[test]
    fn outlier_analyzer_handles_constant_data() {
        let analyzer = OutlierAnalyzer::new();
        let mut data = VecDeque::new();

        for _ in 0..10 {
            data.push_back(create_test_record(1000));
        }

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert_eq!(result.anomalies.len(), 0);
        assert!(result
            .patterns
            .iter()
            .any(|p| p.contains("Zero variance in performance data")));
    }

    #[test]
    fn outlier_analyzer_handles_empty_data() {
        let analyzer = OutlierAnalyzer::new();
        let data = VecDeque::new();

        let result: StatisticalAnalysisResult<f64> =
            analyzer.analyze(&data).expect("analysis runs");

        assert!(result.summary.contains("No data"));
        assert!(result.patterns.iter().any(|p| p.contains("No data")));
    }

    #[test]
    fn custom_parameters_are_stored() {
        let trend_analyzer = TrendAnalyzer::with_min_data_points(3);
        assert_eq!(trend_analyzer.min_data_points, 3);

        let outlier_analyzer = OutlierAnalyzer::with_threshold(3.0);
        assert_eq!(outlier_analyzer.z_threshold, 3.0);
    }

    #[test]
    fn stats_utils_are_nan_safe() {
        use super::stats_utils::*;

        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[]), 0.0);

        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 100.0), 5.0);

        // The previous implementation panicked here via `partial_cmp().expect`.
        let with_nan = [3.0, f64::NAN, 1.0, 2.0, f64::INFINITY];
        assert_eq!(mad(&with_nan), 1.0);
        assert_eq!(mad(&[]), 0.0);
    }

    #[test]
    fn analyzer_names_are_stable() {
        let trend_analyzer = TrendAnalyzer::new();
        let outlier_analyzer = OutlierAnalyzer::new();

        assert_eq!(
            <TrendAnalyzer as crate::regression_tester::types::StatisticalAnalyzer<f64>>::name(
                &trend_analyzer
            ),
            "trend_analyzer"
        );
        assert_eq!(
            <OutlierAnalyzer as crate::regression_tester::types::StatisticalAnalyzer<f64>>::name(
                &outlier_analyzer
            ),
            "outlier_analyzer"
        );
    }
}
