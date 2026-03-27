//! Statistical analysis for evaluation results
//!
//! Provides statistical tests, analysis methods, and confidence intervals.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::benchmark::TestResult;
use super::types::*;
use crate::error::Result;
use crate::EvaluationMetric;

/// Statistical analyzer for evaluation results
#[derive(Debug)]
pub struct StatisticalAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Statistical tests
    statistical_tests: Vec<StatisticalTest<T>>,

    /// Analysis methods
    analysis_methods: Vec<AnalysisMethod>,

    /// Significance thresholds
    significance_thresholds: HashMap<String, T>,

    /// Multiple comparison correction
    multiple_comparison: MultipleComparisonCorrection,
}

/// Statistical test
#[derive(Debug)]
pub struct StatisticalTest<T: Float + Debug + Send + Sync + 'static> {
    /// Test name
    pub name: String,

    /// Test type
    pub test_type: StatisticalTestType,

    /// Test statistic
    pub test_statistic: T,

    /// P-value
    pub p_value: T,

    /// Effect size
    pub effect_size: T,

    /// Confidence interval
    pub confidence_interval: (T, T),
}

impl<T: Float + Debug + Default + std::iter::Sum + Send + Sync> StatisticalAnalyzer<T> {
    pub(crate) fn new() -> Self {
        Self {
            statistical_tests: Vec::new(),
            analysis_methods: vec![
                AnalysisMethod::DescriptiveStatistics,
                AnalysisMethod::CorrelationAnalysis,
            ],
            significance_thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert(
                    "alpha".to_string(),
                    scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(|| T::zero()),
                );
                thresholds
            },
            multiple_comparison: MultipleComparisonCorrection::BenjaminiHochberg,
        }
    }

    pub(crate) fn compute_confidence_intervals(
        &self,
        results: &[TestResult<T>],
    ) -> Result<HashMap<EvaluationMetric, (T, T)>> {
        let mut intervals = HashMap::new();

        if !results.is_empty() {
            let scores: Vec<T> = results.iter().map(|r| r.score).collect();
            let mean = scores.iter().cloned().sum::<T>()
                / T::from(scores.len()).expect("conversion failed");
            let std_dev = if scores.len() > 1 {
                let variance = scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<T>()
                    / T::from(scores.len() - 1).expect("conversion failed");
                variance.sqrt()
            } else {
                T::zero()
            };

            // 95% confidence interval (simplified)
            let margin = std_dev
                * scirs2_core::numeric::NumCast::from(1.96).unwrap_or_else(|| T::zero())
                / T::from((scores.len() as f64).sqrt()).expect("conversion failed");
            intervals.insert(
                EvaluationMetric::FinalPerformance,
                (mean - margin, mean + margin),
            );
        }

        Ok(intervals)
    }

    /// Add a statistical test
    pub fn add_test(&mut self, test: StatisticalTest<T>) {
        self.statistical_tests.push(test);
    }

    /// Get significance threshold
    pub fn get_threshold(&self, name: &str) -> Option<T> {
        self.significance_thresholds.get(name).copied()
    }

    /// Set significance threshold
    pub fn set_threshold(&mut self, name: String, value: T) {
        self.significance_thresholds.insert(name, value);
    }

    /// Compute descriptive statistics
    pub fn compute_descriptive_stats(&self, values: &[T]) -> DescriptiveStats<T> {
        if values.is_empty() {
            return DescriptiveStats {
                mean: T::zero(),
                median: T::zero(),
                std_dev: T::zero(),
                min: T::zero(),
                max: T::zero(),
                count: 0,
            };
        }

        let count = values.len();
        let sum: T = values.iter().cloned().sum();
        let mean = sum / T::from(count).expect("conversion failed");

        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<T>()
            / T::from(count.max(1)).expect("conversion failed");
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted.first().copied().unwrap_or(T::zero());
        let max = sorted.last().copied().unwrap_or(T::zero());
        let median = if count.is_multiple_of(2) {
            (sorted[count / 2 - 1] + sorted[count / 2])
                / scirs2_core::numeric::NumCast::from(2.0).unwrap_or_else(|| T::one())
        } else {
            sorted[count / 2]
        };

        DescriptiveStats {
            mean,
            median,
            std_dev,
            min,
            max,
            count,
        }
    }
}

/// Descriptive statistics result
#[derive(Debug, Clone)]
pub struct DescriptiveStats<T: Float + Debug + Send + Sync + 'static> {
    /// Mean value
    pub mean: T,
    /// Median value
    pub median: T,
    /// Standard deviation
    pub std_dev: T,
    /// Minimum value
    pub min: T,
    /// Maximum value
    pub max: T,
    /// Count
    pub count: usize,
}
