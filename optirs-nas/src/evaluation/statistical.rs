//! Statistical analysis for evaluation results
//!
//! Provides statistical tests, analysis methods, and confidence intervals.

use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use super::benchmark::TestResult;
use super::types::*;
use crate::error::{OptimError, Result};
use crate::numeric::{count_as, scalar_as};
use crate::EvaluationMetric;

/// Statistical analyzer for evaluation results
///
/// `analysis_methods` and `multiple_comparison` were configuration the analyzer
/// stored and never consulted: it always produced descriptive statistics and
/// never corrected a p-value, whatever they said. Both now drive behaviour —
/// see [`StatisticalAnalyzer::analyze`] and
/// [`StatisticalAnalyzer::adjusted_p_values`] — and the recorded
/// [`StatisticalTest`]s they operate on are readable through
/// [`StatisticalAnalyzer::tests`].
#[derive(Debug)]
pub struct StatisticalAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Statistical tests
    statistical_tests: Vec<StatisticalTest<T>>,

    /// Analyses [`StatisticalAnalyzer::analyze`] performs, in order.
    analysis_methods: Vec<AnalysisMethod>,

    /// Significance thresholds
    significance_thresholds: HashMap<String, T>,

    /// Family-wise / false-discovery correction applied by
    /// [`StatisticalAnalyzer::adjusted_p_values`].
    multiple_comparison: MultipleComparisonCorrection,
}

/// Result of running the analyzer's enabled analyses over one sample.
#[derive(Debug, Clone)]
pub struct AnalysisReport<T: Float + Debug + Send + Sync + 'static> {
    /// Descriptive statistics, present when
    /// [`AnalysisMethod::DescriptiveStatistics`] is enabled.
    pub descriptive: Option<DescriptiveStats<T>>,

    /// Lag-1 autocorrelation of the sample, present when
    /// [`AnalysisMethod::CorrelationAnalysis`] is enabled and the sample has at
    /// least two points and non-zero variance. `None` for a constant or
    /// too-short sample: an autocorrelation is undefined there, and reporting
    /// `0` would claim "no correlation" where the truth is "no answer".
    pub lag1_autocorrelation: Option<T>,
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
                // Unreachable for `f32`/`f64` (see `crate::numeric`). If a
                // caller's `Float` type did reject 0.05, a threshold of zero
                // admits nothing — the conservative direction. A fallback of one
                // would declare every test significant, which is precisely the
                // silent-wrong-number failure `crate::numeric` exists to avoid.
                thresholds.insert(
                    "alpha".to_string(),
                    scalar_as::<T>(0.05, "default significance level")
                        .unwrap_or_else(|_| T::zero()),
                );
                thresholds
            },
            multiple_comparison: MultipleComparisonCorrection::BenjaminiHochberg,
        }
    }

    /// Analyses this analyzer performs, in order.
    pub fn analysis_methods(&self) -> &[AnalysisMethod] {
        &self.analysis_methods
    }

    /// Choose which analyses [`StatisticalAnalyzer::analyze`] performs.
    ///
    /// Only the methods this analyzer actually implements are accepted. Enabling
    /// one it cannot perform is reported as [`OptimError::NotImplemented`] rather
    /// than accepted and silently skipped, which would let a caller believe a
    /// regression or cluster analysis had run.
    pub fn set_analysis_methods(&mut self, methods: Vec<AnalysisMethod>) -> Result<()> {
        for method in &methods {
            match method {
                AnalysisMethod::DescriptiveStatistics | AnalysisMethod::CorrelationAnalysis => {}
                other => {
                    return Err(OptimError::NotImplemented(format!(
                        "StatisticalAnalyzer implements DescriptiveStatistics and \
                         CorrelationAnalysis; {:?} has no implementation here",
                        other
                    )))
                }
            }
        }
        self.analysis_methods = methods;
        Ok(())
    }

    /// Run every enabled analysis over `values`.
    pub fn analyze(&self, values: &[T]) -> Result<AnalysisReport<T>> {
        let mut report = AnalysisReport {
            descriptive: None,
            lag1_autocorrelation: None,
        };
        for method in &self.analysis_methods {
            match method {
                AnalysisMethod::DescriptiveStatistics => {
                    report.descriptive = Some(self.compute_descriptive_stats(values)?);
                }
                AnalysisMethod::CorrelationAnalysis => {
                    report.lag1_autocorrelation = self.lag1_autocorrelation(values)?;
                }
                other => {
                    return Err(OptimError::NotImplemented(format!(
                        "StatisticalAnalyzer has no implementation for {:?}",
                        other
                    )))
                }
            }
        }
        Ok(report)
    }

    /// Lag-1 autocorrelation of `values`, or `None` when it is undefined.
    fn lag1_autocorrelation(&self, values: &[T]) -> Result<Option<T>> {
        if values.len() < 2 {
            return Ok(None);
        }
        let count: T = count_as(values.len(), "sample size")?;
        let mean = values.iter().cloned().sum::<T>() / count;

        let mut denominator = T::zero();
        for &value in values {
            let centered = value - mean;
            denominator = denominator + centered * centered;
        }
        if denominator <= T::zero() {
            // A constant series has no autocorrelation to report.
            return Ok(None);
        }

        let mut numerator = T::zero();
        for window in values.windows(2) {
            numerator = numerator + (window[0] - mean) * (window[1] - mean);
        }
        Ok(Some(numerator / denominator))
    }

    /// Recorded statistical tests, in the order they were added.
    pub fn tests(&self) -> &[StatisticalTest<T>] {
        &self.statistical_tests
    }

    /// Multiple-comparison correction applied by
    /// [`StatisticalAnalyzer::adjusted_p_values`].
    pub fn multiple_comparison(&self) -> &MultipleComparisonCorrection {
        &self.multiple_comparison
    }

    /// Choose the multiple-comparison correction.
    pub fn set_multiple_comparison(&mut self, correction: MultipleComparisonCorrection) {
        self.multiple_comparison = correction;
    }

    /// p-values of the recorded tests after the configured multiple-comparison
    /// correction, in the order the tests were added.
    ///
    /// Every correction is closed-form and clamped to `[0, 1]`:
    ///
    /// * `None` — the raw p-values.
    /// * `Bonferroni` — `m * p`.
    /// * `Sidak` — `1 - (1 - p)^m`.
    /// * `HolmBonferroni` — step-down `(m - i) * p`, made monotone.
    /// * `BenjaminiHochberg` — step-up `m * p / (i + 1)`, made monotone.
    /// * `BenjaminiYekutieli` — Benjamini-Hochberg scaled by the harmonic number
    ///   `sum_{k=1..m} 1/k`, which is the dependence-robust variant.
    pub fn adjusted_p_values(&self) -> Result<Vec<T>> {
        let m = self.statistical_tests.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        let raw: Vec<T> = self
            .statistical_tests
            .iter()
            .map(|test| test.p_value)
            .collect();
        let count: T = count_as(m, "number of statistical tests")?;

        let adjusted = match self.multiple_comparison {
            MultipleComparisonCorrection::None => raw,
            MultipleComparisonCorrection::Bonferroni => {
                raw.into_iter().map(|p| (p * count).min(T::one())).collect()
            }
            MultipleComparisonCorrection::Sidak => raw
                .into_iter()
                .map(|p| {
                    let complement = (T::one() - p).max(T::zero());
                    (T::one() - complement.powf(count)).min(T::one())
                })
                .collect(),
            MultipleComparisonCorrection::HolmBonferroni => Self::step_down(&raw, |rank, p| {
                let factor: T = count_as((m - rank).max(1), "Holm rank factor")?;
                Ok((p * factor).min(T::one()))
            })?,
            MultipleComparisonCorrection::BenjaminiHochberg => {
                Self::step_up(&raw, count, T::one())?
            }
            MultipleComparisonCorrection::BenjaminiYekutieli => {
                let mut harmonic = T::zero();
                for k in 1..=m {
                    let denominator: T = count_as(k, "harmonic-series term")?;
                    harmonic = harmonic + T::one() / denominator;
                }
                Self::step_up(&raw, count, harmonic)?
            }
        };
        Ok(adjusted)
    }

    /// Indices of the tests whose adjusted p-value is at or below `alpha`.
    pub fn significant_tests(&self, alpha: T) -> Result<Vec<usize>> {
        let adjusted = self.adjusted_p_values()?;
        Ok(adjusted
            .into_iter()
            .enumerate()
            .filter(|(_, p)| *p <= alpha)
            .map(|(index, _)| index)
            .collect())
    }

    /// Shared machinery for the step-down (Holm) family: sort ascending, scale by
    /// rank, then enforce monotonicity by carrying the running maximum forward.
    fn step_down<F>(raw: &[T], scale: F) -> Result<Vec<T>>
    where
        F: Fn(usize, T) -> Result<T>,
    {
        let mut order: Vec<usize> = (0..raw.len()).collect();
        order.sort_by(|&a, &b| raw[a].partial_cmp(&raw[b]).unwrap_or(Ordering::Equal));

        let mut adjusted = vec![T::zero(); raw.len()];
        let mut running = T::zero();
        for (rank, &index) in order.iter().enumerate() {
            let scaled = scale(rank, raw[index])?;
            running = running.max(scaled);
            adjusted[index] = running.min(T::one());
        }
        Ok(adjusted)
    }

    /// Shared machinery for the step-up (Benjamini) family: sort descending,
    /// scale by `m * factor / rank`, then enforce monotonicity by carrying the
    /// running minimum backwards.
    fn step_up(raw: &[T], count: T, factor: T) -> Result<Vec<T>> {
        let mut order: Vec<usize> = (0..raw.len()).collect();
        order.sort_by(|&a, &b| raw[b].partial_cmp(&raw[a]).unwrap_or(Ordering::Equal));

        let mut adjusted = vec![T::zero(); raw.len()];
        let mut running = T::one();
        for (position, &index) in order.iter().enumerate() {
            let rank = raw.len() - position;
            let rank_value: T = count_as(rank, "Benjamini rank")?;
            let scaled = raw[index] * count * factor / rank_value;
            running = running.min(scaled);
            adjusted[index] = running.min(T::one()).max(T::zero());
        }
        Ok(adjusted)
    }

    pub(crate) fn compute_confidence_intervals(
        &self,
        results: &[TestResult<T>],
    ) -> Result<HashMap<EvaluationMetric, (T, T)>> {
        let mut intervals = HashMap::new();

        if !results.is_empty() {
            let scores: Vec<T> = results.iter().map(|r| r.score).collect();
            let count: T = count_as(scores.len(), "confidence-interval sample size")?;
            let mean = scores.iter().cloned().sum::<T>() / count;
            let std_dev = if scores.len() > 1 {
                let degrees_of_freedom: T =
                    count_as(scores.len() - 1, "confidence-interval degrees of freedom")?;
                let variance =
                    scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<T>() / degrees_of_freedom;
                variance.sqrt()
            } else {
                T::zero()
            };

            // 95% confidence interval (normal approximation).
            let z: T = scalar_as(1.96, "95% normal quantile")?;
            let margin = std_dev * z / count.sqrt();
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

    /// Compute descriptive statistics.
    ///
    /// An empty sample yields an all-zero summary with `count == 0`, which the
    /// caller can distinguish from a genuine all-zero sample. Fallible because the
    /// sample size has to reach the element type to be a divisor; there is no
    /// fallback divisor that would not silently report a different statistic.
    pub fn compute_descriptive_stats(&self, values: &[T]) -> Result<DescriptiveStats<T>> {
        if values.is_empty() {
            return Ok(DescriptiveStats {
                mean: T::zero(),
                median: T::zero(),
                std_dev: T::zero(),
                min: T::zero(),
                max: T::zero(),
                count: 0,
            });
        }

        let count = values.len();
        let count_t: T = count_as(count, "descriptive sample size")?;
        let mean = sum_of(values) / count_t;

        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<T>() / count_t;
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

        Ok(DescriptiveStats {
            mean,
            median,
            std_dev,
            min,
            max,
            count,
        })
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

/// Sum of a sample, factored out so the descriptive path reads the values once.
fn sum_of<T: Float + std::iter::Sum>(values: &[T]) -> T {
    values.iter().cloned().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_with_p(name: &str, p_value: f64) -> StatisticalTest<f64> {
        StatisticalTest {
            name: name.to_string(),
            test_type: StatisticalTestType::TTest,
            test_statistic: 0.0,
            p_value,
            effect_size: 0.0,
            confidence_interval: (0.0, 0.0),
        }
    }

    fn analyzer_with(ps: &[f64]) -> StatisticalAnalyzer<f64> {
        let mut analyzer = StatisticalAnalyzer::<f64>::new();
        for (index, &p) in ps.iter().enumerate() {
            analyzer.add_test(test_with_p(&format!("t{index}"), p));
        }
        analyzer
    }

    #[test]
    fn no_correction_returns_the_raw_p_values() {
        let mut analyzer = analyzer_with(&[0.01, 0.04, 0.20]);
        analyzer.set_multiple_comparison(MultipleComparisonCorrection::None);
        let adjusted = analyzer.adjusted_p_values().expect("adjustment");
        assert_eq!(adjusted, vec![0.01, 0.04, 0.20]);
    }

    #[test]
    fn bonferroni_and_sidak_multiply_by_the_family_size() {
        let mut analyzer = analyzer_with(&[0.01, 0.04, 0.20]);

        analyzer.set_multiple_comparison(MultipleComparisonCorrection::Bonferroni);
        let bonferroni = analyzer.adjusted_p_values().expect("adjustment");
        assert!((bonferroni[0] - 0.03).abs() < 1e-12);
        assert!((bonferroni[1] - 0.12).abs() < 1e-12);
        // Clamped at one rather than reporting a probability above 1.
        assert!((bonferroni[2] - 0.6).abs() < 1e-12);

        analyzer.set_multiple_comparison(MultipleComparisonCorrection::Sidak);
        let sidak = analyzer.adjusted_p_values().expect("adjustment");
        for (s, b) in sidak.iter().zip(bonferroni.iter()) {
            // Sidak is uniformly less conservative than Bonferroni.
            assert!(*s <= *b + 1e-12, "sidak {s} exceeded bonferroni {b}");
            assert!((0.0..=1.0).contains(s));
        }
    }

    #[test]
    fn holm_is_monotone_and_less_conservative_than_bonferroni() {
        let mut analyzer = analyzer_with(&[0.20, 0.01, 0.04]);
        analyzer.set_multiple_comparison(MultipleComparisonCorrection::Bonferroni);
        let bonferroni = analyzer.adjusted_p_values().expect("adjustment");

        analyzer.set_multiple_comparison(MultipleComparisonCorrection::HolmBonferroni);
        let holm = analyzer.adjusted_p_values().expect("adjustment");

        // Smallest raw p (0.01, index 1) is scaled by 3, the next by 2, the last
        // by 1 — then made monotone.
        assert!((holm[1] - 0.03).abs() < 1e-12);
        assert!((holm[2] - 0.08).abs() < 1e-12);
        assert!((holm[0] - 0.20).abs() < 1e-12);
        for (h, b) in holm.iter().zip(bonferroni.iter()) {
            assert!(*h <= *b + 1e-12);
        }
    }

    #[test]
    fn benjamini_hochberg_is_monotone_and_yekutieli_is_stricter() {
        let mut analyzer = analyzer_with(&[0.001, 0.008, 0.039, 0.041, 0.042]);

        analyzer.set_multiple_comparison(MultipleComparisonCorrection::BenjaminiHochberg);
        let bh = analyzer.adjusted_p_values().expect("adjustment");
        // Ordering is preserved: a smaller raw p never gets a larger adjusted p.
        for window in bh.windows(2) {
            assert!(
                window[0] <= window[1] + 1e-12,
                "BH broke monotonicity: {bh:?}"
            );
        }
        // 0.001 * 5 / 1 = 0.005.
        assert!((bh[0] - 0.005).abs() < 1e-12);

        analyzer.set_multiple_comparison(MultipleComparisonCorrection::BenjaminiYekutieli);
        let by = analyzer.adjusted_p_values().expect("adjustment");
        for (y, h) in by.iter().zip(bh.iter()) {
            assert!(*y >= *h - 1e-12, "Yekutieli must not be looser than BH");
            assert!((0.0..=1.0).contains(y));
        }
    }

    #[test]
    fn significance_uses_the_adjusted_values_not_the_raw_ones() {
        let mut analyzer = analyzer_with(&[0.02, 0.03, 0.04]);
        analyzer.set_multiple_comparison(MultipleComparisonCorrection::None);
        assert_eq!(
            analyzer.significant_tests(0.05).expect("significance"),
            vec![0, 1, 2]
        );

        // Under Bonferroni every adjusted value is above 0.05, so nothing survives.
        analyzer.set_multiple_comparison(MultipleComparisonCorrection::Bonferroni);
        assert!(analyzer
            .significant_tests(0.05)
            .expect("significance")
            .is_empty());

        assert_eq!(analyzer.tests().len(), 3);
        assert_eq!(
            analyzer.multiple_comparison(),
            &MultipleComparisonCorrection::Bonferroni
        );
    }

    #[test]
    fn an_empty_family_has_no_adjusted_values() {
        let analyzer = StatisticalAnalyzer::<f64>::new();
        assert!(analyzer.adjusted_p_values().expect("adjustment").is_empty());
        assert!(analyzer
            .significant_tests(0.05)
            .expect("significance")
            .is_empty());
    }

    #[test]
    fn analyze_runs_exactly_the_enabled_methods() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new();
        let series = [1.0, 2.0, 3.0, 4.0, 5.0];

        let report = analyzer.analyze(&series).expect("analysis");
        let descriptive = report.descriptive.expect("descriptive statistics enabled");
        assert_eq!(descriptive.count, 5);
        assert!((descriptive.mean - 3.0).abs() < 1e-12);
        let autocorrelation = report
            .lag1_autocorrelation
            .expect("correlation analysis enabled");
        // Standard (biased) lag-1 estimator on 1..=5: numerator 4, denominator 10.
        assert!(
            (autocorrelation - 0.4).abs() < 1e-12,
            "got {autocorrelation}"
        );

        // An alternating series is negatively autocorrelated, so the sign is real
        // signal and not an artefact of the ramp.
        let alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let flipped = analyzer
            .analyze(&alternating)
            .expect("analysis")
            .lag1_autocorrelation
            .expect("correlation analysis enabled");
        assert!(flipped < 0.0, "got {flipped}");

        analyzer
            .set_analysis_methods(vec![AnalysisMethod::DescriptiveStatistics])
            .expect("descriptive statistics are implemented");
        let report = analyzer.analyze(&series).expect("analysis");
        assert!(report.descriptive.is_some());
        assert!(
            report.lag1_autocorrelation.is_none(),
            "a disabled analysis must not run"
        );
    }

    #[test]
    fn an_unimplemented_analysis_is_rejected_not_silently_skipped() {
        let mut analyzer = StatisticalAnalyzer::<f64>::new();
        let err = analyzer
            .set_analysis_methods(vec![AnalysisMethod::PrincipalComponentAnalysis])
            .expect_err("PCA has no implementation here");
        assert!(matches!(err, OptimError::NotImplemented(_)));
        // The rejected configuration must not have been stored.
        assert_eq!(analyzer.analysis_methods().len(), 2);
    }

    #[test]
    fn autocorrelation_is_undefined_for_short_or_constant_samples() {
        let analyzer = StatisticalAnalyzer::<f64>::new();
        assert!(analyzer
            .analyze(&[])
            .expect("analysis")
            .lag1_autocorrelation
            .is_none());
        assert!(analyzer
            .analyze(&[1.0])
            .expect("analysis")
            .lag1_autocorrelation
            .is_none());
        assert!(
            analyzer
                .analyze(&[2.0, 2.0, 2.0])
                .expect("analysis")
                .lag1_autocorrelation
                .is_none(),
            "a constant series has no autocorrelation to report"
        );
    }

    #[test]
    fn confidence_intervals_survive_degenerate_samples() {
        let analyzer = StatisticalAnalyzer::<f64>::new();
        // No results at all: no interval, and no panic from an empty divisor.
        assert!(analyzer
            .compute_confidence_intervals(&[])
            .expect("intervals")
            .is_empty());
    }
}
