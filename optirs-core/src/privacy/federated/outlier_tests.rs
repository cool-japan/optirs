// Statistical outlier tests used by Byzantine detection.
//
// Every test here works on a vector of per-client *test statistics* (in
// practice, each client's mean squared distance to the rest of the cohort)
// and decides which of them are anomalous. The tests differ in what they
// assume:
//
//   * [`StatisticalTestType::ZScore`] and
//     [`StatisticalTestType::ChauventCriterion`] assume approximate
//     normality and use the sample mean / sample standard deviation.
//   * [`StatisticalTestType::ModifiedZScore`] uses the median and the median
//     absolute deviation, so a few extreme clients cannot inflate the scale
//     estimate and mask each other (Iglewicz & Hoaglin, 1993).
//   * [`StatisticalTestType::IQRTest`] is fully non-parametric (Tukey fences).
//   * [`StatisticalTestType::GrubbsTest`] is the classical single-outlier
//     test, applied per client with a Bonferroni correction.
//
// References
// ----------
//   * Iglewicz, B. and Hoaglin, D. "How to Detect and Handle Outliers." ASQC
//     Basic References in Quality Control, vol. 16, 1993. (modified z-score,
//     the 0.6745 consistency constant and the 3.5 threshold)
//   * Grubbs, F. E. "Procedures for Detecting Outlying Observations in
//     Samples." Technometrics 11(1), 1969.
//   * Chauvenet, W. "A Manual of Spherical and Practical Astronomy," 1863.
//     (reject when the expected number of samples at least as extreme falls
//     below one half)
//   * Tukey, J. W. "Exploratory Data Analysis," 1977. (1.5 x IQR fences)

use super::byzantine_aggregation::StatisticalTestType;
use crate::error::{OptimError, Result};
use scirs2_stats::distributions::{norm, t as student_t, Normal};
use std::cmp::Ordering;

/// Consistency constant making the median absolute deviation an unbiased
/// estimator of sigma for normally distributed data (`Phi^-1(0.75)`).
const MAD_CONSISTENCY: f64 = 0.6745;

/// Iglewicz-Hoaglin cut-off for the modified z-score.
const MODIFIED_Z_THRESHOLD: f64 = 3.5;

/// Tukey fence multiplier.
const IQR_FENCE_MULTIPLIER: f64 = 1.5;

/// Outcome of testing a single sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlierVerdict {
    /// The test statistic for this sample. Its meaning depends on the test:
    /// a z-score, a modified z-score, IQRs from the median, or Grubbs' `G`.
    pub statistic: f64,

    /// Two-sided p-value, when the test defines one. `None` for the
    /// threshold-rule tests (modified z-score, IQR), where reporting a
    /// p-value would imply a distributional claim the rule does not make.
    pub p_value: Option<f64>,

    /// Whether the sample is flagged as an outlier.
    pub is_outlier: bool,
}

/// Evaluate `samples` against the configured test.
///
/// `reference` is the pool used to estimate location and scale. It is
/// normally the same slice as `samples`; when the aggregator's adaptive
/// threshold is enabled it additionally contains statistics retained from
/// previous rounds, so the decision boundary tracks the cohort's recent
/// behaviour instead of being re-derived from one round in isolation.
///
/// Returns one verdict per element of `samples`, in the same order.
pub fn evaluate(
    test_type: StatisticalTestType,
    samples: &[f64],
    reference: &[f64],
    significance_level: f64,
) -> Result<Vec<OutlierVerdict>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if reference.is_empty() {
        return Err(OptimError::InvalidConfig(
            "outlier tests need a non-empty reference sample".to_string(),
        ));
    }
    // Strictly inside (0, 1): alpha = 0 is a test with zero power that can
    // never flag anything, and alpha = 1 flags everything. Both are
    // configuration mistakes, not usable settings.
    if !significance_level.is_finite() || significance_level <= 0.0 || significance_level >= 1.0 {
        return Err(OptimError::InvalidConfig(format!(
            "significance level must be in (0, 1), got {significance_level}"
        )));
    }
    for value in samples.iter().chain(reference.iter()) {
        if !value.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "outlier tests require finite statistics, got {value}"
            )));
        }
    }

    match test_type {
        StatisticalTestType::ZScore => z_score(samples, reference, significance_level),
        StatisticalTestType::ModifiedZScore => modified_z_score(samples, reference),
        StatisticalTestType::IQRTest => iqr_test(samples, reference),
        StatisticalTestType::GrubbsTest => grubbs(samples, reference, significance_level),
        StatisticalTestType::ChauventCriterion => chauvenet(samples, reference),
    }
}

/// Two-sided z-test against the standard normal.
fn z_score(
    samples: &[f64],
    reference: &[f64],
    significance_level: f64,
) -> Result<Vec<OutlierVerdict>> {
    let centre = mean(reference);
    let scale = sample_std(reference);
    let normal = standard_normal()?;

    if scale <= 0.0 {
        return Ok(degenerate_verdicts(samples, centre));
    }
    Ok(samples
        .iter()
        .map(|&value| {
            let z = (value - centre) / scale;
            let p = two_sided_normal_tail(&normal, z);
            OutlierVerdict {
                statistic: z,
                p_value: Some(p),
                is_outlier: p < significance_level,
            }
        })
        .collect())
}

/// Iglewicz-Hoaglin modified z-score. Falls back to the mean absolute
/// deviation when the MAD is zero (which happens whenever more than half the
/// cohort submits identical statistics).
fn modified_z_score(samples: &[f64], reference: &[f64]) -> Result<Vec<OutlierVerdict>> {
    let centre = median(reference);
    let mad = median_absolute_deviation(reference, centre);

    let (scale, constant) = if mad > 0.0 {
        (mad, MAD_CONSISTENCY)
    } else {
        let mean_ad = reference
            .iter()
            .map(|value| (value - centre).abs())
            .sum::<f64>()
            / reference.len() as f64;
        // 1 / E|X - mu| for a standard normal is sqrt(pi / 2) ~= 1.253314.
        (mean_ad, 1.0 / 1.253_314_137_315_5)
    };

    if scale <= 0.0 {
        return Ok(degenerate_verdicts(samples, centre));
    }
    Ok(samples
        .iter()
        .map(|&value| {
            let m = constant * (value - centre) / scale;
            OutlierVerdict {
                statistic: m,
                p_value: None,
                is_outlier: m.abs() > MODIFIED_Z_THRESHOLD,
            }
        })
        .collect())
}

/// Tukey's 1.5 x IQR fences.
fn iqr_test(samples: &[f64], reference: &[f64]) -> Result<Vec<OutlierVerdict>> {
    let q1 = quantile(reference, 0.25);
    let q3 = quantile(reference, 0.75);
    let centre = median(reference);
    let iqr = q3 - q1;

    if iqr <= 0.0 {
        return Ok(degenerate_verdicts(samples, centre));
    }
    let lower = q1 - IQR_FENCE_MULTIPLIER * iqr;
    let upper = q3 + IQR_FENCE_MULTIPLIER * iqr;
    Ok(samples
        .iter()
        .map(|&value| OutlierVerdict {
            statistic: (value - centre) / iqr,
            p_value: None,
            is_outlier: value < lower || value > upper,
        })
        .collect())
}

/// Grubbs' test applied to every sample with a Bonferroni correction over the
/// `n` simultaneous comparisons.
///
/// Grubbs' `G = |x - mean| / s` is mapped back to the Student-t statistic it
/// is derived from,
///
/// ```text
/// t^2 = (n - 2) * (G/c)^2 / (1 - (G/c)^2),   c = (n - 1) / sqrt(n)
/// ```
///
/// whose two-sided tail probability is evaluated with the real Student-t CDF
/// and then multiplied by `n`.
fn grubbs(
    samples: &[f64],
    reference: &[f64],
    significance_level: f64,
) -> Result<Vec<OutlierVerdict>> {
    let n = reference.len();
    if n < 3 {
        return Err(OptimError::InvalidConfig(format!(
            "Grubbs' test needs at least 3 observations to have a defined critical value, got {n}"
        )));
    }
    let centre = mean(reference);
    let scale = sample_std(reference);
    if scale <= 0.0 {
        return Ok(degenerate_verdicts(samples, centre));
    }

    let df = (n - 2) as f64;
    let distribution = student_t(df, 0.0_f64, 1.0_f64).map_err(|error| {
        OptimError::ComputationError(format!(
            "Student-t distribution with {df} degrees of freedom unavailable: {error}"
        ))
    })?;
    let n_f = n as f64;
    let c = (n_f - 1.0) / n_f.sqrt();

    Ok(samples
        .iter()
        .map(|&value| {
            let g = (value - centre).abs() / scale;
            // `g` can only reach `c` in the degenerate case where a single
            // observation carries the whole deviation; clamp just below it so
            // the algebra below stays finite.
            let ratio = (g / c).min(1.0 - 1.0e-12);
            let t_squared = df * ratio * ratio / (1.0 - ratio * ratio);
            let t = t_squared.sqrt();
            let two_sided = 2.0 * (1.0 - distribution.cdf(t));
            let adjusted = (n_f * two_sided).clamp(0.0, 1.0);
            OutlierVerdict {
                statistic: g,
                p_value: Some(adjusted),
                is_outlier: adjusted < significance_level,
            }
        })
        .collect())
}

/// Chauvenet's criterion: reject when the expected number of observations at
/// least as extreme, `n * 2 * (1 - Phi(|z|))`, drops below one half.
fn chauvenet(samples: &[f64], reference: &[f64]) -> Result<Vec<OutlierVerdict>> {
    let n = reference.len() as f64;
    let centre = mean(reference);
    let scale = sample_std(reference);
    if scale <= 0.0 {
        return Ok(degenerate_verdicts(samples, centre));
    }
    let normal = standard_normal()?;

    Ok(samples
        .iter()
        .map(|&value| {
            let z = (value - centre) / scale;
            let tail = two_sided_normal_tail(&normal, z);
            OutlierVerdict {
                statistic: z,
                p_value: Some(tail),
                is_outlier: n * tail < 0.5,
            }
        })
        .collect())
}

/// When the reference sample has zero spread every observation is identical
/// (or the cohort is a single client): nothing can be an outlier, and the
/// statistic is reported as the deviation from the centre, which is zero.
fn degenerate_verdicts(samples: &[f64], centre: f64) -> Vec<OutlierVerdict> {
    samples
        .iter()
        .map(|&value| OutlierVerdict {
            statistic: value - centre,
            p_value: None,
            is_outlier: false,
        })
        .collect()
}

fn standard_normal() -> Result<Normal<f64>> {
    norm(0.0_f64, 1.0_f64).map_err(|error| {
        OptimError::ComputationError(format!("standard normal distribution unavailable: {error}"))
    })
}

/// Two-sided standard-normal tail probability `2 * (1 - Phi(|z|))`, clamped
/// into `[0, 1]` so downstream comparisons cannot see a value outside the
/// probability range because of floating-point round-off.
fn two_sided_normal_tail(normal: &Normal<f64>, z: f64) -> f64 {
    let upper: f64 = 1.0 - normal.cdf(z.abs());
    (2.0 * upper).clamp(0.0, 1.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (Bessel-corrected). Returns 0 for a single
/// observation, which every caller treats as "no spread".
fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let centre = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - centre) * (value - centre))
        .sum::<f64>()
        / (n - 1) as f64;
    variance.max(0.0).sqrt()
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    copy
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let ordered = sorted(values);
    let middle = ordered.len() / 2;
    if ordered.len().is_multiple_of(2) {
        0.5 * (ordered[middle - 1] + ordered[middle])
    } else {
        ordered[middle]
    }
}

fn median_absolute_deviation(values: &[f64], centre: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let deviations: Vec<f64> = values.iter().map(|value| (value - centre).abs()).collect();
    median(&deviations)
}

/// Linear-interpolation quantile (the "type 7" definition used by NumPy and R
/// by default).
fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let ordered = sorted(values);
    if ordered.len() == 1 {
        return ordered[0];
    }
    let position = q.clamp(0.0, 1.0) * (ordered.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return ordered[lower];
    }
    let weight = position - lower as f64;
    ordered[lower] * (1.0 - weight) + ordered[upper] * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nine tightly clustered values plus one far outlier.
    fn contaminated() -> Vec<f64> {
        let mut values: Vec<f64> = (0..9).map(|i| 10.0 + i as f64 * 0.1).collect();
        values.push(1000.0);
        values
    }

    #[test]
    fn z_score_flags_the_contaminant_and_reports_a_real_p_value() {
        let samples = contaminated();
        let verdicts =
            evaluate(StatisticalTestType::ZScore, &samples, &samples, 0.05).expect("z-score");
        assert_eq!(verdicts.len(), samples.len());
        assert!(verdicts[9].is_outlier, "the 1000.0 sample must be flagged");
        assert!(verdicts[..9].iter().all(|v| !v.is_outlier));
        let p = verdicts[9].p_value.expect("z-score defines a p-value");
        assert!((0.0..1.0).contains(&p));
        assert!(p < 0.05);
    }

    #[test]
    fn z_score_statistic_matches_the_hand_computed_value() {
        // mean 2, sample std sqrt(((1)^2+(0)^2+(1)^2)/2) = 1.
        let samples = vec![1.0, 2.0, 3.0];
        let verdicts =
            evaluate(StatisticalTestType::ZScore, &samples, &samples, 0.05).expect("z-score");
        assert!((verdicts[0].statistic + 1.0).abs() < 1e-12);
        assert!(verdicts[1].statistic.abs() < 1e-12);
        assert!((verdicts[2].statistic - 1.0).abs() < 1e-12);
    }

    #[test]
    fn modified_z_score_survives_masking_that_defeats_the_plain_z_score() {
        // Four extreme values inflate the sample standard deviation enough
        // that the plain z-score misses all of them -- the classic masking
        // effect. The median/MAD score is immune and catches every one.
        //
        // n = 14, mean = 147.86, sample sd = 241, so the largest |z| is
        // (530 - 147.86) / 241 = 1.59, whose two-sided p-value is 0.11.
        // Median = 1.025 and MAD = 0.15, so the smallest outlying modified
        // score is 0.6745 * (500 - 1.025) / 0.15 = 2243.
        let samples = vec![
            0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.05, 1.1, 1.15, 1.2, 500.0, 510.0, 520.0, 530.0,
        ];

        let plain =
            evaluate(StatisticalTestType::ZScore, &samples, &samples, 0.05).expect("z-score");
        let modified = evaluate(
            StatisticalTestType::ModifiedZScore,
            &samples,
            &samples,
            0.05,
        )
        .expect("modified z-score");

        assert!(
            plain.iter().all(|v| !v.is_outlier),
            "the plain z-score should be masked by four co-conspirators"
        );
        assert!(
            modified[10..].iter().all(|v| v.is_outlier),
            "the modified z-score must catch every outlier"
        );
        assert!(modified[..10].iter().all(|v| !v.is_outlier));
        // Threshold-rule tests do not claim a p-value.
        assert!(modified[10].p_value.is_none());
    }

    #[test]
    fn iqr_test_uses_tukey_fences() {
        // n = 6: Q1 = 2.25, Q3 = 4.75, IQR = 2.5, so the fences sit at -1.5
        // and 8.5.
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let verdicts =
            evaluate(StatisticalTestType::IQRTest, &samples, &samples, 0.05).expect("iqr");
        assert!(verdicts[5].is_outlier);
        assert!(verdicts[..5].iter().all(|v| !v.is_outlier));
        assert!(verdicts[0].p_value.is_none());
    }

    #[test]
    fn grubbs_flags_the_single_outlier_and_needs_three_observations() {
        let samples = contaminated();
        let verdicts =
            evaluate(StatisticalTestType::GrubbsTest, &samples, &samples, 0.05).expect("grubbs");
        assert!(verdicts[9].is_outlier);
        assert!(verdicts[..9].iter().all(|v| !v.is_outlier));
        let p = verdicts[9].p_value.expect("grubbs defines a p-value");
        assert!(
            p < 0.05,
            "expected a significant Bonferroni p-value, got {p}"
        );

        let tiny = vec![1.0, 2.0];
        let err = evaluate(StatisticalTestType::GrubbsTest, &tiny, &tiny, 0.05)
            .expect_err("two observations are not enough");
        assert!(format!("{err}").contains("at least 3 observations"));
    }

    #[test]
    fn grubbs_statistic_is_the_standardised_deviation() {
        let samples = vec![1.0, 2.0, 3.0];
        let verdicts =
            evaluate(StatisticalTestType::GrubbsTest, &samples, &samples, 0.05).expect("grubbs");
        // mean 2, s = 1 => G = 1, 0, 1.
        assert!((verdicts[0].statistic - 1.0).abs() < 1e-12);
        assert!(verdicts[1].statistic.abs() < 1e-12);
        assert!((verdicts[2].statistic - 1.0).abs() < 1e-12);
        // ... and nothing is significant in such a small clean sample.
        assert!(verdicts.iter().all(|v| !v.is_outlier));
    }

    #[test]
    fn chauvenet_rejects_when_fewer_than_half_an_observation_is_expected() {
        let samples = contaminated();
        let verdicts = evaluate(
            StatisticalTestType::ChauventCriterion,
            &samples,
            &samples,
            0.05,
        )
        .expect("chauvenet");
        assert!(verdicts[9].is_outlier);
        assert!(verdicts[..9].iter().all(|v| !v.is_outlier));

        // A clean normal-ish sample keeps everything.
        let clean = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98];
        let verdicts = evaluate(StatisticalTestType::ChauventCriterion, &clean, &clean, 0.05)
            .expect("chauvenet");
        assert!(verdicts.iter().all(|v| !v.is_outlier));
    }

    #[test]
    fn a_degenerate_reference_flags_nothing() {
        let samples = vec![5.0; 6];
        for test in [
            StatisticalTestType::ZScore,
            StatisticalTestType::ModifiedZScore,
            StatisticalTestType::IQRTest,
            StatisticalTestType::GrubbsTest,
            StatisticalTestType::ChauventCriterion,
        ] {
            let verdicts = evaluate(test, &samples, &samples, 0.05).expect("degenerate");
            assert!(
                verdicts.iter().all(|v| !v.is_outlier),
                "{test:?} flagged an outlier in a constant sample"
            );
        }
    }

    #[test]
    fn the_reference_pool_can_differ_from_the_samples() {
        // Historic rounds sat near 100; this round's client at 100 is normal
        // even though it is far from the current round's other two values.
        let samples = vec![1.0, 2.0, 100.0];
        let reference: Vec<f64> = (0..40).map(|i| 90.0 + i as f64 * 0.5).collect();
        let verdicts =
            evaluate(StatisticalTestType::ZScore, &samples, &reference, 0.05).expect("z-score");
        assert!(!verdicts[2].is_outlier, "100 is typical for the pool");
        assert!(verdicts[0].is_outlier && verdicts[1].is_outlier);
    }

    #[test]
    fn invalid_significance_levels_and_non_finite_samples_are_rejected() {
        let samples = vec![1.0, 2.0, 3.0];
        assert!(evaluate(StatisticalTestType::ZScore, &samples, &samples, 0.0).is_err());
        assert!(evaluate(StatisticalTestType::ZScore, &samples, &samples, 1.0).is_err());
        let bad = vec![1.0, f64::INFINITY];
        assert!(evaluate(StatisticalTestType::ZScore, &bad, &bad, 0.05).is_err());
    }

    #[test]
    fn quantile_matches_the_type_seven_definition() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&values, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&values, 0.25) - 1.75).abs() < 1e-12);
        assert!((quantile(&values, 0.5) - 2.5).abs() < 1e-12);
        assert!((quantile(&values, 0.75) - 3.25).abs() < 1e-12);
        assert!((quantile(&values, 1.0) - 4.0).abs() < 1e-12);
    }
}
