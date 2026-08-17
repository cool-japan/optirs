// Pure-Rust statistical distributions, hypothesis tests and signal-processing
// primitives shared by the benchmarking and regression-detection modules.
//
// Everything in this module is dependency free (no BLAS/LAPACK, no FFI) and is
// written so that the numerical results can be checked against published
// reference values. The unit tests at the bottom of this file pin the accuracy
// of every routine against values computed from standard statistical tables.
//
// Numerical background:
//
// * The incomplete gamma / incomplete beta functions follow the classic
//   continued-fraction and series expansions (Lentz's algorithm) which converge
//   to machine precision for the argument ranges used here.
// * `student_t_sf`, `f_distribution_sf` and `chi_squared_sf` are exact
//   expressions in terms of those special functions - not the "saturating"
//   ad-hoc approximations that previously prevented p-values from ever
//   dropping below 0.24.

use std::cmp::Ordering;

/// Natural log of the absolute value of the gamma function.
///
/// Lanczos approximation with `g = 7` and 9 coefficients; relative accuracy is
/// better than 1e-13 over the range used by the distribution functions.
pub fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if !x.is_finite() {
        return f64::NAN;
    }

    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let sin_term = (std::f64::consts::PI * x).sin().abs();
        if sin_term == 0.0 {
            return f64::INFINITY;
        }
        std::f64::consts::PI.ln() - sin_term.ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut series = COEFFICIENTS[0];
        for (i, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
            series += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
    }
}

/// Regularized lower incomplete gamma function `P(a, x)`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if !(a.is_finite() && x.is_finite()) || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_continued_fraction(a, x)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if !(a.is_finite() && x.is_finite()) || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1000;
    const EPSILON: f64 = 1e-16;

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut term = sum;
    for _ in 0..MAX_ITERATIONS {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * EPSILON {
            break;
        }
    }
    (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1000;
    const EPSILON: f64 = 1e-16;
    const TINY: f64 = 1e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITERATIONS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Error function, accurate to ~1e-14.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let magnitude = gamma_p(0.5, x * x);
    if x > 0.0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Complementary error function, accurate in both tails.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        gamma_q(0.5, x * x)
    } else {
        1.0 + gamma_p(0.5, x * x)
    }
}

/// Standard normal cumulative distribution function.
pub fn normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Standard normal survival function `P(Z > z)`.
pub fn normal_sf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Two-sided p-value for a standard normal test statistic.
pub fn normal_two_sided_p(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    (2.0 * normal_sf(z.abs())).clamp(0.0, 1.0)
}

/// Inverse of the standard normal CDF (probit function).
///
/// Acklam's rational approximation refined with a single Halley step, giving
/// full double precision.
pub fn normal_quantile(p: f64) -> Option<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return None;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement step against the exact CDF.
    let error = normal_cdf(x) - p;
    let density = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if density > 0.0 {
        let u = error / density;
        x -= u / (1.0 + 0.5 * x * u);
    }

    Some(x)
}

/// Continued-fraction evaluation of the incomplete beta function (Lentz).
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 500;
    const EPSILON: f64 = 1e-16;
    const TINY: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let numerator = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let numerator = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }

    h
}

/// Regularized incomplete beta function `I_x(a, b)`.
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if !(a.is_finite() && b.is_finite() && x.is_finite()) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Student's t cumulative distribution function `P(T <= t)`.
pub fn student_t_cdf(t: f64, degrees_of_freedom: f64) -> f64 {
    if !(t.is_finite() && degrees_of_freedom.is_finite()) || degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }
    let tail = 0.5
        * regularized_incomplete_beta(
            degrees_of_freedom / 2.0,
            0.5,
            degrees_of_freedom / (degrees_of_freedom + t * t),
        );
    if t > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Student's t survival function `P(T > t)`.
pub fn student_t_sf(t: f64, degrees_of_freedom: f64) -> f64 {
    1.0 - student_t_cdf(t, degrees_of_freedom)
}

/// Two-sided p-value for a Student's t statistic.
///
/// Uses the exact identity `p = I_{df/(df+t^2)}(df/2, 1/2)`, which is stable
/// even for very large `|t|` (where the p-value underflows smoothly to zero).
pub fn t_two_sided_p(t: f64, degrees_of_freedom: f64) -> f64 {
    if !t.is_finite() || !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }
    let x = degrees_of_freedom / (degrees_of_freedom + t * t);
    regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x).clamp(0.0, 1.0)
}

/// Inverse Student's t CDF: the value `t` with `P(T <= t) = p`.
///
/// Solved by bisection on [`student_t_cdf`], which is monotone.
///
/// Note on accuracy: [`student_t_cdf`] is parameterised through
/// `x = df / (df + t^2)`, which rounds to exactly `1` once `t^2` falls below
/// `eps * df`. The CDF therefore has a numerical plateau of half-width
/// `~sqrt(eps * df)` (about 3e-8 for `df = 5`) around `t = 0`. The median is
/// short-circuited by symmetry so the common cases stay exact; quantiles that
/// land inside the plateau are accurate to that half-width in `t`, i.e. to
/// better than 1e-8 in probability.
pub fn t_quantile(p: f64, degrees_of_freedom: f64) -> Option<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return None;
    }
    if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return None;
    }
    if p == 0.5 {
        // The t distribution is symmetric about zero for every df.
        return Some(0.0);
    }

    // Bracket the root; widen aggressively for heavy tails (small df).
    let mut low = -1.0;
    let mut high = 1.0;
    for _ in 0..200 {
        if student_t_cdf(low, degrees_of_freedom) <= p {
            break;
        }
        low *= 2.0;
        if !low.is_finite() {
            return None;
        }
    }
    for _ in 0..200 {
        if student_t_cdf(high, degrees_of_freedom) >= p {
            break;
        }
        high *= 2.0;
        if !high.is_finite() {
            return None;
        }
    }

    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if mid == low || mid == high {
            break;
        }
        if student_t_cdf(mid, degrees_of_freedom) < p {
            low = mid;
        } else {
            high = mid;
        }
    }

    Some(0.5 * (low + high))
}

/// Upper-tail p-value of the F distribution, `P(F > f)`.
pub fn f_distribution_sf(f: f64, df_numerator: f64, df_denominator: f64) -> f64 {
    if !f.is_finite() || !df_numerator.is_finite() || !df_denominator.is_finite() {
        return f64::NAN;
    }
    if df_numerator <= 0.0 || df_denominator <= 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    let x = df_denominator / (df_denominator + df_numerator * f);
    regularized_incomplete_beta(df_denominator / 2.0, df_numerator / 2.0, x).clamp(0.0, 1.0)
}

/// Upper-tail p-value of the chi-squared distribution, `P(X > x)`.
pub fn chi_squared_sf(x: f64, degrees_of_freedom: f64) -> f64 {
    if !x.is_finite() || !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(degrees_of_freedom / 2.0, x / 2.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Descriptive statistics (NaN-safe)
// ---------------------------------------------------------------------------

/// NaN-safe ascending comparison usable with `sort_by`.
pub fn compare_finite(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// Drop non-finite values and return the remainder sorted ascending.
pub fn sorted_finite(values: &[f64]) -> Vec<f64> {
    let mut filtered: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    filtered.sort_by(compare_finite);
    filtered
}

/// Arithmetic mean of the finite values, or `None` when there are none.
pub fn mean(values: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Unbiased sample variance (n-1 denominator). Requires at least two finite
/// observations.
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return None;
    }
    let m = mean(&finite)?;
    let sum_squares: f64 = finite.iter().map(|v| (v - m) * (v - m)).sum();
    Some(sum_squares / (finite.len() - 1) as f64)
}

/// Unbiased sample standard deviation.
pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(f64::sqrt)
}

/// Linear-interpolated percentile of an already sorted slice.
///
/// `p` is expressed in percent (0..=100).
pub fn percentile_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !p.is_finite() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let clamped = p.clamp(0.0, 100.0);
    let index = (clamped / 100.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        Some(sorted[lower])
    } else {
        let weight = index - lower as f64;
        Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
    }
}

/// Median of an already sorted slice.
pub fn median_sorted(sorted: &[f64]) -> Option<f64> {
    percentile_sorted(sorted, 50.0)
}

/// Interquartile range of an already sorted slice.
pub fn iqr_sorted(sorted: &[f64]) -> Option<f64> {
    let q1 = percentile_sorted(sorted, 25.0)?;
    let q3 = percentile_sorted(sorted, 75.0)?;
    Some(q3 - q1)
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------

/// Result of a t-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTestResult {
    /// t statistic.
    pub t_statistic: f64,
    /// Degrees of freedom used for the p-value.
    pub degrees_of_freedom: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// Welch's unequal-variance t-test from raw samples.
///
/// Returns `None` when either sample has fewer than two finite observations or
/// when both samples are constant (the statistic would be undefined).
pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64]) -> Option<TTestResult> {
    let a: Vec<f64> = sample_a.iter().copied().filter(|v| v.is_finite()).collect();
    let b: Vec<f64> = sample_b.iter().copied().filter(|v| v.is_finite()).collect();
    let mean_a = mean(&a)?;
    let mean_b = mean(&b)?;
    let var_a = sample_variance(&a)?;
    let var_b = sample_variance(&b)?;
    welch_t_test_summary(mean_a, var_a, a.len(), mean_b, var_b, b.len())
}

/// Welch's t-test from summary statistics (means, variances, sample sizes).
pub fn welch_t_test_summary(
    mean_a: f64,
    variance_a: f64,
    n_a: usize,
    mean_b: f64,
    variance_b: f64,
    n_b: usize,
) -> Option<TTestResult> {
    if n_a < 2 || n_b < 2 {
        return None;
    }
    if !(mean_a.is_finite()
        && mean_b.is_finite()
        && variance_a.is_finite()
        && variance_b.is_finite())
    {
        return None;
    }
    if variance_a < 0.0 || variance_b < 0.0 {
        return None;
    }

    let n_a_f = n_a as f64;
    let n_b_f = n_b as f64;
    let term_a = variance_a / n_a_f;
    let term_b = variance_b / n_b_f;
    let standard_error = (term_a + term_b).sqrt();

    if !(standard_error.is_finite()) || standard_error <= 0.0 {
        // Both groups are constant: the difference is either exactly zero
        // (no evidence of a difference) or deterministic (no variance to test).
        return None;
    }

    let t_statistic = (mean_a - mean_b) / standard_error;
    let denominator = term_a * term_a / (n_a_f - 1.0) + term_b * term_b / (n_b_f - 1.0);
    let degrees_of_freedom = if denominator > 0.0 {
        (term_a + term_b).powi(2) / denominator
    } else {
        n_a_f + n_b_f - 2.0
    };

    Some(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value: t_two_sided_p(t_statistic, degrees_of_freedom),
    })
}

/// One-sample t-test of `sample` against the hypothesised mean `mu`.
pub fn one_sample_t_test(sample: &[f64], mu: f64) -> Option<TTestResult> {
    let finite: Vec<f64> = sample.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 || !mu.is_finite() {
        return None;
    }
    let m = mean(&finite)?;
    let sd = sample_std_dev(&finite)?;
    let n = finite.len() as f64;
    let standard_error = sd / n.sqrt();
    if standard_error <= 0.0 || !standard_error.is_finite() {
        return None;
    }
    let t_statistic = (m - mu) / standard_error;
    let degrees_of_freedom = n - 1.0;
    Some(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value: t_two_sided_p(t_statistic, degrees_of_freedom),
    })
}

/// Test whether a single new observation is consistent with a reference
/// distribution summarised by its mean, standard deviation and sample size.
///
/// The standard error used is the *prediction* standard error
/// `sigma * sqrt(1 + 1/n)`, which accounts both for the spread of individual
/// observations and for the uncertainty in the estimated mean. Using
/// `sigma / sqrt(n)` (the standard error *of the mean*) here would be wrong:
/// a single observation is not a mean.
pub fn single_observation_t_test(
    observation: f64,
    reference_mean: f64,
    reference_std_dev: f64,
    reference_sample_count: usize,
) -> Option<TTestResult> {
    if !(observation.is_finite() && reference_mean.is_finite() && reference_std_dev.is_finite()) {
        return None;
    }
    if reference_std_dev <= 0.0 || reference_sample_count < 2 {
        return None;
    }
    let n = reference_sample_count as f64;
    let prediction_error = reference_std_dev * (1.0 + 1.0 / n).sqrt();
    if prediction_error <= 0.0 || !prediction_error.is_finite() {
        return None;
    }
    let t_statistic = (observation - reference_mean) / prediction_error;
    let degrees_of_freedom = n - 1.0;
    Some(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value: t_two_sided_p(t_statistic, degrees_of_freedom),
    })
}

/// Result of a Mann-Whitney U test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MannWhitneyResult {
    /// U statistic of the first sample.
    pub u_statistic: f64,
    /// Normal-approximation z score (continuity corrected).
    pub z_score: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// Assign average (mid-)ranks to a slice, returning the ranks in the original
/// order together with the tie group sizes.
fn average_ranks(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| compare_finite(&values[i], &values[j]));

    let mut ranks = vec![0.0; values.len()];
    let mut tie_sizes = Vec::new();
    let mut position = 0usize;
    while position < order.len() {
        let mut end = position + 1;
        while end < order.len() && values[order[end]] == values[order[position]] {
            end += 1;
        }
        let group = end - position;
        // Ranks are 1-based; the average rank of positions [position, end).
        let average = (position + 1 + end) as f64 / 2.0;
        for &index in &order[position..end] {
            ranks[index] = average;
        }
        if group > 1 {
            tie_sizes.push(group);
        }
        position = end;
    }

    (ranks, tie_sizes)
}

/// Mann-Whitney U test using the normal approximation with tie correction.
///
/// Returns `None` when either sample is empty or the tie-corrected variance
/// collapses to zero (all observations identical).
pub fn mann_whitney_u(sample_a: &[f64], sample_b: &[f64]) -> Option<MannWhitneyResult> {
    let a: Vec<f64> = sample_a.iter().copied().filter(|v| v.is_finite()).collect();
    let b: Vec<f64> = sample_b.iter().copied().filter(|v| v.is_finite()).collect();
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let total = n1 + n2;

    let mut combined = Vec::with_capacity(a.len() + b.len());
    combined.extend_from_slice(&a);
    combined.extend_from_slice(&b);
    let (ranks, tie_sizes) = average_ranks(&combined);

    let rank_sum_a: f64 = ranks[..a.len()].iter().sum();
    let u_statistic = rank_sum_a - n1 * (n1 + 1.0) / 2.0;

    let expected = n1 * n2 / 2.0;
    let tie_term: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let variance = if total > 1.0 {
        (n1 * n2 / 12.0) * ((total + 1.0) - tie_term / (total * (total - 1.0)))
    } else {
        0.0
    };

    if !(variance.is_finite()) || variance <= 0.0 {
        return None;
    }

    let difference = u_statistic - expected;
    // Continuity correction pulls the statistic half a unit toward the mean.
    let corrected = if difference > 0.5 {
        difference - 0.5
    } else if difference < -0.5 {
        difference + 0.5
    } else {
        0.0
    };
    let z_score = corrected / variance.sqrt();

    Some(MannWhitneyResult {
        u_statistic,
        z_score,
        p_value: normal_two_sided_p(z_score),
    })
}

/// Result of a Mann-Kendall trend test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MannKendallResult {
    /// Kendall S statistic.
    pub s_statistic: f64,
    /// Variance of S including the tie correction.
    pub variance: f64,
    /// Standardised test statistic.
    pub z_score: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Kendall's tau-b.
    pub tau: f64,
}

/// Mann-Kendall trend test with the standard tie correction on `Var(S)`.
pub fn mann_kendall(values: &[f64]) -> Option<MannKendallResult> {
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = data.len();
    if n < 3 {
        return None;
    }

    let mut s_statistic = 0.0f64;
    for i in 0..n - 1 {
        for j in i + 1..n {
            if data[j] > data[i] {
                s_statistic += 1.0;
            } else if data[j] < data[i] {
                s_statistic -= 1.0;
            }
        }
    }

    let (_, tie_sizes) = average_ranks(&data);
    let n_f = n as f64;
    let tie_correction: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * (t - 1.0) * (2.0 * t + 5.0)
        })
        .sum();
    let variance = (n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) - tie_correction) / 18.0;

    if !(variance.is_finite()) || variance <= 0.0 {
        return None;
    }

    let z_score = if s_statistic > 0.0 {
        (s_statistic - 1.0) / variance.sqrt()
    } else if s_statistic < 0.0 {
        (s_statistic + 1.0) / variance.sqrt()
    } else {
        0.0
    };

    // tau-b: normalise by the tie-adjusted number of comparable pairs.
    let total_pairs = n_f * (n_f - 1.0) / 2.0;
    let tie_pairs: f64 = tie_sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * (t - 1.0) / 2.0
        })
        .sum();
    let denominator = ((total_pairs - tie_pairs) * total_pairs).sqrt();
    let tau = if denominator > 0.0 {
        s_statistic / denominator
    } else {
        0.0
    };

    Some(MannKendallResult {
        s_statistic,
        variance,
        z_score,
        p_value: normal_two_sided_p(z_score),
        tau,
    })
}

/// Result of a Ljung-Box test for residual autocorrelation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LjungBoxResult {
    /// Q statistic.
    pub q_statistic: f64,
    /// Degrees of freedom (number of lags tested).
    pub degrees_of_freedom: f64,
    /// Upper-tail p-value.
    pub p_value: f64,
}

/// Sample autocorrelation at `lag` for a mean-centred series.
pub fn autocorrelation(values: &[f64], lag: usize) -> Option<f64> {
    if lag == 0 {
        return Some(1.0);
    }
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if data.len() <= lag + 1 {
        return None;
    }
    let m = mean(&data)?;
    let denominator: f64 = data.iter().map(|v| (v - m) * (v - m)).sum();
    if denominator <= 0.0 {
        return None;
    }
    let numerator: f64 = (lag..data.len())
        .map(|i| (data[i] - m) * (data[i - lag] - m))
        .sum();
    Some(numerator / denominator)
}

/// Ljung-Box portmanteau test for autocorrelation up to `lags`.
pub fn ljung_box(values: &[f64], lags: usize) -> Option<LjungBoxResult> {
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = data.len();
    if lags == 0 || n <= lags + 1 {
        return None;
    }

    let n_f = n as f64;
    let mut q_statistic = 0.0;
    for lag in 1..=lags {
        let rho = autocorrelation(&data, lag)?;
        let denominator = n_f - lag as f64;
        if denominator <= 0.0 {
            return None;
        }
        q_statistic += rho * rho / denominator;
    }
    q_statistic *= n_f * (n_f + 2.0);

    let degrees_of_freedom = lags as f64;
    Some(LjungBoxResult {
        q_statistic,
        degrees_of_freedom,
        p_value: chi_squared_sf(q_statistic, degrees_of_freedom),
    })
}

// ---------------------------------------------------------------------------
// Regression / trend analysis
// ---------------------------------------------------------------------------

/// Ordinary-least-squares fit of `y` on `x` with inference on the slope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearTrend {
    /// Fitted slope.
    pub slope: f64,
    /// Fitted intercept.
    pub intercept: f64,
    /// Pearson correlation coefficient between `x` and `y`.
    pub correlation: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// Standard error of the slope estimate.
    pub slope_std_error: f64,
    /// t statistic for `H0: slope == 0`.
    pub t_statistic: f64,
    /// Residual degrees of freedom (`n - 2`).
    pub degrees_of_freedom: f64,
    /// Two-sided p-value for the slope.
    pub p_value: f64,
    /// Number of observations used.
    pub sample_count: usize,
    /// Mean of the response, useful for normalising the slope.
    pub mean_y: f64,
}

impl LinearTrend {
    /// Confidence interval for the slope at the requested confidence level
    /// (e.g. `0.95`).
    pub fn slope_confidence_interval(&self, confidence_level: f64) -> Option<(f64, f64)> {
        if !(0.0..1.0).contains(&confidence_level) || confidence_level <= 0.0 {
            return None;
        }
        let alpha = 1.0 - confidence_level;
        let critical = t_quantile(1.0 - alpha / 2.0, self.degrees_of_freedom)?;
        let margin = critical * self.slope_std_error;
        Some((self.slope - margin, self.slope + margin))
    }

    /// Total relative change implied by the fitted line over the observed
    /// range, expressed as a fraction of the mean response.
    ///
    /// This makes slope thresholds dimensionless: a slope of 5 ns/sample on a
    /// 1000 ns baseline and a slope of 5 s/sample on a 1000 s baseline both map
    /// to the same relative change.
    pub fn relative_total_change(&self) -> Option<f64> {
        if self.sample_count < 2 || self.mean_y == 0.0 || !self.mean_y.is_finite() {
            return None;
        }
        let span = (self.sample_count - 1) as f64;
        Some(self.slope * span / self.mean_y.abs())
    }
}

/// Fit `y` against the supplied `x` values.
pub fn linear_regression_xy(x: &[f64], y: &[f64]) -> Option<LinearTrend> {
    if x.len() != y.len() {
        return None;
    }
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pairs.len();
    if n < 3 {
        return None;
    }

    let n_f = n as f64;
    let mean_x = pairs.iter().map(|(a, _)| *a).sum::<f64>() / n_f;
    let mean_y = pairs.iter().map(|(_, b)| *b).sum::<f64>() / n_f;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (a, b) in &pairs {
        let dx = a - mean_x;
        let dy = b - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if sxx <= 0.0 {
        return None;
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let correlation = if syy > 0.0 {
        (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let residual_sum_squares = (syy - slope * sxy).max(0.0);
    let degrees_of_freedom = n_f - 2.0;
    let residual_variance = residual_sum_squares / degrees_of_freedom;
    let slope_std_error = (residual_variance / sxx).sqrt();

    let (t_statistic, p_value) = if slope_std_error > 0.0 && slope_std_error.is_finite() {
        let t = slope / slope_std_error;
        (t, t_two_sided_p(t, degrees_of_freedom))
    } else if slope == 0.0 {
        // Perfectly flat series: no evidence of a trend.
        (0.0, 1.0)
    } else {
        // Perfect fit with a non-zero slope: the trend is certain.
        (f64::INFINITY, 0.0)
    };

    Some(LinearTrend {
        slope,
        intercept,
        correlation,
        r_squared: correlation * correlation,
        slope_std_error,
        t_statistic,
        degrees_of_freedom,
        p_value,
        sample_count: n,
        mean_y,
    })
}

/// Fit `y` against the implicit index `0, 1, 2, ...`.
pub fn linear_regression(y: &[f64]) -> Option<LinearTrend> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    linear_regression_xy(&x, y)
}

/// Pearson correlation coefficient.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    linear_regression_xy(x, y).map(|trend| trend.correlation)
}

// ---------------------------------------------------------------------------
// Change point detection
// ---------------------------------------------------------------------------

/// A detected change point in a series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangePoint {
    /// Index of the first sample belonging to the *second* segment.
    pub index: usize,
    /// Difference between the segment means (`after - before`).
    pub magnitude: f64,
    /// Relative magnitude with respect to the mean of the earlier segment.
    pub relative_magnitude: f64,
    /// Welch t statistic comparing the two segments.
    pub t_statistic: f64,
    /// Two-sided p-value of the segment comparison.
    pub p_value: f64,
}

fn segment_sum_of_squares(
    prefix_sum: &[f64],
    prefix_square_sum: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    let n = (end - start) as f64;
    if n <= 0.0 {
        return 0.0;
    }
    let sum = prefix_sum[end] - prefix_sum[start];
    let square_sum = prefix_square_sum[end] - prefix_square_sum[start];
    (square_sum - sum * sum / n).max(0.0)
}

/// Binary segmentation change-point detection.
///
/// At each step the split maximising the reduction in within-segment sum of
/// squares (equivalently `n_l*n_r/(n_l+n_r) * (mean_l - mean_r)^2`) is located,
/// then accepted only when a Welch t-test between the two candidate segments is
/// significant at `significance_level`. Recursion continues into both halves
/// until no segment is long enough for two blocks of `min_segment_size`.
///
/// The returned change points are the argmax split locations - not a fixed
/// midpoint.
pub fn binary_segmentation(
    values: &[f64],
    min_segment_size: usize,
    significance_level: f64,
    max_change_points: usize,
) -> Vec<ChangePoint> {
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mut results = Vec::new();
    let min_segment_size = min_segment_size.max(2);

    if data.len() < 2 * min_segment_size || max_change_points == 0 {
        return results;
    }

    let mut prefix_sum = vec![0.0; data.len() + 1];
    let mut prefix_square_sum = vec![0.0; data.len() + 1];
    for (i, &v) in data.iter().enumerate() {
        prefix_sum[i + 1] = prefix_sum[i] + v;
        prefix_square_sum[i + 1] = prefix_square_sum[i] + v * v;
    }

    let mut pending = vec![(0usize, data.len())];
    while let Some((start, end)) = pending.pop() {
        if results.len() >= max_change_points {
            break;
        }
        if end - start < 2 * min_segment_size {
            continue;
        }

        let total_ss = segment_sum_of_squares(&prefix_sum, &prefix_square_sum, start, end);
        let mut best_split = None;
        let mut best_gain = 0.0;
        for split in (start + min_segment_size)..=(end - min_segment_size) {
            let left = segment_sum_of_squares(&prefix_sum, &prefix_square_sum, start, split);
            let right = segment_sum_of_squares(&prefix_sum, &prefix_square_sum, split, end);
            let gain = total_ss - left - right;
            if gain > best_gain {
                best_gain = gain;
                best_split = Some(split);
            }
        }

        let Some(split) = best_split else {
            continue;
        };

        let before = &data[start..split];
        let after = &data[split..end];
        let Some(test) = welch_t_test(after, before) else {
            continue;
        };
        if !(test.p_value < significance_level) {
            continue;
        }

        let (Some(mean_before), Some(mean_after)) = (mean(before), mean(after)) else {
            continue;
        };
        let magnitude = mean_after - mean_before;
        let relative_magnitude = if mean_before.abs() > 0.0 {
            magnitude / mean_before.abs()
        } else {
            0.0
        };

        results.push(ChangePoint {
            index: split,
            magnitude,
            relative_magnitude,
            t_statistic: test.t_statistic,
            p_value: test.p_value,
        });

        pending.push((start, split));
        pending.push((split, end));
    }

    results.sort_by_key(|cp| cp.index);
    results
}

/// CUSUM change-point detection on the mean-centred series.
///
/// Returns the indices where the cumulative sum reaches a local extremum that
/// exceeds `threshold_sigmas` standard deviations of the cumulative statistic.
pub fn cusum_change_points(values: &[f64], threshold_sigmas: f64) -> Vec<usize> {
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mut change_points = Vec::new();
    if data.len() < 4 {
        return change_points;
    }
    let Some(m) = mean(&data) else {
        return change_points;
    };
    let Some(sd) = sample_std_dev(&data) else {
        return change_points;
    };
    if sd <= 0.0 {
        return change_points;
    }

    let n = data.len() as f64;
    // Standard deviation of the cumulative sum at its widest point.
    let cumulative_sigma = sd * (n / 4.0).sqrt();
    let threshold = threshold_sigmas * cumulative_sigma;

    let mut cumulative = 0.0;
    let mut best_index = 0usize;
    let mut best_magnitude = 0.0;
    for (i, &value) in data.iter().enumerate() {
        cumulative += value - m;
        if cumulative.abs() > best_magnitude {
            best_magnitude = cumulative.abs();
            best_index = i;
        }
    }

    if best_magnitude > threshold {
        change_points.push(best_index);
    }

    change_points
}

// ---------------------------------------------------------------------------
// Signal processing
// ---------------------------------------------------------------------------

/// In-place iterative radix-2 Cooley-Tukey FFT.
///
/// `real` and `imaginary` must have equal length and that length must be a
/// power of two; the function returns `false` otherwise and leaves the buffers
/// untouched.
pub fn fft_in_place(real: &mut [f64], imaginary: &mut [f64]) -> bool {
    let n = real.len();
    if n != imaginary.len() || n == 0 || !n.is_power_of_two() {
        return false;
    }
    if n == 1 {
        return true;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            real.swap(i, j);
            imaginary.swap(i, j);
        }
    }

    let mut length = 2usize;
    while length <= n {
        let angle = -2.0 * std::f64::consts::PI / length as f64;
        let (step_sin, step_cos) = angle.sin_cos();
        let mut start = 0usize;
        while start < n {
            let mut w_real = 1.0f64;
            let mut w_imaginary = 0.0f64;
            for offset in 0..length / 2 {
                let even = start + offset;
                let odd = even + length / 2;

                let odd_real = real[odd] * w_real - imaginary[odd] * w_imaginary;
                let odd_imaginary = real[odd] * w_imaginary + imaginary[odd] * w_real;

                real[odd] = real[even] - odd_real;
                imaginary[odd] = imaginary[even] - odd_imaginary;
                real[even] += odd_real;
                imaginary[even] += odd_imaginary;

                let next_real = w_real * step_cos - w_imaginary * step_sin;
                w_imaginary = w_real * step_sin + w_imaginary * step_cos;
                w_real = next_real;
            }
            start += length;
        }
        length <<= 1;
    }

    true
}

/// Magnitude spectrum of a real signal, zero padded to the next power of two.
///
/// Returns the first `n/2` bins (the non-redundant half) or `None` for an empty
/// or entirely non-finite input.
pub fn magnitude_spectrum(signal: &[f64]) -> Option<Vec<f64>> {
    if signal.is_empty() || signal.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let padded_length = signal.len().next_power_of_two();
    let mut real = vec![0.0; padded_length];
    let mut imaginary = vec![0.0; padded_length];
    real[..signal.len()].copy_from_slice(signal);

    if !fft_in_place(&mut real, &mut imaginary) {
        return None;
    }

    let half = (padded_length / 2).max(1);
    Some(
        (0..half)
            .map(|k| (real[k] * real[k] + imaginary[k] * imaginary[k]).sqrt())
            .collect(),
    )
}

/// One level of the discrete wavelet transform.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveletLevel {
    /// Approximation (low-pass) coefficients.
    pub approximation: Vec<f64>,
    /// Detail (high-pass) coefficients.
    pub detail: Vec<f64>,
}

/// Supported wavelet families for [`discrete_wavelet_transform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Haar (Daubechies-1) wavelet.
    Haar,
    /// Daubechies-4 (two vanishing moments) wavelet.
    Daubechies4,
}

fn wavelet_filters(family: WaveletFamily) -> Vec<f64> {
    match family {
        WaveletFamily::Haar => {
            let c = std::f64::consts::FRAC_1_SQRT_2;
            vec![c, c]
        }
        WaveletFamily::Daubechies4 => {
            let sqrt3 = 3.0f64.sqrt();
            let denominator = 4.0 * std::f64::consts::SQRT_2;
            vec![
                (1.0 + sqrt3) / denominator,
                (3.0 + sqrt3) / denominator,
                (3.0 - sqrt3) / denominator,
                (1.0 - sqrt3) / denominator,
            ]
        }
    }
}

/// Multi-level discrete wavelet transform with periodic boundary extension.
///
/// Each level halves the signal length; decomposition stops early when the
/// remaining approximation is shorter than the filter.
pub fn discrete_wavelet_transform(
    signal: &[f64],
    family: WaveletFamily,
    levels: usize,
) -> Option<Vec<WaveletLevel>> {
    if signal.len() < 2 || levels == 0 || signal.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let low_pass = wavelet_filters(family);
    let filter_length = low_pass.len();
    // Quadrature mirror filter: g[k] = (-1)^k * h[L-1-k]
    let high_pass: Vec<f64> = (0..filter_length)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * low_pass[filter_length - 1 - k]
        })
        .collect();

    let mut current: Vec<f64> = signal.to_vec();
    let mut decomposition = Vec::new();

    for _ in 0..levels {
        if current.len() < filter_length || current.len() < 2 {
            break;
        }
        // Work on an even-length signal (drop nothing: pad by repeating last).
        if current.len() % 2 == 1 {
            let last = *current.last()?;
            current.push(last);
        }

        let half = current.len() / 2;
        let mut approximation = Vec::with_capacity(half);
        let mut detail = Vec::with_capacity(half);
        for i in 0..half {
            let mut a = 0.0;
            let mut d = 0.0;
            for k in 0..filter_length {
                let index = (2 * i + k) % current.len();
                a += low_pass[k] * current[index];
                d += high_pass[k] * current[index];
            }
            approximation.push(a);
            detail.push(d);
        }

        current = approximation.clone();
        decomposition.push(WaveletLevel {
            approximation,
            detail,
        });
    }

    if decomposition.is_empty() {
        None
    } else {
        Some(decomposition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
        (actual - expected).abs() <= tolerance
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        // Gamma(5) = 24, Gamma(0.5) = sqrt(pi)
        assert!(close(ln_gamma(5.0), 24.0f64.ln(), 1e-12));
        assert!(close(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            1e-12
        ));
        assert!(close(ln_gamma(1.0), 0.0, 1e-12));
    }

    #[test]
    fn erf_matches_reference_values() {
        assert!(close(erf(0.0), 0.0, 1e-15));
        assert!(close(erf(1.0), 0.842_700_792_949_715, 1e-12));
        assert!(close(erf(-1.0), -0.842_700_792_949_715, 1e-12));
        assert!(close(erf(2.0), 0.995_322_265_018_953, 1e-12));
    }

    #[test]
    fn normal_cdf_matches_reference_values() {
        assert!(close(normal_cdf(0.0), 0.5, 1e-14));
        assert!(close(normal_cdf(1.0), 0.841_344_746_068_543, 1e-12));
        assert!(close(normal_cdf(-1.96), 0.024_997_895_148_220, 1e-12));
        assert!(close(normal_two_sided_p(1.96), 0.049_995_790_296_44, 1e-10));
    }

    #[test]
    fn normal_quantile_inverts_the_cdf() {
        for &p in &[0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999] {
            let z = normal_quantile(p).expect("quantile defined on (0,1)");
            assert!(close(normal_cdf(z), p, 1e-12), "p = {}", p);
        }
        assert!(normal_quantile(0.0).is_none());
        assert!(normal_quantile(1.0).is_none());
        assert!(normal_quantile(f64::NAN).is_none());
    }

    #[test]
    fn incomplete_beta_matches_reference_values() {
        // I_0.5(2, 3) = 0.6875 exactly.
        assert!(close(
            regularized_incomplete_beta(2.0, 3.0, 0.5),
            0.6875,
            1e-12
        ));
        assert!(close(
            regularized_incomplete_beta(1.0, 1.0, 0.25),
            0.25,
            1e-12
        ));
    }

    /// The regression this module exists for: the previous
    /// `t_distribution_cdf` saturated at ~0.876, so `p_two` could never fall
    /// below ~0.248 and no t-test could ever report significance.
    #[test]
    fn student_t_two_sided_p_matches_tables() {
        // df = 10, t = 3.0  =>  p = 0.013343 (standard t table).
        let p = t_two_sided_p(3.0, 10.0);
        assert!(close(p, 0.013_343, 1e-5), "p = {}", p);

        // df = 10, t = 2.228 is the 5% critical value.
        assert!(close(t_two_sided_p(2.228, 10.0), 0.05, 1e-4));

        // Large statistics must reach the deep tail, not saturate.
        let extreme = t_two_sided_p(100.0, 10.0);
        assert!(extreme < 1e-6, "p = {}", extreme);
        assert!(extreme > 0.0);

        // Reachability of the usual alpha levels.
        assert!(t_two_sided_p(3.0, 10.0) < 0.05);
        assert!(t_two_sided_p(1.0, 10.0) > 0.05);
    }

    #[test]
    fn student_t_cdf_is_symmetric_and_monotone() {
        assert!(close(student_t_cdf(0.0, 7.0), 0.5, 1e-12));
        assert!(close(
            student_t_cdf(-1.5, 7.0),
            1.0 - student_t_cdf(1.5, 7.0),
            1e-12
        ));
        assert!(student_t_cdf(1.0, 7.0) > student_t_cdf(0.5, 7.0));
        // Large df approaches the normal distribution.
        assert!(close(student_t_cdf(1.96, 1.0e6), normal_cdf(1.96), 1e-5));
    }

    #[test]
    fn t_quantile_matches_tables() {
        // Two-sided 95% critical value with 10 df is 2.228.
        let critical = t_quantile(0.975, 10.0).expect("defined");
        assert!(close(critical, 2.228, 1e-3), "critical = {}", critical);
        // Two-sided 95% critical value with 1 df is 12.706.
        let heavy = t_quantile(0.975, 1.0).expect("defined");
        assert!(close(heavy, 12.706, 1e-3), "critical = {}", heavy);
        assert!(t_quantile(0.5, 5.0)
            .map(|v| v.abs() < 1e-9)
            .unwrap_or(false));
        assert!(t_quantile(0.975, 0.0).is_none());
    }

    #[test]
    fn f_distribution_survival_matches_tables() {
        // F(3, 12) critical value at alpha = 0.05 is 3.4903.
        let p = f_distribution_sf(3.4903, 3.0, 12.0);
        assert!(close(p, 0.05, 1e-4), "p = {}", p);
        assert!(f_distribution_sf(1.0, 3.0, 12.0) > 0.05);
        assert!(f_distribution_sf(100.0, 3.0, 12.0) < 1e-6);
        assert!(f_distribution_sf(0.0, 3.0, 12.0) == 1.0);
    }

    #[test]
    fn chi_squared_survival_matches_tables() {
        // chi^2 with 3 df: the 95th percentile is 7.8147.
        assert!(close(chi_squared_sf(7.8147, 3.0), 0.05, 1e-4));
        assert!(close(chi_squared_sf(0.0, 3.0), 1.0, 1e-12));
    }

    #[test]
    fn welch_t_test_detects_a_shift() {
        let baseline = [100.0, 101.0, 99.0, 100.5, 100.2, 99.8, 100.1];
        let regressed = [120.0, 121.0, 119.5, 120.4, 120.1, 119.9, 120.3];
        let test = welch_t_test(&regressed, &baseline).expect("test defined");
        assert!(test.t_statistic > 0.0);
        assert!(test.p_value < 1e-6, "p = {}", test.p_value);

        let same = welch_t_test(&baseline, &baseline).expect("test defined");
        assert!(close(same.t_statistic, 0.0, 1e-12));
        assert!(same.p_value > 0.99);
    }

    #[test]
    fn welch_t_test_rejects_degenerate_inputs() {
        assert!(welch_t_test(&[1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(welch_t_test(&[], &[]).is_none());
        assert!(welch_t_test(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]).is_none());
        assert!(welch_t_test(&[f64::NAN, f64::NAN], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn single_observation_test_uses_prediction_error() {
        // sigma = 10, n = 25: prediction SE = 10 * sqrt(1 + 1/25) = 10.198.
        let test = single_observation_t_test(120.0, 100.0, 10.0, 25).expect("defined");
        assert!(close(
            test.t_statistic,
            20.0 / (10.0 * (1.04f64).sqrt()),
            1e-12
        ));
        assert!(test.p_value < 0.1);
        // Guards.
        assert!(single_observation_t_test(1.0, 1.0, 0.0, 25).is_none());
        assert!(single_observation_t_test(1.0, 1.0, 1.0, 0).is_none());
        assert!(single_observation_t_test(f64::NAN, 1.0, 1.0, 5).is_none());
    }

    #[test]
    fn mann_whitney_separates_disjoint_samples() {
        let low = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let high = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let result = mann_whitney_u(&high, &low).expect("defined");
        assert!(close(result.u_statistic, 36.0, 1e-12));
        assert!(result.p_value < 0.01, "p = {}", result.p_value);

        // Identical samples cannot be separated.
        let overlap = mann_whitney_u(&low, &low).expect("defined");
        assert!(overlap.p_value > 0.9, "p = {}", overlap.p_value);
    }

    #[test]
    fn mann_whitney_handles_ties() {
        let a = [1.0, 1.0, 1.0, 2.0];
        let b = [1.0, 1.0, 2.0, 2.0];
        let result = mann_whitney_u(&a, &b).expect("defined");
        assert!(result.p_value > 0.05);
        assert!(result.p_value <= 1.0);
        // A completely constant pooled sample has zero variance.
        assert!(mann_whitney_u(&[1.0, 1.0], &[1.0, 1.0]).is_none());
        assert!(mann_whitney_u(&[], &[1.0]).is_none());
    }

    #[test]
    fn mann_kendall_detects_monotone_trends() {
        let increasing: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = mann_kendall(&increasing).expect("defined");
        assert!(result.s_statistic > 0.0);
        assert!(result.p_value < 0.001, "p = {}", result.p_value);
        assert!(close(result.tau, 1.0, 1e-9));

        let decreasing: Vec<f64> = (0..20).map(|i| -(i as f64)).collect();
        let down = mann_kendall(&decreasing).expect("defined");
        assert!(down.s_statistic < 0.0);
        assert!(down.p_value < 0.001);

        // Constant series: variance collapses to the tie-corrected zero.
        assert!(mann_kendall(&[5.0; 10]).is_none());
        assert!(mann_kendall(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn ljung_box_flags_autocorrelation() {
        // Strongly autocorrelated ramp.
        let ramp: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let correlated = ljung_box(&ramp, 5).expect("defined");
        assert!(correlated.p_value < 0.01, "p = {}", correlated.p_value);

        // Alternating series has strong negative lag-1 autocorrelation, which
        // the test must also flag.
        let alternating: Vec<f64> = (0..60)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let alt = ljung_box(&alternating, 4).expect("defined");
        assert!(alt.p_value < 0.01);

        assert!(ljung_box(&[1.0, 2.0], 5).is_none());
        assert!(ljung_box(&ramp, 0).is_none());
    }

    #[test]
    fn linear_regression_reports_slope_inference() {
        let y: Vec<f64> = (0..12).map(|i| 100.0 + 2.0 * i as f64).collect();
        let trend = linear_regression(&y).expect("defined");
        assert!(close(trend.slope, 2.0, 1e-9));
        assert!(close(trend.correlation, 1.0, 1e-9));
        assert!(trend.p_value < 1e-6);

        // Noisy but still increasing.
        let noisy: Vec<f64> = (0..12)
            .map(|i| 100.0 + 2.0 * i as f64 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let noisy_trend = linear_regression(&noisy).expect("defined");
        assert!(noisy_trend.slope > 1.5);
        assert!(noisy_trend.p_value < 0.001);
        let (low, high) = noisy_trend
            .slope_confidence_interval(0.95)
            .expect("interval defined");
        assert!(low < noisy_trend.slope && noisy_trend.slope < high);
        assert!(low > 0.0, "significant positive slope excludes zero");

        // A flat series is not a trend.
        let flat = vec![7.0; 12];
        let flat_trend = linear_regression(&flat).expect("defined");
        assert!(close(flat_trend.slope, 0.0, 1e-12));
        assert!(flat_trend.p_value > 0.5);

        assert!(linear_regression(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn relative_total_change_is_scale_invariant() {
        let small: Vec<f64> = (0..11).map(|i| 1000.0 + 10.0 * i as f64).collect();
        let large: Vec<f64> = small.iter().map(|v| v * 1.0e9).collect();
        let a = linear_regression(&small)
            .and_then(|t| t.relative_total_change())
            .expect("defined");
        let b = linear_regression(&large)
            .and_then(|t| t.relative_total_change())
            .expect("defined");
        assert!(close(a, b, 1e-9));
    }

    #[test]
    fn binary_segmentation_locates_the_step() {
        let mut series: Vec<f64> = Vec::new();
        for i in 0..30 {
            series.push(100.0 + (i % 3) as f64 * 0.5);
        }
        for i in 0..30 {
            series.push(130.0 + (i % 3) as f64 * 0.5);
        }
        let points = binary_segmentation(&series, 5, 0.01, 4);
        assert!(!points.is_empty());
        let primary = points
            .iter()
            .max_by(|a, b| compare_finite(&a.magnitude.abs(), &b.magnitude.abs()))
            .expect("at least one point");
        assert!(
            (primary.index as i64 - 30).abs() <= 2,
            "detected index {}",
            primary.index
        );
        assert!(primary.magnitude > 25.0);

        // A stationary series must not yield change points.
        let stable: Vec<f64> = (0..60).map(|i| 100.0 + (i % 4) as f64 * 0.1).collect();
        assert!(binary_segmentation(&stable, 5, 0.001, 4).is_empty());
        assert!(binary_segmentation(&[1.0, 2.0], 5, 0.05, 4).is_empty());
    }

    #[test]
    fn cusum_finds_a_level_shift() {
        let mut series: Vec<f64> = vec![10.0; 30];
        series.extend(std::iter::repeat_n(20.0, 30));
        let points = cusum_change_points(&series, 1.0);
        assert!(!points.is_empty());
        assert!(cusum_change_points(&[5.0; 40], 1.0).is_empty());
    }

    #[test]
    fn fft_recovers_a_pure_tone() {
        let n = 64;
        let cycles = 8.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * cycles * i as f64 / n as f64).sin())
            .collect();
        let spectrum = magnitude_spectrum(&signal).expect("defined");
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| compare_finite(a.1, b.1))
            .map(|(i, _)| i)
            .expect("non-empty");
        assert_eq!(peak, cycles as usize);
        assert!(magnitude_spectrum(&[]).is_none());
        assert!(magnitude_spectrum(&[f64::NAN, 1.0]).is_none());
    }

    #[test]
    fn fft_matches_the_naive_transform() {
        let signal: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7).sin() + 0.3).collect();
        let fast = magnitude_spectrum(&signal).expect("defined");
        let n = signal.len();
        for (k, fast_bin) in fast.iter().enumerate() {
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (j, &value) in signal.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                real += value * angle.cos();
                imaginary += value * angle.sin();
            }
            let naive = (real * real + imaginary * imaginary).sqrt();
            assert!(close(*fast_bin, naive, 1e-9), "bin {}", k);
        }
    }

    #[test]
    fn wavelet_transform_separates_scales() {
        // Constant signal: all detail coefficients vanish for Haar.
        let constant = vec![4.0; 16];
        let decomposition =
            discrete_wavelet_transform(&constant, WaveletFamily::Haar, 3).expect("defined");
        assert_eq!(decomposition.len(), 3);
        for level in &decomposition {
            for &d in &level.detail {
                assert!(close(d, 0.0, 1e-12));
            }
        }

        // Alternating signal: level-1 details carry all the energy.
        let alternating: Vec<f64> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let alt =
            discrete_wavelet_transform(&alternating, WaveletFamily::Haar, 1).expect("defined");
        assert!(alt[0].detail.iter().all(|d| d.abs() > 1.0));
        assert!(alt[0].approximation.iter().all(|a| a.abs() < 1e-12));

        let db4 =
            discrete_wavelet_transform(&constant, WaveletFamily::Daubechies4, 2).expect("defined");
        assert_eq!(db4.len(), 2);
        for &d in &db4[0].detail {
            assert!(close(d, 0.0, 1e-12));
        }

        assert!(discrete_wavelet_transform(&[1.0], WaveletFamily::Haar, 1).is_none());
        assert!(discrete_wavelet_transform(&constant, WaveletFamily::Haar, 0).is_none());
    }

    #[test]
    fn descriptive_statistics_are_nan_safe() {
        let data = [3.0, f64::NAN, 1.0, f64::INFINITY, 2.0];
        let sorted = sorted_finite(&data);
        assert_eq!(sorted, vec![1.0, 2.0, 3.0]);
        assert!(close(mean(&data).expect("finite values"), 2.0, 1e-12));
        assert!(close(
            median_sorted(&sorted).expect("non-empty"),
            2.0,
            1e-12
        ));
        assert!(close(
            percentile_sorted(&sorted, 100.0).expect("non-empty"),
            3.0,
            1e-12
        ));
        assert!(close(iqr_sorted(&sorted).expect("non-empty"), 1.0, 1e-12));
        assert!(mean(&[f64::NAN]).is_none());
        assert!(sample_variance(&[1.0]).is_none());
        assert!(percentile_sorted(&[], 50.0).is_none());
    }
}
