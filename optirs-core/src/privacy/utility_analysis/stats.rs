//! Numerical statistics primitives used by the privacy-utility analyzer.
//!
//! Every routine in this module is pure Rust and dependency free. Each one
//! documents the algorithm it implements and the accuracy that algorithm is
//! known to provide, so that callers can reason about the quality of the
//! numbers they surface to users.

use crate::error::{OptimError, Result};

/// Standard normal cumulative distribution function.
///
/// Implements Abramowitz & Stegun, *Handbook of Mathematical Functions*,
/// formula 26.2.17 (Zelen & Severo rational approximation). The absolute
/// error of that approximation is bounded by `7.5e-8` over the whole real
/// line.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let sign_positive = z >= 0.0;
    let z_abs = z.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * z_abs);
    let phi = (-0.5 * z_abs * z_abs).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let upper_tail = 1.0 - phi * poly;
    if sign_positive {
        upper_tail
    } else {
        1.0 - upper_tail
    }
}

/// Inverse of the standard normal CDF (probit function).
///
/// Implements Peter J. Acklam's rational approximation for the inverse normal
/// CDF (2003, "An algorithm for computing the inverse normal cumulative
/// distribution function"). Its relative error is below `1.15e-9` over the
/// whole open interval `(0, 1)`.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] unless `0 < p < 1`.
pub(crate) fn inverse_normal_cdf(p: f64) -> Result<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "inverse normal CDF requires 0 < p < 1, got {p}"
        )));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
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
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    if x.is_finite() {
        Ok(x)
    } else {
        Err(OptimError::ComputationError(format!(
            "inverse normal CDF produced a non-finite value for p={p}"
        )))
    }
}

/// Two-sided critical value `z` for a confidence level.
///
/// For a confidence level `c` (e.g. `0.95`) this returns
/// `Phi^-1(1 - (1 - c) / 2)`, i.e. `1.959964...` for `c = 0.95`.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] unless `0 < level < 1`.
pub(crate) fn z_for_confidence(level: f64) -> Result<f64> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "confidence level must be in (0, 1), got {level}"
        )));
    }
    inverse_normal_cdf(1.0 - (1.0 - level) / 2.0)
}

/// One-sided critical value `z_beta` for a target statistical power.
///
/// For power `1 - beta` this returns `Phi^-1(1 - beta)`, i.e. `0.841621...`
/// for a power of `0.8`.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] unless `0 < power < 1`.
pub(crate) fn z_for_power(power: f64) -> Result<f64> {
    if !power.is_finite() || power <= 0.0 || power >= 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "target power must be in (0, 1), got {power}"
        )));
    }
    inverse_normal_cdf(power)
}

/// Natural logarithm of the gamma function.
///
/// Lanczos approximation with `g = 7` and nine coefficients (the classic
/// Numerical Recipes / Boost coefficient set); the relative error is below
/// `2e-10` for real arguments `x > 0`.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin())
            .abs()
            .ln()
            - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFS[0];
        let t = x + 7.5;
        for (i, c) in COEFFS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued-fraction expansion used by [`regularized_incomplete_beta`].
///
/// Modified Lentz algorithm as given in Numerical Recipes (3rd ed.) §6.4
/// (`betacf`), iterated to a relative tolerance of `1e-15`.
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> Result<f64> {
    const MAX_ITERATIONS: usize = 300;
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // Even step of the recurrence.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step of the recurrence.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return Ok(h);
        }
    }
    Err(OptimError::ComputationError(format!(
        "incomplete beta continued fraction failed to converge for a={a}, b={b}, x={x}"
    )))
}

/// Regularized incomplete beta function `I_x(a, b)`.
///
/// Uses the continued-fraction representation (Numerical Recipes §6.4) with
/// the standard symmetry switch `I_x(a,b) = 1 - I_{1-x}(b,a)` for
/// `x > (a+1)/(a+b+2)`, which keeps the expansion in its fast-converging
/// regime. Accurate to roughly `1e-14` relative for moderate `a`, `b`.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] for non-positive `a`/`b` or
/// `x` outside `[0, 1]`, and [`OptimError::ComputationError`] if the
/// continued fraction does not converge.
pub(crate) fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> Result<f64> {
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "incomplete beta requires a > 0 and b > 0, got a={a}, b={b}"
        )));
    }
    if !x.is_finite() || !(0.0..=1.0).contains(&x) {
        return Err(OptimError::InvalidParameter(format!(
            "incomplete beta requires 0 <= x <= 1, got {x}"
        )));
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    let value = if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x)? / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x)? / b
    };
    Ok(value.clamp(0.0, 1.0))
}

/// Two-sided p-value of Student's t distribution.
///
/// `p = I_{df/(df+t^2)}(df/2, 1/2)`, the exact tail probability
/// `P(|T_df| >= |t|)` expressed through the regularized incomplete beta
/// function. Unlike a normal approximation this is correct for small degrees
/// of freedom (e.g. `df = 2, t = 3` gives `p = 0.09546`, whereas the normal
/// approximation would give `0.0027`).
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] for non-finite `t` or `df <= 0`.
pub(crate) fn student_t_two_sided_p(t: f64, df: f64) -> Result<f64> {
    if !t.is_finite() {
        return Err(OptimError::InvalidParameter(format!(
            "t statistic must be finite, got {t}"
        )));
    }
    if !df.is_finite() || df <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "degrees of freedom must be > 0, got {df}"
        )));
    }
    let x = df / (df + t * t);
    regularized_incomplete_beta(df / 2.0, 0.5, x)
}

/// Bonferroni-adjusted p-values: `min(1, k * p_i)`.
pub(crate) fn bonferroni_adjust(p_values: &[f64]) -> Vec<f64> {
    let k = p_values.len() as f64;
    p_values.iter().map(|p| (p * k).min(1.0)).collect()
}

/// Holm-Bonferroni step-down adjusted p-values (Holm, 1979).
pub(crate) fn holm_adjust(p_values: &[f64]) -> Vec<f64> {
    let k = p_values.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| {
        p_values[i]
            .partial_cmp(&p_values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adjusted_by_rank = Vec::with_capacity(k);
    let mut running_max = 0.0_f64;
    for (rank, &idx) in order.iter().enumerate() {
        let candidate = (p_values[idx] * (k - rank) as f64).min(1.0);
        running_max = running_max.max(candidate);
        adjusted_by_rank.push(running_max);
    }
    let mut adjusted = vec![0.0_f64; k];
    for (rank, &idx) in order.iter().enumerate() {
        adjusted[idx] = adjusted_by_rank[rank];
    }
    adjusted
}

/// Benjamini-Hochberg step-up adjusted p-values (Benjamini & Hochberg, 1995).
///
/// Adjusted value for the i-th smallest of `k` p-values is
/// `min_{j >= i} (k / j) * p_(j)`, clamped to `[0, 1]`. Rejecting every
/// hypothesis whose adjusted p-value is at most `alpha` controls the false
/// discovery rate at `alpha` under independence or positive regression
/// dependency.
pub(crate) fn benjamini_hochberg_adjust(p_values: &[f64]) -> Vec<f64> {
    let k = p_values.len();
    if k == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| {
        p_values[i]
            .partial_cmp(&p_values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adjusted = vec![0.0_f64; k];
    let mut running_min = 1.0_f64;
    for rank in (0..k).rev() {
        let idx = order[rank];
        let candidate = (p_values[idx] * k as f64 / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(candidate);
        adjusted[idx] = running_min;
    }
    adjusted
}

/// Per-group sample size required to detect an effect of size `effect` with a
/// two-sample test at critical values `z_alpha` (two-sided significance) and
/// `z_beta` (power): `n = 2 ((z_alpha + z_beta) / d)^2`.
///
/// Returns `None` when the effect is numerically zero, where the required
/// sample size is unbounded.
pub(crate) fn required_per_group_sample_size(
    effect: f64,
    z_alpha: f64,
    z_beta: f64,
) -> Option<usize> {
    let d = effect.abs();
    if !d.is_finite() || d < 1e-9 {
        return None;
    }
    let n = 2.0 * ((z_alpha + z_beta) / d).powi(2);
    if n.is_finite() && n < usize::MAX as f64 {
        Some(n.ceil() as usize)
    } else {
        None
    }
}

/// Smallest effect a two-sample test detects with per-group sample size `n`
/// at the given critical values: `d = (z_alpha + z_beta) sqrt(2/n)`.
///
/// This is the exact inverse of [`required_per_group_sample_size`], so the two
/// functions are guaranteed to agree on the power they assume.
pub(crate) fn minimum_detectable_effect(n: f64, z_alpha: f64, z_beta: f64) -> f64 {
    if !n.is_finite() || n <= 0.0 {
        return f64::INFINITY;
    }
    (z_alpha + z_beta) * (2.0 / n).sqrt()
}

/// Power of a two-sample test at per-group sample size `n` and effect size
/// `effect`: `Phi(|d| sqrt(n/2) - z_alpha)`.
pub(crate) fn two_sample_power(effect: f64, n: f64, z_alpha: f64) -> f64 {
    if !n.is_finite() || n <= 0.0 {
        return 0.0;
    }
    normal_cdf(effect.abs() * (n / 2.0).sqrt() - z_alpha).clamp(0.0, 1.0)
}

/// FNV-1a 64-bit hash. Used to derive stable configuration fingerprints and
/// per-purpose RNG seeds; it is a non-cryptographic hash.
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Returns `(mean, sample_variance)`; `(0.0, 0.0)` for empty input.
pub(crate) fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let denom = (n - 1).max(1) as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denom;
    (mean, var)
}

/// Pearson correlation coefficient. Returns `0.0` when the lengths differ,
/// the input is empty, or either sample has (numerically) zero spread.
pub(crate) fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let cov: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let sx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>().sqrt();
    if sx < 1e-12 || sy < 1e-12 {
        return 0.0;
    }
    (cov / (sx * sy)).clamp(-0.9999, 0.9999)
}

/// Gauss-Jordan elimination with partial pivoting.
///
/// # Errors
/// Returns [`OptimError::ComputationError`] if the matrix is singular.
pub(crate) fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .abs()
                    .partial_cmp(&a[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if a[pivot_row][col].abs() < 1e-12 {
            return Err(OptimError::ComputationError(
                "singular matrix in least-squares solve".to_string(),
            ));
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let pivot = a[col][col];
        for val in a[col][col..n].iter_mut() {
            *val /= pivot;
        }
        b[col] /= pivot;
        let pivot_row_snap: Vec<f64> = a[col][col..n].to_vec();
        let pivot_b = b[col];
        for row in 0..n {
            if row != col {
                let factor = a[row][col];
                for (j, pv) in pivot_row_snap.iter().enumerate() {
                    a[row][col + j] -= factor * pv;
                }
                b[row] -= factor * pivot_b;
            }
        }
    }
    Ok(b)
}

/// Polynomial least-squares fit through the normal equations.
///
/// Returns coefficients `[c0, c1, ..., c_degree]` with the constant term
/// first.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] when the inputs have different
/// lengths or too few points for the requested degree, and
/// [`OptimError::ComputationError`] when the normal equations are singular
/// (e.g. repeated abscissae).
pub(crate) fn polyfit(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>> {
    if xs.len() != ys.len() {
        return Err(OptimError::InvalidParameter(
            "polyfit requires xs and ys of equal length".to_string(),
        ));
    }
    if xs.len() <= degree {
        return Err(OptimError::InvalidParameter(format!(
            "polyfit of degree {degree} requires at least {} points, got {}",
            degree + 1,
            xs.len()
        )));
    }
    let n = xs.len();
    let d = degree + 1;
    let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(n);
    for &xi in xs {
        let mut row = Vec::with_capacity(d);
        let mut pw = 1.0_f64;
        for _ in 0..d {
            row.push(pw);
            pw *= xi;
        }
        x_mat.push(row);
    }
    let mut xtx: Vec<Vec<f64>> = vec![vec![0.0_f64; d]; d];
    let mut xty: Vec<f64> = vec![0.0_f64; d];
    for i in 0..n {
        for j in 0..d {
            xty[j] += x_mat[i][j] * ys[i];
            for l in 0..d {
                xtx[j][l] += x_mat[i][j] * x_mat[i][l];
            }
        }
    }
    solve_linear_system(xtx, xty)
}

/// Evaluate a polynomial with Horner's method; `coeffs[0]` is the constant.
pub(crate) fn polyeval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0_f64;
    for c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn test_inverse_normal_cdf_round_trip() {
        for p in [0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999] {
            let z = inverse_normal_cdf(p).expect("valid probability");
            assert!(
                (normal_cdf(z) - p).abs() < 1e-6,
                "round trip failed for p={p}: z={z}"
            );
        }
    }

    #[test]
    fn test_inverse_normal_cdf_rejects_out_of_range() {
        assert!(inverse_normal_cdf(0.0).is_err());
        assert!(inverse_normal_cdf(1.0).is_err());
        assert!(inverse_normal_cdf(f64::NAN).is_err());
    }

    #[test]
    fn test_z_for_confidence_matches_textbook_values() {
        let z95 = z_for_confidence(0.95).expect("valid level");
        assert!((z95 - 1.959_964).abs() < 1e-5, "z95={z95}");
        let z99 = z_for_confidence(0.99).expect("valid level");
        assert!((z99 - 2.575_829).abs() < 1e-5, "z99={z99}");
        // The confidence level must actually drive the value.
        assert!(z99 > z95);
        assert!(z_for_confidence(0.0).is_err());
        assert!(z_for_confidence(1.0).is_err());
    }

    #[test]
    fn test_z_for_power_matches_textbook_values() {
        let z80 = z_for_power(0.8).expect("valid power");
        assert!((z80 - 0.841_621).abs() < 1e-5, "z80={z80}");
        let z90 = z_for_power(0.9).expect("valid power");
        assert!((z90 - 1.281_552).abs() < 1e-5, "z90={z90}");
        assert!(z_for_power(1.0).is_err());
    }

    #[test]
    fn test_ln_gamma_known_values() {
        assert!(ln_gamma(1.0).abs() < 1e-9);
        assert!((ln_gamma(2.0)).abs() < 1e-9);
        // Gamma(5) = 24
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        // Gamma(0.5) = sqrt(pi)
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn test_regularized_incomplete_beta_closed_form() {
        // I_x(1, b) = 1 - (1 - x)^b
        let x = 0.3_f64;
        let b = 2.5_f64;
        let expected = 1.0 - (1.0 - x).powf(b);
        let got = regularized_incomplete_beta(1.0, b, x).expect("valid arguments");
        assert!((got - expected).abs() < 1e-12, "got={got}, want={expected}");
        // Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        let lhs = regularized_incomplete_beta(2.0, 3.0, 0.4).expect("valid arguments");
        let rhs = 1.0 - regularized_incomplete_beta(3.0, 2.0, 0.6).expect("valid arguments");
        assert!((lhs - rhs).abs() < 1e-12);
        assert!(regularized_incomplete_beta(0.0, 1.0, 0.5).is_err());
        assert!(regularized_incomplete_beta(1.0, 1.0, 1.5).is_err());
    }

    #[test]
    fn test_student_t_small_df_is_not_normal() {
        // Reference value: P(|T_2| >= 3) = 0.095465...
        let p = student_t_two_sided_p(3.0, 2.0).expect("valid arguments");
        assert!((p - 0.095_465_9).abs() < 1e-6, "p={p}");
        // The normal approximation that this replaces gave ~0.0027, which
        // would have (wrongly) rejected at alpha = 0.05.
        assert!(p > 0.05, "df=2, t=3 must not be significant at 5%");
    }

    #[test]
    fn test_student_t_converges_to_normal() {
        let p_t = student_t_two_sided_p(1.96, 1.0e6).expect("valid arguments");
        let p_normal = 2.0 * (1.0 - normal_cdf(1.96));
        assert!(
            (p_t - p_normal).abs() < 1e-4,
            "p_t={p_t}, p_normal={p_normal}"
        );
    }

    #[test]
    fn test_student_t_zero_statistic() {
        let p = student_t_two_sided_p(0.0, 7.0).expect("valid arguments");
        assert!((p - 1.0).abs() < 1e-12, "p={p}");
        assert!(student_t_two_sided_p(1.0, 0.0).is_err());
        assert!(student_t_two_sided_p(f64::NAN, 4.0).is_err());
    }

    #[test]
    fn test_bonferroni_and_holm() {
        let p = [0.01_f64, 0.04];
        let bonf = bonferroni_adjust(&p);
        assert!((bonf[0] - 0.02).abs() < 1e-12);
        assert!((bonf[1] - 0.08).abs() < 1e-12);
        let holm = holm_adjust(&p);
        assert!((holm[0] - 0.02).abs() < 1e-12);
        assert!((holm[1] - 0.04).abs() < 1e-12);
        // Holm is uniformly no more conservative than Bonferroni.
        for (h, b) in holm.iter().zip(bonf.iter()) {
            assert!(h <= b);
        }
    }

    #[test]
    fn test_benjamini_hochberg_monotone_and_bounded() {
        let p = [0.001_f64, 0.008, 0.039, 0.041, 0.9];
        let adj = benjamini_hochberg_adjust(&p);
        // k/i * p_(i) with the step-up minimum applied.
        assert!((adj[0] - 0.005).abs() < 1e-12, "adj0={}", adj[0]);
        assert!((adj[1] - 0.02).abs() < 1e-12, "adj1={}", adj[1]);
        assert!((adj[2] - 0.05125).abs() < 1e-9, "adj2={}", adj[2]);
        assert!((adj[3] - 0.05125).abs() < 1e-9, "adj3={}", adj[3]);
        assert!((adj[4] - 0.9).abs() < 1e-12, "adj4={}", adj[4]);
        // BH never exceeds Bonferroni and never falls below the raw p-value.
        let bonf = bonferroni_adjust(&p);
        for i in 0..p.len() {
            assert!(adj[i] <= bonf[i] + 1e-12);
            assert!(adj[i] >= p[i] - 1e-12);
        }
        assert!(benjamini_hochberg_adjust(&[]).is_empty());
    }

    #[test]
    fn test_power_analysis_is_self_consistent() {
        let z_alpha = z_for_confidence(0.95).expect("valid level");
        let z_beta = z_for_power(0.8).expect("valid power");
        for effect in [0.2_f64, 0.5, 0.8, 1.3] {
            let n = required_per_group_sample_size(effect, z_alpha, z_beta)
                .expect("non-zero effect has a finite required sample size");
            let mde = minimum_detectable_effect(n as f64, z_alpha, z_beta);
            // Round trip: rounding the required sample size up can only make
            // the detectable effect smaller, never larger, and never by more
            // than the step from one whole sample to the next.
            assert!(mde <= effect + 1e-12, "effect={effect}, n={n}, mde={mde}");
            assert!(
                mde >= minimum_detectable_effect(n as f64 + 1.0, z_alpha, z_beta),
                "effect={effect}, n={n}, mde={mde}"
            );
            // And the power at that sample size is at least the requested one.
            let power = two_sample_power(effect, n as f64, z_alpha);
            assert!(power >= 0.8 - 1e-9, "power={power} for effect={effect}");
            assert!(power <= 0.95, "power={power} for effect={effect}");
        }
        // With a large sample the ceiling is negligible and the round trip is
        // exact to within a fraction of a percent.
        let n = required_per_group_sample_size(0.2, z_alpha, z_beta).expect("finite");
        let mde = minimum_detectable_effect(n as f64, z_alpha, z_beta);
        assert!((mde - 0.2).abs() < 0.002, "n={n}, mde={mde}");
        assert!(required_per_group_sample_size(0.0, z_alpha, z_beta).is_none());
        assert!(minimum_detectable_effect(0.0, z_alpha, z_beta).is_infinite());
    }

    #[test]
    fn test_fnv1a_is_stable_and_sensitive() {
        assert_eq!(fnv1a_64(b"abc"), fnv1a_64(b"abc"));
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"abd"));
    }

    #[test]
    fn test_polyfit_linear_and_quadratic() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![1.0_f64, 3.0, 5.0];
        let coeffs = polyfit(&xs, &ys, 1).expect("well-posed fit");
        assert!((coeffs[0] - 1.0).abs() < 1e-9);
        assert!((coeffs[1] - 2.0).abs() < 1e-9);

        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
        let coeffs = polyfit(&xs, &ys, 2).expect("well-posed fit");
        assert!((coeffs[0] - 1.0).abs() < 1e-6);
        assert!((coeffs[1] - 2.0).abs() < 1e-6);
        assert!((coeffs[2] - 3.0).abs() < 1e-6);
        assert!((polyeval(&coeffs, 4.0) - (1.0 + 8.0 + 48.0)).abs() < 1e-6);
    }

    #[test]
    fn test_polyfit_rejects_degenerate_inputs() {
        assert!(polyfit(&[1.0, 2.0], &[1.0], 1).is_err());
        assert!(polyfit(&[1.0, 2.0], &[1.0, 2.0], 5).is_err());
        // Repeated abscissae make the normal equations singular.
        assert!(polyfit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn test_pearson_and_mean_var() {
        let xs = [1.0_f64, 2.0, 3.0, 4.0];
        let ys = [2.0_f64, 4.0, 6.0, 8.0];
        assert!((pearson_correlation(&xs, &ys) - 0.9999).abs() < 1e-9);
        let (m, v) = mean_var(&xs);
        assert!((m - 2.5).abs() < 1e-12);
        assert!((v - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!(mean_var(&[]), (0.0, 0.0));
    }
}
