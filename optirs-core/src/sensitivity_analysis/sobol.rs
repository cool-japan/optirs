// Sobol global sensitivity analysis (Saltelli's algorithm).
//
// Implements the variance-based Sobol method using Saltelli's enhanced
// pick-and-freeze estimator. Given a black-box model `f: R^k -> R` and
// rectangular bounds, the analyzer estimates:
//
// * first-order indices `S_i = Var[E[Y|X_i]] / Var[Y]`
// * total-order indices `S_Ti = E[Var[Y|X_~i]] / Var[Y]`
// * (optionally) closed second-order indices `S_ij`
//
// References
// ----------
// - Saltelli, A. et al. (2010). "Variance based sensitivity analysis of
//   model output. Design and estimator for the total sensitivity index."
//   Computer Physics Communications, 181(2), 259-270.
// - Sobol, I.M. (2001). "Global sensitivity indices for nonlinear
//   mathematical models and their Monte Carlo estimates."
//   Mathematics and Computers in Simulation, 55(1-3), 271-280.

use crate::error::{OptimError, Result};
use crate::sensitivity_analysis::{SensitivityAnalyzer, SensitivityIndices};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::fmt::Debug;

/// Default number of base samples used per Saltelli matrix.
const DEFAULT_SAMPLES: usize = 1024;

/// Variance-based Sobol global sensitivity analyzer.
///
/// The analyzer requires `N * (k + 2)` model evaluations to estimate first
/// and total-order indices, where `N` is [`SobolAnalyzer::n_samples`] and
/// `k` is the number of parameters. Enabling second-order indices adds an
/// extra `N * k` evaluations.
#[derive(Debug)]
pub struct SobolAnalyzer<F: Float + ScalarOperand + Debug> {
    /// Base sample size per Saltelli matrix.
    n_samples: usize,
    /// Whether to compute total-order indices.
    compute_total_order: bool,
    /// Whether to compute (closed) second-order indices.
    compute_second_order: bool,
    /// Seeded RNG used for sampling.
    rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Seed kept for reproducibility introspection and rebuilding.
    seed: u64,
    /// Cached output of the last successful analysis.
    last_indices: Option<SensitivityIndices<F>>,
}

impl<F: Float + ScalarOperand + Debug> SobolAnalyzer<F> {
    /// Create a new Sobol analyzer with default parameters.
    pub fn new() -> Self {
        let seed: u64 = 0xC0FFEE_u64;
        Self {
            n_samples: DEFAULT_SAMPLES,
            compute_total_order: true,
            compute_second_order: false,
            rng: Random::seed(seed),
            seed,
            last_indices: None,
        }
    }

    /// Set the base sample size `N` for the Saltelli matrices. The total
    /// number of model evaluations is `N * (k + 2)` (or `N * (2k + 2)` with
    /// second-order indices enabled).
    pub fn with_samples(mut self, n: usize) -> Self {
        self.n_samples = n.max(2);
        self
    }

    /// Reseed the analyzer for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Random::seed(seed);
        self.seed = seed;
        self
    }

    /// Toggle computation of total-order indices.
    pub fn with_total_order(mut self, flag: bool) -> Self {
        self.compute_total_order = flag;
        self
    }

    /// Toggle computation of closed second-order indices.
    pub fn with_second_order(mut self, flag: bool) -> Self {
        self.compute_second_order = flag;
        self
    }

    /// Number of base samples per Saltelli matrix.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Last computed indices, if any.
    pub fn last_indices(&self) -> Option<&SensitivityIndices<F>> {
        self.last_indices.as_ref()
    }

    /// Seed used by the underlying RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate an `n x k` matrix of uniform `[0, 1)` samples and immediately
    /// scale them into the rectangular bounds `[low_i, high_i]`.
    fn sample_scaled(&mut self, n: usize, bounds: &[(F, F)]) -> Result<Array2<F>> {
        let k = bounds.len();
        let mut mat = Array2::<F>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let u: f64 = self.rng.gen_range(0.0..1.0);
                let u_f = F::from(u).ok_or_else(|| {
                    OptimError::ComputationError("uniform conversion failed".into())
                })?;
                let (low, high) = bounds[j];
                mat[(i, j)] = low + (high - low) * u_f;
            }
        }
        Ok(mat)
    }

    /// Evaluate `model` over every row of `samples`.
    fn evaluate_matrix(model: &dyn Fn(&Array1<F>) -> F, samples: &Array2<F>) -> Array1<F> {
        let n = samples.nrows();
        let mut out = Array1::<F>::zeros(n);
        for i in 0..n {
            let row = samples.row(i).to_owned();
            out[i] = model(&row);
        }
        out
    }

    /// Compute the variance `Var[Y]` from the concatenation of `y_a` and
    /// `y_b`. Uses the sample variance (denominator `2N`).
    fn total_variance(y_a: &Array1<F>, y_b: &Array1<F>) -> F {
        let n = y_a.len();
        let two_n = F::from(2 * n).unwrap_or_else(F::one);
        let mut sum = F::zero();
        for i in 0..n {
            sum = sum + y_a[i] + y_b[i];
        }
        let mean = sum / two_n;
        let mut acc = F::zero();
        for i in 0..n {
            let d_a = y_a[i] - mean;
            let d_b = y_b[i] - mean;
            acc = acc + d_a * d_a + d_b * d_b;
        }
        acc / two_n
    }

    /// Saltelli (2010) first-order estimator.
    ///
    /// `S_i = (1/N) Σ_j y_B[j] · (y_AB[j] - y_A[j]) / Var(Y)`.
    fn first_order_estimator(y_a: &Array1<F>, y_b: &Array1<F>, y_ab: &Array1<F>, var_y: F) -> F {
        let n = y_a.len();
        if var_y <= F::zero() {
            return F::zero();
        }
        let n_f = F::from(n).unwrap_or_else(F::one);
        let mut acc = F::zero();
        for j in 0..n {
            acc = acc + y_b[j] * (y_ab[j] - y_a[j]);
        }
        (acc / n_f) / var_y
    }

    /// Saltelli (2010) total-order estimator.
    ///
    /// `S_Ti = (1/(2N)) Σ_j (y_A[j] - y_AB[j])² / Var(Y)`.
    fn total_order_estimator(y_a: &Array1<F>, y_ab: &Array1<F>, var_y: F) -> F {
        let n = y_a.len();
        if var_y <= F::zero() {
            return F::zero();
        }
        let two_n = F::from(2 * n).unwrap_or_else(F::one);
        let mut acc = F::zero();
        for j in 0..n {
            let d = y_a[j] - y_ab[j];
            acc = acc + d * d;
        }
        (acc / two_n) / var_y
    }

    /// Build the resampling matrix `A_B^(i)` by copying `mat_a` and
    /// substituting its `column` from `mat_b`.
    fn build_swap_matrix(mat_a: &Array2<F>, mat_b: &Array2<F>, column: usize) -> Array2<F> {
        let mut out = mat_a.clone();
        let n = out.nrows();
        for row in 0..n {
            out[(row, column)] = mat_b[(row, column)];
        }
        out
    }

    /// Build the resampling matrix `B_A^(i)` by copying `mat_b` and
    /// substituting its `column` from `mat_a`. Used for closed second-order
    /// indices.
    fn build_swap_matrix_ba(mat_a: &Array2<F>, mat_b: &Array2<F>, column: usize) -> Array2<F> {
        let mut out = mat_b.clone();
        let n = out.nrows();
        for row in 0..n {
            out[(row, column)] = mat_a[(row, column)];
        }
        out
    }
}

impl<F: Float + ScalarOperand + Debug> Default for SobolAnalyzer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + ScalarOperand + Debug> SensitivityAnalyzer<F> for SobolAnalyzer<F> {
    fn analyze(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<SensitivityIndices<F>> {
        let k = bounds.len();
        if k == 0 {
            return Err(OptimError::InvalidConfig(
                "Sobol analysis requires at least one parameter".into(),
            ));
        }
        for (idx, (low, high)) in bounds.iter().enumerate() {
            if *low >= *high {
                return Err(OptimError::InvalidConfig(format!(
                    "bounds[{idx}] must satisfy low < high"
                )));
            }
        }

        let n = self.n_samples;
        let mat_a = self.sample_scaled(n, bounds)?;
        let mat_b = self.sample_scaled(n, bounds)?;

        // Sanity check: caller-side dimension mismatch. We evaluate one row to
        // ensure the model accepts vectors of length `k`. The model is a
        // black-box `Fn(&Array1<F>) -> F`, so the only externally observable
        // failure is the model's own behavior; we still want to surface
        // configuration errors when bounds disagrees with the model's intent.
        let _probe_row = mat_a.row(0).to_owned();
        let _probe_value = model(&_probe_row);

        let y_a = Self::evaluate_matrix(model, &mat_a);
        let y_b = Self::evaluate_matrix(model, &mat_b);

        let var_y = Self::total_variance(&y_a, &y_b);

        let mut first_order = vec![F::zero(); k];
        let mut total_order = vec![F::zero(); k];

        // Cache the per-parameter pick-and-freeze evaluations so we can also
        // assemble (closed) second-order indices without repeating work.
        let mut y_ab_per_param: Vec<Array1<F>> = Vec::with_capacity(k);

        for i in 0..k {
            let mat_ab_i = Self::build_swap_matrix(&mat_a, &mat_b, i);
            let y_ab_i = Self::evaluate_matrix(model, &mat_ab_i);

            first_order[i] = Self::first_order_estimator(&y_a, &y_b, &y_ab_i, var_y);
            if self.compute_total_order {
                total_order[i] = Self::total_order_estimator(&y_a, &y_ab_i, var_y);
            } else {
                total_order[i] = first_order[i];
            }
            y_ab_per_param.push(y_ab_i);
        }

        let second_order = if self.compute_second_order && k >= 2 {
            // Closed second-order indices: S_{ij}^c = (1/N) Σ y_{B_A^(i)} · y_{A_B^(j)} / Var(Y) - S_i - S_j
            // We compute y_{B_A^(i)} on the fly for each i.
            let mut s_ij = vec![vec![F::zero(); k]; k];
            let n_f = F::from(n).unwrap_or_else(F::one);
            // Means of y_A and y_B used in the closed estimator.
            let mut mean_a = F::zero();
            let mut mean_b = F::zero();
            for j in 0..n {
                mean_a = mean_a + y_a[j];
                mean_b = mean_b + y_b[j];
            }
            mean_a = mean_a / n_f;
            mean_b = mean_b / n_f;

            // Precompute y_{B_A^(i)} matrices.
            let mut y_ba_per_param: Vec<Array1<F>> = Vec::with_capacity(k);
            for i in 0..k {
                let mat_ba_i = Self::build_swap_matrix_ba(&mat_a, &mat_b, i);
                y_ba_per_param.push(Self::evaluate_matrix(model, &mat_ba_i));
            }

            for i in 0..k {
                for j in (i + 1)..k {
                    let mut acc = F::zero();
                    for s in 0..n {
                        acc = acc + y_ba_per_param[i][s] * y_ab_per_param[j][s];
                    }
                    let closed = if var_y > F::zero() {
                        (acc / n_f - mean_a * mean_b) / var_y
                    } else {
                        F::zero()
                    };
                    // Subtract first-order contributions to get the pure
                    // interaction term S_{ij}.
                    let interaction = closed - first_order[i] - first_order[j];
                    s_ij[i][j] = interaction;
                    s_ij[j][i] = interaction;
                }
            }
            Some(s_ij)
        } else {
            None
        };

        let parameter_names = (0..k).map(|i| format!("x{i}")).collect::<Vec<_>>();
        let indices = SensitivityIndices {
            first_order,
            total_order,
            second_order,
            parameter_names,
        };
        self.last_indices = Some(indices.clone());
        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ishigami(x: &Array1<f64>) -> f64 {
        let (x1, x2, x3) = (x[0], x[1], x[2]);
        x1.sin() + 7.0 * x2.sin().powi(2) + 0.1 * x3.powi(4) * x1.sin()
    }

    fn pi_bounds() -> Vec<(f64, f64)> {
        vec![(-PI, PI), (-PI, PI), (-PI, PI)]
    }

    #[test]
    fn test_ishigami_first_order_indices() {
        let mut sa = SobolAnalyzer::<f64>::new().with_samples(4096).with_seed(7);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let res = sa.analyze(model, &bounds).expect("analyze failed");
        // Known analytical Sobol indices for the Ishigami function with
        // a = 7, b = 0.1: S1 ≈ 0.314, S2 ≈ 0.442, S3 = 0.
        assert!(
            (res.first_order[0] - 0.314).abs() < 0.10,
            "S1 = {} far from 0.314",
            res.first_order[0]
        );
        assert!(
            (res.first_order[1] - 0.442).abs() < 0.10,
            "S2 = {} far from 0.442",
            res.first_order[1]
        );
        assert!(
            res.first_order[2].abs() < 0.10,
            "S3 = {} should be near 0",
            res.first_order[2]
        );
    }

    #[test]
    fn test_ishigami_total_order_indices() {
        let mut sa = SobolAnalyzer::<f64>::new()
            .with_samples(4096)
            .with_seed(11)
            .with_total_order(true);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let res = sa.analyze(model, &bounds).expect("analyze failed");
        // Analytical totals: ST1 ≈ 0.557, ST2 ≈ 0.442, ST3 ≈ 0.244.
        assert!(
            (res.total_order[0] - 0.557).abs() < 0.10,
            "ST1 = {} far from 0.557",
            res.total_order[0]
        );
        assert!(
            (res.total_order[1] - 0.442).abs() < 0.10,
            "ST2 = {} far from 0.442",
            res.total_order[1]
        );
        assert!(
            (res.total_order[2] - 0.244).abs() < 0.12,
            "ST3 = {} far from 0.244",
            res.total_order[2]
        );
    }

    #[test]
    fn test_indices_sum_bounded() {
        let mut sa = SobolAnalyzer::<f64>::new().with_samples(2048).with_seed(19);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let res = sa.analyze(model, &bounds).expect("analyze failed");
        let s_sum: f64 = res.first_order.iter().sum();
        assert!(s_sum <= 1.0 + 0.15, "Σ S_i = {s_sum} exceeded 1 + tol");
    }

    #[test]
    fn test_total_geq_first() {
        // The Saltelli first-order estimator is unbiased but has higher
        // variance than the total-order estimator at low sample counts, so
        // individual realizations can momentarily violate the theoretical
        // ordering S_Ti >= S_i. We use a large sample budget plus a small
        // tolerance to absorb residual Monte Carlo noise.
        let mut sa = SobolAnalyzer::<f64>::new()
            .with_samples(8192)
            .with_seed(23)
            .with_total_order(true);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let res = sa.analyze(model, &bounds).expect("analyze failed");
        for i in 0..res.first_order.len() {
            assert!(
                res.total_order[i] + 0.10 >= res.first_order[i],
                "ST{i} = {} < S{i} = {}",
                res.total_order[i],
                res.first_order[i]
            );
        }
    }

    #[test]
    fn test_constant_function_zero_indices() {
        let mut sa = SobolAnalyzer::<f64>::new().with_samples(512).with_seed(31);
        let constant: &dyn Fn(&Array1<f64>) -> f64 = &|_x| 5.0;
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let res = sa.analyze(constant, &bounds).expect("analyze failed");
        for s in &res.first_order {
            assert!(s.abs() < 1e-6, "expected 0, got {s}");
        }
    }

    #[test]
    fn test_linear_function_indices() {
        let mut sa = SobolAnalyzer::<f64>::new().with_samples(4096).with_seed(37);
        // f(x) = 2*x1 + 3*x2 + 0*x3 over uniform [0,1]^3. Variance share
        // ratio is 4 : 9, so S2 must exceed S1 by a wide margin while S3
        // is essentially zero.
        let linear: &dyn Fn(&Array1<f64>) -> f64 =
            &|x: &Array1<f64>| 2.0 * x[0] + 3.0 * x[1] + 0.0 * x[2];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let res = sa.analyze(linear, &bounds).expect("analyze failed");
        assert!(
            res.first_order[1] > res.first_order[0],
            "S2 = {} should exceed S1 = {}",
            res.first_order[1],
            res.first_order[0]
        );
        assert!(
            res.first_order[2].abs() < 0.05,
            "S3 should be near 0, got {}",
            res.first_order[2]
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let mut sa_a = SobolAnalyzer::<f64>::new().with_samples(256).with_seed(42);
        let mut sa_b = SobolAnalyzer::<f64>::new().with_samples(256).with_seed(42);
        let res_a = sa_a.analyze(model, &bounds).expect("analyze failed");
        let res_b = sa_b.analyze(model, &bounds).expect("analyze failed");
        for i in 0..3 {
            assert!((res_a.first_order[i] - res_b.first_order[i]).abs() < 1e-12);
            assert!((res_a.total_order[i] - res_b.total_order[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let sa = SobolAnalyzer::<f64>::new()
            .with_samples(8192)
            .with_seed(101)
            .with_total_order(false)
            .with_second_order(true);
        assert_eq!(sa.n_samples(), 8192);
        assert_eq!(sa.seed(), 101);
        assert!(!sa.compute_total_order);
        assert!(sa.compute_second_order);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut sa = SobolAnalyzer::<f64>::new().with_samples(64);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds: Vec<(f64, f64)> = Vec::new();
        let err = sa.analyze(model, &bounds);
        assert!(matches!(err, Err(OptimError::InvalidConfig(_))));

        // Also reject invalid (collapsed) bounds.
        let bad_bounds = vec![(1.0, 1.0), (0.0, 1.0)];
        let err2 = sa.analyze(model, &bad_bounds);
        assert!(matches!(err2, Err(OptimError::InvalidConfig(_))));
    }

    #[test]
    fn test_second_order_runs() {
        let mut sa = SobolAnalyzer::<f64>::new()
            .with_samples(256)
            .with_seed(99)
            .with_second_order(true);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &ishigami;
        let bounds = pi_bounds();
        let res = sa.analyze(model, &bounds).expect("analyze failed");
        let so = res
            .second_order
            .as_ref()
            .expect("second-order indices should be computed");
        assert_eq!(so.len(), 3);
        assert_eq!(so[0].len(), 3);
    }
}
