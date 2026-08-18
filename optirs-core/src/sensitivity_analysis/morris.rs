// Morris Elementary Effects (EE) screening method.
//
// The Morris method is a one-step-at-a-time global screening technique that
// identifies parameters with negligible, linear, or nonlinear/interacting
// influence on the output. It is typically used as a cheap preliminary step
// before a full variance-based analysis (such as Sobol).
//
// References
// ----------
// - Morris, M.D. (1991). "Factorial sampling plans for preliminary
//   computational experiments." Technometrics, 33(2), 161-174.
// - Campolongo, F., Cariboni, J., Saltelli, A. (2007). "An effective
//   screening design for sensitivity analysis of large models."
//   Environmental Modelling & Software, 22(10), 1509-1518.

use crate::error::{OptimError, Result};
use crate::sensitivity_analysis::{SensitivityAnalyzer, SensitivityIndices};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::fmt::Debug;

/// Default trajectory count `r`.
const DEFAULT_TRAJECTORIES: usize = 10;
/// Default number of levels `p`.
const DEFAULT_LEVELS: usize = 4;

/// Aggregated Morris sensitivity measures.
///
/// All vectors have length `k` (one entry per parameter).
#[derive(Debug, Clone)]
pub struct MorrisIndices<F: Float> {
    /// Mean of the **signed** elementary effects.
    pub mu: Vec<F>,
    /// Mean of the **absolute** elementary effects (Campolongo's μ*).
    pub mu_star: Vec<F>,
    /// Standard deviation of the elementary effects.
    pub sigma: Vec<F>,
    /// Human-readable parameter labels.
    pub parameter_names: Vec<String>,
}

impl<F: Float> MorrisIndices<F> {
    /// Number of parameters described by the indices.
    pub fn num_parameters(&self) -> usize {
        self.mu_star.len()
    }
}

/// Morris Elementary Effects analyzer.
#[derive(Debug)]
pub struct MorrisAnalyzer<F: Float + Debug> {
    /// Number of trajectories `r` (each contributes one EE per parameter).
    n_trajectories: usize,
    /// Number of grid levels `p`.
    n_levels: usize,
    /// Seeded RNG.
    rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Stored seed.
    seed: u64,
    /// Cached Morris indices from the last successful call.
    last_indices: Option<MorrisIndices<F>>,
}

impl<F: Float + Debug> MorrisAnalyzer<F> {
    /// Construct a new analyzer with default settings.
    pub fn new() -> Self {
        let seed: u64 = 0xCAFEBABE_u64;
        Self {
            n_trajectories: DEFAULT_TRAJECTORIES,
            n_levels: DEFAULT_LEVELS,
            rng: Random::seed(seed),
            seed,
            last_indices: None,
        }
    }

    /// Set the number of independent trajectories `r`.
    pub fn with_trajectories(mut self, r: usize) -> Self {
        self.n_trajectories = r.max(1);
        self
    }

    /// Set the grid resolution `p`. Must be even and `>= 2`.
    pub fn with_levels(mut self, p: usize) -> Self {
        let mut levels = p.max(2);
        if !levels.is_multiple_of(2) {
            levels += 1;
        }
        self.n_levels = levels;
        self
    }

    /// Reseed the analyzer for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Random::seed(seed);
        self.seed = seed;
        self
    }

    /// Number of trajectories used per analysis.
    pub fn n_trajectories(&self) -> usize {
        self.n_trajectories
    }

    /// Grid resolution `p`.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Cached indices from the last analysis.
    pub fn last_indices(&self) -> Option<&MorrisIndices<F>> {
        self.last_indices.as_ref()
    }

    /// Sampled seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Run the analysis and store the result on the analyzer.
    pub fn analyze_morris(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<MorrisIndices<F>> {
        let k = bounds.len();
        if k == 0 {
            return Err(OptimError::InvalidConfig(
                "Morris analysis requires at least one parameter".into(),
            ));
        }
        for (idx, (low, high)) in bounds.iter().enumerate() {
            if *low >= *high {
                return Err(OptimError::InvalidConfig(format!(
                    "bounds[{idx}] must satisfy low < high"
                )));
            }
        }

        // Compute the step Δ = p / (2 (p - 1)) in unit-cube coordinates.
        let p_f = F::from(self.n_levels).ok_or_else(|| {
            OptimError::ComputationError("failed to convert n_levels to F".into())
        })?;
        let denom = F::from(2 * (self.n_levels - 1)).ok_or_else(|| {
            OptimError::ComputationError("failed to convert level denominator".into())
        })?;
        let delta = p_f / denom;

        // Storage for per-parameter elementary effects across all trajectories.
        let mut effects: Vec<Vec<F>> = vec![Vec::with_capacity(self.n_trajectories); k];

        for _traj in 0..self.n_trajectories {
            // 1. Random base point in [0, 1 - Δ]^k so that x + Δ stays inside [0, 1].
            let mut x = Array1::<F>::zeros(k);
            for j in 0..k {
                let u: f64 = self.rng.gen_range(0.0..1.0);
                let u_f = F::from(u).ok_or_else(|| {
                    OptimError::ComputationError("uniform conversion failed".into())
                })?;
                // Restrict to [0, 1 - Δ] so a forward step keeps us in the cube.
                let one_minus_delta = F::one() - delta;
                x[j] = u_f * one_minus_delta;
            }

            // 2. Random permutation of parameter order via Fisher-Yates.
            let mut order: Vec<usize> = (0..k).collect();
            for idx in (1..k).rev() {
                let swap_to: usize = self.rng.gen_range(0..(idx + 1));
                order.swap(idx, swap_to);
            }

            // 3. Random direction (+ or -) per parameter.
            let mut direction = vec![F::one(); k];
            for dir in direction.iter_mut().take(k) {
                let s: f64 = self.rng.gen_range(0.0..1.0);
                *dir = if s < 0.5 { -F::one() } else { F::one() };
            }

            // 4. Walk along the trajectory, computing one EE per dimension.
            let f_current = model(&Self::scale_to_bounds(&x, bounds));
            let mut f_prev = f_current;
            for &param_idx in &order {
                // Step ±Δ in unit-cube coords.
                let mut x_new = x.clone();
                let mut step = direction[param_idx] * delta;
                let candidate = x_new[param_idx] + step;
                if candidate > F::one() || candidate < F::zero() {
                    // Reflect the step so we remain inside the unit cube.
                    step = -step;
                    direction[param_idx] = -direction[param_idx];
                }
                x_new[param_idx] = x_new[param_idx] + step;
                let f_next = model(&Self::scale_to_bounds(&x_new, bounds));

                // Elementary effect in unit-cube coordinates. The conversion
                // back to the user-scaled domain cancels out because the
                // perturbation is proportional to (high - low) on both sides
                // of the finite difference.
                let ee = (f_next - f_prev) / step;
                effects[param_idx].push(ee);

                f_prev = f_next;
                x = x_new;
            }
        }

        // 5. Aggregate per-parameter statistics.
        let mut mu = vec![F::zero(); k];
        let mut mu_star = vec![F::zero(); k];
        let mut sigma = vec![F::zero(); k];
        for j in 0..k {
            let ees = &effects[j];
            if ees.is_empty() {
                continue;
            }
            let n_f = F::from(ees.len()).unwrap_or_else(F::one);
            let mut sum = F::zero();
            let mut abs_sum = F::zero();
            for &v in ees {
                sum = sum + v;
                abs_sum = abs_sum + v.abs();
            }
            mu[j] = sum / n_f;
            mu_star[j] = abs_sum / n_f;
            // Sample standard deviation (denominator n - 1 when possible).
            if ees.len() > 1 {
                let denom = F::from(ees.len() - 1).unwrap_or_else(F::one);
                let mean = mu[j];
                let mut acc = F::zero();
                for &v in ees {
                    let d = v - mean;
                    acc = acc + d * d;
                }
                sigma[j] = (acc / denom).sqrt();
            } else {
                sigma[j] = F::zero();
            }
        }

        let parameter_names = (0..k).map(|i| format!("x{i}")).collect::<Vec<_>>();
        let indices = MorrisIndices {
            mu,
            mu_star,
            sigma,
            parameter_names,
        };
        self.last_indices = Some(indices.clone());
        Ok(indices)
    }

    /// Convert a unit-cube point `[0, 1]^k` into the user-provided
    /// rectangular domain.
    fn scale_to_bounds(point: &Array1<F>, bounds: &[(F, F)]) -> Array1<F> {
        let k = bounds.len();
        let mut out = Array1::<F>::zeros(k);
        for j in 0..k {
            let (low, high) = bounds[j];
            out[j] = low + (high - low) * point[j];
        }
        out
    }
}

impl<F: Float + Debug> Default for MorrisAnalyzer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> SensitivityAnalyzer<F> for MorrisAnalyzer<F> {
    fn analyze(
        &mut self,
        model: &dyn Fn(&Array1<F>) -> F,
        bounds: &[(F, F)],
    ) -> Result<SensitivityIndices<F>> {
        let morris = self.analyze_morris(model, bounds)?;
        // Map Morris quantities into the common SensitivityIndices envelope so
        // callers can switch analyzers without changing downstream code:
        //   * `first_order`  ← μ* (importance magnitude)
        //   * `total_order`  ← σ  (interaction/non-linearity proxy)
        Ok(SensitivityIndices {
            first_order: morris.mu_star.clone(),
            total_order: morris.sigma.clone(),
            second_order: None,
            parameter_names: morris.parameter_names.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_function_mu_star() {
        let mut ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(40)
            .with_levels(4)
            .with_seed(1);
        // f(x) = 2 x1 + 3 x2 + 0 x3 on [0, 1]^3.
        let model: &dyn Fn(&Array1<f64>) -> f64 =
            &|x: &Array1<f64>| 2.0 * x[0] + 3.0 * x[1] + 0.0 * x[2];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let idx = ma.analyze_morris(model, &bounds).expect("analyze failed");
        assert!(
            (idx.mu_star[0] - 2.0).abs() < 0.5,
            "μ*₁ = {} far from 2",
            idx.mu_star[0]
        );
        assert!(
            (idx.mu_star[1] - 3.0).abs() < 0.5,
            "μ*₂ = {} far from 3",
            idx.mu_star[1]
        );
        assert!(
            idx.mu_star[2].abs() < 0.5,
            "μ*₃ = {} should be near 0",
            idx.mu_star[2]
        );
    }

    #[test]
    fn test_constant_function_mu_star_zero() {
        let mut ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(20)
            .with_seed(2);
        let constant: &dyn Fn(&Array1<f64>) -> f64 = &|_x| 5.0;
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let idx = ma
            .analyze_morris(constant, &bounds)
            .expect("analyze failed");
        for &v in &idx.mu_star {
            assert!(v.abs() < 1e-8, "μ* = {v} should be 0");
        }
    }

    #[test]
    fn test_sigma_zero_for_linear() {
        let mut ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(20)
            .with_seed(3);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| 2.0 * x[0] + 3.0 * x[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let idx = ma.analyze_morris(model, &bounds).expect("analyze failed");
        for &s in &idx.sigma {
            assert!(s < 1e-8, "linear σ = {s} should be 0");
        }
    }

    #[test]
    fn test_sigma_nonzero_for_nonlinear() {
        let mut ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(40)
            .with_seed(4);
        // f(x) = x1^2 on [0, 1].
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0] * x[0];
        let bounds = vec![(0.0, 1.0)];
        let idx = ma.analyze_morris(model, &bounds).expect("analyze failed");
        assert!(
            idx.sigma[0] > 1e-3,
            "σ = {} should be strictly positive",
            idx.sigma[0]
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0].sin() + 2.0 * x[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut a = MorrisAnalyzer::<f64>::new()
            .with_trajectories(8)
            .with_seed(123);
        let mut b = MorrisAnalyzer::<f64>::new()
            .with_trajectories(8)
            .with_seed(123);
        let res_a = a.analyze_morris(model, &bounds).expect("analyze failed");
        let res_b = b.analyze_morris(model, &bounds).expect("analyze failed");
        for j in 0..2 {
            assert!((res_a.mu_star[j] - res_b.mu_star[j]).abs() < 1e-12);
            assert!((res_a.sigma[j] - res_b.sigma[j]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(25)
            .with_levels(6)
            .with_seed(456);
        assert_eq!(ma.n_trajectories(), 25);
        assert_eq!(ma.n_levels(), 6);
        assert_eq!(ma.seed(), 456);
    }

    #[test]
    fn test_trajectories_correct_count() {
        let mut ma = MorrisAnalyzer::<f64>::new()
            .with_trajectories(10)
            .with_seed(789);
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x: &Array1<f64>| x[0] + x[1] + x[2];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let idx = ma.analyze_morris(model, &bounds).expect("analyze failed");
        // One mu*/sigma per parameter, not per EE.
        assert_eq!(idx.mu_star.len(), 3);
        assert_eq!(idx.sigma.len(), 3);
        assert_eq!(idx.mu.len(), 3);
        assert_eq!(idx.parameter_names.len(), 3);
    }

    #[test]
    fn test_invalid_bounds_error() {
        let mut ma = MorrisAnalyzer::<f64>::new();
        let model: &dyn Fn(&Array1<f64>) -> f64 = &|x| x[0];
        let err = ma.analyze_morris(model, &[]);
        assert!(matches!(err, Err(OptimError::InvalidConfig(_))));
        let err2 = ma.analyze_morris(model, &[(1.0, 1.0)]);
        assert!(matches!(err2, Err(OptimError::InvalidConfig(_))));
    }
}
