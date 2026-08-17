// Variational Quantum Eigensolver (VQE) inspired optimizer based on SPSA.
//
// This module implements a SPSA (Simultaneous Perturbation Stochastic
// Approximation) optimizer with a quantum-inspired ansatz. SPSA is the
// optimizer of choice for hardware VQE because it estimates gradients with
// only two loss evaluations regardless of dimensionality.

use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

use super::DEFAULT_SEED;

/// Default SPSA gain `a` (numerator of the learning-rate schedule).
pub(crate) const DEFAULT_SPSA_A: f64 = 0.1;
/// Default SPSA perturbation `c` (numerator of the perturbation schedule).
pub(crate) const DEFAULT_SPSA_C: f64 = 0.1;
/// Default SPSA exponent `α` (gain decay).
pub(crate) const DEFAULT_SPSA_ALPHA: f64 = 0.602;
/// Default SPSA exponent `γ` (perturbation decay).
pub(crate) const DEFAULT_SPSA_GAMMA: f64 = 0.101;
/// Default SPSA stability `A` (offset that softens early-iteration steps).
pub(crate) const DEFAULT_SPSA_BIG_A: f64 = 10.0;

/// Variational Quantum Optimizer.
///
/// `VariationalQuantumOptimizer` implements a SPSA optimizer with a
/// quantum-inspired ansatz update rule. SPSA approximates the gradient with
///
/// ```text
///     g_i(k) ≈ (L(θ + c_k * Δ) - L(θ - c_k * Δ)) / (2 * c_k * Δ_i)
/// ```
///
/// where `Δ ∈ {-1, +1}^d` is sampled uniformly at every iteration. The gain
/// sequences follow the canonical Spall (1998) recipe:
///
/// ```text
///     a_k = a / (k + 1 + A)^α
///     c_k = c / (k + 1)^γ
/// ```
///
/// The "quantum ansatz" applies a rotation-gate-inspired factor `cos²(θ_i / 2)`
/// to the SPSA update, smoothing updates near `θ_i = 0` (mimicking how a
/// rotation gate has unit effect near identity) and vanishing near `θ_i = π`.
///
/// # Examples
///
/// ```
/// use optirs_core::quantum_inspired::VariationalQuantumOptimizer;
/// use scirs2_core::ndarray::Array1;
///
/// let mut optimizer: VariationalQuantumOptimizer<f64> =
///     VariationalQuantumOptimizer::new(0.1)
///         .with_perturbation(0.05)
///         .with_seed(7);
///
/// let params = Array1::from_vec(vec![0.5, -0.3, 1.2]);
/// let loss_fn = |theta: &Array1<f64>| theta.iter().map(|x| x * x).sum::<f64>();
/// let next = optimizer.step_from_loss(&params, loss_fn).expect("step failed");
/// assert_eq!(next.len(), 3);
/// ```
#[derive(Debug)]
pub struct VariationalQuantumOptimizer<A: Float + ScalarOperand + Debug> {
    /// SPSA gain numerator `a` (also used as the canonical learning rate).
    learning_rate: A,
    /// SPSA perturbation numerator `c`.
    c: A,
    /// SPSA gain decay exponent `α`.
    alpha: A,
    /// SPSA perturbation decay exponent `γ`.
    gamma: A,
    /// SPSA stability `A` that softens the first few learning-rate steps.
    big_a: A,
    /// Current step counter `k`, starting at `0`.
    step: usize,
    /// Last observed loss value (for diagnostics).
    last_loss: Option<A>,
    /// Seed used to initialise the RNG.
    seed: u64,
    /// Seeded RNG.
    rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Optional cached ansatz parameters (used in the trait-driven step).
    ansatz_params: Option<Array1<A>>,
}

impl<A> VariationalQuantumOptimizer<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
{
    /// Create a VQE-inspired SPSA optimizer with the canonical SPSA gain
    /// `a = 0.1`, matching the defaults already used for `c`, `α`, `γ` and
    /// `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optirs_core::quantum_inspired::VariationalQuantumOptimizer;
    ///
    /// let optimizer = VariationalQuantumOptimizer::<f64>::with_default_gain();
    /// assert!((optimizer.learning_rate() - 0.1).abs() < 1e-12);
    /// ```
    pub fn with_default_gain() -> Self {
        Self::new(A::from(DEFAULT_SPSA_A).unwrap_or_else(A::one))
    }

    /// Create a new VQE-inspired SPSA optimizer with the given learning rate.
    pub fn new(learning_rate: A) -> Self {
        let c = A::from(DEFAULT_SPSA_C).unwrap_or_else(|| A::epsilon());
        let alpha = A::from(DEFAULT_SPSA_ALPHA).unwrap_or_else(A::one);
        let gamma = A::from(DEFAULT_SPSA_GAMMA).unwrap_or_else(A::one);
        let big_a = A::from(DEFAULT_SPSA_BIG_A).unwrap_or_else(A::zero);
        Self {
            learning_rate,
            c,
            alpha,
            gamma,
            big_a,
            step: 0,
            last_loss: None,
            seed: DEFAULT_SEED,
            rng: Random::seed(DEFAULT_SEED),
            ansatz_params: None,
        }
    }

    /// Configure the SPSA perturbation magnitude `c`.
    pub fn with_perturbation(mut self, c: A) -> Self {
        self.c = c;
        self
    }

    /// Configure the SPSA decay exponents `α` (gain) and `γ` (perturbation).
    pub fn with_gain_decay(mut self, alpha: A, gamma: A) -> Self {
        self.alpha = alpha;
        self.gamma = gamma;
        self
    }

    /// Configure the SPSA stability offset `A`.
    pub fn with_stability(mut self, big_a: A) -> Self {
        self.big_a = big_a;
        self
    }

    /// Seed the optimizer's RNG.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Random::seed(seed);
        self
    }

    /// Returns the SPSA `α` exponent.
    pub fn alpha(&self) -> A {
        self.alpha
    }

    /// Returns the SPSA `γ` exponent.
    pub fn gamma(&self) -> A {
        self.gamma
    }

    /// Returns the SPSA stability offset `A`.
    pub fn big_a(&self) -> A {
        self.big_a
    }

    /// Returns the SPSA perturbation numerator `c`.
    pub fn c(&self) -> A {
        self.c
    }

    /// Returns the current step counter `k`.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Returns the most recently observed loss value, if any.
    pub fn last_loss(&self) -> Option<A> {
        self.last_loss
    }

    /// Returns the seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the learning rate. Inherent helper that mirrors the trait
    /// method [`Optimizer::get_learning_rate`] so callers do not need to
    /// disambiguate the dimension type.
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Set the learning rate. Inherent helper that mirrors the trait method.
    pub fn set_lr(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }

    /// SPSA gain `a_k`.
    pub fn a_k(&self, k: usize) -> A {
        let k_f = A::from(k).unwrap_or_else(A::zero);
        let one = A::one();
        let denom = (k_f + one + self.big_a).powf(self.alpha);
        if denom <= A::zero() {
            self.learning_rate
        } else {
            self.learning_rate / denom
        }
    }

    /// SPSA perturbation `c_k`.
    pub fn c_k(&self, k: usize) -> A {
        let k_f = A::from(k).unwrap_or_else(A::zero);
        let one = A::one();
        let denom = (k_f + one).powf(self.gamma);
        if denom <= A::zero() {
            self.c
        } else {
            self.c / denom
        }
    }

    /// Reset the step counter and re-seed the RNG.
    pub fn reset(&mut self) {
        self.step = 0;
        self.last_loss = None;
        self.rng = Random::seed(self.seed);
        self.ansatz_params = None;
    }

    /// Quantum-inspired ansatz factor `cos²(θ_i / 2)`. Public for testing.
    pub fn ansatz_factor(theta: A) -> A {
        let half = A::from(0.5).unwrap_or_else(A::one);
        let c = (theta * half).cos();
        c * c
    }

    /// Sample a fresh SPSA perturbation `Δ ∈ {-1, +1}^d`.
    pub(crate) fn sample_perturbation_vector(&mut self, dim: usize) -> Array1<A> {
        let mut buf: Vec<A> = Vec::with_capacity(dim);
        let one = A::one();
        let neg_one = -A::one();
        for _ in 0..dim {
            let u: f64 = self.rng.gen_range(0.0..1.0);
            buf.push(if u < 0.5 { neg_one } else { one });
        }
        Array1::from_vec(buf)
    }

    /// Compute an SPSA gradient estimate using the supplied loss function.
    ///
    /// Returns `(gradient, c_k, delta)`.
    pub fn spsa_gradient<F>(
        &mut self,
        params: &Array1<A>,
        loss_fn: F,
        k: usize,
    ) -> Result<(Array1<A>, A, Array1<A>)>
    where
        F: Fn(&Array1<A>) -> A,
    {
        let dim = params.len();
        if dim == 0 {
            return Err(OptimError::InvalidParameter(
                "VariationalQuantumOptimizer: parameters must be non-empty".to_string(),
            ));
        }
        let c_k = self.c_k(k);
        if c_k <= A::zero() {
            return Err(OptimError::InvalidConfig(
                "VariationalQuantumOptimizer: c_k must be positive".to_string(),
            ));
        }
        let delta = self.sample_perturbation_vector(dim);
        let plus = params + &(&delta * c_k);
        let minus = params - &(&delta * c_k);
        let loss_plus = loss_fn(&plus);
        let loss_minus = loss_fn(&minus);
        let two = A::from(2.0).unwrap_or_else(A::one);
        let numerator = loss_plus - loss_minus;
        let denom = two * c_k;
        let mut grad = Array1::<A>::zeros(dim);
        for i in 0..dim {
            // Δ_i ∈ {-1, +1} so dividing is safe and equivalent to multiplying.
            let d = delta[i];
            grad[i] = numerator / (denom * d);
        }
        Ok((grad, c_k, delta))
    }

    /// Apply the quantum-inspired ansatz update.
    ///
    /// `new_θ_i = θ_i - a_k * g_i * cos²(θ_i / 2)`
    fn apply_ansatz(&self, params: &Array1<A>, grad: &Array1<A>, k: usize) -> Array1<A> {
        let a_k = self.a_k(k);
        let mut updated = params.clone();
        for i in 0..updated.len() {
            let factor = Self::ansatz_factor(updated[i]);
            updated[i] = updated[i] - a_k * grad[i] * factor;
        }
        updated
    }

    /// Perform a loss-driven SPSA step.
    ///
    /// This is the canonical VQE-style update that uses a closed-form loss
    /// rather than relying on user-provided gradients.
    pub fn step_from_loss<F>(&mut self, params: &Array1<A>, loss_fn: F) -> Result<Array1<A>>
    where
        F: Fn(&Array1<A>) -> A,
    {
        let k = self.step;
        let (grad, _c_k, _delta) = self.spsa_gradient(params, &loss_fn, k)?;
        let updated = self.apply_ansatz(params, &grad, k);
        self.last_loss = Some(loss_fn(&updated));
        self.step = self.step.saturating_add(1);
        self.ansatz_params = Some(updated.clone());
        Ok(updated)
    }
}

impl<A, D> Optimizer<A, D> for VariationalQuantumOptimizer<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "VQE optimizer: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let params_dyn = params.to_owned().into_dyn();
        let grads_dyn = gradients.to_owned().into_dyn();
        let k = self.step;
        let a_k = self.a_k(k);

        // Apply the quantum-inspired ansatz on each parameter using the
        // user-supplied gradient directly (so the optimizer remains a true
        // Optimizer<A, D> on top of pre-computed gradients).
        let mut updated = params_dyn.clone();
        for (out, (p, g)) in updated
            .iter_mut()
            .zip(params_dyn.iter().zip(grads_dyn.iter()))
        {
            let factor = Self::ansatz_factor(*p);
            *out = *p - a_k * (*g) * factor;
        }

        self.step = self.step.saturating_add(1);

        updated.into_dimensionality::<D>().map_err(|err| {
            OptimError::ComputationError(format!(
                "VQE optimizer: failed to restore dimension: {err}"
            ))
        })
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn test_default_config_values() {
        let optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.1);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.1);
        assert_abs_diff_eq!(optimizer.c(), DEFAULT_SPSA_C);
        assert_abs_diff_eq!(optimizer.alpha(), DEFAULT_SPSA_ALPHA);
        assert_abs_diff_eq!(optimizer.gamma(), DEFAULT_SPSA_GAMMA);
        assert_abs_diff_eq!(optimizer.big_a(), DEFAULT_SPSA_BIG_A);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_builder_pattern() {
        let optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.05)
            .with_perturbation(0.2)
            .with_gain_decay(0.5, 0.2)
            .with_stability(20.0)
            .with_seed(99);
        assert_abs_diff_eq!(optimizer.c(), 0.2);
        assert_abs_diff_eq!(optimizer.alpha(), 0.5);
        assert_abs_diff_eq!(optimizer.gamma(), 0.2);
        assert_abs_diff_eq!(optimizer.big_a(), 20.0);
        assert_eq!(optimizer.seed(), 99);
    }

    #[test]
    fn test_gain_sequences_endpoints() {
        let optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.1)
            .with_perturbation(0.1)
            .with_gain_decay(0.602, 0.101)
            .with_stability(10.0);
        // a_0 = a / (0 + 1 + A)^α = 0.1 / 11^0.602
        let expected_a0 = 0.1_f64 / 11.0_f64.powf(0.602);
        assert_relative_eq!(optimizer.a_k(0), expected_a0, epsilon = 1e-12);
        // c_0 = c / (0 + 1)^γ = 0.1 / 1 = 0.1
        assert_abs_diff_eq!(optimizer.c_k(0), 0.1);
        // c_k decays at k=10: 0.1 / 11^0.101
        let expected_c10 = 0.1_f64 / 11.0_f64.powf(0.101);
        assert_relative_eq!(optimizer.c_k(10), expected_c10, epsilon = 1e-12);
    }

    #[test]
    fn test_gain_sequences_decay() {
        let optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.1);
        // a_k and c_k must be strictly decreasing because all exponents and
        // denominators are positive.
        for k in 0..50 {
            assert!(
                optimizer.a_k(k + 1) < optimizer.a_k(k),
                "a_k not decreasing at k={k}"
            );
            assert!(
                optimizer.c_k(k + 1) < optimizer.c_k(k),
                "c_k not decreasing at k={k}"
            );
        }
    }

    #[test]
    fn test_perturbation_is_pm_one() {
        let mut optimizer: VariationalQuantumOptimizer<f64> =
            VariationalQuantumOptimizer::new(0.1).with_seed(13);
        for _ in 0..20 {
            let delta = optimizer.sample_perturbation_vector(32);
            for v in delta.iter() {
                assert!(
                    (*v == 1.0) || (*v == -1.0),
                    "perturbation entry {} was not in {{-1, 1}}",
                    *v
                );
            }
        }
    }

    #[test]
    fn test_spsa_gradient_unbiasedness() {
        // For a quadratic loss L(θ) = ||θ||² the analytical gradient at θ is
        // 2θ. The SPSA gradient is exactly unbiased for quadratic objectives
        // (the second-order Taylor term cancels in the central difference) so
        // averaging over many trials should converge in probability to 2θ.
        //
        // We average across many seeds to drive variance down even when each
        // individual mean has substantial residual variance.
        let theta = Array1::from_vec(vec![1.5, -0.5, 0.25]);
        let mut accum = Array1::<f64>::zeros(theta.len());
        let trials_per_seed = 2000;
        let num_seeds = 5;
        let mut total_trials = 0usize;
        for seed in 0..num_seeds {
            let mut optimizer: VariationalQuantumOptimizer<f64> =
                VariationalQuantumOptimizer::new(0.01)
                    .with_perturbation(0.01)
                    .with_seed(2024 + seed as u64);
            for _ in 0..trials_per_seed {
                let (g, _c, _d) = optimizer
                    .spsa_gradient(&theta, |x| x.iter().map(|v| v * v).sum::<f64>(), 0)
                    .expect("spsa gradient failed");
                accum = &accum + &g;
                total_trials += 1;
            }
        }
        let n = total_trials as f64;
        for i in 0..theta.len() {
            let mean = accum[i] / n;
            let expected = 2.0 * theta[i];
            // SPSA is exactly unbiased for quadratic losses, so the mean
            // should converge to 2θ. Variance per trial for θ=[1.5,-0.5,0.25]
            // is O(1) so std of mean with n=10_000 is ~0.04 → 0.2 tolerance
            // is a safe ~5σ bound.
            assert!(
                (mean - expected).abs() < 0.2,
                "SPSA gradient bias too large at index {i}: mean={mean}, expected={expected}, n={n}"
            );
        }
    }

    #[test]
    fn test_step_from_loss_decreases_loss() {
        let mut optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.5)
            .with_perturbation(0.05)
            .with_gain_decay(0.602, 0.101)
            .with_stability(2.0)
            .with_seed(77);
        let mut params = Array1::from_vec(vec![1.2, -0.8, 0.5]);
        let loss_fn = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();
        let initial = loss_fn(&params);
        for _ in 0..150 {
            params = optimizer
                .step_from_loss(&params, loss_fn)
                .expect("step failed");
        }
        let final_loss = loss_fn(&params);
        assert!(
            final_loss < initial,
            "Loss did not decrease: initial={initial}, final={final_loss}"
        );
    }

    #[test]
    fn test_ansatz_smoothness() {
        // cos²(θ/2) → 1 as θ → 0
        let near_zero = VariationalQuantumOptimizer::<f64>::ansatz_factor(0.0);
        assert_abs_diff_eq!(near_zero, 1.0, epsilon = 1e-12);
        // cos²(θ/2) → 0 as θ → π
        let at_pi = VariationalQuantumOptimizer::<f64>::ansatz_factor(PI);
        assert_abs_diff_eq!(at_pi, 0.0, epsilon = 1e-12);
        // intermediate value smoothly between 0 and 1
        let mid = VariationalQuantumOptimizer::<f64>::ansatz_factor(PI / 2.0);
        assert!(mid > 0.4 && mid < 0.6, "midpoint ansatz factor = {mid}");
    }

    #[test]
    fn test_seed_reproducibility() {
        let mut a: VariationalQuantumOptimizer<f64> =
            VariationalQuantumOptimizer::new(0.1).with_seed(321);
        let mut b: VariationalQuantumOptimizer<f64> =
            VariationalQuantumOptimizer::new(0.1).with_seed(321);
        let params = Array1::from_vec(vec![0.2, -0.4, 0.6, -0.8]);
        let loss_fn = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();
        for _ in 0..25 {
            let pa = a.step_from_loss(&params, loss_fn).expect("step failed");
            let pb = b.step_from_loss(&params, loss_fn).expect("step failed");
            for (x, y) in pa.iter().zip(pb.iter()) {
                assert_abs_diff_eq!(*x, *y, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_get_set_learning_rate() {
        let mut optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.3);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.3);
        optimizer.set_lr(0.05);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.05);
    }

    #[test]
    fn test_step_returns_same_shape() {
        let mut optimizer: VariationalQuantumOptimizer<f64> =
            VariationalQuantumOptimizer::new(0.1).with_seed(5);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.5, -0.3, 0.2, 0.1]);
        let updated = <VariationalQuantumOptimizer<f64> as Optimizer<f64, _>>::step(
            &mut optimizer,
            &params,
            &grads,
        )
        .expect("step failed");
        assert_eq!(updated.shape(), params.shape());
    }

    #[test]
    fn test_convergence_on_quadratic() {
        let mut optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(1.0)
            .with_perturbation(0.05)
            .with_gain_decay(0.602, 0.101)
            .with_stability(2.0)
            .with_seed(31);
        let mut params = Array1::from_vec(vec![1.0, -1.0]);
        let loss_fn = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();
        for _ in 0..400 {
            params = optimizer
                .step_from_loss(&params, loss_fn)
                .expect("step failed");
        }
        // Parameters near zero (the global minimum).
        for v in params.iter() {
            assert!(v.abs() < 0.5, "Parameter did not converge: |x|={}", v.abs());
        }
    }

    #[test]
    fn test_step_count_increments() {
        let mut optimizer: VariationalQuantumOptimizer<f64> =
            VariationalQuantumOptimizer::new(0.1).with_seed(1);
        let params = Array1::from_vec(vec![0.5, -0.5]);
        let loss_fn = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();
        assert_eq!(optimizer.step_count(), 0);
        for i in 1..=5 {
            let _ = optimizer
                .step_from_loss(&params, loss_fn)
                .expect("step failed");
            assert_eq!(optimizer.step_count(), i);
        }
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut optimizer: VariationalQuantumOptimizer<f64> = VariationalQuantumOptimizer::new(0.1);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![1.0, 2.0]);
        let result = <VariationalQuantumOptimizer<f64> as Optimizer<f64, _>>::step(
            &mut optimizer,
            &params,
            &grads,
        );
        assert!(result.is_err(), "expected dimension mismatch error");
    }
}
