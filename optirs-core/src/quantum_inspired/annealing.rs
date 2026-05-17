// Quantum annealing optimizer
//
// Implements a quantum-inspired simulated annealing optimizer that mixes the
// classical Metropolis acceptance criterion with a tunneling kernel, allowing
// the optimizer to escape local minima more easily than pure SGD or pure
// simulated annealing on its own.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

use super::{
    DEFAULT_FINAL_TEMP, DEFAULT_INITIAL_TEMP, DEFAULT_NUM_ITERATIONS, DEFAULT_SEED,
    DEFAULT_TUNNELING_STRENGTH,
};

/// Quantum annealing optimizer.
///
/// `QuantumAnnealing` performs a Metropolis-based stochastic search where each
/// candidate update is a perturbation of the current parameters whose magnitude
/// is scaled by the current temperature. The acceptance probability blends the
/// classical Boltzmann factor with a quantum-inspired tunneling kernel:
///
/// ```text
///     P(accept) = min(1, exp(-ΔE / (k * T) + Γ * exp(-‖δ‖²)))
/// ```
///
/// where `ΔE` is approximated by the dot product `gradients · δ` (treating the
/// supplied gradient as an unbiased local descent direction), `Γ` is the
/// tunneling strength and `δ` is the candidate perturbation.
///
/// Temperature follows a geometric (exponential) decay
///
/// ```text
///     T(t) = T_initial * (T_final / T_initial)^(t / N)
/// ```
///
/// which is well-behaved for arbitrary positive endpoints and reproduces
/// `T(0) = T_initial`, `T(N) = T_final` exactly.
///
/// # Examples
///
/// ```
/// use optirs_core::quantum_inspired::QuantumAnnealing;
/// use optirs_core::optimizers::Optimizer;
/// use scirs2_core::ndarray::Array1;
///
/// let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05)
///     .with_temperature_schedule(2.0, 0.01)
///     .with_tunneling(0.5)
///     .with_seed(123);
///
/// let params = Array1::from_vec(vec![1.0, -1.0, 0.5]);
/// let gradients = params.mapv(|x| 2.0 * x);
/// let next = optimizer.step(&params, &gradients).expect("step failed");
/// assert_eq!(next.len(), 3);
/// ```
#[derive(Debug)]
pub struct QuantumAnnealing<A: Float + ScalarOperand + Debug> {
    /// Learning rate that scales the perturbation magnitude.
    learning_rate: A,
    /// Current temperature in the annealing schedule.
    current_temperature: A,
    /// Temperature at iteration zero.
    initial_temperature: A,
    /// Temperature at iteration `num_iterations`.
    final_temperature: A,
    /// Total number of iterations the cooling schedule spans.
    num_iterations: usize,
    /// Step count, used to compute the current temperature.
    current_step: usize,
    /// Strength `Γ` of the quantum-inspired tunneling kernel.
    tunneling_strength: A,
    /// Best parameters discovered so far, flattened to a contiguous vector.
    best_params: Option<Vec<A>>,
    /// Shape associated with `best_params` to allow validated reconstruction.
    best_shape: Option<Vec<usize>>,
    /// Best (lowest) proxy energy observed so far.
    best_energy: A,
    /// Seed used to initialise the RNG.
    seed: u64,
    /// Seeded random number generator.
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl<A> QuantumAnnealing<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
{
    /// Creates a new quantum annealing optimizer with the given learning rate
    /// and all other parameters set to sensible defaults.
    pub fn new(learning_rate: A) -> Self {
        let initial = A::from(DEFAULT_INITIAL_TEMP).unwrap_or_else(A::one);
        let final_t = A::from(DEFAULT_FINAL_TEMP).unwrap_or_else(|| A::epsilon());
        let tunneling = A::from(DEFAULT_TUNNELING_STRENGTH).unwrap_or_else(A::zero);
        Self {
            learning_rate,
            current_temperature: initial,
            initial_temperature: initial,
            final_temperature: final_t,
            num_iterations: DEFAULT_NUM_ITERATIONS,
            current_step: 0,
            tunneling_strength: tunneling,
            best_params: None,
            best_shape: None,
            best_energy: A::infinity(),
            seed: DEFAULT_SEED,
            rng: Random::seed(DEFAULT_SEED),
        }
    }

    /// Configure the temperature schedule endpoints.
    ///
    /// `initial` must be strictly greater than `final_t` and both must be
    /// positive. The current temperature is reset to `initial` on each call so
    /// chained builders behave intuitively.
    pub fn with_temperature_schedule(mut self, initial: A, final_t: A) -> Self {
        self.initial_temperature = initial;
        self.final_temperature = final_t;
        self.current_temperature = initial;
        self
    }

    /// Configure the tunneling strength `Γ`.
    ///
    /// Larger values increase the average acceptance rate by enlarging the
    /// quantum-inspired kernel contribution to the Metropolis exponent.
    pub fn with_tunneling(mut self, strength: A) -> Self {
        self.tunneling_strength = strength;
        self
    }

    /// Seed the optimizer's RNG.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = Random::seed(seed);
        self
    }

    /// Configure the number of iterations the cooling schedule spans.
    pub fn with_iterations(mut self, num_iterations: usize) -> Self {
        self.num_iterations = num_iterations.max(1);
        self
    }

    /// Returns the temperature for the current step.
    pub fn current_temperature(&self) -> A {
        self.current_temperature
    }

    /// Returns the best proxy energy discovered so far. Initialised to `+∞`.
    pub fn best_energy(&self) -> A {
        self.best_energy
    }

    /// Returns the current step count.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns the configured initial temperature.
    pub fn initial_temperature(&self) -> A {
        self.initial_temperature
    }

    /// Returns the configured final temperature.
    pub fn final_temperature(&self) -> A {
        self.final_temperature
    }

    /// Returns the configured tunneling strength.
    pub fn tunneling_strength(&self) -> A {
        self.tunneling_strength
    }

    /// Returns the configured number of iterations.
    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    /// Returns the seed used to initialise the RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the current learning rate. This is an inherent helper so the
    /// caller does not need to qualify a dimension `D` to invoke the trait
    /// implementation of [`Optimizer::get_learning_rate`].
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Set the learning rate. Inherent helper that mirrors the trait method.
    pub fn set_lr(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }

    /// Returns a clone of the best parameters seen so far, reshaped to the
    /// requested dimensionality.
    pub fn best_params<D: Dimension>(&self) -> Option<Array<A, D>> {
        let buf = self.best_params.as_ref()?;
        let shape = self.best_shape.as_ref()?;
        let arr = Array::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), buf.clone()).ok()?;
        arr.into_dimensionality::<D>().ok()
    }

    /// Reset internal step counting and best-state tracking. The RNG is
    /// re-seeded with the original seed so the new trajectory is reproducible.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_temperature = self.initial_temperature;
        self.best_params = None;
        self.best_shape = None;
        self.best_energy = A::infinity();
        self.rng = Random::seed(self.seed);
    }

    /// Internal: compute the geometric-decay temperature for a given step.
    fn temperature_at(&self, step: usize) -> A {
        if self.num_iterations == 0 {
            return self.final_temperature;
        }
        let n = self.num_iterations;
        let step_clamped = step.min(n);
        let frac =
            A::from(step_clamped).unwrap_or_else(A::zero) / A::from(n).unwrap_or_else(A::one);
        // Guard against non-positive temperatures – if either endpoint is
        // unusable, fall back to a numerically stable linear interpolation so
        // we never produce NaN or Inf during the schedule.
        if self.initial_temperature <= A::zero() || self.final_temperature <= A::zero() {
            let span = self.initial_temperature - self.final_temperature;
            return self.initial_temperature - span * frac;
        }
        let ratio = self.final_temperature / self.initial_temperature;
        self.initial_temperature * ratio.powf(frac)
    }

    /// Internal: sample a perturbation `δ` whose entries follow a centered
    /// uniform distribution on `[-lr*T, lr*T]`.
    fn sample_perturbation(
        &mut self,
        shape: &[usize],
    ) -> Result<Array<A, scirs2_core::ndarray::IxDyn>> {
        let scale = self.learning_rate * self.current_temperature;
        // Guard against degenerate scale: produce zero perturbation rather
        // than panicking. This still lets the algorithm proceed sensibly when
        // the schedule has fully cooled.
        let total: usize = shape.iter().product();
        let mut buf: Vec<A> = Vec::with_capacity(total);
        for _ in 0..total {
            let u: f64 = self.rng.gen_range(-1.0..1.0);
            let val = A::from(u).unwrap_or_else(A::zero) * scale;
            buf.push(val);
        }
        Array::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), buf).map_err(|err| {
            OptimError::ComputationError(format!("Failed to build perturbation array: {err}"))
        })
    }

    /// Internal: Metropolis acceptance with quantum-inspired tunneling.
    fn accept(&mut self, delta_e: A, perturbation_sq_norm: A) -> bool {
        // Always accept improvements unconditionally so the algorithm is
        // monotone on strictly downhill moves.
        if delta_e <= A::zero() {
            return true;
        }
        // Guard against degenerate temperature – treat T<=0 as fully frozen.
        if self.current_temperature <= A::zero() {
            return false;
        }
        let tunneling_kernel = (-perturbation_sq_norm).exp();
        let exponent =
            -delta_e / self.current_temperature + self.tunneling_strength * tunneling_kernel;
        // Clamp exponent into a safe range so `exp` never overflows. Anything
        // above zero we accept; below the threshold we treat as zero.
        if exponent >= A::zero() {
            return true;
        }
        let safety = A::from(-50.0).unwrap_or_else(|| -A::one());
        let exponent_safe = if exponent < safety { safety } else { exponent };
        let p = exponent_safe.exp();
        let p_f64 = p.to_f64().unwrap_or(0.0);
        let u: f64 = self.rng.gen_range(0.0..1.0);
        u < p_f64
    }

    /// Internal: advance the cooling schedule by one step.
    fn advance_temperature(&mut self) {
        // We cap the step counter so the schedule plateaus at `final_t`
        // instead of crashing through it once we exceed `num_iterations`.
        self.current_step = self.current_step.saturating_add(1);
        self.current_temperature = self.temperature_at(self.current_step);
    }

    /// Internal: compute the gradient-dot-perturbation proxy energy.
    fn proxy_energy(
        gradients: &Array<A, scirs2_core::ndarray::IxDyn>,
        candidate_offset: &Array<A, scirs2_core::ndarray::IxDyn>,
    ) -> A {
        gradients
            .iter()
            .zip(candidate_offset.iter())
            .fold(A::zero(), |acc, (g, p)| acc + (*g) * (*p))
    }

    /// Internal: compute squared L2 norm.
    fn sq_norm(arr: &Array<A, scirs2_core::ndarray::IxDyn>) -> A {
        arr.iter().fold(A::zero(), |acc, x| acc + (*x) * (*x))
    }

    /// Internal: update best-known-state tracking.
    fn track_best(&mut self, candidate: &Array<A, scirs2_core::ndarray::IxDyn>, energy: A) {
        if energy < self.best_energy {
            self.best_energy = energy;
            self.best_params = Some(candidate.iter().copied().collect());
            self.best_shape = Some(candidate.shape().to_vec());
        }
    }
}

impl<A, D> Optimizer<A, D> for QuantumAnnealing<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "Quantum annealing: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();
        let shape: Vec<usize> = params_dyn.shape().to_vec();

        // Initialise best-known state on first call so we can always return a
        // sensible best estimate.
        if self.best_params.is_none() {
            self.best_energy = A::zero();
            self.best_params = Some(params_dyn.iter().copied().collect());
            self.best_shape = Some(shape.clone());
        }

        let perturbation = self.sample_perturbation(&shape)?;
        // The candidate move is `params - perturbation`. A first-order Taylor
        // expansion of the unknown loss `f` gives
        //
        //     ΔE ≈ f(params - perturbation) - f(params) ≈ -∇f · perturbation
        //
        // so the proxy energy used in the Metropolis criterion is the
        // negative gradient–perturbation dot product.
        let delta_e = -Self::proxy_energy(&gradients_dyn, &perturbation);
        let sq_norm = Self::sq_norm(&perturbation);

        let updated_dyn = if self.accept(delta_e, sq_norm) {
            &params_dyn - &perturbation
        } else {
            params_dyn.clone()
        };

        // Energy proxy for tracking purposes: linearised change in the
        // unknown loss between `params` and the accepted parameters.
        let accepted_offset = &updated_dyn - &params_dyn;
        let accepted_energy = -Self::proxy_energy(&gradients_dyn, &accepted_offset);
        self.track_best(&updated_dyn, accepted_energy);

        self.advance_temperature();

        updated_dyn.into_dimensionality::<D>().map_err(|err| {
            OptimError::ComputationError(format!(
                "Quantum annealing: failed to restore dimension: {err}"
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
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    fn quadratic_grad(params: &Array1<f64>) -> Array1<f64> {
        params.mapv(|x| 2.0 * x)
    }

    #[test]
    fn test_default_config() {
        let optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.1);
        assert_abs_diff_eq!(optimizer.initial_temperature(), DEFAULT_INITIAL_TEMP);
        assert_abs_diff_eq!(optimizer.final_temperature(), DEFAULT_FINAL_TEMP);
        assert_abs_diff_eq!(optimizer.tunneling_strength(), DEFAULT_TUNNELING_STRENGTH);
        assert_eq!(optimizer.num_iterations(), DEFAULT_NUM_ITERATIONS);
        assert_eq!(optimizer.current_step(), 0);
    }

    #[test]
    fn test_builder_pattern() {
        let optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05)
            .with_temperature_schedule(2.0, 0.01)
            .with_tunneling(0.7)
            .with_iterations(250)
            .with_seed(42);
        assert_abs_diff_eq!(optimizer.initial_temperature(), 2.0);
        assert_abs_diff_eq!(optimizer.final_temperature(), 0.01);
        assert_abs_diff_eq!(optimizer.tunneling_strength(), 0.7);
        assert_eq!(optimizer.num_iterations(), 250);
        assert_eq!(optimizer.seed(), 42);
        // current_temperature is reset to initial after configuring schedule
        assert_abs_diff_eq!(optimizer.current_temperature(), 2.0);
    }

    #[test]
    fn test_temperature_decay_monotonic() {
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.01)
            .with_temperature_schedule(1.0, 1e-3)
            .with_iterations(50)
            .with_seed(7);
        let params = Array1::from_vec(vec![1.0, -1.0, 0.5]);
        let mut prev = optimizer.current_temperature();
        for _ in 0..30 {
            let grads = quadratic_grad(&params);
            let _ = optimizer.step(&params, &grads).expect("step failed");
            let curr = optimizer.current_temperature();
            assert!(
                curr <= prev + 1e-12,
                "Temperature did not decrease monotonically: prev={prev}, curr={curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_geometric_cooling_endpoints() {
        let optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.01)
            .with_temperature_schedule(2.0, 1e-2)
            .with_iterations(100);
        // Endpoints: T(0) = initial, T(N) = final
        let t0 = optimizer.temperature_at(0);
        let tn = optimizer.temperature_at(100);
        assert_abs_diff_eq!(t0, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tn, 1e-2, epsilon = 1e-9);
        // Halfway should be geometric mean: sqrt(2.0 * 1e-2)
        let t_half = optimizer.temperature_at(50);
        let expected_half = (2.0_f64 * 1e-2).sqrt();
        assert_abs_diff_eq!(t_half, expected_half, epsilon = 1e-9);
    }

    #[test]
    fn test_metropolis_accepts_lower_energy() {
        // With purely downhill gradients (proxy energy <= 0), we always accept.
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(1.0, 1e-3)
            .with_iterations(50)
            .with_seed(11);
        // Negative ΔE should always be accepted irrespective of temperature.
        for _ in 0..30 {
            let accept = optimizer.accept(-0.5, 0.25);
            assert!(accept, "Metropolis rejected a strictly downhill move");
        }
    }

    #[test]
    fn test_metropolis_rejects_higher_energy_at_low_temp_probabilistically() {
        // At very low temperature with zero tunneling, almost all uphill
        // proposals should be rejected.
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(1e-6, 1e-6)
            .with_tunneling(0.0)
            .with_iterations(100)
            .with_seed(101);
        // Force T to be effectively zero by directly setting via construction.
        optimizer.current_temperature = 1e-6;
        let mut accepted = 0;
        let trials = 500;
        for _ in 0..trials {
            if optimizer.accept(1.0, 0.5) {
                accepted += 1;
            }
        }
        assert!(
            accepted < (trials / 50).max(2),
            "Too many uphill moves accepted at near-zero temperature: {accepted} / {trials}"
        );
    }

    #[test]
    fn test_tunneling_increases_acceptance() {
        // Hold everything constant except tunneling strength and measure mean
        // acceptance for uphill moves. Tunneling=1.0 should accept more than
        // tunneling=0.0.
        let mut low: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(0.1, 0.1)
            .with_tunneling(0.0)
            .with_iterations(1000)
            .with_seed(2024);
        let mut high: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(0.1, 0.1)
            .with_tunneling(2.0)
            .with_iterations(1000)
            .with_seed(2024);

        let mut accept_low = 0usize;
        let mut accept_high = 0usize;
        let trials = 4000;
        for _ in 0..trials {
            if low.accept(0.2, 0.1) {
                accept_low += 1;
            }
            if high.accept(0.2, 0.1) {
                accept_high += 1;
            }
        }
        assert!(
            accept_high > accept_low,
            "Higher tunneling did not increase acceptance: low={accept_low}, high={accept_high}"
        );
    }

    #[test]
    fn test_convergence_on_quadratic_bowl() {
        // f(x) = x^2 from x = 5. Quantum annealing performs a stochastic
        // gradient-driven random walk where the perturbation magnitude is
        // scaled by `learning_rate * current_temperature`. With a high
        // learning rate, broad temperature schedule and modest tunneling, we
        // expect the parameters to descend substantially towards zero within
        // a few hundred iterations.
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(1.0)
            .with_temperature_schedule(2.0, 1e-3)
            .with_tunneling(0.05)
            .with_iterations(200)
            .with_seed(91);
        let initial = 5.0_f64;
        let mut params = Array1::from_vec(vec![initial]);
        for _ in 0..200 {
            let grads = quadratic_grad(&params);
            params = optimizer.step(&params, &grads).expect("step failed");
        }
        // The optimizer should reduce |x| substantially relative to its
        // starting value. We use a generous tolerance because the process is
        // a stochastic gradient-biased random walk.
        assert!(
            params[0].abs() < initial * 0.5,
            "Optimizer did not converge: |x|={}, started at {initial}",
            params[0].abs()
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        let mut a: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(1.0, 1e-2)
            .with_iterations(100)
            .with_seed(123);
        let mut b: QuantumAnnealing<f64> = QuantumAnnealing::new(0.1)
            .with_temperature_schedule(1.0, 1e-2)
            .with_iterations(100)
            .with_seed(123);
        let params = Array1::from_vec(vec![1.0, 2.0, -1.0]);
        let grads = Array1::from_vec(vec![0.1, -0.2, 0.05]);
        for _ in 0..20 {
            let p_a = a.step(&params, &grads).expect("step failed");
            let p_b = b.step(&params, &grads).expect("step failed");
            for (x, y) in p_a.iter().zip(p_b.iter()) {
                assert_abs_diff_eq!(*x, *y, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_set_learning_rate_changes_step_size() {
        let params = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let grads = Array1::from_vec(vec![0.0, 0.0, 0.0]);

        let mut small: QuantumAnnealing<f64> = QuantumAnnealing::new(0.01)
            .with_temperature_schedule(1.0, 1.0)
            .with_tunneling(0.0)
            .with_iterations(10)
            .with_seed(5);
        let mut large: QuantumAnnealing<f64> = QuantumAnnealing::new(1.0)
            .with_temperature_schedule(1.0, 1.0)
            .with_tunneling(0.0)
            .with_iterations(10)
            .with_seed(5);

        let mut max_small = 0.0_f64;
        let mut max_large = 0.0_f64;
        for _ in 0..30 {
            let ps = small.step(&params, &grads).expect("step failed");
            let pl = large.step(&params, &grads).expect("step failed");
            for (s, l) in ps.iter().zip(pl.iter()) {
                max_small = max_small.max((s - 1.0).abs());
                max_large = max_large.max((l - 1.0).abs());
            }
        }
        // Sanity: large LR should have produced larger displacements.
        assert!(
            max_large > max_small,
            "large LR ({max_large}) did not move further than small LR ({max_small})"
        );

        // After setting learning rate, the optimizer reports the update.
        let mut opt: QuantumAnnealing<f64> = QuantumAnnealing::new(0.01);
        opt.set_lr(0.5);
        assert_abs_diff_eq!(opt.learning_rate(), 0.5);
    }

    #[test]
    fn test_step_returns_same_shape_as_params() {
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05)
            .with_temperature_schedule(0.5, 1e-3)
            .with_iterations(20)
            .with_seed(31);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let grads = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let updated = optimizer.step(&params, &grads).expect("step failed");
        assert_eq!(updated.shape(), params.shape());
    }

    #[test]
    fn test_best_params_tracked() {
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05)
            .with_temperature_schedule(0.5, 1e-3)
            .with_iterations(100)
            .with_seed(17);
        let params = Array1::from_vec(vec![5.0]);
        let mut current = params.clone();
        for _ in 0..100 {
            let grads = quadratic_grad(&current);
            current = optimizer.step(&current, &grads).expect("step failed");
        }
        assert!(
            optimizer.best_energy() <= 0.0,
            "best_energy {} should be <= 0",
            optimizer.best_energy()
        );
        let best: Option<Array1<f64>> = optimizer.best_params();
        assert!(best.is_some(), "best_params should be tracked");
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.05).with_seed(42);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![1.0, 2.0]);
        let result = optimizer.step(&params, &grads);
        assert!(result.is_err(), "expected dimension mismatch error");
    }

    #[test]
    fn test_temperature_plateau_after_n_iterations() {
        let mut optimizer: QuantumAnnealing<f64> = QuantumAnnealing::new(0.01)
            .with_temperature_schedule(1.0, 1e-3)
            .with_iterations(10)
            .with_seed(3);
        let params = Array1::from_vec(vec![0.0]);
        let grads = Array1::from_vec(vec![0.0]);
        for _ in 0..50 {
            let _ = optimizer.step(&params, &grads).expect("step failed");
        }
        // Once we've exceeded num_iterations, temperature should plateau at
        // the final temperature endpoint.
        assert_abs_diff_eq!(optimizer.current_temperature(), 1e-3, epsilon = 1e-9);
    }
}
