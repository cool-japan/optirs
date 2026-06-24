//! Quantum-inspired learned optimizers.
//!
//! This module bridges the three quantum-inspired optimizers from
//! [`optirs_core::quantum_inspired`] into the learned-optimizer interface
//! [`crate::domain_optimizers::AdvancedOptimizer`]. The core optimizers operate
//! over arbitrary [`scirs2_core::ndarray::Dimension`]s through the
//! [`optirs_core::optimizers::Optimizer`] trait; here we specialise them to the
//! one-dimensional parameter vectors (`Array1<T>`) used throughout the learned
//! optimizer stack and layer on the bookkeeping the learned interface expects
//! (step counting, gradient-norm EMA, and an [`OptimizerStateInfo`] snapshot).
//!
//! Three backends are wrapped:
//!
//! - **Quantum annealing** ([`QuantumAnnealing`]) — a Metropolis-driven
//!   stochastic search with a quantum-inspired tunneling kernel.
//! - **Variational quantum** ([`VariationalQuantumOptimizer`]) — a SPSA
//!   optimizer with a rotation-gate-inspired ansatz factor.
//! - **Hybrid quantum-classical** ([`HybridQuantumClassical`]) — a two-phase
//!   optimizer that explores with annealing and refines with Adam.
//!
//! All three are exposed behind a single concrete type
//! [`QuantumLearnedOptimizer`] selected at construction time.
//!
//! # Examples
//!
//! ```
//! use optirs_learned::domain_optimizers::AdvancedOptimizer;
//! use optirs_learned::quantum_learned::QuantumLearnedOptimizer;
//! use scirs2_core::ndarray::Array1;
//!
//! let mut optimizer: QuantumLearnedOptimizer<f64> =
//!     QuantumLearnedOptimizer::annealing(0.05);
//! let params = Array1::from_vec(vec![1.0, -1.0, 0.5]);
//! let gradients = params.mapv(|x| 2.0 * x);
//! let next = optimizer.step(&params, &gradients).expect("step succeeds");
//! assert_eq!(next.len(), params.len());
//! ```

use std::fmt::Debug;

use scirs2_core::ndarray::{Array1, Ix1, ScalarOperand};
use scirs2_core::numeric::Float;

use optirs_core::optimizers::Optimizer;
use optirs_core::quantum_inspired::{
    HybridQuantumClassical, QuantumAnnealing, VariationalQuantumOptimizer,
};

use crate::domain_optimizers::{l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};

/// Default decay used for the gradient-norm exponential moving average.
const DEFAULT_NORM_EMA_DECAY: f64 = 0.9;

/// The concrete quantum-inspired optimizer driving a [`QuantumLearnedOptimizer`].
///
/// Each variant holds a fully-configured core optimizer from
/// [`optirs_core::quantum_inspired`]. The wrapper dispatches the
/// [`AdvancedOptimizer`] interface to the held optimizer through the
/// [`Optimizer`] trait specialised to [`Ix1`].
#[derive(Debug)]
pub enum QuantumBackend<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync + 'static,
{
    /// Metropolis-based quantum annealing exploration.
    Annealing(QuantumAnnealing<T>),
    /// SPSA-based variational quantum optimizer.
    Variational(VariationalQuantumOptimizer<T>),
    /// Two-phase hybrid quantum-classical optimizer.
    Hybrid(HybridQuantumClassical<T>),
}

impl<T> QuantumBackend<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync + 'static,
{
    /// Perform one optimization step on a one-dimensional parameter vector.
    ///
    /// Dispatches to the held core optimizer's [`Optimizer`] implementation at
    /// `D = Ix1` and maps any core error into a learned
    /// [`OptimError::ComputationError`].
    fn step_array1(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        let outcome = match self {
            QuantumBackend::Annealing(inner) => {
                <QuantumAnnealing<T> as Optimizer<T, Ix1>>::step(inner, params, gradients)
            }
            QuantumBackend::Variational(inner) => <VariationalQuantumOptimizer<T> as Optimizer<
                T,
                Ix1,
            >>::step(inner, params, gradients),
            QuantumBackend::Hybrid(inner) => {
                <HybridQuantumClassical<T> as Optimizer<T, Ix1>>::step(inner, params, gradients)
            }
        };
        outcome.map_err(|err| OptimError::ComputationError(err.to_string()))
    }

    /// Returns the current learning rate of the held core optimizer.
    fn learning_rate(&self) -> T {
        match self {
            QuantumBackend::Annealing(inner) => {
                <QuantumAnnealing<T> as Optimizer<T, Ix1>>::get_learning_rate(inner)
            }
            QuantumBackend::Variational(inner) => {
                <VariationalQuantumOptimizer<T> as Optimizer<T, Ix1>>::get_learning_rate(inner)
            }
            QuantumBackend::Hybrid(inner) => {
                <HybridQuantumClassical<T> as Optimizer<T, Ix1>>::get_learning_rate(inner)
            }
        }
    }

    /// Sets the learning rate of the held core optimizer.
    fn set_learning_rate(&mut self, learning_rate: T) {
        match self {
            QuantumBackend::Annealing(inner) => {
                <QuantumAnnealing<T> as Optimizer<T, Ix1>>::set_learning_rate(inner, learning_rate)
            }
            QuantumBackend::Variational(inner) => {
                <VariationalQuantumOptimizer<T> as Optimizer<T, Ix1>>::set_learning_rate(
                    inner,
                    learning_rate,
                )
            }
            QuantumBackend::Hybrid(inner) => {
                <HybridQuantumClassical<T> as Optimizer<T, Ix1>>::set_learning_rate(
                    inner,
                    learning_rate,
                )
            }
        }
    }

    /// Returns the stable display name for this backend.
    fn name(&self) -> &'static str {
        match self {
            QuantumBackend::Annealing(_) => "QuantumAnnealingLearned",
            QuantumBackend::Variational(_) => "VariationalQuantumLearned",
            QuantumBackend::Hybrid(_) => "HybridQuantumLearned",
        }
    }
}

/// A learned-interface adapter over the quantum-inspired core optimizers.
///
/// `QuantumLearnedOptimizer` wraps one of the [`QuantumBackend`] variants and
/// implements [`AdvancedOptimizer`], tracking the step count and an
/// exponential moving average of the gradient L2 norm so the learned stack can
/// query optimizer health via [`AdvancedOptimizer::get_state`].
#[derive(Debug)]
pub struct QuantumLearnedOptimizer<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync + 'static,
{
    /// The quantum-inspired optimizer doing the actual work.
    backend: QuantumBackend<T>,
    /// Total number of successful optimization steps taken.
    step_count: usize,
    /// Exponential moving average of the gradient L2 norm.
    grad_norm_ema: T,
    /// Decay used when updating [`Self::grad_norm_ema`]; in `(0, 1)`.
    norm_ema_decay: T,
}

impl<T> QuantumLearnedOptimizer<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync + 'static,
{
    /// Default gradient-norm EMA decay coefficient, materialised in `T`.
    fn default_norm_ema_decay() -> T {
        T::from(DEFAULT_NORM_EMA_DECAY).unwrap_or_else(|| {
            // Fall back to a value that is unambiguously in (0, 1) if the
            // literal cannot be represented exactly in `T`.
            let nine = T::from(9.0).unwrap_or_else(T::one);
            let ten = T::from(10.0).unwrap_or_else(|| T::one() + T::one());
            nine / ten
        })
    }

    /// Construct a wrapper around an arbitrary [`QuantumBackend`].
    fn from_backend(backend: QuantumBackend<T>) -> Self {
        Self {
            backend,
            step_count: 0,
            grad_norm_ema: T::zero(),
            norm_ema_decay: Self::default_norm_ema_decay(),
        }
    }

    /// Create a quantum-annealing-backed learned optimizer.
    ///
    /// The `learning_rate` scales the perturbation magnitude of the underlying
    /// [`QuantumAnnealing`] optimizer.
    pub fn annealing(learning_rate: T) -> Self {
        Self::from_backend(QuantumBackend::Annealing(QuantumAnnealing::new(
            learning_rate,
        )))
    }

    /// Create a variational-quantum (SPSA) backed learned optimizer.
    ///
    /// The `learning_rate` is the SPSA gain numerator `a` of the underlying
    /// [`VariationalQuantumOptimizer`].
    pub fn variational(learning_rate: T) -> Self {
        Self::from_backend(QuantumBackend::Variational(
            VariationalQuantumOptimizer::new(learning_rate),
        ))
    }

    /// Create a hybrid quantum-classical backed learned optimizer.
    ///
    /// The first `switch_step` updates are produced by quantum annealing; later
    /// updates are produced by the classical Adam refinement phase of the
    /// underlying [`HybridQuantumClassical`] optimizer.
    pub fn hybrid(learning_rate: T, switch_step: usize) -> Self {
        Self::from_backend(QuantumBackend::Hybrid(HybridQuantumClassical::new(
            learning_rate,
            switch_step,
        )))
    }

    /// Returns the decay coefficient used for the gradient-norm EMA.
    pub fn norm_ema_decay(&self) -> T {
        self.norm_ema_decay
    }

    /// Override the gradient-norm EMA decay coefficient.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] unless `decay` lies in the open
    /// interval `(0, 1)`; a decay outside that range would make the EMA either
    /// ignore new observations entirely or diverge.
    pub fn set_norm_ema_decay(&mut self, decay: T) -> Result<()> {
        if !(decay > T::zero() && decay < T::one()) {
            return Err(OptimError::InvalidConfig(format!(
                "norm_ema_decay must lie in (0, 1), got {decay:?}"
            )));
        }
        self.norm_ema_decay = decay;
        Ok(())
    }

    /// Immutable access to the underlying backend (useful for diagnostics).
    pub fn backend(&self) -> &QuantumBackend<T> {
        &self.backend
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

impl<T> AdvancedOptimizer<T> for QuantumLearnedOptimizer<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync + 'static,
{
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        // Delegate to the backend; it performs the dimension check and any
        // RNG-driven update. Only on success do we advance our bookkeeping so a
        // failed step leaves the wrapper in a consistent state.
        let updated = self.backend.step_array1(params, gradients)?;

        let grad_norm = l2_norm(gradients);
        self.grad_norm_ema =
            self.norm_ema_decay * self.grad_norm_ema + (T::one() - self.norm_ema_decay) * grad_norm;
        self.step_count = self.step_count.saturating_add(1);

        Ok(updated)
    }

    fn get_learning_rate(&self) -> T {
        self.backend.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.backend.set_learning_rate(lr);
    }

    fn name(&self) -> &str {
        self.backend.name()
    }

    fn get_state(&self) -> OptimizerStateInfo<T> {
        OptimizerStateInfo {
            step_count: self.step_count,
            current_lr: self.get_learning_rate(),
            grad_norm_ema: self.grad_norm_ema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Gradient of the convex quadratic `f(x) = ||x||^2`, i.e. `2 * x`.
    fn quadratic_grad(params: &Array1<f64>) -> Array1<f64> {
        params.mapv(|x| 2.0 * x)
    }

    fn all_finite(arr: &Array1<f64>) -> bool {
        arr.iter().all(|x| x.is_finite())
    }

    #[test]
    fn test_annealing_steps_preserve_shape_and_track_state() {
        let mut opt = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        let mut params = Array1::from_vec(vec![1.0, -2.0, 0.5, 3.0]);
        assert_eq!(opt.step_count(), 0);
        for i in 1..=10 {
            let grads = quadratic_grad(&params);
            params = opt.step(&params, &grads).expect("annealing step succeeds");
            assert_eq!(params.len(), 4, "shape must be preserved");
            assert!(all_finite(&params), "outputs must remain finite");
            assert_eq!(opt.step_count(), i, "step count must increment");
        }
        assert!(
            opt.get_state().grad_norm_ema > 0.0,
            "grad_norm_ema must be positive after nonzero-grad steps"
        );
    }

    #[test]
    fn test_variational_steps_preserve_shape_and_track_state() {
        let mut opt = QuantumLearnedOptimizer::<f64>::variational(0.1);
        let mut params = Array1::from_vec(vec![2.0, -1.0, 0.25]);
        for i in 1..=10 {
            let grads = quadratic_grad(&params);
            params = opt
                .step(&params, &grads)
                .expect("variational step succeeds");
            assert_eq!(params.len(), 3);
            assert!(all_finite(&params));
            assert_eq!(opt.step_count(), i);
        }
        assert!(opt.get_state().grad_norm_ema > 0.0);
    }

    #[test]
    fn test_hybrid_steps_preserve_shape_and_track_state() {
        let mut opt = QuantumLearnedOptimizer::<f64>::hybrid(0.05, 4);
        let mut params = Array1::from_vec(vec![1.5, -0.75]);
        for i in 1..=8 {
            let grads = quadratic_grad(&params);
            params = opt.step(&params, &grads).expect("hybrid step succeeds");
            assert_eq!(params.len(), 2);
            assert!(all_finite(&params));
            assert_eq!(opt.step_count(), i);
        }
        assert!(opt.get_state().grad_norm_ema > 0.0);
    }

    #[test]
    fn test_annealing_learning_rate_round_trip() {
        let mut opt = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        assert!((opt.get_learning_rate() - 0.05).abs() < 1e-12);
        opt.set_learning_rate(0.123);
        assert!((opt.get_learning_rate() - 0.123).abs() < 1e-12);
    }

    #[test]
    fn test_variational_learning_rate_round_trip() {
        let mut opt = QuantumLearnedOptimizer::<f64>::variational(0.2);
        assert!((opt.get_learning_rate() - 0.2).abs() < 1e-12);
        opt.set_learning_rate(0.01);
        assert!((opt.get_learning_rate() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_learning_rate_round_trip() {
        // In the exploration phase the hybrid reports the quantum learning
        // rate; setting the rate updates both sub-optimizers.
        let mut opt = QuantumLearnedOptimizer::<f64>::hybrid(0.05, 10);
        assert!((opt.get_learning_rate() - 0.05).abs() < 1e-12);
        opt.set_learning_rate(0.33);
        assert!((opt.get_learning_rate() - 0.33).abs() < 1e-12);
    }

    #[test]
    fn test_names_per_backend() {
        let annealing = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        let variational = QuantumLearnedOptimizer::<f64>::variational(0.05);
        let hybrid = QuantumLearnedOptimizer::<f64>::hybrid(0.05, 5);
        assert_eq!(annealing.name(), "QuantumAnnealingLearned");
        assert_eq!(variational.name(), "VariationalQuantumLearned");
        assert_eq!(hybrid.name(), "HybridQuantumLearned");
    }

    #[test]
    fn test_annealing_length_mismatch_errors() {
        let mut opt = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![1.0, 2.0]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err(), "mismatched lengths must error");
        // A failed step must not advance bookkeeping.
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_variational_length_mismatch_errors() {
        let mut opt = QuantumLearnedOptimizer::<f64>::variational(0.1);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err());
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_hybrid_length_mismatch_errors() {
        let mut opt = QuantumLearnedOptimizer::<f64>::hybrid(0.05, 5);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![1.0]);
        let result = opt.step(&params, &grads);
        assert!(result.is_err());
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_hybrid_crosses_switch_step_without_error() {
        // Run switch_step + 2 steps to cover the exploration -> refinement
        // transition (and the one-time warm-start handoff).
        let switch_step = 6usize;
        let mut opt = QuantumLearnedOptimizer::<f64>::hybrid(0.05, switch_step);
        let mut params = Array1::from_vec(vec![2.0, -1.0, 0.5]);
        for _ in 0..(switch_step + 2) {
            let grads = quadratic_grad(&params);
            params = opt
                .step(&params, &grads)
                .expect("hybrid step across switch succeeds");
            assert!(all_finite(&params));
        }
        assert_eq!(opt.step_count(), switch_step + 2);
    }

    #[test]
    fn test_get_state_reflects_learning_rate_and_steps() {
        let mut opt = QuantumLearnedOptimizer::<f64>::variational(0.07);
        let params = Array1::from_vec(vec![0.5, -0.5, 0.5]);
        let grads = quadratic_grad(&params);
        let _ = opt.step(&params, &grads).expect("step succeeds");
        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!((state.current_lr - 0.07).abs() < 1e-12);
        assert!(state.grad_norm_ema >= 0.0);
    }

    #[test]
    fn test_norm_ema_decay_validation() {
        let mut opt = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        // Default decay is in (0, 1).
        assert!(opt.norm_ema_decay() > 0.0 && opt.norm_ema_decay() < 1.0);
        assert!(opt.set_norm_ema_decay(0.5).is_ok());
        assert!((opt.norm_ema_decay() - 0.5).abs() < 1e-12);
        // Out-of-range decays are rejected.
        assert!(opt.set_norm_ema_decay(0.0).is_err());
        assert!(opt.set_norm_ema_decay(1.0).is_err());
        assert!(opt.set_norm_ema_decay(-0.1).is_err());
        assert!(opt.set_norm_ema_decay(1.5).is_err());
    }

    #[test]
    fn test_zero_gradient_keeps_norm_ema_zero() {
        // With identically-zero gradients the EMA must remain exactly zero.
        let mut opt = QuantumLearnedOptimizer::<f64>::hybrid(0.05, 3);
        let params = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let grads = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        for _ in 0..5 {
            let _ = opt.step(&params, &grads).expect("zero-grad step succeeds");
        }
        assert!(opt.get_state().grad_norm_ema.abs() < 1e-15);
    }

    #[test]
    fn test_backend_accessor_matches_name() {
        let opt = QuantumLearnedOptimizer::<f64>::annealing(0.05);
        match opt.backend() {
            QuantumBackend::Annealing(_) => {}
            _ => panic!("expected annealing backend"),
        }
    }

    /// Compile-time assertion that the wrapper is `Send + Sync`.
    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<QuantumLearnedOptimizer<f64>>();
    }
}
