// OptiRS - AdaDelta Optimizer
// Adaptive learning rate method without manual learning rate tuning
// Reference: "ADADELTA: An Adaptive Learning Rate Method" by Matthew D. Zeiler (2012)
//
// Algorithm:
//   Accumulate gradients: E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²
//   Compute update: Δθ_t = -RMS[Δθ]_{t-1}/RMS[g]_t * g_t
//   Accumulate updates: E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1 - ρ) * Δθ_t²
//   Apply update: θ_{t+1} = θ_t + Δθ_t

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Ix1, ScalarOperand};
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, Zero};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// AdaDelta optimizer configuration
///
/// AdaDelta adapts learning rates based on a moving window of gradient updates,
/// instead of accumulating all past gradients. This eliminates the need for a
/// manual learning rate parameter.
///
/// # Key Features
/// - No learning rate parameter required (uses adaptive rates)
/// - Uses exponentially decaying average of squared gradients
/// - Uses exponentially decaying average of squared parameter updates
/// - More robust to hyperparameter choice than AdaGrad
///
/// # Type Parameters
/// - `T`: Floating-point type (f32 or f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaDelta<T: Float> {
    /// Decay rate for moving averages (typically 0.9 or 0.95)
    /// Controls the window size for gradient history
    rho: T,

    /// Small constant for numerical stability (typically 1e-6 to 1e-8)
    /// Prevents division by zero
    epsilon: T,

    /// Exponentially decaying average of squared gradients E[g²]
    /// Tracks the magnitude of recent gradients
    accumulated_gradients: Option<Array1<T>>,

    /// Exponentially decaying average of squared parameter updates E[Δθ²]
    /// Tracks the magnitude of recent parameter updates
    accumulated_updates: Option<Array1<T>>,

    /// Number of optimization steps performed
    step_count: usize,

    /// Optional multiplier applied to the first `warmup_steps` updates
    ///
    /// Defaults to `1` (disabled), i.e. plain AdaDelta as published by Zeiler (2012).
    /// When enabled the boost only scales the *applied* update; the accumulator
    /// `E[Δθ²]` is still fed the unboosted update, so the adaptive rate is not
    /// contaminated by the bootstrap factor.
    warmup_boost: T,

    /// Number of initial steps over which `warmup_boost` is applied
    warmup_steps: usize,
}

impl<T: Float> Default for AdaDelta<T> {
    fn default() -> Self {
        Self::new(
            T::from(0.95).expect("unwrap failed"), // rho
            T::from(1e-6).expect("unwrap failed"), // epsilon
        )
        .expect("unwrap failed")
    }
}

impl<T: Float> AdaDelta<T> {
    /// Create a new AdaDelta optimizer
    ///
    /// # Arguments
    /// - `rho`: Decay rate for moving averages (typically 0.9-0.99)
    /// - `epsilon`: Small constant for numerical stability (typically 1e-6 to 1e-8)
    ///
    /// # Returns
    /// Result containing the optimizer or validation error
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::AdaDelta;
    ///
    /// let optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
    /// ```
    pub fn new(rho: T, epsilon: T) -> Result<Self> {
        let rho_f64 = rho.to_f64().expect("unwrap failed");
        let epsilon_f64 = epsilon.to_f64().expect("unwrap failed");

        if rho_f64 <= 0.0 || rho_f64 >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "rho must be in (0, 1), got {}",
                rho_f64
            )));
        }

        if epsilon_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "epsilon must be positive, got {}",
                epsilon_f64
            )));
        }

        Ok(Self {
            rho,
            epsilon,
            accumulated_gradients: None,
            accumulated_updates: None,
            step_count: 0,
            warmup_boost: T::one(),
            warmup_steps: 0,
        })
    }

    /// Enable an opt-in bootstrap multiplier for the first `steps` updates
    ///
    /// Plain AdaDelta starts with `E[Δθ²] = 0`, so the first updates are on the order
    /// of `sqrt(epsilon)` and progress is slow until the update accumulator warms up.
    /// Setting a boost trades strict fidelity to the paper for a faster start.
    ///
    /// The boost scales only the update that is *applied* to the parameters — the
    /// value accumulated into `E[Δθ²]` remains the unboosted update, so the adaptive
    /// learning rate stays a faithful estimate.
    ///
    /// # Errors
    /// Returns an error if `boost` is not strictly positive.
    pub fn with_warmup_boost(mut self, boost: T, steps: usize) -> Result<Self> {
        let boost_f64 = boost.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter("boost is not representable".to_string())
        })?;
        if boost_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "warmup boost must be positive, got {}",
                boost_f64
            )));
        }
        self.warmup_boost = boost;
        self.warmup_steps = steps;
        Ok(self)
    }

    /// Returns the configured warmup boost multiplier (1 when disabled)
    pub fn warmup_boost(&self) -> T {
        self.warmup_boost
    }

    /// Perform a single optimization step
    ///
    /// # Arguments
    /// - `params`: Current parameter values
    /// - `grads`: Gradient values
    ///
    /// # Returns
    /// Result containing updated parameters or error
    ///
    /// # Algorithm
    /// 1. Initialize accumulators on first step
    /// 2. Update exponentially decaying average of squared gradients
    /// 3. Compute RMS of gradients and previous updates
    /// 4. Compute parameter update using adaptive learning rate
    /// 5. Update exponentially decaying average of squared updates
    /// 6. Apply parameter update
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::AdaDelta;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let mut optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
    /// let params = array![1.0, 2.0, 3.0];
    /// let grads = array![0.1, 0.2, 0.3];
    ///
    /// let updated_params = optimizer.step(params.view(), grads.view()).expect("unwrap failed");
    /// ```
    pub fn step<'a, P, G>(&mut self, params: P, grads: G) -> Result<Array1<T>>
    where
        P: Into<ArrayView1<'a, T>>,
        G: Into<ArrayView1<'a, T>>,
        T: 'a,
    {
        self.step_view(params.into(), grads.into())
    }

    /// Perform a single optimization step on borrowed views
    ///
    /// This is the concrete implementation behind [`AdaDelta::step`].
    pub fn step_view(&mut self, params: ArrayView1<T>, grads: ArrayView1<T>) -> Result<Array1<T>> {
        let n = params.len();

        if grads.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected gradient size {}, got {}",
                n,
                grads.len()
            )));
        }

        // Initialize accumulators on first step
        if self.accumulated_gradients.is_none() {
            self.accumulated_gradients = Some(Array1::zeros(n));
            self.accumulated_updates = Some(Array1::zeros(n));
        }

        let acc_grad = self.accumulated_gradients.as_mut().expect("unwrap failed");
        let acc_update = self.accumulated_updates.as_mut().expect("unwrap failed");

        // Update exponentially decaying average of squared gradients
        // E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²
        let one = T::one();
        let one_minus_rho = one - self.rho;

        for i in 0..n {
            let grad = grads[i];
            acc_grad[i] = self.rho * acc_grad[i] + one_minus_rho * grad * grad;
        }

        // Compute RMS[g]_t = sqrt(E[g²]_t + ε)
        // Compute RMS[Δθ]_{t-1} = sqrt(E[Δθ²]_{t-1} + ε)
        // Compute update: Δθ_t = -RMS[Δθ]_{t-1}/RMS[g]_t * g_t
        let mut delta_params = Array1::zeros(n);

        for i in 0..n {
            let rms_grad = (acc_grad[i] + self.epsilon).sqrt();
            let rms_update = (acc_update[i] + self.epsilon).sqrt();

            // Adaptive learning rate: RMS[Δθ]_{t-1}/RMS[g]_t
            delta_params[i] = -(rms_update / rms_grad) * grads[i];
        }

        // Update exponentially decaying average of squared parameter updates using the
        // *unboosted* update, so an opt-in bootstrap multiplier cannot contaminate the
        // adaptive learning rate estimate.
        // E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1 - ρ) * Δθ_t²
        for i in 0..n {
            let delta = delta_params[i];
            acc_update[i] = self.rho * acc_update[i] + one_minus_rho * delta * delta;
        }

        // Opt-in bootstrap multiplier for the first `warmup_steps` updates
        let boost = if self.step_count < self.warmup_steps {
            self.warmup_boost
        } else {
            T::one()
        };

        // Apply update: θ_{t+1} = θ_t + Δθ_t
        let mut updated_params = params.to_owned();
        for i in 0..n {
            updated_params[i] = updated_params[i] + delta_params[i] * boost;
        }

        self.step_count += 1;

        Ok(updated_params)
    }

    /// Get the number of optimization steps performed
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset the optimizer state
    ///
    /// Clears accumulated gradient and update history
    pub fn reset(&mut self) {
        self.accumulated_gradients = None;
        self.accumulated_updates = None;
        self.step_count = 0;
    }

    /// Get the current RMS of gradients for each parameter
    ///
    /// Returns None if no steps have been performed yet
    pub fn rms_gradients(&self) -> Option<Array1<T>> {
        self.accumulated_gradients
            .as_ref()
            .map(|acc_grad| acc_grad.mapv(|x| (x + self.epsilon).sqrt()))
    }

    /// Get the current RMS of parameter updates
    ///
    /// Returns None if no steps have been performed yet
    pub fn rms_updates(&self) -> Option<Array1<T>> {
        self.accumulated_updates
            .as_ref()
            .map(|acc_update| acc_update.mapv(|x| (x + self.epsilon).sqrt()))
    }
}

impl<T> Optimizer<T, Ix1> for AdaDelta<T>
where
    T: Float + ScalarOperand + Debug + Send + Sync,
{
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        self.step_view(params.view(), gradients.view())
    }

    /// AdaDelta has no learning-rate hyperparameter; the effective per-parameter rate
    /// is `RMS[Δθ]/RMS[g]`. This reports `1` as the nominal scale.
    fn get_learning_rate(&self) -> T {
        T::one()
    }

    /// AdaDelta derives its step size from its accumulators, so setting a learning
    /// rate has no effect. The method exists to satisfy the [`Optimizer`] trait.
    fn set_learning_rate(&mut self, _learning_rate: T) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_adadelta_creation() {
        let optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adadelta_invalid_rho() {
        assert!(AdaDelta::<f32>::new(1.5, 1e-6).is_err());
        assert!(AdaDelta::<f32>::new(-0.1, 1e-6).is_err());
    }

    #[test]
    fn test_adadelta_invalid_epsilon() {
        assert!(AdaDelta::<f32>::new(0.95, -1e-6).is_err());
    }

    #[test]
    fn test_adadelta_single_step() {
        let mut optimizer = AdaDelta::<f32>::new(0.9, 1e-6).expect("unwrap failed");
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer
            .step(params.view(), grads.view())
            .expect("unwrap failed");

        // First step should have small updates (RMS[Δθ]_{-1} = 0)
        assert!(updated_params.len() == 3);
        assert_eq!(optimizer.step_count(), 1);

        // Parameters should change (even if slightly on first step)
        for i in 0..3 {
            assert_ne!(updated_params[i], params[i]);
        }
    }

    #[test]
    fn test_adadelta_multiple_steps() {
        let mut optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
        let mut params = array![1.0, 2.0, 3.0];

        for _ in 0..10 {
            let grads = array![0.1, 0.2, 0.3];
            params = optimizer
                .step(params.view(), grads.view())
                .expect("unwrap failed");
        }

        assert_eq!(optimizer.step_count(), 10);

        // After multiple steps, parameters should have changed significantly
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_adadelta_shape_mismatch() {
        let mut optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2]; // Wrong shape

        assert!(optimizer.step(params.view(), grads.view()).is_err());
    }

    #[test]
    fn test_adadelta_reset() {
        let mut optimizer = AdaDelta::<f32>::new(0.95, 1e-6).expect("unwrap failed");
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        optimizer
            .step(params.view(), grads.view())
            .expect("unwrap failed");
        assert_eq!(optimizer.step_count(), 1);
        assert!(optimizer.accumulated_gradients.is_some());

        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
        assert!(optimizer.accumulated_gradients.is_none());
        assert!(optimizer.accumulated_updates.is_none());
    }

    #[test]
    fn test_adadelta_convergence() {
        // Test convergence on a simple quadratic function: f(x) = x²
        // Gradient: f'(x) = 2x
        // Using higher rho (0.99) for better long-term memory.
        //
        // Plain AdaDelta bootstraps from E[Δθ²] = 0, so the first updates are on the
        // order of sqrt(epsilon). It genuinely needs a few thousand steps on this toy
        // problem; that is the published algorithm, not a defect.
        let mut optimizer = AdaDelta::<f64>::new(0.99, 1e-6).expect("unwrap failed");
        let mut params = array![10.0]; // Start far from optimum

        for _ in 0..3000 {
            let grads = params.mapv(|x| 2.0 * x); // Gradient of x²
            params = optimizer
                .step(params.view(), grads.view())
                .expect("unwrap failed");
        }

        assert!(
            params[0].abs() < 0.5,
            "Failed to converge, got {}",
            params[0]
        );
    }

    /// Regression test: the first update must follow the published AdaDelta formula
    /// exactly. The implementation used to multiply the first ten updates by a
    /// hardcoded, undocumented factor of 10 and feed the boosted value back into the
    /// update accumulator.
    #[test]
    fn test_adadelta_first_step_matches_published_formula() {
        let rho = 0.95f64;
        let epsilon = 1e-6f64;
        let mut optimizer = AdaDelta::<f64>::new(rho, epsilon).expect("valid config");

        let params = array![1.0f64];
        let grads = array![0.5f64];

        let updated = optimizer
            .step(params.view(), grads.view())
            .expect("step failed");

        let acc_grad = (1.0 - rho) * 0.5 * 0.5;
        let expected_delta = -((0.0f64 + epsilon).sqrt() / (acc_grad + epsilon).sqrt()) * 0.5;

        assert_relative_eq!(updated[0], 1.0 + expected_delta, epsilon = 1e-12);
        assert_relative_eq!(optimizer.warmup_boost(), 1.0, epsilon = 1e-12);
    }

    /// The opt-in bootstrap multiplier must not leak into `E[Δθ²]`.
    #[test]
    fn test_adadelta_warmup_boost_is_opt_in_and_uncontaminating() {
        let rho = 0.95f64;
        let epsilon = 1e-6f64;

        let mut plain = AdaDelta::<f64>::new(rho, epsilon).expect("valid config");
        let mut boosted = AdaDelta::<f64>::new(rho, epsilon)
            .expect("valid config")
            .with_warmup_boost(10.0, 1)
            .expect("valid boost");

        let params = array![1.0f64];
        let grads = array![0.5f64];

        let plain_out = plain.step(params.view(), grads.view()).expect("plain step");
        let boosted_out = boosted
            .step(params.view(), grads.view())
            .expect("boosted step");

        let plain_delta = plain_out[0] - 1.0;
        let boosted_delta = boosted_out[0] - 1.0;

        // The applied update is scaled...
        assert_relative_eq!(boosted_delta, plain_delta * 10.0, epsilon = 1e-12);

        // ...but the accumulator is identical, i.e. uncontaminated.
        let plain_rms = plain.rms_updates().expect("rms after step");
        let boosted_rms = boosted.rms_updates().expect("rms after step");
        assert_relative_eq!(plain_rms[0], boosted_rms[0], epsilon = 1e-15);
    }

    /// AdaDelta must be usable through the generic `Optimizer` trait.
    #[test]
    fn test_adadelta_optimizer_trait() {
        use crate::optimizers::Optimizer as _;

        let mut optimizer = AdaDelta::<f64>::new(0.95, 1e-6).expect("valid config");
        let params = array![1.0f64, 2.0, 3.0];
        let grads = array![0.1f64, 0.2, 0.3];

        let updated =
            Optimizer::<f64, scirs2_core::ndarray::Ix1>::step(&mut optimizer, &params, &grads)
                .expect("trait step failed");
        assert_eq!(updated.len(), 3);

        // The generic `step` also accepts plain references.
        let again = optimizer.step(&params, &grads).expect("ref step failed");
        assert_eq!(again.len(), 3);
    }

    #[test]
    fn test_adadelta_rms_values() {
        let mut optimizer = AdaDelta::<f32>::new(0.9, 1e-6).expect("unwrap failed");

        // No RMS values before first step
        assert!(optimizer.rms_gradients().is_none());
        assert!(optimizer.rms_updates().is_none());

        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        optimizer
            .step(params.view(), grads.view())
            .expect("unwrap failed");

        // RMS values should exist after first step
        assert!(optimizer.rms_gradients().is_some());
        assert!(optimizer.rms_updates().is_some());

        let rms_grads = optimizer.rms_gradients().expect("unwrap failed");
        assert_eq!(rms_grads.len(), 3);
    }

    #[test]
    fn test_adadelta_f64() {
        let mut optimizer = AdaDelta::<f64>::new(0.95, 1e-8).expect("unwrap failed");
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer
            .step(params.view(), grads.view())
            .expect("unwrap failed");
        assert_eq!(updated_params.len(), 3);
    }
}
