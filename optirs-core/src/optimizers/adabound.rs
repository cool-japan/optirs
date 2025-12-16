// OptiRS - AdaBound Optimizer
// Adaptive Gradient Methods with Dynamic Bound of Learning Rate
// Reference: "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" (ICLR 2019)
//
// Algorithm:
//   AdaBound employs dynamic bounds on learning rates to achieve smooth transition
//   from adaptive methods to SGD. This prevents the generalization gap observed
//   in pure adaptive methods.
//
//   Lower bound: α_l(t) = α_final * (1 - 1/(γ*t + 1))
//   Upper bound: α_u(t) = α_final * (1 + 1/(γ*t))
//   Clipped learning rate: η_t(i) = Clip(α / √(v_t(i) + ε), α_l(t), α_u(t))

use crate::error::{OptimError, Result};
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, Zero};
use serde::{Deserialize, Serialize};

/// AdaBound optimizer configuration
///
/// AdaBound combines the benefits of adaptive learning rate methods (like Adam)
/// with the strong generalization of SGD by dynamically bounding the learning rates.
///
/// # Key Features
/// - Smooth transition from Adam to SGD during training
/// - Dynamic bounds prevent learning rates from becoming too large or too small
/// - Better generalization than pure Adam
/// - Maintains fast convergence of adaptive methods
///
/// # Type Parameters
/// - `T`: Floating-point type (f32 or f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBound<T: Float> {
    /// Initial learning rate (α)
    learning_rate: T,

    /// Final learning rate for SGD convergence
    /// Typically 0.1 * learning_rate
    final_lr: T,

    /// First moment decay rate (β₁) - typically 0.9
    beta1: T,

    /// Second moment decay rate (β₂) - typically 0.999
    beta2: T,

    /// Small constant for numerical stability (ε) - typically 1e-8
    epsilon: T,

    /// Convergence speed parameter (γ) - typically 1e-3
    /// Controls how fast bounds converge to final_lr
    gamma: T,

    /// Weight decay coefficient (L2 regularization)
    weight_decay: T,

    /// Whether to use AMSBound variant (max of v_t)
    amsbound: bool,

    /// First moment vector (m_t)
    momentum: Option<Array1<T>>,

    /// Second moment vector (v_t)
    velocity: Option<Array1<T>>,

    /// Max of second moment (v̂_t) - only for AMSBound
    max_velocity: Option<Array1<T>>,

    /// Number of optimization steps performed
    step_count: usize,
}

use scirs2_core::ndarray::ScalarOperand;

impl<T: Float + ScalarOperand> Default for AdaBound<T> {
    fn default() -> Self {
        Self::new(
            T::from(0.001).unwrap(), // learning_rate
            T::from(0.1).unwrap(),   // final_lr
            T::from(0.9).unwrap(),   // beta1
            T::from(0.999).unwrap(), // beta2
            T::from(1e-8).unwrap(),  // epsilon
            T::from(1e-3).unwrap(),  // gamma
            T::zero(),               // weight_decay
            false,                   // amsbound
        )
        .unwrap()
    }
}

impl<T: Float + ScalarOperand> AdaBound<T> {
    /// Create a new AdaBound optimizer
    ///
    /// # Arguments
    /// - `learning_rate`: Initial learning rate (typically 0.001)
    /// - `final_lr`: Final learning rate for SGD convergence (typically 0.1)
    /// - `beta1`: First moment decay rate (typically 0.9)
    /// - `beta2`: Second moment decay rate (typically 0.999)
    /// - `epsilon`: Small constant for numerical stability (typically 1e-8)
    /// - `gamma`: Convergence speed parameter (typically 1e-3)
    /// - `weight_decay`: L2 regularization coefficient (typically 0.0)
    /// - `amsbound`: Use AMSBound variant if true
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::AdaBound;
    ///
    /// let optimizer = AdaBound::<f32>::new(
    ///     0.001,  // learning_rate
    ///     0.1,    // final_lr
    ///     0.9,    // beta1
    ///     0.999,  // beta2
    ///     1e-8,   // epsilon
    ///     1e-3,   // gamma
    ///     0.0,    // weight_decay
    ///     false   // amsbound
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        learning_rate: T,
        final_lr: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        gamma: T,
        weight_decay: T,
        amsbound: bool,
    ) -> Result<Self> {
        let lr_f64 = learning_rate.to_f64().unwrap();
        let final_f64 = final_lr.to_f64().unwrap();
        let beta1_f64 = beta1.to_f64().unwrap();
        let beta2_f64 = beta2.to_f64().unwrap();
        let eps_f64 = epsilon.to_f64().unwrap();
        let gamma_f64 = gamma.to_f64().unwrap();
        let wd_f64 = weight_decay.to_f64().unwrap();

        if lr_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "learning_rate must be positive, got {}",
                lr_f64
            )));
        }
        if final_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "final_lr must be positive, got {}",
                final_f64
            )));
        }
        if beta1_f64 <= 0.0 || beta1_f64 >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "beta1 must be in (0, 1), got {}",
                beta1_f64
            )));
        }
        if beta2_f64 <= 0.0 || beta2_f64 >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "beta2 must be in (0, 1), got {}",
                beta2_f64
            )));
        }
        if eps_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "epsilon must be positive, got {}",
                eps_f64
            )));
        }
        if gamma_f64 <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "gamma must be positive, got {}",
                gamma_f64
            )));
        }
        if wd_f64 < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "weight_decay must be non-negative, got {}",
                wd_f64
            )));
        }

        Ok(Self {
            learning_rate,
            final_lr,
            beta1,
            beta2,
            epsilon,
            gamma,
            weight_decay,
            amsbound,
            momentum: None,
            velocity: None,
            max_velocity: None,
            step_count: 0,
        })
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
    /// 1. Initialize moments on first step
    /// 2. Apply weight decay if configured
    /// 3. Update biased first moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    /// 4. Update biased second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    /// 5. Compute bias-corrected moments
    /// 6. Compute dynamic bounds: [α_l(t), α_u(t)]
    /// 7. Compute clipped learning rate per parameter
    /// 8. Apply parameter update: θ_{t+1} = θ_t - η_t * m̂_t
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::AdaBound;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let mut optimizer = AdaBound::<f32>::default();
    /// let params = array![1.0, 2.0, 3.0];
    /// let grads = array![0.1, 0.2, 0.3];
    ///
    /// let updated_params = optimizer.step(params.view(), grads.view()).unwrap();
    /// ```
    pub fn step(&mut self, params: ArrayView1<T>, grads: ArrayView1<T>) -> Result<Array1<T>> {
        let n = params.len();

        if grads.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected gradient size {}, got {}",
                n,
                grads.len()
            )));
        }

        // Initialize moments on first step
        if self.momentum.is_none() {
            self.momentum = Some(Array1::zeros(n));
            self.velocity = Some(Array1::zeros(n));
            if self.amsbound {
                self.max_velocity = Some(Array1::zeros(n));
            }
        }

        self.step_count += 1;
        let t = T::from(self.step_count).unwrap();

        let momentum = self.momentum.as_mut().unwrap();
        let velocity = self.velocity.as_mut().unwrap();

        let one = T::one();
        let two = T::from(2).unwrap();

        // Apply weight decay if configured
        let effective_grads = if self.weight_decay > T::zero() {
            grads.to_owned() + &(params.to_owned() * self.weight_decay)
        } else {
            grads.to_owned()
        };

        // Update biased first moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        for i in 0..n {
            momentum[i] = self.beta1 * momentum[i] + (one - self.beta1) * effective_grads[i];
        }

        // Update biased second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        for i in 0..n {
            let grad_sq = effective_grads[i] * effective_grads[i];
            velocity[i] = self.beta2 * velocity[i] + (one - self.beta2) * grad_sq;
        }

        // For AMSBound: v̂_t = max(v̂_{t-1}, v_t)
        if self.amsbound {
            let max_vel = self.max_velocity.as_mut().unwrap();
            for i in 0..n {
                if velocity[i] > max_vel[i] {
                    max_vel[i] = velocity[i];
                }
            }
        }

        // Compute bias correction terms
        let bias_correction1 = one - self.beta1.powf(t);
        let bias_correction2 = one - self.beta2.powf(t);

        // Compute dynamic bounds
        // Lower bound: α_l(t) = α_final * (1 - 1/(γ*t + 1))
        let lower_bound = self.final_lr * (one - one / (self.gamma * t + one));

        // Upper bound: α_u(t) = α_final * (1 + 1/(γ*t))
        let upper_bound = self.final_lr * (one + one / (self.gamma * t));

        // Apply parameter updates with clipped learning rates
        let mut updated_params = params.to_owned();

        for i in 0..n {
            // Bias-corrected first moment
            let m_hat = momentum[i] / bias_correction1;

            // Bias-corrected second moment (or max for AMSBound)
            let v_hat = if self.amsbound {
                self.max_velocity.as_ref().unwrap()[i] / bias_correction2
            } else {
                velocity[i] / bias_correction2
            };

            // Compute adaptive learning rate: α / √(v_t + ε)
            let step_size = self.learning_rate / (v_hat.sqrt() + self.epsilon);

            // Clip learning rate to dynamic bounds
            let clipped_step_size = if step_size < lower_bound {
                lower_bound
            } else if step_size > upper_bound {
                upper_bound
            } else {
                step_size
            };

            // Apply update: θ_{t+1} = θ_t - η_clipped * m̂_t
            updated_params[i] = updated_params[i] - clipped_step_size * m_hat;
        }

        Ok(updated_params)
    }

    /// Get the number of optimization steps performed
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.momentum = None;
        self.velocity = None;
        self.max_velocity = None;
        self.step_count = 0;
    }

    /// Get current dynamic bounds [lower, upper]
    pub fn current_bounds(&self) -> (T, T) {
        if self.step_count == 0 {
            return (self.final_lr, self.final_lr);
        }

        let t = T::from(self.step_count).unwrap();
        let one = T::one();

        let lower_bound = self.final_lr * (one - one / (self.gamma * t + one));
        let upper_bound = self.final_lr * (one + one / (self.gamma * t));

        (lower_bound, upper_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_adabound_creation() {
        let optimizer = AdaBound::<f32>::default();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adabound_single_step() {
        let mut optimizer = AdaBound::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer.step(params.view(), grads.view()).unwrap();

        assert_eq!(updated_params.len(), 3);
        assert_eq!(optimizer.step_count(), 1);

        // Parameters should decrease (gradient descent)
        for i in 0..3 {
            assert!(updated_params[i] < params[i]);
        }
    }

    #[test]
    fn test_adabound_multiple_steps() {
        let mut optimizer = AdaBound::<f32>::default();
        let mut params = array![1.0, 2.0, 3.0];

        for _ in 0..10 {
            let grads = array![0.1, 0.2, 0.3];
            params = optimizer.step(params.view(), grads.view()).unwrap();
        }

        assert_eq!(optimizer.step_count(), 10);
    }

    #[test]
    fn test_adabound_dynamic_bounds() {
        let mut optimizer = AdaBound::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        // Before any steps, bounds should be equal to final_lr
        let (lower0, upper0) = optimizer.current_bounds();
        assert_relative_eq!(lower0, 0.1, epsilon = 1e-6);
        assert_relative_eq!(upper0, 0.1, epsilon = 1e-6);

        // After first step, bounds should widen
        optimizer.step(params.view(), grads.view()).unwrap();
        let (lower1, upper1) = optimizer.current_bounds();
        assert!(lower1 < upper1);
        assert!(lower1 >= 0.0);

        // After many steps, bounds should converge to final_lr
        for _ in 0..10000 {
            // Need many more steps for bound convergence
            optimizer.step(params.view(), grads.view()).unwrap();
        }
        let (lower_final, upper_final) = optimizer.current_bounds();
        assert_relative_eq!(lower_final, 0.1, epsilon = 0.01);
        assert_relative_eq!(upper_final, 0.1, epsilon = 0.01);
    }

    #[test]
    fn test_amsbound() {
        let mut optimizer =
            AdaBound::<f32>::new(0.001, 0.1, 0.9, 0.999, 1e-8, 1e-3, 0.0, true).unwrap();

        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer.step(params.view(), grads.view()).unwrap();
        assert_eq!(updated_params.len(), 3);
        assert!(optimizer.max_velocity.is_some());
    }

    #[test]
    fn test_adabound_weight_decay() {
        let mut optimizer =
            AdaBound::<f32>::new(0.001, 0.1, 0.9, 0.999, 1e-8, 1e-3, 0.01, false).unwrap();

        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer.step(params.view(), grads.view()).unwrap();

        // With weight decay, updates should be larger
        for i in 0..3 {
            assert!(updated_params[i] < params[i]);
        }
    }

    #[test]
    fn test_adabound_convergence() {
        // Test convergence on quadratic function f(x) = x²
        let mut optimizer = AdaBound::<f64>::default();
        let mut params = array![5.0];

        for _ in 0..500 {
            // AdaBound needs more iterations for tight convergence
            let grads = params.mapv(|x| 2.0 * x);
            params = optimizer.step(params.view(), grads.view()).unwrap();
        }

        // Should converge close to zero
        assert!(
            params[0].abs() < 0.1,
            "Failed to converge, got {}",
            params[0]
        );
    }

    #[test]
    fn test_adabound_reset() {
        let mut optimizer = AdaBound::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        optimizer.step(params.view(), grads.view()).unwrap();
        assert_eq!(optimizer.step_count(), 1);

        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
        assert!(optimizer.momentum.is_none());
        assert!(optimizer.velocity.is_none());
    }
}
