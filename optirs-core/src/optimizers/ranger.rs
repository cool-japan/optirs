// OptiRS - Ranger Optimizer
// RAdam + Lookahead combination for improved convergence and stability
// Reference: "Ranger - a synergistic optimizer" by Less Wright (2019)
//
// Ranger combines:
// 1. RAdam (Rectified Adam) - Adaptive learning rate with variance rectification
// 2. Lookahead - Slow and fast weight updates for stability
//
// This combination provides:
// - Fast convergence from RAdam
// - Stability and reduced variance from Lookahead
// - Better generalization than either optimizer alone

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, Zero};
use serde::{Deserialize, Serialize};

/// Ranger optimizer configuration
///
/// Ranger combines RAdam (Rectified Adam) with Lookahead mechanism.
/// This standalone implementation integrates both algorithms efficiently.
///
/// # Key Features
/// - Fast convergence from RAdam's variance rectification
/// - Stability from Lookahead's slow weight trajectory
/// - Reduced sensitivity to hyperparameter choices
/// - Better generalization than Adam or RAdam alone
///
/// # Type Parameters
/// - `T`: Floating-point type (f32 or f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ranger<T: Float + ScalarOperand> {
    // RAdam parameters
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,

    // Lookahead parameters
    lookahead_k: usize,
    lookahead_alpha: T,

    // RAdam state
    momentum: Option<Array1<T>>,
    velocity: Option<Array1<T>>,

    // Lookahead state
    slow_weights: Option<Array1<T>>,

    // Step counters
    step_count: usize,
    slow_update_count: usize,
}

impl<T: Float + ScalarOperand> Default for Ranger<T> {
    fn default() -> Self {
        Self::new(
            T::from(0.001).unwrap(), // learning_rate
            T::from(0.9).unwrap(),   // beta1
            T::from(0.999).unwrap(), // beta2
            T::from(1e-8).unwrap(),  // epsilon
            T::zero(),               // weight_decay
            5,                       // lookahead_k
            T::from(0.5).unwrap(),   // lookahead_alpha
        )
        .unwrap()
    }
}

impl<T: Float + ScalarOperand> Ranger<T> {
    /// Create a new Ranger optimizer
    ///
    /// # Arguments
    /// - `learning_rate`: Learning rate for RAdam (typically 0.001)
    /// - `beta1`: First moment decay rate (typically 0.9)
    /// - `beta2`: Second moment decay rate (typically 0.999)
    /// - `epsilon`: Small constant for numerical stability (typically 1e-8)
    /// - `weight_decay`: L2 regularization coefficient (typically 0.0)
    /// - `lookahead_k`: Number of fast updates per slow update (typically 5-6)
    /// - `lookahead_alpha`: Interpolation factor for slow weights (typically 0.5)
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::Ranger;
    ///
    /// let optimizer = Ranger::<f32>::new(
    ///     0.001,  // learning_rate
    ///     0.9,    // beta1
    ///     0.999,  // beta2
    ///     1e-8,   // epsilon
    ///     0.0,    // weight_decay
    ///     5,      // lookahead_k
    ///     0.5     // lookahead_alpha
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        lookahead_k: usize,
        lookahead_alpha: T,
    ) -> Result<Self> {
        // Validate parameters
        if learning_rate.to_f64().unwrap() <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "learning_rate must be positive, got {}",
                learning_rate.to_f64().unwrap()
            )));
        }
        if beta1.to_f64().unwrap() <= 0.0 || beta1.to_f64().unwrap() >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "beta1 must be in (0, 1), got {}",
                beta1.to_f64().unwrap()
            )));
        }
        if beta2.to_f64().unwrap() <= 0.0 || beta2.to_f64().unwrap() >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "beta2 must be in (0, 1), got {}",
                beta2.to_f64().unwrap()
            )));
        }
        if epsilon.to_f64().unwrap() <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "epsilon must be positive, got {}",
                epsilon.to_f64().unwrap()
            )));
        }
        if weight_decay.to_f64().unwrap() < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "weight_decay must be non-negative, got {}",
                weight_decay.to_f64().unwrap()
            )));
        }
        if lookahead_k == 0 {
            return Err(OptimError::InvalidParameter(
                "lookahead_k must be positive".to_string(),
            ));
        }
        if lookahead_alpha.to_f64().unwrap() <= 0.0 || lookahead_alpha.to_f64().unwrap() > 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "lookahead_alpha must be in (0, 1], got {}",
                lookahead_alpha.to_f64().unwrap()
            )));
        }

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            lookahead_k,
            lookahead_alpha,
            momentum: None,
            velocity: None,
            slow_weights: None,
            step_count: 0,
            slow_update_count: 0,
        })
    }

    /// Perform a single optimization step
    ///
    /// Combines RAdam (fast weights) with Lookahead (slow weights)
    ///
    /// # Example
    /// ```
    /// use optirs_core::optimizers::Ranger;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let mut optimizer = Ranger::<f32>::default();
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

        // Initialize state on first step
        if self.momentum.is_none() {
            self.momentum = Some(Array1::zeros(n));
            self.velocity = Some(Array1::zeros(n));
            self.slow_weights = Some(params.to_owned());
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

        // RAdam: Update biased first moment
        for i in 0..n {
            momentum[i] = self.beta1 * momentum[i] + (one - self.beta1) * effective_grads[i];
        }

        // RAdam: Update biased second moment
        for i in 0..n {
            let grad_sq = effective_grads[i] * effective_grads[i];
            velocity[i] = self.beta2 * velocity[i] + (one - self.beta2) * grad_sq;
        }

        // RAdam: Compute bias correction
        let bias_correction1 = one - self.beta1.powf(t);
        let bias_correction2 = one - self.beta2.powf(t);

        // RAdam: Compute SMA (Simple Moving Average) length
        let rho_inf = two / (one - self.beta2) - one;
        let rho_t = rho_inf - two * t * self.beta2.powf(t) / bias_correction2;

        // RAdam: Apply variance rectification
        let mut updated_params = params.to_owned();

        if rho_t.to_f64().unwrap() > 4.0 {
            // Use adaptive learning rate with variance rectification
            let rect_term = ((rho_t - T::from(4).unwrap()) * (rho_t - two) * rho_inf
                / ((rho_inf - T::from(4).unwrap()) * (rho_inf - two) * rho_t))
                .sqrt();

            for i in 0..n {
                let m_hat = momentum[i] / bias_correction1;
                let v_hat = velocity[i] / bias_correction2;
                let step_size = self.learning_rate * rect_term / (v_hat.sqrt() + self.epsilon);
                updated_params[i] = updated_params[i] - step_size * m_hat;
            }
        } else {
            // Use simple momentum update during warmup
            for i in 0..n {
                let m_hat = momentum[i] / bias_correction1;
                updated_params[i] = updated_params[i] - self.learning_rate * m_hat;
            }
        }

        // Lookahead: Update slow weights every k steps
        if self.step_count.is_multiple_of(self.lookahead_k) {
            let slow = self.slow_weights.as_mut().unwrap();
            for i in 0..n {
                slow[i] = slow[i] + self.lookahead_alpha * (updated_params[i] - slow[i]);
            }
            self.slow_update_count += 1;

            // Synchronize fast weights with slow weights
            // This is the key to Lookahead: we return the slow weights after update
            Ok(slow.clone())
        } else {
            // Between slow updates, return fast weights
            Ok(updated_params)
        }
    }

    /// Get the number of optimization steps performed
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the number of slow weight updates performed
    pub fn slow_update_count(&self) -> usize {
        self.slow_update_count
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.momentum = None;
        self.velocity = None;
        self.slow_weights = None;
        self.step_count = 0;
        self.slow_update_count = 0;
    }

    /// Get the slow weights (Lookahead trajectory)
    pub fn slow_weights(&self) -> Option<&Array1<T>> {
        self.slow_weights.as_ref()
    }

    /// Check if variance rectification is active
    pub fn is_rectified(&self) -> bool {
        if self.step_count == 0 {
            return false;
        }
        let t = T::from(self.step_count).unwrap();
        let one = T::one();
        let two = T::from(2).unwrap();
        let bias_correction2 = one - self.beta2.powf(t);
        let rho_inf = two / (one - self.beta2) - one;
        let rho_t = rho_inf - two * t * self.beta2.powf(t) / bias_correction2;
        rho_t.to_f64().unwrap() > 4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_ranger_creation() {
        let optimizer = Ranger::<f32>::default();
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(optimizer.slow_update_count(), 0);
    }

    #[test]
    fn test_ranger_custom_creation() {
        let optimizer = Ranger::<f32>::new(0.002, 0.95, 0.9999, 1e-7, 0.01, 6, 0.6).unwrap();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_ranger_single_step() {
        let mut optimizer = Ranger::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let updated_params = optimizer.step(params.view(), grads.view()).unwrap();
        assert_eq!(updated_params.len(), 3);
        assert_eq!(optimizer.step_count(), 1);

        for i in 0..3 {
            assert!(updated_params[i] < params[i]);
        }
    }

    #[test]
    fn test_ranger_slow_updates() {
        let mut optimizer = Ranger::<f32>::new(0.001, 0.9, 0.999, 1e-8, 0.0, 3, 0.5).unwrap();
        let mut params = array![1.0, 2.0, 3.0];

        for _ in 0..3 {
            let grads = array![0.1, 0.2, 0.3];
            params = optimizer.step(params.view(), grads.view()).unwrap();
        }
        assert_eq!(optimizer.slow_update_count(), 1);
    }

    #[test]
    fn test_ranger_convergence() {
        // Use higher learning rate for this simple convex problem
        // Default 0.001 is tuned for neural networks
        let mut optimizer = Ranger::<f64>::new(
            0.1,   // learning_rate: higher for simple problem
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            0.0,   // weight_decay
            5,     // lookahead_k
            0.5,   // lookahead_alpha
        )
        .unwrap();
        let mut params = array![5.0];

        // Ranger combines RAdam (adaptive LR) with Lookahead (slow updates)
        for _ in 0..500 {
            let grads = params.mapv(|x| 2.0 * x);
            params = optimizer.step(params.view(), grads.view()).unwrap();
        }

        assert!(
            params[0].abs() < 0.1,
            "Failed to converge, got {}",
            params[0]
        );
    }

    #[test]
    fn test_ranger_reset() {
        let mut optimizer = Ranger::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        for _ in 0..10 {
            optimizer.step(params.view(), grads.view()).unwrap();
        }

        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(optimizer.slow_update_count(), 0);
        assert!(optimizer.slow_weights().is_none());
    }

    #[test]
    fn test_ranger_rectification() {
        let mut optimizer = Ranger::<f32>::default();
        let params = array![1.0];
        let grads = array![0.1];

        // Initially not rectified
        assert!(!optimizer.is_rectified());

        // After several steps, should be rectified
        for _ in 0..10 {
            optimizer.step(params.view(), grads.view()).unwrap();
        }
        assert!(optimizer.is_rectified());
    }
}
