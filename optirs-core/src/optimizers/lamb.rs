// LAMB optimizer implementation
//
// Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
// by You et al. (2019).

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// LAMB (Layer-wise Adaptive Moments) optimizer
///
/// LAMB is designed for large batch optimization. It extends AdamW with layer-wise
/// adaptive learning rates, making it particularly effective for training large models
/// with high batch sizes.
///
/// Formula:
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
/// r1 = ||theta_t||
/// g' = m_hat_t / (sqrt(v_hat_t) + epsilon) + lambda * theta_t
/// r2 = ||g'||
/// ratio = r1/r2 if r1 > 0 and r2 > 0, else 1.0
/// theta_t = theta_{t-1} - lr * ratio * g'
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{LAMB, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create a LAMB optimizer with default hyperparameters
/// let mut optimizer = LAMB::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct LAMB<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the first moment estimates
    beta1: A,
    /// Exponential decay rate for the second moment estimates
    beta2: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Whether to use bias correction
    bias_correction: bool,
    /// First moment vectors, one slot per parameter-tensor index
    m: Option<Vec<Array<A, IxDyn>>>,
    /// Second moment vectors, one slot per parameter-tensor index
    v: Option<Vec<Array<A, IxDyn>>>,
    /// Per-parameter-index timestep counters
    t: Vec<usize>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> LAMB<A> {
    /// Creates a new LAMB optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).expect("LAMB: default beta1 (0.9) must fit in A"),
            beta2: A::from(0.999).expect("LAMB: default beta2 (0.999) must fit in A"),
            epsilon: A::from(1e-6).expect("LAMB: default epsilon (1e-6) must fit in A"),
            weight_decay: A::zero(),
            bias_correction: true,
            m: None,
            v: None,
            t: Vec::new(),
        }
    }

    /// Creates a new LAMB optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-6)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    /// * `bias_correction` - Whether to use bias correction (default: true)
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
        bias_correction: bool,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            bias_correction,
            m: None,
            v: None,
            t: Vec::new(),
        }
    }

    /// Sets the beta1 parameter
    pub fn set_beta1(&mut self, beta1: A) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Gets the beta1 parameter
    pub fn get_beta1(&self) -> A {
        self.beta1
    }

    /// Sets the beta2 parameter
    pub fn set_beta2(&mut self, beta2: A) -> &mut Self {
        self.beta2 = beta2;
        self
    }

    /// Gets the beta2 parameter
    pub fn get_beta2(&self) -> A {
        self.beta2
    }

    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: A) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> A {
        self.epsilon
    }

    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the learning rate
    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t.clear();
    }

    /// Returns the timestep recorded for the parameter tensor at `index`
    ///
    /// Returns `0` when the index has never been stepped.
    pub fn timestep(&self, index: usize) -> usize {
        self.t.get(index).copied().unwrap_or(0)
    }

    /// Ensures state slots exist for `index` and match `dim`, then advances its timestep
    fn advance_state(&mut self, index: usize, dim: &IxDyn) -> Result<usize> {
        let m = self.m.get_or_insert_with(Vec::new);
        let v = self.v.get_or_insert_with(Vec::new);
        while m.len() <= index {
            m.push(Array::zeros(dim.clone()));
        }
        while v.len() <= index {
            v.push(Array::zeros(dim.clone()));
        }
        while self.t.len() <= index {
            self.t.push(0);
        }

        // Reset the slot when the parameter shape for this index changed
        if m[index].raw_dim() != *dim || v[index].raw_dim() != *dim {
            m[index] = Array::zeros(dim.clone());
            v[index] = Array::zeros(dim.clone());
            self.t[index] = 0;
        }

        let next = self.t[index].checked_add(1).ok_or_else(|| {
            OptimError::InvalidConfig(
                "Timestep counter overflow - too many optimization steps".to_string(),
            )
        })?;
        self.t[index] = next;
        Ok(next)
    }

    /// Performs a LAMB update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent moment/timestep slot, so several parameter
    /// tensors can be optimized by a single `LAMB` instance without interference.
    /// LAMB's trust ratio is computed per tensor, which is exactly the layer-wise
    /// adaptation the algorithm prescribes.
    pub fn step_indexed<D: Dimension>(
        &mut self,
        index: usize,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "Incompatible shapes: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let dim = params.raw_dim().into_dyn();
        let t = self.advance_state(index, &dim)?;
        let exp = i32::try_from(t).map_err(|_| {
            OptimError::InvalidConfig(
                "Timestep too large for bias correction calculation".to_string(),
            )
        })?;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.epsilon;
        let weight_decay = self.weight_decay;
        let use_weight_decay = weight_decay > A::zero();
        let one = A::one();
        let (bias_correction1, bias_correction2) = if self.bias_correction {
            (one - beta1.powi(exp), one - beta2.powi(exp))
        } else {
            (one, one)
        };

        let m = self
            .m
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("LAMB state not initialized".to_string()))?;
        let v = self
            .v
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("LAMB state not initialized".to_string()))?;

        let params_view = params.view().into_dyn();
        let gradients_view = gradients.view().into_dyn();

        // Build the (weight-decayed) adaptive update direction, and accumulate the
        // two norms needed for the layer-wise trust ratio in the same traversal.
        let mut update: Array<A, IxDyn> = Array::zeros(dim.clone());
        let mut weight_norm_sq = A::zero();
        let mut update_norm_sq = A::zero();

        Zip::from(&mut update)
            .and(&params_view)
            .and(&gradients_view)
            .and(&mut m[index])
            .and(&mut v[index])
            .for_each(|u, &p, &g, m_i, v_i| {
                *m_i = *m_i * beta1 + g * (one - beta1);
                *v_i = *v_i * beta2 + g * g * (one - beta2);
                let m_hat = *m_i / bias_correction1;
                let v_hat = *v_i / bias_correction2;
                let mut direction = m_hat / (v_hat.sqrt() + eps);
                if use_weight_decay {
                    direction = direction + p * weight_decay;
                }
                *u = direction;
                weight_norm_sq = weight_norm_sq + p * p;
                update_norm_sq = update_norm_sq + direction * direction;
            });

        let weight_norm = weight_norm_sq.sqrt();
        let update_norm = update_norm_sq.sqrt();
        let trust_ratio = if weight_norm > A::zero() && update_norm > A::zero() {
            weight_norm / update_norm
        } else {
            one
        };

        let scale = self.learning_rate * trust_ratio;
        let mut updated = params.to_owned();
        let mut updated_view = updated.view_mut().into_dyn();
        Zip::from(&mut updated_view).and(&update).for_each(|p, &u| {
            *p = *p - u * scale;
        });
        drop(updated_view);

        Ok(updated)
    }
}

impl<A, D> Optimizer<A, D> for LAMB<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        self.step_indexed(0, params, gradients)
    }

    fn step_list(
        &mut self,
        params_list: &[&Array<A, D>],
        gradients_list: &[&Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        if params_list.len() != gradients_list.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params_list.len(),
                gradients_list.len()
            )));
        }

        let mut results = Vec::with_capacity(params_list.len());
        for (index, (params, grads)) in params_list.iter().zip(gradients_list.iter()).enumerate() {
            results.push(self.step_indexed(index, params, grads)?);
        }
        Ok(results)
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

    #[test]
    fn test_lamb_basic_creation() {
        let optimizer: LAMB<f64> = LAMB::new(0.001);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);
        assert_abs_diff_eq!(optimizer.get_beta1(), 0.9);
        assert_abs_diff_eq!(optimizer.get_beta2(), 0.999);
        assert_abs_diff_eq!(optimizer.get_epsilon(), 1e-6);
        assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.0);
        assert!(optimizer.bias_correction);
    }

    #[test]
    fn test_lamb_convergence() {
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);

        // Minimize a simple quadratic function: f(x) = x^2 + y^2
        let mut params = Array1::from_vec(vec![5.0, 3.0]);

        for _ in 0..50 {
            // Gradient of x^2 + y^2 is (2x, 2y)
            let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_lamb_convergence");
        }

        // Should converge towards (0, 0)
        assert!(params[0].abs() < 1.0);
        assert!(params[1].abs() < 1.0);
    }

    #[test]
    fn test_lamb_with_weight_decay() {
        let mut optimizer: LAMB<f64> = LAMB::new_with_config(
            0.1,   // learning_rate
            0.9,   // beta1
            0.999, // beta2
            1e-6,  // epsilon
            0.1,   // weight_decay
            true,  // bias_correction
        );

        // Start from (1.0, 1.0)
        let mut params = Array1::from_vec(vec![1.0, 1.0]);

        // Run optimization with small gradients
        for _ in 0..20 {
            let gradients = Array1::from_vec(vec![0.1, 0.1]);
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_lamb_with_weight_decay");
        }

        // With weight decay, parameters should decrease
        assert!(params[0] < 1.0);
        assert!(params[1] < 1.0);
    }

    #[test]
    fn test_lamb_reset() {
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);

        // Perform a step to initialize state
        let params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![0.5]);
        let _ = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lamb_reset");

        // State should exist
        assert!(optimizer.m.is_some());
        assert!(optimizer.v.is_some());
        assert_eq!(optimizer.timestep(0), 1);

        // Reset
        optimizer.reset();

        // State should be cleared
        assert!(optimizer.m.is_none());
        assert!(optimizer.v.is_none());
        assert_eq!(optimizer.timestep(0), 0);
    }

    #[test]
    fn test_lamb_trust_ratio() {
        // Test with normal gradient and parameters
        let mut optimizer: LAMB<f64> = LAMB::new(0.1);
        let params = Array1::from_vec(vec![2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.4, 0.6]);

        let new_params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lamb_trust_ratio");

        // Parameters should be updated
        assert_ne!(new_params[0], params[0]);
        assert_ne!(new_params[1], params[1]);

        // Check they moved in the right direction
        assert!(new_params[0] < params[0]); // gradient was positive
        assert!(new_params[1] < params[1]); // gradient was positive
    }
}
