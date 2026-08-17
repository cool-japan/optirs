// Lion optimizer implementation
//
// Based on the paper "Symbolic Discovery of Optimization Algorithms"
// by Chen et al. (2023).

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// Lion optimizer
///
/// Implements the Lion (Evolved Sign Momentum) optimization algorithm.
/// Lion is a memory-efficient optimizer that achieves strong performance
/// with only momentum state and uses the sign of the momentum for updates.
///
/// Formula:
/// u_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// theta_t = theta_{t-1} - alpha * (sign(u_t) + lambda * theta_{t-1})
/// m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{Lion, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create a Lion optimizer with default hyperparameters
/// let mut optimizer = Lion::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct Lion<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the momentum
    beta1: A,
    /// Exponential decay rate for the momentum update
    beta2: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Momentum vectors, one slot per parameter-tensor index
    m: Option<Vec<Array<A, IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> Lion<A> {
    /// Creates a new Lion optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).expect("Lion: default beta1 (0.9) must fit in A"),
            beta2: A::from(0.99).expect("Lion: default beta2 (0.99) must fit in A"),
            weight_decay: A::zero(),
            m: None,
        }
    }

    /// Creates a new Lion optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for computing the interpolated update (default: 0.9)
    /// * `beta2` - Exponential decay rate for updating the momentum (default: 0.99)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(learning_rate: A, beta1: A, beta2: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            m: None,
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
    }

    /// Ensures a momentum slot exists for `index` and matches `dim`
    fn ensure_state(&mut self, index: usize, dim: &IxDyn) {
        let m = self.m.get_or_insert_with(Vec::new);
        while m.len() <= index {
            m.push(Array::zeros(dim.clone()));
        }
        if m[index].raw_dim() != *dim {
            m[index] = Array::zeros(dim.clone());
        }
    }

    /// Applies a Lion update in place for the parameter tensor at `index`
    pub fn step_inplace_indexed<D: Dimension>(
        &mut self,
        index: usize,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<()> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "Incompatible shapes: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let dim = params.raw_dim().into_dyn();
        self.ensure_state(index, &dim);

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let lr = self.learning_rate;
        let weight_decay = self.weight_decay;
        let use_weight_decay = weight_decay > A::zero();
        let one = A::one();
        let zero = A::zero();
        let decay_factor = one - weight_decay * lr;

        let m = self
            .m
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("Lion state not initialized".to_string()))?;

        let mut params_view = params.view_mut().into_dyn();
        let gradients_view = gradients.view().into_dyn();

        Zip::from(&mut params_view)
            .and(&gradients_view)
            .and(&mut m[index])
            .for_each(|p, &g, m_i| {
                // Step 1: interpolated update using beta1
                let interpolated = *m_i * beta1 + g * (one - beta1);

                // Step 2: sign of the interpolated update
                let sign_update = if interpolated > zero {
                    one
                } else if interpolated < zero {
                    -one
                } else {
                    zero
                };

                // Step 3: decoupled weight decay, then the sign step
                let decayed = if use_weight_decay {
                    *p * decay_factor
                } else {
                    *p
                };
                *p = decayed - sign_update * lr;

                // Step 4: momentum update using beta2
                *m_i = *m_i * beta2 + g * (one - beta2);
            });

        Ok(())
    }

    /// Applies a Lion update in place using the state slot of the first parameter tensor
    pub fn step_inplace<D: Dimension>(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<()> {
        self.step_inplace_indexed(0, params, gradients)
    }

    /// Performs a Lion update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent momentum slot.
    pub fn step_indexed<D: Dimension>(
        &mut self,
        index: usize,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>> {
        let mut updated = params.to_owned();
        self.step_inplace_indexed(index, &mut updated, gradients)?;
        Ok(updated)
    }
}

impl<A, D> Optimizer<A, D> for Lion<A>
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
    fn test_lion_basic_creation() {
        let optimizer: Lion<f64> = Lion::new(0.001);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);
        assert_abs_diff_eq!(optimizer.get_beta1(), 0.9);
        assert_abs_diff_eq!(optimizer.get_beta2(), 0.99);
        assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.0);
    }

    #[test]
    fn test_lion_convergence() {
        let mut optimizer: Lion<f64> = Lion::new(0.1); // Higher learning rate for testing

        // Minimize a simple quadratic function: f(x) = x^2
        let mut params = Array1::from_vec(vec![5.0]);

        // Lion converges linearly with sign updates
        for _ in 0..40 {
            // Fewer iterations with higher learning rate
            // Gradient of x^2 is 2x
            let gradients = Array1::from_vec(vec![2.0 * params[0]]);
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_lion_convergence");
        }

        // With learning rate 0.1 and 40 iterations, should reach close to 1.0
        assert!(params[0].abs() < 1.1);
    }

    #[test]
    fn test_lion_reset() {
        let mut optimizer: Lion<f64> = Lion::new(0.1);

        // Perform a step to initialize state
        let params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![0.1]);
        let _ = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lion_reset");

        // Reset optimizer
        optimizer.reset();

        // Next step should behave like the first
        let next_step = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lion_reset");

        // Create fresh optimizer for comparison
        let mut fresh_optimizer: Lion<f64> = Lion::new(0.1);
        let fresh_step = fresh_optimizer
            .step(&params, &gradients)
            .expect("step succeeds in test_lion_reset");

        assert_abs_diff_eq!(next_step[0], fresh_step[0], epsilon = 1e-10);
    }
}
