// Adagrad optimizer implementation

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand};
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Adagrad optimizer
///
/// Implements the Adagrad optimization algorithm from the paper:
/// "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al. (2011)
///
/// Adagrad adapts the learning rate to the parameters, performing larger updates for
/// infrequently updated parameters and smaller updates for frequently updated parameters.
///
/// Formula:
/// G_t = G_{t-1} + g_t^2
/// param_t = param_{t-1} - learning_rate * g_t / (sqrt(G_t) + epsilon)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::optimizers::{Adagrad, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an Adagrad optimizer with learning rate 0.01
/// let mut optimizer = Adagrad::new(0.01);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Adagrad<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Sum of squared gradients
    sum_squared_grads: Option<Vec<Array<A, scirs2_core::ndarray_ext::IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> Adagrad<A> {
    /// Creates a new Adagrad optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            epsilon: A::from(1e-10).unwrap(),
            weight_decay: A::zero(),
            sum_squared_grads: None,
        }
    }

    /// Creates a new Adagrad optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `epsilon` - Small constant for numerical stability (default: 1e-10)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(learning_rate: A, epsilon: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay,
            sum_squared_grads: None,
        }
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

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.sum_squared_grads = None;
    }
}

impl<A, D> Optimizer<A, D> for Adagrad<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for storage in state vectors
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Apply weight decay to gradients if needed
        let adjusted_gradients = if self.weight_decay > A::zero() {
            &gradients_dyn + &(&params_dyn * self.weight_decay)
        } else {
            gradients_dyn.clone()
        };

        // Initialize state if this is the first step
        if self.sum_squared_grads.is_none() {
            self.sum_squared_grads = Some(vec![Array::zeros(params_dyn.raw_dim())]);
        }

        let sum_squared_grads = self.sum_squared_grads.as_mut().unwrap();

        // Ensure we have state for this parameter set
        if sum_squared_grads.is_empty() {
            sum_squared_grads.push(Array::zeros(params_dyn.raw_dim()));
        } else if sum_squared_grads[0].raw_dim() != params_dyn.raw_dim() {
            // If the parameter dimensions have changed, reset state
            sum_squared_grads[0] = Array::zeros(params_dyn.raw_dim());
        }

        // Update sum of squared gradients
        // G_t = G_{t-1} + g_t^2
        sum_squared_grads[0] = &sum_squared_grads[0] + &(&adjusted_gradients * &adjusted_gradients);

        // Compute step size
        // step = learning_rate * g_t / (sqrt(G_t) + epsilon)
        let g_sqrt = sum_squared_grads[0].mapv(|x| x.sqrt());
        let step = &adjusted_gradients * self.learning_rate / &(&g_sqrt + self.epsilon);

        // Update parameters
        let updated_params = &params_dyn - step;

        // Convert back to original dimension
        Ok(updated_params.into_dimensionality::<D>().unwrap())
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}
