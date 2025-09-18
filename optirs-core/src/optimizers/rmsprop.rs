// RMSprop optimizer implementation

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand};
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// RMSprop optimizer
///
/// Implements the RMSprop optimization algorithm as proposed by Geoffrey Hinton
/// in his Coursera course "Neural Networks for Machine Learning".
///
/// Formula:
/// v_t = rho * v_{t-1} + (1 - rho) * g_t^2
/// param_t = param_{t-1} - learning_rate * g_t / (sqrt(v_t) + epsilon)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use scirs2_optim::optimizers::{RMSprop, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an RMSprop optimizer with learning rate 0.001
/// let mut optimizer = RMSprop::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RMSprop<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Decay rate for the moving average of squared gradients
    rho: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Moving average of squared gradients
    v: Option<Vec<Array<A, scirs2_core::ndarray_ext::IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> RMSprop<A> {
    /// Creates a new RMSprop optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            rho: A::from(0.9).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: A::zero(),
            v: None,
        }
    }

    /// Creates a new RMSprop optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `rho` - Decay rate for the moving average of squared gradients (default: 0.9)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(learning_rate: A, rho: A, epsilon: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon,
            weight_decay,
            v: None,
        }
    }

    /// Sets the rho parameter
    pub fn set_rho(&mut self, rho: A) -> &mut Self {
        self.rho = rho;
        self
    }

    /// Gets the rho parameter
    pub fn get_rho(&self) -> A {
        self.rho
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
        self.v = None;
    }
}

impl<A, D> Optimizer<A, D> for RMSprop<A>
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
            gradients_dyn
        };

        // Initialize state if this is the first step
        if self.v.is_none() {
            self.v = Some(vec![Array::zeros(params_dyn.raw_dim())]);
        }

        let v = self.v.as_mut().unwrap();

        // Ensure we have state for this parameter set
        if v.is_empty() {
            v.push(Array::zeros(params_dyn.raw_dim()));
        } else if v[0].raw_dim() != params_dyn.raw_dim() {
            // If the parameter dimensions have changed, reset state
            v[0] = Array::zeros(params_dyn.raw_dim());
        }

        // Update moving average of squared gradients
        // v_t = rho * v_{t-1} + (1 - rho) * g_t^2
        v[0] =
            &v[0] * self.rho + &(&adjusted_gradients * &adjusted_gradients * (A::one() - self.rho));

        // Compute step size
        // step = learning_rate * g_t / (sqrt(v_t) + epsilon)
        let v_sqrt = v[0].mapv(|x| x.sqrt());
        let step = &adjusted_gradients * self.learning_rate / &(&v_sqrt + self.epsilon);

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
