// Adagrad optimizer implementation

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
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
/// use scirs2_core::ndarray::Array1;
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
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct Adagrad<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Sum of squared gradients, one slot per parameter-tensor index
    sum_squared_grads: Option<Vec<Array<A, IxDyn>>>,
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
            epsilon: A::from(1e-10).expect("Adagrad: default epsilon (1e-10) must fit in A"),
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

    /// Ensures an accumulator slot exists for `index` and matches `dim`
    fn ensure_state(&mut self, index: usize, dim: &IxDyn) {
        let accumulators = self.sum_squared_grads.get_or_insert_with(Vec::new);
        while accumulators.len() <= index {
            accumulators.push(Array::zeros(dim.clone()));
        }
        if accumulators[index].raw_dim() != *dim {
            accumulators[index] = Array::zeros(dim.clone());
        }
    }

    /// Performs an Adagrad update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent accumulator, so several parameter tensors can
    /// be optimized by a single `Adagrad` instance without their histories mixing.
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
        self.ensure_state(index, &dim);

        let lr = self.learning_rate;
        let eps = self.epsilon;
        let weight_decay = self.weight_decay;
        let use_weight_decay = weight_decay > A::zero();

        let accumulators = self.sum_squared_grads.as_mut().ok_or_else(|| {
            OptimError::InvalidConfig("Adagrad state not initialized".to_string())
        })?;

        let mut updated = params.to_owned();
        let mut params_view = updated.view_mut().into_dyn();
        let gradients_view = gradients.view().into_dyn();

        Zip::from(&mut params_view)
            .and(&gradients_view)
            .and(&mut accumulators[index])
            .for_each(|p, &g, acc| {
                let grad = if use_weight_decay {
                    g + weight_decay * *p
                } else {
                    g
                };
                *acc = *acc + grad * grad;
                *p = *p - lr * grad / (acc.sqrt() + eps);
            });
        drop(params_view);

        Ok(updated)
    }
}

impl<A, D> Optimizer<A, D> for Adagrad<A>
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
