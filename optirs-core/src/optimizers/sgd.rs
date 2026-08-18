// Stochastic Gradient Descent optimizer

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// Stochastic Gradient Descent optimizer
///
/// Implements the classic SGD algorithm with support for momentum and weight decay.
///
/// Formula:
/// v_t = momentum * v_{t-1} + learning_rate * (gradient + weight_decay * param)
/// param_t = param_{t-1} - v_t
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{SGD, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an SGD optimizer with learning rate 0.01 and momentum 0.9
/// let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct SGD<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Momentum factor (0.0 means no momentum)
    momentum: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Velocity (momentum state), one slot per parameter-tensor index
    velocity: Option<Vec<Array<A, IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> SGD<A> {
    /// Creates a new SGD optimizer with the given learning rate and no momentum/weight decay
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::zero(),
            weight_decay: A::zero(),
            velocity: None,
        }
    }

    /// Creates a new SGD optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn new_with_config(learning_rate: A, momentum: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: None,
        }
    }

    /// Sets the momentum factor
    ///
    /// # Arguments
    ///
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    pub fn set_momentum(&mut self, momentum: A) -> &mut Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to set momentum and return self
    ///
    /// # Arguments
    ///
    /// * `momentum` - The momentum factor (0.0 means no momentum)
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Gets the current momentum factor
    pub fn get_momentum(&self) -> A {
        self.momentum
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the weight decay factor
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to set weight decay and return self
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - The weight decay factor (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the current weight decay factor
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Drops the momentum state
    pub fn reset(&mut self) {
        self.velocity = None;
    }

    /// Ensures a velocity slot exists for `index` and matches `dim`
    fn ensure_state(&mut self, index: usize, dim: &IxDyn) {
        let velocity = self.velocity.get_or_insert_with(Vec::new);
        while velocity.len() <= index {
            velocity.push(Array::zeros(dim.clone()));
        }
        if velocity[index].raw_dim() != *dim {
            velocity[index] = Array::zeros(dim.clone());
        }
    }

    /// Applies an SGD update in place for the parameter tensor at `index`
    ///
    /// This is the allocation-free hot path: velocity and parameters are updated in a
    /// single fused [`Zip`] traversal, so no temporary arrays are created per step.
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

        let momentum = self.momentum;
        let lr = self.learning_rate;
        let weight_decay = self.weight_decay;
        let use_weight_decay = weight_decay > A::zero();
        let use_momentum = momentum > A::zero();

        let velocity = self
            .velocity
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("SGD state not initialized".to_string()))?;

        let mut params_view = params.view_mut().into_dyn();
        let gradients_view = gradients.view().into_dyn();

        Zip::from(&mut params_view)
            .and(&gradients_view)
            .and(&mut velocity[index])
            .for_each(|p, &g, v| {
                let grad = if use_weight_decay {
                    g + weight_decay * *p
                } else {
                    g
                };
                *v = if use_momentum {
                    *v * momentum + grad * lr
                } else {
                    grad * lr
                };
                *p = *p - *v;
            });

        Ok(())
    }

    /// Applies an SGD update in place using the state slot of the first parameter tensor
    pub fn step_inplace<D: Dimension>(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<()> {
        self.step_inplace_indexed(0, params, gradients)
    }

    /// Performs an SGD update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent momentum slot, so several parameter tensors
    /// can be optimized by a single `SGD` instance without their velocities mixing.
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

impl<A, D> Optimizer<A, D> for SGD<A>
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
