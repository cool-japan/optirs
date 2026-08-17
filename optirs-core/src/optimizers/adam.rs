// Adam optimizer implementation

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// SciRS2 Integration
// Note: OptiRS receives pre-computed gradients, so scirs2-autograd is not needed
use scirs2_optimize::stochastic::{minimize_adam, AdamOptions};

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// Adam optimizer
///
/// Implements the Adam optimization algorithm from the paper:
/// "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014).
///
/// Formula:
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
/// theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{Adam, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an Adam optimizer with default hyperparameters
/// let mut optimizer = Adam::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("unwrap failed");
/// ```
#[derive(Debug, Clone)]
pub struct Adam<A: Float + ScalarOperand + Debug> {
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
    /// First moment vectors, one slot per parameter-tensor index
    m: Option<Vec<Array<A, IxDyn>>>,
    /// Second moment vectors, one slot per parameter-tensor index
    v: Option<Vec<Array<A, IxDyn>>>,
    /// Per-parameter-index timestep counters
    ///
    /// Each parameter tensor passed through [`Optimizer::step_list`] keeps its own
    /// timestep so that bias correction is computed independently per tensor.
    t: Vec<usize>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> Adam<A> {
    /// Creates a new Adam optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).expect("unwrap failed"),
            beta2: A::from(0.999).expect("unwrap failed"),
            epsilon: A::from(1e-8).expect("unwrap failed"),
            weight_decay: A::zero(),
            m: None,
            v: None,
            t: Vec::new(),
        }
    }

    /// Creates a new Adam optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
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

    /// Builder method to set beta1 and return self
    pub fn with_beta1(mut self, beta1: A) -> Self {
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

    /// Builder method to set beta2 and return self
    pub fn with_beta2(mut self, beta2: A) -> Self {
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

    /// Builder method to set epsilon and return self
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
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

    /// Builder method to set weight decay and return self
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
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
    ///
    /// Returns the new (1-based) timestep for that index.
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

    /// Applies an Adam update in place for the parameter tensor at `index`
    ///
    /// This is the allocation-free hot path: the moments and the parameters are
    /// updated with a single fused [`Zip`] traversal, so no temporary arrays are
    /// created per step.
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
        let t = self.advance_state(index, &dim)?;

        let exp = i32::try_from(t).map_err(|_| {
            OptimError::InvalidConfig(
                "Timestep too large for bias correction calculation".to_string(),
            )
        })?;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let lr = self.learning_rate;
        let eps = self.epsilon;
        let weight_decay = self.weight_decay;
        let one = A::one();
        let bias_correction1 = one - beta1.powi(exp);
        let bias_correction2 = one - beta2.powi(exp);
        let use_weight_decay = weight_decay > A::zero();

        let m = self
            .m
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("Adam state not initialized".to_string()))?;
        let v = self
            .v
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("Adam state not initialized".to_string()))?;

        let mut params_view = params.view_mut().into_dyn();
        let gradients_view = gradients.view().into_dyn();

        Zip::from(&mut params_view)
            .and(&gradients_view)
            .and(&mut m[index])
            .and(&mut v[index])
            .for_each(|p, &g, m_i, v_i| {
                let grad = if use_weight_decay {
                    g + weight_decay * *p
                } else {
                    g
                };
                *m_i = *m_i * beta1 + grad * (one - beta1);
                *v_i = *v_i * beta2 + grad * grad * (one - beta2);
                let m_hat = *m_i / bias_correction1;
                let v_hat = *v_i / bias_correction2;
                *p = *p - lr * m_hat / (v_hat.sqrt() + eps);
            });

        Ok(())
    }

    /// Applies an Adam update in place using the state slot of the first parameter tensor
    pub fn step_inplace<D: Dimension>(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<()> {
        self.step_inplace_indexed(0, params, gradients)
    }

    /// Performs an Adam update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent moment/timestep slot, so several parameter
    /// tensors of different shapes can be optimized by a single `Adam` instance
    /// without their state interfering.
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

impl<A, D> Optimizer<A, D> for Adam<A>
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
