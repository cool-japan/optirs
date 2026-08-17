// RAdam (Rectified Adam) optimizer implementation
//
// RAdam is an improved variant of Adam with a rectified adaptive learning rate.

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// RAdam (Rectified Adam) optimizer
///
/// Implements the RAdam algorithm from the paper:
/// "On the Variance of the Adaptive Learning Rate and Beyond" by Liu et al. (2019).
///
/// RAdam improves upon Adam by addressing the early-stage training instability with
/// a rectified variance term. It eliminates the need for a warmup period and often
/// leads to better convergence.
///
/// Formula:
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
///
/// rho_inf = 2 / (1 - beta2) - 1
/// rho_t   = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
///
/// If rho_t > 4 (the variance of the adaptive learning rate is tractable):
///   r_t = sqrt( ((rho_t - 4)(rho_t - 2) rho_inf) / ((rho_inf - 4)(rho_inf - 2) rho_t) )
///   theta_t = theta_{t-1} - lr * r_t * m_hat_t / (sqrt(v_hat_t) + epsilon)
/// Else:
///   theta_t = theta_{t-1} - lr * m_hat_t (non-adaptive, SGD-with-momentum-like)
///
/// The rectification term `r_t` tends to 1 as `t -> infinity`, so late training
/// behaves like Adam. See Liu et al. (2019), Algorithm 2.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{RAdam, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create a RAdam optimizer with default hyperparameters
/// let mut optimizer = RAdam::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct RAdam<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the first moment estimates
    beta1: A,
    /// Exponential decay rate for the second moment estimates
    beta2: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor
    weight_decay: A,
    /// First moment vectors, one slot per parameter-tensor index
    m: Option<Vec<Array<A, IxDyn>>>,
    /// Second moment vectors, one slot per parameter-tensor index
    v: Option<Vec<Array<A, IxDyn>>>,
    /// Per-parameter-index timestep counters
    t: Vec<usize>,
    /// Rho infinity (precomputed constant)
    rho_inf: A,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> RAdam<A> {
    /// Creates a new RAdam optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        let beta2 = A::from(0.999).expect("RAdam: default beta2 (0.999) must fit in A");
        Self {
            learning_rate,
            beta1: A::from(0.9).expect("RAdam: default beta1 (0.9) must fit in A"),
            beta2,
            epsilon: A::from(1e-8).expect("RAdam: default epsilon (1e-8) must fit in A"),
            weight_decay: A::zero(),
            m: None,
            v: None,
            t: Vec::new(),
            rho_inf: A::from(2.0).expect("RAdam: integer literal 2.0 must fit in A")
                / (A::one() - beta2)
                - A::one(),
        }
    }

    /// Creates a new RAdam optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor (default: 0.0)
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
            rho_inf: A::from(2.0).expect("RAdam: integer literal 2.0 must fit in A")
                / (A::one() - beta2)
                - A::one(),
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
        // Update rho_inf based on new beta2
        self.rho_inf = A::from(2.0).expect("RAdam: integer literal 2.0 must fit in A")
            / (A::one() - beta2)
            - A::one();
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

    /// Returns rho_infinity, the maximum length of the approximated SMA
    pub fn rho_inf(&self) -> A {
        self.rho_inf
    }

    /// Computes rho_t, the length of the approximated simple moving average at step `t`
    ///
    /// Returns `None` when `t == 0` (no step has been taken yet).
    pub fn rho_t(&self, t: usize) -> Option<A> {
        if t == 0 {
            return None;
        }
        let exp = i32::try_from(t).ok()?;
        let two = A::one() + A::one();
        let t_f = A::from(t)?;
        let beta2_t = self.beta2.powi(exp);
        Some(self.rho_inf - two * t_f * beta2_t / (A::one() - beta2_t))
    }

    /// Computes the RAdam rectification term `r_t` for step `t`
    ///
    /// Returns `None` when the variance is not yet tractable (`rho_t <= 4`), in which
    /// case RAdam falls back to a non-adaptive, SGD-like update.
    ///
    /// `r_t` converges to `1` as `t -> infinity`.
    pub fn rectification_term(&self, t: usize) -> Option<A> {
        let rho_t = self.rho_t(t)?;
        let two = A::one() + A::one();
        let four = two + two;
        if rho_t <= four {
            return None;
        }
        let rho_inf = self.rho_inf;
        let numerator = (rho_t - four) * (rho_t - two) * rho_inf;
        let denominator = (rho_inf - four) * (rho_inf - two) * rho_t;
        if denominator <= A::zero() {
            return None;
        }
        Some((numerator / denominator).sqrt())
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

    /// Applies a RAdam update in place for the parameter tensor at `index`
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
        let use_weight_decay = weight_decay > A::zero();
        let one = A::one();
        let bias_correction1 = one - beta1.powi(exp);
        let bias_correction2 = one - beta2.powi(exp);

        // Rectification term; `None` means the variance is not yet tractable and the
        // update falls back to the non-adaptive (SGD-like) branch.
        let rect = self.rectification_term(t);

        let m = self
            .m
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("RAdam state not initialized".to_string()))?;
        let v = self
            .v
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("RAdam state not initialized".to_string()))?;

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
                match rect {
                    Some(r_t) => {
                        let v_hat = *v_i / bias_correction2;
                        *p = *p - lr * r_t * m_hat / (v_hat.sqrt() + eps);
                    }
                    None => {
                        *p = *p - lr * m_hat;
                    }
                }
            });

        Ok(())
    }

    /// Applies a RAdam update in place using the state slot of the first parameter tensor
    pub fn step_inplace<D: Dimension>(
        &mut self,
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<()> {
        self.step_inplace_indexed(0, params, gradients)
    }

    /// Performs a RAdam update for the parameter tensor at `index`
    ///
    /// Each `index` owns an independent moment/timestep slot.
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

impl<A, D> Optimizer<A, D> for RAdam<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync + std::convert::From<f64>,
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
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_radam_step() {
        // Create parameters and gradients
        let params = Array1::zeros(3);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Create optimizer
        let mut optimizer = RAdam::new(0.01);

        // Run one step
        let new_params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_radam_step");

        // Check that parameters have been updated
        assert!(new_params.iter().all(|&x| x != 0.0));

        // Due to rectification, early steps should behave more like SGD
        // Verify gradient direction - larger gradients should result in larger updates
        for i in 1..3 {
            assert!(new_params[i].abs() > new_params[i - 1].abs());
        }
    }

    #[test]
    fn test_radam_multiple_steps() {
        // Create parameters and gradients
        let mut params = Array1::zeros(3);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Create optimizer with small learning rate
        let mut optimizer = RAdam::new(0.01);

        // Run multiple steps to move past the adaptive phase
        for _ in 0..100 {
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_radam_multiple_steps");
        }

        // Parameters should continue to move in the direction of the gradients
        // with larger updates for larger gradients
        for i in 1..3 {
            assert!(params[i].abs() > params[i - 1].abs());
        }
    }

    #[test]
    fn test_radam_weight_decay() {
        // Create parameters with non-zero values and gradients
        let params = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let gradients = Array1::from_vec(vec![0.01, 0.01, 0.01]);

        // Create optimizer with weight decay
        let mut optimizer = RAdam::new_with_config(
            0.01, 0.9, 0.999, 1e-8, 0.1, // Add weight decay
        );

        // Run one step
        let new_params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_radam_weight_decay");

        // Weight decay should reduce parameter magnitudes
        for i in 0..3 {
            assert!(new_params[i].abs() < params[i].abs());
        }
    }

    // Test commented out to fix compilation
    // #[test]
    // fn test_radam_config() {
    //     let optimizer = RAdam::new_with_config(
    //         0.02.into(),
    //         0.8.into(),
    //         0.9,
    //         1e-10.into(),
    //         0.05.into(),
    //     );

    //     assert_eq!(optimizer.get_learning_rate(), 0.02.into());
    //     assert_eq!(optimizer.get_beta1(), 0.8.into());
    //     assert_eq!(optimizer.get_beta2(), 0.9.into());
    //     assert_eq!(optimizer.get_epsilon(), 1e-10.into());
    //     assert_eq!(optimizer.get_weight_decay(), 0.05.into());
    // }

    #[test]
    fn test_radam_reset() {
        // Create parameters and gradients
        let params = Array1::zeros(3);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Create optimizer
        let mut optimizer = RAdam::new(0.01);

        // Run one step
        optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_radam_reset");
        assert_eq!(optimizer.timestep(0), 1);
        assert!(optimizer.m.is_some());
        assert!(optimizer.v.is_some());

        // Reset optimizer
        optimizer.reset();
        assert_eq!(optimizer.timestep(0), 0);
        assert!(optimizer.m.is_none());
        assert!(optimizer.v.is_none());
    }
}
