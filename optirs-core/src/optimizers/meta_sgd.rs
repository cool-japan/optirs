// Meta-SGD optimizer with per-parameter learnable learning rates
//
// Meta-SGD extends MAML by learning not only the model initialization but also
// per-parameter learning rates. This allows the model to adapt more effectively
// to new tasks by using different learning rates for different parameters.
//
// Reference: Li, Z., Zhou, F., Chen, F., & Li, H. (2017).
// "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning"

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Meta-SGD optimizer with per-parameter learnable learning rates
///
/// Implements the Meta-SGD algorithm which learns per-parameter learning rates
/// alongside the model parameters. Each parameter gets its own adaptive learning
/// rate that is updated based on the meta-gradient.
///
/// # Algorithm
///
/// For each step:
/// 1. Initialize per-parameter learning rates alpha_i to base_lr (if first step)
/// 2. Compute parameter update: delta_i = alpha_i * grad_i
/// 3. Update parameters: theta_i = theta_i - delta_i
/// 4. Update per-parameter LRs: alpha_i = alpha_i - alpha_lr * grad_i * delta_i
/// 5. Clamp alpha_i to [1e-8, 10.0]
///
/// The per-parameter learning rates evolve over time, allowing the optimizer to
/// automatically discover the best learning rate for each parameter dimension.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{MetaSGD, Optimizer};
///
/// let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let gradients = Array1::from_vec(vec![0.1, -0.2, 0.3]);
///
/// let mut optimizer = MetaSGD::new(0.01);
/// let new_params = optimizer.step(&params, &gradients).expect("step failed");
/// ```
#[derive(Debug, Clone)]
pub struct MetaSGD<A: Float + ScalarOperand + Debug> {
    /// Base learning rate (used to initialize per-parameter LRs)
    base_lr: A,
    /// Learning rate for updating per-parameter learning rates (meta-learning rate)
    alpha_lr: A,
    /// Number of inner adaptation steps
    inner_steps: usize,
    /// Per-parameter learnable learning rates
    per_param_lr: Option<Array<A, IxDyn>>,
    /// Count of steps taken
    step_count: usize,
}

impl<A: Float + ScalarOperand + Debug> MetaSGD<A> {
    /// Creates a new Meta-SGD optimizer with the given base learning rate
    ///
    /// Defaults:
    /// - alpha_lr: 0.001
    /// - inner_steps: 5
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate for initializing per-parameter LRs
    pub fn new(base_lr: A) -> Self {
        Self {
            base_lr,
            alpha_lr: A::from(0.001).expect("MetaSGD: failed to convert alpha_lr constant"),
            inner_steps: 5,
            per_param_lr: None,
            step_count: 0,
        }
    }

    /// Sets the meta-learning rate for updating per-parameter learning rates
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for the per-parameter LR updates
    pub fn with_alpha_lr(mut self, lr: A) -> Self {
        self.alpha_lr = lr;
        self
    }

    /// Sets the number of inner adaptation steps
    ///
    /// # Arguments
    ///
    /// * `n` - Number of inner steps (must be >= 1)
    pub fn with_inner_steps(mut self, n: usize) -> Self {
        self.inner_steps = if n == 0 { 1 } else { n };
        self
    }

    /// Returns the base learning rate
    pub fn get_base_lr(&self) -> A {
        self.base_lr
    }

    /// Returns the meta-learning rate (alpha_lr)
    pub fn get_alpha_lr(&self) -> A {
        self.alpha_lr
    }

    /// Returns the number of inner adaptation steps
    pub fn get_inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Returns the number of steps taken so far
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }

    /// Returns a reference to the current per-parameter learning rates, if initialized
    pub fn get_per_param_lr(&self) -> Option<&Array<A, IxDyn>> {
        self.per_param_lr.as_ref()
    }

    /// Resets the per-parameter learning rates (they will be re-initialized on next step)
    pub fn reset_per_param_lr(&mut self) {
        self.per_param_lr = None;
    }

    /// Clamp learning rate values to the valid range [min_val, max_val]
    fn clamp_lr_array(lr_array: &mut Array<A, IxDyn>, min_val: A, max_val: A) {
        lr_array.mapv_inplace(|v| {
            if v < min_val {
                min_val
            } else if v > max_val {
                max_val
            } else {
                v
            }
        });
    }
}

impl<A, D> Optimizer<A, D> for MetaSGD<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        let min_lr = A::from(1e-8).expect("MetaSGD: failed to convert min_lr constant");
        let max_lr = A::from(10.0).expect("MetaSGD: failed to convert max_lr constant");

        // Step 1: Initialize per-parameter learning rates if needed
        if self.per_param_lr.is_none() {
            let lr_init = Array::<A, IxDyn>::from_elem(params_dyn.raw_dim(), self.base_lr);
            self.per_param_lr = Some(lr_init);
        }

        // Handle shape mismatch (if params shape changed since last call)
        {
            let current_lr = self
                .per_param_lr
                .as_ref()
                .expect("MetaSGD: per_param_lr should be initialized");
            if current_lr.raw_dim() != params_dyn.raw_dim() {
                self.per_param_lr = Some(Array::<A, IxDyn>::from_elem(
                    params_dyn.raw_dim(),
                    self.base_lr,
                ));
            }
        }

        let per_param_lr = self
            .per_param_lr
            .as_ref()
            .expect("MetaSGD: per_param_lr should be initialized")
            .clone();

        // Step 2-3: Apply inner adaptation steps using per-parameter learning rates
        let mut adapted_params = params_dyn.clone();
        let mut cumulative_delta = Array::<A, IxDyn>::zeros(params_dyn.raw_dim());

        for _ in 0..self.inner_steps {
            // delta = per_param_lr * gradients
            let delta = &per_param_lr * &gradients_dyn;
            // Accumulate total parameter change for meta-gradient
            cumulative_delta = &cumulative_delta + &delta;
            // Update adapted params
            adapted_params = &adapted_params - &delta;
        }

        // Step 4: Update per-parameter learning rates using meta-gradient
        // The meta-gradient for alpha is: grad * cumulative_delta
        // This encourages learning rates that reduce the loss
        let meta_gradient = &gradients_dyn * &cumulative_delta;
        let mut updated_lr = &per_param_lr - &(&meta_gradient * self.alpha_lr);

        // Step 5: Clamp per-parameter learning rates
        Self::clamp_lr_array(&mut updated_lr, min_lr, max_lr);

        self.per_param_lr = Some(updated_lr);
        self.step_count += 1;

        // Convert back to original dimension
        Ok(adapted_params
            .into_dimensionality::<D>()
            .expect("MetaSGD: failed to convert back to original dimensionality"))
    }

    fn get_learning_rate(&self) -> A {
        self.base_lr
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.base_lr = learning_rate;
        // Reset per-param LRs so they re-initialize with new base_lr
        self.per_param_lr = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_meta_sgd_basic_creation() {
        let optimizer: MetaSGD<f64> = MetaSGD::new(0.01);
        assert!((optimizer.get_base_lr() - 0.01).abs() < 1e-10);
        assert!((optimizer.get_alpha_lr() - 0.001).abs() < 1e-10);
        assert_eq!(optimizer.get_inner_steps(), 5);
        assert_eq!(optimizer.get_step_count(), 0);
        assert!(optimizer.get_per_param_lr().is_none());
    }

    #[test]
    fn test_meta_sgd_builder_pattern() {
        let optimizer: MetaSGD<f64> = MetaSGD::new(0.01)
            .with_alpha_lr(0.0001)
            .with_inner_steps(10);

        assert!((optimizer.get_alpha_lr() - 0.0001).abs() < 1e-10);
        assert_eq!(optimizer.get_inner_steps(), 10);
    }

    #[test]
    fn test_meta_sgd_step_works() {
        let mut optimizer = MetaSGD::new(0.1_f64).with_inner_steps(1);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.5, -0.5, 0.0]);

        let new_params = optimizer.step(&params, &gradients).expect("step failed");

        // With inner_steps=1, base_lr=0.1:
        // delta = per_param_lr * gradients = [0.1*0.5, 0.1*(-0.5), 0.1*0.0] = [0.05, -0.05, 0.0]
        // new_params = params - delta = [0.95, 2.05, 3.0]
        assert!((new_params[0] - 0.95).abs() < 1e-10);
        assert!((new_params[1] - 2.05).abs() < 1e-10);
        assert!((new_params[2] - 3.0).abs() < 1e-10);
        assert_eq!(optimizer.get_step_count(), 1);

        // Per-param LR should be initialized now
        assert!(optimizer.get_per_param_lr().is_some());
    }

    #[test]
    fn test_meta_sgd_per_param_lr_adaptation() {
        let mut optimizer = MetaSGD::new(0.1_f64)
            .with_alpha_lr(0.01)
            .with_inner_steps(1);

        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![1.0, 0.001]);

        // First step initializes per-param LRs
        let _ = optimizer.step(&params, &gradients).expect("step failed");

        let lr_after_first = optimizer
            .get_per_param_lr()
            .expect("per_param_lr should exist")
            .clone();

        // The parameter with larger gradient (dim 0) should have its LR adjusted more
        // than the parameter with smaller gradient (dim 1)
        // meta_gradient = grad * delta = grad * (lr * grad) = lr * grad^2
        // For dim 0: meta_grad = 0.1 * 1.0^2 = 0.1
        //   new_lr = 0.1 - 0.01 * 0.1 = 0.099
        // For dim 1: meta_grad = 0.1 * 0.001^2 = 0.0000001
        //   new_lr = 0.1 - 0.01 * 0.0000001 ≈ 0.1
        let lr_diff_0 = (lr_after_first[0] - 0.1_f64).abs();
        let lr_diff_1 = (lr_after_first[1] - 0.1_f64).abs();
        assert!(
            lr_diff_0 > lr_diff_1,
            "Larger gradient dimension should have more LR change: diff_0={lr_diff_0}, diff_1={lr_diff_1}"
        );
    }

    #[test]
    fn test_meta_sgd_convergence_toward_minimum() {
        // Optimize f(x) = x^2, gradient = 2x
        let mut optimizer = MetaSGD::new(0.05_f64)
            .with_alpha_lr(0.0001)
            .with_inner_steps(1);

        let mut params = Array1::from_vec(vec![5.0, -3.0, 2.0]);

        for _ in 0..200 {
            let gradients = &params * 2.0;
            params = optimizer.step(&params, &gradients).expect("step failed");
        }

        // After many steps, params should be close to zero
        for &val in params.iter() {
            assert!(
                val.abs() < 0.5,
                "Parameter {val} did not converge to near zero"
            );
        }
    }

    #[test]
    fn test_meta_sgd_lr_clamping() {
        // Use very large alpha_lr to force per-param LRs to be clamped
        let mut optimizer = MetaSGD::new(0.1_f64)
            .with_alpha_lr(100.0) // Extremely large meta-LR
            .with_inner_steps(1);

        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![1.0, -1.0]);

        // Run a step - the large alpha_lr should cause LRs to hit clamp bounds
        let _ = optimizer.step(&params, &gradients).expect("step failed");

        let per_param_lr = optimizer
            .get_per_param_lr()
            .expect("per_param_lr should exist");

        // All LR values should be within [1e-8, 10.0]
        for &lr in per_param_lr.iter() {
            assert!(
                (1e-8..=10.0).contains(&lr),
                "Per-param LR {lr} is out of clamped range [1e-8, 10.0]"
            );
        }
    }

    #[test]
    fn test_meta_sgd_zero_gradient() {
        let mut optimizer = MetaSGD::new(0.1_f64).with_inner_steps(3);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.0, 0.0, 0.0]);

        let new_params = optimizer.step(&params, &gradients).expect("step failed");

        // With zero gradients, params should not change
        for (p, np) in params.iter().zip(new_params.iter()) {
            assert!(
                (*p - *np).abs() < 1e-12,
                "Params changed with zero gradient"
            );
        }
    }

    #[test]
    fn test_meta_sgd_set_learning_rate_resets_per_param() {
        let mut optimizer = MetaSGD::new(0.1_f64);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2]);

        let _ = optimizer.step(&params, &gradients).expect("step failed");
        assert!(optimizer.get_per_param_lr().is_some());

        // Setting learning rate should reset per-param LRs
        Optimizer::<f64, scirs2_core::ndarray::Ix1>::set_learning_rate(&mut optimizer, 0.05);
        assert!(optimizer.get_per_param_lr().is_none());
        assert!(
            (Optimizer::<f64, scirs2_core::ndarray::Ix1>::get_learning_rate(&optimizer) - 0.05)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_meta_sgd_inner_steps_zero_clamps_to_one() {
        let optimizer: MetaSGD<f64> = MetaSGD::new(0.01).with_inner_steps(0);
        assert_eq!(optimizer.get_inner_steps(), 1);
    }

    #[test]
    fn test_meta_sgd_multiple_steps_count() {
        let mut optimizer = MetaSGD::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2]);

        for i in 0..5 {
            let _ = optimizer.step(&params, &gradients).expect("step failed");
            assert_eq!(optimizer.get_step_count(), i + 1);
        }
    }

    #[test]
    fn test_meta_sgd_reset_per_param_lr() {
        let mut optimizer = MetaSGD::new(0.1_f64);
        let params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![0.1]);

        let _ = optimizer.step(&params, &gradients).expect("step failed");
        assert!(optimizer.get_per_param_lr().is_some());

        optimizer.reset_per_param_lr();
        assert!(optimizer.get_per_param_lr().is_none());
    }
}
