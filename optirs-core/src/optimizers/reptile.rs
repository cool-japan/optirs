// Reptile meta-learning optimizer
//
// Reptile is a meta-learning algorithm that learns a good initialization for
// model parameters. It works by:
// 1. Saving the initial parameters (theta)
// 2. Running N inner SGD steps on a task to get theta_adapted
// 3. Updating: theta += epsilon * (theta_adapted - theta)
//
// Reference: Nichol, A., Achiam, J., & Schulman, J. (2018).
// "On First-Order Meta-Learning Algorithms"

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Reptile meta-learning optimizer
///
/// Implements the Reptile algorithm for meta-learning. Reptile performs multiple
/// inner SGD steps on a task, then interpolates between the original parameters
/// and the adapted parameters using an interpolation factor epsilon.
///
/// # Algorithm
///
/// For each step:
/// 1. Save initial parameters theta_0
/// 2. Perform `inner_steps` SGD updates: theta_k = theta_{k-1} - inner_lr * grad
/// 3. Compute meta-update: theta_new = theta_0 + epsilon * (theta_K - theta_0)
///
/// This effectively moves the initialization point toward a region that is
/// beneficial for fast adaptation across tasks.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{ReptileOptimizer, Optimizer};
///
/// let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let gradients = Array1::from_vec(vec![0.1, -0.2, 0.3]);
///
/// let mut optimizer = ReptileOptimizer::new(0.01);
/// let new_params = optimizer.step(&params, &gradients).expect("step failed");
/// ```
#[derive(Debug, Clone)]
pub struct ReptileOptimizer<A: Float + ScalarOperand + Debug> {
    /// Outer learning rate (used as default for epsilon)
    learning_rate: A,
    /// Inner SGD learning rate for task adaptation
    inner_lr: A,
    /// Number of inner loop SGD steps
    inner_steps: usize,
    /// Interpolation factor between original and adapted parameters
    epsilon: A,
    /// Count of outer steps taken
    step_count: usize,
}

impl<A: Float + ScalarOperand + Debug> ReptileOptimizer<A> {
    /// Creates a new Reptile optimizer with the given outer learning rate
    ///
    /// Defaults:
    /// - inner_steps: 5
    /// - epsilon: same as learning_rate
    /// - inner_lr: same as learning_rate
    ///
    /// # Arguments
    ///
    /// * `lr` - The outer learning rate (also used as default epsilon and inner_lr)
    pub fn new(lr: A) -> Self {
        Self {
            learning_rate: lr,
            inner_lr: lr,
            inner_steps: 5,
            epsilon: lr,
            step_count: 0,
        }
    }

    /// Sets the number of inner SGD steps
    ///
    /// More inner steps allow better task adaptation but increase computation.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of inner SGD steps (must be >= 1)
    pub fn with_inner_steps(mut self, n: usize) -> Self {
        self.inner_steps = if n == 0 { 1 } else { n };
        self
    }

    /// Sets the interpolation factor epsilon
    ///
    /// Controls how much the meta-update moves toward the adapted parameters.
    /// Smaller values mean more conservative updates.
    ///
    /// # Arguments
    ///
    /// * `e` - Interpolation factor (typically in [0, 1])
    pub fn with_epsilon(mut self, e: A) -> Self {
        self.epsilon = e;
        self
    }

    /// Sets the inner SGD learning rate
    ///
    /// This learning rate is used for the inner adaptation steps on each task.
    ///
    /// # Arguments
    ///
    /// * `lr` - Inner learning rate
    pub fn with_inner_lr(mut self, lr: A) -> Self {
        self.inner_lr = lr;
        self
    }

    /// Returns the number of inner steps configured
    pub fn get_inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Returns the current epsilon (interpolation factor)
    pub fn get_epsilon(&self) -> A {
        self.epsilon
    }

    /// Returns the inner learning rate
    pub fn get_inner_lr(&self) -> A {
        self.inner_lr
    }

    /// Returns the number of outer steps taken so far
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }
}

impl<A, D> Optimizer<A, D> for ReptileOptimizer<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for internal computation
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Save original parameters (theta_0)
        let theta_original = params_dyn.clone();

        // Run inner SGD steps: theta_k = theta_{k-1} - inner_lr * gradients
        // In Reptile, we simulate multiple inner steps using the same gradient
        // (in practice, each step would use a gradient from the current params,
        // but with a single gradient call we approximate this)
        let mut theta_adapted = params_dyn;
        for _ in 0..self.inner_steps {
            theta_adapted = &theta_adapted - &(&gradients_dyn * self.inner_lr);
        }

        // Compute the meta-update direction: (theta_adapted - theta_original)
        let meta_direction = &theta_adapted - &theta_original;

        // Apply Reptile update: theta_new = theta_original + epsilon * meta_direction
        let updated_params = &theta_original + &(&meta_direction * self.epsilon);

        self.step_count += 1;

        // Convert back to original dimension
        Ok(updated_params
            .into_dimensionality::<D>()
            .expect("Reptile: failed to convert back to original dimensionality"))
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
        self.epsilon = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_reptile_basic_creation() {
        let optimizer: ReptileOptimizer<f64> = ReptileOptimizer::new(0.01);
        assert!(
            (Optimizer::<f64, scirs2_core::ndarray::Ix1>::get_learning_rate(&optimizer) - 0.01)
                .abs()
                < 1e-10
        );
        assert_eq!(optimizer.get_inner_steps(), 5);
        assert!((optimizer.get_epsilon() - 0.01).abs() < 1e-10);
        assert!((optimizer.get_inner_lr() - 0.01).abs() < 1e-10);
        assert_eq!(optimizer.get_step_count(), 0);
    }

    #[test]
    fn test_reptile_builder_pattern() {
        let optimizer: ReptileOptimizer<f64> = ReptileOptimizer::new(0.01)
            .with_inner_steps(10)
            .with_epsilon(0.05)
            .with_inner_lr(0.001);

        assert_eq!(optimizer.get_inner_steps(), 10);
        assert!((optimizer.get_epsilon() - 0.05).abs() < 1e-10);
        assert!((optimizer.get_inner_lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_reptile_step_works() {
        let mut optimizer = ReptileOptimizer::new(0.1_f64)
            .with_inner_steps(1)
            .with_epsilon(1.0)
            .with_inner_lr(0.1);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.5, -0.5, 0.0]);

        let new_params = optimizer.step(&params, &gradients).expect("step failed");

        // With inner_steps=1, epsilon=1.0:
        // theta_adapted = params - inner_lr * gradients = [1.0 - 0.05, 2.0 + 0.05, 3.0]
        // meta_direction = theta_adapted - params = [-0.05, 0.05, 0.0]
        // result = params + 1.0 * meta_direction = [0.95, 2.05, 3.0]
        assert!((new_params[0] - 0.95).abs() < 1e-10);
        assert!((new_params[1] - 2.05).abs() < 1e-10);
        assert!((new_params[2] - 3.0).abs() < 1e-10);
        assert_eq!(optimizer.get_step_count(), 1);
    }

    #[test]
    fn test_reptile_convergence_toward_minimum() {
        // Optimize f(x) = x^2, gradient = 2x
        // Minimum is at x = 0
        let mut optimizer = ReptileOptimizer::new(0.1_f64)
            .with_inner_steps(3)
            .with_epsilon(0.5)
            .with_inner_lr(0.1);

        let mut params = Array1::from_vec(vec![5.0, -3.0, 2.0]);

        for _ in 0..100 {
            let gradients = &params * 2.0; // gradient of x^2
            params = optimizer.step(&params, &gradients).expect("step failed");
        }

        // After many steps, params should be close to zero
        for &val in params.iter() {
            assert!(
                val.abs() < 0.1,
                "Parameter {val} did not converge to near zero"
            );
        }
    }

    #[test]
    fn test_reptile_multiple_steps_decrement_count() {
        let mut optimizer = ReptileOptimizer::new(0.01_f64);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2]);

        for i in 0..5 {
            let _new_params = optimizer.step(&params, &gradients).expect("step failed");
            assert_eq!(optimizer.get_step_count(), i + 1);
        }
        assert_eq!(optimizer.get_step_count(), 5);
    }

    #[test]
    fn test_reptile_zero_gradient() {
        let mut optimizer = ReptileOptimizer::new(0.1_f64).with_inner_steps(5);

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
    fn test_reptile_inner_steps_zero_clamps_to_one() {
        let optimizer: ReptileOptimizer<f64> = ReptileOptimizer::new(0.01).with_inner_steps(0);
        assert_eq!(optimizer.get_inner_steps(), 1);
    }

    #[test]
    fn test_reptile_set_learning_rate() {
        let mut optimizer: ReptileOptimizer<f64> = ReptileOptimizer::new(0.01);
        Optimizer::<f64, scirs2_core::ndarray::Ix1>::set_learning_rate(&mut optimizer, 0.05);
        assert!(
            (Optimizer::<f64, scirs2_core::ndarray::Ix1>::get_learning_rate(&optimizer) - 0.05)
                .abs()
                < 1e-10
        );
        assert!((optimizer.get_epsilon() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_reptile_multiple_inner_steps_effect() {
        // More inner steps should result in a larger effective update
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let mut opt_1step = ReptileOptimizer::new(0.1_f64)
            .with_inner_steps(1)
            .with_epsilon(1.0)
            .with_inner_lr(0.1);

        let mut opt_5steps = ReptileOptimizer::new(0.1_f64)
            .with_inner_steps(5)
            .with_epsilon(1.0)
            .with_inner_lr(0.1);

        let result_1 = opt_1step.step(&params, &gradients).expect("step failed");
        let result_5 = opt_5steps.step(&params, &gradients).expect("step failed");

        // 5-step version should have moved further from original params
        let diff_1: f64 = params
            .iter()
            .zip(result_1.iter())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum();
        let diff_5: f64 = params
            .iter()
            .zip(result_5.iter())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum();

        assert!(
            diff_5 > diff_1,
            "More inner steps should cause larger displacement: diff_5={diff_5}, diff_1={diff_1}"
        );
    }
}
