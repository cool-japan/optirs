// Layer-wise Adaptive Rate Scaling (LARS) optimizer
//
// LARS is an optimization algorithm specifically designed for large batch training
// in deep neural networks. It scales the learning rate for each layer based on the
// ratio of the weight norm to the gradient norm.
//
// References:
// - [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Layer-wise Adaptive Rate Scaling (LARS) optimizer
///
/// LARS is an optimization algorithm specifically designed for large batch training,
/// which allows scaling up the batch size significantly without loss of accuracy.
/// It works by adapting the learning rate per layer based on the ratio of
/// weight norm to gradient norm.
///
/// # Parameters
///
/// * `learning_rate` - Base learning rate
/// * `momentum` - Momentum factor (default: 0.9)
/// * `weight_decay` - Weight decay factor (default: 0.0001)
/// * `trust_coefficient` - Trust coefficient for scaling (default: 0.001)
/// * `eps` - Small constant for numerical stability (default: 1e-8)
/// * `exclude_bias_and_norm` - Whether to exclude bias and normalization parameters
///   from LARS trust-ratio scaling (default: true). Bias and normalization
///   parameters are identified by their rank: tensors with one dimension or fewer
///   (`ndim() <= 1`) are treated as biases / normalization scales, matching the
///   convention used by the reference LARS and LAMB implementations.
///
/// # Example
///
/// ```no_run
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{LARS, Optimizer};
///
/// let mut optimizer = LARS::new(0.01)
///     .with_momentum(0.9)
///     .with_weight_decay(0.0001)
///     .with_trust_coefficient(0.001);
///
/// let params = Array1::zeros(10);
/// let gradients = Array1::ones(10);
///
/// let updated_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// // Parameters are automatically updated
/// ```
#[derive(Debug, Clone)]
pub struct LARS<A: Float> {
    learning_rate: A,
    momentum: A,
    weight_decay: A,
    trust_coefficient: A,
    eps: A,
    exclude_bias_and_norm: bool,
    /// Momentum buffers, one flat buffer per parameter-tensor index
    velocity: Option<Vec<Vec<A>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> LARS<A> {
    /// Create a new LARS optimizer with the given learning rate
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::from(0.9).expect("LARS: default momentum (0.9) must fit in A"),
            weight_decay: A::from(0.0001)
                .expect("LARS: default weight_decay (0.0001) must fit in A"),
            trust_coefficient: A::from(0.001)
                .expect("LARS: default trust_coefficient (0.001) must fit in A"),
            eps: A::from(1e-8).expect("LARS: default eps (1e-8) must fit in A"),
            exclude_bias_and_norm: true,
            velocity: None,
        }
    }

    /// Set the momentum factor
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the weight decay factor
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the trust coefficient
    pub fn with_trust_coefficient(mut self, trust_coefficient: A) -> Self {
        self.trust_coefficient = trust_coefficient;
        self
    }

    /// Set the epsilon value for numerical stability
    pub fn with_eps(mut self, eps: A) -> Self {
        self.eps = eps;
        self
    }

    /// Set whether to exclude bias and normalization layers from LARS adaptation
    pub fn with_exclude_bias_and_norm(mut self, exclude_bias_and_norm: bool) -> Self {
        self.exclude_bias_and_norm = exclude_bias_and_norm;
        self
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.velocity = None;
    }

    /// Ensures a momentum buffer exists for `index` with `len` elements
    fn ensure_state(&mut self, index: usize, len: usize) {
        let velocity = self.velocity.get_or_insert_with(Vec::new);
        while velocity.len() <= index {
            velocity.push(vec![A::zero(); len]);
        }
        if velocity[index].len() != len {
            velocity[index] = vec![A::zero(); len];
        }
    }

    /// Performs a LARS update for the parameter tensor at `index`
    ///
    /// LARS is a *layer-wise* algorithm: the trust ratio is computed per tensor, and
    /// each tensor keeps its own momentum buffer.
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

        // A bias / normalization parameter is a rank <= 1 tensor.
        let is_bias_or_norm = params.ndim() <= 1;
        let n_params = gradients.len();
        self.ensure_state(index, n_params);

        // Calculate weight norm and gradient norm over this tensor only.
        let weight_norm = params.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();

        // Determine if we should apply LARS scaling.
        // Bias and normalization parameters (rank <= 1) are excluded when configured,
        // and fall back to plain SGD-with-momentum, exactly as the paper prescribes.
        let should_apply_lars = !(self.exclude_bias_and_norm && is_bias_or_norm);

        // Calculate local learning rate using the trust ratio
        let local_lr = if should_apply_lars && weight_norm > A::zero() && grad_norm > A::zero() {
            self.trust_coefficient * weight_norm
                / (grad_norm + self.weight_decay * weight_norm + self.eps)
        } else {
            A::one()
        };

        let scaled_lr = self.learning_rate * local_lr;
        let momentum = self.momentum;
        let weight_decay = self.weight_decay;
        let use_weight_decay = weight_decay > A::zero();

        let velocity = self
            .velocity
            .as_mut()
            .ok_or_else(|| OptimError::InvalidConfig("LARS state not initialized".to_string()))?;
        let buffer = velocity.get_mut(index).ok_or_else(|| {
            OptimError::InvalidConfig(format!("LARS has no velocity buffer for index {}", index))
        })?;

        let mut updated_params = params.clone();
        for (slot, (p, g)) in buffer
            .iter_mut()
            .zip(updated_params.iter_mut().zip(gradients.iter()))
        {
            let grad = if use_weight_decay {
                *g + weight_decay * *p
            } else {
                *g
            };
            *slot = momentum * *slot + grad * scaled_lr;
            *p = *p - *slot;
        }

        Ok(updated_params)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync> Optimizer<A, D>
    for LARS<A>
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

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_lars_creation() {
        let optimizer = LARS::new(0.01);
        assert_abs_diff_eq!(optimizer.learning_rate, 0.01);
        assert_abs_diff_eq!(optimizer.momentum, 0.9);
        assert_abs_diff_eq!(optimizer.weight_decay, 0.0001);
        assert_abs_diff_eq!(optimizer.trust_coefficient, 0.001);
        assert_abs_diff_eq!(optimizer.eps, 1e-8);
        assert!(optimizer.exclude_bias_and_norm);
    }

    #[test]
    fn test_lars_builder() {
        let optimizer = LARS::new(0.01)
            .with_momentum(0.95)
            .with_weight_decay(0.0005)
            .with_trust_coefficient(0.01)
            .with_eps(1e-6)
            .with_exclude_bias_and_norm(false);

        assert_abs_diff_eq!(optimizer.momentum, 0.95);
        assert_abs_diff_eq!(optimizer.weight_decay, 0.0005);
        assert_abs_diff_eq!(optimizer.trust_coefficient, 0.01);
        assert_abs_diff_eq!(optimizer.eps, 1e-6);
        assert!(!optimizer.exclude_bias_and_norm);
    }

    #[test]
    fn test_lars_update() {
        // 1-D tensors are treated as bias / normalization parameters, which LARS
        // excludes by default. Opt in explicitly to exercise the trust-ratio path.
        let mut optimizer = LARS::new(0.1)
            .with_momentum(0.9)
            .with_weight_decay(0.0)
            .with_trust_coefficient(1.0)
            .with_exclude_bias_and_norm(false);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First update
        let updated_params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lars_update");

        // LARS scaling factor with trust_coefficient=1.0 should be:
        // weight_norm / grad_norm = sqrt(14) / sqrt(0.14) ≈ 10
        // So the effective learning rate is 0.1 * 10 = 1.0
        // Scale is approximately 10, but let's check actual value (more precise)
        let weight_norm = params.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        let scale = weight_norm / grad_norm;

        assert_abs_diff_eq!(updated_params[0], 1.0 - 0.1 * scale * 0.1, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[1], 2.0 - 0.1 * scale * 0.2, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[2], 3.0 - 0.1 * scale * 0.3, epsilon = 1e-5);

        // Second update should include momentum
        let updated_params2 = optimizer
            .step(&updated_params, &gradients)
            .expect("step succeeds in test_lars_update");

        // For the second update, the velocity will be updated with momentum
        // Just check that parameters continue to change in the expected direction
        assert!(updated_params2[0] < updated_params[0]);
        assert!(updated_params2[1] < updated_params[1]);
        assert!(updated_params2[2] < updated_params[2]);
    }

    #[test]
    fn test_lars_weight_decay() {
        let mut optimizer = LARS::new(0.01)
            .with_momentum(0.0) // No momentum for clarity
            .with_weight_decay(0.1)
            .with_trust_coefficient(1.0)
            .with_exclude_bias_and_norm(false);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let updated_params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lars_weight_decay");

        // Gradients with weight decay: [0.1, 0.2, 0.3] + 0.1*[1.0, 2.0, 3.0] = [0.2, 0.4, 0.6]
        // LARS scaling factor includes weight decay in denominator
        // weight_norm / (grad_norm + weight_decay * weight_norm)
        // = sqrt(14) / (sqrt(0.56) + 0.1*sqrt(14))
        let weight_norm = params.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        let expected_scale = weight_norm / (grad_norm + 0.1 * weight_norm);

        // Check calculation is approximately correct (allowing for floating point differences)
        let expected_p0 = 1.0 - 0.01 * expected_scale * (0.1 + 0.1 * 1.0);
        let expected_p1 = 2.0 - 0.01 * expected_scale * (0.2 + 0.1 * 2.0);
        let expected_p2 = 3.0 - 0.01 * expected_scale * (0.3 + 0.1 * 3.0);

        assert_abs_diff_eq!(updated_params[0], expected_p0, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[1], expected_p1, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[2], expected_p2, epsilon = 1e-5);
    }

    #[test]
    fn test_zero_gradients() {
        let mut optimizer = LARS::new(0.01);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let zero_gradients = Array1::zeros(3);

        let updated_params = optimizer
            .step(&params, &zero_gradients)
            .expect("step succeeds in test_zero_gradients");

        // With zero gradients, only weight decay should contribute to the update
        // With small weight decay (0.0001), changes should be very small
        assert_abs_diff_eq!(updated_params[0], params[0], epsilon = 1e-3);
        assert_abs_diff_eq!(updated_params[1], params[1], epsilon = 1e-3);
        assert_abs_diff_eq!(updated_params[2], params[2], epsilon = 1e-3);
    }

    #[test]
    fn test_exclude_bias_and_norm() {
        let mut optimizer_excluded = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_exclude_bias_and_norm(true);

        let mut optimizer_included = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_exclude_bias_and_norm(false);

        // Test with parameters that could be bias (small 1D array)
        let bias_params = Array1::from_vec(vec![0.1, 0.2]);
        let bias_grads = Array1::from_vec(vec![0.01, 0.02]);

        let updated_excluded = optimizer_excluded
            .step(&bias_params, &bias_grads)
            .expect("step succeeds in test_exclude_bias_and_norm");
        let updated_included = optimizer_included
            .step(&bias_params, &bias_grads)
            .expect("step succeeds in test_exclude_bias_and_norm");

        // When excluded, should use base learning rate (but still include momentum calculation)
        assert_abs_diff_eq!(updated_excluded[0], 0.1 - 0.01 * 0.01, epsilon = 1e-4);

        // When included, should use LARS scaled learning rate
        let weight_norm = (0.1f64.powi(2) + 0.2f64.powi(2)).sqrt();
        let grad_norm = (0.01f64.powi(2) + 0.02f64.powi(2)).sqrt();
        let expected_factor = 0.001 * weight_norm / grad_norm; // trust_coefficient * weight_norm / grad_norm

        assert_abs_diff_eq!(
            updated_included[0],
            0.1 - 0.01 * expected_factor * 0.01,
            epsilon = 1e-5
        );
    }

    /// Regression test for the `exclude_bias_and_norm` tautology.
    ///
    /// The flag used to be evaluated as `!exclude || weight_norm > 0`, which is true
    /// for every parameter with a non-zero norm, so the exclusion never actually
    /// excluded anything. Bias / normalization parameters are rank <= 1 tensors and
    /// must fall back to the plain (unscaled) learning rate.
    #[test]
    fn test_exclude_bias_and_norm_is_decided_by_rank() {
        use scirs2_core::ndarray::Array2;

        // Rank-1 tensor => treated as a bias, excluded from the trust ratio.
        let mut bias_opt = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_trust_coefficient(1.0)
            .with_exclude_bias_and_norm(true);

        let bias = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let bias_grads = Array1::from_vec(vec![0.1f64, 0.2, 0.3]);
        let updated_bias = bias_opt.step(&bias, &bias_grads).expect("bias step");

        // Plain SGD: p - lr * g
        assert_abs_diff_eq!(updated_bias[0], 1.0 - 0.01 * 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(updated_bias[2], 3.0 - 0.01 * 0.3, epsilon = 1e-12);

        // Rank-2 tensor => a weight matrix, LARS scaling applies even when the
        // exclusion flag is enabled.
        let mut weight_opt = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_trust_coefficient(1.0)
            .with_exclude_bias_and_norm(true);

        let weights =
            Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).expect("valid 3x1 matrix");
        let weight_grads =
            Array2::from_shape_vec((3, 1), vec![0.1f64, 0.2, 0.3]).expect("valid 3x1 matrix");
        let updated_weights = weight_opt
            .step(&weights, &weight_grads)
            .expect("weight step");

        let weight_norm = weights.mapv(|x: f64| x * x).sum().sqrt();
        let grad_norm = weight_grads.mapv(|x: f64| x * x).sum().sqrt();
        let scale = weight_norm / (grad_norm + 1e-8);
        assert_abs_diff_eq!(
            updated_weights[[0, 0]],
            1.0 - 0.01 * scale * 0.1,
            epsilon = 1e-8
        );

        // The two paths must genuinely differ.
        assert!((updated_bias[0] - updated_weights[[0, 0]]).abs() > 1e-6);
    }
}
