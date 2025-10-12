//! SIMD-accelerated SGD optimizer
//!
//! This module provides a SIMD-optimized implementation of Stochastic Gradient Descent
//! for 1D parameter arrays using scirs2_core's SimdUnifiedOps.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;
use crate::simd_optimizer::SimdOptimizer;

/// SIMD-accelerated Stochastic Gradient Descent optimizer
///
/// This is a specialized version of SGD optimized for 1D arrays using SIMD operations.
/// For maximum performance, use this when working with flattened parameter vectors.
///
/// Formula:
/// v_t = momentum * v_{t-1} + learning_rate * (gradient + weight_decay * param)
/// param_t = param_{t-1} - v_t
///
/// # Performance
///
/// This implementation uses SIMD instructions (AVX2/SSE/NEON) for:
/// - Parameter updates
/// - Momentum computation
/// - Weight decay application
///
/// Expected speedup: 2-4x over scalar implementation for large parameter arrays
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{SimdSGD, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(1000);
/// let gradients = Array1::from_elem(1000, 0.1);
///
/// // Create SIMD-accelerated SGD optimizer
/// let mut optimizer = SimdSGD::new(0.01);
/// optimizer.set_momentum(0.9);
///
/// // Update parameters with SIMD acceleration
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SimdSGD<A: Float> {
    /// Learning rate
    learning_rate: A,
    /// Momentum factor (0.0 means no momentum)
    momentum: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Velocity (momentum state)
    velocity: Option<Array1<A>>,
}

impl<A: Float> SimdSGD<A> {
    /// Creates a new SIMD-accelerated SGD optimizer
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

    /// Creates a new SIMD SGD optimizer with full configuration
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
    pub fn set_momentum(&mut self, momentum: A) -> &mut Self {
        self.momentum = momentum;
        self
    }

    /// Builder method to set momentum and return self
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
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to set weight decay and return self
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the current weight decay factor
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Resets the optimizer state
    pub fn reset(&mut self) {
        self.velocity = None;
    }
}

// Specialized SIMD implementation for f32
impl Optimizer<f32, scirs2_core::ndarray::Ix1> for SimdSGD<f32> {
    fn step(&mut self, params: &Array1<f32>, gradients: &Array1<f32>) -> Result<Array1<f32>> {
        // Validate shapes
        if params.shape() != gradients.shape() {
            return Err(crate::error::OptimError::DimensionMismatch(format!(
                "Incompatible shapes: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let params_view = params.view();
        let gradients_view = gradients.view();

        // Apply weight decay if needed
        let adjusted_gradients = if self.weight_decay > 0.0 {
            f32::simd_weight_decay(&gradients_view, &params_view, self.weight_decay)
        } else {
            gradients.to_owned()
        };

        // Initialize velocity if this is the first step
        if self.velocity.is_none() {
            self.velocity = Some(Array1::zeros(params.len()));
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Ensure velocity has correct dimensions
        if velocity.len() != params.len() {
            *velocity = Array1::zeros(params.len());
        }

        // Compute update using SIMD operations
        let new_params = if self.momentum > 0.0 {
            // SIMD-accelerated momentum update
            let (updated_params, updated_velocity) = f32::simd_momentum_update(
                &params_view,
                &adjusted_gradients.view(),
                &velocity.view(),
                self.learning_rate,
                self.momentum,
            );
            *velocity = updated_velocity;
            updated_params
        } else {
            // SIMD-accelerated vanilla SGD
            f32::simd_sgd_update(&params_view, &adjusted_gradients.view(), self.learning_rate)
        };

        Ok(new_params)
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

// Specialized SIMD implementation for f64
impl Optimizer<f64, scirs2_core::ndarray::Ix1> for SimdSGD<f64> {
    fn step(&mut self, params: &Array1<f64>, gradients: &Array1<f64>) -> Result<Array1<f64>> {
        // Validate shapes
        if params.shape() != gradients.shape() {
            return Err(crate::error::OptimError::DimensionMismatch(format!(
                "Incompatible shapes: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let params_view = params.view();
        let gradients_view = gradients.view();

        // Apply weight decay if needed
        let adjusted_gradients = if self.weight_decay > 0.0 {
            f64::simd_weight_decay(&gradients_view, &params_view, self.weight_decay)
        } else {
            gradients.to_owned()
        };

        // Initialize velocity if this is the first step
        if self.velocity.is_none() {
            self.velocity = Some(Array1::zeros(params.len()));
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Ensure velocity has correct dimensions
        if velocity.len() != params.len() {
            *velocity = Array1::zeros(params.len());
        }

        // Compute update using SIMD operations
        let new_params = if self.momentum > 0.0 {
            // SIMD-accelerated momentum update
            let (updated_params, updated_velocity) = f64::simd_momentum_update(
                &params_view,
                &adjusted_gradients.view(),
                &velocity.view(),
                self.learning_rate,
                self.momentum,
            );
            *velocity = updated_velocity;
            updated_params
        } else {
            // SIMD-accelerated vanilla SGD
            f64::simd_sgd_update(&params_view, &adjusted_gradients.view(), self.learning_rate)
        };

        Ok(new_params)
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_sgd_basic() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let mut optimizer = SimdSGD::new(0.1);
        let result = optimizer.step(&params, &gradients).unwrap();

        assert_relative_eq!(result[0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(result[1], 1.98, epsilon = 1e-6);
        assert_relative_eq!(result[2], 2.97, epsilon = 1e-6);
        assert_relative_eq!(result[3], 3.96, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_sgd_momentum() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let mut optimizer = SimdSGD::new_with_config(0.1, 0.9, 0.0);

        // First step
        let result1 = optimizer.step(&params, &gradients).unwrap();

        // Second step - should show momentum effect
        let result2 = optimizer.step(&result1, &gradients).unwrap();

        // With momentum, the second step should move further
        assert!(result2[0] < result1[0]);
    }

    #[test]
    fn test_simd_sgd_weight_decay() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let mut optimizer = SimdSGD::new_with_config(0.1, 0.0, 0.01);
        let result = optimizer.step(&params, &gradients).unwrap();

        // Weight decay should reduce parameters more than vanilla SGD
        let expected_grad = 0.1 + 0.01 * 1.0;
        assert_relative_eq!(result[0], 1.0 - 0.1 * expected_grad, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_sgd_large_array() {
        // Test with large array to ensure SIMD path is taken
        let size = 1000;
        let params: Array1<f32> = Array1::from_vec((0..size).map(|i| i as f32).collect());
        let gradients: Array1<f32> = Array1::from_elem(size, 0.1);

        let mut optimizer = SimdSGD::new(0.01);
        let result = optimizer.step(&params, &gradients).unwrap();

        for i in 0..size {
            assert_relative_eq!(result[i], (i as f32) - 0.01 * 0.1, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_sgd_f64() {
        let params = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let mut optimizer = SimdSGD::new(0.1);
        let result = optimizer.step(&params, &gradients).unwrap();

        assert_relative_eq!(result[0], 0.99, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.98, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.97, epsilon = 1e-10);
        assert_relative_eq!(result[3], 3.96, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sgd_reset() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

        let mut optimizer = SimdSGD::new_with_config(0.1, 0.9, 0.0);

        // Take a step to initialize velocity
        let _ = optimizer.step(&params, &gradients).unwrap();
        assert!(optimizer.velocity.is_some());

        // Reset should clear velocity
        optimizer.reset();
        assert!(optimizer.velocity.is_none());
    }
}
