//! SIMD-accelerated optimizer operations
//!
//! This module provides SIMD-optimized implementations of common optimizer
//! operations using scirs2_core's SimdUnifiedOps infrastructure.
//!
//! The module automatically selects the best SIMD backend available on the
//! target platform (AVX2, SSE, NEON, or scalar fallback).

use scirs2_core::ndarray::{Array, Array1, ArrayView, ArrayView1, Dimension, Ix1};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

/// Trait for SIMD-accelerated optimizer operations
///
/// This trait provides high-performance implementations of common
/// operations found in optimization algorithms.
pub trait SimdOptimizer<T: Float> {
    /// SIMD-accelerated parameter update: params - learning_rate * gradient
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter array
    /// * `gradients` - Gradient array
    /// * `learning_rate` - Learning rate scalar
    ///
    /// # Returns
    ///
    /// Updated parameters
    fn simd_sgd_update(
        params: &ArrayView1<T>,
        gradients: &ArrayView1<T>,
        learning_rate: T,
    ) -> Array1<T>;

    /// SIMD-accelerated momentum update
    ///
    /// velocity = momentum * velocity + learning_rate * gradient
    /// params = params - velocity
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter array
    /// * `gradients` - Gradient array
    /// * `velocity` - Velocity array (momentum state)
    /// * `learning_rate` - Learning rate scalar
    /// * `momentum` - Momentum coefficient
    ///
    /// # Returns
    ///
    /// Tuple of (updated_params, updated_velocity)
    fn simd_momentum_update(
        params: &ArrayView1<T>,
        gradients: &ArrayView1<T>,
        velocity: &ArrayView1<T>,
        learning_rate: T,
        momentum: T,
    ) -> (Array1<T>, Array1<T>);

    /// SIMD-accelerated Adam first moment update
    ///
    /// m = beta1 * m + (1 - beta1) * gradient
    ///
    /// # Arguments
    ///
    /// * `m` - First moment array
    /// * `gradients` - Gradient array
    /// * `beta1` - Exponential decay rate for first moment
    ///
    /// # Returns
    ///
    /// Updated first moment
    fn simd_adam_first_moment(m: &ArrayView1<T>, gradients: &ArrayView1<T>, beta1: T) -> Array1<T>;

    /// SIMD-accelerated Adam second moment update
    ///
    /// v = beta2 * v + (1 - beta2) * gradient^2
    ///
    /// # Arguments
    ///
    /// * `v` - Second moment array
    /// * `gradients` - Gradient array
    /// * `beta2` - Exponential decay rate for second moment
    ///
    /// # Returns
    ///
    /// Updated second moment
    fn simd_adam_second_moment(v: &ArrayView1<T>, gradients: &ArrayView1<T>, beta2: T)
        -> Array1<T>;

    /// SIMD-accelerated Adam parameter update
    ///
    /// params = params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter array
    /// * `m_hat` - Bias-corrected first moment
    /// * `v_hat` - Bias-corrected second moment
    /// * `learning_rate` - Learning rate scalar
    /// * `epsilon` - Small constant for numerical stability
    ///
    /// # Returns
    ///
    /// Updated parameters
    fn simd_adam_update(
        params: &ArrayView1<T>,
        m_hat: &ArrayView1<T>,
        v_hat: &ArrayView1<T>,
        learning_rate: T,
        epsilon: T,
    ) -> Array1<T>;

    /// SIMD-accelerated weight decay application
    ///
    /// gradients = gradients + weight_decay * params
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient array
    /// * `params` - Parameter array
    /// * `weight_decay` - Weight decay coefficient
    ///
    /// # Returns
    ///
    /// Gradients with weight decay applied
    fn simd_weight_decay(
        gradients: &ArrayView1<T>,
        params: &ArrayView1<T>,
        weight_decay: T,
    ) -> Array1<T>;

    /// SIMD-accelerated gradient norm computation
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient array
    ///
    /// # Returns
    ///
    /// L2 norm of gradients
    fn simd_gradient_norm(gradients: &ArrayView1<T>) -> T;
}

/// Implementation of SIMD optimizer operations for f32
impl SimdOptimizer<f32> for f32 {
    fn simd_sgd_update(
        params: &ArrayView1<f32>,
        gradients: &ArrayView1<f32>,
        learning_rate: f32,
    ) -> Array1<f32> {
        // Use SIMD for large arrays, scalar for small ones
        if params.len() >= 16 {
            // SIMD path: params - learning_rate * gradients
            let scaled_grads = f32::simd_scalar_mul(gradients, learning_rate);
            f32::simd_sub(params, &scaled_grads.view())
        } else {
            // Scalar path for small arrays
            params
                .iter()
                .zip(gradients.iter())
                .map(|(&p, &g)| p - learning_rate * g)
                .collect()
        }
    }

    fn simd_momentum_update(
        params: &ArrayView1<f32>,
        gradients: &ArrayView1<f32>,
        velocity: &ArrayView1<f32>,
        learning_rate: f32,
        momentum: f32,
    ) -> (Array1<f32>, Array1<f32>) {
        if params.len() >= 16 {
            // SIMD path
            // velocity = momentum * velocity + learning_rate * gradient
            let scaled_velocity = f32::simd_scalar_mul(velocity, momentum);
            let scaled_gradients = f32::simd_scalar_mul(gradients, learning_rate);
            let new_velocity = f32::simd_add(&scaled_velocity.view(), &scaled_gradients.view());

            // params = params - velocity
            let new_params = f32::simd_sub(params, &new_velocity.view());

            (new_params, new_velocity)
        } else {
            // Scalar path
            let new_velocity: Array1<f32> = velocity
                .iter()
                .zip(gradients.iter())
                .map(|(&v, &g)| momentum * v + learning_rate * g)
                .collect();

            let new_params: Array1<f32> = params
                .iter()
                .zip(new_velocity.iter())
                .map(|(&p, &v)| p - v)
                .collect();

            (new_params, new_velocity)
        }
    }

    fn simd_adam_first_moment(
        m: &ArrayView1<f32>,
        gradients: &ArrayView1<f32>,
        beta1: f32,
    ) -> Array1<f32> {
        if m.len() >= 16 {
            // SIMD path: m = beta1 * m + (1 - beta1) * gradient
            let scaled_m = f32::simd_scalar_mul(m, beta1);
            let scaled_grads = f32::simd_scalar_mul(gradients, 1.0 - beta1);
            f32::simd_add(&scaled_m.view(), &scaled_grads.view())
        } else {
            // Scalar path
            m.iter()
                .zip(gradients.iter())
                .map(|(&m_val, &g)| beta1 * m_val + (1.0 - beta1) * g)
                .collect()
        }
    }

    fn simd_adam_second_moment(
        v: &ArrayView1<f32>,
        gradients: &ArrayView1<f32>,
        beta2: f32,
    ) -> Array1<f32> {
        if v.len() >= 16 {
            // SIMD path: v = beta2 * v + (1 - beta2) * gradient^2
            let scaled_v = f32::simd_scalar_mul(v, beta2);
            let grad_squared = f32::simd_mul(gradients, gradients);
            let scaled_grad_squared = f32::simd_scalar_mul(&grad_squared.view(), 1.0 - beta2);
            f32::simd_add(&scaled_v.view(), &scaled_grad_squared.view())
        } else {
            // Scalar path
            v.iter()
                .zip(gradients.iter())
                .map(|(&v_val, &g)| beta2 * v_val + (1.0 - beta2) * g * g)
                .collect()
        }
    }

    fn simd_adam_update(
        params: &ArrayView1<f32>,
        m_hat: &ArrayView1<f32>,
        v_hat: &ArrayView1<f32>,
        learning_rate: f32,
        epsilon: f32,
    ) -> Array1<f32> {
        if params.len() >= 16 {
            // SIMD path: params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            // Compute sqrt(v_hat) + epsilon
            let v_hat_sqrt: Array1<f32> = v_hat.iter().map(|&v| v.sqrt() + epsilon).collect();

            // Compute m_hat / (sqrt(v_hat) + epsilon)
            let step = f32::simd_div(m_hat, &v_hat_sqrt.view());

            // Scale by learning rate
            let scaled_step = f32::simd_scalar_mul(&step.view(), learning_rate);

            // Update parameters
            f32::simd_sub(params, &scaled_step.view())
        } else {
            // Scalar path
            params
                .iter()
                .zip(m_hat.iter().zip(v_hat.iter()))
                .map(|(&p, (&m, &v))| p - learning_rate * m / (v.sqrt() + epsilon))
                .collect()
        }
    }

    fn simd_weight_decay(
        gradients: &ArrayView1<f32>,
        params: &ArrayView1<f32>,
        weight_decay: f32,
    ) -> Array1<f32> {
        if gradients.len() >= 16 {
            // SIMD path: gradients + weight_decay * params
            let scaled_params = f32::simd_scalar_mul(params, weight_decay);
            f32::simd_add(gradients, &scaled_params.view())
        } else {
            // Scalar path
            gradients
                .iter()
                .zip(params.iter())
                .map(|(&g, &p)| g + weight_decay * p)
                .collect()
        }
    }

    fn simd_gradient_norm(gradients: &ArrayView1<f32>) -> f32 {
        if gradients.len() >= 16 {
            // SIMD path using optimized dot product
            f32::simd_dot(gradients, gradients).sqrt()
        } else {
            // Scalar path
            gradients.iter().map(|&x| x * x).sum::<f32>().sqrt()
        }
    }
}

/// Implementation of SIMD optimizer operations for f64
impl SimdOptimizer<f64> for f64 {
    fn simd_sgd_update(
        params: &ArrayView1<f64>,
        gradients: &ArrayView1<f64>,
        learning_rate: f64,
    ) -> Array1<f64> {
        if params.len() >= 8 {
            // SIMD path
            let scaled_grads = f64::simd_scalar_mul(gradients, learning_rate);
            f64::simd_sub(params, &scaled_grads.view())
        } else {
            // Scalar path
            params
                .iter()
                .zip(gradients.iter())
                .map(|(&p, &g)| p - learning_rate * g)
                .collect()
        }
    }

    fn simd_momentum_update(
        params: &ArrayView1<f64>,
        gradients: &ArrayView1<f64>,
        velocity: &ArrayView1<f64>,
        learning_rate: f64,
        momentum: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        if params.len() >= 8 {
            // SIMD path
            let scaled_velocity = f64::simd_scalar_mul(velocity, momentum);
            let scaled_gradients = f64::simd_scalar_mul(gradients, learning_rate);
            let new_velocity = f64::simd_add(&scaled_velocity.view(), &scaled_gradients.view());
            let new_params = f64::simd_sub(params, &new_velocity.view());
            (new_params, new_velocity)
        } else {
            // Scalar path
            let new_velocity: Array1<f64> = velocity
                .iter()
                .zip(gradients.iter())
                .map(|(&v, &g)| momentum * v + learning_rate * g)
                .collect();
            let new_params: Array1<f64> = params
                .iter()
                .zip(new_velocity.iter())
                .map(|(&p, &v)| p - v)
                .collect();
            (new_params, new_velocity)
        }
    }

    fn simd_adam_first_moment(
        m: &ArrayView1<f64>,
        gradients: &ArrayView1<f64>,
        beta1: f64,
    ) -> Array1<f64> {
        if m.len() >= 8 {
            // SIMD path
            let scaled_m = f64::simd_scalar_mul(m, beta1);
            let scaled_grads = f64::simd_scalar_mul(gradients, 1.0 - beta1);
            f64::simd_add(&scaled_m.view(), &scaled_grads.view())
        } else {
            // Scalar path
            m.iter()
                .zip(gradients.iter())
                .map(|(&m_val, &g)| beta1 * m_val + (1.0 - beta1) * g)
                .collect()
        }
    }

    fn simd_adam_second_moment(
        v: &ArrayView1<f64>,
        gradients: &ArrayView1<f64>,
        beta2: f64,
    ) -> Array1<f64> {
        if v.len() >= 8 {
            // SIMD path
            let scaled_v = f64::simd_scalar_mul(v, beta2);
            let grad_squared = f64::simd_mul(gradients, gradients);
            let scaled_grad_squared = f64::simd_scalar_mul(&grad_squared.view(), 1.0 - beta2);
            f64::simd_add(&scaled_v.view(), &scaled_grad_squared.view())
        } else {
            // Scalar path
            v.iter()
                .zip(gradients.iter())
                .map(|(&v_val, &g)| beta2 * v_val + (1.0 - beta2) * g * g)
                .collect()
        }
    }

    fn simd_adam_update(
        params: &ArrayView1<f64>,
        m_hat: &ArrayView1<f64>,
        v_hat: &ArrayView1<f64>,
        learning_rate: f64,
        epsilon: f64,
    ) -> Array1<f64> {
        if params.len() >= 8 {
            // SIMD path
            let v_hat_sqrt: Array1<f64> = v_hat.iter().map(|&v| v.sqrt() + epsilon).collect();
            let step = f64::simd_div(m_hat, &v_hat_sqrt.view());
            let scaled_step = f64::simd_scalar_mul(&step.view(), learning_rate);
            f64::simd_sub(params, &scaled_step.view())
        } else {
            // Scalar path
            params
                .iter()
                .zip(m_hat.iter().zip(v_hat.iter()))
                .map(|(&p, (&m, &v))| p - learning_rate * m / (v.sqrt() + epsilon))
                .collect()
        }
    }

    fn simd_weight_decay(
        gradients: &ArrayView1<f64>,
        params: &ArrayView1<f64>,
        weight_decay: f64,
    ) -> Array1<f64> {
        if gradients.len() >= 8 {
            // SIMD path
            let scaled_params = f64::simd_scalar_mul(params, weight_decay);
            f64::simd_add(gradients, &scaled_params.view())
        } else {
            // Scalar path
            gradients
                .iter()
                .zip(params.iter())
                .map(|(&g, &p)| g + weight_decay * p)
                .collect()
        }
    }

    fn simd_gradient_norm(gradients: &ArrayView1<f64>) -> f64 {
        if gradients.len() >= 8 {
            // SIMD path
            f64::simd_dot(gradients, gradients).sqrt()
        } else {
            // Scalar path
            gradients.iter().map(|&x| x * x).sum::<f64>().sqrt()
        }
    }
}

/// Helper function to determine if SIMD should be used based on array size
///
/// # Arguments
///
/// * `size` - Size of the array
/// * `dtype_size` - Size of the data type in bytes (4 for f32, 8 for f64)
///
/// # Returns
///
/// True if SIMD should be used, false otherwise
pub fn should_use_simd(size: usize, dtype_size: usize) -> bool {
    // Use SIMD for arrays with at least 16 f32 elements or 8 f64 elements
    let min_simd_size = match dtype_size {
        4 => 16,         // f32
        8 => 8,          // f64
        _ => usize::MAX, // Unknown type, don't use SIMD
    };

    size >= min_simd_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_sgd_update_f32() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let learning_rate = 0.1;

        let result = f32::simd_sgd_update(&params.view(), &gradients.view(), learning_rate);

        assert_relative_eq!(result[0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(result[1], 1.98, epsilon = 1e-6);
        assert_relative_eq!(result[2], 2.97, epsilon = 1e-6);
        assert_relative_eq!(result[3], 3.96, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_sgd_update_f64() {
        let params = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let learning_rate = 0.1;

        let result = f64::simd_sgd_update(&params.view(), &gradients.view(), learning_rate);

        assert_relative_eq!(result[0], 0.99, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.98, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.97, epsilon = 1e-10);
        assert_relative_eq!(result[3], 3.96, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_momentum_update() {
        let params = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let velocity = Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]);
        let learning_rate = 0.1;
        let momentum = 0.9;

        let (new_params, new_velocity) = f32::simd_momentum_update(
            &params.view(),
            &gradients.view(),
            &velocity.view(),
            learning_rate,
            momentum,
        );

        // Check velocity: 0.9 * old_velocity + 0.1 * gradient
        assert_relative_eq!(new_velocity[0], 0.9 * 0.01 + 0.1 * 0.1, epsilon = 1e-6);

        // Check params: old_params - new_velocity
        assert_relative_eq!(new_params[0], 1.0 - new_velocity[0], epsilon = 1e-6);
    }

    #[test]
    fn test_simd_adam_first_moment() {
        let m = Array1::from_vec(vec![0.01f32, 0.02, 0.03, 0.04]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let beta1 = 0.9;

        let result = f32::simd_adam_first_moment(&m.view(), &gradients.view(), beta1);

        assert_relative_eq!(result[0], 0.9 * 0.01 + 0.1 * 0.1, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.9 * 0.02 + 0.1 * 0.2, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_adam_second_moment() {
        let v = Array1::from_vec(vec![0.001f32, 0.002, 0.003, 0.004]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let beta2 = 0.999;

        let result = f32::simd_adam_second_moment(&v.view(), &gradients.view(), beta2);

        assert_relative_eq!(result[0], 0.999 * 0.001 + 0.001 * 0.1 * 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_weight_decay() {
        let gradients = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4]);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let weight_decay = 0.01;

        let result = f32::simd_weight_decay(&gradients.view(), &params.view(), weight_decay);

        assert_relative_eq!(result[0], 0.1 + 0.01 * 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.2 + 0.01 * 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_gradient_norm() {
        let gradients = Array1::from_vec(vec![3.0f32, 4.0]);
        let norm = f32::simd_gradient_norm(&gradients.view());
        assert_relative_eq!(norm, 5.0, epsilon = 1e-6);

        let gradients_f64 = Array1::from_vec(vec![3.0f64, 4.0]);
        let norm_f64 = f64::simd_gradient_norm(&gradients_f64.view());
        assert_relative_eq!(norm_f64, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_should_use_simd() {
        // f32 tests
        assert!(!should_use_simd(8, 4)); // Too small for f32 SIMD
        assert!(should_use_simd(16, 4)); // Exactly at threshold
        assert!(should_use_simd(100, 4)); // Large enough

        // f64 tests
        assert!(!should_use_simd(4, 8)); // Too small for f64 SIMD
        assert!(should_use_simd(8, 8)); // Exactly at threshold
        assert!(should_use_simd(100, 8)); // Large enough
    }

    #[test]
    fn test_simd_large_array() {
        // Test with a large array to ensure SIMD path is taken
        let size = 1000;
        let params: Array1<f32> = Array1::from_vec((0..size).map(|i| i as f32).collect());
        let gradients: Array1<f32> = Array1::from_vec(vec![0.1; size]);
        let learning_rate = 0.01;

        let result = f32::simd_sgd_update(&params.view(), &gradients.view(), learning_rate);

        for i in 0..size {
            assert_relative_eq!(result[i], (i as f32) - learning_rate * 0.1, epsilon = 1e-6);
        }
    }
}
