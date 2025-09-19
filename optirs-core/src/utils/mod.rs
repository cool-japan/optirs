// Utility functions for machine learning optimization
//
// This module provides utility functions and helpers for optimization
// tasks in machine learning.

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand};
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Clip gradient values to a specified range
///
/// # Arguments
///
/// * `gradients` - The gradients to clip
/// * `min_value` - Minimum allowed value
/// * `max_value` - Maximum allowed value
///
/// # Returns
///
/// The clipped gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::utils::clip_gradients;
///
/// let mut gradients = Array1::from_vec(vec![-10.0, 0.5, 8.0, -0.2]);
/// clip_gradients(&mut gradients, -5.0, 5.0);
/// assert_eq!(gradients, Array1::from_vec(vec![-5.0, 0.5, 5.0, -0.2]));
/// ```
#[allow(dead_code)]
pub fn clip_gradients<A, D>(
    gradients: &mut Array<A, D>,
    min_value: A,
    max_value: A,
) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    for grad in gradients.iter_mut() {
        *grad = if *grad < min_value {
            min_value
        } else if *grad > max_value {
            max_value
        } else {
            *grad
        };
    }
    gradients
}

/// Clip gradient norm (global gradient clipping)
///
/// # Arguments
///
/// * `gradients` - The gradients to clip
/// * `max_norm` - Maximum allowed L2 norm
///
/// # Returns
///
/// The clipped gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::utils::clip_gradient_norm;
///
/// let mut gradients = Array1::<f64>::from_vec(vec![3.0, 4.0]); // L2 norm = 5.0
/// clip_gradient_norm(&mut gradients, 1.0f64).unwrap();
/// // After clipping, L2 norm = 1.0
/// let diff0 = (gradients[0] - 0.6f64).abs();
/// let diff1 = (gradients[1] - 0.8f64).abs();
/// assert!(diff0 < 1e-5);
/// assert!(diff1 < 1e-5);
/// ```
#[allow(dead_code)]
pub fn clip_gradient_norm<A, D>(
    gradients: &mut Array<A, D>,
    max_norm: A,
) -> Result<&mut Array<A, D>>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    if max_norm <= A::zero() {
        return Err(OptimError::InvalidConfig(
            "max_norm must be positive".to_string(),
        ));
    }

    // Calculate current L2 _norm
    let _norm = gradients
        .iter()
        .fold(A::zero(), |acc, &x| acc + x * x)
        .sqrt();

    // If _norm exceeds max_norm, scale gradients
    if _norm > max_norm {
        let scale = max_norm / _norm;
        for grad in gradients.iter_mut() {
            *grad = *grad * scale;
        }
    }

    Ok(gradients)
}

/// Compute gradient centralization
///
/// Gradient Centralization is a technique that improves training stability
/// by removing the mean from each gradient tensor.
///
/// # Arguments
///
/// * `gradients` - The gradients to centralize
///
/// # Returns
///
/// The centralized gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::utils::gradient_centralization;
///
/// let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
/// gradient_centralization(&mut gradients);
/// assert_eq!(gradients, Array1::from_vec(vec![-1.0, 0.0, 1.0, 0.0]));
/// ```
#[allow(dead_code)]
pub fn gradient_centralization<A, D>(gradients: &mut Array<A, D>) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    // Calculate mean
    let sum = gradients.iter().fold(A::zero(), |acc, &x| acc + x);
    let mean = sum / A::from(gradients.len()).unwrap_or(A::one());

    // Subtract mean from each element
    for grad in gradients.iter_mut() {
        *grad = *grad - mean;
    }

    gradients
}

/// Zero out small gradient values
///
/// # Arguments
///
/// * `gradients` - The gradients to process
/// * `threshold` - Threshold below which gradients are set to zero
///
/// # Returns
///
/// The processed gradients (in-place modification)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::utils::zero_small_gradients;
///
/// let mut gradients = Array1::from_vec(vec![0.001, 0.02, -0.005, 0.3]);
/// zero_small_gradients(&mut gradients, 0.01);
/// assert_eq!(gradients, Array1::from_vec(vec![0.0, 0.02, 0.0, 0.3]));
/// ```
#[allow(dead_code)]
pub fn zero_small_gradients<A, D>(gradients: &mut Array<A, D>, threshold: A) -> &mut Array<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    let abs_threshold = threshold.abs();

    for grad in gradients.iter_mut() {
        if grad.abs() < abs_threshold {
            *grad = A::zero();
        }
    }

    gradients
}
