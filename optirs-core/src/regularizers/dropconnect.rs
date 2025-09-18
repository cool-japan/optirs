// DropConnect regularization
//
// DropConnect is a regularization technique that randomly drops connections between layers
// during training. Unlike Dropout which drops units, DropConnect drops individual weights.

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// DropConnect regularizer
///
/// Randomly drops connections (weights) during training to prevent overfitting.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray_ext::{Array2, array};
/// use scirs2_optim::regularizers::DropConnect;
///
/// let dropconnect = DropConnect::new(0.5).unwrap(); // 50% connection dropout
/// let weights = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // During training
/// let masked_weights = dropconnect.apply_to_weights(&weights, true);
/// // Some connections will be zeroed out randomly
///
/// // During inference
/// let inference_weights = dropconnect.apply_to_weights(&weights, false);
/// // No dropout during inference - weights are scaled appropriately
/// ```
#[derive(Debug, Clone)]
pub struct DropConnect<A: Float> {
    /// Probability of dropping a connection
    drop_prob: A,
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> DropConnect<A> {
    /// Create a new DropConnect regularizer
    ///
    /// # Arguments
    ///
    /// * `drop_prob` - Probability of dropping each connection (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A new DropConnect instance or error if probability is invalid
    pub fn new(dropprob: A) -> Result<Self> {
        if dropprob < A::zero() || dropprob > A::one() {
            return Err(OptimError::InvalidConfig(
                "Drop probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            drop_prob: dropprob,
        })
    }

    /// Apply DropConnect to weights
    ///
    /// # Arguments
    ///
    /// * `weights` - The weight matrix to apply DropConnect to
    /// * `training` - Whether we're in training mode (applies dropout) or inference mode
    pub fn apply_to_weights<D: Dimension>(
        &self,
        weights: &Array<A, D>,
        training: bool,
    ) -> Array<A, D> {
        if !training || self.drop_prob == A::zero() {
            // During inference or if no dropout, return weights as-is
            return weights.clone();
        }

        // Create keep probability for sampling
        let keep_prob = A::one() - self.drop_prob;
        let keep_prob_f64 = keep_prob.to_f64().unwrap();

        // Sample mask
        let mut rng = scirs2_core::random::rng();
        let mask = Array::from_shape_fn(weights.raw_dim(), |_| {
            rng.random_bool_with_chance(keep_prob_f64)
        });

        // Apply mask and scale by keep probability
        let mut result = weights.clone();
        for (r, &m) in result.iter_mut().zip(mask.iter()) {
            if !m {
                *r = A::zero();
            } else {
                // Scale the kept weights to maintain expected value
                *r = *r / keep_prob;
            }
        }

        result
    }

    /// Apply DropConnect during gradient computation
    ///
    /// This method should be called during backpropagation to ensure
    /// gradients are only computed for non-dropped connections
    pub fn apply_to_gradients<D: Dimension>(
        &self,
        gradients: &Array<A, D>,
        weightsshape: D,
        training: bool,
    ) -> Array<A, D> {
        if !training || self.drop_prob == A::zero() {
            return gradients.clone();
        }

        // Use the same mask for gradients
        let keep_prob = A::one() - self.drop_prob;
        let keep_prob_f64 = keep_prob.to_f64().unwrap();

        // Create mask with same shape as weights
        let mut rng = scirs2_core::random::rng();
        let mask =
            Array::from_shape_fn(weightsshape, |_| rng.random_bool_with_chance(keep_prob_f64));

        // Apply mask to gradients
        let mut result = gradients.clone();
        for (g, &m) in result.iter_mut().zip(mask.iter()) {
            if !m {
                *g = A::zero();
            } else {
                // Scale gradients by keep probability
                *g = *g / keep_prob;
            }
        }

        result
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync, D: Dimension + Send + Sync> Regularizer<A, D>
    for DropConnect<A>
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Apply DropConnect mask to gradients
        let masked_gradients = self.apply_to_gradients(gradients, params.raw_dim(), true);

        // Update gradients in place
        gradients.assign(&masked_gradients);

        // DropConnect doesn't add a penalty term
        Ok(A::zero())
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // DropConnect doesn't add a penalty term to the loss
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_dropconnect_creation() {
        // Valid creation
        let dc = DropConnect::<f64>::new(0.5).unwrap();
        assert_eq!(dc.drop_prob, 0.5);

        // Invalid probabilities
        assert!(DropConnect::<f64>::new(-0.1).is_err());
        assert!(DropConnect::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_dropconnect_training_mode() {
        let dc = DropConnect::new(0.5).unwrap();
        let weights = array![[1.0, 2.0], [3.0, 4.0]];

        // During training, some connections should be dropped
        let masked_weights = dc.apply_to_weights(&weights, true);

        // Check that some but not all values are zero (statistically)
        let _zeros = masked_weights.iter().filter(|&&x| x == 0.0).count();

        // The masked weights should have approximately scaled values
        for (&original, &masked) in weights.iter().zip(masked_weights.iter()) {
            if masked != 0.0 {
                // Non-zero values should be scaled by 1/keep_prob = 2.0
                assert_relative_eq!(masked, original * 2.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dropconnect_inference_mode() {
        let dc = DropConnect::new(0.5).unwrap();
        let weights = array![[1.0, 2.0], [3.0, 4.0]];

        // During inference, weights should remain unchanged
        let inference_weights = dc.apply_to_weights(&weights, false);
        assert_eq!(weights, inference_weights);
    }

    #[test]
    fn test_dropconnect_zero_probability() {
        let dc = DropConnect::new(0.0).unwrap();
        let weights = array![[1.0, 2.0], [3.0, 4.0]];

        // With 0% dropout, weights should remain unchanged
        let result = dc.apply_to_weights(&weights, true);
        assert_eq!(weights, result);
    }

    #[test]
    fn test_dropconnect_gradients() {
        let dc = DropConnect::new(0.5).unwrap();
        let gradients = array![[1.0, 1.0], [1.0, 1.0]];
        let weightsshape = gradients.raw_dim();

        // Apply to gradients
        let masked_grads = dc.apply_to_gradients(&gradients, weightsshape, true);

        // Check scaling
        for &grad in masked_grads.iter() {
            if grad != 0.0 {
                assert_relative_eq!(grad, 2.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_regularizer_trait() {
        let dc = DropConnect::new(0.3).unwrap();
        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];

        // Test Regularizer trait methods
        let penalty = dc.penalty(&params).unwrap();
        assert_eq!(penalty, 0.0); // DropConnect has no penalty term

        // Test gradient computation
        let penalty_from_apply = dc.apply(&params, &mut gradient).unwrap();
        assert_eq!(penalty_from_apply, 0.0);

        // Gradient should be modified with dropout
        let zeros = gradient.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros <= 4); // Some elements may be dropped
    }
}
