// L2 (Ridge) regularization

use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// L2 (Ridge) regularization
///
/// Adds a penalty equal to the sum of the squared values of the parameters,
/// which encourages smaller weights and penalizes large weights.
///
/// Penalty: 0.5 * alpha * sum(params^2)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use scirs2_optim::regularizers::{L2, Regularizer};
///
/// // Create an L2 regularizer with strength 0.01
/// let regularizer = L2::new(0.01);
///
/// // Parameters and gradients
/// let params = Array1::from_vec(vec![0.5, -0.3, 0.0, 0.2]);
/// let mut gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0]);
///
/// // Apply regularization
/// let penalty = regularizer.apply(&params, &mut gradients).unwrap();
///
/// // Gradients will be modified to include the L2 penalty gradient
/// // Penalty will be: 0.5 * 0.01 * (0.5^2 + (-0.3)^2 + 0.0^2 + 0.2^2) = 0.005 * 0.38 = 0.0019
/// ```
#[derive(Debug, Clone, Copy)]
pub struct L2<A: Float + Debug> {
    /// Regularization strength
    alpha: A,
}

impl<A: Float + Debug + Send + Sync> L2<A> {
    /// Create a new L2 regularizer
    ///
    /// # Arguments
    ///
    /// * `alpha` - Regularization strength
    pub fn new(alpha: A) -> Self {
        Self { alpha }
    }

    /// Get the regularization strength
    pub fn alpha(&self) -> A {
        self.alpha
    }

    /// Set the regularization strength
    pub fn set_alpha(&mut self, alpha: A) -> &mut Self {
        self.alpha = alpha;
        self
    }
}

impl<A, D> Regularizer<A, D> for L2<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // L2 gradient: alpha * params
        Zip::from(params).and(gradients).for_each(|&param, grad| {
            *grad = *grad + self.alpha * param;
        });

        self.penalty(params)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // L2 penalty: 0.5 * alpha * sum(params^2)
        let half = A::from(0.5).unwrap();
        let sum_squared = params.iter().fold(A::zero(), |acc, &x| acc + x * x);
        Ok(half * self.alpha * sum_squared)
    }
}
