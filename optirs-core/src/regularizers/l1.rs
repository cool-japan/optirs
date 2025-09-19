// L1 (Lasso) regularization

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array, Dimension, ScalarOperand, Zip};
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// L1 (Lasso) regularization
///
/// Adds a penalty equal to the sum of the absolute values of the parameters,
/// which encourages sparsity (many parameters will be exactly 0).
///
/// Penalty: alpha * sum(abs(params))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array1;
/// use optirs_core::regularizers::{L1, Regularizer};
///
/// // Create an L1 regularizer with strength 0.01
/// let regularizer = L1::new(0.01);
///
/// // Parameters and gradients
/// let params = Array1::from_vec(vec![0.5, -0.3, 0.0, 0.2]);
/// let mut gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0]);
///
/// // Apply regularization
/// let penalty = regularizer.apply(&params, &mut gradients).unwrap();
///
/// // Gradients will be modified to include the L1 penalty gradient
/// // Penalty will be: 0.01 * (|0.5| + |-0.3| + |0.0| + |0.2|) = 0.01 * 1.0 = 0.01
/// ```
#[derive(Debug, Clone, Copy)]
pub struct L1<A: Float + Debug> {
    /// Regularization strength
    alpha: A,
}

impl<A: Float + Debug + Send + Sync> L1<A> {
    /// Create a new L1 regularizer
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

impl<A, D> Regularizer<A, D> for L1<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // L1 gradient: alpha * sign(params)
        Zip::from(params).and(gradients).for_each(|&param, grad| {
            // Sign function: 1 for positive, -1 for negative, 0 for zero
            let sign = if param > A::zero() {
                A::one()
            } else if param < A::zero() {
                -A::one()
            } else {
                A::zero()
            };

            *grad = *grad + self.alpha * sign;
        });

        self.penalty(params)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // L1 penalty: alpha * sum(abs(params))
        let sum_abs = params.iter().fold(A::zero(), |acc, &x| acc + x.abs());
        Ok(self.alpha * sum_abs)
    }
}
