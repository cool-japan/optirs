// ElasticNet regularization (L1 + L2)

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::{Regularizer, L1, L2};

/// ElasticNet regularization
///
/// Combines L1 and L2 regularization to get the benefits of both:
/// - L1 encourages sparsity (many parameters are exactly 0)
/// - L2 discourages large weights for more stable solutions
///
/// Penalty: l1_ratio * L1_penalty + (1 - l1_ratio) * L2_penalty
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::regularizers::{ElasticNet, Regularizer};
///
/// // Create an ElasticNet regularizer with strength 0.01 and l1_ratio 0.5
/// // (equal weight to L1 and L2)
/// let regularizer = ElasticNet::new(0.01, 0.5);
///
/// // Parameters and gradients
/// let params = Array1::from_vec(vec![0.5, -0.3, 0.0, 0.2]);
/// let mut gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0]);
///
/// // Apply regularization
/// let penalty = regularizer.apply(&params, &mut gradients).unwrap();
///
/// // Gradients will be modified to include both L1 and L2 penalty gradients
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ElasticNet<A: Float + Debug> {
    /// Total regularization strength
    alpha: A,
    /// Mixing parameter, with 0 <= l1_ratio <= 1
    /// l1_ratio = 1 means only L1 penalty, l1_ratio = 0 means only L2 penalty
    l1_ratio: A,
    /// L1 regularizer
    l1: L1<A>,
    /// L2 regularizer
    l2: L2<A>,
}

impl<A: Float + Debug + Send + Sync> ElasticNet<A> {
    /// Create a new ElasticNet regularizer
    ///
    /// # Arguments
    ///
    /// * `alpha` - Total regularization strength
    /// * `l1_ratio` - Mixing parameter (0 <= l1_ratio <= 1)
    ///   - l1_ratio = 1: only L1 penalty
    ///   - l1_ratio = 0: only L2 penalty
    pub fn new(alpha: A, l1ratio: A) -> Self {
        // Ensure l1_ratio is between 0 and 1
        let l1_ratio = l1ratio.max(A::zero()).min(A::one());

        // Compute individual strengths for L1 and L2
        let l1_alpha = alpha * l1_ratio;
        let l2_alpha = alpha * (A::one() - l1_ratio);

        Self {
            alpha,
            l1_ratio,
            l1: L1::new(l1_alpha),
            l2: L2::new(l2_alpha),
        }
    }

    /// Get the total regularization strength
    pub fn alpha(&self) -> A {
        self.alpha
    }

    /// Get the L1 ratio
    pub fn l1_ratio(&self) -> A {
        self.l1_ratio
    }

    /// Set the regularization parameters
    ///
    /// # Arguments
    ///
    /// * `alpha` - Total regularization strength
    /// * `l1_ratio` - Mixing parameter (0 <= l1_ratio <= 1)
    pub fn set_params(&mut self, alpha: A, l1ratio: A) -> &mut Self {
        // Ensure l1_ratio is between 0 and 1
        let l1_ratio = l1ratio.max(A::zero()).min(A::one());

        self.alpha = alpha;
        self.l1_ratio = l1_ratio;

        // Update individual strengths
        let l1_alpha = alpha * l1_ratio;
        let l2_alpha = alpha * (A::one() - l1_ratio);

        self.l1.set_alpha(l1_alpha);
        self.l2.set_alpha(l2_alpha);

        self
    }
}

impl<A, D> Regularizer<A, D> for ElasticNet<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Create a copy of gradients for L2 regularization
        let mut l2_grads = gradients.clone();

        // Apply L1 regularization
        let l1_penalty = self.l1.apply(params, gradients)?;

        // Apply L2 regularization to the copy
        let l2_penalty = self.l2.apply(params, &mut l2_grads)?;

        // Combine the gradients according to l1_ratio
        Zip::from(gradients)
            .and(&l2_grads)
            .for_each(|grad, &l2_grad| {
                *grad = self.l1_ratio * *grad + (A::one() - self.l1_ratio) * l2_grad;
            });

        // Return combined penalty
        Ok(l1_penalty + l2_penalty)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // Compute L1 and L2 penalties
        let l1_penalty = self.l1.penalty(params)?;
        let l2_penalty = self.l2.penalty(params)?;

        // Return combined penalty
        Ok(l1_penalty + l2_penalty)
    }
}
