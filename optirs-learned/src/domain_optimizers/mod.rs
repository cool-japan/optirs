//! Domain-Specific Optimizers
//!
//! This module provides specialized optimizers tailored for specific deep learning
//! domains: computer vision, natural language processing, and attention mechanisms.
//!
//! Each optimizer incorporates domain knowledge to improve convergence and
//! generalization compared to generic optimizers.

pub mod attention_optimizer;
pub mod cv_optimizer;
pub mod nlp_optimizer;

pub use attention_optimizer::AttentionOptimizer;
pub use cv_optimizer::CVOptimizer;
pub use nlp_optimizer::NLPOptimizer;

use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Information about the current state of an optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerStateInfo<T: Float + Debug + Send + Sync + 'static> {
    /// Total number of optimization steps taken
    pub step_count: usize,
    /// Current effective learning rate
    pub current_lr: T,
    /// Exponential moving average of gradient norms
    pub grad_norm_ema: T,
}

/// Trait for domain-specific advanced optimizers.
///
/// Provides a common interface for optimizers that incorporate domain knowledge
/// (e.g., spatial awareness for CV, layer-wise decay for NLP, head-wise scaling
/// for attention) on top of standard gradient-based updates.
pub trait AdvancedOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Perform one optimization step, returning the updated parameters.
    ///
    /// # Arguments
    /// * `params` - Current parameter values
    /// * `gradients` - Gradient of the loss with respect to `params`
    ///
    /// # Returns
    /// Updated parameter values after applying the optimizer step.
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>>;

    /// Get the current effective learning rate.
    fn get_learning_rate(&self) -> T;

    /// Set the base learning rate.
    fn set_learning_rate(&mut self, lr: T);

    /// Get the optimizer name.
    fn name(&self) -> &str;

    /// Get a snapshot of the optimizer's internal state.
    fn get_state(&self) -> OptimizerStateInfo<T>;
}

/// Compute the L2 norm of an array.
pub(crate) fn l2_norm<T: Float + Debug + Send + Sync + 'static>(arr: &Array1<T>) -> T {
    let sum_sq = arr.iter().fold(T::zero(), |acc, &x| acc + x * x);
    sum_sq.sqrt()
}

/// Clip a gradient array so that its L2 norm does not exceed `max_norm`.
/// Returns a new array (clipped copy) if the norm exceeds the threshold,
/// otherwise returns a clone.
pub(crate) fn clip_grad_norm<T: Float + Debug + Send + Sync + 'static>(
    grad: &Array1<T>,
    max_norm: T,
) -> Array1<T> {
    let norm = l2_norm(grad);
    if norm > max_norm && norm > T::zero() {
        let scale = max_norm / norm;
        grad.mapv(|g| g * scale)
    } else {
        grad.clone()
    }
}
