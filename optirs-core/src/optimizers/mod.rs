// Optimization algorithms for machine learning
//
// This module provides various optimization algorithms commonly used in machine learning,
// such as Stochastic Gradient Descent (SGD), Adam, RMSprop, and others.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::{Float, ToPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Fallibly converts any primitive numeric value (an `f64` literal, a `usize`
/// step count, ...) into the optimizer's generic scalar type `A`.
///
/// Centralizes what used to be `A::from(x).expect("unwrap failed")` call
/// sites across the optimizer implementations: instead of panicking, a type
/// that genuinely cannot represent `x` now produces an honest [`OptimError`].
pub(crate) fn cast_scalar<A: Float, T: ToPrimitive>(value: T) -> Result<A> {
    A::from(value).ok_or_else(|| {
        OptimError::InvalidConfig(
            "failed to convert a numeric value to the optimizer's scalar type".to_string(),
        )
    })
}

/// Fallibly converts the optimizer's generic scalar type `A` into `f64`.
///
/// Centralizes what used to be `x.to_f64().expect("unwrap failed")` call
/// sites used for hyperparameter validation.
pub(crate) fn scalar_to_f64<A: Float>(value: A) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::InvalidConfig(
            "failed to convert the optimizer's scalar type to f64".to_string(),
        )
    })
}

/// Trait that defines the interface for optimization algorithms
pub trait Optimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Updates parameters using the given gradients
    ///
    /// # Arguments
    ///
    /// * `params` - The current parameter values
    /// * `gradients` - The gradients of the parameters
    ///
    /// # Returns
    ///
    /// The updated parameters
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>>;

    /// Gets the current learning rate
    fn get_learning_rate(&self) -> A;

    /// Sets a new learning rate
    fn set_learning_rate(&mut self, learning_rate: A);

    /// Updates multiple parameter arrays at once
    ///
    /// # State contract
    ///
    /// Position `i` in `params_list` identifies parameter tensor `i` and **must** get
    /// its own optimizer state (moments, accumulators, velocities and any per-tensor
    /// timestep). The caller is expected to pass the tensors in a stable order across
    /// calls, exactly like PyTorch's parameter groups.
    ///
    /// The default implementation below simply forwards to [`Optimizer::step`], which
    /// is only correct for *stateless* optimizers. Every stateful optimizer in this
    /// crate overrides `step_list` and routes each index to a dedicated state slot
    /// (see e.g. `Adam::step_indexed`). Implementors of new stateful optimizers must
    /// do the same: relying on the default makes all tensors share one state slot, so
    /// they reset each other on every shape change and their bias correction advances
    /// once per tensor instead of once per step.
    ///
    /// # Arguments
    ///
    /// * `params_list` - List of parameter arrays
    /// * `gradients_list` - List of gradient arrays corresponding to the parameters
    ///
    /// # Returns
    ///
    /// Updated parameter arrays
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
        for (params, grads) in params_list.iter().zip(gradients_list.iter()) {
            results.push(self.step(params, grads)?);
        }
        Ok(results)
    }
}

// Import specific optimizers
mod adabound;
mod adadelta;
mod adagrad;
mod adam;
mod adamw;
mod grouped_adam;
mod lamb;
mod lars;
mod lbfgs;
mod lion;
mod lookahead;
mod maml;
mod meta_sgd;
mod ntm_optimizer;
mod radam;
mod ranger;
mod reptile;
mod rmsprop;
mod sam;
mod sgd;
mod sgd_simd;
mod sparse_adam;

// Re-export specific optimizers
pub use adabound::AdaBound;
pub use adadelta::AdaDelta;
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use grouped_adam::GroupedAdam;
pub use lamb::LAMB;
pub use lars::LARS;
pub use lbfgs::LBFGS;
pub use lion::Lion;
pub use lookahead::Lookahead;
pub use maml::{MAMLVariant, TaskBatch, MAML};
pub use meta_sgd::MetaSGD;
pub use ntm_optimizer::{AddressingMode, NtmConfig, NtmOptimizer};
pub use radam::RAdam;
pub use ranger::Ranger;
pub use reptile::ReptileOptimizer;
pub use rmsprop::RMSprop;
pub use sam::SAM;
pub use sgd::SGD;
pub use sgd_simd::SimdSGD;
pub use sparse_adam::{SparseAdam, SparseGradient};
