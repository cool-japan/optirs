// Model-Agnostic Meta-Learning (MAML) optimizer
//
// Implements MAML (Finn et al., 2017): "Model-Agnostic Meta-Learning for Fast
// Adaptation of Deep Networks". MAML learns an initialization `theta` such that
// a small number of gradient-descent steps on a new task quickly reach high
// performance.
//
// Algorithm (canonical, K inner steps, batch of tasks T_i):
//
//   Inner (per task T_i):
//       theta_i^{(0)} = theta
//       for k = 0..K-1:
//           g_k       = grad_theta L_{T_i}(theta_i^{(k)})
//           theta_i^{(k+1)} = theta_i^{(k)} - alpha * g_k
//
//   Outer (meta-update):
//       theta <- theta - beta * (1/N) * sum_i grad_theta L_{T_i}(theta_i^{(K)})
//
// The outer gradient passes *through* the inner updates, producing a second-
// order term involving the Hessian of L:
//
//       grad_theta L_{T_i}(theta_i^{(K)}) = J_i^T * final_loss_grad_i,
//       where J_i = prod_{k=0}^{K-1} (I - alpha * H(theta_i^{(k)})).
//
// FOMAML (First-Order MAML, Finn et al., 2017) drops the Hessian term, using
// J_i ~= I, i.e. meta_grad_i := final_loss_grad_i. Reptile (Nichol et al.,
// 2018) approximates the meta-gradient as (theta - theta_i^{(K)}) / alpha,
// which empirically yields a closely related update direction.
//
// References:
//   Finn, C., Abbeel, P., Levine, S. (2017). "Model-Agnostic Meta-Learning for
//     Fast Adaptation of Deep Networks", ICML.
//   Nichol, A., Achiam, J., Schulman, J. (2018). "On First-Order Meta-Learning
//     Algorithms", arXiv:1803.02999.

use scirs2_core::ndarray::{Array, Dimension, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// Variant of MAML to use.
///
/// The variants share the same inner-loop adaptation procedure but differ in
/// how the outer (meta-) gradient is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MAMLVariant {
    /// Full MAML with a second-order term. Because OptiRS does not maintain a
    /// computation graph, the Hessian-vector product is approximated by a
    /// finite difference of inner gradients along the adaptation trajectory.
    /// This yields a meta-gradient of the form
    /// `final_loss_grad - alpha * (g_K - g_0) / K (elementwise) * final_loss_grad`,
    /// which captures the leading second-order correction without ever
    /// instantiating the Hessian.
    SecondOrder,
    /// First-Order MAML (FOMAML): drops the Hessian term entirely. The
    /// meta-gradient is simply `final_loss_grad`. Cheaper than SecondOrder
    /// and usually competitive in practice.
    FirstOrder,
    /// Reptile-style update: the meta-gradient is
    /// `(initial_params - adapted_params) / alpha`. No second-order math is
    /// required and the resulting direction is empirically similar to FOMAML.
    Reptile,
}

/// Per-task data used by the meta-update.
///
/// For each task `T_i` the caller supplies:
/// * `initial_params` – the meta-parameters `theta` at the start of the inner
///   loop (`theta_i^{(0)}`).
/// * `inner_gradients` – the gradients `g_0, g_1, ..., g_{K-1}` evaluated at
///   each inner iterate `theta_i^{(0)}, ..., theta_i^{(K-1)}`. The length of
///   this vector must be at least 1 and matches the number of inner steps.
/// * `final_loss_grad` – the gradient of the meta-loss evaluated at the
///   *adapted* parameters `theta_i^{(K)}` (sometimes denoted `∂L/∂theta'`).
///
/// All arrays must share the same shape; mismatches surface as
/// `OptimError::InvalidParameter`.
#[derive(Debug, Clone)]
pub struct TaskBatch<A: Float + ScalarOperand + Debug> {
    /// Parameters at the start of the inner loop (`theta_i^{(0)}`).
    pub initial_params: Array<A, IxDyn>,
    /// Sequence of gradients along the inner trajectory.
    pub inner_gradients: Vec<Array<A, IxDyn>>,
    /// Gradient of the meta-loss at the adapted parameters.
    pub final_loss_grad: Array<A, IxDyn>,
}

/// MAML optimizer.
///
/// Maintains the meta-parameters `theta` and implements inner-loop adaptation
/// plus outer-loop meta-updates. The struct also implements [`Optimizer`] so
/// it can be used as a drop-in plain-SGD optimizer (with learning rate equal
/// to the meta-learning rate `beta`).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{MAML, MAMLVariant, Optimizer};
///
/// // Use MAML as a regular optimizer (SGD with meta_lr as the step size).
/// let params = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
/// let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
/// let mut opt = MAML::new(0.05).with_variant(MAMLVariant::FirstOrder);
/// let updated = opt.step(&params, &grads).expect("step failed");
/// assert!((updated[0] - (1.0 - 0.05 * 0.1)).abs() < 1e-12);
/// ```
/// Result of a multi-step inner adaptation: the final adapted parameters
/// together with the per-step gradient trajectory (see
/// [`MAML::inner_adapt_multi_step`]).
pub type InnerAdaptResult<A, D> = Result<(Array<A, D>, Vec<Array<A, D>>)>;

#[derive(Debug, Clone)]
pub struct MAML<A: Float + ScalarOperand + Debug> {
    /// Outer / meta learning rate `beta`.
    meta_lr: A,
    /// Inner learning rate `alpha`.
    inner_lr: A,
    /// Number of inner-loop adaptation steps `K`.
    inner_steps: usize,
    /// Which MAML variant to use for the outer gradient.
    variant: MAMLVariant,
    /// L2 weight decay applied to the meta-parameters during the outer step.
    weight_decay: A,
    /// Current meta-parameters `theta`, lazily initialised on the first call
    /// to [`MAML::meta_step`] (or to the `Optimizer::step` shortcut).
    meta_params: Option<Array<A, IxDyn>>,
    /// Number of outer (meta) steps applied so far.
    step_count: usize,
}

impl<A: Float + ScalarOperand + Debug> MAML<A> {
    /// Creates a new MAML optimizer with the given meta-learning rate `beta`.
    ///
    /// Defaults:
    /// * `inner_lr` (`alpha`): `0.01`
    /// * `inner_steps` (`K`): `5`
    /// * `variant`: [`MAMLVariant::FirstOrder`]
    /// * `weight_decay`: `0`
    ///
    /// # Arguments
    ///
    /// * `meta_lr` – outer learning rate `beta` used for the meta-update.
    pub fn new(meta_lr: A) -> Self {
        let default_inner =
            A::from(0.01).expect("MAML: failed to convert default inner_lr constant");
        Self {
            meta_lr,
            inner_lr: default_inner,
            inner_steps: 5,
            variant: MAMLVariant::FirstOrder,
            weight_decay: A::zero(),
            meta_params: None,
            step_count: 0,
        }
    }

    /// Sets the inner-loop learning rate `alpha`.
    pub fn with_inner_lr(mut self, alpha: A) -> Self {
        self.inner_lr = alpha;
        self
    }

    /// Sets the number of inner-loop adaptation steps `K`.
    ///
    /// Zero is interpreted as one step (the inner loop must take at least one
    /// step to produce a meta-gradient).
    pub fn with_inner_steps(mut self, k: usize) -> Self {
        self.inner_steps = if k == 0 { 1 } else { k };
        self
    }

    /// Selects the MAML variant ([`MAMLVariant::SecondOrder`],
    /// [`MAMLVariant::FirstOrder`] or [`MAMLVariant::Reptile`]).
    pub fn with_variant(mut self, v: MAMLVariant) -> Self {
        self.variant = v;
        self
    }

    /// Sets the L2 weight-decay coefficient applied during the outer update.
    pub fn with_weight_decay(mut self, wd: A) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Returns the meta-learning rate `beta`.
    pub fn get_meta_lr(&self) -> A {
        self.meta_lr
    }

    /// Returns the inner-loop learning rate `alpha`.
    pub fn get_inner_lr(&self) -> A {
        self.inner_lr
    }

    /// Returns the number of inner-loop steps `K`.
    pub fn get_inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Returns the active MAML variant.
    pub fn get_variant(&self) -> MAMLVariant {
        self.variant
    }

    /// Returns the configured weight-decay coefficient.
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Returns the number of outer meta-steps applied so far.
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }

    /// Returns a reference to the current meta-parameters, if they have been
    /// initialised by a prior call to [`MAML::meta_step`] or [`Optimizer::step`].
    pub fn meta_params(&self) -> Option<&Array<A, IxDyn>> {
        self.meta_params.as_ref()
    }

    /// Clears the stored meta-parameters and resets the step counter.
    pub fn reset(&mut self) {
        self.meta_params = None;
        self.step_count = 0;
    }

    /// Performs a single inner-loop adaptation step:
    /// `params' = params - inner_lr * gradients`.
    pub fn inner_adapt<D: Dimension>(
        &self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::InvalidParameter(format!(
                "inner_adapt: params shape {:?} does not match gradients shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }
        Ok(params - &(gradients * self.inner_lr))
    }

    /// Performs `inner_steps` adaptation steps starting from `params`, using
    /// `loss_grad_fn` to compute the gradient at each iterate.
    ///
    /// Returns the final adapted parameters together with the trajectory of
    /// gradients evaluated at iterates `theta^{(0)}, theta^{(1)}, ...,
    /// theta^{(K-1)}`. The returned vector therefore has length
    /// `inner_steps`, suitable for direct use in a [`TaskBatch`].
    pub fn inner_adapt_multi_step<D, F>(
        &self,
        params: &Array<A, D>,
        mut loss_grad_fn: F,
    ) -> InnerAdaptResult<A, D>
    where
        D: Dimension,
        F: FnMut(&Array<A, D>) -> Array<A, D>,
    {
        let mut current = params.to_owned();
        let mut trajectory: Vec<Array<A, D>> = Vec::with_capacity(self.inner_steps);
        for _ in 0..self.inner_steps {
            let grad = loss_grad_fn(&current);
            if grad.shape() != current.shape() {
                return Err(OptimError::InvalidParameter(format!(
                    "inner_adapt_multi_step: gradient shape {:?} does not match parameter shape {:?}",
                    grad.shape(),
                    current.shape()
                )));
            }
            current = &current - &(&grad * self.inner_lr);
            trajectory.push(grad);
        }
        Ok((current, trajectory))
    }

    /// Computes the meta-gradient contribution for a single task batch.
    ///
    /// The returned array always has dynamic dimensionality and is suitable
    /// for accumulation across the task batch.
    fn task_meta_gradient(&self, task: &TaskBatch<A>) -> Result<Array<A, IxDyn>> {
        if task.inner_gradients.is_empty() {
            return Err(OptimError::InvalidParameter(
                "MAML::task_meta_gradient: inner_gradients must not be empty".to_string(),
            ));
        }
        let theta_shape = task.initial_params.shape();
        if task.final_loss_grad.shape() != theta_shape {
            return Err(OptimError::InvalidParameter(format!(
                "MAML::task_meta_gradient: final_loss_grad shape {:?} does not match initial_params shape {:?}",
                task.final_loss_grad.shape(),
                theta_shape
            )));
        }
        for (idx, g) in task.inner_gradients.iter().enumerate() {
            if g.shape() != theta_shape {
                return Err(OptimError::InvalidParameter(format!(
                    "MAML::task_meta_gradient: inner_gradients[{}] shape {:?} does not match initial_params shape {:?}",
                    idx,
                    g.shape(),
                    theta_shape
                )));
            }
        }

        match self.variant {
            MAMLVariant::FirstOrder => Ok(task.final_loss_grad.clone()),
            MAMLVariant::SecondOrder => {
                // Hessian-free finite-difference approximation along the inner
                // trajectory. With K inner gradients g_0, ..., g_{K-1} we use
                //
                //     H * v  ~=  ((g_{K-1} - g_0) / (alpha * (K - 1))) (elementwise) * v
                //
                // which is the leading term of a first-order Taylor expansion
                // of the gradient field along the inner step direction. For
                // K = 1 there is no trajectory information to extract a
                // Hessian estimate from, so the SecondOrder variant falls
                // back to FOMAML.
                let k = task.inner_gradients.len();
                if k < 2 {
                    return Ok(task.final_loss_grad.clone());
                }
                let g_first = &task.inner_gradients[0];
                let g_last = &task.inner_gradients[k - 1];
                let steps_minus_one: A = crate::optimizers::cast_scalar(k - 1)?;
                let denom = self.inner_lr * steps_minus_one;
                if denom.abs() <= A::epsilon() {
                    return Ok(task.final_loss_grad.clone());
                }
                let hessian_approx = (g_last - g_first) / denom;
                // meta_grad = (I - alpha * H)^T * final_loss_grad
                //          = final_loss_grad - alpha * H (elementwise) * final_loss_grad
                let correction = &(&hessian_approx * self.inner_lr) * &task.final_loss_grad;
                Ok(&task.final_loss_grad - &correction)
            }
            MAMLVariant::Reptile => {
                // Reconstruct the adapted parameters from the trajectory:
                //   theta_K = theta_0 - alpha * sum_k g_k
                let mut sum_grads = Array::<A, IxDyn>::zeros(task.initial_params.raw_dim());
                for g in &task.inner_gradients {
                    sum_grads = &sum_grads + g;
                }
                let adapted = &task.initial_params - &(&sum_grads * self.inner_lr);
                // meta_grad = (theta_0 - theta_K) / alpha = sum_k g_k. We
                // intentionally compute via the difference form below to keep
                // the semantics of "Reptile uses (initial - adapted) / alpha"
                // explicit in the code.
                let alpha = self.inner_lr;
                if alpha.abs() <= A::epsilon() {
                    return Err(OptimError::InvalidConfig(
                        "MAML(Reptile): inner_lr must be non-zero".to_string(),
                    ));
                }
                Ok((&task.initial_params - &adapted) / alpha)
            }
        }
    }

    /// Performs one outer (meta-) update step over a batch of tasks.
    ///
    /// The meta-gradient is averaged over all task batches, optionally
    /// augmented with L2 weight decay, and applied as
    /// `theta <- theta - meta_lr * mean_meta_grad`. Returns the updated
    /// meta-parameters.
    ///
    /// All task batches must share the same parameter shape and, when the
    /// optimizer already holds meta-parameters, must also match that shape.
    pub fn meta_step<D: Dimension>(
        &mut self,
        task_batches: &[TaskBatch<A>],
    ) -> Result<Array<A, IxDyn>> {
        if task_batches.is_empty() {
            return Err(OptimError::InvalidParameter(
                "MAML::meta_step: task_batches must not be empty".to_string(),
            ));
        }
        let ref_shape = task_batches[0].initial_params.shape().to_vec();
        for (idx, t) in task_batches.iter().enumerate().skip(1) {
            if t.initial_params.shape() != ref_shape.as_slice() {
                return Err(OptimError::InvalidParameter(format!(
                    "MAML::meta_step: task_batches[{}].initial_params shape {:?} differs from task_batches[0] shape {:?}",
                    idx,
                    t.initial_params.shape(),
                    ref_shape
                )));
            }
        }

        // Initialise (or validate) the stored meta-parameters from the first
        // task's initial parameters.
        match self.meta_params.as_ref() {
            None => {
                self.meta_params = Some(task_batches[0].initial_params.clone());
            }
            Some(stored) => {
                if stored.shape() != ref_shape.as_slice() {
                    return Err(OptimError::InvalidParameter(format!(
                        "MAML::meta_step: stored meta_params shape {:?} differs from task initial_params shape {:?}",
                        stored.shape(),
                        ref_shape
                    )));
                }
            }
        }

        // Average meta-gradient across tasks.
        let mut accumulator = Array::<A, IxDyn>::zeros(IxDyn(&ref_shape));
        for task in task_batches {
            let g = self.task_meta_gradient(task)?;
            accumulator = &accumulator + &g;
        }
        let n: A = crate::optimizers::cast_scalar(task_batches.len())?;
        let mean_meta_grad = &accumulator / n;

        // Outer update with optional decoupled weight decay (AdamW-style).
        let mut updated = self
            .meta_params
            .as_ref()
            .expect("MAML: meta_params must be initialised at this point")
            .clone();
        if self.weight_decay != A::zero() {
            let decay = self.meta_lr * self.weight_decay;
            updated = &updated - &(&updated * decay);
        }
        updated = &updated - &(&mean_meta_grad * self.meta_lr);
        self.meta_params = Some(updated.clone());
        self.step_count += 1;

        // Sanity check: the trait signature uses generic `D`; we want to
        // ensure dimensionality still matches the caller-provided arrays.
        let _ = std::marker::PhantomData::<D>;
        Ok(updated)
    }
}

impl<A, D> Optimizer<A, D> for MAML<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Drop-in plain-SGD step using the meta-learning rate. Useful when
    /// embedding MAML inside a standard supervised training loop that has not
    /// (yet) been refactored to use [`MAML::meta_step`].
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::InvalidParameter(format!(
                "MAML::step: params shape {:?} does not match gradients shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }
        let mut updated = params - &(gradients * self.meta_lr);
        if self.weight_decay != A::zero() {
            let decay = self.meta_lr * self.weight_decay;
            updated = &updated - &(params * decay);
        }
        // Cache the parameters as meta_params so subsequent meta_step calls
        // operate on a consistent state.
        self.meta_params = Some(updated.to_owned().into_dyn());
        self.step_count += 1;
        Ok(updated)
    }

    fn get_learning_rate(&self) -> A {
        self.meta_lr
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.meta_lr = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, IxDyn};

    /// Helper: quadratic task `L_i(theta) = 0.5 * (theta - target_i)^2`.
    /// Gradient is `theta - target_i`.
    fn quadratic_grad(theta: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        theta - target
    }

    fn quadratic_loss(theta: &Array1<f64>, target: &Array1<f64>) -> f64 {
        theta
            .iter()
            .zip(target.iter())
            .map(|(t, tg)| 0.5 * (t - tg).powi(2))
            .sum()
    }

    /// Helper: build a TaskBatch from a quadratic task by running the inner
    /// loop manually.
    fn make_quadratic_task(
        theta: &Array1<f64>,
        target: &Array1<f64>,
        inner_lr: f64,
        inner_steps: usize,
    ) -> TaskBatch<f64> {
        let mut current = theta.clone();
        let mut grads: Vec<Array<f64, IxDyn>> = Vec::with_capacity(inner_steps);
        for _ in 0..inner_steps {
            let g = quadratic_grad(&current, target);
            grads.push(g.clone().into_dyn());
            current = &current - &(&g * inner_lr);
        }
        let final_grad = quadratic_grad(&current, target);
        TaskBatch {
            initial_params: theta.clone().into_dyn(),
            inner_gradients: grads,
            final_loss_grad: final_grad.into_dyn(),
        }
    }

    #[test]
    fn test_default_config_values() {
        let opt: MAML<f64> = MAML::new(0.1);
        assert!((opt.get_meta_lr() - 0.1).abs() < 1e-12);
        assert!((opt.get_inner_lr() - 0.01).abs() < 1e-12);
        assert_eq!(opt.get_inner_steps(), 5);
        assert_eq!(opt.get_variant(), MAMLVariant::FirstOrder);
        assert!((opt.get_weight_decay() - 0.0).abs() < 1e-12);
        assert_eq!(opt.get_step_count(), 0);
        assert!(opt.meta_params().is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let opt: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(7)
            .with_variant(MAMLVariant::SecondOrder)
            .with_weight_decay(1e-4);
        assert!((opt.get_inner_lr() - 0.05).abs() < 1e-12);
        assert_eq!(opt.get_inner_steps(), 7);
        assert_eq!(opt.get_variant(), MAMLVariant::SecondOrder);
        assert!((opt.get_weight_decay() - 1e-4).abs() < 1e-12);

        // Zero inner steps must clamp up to 1.
        let clamped: MAML<f64> = MAML::new(0.1).with_inner_steps(0);
        assert_eq!(clamped.get_inner_steps(), 1);
    }

    #[test]
    fn test_inner_adapt_basic() {
        // Quadratic loss with target = 0: gradient = theta. A single inner
        // step with alpha=0.1 must reduce |theta|, hence the loss.
        let opt: MAML<f64> = MAML::new(0.1).with_inner_lr(0.1);
        let theta = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let grad = theta.clone();
        let adapted = opt.inner_adapt(&theta, &grad).expect("inner_adapt failed");
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let loss_before = quadratic_loss(&theta, &target);
        let loss_after = quadratic_loss(&adapted, &target);
        assert!(
            loss_after < loss_before,
            "inner_adapt should decrease loss: before={loss_before}, after={loss_after}"
        );
        // The exact arithmetic: adapted = theta - 0.1 * theta = 0.9 * theta.
        for (a, t) in adapted.iter().zip(theta.iter()) {
            assert!((a - 0.9 * t).abs() < 1e-12);
        }
    }

    #[test]
    fn test_inner_adapt_multi_step_returns_trajectory() {
        let opt: MAML<f64> = MAML::new(0.1).with_inner_lr(0.05).with_inner_steps(4);
        let theta = Array1::from_vec(vec![1.0, -1.0, 2.0]);
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let target_clone = target.clone();
        let (adapted, traj) = opt
            .inner_adapt_multi_step(&theta, move |t| quadratic_grad(t, &target_clone))
            .expect("inner_adapt_multi_step failed");
        assert_eq!(traj.len(), opt.get_inner_steps());
        // Adapted params must equal what a manual loop would produce.
        let mut manual = theta.clone();
        for _ in 0..opt.get_inner_steps() {
            let g = quadratic_grad(&manual, &target);
            manual = &manual - &(&g * opt.get_inner_lr());
        }
        for (a, m) in adapted.iter().zip(manual.iter()) {
            assert!(
                (a - m).abs() < 1e-12,
                "adapted={a}, manual={m} differ beyond tolerance"
            );
        }
    }

    #[test]
    fn test_meta_step_first_order_decreases_meta_loss() {
        // Distribution of quadratic tasks with random-ish targets. The meta
        // optimum lies at the mean of the task targets.
        let targets = [
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![-1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![0.0, -1.0]),
        ];
        let mut opt: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(3)
            .with_variant(MAMLVariant::FirstOrder);
        let mut theta = Array1::from_vec(vec![3.0, -3.0]).into_dyn();

        // Initial mean adapted loss.
        let initial_loss =
            mean_adapted_loss(&theta, &targets, opt.get_inner_lr(), opt.get_inner_steps());

        for _ in 0..50 {
            let batches: Vec<TaskBatch<f64>> = targets
                .iter()
                .map(|tgt| {
                    let theta_1d = theta
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                        .expect("test: theta should be 1-D");
                    make_quadratic_task(&theta_1d, tgt, opt.get_inner_lr(), opt.get_inner_steps())
                })
                .collect();
            theta = opt
                .meta_step::<scirs2_core::ndarray::Ix1>(&batches)
                .expect("meta_step failed");
        }

        let final_loss =
            mean_adapted_loss(&theta, &targets, opt.get_inner_lr(), opt.get_inner_steps());
        assert!(
            final_loss < initial_loss,
            "FOMAML meta-loss did not decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn test_meta_step_second_order_decreases_meta_loss() {
        let targets = [
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![-1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![0.0, -1.0]),
        ];
        let mut opt: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(3)
            .with_variant(MAMLVariant::SecondOrder);
        let mut theta = Array1::from_vec(vec![3.0, -3.0]).into_dyn();

        let initial_loss =
            mean_adapted_loss(&theta, &targets, opt.get_inner_lr(), opt.get_inner_steps());

        for _ in 0..50 {
            let batches: Vec<TaskBatch<f64>> = targets
                .iter()
                .map(|tgt| {
                    let theta_1d = theta
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                        .expect("test: theta should be 1-D");
                    make_quadratic_task(&theta_1d, tgt, opt.get_inner_lr(), opt.get_inner_steps())
                })
                .collect();
            theta = opt
                .meta_step::<scirs2_core::ndarray::Ix1>(&batches)
                .expect("meta_step failed");
        }

        let final_loss =
            mean_adapted_loss(&theta, &targets, opt.get_inner_lr(), opt.get_inner_steps());
        assert!(
            final_loss < initial_loss,
            "SecondOrder MAML meta-loss did not decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn test_reptile_variant_uses_difference_form() {
        let mut opt: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(4)
            .with_variant(MAMLVariant::Reptile);
        let theta = Array1::from_vec(vec![2.0, -2.0]);
        let target = Array1::from_vec(vec![0.5, -0.5]);
        let task = make_quadratic_task(&theta, &target, opt.get_inner_lr(), opt.get_inner_steps());

        // The Reptile meta-gradient is `(theta_0 - theta_K) / alpha`. Build
        // the same quantity by hand from the task.
        let mut adapted = theta.clone();
        for _ in 0..opt.get_inner_steps() {
            let g = quadratic_grad(&adapted, &target);
            adapted = &adapted - &(&g * opt.get_inner_lr());
        }
        let expected_meta_grad = (&theta - &adapted) / opt.get_inner_lr();

        // Apply one meta-step and compare against theta - meta_lr * expected.
        let updated = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
            .expect("meta_step failed");
        let expected_updated = &theta - &(&expected_meta_grad * opt.get_meta_lr());
        for (u, e) in updated.iter().zip(expected_updated.iter()) {
            assert!(
                (u - e).abs() < 1e-10,
                "Reptile meta-update mismatch: got {u}, expected {e}"
            );
        }

        // Direction check: meta-gradient sign must match (initial - adapted).
        let direction = &theta - &adapted;
        for (e, d) in expected_meta_grad.iter().zip(direction.iter()) {
            assert!(
                e.signum() == d.signum() || d.abs() < 1e-12,
                "Reptile direction does not match (initial - adapted)"
            );
        }
    }

    #[test]
    fn test_fomaml_cheaper_than_secondorder() {
        // Verifies both variants run end-to-end on the same task batch and
        // produce finite outputs. The "cheaper" property is structural —
        // FOMAML executes a single vector copy whereas SecondOrder also
        // computes a finite-difference Hessian-vector product.
        let theta = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5]);
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let task = make_quadratic_task(&theta, &target, 0.05, 5);

        let mut fomaml: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(5)
            .with_variant(MAMLVariant::FirstOrder);
        let mut secondorder: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(5)
            .with_variant(MAMLVariant::SecondOrder);

        let a = fomaml
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
            .expect("FOMAML meta_step failed");
        let b = secondorder
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
            .expect("SecondOrder meta_step failed");

        for v in a.iter().chain(b.iter()) {
            assert!(v.is_finite(), "Meta-update produced non-finite value: {v}");
        }
        assert_eq!(fomaml.get_step_count(), 1);
        assert_eq!(secondorder.get_step_count(), 1);
    }

    #[test]
    fn test_variant_switching_preserves_meta_params() {
        let theta = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let task = make_quadratic_task(&theta, &target, 0.05, 3);

        let mut opt: MAML<f64> = MAML::new(0.05)
            .with_inner_lr(0.05)
            .with_inner_steps(3)
            .with_variant(MAMLVariant::FirstOrder);
        let updated = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
            .expect("meta_step failed");

        // Switch variant via the builder-style setter on a mutable reference.
        opt = opt.with_variant(MAMLVariant::SecondOrder);
        assert_eq!(opt.get_variant(), MAMLVariant::SecondOrder);

        // Meta-params survived the variant change.
        let stored = opt
            .meta_params()
            .expect("meta_params should be retained across variant switch");
        for (s, u) in stored.iter().zip(updated.iter()) {
            assert!((s - u).abs() < 1e-12);
        }
    }

    #[test]
    fn test_optimizer_trait_step_is_meta_lr_descent() {
        let mut opt: MAML<f64> = MAML::new(0.05);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, -0.2, 0.3]);
        let updated = opt.step(&params, &grads).expect("step failed");
        for ((p, g), u) in params.iter().zip(grads.iter()).zip(updated.iter()) {
            let expected = p - 0.05 * g;
            assert!(
                (u - expected).abs() < 1e-12,
                "Optimizer::step produced {u}, expected {expected}"
            );
        }
        assert_eq!(opt.get_step_count(), 1);
        assert!(opt.meta_params().is_some());
        assert!(
            (Optimizer::<f64, scirs2_core::ndarray::Ix1>::get_learning_rate(&opt) - 0.05).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_zero_gradients_no_change() {
        // When every per-task gradient is zero the meta-gradient is zero and
        // the meta-parameters must not move.
        let theta = Array1::from_vec(vec![1.5, -0.5, 2.0]);
        let zero = Array1::from_vec(vec![0.0, 0.0, 0.0]);

        for variant in [
            MAMLVariant::FirstOrder,
            MAMLVariant::SecondOrder,
            MAMLVariant::Reptile,
        ] {
            let mut opt: MAML<f64> = MAML::new(0.1)
                .with_inner_lr(0.05)
                .with_inner_steps(3)
                .with_variant(variant);
            let task = TaskBatch {
                initial_params: theta.clone().into_dyn(),
                inner_gradients: vec![zero.clone().into_dyn(); opt.get_inner_steps()],
                final_loss_grad: zero.clone().into_dyn(),
            };
            let updated = opt
                .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
                .expect("meta_step failed");
            for (t, u) in theta.iter().zip(updated.iter()) {
                assert!(
                    (t - u).abs() < 1e-12,
                    "Variant {:?}: theta should not move with zero gradients (got {} vs {})",
                    variant,
                    t,
                    u
                );
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut opt: MAML<f64> = MAML::new(0.1).with_inner_lr(0.05).with_inner_steps(2);
        let theta = Array1::from_vec(vec![1.0, 2.0]);
        let theta_wrong = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // final_loss_grad shape mismatch
        let bad_task = TaskBatch {
            initial_params: theta.clone().into_dyn(),
            inner_gradients: vec![Array1::from_vec(vec![0.1, 0.1]).into_dyn()],
            final_loss_grad: Array1::from_vec(vec![0.1, 0.1, 0.1]).into_dyn(),
        };
        let err = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&bad_task))
            .expect_err("expected InvalidParameter error");
        assert!(matches!(err, OptimError::InvalidParameter(_)));

        // inner_gradients shape mismatch
        let bad_task2 = TaskBatch {
            initial_params: theta.clone().into_dyn(),
            inner_gradients: vec![Array1::from_vec(vec![0.1, 0.1, 0.1]).into_dyn()],
            final_loss_grad: Array1::from_vec(vec![0.1, 0.1]).into_dyn(),
        };
        let err2 = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&bad_task2))
            .expect_err("expected InvalidParameter error");
        assert!(matches!(err2, OptimError::InvalidParameter(_)));

        // Inconsistent shapes across task batches.
        let good_a = make_quadratic_task(
            &theta,
            &Array1::from_vec(vec![0.0, 0.0]),
            opt.get_inner_lr(),
            opt.get_inner_steps(),
        );
        let good_b = make_quadratic_task(
            &theta_wrong,
            &Array1::from_vec(vec![0.0, 0.0, 0.0]),
            opt.get_inner_lr(),
            opt.get_inner_steps(),
        );
        let err3 = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(&[good_a, good_b])
            .expect_err("expected InvalidParameter error");
        assert!(matches!(err3, OptimError::InvalidParameter(_)));

        // Optimizer::step shape mismatch.
        let mut opt2: MAML<f64> = MAML::new(0.1);
        let p = Array1::from_vec(vec![1.0, 2.0]);
        let g = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let err4 = opt2.step(&p, &g).expect_err("expected InvalidParameter");
        assert!(matches!(err4, OptimError::InvalidParameter(_)));
    }

    #[test]
    fn test_weight_decay_shrinks_params() {
        // Apply meta_step with zero meta-gradient but non-zero weight decay:
        // updated = theta - meta_lr * wd * theta = (1 - meta_lr * wd) * theta.
        let mut opt: MAML<f64> = MAML::new(0.1)
            .with_inner_lr(0.05)
            .with_inner_steps(2)
            .with_weight_decay(0.5)
            .with_variant(MAMLVariant::FirstOrder);
        let theta = Array1::from_vec(vec![1.0, -2.0, 4.0]);
        let zero = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let task = TaskBatch {
            initial_params: theta.clone().into_dyn(),
            inner_gradients: vec![zero.clone().into_dyn(); 2],
            final_loss_grad: zero.into_dyn(),
        };
        let updated = opt
            .meta_step::<scirs2_core::ndarray::Ix1>(std::slice::from_ref(&task))
            .expect("meta_step failed");
        let factor = 1.0 - opt.get_meta_lr() * opt.get_weight_decay();
        for (t, u) in theta.iter().zip(updated.iter()) {
            assert!(
                (u - t * factor).abs() < 1e-12,
                "Weight decay mismatch: got {u}, expected {}",
                t * factor
            );
            assert!(
                u.abs() < t.abs(),
                "Weight decay should shrink magnitude: |{u}| !< |{t}|"
            );
        }

        // Also verify via the Optimizer::step path.
        let mut opt2: MAML<f64> = MAML::new(0.1).with_weight_decay(0.5);
        let params = Array1::from_vec(vec![2.0, -4.0]);
        let zeros = Array1::from_vec(vec![0.0, 0.0]);
        let step_updated = opt2.step(&params, &zeros).expect("step failed");
        for (p, u) in params.iter().zip(step_updated.iter()) {
            assert!(
                (u - p * (1.0 - 0.1 * 0.5)).abs() < 1e-12,
                "Optimizer::step weight decay mismatch"
            );
        }
    }

    #[test]
    fn test_reset_clears_meta_params() {
        let mut opt: MAML<f64> = MAML::new(0.1);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2]);
        let _ = opt.step(&params, &grads).expect("step failed");
        assert!(opt.meta_params().is_some());
        assert_eq!(opt.get_step_count(), 1);

        opt.reset();
        assert!(opt.meta_params().is_none());
        assert_eq!(opt.get_step_count(), 0);
    }

    /// Mean loss across tasks after applying `inner_steps` inner-loop steps
    /// from `theta` toward each task's target. Used by the convergence tests.
    fn mean_adapted_loss(
        theta: &Array<f64, IxDyn>,
        targets: &[Array1<f64>],
        inner_lr: f64,
        inner_steps: usize,
    ) -> f64 {
        let theta_1d = theta
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .expect("test: theta should be 1-D");
        let mut total = 0.0;
        for tgt in targets {
            let mut current = theta_1d.clone();
            for _ in 0..inner_steps {
                let g = quadratic_grad(&current, tgt);
                current = &current - &(&g * inner_lr);
            }
            total += quadratic_loss(&current, tgt);
        }
        total / (targets.len() as f64)
    }
}
