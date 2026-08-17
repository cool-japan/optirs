// L-BFGS optimizer implementation
//
// Based on the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm.

use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// L-BFGS optimizer
///
/// Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm.
/// This is a quasi-Newton method that approximates the Hessian inverse using a limited
/// amount of memory by storing only a few vectors from previous iterations.
///
/// # Curvature pairs
///
/// The optimizer stores the previous parameters *and* the previous gradient, so the
/// curvature pair is the true `s = x_k - x_{k-1}`, `y = g_k - g_{k-1}` even when the
/// caller post-processes the returned parameters (projection, clipping, weight decay,
/// a different step size, ...). Pairs with `y·s <= 0` are skipped so the implicit
/// inverse-Hessian stays positive definite.
///
/// # Step size and line search
///
/// [`Optimizer::step`] applies a **fixed** step size (the learning rate) along the
/// two-loop direction: it has no access to the objective, so it cannot run a line
/// search. Use [`LBFGS::step_with_loss`] to get a backtracking Armijo line search that
/// uses the configured `c1` and `max_ls` parameters.
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{LBFGS, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create an L-BFGS optimizer
/// let mut optimizer = LBFGS::new(1.0);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).expect("optimizer.step succeeds");
/// ```
#[derive(Debug, Clone)]
pub struct LBFGS<A: Float + ScalarOperand + Debug> {
    /// Learning rate (also the initial trial step size of the line search)
    learning_rate: A,
    /// History size (number of vectors to store)
    history_size: usize,
    /// Tolerance for gradient norm
    tolerance_grad: A,
    /// Armijo (sufficient-decrease) line search parameter c1
    c1: A,
    /// Wolfe curvature line search parameter c2
    ///
    /// Retained for configuration compatibility and for callers that inspect it via
    /// [`LBFGS::c2`]. The backtracking Armijo search in [`LBFGS::step_with_loss`] only
    /// shrinks the trial step, so it cannot enforce the curvature condition; doing so
    /// requires gradient evaluations at trial points, which this API does not have.
    c2: A,
    /// Maximum number of line search iterations
    max_ls: usize,
    /// Backtracking contraction factor applied to the trial step size
    ls_contraction: A,
    /// History of gradient differences (y = grad_new - grad_old)
    old_dirs: VecDeque<Array1<A>>,
    /// History of step vectors (s = params_new - params_old)
    old_stps: VecDeque<Array1<A>>,
    /// History of 1/(y·s) values
    ro: VecDeque<A>,
    /// Previous parameters (flattened), used to form the true `s = x_k - x_{k-1}`
    prev_params: Option<Array1<A>>,
    /// Previous gradient (flattened), used to form `y = g_k - g_{k-1}`
    prev_grad: Option<Array1<A>>,
    /// Initial Hessian diagonal value
    h_diag: A,
    /// Step counter
    n_iter: usize,
    /// Temporary alpha values for two-loop recursion
    alpha: Vec<A>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> LBFGS<A> {
    /// Creates a new L-BFGS optimizer with the given learning rate
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self::new_with_config(
            learning_rate,
            100,                                      // history_size
            A::from(1e-7).unwrap_or_else(A::epsilon), // tolerance_grad
            A::from(1e-4).unwrap_or_else(A::epsilon), // c1
            A::from(0.9).unwrap_or_else(|| A::one()), // c2
            25,                                       // max_ls
        )
    }

    /// Creates a new L-BFGS optimizer with full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `history_size` - Number of past gradients/steps to store (default: 100)
    /// * `tolerance_grad` - Gradient norm tolerance for convergence (default: 1e-7)
    /// * `c1` - Wolfe line search parameter for Armijo condition (default: 1e-4)
    /// * `c2` - Wolfe line search parameter for curvature condition (default: 0.9)
    /// * `max_ls` - Maximum line search iterations (default: 25)
    pub fn new_with_config(
        learning_rate: A,
        history_size: usize,
        tolerance_grad: A,
        c1: A,
        c2: A,
        max_ls: usize,
    ) -> Self {
        let ls_contraction = A::from(0.5).unwrap_or_else(|| A::one() / (A::one() + A::one()));
        Self {
            learning_rate,
            history_size,
            tolerance_grad,
            c1,
            c2,
            max_ls,
            ls_contraction,
            old_dirs: VecDeque::with_capacity(history_size),
            old_stps: VecDeque::with_capacity(history_size),
            ro: VecDeque::with_capacity(history_size),
            prev_params: None,
            prev_grad: None,
            h_diag: A::one(),
            n_iter: 0,
            alpha: vec![A::zero(); history_size],
        }
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the learning rate
    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    /// Armijo (sufficient-decrease) line search parameter `c1`
    pub fn c1(&self) -> A {
        self.c1
    }

    /// Wolfe curvature line search parameter `c2`
    pub fn c2(&self) -> A {
        self.c2
    }

    /// Maximum number of line search iterations
    pub fn max_ls(&self) -> usize {
        self.max_ls
    }

    /// Number of curvature pairs currently stored
    pub fn history_len(&self) -> usize {
        self.old_stps.len()
    }

    /// The most recently stored curvature pair `(s, y)`, if any.
    ///
    /// `s` is the true parameter difference `x_k - x_{k-1}` (as observed across two
    /// consecutive calls) and `y` is the corresponding gradient difference
    /// `g_k - g_{k-1}`.
    pub fn last_curvature_pair(&self) -> Option<(&Array1<A>, &Array1<A>)> {
        match (self.old_stps.back(), self.old_dirs.back()) {
            (Some(s), Some(y)) => Some((s, y)),
            _ => None,
        }
    }

    /// The current initial inverse-Hessian scaling `gamma_k = (s·y) / (y·y)`.
    pub fn initial_hessian_scale(&self) -> A {
        self.h_diag
    }

    /// Sets the backtracking contraction factor used by [`LBFGS::step_with_loss`].
    ///
    /// Must lie strictly between 0 and 1; other values are rejected.
    pub fn set_line_search_contraction(&mut self, rho: A) -> Result<()> {
        if rho <= A::zero() || rho >= A::one() {
            return Err(OptimError::InvalidConfig(
                "line search contraction factor must lie in (0, 1)".to_string(),
            ));
        }
        self.ls_contraction = rho;
        Ok(())
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.old_dirs.clear();
        self.old_stps.clear();
        self.ro.clear();
        self.prev_params = None;
        self.prev_grad = None;
        self.h_diag = A::one();
        self.n_iter = 0;
        self.alpha.fill(A::zero());
    }

    /// Performs the two-loop recursion to compute the search direction `-H·g`
    fn compute_direction(&mut self, gradient: &Array1<A>) -> Array1<A> {
        let num_old = self.old_dirs.len();

        // Without curvature pairs the approximation is the identity: steepest descent.
        if num_old == 0 {
            return gradient.mapv(|x| -x);
        }

        // First loop: compute alpha values and initial direction
        let mut q = gradient.mapv(|x| -x);

        for i in (0..num_old).rev() {
            self.alpha[i] = self.old_stps[i].dot(&q) * self.ro[i];
            q = &q - &self.old_dirs[i] * self.alpha[i];
        }

        // Scale by initial Hessian
        let mut r = q * self.h_diag;

        // Second loop: compute final direction
        for i in 0..num_old {
            let beta = self.old_dirs[i].dot(&r) * self.ro[i];
            r = &r + &self.old_stps[i] * (self.alpha[i] - beta);
        }

        r
    }

    /// Updates the history with a new curvature pair.
    ///
    /// Returns `true` when the pair passed the curvature test and was stored.
    fn update_history(&mut self, y: Array1<A>, s: Array1<A>) -> bool {
        if self.history_size == 0 || y.len() != s.len() {
            return false;
        }

        let ys = y.dot(&s);

        // Scale-invariant curvature test: y·s > eps·||s||·||y|| keeps the implicit
        // inverse-Hessian approximation positive definite.
        let eps = A::from(1e-10).unwrap_or_else(A::epsilon);
        let threshold = eps * s.dot(&s).sqrt() * y.dot(&y).sqrt();
        if !(ys.is_finite() && ys > A::zero() && ys > threshold) {
            return false;
        }

        // Remove oldest entries if at capacity
        while self.old_dirs.len() >= self.history_size {
            self.old_dirs.pop_front();
            self.old_stps.pop_front();
            self.ro.pop_front();
        }

        // Add new entries
        let yy = y.dot(&y);
        self.old_dirs.push_back(y);
        self.old_stps.push_back(s);
        self.ro.push_back(A::one() / ys);

        // Update initial Hessian approximation: gamma_k = (s·y) / (y·y)
        if yy > A::zero() {
            self.h_diag = ys / yy;
        }
        true
    }

    /// Flattens an array into a 1-D copy, mapping shape failures onto an error.
    fn flatten<D: Dimension>(array: &Array<A, D>, what: &str) -> Result<Array1<A>> {
        array
            .to_owned()
            .into_shape_with_order(array.len())
            .map_err(|e| {
                OptimError::DimensionMismatch(format!(
                    "failed to flatten {} of shape {:?}: {}",
                    what,
                    array.shape(),
                    e
                ))
            })
    }

    /// Reshapes a flat vector back into the shape of `like`.
    fn unflatten<D: Dimension>(flat: Array1<A>, like: &Array<A, D>) -> Result<Array<A, D>> {
        flat.into_shape_with_order(like.raw_dim()).map_err(|e| {
            OptimError::DimensionMismatch(format!(
                "failed to reshape update into {:?}: {}",
                like.shape(),
                e
            ))
        })
    }

    /// Records the curvature pair implied by the caller-visible parameters and
    /// gradients, then returns the L-BFGS search direction for `gradients_flat`.
    fn prepare_direction(
        &mut self,
        params_flat: &Array1<A>,
        gradients_flat: &Array1<A>,
    ) -> Array1<A> {
        // True curvature pair: s = x_k - x_{k-1}, y = g_k - g_{k-1}.
        //
        // Both endpoints come from what the caller actually used, so an externally
        // modified parameter vector (projection, clipping, a different step size, a
        // scheduler) yields the correct `s` instead of a value reconstructed from the
        // optimizer's own assumptions.
        if let (Some(prev_params), Some(prev_grad)) = (&self.prev_params, &self.prev_grad) {
            if prev_params.len() == params_flat.len() && prev_grad.len() == gradients_flat.len() {
                let s = params_flat - prev_params;
                let y = gradients_flat - prev_grad;
                let _accepted = self.update_history(y, s);
            }
        }

        self.compute_direction(gradients_flat)
    }

    /// Performs an L-BFGS step with a backtracking Armijo line search.
    ///
    /// Unlike [`Optimizer::step`], which has no access to the objective and therefore
    /// applies a fixed step size, this method evaluates `loss_fn` at trial points and
    /// accepts the first step size satisfying the Armijo sufficient-decrease condition
    ///
    /// ```text
    /// f(x + alpha * d) <= f(x) + c1 * alpha * g^T d
    /// ```
    ///
    /// starting from `alpha = learning_rate` and contracting by the line search
    /// contraction factor (default `0.5`) for at most `max_ls` iterations. If no trial
    /// step satisfies the condition, the trial with the lowest objective value is used
    /// when it improves on `f(x)`; otherwise the parameters are returned unchanged.
    ///
    /// If the two-loop direction is not a descent direction (which can only happen
    /// through numerical error, since non-positive curvature pairs are never stored),
    /// the search falls back to steepest descent for this step.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::DimensionMismatch`] if `params` and `gradients` have
    /// different shapes, and [`OptimError::InvalidConfig`] if the objective is not
    /// finite at the current parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::Array1;
    /// use optirs_core::optimizers::LBFGS;
    ///
    /// let mut optimizer = LBFGS::new(1.0);
    /// let mut params = Array1::from_vec(vec![2.0_f64, -3.0]);
    /// let loss = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>();
    ///
    /// for _ in 0..30 {
    ///     let grads = params.mapv(|v| 2.0 * v);
    ///     params = optimizer
    ///         .step_with_loss(&params, &grads, loss)
    ///         .expect("step succeeds");
    /// }
    /// assert!(params.iter().all(|v| v.abs() < 1e-6));
    /// ```
    pub fn step_with_loss<D, F>(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
        mut loss_fn: F,
    ) -> Result<Array<A, D>>
    where
        D: Dimension,
        F: FnMut(&Array<A, D>) -> A,
    {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "parameters have shape {:?} but gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        let params_flat = Self::flatten(params, "parameters")?;
        let gradients_flat = Self::flatten(gradients, "gradients")?;

        let grad_norm = gradients_flat.dot(&gradients_flat).sqrt();
        if grad_norm <= self.tolerance_grad {
            self.prev_params = Some(params_flat);
            self.prev_grad = Some(gradients_flat);
            return Ok(params.clone());
        }

        let mut direction = self.prepare_direction(&params_flat, &gradients_flat);

        // Directional derivative g^T d must be negative for a descent direction.
        // A NaN (or otherwise incomparable) `gtd` is not a valid descent direction
        // either, so it must fall into this branch alongside non-negative values.
        let mut gtd = gradients_flat.dot(&direction);
        if !matches!(gtd.partial_cmp(&A::zero()), Some(std::cmp::Ordering::Less)) {
            direction = gradients_flat.mapv(|x| -x);
            gtd = -gradients_flat.dot(&gradients_flat);
        }

        let f0 = loss_fn(params);
        if !f0.is_finite() {
            return Err(OptimError::InvalidConfig(
                "objective is not finite at the current parameters".to_string(),
            ));
        }

        let mut alpha = self.learning_rate;
        let mut best: Option<(A, A, Array<A, D>)> = None; // (loss, alpha, candidate)
        let mut accepted: Option<(A, Array<A, D>)> = None; // (alpha, candidate)

        for _ in 0..self.max_ls.max(1) {
            let candidate_flat = &params_flat + &(&direction * alpha);
            let candidate = Self::unflatten(candidate_flat, params)?;
            let f = loss_fn(&candidate);

            if f.is_finite() && f <= f0 + self.c1 * alpha * gtd {
                accepted = Some((alpha, candidate));
                break;
            }

            if f.is_finite() && f < f0 {
                let improves = match &best {
                    Some((best_f, _, _)) => f < *best_f,
                    None => true,
                };
                if improves {
                    best = Some((f, alpha, candidate));
                }
            }

            alpha = alpha * self.ls_contraction;
            // Stop on a non-positive (or NaN) step size rather than looping forever.
            if !matches!(
                alpha.partial_cmp(&A::zero()),
                Some(std::cmp::Ordering::Greater)
            ) {
                break;
            }
        }

        let (step_size, new_params) = match accepted {
            Some((a, candidate)) => (a, candidate),
            None => match best {
                // No Armijo-acceptable step, but some trial did reduce the objective.
                Some((_, a, candidate)) => (a, candidate),
                // Nothing improved: stay put rather than move to a worse point.
                None => (A::zero(), params.clone()),
            },
        };

        // Record the point the gradient was evaluated at, together with that gradient:
        // the next call forms s = x_{k+1} - x_k and y = g_{k+1} - g_k from them.
        self.prev_params = Some(params_flat);
        self.prev_grad = Some(gradients_flat);
        if step_size > A::zero() {
            self.n_iter += 1;
        }

        Ok(new_params)
    }
}

impl<A, D> Optimizer<A, D> for LBFGS<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension,
{
    /// Performs an L-BFGS step with a **fixed** step size.
    ///
    /// This trait method has no access to the objective, so no line search is possible;
    /// the two-loop direction is scaled by the learning rate (reduced by
    /// `1 / (1 + ||g||)` on the very first step, before any curvature information
    /// exists). Use [`LBFGS::step_with_loss`] for the backtracking Armijo line search
    /// that uses the configured `c1` and `max_ls`.
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "parameters have shape {:?} but gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        // Convert to 1D for computation
        let params_flat = Self::flatten(params, "parameters")?;
        let gradients_flat = Self::flatten(gradients, "gradients")?;

        // Check convergence
        let grad_norm = gradients_flat.dot(&gradients_flat).sqrt();
        if grad_norm <= self.tolerance_grad {
            self.prev_params = Some(params_flat);
            self.prev_grad = Some(gradients_flat);
            return Ok(params.clone());
        }

        // Record the true curvature pair from the previous iterate and compute the
        // search direction for the current gradient.
        let direction = self.prepare_direction(&params_flat, &gradients_flat);

        // Fixed step size: without an objective there is nothing to line search on.
        let step_size = if self.old_stps.is_empty() {
            // No curvature information yet: damp the raw steepest-descent step.
            self.learning_rate / (A::one() + grad_norm)
        } else {
            self.learning_rate
        };

        // Update parameters
        let new_params_flat = &params_flat + &(&direction * step_size);

        // Store the current iterate and gradient for the next curvature pair.
        self.prev_params = Some(params_flat);
        self.prev_grad = Some(gradients_flat);
        self.n_iter += 1;

        // Reshape back to original dimensions
        Self::unflatten(new_params_flat, params)
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_lbfgs_basic_creation() {
        let optimizer: LBFGS<f64> = LBFGS::new(1.0);
        assert_abs_diff_eq!(optimizer.learning_rate(), 1.0);
        assert_eq!(optimizer.history_size, 100);
        assert_abs_diff_eq!(optimizer.tolerance_grad, 1e-7);
    }

    #[test]
    fn test_lbfgs_convergence() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Minimize f(x) = x^2
        let mut params = Array1::from_vec(vec![10.0]);

        for _ in 0..50 {
            let gradients = Array1::from_vec(vec![2.0 * params[0]]);
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_lbfgs_convergence");
        }

        // Should converge close to 0
        assert!(params[0].abs() < 0.1);
    }

    #[test]
    fn test_lbfgs_2d() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Minimize f(x,y) = x^2 + y^2
        let mut params = Array1::from_vec(vec![5.0, 3.0]);

        for _ in 0..50 {
            let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
            params = optimizer
                .step(&params, &gradients)
                .expect("optimizer.step succeeds in test_lbfgs_2d");
        }

        // Should converge close to (0, 0)
        assert!(params[0].abs() < 0.1);
        assert!(params[1].abs() < 0.1);
    }

    #[test]
    fn test_lbfgs_reset() {
        let mut optimizer: LBFGS<f64> = LBFGS::new(0.1);

        // Perform some steps
        let mut params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![2.0]);
        params = optimizer
            .step(&params, &gradients)
            .expect("optimizer.step succeeds in test_lbfgs_reset");

        // Need one more step to actually update history
        let gradients2 = Array1::from_vec(vec![1.5]);
        params = optimizer
            .step(&params, &gradients2)
            .expect("optimizer.step succeeds in test_lbfgs_reset");

        // Third step to populate history
        let gradients3 = Array1::from_vec(vec![1.0]);
        let _ = optimizer
            .step(&params, &gradients3)
            .expect("optimizer.step succeeds in test_lbfgs_reset");

        // Verify state exists
        assert!(!optimizer.old_dirs.is_empty());
        assert!(optimizer.n_iter > 0);

        // Reset
        optimizer.reset();

        // Verify state is cleared
        assert!(optimizer.old_dirs.is_empty());
        assert!(optimizer.old_stps.is_empty());
        assert!(optimizer.ro.is_empty());
        assert!(optimizer.prev_grad.is_none());
        assert_eq!(optimizer.n_iter, 0);
    }
}
