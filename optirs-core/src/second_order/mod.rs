// Second-order optimization methods
//
// This module provides implementations of second-order optimization methods
// that use curvature information (Hessian matrix) to improve convergence.

pub mod kfac;
pub mod newton_cg;

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

pub use self::kfac::{KFACConfig, KFACLayerState, KFACStats, LayerInfo, LayerType, KFAC};
pub use self::newton_cg::NewtonCG;

/// Trait for second-order optimization methods
pub trait SecondOrderOptimizer<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension> {
    /// Update parameters using second-order information
    fn step_second_order(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
        hessian_info: &HessianInfo<A, D>,
    ) -> Result<Array<A, D>>;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Hessian information for second-order methods
#[derive(Debug, Clone)]
pub enum HessianInfo<A: Float, D: Dimension> {
    /// Full Hessian matrix (expensive, rarely used in practice)
    Full(Array2<A>),
    /// Diagonal approximation of Hessian
    Diagonal(Array<A, D>),
    /// L-BFGS style quasi-Newton approximation
    QuasiNewton {
        /// Parameter differences history
        s_history: VecDeque<Array<A, D>>,
        /// Gradient differences history
        y_history: VecDeque<Array<A, D>>,
    },
    /// Gauss-Newton approximation for least squares problems
    GaussNewton(Array2<A>),
}

/// Approximated Hessian computation methods
pub mod hessian_approximation {
    use super::*;

    /// Compute diagonal Hessian approximation using finite differences (1D only)
    pub fn diagonal_finite_difference<A, F>(
        params: &Array1<A>,
        gradient_fn: F,
        epsilon: A,
    ) -> Result<Array1<A>>
    where
        A: Float + ScalarOperand + Debug + Copy,
        F: Fn(&Array1<A>) -> Result<Array1<A>>,
    {
        let mut hessian_diag = Array1::zeros(params.len());
        let _original_grad = gradient_fn(params)?;

        for i in 0..params.len() {
            let mut param_plus = params.clone();
            let mut param_minus = params.clone();

            // Forward difference: f(x + h) - f(x)
            param_plus[i] = params[i] + epsilon;
            let grad_plus = gradient_fn(&param_plus)?;

            // Backward difference: f(x) - f(x - h)
            param_minus[i] = params[i] - epsilon;
            let grad_minus = gradient_fn(&param_minus)?;

            // Hessian diagonal: derivative of gradient using central difference
            let second_deriv =
                (grad_plus[i] - grad_minus[i]) / (A::from(2.0).expect("unwrap failed") * epsilon);
            hessian_diag[i] = second_deriv;
        }

        Ok(hessian_diag)
    }

    /// Relative threshold used by the curvature (positive-definiteness) test.
    ///
    /// A curvature pair `(s, y)` is only usable by the L-BFGS two-loop recursion when
    /// `y·s > 0`. Accepting a pair with `y·s <= 0` destroys the positive-definiteness
    /// of the implicit inverse-Hessian approximation and can turn the resulting
    /// "search direction" into an ascent direction. We use the standard relative
    /// test `y·s > eps * ||s|| * ||y||` so the check is scale invariant.
    fn curvature_threshold<A: Float>() -> A {
        A::from(1e-8).unwrap_or_else(A::epsilon)
    }

    /// Euclidean norm of an array, computed without allocating.
    fn euclidean_norm<A, D>(v: &Array<A, D>) -> A
    where
        A: Float,
        D: Dimension,
    {
        v.iter().fold(A::zero(), |acc, &x| acc + x * x).sqrt()
    }

    /// Dot product of two arrays of identical shape.
    fn dot_product<A, D>(a: &Array<A, D>, b: &Array<A, D>) -> A
    where
        A: Float,
        D: Dimension,
    {
        a.iter()
            .zip(b.iter())
            .fold(A::zero(), |acc, (&x, &y)| acc + x * y)
    }

    /// Returns `true` when the curvature pair `(s, y)` satisfies `y·s > eps·||s||·||y||`
    /// and is therefore safe to store in the L-BFGS history.
    pub fn is_curvature_pair_acceptable<A, D>(
        param_diff: &Array<A, D>,
        grad_diff: &Array<A, D>,
    ) -> bool
    where
        A: Float,
        D: Dimension,
    {
        if param_diff.len() != grad_diff.len() {
            return false;
        }
        let ys = dot_product(param_diff, grad_diff);
        if !ys.is_finite() || ys <= A::zero() {
            return false;
        }
        let threshold =
            curvature_threshold::<A>() * euclidean_norm(param_diff) * euclidean_norm(grad_diff);
        ys > threshold
    }

    /// Update L-BFGS Hessian approximation.
    ///
    /// Curvature pairs that fail the positive-curvature test `y·s > eps·||s||·||y||`
    /// are **skipped** (not stored): storing them would destroy the positive
    /// definiteness of the implicit inverse-Hessian approximation.
    ///
    /// # Returns
    ///
    /// `true` if the pair was accepted and stored, `false` if it was skipped.
    pub fn update_lbfgs_approximation<A, D>(
        s_history: &mut VecDeque<Array<A, D>>,
        y_history: &mut VecDeque<Array<A, D>>,
        param_diff: Array<A, D>,
        grad_diff: Array<A, D>,
        max_history: usize,
    ) -> bool
    where
        A: Float + ScalarOperand + Debug,
        D: Dimension,
    {
        if !is_curvature_pair_acceptable(&param_diff, &grad_diff) {
            return false;
        }

        // Add new differences to the history
        s_history.push_back(param_diff);
        y_history.push_back(grad_diff);

        // Maintain maximum history size
        while s_history.len() > max_history {
            s_history.pop_front();
            y_history.pop_front();
        }
        true
    }

    /// Compute the L-BFGS initial inverse-Hessian scaling `gamma_k = (s·y) / (y·y)`
    /// from the most recent *acceptable* curvature pair.
    ///
    /// Returns `None` when no stored pair passes the curvature test (in which case the
    /// caller should fall back to a user-supplied scale).
    pub fn initial_hessian_scaling<A, D>(
        s_history: &VecDeque<Array<A, D>>,
        y_history: &VecDeque<Array<A, D>>,
    ) -> Option<A>
    where
        A: Float,
        D: Dimension,
    {
        let m = s_history.len().min(y_history.len());
        for i in (0..m).rev() {
            let s_i = &s_history[i];
            let y_i = &y_history[i];
            if !is_curvature_pair_acceptable(s_i, y_i) {
                continue;
            }
            let yy = dot_product(y_i, y_i);
            if yy <= A::zero() || !yy.is_finite() {
                continue;
            }
            let gamma = dot_product(s_i, y_i) / yy;
            if gamma.is_finite() && gamma > A::zero() {
                return Some(gamma);
            }
        }
        None
    }

    /// Apply the L-BFGS two-loop recursion to approximate `H^(-1) * grad`.
    ///
    /// # Curvature filtering
    ///
    /// Pairs that fail the positive-curvature test `y·s > eps·||s||·||y||` are skipped:
    /// they do not correspond to a positive-definite update and including them can turn
    /// the result into an ascent direction. `s_history` / `y_history` populated through
    /// [`update_lbfgs_approximation`] are already filtered, but a caller may also build a
    /// [`super::HessianInfo::QuasiNewton`] history by hand, so the filter is applied here
    /// as well.
    ///
    /// # Initial inverse-Hessian scaling
    ///
    /// `H_0 = gamma_k * I` with `gamma_k = (s·y) / (y·y)` computed from the most recent
    /// acceptable curvature pair (Nocedal & Wright, eq. 7.20). `initial_hessian_scale` is
    /// used as the fallback when no acceptable pair exists (including an empty history).
    pub fn lbfgs_two_loop_recursion<A, D>(
        gradient: &Array<A, D>,
        s_history: &VecDeque<Array<A, D>>,
        y_history: &VecDeque<Array<A, D>>,
        initial_hessian_scale: A,
    ) -> Result<Array<A, D>>
    where
        A: Float + ScalarOperand + Debug,
        D: Dimension,
    {
        if s_history.len() != y_history.len() {
            return Err(OptimError::InvalidConfig(
                "History sizes don't match in L-BFGS".to_string(),
            ));
        }

        let m = s_history.len();
        if m == 0 {
            // No history, return scaled gradient
            return Ok(gradient * initial_hessian_scale);
        }

        // Precompute which pairs are usable and their rho values, so both loops
        // agree and the `alphas` indices stay aligned with the history indices.
        let mut rhos: Vec<Option<A>> = Vec::with_capacity(m);
        for i in 0..m {
            let s_i = &s_history[i];
            let y_i = &y_history[i];
            if s_i.len() != gradient.len() || y_i.len() != gradient.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "L-BFGS history entry {} has length {}/{}, expected {}",
                    i,
                    s_i.len(),
                    y_i.len(),
                    gradient.len()
                )));
            }
            if is_curvature_pair_acceptable(s_i, y_i) {
                rhos.push(Some(A::one() / dot_product(y_i, s_i)));
            } else {
                rhos.push(None);
            }
        }

        // H_0 = gamma_k * I from the latest acceptable pair; fall back to the
        // caller-supplied scale when every pair was rejected.
        let scale = initial_hessian_scaling(s_history, y_history).unwrap_or(initial_hessian_scale);

        let mut q = gradient.clone();
        let mut alphas = vec![A::zero(); m];

        // First loop (newest -> oldest): compute alphas and update q
        for i in (0..m).rev() {
            let rho_i = match rhos[i] {
                Some(rho) => rho,
                None => continue,
            };
            let s_i = &s_history[i];
            let y_i = &y_history[i];

            // alpha_i = rho_i * s_i^T * q
            let alpha_i = rho_i * dot_product(s_i, &q);
            alphas[i] = alpha_i;

            // q = q - alpha_i * y_i
            for (q_val, &y_val) in q.iter_mut().zip(y_i.iter()) {
                *q_val = *q_val - alpha_i * y_val;
            }
        }

        // Scale by the initial inverse-Hessian approximation
        q.mapv_inplace(|x| x * scale);

        // Second loop (oldest -> newest): compute the final result
        for i in 0..m {
            let rho_i = match rhos[i] {
                Some(rho) => rho,
                None => continue,
            };
            let s_i = &s_history[i];
            let y_i = &y_history[i];

            // beta = rho_i * y_i^T * q
            let beta = rho_i * dot_product(y_i, &q);

            // q = q + (alpha_i - beta) * s_i
            let coeff = alphas[i] - beta;
            for (q_val, &s_val) in q.iter_mut().zip(s_i.iter()) {
                *q_val = *q_val + coeff * s_val;
            }
        }

        Ok(q)
    }

    /// Gauss-Newton Hessian approximation for least squares problems
    pub fn gauss_newton_approximation<A>(jacobian: &Array2<A>) -> Result<Array2<A>>
    where
        A: Float + ScalarOperand + Debug,
    {
        // Gauss-Newton approximation: H ≈ J^T * J
        let j_transpose = jacobian.t();
        let hessian_approx = j_transpose.dot(jacobian);
        Ok(hessian_approx)
    }
}

/// Newton's method optimizer
///
/// # Descent safeguarding
///
/// A raw Newton step `-H^{-1} g` is only a descent direction when `H` is positive
/// definite. For a diagonal Hessian approximation this optimizer therefore uses
/// `|h_ii|` (floored at [`Newton::min_curvature`]) as the denominator, which keeps the
/// update a descent direction even where the curvature is negative or vanishing.
#[derive(Debug, Clone)]
pub struct Newton<A: Float> {
    learning_rate: A,
    regularization: A, // For numerical stability
    min_curvature: A,  // Lower bound on |h_ii| used as the step denominator
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + Send + Sync> Newton<A> {
    /// Default lower bound on the absolute diagonal curvature.
    fn default_min_curvature() -> A {
        A::from(1e-8).unwrap_or_else(A::epsilon)
    }

    /// Create a new Newton optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            regularization: A::from(1e-6).unwrap_or_else(A::epsilon),
            min_curvature: Self::default_min_curvature(),
        }
    }

    /// Set regularization parameter for numerical stability
    pub fn with_regularization(mut self, regularization: A) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the lower bound applied to `|h_ii|` before it is used as the step denominator.
    ///
    /// Values `<= 0` are ignored and the default is kept, since a non-positive floor
    /// would re-admit division by (near-)zero curvature.
    pub fn with_min_curvature(mut self, min_curvature: A) -> Self {
        if min_curvature > A::zero() {
            self.min_curvature = min_curvature;
        }
        self
    }

    /// Get the lower bound applied to `|h_ii|`.
    pub fn min_curvature(&self) -> A {
        self.min_curvature
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + Send + Sync>
    SecondOrderOptimizer<A, scirs2_core::ndarray::Ix1> for Newton<A>
{
    fn step_second_order(
        &mut self,
        params: &Array1<A>,
        gradients: &Array1<A>,
        hessian_info: &HessianInfo<A, scirs2_core::ndarray::Ix1>,
    ) -> Result<Array1<A>> {
        match hessian_info {
            HessianInfo::Diagonal(hessian_diag) => {
                if params.len() != hessian_diag.len() || params.len() != gradients.len() {
                    return Err(OptimError::DimensionMismatch(
                        "Parameter, gradient, and Hessian dimensions must match".to_string(),
                    ));
                }

                let mut update = Array1::zeros(params.len());
                for i in 0..params.len() {
                    // Use |h_ii| (floored at `min_curvature`) as the denominator.
                    //
                    // Dividing by a *signed* curvature flips the sign of the update
                    // wherever `h_ii < 0`, which turns the step into an ascent step at
                    // exactly the points (saddles / concave regions) where a descent
                    // step matters most. The absolute value keeps `-lr * g_i / |h_ii|`
                    // a descent direction for every coordinate, and the floor removes
                    // the division-by-(near-)zero case without silently switching to a
                    // differently-scaled fallback.
                    let h_ii = hessian_diag[i] + self.regularization;
                    let denom = h_ii.abs().max(self.min_curvature);
                    update[i] = gradients[i] / denom;
                }

                Ok(params - &(update * self.learning_rate))
            }
            HessianInfo::QuasiNewton {
                s_history,
                y_history,
            } => {
                // Use L-BFGS approximation
                let search_direction = hessian_approximation::lbfgs_two_loop_recursion(
                    gradients,
                    s_history,
                    y_history,
                    A::one(), // Initial Hessian scale
                )?;

                Ok(params - &(search_direction * self.learning_rate))
            }
            _ => Err(OptimError::InvalidConfig(
                "Unsupported Hessian information type for Newton method".to_string(),
            )),
        }
    }

    fn reset(&mut self) {
        // Newton method is stateless, nothing to reset
    }
}

/// Quasi-Newton L-BFGS optimizer
#[derive(Debug)]
pub struct LBFGS<A: Float, D: Dimension> {
    learning_rate: A,
    max_history: usize,
    s_history: VecDeque<Array<A, D>>,
    y_history: VecDeque<Array<A, D>>,
    previous_params: Option<Array<A, D>>,
    previous_grad: Option<Array<A, D>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync> LBFGS<A, D> {
    /// Create a new L-BFGS optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            max_history: 10,
            s_history: VecDeque::new(),
            y_history: VecDeque::new(),
            previous_params: None,
            previous_grad: None,
        }
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Perform L-BFGS step
    pub fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Update history if we have previous step information
        if let (Some(prev_params), Some(prev_grad)) = (&self.previous_params, &self.previous_grad) {
            let s = params - prev_params; // Parameter difference
            let y = gradients - prev_grad; // Gradient difference

            // Pairs failing the curvature test `y·s > eps·||s||·||y||` are skipped by
            // `update_lbfgs_approximation` to preserve positive definiteness.
            let _accepted = hessian_approximation::update_lbfgs_approximation(
                &mut self.s_history,
                &mut self.y_history,
                s,
                y,
                self.max_history,
            );
        }

        // Compute search direction using two-loop recursion
        let search_direction = if self.s_history.is_empty() {
            // No history, use gradient descent
            gradients.clone()
        } else {
            hessian_approximation::lbfgs_two_loop_recursion(
                gradients,
                &self.s_history,
                &self.y_history,
                A::one(),
            )?
        };

        // Update parameters
        let new_params = params - &(search_direction * self.learning_rate);

        // Store current information for next iteration
        self.previous_params = Some(params.clone());
        self.previous_grad = Some(gradients.clone());

        Ok(new_params)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    SecondOrderOptimizer<A, D> for LBFGS<A, D>
{
    fn step_second_order(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
        _hessian_info: &HessianInfo<A, D>, // L-BFGS maintains its own history
    ) -> Result<Array<A, D>> {
        self.step(params, gradients)
    }

    fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.previous_params = None;
        self.previous_grad = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_diagonal_hessian_approximation() {
        // Test on a simple quadratic function: f(x) = x^2
        let params = Array1::from_vec(vec![1.0]);

        // Gradient function for quadratic: grad = 2*x
        let gradient_fn =
            |x: &Array1<f64>| -> Result<Array1<f64>> { Ok(Array1::from_vec(vec![2.0 * x[0]])) };

        let hessian_diag =
            hessian_approximation::diagonal_finite_difference(&params, gradient_fn, 1e-5)
                .expect("unwrap failed");

        // For quadratic function f(x) = x^2, second derivative should be 2.0
        assert_relative_eq!(hessian_diag[0], 2.0, epsilon = 1e-1);
    }

    #[test]
    fn test_lbfgs_two_loop_recursion() {
        let gradient = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut s_history = VecDeque::new();
        let mut y_history = VecDeque::new();

        // Add some history
        s_history.push_back(Array1::from_vec(vec![0.1, 0.1, 0.1]));
        y_history.push_back(Array1::from_vec(vec![0.2, 0.3, 0.4]));

        let result =
            hessian_approximation::lbfgs_two_loop_recursion(&gradient, &s_history, &y_history, 1.0)
                .expect("unwrap failed");

        // Result should be different from original gradient due to curvature information
        assert_ne!(result, gradient);
        assert_eq!(result.len(), gradient.len());
    }

    #[test]
    fn test_newton_method() {
        let mut optimizer = Newton::new(0.1);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2]);
        let hessian_diag = Array1::from_vec(vec![2.0, 4.0]);

        let hessian_info = HessianInfo::Diagonal(hessian_diag);
        let new_params = optimizer
            .step_second_order(&params, &gradients, &hessian_info)
            .expect("unwrap failed");

        // Verify parameters were updated
        assert!(new_params[0] < params[0]);
        assert!(new_params[1] < params[1]);
    }

    #[test]
    fn test_lbfgs_optimizer() {
        let mut optimizer = LBFGS::new(0.01).with_max_history(5);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients1 = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let gradients2 = Array1::from_vec(vec![0.05, 0.15, 0.25]);

        // First step
        params = optimizer.step(&params, &gradients1).expect("unwrap failed");

        // Second step (should use history)
        let new_params = optimizer.step(&params, &gradients2).expect("unwrap failed");

        // Verify parameters were updated
        assert_ne!(new_params, params);
        assert_eq!(optimizer.s_history.len(), 1);
        assert_eq!(optimizer.y_history.len(), 1);
    }

    #[test]
    fn test_gauss_newton_approximation() {
        let jacobian = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("unwrap failed");
        let hessian_approx =
            hessian_approximation::gauss_newton_approximation(&jacobian).expect("unwrap failed");

        // Should be a 2x2 matrix (J^T * J)
        assert_eq!(hessian_approx.dim(), (2, 2));

        // Verify it's positive semidefinite by checking diagonal elements are non-negative
        assert!(hessian_approx[(0, 0)] >= 0.0);
        assert!(hessian_approx[(1, 1)] >= 0.0);
    }
}
