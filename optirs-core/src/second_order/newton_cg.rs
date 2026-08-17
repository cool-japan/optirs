// Newton-CG (Newton Conjugate Gradient) Optimizer
//
// Newton-CG is a second-order optimization method that combines Newton's method
// with conjugate gradient for solving the Newton system. It only requires
// Hessian-vector products rather than the full Hessian matrix, making it
// memory-efficient for large-scale problems.
//
// Algorithm:
//   1. Compute gradient: g = ∇f(x)
//   2. Solve Newton system using CG: H*d = -g
//   3. Update parameters: x_{t+1} = x_t + α*d
//
// References:
//   - Nash, S. G. (1984). Newton-type minimization via the Lanczos method.
//   - Nocedal & Wright (2006). Numerical Optimization, Chapter 7.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Newton-CG optimizer configuration
///
/// Newton-CG uses conjugate gradient to solve the Newton system H*d = -g,
/// where H is the Hessian and g is the gradient. This avoids explicitly
/// computing and storing the full Hessian matrix.
///
/// # Key Features
/// - Memory-efficient: only needs Hessian-vector products
/// - Suitable for large-scale problems
/// - Conjugate gradient with early termination and negative-curvature handling
/// - Optional Steihaug-Toint trust region (see below)
///
/// # Step size control
///
/// Two step-control modes are available and they are *not* interchangeable:
///
/// * [`NewtonCG::step`] applies a **fixed** step: `x + learning_rate * d`, where `d` is
///   the truncated-CG solution of the Newton system. If a trust-region radius has been
///   configured with [`NewtonCG::with_trust_region`], `d` is additionally truncated so
///   that `||d|| <= Delta`, but the radius is **not** adapted, because adapting it
///   requires evaluating the objective. Without a configured radius this is plain
///   (unconstrained) Newton-CG.
/// * [`NewtonCG::step_with_loss`] runs the full Steihaug-Toint trust-region algorithm:
///   truncated CG constrained to `||d|| <= Delta`, then the ratio of actual to predicted
///   reduction decides whether the step is accepted and how `Delta` is updated.
///
/// # Type Parameters
/// - `T`: Floating-point type (f32 or f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtonCG<T: Float> {
    /// Step size applied to the CG direction by [`NewtonCG::step`].
    ///
    /// This is a plain step-size multiplier, not a trust-region radius; the trust
    /// region lives in [`NewtonCG::trust_region_radius`].
    learning_rate: T,

    /// CG termination tolerance, as a *relative* bound on the residual norm:
    /// CG stops once `||r|| <= cg_tolerance * ||r_0||`.
    cg_tolerance: T,

    /// Maximum CG iterations
    cg_max_iters: usize,

    /// Hessian regularization for numerical stability
    hessian_reg: T,

    /// Trust-region radius `Delta`. `None` disables the trust region entirely.
    #[serde(default = "no_trust_region")]
    trust_region_radius: Option<T>,

    /// Upper bound on the adaptively grown trust-region radius.
    #[serde(default = "default_max_trust_region_radius")]
    max_trust_region_radius: T,

    /// Lower bound on the adaptively shrunk trust-region radius.
    #[serde(default = "default_min_trust_region_radius")]
    min_trust_region_radius: T,

    /// Acceptance threshold `eta` for the trust-region ratio `rho`.
    /// A step is rejected when `rho <= eta`.
    #[serde(default = "default_trust_region_eta")]
    trust_region_eta: T,

    /// Whether the most recent [`NewtonCG::step_with_loss`] call accepted its step.
    #[serde(default = "default_last_step_accepted")]
    last_step_accepted: bool,

    /// Step counter
    step_count: usize,
}

fn no_trust_region<T>() -> Option<T> {
    None
}

fn default_max_trust_region_radius<T: Float>() -> T {
    T::from(10.0).unwrap_or_else(T::one)
}

fn default_min_trust_region_radius<T: Float>() -> T {
    T::from(1e-8).unwrap_or_else(T::epsilon)
}

fn default_trust_region_eta<T: Float>() -> T {
    T::from(0.1).unwrap_or_else(T::zero)
}

fn default_last_step_accepted() -> bool {
    true
}

/// Outcome of one truncated (Steihaug-Toint) CG solve.
#[derive(Debug, Clone)]
struct CgSolution<T: Float> {
    /// Approximate solution `d` of `(H + reg*I) d = -g`.
    direction: Array1<T>,
    /// `true` when the solve stopped on the trust-region boundary.
    on_boundary: bool,
    /// `(H + reg*I) * d`, reused for the predicted-reduction model.
    hd: Array1<T>,
}

/// Euclidean inner product of two equally sized vectors.
fn dot<T: Float>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// Positive root `tau` of `||d + tau * p|| == delta`.
///
/// Used by the Steihaug-Toint truncation to step from the interior iterate `d` to the
/// trust-region boundary along `p`. Returns `None` when `p` is (numerically) zero or the
/// quadratic has no real root, in which case the caller keeps the current iterate.
fn boundary_tau<T: Float>(d: &Array1<T>, p: &Array1<T>, delta: T) -> Option<T> {
    let d_view = d.view();
    let p_view = p.view();
    let pp = dot(&p_view, &p_view);
    if !matches!(
        pp.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) {
        return None;
    }
    let dp = dot(&d_view, &p_view);
    let dd = dot(&d_view, &d_view);

    // (p·p) tau^2 + 2 (d·p) tau + (d·d - delta^2) = 0
    let c = dd - delta * delta;
    let discriminant = dp * dp - pp * c;
    if discriminant < T::zero() || !discriminant.is_finite() {
        return None;
    }
    let tau = (-dp + discriminant.sqrt()) / pp;
    if tau.is_finite() && tau > T::zero() {
        Some(tau)
    } else {
        None
    }
}

impl<T: Float + ScalarOperand> Default for NewtonCG<T> {
    fn default() -> Self {
        // Built directly rather than through `new`, so that `Default` is total for every
        // `T: Float` (no conversion is allowed to fail into a panic here).
        Self {
            learning_rate: T::one(),
            cg_tolerance: T::from(1e-6).unwrap_or_else(T::epsilon),
            cg_max_iters: 100,
            hessian_reg: T::from(1e-6).unwrap_or_else(T::epsilon),
            trust_region_radius: None,
            max_trust_region_radius: default_max_trust_region_radius(),
            min_trust_region_radius: default_min_trust_region_radius(),
            trust_region_eta: default_trust_region_eta(),
            last_step_accepted: true,
            step_count: 0,
        }
    }
}

impl<T: Float + ScalarOperand> NewtonCG<T> {
    /// Create a new Newton-CG optimizer
    ///
    /// # Arguments
    /// - `learning_rate`: Step size applied to the CG direction (typically 0.5-2.0)
    /// - `cg_tolerance`: Relative CG convergence tolerance on `||r||/||r_0||` (typically 1e-6)
    /// - `cg_max_iters`: Maximum CG iterations (typically 50-200)
    /// - `hessian_reg`: Hessian regularization (typically 1e-6)
    ///
    /// The trust region is disabled by default; enable it with
    /// [`NewtonCG::with_trust_region`].
    ///
    /// # Example
    /// ```
    /// use optirs_core::second_order::newton_cg::NewtonCG;
    ///
    /// let optimizer = NewtonCG::<f32>::new(1.0, 1e-6, 100, 1e-6).expect("NewtonCG::<f32>::new succeeds");
    /// ```
    pub fn new(
        learning_rate: T,
        cg_tolerance: T,
        cg_max_iters: usize,
        hessian_reg: T,
    ) -> Result<Self> {
        if !matches!(
            learning_rate.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::InvalidParameter(
                "learning_rate must be positive".to_string(),
            ));
        }
        if !matches!(
            cg_tolerance.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::InvalidParameter(
                "cg_tolerance must be positive".to_string(),
            ));
        }
        if cg_max_iters == 0 {
            return Err(OptimError::InvalidParameter(
                "cg_max_iters must be positive".to_string(),
            ));
        }
        if hessian_reg < T::zero() {
            return Err(OptimError::InvalidParameter(
                "hessian_reg must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            learning_rate,
            cg_tolerance,
            cg_max_iters,
            hessian_reg,
            trust_region_radius: None,
            max_trust_region_radius: default_max_trust_region_radius(),
            min_trust_region_radius: default_min_trust_region_radius(),
            trust_region_eta: default_trust_region_eta(),
            last_step_accepted: true,
            step_count: 0,
        })
    }

    /// Enable the Steihaug-Toint trust region with the given initial radius.
    ///
    /// With a radius configured, [`NewtonCG::step`] truncates the CG direction to
    /// `||d|| <= Delta` (without adapting `Delta`), and [`NewtonCG::step_with_loss`]
    /// runs the full trust-region algorithm including radius adaptation and step
    /// rejection.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] if `radius` is not positive.
    pub fn with_trust_region(mut self, radius: T) -> Result<Self> {
        if !matches!(
            radius.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::InvalidParameter(
                "trust region radius must be positive".to_string(),
            ));
        }
        self.trust_region_radius = Some(radius);
        if self.max_trust_region_radius < radius {
            self.max_trust_region_radius = radius;
        }
        Ok(self)
    }

    /// Set the upper bound used when the trust-region radius is grown.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] if `radius` is not positive.
    pub fn with_max_trust_region_radius(mut self, radius: T) -> Result<Self> {
        if !matches!(
            radius.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(OptimError::InvalidParameter(
                "max trust region radius must be positive".to_string(),
            ));
        }
        self.max_trust_region_radius = radius;
        Ok(self)
    }

    /// Set the acceptance threshold `eta`: a trust-region step with ratio
    /// `rho <= eta` is rejected. Must lie in `[0, 1)`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] if `eta` is outside `[0, 1)`.
    pub fn with_trust_region_eta(mut self, eta: T) -> Result<Self> {
        if eta < T::zero() || eta >= T::one() {
            return Err(OptimError::InvalidParameter(
                "trust region eta must lie in [0, 1)".to_string(),
            ));
        }
        self.trust_region_eta = eta;
        Ok(self)
    }

    /// Current trust-region radius, or `None` when the trust region is disabled.
    pub fn trust_region_radius(&self) -> Option<T> {
        self.trust_region_radius
    }

    /// Whether the most recent [`NewtonCG::step_with_loss`] call accepted its step.
    ///
    /// Always `true` before the first trust-region step and for [`NewtonCG::step`],
    /// which never rejects.
    pub fn last_step_accepted(&self) -> bool {
        self.last_step_accepted
    }

    /// Perform a Newton-CG optimization step with a **fixed** step size.
    ///
    /// The Newton system is solved with truncated CG (negative curvature terminates the
    /// solve rather than corrupting the iterate) and the resulting direction is applied
    /// as `x + learning_rate * d`. If a trust-region radius has been configured via
    /// [`NewtonCG::with_trust_region`], `d` is truncated so that `||d|| <= Delta`, but
    /// the radius is *not* adapted and the step is never rejected: both require
    /// evaluating the objective. Use [`NewtonCG::step_with_loss`] for the full
    /// Steihaug-Toint trust-region algorithm.
    ///
    /// # Arguments
    /// - `params`: Current parameter values
    /// - `grads`: Gradient at current parameters
    /// - `hvp_fn`: Function that computes Hessian-vector products
    ///
    /// # Returns
    /// Updated parameters after Newton-CG step
    ///
    /// # Example
    /// ```
    /// use optirs_core::second_order::newton_cg::NewtonCG;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let mut optimizer = NewtonCG::<f32>::default();
    /// let params = array![1.0, 2.0, 3.0];
    /// let grads = array![0.1, 0.2, 0.3];
    ///
    /// // Hessian-vector product function (identity for this example)
    /// let hvp_fn = |v: &[f32]| -> Vec<f32> { v.to_vec() };
    ///
    /// let updated = optimizer.step(params.view(), grads.view(), hvp_fn).expect("optimizer.step succeeds");
    /// ```
    pub fn step<F>(
        &mut self,
        params: ArrayView1<T>,
        grads: ArrayView1<T>,
        hvp_fn: F,
    ) -> Result<Array1<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let n = params.len();

        if grads.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected gradient size {}, got {}",
                n,
                grads.len()
            )));
        }

        self.step_count += 1;
        self.last_step_accepted = true;

        // Solve H*d = -g using truncated conjugate gradient
        // where H is the Hessian and g is the gradient
        let solution = self.truncated_cg(&grads, &hvp_fn, self.trust_region_radius)?;

        // Update parameters: x_{t+1} = x_t + α*d
        Ok(params.to_owned() + &(solution.direction * self.learning_rate))
    }

    /// Perform a Steihaug-Toint trust-region Newton-CG step.
    ///
    /// Unlike [`NewtonCG::step`], this variant can evaluate the objective, so it runs
    /// the complete trust-region algorithm:
    ///
    /// 1. Solve `H d = -g` with CG truncated to `||d|| <= Delta` (Steihaug-Toint):
    ///    the iteration stops at the boundary either when a CG iterate would leave the
    ///    region or when negative curvature is encountered.
    /// 2. Compute `rho = (f(x) - f(x + d)) / (m(0) - m(d))`, where
    ///    `m(d) = f(x) + g^T d + 0.5 d^T H d` is the quadratic model.
    /// 3. Update the radius: `rho < 0.25` shrinks `Delta` to `Delta/4`; `rho > 0.75`
    ///    with the step on the boundary grows it to `min(2*Delta, Delta_max)`.
    /// 4. Accept the step only when `rho > eta`; otherwise the parameters are returned
    ///    unchanged and only the radius shrinks.
    ///
    /// The trust region controls the step length, so `learning_rate` is deliberately
    /// **not** applied here. If no radius was configured with
    /// [`NewtonCG::with_trust_region`], one is initialized to `1.0` on the first call.
    ///
    /// Use [`NewtonCG::last_step_accepted`] to find out whether the step was taken and
    /// [`NewtonCG::trust_region_radius`] to read the updated radius.
    ///
    /// # Arguments
    /// - `params`: Current parameter values
    /// - `grads`: Gradient at current parameters
    /// - `hvp_fn`: Function that computes Hessian-vector products
    /// - `loss_fn`: Objective evaluated at a candidate parameter vector
    ///
    /// # Example
    /// ```
    /// use optirs_core::second_order::newton_cg::NewtonCG;
    /// use scirs2_core::ndarray_ext::{array, Array1};
    ///
    /// // f(x) = x0^2 + x1^2, minimum at the origin.
    /// let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 50, 0.0)
    ///     .and_then(|o| o.with_trust_region(1.0))
    ///     .expect("valid configuration");
    /// let mut params = array![3.0, 4.0];
    /// let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };
    /// let loss_fn = |x: &Array1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
    ///
    /// for _ in 0..25 {
    ///     let grads = array![2.0 * params[0], 2.0 * params[1]];
    ///     params = optimizer
    ///         .step_with_loss(params.view(), grads.view(), hvp_fn, loss_fn)
    ///         .expect("step succeeds");
    /// }
    /// assert!(params[0].abs() < 1e-6 && params[1].abs() < 1e-6);
    /// ```
    pub fn step_with_loss<F, L>(
        &mut self,
        params: ArrayView1<T>,
        grads: ArrayView1<T>,
        hvp_fn: F,
        mut loss_fn: L,
    ) -> Result<Array1<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
        L: FnMut(&Array1<T>) -> T,
    {
        let n = params.len();
        if grads.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected gradient size {}, got {}",
                n,
                grads.len()
            )));
        }

        self.step_count += 1;

        let mut delta = match self.trust_region_radius {
            Some(radius) => radius,
            None => T::one(),
        };

        let solution = self.truncated_cg(&grads, &hvp_fn, Some(delta))?;
        let direction = solution.direction;

        // Predicted reduction from the quadratic model:
        //   m(0) - m(d) = -(g^T d + 0.5 d^T H d)
        let g_dot_d = dot(&grads, &direction.view());
        let d_dot_hd = dot(&direction.view(), &solution.hd.view());
        let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
        let predicted_reduction = -(g_dot_d + half * d_dot_hd);

        let quarter = half * half;
        let two = T::one() + T::one();

        if !predicted_reduction.is_finite() || predicted_reduction <= T::zero() {
            // The model predicts no progress: shrink the region and reject.
            delta = (delta * quarter).max(self.min_trust_region_radius);
            self.trust_region_radius = Some(delta);
            self.last_step_accepted = false;
            return Ok(params.to_owned());
        }

        let current = params.to_owned();
        let candidate = &current + &direction;
        let f_current = loss_fn(&current);
        let f_candidate = loss_fn(&candidate);
        let actual_reduction = f_current - f_candidate;

        let rho = if actual_reduction.is_finite() {
            actual_reduction / predicted_reduction
        } else {
            T::neg_infinity()
        };

        let three_quarters = T::from(0.75).unwrap_or_else(|| T::one() - quarter);

        // Radius update.
        if rho < quarter {
            delta = delta * quarter;
        } else if rho > three_quarters && solution.on_boundary {
            delta = (delta * two).min(self.max_trust_region_radius);
        }
        delta = delta
            .max(self.min_trust_region_radius)
            .min(self.max_trust_region_radius);
        self.trust_region_radius = Some(delta);

        // Acceptance test.
        if rho > self.trust_region_eta {
            self.last_step_accepted = true;
            Ok(candidate)
        } else {
            self.last_step_accepted = false;
            Ok(current)
        }
    }

    /// Truncated (Steihaug-Toint) conjugate gradient solver for `(H + reg*I) d = -g`.
    ///
    /// When `radius` is `Some(delta)` the iteration is constrained to `||d|| <= delta`:
    /// as soon as an iterate would leave the region, or negative curvature is detected,
    /// the step to the region boundary along the current CG direction is taken and the
    /// solve stops. When `radius` is `None` the solve is unconstrained, and negative
    /// curvature terminates the iteration at the current iterate (or, if it occurs on
    /// the very first iteration, at the steepest-descent direction `-g`, which is the
    /// standard Newton-CG safeguard).
    fn truncated_cg<F>(
        &self,
        grads: &ArrayView1<T>,
        hvp_fn: &F,
        radius: Option<T>,
    ) -> Result<CgSolution<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let n = grads.len();

        // Initialize CG: d_0 = 0, r_0 = -g, p_0 = r_0
        let mut d: Array1<T> = Array1::zeros(n);
        let mut r = grads.mapv(|x| -x); // residual: r = -g (since A*0 = 0)
        let mut p = r.clone(); // search direction

        let mut r_norm_sq = dot(&r.view(), &r.view());
        let initial_r_norm_sq = r_norm_sq;

        // Convergence is `||r|| <= tol * ||r_0||`. Both sides are squared here so the
        // comparison is dimensionally consistent: `r^T r <= tol^2 * r_0^T r_0`.
        let tol_sq = self.cg_tolerance * self.cg_tolerance;
        let residual_target = tol_sq * initial_r_norm_sq;

        let mut on_boundary = false;
        let tiny = T::from(1e-12).unwrap_or_else(T::epsilon);

        if r_norm_sq <= residual_target {
            let hd = self.hessian_vector_product(hvp_fn, &d)?;
            return Ok(CgSolution {
                direction: d,
                on_boundary: false,
                hd,
            });
        }

        for cg_iter in 0..self.cg_max_iters {
            // Compute regularized Hessian-vector product: Ap = (H + λI)*p
            let ap_reg = self.hessian_vector_product(hvp_fn, &p)?;

            // Curvature along p.
            let p_dot_ap = dot(&p.view(), &ap_reg.view());

            // Negative-curvature check *before* the iterate is updated. Computing
            // alpha = r^T r / p^T Ap with p^T Ap <= 0 yields a negative step length,
            // i.e. a move along +p that increases the model - the iterate must not be
            // contaminated by it.
            if p_dot_ap <= T::zero() {
                match radius {
                    Some(delta) => {
                        // Steihaug: follow p to the trust-region boundary.
                        if let Some(tau) = boundary_tau(&d, &p, delta) {
                            for i in 0..n {
                                d[i] = d[i] + tau * p[i];
                            }
                        }
                        on_boundary = true;
                    }
                    None => {
                        if cg_iter == 0 {
                            // No usable curvature information yet: fall back to the
                            // steepest-descent direction -g (== r_0).
                            d = grads.mapv(|x| -x);
                        }
                    }
                }
                break;
            }

            if p_dot_ap < tiny {
                // Numerically degenerate curvature; stop with the current iterate.
                break;
            }

            let alpha = r_norm_sq / p_dot_ap;

            // Trust-region boundary check on the *candidate* iterate.
            if let Some(delta) = radius {
                let mut d_next_norm_sq = T::zero();
                for i in 0..n {
                    let v = d[i] + alpha * p[i];
                    d_next_norm_sq = d_next_norm_sq + v * v;
                }
                if d_next_norm_sq >= delta * delta {
                    if let Some(tau) = boundary_tau(&d, &p, delta) {
                        for i in 0..n {
                            d[i] = d[i] + tau * p[i];
                        }
                    }
                    on_boundary = true;
                    break;
                }
            }

            // Update solution: d = d + α*p
            for i in 0..n {
                d[i] = d[i] + alpha * p[i];
            }

            // Update residual: r = r - α*Ap
            for i in 0..n {
                r[i] = r[i] - alpha * ap_reg[i];
            }

            let r_norm_sq_new = dot(&r.view(), &r.view());
            if r_norm_sq_new <= residual_target {
                break;
            }

            // Compute β for the new search direction
            let beta = r_norm_sq_new / r_norm_sq;
            r_norm_sq = r_norm_sq_new;

            // Update search direction: p = r + β*p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
        }

        let hd = self.hessian_vector_product(hvp_fn, &d)?;
        Ok(CgSolution {
            direction: d,
            on_boundary,
            hd,
        })
    }

    /// Apply the regularized Hessian `(H + hessian_reg * I)` to `v`.
    fn hessian_vector_product<F>(&self, hvp_fn: &F, v: &Array1<T>) -> Result<Array1<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let n = v.len();
        let v_vec: Vec<T> = v.iter().copied().collect();
        let hv_vec = hvp_fn(&v_vec);

        if hv_vec.len() != n {
            return Err(OptimError::DimensionMismatch(format!(
                "Hessian-vector product returned wrong size: expected {}, got {}",
                n,
                hv_vec.len()
            )));
        }

        let mut hv = Array1::from_vec(hv_vec);
        for i in 0..n {
            hv[i] = hv[i] + self.hessian_reg * v[i];
        }
        Ok(hv)
    }

    /// Get the number of optimization steps performed
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset optimizer state
    ///
    /// Clears the step counter and the acceptance flag. A configured trust-region
    /// radius is kept (it is configuration, not per-run state); call
    /// [`NewtonCG::with_trust_region`] again to reset it explicitly.
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.last_step_accepted = true;
    }

    /// Get current learning rate
    pub fn get_learning_rate(&self) -> T {
        self.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, learning_rate: T) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_newton_cg_creation() {
        let optimizer = NewtonCG::<f32>::default();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_newton_cg_custom_creation() {
        let optimizer = NewtonCG::<f32>::new(0.5, 1e-8, 50, 1e-5)
            .expect("NewtonCG::<f32>::new succeeds in test_newton_cg_custom_creation");
        assert_eq!(optimizer.step_count(), 0);
        assert_relative_eq!(optimizer.get_learning_rate(), 0.5);
    }

    #[test]
    fn test_newton_cg_invalid_params() {
        assert!(NewtonCG::<f32>::new(-0.1, 1e-6, 100, 1e-6).is_err());
        assert!(NewtonCG::<f32>::new(1.0, -1e-6, 100, 1e-6).is_err());
        assert!(NewtonCG::<f32>::new(1.0, 1e-6, 0, 1e-6).is_err());
        assert!(NewtonCG::<f32>::new(1.0, 1e-6, 100, -1e-6).is_err());
    }

    #[test]
    fn test_newton_cg_quadratic_function() {
        // Minimize f(x) = 0.5 * x^T * H * x - b^T * x
        // where H = [[2, 0], [0, 2]] (Hessian)
        // Gradient: g = H*x - b
        // Optimal: x* = H^{-1}*b = 0.5*b

        let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 50, 0.0)
            .expect("NewtonCG::<f64>::new succeeds in test_newton_cg_quadratic_function");

        // Start at x = [2.0, 2.0], b = [1.0, 1.0]
        // Optimal solution: x* = [0.5, 0.5]
        let mut params = array![2.0, 2.0];
        let b = array![1.0, 1.0];

        // Hessian-vector product: H*v where H = [[2, 0], [0, 2]]
        let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };

        // Single Newton-CG step should converge for quadratic function
        let grads = array![
            2.0 * params[0] - b[0], // ∂f/∂x1 = 2*x1 - 1
            2.0 * params[1] - b[1]  // ∂f/∂x2 = 2*x2 - 1
        ];

        params = optimizer
            .step(params.view(), grads.view(), hvp_fn)
            .expect("step succeeds in test_newton_cg_quadratic_function");

        // Should be close to optimal [0.5, 0.5]
        assert_relative_eq!(params[0], 0.5, epsilon = 0.1);
        assert_relative_eq!(params[1], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_newton_cg_convergence() {
        // Minimize f(x, y) = x² + y²
        // Gradient: [2x, 2y]
        // Hessian: [[2, 0], [0, 2]]
        // Optimal: (0, 0)

        let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 100, 0.0)
            .expect("NewtonCG::<f64>::new succeeds in test_newton_cg_convergence");
        let mut params = array![5.0, 5.0];

        // Hessian-vector product for H = [[2, 0], [0, 2]]
        let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };

        for _ in 0..10 {
            let grads = array![2.0 * params[0], 2.0 * params[1]];
            params = optimizer
                .step(params.view(), grads.view(), hvp_fn)
                .expect("step succeeds in test_newton_cg_convergence");
        }

        // Should converge to near zero
        assert!(
            params[0].abs() < 0.01,
            "Failed to converge, got x = {}",
            params[0]
        );
        assert!(
            params[1].abs() < 0.01,
            "Failed to converge, got y = {}",
            params[1]
        );
    }

    #[test]
    fn test_newton_cg_reset() {
        let mut optimizer = NewtonCG::<f32>::default();
        let params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];

        let hvp_fn = |v: &[f32]| -> Vec<f32> { v.to_vec() };

        optimizer
            .step(params.view(), grads.view(), hvp_fn)
            .expect("step succeeds in test_newton_cg_reset");
        assert_eq!(optimizer.step_count(), 1);

        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_newton_cg_learning_rate() {
        let mut optimizer = NewtonCG::<f32>::default();
        assert_relative_eq!(optimizer.get_learning_rate(), 1.0);

        optimizer.set_learning_rate(0.5);
        assert_relative_eq!(optimizer.get_learning_rate(), 0.5);
    }
}
