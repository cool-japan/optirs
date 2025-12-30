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
use scirs2_core::numeric::{Float, Zero};
use serde::{Deserialize, Serialize};

/// Newton-CG optimizer configuration
///
/// Newton-CG uses conjugate gradient to solve the Newton system H*d = -g,
/// where H is the Hessian and g is the gradient. This avoids explicitly
/// computing and storing the full Hessian matrix.
///
/// # Key Features
/// - Memory-efficient: Only needs Hessian-vector products
/// - Suitable for large-scale problems
/// - Automatic step size control via trust region
/// - Conjugate gradient with early termination
///
/// # Type Parameters
/// - `T`: Floating-point type (f32 or f64)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtonCG<T: Float> {
    /// Learning rate / trust region radius
    learning_rate: T,

    /// CG termination tolerance
    cg_tolerance: T,

    /// Maximum CG iterations
    cg_max_iters: usize,

    /// Hessian regularization for numerical stability
    hessian_reg: T,

    /// Step counter
    step_count: usize,
}

impl<T: Float + ScalarOperand> Default for NewtonCG<T> {
    fn default() -> Self {
        Self::new(
            T::from(1.0).unwrap(),  // learning_rate
            T::from(1e-6).unwrap(), // cg_tolerance
            100,                    // cg_max_iters
            T::from(1e-6).unwrap(), // hessian_reg
        )
        .unwrap()
    }
}

impl<T: Float + ScalarOperand> NewtonCG<T> {
    /// Create a new Newton-CG optimizer
    ///
    /// # Arguments
    /// - `learning_rate`: Trust region radius / step size (typically 0.5-2.0)
    /// - `cg_tolerance`: CG convergence tolerance (typically 1e-6)
    /// - `cg_max_iters`: Maximum CG iterations (typically 50-200)
    /// - `hessian_reg`: Hessian regularization (typically 1e-6)
    ///
    /// # Example
    /// ```
    /// use optirs_core::second_order::newton_cg::NewtonCG;
    ///
    /// let optimizer = NewtonCG::<f32>::new(1.0, 1e-6, 100, 1e-6).unwrap();
    /// ```
    pub fn new(
        learning_rate: T,
        cg_tolerance: T,
        cg_max_iters: usize,
        hessian_reg: T,
    ) -> Result<Self> {
        if learning_rate.to_f64().unwrap() <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "learning_rate must be positive, got {}",
                learning_rate.to_f64().unwrap()
            )));
        }
        if cg_tolerance.to_f64().unwrap() <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "cg_tolerance must be positive, got {}",
                cg_tolerance.to_f64().unwrap()
            )));
        }
        if cg_max_iters == 0 {
            return Err(OptimError::InvalidParameter(
                "cg_max_iters must be positive".to_string(),
            ));
        }
        if hessian_reg.to_f64().unwrap() < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "hessian_reg must be non-negative, got {}",
                hessian_reg.to_f64().unwrap()
            )));
        }

        Ok(Self {
            learning_rate,
            cg_tolerance,
            cg_max_iters,
            hessian_reg,
            step_count: 0,
        })
    }

    /// Perform a Newton-CG optimization step
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
    /// let updated = optimizer.step(params.view(), grads.view(), hvp_fn).unwrap();
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

        // Solve H*d = -g using conjugate gradient
        // where H is the Hessian and g is the gradient
        let direction = self.conjugate_gradient(&grads, hvp_fn)?;

        // Update parameters: x_{t+1} = x_t + α*d
        Ok(params.to_owned() + &(direction * self.learning_rate))
    }

    /// Conjugate gradient solver for H*d = -g
    ///
    /// Uses the conjugate gradient method to solve the Newton system
    /// without explicitly forming the Hessian matrix.
    fn conjugate_gradient<F>(&self, grads: &ArrayView1<T>, hvp_fn: F) -> Result<Array1<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let n = grads.len();

        // Initialize CG: d_0 = 0, r_0 = -g, p_0 = r_0
        let mut d = Array1::zeros(n);
        let mut r = grads.mapv(|x| -x); // residual: r = -g (since Ad = -g, A*0 = 0)
        let mut p = r.clone(); // search direction

        let mut r_norm_sq = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
        let initial_r_norm_sq = r_norm_sq;

        // CG iterations
        for cg_iter in 0..self.cg_max_iters {
            // Check convergence: ||r|| < tol * ||r_0||
            if r_norm_sq < self.cg_tolerance * initial_r_norm_sq {
                break;
            }

            // Compute Hessian-vector product: Ap = H*p
            let p_vec: Vec<T> = p.iter().copied().collect();
            let ap_vec = hvp_fn(&p_vec);

            if ap_vec.len() != n {
                return Err(OptimError::DimensionMismatch(format!(
                    "Hessian-vector product returned wrong size: expected {}, got {}",
                    n,
                    ap_vec.len()
                )));
            }

            let ap = Array1::from_vec(ap_vec);

            // Add regularization: Ap = (H + λI)*p
            let ap_reg = ap + &(p.mapv(|x| x * self.hessian_reg));

            // Compute step size: α = r^T*r / (p^T*Ap)
            let p_dot_ap = p
                .iter()
                .zip(ap_reg.iter())
                .map(|(&pi, &api)| pi * api)
                .fold(T::zero(), |acc, x| acc + x);

            if p_dot_ap.abs() < T::from(1e-12).unwrap() {
                // Numerical issue: p^T*Ap ≈ 0
                break;
            }

            let alpha = r_norm_sq / p_dot_ap;

            // Update solution: d = d + α*p
            for i in 0..n {
                d[i] = d[i] + alpha * p[i];
            }

            // Update residual: r = r - α*Ap
            for i in 0..n {
                r[i] = r[i] - alpha * ap_reg[i];
            }

            // Compute new residual norm squared
            let r_norm_sq_new = r.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);

            // Compute β for new search direction
            let beta = r_norm_sq_new / r_norm_sq;
            r_norm_sq = r_norm_sq_new;

            // Update search direction: p = r + β*p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            // Negative curvature check: if p^T*Ap < 0, stop
            if p_dot_ap < T::zero() {
                break;
            }
        }

        Ok(d)
    }

    /// Get the number of optimization steps performed
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.step_count = 0;
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
        let optimizer = NewtonCG::<f32>::new(0.5, 1e-8, 50, 1e-5).unwrap();
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

        let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 50, 0.0).unwrap();

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

        params = optimizer.step(params.view(), grads.view(), hvp_fn).unwrap();

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

        let mut optimizer = NewtonCG::<f64>::new(1.0, 1e-8, 100, 0.0).unwrap();
        let mut params = array![5.0, 5.0];

        // Hessian-vector product for H = [[2, 0], [0, 2]]
        let hvp_fn = |v: &[f64]| -> Vec<f64> { vec![2.0 * v[0], 2.0 * v[1]] };

        for _ in 0..10 {
            let grads = array![2.0 * params[0], 2.0 * params[1]];
            params = optimizer.step(params.view(), grads.view(), hvp_fn).unwrap();
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

        optimizer.step(params.view(), grads.view(), hvp_fn).unwrap();
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
