// Natural gradient optimization using Fisher information matrix
//
// This module provides natural gradient computation methods that work with
// K-FAC to compute Fisher information matrix approximations and apply
// natural gradient updates.

use crate::error::Result;
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Natural gradient optimizer configuration
#[derive(Debug, Clone)]
pub struct NaturalGradientConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Learning rate for natural gradients
    pub learning_rate: T,

    /// Damping parameter for Fisher information matrix
    pub fisher_damping: T,

    /// Update frequency for Fisher information matrix
    pub fisher_update_freq: usize,

    /// Use empirical Fisher information (vs true Fisher)
    pub use_empirical_fisher: bool,

    /// Maximum rank for low-rank Fisher approximation
    pub max_rank: Option<usize>,

    /// Enable adaptive damping
    pub adaptive_damping: bool,

    /// Use conjugate gradient for matrix inversion
    pub use_conjugate_gradient: bool,

    /// Maximum CG iterations
    pub max_cg_iterations: usize,

    /// CG convergence tolerance
    pub cg_tolerance: T,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for NaturalGradientConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).unwrap_or_else(|| T::zero()),
            fisher_damping: T::from(0.001).unwrap_or_else(|| T::zero()),
            fisher_update_freq: 10,
            use_empirical_fisher: true,
            max_rank: Some(100),
            adaptive_damping: true,
            use_conjugate_gradient: true,
            max_cg_iterations: 100,
            cg_tolerance: T::from(1e-6).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Natural gradient computation utilities
pub struct NaturalGradientCompute;

impl NaturalGradientCompute {
    /// Simplified condition number estimation for static methods
    pub fn estimate_condition_simple<T>(matrix: &Array2<T>) -> T
    where
        T: Float,
    {
        // Simple condition number estimate using ratio of max/min diagonal elements
        let diag = matrix.diag();
        let max_diag = diag.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let min_diag = diag.iter().fold(T::infinity(), |acc, &x| acc.min(x));

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::from(1e12).unwrap_or_else(|| T::zero()) // Large condition number for singular matrices
        }
    }

    /// Robust static matrix inverse.
    ///
    /// Routes through the shared Gauss-Jordan-with-partial-pivoting routine in
    /// `super::utils::general_matrix_inverse`, which applies K-FAC-style Tikhonov
    /// damping if the input is (near-)singular and returns an explicit error if the
    /// damped system is still singular. This replaces the old behavior that
    /// returned a regularized identity for `n > 3` (silently discarding the matrix).
    pub fn safe_matrix_inverse_static<T>(matrix: &Array2<T>) -> Result<Array2<T>>
    where
        T: Float + Default,
    {
        super::utils::general_matrix_inverse(matrix)
    }

    /// Conjugate gradient method for solving Ax = b
    pub fn conjugate_gradient<T>(
        a: &Array2<T>,
        b: &Array2<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<Array2<T>>
    where
        T: Float + std::iter::Sum + scirs2_core::ndarray::ScalarOperand,
    {
        let n = a.nrows();
        if n != a.ncols() || n != b.nrows() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Dimension mismatch in conjugate gradient".to_string(),
            ));
        }

        let mut x = Array2::zeros(b.raw_dim());
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rsold = Self::frobenius_inner_product(&r, &r);

        for _ in 0..max_iterations {
            let ap = a.dot(&p);
            let pap = Self::frobenius_inner_product(&p, &ap);

            if pap <= T::zero() {
                return Err(crate::error::OptimError::ComputationError(
                    "Matrix is not positive definite".to_string(),
                ));
            }

            let alpha = rsold / pap;

            x = x + &(&p * alpha);
            r = r - &(&ap * alpha);

            let rsnew = Self::frobenius_inner_product(&r, &r);

            if rsnew.sqrt() < tolerance {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;
        }

        Ok(x)
    }

    /// Compute Frobenius inner product of two matrices
    fn frobenius_inner_product<T>(a: &Array2<T>, b: &Array2<T>) -> T
    where
        T: Float + std::iter::Sum,
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Apply regularization to a matrix
    pub fn regularize_matrix<T>(matrix: &mut Array2<T>, damping: T)
    where
        T: Float,
    {
        let n = matrix.nrows().min(matrix.ncols());
        for i in 0..n {
            matrix[[i, i]] = matrix[[i, i]] + damping;
        }
    }

    /// Check if matrix is positive definite (approximately)
    pub fn is_positive_definite<T>(matrix: &Array2<T>) -> bool
    where
        T: Float,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return false;
        }

        // Simple check: all diagonal elements should be positive
        // and dominant over off-diagonal elements
        for i in 0..n {
            let diag = matrix[[i, i]];
            if diag <= T::zero() {
                return false;
            }

            let mut row_sum = T::zero();
            for j in 0..n {
                if i != j {
                    row_sum = row_sum + matrix[[i, j]].abs();
                }
            }

            // Diagonal dominance check
            if diag < row_sum {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_gradient_config() {
        let config = NaturalGradientConfig::<f32>::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.fisher_damping >= 0.0);
        assert!(config.max_cg_iterations > 0);
    }

    #[test]
    fn test_condition_number_estimation() {
        let matrix: Array2<f64> = Array2::eye(3);
        let condition = NaturalGradientCompute::estimate_condition_simple(&matrix);
        assert!((condition - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_2x2_matrix_inverse() {
        let matrix: Array2<f64> = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 1.0, 3.0])
            .expect("Array2::from_shape_vec succeeds in test_2x2_matrix_inverse");
        let inverse =
            NaturalGradientCompute::safe_matrix_inverse_static(&matrix).expect("NaturalGradientCompute::safe_matrix_inverse_static succeeds in test_2x2_matrix_inverse");

        // Check that A * A^{-1} ≈ I
        let product = matrix.dot(&inverse);
        let identity: Array2<f64> = Array2::eye(2);

        for i in 0..2 {
            for j in 0..2 {
                let diff = (product[[i, j]] - identity[[i, j]]).abs();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_3x3_matrix_inverse() {
        let matrix: Array2<f64> =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0])
                .expect("Array2::from_shape_vec succeeds in test_3x3_matrix_inverse");

        let inverse =
            NaturalGradientCompute::safe_matrix_inverse_static(&matrix).expect("NaturalGradientCompute::safe_matrix_inverse_static succeeds in test_3x3_matrix_inverse");

        // Check that A * A^{-1} ≈ I
        let product = matrix.dot(&inverse);
        let identity: Array2<f64> = Array2::eye(3);

        for i in 0..3 {
            for j in 0..3 {
                let diff = (product[[i, j]] - identity[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Position ({}, {}): {} vs {}",
                    i,
                    j,
                    product[[i, j]],
                    identity[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_positive_definite_check() {
        // Positive definite matrix
        let pd_matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0])
            .expect("Array2::from_shape_vec succeeds in test_positive_definite_check");
        assert!(NaturalGradientCompute::is_positive_definite(&pd_matrix));

        // Not positive definite matrix
        let npd_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0])
            .expect("Array2::from_shape_vec succeeds in test_positive_definite_check");
        assert!(!NaturalGradientCompute::is_positive_definite(&npd_matrix));
    }

    #[test]
    fn test_regularization() {
        let mut matrix = Array2::zeros((3, 3));
        let damping = 0.1;

        NaturalGradientCompute::regularize_matrix(&mut matrix, damping);

        for i in 0..3 {
            assert!((matrix[[i, i]] - damping).abs() < 1e-10);
        }
    }
}
