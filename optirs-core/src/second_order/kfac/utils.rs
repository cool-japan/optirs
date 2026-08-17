// Utility functions for K-FAC optimization
//
// This module contains helper functions and utilities used throughout
// the K-FAC implementation, including layer-specific operations and
// mathematical utilities.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Default Tikhonov damping applied when a matrix is detected as singular during
/// general inversion. This mirrors the K-FAC regularization convention of
/// inverting `(A + λI)` rather than failing outright on a (numerically) singular
/// factor.
pub(crate) const KFAC_SINGULAR_DAMPING: f64 = 1e-8;

/// Invert a general square matrix using Gauss-Jordan elimination with partial
/// pivoting, with K-FAC-style Tikhonov fallback on (near-)singularity.
///
/// # Algorithm
///
/// The routine forms the augmented system `[A | I]` and reduces the left block to
/// the identity via elementary row operations. At each pivot column the row with
/// the largest absolute pivot at or below the diagonal is swapped into place
/// (partial pivoting) for numerical stability; the right block then holds `A⁻¹`.
///
/// # Singular handling
///
/// If, after partial pivoting, the chosen pivot is effectively zero (|pivot| below
/// a scale-aware tolerance) the matrix is treated as singular. Rather than
/// returning a bogus identity, the routine retries ONCE on the Tikhonov-damped
/// matrix `A + λI` (with `λ = `[`KFAC_SINGULAR_DAMPING`]). If the damped system is
/// still singular an [`OptimError::ComputationError`] is returned. This guarantees
/// the result is either a genuine (possibly damped) inverse or an explicit error —
/// never a silent identity.
///
/// Works for any square size `n ≥ 1` over a generic `T: Float`; no external linear
/// algebra dependency is used.
pub(crate) fn general_matrix_inverse<T>(matrix: &Array2<T>) -> Result<Array2<T>>
where
    T: Float,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(crate::error::OptimError::InvalidParameter(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    if n == 0 {
        return Ok(
            Array2::from_shape_vec((0, 0), Vec::new()).unwrap_or_else(|_| Array2::zeros((0, 0)))
        );
    }

    // First attempt: invert A directly.
    match gauss_jordan_inverse(matrix) {
        Ok(inv) => Ok(inv),
        Err(_) => {
            // Singular: retry once on the Tikhonov-damped matrix (A + λI).
            let lambda = T::from(KFAC_SINGULAR_DAMPING).unwrap_or_else(|| T::zero());
            let mut damped = matrix.clone();
            for i in 0..n {
                damped[[i, i]] = damped[[i, i]] + lambda;
            }
            gauss_jordan_inverse(&damped).map_err(|_| {
                crate::error::OptimError::ComputationError(
                    "Matrix is singular even after Tikhonov damping; inverse does not exist"
                        .to_string(),
                )
            })
        }
    }
}

/// Core Gauss-Jordan elimination with partial pivoting.
///
/// Returns `Err(ComputationError)` if a (near-)zero pivot is encountered after
/// pivoting, signalling that the input is numerically singular. The caller is
/// responsible for any damping/regularization retry.
fn gauss_jordan_inverse<T>(matrix: &Array2<T>) -> Result<Array2<T>>
where
    T: Float,
{
    let n = matrix.nrows();

    // Working copy of A and the augmented identity that becomes A^{-1}.
    let mut a = matrix.clone();
    let mut inv: Array2<T> = Array2::eye(n);

    // Scale-aware singularity tolerance: relative to the largest magnitude entry
    // so that the test is invariant to overall matrix scaling.
    let mut max_abs = T::zero();
    for &v in a.iter() {
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    let base_eps = T::from(1e-12).unwrap_or_else(|| T::zero());
    let tol = if max_abs > T::zero() {
        base_eps * max_abs
    } else {
        // All-zero matrix is singular.
        return Err(crate::error::OptimError::ComputationError(
            "Matrix is singular (zero matrix)".to_string(),
        ));
    };

    for col in 0..n {
        // Partial pivoting: find the row >= col with the largest |pivot| in `col`.
        let mut pivot_row = col;
        let mut pivot_mag = a[[col, col]].abs();
        for row in (col + 1)..n {
            let mag = a[[row, col]].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = row;
            }
        }

        if pivot_mag <= tol {
            return Err(crate::error::OptimError::ComputationError(
                "Matrix is singular (zero pivot after partial pivoting)".to_string(),
            ));
        }

        // Swap the pivot row into position in both A and the augmented matrix.
        if pivot_row != col {
            swap_rows(&mut a, col, pivot_row);
            swap_rows(&mut inv, col, pivot_row);
        }

        // Normalize the pivot row so that a[col, col] == 1.
        let pivot = a[[col, col]];
        let inv_pivot = T::one() / pivot;
        for j in 0..n {
            a[[col, j]] = a[[col, j]] * inv_pivot;
            inv[[col, j]] = inv[[col, j]] * inv_pivot;
        }

        // Eliminate the pivot column from every other row.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[[row, col]];
            if factor == T::zero() {
                continue;
            }
            for j in 0..n {
                a[[row, j]] = a[[row, j]] - factor * a[[col, j]];
                inv[[row, j]] = inv[[row, j]] - factor * inv[[col, j]];
            }
        }
    }

    Ok(inv)
}

/// Swap two rows of a matrix in place.
fn swap_rows<T: Float>(matrix: &mut Array2<T>, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }
    let ncols = matrix.ncols();
    for j in 0..ncols {
        let tmp = matrix[[r1, j]];
        matrix[[r1, j]] = matrix[[r2, j]];
        matrix[[r2, j]] = tmp;
    }
}

/// K-FAC utilities for layer-specific operations
pub struct KFACUtils;

impl KFACUtils {
    /// Compute the K-FAC **weight-gradient statistic** for a convolutional layer
    /// from already-extracted patches — the same quantity
    /// [`super::layer_state::KFACLayerState::weight_gradient`] computes for dense
    /// layers. This is the *un-preconditioned* gradient; feed the result through
    /// [`super::core::KFAC::apply_update_weight`] (`G⁻¹ · grad · A⁻¹`) to get the
    /// actual K-FAC parameter update.
    ///
    /// `input_patches` is the im2col-style patch matrix `[samples, in_channels * kh *
    /// kw]` (one row per `(batch element, output spatial location)` pair), optionally
    /// with one extra trailing bias column, and `output_gradients` is the matching
    /// per-location output gradient `[samples, out_channels]`; the two must
    /// therefore share the same row count.
    ///
    /// `kernel_size`, `stride` and `padding` describe how the caller produced
    /// `input_patches` and are accepted as provenance metadata / a light
    /// consistency check on `input_patches`'s column count. This function has no
    /// [`super::config::LayerInfo`] to consult, so — unlike the dense path — it
    /// cannot know whether the caller included a bias column; `kernel_size` is
    /// therefore checked loosely (with or without one trailing bias column) rather
    /// than rejected outright. `stride`/`padding` do not otherwise enter the
    /// computation: the covariance of already-extracted patches doesn't depend on
    /// how they were extracted. This function does **not** perform patch
    /// extraction itself — pass already-extracted patches.
    ///
    /// The result is `output_gradients^T @ input_patches / samples`, the same
    /// `[out_features, in_features]` convention used by
    /// [`super::layer_state::KFACLayerState::weight_gradient`] for dense layers.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::OptimError::DimensionMismatch`] when `input_patches`
    /// and `output_gradients` disagree on the sample count, or
    /// [`crate::error::OptimError::InvalidParameter`] when `input_patches`'s column
    /// count is not `kernel_size.0 * kernel_size.1 * in_channels` for any positive
    /// `in_channels`, with or without one trailing bias column.
    pub fn conv_kfac_update<T: Float + scirs2_core::ndarray::ScalarOperand + 'static>(
        input_patches: &Array2<T>,
        output_gradients: &Array2<T>,
        kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<Array2<T>> {
        let batch_size = input_patches.nrows();
        let input_dim = input_patches.ncols();
        let output_dim = output_gradients.ncols();

        if batch_size != output_gradients.nrows() {
            return Err(crate::error::OptimError::DimensionMismatch(format!(
                "conv_kfac_update: input_patches has {} samples but output_gradients has {}",
                batch_size,
                output_gradients.nrows()
            )));
        }

        let patch_size = kernel_size.0.saturating_mul(kernel_size.1);
        // `input_dim` must be `patch_size * in_channels` for some positive
        // `in_channels` — i.e. a positive multiple of `patch_size` — optionally
        // plus one trailing bias column (`homogeneous_input`'s convention
        // elsewhere in this module; see `KFACLayerState::homogeneous_input`).
        let without_bias_ok =
            patch_size > 0 && input_dim > 0 && input_dim.is_multiple_of(patch_size);
        let with_bias_ok =
            patch_size > 0 && input_dim > 0 && (input_dim - 1).is_multiple_of(patch_size);
        if !(without_bias_ok || with_bias_ok) {
            return Err(crate::error::OptimError::InvalidParameter(format!(
                "conv_kfac_update: input_patches has {input_dim} columns, which is not \
                 kernel_size {}x{} ({patch_size}) times a positive channel count, with or \
                 without a trailing bias column",
                kernel_size.0, kernel_size.1
            )));
        }

        if batch_size == 0 {
            return Ok(Array2::zeros((output_dim, input_dim)));
        }

        // E[grad_output (x) patch], the Kronecker-factored weight-gradient
        // statistic, in the same [out_features, in_features] convention as the
        // dense-layer path (`KFACLayerState::weight_gradient`).
        let scale = T::one() / T::from(batch_size).unwrap_or_else(T::one);
        Ok(output_gradients.t().dot(input_patches) * scale)
    }

    /// Compute batch normalization statistics for K-FAC
    pub fn batchnorm_statistics<T: Float + scirs2_core::numeric::FromPrimitive>(
        input: &Array2<T>,
        eps: T,
    ) -> Result<(Array1<T>, Array1<T>)> {
        let batch_size = input.nrows();
        let num_features = input.ncols();

        if batch_size == 0 {
            return Ok((Array1::zeros(num_features), Array1::ones(num_features)));
        }

        let batch_size_t = T::from(batch_size).unwrap_or_else(|| T::zero());

        // Compute mean (guaranteed Some: batch_size > 0 was checked above)
        let mean = input
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                OptimError::ComputationError(
                    "batchnorm_statistics: mean_axis returned None for a non-empty batch"
                        .to_string(),
                )
            })?;

        // Compute variance
        let mut var = Array1::zeros(num_features);
        for i in 0..num_features {
            let mut sum_sq_diff = T::zero();
            for j in 0..batch_size {
                let diff = input[[j, i]] - mean[i];
                sum_sq_diff = sum_sq_diff + diff * diff;
            }
            var[i] = sum_sq_diff / batch_size_t + eps;
        }

        Ok((mean, var))
    }

    /// Compute K-FAC update for grouped convolution layers
    pub fn grouped_conv_kfac<T: Float + scirs2_core::ndarray::ScalarOperand>(
        input: &Array2<T>,
        gradients: &Array2<T>,
        num_groups: usize,
    ) -> Result<Array2<T>> {
        let batch_size = input.nrows();
        let input_channels = input.ncols();
        let output_channels = gradients.ncols();

        if num_groups == 0 {
            return Err(crate::error::OptimError::InvalidParameter(
                "Number of groups must be positive".to_string(),
            ));
        }

        let input_per_group = input_channels / num_groups;
        let output_per_group = output_channels / num_groups;

        let mut result = Array2::zeros((input_channels, output_channels));

        // Process each group separately
        for group in 0..num_groups {
            let input_start = group * input_per_group;
            let input_end = input_start + input_per_group;
            let output_start = group * output_per_group;
            let output_end = output_start + output_per_group;

            // Extract group data
            let group_input = input.slice(scirs2_core::ndarray::s![.., input_start..input_end]);
            let group_gradients =
                gradients.slice(scirs2_core::ndarray::s![.., output_start..output_end]);

            // Compute group covariance
            let group_update = group_input.t().dot(&group_gradients);

            // Place back in result
            result
                .slice_mut(scirs2_core::ndarray::s![
                    input_start..input_end,
                    output_start..output_end
                ])
                .assign(&group_update);
        }

        // Normalize by batch size
        if batch_size > 0 {
            let scale = T::one() / T::from(batch_size).unwrap_or_else(|| T::zero());
            result = result * scale;
        }

        Ok(result)
    }

    /// Compute eigenvalue-based regularization
    pub fn eigenvalue_regularization<T: Float + Debug + Send + Sync + 'static>(
        matrix: &Array2<T>,
        min_eigenvalue: T,
    ) -> Array2<T> {
        let n = matrix.nrows();
        let mut regularized = matrix.clone();

        // Simple diagonal regularization (in practice, would use proper eigendecomposition)
        for i in 0..n {
            if regularized[[i, i]] < min_eigenvalue {
                regularized[[i, i]] = min_eigenvalue;
            }
        }

        regularized
    }

    /// Compute Kronecker product approximation for two matrices
    pub fn kronecker_product_approx<T: Float + Debug + Send + Sync + 'static>(
        a: &Array2<T>,
        b: &Array2<T>,
    ) -> Array2<T> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));

        for i in 0..a_rows {
            for j in 0..a_cols {
                let a_val = a[[i, j]];
                for k in 0..b_rows {
                    for l in 0..b_cols {
                        result[[i * b_rows + k, j * b_cols + l]] = a_val * b[[k, l]];
                    }
                }
            }
        }

        result
    }

    /// Compute trace of a matrix
    pub fn trace<T: Float + Debug + Send + Sync + 'static>(matrix: &Array2<T>) -> T {
        let n = matrix.nrows().min(matrix.ncols());
        let mut trace = T::zero();

        for i in 0..n {
            trace = trace + matrix[[i, i]];
        }

        trace
    }

    /// Compute Frobenius norm of a matrix
    pub fn frobenius_norm<T: Float + std::iter::Sum>(matrix: &Array2<T>) -> T {
        matrix.iter().map(|&x| x * x).sum::<T>().sqrt()
    }

    /// Check if two matrices are approximately equal
    pub fn matrices_approx_equal<T: Float + Debug + Send + Sync + 'static>(
        a: &Array2<T>,
        b: &Array2<T>,
        tolerance: T,
    ) -> bool {
        if a.dim() != b.dim() {
            return false;
        }

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            if (*a_val - *b_val).abs() > tolerance {
                return false;
            }
        }

        true
    }

    /// Compute running average with exponential decay
    pub fn exponential_moving_average<T: Float + Debug + Send + Sync + 'static>(
        current_value: T,
        new_value: T,
        decay: T,
    ) -> T {
        decay * current_value + (T::one() - decay) * new_value
    }

    /// Clamp eigenvalues to prevent numerical instability
    pub fn clamp_eigenvalues<T: Float + Debug + Send + Sync + 'static>(
        eigenvalues: &mut Array1<T>,
        min_val: T,
        max_val: T,
    ) {
        for eigenval in eigenvalues.iter_mut() {
            *eigenval = (*eigenval).max(min_val).min(max_val);
        }
    }

    /// Compute condition number using singular values (approximation)
    pub fn condition_number_svd_approx<T: Float + Debug + Send + Sync + 'static>(
        matrix: &Array2<T>,
    ) -> T {
        // Simple approximation using diagonal elements
        let diag = matrix.diag();
        let max_diag = diag
            .iter()
            .fold(T::neg_infinity(), |acc, &x| acc.max(x.abs()));
        let min_diag = diag.iter().fold(T::infinity(), |acc, &x| acc.min(x.abs()));

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }

    /// Extract diagonal elements and create diagonal matrix
    pub fn diag_matrix<T: Float + Clone>(diagonal: &Array1<T>) -> Array2<T> {
        let n = diagonal.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            matrix[[i, i]] = diagonal[i];
        }

        matrix
    }

    /// Symmetrize a matrix: (A + A^T) / 2
    pub fn symmetrize<T: Float + Debug + Send + Sync + 'static>(matrix: &Array2<T>) -> Array2<T> {
        let n = matrix.nrows();
        let mut result = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                result[[i, j]] =
                    (matrix[[i, j]] + matrix[[j, i]]) / T::from(2.0).unwrap_or_else(|| T::zero());
            }
        }

        result
    }
}

/// Ordered float wrapper for comparison operations
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat<T: Float + Debug + Send + Sync + 'static>(pub T);

impl<T: Float + Debug + Send + Sync + 'static> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Eq for OrderedFloat<T> {}

impl<T: Float + Debug + Send + Sync + 'static> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `conv_kfac_update` used to reshape its result to
    /// `(kernel_size.0 * kernel_size.1, output_dim)` and fill it via
    /// `i % input_dim` / `j % output_dim` wraparound indexing — a matrix of the
    /// wrong shape holding a value that is not any real statistic of the inputs.
    /// The correct result is the plain outer-product average
    /// `output_gradients^T @ input_patches / batch`, shaped `(output_dim,
    /// input_dim)`, matching `KFACLayerState::weight_gradient`'s convention.
    #[test]
    fn conv_kfac_update_matches_dense_outer_product_convention() {
        // 2 samples, in_channels * kh * kw = 1 * 2 * 2 = 4, out_channels = 3.
        let patches = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("valid shape");
        let grads = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5])
            .expect("valid shape");

        let update = KFACUtils::conv_kfac_update(&patches, &grads, (2, 2), (1, 1), (0, 0))
            .expect("conv_kfac_update should succeed");

        // Shape must be [out_channels, input_dim], not [kh*kw, out_channels].
        assert_eq!(update.dim(), (3, 4));

        let expected = grads.t().dot(&patches) / 2.0;
        for (actual, expected) in update.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn conv_kfac_update_rejects_column_count_inconsistent_with_kernel_size() {
        // 7 columns cannot come from any positive channel count with a 2x2 kernel,
        // with or without a trailing bias column: 7 % 4 == 3 and (7-1) % 4 == 2.
        let patches = Array2::<f64>::zeros((2, 7));
        let grads = Array2::<f64>::zeros((2, 3));

        let result = KFACUtils::conv_kfac_update(&patches, &grads, (2, 2), (1, 1), (0, 0));
        assert!(result.is_err());
    }

    /// Regression test: the column-count check must not reject the bias-augmented
    /// case. `KFACLayerState::homogeneous_input` appends exactly one trailing bias
    /// column elsewhere in this module (see the F12 covariance tests), so
    /// `conv_kfac_update` must accept `in_channels * kh * kw + 1` columns too, not
    /// just an exact multiple of `kh * kw`.
    #[test]
    fn conv_kfac_update_accepts_a_trailing_bias_column() {
        // in_channels * kh * kw = 1 * 2 * 2 = 4, plus one bias column = 5.
        let patches = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 8.0, 1.0],
        )
        .expect("valid shape");
        let grads = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5])
            .expect("valid shape");

        let update = KFACUtils::conv_kfac_update(&patches, &grads, (2, 2), (1, 1), (0, 0))
            .expect("bias-augmented patches should be accepted");
        assert_eq!(update.dim(), (3, 5));

        let expected = grads.t().dot(&patches) / 2.0;
        for (actual, expected) in update.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn conv_kfac_update_rejects_sample_count_mismatch() {
        let patches = Array2::<f64>::zeros((2, 4));
        let grads = Array2::<f64>::zeros((3, 3)); // wrong row count

        let result = KFACUtils::conv_kfac_update(&patches, &grads, (2, 2), (1, 1), (0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn conv_kfac_update_empty_batch_is_zero_of_correct_shape() {
        let patches = Array2::<f64>::zeros((0, 4));
        let grads = Array2::<f64>::zeros((0, 3));

        let update = KFACUtils::conv_kfac_update(&patches, &grads, (2, 2), (1, 1), (0, 0))
            .expect("empty batch should succeed");
        assert_eq!(update.dim(), (3, 4));
        assert!(update.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_trace_computation() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("Array2::from_shape_vec succeeds in test_trace_computation");
        let trace = KFACUtils::trace(&matrix);
        assert!((trace - 15.0).abs() < 1e-10); // 1 + 5 + 9 = 15
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0])
            .expect("Array2::from_shape_vec succeeds in test_frobenius_norm");
        let norm = KFACUtils::frobenius_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_exponential_moving_average() {
        let current = 10.0;
        let new_val = 20.0;
        let decay = 0.9;

        let result = KFACUtils::exponential_moving_average(current, new_val, decay);
        let expected = 0.9 * 10.0 + 0.1 * 20.0; // 9.0 + 2.0 = 11.0
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matrices_approx_equal() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Array2::from_shape_vec succeeds in test_matrices_approx_equal");
        let b = Array2::from_shape_vec((2, 2), vec![1.001, 2.001, 3.001, 4.001])
            .expect("Array2::from_shape_vec succeeds in test_matrices_approx_equal");

        assert!(KFACUtils::matrices_approx_equal(&a, &b, 0.01));
        assert!(!KFACUtils::matrices_approx_equal(&a, &b, 0.0001));
    }

    #[test]
    fn test_symmetrize() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Array2::from_shape_vec succeeds in test_symmetrize");
        let symmetric = KFACUtils::symmetrize(&matrix);

        assert!((symmetric[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((symmetric[[0, 1]] - 2.5).abs() < 1e-10); // (2 + 3) / 2
        assert!((symmetric[[1, 0]] - 2.5).abs() < 1e-10); // (3 + 2) / 2
        assert!((symmetric[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diag_matrix() {
        let diagonal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let matrix = KFACUtils::diag_matrix(&diagonal);

        assert_eq!(matrix.dim(), (3, 3));
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((matrix[[1, 1]] - 2.0).abs() < 1e-10);
        assert!((matrix[[2, 2]] - 3.0).abs() < 1e-10);
        assert!((matrix[[0, 1]]).abs() < 1e-10); // Off-diagonal should be zero
    }

    #[test]
    fn test_ordered_float() {
        let a = OrderedFloat(1.5);
        let b = OrderedFloat(2.5);
        let c = OrderedFloat(1.5);

        assert!(a < b);
        assert!(a == c);
        assert!(b > a);
    }

    #[test]
    fn test_batchnorm_statistics() {
        let input = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Array2::from_shape_vec succeeds in test_batchnorm_statistics");

        let (mean, var) = KFACUtils::batchnorm_statistics(&input, 1e-8)
            .expect("KFACUtils::batchnorm_statistics succeeds in test_batchnorm_statistics");

        // Expected mean: [4.0, 5.0] (column-wise average)
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 5.0).abs() < 1e-6);

        // Variance should be positive
        assert!(var[0] > 0.0);
        assert!(var[1] > 0.0);
    }

    // ---- General matrix inversion (Gauss-Jordan with partial pivoting) ----

    use approx::assert_abs_diff_eq;

    /// Assert that `A · inv(A) ≈ I` to the given epsilon.
    fn assert_is_inverse(a: &Array2<f64>, inv: &Array2<f64>, eps: f64) {
        let n = a.nrows();
        let product = a.dot(inv);
        let identity: Array2<f64> = Array2::eye(n);
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(product[[i, j]], identity[[i, j]], epsilon = eps);
            }
        }
    }

    #[test]
    fn test_general_inverse_known_2x2() {
        // Hand-chosen matrix with a known inverse.
        // A = [[4, 7], [2, 6]]  =>  inv(A) = [[0.6, -0.7], [-0.2, 0.4]]
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).expect("shape");
        let inv = general_matrix_inverse(&a).expect("invertible");

        assert_abs_diff_eq!(inv[[0, 0]], 0.6, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[[0, 1]], -0.7, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[[1, 0]], -0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[[1, 1]], 0.4, epsilon = 1e-12);
    }

    #[test]
    fn test_general_inverse_4x4_spd() {
        // 4x4 SPD matrix built as M = B^T B + I (guaranteed positive definite).
        let b = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.5, -0.3, 0.2, 0.0, 1.2, 0.7, -0.4, 0.3, -0.1, 0.9, 0.6, -0.2, 0.4, 0.1, 1.1,
            ],
        )
        .expect("shape");
        let mut spd = b.t().dot(&b);
        for i in 0..4 {
            spd[[i, i]] += 1.0;
        }

        let inv = general_matrix_inverse(&spd).expect("invertible");
        assert_is_inverse(&spd, &inv, 1e-6);
    }

    #[test]
    fn test_general_inverse_8x8_spd() {
        // 8x8 SPD matrix M = B^T B + 2I with deterministic, well-conditioned data.
        let n = 8usize;
        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                // A smooth, non-degenerate pattern.
                let v = ((i as f64 + 1.0) * 0.3 - (j as f64) * 0.17).sin()
                    + 0.05 * (i as f64 - j as f64);
                b[[i, j]] = v;
            }
        }
        let mut spd = b.t().dot(&b);
        for i in 0..n {
            spd[[i, i]] += 2.0;
        }

        let inv = general_matrix_inverse(&spd).expect("invertible");
        assert_is_inverse(&spd, &inv, 1e-6);
    }

    #[test]
    fn test_general_inverse_nonsymmetric() {
        // A general (non-symmetric) invertible matrix; partial pivoting is needed
        // because the (0,0) entry is zero.
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 2.0, 1.0, 3.0, 4.0, 1.0, 0.0, 2.0, 1.0, 5.0, 3.0, 0.0, 2.0, 1.0, 4.0, 1.0,
            ],
        )
        .expect("shape");
        let inv = general_matrix_inverse(&a).expect("invertible");
        assert_is_inverse(&a, &inv, 1e-6);

        // inv(A) · A ≈ I as well (left inverse).
        let left = inv.dot(&a);
        let identity: Array2<f64> = Array2::eye(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(left[[i, j]], identity[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_general_inverse_near_singular_damps_to_finite() {
        // A genuinely singular matrix (row 2 = 2 * row 0). The damping path inverts
        // (A + λI) and must yield a finite, well-defined result (no NaN/Inf).
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0])
            .expect("shape");
        let inv = general_matrix_inverse(&a).expect("damping fallback should succeed");
        for &v in inv.iter() {
            assert!(v.is_finite(), "damped inverse contains non-finite entry");
        }
    }

    #[test]
    fn test_general_inverse_not_identity_regression() {
        // Regression guard for the old bug where the inverse silently returned the
        // identity. For a non-identity input the inverse must NOT equal the input's
        // identity-shaped matrix.
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            ],
        )
        .expect("shape");
        let inv = general_matrix_inverse(&a).expect("invertible");
        let identity: Array2<f64> = Array2::eye(4);
        assert!(
            !KFACUtils::matrices_approx_equal(&inv, &identity, 1e-9),
            "inverse of a non-identity matrix must not be the identity"
        );
        // And it must be a real inverse.
        assert_is_inverse(&a, &inv, 1e-6);
    }

    #[test]
    fn test_general_inverse_non_square_errors() {
        let a = Array2::<f64>::zeros((2, 3));
        assert!(general_matrix_inverse(&a).is_err());
    }
}
