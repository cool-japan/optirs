// K-FAC layer state management
//
// This module contains the layer-specific state for K-FAC optimization,
// including covariance matrices and their inverses.

use super::config::LayerInfo;
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// K-FAC optimizer state for a single layer
#[derive(Debug, Clone)]
pub struct KFACLayerState<T: Float + Debug + Send + Sync + 'static> {
    /// Input covariance matrix A = E[a a^T]
    pub a_cov: Array2<T>,

    /// Output gradient covariance matrix G = E[g g^T]
    pub g_cov: Array2<T>,

    /// Inverse of input covariance matrix
    pub a_cov_inv: Option<Array2<T>>,

    /// Inverse of output gradient covariance matrix
    pub g_cov_inv: Option<Array2<T>>,

    /// Number of updates performed
    pub num_updates: usize,

    /// Last update step for covariance matrices
    pub last_cov_update: usize,

    /// Last update step for inverse matrices
    pub last_inv_update: usize,

    /// Damping values for this layer
    pub damping_a: T,
    pub damping_g: T,

    /// Layer information
    pub layerinfo: LayerInfo,

    /// Precomputed Kronecker factors for bias
    pub bias_correction: Option<Array1<T>>,

    /// Moving average statistics
    pub running_mean_a: Option<Array1<T>>,
    pub running_mean_g: Option<Array1<T>>,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + scirs2_core::numeric::FromPrimitive,
    > KFACLayerState<T>
{
    /// Create a new layer state for the given layer
    pub fn new(layer_info: LayerInfo, initial_damping: T) -> Self {
        let input_size = layer_info.input_cov_size();
        let output_size = layer_info.output_cov_size();

        Self {
            a_cov: Array2::eye(input_size),
            g_cov: Array2::eye(output_size),
            a_cov_inv: None,
            g_cov_inv: None,
            num_updates: 0,
            last_cov_update: 0,
            last_inv_update: 0,
            damping_a: initial_damping,
            damping_g: initial_damping,
            layerinfo: layer_info,
            bias_correction: None,
            running_mean_a: None,
            running_mean_g: None,
        }
    }

    /// Initialize moving average statistics
    pub fn init_running_stats(&mut self) {
        let input_size = self.layerinfo.input_cov_size();
        let output_size = self.layerinfo.output_cov_size();

        self.running_mean_a = Some(Array1::zeros(input_size));
        self.running_mean_g = Some(Array1::zeros(output_size));
    }

    /// Update the input covariance matrix with new activations.
    ///
    /// `activations` has shape `[batch, input_dim]`. When the layer carries a bias, a
    /// homogeneous column of ones is appended so the bias is folded into the Kronecker
    /// factor `A`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::DimensionMismatch`] if the activation width does not match
    /// the layer's registered input covariance size.
    pub fn update_input_covariance(&mut self, activations: &Array2<T>, decay: T) -> Result<()> {
        let batch_size = activations.nrows();
        if batch_size == 0 {
            return Ok(());
        }

        // Add bias term if needed
        let input_data = self.homogeneous_input(activations)?;

        // Compute the uncentered second moment
        let batch_cov = Self::second_moment(&input_data);

        // Update running covariance with exponential moving average
        self.a_cov = &self.a_cov * decay + &batch_cov * (T::one() - decay);
        self.num_updates += 1;
        Ok(())
    }

    /// Update the output gradient covariance matrix.
    ///
    /// `gradients` has shape `[batch, output_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::DimensionMismatch`] if the gradient width does not match
    /// the layer's registered output covariance size.
    pub fn update_output_covariance(&mut self, gradients: &Array2<T>, decay: T) -> Result<()> {
        let batch_size = gradients.nrows();
        if batch_size == 0 {
            return Ok(());
        }

        let expected = self.layerinfo.output_cov_size();
        if gradients.ncols() != expected {
            return Err(OptimError::DimensionMismatch(format!(
                "layer '{}': output gradients have {} columns, expected {}",
                self.layerinfo.name,
                gradients.ncols(),
                expected
            )));
        }

        // Compute the uncentered second moment
        let batch_cov = Self::second_moment(gradients);

        // Update running covariance with exponential moving average
        self.g_cov = &self.g_cov * decay + &batch_cov * (T::one() - decay);
        Ok(())
    }

    /// Build the dense-layer weight gradient `[output_dim, input_cov_size]` from
    /// per-sample activations `[batch, input_dim]` and output gradients
    /// `[batch, output_dim]`.
    ///
    /// This is `grad_W = (1/n) * G^T · A_hom`, matching the `A ⊗ G` Kronecker
    /// factorization: the resulting matrix is exactly what
    /// [`super::core::KFAC::apply_update_weight`] preconditions.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::DimensionMismatch`] if the batch dimensions disagree or
    /// either width does not match the registered layer dimensions.
    pub fn weight_gradient(
        &self,
        activations: &Array2<T>,
        output_gradients: &Array2<T>,
    ) -> Result<Array2<T>> {
        let batch = activations.nrows();
        if batch != output_gradients.nrows() {
            return Err(OptimError::DimensionMismatch(format!(
                "layer '{}': activations have {} rows but output gradients have {}",
                self.layerinfo.name,
                batch,
                output_gradients.nrows()
            )));
        }

        let expected_out = self.layerinfo.output_cov_size();
        if output_gradients.ncols() != expected_out {
            return Err(OptimError::DimensionMismatch(format!(
                "layer '{}': output gradients have {} columns, expected {}",
                self.layerinfo.name,
                output_gradients.ncols(),
                expected_out
            )));
        }

        let input_data = self.homogeneous_input(activations)?;

        if batch == 0 {
            return Ok(Array2::zeros((expected_out, input_data.ncols())));
        }

        let scale = T::from_usize(batch).unwrap_or_else(T::one);
        Ok(output_gradients.t().dot(&input_data) / scale)
    }

    /// Compute the inverse of covariance matrices with regularization
    pub fn compute_inverses(&mut self, damping_a: T, damping_g: T) -> Result<()> {
        self.damping_a = damping_a;
        self.damping_g = damping_g;

        // Compute regularized inverse of input covariance
        let mut a_reg = self.a_cov.clone();
        for i in 0..a_reg.nrows() {
            a_reg[[i, i]] = a_reg[[i, i]] + damping_a;
        }

        self.a_cov_inv = Some(self.compute_matrix_inverse(&a_reg)?);

        // Compute regularized inverse of output gradient covariance
        let mut g_reg = self.g_cov.clone();
        for i in 0..g_reg.nrows() {
            g_reg[[i, i]] = g_reg[[i, i]] + damping_g;
        }

        self.g_cov_inv = Some(self.compute_matrix_inverse(&g_reg)?);

        self.last_inv_update = self.num_updates;
        Ok(())
    }

    /// Get the condition number estimate of covariance matrices
    pub fn condition_number_estimate(&self) -> (T, T) {
        let a_cond = self.estimate_condition_number(&self.a_cov);
        let g_cond = self.estimate_condition_number(&self.g_cov);
        (a_cond, g_cond)
    }

    /// Check if the layer state is ready for optimization (inverses computed)
    pub fn is_ready(&self) -> bool {
        self.a_cov_inv.is_some() && self.g_cov_inv.is_some()
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let float_size = std::mem::size_of::<T>();
        let mut size = 0;

        // Covariance matrices
        size += self.a_cov.len() * float_size;
        size += self.g_cov.len() * float_size;

        // Inverse matrices
        if let Some(ref inv) = self.a_cov_inv {
            size += inv.len() * float_size;
        }
        if let Some(ref inv) = self.g_cov_inv {
            size += inv.len() * float_size;
        }

        // Running statistics
        if let Some(ref mean) = self.running_mean_a {
            size += mean.len() * float_size;
        }
        if let Some(ref mean) = self.running_mean_g {
            size += mean.len() * float_size;
        }

        // Bias correction
        if let Some(ref bias) = self.bias_correction {
            size += bias.len() * float_size;
        }

        size
    }

    /// Reset the layer state
    pub fn reset(&mut self) {
        let input_size = self.layerinfo.input_cov_size();
        let output_size = self.layerinfo.output_cov_size();

        self.a_cov = Array2::eye(input_size);
        self.g_cov = Array2::eye(output_size);
        self.a_cov_inv = None;
        self.g_cov_inv = None;
        self.num_updates = 0;
        self.last_cov_update = 0;
        self.last_inv_update = 0;
        self.bias_correction = None;

        if self.running_mean_a.is_some() {
            self.running_mean_a = Some(Array1::zeros(input_size));
        }
        if self.running_mean_g.is_some() {
            self.running_mean_g = Some(Array1::zeros(output_size));
        }
    }

    // Private helper methods

    fn add_bias_column(&self, activations: &Array2<T>) -> Array2<T> {
        let (batch_size, input_dim) = activations.dim();
        let mut result = Array2::ones((batch_size, input_dim + 1));
        result.slice_mut(s![.., ..input_dim]).assign(activations);
        result
    }

    /// Return the activations in the layout the Kronecker factor `A` expects,
    /// appending the homogeneous bias column when the registered covariance size
    /// calls for it.
    fn homogeneous_input(&self, activations: &Array2<T>) -> Result<Array2<T>> {
        let expected = self.layerinfo.input_cov_size();
        let width = activations.ncols();

        if expected == width {
            Ok(activations.clone())
        } else if expected == width + 1 {
            Ok(self.add_bias_column(activations))
        } else {
            Err(OptimError::DimensionMismatch(format!(
                "layer '{}': activations have {} columns, expected {} (or {} plus the bias column)",
                self.layerinfo.name,
                width,
                expected,
                expected.saturating_sub(1)
            )))
        }
    }

    /// Uncentered second moment `E[x x^T] = (1/n) * X^T X`.
    ///
    /// K-FAC's Kronecker factors are second moments, **not** mean-centered
    /// covariances. Centering is wrong twice over here: it discards the mean, which
    /// carries genuine curvature information in the Fisher approximation, and it zeroes
    /// out the homogeneous bias column (whose entries are all ones), making `A`
    /// structurally singular for every layer with a bias.
    fn second_moment(data: &Array2<T>) -> Array2<T> {
        let batch_size = data.nrows();
        if batch_size == 0 {
            return Array2::eye(data.ncols());
        }

        let n = T::from_usize(batch_size).unwrap_or_else(T::one);
        data.t().dot(data) / n
    }

    fn compute_matrix_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        // Robust general inversion: Gauss-Jordan elimination with partial pivoting
        // and K-FAC-style Tikhonov damping on (near-)singular inputs. Implemented
        // once in `kfac::utils` and shared with the natural-gradient path so the
        // Kronecker-factor inverses are real (not a silent identity).
        if matrix.nrows() != matrix.ncols() {
            return Err(OptimError::InvalidParameter(
                "Matrix must be square".to_string(),
            ));
        }

        super::utils::general_matrix_inverse(matrix)
    }

    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::second_order::kfac::config::{LayerInfo, LayerType};

    #[test]
    fn test_layer_state_creation() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 128,
            output_dim: 64,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        let state = KFACLayerState::<f32>::new(layer_info, 0.001);

        assert_eq!(state.a_cov.nrows(), 129); // +1 for bias
        assert_eq!(state.g_cov.nrows(), 64);
        assert!(!state.is_ready()); // No inverses computed yet
    }

    #[test]
    fn test_covariance_update() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 4,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        let mut state = KFACLayerState::<f64>::new(layer_info, 0.001);
        let activations =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .expect("Array2::from_shape_vec succeeds in test_covariance_update");

        state
            .update_input_covariance(&activations, 0.95)
            .expect("covariance update succeeds");

        assert_eq!(state.num_updates, 1);
        assert!(state.a_cov[[0, 0]] != 1.0); // Should have changed from identity
    }

    #[test]
    fn test_input_covariance_is_uncentered_and_keeps_bias_column() {
        // Regression for the mean-centered covariance bug: K-FAC's Kronecker factor is
        // the *uncentered* second moment E[a a^T]. Centering zeroes the homogeneous
        // bias column (all ones), which makes A structurally singular.
        let layer_info = LayerInfo {
            name: "dense".to_string(),
            input_dim: 2,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: true,
        };
        let mut state = KFACLayerState::<f64>::new(layer_info, 0.0);

        // Batch of 3 samples, 2 input features (full rank once homogenized).
        let activations = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
            .expect("shape is valid");

        // decay = 0 so a_cov is exactly the batch statistic.
        state
            .update_input_covariance(&activations, 0.0)
            .expect("covariance update succeeds");

        assert_eq!(state.a_cov.dim(), (3, 3));

        // Expected A = (1/3) * Â^T Â with Â = [[1,2,1],[3,4,1],[5,7,1]].
        let expected = [
            [35.0 / 3.0, 49.0 / 3.0, 3.0],
            [49.0 / 3.0, 23.0, 13.0 / 3.0],
            [3.0, 13.0 / 3.0, 1.0],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (j, &exp) in row.iter().enumerate() {
                assert!(
                    (state.a_cov[[i, j]] - exp).abs() < 1e-12,
                    "A[{},{}] = {}, expected {}",
                    i,
                    j,
                    state.a_cov[[i, j]],
                    exp
                );
            }
        }

        // The bias block must be E[1*1] = 1, not the 0 that centering produces.
        assert!((state.a_cov[[2, 2]] - 1.0).abs() < 1e-12);

        // And the factor must be invertible (centering made it singular).
        state
            .compute_inverses(0.0, 0.0)
            .expect("uncentered A is invertible");
        let a_inv = state.a_cov_inv.as_ref().expect("a_inv present");
        let prod = state.a_cov.dot(a_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected_entry = if i == j { 1.0 } else { 0.0 };
                assert!((prod[[i, j]] - expected_entry).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_weight_gradient_shape_and_value() {
        let layer_info = LayerInfo {
            name: "dense".to_string(),
            input_dim: 2,
            output_dim: 3,
            layer_type: LayerType::Dense,
            has_bias: false,
        };
        let state = KFACLayerState::<f64>::new(layer_info, 0.001);

        // batch = 2
        let activations =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape is valid");
        let out_grads = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0])
            .expect("shape is valid");

        let grad_w = state
            .weight_gradient(&activations, &out_grads)
            .expect("weight gradient computed");
        assert_eq!(grad_w.dim(), (3, 2));

        // grad_W = (1/2) * G^T A
        let expected = [[3.5, 5.0], [1.5, 2.0], [-0.5, -1.0]];
        for i in 0..3 {
            for j in 0..2 {
                assert!((grad_w[[i, j]] - expected[i][j]).abs() < 1e-12);
            }
        }

        // Mismatched batch dimension is an error, not a panic.
        let bad = Array2::<f64>::zeros((3, 3));
        assert!(state.weight_gradient(&activations, &bad).is_err());
    }

    #[test]
    fn test_condition_number_estimation() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 3,
            output_dim: 3,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        let state = KFACLayerState::<f32>::new(layer_info, 0.001);
        let (a_cond, g_cond) = state.condition_number_estimate();

        // Identity matrix should have condition number 1
        assert!((a_cond - 1.0).abs() < 1e-6);
        assert!((g_cond - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_inverses_are_real_not_identity() {
        // End-to-end KFAC path: drive non-trivial covariances into the layer state
        // and confirm the computed inverses are genuine (A · A_inv ≈ I) and are NOT
        // a silent identity (the previous bug).
        let layer_info = LayerInfo {
            name: "dense".to_string(),
            input_dim: 4,
            output_dim: 4,
            layer_type: LayerType::Dense,
            has_bias: false,
        };
        let mut state = KFACLayerState::<f64>::new(layer_info, 0.0);

        // Build a non-identity SPD input covariance: A = B^T B + I.
        let b = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.5, -0.3, 0.2, 0.0, 1.2, 0.7, -0.4, 0.3, -0.1, 0.9, 0.6, -0.2, 0.4, 0.1, 1.1,
            ],
        )
        .expect("shape");
        let mut a_cov = b.t().dot(&b);
        for i in 0..4 {
            a_cov[[i, i]] += 1.0;
        }
        state.a_cov = a_cov.clone();
        // A different non-identity SPD output covariance.
        let mut g_cov = Array2::<f64>::eye(4) * 3.0;
        g_cov[[0, 1]] = 0.5;
        g_cov[[1, 0]] = 0.5;
        g_cov[[2, 3]] = -0.7;
        g_cov[[3, 2]] = -0.7;
        state.g_cov = g_cov.clone();

        // Use zero damping so we can verify the inverse of the raw covariance.
        state.compute_inverses(0.0, 0.0).expect("inverses computed");
        assert!(state.is_ready());

        let a_inv = state.a_cov_inv.as_ref().expect("a_inv present");
        let g_inv = state.g_cov_inv.as_ref().expect("g_inv present");

        // Real inverse: A · A_inv ≈ I and G · G_inv ≈ I.
        let a_prod = a_cov.dot(a_inv);
        let g_prod = g_cov.dot(g_inv);
        let identity: Array2<f64> = Array2::eye(4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((a_prod[[i, j]] - identity[[i, j]]).abs() < 1e-6);
                assert!((g_prod[[i, j]] - identity[[i, j]]).abs() < 1e-6);
            }
        }

        // Regression: the inverse must NOT be the identity for a non-identity input.
        let mut a_inv_is_identity = true;
        for i in 0..4 {
            for j in 0..4 {
                if (a_inv[[i, j]] - identity[[i, j]]).abs() > 1e-9 {
                    a_inv_is_identity = false;
                }
            }
        }
        assert!(
            !a_inv_is_identity,
            "Kronecker-factor inverse collapsed to identity (the old bug)"
        );
    }

    #[test]
    fn test_memory_usage() {
        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 100,
            output_dim: 50,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        let state = KFACLayerState::<f64>::new(layer_info, 0.001);
        let memory_usage = state.memory_usage();

        assert!(memory_usage > 0);
        // Should at least include the covariance matrices
        let expected_minimum = (101 * 101 + 50 * 50) * std::mem::size_of::<f64>();
        assert!(memory_usage >= expected_minimum);
    }
}
