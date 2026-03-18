// Group Lasso regularization with structured sparsity
//
// This module provides Group Lasso regularization which encourages entire groups
// of parameters to be zeroed out, enabling structured sparsity. It also provides
// structured sparsity patterns (column, row, block) for matrix parameters.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Group Lasso regularizer for structured sparsity
///
/// The Group Lasso penalty encourages entire groups of parameters to be zero,
/// rather than individual parameters (as in standard L1/Lasso). This is useful
/// when parameters have a natural grouping structure (e.g., features belonging
/// to the same category, filters in a convolutional layer).
///
/// Penalty: `lambda * sum_g(w_g * ||params[group_g]||_2)`
///
/// where `w_g` is the weight for group `g` and `||.||_2` is the L2 norm.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::regularizers::{GroupLasso, Regularizer};
///
/// // Create a Group Lasso regularizer with two groups
/// let regularizer = GroupLasso::new(0.1_f64)
///     .with_groups(vec![vec![0, 1, 2], vec![3, 4, 5]]);
///
/// let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
/// let penalty = regularizer.penalty(&params).expect("penalty computation failed");
/// ```
#[derive(Debug, Clone)]
pub struct GroupLasso<A: Float + ScalarOperand + Debug> {
    /// Regularization strength
    lambda: A,
    /// Index sets per group - each inner Vec contains parameter indices for that group
    groups: Vec<Vec<usize>>,
    /// Per-group weights (default: all 1.0)
    group_weights: Option<Vec<A>>,
}

impl<A: Float + ScalarOperand + Debug> GroupLasso<A> {
    /// Create a new Group Lasso regularizer
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength (must be non-negative)
    pub fn new(lambda: A) -> Self {
        Self {
            lambda,
            groups: Vec::new(),
            group_weights: None,
        }
    }

    /// Set the groups for the regularizer (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `groups` - A vector of index sets, where each inner vector contains the
    ///   parameter indices belonging to that group
    pub fn with_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.groups = groups;
        self
    }

    /// Set per-group weights (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights, one per group. Must have the same
    ///   length as the number of groups.
    pub fn with_group_weights(mut self, weights: Vec<A>) -> Self {
        self.group_weights = Some(weights);
        self
    }

    /// Automatically create equal-sized groups (builder pattern)
    ///
    /// Partitions parameter indices `[0, param_size)` into groups of `group_size`.
    /// The last group may be smaller if `param_size` is not evenly divisible.
    ///
    /// # Arguments
    ///
    /// * `param_size` - Total number of parameters
    /// * `group_size` - Number of parameters per group
    pub fn auto_groups(mut self, param_size: usize, group_size: usize) -> Self {
        let mut groups = Vec::new();
        let mut start = 0;
        while start < param_size {
            let end = (start + group_size).min(param_size);
            groups.push((start..end).collect());
            start = end;
        }
        self.groups = groups;
        self
    }

    /// Get the regularization strength
    pub fn lambda(&self) -> A {
        self.lambda
    }

    /// Get the groups
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// Get the number of groups
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the weight for a specific group
    fn group_weight(&self, group_idx: usize) -> A {
        self.group_weights
            .as_ref()
            .and_then(|w| w.get(group_idx).copied())
            .unwrap_or_else(A::one)
    }

    /// Compute the L2 norm of parameters at the given indices
    ///
    /// Flattens the array and accesses elements by their linear index.
    fn group_l2_norm(&self, params: &Array<A, impl Dimension>, indices: &[usize]) -> A {
        let flat = params.as_slice_memory_order();
        let sum_sq = indices.iter().fold(A::zero(), |acc, &idx| {
            if let Some(slice) = flat {
                if idx < slice.len() {
                    acc + slice[idx] * slice[idx]
                } else {
                    acc
                }
            } else {
                // Fallback for non-contiguous arrays
                let mut iter = params.iter();
                if let Some(&val) = iter.nth(idx) {
                    acc + val * val
                } else {
                    acc
                }
            }
        });
        sum_sq.sqrt()
    }

    /// Validate that group indices are within bounds for the given parameter array
    fn validate_groups(&self, param_len: usize) -> Result<()> {
        for (g_idx, group) in self.groups.iter().enumerate() {
            for &idx in group {
                if idx >= param_len {
                    return Err(OptimError::InvalidParameter(format!(
                        "Group {} contains index {} which exceeds parameter size {}",
                        g_idx, idx, param_len
                    )));
                }
            }
        }
        if let Some(ref weights) = self.group_weights {
            if weights.len() != self.groups.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "Number of group weights ({}) does not match number of groups ({})",
                    weights.len(),
                    self.groups.len()
                )));
            }
        }
        Ok(())
    }
}

impl<A, D> Regularizer<A, D> for GroupLasso<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        let param_len = params.len();
        self.validate_groups(param_len)?;

        let epsilon = A::from(1e-8).unwrap_or_else(|| A::epsilon());

        // Get mutable slice for gradients
        let grad_slice = gradients.as_slice_memory_order_mut().ok_or_else(|| {
            OptimError::InvalidParameter("Gradients array is not contiguous in memory".to_string())
        })?;

        let param_slice = params.as_slice_memory_order().ok_or_else(|| {
            OptimError::InvalidParameter("Parameters array is not contiguous in memory".to_string())
        })?;

        for (g_idx, group) in self.groups.iter().enumerate() {
            let w_g = self.group_weight(g_idx);

            // Compute L2 norm for this group
            let sum_sq = group.iter().fold(A::zero(), |acc, &idx| {
                if idx < param_len {
                    acc + param_slice[idx] * param_slice[idx]
                } else {
                    acc
                }
            });
            let norm = sum_sq.sqrt();

            // Gradient: lambda * w_g * param[i] / (||group||_2 + epsilon)
            let scale = self.lambda * w_g / (norm + epsilon);

            for &idx in group {
                if idx < param_len {
                    grad_slice[idx] = grad_slice[idx] + scale * param_slice[idx];
                }
            }
        }

        self.penalty(params)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        let param_len = params.len();
        self.validate_groups(param_len)?;

        let mut total = A::zero();

        for (g_idx, group) in self.groups.iter().enumerate() {
            let w_g = self.group_weight(g_idx);
            let norm = self.group_l2_norm(params, group);
            total = total + w_g * norm;
        }

        Ok(self.lambda * total)
    }
}

/// Sparsity pattern for structured sparsity regularization
///
/// Defines how parameters are grouped for structured sparsity.
/// Different patterns encourage different structural zeros in
/// the parameter matrix.
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Column-wise sparsity: entire columns of the parameter matrix are zeroed
    ///
    /// Groups parameters by column index, encouraging entire columns to be sparse.
    Column {
        /// Number of columns in the parameter matrix
        num_columns: usize,
    },
    /// Row-wise sparsity: entire rows of the parameter matrix are zeroed
    ///
    /// Groups parameters by row index, encouraging entire rows to be sparse.
    Row {
        /// Number of rows in the parameter matrix
        num_rows: usize,
    },
    /// Block-wise sparsity: rectangular blocks of parameters are zeroed
    ///
    /// Groups parameters into rectangular blocks, encouraging entire blocks to be sparse.
    Block {
        /// Height of each block (number of rows)
        block_height: usize,
        /// Width of each block (number of columns)
        block_width: usize,
    },
}

/// Structured sparsity regularizer
///
/// Applies Group Lasso regularization with groups defined by a structural
/// pattern (columns, rows, or blocks). This is useful for encouraging
/// structured sparsity in weight matrices, e.g., pruning entire neurons
/// (row sparsity) or features (column sparsity).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::regularizers::{StructuredSparsity, SparsityPattern, Regularizer};
///
/// // Create column-wise structured sparsity for a 3x4 matrix (stored as 12-element vector)
/// let regularizer = StructuredSparsity::new(0.1_f64, SparsityPattern::Column { num_columns: 4 });
///
/// let params = Array1::from_vec(vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0]);
/// let penalty = regularizer.penalty(&params).expect("penalty computation failed");
/// ```
#[derive(Debug, Clone)]
pub struct StructuredSparsity<A: Float + ScalarOperand + Debug> {
    /// Regularization strength
    lambda: A,
    /// The structural sparsity pattern
    pattern: SparsityPattern,
}

impl<A: Float + ScalarOperand + Debug> StructuredSparsity<A> {
    /// Create a new structured sparsity regularizer
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength
    /// * `pattern` - The structural pattern defining how parameters are grouped
    pub fn new(lambda: A, pattern: SparsityPattern) -> Self {
        Self { lambda, pattern }
    }

    /// Get the regularization strength
    pub fn lambda(&self) -> A {
        self.lambda
    }

    /// Get the sparsity pattern
    pub fn pattern(&self) -> &SparsityPattern {
        &self.pattern
    }

    /// Build groups from the sparsity pattern for a given parameter count
    ///
    /// For a matrix with `total_params` elements stored in row-major order:
    /// - Column pattern: groups parameters sharing the same column index
    /// - Row pattern: groups parameters sharing the same row index
    /// - Block pattern: groups parameters into rectangular blocks
    fn build_groups(&self, total_params: usize) -> Result<Vec<Vec<usize>>> {
        match &self.pattern {
            SparsityPattern::Column { num_columns } => {
                if *num_columns == 0 {
                    return Err(OptimError::InvalidConfig(
                        "Number of columns must be greater than 0".to_string(),
                    ));
                }
                let num_rows = total_params / num_columns;
                if num_rows * num_columns != total_params {
                    return Err(OptimError::InvalidConfig(format!(
                        "Total parameters ({}) is not evenly divisible by num_columns ({})",
                        total_params, num_columns
                    )));
                }

                let mut groups = Vec::with_capacity(*num_columns);
                for col in 0..*num_columns {
                    let group: Vec<usize> =
                        (0..num_rows).map(|row| row * num_columns + col).collect();
                    groups.push(group);
                }
                Ok(groups)
            }
            SparsityPattern::Row { num_rows } => {
                if *num_rows == 0 {
                    return Err(OptimError::InvalidConfig(
                        "Number of rows must be greater than 0".to_string(),
                    ));
                }
                let num_columns = total_params / num_rows;
                if num_rows * num_columns != total_params {
                    return Err(OptimError::InvalidConfig(format!(
                        "Total parameters ({}) is not evenly divisible by num_rows ({})",
                        total_params, num_rows
                    )));
                }

                let mut groups = Vec::with_capacity(*num_rows);
                for row in 0..*num_rows {
                    let start = row * num_columns;
                    let group: Vec<usize> = (start..start + num_columns).collect();
                    groups.push(group);
                }
                Ok(groups)
            }
            SparsityPattern::Block {
                block_height,
                block_width,
            } => {
                if *block_height == 0 || *block_width == 0 {
                    return Err(OptimError::InvalidConfig(
                        "Block dimensions must be greater than 0".to_string(),
                    ));
                }

                // Infer the total number of columns from block_width
                // We need to figure out the matrix dimensions. We assume the matrix
                // width is a multiple of block_width. We try to find a valid decomposition.
                // For block sparsity, we need the user to provide a compatible total_params.
                // We compute the number of columns as the smallest multiple of block_width
                // such that total_params / num_cols is a multiple of block_height.
                let num_cols =
                    self.infer_matrix_columns(total_params, *block_height, *block_width)?;
                let num_rows = total_params / num_cols;

                let blocks_per_row = num_cols / block_width;
                let blocks_per_col = num_rows / block_height;

                let mut groups = Vec::with_capacity(blocks_per_row * blocks_per_col);
                for block_row in 0..blocks_per_col {
                    for block_col in 0..blocks_per_row {
                        let mut group = Vec::with_capacity(block_height * block_width);
                        for r in 0..*block_height {
                            for c in 0..*block_width {
                                let row = block_row * block_height + r;
                                let col = block_col * block_width + c;
                                group.push(row * num_cols + col);
                            }
                        }
                        groups.push(group);
                    }
                }
                Ok(groups)
            }
        }
    }

    /// Infer the number of matrix columns for block sparsity
    ///
    /// Tries to find the candidate number of columns (a multiple of block_width)
    /// that produces the most square-like matrix decomposition.
    fn infer_matrix_columns(
        &self,
        total_params: usize,
        block_height: usize,
        block_width: usize,
    ) -> Result<usize> {
        let target = (total_params as f64).sqrt();
        let mut best_candidate: Option<usize> = None;
        let mut best_distance = f64::MAX;

        let mut candidate = block_width;
        while candidate <= total_params {
            if total_params.is_multiple_of(candidate) {
                let rows = total_params / candidate;
                if rows.is_multiple_of(block_height) {
                    let distance = (candidate as f64 - target).abs();
                    if distance < best_distance {
                        best_distance = distance;
                        best_candidate = Some(candidate);
                    }
                }
            }
            candidate += block_width;
        }

        best_candidate.ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "Cannot decompose {} parameters into blocks of {}x{}",
                total_params, block_height, block_width
            ))
        })
    }
}

impl<A, D> Regularizer<A, D> for StructuredSparsity<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        let total_params = params.len();
        let groups = self.build_groups(total_params)?;

        let epsilon = A::from(1e-8).unwrap_or_else(|| A::epsilon());

        let grad_slice = gradients.as_slice_memory_order_mut().ok_or_else(|| {
            OptimError::InvalidParameter("Gradients array is not contiguous in memory".to_string())
        })?;

        let param_slice = params.as_slice_memory_order().ok_or_else(|| {
            OptimError::InvalidParameter("Parameters array is not contiguous in memory".to_string())
        })?;

        for group in &groups {
            // Compute L2 norm for this group
            let sum_sq = group.iter().fold(A::zero(), |acc, &idx| {
                if idx < total_params {
                    acc + param_slice[idx] * param_slice[idx]
                } else {
                    acc
                }
            });
            let norm = sum_sq.sqrt();

            let scale = self.lambda / (norm + epsilon);

            for &idx in group {
                if idx < total_params {
                    grad_slice[idx] = grad_slice[idx] + scale * param_slice[idx];
                }
            }
        }

        self.penalty(params)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        let total_params = params.len();
        let groups = self.build_groups(total_params)?;

        let param_slice = params.as_slice_memory_order().ok_or_else(|| {
            OptimError::InvalidParameter("Parameters array is not contiguous in memory".to_string())
        })?;

        let mut total = A::zero();

        for group in &groups {
            let sum_sq = group.iter().fold(A::zero(), |acc, &idx| {
                if idx < total_params {
                    acc + param_slice[idx] * param_slice[idx]
                } else {
                    acc
                }
            });
            total = total + sum_sq.sqrt();
        }

        Ok(self.lambda * total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_group_lasso_basic_penalty() {
        // Two groups: [0,1,2] and [3,4,5]
        // Group 0: params = [1, 2, 3], ||group||_2 = sqrt(1+4+9) = sqrt(14)
        // Group 1: params = [0, 0, 0], ||group||_2 = 0
        // Penalty = 0.1 * (sqrt(14) + 0) = 0.1 * sqrt(14)
        let regularizer = GroupLasso::new(0.1_f64).with_groups(vec![vec![0, 1, 2], vec![3, 4, 5]]);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        let expected = 0.1 * (14.0_f64).sqrt();
        assert_abs_diff_eq!(penalty, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_group_lasso_with_weights() {
        // Two groups with different weights
        let regularizer = GroupLasso::new(0.5_f64)
            .with_groups(vec![vec![0, 1], vec![2, 3]])
            .with_group_weights(vec![2.0, 0.5]);

        let params = Array1::from_vec(vec![3.0, 4.0, 1.0, 0.0]);
        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        // Group 0: w=2.0, norm=sqrt(9+16)=5.0 => 2.0*5.0=10.0
        // Group 1: w=0.5, norm=sqrt(1+0)=1.0 => 0.5*1.0=0.5
        // Total: 0.5 * (10.0 + 0.5) = 5.25
        let expected = 0.5 * (2.0 * 5.0 + 0.5 * 1.0);
        assert_abs_diff_eq!(penalty, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_group_lasso_auto_groups() {
        let regularizer = GroupLasso::new(0.1_f64).auto_groups(9, 3);

        // Should create 3 groups: [0,1,2], [3,4,5], [6,7,8]
        assert_eq!(regularizer.num_groups(), 3);
        assert_eq!(regularizer.groups()[0], vec![0, 1, 2]);
        assert_eq!(regularizer.groups()[1], vec![3, 4, 5]);
        assert_eq!(regularizer.groups()[2], vec![6, 7, 8]);

        // Test with non-evenly-divisible size
        let regularizer2 = GroupLasso::new(0.1_f64).auto_groups(7, 3);
        assert_eq!(regularizer2.num_groups(), 3);
        assert_eq!(regularizer2.groups()[0], vec![0, 1, 2]);
        assert_eq!(regularizer2.groups()[1], vec![3, 4, 5]);
        assert_eq!(regularizer2.groups()[2], vec![6]); // Remainder group
    }

    #[test]
    fn test_group_lasso_gradient_application() {
        let regularizer = GroupLasso::new(1.0_f64).with_groups(vec![vec![0, 1], vec![2, 3]]);

        let params = Array1::from_vec(vec![3.0, 4.0, 0.0, 0.0]);
        let mut gradients = Array1::zeros(4);

        let penalty = regularizer
            .apply(&params, &mut gradients)
            .expect("apply failed");

        // Group 0: norm = sqrt(9+16) = 5.0
        // Gradient for idx 0: 1.0 * 1.0 * 3.0 / (5.0 + 1e-8) ~ 0.6
        // Gradient for idx 1: 1.0 * 1.0 * 4.0 / (5.0 + 1e-8) ~ 0.8
        let epsilon = 1e-8_f64;
        let norm0 = 5.0_f64;
        assert_abs_diff_eq!(gradients[0], 3.0 / (norm0 + epsilon), epsilon = 1e-6);
        assert_abs_diff_eq!(gradients[1], 4.0 / (norm0 + epsilon), epsilon = 1e-6);

        // Group 1: norm = 0.0, so gradient ~ lambda * 0 / epsilon ~ 0
        assert_abs_diff_eq!(gradients[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(gradients[3], 0.0, epsilon = 1e-6);

        // Penalty: 1.0 * (5.0 + 0.0) = 5.0
        assert_abs_diff_eq!(penalty, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_structured_sparsity_column() {
        // 3x4 matrix stored as 12 elements (row-major)
        // Columns: col0=[0,4,8], col1=[1,5,9], col2=[2,6,10], col3=[3,7,11]
        let regularizer =
            StructuredSparsity::new(0.1_f64, SparsityPattern::Column { num_columns: 4 });

        // Matrix (row-major):
        // [1, 0, 3, 0]
        // [5, 0, 7, 0]
        // [9, 0, 11, 0]
        let params = Array1::from_vec(vec![
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0,
        ]);

        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        // Col 0: [1,5,9] => norm = sqrt(1+25+81) = sqrt(107)
        // Col 1: [0,0,0] => norm = 0
        // Col 2: [3,7,11] => norm = sqrt(9+49+121) = sqrt(179)
        // Col 3: [0,0,0] => norm = 0
        let expected = 0.1 * (107.0_f64.sqrt() + 179.0_f64.sqrt());
        assert_abs_diff_eq!(penalty, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_structured_sparsity_row() {
        // 3x2 matrix stored as 6 elements
        let regularizer = StructuredSparsity::new(0.5_f64, SparsityPattern::Row { num_rows: 3 });

        // [1, 2]
        // [0, 0]
        // [3, 4]
        let params = Array1::from_vec(vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);

        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        // Row 0: [1,2] => norm = sqrt(5)
        // Row 1: [0,0] => norm = 0
        // Row 2: [3,4] => norm = sqrt(25) = 5
        let expected = 0.5 * (5.0_f64.sqrt() + 5.0);
        assert_abs_diff_eq!(penalty, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_structured_sparsity_block() {
        // 4x4 matrix with 2x2 blocks => 4 blocks
        let regularizer = StructuredSparsity::new(
            0.2_f64,
            SparsityPattern::Block {
                block_height: 2,
                block_width: 2,
            },
        );

        // [1, 1, 0, 0]
        // [1, 1, 0, 0]
        // [0, 0, 2, 2]
        // [0, 0, 2, 2]
        let params = Array1::from_vec(vec![
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        ]);

        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        // Block (0,0): [1,1,1,1] => norm = 2.0
        // Block (0,1): [0,0,0,0] => norm = 0.0
        // Block (1,0): [0,0,0,0] => norm = 0.0
        // Block (1,1): [2,2,2,2] => norm = 4.0
        let expected = 0.2 * (2.0 + 4.0);
        assert_abs_diff_eq!(penalty, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_structured_sparsity_gradient_application() {
        let regularizer = StructuredSparsity::new(1.0_f64, SparsityPattern::Row { num_rows: 2 });

        // [3, 4]
        // [0, 0]
        let params = Array1::from_vec(vec![3.0, 4.0, 0.0, 0.0]);
        let mut gradients = Array1::zeros(4);

        let _penalty = regularizer
            .apply(&params, &mut gradients)
            .expect("apply failed");

        // Row 0: norm = 5.0
        // grad[0] = 1.0 * 3.0 / (5.0 + eps) ~ 0.6
        // grad[1] = 1.0 * 4.0 / (5.0 + eps) ~ 0.8
        let epsilon = 1e-8_f64;
        assert_abs_diff_eq!(gradients[0], 3.0 / (5.0 + epsilon), epsilon = 1e-6);
        assert_abs_diff_eq!(gradients[1], 4.0 / (5.0 + epsilon), epsilon = 1e-6);

        // Row 1: norm ~ 0, so gradient ~ 0 / eps ~ 0
        assert_abs_diff_eq!(gradients[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(gradients[3], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_group_lasso_empty_groups() {
        // No groups => penalty should be zero
        let regularizer = GroupLasso::<f64>::new(0.1);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let penalty = regularizer
            .penalty(&params)
            .expect("penalty computation failed");

        assert_abs_diff_eq!(penalty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_group_lasso_out_of_bounds_index() {
        let regularizer = GroupLasso::new(0.1_f64).with_groups(vec![vec![0, 1, 100]]); // index 100 is out of bounds

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = regularizer.penalty(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_group_lasso_weight_mismatch() {
        let regularizer = GroupLasso::new(0.1_f64)
            .with_groups(vec![vec![0, 1], vec![2, 3]])
            .with_group_weights(vec![1.0]); // Only 1 weight for 2 groups

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = regularizer.penalty(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_structured_sparsity_invalid_dimensions() {
        // 7 params cannot be divided into columns of 3
        let regularizer =
            StructuredSparsity::new(0.1_f64, SparsityPattern::Column { num_columns: 3 });

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let result = regularizer.penalty(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_structured_sparsity_zero_columns() {
        let regularizer =
            StructuredSparsity::new(0.1_f64, SparsityPattern::Column { num_columns: 0 });

        let params = Array1::from_vec(vec![1.0, 2.0]);
        let result = regularizer.penalty(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_group_lasso_builder_pattern() {
        let regularizer = GroupLasso::new(0.5_f64)
            .with_groups(vec![vec![0, 1], vec![2, 3]])
            .with_group_weights(vec![1.0, 2.0]);

        assert_eq!(regularizer.lambda(), 0.5);
        assert_eq!(regularizer.num_groups(), 2);
        assert_eq!(regularizer.groups()[0], vec![0, 1]);
        assert_eq!(regularizer.groups()[1], vec![2, 3]);
    }
}
