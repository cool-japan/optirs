// Parameter groups for different learning rates and configurations
//
// This module provides support for parameter groups, allowing different
// sets of parameters to have different hyperparameters (learning rate,
// weight decay, etc.) within the same optimizer.

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, Ix2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

/// Parameter constraints that can be applied to parameter groups
#[derive(Debug, Clone)]
pub enum ParameterConstraint<A: Float> {
    /// Clip values to a range [min, max]
    ValueClip {
        /// Minimum allowed value
        min: A,
        /// Maximum allowed value
        max: A,
    },
    /// Constrain L2 norm to a maximum value
    L2NormConstraint {
        /// Maximum allowed L2 norm
        maxnorm: A,
    },
    /// Constrain L1 norm to a maximum value
    L1NormConstraint {
        /// Maximum allowed L1 norm
        maxnorm: A,
    },
    /// Ensure all values are non-negative
    NonNegative,
    /// Constrain to unit sphere (normalize to unit L2 norm)
    UnitSphere,
    /// Constrain parameters to be within a probability simplex (sum to 1, all non-negative)
    Simplex,
    /// Constrain matrix parameters to be orthogonal
    Orthogonal {
        /// Tolerance for orthogonality check
        tolerance: A,
    },
    /// Constrain symmetric matrices to be positive definite
    PositiveDefinite {
        /// Minimum eigenvalue to ensure positive definiteness
        mineigenvalue: A,
    },
    /// Spectral norm constraint (maximum singular value)
    SpectralNorm {
        /// Maximum allowed spectral norm
        maxnorm: A,
    },
    /// Nuclear norm constraint (sum of singular values)
    NuclearNorm {
        /// Maximum allowed nuclear norm
        maxnorm: A,
    },
    /// Custom constraint function
    Custom {
        /// Name of the custom constraint
        name: String,
    },
}

impl<A: Float + Send + Sync> ParameterConstraint<A> {
    /// Apply the constraint to a parameter array
    pub fn apply<D: Dimension>(&self, params: &mut Array<A, D>) -> Result<()>
    where
        A: ScalarOperand,
    {
        match self {
            ParameterConstraint::ValueClip { min, max } => {
                params.mapv_inplace(|x| {
                    if x < *min {
                        *min
                    } else if x > *max {
                        *max
                    } else {
                        x
                    }
                });
            }
            ParameterConstraint::L2NormConstraint { maxnorm } => {
                let norm = params.mapv(|x| x * x).sum().sqrt();
                if norm > *maxnorm {
                    let scale = *maxnorm / norm;
                    params.mapv_inplace(|x| x * scale);
                }
            }
            ParameterConstraint::L1NormConstraint { maxnorm } => {
                let norm = params.mapv(|x| x.abs()).sum();
                if norm > *maxnorm {
                    let scale = *maxnorm / norm;
                    params.mapv_inplace(|x| x * scale);
                }
            }
            ParameterConstraint::NonNegative => {
                params.mapv_inplace(|x| if x < A::zero() { A::zero() } else { x });
            }
            ParameterConstraint::UnitSphere => {
                let norm = params.mapv(|x| x * x).sum().sqrt();
                if norm > A::zero() {
                    let scale = A::one() / norm;
                    params.mapv_inplace(|x| x * scale);
                }
            }
            ParameterConstraint::Simplex => {
                // First make all values non-negative
                params.mapv_inplace(|x| if x < A::zero() { A::zero() } else { x });

                // Then normalize to sum to 1
                let sum = params.sum();
                if sum > A::zero() {
                    let scale = A::one() / sum;
                    params.mapv_inplace(|x| x * scale);
                } else {
                    // If all values are zero, set to uniform distribution
                    let uniform_val = A::one() / A::from(params.len()).unwrap_or(A::one());
                    params.fill(uniform_val);
                }
            }
            ParameterConstraint::Orthogonal { tolerance } => {
                // Orthonormalize the columns of a 2D matrix via modified Gram-Schmidt.
                if params.ndim() == 2 {
                    let matrix = to_matrix_2d(params)?;
                    let (rows, cols) = matrix.dim();

                    // Skip work if the columns are already orthonormal within tolerance.
                    if rows > 0 && cols > 0 && is_orthonormal(&matrix, *tolerance) {
                        return Ok(());
                    }

                    let orthonormal = modified_gram_schmidt(&matrix);
                    write_matrix_2d(params, &orthonormal)?;
                } else {
                    return Err(OptimError::InvalidConfig(
                        "Orthogonal constraint only applies to 2D arrays (matrices)".to_string(),
                    ));
                }
            }
            ParameterConstraint::PositiveDefinite { mineigenvalue } => {
                // Symmetrize, eigendecompose (cyclic Jacobi), clamp eigenvalues, reconstruct.
                if params.ndim() != 2 {
                    return Err(OptimError::InvalidConfig(
                        "Positive definite constraint only applies to 2D arrays (matrices)"
                            .to_string(),
                    ));
                }
                let matrix = to_matrix_2d(params)?;
                let (rows, cols) = matrix.dim();
                if rows != cols {
                    return Err(OptimError::InvalidConfig(
                        "Positive definite constraint requires a square matrix".to_string(),
                    ));
                }

                let projected = project_positive_definite(&matrix, *mineigenvalue);
                write_matrix_2d(params, &projected)?;
            }
            ParameterConstraint::SpectralNorm { maxnorm } => {
                // Bound the largest singular value via power iteration on MᵀM.
                if params.ndim() != 2 {
                    return Err(OptimError::InvalidConfig(
                        "Spectral norm constraint only applies to 2D arrays (matrices)".to_string(),
                    ));
                }
                let matrix = to_matrix_2d(params)?;
                let sigma_max = power_iteration_spectral_norm(&matrix);
                if sigma_max > *maxnorm && sigma_max > A::zero() {
                    let scale = *maxnorm / sigma_max;
                    params.mapv_inplace(|x| x * scale);
                }
            }
            ParameterConstraint::NuclearNorm { maxnorm } => {
                // Nuclear norm constraint requires SVD computation
                // For now, approximate with L1 norm
                let l1_norm = params.mapv(|x| x.abs()).sum();
                if l1_norm > *maxnorm {
                    let scale = *maxnorm / l1_norm;
                    params.mapv_inplace(|x| x * scale);
                }
            }
            ParameterConstraint::Custom { name } => {
                return Err(OptimError::InvalidConfig(format!(
                    "Custom constraint '{name}' not implemented"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Self-contained linear-algebra helpers for the matrix constraints.
//
// These avoid any external linear-algebra dependency (scirs2-linalg is not
// available here). They operate on small dense 2D matrices using only
// `scirs2_core::ndarray`, and keep the generic `A: Float` bound. `Float` is not
// `Ord`, so all comparisons go through `partial_cmp` / explicit `<`/`>`.
// ---------------------------------------------------------------------------

/// Obtain an owned 2D matrix from a generic n-dimensional array known to be 2D.
///
/// The caller must have already checked `params.ndim() == 2`; the conversion is
/// done on a clone so the original array is untouched until we write back.
fn to_matrix_2d<A, D>(params: &Array<A, D>) -> Result<Array2<A>>
where
    A: Float,
    D: Dimension,
{
    params
        .to_owned()
        .into_dimensionality::<Ix2>()
        .map_err(|e| OptimError::InvalidConfig(format!("Failed to view array as 2D matrix: {e}")))
}

/// Write a 2D matrix back into the generic n-dimensional array in place.
///
/// The matrix must have exactly the same shape as `params`; the values are
/// copied element-by-element in logical (row-major) order so the result is
/// independent of the concrete dimension type `D`.
fn write_matrix_2d<A, D>(params: &mut Array<A, D>, matrix: &Array2<A>) -> Result<()>
where
    A: Float,
    D: Dimension,
{
    if params.len() != matrix.len() {
        return Err(OptimError::InvalidConfig(
            "Internal error: matrix/parameter element count mismatch".to_string(),
        ));
    }
    for (dst, &src) in params.iter_mut().zip(matrix.iter()) {
        *dst = src;
    }
    Ok(())
}

/// Check whether the columns of `matrix` are already orthonormal, i.e.
/// `‖MᵀM − I‖_F ≤ tolerance` (Frobenius norm of the residual).
fn is_orthonormal<A>(matrix: &Array2<A>, tolerance: A) -> bool
where
    A: Float,
{
    let (rows, cols) = matrix.dim();
    let mut residual_sq = A::zero();
    for i in 0..cols {
        for j in 0..cols {
            // (MᵀM)_{ij} = Σ_k M_{ki} M_{kj}
            let mut dot = A::zero();
            for k in 0..rows {
                dot = dot + matrix[[k, i]] * matrix[[k, j]];
            }
            let target = if i == j { A::one() } else { A::zero() };
            let diff = dot - target;
            residual_sq = residual_sq + diff * diff;
        }
    }
    residual_sq.sqrt() <= tolerance
}

/// Orthonormalize the columns of `matrix` using **modified Gram-Schmidt**.
///
/// Numerically stabler than classical Gram-Schmidt because each new column is
/// orthogonalized against the already-finalized basis vectors as it is built.
/// Handles non-square matrices: for an `r × c` matrix only the first
/// `min(r, c)` columns can be linearly independent, so any remaining columns
/// (or columns whose norm underflows) are replaced deterministically by a unit
/// basis vector that is orthogonal to the accumulated basis (falling back to a
/// canonical axis when none is available).
fn modified_gram_schmidt<A>(matrix: &Array2<A>) -> Array2<A>
where
    A: Float,
{
    let (rows, cols) = matrix.dim();
    let mut q: Array2<A> = Array2::zeros((rows, cols));
    if rows == 0 || cols == 0 {
        return q;
    }

    // Threshold below which a column norm is treated as numerical zero.
    let eps = A::epsilon();
    let norm_floor = eps.sqrt();

    // Working copy of the columns; modified in place as we project out
    // previously finalized directions (the "modified" part of MGS).
    let mut work = matrix.clone();

    for j in 0..cols {
        // Re-orthogonalize column j against all finalized columns 0..j.
        for i in 0..j {
            // r_ij = q_i · work_j
            let mut dot = A::zero();
            for k in 0..rows {
                dot = dot + q[[k, i]] * work[[k, j]];
            }
            for k in 0..rows {
                work[[k, j]] = work[[k, j]] - dot * q[[k, i]];
            }
        }

        // Norm of the residual column.
        let mut norm_sq = A::zero();
        for k in 0..rows {
            norm_sq = norm_sq + work[[k, j]] * work[[k, j]];
        }
        let norm = norm_sq.sqrt();

        if norm > norm_floor {
            let inv = A::one() / norm;
            for k in 0..rows {
                q[[k, j]] = work[[k, j]] * inv;
            }
        } else {
            // Degenerate/underflowing column: deterministically substitute a
            // unit vector orthogonal to the existing basis. Try each canonical
            // axis e_a in order, project out the finalized basis, and accept the
            // first with sufficient norm.
            let mut filled = false;
            for axis in 0..rows {
                let mut candidate: Array1<A> = Array1::zeros(rows);
                candidate[axis] = A::one();
                for i in 0..j {
                    let mut dot = A::zero();
                    for k in 0..rows {
                        dot = dot + q[[k, i]] * candidate[k];
                    }
                    for k in 0..rows {
                        candidate[k] = candidate[k] - dot * q[[k, i]];
                    }
                }
                let mut cand_norm_sq = A::zero();
                for k in 0..rows {
                    cand_norm_sq = cand_norm_sq + candidate[k] * candidate[k];
                }
                let cand_norm = cand_norm_sq.sqrt();
                if cand_norm > norm_floor {
                    let inv = A::one() / cand_norm;
                    for k in 0..rows {
                        q[[k, j]] = candidate[k] * inv;
                    }
                    filled = true;
                    break;
                }
            }
            if !filled {
                // No orthogonal axis available (more columns than rows): leave
                // this column as a zero vector, which is the deterministic
                // result of orthonormalizing a rank-deficient set.
                for k in 0..rows {
                    q[[k, j]] = A::zero();
                }
            }
        }
    }

    q
}

/// Estimate σ_max(M) = sqrt(λ_max(MᵀM)) via **power iteration**.
///
/// Uses a fixed, deterministic all-ones start vector (normalized) — no RNG — so
/// results are reproducible. Each iteration applies the symmetric PSD operator
/// `A = MᵀM` to the current vector, then renormalizes. The Rayleigh quotient
/// `vᵀ A v` converges to the dominant eigenvalue λ_max; σ_max is its square root.
fn power_iteration_spectral_norm<A>(matrix: &Array2<A>) -> A
where
    A: Float,
{
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols == 0 {
        return A::zero();
    }

    let eps = A::epsilon();
    let norm_floor = eps.sqrt();

    // Deterministic start: all ones, normalized.
    let mut v: Array1<A> = Array1::from_elem(cols, A::one());
    let start_norm = (A::from(cols).unwrap_or_else(A::one)).sqrt();
    if start_norm > norm_floor {
        let inv = A::one() / start_norm;
        v.mapv_inplace(|x| x * inv);
    }

    let max_iters = 64usize;
    let mut lambda = A::zero();

    for _ in 0..max_iters {
        // w = M v   (length rows)
        let mut w: Array1<A> = Array1::zeros(rows);
        for r in 0..rows {
            let mut acc = A::zero();
            for c in 0..cols {
                acc = acc + matrix[[r, c]] * v[c];
            }
            w[r] = acc;
        }
        // a = Mᵀ w = (MᵀM) v   (length cols)
        let mut a: Array1<A> = Array1::zeros(cols);
        for c in 0..cols {
            let mut acc = A::zero();
            for r in 0..rows {
                acc = acc + matrix[[r, c]] * w[r];
            }
            a[c] = acc;
        }

        // Rayleigh quotient vᵀ(MᵀM)v with v normalized ⇒ estimate of λ_max.
        let mut rayleigh = A::zero();
        for c in 0..cols {
            rayleigh = rayleigh + v[c] * a[c];
        }
        lambda = rayleigh;

        // Renormalize a → next v.
        let mut norm_sq = A::zero();
        for c in 0..cols {
            norm_sq = norm_sq + a[c] * a[c];
        }
        let norm = norm_sq.sqrt();
        if norm <= norm_floor {
            // MᵀM v ≈ 0 ⇒ matrix is (numerically) zero on this direction.
            break;
        }
        let inv = A::one() / norm;
        for c in 0..cols {
            v[c] = a[c] * inv;
        }
    }

    if lambda < A::zero() {
        // MᵀM is PSD; guard against tiny negative round-off.
        A::zero()
    } else {
        lambda.sqrt()
    }
}

/// Symmetric eigendecomposition via the **cyclic Jacobi** algorithm.
///
/// Returns `(eigenvalues, eigenvectors)` where column `k` of the eigenvector
/// matrix is the eigenvector for `eigenvalues[k]`, so that
/// `A ≈ Q · diag(eigenvalues) · Qᵀ`. The input is assumed symmetric; callers
/// should symmetrize first. Reliable and self-contained for the small dense
/// symmetric matrices that arise from parameter constraints.
fn jacobi_eigen_symmetric<A>(input: &Array2<A>) -> (Array1<A>, Array2<A>)
where
    A: Float,
{
    let n = input.nrows();
    let mut a = input.clone();
    let mut v: Array2<A> = Array2::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = A::one();
    }

    if n == 0 {
        return (Array1::zeros(0), v);
    }
    if n == 1 {
        return (Array1::from_elem(1, a[[0, 0]]), v);
    }

    let eps = A::epsilon();
    let two = A::one() + A::one();
    let max_sweeps = 100usize;

    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius magnitude; stop once negligible.
        let mut off = A::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                off = off + a[[p, q]] * a[[p, q]];
            }
        }
        if off.sqrt() <= eps {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() <= eps {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];

                // Compute the Jacobi rotation (c, s) zeroing a[p,q].
                let theta = (aqq - app) / (two * apq);
                let sign = if theta < A::zero() {
                    -A::one()
                } else {
                    A::one()
                };
                let denom = theta.abs() + (theta * theta + A::one()).sqrt();
                let t = sign / denom;
                let c = A::one() / (t * t + A::one()).sqrt();
                let s = t * c;

                // Apply rotation to rows/cols p and q of A.
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[[k, p]];
                        let akq = a[[k, q]];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;
                        a[[k, p]] = new_kp;
                        a[[p, k]] = new_kp;
                        a[[k, q]] = new_kq;
                        a[[q, k]] = new_kq;
                    }
                }

                let new_app = c * c * app - two * s * c * apq + s * s * aqq;
                let new_aqq = s * s * app + two * s * c * apq + c * c * aqq;
                a[[p, p]] = new_app;
                a[[q, q]] = new_aqq;
                a[[p, q]] = A::zero();
                a[[q, p]] = A::zero();

                // Accumulate the rotation into the eigenvector matrix.
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut eigenvalues: Array1<A> = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }
    (eigenvalues, v)
}

/// Project a square matrix onto the cone of matrices with eigenvalues
/// `≥ min_eigenvalue`.
///
/// Symmetrizes the input (`(M + Mᵀ)/2`), eigendecomposes it via cyclic Jacobi,
/// clamps each eigenvalue up to `min_eigenvalue`, then reconstructs
/// `Q · diag(λ_clamped) · Qᵀ`.
fn project_positive_definite<A>(matrix: &Array2<A>, min_eigenvalue: A) -> Array2<A>
where
    A: Float,
{
    let n = matrix.nrows();
    let two = A::one() + A::one();

    // Symmetrize: S = (M + Mᵀ) / 2.
    let mut sym: Array2<A> = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) / two;
        }
    }

    let (mut eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&sym);

    // Clamp eigenvalues to the floor.
    for k in 0..n {
        if eigenvalues[k] < min_eigenvalue {
            eigenvalues[k] = min_eigenvalue;
        }
    }

    // Reconstruct Q Λ Qᵀ.
    let mut result: Array2<A> = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut acc = A::zero();
            for k in 0..n {
                acc = acc + eigenvectors[[i, k]] * eigenvalues[k] * eigenvectors[[j, k]];
            }
            result[[i, j]] = acc;
        }
    }
    result
}

/// Configuration for a parameter group
#[derive(Debug, Clone)]
pub struct ParameterGroupConfig<A: Float> {
    /// Learning rate for this group
    pub learning_rate: Option<A>,
    /// Weight decay for this group
    pub weight_decay: Option<A>,
    /// Momentum for this group (if applicable)
    pub momentum: Option<A>,
    /// Parameter constraints for this group
    pub constraints: Vec<ParameterConstraint<A>>,
    /// Custom parameters as key-value pairs
    pub custom_params: HashMap<String, A>,
}

impl<A: Float + Send + Sync> Default for ParameterGroupConfig<A> {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay: None,
            momentum: None,
            constraints: Vec::new(),
            custom_params: HashMap::new(),
        }
    }
}

impl<A: Float + Send + Sync> ParameterGroupConfig<A> {
    /// Create a new parameter group configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: A) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, wd: A) -> Self {
        self.weight_decay = Some(wd);
        self
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, key: String, value: A) -> Self {
        self.custom_params.insert(key, value);
        self
    }

    /// Add a parameter constraint
    pub fn with_constraint(mut self, constraint: ParameterConstraint<A>) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add value clipping constraint
    pub fn with_value_clip(mut self, min: A, max: A) -> Self {
        self.constraints
            .push(ParameterConstraint::ValueClip { min, max });
        self
    }

    /// Add L2 norm constraint
    pub fn with_l2_norm_constraint(mut self, maxnorm: A) -> Self {
        self.constraints
            .push(ParameterConstraint::L2NormConstraint { maxnorm });
        self
    }

    /// Add L1 norm constraint
    pub fn with_l1_norm_constraint(mut self, maxnorm: A) -> Self {
        self.constraints
            .push(ParameterConstraint::L1NormConstraint { maxnorm });
        self
    }

    /// Add non-negativity constraint
    pub fn with_non_negative(mut self) -> Self {
        self.constraints.push(ParameterConstraint::NonNegative);
        self
    }

    /// Add unit sphere constraint
    pub fn with_unit_sphere(mut self) -> Self {
        self.constraints.push(ParameterConstraint::UnitSphere);
        self
    }

    /// Add simplex constraint (sum to 1, all non-negative)
    pub fn with_simplex(mut self) -> Self {
        self.constraints.push(ParameterConstraint::Simplex);
        self
    }

    /// Add orthogonal constraint for matrices
    pub fn with_orthogonal(mut self, tolerance: A) -> Self {
        self.constraints
            .push(ParameterConstraint::Orthogonal { tolerance });
        self
    }

    /// Add positive definite constraint for symmetric matrices
    pub fn with_positive_definite(mut self, mineigenvalue: A) -> Self {
        self.constraints
            .push(ParameterConstraint::PositiveDefinite { mineigenvalue });
        self
    }

    /// Add spectral norm constraint
    pub fn with_spectral_norm(mut self, maxnorm: A) -> Self {
        self.constraints
            .push(ParameterConstraint::SpectralNorm { maxnorm });
        self
    }

    /// Add nuclear norm constraint
    pub fn with_nuclear_norm(mut self, maxnorm: A) -> Self {
        self.constraints
            .push(ParameterConstraint::NuclearNorm { maxnorm });
        self
    }

    /// Add custom constraint
    pub fn with_custom_constraint(mut self, name: String) -> Self {
        self.constraints.push(ParameterConstraint::Custom { name });
        self
    }
}

/// A parameter group with its own configuration
#[derive(Debug)]
pub struct ParameterGroup<A: Float, D: Dimension> {
    /// Unique identifier for this group
    pub id: usize,
    /// Parameters in this group
    pub params: Vec<Array<A, D>>,
    /// Configuration for this group
    pub config: ParameterGroupConfig<A>,
    /// Internal state for optimization (optimizer-specific)
    pub state: HashMap<String, Vec<Array<A, D>>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> ParameterGroup<A, D> {
    /// Create a new parameter group
    pub fn new(id: usize, params: Vec<Array<A, D>>, config: ParameterGroupConfig<A>) -> Self {
        Self {
            id,
            params,
            config,
            state: HashMap::new(),
        }
    }

    /// Get the number of parameters in this group
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Get learning rate for this group
    pub fn learning_rate(&self, default: A) -> A {
        self.config.learning_rate.unwrap_or(default)
    }

    /// Get weight decay for this group
    pub fn weight_decay(&self, default: A) -> A {
        self.config.weight_decay.unwrap_or(default)
    }

    /// Get momentum for this group
    pub fn momentum(&self, default: A) -> A {
        self.config.momentum.unwrap_or(default)
    }

    /// Get custom parameter
    pub fn get_custom_param(&self, key: &str, default: A) -> A {
        self.config
            .custom_params
            .get(key)
            .copied()
            .unwrap_or(default)
    }

    /// Apply constraints to all parameters in this group
    pub fn apply_constraints(&mut self) -> Result<()>
    where
        A: ScalarOperand + Send + Sync,
    {
        for constraint in &self.config.constraints {
            for param in &mut self.params {
                constraint.apply(param)?;
            }
        }
        Ok(())
    }

    /// Apply constraints to a specific parameter
    pub fn apply_constraints_to_param(&self, param: &mut Array<A, D>) -> Result<()>
    where
        A: ScalarOperand + Send + Sync,
    {
        for constraint in &self.config.constraints {
            constraint.apply(param)?;
        }
        Ok(())
    }

    /// Get the constraints for this group
    pub fn constraints(&self) -> &[ParameterConstraint<A>] {
        &self.config.constraints
    }
}

/// Optimizer with parameter group support
pub trait GroupedOptimizer<A: Float + ScalarOperand + Debug, D: Dimension>:
    Optimizer<A, D>
{
    /// Add a parameter group
    fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> Result<usize>;

    /// Get parameter group by ID
    fn get_group(&self, groupid: usize) -> Result<&ParameterGroup<A, D>>;

    /// Get mutable parameter group by ID
    fn get_group_mut(&mut self, groupid: usize) -> Result<&mut ParameterGroup<A, D>>;

    /// Get all parameter groups
    fn groups(&self) -> &[ParameterGroup<A, D>];

    /// Get all parameter groups mutably
    fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>];

    /// Step for a specific group
    fn step_group(
        &mut self,
        group_id: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>>;

    /// Set learning rate for a specific group
    fn set_group_learning_rate(&mut self, groupid: usize, lr: A) -> Result<()>;

    /// Set weight decay for a specific group
    fn set_group_weight_decay(&mut self, groupid: usize, wd: A) -> Result<()>;
}

/// Helper struct for managing parameter groups
#[derive(Debug)]
pub struct GroupManager<A: Float, D: Dimension> {
    groups: Vec<ParameterGroup<A, D>>,
    next_id: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> Default for GroupManager<A, D> {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            next_id: 0,
        }
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> GroupManager<A, D> {
    /// Create a new group manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new parameter group
    pub fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.push(ParameterGroup::new(id, params, config));
        id
    }

    /// Get group by ID
    pub fn get_group(&self, id: usize) -> Result<&ParameterGroup<A, D>> {
        self.groups
            .iter()
            .find(|g| g.id == id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Group {id} not found")))
    }

    /// Get mutable group by ID
    pub fn get_group_mut(&mut self, id: usize) -> Result<&mut ParameterGroup<A, D>> {
        self.groups
            .iter_mut()
            .find(|g| g.id == id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Group {id} not found")))
    }

    /// Get all groups
    pub fn groups(&self) -> &[ParameterGroup<A, D>] {
        &self.groups
    }

    /// Get all groups mutably
    pub fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>] {
        &mut self.groups
    }

    /// Get total number of parameters across all groups
    pub fn total_params(&self) -> usize {
        self.groups.iter().map(|g| g.num_params()).sum()
    }
}

/// State checkpointing for parameter management
pub mod checkpointing {
    use super::*;

    /// Checkpoint data for optimizer state
    #[derive(Debug, Clone)]
    pub struct OptimizerCheckpoint<A: Float, D: Dimension> {
        /// Step number
        pub step: usize,
        /// Parameter groups
        pub groups: Vec<ParameterGroupCheckpoint<A, D>>,
        /// Global optimizer state
        pub global_state: HashMap<String, String>,
        /// Metadata
        pub metadata: CheckpointMetadata,
    }

    /// Checkpoint data for a parameter group
    #[derive(Debug, Clone)]
    pub struct ParameterGroupCheckpoint<A: Float, D: Dimension> {
        /// Group ID
        pub id: usize,
        /// Parameters
        pub params: Vec<Array<A, D>>,
        /// Group configuration
        pub config: ParameterGroupConfig<A>,
        /// Optimizer-specific state for this group
        pub state: HashMap<String, Vec<Array<A, D>>>,
    }

    /// Metadata for checkpoints
    #[derive(Debug, Clone)]
    pub struct CheckpointMetadata {
        /// Timestamp when checkpoint was created
        pub timestamp: String,
        /// Version of the optimizer
        pub optimizerversion: String,
        /// Custom metadata
        pub custom: HashMap<String, String>,
    }

    impl CheckpointMetadata {
        /// Create new metadata with current timestamp
        pub fn new(optimizerversion: String) -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_string();

            Self {
                timestamp,
                optimizerversion,
                custom: HashMap::new(),
            }
        }

        /// Add custom metadata
        pub fn with_custom(mut self, key: String, value: String) -> Self {
            self.custom.insert(key, value);
            self
        }
    }

    /// Trait for optimizers that support checkpointing
    pub trait Checkpointable<
        A: Float + ToString + std::fmt::Display + std::str::FromStr,
        D: Dimension,
    >
    {
        /// Create a checkpoint of the current optimizer state
        fn create_checkpoint(&self) -> Result<OptimizerCheckpoint<A, D>>;

        /// Restore optimizer state from a checkpoint
        fn restore_checkpoint(&mut self, checkpoint: &OptimizerCheckpoint<A, D>) -> Result<()>;

        /// Save checkpoint to file (simple text format)
        fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            use std::fs::File;
            use std::io::{BufWriter, Write};

            let checkpoint = self.create_checkpoint()?;
            let path = path.as_ref();

            // Create the file
            let file = File::create(path).map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to create checkpoint file: {e}"))
            })?;
            let mut writer = BufWriter::new(file);

            // Write header
            writeln!(writer, "# ScirS2 Optimizer Checkpoint v1.0").map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write checkpoint header: {e}"))
            })?;
            writeln!(writer, "# Timestamp: {}", checkpoint.metadata.timestamp).map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write timestamp: {e}"))
            })?;
            writeln!(
                writer,
                "# Optimizer Version: {}",
                checkpoint.metadata.optimizerversion
            )
            .map_err(|e| OptimError::InvalidConfig(format!("Failed to write version: {e}")))?;
            writeln!(writer, "# Step: {}", checkpoint.step)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write step: {e}")))?;
            writeln!(writer)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write newline: {e}")))?;

            // Write custom metadata
            writeln!(writer, "[METADATA]").map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write metadata section: {e}"))
            })?;
            for (key, value) in &checkpoint.metadata.custom {
                writeln!(writer, "{}={}", key, value).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write metadata entry: {e}"))
                })?;
            }
            writeln!(writer)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write newline: {e}")))?;

            // Write global state
            writeln!(writer, "[GLOBAL_STATE]").map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write global state section: {e}"))
            })?;
            for (key, value) in &checkpoint.global_state {
                writeln!(writer, "{}={}", key, value).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write global state entry: {e}"))
                })?;
            }
            writeln!(writer)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write newline: {e}")))?;

            // Write parameter groups
            writeln!(writer, "[GROUPS]").map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write groups section: {e}"))
            })?;
            writeln!(writer, "count={}", checkpoint.groups.len()).map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to write group count: {e}"))
            })?;
            writeln!(writer)
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write newline: {e}")))?;

            for group in &checkpoint.groups {
                // Write group header
                writeln!(writer, "[GROUP_{}]", group.id).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write group header: {e}"))
                })?;

                // Write group config
                writeln!(
                    writer,
                    "learning_rate={}",
                    group
                        .config
                        .learning_rate
                        .map(|lr| lr.to_string())
                        .unwrap_or_else(|| "None".to_string())
                )
                .map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write learning rate: {e}"))
                })?;
                writeln!(
                    writer,
                    "weight_decay={}",
                    group
                        .config
                        .weight_decay
                        .map(|wd| wd.to_string())
                        .unwrap_or_else(|| "None".to_string())
                )
                .map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write weight decay: {e}"))
                })?;
                writeln!(
                    writer,
                    "momentum={}",
                    group
                        .config
                        .momentum
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "None".to_string())
                )
                .map_err(|e| OptimError::InvalidConfig(format!("Failed to write momentum: {e}")))?;

                // Write custom params
                writeln!(
                    writer,
                    "custom_params_count={}",
                    group.config.custom_params.len()
                )
                .map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write custom params count: {e}"))
                })?;
                for (key, value) in &group.config.custom_params {
                    writeln!(writer, "custom_{}={}", key, value).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write custom param: {e}"))
                    })?;
                }

                // Write parameters
                writeln!(writer, "param_count={}", group.params.len()).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write param count: {e}"))
                })?;
                for (i, param) in group.params.iter().enumerate() {
                    writeln!(writer, "param_{}shape={:?}", i, param.shape()).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write param shape: {e}"))
                    })?;
                    write!(writer, "param_{}_data=", i).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write param data label: {e}"))
                    })?;

                    // Write array data as space-separated values
                    for (j, &val) in param.iter().enumerate() {
                        if j > 0 {
                            write!(writer, " ").map_err(|e| {
                                OptimError::InvalidConfig(format!("Failed to write space: {e}"))
                            })?;
                        }
                        write!(writer, "{}", val).map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to write value: {e}"))
                        })?;
                    }
                    writeln!(writer).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write newline: {e}"))
                    })?;
                }

                // Write optimizer state
                writeln!(writer, "state_count={}", group.state.len()).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write state count: {e}"))
                })?;
                for (state_name, state_arrays) in &group.state {
                    writeln!(writer, "state_name={}", state_name).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write state name: {e}"))
                    })?;
                    writeln!(writer, "state_array_count={}", state_arrays.len()).map_err(|e| {
                        OptimError::InvalidConfig(format!("Failed to write state array count: {e}"))
                    })?;
                    for (i, array) in state_arrays.iter().enumerate() {
                        writeln!(writer, "state_{}shape={:?}", i, array.shape()).map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to write state shape: {e}"))
                        })?;
                        write!(writer, "state_{}_data=", i).map_err(|e| {
                            OptimError::InvalidConfig(format!(
                                "Failed to write state data label: {}",
                                e
                            ))
                        })?;

                        // Write array data
                        for (j, &val) in array.iter().enumerate() {
                            if j > 0 {
                                write!(writer, " ").map_err(|e| {
                                    OptimError::InvalidConfig(format!(
                                        "Failed to write space: {}",
                                        e
                                    ))
                                })?;
                            }
                            write!(writer, "{}", val).map_err(|e| {
                                OptimError::InvalidConfig(format!("Failed to write value: {e}"))
                            })?;
                        }
                        writeln!(writer).map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to write newline: {e}"))
                        })?;
                    }
                }

                writeln!(writer).map_err(|e| {
                    OptimError::InvalidConfig(format!("Failed to write newline: {e}"))
                })?;
            }

            writer.flush().map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to flush checkpoint file: {e}"))
            })?;

            Ok(())
        }

        /// Load checkpoint from file (simple text format)
        fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
            use std::fs::File;
            use std::io::{BufRead, BufReader};

            let path = path.as_ref();
            let file = File::open(path).map_err(|e| {
                OptimError::InvalidConfig(format!("Failed to open checkpoint file: {e}"))
            })?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            // Read header
            let mut step = 0;
            let mut optimizerversion = String::new();
            let mut timestamp = String::new();

            while let Some(Ok(line)) = lines.next() {
                if line.starts_with("# Step: ") {
                    step = line.trim_start_matches("# Step: ").parse().map_err(|_| {
                        OptimError::InvalidConfig("Invalid step format".to_string())
                    })?;
                } else if line.starts_with("# Optimizer Version: ") {
                    optimizerversion = line.trim_start_matches("# Optimizer Version: ").to_string();
                } else if line.starts_with("# Timestamp: ") {
                    timestamp = line.trim_start_matches("# Timestamp: ").to_string();
                } else if line.starts_with("[METADATA]") {
                    break;
                }
            }

            // Read metadata
            let mut custom_metadata = HashMap::new();
            while let Some(Ok(line)) = lines.next() {
                if line.is_empty() || line.starts_with("[") {
                    if line.starts_with("[GLOBAL_STATE]") {
                        break;
                    }
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    custom_metadata.insert(key.to_string(), value.to_string());
                }
            }

            // Read global state
            let mut global_state = HashMap::new();
            while let Some(Ok(line)) = lines.next() {
                if line.is_empty() || line.starts_with("[") {
                    if line.starts_with("[GROUPS]") {
                        break;
                    }
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    global_state.insert(key.to_string(), value.to_string());
                }
            }

            // Read groups count
            let mut group_count = 0;
            while let Some(Ok(line)) = lines.next() {
                if line.starts_with("count=") {
                    group_count = line.trim_start_matches("count=").parse().map_err(|_| {
                        OptimError::InvalidConfig("Invalid group count".to_string())
                    })?;
                    break;
                }
            }

            // Read parameter groups
            let mut groups = Vec::new();
            for _ in 0..group_count {
                // Skip to group header
                let mut group_id = 0;
                while let Some(Ok(line)) = lines.next() {
                    if line.starts_with("[GROUP_") {
                        let id_str = line.trim_start_matches("[GROUP_").trim_end_matches(']');
                        group_id = id_str.parse().map_err(|_| {
                            OptimError::InvalidConfig("Invalid group ID".to_string())
                        })?;
                        break;
                    }
                }

                // Read group config
                let mut learning_rate = None;
                let mut weight_decay = None;
                let mut momentum = None;
                let mut custom_params = HashMap::new();
                let mut _custom_params_count = 0;

                while let Some(Ok(line)) = lines.next() {
                    if line.starts_with("learning_rate=") {
                        let val_str = line.trim_start_matches("learning_rate=");
                        if val_str != "None" {
                            learning_rate = Some(A::from_str(val_str).map_err(|_| {
                                OptimError::InvalidConfig("Invalid learning rate".to_string())
                            })?);
                        }
                    } else if line.starts_with("weight_decay=") {
                        let val_str = line.trim_start_matches("weight_decay=");
                        if val_str != "None" {
                            weight_decay = Some(A::from_str(val_str).map_err(|_| {
                                OptimError::InvalidConfig("Invalid weight decay".to_string())
                            })?);
                        }
                    } else if line.starts_with("momentum=") {
                        let val_str = line.trim_start_matches("momentum=");
                        if val_str != "None" {
                            momentum = Some(A::from_str(val_str).map_err(|_| {
                                OptimError::InvalidConfig("Invalid momentum".to_string())
                            })?);
                        }
                    } else if line.starts_with("custom_params_count=") {
                        _custom_params_count = line
                            .trim_start_matches("custom_params_count=")
                            .parse()
                            .map_err(|_| {
                                OptimError::InvalidConfig("Invalid custom params count".to_string())
                            })?;
                    } else if line.starts_with("custom_") {
                        if let Some((key_with_prefix, value)) = line.split_once('=') {
                            let key = key_with_prefix.trim_start_matches("custom_");
                            custom_params.insert(
                                key.to_string(),
                                A::from_str(value).map_err(|_| {
                                    OptimError::InvalidConfig(
                                        "Invalid custom param value".to_string(),
                                    )
                                })?,
                            );
                        }
                    } else if line.starts_with("param_count=") {
                        break;
                    }
                }

                // Create group config
                let config = ParameterGroupConfig {
                    learning_rate,
                    weight_decay,
                    momentum,
                    constraints: Vec::new(), // Constraints are not persisted in this simple format
                    custom_params,
                };

                // Read parameters
                let param_count: usize = lines
                    .next()
                    .ok_or_else(|| OptimError::InvalidConfig("Missing param count".to_string()))?
                    .map_err(|e| OptimError::InvalidConfig(format!("Failed to read line: {e}")))?
                    .trim_start_matches("param_count=")
                    .parse()
                    .map_err(|_| OptimError::InvalidConfig("Invalid param count".to_string()))?;

                let mut params = Vec::new();
                for i in 0..param_count {
                    // Read shape
                    let shape_line = lines
                        .next()
                        .ok_or_else(|| {
                            OptimError::InvalidConfig("Missing param shape".to_string())
                        })?
                        .map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                        })?;

                    let shape_str = shape_line
                        .trim_start_matches(&format!("param_{}shape=", i))
                        .trim_start_matches('[')
                        .trim_end_matches(']');

                    let shape: Vec<usize> = shape_str
                        .split(", ")
                        .map(|s| {
                            s.parse()
                                .map_err(|_| OptimError::InvalidConfig("Invalid shape".to_string()))
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Read data
                    let data_line = lines
                        .next()
                        .ok_or_else(|| OptimError::InvalidConfig("Missing param data".to_string()))?
                        .map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                        })?;

                    let data_str = data_line.trim_start_matches(&format!("param_{}_data=", i));
                    let data: Vec<A> = data_str
                        .split(' ')
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            A::from_str(s).map_err(|_| {
                                OptimError::InvalidConfig("Invalid data value".to_string())
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Create array from shape and data with dynamic dimensions
                    let array: Array<A, scirs2_core::ndarray::IxDyn> =
                        Array::from_shape_vec(shape, data).map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to create array: {e}"))
                        })?;
                    params.push(array);
                }

                // Read optimizer state
                let state_count: usize = lines
                    .next()
                    .ok_or_else(|| OptimError::InvalidConfig("Missing state count".to_string()))?
                    .map_err(|e| OptimError::InvalidConfig(format!("Failed to read line: {e}")))?
                    .trim_start_matches("state_count=")
                    .parse()
                    .map_err(|_| OptimError::InvalidConfig("Invalid state count".to_string()))?;

                let mut state = HashMap::new();
                for _ in 0..state_count {
                    let state_name = lines
                        .next()
                        .ok_or_else(|| OptimError::InvalidConfig("Missing state name".to_string()))?
                        .map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                        })?
                        .trim_start_matches("state_name=")
                        .to_string();

                    let array_count: usize = lines
                        .next()
                        .ok_or_else(|| {
                            OptimError::InvalidConfig("Missing state array count".to_string())
                        })?
                        .map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                        })?
                        .trim_start_matches("state_array_count=")
                        .parse()
                        .map_err(|_| {
                            OptimError::InvalidConfig("Invalid state array count".to_string())
                        })?;

                    let mut state_arrays = Vec::new();
                    for i in 0..array_count {
                        // Read shape
                        let shape_line = lines
                            .next()
                            .ok_or_else(|| {
                                OptimError::InvalidConfig("Missing state shape".to_string())
                            })?
                            .map_err(|e| {
                                OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                            })?;

                        let shape_str = shape_line
                            .trim_start_matches(&format!("state_{}shape=", i))
                            .trim_start_matches('[')
                            .trim_end_matches(']');

                        let shape: Vec<usize> = shape_str
                            .split(", ")
                            .map(|s| {
                                s.parse().map_err(|_| {
                                    OptimError::InvalidConfig("Invalid state shape".to_string())
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        // Read data
                        let data_line = lines
                            .next()
                            .ok_or_else(|| {
                                OptimError::InvalidConfig("Missing state data".to_string())
                            })?
                            .map_err(|e| {
                                OptimError::InvalidConfig(format!("Failed to read line: {e}"))
                            })?;

                        let data_str = data_line.trim_start_matches(&format!("state_{}_data=", i));
                        let data: Vec<A> = data_str
                            .split(' ')
                            .filter(|s| !s.is_empty())
                            .map(|s| {
                                A::from_str(s).map_err(|_| {
                                    OptimError::InvalidConfig("Invalid state value".to_string())
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        // Create array with dynamic dimensions
                        let array = Array::from_shape_vec(shape, data).map_err(|e| {
                            OptimError::InvalidConfig(format!("Failed to create state array: {e}"))
                        })?;
                        state_arrays.push(array);
                    }

                    state.insert(state_name, state_arrays);
                }

                // Create group checkpoint
                groups.push(ParameterGroupCheckpoint {
                    id: group_id,
                    params,
                    config,
                    state,
                });
            }

            // Create checkpoint metadata
            let mut metadata = CheckpointMetadata::new(optimizerversion);
            metadata.timestamp = timestamp;
            metadata.custom = custom_metadata;

            // Create the checkpoint with dynamic dimensions
            let _dyn_checkpoint = OptimizerCheckpoint::<A, scirs2_core::ndarray::IxDyn> {
                step,
                groups,
                global_state,
                metadata,
            };

            // Dimension conversion from IxDyn to D is a known limitation
            // Checkpoints are saved with dynamic dimensions (IxDyn) for flexibility,
            // but loading requires compile-time dimension type D.
            //
            // DESIGN NOTE: This is intentional for v1.0.0 to maintain type safety.
            // Users should use save_checkpoint() and create a new optimizer instance
            // rather than load_checkpoint() for cross-session restoration.
            //
            // For same-session checkpoint restoration, use CheckpointManager's
            // in-memory storage which preserves dimension types.
            //
            // Future enhancement (v1.1.0+): Add dimension-specific load methods
            // or provide a type-erased checkpoint interface.
            Err(OptimError::InvalidConfig(
                "Checkpoint loading from file with dimension type conversion is not supported in v1.0.0. \
                 Use CheckpointManager for in-memory checkpoints, or save/load with consistent dimension types. \
                 See documentation for checkpoint best practices.".to_string(),
            ))
        }
    }

    /// In-memory checkpoint manager
    #[derive(Debug)]
    pub struct CheckpointManager<A: Float, D: Dimension> {
        checkpoints: HashMap<String, OptimizerCheckpoint<A, D>>,
        _maxcheckpoints: usize,
        checkpoint_keys: Vec<String>, // To maintain order for LRU eviction
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> CheckpointManager<A, D> {
        /// Create a new checkpoint manager
        pub fn new() -> Self {
            Self {
                checkpoints: HashMap::new(),
                _maxcheckpoints: 10,
                checkpoint_keys: Vec::new(),
            }
        }

        /// Create a new checkpoint manager with maximum number of checkpoints
        pub fn with_max_checkpoints(_maxcheckpoints: usize) -> Self {
            Self {
                checkpoints: HashMap::new(),
                _maxcheckpoints,
                checkpoint_keys: Vec::new(),
            }
        }

        /// Store a checkpoint with a given key
        pub fn store_checkpoint(&mut self, key: String, checkpoint: OptimizerCheckpoint<A, D>) {
            // If key already exists, update it
            if self.checkpoints.contains_key(&key) {
                self.checkpoints.insert(key.clone(), checkpoint);
                return;
            }

            // If we're at capacity, remove oldest checkpoint
            if self.checkpoints.len() >= self._maxcheckpoints {
                if let Some(oldest_key) = self.checkpoint_keys.first().cloned() {
                    self.checkpoints.remove(&oldest_key);
                    self.checkpoint_keys.retain(|k| k != &oldest_key);
                }
            }

            // Add new checkpoint
            self.checkpoints.insert(key.clone(), checkpoint);
            self.checkpoint_keys.push(key);
        }

        /// Retrieve a checkpoint by key
        pub fn get_checkpoint(&self, key: &str) -> Option<&OptimizerCheckpoint<A, D>> {
            self.checkpoints.get(key)
        }

        /// Remove a checkpoint by key
        pub fn remove_checkpoint(&mut self, key: &str) -> Option<OptimizerCheckpoint<A, D>> {
            self.checkpoint_keys.retain(|k| k != key);
            self.checkpoints.remove(key)
        }

        /// List all checkpoint keys
        pub fn list_checkpoints(&self) -> &[String] {
            &self.checkpoint_keys
        }

        /// Clear all checkpoints
        pub fn clear(&mut self) {
            self.checkpoints.clear();
            self.checkpoint_keys.clear();
        }

        /// Get number of stored checkpoints
        pub fn len(&self) -> usize {
            self.checkpoints.len()
        }

        /// Check if manager is empty
        pub fn is_empty(&self) -> bool {
            self.checkpoints.is_empty()
        }
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension + Send + Sync> Default
        for CheckpointManager<A, D>
    {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Utility functions for checkpointing
    pub mod utils {
        use super::*;

        /// Create a checkpoint from parameter groups
        pub fn create_checkpoint_from_groups<A: Float + ScalarOperand + Debug, D: Dimension>(
            step: usize,
            groups: &[ParameterGroup<A, D>],
            global_state: HashMap<String, String>,
            optimizerversion: String,
        ) -> OptimizerCheckpoint<A, D> {
            let group_checkpoints = groups
                .iter()
                .map(|group| ParameterGroupCheckpoint {
                    id: group.id,
                    params: group.params.clone(),
                    config: group.config.clone(),
                    state: group.state.clone(),
                })
                .collect();

            OptimizerCheckpoint {
                step,
                groups: group_checkpoints,
                global_state,
                metadata: CheckpointMetadata::new(optimizerversion),
            }
        }

        /// Validate checkpoint compatibility
        pub fn validate_checkpoint<A: Float, D: Dimension>(
            checkpoint: &OptimizerCheckpoint<A, D>,
            expected_groups: usize,
        ) -> Result<()> {
            if checkpoint.groups.len() != expected_groups {
                return Err(OptimError::InvalidConfig(format!(
                    "Checkpoint has {} groups, expected {expected_groups}",
                    checkpoint.groups.len()
                )));
            }

            // Validate that all group IDs are unique
            let mut ids = std::collections::HashSet::new();
            for group in &checkpoint.groups {
                if !ids.insert(group.id) {
                    return Err(OptimError::InvalidConfig(format!(
                        "Duplicate group ID {} in checkpoint",
                        group.id
                    )));
                }
            }

            Ok(())
        }

        /// Get checkpoint summary information
        pub fn checkpoint_summary<A: Float, D: Dimension>(
            checkpoint: &OptimizerCheckpoint<A, D>,
        ) -> String {
            let total_params: usize = checkpoint
                .groups
                .iter()
                .map(|g| g.params.iter().map(|p| p.len()).sum::<usize>())
                .sum();

            format!(
                "Checkpoint at step {}: {} groups, {} total parameters, created at {}",
                checkpoint.step,
                checkpoint.groups.len(),
                total_params,
                checkpoint.metadata.timestamp
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_parameter_group_config() {
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_weight_decay(0.0001)
            .with_momentum(0.9)
            .with_custom_param("beta1".to_string(), 0.9)
            .with_custom_param("beta2".to_string(), 0.999);

        assert_eq!(config.learning_rate, Some(0.01));
        assert_eq!(config.weight_decay, Some(0.0001));
        assert_eq!(config.momentum, Some(0.9));
        assert_eq!(config.custom_params.get("beta1"), Some(&0.9));
        assert_eq!(config.custom_params.get("beta2"), Some(&0.999));
    }

    #[test]
    fn test_parameter_group() {
        let params = vec![Array1::zeros(5), Array1::ones(3)];
        let config = ParameterGroupConfig::new().with_learning_rate(0.01);

        let group = ParameterGroup::new(0, params, config);

        assert_eq!(group.id, 0);
        assert_eq!(group.num_params(), 2);
        assert_eq!(group.learning_rate(0.001), 0.01);
        assert_eq!(group.weight_decay(0.0), 0.0);
    }

    #[test]
    fn test_group_manager() {
        let mut manager: GroupManager<f64, scirs2_core::ndarray::Ix1> = GroupManager::new();

        // Add first group
        let params1 = vec![Array1::zeros(5)];
        let config1 = ParameterGroupConfig::new().with_learning_rate(0.01);
        let id1 = manager.add_group(params1, config1);

        // Add second group
        let params2 = vec![Array1::ones(3), Array1::zeros(4)];
        let config2 = ParameterGroupConfig::new().with_learning_rate(0.001);
        let id2 = manager.add_group(params2, config2);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(manager.groups().len(), 2);
        assert_eq!(manager.total_params(), 3);

        // Test group access
        let group1 = manager.get_group(id1).expect("unwrap failed");
        assert_eq!(group1.learning_rate(0.0), 0.01);

        let group2 = manager.get_group(id2).expect("unwrap failed");
        assert_eq!(group2.learning_rate(0.0), 0.001);
    }

    #[test]
    fn test_parameter_constraints() {
        use approx::assert_relative_eq;

        // Test value clipping
        let mut params = Array1::from_vec(vec![-2.0, 0.5, 3.0]);
        let clip_constraint = ParameterConstraint::ValueClip { min: 0.0, max: 1.0 };
        clip_constraint.apply(&mut params).expect("unwrap failed");
        assert_eq!(params.as_slice().expect("unwrap failed"), &[0.0, 0.5, 1.0]);

        // Test L2 norm constraint
        let mut params = Array1::from_vec(vec![3.0, 4.0]); // norm = 5
        let l2_constraint = ParameterConstraint::L2NormConstraint { maxnorm: 2.0 };
        l2_constraint.apply(&mut params).expect("unwrap failed");
        let new_norm = params.mapv(|x| x * x).sum().sqrt();
        assert_relative_eq!(new_norm, 2.0, epsilon = 1e-6);

        // Test non-negativity constraint
        let mut params = Array1::from_vec(vec![-1.0, 2.0, -3.0]);
        let non_neg_constraint = ParameterConstraint::NonNegative;
        non_neg_constraint
            .apply(&mut params)
            .expect("unwrap failed");
        assert_eq!(params.as_slice().expect("unwrap failed"), &[0.0, 2.0, 0.0]);

        // Test unit sphere constraint
        let mut params = Array1::from_vec(vec![3.0, 4.0]); // norm = 5
        let unit_sphere_constraint = ParameterConstraint::UnitSphere;
        unit_sphere_constraint
            .apply(&mut params)
            .expect("unwrap failed");
        let new_norm = params.mapv(|x| x * x).sum().sqrt();
        assert_relative_eq!(new_norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parameter_group_with_constraints() {
        let params = vec![Array1::from_vec(vec![-2.0, 3.0])];
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_value_clip(0.0, 1.0);

        let mut group = ParameterGroup::new(0, params, config);

        // Apply constraints
        group.apply_constraints().expect("unwrap failed");

        // Check that constraints were applied
        assert_eq!(
            group.params[0].as_slice().expect("unwrap failed"),
            &[0.0, 1.0]
        );
    }

    #[test]
    fn test_parameter_config_builder() {
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_l2_norm_constraint(1.0)
            .with_non_negative()
            .with_custom_param("beta".to_string(), 0.9);

        assert_eq!(config.learning_rate, Some(0.01));
        assert_eq!(config.constraints.len(), 2);
        assert_eq!(config.custom_params.get("beta"), Some(&0.9));
    }

    #[test]
    fn test_simplex_constraint() {
        use approx::assert_relative_eq;

        // Test simplex constraint with positive values
        let mut params = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        let simplex_constraint = ParameterConstraint::Simplex;
        simplex_constraint
            .apply(&mut params)
            .expect("unwrap failed");

        // Check that values sum to 1 and are non-negative
        let sum: f64 = params.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(params.iter().all(|&x| x >= 0.0));

        // Values should be proportional to original
        assert_relative_eq!(params[0], 0.2, epsilon = 1e-6); // 2/10
        assert_relative_eq!(params[1], 0.3, epsilon = 1e-6); // 3/10
        assert_relative_eq!(params[2], 0.5, epsilon = 1e-6); // 5/10
    }

    #[test]
    fn test_simplex_constraint_with_negatives() {
        use approx::assert_relative_eq;

        // Test simplex constraint with negative values
        let mut params = Array1::from_vec(vec![-1.0, 2.0, 3.0]);
        let simplex_constraint = ParameterConstraint::Simplex;
        simplex_constraint
            .apply(&mut params)
            .expect("unwrap failed");

        // Check that values sum to 1 and are non-negative
        let sum: f64 = params.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(params.iter().all(|&x| x >= 0.0));

        // Negative value should become 0, others normalized
        assert_relative_eq!(params[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(params[1], 0.4, epsilon = 1e-6); // 2/5
        assert_relative_eq!(params[2], 0.6, epsilon = 1e-6); // 3/5
    }

    #[test]
    fn test_simplex_constraint_all_zeros() {
        use approx::assert_relative_eq;

        // Test simplex constraint with all zeros
        let mut params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let simplex_constraint = ParameterConstraint::Simplex;
        simplex_constraint
            .apply(&mut params)
            .expect("unwrap failed");

        // Should result in uniform distribution
        let sum: f64 = params.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        for &val in params.iter() {
            assert_relative_eq!(val, 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_spectral_norm_constraint() {
        use approx::assert_relative_eq;
        use scirs2_core::ndarray::arr2;

        // A 1x2 matrix has a single nonzero singular value σ_max = ‖row‖ = 5.
        let mut params = arr2(&[[3.0, 4.0]]);
        let spectral_constraint = ParameterConstraint::SpectralNorm { maxnorm: 2.0 };
        spectral_constraint
            .apply(&mut params)
            .expect("unwrap failed");

        // After scaling by 2/5 the spectral norm equals the cap.
        let sigma = power_iteration_spectral_norm(&params);
        assert_relative_eq!(sigma, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_nuclear_norm_constraint() {
        use approx::assert_relative_eq;

        // Test nuclear norm constraint (approximated with L1 norm)
        let mut params = Array1::from_vec(vec![3.0, -4.0, 2.0]); // L1 norm = 9
        let nuclear_constraint = ParameterConstraint::NuclearNorm { maxnorm: 3.0 };
        nuclear_constraint
            .apply(&mut params)
            .expect("unwrap failed");

        let new_l1_norm = params.mapv(|x| x.abs()).sum();
        assert_relative_eq!(new_l1_norm, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_orthogonal_constraint_error() {
        // Test that orthogonal constraint returns appropriate error
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let orthogonal_constraint = ParameterConstraint::Orthogonal { tolerance: 1e-6 };
        let result = orthogonal_constraint.apply(&mut params);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D arrays"));
    }

    #[test]
    fn test_positive_definite_constraint_error() {
        // A 1D array is not a matrix, so the positive-definite constraint errors.
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let pd_constraint = ParameterConstraint::PositiveDefinite {
            mineigenvalue: 0.01,
        };
        let result = pd_constraint.apply(&mut params);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D arrays"));
    }

    #[test]
    fn test_enhanced_config_builder() {
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_simplex()
            .with_spectral_norm(2.0)
            .with_nuclear_norm(1.5)
            .with_custom_constraint("my_constraint".to_string());

        assert_eq!(config.learning_rate, Some(0.01));
        assert_eq!(config.constraints.len(), 4);

        // Check that the right constraint types were added
        match &config.constraints[0] {
            ParameterConstraint::Simplex => (),
            _ => panic!("Expected Simplex constraint"),
        }

        match &config.constraints[1] {
            ParameterConstraint::SpectralNorm { maxnorm } => {
                assert_eq!(*maxnorm, 2.0);
            }
            _ => panic!("Expected SpectralNorm constraint"),
        }
    }

    #[test]
    fn test_constraint_combination() {
        use approx::assert_relative_eq;

        // Test applying multiple constraints in sequence
        let params = vec![Array1::from_vec(vec![-1.0, 2.0, 3.0])];
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_non_negative()
            .with_simplex();

        let mut group = ParameterGroup::new(0, params, config);

        // Apply constraints
        group.apply_constraints().expect("unwrap failed");

        // Check that both non-negative and simplex constraints were applied
        let result = &group.params[0];
        let sum: f64 = result.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(result.iter().all(|&x| x >= 0.0));

        // Should be [0, 0.4, 0.6] after non-negative then simplex
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.4, epsilon = 1e-6);
        assert_relative_eq!(result[2], 0.6, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // Matrix constraints: Orthogonal, SpectralNorm, PositiveDefinite.
    // -----------------------------------------------------------------------

    /// Compute MᵀM for a 2D array (used to verify orthonormal columns).
    fn gram_matrix(m: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = m.dim();
        let mut g = Array2::<f64>::zeros((cols, cols));
        for i in 0..cols {
            for j in 0..cols {
                let mut dot = 0.0;
                for k in 0..rows {
                    dot += m[[k, i]] * m[[k, j]];
                }
                g[[i, j]] = dot;
            }
        }
        g
    }

    #[test]
    fn test_orthogonal_constraint_square() {
        use approx::assert_abs_diff_eq;
        use scirs2_core::ndarray::arr2;

        // Non-orthonormal 3x3 matrix.
        let mut params = arr2(&[[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]);
        let constraint = ParameterConstraint::Orthogonal { tolerance: 1e-10 };
        constraint.apply(&mut params).expect("constraint failed");

        // Columns must be orthonormal: MᵀM ≈ I.
        let g = gram_matrix(&params);
        for i in 0..3 {
            for j in 0..3 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(g[[i, j]], target, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_orthogonal_constraint_tall() {
        use approx::assert_abs_diff_eq;
        use scirs2_core::ndarray::arr2;

        // Non-square 4x2 matrix: orthonormalize the 2 columns.
        let mut params = arr2(&[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let constraint = ParameterConstraint::Orthogonal { tolerance: 1e-10 };
        constraint.apply(&mut params).expect("constraint failed");

        // MᵀM (2x2) must be the identity.
        let g = gram_matrix(&params);
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(g[[i, j]], target, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_orthogonal_constraint_already_orthonormal_unchanged() {
        use approx::assert_abs_diff_eq;
        use scirs2_core::ndarray::arr2;

        // Identity is already orthonormal; must be left untouched (early return).
        let mut params = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let original = params.clone();
        let constraint = ParameterConstraint::Orthogonal { tolerance: 1e-8 };
        constraint.apply(&mut params).expect("constraint failed");

        for (a, b) in params.iter().zip(original.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_orthogonal_constraint_1d_errors() {
        use scirs2_core::ndarray::Array1;
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let constraint = ParameterConstraint::Orthogonal { tolerance: 1e-6 };
        let result = constraint.apply(&mut params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2D arrays"));
    }

    #[test]
    fn test_spectral_norm_constraint_matrix() {
        use scirs2_core::ndarray::arr2;

        // Diagonal matrix with singular values {5, 1}; cap below the larger one.
        let mut params = arr2(&[[5.0, 0.0], [0.0, 1.0]]);
        let maxnorm = 2.0;
        let constraint = ParameterConstraint::SpectralNorm { maxnorm };
        constraint.apply(&mut params).expect("constraint failed");

        // Recompute the spectral norm (largest singular value) and verify ≤ cap.
        let sigma = power_iteration_spectral_norm(&params);
        assert!(
            sigma <= maxnorm + 1e-6,
            "spectral norm {sigma} exceeds cap {maxnorm}"
        );
        // It should be scaled to (approximately) the cap, not collapsed.
        assert!(
            sigma > maxnorm - 1e-3,
            "spectral norm {sigma} undershot cap"
        );
    }

    #[test]
    fn test_spectral_norm_constraint_nondiagonal() {
        use scirs2_core::ndarray::arr2;

        // A non-diagonal matrix whose true σ_max is well above the cap.
        let mut params = arr2(&[[3.0, 1.0], [1.0, 3.0], [2.0, -2.0]]);
        let maxnorm = 1.5;
        let constraint = ParameterConstraint::SpectralNorm { maxnorm };
        constraint.apply(&mut params).expect("constraint failed");

        let sigma = power_iteration_spectral_norm(&params);
        assert!(
            sigma <= maxnorm + 1e-5,
            "spectral norm {sigma} exceeds cap {maxnorm}"
        );
    }

    #[test]
    fn test_spectral_norm_constraint_under_cap_unchanged() {
        use approx::assert_abs_diff_eq;
        use scirs2_core::ndarray::arr2;

        // σ_max here is 1.0 (identity-like); cap of 10 leaves it untouched.
        let mut params = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let original = params.clone();
        let constraint = ParameterConstraint::SpectralNorm { maxnorm: 10.0 };
        constraint.apply(&mut params).expect("constraint failed");

        for (a, b) in params.iter().zip(original.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_positive_definite_constraint_indefinite() {
        use scirs2_core::ndarray::arr2;

        // Symmetric indefinite matrix: eigenvalues are {3, -1}.
        let mut params = arr2(&[[1.0, 2.0], [2.0, 1.0]]);
        let min_eig = 0.0;
        let constraint = ParameterConstraint::PositiveDefinite {
            mineigenvalue: min_eig,
        };
        constraint.apply(&mut params).expect("constraint failed");

        // Verify all eigenvalues of the result are ≥ min_eig via Jacobi.
        let (eigvals, _) = jacobi_eigen_symmetric(&params);
        for &lambda in eigvals.iter() {
            assert!(
                lambda >= min_eig - 1e-8,
                "eigenvalue {lambda} below floor {min_eig}"
            );
        }

        // And xᵀMx ≥ 0 for several probe vectors (PSD check).
        let probes = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, -3.0]];
        for p in probes.iter() {
            let mut quad = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    quad += p[i] * params[[i, j]] * p[j];
                }
            }
            assert!(quad >= -1e-8, "xᵀMx = {quad} is negative");
        }
    }

    #[test]
    fn test_positive_definite_constraint_positive_floor() {
        use scirs2_core::ndarray::arr2;

        // Same indefinite matrix, but require a strictly positive floor.
        let mut params = arr2(&[[0.0, 1.0], [1.0, 0.0]]); // eigenvalues {1, -1}
        let min_eig = 0.5;
        let constraint = ParameterConstraint::PositiveDefinite {
            mineigenvalue: min_eig,
        };
        constraint.apply(&mut params).expect("constraint failed");

        let (eigvals, _) = jacobi_eigen_symmetric(&params);
        for &lambda in eigvals.iter() {
            assert!(
                lambda >= min_eig - 1e-8,
                "eigenvalue {lambda} below floor {min_eig}"
            );
        }
    }

    #[test]
    fn test_positive_definite_constraint_already_pd_unchanged() {
        use approx::assert_abs_diff_eq;
        use scirs2_core::ndarray::arr2;

        // Already PD (eigenvalues {3, 1}); a floor of 0 must leave it ~unchanged.
        let mut params = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let original = params.clone();
        let constraint = ParameterConstraint::PositiveDefinite { mineigenvalue: 0.0 };
        constraint.apply(&mut params).expect("constraint failed");

        for (a, b) in params.iter().zip(original.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_positive_definite_constraint_non_square_errors() {
        use scirs2_core::ndarray::arr2;
        let mut params = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let constraint = ParameterConstraint::PositiveDefinite { mineigenvalue: 0.0 };
        let result = constraint.apply(&mut params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("square"));
    }
}
