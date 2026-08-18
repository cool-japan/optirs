// Self-contained dense linear-algebra helpers for parameter constraints.
//
// These avoid any external linear-algebra dependency (scirs2-linalg is not
// available here). They operate on small dense 2D matrices using only
// `scirs2_core::ndarray`, and keep the generic `A: Float` bound. `Float` is not
// `Ord`, so all comparisons go through `partial_cmp` / explicit `<`/`>`.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, Ix2};
use scirs2_core::numeric::Float;

/// Obtain an owned 2D matrix from a generic n-dimensional array known to be 2D.
///
/// The caller must have already checked `params.ndim() == 2`; the conversion is
/// done on a clone so the original array is untouched until we write back.
pub(super) fn to_matrix_2d<A, D>(params: &Array<A, D>) -> Result<Array2<A>>
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
pub(super) fn write_matrix_2d<A, D>(params: &mut Array<A, D>, matrix: &Array2<A>) -> Result<()>
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
pub(super) fn is_orthonormal<A>(matrix: &Array2<A>, tolerance: A) -> bool
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
pub(super) fn modified_gram_schmidt<A>(matrix: &Array2<A>) -> Array2<A>
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
pub(super) fn power_iteration_spectral_norm<A>(matrix: &Array2<A>) -> A
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
pub(super) fn jacobi_eigen_symmetric<A>(input: &Array2<A>) -> (Array1<A>, Array2<A>)
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
pub(super) fn project_positive_definite<A>(matrix: &Array2<A>, min_eigenvalue: A) -> Array2<A>
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
