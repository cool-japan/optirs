// Nuclear-norm (trace-norm) operations built on a self-contained truncated SVD.
//
// The nuclear norm of a matrix `M` is the sum of its singular values,
// `‖M‖_* = Σ_i σ_i(M)`. Unlike the entrywise L1 norm it is *not* separable over
// matrix entries, so neither the proximal operator nor the projection onto a
// nuclear-norm ball can be expressed as elementwise shrinkage. Both act on the
// **singular values**:
//
// * `prox_{t‖·‖_*}(M) = U · max(S − t, 0) · Vᵀ` — soft-thresholding of `S`.
// * `Π_{‖·‖_* ≤ τ}(M)  = U · max(S − θ, 0) · Vᵀ` where `θ ≥ 0` is chosen so the
//   surviving singular values sum to exactly `τ` (projection of `S` onto the
//   L1 ball of radius `τ`).
//
// The SVD used here is computed with power iteration plus deflation, using only
// `scirs2_core::ndarray` — no external linear-algebra backend and no FFI. The
// starting vectors come from a deterministic integer hash, so every call on the
// same input produces bit-identical output.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

/// Maximum number of power iterations spent on a single singular triplet.
const MAX_POWER_ITERATIONS: usize = 512;

/// Result of a truncated singular value decomposition, `M ≈ U · diag(s) · Vᵀ`.
#[derive(Debug, Clone)]
pub struct TruncatedSvd<A: Float> {
    /// Extracted singular values, in non-increasing order.
    pub singular_values: Array1<A>,
    /// Left singular vectors stored as columns; shape `(rows, k)`.
    pub u: Array2<A>,
    /// Right singular vectors stored as columns; shape `(cols, k)`.
    pub v: Array2<A>,
}

impl<A: Float> TruncatedSvd<A> {
    /// Number of extracted singular triplets.
    pub fn rank(&self) -> usize {
        self.singular_values.len()
    }

    /// Sum of the extracted singular values (the nuclear norm of the captured part).
    pub fn nuclear_norm(&self) -> A {
        self.singular_values
            .iter()
            .fold(A::zero(), |acc, &s| acc + s)
    }

    /// Reconstruct `U · diag(s) · Vᵀ` for the extracted components.
    pub fn reconstruct(&self) -> Array2<A> {
        let rows = self.u.nrows();
        let cols = self.v.nrows();
        let mut out: Array2<A> = Array2::zeros((rows, cols));
        for (k, &sigma) in self.singular_values.iter().enumerate() {
            if sigma == A::zero() {
                continue;
            }
            for r in 0..rows {
                let scaled = sigma * self.u[[r, k]];
                for c in 0..cols {
                    out[[r, c]] = out[[r, c]] + scaled * self.v[[c, k]];
                }
            }
        }
        out
    }
}

/// Deterministic 64-bit mixing function (SplitMix64 finalizer).
///
/// Used purely to build reproducible, non-degenerate starting vectors for the
/// power iteration; no randomness source is involved, so repeated calls with the
/// same inputs always yield the same decomposition.
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a deterministic, normalized starting vector of length `len`.
///
/// Entries are pseudo-random in `[-1.5, -0.5] ∪ [0.5, 1.5]`, which keeps the
/// vector well away from being orthogonal to the dominant singular direction
/// (the failure mode of a plain all-ones start) while remaining fully
/// reproducible.
fn deterministic_start_vector<A: Float>(len: usize, salt: u64) -> Array1<A> {
    let mut v: Array1<A> = Array1::zeros(len);
    if len == 0 {
        return v;
    }

    let scale = 1.0f64 / (u64::MAX as f64);
    for (i, slot) in v.iter_mut().enumerate() {
        let bits = splitmix64(salt.wrapping_mul(0x0100_0000_01B3).wrapping_add(i as u64));
        let unit = (bits >> 11) as f64 * (scale * 2048.0); // in [0, 1)
        let magnitude = 0.5 + unit;
        let signed = if bits & 1 == 0 { magnitude } else { -magnitude };
        *slot = A::from(signed).unwrap_or_else(A::one);
    }

    let mut norm_sq = A::zero();
    for &x in v.iter() {
        norm_sq = norm_sq + x * x;
    }
    let norm = norm_sq.sqrt();
    if norm > A::zero() {
        let inv = A::one() / norm;
        v.mapv_inplace(|x| x * inv);
    }
    v
}

/// `w = M v`
fn matvec<A: Float>(matrix: &Array2<A>, v: &Array1<A>) -> Array1<A> {
    let (rows, cols) = matrix.dim();
    let mut w: Array1<A> = Array1::zeros(rows);
    for r in 0..rows {
        let mut acc = A::zero();
        for c in 0..cols {
            acc = acc + matrix[[r, c]] * v[c];
        }
        w[r] = acc;
    }
    w
}

/// `a = Mᵀ w`
fn transpose_matvec<A: Float>(matrix: &Array2<A>, w: &Array1<A>) -> Array1<A> {
    let (rows, cols) = matrix.dim();
    let mut a: Array1<A> = Array1::zeros(cols);
    for c in 0..cols {
        let mut acc = A::zero();
        for r in 0..rows {
            acc = acc + matrix[[r, c]] * w[r];
        }
        a[c] = acc;
    }
    a
}

/// Euclidean norm of a vector.
fn vector_norm<A: Float>(v: &Array1<A>) -> A {
    let mut acc = A::zero();
    for &x in v.iter() {
        acc = acc + x * x;
    }
    acc.sqrt()
}

/// Frobenius norm of a matrix.
fn frobenius_norm<A: Float>(matrix: &Array2<A>) -> A {
    let mut acc = A::zero();
    for &x in matrix.iter() {
        acc = acc + x * x;
    }
    acc.sqrt()
}

/// Default stopping threshold for singular values: singular values at or below
/// this magnitude are treated as numerically zero and the deflation stops.
fn default_singular_tolerance<A: Float>(matrix: &Array2<A>) -> A {
    let scale = frobenius_norm(matrix);
    let eps = A::epsilon();
    let relative = eps.sqrt() * scale;
    let floor = eps * eps.sqrt();
    if relative > floor {
        relative
    } else {
        floor
    }
}

/// Compute a truncated SVD of `matrix` via power iteration with deflation.
///
/// Repeatedly extracts the dominant singular triplet `(σ, u, v)` of the current
/// residual by running power iteration on `RᵀR` (which converges at rate
/// `(σ_{k+1}/σ_k)²`), then deflates `R ← R − σ · u vᵀ`. Extraction stops after
/// `max_components` triplets, after `min(rows, cols)` triplets, or as soon as a
/// singular value drops to or below `tolerance`.
///
/// # Arguments
///
/// * `matrix` - Input matrix.
/// * `max_components` - Upper bound on the number of triplets to extract.
/// * `tolerance` - Singular values `≤ tolerance` terminate the extraction. Pass
///   a non-positive value to fall back to a scale-aware default.
///
/// The starting vectors are derived from a deterministic hash, so the result is
/// reproducible across runs and platforms.
pub fn truncated_svd_power_iteration<A: Float>(
    matrix: &Array2<A>,
    max_components: usize,
    tolerance: A,
) -> TruncatedSvd<A> {
    let (rows, cols) = matrix.dim();
    let max_rank = rows.min(cols).min(max_components);

    if rows == 0 || cols == 0 || max_rank == 0 {
        return TruncatedSvd {
            singular_values: Array1::zeros(0),
            u: Array2::zeros((rows, 0)),
            v: Array2::zeros((cols, 0)),
        };
    }

    let tol = if tolerance > A::zero() {
        tolerance
    } else {
        default_singular_tolerance(matrix)
    };
    // Convergence threshold on the right singular vector between iterations.
    let vector_tol = A::epsilon().sqrt();

    let mut residual = matrix.clone();
    let mut sigmas: Vec<A> = Vec::with_capacity(max_rank);
    let mut left: Vec<Array1<A>> = Vec::with_capacity(max_rank);
    let mut right: Vec<Array1<A>> = Vec::with_capacity(max_rank);

    for component in 0..max_rank {
        let mut v = deterministic_start_vector::<A>(cols, component as u64 + 1);

        for _ in 0..MAX_POWER_ITERATIONS {
            let w = matvec(&residual, &v);
            let a = transpose_matvec(&residual, &w);
            let norm = vector_norm(&a);
            if norm <= A::zero() {
                // Residual annihilates this direction: nothing left to extract.
                break;
            }
            let inv = A::one() / norm;
            let mut delta_sq = A::zero();
            for c in 0..cols {
                let next = a[c] * inv;
                let diff = next - v[c];
                delta_sq = delta_sq + diff * diff;
                v[c] = next;
            }
            if delta_sq.sqrt() <= vector_tol {
                break;
            }
        }

        let w = matvec(&residual, &v);
        let sigma = vector_norm(&w);
        if sigma <= tol {
            break;
        }

        let inv_sigma = A::one() / sigma;
        let u = w.mapv(|x| x * inv_sigma);

        // Deflate: R ← R − σ · u vᵀ
        for r in 0..rows {
            let scaled = sigma * u[r];
            for c in 0..cols {
                residual[[r, c]] = residual[[r, c]] - scaled * v[c];
            }
        }

        sigmas.push(sigma);
        left.push(u);
        right.push(v);
    }

    let k = sigmas.len();
    let mut u_mat: Array2<A> = Array2::zeros((rows, k));
    let mut v_mat: Array2<A> = Array2::zeros((cols, k));
    for (idx, (u_vec, v_vec)) in left.iter().zip(right.iter()).enumerate() {
        for r in 0..rows {
            u_mat[[r, idx]] = u_vec[r];
        }
        for c in 0..cols {
            v_mat[[c, idx]] = v_vec[c];
        }
    }

    TruncatedSvd {
        singular_values: Array1::from_vec(sigmas),
        u: u_mat,
        v: v_mat,
    }
}

/// Full-rank truncated SVD with the scale-aware default tolerance.
fn full_svd<A: Float>(matrix: &Array2<A>) -> TruncatedSvd<A> {
    let (rows, cols) = matrix.dim();
    truncated_svd_power_iteration(matrix, rows.min(cols), A::zero())
}

/// Compute the nuclear norm `‖M‖_* = Σ_i σ_i(M)` of a matrix.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::arr2;
/// use optirs_core::parameter_groups::nuclear_norm_of_matrix;
///
/// // A diagonal matrix has singular values equal to |diagonal entries|.
/// let m = arr2(&[[3.0, 0.0], [0.0, -4.0]]);
/// let nn: f64 = nuclear_norm_of_matrix(&m);
/// assert!((nn - 7.0).abs() < 1e-8);
/// ```
pub fn nuclear_norm_of_matrix<A: Float>(matrix: &Array2<A>) -> A {
    full_svd(matrix).nuclear_norm()
}

/// Proximal operator of the nuclear norm: `prox_{t‖·‖_*}(M) = U max(S − t, 0) Vᵀ`.
///
/// This is **singular-value** soft-thresholding, the operation at the heart of
/// low-rank matrix recovery (singular value thresholding / soft-impute). It is
/// deliberately *not* the same as elementwise L1 soft-thresholding of `M`.
///
/// A non-positive `threshold` returns the input unchanged.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::arr2;
/// use optirs_core::parameter_groups::nuclear_norm_prox;
///
/// let m = arr2(&[[3.0f64, 0.0], [0.0, 2.0]]);
/// let shrunk = nuclear_norm_prox(&m, 1.0);
/// assert!((shrunk[[0, 0]] - 2.0).abs() < 1e-6);
/// assert!((shrunk[[1, 1]] - 1.0).abs() < 1e-6);
/// ```
pub fn nuclear_norm_prox<A: Float>(matrix: &Array2<A>, threshold: A) -> Array2<A> {
    if threshold <= A::zero() {
        return matrix.clone();
    }
    let svd = full_svd(matrix);
    subtract_shrinkage(matrix, &svd, |sigma| {
        if sigma < threshold {
            sigma
        } else {
            threshold
        }
    })
}

/// Project a matrix onto the nuclear-norm ball `{X : ‖X‖_* ≤ max_norm}`.
///
/// Computes the singular values, projects them onto the L1 ball of radius
/// `max_norm` (which is soft-thresholding by a data-dependent `θ ≥ 0`), then
/// reconstructs. When the matrix already satisfies the constraint it is
/// returned unchanged. A non-positive `max_norm` collapses the matrix to zero.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::arr2;
/// use optirs_core::parameter_groups::{nuclear_norm_of_matrix, project_onto_nuclear_norm_ball};
///
/// let m = arr2(&[[3.0, 0.0], [0.0, 2.0]]); // nuclear norm 5
/// let projected = project_onto_nuclear_norm_ball(&m, 3.0);
/// let nn: f64 = nuclear_norm_of_matrix(&projected);
/// assert!((nn - 3.0).abs() < 1e-6);
/// ```
pub fn project_onto_nuclear_norm_ball<A: Float>(matrix: &Array2<A>, max_norm: A) -> Array2<A> {
    if max_norm <= A::zero() {
        return Array2::zeros(matrix.dim());
    }

    let svd = full_svd(matrix);
    let total = svd.nuclear_norm();
    if total <= max_norm {
        return matrix.clone();
    }

    let theta = l1_ball_threshold(&svd.singular_values, max_norm);
    if theta <= A::zero() {
        return matrix.clone();
    }

    subtract_shrinkage(
        matrix,
        &svd,
        |sigma| {
            if sigma < theta {
                sigma
            } else {
                theta
            }
        },
    )
}

/// Rebuild `M − Σ_i shrink(σ_i) · u_i v_iᵀ`.
///
/// Expressing the result as a correction to the original matrix (rather than
/// re-accumulating `U diag(σ') Vᵀ`) keeps any singular components that fell
/// below the extraction tolerance intact and avoids amplifying the small
/// orthogonality error of the deflation.
fn subtract_shrinkage<A, F>(matrix: &Array2<A>, svd: &TruncatedSvd<A>, shrink: F) -> Array2<A>
where
    A: Float,
    F: Fn(A) -> A,
{
    let (rows, cols) = matrix.dim();
    let mut out = matrix.clone();
    for (k, &sigma) in svd.singular_values.iter().enumerate() {
        let amount = shrink(sigma);
        if amount <= A::zero() {
            continue;
        }
        for r in 0..rows {
            let scaled = amount * svd.u[[r, k]];
            for c in 0..cols {
                out[[r, c]] = out[[r, c]] - scaled * svd.v[[c, k]];
            }
        }
    }
    out
}

/// Find the soft-threshold `θ ≥ 0` whose shrinkage puts `values` (non-negative)
/// exactly on the L1 sphere of radius `radius`.
///
/// Standard simplex/L1-ball projection: sort descending, walk the cumulative
/// sums, and take the largest prefix `ρ` for which `values[ρ-1] − (cum_ρ − radius)/ρ > 0`.
fn l1_ball_threshold<A: Float>(values: &Array1<A>, radius: A) -> A {
    let mut sorted: Vec<A> = values.iter().copied().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = A::zero();
    let mut theta = A::zero();
    let mut found = false;
    for (idx, &value) in sorted.iter().enumerate() {
        cumulative = cumulative + value;
        let count = A::from(idx + 1).unwrap_or_else(A::one);
        let candidate = (cumulative - radius) / count;
        if value - candidate > A::zero() {
            theta = candidate;
            found = true;
        } else {
            break;
        }
    }

    if !found || theta < A::zero() {
        A::zero()
    } else {
        theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr2, Array2};

    /// Build `U diag(s) Vᵀ` from two orthonormal bases and a spectrum.
    fn synthetic_matrix(spectrum: &[f64]) -> Array2<f64> {
        // 3x3 orthonormal U (rotation about z by 30 degrees composed with a swap).
        let c = (std::f64::consts::PI / 6.0).cos();
        let s = (std::f64::consts::PI / 6.0).sin();
        let u = arr2(&[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]);
        // 3x3 orthonormal V (rotation about x by 45 degrees).
        let c2 = (std::f64::consts::PI / 4.0).cos();
        let s2 = (std::f64::consts::PI / 4.0).sin();
        let v = arr2(&[[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]]);

        let mut m = Array2::<f64>::zeros((3, 3));
        for (k, &sigma) in spectrum.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    m[[i, j]] += sigma * u[[i, k]] * v[[j, k]];
                }
            }
        }
        m
    }

    #[test]
    fn truncated_svd_recovers_known_spectrum() {
        let m = synthetic_matrix(&[3.0, 2.0, 0.5]);
        let svd = truncated_svd_power_iteration(&m, 3, 0.0);
        assert_eq!(svd.rank(), 3);
        assert!((svd.singular_values[0] - 3.0).abs() < 1e-6);
        assert!((svd.singular_values[1] - 2.0).abs() < 1e-6);
        assert!((svd.singular_values[2] - 0.5).abs() < 1e-6);

        // Reconstruction matches the original matrix.
        let recon = svd.reconstruct();
        let mut err = 0.0;
        for (a, b) in recon.iter().zip(m.iter()) {
            err += (a - b) * (a - b);
        }
        assert!(err.sqrt() < 1e-6, "reconstruction error {}", err.sqrt());
    }

    #[test]
    fn truncated_svd_handles_rank_deficiency() {
        let m = synthetic_matrix(&[2.0, 0.0, 0.0]);
        let svd = truncated_svd_power_iteration(&m, 3, 0.0);
        assert_eq!(svd.rank(), 1);
        assert!((svd.singular_values[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn truncated_svd_is_deterministic() {
        let m = synthetic_matrix(&[3.0, 2.0, 0.5]);
        let a = truncated_svd_power_iteration(&m, 3, 0.0);
        let b = truncated_svd_power_iteration(&m, 3, 0.0);
        assert_eq!(a.singular_values, b.singular_values);
        assert_eq!(a.u, b.u);
        assert_eq!(a.v, b.v);
    }

    #[test]
    fn prox_soft_thresholds_singular_values() {
        let m = synthetic_matrix(&[3.0, 2.0, 0.5]);
        let shrunk = nuclear_norm_prox(&m, 1.0);
        let svd = truncated_svd_power_iteration(&shrunk, 3, 1e-9);
        assert!((svd.singular_values[0] - 2.0).abs() < 1e-6);
        assert!((svd.singular_values[1] - 1.0).abs() < 1e-6);
        // Third singular value fully shrunk away.
        assert!(svd.singular_values.iter().skip(2).all(|&s| s < 1e-5));
    }

    #[test]
    fn prox_differs_from_elementwise_l1() {
        let m = synthetic_matrix(&[3.0, 2.0, 0.5]);
        let shrunk = nuclear_norm_prox(&m, 1.0);
        let elementwise = m.mapv(|x: f64| {
            if x > 1.0 {
                x - 1.0
            } else if x < -1.0 {
                x + 1.0
            } else {
                0.0
            }
        });
        let mut diff = 0.0;
        for (a, b) in shrunk.iter().zip(elementwise.iter()) {
            diff += (a - b).abs();
        }
        assert!(
            diff > 1e-3,
            "nuclear prox collapsed to elementwise L1 shrinkage"
        );
    }

    #[test]
    fn projection_hits_the_ball_boundary() {
        let m = synthetic_matrix(&[3.0, 2.0, 0.5]);
        let projected = project_onto_nuclear_norm_ball(&m, 3.0);
        let nn = nuclear_norm_of_matrix(&projected);
        assert!(
            (nn - 3.0).abs() < 1e-6,
            "nuclear norm after projection {nn}"
        );
    }

    #[test]
    fn projection_is_a_no_op_inside_the_ball() {
        let m = synthetic_matrix(&[1.0, 0.5, 0.25]);
        let projected = project_onto_nuclear_norm_ball(&m, 10.0);
        assert_eq!(projected, m);
    }

    #[test]
    fn empty_and_degenerate_shapes_are_safe() {
        let empty: Array2<f64> = Array2::zeros((0, 3));
        let svd = truncated_svd_power_iteration(&empty, 3, 0.0);
        assert_eq!(svd.rank(), 0);

        let zeros: Array2<f64> = Array2::zeros((3, 3));
        assert_eq!(nuclear_norm_of_matrix(&zeros), 0.0);
        assert_eq!(nuclear_norm_prox(&zeros, 1.0), zeros);
    }

    #[test]
    fn non_square_matrices_are_supported() {
        // 2x3 matrix with singular values sqrt(eigenvalues of M Mᵀ).
        let m = arr2(&[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]);
        let nn = nuclear_norm_of_matrix(&m);
        assert!((nn - 7.0).abs() < 1e-6);

        let shrunk = nuclear_norm_prox(&m, 1.0);
        assert!((shrunk[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((shrunk[[1, 1]] - 3.0).abs() < 1e-6);
    }
}
