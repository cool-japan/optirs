//! # Sparse optimizer step (CSR/COO + lazy-state sparse SGD / Adam)
//!
//! This module implements **sparse optimizer updates** that touch only the
//! coordinates with non-zero gradients. It is the CPU reference implementation
//! of the kind of update used to train very large embedding tables, where on
//! every step only a handful of the millions of rows actually receive a
//! gradient. Touching only those coordinates (both for the parameter update and
//! for the optimizer moment state) is the entire point of "sparse / lazy"
//! optimizers.
//!
//! ## Sparse gradient representations
//!
//! * [`CooGradient`] -- a *coordinate list* for a **1-D** parameter / embedding
//!   row view. It stores the touched `indices`, their gradient `values` and the
//!   logical dimension `dim` of the dense vector it represents. It is kept in a
//!   canonical form: indices are strictly ascending (hence sorted and unique).
//! * [`CsrGradient`] -- *compressed sparse row* for a **2-D** embedding table.
//!   It stores `row_offsets`, `col_indices` and `values` together with the dense
//!   `shape`. Within every row the column indices are strictly ascending.
//!
//! The two representations convert losslessly to one another: a `CsrGradient`
//! of shape `(r, c)` corresponds to a `CooGradient` over the flattened
//! `dim = r * c` view using linear indices `row * c + col`. See
//! [`CsrGradient::to_coo`] / [`CsrGradient::from_coo`].
//!
//! ## Sparse SGD
//!
//! [`SparseSgd`] performs `params[i] -= lr * grad[i]` only for the non-zero
//! coordinates `i`, optionally with (coupled) weight decay and momentum applied
//! lazily on the touched coordinates. Untouched coordinates are left
//! bit-identical.
//!
//! ## Lazy Adam (the hard part)
//!
//! [`SparseAdam`] maintains, per coordinate, the first/second moment estimates
//! `m[i]`, `v[i]` and `last_step[i]` -- the global optimizer step at which the
//! coordinate was last updated. Two precisely-defined semantics are supported,
//! selected through [`LazyAdamMode`]:
//!
//! ### `LazyAdamMode::Lazy` (pure TensorFlow `LazyAdam`)
//!
//! When coordinate `i` is touched at global step `t` with gradient `g`, the
//! moments are updated with the **current gradient only**, using a single decay
//! factor regardless of how long the coordinate was dormant:
//!
//! ```text
//!   m[i] = beta1 * m[i] + (1 - beta1) * g
//!   v[i] = beta2 * v[i] + (1 - beta2) * g^2
//! ```
//!
//! Bias correction uses the **global** step `t`:
//!
//! ```text
//!   m_hat = m[i] / (1 - beta1^t)
//!   v_hat = v[i] / (1 - beta2^t)
//!   params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! This is cheap and matches TensorFlow's `LazyAdam`, but the moments do **not**
//! account for the EMA decay that conceptually elapsed while the coordinate was
//! dormant.
//!
//! ### `LazyAdamMode::DormancyDecay` (lazy with dormancy catch-up)
//!
//! Here we model exactly what dense Adam would have done to the moments had the
//! gradient been zero during the dormant steps. If the coordinate was last
//! updated at step `s = last_step[i]` and is now touched at step `t`, dense Adam
//! would multiply the old moments by `beta^(t - s)` (one factor of `beta` per
//! elapsed step, each carrying a zero gradient) before incorporating the new
//! gradient. Hence:
//!
//! ```text
//!   gap  = t - s                      (>= 1)
//!   m[i] = beta1^gap * m[i] + (1 - beta1) * g
//!   v[i] = beta2^gap * v[i] + (1 - beta2) * g^2
//! ```
//!
//! followed by the same global-step bias correction and update as above. For a
//! coordinate touched on **every** step `gap == 1`, so this reduces exactly to
//! the standard Adam recursion; for an intermittently-touched coordinate the
//! moments `m[i]`, `v[i]` at touch time are **identical** to those that dense
//! Adam (fed explicit zero gradients on the dormant steps) would hold. The
//! parameter trajectory still differs from dense Adam, because lazy Adam --- by
//! design --- performs **no** parameter update on the dormant steps.
//!
//! In both modes bias correction uses the global optimizer step count `t`
//! (number of `step` calls so far), never a per-coordinate visit count.

use crate::GpuOptimError;
use scirs2_core::ndarray::{Array1, Array2};

/// Sparse gradient for a **1-D** dense parameter / embedding-row view, stored as
/// a coordinate list (COO).
///
/// The representation is kept canonical: `indices` is strictly ascending, so the
/// entries are sorted and free of duplicates, and every index is `< dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct CooGradient {
    /// Touched coordinate indices, strictly ascending.
    indices: Vec<usize>,
    /// Gradient value for each touched coordinate (parallel to `indices`).
    values: Vec<f64>,
    /// Logical dimension of the dense vector this gradient applies to.
    dim: usize,
}

impl CooGradient {
    /// Build a canonical COO gradient from already sorted, unique indices.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::InvalidState`] when `indices` and `values` have
    /// mismatched lengths or when `indices` is not strictly ascending, and
    /// [`GpuOptimError::DimensionMismatch`] when any index is `>= dim`.
    pub fn new(indices: Vec<usize>, values: Vec<f64>, dim: usize) -> Result<Self, GpuOptimError> {
        let grad = Self {
            indices,
            values,
            dim,
        };
        grad.validate()?;
        Ok(grad)
    }

    /// Build a COO gradient from arbitrary (possibly unsorted, possibly
    /// duplicated) coordinate/value pairs.
    ///
    /// Duplicate indices are accumulated (their gradient values are summed),
    /// which matches the semantics of looking up the same embedding row multiple
    /// times within one mini-batch. The result is canonicalized to strictly
    /// ascending indices.
    ///
    /// # Errors
    /// Returns an error when lengths mismatch or when any index is `>= dim`.
    pub fn new_unsorted(
        indices: Vec<usize>,
        values: Vec<f64>,
        dim: usize,
    ) -> Result<Self, GpuOptimError> {
        if indices.len() != values.len() {
            return Err(GpuOptimError::InvalidState(format!(
                "COO indices/values length mismatch: {} vs {}",
                indices.len(),
                values.len()
            )));
        }
        let mut pairs: Vec<(usize, f64)> = indices.into_iter().zip(values).collect();
        for &(idx, _) in &pairs {
            if idx >= dim {
                return Err(GpuOptimError::DimensionMismatch {
                    expected: vec![dim],
                    actual: vec![idx],
                });
            }
        }
        pairs.sort_by_key(|&(idx, _)| idx);

        let mut out_indices: Vec<usize> = Vec::with_capacity(pairs.len());
        let mut out_values: Vec<f64> = Vec::with_capacity(pairs.len());
        for (idx, val) in pairs {
            if let Some(&last) = out_indices.last() {
                if last == idx {
                    if let Some(slot) = out_values.last_mut() {
                        *slot += val;
                    }
                    continue;
                }
            }
            out_indices.push(idx);
            out_values.push(val);
        }

        Ok(Self {
            indices: out_indices,
            values: out_values,
            dim,
        })
    }

    /// Create an empty (all-zero) sparse gradient of dimension `dim`.
    pub fn empty(dim: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            dim,
        }
    }

    /// Validate the structural invariants: matching lengths, strictly ascending
    /// indices and in-bounds indices.
    ///
    /// # Errors
    /// See [`CooGradient::new`].
    pub fn validate(&self) -> Result<(), GpuOptimError> {
        if self.indices.len() != self.values.len() {
            return Err(GpuOptimError::InvalidState(format!(
                "COO indices/values length mismatch: {} vs {}",
                self.indices.len(),
                self.values.len()
            )));
        }
        for pair in self.indices.windows(2) {
            if pair[0] >= pair[1] {
                return Err(GpuOptimError::InvalidState(format!(
                    "COO indices must be strictly ascending, found {} >= {}",
                    pair[0], pair[1]
                )));
            }
        }
        if let Some(&max_idx) = self.indices.last() {
            if max_idx >= self.dim {
                return Err(GpuOptimError::DimensionMismatch {
                    expected: vec![self.dim],
                    actual: vec![max_idx],
                });
            }
        }
        Ok(())
    }

    /// Logical dimension of the dense vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored (non-zero) coordinates.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// True when no coordinate is touched.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Touched coordinate indices (strictly ascending).
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Gradient values (parallel to [`CooGradient::indices`]).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Iterate over `(index, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
}

/// Sparse gradient for a **2-D** embedding table, stored in compressed sparse
/// row (CSR) form.
///
/// Invariants (checked by [`CsrGradient::validate`]):
/// * `row_offsets.len() == shape.0 + 1`, `row_offsets[0] == 0`, monotonically
///   non-decreasing, and `row_offsets[shape.0] == col_indices.len()`.
/// * `col_indices.len() == values.len()`.
/// * within every row the column indices are strictly ascending and `< shape.1`.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrGradient {
    /// Row pointer array of length `shape.0 + 1`.
    row_offsets: Vec<usize>,
    /// Column index of every stored value.
    col_indices: Vec<usize>,
    /// Stored gradient values (parallel to `col_indices`).
    values: Vec<f64>,
    /// Dense shape `(rows, cols)` of the embedding table.
    shape: (usize, usize),
}

impl CsrGradient {
    /// Build a CSR gradient and validate it.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::InvalidState`] / [`GpuOptimError::DimensionMismatch`]
    /// when any structural invariant is violated.
    pub fn new(
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> Result<Self, GpuOptimError> {
        let grad = Self {
            row_offsets,
            col_indices,
            values,
            shape,
        };
        grad.validate()?;
        Ok(grad)
    }

    /// Create an empty (all-zero) CSR gradient of the given shape.
    pub fn empty(shape: (usize, usize)) -> Self {
        Self {
            row_offsets: vec![0; shape.0 + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    /// Validate all CSR structural invariants.
    ///
    /// # Errors
    /// See the type-level documentation for the checked invariants.
    pub fn validate(&self) -> Result<(), GpuOptimError> {
        let (rows, cols) = self.shape;
        if self.row_offsets.len() != rows + 1 {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![rows + 1],
                actual: vec![self.row_offsets.len()],
            });
        }
        if self.col_indices.len() != self.values.len() {
            return Err(GpuOptimError::InvalidState(format!(
                "CSR col_indices/values length mismatch: {} vs {}",
                self.col_indices.len(),
                self.values.len()
            )));
        }
        if self.row_offsets[0] != 0 {
            return Err(GpuOptimError::InvalidState(format!(
                "CSR row_offsets must start at 0, found {}",
                self.row_offsets[0]
            )));
        }
        for pair in self.row_offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(GpuOptimError::InvalidState(format!(
                    "CSR row_offsets must be non-decreasing, found {} > {}",
                    pair[0], pair[1]
                )));
            }
        }
        if self.row_offsets[rows] != self.col_indices.len() {
            return Err(GpuOptimError::InvalidState(format!(
                "CSR final row_offset {} must equal nnz {}",
                self.row_offsets[rows],
                self.col_indices.len()
            )));
        }
        for r in 0..rows {
            let start = self.row_offsets[r];
            let end = self.row_offsets[r + 1];
            let row_cols = &self.col_indices[start..end];
            for &c in row_cols {
                if c >= cols {
                    return Err(GpuOptimError::DimensionMismatch {
                        expected: vec![cols],
                        actual: vec![c],
                    });
                }
            }
            for pair in row_cols.windows(2) {
                if pair[0] >= pair[1] {
                    return Err(GpuOptimError::InvalidState(format!(
                        "CSR column indices within row {} must be strictly ascending, found {} >= {}",
                        r, pair[0], pair[1]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Dense shape `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Number of rows in the dense table.
    pub fn rows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns in the dense table.
    pub fn cols(&self) -> usize {
        self.shape.1
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Column indices and values stored for row `r`.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] when `r` is out of range.
    pub fn row(&self, r: usize) -> Result<(&[usize], &[f64]), GpuOptimError> {
        if r >= self.shape.0 {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.shape.0],
                actual: vec![r],
            });
        }
        let start = self.row_offsets[r];
        let end = self.row_offsets[r + 1];
        Ok((&self.col_indices[start..end], &self.values[start..end]))
    }

    /// Iterate over `(row, col, value)` triplets in row-major (and within a row,
    /// column-ascending) order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.shape.0).flat_map(move |r| {
            let start = self.row_offsets[r];
            let end = self.row_offsets[r + 1];
            (start..end).map(move |k| (r, self.col_indices[k], self.values[k]))
        })
    }

    /// Convert to a flattened 1-D [`CooGradient`] using linear indices
    /// `row * cols + col` over a `rows * cols` view. The resulting indices are
    /// globally ascending, so the COO gradient is already canonical.
    pub fn to_coo(&self) -> CooGradient {
        let cols = self.shape.1;
        let mut indices = Vec::with_capacity(self.col_indices.len());
        let mut values = Vec::with_capacity(self.values.len());
        for (r, c, val) in self.iter() {
            indices.push(r * cols + c);
            values.push(val);
        }
        CooGradient {
            indices,
            values,
            dim: self.shape.0 * cols,
        }
    }

    /// Build a CSR gradient from a flattened 1-D [`CooGradient`] of dimension
    /// `shape.0 * shape.1`, decoding each linear index into `(row, col)`.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] when `coo.dim()` does not
    /// equal `shape.0 * shape.1`, or [`GpuOptimError::InvalidState`] when the
    /// table would have zero columns while carrying entries.
    pub fn from_coo(coo: &CooGradient, shape: (usize, usize)) -> Result<Self, GpuOptimError> {
        let (rows, cols) = shape;
        let expected_dim = rows * cols;
        if coo.dim() != expected_dim {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![expected_dim],
                actual: vec![coo.dim()],
            });
        }
        if cols == 0 {
            if coo.nnz() == 0 {
                return Ok(Self::empty(shape));
            }
            return Err(GpuOptimError::InvalidState(
                "cannot decode a CSR gradient with zero columns but non-zero entries".to_string(),
            ));
        }

        let mut row_offsets = vec![0usize; rows + 1];
        let mut col_indices = Vec::with_capacity(coo.nnz());
        let mut values = Vec::with_capacity(coo.nnz());
        // Count entries per row first.
        for &lin in coo.indices() {
            let r = lin / cols;
            row_offsets[r + 1] += 1;
        }
        for r in 0..rows {
            row_offsets[r + 1] += row_offsets[r];
        }
        // The COO indices are ascending, so decoded (row, col) pairs are emitted
        // in row-major, column-ascending order: a simple sequential fill is
        // sufficient and preserves the CSR invariants.
        for (lin, val) in coo.iter() {
            let c = lin % cols;
            col_indices.push(c);
            values.push(val);
        }

        let grad = Self {
            row_offsets,
            col_indices,
            values,
            shape,
        };
        grad.validate()?;
        Ok(grad)
    }
}

impl From<&CsrGradient> for CooGradient {
    fn from(csr: &CsrGradient) -> Self {
        csr.to_coo()
    }
}

/// Semantics used by [`SparseAdam`] for catching up the moment estimates of a
/// coordinate that has been dormant. See the module-level documentation for the
/// exact update equations of each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyAdamMode {
    /// Pure TensorFlow `LazyAdam`: moments decay by a single factor of `beta`
    /// regardless of dormancy. Cheapest; does not model the EMA decay that
    /// elapsed while the coordinate was untouched.
    Lazy,
    /// Lazy Adam with dormancy catch-up: moments are decayed by `beta^(t - s)`
    /// (where `s` is the last touched step) before the current gradient is
    /// incorporated, exactly reproducing the moments that dense Adam --- fed
    /// zero gradients on the dormant steps --- would hold.
    DormancyDecay,
}

/// Configuration for [`SparseAdam`] (and its 2-D table variant
/// [`SparseAdamTable`]).
#[derive(Debug, Clone, Copy)]
pub struct SparseAdamConfig {
    /// Learning rate.
    pub lr: f64,
    /// First moment decay `beta1`.
    pub beta1: f64,
    /// Second moment decay `beta2`.
    pub beta2: f64,
    /// Numerical stabilizer `epsilon`.
    pub epsilon: f64,
    /// Coupled (L2) weight decay applied on touched coordinates only.
    pub weight_decay: f64,
    /// Dormancy catch-up semantics.
    pub mode: LazyAdamMode,
}

impl Default for SparseAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            mode: LazyAdamMode::Lazy,
        }
    }
}

/// Apply one lazy-Adam update to a single coordinate, in place.
///
/// `t` is the *global* optimizer step (>= 1). `last_step` is the global step at
/// which this coordinate was previously updated (0 if never). Both moment
/// references and `last_step` are advanced; `param` receives the bias-corrected
/// update.
fn lazy_adam_coordinate(
    param: &mut f64,
    m: &mut f64,
    v: &mut f64,
    last_step: &mut usize,
    grad: f64,
    t: usize,
    cfg: &SparseAdamConfig,
) {
    // Coupled (L2) weight decay: fold into the effective gradient.
    let g = grad + cfg.weight_decay * *param;

    let (decay1, decay2) = match cfg.mode {
        LazyAdamMode::Lazy => (cfg.beta1, cfg.beta2),
        LazyAdamMode::DormancyDecay => {
            let gap = (t - *last_step) as i32;
            (cfg.beta1.powi(gap), cfg.beta2.powi(gap))
        }
    };

    *m = decay1 * *m + (1.0 - cfg.beta1) * g;
    *v = decay2 * *v + (1.0 - cfg.beta2) * g * g;

    let bias1 = 1.0 - cfg.beta1.powi(t as i32);
    let bias2 = 1.0 - cfg.beta2.powi(t as i32);
    let m_hat = *m / bias1;
    let v_hat = *v / bias2;

    *param -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.epsilon);
    *last_step = t;
}

/// Sparse / lazy Adam optimizer over a **1-D** dense parameter vector.
///
/// Per-coordinate moment state (`m`, `v`) and `last_step` are stored densely
/// (Adam always needs one moment pair per parameter) but are only *read and
/// written* for coordinates that carry a gradient on a given step, which is what
/// makes the step cost proportional to the number of non-zero coordinates.
#[derive(Debug, Clone)]
pub struct SparseAdam {
    config: SparseAdamConfig,
    m: Vec<f64>,
    v: Vec<f64>,
    last_step: Vec<usize>,
    dim: usize,
    global_step: usize,
}

impl SparseAdam {
    /// Create a new optimizer. State is allocated lazily on the first
    /// [`SparseAdam::step`] from the length of the parameter vector.
    pub fn new(config: SparseAdamConfig) -> Self {
        Self {
            config,
            m: Vec::new(),
            v: Vec::new(),
            last_step: Vec::new(),
            dim: 0,
            global_step: 0,
        }
    }

    /// The number of optimizer steps performed so far (the global `t`).
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Immutable view of the configuration.
    pub fn config(&self) -> &SparseAdamConfig {
        &self.config
    }

    fn ensure_state(&mut self, dim: usize) -> Result<(), GpuOptimError> {
        if self.global_step == 0 {
            self.dim = dim;
            self.m = vec![0.0; dim];
            self.v = vec![0.0; dim];
            self.last_step = vec![0usize; dim];
        } else if self.dim != dim {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.dim],
                actual: vec![dim],
            });
        }
        Ok(())
    }

    /// Perform one sparse Adam update in place, touching only the coordinates
    /// present in `grad`.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] when the parameter length and
    /// the gradient dimension disagree (or differ from a previous step), and
    /// propagates validation errors from a malformed gradient.
    pub fn step(
        &mut self,
        params: &mut Array1<f64>,
        grad: &CooGradient,
    ) -> Result<(), GpuOptimError> {
        grad.validate()?;
        if grad.dim() != params.len() {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![params.len()],
                actual: vec![grad.dim()],
            });
        }
        self.ensure_state(params.len())?;
        self.global_step += 1;
        let t = self.global_step;

        for (idx, g) in grad.iter() {
            let mut p = params[idx];
            lazy_adam_coordinate(
                &mut p,
                &mut self.m[idx],
                &mut self.v[idx],
                &mut self.last_step[idx],
                g,
                t,
                &self.config,
            );
            params[idx] = p;
        }
        Ok(())
    }
}

/// Sparse / lazy Adam optimizer over a **2-D** embedding table.
///
/// Mirrors [`SparseAdam`] but operates on an [`Array2`] with a [`CsrGradient`].
/// Moment state is stored flat, indexed by `row * cols + col`.
#[derive(Debug, Clone)]
pub struct SparseAdamTable {
    config: SparseAdamConfig,
    m: Vec<f64>,
    v: Vec<f64>,
    last_step: Vec<usize>,
    shape: (usize, usize),
    global_step: usize,
}

impl SparseAdamTable {
    /// Create a new table optimizer. State is allocated lazily on the first
    /// [`SparseAdamTable::step`].
    pub fn new(config: SparseAdamConfig) -> Self {
        Self {
            config,
            m: Vec::new(),
            v: Vec::new(),
            last_step: Vec::new(),
            shape: (0, 0),
            global_step: 0,
        }
    }

    /// The number of optimizer steps performed so far.
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    fn ensure_state(&mut self, shape: (usize, usize)) -> Result<(), GpuOptimError> {
        if self.global_step == 0 {
            self.shape = shape;
            let n = shape.0 * shape.1;
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
            self.last_step = vec![0usize; n];
        } else if self.shape != shape {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.shape.0, self.shape.1],
                actual: vec![shape.0, shape.1],
            });
        }
        Ok(())
    }

    /// Perform one sparse Adam update on the embedding table in place.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] on any shape disagreement and
    /// propagates gradient validation errors.
    pub fn step(
        &mut self,
        params: &mut Array2<f64>,
        grad: &CsrGradient,
    ) -> Result<(), GpuOptimError> {
        grad.validate()?;
        let shape = params.dim();
        if grad.shape() != shape {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![shape.0, shape.1],
                actual: vec![grad.shape().0, grad.shape().1],
            });
        }
        self.ensure_state(shape)?;
        self.global_step += 1;
        let t = self.global_step;
        let cols = shape.1;

        for (r, c, g) in grad.iter() {
            let flat = r * cols + c;
            let mut p = params[[r, c]];
            lazy_adam_coordinate(
                &mut p,
                &mut self.m[flat],
                &mut self.v[flat],
                &mut self.last_step[flat],
                g,
                t,
                &self.config,
            );
            params[[r, c]] = p;
        }
        Ok(())
    }
}

/// Configuration for [`SparseSgd`] (and its 2-D table variant
/// [`SparseSgdTable`]).
#[derive(Debug, Clone, Copy)]
pub struct SparseSgdConfig {
    /// Learning rate.
    pub lr: f64,
    /// Coupled (L2) weight decay applied on touched coordinates only.
    pub weight_decay: f64,
    /// Momentum factor; `0.0` disables the momentum buffer.
    pub momentum: f64,
    /// Whether to use Nesterov momentum (requires `momentum > 0`).
    pub nesterov: bool,
}

impl Default for SparseSgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            weight_decay: 0.0,
            momentum: 0.0,
            nesterov: false,
        }
    }
}

/// Apply one sparse SGD update to a single coordinate, in place.
///
/// Momentum is *lazy*: the per-coordinate buffer is only decayed/advanced when
/// the coordinate is touched (no dormancy catch-up), matching the common sparse
/// SGD behaviour for embeddings.
fn sparse_sgd_coordinate(param: &mut f64, buf: &mut f64, grad: f64, cfg: &SparseSgdConfig) {
    let mut d = grad + cfg.weight_decay * *param;
    if cfg.momentum > 0.0 {
        *buf = cfg.momentum * *buf + d;
        if cfg.nesterov {
            d += cfg.momentum * *buf;
        } else {
            d = *buf;
        }
    }
    *param -= cfg.lr * d;
}

/// Sparse SGD optimizer over a **1-D** dense parameter vector.
///
/// With `momentum == 0` and `weight_decay == 0` this is the canonical sparse
/// update `params[i] -= lr * grad[i]`, touching only the non-zero coordinates
/// and leaving every other coordinate bit-identical.
#[derive(Debug, Clone)]
pub struct SparseSgd {
    config: SparseSgdConfig,
    momentum_buf: Vec<f64>,
    dim: usize,
    initialized: bool,
}

impl SparseSgd {
    /// Create a new optimizer. State (if momentum is used) is allocated lazily.
    pub fn new(config: SparseSgdConfig) -> Self {
        Self {
            config,
            momentum_buf: Vec::new(),
            dim: 0,
            initialized: false,
        }
    }

    /// Immutable view of the configuration.
    pub fn config(&self) -> &SparseSgdConfig {
        &self.config
    }

    fn ensure_state(&mut self, dim: usize) -> Result<(), GpuOptimError> {
        if !self.initialized {
            self.dim = dim;
            if self.config.momentum > 0.0 {
                self.momentum_buf = vec![0.0; dim];
            }
            self.initialized = true;
        } else if self.dim != dim {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.dim],
                actual: vec![dim],
            });
        }
        Ok(())
    }

    /// Perform one sparse SGD update in place, touching only the coordinates
    /// present in `grad`.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] on dimension disagreement and
    /// propagates gradient validation errors.
    pub fn step(
        &mut self,
        params: &mut Array1<f64>,
        grad: &CooGradient,
    ) -> Result<(), GpuOptimError> {
        grad.validate()?;
        if grad.dim() != params.len() {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![params.len()],
                actual: vec![grad.dim()],
            });
        }
        self.ensure_state(params.len())?;

        let use_momentum = self.config.momentum > 0.0;
        for (idx, g) in grad.iter() {
            let mut p = params[idx];
            if use_momentum {
                let mut buf = self.momentum_buf[idx];
                sparse_sgd_coordinate(&mut p, &mut buf, g, &self.config);
                self.momentum_buf[idx] = buf;
            } else {
                let mut scratch = 0.0;
                sparse_sgd_coordinate(&mut p, &mut scratch, g, &self.config);
            }
            params[idx] = p;
        }
        Ok(())
    }
}

/// Sparse SGD optimizer over a **2-D** embedding table (CSR gradients).
#[derive(Debug, Clone)]
pub struct SparseSgdTable {
    config: SparseSgdConfig,
    momentum_buf: Vec<f64>,
    shape: (usize, usize),
    initialized: bool,
}

impl SparseSgdTable {
    /// Create a new table optimizer.
    pub fn new(config: SparseSgdConfig) -> Self {
        Self {
            config,
            momentum_buf: Vec::new(),
            shape: (0, 0),
            initialized: false,
        }
    }

    fn ensure_state(&mut self, shape: (usize, usize)) -> Result<(), GpuOptimError> {
        if !self.initialized {
            self.shape = shape;
            if self.config.momentum > 0.0 {
                self.momentum_buf = vec![0.0; shape.0 * shape.1];
            }
            self.initialized = true;
        } else if self.shape != shape {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![self.shape.0, self.shape.1],
                actual: vec![shape.0, shape.1],
            });
        }
        Ok(())
    }

    /// Perform one sparse SGD update on the embedding table in place.
    ///
    /// # Errors
    /// Returns [`GpuOptimError::DimensionMismatch`] on shape disagreement and
    /// propagates gradient validation errors.
    pub fn step(
        &mut self,
        params: &mut Array2<f64>,
        grad: &CsrGradient,
    ) -> Result<(), GpuOptimError> {
        grad.validate()?;
        let shape = params.dim();
        if grad.shape() != shape {
            return Err(GpuOptimError::DimensionMismatch {
                expected: vec![shape.0, shape.1],
                actual: vec![grad.shape().0, grad.shape().1],
            });
        }
        self.ensure_state(shape)?;

        let use_momentum = self.config.momentum > 0.0;
        let cols = shape.1;
        for (r, c, g) in grad.iter() {
            let mut p = params[[r, c]];
            if use_momentum {
                let flat = r * cols + c;
                let mut buf = self.momentum_buf[flat];
                sparse_sgd_coordinate(&mut p, &mut buf, g, &self.config);
                self.momentum_buf[flat] = buf;
            } else {
                let mut scratch = 0.0;
                sparse_sgd_coordinate(&mut p, &mut scratch, g, &self.config);
            }
            params[[r, c]] = p;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{Array1, Array2};

    /// Dense reference Adam over a small vector; updates every coordinate on
    /// every call (zero gradients included), which is the textbook recursion.
    struct DenseAdam {
        m: Vec<f64>,
        v: Vec<f64>,
        t: usize,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    }

    impl DenseAdam {
        fn new(dim: usize, cfg: &SparseAdamConfig) -> Self {
            Self {
                m: vec![0.0; dim],
                v: vec![0.0; dim],
                t: 0,
                lr: cfg.lr,
                beta1: cfg.beta1,
                beta2: cfg.beta2,
                epsilon: cfg.epsilon,
            }
        }

        fn step(&mut self, params: &mut [f64], grad: &[f64]) {
            self.t += 1;
            let bias1 = 1.0 - self.beta1.powi(self.t as i32);
            let bias2 = 1.0 - self.beta2.powi(self.t as i32);
            for i in 0..params.len() {
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
                let m_hat = self.m[i] / bias1;
                let v_hat = self.v[i] / bias2;
                params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }

    // ----- COO construction / validation -----

    #[test]
    fn test_sparse_coo_new_valid() {
        let g = CooGradient::new(vec![0, 2, 5], vec![1.0, -2.0, 3.0], 8).expect("valid coo");
        assert_eq!(g.nnz(), 3);
        assert_eq!(g.dim(), 8);
        let collected: Vec<(usize, f64)> = g.iter().collect();
        assert_eq!(collected, vec![(0, 1.0), (2, -2.0), (5, 3.0)]);
    }

    #[test]
    fn test_sparse_coo_out_of_bounds_errs() {
        let err = CooGradient::new(vec![0, 9], vec![1.0, 2.0], 8);
        assert!(matches!(err, Err(GpuOptimError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_coo_length_mismatch_errs() {
        let err = CooGradient::new(vec![0, 1, 2], vec![1.0, 2.0], 8);
        assert!(matches!(err, Err(GpuOptimError::InvalidState(_))));
    }

    #[test]
    fn test_sparse_coo_unsorted_errs() {
        let err = CooGradient::new(vec![2, 1], vec![1.0, 2.0], 8);
        assert!(matches!(err, Err(GpuOptimError::InvalidState(_))));
    }

    #[test]
    fn test_sparse_coo_new_unsorted_accumulates_duplicates() {
        let g = CooGradient::new_unsorted(vec![3, 1, 3], vec![1.0, 5.0, 2.0], 8)
            .expect("canonicalizes");
        assert_eq!(g.indices(), &[1, 3]);
        assert_eq!(g.values(), &[5.0, 3.0]);
    }

    // ----- CSR construction / validation -----

    #[test]
    fn test_sparse_csr_new_valid() {
        // shape (3, 4); row0: cols {0,2}; row1: {}; row2: {1,3}
        let csr = CsrGradient::new(
            vec![0, 2, 2, 4],
            vec![0, 2, 1, 3],
            vec![1.0, 2.0, 3.0, 4.0],
            (3, 4),
        )
        .expect("valid csr");
        assert_eq!(csr.nnz(), 4);
        let (cols, vals) = csr.row(2).expect("row 2");
        assert_eq!(cols, &[1, 3]);
        assert_eq!(vals, &[3.0, 4.0]);
    }

    #[test]
    fn test_sparse_csr_bad_offsets_len_errs() {
        let err = CsrGradient::new(vec![0, 2], vec![0, 1], vec![1.0, 2.0], (3, 4));
        assert!(matches!(err, Err(GpuOptimError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_csr_col_out_of_bounds_errs() {
        let err = CsrGradient::new(vec![0, 1, 1, 1], vec![9], vec![1.0], (3, 4));
        assert!(matches!(err, Err(GpuOptimError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_csr_unsorted_cols_errs() {
        let err = CsrGradient::new(vec![0, 2, 2, 2], vec![2, 1], vec![1.0, 2.0], (3, 4));
        assert!(matches!(err, Err(GpuOptimError::InvalidState(_))));
    }

    // ----- COO <-> CSR round trip -----

    #[test]
    fn test_sparse_coo_csr_round_trip() {
        let csr = CsrGradient::new(
            vec![0, 2, 2, 4],
            vec![0, 2, 1, 3],
            vec![1.0, 2.0, 3.0, 4.0],
            (3, 4),
        )
        .expect("valid csr");

        // CSR -> COO -> CSR preserves the matrix.
        let coo = csr.to_coo();
        assert_eq!(coo.dim(), 12);
        assert_eq!(coo.indices(), &[0, 2, 9, 11]); // 0,2, 2*4+1=9, 2*4+3=11
        let csr2 = CsrGradient::from_coo(&coo, (3, 4)).expect("rebuild csr");
        assert_eq!(csr, csr2);

        // COO -> CSR -> COO preserves the vector too.
        let coo2 = csr2.to_coo();
        assert_eq!(coo, coo2);

        // The infallible From impl agrees.
        let coo3: CooGradient = (&csr).into();
        assert_eq!(coo, coo3);
    }

    #[test]
    fn test_sparse_csr_from_coo_dim_mismatch_errs() {
        let coo = CooGradient::new(vec![0, 5], vec![1.0, 2.0], 10).expect("coo");
        let err = CsrGradient::from_coo(&coo, (3, 4)); // 3*4 = 12 != 10
        assert!(matches!(err, Err(GpuOptimError::DimensionMismatch { .. })));
    }

    // ----- Sparse SGD touches only nonzero coordinates -----

    #[test]
    fn test_sparse_sgd_touches_only_nonzero() {
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let original = params.clone();
        let grad = CooGradient::new(vec![1, 3], vec![0.5, -0.5], 5).expect("coo");

        let cfg = SparseSgdConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut opt = SparseSgd::new(cfg);
        opt.step(&mut params, &grad).expect("sgd step");

        // Untouched coordinates are bit-identical.
        assert_eq!(params[0].to_bits(), original[0].to_bits());
        assert_eq!(params[2].to_bits(), original[2].to_bits());
        assert_eq!(params[4].to_bits(), original[4].to_bits());

        // Touched coordinates moved by exactly -lr * g.
        assert_relative_eq!(params[1], 2.0 - 0.1 * 0.5, epsilon = 1e-12);
        assert_relative_eq!(params[3], 4.0 - 0.1 * -0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_sparse_sgd_momentum_lazy() {
        // With momentum, the buffer only advances on touched coordinates.
        let mut params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let cfg = SparseSgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        };
        let mut opt = SparseSgd::new(cfg);

        // Touch coord 1 twice with the same gradient.
        let g = CooGradient::new(vec![1], vec![1.0], 3).expect("coo");
        opt.step(&mut params, &g).expect("step 1");
        // buf = 0.9*0 + 1 = 1; p -= 0.1*1 => -0.1
        assert_relative_eq!(params[1], -0.1, epsilon = 1e-12);
        opt.step(&mut params, &g).expect("step 2");
        // buf = 0.9*1 + 1 = 1.9; p -= 0.1*1.9 => -0.1 - 0.19 = -0.29
        assert_relative_eq!(params[1], -0.29, epsilon = 1e-12);
        // Untouched coords stay exactly zero.
        assert_eq!(params[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(params[2].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn test_sparse_sgd_table_csr() {
        let mut params = Array2::<f64>::zeros((2, 3));
        params[[0, 0]] = 1.0;
        params[[1, 2]] = 2.0;
        let original = params.clone();

        // Touch only (0,1) and (1,0).
        let csr =
            CsrGradient::new(vec![0, 1, 2], vec![1, 0], vec![0.5, -1.0], (2, 3)).expect("csr");
        let cfg = SparseSgdConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut opt = SparseSgdTable::new(cfg);
        opt.step(&mut params, &csr).expect("table step");

        assert_relative_eq!(params[[0, 1]], -0.05, epsilon = 1e-12);
        assert_relative_eq!(params[[1, 0]], 0.1, epsilon = 1e-12);
        // Everything else bit-identical.
        assert_eq!(params[[0, 0]].to_bits(), original[[0, 0]].to_bits());
        assert_eq!(params[[1, 2]].to_bits(), original[[1, 2]].to_bits());
        assert_eq!(params[[0, 2]].to_bits(), original[[0, 2]].to_bits());
    }

    // ----- Lazy Adam: every-step coordinate matches dense Adam -----

    fn assert_every_step_matches_dense(mode: LazyAdamMode) {
        let cfg = SparseAdamConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            mode,
        };
        let dim = 4;
        let touched = 2usize;
        let mut sparse = SparseAdam::new(cfg);
        let mut sparse_params = Array1::from_vec(vec![0.5, -0.3, 0.7, 0.1]);

        let mut dense = DenseAdam::new(1, &cfg);
        let mut dense_param = vec![sparse_params[touched]];

        let grads = [0.4_f64, -0.2, 0.05, 0.0, 0.33, -0.7, 0.15];
        for (k, &g) in grads.iter().enumerate() {
            // Sparse: touch only `touched` (but advance global step each call).
            let coo = CooGradient::new(vec![touched], vec![g], dim).expect("coo");
            sparse.step(&mut sparse_params, &coo).expect("sparse step");

            // Dense scalar Adam on the same coordinate / gradient.
            dense.step(&mut dense_param, &[g]);

            assert_relative_eq!(
                sparse_params[touched],
                dense_param[0],
                epsilon = 1e-12,
                max_relative = 1e-10
            );
            assert_eq!(sparse.global_step(), k + 1);
        }
    }

    #[test]
    fn test_sparse_adam_every_step_matches_dense_lazy() {
        assert_every_step_matches_dense(LazyAdamMode::Lazy);
    }

    #[test]
    fn test_sparse_adam_every_step_matches_dense_dormancy() {
        assert_every_step_matches_dense(LazyAdamMode::DormancyDecay);
    }

    // ----- Lazy Adam: bias correction at small step counts -----

    #[test]
    fn test_sparse_adam_bias_correction_first_step() {
        // At t = 1, m_hat / sqrt(v_hat) = g / |g| = sign(g), so the parameter
        // moves by approximately -lr * sign(g).
        let cfg = SparseAdamConfig {
            lr: 0.1,
            epsilon: 1e-8,
            ..Default::default()
        };
        let mut opt = SparseAdam::new(cfg);
        let mut params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let grad = CooGradient::new(vec![1], vec![2.0], 3).expect("coo");
        opt.step(&mut params, &grad).expect("step");
        assert_relative_eq!(params[1], -0.1, epsilon = 1e-6);

        // A second, negative gradient at t = 2 also yields ~ -lr * sign(g).
        let mut params2 = Array1::from_vec(vec![0.0]);
        let mut opt2 = SparseAdam::new(SparseAdamConfig {
            lr: 0.1,
            epsilon: 1e-8,
            ..Default::default()
        });
        let g_pos = CooGradient::new(vec![0], vec![1.0], 1).expect("coo");
        let g_neg = CooGradient::new(vec![0], vec![-1.0], 1).expect("coo");
        opt2.step(&mut params2, &g_pos).expect("step");
        let after_first = params2[0];
        opt2.step(&mut params2, &g_neg).expect("step");
        // Second step pushes back upward (gradient sign flipped).
        assert!(params2[0] > after_first);
    }

    // ----- Lazy Adam: intermittent coordinate, DormancyDecay semantics -----

    #[test]
    fn test_sparse_adam_intermittent_dormancy_decay() {
        let cfg = SparseAdamConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            mode: LazyAdamMode::DormancyDecay,
        };
        let dim = 2;
        let j = 1usize;
        let mut opt = SparseAdam::new(cfg);
        let mut params = Array1::from_vec(vec![0.0, 1.0]);

        // Dense reference fed explicit zero gradients on the dormant step.
        let mut dense = DenseAdam::new(dim, &cfg);
        let mut dense_params = vec![0.0, 1.0];

        // t=1: touch coord j with g=0.5
        opt.step(
            &mut params,
            &CooGradient::new(vec![j], vec![0.5], dim).expect("coo"),
        )
        .expect("t1");
        dense.step(&mut dense_params, &[0.0, 0.5]);

        // t=2: touch coord 0 only (j dormant)
        opt.step(
            &mut params,
            &CooGradient::new(vec![0], vec![0.3], dim).expect("coo"),
        )
        .expect("t2");
        dense.step(&mut dense_params, &[0.3, 0.0]);

        // t=3: touch coord j again with g=-0.2 (gap = 3 - 1 = 2)
        opt.step(
            &mut params,
            &CooGradient::new(vec![j], vec![-0.2], dim).expect("coo"),
        )
        .expect("t3");
        dense.step(&mut dense_params, &[0.0, -0.2]);

        // Defining property of DormancyDecay: moments at coord j match dense Adam
        // fed zero gradients during dormancy.
        assert_relative_eq!(opt.m[j], dense.m[j], epsilon = 1e-12);
        assert_relative_eq!(opt.v[j], dense.v[j], epsilon = 1e-12);

        // Hand-computed lazy parameter trajectory (independent reference that
        // skips the dormant-step parameter update).
        let (b1, b2, lr, eps) = (cfg.beta1, cfg.beta2, cfg.lr, cfg.epsilon);
        let mut p = 1.0_f64;
        let mut m = 0.0_f64;
        let mut vv = 0.0_f64;
        // t = 1, gap = 1
        m = b1.powi(1) * m + (1.0 - b1) * 0.5;
        vv = b2.powi(1) * vv + (1.0 - b2) * 0.5 * 0.5;
        p -= lr * (m / (1.0 - b1.powi(1))) / ((vv / (1.0 - b2.powi(1))).sqrt() + eps);
        // t = 3, gap = 2 (no update happened at t = 2 for coord j)
        m = b1.powi(2) * m + (1.0 - b1) * -0.2;
        vv = b2.powi(2) * vv + (1.0 - b2) * 0.2 * 0.2;
        p -= lr * (m / (1.0 - b1.powi(3))) / ((vv / (1.0 - b2.powi(3))).sqrt() + eps);

        assert_relative_eq!(params[j], p, epsilon = 1e-12);

        // The lazy parameter must differ from dense (dense nudged j at t = 2).
        assert!((params[j] - dense_params[j]).abs() > 1e-9);
    }

    // ----- Lazy Adam: intermittent coordinate, pure Lazy semantics -----

    #[test]
    fn test_sparse_adam_intermittent_pure_lazy() {
        let cfg = SparseAdamConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            mode: LazyAdamMode::Lazy,
        };
        let dim = 2;
        let j = 1usize;
        let mut opt = SparseAdam::new(cfg);
        let mut params = Array1::from_vec(vec![0.0, 1.0]);

        // t=1 touch j (g=0.5), t=2 touch coord 0, t=3 touch j (g=-0.2)
        opt.step(
            &mut params,
            &CooGradient::new(vec![j], vec![0.5], dim).expect("coo"),
        )
        .expect("t1");
        opt.step(
            &mut params,
            &CooGradient::new(vec![0], vec![0.3], dim).expect("coo"),
        )
        .expect("t2");
        opt.step(
            &mut params,
            &CooGradient::new(vec![j], vec![-0.2], dim).expect("coo"),
        )
        .expect("t3");

        // Pure-lazy reference: single beta decay on the touch at t=3, but bias
        // correction still uses the global step t (1, then 3).
        let (b1, b2, lr, eps) = (cfg.beta1, cfg.beta2, cfg.lr, cfg.epsilon);
        let mut p = 1.0_f64;
        let mut m = 0.0_f64;
        let mut vv = 0.0_f64;
        // visit at t = 1
        m = b1 * m + (1.0 - b1) * 0.5;
        vv = b2 * vv + (1.0 - b2) * 0.5 * 0.5;
        p -= lr * (m / (1.0 - b1.powi(1))) / ((vv / (1.0 - b2.powi(1))).sqrt() + eps);
        // visit at t = 3 (single beta decay, global-step bias correction)
        m = b1 * m + (1.0 - b1) * -0.2;
        vv = b2 * vv + (1.0 - b2) * 0.2 * 0.2;
        p -= lr * (m / (1.0 - b1.powi(3))) / ((vv / (1.0 - b2.powi(3))).sqrt() + eps);

        assert_relative_eq!(params[j], p, epsilon = 1e-12);
    }

    // ----- Lazy Adam: 2-D embedding table variant -----

    #[test]
    fn test_sparse_adam_table_matches_1d() {
        // A 1x4 table with a single touched column must match the 1-D optimizer.
        let cfg = SparseAdamConfig {
            lr: 0.05,
            ..Default::default()
        };
        let mut table_opt = SparseAdamTable::new(cfg);
        let mut table =
            Array2::<f64>::from_shape_vec((1, 4), vec![0.5, -0.3, 0.7, 0.1]).expect("table");

        let mut vec_opt = SparseAdam::new(cfg);
        let mut vec_params = Array1::from_vec(vec![0.5, -0.3, 0.7, 0.1]);

        let col = 2usize;
        let grads = [0.4_f64, -0.2, 0.33];
        for &g in &grads {
            let csr = CsrGradient::new(vec![0, 1], vec![col], vec![g], (1, 4)).expect("csr");
            table_opt.step(&mut table, &csr).expect("table step");

            let coo = CooGradient::new(vec![col], vec![g], 4).expect("coo");
            vec_opt.step(&mut vec_params, &coo).expect("vec step");

            assert_relative_eq!(table[[0, col]], vec_params[col], epsilon = 1e-12);
        }
        // Untouched columns of the table are bit-identical to their originals.
        assert_eq!(table[[0, 0]].to_bits(), 0.5_f64.to_bits());
        assert_eq!(table[[0, 3]].to_bits(), 0.1_f64.to_bits());
    }

    #[test]
    fn test_sparse_adam_dim_mismatch_errs() {
        let mut opt = SparseAdam::new(SparseAdamConfig::default());
        let mut params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        // Gradient claims a different dimension than params.
        let grad = CooGradient::new(vec![0], vec![1.0], 5).expect("coo");
        let err = opt.step(&mut params, &grad);
        assert!(matches!(err, Err(GpuOptimError::DimensionMismatch { .. })));
    }
}
