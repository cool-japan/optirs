// Performance Prediction Models
//
// This module implements ML-based predictors that estimate optimizer performance
// metrics (final loss, convergence steps, peak memory, wall-clock time) from a
// vector of configuration features (learning rate, momentum, weight decay,
// batch-size log, parameter count log, optimizer-type one-hot etc.).
//
// Three estimator flavors are exposed via the `PerformancePredictor` trait:
//
// 1. `LinearRegressionPredictor` - closed-form ordinary least squares (OLS),
//    solved through the normal equations with an L2 ridge term and Gauss-Jordan
//    inversion of `X^T X + reg * I`.
// 2. `RidgeRegressionPredictor`  - identical math, but with an explicit, larger
//    default regularization strength to shrink coefficients.
// 3. `KNearestPredictor`         - non-parametric weighted k-NN over normalised
//    features, with inverse-distance weighting and deterministic tie breaking.
//
// All three respect the workspace policies: snake_case naming, scirs2_core only,
// no `.unwrap()` in production code, and a single file kept under 2000 lines.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// -----------------------------------------------------------------------------
// Public data types
// -----------------------------------------------------------------------------

/// One observed training run - the dataset row used by every predictor.
///
/// `features` is a freeform numeric vector. Suggested layout:
/// `[learning_rate, momentum, weight_decay, batch_size_log2, num_params_log2,
///   optimizer_type_onehot_0, optimizer_type_onehot_1, ...]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    /// Configuration features observed for the run.
    pub features: Vec<f64>,
    /// Final loss reached at the end of training.
    pub final_loss: f64,
    /// Number of optimizer steps until the convergence criterion was met.
    pub convergence_steps: usize,
    /// Peak memory used during the run in megabytes.
    pub peak_memory_mb: f64,
    /// Wall-clock duration of the run in seconds.
    pub wall_clock_seconds: f64,
}

impl PerformanceSample {
    /// Convenience constructor.
    pub fn new(
        features: Vec<f64>,
        final_loss: f64,
        convergence_steps: usize,
        peak_memory_mb: f64,
        wall_clock_seconds: f64,
    ) -> Self {
        Self {
            features,
            final_loss,
            convergence_steps,
            peak_memory_mb,
            wall_clock_seconds,
        }
    }

    /// Return the scalar target value associated with a [`PredictionTarget`].
    pub fn target_value(&self, target: PredictionTarget) -> f64 {
        match target {
            PredictionTarget::FinalLoss => self.final_loss,
            PredictionTarget::ConvergenceSteps => self.convergence_steps as f64,
            PredictionTarget::PeakMemoryMb => self.peak_memory_mb,
            PredictionTarget::WallClockSeconds => self.wall_clock_seconds,
        }
    }
}

/// Which scalar quantity to predict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionTarget {
    /// Final loss reached at the end of training.
    FinalLoss,
    /// Number of steps until convergence.
    ConvergenceSteps,
    /// Peak memory usage in MB.
    PeakMemoryMb,
    /// Wall-clock seconds spent training.
    WallClockSeconds,
}

/// Shared configuration for predictor training.
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    /// L2 ridge regularisation strength.
    pub regularization: f64,
    /// If true, features are zero-mean / unit-variance normalised.
    pub feature_normalize: bool,
    /// If true, a constant intercept column is appended to the design matrix.
    pub fit_intercept: bool,
    /// Maximum iterations for any iterative solver (placeholder for future use).
    pub max_iter: usize,
    /// Convergence tolerance for iterative solvers (placeholder for future use).
    pub tolerance: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            regularization: 1e-4,
            feature_normalize: true,
            fit_intercept: true,
            max_iter: 1000,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

/// Diagnostics on a fitted predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorMetrics {
    /// Coefficient of determination on the training set.
    pub r_squared: f64,
    /// Mean absolute error on the training set.
    pub mean_absolute_error: f64,
    /// Root mean squared error on the training set.
    pub root_mean_squared_error: f64,
    /// Number of training samples used.
    pub training_samples: usize,
    /// Number of features in each sample (excluding the intercept).
    pub num_features: usize,
}

// -----------------------------------------------------------------------------
// Predictor trait
// -----------------------------------------------------------------------------

/// Unified interface for performance predictors.
pub trait PerformancePredictor: Send + Sync + Debug {
    /// Fit the predictor on `samples`, predicting `target`.
    fn fit(
        &mut self,
        samples: &[PerformanceSample],
        target: PredictionTarget,
    ) -> Result<PredictorMetrics>;

    /// Predict the scalar target for a single feature vector.
    fn predict(&self, features: &[f64]) -> Result<f64>;

    /// Predict the scalar target for a batch of feature rows (rows = samples).
    fn predict_batch(&self, feature_matrix: &Array2<f64>) -> Result<Array1<f64>>;

    /// Training metrics, if available (`None` before [`Self::fit`]).
    fn metrics(&self) -> Option<&PredictorMetrics>;

    /// The target this predictor was fit for, if any.
    fn target(&self) -> Option<PredictionTarget>;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

/// Zero-mean / unit-variance feature normalisation along columns.
///
/// Returns the normalised matrix, the per-column means, and the per-column
/// standard deviations. Each `std` is clamped at `1e-12` to avoid division by
/// zero when a feature is constant across all samples.
pub fn normalize_features(matrix: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let (n_rows, n_cols) = matrix.dim();
    let mut means = Array1::<f64>::zeros(n_cols);
    let mut stds = Array1::<f64>::zeros(n_cols);

    if n_rows == 0 {
        // Cannot normalise an empty matrix; return as-is.
        return (matrix.clone(), means, Array1::from_elem(n_cols, 1.0));
    }

    let n_rows_f = n_rows as f64;
    for col in 0..n_cols {
        let mut sum = 0.0_f64;
        for row in 0..n_rows {
            sum += matrix[[row, col]];
        }
        means[col] = sum / n_rows_f;
    }

    for col in 0..n_cols {
        let mut var_sum = 0.0_f64;
        for row in 0..n_rows {
            let d = matrix[[row, col]] - means[col];
            var_sum += d * d;
        }
        let var = var_sum / n_rows_f;
        let std = var.sqrt();
        stds[col] = if std < 1e-12 { 1e-12 } else { std };
    }

    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for row in 0..n_rows {
        for col in 0..n_cols {
            out[[row, col]] = (matrix[[row, col]] - means[col]) / stds[col];
        }
    }

    (out, means, stds)
}

/// Coefficient of determination R^2.
///
/// Returns 1.0 when the predictions exactly equal the targets, and 0.0 when the
/// prediction equals the mean of `y_true`. Negative values are possible when the
/// prediction is worse than predicting the mean.
pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || y_true.len() != y_pred.len() {
        return 0.0;
    }
    let mean: f64 = y_true.iter().copied().sum::<f64>() / (n as f64);
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..n {
        let r = y_true[i] - y_pred[i];
        ss_res += r * r;
        let t = y_true[i] - mean;
        ss_tot += t * t;
    }
    if ss_tot < 1e-30 {
        // y_true is constant: define R^2 = 1 if ss_res is also zero, else 0.
        if ss_res < 1e-30 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Mean absolute error.
pub fn mean_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || y_true.len() != y_pred.len() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += (y_true[i] - y_pred[i]).abs();
    }
    sum / (n as f64)
}

/// Root mean squared error.
pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || y_true.len() != y_pred.len() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let d = y_true[i] - y_pred[i];
        sum += d * d;
    }
    (sum / (n as f64)).sqrt()
}

/// Random train/test split with the supplied seed.
///
/// `test_fraction` must lie in `[0.0, 1.0]`. The samples are shuffled
/// deterministically with the given seed before being split.
pub fn train_test_split(
    samples: &[PerformanceSample],
    test_fraction: f64,
    seed: u64,
) -> Result<(Vec<PerformanceSample>, Vec<PerformanceSample>)> {
    if !(0.0..=1.0).contains(&test_fraction) || test_fraction.is_nan() {
        return Err(OptimError::InvalidConfig(format!(
            "test_fraction must be in [0, 1], got {test_fraction}"
        )));
    }
    let n = samples.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle using the SciRS2 RNG.
    let mut rng = Random::seed(seed);
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }

    let test_size = (n as f64 * test_fraction).round() as usize;
    let test_size = test_size.min(n);
    let train_size = n - test_size;

    let mut train = Vec::with_capacity(train_size);
    let mut test = Vec::with_capacity(test_size);
    for (k, idx) in indices.iter().enumerate() {
        let sample = samples[*idx].clone();
        if k < train_size {
            train.push(sample);
        } else {
            test.push(sample);
        }
    }
    Ok((train, test))
}

// -----------------------------------------------------------------------------
// Internal linear algebra helpers
// -----------------------------------------------------------------------------

/// Build a `(n_samples, n_features)` design matrix from `samples`, optionally
/// prepending a column of ones for the intercept.
fn build_design_matrix(
    samples: &[PerformanceSample],
    fit_intercept: bool,
) -> Result<(Array2<f64>, Array1<f64>, usize)> {
    if samples.is_empty() {
        return Err(OptimError::InvalidConfig(
            "Cannot build design matrix from zero samples".to_string(),
        ));
    }
    let n_features = samples[0].features.len();
    if n_features == 0 {
        return Err(OptimError::InvalidConfig(
            "Each sample must contain at least one feature".to_string(),
        ));
    }
    for (idx, s) in samples.iter().enumerate() {
        if s.features.len() != n_features {
            return Err(OptimError::DimensionMismatch(format!(
                "Sample {idx} has {} features, expected {n_features}",
                s.features.len()
            )));
        }
    }

    let n_samples = samples.len();
    let n_cols = if fit_intercept {
        n_features + 1
    } else {
        n_features
    };
    let mut x = Array2::<f64>::zeros((n_samples, n_cols));
    let mut y = Array1::<f64>::zeros(n_samples);
    for (i, s) in samples.iter().enumerate() {
        let offset = if fit_intercept {
            x[[i, 0]] = 1.0;
            1
        } else {
            0
        };
        for (j, v) in s.features.iter().copied().enumerate() {
            x[[i, j + offset]] = v;
        }
        // y is filled by the caller; we leave zeros here.
        let _ = &mut y; // touch to satisfy borrow-checker style
    }
    Ok((x, y, n_features))
}

/// Compute the matrix inverse of a square symmetric positive (semi)-definite
/// matrix via Gauss-Jordan elimination with partial pivoting. Returns
/// `ComputationError` if a pivot smaller than `1e-12` is encountered, which
/// signals a singular (or near-singular) matrix.
fn invert_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = matrix.dim();
    if n_rows != n_cols {
        return Err(OptimError::DimensionMismatch(format!(
            "invert_matrix: expected square matrix, got {n_rows}x{n_cols}"
        )));
    }
    let n = n_rows;
    if n == 0 {
        return Err(OptimError::InvalidConfig(
            "Cannot invert a 0x0 matrix".to_string(),
        ));
    }

    // Build augmented [A | I] of size n x 2n.
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: find the row in [col..n] with max |aug[row, col]|.
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-12 {
            return Err(OptimError::ComputationError(format!(
                "Singular matrix detected during inversion at column {col} (pivot={pivot_val})"
            )));
        }
        if pivot_row != col {
            // Swap rows pivot_row and col.
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        // Normalise pivot row so that aug[col, col] becomes 1.
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column `col` in every other row.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Extract the right-hand block as the inverse.
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Solve `(X^T X + reg * I_eff) theta = X^T y` and return `theta`.
///
/// When `fit_intercept` is true the first column of `X` is the intercept and is
/// **not** penalised, mirroring the standard scikit-learn convention. `reg` must
/// be non-negative; passing `reg = 0.0` reduces to plain OLS.
fn solve_normal_equations(
    x: &Array2<f64>,
    y: &Array1<f64>,
    reg: f64,
    fit_intercept: bool,
) -> Result<Array1<f64>> {
    if reg < 0.0 || !reg.is_finite() {
        return Err(OptimError::InvalidParameter(format!(
            "Regularisation must be a finite non-negative number, got {reg}"
        )));
    }
    let (n_samples, n_cols) = x.dim();
    if y.len() != n_samples {
        return Err(OptimError::DimensionMismatch(format!(
            "X has {n_samples} rows but y has {} entries",
            y.len()
        )));
    }
    if n_samples == 0 || n_cols == 0 {
        return Err(OptimError::InvalidConfig(
            "Cannot solve normal equations on an empty design matrix".to_string(),
        ));
    }

    // Compute X^T X (n_cols x n_cols).
    let xt = x.t();
    let mut gram = xt.dot(x);

    // Add the ridge penalty on the diagonal, skipping the intercept term.
    if reg > 0.0 {
        let start = if fit_intercept { 1 } else { 0 };
        for i in start..n_cols {
            gram[[i, i]] += reg;
        }
    }

    // Compute X^T y (length n_cols).
    let xt_y = xt.dot(y);

    // Invert the regularised Gram matrix and multiply by X^T y.
    let inv = invert_matrix(&gram)?;
    Ok(inv.dot(&xt_y))
}

// -----------------------------------------------------------------------------
// Linear regression predictor (closed-form OLS / ridge)
// -----------------------------------------------------------------------------

/// Closed-form linear regression via the normal equations.
///
/// When `PredictorConfig::feature_normalize` is `true` the design matrix is
/// standardised (zero mean, unit variance per column) before solving, and the
/// learned mean/std are kept so that [`Self::predict`] can reapply them.
#[derive(Debug, Clone)]
pub struct LinearRegressionPredictor {
    config: PredictorConfig,
    /// Learned coefficient vector. When `fit_intercept` is true, the intercept
    /// is stored as the first element.
    coefficients: Option<Array1<f64>>,
    /// Per-feature mean (only when feature_normalize=true).
    feature_means: Option<Array1<f64>>,
    /// Per-feature standard deviation (only when feature_normalize=true).
    feature_stds: Option<Array1<f64>>,
    metrics: Option<PredictorMetrics>,
    target: Option<PredictionTarget>,
}

impl LinearRegressionPredictor {
    /// Construct a new linear regression predictor.
    pub fn new(config: PredictorConfig) -> Self {
        Self {
            config,
            coefficients: None,
            feature_means: None,
            feature_stds: None,
            metrics: None,
            target: None,
        }
    }

    /// Set whether features should be normalised before fitting.
    pub fn with_feature_normalize(mut self, normalize: bool) -> Self {
        self.config.feature_normalize = normalize;
        self
    }

    /// Set whether an intercept term should be fit.
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the L2 regularisation strength.
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.config.regularization = reg;
        self
    }

    /// Return the learned coefficient vector (intercept first when fit).
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Internal: transform a single feature vector identical to fit-time logic.
    fn transform_features(&self, features: &[f64]) -> Result<Array1<f64>> {
        let coefs = self.coefficients.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "Predictor has not been fit yet; call fit() before predict()".to_string(),
            )
        })?;
        let expected = if self.config.fit_intercept {
            coefs.len() - 1
        } else {
            coefs.len()
        };
        if features.len() != expected {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected {expected} features, got {}",
                features.len()
            )));
        }
        let mut vec = Array1::<f64>::zeros(coefs.len());
        let offset = if self.config.fit_intercept {
            vec[0] = 1.0;
            1
        } else {
            0
        };
        for (i, v) in features.iter().copied().enumerate() {
            let v_norm = if self.config.feature_normalize {
                let means = self.feature_means.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "Feature means missing despite feature_normalize=true".to_string(),
                    )
                })?;
                let stds = self.feature_stds.as_ref().ok_or_else(|| {
                    OptimError::InvalidState(
                        "Feature stds missing despite feature_normalize=true".to_string(),
                    )
                })?;
                (v - means[i]) / stds[i]
            } else {
                v
            };
            vec[i + offset] = v_norm;
        }
        Ok(vec)
    }
}

impl PerformancePredictor for LinearRegressionPredictor {
    fn fit(
        &mut self,
        samples: &[PerformanceSample],
        target: PredictionTarget,
    ) -> Result<PredictorMetrics> {
        if samples.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Cannot fit predictor on zero samples".to_string(),
            ));
        }

        let (mut x_raw, mut y, n_features) =
            build_design_matrix(samples, self.config.fit_intercept)?;

        // Fill y from the chosen target.
        for (i, s) in samples.iter().enumerate() {
            y[i] = s.target_value(target);
        }

        if self.config.feature_normalize {
            // Normalise only the feature columns (skipping the intercept column).
            let offset = if self.config.fit_intercept { 1 } else { 0 };
            let n_samples = samples.len();
            let n_feature_cols = n_features;

            // Slice out the feature submatrix.
            let mut feat_mat = Array2::<f64>::zeros((n_samples, n_feature_cols));
            for i in 0..n_samples {
                for j in 0..n_feature_cols {
                    feat_mat[[i, j]] = x_raw[[i, j + offset]];
                }
            }
            let (norm_mat, means, stds) = normalize_features(&feat_mat);
            for i in 0..n_samples {
                for j in 0..n_feature_cols {
                    x_raw[[i, j + offset]] = norm_mat[[i, j]];
                }
            }
            self.feature_means = Some(means);
            self.feature_stds = Some(stds);
        } else {
            self.feature_means = None;
            self.feature_stds = None;
        }

        let coefs = solve_normal_equations(
            &x_raw,
            &y,
            self.config.regularization,
            self.config.fit_intercept,
        )?;

        // Compute training predictions for metrics.
        let preds = x_raw.dot(&coefs);
        let r2 = r_squared(&y, &preds);
        let mae = mean_absolute_error(&y, &preds);
        let rmse = root_mean_squared_error(&y, &preds);
        let metrics = PredictorMetrics {
            r_squared: r2,
            mean_absolute_error: mae,
            root_mean_squared_error: rmse,
            training_samples: samples.len(),
            num_features: n_features,
        };

        self.coefficients = Some(coefs);
        self.target = Some(target);
        self.metrics = Some(metrics.clone());
        Ok(metrics)
    }

    fn predict(&self, features: &[f64]) -> Result<f64> {
        let coefs = self.coefficients.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "Predictor has not been fit yet; call fit() before predict()".to_string(),
            )
        })?;
        let vec = self.transform_features(features)?;
        Ok(coefs.dot(&vec))
    }

    fn predict_batch(&self, feature_matrix: &Array2<f64>) -> Result<Array1<f64>> {
        let (n_samples, _) = feature_matrix.dim();
        let mut out = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let row: Vec<f64> = feature_matrix.row(i).iter().copied().collect();
            out[i] = self.predict(&row)?;
        }
        Ok(out)
    }

    fn metrics(&self) -> Option<&PredictorMetrics> {
        self.metrics.as_ref()
    }

    fn target(&self) -> Option<PredictionTarget> {
        self.target
    }
}

// -----------------------------------------------------------------------------
// Ridge regression predictor
// -----------------------------------------------------------------------------

/// Closed-form ridge regression.
///
/// Mathematically identical to [`LinearRegressionPredictor`] with a positive
/// regularisation strength; provided as a separate type so that the default
/// regularisation can be meaningfully shrinking (`0.1` rather than `1e-4`) and
/// to make user intent explicit at construction sites.
#[derive(Debug, Clone)]
pub struct RidgeRegressionPredictor {
    inner: LinearRegressionPredictor,
}

impl RidgeRegressionPredictor {
    /// Construct a ridge predictor with the given regularisation strength.
    pub fn new(reg: f64) -> Self {
        let effective_reg = if reg.is_finite() && reg > 0.0 {
            reg
        } else {
            0.1
        };
        let config = PredictorConfig {
            regularization: effective_reg,
            ..PredictorConfig::default()
        };
        Self {
            inner: LinearRegressionPredictor::new(config),
        }
    }

    /// Construct a ridge predictor from a custom config; the `regularization`
    /// field is left untouched, so callers can supply any positive value.
    pub fn with_config(config: PredictorConfig) -> Self {
        Self {
            inner: LinearRegressionPredictor::new(config),
        }
    }

    /// Set feature normalisation.
    pub fn with_feature_normalize(mut self, normalize: bool) -> Self {
        self.inner = self.inner.with_feature_normalize(normalize);
        self
    }

    /// Set intercept fitting.
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.inner = self.inner.with_fit_intercept(fit_intercept);
        self
    }

    /// Access the learned coefficients.
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.inner.coefficients()
    }

    /// Access the wrapped predictor (mostly useful for tests).
    pub fn inner(&self) -> &LinearRegressionPredictor {
        &self.inner
    }
}

impl PerformancePredictor for RidgeRegressionPredictor {
    fn fit(
        &mut self,
        samples: &[PerformanceSample],
        target: PredictionTarget,
    ) -> Result<PredictorMetrics> {
        self.inner.fit(samples, target)
    }

    fn predict(&self, features: &[f64]) -> Result<f64> {
        self.inner.predict(features)
    }

    fn predict_batch(&self, feature_matrix: &Array2<f64>) -> Result<Array1<f64>> {
        self.inner.predict_batch(feature_matrix)
    }

    fn metrics(&self) -> Option<&PredictorMetrics> {
        self.inner.metrics()
    }

    fn target(&self) -> Option<PredictionTarget> {
        self.inner.target()
    }
}

// -----------------------------------------------------------------------------
// K-nearest neighbours predictor
// -----------------------------------------------------------------------------

/// Non-parametric weighted-kNN regression on normalised features.
///
/// Predictions are an inverse-distance-weighted average of the `k` nearest
/// training samples in standardised feature space. Tie-breaking on distance
/// prefers the lower index for deterministic behaviour.
#[derive(Debug, Clone)]
pub struct KNearestPredictor {
    /// Number of neighbours.
    k: usize,
    /// Shared config (we only really use `feature_normalize`).
    config: PredictorConfig,
    /// Normalised training features.
    train_features: Option<Array2<f64>>,
    /// Training targets (after the chosen [`PredictionTarget`] has been applied).
    train_targets: Option<Array1<f64>>,
    /// Per-feature mean / std for re-normalising query points.
    feature_means: Option<Array1<f64>>,
    feature_stds: Option<Array1<f64>>,
    metrics: Option<PredictorMetrics>,
    target: Option<PredictionTarget>,
}

impl KNearestPredictor {
    /// Construct a kNN predictor with `k` neighbours and a default config.
    pub fn new(k: usize) -> Self {
        Self::with_config(k, PredictorConfig::default())
    }

    /// Construct a kNN predictor with a custom config.
    pub fn with_config(k: usize, config: PredictorConfig) -> Self {
        Self {
            k,
            config,
            train_features: None,
            train_targets: None,
            feature_means: None,
            feature_stds: None,
            metrics: None,
            target: None,
        }
    }

    /// Set whether features should be normalised.
    pub fn with_feature_normalize(mut self, normalize: bool) -> Self {
        self.config.feature_normalize = normalize;
        self
    }

    /// Inspect the configured `k`.
    pub fn k(&self) -> usize {
        self.k
    }

    fn normalise_query(&self, features: &[f64]) -> Result<Array1<f64>> {
        let stored = self.train_features.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "KNearestPredictor has not been fit yet; call fit() before predict()".to_string(),
            )
        })?;
        let expected = stored.dim().1;
        if features.len() != expected {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected {expected} features, got {}",
                features.len()
            )));
        }
        let mut out = Array1::<f64>::zeros(expected);
        if self.config.feature_normalize {
            let means = self.feature_means.as_ref().ok_or_else(|| {
                OptimError::InvalidState(
                    "Feature means missing despite feature_normalize=true".to_string(),
                )
            })?;
            let stds = self.feature_stds.as_ref().ok_or_else(|| {
                OptimError::InvalidState(
                    "Feature stds missing despite feature_normalize=true".to_string(),
                )
            })?;
            for (i, v) in features.iter().copied().enumerate() {
                out[i] = (v - means[i]) / stds[i];
            }
        } else {
            for (i, v) in features.iter().copied().enumerate() {
                out[i] = v;
            }
        }
        Ok(out)
    }
}

impl PerformancePredictor for KNearestPredictor {
    fn fit(
        &mut self,
        samples: &[PerformanceSample],
        target: PredictionTarget,
    ) -> Result<PredictorMetrics> {
        if samples.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Cannot fit kNN predictor on zero samples".to_string(),
            ));
        }
        if self.k == 0 {
            return Err(OptimError::InvalidConfig(
                "k must be >= 1 for KNearestPredictor".to_string(),
            ));
        }

        // Build raw feature matrix (no intercept column for kNN).
        let (raw, mut y, n_features) = build_design_matrix(samples, false)?;
        for (i, s) in samples.iter().enumerate() {
            y[i] = s.target_value(target);
        }

        let (normed, means, stds) = if self.config.feature_normalize {
            let (n, m, s) = normalize_features(&raw);
            (n, Some(m), Some(s))
        } else {
            (raw.clone(), None, None)
        };

        // Compute training predictions to populate metrics.
        // Since we have only training points here, predict by leaving one out is
        // expensive; we approximate using the in-sample prediction (which for
        // distance-weighted kNN with epsilon collapses to the same point and
        // therefore yields near-perfect fit). To avoid trivially perfect
        // metrics, we predict using the second-nearest onwards when an exact
        // match is present.
        let mut preds = Array1::<f64>::zeros(samples.len());
        for i in 0..samples.len() {
            let query = normed.row(i).to_owned();
            preds[i] = weighted_knn_predict_internal(&normed, &y, &query, self.k, Some(i))?;
        }
        let r2 = r_squared(&y, &preds);
        let mae = mean_absolute_error(&y, &preds);
        let rmse = root_mean_squared_error(&y, &preds);
        let metrics = PredictorMetrics {
            r_squared: r2,
            mean_absolute_error: mae,
            root_mean_squared_error: rmse,
            training_samples: samples.len(),
            num_features: n_features,
        };

        self.train_features = Some(normed);
        self.train_targets = Some(y);
        self.feature_means = means;
        self.feature_stds = stds;
        self.target = Some(target);
        self.metrics = Some(metrics.clone());
        Ok(metrics)
    }

    fn predict(&self, features: &[f64]) -> Result<f64> {
        let train = self.train_features.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "Predictor has not been fit yet; call fit() before predict()".to_string(),
            )
        })?;
        let targets = self.train_targets.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "Predictor has not been fit yet; call fit() before predict()".to_string(),
            )
        })?;
        let query = self.normalise_query(features)?;
        weighted_knn_predict_internal(train, targets, &query, self.k, None)
    }

    fn predict_batch(&self, feature_matrix: &Array2<f64>) -> Result<Array1<f64>> {
        let (n_samples, _) = feature_matrix.dim();
        let mut out = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let row: Vec<f64> = feature_matrix.row(i).iter().copied().collect();
            out[i] = self.predict(&row)?;
        }
        Ok(out)
    }

    fn metrics(&self) -> Option<&PredictorMetrics> {
        self.metrics.as_ref()
    }

    fn target(&self) -> Option<PredictionTarget> {
        self.target
    }
}

/// Inverse-distance weighted prediction over the `k` nearest rows of `train`.
///
/// When `exclude_idx` is `Some`, that training row is removed from the
/// candidate set (useful for leave-one-out style scoring during fit).
fn weighted_knn_predict_internal(
    train: &Array2<f64>,
    targets: &Array1<f64>,
    query: &Array1<f64>,
    k: usize,
    exclude_idx: Option<usize>,
) -> Result<f64> {
    let (n_rows, n_cols) = train.dim();
    if n_rows == 0 {
        return Err(OptimError::InvalidState(
            "kNN training set is empty".to_string(),
        ));
    }
    if query.len() != n_cols {
        return Err(OptimError::DimensionMismatch(format!(
            "Query vector has {} dimensions but training has {n_cols}",
            query.len()
        )));
    }
    if k == 0 {
        return Err(OptimError::InvalidConfig(
            "k must be >= 1 for KNearestPredictor".to_string(),
        ));
    }

    // Compute squared Euclidean distances.
    let mut dist_idx: Vec<(f64, usize)> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        if Some(i) == exclude_idx {
            continue;
        }
        let mut sum_sq = 0.0_f64;
        for j in 0..n_cols {
            let d = train[[i, j]] - query[j];
            sum_sq += d * d;
        }
        dist_idx.push((sum_sq.sqrt(), i));
    }
    if dist_idx.is_empty() {
        return Err(OptimError::InvalidState(
            "No training samples available after exclusion".to_string(),
        ));
    }

    // Sort by (distance, index) ascending for deterministic tie-breaking.
    dist_idx.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
        Some(std::cmp::Ordering::Equal) | None => a.1.cmp(&b.1),
        Some(o) => o,
    });

    let k_eff = k.min(dist_idx.len());
    let epsilon = 1e-12_f64;
    let mut weighted_sum = 0.0_f64;
    let mut weight_total = 0.0_f64;
    for (dist, idx) in dist_idx.iter().take(k_eff).copied() {
        let w = 1.0 / (dist + epsilon);
        weighted_sum += w * targets[idx];
        weight_total += w;
    }
    if weight_total <= 0.0 || !weight_total.is_finite() {
        return Err(OptimError::ComputationError(
            "kNN weight total is zero or non-finite".to_string(),
        ));
    }
    Ok(weighted_sum / weight_total)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_sample(features: Vec<f64>, y: f64) -> PerformanceSample {
        // The target is stored in `final_loss`; the other fields are placeholders.
        PerformanceSample::new(features, y, 0, 0.0, 0.0)
    }

    /// Generate y = 2*x1 + 3*x2 + 5 perfectly, no noise.
    ///
    /// Uses non-collinear features (a deterministic pseudo-random pattern via
    /// trigonometric phases) so that X^T X is well-conditioned.
    fn linear_dataset(n: usize) -> Vec<PerformanceSample> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let x1 = (t * 0.37).sin() * 2.5 + 0.1 * t;
            let x2 = (t * 0.91 + 0.4).cos() * 1.8 - 0.05 * t + 0.5;
            let y = 2.0 * x1 + 3.0 * x2 + 5.0;
            out.push(make_sample(vec![x1, x2], y));
        }
        out
    }

    #[test]
    fn test_linear_regression_fits_exactly_for_linear_data() {
        let samples = linear_dataset(40);
        let mut predictor = LinearRegressionPredictor::new(PredictorConfig {
            regularization: 0.0,
            feature_normalize: false,
            fit_intercept: true,
            ..PredictorConfig::default()
        });
        let metrics = predictor
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit should succeed");
        let coefs = predictor.coefficients().expect("coefficients available");
        // coefs[0] = intercept, coefs[1] = w1, coefs[2] = w2
        assert_relative_eq!(coefs[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(coefs[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(coefs[2], 3.0, epsilon = 1e-6);
        assert!(
            metrics.mean_absolute_error < 1e-6,
            "MAE too large: {}",
            metrics.mean_absolute_error
        );
        // Spot-check predict()
        let p = predictor.predict(&[1.0, 2.0]).expect("predict ok");
        assert_relative_eq!(p, 2.0 * 1.0 + 3.0 * 2.0 + 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_regression_with_normalization() {
        let samples = linear_dataset(50);
        let mut predictor = LinearRegressionPredictor::new(PredictorConfig {
            regularization: 0.0,
            feature_normalize: true,
            fit_intercept: true,
            ..PredictorConfig::default()
        });
        predictor
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit should succeed");
        // Predictions must still match the true linear function in the
        // *original* feature space because the transform is invertible.
        let p1 = predictor.predict(&[0.5, 1.5]).expect("predict ok");
        let truth = 2.0 * 0.5 + 3.0 * 1.5 + 5.0;
        assert_relative_eq!(p1, truth, epsilon = 1e-6);
        let p2 = predictor.predict(&[-2.0, 4.0]).expect("predict ok");
        let truth2 = 2.0 * -2.0 + 3.0 * 4.0 + 5.0;
        assert_relative_eq!(p2, truth2, epsilon = 1e-6);
    }

    #[test]
    fn test_ridge_shrinks_coefficients_toward_zero() {
        let samples = linear_dataset(30);
        let mut linear = LinearRegressionPredictor::new(PredictorConfig {
            regularization: 0.0,
            feature_normalize: true,
            fit_intercept: true,
            ..PredictorConfig::default()
        });
        linear
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit linear");
        let lin_coefs = linear.coefficients().expect("linear coefs");
        let lin_norm: f64 = lin_coefs.iter().skip(1).map(|c| c * c).sum::<f64>().sqrt();

        let mut ridge = RidgeRegressionPredictor::new(10.0)
            .with_feature_normalize(true)
            .with_fit_intercept(true);
        ridge
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit ridge");
        let ridge_coefs = ridge.coefficients().expect("ridge coefs");
        let ridge_norm: f64 = ridge_coefs
            .iter()
            .skip(1)
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();
        assert!(
            ridge_norm < lin_norm,
            "Ridge should shrink coefficients: ridge_norm={ridge_norm}, lin_norm={lin_norm}"
        );
    }

    #[test]
    fn test_ridge_regularization_extreme_makes_predictions_near_mean() {
        let samples = linear_dataset(40);
        let mean_y: f64 = samples.iter().map(|s| s.final_loss).sum::<f64>() / samples.len() as f64;
        let mut ridge = RidgeRegressionPredictor::new(1e8)
            .with_feature_normalize(true)
            .with_fit_intercept(true);
        ridge
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit ridge");
        // With huge L2 only the intercept survives; predictions tend to mean(y).
        for s in samples.iter().take(5) {
            let p = ridge.predict(&s.features).expect("predict ok");
            assert!(
                (p - mean_y).abs() < 1e-3,
                "Prediction {p} deviates from mean {mean_y}"
            );
        }
    }

    #[test]
    fn test_knn_predict_with_exact_match_returns_exact_value() {
        let samples = linear_dataset(20);
        let mut knn = KNearestPredictor::new(3).with_feature_normalize(true);
        knn.fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit knn");
        // Querying with an exact training sample should yield (almost) that sample's y
        // because its distance is zero and 1/(0 + eps) dominates the weighted average.
        let target = samples[7].final_loss;
        let p = knn.predict(&samples[7].features).expect("predict ok");
        assert_relative_eq!(p, target, epsilon = 1e-3);
    }

    #[test]
    fn test_knn_with_k_equals_n_returns_weighted_mean() {
        // With k = N, kNN reduces to an inverse-distance-weighted mean over the
        // entire training set. For a query "far enough" from any sample (relative
        // to inter-sample spacing), the weights are roughly comparable and the
        // prediction should sit near the unweighted mean.
        let samples = linear_dataset(30);
        let n = samples.len();
        let mut knn = KNearestPredictor::new(n).with_feature_normalize(true);
        knn.fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit knn");
        let mean_y: f64 = samples.iter().map(|s| s.final_loss).sum::<f64>() / n as f64;
        // Query far away in feature space.
        let p = knn.predict(&[100.0, 100.0]).expect("predict ok");
        // The inverse-distance-weighted mean of an evenly-spread dataset around a
        // single point converges to the unweighted mean. Allow a generous epsilon.
        assert!(
            (p - mean_y).abs() < 5.0,
            "kNN(k=n) {p} far from mean {mean_y}"
        );
    }

    #[test]
    fn test_train_test_split_proportions() {
        let samples = linear_dataset(100);
        let (train, test) = train_test_split(&samples, 0.2, 7).expect("split ok");
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_train_test_split_no_overlap() {
        let samples = linear_dataset(50);
        let (train, test) = train_test_split(&samples, 0.3, 11).expect("split ok");
        // Concatenating train+test as a multiset must match the original samples.
        let mut combined_targets: Vec<f64> = train
            .iter()
            .chain(test.iter())
            .map(|s| s.final_loss)
            .collect();
        let mut original_targets: Vec<f64> = samples.iter().map(|s| s.final_loss).collect();
        combined_targets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        original_targets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(combined_targets.len(), original_targets.len());
        for (a, b) in combined_targets.iter().zip(original_targets.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_train_test_split_invalid_fraction_errors() {
        let samples = linear_dataset(10);
        let bad_low = train_test_split(&samples, -0.1, 0);
        assert!(matches!(bad_low, Err(OptimError::InvalidConfig(_))));
        let bad_high = train_test_split(&samples, 1.1, 0);
        assert!(matches!(bad_high, Err(OptimError::InvalidConfig(_))));
    }

    #[test]
    fn test_r_squared_perfect_fit_is_one() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let r2 = r_squared(&y, &y);
        assert_relative_eq!(r2, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_r_squared_mean_predictor_is_zero() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean = y.iter().copied().sum::<f64>() / y.len() as f64;
        let pred = Array1::from_elem(y.len(), mean);
        let r2 = r_squared(&y, &pred);
        assert!(r2.abs() < 1e-12, "R^2 for mean predictor = {r2}");
    }

    #[test]
    fn test_predictor_metrics_populated() {
        let samples = linear_dataset(25);
        let mut predictor = LinearRegressionPredictor::new(PredictorConfig::default());
        let metrics = predictor
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit ok");
        assert_eq!(metrics.training_samples, 25);
        assert_eq!(metrics.num_features, 2);
        let stored = predictor.metrics().expect("metrics stored");
        assert_eq!(stored.training_samples, 25);
        assert_eq!(predictor.target(), Some(PredictionTarget::FinalLoss));
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let lin = LinearRegressionPredictor::new(PredictorConfig::default());
        let err = lin.predict(&[1.0, 2.0]).expect_err("must error");
        assert!(matches!(err, OptimError::InvalidState(_)));
        let ridge = RidgeRegressionPredictor::new(0.1);
        let err2 = ridge.predict(&[1.0, 2.0]).expect_err("must error");
        assert!(matches!(err2, OptimError::InvalidState(_)));
        let knn = KNearestPredictor::new(3);
        let err3 = knn.predict(&[1.0, 2.0]).expect_err("must error");
        assert!(matches!(err3, OptimError::InvalidState(_)));
    }

    #[test]
    fn test_predict_batch_consistency() {
        let samples = linear_dataset(20);
        let mut predictor = LinearRegressionPredictor::new(PredictorConfig {
            regularization: 1e-6,
            feature_normalize: true,
            fit_intercept: true,
            ..PredictorConfig::default()
        });
        predictor
            .fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit ok");
        // Build a small batch matrix.
        let queries = [
            vec![0.0, 0.0],
            vec![1.0, -1.0],
            vec![2.5, 3.5],
            vec![-1.0, 2.0],
        ];
        let mut mat = Array2::<f64>::zeros((queries.len(), 2));
        for (i, q) in queries.iter().enumerate() {
            mat[[i, 0]] = q[0];
            mat[[i, 1]] = q[1];
        }
        let batch = predictor.predict_batch(&mat).expect("batch ok");
        for (i, q) in queries.iter().enumerate() {
            let one = predictor.predict(q).expect("predict ok");
            assert_relative_eq!(batch[i], one, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_features_zero_mean_unit_variance() {
        let mut data = Array2::<f64>::zeros((5, 2));
        for i in 0..5 {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = (i as f64) * 2.0;
        }
        let (norm, means, stds) = normalize_features(&data);
        // Column means should be ~0.
        for col in 0..2 {
            let mean_col: f64 = (0..5).map(|r| norm[[r, col]]).sum::<f64>() / 5.0;
            assert!(mean_col.abs() < 1e-9, "Mean column {col} = {mean_col}");
        }
        // Stored means should match the column means of the original data.
        assert_relative_eq!(means[0], 2.0, epsilon = 1e-9);
        assert_relative_eq!(means[1], 4.0, epsilon = 1e-9);
        assert!(stds[0] > 0.0 && stds[1] > 0.0);
    }

    #[test]
    fn test_invert_matrix_round_trip() {
        // A simple invertible 3x3 matrix.
        let mut m = Array2::<f64>::zeros((3, 3));
        m[[0, 0]] = 4.0;
        m[[0, 1]] = 7.0;
        m[[0, 2]] = 2.0;
        m[[1, 0]] = 3.0;
        m[[1, 1]] = 6.0;
        m[[1, 2]] = 1.0;
        m[[2, 0]] = 2.0;
        m[[2, 1]] = 5.0;
        m[[2, 2]] = 3.0;
        let inv = invert_matrix(&m).expect("invertible");
        let prod = m.dot(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(prod[[i, j]], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_invert_matrix_singular_errors() {
        // Rank-1 matrix is singular.
        let mut m = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                m[[i, j]] = 1.0;
            }
        }
        let err = invert_matrix(&m).expect_err("singular must fail");
        assert!(matches!(err, OptimError::ComputationError(_)));
    }

    #[test]
    fn test_dimension_mismatch_on_predict() {
        let samples = linear_dataset(10);
        let mut lin = LinearRegressionPredictor::new(PredictorConfig::default());
        lin.fit(&samples, PredictionTarget::FinalLoss)
            .expect("fit ok");
        // Wrong number of features.
        let err = lin.predict(&[1.0]).expect_err("must error");
        assert!(matches!(err, OptimError::DimensionMismatch(_)));
    }

    #[test]
    fn test_all_targets_can_be_fit() {
        // Build samples with explicit target values to exercise every PredictionTarget variant.
        // To keep the design matrix well-conditioned and the relationship to every target
        // exactly linear, derive each target as a direct linear function of the features.
        let mut samples = Vec::new();
        for i in 0..15 {
            let t = i as f64;
            let x1 = (t * 0.21).sin() * 1.2 + 0.3 * t;
            let x2 = (t * 0.77).cos() * 0.9 - 0.1 * t;
            let loss = 10.0 - 0.3 * x1 + 0.2 * x2;
            let steps_f = 5.0 + 2.0 * x1 + 0.5 * x2;
            // round-half-away-from-zero to nearest non-negative usize for steps.
            let steps = if steps_f < 0.0 {
                0
            } else {
                steps_f.round() as usize
            };
            let memory = 100.0 + 1.5 * x1 - 0.7 * x2;
            let wall = 1.0 + 0.4 * x1 + 0.1 * x2;
            samples.push(PerformanceSample::new(
                vec![x1, x2],
                loss,
                steps,
                memory,
                wall,
            ));
        }
        for target in [
            PredictionTarget::FinalLoss,
            PredictionTarget::ConvergenceSteps,
            PredictionTarget::PeakMemoryMb,
            PredictionTarget::WallClockSeconds,
        ] {
            let mut lin = LinearRegressionPredictor::new(PredictorConfig {
                regularization: 0.0,
                feature_normalize: false,
                fit_intercept: true,
                ..PredictorConfig::default()
            });
            let metrics = lin.fit(&samples, target).expect("fit ok");
            // ConvergenceSteps is rounded to a usize, so it carries quantisation
            // noise; the other targets are exact linear functions of the features.
            let threshold = match target {
                PredictionTarget::ConvergenceSteps => 0.85,
                _ => 0.99,
            };
            assert!(
                metrics.r_squared > threshold,
                "R^2 too low for {target:?}: got {}",
                metrics.r_squared
            );
        }
    }

    #[test]
    fn test_ridge_default_regularization() {
        // Even when reg is non-positive or non-finite, the constructor must
        // fall back to a positive default rather than panicking.
        let r1 = RidgeRegressionPredictor::new(-1.0);
        assert!(r1.inner().config.regularization > 0.0);
        let r2 = RidgeRegressionPredictor::new(f64::NAN);
        assert!(r2.inner().config.regularization > 0.0);
        let r3 = RidgeRegressionPredictor::new(0.0);
        assert!(r3.inner().config.regularization > 0.0);
    }
}
