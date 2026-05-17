//! Numerical primitives used by the AutoML pipeline coordinator.
//!
//! This module groups the *pure* helpers that operate on raw tensors:
//! preprocessing scalers, feature-engineering expansions, train-/-val
//! fit-and-apply helpers, ensembling reducers, sampling helpers, and a
//! handful of misc utilities. None of them depend on coordinator state,
//! so they can be unit-tested and reused independently.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, Rng};

use super::{EnsembleStrategy, FeatureEngineeringStep, PreprocessingStep};

/// Epsilon used in `Reciprocal` feature engineering to avoid division by
/// zero.
pub(super) const RECIPROCAL_EPSILON: f64 = 1e-6;

/// Lower bound for any variance / stddev appearing in a denominator.
pub(super) const VARIANCE_FLOOR: f64 = 1e-12;

// ============================================================================
// Preprocessing primitives.
// ============================================================================

pub(super) fn standard_scaler(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows() as f64;
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let col = x.column(j);
        let mean: f64 = col.iter().sum::<f64>() / n;
        let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let denom = var.sqrt().max(VARIANCE_FLOOR);
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - mean) / denom;
        }
    }
    out
}

pub(super) fn min_max_scaler(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let col = x.column(j);
        let min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom = (max - min).max(VARIANCE_FLOOR);
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - min) / denom;
        }
    }
    out
}

pub(super) fn robust_scaler(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let col: Vec<f64> = x.column(j).iter().copied().collect();
        let median = percentile(&col, 50.0);
        let q25 = percentile(&col, 25.0);
        let q75 = percentile(&col, 75.0);
        let iqr = (q75 - q25).max(VARIANCE_FLOOR);
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - median) / iqr;
        }
    }
    out
}

pub(super) fn impute_mean(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let col = x.column(j);
        let (sum, count) = col.iter().fold((0.0_f64, 0_usize), |(s, c), v| {
            if v.is_finite() {
                (s + *v, c + 1)
            } else {
                (s, c)
            }
        });
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        for i in 0..x.nrows() {
            if !x[[i, j]].is_finite() {
                out[[i, j]] = mean;
            }
        }
    }
    out
}

pub(super) fn impute_median(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let finite: Vec<f64> = x
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let median = if finite.is_empty() {
            0.0
        } else {
            percentile(&finite, 50.0)
        };
        for i in 0..x.nrows() {
            if !x[[i, j]].is_finite() {
                out[[i, j]] = median;
            }
        }
    }
    out
}

/// Compute the `p`-th percentile (`p ∈ [0, 100]`) of `values` using
/// linear interpolation between observed quantiles. Non-finite entries
/// are filtered out before sorting.
pub(super) fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * ((n - 1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

// ============================================================================
// Feature engineering primitives.
// ============================================================================

/// Append per-element-transformed columns to `x`. Output has
/// `2 * x.ncols()` columns: the original features come first, followed
/// by the transformed ones in the same column order.
pub(super) fn append_per_feature<F: Fn(f64) -> f64>(x: &Array2<f64>, f: F) -> Array2<f64> {
    let n_rows = x.nrows();
    let n_cols = x.ncols();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols * 2));
    for i in 0..n_rows {
        for j in 0..n_cols {
            out[[i, j]] = x[[i, j]];
            out[[i, n_cols + j]] = f(x[[i, j]]);
        }
    }
    out
}

/// Polynomial-degree-2 expansion: returns a matrix with
/// `n_cols + n_cols * (n_cols + 1) / 2` columns. Original features come
/// first, followed by all `x_i * x_j` for `i <= j` (lex order on
/// `(i, j)`).
pub(super) fn polynomial_degree_2(x: &Array2<f64>) -> Array2<f64> {
    let n_rows = x.nrows();
    let n_cols = x.ncols();
    let n_poly = n_cols * (n_cols + 1) / 2;
    let mut out = Array2::<f64>::zeros((n_rows, n_cols + n_poly));
    for i in 0..n_rows {
        for j in 0..n_cols {
            out[[i, j]] = x[[i, j]];
        }
        let mut k = n_cols;
        for a in 0..n_cols {
            for b in a..n_cols {
                out[[i, k]] = x[[i, a]] * x[[i, b]];
                k += 1;
            }
        }
    }
    out
}

/// Append pairwise products of the top-`k` features ranked by absolute
/// Pearson correlation with `y`. The output preserves the original
/// features and appends `k * (k - 1) / 2` product columns (pairs with
/// `i < j`, *excluding* squares which are already covered by the
/// polynomial path).
pub(super) fn pairwise_top_k_products(
    x: &Array2<f64>,
    y: &Array1<f64>,
    top_k: usize,
) -> Array2<f64> {
    let n_rows = x.nrows();
    let n_cols = x.ncols();
    let k = top_k.min(n_cols);
    if k == 0 {
        return x.clone();
    }
    let (top_indices, _) = top_k_correlated_indices(x, y, k);
    append_pairwise_products(x, &top_indices)
}

// ============================================================================
// Train / val fit-and-apply helpers.
// ============================================================================

pub(super) fn apply_preprocessing_train_val(
    step: PreprocessingStep,
    x_train: &Array2<f64>,
    x_val: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if x_train.ncols() != x_val.ncols() {
        return Err(OptimError::InvalidParameter(format!(
            "train ({}) and val ({}) column counts differ",
            x_train.ncols(),
            x_val.ncols()
        )));
    }
    match step {
        PreprocessingStep::Identity => Ok((x_train.clone(), x_val.clone())),
        PreprocessingStep::StandardScaler => {
            let (means, stds) = fit_mean_std(x_train);
            Ok((
                apply_mean_std(x_train, &means, &stds),
                apply_mean_std(x_val, &means, &stds),
            ))
        }
        PreprocessingStep::MinMaxScaler => {
            let (mins, ranges) = fit_min_range(x_train);
            Ok((
                apply_min_range(x_train, &mins, &ranges),
                apply_min_range(x_val, &mins, &ranges),
            ))
        }
        PreprocessingStep::RobustScaler => {
            let (medians, iqrs) = fit_median_iqr(x_train);
            Ok((
                apply_median_iqr(x_train, &medians, &iqrs),
                apply_median_iqr(x_val, &medians, &iqrs),
            ))
        }
        PreprocessingStep::ImputeMean => {
            let means = fit_finite_mean(x_train);
            Ok((apply_impute(x_train, &means), apply_impute(x_val, &means)))
        }
        PreprocessingStep::ImputeMedian => {
            let medians = fit_finite_median(x_train);
            Ok((
                apply_impute(x_train, &medians),
                apply_impute(x_val, &medians),
            ))
        }
    }
}

pub(super) fn apply_feature_engineering_train_val(
    step: FeatureEngineeringStep,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    x_val: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    match step {
        FeatureEngineeringStep::Identity => Ok((x_train.clone(), x_val.clone())),
        FeatureEngineeringStep::PolynomialDegree2 => {
            Ok((polynomial_degree_2(x_train), polynomial_degree_2(x_val)))
        }
        FeatureEngineeringStep::LogTransform => Ok((
            append_per_feature(x_train, |v| (v.abs() + 1.0).ln()),
            append_per_feature(x_val, |v| (v.abs() + 1.0).ln()),
        )),
        FeatureEngineeringStep::SqrtAbsTransform => {
            let f = |v: f64| {
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                sign * v.abs().sqrt()
            };
            Ok((append_per_feature(x_train, f), append_per_feature(x_val, f)))
        }
        FeatureEngineeringStep::Reciprocal => {
            let f = |v: f64| 1.0 / (v.abs() + RECIPROCAL_EPSILON);
            Ok((append_per_feature(x_train, f), append_per_feature(x_val, f)))
        }
        FeatureEngineeringStep::PairwiseProducts { top_k } => {
            // Fit the top-k ranking on the training matrix only.
            let (top_indices, _) = top_k_correlated_indices(x_train, y_train, top_k as usize);
            Ok((
                append_pairwise_products(x_train, &top_indices),
                append_pairwise_products(x_val, &top_indices),
            ))
        }
    }
}

pub(super) fn fit_mean_std(x: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = x.nrows() as f64;
    let mut means = Vec::with_capacity(x.ncols());
    let mut stds = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        let col = x.column(j);
        let mean: f64 = col.iter().sum::<f64>() / n;
        let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        means.push(mean);
        stds.push(var.sqrt().max(VARIANCE_FLOOR));
    }
    (means, stds)
}

pub(super) fn apply_mean_std(x: &Array2<f64>, means: &[f64], stds: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
        }
    }
    out
}

pub(super) fn fit_min_range(x: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut mins = Vec::with_capacity(x.ncols());
    let mut ranges = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        let col = x.column(j);
        let min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        mins.push(min);
        ranges.push((max - min).max(VARIANCE_FLOOR));
    }
    (mins, ranges)
}

pub(super) fn apply_min_range(x: &Array2<f64>, mins: &[f64], ranges: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - mins[j]) / ranges[j];
        }
    }
    out
}

pub(super) fn fit_median_iqr(x: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut medians = Vec::with_capacity(x.ncols());
    let mut iqrs = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        let col: Vec<f64> = x.column(j).iter().copied().collect();
        let median = percentile(&col, 50.0);
        let q25 = percentile(&col, 25.0);
        let q75 = percentile(&col, 75.0);
        medians.push(median);
        iqrs.push((q75 - q25).max(VARIANCE_FLOOR));
    }
    (medians, iqrs)
}

pub(super) fn apply_median_iqr(x: &Array2<f64>, medians: &[f64], iqrs: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            out[[i, j]] = (x[[i, j]] - medians[j]) / iqrs[j];
        }
    }
    out
}

pub(super) fn fit_finite_mean(x: &Array2<f64>) -> Vec<f64> {
    let mut means = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        let col = x.column(j);
        let (sum, count) = col.iter().fold((0.0_f64, 0_usize), |(s, c), v| {
            if v.is_finite() {
                (s + *v, c + 1)
            } else {
                (s, c)
            }
        });
        means.push(if count > 0 { sum / count as f64 } else { 0.0 });
    }
    means
}

pub(super) fn fit_finite_median(x: &Array2<f64>) -> Vec<f64> {
    let mut medians = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        let finite: Vec<f64> = x
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        medians.push(if finite.is_empty() {
            0.0
        } else {
            percentile(&finite, 50.0)
        });
    }
    medians
}

pub(super) fn apply_impute(x: &Array2<f64>, fills: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[[i, j]].is_finite() {
                out[[i, j]] = fills[j];
            }
        }
    }
    out
}

pub(super) fn top_k_correlated_indices(
    x: &Array2<f64>,
    y: &Array1<f64>,
    top_k: usize,
) -> (Vec<usize>, Vec<f64>) {
    let n_rows = x.nrows();
    let n_cols = x.ncols();
    let k = top_k.min(n_cols);
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let y_var: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / y.len() as f64;
    let y_std = y_var.sqrt().max(VARIANCE_FLOOR);

    let mut scored: Vec<(usize, f64)> = (0..n_cols)
        .map(|j| {
            let col = x.column(j);
            let mean = col.iter().sum::<f64>() / n_rows as f64;
            let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_rows as f64;
            let std = var.sqrt().max(VARIANCE_FLOOR);
            let cov: f64 = col
                .iter()
                .zip(y.iter())
                .map(|(xi, yi)| (xi - mean) * (yi - y_mean))
                .sum::<f64>()
                / n_rows as f64;
            let corr = cov / (std * y_std);
            (j, corr.abs())
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let indices: Vec<usize> = scored.iter().take(k).map(|(j, _)| *j).collect();
    let scores: Vec<f64> = scored.iter().take(k).map(|(_, s)| *s).collect();
    (indices, scores)
}

pub(super) fn append_pairwise_products(x: &Array2<f64>, top_indices: &[usize]) -> Array2<f64> {
    let n_rows = x.nrows();
    let n_cols = x.ncols();
    let k = top_indices.len();
    let n_pairs = if k > 1 { k * (k - 1) / 2 } else { 0 };
    let mut out = Array2::<f64>::zeros((n_rows, n_cols + n_pairs));
    for i in 0..n_rows {
        for j in 0..n_cols {
            out[[i, j]] = x[[i, j]];
        }
        let mut col_ptr = n_cols;
        for a in 0..k {
            for b in (a + 1)..k {
                let ja = top_indices[a];
                let jb = top_indices[b];
                out[[i, col_ptr]] = x[[i, ja]] * x[[i, jb]];
                col_ptr += 1;
            }
        }
    }
    out
}

// ============================================================================
// Ensembling
// ============================================================================

pub(super) fn aggregate_losses(losses: &[f64], strategy: EnsembleStrategy) -> f64 {
    if losses.is_empty() {
        return f64::INFINITY;
    }
    match strategy {
        EnsembleStrategy::BestOnly => losses.iter().copied().fold(f64::INFINITY, f64::min),
        EnsembleStrategy::Average => losses.iter().sum::<f64>() / losses.len() as f64,
        EnsembleStrategy::Weighted => {
            // Inverse-loss weighting. Convert losses to non-negative
            // weights `w_i = 1 / (loss_i + eps)`, normalise, then take
            // the *weighted average loss*. This produces a result in
            // the same minimisation-oriented convention.
            let eps = 1e-9_f64;
            let raw_weights: Vec<f64> = losses.iter().map(|l| 1.0 / (l + eps)).collect();
            let sum_w: f64 = raw_weights.iter().sum();
            if sum_w <= 0.0 || !sum_w.is_finite() {
                return losses.iter().sum::<f64>() / losses.len() as f64;
            }
            losses
                .iter()
                .zip(raw_weights.iter())
                .map(|(l, w)| l * (w / sum_w))
                .sum()
        }
        EnsembleStrategy::Stacking => {
            // Simplified: full stacking would require training a meta
            // learner on the per-candidate predictions, which lives in
            // the user's callback. Fall back to the mean here.
            losses.iter().sum::<f64>() / losses.len() as f64
        }
        EnsembleStrategy::Median => {
            let mut sorted = losses.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
                0.5 * (sorted[mid - 1] + sorted[mid])
            } else {
                sorted[mid]
            }
        }
    }
}

// ============================================================================
// Sampling helpers
// ============================================================================

/// Sample up to `max_take` distinct elements from `pool` without
/// replacement. The number of samples drawn is uniform in
/// `[1, min(max_take, pool.len())]`.
pub(super) fn sample_distinct<R: Rng, T: Clone>(
    rng: &mut Random<R>,
    pool: &[T],
    max_take: usize,
) -> Result<Vec<T>> {
    if pool.is_empty() {
        return Err(OptimError::SearchSpaceError(
            "cannot sample from an empty pool".to_string(),
        ));
    }
    let upper = max_take.min(pool.len()).max(1);
    let n_take: usize = rng.gen_range(1..=upper);
    let mut indices: Vec<usize> = (0..pool.len()).collect();
    // Fisher-Yates partial shuffle: we only need the first `n_take`
    // elements to be uniformly sampled.
    for i in 0..n_take {
        let j: usize = rng.gen_range(i..pool.len());
        indices.swap(i, j);
    }
    Ok(indices
        .into_iter()
        .take(n_take)
        .map(|i| pool[i].clone())
        .collect())
}

pub(super) fn pick_ensemble<R: Rng>(rng: &mut Random<R>) -> EnsembleStrategy {
    const OPTIONS: [EnsembleStrategy; 5] = [
        EnsembleStrategy::Average,
        EnsembleStrategy::Weighted,
        EnsembleStrategy::Median,
        EnsembleStrategy::BestOnly,
        EnsembleStrategy::Stacking,
    ];
    let idx: usize = rng.gen_range(0..OPTIONS.len());
    OPTIONS[idx]
}

// ============================================================================
// Misc
// ============================================================================

pub(super) fn instant_to_ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub(super) fn compute_variance(y: &Array1<f64>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64
}
