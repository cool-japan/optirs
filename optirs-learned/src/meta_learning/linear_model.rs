//! The parametric task model shared by every meta-learner in this module.
//!
//! Meta-learning only means something if all the learners that
//! [`super::framework::MetaLearningFramework`] can dispatch to agree on what the
//! meta-parameters *are*. This module fixes that contract:
//!
//! ```text
//! prediction(x; theta) = (sum_i x_i * w_i) / len(x) + b
//! ```
//!
//! where `w = theta["weights"]` and `b = theta["bias"][0]` when a bias entry is
//! present (it is optional; without it the model is exactly the linear model
//! [`super::reptile_learner::ReptileLearner`] has always used, so the two
//! algorithms remain interchangeable).
//!
//! Losses are mean squared error and every gradient here is **analytic** — no
//! finite differences, no per-coordinate re-evaluation of the whole dataset.

use std::collections::HashMap;
use std::fmt::Debug;

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};

/// Parameter key holding the linear weights.
pub const WEIGHTS_KEY: &str = "weights";

/// Parameter key holding the (optional, length-1) bias.
pub const BIAS_KEY: &str = "bias";

/// Evaluate the shared model on a single feature vector.
///
/// Returns `Err` when `features` is empty (the prediction would be `0/0`).
pub fn predict<T: Float + Debug + Send + Sync + 'static>(
    features: &Array1<T>,
    parameters: &HashMap<String, Array1<T>>,
) -> Result<T> {
    if features.is_empty() {
        return Err(OptimError::InsufficientData(
            "cannot predict from an empty feature vector".to_string(),
        ));
    }
    let feat_len = T::from(features.len()).ok_or_else(|| {
        OptimError::ComputationError("failed to convert feature length".to_string())
    })?;

    let linear = match parameters.get(WEIGHTS_KEY) {
        Some(weights) => {
            let n = features.len().min(weights.len());
            let mut sum = T::zero();
            for i in 0..n {
                sum = sum + features[i] * weights[i];
            }
            sum / feat_len
        }
        None => {
            // No weights supplied: fall back to the unweighted feature mean,
            // which is what a zero-information model predicts.
            features.iter().copied().fold(T::zero(), |a, b| a + b) / feat_len
        }
    };

    let bias = parameters
        .get(BIAS_KEY)
        .and_then(|b| b.first().copied())
        .unwrap_or_else(T::zero);

    Ok(linear + bias)
}

/// Mean squared error of the shared model over a dataset.
///
/// Returns `Err` on an empty dataset instead of producing `0/0 = NaN`.
pub fn mse_loss<T: Float + Debug + Send + Sync + 'static>(
    features: &[Array1<T>],
    targets: &[T],
    parameters: &HashMap<String, Array1<T>>,
) -> Result<T> {
    if features.is_empty() {
        return Err(OptimError::InsufficientData(
            "cannot compute a loss over an empty dataset".to_string(),
        ));
    }
    if features.len() != targets.len() {
        return Err(OptimError::InvalidConfig(format!(
            "dataset has {} feature vectors but {} targets",
            features.len(),
            targets.len()
        )));
    }

    let mut total = T::zero();
    for (feat, target) in features.iter().zip(targets.iter()) {
        let diff = predict(feat, parameters)? - *target;
        total = total + diff * diff;
    }
    let n = T::from(features.len()).ok_or_else(|| {
        OptimError::ComputationError("failed to convert dataset size".to_string())
    })?;
    Ok(total / n)
}

/// Analytic gradient of [`mse_loss`] with respect to every supplied parameter.
///
/// For `p_n = (sum_i x_n[i] w_i)/len(x_n) + b` and `L = (1/N) sum_n (p_n - y_n)^2`:
///
/// ```text
/// dL/dw_i = (2/N) sum_n (p_n - y_n) * x_n[i] / len(x_n)
/// dL/db   = (2/N) sum_n (p_n - y_n)
/// ```
pub fn mse_gradients<T: Float + Debug + Send + Sync + 'static>(
    features: &[Array1<T>],
    targets: &[T],
    parameters: &HashMap<String, Array1<T>>,
) -> Result<HashMap<String, Array1<T>>> {
    if features.is_empty() {
        return Err(OptimError::InsufficientData(
            "cannot compute gradients over an empty dataset".to_string(),
        ));
    }
    if features.len() != targets.len() {
        return Err(OptimError::InvalidConfig(format!(
            "dataset has {} feature vectors but {} targets",
            features.len(),
            targets.len()
        )));
    }

    let n = T::from(features.len()).ok_or_else(|| {
        OptimError::ComputationError("failed to convert dataset size".to_string())
    })?;
    let two = T::one() + T::one();
    let scale = two / n;

    let mut gradients: HashMap<String, Array1<T>> = parameters
        .iter()
        .map(|(name, param)| (name.clone(), Array1::zeros(param.len())))
        .collect();

    for (feat, target) in features.iter().zip(targets.iter()) {
        if feat.is_empty() {
            return Err(OptimError::InsufficientData(
                "dataset contains an empty feature vector".to_string(),
            ));
        }
        let feat_len = T::from(feat.len()).ok_or_else(|| {
            OptimError::ComputationError("failed to convert feature length".to_string())
        })?;
        let residual = predict(feat, parameters)? - *target;

        if let (Some(weights), Some(grad)) =
            (parameters.get(WEIGHTS_KEY), gradients.get_mut(WEIGHTS_KEY))
        {
            let m = feat.len().min(weights.len());
            for i in 0..m {
                grad[i] = grad[i] + scale * residual * feat[i] / feat_len;
            }
        }
        if let Some(grad) = gradients.get_mut(BIAS_KEY) {
            if !grad.is_empty() {
                grad[0] = grad[0] + scale * residual;
            }
        }
    }

    Ok(gradients)
}

/// Hessian-vector product of [`mse_loss`] with respect to the weights (and the
/// optional bias), evaluated exactly.
///
/// The model is linear in its parameters, so the MSE Hessian is a constant
/// matrix `H = (2/N) sum_n phi_n phi_n^T` where `phi_n` is the feature vector
/// scaled by `1/len(x_n)` (with a trailing `1` for the bias). That makes the
/// product exact rather than a finite-difference approximation, and lets the
/// second-order MAML meta-gradient be computed without a tape.
pub fn mse_hessian_vector_product<T: Float + Debug + Send + Sync + 'static>(
    features: &[Array1<T>],
    parameters: &HashMap<String, Array1<T>>,
    vector: &HashMap<String, Array1<T>>,
) -> Result<HashMap<String, Array1<T>>> {
    if features.is_empty() {
        return Err(OptimError::InsufficientData(
            "cannot compute a Hessian-vector product over an empty dataset".to_string(),
        ));
    }
    let n = T::from(features.len()).ok_or_else(|| {
        OptimError::ComputationError("failed to convert dataset size".to_string())
    })?;
    let two = T::one() + T::one();
    let scale = two / n;

    let mut product: HashMap<String, Array1<T>> = parameters
        .iter()
        .map(|(name, param)| (name.clone(), Array1::zeros(param.len())))
        .collect();

    let weight_len = parameters.get(WEIGHTS_KEY).map(|w| w.len()).unwrap_or(0);
    let has_bias = parameters.contains_key(BIAS_KEY);

    for feat in features.iter() {
        if feat.is_empty() {
            return Err(OptimError::InsufficientData(
                "dataset contains an empty feature vector".to_string(),
            ));
        }
        let feat_len = T::from(feat.len()).ok_or_else(|| {
            OptimError::ComputationError("failed to convert feature length".to_string())
        })?;
        let m = feat.len().min(weight_len);

        // phi . v
        let mut dot = T::zero();
        if let Some(v) = vector.get(WEIGHTS_KEY) {
            let limit = m.min(v.len());
            for i in 0..limit {
                dot = dot + (feat[i] / feat_len) * v[i];
            }
        }
        if has_bias {
            if let Some(v) = vector.get(BIAS_KEY) {
                if let Some(&v0) = v.first() {
                    dot = dot + v0;
                }
            }
        }

        if let Some(out) = product.get_mut(WEIGHTS_KEY) {
            let limit = m.min(out.len());
            for i in 0..limit {
                out[i] = out[i] + scale * dot * (feat[i] / feat_len);
            }
        }
        if let Some(out) = product.get_mut(BIAS_KEY) {
            if !out.is_empty() {
                out[0] = out[0] + scale * dot;
            }
        }
    }

    Ok(product)
}

/// Predictions of the shared model over a whole dataset.
pub fn predict_all<T: Float + Debug + Send + Sync + 'static>(
    features: &[Array1<T>],
    parameters: &HashMap<String, Array1<T>>,
) -> Result<Vec<T>> {
    features.iter().map(|f| predict(f, parameters)).collect()
}

/// Coefficient of determination (R^2) of the predictions against the targets.
///
/// Returns `None` when the targets have zero variance, in which case R^2 is
/// undefined rather than "perfect".
pub fn r_squared<T: Float + Debug + Send + Sync + 'static>(
    predictions: &[T],
    targets: &[T],
) -> Option<T> {
    if predictions.len() != targets.len() || targets.is_empty() {
        return None;
    }
    let n = T::from(targets.len())?;
    let mean = targets.iter().copied().fold(T::zero(), |a, b| a + b) / n;
    let mut ss_res = T::zero();
    let mut ss_tot = T::zero();
    for (p, y) in predictions.iter().zip(targets.iter()) {
        let r = *p - *y;
        ss_res = ss_res + r * r;
        let d = *y - mean;
        ss_tot = ss_tot + d * d;
    }
    if ss_tot <= T::zero() {
        return None;
    }
    Some(T::one() - ss_res / ss_tot)
}

/// Fraction of predictions that round to the same integer label as the target.
///
/// Returns `None` for an empty dataset.
pub fn label_accuracy<T: Float + Debug + Send + Sync + 'static>(
    predictions: &[T],
    targets: &[T],
) -> Option<T> {
    if predictions.len() != targets.len() || targets.is_empty() {
        return None;
    }
    let mut correct = 0usize;
    for (p, y) in predictions.iter().zip(targets.iter()) {
        if p.round() == y.round() {
            correct += 1;
        }
    }
    let n = T::from(targets.len())?;
    T::from(correct).map(|c| c / n)
}

/// Full-curve area under the ROC curve for binary labels.
///
/// The model's raw predictions double as the ranking scores, so no separate
/// score-retention channel is needed: `evaluate_query_set` already keeps every
/// prediction in [`super::types::QueryEvaluationResult::predictions`] and hands
/// that same slice here.
///
/// The area is computed exactly, over the *whole* curve, via the rank
/// (Mann-Whitney U) identity
///
/// ```text
/// AUC = (sum of ranks of the positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
/// ```
///
/// Tied scores share their mean rank, which is what makes this identity equal
/// the trapezoidal area of the ROC curve rather than an optimistic step area.
/// It is therefore a true full-curve AUC, not a single-operating-point
/// approximation such as `(tpr + tnr) / 2`.
///
/// Returns `None` — never a fabricated number — when the area is undefined:
///
/// * the lengths disagree or the set is empty,
/// * some label does not round to `0` or `1` (the labels are not binary),
/// * every label is the same class (no positive/negative pair to rank),
/// * some score is NaN (the ranking would be ill-defined).
pub fn roc_auc<T: Float + Debug + Send + Sync + 'static>(scores: &[T], labels: &[T]) -> Option<T> {
    if scores.len() != labels.len() || labels.is_empty() {
        return None;
    }

    // Binary-label check: every target must round to exactly 0 or 1.
    let mut positives = Vec::with_capacity(labels.len());
    for y in labels {
        let r = y.round();
        if r == T::zero() {
            positives.push(false);
        } else if r == T::one() {
            positives.push(true);
        } else {
            return None;
        }
    }
    let n_pos = positives.iter().filter(|p| **p).count();
    let n_neg = positives.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    if scores.iter().any(|s| s.is_nan()) {
        return None;
    }

    // Ascending order by score; NaN was rejected above so the comparison is total.
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| {
        scores[*a]
            .partial_cmp(&scores[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mean rank within each tied block (ranks are 1-based).
    let mut rank_sum_pos = T::zero();
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        // Ranks i+1 ..= j average to (i + 1 + j) / 2.
        let mean_rank = T::from(i + 1 + j)? / T::from(2.0)?;
        for k in order.iter().take(j).skip(i) {
            if positives[*k] {
                rank_sum_pos = rank_sum_pos + mean_rank;
            }
        }
        i = j;
    }

    let n_pos_t = T::from(n_pos)?;
    let n_neg_t = T::from(n_neg)?;
    let two = T::from(2.0)?;
    let min_rank_sum = n_pos_t * (n_pos_t + T::one()) / two;
    Some((rank_sum_pos - min_rank_sum) / (n_pos_t * n_neg_t))
}

/// L2 norm of a parameter/gradient map.
pub fn map_norm<T: Float + Debug + Send + Sync + 'static>(map: &HashMap<String, Array1<T>>) -> T {
    map.values()
        .flat_map(|v| v.iter().copied())
        .fold(T::zero(), |a, b| a + b * b)
        .sqrt()
}

/// Cosine alignment between two parameter/gradient maps.
///
/// Returns `None` when either map has zero norm.
pub fn map_cosine<T: Float + Debug + Send + Sync + 'static>(
    lhs: &HashMap<String, Array1<T>>,
    rhs: &HashMap<String, Array1<T>>,
) -> Option<T> {
    let mut dot = T::zero();
    for (name, a) in lhs {
        if let Some(b) = rhs.get(name) {
            let n = a.len().min(b.len());
            for i in 0..n {
                dot = dot + a[i] * b[i];
            }
        }
    }
    let norm = map_norm(lhs) * map_norm(rhs);
    if norm <= T::zero() {
        None
    } else {
        Some(dot / norm)
    }
}

/// In-place gradient-descent step `theta <- theta - lr * grad`, returning the
/// L2 norm of the applied change.
pub fn descend<T: Float + Debug + Send + Sync + 'static>(
    parameters: &mut HashMap<String, Array1<T>>,
    gradients: &HashMap<String, Array1<T>>,
    lr: T,
) -> Result<T> {
    let mut change_sq = T::zero();
    for (name, param) in parameters.iter_mut() {
        if let Some(grad) = gradients.get(name) {
            if grad.len() != param.len() {
                return Err(OptimError::ComputationError(format!(
                    "gradient for '{name}' has {} elements but the parameter has {}",
                    grad.len(),
                    param.len()
                )));
            }
            for i in 0..param.len() {
                let change = lr * grad[i];
                param[i] = param[i] - change;
                change_sq = change_sq + change * change;
            }
        }
    }
    Ok(change_sq.sqrt())
}

/// Clip a gradient map in place so its global L2 norm does not exceed `max_norm`.
pub fn clip_gradients<T: Float + Debug + Send + Sync + 'static>(
    gradients: &mut HashMap<String, Array1<T>>,
    max_norm: T,
) {
    if max_norm <= T::zero() {
        return;
    }
    let norm = map_norm(gradients);
    if norm <= max_norm || norm <= T::zero() {
        return;
    }
    let scale = max_norm / norm;
    for grad in gradients.values_mut() {
        for g in grad.iter_mut() {
            *g = *g * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(weights: Vec<f64>, bias: Option<f64>) -> HashMap<String, Array1<f64>> {
        let mut p = HashMap::new();
        p.insert(WEIGHTS_KEY.to_string(), Array1::from_vec(weights));
        if let Some(b) = bias {
            p.insert(BIAS_KEY.to_string(), Array1::from_vec(vec![b]));
        }
        p
    }

    #[test]
    fn test_predict_matches_documented_formula() {
        let p = params(vec![2.0, 4.0], Some(0.5));
        let x = Array1::from_vec(vec![1.0, 3.0]);
        // (1*2 + 3*4)/2 + 0.5 = 7.5
        approx::assert_abs_diff_eq!(predict(&x, &p).expect("predict"), 7.5, epsilon = 1e-12);
    }

    #[test]
    fn test_empty_dataset_is_an_error_not_nan() {
        let p = params(vec![1.0], None);
        assert!(mse_loss::<f64>(&[], &[], &p).is_err());
        assert!(mse_gradients::<f64>(&[], &[], &p).is_err());
        assert!(predict(&Array1::from_vec(Vec::<f64>::new()), &p).is_err());
    }

    #[test]
    fn test_analytic_gradients_match_central_differences() {
        let p = params(vec![0.3, -1.2, 0.7], Some(-0.4));
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0, -1.0]),
            Array1::from_vec(vec![0.5, -0.5, 2.0]),
            Array1::from_vec(vec![-1.5, 1.0, 0.25]),
        ];
        let targets = vec![0.8, -0.3, 1.4];

        let analytic = mse_gradients(&features, &targets, &p).expect("gradients");
        let h = 1e-6;
        for key in [WEIGHTS_KEY, BIAS_KEY] {
            let len = p[key].len();
            for i in 0..len {
                let mut plus = p.clone();
                let mut minus = p.clone();
                if let Some(v) = plus.get_mut(key) {
                    v[i] += h;
                }
                if let Some(v) = minus.get_mut(key) {
                    v[i] -= h;
                }
                let fd = (mse_loss(&features, &targets, &plus).expect("l+")
                    - mse_loss(&features, &targets, &minus).expect("l-"))
                    / (2.0 * h);
                approx::assert_abs_diff_eq!(analytic[key][i], fd, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_hessian_vector_product_matches_gradient_difference() {
        let p = params(vec![0.1, -0.6], Some(0.2));
        let features = vec![
            Array1::from_vec(vec![1.0, -2.0]),
            Array1::from_vec(vec![0.4, 0.9]),
        ];
        let targets = vec![0.2, -1.1];

        let mut v = HashMap::new();
        v.insert(WEIGHTS_KEY.to_string(), Array1::from_vec(vec![0.7, -0.3]));
        v.insert(BIAS_KEY.to_string(), Array1::from_vec(vec![0.5]));

        let hvp = mse_hessian_vector_product(&features, &p, &v).expect("hvp");

        // (grad(theta + h v) - grad(theta - h v)) / 2h
        let h = 1e-6;
        let mut plus = p.clone();
        let mut minus = p.clone();
        for (name, vec_part) in &v {
            if let Some(a) = plus.get_mut(name) {
                for i in 0..a.len() {
                    a[i] += h * vec_part[i];
                }
            }
            if let Some(a) = minus.get_mut(name) {
                for i in 0..a.len() {
                    a[i] -= h * vec_part[i];
                }
            }
        }
        let gp = mse_gradients(&features, &targets, &plus).expect("g+");
        let gm = mse_gradients(&features, &targets, &minus).expect("g-");
        for name in [WEIGHTS_KEY, BIAS_KEY] {
            for i in 0..hvp[name].len() {
                let fd = (gp[name][i] - gm[name][i]) / (2.0 * h);
                approx::assert_abs_diff_eq!(hvp[name][i], fd, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_descend_and_clip() {
        let mut p = params(vec![1.0, 1.0], None);
        let mut g = HashMap::new();
        g.insert(WEIGHTS_KEY.to_string(), Array1::from_vec(vec![3.0, 4.0]));
        clip_gradients(&mut g, 1.0);
        approx::assert_abs_diff_eq!(map_norm(&g), 1.0, epsilon = 1e-12);
        let change = descend(&mut p, &g, 0.5).expect("descend");
        approx::assert_abs_diff_eq!(change, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_r_squared_and_accuracy_edge_cases() {
        assert!(r_squared::<f64>(&[1.0, 1.0], &[2.0, 2.0]).is_none());
        let acc = label_accuracy::<f64>(&[0.9, 2.1, 0.2], &[1.0, 2.0, 1.0]).expect("acc");
        approx::assert_abs_diff_eq!(acc, 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_roc_auc_perfect_and_inverted_ranking() {
        // Every positive outranks every negative -> area 1.
        let scores = [0.1, 0.2, 0.8, 0.9];
        let labels = [0.0, 0.0, 1.0, 1.0];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&scores, &labels).expect("binary labels give an AUC"),
            1.0,
            epsilon = 1e-12
        );
        // Reversing the scores mirrors the curve -> area 0.
        let inverted = [0.9, 0.8, 0.2, 0.1];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&inverted, &labels).expect("binary labels give an AUC"),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_roc_auc_uses_full_curve_not_one_operating_point() {
        // 2 positives (0.3, 0.6) and 2 negatives (0.1, 0.4) give 4 pairs. Only
        // (0.3 vs 0.4) is mis-ordered, so the area is 3/4.
        let labels = [0.0, 0.0, 1.0, 1.0];
        let scores = [0.1, 0.4, 0.3, 0.6];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&scores, &labels).expect("binary labels give an AUC"),
            0.75,
            epsilon = 1e-12
        );
        // Now only (0.6 vs 0.5) survives, so the area drops to 1/4 even though a
        // single-threshold summary at 0.55 would score both sets identically.
        let scores_worse = [0.5, 0.7, 0.3, 0.6];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&scores_worse, &labels).expect("binary labels give an AUC"),
            0.25,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_roc_auc_ties_take_mean_rank() {
        // All scores tied: every pair is a coin flip -> exactly 0.5.
        let scores = [0.5, 0.5, 0.5, 0.5];
        let labels = [0.0, 1.0, 0.0, 1.0];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&scores, &labels).expect("binary labels give an AUC"),
            0.5,
            epsilon = 1e-12
        );
        // One tied pair between an otherwise perfect split: 3 clean pairs + one
        // half-credit tie out of 4 -> 0.875.
        let partial = [0.1, 0.5, 0.5, 0.9];
        approx::assert_abs_diff_eq!(
            roc_auc::<f64>(&partial, &labels).expect("binary labels give an AUC"),
            0.875,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_roc_auc_returns_none_when_undefined() {
        // Non-binary labels (regression targets).
        assert!(roc_auc::<f64>(&[0.1, 0.2, 0.3], &[2.0, 5.0, 8.0]).is_none());
        // Single class: no positive/negative pair exists.
        assert!(roc_auc::<f64>(&[0.1, 0.2], &[1.0, 1.0]).is_none());
        assert!(roc_auc::<f64>(&[0.1, 0.2], &[0.0, 0.0]).is_none());
        // Empty and mismatched inputs.
        assert!(roc_auc::<f64>(&[], &[]).is_none());
        assert!(roc_auc::<f64>(&[0.1], &[0.0, 1.0]).is_none());
        // NaN scores make the ranking ill-defined.
        assert!(roc_auc::<f64>(&[f64::NAN, 0.2], &[0.0, 1.0]).is_none());
    }
}
