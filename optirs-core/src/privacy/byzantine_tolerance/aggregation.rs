//! Aggregation algorithms: coordinate-wise trimmed mean and median, the Krum
//! family (Krum, Multi-Krum, Bulyan) and geometric median.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use super::aggregator::ByzantineTolerantAggregator;
use super::helpers::{euclidean_distance, to_scalar};

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Coordinate-wise trimmed mean (Yin et al., 2018).
    ///
    /// `trim_per_tail` values are dropped from each tail of every coordinate; the
    /// robustness bound requires `trim_per_tail >= f`, hence the pipeline passes
    /// `config.max_byzantine`. Requires `n > 2 * trim_per_tail` so at least one
    /// value survives - smaller cohorts are rejected instead of silently
    /// aggregating to zero.
    pub(super) fn trimmed_mean_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        trim_per_tail: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = values.len();
        if n <= 2 * trim_per_tail {
            return Err(OptimError::InvalidState(format!(
                "trimmed mean needs more than {} gradients to trim {} from each tail, got {}",
                2 * trim_per_tail,
                trim_per_tail,
                n
            )));
        }
        let dim = values[0].len();
        let mut result = Array1::zeros(dim);
        let kept = n - 2 * trim_per_tail;
        let divisor: T = to_scalar(kept as f64)?;
        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let sum: T = coord_values[trim_per_tail..n - trim_per_tail]
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            result[i] = sum / divisor;
        }
        Ok(result)
    }
    /// Coordinate-wise median aggregation
    pub(super) fn coordinate_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let dim = values[0].len();
        let mut result = Array1::zeros(dim);
        let two: T = to_scalar(2.0)?;
        for i in 0..dim {
            let mut coord_values: Vec<T> = values.iter().map(|g| g[i]).collect();
            coord_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            result[i] = if coord_values.len().is_multiple_of(2) {
                let mid = coord_values.len() / 2;
                (coord_values[mid - 1] + coord_values[mid]) / two
            } else {
                coord_values[coord_values.len() / 2]
            };
        }
        Ok(result)
    }
    /// Krum scores of every gradient in `grads`.
    ///
    /// The score of gradient `i` is the sum of its `n - f - 2` smallest squared-free
    /// Euclidean distances to the other gradients. The window is computed with
    /// saturating arithmetic and clamped to at least one neighbour, so no call can
    /// underflow even if a caller bypasses the cohort-size gate.
    pub(super) fn krum_scores(&self, grads: &[&Array1<T>], f: usize) -> Result<Vec<T>> {
        let n = grads.len();
        let take_count = n.saturating_sub(f).saturating_sub(2).max(1);
        let mut scores = Vec::with_capacity(n);
        for (i, gradient) in grads.iter().enumerate() {
            let mut distances = Vec::with_capacity(n.saturating_sub(1));
            for (j, other) in grads.iter().enumerate() {
                if i != j {
                    distances.push(euclidean_distance(gradient, other)?);
                }
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let score = distances
                .iter()
                .take(take_count)
                .copied()
                .fold(T::zero(), |acc, d| acc + d);
            scores.push(score);
        }
        Ok(scores)
    }
    /// Index of the minimum-score gradient; ties resolve to the lowest index.
    pub(super) fn krum_select(&self, grads: &[&Array1<T>], f: usize) -> Result<usize> {
        let scores = self.krum_scores(grads, f)?;
        let mut best = 0usize;
        for (i, score) in scores.iter().enumerate().skip(1) {
            if *score < scores[best] {
                best = i;
            }
        }
        if scores.is_empty() {
            return Err(OptimError::InvalidState(
                "cannot run Krum on an empty cohort".to_string(),
            ));
        }
        Ok(best)
    }
    /// Reject cohorts smaller than `required` for the named algorithm.
    pub(super) fn require_cohort(
        n: usize,
        required: usize,
        algorithm: &str,
        f: usize,
    ) -> Result<()> {
        if n < required {
            return Err(OptimError::InvalidState(format!(
                "{algorithm} tolerating {f} Byzantine participants requires at least {required} \
                 gradients, got {n}"
            )));
        }
        Ok(())
    }
    /// Krum aggregation (Blanchard et al., 2017): select the single most
    /// representative gradient. Requires `n >= 2f + 3`.
    pub(super) fn krum_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        Self::require_cohort(grads.len(), 2 * f + 3, "Krum", f)?;
        let selected = self.krum_select(&grads, f)?;
        Ok(grads[selected].clone())
    }
    /// Multi-Krum aggregation: average the `n - f` gradients with the lowest Krum
    /// scores. Requires `n >= 2f + 3`.
    pub(super) fn multi_krum_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        Self::require_cohort(n, 2 * f + 3, "Multi-Krum", f)?;
        let k = n.saturating_sub(f).max(1);
        let scores = self.krum_scores(&grads, f)?;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut result: Array1<T> = Array1::zeros(grads[0].len());
        for &index in indices.iter().take(k) {
            result = result + grads[index];
        }
        let divisor: T = to_scalar(k as f64)?;
        Ok(result / divisor)
    }
    /// Bulyan aggregation (El Mhamdi et al., 2018).
    ///
    /// Stage one runs Krum `theta = n - 2f` times, removing the selected gradient
    /// each time. Stage two averages, per coordinate, the `beta = theta - 2f`
    /// values closest to the median of the selection. Requires `n >= 4f + 3`.
    pub(super) fn bulyan_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        f: usize,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        Self::require_cohort(n, 4 * f + 3, "Bulyan", f)?;
        let theta = n - 2 * f;
        let beta = theta - 2 * f;
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut selection: Vec<usize> = Vec::with_capacity(theta);
        for _ in 0..theta {
            if remaining.is_empty() {
                break;
            }
            let subset: Vec<&Array1<T>> = remaining.iter().map(|&i| grads[i]).collect();
            let local = self.krum_select(&subset, f)?;
            selection.push(remaining[local]);
            remaining.remove(local);
        }
        if selection.len() < beta || beta == 0 {
            return Err(OptimError::InvalidState(format!(
                "Bulyan selected {} gradients but needs at least {} for the median stage",
                selection.len(),
                beta.max(1)
            )));
        }
        let dim = grads[0].len();
        let mut result = Array1::zeros(dim);
        let two: T = to_scalar(2.0)?;
        let divisor: T = to_scalar(beta as f64)?;
        for c in 0..dim {
            let mut coord: Vec<T> = selection.iter().map(|&i| grads[i][c]).collect();
            let mut sorted = coord.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = if sorted.len().is_multiple_of(2) {
                let mid = sorted.len() / 2;
                (sorted[mid - 1] + sorted[mid]) / two
            } else {
                sorted[sorted.len() / 2]
            };
            coord.sort_by(|a, b| {
                (*a - median)
                    .abs()
                    .partial_cmp(&(*b - median).abs())
                    .unwrap_or(Ordering::Equal)
            });
            let sum = coord
                .iter()
                .take(beta)
                .copied()
                .fold(T::zero(), |acc, x| acc + x);
            result[c] = sum / divisor;
        }
        Ok(result)
    }
    /// Simple median aggregation
    pub(super) fn median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        self.coordinate_median_aggregation(gradients)
    }
    /// Geometric median aggregation via Weiszfeld's algorithm.
    pub(super) fn geometric_median_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let values: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let mut current = values[0].clone();
        let tolerance: T = to_scalar(1e-6)?;
        for _ in 0..100 {
            let mut numerator: Array1<T> = Array1::zeros(current.len());
            let mut denominator = T::zero();
            for &gradient in &values {
                let distance = euclidean_distance(&current, gradient)?;
                if distance > T::zero() {
                    let weight = T::one() / distance;
                    numerator = numerator + gradient * weight;
                    denominator = denominator + weight;
                }
            }
            if denominator <= T::zero() {
                break;
            }
            let new_estimate = numerator / denominator;
            let change = euclidean_distance(&current, &new_estimate)?;
            current = new_estimate;
            if change < tolerance {
                break;
            }
        }
        Ok(current)
    }
}
