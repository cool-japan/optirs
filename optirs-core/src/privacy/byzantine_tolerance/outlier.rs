//! Outlier detection

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::fmt::Debug;

use super::aggregator::ByzantineTolerantAggregator;
use super::helpers::{
    average_path_length, euclidean_distance, from_scalar, isolation_path, to_f64_vec,
    ISOLATION_TREES,
};
use super::types::{OutlierDetectionMethod, OutlierScore, SplitMix64, StatisticalMeasures};

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// Outlier score in `[0, 1]` for `cohort[index]` under the configured method.
    ///
    /// The match is exhaustive on purpose: adding a variant to
    /// [`OutlierDetectionMethod`] must be a compile error rather than fall into a
    /// wildcard arm.
    pub(super) fn compute_outlier_score(
        &self,
        index: usize,
        cohort: &[&Array1<T>],
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let gradient = cohort.get(index).ok_or_else(|| {
            OptimError::InvalidState(format!("outlier index {index} is outside the cohort"))
        })?;
        match self.config.outlier_detection {
            OutlierDetectionMethod::ZScore => Self::zscore_outlier_score(gradient, stats),
            OutlierDetectionMethod::IQR => Self::iqr_outlier_score(gradient, stats),
            OutlierDetectionMethod::MahalanobisDistance => {
                Self::mahalanobis_outlier_score(gradient, stats)
            }
            OutlierDetectionMethod::LocalOutlierFactor => Self::lof_outlier_score(index, cohort),
            OutlierDetectionMethod::IsolationForest => {
                Self::isolation_forest_outlier_score(index, cohort)
            }
        }
    }
    /// Maximum per-coordinate z-score, normalised by the three-sigma rule.
    pub(super) fn zscore_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut max_z_score = 0.0f64;
        for i in 0..gradient.len().min(stats.std_dev.len()) {
            if stats.std_dev[i] > T::zero() {
                let z_score = ((gradient[i] - stats.mean[i]) / stats.std_dev[i]).abs();
                let z_score = from_scalar(z_score)?;
                if z_score > max_z_score {
                    max_z_score = z_score;
                }
            }
        }
        Ok(OutlierScore {
            score: (max_z_score / 3.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::ZScore,
            details: format!("Max Z-score: {max_z_score:.4}"),
        })
    }
    /// Interquartile-range score.
    ///
    /// A coordinate on which at least three quarters of the cohort agree has a
    /// zero IQR; instead of dividing by it (which produced `+inf`), such a
    /// coordinate falls back to the absolute deviation from the mean normalised by
    /// the coordinate's standard deviation, and contributes nothing at all when the
    /// coordinate is constant across the cohort.
    pub(super) fn iqr_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut max_score = 0.0f64;
        let mut degenerate = 0usize;
        for i in 0..gradient.len().min(stats.iqr.len()) {
            let value = gradient[i];
            let iqr = stats.iqr[i];
            let score = if iqr > T::zero() {
                let excess = if value < stats.q1[i] {
                    stats.q1[i] - value
                } else if value > stats.q3[i] {
                    value - stats.q3[i]
                } else {
                    T::zero()
                };
                from_scalar(excess / iqr)? / 1.5
            } else {
                degenerate += 1;
                if stats.std_dev[i] > T::zero() {
                    from_scalar((value - stats.mean[i]).abs() / stats.std_dev[i])? / 3.0
                } else {
                    0.0
                }
            };
            if score > max_score {
                max_score = score;
            }
        }
        Ok(OutlierScore {
            score: max_score.clamp(0.0, 1.0),
            method: OutlierDetectionMethod::IQR,
            details: format!(
                "Max IQR score: {max_score:.4} ({degenerate} degenerate coordinate(s))"
            ),
        })
    }
    /// Diagonal-covariance Mahalanobis distance, reported as the root-mean-square
    /// z-score and normalised by the three-sigma rule.
    pub(super) fn mahalanobis_outlier_score(
        gradient: &Array1<T>,
        stats: &StatisticalMeasures<T>,
    ) -> Result<OutlierScore> {
        let mut accumulated = 0.0f64;
        let mut used = 0usize;
        for i in 0..gradient.len().min(stats.std_dev.len()) {
            if stats.std_dev[i] > T::zero() {
                let z = from_scalar((gradient[i] - stats.mean[i]) / stats.std_dev[i])?;
                accumulated += z * z;
                used += 1;
            }
        }
        let distance = if used == 0 {
            0.0
        } else {
            (accumulated / used as f64).sqrt()
        };
        Ok(OutlierScore {
            score: (distance / 3.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::MahalanobisDistance,
            details: format!(
                "Diagonal Mahalanobis (RMS z) distance: {distance:.4} over {used} coordinate(s)"
            ),
        })
    }
    /// Local outlier factor (Breunig et al., 2000) over the current cohort.
    pub(super) fn lof_outlier_score(index: usize, cohort: &[&Array1<T>]) -> Result<OutlierScore> {
        let n = cohort.len();
        if n < 3 {
            return Ok(OutlierScore {
                score: 0.0,
                method: OutlierDetectionMethod::LocalOutlierFactor,
                details: format!("cohort of {n} is too small for a local outlier factor"),
            });
        }
        let k = ((n - 1) / 2).clamp(1, 20);
        let mut distance = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = from_scalar(euclidean_distance(cohort[i], cohort[j])?)?;
                distance[i][j] = d;
                distance[j][i] = d;
            }
        }
        let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut k_distance = vec![0.0f64; n];
        for i in 0..n {
            let mut others: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance[i][j]))
                .collect();
            others.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            k_distance[i] = others[k - 1].1;
            neighbours.push(others.into_iter().take(k).map(|(j, _)| j).collect());
        }
        let mut lrd = vec![0.0f64; n];
        for i in 0..n {
            let sum: f64 = neighbours[i]
                .iter()
                .map(|&j| k_distance[j].max(distance[i][j]))
                .sum();
            let mean = sum / neighbours[i].len() as f64;
            lrd[i] = 1.0 / mean.max(f64::EPSILON);
        }
        let neighbourhood = &neighbours[index];
        let lof = neighbourhood.iter().map(|&j| lrd[j]).sum::<f64>()
            / (neighbourhood.len() as f64 * lrd[index]);
        let lof = if lof.is_finite() { lof } else { 1.0 };
        Ok(OutlierScore {
            score: (lof - 1.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::LocalOutlierFactor,
            details: format!("Local outlier factor: {lof:.4} (k = {k})"),
        })
    }
    /// Isolation forest (Liu et al., 2008) over the current cohort.
    ///
    /// The splits are drawn from a deterministically seeded generator, so repeated
    /// evaluation of the same cohort yields the same score.
    pub(super) fn isolation_forest_outlier_score(
        index: usize,
        cohort: &[&Array1<T>],
    ) -> Result<OutlierScore> {
        let n = cohort.len();
        if n < 3 {
            return Ok(OutlierScore {
                score: 0.0,
                method: OutlierDetectionMethod::IsolationForest,
                details: format!("cohort of {n} is too small for an isolation forest"),
            });
        }
        let points: Vec<Vec<f64>> = cohort
            .iter()
            .map(|g| to_f64_vec(g))
            .collect::<Result<Vec<_>>>()?;
        let indices: Vec<usize> = (0..n).collect();
        let depth_limit = ((n as f64).log2().ceil() as usize).max(1);
        let mut totals = vec![0.0f64; n];
        let mut rng = SplitMix64::new(0x15_01A7_10F0_2E57);
        for _ in 0..ISOLATION_TREES {
            isolation_path(&points, &indices, 0, depth_limit, &mut rng, &mut totals);
        }
        let expected = totals[index] / ISOLATION_TREES as f64;
        let normaliser = average_path_length(n);
        let raw = if normaliser > 0.0 {
            2f64.powf(-expected / normaliser)
        } else {
            0.5
        };
        Ok(OutlierScore {
            score: ((raw - 0.5) * 2.0).clamp(0.0, 1.0),
            method: OutlierDetectionMethod::IsolationForest,
            details: format!("Isolation score: {raw:.4}, mean path length: {expected:.4}"),
        })
    }
}
