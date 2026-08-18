//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use scirs2_core::numeric::Float;
use scirs2_core::random::{rngs::StdRng, Random};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::constants::ISOLATION_FOREST_DEFAULT_SEED;
use super::functions::isolation_forest_path_normalizer;
use super::types::{
    AnomalyContext, AnomalyResult, AnomalySeverity, AnomalyType, ClassificationModel,
    FrequencyAnalyzer, SeasonalDecomposition, TimeSeriesAnalyzer,
};

/// Outlier detection using multiple methods
pub struct OutlierDetector<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    /// Seeded RNG driving [`Self::isolation_forest_detection`]'s random
    /// subsampling and split points. Fixed-seeded by default (rather than
    /// system-entropy-seeded) so a given detector instance produces
    /// reproducible anomaly calls across a run -- important for tests and
    /// for not having "is this an anomaly" flicker between two calls with
    /// identical history purely due to RNG draw. `Mutex`-wrapped because
    /// `detect_outlier`/`isolation_forest_detection` take `&self` (the
    /// wider `OutlierDetector` API is shared via `&self`, not `&mut self`).
    pub(super) rng: Mutex<Random<StdRng>>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> OutlierDetector<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            rng: Mutex::new(Random::seed(ISOLATION_FOREST_DEFAULT_SEED)),
            _phantom: PhantomData,
        }
    }
    /// Create a detector whose isolation-forest RNG is seeded explicitly
    /// (primarily for deterministic tests that need a specific draw
    /// sequence rather than just "some fixed seed").
    #[cfg(test)]
    pub(super) fn with_seed(config: AnomalyConfig<T>, seed: u64) -> Self {
        Self {
            config,
            rng: Mutex::new(Random::seed(seed)),
            _phantom: PhantomData,
        }
    }
    pub fn detect_outlier(&self, value: T, history: &VecDeque<(Instant, T)>) -> AnomalyResult<T> {
        let values: Vec<T> = history.iter().map(|(_, v)| *v).collect();
        if values.len() < self.config.min_data_points {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp: Instant::now(),
                context: AnomalyContext {
                    baseline_mean: T::zero(),
                    baseline_std: T::zero(),
                    current_value: value,
                    deviation_magnitude: T::zero(),
                    trend_deviation: T::zero(),
                    pattern_match_score: T::zero(),
                    historical_frequency: T::zero(),
                },
                suggested_actions: vec![],
            };
        }
        let mut method_results = Vec::new();
        for method in &self.config.outlier_methods {
            let result = match method {
                OutlierMethod::ZScore => self.zscore_detection(value, &values),
                OutlierMethod::ModifiedZScore => self.modified_zscore_detection(value, &values),
                OutlierMethod::IQR => self.iqr_detection(value, &values),
                OutlierMethod::Hampel => self.hampel_detection(value, &values),
                OutlierMethod::IsolationForest => self.isolation_forest_detection(value, &values),
                OutlierMethod::LocalOutlierFactor => self.lof_detection(value, &values),
                OutlierMethod::OneClassSVM => self.svm_detection(value, &values),
                OutlierMethod::DBSCAN => self.dbscan_detection(value, &values),
            };
            method_results.push(result);
        }
        self.combine_outlier_results(method_results, value)
    }
    pub(super) fn zscore_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(values.len()).expect("unwrap failed");
        let variance = values
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(values.len()).expect("unwrap failed");
        let std_dev = variance.sqrt();
        if std_dev < T::epsilon() {
            return (false, T::zero(), T::zero());
        }
        let z_score = (value - mean).abs() / std_dev;
        let is_outlier = z_score > self.config.statistical_threshold;
        let confidence = if is_outlier {
            (z_score / self.config.statistical_threshold).min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, z_score)
    }
    pub(super) fn modified_zscore_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_values.len().is_multiple_of(2) {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid])
                / T::from(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[sorted_values.len() / 2]
        };
        let mad = {
            let deviations: Vec<T> = values.iter().map(|&x| (x - median).abs()).collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted_deviations.len().is_multiple_of(2) {
                let mid = sorted_deviations.len() / 2;
                (sorted_deviations[mid - 1] + sorted_deviations[mid])
                    / T::from(2.0).unwrap_or_else(|| T::zero())
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            }
        };
        if mad < T::epsilon() {
            return (false, T::zero(), T::zero());
        }
        let modified_z =
            T::from(0.6745).unwrap_or_else(|| T::zero()) * (value - median).abs() / mad;
        let is_outlier = modified_z > T::from(3.5).unwrap_or_else(|| T::zero());
        let confidence = if is_outlier {
            (modified_z / T::from(3.5).unwrap_or_else(|| T::zero())).min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, modified_z)
    }
    pub(super) fn iqr_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        let lower_bound = q1 - T::from(1.5).unwrap_or_else(|| T::zero()) * iqr;
        let upper_bound = q3 + T::from(1.5).unwrap_or_else(|| T::zero()) * iqr;
        let is_outlier = value < lower_bound || value > upper_bound;
        let score = if value < lower_bound {
            (lower_bound - value) / iqr
        } else if value > upper_bound {
            (value - upper_bound) / iqr
        } else {
            T::zero()
        };
        let confidence = if is_outlier {
            score.min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, score)
    }
    pub(super) fn hampel_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        if values.len() < 3 {
            return (false, T::zero(), T::zero());
        }
        let window_size = 7.min(values.len());
        let recent_values = &values[values.len().saturating_sub(window_size)..];
        let mut sorted_values = recent_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_values.len().is_multiple_of(2) {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid])
                / T::from(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[sorted_values.len() / 2]
        };
        let mad = {
            let deviations: Vec<T> = recent_values.iter().map(|&x| (x - median).abs()).collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted_deviations.len().is_multiple_of(2) {
                let mid = sorted_deviations.len() / 2;
                (sorted_deviations[mid - 1] + sorted_deviations[mid])
                    / T::from(2.0).unwrap_or_else(|| T::zero())
            } else {
                sorted_deviations[sorted_deviations.len() / 2]
            }
        };
        if mad < T::epsilon() {
            return (false, T::zero(), T::zero());
        }
        let hampel_score = (value - median).abs() / mad;
        let threshold = T::from(3.0).unwrap_or_else(|| T::zero());
        let is_outlier = hampel_score > threshold;
        let confidence = if is_outlier {
            (hampel_score / threshold).min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, hampel_score)
    }
    /// Isolation-forest anomaly score for `value` against the reference
    /// `values` window.
    ///
    /// Regression fix (F60): the previous version called a fully
    /// deterministic midpoint-split helper (no randomness anywhere) on the
    /// *full, unsampled* `values` slice for every one of its "10 trees", so
    /// every tree computed the exact identical path length and the
    /// "ensemble" was a single evaluation repeated 10 times. It also
    /// returned the raw, un-normalized `depth / max_depth` ratio and fed
    /// the subsample size in as the depth limit -- both bugs compounded
    /// into an *inverted* detector: a point in a dense cluster needs more
    /// splits to separate (longer path -> higher raw ratio -> flagged),
    /// while a genuine outlier separates in one or two splits (shorter path
    /// -> lower raw ratio -> not flagged).
    ///
    /// This version builds `NUM_TREES` real, independently-randomized
    /// isolation trees -- each over a fresh random subsample (via
    /// `self.rng`, see [`Self::isolation_tree_depth`]) with random split
    /// points -- and combines their average path length via the real
    /// Isolation Forest normalization from Liu, Ting & Zhou (2008):
    /// `s(x, n) = 2^(-E[h(x)] / c(n))`, where `c(n)` is the expected path
    /// length of an unsuccessful binary-search-tree lookup over `n` points
    /// ([`isolation_forest_path_normalizer`]). A short average path
    /// (isolated quickly) now correctly yields a score near 1; a path near
    /// `c(n)` (typical of the bulk of the data) yields a score near 0.5.
    ///
    /// Known limitation shared with the real algorithm (see
    /// `test_isolation_forest_duplicate_heavy_data_is_a_known_soft_spot`):
    /// values with many exact duplicates in `values` can't be axis-split
    /// apart from each other, so their path terminates once their shared
    /// value is isolated rather than once each point is -- this can score
    /// a heavily duplicated in-cluster value close to a genuine outlier's.
    /// Real optimizer metrics essentially never repeat a float value
    /// dozens of times, so this is not expected to matter in practice.
    pub(super) fn isolation_forest_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        if values.len() < 2 {
            return (false, T::zero(), T::zero());
        }
        const NUM_TREES: usize = 50;
        let subsample_size = values.len().clamp(2, 256);
        let max_depth = ((subsample_size as f64).log2().ceil() as usize).max(1);
        let mut rng = self
            .rng
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut total_depth = 0.0_f64;
        for _ in 0..NUM_TREES {
            let mut indices: Vec<usize> = (0..values.len()).collect();
            rng.shuffle(&mut indices);
            let subsample: Vec<T> = indices[..subsample_size]
                .iter()
                .map(|&i| values[i])
                .collect();
            total_depth +=
                Self::isolation_tree_depth(value, &subsample, max_depth, 0, &mut rng) as f64;
        }
        drop(rng);
        let avg_path_length = total_depth / NUM_TREES as f64;
        let c_n = isolation_forest_path_normalizer(subsample_size);
        let score = if c_n > 0.0 {
            2.0_f64.powf(-avg_path_length / c_n)
        } else {
            0.0
        };
        let score_t = T::from(score).unwrap_or_else(T::zero);
        let threshold = T::from(0.6).unwrap_or_else(|| T::zero());
        let is_outlier = score_t > threshold;
        let confidence = if is_outlier { score_t } else { T::zero() };
        (is_outlier, confidence, score_t)
    }
    /// Depth at which `value` becomes isolated within `subset`, splitting
    /// on a uniformly random point between the subset's min and max at
    /// each step (the real isolation-forest splitting rule -- the previous
    /// `isolation_tree_score` always split at the exact midpoint, so
    /// repeated calls were never actually independent trees).
    pub(super) fn isolation_tree_depth(
        value: T,
        subset: &[T],
        max_depth: usize,
        current_depth: usize,
        rng: &mut Random<StdRng>,
    ) -> usize {
        if current_depth >= max_depth || subset.len() <= 1 {
            return current_depth;
        }
        let min_val = subset.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        let max_val = subset.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        if (max_val - min_val).abs() < T::epsilon() {
            return current_depth;
        }
        let min_f = min_val.to_f64().unwrap_or(0.0);
        let max_f = max_val.to_f64().unwrap_or(0.0);
        let split_f = rng.gen_range(min_f..max_f);
        let split_point = T::from(split_f).unwrap_or(min_val);
        let next_subset: Vec<T> = if value <= split_point {
            subset
                .iter()
                .filter(|&&x| x <= split_point)
                .cloned()
                .collect()
        } else {
            subset
                .iter()
                .filter(|&&x| x > split_point)
                .cloned()
                .collect()
        };
        if next_subset.len() >= subset.len() {
            return current_depth + 1;
        }
        Self::isolation_tree_depth(value, &next_subset, max_depth, current_depth + 1, rng)
    }
    /// Approximates the Local Outlier Factor for a 1-D scalar stream:
    /// `k`-distance-based local density of `value` versus the average
    /// local density of its own `k` nearest neighbors, the same ratio a
    /// real (multivariate) LOF computes, just with plain absolute
    /// difference standing in for a general distance metric since there is
    /// only one feature here. This is a real, direct simplification of
    /// LOF (not a different algorithm wearing its name), unlike
    /// [`Self::svm_detection`] below.
    pub(super) fn lof_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let k = 5.min(values.len());
        if k == 0 {
            return (false, T::zero(), T::zero());
        }
        let mut distances: Vec<T> = values.iter().map(|&x| (x - value).abs()).collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_distance = distances[k.min(distances.len() - 1)];
        let neighbors: Vec<T> = values
            .iter()
            .filter(|&&x| (x - value).abs() <= k_distance)
            .cloned()
            .collect();
        if neighbors.is_empty() {
            return (false, T::zero(), T::zero());
        }
        let local_density =
            T::from(neighbors.len()).expect("unwrap failed") / (k_distance + T::epsilon());
        let neighbor_densities: Vec<T> = neighbors
            .iter()
            .map(|&neighbor| {
                let neighbor_distances: Vec<T> =
                    values.iter().map(|&x| (x - neighbor).abs()).collect();
                let mut sorted_neighbor_distances = neighbor_distances;
                sorted_neighbor_distances
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let neighbor_k_distance =
                    sorted_neighbor_distances[k.min(sorted_neighbor_distances.len() - 1)];
                T::from(k).unwrap_or_else(|| T::zero()) / (neighbor_k_distance + T::epsilon())
            })
            .collect();
        let avg_neighbor_density = neighbor_densities.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(neighbor_densities.len()).expect("unwrap failed");
        let lof_score = if local_density > T::epsilon() {
            avg_neighbor_density / local_density
        } else {
            T::one()
        };
        let threshold = T::from(1.5).unwrap_or_else(|| T::zero());
        let is_outlier = lof_score > threshold;
        let confidence = if is_outlier {
            ((lof_score - T::one()) / threshold).min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, lof_score)
    }
    /// Distance-from-centroid outlier score, dispatched under
    /// [`OutlierMethod::OneClassSVM`].
    ///
    /// Honesty note: this is **not** a One-Class SVM. A real one-class SVM
    /// (Scholkopf et al., 2001) solves a quadratic program to find a
    /// maximum-margin hyperplane (or hypersphere, for SVDD) separating the
    /// data from the origin in kernel feature space -- it has no closed
    /// form and needs an actual QP/SMO solver. What this computes instead
    /// is `|value - mean(values)| / mean_absolute_deviation(values)`,
    /// i.e. a mean-absolute-deviation-scaled distance from the centroid: a
    /// legitimate, cheap statistical outlier score in its own right (in
    /// the same family as the z-score/Hampel methods elsewhere in this
    /// file), just not the algorithm its enum variant name promises. It is
    /// kept under the existing [`OutlierMethod::OneClassSVM`] name rather
    /// than introduced as a new variant to avoid an API break; treat the
    /// name as legacy and this comment as the ground truth for what it
    /// actually does.
    pub(super) fn svm_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(values.len()).expect("unwrap failed");
        let distances: Vec<T> = values.iter().map(|&x| (x - mean).abs()).collect();
        let avg_distance = distances.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(distances.len()).expect("unwrap failed");
        let current_distance = (value - mean).abs();
        let score = current_distance / (avg_distance + T::epsilon());
        let threshold = T::from(2.0).unwrap_or_else(|| T::zero());
        let is_outlier = score > threshold;
        let confidence = if is_outlier {
            (score / threshold).min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, score)
    }
    /// Applies DBSCAN's core-point/noise-point test directly to `value`:
    /// flagged as an outlier ("noise", in DBSCAN terms) when fewer than
    /// `min_points` other values fall within `eps` of it. `eps` is
    /// estimated per-call from the pairwise-distance distribution (see
    /// [`Self::compute_eps`]) rather than fixed. This is real DBSCAN
    /// noise-point logic, just evaluated locally for one point instead of
    /// clustering the whole dataset and propagating density-reachability
    /// between core points -- there is only one cluster's worth of
    /// structure to find in a 1-D scalar stream, so cluster propagation
    /// would not change which points end up classified as noise.
    pub(super) fn dbscan_detection(&self, value: T, values: &[T]) -> (bool, T, T) {
        let eps = self.compute_eps(values);
        let min_points = 3;
        let neighbors: Vec<T> = values
            .iter()
            .filter(|&&x| (x - value).abs() <= eps)
            .cloned()
            .collect();
        let is_outlier = neighbors.len() < min_points;
        let score = T::from(min_points).unwrap_or_else(|| T::zero())
            / (T::from(neighbors.len()).expect("unwrap failed") + T::one());
        let confidence = if is_outlier {
            score.min(T::one())
        } else {
            T::zero()
        };
        (is_outlier, confidence, score)
    }
    pub(super) fn compute_eps(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::one();
        }
        let mut distances = Vec::new();
        for i in 0..values.len() {
            for j in i + 1..values.len() {
                distances.push((values[i] - values[j]).abs());
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile_90 = distances[(distances.len() * 9 / 10).min(distances.len() - 1)];
        percentile_90 / T::from(2.0).unwrap_or_else(|| T::zero())
    }
    pub(super) fn combine_outlier_results(
        &self,
        results: Vec<(bool, T, T)>,
        value: T,
    ) -> AnomalyResult<T> {
        if results.is_empty() {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp: Instant::now(),
                context: AnomalyContext {
                    baseline_mean: T::zero(),
                    baseline_std: T::zero(),
                    current_value: value,
                    deviation_magnitude: T::zero(),
                    trend_deviation: T::zero(),
                    pattern_match_score: T::zero(),
                    historical_frequency: T::zero(),
                },
                suggested_actions: vec![],
            };
        }
        let outlier_count = results
            .iter()
            .filter(|(is_outlier, _, _)| *is_outlier)
            .count();
        let total_confidence = results
            .iter()
            .map(|(_, confidence, _)| *confidence)
            .fold(T::zero(), |acc, x| acc + x);
        let total_score = results
            .iter()
            .map(|(_, _, score)| *score)
            .fold(T::zero(), |acc, x| acc + x);
        let avg_confidence = total_confidence / T::from(results.len()).expect("unwrap failed");
        let avg_score = total_score / T::from(results.len()).expect("unwrap failed");
        let is_anomaly = T::from(outlier_count).unwrap_or_else(|| T::zero())
            > T::from(results.len()).expect("unwrap failed")
                / T::from(2.0).unwrap_or_else(|| T::zero());
        let severity = if avg_score > T::from(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if avg_score > T::from(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if avg_score > T::from(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };
        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::StatisticalOutlier,
            severity,
            confidence: avg_confidence,
            anomaly_score: avg_score,
            timestamp: Instant::now(),
            context: AnomalyContext {
                baseline_mean: T::zero(),
                baseline_std: T::zero(),
                current_value: value,
                deviation_magnitude: avg_score,
                trend_deviation: T::zero(),
                pattern_match_score: T::zero(),
                historical_frequency: T::zero(),
            },
            suggested_actions: vec![
                "Review data collection process".to_string(),
                "Check for measurement errors".to_string(),
                "Investigate potential system issues".to_string(),
            ],
        }
    }
}
/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyConfig<T: Float + Debug + Send + Sync + 'static> {
    pub statistical_threshold: T,
    pub trend_sensitivity: T,
    pub pattern_window: usize,
    pub baseline_window: usize,
    pub min_data_points: usize,
    pub confidence_threshold: T,
    pub enable_adaptive_thresholds: bool,
    pub seasonal_analysis: bool,
    pub outlier_methods: Vec<OutlierMethod>,
    pub alert_cooldown: Duration,
}
/// Adaptive threshold management
#[derive(Debug)]
pub(super) struct AdaptiveThresholds<T: Float + Debug + Send + Sync + 'static> {
    pub(super) statistical_threshold: T,
    pub(super) trend_threshold: T,
    pub(super) pattern_threshold: T,
    pub(super) adaptation_rate: T,
    pub(super) false_positive_count: usize,
    pub(super) false_negative_count: usize,
}
impl<T: Float + Debug + Send + Sync + 'static> AdaptiveThresholds<T> {
    pub(super) fn new() -> Self {
        Self {
            statistical_threshold: T::from(2.5).unwrap_or_else(|| T::zero()),
            trend_threshold: T::from(0.1).unwrap_or_else(|| T::zero()),
            pattern_threshold: T::from(0.3).unwrap_or_else(|| T::zero()),
            adaptation_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
            false_positive_count: 0,
            false_negative_count: 0,
        }
    }
    pub(super) fn update(&mut self, _value: T, detected_anomaly: bool) {
        if detected_anomaly {
            self.statistical_threshold =
                self.statistical_threshold * (T::one() + self.adaptation_rate);
        } else {
            self.statistical_threshold = self.statistical_threshold
                * (T::one() - self.adaptation_rate / T::from(2.0).unwrap_or_else(|| T::zero()));
        }
        self.statistical_threshold = self
            .statistical_threshold
            .max(T::from(1.5).unwrap_or_else(|| T::zero()))
            .min(T::from(4.0).unwrap_or_else(|| T::zero()));
    }
    pub(super) fn get_statistical_threshold(&self) -> T {
        self.statistical_threshold
    }
}
/// Available outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierMethod {
    ZScore,
    ModifiedZScore,
    IQR,
    Hampel,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
}
/// Anomaly classification system
pub struct AnomalyClassifier<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) classification_models: HashMap<AnomalyType, ClassificationModel<T>>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> AnomalyClassifier<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        let mut classifier = Self {
            config,
            classification_models: HashMap::new(),
            _phantom: PhantomData,
        };
        classifier.initialize_models();
        classifier
    }
    pub(super) fn initialize_models(&mut self) {
        let anomaly_types = [
            AnomalyType::StatisticalOutlier,
            AnomalyType::TrendAnomaly,
            AnomalyType::PerformanceAnomaly,
            AnomalyType::ConvergenceAnomaly,
            AnomalyType::ResourceAnomaly,
            AnomalyType::PatternAnomaly,
            AnomalyType::SeasonalAnomaly,
            AnomalyType::SystemAnomaly,
        ];
        for anomaly_type in &anomaly_types {
            self.classification_models.insert(
                anomaly_type.clone(),
                ClassificationModel::new(anomaly_type.clone()),
            );
        }
    }
    pub fn classify_anomaly(&self, features: &AnomalyFeatures<T>) -> AnomalyType {
        let mut best_score = T::zero();
        let mut best_type = AnomalyType::StatisticalOutlier;
        for (anomaly_type, model) in &self.classification_models {
            let score = model.classify(features);
            if score > best_score {
                best_score = score;
                best_type = anomaly_type.clone();
            }
        }
        best_type
    }
    pub fn update_model(
        &mut self,
        anomaly_type: &AnomalyType,
        features: &AnomalyFeatures<T>,
        label: bool,
    ) {
        if let Some(model) = self.classification_models.get_mut(anomaly_type) {
            model.update(features, label);
        }
    }
}
/// Advanced anomaly analyzer
pub struct AnomalyAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) time_series_analyzer: TimeSeriesAnalyzer<T>,
    pub(super) frequency_analyzer: FrequencyAnalyzer<T>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> AnomalyAnalyzer<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            time_series_analyzer: TimeSeriesAnalyzer::new(config.clone()),
            frequency_analyzer: FrequencyAnalyzer::new(config.clone()),
            config,
            _phantom: PhantomData,
        }
    }
    pub fn analyze_time_series(&self, data: &[(Instant, T)]) -> Vec<AnomalyResult<T>> {
        self.time_series_analyzer.analyze(data)
    }
    pub fn analyze_frequency_domain(&self, values: &[T]) -> Vec<AnomalyResult<T>> {
        self.frequency_analyzer.analyze(values)
    }
    pub fn seasonal_decomposition(&self, data: &[(Instant, T)]) -> SeasonalDecomposition<T> {
        self.time_series_analyzer.seasonal_decomposition(data)
    }
}
/// Baseline statistics computation
#[derive(Debug)]
pub(super) struct BaselineStats<T: Float + Debug + Send + Sync + 'static> {
    pub(super) mean: T,
    pub(super) std_dev: T,
    pub(super) min: T,
    pub(super) max: T,
    pub(super) median: T,
}
impl<T: Float + Debug + Send + Sync + 'static> BaselineStats<T> {
    pub(super) fn new() -> Self {
        Self {
            mean: T::zero(),
            std_dev: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
            median: T::zero(),
        }
    }
    pub(super) fn update(&mut self, values: &[T]) {
        if values.is_empty() {
            return;
        }
        self.mean = values.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(values.len()).expect("unwrap failed");
        let variance = values
            .iter()
            .map(|&x| (x - self.mean) * (x - self.mean))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(values.len()).expect("unwrap failed");
        self.std_dev = variance.sqrt();
        self.min = values.iter().fold(T::infinity(), |acc, &x| acc.min(x));
        self.max = values.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted_values.len() / 2;
        self.median = if sorted_values.len().is_multiple_of(2) {
            (sorted_values[mid - 1] + sorted_values[mid])
                / T::from(2.0).unwrap_or_else(|| T::zero())
        } else {
            sorted_values[mid]
        };
    }
}
/// Features extracted for anomaly classification
#[derive(Debug, Clone)]
pub struct AnomalyFeatures<T: Float + Debug + Send + Sync + 'static> {
    pub statistical_score: T,
    pub trend_score: T,
    pub pattern_score: T,
    pub volatility: T,
    pub magnitude: T,
    pub frequency_features: Vec<T>,
    pub temporal_features: Vec<T>,
}
