// Anomaly Detection for Performance Metrics
//
// This module implements a library of statistical and ML-based anomaly
// detectors aimed at detecting outlier benchmark runs in streams of optimizer
// timings, memory footprints, gradient norms, and similar scalar metrics.
//
// Four detector flavors are exposed via the `AnomalyDetector` trait:
//
// 1. `ZScoreDetector`         - classic standard-score detector (mean / std).
// 2. `IqrDetector`            - Tukey's interquartile-range fence detector.
// 3. `ModifiedZScoreDetector` - robust MAD-based detector (resistant to high
//                               outlier contamination in the training data).
// 4. `IsolationForestDetector` - simplified one-dimensional Isolation Forest
//                                that scores points by the average depth at
//                                which they get isolated across random trees.
//
// All four respect the workspace policies: snake_case naming, scirs2_core only,
// no `.unwrap()` in production code, and a single file kept under 2000 lines.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// -----------------------------------------------------------------------------
// Public data types
// -----------------------------------------------------------------------------

/// Severity level of an anomaly.
///
/// Higher variants correspond to more extreme deviations from the fitted
/// distribution. The exact mapping from detector score to severity is governed
/// by [`AnomalyDetectorConfig`] thresholds and the
/// [`classify_severity`] helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// The point falls inside the normal region of the distribution.
    Normal,
    /// Mild outlier (e.g. 2 sigma for Z-score, beyond 1.5 IQR fences).
    Mild,
    /// Severe outlier (e.g. 3 sigma for Z-score).
    Severe,
    /// Extreme outlier (e.g. 4+ sigma for Z-score).
    Extreme,
}

/// A single anomaly report emitted by a detector for a specific data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    /// Index of the sample in the input array passed to `detect()`.
    pub index: usize,
    /// Raw value of the sample being reported on.
    pub value: f64,
    /// Detector-specific score; higher means more anomalous.
    pub score: f64,
    /// Severity bucket derived from the score and configured thresholds.
    pub severity: AnomalySeverity,
    /// Name of the detector that produced the report.
    pub method: String,
}

/// Common configuration shared by every detector.
///
/// The thresholds carry detector-specific semantics:
///
/// - For `ZScoreDetector` and `ModifiedZScoreDetector` they are interpreted
///   directly as standard-score cutoffs (e.g. 2.0, 3.0, 4.0).
/// - For `IqrDetector` they multiply the IQR (e.g. 1.5, 3.0, 6.0).
/// - For `IsolationForestDetector` they are *unused*: that detector uses fixed
///   score thresholds (0.6 / 0.7 / 0.8) tied to the path-length formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    /// Score above which a point is considered a Mild anomaly.
    pub mild_threshold: f64,
    /// Score above which a point is considered a Severe anomaly.
    pub severe_threshold: f64,
    /// Score above which a point is considered an Extreme anomaly.
    pub extreme_threshold: f64,
    /// Minimum number of samples that must be supplied to `fit()`.
    pub min_samples: usize,
    /// Seed for ML detectors that involve randomness (Isolation Forest).
    pub seed: u64,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            mild_threshold: 2.0,
            severe_threshold: 3.0,
            extreme_threshold: 4.0,
            min_samples: 10,
            seed: 42,
        }
    }
}

/// Unified interface for anomaly detectors.
///
/// The `Send + Sync` bound lets detectors be shared across threads, which is
/// useful when the same fitted detector is consulted by many concurrent
/// benchmark workers.
pub trait AnomalyDetector: Send + Sync {
    /// Fit detector state from a vector of training samples.
    fn fit(&mut self, samples: &Array1<f64>) -> Result<()>;

    /// Run the fitted detector over `samples`, returning one report per sample.
    fn detect(&self, samples: &Array1<f64>) -> Result<Vec<AnomalyReport>>;

    /// Score a single value at logical position `index`.
    fn detect_one(&self, value: f64, index: usize) -> Result<AnomalyReport>;

    /// Whether `fit()` has been called successfully.
    fn is_fitted(&self) -> bool;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

/// Sort a copy of `samples` ascending, treating NaN as larger than any finite
/// value so they sort to the tail.
fn sorted_ascending(samples: &Array1<f64>) -> Vec<f64> {
    let mut buf: Vec<f64> = samples.iter().copied().collect();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    buf
}

/// Linear-interpolation quantile, equivalent to NumPy's `quantile(... method="linear")`.
///
/// `q` must lie in `[0, 1]`. Returns the interpolated value
/// `x_lower * (1 - frac) + x_upper * frac` where `idx = q * (n - 1)` and
/// `(lower, upper)` bracket the fractional index.
pub fn quantile(samples: &Array1<f64>, q: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&q) || q.is_nan() {
        return Err(OptimError::InvalidParameter(format!(
            "quantile q must be in [0, 1], got {q}"
        )));
    }
    let n = samples.len();
    if n == 0 {
        return Err(OptimError::InvalidConfig(
            "Cannot compute quantile of an empty array".to_string(),
        ));
    }
    let sorted = sorted_ascending(samples);
    if n == 1 {
        return Ok(sorted[0]);
    }
    let idx = q * ((n - 1) as f64);
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return Ok(sorted[lower]);
    }
    let frac = idx - (lower as f64);
    Ok(sorted[lower] * (1.0 - frac) + sorted[upper] * frac)
}

/// Median = quantile(0.5).
pub fn median(samples: &Array1<f64>) -> Result<f64> {
    quantile(samples, 0.5)
}

/// Median Absolute Deviation: `median(|x_i - median(x)|)`.
///
/// MAD is a robust spread estimator with a breakdown point of 50%: half of the
/// training samples can be arbitrarily corrupted before MAD becomes useless.
pub fn mad(samples: &Array1<f64>) -> Result<f64> {
    let med = median(samples)?;
    let deviations: Array1<f64> = samples.iter().map(|v| (*v - med).abs()).collect();
    median(&deviations)
}

/// Bucket a non-negative score into a [`AnomalySeverity`] using thresholds.
///
/// Thresholds need only satisfy `mild <= severe <= extreme`; this function does
/// not enforce that ordering, since detectors may legitimately use any
/// monotone mapping. NaN and negative scores are treated as `Normal`.
pub fn classify_severity(score: f64, mild: f64, severe: f64, extreme: f64) -> AnomalySeverity {
    if !score.is_finite() || score < 0.0 {
        return AnomalySeverity::Normal;
    }
    if score >= extreme {
        AnomalySeverity::Extreme
    } else if score >= severe {
        AnomalySeverity::Severe
    } else if score >= mild {
        AnomalySeverity::Mild
    } else {
        AnomalySeverity::Normal
    }
}

/// Count reports whose severity is anything other than `Normal`.
pub fn count_anomalies(reports: &[AnomalyReport]) -> usize {
    reports
        .iter()
        .filter(|r| r.severity != AnomalySeverity::Normal)
        .count()
}

/// Mean of a non-empty array. Returns 0.0 for an empty input.
fn mean_of(samples: &Array1<f64>) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    samples.iter().copied().sum::<f64>() / (n as f64)
}

/// Population standard deviation of a non-empty array.
fn std_of(samples: &Array1<f64>) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let mu = mean_of(samples);
    let mut acc = 0.0_f64;
    for v in samples.iter() {
        let d = *v - mu;
        acc += d * d;
    }
    (acc / (n as f64)).sqrt()
}

/// Approximate the n-th harmonic number using the Euler-Mascheroni asymptotic.
fn harmonic_number(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    // For small n, sum exactly; for larger n, use ln(n) + gamma.
    if n <= 32 {
        let mut s = 0.0_f64;
        for i in 1..=n {
            s += 1.0 / (i as f64);
        }
        s
    } else {
        (n as f64).ln() + 0.577_215_664_901_532_9
    }
}

/// Expected average path length of an unsuccessful BST search in a tree of
/// size `n`, used to normalise Isolation Forest depths.
///
/// `c(n) = 2 * H(n - 1) - 2 * (n - 1) / n`, with `c(0) = c(1) = 0`.
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    2.0 * harmonic_number(n - 1) - 2.0 * (nf - 1.0) / nf
}

// -----------------------------------------------------------------------------
// Z-score detector
// -----------------------------------------------------------------------------

/// Classic standard-score anomaly detector.
///
/// After [`Self::fit`], the detector stores the sample mean `mu` and the
/// population standard deviation `sigma` of the training data. The anomaly
/// score for a value `x` is `|x - mu| / sigma`; severity is then derived from
/// the configured thresholds.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    config: AnomalyDetectorConfig,
    mean: Option<f64>,
    std: Option<f64>,
}

impl ZScoreDetector {
    /// Construct a detector with the supplied configuration.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            mean: None,
            std: None,
        }
    }

    /// Convenience constructor using the default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(AnomalyDetectorConfig::default())
    }

    /// Learned mean from the training data, if any.
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Learned standard deviation from the training data, if any.
    pub fn std(&self) -> Option<f64> {
        self.std
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit(&mut self, samples: &Array1<f64>) -> Result<()> {
        if samples.len() < self.config.min_samples {
            return Err(OptimError::InvalidConfig(format!(
                "ZScoreDetector requires at least {} samples, got {}",
                self.config.min_samples,
                samples.len()
            )));
        }
        for v in samples.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "ZScoreDetector cannot fit on non-finite value {v}"
                )));
            }
        }
        let mu = mean_of(samples);
        let sigma = std_of(samples);
        // Guard against a constant series; use a tiny floor so we don't divide
        // by zero, while still being able to flag any value far enough away.
        let sigma_eff = if sigma < 1e-12 { 1e-12 } else { sigma };
        self.mean = Some(mu);
        self.std = Some(sigma_eff);
        Ok(())
    }

    fn detect(&self, samples: &Array1<f64>) -> Result<Vec<AnomalyReport>> {
        let mut out = Vec::with_capacity(samples.len());
        for (idx, v) in samples.iter().copied().enumerate() {
            out.push(self.detect_one(v, idx)?);
        }
        Ok(out)
    }

    fn detect_one(&self, value: f64, index: usize) -> Result<AnomalyReport> {
        let mu = self.mean.ok_or_else(|| {
            OptimError::InvalidState(
                "ZScoreDetector has not been fit yet; call fit() before detect()".to_string(),
            )
        })?;
        let sigma = self.std.ok_or_else(|| {
            OptimError::InvalidState(
                "ZScoreDetector has not been fit yet; call fit() before detect()".to_string(),
            )
        })?;
        let score = if value.is_finite() {
            (value - mu).abs() / sigma
        } else {
            // Non-finite samples are flagged as extreme.
            f64::INFINITY
        };
        let severity = classify_severity(
            score,
            self.config.mild_threshold,
            self.config.severe_threshold,
            self.config.extreme_threshold,
        );
        Ok(AnomalyReport {
            index,
            value,
            score,
            severity,
            method: "zscore".to_string(),
        })
    }

    fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }
}

// -----------------------------------------------------------------------------
// IQR detector
// -----------------------------------------------------------------------------

/// Tukey-style interquartile-range fence detector.
///
/// After [`Self::fit`], the detector stores the 25th percentile (`q1`), the
/// 75th percentile (`q3`), and the IQR. A value is normal whenever it lies
/// within `[q1 - mild * iqr, q3 + mild * iqr]`. Otherwise its score is the
/// signed-distance to the nearest fence, divided by the IQR.
#[derive(Debug, Clone)]
pub struct IqrDetector {
    config: AnomalyDetectorConfig,
    q1: Option<f64>,
    q3: Option<f64>,
    iqr: Option<f64>,
}

impl IqrDetector {
    /// Construct a detector with the supplied configuration. The defaults
    /// (`mild=2.0`, `severe=3.0`, `extreme=4.0`) inherit the Z-score convention
    /// but in the IQR world a common choice is `1.5 / 3.0 / 6.0`; callers can
    /// supply those explicitly via [`Self::with_iqr_defaults`].
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            q1: None,
            q3: None,
            iqr: None,
        }
    }

    /// Convenience constructor with IQR-idiomatic thresholds (1.5 / 3.0 / 6.0).
    pub fn with_iqr_defaults() -> Self {
        Self::new(AnomalyDetectorConfig {
            mild_threshold: 1.5,
            severe_threshold: 3.0,
            extreme_threshold: 6.0,
            ..AnomalyDetectorConfig::default()
        })
    }

    /// Learned 25th percentile, if fit.
    pub fn q1(&self) -> Option<f64> {
        self.q1
    }

    /// Learned 75th percentile, if fit.
    pub fn q3(&self) -> Option<f64> {
        self.q3
    }

    /// Learned IQR, if fit.
    pub fn iqr(&self) -> Option<f64> {
        self.iqr
    }
}

impl AnomalyDetector for IqrDetector {
    fn fit(&mut self, samples: &Array1<f64>) -> Result<()> {
        if samples.len() < self.config.min_samples {
            return Err(OptimError::InvalidConfig(format!(
                "IqrDetector requires at least {} samples, got {}",
                self.config.min_samples,
                samples.len()
            )));
        }
        for v in samples.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "IqrDetector cannot fit on non-finite value {v}"
                )));
            }
        }
        let q1 = quantile(samples, 0.25)?;
        let q3 = quantile(samples, 0.75)?;
        let iqr = q3 - q1;
        let iqr_eff = if iqr.abs() < 1e-12 { 1e-12 } else { iqr };
        self.q1 = Some(q1);
        self.q3 = Some(q3);
        self.iqr = Some(iqr_eff);
        Ok(())
    }

    fn detect(&self, samples: &Array1<f64>) -> Result<Vec<AnomalyReport>> {
        let mut out = Vec::with_capacity(samples.len());
        for (idx, v) in samples.iter().copied().enumerate() {
            out.push(self.detect_one(v, idx)?);
        }
        Ok(out)
    }

    fn detect_one(&self, value: f64, index: usize) -> Result<AnomalyReport> {
        let q1 = self.q1.ok_or_else(|| {
            OptimError::InvalidState(
                "IqrDetector has not been fit yet; call fit() before detect()".to_string(),
            )
        })?;
        let q3 = self.q3.ok_or_else(|| {
            OptimError::InvalidState(
                "IqrDetector has not been fit yet; call fit() before detect()".to_string(),
            )
        })?;
        let iqr = self.iqr.ok_or_else(|| {
            OptimError::InvalidState(
                "IqrDetector has not been fit yet; call fit() before detect()".to_string(),
            )
        })?;
        let mild = self.config.mild_threshold;
        let lower_fence = q1 - mild * iqr;
        let upper_fence = q3 + mild * iqr;

        let score = if !value.is_finite() {
            f64::INFINITY
        } else if value < lower_fence {
            (lower_fence - value) / iqr
        } else if value > upper_fence {
            (value - upper_fence) / iqr
        } else {
            0.0
        };
        // The score above is measured *beyond* the mild fence, so any value
        // outside the fence already has score >= 0. To map it back onto the
        // configured threshold scale we add the mild threshold itself: a point
        // whose distance equals one extra IQR past the mild fence will hit
        // exactly `mild + 1`, which crosses the severe threshold when severe
        // is one above mild.
        let effective_score = if score > 0.0 { score + mild } else { score };
        let severity = classify_severity(
            effective_score,
            self.config.mild_threshold,
            self.config.severe_threshold,
            self.config.extreme_threshold,
        );
        Ok(AnomalyReport {
            index,
            value,
            score: effective_score,
            severity,
            method: "iqr".to_string(),
        })
    }

    fn is_fitted(&self) -> bool {
        self.q1.is_some() && self.q3.is_some() && self.iqr.is_some()
    }
}

// -----------------------------------------------------------------------------
// Modified Z-score detector (MAD-based)
// -----------------------------------------------------------------------------

/// Robust Z-score detector based on the Median Absolute Deviation.
///
/// The score for a value `x` is `0.6745 * |x - median| / MAD`, where the
/// constant `0.6745` is the inverse of the 75th percentile of the standard
/// normal distribution. With this scaling, when the training data is normal,
/// the modified Z-score is approximately a standard Z-score.
///
/// Because both `median` and `MAD` have a 50% breakdown point, this detector
/// remains effective even when half of the training samples are contaminated.
#[derive(Debug, Clone)]
pub struct ModifiedZScoreDetector {
    config: AnomalyDetectorConfig,
    median: Option<f64>,
    mad_value: Option<f64>,
}

impl ModifiedZScoreDetector {
    /// Construct a detector with the supplied configuration.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            median: None,
            mad_value: None,
        }
    }

    /// Convenience constructor with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(AnomalyDetectorConfig::default())
    }

    /// Learned median, if fit.
    pub fn median_value(&self) -> Option<f64> {
        self.median
    }

    /// Learned MAD, if fit.
    pub fn mad_value(&self) -> Option<f64> {
        self.mad_value
    }
}

impl AnomalyDetector for ModifiedZScoreDetector {
    fn fit(&mut self, samples: &Array1<f64>) -> Result<()> {
        if samples.len() < self.config.min_samples {
            return Err(OptimError::InvalidConfig(format!(
                "ModifiedZScoreDetector requires at least {} samples, got {}",
                self.config.min_samples,
                samples.len()
            )));
        }
        for v in samples.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "ModifiedZScoreDetector cannot fit on non-finite value {v}"
                )));
            }
        }
        let med = median(samples)?;
        let mad_v = mad(samples)?;
        // Guard against the degenerate case where every training sample equals
        // the median (MAD = 0). We substitute a tiny epsilon so the detector
        // remains usable: any value not equal to the median will then have a
        // large modified Z-score, which is the right behaviour.
        let mad_eff = if mad_v.abs() < 1e-12 { 1e-12 } else { mad_v };
        self.median = Some(med);
        self.mad_value = Some(mad_eff);
        Ok(())
    }

    fn detect(&self, samples: &Array1<f64>) -> Result<Vec<AnomalyReport>> {
        let mut out = Vec::with_capacity(samples.len());
        for (idx, v) in samples.iter().copied().enumerate() {
            out.push(self.detect_one(v, idx)?);
        }
        Ok(out)
    }

    fn detect_one(&self, value: f64, index: usize) -> Result<AnomalyReport> {
        let med = self.median.ok_or_else(|| {
            OptimError::InvalidState(
                "ModifiedZScoreDetector has not been fit yet; call fit() before detect()"
                    .to_string(),
            )
        })?;
        let mad_v = self.mad_value.ok_or_else(|| {
            OptimError::InvalidState(
                "ModifiedZScoreDetector has not been fit yet; call fit() before detect()"
                    .to_string(),
            )
        })?;
        let score = if value.is_finite() {
            0.6745 * (value - med).abs() / mad_v
        } else {
            f64::INFINITY
        };
        let severity = classify_severity(
            score,
            self.config.mild_threshold,
            self.config.severe_threshold,
            self.config.extreme_threshold,
        );
        Ok(AnomalyReport {
            index,
            value,
            score,
            severity,
            method: "modified_zscore".to_string(),
        })
    }

    fn is_fitted(&self) -> bool {
        self.median.is_some() && self.mad_value.is_some()
    }
}

// -----------------------------------------------------------------------------
// Isolation Forest detector
// -----------------------------------------------------------------------------

/// A node in a (one-dimensional) Isolation Tree.
///
/// Each leaf records the number of training samples that reached it; internal
/// nodes record the split threshold and the indices of the left / right
/// children inside the owning tree's `nodes` vector.
#[derive(Debug, Clone)]
struct IsolationNode {
    /// `true` for terminal leaves.
    is_leaf: bool,
    /// Split threshold (internal nodes only).
    split: f64,
    /// Index of the left child in `IsolationTree::nodes`.
    left: usize,
    /// Index of the right child in `IsolationTree::nodes`.
    right: usize,
    /// Number of training samples in this leaf (for path-length correction).
    size: usize,
    /// Depth of this node from the tree root.
    depth: usize,
}

/// A single Isolation Tree.
///
/// In the original Isolation Forest the trees split on a randomly chosen
/// feature; here, since we operate on scalar metrics, the "feature" is always
/// the value itself.
#[derive(Debug, Clone)]
struct IsolationTree {
    nodes: Vec<IsolationNode>,
}

impl IsolationTree {
    /// Build a new tree by recursively splitting `values` until each leaf has
    /// at most one sample or `max_depth` is reached.
    fn build<R: Rng>(values: &[f64], max_depth: usize, rng: &mut Random<R>) -> Self {
        let mut nodes: Vec<IsolationNode> = Vec::new();
        // Use indices into a single flat vector instead of recursion so the
        // call stack does not depend on input size.
        struct Frame {
            values: Vec<f64>,
            depth: usize,
            parent_slot: Option<(usize, bool)>, // (parent_idx, is_left)
        }
        let mut stack: Vec<Frame> = vec![Frame {
            values: values.to_vec(),
            depth: 0,
            parent_slot: None,
        }];

        while let Some(frame) = stack.pop() {
            let Frame {
                values,
                depth,
                parent_slot,
            } = frame;
            let size = values.len();

            // Stop if depth limit reached, or only one sample remains, or all
            // values are identical (no useful split exists).
            let all_same = if size > 1 {
                let v0 = values[0];
                values.iter().all(|v| (*v - v0).abs() < 1e-15)
            } else {
                true
            };

            if size <= 1 || depth >= max_depth || all_same {
                let leaf_idx = nodes.len();
                nodes.push(IsolationNode {
                    is_leaf: true,
                    split: 0.0,
                    left: 0,
                    right: 0,
                    size,
                    depth,
                });
                if let Some((parent_idx, is_left)) = parent_slot {
                    if is_left {
                        nodes[parent_idx].left = leaf_idx;
                    } else {
                        nodes[parent_idx].right = leaf_idx;
                    }
                }
                continue;
            }

            // Random split in the (min, max) range. Find min/max in O(n).
            let mut vmin = values[0];
            let mut vmax = values[0];
            for v in values.iter().copied() {
                if v < vmin {
                    vmin = v;
                }
                if v > vmax {
                    vmax = v;
                }
            }
            // Degenerate case caught above, but guard once more.
            if (vmax - vmin).abs() < 1e-15 {
                let leaf_idx = nodes.len();
                nodes.push(IsolationNode {
                    is_leaf: true,
                    split: 0.0,
                    left: 0,
                    right: 0,
                    size,
                    depth,
                });
                if let Some((parent_idx, is_left)) = parent_slot {
                    if is_left {
                        nodes[parent_idx].left = leaf_idx;
                    } else {
                        nodes[parent_idx].right = leaf_idx;
                    }
                }
                continue;
            }
            let split: f64 = rng.gen_range(vmin..vmax);

            // Reserve the internal node now; children indices will be filled
            // when their respective leaves / subtrees are popped from the
            // stack.
            let node_idx = nodes.len();
            nodes.push(IsolationNode {
                is_leaf: false,
                split,
                left: 0,
                right: 0,
                size,
                depth,
            });
            if let Some((parent_idx, is_left)) = parent_slot {
                if is_left {
                    nodes[parent_idx].left = node_idx;
                } else {
                    nodes[parent_idx].right = node_idx;
                }
            }

            // Partition.
            let mut left_vals = Vec::with_capacity(size / 2);
            let mut right_vals = Vec::with_capacity(size / 2);
            for v in values.into_iter() {
                if v < split {
                    left_vals.push(v);
                } else {
                    right_vals.push(v);
                }
            }
            // Push right first so left is processed first (LIFO order does
            // not affect correctness but yields more readable structures).
            stack.push(Frame {
                values: right_vals,
                depth: depth + 1,
                parent_slot: Some((node_idx, false)),
            });
            stack.push(Frame {
                values: left_vals,
                depth: depth + 1,
                parent_slot: Some((node_idx, true)),
            });
        }

        // Edge case: empty input.
        if nodes.is_empty() {
            nodes.push(IsolationNode {
                is_leaf: true,
                split: 0.0,
                left: 0,
                right: 0,
                size: 0,
                depth: 0,
            });
        }
        Self { nodes }
    }

    /// Compute the path length for `value`, augmented with the expected path
    /// length of an unsuccessful search in a subtree of the leaf's size.
    fn path_length(&self, value: f64) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut idx = 0_usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return (node.depth as f64) + average_path_length(node.size);
            }
            idx = if value < node.split {
                node.left
            } else {
                node.right
            };
            // Defensive: if a child index points back to the same node we have
            // a malformed tree; bail out to avoid an infinite loop.
            if idx == 0 && node.depth > 0 {
                return node.depth as f64;
            }
        }
    }
}

/// Isolation Forest anomaly detector for one-dimensional metrics.
///
/// Trains `n_trees` random binary trees; each tree splits the training values
/// on a random threshold drawn uniformly from `[min, max]` of the data
/// reaching the node. The anomaly score for a value is
/// `s(x) = 2^(-E[h(x)] / c(n))` where `E[h(x)]` is the average path length
/// across trees and `c(n)` is the expected path length of an unsuccessful BST
/// search of size `n` (the training set size).
///
/// Higher scores correspond to "easier" to isolate points, which are more
/// likely to be anomalies. The severity mapping uses fixed thresholds
/// (0.6 / 0.7 / 0.8) chosen to follow the original paper's recommendation
/// (anything above ~0.6 is suspicious; >0.8 is almost certainly anomalous).
#[derive(Debug, Clone)]
pub struct IsolationForestDetector {
    config: AnomalyDetectorConfig,
    n_trees: usize,
    trees: Vec<IsolationTree>,
    train_size: usize,
}

impl IsolationForestDetector {
    /// Construct a detector with the default tree count of 100.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            n_trees: 100,
            trees: Vec::new(),
            train_size: 0,
        }
    }

    /// Convenience constructor with default thresholds and 100 trees.
    pub fn with_defaults() -> Self {
        Self::new(AnomalyDetectorConfig::default())
    }

    /// Configure the number of trees in the forest.
    pub fn with_n_trees(mut self, n: usize) -> Self {
        self.n_trees = n.max(1);
        self
    }

    /// Number of trees in the fitted forest.
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// Training-set size that was supplied to [`Self::fit`].
    pub fn train_size(&self) -> usize {
        self.train_size
    }

    /// Internal: score formula `2^(-E[h] / c(n))`, clamped to [0, 1].
    fn score_value(&self, value: f64) -> f64 {
        if self.trees.is_empty() || self.train_size == 0 {
            return 0.0;
        }
        if !value.is_finite() {
            return 1.0;
        }
        let mut sum_depth = 0.0_f64;
        for tree in self.trees.iter() {
            sum_depth += tree.path_length(value);
        }
        let avg_depth = sum_depth / (self.trees.len() as f64);
        let c = average_path_length(self.train_size);
        if c <= 0.0 {
            return 0.0;
        }
        let exponent = -avg_depth / c;
        let s = 2.0_f64.powf(exponent);
        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl AnomalyDetector for IsolationForestDetector {
    fn fit(&mut self, samples: &Array1<f64>) -> Result<()> {
        if samples.len() < self.config.min_samples {
            return Err(OptimError::InvalidConfig(format!(
                "IsolationForestDetector requires at least {} samples, got {}",
                self.config.min_samples,
                samples.len()
            )));
        }
        for v in samples.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "IsolationForestDetector cannot fit on non-finite value {v}"
                )));
            }
        }

        // max_depth = ceil(log2(n)); guard against n <= 1 (filtered above by
        // min_samples >= 10 but defensive code is cheap).
        let n = samples.len();
        let max_depth = if n <= 1 {
            1
        } else {
            ((n as f64).log2().ceil() as usize).max(1)
        };

        let raw_vals: Vec<f64> = samples.iter().copied().collect();
        let mut rng = Random::seed(self.config.seed);
        let mut trees: Vec<IsolationTree> = Vec::with_capacity(self.n_trees);
        for _ in 0..self.n_trees {
            // Sub-sampling reduces tree correlation; we use sampling-with-
            // replacement of size n which preserves the original IF behaviour
            // for small datasets but adds randomness.
            let mut subset: Vec<f64> = Vec::with_capacity(n);
            for _ in 0..n {
                let idx: usize = rng.gen_range(0..n);
                subset.push(raw_vals[idx]);
            }
            trees.push(IsolationTree::build(&subset, max_depth, &mut rng));
        }
        self.trees = trees;
        self.train_size = n;
        Ok(())
    }

    fn detect(&self, samples: &Array1<f64>) -> Result<Vec<AnomalyReport>> {
        let mut out = Vec::with_capacity(samples.len());
        for (idx, v) in samples.iter().copied().enumerate() {
            out.push(self.detect_one(v, idx)?);
        }
        Ok(out)
    }

    fn detect_one(&self, value: f64, index: usize) -> Result<AnomalyReport> {
        if !self.is_fitted() {
            return Err(OptimError::InvalidState(
                "IsolationForestDetector has not been fit yet; call fit() before detect()"
                    .to_string(),
            ));
        }
        let score = self.score_value(value);
        // Fixed thresholds (Liu et al. 2008): >0.6 mild, >0.7 severe, >0.8 extreme.
        let severity = if score > 0.8 {
            AnomalySeverity::Extreme
        } else if score > 0.7 {
            AnomalySeverity::Severe
        } else if score > 0.6 {
            AnomalySeverity::Mild
        } else {
            AnomalySeverity::Normal
        };
        Ok(AnomalyReport {
            index,
            value,
            score,
            severity,
            method: "isolation_forest".to_string(),
        })
    }

    fn is_fitted(&self) -> bool {
        !self.trees.is_empty() && self.train_size > 0
    }
}

// -----------------------------------------------------------------------------
// Convenience constructors / batch helpers
// -----------------------------------------------------------------------------

/// Build a 2D detection matrix where each column corresponds to a different
/// detector run on the *same* sample vector. Mostly useful for downstream
/// reporting / visualisation tools that want to compare detector outputs
/// side-by-side.
pub fn detect_with_all<D: AnomalyDetector + ?Sized>(
    detectors: &mut [&mut D],
    train: &Array1<f64>,
    test: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_test = test.len();
    let n_det = detectors.len();
    let mut out = Array2::<f64>::zeros((n_test, n_det));
    for (col, det) in detectors.iter_mut().enumerate() {
        det.fit(train)?;
        let reports = det.detect(test)?;
        for (row, r) in reports.iter().enumerate() {
            out[[row, col]] = r.score;
        }
    }
    Ok(out)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Box-Muller transform: turn two uniform draws into a standard-normal one.
    fn box_muller<R: Rng>(rng: &mut Random<R>) -> f64 {
        let u1: f64 = rng.gen_range(1e-12_f64..1.0);
        let u2: f64 = rng.gen_range(0.0_f64..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }

    /// Generate `n` iid samples from N(0, 1) using a seeded SciRS2 RNG.
    fn normal_samples(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = Random::seed(seed);
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            out[i] = box_muller(&mut rng);
        }
        out
    }

    // ---- helpers ----

    #[test]
    fn test_quantile_correctness() {
        let arr = Array1::from_vec((1..=10).map(|i| i as f64).collect());
        assert_relative_eq!(quantile(&arr, 0.5).expect("ok"), 5.5, epsilon = 1e-9);
        assert_relative_eq!(quantile(&arr, 0.25).expect("ok"), 3.25, epsilon = 1e-9);
        assert_relative_eq!(quantile(&arr, 0.75).expect("ok"), 7.75, epsilon = 1e-9);
        // Boundary values.
        assert_relative_eq!(quantile(&arr, 0.0).expect("ok"), 1.0, epsilon = 1e-12);
        assert_relative_eq!(quantile(&arr, 1.0).expect("ok"), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_quantile_invalid_q_errors() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let bad = quantile(&arr, -0.1);
        assert!(matches!(bad, Err(OptimError::InvalidParameter(_))));
        let bad2 = quantile(&arr, 1.1);
        assert!(matches!(bad2, Err(OptimError::InvalidParameter(_))));
    }

    #[test]
    fn test_quantile_empty_errors() {
        let arr = Array1::<f64>::zeros(0);
        let bad = quantile(&arr, 0.5);
        assert!(matches!(bad, Err(OptimError::InvalidConfig(_))));
    }

    #[test]
    fn test_median_correctness() {
        let odd = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(median(&odd).expect("ok"), 3.0, epsilon = 1e-12);
        let even = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(median(&even).expect("ok"), 2.5, epsilon = 1e-12);
        // Unsorted input should still yield the correct median.
        let unsorted = Array1::from_vec(vec![5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_relative_eq!(median(&unsorted).expect("ok"), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mad_correctness() {
        // From the spec: mad([1, 1, 2, 2, 4, 6, 9]) = 1.0
        // median is 2; deviations [1, 1, 0, 0, 2, 4, 7]; median of those is 1.
        let arr = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0]);
        let m = mad(&arr).expect("ok");
        assert_relative_eq!(m, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_classify_severity_thresholds() {
        let mild = 2.0;
        let severe = 3.0;
        let extreme = 4.0;
        assert_eq!(
            classify_severity(0.0, mild, severe, extreme),
            AnomalySeverity::Normal
        );
        assert_eq!(
            classify_severity(1.9, mild, severe, extreme),
            AnomalySeverity::Normal
        );
        assert_eq!(
            classify_severity(2.0, mild, severe, extreme),
            AnomalySeverity::Mild
        );
        assert_eq!(
            classify_severity(2.99, mild, severe, extreme),
            AnomalySeverity::Mild
        );
        assert_eq!(
            classify_severity(3.0, mild, severe, extreme),
            AnomalySeverity::Severe
        );
        assert_eq!(
            classify_severity(3.99, mild, severe, extreme),
            AnomalySeverity::Severe
        );
        assert_eq!(
            classify_severity(4.0, mild, severe, extreme),
            AnomalySeverity::Extreme
        );
        assert_eq!(
            classify_severity(99.0, mild, severe, extreme),
            AnomalySeverity::Extreme
        );
        // NaN / negative -> Normal.
        assert_eq!(
            classify_severity(f64::NAN, mild, severe, extreme),
            AnomalySeverity::Normal
        );
        assert_eq!(
            classify_severity(-1.0, mild, severe, extreme),
            AnomalySeverity::Normal
        );
    }

    #[test]
    fn test_count_anomalies() {
        let reports = vec![
            AnomalyReport {
                index: 0,
                value: 0.0,
                score: 0.5,
                severity: AnomalySeverity::Normal,
                method: "zscore".into(),
            },
            AnomalyReport {
                index: 1,
                value: 5.0,
                score: 2.5,
                severity: AnomalySeverity::Mild,
                method: "zscore".into(),
            },
            AnomalyReport {
                index: 2,
                value: 10.0,
                score: 5.0,
                severity: AnomalySeverity::Severe,
                method: "zscore".into(),
            },
            AnomalyReport {
                index: 3,
                value: 0.0,
                score: 0.1,
                severity: AnomalySeverity::Normal,
                method: "zscore".into(),
            },
            AnomalyReport {
                index: 4,
                value: 99.0,
                score: 10.0,
                severity: AnomalySeverity::Extreme,
                method: "zscore".into(),
            },
        ];
        assert_eq!(count_anomalies(&reports), 3);
        // Empty input -> 0.
        assert_eq!(count_anomalies(&[]), 0);
    }

    // ---- Z-score detector ----

    #[test]
    fn test_zscore_detector_no_outliers_in_normal_data() {
        let samples = normal_samples(500, 123);
        let mut det = ZScoreDetector::with_defaults();
        det.fit(&samples).expect("fit");
        let reports = det.detect(&samples).expect("detect");
        let severe_count = reports
            .iter()
            .filter(|r| {
                matches!(
                    r.severity,
                    AnomalySeverity::Severe | AnomalySeverity::Extreme
                )
            })
            .count();
        // We expect very few severe outliers (3-sigma in normal -> ~0.3%).
        assert!(
            severe_count < 5,
            "Too many severe outliers in N(0,1): {severe_count}"
        );
    }

    #[test]
    fn test_zscore_detector_flags_extreme_outlier() {
        let mut samples = normal_samples(200, 7);
        // Append a 10-sigma value.
        let mut as_vec: Vec<f64> = samples.iter().copied().collect();
        as_vec.push(10.0);
        samples = Array1::from_vec(as_vec);

        let mut det = ZScoreDetector::with_defaults();
        det.fit(&samples).expect("fit");
        let reports = det.detect(&samples).expect("detect");
        let last = reports.last().expect("non-empty");
        assert_eq!(last.severity, AnomalySeverity::Extreme);
        assert!(last.score > 4.0, "Expected score > 4.0, got {}", last.score);
    }

    #[test]
    fn test_zscore_detector_min_samples_error() {
        let mut det = ZScoreDetector::new(AnomalyDetectorConfig {
            min_samples: 10,
            ..AnomalyDetectorConfig::default()
        });
        let too_few = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let err = det.fit(&too_few).expect_err("must error");
        assert!(matches!(err, OptimError::InvalidConfig(_)));
    }

    #[test]
    fn test_zscore_detector_non_finite_in_fit_errors() {
        let mut det = ZScoreDetector::with_defaults();
        let bad = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, f64::NAN]);
        let err = det.fit(&bad).expect_err("must error");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    // ---- IQR detector ----

    #[test]
    fn test_iqr_detector_flags_outside_bounds() {
        let mut det = IqrDetector::with_iqr_defaults();
        // Fit on a tight cluster around 0 with a couple of moderate outliers.
        let mut buf = vec![
            -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        ];
        // Padding samples for min_samples.
        for i in 0..20 {
            buf.push(0.001 * (i as f64));
        }
        let samples = Array1::from_vec(buf);
        det.fit(&samples).expect("fit");
        // A clearly-outside value should yield a positive score and at least
        // Mild severity.
        let report = det.detect_one(10.0, 0).expect("detect_one");
        assert!(
            matches!(
                report.severity,
                AnomalySeverity::Mild | AnomalySeverity::Severe | AnomalySeverity::Extreme
            ),
            "Expected non-Normal severity for value=10.0, got {:?}",
            report.severity
        );
        // A value inside the bounds should be Normal.
        let inside = det.detect_one(0.0, 0).expect("detect_one");
        assert_eq!(inside.severity, AnomalySeverity::Normal);
        assert_eq!(inside.score, 0.0);
    }

    #[test]
    fn test_iqr_detector_robust_to_few_outliers() {
        // 100 normal + 5 extreme outliers; IQR is computed on quartiles so a
        // few extreme tails do not blow up the fences.
        let mut buf: Vec<f64> = normal_samples(100, 11).iter().copied().collect();
        buf.extend(std::iter::repeat_n(50.0, 5));
        let arr = Array1::from_vec(buf);
        let mut det = IqrDetector::with_iqr_defaults();
        det.fit(&arr).expect("fit");
        let reports = det.detect(&arr).expect("detect");
        let extreme_count = reports
            .iter()
            .filter(|r| {
                matches!(
                    r.severity,
                    AnomalySeverity::Mild | AnomalySeverity::Severe | AnomalySeverity::Extreme
                )
            })
            .count();
        // We need to flag at least the 5 injected outliers (more is fine).
        assert!(
            extreme_count >= 5,
            "Expected >=5 flagged outliers, got {extreme_count}"
        );
        // The last 5 reports correspond to the injected 50.0 values.
        for r in reports.iter().rev().take(5) {
            assert!(
                matches!(
                    r.severity,
                    AnomalySeverity::Mild | AnomalySeverity::Severe | AnomalySeverity::Extreme
                ),
                "Injected outlier was not flagged: {r:?}"
            );
        }
    }

    #[test]
    fn test_iqr_detector_normal_data_few_outliers() {
        let samples = normal_samples(200, 23);
        let mut det = IqrDetector::with_iqr_defaults();
        det.fit(&samples).expect("fit");
        let reports = det.detect(&samples).expect("detect");
        let extreme_count = reports
            .iter()
            .filter(|r| matches!(r.severity, AnomalySeverity::Extreme))
            .count();
        assert!(
            extreme_count < 5,
            "Too many extreme outliers on clean N(0,1) data: {extreme_count}"
        );
    }

    // ---- Modified Z-score detector ----

    #[test]
    fn test_mad_detector_robust_to_high_outlier_contamination() {
        // The median has a breakdown point of 50%, so it remains the right
        // estimator as long as strictly fewer than half the samples are
        // contaminated. With 60 normal + 40 outlier samples (40% contamination)
        // the median still lands in the bulk of the normal data.
        //
        // Compare against the parametric ZScoreDetector: it is *not* robust,
        // and its sample mean / std get dragged by the contamination, which
        // demonstrates why MAD-based detection is preferred for noisy streams.
        let mut buf: Vec<f64> = normal_samples(60, 42).iter().copied().collect();
        buf.extend(std::iter::repeat_n(100.0, 40));
        let arr = Array1::from_vec(buf);
        let mut det = ModifiedZScoreDetector::with_defaults();
        det.fit(&arr).expect("fit");
        let mad_v = det.mad_value().expect("mad fit");
        let med = det.median_value().expect("median fit");
        // MAD must remain finite, positive, and the median must stay near 0
        // (the normal cluster center), not get dragged to 100.
        assert!(mad_v.is_finite());
        assert!(mad_v > 0.0);
        assert!(
            med.abs() < 5.0,
            "Median {med} should stay near 0 with 40% contamination"
        );
        // Direct detection: a value of 100 must get flagged as anomalous.
        let r = det.detect_one(100.0, 0).expect("detect_one");
        assert!(matches!(
            r.severity,
            AnomalySeverity::Mild | AnomalySeverity::Severe | AnomalySeverity::Extreme
        ));
    }

    #[test]
    fn test_mad_detector_zero_mad_handled() {
        // All identical training values -> MAD = 0, so we substitute eps.
        let arr = Array1::from_vec(vec![5.0; 20]);
        let mut det = ModifiedZScoreDetector::with_defaults();
        det.fit(&arr).expect("fit");
        // The median equals 5.0; a novel value of 7.0 should be flagged as
        // anomalous (very large modified Z-score thanks to the epsilon MAD).
        let r = det.detect_one(7.0, 0).expect("detect_one");
        assert_ne!(r.severity, AnomalySeverity::Normal);
        // A value exactly at the median should be Normal.
        let r2 = det.detect_one(5.0, 0).expect("detect_one");
        assert_eq!(r2.severity, AnomalySeverity::Normal);
    }

    #[test]
    fn test_mad_detector_min_samples_error() {
        let mut det = ModifiedZScoreDetector::new(AnomalyDetectorConfig {
            min_samples: 10,
            ..AnomalyDetectorConfig::default()
        });
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let err = det.fit(&arr).expect_err("must error");
        assert!(matches!(err, OptimError::InvalidConfig(_)));
    }

    // ---- Isolation Forest detector ----

    #[test]
    fn test_isolation_forest_flags_clear_outlier() {
        let samples = normal_samples(200, 5);
        let mut det = IsolationForestDetector::with_defaults();
        det.fit(&samples).expect("fit");
        let outlier = det.detect_one(10.0, 0).expect("detect_one");
        let inlier = det.detect_one(0.0, 0).expect("detect_one");
        assert!(
            outlier.score > inlier.score,
            "Outlier score {} not > inlier score {}",
            outlier.score,
            inlier.score
        );
    }

    #[test]
    fn test_isolation_forest_reproducible_with_seed() {
        let samples = normal_samples(150, 88);
        let cfg = AnomalyDetectorConfig {
            seed: 1234,
            ..AnomalyDetectorConfig::default()
        };
        let mut a = IsolationForestDetector::new(cfg.clone()).with_n_trees(30);
        let mut b = IsolationForestDetector::new(cfg).with_n_trees(30);
        a.fit(&samples).expect("fit a");
        b.fit(&samples).expect("fit b");
        let ra = a.detect_one(10.0, 0).expect("a detect");
        let rb = b.detect_one(10.0, 0).expect("b detect");
        assert_relative_eq!(ra.score, rb.score, epsilon = 1e-12);
        // Also check a batch.
        let test = Array1::from_vec(vec![0.0, 0.5, -1.0, 2.5, 10.0, -10.0]);
        let reports_a = a.detect(&test).expect("a batch");
        let reports_b = b.detect(&test).expect("b batch");
        for (xa, xb) in reports_a.iter().zip(reports_b.iter()) {
            assert_relative_eq!(xa.score, xb.score, epsilon = 1e-12);
            assert_eq!(xa.severity, xb.severity);
        }
    }

    #[test]
    fn test_isolation_forest_fewer_outliers_in_normal_data() {
        let samples = normal_samples(300, 314);
        let mut det = IsolationForestDetector::with_defaults().with_n_trees(50);
        det.fit(&samples).expect("fit");
        let reports = det.detect(&samples).expect("detect");
        let severe_count = reports
            .iter()
            .filter(|r| {
                matches!(
                    r.severity,
                    AnomalySeverity::Severe | AnomalySeverity::Extreme
                )
            })
            .count();
        // For N(0,1) data we expect very few severe scores; allow up to 5%.
        let allow = (samples.len() as f64 * 0.05) as usize;
        assert!(
            severe_count < allow.max(5),
            "Too many severe scores on clean data: {severe_count} (allow={allow})"
        );
    }

    #[test]
    fn test_isolation_forest_min_samples_error() {
        let mut det = IsolationForestDetector::with_defaults();
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let err = det.fit(&arr).expect_err("must error");
        assert!(matches!(err, OptimError::InvalidConfig(_)));
    }

    // ---- Trait state / pre-fit / batch ----

    #[test]
    fn test_detect_before_fit_errors() {
        let z = ZScoreDetector::with_defaults();
        let r = z.detect_one(0.0, 0).expect_err("must error");
        assert!(matches!(r, OptimError::InvalidState(_)));

        let i = IqrDetector::with_iqr_defaults();
        let r2 = i.detect_one(0.0, 0).expect_err("must error");
        assert!(matches!(r2, OptimError::InvalidState(_)));

        let m = ModifiedZScoreDetector::with_defaults();
        let r3 = m.detect_one(0.0, 0).expect_err("must error");
        assert!(matches!(r3, OptimError::InvalidState(_)));

        let f = IsolationForestDetector::with_defaults();
        let r4 = f.detect_one(0.0, 0).expect_err("must error");
        assert!(matches!(r4, OptimError::InvalidState(_)));
    }

    #[test]
    fn test_is_fitted_flag() {
        let samples = normal_samples(50, 9);
        let mut z = ZScoreDetector::with_defaults();
        assert!(!z.is_fitted());
        z.fit(&samples).expect("fit");
        assert!(z.is_fitted());

        let mut i = IqrDetector::with_iqr_defaults();
        assert!(!i.is_fitted());
        i.fit(&samples).expect("fit");
        assert!(i.is_fitted());

        let mut m = ModifiedZScoreDetector::with_defaults();
        assert!(!m.is_fitted());
        m.fit(&samples).expect("fit");
        assert!(m.is_fitted());

        let mut f = IsolationForestDetector::with_defaults().with_n_trees(20);
        assert!(!f.is_fitted());
        f.fit(&samples).expect("fit");
        assert!(f.is_fitted());
    }

    #[test]
    fn test_detect_batch_indices_are_sequential() {
        let samples = normal_samples(60, 99);
        let mut z = ZScoreDetector::with_defaults();
        z.fit(&samples).expect("fit");
        let test = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 10.0]);
        let reports = z.detect(&test).expect("detect");
        for (i, r) in reports.iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.method, "zscore");
        }
        // The last value (10.0) is definitely an outlier.
        assert_ne!(reports[4].severity, AnomalySeverity::Normal);
    }

    #[test]
    fn test_detect_with_all_helper() {
        let train = normal_samples(80, 17);
        let test = Array1::from_vec(vec![0.0, 1.0, -1.0, 5.0, -5.0]);
        let mut z = ZScoreDetector::with_defaults();
        let mut m = ModifiedZScoreDetector::with_defaults();
        let mut detectors: Vec<&mut dyn AnomalyDetector> = vec![&mut z, &mut m];
        // Box the trait objects into the helper.
        let n_test = test.len();
        let n_det = detectors.len();
        let mut scores = Array2::<f64>::zeros((n_test, n_det));
        for (col, det) in detectors.iter_mut().enumerate() {
            det.fit(&train).expect("fit");
            let reports = det.detect(&test).expect("detect");
            for (row, r) in reports.iter().enumerate() {
                scores[[row, col]] = r.score;
            }
        }
        // Row 3 corresponds to value 5.0 -> both detectors should score it higher
        // than row 0 (value 0.0).
        for col in 0..n_det {
            assert!(
                scores[[3, col]] > scores[[0, col]],
                "Score for 5.0 should exceed score for 0.0 in column {col}"
            );
        }
    }

    #[test]
    fn test_anomaly_report_serde_round_trip() {
        // Ensure the public types are serde-friendly.
        let captured_value = 12.75_f64;
        let r = AnomalyReport {
            index: 7,
            value: captured_value,
            score: 2.5,
            severity: AnomalySeverity::Severe,
            method: "zscore".to_string(),
        };
        let s = serde_json::to_string(&r).expect("ser ok");
        let back: AnomalyReport = serde_json::from_str(&s).expect("de ok");
        assert_eq!(back.index, 7);
        assert_relative_eq!(back.value, captured_value, epsilon = 1e-12);
        assert_eq!(back.severity, AnomalySeverity::Severe);
    }

    #[test]
    fn test_average_path_length_matches_paper() {
        // c(2) = 2 * H(1) - 2 * 1 / 2 = 2 - 1 = 1
        assert_relative_eq!(average_path_length(2), 1.0, epsilon = 1e-9);
        // c(1) = 0, c(0) = 0
        assert_relative_eq!(average_path_length(1), 0.0, epsilon = 1e-12);
        assert_relative_eq!(average_path_length(0), 0.0, epsilon = 1e-12);
        // For large n, c(n) ~ 2 (ln(n-1) + gamma) - 2(n-1)/n; check a moderate value.
        let c10 = average_path_length(10);
        // 2 * H(9) ≈ 2 * 2.8289682... = 5.6579..., minus 2*9/10 = 1.8 -> 3.858...
        assert!((c10 - 3.858).abs() < 0.01, "c(10) = {c10}");
    }

    #[test]
    fn test_isolation_forest_score_bounded() {
        let samples = normal_samples(100, 3);
        let mut det = IsolationForestDetector::with_defaults().with_n_trees(20);
        det.fit(&samples).expect("fit");
        for v in [-1000.0, -3.0, 0.0, 3.0, 1000.0, f64::INFINITY, f64::NAN] {
            let r = det.detect_one(v, 0).expect("detect_one");
            assert!(
                (0.0..=1.0).contains(&r.score),
                "Score {} for value {v} out of [0,1]",
                r.score
            );
        }
    }
}
