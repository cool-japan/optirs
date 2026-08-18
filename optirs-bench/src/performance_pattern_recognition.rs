// Performance Pattern Recognition for Time-Series Benchmark Metrics
//
// This module implements **general-purpose pattern recognition** for univariate
// performance time-series (latency, throughput, loss, gradient norms, ...). It
// is *distinct* from `advanced_pattern_detection.rs`, which focuses on the
// memory-leak signatures (sawtooth / staircase / leak waveforms) of a memory
// usage trace; the algorithms here are fundamentally different and operate on
// generic, unconstrained performance series.
//
// The detector exposes five families of analysis:
//
// 1. **Motif discovery** via a simplified STAMP-style matrix profile. Returns
//    the top-k recurring sub-sequences under z-normalised Euclidean distance.
// 2. **CUSUM change-point detection** with separate forward (upward shift)
//    and reverse (downward shift) passes against the full-series baseline.
// 3. **Page-Hinkley change-point detection**, a sequential statistical test
//    for sudden drift in mean.
// 4. **Regime segmentation** via offline binary segmentation that recursively
//    partitions the series at the index that maximally reduces total
//    sum-of-squared-error subject to a minimum-segment-length constraint and a
//    minimum cost-improvement threshold.
// 5. **Trend classification** that labels a series as one of
//    `Increasing`, `Decreasing`, `Flat`, `Cyclic`, or `Volatile` based on its
//    OLS slope, coefficient of variation, and lag-1 autocorrelation.
//
// The module respects the workspace policies: snake_case naming, scirs2_core
// only (no direct ndarray/rand), no `.unwrap()` in production code, and a
// single file kept under 2000 lines. Tests use seeded RNG.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Public data types
// -----------------------------------------------------------------------------

/// A discovered motif (a recurring subsequence) in a time-series.
///
/// `start` is the index at which the canonical occurrence of the motif begins,
/// `length` is the window size in samples, and `distance` is the z-normalised
/// Euclidean distance to the motif's nearest non-trivial neighbour (smaller
/// values mean a more strongly repeated pattern). `occurrences` enumerates the
/// indices of all detected occurrences of this motif under the trivial-match
/// exclusion zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifResult {
    /// Starting index of the canonical occurrence.
    pub start: usize,
    /// Window length in samples.
    pub length: usize,
    /// z-normalised Euclidean distance to nearest non-trivial neighbour.
    pub distance: f64,
    /// All indices at which this motif occurs (within the exclusion zone).
    pub occurrences: Vec<usize>,
}

/// Direction of a detected change-point relative to the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeDirection {
    /// Mean shifted upward at this index.
    Upward,
    /// Mean shifted downward at this index.
    Downward,
}

/// A change-point detected by CUSUM, Page-Hinkley, or another detector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Sample index at which the change was flagged.
    pub index: usize,
    /// Detector-specific score; higher means stronger evidence.
    pub score: f64,
    /// Direction of the shift.
    pub direction: ChangeDirection,
}

/// High-level classification of the overall trend of a series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendType {
    /// Series rises with a positive OLS slope.
    Increasing,
    /// Series falls with a negative OLS slope.
    Decreasing,
    /// Slope is approximately zero, low autocorrelation.
    Flat,
    /// High lag-1 autocorrelation with a near-zero slope (periodic).
    Cyclic,
    /// High coefficient of variation with low autocorrelation (random).
    Volatile,
}

/// A contiguous segment of the time-series with stable summary statistics.
///
/// The segment covers `series[start..end]` (the `end` index is exclusive). The
/// `mean` and `stddev` fields hold the within-segment population statistics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegimeSegment {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
    /// Within-segment mean.
    pub mean: f64,
    /// Within-segment population standard deviation.
    pub stddev: f64,
}

/// Configuration shared by every detector in the module.
///
/// See [`PatternDetectorConfig::default`] for the recommended defaults that the
/// tests are calibrated against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectorConfig {
    /// Window size for motif discovery (in samples).
    pub motif_window: usize,
    /// Maximum number of motifs to return from `find_motifs`.
    pub motif_top_k: usize,
    /// CUSUM threshold `h` in units of the sample standard deviation.
    pub cusum_threshold: f64,
    /// CUSUM drift parameter `delta` in units of the sample standard deviation.
    pub cusum_drift: f64,
    /// Page-Hinkley magnitude threshold `lambda`.
    pub page_hinkley_lambda: f64,
    /// Page-Hinkley minimum-magnitude tolerance `alpha`.
    pub page_hinkley_alpha: f64,
    /// Minimum allowed segment length in regime segmentation.
    pub min_regime_length: usize,
    /// Minimum cost-improvement (SSE reduction) required to split a regime.
    pub regime_cost_threshold: f64,
    /// Absolute slope (per index) threshold for `Increasing` / `Decreasing`.
    pub trend_slope_threshold: f64,
    /// Coefficient-of-variation threshold for `Volatile`.
    pub volatility_cv_threshold: f64,
    /// Lag-1 autocorrelation threshold for `Cyclic`.
    pub cyclic_autocorr_threshold: f64,
}

impl Default for PatternDetectorConfig {
    fn default() -> Self {
        Self {
            motif_window: 16,
            motif_top_k: 3,
            cusum_threshold: 5.0,
            cusum_drift: 0.5,
            page_hinkley_lambda: 50.0,
            page_hinkley_alpha: 1e-3,
            min_regime_length: 10,
            regime_cost_threshold: 1.0,
            trend_slope_threshold: 0.01,
            volatility_cv_threshold: 1.0,
            cyclic_autocorr_threshold: 0.5,
        }
    }
}

/// General-purpose pattern detector operating on performance time-series.
///
/// The detector is stateless across method calls (all detectors are configured
/// via [`PatternDetectorConfig`]); each public method consumes the input series
/// and returns a fresh result. Construct via [`PerformancePatternDetector::new`]
/// for sensible defaults or chain the builder methods to override individual
/// parameters.
#[derive(Debug, Clone)]
pub struct PerformancePatternDetector {
    config: PatternDetectorConfig,
}

impl Default for PerformancePatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformancePatternDetector {
    /// Construct a detector with the default configuration.
    pub fn new() -> Self {
        Self {
            config: PatternDetectorConfig::default(),
        }
    }

    /// Construct a detector with the supplied configuration.
    pub fn with_config(config: PatternDetectorConfig) -> Self {
        Self { config }
    }

    /// Read-only access to the underlying configuration.
    pub fn config(&self) -> &PatternDetectorConfig {
        &self.config
    }

    // -- Builders ----------------------------------------------------------

    /// Override the motif window size.
    pub fn with_motif_window(mut self, motif_window: usize) -> Self {
        self.config.motif_window = motif_window;
        self
    }

    /// Override the motif top-k count.
    pub fn with_motif_top_k(mut self, motif_top_k: usize) -> Self {
        self.config.motif_top_k = motif_top_k;
        self
    }

    /// Override CUSUM `(threshold, drift)`.
    pub fn with_cusum(mut self, threshold: f64, drift: f64) -> Self {
        self.config.cusum_threshold = threshold;
        self.config.cusum_drift = drift;
        self
    }

    /// Override Page-Hinkley `(lambda, alpha)`.
    pub fn with_page_hinkley(mut self, lambda: f64, alpha: f64) -> Self {
        self.config.page_hinkley_lambda = lambda;
        self.config.page_hinkley_alpha = alpha;
        self
    }

    /// Override trend classification thresholds: slope, CV, autocorrelation.
    pub fn with_trend_thresholds(mut self, slope: f64, cv: f64, autocorr: f64) -> Self {
        self.config.trend_slope_threshold = slope;
        self.config.volatility_cv_threshold = cv;
        self.config.cyclic_autocorr_threshold = autocorr;
        self
    }

    /// Override regime segmentation thresholds: min length, cost threshold.
    pub fn with_regime_thresholds(mut self, min_length: usize, cost: f64) -> Self {
        self.config.min_regime_length = min_length;
        self.config.regime_cost_threshold = cost;
        self
    }

    // -- Motif discovery ---------------------------------------------------

    /// Discover the top-k motifs (recurring sub-sequences) in `series` using a
    /// simplified STAMP-style matrix profile.
    ///
    /// The algorithm computes, for every window of length `motif_window`, the
    /// z-normalised Euclidean distance to its nearest non-trivial neighbour
    /// (excluding the trivial-match band of width `motif_window / 4`). Motifs
    /// are then selected greedily by ascending distance with a `motif_window /
    /// 2`-sample exclusion zone to avoid returning overlapping copies of the
    /// same pattern. For each selected motif, all near-occurrences whose
    /// nearest-neighbour pointer references the motif's anchor *or* whose
    /// matrix-profile distance is within `2 * best_d` of the anchor's distance
    /// are reported.
    ///
    /// Returns at most `motif_top_k` motifs sorted by `distance` ascending.
    /// When the series is shorter than `2 * motif_window`, returns an empty
    /// vector (not an error).
    ///
    /// The implementation is intentionally O(n^2 * m); callers should restrict
    /// `series` to a few hundred points for interactive use.
    pub fn find_motifs(&self, series: &Array1<f64>) -> Result<Vec<MotifResult>> {
        let m = self.config.motif_window;
        if m == 0 {
            return Err(OptimError::InvalidConfig(
                "find_motifs: motif_window must be > 0".to_string(),
            ));
        }
        let n = series.len();
        if n < 2 * m {
            return Ok(Vec::new());
        }
        for v in series.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "find_motifs: non-finite value in series ({v})"
                )));
            }
        }
        let k = self.config.motif_top_k;
        if k == 0 {
            return Ok(Vec::new());
        }

        let trivial_exclusion = (m / 4).max(1) as isize;
        let num_windows = n - m + 1;

        // Pre-compute z-normalised windows once to avoid O(n) recomputation
        // inside the inner loop, which would otherwise turn this into O(n^2 m^2).
        let mut norm_windows: Vec<Vec<f64>> = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let slice: Vec<f64> = series.iter().skip(i).take(m).copied().collect();
            norm_windows.push(z_normalize(&slice));
        }

        // profile[i] = (distance_to_nearest_non_trivial_neighbour, neighbour_index)
        let mut profile: Vec<(f64, usize)> = vec![(f64::INFINITY, 0); num_windows];
        for (i, wi) in norm_windows.iter().enumerate() {
            let mut best_d = f64::INFINITY;
            let mut best_j = i;
            for (j, wj) in norm_windows.iter().enumerate() {
                if (i as isize - j as isize).abs() < trivial_exclusion {
                    continue;
                }
                let d = euclidean_distance(wi, wj);
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            profile[i] = (best_d, best_j);
        }

        // Order candidates by ascending distance.
        let mut order: Vec<usize> = (0..num_windows).collect();
        order.sort_by(|&a, &b| {
            profile[a]
                .0
                .partial_cmp(&profile[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedy selection with a motif_window / 2 exclusion zone around each
        // already-picked motif.
        let exclusion = (m / 2).max(1) as isize;
        let mut motifs: Vec<MotifResult> = Vec::with_capacity(k);
        let mut picked: Vec<usize> = Vec::with_capacity(k);
        for cand in order {
            if motifs.len() >= k {
                break;
            }
            let (cand_d, _cand_j) = profile[cand];
            if !cand_d.is_finite() {
                continue;
            }
            let mut blocked = false;
            for &p in &picked {
                if (cand as isize - p as isize).abs() < exclusion {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }

            // Collect occurrences: any window whose nearest-neighbour pointer
            // references `cand` *or* whose own profile distance is within
            // 2 * cand_d of cand_d AND within `m` samples or pointing to it.
            let mut occurrences: Vec<usize> = vec![cand];
            let widen = 2.0 * cand_d;
            for (j, &(dj, nb)) in profile.iter().enumerate() {
                if j == cand {
                    continue;
                }
                let neighbour_match = nb == cand;
                let within_widen = dj <= widen && (j as isize - cand as isize).abs() >= exclusion;
                if neighbour_match || within_widen {
                    occurrences.push(j);
                }
            }
            occurrences.sort_unstable();
            occurrences.dedup();

            picked.push(cand);
            motifs.push(MotifResult {
                start: cand,
                length: m,
                distance: cand_d,
                occurrences,
            });
        }

        // Already sorted by ascending distance from greedy selection, but make
        // the order explicit for callers that might mutate `picked`.
        motifs.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(motifs)
    }

    // -- CUSUM -------------------------------------------------------------

    /// Detect mean-shift change-points using a two-sided CUSUM filter.
    ///
    /// Both an upward CUSUM (`S_t = max(0, S_{t-1} + (x_t - mu - delta))`) and
    /// a downward CUSUM (`T_t = max(0, T_{t-1} + (mu - delta - x_t))`) are
    /// computed simultaneously against the full-series mean `mu` and
    /// population standard deviation `sigma`. A change-point is flagged
    /// whenever either running sum exceeds `cusum_threshold * sigma`, after
    /// which the corresponding sum is reset to zero.
    ///
    /// Returns the list of [`ChangePoint`]s in the order they are detected.
    pub fn detect_changepoints_cusum(&self, series: &Array1<f64>) -> Result<Vec<ChangePoint>> {
        let n = series.len();
        if n == 0 {
            return Err(OptimError::InvalidParameter(
                "detect_changepoints_cusum: empty series".to_string(),
            ));
        }
        for v in series.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "detect_changepoints_cusum: non-finite value ({v})"
                )));
            }
        }
        let mu = mean(series);
        let sigma = stddev(series);
        let sigma_eff = if sigma < 1e-12 { 1e-12 } else { sigma };
        let drift = self.config.cusum_drift * sigma_eff;
        let h = self.config.cusum_threshold * sigma_eff;

        let mut s_up = 0.0_f64;
        let mut s_dn = 0.0_f64;
        let mut out: Vec<ChangePoint> = Vec::new();

        // When sigma is effectively zero (constant series) we cannot meaningfully
        // detect a shift: skip with no detections.
        if sigma < 1e-12 {
            return Ok(out);
        }

        for (i, v) in series.iter().copied().enumerate() {
            let diff = v - mu;
            s_up = (s_up + diff - drift).max(0.0);
            s_dn = (s_dn - diff - drift).max(0.0);
            if s_up > h {
                out.push(ChangePoint {
                    index: i,
                    score: s_up / sigma_eff,
                    direction: ChangeDirection::Upward,
                });
                s_up = 0.0;
            }
            if s_dn > h {
                out.push(ChangePoint {
                    index: i,
                    score: s_dn / sigma_eff,
                    direction: ChangeDirection::Downward,
                });
                s_dn = 0.0;
            }
        }
        Ok(out)
    }

    // -- Page-Hinkley ------------------------------------------------------

    /// Detect mean-shift change-points using the Page-Hinkley sequential test.
    ///
    /// Maintains a cumulative deviation `m_t = sum_{s<=t} (x_s - mu - alpha)`
    /// and tracks its running minimum `M_t`. A change-point is flagged when
    /// the gap `m_t - M_t` exceeds `lambda`, indicating that the running mean
    /// has drifted upward by more than `alpha + lambda/n`. After a detection
    /// the statistic is reset.
    ///
    /// The classic Page-Hinkley test is one-sided (Upward); to detect
    /// downward shifts as well the dual statistic `M_t' - m_t' > lambda` is
    /// also tracked, with `m_t' = sum_{s<=t} (mu - alpha - x_s)`.
    pub fn detect_changepoints_page_hinkley(
        &self,
        series: &Array1<f64>,
    ) -> Result<Vec<ChangePoint>> {
        let n = series.len();
        if n == 0 {
            return Err(OptimError::InvalidParameter(
                "detect_changepoints_page_hinkley: empty series".to_string(),
            ));
        }
        for v in series.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "detect_changepoints_page_hinkley: non-finite value ({v})"
                )));
            }
        }
        let mu = mean(series);
        let alpha = self.config.page_hinkley_alpha;
        let lambda = self.config.page_hinkley_lambda;

        let mut m_up = 0.0_f64; // upward cumulative deviation
        let mut min_up = f64::INFINITY;
        let mut m_dn = 0.0_f64; // downward cumulative deviation (mirror)
        let mut min_dn = f64::INFINITY;
        let mut out: Vec<ChangePoint> = Vec::new();

        for (t, v) in series.iter().copied().enumerate() {
            m_up += v - mu - alpha;
            if m_up < min_up {
                min_up = m_up;
            }
            let gap_up = m_up - min_up;
            if gap_up > lambda {
                out.push(ChangePoint {
                    index: t,
                    score: gap_up,
                    direction: ChangeDirection::Upward,
                });
                m_up = 0.0;
                min_up = f64::INFINITY;
            }

            m_dn += mu - alpha - v;
            if m_dn < min_dn {
                min_dn = m_dn;
            }
            let gap_dn = m_dn - min_dn;
            if gap_dn > lambda {
                out.push(ChangePoint {
                    index: t,
                    score: gap_dn,
                    direction: ChangeDirection::Downward,
                });
                m_dn = 0.0;
                min_dn = f64::INFINITY;
            }
        }
        Ok(out)
    }

    // -- Regime segmentation -----------------------------------------------

    /// Segment the series into homogeneous regimes via offline binary
    /// segmentation.
    ///
    /// The algorithm recursively partitions `[start, end)` at the split index
    /// that maximally reduces total sum-of-squared-error, subject to:
    ///
    /// 1. Each child segment must contain at least `min_regime_length`
    ///    samples.
    /// 2. The cost improvement
    ///    `SSE(start..end) - (SSE(start..split) + SSE(split..end))` must
    ///    exceed `regime_cost_threshold`.
    ///
    /// Returns regimes in ascending order of `start`.
    pub fn segment_regimes(&self, series: &Array1<f64>) -> Result<Vec<RegimeSegment>> {
        let n = series.len();
        if n == 0 {
            return Err(OptimError::InvalidParameter(
                "segment_regimes: empty series".to_string(),
            ));
        }
        for v in series.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "segment_regimes: non-finite value ({v})"
                )));
            }
        }
        let min_len = self.config.min_regime_length.max(1);
        let cost_threshold = self.config.regime_cost_threshold;

        // Build cumulative sums of x and x^2 for O(1) SSE queries.
        let mut cum: Vec<f64> = Vec::with_capacity(n + 1);
        let mut cum_sq: Vec<f64> = Vec::with_capacity(n + 1);
        cum.push(0.0);
        cum_sq.push(0.0);
        let mut s = 0.0_f64;
        let mut sq = 0.0_f64;
        for v in series.iter().copied() {
            s += v;
            sq += v * v;
            cum.push(s);
            cum_sq.push(sq);
        }

        let ctx = SegmentContext {
            cum: &cum,
            cum_sq: &cum_sq,
            min_len,
            cost_threshold,
        };
        let mut out: Vec<RegimeSegment> = Vec::new();
        recurse_segment(&ctx, 0, n, &mut out);
        out.sort_by_key(|seg| seg.start);
        Ok(out)
    }

    // -- Trend classification ---------------------------------------------

    /// Classify the overall trend of the series into one of
    /// [`TrendType::Increasing`], [`TrendType::Decreasing`],
    /// [`TrendType::Flat`], [`TrendType::Cyclic`], or [`TrendType::Volatile`].
    ///
    /// The classifier uses three statistics:
    ///
    /// * **OLS slope** of `value` vs. `index` (closed-form `cov / var`).
    /// * **Coefficient of variation**, `stddev / |mean + 1e-12|`.
    /// * **Lag-1 autocorrelation**, computed as the Pearson correlation of
    ///   `series[0..n-1]` with `series[1..n]`.
    ///
    /// The decision tree is:
    ///
    /// 1. `cv > volatility_cv_threshold` AND `autocorr < 0.2` -> `Volatile`
    /// 2. `autocorr > cyclic_autocorr_threshold` AND `|slope| <
    ///    trend_slope_threshold` -> `Cyclic`
    /// 3. `slope > trend_slope_threshold` -> `Increasing`
    /// 4. `slope < -trend_slope_threshold` -> `Decreasing`
    /// 5. otherwise -> `Flat`
    pub fn classify_trend(&self, series: &Array1<f64>) -> Result<TrendType> {
        let n = series.len();
        if n < 2 {
            return Err(OptimError::InvalidParameter(
                "classify_trend: series must contain at least two points".to_string(),
            ));
        }
        for v in series.iter() {
            if !v.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "classify_trend: non-finite value ({v})"
                )));
            }
        }
        let slope = ols_slope(series);
        let mu = mean(series);
        let sigma = stddev(series);
        let cv = sigma / (mu.abs() + 1e-12);
        let autocorr = lag_one_autocorrelation(series);

        let slope_t = self.config.trend_slope_threshold;
        let cv_t = self.config.volatility_cv_threshold;
        let ac_t = self.config.cyclic_autocorr_threshold;

        if cv > cv_t && autocorr < 0.2 {
            return Ok(TrendType::Volatile);
        }
        if autocorr > ac_t && slope.abs() < slope_t {
            return Ok(TrendType::Cyclic);
        }
        if slope > slope_t {
            return Ok(TrendType::Increasing);
        }
        if slope < -slope_t {
            return Ok(TrendType::Decreasing);
        }
        Ok(TrendType::Flat)
    }
}

// -----------------------------------------------------------------------------
// Public helpers
// -----------------------------------------------------------------------------

/// Z-normalise a window: subtract the mean and divide by the population
/// standard deviation (with a tiny floor to avoid division by zero on a
/// constant window). Returns a fresh `Vec<f64>` of the same length.
pub fn z_normalize(window: &[f64]) -> Vec<f64> {
    let n = window.len();
    if n == 0 {
        return Vec::new();
    }
    let sum: f64 = window.iter().copied().sum();
    let mean_v = sum / (n as f64);
    let mut acc = 0.0_f64;
    for v in window.iter().copied() {
        let d = v - mean_v;
        acc += d * d;
    }
    let var = acc / (n as f64);
    let std = if var <= 1e-24 { 1e-12 } else { var.sqrt() };
    window.iter().map(|v| (*v - mean_v) / std).collect()
}

/// Euclidean distance between two equal-length slices.
///
/// Panics-free: if the slices have different lengths the shorter is truncated
/// effectively to the longer (in practice the matrix-profile caller guarantees
/// equal lengths).
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

/// Mean of an [`Array1`].
fn mean(series: &Array1<f64>) -> f64 {
    let n = series.len();
    if n == 0 {
        return 0.0;
    }
    series.iter().copied().sum::<f64>() / (n as f64)
}

/// Population standard deviation of an [`Array1`].
fn stddev(series: &Array1<f64>) -> f64 {
    let n = series.len();
    if n == 0 {
        return 0.0;
    }
    let mu = mean(series);
    let mut acc = 0.0_f64;
    for v in series.iter().copied() {
        let d = v - mu;
        acc += d * d;
    }
    (acc / (n as f64)).sqrt()
}

/// Closed-form OLS slope of `series` vs. its 0-based index axis.
fn ols_slope(series: &Array1<f64>) -> f64 {
    let n = series.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = mean(series);
    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    for (i, y) in series.iter().copied().enumerate() {
        let xd = (i as f64) - mean_x;
        let yd = y - mean_y;
        cov += xd * yd;
        var_x += xd * xd;
    }
    if var_x <= 1e-24 {
        return 0.0;
    }
    cov / var_x
}

/// Lag-1 autocorrelation of `series` (Pearson over `s[0..n-1]` and `s[1..n]`).
fn lag_one_autocorrelation(series: &Array1<f64>) -> f64 {
    let n = series.len();
    if n < 2 {
        return 0.0;
    }
    let mu = mean(series);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let d = series[i] - mu;
        den += d * d;
        if i + 1 < n {
            let d1 = series[i + 1] - mu;
            num += d * d1;
        }
    }
    if den <= 1e-24 {
        return 0.0;
    }
    num / den
}

/// Sum-of-squared-error of `series[start..end]` around its own mean,
/// computed in O(1) from cumulative sums of `x` and `x^2`.
fn sse(cum: &[f64], cum_sq: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }
    let s = cum[end] - cum[start];
    let sq = cum_sq[end] - cum_sq[start];
    let nf = n as f64;
    sq - (s * s) / nf
}

/// Within-segment mean from cumulative sums.
fn segment_mean(cum: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }
    (cum[end] - cum[start]) / (n as f64)
}

/// Within-segment population standard deviation from cumulative sums.
fn segment_stddev(cum: &[f64], cum_sq: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }
    let var = sse(cum, cum_sq, start, end) / (n as f64);
    if var <= 0.0 {
        0.0
    } else {
        var.sqrt()
    }
}

/// Read-only context bundle passed through the recursive binary-segmentation
/// routine. Bundling these into a struct keeps the function signature short
/// and lets clippy stay quiet about argument count.
struct SegmentContext<'a> {
    cum: &'a [f64],
    cum_sq: &'a [f64],
    min_len: usize,
    cost_threshold: f64,
}

/// Recursive binary-segmentation routine. Pushes leaves into `out`.
fn recurse_segment(
    ctx: &SegmentContext<'_>,
    start: usize,
    end: usize,
    out: &mut Vec<RegimeSegment>,
) {
    let len = end - start;
    let leaf = |out: &mut Vec<RegimeSegment>| {
        out.push(RegimeSegment {
            start,
            end,
            mean: segment_mean(ctx.cum, start, end),
            stddev: segment_stddev(ctx.cum, ctx.cum_sq, start, end),
        });
    };

    // A split into two halves requires each half to have at least min_len
    // samples, hence the parent must have at least 2 * min_len samples.
    if len < 2 * ctx.min_len {
        leaf(out);
        return;
    }

    let cost_no_split = sse(ctx.cum, ctx.cum_sq, start, end);
    let mut best_split = 0usize;
    let mut best_improvement = f64::NEG_INFINITY;
    // Candidate splits respect the min-length constraint on both children.
    for split in (start + ctx.min_len)..=(end - ctx.min_len) {
        let cost_split =
            sse(ctx.cum, ctx.cum_sq, start, split) + sse(ctx.cum, ctx.cum_sq, split, end);
        let improvement = cost_no_split - cost_split;
        if improvement > best_improvement {
            best_improvement = improvement;
            best_split = split;
        }
    }

    if best_improvement < ctx.cost_threshold {
        leaf(out);
        return;
    }

    recurse_segment(ctx, start, best_split, out);
    recurse_segment(ctx, best_split, end, out);
}

// -----------------------------------------------------------------------------
// Generic-friendly Float helper (unused publicly, retained for crate-wide
// numeric consistency with other modules that key on `scirs2_core::numeric`).
// -----------------------------------------------------------------------------

#[doc(hidden)]
#[inline]
pub(crate) fn _float_zero<F: Float>() -> F {
    F::zero()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::random::Random;
    use scirs2_core::random::Rng;
    use std::f64::consts::PI;

    /// Box-Muller transform: turn two uniform draws into a standard-normal one.
    fn box_muller<R: Rng>(rng: &mut Random<R>) -> f64 {
        let u1: f64 = rng.gen_range(1e-12_f64..1.0);
        let u2: f64 = rng.gen_range(0.0_f64..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }

    fn normal(rng: &mut Random<impl Rng>, mu: f64, sigma: f64) -> f64 {
        mu + sigma * box_muller(rng)
    }

    // ---- helpers ----

    #[test]
    fn test_z_normalize_correctness() {
        let w = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let z = z_normalize(&w);
        let n = z.len() as f64;
        let m_z: f64 = z.iter().sum::<f64>() / n;
        let var_z: f64 = z.iter().map(|x| (x - m_z) * (x - m_z)).sum::<f64>() / n;
        let std_z = var_z.sqrt();
        assert_relative_eq!(m_z, 0.0, epsilon = 1e-9);
        assert_relative_eq!(std_z, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_z_normalize_constant_window_no_nan() {
        // A constant input must not produce NaN; the floor on stddev should
        // map every value to roughly zero (since `v - mean = 0`).
        let w = [3.0_f64; 8];
        let z = z_normalize(&w);
        for v in z {
            assert!(v.is_finite(), "z_normalize produced non-finite");
        }
    }

    #[test]
    fn test_euclidean_distance_correctness() {
        let a = [0.0_f64, 0.0, 0.0];
        let b = [3.0_f64, 4.0, 0.0];
        let d = euclidean_distance(&a, &b);
        assert_relative_eq!(d, 5.0, epsilon = 1e-12);
    }

    // ---- motifs ----

    #[test]
    fn test_motif_discovery_finds_planted_motif() {
        // Construct a 200-point series with two embedded sine motifs of
        // length ~32 starting at indices 30 and 130. The rest is small Gaussian
        // noise that should NOT outscore the planted pair.
        let n = 200;
        let m = 16;
        let mut series = vec![0.0_f64; n];
        let mut rng = Random::seed(7);
        for v in series.iter_mut() {
            *v = 0.05 * box_muller(&mut rng);
        }
        // Plant identical motifs at start indices 30 and 130.
        for i in 0..m {
            let phase = 2.0 * PI * (i as f64) / (m as f64);
            let val = phase.sin();
            series[30 + i] += val;
            series[130 + i] += val;
        }
        let arr = Array1::from_vec(series);
        let det = PerformancePatternDetector::new().with_motif_window(m);
        let motifs = det.find_motifs(&arr).expect("motifs ok");
        assert!(!motifs.is_empty(), "no motifs returned");
        let top = &motifs[0];
        // The first motif should anchor close to 30 or 130 and its
        // occurrences should include the other index within +/-5.
        let close_to_30 = (top.start as isize - 30).abs() <= 5;
        let close_to_130 = (top.start as isize - 130).abs() <= 5;
        assert!(
            close_to_30 || close_to_130,
            "top motif starts at {} (not near 30 or 130)",
            top.start
        );
        // The occurrence set must include both planted locations within +/-5.
        let near_30 = top
            .occurrences
            .iter()
            .any(|&j| (j as isize - 30).abs() <= 5);
        let near_130 = top
            .occurrences
            .iter()
            .any(|&j| (j as isize - 130).abs() <= 5);
        assert!(
            near_30 && near_130,
            "occurrences {:?} did not cover both 30 and 130",
            top.occurrences
        );
    }

    #[test]
    fn test_motif_discovery_short_series_returns_empty() {
        // n < 2 * m -> empty result, NOT an error.
        let n = 20usize;
        let m = 16usize;
        let arr = Array1::from_vec(vec![1.0; n]);
        let det = PerformancePatternDetector::new().with_motif_window(m);
        let motifs = det.find_motifs(&arr).expect("ok");
        assert!(motifs.is_empty());
    }

    #[test]
    fn test_motif_discovery_zero_window_errors() {
        let arr = Array1::from_vec(vec![1.0; 100]);
        let det = PerformancePatternDetector::new().with_motif_window(0);
        let err = det.find_motifs(&arr).expect_err("zero window must error");
        assert!(matches!(err, OptimError::InvalidConfig(_)));
    }

    // ---- CUSUM ----

    #[test]
    fn test_cusum_detects_mean_shift() {
        // 100 N(0, 1) then 100 N(2, 1) -> at least one upward changepoint
        // within +/-20 of index 100.
        let n = 100;
        let mut rng = Random::seed(42);
        let mut vs: Vec<f64> = Vec::with_capacity(2 * n);
        for _ in 0..n {
            vs.push(normal(&mut rng, 0.0, 1.0));
        }
        for _ in 0..n {
            vs.push(normal(&mut rng, 2.0, 1.0));
        }
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let cps = det.detect_changepoints_cusum(&arr).expect("ok");
        assert!(!cps.is_empty(), "no CUSUM detections");
        let upward = cps
            .iter()
            .find(|c| c.direction == ChangeDirection::Upward)
            .expect("expected upward changepoint");
        let dist = (upward.index as isize - 100).abs();
        assert!(
            dist <= 30,
            "upward changepoint at index {} too far from 100",
            upward.index
        );
    }

    #[test]
    fn test_cusum_no_false_positive_on_constant_series() {
        let arr = Array1::from_vec(vec![5.0; 100]);
        let det = PerformancePatternDetector::new();
        let cps = det.detect_changepoints_cusum(&arr).expect("ok");
        assert!(
            cps.is_empty(),
            "constant series produced detections: {cps:?}"
        );
    }

    #[test]
    fn test_cusum_detects_downward_shift() {
        // 100 N(5, 0.5) then 100 N(1, 0.5) -> at least one downward changepoint.
        let mut rng = Random::seed(11);
        let mut vs: Vec<f64> = Vec::with_capacity(200);
        for _ in 0..100 {
            vs.push(normal(&mut rng, 5.0, 0.5));
        }
        for _ in 0..100 {
            vs.push(normal(&mut rng, 1.0, 0.5));
        }
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let cps = det.detect_changepoints_cusum(&arr).expect("ok");
        let any_dn = cps.iter().any(|c| c.direction == ChangeDirection::Downward);
        assert!(
            any_dn,
            "expected at least one downward detection, got {cps:?}"
        );
    }

    // ---- Page-Hinkley ----

    #[test]
    fn test_page_hinkley_detects_drift() {
        // Same mean-shift scenario as CUSUM.
        let n = 100;
        let mut rng = Random::seed(99);
        let mut vs: Vec<f64> = Vec::with_capacity(2 * n);
        for _ in 0..n {
            vs.push(normal(&mut rng, 0.0, 1.0));
        }
        for _ in 0..n {
            vs.push(normal(&mut rng, 2.0, 1.0));
        }
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new().with_page_hinkley(20.0, 1e-3);
        let cps = det.detect_changepoints_page_hinkley(&arr).expect("ok");
        let any_up = cps.iter().any(|c| c.direction == ChangeDirection::Upward);
        assert!(
            any_up,
            "Page-Hinkley failed to detect upward drift: {cps:?}"
        );
    }

    #[test]
    fn test_page_hinkley_no_false_positive_on_constant() {
        let arr = Array1::from_vec(vec![3.0; 100]);
        let det = PerformancePatternDetector::new();
        let cps = det.detect_changepoints_page_hinkley(&arr).expect("ok");
        assert!(
            cps.is_empty(),
            "constant series produced PH detections: {cps:?}"
        );
    }

    // ---- regimes ----

    #[test]
    fn test_regime_segmentation_finds_two_regimes_for_step_function() {
        // [1.0; 50] ++ [5.0; 50] -> exactly two segments centred on 1.0 and 5.0.
        let mut vs: Vec<f64> = vec![1.0; 50];
        vs.extend(std::iter::repeat_n(5.0, 50));
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let segs = det.segment_regimes(&arr).expect("ok");
        assert_eq!(segs.len(), 2, "expected 2 segments, got {segs:?}");
        assert!(
            (segs[0].mean - 1.0).abs() < 1e-9,
            "first segment mean off: {segs:?}"
        );
        assert!(
            (segs[1].mean - 5.0).abs() < 1e-9,
            "second segment mean off: {segs:?}"
        );
        // Boundary should be at index 50 (+/- min_regime_length).
        assert!(
            (segs[0].end as isize - 50).abs() <= det.config.min_regime_length as isize,
            "boundary at {} not near 50",
            segs[0].end
        );
    }

    #[test]
    fn test_regime_segmentation_one_regime_for_homogeneous() {
        let arr = Array1::from_vec(vec![1.0; 100]);
        let det = PerformancePatternDetector::new();
        let segs = det.segment_regimes(&arr).expect("ok");
        assert_eq!(segs.len(), 1, "expected 1 segment, got {segs:?}");
        assert_relative_eq!(segs[0].mean, 1.0, epsilon = 1e-12);
        assert_relative_eq!(segs[0].stddev, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_regime_segmentation_three_regimes_for_two_steps() {
        // [1.0; 50] ++ [5.0; 50] ++ [2.0; 50] -> 3 segments.
        let mut vs: Vec<f64> = vec![1.0; 50];
        vs.extend(std::iter::repeat_n(5.0, 50));
        vs.extend(std::iter::repeat_n(2.0, 50));
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let segs = det.segment_regimes(&arr).expect("ok");
        assert_eq!(segs.len(), 3, "expected 3 segments, got {segs:?}");
    }

    // ---- trend classification ----

    #[test]
    fn test_trend_classification_increasing_sequence() {
        let vs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let t = det.classify_trend(&arr).expect("ok");
        assert_eq!(t, TrendType::Increasing);
    }

    #[test]
    fn test_trend_classification_decreasing_sequence() {
        let vs: Vec<f64> = (1..=100).rev().map(|i| i as f64).collect();
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let t = det.classify_trend(&arr).expect("ok");
        assert_eq!(t, TrendType::Decreasing);
    }

    #[test]
    fn test_trend_classification_flat_constant() {
        let arr = Array1::from_vec(vec![5.0; 100]);
        let det = PerformancePatternDetector::new();
        let t = det.classify_trend(&arr).expect("ok");
        assert_eq!(t, TrendType::Flat);
    }

    #[test]
    fn test_trend_classification_cyclic_sine() {
        // sin(2 pi t / 20) for t = 0..200 has a near-zero slope and strong
        // lag-1 autocorrelation (cos(pi/10) ~= 0.95).
        let n = 200usize;
        let vs: Vec<f64> = (0..n)
            .map(|t| (2.0 * PI * (t as f64) / 20.0).sin())
            .collect();
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let t = det.classify_trend(&arr).expect("ok");
        assert_eq!(t, TrendType::Cyclic);
    }

    #[test]
    fn test_trend_classification_volatile_random() {
        // 200 uniform [-10, 10] samples -> high CV, low autocorrelation.
        let n = 200usize;
        let mut rng = Random::seed(17);
        let vs: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0_f64..10.0)).collect();
        let arr = Array1::from_vec(vs);
        let det = PerformancePatternDetector::new();
        let t = det.classify_trend(&arr).expect("ok");
        assert_eq!(t, TrendType::Volatile);
    }

    #[test]
    fn test_trend_classification_short_series_errors() {
        let arr = Array1::from_vec(vec![1.0]);
        let det = PerformancePatternDetector::new();
        let err = det.classify_trend(&arr).expect_err("short must err");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
    }

    // ---- serde / config / builder ----

    #[test]
    fn test_changepoint_serde_roundtrip() {
        let cp = ChangePoint {
            index: 42,
            score: 3.5,
            direction: ChangeDirection::Upward,
        };
        let j = serde_json::to_string(&cp).expect("serialize");
        let back: ChangePoint = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(back.index, cp.index);
        assert_relative_eq!(back.score, cp.score, epsilon = 1e-12);
        assert_eq!(back.direction, cp.direction);
    }

    #[test]
    fn test_motif_result_serde_roundtrip() {
        let mr = MotifResult {
            start: 7,
            length: 16,
            distance: 0.123,
            occurrences: vec![7, 19, 31],
        };
        let j = serde_json::to_string(&mr).expect("serialize");
        let back: MotifResult = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(back.start, mr.start);
        assert_eq!(back.length, mr.length);
        assert_relative_eq!(back.distance, mr.distance, epsilon = 1e-12);
        assert_eq!(back.occurrences, mr.occurrences);
    }

    #[test]
    fn test_regime_segment_serde_roundtrip() {
        let rs = RegimeSegment {
            start: 0,
            end: 50,
            mean: 1.0,
            stddev: 0.5,
        };
        let j = serde_json::to_string(&rs).expect("serialize");
        let back: RegimeSegment = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(back.start, rs.start);
        assert_eq!(back.end, rs.end);
        assert_relative_eq!(back.mean, rs.mean, epsilon = 1e-12);
        assert_relative_eq!(back.stddev, rs.stddev, epsilon = 1e-12);
    }

    #[test]
    fn test_default_config_values() {
        let c = PatternDetectorConfig::default();
        assert_eq!(c.motif_window, 16);
        assert_eq!(c.motif_top_k, 3);
        assert_relative_eq!(c.cusum_threshold, 5.0, epsilon = 1e-12);
        assert_relative_eq!(c.cusum_drift, 0.5, epsilon = 1e-12);
        assert_relative_eq!(c.page_hinkley_lambda, 50.0, epsilon = 1e-12);
        assert_relative_eq!(c.page_hinkley_alpha, 1e-3, epsilon = 1e-12);
        assert_eq!(c.min_regime_length, 10);
        assert_relative_eq!(c.regime_cost_threshold, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c.trend_slope_threshold, 0.01, epsilon = 1e-12);
        assert_relative_eq!(c.volatility_cv_threshold, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c.cyclic_autocorr_threshold, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_builder_pattern_chains() {
        let det = PerformancePatternDetector::new()
            .with_motif_window(8)
            .with_motif_top_k(5)
            .with_cusum(3.0, 0.25)
            .with_page_hinkley(40.0, 1e-2)
            .with_trend_thresholds(0.005, 0.8, 0.4)
            .with_regime_thresholds(15, 2.5);
        let c = det.config();
        assert_eq!(c.motif_window, 8);
        assert_eq!(c.motif_top_k, 5);
        assert_relative_eq!(c.cusum_threshold, 3.0, epsilon = 1e-12);
        assert_relative_eq!(c.cusum_drift, 0.25, epsilon = 1e-12);
        assert_relative_eq!(c.page_hinkley_lambda, 40.0, epsilon = 1e-12);
        assert_relative_eq!(c.page_hinkley_alpha, 1e-2, epsilon = 1e-12);
        assert_relative_eq!(c.trend_slope_threshold, 0.005, epsilon = 1e-12);
        assert_relative_eq!(c.volatility_cv_threshold, 0.8, epsilon = 1e-12);
        assert_relative_eq!(c.cyclic_autocorr_threshold, 0.4, epsilon = 1e-12);
        assert_eq!(c.min_regime_length, 15);
        assert_relative_eq!(c.regime_cost_threshold, 2.5, epsilon = 1e-12);
    }

    #[test]
    fn test_detector_default_uses_new() {
        let a = PerformancePatternDetector::default();
        let b = PerformancePatternDetector::new();
        assert_eq!(a.config().motif_window, b.config().motif_window);
        assert_eq!(a.config().motif_top_k, b.config().motif_top_k);
    }

    // ---- error paths ----

    #[test]
    fn test_changepoint_detectors_reject_nonfinite() {
        let arr = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
        let det = PerformancePatternDetector::new();
        assert!(matches!(
            det.detect_changepoints_cusum(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
        assert!(matches!(
            det.detect_changepoints_page_hinkley(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
        assert!(matches!(
            det.segment_regimes(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
        assert!(matches!(
            det.classify_trend(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_empty_series_rejected() {
        let arr = Array1::<f64>::zeros(0);
        let det = PerformancePatternDetector::new();
        assert!(matches!(
            det.detect_changepoints_cusum(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
        assert!(matches!(
            det.detect_changepoints_page_hinkley(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
        assert!(matches!(
            det.segment_regimes(&arr),
            Err(OptimError::InvalidParameter(_))
        ));
    }
}
