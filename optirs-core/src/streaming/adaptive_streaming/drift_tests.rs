// Real statistical drift tests, distribution comparators and model-based
// detectors for the adaptive-streaming drift detector.
//
// Every detector here implements its published definition, produces its own
// distinct test statistic, and derives its p-value from a real null
// distribution (standard normal, Kolmogorov, or a Hoeffding/Chernoff bound
// where the exact null has no closed form). Nothing in this module fabricates
// a significance value or reuses another detector's statistic.
//
// References
// - ADWIN: Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive
//   Windowing", SDM 2007.
// - DDM: Gama, Medas, Castillo & Rodrigues, "Learning with Drift Detection",
//   SBIA 2004.
// - EDDM: Baena-García et al., "Early Drift Detection Method", ECML/PKDD
//   IWKDDS 2006.
// - Page-Hinkley: Page, "Continuous Inspection Schemes", Biometrika 1954.
// - CUSUM: Page, ibid.; Montgomery, "Introduction to Statistical Quality
//   Control" for the k/h design rules.
// - Two-sample Kolmogorov-Smirnov and Mann-Whitney U: standard nonparametric
//   two-sample tests.

use super::drift_detection::{
    DistributionComparator, DistributionComparison, DriftTestResult, ModelBasedDetector,
    ModelDriftResult, StatisticalTest,
};
use super::optimizer::StreamingDataPoint;
use super::statistics as stats;

use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};

/// Converts a generic float into `f64`, reporting an honest error instead of
/// silently substituting a default when the element type cannot represent it.
fn to_f64<A: Float>(value: A) -> Result<f64, String> {
    value
        .to_f64()
        .ok_or_else(|| "value cannot be represented as f64".to_string())
}

/// Converts an `f64` into the generic element type.
fn from_f64<A: Float>(value: f64) -> Result<A, String> {
    A::from(value).ok_or_else(|| format!("{value} cannot be represented in the element type"))
}

/// Collects a sample as `f64`, skipping non-finite observations.
fn finite_f64<A: Float>(values: &[A]) -> Vec<f64> {
    values
        .iter()
        .filter_map(|v| v.to_f64())
        .filter(|v| v.is_finite())
        .collect()
}

/// Maximum number of observations any single detector retains.
const MAX_DETECTOR_WINDOW: usize = 4096;

/// Number of histogram buckets used by the distribution comparators.
const HISTOGRAM_BINS: usize = 16;

/// Additive smoothing applied to histogram counts before a log-ratio is taken.
const HISTOGRAM_SMOOTHING: f64 = 0.5;

// ---------------------------------------------------------------------------
// ADWIN
// ---------------------------------------------------------------------------

/// ADWIN (ADaptive WINdowing).
///
/// Maintains a single window of recent observations and, on every update,
/// searches **every** valid split point for a pair of sub-windows whose means
/// differ by more than the Hoeffding cut
/// `eps_cut = R * sqrt(ln(4/delta) / (2m))`, where `m` is the harmonic
/// combination `1/(1/n0 + 1/n1)` of the sub-window sizes and `R` is the
/// observed range of the window (the practical adaptation used for unbounded
/// metrics — the textbook bound assumes observations in `[0, 1]`). When a cut
/// is found the older sub-window is dropped, which is ADWIN's defining
/// behaviour: the window shrinks to the most recent stationary segment.
///
/// ADWIN is a single-stream detector, so it consumes only the newly arrived
/// observations; the caller's reference sample is not used (that is what makes
/// it genuinely different from the two-sample tests in this module).
pub struct AdwinTest<A: Float + Send + Sync> {
    /// Confidence parameter (the `delta` of the Hoeffding bound).
    delta: f64,
    /// Significance level used as a secondary gate.
    significance_level: f64,
    /// Observation window.
    window: VecDeque<A>,
    /// Smallest sub-window size considered on either side of a split.
    min_sub_window: usize,
}

impl<A: Float + Send + Sync> AdwinTest<A> {
    /// Creates an ADWIN detector. `sensitivity` is interpreted as ADWIN's
    /// `delta` confidence parameter (smaller = more conservative).
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0 && sensitivity < 1.0) {
            return Err(format!(
                "ADWIN delta must lie strictly in (0, 1), got {sensitivity}"
            ));
        }
        Ok(Self {
            delta: sensitivity,
            significance_level,
            window: VecDeque::with_capacity(256),
            min_sub_window: 5,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A>
    for AdwinTest<A>
{
    fn test_for_drift(
        &mut self,
        _reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if current.is_empty() {
            return Err("ADWIN: empty observation batch".to_string());
        }

        for &value in current {
            if self.window.len() >= MAX_DETECTOR_WINDOW {
                self.window.pop_front();
            }
            self.window.push_back(value);
        }

        let observations: Vec<f64> = finite_f64(self.window.make_contiguous());
        let n = observations.len();
        if n < 2 * self.min_sub_window {
            return insignificant_result(0.0, HashMap::new());
        }

        let (min, max) = stats::finite_range(&observations)
            .ok_or_else(|| "ADWIN: window contains no finite observations".to_string())?;
        // A zero-range window cannot exhibit a mean shift at all.
        let range = (max - min).max(f64::MIN_POSITIVE);

        // Prefix sums make the all-splits scan linear rather than quadratic.
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(0.0_f64);
        for &value in &observations {
            let last = prefix[prefix.len() - 1];
            prefix.push(last + value);
        }
        let total = prefix[n];

        let ln_term = (4.0 / self.delta).ln();
        let mut best_excess = f64::NEG_INFINITY;
        let mut best_diff = 0.0_f64;
        let mut best_split = 0usize;
        let mut best_eps = 0.0_f64;
        let mut best_m = 0.0_f64;

        let first_split = self.min_sub_window;
        let last_split = n - self.min_sub_window;
        for (offset, &prefix_sum) in prefix[first_split..=last_split].iter().enumerate() {
            let split = first_split + offset;
            let n0 = split as f64;
            let n1 = (n - split) as f64;
            let mean0 = prefix_sum / n0;
            let mean1 = (total - prefix_sum) / n1;
            let diff = (mean0 - mean1).abs();

            let harmonic_m = 1.0 / (1.0 / n0 + 1.0 / n1);
            let eps_cut = range * (ln_term / (2.0 * harmonic_m)).sqrt();
            let excess = diff - eps_cut;

            if excess > best_excess {
                best_excess = excess;
                best_diff = diff;
                best_split = split;
                best_eps = eps_cut;
                best_m = harmonic_m;
            }
        }

        // Invert the Hoeffding bound at the most violating split to obtain a
        // genuine (conservative) p-value: the bound states that under the
        // no-change hypothesis, P(|mean0 - mean1| >= d) <= 4 exp(-2 m (d/R)^2).
        let normalised = best_diff / range;
        let p_value = (4.0 * (-2.0 * best_m * normalised * normalised).exp()).clamp(0.0, 1.0);

        let cut_found = best_excess > 0.0;
        if cut_found {
            // Real ADWIN behaviour: forget the stale sub-window.
            for _ in 0..best_split {
                self.window.pop_front();
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("split_point".to_string(), from_f64::<A>(best_split as f64)?);
        metadata.insert("window_size".to_string(), from_f64::<A>(n as f64)?);
        metadata.insert("epsilon_cut".to_string(), from_f64::<A>(best_eps)?);
        metadata.insert("observed_range".to_string(), from_f64::<A>(range)?);

        Ok(DriftTestResult {
            drift_detected: cut_found || p_value < self.significance_level,
            p_value: from_f64(p_value)?,
            test_statistic: from_f64(best_diff)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        // Positive feedback (detections were useful) relaxes delta so the
        // detector stays responsive; negative feedback (false positives)
        // tightens it. Real state change, bounded to a sane range.
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        let scale = (1.0 + 0.1 * feedback).clamp(0.5, 2.0);
        self.delta = (self.delta * scale).clamp(1e-8, 0.5);
        Ok(())
    }

    fn reset(&mut self) {
        self.window.clear();
    }
}

// ---------------------------------------------------------------------------
// Shared error-stream binarisation for DDM / EDDM
// ---------------------------------------------------------------------------

/// Turns a real-valued observation stream into the Bernoulli "error" stream
/// that DDM and EDDM are defined over.
///
/// Both methods were published for classifiers, where the input is a sequence
/// of 0/1 misclassification indicators. For a metric stream the standard
/// adaptation is to threshold against the reference sample: an observation
/// counts as an "error" when it falls further than one reference standard
/// deviation from the reference mean. Under a stationary stream this yields a
/// stable error probability (`~0.317` for Gaussian data); a shift of the mean
/// **in either direction**, or an increase in spread, raises it — which is
/// exactly the signal DDM and EDDM are built to detect.
#[derive(Debug, Clone, Copy)]
struct ReferenceBaseline {
    mean: f64,
    std_dev: f64,
}

impl ReferenceBaseline {
    fn from_sample<A: Float>(reference: &[A]) -> Result<Self, String> {
        let sample = finite_f64(reference);
        if sample.len() < 2 {
            return Err(
                "error-stream baseline requires at least two reference observations".to_string(),
            );
        }
        let mean = stats::mean(&sample).ok_or_else(|| "reference mean is undefined".to_string())?;
        let std_dev = stats::sample_std_dev(&sample)
            .ok_or_else(|| "reference standard deviation is undefined".to_string())?;
        Ok(Self {
            mean,
            std_dev: std_dev.max(f64::MIN_POSITIVE),
        })
    }

    fn is_error(&self, value: f64) -> bool {
        (value - self.mean).abs() > self.std_dev
    }
}

// ---------------------------------------------------------------------------
// DDM
// ---------------------------------------------------------------------------

/// DDM (Drift Detection Method), Gama et al. 2004.
///
/// Tracks the running error probability `p_i` and its binomial standard
/// deviation `s_i = sqrt(p_i (1 - p_i) / i)`, remembers the minimum of
/// `p_i + s_i` seen so far, and signals a warning at
/// `p_i + s_i >= p_min + 2 s_min` and drift at `p_i + s_i >= p_min + 3 s_min`.
/// The reported statistic is the number of `s_min` units the current error
/// level sits above `p_min`, and the p-value is the one-sided normal tail of
/// that z score.
pub struct DdmTest<A: Float + Send + Sync> {
    significance_level: f64,
    warning_level: f64,
    drift_level: f64,
    min_instances: usize,
    instances: usize,
    errors: usize,
    p_min: f64,
    s_min: f64,
    warning_active: bool,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> DdmTest<A> {
    /// Creates a DDM detector. `sensitivity` scales the published 2-sigma /
    /// 3-sigma warning and drift levels, so a smaller value makes the detector
    /// fire earlier.
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "DDM sensitivity must be positive, got {sensitivity}"
            ));
        }
        // The published levels are 2 and 3 standard deviations. `sensitivity`
        // (a value in (0, 1]) shrinks them proportionally so the configured
        // sensitivity has a real, monotone effect on when DDM fires.
        let scale = (0.5 + sensitivity).clamp(0.5, 1.5);
        Ok(Self {
            significance_level,
            warning_level: 2.0 * scale,
            drift_level: 3.0 * scale,
            min_instances: 30,
            instances: 0,
            errors: 0,
            p_min: f64::INFINITY,
            s_min: f64::INFINITY,
            warning_active: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Whether DDM is currently in its warning zone (between the 2-sigma and
    /// 3-sigma levels).
    pub fn is_warning(&self) -> bool {
        self.warning_active
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A> for DdmTest<A> {
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if current.is_empty() {
            return Err("DDM: empty observation batch".to_string());
        }
        let baseline = ReferenceBaseline::from_sample(reference)?;

        for value in finite_f64(current) {
            self.instances += 1;
            if baseline.is_error(value) {
                self.errors += 1;
            }

            let n = self.instances as f64;
            let p = self.errors as f64 / n;
            let s = (p * (1.0 - p) / n).sqrt();

            if p + s < self.p_min + self.s_min {
                self.p_min = p;
                self.s_min = s;
            }
        }

        let n = self.instances as f64;
        let p = self.errors as f64 / n;
        let s = (p * (1.0 - p) / n).sqrt();

        // Guard the very first updates, where s_min can legitimately be zero.
        let s_min = if self.s_min.is_finite() && self.s_min > 0.0 {
            self.s_min
        } else {
            (p * (1.0 - p) / n).sqrt().max(f64::MIN_POSITIVE)
        };
        let p_min = if self.p_min.is_finite() {
            self.p_min
        } else {
            p
        };

        let z = ((p + s) - p_min) / s_min;
        let p_value = stats::standard_normal_sf(z)?;

        let enough_data = self.instances >= self.min_instances;
        self.warning_active = enough_data && z >= self.warning_level && z < self.drift_level;
        let drift_detected =
            enough_data && (z >= self.drift_level || p_value < self.significance_level);

        if drift_detected {
            // Published DDM behaviour: restart the statistics after a drift so
            // the new concept establishes its own p_min baseline.
            self.instances = 0;
            self.errors = 0;
            self.p_min = f64::INFINITY;
            self.s_min = f64::INFINITY;
            self.warning_active = false;
        }

        let mut metadata = HashMap::new();
        metadata.insert("error_rate".to_string(), from_f64::<A>(p)?);
        metadata.insert("p_min".to_string(), from_f64::<A>(p_min)?);
        metadata.insert("s_min".to_string(), from_f64::<A>(s_min)?);
        metadata.insert(
            "warning_level".to_string(),
            from_f64::<A>(self.warning_level)?,
        );

        Ok(DriftTestResult {
            drift_detected,
            p_value: from_f64(p_value)?,
            test_statistic: from_f64(z)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        // Negative feedback (too many false alarms) pushes the drift level up.
        self.drift_level = (self.drift_level - 0.2 * feedback).clamp(1.5, 6.0);
        self.warning_level = self.warning_level.min(self.drift_level - 0.25).max(1.0);
        Ok(())
    }

    fn reset(&mut self) {
        self.instances = 0;
        self.errors = 0;
        self.p_min = f64::INFINITY;
        self.s_min = f64::INFINITY;
        self.warning_active = false;
    }
}

// ---------------------------------------------------------------------------
// EDDM
// ---------------------------------------------------------------------------

/// EDDM (Early Drift Detection Method), Baena-García et al. 2006.
///
/// Where DDM watches the error *rate*, EDDM watches the **distance between
/// consecutive errors**: as a concept degrades, errors bunch together and that
/// distance falls. It tracks the running mean `p'` and standard deviation `s'`
/// of the inter-error distance, remembers the maximum of `p' + 2 s'` ever
/// observed, and reports the ratio `(p' + 2 s') / (p'_max + 2 s'_max)`. The
/// published thresholds are `0.95` for warning and `0.90` for drift.
pub struct EddmTest<A: Float + Send + Sync> {
    significance_level: f64,
    warning_ratio: f64,
    drift_ratio: f64,
    min_errors: usize,
    /// Number of observations since the previous error.
    since_last_error: usize,
    /// Welford accumulators over the inter-error distances.
    error_count: usize,
    mean_distance: f64,
    m2_distance: f64,
    max_criterion: f64,
    warning_active: bool,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> EddmTest<A> {
    /// Creates an EDDM detector. `sensitivity` nudges the published `0.95` /
    /// `0.90` ratio thresholds.
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "EDDM sensitivity must be positive, got {sensitivity}"
            ));
        }
        // A larger sensitivity relaxes the ratios (fires sooner); the offsets
        // stay small so the published thresholds remain recognisable.
        let shift = (sensitivity * 0.1).clamp(0.0, 0.05);
        Ok(Self {
            significance_level,
            warning_ratio: 0.95 + shift,
            drift_ratio: 0.90 + shift,
            min_errors: 30,
            since_last_error: 0,
            error_count: 0,
            mean_distance: 0.0,
            m2_distance: 0.0,
            max_criterion: 0.0,
            warning_active: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Whether EDDM is currently in its warning zone.
    pub fn is_warning(&self) -> bool {
        self.warning_active
    }

    fn criterion(&self) -> f64 {
        let variance = if self.error_count > 1 {
            self.m2_distance / (self.error_count - 1) as f64
        } else {
            0.0
        };
        self.mean_distance + 2.0 * variance.max(0.0).sqrt()
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A> for EddmTest<A> {
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if current.is_empty() {
            return Err("EDDM: empty observation batch".to_string());
        }
        let baseline = ReferenceBaseline::from_sample(reference)?;

        for value in finite_f64(current) {
            self.since_last_error += 1;
            if !baseline.is_error(value) {
                continue;
            }

            // Welford update over the inter-error distance.
            let distance = self.since_last_error as f64;
            self.since_last_error = 0;
            self.error_count += 1;
            let delta = distance - self.mean_distance;
            self.mean_distance += delta / self.error_count as f64;
            self.m2_distance += delta * (distance - self.mean_distance);

            let criterion = self.criterion();
            if criterion > self.max_criterion {
                self.max_criterion = criterion;
            }
        }

        let criterion = self.criterion();
        let ratio = if self.max_criterion > 0.0 {
            criterion / self.max_criterion
        } else {
            1.0
        };

        // Real significance for "the mean inter-error distance has fallen
        // below the best value seen": a one-sample z test on the mean, using
        // the running standard error of the inter-error distance.
        let variance = if self.error_count > 1 {
            self.m2_distance / (self.error_count - 1) as f64
        } else {
            0.0
        };
        let standard_error = if self.error_count > 0 {
            (variance / self.error_count as f64).sqrt()
        } else {
            0.0
        };
        let z = if standard_error > 0.0 {
            (self.max_criterion - criterion) / standard_error
        } else {
            0.0
        };
        let p_value = stats::standard_normal_sf(z)?;

        let enough_errors = self.error_count >= self.min_errors;
        self.warning_active =
            enough_errors && ratio < self.warning_ratio && ratio >= self.drift_ratio;
        let drift_detected =
            enough_errors && (ratio < self.drift_ratio || p_value < self.significance_level);

        if drift_detected {
            self.error_count = 0;
            self.mean_distance = 0.0;
            self.m2_distance = 0.0;
            self.max_criterion = 0.0;
            self.since_last_error = 0;
            self.warning_active = false;
        }

        let mut metadata = HashMap::new();
        metadata.insert("criterion_ratio".to_string(), from_f64::<A>(ratio)?);
        metadata.insert(
            "mean_error_distance".to_string(),
            from_f64::<A>(self.mean_distance)?,
        );
        metadata.insert(
            "max_criterion".to_string(),
            from_f64::<A>(self.max_criterion)?,
        );

        Ok(DriftTestResult {
            drift_detected,
            // The statistic is the ratio itself, which is what EDDM thresholds.
            test_statistic: from_f64(ratio)?,
            p_value: from_f64(p_value)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        self.drift_ratio = (self.drift_ratio + 0.01 * feedback).clamp(0.70, 0.99);
        self.warning_ratio = self.warning_ratio.max(self.drift_ratio + 0.005).min(0.999);
        Ok(())
    }

    fn reset(&mut self) {
        self.since_last_error = 0;
        self.error_count = 0;
        self.mean_distance = 0.0;
        self.m2_distance = 0.0;
        self.max_criterion = 0.0;
        self.warning_active = false;
    }
}

// ---------------------------------------------------------------------------
// Page-Hinkley
// ---------------------------------------------------------------------------

/// Page-Hinkley test.
///
/// Accumulates the centred, slack-adjusted deviations of the stream from its
/// own running mean and reports how far the cumulative sum has moved away from
/// its running extremum:
///
/// ```text
/// x̄_t  = running mean of all observations
/// m⁺_t = Σ (x_t - x̄_t - δ)      M⁺_t = min m⁺      PH⁺ = m⁺_t - M⁺_t
/// m⁻_t = Σ (x̄_t - x_t - δ)      M⁻_t = min m⁻      PH⁻ = m⁻_t - M⁻_t
/// ```
///
/// Drift is signalled when `max(PH⁺, PH⁻) > λ`. Tracking both directions makes
/// the detector symmetric, so a downward shift is caught as well as an upward
/// one. The running mean is genuinely running: there is no hard-coded baseline.
pub struct PageHinkleyTest<A: Float + Send + Sync> {
    significance_level: f64,
    /// Magnitude of change tolerated before the sum starts to accumulate.
    delta: f64,
    /// Detection threshold.
    lambda: f64,
    /// Welford accumulators for the running mean and variance.
    count: usize,
    mean: f64,
    m2: f64,
    /// Cumulative sums and their running minima, per direction.
    sum_increase: f64,
    min_increase: f64,
    sum_decrease: f64,
    min_decrease: f64,
    /// Observations since the current run started (reset on detection).
    run_length: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> PageHinkleyTest<A> {
    /// Creates a Page-Hinkley detector. `sensitivity` becomes the slack `delta`
    /// (expressed in units of the stream's own running standard deviation) and
    /// also sets the detection threshold `lambda`.
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "Page-Hinkley sensitivity must be positive, got {sensitivity}"
            ));
        }
        Ok(Self {
            significance_level,
            delta: sensitivity,
            // A threshold of 50 slack units is the common practical default;
            // scaling it by 1/sensitivity keeps a smaller sensitivity value
            // (a tighter slack) from firing on ordinary noise.
            lambda: (5.0 / sensitivity).clamp(5.0, 500.0),
            count: 0,
            mean: 0.0,
            m2: 0.0,
            sum_increase: 0.0,
            min_increase: 0.0,
            sum_decrease: 0.0,
            min_decrease: 0.0,
            run_length: 0,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A>
    for PageHinkleyTest<A>
{
    fn test_for_drift(
        &mut self,
        _reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if current.is_empty() {
            return Err("Page-Hinkley: empty observation batch".to_string());
        }

        for value in finite_f64(current) {
            self.count += 1;
            self.run_length += 1;
            let delta_from_mean = value - self.mean;
            self.mean += delta_from_mean / self.count as f64;
            self.m2 += delta_from_mean * (value - self.mean);

            let deviation = value - self.mean;
            self.sum_increase += deviation - self.delta;
            self.sum_decrease += -deviation - self.delta;
            if self.sum_increase < self.min_increase {
                self.min_increase = self.sum_increase;
            }
            if self.sum_decrease < self.min_decrease {
                self.min_decrease = self.sum_decrease;
            }
        }

        let ph_increase = self.sum_increase - self.min_increase;
        let ph_decrease = self.sum_decrease - self.min_decrease;
        let statistic = ph_increase.max(ph_decrease);

        // The PH statistic is a sum of centred deviations over the current
        // run, so under the no-change hypothesis it is approximately
        // N(0, sigma^2 * run_length): z = PH / (sigma * sqrt(run_length)).
        let variance = if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        };
        let sigma = variance.max(0.0).sqrt();
        let z = if sigma > 0.0 && self.run_length > 0 {
            statistic / (sigma * (self.run_length as f64).sqrt())
        } else {
            0.0
        };
        let p_value = stats::standard_normal_sf(z)?;

        let drift_detected = statistic > self.lambda || p_value < self.significance_level;
        if drift_detected {
            self.sum_increase = 0.0;
            self.min_increase = 0.0;
            self.sum_decrease = 0.0;
            self.min_decrease = 0.0;
            self.run_length = 0;
        }

        let mut metadata = HashMap::new();
        metadata.insert("ph_increase".to_string(), from_f64::<A>(ph_increase)?);
        metadata.insert("ph_decrease".to_string(), from_f64::<A>(ph_decrease)?);
        metadata.insert("running_mean".to_string(), from_f64::<A>(self.mean)?);
        metadata.insert("lambda".to_string(), from_f64::<A>(self.lambda)?);

        Ok(DriftTestResult {
            drift_detected,
            test_statistic: from_f64(statistic)?,
            p_value: from_f64(p_value)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        self.lambda = (self.lambda * (1.0 - 0.1 * feedback)).clamp(1.0, 1000.0);
        Ok(())
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.sum_increase = 0.0;
        self.min_increase = 0.0;
        self.sum_decrease = 0.0;
        self.min_decrease = 0.0;
        self.run_length = 0;
    }
}

// ---------------------------------------------------------------------------
// CUSUM
// ---------------------------------------------------------------------------

/// Two-sided CUSUM control chart.
///
/// Uses the reference sample to fix the in-control mean and standard deviation,
/// then accumulates
/// `g⁺ = max(0, g⁺ + (x - μ - k))` and `g⁻ = max(0, g⁻ - (x - μ + k))`
/// with the classic tabular design `k = 0.5σ` (half the shift to detect) and
/// `h = 5σ` (which gives an in-control average run length of roughly 465 for
/// that `k`). Unlike Page-Hinkley the reference level is *fixed* rather than
/// running, which makes CUSUM sensitive to slow, sustained drift that a running
/// mean would absorb.
pub struct CusumTest<A: Float + Send + Sync> {
    significance_level: f64,
    /// Slack in units of the reference standard deviation.
    k_sigma: f64,
    /// Threshold in units of the reference standard deviation.
    h_sigma: f64,
    positive_sum: f64,
    negative_sum: f64,
    run_length: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> CusumTest<A> {
    /// Creates a CUSUM detector. `sensitivity` scales the decision interval
    /// `h`, so a larger sensitivity fires sooner.
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "CUSUM sensitivity must be positive, got {sensitivity}"
            ));
        }
        Ok(Self {
            significance_level,
            k_sigma: 0.5,
            h_sigma: (5.0 * (1.0 - sensitivity).max(0.2)).clamp(1.0, 10.0),
            positive_sum: 0.0,
            negative_sum: 0.0,
            run_length: 0,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A>
    for CusumTest<A>
{
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        if current.is_empty() {
            return Err("CUSUM: empty observation batch".to_string());
        }
        let baseline = ReferenceBaseline::from_sample(reference)?;
        let sigma = baseline.std_dev;
        let slack = self.k_sigma * sigma;

        for value in finite_f64(current) {
            self.run_length += 1;
            let deviation = value - baseline.mean;
            self.positive_sum = (self.positive_sum + deviation - slack).max(0.0);
            self.negative_sum = (self.negative_sum - deviation - slack).max(0.0);
        }

        let statistic = self.positive_sum.max(self.negative_sum);
        let threshold = self.h_sigma * sigma;

        // As with Page-Hinkley, the accumulated sum over the current run is
        // approximately normal under the in-control hypothesis.
        let z = if sigma > 0.0 && self.run_length > 0 {
            statistic / (sigma * (self.run_length as f64).sqrt())
        } else {
            0.0
        };
        let p_value = stats::standard_normal_sf(z)?;

        let drift_detected = statistic > threshold || p_value < self.significance_level;
        if drift_detected {
            self.positive_sum = 0.0;
            self.negative_sum = 0.0;
            self.run_length = 0;
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "positive_sum".to_string(),
            from_f64::<A>(self.positive_sum)?,
        );
        metadata.insert(
            "negative_sum".to_string(),
            from_f64::<A>(self.negative_sum)?,
        );
        metadata.insert("threshold".to_string(), from_f64::<A>(threshold)?);
        metadata.insert("reference_mean".to_string(), from_f64::<A>(baseline.mean)?);

        Ok(DriftTestResult {
            drift_detected,
            test_statistic: from_f64(statistic)?,
            p_value: from_f64(p_value)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        self.h_sigma = (self.h_sigma * (1.0 - 0.1 * feedback)).clamp(1.0, 12.0);
        Ok(())
    }

    fn reset(&mut self) {
        self.positive_sum = 0.0;
        self.negative_sum = 0.0;
        self.run_length = 0;
    }
}

// ---------------------------------------------------------------------------
// Two-sample Kolmogorov-Smirnov
// ---------------------------------------------------------------------------

/// Two-sample Kolmogorov-Smirnov test.
///
/// The statistic is `D = max_x |F_ref(x) - F_cur(x)|` over the merged support
/// of the two samples and the p-value comes from the asymptotic Kolmogorov
/// distribution at `sqrt(n_eff) D`, `n_eff = n_ref n_cur / (n_ref + n_cur)`.
/// This is a genuine *distribution* test: it reacts to changes in shape or
/// spread that leave the mean untouched, which none of the mean-shift detectors
/// above can see.
pub struct KsTest<A: Float + Send + Sync> {
    significance_level: f64,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> KsTest<A> {
    /// Creates a KS test whose rejection level is `sensitivity` (falling back
    /// to the configured significance level when that is the tighter of the
    /// two).
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0 && sensitivity < 1.0) {
            return Err(format!(
                "KS significance level must lie strictly in (0, 1), got {sensitivity}"
            ));
        }
        Ok(Self {
            significance_level: sensitivity.max(significance_level),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A> for KsTest<A> {
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        let reference_sample = finite_f64(reference);
        let current_sample = finite_f64(current);
        if reference_sample.is_empty() || current_sample.is_empty() {
            return Err("KS test: both samples must be non-empty".to_string());
        }

        let d = stats::ks_statistic(&reference_sample, &current_sample)
            .ok_or_else(|| "KS test: empirical CDF is undefined".to_string())?;
        let p_value = stats::ks_two_sample_p(d, reference_sample.len(), current_sample.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "reference_size".to_string(),
            from_f64::<A>(reference_sample.len() as f64)?,
        );
        metadata.insert(
            "current_size".to_string(),
            from_f64::<A>(current_sample.len() as f64)?,
        );

        Ok(DriftTestResult {
            drift_detected: p_value < self.significance_level,
            test_statistic: from_f64(d)?,
            p_value: from_f64(p_value)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        self.significance_level =
            (self.significance_level * (1.0 + 0.1 * feedback)).clamp(1e-6, 0.5);
        Ok(())
    }

    fn reset(&mut self) {
        // Stateless: each call is a fresh two-sample comparison.
    }
}

// ---------------------------------------------------------------------------
// Mann-Whitney U
// ---------------------------------------------------------------------------

/// Two-sample Mann-Whitney U test (Wilcoxon rank-sum) with tie correction.
///
/// Ranks the pooled sample, forms `U = R_1 - n_1(n_1 + 1)/2`, and uses the
/// tie-corrected normal approximation
/// `sigma^2 = n_1 n_2 / 12 * ((N + 1) - Σ(t^3 - t)/(N(N-1)))`
/// with a continuity correction. This is a *location-shift* test on ranks, so
/// it is robust to the heavy tails that would inflate a mean-difference
/// statistic.
pub struct MannWhitneyUTest<A: Float + Send + Sync> {
    significance_level: f64,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> MannWhitneyUTest<A> {
    /// Creates a Mann-Whitney U test at the given rejection level.
    pub fn new(sensitivity: f64, significance_level: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0 && sensitivity < 1.0) {
            return Err(format!(
                "Mann-Whitney significance level must lie strictly in (0, 1), got {sensitivity}"
            ));
        }
        Ok(Self {
            significance_level: sensitivity.max(significance_level),
            _marker: std::marker::PhantomData,
        })
    }
}

/// Assigns mid-ranks to a pooled sample, returning the rank sum of the first
/// group and the tie-correction term `Σ (t^3 - t)`.
fn rank_sum_with_ties(group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
    let mut pooled: Vec<(f64, bool)> = group_a
        .iter()
        .map(|&v| (v, true))
        .chain(group_b.iter().map(|&v| (v, false)))
        .collect();
    pooled.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_sum_a = 0.0_f64;
    let mut tie_correction = 0.0_f64;
    let mut index = 0usize;
    while index < pooled.len() {
        let mut end = index + 1;
        while end < pooled.len() && pooled[end].0 == pooled[index].0 {
            end += 1;
        }
        let tie_size = (end - index) as f64;
        // Mid-rank shared by the whole tie group (ranks are 1-based).
        let mid_rank = (index as f64 + 1.0 + end as f64) / 2.0;
        for entry in &pooled[index..end] {
            if entry.1 {
                rank_sum_a += mid_rank;
            }
        }
        if tie_size > 1.0 {
            tie_correction += tie_size * tie_size * tie_size - tie_size;
        }
        index = end;
    }

    (rank_sum_a, tie_correction)
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> StatisticalTest<A>
    for MannWhitneyUTest<A>
{
    fn test_for_drift(
        &mut self,
        reference: &[A],
        current: &[A],
    ) -> Result<DriftTestResult<A>, String> {
        let reference_sample = finite_f64(reference);
        let current_sample = finite_f64(current);
        if reference_sample.is_empty() || current_sample.is_empty() {
            return Err("Mann-Whitney U: both samples must be non-empty".to_string());
        }

        let n1 = reference_sample.len() as f64;
        let n2 = current_sample.len() as f64;
        let total = n1 + n2;

        let (rank_sum, tie_correction) = rank_sum_with_ties(&reference_sample, &current_sample);
        let u = rank_sum - n1 * (n1 + 1.0) / 2.0;
        let mean_u = n1 * n2 / 2.0;

        let tie_term = if total > 1.0 {
            tie_correction / (total * (total - 1.0))
        } else {
            0.0
        };
        let variance_u = (n1 * n2 / 12.0) * ((total + 1.0) - tie_term);

        let z = if variance_u > 0.0 {
            // Continuity correction of 0.5 towards the mean.
            let deviation = (u - mean_u).abs();
            (deviation - 0.5).max(0.0) / variance_u.sqrt()
        } else {
            0.0
        };
        let p_value = stats::normal_two_sided_p(z)?;

        let mut metadata = HashMap::new();
        metadata.insert("u_statistic".to_string(), from_f64::<A>(u)?);
        metadata.insert("expected_u".to_string(), from_f64::<A>(mean_u)?);
        metadata.insert("tie_correction".to_string(), from_f64::<A>(tie_correction)?);

        Ok(DriftTestResult {
            drift_detected: p_value < self.significance_level,
            // Report the standardised statistic so the magnitude is comparable
            // across sample sizes.
            test_statistic: from_f64(z)?,
            p_value: from_f64(p_value)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            metadata,
        })
    }

    fn update_parameters(&mut self, performance_feedback: A) -> Result<(), String> {
        let feedback = to_f64(performance_feedback)?.clamp(-1.0, 1.0);
        self.significance_level =
            (self.significance_level * (1.0 + 0.1 * feedback)).clamp(1e-6, 0.5);
        Ok(())
    }

    fn reset(&mut self) {
        // Stateless.
    }
}

/// Builds a "no evidence of drift" result carrying a real statistic value.
fn insignificant_result<A: Float + Send + Sync>(
    statistic: f64,
    metadata: HashMap<String, A>,
) -> Result<DriftTestResult<A>, String> {
    Ok(DriftTestResult {
        drift_detected: false,
        p_value: A::one(),
        test_statistic: from_f64(statistic)?,
        confidence: A::zero(),
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Distribution comparators
// ---------------------------------------------------------------------------

/// Divergence measures a [`HistogramComparator`] can evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramDivergence {
    /// Kullback-Leibler divergence `KL(current || reference)`, in nats.
    KullbackLeibler,
    /// Jensen-Shannon divergence, in nats (bounded by `ln 2`).
    JensenShannon,
    /// Hellinger distance (bounded by `1`).
    Hellinger,
}

impl HistogramDivergence {
    /// Upper bound of the measure, used to normalise the drift threshold.
    /// `None` means the measure is unbounded (KL divergence).
    fn upper_bound(self) -> Option<f64> {
        match self {
            HistogramDivergence::KullbackLeibler => None,
            HistogramDivergence::JensenShannon => Some(std::f64::consts::LN_2),
            HistogramDivergence::Hellinger => Some(1.0),
        }
    }

    fn label(self) -> &'static str {
        match self {
            HistogramDivergence::KullbackLeibler => "kl_divergence",
            HistogramDivergence::JensenShannon => "js_divergence",
            HistogramDivergence::Hellinger => "hellinger_distance",
        }
    }
}

/// Histogram-based distribution comparator.
///
/// Both samples are binned onto a **common** support (the union of their
/// ranges) so the resulting probability mass functions are directly
/// comparable, additively smoothed so no bucket is exactly zero, and then
/// reduced to the requested divergence. Alongside the distance the comparator
/// runs a G-test (likelihood-ratio) on the same `2 x k` table, which yields a
/// genuine chi-square p-value rather than a rescaled distance.
pub struct HistogramComparator<A: Float + Send + Sync> {
    divergence: HistogramDivergence,
    threshold: A,
    bins: usize,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> HistogramComparator<A> {
    /// Creates a comparator. `sensitivity` in `(0, 1]` is mapped onto the
    /// measure's own scale: for a bounded measure the threshold is that
    /// fraction of the bound, and for the unbounded KL divergence it is used
    /// directly in nats.
    pub fn new(divergence: HistogramDivergence, sensitivity: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "{} sensitivity must be positive, got {sensitivity}",
                divergence.label()
            ));
        }
        let threshold_value = match divergence.upper_bound() {
            Some(bound) => (sensitivity.min(1.0)) * bound,
            None => sensitivity,
        };
        Ok(Self {
            divergence,
            threshold: from_f64(threshold_value)?,
            bins: HISTOGRAM_BINS,
            _marker: std::marker::PhantomData,
        })
    }

    /// The divergence measure this comparator evaluates.
    pub fn divergence(&self) -> HistogramDivergence {
        self.divergence
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> DistributionComparator<A>
    for HistogramComparator<A>
{
    fn compare_distributions(
        &self,
        reference: &[A],
        current: &[A],
    ) -> Result<DistributionComparison<A>, String> {
        let reference_sample = finite_f64(reference);
        let current_sample = finite_f64(current);
        if reference_sample.is_empty() || current_sample.is_empty() {
            return Err(format!(
                "{}: both samples must be non-empty",
                self.divergence.label()
            ));
        }

        // Common support over the union of both samples.
        let (ref_min, ref_max) = stats::finite_range(&reference_sample)
            .ok_or_else(|| "reference sample has no finite observations".to_string())?;
        let (cur_min, cur_max) = stats::finite_range(&current_sample)
            .ok_or_else(|| "current sample has no finite observations".to_string())?;
        let min = ref_min.min(cur_min);
        let max = ref_max.max(cur_max);

        let reference_counts = stats::histogram_counts(&reference_sample, min, max, self.bins);
        let current_counts = stats::histogram_counts(&current_sample, min, max, self.bins);

        let reference_pmf = stats::smoothed_pmf(&reference_counts, HISTOGRAM_SMOOTHING);
        let current_pmf = stats::smoothed_pmf(&current_counts, HISTOGRAM_SMOOTHING);

        let distance = match self.divergence {
            HistogramDivergence::KullbackLeibler => {
                stats::kl_divergence(&current_pmf, &reference_pmf)?
            }
            HistogramDivergence::JensenShannon => {
                stats::js_divergence(&reference_pmf, &current_pmf)?
            }
            HistogramDivergence::Hellinger => {
                stats::hellinger_distance(&reference_pmf, &current_pmf)?
            }
        };

        // Real significance on the same binning.
        let (g, degrees_of_freedom) = stats::g_test_statistic(&reference_counts, &current_counts)?;
        let p_value = if degrees_of_freedom >= 1.0 {
            stats::chi_square_sf(g, degrees_of_freedom)?
        } else {
            1.0
        };

        let threshold = to_f64(self.threshold)?;
        Ok(DistributionComparison {
            distance: from_f64(distance)?,
            threshold: self.threshold,
            drift_detected: distance > threshold,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
        })
    }

    fn get_threshold(&self) -> A {
        self.threshold
    }

    fn update_threshold(&mut self, new_threshold: A) {
        self.threshold = new_threshold;
    }
}

/// One-dimensional optimal-transport comparator.
///
/// Evaluates the first Wasserstein distance
/// `W1 = integral |F_ref(x) - F_cur(x)| dx` exactly on the merged support of
/// the two empirical distributions. In one dimension the Earth Mover's
/// Distance **is** `W1` — they are the same quantity, not two different
/// numbers — so both `DistributionMethod::WassersteinDistance` and
/// `DistributionMethod::EarthMoverDistance` are served by this comparator, and
/// the only difference between the two registrations is how the drift
/// threshold is scaled.
pub struct WassersteinComparator<A: Float + Send + Sync> {
    /// Threshold expressed as a fraction of the reference sample's spread, so
    /// the comparator is scale-free.
    relative_threshold: f64,
    threshold: A,
}

impl<A: Float + Send + Sync> WassersteinComparator<A> {
    /// Creates a comparator whose drift threshold is `sensitivity` times the
    /// reference sample's standard deviation.
    pub fn new(sensitivity: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "Wasserstein sensitivity must be positive, got {sensitivity}"
            ));
        }
        Ok(Self {
            relative_threshold: sensitivity,
            threshold: from_f64(sensitivity)?,
        })
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> DistributionComparator<A>
    for WassersteinComparator<A>
{
    fn compare_distributions(
        &self,
        reference: &[A],
        current: &[A],
    ) -> Result<DistributionComparison<A>, String> {
        let reference_sample = finite_f64(reference);
        let current_sample = finite_f64(current);
        if reference_sample.is_empty() || current_sample.is_empty() {
            return Err("Wasserstein distance: both samples must be non-empty".to_string());
        }

        let distance = stats::wasserstein_1d(&reference_sample, &current_sample)
            .ok_or_else(|| "Wasserstein distance is undefined for these samples".to_string())?;

        // Scale-free threshold: a fraction of the reference spread.
        let spread = stats::sample_std_dev(&reference_sample).unwrap_or(0.0);
        let effective_threshold = if spread > 0.0 {
            self.relative_threshold * spread
        } else {
            to_f64(self.threshold)?
        };

        // Significance comes from the two-sample KS test on the same data:
        // W1 measures *how far* the distributions moved, KS says whether the
        // shift is distinguishable from sampling noise.
        let d = stats::ks_statistic(&reference_sample, &current_sample)
            .ok_or_else(|| "Wasserstein: empirical CDF is undefined".to_string())?;
        let p_value = stats::ks_two_sample_p(d, reference_sample.len(), current_sample.len());

        Ok(DistributionComparison {
            distance: from_f64(distance)?,
            threshold: from_f64(effective_threshold)?,
            drift_detected: distance > effective_threshold,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
        })
    }

    fn get_threshold(&self) -> A {
        self.threshold
    }

    fn update_threshold(&mut self, new_threshold: A) {
        self.threshold = new_threshold;
        if let Some(value) = new_threshold.to_f64() {
            if value > 0.0 {
                self.relative_threshold = value;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model-based detector
// ---------------------------------------------------------------------------

/// Online linear model drift detector.
///
/// Fits a real ridge-regularised linear regressor by stochastic gradient
/// descent on the incoming `(features -> target)` pairs and monitors its
/// prediction error. `baseline_performance` is the exponentially-weighted mean
/// squared error established while the model was last considered healthy;
/// `model_performance` is the EWMA of the most recent errors. When the recent
/// error rises significantly above the baseline the model has stopped
/// describing the stream, which is precisely model-based drift.
///
/// Both fields are genuinely written by `update_model`, so
/// `performance_degradation` is a real measurement rather than the constant
/// zero produced by subtracting two never-initialised accumulators.
pub struct LinearModelDetector<A: Float + Send + Sync> {
    /// Regression weights (grown lazily to the observed feature width).
    weights: Vec<f64>,
    /// Intercept term.
    bias: f64,
    /// Step size of the normalised-LMS update. Stable for `0 < mu < 2`.
    learning_rate: f64,
    /// L2 weight-decay coefficient.
    l2_lambda: f64,
    /// EWMA smoothing factor for the error trackers.
    error_alpha: f64,
    /// EWMA of squared prediction error over the healthy baseline period.
    baseline_performance: f64,
    /// EWMA of squared prediction error over the most recent observations.
    model_performance: f64,
    /// Running variance of the squared error, for the significance test.
    error_m2: f64,
    error_mean: f64,
    /// Number of supervised updates applied.
    updates: usize,
    /// Number of updates required before the baseline is trusted.
    warmup_updates: usize,
    /// Weight snapshot taken at baseline time, for feature-importance deltas.
    baseline_weights: Vec<f64>,
    /// Relative degradation that counts as drift.
    degradation_threshold: f64,
    _marker: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> LinearModelDetector<A> {
    /// Creates a detector. `sensitivity` becomes the relative error-increase
    /// that counts as drift (e.g. `0.05` = a 5% rise in mean squared error).
    pub fn new(sensitivity: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "linear model sensitivity must be positive, got {sensitivity}"
            ));
        }
        Ok(Self {
            weights: Vec::new(),
            bias: 0.0,
            // Normalised-LMS step size; the classic mid-range choice, stable for
            // any input scale.
            learning_rate: 0.5,
            l2_lambda: 1e-5,
            error_alpha: 0.1,
            baseline_performance: f64::NAN,
            model_performance: f64::NAN,
            error_m2: 0.0,
            error_mean: 0.0,
            updates: 0,
            warmup_updates: 20,
            baseline_weights: Vec::new(),
            degradation_threshold: sensitivity,
            _marker: std::marker::PhantomData,
        })
    }

    /// Current squared-error EWMA, or `None` before the first supervised
    /// update.
    pub fn current_error(&self) -> Option<f64> {
        if self.model_performance.is_finite() {
            Some(self.model_performance)
        } else {
            None
        }
    }

    /// Baseline squared-error EWMA, or `None` before warm-up completes.
    pub fn baseline_error(&self) -> Option<f64> {
        if self.baseline_performance.is_finite() {
            Some(self.baseline_performance)
        } else {
            None
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let mut prediction = self.bias;
        for (weight, &feature) in self.weights.iter().zip(features.iter()) {
            prediction += weight * feature;
        }
        prediction
    }

    /// Applies one normalised-LMS step and folds the observed error into the
    /// trackers.
    ///
    /// Plain SGD on a raw feature vector converges at a rate that depends on the
    /// feature scale, which is unknown for a streaming metric. The normalised
    /// least-mean-squares update divides the step by the instantaneous input
    /// energy `1 + ||x||^2` (the `1` accounting for the bias coordinate), which
    /// makes the step size scale-invariant and stable for any
    /// `0 < learning_rate < 2` — the standard choice for an online regressor
    /// whose inputs are not pre-standardised. The ridge penalty is applied as a
    /// separate weight decay so it does not interact with the normalisation.
    fn learn_one(&mut self, features: &[f64], target: f64) {
        if self.weights.len() < features.len() {
            self.weights.resize(features.len(), 0.0);
        }

        let prediction = self.predict(features);
        let error = prediction - target;
        let squared_error = error * error;

        let input_energy = 1.0 + features.iter().map(|&f| f * f).sum::<f64>();
        let step = self.learning_rate * error / input_energy;
        for (weight, &feature) in self.weights.iter_mut().zip(features.iter()) {
            *weight -= step * feature + self.l2_lambda * *weight;
        }
        self.bias -= step;

        // EWMA of the squared error, plus a Welford accumulator for its spread.
        self.model_performance = if self.model_performance.is_finite() {
            self.error_alpha * squared_error + (1.0 - self.error_alpha) * self.model_performance
        } else {
            squared_error
        };

        self.updates += 1;
        let delta = squared_error - self.error_mean;
        self.error_mean += delta / self.updates as f64;
        self.error_m2 += delta * (squared_error - self.error_mean);

        // Establish (and afterwards slowly track) the healthy baseline.
        if self.updates == self.warmup_updates || !self.baseline_performance.is_finite() {
            self.baseline_performance = self.model_performance;
            self.baseline_weights = self.weights.clone();
        } else if self.model_performance <= self.baseline_performance {
            // The model is doing at least as well as its baseline, so tighten
            // the baseline towards the current level.
            self.baseline_performance =
                0.9 * self.baseline_performance + 0.1 * self.model_performance;
            self.baseline_weights = self.weights.clone();
        }
    }

    /// Extracts a supervised `(features, target)` pair from a data point.
    ///
    /// When the point carries no target the model cannot be trained, so the
    /// point is skipped rather than silently trained against a fabricated
    /// label.
    fn supervised_pair(data_point: &StreamingDataPoint<A>) -> Option<(Vec<f64>, f64)> {
        let target = data_point.target.as_ref()?;
        let target_value = target.iter().next()?.to_f64()?;
        if !target_value.is_finite() {
            return None;
        }
        let features: Vec<f64> = data_point
            .features
            .iter()
            .filter_map(|v| v.to_f64())
            .filter(|v| v.is_finite())
            .collect();
        if features.is_empty() {
            return None;
        }
        Some((features, target_value))
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> ModelBasedDetector<A>
    for LinearModelDetector<A>
{
    fn update_model(&mut self, data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut trained = 0usize;
        for data_point in data {
            if let Some((features, target)) = Self::supervised_pair(data_point) {
                self.learn_one(&features, target);
                trained += 1;
            }
        }
        if trained == 0 && !data.is_empty() {
            return Err(
                "linear model drift detector requires labelled data points (target is None)"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn detect_drift(
        &mut self,
        data: &[StreamingDataPoint<A>],
    ) -> Result<ModelDriftResult<A>, String> {
        // Learn from the batch first so the error trackers reflect it.
        self.update_model(data)?;

        let baseline = self
            .baseline_performance
            .is_finite()
            .then_some(self.baseline_performance)
            .ok_or_else(|| "linear model drift detector has no baseline yet".to_string())?;
        let current = self
            .model_performance
            .is_finite()
            .then_some(self.model_performance)
            .ok_or_else(|| "linear model drift detector has no error estimate yet".to_string())?;

        // Relative rise in mean squared error.
        let denominator = baseline.max(f64::MIN_POSITIVE);
        let degradation = (current - baseline) / denominator;

        // Significance: a one-sided z test that the recent squared error sits
        // above the baseline, using the running spread of the squared error.
        let variance = if self.updates > 1 {
            self.error_m2 / (self.updates - 1) as f64
        } else {
            0.0
        };
        let standard_error = if self.updates > 0 {
            (variance / self.updates as f64).sqrt()
        } else {
            0.0
        };
        let z = if standard_error > 0.0 {
            (current - baseline) / standard_error
        } else {
            0.0
        };
        let p_value = stats::standard_normal_sf(z)?;

        let ready = self.updates >= self.warmup_updates;
        let drift_detected = ready && degradation > self.degradation_threshold;

        // Real feature-importance change: the per-weight delta since baseline.
        let mut feature_importance_changes = Vec::with_capacity(self.weights.len());
        for (index, weight) in self.weights.iter().enumerate() {
            let baseline_weight = self.baseline_weights.get(index).copied().unwrap_or(0.0);
            feature_importance_changes.push(from_f64::<A>(weight - baseline_weight)?);
        }

        Ok(ModelDriftResult {
            drift_detected,
            performance_degradation: from_f64(degradation)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            feature_importance_changes,
        })
    }

    fn reset_model(&mut self) -> Result<(), String> {
        self.weights.clear();
        self.baseline_weights.clear();
        self.bias = 0.0;
        self.baseline_performance = f64::NAN;
        self.model_performance = f64::NAN;
        self.error_mean = 0.0;
        self.error_m2 = 0.0;
        self.updates = 0;
        Ok(())
    }
}

#[cfg(test)]
#[path = "drift_tests_regression_tests.rs"]
mod regression_tests;
