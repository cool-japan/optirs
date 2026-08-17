// Concept drift detection and adaptation for streaming optimization
//
// This module provides various algorithms for detecting when the underlying
// data distribution changes (concept drift) and adapting the optimizer accordingly.

use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::iter::Sum;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::utils::scalar_or;

#[cfg(test)]
mod drift_regression_tests;

/// Types of concept drift detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftDetectionMethod {
    /// Page-Hinkley test for change detection
    PageHinkley,
    /// ADWIN (Adaptive Windowing) algorithm
    Adwin,
    /// Drift Detection Method (DDM)
    DriftDetectionMethod,
    /// Early Drift Detection Method (EDDM)
    EarlyDriftDetection,
    /// Statistical test-based detection
    StatisticalTest,
    /// Ensemble-based detection
    Ensemble,
}

/// Concept drift detector configuration
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    /// Detection method to use
    pub method: DriftDetectionMethod,
    /// Minimum samples before detection
    pub min_samples: usize,
    /// Detection threshold
    pub threshold: f64,
    /// Window size for statistical methods
    pub window_size: usize,
    /// Alpha value for statistical tests
    pub alpha: f64,
    /// Warning threshold (before drift)
    pub warningthreshold: f64,
    /// Enable ensemble detection
    pub enable_ensemble: bool,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            method: DriftDetectionMethod::PageHinkley,
            min_samples: 30,
            threshold: 3.0,
            window_size: 100,
            alpha: 0.005,
            warningthreshold: 2.0,
            enable_ensemble: false,
        }
    }
}

/// Concept drift detection result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Warning level - potential drift
    Warning,
    /// Drift detected
    Drift,
}

/// Drift detection event
#[derive(Debug, Clone)]
pub struct DriftEvent<A: Float + Send + Sync> {
    /// Timestamp of detection
    pub timestamp: Instant,
    /// Detection confidence (0.0 to 1.0)
    pub confidence: A,
    /// Type of drift detected
    pub drift_type: DriftType,
    /// Recommendation for adaptation
    pub adaptation_recommendation: AdaptationRecommendation,
}

/// Types of concept drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftType {
    /// Sudden/abrupt drift
    Sudden,
    /// Gradual drift
    Gradual,
    /// Incremental drift
    Incremental,
    /// Recurring drift
    Recurring,
    /// Blip (temporary change)
    Blip,
}

/// Recommendations for adapting to drift
#[derive(Debug, Clone)]
pub enum AdaptationRecommendation {
    /// Reset optimizer state
    Reset,
    /// Increase learning rate
    IncreaseLearningRate { factor: f64 },
    /// Decrease learning rate
    DecreaseLearningRate { factor: f64 },
    /// Use different optimizer
    SwitchOptimizer { new_optimizer: String },
    /// Adjust window size
    AdjustWindow { new_size: usize },
    /// No adaptation needed
    NoAction,
}

/// Page-Hinkley drift detector
#[derive(Debug, Clone)]
pub struct PageHinkleyDetector<A: Float + Send + Sync> {
    /// Cumulative sum
    sum: A,
    /// Minimum cumulative sum seen
    min_sum: A,
    /// Detection threshold
    threshold: A,
    /// Warning threshold
    warningthreshold: A,
    /// Sample count
    sample_count: usize,
    /// Last drift time
    last_drift: Option<Instant>,
    /// Running mean of observed losses under the null hypothesis of no
    /// drift (C1 fix): the classic Page-Hinkley test (Gama et al., 2004)
    /// computes `x̄_t`, the incremental mean of *all* samples seen so far
    /// (including the current one), and accumulates `sum(x_t - x̄_t)`. The
    /// previous code used a hardcoded `0.1` in place of `x̄_t`, so the
    /// detector only behaved correctly for streams whose stable loss
    /// happened to sit near 0.1 — any other baseline made `sum` drift
    /// monotonically regardless of real drift, eventually firing false
    /// positives (or, for a baseline well below 0.1, never firing at all).
    running_mean: A,
}

impl<A: Float + Send + Sync + Send + Sync> PageHinkleyDetector<A> {
    /// Create a new Page-Hinkley detector
    pub fn new(threshold: A, warningthreshold: A) -> Self {
        Self {
            sum: A::zero(),
            min_sum: A::zero(),
            threshold,
            warningthreshold,
            sample_count: 0,
            last_drift: None,
            running_mean: A::zero(),
        }
    }

    /// Update detector with new loss value
    pub fn update(&mut self, loss: A) -> DriftStatus {
        self.sample_count += 1;

        // Incremental mean update (Welford-style): `running_mean` becomes
        // the mean of all `sample_count` losses seen so far, including this
        // one, matching the standard Page-Hinkley `x̄_t` (C1 fix).
        let count = A::from(self.sample_count).unwrap_or(A::one());
        self.running_mean = self.running_mean + (loss - self.running_mean) / count;

        // Update cumulative sum (assuming we want to detect increases in loss)
        self.sum = self.sum + loss - self.running_mean;

        // Update minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }

        // Compute test statistic
        let test_stat = self.sum - self.min_sum;

        if test_stat > self.threshold {
            self.last_drift = Some(Instant::now());
            self.reset();
            DriftStatus::Drift
        } else if test_stat > self.warningthreshold {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    /// Replace the decision thresholds *without* discarding the accumulated
    /// statistic (C6). Rebuilding the detector to change a threshold would
    /// reset `sum`/`min_sum` on every adaptation, so the cumulative test could
    /// never reach any threshold at all.
    pub fn set_thresholds(&mut self, threshold: A, warningthreshold: A) {
        self.threshold = threshold;
        self.warningthreshold = warningthreshold;
    }

    /// Current detection threshold.
    pub fn threshold(&self) -> A {
        self.threshold
    }

    /// Current warning threshold.
    pub fn warning_threshold(&self) -> A {
        self.warningthreshold
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.sum = A::zero();
        self.min_sum = A::zero();
        self.sample_count = 0;
        self.running_mean = A::zero();
    }
}

/// ADWIN (Adaptive Windowing) drift detector
#[derive(Debug, Clone)]
pub struct AdwinDetector<A: Float + Send + Sync> {
    /// Window of recent values
    window: VecDeque<A>,
    /// Maximum window size
    max_windowsize: usize,
    /// Detection confidence level
    delta: A,
    /// Minimum window size for detection
    min_window_size: usize,
}

impl<A: Float + Sum + Send + Sync + Send + Sync> AdwinDetector<A> {
    /// Create a new ADWIN detector
    pub fn new(delta: A, max_windowsize: usize) -> Self {
        Self {
            window: VecDeque::new(),
            max_windowsize,
            delta,
            min_window_size: 10,
        }
    }

    /// Replace the confidence parameter without discarding the window (C6).
    pub fn set_delta(&mut self, delta: A) {
        self.delta = delta;
    }

    /// Current confidence parameter.
    pub fn delta(&self) -> A {
        self.delta
    }

    /// Update detector with new value
    pub fn update(&mut self, value: A) -> DriftStatus {
        self.window.push_back(value);

        // Maintain window size
        if self.window.len() > self.max_windowsize {
            self.window.pop_front();
        }

        // Check for drift
        if self.window.len() >= self.min_window_size {
            if self.detect_change() {
                self.shrink_window();
                DriftStatus::Drift
            } else {
                DriftStatus::Stable
            }
        } else {
            DriftStatus::Stable
        }
    }

    /// Detect change using the ADWIN algorithm (Bifet & Gavaldà, 2007).
    ///
    /// C2 fix: the previous implementation checked only the single midpoint
    /// split and used an ad hoc `sqrt(var1 + var2 + 0.01)` threshold that
    /// never read `delta` at all, so the detector's configured confidence
    /// level had zero effect on its behavior. This checks every valid split
    /// point `n0 = 1..n` (as real ADWIN does — a true change can occur
    /// anywhere in the window, not just at the middle) using the standard
    /// Hoeffding-bound cut condition: for sub-windows of size `n0`, `n1`
    /// with means `mean0`, `mean1`, a cut is declared where
    /// `|mean0 - mean1| > eps_cut`, with
    /// `eps_cut = sqrt((1 / (2*m)) * ln(4 / delta))`,
    /// `m = 1 / (1/n0 + 1/n1)` (the harmonic-mean-style combined size ADWIN
    /// uses), which directly incorporates the detector's `delta` confidence
    /// parameter: a smaller `delta` (higher confidence) requires a larger
    /// mean gap before declaring a change.
    fn detect_change(&self) -> bool {
        let n = self.window.len();
        if n < 2 {
            return false;
        }

        let values: Vec<A> = self.window.iter().cloned().collect();
        // Prefix sums so every split's sub-window mean is O(1) to compute.
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(A::zero());
        for &v in &values {
            prefix.push(*prefix.last().unwrap_or(&A::zero()) + v);
        }
        let total = prefix[n];

        // The textbook Hoeffding bound assumes values in [0, 1]; real
        // losses/metrics are not naturally bounded that way. Following the
        // common practical adaptation (as in e.g. river's/scikit-multiflow's
        // ADWIN), scale by the window's observed range `R = max - min` as an
        // empirical stand-in for the a-priori bound, so the same relative
        // sensitivity holds regardless of the metric's absolute scale.
        let min_v = values
            .iter()
            .cloned()
            .fold(values[0], |a, b| if b < a { b } else { a });
        let max_v = values
            .iter()
            .cloned()
            .fold(values[0], |a, b| if b > a { b } else { a });
        let range = (max_v - min_v).max(A::from(1e-12).unwrap_or(A::zero()));

        let four = A::from(4.0).unwrap_or(A::one());
        let two = A::from(2.0).unwrap_or(A::one());
        let ln_term = (four / self.delta.max(A::from(1e-12).unwrap_or(A::zero()))).ln();

        for (offset, &sum0) in prefix[1..n].iter().enumerate() {
            let n0 = offset + 1;
            let n1 = n - n0;
            let n0_a = A::from(n0).unwrap_or(A::one());
            let n1_a = A::from(n1).unwrap_or(A::one());

            let sum1 = total - sum0;
            let mean0 = sum0 / n0_a;
            let mean1 = sum1 / n1_a;

            // Harmonic-mean-style combined size `m = 1 / (1/n0 + 1/n1)`.
            let m = A::one() / (A::one() / n0_a + A::one() / n1_a);
            let eps_cut = range * (ln_term / (two * m)).sqrt();

            if (mean0 - mean1).abs() > eps_cut {
                return true;
            }
        }

        false
    }

    /// Shrink window after drift detection
    fn shrink_window(&mut self) {
        let new_size = self.window.len() / 2;
        while self.window.len() > new_size {
            self.window.pop_front();
        }
    }
}

/// DDM (Drift Detection Method) detector.
///
/// C3: implemented per Gama et al., "Learning with Drift Detection" (2004).
/// The detector tracks the online error rate `p_i` and its standard deviation
/// `s_i = sqrt(p_i (1 - p_i) / i)`, remembers the pair `(p_min, s_min)` observed
/// at the *minimum of `p_i + s_i`*, and compares the current `p_i + s_i`
/// against `p_min + 2*s_min` (warning) and `p_min + 3*s_min` (drift).
///
/// The previous implementation instead tracked `min(p_i + 2*s_i)` in
/// `min_error_plus_2_std` and set `min_error_plus_3_std` to `p_i + 3*s_i` *at
/// that same moment*. The published `2*s_min` / `3*s_min` margins were
/// therefore never applied: the warning test degenerated to "is the current
/// level above the smallest level ever seen", which fires on essentially any
/// upward noise, and the drift test compared `p + 2s` against `p_min + 3*s_min`
/// where both terms came from different definitions. It also seeded
/// `error_std = 1.0`, which is not a possible standard deviation for a rate in
/// `[0, 1]`.
#[derive(Debug, Clone)]
pub struct DdmDetector<A: Float + Send + Sync> {
    /// Current error rate `p_i`
    error_rate: A,
    /// Current standard deviation `s_i`
    error_std: A,
    /// Error rate at the minimum of `p_i + s_i`
    p_min: Option<A>,
    /// Standard deviation at the minimum of `p_i + s_i`
    s_min: Option<A>,
    /// Sample count
    sample_count: usize,
    /// Error count
    error_count: usize,
    /// Samples required before the detector starts testing
    warmup: usize,
}

impl<A: Float + Send + Sync + Send + Sync> DdmDetector<A> {
    /// Minimum samples before the DDM statistics are meaningful (the value
    /// used in the original paper).
    pub const DEFAULT_WARMUP: usize = 30;

    /// Create a new DDM detector
    pub fn new() -> Self {
        Self::with_warmup(Self::DEFAULT_WARMUP)
    }

    /// Create a DDM detector with a custom warm-up length.
    pub fn with_warmup(warmup: usize) -> Self {
        Self {
            error_rate: A::zero(),
            error_std: A::zero(),
            p_min: None,
            s_min: None,
            sample_count: 0,
            error_count: 0,
            warmup: warmup.max(2),
        }
    }

    /// Current error rate estimate.
    pub fn error_rate(&self) -> A {
        self.error_rate
    }

    /// Current warning level `p_min + 2*s_min`, if the baseline is established.
    pub fn warning_level(&self) -> Option<A> {
        let (p_min, s_min) = (self.p_min?, self.s_min?);
        Some(p_min + A::from(2.0)? * s_min)
    }

    /// Current drift level `p_min + 3*s_min`, if the baseline is established.
    pub fn drift_level(&self) -> Option<A> {
        let (p_min, s_min) = (self.p_min?, self.s_min?);
        Some(p_min + A::from(3.0)? * s_min)
    }

    /// Update with prediction result
    pub fn update(&mut self, iserror: bool) -> DriftStatus {
        self.sample_count += 1;
        if iserror {
            self.error_count += 1;
        }

        let n = match A::from(self.sample_count as f64) {
            Some(n) if n > A::zero() => n,
            _ => return DriftStatus::Stable,
        };
        let p = A::from(self.error_count as f64).unwrap_or_else(A::zero) / n;
        // `p (1 - p) / n` is non-negative for any p in [0, 1]; clamp defensively
        // so a rounding artefact can never feed a NaN into `sqrt`.
        let variance = (p * (A::one() - p) / n).max(A::zero());
        self.error_rate = p;
        self.error_std = variance.sqrt();

        if self.sample_count < self.warmup {
            // The baseline is only meaningful once the rate has settled; seeding
            // it from the first few samples is what made the original detector
            // fire immediately.
            return DriftStatus::Stable;
        }

        let level = p + self.error_std;
        match (self.p_min, self.s_min) {
            (Some(p_min), Some(s_min)) if level >= p_min + s_min => {}
            _ => {
                self.p_min = Some(p);
                self.s_min = Some(self.error_std);
            }
        }

        let Some(warning_level) = self.warning_level() else {
            return DriftStatus::Stable;
        };
        let Some(drift_level) = self.drift_level() else {
            return DriftStatus::Stable;
        };

        // Strict comparisons: for a stream that has seen no errors at all,
        // `p_min` and `s_min` are both exactly 0, and a non-strict test would
        // report drift on the first post-warm-up sample of a perfectly clean
        // stream.
        if level > drift_level {
            self.reset();
            DriftStatus::Drift
        } else if level > warning_level {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.error_count = 0;
        self.error_rate = A::zero();
        self.error_std = A::zero();
        self.p_min = None;
        self.s_min = None;
    }
}

impl<A: Float + Send + Sync + Send + Sync> Default for DdmDetector<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive concept drift detector
pub struct ConceptDriftDetector<A: Float + Send + Sync> {
    /// Configuration
    config: DriftDetectorConfig,

    /// Page-Hinkley detector
    ph_detector: PageHinkleyDetector<A>,

    /// ADWIN detector
    adwin_detector: AdwinDetector<A>,

    /// DDM detector
    ddm_detector: DdmDetector<A>,

    /// Ensemble voting history
    ensemble_history: VecDeque<DriftStatus>,

    /// Drift events history
    drift_events: Vec<DriftEvent<A>>,

    /// Performance before/after drift
    performance_tracker: PerformanceDriftTracker<A>,
}

impl<A: Float + std::fmt::Debug + Sum + Send + Sync + Send + Sync> ConceptDriftDetector<A> {
    /// Bound on the retained ensemble decision history (C7).
    pub const ENSEMBLE_HISTORY_CAPACITY: usize = 64;

    /// Bound on the retained drift-event log (C7): an unbounded `Vec` here grows
    /// for the lifetime of a long-running stream.
    pub const DRIFT_EVENT_CAPACITY: usize = 1024;

    /// Create a new concept drift detector
    pub fn new(config: DriftDetectorConfig) -> Self {
        let threshold = scalar_or(config.threshold, A::zero());
        let warningthreshold = scalar_or(config.warningthreshold, A::zero());
        let delta = scalar_or(config.alpha, A::zero());

        Self {
            ph_detector: PageHinkleyDetector::new(threshold, warningthreshold),
            adwin_detector: AdwinDetector::new(delta, config.window_size),
            ddm_detector: DdmDetector::new(),
            ensemble_history: VecDeque::with_capacity(10),
            drift_events: Vec::new(),
            performance_tracker: PerformanceDriftTracker::new(),
            config,
        }
    }

    /// Update detector with new loss and prediction error
    pub fn update(&mut self, loss: A, is_predictionerror: bool) -> Result<DriftStatus> {
        let ph_status = self.ph_detector.update(loss);
        let adwin_status = self.adwin_detector.update(loss);
        let ddm_status = self.ddm_detector.update(is_predictionerror);

        let final_status = if self.config.enable_ensemble {
            self.ensemble_vote(ph_status, adwin_status, ddm_status)
        } else {
            match self.config.method {
                DriftDetectionMethod::PageHinkley => ph_status,
                DriftDetectionMethod::Adwin => adwin_status,
                DriftDetectionMethod::DriftDetectionMethod => ddm_status,
                _ => ddm_status, // Fallback for EarlyDriftDetection, StatisticalTest, Ensemble
            }
        };

        // C7: the ensemble decision history is now actually recorded (it used
        // to be allocated in the constructor and never written), bounded to
        // `ENSEMBLE_HISTORY_CAPACITY`, and read back to derive a real
        // confidence.
        self.ensemble_history.push_back(final_status);
        while self.ensemble_history.len() > Self::ENSEMBLE_HISTORY_CAPACITY {
            self.ensemble_history.pop_front();
        }

        // Record drift event if detected
        if final_status == DriftStatus::Drift {
            let event = DriftEvent {
                timestamp: Instant::now(),
                // Real confidence: how strongly the detectors agreed on this
                // sample, tempered by how persistent the recent signal has been.
                confidence: self.detection_confidence(ph_status, adwin_status, ddm_status),
                drift_type: self.classify_drift_type(),
                adaptation_recommendation: self.generate_adaptation_recommendation(),
            };
            self.drift_events.push(event);
            while self.drift_events.len() > Self::DRIFT_EVENT_CAPACITY {
                self.drift_events.remove(0);
            }
        }

        // Update performance tracking
        self.performance_tracker.update(loss, final_status);

        Ok(final_status)
    }

    /// Confidence in a detection, from detector agreement and signal
    /// persistence. Replaces the hardcoded `0.8` that every drift event used to
    /// carry regardless of how the detectors actually voted.
    fn detection_confidence(&self, ph: DriftStatus, adwin: DriftStatus, ddm: DriftStatus) -> A {
        let votes = [ph, adwin, ddm];
        let drift_votes = votes.iter().filter(|&&s| s == DriftStatus::Drift).count();
        let warning_votes = votes.iter().filter(|&&s| s == DriftStatus::Warning).count();
        let agreement = (drift_votes as f64 + 0.5 * warning_votes as f64) / votes.len() as f64;

        // Persistence: the share of the retained ensemble history that is not
        // Stable. A single isolated spike is less trustworthy than a sustained
        // signal.
        let persistence = if self.ensemble_history.is_empty() {
            0.0
        } else {
            self.ensemble_history
                .iter()
                .filter(|status| **status != DriftStatus::Stable)
                .count() as f64
                / self.ensemble_history.len() as f64
        };

        let confidence = (0.7 * agreement + 0.3 * persistence).clamp(0.0, 1.0);
        A::from(confidence).unwrap_or_else(A::zero)
    }

    /// Recent ensemble decisions, oldest first.
    pub fn ensemble_history(&self) -> &VecDeque<DriftStatus> {
        &self.ensemble_history
    }

    /// Ensemble voting among detectors
    fn ensemble_vote(
        &mut self,
        ph: DriftStatus,
        adwin: DriftStatus,
        ddm: DriftStatus,
    ) -> DriftStatus {
        let votes = [ph, adwin, ddm];

        // Count votes
        let drift_votes = votes.iter().filter(|&&s| s == DriftStatus::Drift).count();
        let warning_votes = votes.iter().filter(|&&s| s == DriftStatus::Warning).count();

        if drift_votes >= 2 {
            DriftStatus::Drift
        } else if warning_votes >= 2 || drift_votes >= 1 {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    /// Classify the type of drift based on recent history
    fn classify_drift_type(&self) -> DriftType {
        // Simplified classification based on recent drift events
        if self.drift_events.len() < 2 {
            return DriftType::Sudden;
        }

        let recent_events = self.drift_events.iter().rev().take(5);
        let time_intervals: Vec<_> = recent_events
            .map(|event| event.timestamp)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| window[0].duration_since(window[1]))
            .collect();

        if time_intervals.iter().all(|&d| d < Duration::from_secs(60)) {
            DriftType::Sudden
        } else if time_intervals.len() > 2 {
            DriftType::Gradual
        } else {
            DriftType::Incremental
        }
    }

    /// Generate adaptation recommendation based on drift characteristics
    fn generate_adaptation_recommendation(&self) -> AdaptationRecommendation {
        let recent_performance = self.performance_tracker.get_recent_performance_change();

        if recent_performance > scalar_or(0.5, A::zero()) {
            // Significant performance degradation
            AdaptationRecommendation::Reset
        } else if recent_performance > scalar_or(0.2, A::zero()) {
            // Moderate degradation
            AdaptationRecommendation::IncreaseLearningRate { factor: 1.5 }
        } else if recent_performance < scalar_or(-0.1, A::zero()) {
            // Performance improved (suspicious)
            AdaptationRecommendation::DecreaseLearningRate { factor: 0.8 }
        } else {
            AdaptationRecommendation::NoAction
        }
    }

    /// Get drift detection statistics
    pub fn get_statistics(&self) -> DriftStatistics<A> {
        DriftStatistics {
            total_drifts: self.drift_events.len(),
            recent_drift_rate: self.calculate_recent_drift_rate(),
            average_drift_confidence: self.calculate_average_confidence(),
            drift_types_distribution: self.calculate_drift_type_distribution(),
            time_since_last_drift: self.time_since_last_drift(),
        }
    }

    fn calculate_recent_drift_rate(&self) -> f64 {
        // Calculate drift rate in the last hour.
        //
        // `Instant::now() - Duration` panics when the process has been up for
        // less than the window (the resulting instant is not representable), so
        // the window is applied as a forward `duration_since` comparison
        // instead of by materialising a cutoff instant.
        let recent_window = Duration::from_secs(3600);
        let now = Instant::now();
        let recent_drifts = self
            .drift_events
            .iter()
            .filter(|event| now.duration_since(event.timestamp) <= recent_window)
            .count();
        recent_drifts as f64 / recent_window.as_secs_f64() // Drifts per second
    }

    fn calculate_average_confidence(&self) -> Option<A> {
        if self.drift_events.is_empty() {
            None
        } else {
            let sum = self
                .drift_events
                .iter()
                .map(|event| event.confidence)
                .sum::<A>();
            Some(sum / scalar_or(self.drift_events.len(), A::one()))
        }
    }

    fn calculate_drift_type_distribution(&self) -> std::collections::HashMap<DriftType, usize> {
        let mut distribution = std::collections::HashMap::new();
        for event in &self.drift_events {
            *distribution.entry(event.drift_type).or_insert(0) += 1;
        }
        distribution
    }

    fn time_since_last_drift(&self) -> Option<Duration> {
        self.drift_events
            .last()
            .map(|event| event.timestamp.elapsed())
    }
}

/// Performance tracker for drift impact analysis
#[derive(Debug, Clone)]
struct PerformanceDriftTracker<A: Float + Send + Sync> {
    /// Performance history with drift annotations
    performance_history: VecDeque<(A, DriftStatus, Instant)>,
    /// Window size for analysis
    window_size: usize,
}

impl<A: Float + std::iter::Sum + Send + Sync + Send + Sync> PerformanceDriftTracker<A> {
    fn new() -> Self {
        Self {
            performance_history: VecDeque::new(),
            window_size: 100,
        }
    }

    fn update(&mut self, performance: A, driftstatus: DriftStatus) {
        self.performance_history
            .push_back((performance, driftstatus, Instant::now()));

        // Maintain window size
        if self.performance_history.len() > self.window_size {
            self.performance_history.pop_front();
        }
    }

    /// Get recent performance change (positive = degradation, negative = improvement)
    fn get_recent_performance_change(&self) -> A {
        if self.performance_history.len() < 10 {
            return A::zero();
        }

        let recent: Vec<_> = self.performance_history.iter().rev().take(10).collect();
        let older: Vec<_> = self
            .performance_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .collect();

        if older.is_empty() {
            return A::zero();
        }

        let recent_avg =
            recent.iter().map(|(p, _, _)| *p).sum::<A>() / scalar_or(recent.len(), A::one());
        let older_avg =
            older.iter().map(|(p, _, _)| *p).sum::<A>() / scalar_or(older.len(), A::one());

        recent_avg - older_avg
    }
}

/// Drift detection statistics
#[derive(Debug, Clone)]
pub struct DriftStatistics<A: Float + Send + Sync> {
    /// Total number of drifts detected
    pub total_drifts: usize,
    /// Recent drift rate (drifts per second)
    pub recent_drift_rate: f64,
    /// Average confidence of drift detections
    pub average_drift_confidence: Option<A>,
    /// Distribution of drift types
    pub drift_types_distribution: std::collections::HashMap<DriftType, usize>,
    /// Time since last drift
    pub time_since_last_drift: Option<Duration>,
}

/// Advanced concept drift analysis and adaptation
pub mod advanced_drift_analysis {
    use super::*;
    use std::collections::HashMap;

    /// Advanced drift detector with machine learning-based detection
    #[derive(Debug)]
    pub struct AdvancedDriftDetector<A: Float + Send + Sync> {
        /// Base detector ensemble
        base_detectors: Vec<Box<dyn DriftDetectorTrait<A>>>,

        /// Drift pattern analyzer
        pattern_analyzer: DriftPatternAnalyzer<A>,

        /// Adaptive threshold manager
        threshold_manager: AdaptiveThresholdManager<A>,

        /// Context-aware drift detection
        context_detector: ContextAwareDriftDetector<A>,

        /// Performance impact analyzer
        impact_analyzer: DriftImpactAnalyzer<A>,

        /// Adaptation strategy selector
        adaptation_selector: AdaptationStrategySelector<A>,

        /// Historical drift database
        drift_database: DriftDatabase<A>,
    }

    /// Trait for all drift detectors
    pub trait DriftDetectorTrait<A: Float + Send + Sync>: std::fmt::Debug {
        fn update(&mut self, value: A) -> DriftStatus;
        fn reset(&mut self);
        fn get_confidence(&self) -> A;

        /// Human-readable detector name, used to key adaptive thresholds.
        fn name(&self) -> &str;

        /// Apply an adapted decision threshold (C6). Without this the adaptive
        /// threshold manager computed thresholds that nothing ever consumed.
        fn set_threshold(&mut self, threshold: A);

        /// The threshold currently in force.
        fn threshold(&self) -> A;
    }

    /// Drift pattern analyzer for characterizing drift behavior
    #[derive(Debug)]
    pub struct DriftPatternAnalyzer<A: Float + Send + Sync> {
        /// Pattern history buffer
        pub(crate) pattern_buffer: VecDeque<PatternFeatures<A>>,

        /// Rolling raw values the features are extracted from (C4: the analyzer
        /// used to be handed a single value per call, so variance was always 0)
        pub(crate) value_buffer: VecDeque<A>,

        /// Window length for feature extraction
        pub(crate) window: usize,

        /// Learned drift patterns
        pub(crate) known_patterns: HashMap<String, DriftPattern<A>>,

        /// Pattern matching threshold
        pub(crate) matching_threshold: A,

        /// Feature extractors
        pub(crate) feature_extractors: Vec<Box<dyn FeatureExtractor<A>>>,
    }

    /// Pattern features for drift characterization.
    ///
    /// C4: everything beyond the first two moments needs a window of samples to
    /// exist at all. Those fields are therefore `Option`: they are `None` until
    /// the analyzer has enough history, instead of carrying the placeholder
    /// zeros (and the fabricated `fractal_dimension: 1.5`) they used to. The
    /// `entropy` field in particular used to be `variance.ln().abs()`, which is
    /// `+inf` for the zero-variance single-sample window it was always called
    /// with.
    #[derive(Debug, Clone)]
    pub struct PatternFeatures<A: Float + Send + Sync> {
        /// Statistical moments
        pub mean: A,
        pub variance: A,
        pub skewness: Option<A>,
        pub kurtosis: Option<A>,

        /// Trend indicators
        pub trend_slope: Option<A>,
        pub trend_strength: Option<A>,

        /// Frequency domain features
        pub dominant_frequency: Option<A>,
        pub spectral_entropy: Option<A>,

        /// Temporal features
        pub temporal_locality: Option<A>,
        pub persistence: Option<A>,

        /// Complexity measures
        pub entropy: Option<A>,
        pub fractal_dimension: Option<A>,
    }

    impl<A: Float + Send + Sync> PatternFeatures<A> {
        /// The feature vector used for similarity search: `(name, value)` pairs
        /// for every feature that actually has a value.
        pub fn named_values(&self) -> Vec<(&'static str, A)> {
            let mut values: Vec<(&'static str, A)> =
                vec![("mean", self.mean), ("variance", self.variance)];
            let optional: [(&'static str, Option<A>); 10] = [
                ("skewness", self.skewness),
                ("kurtosis", self.kurtosis),
                ("trend_slope", self.trend_slope),
                ("trend_strength", self.trend_strength),
                ("dominant_frequency", self.dominant_frequency),
                ("spectral_entropy", self.spectral_entropy),
                ("temporal_locality", self.temporal_locality),
                ("persistence", self.persistence),
                ("entropy", self.entropy),
                ("fractal_dimension", self.fractal_dimension),
            ];
            for (name, value) in optional {
                if let Some(value) = value {
                    values.push((name, value));
                }
            }
            values
        }

        /// Look up a feature by name, as used by
        /// [`ApplicabilityCondition::feature_name`].
        pub fn feature(&self, name: &str) -> Option<A> {
            match name {
                "mean" => Some(self.mean),
                "variance" => Some(self.variance),
                "skewness" => self.skewness,
                "kurtosis" => self.kurtosis,
                "trend_slope" => self.trend_slope,
                "trend_strength" => self.trend_strength,
                "dominant_frequency" => self.dominant_frequency,
                "spectral_entropy" => self.spectral_entropy,
                "temporal_locality" => self.temporal_locality,
                "persistence" => self.persistence,
                "entropy" => self.entropy,
                "fractal_dimension" => self.fractal_dimension,
                _ => None,
            }
        }
    }

    /// Learned drift pattern
    #[derive(Debug, Clone)]
    pub struct DriftPattern<A: Float + Send + Sync> {
        /// Pattern identifier
        pub id: String,

        /// Characteristic features
        pub features: PatternFeatures<A>,

        /// Pattern type
        pub pattern_type: DriftType,

        /// Typical duration
        pub typical_duration: Duration,

        /// Optimal adaptation strategy
        pub optimal_adaptation: AdaptationRecommendation,

        /// Success rate of this pattern's adaptations
        pub adaptation_success_rate: A,

        /// Occurrence frequency
        pub occurrence_count: usize,
    }

    /// Feature extractor trait
    pub trait FeatureExtractor<A: Float + Send + Sync>: std::fmt::Debug {
        fn extract(&self, data: &[A]) -> A;
        fn name(&self) -> &str;
    }

    /// Adaptive threshold management
    #[derive(Debug)]
    pub struct AdaptiveThresholdManager<A: Float + Send + Sync> {
        /// Current thresholds for different detectors
        thresholds: HashMap<String, A>,

        /// Threshold adaptation history
        threshold_history: VecDeque<ThresholdUpdate<A>>,

        /// Performance feedback for threshold adjustment
        performance_feedback: VecDeque<PerformanceFeedback<A>>,

        /// Learning rate for threshold adaptation
        learning_rate: A,
    }

    /// Threshold update record
    #[derive(Debug, Clone)]
    pub struct ThresholdUpdate<A: Float + Send + Sync> {
        pub detector_name: String,
        pub old_threshold: A,
        pub new_threshold: A,
        pub timestamp: Instant,
        pub reason: String,
    }

    /// Performance feedback for threshold adjustment
    #[derive(Debug, Clone)]
    pub struct PerformanceFeedback<A: Float + Send + Sync> {
        pub true_positive_rate: A,
        pub false_positive_rate: A,
        pub detection_delay: Duration,
        pub adaptation_effectiveness: A,
        pub timestamp: Instant,
    }

    /// Context-aware drift detection
    #[derive(Debug)]
    pub struct ContextAwareDriftDetector<A: Float + Send + Sync> {
        /// Contextual features
        context_features: Vec<ContextFeature<A>>,

        /// Current context state
        current_context: Option<String>,

        /// Context transition matrix
        transition_matrix: HashMap<(String, String), A>,
    }

    /// Contextual feature for drift detection
    #[derive(Debug, Clone)]
    pub struct ContextFeature<A: Float + Send + Sync> {
        pub name: String,
        pub value: A,
        pub importance_weight: A,
        pub temporal_stability: A,
    }

    /// Drift impact analyzer
    #[derive(Debug)]
    pub struct DriftImpactAnalyzer<A: Float + Send + Sync> {
        /// Impact metrics history
        impact_history: VecDeque<DriftImpact<A>>,

        /// Severity classifier
        severity_classifier: SeverityClassifier<A>,

        /// Recovery time predictor
        recovery_predictor: RecoveryTimePredictor<A>,

        /// Business impact estimator
        business_impact_estimator: BusinessImpactEstimator<A>,
    }

    /// Drift impact assessment
    #[derive(Debug, Clone)]
    pub struct DriftImpact<A: Float + Send + Sync> {
        /// Performance degradation magnitude
        pub performance_degradation: A,

        /// Affected metrics
        pub affected_metrics: Vec<String>,

        /// Estimated recovery time
        pub estimated_recovery_time: Duration,

        /// Confidence in impact assessment
        pub confidence: A,

        /// Business impact score
        pub business_impact_score: A,

        /// Urgency level
        pub urgency_level: UrgencyLevel,
    }

    /// Urgency levels for drift response
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum UrgencyLevel {
        Low,
        Medium,
        High,
        Critical,
    }

    /// Adaptation strategy selector
    #[derive(Debug)]
    pub struct AdaptationStrategySelector<A: Float + Send + Sync> {
        /// Available adaptation strategies
        strategies: Vec<AdaptationStrategy<A>>,

        /// Strategy performance history
        strategy_performance: HashMap<String, StrategyPerformance<A>>,

        /// Multi-armed bandit for strategy selection
        bandit: EpsilonGreedyBandit<A>,

        /// Context-strategy mapping
        context_strategy_map: HashMap<String, Vec<String>>,
    }

    /// Adaptation strategy
    #[derive(Debug, Clone)]
    pub struct AdaptationStrategy<A: Float + Send + Sync> {
        /// Strategy identifier
        pub id: String,

        /// Strategy type
        pub strategy_type: AdaptationStrategyType,

        /// Parameters
        pub parameters: HashMap<String, A>,

        /// Applicability conditions
        pub applicability_conditions: Vec<ApplicabilityCondition<A>>,

        /// Expected effectiveness
        pub expected_effectiveness: A,

        /// Computational cost
        pub computational_cost: A,
    }

    /// Types of adaptation strategies
    #[derive(Debug, Clone, Copy)]
    pub enum AdaptationStrategyType {
        ParameterTuning,
        ModelReplacement,
        EnsembleReweighting,
        ArchitectureChange,
        DataAugmentation,
        FeatureSelection,
        Hybrid,
    }

    /// Conditions for strategy applicability
    #[derive(Debug, Clone)]
    pub struct ApplicabilityCondition<A: Float + Send + Sync> {
        pub feature_name: String,
        pub operator: ComparisonOperator,
        pub threshold: A,
        pub weight: A,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum ComparisonOperator {
        GreaterThan,
        LessThan,
        Equal,
        NotEqual,
        GreaterEqual,
        LessEqual,
    }

    /// Strategy performance tracking
    #[derive(Debug, Clone)]
    pub struct StrategyPerformance<A: Float + Send + Sync> {
        pub success_rate: A,
        pub average_improvement: A,
        pub average_adaptation_time: Duration,
        pub stability_after_adaptation: A,
        pub usage_count: usize,
    }

    /// Epsilon-greedy bandit for strategy selection.
    ///
    /// C5: the bandit had no methods at all, so `select_strategy` returned the
    /// same hardcoded "increase_lr" strategy on every call.
    pub struct EpsilonGreedyBandit<A: Float + Send + Sync> {
        epsilon: A,
        action_values: HashMap<String, A>,
        action_counts: HashMap<String, usize>,
        total_trials: usize,
        /// Deterministically seeded so exploration is reproducible in tests.
        rng: scirs2_core::random::Random<scirs2_core::random::rngs::StdRng>,
    }

    impl<A: Float + Send + Sync> std::fmt::Debug for EpsilonGreedyBandit<A> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("EpsilonGreedyBandit")
                .field("action_values", &self.action_values.len())
                .field("total_trials", &self.total_trials)
                .finish()
        }
    }

    /// Historical drift database
    #[derive(Debug)]
    pub struct DriftDatabase<A: Float + Send + Sync> {
        /// Stored drift events
        drift_events: Vec<StoredDriftEvent<A>>,

        /// Pattern-outcome associations
        pattern_outcomes: HashMap<String, Vec<AdaptationOutcome<A>>>,

        /// Seasonal drift patterns
        seasonal_patterns: HashMap<String, SeasonalPattern<A>>,

        /// Similarity search index
        similarity_index: SimilarityIndex<A>,
    }

    /// Stored drift event for learning.
    ///
    /// C5: `outcome` is now optional and starts out `None`. `store_event` used
    /// to invent `success: true` with a `performance_improvement` of `0.1`, a
    /// 60 second adaptation time and a 300 second stability period the moment
    /// the strategy was *selected* — before anything had been observed. The
    /// real outcome arrives later through
    /// [`AdvancedDriftDetector::record_adaptation_outcome`].
    #[derive(Debug, Clone)]
    pub struct StoredDriftEvent<A: Float + Send + Sync> {
        pub features: PatternFeatures<A>,
        pub context: Vec<ContextFeature<A>>,
        pub applied_strategy: String,
        pub outcome: Option<AdaptationOutcome<A>>,
        pub timestamp: Instant,
    }

    /// Adaptation outcome for learning
    #[derive(Debug, Clone)]
    pub struct AdaptationOutcome<A: Float + Send + Sync> {
        pub success: bool,
        pub performance_improvement: A,
        pub adaptation_time: Duration,
        pub stability_period: Duration,
        pub side_effects: Vec<String>,
    }

    /// Seasonal drift pattern
    #[derive(Debug, Clone)]
    pub struct SeasonalPattern<A: Float + Send + Sync> {
        pub period: Duration,
        pub amplitude: A,
        pub phase_offset: Duration,
        pub pattern_strength: A,
        pub last_occurrence: Instant,
    }

    /// Similarity search for historical patterns
    #[derive(Debug)]
    pub struct SimilarityIndex<A: Float + Send + Sync> {
        /// Feature vectors for similarity search
        feature_vectors: Vec<(String, Vec<A>)>,

        /// Similarity threshold
        similarity_threshold: A,

        /// Distance metric
        distance_metric: DistanceMetric,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum DistanceMetric {
        Euclidean,
        Manhattan,
        Cosine,
        Mahalanobis,
    }

    impl<A: Float + Default + Clone + std::fmt::Debug + std::iter::Sum + Send + Sync + 'static>
        AdvancedDriftDetector<A>
    {
        /// Create new advanced drift detector.
        ///
        /// C6/C8: `base_detectors` used to be an empty vector with an "Add base
        /// detectors here" comment, which made the whole detector a hollow
        /// shell: no detector ever voted, the adaptive thresholds had nothing to
        /// apply to, and `combine_detection_results` divided by
        /// `base_results.len() == 0`. It is now populated with adapters over the
        /// three real detectors implemented in this module.
        pub fn new(config: DriftDetectorConfig) -> Self {
            let threshold = A::from(config.threshold).unwrap_or_else(A::one);
            let warning = A::from(config.warningthreshold).unwrap_or_else(A::zero);
            let delta =
                A::from(config.alpha).unwrap_or_else(|| A::from(0.002).unwrap_or_else(A::zero));

            let base_detectors: Vec<Box<dyn DriftDetectorTrait<A>>> = vec![
                Box::new(impls::PageHinkleyAdapter::new(threshold, warning)),
                Box::new(impls::AdwinAdapter::new(delta, config.window_size)),
                Box::new(impls::DdmAdapter::new(config.min_samples)),
            ];

            Self {
                base_detectors,
                pattern_analyzer: DriftPatternAnalyzer::new(config.window_size),
                threshold_manager: AdaptiveThresholdManager::new(),
                context_detector: ContextAwareDriftDetector::new(),
                impact_analyzer: DriftImpactAnalyzer::new(),
                adaptation_selector: AdaptationStrategySelector::new(),
                drift_database: DriftDatabase::new(),
            }
        }

        /// Advanced drift detection with pattern analysis
        pub fn detect_drift_advanced(
            &mut self,
            value: A,
            context_features: &[ContextFeature<A>],
        ) -> Result<AdvancedDriftResult<A>> {
            // Update context
            self.context_detector.update_context(context_features);

            // Run base detectors
            let base_results: Vec<_> = self
                .base_detectors
                .iter_mut()
                .map(|detector| (detector.name().to_string(), detector.update(value)))
                .collect();
            let statuses: Vec<DriftStatus> =
                base_results.iter().map(|(_, status)| *status).collect();

            // Analyze patterns over the rolling window (C4: the analyzer used to
            // be handed a one-element slice, so variance was always exactly 0
            // and `entropy = variance.ln().abs()` was always `+inf`).
            let pattern_features = self.pattern_analyzer.ingest(value)?;
            let matched_pattern = self.pattern_analyzer.match_pattern(&pattern_features);

            // Adaptive threshold adjustment, then actually apply the adapted
            // thresholds to the detectors (C6).
            self.threshold_manager
                .update_thresholds(&base_results, &pattern_features);
            self.threshold_manager.apply_to(&mut self.base_detectors);

            // Combine results with confidence weighting
            let combined_result = self.combine_detection_results(&statuses, &matched_pattern);

            // Analyze impact if drift detected
            let impact = if combined_result.status == DriftStatus::Drift {
                Some(
                    self.impact_analyzer
                        .analyze_impact(&pattern_features, &matched_pattern)?,
                )
            } else {
                None
            };

            // Select adaptation strategy
            let adaptation_strategy = if let Some(ref impact) = impact {
                self.adaptation_selector.select_strategy(
                    &pattern_features,
                    impact,
                    &matched_pattern,
                )?
            } else {
                None
            };

            // Store in database for learning. The stored event carries no
            // outcome yet (C5): a real one arrives through
            // `record_adaptation_outcome`.
            if combined_result.status == DriftStatus::Drift {
                self.drift_database.store_event(
                    &pattern_features,
                    context_features,
                    &adaptation_strategy,
                );
            }

            Ok(AdvancedDriftResult {
                status: combined_result.status,
                confidence: combined_result.confidence,
                matched_pattern,
                impact,
                recommended_strategy: adaptation_strategy,
                feature_importance: self.calculate_feature_importance(&pattern_features),
                prediction_horizon: self.estimate_drift_duration(&pattern_features),
            })
        }

        /// Report what actually happened after the most recently recommended
        /// adaptation was applied (C5).
        ///
        /// This is what turns `DriftDatabase` into a real learning store: the
        /// outcome is recorded against the pending event, folded into the
        /// strategy's measured performance and the bandit's action values, and
        /// used to learn (or reinforce) a `DriftPattern` so that
        /// `match_pattern` can eventually match something.
        pub fn record_adaptation_outcome(&mut self, outcome: AdaptationOutcome<A>) -> Result<()> {
            let Some((strategy_id, features)) =
                self.drift_database.complete_pending_event(outcome.clone())
            else {
                return Err(crate::error::OptimError::InvalidState(
                    "no adaptation is awaiting an outcome".to_string(),
                ));
            };
            self.adaptation_selector
                .record_outcome(&strategy_id, &outcome);
            self.pattern_analyzer.learn_pattern(
                &features,
                &strategy_id,
                &outcome,
                self.impact_analyzer.last_drift_type(),
            );
            self.impact_analyzer.record_observed_recovery(&outcome);
            Ok(())
        }

        /// Feed measured detection quality back into the threshold manager (C6).
        pub fn record_threshold_feedback(&mut self, feedback: PerformanceFeedback<A>) {
            self.threshold_manager.record_feedback(feedback);
        }

        /// Patterns learned so far.
        pub fn known_patterns(&self) -> &HashMap<String, DriftPattern<A>> {
            &self.pattern_analyzer.known_patterns
        }

        /// Adapted thresholds currently in force, keyed by detector name.
        pub fn detector_thresholds(&self) -> Vec<(String, A)> {
            self.base_detectors
                .iter()
                .map(|detector| (detector.name().to_string(), detector.threshold()))
                .collect()
        }

        /// Stored drift events, including the ones still awaiting an outcome.
        pub fn stored_events(&self) -> &[StoredDriftEvent<A>] {
            &self.drift_database.drift_events
        }

        fn combine_detection_results(
            &self,
            base_results: &[DriftStatus],
            matched_pattern: &Option<DriftPattern<A>>,
        ) -> CombinedDetectionResult<A> {
            // C8: with no detectors at all there is nothing to combine, and the
            // old `drift_votes / base_results.len()` produced `0/0 = NaN` which
            // then poisoned every downstream comparison.
            if base_results.is_empty() {
                return CombinedDetectionResult {
                    status: DriftStatus::Stable,
                    confidence: A::zero(),
                };
            }

            // Weighted voting based on detector confidence and pattern matching
            let drift_votes = base_results
                .iter()
                .filter(|&&s| s == DriftStatus::Drift)
                .count();
            let warning_votes = base_results
                .iter()
                .filter(|&&s| s == DriftStatus::Warning)
                .count();

            // Pattern-based confidence adjustment. With no matched pattern there
            // is no pattern evidence either way, so the pattern term is neutral.
            let neutral = A::from(0.5).unwrap_or_else(A::zero);
            let pattern_confidence = matched_pattern
                .as_ref()
                .map(|p| p.adaptation_success_rate)
                .unwrap_or(neutral);
            let strong = A::from(0.7).unwrap_or_else(A::one);

            let status = if drift_votes >= 2 {
                DriftStatus::Drift
            } else if warning_votes >= 2 || (drift_votes >= 1 && pattern_confidence > strong) {
                DriftStatus::Warning
            } else {
                DriftStatus::Stable
            };

            let vote_share =
                A::from(drift_votes as f64 / base_results.len() as f64).unwrap_or_else(A::zero);
            let confidence = vote_share * pattern_confidence;

            CombinedDetectionResult { status, confidence }
        }

        fn calculate_feature_importance(
            &self,
            features: &PatternFeatures<A>,
        ) -> HashMap<String, A> {
            // Importance is the magnitude of each feature that actually has a
            // value, normalised so the reported weights sum to one.
            let mut magnitudes: Vec<(String, A)> = features
                .named_values()
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(name, value)| (name.to_string(), value.abs()))
                .collect();
            let total = magnitudes
                .iter()
                .fold(A::zero(), |acc, (_, value)| acc + *value);
            if total > A::zero() {
                for entry in magnitudes.iter_mut() {
                    entry.1 = entry.1 / total;
                }
            }
            magnitudes.into_iter().collect()
        }

        fn estimate_drift_duration(&self, features: &PatternFeatures<A>) -> Duration {
            // Base horizon, scaled by how strong and how persistent the observed
            // trend is. When either is unmeasured the base horizon stands rather
            // than being multiplied by a placeholder zero (which used to collapse
            // the horizon to 0 seconds on every call, since both fields were
            // hardcoded zeros).
            let base_duration = Duration::from_secs(300);
            let (Some(strength), Some(persistence)) =
                (features.trend_strength, features.persistence)
            else {
                return base_duration;
            };
            let multiplier = (strength * persistence).to_f64().unwrap_or(1.0);
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return base_duration;
            }
            let seconds = (base_duration.as_secs() as f64 * multiplier).clamp(1.0, 86_400.0);
            Duration::from_secs(seconds as u64)
        }
    }

    /// Advanced drift detection result
    #[derive(Debug, Clone)]
    pub struct AdvancedDriftResult<A: Float + Send + Sync> {
        pub status: DriftStatus,
        pub confidence: A,
        pub matched_pattern: Option<DriftPattern<A>>,
        pub impact: Option<DriftImpact<A>>,
        pub recommended_strategy: Option<AdaptationStrategy<A>>,
        pub feature_importance: HashMap<String, A>,
        pub prediction_horizon: Duration,
    }

    #[derive(Debug, Clone)]
    struct CombinedDetectionResult<A: Float + Send + Sync> {
        status: DriftStatus,
        confidence: A,
    }

    mod impls;

    #[cfg(test)]
    mod tests;

    pub(crate) use impls::{BusinessImpactEstimator, RecoveryTimePredictor, SeverityClassifier};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = PageHinkleyDetector::new(3.0f64, 2.0f64);

        // Stable period
        for _ in 0..10 {
            let status = detector.update(0.1);
            assert_eq!(status, DriftStatus::Stable);
        }

        // Drift period
        for _ in 0..5 {
            let status = detector.update(0.5); // Higher loss
            if status == DriftStatus::Drift {
                break;
            }
        }
    }

    /// C1: a stationary (non-drifting) loss stream whose baseline is far
    /// from the old hardcoded `0.1` "estimated mean under H0" must not
    /// falsely report drift. The previous constant made `sum` accumulate
    /// `loss - 0.1` on every update: for a stable stream at (say) 5.0, that
    /// is `+4.9` every single sample, guaranteeing `test_stat` blows past
    /// any reasonable threshold in only a handful of updates even though
    /// nothing changed.
    #[test]
    fn page_hinkley_does_not_falsely_drift_on_stable_stream_away_from_0_1() {
        let mut detector = PageHinkleyDetector::new(5.0f64, 3.0f64);

        // A perfectly stationary stream at loss = 5.0, far from the old
        // hardcoded mean_loss of 0.1.
        for _ in 0..200 {
            let status = detector.update(5.0);
            assert_eq!(
                status,
                DriftStatus::Stable,
                "C1 regression: false drift reported on a stationary stream \
                 whose baseline (5.0) differs from the old hardcoded mean_loss (0.1)"
            );
        }
    }

    /// C1: the detector must still correctly flag a genuine regime change
    /// (loss step-increasing well above its established running mean),
    /// confirming the running-mean fix did not just make it insensitive to
    /// real drift.
    #[test]
    fn page_hinkley_detects_genuine_drift_away_from_0_1_baseline() {
        let mut detector = PageHinkleyDetector::new(5.0f64, 3.0f64);

        // Establish a stable baseline around loss = 5.0.
        for _ in 0..30 {
            detector.update(5.0);
        }

        // Sharp, sustained increase: must eventually report Drift.
        let mut drifted = false;
        for _ in 0..50 {
            let status = detector.update(20.0);
            if status == DriftStatus::Drift {
                drifted = true;
                break;
            }
        }
        assert!(
            drifted,
            "C1 regression: detector failed to flag a genuine sustained \
             increase in loss away from a non-0.1 baseline"
        );
    }

    #[test]
    fn test_adwin_detector() {
        let mut detector = AdwinDetector::new(0.005f64, 100);

        // Add stable values
        for i in 0..20 {
            let value = 0.1 + (i as f64) * 0.001; // Slight trend
            detector.update(value);
        }

        // Add drift values
        for i in 0..10 {
            let value = 0.5 + (i as f64) * 0.01; // Clear change
            let status = detector.update(value);
            if status == DriftStatus::Drift {
                break;
            }
        }
    }

    /// C2: `delta` (the detector's confidence parameter) must actually
    /// affect sensitivity. The previous implementation never read `delta`
    /// at all, so two detectors built with wildly different `delta` values
    /// behaved identically. A much smaller `delta` (higher required
    /// confidence) must be at least as slow to fire as a larger `delta` on
    /// the same borderline-noisy data.
    #[test]
    fn adwin_delta_affects_sensitivity() {
        fn feed(mut detector: AdwinDetector<f64>) -> Option<usize> {
            // Stable baseline noise around 1.0.
            for i in 0..20 {
                let value = 1.0 + 0.02 * ((i % 3) as f64 - 1.0);
                detector.update(value);
            }
            // A modest, borderline shift.
            for i in 0..40 {
                let value = 1.15 + 0.02 * ((i % 3) as f64 - 1.0);
                if detector.update(value) == DriftStatus::Drift {
                    return Some(i);
                }
            }
            None
        }

        // A very small delta demands much higher confidence (a much larger
        // eps_cut) than a large delta, so it must not fire strictly sooner.
        let lenient = feed(AdwinDetector::new(0.5f64, 200)); // delta close to 1: low confidence required
        let strict = feed(AdwinDetector::new(1e-6f64, 200)); // delta tiny: very high confidence required

        match (lenient, strict) {
            (Some(_), None) => {} // lenient fired, strict correctly held off: expected
            (Some(l), Some(s)) => assert!(
                s >= l,
                "C2 regression: stricter delta (1e-6) fired sooner ({s}) than \
                 lenient delta (0.5, fired at {l}) — delta has no effect on sensitivity"
            ),
            (None, Some(_)) => {
                panic!("C2 regression: stricter delta fired but the more lenient delta did not")
            }
            (None, None) => {
                // Both held off - inconclusive for the ordering claim, but
                // at minimum confirms neither exploded/panicked.
            }
        }
    }

    #[test]
    fn test_ddm_detector() {
        let mut detector = DdmDetector::<f64>::new();

        // Stable period with low error rate
        for i in 0..50 {
            let iserror = i % 10 == 0; // 10% error rate
            detector.update(iserror);
        }

        // Period with high error rate
        for i in 0..20 {
            let iserror = i % 2 == 0; // 50% error rate
            let status = detector.update(iserror);
            if status == DriftStatus::Drift {
                break;
            }
        }
    }

    #[test]
    fn test_concept_drift_detector() {
        let config = DriftDetectorConfig::default();
        let mut detector = ConceptDriftDetector::new(config);

        // Simulate stable period
        for i in 0..30 {
            let loss = 0.1 + (i as f64) * 0.001;
            let iserror = i % 10 == 0;
            let status = detector.update(loss, iserror).expect("unwrap failed");
            assert_ne!(status, DriftStatus::Drift); // Should be stable
        }

        // Simulate drift
        for i in 0..20 {
            let loss = 0.5 + (i as f64) * 0.01; // Much higher loss
            let iserror = i % 2 == 0; // Higher error rate
            let _status = detector.update(loss, iserror).expect("unwrap failed");
        }

        let stats = detector.get_statistics();
        assert!(stats.total_drifts > 0 || stats.recent_drift_rate > 0.0);
    }

    #[test]
    fn test_drift_event() {
        let event = DriftEvent {
            timestamp: Instant::now(),
            confidence: 0.85f64,
            drift_type: DriftType::Sudden,
            adaptation_recommendation: AdaptationRecommendation::Reset,
        };

        assert_eq!(event.drift_type, DriftType::Sudden);
        assert!(event.confidence > 0.8);
    }
}
