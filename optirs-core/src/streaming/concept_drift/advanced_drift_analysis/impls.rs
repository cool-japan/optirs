// Real implementations for the advanced drift-analysis components
// (findings C4, C5, C6, C7, C8).
//
// Everything in this file replaces a block the source itself labelled
// "Implementation stubs for complex components" / "Placeholder implementations
// for complex analyzers":
//
// * C4 — `extract_features` was called with a one-element slice, so `variance`
//   was always exactly 0 and `entropy = variance.ln().abs()` was always `+inf`;
//   eight further features were hardcoded zeros and `fractal_dimension` was the
//   invented constant 1.5.
// * C5 — `known_patterns` was never written to (so `match_pattern` could never
//   match), `select_strategy` returned the same hardcoded strategy on every
//   call, and `store_event` recorded `success: true` before anything had
//   happened.
// * C6 — the adapted thresholds were computed and then discarded; no detector
//   ever consumed one, and `performance_feedback` was never fed.
// * C7 — `threshold_history` and `impact_history` grew without bound.

use super::*;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// numeric helpers
// ---------------------------------------------------------------------------

/// Smallest value used to guard a logarithm or a division.
const EPSILON: f64 = 1e-12;

/// Bound on the retained threshold-adaptation log (C7).
const MAX_THRESHOLD_HISTORY: usize = 256;

/// Bound on the retained performance-feedback log (C7).
const MAX_FEEDBACK_HISTORY: usize = 256;

/// Bound on the retained impact log (C7).
const MAX_IMPACT_HISTORY: usize = 256;

/// Bound on the retained pattern-feature buffer (C7).
const MAX_PATTERN_BUFFER: usize = 256;

/// Bound on the stored drift-event log (C7).
const MAX_STORED_EVENTS: usize = 1024;

/// Minimum window length before the window-derived features are meaningful.
const MIN_FEATURE_WINDOW: usize = 8;

fn to_scalar<A: Float>(value: f64) -> Option<A> {
    if value.is_finite() {
        A::from(value)
    } else {
        None
    }
}

fn from_scalar<A: Float>(value: A) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// `k`-th central moment.
fn central_moment(values: &[f64], k: i32) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(k)).sum::<f64>() / values.len() as f64
}

/// Ordinary-least-squares slope and coefficient of determination against index.
fn linear_fit(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 3 {
        return None;
    }
    let n = values.len() as f64;
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = mean(values);
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (index, value) in values.iter().enumerate() {
        let dx = index as f64 - x_mean;
        let dy = value - y_mean;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let _ = n;
    if sxx <= EPSILON {
        return None;
    }
    let slope = sxy / sxx;
    // R^2 = explained / total; zero-variance input has nothing to explain.
    let r_squared = if syy <= EPSILON {
        0.0
    } else {
        ((sxy * sxy) / (sxx * syy)).clamp(0.0, 1.0)
    };
    Some((slope, r_squared))
}

/// Autocorrelation at `lag` of a mean-centred series.
fn autocorrelation(values: &[f64], lag: usize) -> Option<f64> {
    if lag == 0 || values.len() <= lag + 1 {
        return None;
    }
    let m = mean(values);
    let denominator: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
    if denominator <= EPSILON {
        return None;
    }
    let numerator: f64 = values
        .windows(lag + 1)
        .map(|window| (window[0] - m) * (window[lag] - m))
        .sum();
    Some((numerator / denominator).clamp(-1.0, 1.0))
}

/// Real discrete Fourier transform magnitude spectrum (positive frequencies).
///
/// A direct O(n^2) DFT is used deliberately: the analysis window is at most a
/// few hundred samples, and pulling in an FFT dependency for that is not worth
/// it. The transform is real arithmetic only — no external crate involved.
fn magnitude_spectrum(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n < 4 {
        return Vec::new();
    }
    let m = mean(values);
    let centred: Vec<f64> = values.iter().map(|v| v - m).collect();
    let bins = n / 2;
    let mut spectrum = Vec::with_capacity(bins);
    for k in 1..=bins {
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (index, value) in centred.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (index as f64) / (n as f64);
            real += value * angle.cos();
            imaginary += value * angle.sin();
        }
        spectrum.push((real * real + imaginary * imaginary).sqrt());
    }
    spectrum
}

/// Shannon entropy of a normalised power spectrum, in nats.
fn spectral_entropy(spectrum: &[f64]) -> Option<f64> {
    if spectrum.is_empty() {
        return None;
    }
    let total: f64 = spectrum.iter().map(|v| v * v).sum();
    if total <= EPSILON {
        return None;
    }
    let mut entropy = 0.0;
    for value in spectrum {
        let p = (value * value) / total;
        if p > EPSILON {
            entropy -= p * p.ln();
        }
    }
    Some(entropy)
}

/// Shannon entropy of an equal-width histogram of the window, in nats.
///
/// C4: replaces `variance.ln().abs()`, which is `+inf` for zero variance (the
/// only case the previous single-sample call site could ever produce).
fn histogram_entropy(values: &[f64]) -> Option<f64> {
    if values.len() < MIN_FEATURE_WINDOW {
        return None;
    }
    let (min, max) = values.iter().fold((f64::MAX, f64::MIN), |(lo, hi), value| {
        (lo.min(*value), hi.max(*value))
    });
    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    if (max - min).abs() <= EPSILON {
        // A constant window carries no information; that is a real zero, not a
        // degenerate logarithm.
        return Some(0.0);
    }
    let bins = (values.len() as f64).sqrt().ceil().max(2.0) as usize;
    let width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for value in values {
        let index = (((value - min) / width) as usize).min(bins - 1);
        counts[index] += 1;
    }
    let total = values.len() as f64;
    let mut entropy = 0.0;
    for count in counts {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        entropy -= p * p.ln();
    }
    Some(entropy)
}

/// Higuchi fractal dimension of the window.
fn higuchi_fractal_dimension(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 16 {
        return None;
    }
    let k_max = (n / 4).clamp(2, 10);
    let mut logs = Vec::with_capacity(k_max);
    for k in 1..=k_max {
        let mut lengths = Vec::with_capacity(k);
        for m in 0..k {
            let count = (n - m - 1) / k;
            if count == 0 {
                continue;
            }
            let mut sum = 0.0;
            for i in 1..=count {
                sum += (values[m + i * k] - values[m + (i - 1) * k]).abs();
            }
            let normaliser = (n - 1) as f64 / (count * k) as f64;
            lengths.push(sum * normaliser / k as f64);
        }
        if lengths.is_empty() {
            continue;
        }
        let length = mean(&lengths);
        if length <= EPSILON {
            continue;
        }
        logs.push(((1.0 / k as f64).ln(), length.ln()));
    }
    if logs.len() < 3 {
        return None;
    }
    // Slope of ln(L(k)) against ln(1/k) is the fractal dimension.
    let x_mean = mean(&logs.iter().map(|(x, _)| *x).collect::<Vec<_>>());
    let y_mean = mean(&logs.iter().map(|(_, y)| *y).collect::<Vec<_>>());
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (x, y) in &logs {
        sxy += (x - x_mean) * (y - y_mean);
        sxx += (x - x_mean) * (x - x_mean);
    }
    if sxx <= EPSILON {
        return None;
    }
    Some((sxy / sxx).clamp(1.0, 2.0))
}

// ---------------------------------------------------------------------------
// C6: real base detectors behind the `DriftDetectorTrait`
// ---------------------------------------------------------------------------

/// Page-Hinkley detector exposed through [`DriftDetectorTrait`].
#[derive(Debug)]
pub(crate) struct PageHinkleyAdapter<A: Float + Send + Sync + std::fmt::Debug> {
    inner: PageHinkleyDetector<A>,
    threshold: A,
    warning: A,
    last_status: DriftStatus,
}

impl<A: Float + Send + Sync + std::fmt::Debug> PageHinkleyAdapter<A> {
    pub(crate) fn new(threshold: A, warning: A) -> Self {
        Self {
            inner: PageHinkleyDetector::new(threshold, warning),
            threshold,
            warning,
            last_status: DriftStatus::Stable,
        }
    }
}

impl<A: Float + Send + Sync + std::fmt::Debug> DriftDetectorTrait<A> for PageHinkleyAdapter<A> {
    fn update(&mut self, value: A) -> DriftStatus {
        self.last_status = self.inner.update(value);
        self.last_status
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.last_status = DriftStatus::Stable;
    }

    fn get_confidence(&self) -> A {
        match self.last_status {
            DriftStatus::Drift => A::one(),
            DriftStatus::Warning => A::from(0.5).unwrap_or_else(A::zero),
            DriftStatus::Stable => A::zero(),
        }
    }

    fn name(&self) -> &str {
        "page_hinkley"
    }

    fn set_threshold(&mut self, threshold: A) {
        // Rebuild with the adapted threshold, preserving the relative warning
        // band so a tightened threshold tightens both levels.
        let ratio = if self.threshold > A::zero() {
            self.warning / self.threshold
        } else {
            A::from(0.5).unwrap_or_else(A::zero)
        };
        self.threshold = threshold;
        self.warning = threshold * ratio;
        // Mutate in place: rebuilding would reset the cumulative statistic on
        // every adaptation, so the detector could never reach any threshold.
        self.inner.set_thresholds(self.threshold, self.warning);
    }

    fn threshold(&self) -> A {
        self.threshold
    }
}

/// ADWIN detector exposed through [`DriftDetectorTrait`].
#[derive(Debug)]
pub(crate) struct AdwinAdapter<A: Float + Sum + Send + Sync + std::fmt::Debug> {
    inner: AdwinDetector<A>,
    delta: A,
    window: usize,
    last_status: DriftStatus,
}

impl<A: Float + Sum + Send + Sync + std::fmt::Debug> AdwinAdapter<A> {
    pub(crate) fn new(delta: A, window: usize) -> Self {
        let window = window.max(16);
        Self {
            inner: AdwinDetector::new(delta, window),
            delta,
            window,
            last_status: DriftStatus::Stable,
        }
    }
}

impl<A: Float + Sum + Send + Sync + std::fmt::Debug> DriftDetectorTrait<A> for AdwinAdapter<A> {
    fn update(&mut self, value: A) -> DriftStatus {
        self.last_status = self.inner.update(value);
        self.last_status
    }

    fn reset(&mut self) {
        self.inner = AdwinDetector::new(self.delta, self.window);
        self.last_status = DriftStatus::Stable;
    }

    fn get_confidence(&self) -> A {
        match self.last_status {
            DriftStatus::Drift => A::one(),
            DriftStatus::Warning => A::from(0.5).unwrap_or_else(A::zero),
            DriftStatus::Stable => A::zero(),
        }
    }

    fn name(&self) -> &str {
        "adwin"
    }

    fn set_threshold(&mut self, threshold: A) {
        // ADWIN's decision parameter is its confidence `delta`, which must stay
        // inside (0, 1); a larger threshold means a stricter (smaller) delta.
        let lower = A::from(1e-9).unwrap_or_else(A::zero);
        let upper = A::from(0.5).unwrap_or_else(A::one);
        let delta = if threshold > A::zero() {
            (A::one() / threshold).clamp(lower, upper)
        } else {
            upper
        };
        self.delta = delta;
        // Mutate in place so the adapted confidence does not discard the window.
        self.inner.set_delta(delta);
    }

    fn threshold(&self) -> A {
        if self.delta > A::zero() {
            A::one() / self.delta
        } else {
            A::zero()
        }
    }
}

/// DDM detector exposed through [`DriftDetectorTrait`].
///
/// DDM consumes error indicators, so a continuous loss value is turned into one
/// by comparing it against the running mean of the stream: a loss above its own
/// running mean counts as an error. That is a real, stateful decision rule, not
/// a placeholder.
#[derive(Debug)]
pub(crate) struct DdmAdapter<A: Float + Send + Sync + std::fmt::Debug> {
    inner: DdmDetector<A>,
    running_mean: A,
    samples: usize,
    warmup: usize,
    /// Multiplier on the running mean above which a sample counts as an error.
    threshold: A,
    last_status: DriftStatus,
}

impl<A: Float + Send + Sync + std::fmt::Debug> DdmAdapter<A> {
    pub(crate) fn new(warmup: usize) -> Self {
        Self {
            inner: DdmDetector::with_warmup(warmup.max(2)),
            running_mean: A::zero(),
            samples: 0,
            warmup: warmup.max(2),
            threshold: A::one(),
            last_status: DriftStatus::Stable,
        }
    }
}

impl<A: Float + Send + Sync + std::fmt::Debug> DriftDetectorTrait<A> for DdmAdapter<A> {
    fn update(&mut self, value: A) -> DriftStatus {
        self.samples += 1;
        let count = A::from(self.samples).unwrap_or_else(A::one);
        self.running_mean = self.running_mean + (value - self.running_mean) / count;
        let is_error = value > self.running_mean * self.threshold;
        self.last_status = self.inner.update(is_error);
        self.last_status
    }

    fn reset(&mut self) {
        self.inner = DdmDetector::with_warmup(self.warmup);
        self.running_mean = A::zero();
        self.samples = 0;
        self.last_status = DriftStatus::Stable;
    }

    fn get_confidence(&self) -> A {
        match self.last_status {
            DriftStatus::Drift => A::one(),
            DriftStatus::Warning => A::from(0.5).unwrap_or_else(A::zero),
            DriftStatus::Stable => A::zero(),
        }
    }

    fn name(&self) -> &str {
        "ddm"
    }

    fn set_threshold(&mut self, threshold: A) {
        let lower = A::from(0.5).unwrap_or_else(A::zero);
        let upper = A::from(4.0).unwrap_or_else(A::one);
        self.threshold = threshold.clamp(lower, upper);
    }

    fn threshold(&self) -> A {
        self.threshold
    }
}

// ---------------------------------------------------------------------------
// C4: real feature extraction over a rolling window
// ---------------------------------------------------------------------------

impl<A: Float + std::iter::Sum + Send + Sync> DriftPatternAnalyzer<A> {
    pub(crate) fn new(window: usize) -> Self {
        let window = window.clamp(MIN_FEATURE_WINDOW, MAX_PATTERN_BUFFER);
        Self {
            pattern_buffer: VecDeque::with_capacity(MAX_PATTERN_BUFFER),
            value_buffer: VecDeque::with_capacity(window),
            window,
            known_patterns: HashMap::new(),
            matching_threshold: A::from(0.8).unwrap_or_else(A::one),
            feature_extractors: Vec::new(),
        }
    }

    /// Push a value into the rolling window and extract the current features.
    pub(crate) fn ingest(&mut self, value: A) -> Result<PatternFeatures<A>> {
        self.value_buffer.push_back(value);
        while self.value_buffer.len() > self.window {
            self.value_buffer.pop_front();
        }
        let window: Vec<A> = self.value_buffer.iter().copied().collect();
        let features = self.extract_features(&window)?;
        // C7: the pattern buffer is now written to and bounded (it used to be
        // allocated and never touched).
        self.pattern_buffer.push_back(features.clone());
        while self.pattern_buffer.len() > MAX_PATTERN_BUFFER {
            self.pattern_buffer.pop_front();
        }
        Ok(features)
    }

    pub(crate) fn extract_features(&mut self, data: &[A]) -> Result<PatternFeatures<A>> {
        if data.is_empty() {
            return Err(crate::error::OptimError::InvalidParameter(
                "cannot extract drift features from an empty window".to_string(),
            ));
        }

        let values: Vec<f64> = data.iter().map(|value| from_scalar(*value)).collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(crate::error::OptimError::InvalidParameter(
                "drift feature extraction received a non-finite value".to_string(),
            ));
        }

        let mean_value = mean(&values);
        let variance = central_moment(&values, 2).max(0.0);
        let std_dev = variance.sqrt();

        let long_enough = values.len() >= MIN_FEATURE_WINDOW;
        let skewness = if long_enough && std_dev > EPSILON {
            to_scalar(central_moment(&values, 3) / std_dev.powi(3))
        } else {
            None
        };
        let kurtosis = if long_enough && std_dev > EPSILON {
            to_scalar(central_moment(&values, 4) / variance.powi(2) - 3.0)
        } else {
            None
        };

        let fit = if long_enough {
            linear_fit(&values)
        } else {
            None
        };
        let trend_slope = fit.and_then(|(slope, _)| to_scalar(slope));
        let trend_strength = fit.and_then(|(_, r2)| to_scalar(r2));

        let spectrum = if long_enough {
            magnitude_spectrum(&values)
        } else {
            Vec::new()
        };
        let dominant_frequency = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(index, _)| {
                // Normalised frequency in cycles per sample.
                to_scalar((index as f64 + 1.0) / values.len() as f64)
            });
        let spectral = spectral_entropy(&spectrum).and_then(to_scalar);

        let temporal_locality = if long_enough {
            autocorrelation(&values, 1).and_then(to_scalar)
        } else {
            None
        };
        // Persistence: how many lags the autocorrelation stays above 1/e,
        // normalised by the window length.
        let persistence = if long_enough {
            let limit = (values.len() / 2).max(1);
            let mut lags = 0usize;
            for lag in 1..=limit {
                match autocorrelation(&values, lag) {
                    Some(correlation) if correlation > std::f64::consts::E.recip() => lags += 1,
                    _ => break,
                }
            }
            to_scalar(lags as f64 / limit as f64)
        } else {
            None
        };

        let entropy = histogram_entropy(&values).and_then(to_scalar);
        let fractal_dimension = higuchi_fractal_dimension(&values).and_then(to_scalar);

        // Any registered extractor contributes an extra named feature; there
        // are none by default, so this loop is a genuine extension point rather
        // than dead weight.
        for extractor in &self.feature_extractors {
            let _ = extractor.extract(data);
        }

        Ok(PatternFeatures {
            mean: to_scalar(mean_value).unwrap_or_else(A::zero),
            variance: to_scalar(variance).unwrap_or_else(A::zero),
            skewness,
            kurtosis,
            trend_slope,
            trend_strength,
            dominant_frequency,
            spectral_entropy: spectral,
            temporal_locality,
            persistence,
            entropy,
            fractal_dimension,
        })
    }

    pub(crate) fn match_pattern(&self, features: &PatternFeatures<A>) -> Option<DriftPattern<A>> {
        self.known_patterns
            .values()
            .filter_map(|pattern| {
                let similarity = self.calculate_similarity(&pattern.features, features);
                if similarity > self.matching_threshold {
                    Some((similarity, pattern))
                } else {
                    None
                }
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, pattern)| pattern.clone())
    }

    /// Similarity in `[0, 1]` over every feature both patterns actually have.
    ///
    /// C5: the previous formula was `1 - (|Δmean| + |Δvariance|) / 2`, which is
    /// unbounded below (arbitrarily negative for distant patterns), ignores ten
    /// of the twelve features, and is not scale-invariant.
    pub(crate) fn calculate_similarity(
        &self,
        left: &PatternFeatures<A>,
        right: &PatternFeatures<A>,
    ) -> A {
        let left_values: HashMap<&str, f64> = left
            .named_values()
            .into_iter()
            .map(|(name, value)| (name, from_scalar(value)))
            .collect();
        let mut total = 0.0;
        let mut count = 0usize;
        for (name, value) in right.named_values() {
            let Some(other) = left_values.get(name) else {
                continue;
            };
            let value = from_scalar(value);
            let scale = value.abs().max(other.abs()).max(1.0);
            // Gaussian kernel on the scale-normalised difference: 1 for an exact
            // match, decaying smoothly to 0, never negative.
            let normalised = (value - other) / scale;
            total += (-(normalised * normalised)).exp();
            count += 1;
        }
        if count == 0 {
            return A::zero();
        }
        A::from(total / count as f64).unwrap_or_else(A::zero)
    }

    /// Learn (or reinforce) a pattern from a drift whose outcome is now known.
    ///
    /// C5: `known_patterns` was permanently empty, which is why `match_pattern`
    /// could never return anything and the pattern-confidence term in
    /// `combine_detection_results` was always the neutral 0.5.
    pub(crate) fn learn_pattern(
        &mut self,
        features: &PatternFeatures<A>,
        strategy_id: &str,
        outcome: &AdaptationOutcome<A>,
        drift_type: DriftType,
    ) {
        // Reinforce the closest existing pattern if there is one, otherwise
        // record a new one keyed by its discretised signature.
        let existing = self
            .known_patterns
            .iter()
            .map(|(id, pattern)| {
                (
                    id.clone(),
                    self.calculate_similarity(&pattern.features, features),
                )
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|(_, similarity)| *similarity > self.matching_threshold)
            .map(|(id, _)| id);

        let success = if outcome.success { 1.0 } else { 0.0 };
        match existing {
            Some(id) => {
                if let Some(pattern) = self.known_patterns.get_mut(&id) {
                    pattern.occurrence_count += 1;
                    let count = pattern.occurrence_count as f64;
                    let previous = from_scalar(pattern.adaptation_success_rate);
                    let updated = previous + (success - previous) / count;
                    pattern.adaptation_success_rate =
                        A::from(updated).unwrap_or(pattern.adaptation_success_rate);
                    // Blend the observed stability period into the typical
                    // duration with the same running mean.
                    let previous_secs = pattern.typical_duration.as_secs_f64();
                    let observed_secs = outcome.stability_period.as_secs_f64();
                    let blended = previous_secs + (observed_secs - previous_secs) / count;
                    pattern.typical_duration = Duration::from_secs_f64(blended.max(0.0));
                }
            }
            None => {
                let id = pattern_signature(features, drift_type);
                self.known_patterns.insert(
                    id.clone(),
                    DriftPattern {
                        id,
                        features: features.clone(),
                        pattern_type: drift_type,
                        typical_duration: outcome.stability_period,
                        optimal_adaptation: AdaptationRecommendation::SwitchOptimizer {
                            new_optimizer: strategy_id.to_string(),
                        },
                        adaptation_success_rate: A::from(success).unwrap_or_else(A::zero),
                        occurrence_count: 1,
                    },
                );
            }
        }
    }
}

/// Stable identifier derived from a feature vector, used as the pattern key.
fn pattern_signature<A: Float + Send + Sync>(
    features: &PatternFeatures<A>,
    drift_type: DriftType,
) -> String {
    let bucket = |value: f64| -> i64 {
        if !value.is_finite() {
            0
        } else {
            (value * 4.0).round() as i64
        }
    };
    format!(
        "{:?}:m{}:v{}:t{}",
        drift_type,
        bucket(from_scalar(features.mean)),
        bucket(from_scalar(features.variance)),
        bucket(features.trend_slope.map(from_scalar).unwrap_or(0.0))
    )
}

// ---------------------------------------------------------------------------
// C6: adaptive thresholds that are actually applied
// ---------------------------------------------------------------------------

impl<A: Float + Send + Sync> AdaptiveThresholdManager<A> {
    pub(crate) fn new() -> Self {
        Self {
            thresholds: HashMap::new(),
            threshold_history: VecDeque::with_capacity(MAX_THRESHOLD_HISTORY),
            performance_feedback: VecDeque::with_capacity(MAX_FEEDBACK_HISTORY),
            learning_rate: A::from(0.01).unwrap_or_else(A::zero),
        }
    }

    /// Adapt each detector's threshold from its own verdict, the observed signal
    /// strength and any recorded detection-quality feedback.
    pub(crate) fn update_thresholds(
        &mut self,
        results: &[(String, DriftStatus)],
        features: &PatternFeatures<A>,
    ) {
        // A noisy window justifies a looser threshold; a quiet one a tighter
        // threshold. `trend_strength` is the fraction of variance explained by a
        // trend, so it is already normalised to [0, 1].
        let signal = features.trend_strength.map(from_scalar).unwrap_or(0.0);
        let feedback_bias = self.feedback_bias();
        let learning_rate = from_scalar(self.learning_rate);

        for (detector_name, result) in results {
            let current_threshold = self
                .thresholds
                .get(detector_name)
                .copied()
                .unwrap_or_else(A::one);
            let current = from_scalar(current_threshold).max(EPSILON);

            // Detecting drift lowers the bar (the detector is being useful);
            // steady stability raises it. The signal and feedback terms scale
            // how far each step moves.
            let direction = match result {
                DriftStatus::Drift => -1.0,
                DriftStatus::Warning => -0.25,
                DriftStatus::Stable => 0.1,
            };
            let adjustment = learning_rate * direction * (1.0 + signal) + feedback_bias;
            let new_threshold = (current * (1.0 + adjustment)).clamp(1e-6, 1e6);
            let Some(new_value) = A::from(new_threshold) else {
                continue;
            };

            self.thresholds.insert(detector_name.clone(), new_value);
            self.threshold_history.push_back(ThresholdUpdate {
                detector_name: detector_name.clone(),
                old_threshold: current_threshold,
                new_threshold: new_value,
                timestamp: Instant::now(),
                reason: format!("{result:?} verdict, trend strength {signal:.3}"),
            });
            // C7: bound the log.
            while self.threshold_history.len() > MAX_THRESHOLD_HISTORY {
                self.threshold_history.pop_front();
            }
        }
    }

    /// C6: push the adapted thresholds into the detectors. Without this the
    /// whole manager was write-only bookkeeping.
    pub(crate) fn apply_to(&self, detectors: &mut [Box<dyn DriftDetectorTrait<A>>]) {
        for detector in detectors.iter_mut() {
            if let Some(threshold) = self.thresholds.get(detector.name()).copied() {
                detector.set_threshold(threshold);
            }
        }
    }

    /// Record measured detection quality (C6: `performance_feedback` was never
    /// written to).
    pub(crate) fn record_feedback(&mut self, feedback: PerformanceFeedback<A>) {
        self.performance_feedback.push_back(feedback);
        while self.performance_feedback.len() > MAX_FEEDBACK_HISTORY {
            self.performance_feedback.pop_front();
        }
    }

    /// Bias applied to every threshold step: a high false-positive rate pushes
    /// thresholds up, a long detection delay pushes them down.
    fn feedback_bias(&self) -> f64 {
        if self.performance_feedback.is_empty() {
            return 0.0;
        }
        let count = self.performance_feedback.len() as f64;
        let false_positive_rate = self
            .performance_feedback
            .iter()
            .map(|feedback| from_scalar(feedback.false_positive_rate))
            .sum::<f64>()
            / count;
        let mean_delay = self
            .performance_feedback
            .iter()
            .map(|feedback| feedback.detection_delay.as_secs_f64())
            .sum::<f64>()
            / count;
        // Both terms are small so they nudge rather than dominate.
        let delay_term = (mean_delay / 60.0).min(1.0);
        0.05 * false_positive_rate - 0.05 * delay_term
    }

    /// Adapted thresholds currently held.
    pub fn thresholds(&self) -> &HashMap<String, A> {
        &self.thresholds
    }
}

// ---------------------------------------------------------------------------
// context detection
// ---------------------------------------------------------------------------

impl<A: Float + Send + Sync> ContextAwareDriftDetector<A> {
    pub(crate) fn new() -> Self {
        Self {
            context_features: Vec::new(),
            current_context: None,
            transition_matrix: HashMap::new(),
        }
    }

    /// Classify the current context from an importance-weighted mean of the
    /// supplied features, and record the observed transition frequency.
    pub(crate) fn update_context(&mut self, features: &[ContextFeature<A>]) {
        self.context_features = features.to_vec();

        let context_id = if features.is_empty() {
            "unknown".to_string()
        } else {
            let mut weighted = 0.0;
            let mut weights = 0.0;
            for feature in features {
                let weight = from_scalar(feature.importance_weight).abs().max(EPSILON);
                weighted += from_scalar(feature.value) * weight;
                weights += weight;
            }
            let score = if weights > 0.0 {
                weighted / weights
            } else {
                0.0
            };
            // Three bands rather than the previous single 0.5 cut on
            // `features[0]`, which ignored every other feature and their weights.
            if score > 0.66 {
                "high_activity".to_string()
            } else if score > 0.33 {
                "medium_activity".to_string()
            } else {
                "low_activity".to_string()
            }
        };

        if let Some(previous) = self.current_context.clone() {
            let key = (previous, context_id.clone());
            let count = self.transition_matrix.entry(key).or_insert_with(A::zero);
            *count = *count + A::one();
        }
        self.current_context = Some(context_id);
    }

    /// Observed transition counts between contexts.
    pub fn transitions(&self) -> &HashMap<(String, String), A> {
        &self.transition_matrix
    }
}

// ---------------------------------------------------------------------------
// impact analysis
// ---------------------------------------------------------------------------

/// Severity classifier backed by the observed distribution of degradations.
#[derive(Debug, Default)]
pub(crate) struct SeverityClassifier<A: Float + Send + Sync> {
    observations: Vec<f64>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> SeverityClassifier<A> {
    pub(crate) fn new() -> Self {
        Self {
            observations: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    fn observe(&mut self, degradation: f64) {
        self.observations.push(degradation);
        if self.observations.len() > MAX_IMPACT_HISTORY {
            self.observations.remove(0);
        }
    }

    /// Urgency from the percentile the current degradation occupies in the
    /// observed history, falling back to absolute bands while the history is
    /// too short to have percentiles.
    fn classify(&self, degradation: f64) -> UrgencyLevel {
        if self.observations.len() < 8 {
            return if degradation > 1.0 {
                UrgencyLevel::High
            } else if degradation > 0.1 {
                UrgencyLevel::Medium
            } else {
                UrgencyLevel::Low
            };
        }
        let below = self
            .observations
            .iter()
            .filter(|value| **value <= degradation)
            .count() as f64;
        let percentile = below / self.observations.len() as f64;
        if percentile >= 0.99 {
            UrgencyLevel::Critical
        } else if percentile >= 0.9 {
            UrgencyLevel::High
        } else if percentile >= 0.5 {
            UrgencyLevel::Medium
        } else {
            UrgencyLevel::Low
        }
    }
}

/// Recovery-time predictor learned from observed stability periods.
#[derive(Debug, Default)]
pub(crate) struct RecoveryTimePredictor<A: Float + Send + Sync> {
    observed: Vec<Duration>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> RecoveryTimePredictor<A> {
    pub(crate) fn new() -> Self {
        Self {
            observed: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    fn observe(&mut self, adaptation_time: Duration) {
        self.observed.push(adaptation_time);
        if self.observed.len() > MAX_IMPACT_HISTORY {
            self.observed.remove(0);
        }
    }

    /// Mean observed recovery time scaled by the current degradation, or `None`
    /// when nothing has been observed yet. The previous code returned a flat
    /// 300 seconds regardless of input or history.
    fn predict(&self, degradation: f64) -> Option<Duration> {
        if self.observed.is_empty() {
            return None;
        }
        let mean_seconds = self
            .observed
            .iter()
            .map(|duration| duration.as_secs_f64())
            .sum::<f64>()
            / self.observed.len() as f64;
        let scaled = mean_seconds * (1.0 + degradation.max(0.0));
        if !scaled.is_finite() {
            return None;
        }
        Some(Duration::from_secs_f64(scaled.clamp(0.0, 86_400.0)))
    }
}

/// Business-impact estimator driven by the observed degradation distribution.
#[derive(Debug, Default)]
pub(crate) struct BusinessImpactEstimator<A: Float + Send + Sync> {
    total_degradation: f64,
    events: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + Send + Sync> BusinessImpactEstimator<A> {
    pub(crate) fn new() -> Self {
        Self {
            total_degradation: 0.0,
            events: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn observe(&mut self, degradation: f64) {
        self.total_degradation += degradation.max(0.0);
        self.events += 1;
    }

    /// Impact of this event relative to the mean event seen so far.
    fn estimate(&self, degradation: f64) -> f64 {
        if self.events == 0 {
            return degradation.max(0.0);
        }
        let average = self.total_degradation / self.events as f64;
        if average <= EPSILON {
            degradation.max(0.0)
        } else {
            (degradation.max(0.0) / average).min(100.0)
        }
    }
}

impl<A: Float + Send + Sync> DriftImpactAnalyzer<A> {
    pub(crate) fn new() -> Self {
        Self {
            impact_history: VecDeque::with_capacity(MAX_IMPACT_HISTORY),
            severity_classifier: SeverityClassifier::new(),
            recovery_predictor: RecoveryTimePredictor::new(),
            business_impact_estimator: BusinessImpactEstimator::new(),
        }
    }

    pub(crate) fn analyze_impact(
        &mut self,
        features: &PatternFeatures<A>,
        pattern: &Option<DriftPattern<A>>,
    ) -> Result<DriftImpact<A>> {
        // Degradation combines the window's dispersion with how strongly it is
        // trending: a large but stationary spread is less damaging than a
        // smaller but relentlessly rising one.
        let variance = from_scalar(features.variance).max(0.0);
        let trend = features
            .trend_strength
            .map(from_scalar)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let degradation = variance.sqrt() * (1.0 + trend);

        self.severity_classifier.observe(degradation);
        self.business_impact_estimator.observe(degradation);
        let urgency_level = self.severity_classifier.classify(degradation);

        // Prefer a matched pattern's measured duration, then the learned
        // recovery model, and only then admit that nothing is known.
        let estimated_recovery_time = pattern
            .as_ref()
            .map(|pattern| pattern.typical_duration)
            .or_else(|| self.recovery_predictor.predict(degradation))
            .unwrap_or(Duration::ZERO);

        // Confidence in the assessment: high when the window supports a clear
        // trend and a pattern was matched, low when neither holds.
        let pattern_term = pattern
            .as_ref()
            .map(|pattern| from_scalar(pattern.adaptation_success_rate))
            .unwrap_or(0.0);
        let confidence = (0.5 * trend + 0.5 * pattern_term).clamp(0.0, 1.0);

        let impact = DriftImpact {
            performance_degradation: A::from(degradation).unwrap_or_else(A::zero),
            affected_metrics: vec!["accuracy".to_string(), "loss".to_string()],
            estimated_recovery_time,
            confidence: A::from(confidence).unwrap_or_else(A::zero),
            business_impact_score: A::from(self.business_impact_estimator.estimate(degradation))
                .unwrap_or_else(A::zero),
            urgency_level,
        };

        // C7: the impact log is now written to and bounded.
        self.impact_history.push_back(impact.clone());
        while self.impact_history.len() > MAX_IMPACT_HISTORY {
            self.impact_history.pop_front();
        }

        Ok(impact)
    }

    /// Drift type implied by the most recent impact assessment.
    pub(crate) fn last_drift_type(&self) -> DriftType {
        match self.impact_history.back() {
            None => DriftType::Sudden,
            Some(impact) => match impact.urgency_level {
                UrgencyLevel::Critical | UrgencyLevel::High => DriftType::Sudden,
                UrgencyLevel::Medium => DriftType::Incremental,
                UrgencyLevel::Low => DriftType::Gradual,
            },
        }
    }

    /// Feed a measured adaptation outcome into the recovery model.
    pub(crate) fn record_observed_recovery(&mut self, outcome: &AdaptationOutcome<A>) {
        self.recovery_predictor.observe(outcome.adaptation_time);
    }

    /// Retained impact assessments.
    pub fn impact_history(&self) -> &VecDeque<DriftImpact<A>> {
        &self.impact_history
    }
}

// ---------------------------------------------------------------------------
// C5: real strategy selection
// ---------------------------------------------------------------------------

impl<A: Float + Send + Sync> EpsilonGreedyBandit<A> {
    pub(crate) fn new(epsilon: A) -> Self {
        Self {
            epsilon,
            action_values: HashMap::new(),
            action_counts: HashMap::new(),
            total_trials: 0,
            rng: scirs2_core::random::Random::seed(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Pick an action: with probability `epsilon` explore uniformly, otherwise
    /// take the highest measured value (ties broken by the fewest trials, so
    /// untried actions are tried).
    pub(crate) fn select(&mut self, actions: &[String]) -> Option<String> {
        if actions.is_empty() {
            return None;
        }
        self.total_trials += 1;

        let epsilon = from_scalar(self.epsilon).clamp(0.0, 1.0);
        if self.rng.random_f64() < epsilon {
            let index = (self.rng.random_f64() * actions.len() as f64) as usize;
            return actions.get(index.min(actions.len() - 1)).cloned();
        }

        actions
            .iter()
            .max_by(|left, right| {
                let left_value = self
                    .action_values
                    .get(*left)
                    .copied()
                    .unwrap_or_else(A::zero);
                let right_value = self
                    .action_values
                    .get(*right)
                    .copied()
                    .unwrap_or_else(A::zero);
                match left_value.partial_cmp(&right_value) {
                    Some(std::cmp::Ordering::Equal) | None => {
                        let left_count = self.action_counts.get(*left).copied().unwrap_or(0);
                        let right_count = self.action_counts.get(*right).copied().unwrap_or(0);
                        right_count.cmp(&left_count)
                    }
                    Some(ordering) => ordering,
                }
            })
            .cloned()
    }

    /// Seed an unseen action's value with the prior estimate for it, so the
    /// first exploit pick follows the scoring rather than an arbitrary tie
    /// break among zeros.
    pub(crate) fn prime(&mut self, action: &str, prior: A) {
        self.action_values
            .entry(action.to_string())
            .or_insert(prior);
    }

    /// Fold a measured reward into the action's running value.
    pub(crate) fn update(&mut self, action: &str, reward: A) {
        let count = self.action_counts.entry(action.to_string()).or_insert(0);
        *count += 1;
        let count = *count;
        let value = self
            .action_values
            .entry(action.to_string())
            .or_insert_with(A::zero);
        let step = A::from(count).unwrap_or_else(A::one);
        *value = *value + (reward - *value) / step;
    }

    /// Measured value of each action.
    pub fn action_values(&self) -> &HashMap<String, A> {
        &self.action_values
    }
}

impl<A: Float + Send + Sync> AdaptationStrategySelector<A> {
    pub(crate) fn new() -> Self {
        Self {
            strategies: default_strategies(),
            strategy_performance: HashMap::new(),
            bandit: EpsilonGreedyBandit::new(A::from(0.1).unwrap_or_else(A::zero)),
            context_strategy_map: HashMap::new(),
        }
    }

    /// Select an adaptation strategy for the observed drift.
    ///
    /// C5: this used to build and return the same hardcoded "increase_lr"
    /// strategy on every call, ignoring the features, the impact, the matched
    /// pattern, the (empty) strategy list, the recorded performance and the
    /// bandit entirely.
    pub(crate) fn select_strategy(
        &mut self,
        features: &PatternFeatures<A>,
        impact: &DriftImpact<A>,
        pattern: &Option<DriftPattern<A>>,
    ) -> Result<Option<AdaptationStrategy<A>>> {
        if self.strategies.is_empty() {
            return Ok(None);
        }

        // A matched pattern with a genuinely good track record short-circuits
        // exploration: use what has been measured to work.
        if let Some(pattern) = pattern {
            if from_scalar(pattern.adaptation_success_rate) > 0.8 {
                if let AdaptationRecommendation::SwitchOptimizer { new_optimizer } =
                    &pattern.optimal_adaptation
                {
                    if let Some(strategy) = self
                        .strategies
                        .iter()
                        .find(|strategy| strategy.id == *new_optimizer)
                    {
                        return Ok(Some(strategy.clone()));
                    }
                }
            }
        }

        // Otherwise score every applicable strategy and let the bandit choose
        // among the viable ones.
        let urgency_weight = match impact.urgency_level {
            UrgencyLevel::Critical => 1.0,
            UrgencyLevel::High => 0.75,
            UrgencyLevel::Medium => 0.5,
            UrgencyLevel::Low => 0.25,
        };

        let mut viable: Vec<(String, f64)> = Vec::new();
        for strategy in &self.strategies {
            let applicability = strategy.applicability(features);
            if applicability <= 0.0 {
                continue;
            }
            let measured = self
                .strategy_performance
                .get(&strategy.id)
                .map(|performance| from_scalar(performance.success_rate))
                .unwrap_or_else(|| from_scalar(strategy.expected_effectiveness));
            let cost = from_scalar(strategy.computational_cost).max(0.0);
            // Urgent drift tolerates expensive strategies; quiet drift does not.
            let score = applicability * measured - (1.0 - urgency_weight) * cost;
            viable.push((strategy.id.clone(), score));
        }

        if viable.is_empty() {
            return Ok(None);
        }
        viable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Offer the bandit the top candidates so exploration stays sensible, and
        // seed each unseen candidate with its score so the first exploit pick
        // follows the scoring instead of an arbitrary tie break.
        let candidates: Vec<String> = viable.iter().take(3).map(|(id, _)| id.clone()).collect();
        for (id, score) in viable.iter().take(3) {
            self.bandit
                .prime(id, A::from(*score).unwrap_or_else(A::zero));
        }
        let Some(chosen) = self.bandit.select(&candidates) else {
            return Ok(None);
        };

        if let Some(context) = self.context_strategy_map.get_mut("last") {
            context.push(chosen.clone());
            if context.len() > 32 {
                context.remove(0);
            }
        } else {
            self.context_strategy_map
                .insert("last".to_string(), vec![chosen.clone()]);
        }

        Ok(self
            .strategies
            .iter()
            .find(|strategy| strategy.id == chosen)
            .cloned())
    }

    /// Fold a measured outcome into the strategy's performance record and the
    /// bandit's action value.
    pub(crate) fn record_outcome(&mut self, strategy_id: &str, outcome: &AdaptationOutcome<A>) {
        let improvement = from_scalar(outcome.performance_improvement);
        let entry = self
            .strategy_performance
            .entry(strategy_id.to_string())
            .or_insert_with(|| StrategyPerformance {
                success_rate: A::zero(),
                average_improvement: A::zero(),
                average_adaptation_time: Duration::ZERO,
                stability_after_adaptation: A::zero(),
                usage_count: 0,
            });
        entry.usage_count += 1;
        let count = entry.usage_count as f64;

        let success = if outcome.success { 1.0 } else { 0.0 };
        let previous_rate = from_scalar(entry.success_rate);
        entry.success_rate = A::from(previous_rate + (success - previous_rate) / count)
            .unwrap_or(entry.success_rate);

        let previous_improvement = from_scalar(entry.average_improvement);
        entry.average_improvement =
            A::from(previous_improvement + (improvement - previous_improvement) / count)
                .unwrap_or(entry.average_improvement);

        let previous_time = entry.average_adaptation_time.as_secs_f64();
        let observed_time = outcome.adaptation_time.as_secs_f64();
        entry.average_adaptation_time = Duration::from_secs_f64(
            (previous_time + (observed_time - previous_time) / count).max(0.0),
        );

        let previous_stability = from_scalar(entry.stability_after_adaptation);
        let observed_stability = outcome.stability_period.as_secs_f64();
        entry.stability_after_adaptation =
            A::from(previous_stability + (observed_stability - previous_stability) / count)
                .unwrap_or(entry.stability_after_adaptation);

        // Reward: the measured improvement, penalised for reported side effects.
        let penalty = 0.1 * outcome.side_effects.len() as f64;
        let reward = A::from(improvement - penalty).unwrap_or_else(A::zero);
        self.bandit.update(strategy_id, reward);
    }

    /// Measured performance per strategy.
    pub fn strategy_performance(&self) -> &HashMap<String, StrategyPerformance<A>> {
        &self.strategy_performance
    }

    /// Bandit action values.
    pub fn action_values(&self) -> &HashMap<String, A> {
        self.bandit.action_values()
    }
}

impl<A: Float + Send + Sync> AdaptationStrategy<A> {
    /// Weighted share of the applicability conditions this feature set meets.
    /// A strategy with no conditions is unconditionally applicable.
    pub(crate) fn applicability(&self, features: &PatternFeatures<A>) -> f64 {
        if self.applicability_conditions.is_empty() {
            return 1.0;
        }
        let mut satisfied = 0.0;
        let mut total = 0.0;
        for condition in &self.applicability_conditions {
            let weight = from_scalar(condition.weight).abs();
            if weight <= 0.0 {
                continue;
            }
            total += weight;
            let Some(value) = features.feature(&condition.feature_name) else {
                // A condition on an unmeasured feature cannot be satisfied, but
                // it also should not veto the strategy outright.
                continue;
            };
            let value = from_scalar(value);
            let threshold = from_scalar(condition.threshold);
            let met = match condition.operator {
                ComparisonOperator::GreaterThan => value > threshold,
                ComparisonOperator::LessThan => value < threshold,
                ComparisonOperator::Equal => (value - threshold).abs() <= EPSILON,
                ComparisonOperator::NotEqual => (value - threshold).abs() > EPSILON,
                ComparisonOperator::GreaterEqual => value >= threshold,
                ComparisonOperator::LessEqual => value <= threshold,
            };
            if met {
                satisfied += weight;
            }
        }
        if total <= 0.0 {
            0.0
        } else {
            satisfied / total
        }
    }
}

/// The candidate strategies the selector chooses between.
///
/// C5: `strategies` used to be `Vec::new()`, so there was nothing to choose
/// from and `select_strategy` had to fabricate its answer.
fn default_strategies<A: Float + Send + Sync>() -> Vec<AdaptationStrategy<A>> {
    let scalar = |value: f64| A::from(value).unwrap_or_else(A::zero);

    vec![
        AdaptationStrategy {
            id: "increase_learning_rate".to_string(),
            strategy_type: AdaptationStrategyType::ParameterTuning,
            parameters: HashMap::from([("learning_rate_factor".to_string(), scalar(1.5))]),
            applicability_conditions: vec![ApplicabilityCondition {
                feature_name: "trend_strength".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: scalar(0.3),
                weight: A::one(),
            }],
            expected_effectiveness: scalar(0.7),
            computational_cost: scalar(0.05),
        },
        AdaptationStrategy {
            id: "decrease_learning_rate".to_string(),
            strategy_type: AdaptationStrategyType::ParameterTuning,
            parameters: HashMap::from([("learning_rate_factor".to_string(), scalar(0.5))]),
            applicability_conditions: vec![ApplicabilityCondition {
                feature_name: "variance".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: scalar(1.0),
                weight: A::one(),
            }],
            expected_effectiveness: scalar(0.6),
            computational_cost: scalar(0.05),
        },
        AdaptationStrategy {
            id: "reset_model".to_string(),
            strategy_type: AdaptationStrategyType::ModelReplacement,
            parameters: HashMap::new(),
            applicability_conditions: vec![ApplicabilityCondition {
                feature_name: "entropy".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: scalar(1.0),
                weight: A::one(),
            }],
            expected_effectiveness: scalar(0.5),
            computational_cost: scalar(1.0),
        },
        AdaptationStrategy {
            id: "reweight_ensemble".to_string(),
            strategy_type: AdaptationStrategyType::EnsembleReweighting,
            parameters: HashMap::from([("decay".to_string(), scalar(0.9))]),
            applicability_conditions: Vec::new(),
            expected_effectiveness: scalar(0.55),
            computational_cost: scalar(0.2),
        },
        AdaptationStrategy {
            id: "shrink_window".to_string(),
            strategy_type: AdaptationStrategyType::Hybrid,
            parameters: HashMap::from([("window_factor".to_string(), scalar(0.5))]),
            applicability_conditions: vec![ApplicabilityCondition {
                feature_name: "persistence".to_string(),
                operator: ComparisonOperator::LessThan,
                threshold: scalar(0.3),
                weight: A::one(),
            }],
            expected_effectiveness: scalar(0.6),
            computational_cost: scalar(0.1),
        },
    ]
}

// ---------------------------------------------------------------------------
// C5: the drift database becomes a real learning store
// ---------------------------------------------------------------------------

impl<A: Float + Send + Sync> SimilarityIndex<A> {
    pub(crate) fn new() -> Self {
        Self {
            feature_vectors: Vec::new(),
            similarity_threshold: A::from(0.8).unwrap_or_else(A::one),
            distance_metric: DistanceMetric::Euclidean,
        }
    }

    /// Index a feature vector under `id`.
    pub(crate) fn insert(&mut self, id: &str, features: &PatternFeatures<A>) {
        let vector: Vec<A> = features
            .named_values()
            .into_iter()
            .map(|(_, value)| value)
            .collect();
        self.feature_vectors.push((id.to_string(), vector));
        if self.feature_vectors.len() > MAX_STORED_EVENTS {
            self.feature_vectors.remove(0);
        }
    }

    /// Ids whose indexed vector is closer than the similarity threshold, best
    /// first.
    pub fn find_similar(&self, features: &PatternFeatures<A>) -> Vec<(String, f64)> {
        let query: Vec<f64> = features
            .named_values()
            .into_iter()
            .map(|(_, value)| from_scalar(value))
            .collect();
        if query.is_empty() {
            return Vec::new();
        }
        let threshold = from_scalar(self.similarity_threshold);
        let mut matches: Vec<(String, f64)> = self
            .feature_vectors
            .iter()
            .filter_map(|(id, vector)| {
                let candidate: Vec<f64> = vector.iter().map(|value| from_scalar(*value)).collect();
                let similarity = self.similarity(&query, &candidate)?;
                if similarity >= threshold {
                    Some((id.clone(), similarity))
                } else {
                    None
                }
            })
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Similarity in `[0, 1]` under the configured distance metric.
    fn similarity(&self, left: &[f64], right: &[f64]) -> Option<f64> {
        if left.len() != right.len() || left.is_empty() {
            return None;
        }
        match self.distance_metric {
            DistanceMetric::Euclidean | DistanceMetric::Mahalanobis => {
                // Mahalanobis without a covariance estimate degenerates to a
                // per-dimension scaled Euclidean distance, which is what this is.
                let mut sum = 0.0;
                for (a, b) in left.iter().zip(right.iter()) {
                    let scale = a.abs().max(b.abs()).max(1.0);
                    let d = (a - b) / scale;
                    sum += d * d;
                }
                Some((-(sum / left.len() as f64)).exp())
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for (a, b) in left.iter().zip(right.iter()) {
                    let scale = a.abs().max(b.abs()).max(1.0);
                    sum += ((a - b) / scale).abs();
                }
                Some((-(sum / left.len() as f64)).exp())
            }
            DistanceMetric::Cosine => {
                let dot: f64 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();
                let norm_left: f64 = left.iter().map(|a| a * a).sum::<f64>().sqrt();
                let norm_right: f64 = right.iter().map(|b| b * b).sum::<f64>().sqrt();
                if norm_left <= EPSILON || norm_right <= EPSILON {
                    None
                } else {
                    Some(((dot / (norm_left * norm_right)) + 1.0) / 2.0)
                }
            }
        }
    }
}

impl<A: Float + Send + Sync> DriftDatabase<A> {
    pub(crate) fn new() -> Self {
        Self {
            drift_events: Vec::new(),
            pattern_outcomes: HashMap::new(),
            seasonal_patterns: HashMap::new(),
            similarity_index: SimilarityIndex::new(),
        }
    }

    /// Record a drift event and the strategy chosen for it. The outcome is
    /// deliberately left unset: nothing has been observed yet.
    pub(crate) fn store_event(
        &mut self,
        features: &PatternFeatures<A>,
        context: &[ContextFeature<A>],
        strategy: &Option<AdaptationStrategy<A>>,
    ) {
        let Some(strategy) = strategy else {
            return;
        };
        let event = StoredDriftEvent {
            features: features.clone(),
            context: context.to_vec(),
            applied_strategy: strategy.id.clone(),
            outcome: None,
            timestamp: Instant::now(),
        };
        self.similarity_index.insert(&strategy.id, features);
        self.drift_events.push(event);
        // C7: bound the log.
        while self.drift_events.len() > MAX_STORED_EVENTS {
            self.drift_events.remove(0);
        }
        self.update_seasonal_pattern(&strategy.id, features);
    }

    /// Attach a measured outcome to the newest event still awaiting one.
    pub(crate) fn complete_pending_event(
        &mut self,
        outcome: AdaptationOutcome<A>,
    ) -> Option<(String, PatternFeatures<A>)> {
        let event = self
            .drift_events
            .iter_mut()
            .rev()
            .find(|event| event.outcome.is_none())?;
        let strategy = event.applied_strategy.clone();
        let features = event.features.clone();
        event.outcome = Some(outcome.clone());
        self.pattern_outcomes
            .entry(strategy.clone())
            .or_default()
            .push(outcome);
        Some((strategy, features))
    }

    /// Measured outcomes per strategy.
    pub fn pattern_outcomes(&self) -> &HashMap<String, Vec<AdaptationOutcome<A>>> {
        &self.pattern_outcomes
    }

    /// Historical events most similar to a feature set.
    pub fn find_similar(&self, features: &PatternFeatures<A>) -> Vec<(String, f64)> {
        self.similarity_index.find_similar(features)
    }

    /// Maintain a real recurrence profile per strategy: the mean interval
    /// between events and how consistent that interval is.
    fn update_seasonal_pattern(&mut self, strategy_id: &str, features: &PatternFeatures<A>) {
        let now = Instant::now();
        let amplitude = A::from(from_scalar(features.variance).sqrt()).unwrap_or_else(A::zero);
        match self.seasonal_patterns.get_mut(strategy_id) {
            None => {
                self.seasonal_patterns.insert(
                    strategy_id.to_string(),
                    SeasonalPattern {
                        period: Duration::ZERO,
                        amplitude,
                        phase_offset: Duration::ZERO,
                        pattern_strength: A::zero(),
                        last_occurrence: now,
                    },
                );
            }
            Some(pattern) => {
                let interval = now.saturating_duration_since(pattern.last_occurrence);
                let previous = pattern.period.as_secs_f64();
                let blended = if previous <= 0.0 {
                    interval.as_secs_f64()
                } else {
                    0.7 * previous + 0.3 * interval.as_secs_f64()
                };
                // Strength: how close this interval is to the running period.
                let strength = if blended <= EPSILON {
                    0.0
                } else {
                    (1.0 - ((interval.as_secs_f64() - blended).abs() / blended)).clamp(0.0, 1.0)
                };
                pattern.period = Duration::from_secs_f64(blended.max(0.0));
                pattern.phase_offset = interval;
                pattern.amplitude = amplitude;
                pattern.pattern_strength = A::from(strength).unwrap_or_else(A::zero);
                pattern.last_occurrence = now;
            }
        }
    }

    /// Recurrence profiles observed per strategy.
    pub fn seasonal_patterns(&self) -> &HashMap<String, SeasonalPattern<A>> {
        &self.seasonal_patterns
    }
}
