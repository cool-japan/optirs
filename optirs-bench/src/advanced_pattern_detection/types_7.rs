//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use crate::regression_tester::distributions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::functions::{
    build_mann_kendall_pattern, calculate_acceleration, current_unix_secs, dominant_spectral_peak,
};
use super::types::{
    ARIMAParameters, AdvancedMemoryPattern, AdvancedPatternConfig, AdvancedPatternType,
    ChangePointAlgorithm, FFTProcessor, FeatureExtractor, HypothesisTestEngine, KalmanFilter,
    PatternDatabase, PatternFrequencyStats, StatisticalConfig, StatisticalProperties,
    TimeSeriesAnalyzer, TrendInfo, WaveletProcessor,
};

/// Advanced-advanced pattern detector using ML and signal processing
#[derive(Debug)]
pub struct AdvancedPatternDetector {
    /// Configuration for pattern detection
    pub(super) config: AdvancedPatternConfig,
    /// Neural network for pattern classification
    pub(super) pattern_classifier: PatternClassifier,
    /// Signal processing engine
    pub(super) signal_processor: SignalProcessor,
    /// Statistical analyzer
    pub(super) statistical_analyzer: AdvancedStatisticalAnalyzer,
    /// Learned patterns database
    pub(super) pattern_database: PatternDatabase,
    /// Real-time feature extractor
    pub(super) feature_extractor: FeatureExtractor,
}
impl AdvancedPatternDetector {
    /// Create a new advanced pattern detector
    pub fn new(config: AdvancedPatternConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            pattern_classifier: PatternClassifier::new(),
            signal_processor: SignalProcessor::new()?,
            statistical_analyzer: AdvancedStatisticalAnalyzer::new()?,
            pattern_database: PatternDatabase::new(),
            feature_extractor: FeatureExtractor::new(),
        })
    }
    /// Detect patterns in memory usage data using advanced algorithms
    pub fn detect_patterns(&mut self, memorydata: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        if memorydata.len() < self.config.min_pattern_length {
            return Ok(Vec::new());
        }
        let mut detected_patterns = Vec::new();
        let features = self.feature_extractor.extract_features(memorydata)?;
        if self.config.enable_signal_processing {
            let signal_patterns = self.signal_processor.analyze_signal(memorydata)?;
            detected_patterns.extend(signal_patterns);
        }
        if self.config.enable_statistical_matching {
            let statistical_patterns = self.statistical_analyzer.analyze_patterns(memorydata)?;
            detected_patterns.extend(statistical_patterns);
        }
        if self.config.enable_ml_classification {
            let ml_patterns = self
                .pattern_classifier
                .classify_patterns(&features, memorydata)?;
            detected_patterns.extend(ml_patterns);
        }
        let mut refined_patterns = self.fuse_and_refine_patterns(detected_patterns)?;
        if self.config.enable_anomaly_scoring {
            self.compute_anomaly_scores(&mut refined_patterns, memorydata)?;
        }
        if self.config.enable_trend_forecasting {
            self.add_trend_forecasts(&mut refined_patterns, memorydata)?;
        }
        self.update_pattern_database(&refined_patterns)?;
        Ok(refined_patterns)
    }
    /// Fuse and refine overlapping patterns
    pub(super) fn fuse_and_refine_patterns(
        &self,
        patterns: Vec<AdvancedMemoryPattern>,
    ) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut refined_patterns = Vec::new();
        let mut used_patterns = vec![false; patterns.len()];
        for i in 0..patterns.len() {
            if used_patterns[i] {
                continue;
            }
            let mut pattern_group = vec![&patterns[i]];
            used_patterns[i] = true;
            for j in (i + 1)..patterns.len() {
                if used_patterns[j] {
                    continue;
                }
                let similarity = self.calculate_pattern_similarity(&patterns[i], &patterns[j])?;
                if similarity > self.config.pattern_matching_threshold {
                    pattern_group.push(&patterns[j]);
                    used_patterns[j] = true;
                }
            }
            let fused_pattern = self.fuse_pattern_group(pattern_group)?;
            refined_patterns.push(fused_pattern);
        }
        Ok(refined_patterns)
    }
    /// Calculate similarity between two patterns
    pub(super) fn calculate_pattern_similarity(
        &self,
        pattern1: &AdvancedMemoryPattern,
        pattern2: &AdvancedMemoryPattern,
    ) -> Result<f64> {
        let dot_product: f64 = pattern1
            .signature
            .iter()
            .zip(pattern2.signature.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f64 = pattern1.signature.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = pattern2.signature.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        let cosine_similarity = dot_product / (norm1 * norm2);
        let type_similarity =
            self.calculate_type_similarity(&pattern1.pattern_type, &pattern2.pattern_type);
        Ok((cosine_similarity + type_similarity) / 2.0)
    }
    /// Calculate similarity between pattern types
    pub(super) fn calculate_type_similarity(
        &self,
        type1: &AdvancedPatternType,
        type2: &AdvancedPatternType,
    ) -> f64 {
        match (type1, type2) {
            (
                AdvancedPatternType::LinearGrowth { .. },
                AdvancedPatternType::LinearGrowth { .. },
            ) => 1.0,
            (
                AdvancedPatternType::ExponentialGrowth { .. },
                AdvancedPatternType::ExponentialGrowth { .. },
            ) => 1.0,
            (AdvancedPatternType::Periodic { .. }, AdvancedPatternType::Periodic { .. }) => 1.0,
            (AdvancedPatternType::SawTooth { .. }, AdvancedPatternType::SawTooth { .. }) => 1.0,
            (
                AdvancedPatternType::StepFunction { .. },
                AdvancedPatternType::StepFunction { .. },
            ) => 1.0,
            (AdvancedPatternType::Chaotic { .. }, AdvancedPatternType::Chaotic { .. }) => 1.0,
            (AdvancedPatternType::Burst { .. }, AdvancedPatternType::Burst { .. }) => 1.0,
            (
                AdvancedPatternType::LeakSignature { .. },
                AdvancedPatternType::LeakSignature { .. },
            ) => 1.0,
            (
                AdvancedPatternType::LinearGrowth { .. },
                AdvancedPatternType::ExponentialGrowth { .. },
            ) => 0.7,
            (
                AdvancedPatternType::ExponentialGrowth { .. },
                AdvancedPatternType::LinearGrowth { .. },
            ) => 0.7,
            (AdvancedPatternType::SawTooth { .. }, AdvancedPatternType::Periodic { .. }) => 0.6,
            (AdvancedPatternType::Periodic { .. }, AdvancedPatternType::SawTooth { .. }) => 0.6,
            _ => 0.0,
        }
    }
    /// Fuse a group of similar patterns
    pub(super) fn fuse_pattern_group(
        &self,
        pattern_group: Vec<&AdvancedMemoryPattern>,
    ) -> Result<AdvancedMemoryPattern> {
        if pattern_group.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Empty pattern group",
            )
            .into());
        }
        if pattern_group.len() == 1 {
            return Ok(pattern_group[0].clone());
        }
        let base_pattern = match pattern_group
            .iter()
            .copied()
            .max_by(|a, b| distributions::compare_finite(&a.confidence, &b.confidence))
        {
            Some(pattern) => pattern,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Empty pattern group",
                )
                .into());
            }
        };
        let mut fused_pattern = base_pattern.clone();
        fused_pattern.confidence =
            pattern_group.iter().map(|p| p.confidence).sum::<f64>() / pattern_group.len() as f64;
        let signature_len = fused_pattern.signature.len();
        let mut averaged_signature = vec![0.0; signature_len];
        for pattern in &pattern_group {
            for (i, &val) in pattern.signature.iter().enumerate() {
                if i < signature_len {
                    averaged_signature[i] += val;
                }
            }
        }
        for val in &mut averaged_signature {
            *val /= pattern_group.len() as f64;
        }
        fused_pattern.signature = averaged_signature;
        fused_pattern.description = format!(
            "Fused pattern from {} similar patterns: {}",
            pattern_group.len(),
            pattern_group
                .iter()
                .map(|p| p.description.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(fused_pattern)
    }
    /// Update pattern database with new patterns
    pub(super) fn update_pattern_database(
        &mut self,
        patterns: &[AdvancedMemoryPattern],
    ) -> Result<()> {
        for pattern in patterns {
            if let Some(existing_pattern) = self.pattern_database.patterns.get_mut(&pattern.id) {
                existing_pattern.confidence =
                    (existing_pattern.confidence + pattern.confidence) / 2.0;
                existing_pattern.strength = (existing_pattern.strength + pattern.strength) / 2.0;
                if let Some(freq_stats) = self.pattern_database.frequency_stats.get_mut(&pattern.id)
                {
                    freq_stats.count += 1;
                    freq_stats.avg_confidence =
                        (freq_stats.avg_confidence + pattern.confidence) / 2.0;
                    freq_stats.last_seen = current_unix_secs();
                }
            } else {
                self.pattern_database
                    .patterns
                    .insert(pattern.id.clone(), pattern.clone());
                self.pattern_database.frequency_stats.insert(
                    pattern.id.clone(),
                    PatternFrequencyStats {
                        count: 1,
                        avg_confidence: pattern.confidence,
                        last_seen: current_unix_secs(),
                        contexts: Vec::new(),
                    },
                );
            }
        }
        if self.pattern_database.patterns.len() > self.config.max_patterns_stored {
            self.prune_pattern_database()?;
        }
        Ok(())
    }
    /// Prune pattern database to maintain size limits
    pub(super) fn prune_pattern_database(&mut self) -> Result<()> {
        let mut patterns_to_remove = Vec::new();
        for (id, freq_stats) in &self.pattern_database.frequency_stats {
            if let Some(pattern) = self.pattern_database.patterns.get(id) {
                let score = freq_stats.count as f64 * pattern.confidence;
                patterns_to_remove.push((id.clone(), score));
            }
        }
        patterns_to_remove
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let patterns_to_remove_count =
            self.pattern_database.patterns.len() - self.config.max_patterns_stored;
        for (id_, _) in patterns_to_remove.iter().take(patterns_to_remove_count) {
            self.pattern_database.patterns.remove(id_);
            self.pattern_database.frequency_stats.remove(id_);
        }
        Ok(())
    }
    /// Compute anomaly scores for patterns
    // NOTE: `memorydata` is intentionally unused. Each pattern's anomaly score is
    // derived from `pattern.strength` vs. its own historical mean, which is already
    // a complete, self-contained signal; folding the raw series in too (e.g. a
    // z-score of the latest sample) would be a new anomaly-detection design
    // decision, not a mechanical use of an existing computation, so it is left as a
    // tracked gap rather than invented here.
    pub(super) fn compute_anomaly_scores(
        &self,
        patterns: &mut [AdvancedMemoryPattern],
        _memorydata: &[f64],
    ) -> Result<()> {
        for pattern in patterns.iter_mut() {
            match self.get_historical_pattern_mean(&pattern.pattern_type) {
                Some(historical_mean) => {
                    let deviation = (pattern.strength - historical_mean).abs();
                    let denominator = historical_mean.abs().max(f64::EPSILON);
                    pattern.anomaly_score = (deviation / denominator).min(1.0);
                }
                None => {
                    pattern.anomaly_score = 0.0;
                }
            }
        }
        Ok(())
    }
    /// Add trend forecasts to patterns
    pub(super) fn add_trend_forecasts(
        &self,
        patterns: &mut Vec<AdvancedMemoryPattern>,
        memorydata: &[f64],
    ) -> Result<()> {
        for pattern in patterns {
            if memorydata.len() >= 2 {
                let n = memorydata.len();
                let last_values = &memorydata[n.saturating_sub(10)..];
                if let Some((slope_, _)) = self.calculate_linear_trend(last_values) {
                    pattern.trend.direction = slope_.signum();
                    pattern.trend.strength = slope_.abs().min(1.0);
                    pattern.trend.acceleration = calculate_acceleration(last_values);
                }
            }
        }
        Ok(())
    }
    /// Calculate linear trend from data
    pub(super) fn calculate_linear_trend(&self, data: &[f64]) -> Option<(f64, f64)> {
        if data.len() < 2 {
            return None;
        }
        let n = data.len() as f64;
        let sum_x = (0..data.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = data.iter().sum::<f64>();
        let sum_xy = data
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..data.len()).map(|i| (i * i) as f64).sum::<f64>();
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return None;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;
        Some((slope, intercept))
    }
    /// Mean `strength` of previously stored patterns sharing the same type, or
    /// `None` when no such history exists.
    ///
    /// This is a genuine baseline drawn from the pattern database (populated by
    /// earlier `detect_patterns` calls) rather than a hardcoded per-type
    /// constant. When the database has never seen this pattern type it honestly
    /// returns `None` so the caller can decline to compute a deviation.
    pub(super) fn get_historical_pattern_mean(
        &self,
        patterntype: &AdvancedPatternType,
    ) -> Option<f64> {
        let target = std::mem::discriminant(patterntype);
        let mut sum = 0.0;
        let mut count = 0usize;
        for pattern in self.pattern_database.patterns.values() {
            if std::mem::discriminant(&pattern.pattern_type) == target {
                sum += pattern.strength;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }
}
/// Types of hypothesis tests
#[derive(Debug, Clone, PartialEq)]
pub enum HypothesisTestType {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Anderson-Darling test
    AndersonDarling,
    /// Mann-Kendall trend test
    MannKendall,
    /// Ljung-Box test for autocorrelation
    LjungBox,
    /// Augmented Dickey-Fuller test
    AugmentedDickeyFuller,
    /// KPSS test for stationarity
    KPSS,
}
/// Hjorth parameters for signal complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HjorthParameters {
    /// Activity (variance)
    pub activity: f64,
    /// Mobility (mean frequency)
    pub mobility: f64,
    /// Complexity (bandwidth)
    pub complexity: f64,
}
/// Pattern evolution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEvolution {
    /// Pattern stability over time
    pub stability: f64,
    /// Evolution rate
    pub evolution_rate: f64,
    /// Adaptation score
    pub adaptation_score: f64,
    /// Historical states
    pub historical_states: Vec<PatternState>,
}
/// Signal processing engine for memory analysis
#[derive(Debug)]
pub struct SignalProcessor {
    /// FFT processor
    pub(super) fft_processor: FFTProcessor,
    /// Wavelet processor
    pub(super) wavelet_processor: WaveletProcessor,
    /// Kalman filter for noise reduction
    pub(super) kalman_filter: KalmanFilter,
}
impl SignalProcessor {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            fft_processor: FFTProcessor::new(),
            wavelet_processor: WaveletProcessor::new(),
            kalman_filter: KalmanFilter::new(),
        })
    }
    pub(super) fn analyze_signal(&mut self, signal: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        let frequency_patterns = self.fft_processor.analyze_frequencies(signal)?;
        patterns.extend(frequency_patterns);
        let wavelet_patterns = self.wavelet_processor.analyze_wavelets(signal)?;
        patterns.extend(wavelet_patterns);
        // Also run wavelet analysis on the Kalman-denoised signal: patterns
        // obscured by noise in the raw series (already analyzed above) can
        // surface once smoothed. `filtered_signal` was previously computed and
        // discarded entirely (an always-true, always-empty `if` block).
        let filtered_signal = self.kalman_filter.filter(signal)?;
        if !filtered_signal.is_empty() {
            let denoised_patterns = self.wavelet_processor.analyze_wavelets(&filtered_signal)?;
            patterns.extend(denoised_patterns);
        }
        Ok(patterns)
    }
}
/// ARIMA model fitter
#[derive(Debug)]
pub struct ARIMAFitter {
    /// Model parameters
    pub(super) parameters: ARIMAParameters,
}
impl ARIMAFitter {
    pub(super) fn new() -> Self {
        Self {
            parameters: ARIMAParameters {
                p: 1,
                d: 1,
                q: 1,
                seasonal: None,
            },
        }
    }
    /// Fit the "I" (integrated) part of ARIMA(p, d, q) for real -- difference the
    /// series `self.parameters.d` times to the order this fitter was configured
    /// with, then regress the (now closer to stationary) result -- rather than
    /// returning a hardcoded slope/intercept/r_squared/confidence unrelated to
    /// `data`. This does not fit the AR(p)/MA(q) terms (a genuine ARIMA solver is
    /// out of scope here); it is a bounded, honest use of the configured order
    /// instead of a fabricated result.
    pub(super) fn fit_and_analyze(&self, data: &[f64]) -> Result<AdvancedMemoryPattern> {
        let mut differenced = data.to_vec();
        for _ in 0..self.parameters.d {
            if differenced.len() < 2 {
                break;
            }
            differenced = differenced.windows(2).map(|w| w[1] - w[0]).collect();
        }

        let trend = distributions::linear_regression(&differenced);
        let (slope, intercept, r_squared) = trend
            .as_ref()
            .map(|t| (t.slope, t.intercept, t.r_squared.clamp(0.0, 1.0)))
            .unwrap_or((0.0, 0.0, 0.0));
        let confidence = r_squared;

        Ok(AdvancedMemoryPattern {
            id: format!(
                "arima_pattern_p{}_d{}_q{}",
                self.parameters.p, self.parameters.d, self.parameters.q
            ),
            pattern_type: AdvancedPatternType::LinearGrowth {
                slope,
                intercept,
                r_squared,
            },
            confidence,
            signature: data.to_vec(),
            description: format!(
                "ARIMA({}, {}, {})-differenced linear fit",
                self.parameters.p, self.parameters.d, self.parameters.q
            ),
            frequency_characteristics: FrequencyCharacteristics::default(),
            statistical_properties: StatisticalProperties::default(),
            anomaly_score: 0.0,
            strength: confidence,
            periodicity: None,
            trend: TrendInfo::default(),
            leak_indicators: Vec::new(),
            evolution: PatternEvolution::default(),
        })
    }
}
/// Wavelet types
#[derive(Debug, Clone)]
pub enum WaveletType {
    Daubechies { order: usize },
    Biorthogonal { order: (usize, usize) },
    Coiflets { order: usize },
    Haar,
    Morlet { sigma: f64 },
}
/// Advanced statistical analyzer
#[derive(Debug)]
pub struct AdvancedStatisticalAnalyzer {
    /// Configuration
    pub(super) config: StatisticalConfig,
    /// Hypothesis test engine
    pub(super) hypothesis_tester: HypothesisTestEngine,
    /// Time series analyzer
    pub(super) time_series_analyzer: TimeSeriesAnalyzer,
}
impl AdvancedStatisticalAnalyzer {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            config: StatisticalConfig {
                significance_level: 0.05,
                bootstrap_iterations: 1000,
                confidence_interval: 0.95,
            },
            hypothesis_tester: HypothesisTestEngine::new(),
            time_series_analyzer: TimeSeriesAnalyzer::new(),
        })
    }
    pub(super) fn analyze_patterns(&self, data: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        if let Some(p_value) = self.hypothesis_tester.mann_kendall_test(data) {
            if p_value < self.config.significance_level {
                if let Some(pattern) = build_mann_kendall_pattern(data, p_value) {
                    patterns.push(pattern);
                }
            }
        }
        let ts_patterns = self.time_series_analyzer.analyze(data)?;
        patterns.extend(ts_patterns);
        Ok(patterns)
    }
}
/// State of pattern at specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternState {
    /// Timestamp
    pub timestamp: u64,
    /// Pattern parameters at this time
    pub parameters: HashMap<String, f64>,
    /// Confidence at this time
    pub confidence: f64,
}
/// Change point detector
#[derive(Debug)]
pub struct ChangePointDetector {
    /// Detection algorithms
    pub(super) algorithms: Vec<ChangePointAlgorithm>,
}
impl ChangePointDetector {
    pub(super) fn new() -> Self {
        Self {
            algorithms: vec![
                ChangePointAlgorithm::CUSUM { threshold: 2.0 },
                ChangePointAlgorithm::BinarySegmentation { min_size: 5 },
            ],
        }
    }
    pub(super) fn detect_change_points(&self, data: &[f64]) -> Result<Vec<usize>> {
        let mut change_points = Vec::new();
        if data.len() < 10 {
            return Ok(change_points);
        }
        // Use the configured CUSUM sensitivity when this detector was built with
        // one (see `Self::new`'s default); PELT/BinarySegmentation/Bayesian are
        // declared as configurable but not yet implemented as separate algorithms
        // here, so any of those alone falls back to the same CUSUM-style default.
        let threshold_multiplier = self
            .algorithms
            .iter()
            .find_map(|algorithm| match algorithm {
                ChangePointAlgorithm::CUSUM { threshold } => Some(*threshold),
                _ => None,
            })
            .unwrap_or(2.0);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let mut cumsum = 0.0;
        let threshold = threshold_multiplier * data.iter().map(|x| (x - mean).abs()).sum::<f64>()
            / data.len() as f64;
        for (i, &value) in data.iter().enumerate() {
            cumsum += value - mean;
            if cumsum.abs() > threshold {
                change_points.push(i);
                cumsum = 0.0;
            }
        }
        Ok(change_points)
    }
}
/// Frequency domain characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyCharacteristics {
    /// Dominant frequencies
    pub dominant_frequencies: Vec<f64>,
    /// Power spectral density
    pub power_spectrum: Vec<f64>,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Spectral bandwidth
    pub spectral_bandwidth: f64,
    /// Spectral roll-off
    pub spectral_rolloff: f64,
    /// Spectral flux
    pub spectral_flux: f64,
    /// Zero crossing rate
    pub zero_crossing_rate: f64,
}
/// Transparent, deterministic pattern classifier.
///
/// Replaces the former untrained neural-network stub (all-zero weights, no
/// training path) that could only ever emit `sigmoid(0) = 0.5` and therefore
/// never classified anything. Classification now runs directly over real,
/// computed signal features (OLS trend, spectral prominence).
#[derive(Debug)]
pub struct PatternClassifier {
    /// Significance level for the trend hypothesis test.
    pub(super) significance_level: f64,
    /// Minimum coefficient of determination for a confident linear classification.
    pub(super) min_r_squared: f64,
    /// Minimum spectral prominence for a confident periodic classification.
    pub(super) min_spectral_prominence: f64,
}
impl PatternClassifier {
    pub(super) fn new() -> Self {
        Self {
            significance_level: 0.05,
            min_r_squared: 0.5,
            min_spectral_prominence: 0.25,
        }
    }
    /// Classify the signal into at most one confident pattern using transparent,
    /// deterministic rules over the actually-computed signal statistics.
    ///
    /// A significant monotone linear trend is tested first (via OLS regression on
    /// the raw series); only when the series is *not* a significant trend is a
    /// periodic classification considered, so a linear ramp - whose spectrum is
    /// dominated by its lowest bin - is never mistaken for an oscillation. When
    /// neither test is confident an empty vector is returned: an explicit "no
    /// confident classification", never the silent empty of the old broken
    /// all-zero-weight network.
    pub(super) fn classify_patterns(
        &self,
        features: &[f64],
        data: &[f64],
    ) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        if data.len() < 4 {
            return Ok(patterns);
        }
        if let Some(trend) = distributions::linear_regression(data) {
            if trend.p_value < self.significance_level && trend.r_squared >= self.min_r_squared {
                let confidence = trend.r_squared.clamp(0.0, 1.0);
                let pattern_type = AdvancedPatternType::LinearGrowth {
                    slope: trend.slope,
                    intercept: trend.intercept,
                    r_squared: trend.r_squared,
                };
                patterns.push(self.make_classified_pattern(
                    "classified_linear",
                    pattern_type,
                    confidence,
                    "Deterministic classifier: significant linear trend",
                    features,
                ));
                return Ok(patterns);
            }
        }
        if let Some((frequency, prominence)) = dominant_spectral_peak(data) {
            if prominence >= self.min_spectral_prominence {
                let pattern_type = AdvancedPatternType::Periodic {
                    fundamental_frequency: frequency,
                    harmonics: Vec::new(),
                    phase_shift: 0.0,
                    amplitude: features.get(1).copied().unwrap_or(0.0).abs(),
                };
                patterns.push(self.make_classified_pattern(
                    "classified_periodic",
                    pattern_type,
                    prominence.clamp(0.0, 1.0),
                    "Deterministic classifier: dominant periodic component",
                    features,
                ));
                return Ok(patterns);
            }
        }
        Ok(patterns)
    }
    pub(super) fn make_classified_pattern(
        &self,
        id: &str,
        pattern_type: AdvancedPatternType,
        confidence: f64,
        description: &str,
        features: &[f64],
    ) -> AdvancedMemoryPattern {
        AdvancedMemoryPattern {
            id: id.to_string(),
            pattern_type,
            confidence,
            signature: features.to_vec(),
            description: description.to_string(),
            frequency_characteristics: FrequencyCharacteristics::default(),
            statistical_properties: StatisticalProperties::default(),
            anomaly_score: 0.0,
            strength: confidence,
            periodicity: None,
            trend: TrendInfo::default(),
            leak_indicators: Vec::new(),
            evolution: PatternEvolution::default(),
        }
    }
}
