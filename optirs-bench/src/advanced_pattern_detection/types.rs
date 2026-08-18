//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use crate::regression_tester::distributions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

use super::constants::WAVELET_MIN_CONCENTRATION;
use super::types_7::{
    ARIMAFitter, ChangePointDetector, FrequencyCharacteristics, HjorthParameters,
    HypothesisTestType, PatternEvolution, WaveletType,
};

/// FFT processor for frequency analysis
#[derive(Debug)]
pub struct FFTProcessor {
    /// Window function type
    pub(super) window_type: WindowType,
}
impl FFTProcessor {
    pub(super) fn new() -> Self {
        Self {
            window_type: WindowType::Hanning,
        }
    }
    pub(super) fn analyze_frequencies(&self, signal: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        if signal.len() < 8 {
            return Ok(patterns);
        }
        let windowed_signal = self.apply_window(signal);
        let spectrum = self.compute_fft(&windowed_signal)?;
        let dominant_freqs = self.find_dominant_frequencies(&spectrum);
        if !dominant_freqs.is_empty() {
            patterns.push(AdvancedMemoryPattern {
                id: "fft_pattern".to_string(),
                pattern_type: AdvancedPatternType::Periodic {
                    fundamental_frequency: dominant_freqs[0],
                    harmonics: dominant_freqs[1..].to_vec(),
                    phase_shift: 0.0,
                    amplitude: spectrum.iter().sum::<f64>() / spectrum.len() as f64,
                },
                confidence: 0.8,
                signature: spectrum.clone(),
                description: "FFT-detected periodic pattern".to_string(),
                frequency_characteristics: FrequencyCharacteristics {
                    dominant_frequencies: dominant_freqs,
                    power_spectrum: spectrum,
                    spectral_centroid: 0.0,
                    spectral_bandwidth: 0.0,
                    spectral_rolloff: 0.0,
                    spectral_flux: 0.0,
                    zero_crossing_rate: 0.0,
                },
                statistical_properties: StatisticalProperties::default(),
                anomaly_score: 0.0,
                strength: 0.8,
                periodicity: None,
                trend: TrendInfo::default(),
                leak_indicators: Vec::new(),
                evolution: PatternEvolution::default(),
            });
        }
        Ok(patterns)
    }
    pub(super) fn apply_window(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        match &self.window_type {
            WindowType::Hanning => signal
                .iter()
                .enumerate()
                .map(|(i, &x)| x * 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                .collect(),
            WindowType::Hamming => signal
                .iter()
                .enumerate()
                .map(|(i, &x)| x * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
                .collect(),
            _ => signal.to_vec(),
        }
    }
    pub(super) fn compute_fft(&self, signal: &[f64]) -> Result<Vec<f64>> {
        Ok(distributions::magnitude_spectrum(signal).unwrap_or_default())
    }
    pub(super) fn find_dominant_frequencies(&self, spectrum: &[f64]) -> Vec<f64> {
        let mut freq_mag_pairs: Vec<(usize, f64)> = spectrum
            .iter()
            .enumerate()
            .map(|(i, &mag)| (i, mag))
            .collect();
        freq_mag_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        freq_mag_pairs
            .iter()
            .take(5)
            .map(|(i_, _)| *i_ as f64 / (2.0 * spectrum.len() as f64))
            .collect()
    }
}
/// Wavelet processor for time-frequency analysis
#[derive(Debug)]
pub struct WaveletProcessor {
    /// Wavelet type
    pub(super) wavelet_type: WaveletType,
    /// Number of decomposition levels
    pub(super) levels: usize,
}
impl WaveletProcessor {
    pub(super) fn new() -> Self {
        Self {
            wavelet_type: WaveletType::Daubechies { order: 4 },
            levels: 5,
        }
    }
    pub(super) fn analyze_wavelets(&self, signal: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        let family = match &self.wavelet_type {
            WaveletType::Haar => distributions::WaveletFamily::Haar,
            WaveletType::Daubechies { order } if *order <= 1 => distributions::WaveletFamily::Haar,
            _ => distributions::WaveletFamily::Daubechies4,
        };
        let Some(levels) = distributions::discrete_wavelet_transform(signal, family, self.levels)
        else {
            return Ok(patterns);
        };
        let energies: Vec<f64> = levels
            .iter()
            .map(|level| level.detail.iter().map(|d| d * d).sum::<f64>())
            .collect();
        let total: f64 = energies.iter().sum();
        // `total` could in principle be NaN; treat that the same as "no energy"
        // rather than silently comparing it as greater/less-than 0.0.
        if total.is_nan() || total <= 0.0 {
            return Ok(patterns);
        }
        let Some((dominant_level, &dominant_energy)) = energies
            .iter()
            .enumerate()
            .max_by(|a, b| distributions::compare_finite(a.1, b.1))
        else {
            return Ok(patterns);
        };
        let concentration = (dominant_energy / total).clamp(0.0, 1.0);
        if concentration < WAVELET_MIN_CONCENTRATION {
            return Ok(patterns);
        }
        let period = 2.0f64.powi(dominant_level as i32 + 1);
        let fundamental_frequency = if period > 0.0 { 1.0 / period } else { 0.0 };
        let detail_count = levels
            .get(dominant_level)
            .map(|level| level.detail.len())
            .unwrap_or(0)
            .max(1) as f64;
        let amplitude = (dominant_energy / detail_count).sqrt();
        patterns.push(AdvancedMemoryPattern {
            id: "wavelet_pattern".to_string(),
            pattern_type: AdvancedPatternType::Periodic {
                fundamental_frequency,
                harmonics: Vec::new(),
                phase_shift: 0.0,
                amplitude,
            },
            confidence: concentration,
            signature: energies.clone(),
            description: format!(
                "Wavelet detail energy concentrated at level {} (~{:.0}-sample scale)",
                dominant_level + 1,
                period
            ),
            frequency_characteristics: FrequencyCharacteristics {
                dominant_frequencies: vec![fundamental_frequency],
                power_spectrum: energies,
                ..FrequencyCharacteristics::default()
            },
            statistical_properties: StatisticalProperties::default(),
            anomaly_score: 0.0,
            strength: concentration,
            periodicity: Some(PeriodicityInfo {
                period,
                strength: concentration,
                phase_coherence: 0.0,
                stability: 0.0,
            }),
            trend: TrendInfo::default(),
            leak_indicators: Vec::new(),
            evolution: PatternEvolution::default(),
        });
        Ok(patterns)
    }
}
/// Advanced memory pattern with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMemoryPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern type classification
    pub pattern_type: AdvancedPatternType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern signature (feature vector)
    pub signature: Vec<f64>,
    /// Pattern description
    pub description: String,
    /// Frequency domain characteristics
    pub frequency_characteristics: FrequencyCharacteristics,
    /// Statistical properties
    pub statistical_properties: StatisticalProperties,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Pattern strength
    pub strength: f64,
    /// Periodicity information
    pub periodicity: Option<PeriodicityInfo>,
    /// Trend information
    pub trend: TrendInfo,
    /// Associated leak indicators
    pub leak_indicators: Vec<LeakIndicator>,
    /// Pattern evolution over time
    pub evolution: PatternEvolution,
}
/// Kalman filter for signal denoising
#[derive(Debug)]
pub struct KalmanFilter {
    /// State estimate
    pub(super) state: Vec<f64>,
    /// Covariance matrix
    pub(super) covariance: Vec<Vec<f64>>,
    /// Process noise
    pub(super) process_noise: f64,
    /// Measurement noise
    pub(super) measurement_noise: f64,
}
impl KalmanFilter {
    pub(super) fn new() -> Self {
        Self {
            state: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            process_noise: 0.01,
            measurement_noise: 0.1,
        }
    }
    pub(super) fn filter(&mut self, signal: &[f64]) -> Result<Vec<f64>> {
        let mut filtered = Vec::new();
        for &measurement in signal {
            self.predict();
            self.update(measurement);
            filtered.push(self.state[0]);
        }
        Ok(filtered)
    }
    pub(super) fn predict(&mut self) {
        self.state[0] += self.state[1];
        self.covariance[0][0] += self.process_noise;
        self.covariance[1][1] += self.process_noise;
    }
    pub(super) fn update(&mut self, measurement: f64) {
        let gain = self.covariance[0][0] / (self.covariance[0][0] + self.measurement_noise);
        let innovation = measurement - self.state[0];
        self.state[0] += gain * innovation;
        self.covariance[0][0] *= 1.0 - gain;
    }
}
/// Types of features to extract
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Statistical moments
    StatisticalMoments,
    /// Frequency domain features
    FrequencyDomain,
    /// Time domain features
    TimeDomain,
    /// Wavelet features
    WaveletFeatures,
    /// Fractal features
    FractalFeatures,
    /// Information theoretic features
    InformationTheoretic,
    /// Shape features
    ShapeFeatures,
}
/// Periodicity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicityInfo {
    /// Period length
    pub period: f64,
    /// Periodicity strength
    pub strength: f64,
    /// Phase coherence
    pub phase_coherence: f64,
    /// Period stability
    pub stability: f64,
}
/// Window function types for FFT
#[derive(Debug, Clone)]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    Kaiser { beta: f64 },
}
/// Leak indicator from pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakIndicator {
    /// Indicator type
    pub indicator_type: LeakIndicatorType,
    /// Strength of indicator (0.0 to 1.0)
    pub strength: f64,
    /// Time to critical threshold
    pub time_to_critical: Option<f64>,
    /// Confidence in indicator
    pub confidence: f64,
}
/// Trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInfo {
    /// Trend direction (-1: decreasing, 0: stable, 1: increasing)
    pub direction: f64,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend acceleration
    pub acceleration: f64,
    /// Trend stability
    pub stability: f64,
    /// Change points
    pub change_points: Vec<usize>,
}
/// Change point detection algorithms
#[derive(Debug, Clone)]
pub enum ChangePointAlgorithm {
    /// CUSUM algorithm
    CUSUM { threshold: f64 },
    /// PELT (Pruned Exact Linear Time)
    PELT { penalty: f64 },
    /// Binary segmentation
    BinarySegmentation { min_size: usize },
    /// Bayesian change point detection
    Bayesian { prior_scale: f64 },
}
/// Types of leak indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakIndicatorType {
    /// Monotonic increase in memory
    MonotonicIncrease,
    /// Accelerating growth
    AcceleratingGrowth,
    /// Irregular spikes
    IrregularSpikes,
    /// Baseline drift
    BaselineDrift,
    /// Fragmentation signature
    FragmentationSignature,
    /// Cache thrashing pattern
    CacheThrashing,
    /// Resource exhaustion pattern
    ResourceExhaustion,
}
/// Hypothesis test engine
#[derive(Debug)]
pub struct HypothesisTestEngine {
    /// Available test types
    pub(super) test_types: Vec<HypothesisTestType>,
}
impl HypothesisTestEngine {
    pub(super) fn new() -> Self {
        Self {
            test_types: vec![
                HypothesisTestType::MannKendall,
                HypothesisTestType::LjungBox,
            ],
        }
    }
    /// Two-sided p-value of the Mann-Kendall trend test, or `None` when this
    /// engine was not configured with [`HypothesisTestType::MannKendall`] in
    /// `test_types` (an engine can be built with a narrower allowlist than
    /// `new`'s default).
    ///
    /// Delegates to the validated implementation in
    /// `regression_tester::distributions`, which computes a real p-value with tie
    /// correction (the previous version returned a normalized `S` statistic in
    /// `[-1, 1]` that was incorrectly treated as a p-value). Returns `None` when
    /// there are fewer than three finite points (the test is undefined) or the
    /// series is constant (zero variance).
    pub(super) fn mann_kendall_test(&self, data: &[f64]) -> Option<f64> {
        if !self.test_types.contains(&HypothesisTestType::MannKendall) {
            return None;
        }
        distributions::mann_kendall(data).map(|result| result.p_value)
    }
}
/// Training example for pattern classifier
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input features
    pub features: Vec<f64>,
    /// Expected pattern type
    pub pattern_type: AdvancedPatternType,
    /// Confidence weight
    pub weight: f64,
}
/// Advanced pattern types with detailed classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdvancedPatternType {
    /// Linear growth pattern
    LinearGrowth {
        slope: f64,
        intercept: f64,
        r_squared: f64,
    },
    /// Exponential growth pattern
    ExponentialGrowth {
        growth_rate: f64,
        base_value: f64,
        doubling_time: f64,
    },
    /// Periodic pattern with harmonics
    Periodic {
        fundamental_frequency: f64,
        harmonics: Vec<f64>,
        phase_shift: f64,
        amplitude: f64,
    },
    /// Saw-tooth allocation/deallocation pattern
    SawTooth {
        peak_height: f64,
        cycle_duration: f64,
        duty_cycle: f64,
        baseline: f64,
    },
    /// Step function pattern
    StepFunction {
        step_size: f64,
        step_frequency: f64,
        plateaus: Vec<f64>,
    },
    /// Chaotic/fractal pattern
    Chaotic {
        lyapunov_exponent: f64,
        correlation_dimension: f64,
        hurst_exponent: f64,
    },
    /// Burst pattern
    Burst {
        burst_intensity: f64,
        burst_duration: f64,
        inter_burst_interval: f64,
        baseline_level: f64,
    },
    /// Memory leak signature
    LeakSignature {
        leak_rate: f64,
        leak_acceleration: f64,
        leak_confidence: f64,
    },
    /// Composite pattern (combination of multiple patterns)
    Composite {
        components: Vec<Box<AdvancedPatternType>>,
        weights: Vec<f64>,
    },
}
/// Statistical analysis configuration
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Significance level for tests
    pub significance_level: f64,
    /// Bootstrap iterations
    pub bootstrap_iterations: usize,
    /// Confidence interval level
    pub confidence_interval: f64,
}
/// Time series analyzer
#[derive(Debug)]
pub struct TimeSeriesAnalyzer {
    /// ARIMA model fitter
    pub(super) arima_fitter: ARIMAFitter,
    /// Change point detector
    pub(super) change_point_detector: ChangePointDetector,
}
impl TimeSeriesAnalyzer {
    pub(super) fn new() -> Self {
        Self {
            arima_fitter: ARIMAFitter::new(),
            change_point_detector: ChangePointDetector::new(),
        }
    }
    pub(super) fn analyze(&self, data: &[f64]) -> Result<Vec<AdvancedMemoryPattern>> {
        let mut patterns = Vec::new();
        if let Ok(arima_pattern) = self.arima_fitter.fit_and_analyze(data) {
            patterns.push(arima_pattern);
        }
        let change_points = self.change_point_detector.detect_change_points(data)?;
        if !change_points.is_empty() {
            patterns.push(AdvancedMemoryPattern {
                id: "change_points".to_string(),
                pattern_type: AdvancedPatternType::StepFunction {
                    step_size: 0.0,
                    step_frequency: change_points.len() as f64 / data.len() as f64,
                    plateaus: Vec::new(),
                },
                confidence: 0.7,
                signature: change_points.iter().map(|&x| x as f64).collect(),
                description: "Change point pattern detected".to_string(),
                frequency_characteristics: FrequencyCharacteristics::default(),
                statistical_properties: StatisticalProperties::default(),
                anomaly_score: 0.0,
                strength: 0.7,
                periodicity: None,
                trend: TrendInfo {
                    direction: 0.0,
                    strength: 0.0,
                    acceleration: 0.0,
                    stability: 0.0,
                    change_points,
                },
                leak_indicators: Vec::new(),
                evolution: PatternEvolution::default(),
            });
        }
        Ok(patterns)
    }
}
/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ARIMAParameters {
    /// Autoregressive order
    pub p: usize,
    /// Differencing order
    pub d: usize,
    /// Moving average order
    pub q: usize,
    /// Seasonal parameters
    pub seasonal: Option<SeasonalParameters>,
}
/// Statistical properties of patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProperties {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Entropy
    pub entropy: f64,
    /// Autocorrelation function
    pub autocorrelation: Vec<f64>,
    /// Partial autocorrelation
    pub partial_autocorrelation: Vec<f64>,
    /// Hjorth parameters
    pub hjorth_parameters: HjorthParameters,
}
/// Seasonal ARIMA parameters
#[derive(Debug, Clone)]
pub struct SeasonalParameters {
    /// Seasonal autoregressive order
    pub p: usize,
    /// Seasonal differencing order
    pub d: usize,
    /// Seasonal moving average order
    pub q: usize,
    /// Seasonal period
    pub period: usize,
}
/// Pattern frequency statistics
#[derive(Debug, Clone)]
pub struct PatternFrequencyStats {
    /// Occurrence count
    pub count: usize,
    /// Average confidence
    pub avg_confidence: f64,
    /// Last seen timestamp
    pub last_seen: u64,
    /// Context information
    pub contexts: Vec<String>,
}
/// Configuration for advanced pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPatternConfig {
    /// Enable machine learning classification
    pub enable_ml_classification: bool,
    /// Enable signal processing analysis
    pub enable_signal_processing: bool,
    /// Enable statistical pattern matching
    pub enable_statistical_matching: bool,
    /// Minimum pattern length for detection
    pub min_pattern_length: usize,
    /// Pattern matching threshold
    pub pattern_matching_threshold: f64,
    /// Feature extraction window size
    pub feature_window_size: usize,
    /// Maximum patterns to store
    pub max_patterns_stored: usize,
    /// Learning rate for adaptive patterns
    pub learning_rate: f64,
    /// Enable anomaly scoring
    pub enable_anomaly_scoring: bool,
    /// Enable trend forecasting
    pub enable_trend_forecasting: bool,
}
/// Pattern database for storing learned patterns
#[derive(Debug)]
pub struct PatternDatabase {
    /// Stored patterns
    pub(super) patterns: HashMap<String, AdvancedMemoryPattern>,
    /// Pattern frequency statistics
    pub(super) frequency_stats: HashMap<String, PatternFrequencyStats>,
}
impl PatternDatabase {
    pub(super) fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            frequency_stats: HashMap::new(),
        }
    }
}
/// Feature extractor for pattern analysis
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Feature types to extract
    pub(super) feature_types: Vec<FeatureType>,
}
impl FeatureExtractor {
    pub(super) fn new() -> Self {
        Self {
            feature_types: vec![
                FeatureType::StatisticalMoments,
                FeatureType::FrequencyDomain,
                FeatureType::TimeDomain,
            ],
        }
    }
    pub(super) fn extract_features(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        for feature_type in &self.feature_types {
            match feature_type {
                FeatureType::StatisticalMoments => {
                    features.extend(self.extract_statistical_moments(data)?);
                }
                FeatureType::FrequencyDomain => {
                    features.extend(self.extract_frequency_features(data)?);
                }
                FeatureType::TimeDomain => {
                    features.extend(self.extract_time_domain_features(data)?);
                }
                _ => {}
            }
        }
        Ok(features)
    }
    pub(super) fn extract_statistical_moments(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0; 4]);
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let skewness = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };
        let kurtosis = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };
        Ok(vec![mean, std_dev, skewness, kurtosis])
    }
    pub(super) fn extract_frequency_features(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.len() < 4 {
            return Ok(vec![0.0; 3]);
        }
        let mut zero_crossings = 0;
        for i in 1..data.len() {
            if (data[i] >= 0.0) != (data[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zero_crossing_rate = zero_crossings as f64 / data.len() as f64;
        let spectral_centroid = data
            .iter()
            .enumerate()
            .map(|(i, &x)| i as f64 * x.abs())
            .sum::<f64>()
            / data.iter().map(|&x| x.abs()).sum::<f64>().max(1.0);
        let total_energy = data.iter().map(|&x| x * x).sum::<f64>();
        let mut cumulative_energy = 0.0;
        let mut rolloff_index = 0;
        for (i, &x) in data.iter().enumerate() {
            cumulative_energy += x * x;
            if cumulative_energy >= 0.85 * total_energy {
                rolloff_index = i;
                break;
            }
        }
        let spectral_rolloff = rolloff_index as f64 / data.len() as f64;
        Ok(vec![
            zero_crossing_rate,
            spectral_centroid,
            spectral_rolloff,
        ])
    }
    pub(super) fn extract_time_domain_features(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0; 3]);
        }
        let energy = data.iter().map(|&x| x * x).sum::<f64>();
        let rms = (energy / data.len() as f64).sqrt();
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let peak_to_peak = max_val - min_val;
        Ok(vec![energy, rms, peak_to_peak])
    }
}
