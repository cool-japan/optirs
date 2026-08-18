//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(dead_code)]
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::Instant;

use super::types_3::{
    AdaptiveThresholds, AnomalyAnalyzer, AnomalyClassifier, AnomalyConfig, AnomalyFeatures,
    BaselineStats, OutlierDetector,
};

/// Anomaly reporting and alerting system
pub struct AnomalyReporter<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) alerts: VecDeque<AnomalyAlert<T>>,
    pub(super) alert_history: HashMap<AnomalyType, Vec<Instant>>,
    pub(super) last_alert_times: HashMap<AnomalyType, Instant>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> AnomalyReporter<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            alerts: VecDeque::with_capacity(1000),
            alert_history: HashMap::new(),
            last_alert_times: HashMap::new(),
            _phantom: PhantomData,
        }
    }
    pub fn report_anomaly(&mut self, result: &AnomalyResult<T>) -> Option<AnomalyAlert<T>> {
        if !result.is_anomaly {
            return None;
        }
        let now = Instant::now();
        if let Some(&last_alert) = self.last_alert_times.get(&result.anomaly_type) {
            if now.duration_since(last_alert) < self.config.alert_cooldown {
                return None;
            }
        }
        let alert_id = format!(
            "anomaly_{}_{}",
            self.anomaly_type_to_string(&result.anomaly_type),
            now.elapsed().as_nanos()
        );
        let alert = AnomalyAlert {
            id: alert_id,
            anomaly_type: result.anomaly_type.clone(),
            severity: result.severity.clone(),
            timestamp: now,
            message: self.generate_alert_message(result),
            confidence: result.confidence,
            data_point: result.context.current_value,
            context: result.context.clone(),
            suggested_actions: result.suggested_actions.clone(),
            acknowledged: false,
            resolved: false,
        };
        self.alerts.push_back(alert.clone());
        if self.alerts.len() > 1000 {
            self.alerts.pop_front();
        }
        self.alert_history
            .entry(result.anomaly_type.clone())
            .or_default()
            .push(now);
        self.last_alert_times
            .insert(result.anomaly_type.clone(), now);
        Some(alert)
    }
    pub(super) fn generate_alert_message(&self, result: &AnomalyResult<T>) -> String {
        match result.anomaly_type {
            AnomalyType::StatisticalOutlier => {
                format!(
                    "Statistical outlier detected: value {:.4} deviates by {:.2} standard deviations",
                    result.context.current_value.to_f64().unwrap_or(0.0), (result.context
                    .deviation_magnitude / result.context.baseline_std).to_f64()
                    .unwrap_or(0.0)
                )
            }
            AnomalyType::TrendAnomaly => {
                format!(
                    "Trend anomaly detected: unexpected deviation of {:.4} from expected trend",
                    result.context.trend_deviation.to_f64().unwrap_or(0.0)
                )
            }
            AnomalyType::PatternAnomaly => {
                format!(
                    "Pattern anomaly detected: pattern match score {:.4} below threshold",
                    result.context.pattern_match_score.to_f64().unwrap_or(0.0)
                )
            }
            _ => {
                format!(
                    "{:?} anomaly detected with confidence {:.4}",
                    result.anomaly_type,
                    result.confidence.to_f64().unwrap_or(0.0)
                )
            }
        }
    }
    pub(super) fn anomaly_type_to_string(&self, anomaly_type: &AnomalyType) -> &str {
        match anomaly_type {
            AnomalyType::StatisticalOutlier => "statistical",
            AnomalyType::TrendAnomaly => "trend",
            AnomalyType::PerformanceAnomaly => "performance",
            AnomalyType::ConvergenceAnomaly => "convergence",
            AnomalyType::ResourceAnomaly => "resource",
            AnomalyType::PatternAnomaly => "pattern",
            AnomalyType::SeasonalAnomaly => "seasonal",
            AnomalyType::SystemAnomaly => "system",
        }
    }
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> bool {
        for alert in &mut self.alerts {
            if alert.id == alert_id {
                alert.acknowledged = true;
                return true;
            }
        }
        false
    }
    pub fn resolve_alert(&mut self, alert_id: &str) -> bool {
        for alert in &mut self.alerts {
            if alert.id == alert_id {
                alert.resolved = true;
                return true;
            }
        }
        false
    }
    pub fn get_active_alerts(&self) -> Vec<&AnomalyAlert<T>> {
        self.alerts.iter().filter(|alert| !alert.resolved).collect()
    }
    pub fn get_unacknowledged_alerts(&self) -> Vec<&AnomalyAlert<T>> {
        self.alerts
            .iter()
            .filter(|alert| !alert.acknowledged && !alert.resolved)
            .collect()
    }
    pub fn get_alert_statistics(&self) -> HashMap<AnomalyType, usize> {
        let mut stats = HashMap::new();
        for alert in &self.alerts {
            *stats.entry(alert.anomaly_type.clone()).or_insert(0) += 1;
        }
        stats
    }
    pub fn clear_resolved_alerts(&mut self) {
        self.alerts.retain(|alert| !alert.resolved);
    }
    pub fn generate_summary_report(&self) -> String {
        let total_alerts = self.alerts.len();
        let active_alerts = self.get_active_alerts().len();
        let unacknowledged_alerts = self.get_unacknowledged_alerts().len();
        let mut report = format!(
            "Anomaly Detection Summary:\n\
             - Total Alerts: {}\n\
             - Active Alerts: {}\n\
             - Unacknowledged Alerts: {}\n\
             \n\
             Alert Breakdown by Type:\n",
            total_alerts, active_alerts, unacknowledged_alerts
        );
        let stats = self.get_alert_statistics();
        for (anomaly_type, count) in stats {
            report.push_str(&format!("- {:?}: {}\n", anomaly_type, count));
        }
        report
    }
}
/// Time series analysis for anomaly detection
pub(super) struct TimeSeriesAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> TimeSeriesAnalyzer<T> {
    pub(super) fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }
    pub(super) fn analyze(&self, data: &[(Instant, T)]) -> Vec<AnomalyResult<T>> {
        let mut results = Vec::new();
        if data.len() < self.config.min_data_points {
            return results;
        }
        let decomposition = self.seasonal_decomposition(data);
        for (i, &(timestamp, value)) in data.iter().enumerate() {
            if i < decomposition.residuals.len() {
                let residual = decomposition.residuals[i];
                let threshold = self.compute_residual_threshold(&decomposition.residuals);
                if residual.abs() > threshold {
                    results.push(AnomalyResult {
                        is_anomaly: true,
                        anomaly_type: AnomalyType::SeasonalAnomaly,
                        severity: self.determine_severity(residual.abs(), threshold),
                        confidence: (residual.abs() / threshold).min(T::one()),
                        anomaly_score: residual.abs(),
                        timestamp,
                        context: AnomalyContext {
                            baseline_mean: T::zero(),
                            baseline_std: threshold,
                            current_value: value,
                            deviation_magnitude: residual.abs(),
                            trend_deviation: T::zero(),
                            pattern_match_score: T::zero(),
                            historical_frequency: T::zero(),
                        },
                        suggested_actions: vec![
                            "Investigate seasonal pattern disruption".to_string()
                        ],
                    });
                }
            }
        }
        results
    }
    pub(super) fn seasonal_decomposition(&self, data: &[(Instant, T)]) -> SeasonalDecomposition<T> {
        let values: Vec<T> = data.iter().map(|(_, v)| *v).collect();
        if values.len() < 10 {
            return SeasonalDecomposition {
                trend: values.clone(),
                seasonal: vec![T::zero(); values.len()],
                residuals: vec![T::zero(); values.len()],
            };
        }
        let window_size = (values.len() / 4).max(3);
        let mut trend = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(values.len());
            let window_mean = values[start..end].iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(end - start).unwrap_or_else(|| T::zero());
            trend.push(window_mean);
        }
        let detrended: Vec<T> = values
            .iter()
            .zip(trend.iter())
            .map(|(&val, &tr)| val - tr)
            .collect();
        let seasonal = match Self::estimate_seasonal_period(&detrended) {
            Some(period) if period >= 2 && period <= detrended.len() / 2 => {
                Self::extract_seasonal_component(&detrended, period)
            }
            _ => vec![T::zero(); values.len()],
        };
        let residuals: Vec<T> = detrended
            .iter()
            .zip(seasonal.iter())
            .map(|(&det, &seas)| det - seas)
            .collect();
        SeasonalDecomposition {
            trend,
            seasonal,
            residuals,
        }
    }
    /// Estimate the dominant seasonal period from the autocorrelation function of a
    /// (de-trended) series. Returns the lag `>= 2` of the first significant local
    /// ACF peak, or `None` when no clear periodicity is present.
    pub(super) fn estimate_seasonal_period(series: &[T]) -> Option<usize> {
        let n = series.len();
        if n < 4 {
            return None;
        }
        let mean = series.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(n).unwrap_or_else(|| T::one());
        let variance = series
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x);
        if variance <= T::epsilon() {
            return None;
        }
        let max_lag = n / 2;
        let mut acf = vec![T::zero(); max_lag + 1];
        acf[0] = T::one();
        for lag in 1..=max_lag {
            let mut sum = T::zero();
            for i in lag..n {
                sum = sum + (series[i] - mean) * (series[i - lag] - mean);
            }
            acf[lag] = sum / variance;
        }
        let significance = T::from(0.2).unwrap_or_else(|| T::zero());
        for lag in 2..max_lag {
            let val = acf[lag];
            let is_local_peak = val > acf[lag - 1] && val >= acf[lag + 1];
            if is_local_peak && val > significance {
                return Some(lag);
            }
        }
        None
    }
    /// Extract a centered additive seasonal component for the given period. The
    /// seasonal index for phase `j` is the mean of `detrended[i]` over all
    /// `i ≡ j (mod period)`; the indices are then centered (their mean subtracted)
    /// so the seasonal component sums to ~0 over one period, and tiled across the
    /// full series length.
    pub(super) fn extract_seasonal_component(detrended: &[T], period: usize) -> Vec<T> {
        let n = detrended.len();
        let mut sums = vec![T::zero(); period];
        let mut counts = vec![0usize; period];
        for (i, &value) in detrended.iter().enumerate() {
            let phase = i % period;
            sums[phase] = sums[phase] + value;
            counts[phase] += 1;
        }
        let mut indices = vec![T::zero(); period];
        for phase in 0..period {
            if counts[phase] > 0 {
                indices[phase] = sums[phase] / T::from(counts[phase]).unwrap_or_else(|| T::one());
            }
        }
        let index_mean = indices.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(period).unwrap_or_else(|| T::one());
        for index in indices.iter_mut() {
            *index = *index - index_mean;
        }
        (0..n).map(|i| indices[i % period]).collect()
    }
    pub(super) fn compute_residual_threshold(&self, residuals: &[T]) -> T {
        if residuals.is_empty() {
            return T::zero();
        }
        // A count is representable in every real float type; `scalar_or`
        // keeps this infallible function panic-free rather than aborting the
        // process on an exotic `T`. A divisor of 1 degrades the statistic, it
        // does not corrupt it.
        let count = crate::utils::scalar_or(residuals.len(), T::one());
        let mean = residuals.iter().fold(T::zero(), |acc, &x| acc + x) / count;
        let variance = residuals
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / count;
        variance.sqrt() * self.config.statistical_threshold
    }
    pub(super) fn determine_severity(&self, score: T, threshold: T) -> AnomalySeverity {
        if threshold < T::epsilon() {
            return AnomalySeverity::Low;
        }
        let ratio = score / threshold;
        if ratio > T::from(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if ratio > T::from(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if ratio > T::from(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
}
/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult<T: Float + Debug + Send + Sync + 'static> {
    pub is_anomaly: bool,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub confidence: T,
    pub anomaly_score: T,
    pub timestamp: Instant,
    pub context: AnomalyContext<T>,
    pub suggested_actions: Vec<String>,
}
/// Context information for anomalies
#[derive(Debug, Clone)]
pub struct AnomalyContext<T: Float + Debug + Send + Sync + 'static> {
    pub baseline_mean: T,
    pub baseline_std: T,
    pub current_value: T,
    pub deviation_magnitude: T,
    pub trend_deviation: T,
    pub pattern_match_score: T,
    pub historical_frequency: T,
}
/// Seasonal decomposition result
#[derive(Debug)]
pub struct SeasonalDecomposition<T: Float + Debug + Send + Sync + 'static> {
    pub trend: Vec<T>,
    pub seasonal: Vec<T>,
    pub residuals: Vec<T>,
}
/// Simple classification model
pub(super) struct ClassificationModel<T: Float + Debug + Send + Sync + 'static> {
    pub(super) anomaly_type: AnomalyType,
    pub(super) weights: Vec<T>,
    pub(super) bias: T,
    pub(super) learning_rate: T,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ClassificationModel<T> {
    pub(super) fn new(anomaly_type: AnomalyType) -> Self {
        Self {
            anomaly_type,
            weights: vec![T::from(0.1).unwrap_or_else(|| T::zero()); 7],
            bias: T::zero(),
            learning_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
            _phantom: PhantomData,
        }
    }
    pub(super) fn classify(&self, features: &AnomalyFeatures<T>) -> T {
        let feature_vec = [
            features.statistical_score,
            features.trend_score,
            features.pattern_score,
            features.volatility,
            features.magnitude,
            features
                .frequency_features
                .first()
                .copied()
                .unwrap_or(T::zero()),
            features
                .temporal_features
                .first()
                .copied()
                .unwrap_or(T::zero()),
        ];
        let score = feature_vec
            .iter()
            .zip(self.weights.iter())
            .map(|(&f, &w)| f * w)
            .fold(T::zero(), |acc, x| acc + x)
            + self.bias;
        T::one() / (T::one() + (-score).exp())
    }
    pub(super) fn update(&mut self, features: &AnomalyFeatures<T>, label: bool) {
        let feature_vec = [
            features.statistical_score,
            features.trend_score,
            features.pattern_score,
            features.volatility,
            features.magnitude,
            features
                .frequency_features
                .first()
                .copied()
                .unwrap_or(T::zero()),
            features
                .temporal_features
                .first()
                .copied()
                .unwrap_or(T::zero()),
        ];
        let prediction = self.classify(features);
        let target = if label { T::one() } else { T::zero() };
        let error = prediction - target;
        for (weight, &feature) in self.weights.iter_mut().zip(feature_vec.iter()) {
            *weight = *weight - self.learning_rate * error * feature;
        }
        self.bias = self.bias - self.learning_rate * error;
    }
}
/// Frequency domain analysis for anomaly detection
pub(super) struct FrequencyAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> FrequencyAnalyzer<T> {
    pub(super) fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }
    pub(super) fn analyze(&self, values: &[T]) -> Vec<AnomalyResult<T>> {
        let mut results = Vec::new();
        if values.len() < self.config.min_data_points {
            return results;
        }
        let autocorr = self.compute_autocorrelation(values);
        let anomaly_threshold = T::from(0.1).unwrap_or_else(|| T::zero());
        for (lag, &corr) in autocorr.iter().enumerate() {
            if corr.abs() > anomaly_threshold && lag > 0 {
                results.push(AnomalyResult {
                    is_anomaly: true,
                    anomaly_type: AnomalyType::PatternAnomaly,
                    severity: AnomalySeverity::Medium,
                    confidence: corr.abs(),
                    anomaly_score: corr.abs(),
                    timestamp: Instant::now(),
                    context: AnomalyContext {
                        baseline_mean: T::zero(),
                        baseline_std: T::zero(),
                        current_value: T::zero(),
                        deviation_magnitude: corr.abs(),
                        trend_deviation: T::zero(),
                        pattern_match_score: corr.abs(),
                        historical_frequency: T::from(lag).unwrap_or_else(|| T::zero()),
                    },
                    suggested_actions: vec![format!(
                        "Investigate periodic pattern with lag {}",
                        lag
                    )],
                });
            }
        }
        results
    }
    pub(super) fn compute_autocorrelation(&self, values: &[T]) -> Vec<T> {
        let n = values.len();
        let mut autocorr = vec![T::zero(); n / 2];
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(n).unwrap_or_else(|| T::zero());
        for lag in 0..autocorr.len() {
            let mut sum = T::zero();
            let mut count = 0;
            for i in lag..n {
                sum = sum + (values[i] - mean) * (values[i - lag] - mean);
                count += 1;
            }
            if count > 0 {
                autocorr[lag] = sum / T::from(count).unwrap_or_else(|| T::zero());
            }
        }
        if autocorr[0] > T::epsilon() {
            let first_value = autocorr[0];
            for corr in &mut autocorr {
                *corr = *corr / first_value;
            }
        }
        autocorr
    }
}
/// Anomaly classification types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    StatisticalOutlier,
    TrendAnomaly,
    PerformanceAnomaly,
    ConvergenceAnomaly,
    ResourceAnomaly,
    PatternAnomaly,
    SeasonalAnomaly,
    SystemAnomaly,
}
/// Pattern memory for anomaly detection
#[derive(Debug)]
pub(super) struct PatternMemory<T: Float + Debug + Send + Sync + 'static> {
    pub(super) patterns: VecDeque<T>,
    pub(super) window_size: usize,
    pub(super) pattern_weights: Vec<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> PatternMemory<T> {
    pub(super) fn new(window_size: usize) -> Self {
        Self {
            patterns: VecDeque::with_capacity(window_size),
            window_size,
            pattern_weights: vec![T::one(); window_size],
        }
    }
    pub(super) fn update(&mut self, value: T, is_anomaly: bool) {
        self.patterns.push_back(value);
        if self.patterns.len() > self.window_size {
            self.patterns.pop_front();
        }
        if is_anomaly && !self.pattern_weights.is_empty() {
            let last_idx = self.pattern_weights.len() - 1;
            self.pattern_weights[last_idx] =
                self.pattern_weights[last_idx] * T::from(0.9).unwrap_or_else(|| T::zero());
        }
    }
    pub(super) fn compute_pattern_score(&self, value: T) -> T {
        if self.patterns.is_empty() {
            return T::one();
        }
        let similarities: Vec<T> = self
            .patterns
            .iter()
            .map(|&pattern| {
                let diff = (value - pattern).abs();
                (-diff).exp()
            })
            .collect();
        let weighted_similarity = similarities
            .iter()
            .zip(self.pattern_weights.iter())
            .map(|(&sim, &weight)| sim * weight)
            .fold(T::zero(), |acc, x| acc + x);
        let total_weight = self
            .pattern_weights
            .iter()
            .fold(T::zero(), |acc, &w| acc + w);
        if total_weight > T::epsilon() {
            weighted_similarity / total_weight
        } else {
            T::zero()
        }
    }
}
/// Primary anomaly detector
pub struct AnomalyDetector<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: AnomalyConfig<T>,
    pub(super) analyzer: AnomalyAnalyzer<T>,
    pub(super) classifier: AnomalyClassifier<T>,
    pub(super) outlier_detector: OutlierDetector<T>,
    pub(super) reporter: AnomalyReporter<T>,
    pub(super) data_history: VecDeque<(Instant, T)>,
    pub(super) baseline_stats: BaselineStats<T>,
    pub(super) pattern_memory: PatternMemory<T>,
    pub(super) adaptive_thresholds: AdaptiveThresholds<T>,
    pub(super) _phantom: PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> AnomalyDetector<T> {
    pub fn new(config: AnomalyConfig<T>) -> Self {
        Self {
            analyzer: AnomalyAnalyzer::new(config.clone()),
            classifier: AnomalyClassifier::new(config.clone()),
            outlier_detector: OutlierDetector::new(config.clone()),
            reporter: AnomalyReporter::new(config.clone()),
            data_history: VecDeque::with_capacity(config.baseline_window * 2),
            baseline_stats: BaselineStats::new(),
            pattern_memory: PatternMemory::new(config.pattern_window),
            adaptive_thresholds: AdaptiveThresholds::new(),
            config,
            _phantom: PhantomData,
        }
    }
    pub fn detect_anomaly(&mut self, value: T) -> AnomalyResult<T> {
        let timestamp = Instant::now();
        self.data_history.push_back((timestamp, value));
        if self.data_history.len() > self.config.baseline_window * 2 {
            self.data_history.pop_front();
        }
        self.update_baseline_stats();
        if self.data_history.len() < self.config.min_data_points {
            return AnomalyResult {
                is_anomaly: false,
                anomaly_type: AnomalyType::StatisticalOutlier,
                severity: AnomalySeverity::Low,
                confidence: T::zero(),
                anomaly_score: T::zero(),
                timestamp,
                context: self.create_context(value),
                suggested_actions: vec![],
            };
        }
        let statistical_result = self.statistical_anomaly_detection(value);
        let trend_result = self.trend_anomaly_detection(value);
        let pattern_result = self.pattern_anomaly_detection(value);
        let outlier_result = self
            .outlier_detector
            .detect_outlier(value, &self.data_history);
        let combined_result = self.combine_detection_results(
            statistical_result,
            trend_result,
            pattern_result,
            outlier_result,
            timestamp,
            value,
        );
        self.pattern_memory
            .update(value, combined_result.is_anomaly);
        if self.config.enable_adaptive_thresholds {
            self.adaptive_thresholds
                .update(value, combined_result.is_anomaly);
        }
        combined_result
    }
    pub(super) fn update_baseline_stats(&mut self) {
        if self.data_history.is_empty() {
            return;
        }
        let values: Vec<T> = self.data_history.iter().map(|(_, v)| *v).collect();
        let baseline_window = self.config.baseline_window.min(values.len());
        let baseline_values = if values.len() > baseline_window {
            &values[values.len() - baseline_window..]
        } else {
            &values
        };
        self.baseline_stats.update(baseline_values);
    }
    pub(super) fn statistical_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        let z_score = if self.baseline_stats.std_dev > T::epsilon() {
            (value - self.baseline_stats.mean).abs() / self.baseline_stats.std_dev
        } else {
            T::zero()
        };
        let threshold = if self.config.enable_adaptive_thresholds {
            self.adaptive_thresholds.get_statistical_threshold()
        } else {
            self.config.statistical_threshold
        };
        let is_anomaly = z_score > threshold;
        let confidence = if is_anomaly {
            (z_score / threshold).min(T::one())
        } else {
            T::zero()
        };
        let severity = self.determine_severity(z_score, threshold);
        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::StatisticalOutlier,
            severity,
            confidence,
            anomaly_score: z_score,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_statistical_actions(z_score, threshold),
        }
    }
    pub(super) fn trend_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        if self.data_history.len() < 3 {
            return self.create_no_anomaly_result(value, AnomalyType::TrendAnomaly);
        }
        let recent_values: Vec<T> = self
            .data_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, v)| *v)
            .collect();
        let trend = self.compute_trend(&recent_values);
        let expected_value = self.predict_next_value(&recent_values, trend);
        let trend_deviation = (value - expected_value).abs();
        let threshold = self.baseline_stats.std_dev * self.config.trend_sensitivity;
        let is_anomaly = trend_deviation > threshold && threshold > T::epsilon();
        let confidence = if is_anomaly && threshold > T::epsilon() {
            (trend_deviation / threshold).min(T::one())
        } else {
            T::zero()
        };
        let severity = self.determine_severity(trend_deviation, threshold);
        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::TrendAnomaly,
            severity,
            confidence,
            anomaly_score: trend_deviation,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_trend_actions(trend_deviation, threshold),
        }
    }
    pub(super) fn pattern_anomaly_detection(&self, value: T) -> AnomalyResult<T> {
        let pattern_score = self.pattern_memory.compute_pattern_score(value);
        let threshold = T::from(0.3).unwrap_or_else(|| T::zero());
        let is_anomaly = pattern_score < threshold;
        let confidence = if is_anomaly {
            T::one() - pattern_score
        } else {
            T::zero()
        };
        let severity = if pattern_score < T::from(0.1).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if pattern_score < T::from(0.2).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };
        AnomalyResult {
            is_anomaly,
            anomaly_type: AnomalyType::PatternAnomaly,
            severity,
            confidence,
            anomaly_score: T::one() - pattern_score,
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: self.get_pattern_actions(pattern_score),
        }
    }
    pub(super) fn combine_detection_results(
        &self,
        statistical: AnomalyResult<T>,
        trend: AnomalyResult<T>,
        pattern: AnomalyResult<T>,
        outlier: AnomalyResult<T>,
        timestamp: Instant,
        value: T,
    ) -> AnomalyResult<T> {
        let weights = [
            T::from(0.3).unwrap_or_else(|| T::zero()),
            T::from(0.25).unwrap_or_else(|| T::zero()),
            T::from(0.25).unwrap_or_else(|| T::zero()),
            T::from(0.2).unwrap_or_else(|| T::zero()),
        ];
        let results = [&statistical, &trend, &pattern, &outlier];
        let combined_confidence = results
            .iter()
            .zip(weights.iter())
            .map(|(result, &weight)| result.confidence * weight)
            .fold(T::zero(), |acc, x| acc + x);
        let combined_score = results
            .iter()
            .zip(weights.iter())
            .map(|(result, &weight)| result.anomaly_score * weight)
            .fold(T::zero(), |acc, x| acc + x);
        let is_anomaly = combined_confidence > self.config.confidence_threshold;
        let anomaly_type = results
            .iter()
            // A NaN confidence must order deterministically rather than
            // panicking the comparator: treat it as the smallest value, so a
            // result whose confidence could not be computed never wins.
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.anomaly_type.clone())
            .unwrap_or(AnomalyType::StatisticalOutlier);
        let severity = self.determine_severity(combined_score, self.config.statistical_threshold);
        let mut suggested_actions = Vec::new();
        for result in &results {
            if result.is_anomaly {
                suggested_actions.extend(result.suggested_actions.clone());
            }
        }
        AnomalyResult {
            is_anomaly,
            anomaly_type,
            severity,
            confidence: combined_confidence,
            anomaly_score: combined_score,
            timestamp,
            context: self.create_context(value),
            suggested_actions,
        }
    }
    pub(super) fn create_no_anomaly_result(
        &self,
        value: T,
        anomaly_type: AnomalyType,
    ) -> AnomalyResult<T> {
        AnomalyResult {
            is_anomaly: false,
            anomaly_type,
            severity: AnomalySeverity::Low,
            confidence: T::zero(),
            anomaly_score: T::zero(),
            timestamp: Instant::now(),
            context: self.create_context(value),
            suggested_actions: vec![],
        }
    }
    pub(super) fn create_context(&self, value: T) -> AnomalyContext<T> {
        AnomalyContext {
            baseline_mean: self.baseline_stats.mean,
            baseline_std: self.baseline_stats.std_dev,
            current_value: value,
            deviation_magnitude: (value - self.baseline_stats.mean).abs(),
            trend_deviation: T::zero(),
            pattern_match_score: self.pattern_memory.compute_pattern_score(value),
            historical_frequency: T::zero(),
        }
    }
    pub(super) fn compute_trend(&self, values: &[T]) -> T {
        if values.len() < 2 {
            return T::zero();
        }
        // Indices and counts are representable in every real float type; the
        // fallbacks keep this infallible function panic-free.
        let n = crate::utils::scalar_or(values.len(), T::one());
        let sum_x = (0..values.len()).fold(T::zero(), |acc, i| {
            acc + crate::utils::scalar_or(i, T::zero())
        });
        let sum_y = values.iter().fold(T::zero(), |acc, &y| acc + y);
        let sum_xy = values.iter().enumerate().fold(T::zero(), |acc, (i, &y)| {
            acc + crate::utils::scalar_or(i, T::zero()) * y
        });
        let sum_x2 = (0..values.len()).fold(T::zero(), |acc, i| {
            let i_t = crate::utils::scalar_or(i, T::zero());
            acc + i_t * i_t
        });
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < T::epsilon() {
            return T::zero();
        }
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    pub(super) fn predict_next_value(&self, values: &[T], trend: T) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let last_value = values[values.len() - 1];
        last_value + trend
    }
    pub(super) fn determine_severity(&self, score: T, threshold: T) -> AnomalySeverity {
        if threshold < T::epsilon() {
            return AnomalySeverity::Low;
        }
        let ratio = score / threshold;
        if ratio > T::from(3.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Critical
        } else if ratio > T::from(2.0).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::High
        } else if ratio > T::from(1.5).unwrap_or_else(|| T::zero()) {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
    pub(super) fn get_statistical_actions(&self, z_score: T, threshold: T) -> Vec<String> {
        let mut actions = Vec::new();
        if z_score > threshold * T::from(2.0).unwrap_or_else(|| T::zero()) {
            actions.push("Investigate data source for potential errors".to_string());
            actions.push("Check for system or measurement anomalies".to_string());
        } else if z_score > threshold {
            actions.push("Monitor closely for recurring pattern".to_string());
            actions.push("Consider adjusting optimization parameters".to_string());
        }
        actions
    }
    pub(super) fn get_trend_actions(&self, deviation: T, threshold: T) -> Vec<String> {
        let mut actions = Vec::new();
        if deviation > threshold {
            actions.push("Analyze trend disruption causes".to_string());
            actions.push("Consider adaptive optimization strategies".to_string());
            actions.push("Review recent parameter changes".to_string());
        }
        actions
    }
    pub(super) fn get_pattern_actions(&self, pattern_score: T) -> Vec<String> {
        let mut actions = Vec::new();
        if pattern_score < T::from(0.2).unwrap_or_else(|| T::zero()) {
            actions.push("Investigate unusual behavioral patterns".to_string());
            actions.push("Check for optimization convergence issues".to_string());
        }
        actions
    }
    pub fn get_config(&self) -> &AnomalyConfig<T> {
        &self.config
    }
    pub fn update_config(&mut self, config: AnomalyConfig<T>) {
        self.config = config;
    }
    pub fn reset(&mut self) {
        self.data_history.clear();
        self.baseline_stats = BaselineStats::new();
        self.pattern_memory = PatternMemory::new(self.config.pattern_window);
        self.adaptive_thresholds = AdaptiveThresholds::new();
    }
}
/// Anomaly alert with detailed information
#[derive(Debug, Clone)]
pub struct AnomalyAlert<T: Float + Debug + Send + Sync + 'static> {
    pub id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub timestamp: Instant,
    pub message: String,
    pub confidence: T,
    pub data_point: T,
    pub context: AnomalyContext<T>,
    pub suggested_actions: Vec<String>,
    pub acknowledged: bool,
    pub resolved: bool,
}
