//! Behavior for [`TopologyPerformanceMonitor`]: metrics collection,
//! health checks, rolling-history z-score anomaly detection, and report
//! generation.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{OptimError, Result};
use scirs2_core::error::ErrorContext;

use super::types::*;

impl TopologyPerformanceMonitor {
    /// Create a new topology performance monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Start monitoring with specified configuration
    pub fn start_monitoring(&mut self, config: TopologyMonitoringSettings) -> Result<()> {
        self.performance_monitoring = config.performance_monitoring;
        self.health_monitoring = config.health_monitoring;
        self.traffic_monitoring = config.traffic_monitoring;
        self.alert_system.config = config.alert_settings;
        Ok(())
    }

    /// Collect current metrics
    pub fn collect_metrics(&mut self) -> Result<TopologyMetrics> {
        let mut metrics = HashMap::new();

        // Add a real wall-clock timestamp (fractional seconds since the Unix
        // epoch). `Instant` is only meaningful as a delta and would yield ~0
        // here, so `SystemTime` is used and the clock error is propagated
        // honestly rather than swallowed.
        metrics.insert(
            "collection_timestamp".to_string(),
            Self::unix_timestamp_secs()?,
        );

        // Update metrics collector
        self.metrics_collector.metrics = metrics.clone();

        Ok(metrics)
    }

    /// Process health checks
    pub fn process_health_checks(&mut self) -> Result<Vec<HealthCheckResult>> {
        let mut results = Vec::new();

        for indicator in &self.health_monitoring.health_indicators {
            let result = self.perform_health_check(indicator)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Perform an individual health check driven by the most recently collected
    /// metrics.
    ///
    /// Health is derived from real data rather than assumed. When no metric
    /// backing the requested indicator is present, [`HealthStatus::Unknown`] is
    /// returned (never `Healthy` without evidence). When data is present it is
    /// classified against the configured [`PerformanceThresholds`].
    ///
    /// Metric-key convention (snake_case, aligned with [`MetricType`]):
    /// - `LinkConnectivity`   -> `"link_connectivity"` (fraction of links up in
    ///   `[0, 1]`, higher is better)
    /// - `DeviceResponsiveness` -> `"latency"` (ms, higher is worse)
    /// - `PerformanceDegradation` -> `"bandwidth_utilization"` (percent, higher
    ///   is worse)
    /// - `ErrorRateIncrease`  -> `"packet_loss"` (rate, higher is worse)
    /// - `Custom { indicator_name }` -> a metric named `indicator_name`, treated
    ///   as a `[0, 1]` higher-is-better health ratio.
    ///
    /// The `HealthStatus` enum has no dedicated `Degraded`/`Unhealthy` variants,
    /// so a degraded reading maps to [`HealthStatus::Warning`] and an unhealthy
    /// reading to [`HealthStatus::Critical`].
    ///
    /// `pub(super)`: called directly by the `monitoring` test module (a
    /// sibling submodule), in addition to internal use in
    /// [`Self::process_health_checks`] and [`Self::compute_overall_score`].
    pub(super) fn perform_health_check(
        &self,
        indicator: &HealthIndicator,
    ) -> Result<HealthCheckResult> {
        let metrics = &self.metrics_collector.metrics;
        let thresholds = &self.performance_monitoring.performance_thresholds;

        let (status, details) = match indicator {
            HealthIndicator::LinkConnectivity => match metrics.get("link_connectivity") {
                Some(&ratio) => (
                    Self::classify_higher_is_better(ratio),
                    format!("link connectivity ratio = {ratio:.4}"),
                ),
                None => (
                    HealthStatus::Unknown,
                    "no link_connectivity data collected".to_string(),
                ),
            },
            HealthIndicator::DeviceResponsiveness => match metrics.get("latency") {
                Some(&latency) => (
                    Self::classify_higher_is_worse(latency, &thresholds.latency_thresholds),
                    format!("device latency = {latency:.4} ms"),
                ),
                None => (
                    HealthStatus::Unknown,
                    "no latency data collected".to_string(),
                ),
            },
            HealthIndicator::PerformanceDegradation => match metrics.get("bandwidth_utilization") {
                Some(&util) => (
                    Self::classify_higher_is_worse(util, &thresholds.utilization_thresholds),
                    format!("bandwidth utilization = {util:.4}%"),
                ),
                None => (
                    HealthStatus::Unknown,
                    "no bandwidth_utilization data collected".to_string(),
                ),
            },
            HealthIndicator::ErrorRateIncrease => match metrics.get("packet_loss") {
                Some(&loss) => (
                    Self::classify_higher_is_worse(loss, &thresholds.error_thresholds),
                    format!("packet loss rate = {loss:.4}"),
                ),
                None => (
                    HealthStatus::Unknown,
                    "no packet_loss data collected".to_string(),
                ),
            },
            HealthIndicator::Custom { indicator_name } => match metrics.get(indicator_name) {
                Some(&value) => (
                    Self::classify_higher_is_better(value),
                    format!("{indicator_name} health ratio = {value:.4}"),
                ),
                None => (
                    HealthStatus::Unknown,
                    format!("no {indicator_name} data collected"),
                ),
            },
        };

        Ok(HealthCheckResult {
            indicator: indicator.clone(),
            status,
            timestamp: Instant::now(),
            details,
        })
    }

    /// Detect anomalies in the supplied metrics using a rolling z-score test.
    ///
    /// For each metric the current value is scored against the previously
    /// observed distribution for that metric, then folded into the rolling
    /// history. Scoring happens *before* recording so a sample is compared to
    /// its prior baseline rather than to itself.
    pub fn detect_anomalies(&mut self, metrics: &TopologyMetrics) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();

        for (metric_name, value) in metrics {
            // Score against the existing baseline before folding in the sample.
            let anomalous = self.is_anomalous_value(metric_name, *value);
            let z_score = self.metric_z_score(metric_name, *value);
            self.record_metric_value(metric_name, *value);

            if anomalous {
                let anomaly = DetectedAnomaly {
                    anomaly_id: format!(
                        "anomaly_{metric_name}_{}",
                        self.anomaly_detector.statistics.total_detected + anomalies.len()
                    ),
                    timestamp: Instant::now(),
                    anomaly_type: AnomalyType::Performance,
                    // `is_anomalous_value` is true only when a z-score exists, so
                    // the fallback is unreachable; 1.0 is the max-severity default.
                    score: z_score.map_or(1.0, Self::anomaly_score_from_z),
                    metrics: [(metric_name.clone(), *value)].into_iter().collect(),
                    status: AnomalyStatus::New,
                };
                anomalies.push(anomaly);
            }
        }

        self.anomaly_detector.statistics.total_detected += anomalies.len();
        self.anomaly_detector.anomalies.extend(anomalies.clone());
        Ok(anomalies)
    }

    /// Check whether a metric value is anomalous relative to its rolling history.
    ///
    /// Returns `false` when there is insufficient history to form a reliable
    /// estimate (honest: not enough data), and otherwise flags the value when
    /// its z-score magnitude exceeds [`Self::ANOMALY_Z_THRESHOLD`].
    fn is_anomalous_value(&self, metric_name: &str, value: f64) -> bool {
        match self.metric_z_score(metric_name, value) {
            Some(z) => z.abs() > Self::ANOMALY_Z_THRESHOLD,
            None => false,
        }
    }

    /// Generate a performance report with a unique id and a data-derived score.
    ///
    /// The `report_id` combines a monotonically increasing sequence number
    /// (guaranteeing uniqueness even within the same second) with a wall-clock
    /// timestamp. The `overall_score` is computed from the live health checks
    /// and the anomaly baseline rather than hard-coded.
    pub fn generate_report(&self, period: Duration) -> Result<PerformanceReport> {
        // fetch_add returns the previous value, so ids start at 0 and never
        // repeat for the lifetime of this monitor.
        let sequence = self.report_counter.fetch_add(1, Ordering::Relaxed);
        let minted_at = Self::unix_timestamp_secs()?;

        let (overall_score, critical_issues) = self.compute_overall_score();

        Ok(PerformanceReport {
            report_id: format!("report_{sequence}_{}", minted_at as u64),
            timestamp: Instant::now(),
            period,
            summary: PerformanceSummary {
                overall_score,
                kpis: HashMap::new(),
                trends: Vec::new(),
                critical_issues,
            },
            detailed_metrics: self.metrics_collector.metrics.clone(),
            recommendations: Vec::new(),
        })
    }

    /// Compute an overall performance score in `[0, 1]` from real evidence.
    ///
    /// Two evidence sources are averaged when available:
    /// - the mean of the current health-check scores
    ///   (`Healthy` = 1.0, `Warning` = 0.5, `Critical`/`Error` = 0.0; `Unknown`
    ///   indicators are excluded because they carry no evidence), and
    /// - an anomaly component `1 - penalty`, where the penalty is the fraction
    ///   of tracked metrics currently flagged by unresolved anomalies.
    ///
    /// When neither source has data the score is `0.0` and a self-describing
    /// critical issue is recorded so the value reads as "no evidence" rather
    /// than a catastrophic reading.
    fn compute_overall_score(&self) -> (f64, Vec<String>) {
        let mut components: Vec<f64> = Vec::new();
        let mut critical_issues: Vec<String> = Vec::new();

        // Health component: mean over indicators that actually have data.
        let mut health_scores: Vec<f64> = Vec::new();
        for indicator in &self.health_monitoring.health_indicators {
            if let Ok(result) = self.perform_health_check(indicator) {
                let score = match result.status {
                    HealthStatus::Healthy => Some(1.0),
                    HealthStatus::Warning => Some(0.5),
                    HealthStatus::Error | HealthStatus::Critical => Some(0.0),
                    HealthStatus::Unknown => None,
                };
                if let Some(score) = score {
                    health_scores.push(score);
                    if matches!(result.status, HealthStatus::Error | HealthStatus::Critical) {
                        critical_issues.push(result.details.clone());
                    }
                }
            }
        }
        if !health_scores.is_empty() {
            components.push(health_scores.iter().sum::<f64>() / health_scores.len() as f64);
        }

        // Anomaly component: only meaningful once a baseline has been observed.
        if !self.anomaly_detector.value_history.is_empty()
            || !self.anomaly_detector.anomalies.is_empty()
        {
            let active_anomalies = self
                .anomaly_detector
                .anomalies
                .iter()
                .filter(|a| {
                    matches!(
                        a.status,
                        AnomalyStatus::New
                            | AnomalyStatus::Investigating
                            | AnomalyStatus::Confirmed
                    )
                })
                .count();
            let tracked = self.anomaly_detector.value_history.len().max(1);
            let penalty = (active_anomalies as f64 / tracked as f64).min(1.0);
            components.push(1.0 - penalty);
        }

        if components.is_empty() {
            critical_issues
                .push("insufficient monitoring data to compute a performance score".to_string());
            (0.0, critical_issues)
        } else {
            let score = components.iter().sum::<f64>() / components.len() as f64;
            (score, critical_issues)
        }
    }

    /// Number of standard deviations beyond which a sample is treated as
    /// anomalous.
    ///
    /// The configured [`AlertThresholds`] express anomaly *scores* in `[0, 1]`,
    /// not standard-deviation multiples, so there is no sigma value to source
    /// from the existing configuration. A 3-sigma rule (~99.7% of a normal
    /// distribution lies within this band) is used as a documented, widely
    /// established default.
    const ANOMALY_Z_THRESHOLD: f64 = 3.0;

    /// Minimum rolling-history length required before z-score anomaly detection
    /// is meaningful. With fewer samples the mean/standard-deviation estimate is
    /// too noisy, so values are reported as non-anomalous (honest: not enough
    /// data).
    const ANOMALY_MIN_SAMPLES: usize = 5;

    /// Upper bound on the per-metric rolling history retained for anomaly
    /// detection, keeping memory usage bounded for long-running monitors.
    pub const ANOMALY_HISTORY_CAPACITY: usize = 256;

    /// Current wall-clock time as fractional seconds since the Unix epoch.
    ///
    /// `SystemTime` is used rather than `Instant` because `Instant` is only
    /// meaningful as a delta and yields ~0 against a freshly captured `now()`.
    /// A clock set before the epoch is surfaced as an error rather than masked.
    fn unix_timestamp_secs() -> Result<f64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .map_err(|error| {
                OptimError::ComputationError(ErrorContext::new(format!(
                    "system clock is set before the Unix epoch: {error}"
                )))
            })
    }

    /// Compute the z-score of `value` against the rolling history for
    /// `metric_name`, or `None` when there is not yet enough history.
    fn metric_z_score(&self, metric_name: &str, value: f64) -> Option<f64> {
        let history = self.anomaly_detector.value_history.get(metric_name)?;
        if history.len() < Self::ANOMALY_MIN_SAMPLES {
            return None;
        }

        let n = history.len() as f64;
        let mean = history.iter().sum::<f64>() / n;
        let variance = history
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        if std_dev <= f64::EPSILON {
            // Degenerate (constant) history: any deviation from the constant is
            // infinitely surprising; an exact match is perfectly normal.
            return Some(if (value - mean).abs() > f64::EPSILON {
                f64::INFINITY
            } else {
                0.0
            });
        }

        Some((value - mean) / std_dev)
    }

    /// Append a sample to the bounded rolling history for `metric_name`.
    fn record_metric_value(&mut self, metric_name: &str, value: f64) {
        let history = self
            .anomaly_detector
            .value_history
            .entry(metric_name.to_string())
            .or_default();
        history.push_back(value);
        while history.len() > Self::ANOMALY_HISTORY_CAPACITY {
            history.pop_front();
        }
    }

    /// Map a z-score to a bounded anomaly score in `[0, 1)`.
    ///
    /// The map `|z| / (|z| + threshold)` is monotonic: `0` at `z = 0`, `0.5` at
    /// the detection threshold, and approaching `1` for extreme outliers.
    fn anomaly_score_from_z(z: f64) -> f64 {
        let magnitude = z.abs();
        if !magnitude.is_finite() {
            return 1.0;
        }
        magnitude / (magnitude + Self::ANOMALY_Z_THRESHOLD)
    }

    /// Classify a "higher-is-worse" metric value against threshold levels.
    fn classify_higher_is_worse(value: f64, thresholds: &ThresholdLevels) -> HealthStatus {
        if value >= thresholds.critical {
            HealthStatus::Critical
        } else if value >= thresholds.warning {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }

    /// Classify a `[0, 1]` "higher-is-better" health ratio using documented
    /// bands: `>= 0.99` healthy, `>= 0.90` degraded (warning), otherwise
    /// unhealthy (critical).
    fn classify_higher_is_better(ratio: f64) -> HealthStatus {
        if ratio >= 0.99 {
            HealthStatus::Healthy
        } else if ratio >= 0.90 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        }
    }
}
