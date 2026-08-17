//! Real-time monitoring dashboard.
//!
//! # The defect this replaces
//!
//! `MonitoringDashboard::update_metrics` was
//!
//! ```text
//! if let Some(metric) = self.metrics.get_mut("privacy_budget_epsilon") { ... }
//! ```
//!
//! and nothing ever inserted `"privacy_budget_epsilon"`, so the branch never
//! ran: the dashboard was permanently empty, no history accumulated, and the
//! `alerts` queue and `config.alert_thresholds` were never consulted.
//!
//! What is implemented: metrics are created on first sight, gauges and counters
//! are updated per event, history is trimmed by the configured retention
//! window, and configured thresholds actually raise alerts.

use crate::error::{OptimError, Result};
use std::collections::{HashMap, VecDeque};

use super::hashing::event_type_key;
use super::types::{
    AlertSeverity, AlertStatus, AlertThreshold, AuditEvent, DashboardAlert, DashboardConfig,
    DashboardMetric, MetricType, ThresholdDirection,
};

/// Metric name for the epsilon reported by the most recent event.
pub const METRIC_EPSILON: &str = "privacy_budget_epsilon";
/// Metric name for the delta reported by the most recent event.
pub const METRIC_DELTA: &str = "privacy_budget_delta";
/// Metric name for the cumulative epsilon reported across all events.
pub const METRIC_CUMULATIVE_EPSILON: &str = "privacy_budget_epsilon_cumulative";
/// Metric name for the total number of events observed.
pub const METRIC_EVENTS: &str = "audit_events_total";

/// Maximum number of samples retained per metric.
const MAX_SAMPLES: usize = 1_000;
/// Maximum number of alerts retained.
const MAX_ALERTS: usize = 1_000;

/// Real-time monitoring dashboard.
pub struct MonitoringDashboard {
    /// Metrics by name.
    metrics: HashMap<String, DashboardMetric>,
    /// Alerts, oldest first.
    alerts: VecDeque<DashboardAlert>,
    /// Dashboard configuration.
    config: DashboardConfig,
}

impl MonitoringDashboard {
    /// Create a dashboard with the default configuration.
    pub fn new() -> Self {
        Self::with_config(DashboardConfig::default())
    }

    /// Create a dashboard with an explicit configuration.
    pub fn with_config(config: DashboardConfig) -> Self {
        Self {
            metrics: HashMap::new(),
            alerts: VecDeque::new(),
            config,
        }
    }

    /// The dashboard configuration.
    pub fn config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Install or replace an alert threshold.
    pub fn set_threshold(&mut self, threshold: AlertThreshold) -> Result<()> {
        if !threshold.warning.is_finite() || !threshold.critical.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "the thresholds for metric `{}` must be finite, got warning={} critical={}",
                threshold.metric, threshold.warning, threshold.critical
            )));
        }
        match threshold.direction {
            ThresholdDirection::Above if threshold.critical < threshold.warning => {
                return Err(OptimError::InvalidParameter(format!(
                    "for an upper bound on `{}` the critical threshold must not be below the \
                     warning threshold",
                    threshold.metric
                )))
            }
            ThresholdDirection::Below if threshold.critical > threshold.warning => {
                return Err(OptimError::InvalidParameter(format!(
                    "for a lower bound on `{}` the critical threshold must not be above the \
                     warning threshold",
                    threshold.metric
                )))
            }
            _ => {}
        }
        self.config
            .alert_thresholds
            .insert(threshold.metric.clone(), threshold);
        Ok(())
    }

    /// A metric by name.
    pub fn metric(&self, name: &str) -> Option<&DashboardMetric> {
        self.metrics.get(name)
    }

    /// Names of the tracked metrics, sorted.
    pub fn metric_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.metrics.keys().cloned().collect();
        names.sort();
        names
    }

    /// Alerts, oldest first.
    pub fn alerts(&self) -> impl Iterator<Item = &DashboardAlert> {
        self.alerts.iter()
    }

    /// Number of retained alerts.
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }

    /// Acknowledge an alert by identifier.
    pub fn acknowledge_alert(&mut self, id: &str) -> Result<()> {
        for alert in self.alerts.iter_mut() {
            if alert.id == id {
                alert.status = AlertStatus::Acknowledged;
                return Ok(());
            }
        }
        Err(OptimError::InvalidParameter(format!(
            "no dashboard alert with identifier `{id}`"
        )))
    }

    /// Update the dashboard from an audit event.
    pub fn update_metrics(&mut self, event: &AuditEvent) -> Result<()> {
        let epsilon = event.privacy_context.epsilon_budget;
        let delta = event.privacy_context.delta_budget;
        if !epsilon.is_finite() || !delta.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "event `{}` reports a non-finite privacy parameter (epsilon={epsilon}, \
                 delta={delta}), which cannot be charted",
                event.id
            )));
        }

        let previous_cumulative = self
            .metrics
            .get(METRIC_CUMULATIVE_EPSILON)
            .map(|metric| metric.current_value)
            .unwrap_or(0.0);
        let previous_count = self
            .metrics
            .get(METRIC_EVENTS)
            .map(|metric| metric.current_value)
            .unwrap_or(0.0);

        let type_metric = format!("audit_events_{}", event_type_key(&event.event_type));
        let previous_type_count = self
            .metrics
            .get(&type_metric)
            .map(|metric| metric.current_value)
            .unwrap_or(0.0);

        let updates: [(String, f64, &str, MetricType); 5] = [
            (
                METRIC_EPSILON.to_string(),
                epsilon,
                "epsilon",
                MetricType::Gauge,
            ),
            (METRIC_DELTA.to_string(), delta, "delta", MetricType::Gauge),
            (
                METRIC_CUMULATIVE_EPSILON.to_string(),
                previous_cumulative + epsilon,
                "epsilon",
                MetricType::Counter,
            ),
            (
                METRIC_EVENTS.to_string(),
                previous_count + 1.0,
                "events",
                MetricType::Counter,
            ),
            (
                type_metric,
                previous_type_count + 1.0,
                "events",
                MetricType::Counter,
            ),
        ];

        for (name, value, unit, metric_type) in updates {
            self.record_sample(&name, value, unit, metric_type, event.timestamp);
            self.evaluate_thresholds(&name, value, event.timestamp);
        }
        Ok(())
    }

    /// Record one sample of a metric, creating it if necessary.
    fn record_sample(
        &mut self,
        name: &str,
        value: f64,
        unit: &str,
        metric_type: MetricType,
        timestamp: u64,
    ) {
        let retention_seconds = u64::from(self.config.history_retention_hours) * 3_600;
        let metric = self
            .metrics
            .entry(name.to_string())
            .or_insert_with(|| DashboardMetric {
                name: name.to_string(),
                current_value: value,
                historical_values: VecDeque::new(),
                unit: unit.to_string(),
                metric_type,
            });
        metric.current_value = value;
        metric.historical_values.push_back((timestamp, value));

        // Retention window first, then the hard sample cap.
        if retention_seconds > 0 {
            let cutoff = timestamp.saturating_sub(retention_seconds);
            while metric
                .historical_values
                .front()
                .is_some_and(|(sample_time, _)| *sample_time < cutoff)
            {
                let _ = metric.historical_values.pop_front();
            }
        }
        while metric.historical_values.len() > MAX_SAMPLES {
            let _ = metric.historical_values.pop_front();
        }
    }

    /// Raise alerts for a metric that has crossed its configured thresholds.
    fn evaluate_thresholds(&mut self, name: &str, value: f64, timestamp: u64) {
        let Some(threshold) = self.config.alert_thresholds.get(name) else {
            return;
        };
        let breach = match threshold.direction {
            ThresholdDirection::Above => {
                if value >= threshold.critical {
                    Some(AlertSeverity::Critical)
                } else if value >= threshold.warning {
                    Some(AlertSeverity::Warning)
                } else {
                    None
                }
            }
            ThresholdDirection::Below => {
                if value <= threshold.critical {
                    Some(AlertSeverity::Critical)
                } else if value <= threshold.warning {
                    Some(AlertSeverity::Warning)
                } else {
                    None
                }
            }
        };
        let Some(severity) = breach else {
            return;
        };
        let comparison = match threshold.direction {
            ThresholdDirection::Above => "at or above",
            ThresholdDirection::Below => "at or below",
        };
        let bound = match severity {
            AlertSeverity::Critical => threshold.critical,
            _ => threshold.warning,
        };
        let message =
            format!("metric `{name}` is {value}, {comparison} its {severity:?} bound {bound}");
        self.alerts.push_back(DashboardAlert {
            id: format!("dashboard_alert_{}", self.alerts.len()),
            message,
            severity,
            timestamp,
            metric: Some(name.to_string()),
            status: AlertStatus::Active,
        });
        while self.alerts.len() > MAX_ALERTS {
            let _ = self.alerts.pop_front();
        }
    }
}

impl Default for MonitoringDashboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::enhanced_audit::types::{AuditEventData, AuditEventType, PrivacyContext};

    fn event(id: &str, epsilon: f64, timestamp: u64) -> AuditEvent {
        AuditEvent {
            id: id.to_string(),
            timestamp,
            event_type: AuditEventType::PrivacyBudgetConsumption,
            actor: "trainer".to_string(),
            data: AuditEventData {
                description: "spend".to_string(),
                affected_data_subjects: Vec::new(),
                data_categories: Vec::new(),
                processing_purposes: vec!["ml_training".to_string()],
                legal_basis: vec!["consent".to_string()],
                technical_measures: vec!["differential_privacy".to_string()],
                metadata: HashMap::new(),
            },
            privacy_context: PrivacyContext {
                epsilon_budget: epsilon,
                delta_budget: 1e-6,
                privacy_mechanism: "dp_sgd".to_string(),
                data_minimization: true,
                purpose_limitation: true,
                storage_limitation: true,
            },
            signature: None,
            compliance_annotations: HashMap::new(),
        }
    }

    #[test]
    fn metrics_are_created_on_first_sight() {
        // Regression: the update only ran when the metric already existed, and
        // nothing ever created it, so the dashboard stayed empty forever.
        let mut dashboard = MonitoringDashboard::new();
        assert!(dashboard.metric_names().is_empty());

        let ok = dashboard.update_metrics(&event("a", 0.25, 100));
        assert!(ok.is_ok(), "update failed");

        let epsilon = match dashboard.metric(METRIC_EPSILON) {
            Some(metric) => metric,
            None => panic!("the epsilon gauge must exist after one event"),
        };
        assert_eq!(epsilon.current_value, 0.25);
        assert_eq!(epsilon.historical_values.len(), 1);
        let events = match dashboard.metric(METRIC_EVENTS) {
            Some(metric) => metric,
            None => panic!("the event counter must exist"),
        };
        assert_eq!(events.current_value, 1.0);
    }

    #[test]
    fn counters_accumulate_and_gauges_track_the_latest_value() {
        let mut dashboard = MonitoringDashboard::new();
        for (index, epsilon) in [0.1, 0.2, 0.4].iter().enumerate() {
            let ok = dashboard.update_metrics(&event(
                &format!("e{index}"),
                *epsilon,
                100 + index as u64,
            ));
            assert!(ok.is_ok());
        }
        let gauge = match dashboard.metric(METRIC_EPSILON) {
            Some(metric) => metric,
            None => panic!("missing gauge"),
        };
        assert_eq!(gauge.current_value, 0.4, "a gauge holds the latest value");
        let cumulative = match dashboard.metric(METRIC_CUMULATIVE_EPSILON) {
            Some(metric) => metric,
            None => panic!("missing counter"),
        };
        assert!((cumulative.current_value - 0.7).abs() < 1e-12);
        let events = match dashboard.metric(METRIC_EVENTS) {
            Some(metric) => metric,
            None => panic!("missing counter"),
        };
        assert_eq!(events.current_value, 3.0);
    }

    #[test]
    fn per_event_type_counters_are_kept_separately() {
        let mut dashboard = MonitoringDashboard::new();
        let ok = dashboard.update_metrics(&event("a", 0.1, 1));
        assert!(ok.is_ok());
        let mut access = event("b", 0.1, 2);
        access.event_type = AuditEventType::DataAccess;
        let ok = dashboard.update_metrics(&access);
        assert!(ok.is_ok());

        assert_eq!(
            dashboard
                .metric("audit_events_PrivacyBudgetConsumption")
                .map(|metric| metric.current_value),
            Some(1.0)
        );
        assert_eq!(
            dashboard
                .metric("audit_events_DataAccess")
                .map(|metric| metric.current_value),
            Some(1.0)
        );
    }

    #[test]
    fn a_configured_threshold_actually_raises_an_alert() {
        let mut dashboard = MonitoringDashboard::new();
        let ok = dashboard.set_threshold(AlertThreshold {
            metric: METRIC_CUMULATIVE_EPSILON.to_string(),
            warning: 0.5,
            critical: 1.0,
            direction: ThresholdDirection::Above,
        });
        assert!(ok.is_ok());

        let ok = dashboard.update_metrics(&event("a", 0.2, 1));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 0, "0.2 is below the warning bound");

        let ok = dashboard.update_metrics(&event("b", 0.4, 2));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 1, "0.6 crosses the warning bound");
        let warning = match dashboard.alerts().next() {
            Some(alert) => alert,
            None => panic!("expected an alert"),
        };
        assert!(matches!(warning.severity, AlertSeverity::Warning));
        assert_eq!(warning.metric.as_deref(), Some(METRIC_CUMULATIVE_EPSILON));

        let ok = dashboard.update_metrics(&event("c", 0.5, 3));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 2, "1.1 crosses the critical bound");
        let critical = match dashboard.alerts().last() {
            Some(alert) => alert,
            None => panic!("expected an alert"),
        };
        assert!(matches!(critical.severity, AlertSeverity::Critical));
    }

    #[test]
    fn a_lower_bound_threshold_fires_on_the_way_down() {
        let mut dashboard = MonitoringDashboard::new();
        let ok = dashboard.set_threshold(AlertThreshold {
            metric: METRIC_EPSILON.to_string(),
            warning: 0.5,
            critical: 0.1,
            direction: ThresholdDirection::Below,
        });
        assert!(ok.is_ok());
        let ok = dashboard.update_metrics(&event("a", 0.9, 1));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 0);
        let ok = dashboard.update_metrics(&event("b", 0.4, 2));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 1);
        let ok = dashboard.update_metrics(&event("c", 0.05, 3));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 2);
    }

    #[test]
    fn an_inconsistent_threshold_is_refused() {
        let mut dashboard = MonitoringDashboard::new();
        assert!(dashboard
            .set_threshold(AlertThreshold {
                metric: METRIC_EPSILON.to_string(),
                warning: 1.0,
                critical: 0.5,
                direction: ThresholdDirection::Above,
            })
            .is_err());
        assert!(dashboard
            .set_threshold(AlertThreshold {
                metric: METRIC_EPSILON.to_string(),
                warning: f64::NAN,
                critical: 1.0,
                direction: ThresholdDirection::Above,
            })
            .is_err());
    }

    #[test]
    fn samples_outside_the_retention_window_are_dropped() {
        let config = DashboardConfig {
            history_retention_hours: 1,
            ..DashboardConfig::default()
        };
        let mut dashboard = MonitoringDashboard::with_config(config);

        let ok = dashboard.update_metrics(&event("old", 0.1, 0));
        assert!(ok.is_ok());
        let ok = dashboard.update_metrics(&event("recent", 0.1, 7_200));
        assert!(ok.is_ok());

        let gauge = match dashboard.metric(METRIC_EPSILON) {
            Some(metric) => metric,
            None => panic!("missing gauge"),
        };
        assert_eq!(
            gauge.historical_values.len(),
            1,
            "the two-hour-old sample must fall outside a one-hour window"
        );
    }

    #[test]
    fn a_non_finite_privacy_parameter_is_refused() {
        let mut dashboard = MonitoringDashboard::new();
        assert!(dashboard
            .update_metrics(&event("bad", f64::INFINITY, 1))
            .is_err());
    }

    #[test]
    fn alerts_can_be_acknowledged() {
        let mut dashboard = MonitoringDashboard::new();
        let ok = dashboard.set_threshold(AlertThreshold {
            metric: METRIC_EVENTS.to_string(),
            warning: 1.0,
            critical: 10.0,
            direction: ThresholdDirection::Above,
        });
        assert!(ok.is_ok());
        let ok = dashboard.update_metrics(&event("a", 0.1, 1));
        assert!(ok.is_ok());
        assert_eq!(dashboard.alert_count(), 1);
        assert!(dashboard.acknowledge_alert("dashboard_alert_0").is_ok());
        assert!(dashboard.acknowledge_alert("nope").is_err());
        let alert = match dashboard.alerts().next() {
            Some(alert) => alert,
            None => panic!("expected an alert"),
        };
        assert!(matches!(alert.status, AlertStatus::Acknowledged));
    }
}
