// Real alert-rule evaluation for the streaming metrics collector (finding M1).
//
// `AlertSystem::evaluate_rules` used to be `Ok(())` with a comment, so every
// configured rule was silently inert. It now resolves the rule's metric path
// against the live metrics, evaluates the condition, fires and *resolves*
// alerts, and dispatches to the notification channels that this crate can
// actually reach.

use super::*;
use crate::error::OptimError;
use std::collections::VecDeque;

/// Every metric path the resolver understands. Used by alert rules,
/// dashboards and the aggregation layer.
pub const KNOWN_METRIC_PATHS: &[&str] = &[
    "performance.throughput.samples_per_second",
    "performance.throughput.updates_per_second",
    "performance.throughput.gradients_per_second",
    "performance.throughput.peak_throughput",
    "performance.throughput.min_throughput",
    "performance.throughput.throughput_variance",
    "performance.throughput.throughput_trend",
    "performance.latency.end_to_end.mean",
    "performance.latency.end_to_end.median",
    "performance.latency.end_to_end.p95",
    "performance.latency.end_to_end.p99",
    "performance.latency.end_to_end.p999",
    "performance.latency.end_to_end.max",
    "performance.latency.end_to_end.min",
    "performance.latency.end_to_end.std_dev",
    "performance.latency.jitter",
    "performance.accuracy.current_loss",
    "performance.accuracy.loss_reduction_rate",
    "performance.accuracy.convergence_rate",
    "performance.accuracy.prediction_accuracy",
    "performance.accuracy.gradient_magnitude",
    "performance.accuracy.parameter_stability",
    "performance.accuracy.learning_progress",
    "performance.stability.loss_variance",
    "performance.stability.gradient_variance",
    "performance.stability.parameter_drift",
    "performance.stability.oscillation_score",
    "performance.stability.divergence_probability",
    "performance.stability.stability_confidence",
    "performance.efficiency.computational_efficiency",
    "performance.efficiency.memory_efficiency",
    "performance.efficiency.communication_efficiency",
    "performance.efficiency.energy_efficiency",
    "performance.efficiency.resource_utilization",
    "performance.efficiency.cost_efficiency",
    "resource.cpu_utilization",
    "resource.gpu_utilization",
    "resource.network_bandwidth",
    "resource.disk_io",
    "resource.thread_utilization",
    "resource.memory.current_used",
    "resource.memory.peak_usage",
    "resource.memory.total_allocated",
    "resource.memory.fragmentation_ratio",
    "resource.memory.efficiency",
    "quality.data_quality",
    "quality.model.training_quality",
    "quality.model.generalization_score",
    "quality.model.overfitting_score",
    "quality.model.underfitting_score",
    "quality.model.complexity_score",
    "quality.drift.drift_confidence",
    "quality.drift.drift_magnitude",
    "quality.drift.drift_frequency",
    "quality.drift.adaptation_effectiveness",
    "quality.drift.detection_latency",
    "quality.anomaly.anomaly_score",
    "quality.anomaly.anomaly_frequency",
    "quality.robustness.noise_tolerance",
    "quality.robustness.adversarial_robustness",
    "quality.robustness.perturbation_sensitivity",
    "quality.robustness.recovery_capability",
    "quality.robustness.fault_tolerance",
    "business.availability",
    "business.slo_compliance",
    "business.user_satisfaction",
    "business.business_value",
    "business.cost.computational_cost",
    "business.cost.infrastructure_cost",
    "business.cost.energy_cost",
    "business.cost.opportunity_cost",
    "business.cost.total_cost",
];

fn scalar<A: Float + Send + Sync>(value: A) -> Option<f64> {
    value.to_f64()
}

/// Resolve a dotted metric path against a metrics summary.
pub fn resolve_metric<A: Float + Send + Sync>(
    summary: &MetricsSummary<A>,
    path: &str,
) -> Option<f64> {
    resolve_components(
        &summary.performance,
        &summary.resource,
        &summary.quality,
        &summary.business,
        path,
    )
}

/// Resolve a dotted metric path against a stored snapshot.
pub(crate) fn resolve_snapshot_metric<A: Float + Send + Sync>(
    snapshot: &MetricsSnapshot<A>,
    path: &str,
) -> Option<f64> {
    resolve_components(
        &snapshot.performance,
        &snapshot.resource,
        &snapshot.quality,
        &snapshot.business,
        path,
    )
}

fn resolve_components<A: Float + Send + Sync>(
    performance: &PerformanceMetrics<A>,
    resource: &ResourceMetrics,
    quality: &QualityMetrics<A>,
    business: &BusinessMetrics<A>,
    path: &str,
) -> Option<f64> {
    match path {
        "performance.throughput.samples_per_second" => {
            Some(performance.throughput.samples_per_second)
        }
        "performance.throughput.updates_per_second" => {
            Some(performance.throughput.updates_per_second)
        }
        "performance.throughput.gradients_per_second" => {
            Some(performance.throughput.gradients_per_second)
        }
        "performance.throughput.peak_throughput" => Some(performance.throughput.peak_throughput),
        "performance.throughput.min_throughput" => Some(performance.throughput.min_throughput),
        "performance.throughput.throughput_variance" => {
            Some(performance.throughput.throughput_variance)
        }
        "performance.throughput.throughput_trend" => Some(performance.throughput.throughput_trend),
        "performance.latency.end_to_end.mean" => {
            Some(performance.latency.end_to_end.mean.as_secs_f64())
        }
        "performance.latency.end_to_end.median" => {
            Some(performance.latency.end_to_end.median.as_secs_f64())
        }
        "performance.latency.end_to_end.p95" => {
            Some(performance.latency.end_to_end.p95.as_secs_f64())
        }
        "performance.latency.end_to_end.p99" => {
            Some(performance.latency.end_to_end.p99.as_secs_f64())
        }
        "performance.latency.end_to_end.p999" => {
            Some(performance.latency.end_to_end.p999.as_secs_f64())
        }
        "performance.latency.end_to_end.max" => {
            Some(performance.latency.end_to_end.max.as_secs_f64())
        }
        "performance.latency.end_to_end.min" => {
            Some(performance.latency.end_to_end.min.as_secs_f64())
        }
        "performance.latency.end_to_end.std_dev" => {
            Some(performance.latency.end_to_end.std_dev.as_secs_f64())
        }
        "performance.latency.jitter" => Some(performance.latency.jitter),
        "performance.accuracy.current_loss" => scalar(performance.accuracy.current_loss),
        "performance.accuracy.loss_reduction_rate" => {
            scalar(performance.accuracy.loss_reduction_rate)
        }
        "performance.accuracy.convergence_rate" => scalar(performance.accuracy.convergence_rate),
        "performance.accuracy.prediction_accuracy" => {
            performance.accuracy.prediction_accuracy.and_then(scalar)
        }
        "performance.accuracy.gradient_magnitude" => {
            scalar(performance.accuracy.gradient_magnitude)
        }
        "performance.accuracy.parameter_stability" => {
            scalar(performance.accuracy.parameter_stability)
        }
        "performance.accuracy.learning_progress" => scalar(performance.accuracy.learning_progress),
        "performance.stability.loss_variance" => scalar(performance.stability.loss_variance),
        "performance.stability.gradient_variance" => {
            scalar(performance.stability.gradient_variance)
        }
        "performance.stability.parameter_drift" => scalar(performance.stability.parameter_drift),
        "performance.stability.oscillation_score" => {
            scalar(performance.stability.oscillation_score)
        }
        "performance.stability.divergence_probability" => {
            scalar(performance.stability.divergence_probability)
        }
        "performance.stability.stability_confidence" => {
            scalar(performance.stability.stability_confidence)
        }
        "performance.efficiency.computational_efficiency" => performance
            .efficiency
            .computational_efficiency
            .and_then(scalar),
        "performance.efficiency.memory_efficiency" => {
            performance.efficiency.memory_efficiency.and_then(scalar)
        }
        "performance.efficiency.communication_efficiency" => performance
            .efficiency
            .communication_efficiency
            .and_then(scalar),
        "performance.efficiency.energy_efficiency" => {
            performance.efficiency.energy_efficiency.and_then(scalar)
        }
        "performance.efficiency.resource_utilization" => {
            scalar(performance.efficiency.resource_utilization)
        }
        "performance.efficiency.cost_efficiency" => {
            performance.efficiency.cost_efficiency.and_then(scalar)
        }
        "resource.cpu_utilization" => resource.cpu_utilization,
        "resource.gpu_utilization" => resource.gpu_utilization,
        "resource.network_bandwidth" => resource.network_bandwidth,
        "resource.disk_io" => resource.disk_io,
        "resource.thread_utilization" => resource.thread_utilization,
        "resource.memory.current_used" => Some(resource.memory_usage.current_used as f64),
        "resource.memory.peak_usage" => Some(resource.memory_usage.peak_usage as f64),
        "resource.memory.total_allocated" => {
            resource.memory_usage.total_allocated.map(|v| v as f64)
        }
        "resource.memory.fragmentation_ratio" => resource.memory_usage.fragmentation_ratio,
        "resource.memory.efficiency" => resource.memory_usage.efficiency,
        "quality.data_quality" => scalar(quality.data_quality),
        "quality.model.training_quality" => scalar(quality.model_quality.training_quality),
        "quality.model.generalization_score" => {
            quality.model_quality.generalization_score.and_then(scalar)
        }
        "quality.model.overfitting_score" => {
            quality.model_quality.overfitting_score.and_then(scalar)
        }
        "quality.model.underfitting_score" => {
            quality.model_quality.underfitting_score.and_then(scalar)
        }
        "quality.model.complexity_score" => quality.model_quality.complexity_score.and_then(scalar),
        "quality.drift.drift_confidence" => quality.concept_drift.drift_confidence.and_then(scalar),
        "quality.drift.drift_magnitude" => quality.concept_drift.drift_magnitude.and_then(scalar),
        "quality.drift.drift_frequency" => Some(quality.concept_drift.drift_frequency),
        "quality.drift.adaptation_effectiveness" => quality
            .concept_drift
            .adaptation_effectiveness
            .and_then(scalar),
        "quality.drift.detection_latency" => quality
            .concept_drift
            .detection_latency
            .map(|d| d.as_secs_f64()),
        "quality.anomaly.anomaly_score" => scalar(quality.anomaly_detection.anomaly_score),
        "quality.anomaly.anomaly_frequency" => Some(quality.anomaly_detection.anomaly_frequency),
        "quality.robustness.noise_tolerance" => quality.robustness.noise_tolerance.and_then(scalar),
        "quality.robustness.adversarial_robustness" => {
            quality.robustness.adversarial_robustness.and_then(scalar)
        }
        "quality.robustness.perturbation_sensitivity" => {
            quality.robustness.perturbation_sensitivity.and_then(scalar)
        }
        "quality.robustness.recovery_capability" => {
            quality.robustness.recovery_capability.and_then(scalar)
        }
        "quality.robustness.fault_tolerance" => quality.robustness.fault_tolerance.and_then(scalar),
        "business.availability" => business.availability,
        "business.slo_compliance" => business.slo_compliance,
        "business.user_satisfaction" => business.user_satisfaction.and_then(scalar),
        "business.business_value" => business.business_value.and_then(scalar),
        "business.cost.computational_cost" => {
            business.cost_metrics.computational_cost.and_then(scalar)
        }
        "business.cost.infrastructure_cost" => {
            business.cost_metrics.infrastructure_cost.and_then(scalar)
        }
        "business.cost.energy_cost" => business.cost_metrics.energy_cost.and_then(scalar),
        "business.cost.opportunity_cost" => business.cost_metrics.opportunity_cost.and_then(scalar),
        "business.cost.total_cost" => business.cost_metrics.total_cost.and_then(scalar),
        _ => None,
    }
}

/// Resolve a path that may additionally refer to the sample being ingested.
fn resolve_rule_value<A: Float + Send + Sync>(
    summary: &MetricsSummary<A>,
    sample: &MetricsSample<A>,
    path: &str,
) -> Option<f64> {
    match path {
        "sample.loss" => scalar(sample.loss),
        "sample.gradient_magnitude" => scalar(sample.gradient_magnitude),
        "sample.processing_time_seconds" => Some(sample.processing_time.as_secs_f64()),
        "sample.memory_usage" => Some(sample.memory_usage as f64),
        other => {
            if let Some(name) = other.strip_prefix("custom.") {
                sample.custom_metrics.get(name).copied().and_then(scalar)
            } else {
                resolve_metric(summary, other)
            }
        }
    }
}

/// Per-rule bookkeeping kept by the evaluator.
#[derive(Debug, Default, Clone)]
pub(crate) struct RuleState {
    last_evaluated: Option<SystemTime>,
    history: VecDeque<(SystemTime, f64)>,
    observations: u64,
    mean: f64,
    m2: f64,
    active_alert_id: Option<String>,
}

impl RuleState {
    const MAX_HISTORY: usize = 512;

    fn observe(&mut self, at: SystemTime, value: f64) {
        self.history.push_back((at, value));
        while self.history.len() > Self::MAX_HISTORY {
            self.history.pop_front();
        }
        if value.is_finite() {
            self.observations += 1;
            let delta = value - self.mean;
            self.mean += delta / self.observations as f64;
            self.m2 += delta * (value - self.mean);
        }
    }

    fn std_dev(&self) -> f64 {
        if self.observations < 2 {
            0.0
        } else {
            (self.m2 / self.observations as f64).sqrt()
        }
    }

    /// Change of the observed value over `window`, or `None` when the history
    /// does not span the window yet.
    fn rate_of_change(&self, now: SystemTime, window: Duration) -> Option<f64> {
        let (_, latest) = *self.history.back()?;
        let oldest_in_window = self
            .history
            .iter()
            .find(|(at, _)| saturating_elapsed(now, *at) <= window)?;
        let span = saturating_elapsed(now, oldest_in_window.0).as_secs_f64();
        if span <= 0.0 {
            return None;
        }
        Some((latest - oldest_in_window.1) / span)
    }
}

fn compare(operator: ComparisonOperator, value: f64, threshold: f64) -> bool {
    match operator {
        ComparisonOperator::GreaterThan => value > threshold,
        ComparisonOperator::LessThan => value < threshold,
        ComparisonOperator::GreaterThanOrEqual => value >= threshold,
        ComparisonOperator::LessThanOrEqual => value <= threshold,
        ComparisonOperator::Equal => (value - threshold).abs() <= f64::EPSILON,
        ComparisonOperator::NotEqual => (value - threshold).abs() > f64::EPSILON,
    }
}

/// A human-readable identifier for a notification channel.
fn channel_id(channel: &NotificationChannel) -> String {
    match channel {
        NotificationChannel::Email { .. } => "email".to_string(),
        NotificationChannel::Webhook { url, .. } => format!("webhook:{url}"),
        NotificationChannel::Slack { channel, .. } => format!("slack:{channel}"),
        NotificationChannel::PagerDuty { .. } => "pagerduty".to_string(),
        NotificationChannel::Custom { config } => format!(
            "custom:{}",
            config.get("name").map(String::as_str).unwrap_or("unnamed")
        ),
    }
}

impl<A: Float + Send + Sync> AlertSystem<A> {
    pub(crate) fn new() -> Self {
        Self {
            rules: Vec::new(),
            active_alerts: Vec::new(),
            alert_history: Vec::new(),
            notification_channels: Vec::new(),
            rule_state: HashMap::new(),
            next_alert_id: 0,
            max_history: 1000,
        }
    }

    /// Register a rule, rejecting conditions and metric paths this crate
    /// cannot evaluate instead of silently ignoring them later.
    pub(crate) fn add_rule(&mut self, rule: AlertRule<A>) -> Result<()> {
        if let AlertCondition::Custom { expression } = &rule.condition {
            return Err(OptimError::UnsupportedOperation(format!(
                "alert rule '{}' uses a custom expression ('{expression}'); this crate has no \
                 expression evaluator, so the rule would never fire",
                rule.name
            )));
        }
        let path = rule.metric_path.as_str();
        let known = KNOWN_METRIC_PATHS.contains(&path)
            || path.starts_with("custom.")
            || matches!(
                path,
                "sample.loss"
                    | "sample.gradient_magnitude"
                    | "sample.processing_time_seconds"
                    | "sample.memory_usage"
            );
        if !known {
            return Err(OptimError::InvalidConfig(format!(
                "alert rule '{}' watches unknown metric path '{}'",
                rule.name, path
            )));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Register a notification channel this crate can actually reach.
    pub(crate) fn add_notification_channel(&mut self, channel: NotificationChannel) -> Result<()> {
        match &channel {
            NotificationChannel::Custom { config } if config.contains_key("path") => {
                self.notification_channels.push(channel);
                Ok(())
            }
            NotificationChannel::Custom { .. } => Err(OptimError::InvalidConfig(
                "a custom notification channel needs a 'path' entry naming the sink file"
                    .to_string(),
            )),
            _ => Err(OptimError::UnsupportedOperation(
                "optirs-core has no network client; email, webhook, Slack and PagerDuty \
                 notifications cannot be delivered from this crate"
                    .to_string(),
            )),
        }
    }

    pub(crate) fn evaluate_rules(
        &mut self,
        sample: &MetricsSample<A>,
        summary: &MetricsSummary<A>,
    ) -> Result<()> {
        let now = SystemTime::now();
        let rules = std::mem::take(&mut self.rules);

        for rule in &rules {
            // Honour the rule's own evaluation cadence.
            {
                let state = self.rule_state.entry(rule.name.clone()).or_default();
                if let Some(last) = state.last_evaluated {
                    if saturating_elapsed(now, last) < rule.evaluation_frequency {
                        continue;
                    }
                }
                state.last_evaluated = Some(now);
            }

            let Some(value) = resolve_rule_value(summary, sample, &rule.metric_path) else {
                // The metric has not been measured yet; nothing to assert.
                continue;
            };

            let breach = {
                let state = self.rule_state.entry(rule.name.clone()).or_default();
                state.observe(now, value);
                match &rule.condition {
                    AlertCondition::Threshold { operator, value: t } => {
                        let threshold = t.to_f64().unwrap_or(f64::NAN);
                        if threshold.is_finite() && compare(*operator, value, threshold) {
                            Some((value, threshold))
                        } else {
                            None
                        }
                    }
                    AlertCondition::RateOfChange {
                        threshold,
                        time_window,
                    } => {
                        let limit = threshold.to_f64().unwrap_or(f64::NAN);
                        match state.rate_of_change(now, *time_window) {
                            Some(rate) if limit.is_finite() && rate.abs() > limit.abs() => {
                                Some((rate, limit))
                            }
                            _ => None,
                        }
                    }
                    AlertCondition::Anomaly { sensitivity } => {
                        let sensitivity = sensitivity.to_f64().unwrap_or(f64::NAN);
                        let std = state.std_dev();
                        if sensitivity.is_finite() && std > 0.0 && state.observations >= 3 {
                            let z = (value - state.mean).abs() / std;
                            if z > sensitivity {
                                Some((z, sensitivity))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    AlertCondition::Custom { expression } => {
                        self.rules = rules.clone();
                        return Err(OptimError::UnsupportedOperation(format!(
                            "alert rule '{}' uses a custom expression ('{expression}') that \
                             cannot be evaluated",
                            rule.name
                        )));
                    }
                }
            };

            match breach {
                Some((observed, threshold)) => self.fire(rule, observed, threshold, now)?,
                None => self.resolve(rule, now),
            }
        }

        self.rules = rules;
        Ok(())
    }

    fn fire(
        &mut self,
        rule: &AlertRule<A>,
        value: f64,
        threshold: f64,
        now: SystemTime,
    ) -> Result<()> {
        let already_active = self
            .rule_state
            .get(&rule.name)
            .and_then(|state| state.active_alert_id.clone());

        if let Some(id) = already_active {
            if let Some(alert) = self.active_alerts.iter_mut().find(|alert| alert.id == id) {
                alert.current_value = A::from(value).unwrap_or_else(A::zero);
                return Ok(());
            }
        }

        self.next_alert_id += 1;
        let id = format!("{}#{}", rule.name, self.next_alert_id);
        let alert = Alert {
            id: id.clone(),
            rule_name: rule.name.clone(),
            triggered_at: now,
            resolved_at: None,
            current_value: A::from(value).unwrap_or_else(A::zero),
            threshold: A::from(threshold).unwrap_or_else(A::zero),
            severity: rule.severity,
            message: format!(
                "{} = {:.6} breached threshold {:.6} ({})",
                rule.metric_path, value, threshold, rule.name
            ),
        };
        if let Some(state) = self.rule_state.get_mut(&rule.name) {
            state.active_alert_id = Some(id);
        }
        self.dispatch(rule, &alert)?;
        self.active_alerts.push(alert);
        Ok(())
    }

    fn resolve(&mut self, rule: &AlertRule<A>, now: SystemTime) {
        let Some(state) = self.rule_state.get_mut(&rule.name) else {
            return;
        };
        let Some(id) = state.active_alert_id.take() else {
            return;
        };
        if let Some(position) = self.active_alerts.iter().position(|alert| alert.id == id) {
            let mut alert = self.active_alerts.remove(position);
            alert.resolved_at = Some(now);
            self.alert_history.push(alert);
            while self.alert_history.len() > self.max_history {
                self.alert_history.remove(0);
            }
        }
    }

    fn dispatch(&self, rule: &AlertRule<A>, alert: &Alert<A>) -> Result<()> {
        for channel in &self.notification_channels {
            let id = channel_id(channel);
            if !rule.notifications.is_empty()
                && !rule.notifications.iter().any(|name| id.contains(name))
            {
                continue;
            }
            match channel {
                NotificationChannel::Custom { config } => {
                    let Some(path) = config.get("path") else {
                        return Err(OptimError::InvalidConfig(
                            "custom notification channel is missing its 'path' entry".to_string(),
                        ));
                    };
                    use std::io::Write;
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .map_err(OptimError::IO)?;
                    writeln!(
                        file,
                        "{} [{:?}] {}",
                        unix_timestamp(alert.triggered_at),
                        alert.severity,
                        alert.message
                    )
                    .map_err(OptimError::IO)?;
                }
                other => {
                    return Err(OptimError::UnsupportedOperation(format!(
                        "cannot deliver alert '{}' through {}: optirs-core has no network client",
                        alert.id,
                        channel_id(other)
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> StreamingMetricsCollector<A> {
    /// Register a notification channel; remote channels are rejected because
    /// this crate cannot deliver to them.
    pub fn add_notification_channel(&mut self, channel: NotificationChannel) -> Result<()> {
        self.alert_system.add_notification_channel(channel)
    }

    /// Resolve a metric path against the current metrics.
    pub fn resolve_metric(&self, path: &str) -> Option<f64> {
        resolve_metric(&self.current_summary(), path)
    }
}
