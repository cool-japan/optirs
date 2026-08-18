// Alert system for performance regression notifications
//
// This module provides a comprehensive alerting system that can notify stakeholders
// through multiple channels (email, Slack, GitHub issues) when performance
// regressions are detected.
//
// Delivery goes through `crate::notification_transport`: no HTTP/SMTP client
// dependency is linked into this crate. By default (`TransportKind::Command`)
// delivery shells out to the system `curl`; set
// `OPTIRS_NOTIFICATION_TRANSPORT=file:<dir>`, `=log`, or `=disabled` to
// redirect delivery for tests/offline CI. A channel that is enabled but has
// no destination configured (see `AlertConfig::email`/`slack`/`github`)
// returns an explicit `Err` -- it is never reported as "sent successfully".

use crate::error::{OptimError, Result};
use crate::notification_transport::{
    self, DeliveryTarget, SmtpTarget, TransportKind, TransportMethod,
};
use crate::regression_tester::config::{Alert, AlertConfig, AlertSeverity};
use crate::regression_tester::types::RegressionResult;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Alert system for regression notifications
///
/// Manages alert generation, notification delivery, cooldown periods,
/// and integration with external services like email, Slack, and GitHub.
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Alert history for tracking and cooldown management
    alert_history: VecDeque<Alert>,
    /// Transport used for outbound delivery (curl by default; see module docs).
    transport_kind: TransportKind,
}

impl AlertSystem {
    /// Create a new alert system with default configuration
    pub fn new() -> Self {
        Self {
            config: AlertConfig::default(),
            alert_history: VecDeque::new(),
            transport_kind: notification_transport::transport_kind_from_env(),
        }
    }

    /// Create a new alert system with custom configuration
    pub fn with_config(config: AlertConfig) -> Self {
        Self {
            config,
            alert_history: VecDeque::new(),
            transport_kind: notification_transport::transport_kind_from_env(),
        }
    }

    /// Get the current alert configuration
    pub fn config(&self) -> &AlertConfig {
        &self.config
    }

    /// Update the alert configuration
    pub fn update_config(&mut self, config: AlertConfig) {
        self.config = config;
    }

    /// Get alert history
    pub fn alert_history(&self) -> &VecDeque<Alert> {
        &self.alert_history
    }

    /// Send an alert for a regression
    pub fn send_alert<A: Float>(&mut self, regression: &RegressionResult<A>) -> Result<()> {
        if regression.severity < self.config.severity_threshold {
            return Ok(()); // Below threshold
        }

        let alert = Alert::new(
            format!(
                "alert_{}",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            ),
            self.map_severity(regression.severity),
            format!(
                "Performance regression detected in {}: {:.2}% degradation",
                regression.test_id, regression.performance_change_percent
            ),
            regression.test_id.clone(),
        );

        self.alert_history.push_back(alert.clone());

        // Maintain alert history size
        if self.alert_history.len() > 100 {
            self.alert_history.pop_front();
        }

        // Send actual alerts through configured channels
        self.send_alert_notifications(&alert)?;

        Ok(())
    }

    /// Map numeric severity to AlertSeverity enum
    fn map_severity(&self, severity: f64) -> AlertSeverity {
        match severity {
            s if s >= 0.8 => AlertSeverity::Critical,
            s if s >= 0.6 => AlertSeverity::High,
            s if s >= 0.3 => AlertSeverity::Medium,
            _ => AlertSeverity::Low,
        }
    }

    /// Send alert notifications through configured channels. Every attempted
    /// channel's real outcome is logged via `log::debug!`/`log::error!`
    /// (never `println!`/`eprintln!`), and a channel that is enabled without
    /// a matching destination is treated as a delivery failure, not skipped
    /// silently.
    fn send_alert_notifications(&self, alert: &Alert) -> Result<()> {
        if !self.config.enable_alerts || self.severity_below_threshold(alert) {
            return Ok(());
        }

        if self.is_in_cooldown_period(alert)? {
            return Ok(());
        }

        let mut any_attempted = false;
        let mut any_failed = false;

        if self.config.enable_email {
            any_attempted = true;
            match self.send_email_notification(alert) {
                Ok(()) => log::debug!("alert {}: email sent successfully", alert.id),
                Err(e) => {
                    any_failed = true;
                    log::error!("alert {}: email delivery failed: {e}", alert.id);
                }
            }
        }

        if self.config.enable_slack {
            any_attempted = true;
            match self.send_slack_notification(alert) {
                Ok(()) => log::debug!("alert {}: Slack notification sent successfully", alert.id),
                Err(e) => {
                    any_failed = true;
                    log::error!("alert {}: Slack delivery failed: {e}", alert.id);
                }
            }
        }

        if self.config.enable_github_issues {
            any_attempted = true;
            match self.create_github_issue(alert) {
                Ok(()) => log::debug!("alert {}: GitHub issue created successfully", alert.id),
                Err(e) => {
                    any_failed = true;
                    log::error!("alert {}: GitHub issue creation failed: {e}", alert.id);
                }
            }
        }

        if any_attempted && any_failed {
            return Err(OptimError::InvalidConfig(format!(
                "one or more alert channels failed to deliver for alert {}; see log output",
                alert.id
            )));
        }

        Ok(())
    }

    /// Check if alert severity is below configured threshold
    fn severity_below_threshold(&self, alert: &Alert) -> bool {
        let alert_severity_value = match alert.severity {
            AlertSeverity::Critical => 1.0,
            AlertSeverity::High => 0.75,
            AlertSeverity::Medium => 0.5,
            AlertSeverity::Low => 0.25,
        };
        alert_severity_value < self.config.severity_threshold
    }

    /// Check if we're in cooldown period for similar alerts
    fn is_in_cooldown_period(&self, alert: &Alert) -> Result<bool> {
        let cooldown_duration = Duration::from_secs(self.config.cooldown_minutes * 60);
        let current_time = SystemTime::now();

        for recent_alert in self.alert_history.iter().rev().take(10) {
            if recent_alert.regression_id == alert.regression_id {
                let recent_time = UNIX_EPOCH + Duration::from_secs(recent_alert.timestamp);
                if current_time.duration_since(recent_time)? < cooldown_duration {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Build the "view details" link for an alert, or `None` when no
    /// `dashboard_base_url` is configured (never a fabricated domain).
    fn dashboard_link(&self, alert: &Alert) -> Option<String> {
        self.config
            .dashboard_base_url
            .as_ref()
            .map(|base| format!("{}/alerts/{}", base.trim_end_matches('/'), alert.id))
    }

    /// Send email notification via SMTP (through `notification_transport`,
    /// system `curl` by default). Returns `Err` when `enable_email` is set
    /// without a matching `AlertConfig::email` destination -- never `Ok(())`.
    fn send_email_notification(&self, alert: &Alert) -> Result<()> {
        let email_config = self.config.email.as_ref().ok_or_else(|| {
            OptimError::InvalidConfig(
                "enable_email is set but AlertConfig::email has no destination configured"
                    .to_string(),
            )
        })?;
        if email_config.recipients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "AlertConfig::email has no recipients configured".to_string(),
            ));
        }

        let subject = format!("Performance Regression Alert: {}", alert.regression_id);
        let body = self.format_email_body(alert);
        let message = format!(
            "From: {}\r\nTo: {}\r\nSubject: {}\r\n\r\n{}\r\n",
            email_config.from_address,
            email_config.recipients.join(", "),
            subject,
            body
        );

        let smtp_target = SmtpTarget {
            host: email_config.smtp_host.clone(),
            port: email_config.smtp_port,
            use_tls: email_config.use_tls,
            username: email_config.username.clone(),
            password: email_config.password.clone(),
            from: email_config.from_address.clone(),
            to: email_config.recipients.clone(),
            timeout: Duration::from_secs(30),
        };

        let outcome =
            notification_transport::deliver_email(&self.transport_kind, &smtp_target, &message)?;
        if outcome.is_success() {
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "email transport reported failure: {}",
                outcome.detail()
            )))
        }
    }

    /// Send Slack notification via the configured incoming webhook.
    fn send_slack_notification(&self, alert: &Alert) -> Result<()> {
        let slack_config = self.config.slack.as_ref().ok_or_else(|| {
            OptimError::InvalidConfig(
                "enable_slack is set but AlertConfig::slack has no destination configured"
                    .to_string(),
            )
        })?;
        if slack_config.webhook_url.is_empty() {
            return Err(OptimError::InvalidConfig(
                "AlertConfig::slack has no webhook_url configured".to_string(),
            ));
        }

        let slack_message = self.format_slack_message(alert);
        let payload = serde_json::json!({
            "channel": slack_config.channel,
            "username": "Performance Bot",
            "text": slack_message,
        });

        let target = DeliveryTarget::json_post(
            slack_config.webhook_url.clone(),
            slack_config.channel.clone(),
        );
        let outcome =
            notification_transport::deliver(&self.transport_kind, &target, &payload.to_string())?;
        if outcome.is_success() {
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "Slack transport reported failure: {}",
                outcome.detail()
            )))
        }
    }

    /// Create GitHub issue via the GitHub REST API.
    fn create_github_issue(&self, alert: &Alert) -> Result<()> {
        let github_config = self.config.github.as_ref().ok_or_else(|| {
            OptimError::InvalidConfig(
                "enable_github_issues is set but AlertConfig::github has no destination configured"
                    .to_string(),
            )
        })?;
        if github_config.repository.is_empty() || github_config.token.is_empty() {
            return Err(OptimError::InvalidConfig(
                "AlertConfig::github requires both a non-empty token and repository".to_string(),
            ));
        }

        let issue_title = format!("Performance regression in {}", alert.regression_id);
        let issue_body = self.format_github_issue_body(alert);
        let payload = serde_json::json!({
            "title": issue_title,
            "body": issue_body,
            "labels": ["performance", "regression", "automated"],
        });

        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", github_config.token),
        );
        headers.insert(
            "Accept".to_string(),
            "application/vnd.github+json".to_string(),
        );

        let target = DeliveryTarget {
            url: format!(
                "https://api.github.com/repos/{}/issues",
                github_config.repository
            ),
            method: TransportMethod::Post,
            headers,
            timeout: Duration::from_secs(30),
            channel: github_config.repository.clone(),
        };
        let outcome =
            notification_transport::deliver(&self.transport_kind, &target, &payload.to_string())?;
        if outcome.is_success() {
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "GitHub transport reported failure: {}",
                outcome.detail()
            )))
        }
    }

    /// Format email body for alert
    fn format_email_body(&self, alert: &Alert) -> String {
        let link = self
            .dashboard_link(alert)
            .map(|l| format!("\nView full details at: {l}\n"))
            .unwrap_or_default();
        format!(
            "Performance Regression Alert\n\
            =============================\n\n\
            Alert ID: {}\n\
            Timestamp: {}\n\
            Severity: {:?}\n\
            Test: {}\n\n\
            Details:\n\
            {}\n\
            {}\n\
            Please investigate this performance regression.\n",
            alert.id, alert.timestamp, alert.severity, alert.regression_id, alert.message, link
        )
    }

    /// Format Slack message for alert
    fn format_slack_message(&self, alert: &Alert) -> String {
        let severity_emoji = match alert.severity {
            AlertSeverity::Critical => "🚨",
            AlertSeverity::High => "⚠️",
            AlertSeverity::Medium => "🟡",
            AlertSeverity::Low => "🔵",
        };
        let link = self
            .dashboard_link(alert)
            .map(|l| format!("\n<{l}|View Details>"))
            .unwrap_or_default();

        format!(
            "{} *Performance Regression Alert*\n\
            *Test:* {}\n\
            *Severity:* {:?}\n\
            *Details:* {}\n\
            *Time:* <t:{}:F>{}",
            severity_emoji,
            alert.regression_id,
            alert.severity,
            alert.message,
            alert.timestamp,
            link
        )
    }

    /// Format GitHub issue body for alert
    fn format_github_issue_body(&self, alert: &Alert) -> String {
        let link = self
            .dashboard_link(alert)
            .map(|l| format!("- [Performance Dashboard]({l})\n"))
            .unwrap_or_default();
        format!(
            "## Performance Regression Detected\n\n\
            **Alert ID:** {}\n\
            **Timestamp:** {}\n\
            **Severity:** {:?}\n\
            **Test:** {}\n\n\
            ### Description\n\
            {}\n\n\
            ### Investigation Steps\n\
            - [ ] Review recent code changes that might affect performance\n\
            - [ ] Check system resource utilization during test execution\n\
            - [ ] Run additional test iterations to confirm regression\n\
            - [ ] Analyze profiling data for performance bottlenecks\n\
            - [ ] Compare with baseline performance metrics\n\n\
            ### Links\n\
            {}\n\
            ---\n\
            *This issue was automatically created by the performance monitoring system.*",
            alert.id, alert.timestamp, alert.severity, alert.regression_id, alert.message, link
        )
    }

    /// Get active alerts (not acknowledged or resolved)
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.alert_history
            .iter()
            .filter(|alert| alert.is_active())
            .collect()
    }

    /// Get recent alerts within the specified duration
    pub fn get_recent_alerts(&self, duration: Duration) -> Vec<&Alert> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(duration.as_secs());

        self.alert_history
            .iter()
            .filter(|alert| alert.timestamp >= cutoff_time)
            .collect()
    }

    /// Acknowledge an alert by ID
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<()> {
        for alert in &mut self.alert_history {
            if alert.id == alert_id {
                alert.acknowledge();
                return Ok(());
            }
        }
        Err(crate::error::OptimError::InvalidParameter(format!(
            "Alert with ID {} not found",
            alert_id
        )))
    }

    /// Resolve an alert by ID
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<()> {
        for alert in &mut self.alert_history {
            if alert.id == alert_id {
                alert.resolve();
                return Ok(());
            }
        }
        Err(crate::error::OptimError::InvalidParameter(format!(
            "Alert with ID {} not found",
            alert_id
        )))
    }

    /// Clear old alerts from history
    pub fn cleanup_old_alerts(&mut self, max_age: Duration) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let cutoff_time = now.saturating_sub(max_age.as_secs());

        log::debug!(
            "cleanup_old_alerts: now={now}, max_age={:?}, cutoff_time={cutoff_time}",
            max_age
        );

        let original_len = self.alert_history.len();
        self.alert_history.retain(|alert| {
            let keep = alert.timestamp >= cutoff_time;
            log::debug!(
                "alert {} timestamp {} >= cutoff {cutoff_time}: {keep}",
                alert.id,
                alert.timestamp
            );
            keep
        });
        let removed = original_len - self.alert_history.len();
        log::debug!("cleanup_old_alerts: removed {removed} alert(s)");
        removed
    }

    /// Get statistics about alerts
    pub fn get_alert_statistics(&self) -> AlertStatistics {
        let total_alerts = self.alert_history.len();
        let active_alerts = self.get_active_alerts().len();

        let severity_counts =
            self.alert_history
                .iter()
                .fold(SeverityCounts::default(), |mut counts, alert| {
                    match alert.severity {
                        AlertSeverity::Critical => counts.critical += 1,
                        AlertSeverity::High => counts.high += 1,
                        AlertSeverity::Medium => counts.medium += 1,
                        AlertSeverity::Low => counts.low += 1,
                    }
                    counts
                });

        AlertStatistics {
            total_alerts,
            active_alerts,
            severity_counts,
        }
    }
}

impl Default for AlertSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    /// Total number of alerts in history
    pub total_alerts: usize,
    /// Number of active (unresolved) alerts
    pub active_alerts: usize,
    /// Count of alerts by severity level
    pub severity_counts: SeverityCounts,
}

/// Count of alerts by severity level
#[derive(Debug, Clone, Default)]
pub struct SeverityCounts {
    /// Number of critical alerts
    pub critical: usize,
    /// Number of high severity alerts
    pub high: usize,
    /// Number of medium severity alerts
    pub medium: usize,
    /// Number of low severity alerts
    pub low: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression_tester::config::{
        AlertStatus, EmailAlertConfig, GitHubAlertConfig, SlackAlertConfig,
    };
    use crate::regression_tester::types::{
        ChangePointAnalysis, OutlierAnalysis, RegressionAnalysis, TrendAnalysis, TrendDirection,
    };

    fn create_test_regression(severity: f64, test_id: &str) -> RegressionResult<f64> {
        RegressionResult {
            test_id: test_id.to_string(),
            regression_detected: true,
            severity,
            confidence: 0.95,
            performance_change_percent: 15.0,
            memory_change_percent: 5.0,
            affected_metrics: vec!["timing".to_string()],
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: TrendDirection::Degrading,
                    magnitude: 15.0,
                    significance: 0.95,
                    start_point: None,
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: vec![],
        }
    }

    #[test]
    fn test_alert_system_creation() {
        let alert_system = AlertSystem::new();
        assert!(alert_system.config().enable_alerts);
        assert_eq!(alert_system.alert_history().len(), 0);
    }

    #[test]
    fn test_send_alert_above_threshold() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.8, "test_high_severity");

        let result = alert_system.send_alert(&regression);
        assert!(result.is_ok());
        assert_eq!(alert_system.alert_history().len(), 1);

        let alert = &alert_system.alert_history()[0];
        assert!(matches!(alert.severity, AlertSeverity::Critical));
        assert_eq!(alert.regression_id, "test_high_severity");
    }

    #[test]
    fn test_send_alert_below_threshold() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.01, "test_low_severity"); // Below default threshold of 0.05

        let result = alert_system.send_alert(&regression);
        assert!(result.is_ok());
        assert_eq!(alert_system.alert_history().len(), 0); // Below threshold
    }

    #[test]
    fn test_severity_mapping() {
        let alert_system = AlertSystem::new();

        assert!(matches!(
            alert_system.map_severity(0.9),
            AlertSeverity::Critical
        ));
        assert!(matches!(
            alert_system.map_severity(0.7),
            AlertSeverity::High
        ));
        assert!(matches!(
            alert_system.map_severity(0.4),
            AlertSeverity::Medium
        ));
        assert!(matches!(alert_system.map_severity(0.1), AlertSeverity::Low));
    }

    #[test]
    fn test_alert_history_limit() {
        let mut alert_system = AlertSystem::new();

        // Add more than 100 alerts
        for i in 0..105 {
            let regression = create_test_regression(0.8, &format!("test_{}", i));
            let _ = alert_system.send_alert(&regression);
        }

        // Should maintain limit of 100
        assert_eq!(alert_system.alert_history().len(), 100);
    }

    #[test]
    fn test_custom_config() {
        let custom_config = AlertConfig {
            enable_alerts: true,
            enable_email: true,
            enable_slack: true,
            enable_github_issues: false,
            severity_threshold: 0.8,
            cooldown_minutes: 30,
            ..Default::default()
        };

        let alert_system = AlertSystem::with_config(custom_config.clone());
        assert_eq!(alert_system.config().severity_threshold, 0.8);
        assert_eq!(alert_system.config().cooldown_minutes, 30);
        assert!(alert_system.config().enable_email);
        assert!(alert_system.config().enable_slack);
        assert!(!alert_system.config().enable_github_issues);
    }

    #[test]
    fn test_alert_statistics() {
        let mut alert_system = AlertSystem::new();

        let _ = alert_system.send_alert(&create_test_regression(0.9, "critical"));
        let _ = alert_system.send_alert(&create_test_regression(0.7, "high"));
        let _ = alert_system.send_alert(&create_test_regression(0.4, "medium"));
        let _ = alert_system.send_alert(&create_test_regression(0.2, "low"));

        let stats = alert_system.get_alert_statistics();
        assert_eq!(stats.total_alerts, 4);
        assert_eq!(stats.active_alerts, 4);
        assert_eq!(stats.severity_counts.critical, 1);
        assert_eq!(stats.severity_counts.high, 1);
        assert_eq!(stats.severity_counts.medium, 1);
        assert_eq!(stats.severity_counts.low, 1);
    }

    #[test]
    fn test_alert_acknowledgment() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.8, "test_ack");

        let _ = alert_system.send_alert(&regression);
        let alert_id = alert_system.alert_history()[0].id.clone();

        let result = alert_system.acknowledge_alert(&alert_id);
        assert!(result.is_ok());

        let alert = &alert_system.alert_history()[0];
        assert!(!alert.is_active());
        assert!(matches!(alert.status, AlertStatus::Acknowledged));
    }

    #[test]
    fn test_cleanup_old_alerts() {
        let mut alert_system = AlertSystem::new();

        for i in 0..5 {
            let regression = create_test_regression(0.8, &format!("test_{}", i));
            let _ = alert_system.send_alert(&regression);
        }

        assert_eq!(alert_system.alert_history().len(), 5);

        std::thread::sleep(Duration::from_secs(2));
        let removed = alert_system.cleanup_old_alerts(Duration::from_secs(1));
        assert_eq!(removed, 5);
        assert_eq!(alert_system.alert_history().len(), 0);
    }

    #[test]
    fn enabled_email_without_destination_is_explicit_err() {
        let mut alert_system = AlertSystem::with_config(AlertConfig {
            enable_alerts: true,
            enable_email: true,
            severity_threshold: 0.0,
            cooldown_minutes: 0,
            ..Default::default()
        });
        let regression = create_test_regression(0.9, "no_email_destination");
        let result = alert_system.send_alert(&regression);
        assert!(
            result.is_err(),
            "enabling email without a destination must never report success"
        );
    }

    #[test]
    fn enabled_channels_with_destinations_deliver_via_file_transport() {
        let dir = std::env::temp_dir().join(format!(
            "optirs_bench_alert_transport_test_{}",
            std::process::id()
        ));
        std::env::set_var(
            "OPTIRS_NOTIFICATION_TRANSPORT",
            format!("file:{}", dir.display()),
        );

        let mut alert_system = AlertSystem::with_config(AlertConfig {
            enable_alerts: true,
            enable_email: true,
            enable_slack: true,
            enable_github_issues: true,
            severity_threshold: 0.0,
            cooldown_minutes: 0,
            email: Some(EmailAlertConfig {
                smtp_host: "smtp.example.invalid".to_string(),
                smtp_port: 587,
                use_tls: false,
                username: None,
                password: None,
                from_address: "alerts@example.invalid".to_string(),
                recipients: vec!["oncall@example.invalid".to_string()],
            }),
            slack: Some(SlackAlertConfig {
                webhook_url: "https://hooks.slack.example.invalid/services/T/B/X".to_string(),
                channel: "#performance-alerts".to_string(),
            }),
            github: Some(GitHubAlertConfig {
                token: "test-token".to_string(),
                repository: "example/repo".to_string(),
            }),
            dashboard_base_url: Some("https://dashboards.example.invalid".to_string()),
        });

        let regression = create_test_regression(0.9, "all_channels");
        let result = alert_system.send_alert(&regression);
        std::env::remove_var("OPTIRS_NOTIFICATION_TRANSPORT");

        assert!(
            result.is_ok(),
            "all channels configured should succeed: {result:?}"
        );
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("outbox dir should exist")
            .collect();
        assert!(
            entries.len() >= 3,
            "expected at least 3 outbox files (email, slack, github), found {}",
            entries.len()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
