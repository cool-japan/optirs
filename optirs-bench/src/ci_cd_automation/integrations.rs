// External Service Integrations
//
// This module provides comprehensive integration capabilities with external services
// including GitHub, Slack, email, webhooks, and custom integrations for CI/CD automation.
//
// Every integration delivers through `crate::notification_transport`: no HTTP client
// dependency is linked into this crate. By default (`TransportKind::Command`) delivery
// shells out to the system `curl`; set `OPTIRS_NOTIFICATION_TRANSPORT=file:<dir>`,
// `=log`, or `=disabled` to redirect delivery for tests/offline CI. An integration that
// is not configured (no token/webhook URL/etc.) or whose transport is `Disabled` always
// returns `IntegrationStatus::Disabled` plus an `Err` -- it never reports `Ok(())` for a
// notification that was not actually sent.

use crate::error::{OptimError, Result};
use crate::notification_transport::{
    self, DeliveryOutcome, DeliveryTarget, SmtpTarget, TransportKind, TransportMethod,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::config::{
    EmailIntegration, GitHubIntegration, HttpMethod, IntegrationConfig, PayloadFormat,
    SlackIntegration, WebhookAuth, WebhookIntegration, WebhookPayloadConfig, WebhookRetryConfig,
};
// Only exercised by the unit tests below -- gated so a non-test build does not
// warn about unused imports.
#[cfg(test)]
use super::config::WebhookTriggerConfig;
use super::reporting::GeneratedReport;
use super::test_execution::{CiCdTestResult, TestExecutionStatus, TestSuiteStatistics};

/// Integration manager for handling external service connections
#[derive(Debug)]
pub struct IntegrationManager {
    /// Configuration settings
    pub config: IntegrationConfig,
    /// GitHub integration client
    pub github_client: Option<GitHubClient>,
    /// Slack integration client
    pub slack_client: Option<SlackClient>,
    /// Email integration client
    pub email_client: Option<EmailClient>,
    /// Webhook clients
    pub webhook_clients: Vec<WebhookClient>,
    /// Custom integration handlers
    pub custom_integrations: HashMap<String, Box<dyn CustomIntegration>>,
    /// Integration statistics, refreshed from each client after every send.
    pub statistics: IntegrationStatistics,
}

/// GitHub integration client
#[derive(Debug, Clone)]
pub struct GitHubClient {
    /// GitHub configuration
    pub config: GitHubIntegration,
    /// HTTP client for API calls
    pub http_client: HttpClient,
    /// Rate limiter
    pub rate_limiter: RateLimiter,
    /// Delivery statistics, updated after every real request attempt.
    pub statistics: GitHubStatistics,
}

/// Slack integration client
#[derive(Debug, Clone)]
pub struct SlackClient {
    /// Slack configuration
    pub config: SlackIntegration,
    /// HTTP client for API calls
    pub http_client: HttpClient,
    /// Delivery statistics, updated after every real request attempt.
    pub statistics: SlackStatistics,
}

/// Email integration client
#[derive(Debug, Clone)]
pub struct EmailClient {
    /// Email configuration
    pub config: EmailIntegration,
    /// SMTP transport selection (curl SMTP submission by default).
    pub transport_kind: TransportKind,
    /// Delivery statistics, updated after every real send attempt.
    pub statistics: EmailStatistics,
}

/// Webhook integration client
#[derive(Debug, Clone)]
pub struct WebhookClient {
    /// Webhook configuration
    pub config: WebhookIntegration,
    /// HTTP client for requests
    pub http_client: HttpClient,
    /// Retry manager
    pub retry_manager: RetryManager,
    /// Delivery statistics, updated after every real request attempt.
    pub statistics: WebhookStatistics,
}

/// Custom integration trait
pub trait CustomIntegration: std::fmt::Debug + Send + Sync {
    /// Initialize the integration
    fn initialize(&mut self, config: &HashMap<String, String>) -> Result<()>;

    /// Send notification
    fn send_notification(&self, notification: &IntegrationNotification) -> Result<()>;

    /// Handle test results
    fn handle_test_results(
        &self,
        results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
    ) -> Result<()>;

    /// Handle report generation
    fn handle_report_generated(&self, report: &GeneratedReport) -> Result<()>;

    /// Get integration status
    fn get_status(&self) -> IntegrationStatus;

    /// Validate configuration
    fn validate_config(&self, config: &HashMap<String, String>) -> Result<()>;
}

/// HTTP client for making API requests. Delivery is performed by
/// `crate::notification_transport` (system `curl` by default); this struct
/// only holds request defaults (base URL, headers, timeout) and the chosen
/// transport.
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Base URL for requests
    pub base_url: String,
    /// Default headers
    pub default_headers: HashMap<String, String>,
    /// Request timeout
    pub timeout: Duration,
    /// User agent string
    pub user_agent: String,
    /// Transport used to actually deliver requests.
    pub transport_kind: TransportKind,
}

/// Rate limiter for API calls. Tracks requests against a configurable
/// window (per-minute or per-hour), not a hardcoded 60-second bucket.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Requests allowed per window
    pub limit: u32,
    /// Length of the rate-limit window
    pub window: Duration,
    /// Current request count within the window
    pub current_requests: u32,
    /// Reset time
    pub reset_time: SystemTime,
    /// Request history
    pub request_history: Vec<SystemTime>,
}

/// Retry manager for failed requests
#[derive(Debug, Clone)]
pub struct RetryManager {
    /// Retry configuration
    pub config: WebhookRetryConfig,
    /// Transport used to re-issue retried requests.
    pub transport_kind: TransportKind,
    /// Failed requests queue
    pub failed_requests: Vec<FailedRequest>,
    /// Retry statistics
    pub statistics: RetryStatistics,
}

/// Failed request information
#[derive(Debug, Clone)]
pub struct FailedRequest {
    /// Request ID
    pub id: String,
    /// Original request data
    pub request_data: RequestData,
    /// Failure timestamp
    pub failed_at: SystemTime,
    /// Number of retry attempts
    pub retry_attempts: u32,
    /// Last error message
    pub last_error: String,
    /// Next retry time
    pub next_retry_at: SystemTime,
}

/// Request data for retries
#[derive(Debug, Clone)]
pub struct RequestData {
    /// HTTP method
    pub method: HttpMethod,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body
    pub body: String,
}

/// Retry statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetryStatistics {
    /// Total retry attempts
    pub total_retries: u64,
    /// Successful retries
    pub successful_retries: u64,
    /// Failed retries
    pub failed_retries: u64,
    /// Average retry delay
    pub average_retry_delay_sec: f64,
}

/// Integration notification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationNotification {
    /// Notification type
    pub notification_type: NotificationType,
    /// Notification title
    pub title: String,
    /// Notification message
    pub message: String,
    /// Notification priority
    pub priority: NotificationPriority,
    /// Additional data. Recognized keys used for GitHub delivery:
    /// `commit_sha` (status checks) and `pr_number` (PR comments); both fall
    /// back to the `GITHUB_SHA`/`PR_NUMBER`/`GITHUB_REF` environment
    /// variables when absent.
    pub data: HashMap<String, String>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Types of notifications
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NotificationType {
    /// Test completion notification
    TestCompletion,
    /// Test failure notification
    TestFailure,
    /// Performance regression detected
    PerformanceRegression,
    /// Performance improvement detected
    PerformanceImprovement,
    /// Report generated notification
    ReportGenerated,
    /// System alert
    SystemAlert,
    /// Custom notification
    Custom(String),
}

/// Notification priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Integration status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntegrationStatus {
    /// Integration is healthy
    Healthy,
    /// Integration has warnings
    Warning,
    /// Integration has errors
    Error,
    /// Integration is disabled
    Disabled,
    /// Integration is not configured
    NotConfigured,
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntegrationStatistics {
    /// GitHub statistics
    pub github: Option<GitHubStatistics>,
    /// Slack statistics
    pub slack: Option<SlackStatistics>,
    /// Email statistics
    pub email: Option<EmailStatistics>,
    /// Webhook statistics
    pub webhook: WebhookStatistics,
    /// Custom integration statistics
    pub custom: HashMap<String, CustomIntegrationStatistics>,
}

/// GitHub integration statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GitHubStatistics {
    /// Status checks created
    pub status_checks_created: u64,
    /// PR comments created
    pub pr_comments_created: u64,
    /// Issues created
    pub issues_created: u64,
    /// API requests made
    pub api_requests: u64,
    /// Rate limit hits
    pub rate_limit_hits: u64,
    /// Last activity timestamp
    pub last_activity: Option<SystemTime>,
}

/// Slack integration statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlackStatistics {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages that failed to send
    pub messages_failed: u64,
    /// Files uploaded
    pub files_uploaded: u64,
    /// Channels used
    pub channels_used: Vec<String>,
    /// Last activity timestamp
    pub last_activity: Option<SystemTime>,
}

/// Email integration statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmailStatistics {
    /// Emails sent
    pub emails_sent: u64,
    /// Emails failed
    pub emails_failed: u64,
    /// Recipients contacted
    pub recipients_contacted: Vec<String>,
    /// Last activity timestamp
    pub last_activity: Option<SystemTime>,
}

/// Webhook integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookStatistics {
    /// Requests sent
    pub requests_sent: u64,
    /// Requests successful
    pub requests_successful: u64,
    /// Requests failed
    pub requests_failed: u64,
    /// Average response time
    pub average_response_time_ms: f64,
    /// Retry statistics
    pub retry_stats: RetryStatistics,
}

/// Custom integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIntegrationStatistics {
    /// Integration name
    pub name: String,
    /// Operations performed
    pub operations_performed: u64,
    /// Operations successful
    pub operations_successful: u64,
    /// Operations failed
    pub operations_failed: u64,
    /// Last activity timestamp
    pub last_activity: Option<SystemTime>,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new(config: IntegrationConfig) -> Result<Self> {
        let mut manager = Self {
            config: config.clone(),
            github_client: None,
            slack_client: None,
            email_client: None,
            webhook_clients: Vec::new(),
            custom_integrations: HashMap::new(),
            statistics: IntegrationStatistics::default(),
        };

        manager.initialize_integrations()?;
        Ok(manager)
    }

    /// Initialize all configured integrations
    fn initialize_integrations(&mut self) -> Result<()> {
        if let Some(github_config) = &self.config.github {
            self.github_client = Some(GitHubClient::new(github_config.clone())?);
        }

        if let Some(slack_config) = &self.config.slack {
            self.slack_client = Some(SlackClient::new(slack_config.clone())?);
        }

        if let Some(email_config) = &self.config.email {
            self.email_client = Some(EmailClient::new(email_config.clone())?);
        }

        for webhook_config in &self.config.webhooks {
            let client = WebhookClient::new(webhook_config.clone())?;
            self.webhook_clients.push(client);
        }

        // Custom integrations are provided externally via `register_custom_integration`;
        // nothing further to do for entries that are merely configured but not yet
        // registered (`enabled: false` explicitly means "do not attempt to use it").

        Ok(())
    }

    /// Register a concrete custom integration handler for a name declared in
    /// `config.custom`. Returns an error if no matching (enabled)
    /// configuration entry exists, rather than silently accepting handlers
    /// for unconfigured names.
    pub fn register_custom_integration(
        &mut self,
        name: impl Into<String>,
        mut handler: Box<dyn CustomIntegration>,
    ) -> Result<()> {
        let name = name.into();
        let entry = self.config.custom.get(&name).ok_or_else(|| {
            OptimError::InvalidConfig(format!("no custom integration configured named '{name}'"))
        })?;
        if !entry.enabled {
            return Err(OptimError::InvalidConfig(format!(
                "custom integration '{name}' is configured but disabled"
            )));
        }
        handler.initialize(&entry.parameters)?;
        self.custom_integrations.insert(name, handler);
        Ok(())
    }

    /// Send notification to all configured integrations. Returns `Err` if
    /// any *configured* integration fails to deliver; partially-succeeded
    /// deliveries are still reflected truthfully in `self.statistics`
    /// before the error is returned.
    pub fn send_notification(&mut self, notification: &IntegrationNotification) -> Result<()> {
        let mut first_error: Option<OptimError> = None;

        if self.should_send_to_github(&notification.notification_type) {
            if let Some(github_client) = &mut self.github_client {
                if let Err(err) = github_client.send_notification(notification) {
                    first_error.get_or_insert(err);
                }
            }
        }

        if self.should_send_to_slack(&notification.notification_type) {
            if let Some(slack_client) = &mut self.slack_client {
                if let Err(err) = slack_client.send_notification(notification) {
                    first_error.get_or_insert(err);
                }
            }
        }

        if self.should_send_to_email(&notification.notification_type) {
            if let Some(email_client) = &mut self.email_client {
                if let Err(err) = email_client.send_notification(notification) {
                    first_error.get_or_insert(err);
                }
            }
        }

        for webhook_client in &mut self.webhook_clients {
            if webhook_client.should_trigger(&notification.notification_type) {
                if let Err(err) = webhook_client.send_notification(notification) {
                    first_error.get_or_insert(err);
                }
            }
        }

        for integration in self.custom_integrations.values() {
            if let Err(err) = integration.send_notification(notification) {
                first_error.get_or_insert(err);
            }
        }

        self.refresh_statistics();

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Handle test completion
    pub fn handle_test_completion(
        &mut self,
        results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
    ) -> Result<()> {
        let notification_type = if statistics.failed > 0 {
            NotificationType::TestFailure
        } else {
            NotificationType::TestCompletion
        };

        let priority = if statistics.failed > 0 {
            NotificationPriority::High
        } else {
            NotificationPriority::Normal
        };

        let notification = IntegrationNotification {
            notification_type,
            title: format!(
                "Test Suite Completed: {}/{} Passed",
                statistics.passed, statistics.total_tests
            ),
            message: self.create_test_summary_message(results, statistics),
            priority,
            data: self.create_test_data_map(statistics),
            timestamp: SystemTime::now(),
        };

        self.send_notification(&notification)?;

        for integration in self.custom_integrations.values() {
            integration.handle_test_results(results, statistics)?;
        }

        Ok(())
    }

    /// Handle report generation
    pub fn handle_report_generated(&mut self, report: &GeneratedReport) -> Result<()> {
        let notification = IntegrationNotification {
            notification_type: NotificationType::ReportGenerated,
            title: format!("Performance Report Generated: {:?}", report.report_type),
            message: format!("Report available at: {:?}", report.file_path),
            priority: NotificationPriority::Normal,
            data: HashMap::new(),
            timestamp: SystemTime::now(),
        };

        self.send_notification(&notification)?;

        for integration in self.custom_integrations.values() {
            integration.handle_report_generated(report)?;
        }

        Ok(())
    }

    /// Get overall integration health status
    pub fn get_health_status(&self) -> IntegrationStatus {
        let mut has_errors = false;
        let mut has_warnings = false;

        if let Some(github_client) = &self.github_client {
            match github_client.get_health_status() {
                IntegrationStatus::Error => has_errors = true,
                IntegrationStatus::Warning => has_warnings = true,
                _ => {}
            }
        }

        if let Some(slack_client) = &self.slack_client {
            match slack_client.get_health_status() {
                IntegrationStatus::Error => has_errors = true,
                IntegrationStatus::Warning => has_warnings = true,
                _ => {}
            }
        }

        if let Some(email_client) = &self.email_client {
            match email_client.get_health_status() {
                IntegrationStatus::Error => has_errors = true,
                IntegrationStatus::Warning => has_warnings = true,
                _ => {}
            }
        }

        for webhook_client in &self.webhook_clients {
            match webhook_client.get_health_status() {
                IntegrationStatus::Error => has_errors = true,
                IntegrationStatus::Warning => has_warnings = true,
                _ => {}
            }
        }

        for integration in self.custom_integrations.values() {
            match integration.get_status() {
                IntegrationStatus::Error => has_errors = true,
                IntegrationStatus::Warning => has_warnings = true,
                _ => {}
            }
        }

        if has_errors {
            IntegrationStatus::Error
        } else if has_warnings {
            IntegrationStatus::Warning
        } else {
            IntegrationStatus::Healthy
        }
    }

    /// Rebuild `self.statistics` from each live client's own counters. This
    /// is the only place `self.statistics` is written, so it can never drift
    /// from what actually happened during delivery.
    fn refresh_statistics(&mut self) {
        let webhook =
            self.webhook_clients
                .iter()
                .fold(WebhookStatistics::default(), |mut acc, client| {
                    acc.requests_sent += client.statistics.requests_sent;
                    acc.requests_successful += client.statistics.requests_successful;
                    acc.requests_failed += client.statistics.requests_failed;
                    acc.retry_stats.total_retries += client.statistics.retry_stats.total_retries;
                    acc.retry_stats.successful_retries +=
                        client.statistics.retry_stats.successful_retries;
                    acc.retry_stats.failed_retries += client.statistics.retry_stats.failed_retries;
                    acc
                });

        self.statistics = IntegrationStatistics {
            github: self.github_client.as_ref().map(|c| c.statistics.clone()),
            slack: self.slack_client.as_ref().map(|c| c.statistics.clone()),
            email: self.email_client.as_ref().map(|c| c.statistics.clone()),
            webhook,
            custom: self.statistics.custom.clone(),
        };
    }

    /// Determine if notification should be sent to GitHub
    fn should_send_to_github(&self, notification_type: &NotificationType) -> bool {
        if let Some(github_config) = &self.config.github {
            match notification_type {
                NotificationType::TestCompletion | NotificationType::TestFailure => {
                    github_config.create_status_checks || github_config.create_pr_comments
                }
                NotificationType::PerformanceRegression => github_config.create_regression_issues,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Determine if notification should be sent to Slack
    fn should_send_to_slack(&self, notification_type: &NotificationType) -> bool {
        if let Some(slack_config) = &self.config.slack {
            match notification_type {
                NotificationType::TestCompletion => slack_config.notifications.notify_on_completion,
                NotificationType::TestFailure => slack_config.notifications.notify_on_failure,
                NotificationType::PerformanceRegression => {
                    slack_config.notifications.notify_on_regression
                }
                NotificationType::PerformanceImprovement => {
                    slack_config.notifications.notify_on_improvement
                }
                _ => true, // Send other notifications by default
            }
        } else {
            false
        }
    }

    /// Determine if notification should be sent to Email
    fn should_send_to_email(&self, notification_type: &NotificationType) -> bool {
        // For simplicity, send high and critical priority notifications via email
        matches!(
            notification_type,
            NotificationType::TestFailure
                | NotificationType::PerformanceRegression
                | NotificationType::SystemAlert
        )
    }

    /// Create test summary message
    fn create_test_summary_message(
        &self,
        results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
    ) -> String {
        let mut message = format!(
            "Test Suite Results:\n• Total: {}\n• Passed: {}\n• Failed: {}\n• Skipped: {}\n• Success Rate: {:.1}%\n",
            statistics.total_tests,
            statistics.passed,
            statistics.failed,
            statistics.skipped,
            statistics.success_rate * 100.0
        );

        if statistics.failed > 0 {
            message.push_str("\nFailed Tests:\n");
            for result in results
                .iter()
                .filter(|r| r.status == TestExecutionStatus::Failed)
                .take(5)
            {
                message.push_str(&format!("• {}\n", result.test_name));
            }
            if statistics.failed > 5 {
                message.push_str(&format!("• ... and {} more\n", statistics.failed - 5));
            }
        }

        message
    }

    /// Create test data map for notification
    fn create_test_data_map(&self, statistics: &TestSuiteStatistics) -> HashMap<String, String> {
        let mut data = HashMap::new();
        data.insert(
            "total_tests".to_string(),
            statistics.total_tests.to_string(),
        );
        data.insert("passed_tests".to_string(), statistics.passed.to_string());
        data.insert("failed_tests".to_string(), statistics.failed.to_string());
        data.insert("skipped_tests".to_string(), statistics.skipped.to_string());
        data.insert(
            "success_rate".to_string(),
            format!("{:.1}", statistics.success_rate * 100.0),
        );
        data.insert(
            "duration_seconds".to_string(),
            statistics.total_duration.as_secs().to_string(),
        );
        data
    }
}

/// Resolve a commit SHA for a GitHub status check: explicit
/// `notification.data["commit_sha"]` first, then the `GITHUB_SHA`/
/// `CI_COMMIT_SHA` environment variables set by GitHub Actions/GitLab CI.
fn resolve_commit_sha(notification: &IntegrationNotification) -> Option<String> {
    notification
        .data
        .get("commit_sha")
        .cloned()
        .or_else(|| std::env::var("GITHUB_SHA").ok())
        .or_else(|| std::env::var("CI_COMMIT_SHA").ok())
        .filter(|s| !s.is_empty())
}

/// Resolve a PR/issue number for a GitHub PR comment: explicit
/// `notification.data["pr_number"]` first, then the `PR_NUMBER` environment
/// variable, then parsed out of `GITHUB_REF` (`refs/pull/<n>/merge`).
fn resolve_pr_number(notification: &IntegrationNotification) -> Option<String> {
    if let Some(number) = notification.data.get("pr_number") {
        if !number.is_empty() {
            return Some(number.clone());
        }
    }
    if let Ok(number) = std::env::var("PR_NUMBER") {
        if !number.is_empty() {
            return Some(number);
        }
    }
    if let Ok(reference) = std::env::var("GITHUB_REF") {
        let parts: Vec<&str> = reference.split('/').collect();
        if parts.len() >= 3 && parts.first() == Some(&"refs") && parts.get(1) == Some(&"pull") {
            return Some(parts[2].to_string());
        }
    }
    None
}

impl GitHubClient {
    /// Create a new GitHub client
    pub fn new(config: GitHubIntegration) -> Result<Self> {
        let http_client = HttpClient::new("https://api.github.com".to_string())?;
        // GitHub's REST API allows 5000 authenticated requests per hour.
        let rate_limiter = RateLimiter::per_hour(5000);

        Ok(Self {
            config,
            http_client,
            rate_limiter,
            statistics: GitHubStatistics::default(),
        })
    }

    fn auth_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.config.token),
        );
        headers.insert(
            "Accept".to_string(),
            "application/vnd.github+json".to_string(),
        );
        headers.insert("X-GitHub-Api-Version".to_string(), "2022-11-28".to_string());
        headers
    }

    fn check_rate_limit(&mut self) -> Result<()> {
        if !self.rate_limiter.is_allowed() {
            self.statistics.rate_limit_hits += 1;
            return Err(OptimError::ResourceUnavailable(format!(
                "GitHub API rate limit exceeded ({} requests per {:?})",
                self.rate_limiter.limit, self.rate_limiter.window
            )));
        }
        Ok(())
    }

    /// Send notification to GitHub.
    ///
    /// Every GitHub REST call returns a [`DeliveryOutcome`]; a reachable-but-
    /// rejecting API (401/403/404/5xx) surfaces as `Ok(DeliveryOutcome { status:
    /// Failed, .. })` rather than a transport `Err`. This method therefore
    /// inspects each outcome and returns an honest `Err` when the API rejected
    /// the request -- it never reports success for a call GitHub refused.
    pub fn send_notification(&mut self, notification: &IntegrationNotification) -> Result<()> {
        match &notification.notification_type {
            NotificationType::TestCompletion | NotificationType::TestFailure => {
                if self.config.create_status_checks {
                    let outcome = self.create_status_check(notification)?;
                    Self::require_delivered(&outcome, "GitHub status check")?;
                }
                if self.config.create_pr_comments {
                    let outcome = self.create_pr_comment(notification)?;
                    Self::require_delivered(&outcome, "GitHub PR comment")?;
                }
            }
            NotificationType::PerformanceRegression if self.config.create_regression_issues => {
                let outcome = self.create_issue(notification)?;
                Self::require_delivered(&outcome, "GitHub issue")?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Turn a non-success [`DeliveryOutcome`] into an honest `Err`.
    fn require_delivered(outcome: &DeliveryOutcome, what: &str) -> Result<()> {
        if outcome.is_success() {
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "{what} delivery failed: {}",
                outcome.detail()
            )))
        }
    }

    /// Create status check
    fn create_status_check(
        &mut self,
        notification: &IntegrationNotification,
    ) -> Result<DeliveryOutcome> {
        let sha = resolve_commit_sha(notification).ok_or_else(|| {
            OptimError::InvalidConfig(
                "GitHub status check requires a commit SHA: set notification.data[\"commit_sha\"] \
                 or the GITHUB_SHA/CI_COMMIT_SHA environment variable"
                    .to_string(),
            )
        })?;

        let state = match &notification.notification_type {
            NotificationType::TestCompletion => "success",
            NotificationType::TestFailure => "failure",
            _ => "pending",
        };

        let payload = serde_json::json!({
            "state": state,
            "description": &notification.message,
            "context": self.config.status_checks.context
        });

        self.check_rate_limit()?;
        let path = format!(
            "/repos/{}/{}/statuses/{}",
            self.config.owner, self.config.repository, sha
        );
        let outcome = self.http_client.send(
            &path,
            TransportMethod::Post,
            &self.auth_headers(),
            &format!(
                "github-status:{}/{}",
                self.config.owner, self.config.repository
            ),
            &payload.to_string(),
        )?;
        self.statistics.api_requests += 1;
        self.statistics.last_activity = Some(SystemTime::now());
        if outcome.is_success() {
            self.statistics.status_checks_created += 1;
        }
        Ok(outcome)
    }

    /// Create PR comment
    fn create_pr_comment(
        &mut self,
        notification: &IntegrationNotification,
    ) -> Result<DeliveryOutcome> {
        let pr_number = resolve_pr_number(notification).ok_or_else(|| {
            OptimError::InvalidConfig(
                "GitHub PR comment requires a PR number: set notification.data[\"pr_number\"], \
                 the PR_NUMBER environment variable, or run in a GitHub Actions pull_request context"
                    .to_string(),
            )
        })?;

        let comment_body = format!("## {}\n\n{}", notification.title, notification.message);
        let payload = serde_json::json!({ "body": comment_body });

        self.check_rate_limit()?;
        let path = format!(
            "/repos/{}/{}/issues/{}/comments",
            self.config.owner, self.config.repository, pr_number
        );
        let outcome = self.http_client.send(
            &path,
            TransportMethod::Post,
            &self.auth_headers(),
            &format!(
                "github-comment:{}/{}",
                self.config.owner, self.config.repository
            ),
            &payload.to_string(),
        )?;
        self.statistics.api_requests += 1;
        self.statistics.last_activity = Some(SystemTime::now());
        if outcome.is_success() {
            self.statistics.pr_comments_created += 1;
        }
        Ok(outcome)
    }

    /// Create issue
    fn create_issue(&mut self, notification: &IntegrationNotification) -> Result<DeliveryOutcome> {
        let payload = serde_json::json!({
            "title": &notification.title,
            "body": &notification.message,
            "labels": [self.config.labels.performance_regression]
        });

        self.check_rate_limit()?;
        let path = format!(
            "/repos/{}/{}/issues",
            self.config.owner, self.config.repository
        );
        let outcome = self.http_client.send(
            &path,
            TransportMethod::Post,
            &self.auth_headers(),
            &format!(
                "github-issue:{}/{}",
                self.config.owner, self.config.repository
            ),
            &payload.to_string(),
        )?;
        self.statistics.api_requests += 1;
        self.statistics.last_activity = Some(SystemTime::now());
        if outcome.is_success() {
            self.statistics.issues_created += 1;
        }
        Ok(outcome)
    }

    /// Get health status, derived from the most recent delivery outcome
    /// rather than an unconditional constant.
    pub fn get_health_status(&self) -> IntegrationStatus {
        if self.config.token.is_empty() {
            return IntegrationStatus::NotConfigured;
        }
        if self.statistics.last_activity.is_none() {
            return IntegrationStatus::Healthy; // configured, nothing sent yet
        }
        if self.statistics.status_checks_created == 0
            && self.statistics.pr_comments_created == 0
            && self.statistics.issues_created == 0
            && self.statistics.api_requests > 0
        {
            IntegrationStatus::Error // every attempted request failed
        } else {
            IntegrationStatus::Healthy
        }
    }
}

impl SlackClient {
    /// Create a new Slack client
    pub fn new(config: SlackIntegration) -> Result<Self> {
        // The webhook URL *is* the full endpoint (e.g.
        // `https://hooks.slack.com/services/T000/B000/XXXX`); it must be the
        // HttpClient's base URL, not a bare host, or requests would 404.
        let http_client = HttpClient::new(config.webhook_url.clone())?;

        Ok(Self {
            config,
            http_client,
            statistics: SlackStatistics::default(),
        })
    }

    /// Send notification to Slack
    pub fn send_notification(&mut self, notification: &IntegrationNotification) -> Result<()> {
        if self.config.webhook_url.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Slack integration has no webhook_url configured".to_string(),
            ));
        }

        let color = match notification.priority {
            NotificationPriority::Critical => "#FF0000",
            NotificationPriority::High => "#FF8C00",
            NotificationPriority::Normal => "#36A64F",
            NotificationPriority::Low => "#808080",
        };

        let payload = serde_json::json!({
            "channel": self.config.default_channel,
            "username": self.config.username.as_deref().unwrap_or("CI/CD Bot"),
            "icon_emoji": self.config.icon_emoji.as_deref().unwrap_or(":robot_face:"),
            "attachments": [{
                "color": color,
                "title": &notification.title,
                "text": &notification.message,
                "timestamp": notification.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default().as_secs()
            }]
        });

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        let outcome = self.http_client.send(
            "",
            TransportMethod::Post,
            &headers,
            &self.config.default_channel,
            &payload.to_string(),
        )?;

        self.statistics.last_activity = Some(SystemTime::now());
        if !self
            .statistics
            .channels_used
            .contains(&self.config.default_channel)
        {
            self.statistics
                .channels_used
                .push(self.config.default_channel.clone());
        }
        if outcome.is_success() {
            self.statistics.messages_sent += 1;
            Ok(())
        } else {
            self.statistics.messages_failed += 1;
            Err(OptimError::InvalidConfig(format!(
                "Slack delivery failed: {}",
                outcome.detail()
            )))
        }
    }

    /// Get health status, derived from delivery counters.
    pub fn get_health_status(&self) -> IntegrationStatus {
        if self.config.webhook_url.is_empty() {
            return IntegrationStatus::NotConfigured;
        }
        if self.statistics.messages_sent == 0 && self.statistics.messages_failed > 0 {
            IntegrationStatus::Error
        } else if self.statistics.messages_failed > 0 {
            IntegrationStatus::Warning
        } else {
            IntegrationStatus::Healthy
        }
    }
}

impl EmailClient {
    /// Create a new email client
    pub fn new(config: EmailIntegration) -> Result<Self> {
        Ok(Self {
            config,
            transport_kind: notification_transport::transport_kind_from_env(),
            statistics: EmailStatistics::default(),
        })
    }

    /// Send notification via email
    pub fn send_notification(&mut self, notification: &IntegrationNotification) -> Result<()> {
        if self.config.default_recipients.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Email integration has no default_recipients configured".to_string(),
            ));
        }

        let subject = match notification.priority {
            NotificationPriority::Critical => format!("[CRITICAL] {}", notification.title),
            NotificationPriority::High => format!("[HIGH] {}", notification.title),
            _ => notification.title.clone(),
        };

        let body = if self.config.templates.use_html {
            self.create_html_email(&notification.message)
        } else {
            notification.message.clone()
        };

        let message = format!(
            "From: {}\r\nTo: {}\r\nSubject: {}\r\n\r\n{}\r\n",
            self.config.from_email,
            self.config.default_recipients.join(", "),
            subject,
            body
        );

        let smtp_target = SmtpTarget {
            host: self.config.smtp.host.clone(),
            port: self.config.smtp.port,
            use_tls: self.config.smtp.use_tls,
            username: self.config.smtp.username.clone(),
            password: self.config.smtp.password.clone(),
            from: self.config.from_email.clone(),
            to: self.config.default_recipients.clone(),
            timeout: Duration::from_secs(self.config.smtp.timeout_sec),
        };

        let outcome =
            notification_transport::deliver_email(&self.transport_kind, &smtp_target, &message)?;

        self.statistics.last_activity = Some(SystemTime::now());
        for recipient in &self.config.default_recipients {
            if !self.statistics.recipients_contacted.contains(recipient) {
                self.statistics.recipients_contacted.push(recipient.clone());
            }
        }

        if outcome.is_success() {
            self.statistics.emails_sent += 1;
            Ok(())
        } else {
            self.statistics.emails_failed += 1;
            Err(OptimError::InvalidConfig(format!(
                "Email delivery failed: {}",
                outcome.detail()
            )))
        }
    }

    /// Create HTML email body
    fn create_html_email(&self, message: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head><title>CI/CD Notification</title></head>
<body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<h2>CI/CD Automation Notification</h2>
<p>{}</p>
<hr>
<p><small>This is an automated message from the CI/CD system.</small></p>
</div>
</body>
</html>"#,
            message.replace('\n', "<br>")
        )
    }

    /// Get health status, derived from delivery counters.
    pub fn get_health_status(&self) -> IntegrationStatus {
        if self.config.default_recipients.is_empty() {
            return IntegrationStatus::NotConfigured;
        }
        if self.statistics.emails_sent == 0 && self.statistics.emails_failed > 0 {
            IntegrationStatus::Error
        } else if self.statistics.emails_failed > 0 {
            IntegrationStatus::Warning
        } else {
            IntegrationStatus::Healthy
        }
    }
}

impl WebhookClient {
    /// Create a new webhook client
    pub fn new(config: WebhookIntegration) -> Result<Self> {
        let http_client = HttpClient::new(config.url.clone())?;
        let retry_manager = RetryManager::new(
            config.payload.clone().into(),
            notification_transport::transport_kind_from_env(),
        );

        Ok(Self {
            config,
            http_client,
            retry_manager,
            statistics: WebhookStatistics::default(),
        })
    }

    /// Check if webhook should trigger for notification type
    pub fn should_trigger(&self, notification_type: &NotificationType) -> bool {
        match notification_type {
            NotificationType::TestCompletion => self.config.triggers.on_completion,
            NotificationType::TestFailure => self.config.triggers.on_failure,
            NotificationType::PerformanceRegression => self.config.triggers.on_regression,
            NotificationType::PerformanceImprovement => self.config.triggers.on_improvement,
            _ => false,
        }
    }

    /// Send notification via webhook
    pub fn send_notification(&mut self, notification: &IntegrationNotification) -> Result<()> {
        let payload = self.create_webhook_payload(notification)?;
        let method = transport_method_from_http_method(&self.config.method);
        let mut headers = self.config.headers.clone();
        apply_webhook_auth(&mut headers, self.config.auth.as_ref());

        let start = SystemTime::now();
        let send_result = self
            .http_client
            .send("", method, &headers, &self.config.name, &payload);
        let elapsed_ms = SystemTime::now()
            .duration_since(start)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;

        self.statistics.requests_sent += 1;
        self.statistics.average_response_time_ms = if self.statistics.requests_sent <= 1 {
            elapsed_ms
        } else {
            (self.statistics.average_response_time_ms * (self.statistics.requests_sent - 1) as f64
                + elapsed_ms)
                / self.statistics.requests_sent as f64
        };

        match send_result {
            Ok(outcome) if outcome.is_success() => {
                self.statistics.requests_successful += 1;
                Ok(())
            }
            Ok(outcome) => {
                self.statistics.requests_failed += 1;
                self.retry_manager.add_failed_request(
                    RequestData {
                        method: self.config.method.clone(),
                        url: self.config.url.clone(),
                        headers,
                        body: payload,
                    },
                    outcome.detail().to_string(),
                );
                Err(OptimError::InvalidConfig(format!(
                    "webhook '{}' delivery failed: {}",
                    self.config.name,
                    outcome.detail()
                )))
            }
            Err(err) => {
                self.statistics.requests_failed += 1;
                self.retry_manager.add_failed_request(
                    RequestData {
                        method: self.config.method.clone(),
                        url: self.config.url.clone(),
                        headers,
                        body: payload,
                    },
                    err.to_string(),
                );
                Err(err)
            }
        }
    }

    /// Create webhook payload
    fn create_webhook_payload(&self, notification: &IntegrationNotification) -> Result<String> {
        match self.config.payload.format {
            PayloadFormat::JSON => {
                let payload = serde_json::json!({
                    "type": format!("{:?}", notification.notification_type),
                    "title": notification.title,
                    "message": notification.message,
                    "priority": format!("{:?}", notification.priority),
                    "timestamp": notification.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default().as_secs(),
                    "data": notification.data
                });
                Ok(serde_json::to_string(&payload)?)
            }
            PayloadFormat::XML => Ok(format!(
                r#"<?xml version="1.0"?>
<notification>
    <type>{:?}</type>
    <title>{}</title>
    <message>{}</message>
    <priority>{:?}</priority>
    <timestamp>{}</timestamp>
</notification>"#,
                notification.notification_type,
                notification.title,
                notification.message,
                notification.priority,
                notification
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )),
            _ => Ok(notification.message.clone()),
        }
    }

    /// Get health status, derived from delivery counters.
    pub fn get_health_status(&self) -> IntegrationStatus {
        if self.config.url.is_empty() {
            return IntegrationStatus::NotConfigured;
        }
        if self.statistics.requests_sent == 0 {
            IntegrationStatus::Healthy // configured, nothing sent yet
        } else if self.statistics.requests_successful == 0 {
            IntegrationStatus::Error
        } else if self.statistics.requests_failed > 0 {
            IntegrationStatus::Warning
        } else {
            IntegrationStatus::Healthy
        }
    }
}

fn transport_method_from_http_method(method: &HttpMethod) -> TransportMethod {
    match method {
        HttpMethod::GET => TransportMethod::Get,
        HttpMethod::POST => TransportMethod::Post,
        HttpMethod::PUT => TransportMethod::Put,
        HttpMethod::PATCH => TransportMethod::Patch,
        HttpMethod::DELETE => TransportMethod::Delete,
    }
}

fn apply_webhook_auth(headers: &mut HashMap<String, String>, auth: Option<&WebhookAuth>) {
    match auth {
        Some(WebhookAuth::Bearer { token }) => {
            headers.insert("Authorization".to_string(), format!("Bearer {token}"));
        }
        Some(WebhookAuth::Basic { username, password }) => {
            let credentials = format!("{username}:{password}");
            headers.insert(
                "Authorization".to_string(),
                format!("Basic {}", base64_encode(credentials.as_bytes())),
            );
        }
        Some(WebhookAuth::ApiKey { key, header }) => {
            headers.insert(header.clone(), key.clone());
        }
        Some(WebhookAuth::Custom { headers: custom }) => {
            for (k, v) in custom {
                headers.insert(k.clone(), v.clone());
            }
        }
        None => {}
    }
}

/// Minimal RFC 4648 base64 encoder (standard alphabet, with padding) so
/// HTTP Basic auth headers can be built without a base64 crate dependency.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);

        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(b2 & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

impl HttpClient {
    /// Create a new HTTP client
    pub fn new(base_url: String) -> Result<Self> {
        Ok(Self {
            base_url,
            default_headers: HashMap::new(),
            timeout: Duration::from_secs(30),
            user_agent: "optirs-bench-ci-cd-automation/1.0".to_string(),
            transport_kind: notification_transport::transport_kind_from_env(),
        })
    }

    /// Make an HTTP request through the configured transport. `path` is
    /// appended to `base_url` verbatim (pass `""` when `base_url` is already
    /// the full endpoint, as with Slack/generic webhooks).
    pub fn send(
        &self,
        path: &str,
        method: TransportMethod,
        extra_headers: &HashMap<String, String>,
        channel: &str,
        body: &str,
    ) -> Result<DeliveryOutcome> {
        let mut headers = self.default_headers.clone();
        headers.insert("User-Agent".to_string(), self.user_agent.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        for (k, v) in extra_headers {
            headers.insert(k.clone(), v.clone());
        }

        let target = DeliveryTarget {
            url: format!("{}{}", self.base_url, path),
            method,
            headers,
            timeout: self.timeout,
            channel: channel.to_string(),
        };

        notification_transport::deliver(&self.transport_kind, &target, body)
    }
}

impl RateLimiter {
    /// Create a new rate limiter with a per-minute window (the historical
    /// default for this constructor; kept for API/test compatibility).
    pub fn new(requests_per_minute: u32) -> Self {
        Self::with_window(requests_per_minute, Duration::from_secs(60))
    }

    /// Create a rate limiter matching GitHub's per-hour budget.
    pub fn per_hour(requests_per_hour: u32) -> Self {
        Self::with_window(requests_per_hour, Duration::from_secs(3600))
    }

    /// Create a rate limiter with an explicit limit and window.
    pub fn with_window(limit: u32, window: Duration) -> Self {
        Self {
            limit,
            window,
            current_requests: 0,
            reset_time: SystemTime::now() + window,
            request_history: Vec::new(),
        }
    }

    /// Check if request is allowed
    pub fn is_allowed(&mut self) -> bool {
        let now = SystemTime::now();

        if now >= self.reset_time {
            self.current_requests = 0;
            self.reset_time = now + self.window;
            self.request_history.clear();
        }

        if self.current_requests < self.limit {
            self.current_requests += 1;
            self.request_history.push(now);
            true
        } else {
            false
        }
    }
}

impl RetryManager {
    /// Create a new retry manager
    pub fn new(config: WebhookRetryConfig, transport_kind: TransportKind) -> Self {
        Self {
            config,
            transport_kind,
            failed_requests: Vec::new(),
            statistics: RetryStatistics::default(),
        }
    }

    /// Add failed request for retry
    pub fn add_failed_request(&mut self, request_data: RequestData, error: String) {
        let failed_request = FailedRequest {
            id: uuid::Uuid::new_v4().to_string(),
            request_data,
            failed_at: SystemTime::now(),
            retry_attempts: 0,
            last_error: error,
            next_retry_at: SystemTime::now() + Duration::from_secs(self.config.initial_delay_sec),
        };

        self.failed_requests.push(failed_request);
    }

    /// Process the retry queue, re-issuing each due request through the
    /// real transport and branching on the actual outcome. A hard 4xx
    /// failure (other than 429) is treated as permanent and is not retried
    /// further; 429/5xx and transport-level errors are retried with
    /// exponential backoff up to `max_retries`.
    pub fn process_retries(&mut self) -> Result<()> {
        let now = SystemTime::now();
        let mut completed = Vec::new();

        for (index, failed_request) in self.failed_requests.iter_mut().enumerate() {
            if now < failed_request.next_retry_at {
                continue;
            }
            if failed_request.retry_attempts >= self.config.max_retries {
                self.statistics.failed_retries += 1;
                completed.push(index);
                continue;
            }

            failed_request.retry_attempts += 1;
            self.statistics.total_retries += 1;

            let target = DeliveryTarget {
                url: failed_request.request_data.url.clone(),
                method: transport_method_from_http_method(&failed_request.request_data.method),
                headers: failed_request.request_data.headers.clone(),
                timeout: Duration::from_secs(30),
                channel: failed_request.id.clone(),
            };

            let outcome = notification_transport::deliver(
                &self.transport_kind,
                &target,
                &failed_request.request_data.body,
            );

            let delay = (self.config.initial_delay_sec as f64
                * self
                    .config
                    .backoff_multiplier
                    .powi(failed_request.retry_attempts as i32))
            .min(self.config.max_delay_sec as f64) as u64;
            failed_request.next_retry_at = now + Duration::from_secs(delay);

            match outcome {
                Ok(outcome) if outcome.is_success() => {
                    self.statistics.successful_retries += 1;
                    completed.push(index);
                }
                Ok(outcome) => {
                    failed_request.last_error = outcome.detail().to_string();
                    let permanent_failure = matches!(
                        outcome.http_status,
                        Some(code) if (400..500).contains(&code) && code != 429
                    );
                    if permanent_failure || failed_request.retry_attempts >= self.config.max_retries
                    {
                        self.statistics.failed_retries += 1;
                        completed.push(index);
                    }
                }
                Err(err) => {
                    // Transport itself is broken (e.g. curl missing): retrying
                    // immediately cannot help, so stop retrying this request.
                    failed_request.last_error = err.to_string();
                    self.statistics.failed_retries += 1;
                    completed.push(index);
                }
            }
        }

        for &index in completed.iter().rev() {
            self.failed_requests.remove(index);
        }

        Ok(())
    }
}

// Default implementations

impl Default for WebhookStatistics {
    fn default() -> Self {
        Self {
            requests_sent: 0,
            requests_successful: 0,
            requests_failed: 0,
            average_response_time_ms: 0.0,
            retry_stats: RetryStatistics::default(),
        }
    }
}

impl From<WebhookPayloadConfig> for WebhookRetryConfig {
    fn from(_payload_config: WebhookPayloadConfig) -> Self {
        Self {
            max_retries: 3,
            initial_delay_sec: 5,
            backoff_multiplier: 2.0,
            max_delay_sec: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_creation() {
        let notification = IntegrationNotification {
            notification_type: NotificationType::TestCompletion,
            title: "Tests Completed".to_string(),
            message: "All tests passed successfully".to_string(),
            priority: NotificationPriority::Normal,
            data: HashMap::new(),
            timestamp: SystemTime::now(),
        };

        assert_eq!(
            notification.notification_type,
            NotificationType::TestCompletion
        );
        assert_eq!(notification.priority, NotificationPriority::Normal);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(5);

        for _ in 0..5 {
            assert!(limiter.is_allowed());
        }

        assert!(!limiter.is_allowed());
    }

    #[test]
    fn rate_limiter_per_hour_uses_hour_window() {
        let limiter = RateLimiter::per_hour(5000);
        assert_eq!(limiter.limit, 5000);
        assert_eq!(limiter.window, Duration::from_secs(3600));
    }

    #[test]
    fn test_integration_status() {
        assert!(IntegrationStatus::Healthy < IntegrationStatus::Warning);
        assert!(IntegrationStatus::Warning < IntegrationStatus::Error);
    }

    #[test]
    fn test_notification_priority() {
        assert!(NotificationPriority::Low < NotificationPriority::Normal);
        assert!(NotificationPriority::Normal < NotificationPriority::High);
        assert!(NotificationPriority::High < NotificationPriority::Critical);
    }

    fn sample_webhook_config() -> WebhookIntegration {
        WebhookIntegration {
            name: "test".to_string(),
            url: "https://example.invalid/webhook".to_string(),
            method: HttpMethod::POST,
            headers: HashMap::new(),
            auth: None,
            triggers: WebhookTriggerConfig {
                on_completion: true,
                on_regression: false,
                on_failure: false,
                on_improvement: false,
                custom_conditions: Vec::new(),
            },
            payload: WebhookPayloadConfig {
                format: PayloadFormat::JSON,
                include_results: true,
                include_metrics: true,
                include_environment: false,
                custom_template: None,
            },
        }
    }

    #[test]
    fn test_webhook_payload_format() {
        let client = WebhookClient::new(sample_webhook_config());
        assert!(client.is_ok());
    }

    #[test]
    fn disabled_transport_webhook_send_is_explicit_err() {
        std::env::set_var("OPTIRS_NOTIFICATION_TRANSPORT", "disabled");
        let mut client = WebhookClient::new(sample_webhook_config()).expect("client should build");
        let notification = IntegrationNotification {
            notification_type: NotificationType::TestCompletion,
            title: "Test".to_string(),
            message: "Test message".to_string(),
            priority: NotificationPriority::Normal,
            data: HashMap::new(),
            timestamp: SystemTime::now(),
        };
        let result = client.send_notification(&notification);
        std::env::remove_var("OPTIRS_NOTIFICATION_TRANSPORT");
        assert!(
            result.is_err(),
            "disabled transport must never claim success"
        );
        assert_eq!(client.statistics.requests_failed, 1);
    }

    #[test]
    fn file_transport_webhook_send_reports_success_and_updates_statistics() {
        let dir = std::env::temp_dir().join(format!(
            "optirs_bench_webhook_transport_test_{}",
            std::process::id()
        ));
        std::env::set_var(
            "OPTIRS_NOTIFICATION_TRANSPORT",
            format!("file:{}", dir.display()),
        );
        let mut client = WebhookClient::new(sample_webhook_config()).expect("client should build");
        let notification = IntegrationNotification {
            notification_type: NotificationType::TestCompletion,
            title: "Test".to_string(),
            message: "Test message".to_string(),
            priority: NotificationPriority::Normal,
            data: HashMap::new(),
            timestamp: SystemTime::now(),
        };
        let result = client.send_notification(&notification);
        std::env::remove_var("OPTIRS_NOTIFICATION_TRANSPORT");

        assert!(
            result.is_ok(),
            "file transport delivery should succeed: {result:?}"
        );
        assert_eq!(client.statistics.requests_successful, 1);
        assert_eq!(client.statistics.requests_failed, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b"user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
    }
}
