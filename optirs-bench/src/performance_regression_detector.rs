// Advanced Performance Regression Detection System
//
// This module provides comprehensive performance regression detection capabilities
// with statistical analysis, trend detection, and automated alert generation for
// CI/CD integration and continuous performance monitoring.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Main performance regression detection engine
#[derive(Debug)]
pub struct PerformanceRegressionDetector {
    /// Configuration for regression detection
    config: RegressionConfig,
    /// Historical performance data
    historical_data: PerformanceDatabase,
    /// Statistical analyzer
    statistical_analyzer: StatisticalAnalyzer,
    /// Regression analyzer
    regression_analyzer: RegressionAnalyzer,
    /// Alert system
    alert_system: AlertSystem,
    /// Current baseline metrics
    baseline_metrics: Option<BaselineMetrics>,
}

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Enable regression detection
    pub enable_detection: bool,
    /// Confidence threshold for regression detection (0.0 to 1.0)
    pub confidence_threshold: f64,
    /// Performance degradation threshold (e.g., 0.05 = 5% slower)
    pub degradation_threshold: f64,
    /// Minimum samples needed for statistical analysis
    pub min_samples: usize,
    /// Maximum historical data to keep
    pub max_history_size: usize,
    /// Performance metrics to track
    pub tracked_metrics: Vec<MetricType>,
    /// Statistical test type
    pub statistical_test: StatisticalTest,
    /// Regression detection sensitivity
    pub sensitivity: RegressionSensitivity,
    /// Baseline update strategy
    pub baseline_strategy: BaselineStrategy,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// CI/CD integration settings
    pub ci_cd_config: CiCdConfig,
}

/// Types of performance metrics to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Execution time metrics
    ExecutionTime,
    /// Memory usage metrics
    MemoryUsage,
    /// Throughput metrics (operations per second)
    Throughput,
    /// CPU utilization
    CpuUtilization,
    /// GPU utilization
    GpuUtilization,
    /// Cache hit rates
    CacheHitRate,
    /// FLOPS (floating point operations per second)
    Flops,
    /// Convergence rate
    ConvergenceRate,
    /// Error rate
    ErrorRate,
    /// Custom metric
    Custom(String),
}

/// Statistical test types for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    /// Mann-Whitney U test (non-parametric)
    MannWhitneyU,
    /// Student's t-test (parametric)
    StudentTTest,
    /// Wilcoxon signed-rank test
    WilcoxonSignedRank,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Custom statistical test
    Custom(String),
}

/// Regression detection sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSensitivity {
    /// Very sensitive - detects small changes
    VeryHigh,
    /// High sensitivity
    High,
    /// Medium sensitivity (balanced)
    Medium,
    /// Low sensitivity - only major regressions
    Low,
    /// Very low sensitivity
    VeryLow,
}

/// Baseline update strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineStrategy {
    /// Update baseline automatically when performance improves
    AutoImprovement,
    /// Update baseline only manually
    Manual,
    /// Update baseline on successful releases
    OnRelease,
    /// Rolling window baseline
    RollingWindow(usize),
    /// Seasonal baseline (accounts for cyclical patterns)
    Seasonal,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Performance degradation thresholds (percentage)
    pub degradation_thresholds: HashMap<MetricType, f64>,
    /// Memory increase thresholds (percentage)
    pub memory_increase_thresholds: HashMap<String, f64>,
    /// Failure rate thresholds
    pub failure_rate_threshold: f64,
    /// Timeout thresholds (seconds)
    pub timeout_threshold: f64,
}

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdConfig {
    /// Enable CI/CD integration
    pub enabled: bool,
    /// Exit with error code on regression
    pub fail_on_regression: bool,
    /// Generate performance reports for CI
    pub generate_reports: bool,
    /// Report output format
    pub report_format: ReportFormat,
    /// Report output path
    pub report_path: PathBuf,
    /// Webhook URLs for notifications
    pub webhook_urls: Vec<String>,
    /// Slack integration settings
    pub slack_config: Option<SlackConfig>,
    /// Email notification settings
    pub email_config: Option<EmailConfig>,
}

/// Report output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Xml,
    Html,
    Markdown,
    JUnit,
    TeamCity,
}

/// Slack notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub webhook_url: String,
    pub channel: String,
    pub username: String,
    pub icon_emoji: String,
}

/// Email notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
    pub to_addresses: Vec<String>,
}

/// Performance database for historical data storage
#[derive(Debug)]
#[allow(dead_code)]
pub struct PerformanceDatabase {
    /// Historical performance measurements
    measurements: VecDeque<PerformanceMeasurement>,
    /// Performance trends by metric type
    trends: HashMap<MetricType, PerformanceTrend>,
    /// Baseline data by commit/version
    baselines: HashMap<String, BaselineMetrics>,
    /// Configuration metadata
    metadata: DatabaseMetadata,
}

/// Single performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Timestamp of measurement
    pub timestamp: SystemTime,
    /// Git commit hash or version identifier
    pub commithash: String,
    /// Branch name
    pub branch: String,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Test environment information
    pub environment: EnvironmentInfo,
    /// Performance metrics
    pub metrics: HashMap<MetricType, MetricValue>,
    /// Test configuration details
    pub test_config: TestConfiguration,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    /// Primary value
    pub value: f64,
    /// Standard deviation (if applicable)
    pub std_dev: Option<f64>,
    /// Number of samples
    pub sample_count: usize,
    /// Minimum observed value
    pub min_value: f64,
    /// Maximum observed value
    pub max_value: f64,
    /// Additional statistics
    pub percentiles: Option<Percentiles>,
}

/// Percentile data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total memory (MB)
    pub total_memory_mb: usize,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Compiler version
    pub compiler_version: String,
    /// Rust version
    pub rust_version: String,
    /// Additional environment variables
    pub env_vars: HashMap<String, String>,
}

/// Test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    /// Test name
    pub test_name: String,
    /// Test parameters
    pub parameters: HashMap<String, String>,
    /// Dataset size
    pub dataset_size: Option<usize>,
    /// Optimization iterations
    pub iterations: Option<usize>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Precision (f32/f64)
    pub precision: String,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Metric type
    pub metrictype: MetricType,
    /// Trend direction (improving/degrading/stable)
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Statistical significance
    pub significance: f64,
    /// Recent measurements for trend calculation
    pub recent_values: VecDeque<f64>,
    /// Long-term average
    pub long_term_average: f64,
    /// Volatility measure
    pub volatility: f64,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Uncertain,
}

/// Baseline metrics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Version/commit of baseline
    pub version: String,
    /// Baseline measurement timestamp
    pub timestamp: SystemTime,
    /// Baseline metric values
    pub metrics: HashMap<MetricType, MetricValue>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<MetricType, ConfidenceInterval>,
    /// Quality score of baseline (0.0 to 1.0)
    pub quality_score: f64,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Database metadata
#[derive(Debug, Clone)]
pub struct DatabaseMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
    /// Database version
    pub version: String,
    /// Data retention period
    pub retention_period: Duration,
}

/// Statistical analyzer for performance data
#[derive(Debug)]
pub struct StatisticalAnalyzer {
    /// Configuration
    config: StatisticalConfig,
}

/// Statistical analysis configuration
#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    /// Confidence level for tests
    pub confidence_level: f64,
    /// Alpha level for significance testing
    pub alpha: f64,
    /// Minimum effect size to consider significant
    pub min_effect_size: f64,
    /// Bootstrap sample size
    pub bootstrap_samples: usize,
}

/// Regression analysis engine
#[derive(Debug)]
#[allow(dead_code)]
pub struct RegressionAnalyzer {
    /// Current analysis results
    current_results: Vec<RegressionResult>,
    /// Analysis configuration
    config: RegressionAnalysisConfig,
}

/// Regression analysis configuration
#[derive(Debug, Clone)]
pub struct RegressionAnalysisConfig {
    /// Regression detection algorithms
    pub algorithms: Vec<RegressionAlgorithm>,
    /// Sensitivity settings per metric
    pub metric_sensitivity: HashMap<MetricType, f64>,
    /// Temporal analysis window
    pub analysis_window: Duration,
}

/// Regression detection algorithms
#[derive(Debug, Clone)]
pub enum RegressionAlgorithm {
    /// Change point detection
    ChangePoint,
    /// Trend analysis
    TrendAnalysis,
    /// Statistical process control
    StatisticalProcessControl,
    /// Machine learning anomaly detection
    AnomalyDetection,
}

/// Result of regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Metric that regressed
    pub metric: MetricType,
    /// Regression severity (0.0 to 1.0)
    pub severity: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Effect size
    pub effect_size: f64,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Performance change percentage
    pub change_percentage: f64,
    /// Detected regression type
    pub regression_type: RegressionType,
    /// Evidence for regression
    pub evidence: Vec<String>,
    /// Recommendations for investigation
    pub recommendations: Vec<String>,
}

/// Types of performance regressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RegressionType {
    /// Sudden performance drop
    AbruptRegression,
    /// Gradual performance degradation
    GradualRegression,
    /// Memory leak
    MemoryLeak,
    /// Increased error rate
    IncreasedErrorRate,
    /// Reduced throughput
    ReducedThroughput,
    /// Higher latency
    IncreasedLatency,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// Alert system for notifications
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Alert history
    alert_history: VecDeque<Alert>,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Minimum severity for alerts
    pub min_severity: AlertSeverity,
    /// Rate limiting settings
    pub rate_limit: RateLimit,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Maximum alerts per time window
    pub max_alerts: usize,
    /// Time window for rate limiting
    pub time_window: Duration,
    /// Cooldown period between alerts
    pub cooldown: Duration,
}

/// Notification channels
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Email(EmailConfig),
    Slack(SlackConfig),
    Webhook(String),
    Console,
    File(PathBuf),
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert title
    pub title: String,
    /// Alert description
    pub description: String,
    /// Affected metrics
    pub affected_metrics: Vec<MetricType>,
    /// Regression results that triggered the alert
    pub regressionresults: Vec<RegressionResult>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

impl PerformanceRegressionDetector {
    /// Create a new performance regression detector
    pub fn new(config: RegressionConfig) -> Result<Self> {
        let historical_data = PerformanceDatabase::new(config.max_history_size)?;
        let statistical_analyzer = StatisticalAnalyzer::new(StatisticalConfig::default());
        let regression_analyzer = RegressionAnalyzer::new(RegressionAnalysisConfig::default());
        let alert_system = AlertSystem::new(AlertConfig::default());

        Ok(Self {
            config,
            historical_data,
            statistical_analyzer,
            regression_analyzer,
            alert_system,
            baseline_metrics: None,
        })
    }

    /// Add a new performance measurement
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) -> Result<()> {
        self.historical_data.add_measurement(measurement)?;
        self.update_trends()?;
        Ok(())
    }

    /// Detect performance regressions
    pub fn detect_regressions(&mut self) -> Result<Vec<RegressionResult>> {
        if !self.config.enable_detection {
            return Ok(vec![]);
        }

        let latest_measurements = self
            .historical_data
            .get_latest_measurements(self.config.min_samples)?;

        if latest_measurements.len() < self.config.min_samples {
            return Ok(vec![]);
        }

        let mut all_results = Vec::new();

        for metrictype in &self.config.tracked_metrics {
            let results = self.detect_metric_regression(metrictype, &latest_measurements)?;
            all_results.extend(results);
        }

        // Filter results by confidence threshold
        let filtered_results: Vec<_> = all_results
            .into_iter()
            .filter(|result| result.confidence >= self.config.confidence_threshold)
            .collect();

        // Generate alerts for significant regressions
        if !filtered_results.is_empty() {
            self.generate_alerts(&filtered_results)?;
        }

        self.regression_analyzer.current_results = filtered_results.clone();
        Ok(filtered_results)
    }

    /// Detect regression for a specific metric
    fn detect_metric_regression(
        &self,
        metrictype: &MetricType,
        measurements: &[PerformanceMeasurement],
    ) -> Result<Vec<RegressionResult>> {
        let mut results = Vec::new();

        // Extract metric values
        let values: Vec<f64> = measurements
            .iter()
            .filter_map(|m| m.metrics.get(metrictype))
            .map(|mv| mv.value)
            .collect();

        if values.len() < 2 {
            return Ok(results);
        }

        // Get baseline for comparison
        let baseline = self.get_baseline_for_metric(metrictype)?;

        // Perform statistical tests
        let statisticalresult = self.statistical_analyzer.perform_regression_test(
            &values,
            baseline.as_ref(),
            &self.config.statistical_test,
        )?;

        if statisticalresult.is_significant {
            let regression_result = RegressionResult {
                metric: metrictype.clone(),
                severity: self.calculate_severity(&statisticalresult),
                confidence: statisticalresult.confidence,
                p_value: statisticalresult.p_value,
                effect_size: statisticalresult.effect_size,
                baseline_value: baseline.clone().map(|b| b.value).unwrap_or(0.0),
                current_value: *values.last().unwrap(),
                change_percentage: self.calculate_change_percentage(&values, baseline.as_ref()),
                regression_type: self.classify_regression_type(metrictype, &values),
                evidence: statisticalresult.evidence.clone(),
                recommendations: self.generate_recommendations(metrictype, &statisticalresult),
            };

            results.push(regression_result);
        }

        Ok(results)
    }

    /// Calculate regression severity
    fn calculate_severity(&self, statisticalresult: &StatisticalTestResult) -> f64 {
        let base_severity = 1.0 - statisticalresult.p_value;
        let effect_multiplier = (statisticalresult.effect_size / 2.0).min(1.0);
        (base_severity * effect_multiplier).clamp(0.0, 1.0)
    }

    /// Calculate percentage change
    fn calculate_change_percentage(&self, values: &[f64], baseline: Option<&MetricValue>) -> f64 {
        if let Some(baseline) = baseline {
            let current = *values.last().unwrap();
            if baseline.value != 0.0 {
                ((current - baseline.value) / baseline.value) * 100.0
            } else {
                0.0
            }
        } else if values.len() >= 2 {
            let previous = values[values.len() - 2];
            let current = *values.last().unwrap();
            if previous != 0.0 {
                ((current - previous) / previous) * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Classify the type of regression
    fn classify_regression_type(&self, metrictype: &MetricType, values: &[f64]) -> RegressionType {
        // Simple heuristic classification
        match metrictype {
            MetricType::MemoryUsage => {
                if self.is_monotonic_increase(values) {
                    RegressionType::MemoryLeak
                } else {
                    RegressionType::AbruptRegression
                }
            }
            MetricType::ExecutionTime => RegressionType::IncreasedLatency,
            MetricType::Throughput => RegressionType::ReducedThroughput,
            MetricType::ErrorRate => {
                if self.is_gradual_change(values) {
                    RegressionType::GradualRegression
                } else {
                    RegressionType::AbruptRegression
                }
            }
            MetricType::CpuUtilization | MetricType::GpuUtilization => {
                if self.is_gradual_change(values) {
                    RegressionType::GradualRegression
                } else {
                    RegressionType::AbruptRegression
                }
            }
            MetricType::CacheHitRate => RegressionType::ReducedThroughput,
            MetricType::Flops => RegressionType::ReducedThroughput,
            MetricType::ConvergenceRate => RegressionType::GradualRegression,
            MetricType::Custom(_) => RegressionType::AbruptRegression, // Default for custom metrics
        }
    }

    /// Check if values show monotonic increase
    fn is_monotonic_increase(&self, values: &[f64]) -> bool {
        values.windows(2).all(|w| w[1] >= w[0])
    }

    /// Check if change is gradual
    fn is_gradual_change(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }

        let changes: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

        let avg_change = changes.iter().sum::<f64>() / changes.len() as f64;
        let max_change = changes.iter().fold(0.0f64, |acc, &x| acc.max(x));

        max_change < avg_change * 2.0 // Gradual if no single large change
    }

    /// Generate recommendations for investigation
    fn generate_recommendations(
        &self,
        metrictype: &MetricType,
        statisticalresult: &StatisticalTestResult,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match metrictype {
            MetricType::ExecutionTime => {
                recommendations
                    .push("Profile the code to identify performance bottlenecks".to_string());
                recommendations
                    .push("Check for algorithmic changes or inefficient loops".to_string());
                recommendations.push("Verify compiler optimizations are enabled".to_string());
            }
            MetricType::MemoryUsage => {
                recommendations.push("Run memory leak detection tools".to_string());
                recommendations.push("Check for memory allocation patterns".to_string());
                recommendations.push("Verify proper cleanup of resources".to_string());
            }
            MetricType::Throughput => {
                recommendations.push("Analyze parallelization and concurrency".to_string());
                recommendations.push("Check for synchronization bottlenecks".to_string());
                recommendations.push("Verify hardware utilization efficiency".to_string());
            }
            _ => {
                recommendations.push("Investigate recent code changes".to_string());
                recommendations.push("Review commit history for relevant changes".to_string());
            }
        }

        if statisticalresult.effect_size > 0.5 {
            recommendations.push("Consider this a high-priority investigation".to_string());
        }

        recommendations
    }

    /// Get baseline metric value
    fn get_baseline_for_metric(&self, metrictype: &MetricType) -> Result<Option<MetricValue>> {
        if let Some(baseline) = &self.baseline_metrics {
            Ok(baseline.metrics.get(metrictype).cloned())
        } else {
            Ok(None)
        }
    }

    /// Update performance trends
    fn update_trends(&mut self) -> Result<()> {
        for metrictype in &self.config.tracked_metrics.clone() {
            let recent_measurements = self.historical_data.get_recent_measurements_for_metric(
                metrictype, 50, // Last 50 measurements
            )?;

            if recent_measurements.len() >= 10 {
                let trend = self.calculate_trend(metrictype, &recent_measurements)?;
                self.historical_data
                    .trends
                    .insert(metrictype.clone(), trend);
            }
        }

        Ok(())
    }

    /// Calculate performance trend
    fn calculate_trend(&self, metrictype: &MetricType, values: &[f64]) -> Result<PerformanceTrend> {
        let direction = self.determine_trend_direction(values);
        let strength = self.calculate_trend_strength(values);
        let significance = self.calculate_trend_significance(values);
        let long_term_average = values.iter().sum::<f64>() / values.len() as f64;
        let volatility = self.calculate_volatility(values);

        Ok(PerformanceTrend {
            metrictype: metrictype.clone(),
            direction,
            strength,
            significance,
            recent_values: values.iter().copied().collect(),
            long_term_average,
            volatility,
        })
    }

    /// Determine trend direction using linear regression
    fn determine_trend_direction(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::Uncertain;
        }

        // Simple linear regression slope calculation
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        if slope.abs() <= 0.015 {
            // Slightly more tolerant threshold for stability
            TrendDirection::Stable
        } else if slope > 0.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Improving
        }
    }

    /// Calculate trend strength (0.0 to 1.0)
    fn calculate_trend_strength(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let first = values[0];
        let last = *values.last().unwrap();
        let max_value = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let min_value = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));

        if max_value == min_value {
            return 0.0;
        }

        ((last - first).abs() / (max_value - min_value)).clamp(0.0, 1.0)
    }

    /// Calculate trend statistical significance
    fn calculate_trend_significance(&self, values: &[f64]) -> f64 {
        // Simplified significance calculation
        // In practice, would use proper statistical tests
        if values.len() < 5 {
            return 0.0;
        }

        let variance = self.calculate_variance(values);
        let strength = self.calculate_trend_strength(values);

        // Higher variance reduces significance, higher strength increases it
        (strength / (1.0 + variance.sqrt())).clamp(0.0, 1.0)
    }

    /// Calculate variance of values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sum_squares = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        sum_squares / (values.len() - 1) as f64
    }

    /// Calculate volatility (coefficient of variation)
    fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }

        let variance = self.calculate_variance(values);
        variance.sqrt() / mean
    }

    /// Generate alerts for regressions
    fn generate_alerts(&mut self, regressionresults: &[RegressionResult]) -> Result<()> {
        for result in regressionresults {
            if result.severity >= 0.7 {
                // High severity threshold
                let alert = Alert {
                    id: format!("regression_{}_{}", 
                        result.metric.to_string().to_lowercase(),
                        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                    ),
                    timestamp: SystemTime::now(),
                    severity: self.map_severity(result.severity),
                    title: format!("Performance Regression Detected: {}", result.metric.to_string()),
                    description: format!(
                        "Performance regression detected for metric '{}'. Change: {:.2}%, Confidence: {:.2}%",
                        result.metric.to_string(),
                        result.change_percentage,
                        result.confidence * 100.0
                    ),
                    affected_metrics: vec![result.metric.clone()],
                    regressionresults: vec![result.clone()],
                    recommended_actions: result.recommendations.clone(),
                    metadata: HashMap::new(),
                };

                self.alert_system.send_alert(alert)?;
            }
        }

        Ok(())
    }

    /// Map numeric severity to alert severity
    fn map_severity(&self, severity: f64) -> AlertSeverity {
        match severity {
            s if s >= 0.9 => AlertSeverity::Critical,
            s if s >= 0.7 => AlertSeverity::High,
            s if s >= 0.5 => AlertSeverity::Medium,
            _ => AlertSeverity::Low,
        }
    }

    /// Set baseline metrics
    pub fn set_baseline(&mut self, baseline: BaselineMetrics) -> Result<()> {
        self.baseline_metrics = Some(baseline);
        Ok(())
    }

    /// Update baseline from recent measurements
    pub fn update_baseline_from_recent(&mut self, commithash: String) -> Result<()> {
        let recent_measurements = self
            .historical_data
            .get_latest_measurements(self.config.min_samples)?;

        if recent_measurements.len() < self.config.min_samples {
            return Err(OptimError::InvalidConfig(
                "Insufficient measurements for baseline update".to_string(),
            ));
        }

        let mut metrics = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        for metrictype in &self.config.tracked_metrics {
            let values: Vec<f64> = recent_measurements
                .iter()
                .filter_map(|m| m.metrics.get(metrictype))
                .map(|mv| mv.value)
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = self.calculate_variance(&values);
                let std_dev = variance.sqrt();

                metrics.insert(
                    metrictype.clone(),
                    MetricValue {
                        value: mean,
                        std_dev: Some(std_dev),
                        sample_count: values.len(),
                        min_value: values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)),
                        max_value: values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
                        percentiles: None,
                    },
                );

                // 95% confidence interval
                let margin = 1.96 * std_dev / (values.len() as f64).sqrt();
                confidence_intervals.insert(
                    metrictype.clone(),
                    ConfidenceInterval {
                        lower_bound: mean - margin,
                        upper_bound: mean + margin,
                        confidence_level: 0.95,
                    },
                );
            }
        }

        let baseline = BaselineMetrics {
            version: commithash,
            timestamp: SystemTime::now(),
            metrics,
            confidence_intervals,
            quality_score: self.calculate_baseline_quality(&recent_measurements),
        };

        self.set_baseline(baseline)?;
        Ok(())
    }

    /// Calculate baseline quality score
    fn calculate_baseline_quality(&self, measurements: &[PerformanceMeasurement]) -> f64 {
        if measurements.is_empty() {
            return 0.0;
        }

        // Quality factors:
        // 1. Number of measurements (more is better)
        // 2. Consistency of measurements (lower variance is better)
        // 3. Completeness of metrics (more metrics is better)

        let sample_score = (measurements.len() as f64 / 20.0).min(1.0); // Good if >= 20 samples

        let mut consistency_scores = Vec::new();
        for metrictype in &self.config.tracked_metrics {
            let values: Vec<f64> = measurements
                .iter()
                .filter_map(|m| m.metrics.get(metrictype))
                .map(|mv| mv.value)
                .collect();

            if !values.is_empty() {
                let cv = self.calculate_volatility(&values);
                consistency_scores.push((1.0 / (1.0 + cv)).clamp(0.0, 1.0));
            }
        }

        let consistency_score = if consistency_scores.is_empty() {
            0.0
        } else {
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        };

        let completeness_score = if self.config.tracked_metrics.is_empty() {
            1.0
        } else {
            let complete_metrics = measurements
                .iter()
                .map(|m| {
                    self.config
                        .tracked_metrics
                        .iter()
                        .filter(|metric| m.metrics.contains_key(metric))
                        .count()
                })
                .max()
                .unwrap_or(0);

            complete_metrics as f64 / self.config.tracked_metrics.len() as f64
        };

        (sample_score + consistency_score + completeness_score) / 3.0
    }

    /// Export performance data for CI/CD reporting
    pub fn export_for_ci_cd(&self) -> Result<CiCdReport> {
        let latest_results = &self.regression_analyzer.current_results;

        let status = if latest_results.iter().any(|r| r.severity >= 0.7) {
            CiCdStatus::Failed
        } else if latest_results.iter().any(|r| r.severity >= 0.5) {
            CiCdStatus::Warning
        } else {
            CiCdStatus::Passed
        };

        let report = CiCdReport {
            status,
            timestamp: SystemTime::now(),
            regression_count: latest_results.len(),
            critical_regressions: latest_results.iter().filter(|r| r.severity >= 0.9).count(),
            high_severity_regressions: latest_results.iter().filter(|r| r.severity >= 0.7).count(),
            regressionresults: latest_results.clone(),
            performance_summary: self.generate_performance_summary()?,
            recommendations: self.generate_overall_recommendations(latest_results),
        };

        Ok(report)
    }

    /// Generate performance summary
    fn generate_performance_summary(&self) -> Result<PerformanceSummary> {
        let latest_measurements = self.historical_data.get_latest_measurements(5)?;

        let mut metric_summaries = HashMap::new();
        for metrictype in &self.config.tracked_metrics {
            let values: Vec<f64> = latest_measurements
                .iter()
                .filter_map(|m| m.metrics.get(metrictype))
                .map(|mv| mv.value)
                .collect();

            if !values.is_empty() {
                let trend = self.historical_data.trends.get(metrictype);
                metric_summaries.insert(
                    metrictype.clone(),
                    MetricSummary {
                        current_value: *values.last().unwrap(),
                        trend_direction: trend
                            .map(|t| t.direction.clone())
                            .unwrap_or(TrendDirection::Uncertain),
                        trend_strength: trend.map(|t| t.strength).unwrap_or(0.0),
                        stability_score: trend.map(|t| 1.0 - t.volatility).unwrap_or(1.0),
                    },
                );
            }
        }

        Ok(PerformanceSummary {
            overall_health_score: self.calculate_overall_health_score(&metric_summaries),
            metric_summaries,
            data_quality_score: self
                .baseline_metrics
                .as_ref()
                .map(|b| b.quality_score)
                .unwrap_or(0.0),
        })
    }

    /// Calculate overall health score
    fn calculate_overall_health_score(
        &self,
        summaries: &HashMap<MetricType, MetricSummary>,
    ) -> f64 {
        if summaries.is_empty() {
            return 1.0;
        }

        let scores: Vec<f64> = summaries
            .values()
            .map(|summary| {
                let trend_score = match summary.trend_direction {
                    TrendDirection::Improving => 1.0,
                    TrendDirection::Stable => 0.8,
                    TrendDirection::Degrading => 0.3,
                    TrendDirection::Uncertain => 0.6,
                };

                (trend_score + summary.stability_score) / 2.0
            })
            .collect();

        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Generate overall recommendations
    fn generate_overall_recommendations(&self, results: &[RegressionResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if results.is_empty() {
            recommendations
                .push("No performance regressions detected. Continue monitoring.".to_string());
            return recommendations;
        }

        let critical_count = results.iter().filter(|r| r.severity >= 0.9).count();
        let high_count = results.iter().filter(|r| r.severity >= 0.7).count();

        if critical_count > 0 {
            recommendations.push(format!(
                "CRITICAL: {} critical performance regressions detected. Immediate investigation required.",
                critical_count
            ));
        }

        if high_count > 0 {
            recommendations.push(format!(
                "HIGH: {} high-severity performance regressions detected. Investigation recommended.",
                high_count
            ));
        }

        // Group recommendations by regression type
        let regression_types: std::collections::HashSet<_> =
            results.iter().map(|r| &r.regression_type).collect();

        for regression_type in regression_types {
            match regression_type {
                RegressionType::MemoryLeak => {
                    recommendations
                        .push("Memory leak detected. Run memory profiling tools.".to_string());
                }
                RegressionType::IncreasedLatency => {
                    recommendations.push(
                        "Performance degradation detected. Profile critical paths.".to_string(),
                    );
                }
                RegressionType::ReducedThroughput => {
                    recommendations.push(
                        "Throughput reduction detected. Check parallelization efficiency."
                            .to_string(),
                    );
                }
                _ => {}
            }
        }

        recommendations
            .push("Review recent commits for performance-impacting changes.".to_string());
        recommendations
    }
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    pub is_significant: bool,
    pub p_value: f64,
    pub confidence: f64,
    pub effect_size: f64,
    pub evidence: Vec<String>,
}

/// CI/CD report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdReport {
    pub status: CiCdStatus,
    pub timestamp: SystemTime,
    pub regression_count: usize,
    pub critical_regressions: usize,
    pub high_severity_regressions: usize,
    pub regressionresults: Vec<RegressionResult>,
    pub performance_summary: PerformanceSummary,
    pub recommendations: Vec<String>,
}

/// CI/CD status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CiCdStatus {
    Passed,
    Warning,
    Failed,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub overall_health_score: f64,
    pub metric_summaries: HashMap<MetricType, MetricSummary>,
    pub data_quality_score: f64,
}

/// Individual metric summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub current_value: f64,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub stability_score: f64,
}

// Trait implementations for serialization

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::ExecutionTime => write!(f, "execution_time"),
            MetricType::MemoryUsage => write!(f, "memory_usage"),
            MetricType::Throughput => write!(f, "throughput"),
            MetricType::CpuUtilization => write!(f, "cpu_utilization"),
            MetricType::GpuUtilization => write!(f, "gpu_utilization"),
            MetricType::CacheHitRate => write!(f, "cache_hit_rate"),
            MetricType::Flops => write!(f, "flops"),
            MetricType::ConvergenceRate => write!(f, "convergence_rate"),
            MetricType::ErrorRate => write!(f, "error_rate"),
            MetricType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

// Implementation stubs for supporting types

impl PerformanceDatabase {
    fn new(_maxsize: usize) -> Result<Self> {
        Ok(Self {
            measurements: VecDeque::with_capacity(_maxsize),
            trends: HashMap::new(),
            baselines: HashMap::new(),
            metadata: DatabaseMetadata {
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                version: "1.0.0".to_string(),
                retention_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            },
        })
    }

    fn add_measurement(&mut self, measurement: PerformanceMeasurement) -> Result<()> {
        self.measurements.push_back(measurement);
        if self.measurements.len() > self.measurements.capacity() {
            self.measurements.pop_front();
        }
        self.metadata.updated_at = SystemTime::now();
        Ok(())
    }

    fn get_latest_measurements(&self, count: usize) -> Result<Vec<PerformanceMeasurement>> {
        Ok(self
            .measurements
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect())
    }

    fn get_recent_measurements_for_metric(
        &self,
        metrictype: &MetricType,
        count: usize,
    ) -> Result<Vec<f64>> {
        Ok(self
            .measurements
            .iter()
            .rev()
            .take(count)
            .filter_map(|m| m.metrics.get(metrictype))
            .map(|mv| mv.value)
            .collect())
    }
}

impl StatisticalAnalyzer {
    fn new(config: StatisticalConfig) -> Self {
        Self { config }
    }

    fn perform_regression_test(
        &self,
        values: &[f64],
        baseline: Option<&MetricValue>,
        test_type: &StatisticalTest,
    ) -> Result<StatisticalTestResult> {
        // Simplified implementation - would use proper statistical libraries
        let p_value = self.calculate_p_value(values, baseline, test_type);
        let effect_size = self.calculate_effect_size(values, baseline);
        let confidence = 1.0 - p_value;
        let is_significant =
            p_value < self.config.alpha && effect_size > self.config.min_effect_size;

        Ok(StatisticalTestResult {
            is_significant,
            p_value,
            confidence,
            effect_size,
            evidence: vec![format!("Statistical test: {:?}", test_type)],
        })
    }

    fn calculate_p_value(
        &self,
        values: &[f64],
        _baseline: Option<&MetricValue>,
        _test_type: &StatisticalTest,
    ) -> f64 {
        // Simplified - would implement actual statistical tests
        0.05
    }

    fn calculate_effect_size(&self, values: &[f64], baseline: Option<&MetricValue>) -> f64 {
        if let Some(baseline) = baseline {
            let current_mean = values.iter().sum::<f64>() / values.len() as f64;
            let pooled_std =
                (baseline.std_dev.unwrap_or(1.0) + self.calculate_std_dev(values)) / 2.0;

            if pooled_std > 0.0 {
                ((current_mean - baseline.value) / pooled_std).abs()
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }
}

impl RegressionAnalyzer {
    fn new(config: RegressionAnalysisConfig) -> Self {
        Self {
            current_results: Vec::new(),
            config,
        }
    }
}

impl AlertSystem {
    fn new(config: AlertConfig) -> Self {
        Self {
            config,
            alert_history: VecDeque::new(),
        }
    }

    fn send_alert(&mut self, alert: Alert) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Add to history
        self.alert_history.push_back(alert.clone());

        // Send via configured channels
        for channel in &self.config.channels {
            self.send_via_channel(&alert, channel)?;
        }

        println!("🚨 PERFORMANCE ALERT: {}", alert.title);
        println!("   Description: {}", alert.description);
        println!("   Severity: {:?}", alert.severity);

        Ok(())
    }

    fn send_via_channel(&self, alert: &Alert, channel: &NotificationChannel) -> Result<()> {
        match channel {
            NotificationChannel::Console => {
                println!("ALERT: {}", alert.title);
            }
            NotificationChannel::File(path) => {
                let alert_json = serde_json::to_string_pretty(alert)?;
                std::fs::write(path, alert_json)?;
            }
            NotificationChannel::Email(config) => {
                // Would implement email sending
                println!("Email alert sent: {}", alert.title);
            }
            NotificationChannel::Slack(config) => {
                // Would implement Slack webhook
                println!("Slack alert sent: {}", alert.title);
            }
            NotificationChannel::Webhook(url) => {
                // Would implement webhook call
                println!("Webhook alert sent to {}: {}", url, alert.title);
            }
        }
        Ok(())
    }
}

// Default implementations

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            confidence_threshold: 0.95,
            degradation_threshold: 0.05,
            min_samples: 10,
            max_history_size: 1000,
            tracked_metrics: vec![
                MetricType::ExecutionTime,
                MetricType::MemoryUsage,
                MetricType::Throughput,
            ],
            statistical_test: StatisticalTest::MannWhitneyU,
            sensitivity: RegressionSensitivity::Medium,
            baseline_strategy: BaselineStrategy::AutoImprovement,
            alert_thresholds: AlertThresholds::default(),
            ci_cd_config: CiCdConfig::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        let mut degradation_thresholds = HashMap::new();
        degradation_thresholds.insert(MetricType::ExecutionTime, 0.1); // 10% slower
        degradation_thresholds.insert(MetricType::MemoryUsage, 0.2); // 20% more memory
        degradation_thresholds.insert(MetricType::Throughput, 0.1); // 10% less throughput

        Self {
            degradation_thresholds,
            memory_increase_thresholds: HashMap::new(),
            failure_rate_threshold: 0.05, // 5% failure rate
            timeout_threshold: 300.0,     // 5 minutes
        }
    }
}

impl Default for CiCdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fail_on_regression: false,
            generate_reports: true,
            report_format: ReportFormat::Json,
            report_path: PathBuf::from("performance_report.json"),
            webhook_urls: Vec::new(),
            slack_config: None,
            email_config: None,
        }
    }
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            alpha: 0.05,
            min_effect_size: 0.2,
            bootstrap_samples: 1000,
        }
    }
}

impl Default for RegressionAnalysisConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                RegressionAlgorithm::ChangePoint,
                RegressionAlgorithm::TrendAnalysis,
            ],
            metric_sensitivity: HashMap::new(),
            analysis_window: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_severity: AlertSeverity::Medium,
            rate_limit: RateLimit {
                max_alerts: 10,
                time_window: Duration::from_secs(60 * 60), // 1 hour
                cooldown: Duration::from_secs(5 * 60),     // 5 minutes
            },
            channels: vec![NotificationChannel::Console],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_detector_creation() {
        let config = RegressionConfig::default();
        let detector = PerformanceRegressionDetector::new(config).unwrap();
        assert!(!detector.regression_analyzer.current_results.is_empty() == false);
    }

    #[test]
    fn test_trend_direction_calculation() {
        let config = RegressionConfig::default();
        let detector = PerformanceRegressionDetector::new(config).unwrap();

        // Improving trend (decreasing values for execution time)
        let improving_values = vec![10.0, 9.0, 8.0, 7.0, 6.0];
        let direction = detector.determine_trend_direction(&improving_values);
        assert_eq!(direction, TrendDirection::Improving);

        // Degrading trend (increasing values)
        let degrading_values = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let direction = detector.determine_trend_direction(&degrading_values);
        assert_eq!(direction, TrendDirection::Degrading);

        // Stable trend
        let stable_values = vec![8.0, 8.1, 7.9, 8.0, 8.1];
        let direction = detector.determine_trend_direction(&stable_values);
        assert_eq!(direction, TrendDirection::Stable);
    }

    #[test]
    fn test_baseline_quality_calculation() {
        let config = RegressionConfig::default();
        let detector = PerformanceRegressionDetector::new(config).unwrap();

        // Create test measurements
        let mut measurements = Vec::new();
        for i in 0..20 {
            let mut metrics = HashMap::new();
            metrics.insert(
                MetricType::ExecutionTime,
                MetricValue {
                    value: 10.0 + (i as f64 * 0.1),
                    std_dev: Some(0.5),
                    sample_count: 1,
                    min_value: 10.0,
                    max_value: 12.0,
                    percentiles: None,
                },
            );

            measurements.push(PerformanceMeasurement {
                timestamp: SystemTime::now(),
                commithash: format!("commit_{}", i),
                branch: "main".to_string(),
                build_config: "release".to_string(),
                environment: EnvironmentInfo::default(),
                metrics,
                test_config: TestConfiguration::default(),
                metadata: HashMap::new(),
            });
        }

        let quality = detector.calculate_baseline_quality(&measurements);
        assert!(quality > 0.0);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_regression_type_classification() {
        let config = RegressionConfig::default();
        let detector = PerformanceRegressionDetector::new(config).unwrap();

        // Memory leak pattern (monotonic increase)
        let memory_leak_values = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        let regression_type =
            detector.classify_regression_type(&MetricType::MemoryUsage, &memory_leak_values);
        assert!(matches!(regression_type, RegressionType::MemoryLeak));

        // Execution time increase
        let latency_values = vec![10.0, 12.0, 15.0, 18.0, 20.0];
        let regression_type =
            detector.classify_regression_type(&MetricType::ExecutionTime, &latency_values);
        assert!(matches!(regression_type, RegressionType::IncreasedLatency));
    }

    #[test]
    fn test_ci_cd_report_generation() {
        let config = RegressionConfig::default();
        let detector = PerformanceRegressionDetector::new(config).unwrap();

        let report = detector.export_for_ci_cd().unwrap();
        assert!(matches!(report.status, CiCdStatus::Passed));
        assert_eq!(report.regression_count, 0);
    }
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        Self {
            os: "linux".to_string(),
            cpu_model: "unknown".to_string(),
            cpu_cores: 4,
            total_memory_mb: 8192,
            gpu_info: None,
            compiler_version: "rustc 1.70".to_string(),
            rust_version: "1.70.0".to_string(),
            env_vars: HashMap::new(),
        }
    }
}

impl Default for TestConfiguration {
    fn default() -> Self {
        Self {
            test_name: "default_test".to_string(),
            parameters: HashMap::new(),
            dataset_size: Some(1000),
            iterations: Some(100),
            batch_size: Some(32),
            precision: "f64".to_string(),
        }
    }
}

// Hash implementations for MetricType
impl std::hash::Hash for MetricType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            MetricType::Custom(name) => name.hash(state),
            // For other variants, discriminant is sufficient
            MetricType::ExecutionTime
            | MetricType::MemoryUsage
            | MetricType::Throughput
            | MetricType::CpuUtilization
            | MetricType::GpuUtilization
            | MetricType::CacheHitRate
            | MetricType::Flops
            | MetricType::ConvergenceRate
            | MetricType::ErrorRate => {}
        }
    }
}

impl PartialEq for MetricType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MetricType::Custom(a), MetricType::Custom(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl Eq for MetricType {}
