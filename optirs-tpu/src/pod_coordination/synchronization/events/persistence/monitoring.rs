// Persistence Monitoring Configuration
//
// This module provides comprehensive monitoring configurations including performance,
// health, capacity, and error monitoring for persistence systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::backup::NotificationChannel;

/// Persistence monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceMonitoring {
    /// Performance monitoring
    pub performance: PerformanceMonitoring,
    /// Health monitoring
    pub health: HealthMonitoring,
    /// Capacity monitoring
    pub capacity: CapacityMonitoring,
    /// Error monitoring
    pub error_monitoring: ErrorMonitoring,
}

impl Default for PersistenceMonitoring {
    fn default() -> Self {
        Self {
            performance: PerformanceMonitoring::default(),
            health: HealthMonitoring::default(),
            capacity: CapacityMonitoring::default(),
            error_monitoring: ErrorMonitoring::default(),
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoring {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Metrics collection
    pub metrics: Vec<PerformanceMetric>,
    /// Performance alerts
    pub alerts: Vec<PerformanceAlert>,
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            metrics: vec![
                PerformanceMetric::Latency,
                PerformanceMetric::Throughput,
                PerformanceMetric::IOPS,
                PerformanceMetric::ErrorRate,
            ],
            alerts: Vec::new(),
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Read/write latency
    Latency,
    /// Data throughput
    Throughput,
    /// Input/output operations per second
    IOPS,
    /// Error rate
    ErrorRate,
    /// Cache hit rate
    CacheHitRate,
    /// Connection pool utilization
    PoolUtilization,
}

/// Performance alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert name
    pub name: String,
    /// Metric to monitor
    pub metric: PerformanceMetric,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    /// Log message
    Log(String),
    /// Send notification
    Notify(NotificationChannel),
    /// Execute command
    Execute(String),
    /// Scale resources
    Scale(ScaleAction),
}

/// Scale actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleAction {
    /// Scale up
    ScaleUp(f32),
    /// Scale down
    ScaleDown(f32),
    /// Auto-scale
    AutoScale,
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoring {
    /// Enable health monitoring
    pub enabled: bool,
    /// Health check interval
    pub check_interval: Duration,
    /// Health checks
    pub checks: Vec<HealthCheck>,
    /// Health alerts
    pub alerts: Vec<HealthAlert>,
}

impl Default for HealthMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(30),
            checks: vec![
                HealthCheck::StorageConnectivity,
                HealthCheck::DiskSpace,
                HealthCheck::MemoryUsage,
                HealthCheck::ProcessHealth,
            ],
            alerts: Vec::new(),
        }
    }
}

/// Health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheck {
    /// Storage connectivity
    StorageConnectivity,
    /// Disk space availability
    DiskSpace,
    /// Memory usage
    MemoryUsage,
    /// Process health
    ProcessHealth,
    /// Network connectivity
    NetworkConnectivity,
    /// Service dependencies
    ServiceDependencies,
}

/// Health alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    /// Alert name
    pub name: String,
    /// Health check
    pub check: HealthCheck,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Capacity monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityMonitoring {
    /// Enable capacity monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Capacity metrics
    pub metrics: Vec<CapacityMetric>,
    /// Capacity alerts
    pub alerts: Vec<CapacityAlert>,
    /// Forecasting
    pub forecasting: CapacityForecasting,
}

impl Default for CapacityMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(300), // 5 minutes
            metrics: vec![
                CapacityMetric::StorageUtilization,
                CapacityMetric::MemoryUtilization,
                CapacityMetric::ConnectionUtilization,
            ],
            alerts: Vec::new(),
            forecasting: CapacityForecasting::default(),
        }
    }
}

/// Capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapacityMetric {
    /// Storage utilization percentage
    StorageUtilization,
    /// Memory utilization percentage
    MemoryUtilization,
    /// Connection pool utilization
    ConnectionUtilization,
    /// CPU utilization
    CpuUtilization,
    /// Network utilization
    NetworkUtilization,
}

/// Capacity alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityAlert {
    /// Alert name
    pub name: String,
    /// Capacity metric
    pub metric: CapacityMetric,
    /// Warning threshold
    pub warning_threshold: f32,
    /// Critical threshold
    pub critical_threshold: f32,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}

/// Capacity forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityForecasting {
    /// Enable forecasting
    pub enabled: bool,
    /// Forecasting horizon
    pub horizon: Duration,
    /// Forecasting model
    pub model: ForecastingModel,
    /// Forecast accuracy threshold
    pub accuracy_threshold: f32,
}

impl Default for CapacityForecasting {
    fn default() -> Self {
        Self {
            enabled: true,
            horizon: Duration::from_secs(86400 * 30), // 30 days
            model: ForecastingModel::LinearRegression,
            accuracy_threshold: 0.8, // 80%
        }
    }
}

/// Forecasting models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastingModel {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// Machine learning model
    MachineLearning(String),
}

/// Error monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMonitoring {
    /// Enable error monitoring
    pub enabled: bool,
    /// Error tracking
    pub tracking: ErrorTracking,
    /// Error analysis
    pub analysis: ErrorAnalysis,
    /// Error alerts
    pub alerts: Vec<ErrorAlert>,
}

impl Default for ErrorMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            tracking: ErrorTracking::default(),
            analysis: ErrorAnalysis::default(),
            alerts: Vec::new(),
        }
    }
}

/// Error tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTracking {
    /// Track error frequency
    pub frequency: bool,
    /// Track error patterns
    pub patterns: bool,
    /// Track error correlation
    pub correlation: bool,
    /// Error history retention
    pub retention: Duration,
}

impl Default for ErrorTracking {
    fn default() -> Self {
        Self {
            frequency: true,
            patterns: true,
            correlation: true,
            retention: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

/// Error analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Root cause analysis
    pub root_cause_analysis: bool,
    /// Error classification
    pub classification: bool,
    /// Trend analysis
    pub trend_analysis: bool,
    /// Impact analysis
    pub impact_analysis: bool,
}

impl Default for ErrorAnalysis {
    fn default() -> Self {
        Self {
            root_cause_analysis: true,
            classification: true,
            trend_analysis: true,
            impact_analysis: true,
        }
    }
}

/// Error alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAlert {
    /// Alert name
    pub name: String,
    /// Error type pattern
    pub error_pattern: String,
    /// Error rate threshold
    pub rate_threshold: f32,
    /// Time window
    pub time_window: Duration,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}
