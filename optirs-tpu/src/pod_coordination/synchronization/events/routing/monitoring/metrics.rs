// Routing metrics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Routing analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingAnalytics {
    /// Analytics enabled
    pub enabled: bool,
    /// Metrics collection
    pub metrics: RoutingMetrics,
    /// Reporting configuration
    pub reporting: AnalyticsReporting,
    /// Data retention
    pub retention: DataRetention,
}

impl Default for RoutingAnalytics {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: RoutingMetrics::default(),
            reporting: AnalyticsReporting::default(),
            retention: DataRetention::default(),
        }
    }
}

/// Routing metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetrics {
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Quality metrics
    pub quality: QualityMetrics,
    /// Usage metrics
    pub usage: UsageMetrics,
}

impl Default for RoutingMetrics {
    fn default() -> Self {
        Self {
            performance: PerformanceMetrics::default(),
            quality: QualityMetrics::default(),
            usage: UsageMetrics::default(),
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Response time
    pub response_time: bool,
    /// Throughput
    pub throughput: bool,
    /// Latency
    pub latency: bool,
    /// Error rate
    pub error_rate: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time: true,
            throughput: true,
            latency: true,
            error_rate: true,
        }
    }
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Availability
    pub availability: bool,
    /// Reliability
    pub reliability: bool,
    /// Consistency
    pub consistency: bool,
    /// Accuracy
    pub accuracy: bool,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            availability: true,
            reliability: true,
            consistency: true,
            accuracy: true,
        }
    }
}

/// Usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// Request count
    pub request_count: bool,
    /// Route utilization
    pub route_utilization: bool,
    /// Strategy effectiveness
    pub strategy_effectiveness: bool,
    /// Resource consumption
    pub resource_consumption: bool,
}

impl Default for UsageMetrics {
    fn default() -> Self {
        Self {
            request_count: true,
            route_utilization: true,
            strategy_effectiveness: true,
            resource_consumption: true,
        }
    }
}

/// Analytics reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReporting {
    /// Reporting interval
    pub interval: Duration,
    /// Report formats
    pub formats: Vec<ReportFormat>,
    /// Report destinations
    pub destinations: Vec<String>,
}

impl Default for AnalyticsReporting {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(300),
            formats: vec![ReportFormat::JSON],
            destinations: vec!["logs".to_string()],
        }
    }
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// XML format
    XML,
    /// Custom format
    Custom(String),
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetention {
    /// Retention period
    pub period: Duration,
    /// Storage backend
    pub storage: StorageBackend,
    /// Compression enabled
    pub compression: bool,
}

impl Default for DataRetention {
    fn default() -> Self {
        Self {
            period: Duration::from_secs(86400 * 7), // 7 days
            storage: StorageBackend::Memory,
            compression: true,
        }
    }
}

/// Storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory storage
    Memory,
    /// File system storage
    FileSystem,
    /// Database storage
    Database,
    /// Custom storage
    Custom(String),
}