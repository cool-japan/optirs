// Load balancing algorithms

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancing {
    /// Load balancing algorithms
    pub algorithms: Vec<LoadBalancingAlgorithm>,
    /// Default algorithm
    pub default_algorithm: LoadBalancingAlgorithm,
    /// Algorithm selection criteria
    pub selection_criteria: AlgorithmSelectionCriteria,
    /// Load monitoring
    pub load_monitoring: LoadMonitoring,
}

impl Default for LoadBalancing {
    fn default() -> Self {
        Self {
            algorithms: vec![
                LoadBalancingAlgorithm::RoundRobin,
                LoadBalancingAlgorithm::WeightedRoundRobin,
                LoadBalancingAlgorithm::LeastConnections,
            ],
            default_algorithm: LoadBalancingAlgorithm::RoundRobin,
            selection_criteria: AlgorithmSelectionCriteria::default(),
            load_monitoring: LoadMonitoring::default(),
        }
    }
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin load balancing
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Resource-based
    ResourceBased,
    /// IP hash
    IpHash,
    /// Custom algorithm
    Custom(String),
}

/// Algorithm selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSelectionCriteria {
    /// Selection strategy
    pub strategy: SelectionStrategy,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    /// Adaptation settings
    pub adaptation: AdaptationSettings,
}

impl Default for AlgorithmSelectionCriteria {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::Static,
            thresholds: PerformanceThresholds::default(),
            adaptation: AdaptationSettings::default(),
        }
    }
}

/// Selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Static selection
    Static,
    /// Dynamic selection
    Dynamic,
    /// Adaptive selection
    Adaptive,
    /// Performance-based selection
    PerformanceBased,
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Response time threshold
    pub response_time: Duration,
    /// Throughput threshold
    pub throughput: f64,
    /// Error rate threshold
    pub error_rate: f64,
    /// Resource utilization threshold
    pub resource_utilization: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            response_time: Duration::from_millis(100),
            throughput: 1000.0,
            error_rate: 0.01,
            resource_utilization: 0.8,
        }
    }
}

/// Adaptation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationSettings {
    /// Enable adaptation
    pub enabled: bool,
    /// Adaptation interval
    pub interval: Duration,
    /// Learning rate
    pub learning_rate: f64,
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
}

impl Default for AdaptationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            learning_rate: 0.1,
            strategy: AdaptationStrategy::Gradual,
        }
    }
}

/// Adaptation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Gradual adaptation
    Gradual,
    /// Immediate adaptation
    Immediate,
    /// Threshold-based adaptation
    ThresholdBased,
    /// Custom adaptation
    Custom(String),
}

/// Load monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMonitoring {
    /// Monitoring enabled
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Metrics collection
    pub metrics: LoadMetrics,
    /// Alerting configuration
    pub alerting: LoadAlerting,
}

impl Default for LoadMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            metrics: LoadMetrics::default(),
            alerting: LoadAlerting::default(),
        }
    }
}

/// Load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// Connection count
    pub connection_count: bool,
    /// Request rate
    pub request_rate: bool,
    /// Response time
    pub response_time: bool,
    /// Throughput
    pub throughput: bool,
    /// Error rate
    pub error_rate: bool,
}

impl Default for LoadMetrics {
    fn default() -> Self {
        Self {
            connection_count: true,
            request_rate: true,
            response_time: true,
            throughput: true,
            error_rate: true,
        }
    }
}

/// Load alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAlerting {
    /// Alerting enabled
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
    /// Alert destinations
    pub destinations: Vec<String>,
    /// Alert frequency
    pub frequency: Duration,
}

impl Default for LoadAlerting {
    fn default() -> Self {
        Self {
            enabled: false,
            thresholds: HashMap::new(),
            destinations: Vec::new(),
            frequency: Duration::from_secs(300),
        }
    }
}
