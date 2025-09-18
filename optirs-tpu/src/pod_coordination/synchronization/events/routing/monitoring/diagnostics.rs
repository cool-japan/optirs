// Routing diagnostics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Traffic management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficManagement {
    /// Traffic shaping
    pub shaping: TrafficShaping,
    /// Load distribution
    pub distribution: LoadDistribution,
    /// Priority handling
    pub priority: PriorityHandling,
    /// Flow control
    pub flow_control: FlowControl,
}

impl Default for TrafficManagement {
    fn default() -> Self {
        Self {
            shaping: TrafficShaping::default(),
            distribution: LoadDistribution::default(),
            priority: PriorityHandling::default(),
            flow_control: FlowControl::default(),
        }
    }
}

/// Traffic shaping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficShaping {
    /// Shaping enabled
    pub enabled: bool,
    /// Rate limits
    pub rate_limits: HashMap<String, f64>,
    /// Burst limits
    pub burst_limits: HashMap<String, u64>,
    /// Shaping algorithm
    pub algorithm: ShapingAlgorithm,
}

impl Default for TrafficShaping {
    fn default() -> Self {
        Self {
            enabled: false,
            rate_limits: HashMap::new(),
            burst_limits: HashMap::new(),
            algorithm: ShapingAlgorithm::TokenBucket,
        }
    }
}

/// Shaping algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapingAlgorithm {
    /// Token bucket
    TokenBucket,
    /// Leaky bucket
    LeakyBucket,
    /// Fixed window
    FixedWindow,
    /// Custom algorithm
    Custom(String),
}

/// Load distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadDistribution {
    /// Distribution strategy
    pub strategy: DistributionStrategy,
    /// Weight assignments
    pub weights: HashMap<String, f64>,
    /// Auto-scaling
    pub auto_scaling: AutoScaling,
}

impl Default for LoadDistribution {
    fn default() -> Self {
        Self {
            strategy: DistributionStrategy::EvenDistribution,
            weights: HashMap::new(),
            auto_scaling: AutoScaling::default(),
        }
    }
}

/// Distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionStrategy {
    /// Even distribution
    EvenDistribution,
    /// Weighted distribution
    WeightedDistribution,
    /// Performance-based distribution
    PerformanceBased,
    /// Custom distribution
    Custom(String),
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScaling {
    /// Auto-scaling enabled
    pub enabled: bool,
    /// Scale-up threshold
    pub scale_up_threshold: f64,
    /// Scale-down threshold
    pub scale_down_threshold: f64,
    /// Scaling policies
    pub policies: Vec<ScalingPolicy>,
}

impl Default for AutoScaling {
    fn default() -> Self {
        Self {
            enabled: false,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            policies: vec![],
        }
    }
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Policy name
    pub name: String,
    /// Trigger condition
    pub trigger: String,
    /// Scaling action
    pub action: ScalingAction,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up
    ScaleUp { count: u32 },
    /// Scale down
    ScaleDown { count: u32 },
    /// Auto-adjust
    AutoAdjust,
    /// Custom action
    Custom(String),
}

/// Priority handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityHandling {
    /// Priority enabled
    pub enabled: bool,
    /// Priority levels
    pub levels: u32,
    /// Queue management
    pub queue_management: QueueManagement,
    /// Preemption policy
    pub preemption: PreemptionPolicy,
}

impl Default for PriorityHandling {
    fn default() -> Self {
        Self {
            enabled: false,
            levels: 5,
            queue_management: QueueManagement::default(),
            preemption: PreemptionPolicy::NoPreemption,
        }
    }
}

/// Queue management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueManagement {
    /// Queue strategy
    pub strategy: QueueStrategy,
    /// Queue limits
    pub limits: HashMap<String, u32>,
    /// Overflow handling
    pub overflow: OverflowHandling,
}

impl Default for QueueManagement {
    fn default() -> Self {
        Self {
            strategy: QueueStrategy::FIFO,
            limits: HashMap::new(),
            overflow: OverflowHandling::Drop,
        }
    }
}

/// Queue strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueStrategy {
    /// First In First Out
    FIFO,
    /// Last In First Out
    LIFO,
    /// Priority queue
    Priority,
    /// Round robin
    RoundRobin,
    /// Custom strategy
    Custom(String),
}

/// Overflow handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowHandling {
    /// Drop requests
    Drop,
    /// Block requests
    Block,
    /// Redirect requests
    Redirect,
    /// Custom handling
    Custom(String),
}

/// Preemption policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// No preemption
    NoPreemption,
    /// Priority-based preemption
    PriorityBased,
    /// Age-based preemption
    AgeBased,
    /// Custom preemption
    Custom(String),
}

/// Flow control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControl {
    /// Flow control enabled
    pub enabled: bool,
    /// Control algorithm
    pub algorithm: FlowControlAlgorithm,
    /// Window size
    pub window_size: u32,
    /// Congestion control
    pub congestion_control: CongestionControl,
}

impl Default for FlowControl {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: FlowControlAlgorithm::SlidingWindow,
            window_size: 1000,
            congestion_control: CongestionControl::default(),
        }
    }
}

/// Flow control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowControlAlgorithm {
    /// Stop-and-wait
    StopAndWait,
    /// Sliding window
    SlidingWindow,
    /// Credit-based
    CreditBased,
    /// Custom algorithm
    Custom(String),
}

/// Congestion control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControl {
    /// Detection method
    pub detection: CongestionDetection,
    /// Avoidance strategy
    pub avoidance: CongestionAvoidance,
    /// Recovery mechanism
    pub recovery: CongestionRecovery,
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self {
            detection: CongestionDetection::default(),
            avoidance: CongestionAvoidance::default(),
            recovery: CongestionRecovery::default(),
        }
    }
}

/// Congestion detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionDetection {
    /// Detection enabled
    pub enabled: bool,
    /// Detection thresholds
    pub thresholds: HashMap<String, f64>,
    /// Detection window
    pub window: Duration,
}

impl Default for CongestionDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: HashMap::new(),
            window: Duration::from_secs(30),
        }
    }
}

/// Congestion avoidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionAvoidance {
    /// Avoidance strategy
    pub strategy: AvoidanceStrategy,
    /// Rate adjustment
    pub rate_adjustment: f64,
    /// Window adjustment
    pub window_adjustment: u32,
}

impl Default for CongestionAvoidance {
    fn default() -> Self {
        Self {
            strategy: AvoidanceStrategy::BackPressure,
            rate_adjustment: 0.5,
            window_adjustment: 100,
        }
    }
}

/// Avoidance strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AvoidanceStrategy {
    /// Back pressure
    BackPressure,
    /// Rate limiting
    RateLimiting,
    /// Load shedding
    LoadShedding,
    /// Circuit breaker
    CircuitBreaker,
    /// Custom strategy
    Custom(String),
}

/// Congestion recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionRecovery {
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// Recovery timeout
    pub timeout: Duration,
    /// Gradual recovery
    pub gradual: bool,
}

impl Default for CongestionRecovery {
    fn default() -> Self {
        Self {
            strategy: RecoveryStrategy::GradualIncrease,
            timeout: Duration::from_secs(60),
            gradual: true,
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Gradual increase
    GradualIncrease,
    /// Immediate recovery
    ImmediateRecovery,
    /// Exponential backoff
    ExponentialBackoff,
    /// Custom recovery
    Custom(String),
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoring {
    /// Health checks enabled
    pub enabled: bool,
    /// Check interval
    pub interval: Duration,
    /// Health metrics
    pub metrics: HealthMetrics,
    /// Alerting configuration
    pub alerting: HealthAlerting,
}

impl Default for HealthMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            metrics: HealthMetrics::default(),
            alerting: HealthAlerting::default(),
        }
    }
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Endpoint health
    pub endpoint_health: bool,
    /// Route health
    pub route_health: bool,
    /// System health
    pub system_health: bool,
    /// Performance health
    pub performance_health: bool,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            endpoint_health: true,
            route_health: true,
            system_health: true,
            performance_health: true,
        }
    }
}

/// Health alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlerting {
    /// Alerting enabled
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
    /// Alert destinations
    pub destinations: Vec<String>,
    /// Alert frequency
    pub frequency: Duration,
}

impl Default for HealthAlerting {
    fn default() -> Self {
        Self {
            enabled: false,
            thresholds: HashMap::new(),
            destinations: vec![],
            frequency: Duration::from_secs(300),
        }
    }
}