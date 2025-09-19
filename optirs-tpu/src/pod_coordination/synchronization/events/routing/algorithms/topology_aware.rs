// Topology-aware routing algorithms

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Topology-aware routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAwareRouting {
    /// Topology analysis
    pub analysis: TopologyAnalysis,
    /// Routing optimization
    pub optimization: RoutingOptimization,
    /// Network awareness
    pub network_awareness: NetworkAwareness,
}

impl Default for TopologyAwareRouting {
    fn default() -> Self {
        Self {
            analysis: TopologyAnalysis::default(),
            optimization: RoutingOptimization::default(),
            network_awareness: NetworkAwareness::default(),
        }
    }
}

/// Topology analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysis {
    /// Analysis methods
    pub methods: Vec<AnalysisMethod>,
    /// Network mapping
    pub network_mapping: NetworkMapping,
    /// Distance calculation
    pub distance_calculation: DistanceCalculation,
}

impl Default for TopologyAnalysis {
    fn default() -> Self {
        Self {
            methods: vec![AnalysisMethod::Static, AnalysisMethod::Dynamic],
            network_mapping: NetworkMapping::default(),
            distance_calculation: DistanceCalculation::default(),
        }
    }
}

/// Analysis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisMethod {
    /// Static analysis
    Static,
    /// Dynamic analysis
    Dynamic,
    /// Hybrid analysis
    Hybrid,
    /// Custom analysis
    Custom(String),
}

/// Network mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMapping {
    /// Mapping strategy
    pub strategy: MappingStrategy,
    /// Update interval
    pub update_interval: std::time::Duration,
    /// Mapping cache
    pub cache_size: usize,
}

impl Default for NetworkMapping {
    fn default() -> Self {
        Self {
            strategy: MappingStrategy::Hierarchical,
            update_interval: std::time::Duration::from_secs(300),
            cache_size: 1000,
        }
    }
}

/// Mapping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingStrategy {
    /// Flat mapping
    Flat,
    /// Hierarchical mapping
    Hierarchical,
    /// Geographic mapping
    Geographic,
    /// Custom mapping
    Custom(String),
}

/// Distance calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceCalculation {
    /// Calculation method
    pub method: DistanceMethod,
    /// Weight factors
    pub factors: DistanceFactors,
}

impl Default for DistanceCalculation {
    fn default() -> Self {
        Self {
            method: DistanceMethod::NetworkLatency,
            factors: DistanceFactors::default(),
        }
    }
}

/// Distance calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMethod {
    /// Network latency
    NetworkLatency,
    /// Geographic distance
    GeographicDistance,
    /// Hop count
    HopCount,
    /// Combined distance
    Combined,
}

/// Distance factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceFactors {
    /// Latency weight
    pub latency_weight: f64,
    /// Geographic weight
    pub geographic_weight: f64,
    /// Hop count weight
    pub hop_count_weight: f64,
    /// Custom weights
    pub custom_weights: HashMap<String, f64>,
}

impl Default for DistanceFactors {
    fn default() -> Self {
        Self {
            latency_weight: 0.5,
            geographic_weight: 0.3,
            hop_count_weight: 0.2,
            custom_weights: HashMap::new(),
        }
    }
}

/// Routing optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOptimization {
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Optimization algorithms
    pub algorithms: Vec<OptimizationAlgorithm>,
    /// Constraints
    pub constraints: OptimizationConstraints,
}

impl Default for RoutingOptimization {
    fn default() -> Self {
        Self {
            objectives: vec![
                OptimizationObjective::MinimizeLatency,
                OptimizationObjective::MaximizeThroughput,
            ],
            algorithms: vec![OptimizationAlgorithm::Dijkstra],
            constraints: OptimizationConstraints::default(),
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize latency
    MinimizeLatency,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize cost
    MinimizeCost,
    /// Load balancing
    LoadBalancing,
    /// Custom objective
    Custom(String),
}

/// Optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Dijkstra's algorithm
    Dijkstra,
    /// A* algorithm
    AStar,
    /// Genetic algorithm
    Genetic,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Custom algorithm
    Custom(String),
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    /// Maximum latency
    pub max_latency: Option<std::time::Duration>,
    /// Minimum throughput
    pub min_throughput: Option<f64>,
    /// Maximum cost
    pub max_cost: Option<f64>,
    /// Custom constraints
    pub custom_constraints: HashMap<String, f64>,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            max_latency: Some(std::time::Duration::from_millis(100)),
            min_throughput: Some(1000.0),
            max_cost: None,
            custom_constraints: HashMap::new(),
        }
    }
}

/// Network awareness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAwareness {
    /// Network conditions monitoring
    pub monitoring: NetworkMonitoring,
    /// Adaptive routing
    pub adaptive_routing: AdaptiveRouting,
    /// QoS awareness
    pub qos_awareness: QoSAwareness,
}

impl Default for NetworkAwareness {
    fn default() -> Self {
        Self {
            monitoring: NetworkMonitoring::default(),
            adaptive_routing: AdaptiveRouting::default(),
            qos_awareness: QoSAwareness::default(),
        }
    }
}

/// Network monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMonitoring {
    /// Monitoring interval
    pub interval: std::time::Duration,
    /// Monitored metrics
    pub metrics: Vec<NetworkMetric>,
    /// Alerting thresholds
    pub thresholds: HashMap<String, f64>,
}

impl Default for NetworkMonitoring {
    fn default() -> Self {
        Self {
            interval: std::time::Duration::from_secs(30),
            metrics: vec![
                NetworkMetric::Latency,
                NetworkMetric::Bandwidth,
                NetworkMetric::PacketLoss,
            ],
            thresholds: HashMap::new(),
        }
    }
}

/// Network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMetric {
    /// Network latency
    Latency,
    /// Available bandwidth
    Bandwidth,
    /// Packet loss rate
    PacketLoss,
    /// Jitter
    Jitter,
    /// Custom metric
    Custom(String),
}

/// Adaptive routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRouting {
    /// Adaptation triggers
    pub triggers: Vec<AdaptationTrigger>,
    /// Adaptation strategies
    pub strategies: Vec<AdaptationStrategy>,
    /// Learning parameters
    pub learning: LearningParameters,
}

impl Default for AdaptiveRouting {
    fn default() -> Self {
        Self {
            triggers: vec![
                AdaptationTrigger::PerformanceDegradation,
                AdaptationTrigger::NetworkChange,
            ],
            strategies: vec![AdaptationStrategy::GradualAdjustment],
            learning: LearningParameters::default(),
        }
    }
}

/// Adaptation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationTrigger {
    /// Performance degradation
    PerformanceDegradation,
    /// Network topology change
    NetworkChange,
    /// Load imbalance
    LoadImbalance,
    /// Custom trigger
    Custom(String),
}

/// Adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Gradual adjustment
    GradualAdjustment,
    /// Immediate switch
    ImmediateSwitch,
    /// Weighted transition
    WeightedTransition,
    /// Custom strategy
    Custom(String),
}

/// Learning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParameters {
    /// Learning rate
    pub learning_rate: f64,
    /// Memory window
    pub memory_window: std::time::Duration,
    /// Exploration rate
    pub exploration_rate: f64,
}

impl Default for LearningParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            memory_window: std::time::Duration::from_secs(3600),
            exploration_rate: 0.05,
        }
    }
}

/// QoS awareness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSAwareness {
    /// QoS requirements
    pub requirements: QoSRequirements,
    /// QoS monitoring
    pub monitoring: QoSMonitoring,
    /// QoS enforcement
    pub enforcement: QoSEnforcement,
}

impl Default for QoSAwareness {
    fn default() -> Self {
        Self {
            requirements: QoSRequirements::default(),
            monitoring: QoSMonitoring::default(),
            enforcement: QoSEnforcement::default(),
        }
    }
}

/// QoS requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSRequirements {
    /// Latency requirements
    pub latency: Option<std::time::Duration>,
    /// Bandwidth requirements
    pub bandwidth: Option<f64>,
    /// Reliability requirements
    pub reliability: Option<f64>,
    /// Custom requirements
    pub custom: HashMap<String, f64>,
}

impl Default for QoSRequirements {
    fn default() -> Self {
        Self {
            latency: Some(std::time::Duration::from_millis(50)),
            bandwidth: Some(1000.0),
            reliability: Some(0.99),
            custom: HashMap::new(),
        }
    }
}

/// QoS monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSMonitoring {
    /// Monitoring enabled
    pub enabled: bool,
    /// Monitoring interval
    pub interval: std::time::Duration,
    /// SLA tracking
    pub sla_tracking: bool,
}

impl Default for QoSMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: std::time::Duration::from_secs(30),
            sla_tracking: true,
        }
    }
}

/// QoS enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSEnforcement {
    /// Enforcement enabled
    pub enabled: bool,
    /// Enforcement actions
    pub actions: Vec<EnforcementAction>,
    /// Violation handling
    pub violation_handling: ViolationHandling,
}

impl Default for QoSEnforcement {
    fn default() -> Self {
        Self {
            enabled: false,
            actions: vec![EnforcementAction::Reroute],
            violation_handling: ViolationHandling::default(),
        }
    }
}

/// Enforcement actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementAction {
    /// Reroute traffic
    Reroute,
    /// Throttle traffic
    Throttle,
    /// Drop traffic
    Drop,
    /// Alert only
    Alert,
    /// Custom action
    Custom(String),
}

/// Violation handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationHandling {
    /// Violation threshold
    pub threshold: f64,
    /// Grace period
    pub grace_period: std::time::Duration,
    /// Escalation policy
    pub escalation: Vec<String>,
}

impl Default for ViolationHandling {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            grace_period: std::time::Duration::from_secs(60),
            escalation: vec!["alert".to_string(), "reroute".to_string()],
        }
    }
}
