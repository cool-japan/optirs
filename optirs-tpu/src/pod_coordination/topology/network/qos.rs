// Quality of Service Settings and Traffic Classes
//
// This module handles QoS configuration, traffic classification,
// and service level agreements for TPU pod networks.

use std::collections::HashMap;
use std::time::Duration;

/// Network Quality of Service settings
#[derive(Debug, Clone)]
pub struct NetworkQoSSettings {
    /// Enable QoS
    pub enabled: bool,
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
    /// QoS policies
    pub policies: Vec<QoSPolicy>,
    /// Service levels
    pub service_levels: Vec<ServiceLevel>,
    /// Bandwidth allocation
    pub bandwidth_allocation: BandwidthAllocation,
}

impl Default for NetworkQoSSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            traffic_classes: vec![
                TrafficClass::control(),
                TrafficClass::data(),
                TrafficClass::best_effort(),
            ],
            policies: vec![QoSPolicy::default()],
            service_levels: vec![ServiceLevel::default()],
            bandwidth_allocation: BandwidthAllocation::default(),
        }
    }
}

impl NetworkQoSSettings {
    /// High-performance QoS configuration
    pub fn high_performance() -> Self {
        Self {
            enabled: true,
            traffic_classes: vec![
                TrafficClass::high_priority(),
                TrafficClass::medium_priority(),
                TrafficClass::low_priority(),
            ],
            policies: vec![QoSPolicy::high_performance()],
            service_levels: vec![ServiceLevel::premium()],
            bandwidth_allocation: BandwidthAllocation::high_performance(),
        }
    }

    /// Low-latency QoS configuration
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            traffic_classes: vec![TrafficClass::ultra_low_latency()],
            policies: vec![QoSPolicy::low_latency()],
            service_levels: vec![ServiceLevel::real_time()],
            bandwidth_allocation: BandwidthAllocation::low_latency(),
        }
    }

    /// High-bandwidth QoS configuration
    pub fn high_bandwidth() -> Self {
        Self {
            enabled: true,
            traffic_classes: vec![TrafficClass::high_bandwidth()],
            policies: vec![QoSPolicy::high_bandwidth()],
            service_levels: vec![ServiceLevel::bulk_transfer()],
            bandwidth_allocation: BandwidthAllocation::high_bandwidth(),
        }
    }
}

/// Traffic class configuration
#[derive(Debug, Clone)]
pub struct TrafficClass {
    /// Class name
    pub name: String,
    /// Traffic priority
    pub priority: TrafficPriority,
    /// Bandwidth guarantee (percentage)
    pub bandwidth_guarantee: f64,
    /// Maximum bandwidth (percentage)
    pub max_bandwidth: f64,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
    /// Jitter requirements
    pub jitter_requirements: JitterRequirements,
    /// Packet loss tolerance
    pub packet_loss_tolerance: f64,
}

impl TrafficClass {
    /// Control traffic class
    pub fn control() -> Self {
        Self {
            name: "control".to_string(),
            priority: TrafficPriority::Critical,
            bandwidth_guarantee: 5.0,
            max_bandwidth: 10.0,
            latency_requirements: LatencyRequirements::strict(),
            jitter_requirements: JitterRequirements::strict(),
            packet_loss_tolerance: 0.001,
        }
    }

    /// Data traffic class
    pub fn data() -> Self {
        Self {
            name: "data".to_string(),
            priority: TrafficPriority::High,
            bandwidth_guarantee: 70.0,
            max_bandwidth: 90.0,
            latency_requirements: LatencyRequirements::moderate(),
            jitter_requirements: JitterRequirements::moderate(),
            packet_loss_tolerance: 0.01,
        }
    }

    /// Best effort traffic class
    pub fn best_effort() -> Self {
        Self {
            name: "best_effort".to_string(),
            priority: TrafficPriority::BestEffort,
            bandwidth_guarantee: 10.0,
            max_bandwidth: 100.0,
            latency_requirements: LatencyRequirements::relaxed(),
            jitter_requirements: JitterRequirements::relaxed(),
            packet_loss_tolerance: 0.1,
        }
    }

    /// High priority traffic class
    pub fn high_priority() -> Self {
        Self {
            name: "high_priority".to_string(),
            priority: TrafficPriority::RealTimeCritical,
            bandwidth_guarantee: 30.0,
            max_bandwidth: 50.0,
            latency_requirements: LatencyRequirements::ultra_strict(),
            jitter_requirements: JitterRequirements::ultra_strict(),
            packet_loss_tolerance: 0.0001,
        }
    }

    /// Medium priority traffic class
    pub fn medium_priority() -> Self {
        Self {
            name: "medium_priority".to_string(),
            priority: TrafficPriority::High,
            bandwidth_guarantee: 40.0,
            max_bandwidth: 70.0,
            latency_requirements: LatencyRequirements::strict(),
            jitter_requirements: JitterRequirements::strict(),
            packet_loss_tolerance: 0.001,
        }
    }

    /// Low priority traffic class
    pub fn low_priority() -> Self {
        Self {
            name: "low_priority".to_string(),
            priority: TrafficPriority::Low,
            bandwidth_guarantee: 10.0,
            max_bandwidth: 30.0,
            latency_requirements: LatencyRequirements::relaxed(),
            jitter_requirements: JitterRequirements::relaxed(),
            packet_loss_tolerance: 0.05,
        }
    }

    /// Ultra low latency traffic class
    pub fn ultra_low_latency() -> Self {
        Self {
            name: "ultra_low_latency".to_string(),
            priority: TrafficPriority::RealTimeCritical,
            bandwidth_guarantee: 50.0,
            max_bandwidth: 100.0,
            latency_requirements: LatencyRequirements::ultra_strict(),
            jitter_requirements: JitterRequirements::ultra_strict(),
            packet_loss_tolerance: 0.0,
        }
    }

    /// High bandwidth traffic class
    pub fn high_bandwidth() -> Self {
        Self {
            name: "high_bandwidth".to_string(),
            priority: TrafficPriority::Medium,
            bandwidth_guarantee: 80.0,
            max_bandwidth: 100.0,
            latency_requirements: LatencyRequirements::moderate(),
            jitter_requirements: JitterRequirements::moderate(),
            packet_loss_tolerance: 0.01,
        }
    }
}

/// Traffic priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrafficPriority {
    /// Best effort traffic (lowest priority)
    BestEffort = 0,
    /// Low priority traffic
    Low = 1,
    /// Medium priority traffic
    Medium = 2,
    /// High priority traffic
    High = 3,
    /// Critical priority traffic
    Critical = 4,
    /// Real-time traffic
    RealTime = 5,
    /// Real-time critical traffic
    RealTimeCritical = 6,
    /// Network control traffic (highest priority)
    NetworkControl = 7,
}

/// Latency requirements
#[derive(Debug, Clone)]
pub struct LatencyRequirements {
    /// Maximum latency
    pub max_latency: Duration,
    /// Target latency
    pub target_latency: Duration,
    /// Latency variation tolerance
    pub variation_tolerance: Duration,
}

impl LatencyRequirements {
    /// Ultra strict latency requirements
    pub fn ultra_strict() -> Self {
        Self {
            max_latency: Duration::from_micros(100),
            target_latency: Duration::from_micros(50),
            variation_tolerance: Duration::from_micros(10),
        }
    }

    /// Strict latency requirements
    pub fn strict() -> Self {
        Self {
            max_latency: Duration::from_millis(1),
            target_latency: Duration::from_micros(500),
            variation_tolerance: Duration::from_micros(100),
        }
    }

    /// Moderate latency requirements
    pub fn moderate() -> Self {
        Self {
            max_latency: Duration::from_millis(10),
            target_latency: Duration::from_millis(5),
            variation_tolerance: Duration::from_millis(1),
        }
    }

    /// Relaxed latency requirements
    pub fn relaxed() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            target_latency: Duration::from_millis(50),
            variation_tolerance: Duration::from_millis(10),
        }
    }
}

/// Jitter requirements
#[derive(Debug, Clone)]
pub struct JitterRequirements {
    /// Maximum jitter
    pub max_jitter: Duration,
    /// Target jitter
    pub target_jitter: Duration,
    /// Jitter buffer size
    pub buffer_size: Duration,
}

impl JitterRequirements {
    /// Ultra strict jitter requirements
    pub fn ultra_strict() -> Self {
        Self {
            max_jitter: Duration::from_micros(10),
            target_jitter: Duration::from_micros(1),
            buffer_size: Duration::from_micros(50),
        }
    }

    /// Strict jitter requirements
    pub fn strict() -> Self {
        Self {
            max_jitter: Duration::from_micros(100),
            target_jitter: Duration::from_micros(10),
            buffer_size: Duration::from_millis(1),
        }
    }

    /// Moderate jitter requirements
    pub fn moderate() -> Self {
        Self {
            max_jitter: Duration::from_millis(1),
            target_jitter: Duration::from_micros(100),
            buffer_size: Duration::from_millis(5),
        }
    }

    /// Relaxed jitter requirements
    pub fn relaxed() -> Self {
        Self {
            max_jitter: Duration::from_millis(10),
            target_jitter: Duration::from_millis(1),
            buffer_size: Duration::from_millis(50),
        }
    }
}

/// QoS policy configuration
#[derive(Debug, Clone)]
pub struct QoSPolicy {
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<QoSRule>,
    /// Policy enforcement
    pub enforcement: PolicyEnforcement,
}

impl Default for QoSPolicy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            rules: vec![QoSRule::default()],
            enforcement: PolicyEnforcement::Strict,
        }
    }
}

impl QoSPolicy {
    /// High-performance QoS policy
    pub fn high_performance() -> Self {
        Self {
            name: "high_performance".to_string(),
            rules: vec![
                QoSRule::priority_based(),
                QoSRule::bandwidth_based(),
            ],
            enforcement: PolicyEnforcement::Strict,
        }
    }

    /// Low-latency QoS policy
    pub fn low_latency() -> Self {
        Self {
            name: "low_latency".to_string(),
            rules: vec![QoSRule::latency_based()],
            enforcement: PolicyEnforcement::Strict,
        }
    }

    /// High-bandwidth QoS policy
    pub fn high_bandwidth() -> Self {
        Self {
            name: "high_bandwidth".to_string(),
            rules: vec![QoSRule::bandwidth_based()],
            enforcement: PolicyEnforcement::BestEffort,
        }
    }
}

/// QoS rule configuration
#[derive(Debug, Clone)]
pub struct QoSRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: RuleCondition,
    /// Rule action
    pub action: RuleAction,
    /// Rule priority
    pub priority: u32,
}

impl Default for QoSRule {
    fn default() -> Self {
        Self {
            name: "default_rule".to_string(),
            condition: RuleCondition::TrafficClass("data".to_string()),
            action: RuleAction::SetPriority(TrafficPriority::Medium),
            priority: 100,
        }
    }
}

impl QoSRule {
    /// Priority-based QoS rule
    pub fn priority_based() -> Self {
        Self {
            name: "priority_based".to_string(),
            condition: RuleCondition::Priority(TrafficPriority::High),
            action: RuleAction::GuaranteeBandwidth(30.0),
            priority: 10,
        }
    }

    /// Bandwidth-based QoS rule
    pub fn bandwidth_based() -> Self {
        Self {
            name: "bandwidth_based".to_string(),
            condition: RuleCondition::BandwidthUsage(80.0),
            action: RuleAction::ThrottleBandwidth(90.0),
            priority: 20,
        }
    }

    /// Latency-based QoS rule
    pub fn latency_based() -> Self {
        Self {
            name: "latency_based".to_string(),
            condition: RuleCondition::Latency(Duration::from_millis(5)),
            action: RuleAction::SetPriority(TrafficPriority::RealTimeCritical),
            priority: 5,
        }
    }
}

/// QoS rule conditions
#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// Traffic class condition
    TrafficClass(String),
    /// Priority condition
    Priority(TrafficPriority),
    /// Bandwidth usage condition
    BandwidthUsage(f64),
    /// Latency condition
    Latency(Duration),
    /// Packet loss condition
    PacketLoss(f64),
    /// Source/destination condition
    SourceDestination { source: String, destination: String },
}

/// QoS rule actions
#[derive(Debug, Clone)]
pub enum RuleAction {
    /// Set traffic priority
    SetPriority(TrafficPriority),
    /// Guarantee bandwidth percentage
    GuaranteeBandwidth(f64),
    /// Throttle bandwidth to percentage
    ThrottleBandwidth(f64),
    /// Drop packets
    DropPackets,
    /// Delay packets
    DelayPackets(Duration),
    /// Redirect traffic
    RedirectTraffic(String),
}

/// Policy enforcement levels
#[derive(Debug, Clone)]
pub enum PolicyEnforcement {
    /// Strict enforcement
    Strict,
    /// Best effort enforcement
    BestEffort,
    /// Monitoring only
    MonitorOnly,
}

/// Service level configuration
#[derive(Debug, Clone)]
pub struct ServiceLevel {
    /// Service level name
    pub name: String,
    /// Service level agreement
    pub sla: ServiceLevelAgreement,
    /// Monitoring configuration
    pub monitoring: ServiceLevelMonitoring,
}

impl Default for ServiceLevel {
    fn default() -> Self {
        Self {
            name: "standard".to_string(),
            sla: ServiceLevelAgreement::default(),
            monitoring: ServiceLevelMonitoring::default(),
        }
    }
}

impl ServiceLevel {
    /// Premium service level
    pub fn premium() -> Self {
        Self {
            name: "premium".to_string(),
            sla: ServiceLevelAgreement::premium(),
            monitoring: ServiceLevelMonitoring::comprehensive(),
        }
    }

    /// Real-time service level
    pub fn real_time() -> Self {
        Self {
            name: "real_time".to_string(),
            sla: ServiceLevelAgreement::real_time(),
            monitoring: ServiceLevelMonitoring::real_time(),
        }
    }

    /// Bulk transfer service level
    pub fn bulk_transfer() -> Self {
        Self {
            name: "bulk_transfer".to_string(),
            sla: ServiceLevelAgreement::bulk_transfer(),
            monitoring: ServiceLevelMonitoring::basic(),
        }
    }
}

/// Service Level Agreement
#[derive(Debug, Clone)]
pub struct ServiceLevelAgreement {
    /// Availability guarantee
    pub availability: f64,
    /// Latency guarantee
    pub latency_guarantee: Duration,
    /// Throughput guarantee
    pub throughput_guarantee: f64,
    /// Packet loss guarantee
    pub packet_loss_guarantee: f64,
    /// Jitter guarantee
    pub jitter_guarantee: Duration,
}

impl Default for ServiceLevelAgreement {
    fn default() -> Self {
        Self {
            availability: 0.99, // 99%
            latency_guarantee: Duration::from_millis(10),
            throughput_guarantee: 100_000_000.0, // 100 Mbps
            packet_loss_guarantee: 0.01, // 1%
            jitter_guarantee: Duration::from_millis(1),
        }
    }
}

impl ServiceLevelAgreement {
    /// Premium SLA
    pub fn premium() -> Self {
        Self {
            availability: 0.9999, // 99.99%
            latency_guarantee: Duration::from_millis(1),
            throughput_guarantee: 10_000_000_000.0, // 10 Gbps
            packet_loss_guarantee: 0.0001, // 0.01%
            jitter_guarantee: Duration::from_micros(100),
        }
    }

    /// Real-time SLA
    pub fn real_time() -> Self {
        Self {
            availability: 0.999, // 99.9%
            latency_guarantee: Duration::from_micros(500),
            throughput_guarantee: 1_000_000_000.0, // 1 Gbps
            packet_loss_guarantee: 0.0, // 0%
            jitter_guarantee: Duration::from_micros(10),
        }
    }

    /// Bulk transfer SLA
    pub fn bulk_transfer() -> Self {
        Self {
            availability: 0.99, // 99%
            latency_guarantee: Duration::from_millis(100),
            throughput_guarantee: 10_000_000_000.0, // 10 Gbps
            packet_loss_guarantee: 0.01, // 1%
            jitter_guarantee: Duration::from_millis(10),
        }
    }
}

/// Service level monitoring
#[derive(Debug, Clone)]
pub struct ServiceLevelMonitoring {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring frequency
    pub frequency: Duration,
    /// Monitored metrics
    pub metrics: Vec<String>,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
}

impl Default for ServiceLevelMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(60),
            metrics: vec![
                "availability".to_string(),
                "latency".to_string(),
                "throughput".to_string(),
            ],
            thresholds: HashMap::new(),
        }
    }
}

impl ServiceLevelMonitoring {
    /// Comprehensive monitoring
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(10),
            metrics: vec![
                "availability".to_string(),
                "latency".to_string(),
                "throughput".to_string(),
                "packet_loss".to_string(),
                "jitter".to_string(),
            ],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("availability".to_string(), 0.9999);
                thresholds.insert("latency".to_string(), 1.0); // 1ms
                thresholds.insert("packet_loss".to_string(), 0.0001);
                thresholds
            },
        }
    }

    /// Real-time monitoring
    pub fn real_time() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(1),
            metrics: vec!["latency".to_string(), "jitter".to_string()],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("latency".to_string(), 0.5); // 500μs
                thresholds.insert("jitter".to_string(), 0.01); // 10μs
                thresholds
            },
        }
    }

    /// Basic monitoring
    pub fn basic() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(300),
            metrics: vec!["throughput".to_string()],
            thresholds: HashMap::new(),
        }
    }
}

/// Bandwidth allocation configuration
#[derive(Debug, Clone)]
pub struct BandwidthAllocation {
    /// Total available bandwidth
    pub total_bandwidth: f64,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Fair sharing configuration
    pub fair_sharing: FairSharingConfig,
}

impl Default for BandwidthAllocation {
    fn default() -> Self {
        Self {
            total_bandwidth: 10_000_000_000.0, // 10 Gbps
            strategy: AllocationStrategy::Proportional,
            fair_sharing: FairSharingConfig::default(),
        }
    }
}

impl BandwidthAllocation {
    /// High-performance bandwidth allocation
    pub fn high_performance() -> Self {
        Self {
            total_bandwidth: 100_000_000_000.0, // 100 Gbps
            strategy: AllocationStrategy::Priority,
            fair_sharing: FairSharingConfig::performance_focused(),
        }
    }

    /// Low-latency bandwidth allocation
    pub fn low_latency() -> Self {
        Self {
            total_bandwidth: 10_000_000_000.0, // 10 Gbps
            strategy: AllocationStrategy::Priority,
            fair_sharing: FairSharingConfig::latency_focused(),
        }
    }

    /// High-bandwidth allocation
    pub fn high_bandwidth() -> Self {
        Self {
            total_bandwidth: 400_000_000_000.0, // 400 Gbps
            strategy: AllocationStrategy::MaxMin,
            fair_sharing: FairSharingConfig::bandwidth_focused(),
        }
    }
}

/// Bandwidth allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// Equal allocation
    Equal,
    /// Proportional allocation
    Proportional,
    /// Priority-based allocation
    Priority,
    /// Max-min fair allocation
    MaxMin,
    /// Weighted fair queuing
    WeightedFairQueuing,
}

/// Fair sharing configuration
#[derive(Debug, Clone)]
pub struct FairSharingConfig {
    /// Enable fair sharing
    pub enabled: bool,
    /// Sharing granularity
    pub granularity: SharingGranularity,
    /// Fairness algorithm
    pub algorithm: FairnessAlgorithm,
    /// Monitoring configuration
    pub monitoring: FairnessMonitoring,
}

impl Default for FairSharingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            granularity: SharingGranularity::PerFlow,
            algorithm: FairnessAlgorithm::WeightedFairQueuing,
            monitoring: FairnessMonitoring::default(),
        }
    }
}

impl FairSharingConfig {
    /// Performance-focused fair sharing
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            granularity: SharingGranularity::PerClass,
            algorithm: FairnessAlgorithm::StrictPriority,
            monitoring: FairnessMonitoring::performance_focused(),
        }
    }

    /// Latency-focused fair sharing
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            granularity: SharingGranularity::PerFlow,
            algorithm: FairnessAlgorithm::EarliestDeadlineFirst,
            monitoring: FairnessMonitoring::latency_focused(),
        }
    }

    /// Bandwidth-focused fair sharing
    pub fn bandwidth_focused() -> Self {
        Self {
            enabled: true,
            granularity: SharingGranularity::PerUser,
            algorithm: FairnessAlgorithm::DeficitRoundRobin,
            monitoring: FairnessMonitoring::bandwidth_focused(),
        }
    }
}

/// Sharing granularity levels
#[derive(Debug, Clone)]
pub enum SharingGranularity {
    /// Per flow sharing
    PerFlow,
    /// Per user sharing
    PerUser,
    /// Per application sharing
    PerApplication,
    /// Per traffic class sharing
    PerClass,
}

/// Fairness algorithms
#[derive(Debug, Clone)]
pub enum FairnessAlgorithm {
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Deficit round robin
    DeficitRoundRobin,
    /// Strict priority
    StrictPriority,
    /// Earliest deadline first
    EarliestDeadlineFirst,
    /// Start-time fair queuing
    StartTimeFairQueuing,
}

/// Fairness monitoring configuration
#[derive(Debug, Clone)]
pub struct FairnessMonitoring {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Fairness metrics
    pub metrics: Vec<String>,
    /// Fairness thresholds
    pub thresholds: HashMap<String, f64>,
}

impl Default for FairnessMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            metrics: vec![
                "bandwidth_fairness".to_string(),
                "latency_fairness".to_string(),
            ],
            thresholds: HashMap::new(),
        }
    }
}

impl FairnessMonitoring {
    /// Performance-focused fairness monitoring
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            metrics: vec!["bandwidth_fairness".to_string()],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("bandwidth_fairness".to_string(), 0.9);
                thresholds
            },
        }
    }

    /// Latency-focused fairness monitoring
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(5),
            metrics: vec!["latency_fairness".to_string()],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("latency_fairness".to_string(), 0.95);
                thresholds
            },
        }
    }

    /// Bandwidth-focused fairness monitoring
    pub fn bandwidth_focused() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            metrics: vec!["bandwidth_fairness".to_string()],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("bandwidth_fairness".to_string(), 0.99);
                thresholds
            },
        }
    }
}