// Network Interfaces and Reliability Management
//
// This module handles network interfaces, reliability metrics, interface management,
// and network health monitoring for TPU pod communication.

use scirs2_core::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::config::{DeviceId, TopologyId};

/// Network configuration for device interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfiguration {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Routing table
    pub routing_table: RoutingTable,
    /// Default gateway
    pub default_gateway: Option<DeviceId>,
    /// DNS configuration
    pub dns_config: DNSConfiguration,
    /// Network security settings
    pub security_settings: NetworkSecuritySettings,
    /// Monitoring configuration
    pub monitoring_config: NetworkMonitoringConfig,
}

impl Default for NetworkConfiguration {
    fn default() -> Self {
        Self {
            interfaces: Vec::new(),
            routing_table: RoutingTable::default(),
            default_gateway: None,
            dns_config: DNSConfiguration::default(),
            security_settings: NetworkSecuritySettings::default(),
            monitoring_config: NetworkMonitoringConfig::default(),
        }
    }
}

/// Network interface definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface identifier
    pub interface_id: String,
    /// Interface name
    pub name: String,
    /// Interface type
    pub interface_type: InterfaceType,
    /// Interface configuration
    pub config: InterfaceConfiguration,
    /// Interface status
    pub status: InterfaceStatus,
    /// Interface metrics
    pub metrics: InterfaceMetrics,
    /// Quality of service settings
    pub qos_settings: InterfaceQoSSettings,
    /// Reliability configuration
    pub reliability_config: InterfaceReliabilityConfig,
}

impl NetworkInterface {
    /// Create a new network interface
    pub fn new(interface_id: String, name: String, interface_type: InterfaceType) -> Self {
        Self {
            interface_id,
            name,
            interface_type,
            config: InterfaceConfiguration::default(),
            status: InterfaceStatus::Down,
            metrics: InterfaceMetrics::new(),
            qos_settings: InterfaceQoSSettings::default(),
            reliability_config: InterfaceReliabilityConfig::default(),
        }
    }

    /// Bring interface up
    pub fn bring_up(&mut self) -> Result<()> {
        self.validate_configuration()?;
        self.status = InterfaceStatus::Up;
        self.metrics.last_state_change = Instant::now();
        Ok(())
    }

    /// Bring interface down
    pub fn bring_down(&mut self) -> Result<()> {
        self.status = InterfaceStatus::Down;
        self.metrics.last_state_change = Instant::now();
        Ok(())
    }

    /// Update interface metrics
    pub fn update_metrics(&mut self, new_metrics: InterfaceMetrics) {
        self.metrics = new_metrics;
        self.metrics.last_update = Instant::now();
    }

    /// Check interface health
    pub fn check_health(&self) -> InterfaceHealthStatus {
        let error_rate = self.metrics.error_rate;
        let utilization = self.metrics.utilization;
        let latency = self.metrics.latency;

        if error_rate > 0.1 || latency > Duration::from_millis(100) {
            InterfaceHealthStatus::Degraded
        } else if utilization > 0.9 {
            InterfaceHealthStatus::Congested
        } else if self.status != InterfaceStatus::Up {
            InterfaceHealthStatus::Down
        } else {
            InterfaceHealthStatus::Healthy
        }
    }

    /// Validate interface configuration
    fn validate_configuration(&self) -> Result<()> {
        // Implementation would validate interface configuration
        Ok(())
    }
}

/// Types of network interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceType {
    /// Ethernet interface
    Ethernet {
        speed: EthernetSpeed,
        duplex: DuplexMode,
    },
    /// InfiniBand interface
    InfiniBand {
        rate: InfiniBandRate,
        link_layer: InfiniBandLinkLayer,
    },
    /// Wireless interface
    Wireless {
        standard: WirelessStandard,
        frequency: f64,
    },
    /// Virtual interface
    Virtual { virtual_type: VirtualInterfaceType },
    /// Loopback interface
    Loopback,
    /// Custom interface type
    Custom {
        type_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Ethernet speeds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EthernetSpeed {
    Speed10M,
    Speed100M,
    Speed1G,
    Speed10G,
    Speed25G,
    Speed40G,
    Speed50G,
    Speed100G,
    Speed200G,
    Speed400G,
}

/// Duplex modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplexMode {
    Half,
    Full,
    Auto,
}

/// InfiniBand rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfiniBandRate {
    SDR, // 2.5 Gbps
    DDR, // 5 Gbps
    QDR, // 10 Gbps
    FDR, // 14 Gbps
    EDR, // 25 Gbps
    HDR, // 50 Gbps
    NDR, // 100 Gbps
    XDR, // 200 Gbps
}

/// InfiniBand link layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfiniBandLinkLayer {
    IB,       // InfiniBand
    Ethernet, // Ethernet over InfiniBand
    OPA,      // Omni-Path Architecture
}

/// Wireless standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WirelessStandard {
    IEEE80211a,
    IEEE80211b,
    IEEE80211g,
    IEEE80211n,
    IEEE80211ac,
    IEEE80211ax,
    IEEE80211be,
}

/// Virtual interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirtualInterfaceType {
    VLAN { vlan_id: u16 },
    Bridge,
    Tunnel { tunnel_type: TunnelType },
    Bond { bond_mode: BondMode },
}

/// Tunnel types for virtual interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TunnelType {
    GRE,
    VXLAN,
    IPSec,
    WireGuard,
    OpenVPN,
}

/// Bonding modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondMode {
    ActiveBackup,
    LoadBalancing,
    IEEE8023AD,
    Broadcast,
    IEEE8023AD_Dynamic,
}

/// Interface status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterfaceStatus {
    /// Interface is up and operational
    Up,
    /// Interface is down
    Down,
    /// Interface is testing
    Testing,
    /// Interface has unknown status
    Unknown,
    /// Interface is dormant
    Dormant,
    /// Interface is not present
    NotPresent,
    /// Interface is in lower layer down state
    LowerLayerDown,
}

/// Interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfiguration {
    /// IP addresses assigned to interface
    pub ip_addresses: Vec<IPAddress>,
    /// Maximum transmission unit
    pub mtu: usize,
    /// Hardware address (MAC)
    pub hardware_address: Option<String>,
    /// Interface speed (bps)
    pub speed: Option<f64>,
    /// Enable/disable interface
    pub enabled: bool,
    /// Auto-negotiation settings
    pub auto_negotiation: AutoNegotiationSettings,
    /// Flow control settings
    pub flow_control: FlowControlSettings,
    /// Jumbo frame support
    pub jumbo_frames: bool,
}

impl Default for InterfaceConfiguration {
    fn default() -> Self {
        Self {
            ip_addresses: Vec::new(),
            mtu: 1500,
            hardware_address: None,
            speed: None,
            enabled: true,
            auto_negotiation: AutoNegotiationSettings::default(),
            flow_control: FlowControlSettings::default(),
            jumbo_frames: false,
        }
    }
}

/// IP address configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPAddress {
    /// IP address
    pub address: String,
    /// Subnet mask or prefix length
    pub prefix_length: u8,
    /// Address type
    pub address_type: IPAddressType,
    /// Address scope
    pub scope: IPAddressScope,
}

/// IP address types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IPAddressType {
    IPv4,
    IPv6,
}

/// IP address scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IPAddressScope {
    Global,
    Link,
    Host,
    Site,
}

/// Auto-negotiation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoNegotiationSettings {
    /// Enable auto-negotiation
    pub enabled: bool,
    /// Advertised capabilities
    pub advertised_speeds: Vec<u64>,
    /// Advertised duplex modes
    pub advertised_duplex: Vec<DuplexMode>,
    /// Auto-negotiation timeout
    pub negotiation_timeout: Duration,
}

impl Default for AutoNegotiationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            advertised_speeds: vec![1_000_000_000, 10_000_000_000], // 1G, 10G
            advertised_duplex: vec![DuplexMode::Full],
            negotiation_timeout: Duration::from_secs(10),
        }
    }
}

/// Flow control settings for interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlSettings {
    /// Enable flow control
    pub enabled: bool,
    /// Receive flow control
    pub rx_flow_control: bool,
    /// Transmit flow control
    pub tx_flow_control: bool,
    /// Flow control type
    pub flow_control_type: FlowControlType,
}

impl Default for FlowControlSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            rx_flow_control: true,
            tx_flow_control: true,
            flow_control_type: FlowControlType::IEEE8023X,
        }
    }
}

/// Flow control types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowControlType {
    /// IEEE 802.3x pause frames
    IEEE8023X,
    /// Priority-based flow control
    PFC,
    /// Credit-based flow control
    CreditBased,
    /// Custom flow control
    Custom(String),
}

/// Interface metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceMetrics {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes transmitted
    pub bytes_transmitted: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets transmitted
    pub packets_transmitted: u64,
    /// Receive errors
    pub receive_errors: u64,
    /// Transmit errors
    pub transmit_errors: u64,
    /// Dropped packets (receive)
    pub rx_dropped: u64,
    /// Dropped packets (transmit)
    pub tx_dropped: u64,
    /// Interface utilization (0.0-1.0)
    pub utilization: f64,
    /// Current latency
    pub latency: Duration,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
    /// Last update timestamp
    pub last_update: Instant,
    /// Last state change timestamp
    pub last_state_change: Instant,
}

impl InterfaceMetrics {
    /// Create new interface metrics
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            bytes_received: 0,
            bytes_transmitted: 0,
            packets_received: 0,
            packets_transmitted: 0,
            receive_errors: 0,
            transmit_errors: 0,
            rx_dropped: 0,
            tx_dropped: 0,
            utilization: 0.0,
            latency: Duration::from_millis(0),
            error_rate: 0.0,
            last_update: now,
            last_state_change: now,
        }
    }

    /// Calculate throughput (bps)
    pub fn calculate_throughput(&self, time_interval: Duration) -> f64 {
        if time_interval.as_secs_f64() == 0.0 {
            return 0.0;
        }

        let total_bytes = self.bytes_received + self.bytes_transmitted;
        (total_bytes as f64 * 8.0) / time_interval.as_secs_f64()
    }

    /// Calculate packet rate (packets per second)
    pub fn calculate_packet_rate(&self, time_interval: Duration) -> f64 {
        if time_interval.as_secs_f64() == 0.0 {
            return 0.0;
        }

        let total_packets = self.packets_received + self.packets_transmitted;
        total_packets as f64 / time_interval.as_secs_f64()
    }

    /// Update error rate
    pub fn update_error_rate(&mut self) {
        let total_packets = self.packets_received + self.packets_transmitted;
        let total_errors = self.receive_errors + self.transmit_errors;

        if total_packets > 0 {
            self.error_rate = total_errors as f64 / total_packets as f64;
        } else {
            self.error_rate = 0.0;
        }
    }
}

/// Interface QoS settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceQoSSettings {
    /// Traffic classes
    pub traffic_classes: Vec<String>,
    /// Queue configuration
    pub queue_config: QueueConfiguration,
    /// Rate limiting
    pub rate_limiting: RateLimitingSettings,
    /// Buffer management
    pub buffer_management: BufferManagementSettings,
}

impl Default for InterfaceQoSSettings {
    fn default() -> Self {
        Self {
            traffic_classes: vec![
                "realtime".to_string(),
                "high_priority".to_string(),
                "normal".to_string(),
                "best_effort".to_string(),
            ],
            queue_config: QueueConfiguration::default(),
            rate_limiting: RateLimitingSettings::default(),
            buffer_management: BufferManagementSettings::default(),
        }
    }
}

/// Queue configuration for interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfiguration {
    /// Number of queues
    pub queue_count: usize,
    /// Queue sizes
    pub queue_sizes: Vec<usize>,
    /// Queue scheduling algorithm
    pub scheduling_algorithm: QueueSchedulingAlgorithm,
    /// Queue weights for weighted algorithms
    pub queue_weights: Vec<f64>,
}

impl Default for QueueConfiguration {
    fn default() -> Self {
        Self {
            queue_count: 4,
            queue_sizes: vec![1000, 2000, 4000, 8000],
            scheduling_algorithm: QueueSchedulingAlgorithm::WeightedRoundRobin,
            queue_weights: vec![0.4, 0.3, 0.2, 0.1],
        }
    }
}

/// Queue scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueSchedulingAlgorithm {
    FIFO,
    PriorityQueuing,
    RoundRobin,
    WeightedRoundRobin,
    DeficitRoundRobin,
    StochasticFairQueuing,
}

/// Rate limiting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingSettings {
    /// Enable rate limiting
    pub enabled: bool,
    /// Rate limit (bps)
    pub rate_limit: f64,
    /// Burst size (bytes)
    pub burst_size: usize,
    /// Rate limiting algorithm
    pub algorithm: RateLimitingAlgorithm,
}

impl Default for RateLimitingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            rate_limit: 1_000_000_000.0, // 1 Gbps
            burst_size: 65536,
            algorithm: RateLimitingAlgorithm::TokenBucket,
        }
    }
}

/// Rate limiting algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitingAlgorithm {
    TokenBucket,
    LeakyBucket,
    SlidingWindow,
    FixedWindow,
}

/// Buffer management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferManagementSettings {
    /// Total buffer size (bytes)
    pub total_buffer_size: usize,
    /// Buffer allocation strategy
    pub allocation_strategy: BufferAllocationStrategy,
    /// Drop policy when buffer is full
    pub drop_policy: BufferDropPolicy,
    /// Buffer thresholds
    pub thresholds: BufferThresholds,
}

impl Default for BufferManagementSettings {
    fn default() -> Self {
        Self {
            total_buffer_size: 1_048_576, // 1 MB
            allocation_strategy: BufferAllocationStrategy::Dynamic,
            drop_policy: BufferDropPolicy::TailDrop,
            thresholds: BufferThresholds::default(),
        }
    }
}

/// Buffer allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferAllocationStrategy {
    Static,
    Dynamic,
    Proportional,
    PriorityBased,
}

/// Buffer drop policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferDropPolicy {
    TailDrop,
    HeadDrop,
    RandomDrop,
    PriorityDrop,
    RED,  // Random Early Detection
    WRED, // Weighted Random Early Detection
}

/// Buffer thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferThresholds {
    /// Warning threshold (0.0-1.0)
    pub warning_threshold: f64,
    /// Critical threshold (0.0-1.0)
    pub critical_threshold: f64,
    /// Drop threshold (0.0-1.0)
    pub drop_threshold: f64,
}

impl Default for BufferThresholds {
    fn default() -> Self {
        Self {
            warning_threshold: 0.7,
            critical_threshold: 0.85,
            drop_threshold: 0.95,
        }
    }
}

/// Interface reliability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceReliabilityConfig {
    /// Redundancy settings
    pub redundancy: RedundancySettings,
    /// Fault detection settings
    pub fault_detection: FaultDetectionSettings,
    /// Recovery settings
    pub recovery: RecoverySettings,
    /// Health monitoring settings
    pub health_monitoring: HealthMonitoringSettings,
}

impl Default for InterfaceReliabilityConfig {
    fn default() -> Self {
        Self {
            redundancy: RedundancySettings::default(),
            fault_detection: FaultDetectionSettings::default(),
            recovery: RecoverySettings::default(),
            health_monitoring: HealthMonitoringSettings::default(),
        }
    }
}

/// Redundancy settings for interface reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancySettings {
    /// Enable redundancy
    pub enabled: bool,
    /// Redundancy type
    pub redundancy_type: RedundancyType,
    /// Backup interfaces
    pub backup_interfaces: Vec<String>,
    /// Failover settings
    pub failover: FailoverSettings,
}

impl Default for RedundancySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            redundancy_type: RedundancyType::ActiveBackup,
            backup_interfaces: Vec::new(),
            failover: FailoverSettings::default(),
        }
    }
}

/// Types of redundancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyType {
    /// Active-backup configuration
    ActiveBackup,
    /// Load balancing across interfaces
    LoadBalancing,
    /// Link aggregation
    LinkAggregation,
    /// Hot standby
    HotStandby,
}

/// Failover settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverSettings {
    /// Failover detection time
    pub detection_time: Duration,
    /// Failover switch time
    pub switch_time: Duration,
    /// Automatic failover enabled
    pub automatic: bool,
    /// Failback settings
    pub failback: FailbackSettings,
}

impl Default for FailoverSettings {
    fn default() -> Self {
        Self {
            detection_time: Duration::from_secs(3),
            switch_time: Duration::from_secs(1),
            automatic: true,
            failback: FailbackSettings::default(),
        }
    }
}

/// Failback settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailbackSettings {
    /// Enable automatic failback
    pub enabled: bool,
    /// Failback delay
    pub delay: Duration,
    /// Preemptive failback
    pub preemptive: bool,
}

impl Default for FailbackSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            delay: Duration::from_secs(10),
            preemptive: false,
        }
    }
}

/// Fault detection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultDetectionSettings {
    /// Detection methods
    pub detection_methods: Vec<FaultDetectionMethod>,
    /// Detection interval
    pub detection_interval: Duration,
    /// Detection thresholds
    pub thresholds: FaultDetectionThresholds,
    /// Correlation settings
    pub correlation: FaultCorrelationSettings,
}

impl Default for FaultDetectionSettings {
    fn default() -> Self {
        Self {
            detection_methods: vec![
                FaultDetectionMethod::LinkState,
                FaultDetectionMethod::KeepAlive,
                FaultDetectionMethod::ErrorRate,
            ],
            detection_interval: Duration::from_secs(1),
            thresholds: FaultDetectionThresholds::default(),
            correlation: FaultCorrelationSettings::default(),
        }
    }
}

/// Fault detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultDetectionMethod {
    /// Link state monitoring
    LinkState,
    /// Keep-alive packets
    KeepAlive,
    /// Error rate monitoring
    ErrorRate,
    /// Latency monitoring
    LatencyMonitoring,
    /// Throughput monitoring
    ThroughputMonitoring,
    /// SNMP monitoring
    SNMP,
    /// Custom monitoring
    Custom(String),
}

/// Fault detection thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultDetectionThresholds {
    /// Maximum acceptable error rate (0.0-1.0)
    pub max_error_rate: f64,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Minimum acceptable throughput (bps)
    pub min_throughput: f64,
    /// Keep-alive timeout
    pub keepalive_timeout: Duration,
}

impl Default for FaultDetectionThresholds {
    fn default() -> Self {
        Self {
            max_error_rate: 0.01,
            max_latency: Duration::from_millis(100),
            min_throughput: 1_000_000.0, // 1 Mbps
            keepalive_timeout: Duration::from_secs(5),
        }
    }
}

/// Fault correlation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultCorrelationSettings {
    /// Enable fault correlation
    pub enabled: bool,
    /// Correlation window
    pub correlation_window: Duration,
    /// Correlation threshold
    pub correlation_threshold: f64,
    /// Suppress correlated faults
    pub suppress_correlated: bool,
}

impl Default for FaultCorrelationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            correlation_window: Duration::from_secs(30),
            correlation_threshold: 0.8,
            suppress_correlated: true,
        }
    }
}

/// Recovery settings for interface faults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySettings {
    /// Recovery strategies
    pub strategies: Vec<RecoveryStrategy>,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Maximum recovery attempts
    pub max_attempts: usize,
    /// Recovery verification
    pub verification: RecoveryVerificationSettings,
}

impl Default for RecoverySettings {
    fn default() -> Self {
        Self {
            strategies: vec![
                RecoveryStrategy::InterfaceReset,
                RecoveryStrategy::ParameterReset,
                RecoveryStrategy::Failover,
            ],
            recovery_timeout: Duration::from_secs(30),
            max_attempts: 3,
            verification: RecoveryVerificationSettings::default(),
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Reset interface
    InterfaceReset,
    /// Reset interface parameters
    ParameterReset,
    /// Failover to backup interface
    Failover,
    /// Restart network service
    ServiceRestart,
    /// Physical layer reset
    PhysicalReset,
    /// Custom recovery action
    Custom(String),
}

/// Recovery verification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryVerificationSettings {
    /// Enable recovery verification
    pub enabled: bool,
    /// Verification tests
    pub tests: Vec<VerificationTest>,
    /// Verification timeout
    pub timeout: Duration,
    /// Verification interval
    pub interval: Duration,
}

impl Default for RecoveryVerificationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tests: vec![
                VerificationTest::Connectivity,
                VerificationTest::Throughput,
                VerificationTest::Latency,
            ],
            timeout: Duration::from_secs(10),
            interval: Duration::from_secs(1),
        }
    }
}

/// Verification tests for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationTest {
    /// Basic connectivity test
    Connectivity,
    /// Throughput test
    Throughput,
    /// Latency test
    Latency,
    /// Error rate test
    ErrorRate,
    /// Custom test
    Custom(String),
}

/// Health monitoring settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoringSettings {
    /// Enable health monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Health metrics to monitor
    pub metrics: Vec<HealthMetric>,
    /// Health scoring settings
    pub scoring: HealthScoringSettings,
}

impl Default for HealthMonitoringSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(5),
            metrics: vec![
                HealthMetric::ErrorRate,
                HealthMetric::Latency,
                HealthMetric::Throughput,
                HealthMetric::Utilization,
            ],
            scoring: HealthScoringSettings::default(),
        }
    }
}

/// Health metrics to monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthMetric {
    ErrorRate,
    Latency,
    Throughput,
    Utilization,
    PacketLoss,
    Jitter,
    Availability,
}

/// Health scoring settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthScoringSettings {
    /// Scoring algorithm
    pub algorithm: HealthScoringAlgorithm,
    /// Metric weights
    pub metric_weights: HashMap<String, f64>,
    /// Score aggregation method
    pub aggregation: ScoreAggregationMethod,
}

impl Default for HealthScoringSettings {
    fn default() -> Self {
        let mut metric_weights = HashMap::new();
        metric_weights.insert("error_rate".to_string(), 0.3);
        metric_weights.insert("latency".to_string(), 0.25);
        metric_weights.insert("throughput".to_string(), 0.2);
        metric_weights.insert("utilization".to_string(), 0.15);
        metric_weights.insert("availability".to_string(), 0.1);

        Self {
            algorithm: HealthScoringAlgorithm::WeightedAverage,
            metric_weights,
            aggregation: ScoreAggregationMethod::WeightedSum,
        }
    }
}

/// Health scoring algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthScoringAlgorithm {
    WeightedAverage,
    MinimumScore,
    MaximumScore,
    MedianScore,
    Custom(String),
}

/// Score aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreAggregationMethod {
    WeightedSum,
    GeometricMean,
    HarmonicMean,
    MinScore,
    MaxScore,
}

/// Interface health status
#[derive(Debug, Clone, PartialEq)]
pub enum InterfaceHealthStatus {
    Healthy,
    Degraded,
    Congested,
    Down,
    Unknown,
}

/// Network reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkReliabilityMetrics {
    /// Overall network availability (0.0-1.0)
    pub availability: f64,
    /// Mean time between failures (seconds)
    pub mtbf: f64,
    /// Mean time to repair (seconds)
    pub mttr: f64,
    /// Reliability score (0.0-1.0)
    pub reliability_score: f64,
    /// Fault tolerance score (0.0-1.0)
    pub fault_tolerance_score: f64,
    /// Recovery time statistics
    pub recovery_time_stats: RecoveryTimeStatistics,
}

impl Default for NetworkReliabilityMetrics {
    fn default() -> Self {
        Self {
            availability: 0.999,
            mtbf: 8760.0 * 3600.0, // 1 year in seconds
            mttr: 3600.0,          // 1 hour in seconds
            reliability_score: 0.99,
            fault_tolerance_score: 0.95,
            recovery_time_stats: RecoveryTimeStatistics::default(),
        }
    }
}

/// Recovery time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTimeStatistics {
    /// Average recovery time
    pub average_recovery_time: Duration,
    /// Minimum recovery time
    pub min_recovery_time: Duration,
    /// Maximum recovery time
    pub max_recovery_time: Duration,
    /// Recovery time standard deviation
    pub recovery_time_stddev: Duration,
    /// Recovery success rate (0.0-1.0)
    pub success_rate: f64,
}

impl Default for RecoveryTimeStatistics {
    fn default() -> Self {
        Self {
            average_recovery_time: Duration::from_secs(30),
            min_recovery_time: Duration::from_secs(5),
            max_recovery_time: Duration::from_secs(120),
            recovery_time_stddev: Duration::from_secs(15),
            success_rate: 0.95,
        }
    }
}

/// Routing table for network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    /// Static routes
    pub static_routes: Vec<StaticRoute>,
    /// Dynamic routes
    pub dynamic_routes: Vec<DynamicRoute>,
    /// Default route
    pub default_route: Option<Route>,
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self {
            static_routes: Vec::new(),
            dynamic_routes: Vec::new(),
            default_route: None,
        }
    }
}

/// Static route definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticRoute {
    /// Destination network
    pub destination: String,
    /// Next hop
    pub next_hop: DeviceId,
    /// Route metric
    pub metric: u32,
    /// Interface to use
    pub interface: Option<String>,
}

/// Dynamic route definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRoute {
    /// Destination network
    pub destination: String,
    /// Next hop
    pub next_hop: DeviceId,
    /// Route metric
    pub metric: u32,
    /// Route protocol
    pub protocol: RoutingProtocol,
    /// Route age
    pub age: Duration,
}

/// Basic route definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// Destination network
    pub destination: String,
    /// Next hop
    pub next_hop: DeviceId,
    /// Route metric
    pub metric: u32,
}

/// Routing protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingProtocol {
    Static,
    OSPF,
    BGP,
    RIP,
    EIGRP,
    ISIS,
}

/// DNS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNSConfiguration {
    /// DNS servers
    pub servers: Vec<String>,
    /// Search domains
    pub search_domains: Vec<String>,
    /// DNS timeout
    pub timeout: Duration,
    /// DNS retries
    pub retries: usize,
}

impl Default for DNSConfiguration {
    fn default() -> Self {
        Self {
            servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            search_domains: Vec::new(),
            timeout: Duration::from_secs(5),
            retries: 3,
        }
    }
}

/// Network security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSecuritySettings {
    /// Enable encryption
    pub encryption_enabled: bool,
    /// Encryption algorithms
    pub encryption_algorithms: Vec<String>,
    /// Authentication settings
    pub authentication: AuthenticationSettings,
    /// Firewall settings
    pub firewall: FirewallSettings,
}

impl Default for NetworkSecuritySettings {
    fn default() -> Self {
        Self {
            encryption_enabled: true,
            encryption_algorithms: vec!["AES-256".to_string(), "ChaCha20".to_string()],
            authentication: AuthenticationSettings::default(),
            firewall: FirewallSettings::default(),
        }
    }
}

/// Authentication settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationSettings {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Authentication timeout
    pub timeout: Duration,
    /// Maximum authentication attempts
    pub max_attempts: usize,
}

impl Default for AuthenticationSettings {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::CertificateBased,
            timeout: Duration::from_secs(30),
            max_attempts: 3,
        }
    }
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    None,
    Password,
    CertificateBased,
    TokenBased,
    MultiFactor,
}

/// Firewall settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallSettings {
    /// Enable firewall
    pub enabled: bool,
    /// Default policy
    pub default_policy: FirewallPolicy,
    /// Firewall rules
    pub rules: Vec<FirewallRule>,
}

impl Default for FirewallSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            default_policy: FirewallPolicy::Deny,
            rules: Vec::new(),
        }
    }
}

/// Firewall policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FirewallPolicy {
    Allow,
    Deny,
    Drop,
}

/// Firewall rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    /// Rule identifier
    pub id: String,
    /// Source address
    pub source: String,
    /// Destination address
    pub destination: String,
    /// Protocol
    pub protocol: String,
    /// Port range
    pub port_range: Option<(u16, u16)>,
    /// Action
    pub action: FirewallPolicy,
}

/// Network monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Metrics to collect
    pub metrics: Vec<String>,
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

impl Default for NetworkMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            metrics: vec![
                "throughput".to_string(),
                "latency".to_string(),
                "error_rate".to_string(),
                "utilization".to_string(),
            ],
            alerting: AlertingConfig::default(),
        }
    }
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: HashMap<String, f64>,
    /// Alert destinations
    pub destinations: Vec<String>,
    /// Alert cooldown period
    pub cooldown: Duration,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("error_rate".to_string(), 0.01);
        thresholds.insert("latency".to_string(), 100.0);
        thresholds.insert("utilization".to_string(), 0.9);

        Self {
            enabled: true,
            thresholds,
            destinations: Vec::new(),
            cooldown: Duration::from_minutes(5),
        }
    }
}
