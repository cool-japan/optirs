// Power Distribution Management
//
// This module handles power distribution units (PDUs), power ports, distribution logic,
// and load management for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::super::device_layout::Position3D;

/// Power distribution unit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDistributionUnit {
    /// PDU identifier
    pub pdu_id: String,
    /// Output ports
    pub ports: Vec<PowerPort>,
    /// Total capacity (watts)
    pub total_capacity: f64,
    /// Current usage (watts)
    pub current_usage: f64,
    /// PDU location
    pub location: Option<Position3D>,
    /// PDU specifications
    pub specifications: PowerDistributionUnitSpecs,
    /// PDU metrics
    pub metrics: PowerDistributionUnitMetrics,
    /// PDU configuration
    pub config: PowerDistributionUnitConfig,
    /// PDU status
    pub status: PowerDistributionUnitStatus,
}

impl Default for PowerDistributionUnit {
    fn default() -> Self {
        Self {
            pdu_id: String::new(),
            ports: Vec::new(),
            total_capacity: 0.0,
            current_usage: 0.0,
            location: None,
            specifications: PowerDistributionUnitSpecs::default(),
            metrics: PowerDistributionUnitMetrics::default(),
            config: PowerDistributionUnitConfig::default(),
            status: PowerDistributionUnitStatus::Normal,
        }
    }
}

/// Power distribution unit specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDistributionUnitSpecs {
    /// Number of outlets
    pub outlet_count: usize,
    /// Rated current (amperes)
    pub rated_current: f64,
    /// Input voltage (volts)
    pub input_voltage: f64,
    /// Output voltage (volts)
    pub output_voltage: f64,
    /// Phase configuration
    pub phase_config: PhaseConfiguration,
    /// Switching capability
    pub remote_switching: bool,
    /// Monitoring capabilities
    pub monitoring_capabilities: Vec<MonitoringCapability>,
    /// Form factor
    pub form_factor: PDUFormFactor,
    /// Mounting type
    pub mounting_type: MountingType,
    /// Input connector type
    pub input_connector: ConnectorType,
    /// Protection features
    pub protection_features: Vec<ProtectionFeature>,
}

impl Default for PowerDistributionUnitSpecs {
    fn default() -> Self {
        Self {
            outlet_count: 8,
            rated_current: 20.0,
            input_voltage: 120.0,
            output_voltage: 120.0,
            phase_config: PhaseConfiguration::SinglePhase,
            remote_switching: true,
            monitoring_capabilities: vec![
                MonitoringCapability::Voltage,
                MonitoringCapability::Current,
                MonitoringCapability::Power,
            ],
            form_factor: PDUFormFactor::Rack1U,
            mounting_type: MountingType::Horizontal,
            input_connector: ConnectorType::IEC320C20,
            protection_features: vec![
                ProtectionFeature::OverCurrentProtection,
                ProtectionFeature::SurgeProtection,
            ],
        }
    }
}

/// Phase configuration for power systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseConfiguration {
    /// Single phase
    SinglePhase,
    /// Three phase delta
    ThreePhaseDelta,
    /// Three phase wye
    ThreePhaseWye,
    /// Split phase
    SplitPhase,
    /// High-leg delta
    HighLegDelta,
}

/// Monitoring capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringCapability {
    /// Voltage monitoring
    Voltage,
    /// Current monitoring
    Current,
    /// Power monitoring
    Power,
    /// Energy monitoring
    Energy,
    /// Temperature monitoring
    Temperature,
    /// Frequency monitoring
    Frequency,
    /// Power factor monitoring
    PowerFactor,
    /// Harmonic monitoring
    Harmonics,
    /// Environmental monitoring
    Environmental,
}

/// PDU form factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PDUFormFactor {
    Rack1U,
    Rack2U,
    ZeroU,
    Tower,
    Cabinet,
    Inline,
    Custom(String),
}

/// Mounting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MountingType {
    Horizontal,
    Vertical,
    ZeroU,
    Toolless,
    Custom(String),
}

/// Protection features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtectionFeature {
    OverCurrentProtection,
    SurgeProtection,
    OverVoltageProtection,
    UnderVoltageProtection,
    ArcFaultProtection,
    GroundFaultProtection,
    ThermalProtection,
    ShortCircuitProtection,
}

/// Power distribution unit metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDistributionUnitMetrics {
    /// Input voltage (volts)
    pub input_voltage: f64,
    /// Input current (amperes)
    pub input_current: f64,
    /// Input frequency (Hz)
    pub input_frequency: f64,
    /// Power factor
    pub power_factor: f64,
    /// Apparent power (VA)
    pub apparent_power: f64,
    /// Real power (watts)
    pub real_power: f64,
    /// Reactive power (VAR)
    pub reactive_power: f64,
    /// Total harmonic distortion
    pub thd: f64,
    /// Crest factor
    pub crest_factor: f64,
    /// Load percentage
    pub load_percentage: f64,
    /// Temperature (Celsius)
    pub temperature: f64,
    /// Humidity (percentage)
    pub humidity: f64,
}

impl Default for PowerDistributionUnitMetrics {
    fn default() -> Self {
        Self {
            input_voltage: 120.0,
            input_current: 0.0,
            input_frequency: 60.0,
            power_factor: 1.0,
            apparent_power: 0.0,
            real_power: 0.0,
            reactive_power: 0.0,
            thd: 0.0,
            crest_factor: 1.414,
            load_percentage: 0.0,
            temperature: 25.0,
            humidity: 50.0,
        }
    }
}

/// Power distribution unit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDistributionUnitConfig {
    /// Enable remote switching
    pub remote_switching_enabled: bool,
    /// Power-on delay (seconds)
    pub power_on_delay: Duration,
    /// Power-off delay (seconds)
    pub power_off_delay: Duration,
    /// Overcurrent threshold (percentage)
    pub overcurrent_threshold: f64,
    /// Under-voltage threshold (volts)
    pub under_voltage_threshold: f64,
    /// Over-voltage threshold (volts)
    pub over_voltage_threshold: f64,
    /// Temperature threshold (Celsius)
    pub temperature_threshold: f64,
    /// Load balancing mode
    pub load_balancing_mode: LoadBalancingMode,
    /// Alarm settings
    pub alarm_settings: PDUAlarmSettings,
}

impl Default for PowerDistributionUnitConfig {
    fn default() -> Self {
        Self {
            remote_switching_enabled: true,
            power_on_delay: Duration::from_secs(5),
            power_off_delay: Duration::from_secs(3),
            overcurrent_threshold: 90.0,
            under_voltage_threshold: 100.0,
            over_voltage_threshold: 140.0,
            temperature_threshold: 50.0,
            load_balancing_mode: LoadBalancingMode::Automatic,
            alarm_settings: PDUAlarmSettings::default(),
        }
    }
}

/// Load balancing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingMode {
    /// Automatic load balancing
    Automatic,
    /// Manual load distribution
    Manual,
    /// Priority-based distribution
    Priority,
    /// Round-robin distribution
    RoundRobin,
    /// Efficiency-optimized distribution
    EfficiencyOptimized,
}

/// PDU alarm settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDUAlarmSettings {
    /// Enable current alarms
    pub current_alarms: bool,
    /// Enable voltage alarms
    pub voltage_alarms: bool,
    /// Enable temperature alarms
    pub temperature_alarms: bool,
    /// Enable power factor alarms
    pub power_factor_alarms: bool,
    /// Alarm notification methods
    pub notification_methods: Vec<NotificationMethod>,
}

impl Default for PDUAlarmSettings {
    fn default() -> Self {
        Self {
            current_alarms: true,
            voltage_alarms: true,
            temperature_alarms: true,
            power_factor_alarms: false,
            notification_methods: vec![NotificationMethod::Email, NotificationMethod::SNMP],
        }
    }
}

/// Notification methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationMethod {
    Email,
    SMS,
    SNMP,
    Syslog,
    HTTP,
    WebHook,
}

/// Power distribution unit status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerDistributionUnitStatus {
    Normal,
    Warning,
    Critical,
    Fault,
    Maintenance,
    Offline,
}

/// Power port information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPort {
    /// Port identifier
    pub port_id: String,
    /// Port number
    pub port_number: usize,
    /// Port status
    pub status: PowerPortStatus,
    /// Port specifications
    pub specifications: PowerPortSpecs,
    /// Port metrics
    pub metrics: PowerPortMetrics,
    /// Connected device
    pub connected_device: Option<String>,
    /// Port configuration
    pub config: PowerPortConfig,
}

impl Default for PowerPort {
    fn default() -> Self {
        Self {
            port_id: String::new(),
            port_number: 0,
            status: PowerPortStatus::Available,
            specifications: PowerPortSpecs::default(),
            metrics: PowerPortMetrics::default(),
            connected_device: None,
            config: PowerPortConfig::default(),
        }
    }
}

/// Power port status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerPortStatus {
    /// Port is available for use
    Available,
    /// Port is in use
    InUse,
    /// Port is disabled
    Disabled,
    /// Port has fault
    Fault,
    /// Port is in maintenance
    Maintenance,
    /// Port is overloaded
    Overloaded,
    /// Port is in standby
    Standby,
}

/// Power port specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPortSpecs {
    /// Rated voltage (volts)
    pub rated_voltage: f64,
    /// Rated current (amperes)
    pub rated_current: f64,
    /// Maximum power (watts)
    pub max_power: f64,
    /// Connector type
    pub connector_type: ConnectorType,
    /// Switching capability
    pub switchable: bool,
    /// Metering capability
    pub metered: bool,
    /// Protection features
    pub protection_features: Vec<PortProtectionFeature>,
}

impl Default for PowerPortSpecs {
    fn default() -> Self {
        Self {
            rated_voltage: 120.0,
            rated_current: 15.0,
            max_power: 1800.0,
            connector_type: ConnectorType::NEMA515R,
            switchable: true,
            metered: true,
            protection_features: vec![
                PortProtectionFeature::OverCurrent,
                PortProtectionFeature::SurgeProtection,
            ],
        }
    }
}

/// Connector types for power ports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectorType {
    /// NEMA 5-15R (Standard US outlet)
    NEMA515R,
    /// NEMA 5-20R (20A US outlet)
    NEMA520R,
    /// NEMA L5-20R (Locking 20A outlet)
    NEMAL520R,
    /// IEC 320 C13 (Standard computer outlet)
    IEC320C13,
    /// IEC 320 C19 (High current computer outlet)
    IEC320C19,
    /// IEC 320 C20 (High current computer inlet)
    IEC320C20,
    /// CEE 7/3 (European outlet)
    CEE73,
    /// Custom connector
    Custom(String),
}

/// Port protection features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PortProtectionFeature {
    OverCurrent,
    SurgeProtection,
    ArcFault,
    GroundFault,
    ThermalProtection,
    FastBlowFuse,
}

/// Power port metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPortMetrics {
    /// Current voltage (volts)
    pub voltage: f64,
    /// Current current (amperes)
    pub current: f64,
    /// Current power (watts)
    pub power: f64,
    /// Energy consumption (kWh)
    pub energy: f64,
    /// Power factor
    pub power_factor: f64,
    /// Load percentage
    pub load_percentage: f64,
    /// Connection time (duration since last connection)
    pub connection_time: Duration,
    /// Total usage time
    pub total_usage_time: Duration,
}

impl Default for PowerPortMetrics {
    fn default() -> Self {
        Self {
            voltage: 0.0,
            current: 0.0,
            power: 0.0,
            energy: 0.0,
            power_factor: 1.0,
            load_percentage: 0.0,
            connection_time: Duration::from_secs(0),
            total_usage_time: Duration::from_secs(0),
        }
    }
}

/// Power port configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPortConfig {
    /// Auto power-on enabled
    pub auto_power_on: bool,
    /// Power-on delay (seconds)
    pub power_on_delay: Duration,
    /// Power-off delay (seconds)
    pub power_off_delay: Duration,
    /// Current limit (amperes)
    pub current_limit: f64,
    /// Power limit (watts)
    pub power_limit: f64,
    /// Enable scheduling
    pub scheduling_enabled: bool,
    /// Power schedule
    pub power_schedule: Vec<PowerScheduleEntry>,
}

impl Default for PowerPortConfig {
    fn default() -> Self {
        Self {
            auto_power_on: false,
            power_on_delay: Duration::from_secs(0),
            power_off_delay: Duration::from_secs(0),
            current_limit: 15.0,
            power_limit: 1800.0,
            scheduling_enabled: false,
            power_schedule: Vec::new(),
        }
    }
}

/// Power schedule entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerScheduleEntry {
    /// Schedule name
    pub name: String,
    /// Start time (hour:minute)
    pub start_time: (u8, u8),
    /// End time (hour:minute)
    pub end_time: (u8, u8),
    /// Days of week (0=Sunday, 6=Saturday)
    pub days_of_week: Vec<u8>,
    /// Action to perform
    pub action: ScheduleAction,
    /// Enabled
    pub enabled: bool,
}

/// Schedule actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleAction {
    PowerOn,
    PowerOff,
    SetCurrentLimit(f64),
    SetPowerLimit(f64),
    EnablePort,
    DisablePort,
}

/// Power distribution manager
#[derive(Debug, Clone)]
pub struct PowerDistributionManager {
    /// Managed PDUs
    pub pdus: HashMap<String, PowerDistributionUnit>,
    /// Distribution topology
    pub topology: DistributionTopology,
    /// Load balancer
    pub load_balancer: DistributionLoadBalancer,
    /// Manager configuration
    pub config: DistributionManagerConfig,
}

impl Default for PowerDistributionManager {
    fn default() -> Self {
        Self {
            pdus: HashMap::new(),
            topology: DistributionTopology::default(),
            load_balancer: DistributionLoadBalancer::default(),
            config: DistributionManagerConfig::default(),
        }
    }
}

/// Distribution topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionTopology {
    /// Topology graph (PDU connections)
    pub connections: HashMap<String, Vec<String>>,
    /// Load paths
    pub load_paths: Vec<LoadPath>,
    /// Redundancy groups
    pub redundancy_groups: Vec<RedundancyGroup>,
    /// Isolation zones
    pub isolation_zones: Vec<IsolationZone>,
}

impl Default for DistributionTopology {
    fn default() -> Self {
        Self {
            connections: HashMap::new(),
            load_paths: Vec::new(),
            redundancy_groups: Vec::new(),
            isolation_zones: Vec::new(),
        }
    }
}

/// Load path in distribution topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPath {
    /// Path identifier
    pub path_id: String,
    /// Source PDU
    pub source_pdu: String,
    /// Destination devices
    pub destinations: Vec<String>,
    /// Path capacity (watts)
    pub capacity: f64,
    /// Current load (watts)
    pub current_load: f64,
    /// Path priority
    pub priority: PathPriority,
}

/// Path priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathPriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Redundancy group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyGroup {
    /// Group identifier
    pub group_id: String,
    /// Member PDUs
    pub members: Vec<String>,
    /// Redundancy level
    pub redundancy_level: RedundancyLevel,
    /// Failover strategy
    pub failover_strategy: FailoverStrategy,
}

/// Redundancy levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyLevel {
    None,
    NPlusOne,
    TwoN,
    Custom(String),
}

/// Failover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverStrategy {
    Automatic,
    Manual,
    LoadBased,
    PriorityBased,
}

/// Isolation zone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationZone {
    /// Zone identifier
    pub zone_id: String,
    /// Zone members (PDUs)
    pub members: Vec<String>,
    /// Isolation type
    pub isolation_type: IsolationType,
    /// Emergency actions
    pub emergency_actions: Vec<EmergencyAction>,
}

/// Isolation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationType {
    Electrical,
    Thermal,
    Safety,
    Maintenance,
}

/// Emergency actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    PowerOff,
    IsolatePDU,
    SwitchToBackup,
    NotifyOperator,
    TriggerAlarm,
}

/// Distribution load balancer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionLoadBalancer {
    /// Balancing strategy
    pub strategy: DistributionBalancingStrategy,
    /// Load thresholds
    pub thresholds: LoadThresholds,
    /// Balancing intervals
    pub balancing_interval: Duration,
    /// Performance metrics
    pub metrics: LoadBalancingMetrics,
}

impl Default for DistributionLoadBalancer {
    fn default() -> Self {
        Self {
            strategy: DistributionBalancingStrategy::EqualDistribution,
            thresholds: LoadThresholds::default(),
            balancing_interval: Duration::from_secs(60),
            metrics: LoadBalancingMetrics::default(),
        }
    }
}

/// Distribution balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionBalancingStrategy {
    EqualDistribution,
    CapacityProportional,
    EfficiencyOptimized,
    ThermalAware,
    PriorityBased,
}

/// Load thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadThresholds {
    /// Warning threshold (percentage)
    pub warning_threshold: f64,
    /// Critical threshold (percentage)
    pub critical_threshold: f64,
    /// Imbalance threshold (percentage difference)
    pub imbalance_threshold: f64,
}

impl Default for LoadThresholds {
    fn default() -> Self {
        Self {
            warning_threshold: 80.0,
            critical_threshold: 95.0,
            imbalance_threshold: 20.0,
        }
    }
}

/// Load balancing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingMetrics {
    /// Total load balanced (watts)
    pub total_load_balanced: f64,
    /// Balancing operations count
    pub balancing_operations: u64,
    /// Average imbalance
    pub average_imbalance: f64,
    /// Efficiency improvement
    pub efficiency_improvement: f64,
}

impl Default for LoadBalancingMetrics {
    fn default() -> Self {
        Self {
            total_load_balanced: 0.0,
            balancing_operations: 0,
            average_imbalance: 0.0,
            efficiency_improvement: 0.0,
        }
    }
}

/// Distribution manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionManagerConfig {
    /// Enable automatic load balancing
    pub auto_load_balancing: bool,
    /// Enable redundancy management
    pub redundancy_management: bool,
    /// Enable fault detection
    pub fault_detection: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

impl Default for DistributionManagerConfig {
    fn default() -> Self {
        Self {
            auto_load_balancing: true,
            redundancy_management: true,
            fault_detection: true,
            health_check_interval: Duration::from_secs(30),
            performance_monitoring: true,
        }
    }
}
