// Power Monitoring and Measurement
//
// This module handles power monitoring systems, meters, data collection,
// and measurement analysis for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Power consumption monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMonitoring {
    /// Power meters
    pub meters: HashMap<String, PowerMeter>,
    /// Monitoring configuration
    pub config: PowerMonitoringConfig,
    /// Data collector
    pub data_collector: DataCollector,
    /// Alert manager
    pub alert_manager: AlertManager,
    /// Analytics engine
    pub analytics: PowerAnalytics,
}

impl Default for PowerMonitoring {
    fn default() -> Self {
        Self {
            meters: HashMap::new(),
            config: PowerMonitoringConfig::default(),
            data_collector: DataCollector::default(),
            alert_manager: AlertManager::default(),
            analytics: PowerAnalytics::default(),
        }
    }
}

/// Power meter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMeter {
    /// Meter identifier
    pub meter_id: String,
    /// Meter type
    pub meter_type: PowerMeterType,
    /// Meter status
    pub status: PowerMeterStatus,
    /// Meter specifications
    pub specifications: PowerMeterSpecs,
    /// Current readings
    pub current_readings: PowerReadings,
    /// Meter configuration
    pub config: PowerMeterConfig,
    /// Calibration information
    pub calibration: CalibrationInfo,
    /// Communication interface
    pub communication: CommunicationInterface,
}

impl Default for PowerMeter {
    fn default() -> Self {
        Self {
            meter_id: String::new(),
            meter_type: PowerMeterType::Digital,
            status: PowerMeterStatus::Normal,
            specifications: PowerMeterSpecs::default(),
            current_readings: PowerReadings::default(),
            config: PowerMeterConfig::default(),
            calibration: CalibrationInfo::default(),
            communication: CommunicationInterface::Modbus,
        }
    }
}

/// Power meter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerMeterType {
    /// Digital power meter
    Digital,
    /// Analog power meter
    Analog,
    /// Smart meter with advanced features
    Smart,
    /// Revenue-grade meter
    RevenueGrade,
    /// Submetering device
    Submeter,
    /// Current transformer based
    CTBased,
    /// Rogowski coil based
    RogowskiBased,
}

/// Power meter specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMeterSpecs {
    /// Accuracy class
    pub accuracy_class: AccuracyClass,
    /// Measurement range (voltage)
    pub voltage_range: (f64, f64),
    /// Measurement range (current)
    pub current_range: (f64, f64),
    /// Frequency range (Hz)
    pub frequency_range: (f64, f64),
    /// Power measurement accuracy (percentage)
    pub power_accuracy: f64,
    /// Energy measurement accuracy (percentage)
    pub energy_accuracy: f64,
    /// Sampling rate (samples per second)
    pub sampling_rate: f64,
    /// Communication interfaces
    pub communication_interfaces: Vec<CommunicationInterface>,
    /// Operating conditions
    pub operating_conditions: OperatingConditions,
    /// Certification compliance
    pub certifications: Vec<String>,
}

impl Default for PowerMeterSpecs {
    fn default() -> Self {
        Self {
            accuracy_class: AccuracyClass::Class1,
            voltage_range: (80.0, 300.0),
            current_range: (0.1, 100.0),
            frequency_range: (45.0, 65.0),
            power_accuracy: 0.5,
            energy_accuracy: 0.2,
            sampling_rate: 1000.0,
            communication_interfaces: vec![
                CommunicationInterface::Modbus,
                CommunicationInterface::Ethernet,
            ],
            operating_conditions: OperatingConditions::default(),
            certifications: vec!["IEC 62053-22".to_string(), "ANSI C12.20".to_string()],
        }
    }
}

/// Accuracy classes for power meters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyClass {
    /// Class 0.1 (0.1% accuracy)
    Class01,
    /// Class 0.2 (0.2% accuracy)
    Class02,
    /// Class 0.5 (0.5% accuracy)
    Class05,
    /// Class 1 (1% accuracy)
    Class1,
    /// Class 2 (2% accuracy)
    Class2,
    /// Class 3 (3% accuracy)
    Class3,
    /// Custom accuracy
    Custom(f64),
}

/// Communication interfaces for power meters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationInterface {
    /// Modbus RTU/TCP
    Modbus,
    /// Ethernet TCP/IP
    Ethernet,
    /// Serial RS-485
    RS485,
    /// Serial RS-232
    RS232,
    /// DNP3 protocol
    DNP3,
    /// IEC 61850
    IEC61850,
    /// BACnet
    BACnet,
    /// SNMP
    SNMP,
    /// HTTP/REST API
    HTTP,
    /// MQTT
    MQTT,
    /// Wireless (WiFi)
    WiFi,
    /// Wireless (Zigbee)
    Zigbee,
    /// CAN Bus
    CANBus,
}

/// Operating conditions for power meters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingConditions {
    /// Operating temperature range (Celsius)
    pub temperature_range: (f64, f64),
    /// Operating humidity range (percentage)
    pub humidity_range: (f64, f64),
    /// Altitude range (meters)
    pub altitude_range: (f64, f64),
    /// Vibration tolerance
    pub vibration_tolerance: VibrationTolerance,
    /// IP rating (ingress protection)
    pub ip_rating: String,
    /// EMC compliance
    pub emc_compliance: Vec<String>,
}

impl Default for OperatingConditions {
    fn default() -> Self {
        Self {
            temperature_range: (-20.0, 70.0),
            humidity_range: (5.0, 95.0),
            altitude_range: (0.0, 2000.0),
            vibration_tolerance: VibrationTolerance::default(),
            ip_rating: "IP54".to_string(),
            emc_compliance: vec!["EN 61326-1".to_string(), "FCC Part 15".to_string()],
        }
    }
}

/// Vibration tolerance specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibrationTolerance {
    /// Frequency range (Hz)
    pub frequency_range: (f64, f64),
    /// Acceleration limit (g)
    pub acceleration_limit: f64,
    /// Displacement limit (mm)
    pub displacement_limit: f64,
}

impl Default for VibrationTolerance {
    fn default() -> Self {
        Self {
            frequency_range: (10.0, 150.0),
            acceleration_limit: 2.0,
            displacement_limit: 0.15,
        }
    }
}

/// Power meter status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerMeterStatus {
    /// Normal operation
    Normal,
    /// Warning condition
    Warning,
    /// Error condition
    Error,
    /// Calibration required
    CalibrationRequired,
    /// Communication failure
    CommunicationFailure,
    /// Maintenance mode
    Maintenance,
    /// Offline
    Offline,
}

/// Power readings from meter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerReadings {
    /// Timestamp of readings
    pub timestamp: SystemTime,
    /// Voltage readings (volts)
    pub voltages: VoltageReadings,
    /// Current readings (amperes)
    pub currents: CurrentReadings,
    /// Power readings (watts)
    pub powers: PowerValues,
    /// Energy readings (kWh)
    pub energy: EnergyReadings,
    /// Power quality metrics
    pub power_quality: PowerQualityMetrics,
    /// Environmental readings
    pub environmental: EnvironmentalReadings,
}

impl Default for PowerReadings {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            voltages: VoltageReadings::default(),
            currents: CurrentReadings::default(),
            powers: PowerValues::default(),
            energy: EnergyReadings::default(),
            power_quality: PowerQualityMetrics::default(),
            environmental: EnvironmentalReadings::default(),
        }
    }
}

/// Voltage readings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageReadings {
    /// Phase A voltage (volts)
    pub phase_a: f64,
    /// Phase B voltage (volts)
    pub phase_b: Option<f64>,
    /// Phase C voltage (volts)
    pub phase_c: Option<f64>,
    /// Neutral voltage (volts)
    pub neutral: Option<f64>,
    /// Line-to-line voltages
    pub line_to_line: Vec<f64>,
    /// Average voltage
    pub average: f64,
    /// Voltage unbalance (percentage)
    pub unbalance: f64,
}

impl Default for VoltageReadings {
    fn default() -> Self {
        Self {
            phase_a: 0.0,
            phase_b: None,
            phase_c: None,
            neutral: None,
            line_to_line: Vec::new(),
            average: 0.0,
            unbalance: 0.0,
        }
    }
}

/// Current readings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentReadings {
    /// Phase A current (amperes)
    pub phase_a: f64,
    /// Phase B current (amperes)
    pub phase_b: Option<f64>,
    /// Phase C current (amperes)
    pub phase_c: Option<f64>,
    /// Neutral current (amperes)
    pub neutral: Option<f64>,
    /// Ground current (amperes)
    pub ground: Option<f64>,
    /// Average current
    pub average: f64,
    /// Current unbalance (percentage)
    pub unbalance: f64,
    /// Peak current
    pub peak: f64,
}

impl Default for CurrentReadings {
    fn default() -> Self {
        Self {
            phase_a: 0.0,
            phase_b: None,
            phase_c: None,
            neutral: None,
            ground: None,
            average: 0.0,
            unbalance: 0.0,
            peak: 0.0,
        }
    }
}

/// Power values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerValues {
    /// Active power (watts)
    pub active: f64,
    /// Reactive power (VAR)
    pub reactive: f64,
    /// Apparent power (VA)
    pub apparent: f64,
    /// Power factor
    pub power_factor: f64,
    /// Power factor by phase
    pub phase_power_factors: Vec<f64>,
    /// Total harmonic distortion (current)
    pub thd_current: f64,
    /// Total harmonic distortion (voltage)
    pub thd_voltage: f64,
    /// Individual harmonic components
    pub harmonics: Vec<HarmonicComponent>,
}

impl Default for PowerValues {
    fn default() -> Self {
        Self {
            active: 0.0,
            reactive: 0.0,
            apparent: 0.0,
            power_factor: 1.0,
            phase_power_factors: Vec::new(),
            thd_current: 0.0,
            thd_voltage: 0.0,
            harmonics: Vec::new(),
        }
    }
}

/// Harmonic component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicComponent {
    /// Harmonic order (2nd, 3rd, 5th, etc.)
    pub order: u32,
    /// Magnitude (percentage of fundamental)
    pub magnitude: f64,
    /// Phase angle (degrees)
    pub phase_angle: f64,
}

/// Energy readings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReadings {
    /// Active energy imported (kWh)
    pub active_imported: f64,
    /// Active energy exported (kWh)
    pub active_exported: f64,
    /// Reactive energy imported (kVARh)
    pub reactive_imported: f64,
    /// Reactive energy exported (kVARh)
    pub reactive_exported: f64,
    /// Apparent energy (kVAh)
    pub apparent: f64,
    /// Peak demand (kW)
    pub peak_demand: f64,
    /// Peak demand timestamp
    pub peak_demand_time: SystemTime,
}

impl Default for EnergyReadings {
    fn default() -> Self {
        Self {
            active_imported: 0.0,
            active_exported: 0.0,
            reactive_imported: 0.0,
            reactive_exported: 0.0,
            apparent: 0.0,
            peak_demand: 0.0,
            peak_demand_time: SystemTime::now(),
        }
    }
}

/// Power quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerQualityMetrics {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Voltage sag events
    pub voltage_sag_events: u32,
    /// Voltage swell events
    pub voltage_swell_events: u32,
    /// Power interruption events
    pub interruption_events: u32,
    /// Transient events
    pub transient_events: u32,
    /// Flicker severity
    pub flicker_severity: f64,
    /// Unbalance factor
    pub unbalance_factor: f64,
}

impl Default for PowerQualityMetrics {
    fn default() -> Self {
        Self {
            frequency: 60.0,
            voltage_sag_events: 0,
            voltage_swell_events: 0,
            interruption_events: 0,
            transient_events: 0,
            flicker_severity: 0.0,
            unbalance_factor: 0.0,
        }
    }
}

/// Environmental readings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalReadings {
    /// Temperature (Celsius)
    pub temperature: f64,
    /// Humidity (percentage)
    pub humidity: f64,
    /// Atmospheric pressure (hPa)
    pub pressure: f64,
    /// Ambient light level (lux)
    pub light_level: f64,
}

impl Default for EnvironmentalReadings {
    fn default() -> Self {
        Self {
            temperature: 25.0,
            humidity: 50.0,
            pressure: 1013.25,
            light_level: 500.0,
        }
    }
}

/// Power meter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMeterConfig {
    /// Data collection interval
    pub collection_interval: Duration,
    /// Data logging enabled
    pub logging_enabled: bool,
    /// Alert thresholds
    pub alert_thresholds: MeterAlertThresholds,
    /// Communication settings
    pub communication_settings: CommunicationSettings,
    /// Calibration settings
    pub calibration_settings: CalibrationSettings,
}

impl Default for PowerMeterConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(1),
            logging_enabled: true,
            alert_thresholds: MeterAlertThresholds::default(),
            communication_settings: CommunicationSettings::default(),
            calibration_settings: CalibrationSettings::default(),
        }
    }
}

/// Meter alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeterAlertThresholds {
    /// Voltage threshold (percentage deviation)
    pub voltage_threshold: f64,
    /// Current threshold (percentage of rated)
    pub current_threshold: f64,
    /// Power threshold (percentage of rated)
    pub power_threshold: f64,
    /// Frequency threshold (Hz deviation)
    pub frequency_threshold: f64,
    /// THD threshold (percentage)
    pub thd_threshold: f64,
    /// Power factor threshold
    pub power_factor_threshold: f64,
}

impl Default for MeterAlertThresholds {
    fn default() -> Self {
        Self {
            voltage_threshold: 10.0,
            current_threshold: 90.0,
            power_threshold: 90.0,
            frequency_threshold: 1.0,
            thd_threshold: 5.0,
            power_factor_threshold: 0.8,
        }
    }
}

/// Communication settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationSettings {
    /// Primary interface
    pub primary_interface: CommunicationInterface,
    /// Backup interfaces
    pub backup_interfaces: Vec<CommunicationInterface>,
    /// Communication timeout
    pub timeout: Duration,
    /// Retry attempts
    pub retry_attempts: u32,
    /// Data transmission rate
    pub baud_rate: u32,
    /// Network settings
    pub network_settings: NetworkSettings,
}

impl Default for CommunicationSettings {
    fn default() -> Self {
        Self {
            primary_interface: CommunicationInterface::Modbus,
            backup_interfaces: vec![CommunicationInterface::Ethernet],
            timeout: Duration::from_secs(5),
            retry_attempts: 3,
            baud_rate: 9600,
            network_settings: NetworkSettings::default(),
        }
    }
}

/// Network settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    /// IP address
    pub ip_address: String,
    /// Subnet mask
    pub subnet_mask: String,
    /// Gateway
    pub gateway: String,
    /// DNS servers
    pub dns_servers: Vec<String>,
    /// DHCP enabled
    pub dhcp_enabled: bool,
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            ip_address: "192.168.1.100".to_string(),
            subnet_mask: "255.255.255.0".to_string(),
            gateway: "192.168.1.1".to_string(),
            dns_servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            dhcp_enabled: true,
        }
    }
}

/// Calibration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    /// Last calibration date
    pub last_calibration: SystemTime,
    /// Next calibration due
    pub next_calibration_due: SystemTime,
    /// Calibration certificate
    pub certificate_number: String,
    /// Calibration authority
    pub calibration_authority: String,
    /// Calibration constants
    pub calibration_constants: CalibrationConstants,
    /// Calibration status
    pub status: CalibrationStatus,
}

impl Default for CalibrationInfo {
    fn default() -> Self {
        Self {
            last_calibration: SystemTime::now(),
            next_calibration_due: SystemTime::now(),
            certificate_number: String::new(),
            calibration_authority: String::new(),
            calibration_constants: CalibrationConstants::default(),
            status: CalibrationStatus::Valid,
        }
    }
}

/// Calibration constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConstants {
    /// Voltage calibration factor
    pub voltage_factor: f64,
    /// Current calibration factor
    pub current_factor: f64,
    /// Power calibration factor
    pub power_factor: f64,
    /// Phase correction
    pub phase_correction: f64,
    /// Offset corrections
    pub offset_corrections: Vec<f64>,
}

impl Default for CalibrationConstants {
    fn default() -> Self {
        Self {
            voltage_factor: 1.0,
            current_factor: 1.0,
            power_factor: 1.0,
            phase_correction: 0.0,
            offset_corrections: Vec::new(),
        }
    }
}

/// Calibration status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationStatus {
    Valid,
    Expired,
    RequiresCalibration,
    InProgress,
    Failed,
}

/// Calibration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSettings {
    /// Auto-calibration enabled
    pub auto_calibration: bool,
    /// Calibration interval
    pub calibration_interval: Duration,
    /// Calibration standards
    pub standards: Vec<CalibrationStandard>,
    /// Temperature compensation
    pub temperature_compensation: bool,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            auto_calibration: false,
            calibration_interval: Duration::from_secs(365 * 24 * 3600), // 1 year
            standards: vec![CalibrationStandard::default()],
            temperature_compensation: true,
        }
    }
}

/// Calibration standard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStandard {
    /// Standard name
    pub name: String,
    /// Reference value
    pub reference_value: f64,
    /// Uncertainty
    pub uncertainty: f64,
    /// Traceability
    pub traceability: String,
}

impl Default for CalibrationStandard {
    fn default() -> Self {
        Self {
            name: "NIST Reference".to_string(),
            reference_value: 0.0,
            uncertainty: 0.01,
            traceability: "NIST".to_string(),
        }
    }
}

/// Power monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMonitoringConfig {
    /// Global monitoring interval
    pub monitoring_interval: Duration,
    /// Data retention period
    pub data_retention_period: Duration,
    /// Alert policies
    pub alert_policies: Vec<MonitoringPolicy>,
    /// Quality standards
    pub quality_standards: QualityStandards,
    /// Reporting configuration
    pub reporting: ReportingConfig,
}

impl Default for PowerMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(10),
            data_retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
            alert_policies: vec![MonitoringPolicy::default()],
            quality_standards: QualityStandards::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

/// Monitoring policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringPolicy {
    /// Policy name
    pub name: String,
    /// Policy enabled
    pub enabled: bool,
    /// Conditions
    pub conditions: Vec<PolicyCondition>,
    /// Actions
    pub actions: Vec<PolicyAction>,
    /// Priority
    pub priority: PolicyPriority,
}

impl Default for MonitoringPolicy {
    fn default() -> Self {
        Self {
            name: "Default Policy".to_string(),
            enabled: true,
            conditions: Vec::new(),
            actions: Vec::new(),
            priority: PolicyPriority::Medium,
        }
    }
}

/// Policy condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    /// Parameter to monitor
    pub parameter: MonitoringParameter,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f64,
    /// Duration requirement
    pub duration: Duration,
}

/// Monitoring parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringParameter {
    Voltage,
    Current,
    Power,
    Energy,
    Frequency,
    PowerFactor,
    THD,
    Temperature,
    Humidity,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Policy action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    SendAlert,
    LogEvent,
    TriggerAlarm,
    NotifyOperator,
    ExecuteScript(String),
    ShutdownDevice(String),
}

/// Policy priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStandards {
    /// Voltage quality standards
    pub voltage_standards: VoltageStandards,
    /// Frequency standards
    pub frequency_standards: FrequencyStandards,
    /// Harmonic standards
    pub harmonic_standards: HarmonicStandards,
    /// Power factor standards
    pub power_factor_standards: PowerFactorStandards,
}

impl Default for QualityStandards {
    fn default() -> Self {
        Self {
            voltage_standards: VoltageStandards::default(),
            frequency_standards: FrequencyStandards::default(),
            harmonic_standards: HarmonicStandards::default(),
            power_factor_standards: PowerFactorStandards::default(),
        }
    }
}

/// Voltage standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageStandards {
    /// Nominal voltage
    pub nominal: f64,
    /// Acceptable range (percentage)
    pub acceptable_range: (f64, f64),
    /// Warning range (percentage)
    pub warning_range: (f64, f64),
}

impl Default for VoltageStandards {
    fn default() -> Self {
        Self {
            nominal: 120.0,
            acceptable_range: (-5.0, 5.0),
            warning_range: (-10.0, 10.0),
        }
    }
}

/// Frequency standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyStandards {
    /// Nominal frequency
    pub nominal: f64,
    /// Acceptable range (Hz)
    pub acceptable_range: (f64, f64),
    /// Warning range (Hz)
    pub warning_range: (f64, f64),
}

impl Default for FrequencyStandards {
    fn default() -> Self {
        Self {
            nominal: 60.0,
            acceptable_range: (59.5, 60.5),
            warning_range: (59.0, 61.0),
        }
    }
}

/// Harmonic standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicStandards {
    /// Total harmonic distortion limit (percentage)
    pub thd_limit: f64,
    /// Individual harmonic limits
    pub individual_limits: HashMap<u32, f64>,
    /// Standard reference
    pub standard_reference: String,
}

impl Default for HarmonicStandards {
    fn default() -> Self {
        let mut individual_limits = HashMap::new();
        individual_limits.insert(3, 5.0); // 3rd harmonic: 5%
        individual_limits.insert(5, 4.0); // 5th harmonic: 4%
        individual_limits.insert(7, 3.0); // 7th harmonic: 3%

        Self {
            thd_limit: 8.0,
            individual_limits,
            standard_reference: "IEEE 519".to_string(),
        }
    }
}

/// Power factor standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerFactorStandards {
    /// Minimum power factor
    pub minimum: f64,
    /// Target power factor
    pub target: f64,
    /// Penalty threshold
    pub penalty_threshold: f64,
}

impl Default for PowerFactorStandards {
    fn default() -> Self {
        Self {
            minimum: 0.8,
            target: 0.95,
            penalty_threshold: 0.85,
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable automatic reports
    pub auto_reports: bool,
    /// Report generation interval
    pub report_interval: Duration,
    /// Report types
    pub report_types: Vec<ReportType>,
    /// Report format
    pub report_format: ReportFormat,
    /// Distribution list
    pub distribution_list: Vec<String>,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            auto_reports: true,
            report_interval: Duration::from_secs(24 * 3600), // Daily
            report_types: vec![ReportType::PowerQuality, ReportType::EnergyUsage],
            report_format: ReportFormat::PDF,
            distribution_list: Vec::new(),
        }
    }
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    PowerQuality,
    EnergyUsage,
    Efficiency,
    LoadProfile,
    Alarms,
    Maintenance,
    Custom(String),
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    Excel,
    CSV,
    JSON,
    XML,
}

/// Data collector for power monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollector {
    /// Collection configuration
    pub config: DataCollectionConfig,
    /// Data buffer
    pub buffer: VecDeque<PowerReadings>,
    /// Storage manager
    pub storage: DataStorage,
    /// Statistics
    pub statistics: CollectionStatistics,
}

impl Default for DataCollector {
    fn default() -> Self {
        Self {
            config: DataCollectionConfig::default(),
            buffer: VecDeque::new(),
            storage: DataStorage::default(),
            statistics: CollectionStatistics::default(),
        }
    }
}

/// Data collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionConfig {
    /// Buffer size
    pub buffer_size: usize,
    /// Collection rate
    pub collection_rate: Duration,
    /// Data compression
    pub compression_enabled: bool,
    /// Data validation
    pub validation_enabled: bool,
}

impl Default for DataCollectionConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            collection_rate: Duration::from_secs(1),
            compression_enabled: true,
            validation_enabled: true,
        }
    }
}

/// Data storage for collected measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStorage {
    /// Storage type
    pub storage_type: StorageType,
    /// Storage location
    pub location: String,
    /// Retention policy
    pub retention_policy: RetentionPolicy,
    /// Backup configuration
    pub backup_config: BackupConfig,
}

impl Default for DataStorage {
    fn default() -> Self {
        Self {
            storage_type: StorageType::Database,
            location: "power_monitoring.db".to_string(),
            retention_policy: RetentionPolicy::default(),
            backup_config: BackupConfig::default(),
        }
    }
}

/// Storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    Database,
    FileSystem,
    Cloud,
    DistributedStorage,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Raw data retention (days)
    pub raw_data_days: u32,
    /// Aggregated data retention (days)
    pub aggregated_data_days: u32,
    /// Archive after (days)
    pub archive_after_days: u32,
    /// Compression after (days)
    pub compress_after_days: u32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            raw_data_days: 30,
            aggregated_data_days: 365,
            archive_after_days: 90,
            compress_after_days: 7,
        }
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup enabled
    pub enabled: bool,
    /// Backup frequency
    pub frequency: Duration,
    /// Backup location
    pub backup_location: String,
    /// Number of backups to keep
    pub backup_count: u32,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(24 * 3600), // Daily
            backup_location: "backups/".to_string(),
            backup_count: 7,
        }
    }
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    /// Total samples collected
    pub total_samples: u64,
    /// Successful collections
    pub successful_collections: u64,
    /// Failed collections
    pub failed_collections: u64,
    /// Data quality score
    pub data_quality_score: f64,
    /// Collection efficiency
    pub collection_efficiency: f64,
}

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self {
            total_samples: 0,
            successful_collections: 0,
            failed_collections: 0,
            data_quality_score: 1.0,
            collection_efficiency: 1.0,
        }
    }
}

/// Alert manager for power monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManager {
    /// Alert configuration
    pub config: AlertManagerConfig,
    /// Active alerts
    pub active_alerts: Vec<PowerAlert>,
    /// Alert history
    pub alert_history: VecDeque<PowerAlert>,
    /// Notification system
    pub notification_system: NotificationSystem,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self {
            config: AlertManagerConfig::default(),
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
            notification_system: NotificationSystem::default(),
        }
    }
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Alert escalation enabled
    pub escalation_enabled: bool,
    /// Alert suppression duration
    pub suppression_duration: Duration,
    /// Maximum active alerts
    pub max_active_alerts: usize,
    /// Alert history size
    pub history_size: usize,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            escalation_enabled: true,
            suppression_duration: Duration::from_secs(300), // 5 minutes
            max_active_alerts: 100,
            history_size: 1000,
        }
    }
}

/// Power alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Source meter
    pub source_meter: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Acknowledged
    pub acknowledged: bool,
    /// Resolved
    pub resolved: bool,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    VoltageOutOfRange,
    CurrentOverload,
    PowerFactor,
    Frequency,
    THD,
    Temperature,
    CommunicationFailure,
    CalibrationDue,
    DataQuality,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Notification system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSystem {
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Escalation rules
    pub escalation_rules: Vec<EscalationRule>,
    /// Notification templates
    pub templates: HashMap<AlertType, NotificationTemplate>,
}

impl Default for NotificationSystem {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            escalation_rules: Vec::new(),
            templates: HashMap::new(),
        }
    }
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: ChannelType,
    /// Channel configuration
    pub config: ChannelConfig,
    /// Enabled
    pub enabled: bool,
}

/// Channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    Email,
    SMS,
    Slack,
    Teams,
    WebHook,
    SNMP,
    Syslog,
}

/// Channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel-specific settings
    pub settings: HashMap<String, String>,
}

/// Escalation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    /// Rule name
    pub name: String,
    /// Trigger conditions
    pub conditions: Vec<EscalationCondition>,
    /// Escalation delay
    pub delay: Duration,
    /// Target channels
    pub target_channels: Vec<String>,
}

/// Escalation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCondition {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Duration unresolved
    pub duration_unresolved: Duration,
    /// Acknowledgment required
    pub acknowledgment_required: bool,
}

/// Notification template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    /// Template subject
    pub subject: String,
    /// Template body
    pub body: String,
    /// Template variables
    pub variables: Vec<String>,
}

/// Power analytics engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalytics {
    /// Analytics configuration
    pub config: AnalyticsConfig,
    /// Statistical models
    pub models: Vec<AnalyticsModel>,
    /// Analysis results
    pub results: AnalyticsResults,
    /// Trending data
    pub trends: TrendingData,
}

impl Default for PowerAnalytics {
    fn default() -> Self {
        Self {
            config: AnalyticsConfig::default(),
            models: Vec::new(),
            results: AnalyticsResults::default(),
            trends: TrendingData::default(),
        }
    }
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time analytics
    pub real_time_analytics: bool,
    /// Analysis interval
    pub analysis_interval: Duration,
    /// Trend detection sensitivity
    pub trend_sensitivity: f64,
    /// Anomaly detection enabled
    pub anomaly_detection: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            real_time_analytics: true,
            analysis_interval: Duration::from_secs(300), // 5 minutes
            trend_sensitivity: 0.05,
            anomaly_detection: true,
        }
    }
}

/// Analytics model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsModel {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f64,
}

/// Model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LinearRegression,
    TimeSeries,
    AnomalyDetection,
    Forecasting,
    Classification,
}

/// Analytics results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsResults {
    /// Power quality scores
    pub power_quality_scores: HashMap<String, f64>,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Anomaly detections
    pub anomalies: Vec<AnomalyDetection>,
    /// Predictions
    pub predictions: Vec<PowerPrediction>,
}

impl Default for AnalyticsResults {
    fn default() -> Self {
        Self {
            power_quality_scores: HashMap::new(),
            efficiency_metrics: EfficiencyMetrics::default(),
            anomalies: Vec::new(),
            predictions: Vec::new(),
        }
    }
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Overall efficiency
    pub overall_efficiency: f64,
    /// Power factor efficiency
    pub power_factor_efficiency: f64,
    /// Load efficiency
    pub load_efficiency: f64,
    /// Energy waste (kWh)
    pub energy_waste: f64,
}

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            overall_efficiency: 0.0,
            power_factor_efficiency: 0.0,
            load_efficiency: 0.0,
            energy_waste: 0.0,
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Confidence score
    pub confidence: f64,
    /// Affected meter
    pub affected_meter: String,
    /// Detection timestamp
    pub timestamp: SystemTime,
    /// Description
    pub description: String,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PowerSpike,
    UnexpectedLoad,
    EfficiencyDrop,
    VoltageAnomaly,
    FrequencyDeviation,
    PatternChange,
}

/// Power prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPrediction {
    /// Prediction type
    pub prediction_type: PredictionType,
    /// Predicted value
    pub predicted_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Prediction horizon
    pub horizon: Duration,
    /// Meter ID
    pub meter_id: String,
}

/// Prediction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionType {
    PowerDemand,
    EnergyConsumption,
    PeakLoad,
    PowerQuality,
    EquipmentFailure,
}

/// Trending data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingData {
    /// Power trends
    pub power_trends: Vec<TrendData>,
    /// Energy trends
    pub energy_trends: Vec<TrendData>,
    /// Quality trends
    pub quality_trends: Vec<TrendData>,
    /// Efficiency trends
    pub efficiency_trends: Vec<TrendData>,
}

impl Default for TrendingData {
    fn default() -> Self {
        Self {
            power_trends: Vec::new(),
            energy_trends: Vec::new(),
            quality_trends: Vec::new(),
            efficiency_trends: Vec::new(),
        }
    }
}

/// Trend data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    /// Parameter name
    pub parameter: String,
    /// Time series data
    pub data_points: Vec<(SystemTime, f64)>,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}
