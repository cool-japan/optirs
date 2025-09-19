// Thermal Management and Cooling Integration
//
// This module handles thermal management, cooling systems integration,
// and temperature-based power optimization for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

use super::super::device_layout::Position3D;

/// Thermal management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalManagement {
    /// Thermal configuration
    pub config: ThermalConfig,
    /// Temperature sensors
    pub sensors: HashMap<String, TemperatureSensor>,
    /// Cooling systems
    pub cooling_systems: HashMap<String, CoolingSystem>,
    /// Thermal zones
    pub thermal_zones: Vec<ThermalZone>,
    /// Thermal controller
    pub controller: ThermalController,
    /// Thermal analytics
    pub analytics: ThermalAnalytics,
}

impl Default for ThermalManagement {
    fn default() -> Self {
        Self {
            config: ThermalConfig::default(),
            sensors: HashMap::new(),
            cooling_systems: HashMap::new(),
            thermal_zones: Vec::new(),
            controller: ThermalController::default(),
            analytics: ThermalAnalytics::default(),
        }
    }
}

/// Thermal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Global temperature limits
    pub global_limits: TemperatureLimits,
    /// Thermal management mode
    pub management_mode: ThermalManagementMode,
    /// Control algorithm
    pub control_algorithm: ThermalControlAlgorithm,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Emergency procedures
    pub emergency_procedures: EmergencyProcedures,
    /// Power throttling settings
    pub power_throttling: PowerThrottlingSettings,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            global_limits: TemperatureLimits::default(),
            management_mode: ThermalManagementMode::Automatic,
            control_algorithm: ThermalControlAlgorithm::PID,
            monitoring_interval: Duration::from_secs(5),
            emergency_procedures: EmergencyProcedures::default(),
            power_throttling: PowerThrottlingSettings::default(),
        }
    }
}

/// Temperature limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureLimits {
    /// Warning temperature (Celsius)
    pub warning_temp: f64,
    /// Critical temperature (Celsius)
    pub critical_temp: f64,
    /// Emergency shutdown temperature (Celsius)
    pub shutdown_temp: f64,
    /// Optimal operating range
    pub optimal_range: (f64, f64),
    /// Maximum gradient (degrees per minute)
    pub max_gradient: f64,
}

impl Default for TemperatureLimits {
    fn default() -> Self {
        Self {
            warning_temp: 70.0,
            critical_temp: 85.0,
            shutdown_temp: 95.0,
            optimal_range: (20.0, 65.0),
            max_gradient: 10.0,
        }
    }
}

/// Thermal management modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalManagementMode {
    /// Automatic thermal management
    Automatic,
    /// Manual thermal management
    Manual,
    /// Performance-priority mode
    PerformancePriority,
    /// Efficiency-priority mode
    EfficiencyPriority,
    /// Silent operation mode
    SilentMode,
    /// Emergency cooling mode
    EmergencyMode,
}

/// Thermal control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalControlAlgorithm {
    /// PID controller
    PID,
    /// Fuzzy logic controller
    FuzzyLogic,
    /// Model predictive control
    ModelPredictive,
    /// Adaptive control
    Adaptive,
    /// Neural network control
    NeuralNetwork,
    /// Bang-bang control
    BangBang,
}

/// Emergency procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyProcedures {
    /// Emergency cooling activation
    pub emergency_cooling: bool,
    /// Power reduction levels
    pub power_reduction_levels: Vec<PowerReductionLevel>,
    /// Device shutdown sequence
    pub shutdown_sequence: ShutdownSequence,
    /// Notification procedures
    pub notification_procedures: NotificationProcedures,
}

impl Default for EmergencyProcedures {
    fn default() -> Self {
        Self {
            emergency_cooling: true,
            power_reduction_levels: vec![
                PowerReductionLevel {
                    trigger_temp: 75.0,
                    reduction_percentage: 20.0,
                    description: "Light throttling".to_string(),
                },
                PowerReductionLevel {
                    trigger_temp: 80.0,
                    reduction_percentage: 50.0,
                    description: "Heavy throttling".to_string(),
                },
                PowerReductionLevel {
                    trigger_temp: 90.0,
                    reduction_percentage: 80.0,
                    description: "Emergency throttling".to_string(),
                },
            ],
            shutdown_sequence: ShutdownSequence::default(),
            notification_procedures: NotificationProcedures::default(),
        }
    }
}

/// Power reduction level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerReductionLevel {
    /// Trigger temperature (Celsius)
    pub trigger_temp: f64,
    /// Power reduction percentage
    pub reduction_percentage: f64,
    /// Description
    pub description: String,
}

/// Shutdown sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownSequence {
    /// Enable graceful shutdown
    pub graceful_shutdown: bool,
    /// Shutdown priority order
    pub priority_order: Vec<ShutdownPriority>,
    /// Shutdown delays
    pub shutdown_delays: HashMap<String, Duration>,
    /// Force shutdown temperature
    pub force_shutdown_temp: f64,
}

impl Default for ShutdownSequence {
    fn default() -> Self {
        Self {
            graceful_shutdown: true,
            priority_order: vec![
                ShutdownPriority::NonCriticalWorkloads,
                ShutdownPriority::BackgroundTasks,
                ShutdownPriority::NormalWorkloads,
                ShutdownPriority::HighPriorityWorkloads,
                ShutdownPriority::CriticalSystems,
            ],
            shutdown_delays: HashMap::new(),
            force_shutdown_temp: 100.0,
        }
    }
}

/// Shutdown priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShutdownPriority {
    NonCriticalWorkloads,
    BackgroundTasks,
    NormalWorkloads,
    HighPriorityWorkloads,
    CriticalSystems,
    EmergencyOnly,
}

/// Notification procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationProcedures {
    /// Email notifications
    pub email_enabled: bool,
    /// SMS notifications
    pub sms_enabled: bool,
    /// SNMP traps
    pub snmp_enabled: bool,
    /// Syslog messages
    pub syslog_enabled: bool,
    /// Notification recipients
    pub recipients: Vec<NotificationRecipient>,
}

impl Default for NotificationProcedures {
    fn default() -> Self {
        Self {
            email_enabled: true,
            sms_enabled: false,
            snmp_enabled: true,
            syslog_enabled: true,
            recipients: Vec::new(),
        }
    }
}

/// Notification recipient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRecipient {
    /// Recipient name
    pub name: String,
    /// Contact method
    pub contact_method: ContactMethod,
    /// Contact details
    pub contact_details: String,
    /// Severity threshold
    pub severity_threshold: ThermalSeverity,
}

/// Contact methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContactMethod {
    Email,
    SMS,
    Phone,
    Slack,
    Teams,
    WebHook,
}

/// Thermal severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Power throttling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerThrottlingSettings {
    /// Enable power throttling
    pub enabled: bool,
    /// Throttling algorithm
    pub algorithm: ThrottlingAlgorithm,
    /// Throttling aggressiveness
    pub aggressiveness: f64,
    /// Recovery settings
    pub recovery_settings: ThrottlingRecoverySettings,
    /// Device-specific settings
    pub device_settings: HashMap<String, DeviceThrottlingSettings>,
}

impl Default for PowerThrottlingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: ThrottlingAlgorithm::Proportional,
            aggressiveness: 0.7,
            recovery_settings: ThrottlingRecoverySettings::default(),
            device_settings: HashMap::new(),
        }
    }
}

/// Throttling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThrottlingAlgorithm {
    /// Linear throttling
    Linear,
    /// Proportional throttling
    Proportional,
    /// Exponential throttling
    Exponential,
    /// Stepped throttling
    Stepped,
    /// Adaptive throttling
    Adaptive,
}

/// Throttling recovery settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottlingRecoverySettings {
    /// Recovery temperature threshold
    pub recovery_temp_threshold: f64,
    /// Recovery delay
    pub recovery_delay: Duration,
    /// Gradual recovery enabled
    pub gradual_recovery: bool,
    /// Recovery rate
    pub recovery_rate: f64,
}

impl Default for ThrottlingRecoverySettings {
    fn default() -> Self {
        Self {
            recovery_temp_threshold: 5.0, // 5 degrees below trigger
            recovery_delay: Duration::from_secs(30),
            gradual_recovery: true,
            recovery_rate: 0.1, // 10% per step
        }
    }
}

/// Device-specific throttling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceThrottlingSettings {
    /// Device-specific temperature thresholds
    pub temp_thresholds: TemperatureLimits,
    /// Throttling sensitivity
    pub sensitivity: f64,
    /// Maximum throttling percentage
    pub max_throttling: f64,
    /// Minimum power level
    pub min_power_level: f64,
}

impl Default for DeviceThrottlingSettings {
    fn default() -> Self {
        Self {
            temp_thresholds: TemperatureLimits::default(),
            sensitivity: 1.0,
            max_throttling: 80.0,
            min_power_level: 20.0,
        }
    }
}

/// Temperature sensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureSensor {
    /// Sensor identifier
    pub sensor_id: String,
    /// Sensor type
    pub sensor_type: SensorType,
    /// Sensor location
    pub location: Position3D,
    /// Current temperature reading
    pub current_temperature: f64,
    /// Sensor specifications
    pub specifications: SensorSpecifications,
    /// Sensor status
    pub status: SensorStatus,
    /// Calibration information
    pub calibration: SensorCalibration,
    /// Historical readings
    pub history: SensorHistory,
}

impl Default for TemperatureSensor {
    fn default() -> Self {
        Self {
            sensor_id: String::new(),
            sensor_type: SensorType::Digital,
            location: Position3D::default(),
            current_temperature: 25.0,
            specifications: SensorSpecifications::default(),
            status: SensorStatus::Normal,
            calibration: SensorCalibration::default(),
            history: SensorHistory::default(),
        }
    }
}

/// Sensor types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorType {
    /// Digital temperature sensor
    Digital,
    /// Analog thermistor
    Thermistor,
    /// Thermocouple
    Thermocouple,
    /// RTD (Resistance Temperature Detector)
    RTD,
    /// Infrared sensor
    Infrared,
    /// Semiconductor sensor
    Semiconductor,
}

/// Sensor specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSpecifications {
    /// Measurement range (Celsius)
    pub measurement_range: (f64, f64),
    /// Accuracy (degrees)
    pub accuracy: f64,
    /// Resolution (degrees)
    pub resolution: f64,
    /// Response time (seconds)
    pub response_time: f64,
    /// Sampling rate (Hz)
    pub sampling_rate: f64,
    /// Operating voltage range
    pub voltage_range: (f64, f64),
    /// Interface type
    pub interface_type: SensorInterface,
}

impl Default for SensorSpecifications {
    fn default() -> Self {
        Self {
            measurement_range: (-40.0, 125.0),
            accuracy: 0.5,
            resolution: 0.1,
            response_time: 1.0,
            sampling_rate: 1.0,
            voltage_range: (3.0, 5.0),
            interface_type: SensorInterface::I2C,
        }
    }
}

/// Sensor interface types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorInterface {
    I2C,
    SPI,
    OneWire,
    Analog,
    UART,
    Modbus,
    CAN,
}

/// Sensor status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SensorStatus {
    Normal,
    Warning,
    Error,
    Calibrating,
    Offline,
    MaintenanceMode,
}

/// Sensor calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorCalibration {
    /// Last calibration date
    pub last_calibration: SystemTime,
    /// Calibration offset
    pub offset: f64,
    /// Calibration scale
    pub scale: f64,
    /// Calibration accuracy
    pub calibration_accuracy: f64,
    /// Next calibration due
    pub next_calibration_due: SystemTime,
}

impl Default for SensorCalibration {
    fn default() -> Self {
        Self {
            last_calibration: SystemTime::now(),
            offset: 0.0,
            scale: 1.0,
            calibration_accuracy: 0.1,
            next_calibration_due: SystemTime::now(),
        }
    }
}

/// Sensor history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorHistory {
    /// Historical temperature readings
    pub readings: Vec<TemperatureReading>,
    /// Statistics
    pub statistics: TemperatureStatistics,
    /// Trend data
    pub trends: TemperatureTrends,
}

impl Default for SensorHistory {
    fn default() -> Self {
        Self {
            readings: Vec::new(),
            statistics: TemperatureStatistics::default(),
            trends: TemperatureTrends::default(),
        }
    }
}

/// Temperature reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureReading {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Temperature value (Celsius)
    pub temperature: f64,
    /// Data quality
    pub quality: DataQuality,
}

/// Data quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataQuality {
    Good,
    Questionable,
    Poor,
    Invalid,
}

/// Temperature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureStatistics {
    /// Minimum temperature
    pub min_temp: f64,
    /// Maximum temperature
    pub max_temp: f64,
    /// Average temperature
    pub avg_temp: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// Number of readings
    pub reading_count: u64,
}

impl Default for TemperatureStatistics {
    fn default() -> Self {
        Self {
            min_temp: f64::MAX,
            max_temp: f64::MIN,
            avg_temp: 0.0,
            std_deviation: 0.0,
            reading_count: 0,
        }
    }
}

/// Temperature trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureTrends {
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength
    pub trend_strength: f64,
    /// Rate of change (degrees per hour)
    pub rate_of_change: f64,
    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

impl Default for TemperatureTrends {
    fn default() -> Self {
        Self {
            trend_direction: TrendDirection::Stable,
            trend_strength: 0.0,
            rate_of_change: 0.0,
            seasonal_patterns: Vec::new(),
        }
    }
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

/// Seasonal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern name
    pub name: String,
    /// Time period
    pub period: Duration,
    /// Amplitude
    pub amplitude: f64,
    /// Phase offset
    pub phase_offset: f64,
}

/// Cooling system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingSystem {
    /// System identifier
    pub system_id: String,
    /// Cooling type
    pub cooling_type: CoolingType,
    /// System specifications
    pub specifications: CoolingSpecifications,
    /// Current operating state
    pub operating_state: CoolingOperatingState,
    /// Performance metrics
    pub performance: CoolingPerformance,
    /// Control settings
    pub control_settings: CoolingControlSettings,
    /// Maintenance information
    pub maintenance: CoolingMaintenance,
}

impl Default for CoolingSystem {
    fn default() -> Self {
        Self {
            system_id: String::new(),
            cooling_type: CoolingType::AirCooling,
            specifications: CoolingSpecifications::default(),
            operating_state: CoolingOperatingState::default(),
            performance: CoolingPerformance::default(),
            control_settings: CoolingControlSettings::default(),
            maintenance: CoolingMaintenance::default(),
        }
    }
}

/// Cooling types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingType {
    /// Air cooling (fans)
    AirCooling,
    /// Liquid cooling (pumps, radiators)
    LiquidCooling,
    /// Immersion cooling
    ImmersionCooling,
    /// Thermoelectric cooling (Peltier)
    ThermoelectricCooling,
    /// Phase change cooling
    PhaseChangeCooling,
    /// Hybrid cooling system
    HybridCooling,
}

/// Cooling specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingSpecifications {
    /// Maximum cooling capacity (watts)
    pub max_cooling_capacity: f64,
    /// Operating temperature range
    pub operating_temp_range: (f64, f64),
    /// Power consumption (watts)
    pub power_consumption: f64,
    /// Noise level (dBA)
    pub noise_level: f64,
    /// Airflow capacity (CFM)
    pub airflow_capacity: Option<f64>,
    /// Coolant flow rate (L/min)
    pub coolant_flow_rate: Option<f64>,
    /// Efficiency rating
    pub efficiency_rating: f64,
}

impl Default for CoolingSpecifications {
    fn default() -> Self {
        Self {
            max_cooling_capacity: 1000.0,
            operating_temp_range: (0.0, 50.0),
            power_consumption: 100.0,
            noise_level: 35.0,
            airflow_capacity: Some(200.0),
            coolant_flow_rate: None,
            efficiency_rating: 0.8,
        }
    }
}

/// Cooling operating state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingOperatingState {
    /// System status
    pub status: CoolingStatus,
    /// Current cooling output (percentage)
    pub current_output: f64,
    /// Fan speeds (RPM)
    pub fan_speeds: Vec<f64>,
    /// Pump speeds (RPM)
    pub pump_speeds: Vec<f64>,
    /// Coolant temperatures
    pub coolant_temperatures: Vec<f64>,
    /// Operating mode
    pub operating_mode: CoolingMode,
}

impl Default for CoolingOperatingState {
    fn default() -> Self {
        Self {
            status: CoolingStatus::Normal,
            current_output: 0.0,
            fan_speeds: Vec::new(),
            pump_speeds: Vec::new(),
            coolant_temperatures: Vec::new(),
            operating_mode: CoolingMode::Automatic,
        }
    }
}

/// Cooling status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoolingStatus {
    Normal,
    Warning,
    Critical,
    Fault,
    Maintenance,
    Offline,
}

/// Cooling modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingMode {
    /// Automatic temperature-based control
    Automatic,
    /// Manual fixed output
    Manual,
    /// Performance mode (maximum cooling)
    Performance,
    /// Quiet mode (minimum noise)
    Quiet,
    /// Eco mode (energy efficient)
    Eco,
    /// Emergency mode (maximum output)
    Emergency,
}

/// Cooling performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingPerformance {
    /// Current cooling efficiency
    pub efficiency: f64,
    /// Heat removal rate (watts)
    pub heat_removal_rate: f64,
    /// Power consumption (watts)
    pub power_consumption: f64,
    /// Coefficient of performance (COP)
    pub coefficient_of_performance: f64,
    /// Operating hours
    pub operating_hours: f64,
    /// Performance degradation
    pub performance_degradation: f64,
}

impl Default for CoolingPerformance {
    fn default() -> Self {
        Self {
            efficiency: 0.8,
            heat_removal_rate: 0.0,
            power_consumption: 0.0,
            coefficient_of_performance: 3.0,
            operating_hours: 0.0,
            performance_degradation: 0.0,
        }
    }
}

/// Cooling control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingControlSettings {
    /// Control mode
    pub control_mode: CoolingControlMode,
    /// Target temperature
    pub target_temperature: f64,
    /// Control parameters
    pub control_parameters: ControlParameters,
    /// Safety limits
    pub safety_limits: CoolingSafetyLimits,
}

impl Default for CoolingControlSettings {
    fn default() -> Self {
        Self {
            control_mode: CoolingControlMode::TemperatureBased,
            target_temperature: 45.0,
            control_parameters: ControlParameters::default(),
            safety_limits: CoolingSafetyLimits::default(),
        }
    }
}

/// Cooling control modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingControlMode {
    TemperatureBased,
    LoadBased,
    PredictiveBased,
    HybridControl,
    ManualControl,
}

/// Control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlParameters {
    /// PID parameters
    pub pid: PIDParameters,
    /// Response time settings
    pub response_time: ResponseTimeSettings,
    /// Hysteresis settings
    pub hysteresis: HysteresisSettings,
}

impl Default for ControlParameters {
    fn default() -> Self {
        Self {
            pid: PIDParameters::default(),
            response_time: ResponseTimeSettings::default(),
            hysteresis: HysteresisSettings::default(),
        }
    }
}

/// PID parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDParameters {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Integral windup limit
    pub windup_limit: f64,
}

impl Default for PIDParameters {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.01,
            windup_limit: 100.0,
        }
    }
}

/// Response time settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeSettings {
    /// Ramp-up time
    pub ramp_up_time: Duration,
    /// Ramp-down time
    pub ramp_down_time: Duration,
    /// Settling time
    pub settling_time: Duration,
}

impl Default for ResponseTimeSettings {
    fn default() -> Self {
        Self {
            ramp_up_time: Duration::from_secs(30),
            ramp_down_time: Duration::from_secs(60),
            settling_time: Duration::from_secs(120),
        }
    }
}

/// Hysteresis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisSettings {
    /// Temperature hysteresis (degrees)
    pub temperature_hysteresis: f64,
    /// Output hysteresis (percentage)
    pub output_hysteresis: f64,
    /// Enable hysteresis
    pub enabled: bool,
}

impl Default for HysteresisSettings {
    fn default() -> Self {
        Self {
            temperature_hysteresis: 2.0,
            output_hysteresis: 5.0,
            enabled: true,
        }
    }
}

/// Cooling safety limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingSafetyLimits {
    /// Maximum output percentage
    pub max_output: f64,
    /// Minimum output percentage
    pub min_output: f64,
    /// Maximum temperature differential
    pub max_temp_differential: f64,
    /// Emergency shutdown temperature
    pub emergency_shutdown_temp: f64,
}

impl Default for CoolingSafetyLimits {
    fn default() -> Self {
        Self {
            max_output: 100.0,
            min_output: 0.0,
            max_temp_differential: 50.0,
            emergency_shutdown_temp: 100.0,
        }
    }
}

/// Cooling maintenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingMaintenance {
    /// Last maintenance date
    pub last_maintenance: SystemTime,
    /// Next maintenance due
    pub next_maintenance_due: SystemTime,
    /// Maintenance history
    pub maintenance_history: Vec<MaintenanceRecord>,
    /// Filter status
    pub filter_status: FilterStatus,
    /// Coolant status
    pub coolant_status: Option<CoolantStatus>,
}

impl Default for CoolingMaintenance {
    fn default() -> Self {
        Self {
            last_maintenance: SystemTime::now(),
            next_maintenance_due: SystemTime::now(),
            maintenance_history: Vec::new(),
            filter_status: FilterStatus::default(),
            coolant_status: None,
        }
    }
}

/// Maintenance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceRecord {
    /// Maintenance date
    pub date: SystemTime,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Description
    pub description: String,
    /// Technician
    pub technician: String,
    /// Parts replaced
    pub parts_replaced: Vec<String>,
    /// Cost
    pub cost: f64,
}

/// Maintenance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
    Preventive,
    Corrective,
    Predictive,
    Emergency,
    Cleaning,
    Calibration,
    Upgrade,
}

/// Filter status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStatus {
    /// Filter type
    pub filter_type: FilterType,
    /// Cleanliness level (percentage)
    pub cleanliness_level: f64,
    /// Replacement due date
    pub replacement_due: SystemTime,
    /// Operating hours since replacement
    pub operating_hours: f64,
}

impl Default for FilterStatus {
    fn default() -> Self {
        Self {
            filter_type: FilterType::HEPA,
            cleanliness_level: 100.0,
            replacement_due: SystemTime::now(),
            operating_hours: 0.0,
        }
    }
}

/// Filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    HEPA,
    PreFilter,
    CarbonFilter,
    ElectrostaticFilter,
    Custom(String),
}

/// Coolant status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolantStatus {
    /// Coolant type
    pub coolant_type: CoolantType,
    /// Coolant level (percentage)
    pub level: f64,
    /// Coolant temperature
    pub temperature: f64,
    /// Coolant quality
    pub quality: CoolantQuality,
    /// pH level
    pub ph_level: f64,
    /// Additive concentrations
    pub additive_concentrations: HashMap<String, f64>,
}

/// Coolant types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolantType {
    Water,
    EthyleneGlycol,
    PropyleneGlycol,
    Fluorinert,
    Mineral_Oil,
    Custom(String),
}

/// Coolant quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolantQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Replace,
}

/// Thermal zone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalZone {
    /// Zone identifier
    pub zone_id: String,
    /// Zone boundaries
    pub boundaries: ZoneBoundaries,
    /// Zone sensors
    pub sensors: Vec<String>,
    /// Zone cooling systems
    pub cooling_systems: Vec<String>,
    /// Zone devices
    pub devices: Vec<String>,
    /// Zone thermal properties
    pub thermal_properties: ThermalProperties,
    /// Zone control strategy
    pub control_strategy: ZoneControlStrategy,
}

impl Default for ThermalZone {
    fn default() -> Self {
        Self {
            zone_id: String::new(),
            boundaries: ZoneBoundaries::default(),
            sensors: Vec::new(),
            cooling_systems: Vec::new(),
            devices: Vec::new(),
            thermal_properties: ThermalProperties::default(),
            control_strategy: ZoneControlStrategy::default(),
        }
    }
}

/// Zone boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneBoundaries {
    /// Minimum position
    pub min_position: Position3D,
    /// Maximum position
    pub max_position: Position3D,
    /// Zone shape
    pub shape: ZoneShape,
}

impl Default for ZoneBoundaries {
    fn default() -> Self {
        Self {
            min_position: Position3D::default(),
            max_position: Position3D::default(),
            shape: ZoneShape::Rectangular,
        }
    }
}

/// Zone shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZoneShape {
    Rectangular,
    Cylindrical,
    Spherical,
    Irregular,
}

/// Thermal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalProperties {
    /// Thermal mass (J/K)
    pub thermal_mass: f64,
    /// Thermal resistance (K/W)
    pub thermal_resistance: f64,
    /// Heat capacity (J/kg·K)
    pub heat_capacity: f64,
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
    /// Convection coefficient (W/m²·K)
    pub convection_coefficient: f64,
    /// Emissivity
    pub emissivity: f64,
}

impl Default for ThermalProperties {
    fn default() -> Self {
        Self {
            thermal_mass: 1000.0,
            thermal_resistance: 0.1,
            heat_capacity: 1000.0,
            thermal_conductivity: 1.0,
            convection_coefficient: 10.0,
            emissivity: 0.9,
        }
    }
}

/// Zone control strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneControlStrategy {
    /// Control objective
    pub objective: ControlObjective,
    /// Priority level
    pub priority: ZonePriority,
    /// Temperature targets
    pub temperature_targets: TemperatureTargets,
    /// Load balancing
    pub load_balancing: bool,
    /// Adaptive control
    pub adaptive_control: bool,
}

impl Default for ZoneControlStrategy {
    fn default() -> Self {
        Self {
            objective: ControlObjective::TemperatureControl,
            priority: ZonePriority::Normal,
            temperature_targets: TemperatureTargets::default(),
            load_balancing: true,
            adaptive_control: true,
        }
    }
}

/// Control objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlObjective {
    TemperatureControl,
    EnergyEfficiency,
    UniformityControl,
    HotspotPrevention,
    NoiseMinimization,
}

/// Zone priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZonePriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Temperature targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureTargets {
    /// Target temperature
    pub target: f64,
    /// Acceptable range
    pub acceptable_range: (f64, f64),
    /// Setpoint scheduling
    pub setpoint_schedule: Vec<SetpointSchedule>,
}

impl Default for TemperatureTargets {
    fn default() -> Self {
        Self {
            target: 45.0,
            acceptable_range: (40.0, 50.0),
            setpoint_schedule: Vec::new(),
        }
    }
}

/// Setpoint schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetpointSchedule {
    /// Time of day (hour)
    pub hour: u8,
    /// Temperature setpoint
    pub setpoint: f64,
    /// Days of week
    pub days_of_week: Vec<u8>,
}

/// Thermal controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalController {
    /// Controller configuration
    pub config: ThermalControllerConfig,
    /// Control loops
    pub control_loops: Vec<ControlLoop>,
    /// Controller state
    pub state: ControllerState,
    /// Performance metrics
    pub performance: ControllerPerformance,
}

impl Default for ThermalController {
    fn default() -> Self {
        Self {
            config: ThermalControllerConfig::default(),
            control_loops: Vec::new(),
            state: ControllerState::default(),
            performance: ControllerPerformance::default(),
        }
    }
}

/// Thermal controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalControllerConfig {
    /// Control strategy
    pub strategy: ThermalControlStrategy,
    /// Update frequency
    pub update_frequency: Duration,
    /// Predictive horizon
    pub predictive_horizon: Duration,
    /// Optimization enabled
    pub optimization_enabled: bool,
    /// Fault tolerance
    pub fault_tolerance: FaultToleranceSettings,
}

impl Default for ThermalControllerConfig {
    fn default() -> Self {
        Self {
            strategy: ThermalControlStrategy::Model_Predictive,
            update_frequency: Duration::from_secs(5),
            predictive_horizon: Duration::from_secs(300),
            optimization_enabled: true,
            fault_tolerance: FaultToleranceSettings::default(),
        }
    }
}

/// Thermal control strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalControlStrategy {
    Reactive,
    Predictive,
    Adaptive,
    Model_Predictive,
    Fuzzy_Logic,
    Neural_Network,
}

/// Fault tolerance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceSettings {
    /// Sensor fault detection
    pub sensor_fault_detection: bool,
    /// Actuator fault detection
    pub actuator_fault_detection: bool,
    /// Backup control strategies
    pub backup_strategies: Vec<ThermalControlStrategy>,
    /// Graceful degradation
    pub graceful_degradation: bool,
}

impl Default for FaultToleranceSettings {
    fn default() -> Self {
        Self {
            sensor_fault_detection: true,
            actuator_fault_detection: true,
            backup_strategies: vec![ThermalControlStrategy::Reactive],
            graceful_degradation: true,
        }
    }
}

/// Control loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlLoop {
    /// Loop identifier
    pub loop_id: String,
    /// Controlled zone
    pub zone_id: String,
    /// Input sensors
    pub input_sensors: Vec<String>,
    /// Output actuators
    pub output_actuators: Vec<String>,
    /// Controller type
    pub controller_type: ControllerType,
    /// Loop parameters
    pub parameters: LoopParameters,
    /// Loop status
    pub status: LoopStatus,
}

/// Controller types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerType {
    PID,
    PI,
    PD,
    Fuzzy,
    Neural,
    Model_Predictive,
    Adaptive,
}

/// Loop parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopParameters {
    /// Setpoint
    pub setpoint: f64,
    /// Output limits
    pub output_limits: (f64, f64),
    /// Rate limits
    pub rate_limits: (f64, f64),
    /// Deadband
    pub deadband: f64,
}

impl Default for LoopParameters {
    fn default() -> Self {
        Self {
            setpoint: 45.0,
            output_limits: (0.0, 100.0),
            rate_limits: (-10.0, 10.0),
            deadband: 1.0,
        }
    }
}

/// Loop status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopStatus {
    Active,
    Inactive,
    Manual,
    Fault,
    Tuning,
}

/// Controller state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerState {
    /// Current mode
    pub mode: ControllerMode,
    /// System health
    pub health: ControllerHealth,
    /// Active alarms
    pub active_alarms: Vec<ThermalAlarm>,
    /// Last update timestamp
    pub last_update: SystemTime,
}

impl Default for ControllerState {
    fn default() -> Self {
        Self {
            mode: ControllerMode::Automatic,
            health: ControllerHealth::Good,
            active_alarms: Vec::new(),
            last_update: SystemTime::now(),
        }
    }
}

/// Controller modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerMode {
    Automatic,
    Manual,
    Tuning,
    Diagnostic,
    Emergency,
    Maintenance,
}

/// Controller health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControllerHealth {
    Good,
    Degraded,
    Poor,
    Failed,
}

/// Thermal alarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAlarm {
    /// Alarm identifier
    pub alarm_id: String,
    /// Alarm type
    pub alarm_type: ThermalAlarmType,
    /// Severity
    pub severity: ThermalSeverity,
    /// Source location
    pub source: String,
    /// Alarm message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Acknowledged
    pub acknowledged: bool,
}

/// Thermal alarm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalAlarmType {
    HighTemperature,
    LowTemperature,
    RapidTemperatureRise,
    SensorFailure,
    CoolingSystemFailure,
    ControllerFailure,
    ZoneImbalance,
    PowerThrottling,
}

/// Controller performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerPerformance {
    /// Control accuracy
    pub accuracy: f64,
    /// Response time
    pub response_time: Duration,
    /// Stability metric
    pub stability: f64,
    /// Energy efficiency
    pub energy_efficiency: f64,
    /// Uptime
    pub uptime: f64,
}

impl Default for ControllerPerformance {
    fn default() -> Self {
        Self {
            accuracy: 0.95,
            response_time: Duration::from_secs(30),
            stability: 0.9,
            energy_efficiency: 0.8,
            uptime: 0.99,
        }
    }
}

/// Thermal analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAnalytics {
    /// Analytics configuration
    pub config: ThermalAnalyticsConfig,
    /// Thermal models
    pub models: Vec<ThermalModel>,
    /// Predictive analytics
    pub predictive: PredictiveAnalytics,
    /// Optimization results
    pub optimization: ThermalOptimizationResults,
}

impl Default for ThermalAnalytics {
    fn default() -> Self {
        Self {
            config: ThermalAnalyticsConfig::default(),
            models: Vec::new(),
            predictive: PredictiveAnalytics::default(),
            optimization: ThermalOptimizationResults::default(),
        }
    }
}

/// Thermal analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAnalyticsConfig {
    /// Enable real-time analytics
    pub real_time_enabled: bool,
    /// Analytics interval
    pub analytics_interval: Duration,
    /// Model update frequency
    pub model_update_frequency: Duration,
    /// Prediction horizon
    pub prediction_horizon: Duration,
}

impl Default for ThermalAnalyticsConfig {
    fn default() -> Self {
        Self {
            real_time_enabled: true,
            analytics_interval: Duration::from_secs(60),
            model_update_frequency: Duration::from_secs(3600),
            prediction_horizon: Duration::from_secs(1800),
        }
    }
}

/// Thermal model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalModel {
    /// Model identifier
    pub model_id: String,
    /// Model type
    pub model_type: ThermalModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f64,
    /// Last update
    pub last_update: SystemTime,
}

/// Thermal model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalModelType {
    LinearRegression,
    NonlinearRegression,
    NeuralNetwork,
    Physics_Based,
    Hybrid,
    Statistical,
}

/// Predictive analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnalytics {
    /// Temperature predictions
    pub temperature_predictions: Vec<TemperaturePrediction>,
    /// Hotspot predictions
    pub hotspot_predictions: Vec<HotspotPrediction>,
    /// Cooling load predictions
    pub cooling_load_predictions: Vec<CoolingLoadPrediction>,
    /// Failure predictions
    pub failure_predictions: Vec<FailurePrediction>,
}

impl Default for PredictiveAnalytics {
    fn default() -> Self {
        Self {
            temperature_predictions: Vec::new(),
            hotspot_predictions: Vec::new(),
            cooling_load_predictions: Vec::new(),
            failure_predictions: Vec::new(),
        }
    }
}

/// Temperature prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperaturePrediction {
    /// Location
    pub location: String,
    /// Predicted temperature
    pub predicted_temperature: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Prediction time
    pub prediction_time: SystemTime,
    /// Prediction horizon
    pub horizon: Duration,
}

/// Hotspot prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotspotPrediction {
    /// Predicted hotspot location
    pub location: Position3D,
    /// Predicted maximum temperature
    pub max_temperature: f64,
    /// Probability of occurrence
    pub probability: f64,
    /// Time to occurrence
    pub time_to_occurrence: Duration,
}

/// Cooling load prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingLoadPrediction {
    /// Zone identifier
    pub zone_id: String,
    /// Predicted cooling load (watts)
    pub predicted_load: f64,
    /// Load factor
    pub load_factor: f64,
    /// Prediction confidence
    pub confidence: f64,
}

/// Failure prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePrediction {
    /// Component identifier
    pub component_id: String,
    /// Failure type
    pub failure_type: FailureType,
    /// Probability of failure
    pub probability: f64,
    /// Time to failure
    pub time_to_failure: Duration,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Failure types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    SensorFailure,
    FanFailure,
    PumpFailure,
    ThermalOverload,
    ControllerFailure,
    CoolantLeak,
    FilterClogging,
}

/// Thermal optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalOptimizationResults {
    /// Optimization objectives achieved
    pub objectives_achieved: Vec<OptimizationObjective>,
    /// Energy savings (percentage)
    pub energy_savings: f64,
    /// Temperature uniformity improvement
    pub uniformity_improvement: f64,
    /// Performance improvement
    pub performance_improvement: f64,
    /// Cost savings
    pub cost_savings: f64,
}

impl Default for ThermalOptimizationResults {
    fn default() -> Self {
        Self {
            objectives_achieved: Vec::new(),
            energy_savings: 0.0,
            uniformity_improvement: 0.0,
            performance_improvement: 0.0,
            cost_savings: 0.0,
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MinimizeEnergyConsumption,
    MaximizeTemperatureUniformity,
    MinimizeHotspots,
    MaximizeSystemReliability,
    MinimizeNoise,
    MaximizeEfficiency,
}
