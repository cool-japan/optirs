// Power Supply Management
//
// This module handles power supply units (PSUs), their specifications, metrics,
// and management functionality for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::super::device_layout::Position3D;

/// Power supply unit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupply {
    /// PSU identifier
    pub psu_id: String,
    /// Power capacity (watts)
    pub capacity: f64,
    /// Current load (watts)
    pub current_load: f64,
    /// Efficiency rating (0.0 to 1.0)
    pub efficiency: f64,
    /// PSU status
    pub status: PowerSupplyStatus,
    /// PSU location
    pub location: Option<Position3D>,
    /// PSU specifications
    pub specifications: PowerSupplySpecifications,
    /// PSU metrics
    pub metrics: PowerSupplyMetrics,
    /// PSU configuration
    pub config: PowerSupplyConfig,
    /// PSU history
    pub history: PowerSupplyHistory,
}

impl Default for PowerSupply {
    fn default() -> Self {
        Self {
            psu_id: String::new(),
            capacity: 0.0,
            current_load: 0.0,
            efficiency: 0.9,
            status: PowerSupplyStatus::Normal,
            location: None,
            specifications: PowerSupplySpecifications::default(),
            metrics: PowerSupplyMetrics::default(),
            config: PowerSupplyConfig::default(),
            history: PowerSupplyHistory::default(),
        }
    }
}

/// Power supply status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerSupplyStatus {
    /// PSU is operating normally
    Normal,
    /// PSU is overloaded
    Overloaded,
    /// PSU has failed
    Failed,
    /// PSU is in maintenance mode
    Maintenance,
    /// PSU is starting up
    Starting,
    /// PSU is shutting down
    ShuttingDown,
    /// PSU is in standby mode
    Standby,
    /// PSU is in power save mode
    PowerSave,
    /// PSU has warning conditions
    Warning,
    /// PSU is critical
    Critical,
}

/// Power supply specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupplySpecifications {
    /// Rated power output (watts)
    pub rated_power: f64,
    /// Input voltage range (volts)
    pub input_voltage_range: (f64, f64),
    /// Output voltage (volts)
    pub output_voltage: f64,
    /// Efficiency curve
    pub efficiency_curve: Vec<(f64, f64)>, // (load_percentage, efficiency)
    /// Operating temperature range (Celsius)
    pub operating_temp_range: (f64, f64),
    /// Mean time between failures (hours)
    pub mtbf: f64,
    /// Power factor correction
    pub power_factor_correction: bool,
    /// Redundancy level
    pub redundancy_level: RedundancyLevel,
    /// Form factor
    pub form_factor: FormFactor,
    /// Certification standards
    pub certifications: Vec<String>,
    /// Cooling requirements
    pub cooling_requirements: CoolingRequirements,
}

impl Default for PowerSupplySpecifications {
    fn default() -> Self {
        Self {
            rated_power: 1000.0,
            input_voltage_range: (100.0, 240.0),
            output_voltage: 12.0,
            efficiency_curve: vec![
                (10.0, 0.85),
                (20.0, 0.90),
                (50.0, 0.94),
                (80.0, 0.92),
                (100.0, 0.89),
            ],
            operating_temp_range: (0.0, 50.0),
            mtbf: 100000.0,
            power_factor_correction: true,
            redundancy_level: RedundancyLevel::None,
            form_factor: FormFactor::ATX,
            certifications: vec!["80Plus Gold".to_string()],
            cooling_requirements: CoolingRequirements::default(),
        }
    }
}

/// Redundancy levels for power supplies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyLevel {
    /// No redundancy
    None,
    /// N+1 redundancy
    NPlusOne,
    /// N+N redundancy
    NPlusN,
    /// 2N redundancy
    TwoN,
    /// 2N+1 redundancy
    TwoNPlusOne,
}

/// Form factors for power supplies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormFactor {
    ATX,
    SFX,
    TFX,
    FlexATX,
    Custom(String),
}

/// Cooling requirements for power supplies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingRequirements {
    /// Required airflow (CFM)
    pub airflow_cfm: f64,
    /// Fan count
    pub fan_count: usize,
    /// Fan specifications
    pub fan_specs: Vec<FanSpecification>,
    /// Thermal design power (watts)
    pub tdp: f64,
    /// Maximum operating temperature (Celsius)
    pub max_operating_temp: f64,
}

impl Default for CoolingRequirements {
    fn default() -> Self {
        Self {
            airflow_cfm: 50.0,
            fan_count: 1,
            fan_specs: vec![FanSpecification::default()],
            tdp: 100.0,
            max_operating_temp: 60.0,
        }
    }
}

/// Fan specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanSpecification {
    /// Fan diameter (mm)
    pub diameter_mm: f64,
    /// Maximum RPM
    pub max_rpm: f64,
    /// Noise level (dBA)
    pub noise_level_dba: f64,
    /// Bearing type
    pub bearing_type: BearingType,
    /// Lifespan (hours)
    pub lifespan_hours: f64,
}

impl Default for FanSpecification {
    fn default() -> Self {
        Self {
            diameter_mm: 80.0,
            max_rpm: 3000.0,
            noise_level_dba: 35.0,
            bearing_type: BearingType::BallBearing,
            lifespan_hours: 50000.0,
        }
    }
}

/// Bearing types for fans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BearingType {
    BallBearing,
    SleeveBearing,
    HydraulicBearing,
    MagneticBearing,
    FluidDynamic,
}

/// Power supply metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupplyMetrics {
    /// Input power (watts)
    pub input_power: f64,
    /// Output power (watts)
    pub output_power: f64,
    /// Current efficiency (0.0 to 1.0)
    pub current_efficiency: f64,
    /// Power factor
    pub power_factor: f64,
    /// Temperature (Celsius)
    pub temperature: f64,
    /// Fan speed (RPM)
    pub fan_speed: Option<f64>,
    /// Lifetime operating hours
    pub operating_hours: f64,
    /// Input voltage (volts)
    pub input_voltage: f64,
    /// Input current (amperes)
    pub input_current: f64,
    /// Output voltage (volts)
    pub output_voltage: f64,
    /// Output current (amperes)
    pub output_current: f64,
    /// Power factor correction status
    pub pfc_status: bool,
    /// Load percentage
    pub load_percentage: f64,
    /// Estimated remaining life (hours)
    pub remaining_life_hours: f64,
}

impl Default for PowerSupplyMetrics {
    fn default() -> Self {
        Self {
            input_power: 0.0,
            output_power: 0.0,
            current_efficiency: 0.9,
            power_factor: 0.95,
            temperature: 25.0,
            fan_speed: Some(1000.0),
            operating_hours: 0.0,
            input_voltage: 120.0,
            input_current: 0.0,
            output_voltage: 12.0,
            output_current: 0.0,
            pfc_status: true,
            load_percentage: 0.0,
            remaining_life_hours: 100000.0,
        }
    }
}

/// Power supply configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupplyConfig {
    /// Enable power factor correction
    pub enable_pfc: bool,
    /// Power saving mode enabled
    pub power_saving_enabled: bool,
    /// Over-temperature protection threshold (Celsius)
    pub over_temp_threshold: f64,
    /// Over-current protection threshold (amperes)
    pub over_current_threshold: f64,
    /// Under-voltage protection threshold (volts)
    pub under_voltage_threshold: f64,
    /// Over-voltage protection threshold (volts)
    pub over_voltage_threshold: f64,
    /// Fan control mode
    pub fan_control_mode: FanControlMode,
    /// Monitoring interval (seconds)
    pub monitoring_interval: Duration,
    /// Alarm thresholds
    pub alarm_thresholds: AlarmThresholds,
}

impl Default for PowerSupplyConfig {
    fn default() -> Self {
        Self {
            enable_pfc: true,
            power_saving_enabled: false,
            over_temp_threshold: 70.0,
            over_current_threshold: 100.0,
            under_voltage_threshold: 10.8,
            over_voltage_threshold: 13.2,
            fan_control_mode: FanControlMode::Automatic,
            monitoring_interval: Duration::from_secs(10),
            alarm_thresholds: AlarmThresholds::default(),
        }
    }
}

/// Fan control modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FanControlMode {
    /// Automatic temperature-based control
    Automatic,
    /// Manual fixed speed
    Manual(f64), // RPM
    /// Eco mode (minimum speed)
    Eco,
    /// Performance mode (maximum cooling)
    Performance,
    /// Silent mode (minimum noise)
    Silent,
}

/// Alarm thresholds for power supplies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmThresholds {
    /// Temperature warning threshold (Celsius)
    pub temp_warning: f64,
    /// Temperature critical threshold (Celsius)
    pub temp_critical: f64,
    /// Load warning threshold (percentage)
    pub load_warning: f64,
    /// Load critical threshold (percentage)
    pub load_critical: f64,
    /// Efficiency warning threshold (percentage)
    pub efficiency_warning: f64,
    /// Fan failure detection
    pub fan_failure_detection: bool,
}

impl Default for AlarmThresholds {
    fn default() -> Self {
        Self {
            temp_warning: 60.0,
            temp_critical: 75.0,
            load_warning: 80.0,
            load_critical: 95.0,
            efficiency_warning: 85.0,
            fan_failure_detection: true,
        }
    }
}

/// Power supply history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupplyHistory {
    /// Historical metrics samples
    pub metrics_history: Vec<HistoricalMetrics>,
    /// Fault history
    pub fault_history: Vec<FaultRecord>,
    /// Maintenance history
    pub maintenance_history: Vec<MaintenanceRecord>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}

impl Default for PowerSupplyHistory {
    fn default() -> Self {
        Self {
            metrics_history: Vec::new(),
            fault_history: Vec::new(),
            maintenance_history: Vec::new(),
            performance_trends: PerformanceTrends::default(),
        }
    }
}

/// Historical metrics sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMetrics {
    /// Timestamp
    pub timestamp: Instant,
    /// Metrics snapshot
    pub metrics: PowerSupplyMetrics,
}

/// Fault record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultRecord {
    /// Fault timestamp
    pub timestamp: Instant,
    /// Fault type
    pub fault_type: FaultType,
    /// Fault description
    pub description: String,
    /// Fault severity
    pub severity: FaultSeverity,
    /// Resolution status
    pub resolved: bool,
    /// Resolution timestamp
    pub resolution_timestamp: Option<Instant>,
}

/// Fault types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultType {
    OverTemperature,
    OverCurrent,
    OverVoltage,
    UnderVoltage,
    FanFailure,
    EfficiencyDegradation,
    CommunicationFailure,
    PowerFactorIssue,
    Other(String),
}

/// Fault severities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Maintenance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceRecord {
    /// Maintenance timestamp
    pub timestamp: Instant,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Description
    pub description: String,
    /// Technician
    pub technician: String,
    /// Duration
    pub duration: Duration,
}

/// Maintenance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
    Preventive,
    Corrective,
    Predictive,
    Emergency,
    Upgrade,
    Cleaning,
    Calibration,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Efficiency trend (slope per hour)
    pub efficiency_trend: f64,
    /// Temperature trend (slope per hour)
    pub temperature_trend: f64,
    /// Load trend (slope per hour)
    pub load_trend: f64,
    /// Failure prediction score (0.0 to 1.0)
    pub failure_prediction_score: f64,
    /// Predicted remaining useful life (hours)
    pub predicted_rul_hours: f64,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            efficiency_trend: 0.0,
            temperature_trend: 0.0,
            load_trend: 0.0,
            failure_prediction_score: 0.0,
            predicted_rul_hours: 100000.0,
        }
    }
}

/// Power supply manager
#[derive(Debug, Clone)]
pub struct PowerSupplyManager {
    /// Managed power supplies
    pub supplies: HashMap<String, PowerSupply>,
    /// Manager configuration
    pub config: PowerSupplyManagerConfig,
    /// Load balancing strategy
    pub load_balancer: LoadBalancingStrategy,
    /// Redundancy manager
    pub redundancy_manager: RedundancyManager,
}

impl Default for PowerSupplyManager {
    fn default() -> Self {
        Self {
            supplies: HashMap::new(),
            config: PowerSupplyManagerConfig::default(),
            load_balancer: LoadBalancingStrategy::default(),
            redundancy_manager: RedundancyManager::default(),
        }
    }
}

/// Power supply manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSupplyManagerConfig {
    /// Automatic load balancing
    pub auto_load_balancing: bool,
    /// Redundancy enabled
    pub redundancy_enabled: bool,
    /// Health monitoring interval
    pub health_check_interval: Duration,
    /// Fault detection sensitivity
    pub fault_detection_sensitivity: f64,
    /// Performance optimization enabled
    pub performance_optimization: bool,
}

impl Default for PowerSupplyManagerConfig {
    fn default() -> Self {
        Self {
            auto_load_balancing: true,
            redundancy_enabled: true,
            health_check_interval: Duration::from_secs(30),
            fault_detection_sensitivity: 0.8,
            performance_optimization: true,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Equal distribution
    EqualDistribution,
    /// Efficiency-based
    EfficiencyBased,
    /// Temperature-based
    TemperatureBased,
    /// Lifetime-based
    LifetimeBased,
    /// Custom algorithm
    Custom(String),
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        LoadBalancingStrategy::EfficiencyBased
    }
}

/// Redundancy manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyManager {
    /// Redundancy configuration
    pub config: RedundancyConfig,
    /// Active supplies
    pub active_supplies: Vec<String>,
    /// Standby supplies
    pub standby_supplies: Vec<String>,
    /// Failover strategy
    pub failover_strategy: FailoverStrategy,
}

impl Default for RedundancyManager {
    fn default() -> Self {
        Self {
            config: RedundancyConfig::default(),
            active_supplies: Vec::new(),
            standby_supplies: Vec::new(),
            failover_strategy: FailoverStrategy::default(),
        }
    }
}

/// Redundancy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyConfig {
    /// Redundancy level
    pub level: RedundancyLevel,
    /// Automatic failover
    pub auto_failover: bool,
    /// Failover timeout
    pub failover_timeout: Duration,
    /// Health check frequency
    pub health_check_frequency: Duration,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            level: RedundancyLevel::NPlusOne,
            auto_failover: true,
            failover_timeout: Duration::from_secs(5),
            health_check_frequency: Duration::from_secs(10),
        }
    }
}

/// Failover strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverStrategy {
    /// Immediate failover
    Immediate,
    /// Graceful failover
    Graceful,
    /// Load-aware failover
    LoadAware,
    /// Priority-based failover
    PriorityBased,
}

impl Default for FailoverStrategy {
    fn default() -> Self {
        FailoverStrategy::Graceful
    }
}
