// Power Budget Management
//
// This module handles power budgeting, allocation, optimization,
// and power planning for TPU pod coordination systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Power budget allocation and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBudget {
    /// Budget configuration
    pub config: PowerBudgetConfig,
    /// Device allocations
    pub device_allocations: HashMap<String, DevicePowerAllocation>,
    /// Budget constraints
    pub constraints: BudgetConstraints,
    /// Budget optimizer
    pub optimizer: BudgetOptimizer,
    /// Budget monitor
    pub monitor: BudgetMonitor,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

impl Default for PowerBudget {
    fn default() -> Self {
        Self {
            config: PowerBudgetConfig::default(),
            device_allocations: HashMap::new(),
            constraints: BudgetConstraints::default(),
            optimizer: BudgetOptimizer::default(),
            monitor: BudgetMonitor::default(),
            allocation_strategy: AllocationStrategy::default(),
        }
    }
}

/// Power budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBudgetConfig {
    /// Total available power (watts)
    pub total_power_available: f64,
    /// Safety margin (percentage)
    pub safety_margin: f64,
    /// Emergency reserve (percentage)
    pub emergency_reserve: f64,
    /// Budget update interval
    pub update_interval: Duration,
    /// Dynamic allocation enabled
    pub dynamic_allocation: bool,
    /// Power capping enabled
    pub power_capping: bool,
    /// Budget enforcement mode
    pub enforcement_mode: EnforcementMode,
    /// Oversubscription allowed
    pub oversubscription_allowed: bool,
    /// Oversubscription ratio
    pub oversubscription_ratio: f64,
}

impl Default for PowerBudgetConfig {
    fn default() -> Self {
        Self {
            total_power_available: 10000.0, // 10kW
            safety_margin: 10.0,
            emergency_reserve: 5.0,
            update_interval: Duration::from_secs(60),
            dynamic_allocation: true,
            power_capping: true,
            enforcement_mode: EnforcementMode::Soft,
            oversubscription_allowed: true,
            oversubscription_ratio: 1.2,
        }
    }
}

/// Budget enforcement modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementMode {
    /// Soft enforcement (warnings only)
    Soft,
    /// Hard enforcement (power limiting)
    Hard,
    /// Adaptive enforcement
    Adaptive,
    /// Emergency only
    EmergencyOnly,
}

/// Device power allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePowerAllocation {
    /// Device identifier
    pub device_id: String,
    /// Allocated power (watts)
    pub allocated_power: f64,
    /// Minimum required power (watts)
    pub minimum_power: f64,
    /// Maximum allowed power (watts)
    pub maximum_power: f64,
    /// Current consumption (watts)
    pub current_consumption: f64,
    /// Priority level
    pub priority: AllocationPriority,
    /// Power profile
    pub power_profile: PowerProfile,
    /// Allocation status
    pub status: AllocationStatus,
    /// Power constraints
    pub constraints: DevicePowerConstraints,
    /// Performance impact
    pub performance_impact: PerformanceImpact,
}

impl Default for DevicePowerAllocation {
    fn default() -> Self {
        Self {
            device_id: String::new(),
            allocated_power: 0.0,
            minimum_power: 0.0,
            maximum_power: 0.0,
            current_consumption: 0.0,
            priority: AllocationPriority::Normal,
            power_profile: PowerProfile::default(),
            status: AllocationStatus::Active,
            constraints: DevicePowerConstraints::default(),
            performance_impact: PerformanceImpact::default(),
        }
    }
}

/// Allocation priorities
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum AllocationPriority {
    /// Critical system components
    Critical,
    /// High priority workloads
    High,
    /// Normal priority workloads
    Normal,
    /// Low priority workloads
    Low,
    /// Background tasks
    Background,
    /// Best effort
    BestEffort,
}

impl Default for AllocationPriority {
    fn default() -> Self {
        AllocationPriority::Normal
    }
}

/// Power profile for devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerProfile {
    /// Profile type
    pub profile_type: ProfileType,
    /// Base power consumption (watts)
    pub base_power: f64,
    /// Dynamic power range (watts)
    pub dynamic_power_range: (f64, f64),
    /// Power states
    pub power_states: Vec<PowerState>,
    /// Transition characteristics
    pub transition_characteristics: TransitionCharacteristics,
    /// Load characteristics
    pub load_characteristics: LoadCharacteristics,
}

impl Default for PowerProfile {
    fn default() -> Self {
        Self {
            profile_type: ProfileType::Constant,
            base_power: 0.0,
            dynamic_power_range: (0.0, 0.0),
            power_states: Vec::new(),
            transition_characteristics: TransitionCharacteristics::default(),
            load_characteristics: LoadCharacteristics::default(),
        }
    }
}

/// Profile types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileType {
    /// Constant power consumption
    Constant,
    /// Variable power consumption
    Variable,
    /// Stepped power levels
    Stepped,
    /// Burst workload pattern
    Burst,
    /// Periodic workload pattern
    Periodic,
    /// Custom profile
    Custom(String),
}

/// Power state definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerState {
    /// State name
    pub name: String,
    /// Power consumption (watts)
    pub power_consumption: f64,
    /// Performance level (0.0 to 1.0)
    pub performance_level: f64,
    /// Transition time to this state
    pub transition_time: Duration,
    /// Minimum time in state
    pub minimum_duration: Duration,
    /// State description
    pub description: String,
}

/// Transition characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionCharacteristics {
    /// Power-up time
    pub power_up_time: Duration,
    /// Power-down time
    pub power_down_time: Duration,
    /// Inrush current (amperes)
    pub inrush_current: f64,
    /// Inrush duration
    pub inrush_duration: Duration,
    /// Graceful shutdown supported
    pub graceful_shutdown: bool,
    /// Fast shutdown time
    pub fast_shutdown_time: Duration,
}

impl Default for TransitionCharacteristics {
    fn default() -> Self {
        Self {
            power_up_time: Duration::from_secs(30),
            power_down_time: Duration::from_secs(10),
            inrush_current: 0.0,
            inrush_duration: Duration::from_millis(100),
            graceful_shutdown: true,
            fast_shutdown_time: Duration::from_secs(1),
        }
    }
}

/// Load characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCharacteristics {
    /// Load pattern type
    pub pattern_type: LoadPatternType,
    /// Utilization factor
    pub utilization_factor: f64,
    /// Peak-to-average ratio
    pub peak_to_average_ratio: f64,
    /// Load predictability
    pub predictability: f64,
    /// Seasonal variations
    pub seasonal_variations: bool,
    /// Daily patterns
    pub daily_patterns: Vec<DailyPattern>,
}

impl Default for LoadCharacteristics {
    fn default() -> Self {
        Self {
            pattern_type: LoadPatternType::Steady,
            utilization_factor: 0.7,
            peak_to_average_ratio: 1.2,
            predictability: 0.8,
            seasonal_variations: false,
            daily_patterns: Vec::new(),
        }
    }
}

/// Load pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadPatternType {
    Steady,
    Variable,
    Periodic,
    Random,
    Spiky,
    Batched,
}

/// Daily pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPattern {
    /// Time of day (hour)
    pub hour: u8,
    /// Load factor (0.0 to 1.0)
    pub load_factor: f64,
    /// Duration (hours)
    pub duration: f64,
}

/// Allocation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Allocation is active
    Active,
    /// Allocation is suspended
    Suspended,
    /// Allocation is pending
    Pending,
    /// Allocation has been revoked
    Revoked,
    /// Allocation is in violation
    Violation,
    /// Allocation is capped
    Capped,
}

/// Device power constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePowerConstraints {
    /// Thermal constraints
    pub thermal_constraints: ThermalConstraints,
    /// Electrical constraints
    pub electrical_constraints: ElectricalConstraints,
    /// Physical constraints
    pub physical_constraints: PhysicalConstraints,
    /// Time-based constraints
    pub time_constraints: TimeConstraints,
    /// Dependency constraints
    pub dependency_constraints: Vec<DependencyConstraint>,
}

impl Default for DevicePowerConstraints {
    fn default() -> Self {
        Self {
            thermal_constraints: ThermalConstraints::default(),
            electrical_constraints: ElectricalConstraints::default(),
            physical_constraints: PhysicalConstraints::default(),
            time_constraints: TimeConstraints::default(),
            dependency_constraints: Vec::new(),
        }
    }
}

/// Thermal constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConstraints {
    /// Maximum operating temperature (Celsius)
    pub max_temperature: f64,
    /// Thermal design power (watts)
    pub thermal_design_power: f64,
    /// Cooling requirements
    pub cooling_requirements: CoolingRequirements,
    /// Temperature derating curve
    pub derating_curve: Vec<(f64, f64)>, // (temperature, power_factor)
}

impl Default for ThermalConstraints {
    fn default() -> Self {
        Self {
            max_temperature: 85.0,
            thermal_design_power: 100.0,
            cooling_requirements: CoolingRequirements::default(),
            derating_curve: vec![(25.0, 1.0), (50.0, 0.9), (75.0, 0.7), (85.0, 0.5)],
        }
    }
}

/// Cooling requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingRequirements {
    /// Required airflow (CFM)
    pub airflow_cfm: f64,
    /// Cooling method
    pub cooling_method: CoolingMethod,
    /// Heat dissipation (watts)
    pub heat_dissipation: f64,
    /// Ambient temperature limit (Celsius)
    pub ambient_temp_limit: f64,
}

impl Default for CoolingRequirements {
    fn default() -> Self {
        Self {
            airflow_cfm: 50.0,
            cooling_method: CoolingMethod::AirCooling,
            heat_dissipation: 100.0,
            ambient_temp_limit: 35.0,
        }
    }
}

/// Cooling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingMethod {
    AirCooling,
    LiquidCooling,
    PassiveCooling,
    ImmersionCooling,
    ThermoelectricCooling,
    Hybrid,
}

/// Electrical constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricalConstraints {
    /// Maximum current (amperes)
    pub max_current: f64,
    /// Voltage requirements
    pub voltage_requirements: VoltageRequirements,
    /// Power factor requirements
    pub power_factor_requirements: PowerFactorRequirements,
    /// Harmonics limitations
    pub harmonics_limitations: HarmonicsLimitations,
}

impl Default for ElectricalConstraints {
    fn default() -> Self {
        Self {
            max_current: 100.0,
            voltage_requirements: VoltageRequirements::default(),
            power_factor_requirements: PowerFactorRequirements::default(),
            harmonics_limitations: HarmonicsLimitations::default(),
        }
    }
}

/// Voltage requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageRequirements {
    /// Nominal voltage (volts)
    pub nominal_voltage: f64,
    /// Voltage tolerance (percentage)
    pub voltage_tolerance: f64,
    /// Voltage regulation
    pub voltage_regulation: f64,
    /// Multiple voltage rails
    pub voltage_rails: Vec<VoltageRail>,
}

impl Default for VoltageRequirements {
    fn default() -> Self {
        Self {
            nominal_voltage: 12.0,
            voltage_tolerance: 5.0,
            voltage_regulation: 1.0,
            voltage_rails: Vec::new(),
        }
    }
}

/// Voltage rail specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageRail {
    /// Rail name
    pub name: String,
    /// Voltage level (volts)
    pub voltage: f64,
    /// Current requirement (amperes)
    pub current: f64,
    /// Power requirement (watts)
    pub power: f64,
    /// Regulation requirement (percentage)
    pub regulation: f64,
}

/// Power factor requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerFactorRequirements {
    /// Minimum power factor
    pub minimum_power_factor: f64,
    /// Target power factor
    pub target_power_factor: f64,
    /// Power factor correction required
    pub correction_required: bool,
    /// Correction method
    pub correction_method: PowerFactorCorrectionMethod,
}

impl Default for PowerFactorRequirements {
    fn default() -> Self {
        Self {
            minimum_power_factor: 0.8,
            target_power_factor: 0.95,
            correction_required: false,
            correction_method: PowerFactorCorrectionMethod::Active,
        }
    }
}

/// Power factor correction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerFactorCorrectionMethod {
    Active,
    Passive,
    Hybrid,
    None,
}

/// Harmonics limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicsLimitations {
    /// Total harmonic distortion limit (percentage)
    pub thd_limit: f64,
    /// Individual harmonic limits
    pub individual_limits: HashMap<u32, f64>,
    /// Compliance standard
    pub compliance_standard: String,
}

impl Default for HarmonicsLimitations {
    fn default() -> Self {
        let mut individual_limits = HashMap::new();
        individual_limits.insert(3, 5.0);
        individual_limits.insert(5, 4.0);
        individual_limits.insert(7, 3.0);

        Self {
            thd_limit: 8.0,
            individual_limits,
            compliance_standard: "IEEE 519".to_string(),
        }
    }
}

/// Physical constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalConstraints {
    /// Space requirements (cubic meters)
    pub space_requirements: f64,
    /// Weight limit (kg)
    pub weight_limit: f64,
    /// Mounting requirements
    pub mounting_requirements: MountingRequirements,
    /// Environmental requirements
    pub environmental_requirements: EnvironmentalRequirements,
}

impl Default for PhysicalConstraints {
    fn default() -> Self {
        Self {
            space_requirements: 1.0,
            weight_limit: 50.0,
            mounting_requirements: MountingRequirements::default(),
            environmental_requirements: EnvironmentalRequirements::default(),
        }
    }
}

/// Mounting requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountingRequirements {
    /// Mounting type
    pub mounting_type: MountingType,
    /// Orientation requirements
    pub orientation: OrientationRequirement,
    /// Vibration tolerance
    pub vibration_tolerance: f64,
    /// Shock tolerance
    pub shock_tolerance: f64,
}

impl Default for MountingRequirements {
    fn default() -> Self {
        Self {
            mounting_type: MountingType::Rack,
            orientation: OrientationRequirement::Any,
            vibration_tolerance: 2.0,
            shock_tolerance: 10.0,
        }
    }
}

/// Mounting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MountingType {
    Rack,
    Floor,
    Wall,
    Ceiling,
    Pole,
    Custom(String),
}

/// Orientation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrientationRequirement {
    Any,
    Horizontal,
    Vertical,
    Specific(f64), // Angle in degrees
}

/// Environmental requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalRequirements {
    /// Operating temperature range (Celsius)
    pub temperature_range: (f64, f64),
    /// Operating humidity range (percentage)
    pub humidity_range: (f64, f64),
    /// Altitude limit (meters)
    pub altitude_limit: f64,
    /// Dust resistance required
    pub dust_resistance: bool,
    /// Water resistance required
    pub water_resistance: bool,
    /// EMI shielding required
    pub emi_shielding: bool,
}

impl Default for EnvironmentalRequirements {
    fn default() -> Self {
        Self {
            temperature_range: (0.0, 50.0),
            humidity_range: (10.0, 90.0),
            altitude_limit: 2000.0,
            dust_resistance: false,
            water_resistance: false,
            emi_shielding: false,
        }
    }
}

/// Time-based constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Operating schedule
    pub operating_schedule: Vec<TimeWindow>,
    /// Maintenance windows
    pub maintenance_windows: Vec<MaintenanceWindow>,
    /// Peak hours restrictions
    pub peak_hours_restrictions: PeakHoursRestrictions,
    /// Startup/shutdown scheduling
    pub startup_shutdown_scheduling: StartupShutdownScheduling,
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self {
            operating_schedule: Vec::new(),
            maintenance_windows: Vec::new(),
            peak_hours_restrictions: PeakHoursRestrictions::default(),
            startup_shutdown_scheduling: StartupShutdownScheduling::default(),
        }
    }
}

/// Time window specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Window name
    pub name: String,
    /// Start time (hour:minute)
    pub start_time: (u8, u8),
    /// End time (hour:minute)
    pub end_time: (u8, u8),
    /// Days of week
    pub days_of_week: Vec<u8>,
    /// Power budget for this window
    pub power_budget: f64,
}

/// Maintenance window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    /// Window identifier
    pub window_id: String,
    /// Scheduled start time
    pub start_time: SystemTime,
    /// Duration
    pub duration: Duration,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Power requirements during maintenance
    pub power_requirements: f64,
    /// Critical systems exempt
    pub critical_exempt: bool,
}

/// Maintenance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaintenanceType {
    Preventive,
    Corrective,
    Predictive,
    Emergency,
    Upgrade,
    Inspection,
}

/// Peak hours restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakHoursRestrictions {
    /// Peak hours definition
    pub peak_hours: Vec<TimeWindow>,
    /// Power reduction during peak hours (percentage)
    pub power_reduction: f64,
    /// Critical systems exempt
    pub critical_exempt: bool,
    /// Demand response participation
    pub demand_response: bool,
}

impl Default for PeakHoursRestrictions {
    fn default() -> Self {
        Self {
            peak_hours: Vec::new(),
            power_reduction: 0.0,
            critical_exempt: true,
            demand_response: false,
        }
    }
}

/// Startup/shutdown scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupShutdownScheduling {
    /// Staggered startup enabled
    pub staggered_startup: bool,
    /// Startup delay between devices
    pub startup_delay: Duration,
    /// Graceful shutdown sequence
    pub graceful_shutdown: bool,
    /// Shutdown priority order
    pub shutdown_priority: Vec<String>,
}

impl Default for StartupShutdownScheduling {
    fn default() -> Self {
        Self {
            staggered_startup: true,
            startup_delay: Duration::from_secs(10),
            graceful_shutdown: true,
            shutdown_priority: Vec::new(),
        }
    }
}

/// Dependency constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConstraint {
    /// Constraint type
    pub constraint_type: DependencyType,
    /// Dependent device
    pub dependent_device: String,
    /// Dependency relationship
    pub relationship: DependencyRelationship,
    /// Power impact
    pub power_impact: f64,
}

/// Dependency types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Power prerequisite
    PowerPrerequisite,
    /// Thermal dependency
    ThermalDependency,
    /// Operational dependency
    OperationalDependency,
    /// Resource sharing
    ResourceSharing,
    /// Redundancy group
    RedundancyGroup,
}

/// Dependency relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyRelationship {
    /// Must start before
    MustStartBefore,
    /// Must start after
    MustStartAfter,
    /// Must run together
    MustRunTogether,
    /// Cannot run together
    CannotRunTogether,
    /// Shares power budget
    SharesPowerBudget,
}

/// Performance impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Performance degradation curve
    pub degradation_curve: Vec<(f64, f64)>, // (power_reduction_%, performance_impact_%)
    /// Critical performance threshold
    pub critical_threshold: f64,
    /// Acceptable performance range
    pub acceptable_range: (f64, f64),
    /// Performance recovery time
    pub recovery_time: Duration,
    /// Quality of service impact
    pub qos_impact: QualityOfServiceImpact,
}

impl Default for PerformanceImpact {
    fn default() -> Self {
        Self {
            degradation_curve: vec![(0.0, 0.0), (10.0, 5.0), (20.0, 15.0), (50.0, 60.0)],
            critical_threshold: 0.5,
            acceptable_range: (0.8, 1.0),
            recovery_time: Duration::from_secs(60),
            qos_impact: QualityOfServiceImpact::default(),
        }
    }
}

/// Quality of service impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityOfServiceImpact {
    /// Latency impact
    pub latency_impact: f64,
    /// Throughput impact
    pub throughput_impact: f64,
    /// Reliability impact
    pub reliability_impact: f64,
    /// Availability impact
    pub availability_impact: f64,
}

impl Default for QualityOfServiceImpact {
    fn default() -> Self {
        Self {
            latency_impact: 0.0,
            throughput_impact: 0.0,
            reliability_impact: 0.0,
            availability_impact: 0.0,
        }
    }
}

/// Budget constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConstraints {
    /// Hard power limit (watts)
    pub hard_power_limit: f64,
    /// Soft power limit (watts)
    pub soft_power_limit: f64,
    /// Peak power limit (watts)
    pub peak_power_limit: f64,
    /// Average power limit (watts)
    pub average_power_limit: f64,
    /// Time-based limits
    pub time_based_limits: Vec<TimePowerLimit>,
    /// Seasonal adjustments
    pub seasonal_adjustments: SeasonalAdjustments,
    /// Emergency power levels
    pub emergency_levels: EmergencyPowerLevels,
}

impl Default for BudgetConstraints {
    fn default() -> Self {
        Self {
            hard_power_limit: 10000.0,
            soft_power_limit: 9000.0,
            peak_power_limit: 12000.0,
            average_power_limit: 8000.0,
            time_based_limits: Vec::new(),
            seasonal_adjustments: SeasonalAdjustments::default(),
            emergency_levels: EmergencyPowerLevels::default(),
        }
    }
}

/// Time-based power limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePowerLimit {
    /// Time window
    pub time_window: TimeWindow,
    /// Power limit for this window
    pub power_limit: f64,
    /// Enforcement priority
    pub enforcement_priority: EnforcementPriority,
}

/// Enforcement priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementPriority {
    Mandatory,
    High,
    Medium,
    Low,
    Advisory,
}

/// Seasonal adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalAdjustments {
    /// Summer adjustment (percentage)
    pub summer_adjustment: f64,
    /// Winter adjustment (percentage)
    pub winter_adjustment: f64,
    /// Spring/Fall adjustment (percentage)
    pub spring_fall_adjustment: f64,
    /// Temperature-based adjustments
    pub temperature_based: bool,
    /// Humidity-based adjustments
    pub humidity_based: bool,
}

impl Default for SeasonalAdjustments {
    fn default() -> Self {
        Self {
            summer_adjustment: -10.0, // Reduce power in summer for cooling
            winter_adjustment: 5.0,   // Slight increase in winter
            spring_fall_adjustment: 0.0,
            temperature_based: true,
            humidity_based: false,
        }
    }
}

/// Emergency power levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyPowerLevels {
    /// Emergency level definitions
    pub levels: Vec<EmergencyLevel>,
    /// Automatic level transition
    pub auto_transition: bool,
    /// Recovery procedures
    pub recovery_procedures: Vec<RecoveryProcedure>,
}

impl Default for EmergencyPowerLevels {
    fn default() -> Self {
        Self {
            levels: vec![
                EmergencyLevel {
                    name: "Normal".to_string(),
                    power_percentage: 100.0,
                    duration_limit: None,
                    description: "Normal operation".to_string(),
                },
                EmergencyLevel {
                    name: "Conservative".to_string(),
                    power_percentage: 80.0,
                    duration_limit: None,
                    description: "Conservative power usage".to_string(),
                },
                EmergencyLevel {
                    name: "Emergency".to_string(),
                    power_percentage: 50.0,
                    duration_limit: Some(Duration::from_secs(3600)),
                    description: "Emergency power reduction".to_string(),
                },
                EmergencyLevel {
                    name: "Critical".to_string(),
                    power_percentage: 25.0,
                    duration_limit: Some(Duration::from_secs(900)),
                    description: "Critical systems only".to_string(),
                },
            ],
            auto_transition: true,
            recovery_procedures: Vec::new(),
        }
    }
}

/// Emergency level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyLevel {
    /// Level name
    pub name: String,
    /// Power percentage of normal operation
    pub power_percentage: f64,
    /// Maximum duration at this level
    pub duration_limit: Option<Duration>,
    /// Level description
    pub description: String,
}

/// Recovery procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryProcedure {
    /// Procedure name
    pub name: String,
    /// Trigger conditions
    pub trigger_conditions: Vec<TriggerCondition>,
    /// Recovery steps
    pub steps: Vec<RecoveryStep>,
    /// Estimated recovery time
    pub estimated_time: Duration,
}

/// Trigger condition for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Threshold value
    pub threshold: f64,
    /// Duration requirement
    pub duration: Duration,
}

/// Condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    PowerAvailable,
    TemperatureNormal,
    SystemStable,
    LoadReduced,
    TimeElapsed,
    ManualTrigger,
}

/// Recovery step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    /// Step description
    pub description: String,
    /// Step action
    pub action: RecoveryAction,
    /// Expected duration
    pub duration: Duration,
    /// Prerequisites
    pub prerequisites: Vec<String>,
}

/// Recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    RestorePower(String),    // Device ID
    IncreasePowerLimit(f64), // Power increase in watts
    EnableDevice(String),    // Device ID
    VerifyStability,
    NotifyOperator(String), // Message
    WaitForConfirmation,
}

/// Budget optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetOptimizer {
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Optimization constraints
    pub constraints: OptimizationConstraints,
    /// Optimization history
    pub history: OptimizationHistory,
    /// Performance metrics
    pub metrics: OptimizationMetrics,
}

impl Default for BudgetOptimizer {
    fn default() -> Self {
        Self {
            objectives: vec![
                OptimizationObjective::MaximizePerformance,
                OptimizationObjective::MinimizePowerConsumption,
            ],
            algorithm: OptimizationAlgorithm::GeneticAlgorithm,
            constraints: OptimizationConstraints::default(),
            history: OptimizationHistory::default(),
            metrics: OptimizationMetrics::default(),
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MaximizePerformance,
    MinimizePowerConsumption,
    MaximizeEfficiency,
    MinimizeCost,
    MaximizeReliability,
    MinimizeTemperature,
    BalanceWorkload,
    Custom(String),
}

/// Optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    GeneticAlgorithm,
    SimulatedAnnealing,
    ParticleSwarmOptimization,
    LinearProgramming,
    DynamicProgramming,
    HeuristicSearch,
    MachineLearning,
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    /// Maximum optimization time
    pub max_optimization_time: Duration,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
    /// Solution quality requirements
    pub quality_requirements: QualityRequirements,
    /// Stability requirements
    pub stability_requirements: StabilityRequirements,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            max_optimization_time: Duration::from_secs(60),
            convergence_criteria: ConvergenceCriteria::default(),
            quality_requirements: QualityRequirements::default(),
            stability_requirements: StabilityRequirements::default(),
        }
    }
}

/// Convergence criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: u32,
    /// Improvement threshold
    pub improvement_threshold: f64,
    /// Stability window
    pub stability_window: u32,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            improvement_threshold: 0.001,
            stability_window: 10,
        }
    }
}

/// Quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum solution quality
    pub min_solution_quality: f64,
    /// Solution diversity required
    pub diversity_required: bool,
    /// Robustness testing
    pub robustness_testing: bool,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_solution_quality: 0.8,
            diversity_required: true,
            robustness_testing: true,
        }
    }
}

/// Stability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRequirements {
    /// Solution stability period
    pub stability_period: Duration,
    /// Maximum change rate
    pub max_change_rate: f64,
    /// Oscillation detection
    pub oscillation_detection: bool,
}

impl Default for StabilityRequirements {
    fn default() -> Self {
        Self {
            stability_period: Duration::from_secs(300),
            max_change_rate: 0.1,
            oscillation_detection: true,
        }
    }
}

/// Optimization history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHistory {
    /// Historical optimization runs
    pub runs: Vec<OptimizationRun>,
    /// Performance trends
    pub trends: Vec<PerformanceTrend>,
    /// Best solutions
    pub best_solutions: Vec<OptimizationSolution>,
}

impl Default for OptimizationHistory {
    fn default() -> Self {
        Self {
            runs: Vec::new(),
            trends: Vec::new(),
            best_solutions: Vec::new(),
        }
    }
}

/// Optimization run record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRun {
    /// Run timestamp
    pub timestamp: SystemTime,
    /// Run duration
    pub duration: Duration,
    /// Objective function value
    pub objective_value: f64,
    /// Number of iterations
    pub iterations: u32,
    /// Convergence achieved
    pub converged: bool,
    /// Solution quality
    pub solution_quality: f64,
}

/// Performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric: String,
    /// Trend data points
    pub data_points: Vec<(SystemTime, f64)>,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Oscillating,
}

/// Optimization solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSolution {
    /// Solution identifier
    pub solution_id: String,
    /// Device allocations
    pub allocations: HashMap<String, f64>,
    /// Objective function value
    pub objective_value: f64,
    /// Constraint violations
    pub constraint_violations: Vec<ConstraintViolation>,
    /// Solution timestamp
    pub timestamp: SystemTime,
}

/// Constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Constraint type
    pub constraint_type: String,
    /// Violation magnitude
    pub magnitude: f64,
    /// Affected devices
    pub affected_devices: Vec<String>,
}

/// Optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    /// Average optimization time
    pub avg_optimization_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Solution quality average
    pub avg_solution_quality: f64,
    /// Improvement achieved
    pub improvement_achieved: f64,
    /// Stability score
    pub stability_score: f64,
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            avg_optimization_time: Duration::from_secs(30),
            success_rate: 0.95,
            avg_solution_quality: 0.85,
            improvement_achieved: 0.15,
            stability_score: 0.9,
        }
    }
}

/// Budget monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetMonitor {
    /// Monitoring configuration
    pub config: MonitoringConfig,
    /// Real-time metrics
    pub real_time_metrics: RealTimeMetrics,
    /// Violation tracking
    pub violation_tracker: ViolationTracker,
    /// Alert system
    pub alert_system: BudgetAlertSystem,
    /// Reporting system
    pub reporting: BudgetReporting,
}

impl Default for BudgetMonitor {
    fn default() -> Self {
        Self {
            config: MonitoringConfig::default(),
            real_time_metrics: RealTimeMetrics::default(),
            violation_tracker: ViolationTracker::default(),
            alert_system: BudgetAlertSystem::default(),
            reporting: BudgetReporting::default(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Real-time monitoring enabled
    pub real_time_enabled: bool,
    /// Historical data retention
    pub data_retention: Duration,
    /// Alert thresholds
    pub alert_thresholds: BudgetAlertThresholds,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(10),
            real_time_enabled: true,
            data_retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            alert_thresholds: BudgetAlertThresholds::default(),
        }
    }
}

/// Real-time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    /// Current total power consumption
    pub total_power_consumption: f64,
    /// Power budget utilization (percentage)
    pub budget_utilization: f64,
    /// Available power headroom
    pub available_headroom: f64,
    /// Power efficiency
    pub power_efficiency: f64,
    /// Active device count
    pub active_device_count: usize,
    /// Critical violations count
    pub critical_violations: u32,
    /// Performance impact score
    pub performance_impact_score: f64,
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self {
            total_power_consumption: 0.0,
            budget_utilization: 0.0,
            available_headroom: 0.0,
            power_efficiency: 0.0,
            active_device_count: 0,
            critical_violations: 0,
            performance_impact_score: 0.0,
        }
    }
}

/// Violation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationTracker {
    /// Active violations
    pub active_violations: Vec<BudgetViolation>,
    /// Violation history
    pub violation_history: Vec<BudgetViolation>,
    /// Violation statistics
    pub statistics: ViolationStatistics,
}

impl Default for ViolationTracker {
    fn default() -> Self {
        Self {
            active_violations: Vec::new(),
            violation_history: Vec::new(),
            statistics: ViolationStatistics::default(),
        }
    }
}

/// Budget violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetViolation {
    /// Violation ID
    pub violation_id: String,
    /// Violation type
    pub violation_type: ViolationType,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Affected device
    pub affected_device: String,
    /// Violation magnitude
    pub magnitude: f64,
    /// Start timestamp
    pub start_time: SystemTime,
    /// End timestamp (if resolved)
    pub end_time: Option<SystemTime>,
    /// Resolution actions taken
    pub resolution_actions: Vec<String>,
}

/// Violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    PowerExceeded,
    ThermalLimit,
    CurrentOverload,
    VoltageOutOfRange,
    EfficiencyTooLow,
    UnauthorizedUsage,
    ScheduleViolation,
    DependencyViolation,
}

/// Violation severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Violation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationStatistics {
    /// Total violations
    pub total_violations: u64,
    /// Violations by type
    pub violations_by_type: HashMap<String, u64>,
    /// Average resolution time
    pub avg_resolution_time: Duration,
    /// Violation rate
    pub violation_rate: f64,
    /// Most frequent violators
    pub frequent_violators: Vec<String>,
}

impl Default for ViolationStatistics {
    fn default() -> Self {
        Self {
            total_violations: 0,
            violations_by_type: HashMap::new(),
            avg_resolution_time: Duration::from_secs(0),
            violation_rate: 0.0,
            frequent_violators: Vec::new(),
        }
    }
}

/// Budget alert system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlertSystem {
    /// Alert configuration
    pub config: BudgetAlertConfig,
    /// Active alerts
    pub active_alerts: Vec<BudgetAlert>,
    /// Alert escalation rules
    pub escalation_rules: Vec<AlertEscalationRule>,
}

impl Default for BudgetAlertSystem {
    fn default() -> Self {
        Self {
            config: BudgetAlertConfig::default(),
            active_alerts: Vec::new(),
            escalation_rules: Vec::new(),
        }
    }
}

/// Budget alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlertConfig {
    /// Alert thresholds
    pub thresholds: BudgetAlertThresholds,
    /// Notification channels
    pub notification_channels: Vec<String>,
    /// Alert suppression rules
    pub suppression_rules: Vec<AlertSuppressionRule>,
}

impl Default for BudgetAlertConfig {
    fn default() -> Self {
        Self {
            thresholds: BudgetAlertThresholds::default(),
            notification_channels: Vec::new(),
            suppression_rules: Vec::new(),
        }
    }
}

/// Budget alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlertThresholds {
    /// Budget utilization warning (percentage)
    pub utilization_warning: f64,
    /// Budget utilization critical (percentage)
    pub utilization_critical: f64,
    /// Efficiency warning threshold
    pub efficiency_warning: f64,
    /// Violation count threshold
    pub violation_count_threshold: u32,
    /// Performance impact threshold
    pub performance_impact_threshold: f64,
}

impl Default for BudgetAlertThresholds {
    fn default() -> Self {
        Self {
            utilization_warning: 80.0,
            utilization_critical: 95.0,
            efficiency_warning: 0.7,
            violation_count_threshold: 5,
            performance_impact_threshold: 0.2,
        }
    }
}

/// Budget alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: BudgetAlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Acknowledgment status
    pub acknowledged: bool,
    /// Resolution status
    pub resolved: bool,
}

/// Budget alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetAlertType {
    BudgetExceeded,
    EfficiencyLow,
    ViolationCountHigh,
    PerformanceImpact,
    AllocationFailure,
    OptimizationFailure,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert escalation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalationRule {
    /// Rule name
    pub name: String,
    /// Trigger conditions
    pub conditions: Vec<EscalationCondition>,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
    /// Escalation delay
    pub delay: Duration,
}

/// Escalation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCondition {
    /// Alert type
    pub alert_type: BudgetAlertType,
    /// Duration unresolved
    pub duration_unresolved: Duration,
    /// Severity level
    pub severity_level: AlertSeverity,
}

/// Escalation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationAction {
    NotifyManager,
    IncreasePriority,
    TriggerEmergencyProcedure,
    AutoResolveAttempt,
    CallOperator,
}

/// Alert suppression rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSuppressionRule {
    /// Rule name
    pub name: String,
    /// Suppression conditions
    pub conditions: Vec<SuppressionCondition>,
    /// Suppression duration
    pub duration: Duration,
    /// Applies to alert types
    pub alert_types: Vec<BudgetAlertType>,
}

/// Suppression condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionCondition {
    /// Condition type
    pub condition_type: SuppressionConditionType,
    /// Condition value
    pub value: String,
}

/// Suppression condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionConditionType {
    MaintenanceWindow,
    EmergencyMode,
    DeviceShutdown,
    ScheduledActivity,
}

/// Budget reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReporting {
    /// Reporting configuration
    pub config: ReportingConfig,
    /// Scheduled reports
    pub scheduled_reports: Vec<ScheduledReport>,
    /// Report templates
    pub templates: Vec<ReportTemplate>,
}

impl Default for BudgetReporting {
    fn default() -> Self {
        Self {
            config: ReportingConfig::default(),
            scheduled_reports: Vec::new(),
            templates: Vec::new(),
        }
    }
}

/// Report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template type
    pub template_type: ReportTemplateType,
    /// Content sections
    pub sections: Vec<ReportSection>,
    /// Output format
    pub format: ReportFormat,
}

/// Report template types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportTemplateType {
    BudgetUtilization,
    PowerEfficiency,
    ViolationSummary,
    PerformanceImpact,
    CostAnalysis,
    TrendAnalysis,
}

/// Report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section name
    pub name: String,
    /// Section content
    pub content: Vec<ReportContent>,
    /// Visualization type
    pub visualization: VisualizationType,
}

/// Report content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportContent {
    Summary(SummaryMetrics),
    Table(TableData),
    Chart(ChartData),
    Text(String),
}

/// Summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetrics {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit
    pub unit: String,
    /// Trend direction
    pub trend: TrendDirection,
}

/// Table data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    /// Column headers
    pub headers: Vec<String>,
    /// Row data
    pub rows: Vec<Vec<String>>,
}

/// Chart data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    /// Chart type
    pub chart_type: ChartType,
    /// Data series
    pub series: Vec<DataSeries>,
    /// Chart title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
}

/// Chart types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Histogram,
    HeatMap,
}

/// Data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub data: Vec<(f64, f64)>,
    /// Series color
    pub color: String,
}

/// Visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    Table,
    Chart,
    Gauge,
    Text,
    Mixed,
}

/// Report format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    HTML,
    Excel,
    CSV,
    JSON,
}

/// Scheduled report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledReport {
    /// Report name
    pub name: String,
    /// Template to use
    pub template: String,
    /// Schedule specification
    pub schedule: ReportSchedule,
    /// Recipients
    pub recipients: Vec<String>,
    /// Enabled
    pub enabled: bool,
}

/// Report schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Frequency
    pub frequency: ReportFrequency,
    /// Time of day
    pub time_of_day: (u8, u8), // hour, minute
    /// Days of week (for weekly reports)
    pub days_of_week: Vec<u8>,
}

/// Schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Recurring,
    OneTime,
    Triggered,
}

/// Report frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    Custom(Duration),
}

/// Allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStrategy {
    /// Strategy type
    pub strategy_type: AllocationStrategyType,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Fairness policy
    pub fairness_policy: FairnessPolicy,
    /// Priority weighting
    pub priority_weighting: PriorityWeighting,
    /// Dynamic adjustment
    pub dynamic_adjustment: DynamicAdjustment,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self {
            strategy_type: AllocationStrategyType::ProportionalFair,
            parameters: HashMap::new(),
            fairness_policy: FairnessPolicy::default(),
            priority_weighting: PriorityWeighting::default(),
            dynamic_adjustment: DynamicAdjustment::default(),
        }
    }
}

/// Allocation strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategyType {
    FirstComeFirstServed,
    PriorityBased,
    ProportionalFair,
    MaxMin,
    WorstCaseOptimal,
    UtilityMaximizing,
    EnergyEfficient,
    PerformanceOptimal,
}

/// Fairness policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessPolicy {
    /// Fairness metric
    pub fairness_metric: FairnessMetric,
    /// Minimum allocation guarantee
    pub min_allocation_guarantee: f64,
    /// Proportional sharing enabled
    pub proportional_sharing: bool,
    /// Starvation prevention
    pub starvation_prevention: bool,
}

impl Default for FairnessPolicy {
    fn default() -> Self {
        Self {
            fairness_metric: FairnessMetric::ProportionalFairness,
            min_allocation_guarantee: 0.1,
            proportional_sharing: true,
            starvation_prevention: true,
        }
    }
}

/// Fairness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessMetric {
    ProportionalFairness,
    MaxMinFairness,
    AlphaFairness(f64),
    JainFairnessIndex,
    EqualAllocation,
}

/// Priority weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeighting {
    /// Priority weights
    pub weights: HashMap<AllocationPriority, f64>,
    /// Adaptive weighting
    pub adaptive_weighting: bool,
    /// Weight adjustment factors
    pub adjustment_factors: WeightAdjustmentFactors,
}

impl Default for PriorityWeighting {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(AllocationPriority::Critical, 10.0);
        weights.insert(AllocationPriority::High, 5.0);
        weights.insert(AllocationPriority::Normal, 1.0);
        weights.insert(AllocationPriority::Low, 0.5);
        weights.insert(AllocationPriority::Background, 0.1);
        weights.insert(AllocationPriority::BestEffort, 0.05);

        Self {
            weights,
            adaptive_weighting: true,
            adjustment_factors: WeightAdjustmentFactors::default(),
        }
    }
}

/// Weight adjustment factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAdjustmentFactors {
    /// Performance impact factor
    pub performance_impact_factor: f64,
    /// Efficiency factor
    pub efficiency_factor: f64,
    /// Deadline urgency factor
    pub deadline_urgency_factor: f64,
    /// Resource scarcity factor
    pub resource_scarcity_factor: f64,
}

impl Default for WeightAdjustmentFactors {
    fn default() -> Self {
        Self {
            performance_impact_factor: 1.5,
            efficiency_factor: 1.2,
            deadline_urgency_factor: 2.0,
            resource_scarcity_factor: 1.8,
        }
    }
}

/// Dynamic adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAdjustment {
    /// Enable dynamic adjustment
    pub enabled: bool,
    /// Adjustment interval
    pub adjustment_interval: Duration,
    /// Learning algorithm
    pub learning_algorithm: LearningAlgorithm,
    /// Feedback mechanisms
    pub feedback_mechanisms: Vec<FeedbackMechanism>,
    /// Stability constraints
    pub stability_constraints: StabilityConstraints,
}

impl Default for DynamicAdjustment {
    fn default() -> Self {
        Self {
            enabled: true,
            adjustment_interval: Duration::from_secs(300), // 5 minutes
            learning_algorithm: LearningAlgorithm::ReinforcementLearning,
            feedback_mechanisms: vec![
                FeedbackMechanism::PerformanceMetrics,
                FeedbackMechanism::EfficiencyMetrics,
            ],
            stability_constraints: StabilityConstraints::default(),
        }
    }
}

/// Learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    ReinforcementLearning,
    GradientDescent,
    BayesianOptimization,
    EvolutionaryAlgorithm,
    OnlineOptimization,
    ModelPredictiveControl,
}

/// Feedback mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackMechanism {
    PerformanceMetrics,
    EfficiencyMetrics,
    UserSatisfaction,
    SystemStability,
    CostMetrics,
    QualityOfService,
}

/// Stability constraints for dynamic adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConstraints {
    /// Maximum change per adjustment
    pub max_change_per_adjustment: f64,
    /// Minimum stability period
    pub min_stability_period: Duration,
    /// Oscillation prevention
    pub oscillation_prevention: bool,
    /// Change rate limiting
    pub change_rate_limiting: bool,
}

impl Default for StabilityConstraints {
    fn default() -> Self {
        Self {
            max_change_per_adjustment: 0.1,                 // 10%
            min_stability_period: Duration::from_secs(600), // 10 minutes
            oscillation_prevention: true,
            change_rate_limiting: true,
        }
    }
}
