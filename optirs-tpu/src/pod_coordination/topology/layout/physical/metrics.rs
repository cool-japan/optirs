// Physical Layout Metrics
//
// This module handles comprehensive metrics collection and management for physical layouts
// including space utilization, power, thermal, connectivity, and performance metrics.

/// Physical layout metrics
#[derive(Debug, Clone)]
pub struct PhysicalLayoutMetrics {
    /// Space utilization
    pub space_utilization: SpaceUtilizationMetrics,
    /// Power metrics
    pub power_metrics: PhysicalPowerMetrics,
    /// Thermal metrics
    pub thermal_metrics: PhysicalThermalMetrics,
    /// Connectivity metrics
    pub connectivity_metrics: PhysicalConnectivityMetrics,
    /// Performance metrics
    pub performance_metrics: PhysicalPerformanceMetrics,
}

impl Default for PhysicalLayoutMetrics {
    fn default() -> Self {
        Self {
            space_utilization: SpaceUtilizationMetrics::default(),
            power_metrics: PhysicalPowerMetrics::default(),
            thermal_metrics: PhysicalThermalMetrics::default(),
            connectivity_metrics: PhysicalConnectivityMetrics::default(),
            performance_metrics: PhysicalPerformanceMetrics::default(),
        }
    }
}

/// Space utilization metrics
#[derive(Debug, Clone)]
pub struct SpaceUtilizationMetrics {
    /// Total space available (cubic meters)
    pub total_space: f64,
    /// Space occupied (cubic meters)
    pub occupied_space: f64,
    /// Space utilization percentage
    pub utilization_percentage: f64,
    /// Space efficiency score
    pub efficiency_score: f64,
}

impl Default for SpaceUtilizationMetrics {
    fn default() -> Self {
        Self {
            total_space: 100.0,
            occupied_space: 50.0,
            utilization_percentage: 50.0,
            efficiency_score: 0.8,
        }
    }
}

/// Physical power metrics
#[derive(Debug, Clone)]
pub struct PhysicalPowerMetrics {
    /// Total power consumption (W)
    pub total_power: f64,
    /// Power density (W/cubic meter)
    pub power_density: f64,
    /// Power efficiency
    pub power_efficiency: f64,
    /// Power distribution efficiency
    pub distribution_efficiency: f64,
}

impl Default for PhysicalPowerMetrics {
    fn default() -> Self {
        Self {
            total_power: 5000.0,
            power_density: 50.0,
            power_efficiency: 0.9,
            distribution_efficiency: 0.95,
        }
    }
}

/// Physical thermal metrics
#[derive(Debug, Clone)]
pub struct PhysicalThermalMetrics {
    /// Average temperature (Celsius)
    pub average_temperature: f64,
    /// Maximum temperature (Celsius)
    pub max_temperature: f64,
    /// Temperature variance
    pub temperature_variance: f64,
    /// Cooling efficiency
    pub cooling_efficiency: f64,
}

impl Default for PhysicalThermalMetrics {
    fn default() -> Self {
        Self {
            average_temperature: 25.0,
            max_temperature: 35.0,
            temperature_variance: 5.0,
            cooling_efficiency: 0.8,
        }
    }
}

/// Physical connectivity metrics
#[derive(Debug, Clone)]
pub struct PhysicalConnectivityMetrics {
    /// Total connections
    pub total_connections: usize,
    /// Average connection length (meters)
    pub average_connection_length: f64,
    /// Connection reliability
    pub connection_reliability: f64,
    /// Network diameter
    pub network_diameter: f64,
}

impl Default for PhysicalConnectivityMetrics {
    fn default() -> Self {
        Self {
            total_connections: 100,
            average_connection_length: 2.0,
            connection_reliability: 0.99,
            network_diameter: 10.0,
        }
    }
}

/// Physical performance metrics
#[derive(Debug, Clone)]
pub struct PhysicalPerformanceMetrics {
    /// Overall layout score
    pub layout_score: f64,
    /// Optimization convergence
    pub optimization_convergence: f64,
    /// Validation pass rate
    pub validation_pass_rate: f64,
    /// Layout stability
    pub layout_stability: f64,
}

impl Default for PhysicalPerformanceMetrics {
    fn default() -> Self {
        Self {
            layout_score: 85.0,
            optimization_convergence: 0.95,
            validation_pass_rate: 0.98,
            layout_stability: 0.9,
        }
    }
}