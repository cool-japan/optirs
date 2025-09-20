// Power Management Module
//
// This module provides comprehensive power management functionality for TPU topology

pub mod allocation_strategies;
pub mod budget;
pub mod budget_config;
pub mod budget_monitoring;
pub mod budget_optimization;
pub mod device_allocation;
pub mod distribution;
pub mod efficiency;
pub mod emergency_management;
pub mod monitoring;
pub mod power_constraints;
pub mod scheduling_management;
pub mod supplies;
pub mod thermal;

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Core power management types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerRequirements {
    pub min_watts: f64,
    pub max_watts: f64,
    pub typical_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerBudget {
    pub total_watts: f64,
    pub allocated_watts: f64,
    pub reserved_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerConfiguration {
    pub power_requirements: PowerRequirements,
    pub power_budget: PowerBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerDistribution {
    pub distribution_units: Vec<PowerDistributionUnit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerDistributionUnit {
    pub id: String,
    pub capacity_watts: f64,
    pub current_load_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerEfficiencyManager {
    pub efficiency_target: f64,
    pub current_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerManagementSystem {
    pub configuration: PowerConfiguration,
    pub distribution: PowerDistribution,
    pub efficiency_manager: PowerEfficiencyManager,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerMonitoring {
    pub sample_interval_ms: u64,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerSupply {
    pub id: String,
    pub capacity_watts: f64,
    pub efficiency_rating: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalManagement {
    pub max_temperature_celsius: f64,
    pub current_temperature_celsius: f64,
    pub cooling_capacity_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnergyHarvesting {
    pub enabled: bool,
    pub harvested_watts: f64,
}

// Re-export submodule types
pub use allocation_strategies::*;
pub use budget::*;
pub use budget_config::*;
pub use budget_monitoring::*;
pub use budget_optimization::*;
pub use device_allocation::*;
pub use distribution::*;
pub use efficiency::*;
pub use emergency_management::*;
pub use monitoring::*;
pub use power_constraints::*;
pub use scheduling_management::*;
pub use supplies::*;
pub use thermal::*;
