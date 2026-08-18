// Power Management Module
//
// This module provides comprehensive power management functionality for TPU topology

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
