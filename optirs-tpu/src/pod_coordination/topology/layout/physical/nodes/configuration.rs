// Node Configuration Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeConfiguration {
    pub max_power_watts: f64,
    pub cooling_type: CoolingType,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CoolingType {
    #[default]
    Air,
    Liquid,
    Hybrid,
}
