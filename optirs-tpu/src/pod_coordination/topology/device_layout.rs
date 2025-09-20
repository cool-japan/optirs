// Device Layout Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceCapabilities {
    pub compute_flops: f64,
    pub memory_gb: f64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceConfig {
    pub device_id: DeviceId,
    pub capabilities: DeviceCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceGroup {
    pub group_id: String,
    pub devices: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub name: String,
    pub capabilities: DeviceCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceLayoutManager {
    pub devices: HashMap<DeviceId, DeviceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceNode {
    pub node_id: String,
    pub devices: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayoutOptimizer {
    pub optimization_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayoutStatistics {
    pub total_devices: usize,
    pub active_devices: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LogicalLayout {
    pub topology_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhysicalLayout {
    pub rack_layout: HashMap<String, Vec<DeviceId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlacementPolicy {
    pub policy_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalStatus {
    pub temperature_celsius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalZone {
    pub zone_id: String,
    pub devices: Vec<DeviceId>,
    pub status: ThermalStatus,
}
