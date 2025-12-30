// Core topology module
//
// This module provides the core topology management functionality

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopologyManager {
    pub config: HashMap<String, String>,
    pub devices: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopologyEventManager {
    pub events: Vec<TopologyEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopologyPerformanceMonitor {
    pub metrics: HashMap<String, f64>,
    pub sampling_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TopologyEvent {
    DeviceAdded(DeviceId),
    DeviceRemoved(DeviceId),
    #[default]
    TopologyChanged,
    PerformanceUpdate(HashMap<String, f64>),
}
