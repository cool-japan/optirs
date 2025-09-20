// QoS Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BandwidthAllocation {
    pub allocated_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FlowControl;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriorityScheduling;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QoSClass {
    RealTime,
    Interactive,
    BestEffort,
}

impl Default for QoSClass {
    fn default() -> Self {
        Self::BestEffort
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QoSConfig {
    pub class: QoSClass,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QoSRequirements {
    pub min_bandwidth_gbps: f64,
    pub max_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReliabilityRequirements;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficClass {
    Control,
    Data,
    Management,
}

impl Default for TrafficClass {
    fn default() -> Self {
        Self::Data
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPriority {
    High,
    Medium,
    Low,
}

impl Default for TrafficPriority {
    fn default() -> Self {
        Self::Medium
    }
}
