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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum QoSClass {
    RealTime,
    Interactive,
    #[default]
    BestEffort,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TrafficClass {
    Control,
    #[default]
    Data,
    Management,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TrafficPriority {
    High,
    #[default]
    Medium,
    Low,
}
