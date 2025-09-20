// Network Traffic Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type FlowId = u64;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrafficFlow {
    pub flow_id: FlowId,
    pub bandwidth_mbps: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TrafficManager {
    pub flows: HashMap<FlowId, TrafficFlow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPattern {
    Uniform,
    Burst,
    Adaptive,
}

impl Default for TrafficPattern {
    fn default() -> Self { Self::Uniform }
}
