// Node Metrics Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeMetrics {
    pub utilization_percent: f64,
    pub temperature_celsius: f64,
}
