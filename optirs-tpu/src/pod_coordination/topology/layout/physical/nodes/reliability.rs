// Reliability Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReliabilityMetrics {
    pub uptime_hours: f64,
    pub mtbf_hours: f64,
}
