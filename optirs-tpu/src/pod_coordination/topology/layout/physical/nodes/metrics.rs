// Node Metrics Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeMetrics {
    pub utilization_percent: f64,
    pub temperature_celsius: f64,
}
