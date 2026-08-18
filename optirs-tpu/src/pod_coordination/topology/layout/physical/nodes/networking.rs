// Networking Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkCapabilities {
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}
