// Memory Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryCapabilities {
    pub total_gb: u64,
    pub bandwidth_gbps: f64,
}
