// I/O Interfaces Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IOCapabilities {
    pub pcie_lanes: u32,
    pub network_interfaces: u32,
}
