// I/O Interfaces Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IOCapabilities {
    pub pcie_lanes: u32,
    pub network_interfaces: u32,
}
