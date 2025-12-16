// Topology Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod communication;
pub mod config;
pub mod core;
pub mod device_layout;
pub mod graph_management;
pub mod layout;
pub mod monitoring;
pub mod network;
pub mod optimization;
pub mod power;
pub mod power_management;

pub use communication::*;
pub use config::*;
pub use core::*;
pub use device_layout::*;
pub use graph_management::*;
pub use monitoring::*;
pub use network::*;
pub use optimization::*;
pub use power::*;
pub use power_management::*;

// Define missing types
pub type BandwidthMatrix = HashMap<(DeviceId, DeviceId), f64>;
pub type LatencyMatrix = HashMap<(DeviceId, DeviceId), f64>;

#[derive(Debug, Clone, Default)]
pub struct CommunicationStep {
    pub step_type: CommunicationStepType,
    pub devices: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CommunicationStepType {
    #[default]
    Send,
    Receive,
    Broadcast,
    AllReduce,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceLayout {
    pub devices: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LinkType {
    #[default]
    Direct,
    Switch,
    Hierarchical,
}

pub type RoutingTable = HashMap<(DeviceId, DeviceId), Vec<DeviceId>>;

#[derive(Debug, Clone, Default)]
pub struct TopologyProperties {
    pub diameter: usize,
    pub bisection_bandwidth: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TopologyStatistics {
    pub total_links: usize,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationTopology {
    pub bandwidth_matrix: BandwidthMatrix,
    pub latency_matrix: LatencyMatrix,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationLink {
    pub source: DeviceId,
    pub target: DeviceId,
    pub link_type: LinkType,
}

#[derive(Debug, Clone, Default)]
pub struct TopologyManager {
    pub topology: CommunicationTopology,
}
