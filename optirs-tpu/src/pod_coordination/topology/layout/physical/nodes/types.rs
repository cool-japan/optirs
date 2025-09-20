// Core Node Types Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// Import from parent module
use super::super::positioning::NodeId;

// Re-exports from sibling modules
use super::configuration::NodeConfiguration;
use super::interfaces::IOCapabilities;
use super::memory::MemoryCapabilities;
use super::metrics::NodeMetrics;
use super::networking::NetworkCapabilities;
use super::physical::NodePhysicalProperties;
use super::processing::ProcessingCapabilities;
use super::reliability::ReliabilityMetrics;
use super::storage::StorageCapabilities;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Compute,
    Storage,
    Network,
    Management,
}

impl Default for NodeType {
    fn default() -> Self {
        Self::Compute
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Offline,
    Maintenance,
    Failed,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self::Offline
    }
}

#[derive(Debug, Clone, Default)]
pub struct NodeCapabilities {
    pub processing: ProcessingCapabilities,
    pub memory: MemoryCapabilities,
    pub storage: StorageCapabilities,
    pub networking: NetworkCapabilities,
    pub io: IOCapabilities,
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub device_id: DeviceId,
    pub node_type: NodeType,
    pub status: NodeStatus,
    pub capabilities: NodeCapabilities,
    pub physical_properties: NodePhysicalProperties,
    pub configuration: NodeConfiguration,
    pub metrics: NodeMetrics,
    pub reliability: ReliabilityMetrics,
    pub last_update: Instant,
}

impl Default for NodeInfo {
    fn default() -> Self {
        Self {
            node_id: 0,
            device_id: DeviceId::default(),
            node_type: NodeType::default(),
            status: NodeStatus::default(),
            capabilities: NodeCapabilities::default(),
            physical_properties: NodePhysicalProperties::default(),
            configuration: NodeConfiguration::default(),
            metrics: NodeMetrics::default(),
            reliability: ReliabilityMetrics::default(),
            last_update: Instant::now(),
        }
    }
}
