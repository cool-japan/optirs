// Network Sync Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingAlgorithm {
    #[default]
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessagePassingConfig {
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MessagePriority {
    Low,
    #[default]
    Normal,
    High,
}

#[derive(Debug, Clone, Default)]
pub struct NetworkFaultTolerance {
    pub redundancy_factor: u32,
}

#[derive(Debug, Clone, Default)]
pub struct NetworkLoadBalancing {
    pub algorithm: LoadBalancingAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkSyncConfig {
    pub timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct NetworkSyncError;

impl std::fmt::Display for NetworkSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Network synchronization error")
    }
}

#[derive(Debug, Clone, Default)]
pub struct NetworkSynchronizationManager {
    pub config: NetworkSyncConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum NetworkTopology {
    Star,
    Ring,
    #[default]
    Mesh,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SyncMessageType {
    #[default]
    Request,
    Response,
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum NetworkTimeProtocol {
    #[default]
    NTP,
    SNTP,
    PTP,
}
