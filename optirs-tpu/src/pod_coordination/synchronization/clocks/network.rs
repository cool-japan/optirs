// Network Sync Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
}

impl Default for LoadBalancingAlgorithm {
    fn default() -> Self {
        Self::RoundRobin
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessagePassingConfig {
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTopology {
    Star,
    Ring,
    Mesh,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::Mesh
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMessageType {
    Request,
    Response,
    Heartbeat,
}

impl Default for SyncMessageType {
    fn default() -> Self {
        Self::Request
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkTimeProtocol {
    NTP,
    SNTP,
    PTP,
}

impl Default for NetworkTimeProtocol {
    fn default() -> Self {
        Self::NTP
    }
}
