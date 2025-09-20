// Coordination Configuration Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchParallelizationStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
    Adaptive,
}

impl Default for BatchParallelizationStrategy {
    fn default() -> Self {
        Self::DataParallel
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationPattern {
    AllReduce,
    AllGather,
    AllToAll,
    Ring,
    Broadcast,
}

impl Default for CommunicationPattern {
    fn default() -> Self {
        Self::AllReduce
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    Centralized,
    Decentralized,
    Hierarchical,
    Adaptive,
    Mesh,
}

impl Default for CoordinationStrategy {
    fn default() -> Self {
        Self::Centralized
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientAggregationMethod {
    Average,
    Sum,
    WeightedAverage,
    LocalSGD,
}

impl Default for GradientAggregationMethod {
    fn default() -> Self {
        Self::Average
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    Adaptive,
    LatencyAware,
    BandwidthAware,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::Dynamic
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryManagementStrategy {
    StaticPartitioning,
    DynamicPartitioning,
    Pooling,
}

impl Default for MemoryManagementStrategy {
    fn default() -> Self {
        Self::DynamicPartitioning
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PodCoordinationConfig {
    pub topology: crate::pod_coordination::PodTopology,
    pub num_devices: usize,
    pub coordination_strategy: CoordinationStrategy,
    pub communication_pattern: CommunicationPattern,
    pub synchronization_mode: SynchronizationMode,
    pub batch_strategy: BatchParallelizationStrategy,
    pub gradient_aggregation: GradientAggregationMethod,
    pub enable_fault_tolerance: bool,
    pub heartbeat_interval_ms: u64,
    pub operation_timeout_ms: u64,
    pub enable_performance_monitoring: bool,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub memory_management: MemoryManagementStrategy,
    pub adaptive_optimization: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynchronizationMode {
    Synchronous,
    Asynchronous,
    BulkSynchronous,
    Adaptive,
}

impl Default for SynchronizationMode {
    fn default() -> Self {
        Self::Synchronous
    }
}
