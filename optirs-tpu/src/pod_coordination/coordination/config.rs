// Coordination Configuration Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BatchParallelizationStrategy {
    #[default]
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
    Adaptive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CommunicationPattern {
    #[default]
    AllReduce,
    AllGather,
    AllToAll,
    Ring,
    Broadcast,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CoordinationStrategy {
    #[default]
    Centralized,
    Decentralized,
    Hierarchical,
    Adaptive,
    Mesh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GradientAggregationMethod {
    #[default]
    Average,
    Sum,
    WeightedAverage,
    LocalSGD,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LoadBalancingStrategy {
    Static,
    #[default]
    Dynamic,
    Adaptive,
    LatencyAware,
    BandwidthAware,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryManagementStrategy {
    StaticPartitioning,
    #[default]
    DynamicPartitioning,
    Pooling,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SynchronizationMode {
    #[default]
    Synchronous,
    Asynchronous,
    BulkSynchronous,
    Adaptive,
}
