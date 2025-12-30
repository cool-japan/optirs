// Batch Coordination Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AggregationMethod {
    Sum,
    #[default]
    Average,
    Max,
    Min,
}

#[derive(Debug, Clone, Default)]
pub struct BatchCoordinator {
    pub batch_size: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BatchData {
    pub batch_id: u64,
    pub size: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BatchExecution {
    pub batch_id: u64,
    pub status: BatchExecutionResult,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum BatchExecutionResult {
    #[default]
    Success,
    PartialSuccess,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct BatchMetadata {
    pub batch_id: u64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone)]
pub struct BatchPartition<T> {
    pub data: scirs2_core::ndarray::Array<T, scirs2_core::ndarray::IxDyn>,
    pub indices: Vec<usize>,
    pub status: PartitionStatus,
    pub device: DeviceId,
    pub dependencies: Vec<PartitionId>,
    pub created_at: std::time::Instant,
    pub processing_start: Option<std::time::Instant>,
    pub completed_at: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum BatchPriority {
    Low,
    #[default]
    Normal,
    High,
}

#[derive(Debug, Clone, Default)]
pub struct BatchProgress {
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CachingStrategy {
    None,
    #[default]
    LRU,
    Adaptive,
}

#[derive(Debug, Clone, Default)]
pub struct DataPartitioning {
    pub num_partitions: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceExecutionStatistics {
    pub device_id: DeviceId,
    pub batches_processed: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum DistributionStrategy {
    Broadcast,
    #[default]
    Scatter,
    AllGather,
}

pub type PartitionId = u64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PartitionStatus {
    #[default]
    Ready,
    Processing,
    Complete,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineStage {
    pub stage_id: u64,
    pub status: PipelineStageStatus,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PipelineStageStatus {
    #[default]
    Idle,
    Running,
    Complete,
}

#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub accuracy: f64,
    pub loss: f64,
}

#[derive(Debug, Clone, Default)]
pub struct BatchCoordinationStatistics {
    pub total_batches: u64,
    pub avg_batch_time_ms: f64,
}
