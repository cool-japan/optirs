// Batch Coordination Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Sum,
    Average,
    Max,
    Min,
}

impl Default for AggregationMethod {
    fn default() -> Self {
        Self::Average
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchExecutionResult {
    Success,
    PartialSuccess,
    Failed,
}

impl Default for BatchExecutionResult {
    fn default() -> Self {
        Self::Success
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchPriority {
    Low,
    Normal,
    High,
}

impl Default for BatchPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Default)]
pub struct BatchProgress {
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachingStrategy {
    None,
    LRU,
    Adaptive,
}

impl Default for CachingStrategy {
    fn default() -> Self {
        Self::LRU
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionStrategy {
    Broadcast,
    Scatter,
    AllGather,
}

impl Default for DistributionStrategy {
    fn default() -> Self {
        Self::Scatter
    }
}

pub type PartitionId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStatus {
    Ready,
    Processing,
    Complete,
}

impl Default for PartitionStatus {
    fn default() -> Self {
        Self::Ready
    }
}

#[derive(Debug, Clone, Default)]
pub struct PipelineStage {
    pub stage_id: u64,
    pub status: PipelineStageStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStageStatus {
    Idle,
    Running,
    Complete,
}

impl Default for PipelineStageStatus {
    fn default() -> Self {
        Self::Idle
    }
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
