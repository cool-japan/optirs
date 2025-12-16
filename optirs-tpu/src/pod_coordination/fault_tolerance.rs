// Fault Tolerance Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct CheckpointInfo {
    pub checkpoint_id: u64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CheckpointType {
    #[default]
    Full,
    Incremental,
    Differential,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConsistencyLevel {
    #[default]
    Strong,
    Eventual,
    Weak,
}

#[derive(Debug, Clone, Default)]
pub struct DetectionConfig {
    pub timeout_ms: u64,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FailureDetectionAlgorithm {
    #[default]
    Heartbeat,
    Gossip,
    Adaptive,
}

#[derive(Debug, Clone, Default)]
pub struct FailureDetector {
    pub algorithm: FailureDetectionAlgorithm,
    pub config: DetectionConfig,
}

#[derive(Debug, Clone, Default)]
pub struct FailureInfo {
    pub device_id: DeviceId,
    pub timestamp_ms: u64,
    pub failure_type: FailureType,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FailureStatus {
    #[default]
    Detected,
    Recovering,
    Recovered,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FailureType {
    Hardware,
    Software,
    Network,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Default)]
pub struct FaultToleranceManager {
    pub detector: FailureDetector,
    pub recovery_strategy: RecoveryStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecoveryAction {
    #[default]
    Restart,
    Migrate,
    Replace,
    NoOp,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecoveryPriority {
    Critical,
    High,
    #[default]
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecoveryStrategy {
    #[default]
    Immediate,
    Delayed,
    Manual,
}

#[derive(Debug, Clone, Default)]
pub struct RedundancyConfig {
    pub replication_factor: u32,
    pub strategy: RedundancyStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RedundancyStrategy {
    #[default]
    Active,
    Passive,
    Hybrid,
}

#[derive(Debug, Clone, Default)]
pub struct FaultToleranceStatistics {
    pub total_failures: u64,
    pub recovered_failures: u64,
}
