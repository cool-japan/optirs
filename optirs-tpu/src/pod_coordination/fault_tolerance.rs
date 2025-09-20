// Fault Tolerance Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct CheckpointInfo {
    pub checkpoint_id: u64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointType {
    Full,
    Incremental,
    Differential,
}

impl Default for CheckpointType {
    fn default() -> Self {
        Self::Full
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Weak,
}

impl Default for ConsistencyLevel {
    fn default() -> Self {
        Self::Strong
    }
}

#[derive(Debug, Clone, Default)]
pub struct DetectionConfig {
    pub timeout_ms: u64,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureDetectionAlgorithm {
    Heartbeat,
    Gossip,
    Adaptive,
}

impl Default for FailureDetectionAlgorithm {
    fn default() -> Self {
        Self::Heartbeat
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureStatus {
    Detected,
    Recovering,
    Recovered,
    Failed,
}

impl Default for FailureStatus {
    fn default() -> Self {
        Self::Detected
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    Hardware,
    Software,
    Network,
    Unknown,
}

impl Default for FailureType {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Default)]
pub struct FaultToleranceManager {
    pub detector: FailureDetector,
    pub recovery_strategy: RecoveryStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    Restart,
    Migrate,
    Replace,
    NoOp,
}

impl Default for RecoveryAction {
    fn default() -> Self {
        Self::Restart
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryPriority {
    Critical,
    High,
    Normal,
    Low,
}

impl Default for RecoveryPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Immediate,
    Delayed,
    Manual,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::Immediate
    }
}

#[derive(Debug, Clone, Default)]
pub struct RedundancyConfig {
    pub replication_factor: u32,
    pub strategy: RedundancyStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyStrategy {
    Active,
    Passive,
    Hybrid,
}

impl Default for RedundancyStrategy {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Debug, Clone, Default)]
pub struct FaultToleranceStatistics {
    pub total_failures: u64,
    pub recovered_failures: u64,
}
