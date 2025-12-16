// Reliability Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorDetectionConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ErrorDetectionMethod {
    #[default]
    CRC,
    Checksum,
    Parity,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FaultToleranceConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecoveryConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecoveryStrategy {
    #[default]
    Retry,
    Failover,
    Rebuild,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RedundancyConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReliabilityConfig;
