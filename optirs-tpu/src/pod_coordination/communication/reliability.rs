// Reliability Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorDetectionConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorDetectionMethod {
    CRC,
    Checksum,
    Parity,
}

impl Default for ErrorDetectionMethod {
    fn default() -> Self {
        Self::CRC
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FaultToleranceConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecoveryConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Retry,
    Failover,
    Rebuild,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::Retry
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RedundancyConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReliabilityConfig;
