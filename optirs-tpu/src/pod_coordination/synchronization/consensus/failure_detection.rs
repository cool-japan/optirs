// Failure Detection Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    Crash,
    Byzantine,
    Network,
}

impl Default for FailureType {
    fn default() -> Self { Self::Crash }
}

#[derive(Debug, Clone, Default)]
pub struct FailureDetector {
    pub suspected_failures: Vec<DeviceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionMethod {
    Heartbeat,
    Gossip,
    Adaptive,
}

impl Default for DetectionMethod {
    fn default() -> Self { Self::Heartbeat }
}
