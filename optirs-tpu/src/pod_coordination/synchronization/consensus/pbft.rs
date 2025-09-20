// Pbft Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftState {
    Follower,
    Candidate,
    Leader,
}

impl Default for PbftState {
    fn default() -> Self { Self::Follower }
}

#[derive(Debug, Clone, Default)]
pub struct PbftNode {
    pub state: PbftState,
    pub term: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PbftConfig {
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
}
