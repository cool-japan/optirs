// Paxos Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaxosState {
    Follower,
    Candidate,
    Leader,
}

impl Default for PaxosState {
    fn default() -> Self { Self::Follower }
}

#[derive(Debug, Clone, Default)]
pub struct PaxosNode {
    pub state: PaxosState,
    pub term: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PaxosConfig {
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
}
