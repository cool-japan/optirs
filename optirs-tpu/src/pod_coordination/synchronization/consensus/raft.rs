// Raft Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

impl Default for RaftState {
    fn default() -> Self { Self::Follower }
}

#[derive(Debug, Clone, Default)]
pub struct RaftNode {
    pub state: RaftState,
    pub term: u64,
}

#[derive(Debug, Clone, Default)]
pub struct RaftConfig {
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
}
