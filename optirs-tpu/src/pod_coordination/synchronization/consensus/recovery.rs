// Recovery Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryState {
    Normal,
    Recovering,
    Failed,
}

impl Default for RecoveryState {
    fn default() -> Self { Self::Normal }
}

#[derive(Debug, Clone, Default)]
pub struct RecoveryManager {
    pub state: RecoveryState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Checkpoint,
    LogReplay,
    FullRestore,
}

impl Default for RecoveryStrategy {
    fn default() -> Self { Self::Checkpoint }
}
