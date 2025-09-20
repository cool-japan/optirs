// Sync Manager Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SyncError;

pub type Result<T> = std::result::Result<T, SyncError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMode {
    Blocking,
    NonBlocking,
    Adaptive,
}

impl Default for SyncMode {
    fn default() -> Self { Self::Blocking }
}

#[derive(Debug, Clone, Default)]
pub struct SyncManager {
    pub mode: SyncMode,
}
