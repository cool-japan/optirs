// Synchronization Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod barriers;
pub mod clocks;
pub mod config;
pub mod consensus;
pub mod core;
pub mod deadlock;
pub mod events;

pub use barriers::*;
pub use config::*;
pub use core::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BarrierId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BarrierState {
    Active,
    #[default]
    Waiting,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BarrierType {
    #[default]
    Global,
    Local,
    Hierarchical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeviceStatus {
    Active,
    #[default]
    Idle,
    Failed,
}

#[derive(Debug, Clone)]
pub struct SyncEvent;

#[derive(Debug, Clone)]
pub enum SyncEventType {
    Started,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct SyncEventData {
    pub event_type: SyncEventType,
}

#[derive(Debug, Clone, Default)]
pub struct SynchronizationManager;

// Re-export from submodules
pub use events::SynchronizationStatistics;
