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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BarrierState {
    Active,
    Waiting,
    Complete,
}

impl Default for BarrierState {
    fn default() -> Self {
        Self::Waiting
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BarrierType {
    Global,
    Local,
    Hierarchical,
}

impl Default for BarrierType {
    fn default() -> Self {
        Self::Global
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceStatus {
    Active,
    Idle,
    Failed,
}

impl Default for DeviceStatus {
    fn default() -> Self {
        Self::Idle
    }
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
