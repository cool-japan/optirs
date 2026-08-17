// Events Module

use crate::pod_coordination::types::*;
use std::collections::HashMap;

pub mod handlers;
pub mod queue;

// Re-export main types explicitly to avoid ambiguous glob re-exports
// Submodules with conflicts (monitoring, performance, storage) can be accessed via full paths
pub use queue::*;

pub type EventStatistics = HashMap<String, f64>;

#[derive(Debug, Clone, Default)]
pub struct SynchronizationStatistics {
    pub event_stats: EventStatistics,
}
