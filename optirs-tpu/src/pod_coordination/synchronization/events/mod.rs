// Events Module

use crate::pod_coordination::types::*;
use std::collections::HashMap;

pub mod compression;
pub mod core;
pub mod delivery;
pub mod filtering;
pub mod handlers;
pub mod persistence;
pub mod queue;
pub mod routing;
pub mod utilities;

// Re-export main types explicitly to avoid ambiguous glob re-exports
// Submodules with conflicts (monitoring, performance, storage) can be accessed via full paths
pub use compression::*;
pub use delivery::*;
pub use queue::*;
pub use routing::*;
pub use utilities::*;

pub type EventStatistics = HashMap<String, f64>;

#[derive(Debug, Clone, Default)]
pub struct SynchronizationStatistics {
    pub event_stats: EventStatistics,
}
