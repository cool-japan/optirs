// Timeline profiling: per-operation event/resource-usage timelines.
//
// Split out of `profiling_integration/mod.rs` per the 2000-line file-size
// convention. Struct definitions and their `impl` are kept together in this
// one file (rather than split further) because nothing outside this file's
// own `impl TimelineProfiler` touches these types' private fields, so no
// visibility widening (`pub(super)`, ...) was needed for the split.

use std::collections::HashMap;
use std::time::{Instant, SystemTime};

use crate::error::Result;

use super::{CounterValue, ProfilingConfig};

/// Timeline profiler
pub struct TimelineProfiler<T> {
    /// Timeline sessions
    sessions: HashMap<String, TimelineSession>,

    /// Timeline data
    timeline_data: Vec<TimelineEntry>,

    _phantom: std::marker::PhantomData<T>,
}

/// Timeline session
#[derive(Debug)]
pub struct TimelineSession {
    /// Session ID
    pub id: String,

    /// Session start time
    pub start_time: Instant,

    /// Tracked operations
    pub operations: HashMap<String, OperationTimeline>,

    /// Session metadata
    pub metadata: TimelineMetadata,
}

/// Operation timeline
#[derive(Debug)]
pub struct OperationTimeline {
    /// Operation ID
    pub operation_id: String,

    /// Start time
    pub start_time: Instant,

    /// End time
    pub end_time: Option<Instant>,

    /// Timeline events
    pub events: Vec<TimelineEvent>,

    /// Resource usage timeline
    pub resource_usage: Vec<ResourceUsagePoint>,
}

/// Timeline event
#[derive(Debug)]
pub struct TimelineEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Event description
    pub description: String,

    /// Event data
    pub data: HashMap<String, String>,
}

/// Resource usage point in timeline
#[derive(Debug)]
pub struct ResourceUsagePoint {
    /// Timestamp
    pub timestamp: Instant,

    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// TPU utilization (0.0-1.0)
    pub tpu_utilization: f64,

    /// Power consumption (watts)
    pub power_consumption: f64,
}

/// Timeline entry
#[derive(Debug)]
pub struct TimelineEntry {
    /// Entry timestamp
    pub timestamp: Instant,

    /// Entry type
    pub entry_type: TimelineEntryType,

    /// Associated operation
    pub operation_id: Option<String>,

    /// Entry data
    pub data: TimelineEntryData,
}

/// Timeline entry types
#[derive(Debug)]
pub enum TimelineEntryType {
    /// Operation start
    OperationStart,

    /// Operation end
    OperationEnd,

    /// Resource allocation
    ResourceAllocation,

    /// Memory event
    MemoryEvent,

    /// Performance counter event
    CounterEvent,
}

/// Timeline entry data
#[derive(Debug)]
pub enum TimelineEntryData {
    /// Operation data
    Operation(OperationTimelineData),

    /// Resource data
    Resource(ResourceTimelineData),

    /// Memory data
    Memory(MemoryTimelineData),

    /// Counter data
    Counter(CounterTimelineData),
}

/// Operation timeline data
#[derive(Debug)]
pub struct OperationTimelineData {
    /// Operation name
    pub name: String,

    /// Input sizes
    pub input_sizes: Vec<usize>,

    /// Output sizes
    pub output_sizes: Vec<usize>,

    /// Compute intensity
    pub compute_intensity: f64,
}

/// Resource timeline data
#[derive(Debug)]
pub struct ResourceTimelineData {
    /// Resource type
    pub resource_type: String,

    /// Resource amount
    pub amount: usize,

    /// Utilization
    pub utilization: f64,
}

/// Memory timeline data
#[derive(Debug)]
pub struct MemoryTimelineData {
    /// Memory operation type
    pub operation_type: String,

    /// Memory address
    pub address: usize,

    /// Operation size
    pub size: usize,
}

/// Counter timeline data
#[derive(Debug)]
pub struct CounterTimelineData {
    /// Counter name
    pub counter_name: String,

    /// Counter value
    pub value: CounterValue,

    /// Counter delta
    pub delta: Option<f64>,
}

/// Timeline metadata
#[derive(Debug, Default)]
pub struct TimelineMetadata {
    /// Session name
    pub session_name: String,

    /// Start time
    pub start_time: Option<SystemTime>,

    /// End time
    pub end_time: Option<SystemTime>,

    /// Total operations
    pub total_operations: usize,
}

/// Timeline configuration
#[derive(Debug)]
pub struct TimelineConfig {
    /// Enable detailed operation tracking
    pub detailed_operations: bool,

    /// Include resource usage
    pub include_resources: bool,

    /// Timeline resolution (microseconds)
    pub resolution_us: u64,

    /// Maximum timeline entries
    pub max_entries: usize,
}

impl<T> TimelineProfiler<T> {
    /// Create new timeline profiler
    pub fn new(_config: &ProfilingConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            timeline_data: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Start timeline session
    pub fn start_timeline(&mut self, session_id: &str) -> Result<()> {
        let session = TimelineSession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            operations: HashMap::new(),
            metadata: TimelineMetadata::default(),
        };

        self.sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Reset timeline profiler
    pub fn reset(&mut self) {
        self.sessions.clear();
        self.timeline_data.clear();
    }
}
