//! TPU Synchronization and Communication Primitives
//!
//! This module provides synchronization mechanisms for TPU pods,
//! including barriers, all-reduce operations, point-to-point communication,
//! and collective operations for distributed optimization algorithms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::coordination::TpuDeviceId;

/// Synchronization manager for TPU pods
#[derive(Debug)]
pub struct SynchronizationManager {
    /// Device ID for this manager instance
    device_id: TpuDeviceId,

    /// All devices in the pod
    all_devices: Vec<TpuDeviceId>,

    /// Active synchronization barriers
    active_barriers: Arc<Mutex<HashMap<String, Barrier>>>,

    /// Collective operation handlers
    collective_handlers: HashMap<CollectiveOpType, Box<dyn CollectiveHandler>>,

    /// Communication topology
    topology: CommunicationTopology,

    /// Synchronization statistics
    stats: Arc<Mutex<SynchronizationStats>>,
}

/// Synchronization barrier for coordinating multiple TPU devices
#[derive(Debug)]
pub struct Barrier {
    /// Barrier identifier
    pub id: String,

    /// Devices that must reach this barrier
    pub required_devices: Vec<TpuDeviceId>,

    /// Shared, reusable episode state guarded by the barrier's own mutex.
    ///
    /// Arrival bookkeeping lives here rather than in the `active_barriers` map
    /// so that a waiting thread never holds the map lock while blocked.
    pub condition: Arc<(Mutex<BarrierEpisode>, Condvar)>,

    /// Timeout for the barrier
    pub timeout: Duration,

    /// Creation timestamp
    pub created_at: Instant,

    /// Barrier state
    pub state: BarrierState,
}

/// Reusable per-round state of a [`Barrier`].
///
/// The barrier is a generation-counter barrier: every completed round bumps
/// `generation`, and a waiter blocks until it observes a generation different
/// from the one it arrived in. This is what makes the barrier immune to
/// spurious condvar wakeups and safe to reuse for consecutive rounds.
///
/// The previous implementation never mutated the condvar's predicate at all, so
/// every waiter other than the last to arrive blocked until the timeout expired
/// and then reported `BarrierTimeout`.
#[derive(Debug, Default)]
pub struct BarrierEpisode {
    /// Number of rounds this barrier has completed.
    pub generation: u64,

    /// Devices that have arrived in the round currently in progress.
    pub arrived: Vec<TpuDeviceId>,

    /// Set when the barrier is cancelled so that waiters are released with an
    /// error instead of hanging until their timeout.
    pub aborted: bool,
}

/// State of a synchronization barrier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierState {
    /// Barrier is active and waiting for devices
    Active,
    /// All devices have arrived
    Complete,
    /// Barrier timed out
    TimedOut,
    /// Barrier was cancelled
    Cancelled,
}

/// Communication topology for TPU pod
#[derive(Debug, Clone)]
pub struct CommunicationTopology {
    /// Topology type
    pub topology_type: TopologyType,

    /// Device connections
    pub connections: HashMap<TpuDeviceId, Vec<TpuDeviceId>>,

    /// Communication rings (for ring all-reduce)
    pub rings: Vec<Vec<TpuDeviceId>>,

    /// Tree structure (for tree-based operations)
    pub tree: Option<CommunicationTree>,
}

/// Type of communication topology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Linear topology (chain)
    Linear,
    /// Ring topology
    Ring,
    /// Tree topology
    Tree,
    /// Mesh topology
    Mesh,
    /// Torus topology
    Torus,
    /// Custom topology
    Custom,
}

/// Tree structure for hierarchical communication
#[derive(Debug, Clone)]
pub struct CommunicationTree {
    /// Root device
    pub root: TpuDeviceId,

    /// Parent-child relationships
    pub parent_child: HashMap<TpuDeviceId, Vec<TpuDeviceId>>,

    /// Child-parent relationships
    pub child_parent: HashMap<TpuDeviceId, TpuDeviceId>,

    /// Tree depth
    pub depth: u32,
}

/// Collective operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectiveOpType {
    /// All-reduce operation
    AllReduce,
    /// All-gather operation
    AllGather,
    /// Reduce-scatter operation
    ReduceScatter,
    /// Broadcast operation
    Broadcast,
    /// All-to-all operation
    AllToAll,
    /// Barrier synchronization
    Barrier,
}

/// Reduction operations for collective operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
    /// Product reduction
    Product,
    /// Logical AND
    LogicalAnd,
    /// Logical OR
    LogicalOr,
}

/// Data buffers a collective operation reads and writes.
///
/// Each entry is one participating rank's local buffer, in the same order as
/// [`CollectiveOpRequest::devices`]. There is no TPU in this environment, so
/// "ranks" are simulated buffers held in this process; the reduction arithmetic
/// and the ring communication schedule are nonetheless real, and the result is
/// bit-for-bit what a hardware ring all-reduce would produce.
#[derive(Debug, Clone, Default)]
pub struct CollectiveBuffers {
    /// One buffer per rank.
    pub ranks: Vec<Vec<f64>>,
}

impl CollectiveBuffers {
    /// Create buffers from per-rank data.
    pub fn new(ranks: Vec<Vec<f64>>) -> Self {
        Self { ranks }
    }

    /// Number of participating ranks.
    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    /// Common buffer length, or an error when the ranks disagree.
    pub fn uniform_len(&self) -> Result<usize, SynchronizationError> {
        let mut lengths = self.ranks.iter().map(|buffer| buffer.len());
        let Some(first) = lengths.next() else {
            return Ok(0);
        };
        if lengths.all(|len| len == first) {
            Ok(first)
        } else {
            Err(SynchronizationError::InvalidOperation {
                reason: "collective buffers must all have the same length".to_string(),
            })
        }
    }

    /// Total bytes held across all ranks.
    pub fn total_bytes(&self) -> usize {
        self.ranks
            .iter()
            .map(|buffer| buffer.len() * std::mem::size_of::<f64>())
            .sum()
    }
}

/// Shared handle to the buffers a collective operates on.
pub type SharedBuffers = Arc<Mutex<CollectiveBuffers>>;

/// Collective operation request
#[derive(Debug, Clone)]
pub struct CollectiveOpRequest {
    /// Operation identifier
    pub id: String,

    /// Operation type
    pub op_type: CollectiveOpType,

    /// Participating devices
    pub devices: Vec<TpuDeviceId>,

    /// Root device (for operations like broadcast)
    pub root_device: Option<TpuDeviceId>,

    /// Reduction operation (for reduce-type operations)
    pub reduction_op: Option<ReductionOp>,

    /// Data size in bytes
    ///
    /// Used for reporting only when `buffers` is present; the real size is
    /// derived from the buffers themselves.
    pub data_size: usize,

    /// Buffers the operation reduces or moves.
    ///
    /// `None` means the caller only wants the operation validated and
    /// scheduled; handlers reject that rather than pretending to move data.
    pub buffers: Option<SharedBuffers>,

    /// Timeout for the operation
    pub timeout: Duration,

    /// Priority level
    pub priority: OperationPriority,
}

/// Priority levels for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Result of a collective operation
#[derive(Debug)]
pub struct CollectiveOpResult {
    /// Operation ID
    pub id: String,

    /// Success/failure status
    pub status: OperationStatus,

    /// Duration of the operation
    pub duration: Duration,

    /// Bandwidth achieved (GB/s)
    pub bandwidth_gb_s: f64,

    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Status of an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationStatus {
    Success,
    Failed,
    TimedOut,
    Cancelled,
}

/// Trait for collective operation handlers
pub trait CollectiveHandler: Send + Sync + std::fmt::Debug {
    /// Execute collective operation
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError>;

    /// Get estimated execution time
    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration;

    /// Check if handler supports the operation
    fn supports_operation(&self, op_type: CollectiveOpType) -> bool;
}

/// Synchronization statistics
#[derive(Debug, Default, Clone)]
pub struct SynchronizationStats {
    /// Total number of barriers created
    pub barriers_created: u64,

    /// Total number of barriers completed
    pub barriers_completed: u64,

    /// Total number of barriers timed out
    pub barriers_timed_out: u64,

    /// Total collective operations
    pub collective_ops_total: u64,

    /// Successful collective operations
    pub collective_ops_success: u64,

    /// Failed collective operations
    pub collective_ops_failed: u64,

    /// Total synchronization time (seconds)
    pub total_sync_time_seconds: f64,

    /// Average barrier wait time (seconds)
    pub avg_barrier_wait_time: f64,

    /// Total data transferred (bytes)
    pub total_data_transferred: u64,

    /// Average bandwidth (GB/s)
    pub avg_bandwidth_gb_s: f64,

    /// Total end-to-end time spent dispatching collective operations
    /// (seconds).
    ///
    /// Distinct from [`Self::total_sync_time_seconds`], which accumulates
    /// barrier *wait* time only. This one is measured by
    /// [`SynchronizationManager::execute_collective_op`] around the whole
    /// dispatch -- handler lookup, statistics locking and the data movement
    /// itself -- so it includes the coordination overhead a caller actually
    /// pays, not just the transfer kernel that
    /// [`CollectiveOpResult::duration`] reports.
    pub total_collective_time_seconds: f64,

    /// Mean of [`Self::total_collective_time_seconds`] over every collective
    /// operation attempted (successful or not).
    pub avg_collective_op_time: f64,
}

/// Errors that can occur during synchronization
#[derive(Debug, Error)]
pub enum SynchronizationError {
    #[error("Barrier timeout: {barrier_id}")]
    BarrierTimeout { barrier_id: String },

    #[error("Device not found: {device_id:?}")]
    DeviceNotFound { device_id: TpuDeviceId },

    #[error("Collective operation failed: {reason}")]
    CollectiveOpFailed { reason: String },

    #[error("Communication error: {reason}")]
    CommunicationError { reason: String },

    #[error("Topology error: {reason}")]
    TopologyError { reason: String },

    #[error("Operation cancelled: {operation_id}")]
    OperationCancelled { operation_id: String },

    #[error("Invalid operation: {reason}")]
    InvalidOperation { reason: String },
}

/// Lock the barrier registry, turning poisoning into an error instead of a panic.
fn lock_map(
    barriers: &Arc<Mutex<HashMap<String, Barrier>>>,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, Barrier>>, SynchronizationError> {
    barriers
        .lock()
        .map_err(|_| SynchronizationError::InvalidOperation {
            reason: "barrier registry lock is poisoned".to_string(),
        })
}

/// Lock the statistics record, turning poisoning into an error instead of a panic.
fn lock_stats(
    stats: &Arc<Mutex<SynchronizationStats>>,
) -> Result<std::sync::MutexGuard<'_, SynchronizationStats>, SynchronizationError> {
    stats
        .lock()
        .map_err(|_| SynchronizationError::InvalidOperation {
            reason: "synchronization statistics lock is poisoned".to_string(),
        })
}

impl SynchronizationManager {
    /// Create a new synchronization manager
    pub fn new(
        device_id: TpuDeviceId,
        all_devices: Vec<TpuDeviceId>,
        topology: CommunicationTopology,
    ) -> Self {
        Self::with_barrier_registry(
            device_id,
            all_devices,
            topology,
            Arc::new(Mutex::new(HashMap::new())),
        )
    }

    /// Create a manager that shares an existing barrier registry.
    ///
    /// Participants only rendezvous with each other if they observe the same
    /// [`Barrier`], so every manager taking part in a barrier must be built
    /// from the same registry handle.
    pub fn with_barrier_registry(
        device_id: TpuDeviceId,
        all_devices: Vec<TpuDeviceId>,
        topology: CommunicationTopology,
        active_barriers: Arc<Mutex<HashMap<String, Barrier>>>,
    ) -> Self {
        let mut collective_handlers: HashMap<CollectiveOpType, Box<dyn CollectiveHandler>> =
            HashMap::new();

        // Register default handlers
        collective_handlers.insert(
            CollectiveOpType::AllReduce,
            Box::new(AllReduceHandler::new()),
        );
        collective_handlers.insert(
            CollectiveOpType::AllGather,
            Box::new(AllGatherHandler::new()),
        );
        collective_handlers.insert(
            CollectiveOpType::Broadcast,
            Box::new(BroadcastHandler::new()),
        );
        collective_handlers.insert(
            CollectiveOpType::ReduceScatter,
            Box::new(ReduceScatterHandler::new()),
        );
        collective_handlers.insert(CollectiveOpType::AllToAll, Box::new(AllToAllHandler::new()));
        collective_handlers.insert(CollectiveOpType::Barrier, Box::new(BarrierHandler::new()));

        Self {
            device_id,
            all_devices,
            active_barriers,
            collective_handlers,
            topology,
            stats: Arc::new(Mutex::new(SynchronizationStats::default())),
        }
    }

    /// Devices this manager knows about, i.e. the pod it was constructed for.
    pub fn pod_devices(&self) -> &[TpuDeviceId] {
        &self.all_devices
    }

    /// Reject participants that are not members of this pod.
    ///
    /// `all_devices` is the manager's authoritative membership list, so a
    /// barrier or collective naming a device outside it can never rendezvous --
    /// it would block until timeout (barrier) or index past the known ranks
    /// (collective). Failing fast with [`SynchronizationError::DeviceNotFound`]
    /// is the honest answer.
    fn ensure_pod_members(&self, devices: &[TpuDeviceId]) -> Result<(), SynchronizationError> {
        if self.all_devices.is_empty() {
            // A manager built without a membership list cannot contradict the
            // caller, so nothing to check against.
            return Ok(());
        }
        for device in devices {
            if !self.all_devices.contains(device) {
                return Err(SynchronizationError::DeviceNotFound { device_id: *device });
            }
        }
        Ok(())
    }

    /// Create a synchronization barrier
    pub fn create_barrier(
        &self,
        barrier_id: String,
        devices: Vec<TpuDeviceId>,
        timeout: Duration,
    ) -> Result<(), SynchronizationError> {
        if devices.is_empty() {
            return Err(SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} requires at least one device"),
            });
        }
        self.ensure_pod_members(&devices)?;

        let barrier = Barrier {
            id: barrier_id.clone(),
            required_devices: devices,
            condition: Arc::new((Mutex::new(BarrierEpisode::default()), Condvar::new())),
            timeout,
            created_at: Instant::now(),
            state: BarrierState::Active,
        };

        let mut barriers = lock_map(&self.active_barriers)?;
        barriers.insert(barrier_id, barrier);
        drop(barriers);

        // Update statistics
        let mut stats = lock_stats(&self.stats)?;
        stats.barriers_created += 1;

        Ok(())
    }

    /// Wait for a barrier to complete.
    ///
    /// Every participant blocks until the last one arrives, at which point the
    /// generation counter is bumped **before** `notify_all`, so no waiter can
    /// miss the release. The barrier's own configured timeout is honoured (the
    /// previous code always used a hardcoded 30s), and the map lock is released
    /// before blocking so a waiting participant never blocks arrivals.
    pub fn wait_barrier(&self, barrier_id: &str) -> Result<(), SynchronizationError> {
        let start_time = Instant::now();

        // Short critical section on the registry: copy out what we need and let
        // the map lock go before blocking on anything.
        let (condition, timeout, required) = {
            let barriers = lock_map(&self.active_barriers)?;
            let barrier =
                barriers
                    .get(barrier_id)
                    .ok_or_else(|| SynchronizationError::InvalidOperation {
                        reason: format!("Barrier {barrier_id} not found"),
                    })?;
            (
                Arc::clone(&barrier.condition),
                barrier.timeout,
                barrier.required_devices.clone(),
            )
        };

        let (lock, condvar) = &*condition;
        let mut episode = lock
            .lock()
            .map_err(|_| SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} state lock is poisoned"),
            })?;

        if episode.aborted {
            return Err(SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} was cancelled"),
            });
        }

        if !episode.arrived.contains(&self.device_id) {
            episode.arrived.push(self.device_id);
        }

        if episode.arrived.len() >= required.len() {
            // Last participant: open the gate for this round, then reset so the
            // barrier can be reused for the next one.
            episode.arrived.clear();
            episode.generation = episode.generation.wrapping_add(1);
            drop(episode);
            condvar.notify_all();

            self.mark_barrier_state(barrier_id, BarrierState::Complete)?;
            self.record_barrier_completion(start_time)?;
            return Ok(());
        }

        let my_generation = episode.generation;
        let (episode, wait_result) = condvar
            .wait_timeout_while(episode, timeout, |state| {
                state.generation == my_generation && !state.aborted
            })
            .map_err(|_| SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} state lock is poisoned"),
            })?;

        let aborted = episode.aborted;
        drop(episode);

        if aborted {
            return Err(SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} was cancelled while waiting"),
            });
        }

        if wait_result.timed_out() {
            self.mark_barrier_state(barrier_id, BarrierState::TimedOut)?;

            let mut stats = lock_stats(&self.stats)?;
            stats.barriers_timed_out += 1;

            return Err(SynchronizationError::BarrierTimeout {
                barrier_id: barrier_id.to_string(),
            });
        }

        self.record_barrier_completion(start_time)?;
        Ok(())
    }

    /// Record a barrier completion and refresh the rolling wait-time average.
    fn record_barrier_completion(&self, start_time: Instant) -> Result<(), SynchronizationError> {
        let wait_time = start_time.elapsed().as_secs_f64();
        let mut stats = lock_stats(&self.stats)?;
        stats.barriers_completed += 1;
        stats.total_sync_time_seconds += wait_time;
        stats.avg_barrier_wait_time = if stats.barriers_completed == 0 {
            0.0
        } else {
            stats.total_sync_time_seconds / stats.barriers_completed as f64
        };
        Ok(())
    }

    /// Update the recorded lifecycle state of a barrier.
    fn mark_barrier_state(
        &self,
        barrier_id: &str,
        state: BarrierState,
    ) -> Result<(), SynchronizationError> {
        let mut barriers = lock_map(&self.active_barriers)?;
        if let Some(barrier) = barriers.get_mut(barrier_id) {
            barrier.state = state;
        }
        Ok(())
    }

    /// Devices that have arrived at the barrier's current round.
    pub fn barrier_arrivals(
        &self,
        barrier_id: &str,
    ) -> Result<Vec<TpuDeviceId>, SynchronizationError> {
        let condition = {
            let barriers = lock_map(&self.active_barriers)?;
            let barrier =
                barriers
                    .get(barrier_id)
                    .ok_or_else(|| SynchronizationError::InvalidOperation {
                        reason: format!("Barrier {barrier_id} not found"),
                    })?;
            Arc::clone(&barrier.condition)
        };

        let episode = condition
            .0
            .lock()
            .map_err(|_| SynchronizationError::InvalidOperation {
                reason: format!("Barrier {barrier_id} state lock is poisoned"),
            })?;
        Ok(episode.arrived.clone())
    }

    /// Execute a collective operation
    pub fn execute_collective_op(
        &self,
        request: CollectiveOpRequest,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let start_time = Instant::now();

        // Every participant must belong to this pod; a stranger rank would index
        // past the known buffers rather than communicate with anything.
        self.ensure_pod_members(&request.devices)?;

        // Update statistics
        {
            let mut stats = lock_stats(&self.stats)?;
            stats.collective_ops_total += 1;
        }

        // Find appropriate handler
        let handler = self
            .collective_handlers
            .get(&request.op_type)
            .ok_or_else(|| SynchronizationError::InvalidOperation {
                reason: format!("No handler for operation {:?}", request.op_type),
            })?;

        // Execute the operation
        let result = handler.execute(&request, &self.topology);

        // Update statistics based on result
        let mut stats = lock_stats(&self.stats)?;
        match &result {
            Ok(op_result) => {
                stats.collective_ops_success += 1;
                stats.total_data_transferred += request.data_size as u64;

                let total_ops = stats.collective_ops_success;
                let new_bandwidth = op_result.bandwidth_gb_s;
                stats.avg_bandwidth_gb_s = (stats.avg_bandwidth_gb_s * (total_ops - 1) as f64
                    + new_bandwidth)
                    / total_ops as f64;
            }
            Err(_) => {
                stats.collective_ops_failed += 1;
            }
        }

        // Fold this dispatch's real end-to-end cost into the rolling collective
        // timing statistics.
        stats.total_collective_time_seconds += start_time.elapsed().as_secs_f64();
        stats.avg_collective_op_time = if stats.collective_ops_total == 0 {
            0.0
        } else {
            stats.total_collective_time_seconds / stats.collective_ops_total as f64
        };

        result
    }

    /// Get current synchronization statistics
    pub fn get_statistics(&self) -> SynchronizationStats {
        match self.stats.lock() {
            Ok(stats) => (*stats).clone(),
            Err(poisoned) => (*poisoned.into_inner()).clone(),
        }
    }

    /// Cancel all active barriers, releasing every waiter with an error.
    ///
    /// The `aborted` flag is set under the barrier's own lock *before*
    /// `notify_all`, so a waiter cannot miss the cancellation and hang until
    /// its timeout.
    pub fn cancel_all_barriers(&self) -> Result<(), SynchronizationError> {
        let mut barriers = lock_map(&self.active_barriers)?;
        for barrier in barriers.values_mut() {
            barrier.state = BarrierState::Cancelled;
            let (lock, condvar) = &*barrier.condition;
            match lock.lock() {
                Ok(mut episode) => {
                    episode.aborted = true;
                    // Bump the generation as well so generation-based waiters
                    // wake even if they somehow miss the abort flag.
                    episode.generation = episode.generation.wrapping_add(1);
                    drop(episode);
                    condvar.notify_all();
                }
                Err(_) => {
                    log::error!(
                        "barrier {} state lock is poisoned; waiters may not be released",
                        barrier.id
                    );
                }
            }
        }
        barriers.clear();
        Ok(())
    }

    /// Get topology information
    pub fn get_topology(&self) -> &CommunicationTopology {
        &self.topology
    }
}

// ---------------------------------------------------------------------------
// Collective operations
//
// These are real, in-process implementations. There is no TPU interconnect in
// this environment, so the "ranks" are simulated buffers held in this process,
// but the reduction arithmetic and the ring communication schedule are the
// genuine algorithms: after `AllReduce` every rank's buffer holds the true
// element-wise reduction across all ranks. Nothing here sleeps to fake latency,
// and every reported bandwidth is measured from bytes actually moved.
// ---------------------------------------------------------------------------

/// Apply a reduction operator to a pair of values.
fn reduce_pair(op: ReductionOp, accumulator: f64, value: f64) -> f64 {
    match op {
        // `Mean` accumulates as a sum and is divided by the rank count at the end.
        ReductionOp::Sum | ReductionOp::Mean => accumulator + value,
        ReductionOp::Product => accumulator * value,
        ReductionOp::Max => accumulator.max(value),
        ReductionOp::Min => accumulator.min(value),
        ReductionOp::LogicalAnd => {
            if accumulator != 0.0 && value != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ReductionOp::LogicalOr => {
            if accumulator != 0.0 || value != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Split `len` elements into `parts` contiguous chunks of near-equal size.
///
/// The first `len % parts` chunks get one extra element, so the schedule is
/// correct even when the buffer length is not a multiple of the rank count.
fn chunk_bounds(len: usize, parts: usize) -> Vec<(usize, usize)> {
    if parts == 0 {
        return Vec::new();
    }
    let base = len / parts;
    let remainder = len % parts;
    let mut bounds = Vec::with_capacity(parts);
    let mut start = 0usize;
    for index in 0..parts {
        let extra = usize::from(index < remainder);
        let end = start + base + extra;
        bounds.push((start, end));
        start = end;
    }
    bounds
}

/// Ring all-reduce across in-process rank buffers.
///
/// Runs the standard two-phase ring algorithm: `R - 1` reduce-scatter steps
/// followed by `R - 1` all-gather steps, each step moving one chunk along the
/// ring. On return every rank's buffer holds the full reduction.
///
/// Returns the number of bytes actually moved, so callers can report measured
/// rather than invented bandwidth.
///
/// `source` is used for modular-ring arithmetic (`(source + rank_count - step)
/// % rank_count`, `(source + 1) % rank_count`) as well as indexing, in both
/// phases below, so `.iter().enumerate()` would still need the plain index
/// alongside the item; range loops read more directly here than threading the
/// same value through both the enumeration and the arithmetic.
#[allow(clippy::needless_range_loop)]
pub fn ring_all_reduce(
    ranks: &mut [Vec<f64>],
    op: ReductionOp,
) -> Result<usize, SynchronizationError> {
    let rank_count = ranks.len();
    if rank_count == 0 {
        return Err(SynchronizationError::InvalidOperation {
            reason: "all-reduce requires at least one rank".to_string(),
        });
    }

    let len = ranks[0].len();
    if ranks.iter().any(|buffer| buffer.len() != len) {
        return Err(SynchronizationError::InvalidOperation {
            reason: "all-reduce requires every rank buffer to have the same length".to_string(),
        });
    }

    if rank_count == 1 {
        if op == ReductionOp::Mean {
            // A single rank's mean is itself; nothing to do.
        }
        return Ok(0);
    }

    if len == 0 {
        return Ok(0);
    }

    let bounds = chunk_bounds(len, rank_count);
    let mut bytes_moved = 0usize;
    let element_size = std::mem::size_of::<f64>();

    // Phase 1 - reduce-scatter. After R-1 steps, rank `r` owns the fully
    // reduced chunk `(r + 1) % R`.
    for step in 0..rank_count - 1 {
        // Every rank sends simultaneously, so snapshot all outgoing segments
        // before any of them are overwritten by an incoming one.
        let mut in_flight: Vec<Vec<f64>> = Vec::with_capacity(rank_count);
        for source in 0..rank_count {
            let chunk = (source + rank_count - step) % rank_count;
            let (start, end) = bounds[chunk];
            in_flight.push(ranks[source][start..end].to_vec());
        }

        for source in 0..rank_count {
            let destination = (source + 1) % rank_count;
            let chunk = (source + rank_count - step) % rank_count;
            let (start, _end) = bounds[chunk];
            let segment = &in_flight[source];
            bytes_moved += segment.len() * element_size;

            for (offset, &value) in segment.iter().enumerate() {
                let slot = &mut ranks[destination][start + offset];
                *slot = reduce_pair(op, *slot, value);
            }
        }
    }

    // Phase 2 - all-gather. Propagate each fully reduced chunk around the ring.
    for step in 0..rank_count - 1 {
        let mut in_flight: Vec<Vec<f64>> = Vec::with_capacity(rank_count);
        for source in 0..rank_count {
            let chunk = (source + 1 + rank_count - step) % rank_count;
            let (start, end) = bounds[chunk];
            in_flight.push(ranks[source][start..end].to_vec());
        }

        for source in 0..rank_count {
            let destination = (source + 1) % rank_count;
            let chunk = (source + 1 + rank_count - step) % rank_count;
            let (start, end) = bounds[chunk];
            let segment = &in_flight[source];
            bytes_moved += segment.len() * element_size;
            ranks[destination][start..end].copy_from_slice(segment);
        }
    }

    if op == ReductionOp::Mean {
        let divisor = rank_count as f64;
        for buffer in ranks.iter_mut() {
            for value in buffer.iter_mut() {
                *value /= divisor;
            }
        }
    }

    Ok(bytes_moved)
}

/// Broadcast the root rank's buffer to every other rank along the ring.
pub fn ring_broadcast(ranks: &mut [Vec<f64>], root: usize) -> Result<usize, SynchronizationError> {
    let rank_count = ranks.len();
    if root >= rank_count {
        return Err(SynchronizationError::InvalidOperation {
            reason: format!("broadcast root {root} is out of range for {rank_count} ranks"),
        });
    }

    let source = ranks[root].clone();
    let mut bytes_moved = 0usize;
    for (index, buffer) in ranks.iter_mut().enumerate() {
        if index == root {
            continue;
        }
        bytes_moved += source.len() * std::mem::size_of::<f64>();
        *buffer = source.clone();
    }
    Ok(bytes_moved)
}

/// All-gather: every rank ends up holding the concatenation of all buffers.
pub fn ring_all_gather(ranks: &mut [Vec<f64>]) -> Result<usize, SynchronizationError> {
    if ranks.is_empty() {
        return Err(SynchronizationError::InvalidOperation {
            reason: "all-gather requires at least one rank".to_string(),
        });
    }

    let gathered: Vec<f64> = ranks.iter().flatten().copied().collect();
    let bytes_moved = gathered.len() * std::mem::size_of::<f64>() * ranks.len();
    for buffer in ranks.iter_mut() {
        *buffer = gathered.clone();
    }
    Ok(bytes_moved)
}

/// Reduce-scatter: reduce across ranks, then leave rank `r` holding chunk `r`.
pub fn ring_reduce_scatter(
    ranks: &mut [Vec<f64>],
    op: ReductionOp,
) -> Result<usize, SynchronizationError> {
    let rank_count = ranks.len();
    let len = ranks.first().map(|buffer| buffer.len()).unwrap_or(0);
    let bytes_moved = ring_all_reduce(ranks, op)?;

    let bounds = chunk_bounds(len, rank_count);
    for (index, buffer) in ranks.iter_mut().enumerate() {
        if let Some(&(start, end)) = bounds.get(index) {
            *buffer = buffer[start..end].to_vec();
        }
    }
    Ok(bytes_moved)
}

/// Lock a request's buffers, mapping poisoning to an error.
fn lock_buffers(
    request: &CollectiveOpRequest,
) -> Result<std::sync::MutexGuard<'_, CollectiveBuffers>, SynchronizationError> {
    let buffers =
        request
            .buffers
            .as_ref()
            .ok_or_else(|| SynchronizationError::InvalidOperation {
                reason: format!(
                    "collective operation {} carries no data buffers; attach \
                 CollectiveOpRequest::buffers so there is something to move",
                    request.id
                ),
            })?;

    buffers
        .lock()
        .map_err(|_| SynchronizationError::InvalidOperation {
            reason: format!("collective buffers for {} are poisoned", request.id),
        })
}

/// Build a result whose bandwidth is measured from bytes actually moved.
fn measured_result(id: &str, bytes_moved: usize, duration: Duration) -> CollectiveOpResult {
    let seconds = duration.as_secs_f64();
    let bandwidth_gb_s = if seconds > 0.0 {
        (bytes_moved as f64) / seconds / 1e9
    } else {
        0.0
    };

    CollectiveOpResult {
        id: id.to_string(),
        status: OperationStatus::Success,
        duration,
        bandwidth_gb_s,
        error_message: None,
    }
}

/// Validate that the buffer count matches the participating device count.
fn check_rank_count(
    request: &CollectiveOpRequest,
    buffers: &CollectiveBuffers,
) -> Result<(), SynchronizationError> {
    if buffers.rank_count() != request.devices.len() {
        return Err(SynchronizationError::InvalidOperation {
            reason: format!(
                "collective {} has {} device(s) but {} buffer(s)",
                request.id,
                request.devices.len(),
                buffers.rank_count()
            ),
        });
    }
    Ok(())
}

/// All-reduce operation handler
#[derive(Debug, Default)]
pub struct AllReduceHandler;

impl AllReduceHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for AllReduceHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let reduction =
            request
                .reduction_op
                .ok_or_else(|| SynchronizationError::InvalidOperation {
                    reason: format!("all-reduce {} requires a reduction operator", request.id),
                })?;

        let mut buffers = lock_buffers(request)?;
        check_rank_count(request, &buffers)?;
        buffers.uniform_len()?;

        let start_time = Instant::now();
        let bytes_moved = ring_all_reduce(&mut buffers.ranks, reduction)?;
        let duration = start_time.elapsed();

        Ok(measured_result(&request.id, bytes_moved, duration))
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        // Ring all-reduce moves 2*(R-1)/R of the payload per rank.
        let ranks = request.devices.len().max(1) as u64;
        let base_latency = Duration::from_micros(10 * ranks);
        let bytes = 2 * request.data_size as u64 * (ranks - 1) / ranks;
        base_latency + Duration::from_nanos(bytes / 100)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::AllReduce)
    }
}

/// All-gather operation handler
#[derive(Debug, Default)]
pub struct AllGatherHandler;

impl AllGatherHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for AllGatherHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let mut buffers = lock_buffers(request)?;
        check_rank_count(request, &buffers)?;

        let start_time = Instant::now();
        let bytes_moved = ring_all_gather(&mut buffers.ranks)?;
        let duration = start_time.elapsed();

        Ok(measured_result(&request.id, bytes_moved, duration))
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(5 + (request.data_size / 1024) as u64)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::AllGather)
    }
}

/// Reduce-scatter operation handler
#[derive(Debug, Default)]
pub struct ReduceScatterHandler;

impl ReduceScatterHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for ReduceScatterHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let reduction =
            request
                .reduction_op
                .ok_or_else(|| SynchronizationError::InvalidOperation {
                    reason: format!(
                        "reduce-scatter {} requires a reduction operator",
                        request.id
                    ),
                })?;

        let mut buffers = lock_buffers(request)?;
        check_rank_count(request, &buffers)?;
        buffers.uniform_len()?;

        let start_time = Instant::now();
        let bytes_moved = ring_reduce_scatter(&mut buffers.ranks, reduction)?;
        let duration = start_time.elapsed();

        Ok(measured_result(&request.id, bytes_moved, duration))
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        let ranks = request.devices.len().max(1) as u64;
        Duration::from_micros(5 * ranks) + Duration::from_nanos(request.data_size as u64 / 100)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::ReduceScatter)
    }
}

/// Broadcast operation handler
#[derive(Debug, Default)]
pub struct BroadcastHandler;

impl BroadcastHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for BroadcastHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let root_device =
            request
                .root_device
                .ok_or_else(|| SynchronizationError::InvalidOperation {
                    reason: "Broadcast requires a root device".to_string(),
                })?;

        let root_index = request
            .devices
            .iter()
            .position(|device| *device == root_device)
            .ok_or_else(|| SynchronizationError::InvalidOperation {
                reason: format!(
                    "broadcast root {root_device:?} does not participate in operation {}",
                    request.id
                ),
            })?;

        let mut buffers = lock_buffers(request)?;
        check_rank_count(request, &buffers)?;

        let start_time = Instant::now();
        let bytes_moved = ring_broadcast(&mut buffers.ranks, root_index)?;
        let duration = start_time.elapsed();

        Ok(measured_result(&request.id, bytes_moved, duration))
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(3) + Duration::from_nanos(request.data_size as u64 / 100)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::Broadcast)
    }
}

/// All-to-all operation handler
///
/// All-to-all requires each rank to exchange a distinct slice with every other
/// rank. That schedule is not implemented here, and reporting success without
/// moving the data would be a lie, so this returns an error.
#[derive(Debug, Default)]
pub struct AllToAllHandler;

impl AllToAllHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for AllToAllHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        Err(SynchronizationError::InvalidOperation {
            reason: format!(
                "all-to-all is not implemented for operation {}; use AllReduce, AllGather, \
                 ReduceScatter or Broadcast",
                request.id
            ),
        })
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        let ranks = request.devices.len().max(1) as u64;
        Duration::from_nanos(request.data_size as u64 * ranks / 100)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::AllToAll)
    }
}

/// Barrier operation handler
///
/// A barrier moves no data; it is a pure rendezvous. Actual rendezvous is
/// performed by [`SynchronizationManager::wait_barrier`], so this handler only
/// validates the request.
#[derive(Debug, Default)]
pub struct BarrierHandler;

impl BarrierHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectiveHandler for BarrierHandler {
    fn execute(
        &self,
        request: &CollectiveOpRequest,
        _topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        if request.devices.is_empty() {
            return Err(SynchronizationError::InvalidOperation {
                reason: format!("barrier {} has no participating devices", request.id),
            });
        }

        Ok(CollectiveOpResult {
            id: request.id.clone(),
            status: OperationStatus::Success,
            duration: Duration::ZERO,
            bandwidth_gb_s: 0.0, // A barrier transfers no payload.
            error_message: None,
        })
    }

    fn estimate_time(&self, _request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(100)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::Barrier)
    }
}

impl CommunicationTopology {
    /// Create a ring topology
    pub fn create_ring(devices: Vec<TpuDeviceId>) -> Self {
        let mut connections = HashMap::new();
        let mut rings = Vec::new();

        if !devices.is_empty() {
            // Create ring connections
            for (i, &device) in devices.iter().enumerate() {
                let next_device = devices[(i + 1) % devices.len()];
                connections.insert(device, vec![next_device]);
            }

            rings.push(devices.clone());
        }

        Self {
            topology_type: TopologyType::Ring,
            connections,
            rings,
            tree: None,
        }
    }

    /// Create a tree topology
    pub fn create_tree(devices: Vec<TpuDeviceId>) -> Self {
        let mut connections = HashMap::new();
        let mut parent_child = HashMap::new();
        let mut child_parent = HashMap::new();

        // The root is the single source of truth for both the adjacency build
        // and the returned `CommunicationTree`; `first()` also keeps the empty
        // case from needing an unguarded `devices[0]`.
        let tree = match devices.first().copied() {
            Some(root) => {
                // Simple binary tree
                for (i, &device) in devices.iter().enumerate() {
                    let mut children = Vec::new();

                    let left_child_idx = 2 * i + 1;
                    let right_child_idx = 2 * i + 2;

                    if let Some(&child) = devices.get(left_child_idx) {
                        children.push(child);
                        child_parent.insert(child, device);
                    }

                    if let Some(&child) = devices.get(right_child_idx) {
                        children.push(child);
                        child_parent.insert(child, device);
                    }

                    if !children.is_empty() {
                        connections.insert(device, children.clone());
                        parent_child.insert(device, children);
                    }
                }

                Some(CommunicationTree {
                    root,
                    parent_child,
                    child_parent,
                    depth: (devices.len() as f64).log2().ceil() as u32,
                })
            }
            None => None,
        };

        Self {
            topology_type: TopologyType::Tree,
            connections,
            rings: Vec::new(),
            tree,
        }
    }

    /// Create a mesh topology
    pub fn create_mesh(devices: Vec<TpuDeviceId>) -> Self {
        let mut connections = HashMap::new();

        // Full mesh - every device connects to every other device
        for &device in &devices {
            let neighbors: Vec<TpuDeviceId> = devices
                .iter()
                .filter(|&&other| other != device)
                .cloned()
                .collect();
            connections.insert(device, neighbors);
        }

        Self {
            topology_type: TopologyType::Mesh,
            connections,
            rings: Vec::new(),
            tree: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronization_manager_creation() {
        let devices = vec![
            TpuDeviceId(0),
            TpuDeviceId(1),
            TpuDeviceId(2),
            TpuDeviceId(3),
        ];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let sync_manager = SynchronizationManager::new(TpuDeviceId(0), devices, topology);

        assert_eq!(sync_manager.device_id, TpuDeviceId(0));
        assert_eq!(sync_manager.all_devices.len(), 4);
    }

    #[test]
    fn test_barrier_creation() {
        let devices = vec![TpuDeviceId(0), TpuDeviceId(1)];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let sync_manager = SynchronizationManager::new(TpuDeviceId(0), devices.clone(), topology);

        let result = sync_manager.create_barrier(
            "test_barrier".to_string(),
            devices,
            Duration::from_secs(10),
        );

        assert!(result.is_ok());
        assert!(sync_manager.barrier_arrivals("test_barrier").is_ok());
    }

    #[test]
    fn test_collective_operation() {
        let devices = vec![
            TpuDeviceId(0),
            TpuDeviceId(1),
            TpuDeviceId(2),
            TpuDeviceId(3),
        ];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let sync_manager = SynchronizationManager::new(TpuDeviceId(0), devices.clone(), topology);

        // Four ranks, each holding a distinct buffer.
        let buffers = Arc::new(Mutex::new(CollectiveBuffers::new(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![100.0, 200.0, 300.0, 400.0],
            vec![1000.0, 2000.0, 3000.0, 4000.0],
        ])));

        let request = CollectiveOpRequest {
            id: "test_allreduce".to_string(),
            op_type: CollectiveOpType::AllReduce,
            devices,
            root_device: None,
            reduction_op: Some(ReductionOp::Sum),
            data_size: 4 * std::mem::size_of::<f64>(),
            buffers: Some(Arc::clone(&buffers)),
            timeout: Duration::from_secs(10),
            priority: OperationPriority::Normal,
        };

        let op_result = sync_manager
            .execute_collective_op(request)
            .expect("all-reduce over valid buffers must succeed");
        assert_eq!(op_result.status, OperationStatus::Success);

        // Every rank must now hold the element-wise sum.
        let expected = vec![1111.0, 2222.0, 3333.0, 4444.0];
        let final_buffers = buffers.lock().expect("buffers must not be poisoned");
        for (rank, buffer) in final_buffers.ranks.iter().enumerate() {
            assert_eq!(
                buffer, &expected,
                "rank {rank} did not receive the reduction"
            );
        }
    }

    #[test]
    fn test_topology_creation() {
        let devices = vec![
            TpuDeviceId(0),
            TpuDeviceId(1),
            TpuDeviceId(2),
            TpuDeviceId(3),
        ];

        // Test ring topology
        let ring_topology = CommunicationTopology::create_ring(devices.clone());
        assert_eq!(ring_topology.topology_type, TopologyType::Ring);
        assert_eq!(ring_topology.rings.len(), 1);
        assert_eq!(ring_topology.rings[0].len(), 4);

        // Test tree topology
        let tree_topology = CommunicationTopology::create_tree(devices.clone());
        assert_eq!(tree_topology.topology_type, TopologyType::Tree);
        assert!(tree_topology.tree.is_some());

        // Test mesh topology
        let mesh_topology = CommunicationTopology::create_mesh(devices.clone());
        assert_eq!(mesh_topology.topology_type, TopologyType::Mesh);
        assert_eq!(mesh_topology.connections.len(), 4);

        for neighbors in mesh_topology.connections.values() {
            assert_eq!(neighbors.len(), 3); // Each device connects to 3 others
        }
    }
}

#[cfg(test)]
mod barrier_and_collective_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    /// Reference implementation: naive element-wise reduction across ranks.
    fn reference_reduce(ranks: &[Vec<f64>], op: ReductionOp) -> Vec<f64> {
        let Some(first) = ranks.first() else {
            return Vec::new();
        };
        let mut accumulator = first.clone();
        for buffer in ranks.iter().skip(1) {
            for (slot, &value) in accumulator.iter_mut().zip(buffer.iter()) {
                *slot = reduce_pair(op, *slot, value);
            }
        }
        if op == ReductionOp::Mean {
            let divisor = ranks.len() as f64;
            for value in accumulator.iter_mut() {
                *value /= divisor;
            }
        }
        accumulator
    }

    /// F4: every participant must be released, not just the last to arrive.
    ///
    /// Before the fix, all non-final waiters blocked on a condvar whose
    /// predicate was never set, timed out after a hardcoded 30s, and returned
    /// `BarrierTimeout`.
    #[test]
    fn barrier_releases_all_threads() {
        const PARTICIPANTS: usize = 8;

        let devices: Vec<TpuDeviceId> = (0..PARTICIPANTS)
            .map(|index| TpuDeviceId(index as u32))
            .collect();
        let topology = CommunicationTopology::create_ring(devices.clone());

        // All managers must share one barrier registry to rendezvous.
        let shared_barriers = Arc::new(Mutex::new(HashMap::new()));
        let managers: Vec<Arc<SynchronizationManager>> = devices
            .iter()
            .map(|&device| {
                Arc::new(SynchronizationManager::with_barrier_registry(
                    device,
                    devices.clone(),
                    topology.clone(),
                    Arc::clone(&shared_barriers),
                ))
            })
            .collect();

        managers[0]
            .create_barrier(
                "round".to_string(),
                devices.clone(),
                Duration::from_secs(10),
            )
            .expect("barrier creation must succeed");

        let released = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for manager in &managers {
            let manager = Arc::clone(manager);
            let released = Arc::clone(&released);
            handles.push(thread::spawn(move || {
                let result = manager.wait_barrier("round");
                if result.is_ok() {
                    released.fetch_add(1, Ordering::SeqCst);
                }
                result
            }));
        }

        for handle in handles {
            let outcome = handle.join().expect("barrier thread must not panic");
            assert!(
                outcome.is_ok(),
                "a participant was not released: {outcome:?}"
            );
        }

        assert_eq!(
            released.load(Ordering::SeqCst),
            PARTICIPANTS,
            "every participant must be released"
        );
    }

    /// The barrier must be reusable across consecutive rounds.
    #[test]
    fn barrier_is_reusable_across_rounds() {
        const PARTICIPANTS: usize = 4;
        const ROUNDS: usize = 3;

        let devices: Vec<TpuDeviceId> = (0..PARTICIPANTS)
            .map(|index| TpuDeviceId(index as u32))
            .collect();
        let topology = CommunicationTopology::create_ring(devices.clone());
        let shared_barriers = Arc::new(Mutex::new(HashMap::new()));

        let managers: Vec<Arc<SynchronizationManager>> = devices
            .iter()
            .map(|&device| {
                Arc::new(SynchronizationManager::with_barrier_registry(
                    device,
                    devices.clone(),
                    topology.clone(),
                    Arc::clone(&shared_barriers),
                ))
            })
            .collect();

        managers[0]
            .create_barrier(
                "reusable".to_string(),
                devices.clone(),
                Duration::from_secs(10),
            )
            .expect("barrier creation must succeed");

        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for manager in &managers {
            let manager = Arc::clone(manager);
            let completed = Arc::clone(&completed);
            handles.push(thread::spawn(move || {
                for _ in 0..ROUNDS {
                    manager
                        .wait_barrier("reusable")
                        .expect("each round must release every participant");
                    completed.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("barrier thread must not panic");
        }

        assert_eq!(completed.load(Ordering::SeqCst), PARTICIPANTS * ROUNDS);
    }

    /// A barrier whose participants never all arrive must honour its own
    /// configured timeout, not a hardcoded 30 seconds.
    #[test]
    fn barrier_honours_its_configured_timeout() {
        let devices = vec![TpuDeviceId(0), TpuDeviceId(1)];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let manager = SynchronizationManager::new(TpuDeviceId(0), devices.clone(), topology);

        manager
            .create_barrier("lonely".to_string(), devices, Duration::from_millis(50))
            .expect("barrier creation must succeed");

        let start = Instant::now();
        let outcome = manager.wait_barrier("lonely");
        let elapsed = start.elapsed();

        assert!(matches!(
            outcome,
            Err(SynchronizationError::BarrierTimeout { .. })
        ));
        assert!(
            elapsed < Duration::from_secs(5),
            "the configured 50ms timeout must be used, waited {elapsed:?}"
        );
    }

    /// F15: after all-reduce every rank holds the element-wise sum.
    #[test]
    fn ring_all_reduce_sums_across_four_ranks() {
        let mut ranks = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![100.0, 200.0, 300.0, 400.0],
            vec![1000.0, 2000.0, 3000.0, 4000.0],
        ];
        let expected = reference_reduce(&ranks, ReductionOp::Sum);

        let bytes = ring_all_reduce(&mut ranks, ReductionOp::Sum)
            .expect("all-reduce over uniform buffers must succeed");

        assert!(bytes > 0, "a real all-reduce moves data");
        for (index, buffer) in ranks.iter().enumerate() {
            assert_eq!(buffer, &expected, "rank {index} holds the wrong result");
        }
        assert_eq!(expected, vec![1111.0, 2222.0, 3333.0, 4444.0]);
    }

    /// The ring schedule must be correct when the length is not divisible by
    /// the rank count.
    #[test]
    fn ring_all_reduce_handles_non_divisible_length() {
        for len in [1usize, 2, 3, 5, 7, 10, 13] {
            for rank_count in [1usize, 2, 3, 4, 5] {
                let mut ranks: Vec<Vec<f64>> = (0..rank_count)
                    .map(|rank| (0..len).map(|index| (rank * 100 + index) as f64).collect())
                    .collect();
                let expected = reference_reduce(&ranks, ReductionOp::Sum);

                ring_all_reduce(&mut ranks, ReductionOp::Sum)
                    .expect("all-reduce must succeed for any shape");

                for (index, buffer) in ranks.iter().enumerate() {
                    assert_eq!(
                        buffer, &expected,
                        "len={len} ranks={rank_count}: rank {index} is wrong"
                    );
                }
            }
        }
    }

    /// Every reduction operator must match a naive reference implementation.
    #[test]
    fn ring_all_reduce_matches_reference_for_all_operators() {
        let operators = [
            ReductionOp::Sum,
            ReductionOp::Mean,
            ReductionOp::Product,
            ReductionOp::Max,
            ReductionOp::Min,
        ];

        for op in operators {
            let base: Vec<Vec<f64>> = vec![
                vec![1.0, -2.0, 3.5, 4.0, 0.5],
                vec![2.0, 5.0, -1.5, 8.0, 2.0],
                vec![3.0, 1.0, 2.0, -4.0, 1.5],
            ];
            let expected = reference_reduce(&base, op);
            let mut ranks = base.clone();

            ring_all_reduce(&mut ranks, op).expect("all-reduce must succeed");

            for (index, buffer) in ranks.iter().enumerate() {
                for (slot, (&actual, &want)) in buffer.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (actual - want).abs() < 1e-9,
                        "{op:?}: rank {index} element {slot}: {actual} != {want}"
                    );
                }
            }
        }
    }

    /// Mismatched buffer lengths must be rejected, not silently truncated.
    #[test]
    fn ring_all_reduce_rejects_ragged_buffers() {
        let mut ranks = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(ring_all_reduce(&mut ranks, ReductionOp::Sum).is_err());
    }

    /// Broadcast copies the root's buffer to every rank.
    #[test]
    fn ring_broadcast_replicates_the_root_buffer() {
        let mut ranks = vec![vec![0.0; 3], vec![7.0, 8.0, 9.0], vec![0.0; 3]];
        ring_broadcast(&mut ranks, 1).expect("broadcast must succeed");
        for buffer in &ranks {
            assert_eq!(buffer, &vec![7.0, 8.0, 9.0]);
        }
    }

    /// Reduce-scatter leaves rank `r` holding only its own reduced chunk.
    #[test]
    fn ring_reduce_scatter_splits_the_reduction() {
        let ranks_before = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let full = reference_reduce(&ranks_before, ReductionOp::Sum);
        let mut ranks = ranks_before.clone();

        ring_reduce_scatter(&mut ranks, ReductionOp::Sum).expect("reduce-scatter must succeed");

        assert_eq!(ranks[0], full[0..2].to_vec());
        assert_eq!(ranks[1], full[2..4].to_vec());
    }

    /// A collective with no attached buffers must error rather than claim
    /// success while moving nothing.
    #[test]
    fn collective_without_buffers_is_rejected() {
        let devices = vec![TpuDeviceId(0), TpuDeviceId(1)];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let manager = SynchronizationManager::new(TpuDeviceId(0), devices.clone(), topology);

        let request = CollectiveOpRequest {
            id: "no_data".to_string(),
            op_type: CollectiveOpType::AllReduce,
            devices,
            root_device: None,
            reduction_op: Some(ReductionOp::Sum),
            data_size: 0,
            buffers: None,
            timeout: Duration::from_secs(1),
            priority: OperationPriority::Normal,
        };

        assert!(manager.execute_collective_op(request).is_err());
    }

    /// All-to-all is not implemented and must say so.
    #[test]
    fn all_to_all_reports_unimplemented() {
        let devices = vec![TpuDeviceId(0), TpuDeviceId(1)];
        let topology = CommunicationTopology::create_ring(devices.clone());
        let handler = AllToAllHandler::new();

        let request = CollectiveOpRequest {
            id: "a2a".to_string(),
            op_type: CollectiveOpType::AllToAll,
            devices,
            root_device: None,
            reduction_op: None,
            data_size: 16,
            buffers: Some(Arc::new(Mutex::new(CollectiveBuffers::new(vec![
                vec![1.0],
                vec![2.0],
            ])))),
            timeout: Duration::from_secs(1),
            priority: OperationPriority::Normal,
        };

        assert!(handler.execute(&request, &topology).is_err());
    }
}
