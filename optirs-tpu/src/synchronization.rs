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

use crate::coordination::{PodCoordinator, TpuDeviceId};

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

    /// Devices that have reached the barrier
    pub arrived_devices: Vec<TpuDeviceId>,

    /// Condition variable for waiting
    pub condition: Arc<(Mutex<bool>, Condvar)>,

    /// Timeout for the barrier
    pub timeout: Duration,

    /// Creation timestamp
    pub created_at: Instant,

    /// Barrier state
    pub state: BarrierState,
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
    pub data_size: usize,

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
pub trait CollectiveHandler: Send + Sync {
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
#[derive(Debug, Default)]
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

impl SynchronizationManager {
    /// Create a new synchronization manager
    pub fn new(
        device_id: TpuDeviceId,
        all_devices: Vec<TpuDeviceId>,
        topology: CommunicationTopology,
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
        collective_handlers.insert(CollectiveOpType::Barrier, Box::new(BarrierHandler::new()));

        Self {
            device_id,
            all_devices,
            active_barriers: Arc::new(Mutex::new(HashMap::new())),
            collective_handlers,
            topology,
            stats: Arc::new(Mutex::new(SynchronizationStats::default())),
        }
    }

    /// Create a synchronization barrier
    pub fn create_barrier(
        &self,
        barrier_id: String,
        devices: Vec<TpuDeviceId>,
        timeout: Duration,
    ) -> Result<(), SynchronizationError> {
        let barrier = Barrier {
            id: barrier_id.clone(),
            required_devices: devices,
            arrived_devices: Vec::new(),
            condition: Arc::new((Mutex::new(false), Condvar::new())),
            timeout,
            created_at: Instant::now(),
            state: BarrierState::Active,
        };

        let mut barriers = self.active_barriers.lock().unwrap();
        barriers.insert(barrier_id, barrier);

        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.barriers_created += 1;

        Ok(())
    }

    /// Wait for a barrier to complete
    pub fn wait_barrier(&self, barrier_id: &str) -> Result<(), SynchronizationError> {
        let start_time = Instant::now();

        // Signal arrival at barrier
        {
            let mut barriers = self.active_barriers.lock().unwrap();
            if let Some(barrier) = barriers.get_mut(barrier_id) {
                if !barrier.arrived_devices.contains(&self.device_id) {
                    barrier.arrived_devices.push(self.device_id);
                }

                // Check if all devices have arrived
                if barrier.arrived_devices.len() == barrier.required_devices.len() {
                    barrier.state = BarrierState::Complete;
                    let (_, condvar) = &*barrier.condition;
                    condvar.notify_all();

                    // Update statistics
                    let mut stats = self.stats.lock().unwrap();
                    stats.barriers_completed += 1;

                    return Ok(());
                }
            } else {
                return Err(SynchronizationError::InvalidOperation {
                    reason: format!("Barrier {} not found", barrier_id),
                });
            }
        }

        // Wait for barrier completion or timeout
        let barrier_condition = {
            let barriers = self.active_barriers.lock().unwrap();
            barriers.get(barrier_id).unwrap().condition.clone()
        };

        let (lock, condvar) = &*barrier_condition;
        let mut completed = lock.lock().unwrap();

        let timeout_result = condvar
            .wait_timeout_while(
                completed,
                Duration::from_secs(30), // Default timeout
                |&mut completed| !completed,
            )
            .unwrap();

        if timeout_result.1.timed_out() {
            // Mark barrier as timed out
            let mut barriers = self.active_barriers.lock().unwrap();
            if let Some(barrier) = barriers.get_mut(barrier_id) {
                barrier.state = BarrierState::TimedOut;
            }

            let mut stats = self.stats.lock().unwrap();
            stats.barriers_timed_out += 1;

            return Err(SynchronizationError::BarrierTimeout {
                barrier_id: barrier_id.to_string(),
            });
        }

        // Update statistics
        let wait_time = start_time.elapsed().as_secs_f64();
        let mut stats = self.stats.lock().unwrap();
        stats.total_sync_time_seconds += wait_time;
        stats.avg_barrier_wait_time =
            stats.total_sync_time_seconds / stats.barriers_completed as f64;

        Ok(())
    }

    /// Execute a collective operation
    pub fn execute_collective_op(
        &self,
        request: CollectiveOpRequest,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let start_time = Instant::now();

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
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
        let mut stats = self.stats.lock().unwrap();
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

        result
    }

    /// Get current synchronization statistics
    pub fn get_statistics(&self) -> SynchronizationStats {
        self.stats.lock().unwrap().clone()
    }

    /// Cancel all active barriers
    pub fn cancel_all_barriers(&self) {
        let mut barriers = self.active_barriers.lock().unwrap();
        for (_, barrier) in barriers.iter_mut() {
            barrier.state = BarrierState::Cancelled;
            let (_, condvar) = &*barrier.condition;
            condvar.notify_all();
        }
        barriers.clear();
    }

    /// Get topology information
    pub fn get_topology(&self) -> &CommunicationTopology {
        &self.topology
    }
}

// Collective operation handlers

/// All-reduce operation handler
#[derive(Debug)]
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
        topology: &CommunicationTopology,
    ) -> Result<CollectiveOpResult, SynchronizationError> {
        let start_time = Instant::now();

        // Simulate all-reduce operation
        // In a real implementation, this would coordinate the actual data reduction
        std::thread::sleep(Duration::from_millis(10));

        let duration = start_time.elapsed();
        let bandwidth_gb_s =
            (request.data_size as f64) / duration.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

        Ok(CollectiveOpResult {
            id: request.id.clone(),
            status: OperationStatus::Success,
            duration,
            bandwidth_gb_s,
            error_message: None,
        })
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        // Estimate based on data size and number of devices
        let base_latency = Duration::from_micros(10);
        let transfer_time = Duration::from_nanos(
            (request.data_size as u64 * request.devices.len() as u64) / 100, // Assume 100 GB/s
        );
        base_latency + transfer_time
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::AllReduce)
    }
}

/// All-gather operation handler
#[derive(Debug)]
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
        let start_time = Instant::now();

        // Simulate all-gather operation
        std::thread::sleep(Duration::from_millis(5));

        let duration = start_time.elapsed();
        let bandwidth_gb_s = (request.data_size as f64 * request.devices.len() as f64)
            / duration.as_secs_f64()
            / (1024.0 * 1024.0 * 1024.0);

        Ok(CollectiveOpResult {
            id: request.id.clone(),
            status: OperationStatus::Success,
            duration,
            bandwidth_gb_s,
            error_message: None,
        })
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(5 + (request.data_size / 1024) as u64)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::AllGather)
    }
}

/// Broadcast operation handler
#[derive(Debug)]
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
        let start_time = Instant::now();

        if request.root_device.is_none() {
            return Err(SynchronizationError::InvalidOperation {
                reason: "Broadcast requires a root device".to_string(),
            });
        }

        // Simulate broadcast operation
        std::thread::sleep(Duration::from_millis(3));

        let duration = start_time.elapsed();
        let bandwidth_gb_s =
            (request.data_size as f64) / duration.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

        Ok(CollectiveOpResult {
            id: request.id.clone(),
            status: OperationStatus::Success,
            duration,
            bandwidth_gb_s,
            error_message: None,
        })
    }

    fn estimate_time(&self, request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(3 + (request.data_size / 2048) as u64)
    }

    fn supports_operation(&self, op_type: CollectiveOpType) -> bool {
        matches!(op_type, CollectiveOpType::Broadcast)
    }
}

/// Barrier operation handler
#[derive(Debug)]
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
        let start_time = Instant::now();

        // Simulate barrier synchronization
        std::thread::sleep(Duration::from_millis(1));

        let duration = start_time.elapsed();

        Ok(CollectiveOpResult {
            id: request.id.clone(),
            status: OperationStatus::Success,
            duration,
            bandwidth_gb_s: 0.0, // No data transfer for barriers
            error_message: None,
        })
    }

    fn estimate_time(&self, _request: &CollectiveOpRequest) -> Duration {
        Duration::from_micros(100) // Fast synchronization
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

        if !devices.is_empty() {
            let root = devices[0];

            // Simple binary tree
            for (i, &device) in devices.iter().enumerate() {
                let mut children = Vec::new();

                let left_child_idx = 2 * i + 1;
                let right_child_idx = 2 * i + 2;

                if left_child_idx < devices.len() {
                    children.push(devices[left_child_idx]);
                    child_parent.insert(devices[left_child_idx], device);
                }

                if right_child_idx < devices.len() {
                    children.push(devices[right_child_idx]);
                    child_parent.insert(devices[right_child_idx], device);
                }

                if !children.is_empty() {
                    connections.insert(device, children.clone());
                    parent_child.insert(device, children);
                }
            }
        }

        let tree = if !devices.is_empty() {
            Some(CommunicationTree {
                root: devices[0],
                parent_child,
                child_parent,
                depth: (devices.len() as f64).log2().ceil() as u32,
            })
        } else {
            None
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

        let barriers = sync_manager.active_barriers.lock().unwrap();
        assert!(barriers.contains_key("test_barrier"));
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

        let request = CollectiveOpRequest {
            id: "test_allreduce".to_string(),
            op_type: CollectiveOpType::AllReduce,
            devices,
            root_device: None,
            reduction_op: Some(ReductionOp::Sum),
            data_size: 1024,
            timeout: Duration::from_secs(10),
            priority: OperationPriority::Normal,
        };

        let result = sync_manager.execute_collective_op(request);
        assert!(result.is_ok());

        let op_result = result.unwrap();
        assert_eq!(op_result.status, OperationStatus::Success);
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

        for (_, neighbors) in &mesh_topology.connections {
            assert_eq!(neighbors.len(), 3); // Each device connects to 3 others
        }
    }
}
