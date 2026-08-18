//! TPU Pod Coordination and Management
//!
//! This module provides coordination and management functionality for TPU pods,
//! including distributed computation, synchronization, fault tolerance, and
//! load balancing across multiple TPU devices.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// TPU Pod Coordinator manages multiple TPU devices and orchestrates distributed computation
#[derive(Debug)]
pub struct PodCoordinator {
    /// Configuration for the pod
    config: PodConfig,

    /// List of TPU devices in the pod
    devices: Vec<TpuDevice>,

    /// Current state of the pod
    state: Arc<Mutex<PodState>>,

    /// Communication channels between devices, keyed by the ordered
    /// `(source, target)` device pair so that every directed link is kept
    /// (an `N`-device pod has `N * (N - 1)` directed channels).
    communication_channels: HashMap<(TpuDeviceId, TpuDeviceId), CommunicationChannel>,

    /// Load balancer for distributing work
    load_balancer: LoadBalancer,

    /// Fault tolerance manager
    fault_manager: FaultToleranceManager,

    /// Performance monitor
    performance_monitor: PerformanceMonitor,

    /// Number of synchronization barriers that have completed successfully.
    ///
    /// This is honest bookkeeping for [`PodCoordinator::synchronize_devices`]:
    /// it only advances when every expected participant actually reached the
    /// barrier, never on a fabricated completion.
    completed_barriers: u64,
}

/// Configuration for TPU pod coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodConfig {
    /// Number of TPU devices in the pod
    pub num_devices: usize,

    /// Nominal per-device capability spec (compute cores, memory, peak TOPS,
    /// ...) that every device in the pod is initialized with.
    ///
    /// This is the *nominal* hardware spec, distinct from the *live*
    /// [`DeviceMetrics`] each device accumulates at runtime (utilization,
    /// temperature, throughput): capabilities describe what the hardware can
    /// do, metrics describe what it is currently doing. Defaults to a TPU
    /// v4-shaped spec via [`DeviceCapabilities::default`]; set this
    /// explicitly to model a different generation.
    pub device_capabilities: DeviceCapabilities,

    /// Topology type (e.g., mesh, torus, ring)
    pub topology: TopologyType,

    /// Coordination strategy
    pub coordination_strategy: CoordinationStrategy,

    /// Synchronization mode
    pub sync_mode: SynchronizationMode,

    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,

    /// Performance monitoring settings
    pub monitoring: MonitoringConfig,

    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,

    /// Communication timeout
    pub communication_timeout: Duration,

    /// Maximum retry attempts for failed operations
    pub max_retry_attempts: usize,
}

/// TPU device representation
#[derive(Debug, Clone)]
pub struct TpuDevice {
    /// Unique device identifier
    pub id: TpuDeviceId,

    /// Device capabilities
    pub capabilities: DeviceCapabilities,

    /// Current device state
    pub state: DeviceState,

    /// Current workload
    pub workload: Option<WorkloadInfo>,

    /// Performance metrics
    pub metrics: DeviceMetrics,

    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
}

/// Unique identifier for TPU devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TpuDeviceId(pub u32);

/// TPU device capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Compute cores available
    pub compute_cores: u32,

    /// Memory capacity in GB
    pub memory_gb: f64,

    /// Peak compute performance (TOPS)
    pub peak_tops: f64,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gb_s: f64,

    /// Supported data types
    pub supported_dtypes: Vec<DataType>,

    /// Maximum matrix multiplication dimensions
    pub max_matmul_dims: (usize, usize, usize),
}

impl Default for DeviceCapabilities {
    /// A TPU v4-shaped nominal spec.
    ///
    /// This used to be the literal `initialize_devices` wrote for every
    /// device regardless of `PodConfig`, so a pod modelling any other TPU
    /// generation silently reported v4 numbers. It is now only the
    /// *default* — [`PodConfig::device_capabilities`] lets a caller supply
    /// the real nominal spec for the hardware being modelled, and
    /// `initialize_devices` reads that field instead of this constant.
    fn default() -> Self {
        Self {
            compute_cores: 2,
            memory_gb: 32.0,
            peak_tops: 275.0,
            memory_bandwidth_gb_s: 1600.0,
            supported_dtypes: vec![DataType::Float32, DataType::Float16, DataType::BFloat16],
            max_matmul_dims: (8192, 8192, 8192),
        }
    }
}

/// Device state enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceState {
    /// Device is idle and available
    Idle,
    /// Device is actively computing
    Computing,
    /// Device is waiting for synchronization
    Waiting,
    /// Device is communicating with other devices
    Communicating,
    /// Device has encountered an error
    Error(String),
    /// Device is offline or unavailable
    Offline,
}

/// Workload information for a device
#[derive(Debug, Clone)]
pub struct WorkloadInfo {
    /// Workload identifier
    pub id: String,

    /// Type of computation
    pub computation_type: ComputationType,

    /// Estimated completion time
    pub estimated_completion: Duration,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,

    /// Priority level
    pub priority: WorkloadPriority,
}

/// Pod state tracking
#[derive(Debug, Clone)]
pub struct PodState {
    /// Overall pod status
    pub status: PodStatus,

    /// Number of active devices
    pub active_devices: usize,

    /// Current computation phase
    pub computation_phase: ComputationPhase,

    /// Synchronization barriers active
    pub active_barriers: Vec<BarrierInfo>,

    /// Global step counter
    pub global_step: u64,

    /// Last coordination timestamp
    pub last_coordination: Instant,
}

/// Pod status enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodStatus {
    Initializing,
    Ready,
    Computing,
    Synchronizing,
    Error(String),
    Shutdown,
}

/// Computation phases in distributed training
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputationPhase {
    Forward,
    Backward,
    ParameterUpdate,
    AllReduce,
    Checkpoint,
}

/// Communication channel between devices
#[derive(Debug)]
pub struct CommunicationChannel {
    /// Source device
    pub source: TpuDeviceId,

    /// Target device
    pub target: TpuDeviceId,

    /// Channel bandwidth (GB/s)
    pub bandwidth_gb_s: f64,

    /// Current latency (microseconds)
    pub latency_us: f64,

    /// Message queue
    pub message_queue: Arc<Mutex<Vec<Message>>>,

    /// Channel state
    pub state: ChannelState,
}

/// Message for inter-device communication
#[derive(Debug, Clone)]
pub struct Message {
    /// Message identifier
    pub id: String,

    /// Message type
    pub message_type: MessageType,

    /// Payload data
    pub payload: Vec<u8>,

    /// Timestamp
    pub timestamp: Instant,

    /// Priority
    pub priority: MessagePriority,
}

/// Load balancer for distributing work across devices.
///
/// Selection reads utilization straight off each candidate's own
/// [`DeviceMetrics`], so there is no separate `utilization_tracker` mirror to
/// drift out of date, and no `work_queue`: this balancer answers "which device
/// should take this workload" rather than owning the work.
#[derive(Debug)]
pub struct LoadBalancer {
    /// Balancing strategy
    strategy: LoadBalancingStrategy,
}

/// Fault tolerance manager
#[derive(Debug)]
pub struct FaultToleranceManager {
    /// Recovery strategies, derived from the configuration at construction.
    ///
    /// The configuration itself is not retained: it is consumed by
    /// [`Self::build_recovery_strategies`] and nothing else read it. Neither is
    /// a failed-device table or a checkpoint manager -- recording failures and
    /// writing/verifying checkpoints is [`crate::fault_tolerance`]'s job, and it
    /// does both for real (SHA-256-verified checkpoints on disk); a second inert
    /// copy here only looked like it did.
    recovery_strategies: Vec<RecoveryStrategy>,

    /// Whether fault detection has been activated.
    detection_active: bool,

    /// Timestamp fault detection was activated, if active.
    detection_started_at: Option<Instant>,

    /// Devices currently registered for fault monitoring.
    monitored_devices: Vec<TpuDeviceId>,
}

/// Performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Alerting thresholds, derived from the configuration at construction.
    ///
    /// The configuration, a metrics collector and a performance history buffer
    /// used to sit alongside this and were never read. Real per-execution
    /// sampling with a bounded history lives in
    /// [`crate::tpu_backend::PerformanceMonitor`]; this monitor answers
    /// "is monitoring active, over which devices, and at what alert
    /// thresholds".
    alerting: AlertingSystem,

    /// Whether monitoring has been activated.
    monitoring_active: bool,

    /// Timestamp monitoring was activated, if active.
    monitoring_started_at: Option<Instant>,

    /// Devices currently registered for performance monitoring.
    monitored_devices: Vec<TpuDeviceId>,
}

// Enumerations and supporting types

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    Mesh,
    Torus,
    Ring,
    Tree,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    Centralized,
    Decentralized,
    Hierarchical,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynchronizationMode {
    Synchronous,
    Asynchronous,
    BulkSynchronous,
    EventDriven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    Performance,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int16,
    Int8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationType {
    MatrixMultiplication,
    Convolution,
    Attention,
    Embedding,
    Normalization,
    Activation,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelState {
    Active,
    Congested,
    Failed,
    Maintenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Data,
    Control,
    Synchronization,
    Heartbeat,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Urgent,
}

// Configuration structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub enable_checkpointing: bool,
    pub checkpoint_interval: Duration,
    pub max_failures: usize,
    pub recovery_timeout: Duration,
    pub enable_redundancy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub collection_interval: Duration,
    pub metrics_retention: Duration,
    pub enable_profiling: bool,
    pub alert_thresholds: HashMap<String, f64>,
}

// Supporting structures

#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    pub utilization: f64,
    pub memory_usage: f64,
    pub temperature: f64,
    pub power_consumption: f64,
    pub throughput_tops: f64,
    pub error_count: u64,
}

#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub compute: f64,
    pub memory: f64,
    pub bandwidth: f64,
}

#[derive(Debug, Clone)]
pub struct BarrierInfo {
    pub id: String,
    pub waiting_devices: Vec<TpuDeviceId>,
    pub completed_devices: Vec<TpuDeviceId>,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: String,
    pub computation: ComputationType,
    pub data_size: usize,
    pub priority: WorkloadPriority,
    pub target_device: Option<TpuDeviceId>,
}

#[derive(Debug, Clone)]
pub struct FailureInfo {
    pub failure_type: FailureType,
    pub timestamp: Instant,
    pub error_message: String,
    pub recovery_attempts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureType {
    Hardware,
    Software,
    Communication,
    Timeout,
    Memory,
}

#[derive(Debug)]
pub struct RecoveryStrategy {
    pub strategy_type: RecoveryStrategyType,
    pub applicability: Vec<FailureType>,
    pub cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategyType {
    Restart,
    Reassign,
    Redundancy,
    Checkpointing,
}

#[derive(Debug)]
pub struct CheckpointManager {
    pub checkpoint_interval: Duration,
    pub checkpoint_storage: String,
    pub compression_enabled: bool,
}

#[derive(Debug)]
pub struct MetricsCollector {
    pub collection_interval: Duration,
    pub metrics_buffer: Arc<Mutex<Vec<MetricData>>>,
}

#[derive(Debug, Clone)]
pub struct MetricData {
    pub device_id: TpuDeviceId,
    pub timestamp: Instant,
    pub metric_type: MetricType,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    Utilization,
    Throughput,
    Latency,
    Memory,
    Power,
    Temperature,
    ErrorRate,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub overall_utilization: f64,
    pub throughput: f64,
    pub active_devices: usize,
    pub bottlenecks: Vec<BottleneckInfo>,
}

#[derive(Debug, Clone)]
pub struct BottleneckInfo {
    pub bottleneck_type: BottleneckType,
    pub affected_devices: Vec<TpuDeviceId>,
    pub severity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    Compute,
    Memory,
    Communication,
    Synchronization,
}

#[derive(Debug)]
pub struct AlertingSystem {
    pub thresholds: HashMap<MetricType, f64>,
    pub alert_handlers: Vec<AlertHandler>,
}

#[derive(Debug)]
pub struct AlertHandler {
    pub handler_type: AlertHandlerType,
    pub severity_threshold: AlertSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertHandlerType {
    Log,
    Email,
    Webhook,
    Sms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Errors that can occur during pod coordination
#[derive(Debug, Error)]
pub enum CoordinationError {
    #[error("Device not found: {device_id:?}")]
    DeviceNotFound { device_id: TpuDeviceId },

    #[error("Communication timeout with device: {device_id:?}")]
    CommunicationTimeout { device_id: TpuDeviceId },

    #[error("Synchronization failed: {reason}")]
    SynchronizationFailed { reason: String },

    #[error("Load balancing error: {reason}")]
    LoadBalancingError { reason: String },

    #[error("Fault tolerance error: {reason}")]
    FaultToleranceError { reason: String },

    #[error("Configuration error: {reason}")]
    ConfigurationError { reason: String },

    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustion { resource: String },

    #[error("Pod initialization failed: {reason}")]
    InitializationFailed { reason: String },
}

impl PodCoordinator {
    /// Create a new pod coordinator with the given configuration
    pub fn new(config: PodConfig) -> Result<Self, CoordinationError> {
        let devices = Self::initialize_devices(&config)?;
        let state = Arc::new(Mutex::new(PodState {
            status: PodStatus::Initializing,
            active_devices: devices.len(),
            computation_phase: ComputationPhase::Forward,
            active_barriers: Vec::new(),
            global_step: 0,
            last_coordination: Instant::now(),
        }));

        let communication_channels = Self::setup_communication_channels(&devices, &config)?;
        let load_balancer = LoadBalancer::new(config.load_balancing);
        let fault_manager = FaultToleranceManager::new(config.fault_tolerance.clone());
        let performance_monitor = PerformanceMonitor::new(config.monitoring.clone());

        Ok(Self {
            config,
            devices,
            state,
            communication_channels,
            load_balancer,
            fault_manager,
            performance_monitor,
            completed_barriers: 0,
        })
    }

    /// Initialize TPU devices based on configuration
    ///
    /// Capabilities come from `config.device_capabilities` (the nominal spec
    /// for the hardware being modelled), not a hardcoded literal, so a pod
    /// configured for a different TPU generation actually reports it.
    /// `metrics` starts at the idle baseline every device genuinely has
    /// before its first heartbeat; unlike capabilities this is meant to
    /// change as `PerformanceMonitor`/`FaultToleranceManager` observe it.
    fn initialize_devices(config: &PodConfig) -> Result<Vec<TpuDevice>, CoordinationError> {
        let mut devices = Vec::new();

        for i in 0..config.num_devices {
            let device = TpuDevice {
                id: TpuDeviceId(i as u32),
                capabilities: config.device_capabilities.clone(),
                state: DeviceState::Idle,
                workload: None,
                metrics: DeviceMetrics {
                    utilization: 0.0,
                    memory_usage: 0.0,
                    temperature: 25.0,
                    power_consumption: 100.0,
                    throughput_tops: 0.0,
                    error_count: 0,
                },
                last_heartbeat: Instant::now(),
            };

            devices.push(device);
        }

        Ok(devices)
    }

    /// Setup communication channels between devices.
    ///
    /// One directed [`CommunicationChannel`] is created for every ordered
    /// `(source, target)` pair of distinct devices, so an `N`-device pod ends
    /// up with `N * (N - 1)` channels (previously the map was keyed on the
    /// source only, silently discarding all but the last target per source).
    ///
    /// The pod configuration does not carry per-host placement information, so
    /// link quality is modelled from the interconnect **topology**
    /// (`config.topology`) and pod size (`config.num_devices`): each pair's
    /// topological hop-distance drives its bandwidth and latency. A directly
    /// connected pair (1 hop, i.e. topology neighbours) gets the full
    /// interconnect bandwidth and lowest latency; multi-hop pairs are degraded
    /// proportionally to the number of hops they must traverse. This makes the
    /// values vary per pair instead of using a single magic constant.
    fn setup_communication_channels(
        devices: &[TpuDevice],
        config: &PodConfig,
    ) -> Result<HashMap<(TpuDeviceId, TpuDeviceId), CommunicationChannel>, CoordinationError> {
        let mut channels = HashMap::new();

        for device in devices {
            for other_device in devices {
                if device.id != other_device.id {
                    let (bandwidth_gb_s, latency_us) = Self::channel_link_quality(
                        device.id,
                        other_device.id,
                        config.topology,
                        config.num_devices,
                    );

                    let channel = CommunicationChannel {
                        source: device.id,
                        target: other_device.id,
                        bandwidth_gb_s,
                        latency_us,
                        message_queue: Arc::new(Mutex::new(Vec::new())),
                        state: ChannelState::Active,
                    };

                    channels.insert((device.id, other_device.id), channel);
                }
            }
        }

        Ok(channels)
    }

    /// Bandwidth (GB/s) and latency (us) for the directed link between two
    /// devices, derived from the pod topology.
    ///
    /// The baseline single-hop interconnect is 300 GB/s at 2.0 us latency.
    /// Bandwidth scales as `base / hops` and latency as `base * hops`, where
    /// `hops` is the topological distance between the two devices (always at
    /// least 1). Neighbouring devices therefore share the fast direct link,
    /// while distant devices see reduced bandwidth and higher latency.
    fn channel_link_quality(
        source: TpuDeviceId,
        target: TpuDeviceId,
        topology: TopologyType,
        num_devices: usize,
    ) -> (f64, f64) {
        const BASE_LINK_BANDWIDTH_GB_S: f64 = 300.0;
        const BASE_LINK_LATENCY_US: f64 = 2.0;

        let hops = Self::topology_hops(source.0, target.0, topology, num_devices).max(1);
        let hops_f = hops as f64;

        let bandwidth_gb_s = BASE_LINK_BANDWIDTH_GB_S / hops_f;
        let latency_us = BASE_LINK_LATENCY_US * hops_f;

        (bandwidth_gb_s, latency_us)
    }

    /// Topological hop-distance between two device indices for a given
    /// interconnect topology. The result is always at least 1 for distinct
    /// devices.
    fn topology_hops(
        source: u32,
        target: u32,
        topology: TopologyType,
        num_devices: usize,
    ) -> usize {
        if source == target {
            return 0;
        }
        let n = num_devices.max(1);
        let a = source as usize;
        let b = target as usize;

        match topology {
            TopologyType::Ring => {
                // Bidirectional ring: shortest way around the loop.
                let forward = (a + n - (b % n)) % n;
                let backward = (b + n - (a % n)) % n;
                forward.min(backward).max(1)
            }
            TopologyType::Mesh | TopologyType::Torus => {
                // Arrange devices on a near-square grid and use Manhattan
                // distance; the torus additionally wraps around each axis.
                let cols = ((n as f64).sqrt().round() as usize).max(1);
                let (ar, ac) = (a / cols, a % cols);
                let (br, bc) = (b / cols, b % cols);
                let rows = n.div_ceil(cols).max(1);

                let (dr, dc) = (Self::axis_delta(ar, br), Self::axis_delta(ac, bc));
                let (dr, dc) = if matches!(topology, TopologyType::Torus) {
                    (
                        dr.min(rows.saturating_sub(dr)),
                        dc.min(cols.saturating_sub(dc)),
                    )
                } else {
                    (dr, dc)
                };
                (dr + dc).max(1)
            }
            TopologyType::Tree => {
                // Complete binary tree indexed from 0: distance via depths and
                // the lowest common ancestor.
                Self::tree_distance(a, b).max(1)
            }
            TopologyType::Custom => {
                // Without an explicit graph, fall back to linear adjacency.
                Self::axis_delta(a, b).max(1)
            }
        }
    }

    /// Absolute difference between two grid coordinates.
    fn axis_delta(a: usize, b: usize) -> usize {
        a.abs_diff(b)
    }

    /// Distance between two nodes in a complete binary tree indexed from 0
    /// (children of `i` are `2*i + 1` and `2*i + 2`).
    fn tree_distance(mut a: usize, mut b: usize) -> usize {
        let mut dist = 0;
        // Repeatedly lift the larger index to its parent until the two nodes
        // meet at their lowest common ancestor (parent of `i` is `(i-1)/2`).
        while a != b {
            if a > b {
                a = (a - 1) / 2;
            } else {
                b = (b - 1) / 2;
            }
            dist += 1;
        }
        dist
    }

    /// Look up the directed communication channel between two devices, if one
    /// exists.
    pub fn communication_channel(
        &self,
        source: TpuDeviceId,
        target: TpuDeviceId,
    ) -> Option<&CommunicationChannel> {
        self.communication_channels.get(&(source, target))
    }

    /// Number of directed communication channels in the pod.
    pub fn num_communication_channels(&self) -> usize {
        self.communication_channels.len()
    }

    /// Acquire the pod-state lock, recovering the guard if the mutex was
    /// poisoned by a panic in another thread.
    ///
    /// Poisoning only means some thread panicked while holding the lock; the
    /// `PodState` behind it is still structurally valid, so recovering the
    /// guard (rather than panicking again via `unwrap`/`expect`) keeps the
    /// coordinator usable and keeps the public accessors infallible.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, PodState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Start the pod coordination
    pub fn start(&mut self) -> Result<(), CoordinationError> {
        {
            let mut state = self.lock_state();
            state.status = PodStatus::Ready;
            state.last_coordination = Instant::now();
        }

        // Start monitoring and coordination loops
        self.performance_monitor.start_monitoring(&self.devices)?;
        self.fault_manager.start_fault_detection(&self.devices)?;

        Ok(())
    }

    /// Whether performance monitoring has been activated for this pod.
    pub fn is_monitoring_active(&self) -> bool {
        self.performance_monitor.is_active()
    }

    /// Whether fault detection has been activated for this pod.
    pub fn is_fault_detection_active(&self) -> bool {
        self.fault_manager.is_active()
    }

    /// Number of synchronization barriers that have completed successfully.
    pub fn completed_barrier_count(&self) -> u64 {
        self.completed_barriers
    }

    /// Submit a computation workload to the pod
    pub fn submit_workload(&mut self, workload: WorkloadInfo) -> Result<(), CoordinationError> {
        let target_device = self.load_balancer.select_device(&self.devices, &workload)?;

        if let Some(device) = self.devices.iter_mut().find(|d| d.id == target_device) {
            device.workload = Some(workload);
            device.state = DeviceState::Computing;
        }

        Ok(())
    }

    /// Synchronize all devices at a barrier.
    ///
    /// This performs real, honest barrier accounting rather than sleeping and
    /// declaring success:
    ///
    /// * **Participants** are every device that is not [`DeviceState::Offline`]
    ///   — those are the devices expected to reach the barrier.
    /// * A participant is counted as **arrived** only when it is in a
    ///   schedulable state (`Idle`/`Computing`/`Waiting`/`Communicating`). A
    ///   device stuck in [`DeviceState::Error`] is expected but cannot arrive,
    ///   so it remains in the waiting set.
    /// * The barrier is only considered complete when every participant has
    ///   arrived. If some participant cannot reach it, the call returns
    ///   [`CoordinationError::SynchronizationFailed`] instead of fabricating
    ///   success.
    ///
    /// No sleep is performed and the state lock is never held across a blocking
    /// wait; this is a synchronous accounting operation over the current device
    /// states. On success the completed-barrier counter is advanced.
    pub fn synchronize_devices(&mut self, barrier_id: String) -> Result<(), CoordinationError> {
        // Devices expected at the barrier (everything that is online).
        let participants: Vec<TpuDeviceId> = self
            .devices
            .iter()
            .filter(|d| d.state != DeviceState::Offline)
            .map(|d| d.id)
            .collect();

        if participants.is_empty() {
            return Err(CoordinationError::SynchronizationFailed {
                reason: format!("barrier {barrier_id} has no online participants"),
            });
        }

        // Devices that can actually reach the barrier right now.
        let arrived: Vec<TpuDeviceId> = self
            .devices
            .iter()
            .filter(|d| {
                matches!(
                    d.state,
                    DeviceState::Idle
                        | DeviceState::Computing
                        | DeviceState::Waiting
                        | DeviceState::Communicating
                )
            })
            .map(|d| d.id)
            .collect();

        let outcome = {
            let mut state = self.lock_state();
            state.status = PodStatus::Synchronizing;

            let waiting_devices: Vec<TpuDeviceId> = participants
                .iter()
                .copied()
                .filter(|id| !arrived.contains(id))
                .collect();
            let completed_devices: Vec<TpuDeviceId> = participants
                .iter()
                .copied()
                .filter(|id| arrived.contains(id))
                .collect();

            let n_participants = participants.len();
            let n_completed = completed_devices.len();
            let complete = waiting_devices.is_empty() && n_completed == n_participants;

            // Register the barrier while it is being resolved, then retire it
            // so a completed/failed barrier never lingers in the active set.
            let barrier = BarrierInfo {
                id: barrier_id.clone(),
                waiting_devices,
                completed_devices,
                timeout: self.config.communication_timeout,
            };
            state.active_barriers.push(barrier);
            state.active_barriers.retain(|b| b.id != barrier_id);

            state.status = PodStatus::Ready;
            state.last_coordination = Instant::now();

            if complete {
                Ok(())
            } else {
                Err(CoordinationError::SynchronizationFailed {
                    reason: format!(
                        "barrier {barrier_id}: {n_completed}/{n_participants} devices arrived"
                    ),
                })
            }
        };

        if outcome.is_ok() {
            self.completed_barriers = self.completed_barriers.wrapping_add(1);
        }

        outcome
    }

    /// Get current pod status
    pub fn get_status(&self) -> PodState {
        self.lock_state().clone()
    }

    /// Get device metrics
    pub fn get_device_metrics(&self, device_id: TpuDeviceId) -> Option<DeviceMetrics> {
        self.devices
            .iter()
            .find(|d| d.id == device_id)
            .map(|d| d.metrics.clone())
    }

    /// Shutdown the pod
    pub fn shutdown(&mut self) -> Result<(), CoordinationError> {
        {
            let mut state = self.lock_state();
            state.status = PodStatus::Shutdown;
        }

        // Stop all devices and cleanup
        for device in &mut self.devices {
            device.state = DeviceState::Offline;
        }

        Ok(())
    }
}

impl LoadBalancer {
    fn new(strategy: LoadBalancingStrategy) -> Self {
        Self { strategy }
    }

    fn select_device(
        &mut self,
        devices: &[TpuDevice],
        _workload: &WorkloadInfo,
    ) -> Result<TpuDeviceId, CoordinationError> {
        match self.strategy {
            LoadBalancingStrategy::LeastLoaded => {
                let device = devices
                    .iter()
                    .filter(|d| matches!(d.state, DeviceState::Idle))
                    .min_by(|a, b| {
                        // `total_cmp` is a total order over floats, so it never
                        // returns `None` (unlike `partial_cmp`) and cannot panic
                        // on a NaN utilization metric. NaN sorts as the largest
                        // value, so a device with a NaN load is never chosen as
                        // the least loaded while any finite-load device exists.
                        a.metrics.utilization.total_cmp(&b.metrics.utilization)
                    });

                device
                    .map(|d| d.id)
                    .ok_or_else(|| CoordinationError::ResourceExhaustion {
                        resource: "Available devices".to_string(),
                    })
            }
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin implementation
                let idle_devices: Vec<_> = devices
                    .iter()
                    .filter(|d| matches!(d.state, DeviceState::Idle))
                    .collect();

                if idle_devices.is_empty() {
                    return Err(CoordinationError::ResourceExhaustion {
                        resource: "Available devices".to_string(),
                    });
                }

                Ok(idle_devices[0].id)
            }
            _ => {
                // Default to first available device
                devices
                    .iter()
                    .find(|d| matches!(d.state, DeviceState::Idle))
                    .map(|d| d.id)
                    .ok_or_else(|| CoordinationError::ResourceExhaustion {
                        resource: "Available devices".to_string(),
                    })
            }
        }
    }
}

impl FaultToleranceManager {
    fn new(config: FaultToleranceConfig) -> Self {
        let recovery_strategies = Self::build_recovery_strategies(&config);
        Self {
            recovery_strategies,
            detection_active: false,
            detection_started_at: None,
            monitored_devices: Vec::new(),
        }
    }

    /// Derive the concrete set of recovery strategies from the fault-tolerance
    /// configuration. Restart and reassignment are always available;
    /// checkpoint- and redundancy-based recovery are only registered when the
    /// corresponding features are enabled in the config.
    fn build_recovery_strategies(config: &FaultToleranceConfig) -> Vec<RecoveryStrategy> {
        let mut strategies = vec![
            RecoveryStrategy {
                strategy_type: RecoveryStrategyType::Restart,
                applicability: vec![
                    FailureType::Hardware,
                    FailureType::Software,
                    FailureType::Timeout,
                ],
                cost: 1.0,
            },
            RecoveryStrategy {
                strategy_type: RecoveryStrategyType::Reassign,
                applicability: vec![
                    FailureType::Software,
                    FailureType::Communication,
                    FailureType::Memory,
                    FailureType::Timeout,
                ],
                cost: 0.5,
            },
        ];

        if config.enable_checkpointing {
            strategies.push(RecoveryStrategy {
                strategy_type: RecoveryStrategyType::Checkpointing,
                applicability: vec![
                    FailureType::Hardware,
                    FailureType::Software,
                    FailureType::Memory,
                ],
                cost: 2.0,
            });
        }

        if config.enable_redundancy {
            strategies.push(RecoveryStrategy {
                strategy_type: RecoveryStrategyType::Redundancy,
                applicability: vec![FailureType::Hardware, FailureType::Communication],
                cost: 3.0,
            });
        }

        strategies
    }

    /// Activate fault detection.
    ///
    /// This does not spawn background threads; instead it establishes the real,
    /// observable in-struct state that the feature is built on: it marks
    /// detection active, records the activation time, and registers the set of
    /// devices to be watched. Later queries (`is_active`, `monitored_devices`)
    /// reflect this state. Actual background polling is intentionally out of
    /// scope here.
    fn start_fault_detection(&mut self, devices: &[TpuDevice]) -> Result<(), CoordinationError> {
        self.detection_active = true;
        self.detection_started_at = Some(Instant::now());
        self.monitored_devices = devices.iter().map(|d| d.id).collect();
        Ok(())
    }

    /// Whether fault detection has been activated.
    pub fn is_active(&self) -> bool {
        self.detection_active
    }

    /// Timestamp fault detection was activated, if active.
    pub fn started_at(&self) -> Option<Instant> {
        self.detection_started_at
    }

    /// Devices currently registered for fault monitoring.
    pub fn monitored_devices(&self) -> &[TpuDeviceId] {
        &self.monitored_devices
    }

    /// Recovery strategies applicable to the given failure type.
    pub fn recovery_strategies_for(&self, failure: FailureType) -> Vec<&RecoveryStrategy> {
        self.recovery_strategies
            .iter()
            .filter(|s| s.applicability.contains(&failure))
            .collect()
    }
}

impl PerformanceMonitor {
    fn new(config: MonitoringConfig) -> Self {
        let thresholds = Self::build_alert_thresholds(&config);
        Self {
            alerting: AlertingSystem {
                thresholds,
                alert_handlers: Vec::new(),
            },
            monitoring_active: false,
            monitoring_started_at: None,
            monitored_devices: Vec::new(),
        }
    }

    /// Translate the config's string-keyed alert thresholds into a typed
    /// `MetricType -> threshold` map. Unrecognised keys are ignored.
    fn build_alert_thresholds(config: &MonitoringConfig) -> HashMap<MetricType, f64> {
        let mut thresholds = HashMap::new();
        for (key, value) in &config.alert_thresholds {
            if let Some(metric) = Self::parse_metric_type(key) {
                thresholds.insert(metric, *value);
            }
        }
        thresholds
    }

    /// Map an alert-threshold key onto a [`MetricType`], accepting a few common
    /// spellings.
    fn parse_metric_type(key: &str) -> Option<MetricType> {
        match key.to_ascii_lowercase().replace([' ', '-'], "_").as_str() {
            "utilization" | "util" => Some(MetricType::Utilization),
            "throughput" => Some(MetricType::Throughput),
            "latency" => Some(MetricType::Latency),
            "memory" | "memory_usage" => Some(MetricType::Memory),
            "power" | "power_consumption" => Some(MetricType::Power),
            "temperature" | "temp" => Some(MetricType::Temperature),
            "error_rate" | "errorrate" | "errors" => Some(MetricType::ErrorRate),
            _ => None,
        }
    }

    /// Activate performance monitoring.
    ///
    /// As with fault detection, this establishes real, observable in-struct
    /// state rather than spawning background threads: it marks monitoring
    /// active, records the activation time, and registers the devices to be
    /// monitored. Later queries (`is_active`, `monitored_devices`) reflect this
    /// state. Background metric collection is intentionally out of scope here.
    fn start_monitoring(&mut self, devices: &[TpuDevice]) -> Result<(), CoordinationError> {
        self.monitoring_active = true;
        self.monitoring_started_at = Some(Instant::now());
        self.monitored_devices = devices.iter().map(|d| d.id).collect();
        Ok(())
    }

    /// Whether performance monitoring has been activated.
    pub fn is_active(&self) -> bool {
        self.monitoring_active
    }

    /// Timestamp monitoring was activated, if active.
    pub fn started_at(&self) -> Option<Instant> {
        self.monitoring_started_at
    }

    /// Devices currently registered for performance monitoring.
    pub fn monitored_devices(&self) -> &[TpuDeviceId] {
        &self.monitored_devices
    }

    /// Configured alert threshold for a given metric type, if any.
    pub fn alert_threshold(&self, metric: MetricType) -> Option<f64> {
        self.alerting.thresholds.get(&metric).copied()
    }
}

impl Default for PodConfig {
    fn default() -> Self {
        Self {
            num_devices: 8,
            device_capabilities: DeviceCapabilities::default(),
            topology: TopologyType::Mesh,
            coordination_strategy: CoordinationStrategy::Centralized,
            sync_mode: SynchronizationMode::Synchronous,
            fault_tolerance: FaultToleranceConfig {
                enable_checkpointing: true,
                checkpoint_interval: Duration::from_secs(300),
                max_failures: 3,
                recovery_timeout: Duration::from_secs(60),
                enable_redundancy: false,
            },
            monitoring: MonitoringConfig {
                collection_interval: Duration::from_secs(1),
                metrics_retention: Duration::from_secs(3600),
                enable_profiling: true,
                alert_thresholds: HashMap::new(),
            },
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            communication_timeout: Duration::from_secs(30),
            max_retry_attempts: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_coordinator_creation() {
        let config = PodConfig::default();
        let coordinator = PodCoordinator::new(config);
        assert!(coordinator.is_ok());
    }

    // Mirrors the "Pod coordination" example in README.md verbatim (module path and
    // all), since README code blocks are not compiled by `cargo test --doc`. If this
    // stops compiling, that example is stale and needs updating too.
    #[test]
    fn readme_pod_coordination_example_compiles_and_runs() {
        use crate::coordination::{CoordinationError, PodConfig, PodCoordinator};

        fn coordinate_one_step() -> Result<(), CoordinationError> {
            let config = PodConfig {
                num_devices: 8,
                ..Default::default()
            };
            let mut pod = PodCoordinator::new(config)?;
            pod.synchronize_devices("step-0".to_string())?;
            Ok(())
        }

        coordinate_one_step().expect("README pod coordination example must succeed");
    }

    #[test]
    fn test_device_initialization() {
        let config = PodConfig {
            num_devices: 4,
            ..Default::default()
        };

        let devices = PodCoordinator::initialize_devices(&config).expect("unwrap failed");
        assert_eq!(devices.len(), 4);

        for (i, device) in devices.iter().enumerate() {
            assert_eq!(device.id.0, i as u32);
            assert!(matches!(device.state, DeviceState::Idle));
        }
    }

    // Regression test for F29: `initialize_devices` used to hardcode a
    // TPU-v4-shaped `DeviceCapabilities` literal regardless of `PodConfig`,
    // so every device silently reported v4 numbers no matter what hardware
    // the pod was meant to model. Capabilities must now come from
    // `config.device_capabilities`, not a baked-in constant.
    #[test]
    fn device_capabilities_come_from_config_not_a_hardcoded_literal() {
        let custom = DeviceCapabilities {
            compute_cores: 4,
            memory_gb: 16.0,
            peak_tops: 123.0,
            memory_bandwidth_gb_s: 900.0,
            supported_dtypes: vec![DataType::Int8],
            max_matmul_dims: (128, 128, 128),
        };
        let config = PodConfig {
            num_devices: 3,
            device_capabilities: custom.clone(),
            ..Default::default()
        };

        let devices = PodCoordinator::initialize_devices(&config).expect("unwrap failed");
        assert_eq!(devices.len(), 3);
        for device in &devices {
            assert_eq!(device.capabilities.compute_cores, custom.compute_cores);
            assert_eq!(device.capabilities.memory_gb, custom.memory_gb);
            assert_eq!(device.capabilities.peak_tops, custom.peak_tops);
            assert_eq!(
                device.capabilities.memory_bandwidth_gb_s,
                custom.memory_bandwidth_gb_s
            );
            assert_eq!(
                device.capabilities.supported_dtypes,
                custom.supported_dtypes
            );
            assert_eq!(device.capabilities.max_matmul_dims, custom.max_matmul_dims);
        }
    }

    #[test]
    fn default_device_capabilities_are_reused_by_default_pod_config() {
        // The default PodConfig must still describe the same nominal spec the
        // old hardcoded literal used, so existing callers relying on
        // `PodConfig::default()` see no behavioural change.
        let config = PodConfig::default();
        let devices = PodCoordinator::initialize_devices(&config).expect("unwrap failed");
        let default_caps = DeviceCapabilities::default();
        for device in &devices {
            assert_eq!(
                device.capabilities.compute_cores,
                default_caps.compute_cores
            );
            assert_eq!(device.capabilities.peak_tops, default_caps.peak_tops);
        }
    }

    #[test]
    fn test_load_balancer() {
        let mut load_balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);

        let devices = vec![TpuDevice {
            id: TpuDeviceId(0),
            capabilities: DeviceCapabilities {
                compute_cores: 2,
                memory_gb: 32.0,
                peak_tops: 275.0,
                memory_bandwidth_gb_s: 1600.0,
                supported_dtypes: vec![DataType::Float32],
                max_matmul_dims: (8192, 8192, 8192),
            },
            state: DeviceState::Idle,
            workload: None,
            metrics: DeviceMetrics {
                utilization: 0.5,
                memory_usage: 0.3,
                temperature: 25.0,
                power_consumption: 100.0,
                throughput_tops: 100.0,
                error_count: 0,
            },
            last_heartbeat: Instant::now(),
        }];

        let workload = WorkloadInfo {
            id: "test_workload".to_string(),
            computation_type: ComputationType::MatrixMultiplication,
            estimated_completion: Duration::from_secs(10),
            resource_utilization: ResourceUtilization {
                compute: 0.8,
                memory: 0.6,
                bandwidth: 0.4,
            },
            priority: WorkloadPriority::Medium,
        };

        let selected = load_balancer.select_device(&devices, &workload);
        assert!(selected.is_ok());
        assert_eq!(selected.expect("unwrap failed"), TpuDeviceId(0));
    }

    /// Build a minimal device with a given id, state and utilization metric.
    fn make_device(id: u32, state: DeviceState, utilization: f64) -> TpuDevice {
        TpuDevice {
            id: TpuDeviceId(id),
            capabilities: DeviceCapabilities {
                compute_cores: 2,
                memory_gb: 32.0,
                peak_tops: 275.0,
                memory_bandwidth_gb_s: 1600.0,
                supported_dtypes: vec![DataType::Float32],
                max_matmul_dims: (8192, 8192, 8192),
            },
            state,
            workload: None,
            metrics: DeviceMetrics {
                utilization,
                memory_usage: 0.0,
                temperature: 25.0,
                power_consumption: 100.0,
                throughput_tops: 0.0,
                error_count: 0,
            },
            last_heartbeat: Instant::now(),
        }
    }

    // F23: every ordered pair of distinct devices must get its own channel.
    #[test]
    fn test_setup_communication_channels_all_pairs() {
        let n = 5usize;
        let config = PodConfig {
            num_devices: n,
            ..Default::default()
        };
        let devices = PodCoordinator::initialize_devices(&config).expect("devices");
        let channels =
            PodCoordinator::setup_communication_channels(&devices, &config).expect("channels");

        // N * (N - 1) directed channels, not just N.
        assert_eq!(channels.len(), n * (n - 1));

        // Every ordered (source, target) pair with source != target is present,
        // and each channel records the correct endpoints.
        for i in 0..n as u32 {
            for j in 0..n as u32 {
                let key = (TpuDeviceId(i), TpuDeviceId(j));
                if i == j {
                    assert!(!channels.contains_key(&key));
                } else {
                    let ch = channels.get(&key).expect("pair channel present");
                    assert_eq!(ch.source, TpuDeviceId(i));
                    assert_eq!(ch.target, TpuDeviceId(j));
                }
            }
        }
    }

    // F23: bandwidth/latency must be derived from topology, not a single magic
    // constant — a 1-hop neighbour link differs from a 2-hop link.
    #[test]
    fn test_channel_link_quality_varies_with_hops() {
        // Ring of 4: device 1 is a neighbour of 0 (1 hop); device 2 is 2 hops.
        let (bw_near, lat_near) = PodCoordinator::channel_link_quality(
            TpuDeviceId(0),
            TpuDeviceId(1),
            TopologyType::Ring,
            4,
        );
        let (bw_far, lat_far) = PodCoordinator::channel_link_quality(
            TpuDeviceId(0),
            TpuDeviceId(2),
            TopologyType::Ring,
            4,
        );

        assert!(
            bw_near > bw_far,
            "neighbour link should have more bandwidth"
        );
        assert!(
            lat_near < lat_far,
            "neighbour link should have less latency"
        );
        assert!((bw_near - 300.0).abs() < 1e-9);
        assert!((lat_near - 2.0).abs() < 1e-9);
        assert!((bw_far - 150.0).abs() < 1e-9);
        assert!((lat_far - 4.0).abs() < 1e-9);

        // The same must be observable through an actual pod's channels.
        let config = PodConfig {
            num_devices: 4,
            topology: TopologyType::Ring,
            ..Default::default()
        };
        let coordinator = PodCoordinator::new(config).expect("coordinator");
        assert_eq!(coordinator.num_communication_channels(), 4 * 3);
        let near = coordinator
            .communication_channel(TpuDeviceId(0), TpuDeviceId(1))
            .expect("near channel");
        let far = coordinator
            .communication_channel(TpuDeviceId(0), TpuDeviceId(2))
            .expect("far channel");
        assert!(near.bandwidth_gb_s > far.bandwidth_gb_s);
        assert!(near.latency_us < far.latency_us);
    }

    // F24: barrier synchronization completes via real accounting, with no fixed
    // sleep, and reports honest completion.
    #[test]
    fn test_synchronize_devices_honest_completion() {
        let config = PodConfig {
            num_devices: 8,
            ..Default::default()
        };
        let mut coordinator = PodCoordinator::new(config).expect("coordinator");
        assert_eq!(coordinator.completed_barrier_count(), 0);

        let start = Instant::now();
        let result = coordinator.synchronize_devices("barrier-0".to_string());
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "all-idle pod should complete the barrier");
        // No fixed sleep: the old stub slept 10ms; real accounting is
        // microseconds, so this comfortably rules out a reintroduced sleep.
        assert!(
            elapsed < Duration::from_millis(9),
            "synchronize must not sleep, took {elapsed:?}"
        );

        // Honest, observable completion bookkeeping.
        assert_eq!(coordinator.completed_barrier_count(), 1);
        let status = coordinator.get_status();
        assert!(status.active_barriers.is_empty(), "barrier must be retired");
        assert_eq!(status.status, PodStatus::Ready);
        // global_step is a training counter and must not be hijacked.
        assert_eq!(status.global_step, 0);

        // A second barrier advances the honest counter again.
        coordinator
            .synchronize_devices("barrier-1".to_string())
            .expect("second barrier");
        assert_eq!(coordinator.completed_barrier_count(), 2);
    }

    // F24: a participant that cannot reach the barrier makes the barrier fail,
    // rather than fabricating success.
    #[test]
    fn test_synchronize_devices_fails_when_device_errored() {
        let config = PodConfig {
            num_devices: 4,
            ..Default::default()
        };
        let mut coordinator = PodCoordinator::new(config).expect("coordinator");
        coordinator.devices[1].state = DeviceState::Error("stuck".to_string());

        let result = coordinator.synchronize_devices("barrier-err".to_string());
        assert!(
            matches!(result, Err(CoordinationError::SynchronizationFailed { .. })),
            "errored participant must fail the barrier"
        );
        // No fabricated completion.
        assert_eq!(coordinator.completed_barrier_count(), 0);
        assert!(coordinator.get_status().active_barriers.is_empty());
    }

    // F24: an empty participant set is an honest failure, not a silent success.
    #[test]
    fn test_synchronize_devices_no_participants() {
        let config = PodConfig {
            num_devices: 0,
            ..Default::default()
        };
        let mut coordinator = PodCoordinator::new(config).expect("coordinator");
        let result = coordinator.synchronize_devices("barrier-empty".to_string());
        assert!(result.is_err());
        assert_eq!(coordinator.completed_barrier_count(), 0);
    }

    // F24: lifecycle methods leave real, observable state (enabled flags true,
    // devices registered).
    #[test]
    fn test_start_activates_monitoring_and_fault_detection() {
        let config = PodConfig {
            num_devices: 3,
            ..Default::default()
        };
        let mut coordinator = PodCoordinator::new(config).expect("coordinator");

        // Before start, nothing is active.
        assert!(!coordinator.is_monitoring_active());
        assert!(!coordinator.is_fault_detection_active());

        coordinator.start().expect("start");

        assert!(coordinator.is_monitoring_active());
        assert!(coordinator.is_fault_detection_active());
        assert_eq!(coordinator.performance_monitor.monitored_devices().len(), 3);
        assert_eq!(coordinator.fault_manager.monitored_devices().len(), 3);
        assert!(coordinator.performance_monitor.started_at().is_some());
        assert!(coordinator.fault_manager.started_at().is_some());
    }

    // F24 config-driven state: recovery strategies and alert thresholds are
    // populated from configuration rather than left empty.
    #[test]
    fn test_config_derived_state_is_populated() {
        let mut thresholds = HashMap::new();
        thresholds.insert("utilization".to_string(), 0.9);
        thresholds.insert("temperature".to_string(), 80.0);

        let config = PodConfig {
            num_devices: 2,
            fault_tolerance: FaultToleranceConfig {
                enable_checkpointing: true,
                checkpoint_interval: Duration::from_secs(300),
                max_failures: 3,
                recovery_timeout: Duration::from_secs(60),
                enable_redundancy: true,
            },
            monitoring: MonitoringConfig {
                collection_interval: Duration::from_secs(1),
                metrics_retention: Duration::from_secs(3600),
                enable_profiling: true,
                alert_thresholds: thresholds,
            },
            ..Default::default()
        };
        let coordinator = PodCoordinator::new(config).expect("coordinator");

        // Checkpointing + redundancy strategies both registered from config.
        assert!(!coordinator
            .fault_manager
            .recovery_strategies_for(FailureType::Hardware)
            .is_empty());

        // Typed thresholds mapped from string keys.
        assert_eq!(
            coordinator
                .performance_monitor
                .alert_threshold(MetricType::Utilization),
            Some(0.9)
        );
        assert_eq!(
            coordinator
                .performance_monitor
                .alert_threshold(MetricType::Temperature),
            Some(80.0)
        );
    }

    // F22: a NaN utilization metric must not panic and must not be selected as
    // the "least loaded" device.
    #[test]
    fn test_least_loaded_handles_nan_without_panic() {
        let mut load_balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);
        let devices = vec![
            make_device(0, DeviceState::Idle, f64::NAN),
            make_device(1, DeviceState::Idle, 0.3),
        ];
        let workload = WorkloadInfo {
            id: "nan_probe".to_string(),
            computation_type: ComputationType::MatrixMultiplication,
            estimated_completion: Duration::from_secs(1),
            resource_utilization: ResourceUtilization {
                compute: 0.1,
                memory: 0.1,
                bandwidth: 0.1,
            },
            priority: WorkloadPriority::Low,
        };

        // total_cmp gives a total order over floats, so this cannot panic.
        let selected = load_balancer
            .select_device(&devices, &workload)
            .expect("selection must not panic on NaN");
        // The finite-utilization device is the genuine minimum.
        assert_eq!(selected, TpuDeviceId(1));
    }
}
