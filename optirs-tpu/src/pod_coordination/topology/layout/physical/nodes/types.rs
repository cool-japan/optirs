// Core Node Types Module
//
// This module defines the fundamental node types, status enums, and the main
// NodeInfo structure for TPU pod coordination systems.

use std::time::Instant;
use super::super::super::super::super::tpu_backend::DeviceId;
use super::super::positioning::{NodeId};

// Re-exports from other modules
use super::processing::ProcessingCapabilities;
use super::interfaces::IOCapabilities;
use super::memory::MemoryCapabilities;
use super::storage::StorageCapabilities;
use super::networking::NetworkCapabilities;
use super::reliability::ReliabilityMetrics;
use super::physical::NodePhysicalProperties;
use super::configuration::NodeConfiguration;
use super::metrics::NodeMetrics;

/// Node information in the physical layout
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node identifier
    pub node_id: NodeId,
    /// Associated device
    pub device_id: DeviceId,
    /// Node type
    pub node_type: NodeType,
    /// Current status
    pub status: NodeStatus,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Physical properties
    pub physical_properties: NodePhysicalProperties,
    /// Node configuration
    pub configuration: NodeConfiguration,
    /// Node metrics
    pub metrics: NodeMetrics,
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for NodeInfo {
    fn default() -> Self {
        Self {
            node_id: 0,
            device_id: DeviceId::default(),
            node_type: NodeType::default(),
            status: NodeStatus::default(),
            capabilities: NodeCapabilities::default(),
            physical_properties: NodePhysicalProperties::default(),
            configuration: NodeConfiguration::default(),
            metrics: NodeMetrics::default(),
            last_update: Instant::now(),
        }
    }
}

/// Node capabilities
#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    /// Processing capabilities
    pub processing: ProcessingCapabilities,
    /// I/O capabilities
    pub io: IOCapabilities,
    /// Memory capabilities
    pub memory: MemoryCapabilities,
    /// Storage capabilities
    pub storage: StorageCapabilities,
    /// Network capabilities
    pub network: NetworkCapabilities,
    /// Reliability metrics
    pub reliability: ReliabilityMetrics,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            processing: ProcessingCapabilities::default(),
            io: IOCapabilities::default(),
            memory: MemoryCapabilities::default(),
            storage: StorageCapabilities::default(),
            network: NetworkCapabilities::default(),
            reliability: ReliabilityMetrics::default(),
        }
    }
}

/// Types of nodes in the topology
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// Compute node
    Compute,
    /// Storage node
    Storage,
    /// Network node
    Network,
    /// Control node
    Control,
    /// Hybrid node
    Hybrid { capabilities: Vec<String> },
    /// Custom node type
    Custom { node_type: String },
}

impl Default for NodeType {
    fn default() -> Self {
        NodeType::Compute
    }
}

/// Node status in the topology
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Node is online and operational
    Online,
    /// Node is offline
    Offline,
    /// Node is in initialization phase
    Initializing,
    /// Node is shutting down
    ShuttingDown,
    /// Node is in maintenance mode
    Maintenance,
    /// Node has encountered an error
    Error { error_code: u32, message: String },
    /// Node is in degraded mode
    Degraded { reason: String },
    /// Unknown status
    Unknown,
}

impl Default for NodeStatus {
    fn default() -> Self {
        NodeStatus::Unknown
    }
}

/// Node role in the cluster
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    /// Primary node
    Primary,
    /// Secondary node
    Secondary,
    /// Worker node
    Worker,
    /// Coordinator node
    Coordinator,
    /// Observer node
    Observer,
    /// Custom role
    Custom { role: String },
}

impl Default for NodeRole {
    fn default() -> Self {
        NodeRole::Worker
    }
}

/// Node priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum NodePriority {
    /// Critical priority
    Critical,
    /// High priority
    High,
    /// Normal priority
    Normal,
    /// Low priority
    Low,
    /// Background priority
    Background,
}

impl Default for NodePriority {
    fn default() -> Self {
        NodePriority::Normal
    }
}

/// Node health status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeHealthStatus {
    /// Healthy node
    Healthy,
    /// Warning status
    Warning { message: String },
    /// Critical status
    Critical { message: String },
    /// Failed status
    Failed { error: String },
    /// Unknown health status
    Unknown,
}

impl Default for NodeHealthStatus {
    fn default() -> Self {
        NodeHealthStatus::Unknown
    }
}

/// Node lifecycle state
#[derive(Debug, Clone, PartialEq)]
pub enum NodeLifecycleState {
    /// Node is being provisioned
    Provisioning,
    /// Node is ready for use
    Ready,
    /// Node is active and processing
    Active,
    /// Node is being updated
    Updating,
    /// Node is being migrated
    Migrating,
    /// Node is being decommissioned
    Decommissioning,
    /// Node has been terminated
    Terminated,
}

impl Default for NodeLifecycleState {
    fn default() -> Self {
        NodeLifecycleState::Provisioning
    }
}

/// Node availability status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeAvailability {
    /// Node is available
    Available,
    /// Node is busy
    Busy,
    /// Node is reserved
    Reserved { by: String, until: Instant },
    /// Node is draining
    Draining,
    /// Node is unavailable
    Unavailable { reason: String },
}

impl Default for NodeAvailability {
    fn default() -> Self {
        NodeAvailability::Available
    }
}

/// Node cluster membership
#[derive(Debug, Clone)]
pub struct NodeMembership {
    /// Cluster ID
    pub cluster_id: String,
    /// Pod ID
    pub pod_id: String,
    /// Rack ID
    pub rack_id: String,
    /// Zone ID
    pub zone_id: String,
    /// Region ID
    pub region_id: String,
}

impl Default for NodeMembership {
    fn default() -> Self {
        Self {
            cluster_id: "default".to_string(),
            pod_id: "pod-0".to_string(),
            rack_id: "rack-0".to_string(),
            zone_id: "zone-a".to_string(),
            region_id: "region-1".to_string(),
        }
    }
}

/// Node tags for classification and filtering
#[derive(Debug, Clone)]
pub struct NodeTags {
    /// Environment tags
    pub environment: Vec<String>,
    /// Application tags
    pub application: Vec<String>,
    /// Owner tags
    pub owner: Vec<String>,
    /// Cost center tags
    pub cost_center: Vec<String>,
    /// Custom tags
    pub custom: Vec<(String, String)>,
}

impl Default for NodeTags {
    fn default() -> Self {
        Self {
            environment: Vec::new(),
            application: Vec::new(),
            owner: Vec::new(),
            cost_center: Vec::new(),
            custom: Vec::new(),
        }
    }
}

/// Node maintenance window
#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    /// Start time
    pub start: Instant,
    /// Duration
    pub duration: std::time::Duration,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Reason for maintenance
    pub reason: String,
    /// Scheduled by
    pub scheduled_by: String,
}

/// Maintenance types
#[derive(Debug, Clone, PartialEq)]
pub enum MaintenanceType {
    /// Software update
    SoftwareUpdate,
    /// Hardware maintenance
    HardwareMaintenance,
    /// Security patch
    SecurityPatch,
    /// Performance tuning
    PerformanceTuning,
    /// Preventive maintenance
    Preventive,
    /// Emergency maintenance
    Emergency,
    /// Custom maintenance
    Custom { maintenance_type: String },
}

impl Default for MaintenanceType {
    fn default() -> Self {
        MaintenanceType::Preventive
    }
}

/// Node resource allocation
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU allocation (percentage)
    pub cpu_allocation: f64,
    /// Memory allocation (bytes)
    pub memory_allocation: u64,
    /// Storage allocation (bytes)
    pub storage_allocation: u64,
    /// Network bandwidth allocation (bytes/sec)
    pub network_allocation: u64,
    /// GPU allocation (percentage)
    pub gpu_allocation: f64,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_allocation: 100.0,
            memory_allocation: u64::MAX,
            storage_allocation: u64::MAX,
            network_allocation: u64::MAX,
            gpu_allocation: 100.0,
        }
    }
}

/// Node resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum CPU usage (percentage)
    pub max_cpu: f64,
    /// Maximum memory usage (bytes)
    pub max_memory: u64,
    /// Maximum storage usage (bytes)
    pub max_storage: u64,
    /// Maximum network bandwidth (bytes/sec)
    pub max_network: u64,
    /// Maximum GPU usage (percentage)
    pub max_gpu: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu: 100.0,
            max_memory: u64::MAX,
            max_storage: u64::MAX,
            max_network: u64::MAX,
            max_gpu: 100.0,
        }
    }
}