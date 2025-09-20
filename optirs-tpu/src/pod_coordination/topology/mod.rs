// Topology management module for TPU pod coordination
//
// This module provides comprehensive topology management functionality including:
// - Device layout and placement optimization
// - Communication topology and network management
// - Power distribution and thermal management
// - Graph-based topology algorithms
// - Multi-objective optimization strategies
// - Centralized coordination and monitoring

// Module declarations
pub mod communication;
pub mod config;
pub mod core;
pub mod device_layout;
pub mod graph_management;
pub mod layout;
pub mod monitoring;
pub mod network;
pub mod optimization;
pub mod power;
pub mod power_management;
pub mod traffic;

// Re-exports
pub use config::*;
pub use core::{TopologyEventManager, TopologyManager, TopologyPerformanceMonitor};
pub use device_layout::{
    DeviceCapabilities, DeviceConfig, DeviceGroup, DeviceInfo, DeviceLayoutManager, DeviceNode,
    LayoutOptimizer, LayoutStatistics, LogicalLayout, PhysicalLayout, PlacementPolicy,
    ThermalStatus, ThermalZone,
};
pub use power_management::{
    EnergyHarvesting, PowerBudget, PowerConfiguration, PowerDistribution, PowerDistributionUnit,
    PowerEfficiencyManager, PowerManagementSystem, PowerMonitoring, PowerRequirements, PowerSupply,
    ThermalManagement,
};
pub use graph_management::{
    CentralityAlgorithms, ClusteringAlgorithms, CommunityDetection, GraphAlgorithms, GraphEdge,
    GraphMetrics, GraphNode, GraphOptimization, GraphPartitioning, GraphStatistics, GraphTraversal,
    TopologyGraph, TopologyGraphBuilder,
};
pub use communication::{
    CommunicationChannel, CommunicationInterface, CommunicationLink, CommunicationPath,
    CommunicationPattern, CommunicationQoS, CommunicationRouting, CommunicationTopology,
};
pub use optimization::{
    LayoutOptimizationAlgorithm, LayoutOptimizationObjective, LayoutOptimizerConfig,
    NetworkFlowOptimizer, OptimizationConstraint, OptimizationObjective, OptimizationResult,
    PowerAwareOptimizer, TopologyOptimizer,
};

// Type imports
use crate::pod_coordination::types::*;

// Statistics structure
#[derive(Debug, Clone, Default)]
pub struct TopologyStatistics {
    pub device_count: usize,
    pub connection_count: usize,
    pub average_latency: f64,
    pub total_bandwidth: f64,
}