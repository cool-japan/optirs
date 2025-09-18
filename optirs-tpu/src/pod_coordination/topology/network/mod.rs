// Network Topology and Communication Management
//
// This module provides comprehensive network topology management for TPU pod coordination
// including routing, quality of service, traffic management, and communication optimization.
//
// # Architecture
//
// The network management system is organized into focused sub-modules:
//
// - [`topology`] - Network topology types and configuration
// - [`routing`] - Routing protocols and path management
// - [`qos`] - Quality of Service settings and traffic classes
// - [`traffic`] - Traffic management and flow control
// - [`monitoring`] - Performance monitoring and analytics
//
// # Usage
//
// ```rust
// use crate::topology::network::{CommunicationTopologyManager, CommunicationTopologyConfig};
//
// let config = CommunicationTopologyConfig::default();
// let topology_manager = CommunicationTopologyManager::new(config)?;
// ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::super::tpu_backend::DeviceId;
use crate::error::{OptimError, Result};

// Import and re-export all sub-modules
pub mod topology;
pub mod routing;
pub mod qos;
pub mod traffic;
pub mod monitoring;

// Re-export core types and functionality
pub use topology::{
    NetworkTopology, NetworkTopologyType, TopologyConfiguration,
    NodeConfiguration, LinkConfiguration, TopologyBuilder,
};

pub use routing::{
    RoutingManager, RoutingProtocol, RoutingTable, RouteEntry,
    PathSelection, LoadBalancing, FailoverStrategy,
};

pub use qos::{
    NetworkQoSSettings, TrafficClass, TrafficPriority,
    BandwidthAllocation, QoSPolicy, ServiceLevel,
};

pub use traffic::{
    TrafficManager, TrafficManagementSettings, FlowControl,
    TrafficShaping, CongestionControl, TrafficAnalytics,
};

pub use monitoring::{
    TopologyPerformanceMonitor, TopologyMonitoringSettings,
    NetworkMetrics, PerformanceAnalytics, HealthMonitoring,
};

/// Communication topology manager for network coordination
#[derive(Debug)]
pub struct CommunicationTopologyManager {
    /// Communication topology configuration
    pub config: CommunicationTopologyConfig,
    /// Network topology
    pub network_topology: NetworkTopology,
    /// Routing manager
    pub routing_manager: RoutingManager,
    /// Traffic manager
    pub traffic_manager: TrafficManager,
    /// Performance monitor
    pub performance_monitor: TopologyPerformanceMonitor,
}

impl CommunicationTopologyManager {
    /// Create a new communication topology manager
    pub fn new(config: CommunicationTopologyConfig) -> Result<Self> {
        let network_topology = NetworkTopology::new(config.topology_type.clone())?;
        let routing_manager = RoutingManager::new(config.routing_protocol.clone())?;
        let traffic_manager = TrafficManager::new(config.traffic_management.clone())?;
        let performance_monitor = TopologyPerformanceMonitor::new(config.monitoring_settings.clone())?;

        Ok(Self {
            config,
            network_topology,
            routing_manager,
            traffic_manager,
            performance_monitor,
        })
    }

    /// Update the network topology
    pub fn update_topology(&mut self, new_config: TopologyConfiguration) -> Result<()> {
        self.network_topology.update(new_config)?;
        self.routing_manager.rebuild_routes(&self.network_topology)?;
        Ok(())
    }

    /// Get current network metrics
    pub fn get_metrics(&self) -> NetworkMetrics {
        self.performance_monitor.get_current_metrics()
    }

    /// Optimize network performance
    pub fn optimize_performance(&mut self) -> Result<()> {
        let metrics = self.get_metrics();

        // Optimize routing based on current performance
        self.routing_manager.optimize_routes(&metrics)?;

        // Adjust traffic management
        self.traffic_manager.optimize_traffic_flow(&metrics)?;

        Ok(())
    }
}

/// Configuration for communication topology
#[derive(Debug, Clone)]
pub struct CommunicationTopologyConfig {
    /// Network topology type
    pub topology_type: NetworkTopologyType,
    /// Routing protocol
    pub routing_protocol: RoutingProtocol,
    /// Quality of service settings
    pub qos_settings: NetworkQoSSettings,
    /// Traffic management settings
    pub traffic_management: TrafficManagementSettings,
    /// Performance monitoring settings
    pub monitoring_settings: TopologyMonitoringSettings,
}

impl Default for CommunicationTopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: NetworkTopologyType::default(),
            routing_protocol: RoutingProtocol::default(),
            qos_settings: NetworkQoSSettings::default(),
            traffic_management: TrafficManagementSettings::default(),
            monitoring_settings: TopologyMonitoringSettings::default(),
        }
    }
}

impl CommunicationTopologyConfig {
    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            topology_type: NetworkTopologyType::high_performance(),
            routing_protocol: RoutingProtocol::high_performance(),
            qos_settings: NetworkQoSSettings::high_performance(),
            traffic_management: TrafficManagementSettings::high_performance(),
            monitoring_settings: TopologyMonitoringSettings::high_performance(),
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            topology_type: NetworkTopologyType::low_latency(),
            routing_protocol: RoutingProtocol::low_latency(),
            qos_settings: NetworkQoSSettings::low_latency(),
            traffic_management: TrafficManagementSettings::low_latency(),
            monitoring_settings: TopologyMonitoringSettings::low_latency(),
        }
    }

    /// Create a high-bandwidth configuration
    pub fn high_bandwidth() -> Self {
        Self {
            topology_type: NetworkTopologyType::high_bandwidth(),
            routing_protocol: RoutingProtocol::high_bandwidth(),
            qos_settings: NetworkQoSSettings::high_bandwidth(),
            traffic_management: TrafficManagementSettings::high_bandwidth(),
            monitoring_settings: TopologyMonitoringSettings::high_bandwidth(),
        }
    }
}

/// Network topology configuration builder
pub struct NetworkConfigurationBuilder {
    config: CommunicationTopologyConfig,
}

impl NetworkConfigurationBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: CommunicationTopologyConfig::default(),
        }
    }

    /// Set the network topology type
    pub fn with_topology_type(mut self, topology_type: NetworkTopologyType) -> Self {
        self.config.topology_type = topology_type;
        self
    }

    /// Set the routing protocol
    pub fn with_routing_protocol(mut self, routing_protocol: RoutingProtocol) -> Self {
        self.config.routing_protocol = routing_protocol;
        self
    }

    /// Set QoS settings
    pub fn with_qos_settings(mut self, qos_settings: NetworkQoSSettings) -> Self {
        self.config.qos_settings = qos_settings;
        self
    }

    /// Set traffic management settings
    pub fn with_traffic_management(mut self, traffic_management: TrafficManagementSettings) -> Self {
        self.config.traffic_management = traffic_management;
        self
    }

    /// Set monitoring settings
    pub fn with_monitoring_settings(mut self, monitoring_settings: TopologyMonitoringSettings) -> Self {
        self.config.monitoring_settings = monitoring_settings;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> CommunicationTopologyConfig {
        self.config
    }
}

impl Default for NetworkConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Network topology presets for common configurations
pub struct NetworkTopologyPresets;

impl NetworkTopologyPresets {
    /// High-performance network configuration
    pub fn high_performance() -> CommunicationTopologyConfig {
        NetworkConfigurationBuilder::new()
            .with_topology_type(NetworkTopologyType::high_performance())
            .with_routing_protocol(RoutingProtocol::high_performance())
            .with_qos_settings(NetworkQoSSettings::high_performance())
            .with_traffic_management(TrafficManagementSettings::high_performance())
            .with_monitoring_settings(TopologyMonitoringSettings::high_performance())
            .build()
    }

    /// Low-latency network configuration
    pub fn low_latency() -> CommunicationTopologyConfig {
        NetworkConfigurationBuilder::new()
            .with_topology_type(NetworkTopologyType::low_latency())
            .with_routing_protocol(RoutingProtocol::low_latency())
            .with_qos_settings(NetworkQoSSettings::low_latency())
            .with_traffic_management(TrafficManagementSettings::low_latency())
            .with_monitoring_settings(TopologyMonitoringSettings::low_latency())
            .build()
    }

    /// High-bandwidth network configuration
    pub fn high_bandwidth() -> CommunicationTopologyConfig {
        NetworkConfigurationBuilder::new()
            .with_topology_type(NetworkTopologyType::high_bandwidth())
            .with_routing_protocol(RoutingProtocol::high_bandwidth())
            .with_qos_settings(NetworkQoSSettings::high_bandwidth())
            .with_traffic_management(TrafficManagementSettings::high_bandwidth())
            .with_monitoring_settings(TopologyMonitoringSettings::high_bandwidth())
            .build()
    }

    /// Balanced network configuration
    pub fn balanced() -> CommunicationTopologyConfig {
        CommunicationTopologyConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_manager_creation() {
        let config = CommunicationTopologyConfig::default();
        let manager = CommunicationTopologyManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_configuration_builder() {
        let config = NetworkConfigurationBuilder::new()
            .with_topology_type(NetworkTopologyType::default())
            .with_routing_protocol(RoutingProtocol::default())
            .build();

        assert!(matches!(config.topology_type, NetworkTopologyType::Flat));
    }

    #[test]
    fn test_high_performance_preset() {
        let config = NetworkTopologyPresets::high_performance();
        // Verify high-performance settings are applied
        assert!(config.monitoring_settings.enabled);
    }

    #[test]
    fn test_low_latency_preset() {
        let config = NetworkTopologyPresets::low_latency();
        // Verify low-latency optimizations are applied
        assert!(config.monitoring_settings.enabled);
    }
}