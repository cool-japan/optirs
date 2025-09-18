// Event Routing Strategies, Load Balancing, and Failover
//
// This module provides comprehensive event routing capabilities for TPU synchronization
// including intelligent routing strategies, load balancing algorithms, failover mechanisms,
// health monitoring, traffic management, and routing analytics.
//
// # Architecture
//
// The routing system is organized into focused sub-modules:
//
// - [`core`] - Main routing engine, tables, and strategies
// - [`algorithms`] - Load balancing, failover, and topology-aware algorithms
// - [`optimization`] - Path optimization, caching, and adaptive routing
// - [`monitoring`] - Metrics collection and diagnostics
//
// # Usage
//
// ```rust
// use crate::events::routing::{EventRouting, EventRoutingBuilder};
//
// let routing = EventRoutingBuilder::new()
//     .with_load_balancing()
//     .with_failover()
//     .with_health_monitoring()
//     .build();
// ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Core routing modules
pub mod core;
pub mod algorithms;
pub mod optimization;
pub mod monitoring;

// Re-export core types
pub use core::{
    RoutingError, RoutingResult, Router, RouterConfig, RouterSettings,
    RoutingTables, StaticRoute, DynamicRoute, RouteMetadata,
    RoutingStrategy, RoundRobinConfig, WeightedRoundRobinConfig,
    LeastConnectionsConfig, HashBasedConfig, GeographicConfig,
    PerformanceBasedConfig, ContentBasedConfig, RandomConfig,
    PriorityBasedConfig, AdaptiveConfig, CustomRoutingConfig,
};

// Re-export algorithm types
pub use algorithms::{
    LoadBalancing, LoadBalancingAlgorithm, Failover, FailoverStrategy,
    TopologyAwareRouting, FailureDetection, FailoverRecovery,
    HealthChecking, NetworkAwareness, QoSAwareness,
};

// Re-export optimization types
pub use optimization::{
    PathOptimization, OptimizationCriteria, RoutingCache, CacheStrategy,
    AdaptiveRouting as AdaptiveOptimization, LearningAlgorithm,
};

// Re-export monitoring types
pub use monitoring::{
    RoutingAnalytics, RoutingMetrics, TrafficManagement, HealthMonitoring,
    PerformanceMetrics, QualityMetrics, UsageMetrics, TrafficShaping,
    LoadDistribution, PriorityHandling, FlowControl,
};

/// Event routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRouting {
    /// Routing strategies
    pub routing_strategies: RoutingStrategies,
    /// Load balancing configuration
    pub load_balancing: LoadBalancing,
    /// Failover configuration
    pub failover: Failover,
    /// Traffic management
    pub traffic_management: TrafficManagement,
    /// Health monitoring
    pub health_monitoring: HealthMonitoring,
    /// Routing analytics
    pub analytics: RoutingAnalytics,
}

impl Default for EventRouting {
    fn default() -> Self {
        Self {
            routing_strategies: RoutingStrategies::default(),
            load_balancing: LoadBalancing::default(),
            failover: Failover::default(),
            traffic_management: TrafficManagement::default(),
            health_monitoring: HealthMonitoring::default(),
            analytics: RoutingAnalytics::default(),
        }
    }
}

/// Routing strategies configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStrategies {
    /// Available strategies
    pub strategies: Vec<RoutingStrategy>,
    /// Default strategy
    pub default_strategy: RoutingStrategy,
    /// Strategy selection
    pub strategy_selection: StrategySelection,
    /// Routing tables
    pub routing_tables: RoutingTables,
}

impl Default for RoutingStrategies {
    fn default() -> Self {
        Self {
            strategies: vec![
                RoutingStrategy::RoundRobin(RoundRobinConfig::default()),
                RoutingStrategy::WeightedRoundRobin(WeightedRoundRobinConfig::default()),
                RoutingStrategy::LeastConnections(LeastConnectionsConfig::default()),
                RoutingStrategy::HashBased(HashBasedConfig::default()),
            ],
            default_strategy: RoutingStrategy::RoundRobin(RoundRobinConfig::default()),
            strategy_selection: StrategySelection::default(),
            routing_tables: RoutingTables::default(),
        }
    }
}

/// Strategy selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelection {
    /// Selection algorithm
    pub algorithm: SelectionAlgorithm,
    /// Selection criteria
    pub criteria: SelectionCriteria,
    /// Fallback strategy
    pub fallback: String,
}

impl Default for StrategySelection {
    fn default() -> Self {
        Self {
            algorithm: SelectionAlgorithm::Performance,
            criteria: SelectionCriteria::default(),
            fallback: "round_robin".to_string(),
        }
    }
}

/// Strategy selection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Performance-based selection
    Performance,
    /// Load-based selection
    Load,
    /// Random selection
    Random,
    /// Round-robin selection
    RoundRobin,
    /// Custom selection
    Custom(String),
}

/// Strategy selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Performance weight
    pub performance_weight: f64,
    /// Load weight
    pub load_weight: f64,
    /// Latency weight
    pub latency_weight: f64,
    /// Availability weight
    pub availability_weight: f64,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            performance_weight: 0.4,
            load_weight: 0.3,
            latency_weight: 0.2,
            availability_weight: 0.1,
        }
    }
}

/// Event routing builder for easy configuration
pub struct EventRoutingBuilder {
    config: EventRouting,
}

impl EventRoutingBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EventRouting::default(),
        }
    }

    /// Configure routing strategies
    pub fn with_strategies(mut self, strategies: RoutingStrategies) -> Self {
        self.config.routing_strategies = strategies;
        self
    }

    /// Configure load balancing
    pub fn with_load_balancing(mut self) -> Self {
        self.config.load_balancing.algorithms = vec![
            LoadBalancingAlgorithm::RoundRobin,
            LoadBalancingAlgorithm::WeightedRoundRobin,
            LoadBalancingAlgorithm::LeastConnections,
        ];
        self
    }

    /// Configure failover
    pub fn with_failover(mut self) -> Self {
        self.config.failover.strategies = vec![
            FailoverStrategy::Automatic,
            FailoverStrategy::Manual,
        ];
        self
    }

    /// Configure health monitoring
    pub fn with_health_monitoring(mut self) -> Self {
        self.config.health_monitoring.enabled = true;
        self
    }

    /// Configure analytics
    pub fn with_analytics(mut self) -> Self {
        self.config.analytics.enabled = true;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> EventRouting {
        self.config
    }
}

impl Default for EventRoutingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Routing presets for common configurations
pub struct RoutingPresets;

impl RoutingPresets {
    /// High-performance routing configuration
    pub fn high_performance() -> EventRouting {
        EventRoutingBuilder::new()
            .with_load_balancing()
            .with_health_monitoring()
            .with_analytics()
            .build()
    }

    /// High-availability routing configuration
    pub fn high_availability() -> EventRouting {
        EventRoutingBuilder::new()
            .with_failover()
            .with_health_monitoring()
            .with_analytics()
            .build()
    }

    /// Balanced routing configuration
    pub fn balanced() -> EventRouting {
        EventRoutingBuilder::new()
            .with_load_balancing()
            .with_failover()
            .with_health_monitoring()
            .build()
    }
}