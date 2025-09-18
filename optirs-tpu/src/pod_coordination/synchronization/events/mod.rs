// Event Synchronization System
//
// This module provides a comprehensive event synchronization system for TPU pod coordination.
// It includes delivery guarantees, ordering mechanisms, filtering capabilities, persistence layers,
// compression algorithms, routing strategies, queue management, and handler coordination.
//
// # Architecture Overview
//
// The event synchronization system is composed of nine focused modules:
//
// - **Delivery**: Event delivery guarantees, acknowledgments, retries, and timeouts
// - **Ordering**: Event ordering, sequence management, gap detection, and buffering
// - **Filtering**: Event filtering rules, conditions, and optimization
// - **Persistence**: Event persistence, storage backends, and retention policies
// - **Compression**: Event compression algorithms and adaptive compression
// - **Routing**: Event routing strategies, load balancing, and failover
// - **Queue**: Event queue management, statistics, and overflow handling
// - **Handlers**: Event handler management, capabilities, and metrics
//
// # Usage Examples
//
// ## Basic Event Synchronization Setup
//
// ```rust
// use crate::events::{EventSynchronization, EventSyncBuilder};
//
// let sync = EventSyncBuilder::new()
//     .with_high_performance()
//     .with_reliable_delivery()
//     .with_compression_enabled()
//     .build();
// ```
//
// ## Advanced Configuration
//
// ```rust
// use crate::events::{EventSynchronization, delivery::*, ordering::*, filtering::*};
//
// let delivery_config = EventDeliveryBuilder::new()
//     .with_at_least_once_delivery()
//     .with_acknowledgment_timeout(Duration::from_secs(30))
//     .with_retry_policy(RetryPolicy::exponential_backoff(5))
//     .build();
//
// let ordering_config = EventOrderingBuilder::new()
//     .with_total_ordering()
//     .with_sequence_validation()
//     .with_gap_detection()
//     .build();
//
// let filtering_config = EventFilteringBuilder::new()
//     .with_content_based_filtering()
//     .with_performance_optimization()
//     .build();
//
// let sync = EventSyncBuilder::new()
//     .with_delivery(delivery_config)
//     .with_ordering(ordering_config)
//     .with_filtering(filtering_config)
//     .build();
// ```
//
// ## Preset Configurations
//
// ```rust
// use crate::events::EventSyncPresets;
//
// // High-performance configuration for low-latency scenarios
// let high_perf = EventSyncPresets::high_performance();
//
// // Reliable configuration with strong durability guarantees
// let reliable = EventSyncPresets::reliable();
//
// // Memory-optimized configuration
// let memory_opt = EventSyncPresets::memory_optimized();
//
// // Development configuration with enhanced debugging
// let dev = EventSyncPresets::development();
// ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Core module with types, configuration, and builders
pub mod core;

// Utilities module
pub mod utilities;

// Re-export all event synchronization modules
pub mod delivery;
pub mod ordering;
pub mod filtering;
pub mod persistence;
pub mod compression;
pub mod routing;
pub mod queue;
pub mod handlers;

// Re-export core types and builders for convenient access
pub use core::{
    types::*,
    builder::*,
    config::*,
};

// Re-export utilities
pub use utilities::helpers::EventSyncUtils;

// Re-export key types from each module for convenient access
pub use delivery::{
    EventDelivery, EventDeliveryBuilder, DeliveryGuarantees, AcknowledgmentConfig,
    RetryPolicies, TimeoutConfig, DeliveryMetrics, DeliveryPresets,
};

pub use ordering::{
    EventOrdering, EventOrderingBuilder, OrderingStrategies, SequenceManagement,
    GapDetection, OrderingBuffering, OrderingMetrics, OrderingPresets,
};

pub use filtering::{
    EventFiltering, EventFilteringBuilder, FilterRulesConfig, FilterOptimization,
    FilterPerformanceMonitoring, FilterComposition, FilteringPresets,
};

pub use persistence::{
    EventPersistence, EventPersistenceBuilder, StorageBackendConfig, RetentionPolicies,
    BackupRecoveryConfig, PerformanceOptimization, PersistencePresets,
};

pub use compression::{
    EventCompression, EventCompressionBuilder, CompressionAlgorithms, AdaptiveCompression,
    StreamingCompression, CompressionAnalytics, CompressionPresets,
};

pub use routing::{
    EventRouting, EventRoutingBuilder, RoutingStrategies, LoadBalancing,
    Failover, TrafficManagement, RoutingAnalytics, RoutingPresets,
};

pub use queue::{
    EventQueue, EventQueueBuilder, QueueManagement, QueueStatistics,
    OverflowHandling, QueuePerformance, QueuePresets,
};

pub use handlers::{
    EventHandlers, EventHandlersBuilder, HandlerManagement, HandlerCapabilities,
    HandlerRouting, HandlerMetrics, EventHandlerPresets,
};