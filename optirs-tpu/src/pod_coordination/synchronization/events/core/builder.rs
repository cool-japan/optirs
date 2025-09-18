// Builder patterns for event synchronization system

use std::time::Duration;
use super::types::*;

/// Event synchronization builder for comprehensive configuration
pub struct EventSyncBuilder {
    config: EventSynchronization,
}

impl EventSyncBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EventSynchronization::default(),
        }
    }

    /// Configure event delivery
    pub fn with_delivery(mut self, delivery: EventDelivery) -> Self {
        self.config.delivery = delivery;
        self
    }

    /// Configure event ordering
    pub fn with_ordering(mut self, ordering: EventOrdering) -> Self {
        self.config.ordering = ordering;
        self
    }

    /// Configure event filtering
    pub fn with_filtering(mut self, filtering: EventFiltering) -> Self {
        self.config.filtering = filtering;
        self
    }

    /// Configure event persistence
    pub fn with_persistence(mut self, persistence: EventPersistence) -> Self {
        self.config.persistence = persistence;
        self
    }

    /// Configure event compression
    pub fn with_compression(mut self, compression: EventCompression) -> Self {
        self.config.compression = compression;
        self
    }

    /// Configure event routing
    pub fn with_routing(mut self, routing: EventRouting) -> Self {
        self.config.routing = routing;
        self
    }

    /// Configure event queue management
    pub fn with_queue(mut self, queue: EventQueue) -> Self {
        self.config.queue = queue;
        self
    }

    /// Configure event handlers
    pub fn with_handlers(mut self, handlers: EventHandlers) -> Self {
        self.config.handlers = handlers;
        self
    }

    /// Configure global synchronization settings
    pub fn with_global_settings(mut self, global_settings: GlobalSyncSettings) -> Self {
        self.config.global_settings = global_settings;
        self
    }

    /// Configure integration settings
    pub fn with_integration(mut self, integration: IntegrationSettings) -> Self {
        self.config.integration = integration;
        self
    }

    /// Enable high-performance configuration
    pub fn with_high_performance(mut self) -> Self {
        // Configure for high performance
        self.config.global_settings.performance.enabled = true;
        self.config.global_settings.performance.targets.latency = Some(Duration::from_millis(1));
        self.config.global_settings.performance.targets.throughput = Some(100000.0);
        self
    }

    /// Enable reliable delivery
    pub fn with_reliable_delivery(mut self) -> Self {
        // Configure for reliable delivery - this would need to access delivery config
        self
    }

    /// Enable compression
    pub fn with_compression_enabled(mut self) -> Self {
        // Configure compression - this would need to access compression config
        self
    }

    /// Build the final configuration
    pub fn build(self) -> EventSynchronization {
        self.config
    }
}

impl Default for EventSyncBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Event synchronization presets for common configurations
pub struct EventSyncPresets;

impl EventSyncPresets {
    /// High-performance configuration for low-latency scenarios
    pub fn high_performance() -> EventSynchronization {
        EventSyncBuilder::new()
            .with_high_performance()
            .build()
    }

    /// Reliable configuration with strong durability guarantees
    pub fn reliable() -> EventSynchronization {
        EventSyncBuilder::new()
            .with_reliable_delivery()
            .build()
    }

    /// Memory-optimized configuration
    pub fn memory_optimized() -> EventSynchronization {
        EventSyncBuilder::new()
            .with_compression_enabled()
            .build()
    }

    /// Development configuration with enhanced debugging
    pub fn development() -> EventSynchronization {
        let mut config = EventSyncBuilder::new().build();
        config.global_settings.monitoring.enabled = true;
        config.global_settings.error_handling.reporting.enabled = true;
        config
    }
}

/// Global sync settings builder
pub struct GlobalSyncSettingsBuilder {
    config: GlobalSyncSettings,
}

impl GlobalSyncSettingsBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: GlobalSyncSettings::default(),
        }
    }

    /// Configure event ID format
    pub fn with_event_id_format(mut self, format: EventIdFormat) -> Self {
        self.config.event_id_format = format;
        self
    }

    /// Configure global timeouts
    pub fn with_timeouts(mut self, timeouts: GlobalTimeouts) -> Self {
        self.config.timeouts = timeouts;
        self
    }

    /// Configure cross-module coordination
    pub fn with_coordination(mut self, coordination: CrossModuleCoordination) -> Self {
        self.config.coordination = coordination;
        self
    }

    /// Configure global error handling
    pub fn with_error_handling(mut self, error_handling: GlobalErrorHandling) -> Self {
        self.config.error_handling = error_handling;
        self
    }

    /// Configure system monitoring
    pub fn with_monitoring(mut self, monitoring: GlobalMonitoring) -> Self {
        self.config.monitoring = monitoring;
        self
    }

    /// Configure performance tuning
    pub fn with_performance(mut self, performance: GlobalPerformance) -> Self {
        self.config.performance = performance;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> GlobalSyncSettings {
        self.config
    }
}

impl Default for GlobalSyncSettingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Event ID format builder
pub struct EventIdFormatBuilder {
    config: EventIdFormat,
}

impl EventIdFormatBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EventIdFormat::default(),
        }
    }

    /// Configure ID type
    pub fn with_id_type(mut self, id_type: EventIdType) -> Self {
        self.config.id_type = id_type;
        self
    }

    /// Configure generation strategy
    pub fn with_generation_strategy(mut self, strategy: IdGenerationStrategy) -> Self {
        self.config.generation_strategy = strategy;
        self
    }

    /// Configure validation
    pub fn with_validation(mut self, validation: IdValidation) -> Self {
        self.config.validation = validation;
        self
    }

    /// Configure uniqueness
    pub fn with_uniqueness(mut self, uniqueness: IdUniqueness) -> Self {
        self.config.uniqueness = uniqueness;
        self
    }

    /// Configure for UUID v4
    pub fn with_uuid_v4(mut self) -> Self {
        self.config.id_type = EventIdType::UUID { version: 4 };
        self.config.generation_strategy = IdGenerationStrategy::Local { seed: None };
        self
    }

    /// Configure for timestamp-based IDs
    pub fn with_timestamp(mut self, precision: TimestampPrecision) -> Self {
        self.config.id_type = EventIdType::Timestamp { precision };
        self.config.generation_strategy = IdGenerationStrategy::Local { seed: None };
        self
    }

    /// Configure for sequential IDs
    pub fn with_sequential(mut self, start: u64, increment: u64) -> Self {
        self.config.id_type = EventIdType::Sequential { start, increment };
        self.config.generation_strategy = IdGenerationStrategy::Centralized {
            generator_endpoint: "http://localhost:8080/id-generator".to_string(),
        };
        self
    }

    /// Build the final configuration
    pub fn build(self) -> EventIdFormat {
        self.config
    }
}

impl Default for EventIdFormatBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Global timeouts builder
pub struct GlobalTimeoutsBuilder {
    config: GlobalTimeouts,
}

impl GlobalTimeoutsBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: GlobalTimeouts::default(),
        }
    }

    /// Configure default operation timeout
    pub fn with_default_operation(mut self, timeout: Duration) -> Self {
        self.config.default_operation = timeout;
        self
    }

    /// Configure long-running operation timeout
    pub fn with_long_running_operation(mut self, timeout: Duration) -> Self {
        self.config.long_running_operation = timeout;
        self
    }

    /// Configure critical operation timeout
    pub fn with_critical_operation(mut self, timeout: Duration) -> Self {
        self.config.critical_operation = timeout;
        self
    }

    /// Add module-specific timeout
    pub fn with_module_timeout(mut self, module: String, timeout: Duration) -> Self {
        self.config.module_timeouts.insert(module, timeout);
        self
    }

    /// Configure timeout escalation
    pub fn with_escalation(mut self, escalation: TimeoutEscalation) -> Self {
        self.config.escalation = escalation;
        self
    }

    /// Enable escalation with defaults
    pub fn with_escalation_enabled(mut self) -> Self {
        self.config.escalation.enabled = true;
        self.config.escalation.factor = 2.0;
        self.config.escalation.max_timeout = Duration::from_secs(300); // 5 minutes
        self.config.escalation.steps = 3;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> GlobalTimeouts {
        self.config
    }
}

impl Default for GlobalTimeoutsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration settings builder
pub struct IntegrationSettingsBuilder {
    config: IntegrationSettings,
}

impl IntegrationSettingsBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: IntegrationSettings::default(),
        }
    }

    /// Add external system configuration
    pub fn with_external_system(mut self, name: String, config: ExternalSystemConfig) -> Self {
        self.config.external_systems.insert(name, config);
        self
    }

    /// Configure APIs
    pub fn with_apis(mut self, apis: ApiConfigurations) -> Self {
        self.config.apis = apis;
        self
    }

    /// Configure message queues
    pub fn with_message_queues(mut self, queues: MessageQueueIntegrations) -> Self {
        self.config.message_queues = queues;
        self
    }

    /// Configure databases
    pub fn with_databases(mut self, databases: DatabaseIntegrations) -> Self {
        self.config.databases = databases;
        self
    }

    /// Configure monitoring
    pub fn with_monitoring(mut self, monitoring: MonitoringIntegrations) -> Self {
        self.config.monitoring = monitoring;
        self
    }

    /// Enable REST API
    pub fn with_rest_api_enabled(mut self, base_url: String) -> Self {
        self.config.apis.rest.enabled = true;
        self.config.apis.rest.base_url = base_url;
        self
    }

    /// Enable GraphQL API
    pub fn with_graphql_api_enabled(mut self, endpoint: String) -> Self {
        self.config.apis.graphql.enabled = true;
        self.config.apis.graphql.endpoint = endpoint;
        self
    }

    /// Enable gRPC API
    pub fn with_grpc_api_enabled(mut self, address: String) -> Self {
        self.config.apis.grpc.enabled = true;
        self.config.apis.grpc.address = address;
        self
    }

    /// Enable WebSocket API
    pub fn with_websocket_api_enabled(mut self, endpoint: String) -> Self {
        self.config.apis.websocket.enabled = true;
        self.config.apis.websocket.endpoint = endpoint;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> IntegrationSettings {
        self.config
    }
}

impl Default for IntegrationSettingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}