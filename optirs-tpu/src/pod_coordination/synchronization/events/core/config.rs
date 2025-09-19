// Configuration defaults and utilities for event synchronization system

use super::types::*;
use std::collections::HashMap;
use std::time::Duration;

// Default implementations for core types

impl Default for EventSynchronization {
    fn default() -> Self {
        Self {
            delivery: EventDelivery::default(),
            ordering: EventOrdering::default(),
            filtering: EventFiltering::default(),
            persistence: EventPersistence::default(),
            compression: EventCompression::default(),
            routing: EventRouting::default(),
            queue: EventQueue::default(),
            handlers: EventHandlers::default(),
            global_settings: GlobalSyncSettings::default(),
            integration: IntegrationSettings::default(),
        }
    }
}

impl Default for GlobalSyncSettings {
    fn default() -> Self {
        Self {
            event_id_format: EventIdFormat::default(),
            timeouts: GlobalTimeouts::default(),
            coordination: CrossModuleCoordination::default(),
            error_handling: GlobalErrorHandling::default(),
            monitoring: GlobalMonitoring::default(),
            performance: GlobalPerformance::default(),
        }
    }
}

impl Default for EventIdFormat {
    fn default() -> Self {
        Self {
            id_type: EventIdType::UUID { version: 4 },
            generation_strategy: IdGenerationStrategy::Local { seed: None },
            validation: IdValidation::default(),
            uniqueness: IdUniqueness::default(),
        }
    }
}

impl Default for IdValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            performance: IdValidationPerformance::default(),
            custom_validators: vec![],
        }
    }
}

impl Default for IdValidationPerformance {
    fn default() -> Self {
        Self {
            caching: true,
            cache_size: 10000,
            parallel_validation: true,
            timeout: Duration::from_millis(100),
        }
    }
}

impl Default for IdUniqueness {
    fn default() -> Self {
        Self {
            scope: UniquenessScope::Global,
            conflict_detection: ConflictDetection::Immediate,
            conflict_resolution: ConflictResolution::Reject,
            monitoring: UniquenessMonitoring::default(),
        }
    }
}

impl Default for UniquenessMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            alert_thresholds: HashMap::new(),
            interval: Duration::from_secs(60),
            reporting: UniquenessReporting::default(),
        }
    }
}

impl Default for UniquenessReporting {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
            destinations: vec!["logs".to_string()],
            frequency: Duration::from_secs(300),
            include_statistics: true,
        }
    }
}

impl Default for GlobalTimeouts {
    fn default() -> Self {
        Self {
            default_operation: Duration::from_secs(30),
            long_running_operation: Duration::from_secs(300),
            critical_operation: Duration::from_secs(10),
            module_timeouts: HashMap::new(),
            escalation: TimeoutEscalation::default(),
        }
    }
}

impl Default for TimeoutEscalation {
    fn default() -> Self {
        Self {
            enabled: false,
            factor: 2.0,
            max_timeout: Duration::from_secs(600),
            steps: 3,
            actions: vec![],
        }
    }
}

impl Default for CrossModuleCoordination {
    fn default() -> Self {
        Self {
            protocol: CoordinationProtocol::Centralized {
                coordinator: "main".to_string(),
            },
            dependencies: DependencyManagement::default(),
            resource_sharing: ResourceSharing::default(),
            state_sync: StateSynchronization::default(),
            event_propagation: EventPropagation::default(),
        }
    }
}

impl Default for DependencyManagement {
    fn default() -> Self {
        Self {
            resolution: DependencyResolution::Lazy,
            circular_handling: CircularDependencyHandling::Reject,
            monitoring: DependencyMonitoring::default(),
            injection: DependencyInjection::default(),
        }
    }
}

impl Default for DependencyMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            health_checks: true,
            performance: true,
            availability: true,
        }
    }
}

impl Default for DependencyInjection {
    fn default() -> Self {
        Self {
            strategy: InjectionStrategy::Constructor,
            scope_management: ScopeManagement::Singleton,
            lifecycle: LifecycleManagement::default(),
            configuration_injection: true,
        }
    }
}

impl Default for LifecycleManagement {
    fn default() -> Self {
        Self {
            initialization_order: vec![],
            shutdown_order: vec![],
            hooks: LifecycleHooks::default(),
            cleanup: ResourceCleanup::default(),
        }
    }
}

impl Default for LifecycleHooks {
    fn default() -> Self {
        Self {
            pre_init: vec![],
            post_init: vec![],
            pre_shutdown: vec![],
            post_shutdown: vec![],
        }
    }
}

impl Default for ResourceCleanup {
    fn default() -> Self {
        Self {
            automatic: true,
            timeout: Duration::from_secs(30),
            order: vec![],
            verification: true,
        }
    }
}

impl Default for ResourceSharing {
    fn default() -> Self {
        Self {
            resources: HashMap::new(),
            access_control: ResourceAccessControl::default(),
            pooling: ResourcePooling::default(),
            monitoring: ResourceMonitoring::default(),
        }
    }
}

impl Default for ResourceAccessControl {
    fn default() -> Self {
        Self {
            authentication: false,
            authorization: AuthorizationPolicies::default(),
            logging: AccessLogging::default(),
            rate_limiting: RateLimiting::default(),
        }
    }
}

impl Default for AuthorizationPolicies {
    fn default() -> Self {
        Self {
            rbac: false,
            abac: false,
            rules: vec![],
            evaluation: PolicyEvaluation::FailClosed,
        }
    }
}

impl Default for AccessLogging {
    fn default() -> Self {
        Self {
            enabled: false,
            level: "info".to_string(),
            format: "json".to_string(),
            destinations: vec!["stdout".to_string()],
            retention: Duration::from_days(30),
        }
    }
}

impl Default for RateLimiting {
    fn default() -> Self {
        Self {
            enabled: false,
            limits: HashMap::new(),
            algorithm: RateLimitingAlgorithm::TokenBucket {
                capacity: 1000,
                refill_rate: 100.0,
            },
            violation_handling: ViolationHandling::Drop,
        }
    }
}

impl Default for ResourcePooling {
    fn default() -> Self {
        Self {
            enabled: false,
            pools: HashMap::new(),
            monitoring: PoolMonitoring::default(),
            optimization: PoolOptimization::default(),
        }
    }
}

impl Default for PoolMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: PoolMetrics::default(),
            health_monitoring: true,
            performance_monitoring: true,
        }
    }
}

impl Default for PoolMetrics {
    fn default() -> Self {
        Self {
            size: true,
            usage: true,
            performance: true,
            errors: true,
        }
    }
}

impl Default for PoolOptimization {
    fn default() -> Self {
        Self {
            enabled: false,
            strategies: vec![OptimizationStrategy::PerformanceOptimization],
            interval: Duration::from_secs(300),
            targets: PerformanceTargets::default(),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            latency: Some(Duration::from_millis(100)),
            throughput: Some(1000.0),
            utilization: Some(0.8),
            availability: Some(0.99),
        }
    }
}

impl Default for ResourceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            metrics: ResourceMetrics::default(),
            alerting: ResourceAlerting::default(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            usage: true,
            performance: true,
            availability: true,
            errors: true,
        }
    }
}

impl Default for ResourceAlerting {
    fn default() -> Self {
        Self {
            enabled: false,
            thresholds: HashMap::new(),
            destinations: vec!["logs".to_string()],
            frequency: AlertFrequency::Immediate,
        }
    }
}

impl Default for StateSynchronization {
    fn default() -> Self {
        Self {
            strategy: StateSyncStrategy::EventDriven,
            consistency: ConsistencyModel::Eventual,
            conflict_resolution: StateSyncConflictResolution::LastWriterWins,
            versioning: StateVersioning::default(),
        }
    }
}

impl Default for StateVersioning {
    fn default() -> Self {
        Self {
            enabled: true,
            format: VersionFormat::Timestamp,
            history: VersionHistory::default(),
            comparison: VersionComparison::Timestamp,
        }
    }
}

impl Default for VersionHistory {
    fn default() -> Self {
        Self {
            enabled: true,
            max_versions: Some(100),
            retention_period: Some(Duration::from_days(7)),
            compression: true,
        }
    }
}

impl Default for EventPropagation {
    fn default() -> Self {
        Self {
            strategy: PropagationStrategy::Selective {
                targets: vec!["all".to_string()],
            },
            filtering: PropagationFiltering::default(),
            transformation: EventTransformation::default(),
            monitoring: PropagationMonitoring::default(),
        }
    }
}

impl Default for PropagationFiltering {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            default_action: FilterAction::Allow,
            optimization: true,
        }
    }
}

impl Default for EventTransformation {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            default_transformer: None,
            optimization: true,
        }
    }
}

impl Default for PropagationMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: PropagationMetrics::default(),
            performance: true,
            error_tracking: true,
        }
    }
}

impl Default for PropagationMetrics {
    fn default() -> Self {
        Self {
            event_counts: true,
            latency: true,
            success_rate: true,
            error_rate: true,
        }
    }
}

impl Default for GlobalErrorHandling {
    fn default() -> Self {
        Self {
            strategy: ErrorHandlingStrategy::GracefulDegradation,
            classification: ErrorClassification::default(),
            recovery: ErrorRecovery::default(),
            reporting: ErrorReporting::default(),
        }
    }
}

impl Default for ErrorClassification {
    fn default() -> Self {
        Self {
            enabled: true,
            categories: HashMap::new(),
            rules: vec![],
            default_category: "general".to_string(),
        }
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: HashMap::new(),
            retry: RetryConfiguration::default(),
            fallback: FallbackConfiguration::default(),
        }
    }
}

impl Default for RetryConfiguration {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            delay: Duration::from_millis(100),
            backoff: BackoffStrategy::Exponential { factor: 2.0 },
            conditions: vec!["transient_error".to_string()],
        }
    }
}

impl Default for FallbackConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            targets: vec![],
            timeout: Duration::from_secs(5),
            quality_threshold: 0.8,
        }
    }
}

impl Default for ErrorReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            destinations: vec![],
            format: ReportFormat::Json,
            aggregation: ReportAggregation::default(),
        }
    }
}

impl Default for ReportAggregation {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(60),
            rules: vec![],
            max_size: 1000,
        }
    }
}

impl Default for GlobalMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: GlobalMetrics::default(),
            health: GlobalHealthMonitoring::default(),
            performance: GlobalPerformanceMonitoring::default(),
            alerting: GlobalAlerting::default(),
        }
    }
}

impl Default for GlobalMetrics {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            categories: vec![
                "performance".to_string(),
                "errors".to_string(),
                "usage".to_string(),
            ],
            custom: HashMap::new(),
        }
    }
}

impl Default for GlobalHealthMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            checks: vec![],
            aggregation: HealthAggregation::default(),
            reporting: HealthReporting::default(),
        }
    }
}

impl Default for HealthAggregation {
    fn default() -> Self {
        Self {
            strategy: HealthAggregationStrategy::Majority,
            weights: HashMap::new(),
            thresholds: HealthThresholds::default(),
        }
    }
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            healthy: 0.9,
            warning: 0.7,
            critical: 0.5,
        }
    }
}

impl Default for HealthReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            format: "json".to_string(),
            destinations: vec!["logs".to_string()],
        }
    }
}

impl Default for GlobalPerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: GlobalPerformanceMetrics::default(),
            benchmarking: GlobalBenchmarking::default(),
            profiling: GlobalProfiling::default(),
        }
    }
}

impl Default for GlobalPerformanceMetrics {
    fn default() -> Self {
        Self {
            latency: true,
            throughput: true,
            resource_utilization: true,
            error_rate: true,
        }
    }
}

impl Default for GlobalBenchmarking {
    fn default() -> Self {
        Self {
            enabled: false,
            suites: vec![],
            schedule: BenchmarkSchedule::Manual,
            storage: BenchmarkStorage::default(),
        }
    }
}

impl Default for BenchmarkStorage {
    fn default() -> Self {
        Self {
            backend: "filesystem".to_string(),
            configuration: HashMap::new(),
            retention: Duration::from_days(30),
            compression: true,
        }
    }
}

impl Default for GlobalProfiling {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ProfilingMode::CPU,
            targets: vec![],
            storage: ProfileStorage::default(),
        }
    }
}

impl Default for ProfileStorage {
    fn default() -> Self {
        Self {
            location: "/tmp/profiles".to_string(),
            format: "pprof".to_string(),
            retention: Duration::from_days(7),
            compression: true,
        }
    }
}

impl Default for GlobalAlerting {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            channels: vec![],
            aggregation: GlobalAlertAggregation::default(),
            escalation: GlobalAlertEscalation::default(),
        }
    }
}

impl Default for GlobalAlertAggregation {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(300),
            strategy: GlobalAggregationStrategy::Count,
            max_alerts: 100,
        }
    }
}

impl Default for GlobalAlertEscalation {
    fn default() -> Self {
        Self {
            enabled: false,
            policies: vec![],
            default_policy: "default".to_string(),
        }
    }
}

impl Default for GlobalPerformance {
    fn default() -> Self {
        Self {
            enabled: true,
            targets: PerformanceTargets::default(),
            strategies: vec![
                OptimizationStrategy::PerformanceOptimization,
                OptimizationStrategy::MemoryOptimization,
            ],
            monitoring: PerformanceMonitoring::default(),
        }
    }
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            metrics: PerformanceMetrics::default(),
            alerting: PerformanceAlerting::default(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu: true,
            memory: true,
            network: true,
            disk: true,
            application: true,
        }
    }
}

impl Default for PerformanceAlerting {
    fn default() -> Self {
        Self {
            enabled: false,
            thresholds: HashMap::new(),
            channels: vec![],
            frequency: Duration::from_secs(300),
        }
    }
}

impl Default for IntegrationSettings {
    fn default() -> Self {
        Self {
            external_systems: HashMap::new(),
            apis: ApiConfigurations::default(),
            message_queues: MessageQueueIntegrations::default(),
            databases: DatabaseIntegrations::default(),
            monitoring: MonitoringIntegrations::default(),
        }
    }
}

impl Default for ApiConfigurations {
    fn default() -> Self {
        Self {
            rest: RestApiConfig::default(),
            graphql: GraphQLApiConfig::default(),
            grpc: GrpcApiConfig::default(),
            websocket: WebSocketApiConfig::default(),
        }
    }
}

impl Default for RestApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:8080/api".to_string(),
            version: "v1".to_string(),
            authentication: RestAuthentication::default(),
            rate_limiting: RestRateLimiting::default(),
        }
    }
}

impl Default for RestAuthentication {
    fn default() -> Self {
        Self {
            auth_type: "none".to_string(),
            api_key: None,
            oauth: None,
            jwt: None,
        }
    }
}

impl Default for RestRateLimiting {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_minute: 1000,
            burst_limit: 100,
            algorithm: "token_bucket".to_string(),
        }
    }
}

impl Default for GraphQLApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:8080/graphql".to_string(),
            schema: GraphQLSchema::default(),
            complexity_limits: GraphQLComplexityLimits::default(),
        }
    }
}

impl Default for GraphQLSchema {
    fn default() -> Self {
        Self {
            definition: "".to_string(),
            validation: true,
            introspection: true,
            custom_scalars: vec![],
        }
    }
}

impl Default for GraphQLComplexityLimits {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_complexity: 1000,
            timeout: Duration::from_secs(30),
            custom_calculator: None,
        }
    }
}

impl Default for GrpcApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            address: "127.0.0.1:50051".to_string(),
            services: vec![],
            security: GrpcSecurity::default(),
        }
    }
}

impl Default for GrpcSecurity {
    fn default() -> Self {
        Self {
            tls_enabled: false,
            certificates: GrpcCertificates::default(),
            authentication: vec![],
            authorization: vec![],
        }
    }
}

impl Default for GrpcCertificates {
    fn default() -> Self {
        Self {
            server_cert: "server.crt".to_string(),
            server_key: "server.key".to_string(),
            ca_cert: None,
            client_certs_required: false,
        }
    }
}

impl Default for WebSocketApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "ws://localhost:8080/ws".to_string(),
            protocols: vec!["event-sync".to_string()],
            connection_limits: WebSocketLimits::default(),
            message_handling: WebSocketMessageHandling::default(),
        }
    }
}

impl Default for WebSocketLimits {
    fn default() -> Self {
        Self {
            max_connections: 1000,
            max_message_size: 1024 * 1024, // 1MB
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for WebSocketMessageHandling {
    fn default() -> Self {
        Self {
            message_types: vec![
                "event".to_string(),
                "command".to_string(),
                "query".to_string(),
            ],
            routing: WebSocketRouting::default(),
            validation: true,
            compression: false,
        }
    }
}

impl Default for WebSocketRouting {
    fn default() -> Self {
        Self {
            strategy: "type_based".to_string(),
            routes: HashMap::new(),
            default_handler: "default".to_string(),
        }
    }
}

impl Default for MessageQueueIntegrations {
    fn default() -> Self {
        Self {
            kafka: None,
            rabbitmq: None,
            redis_streams: None,
            custom: HashMap::new(),
        }
    }
}

impl Default for DatabaseIntegrations {
    fn default() -> Self {
        Self {
            postgresql: None,
            mysql: None,
            mongodb: None,
            redis: None,
            custom: HashMap::new(),
        }
    }
}

impl Default for MonitoringIntegrations {
    fn default() -> Self {
        Self {
            prometheus: None,
            grafana: None,
            jaeger: None,
            zipkin: None,
            custom: HashMap::new(),
        }
    }
}

// Extension trait for Duration to add convenience methods
trait DurationExt {
    fn from_minutes(minutes: u64) -> Duration;
    fn from_hours(hours: u64) -> Duration;
    fn from_days(days: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_minutes(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }

    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn from_days(days: u64) -> Duration {
        Duration::from_secs(days * 86400)
    }
}
