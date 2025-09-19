// Helper functions and utilities for event synchronization system

use super::super::core::types::EventSynchronization;

/// Event synchronization utilities
pub struct EventSyncUtils;

impl EventSyncUtils {
    /// Validate module interdependencies
    pub fn validate_dependencies(config: &EventSynchronization) -> Result<(), String> {
        // Check if compression is enabled but persistence doesn't support it
        // Note: This is a placeholder check - actual implementation would need access to compression and persistence details
        // This demonstrates the validation pattern without requiring actual submodule implementation

        // Add more validation logic as needed
        Ok(())
    }

    /// Generate performance report
    pub fn generate_performance_report(config: &EventSynchronization) -> String {
        format!(
            "Event Synchronization Performance Configuration Report\n\
                Delivery: {:?}\n\
                Ordering: {:?}\n\
                Filtering: {:?}\n\
                Compression: {:?}\n\
                Routing: {:?}\n\
                Queue: {:?}\n\
                Handlers: {:?}",
            config.delivery,
            config.ordering,
            config.filtering,
            config.compression,
            config.routing,
            config.queue,
            config.handlers
        )
    }

    /// Validate configuration consistency
    pub fn validate_configuration(config: &EventSynchronization) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate global settings
        if config.global_settings.timeouts.default_operation.as_secs() == 0 {
            errors.push("Default operation timeout cannot be zero".to_string());
        }

        // Validate integration settings
        if config.integration.apis.rest.enabled && config.integration.apis.rest.base_url.is_empty()
        {
            errors.push("REST API base URL cannot be empty when REST API is enabled".to_string());
        }

        if config.integration.apis.graphql.enabled
            && config.integration.apis.graphql.endpoint.is_empty()
        {
            errors.push("GraphQL endpoint cannot be empty when GraphQL API is enabled".to_string());
        }

        if config.integration.apis.grpc.enabled && config.integration.apis.grpc.address.is_empty() {
            errors.push("gRPC address cannot be empty when gRPC API is enabled".to_string());
        }

        if config.integration.apis.websocket.enabled
            && config.integration.apis.websocket.endpoint.is_empty()
        {
            errors.push(
                "WebSocket endpoint cannot be empty when WebSocket API is enabled".to_string(),
            );
        }

        // Validate monitoring settings
        if config.global_settings.monitoring.enabled
            && config.global_settings.monitoring.metrics.interval.as_secs() == 0
        {
            errors
                .push("Monitoring interval cannot be zero when monitoring is enabled".to_string());
        }

        // Validate performance settings
        if config.global_settings.performance.enabled {
            if let Some(latency) = config.global_settings.performance.targets.latency {
                if latency.as_millis() == 0 {
                    errors.push("Performance target latency cannot be zero".to_string());
                }
            }
            if let Some(throughput) = config.global_settings.performance.targets.throughput {
                if throughput <= 0.0 {
                    errors.push("Performance target throughput must be positive".to_string());
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get configuration summary
    pub fn get_configuration_summary(config: &EventSynchronization) -> String {
        let mut summary = String::new();

        summary.push_str("Event Synchronization Configuration Summary\n");
        summary.push_str("=========================================\n\n");

        // Global settings summary
        summary.push_str(&format!("Global Settings:\n"));
        summary.push_str(&format!(
            "  - Event ID Type: {:?}\n",
            config.global_settings.event_id_format.id_type
        ));
        summary.push_str(&format!(
            "  - Default Timeout: {:?}\n",
            config.global_settings.timeouts.default_operation
        ));
        summary.push_str(&format!(
            "  - Monitoring Enabled: {}\n",
            config.global_settings.monitoring.enabled
        ));
        summary.push_str(&format!(
            "  - Performance Enabled: {}\n",
            config.global_settings.performance.enabled
        ));
        summary.push_str("\n");

        // Integration settings summary
        summary.push_str("Integration Settings:\n");
        summary.push_str(&format!(
            "  - REST API: {}\n",
            if config.integration.apis.rest.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        summary.push_str(&format!(
            "  - GraphQL API: {}\n",
            if config.integration.apis.graphql.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        summary.push_str(&format!(
            "  - gRPC API: {}\n",
            if config.integration.apis.grpc.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        summary.push_str(&format!(
            "  - WebSocket API: {}\n",
            if config.integration.apis.websocket.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        summary.push_str("\n");

        // External systems summary
        summary.push_str(&format!(
            "External Systems: {} configured\n",
            config.integration.external_systems.len()
        ));

        summary
    }

    /// Check if high performance mode is enabled
    pub fn is_high_performance_mode(config: &EventSynchronization) -> bool {
        config.global_settings.performance.enabled
            && config
                .global_settings
                .performance
                .targets
                .latency
                .map_or(false, |l| l.as_millis() <= 10)
            && config
                .global_settings
                .performance
                .targets
                .throughput
                .map_or(false, |t| t >= 10000.0)
    }

    /// Check if monitoring is fully configured
    pub fn is_monitoring_configured(config: &EventSynchronization) -> bool {
        config.global_settings.monitoring.enabled
            && config.global_settings.monitoring.metrics.enabled
            && config.global_settings.monitoring.health.enabled
    }

    /// Get recommended configuration improvements
    pub fn get_recommendations(config: &EventSynchronization) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if !config.global_settings.performance.enabled {
            recommendations.push(
                "Consider enabling performance monitoring for better observability".to_string(),
            );
        }

        // Monitoring recommendations
        if !config.global_settings.monitoring.enabled {
            recommendations.push("Enable monitoring for better system visibility".to_string());
        }

        // Error handling recommendations
        if !config.global_settings.error_handling.recovery.enabled {
            recommendations.push("Enable error recovery for better resilience".to_string());
        }

        // Integration recommendations
        if config.integration.external_systems.is_empty() {
            recommendations
                .push("Consider configuring external system integrations if needed".to_string());
        }

        // Security recommendations
        if config.integration.apis.rest.enabled
            && config.integration.apis.rest.authentication.auth_type == "none"
        {
            recommendations
                .push("Consider enabling authentication for REST API security".to_string());
        }

        if config.integration.apis.grpc.enabled
            && !config.integration.apis.grpc.security.tls_enabled
        {
            recommendations.push("Consider enabling TLS for gRPC API security".to_string());
        }

        recommendations
    }

    /// Export configuration to different formats
    pub fn export_configuration(
        config: &EventSynchronization,
        format: &str,
    ) -> Result<String, String> {
        match format.to_lowercase().as_str() {
            "json" => serde_json::to_string_pretty(config)
                .map_err(|e| format!("JSON serialization error: {}", e)),
            "toml" => {
                // Note: This would require the toml crate
                Err("TOML export not implemented".to_string())
            }
            "yaml" => {
                // Note: This would require the serde_yaml crate
                Err("YAML export not implemented".to_string())
            }
            _ => Err(format!("Unsupported export format: {}", format)),
        }
    }

    /// Import configuration from different formats
    pub fn import_configuration(data: &str, format: &str) -> Result<EventSynchronization, String> {
        match format.to_lowercase().as_str() {
            "json" => {
                serde_json::from_str(data).map_err(|e| format!("JSON deserialization error: {}", e))
            }
            "toml" => {
                // Note: This would require the toml crate
                Err("TOML import not implemented".to_string())
            }
            "yaml" => {
                // Note: This would require the serde_yaml crate
                Err("YAML import not implemented".to_string())
            }
            _ => Err(format!("Unsupported import format: {}", format)),
        }
    }
}
