// Clock Synchronization Module
//
// This module provides comprehensive clock synchronization capabilities for TPU pod coordination.
// The module is organized into focused sub-modules that handle different aspects of time synchronization:
//
// - [`core`] - Main synchronization manager and coordination logic
// - [`protocols`] - Synchronization protocols (NTP, PTP, GPS, etc.)
// - [`sources`] - Time source management and selection algorithms
// - [`gps`] - GPS signal processing and error correction
// - [`network`] - Network synchronization, messaging, and load balancing
// - [`quality`] - Quality monitoring and assessment
// - [`drift`] - Drift compensation and prediction
// - [`health`] - Health monitoring and recovery
// - [`statistics`] - Performance tracking and reporting
//
// # Architecture
//
// The clock synchronization system follows a modular architecture where each component
// has a specific responsibility but can work together to provide robust time synchronization:
//
// ```text
// ┌─────────────────────────────────────────────────────────────────┐
// │                    ClockSynchronizationManager                  │
// │                         (core module)                          │
// └─────────────────────────┬───────────────────────────────────────┘
//                           │
// ┌─────────────────────────┼───────────────────────────────────────┐
// │         TimeSourceManager        │        ProtocolManager        │
// │         (sources module)         │       (protocols module)      │
// └─────────────────────────┬───────┴───────┬───────────────────────┘
//                           │               │
// ┌─────────────────────────┼───────────────┼───────────────────────┐
// │      GPS Processing     │   Network     │    Quality & Health   │
// │      (gps module)       │  (network)    │  (quality & health)   │
// └─────────────────────────┼───────────────┼───────────────────────┘
//                           │               │
// ┌─────────────────────────┼───────────────┼───────────────────────┐
// │   Drift Compensation    │  Statistics   │      Reporting        │
// │     (drift module)      │ (statistics)  │    (statistics)       │
// └─────────────────────────┴───────────────┴───────────────────────┘
// ```
//
// # Usage
//
// Basic usage of the clock synchronization system:
//
// ```rust
// use crate::pod_coordination::synchronization::clocks::{
//     ClockSynchronizationManager, ClockSynchronizationConfig
// };
//
// # fn example() -> Result<(), Box<dyn std::error::Error>> {
// // Create synchronization manager with default configuration
// let mut sync_manager = ClockSynchronizationManager::new(
//     ClockSynchronizationConfig::default()
// )?;
//
// // Start synchronization
// sync_manager.start_synchronization()?;
//
// // Perform synchronization
// sync_manager.synchronize()?;
//
// // Get synchronization status
// let status = sync_manager.get_synchronization_status();
// println!("Sync status: {:?}", status);
//
// // Stop synchronization
// sync_manager.stop_synchronization()?;
// # Ok(())
// # }
// ```
//
// # Performance Considerations
//
// The clock synchronization system is designed for high-performance operation with:
// - Minimal latency overhead
// - Efficient memory usage
// - Scalable to large TPU clusters
// - Real-time operation capabilities
// - Adaptive algorithms for varying network conditions

// Core synchronization components
pub mod core;
pub mod protocols;
pub mod sources;

// Specialized synchronization modules
pub mod gps;
pub mod network;

// Monitoring and analysis modules
pub mod drift;
pub mod health;
pub mod quality;
pub mod statistics;

// Re-export main types from core module
pub use core::{
    ClockOffset, ClockSynchronizationConfig, ClockSynchronizationManager,
    ClockSynchronizationState, ClockSynchronizationStatus, ClockSynchronizer, SynchronizationEvent,
    SynchronizationResult,
};

// Re-export protocol types
pub use protocols::{
    BerkeleyConfig, ClockSyncProtocol, CristianConfig, CustomProtocolConfig, NtpConfig, NtpPeer,
    NtpSynchronizer, NtpTimestamps, ProtocolError, ProtocolManager, PtpConfig, PtpSynchronizer,
    SntpConfig, SntpSynchronizer,
};

// Re-export source management types
pub use sources::{
    AtomicClockType, ClockSource, RadioTimeStation, SourceSelectionAlgorithm,
    SourceSelectionCriteria, SourceValidation, SystemClockConfig, TimeSource, TimeSourceConfig,
    TimeSourceManager,
};

// Re-export GPS synchronization types
pub use gps::{
    AntennaConfig, GpsConfig, GpsError, GpsErrorCorrection, GpsReceiverType, GpsSignalProcessing,
    GpsSynchronizationManager, GpsTime, IonosphericCorrection, SatelliteClockCorrection,
    TroposphericCorrection,
};

// Re-export network synchronization types
pub use network::{
    LoadBalancingAlgorithm, MessagePassingConfig, MessagePriority, NetworkFaultTolerance,
    NetworkLoadBalancing, NetworkSyncConfig, NetworkSyncError, NetworkSynchronizationManager,
    NetworkTopology, SyncMessageType,
};

// Re-export quality monitoring types
pub use quality::{
    ClockAccuracyRequirements, ClockQualityMonitor, QualityAssessment, QualityGrade, QualityMetric,
    QualityMonitoringConfig, QualityRequirements, QualitySnapshot, QualityThresholds,
    SourceQualityMonitoring,
};

// Re-export drift compensation types
pub use drift::{
    DriftCompensationAlgorithm, DriftCompensationConfig, DriftCompensationError,
    DriftCompensationStatus, DriftCompensator, DriftMeasurement, DriftMeasurementConfig,
    DriftModel, DriftPredictionConfig, DriftPredictionEngine,
};

// Re-export health monitoring types
pub use health::{
    AlertConfiguration, AlertSeverity, HealthAlert, HealthCheck, HealthCheckType,
    HealthMonitorConfig, HealthMonitorError, HealthStatus, HealthThresholds, RecoveryConfiguration,
    SourceFailoverConfig, SourceHealthMonitor,
};

// Re-export statistics and reporting types
pub use statistics::{
    ClockStatistics, PerformanceHistory, PerformanceMeasurement, PerformanceReport,
    PerformanceTracking, QualityReporting, ReliabilityStatistics, ReportGeneration,
    StatisticsCollector, StatisticsError, TrendDirection,
};

// Convenience type aliases
pub type Result<T> = std::result::Result<T, ClockSynchronizationError>;
pub type Duration = std::time::Duration;
pub type Instant = std::time::Instant;

/// Main error type for clock synchronization operations
#[derive(Debug)]
pub enum ClockSynchronizationError {
    /// Core synchronization error
    CoreError(core::ClockSynchronizationError),
    /// Protocol error
    ProtocolError(protocols::ProtocolError),
    /// Source management error
    SourceError(sources::SourceManagementError),
    /// GPS synchronization error
    GpsError(gps::GpsError),
    /// Network synchronization error
    NetworkError(network::NetworkSyncError),
    /// Quality monitoring error
    QualityError(quality::QualityMonitorError),
    /// Drift compensation error
    DriftError(drift::DriftCompensationError),
    /// Health monitoring error
    HealthError(health::HealthMonitorError),
    /// Statistics error
    StatisticsError(statistics::StatisticsError),
    /// Configuration error
    ConfigurationError(String),
    /// System error
    SystemError(String),
}

impl std::fmt::Display for ClockSynchronizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClockSynchronizationError::CoreError(e) => {
                write!(f, "Core synchronization error: {}", e)
            }
            // No prefix here: `ProtocolError`'s own `Display` already starts
            // with "Protocol error", and prefixing again rendered
            // "Protocol error: Protocol error: ...". The inner type stays
            // self-describing because it is also returned standalone from
            // `NtpTimestamps::round_trip_delay`/`offset`.
            ClockSynchronizationError::ProtocolError(e) => write!(f, "{}", e),
            ClockSynchronizationError::SourceError(e) => {
                write!(f, "Source management error: {}", e)
            }
            ClockSynchronizationError::GpsError(e) => write!(f, "GPS synchronization error: {}", e),
            ClockSynchronizationError::NetworkError(e) => {
                write!(f, "Network synchronization error: {}", e)
            }
            ClockSynchronizationError::QualityError(e) => {
                write!(f, "Quality monitoring error: {}", e)
            }
            ClockSynchronizationError::DriftError(e) => {
                write!(f, "Drift compensation error: {}", e)
            }
            ClockSynchronizationError::HealthError(e) => {
                write!(f, "Health monitoring error: {}", e)
            }
            ClockSynchronizationError::StatisticsError(e) => write!(f, "Statistics error: {}", e),
            ClockSynchronizationError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            ClockSynchronizationError::SystemError(msg) => write!(f, "System error: {}", msg),
        }
    }
}

impl std::error::Error for ClockSynchronizationError {}

// Error conversions for seamless error handling
impl From<core::ClockSynchronizationError> for ClockSynchronizationError {
    fn from(err: core::ClockSynchronizationError) -> Self {
        ClockSynchronizationError::CoreError(err)
    }
}

impl From<protocols::ProtocolError> for ClockSynchronizationError {
    fn from(err: protocols::ProtocolError) -> Self {
        ClockSynchronizationError::ProtocolError(err)
    }
}

impl From<sources::SourceManagementError> for ClockSynchronizationError {
    fn from(err: sources::SourceManagementError) -> Self {
        ClockSynchronizationError::SourceError(err)
    }
}

impl From<gps::GpsError> for ClockSynchronizationError {
    fn from(err: gps::GpsError) -> Self {
        ClockSynchronizationError::GpsError(err)
    }
}

impl From<network::NetworkSyncError> for ClockSynchronizationError {
    fn from(err: network::NetworkSyncError) -> Self {
        ClockSynchronizationError::NetworkError(err)
    }
}

impl From<quality::QualityMonitorError> for ClockSynchronizationError {
    fn from(err: quality::QualityMonitorError) -> Self {
        ClockSynchronizationError::QualityError(err)
    }
}

impl From<drift::DriftCompensationError> for ClockSynchronizationError {
    fn from(err: drift::DriftCompensationError) -> Self {
        ClockSynchronizationError::DriftError(err)
    }
}

impl From<health::HealthMonitorError> for ClockSynchronizationError {
    fn from(err: health::HealthMonitorError) -> Self {
        ClockSynchronizationError::HealthError(err)
    }
}

impl From<statistics::StatisticsError> for ClockSynchronizationError {
    fn from(err: statistics::StatisticsError) -> Self {
        ClockSynchronizationError::StatisticsError(err)
    }
}

impl From<scirs2_core::CoreError> for ClockSynchronizationError {
    fn from(err: scirs2_core::CoreError) -> Self {
        ClockSynchronizationError::SystemError(err.to_string())
    }
}

/// Builder for configuring clock synchronization
///
/// Provides a fluent interface for configuring the various aspects
/// of clock synchronization with sensible defaults.
#[derive(Debug)]
pub struct ClockSynchronizationBuilder {
    core_config: Option<core::ClockSynchronizationConfig>,
    protocol_configs: Vec<protocols::ClockSyncProtocol>,
    source_configs: Vec<sources::TimeSource>,
    gps_config: Option<gps::GpsConfig>,
    network_config: Option<network::NetworkSyncConfig>,
    quality_config: Option<quality::QualityMonitoringConfig>,
    drift_config: Option<drift::DriftCompensationConfig>,
    health_config: Option<health::HealthMonitorConfig>,
    statistics_config: Option<statistics::StatisticsCollectionConfig>,
}

impl ClockSynchronizationBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            core_config: None,
            protocol_configs: Vec::new(),
            source_configs: Vec::new(),
            gps_config: None,
            network_config: None,
            quality_config: None,
            drift_config: None,
            health_config: None,
            statistics_config: None,
        }
    }

    /// Set core synchronization configuration
    pub fn with_core_config(mut self, config: core::ClockSynchronizationConfig) -> Self {
        self.core_config = Some(config);
        self
    }

    /// Add synchronization protocol
    pub fn with_protocol(mut self, protocol: protocols::ClockSyncProtocol) -> Self {
        self.protocol_configs.push(protocol);
        self
    }

    /// Add time source
    pub fn with_source(mut self, source: sources::TimeSource) -> Self {
        self.source_configs.push(source);
        self
    }

    /// Set GPS configuration
    pub fn with_gps_config(mut self, config: gps::GpsConfig) -> Self {
        self.gps_config = Some(config);
        self
    }

    /// Set network synchronization configuration
    pub fn with_network_config(mut self, config: network::NetworkSyncConfig) -> Self {
        self.network_config = Some(config);
        self
    }

    /// Set quality monitoring configuration
    pub fn with_quality_config(mut self, config: quality::QualityMonitoringConfig) -> Self {
        self.quality_config = Some(config);
        self
    }

    /// Set drift compensation configuration
    pub fn with_drift_config(mut self, config: drift::DriftCompensationConfig) -> Self {
        self.drift_config = Some(config);
        self
    }

    /// Set health monitoring configuration
    pub fn with_health_config(mut self, config: health::HealthMonitorConfig) -> Self {
        self.health_config = Some(config);
        self
    }

    /// Set statistics collection configuration
    pub fn with_statistics_config(
        mut self,
        config: statistics::StatisticsCollectionConfig,
    ) -> Self {
        self.statistics_config = Some(config);
        self
    }

    /// Build the clock synchronization manager
    pub fn build(self) -> Result<ClockSynchronizationManager> {
        let core_config = self.core_config.unwrap_or_default();

        // Create and configure the synchronization manager
        let mut manager = ClockSynchronizationManager::new();
        manager.config = core_config;

        // Configure protocols
        for protocol in self.protocol_configs {
            manager.add_protocol(protocol)?;
        }

        // Configure sources
        for source in self.source_configs {
            manager.add_time_source(source)?;
        }

        // Apply additional configurations
        if let Some(gps_config) = self.gps_config {
            manager.configure_gps(gps_config)?;
        }

        if let Some(network_config) = self.network_config {
            manager.configure_network(network_config)?;
        }

        if let Some(quality_config) = self.quality_config {
            manager.configure_quality_monitoring(quality_config)?;
        }

        if let Some(drift_config) = self.drift_config {
            manager.configure_drift_compensation(drift_config)?;
        }

        if let Some(health_config) = self.health_config {
            manager.configure_health_monitoring(health_config)?;
        }

        if let Some(statistics_config) = self.statistics_config {
            manager.configure_statistics(statistics_config)?;
        }

        Ok(manager)
    }
}

impl Default for ClockSynchronizationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for clock synchronization
pub mod utils {
    use super::*;

    /// Create a basic NTP-based synchronization setup, one time source per
    /// supplied server.
    ///
    /// The server list is the whole point of an NTP setup, so an empty list is
    /// an honest configuration error rather than a manager with nothing to
    /// synchronize against. (This used to loop over
    /// `builder.source_configs.len()` -- which is zero on a fresh builder --
    /// and so added no sources at all while ignoring `ntp_servers` entirely.)
    pub fn create_ntp_sync_manager(
        ntp_servers: Vec<String>,
    ) -> Result<ClockSynchronizationManager> {
        if ntp_servers.is_empty() {
            return Err(ClockSynchronizationError::ConfigurationError(
                "an NTP synchronization manager needs at least one server address".to_string(),
            ));
        }

        let mut builder = ClockSynchronizationBuilder::new();

        // Add NTP protocol
        builder = builder.with_protocol(protocols::ClockSyncProtocol::NTP);

        // One addressable network time source per configured server.
        for server in ntp_servers {
            builder = builder.with_source(sources::TimeSource {
                source_type: sources::ClockSource::NTP,
                address: Some(server),
            });
        }

        // Enable basic monitoring
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());

        builder.build()
    }

    /// Create a GPS-based synchronization setup
    pub fn create_gps_sync_manager(
        gps_config: gps::GpsConfig,
    ) -> Result<ClockSynchronizationManager> {
        let mut builder = ClockSynchronizationBuilder::new();

        // Add GPS configuration
        builder = builder.with_gps_config(gps_config.clone());

        // Add GPS time source. The receiver's device path comes from the GPS
        // configuration rather than being invented here.
        let source = sources::TimeSource {
            source_type: sources::ClockSource::GPS,
            address: None,
        };
        builder = builder.with_source(source);

        // Enable comprehensive monitoring for GPS
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_drift_config(drift::DriftCompensationConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());

        builder.build()
    }

    /// Create a high-precision synchronization setup
    pub fn create_precision_sync_manager() -> Result<ClockSynchronizationManager> {
        let mut builder = ClockSynchronizationBuilder::new();

        // Use PTP for high precision
        builder = builder.with_protocol(protocols::ClockSyncProtocol::PTP);

        // Add atomic clock source. A locally attached reference needs no
        // network address.
        let source = sources::TimeSource {
            source_type: sources::ClockSource::Atomic,
            address: None,
        };
        builder = builder.with_source(source);

        // Enable all monitoring and compensation
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_drift_config(drift::DriftCompensationConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());
        builder = builder.with_statistics_config(statistics::StatisticsCollectionConfig::default());

        builder.build()
    }

    /// Convert duration to human-readable string
    pub fn duration_to_string(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();
        let micros = duration.subsec_micros() % 1000;
        let nanos = duration.subsec_nanos() % 1000;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else if millis > 0 {
            format!("{}.{:03}ms", millis, micros)
        } else if micros > 0 {
            format!("{}.{:03}μs", micros, nanos)
        } else {
            format!("{}ns", nanos)
        }
    }

    /// Validate clock offset against requirements
    pub fn validate_clock_offset(
        offset: ClockOffset,
        requirements: &quality::ClockAccuracyRequirements,
    ) -> bool {
        offset.offset_ns.abs() as f64 <= requirements.max_drift_ppm
    }

    /// Calculate quality score from multiple metrics
    pub fn calculate_quality_score(metrics: &std::collections::HashMap<String, f64>) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let sum: f64 = metrics.values().sum();
        sum / metrics.len() as f64
    }

    /// Wall-clock time elapsed since this process-uptime tracker was first
    /// consulted, as a real, live measurement.
    ///
    /// There is no portable, pure-Rust way (no FFI, no platform-specific
    /// `/proc`/`sysctl` parsing) to ask the OS for the true process start
    /// time on every platform this crate targets. Rather than fabricate a
    /// plausible-looking constant, this establishes a monotonic anchor the
    /// first time it is called (via [`std::sync::OnceLock`]) and returns
    /// [`Instant::elapsed`] against that anchor on every call thereafter
    /// (including the first, which returns a value very close to zero).
    ///
    /// Renamed from the former `get_system_uptime`: that name promised the
    /// OS-level system uptime, which this never measured (it always
    /// returned a hardcoded `Duration::from_secs(86400)`). `process_uptime`
    /// accurately describes what a pure-Rust anchor-based measurement can
    /// honestly provide.
    pub fn process_uptime() -> Duration {
        static PROCESS_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = *PROCESS_START.get_or_init(Instant::now);
        start.elapsed()
    }
}

/// Prelude module for common imports
pub mod prelude {}

// Module-level documentation tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let builder =
            ClockSynchronizationBuilder::new().with_protocol(protocols::ClockSyncProtocol::NTP);

        // Builder should be constructible
        assert!(builder.protocol_configs.len() == 1);
    }

    #[test]
    fn test_error_conversions() {
        let core_error = core::ClockSynchronizationError;
        let sync_error: ClockSynchronizationError = core_error.into();

        // The conversion should work
        match sync_error {
            ClockSynchronizationError::CoreError(_) => {}
            _ => panic!("Error conversion failed"),
        }
    }

    #[test]
    fn test_utility_functions() {
        // Test duration formatting
        let duration = Duration::from_millis(1500);
        let formatted = utils::duration_to_string(duration);
        assert!(formatted.contains("s"));

        // Test quality score calculation
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("accuracy".to_string(), 0.9);
        metrics.insert("stability".to_string(), 0.8);
        let score = utils::calculate_quality_score(&metrics);
        assert!((score - 0.85).abs() < 1e-10);
    }

    // Regression test: `process_uptime` (formerly `get_system_uptime`) used
    // to always return a hardcoded `Duration::from_secs(86400)`. A fake
    // constant would pass a naive "returns a Duration" check but can never
    // reflect real elapsed time; this asserts it strictly increases with
    // real wall-clock time and never equals the old fabricated value.
    #[test]
    fn process_uptime_reflects_real_elapsed_time_not_a_fixed_constant() {
        let first = utils::process_uptime();
        std::thread::sleep(Duration::from_millis(30));
        let second = utils::process_uptime();

        assert!(
            second > first,
            "process_uptime must strictly increase with real elapsed time \
             (first={first:?}, second={second:?})"
        );
        assert!(
            second - first >= Duration::from_millis(20),
            "process_uptime delta should reflect the real 30ms sleep, got {:?}",
            second - first
        );
        assert_ne!(
            second,
            Duration::from_secs(86400),
            "must not be the old hardcoded fabricated constant"
        );
    }
}
