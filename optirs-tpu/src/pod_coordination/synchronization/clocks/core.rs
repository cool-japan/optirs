// Clock Core Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClockOffset {
    pub offset_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClockSynchronizationConfig {
    pub sync_interval_ms: u64,
}

/// Clock synchronization manager.
///
/// Holds every configuration handed to it through `add_*`/`configure_*`. The
/// previous implementation accepted all of these calls, returned `Ok(())`,
/// and silently discarded the argument, so a caller had no way to observe
/// what — if anything — had actually been configured. Each setter below now
/// stores its argument, and the matching accessor lets a caller (or a test)
/// confirm the stored value round-trips.
#[derive(Debug, Clone, Default)]
pub struct ClockSynchronizationManager {
    pub config: ClockSynchronizationConfig,
    protocols: Vec<super::protocols::ClockSyncProtocol>,
    time_sources: Vec<super::sources::TimeSource>,
    gps_config: Option<super::gps::GpsConfig>,
    network_config: Option<super::network::NetworkSyncConfig>,
    quality_config: Option<super::quality::QualityMonitoringConfig>,
    drift_config: Option<super::drift::DriftCompensationConfig>,
    health_config: Option<super::health::HealthMonitorConfig>,
    statistics_config: Option<super::statistics::StatisticsCollectionConfig>,
}

impl ClockSynchronizationManager {
    /// Create a new clock synchronization manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a protocol configuration
    pub fn add_protocol(
        &mut self,
        protocol: super::protocols::ClockSyncProtocol,
    ) -> crate::error::Result<()> {
        self.protocols.push(protocol);
        Ok(())
    }

    /// Protocols registered via [`Self::add_protocol`], in registration order.
    pub fn protocols(&self) -> &[super::protocols::ClockSyncProtocol] {
        &self.protocols
    }

    /// Add a time source
    pub fn add_time_source(
        &mut self,
        source: super::sources::TimeSource,
    ) -> crate::error::Result<()> {
        self.time_sources.push(source);
        Ok(())
    }

    /// Time sources registered via [`Self::add_time_source`], in registration order.
    pub fn time_sources(&self) -> &[super::sources::TimeSource] {
        &self.time_sources
    }

    /// Configure GPS settings
    pub fn configure_gps(&mut self, config: super::gps::GpsConfig) -> crate::error::Result<()> {
        self.gps_config = Some(config);
        Ok(())
    }

    /// The GPS configuration set by [`Self::configure_gps`], if any.
    pub fn gps_config(&self) -> Option<&super::gps::GpsConfig> {
        self.gps_config.as_ref()
    }

    /// Configure network settings
    pub fn configure_network(
        &mut self,
        config: super::network::NetworkSyncConfig,
    ) -> crate::error::Result<()> {
        self.network_config = Some(config);
        Ok(())
    }

    /// The network configuration set by [`Self::configure_network`], if any.
    pub fn network_config(&self) -> Option<&super::network::NetworkSyncConfig> {
        self.network_config.as_ref()
    }

    /// Configure quality monitoring
    pub fn configure_quality_monitoring(
        &mut self,
        config: super::quality::QualityMonitoringConfig,
    ) -> crate::error::Result<()> {
        self.quality_config = Some(config);
        Ok(())
    }

    /// The quality-monitoring configuration set by
    /// [`Self::configure_quality_monitoring`], if any.
    pub fn quality_config(&self) -> Option<&super::quality::QualityMonitoringConfig> {
        self.quality_config.as_ref()
    }

    /// Configure drift compensation
    pub fn configure_drift_compensation(
        &mut self,
        config: super::drift::DriftCompensationConfig,
    ) -> crate::error::Result<()> {
        self.drift_config = Some(config);
        Ok(())
    }

    /// The drift-compensation configuration set by
    /// [`Self::configure_drift_compensation`], if any.
    pub fn drift_config(&self) -> Option<&super::drift::DriftCompensationConfig> {
        self.drift_config.as_ref()
    }

    /// Configure health monitoring
    pub fn configure_health_monitoring(
        &mut self,
        config: super::health::HealthMonitorConfig,
    ) -> crate::error::Result<()> {
        self.health_config = Some(config);
        Ok(())
    }

    /// The health-monitoring configuration set by
    /// [`Self::configure_health_monitoring`], if any.
    pub fn health_config(&self) -> Option<&super::health::HealthMonitorConfig> {
        self.health_config.as_ref()
    }

    /// Configure statistics collection
    pub fn configure_statistics(
        &mut self,
        config: super::statistics::StatisticsCollectionConfig,
    ) -> crate::error::Result<()> {
        self.statistics_config = Some(config);
        Ok(())
    }

    /// The statistics-collection configuration set by
    /// [`Self::configure_statistics`], if any.
    pub fn statistics_config(&self) -> Option<&super::statistics::StatisticsCollectionConfig> {
        self.statistics_config.as_ref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ClockSynchronizationState {
    Synced,
    Syncing,
    #[default]
    OutOfSync,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ClockSynchronizationStatus {
    Active,
    #[default]
    Inactive,
    Error,
}

#[derive(Debug, Clone, Default)]
pub struct ClockSynchronizer {
    pub state: ClockSynchronizationState,
    pub status: ClockSynchronizationStatus,
}

#[derive(Debug, Clone)]
pub struct SynchronizationEvent;

#[derive(Debug, Clone, Default)]
pub struct SynchronizationResult {
    pub success: bool,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ClockSynchronizationError;

impl std::fmt::Display for ClockSynchronizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Clock synchronization error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression test for F26: every `add_*`/`configure_*` call used to
    // return `Ok(())` while dropping its argument, so a manager built
    // through several configuration calls was indistinguishable from a
    // freshly created one. Each setter's stored value must now round-trip
    // through its accessor.
    #[test]
    fn configuration_round_trips_instead_of_being_discarded() {
        let mut manager = ClockSynchronizationManager::new();

        manager
            .add_protocol(super::super::protocols::ClockSyncProtocol::NTP)
            .expect("add_protocol must succeed");
        manager
            .add_protocol(super::super::protocols::ClockSyncProtocol::PTP)
            .expect("add_protocol must succeed");
        assert_eq!(manager.protocols().len(), 2);

        manager
            .add_time_source(super::super::sources::TimeSource::default())
            .expect("add_time_source must succeed");
        assert_eq!(manager.time_sources().len(), 1);

        assert!(manager.gps_config().is_none());
        manager
            .configure_gps(super::super::gps::GpsConfig::default())
            .expect("configure_gps must succeed");
        assert!(manager.gps_config().is_some());

        assert!(manager.network_config().is_none());
        manager
            .configure_network(super::super::network::NetworkSyncConfig::default())
            .expect("configure_network must succeed");
        assert!(manager.network_config().is_some());

        assert!(manager.quality_config().is_none());
        manager
            .configure_quality_monitoring(super::super::quality::QualityMonitoringConfig::default())
            .expect("configure_quality_monitoring must succeed");
        assert!(manager.quality_config().is_some());

        assert!(manager.drift_config().is_none());
        manager
            .configure_drift_compensation(super::super::drift::DriftCompensationConfig::default())
            .expect("configure_drift_compensation must succeed");
        assert!(manager.drift_config().is_some());

        assert!(manager.health_config().is_none());
        manager
            .configure_health_monitoring(super::super::health::HealthMonitorConfig::default())
            .expect("configure_health_monitoring must succeed");
        assert!(manager.health_config().is_some());

        assert!(manager.statistics_config().is_none());
        manager
            .configure_statistics(super::super::statistics::StatisticsCollectionConfig::default())
            .expect("configure_statistics must succeed");
        assert!(manager.statistics_config().is_some());
    }

    #[test]
    fn fresh_manager_has_no_configuration() {
        let manager = ClockSynchronizationManager::new();
        assert!(manager.protocols().is_empty());
        assert!(manager.time_sources().is_empty());
        assert!(manager.gps_config().is_none());
        assert!(manager.network_config().is_none());
        assert!(manager.quality_config().is_none());
        assert!(manager.drift_config().is_none());
        assert!(manager.health_config().is_none());
        assert!(manager.statistics_config().is_none());
    }
}
