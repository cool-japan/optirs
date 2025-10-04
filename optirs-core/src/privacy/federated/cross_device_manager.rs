// Cross-Device Privacy Manager Module
//
// This module implements privacy management across different devices and users
// in federated learning scenarios, including user-level privacy guarantees,
// device clustering, and temporal correlation tracking.

use super::super::PrivacyBudget;
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Cross-device privacy configuration
#[derive(Debug, Clone, Default)]
pub struct CrossDeviceConfig {
    /// User-level privacy guarantees
    pub user_level_privacy: bool,

    /// Device clustering for privacy
    pub device_clustering: bool,

    /// Temporal privacy across rounds
    pub temporal_privacy: bool,

    /// Geographic privacy considerations
    pub geographic_privacy: bool,

    /// Demographic privacy protection
    pub demographic_privacy: bool,
}

/// Cross-device privacy manager
pub struct CrossDevicePrivacyManager<T: Float + Debug + Send + Sync + 'static> {
    config: CrossDeviceConfig,
    user_clusters: HashMap<String, Vec<String>>,
    device_profiles: HashMap<String, DeviceProfile<T>>,
    temporal_correlations: HashMap<String, Vec<TemporalEvent>>,
}

/// Device profile for cross-device privacy
#[derive(Debug, Clone)]
pub struct DeviceProfile<T: Float + Debug + Send + Sync + 'static> {
    pub device_id: String,
    pub user_id: String,
    pub device_type: DeviceType,
    pub location_cluster: String,
    pub participation_frequency: f64,
    pub local_privacy_budget: PrivacyBudget,
    pub sensitivity_estimate: T,
}

/// Device types for privacy analysis
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Default)]
pub enum DeviceType {
    #[default]
    Mobile,
    Desktop,
    IoT,
    Edge,
    Server,
}

/// Temporal event for privacy tracking
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    pub timestamp: u64,
    pub event_type: TemporalEventType,
    pub privacy_impact: f64,
}

#[derive(Debug, Clone)]
pub enum TemporalEventType {
    ClientParticipation,
    ModelUpdate,
    PrivacyBudgetConsumption,
    AggregationEvent,
}

impl<T: Float + Debug + Send + Sync + 'static> CrossDevicePrivacyManager<T> {
    pub fn new(config: CrossDeviceConfig) -> Self {
        Self {
            config,
            user_clusters: HashMap::new(),
            device_profiles: HashMap::new(),
            temporal_correlations: HashMap::new(),
        }
    }

    pub fn update_participation(&mut self, clientid: String, round: usize) {
        // Update device profile
        if let Some(profile) = self.device_profiles.get_mut(&clientid) {
            profile.participation_frequency += 0.1; // Simple increment
        } else {
            // Create new profile
            let profile = DeviceProfile {
                device_id: clientid.clone(),
                user_id: clientid.clone(),       // Simplified
                device_type: DeviceType::Mobile, // Default
                location_cluster: "default".to_string(),
                participation_frequency: 1.0,
                local_privacy_budget: PrivacyBudget::default(),
                sensitivity_estimate: T::one(),
            };
            self.device_profiles.insert(clientid.clone(), profile);
        }

        // Record temporal event
        self.temporal_correlations
            .entry(clientid)
            .or_default()
            .push(TemporalEvent {
                timestamp: round as u64, // Simplified timestamp
                event_type: TemporalEventType::ClientParticipation,
                privacy_impact: 1.0,
            });
    }

    /// Get device profile for a client
    pub fn get_device_profile(&self, client_id: &str) -> Option<&DeviceProfile<T>> {
        self.device_profiles.get(client_id)
    }

    /// Get temporal correlations for a client
    pub fn get_temporal_correlations(&self, client_id: &str) -> Option<&Vec<TemporalEvent>> {
        self.temporal_correlations.get(client_id)
    }

    /// Create user cluster for related devices
    pub fn create_user_cluster(&mut self, user_id: String, device_ids: Vec<String>) {
        self.user_clusters.insert(user_id, device_ids);
    }

    /// Get devices in a user cluster
    pub fn get_user_cluster(&self, user_id: &str) -> Option<&Vec<String>> {
        self.user_clusters.get(user_id)
    }

    /// Check if user-level privacy is enabled
    pub fn is_user_level_privacy_enabled(&self) -> bool {
        self.config.user_level_privacy
    }

    /// Check if device clustering is enabled
    pub fn is_device_clustering_enabled(&self) -> bool {
        self.config.device_clustering
    }

    /// Check if temporal privacy is enabled
    pub fn is_temporal_privacy_enabled(&self) -> bool {
        self.config.temporal_privacy
    }

    /// Get configuration
    pub fn config(&self) -> &CrossDeviceConfig {
        &self.config
    }

    /// Get number of tracked devices
    pub fn device_count(&self) -> usize {
        self.device_profiles.len()
    }

    /// Get participation frequency for a device
    pub fn get_participation_frequency(&self, client_id: &str) -> Option<f64> {
        self.device_profiles
            .get(client_id)
            .map(|p| p.participation_frequency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_device_config() {
        let config = CrossDeviceConfig::default();
        assert!(!config.user_level_privacy);
        assert!(!config.device_clustering);
        assert!(!config.temporal_privacy);
    }

    #[test]
    fn test_device_profile_creation() {
        let profile = DeviceProfile {
            device_id: "device_1".to_string(),
            user_id: "user_1".to_string(),
            device_type: DeviceType::Mobile,
            location_cluster: "cluster_a".to_string(),
            participation_frequency: 0.5,
            local_privacy_budget: PrivacyBudget::default(),
            sensitivity_estimate: 1.0f64,
        };

        assert_eq!(profile.device_id, "device_1");
        assert!(matches!(profile.device_type, DeviceType::Mobile));
    }

    #[test]
    fn test_cross_device_manager() {
        let config = CrossDeviceConfig::default();
        let mut manager = CrossDevicePrivacyManager::<f64>::new(config);

        assert_eq!(manager.device_count(), 0);

        manager.update_participation("device1".to_string(), 1);
        assert_eq!(manager.device_count(), 1);

        let frequency = manager.get_participation_frequency("device1");
        assert!(frequency.is_some());
        assert_eq!(frequency.unwrap(), 1.0);
    }

    #[test]
    fn test_user_clustering() {
        let config = CrossDeviceConfig::default();
        let mut manager = CrossDevicePrivacyManager::<f64>::new(config);

        let user_id = "user1".to_string();
        let devices = vec!["device1".to_string(), "device2".to_string()];

        manager.create_user_cluster(user_id.clone(), devices.clone());

        let cluster = manager.get_user_cluster(&user_id);
        assert!(cluster.is_some());
        assert_eq!(cluster.unwrap().len(), 2);
    }

    #[test]
    fn test_temporal_events() {
        let config = CrossDeviceConfig::default();
        let mut manager = CrossDevicePrivacyManager::<f64>::new(config);

        manager.update_participation("device1".to_string(), 1);
        manager.update_participation("device1".to_string(), 2);

        let events = manager.get_temporal_correlations("device1");
        assert!(events.is_some());
        assert_eq!(events.unwrap().len(), 2);
    }
}
