// Federated Privacy Modules
//
// This module contains privacy-preserving algorithms specifically designed
// for federated learning scenarios, including secure aggregation, cross-device
// privacy management, composition analysis, and Byzantine-robust aggregation.

pub mod byzantine_aggregation;
pub mod composition_analyzer;
pub mod cross_device_manager;
pub mod secure_aggregation;

// Re-export main types and traits
pub use secure_aggregation::{
    SecureAggregationConfig, SecureAggregationPlan, SecureAggregator, SeedSharingMethod,
};

pub use cross_device_manager::{
    CrossDeviceConfig, CrossDevicePrivacyManager, DeviceProfile, DeviceType, TemporalEvent,
    TemporalEventType,
};

pub use composition_analyzer::{
    ClientComposition, CompositionStats, FederatedCompositionAnalyzer, FederatedCompositionMethod,
    RoundComposition,
};

pub use byzantine_aggregation::{
    AdaptivePrivacyAllocation, ByzantineRobustAggregator, ByzantineRobustConfig,
    ByzantineRobustMethod, OutlierDetectionResult, ReputationSystemConfig, RobustEstimators,
    StatisticalAnalyzer, StatisticalTestConfig, StatisticalTestType, TestStatistic,
};
