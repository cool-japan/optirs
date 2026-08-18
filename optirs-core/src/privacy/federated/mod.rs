// Federated Privacy Modules
//
// This module contains privacy-preserving algorithms specifically designed
// for federated learning scenarios, including secure aggregation, cross-device
// privacy management, composition analysis, and Byzantine-robust aggregation.

pub mod byzantine_aggregation;
pub mod composition_analyzer;
pub mod cross_device_manager;
pub mod outlier_tests;
pub mod pairwise_masking;
pub mod robust_ops;
pub mod secure_aggregation;

// Re-export main types and traits
pub use secure_aggregation::{
    disclose_dropout_masks, mask_client_update, DropoutDisclosure, MaskedClientUpdate,
    SecureAggregationConfig, SecureAggregationPlan, SecureAggregator, SeedSharingMethod,
    DEFAULT_MODULUS_BITS, MAX_MODULUS_BITS, MIN_MODULUS_BITS,
};

pub use pairwise_masking::{ClientKeyPair, ClientPublicKey};

pub use cross_device_manager::{
    CrossDeviceConfig, CrossDevicePrivacyManager, DeviceProfile, DeviceRegistration, DeviceType,
    ParticipationRecord, TemporalEvent, TemporalEventType, DEFAULT_PARTICIPATION_WINDOW_ROUNDS,
    DEFAULT_SUBJECT_EPSILON_BUDGET,
};

pub use composition_analyzer::{
    ClientComposition, CompositionStats, FederatedCompositionAnalyzer, FederatedCompositionMethod,
    RoundComposition,
};

pub use byzantine_aggregation::{
    AdaptivePrivacyAllocation, ByzantineRobustAggregator, ByzantineRobustConfig,
    ByzantineRobustMethod, OutlierDetectionResult, ReputationSystemConfig, RobustEstimators,
    StatisticalAnalyzer, StatisticalTestConfig, StatisticalTestType, TestStatistic,
    OUTLIER_HISTORY_CAPACITY,
};

pub use outlier_tests::OutlierVerdict;
