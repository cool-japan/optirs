//! Byzantine Fault Tolerance for Federated Learning
//!
//! This module implements Byzantine-robust aggregation algorithms that can
//! tolerate malicious participants in federated learning scenarios.
//!
//! # Determinism
//!
//! Every algorithm in this module orders its cohort by participant id before doing
//! any work, so results never depend on `HashMap` iteration order. The two
//! randomised components (FLAME's calibrated noise and the isolation-forest outlier
//! detector) draw from an internal, deterministically seeded SplitMix64 generator.
//!
//! # Threat model notes
//!
//! The FLAME noise term is a *backdoor mitigation* measure taken from Nguyen et al.,
//! 2022. It is **not** a differential-privacy mechanism and provides no formal
//! privacy guarantee; use [`crate::privacy`]'s differential privacy machinery for
//! that.

mod aggregation;
mod aggregator;
mod foolsgold;
mod helpers;
mod outlier;
mod scoring;
mod types;

pub use aggregator::ByzantineTolerantAggregator;
pub use types::{
    AnomalyDetector, AnomalyScore, BehaviorHistory, ByzantineAggregationMethod,
    ByzantineAggregationResult, ByzantineConfig, GradientProperties, GradientStatistics,
    GradientVerifier, OutlierDetectionMethod, OutlierScore, PatternModel, ReputationScore,
    StatisticalAnalysis, StatisticalMeasures, TrustLevel, VerificationRule, VerificationScore,
};

#[cfg(test)]
mod tests;
