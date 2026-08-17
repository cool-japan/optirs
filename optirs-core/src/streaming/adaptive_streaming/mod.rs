// Adaptive Streaming Optimization Module
//
// This module provides comprehensive adaptive streaming optimization for ML workloads.

pub mod anomaly_detection;
pub mod anomaly_ml;
pub mod anomaly_statistical;
pub mod buffering;
pub mod config;
pub mod drift_detection;
pub mod drift_tests;
pub mod meta_bandit;
pub mod meta_learning;
pub mod meta_transfer;
pub mod optimizer;
pub mod performance;
pub mod resource_management;
pub mod statistics;

#[cfg(test)]
mod config_wiring_tests;

// NOTE: `anomaly_ml`, `anomaly_statistical`, `drift_tests`, `meta_bandit`
// and `statistics` are deliberately NOT glob-re-exported. The glob exports below already
// collide across modules (see the aliased re-exports further down), and
// adding four more globs would reintroduce ambiguous names for every
// downstream consumer. Reach for them through their module path instead.

// Selective exports to avoid import conflicts
pub use buffering::*;
pub use config::*;
pub use meta_learning::*;
pub use optimizer::*;
pub use resource_management::*;

// Selective re-exports to avoid conflicts
// Anomaly detection module exports
pub use anomaly_detection::{
    AdaptiveThresholdManager, AnomalyContext, AnomalyDetectionResult, AnomalyDetector,
    AnomalyEvent, AnomalyResponseSystem, AnomalySeverity as AnomalyDetectionSeverity,
    AnomalyType as AnomalyDetectionType, ContextPattern,
    DataStatistics as AnomalyDetectionDataStatistics, DetectionResult, DetectorPerformance,
    EffectivenessMetrics, EnsembleAnomalyDetector, EnsembleConfig, EnsembleVotingStrategy,
    EscalationCondition, EscalationRule, FPMitigationStrategy, FPRateCalculator,
    FalsePositiveEvent, FalsePositivePatterns, FalsePositiveTracker as AnomalyDetectionFPTracker,
    MLModelMetrics, OutcomeMeasurement, PendingResponse, ResponseAction, ResponseExecution,
    ResponseExecutor, ResponseOutcome, ResponsePriority, ResponseResourceLimits, TemporalPattern,
    TemporalPatternType, ThresholdAdaptationParams, ThresholdAdaptationStrategy,
    ThresholdPerformanceFeedback, TrendAnalysis, TrendDirection,
};

// Drift detection module exports
pub use drift_detection::{
    DistributionComparison, DriftDiagnostics, DriftEvent, DriftSeverity, DriftState,
    DriftTestResult, EnhancedDriftDetector, FalsePositiveTracker as DriftDetectionFPTracker,
    ModelDriftResult,
};

// Performance module exports
pub use performance::{
    AnomalySeverity as PerformanceAnomalySeverity, AnomalyType as PerformanceAnomalyType,
    DataStatistics as PerformanceDataStatistics, ImprovementEvent, MetricStatistics,
    PerformanceAnomaly, PerformanceAnomalyDetector, PerformanceContext, PerformanceDiagnostics,
    PerformanceImprovementTracker, PerformanceMetric, PerformancePredictor, PerformanceSnapshot,
    PerformanceTracker, PerformanceTrendAnalyzer, PlateauDetector, PredictionMethod,
    PredictionResult, TrendData, TrendMethod,
};

// Utility functions for common configurations
pub fn create_default_optimizer<A, D>(
) -> StreamingResult<AdaptiveStreamingOptimizer<crate::optimizers::Adam<A>, A, D>>
where
    A: scirs2_core::ndarray::ScalarOperand
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::numeric::Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::ops::DivAssign,
    // `Data` is ndarray's *storage* trait, not a dimension trait: no type
    // implements both it and `Dimension`, so this bound was unsatisfiable and
    // neither factory could ever be instantiated by any caller.
    D: scirs2_core::ndarray::Dimension + Send + Sync + 'static,
{
    let config = StreamingConfig::default();
    let default_learning_rate = A::from(DEFAULT_LEARNING_RATE).ok_or_else(|| {
        format!("element type cannot represent the default learning rate {DEFAULT_LEARNING_RATE}")
    })?;
    let base_optimizer = crate::optimizers::Adam::new(default_learning_rate);
    Ok(AdaptiveStreamingOptimizer::new(base_optimizer, config)?)
}

pub fn create_optimizer_with_config<A, D>(
    config: StreamingConfig,
) -> StreamingResult<AdaptiveStreamingOptimizer<crate::optimizers::Adam<A>, A, D>>
where
    A: scirs2_core::ndarray::ScalarOperand
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + scirs2_core::numeric::Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::ops::DivAssign,
    // `Data` is ndarray's *storage* trait, not a dimension trait: no type
    // implements both it and `Dimension`, so this bound was unsatisfiable and
    // neither factory could ever be instantiated by any caller.
    D: scirs2_core::ndarray::Dimension + Send + Sync + 'static,
{
    let default_learning_rate = A::from(DEFAULT_LEARNING_RATE).ok_or_else(|| {
        format!("element type cannot represent the default learning rate {DEFAULT_LEARNING_RATE}")
    })?;
    let base_optimizer = crate::optimizers::Adam::new(default_learning_rate);
    Ok(AdaptiveStreamingOptimizer::new(base_optimizer, config)?)
}

/// Learning rate used by the convenience constructors above when the caller
/// does not supply one.
pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

// Result type alias
pub type StreamingResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
