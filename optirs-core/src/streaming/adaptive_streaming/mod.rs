// Adaptive Streaming Optimization Module
//
// This module provides comprehensive adaptive streaming optimization for ML workloads.

pub mod anomaly_detection;
pub mod buffering;
pub mod config;
pub mod drift_detection;
pub mod meta_learning;
pub mod optimizer;
pub mod performance;
pub mod resource_management;

// Selective exports to avoid import conflicts
pub use buffering::*;
pub use config::*;
pub use meta_learning::*;
pub use optimizer::*;
pub use resource_management::*;

// Selective re-exports to avoid conflicts
// Anomaly detection module exports
pub use anomaly_detection::{
    AnomalyDetector, AnomalyEvent, AnomalyContext,
    AnomalyDetectionResult, MLModelMetrics,
    EnsembleAnomalyDetector, EnsembleVotingStrategy, DetectorPerformance,
    EnsembleConfig, AdaptiveThresholdManager, ThresholdAdaptationStrategy,
    ThresholdPerformanceFeedback, ThresholdAdaptationParams,
    FalsePositiveEvent, FPRateCalculator, DetectionResult,
    FalsePositivePatterns, TemporalPattern, TemporalPatternType,
    ContextPattern, FPMitigationStrategy, AnomalyResponseSystem,
    ResponseAction, ResponseExecutor, PendingResponse,
    ResponseExecution, ResponsePriority, ResponseResourceLimits,
    ResponseEffectivenessTracker, EffectivenessMetrics,
    ResponseOutcome, OutcomeMeasurement, TrendAnalysis,
    TrendDirection, EscalationRule, EscalationCondition,
    AnomalyType as AnomalyDetectionType,
    AnomalySeverity as AnomalyDetectionSeverity,
    DataStatistics as AnomalyDetectionDataStatistics,
    FalsePositiveTracker as AnomalyDetectionFPTracker
};

// Drift detection module exports
pub use drift_detection::{
    EnhancedDriftDetector, DriftEvent, DriftSeverity, DriftState,
    DriftTestResult, DistributionComparison, ModelDriftResult,
    DriftDiagnostics,
    FalsePositiveTracker as DriftDetectionFPTracker
};

// Performance module exports
pub use performance::{
    PerformanceSnapshot, PerformanceMetric, PerformanceContext,
    PerformanceTracker, PerformanceTrendAnalyzer, TrendData, TrendMethod,
    PerformancePredictor, PredictionMethod, PredictionResult,
    PerformanceImprovementTracker, ImprovementEvent, PlateauDetector,
    PerformanceAnomalyDetector, MetricStatistics, PerformanceAnomaly,
    PerformanceDiagnostics,
    AnomalyType as PerformanceAnomalyType,
    AnomalySeverity as PerformanceAnomalySeverity,
    DataStatistics as PerformanceDataStatistics
};

// Utility functions for common configurations
pub fn create_default_optimizer<A, D>(
) -> StreamingResult<AdaptiveStreamingOptimizer<crate::optimizers::Adam<A>, A, D>>
where
    A: scirs2_core::ndarray_ext::ScalarOperand
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::ops::DivAssign,
    D: scirs2_core::ndarray_ext::Data<Elem = A>
        + scirs2_core::ndarray_ext::Dimension
        + Send
        + Sync
        + 'static,
{
    let config = StreamingConfig::default();
    let base_optimizer = crate::optimizers::Adam::new(A::from(0.001).unwrap()); // Default learning rate
    Ok(AdaptiveStreamingOptimizer::new(base_optimizer, config)?)
}

pub fn create_optimizer_with_config<A, D>(
    config: StreamingConfig,
) -> StreamingResult<AdaptiveStreamingOptimizer<crate::optimizers::Adam<A>, A, D>>
where
    A: scirs2_core::ndarray_ext::ScalarOperand
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Float
        + std::iter::Sum
        + std::fmt::Debug
        + std::ops::DivAssign,
    D: scirs2_core::ndarray_ext::Data<Elem = A>
        + scirs2_core::ndarray_ext::Dimension
        + Send
        + Sync
        + 'static,
{
    let base_optimizer = crate::optimizers::Adam::new(A::from(0.001).unwrap()); // Default learning rate
    Ok(AdaptiveStreamingOptimizer::new(base_optimizer, config)?)
}

// Result type alias
pub type StreamingResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
