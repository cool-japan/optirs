//! Streaming optimizer types, configs, and coordination structures.
//!
//! Split via SplitRS from a single `types.rs` (COOLJAPAN 2000-line policy).
//! This root stays a thin re-export layer: all real code lives in the
//! private submodules below and every item that was previously public is
//! re-exported here unchanged, so `streaming::types::*` keeps the exact same
//! public surface as before the split.

mod constants;
mod functions;
mod pipeline;
mod prediction;
mod primitives;
mod realtime;
mod types_2;
mod types_3;
mod wiring;

pub use pipeline::{PipelineExecutionManager, StageFunction};
pub use prediction::PredictionModel;
pub use realtime::RealTimeOptimizer;

pub use primitives::{
    CoordinationStrategy, FusedOptimizationStep, FusionStrategy, LearningRateAdaptation,
    LoadBalancingStrategy, MultiStreamCoordinator, PipelineStage, PredictiveStreamingEngine,
    QoSMetric, RTOptimizationResult, RTOptimizationState, RealTimeMetrics, ResourceAllocation,
    ResourceConstraints, ResourceReservationStrategy, ResourceUsage, ServiceLevelObjective,
    StageCoordinator, StageMetrics, StreamConfig, StreamPriority, StreamingConfig,
    StreamingMetrics, DEFAULT_STREAM_ID, STREAM_ID_METADATA_KEY,
};

pub use types_2::{ResourceAllocationStrategy, StreamingDataPoint, StreamingOptimizer};

pub use types_3::{
    AdaptiveResourceManager, AdvancedQoSConfig, AdvancedQoSManager, ConsensusAlgorithm,
    InterruptStrategy, QoSObservation, QoSStatus, QoSViolation, RealTimeConfig,
    StreamFusionOptimizer, StreamingHealthStatus, SyncBarrier,
};
