// Streaming optimization for real-time learning
//
// This module provides streaming gradient descent and other online optimization
// algorithms designed for real-time data processing and low-latency inference.

// Existing modules
pub mod adaptive_streaming;
pub mod concept_drift;
pub mod enhanced_adaptive_lr;
pub mod low_latency;
pub mod streaming_metrics;

// New split modules from original mod.rs content
mod advancedqosconfig_traits;
mod functions;
mod realtimeconfig_traits;
mod streamingconfig_traits;
mod streamingmetrics_traits;
mod types;

// Re-export key types from existing modules
pub use concept_drift::{ConceptDriftDetector, DriftDetectorConfig, DriftEvent, DriftStatus};
pub use enhanced_adaptive_lr::{
    AdaptationStatistics, AdaptiveLRConfig, EnhancedAdaptiveLRController,
};
pub use low_latency::{LowLatencyConfig, LowLatencyMetrics, LowLatencyOptimizer};
pub use streaming_metrics::{MetricsSample, MetricsSummary, StreamingMetricsCollector};

// Re-export split module types
pub use advancedqosconfig_traits::*;
pub use functions::*;
pub use realtimeconfig_traits::*;
pub use streamingconfig_traits::*;
pub use streamingmetrics_traits::*;
pub use types::*;
