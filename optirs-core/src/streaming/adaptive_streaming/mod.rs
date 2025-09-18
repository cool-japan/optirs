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

// Simplified exports to avoid import conflicts during refactoring
pub use anomaly_detection::*;
pub use buffering::*;
pub use config::*;
pub use drift_detection::*;
pub use meta_learning::*;
pub use optimizer::*;
pub use performance::*;
pub use resource_management::*;

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
