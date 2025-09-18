// Optimization strategies for transformer-based learned optimizers
//
// This module contains various optimization strategies that work in conjunction
// with the transformer architecture to improve optimization performance.

pub mod gradient_processing;
pub mod learning_rate_adaptation;
pub mod momentum_integration;
pub mod regularization;

// Re-export key types for convenience
pub use gradient_processing::{
    GradientProcessingParams, GradientProcessingStrategy, GradientProcessor, GradientStatistics,
};
pub use learning_rate_adaptation::{
    LRAdaptationParams, LearningRateAdaptationStrategy, LearningRateAdapter, ScheduleState,
};
pub use momentum_integration::{
    MomentumIntegrator, MomentumParams, MomentumState, MomentumStatistics, MomentumStrategy,
};
pub use regularization::{
    ParameterStatistics, RegularizationParams, RegularizationStrategy, TransformerRegularizer,
};
