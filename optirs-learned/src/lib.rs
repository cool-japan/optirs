//! # OptiRS Learned
//!
//! Learned optimizers and meta-learning for OptiRS.
//!
//! This crate provides:
//! - Transformer-based optimizers
//! - LSTM optimizers
//! - Meta-learning algorithms
//! - Few-shot optimization
//! - Adaptive enhancement systems

pub mod adaptive;
pub mod common;
pub mod error;
pub mod few_shot;
pub mod lstm;
pub mod meta_learning;
pub mod transformer;
pub mod transformer_based_optimizer;

pub use common::{
    LearnedOptimizerConfig, MetaOptimizationStrategy, NeuralOptimizerMetrics, NeuralOptimizerType,
    OptimizerState, StateMetadata, TaskContext, TaskPerformance,
};
pub use error::{OptimError, Result};
pub use lstm::LSTMOptimizer;
pub use transformer::TransformerOptimizer;
pub use transformer_based_optimizer::TransformerOptimizer as TransformerBasedOptimizer;
