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

pub mod error;
pub mod common;
pub mod adaptive;
pub mod few_shot;
pub mod lstm;
pub mod meta_learning;
pub mod transformer;
pub mod transformer_based_optimizer;

pub use error::{OptimError, Result};
pub use common::{
    LearnedOptimizerConfig, MetaOptimizationStrategy, OptimizerState, StateMetadata,
    NeuralOptimizerType, TaskContext, NeuralOptimizerMetrics, TaskPerformance,
};
pub use lstm::LSTMOptimizer;
pub use transformer::TransformerOptimizer;
pub use transformer_based_optimizer::TransformerOptimizer as TransformerBasedOptimizer;
