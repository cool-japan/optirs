//! # OptiRS NAS
//!
//! Neural Architecture Search and hyperparameter optimization for OptiRS.
//!
//! This crate provides:
//! - Neural architecture search algorithms
//! - Hyperparameter optimization
//! - Multi-objective optimization
//! - Progressive search strategies
//! - Architecture encoding and evaluation

pub mod architecture;
pub mod error;
pub mod evaluation;
pub mod hyperparameter;
pub mod multi_objective;
pub mod progressive;
pub mod search_strategies;

pub use architecture::ArchitectureSpace;
pub use error::{OptimError, Result};
pub use search_strategies::SearchStrategy;

// Re-export key types
use serde::{Deserialize, Serialize};

/// Evaluation configuration for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Number of epochs for evaluation
    pub epochs: u32,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate for evaluation
    pub learning_rate: f64,
}

/// Evaluation metrics for architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetric {
    /// Accuracy score
    pub accuracy: f64,
    /// Training time in seconds
    pub training_time: f64,
    /// Model size in parameters
    pub model_size: usize,
}

/// Evaluation results for an architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    /// Final evaluation metrics
    pub metrics: EvaluationMetric,
    /// Training history
    pub history: Vec<f64>,
    /// Validation accuracy
    pub validation_accuracy: f64,
}

/// Optimizer architecture representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerArchitecture {
    /// Architecture identifier
    pub id: String,
    /// Architecture parameters
    pub parameters: std::collections::HashMap<String, f64>,
    /// Architecture structure
    pub structure: Vec<String>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Compute time in seconds
    pub compute_time: f64,
    /// Energy consumption in joules
    pub energy_consumption: f64,
}

/// NAS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Search budget (number of architectures to evaluate)
    pub search_budget: usize,
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
}
