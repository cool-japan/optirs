//! # OptiRS NAS - Neural Architecture Search
//!
//! **Version:** 0.1.0-rc.1
//! **Status:** Research Phase (Early Development)
//!
//! ⚠️ **Warning:** This crate is in early research phase. APIs are unstable and may change
//! significantly. Not recommended for production use.
//!
//! `optirs-nas` provides neural architecture search and automated optimizer discovery
//! built on [SciRS2](https://github.com/cool-japan/scirs).
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.1.0-rc.2 - Required foundation
//! - `optirs-core` 0.1.0-rc.1 - Core optimizers
//!
//! ## Implementation Status (RC.1)
//!
//! - 🚧 Bayesian optimization (in development)
//! - 🚧 Evolutionary algorithms (planned)
//! - 🚧 RL-based search (planned)
//! - 🚧 Multi-objective optimization (in development)
//! - 📝 Research framework only
//! - 📝 No production-ready implementations yet
//!
//! ## Status: Research Phase
//!
//! This crate implements state-of-the-art architecture search algorithms.
//!
//! ## Features
//!
//! ### Search Strategies
//! - **Bayesian Optimization** - Gaussian processes for efficient search
//! - **Evolutionary Algorithms** - Population-based architecture evolution
//! - **Reinforcement Learning** - Neural controller for architecture sampling
//! - **Gradient-Based** - DARTS and differentiable NAS
//!
//! ### Multi-Objective Optimization
//! - **Pareto Frontier** - Balance accuracy, speed, memory
//! - **Weighted Sum** - Customizable objective functions
//! - **NSGA-II** - Non-dominated sorting genetic algorithm
//! - **Constraint Satisfaction** - Hardware and resource constraints
//!
//! ### Progressive Search
//! - **Start Simple** - Begin with small architectures
//! - **Gradual Complexity** - Incrementally increase depth/width
//! - **Early Stopping** - Prune unpromising architectures
//! - **Transfer Learning** - Reuse knowledge from previous searches
//!
//! ### Hardware-Aware NAS
//! - **Latency Prediction** - Model inference time on target hardware
//! - **Memory Estimation** - Predict GPU/TPU memory usage
//! - **Energy Consumption** - Optimize for mobile/edge devices
//! - **Cost Optimization** - Minimize cloud compute costs
//!
//! ## Example Usage (Future)
//!
//! ```rust,ignore
//! use optirs_nas::{NASEngine, SearchConfig, SearchStrategy};
//!
//! // Configure architecture search
//! let config = SearchConfig {
//!     search_budget: 1000,
//!     objectives: vec!["accuracy", "latency", "memory"],
//!     strategy: SearchStrategy::BayesianOptimization,
//! };
//!
//! let mut engine = NASEngine::new(config)?;
//!
//! // Define search space
//! let space = engine.optimizer_search_space()?;
//!
//! // Run search
//! let best_architecture = engine.search(&space, &dataset)?;
//!
//! // Use discovered optimizer
//! let optimizer = best_architecture.instantiate()?;
//! ```
//!
//! ## Supported Search Spaces
//!
//! - **Optimizer Selection** - Choose between SGD, Adam, RMSprop, etc.
//! - **Hyperparameters** - Learning rates, momentum, weight decay
//! - **Learning Rate Schedules** - Warmup, decay, cosine annealing
//! - **Regularization** - L1/L2, dropout rates, gradient clipping
//! - **Architecture Components** - Optimizer composition and ensembles
//!
//! ## Performance
//!
//! - **Automated Discovery** - Find better optimizers than hand-tuning
//! - **Hardware-Specific** - Optimized for your exact hardware
//! - **Multi-Objective** - No trade-offs between accuracy and speed
//! - **Generalization** - Architectures transfer across tasks
//!
//! ## Architecture
//!
//! Built exclusively on SciRS2:
//! - **NAS**: `scirs2_core::neural_architecture_search`
//! - **Quantum Opt**: `scirs2_core::quantum_optimization`
//! - **Parallel**: `scirs2_core::parallel::LoadBalancer`
//! - **Search Space**: `scirs2_core::neural_architecture_search::SearchSpace`
//!
//! ## References
//!
//! - DARTS: Differentiable Architecture Search (Liu et al., 2019)
//! - EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)
//! - Once-for-All: Train One Network and Specialize it for Efficient Deployment (Cai et al., 2020)
//!
//! ## Contributing
//!
//! Research contributions welcome! Follow SciRS2 integration guidelines.

pub mod architecture;
pub mod error;
pub mod evaluation;
pub mod hyperparameter;
pub mod multi_objective;
pub mod nas_engine;
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
    /// Use performance prediction
    pub performance_prediction: bool,
}

/// Evaluation metrics for architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationMetric {
    /// Final performance metric
    FinalPerformance,
    /// Convergence speed metric
    ConvergenceSpeed,
    /// Model stability metric
    Stability,
    /// Robustness metric
    Robustness,
    /// Efficiency metric
    Efficiency,
    /// Generalization metric
    Generalization,
    /// Memory usage metric
    MemoryUsage,
    /// Computation time metric
    ComputationTime,
    /// Training stability metric
    TrainingStability,
    /// Memory efficiency metric
    MemoryEfficiency,
    /// Computational efficiency metric
    ComputationalEfficiency,
    /// Accuracy metric
    Accuracy,
    /// Training time metric
    TrainingTime,
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
    /// Metric scores by type
    pub metric_scores: std::collections::HashMap<EvaluationMetric, f64>,
    /// Overall aggregated score
    pub overall_score: f64,
    /// Confidence intervals for metrics
    pub confidence_intervals: std::collections::HashMap<EvaluationMetric, (f64, f64)>,
    /// Time taken for evaluation
    pub evaluation_time: std::time::Duration,
    /// Whether evaluation was successful
    pub success: bool,
    /// Error message if evaluation failed
    pub error_message: Option<String>,
    /// Benchmark results
    pub benchmark_results: Vec<BenchmarkResult>,
    /// Cross-validation results
    pub cv_results: CrossValidationResults,
    /// Training trajectory over time
    pub training_trajectory: Vec<TrainingPoint>,
}

/// Benchmark result for a specific task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Score achieved
    pub score: f64,
    /// Execution time
    pub execution_time: std::time::Duration,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation across folds
    pub std_score: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Number of folds used
    pub num_folds: usize,
}

/// Training point for trajectory tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPoint {
    /// Training step/epoch
    pub step: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: Option<f64>,
    /// Timestamp (elapsed time in seconds since start)
    pub timestamp_secs: f64,
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
    /// Memory usage in GB
    pub memory_gb: f64,
    /// CPU time in seconds
    pub cpu_time_seconds: f64,
    /// GPU time in seconds
    pub gpu_time_seconds: f64,
    /// Energy consumption in kWh
    pub energy_kwh: f64,
    /// Cost in USD
    pub cost_usd: f64,
    /// Network usage in GB
    pub network_gb: f64,
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

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            search_budget: 1000,
            population_size: 50,
            generations: 20,
        }
    }
}
