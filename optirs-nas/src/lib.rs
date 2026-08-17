//! # OptiRS NAS - Neural Architecture Search
//!
//! **Version:** 0.3.2
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
//! - `scirs2-core` 0.1.1 - Required foundation
//! - `optirs-core` 0.1.0 - Core optimizers
//!
//! ## Implementation Status (v0.1.0)
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
pub mod architecture_embedding;
pub mod architecture_knowledge_graph;
pub mod automl_pipeline;
pub mod cross_domain_transfer;
pub mod domain_specific_nas;
pub mod error;
pub mod evaluation;
pub mod few_shot_architecture;
pub mod hardware_cost;
pub mod hyperparameter;
pub mod multi_objective;
pub mod multimodal_nas;
pub mod nas_engine;
pub mod progressive;
pub mod search_strategies;
pub mod speech_nas;

pub use architecture::ArchitectureSpace;
pub use architecture_embedding::{AggregationMethod, ArchitectureEmbedder};
pub use architecture_knowledge_graph::{
    ArchKnowledgeEdge, ArchKnowledgeNode, ArchitectureKnowledgeGraph, NodeId, PerformanceRecord,
    RelationType,
};
pub use automl_pipeline::{
    AutomlPipelineConfig, AutomlPipelineCoordinator, CandidateModel, EnsembleStrategy,
    FeatureEngineeringStep, PipelineEvaluation, PreprocessingStep, ScoredPipeline, SvmKernel,
    TrainValSplit,
};
pub use cross_domain_transfer::{
    CrossDomainTransferEngine, DomainProfile, DomainSimilarityMetric, DomainSimilarityReport,
    TransferRecommendation, TransferabilityWeights,
};
pub use domain_specific_nas::{
    DomainNASEngine, DomainSearchSpace, DomainType as NASDomainType, NASConstraint,
};
pub use error::{OptimError, Result};
pub use few_shot_architecture::{
    ArchitectureExample, DistanceMetric, FewShotAlgorithm, FewShotArchitectureOptimizer,
    FewShotConfig, FewShotPrediction,
};
pub use hardware_cost::{
    ActivationKind, Bottleneck, CostReport, HardwareCostModel, HardwareProfile, LatencyLookupTable,
    LatencyPrediction, LatencySource, LayerKind, LayerSignature, LayerSpec, PerLayerCost, PoolKind,
    RooflineResult,
};
pub use multimodal_nas::{
    FusionOp, Modality, ModalityEncoder, MultimodalArchitecture, MultimodalEvaluation,
    MultimodalLayer, MultimodalNasEngine, MultimodalSearchSpace, MultimodalValidationError,
};
pub use search_strategies::SearchStrategy;
pub use speech_nas::{
    layer_position_constraint, LayerPositionConstraint, SpeechLayerType, SpeechModelConfig,
    SpeechModelEvaluation, SpeechNasEngine, SpeechSearchSpace,
};

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

// ---------------------------------------------------------------------------
// Canonical result / architecture types
// ---------------------------------------------------------------------------
//
// The generic, element-type-parameterised definitions in
// [`nas_engine::results`] are the single source of truth for architectures,
// evaluation results and resource accounting. They are re-exported here so
// `optirs_nas::OptimizerArchitecture<T>` and
// `nas_engine::results::OptimizerArchitecture<T>` name the *same* type. Earlier
// releases carried a second, `f64`-only copy of each of these structs at the
// crate root; those duplicates are gone.

pub use nas_engine::config::NASConfig;
pub use nas_engine::results::{
    BenchmarkResult, CrossValidationResults, EvaluationResults, OptimizerArchitecture,
    ResourceUsage, TrainingSnapshot,
};

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 32,
            learning_rate: 1e-3,
            performance_prediction: false,
        }
    }
}

impl EvaluationConfig {
    /// Derive an evaluation configuration from the richer, generic
    /// [`nas_engine::config::EvaluationConfig`] carried by [`NASConfig`].
    ///
    /// The engine-level configuration describes budgets and statistical
    /// testing; the evaluation subsystem only needs the step budget, batch size
    /// and base learning rate, plus whether performance prediction is enabled.
    /// `max_epochs` is clamped to at least one step so an evaluation always
    /// performs real work.
    pub fn from_engine_config<T>(config: &nas_engine::config::EvaluationConfig<T>) -> Self
    where
        T: scirs2_core::numeric::Float + std::fmt::Debug + Send + Sync + 'static,
    {
        Self {
            epochs: config.evaluation_budget.max_epochs.max(1) as u32,
            batch_size: 32,
            learning_rate: 1e-3,
            performance_prediction: false,
        }
    }
}
