//! # OptiRS NAS - Neural Architecture Search
//!
//! **Version:** 0.3.2
//! **Status:** Research-grade implementations; APIs may still change between releases
//!
//! ⚠️ **Warning:** Architecture search is compute-heavy and the search-space / objective
//! APIs are still evolving. Validate a discovered architecture against your own baselines
//! before relying on it in production.
//!
//! `optirs-nas` provides neural architecture search and automated optimizer discovery
//! built on [SciRS2](https://github.com/cool-japan/scirs).
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.6.5 - required foundation (arrays, RNG, numeric traits)
//!
//! This crate has **no intra-workspace dependencies**: `optirs-core` and
//! `optirs-learned` were declared once but never referenced, and were removed so
//! `optirs-nas` builds and tests standalone. The benchmark harness in
//! [`evaluation::benchmark`] implements its optimizer update rules directly, which
//! also makes NAS scores reproducible independently of sibling-crate changes.
//!
//! ## Implementation Status (v0.3.2)
//!
//! - ✅ Bayesian optimization ([`search_strategies::bayesian`] - Gaussian-process surrogate search)
//! - ✅ Hyperparameter search ([`hyperparameter`] - grid enumeration, TPE, a GP-free
//!   kernel-regression surrogate, and evolutionary search)
//! - ✅ Evolutionary algorithms ([`search_strategies::evolutionary`] - population-based search)
//! - ✅ RL-based search ([`search_strategies::rl_search`] - neural-controller sampling)
//! - ✅ Differentiable search ([`search_strategies::differentiable`] - DARTS-style gradient search)
//! - ✅ Multi-objective optimization ([`multi_objective`] - NSGA-II, weighted-sum, Pareto frontier)
//! - ✅ Hardware-aware cost modeling ([`hardware_cost`] - latency/memory/energy estimation)
//! - ✅ AutoML pipeline coordination ([`automl_pipeline`]), cross-domain transfer, few-shot and
//!   progressive search
//! - 📝 Real, tested algorithms throughout - still labeled research-grade because search-space
//!   and objective APIs may change across releases
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
//! ## Example Usage
//!
//! The engine type is [`nas_engine::NeuralArchitectureSearch`], configured with a
//! [`nas_engine::NASConfig`]. This example compiles and runs as a doctest, so it
//! cannot drift away from the real API.
//!
//! ```rust
//! use optirs_nas::nas_engine::{
//!     create_minimal_nas_config, NeuralArchitectureSearch, SearchStrategyType,
//! };
//!
//! # fn main() -> Result<(), optirs_nas::error::OptimError> {
//! // A small, ready-made configuration: a populated search space, a Random
//! // strategy and a tiny budget.
//! let mut config = create_minimal_nas_config::<f64>();
//! config.search_strategy = SearchStrategyType::Evolutionary;
//! config.search_budget = 2;
//! config.population_size = 4;
//!
//! let mut engine = NeuralArchitectureSearch::new(config)?;
//! // The engine reports which strategy is *actually* running, not just what was
//! // requested.
//! assert_eq!(engine.search_strategy_name(), "EvolutionaryStrategy");
//!
//! // A short search: two generations against the built-in evaluator.
//! let results = engine.run_search()?;
//! assert!(!results.search_history.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! Hyperparameter search is independent of the engine and can be driven directly:
//!
//! ```rust
//! use optirs_nas::hyperparameter::{
//!     DistributionType, HyperparameterOptimizer, HyperparameterSpace,
//!     OptimizationStrategy, ParameterRange,
//! };
//!
//! # fn main() -> Result<(), optirs_nas::error::OptimError> {
//! let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
//! space.add_parameter(
//!     "learning_rate".to_string(),
//!     ParameterRange {
//!         name: "learning_rate".to_string(),
//!         min_value: 1e-5,
//!         max_value: 1e-1,
//!         distribution: DistributionType::Uniform,
//!         log_scale: true,
//!         discrete_values: None,
//!     },
//! );
//!
//! // Grid search enumerates the grid; it does not sample at random.
//! let mut optimizer =
//!     HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Grid, 7);
//! optimizer.set_grid_resolution(4);
//! assert_eq!(optimizer.grid_size(), Some(4));
//!
//! let suggestion = optimizer.suggest_configuration()?;
//! let learning_rate = suggestion.parameters["learning_rate"];
//! assert!((1e-5..=1e-1).contains(&learning_rate));
//! # Ok(())
//! # }
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
//! Every algorithm in this crate is implemented here; `scirs2-core` supplies the
//! numeric substrate only:
//!
//! - **Arrays**: `scirs2_core::ndarray` ([`scirs2_core::ndarray::Array1`] / `Array2` / `Array3`)
//! - **Numeric traits**: `scirs2_core::numeric` ([`scirs2_core::numeric::Float`], `NumCast`)
//! - **RNG**: `scirs2_core::random` (`Random`, `Rng`)
//!
//! The search machinery itself lives in this crate:
//! - **Search space**: [`nas_engine::SearchSpaceConfig`]
//! - **Engine**: [`nas_engine::NeuralArchitectureSearch`]
//! - **Strategies**: [`search_strategies`]
//! - **Multi-objective**: [`multi_objective`]
//!
//! (Earlier revisions of this document referenced
//! `scirs2_core::neural_architecture_search` and
//! `scirs2_core::quantum_optimization`; neither module exists.)
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
