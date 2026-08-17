//! # OptiRS Learned - Learned Optimizers and Meta-Learning
//!
//! **Version:** 0.3.2
//! **Status:** Research-grade implementations; APIs may still change between releases
//!
//! ⚠️ **Warning:** Learned optimizers are inherently sensitive to the distribution of
//! tasks they were meta-trained on. Benchmark against `optirs-core`'s hand-designed
//! optimizers on your own workload before depending on a learned one in production.
//!
//! `optirs-learned` provides learned optimizers, meta-learning algorithms, and adaptive
//! optimization systems built on [SciRS2](https://github.com/cool-japan/scirs).
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.6.5 - Required foundation
//! - `optirs-core` 0.3.2 - Core optimizers
//!
//! ## Implementation Status (v0.3.2)
//!
//! - ✅ Transformer-based optimizers ([`transformer`], [`transformer_based_optimizer`] -
//!   self-/cross-attention over parameters)
//! - ✅ LSTM optimizers ([`lstm`] - recurrent per-parameter update rule)
//! - ✅ Meta-learning framework ([`meta_learning`] - MAML, Reptile, Meta-SGD)
//! - ✅ Graph-neural-network optimizer ([`gnn_optimizer`]) and Neural Turing Machine
//!   optimizer ([`ntm_optimizer`])
//! - ✅ Differentiable optimizer search ([`darts_optimizer_search`] - DARTS-style)
//! - ✅ Forward/reverse-mode autodiff engines ([`forward_mode`], [`reverse_mode`])
//! - ✅ Continual learning ([`continual_learning`] - EWC, progressive networks), few-shot,
//!   zero-shot and realtime drift adaptation ([`realtime_adaptation`])
//! - 📝 Real, tested algorithms throughout - still labeled research-grade because learned
//!   optimizers carry that distribution-sensitivity caveat, not because they are stubs
//!
//! ## Features
//!
//! ### Transformer-Based Optimizers
//! - **Self-Attention** - Learn optimization patterns across parameters
//! - **Cross-Attention** - Share optimization knowledge between layers
//! - **Positional Encoding** - Parameter-aware optimization
//! - **Multi-Head** - Diverse optimization strategies
//!
//! ### LSTM Optimizers
//! - **Recurrent State** - Maintain long-term optimization memory
//! - **Gating Mechanisms** - Adaptive learning rate control
//! - **Sequence Modeling** - Learn optimization trajectories
//! - **Stateful Updates** - Context-aware parameter updates
//!
//! ### Meta-Learning
//! - **MAML** - Model-Agnostic Meta-Learning
//! - **Reptile** - First-order meta-learning
//! - **Meta-SGD** - Learn learning rates and update rules
//! - **Task Adaptation** - Rapid fine-tuning on new tasks
//!
//! ### Few-Shot Optimization
//! - **Fast Adaptation** - Few-step convergence on new problems
//! - **Transfer Learning** - Knowledge transfer across domains
//! - **Online Learning** - Continuous adaptation during training
//! - **Hypernetworks** - Generate optimizer parameters on-the-fly
//!
//! ## Example Usage (Future)
//!
//! ```rust,ignore
//! use optirs_learned::{TransformerOptimizer, MetaLearningConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create transformer-based optimizer
//! let config = MetaLearningConfig {
//!     num_heads: 8,
//!     hidden_dim: 256,
//!     num_layers: 4,
//! };
//!
//! let mut optimizer = TransformerOptimizer::new(config)?;
//!
//! // Meta-train on multiple tasks
//! for task in tasks {
//!     optimizer.meta_train(&task)?;
//! }
//!
//! // Rapid adaptation to new task
//! let params = Array1::from_elem(1000, 1.0);
//! let grads = Array1::from_elem(1000, 0.01);
//! let updated = optimizer.step(&params, &grads)?;  // Fast convergence
//! ```
//!
//! ## Research Highlights
//!
//! - **Outperforms Hand-Designed** - Better than Adam on many tasks
//! - **Generalizes Across Domains** - Vision, NLP, RL all benefit
//! - **Few-Shot Learning** - Converges in 10-100 steps vs thousands
//! - **Adaptive Schedules** - Learns optimal learning rate schedules
//!
//! ## Architecture
//!
//! Built exclusively on SciRS2:
//! - **ML Pipeline**: `scirs2_core::ml_pipeline::MLPipeline`
//! - **Neural**: `scirs2_core::neural_architecture_search`
//! - **Memory**: `scirs2_core::memory_efficient::LazyArray`
//! - **Metrics**: `scirs2_core::ml_pipeline::PipelineMetrics`
//!
//! ## References
//!
//! - Learning to Learn by Gradient Descent by Gradient Descent (Andrychowicz et al., 2016)
//! - Learned Optimizers that Scale and Generalize (Metz et al., 2022)
//! - VeLO: Training Versatile Learned Optimizers (Metz et al., 2023)
//!
//! ## Contributing
//!
//! Research contributions welcome! Follow SciRS2 integration guidelines.

pub mod adaptive;
pub mod common;
pub mod continual_learning;
pub mod cross_domain_transfer;
pub mod darts_optimizer_search;
pub mod domain_objectives;
pub mod domain_optimizers;
pub mod episodic_memory_impl;
pub mod error;
pub mod es_meta_training;
pub mod few_shot;
pub mod few_shot_impl;
pub mod forward_mode;
pub mod gnn_optimizer;
pub mod higher_order;
pub mod lstm;
pub mod meta_learning;
pub mod ntm_optimizer;
pub mod online_maml;
pub mod quantum_learned;
pub mod realtime_adaptation;
pub mod reverse_mode;
pub mod transformer;
pub mod transformer_based_optimizer;
pub mod zero_shot;

pub use common::{
    LearnedOptimizerConfig, MetaOptimizationStrategy, NeuralOptimizerMetrics, NeuralOptimizerType,
    OptimizerState, StateMetadata, TaskContext, TaskPerformance,
};
pub use continual_learning::{ElasticWeightConsolidation, NetworkColumn, ProgressiveNetworks};
pub use darts_optimizer_search::{
    ClosureObjective, DartsConfig, DartsOptimizerSearch, DifferentiableObjective,
    DiscoveredOptimizer, PrimitiveHyperparams, PrimitiveState, QuadraticBowl, Rosenbrock,
    SearchOutcome, UpdatePrimitive,
};
pub use domain_objectives::{MetaObjective, QuadraticObjective};
pub use error::{OptimError, Result};
pub use es_meta_training::{
    EsMetaTrainer, MetaTrainable, MetaTrainingConfig, MetaTrainingReport, SelectionMetric,
};
pub use forward_mode::{DualNumber, ForwardModeEngine, ForwardModeStats, VectorDual};
pub use gnn_optimizer::{
    GnnOptimizer, GnnOptimizerConfig, GraphTopology, MessageActivation, MessageAggregation,
};
pub use higher_order::{
    HessianConfig, HigherOrderConfig, HigherOrderEngine, HigherOrderStats, HvpMode, LayerInfo,
    LayerType, MixedPartialMethod, MixedPartials, SparseHessian, ThirdOrderTensor,
};
pub use lstm::LSTMOptimizer;
pub use ntm_optimizer::{NtmOptimizer, NtmOptimizerConfig};
pub use quantum_learned::{QuantumBackend, QuantumLearnedOptimizer};
pub use realtime_adaptation::{
    AdaptationDecision, AdaptationReason, DriftDetector, DriftDetectorConfig, DriftSignal,
    RealtimeAdaptationConfig, RealtimeAdaptationController,
};
pub use reverse_mode::{GradientAccumulator, GradientContext, ReverseModeEngine, ReverseModeStats};
pub use transformer::TransformerOptimizer;
pub use transformer_based_optimizer::TransformerOptimizer as TransformerBasedOptimizer;
pub use zero_shot::{
    MetaExample, MetaFeatures, OptimizerHyperparameters, OptimizerKind, OptimizerRecommendation,
    QuadraticProbe, RosenbrockProbe, TaskProbe, ZeroShotConfig, ZeroShotSelector,
};
