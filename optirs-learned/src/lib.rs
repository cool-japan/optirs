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
//! ### Few-Shot Optimization (`few_shot` module)
//! - **PrototypicalNetwork** - Encode inputs into prototypes for rapid task identification
//! - **FastAdaptationEngine** - Few-step adaptation with dynamic algorithm selection
//! - **TaskSimilarityCalculator** / **EpisodicMemoryBank** - Find and reuse similar past tasks
//! - **OnlineMAML** (`online_maml` module) - Continuous task-stream meta-learning with
//!   staleness decay
//!
//! ## Example Usage
//!
//! Requires the `transformer` feature (on by default). Parameters and gradients are keyed
//! by name, one entry per tensor, matching how a real training loop hands over named
//! layers rather than a single flat array.
//!
//! ```rust,no_run
//! use optirs_learned::transformer::{TransformerOptimizer, TransformerOptimizerConfig};
//! use scirs2_core::ndarray::Array2;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a transformer-based optimizer (default architecture)
//! let config = TransformerOptimizerConfig::default();
//! let mut optimizer = TransformerOptimizer::<f64>::new(config)?;
//!
//! let mut params = HashMap::new();
//! params.insert("layer1.weight".to_string(), Array2::from_elem((4, 4), 1.0));
//!
//! let mut grads = HashMap::new();
//! grads.insert("layer1.weight".to_string(), Array2::from_elem((4, 4), 0.01));
//!
//! // `step` takes the current loss and hands it back, so callers can chain steps
//! let loss = optimizer.step(&mut params, &mut grads, 0.5)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Gates | Default |
//! |---|---|---|
//! | `transformer` | [`adaptive`], [`transformer`], [`transformer_based_optimizer`] modules and the [`TransformerOptimizer`] / `TransformerBasedOptimizer` re-exports | yes |
//! | `lstm` | [`lstm`] module and the [`LSTMOptimizer`] re-export | yes |
//! | `meta_learning` | [`meta_learning`] module (MAML, Reptile, Meta-SGD) | yes |
//!
//! All other modules ([`common`], [`continual_learning`], [`darts_optimizer_search`],
//! [`forward_mode`], [`reverse_mode`], [`gnn_optimizer`], [`higher_order`],
//! [`ntm_optimizer`], [`online_maml`], [`quantum_learned`], [`realtime_adaptation`],
//! [`zero_shot`], `few_shot`, `es_meta_training`, `domain_objectives`, `domain_optimizers`,
//! `episodic_memory_impl`, `cross_domain_transfer`) are ungated and always available.
//! Four integration tests declare `required-features` in `Cargo.toml` (`lstm_layers`,
//! `lstm_meta_training`: `lstm`; `adaptive_enhancement`: `transformer`;
//! `xavier_initialization`: `lstm` + `transformer`), which is why
//! `--no-default-features --features <one family>` is a real, smaller build.
//!
//! ## Architecture
//!
//! Built on [`optirs-core`](https://docs.rs/optirs-core) and `scirs2-core`
//! (`scirs2_core::ndarray`, `scirs2_core::numeric`, `scirs2_core::random`) - no direct
//! `ndarray`/`rand` dependency.
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

/// Adaptive architecture / performance-prediction layer for the transformer
/// optimizers. Written directly against
/// [`transformer_based_optimizer`], so it lives behind the same feature.
#[cfg(feature = "transformer")]
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
#[cfg(feature = "lstm")]
pub mod lstm;
#[cfg(feature = "meta_learning")]
pub mod meta_learning;
pub mod ntm_optimizer;
pub mod online_maml;
pub mod quantum_learned;
pub mod realtime_adaptation;
pub mod reverse_mode;
#[cfg(feature = "transformer")]
pub mod transformer;
#[cfg(feature = "transformer")]
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
#[cfg(feature = "lstm")]
pub use lstm::LSTMOptimizer;
pub use ntm_optimizer::{NtmOptimizer, NtmOptimizerConfig};
pub use quantum_learned::{QuantumBackend, QuantumLearnedOptimizer};
pub use realtime_adaptation::{
    AdaptationDecision, AdaptationReason, DriftDetector, DriftDetectorConfig, DriftSignal,
    RealtimeAdaptationConfig, RealtimeAdaptationController,
};
pub use reverse_mode::{GradientAccumulator, GradientContext, ReverseModeEngine, ReverseModeStats};
#[cfg(feature = "transformer")]
pub use transformer::TransformerOptimizer;
#[cfg(feature = "transformer")]
pub use transformer_based_optimizer::TransformerOptimizer as TransformerBasedOptimizer;
pub use zero_shot::{
    MetaExample, MetaFeatures, OptimizerHyperparameters, OptimizerKind, OptimizerRecommendation,
    QuadraticProbe, RosenbrockProbe, TaskProbe, ZeroShotConfig, ZeroShotSelector,
};
