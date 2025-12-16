//! # OptiRS Learned - Learned Optimizers and Meta-Learning
//!
//! **Version:** 0.1.0-rc.1
//! **Status:** Research Phase (Early Implementation)
//!
//! ⚠️ **Warning:** This crate is in early research phase. APIs may change significantly
//! in future releases. Not recommended for production use.
//!
//! `optirs-learned` provides learned optimizers, meta-learning algorithms, and adaptive
//! optimization systems built on [SciRS2](https://github.com/cool-japan/scirs).
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.1.0-rc.2 - Required foundation
//! - `optirs-core` 0.1.0-rc.1 - Core optimizers
//!
//! ## Implementation Status (RC.1)
//!
//! - 🚧 Transformer-based optimizers (in development)
//! - 🚧 LSTM optimizers (planned)
//! - 🚧 Meta-learning framework (in development)
//! - 📝 Research prototypes only
//! - 📝 No production-ready implementations yet
//!
//! ## Status: Research Phase
//!
//! This crate implements cutting-edge research in learned optimization.
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
