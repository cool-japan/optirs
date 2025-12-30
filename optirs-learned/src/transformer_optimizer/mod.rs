// Transformer-based Neural Optimizer
//
// This module implements a learned optimizer using Transformer architecture
// to adaptively update optimization parameters. The Transformer leverages
// self-attention mechanisms to capture long-range dependencies in optimization
// trajectories and learn sophisticated optimization strategies.
//
// The system is organized into focused modules:
//
// - `config`: Configuration types and settings (✓ v1.0.0)
// - `network`: Core transformer network architecture (ROADMAP v1.1.0+)
// - `attention`: Attention mechanisms and optimizations (ROADMAP v1.1.0+)
// - `layers`: Neural network layers (feed-forward, layer norm, etc.) (ROADMAP v1.1.0+)
// - `embedding`: Input/output embeddings and positional encoding (ROADMAP v1.1.0+)
// - `sequence`: Sequence buffer and history management (ROADMAP v1.1.0+)
// - `strategy`: Strategy prediction and selection (ROADMAP v1.1.0+)
// - `meta_learning`: Meta-learning components (ROADMAP v1.1.0+)
// - `domain`: Domain adaptation mechanisms (ROADMAP v1.1.0+)
// - `metrics`: Performance tracking and evaluation (ROADMAP v1.1.0+)

pub mod config;

// ROADMAP (v1.1.0+): Additional modules planned for future releases from transformer_optimizer.rs
// pub mod network;      // TransformerNetwork, TransformerLayer, etc.
// pub mod attention;    // MultiHeadAttention, RelativePositionBias, RoPEEmbeddings, etc.
// pub mod layers;       // FeedForwardNetwork, LayerNorm, DropoutLayer, etc.
// pub mod embedding;    // InputEmbedding, OutputProjectionLayer, PositionalEncoder, etc.
// pub mod sequence;     // SequenceBuffer, etc.
// pub mod strategy;     // StrategyPredictor, StrategyNetwork, etc.
// pub mod meta_learning;// TransformerMetaLearner, MetaTrainingEvent, etc.
// pub mod domain;       // DomainAdapter, DomainSpecificAdapter, etc.
// pub mod metrics;      // TransformerOptimizerMetrics, etc.

// Re-export configuration types
pub use config::{
    AttentionOptimization, PositionalEncodingType, TransformerOptimizerConfig,
};

// ROADMAP (v1.1.0+): Additional re-exports once modules are created
// pub use network::*;
// pub use attention::*;
// pub use layers::*;
// pub use embedding::*;
// pub use sequence::*;
// pub use strategy::*;
// pub use meta_learning::*;
// pub use domain::*;
// pub use metrics::*;

// ROADMAP (v1.1.0+): Main TransformerOptimizer struct to be moved here from transformer_optimizer.rs
// once modular refactoring is complete. For v1.0.0, the implementation remains in the parent
// transformer_optimizer.rs file to maintain stability.