// Configuration types for the Transformer-based Neural Optimizer
//
// This module defines configuration structures and enums that control
// the behavior of the Transformer optimizer including architecture settings,
// attention mechanisms, and optimization strategies.

use super::super::{LearnedOptimizerConfig, MetaOptimizationStrategy};

/// Configuration specific to Transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerOptimizerConfig {
    /// Base learned optimizer config
    pub base_config: LearnedOptimizerConfig,

    /// Model dimension (d_model)
    pub modeldim: usize,

    /// Number of attention heads
    pub numheads: usize,

    /// Feed-forward network dimension
    pub ff_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Attention dropout rate
    pub attention_dropout: f64,

    /// Feed-forward dropout rate
    pub ff_dropout: f64,

    /// Layer normalization epsilon
    pub layer_norm_eps: f64,

    /// Use pre-layer normalization
    pub pre_layer_norm: bool,

    /// Positional encoding type
    pub pos_encoding_type: PositionalEncodingType,

    /// Enable relative position bias
    pub relative_position_bias: bool,

    /// Use rotary position embedding
    pub use_rope: bool,

    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,

    /// Attention pattern optimization
    pub attention_optimization: AttentionOptimization,

    /// Multi-scale attention
    pub multi_scale_attention: bool,

    /// Cross-attention for multi-task learning
    pub cross_attention: bool,

    /// Memory efficiency mode
    pub memory_efficient: bool,
}

/// Types of positional encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Sinusoidal position encoding
    Sinusoidal,

    /// Learned position embedding
    Learned,

    /// Rotary position embedding (RoPE)
    Rotary,

    /// Relative position encoding
    Relative,

    /// ALiBi (Attention with Linear Biases)
    ALiBi,
}

/// Attention optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionOptimization {
    /// Standard full attention
    Full,

    /// Sparse attention patterns
    Sparse,

    /// Linear attention approximation
    Linear,

    /// Local attention windows
    Local,

    /// Hierarchical attention
    Hierarchical,

    /// Adaptive attention sparsity
    Adaptive,
}

impl Default for TransformerOptimizerConfig {
    fn default() -> Self {
        Self {
            base_config: LearnedOptimizerConfig::default(),
            modeldim: 512,
            numheads: 8,
            ff_dim: 2048,
            num_layers: 6,
            max_sequence_length: 1024,
            attention_dropout: 0.1,
            ff_dropout: 0.1,
            layer_norm_eps: 1e-5,
            pre_layer_norm: true,
            pos_encoding_type: PositionalEncodingType::Sinusoidal,
            relative_position_bias: false,
            use_rope: false,
            gradient_checkpointing: false,
            attention_optimization: AttentionOptimization::Full,
            multi_scale_attention: false,
            cross_attention: false,
            memory_efficient: false,
        }
    }
}

impl Default for PositionalEncodingType {
    fn default() -> Self {
        Self::Sinusoidal
    }
}

impl Default for AttentionOptimization {
    fn default() -> Self {
        Self::Full
    }
}