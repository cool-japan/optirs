// Architecture search space definition
//
// This module defines the search space for neural architecture search,
// including available operations, constraints, and sampling methods.

use std::collections::HashMap;

use super::architecture::*;

/// Architecture search space definition
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Layer types available
    pub layer_types: Vec<LayerType>,

    /// Hidden size options
    pub hidden_sizes: Vec<usize>,

    /// Number of layers range
    pub num_layers_range: (usize, usize),

    /// Activation functions
    pub activation_functions: Vec<ActivationType>,

    /// Connection patterns
    pub connection_patterns: Vec<ConnectionPattern>,

    /// Attention mechanisms
    pub attention_mechanisms: Vec<AttentionType>,

    /// Normalization options
    pub normalization_options: Vec<NormalizationType>,

    /// Optimization components
    pub optimizer_components: Vec<OptimizerComponent>,

    /// Memory mechanisms
    pub memory_mechanisms: Vec<MemoryType>,

    /// Skip connection options
    pub skip_connections: Vec<SkipConnectionType>,
}

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            layer_types: vec![
                LayerType::Linear,
                LayerType::LSTM,
                LayerType::GRU,
                LayerType::Transformer,
                LayerType::Attention,
                LayerType::Dense,
            ],
            hidden_sizes: vec![32, 64, 128, 256, 512],
            num_layers_range: (1, 10),
            activation_functions: vec![
                ActivationType::ReLU,
                ActivationType::GELU,
                ActivationType::Tanh,
                ActivationType::Sigmoid,
                ActivationType::Swish,
            ],
            connection_patterns: vec![
                ConnectionPattern::Sequential,
                ConnectionPattern::Residual,
                ConnectionPattern::DenseNet,
                ConnectionPattern::Attention,
            ],
            attention_mechanisms: vec![
                AttentionType::None,
                AttentionType::SelfAttention,
                AttentionType::MultiHeadAttention,
                AttentionType::LocalAttention,
            ],
            normalization_options: vec![
                NormalizationType::None,
                NormalizationType::LayerNorm,
                NormalizationType::BatchNorm,
            ],
            optimizer_components: vec![
                OptimizerComponent::MomentumTracker,
                OptimizerComponent::AdaptiveLearningRate,
                OptimizerComponent::GradientClipping,
            ],
            memory_mechanisms: vec![
                MemoryType::None,
                MemoryType::ShortTerm,
                MemoryType::LongTerm,
                MemoryType::WorkingMemory,
            ],
            skip_connections: vec![
                SkipConnectionType::None,
                SkipConnectionType::Residual,
                SkipConnectionType::Dense,
                SkipConnectionType::Highway,
            ],
        }
    }
}

impl ArchitectureSearchSpace {
    /// Get total search space size (approximate)
    pub fn search_space_size(&self) -> u64 {
        let layer_combinations = self.layer_types.len().pow(self.num_layers_range.1 as u32) as u64;
        let hidden_size_combinations = self.hidden_sizes.len().pow(self.num_layers_range.1 as u32) as u64;
        let activation_combinations = self.activation_functions.len().pow(self.num_layers_range.1 as u32) as u64;
        
        layer_combinations * hidden_size_combinations * activation_combinations
    }

    /// Sample random architecture from search space
    pub fn sample_random(&self) -> ArchitectureSpec {
        use scirs2_core::random::seq::SliceRandom;
        
        let num_layers = scirs2_core::random::random::<usize>() % (self.num_layers_range.1 - self.num_layers_range.0 + 1) + self.num_layers_range.0;
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let layer_type = *self.layer_types.choose(&mut scirs2_core::random::thread_rng()).unwrap_or(&LayerType::Linear);
            let hidden_size = *self.hidden_sizes.choose(&mut scirs2_core::random::thread_rng()).unwrap_or(&128);
            let activation = *self.activation_functions.choose(&mut scirs2_core::random::thread_rng()).unwrap_or(&ActivationType::ReLU);

            let dimensions = LayerDimensions {
                input_dim: if i == 0 { hidden_size } else { layers[i-1].dimensions.output_dim },
                output_dim: hidden_size,
                hidden_dims: vec![],
            };

            layers.push(LayerSpec::new(layer_type, dimensions, activation));
        }

        ArchitectureSpec::new(layers, GlobalArchitectureConfig::default())
    }
}