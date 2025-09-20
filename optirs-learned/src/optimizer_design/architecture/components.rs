// Architecture components and basic building blocks
//
// This module defines the fundamental architectural components used in
// neural architecture search for learned optimizers.

/// Types of neural network layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Linear,
    LSTM,
    GRU,
    Transformer,
    Convolutional1D,
    Attention,
    Recurrent,
    Highway,
    Residual,
    Dense,
    Embedding,
    Custom,
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
    Swish,
    Mish,
    ELU,
    LeakyReLU,
    PReLU,
    Linear,
}

/// Connection patterns between layers
#[derive(Debug, Clone, Copy)]
pub enum ConnectionPattern {
    Sequential,
    Residual,
    DenseNet,
    UNet,
    Attention,
    Recurrent,
    Hybrid,
    Custom,
}

/// Types of attention mechanisms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionType {
    None,
    SelfAttention,
    MultiHeadAttention,
    CrossAttention,
    LocalAttention,
    SparseAttention,
    AdaptiveAttention,
}

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationType {
    None,
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    AdaptiveNorm,
}

/// Optimizer-specific components
#[derive(Debug, Clone, Copy)]
pub enum OptimizerComponent {
    MomentumTracker,
    AdaptiveLearningRate,
    GradientClipping,
    NoiseInjection,
    CurvatureEstimation,
    SecondOrderInfo,
    MetaGradients,
}

/// Memory mechanism types
#[derive(Debug, Clone, Copy)]
pub enum MemoryType {
    None,
    ShortTerm,
    LongTerm,
    Episodic,
    WorkingMemory,
    ExternalMemory,
    AdaptiveMemory,
}

/// Skip connection types
#[derive(Debug, Clone, Copy)]
pub enum SkipConnectionType {
    None,
    Residual,
    Dense,
    Highway,
    Gated,
    Attention,
    Adaptive,
}

/// Memory management strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryManagementStrategy {
    Standard,
    Optimized,
    LowMemory,
}

/// Sparsity patterns
#[derive(Debug, Clone, Copy)]
pub enum SparsityPattern {
    Random,
    Local,
    Strided,
    Block,
    Learned,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Hierarchical,
    ContentAddressable,
    Adaptive,
}

impl Default for LayerType {
    fn default() -> Self {
        LayerType::Linear
    }
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::ReLU
    }
}

impl Default for ConnectionPattern {
    fn default() -> Self {
        ConnectionPattern::Sequential
    }
}

impl Default for AttentionType {
    fn default() -> Self {
        AttentionType::None
    }
}

impl Default for NormalizationType {
    fn default() -> Self {
        NormalizationType::None
    }
}

impl Default for OptimizerComponent {
    fn default() -> Self {
        OptimizerComponent::MomentumTracker
    }
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::None
    }
}

impl Default for SkipConnectionType {
    fn default() -> Self {
        SkipConnectionType::None
    }
}

impl Default for MemoryManagementStrategy {
    fn default() -> Self {
        MemoryManagementStrategy::Standard
    }
}

impl Default for SparsityPattern {
    fn default() -> Self {
        SparsityPattern::Random
    }
}

impl Default for MemoryAccessPattern {
    fn default() -> Self {
        MemoryAccessPattern::Sequential
    }
}

// Component validation and utility methods
impl LayerType {
    /// Check if the layer type supports attention mechanisms
    pub fn supports_attention(&self) -> bool {
        matches!(
            self,
            LayerType::Transformer | LayerType::Attention | LayerType::Custom
        )
    }

    /// Check if the layer type is recurrent
    pub fn is_recurrent(&self) -> bool {
        matches!(self, LayerType::LSTM | LayerType::GRU | LayerType::Recurrent)
    }

    /// Get the typical parameter count multiplier for this layer type
    pub fn parameter_multiplier(&self) -> f64 {
        match self {
            LayerType::Linear => 1.0,
            LayerType::LSTM => 4.0,
            LayerType::GRU => 3.0,
            LayerType::Transformer => 4.0,
            LayerType::Convolutional1D => 0.5,
            LayerType::Attention => 3.0,
            LayerType::Recurrent => 3.0,
            LayerType::Highway => 2.0,
            LayerType::Residual => 1.2,
            LayerType::Dense => 1.0,
            LayerType::Embedding => 0.3,
            LayerType::Custom => 1.0,
        }
    }
}

impl ActivationType {
    /// Check if the activation function is differentiable
    pub fn is_differentiable(&self) -> bool {
        !matches!(self, ActivationType::Linear)
    }

    /// Get the computational cost factor
    pub fn computational_cost(&self) -> f64 {
        match self {
            ActivationType::ReLU => 0.1,
            ActivationType::Tanh => 0.8,
            ActivationType::Sigmoid => 0.7,
            ActivationType::GELU => 1.0,
            ActivationType::Swish => 0.9,
            ActivationType::Mish => 1.1,
            ActivationType::ELU => 0.6,
            ActivationType::LeakyReLU => 0.2,
            ActivationType::PReLU => 0.3,
            ActivationType::Linear => 0.0,
        }
    }

    /// Check if the activation requires learnable parameters
    pub fn has_learnable_parameters(&self) -> bool {
        matches!(self, ActivationType::PReLU)
    }
}

impl ConnectionPattern {
    /// Check if the pattern requires special connectivity handling
    pub fn requires_special_connectivity(&self) -> bool {
        matches!(
            self,
            ConnectionPattern::Residual
                | ConnectionPattern::DenseNet
                | ConnectionPattern::UNet
                | ConnectionPattern::Attention
                | ConnectionPattern::Hybrid
                | ConnectionPattern::Custom
        )
    }

    /// Get the memory overhead factor for this connection pattern
    pub fn memory_overhead(&self) -> f64 {
        match self {
            ConnectionPattern::Sequential => 1.0,
            ConnectionPattern::Residual => 1.1,
            ConnectionPattern::DenseNet => 1.5,
            ConnectionPattern::UNet => 1.3,
            ConnectionPattern::Attention => 1.4,
            ConnectionPattern::Recurrent => 1.2,
            ConnectionPattern::Hybrid => 1.6,
            ConnectionPattern::Custom => 1.0,
        }
    }
}

impl AttentionType {
    /// Get the computational complexity factor
    pub fn complexity_factor(&self) -> f64 {
        match self {
            AttentionType::None => 0.0,
            AttentionType::SelfAttention => 1.0,
            AttentionType::MultiHeadAttention => 1.5,
            AttentionType::CrossAttention => 1.2,
            AttentionType::LocalAttention => 0.8,
            AttentionType::SparseAttention => 0.6,
            AttentionType::AdaptiveAttention => 1.8,
        }
    }

    /// Check if the attention type supports multi-head operation
    pub fn supports_multi_head(&self) -> bool {
        matches!(
            self,
            AttentionType::MultiHeadAttention | AttentionType::AdaptiveAttention
        )
    }
}

impl OptimizerComponent {
    /// Get the computational overhead for this component
    pub fn computational_overhead(&self) -> f64 {
        match self {
            OptimizerComponent::MomentumTracker => 0.1,
            OptimizerComponent::AdaptiveLearningRate => 0.2,
            OptimizerComponent::GradientClipping => 0.05,
            OptimizerComponent::NoiseInjection => 0.08,
            OptimizerComponent::CurvatureEstimation => 0.5,
            OptimizerComponent::SecondOrderInfo => 0.8,
            OptimizerComponent::MetaGradients => 0.3,
        }
    }

    /// Get the memory overhead for this component
    pub fn memory_overhead(&self) -> f64 {
        match self {
            OptimizerComponent::MomentumTracker => 1.0,
            OptimizerComponent::AdaptiveLearningRate => 0.5,
            OptimizerComponent::GradientClipping => 0.1,
            OptimizerComponent::NoiseInjection => 0.0,
            OptimizerComponent::CurvatureEstimation => 2.0,
            OptimizerComponent::SecondOrderInfo => 3.0,
            OptimizerComponent::MetaGradients => 1.5,
        }
    }
}

impl MemoryType {
    /// Get the memory capacity multiplier
    pub fn capacity_multiplier(&self) -> f64 {
        match self {
            MemoryType::None => 0.0,
            MemoryType::ShortTerm => 0.1,
            MemoryType::LongTerm => 1.0,
            MemoryType::Episodic => 0.5,
            MemoryType::WorkingMemory => 0.3,
            MemoryType::ExternalMemory => 2.0,
            MemoryType::AdaptiveMemory => 0.8,
        }
    }

    /// Check if the memory type requires persistence
    pub fn requires_persistence(&self) -> bool {
        matches!(
            self,
            MemoryType::LongTerm | MemoryType::Episodic | MemoryType::ExternalMemory
        )
    }
}

// Compatibility checking functions
pub fn are_components_compatible(
    layer_type: LayerType,
    activation: ActivationType,
    attention: AttentionType,
) -> bool {
    // Check if layer type supports the attention mechanism
    if attention != AttentionType::None && !layer_type.supports_attention() {
        return false;
    }

    // Check activation compatibility with layer type
    match layer_type {
        LayerType::LSTM | LayerType::GRU => {
            // Recurrent layers work better with saturating activations
            matches!(
                activation,
                ActivationType::Tanh | ActivationType::Sigmoid | ActivationType::GELU
            )
        }
        LayerType::Transformer => {
            // Transformers typically use GELU or ReLU
            matches!(
                activation,
                ActivationType::GELU | ActivationType::ReLU | ActivationType::Swish
            )
        }
        _ => true, // Most combinations are valid
    }
}

pub fn estimate_component_cost(
    layer_type: LayerType,
    activation: ActivationType,
    attention: AttentionType,
    memory_type: MemoryType,
) -> f64 {
    let layer_cost = layer_type.parameter_multiplier();
    let activation_cost = activation.computational_cost();
    let attention_cost = attention.complexity_factor();
    let memory_cost = memory_type.capacity_multiplier();

    layer_cost + activation_cost + attention_cost + memory_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_properties() {
        assert!(LayerType::Transformer.supports_attention());
        assert!(!LayerType::Linear.supports_attention());
        assert!(LayerType::LSTM.is_recurrent());
        assert!(!LayerType::Linear.is_recurrent());
    }

    #[test]
    fn test_activation_properties() {
        assert!(ActivationType::ReLU.is_differentiable());
        assert!(!ActivationType::Linear.is_differentiable());
        assert!(ActivationType::PReLU.has_learnable_parameters());
        assert!(!ActivationType::ReLU.has_learnable_parameters());
    }

    #[test]
    fn test_component_compatibility() {
        assert!(are_components_compatible(
            LayerType::Transformer,
            ActivationType::GELU,
            AttentionType::MultiHeadAttention
        ));
        
        assert!(!are_components_compatible(
            LayerType::Linear,
            ActivationType::ReLU,
            AttentionType::SelfAttention
        ));
    }

    #[test]
    fn test_cost_estimation() {
        let cost = estimate_component_cost(
            LayerType::LSTM,
            ActivationType::Tanh,
            AttentionType::None,
            MemoryType::ShortTerm
        );
        assert!(cost > 0.0);
    }
}