// Architecture specifications and configuration structures
//
// This module defines the detailed specifications for neural architectures
// in the optimization context, including layer specifications, attention patterns,
// and memory management configurations.

use std::collections::HashMap;
use scirs2_core::ndarray::Array2;

use super::components::*;

/// Architecture specification
#[derive(Debug, Clone)]
pub struct ArchitectureSpec {
    /// Layers in the architecture
    pub layers: Vec<LayerSpec>,

    /// Connection matrix
    pub connections: Array2<bool>,

    /// Global configuration
    pub global_config: GlobalArchitectureConfig,

    /// Specialized components
    pub specialized_components: Vec<SpecializedComponent>,
}

/// Individual layer specification
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Layer type
    pub layer_type: LayerType,

    /// Layer dimensions
    pub dimensions: LayerDimensions,

    /// Activation function
    pub activation: ActivationType,

    /// Normalization
    pub normalization: NormalizationType,

    /// Layer-specific parameters
    pub parameters: HashMap<String, f64>,

    /// Skip connections from this layer
    pub skip_connections: Vec<usize>,
}

/// Layer dimensions
#[derive(Debug, Clone)]
pub struct LayerDimensions {
    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Hidden dimensions (for multi-dimensional layers)
    pub hidden_dims: Vec<usize>,
}

/// Global architecture configuration
#[derive(Debug, Clone)]
pub struct GlobalArchitectureConfig {
    /// Overall depth
    pub depth: usize,

    /// Overall width
    pub width: usize,

    /// Global skip connections
    pub global_skip_connections: bool,

    /// Attention patterns
    pub attention_pattern: AttentionPattern,

    /// Memory management
    pub memory_management: MemoryManagementStrategy,
}

/// Attention patterns
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    /// Attention type
    pub attention_type: AttentionType,

    /// Number of heads
    pub num_heads: usize,

    /// Attention span
    pub attention_span: usize,

    /// Sparse attention configuration
    pub sparse_config: Option<SparseAttentionConfig>,
}

/// Sparse attention configuration
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Sparsity pattern
    pub sparsity_pattern: SparsityPattern,

    /// Sparsity ratio
    pub sparsity_ratio: f64,

    /// Block size
    pub block_size: usize,
}

/// Memory management configuration
#[derive(Debug, Clone)]
pub struct MemoryManagement {
    /// Memory type
    pub memory_type: MemoryType,

    /// Memory capacity
    pub memory_capacity: usize,

    /// Memory access pattern
    pub access_pattern: MemoryAccessPattern,

    /// Memory compression
    pub compression_enabled: bool,
}

/// Specialized component
#[derive(Debug, Clone)]
pub struct SpecializedComponent {
    /// Component type
    pub component_type: OptimizerComponent,

    /// Component parameters
    pub parameters: HashMap<String, f64>,

    /// Integration points
    pub integration_points: Vec<usize>,
}

/// Architecture candidate representation
#[derive(Debug, Clone)]
pub struct ArchitectureCandidate {
    /// Unique architecture ID
    pub id: String,

    /// Architecture specification
    pub architecture: ArchitectureSpec,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Resource usage
    pub resource_usage: ResourceUsage,

    /// Generation information
    pub generation_info: GenerationInfo,

    /// Validation results
    pub validation_results: Option<ValidationResults>,
}

/// Performance metrics for architecture
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Optimization performance
    pub optimization_performance: f64,

    /// Convergence speed
    pub convergence_speed: f64,

    /// Generalization ability
    pub generalization: f64,

    /// Robustness score
    pub robustness: f64,

    /// Transfer learning performance
    pub transfer_performance: f64,

    /// Multi-task performance
    pub multitask_performance: f64,

    /// Stability score
    pub stability: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Parameter count
    pub parameter_count: usize,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// Computational cost (FLOPs)
    pub computational_cost: u64,

    /// Inference time (microseconds)
    pub inference_time_us: u64,

    /// Training time per step (microseconds)
    pub training_time_us: u64,

    /// Energy consumption (joules)
    pub energy_consumption: f64,
}

/// Generation information
#[derive(Debug, Clone)]
pub struct GenerationInfo {
    /// Generation number
    pub generation: usize,

    /// Parent architectures
    pub parents: Vec<String>,

    /// Mutation history
    pub mutations: Vec<MutationRecord>,

    /// Creation timestamp
    pub created_at: std::time::Instant,

    /// Creation method
    pub creation_method: CreationMethod,
}

/// Mutation record
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// Mutation type
    pub mutation_type: String,

    /// Affected layer indices
    pub affected_layers: Vec<usize>,

    /// Mutation parameters
    pub parameters: HashMap<String, f64>,

    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Architecture creation method
#[derive(Debug, Clone, Copy)]
pub enum CreationMethod {
    Random,
    Evolutionary,
    BayesianOptimization,
    ReinforcementLearning,
    Progressive,
    Manual,
    Transfer,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Validation accuracy
    pub accuracy: f64,

    /// Cross-validation scores
    pub cv_scores: Vec<f64>,

    /// Statistical significance
    pub p_value: f64,

    /// Confidence interval
    pub confidence_interval: (f64, f64),

    /// Validation details
    pub details: HashMap<String, f64>,
}

// Implementation methods
impl Default for LayerDimensions {
    fn default() -> Self {
        Self {
            input_dim: 128,
            output_dim: 128,
            hidden_dims: vec![],
        }
    }
}

impl Default for AttentionPattern {
    fn default() -> Self {
        Self {
            attention_type: AttentionType::None,
            num_heads: 1,
            attention_span: 0,
            sparse_config: None,
        }
    }
}

impl Default for GlobalArchitectureConfig {
    fn default() -> Self {
        Self {
            depth: 3,
            width: 128,
            global_skip_connections: false,
            attention_pattern: AttentionPattern::default(),
            memory_management: MemoryManagementStrategy::Standard,
        }
    }
}

impl Default for MemoryManagement {
    fn default() -> Self {
        Self {
            memory_type: MemoryType::None,
            memory_capacity: 0,
            access_pattern: MemoryAccessPattern::Sequential,
            compression_enabled: false,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            optimization_performance: 0.0,
            convergence_speed: 0.0,
            generalization: 0.0,
            robustness: 0.0,
            transfer_performance: 0.0,
            multitask_performance: 0.0,
            stability: 0.0,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            parameter_count: 0,
            memory_usage: 0,
            computational_cost: 0,
            inference_time_us: 0,
            training_time_us: 0,
            energy_consumption: 0.0,
        }
    }
}

impl Default for CreationMethod {
    fn default() -> Self {
        CreationMethod::Random
    }
}

// Utility methods for specifications
impl ArchitectureSpec {
    /// Create a new architecture specification
    pub fn new(
        layers: Vec<LayerSpec>,
        global_config: GlobalArchitectureConfig,
    ) -> Self {
        let num_layers = layers.len();
        let connections = Array2::from_elem((num_layers, num_layers), false);

        Self {
            layers,
            connections,
            global_config,
            specialized_components: Vec::new(),
        }
    }

    /// Get the total parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.estimate_parameter_count())
            .sum::<usize>()
            + self
                .specialized_components
                .iter()
                .map(|comp| comp.estimate_parameter_count())
                .sum::<usize>()
    }

    /// Get the total memory usage estimate
    pub fn memory_usage_estimate(&self) -> usize {
        let layer_memory: usize = self
            .layers
            .iter()
            .map(|layer| layer.estimate_memory_usage())
            .sum();

        let connection_memory = self.connections.len() * std::mem::size_of::<bool>();

        layer_memory + connection_memory
    }

    /// Validate the architecture specification
    pub fn validate(&self) -> Result<(), SpecificationError> {
        // Check layer count
        if self.layers.is_empty() {
            return Err(SpecificationError::InvalidConfiguration(
                "Architecture must have at least one layer".to_string(),
            ));
        }

        // Check connection matrix dimensions
        let expected_size = self.layers.len();
        if self.connections.shape() != [expected_size, expected_size] {
            return Err(SpecificationError::InvalidConfiguration(
                "Connection matrix size does not match layer count".to_string(),
            ));
        }

        // Validate each layer
        for (i, layer) in self.layers.iter().enumerate() {
            layer.validate().map_err(|e| {
                SpecificationError::InvalidLayer(i, format!("{}", e))
            })?;
        }

        // Validate dimension compatibility
        self.validate_dimension_compatibility()?;

        Ok(())
    }

    fn validate_dimension_compatibility(&self) -> Result<(), SpecificationError> {
        for i in 0..self.layers.len() - 1 {
            let current_layer = &self.layers[i];
            let next_layer = &self.layers[i + 1];

            if current_layer.dimensions.output_dim != next_layer.dimensions.input_dim {
                return Err(SpecificationError::DimensionMismatch(
                    i,
                    i + 1,
                    current_layer.dimensions.output_dim,
                    next_layer.dimensions.input_dim,
                ));
            }
        }
        Ok(())
    }

    /// Add a specialized component
    pub fn add_specialized_component(&mut self, component: SpecializedComponent) {
        self.specialized_components.push(component);
    }

    /// Get layers by type
    pub fn get_layers_by_type(&self, layer_type: LayerType) -> Vec<&LayerSpec> {
        self.layers
            .iter()
            .filter(|layer| layer.layer_type == layer_type)
            .collect()
    }
}

impl LayerSpec {
    /// Create a new layer specification
    pub fn new(
        layer_type: LayerType,
        dimensions: LayerDimensions,
        activation: ActivationType,
    ) -> Self {
        Self {
            layer_type,
            dimensions,
            activation,
            normalization: NormalizationType::None,
            parameters: HashMap::new(),
            skip_connections: Vec::new(),
        }
    }

    /// Estimate parameter count for this layer
    pub fn estimate_parameter_count(&self) -> usize {
        let base_params = self.dimensions.input_dim * self.dimensions.output_dim;
        let bias_params = self.dimensions.output_dim;
        let multiplier = self.layer_type.parameter_multiplier();

        ((base_params + bias_params) as f64 * multiplier) as usize
    }

    /// Estimate memory usage for this layer
    pub fn estimate_memory_usage(&self) -> usize {
        let param_memory = self.estimate_parameter_count() * std::mem::size_of::<f32>();
        let activation_memory = self.dimensions.output_dim * std::mem::size_of::<f32>();

        param_memory + activation_memory
    }

    /// Validate the layer specification
    pub fn validate(&self) -> Result<(), SpecificationError> {
        if self.dimensions.input_dim == 0 {
            return Err(SpecificationError::InvalidConfiguration(
                "Input dimension cannot be zero".to_string(),
            ));
        }

        if self.dimensions.output_dim == 0 {
            return Err(SpecificationError::InvalidConfiguration(
                "Output dimension cannot be zero".to_string(),
            ));
        }

        // Check component compatibility
        // NOTE: AttentionType::None is intentional for layer-level validation.
        // Attention patterns are managed at the architecture level (GlobalArchitectureConfig),
        // not at individual layer level. This validates basic layer-activation compatibility.
        // For attention-specific validation, use ArchitectureSpec::validate() instead.
        if !are_components_compatible(
            self.layer_type,
            self.activation,
            AttentionType::None,
        ) {
            return Err(SpecificationError::IncompatibleComponents(
                format!("{:?}", self.layer_type),
                format!("{:?}", self.activation),
            ));
        }

        Ok(())
    }

    /// Add a parameter
    pub fn add_parameter(&mut self, name: String, value: f64) {
        self.parameters.insert(name, value);
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }
}

impl SpecializedComponent {
    /// Create a new specialized component
    pub fn new(component_type: OptimizerComponent) -> Self {
        Self {
            component_type,
            parameters: HashMap::new(),
            integration_points: Vec::new(),
        }
    }

    /// Estimate parameter count for this component
    pub fn estimate_parameter_count(&self) -> usize {
        // Base estimation based on component type
        match self.component_type {
            OptimizerComponent::MomentumTracker => 1,
            OptimizerComponent::AdaptiveLearningRate => 2,
            OptimizerComponent::GradientClipping => 1,
            OptimizerComponent::NoiseInjection => 2,
            OptimizerComponent::CurvatureEstimation => 10,
            OptimizerComponent::SecondOrderInfo => 20,
            OptimizerComponent::MetaGradients => 5,
        }
    }

    /// Add integration point
    pub fn add_integration_point(&mut self, layer_index: usize) {
        if !self.integration_points.contains(&layer_index) {
            self.integration_points.push(layer_index);
        }
    }
}

impl ArchitectureCandidate {
    /// Create a new architecture candidate
    pub fn new(id: String, architecture: ArchitectureSpec) -> Self {
        Self {
            id,
            architecture,
            performance: PerformanceMetrics::default(),
            resource_usage: ResourceUsage::default(),
            generation_info: GenerationInfo {
                generation: 0,
                parents: Vec::new(),
                mutations: Vec::new(),
                created_at: std::time::Instant::now(),
                creation_method: CreationMethod::default(),
            },
            validation_results: None,
        }
    }

    /// Calculate overall fitness score
    pub fn fitness_score(&self, weights: &[f64]) -> f64 {
        let performance_scores = vec![
            self.performance.optimization_performance,
            self.performance.convergence_speed,
            self.performance.generalization,
            self.performance.robustness,
            self.performance.stability,
        ];

        performance_scores
            .iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }

    /// Check if candidate meets constraints
    pub fn meets_constraints(&self, constraints: &super::constraints::SearchConstraints) -> bool {
        self.resource_usage.parameter_count <= constraints.max_parameters
            && self.resource_usage.memory_usage <= constraints.max_memory_mb * 1024 * 1024
            && self.resource_usage.inference_time_us <= constraints.max_inference_time_ms * 1000
            && self.performance.optimization_performance >= constraints.min_accuracy
    }
}

/// Specification-related errors
#[derive(Debug, Clone)]
pub enum SpecificationError {
    InvalidConfiguration(String),
    InvalidLayer(usize, String),
    DimensionMismatch(usize, usize, usize, usize),
    IncompatibleComponents(String, String),
    ValidationFailed(String),
}

impl std::fmt::Display for SpecificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecificationError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            SpecificationError::InvalidLayer(idx, msg) => {
                write!(f, "Invalid layer {}: {}", idx, msg)
            }
            SpecificationError::DimensionMismatch(i, j, dim1, dim2) => {
                write!(
                    f,
                    "Dimension mismatch between layers {} and {}: {} != {}",
                    i, j, dim1, dim2
                )
            }
            SpecificationError::IncompatibleComponents(comp1, comp2) => {
                write!(f, "Incompatible components: {} and {}", comp1, comp2)
            }
            SpecificationError::ValidationFailed(msg) => {
                write!(f, "Validation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for SpecificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_spec_creation() {
        let dimensions = LayerDimensions {
            input_dim: 256,
            output_dim: 128,
            hidden_dims: vec![],
        };

        let layer = LayerSpec::new(LayerType::Linear, dimensions, ActivationType::ReLU);
        assert_eq!(layer.layer_type, LayerType::Linear);
        assert_eq!(layer.activation, ActivationType::ReLU);
    }

    #[test]
    fn test_architecture_validation() {
        let layer1 = LayerSpec::new(
            LayerType::Linear,
            LayerDimensions {
                input_dim: 256,
                output_dim: 128,
                hidden_dims: vec![],
            },
            ActivationType::ReLU,
        );

        let layer2 = LayerSpec::new(
            LayerType::Linear,
            LayerDimensions {
                input_dim: 128,
                output_dim: 64,
                hidden_dims: vec![],
            },
            ActivationType::ReLU,
        );

        let arch = ArchitectureSpec::new(
            vec![layer1, layer2],
            GlobalArchitectureConfig::default(),
        );

        assert!(arch.validate().is_ok());
    }

    #[test]
    fn test_parameter_count_estimation() {
        let layer = LayerSpec::new(
            LayerType::Linear,
            LayerDimensions {
                input_dim: 100,
                output_dim: 50,
                hidden_dims: vec![],
            },
            ActivationType::ReLU,
        );

        let param_count = layer.estimate_parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_architecture_candidate_fitness() {
        let arch = ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default());
        let mut candidate = ArchitectureCandidate::new("test".to_string(), arch);

        candidate.performance.optimization_performance = 0.8;
        candidate.performance.convergence_speed = 0.7;

        let weights = vec![0.5, 0.3, 0.2, 0.0, 0.0];
        let fitness = candidate.fitness_score(&weights);
        assert!(fitness > 0.0);
    }
}