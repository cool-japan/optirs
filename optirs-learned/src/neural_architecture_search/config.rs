// Configuration structures for Neural Architecture Search

use std::collections::HashMap;

/// NAS configuration
#[derive(Debug, Clone, Default)]
pub struct NASConfig {
    /// Search strategy type
    pub search_strategy: SearchStrategyType,

    /// Maximum search iterations
    pub max_iterations: usize,

    /// Population size (for evolutionary strategies)
    pub population_size: usize,

    /// Number of top architectures to keep
    pub elite_size: usize,

    /// Mutation rate for evolutionary search
    pub mutation_rate: f64,

    /// Crossover rate for evolutionary search
    pub crossover_rate: f64,

    /// Early stopping patience
    pub early_stopping_patience: usize,

    /// Evaluation budget (computational resources)
    pub evaluation_budget: usize,

    /// Multi-objective weights
    pub objective_weights: Vec<f64>,

    /// Enable performance prediction
    pub enable_performance_prediction: bool,

    /// Enable progressive search
    pub progressive_search: bool,

    /// Search space constraints
    pub constraints: SearchConstraints,

    /// Parallelization level
    pub parallelization_level: usize,

    /// Enable transfer learning
    pub enable_transfer_learning: bool,

    /// Warm start from existing architectures
    pub warm_start_architectures: Vec<String>,
}

impl NASConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self {
            search_strategy: SearchStrategyType::Evolutionary,
            max_iterations: 100,
            population_size: 50,
            elite_size: 10,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            early_stopping_patience: 20,
            evaluation_budget: 1000,
            objective_weights: vec![1.0, 0.5, 0.3], // accuracy, efficiency, complexity
            enable_performance_prediction: true,
            progressive_search: false,
            constraints: SearchConstraints::default(),
            parallelization_level: 4,
            enable_transfer_learning: false,
            warm_start_architectures: Vec::new(),
        }
    }

    /// Create configuration for quick search
    pub fn quick_search() -> Self {
        Self {
            max_iterations: 20,
            population_size: 10,
            elite_size: 3,
            evaluation_budget: 100,
            early_stopping_patience: 10,
            ..Self::new()
        }
    }

    /// Create configuration for thorough search
    pub fn thorough_search() -> Self {
        Self {
            max_iterations: 500,
            population_size: 100,
            elite_size: 20,
            evaluation_budget: 5000,
            early_stopping_patience: 50,
            parallelization_level: 8,
            ..Self::new()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_iterations == 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }

        if self.population_size == 0 {
            return Err("population_size must be greater than 0".to_string());
        }

        if self.elite_size > self.population_size {
            return Err("elite_size cannot be larger than population_size".to_string());
        }

        if self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err("mutation_rate must be between 0.0 and 1.0".to_string());
        }

        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err("crossover_rate must be between 0.0 and 1.0".to_string());
        }

        if self.evaluation_budget == 0 {
            return Err("evaluation_budget must be greater than 0".to_string());
        }

        if self.objective_weights.is_empty() {
            return Err("objective_weights cannot be empty".to_string());
        }

        if self.parallelization_level == 0 {
            return Err("parallelization_level must be greater than 0".to_string());
        }

        self.constraints.validate()?;

        Ok(())
    }
}

/// Search strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchStrategyType {
    /// Random search
    Random,
    /// Evolutionary algorithm
    Evolutionary,
    /// Bayesian optimization
    Bayesian,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Gradient-based search
    GradientBased,
    /// Progressive search
    Progressive,
    /// Hybrid approach
    Hybrid,
}

impl Default for SearchStrategyType {
    fn default() -> Self {
        SearchStrategyType::Evolutionary
    }
}

/// Search space constraints
#[derive(Debug, Clone)]
pub struct SearchConstraints {
    /// Maximum number of layers
    pub max_layers: usize,
    /// Minimum number of layers
    pub min_layers: usize,
    /// Maximum parameters per layer
    pub max_params_per_layer: usize,
    /// Allowed layer types
    pub allowed_layer_types: Vec<LayerType>,
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum FLOPs
    pub max_flops: u64,
    /// Target accuracy threshold
    pub min_accuracy: f64,
    /// Maximum training time (seconds)
    pub max_training_time_secs: u64,
}

impl Default for SearchConstraints {
    fn default() -> Self {
        Self {
            max_layers: 20,
            min_layers: 2,
            max_params_per_layer: 1000000,
            allowed_layer_types: vec![
                LayerType::Dense,
                LayerType::Conv2D,
                LayerType::LSTM,
                LayerType::Attention,
                LayerType::Residual,
            ],
            max_memory_mb: 1024,
            max_flops: 1_000_000_000,
            min_accuracy: 0.8,
            max_training_time_secs: 3600,
        }
    }
}

impl SearchConstraints {
    /// Validate constraints
    pub fn validate(&self) -> Result<(), String> {
        if self.min_layers > self.max_layers {
            return Err("min_layers cannot be greater than max_layers".to_string());
        }

        if self.max_layers == 0 {
            return Err("max_layers must be greater than 0".to_string());
        }

        if self.max_params_per_layer == 0 {
            return Err("max_params_per_layer must be greater than 0".to_string());
        }

        if self.allowed_layer_types.is_empty() {
            return Err("allowed_layer_types cannot be empty".to_string());
        }

        if self.min_accuracy < 0.0 || self.min_accuracy > 1.0 {
            return Err("min_accuracy must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Check if architecture satisfies constraints
    pub fn satisfies(&self, architecture: &ArchitectureSpec) -> bool {
        if architecture.layers.len() < self.min_layers || architecture.layers.len() > self.max_layers {
            return false;
        }

        for layer in &architecture.layers {
            if !self.allowed_layer_types.contains(&layer.layer_type) {
                return false;
            }

            if layer.params > self.max_params_per_layer {
                return false;
            }
        }

        if architecture.estimated_memory_mb > self.max_memory_mb {
            return false;
        }

        if architecture.estimated_flops > self.max_flops {
            return false;
        }

        true
    }
}

/// Layer types for neural architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Dense,
    Conv2D,
    Conv1D,
    LSTM,
    GRU,
    Attention,
    MultiHeadAttention,
    Transformer,
    Residual,
    BatchNorm,
    Dropout,
    Activation,
}

/// Architecture specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchitectureSpec {
    pub layers: Vec<LayerSpec>,
    pub estimated_memory_mb: usize,
    pub estimated_flops: u64,
    pub estimated_params: usize,
}

/// Layer specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerSpec {
    pub layer_type: LayerType,
    pub params: usize,
    pub config: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_config_creation() {
        let config = NASConfig::new();
        assert!(config.validate().is_ok());
        assert_eq!(config.search_strategy, SearchStrategyType::Evolutionary);
    }

    #[test]
    fn test_config_validation() {
        let mut config = NASConfig::new();

        // Test invalid population size
        config.population_size = 0;
        assert!(config.validate().is_err());

        // Test invalid elite size
        config.population_size = 10;
        config.elite_size = 20;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_search_constraints() {
        let constraints = SearchConstraints::default();
        assert!(constraints.validate().is_ok());

        let mut invalid_constraints = constraints.clone();
        invalid_constraints.min_layers = 10;
        invalid_constraints.max_layers = 5;
        assert!(invalid_constraints.validate().is_err());
    }

    #[test]
    fn test_quick_search_config() {
        let config = NASConfig::quick_search();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.population_size, 10);
    }
}