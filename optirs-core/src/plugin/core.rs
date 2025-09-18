// Core plugin traits and interfaces for optimizer development
//
// This module defines the fundamental traits and structures that custom optimizers
// must implement to integrate with the plugin system.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

/// Main trait for optimizer plugins
pub trait OptimizerPlugin<A: Float>: Debug + Send + Sync {
    /// Perform a single optimization step
    fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>>;

    /// Get optimizer name
    fn name(&self) -> &str;

    /// Get optimizer version
    fn version(&self) -> &str;

    /// Get plugin information
    fn plugin_info(&self) -> PluginInfo;

    /// Get optimizer capabilities
    fn capabilities(&self) -> PluginCapabilities;

    /// Initialize optimizer with parameters
    fn initialize(&mut self, paramshape: &[usize]) -> Result<()>;

    /// Reset optimizer state
    fn reset(&mut self) -> Result<()>;

    /// Get optimizer configuration
    fn get_config(&self) -> OptimizerConfig;

    /// Set optimizer configuration
    fn set_config(&mut self, config: OptimizerConfig) -> Result<()>;

    /// Get optimizer state for serialization
    fn get_state(&self) -> Result<OptimizerState>;

    /// Set optimizer state from deserialization
    fn set_state(&mut self, state: OptimizerState) -> Result<()>;

    /// Clone the optimizer plugin
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>>;

    /// Get memory usage information
    fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage::default()
    }

    /// Get performance metrics
    fn performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics::default()
    }
}

/// Extended plugin trait for optimizers with advanced features
pub trait ExtendedOptimizerPlugin<A: Float>: OptimizerPlugin<A> {
    /// Perform batch optimization step
    fn batch_step(&mut self, params: &Array2<A>, gradients: &Array2<A>) -> Result<Array2<A>>;

    /// Compute adaptive learning rate
    fn adaptive_learning_rate(&self, gradients: &Array1<A>) -> A;

    /// Gradient preprocessing
    fn preprocess_gradients(&self, gradients: &Array1<A>) -> Result<Array1<A>>;

    /// Parameter postprocessing
    fn postprocess_parameters(&self, params: &Array1<A>) -> Result<Array1<A>>;

    /// Get optimization trajectory
    fn get_trajectory(&self) -> Vec<Array1<A>>;

    /// Compute convergence metrics
    fn convergence_metrics(&self) -> ConvergenceMetrics;
}

/// Plugin information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Plugin homepage/repository
    pub homepage: Option<String>,
    /// Plugin license
    pub license: String,
    /// Supported data types
    pub supported_types: Vec<DataType>,
    /// Plugin category
    pub category: PluginCategory,
    /// Plugin tags for search/filtering
    pub tags: Vec<String>,
    /// Minimum SDK version required
    pub min_sdk_version: String,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
}

/// Plugin capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Supports sparse gradients
    pub sparse_gradients: bool,
    /// Supports parameter groups
    pub parameter_groups: bool,
    /// Supports momentum
    pub momentum: bool,
    /// Supports adaptive learning rates
    pub adaptive_learning_rate: bool,
    /// Supports weight decay
    pub weight_decay: bool,
    /// Supports gradient clipping
    pub gradient_clipping: bool,
    /// Supports batch processing
    pub batch_processing: bool,
    /// Supports state serialization
    pub state_serialization: bool,
    /// Thread safety
    pub thread_safe: bool,
    /// Memory efficient
    pub memory_efficient: bool,
    /// GPU acceleration support
    pub gpu_support: bool,
    /// SIMD optimization
    pub simd_optimized: bool,
    /// Supports custom loss functions
    pub custom_loss_functions: bool,
    /// Supports regularization
    pub regularization: bool,
}

/// Supported data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    Complex32,
    Complex64,
    Custom(String),
}

/// Plugin categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PluginCategory {
    /// First-order optimizers (SGD, Adam, etc.)
    FirstOrder,
    /// Second-order optimizers (Newton, BFGS, etc.)
    SecondOrder,
    /// Specialized optimizers (domain-specific)
    Specialized,
    /// Meta-learning optimizers
    MetaLearning,
    /// Experimental optimizers
    Experimental,
    /// Utility/helper plugins
    Utility,
}

/// Plugin dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: String,
    /// Whether dependency is optional
    pub optional: bool,
    /// Dependency type
    pub dependency_type: DependencyType,
}

/// Types of plugin dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Another plugin
    Plugin,
    /// System library
    SystemLibrary,
    /// Rust crate
    Crate,
    /// Runtime requirement
    Runtime,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Custom parameters
    pub custom_params: HashMap<String, ConfigValue>,
}

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
}

/// Optimizer state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Internal state vectors
    pub state_vectors: HashMap<String, Vec<f64>>,
    /// Step count
    pub step_count: usize,
    /// Custom state data
    pub custom_state: HashMap<String, StateValue>,
}

/// State value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}

/// Memory usage information
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average step time (seconds)
    pub avg_step_time: f64,
    /// Total steps performed
    pub total_steps: usize,
    /// Throughput (steps per second)
    pub throughput: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
}

/// Convergence metrics
#[derive(Debug, Clone, Default)]
pub struct ConvergenceMetrics {
    /// Gradient norm
    pub gradient_norm: f64,
    /// Parameter change norm
    pub parameter_change_norm: f64,
    /// Loss improvement rate
    pub loss_improvement_rate: f64,
    /// Convergence score (0.0 to 1.0)
    pub convergence_score: f64,
}

/// Plugin validation result
#[derive(Debug, Clone)]
pub struct PluginValidationResult {
    /// Whether plugin is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Performance benchmark results
    pub benchmark_results: Option<BenchmarkResults>,
}

/// Benchmark results for plugin validation
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Execution time benchmarks
    pub execution_times: Vec<Duration>,
    /// Memory usage benchmarks
    pub memory_usage: Vec<usize>,
    /// Accuracy benchmarks
    pub accuracy_scores: Vec<f64>,
    /// Convergence benchmarks
    pub convergence_rates: Vec<f64>,
}

/// Plugin factory trait for creating optimizer instances
pub trait OptimizerPluginFactory<A: Float>: Debug + Send + Sync {
    /// Create a new optimizer instance
    fn create_optimizer(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<A>>>;

    /// Get factory information
    fn factory_info(&self) -> PluginInfo;

    /// Validate configuration
    fn validate_config(&self, config: &OptimizerConfig) -> Result<()>;

    /// Get default configuration
    fn default_config(&self) -> OptimizerConfig;

    /// Get configuration schema
    fn config_schema(&self) -> ConfigSchema;
}

/// Configuration schema for validation and UI generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    /// Schema fields
    pub fields: HashMap<String, FieldSchema>,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Schema version
    pub version: String,
}

/// Individual field schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field type
    pub field_type: FieldType,
    /// Field description
    pub description: String,
    /// Default value
    pub default_value: Option<ConfigValue>,
    /// Validation constraints
    pub constraints: Vec<ValidationConstraint>,
    /// Whether field is required
    pub required: bool,
}

/// Field types for schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Float {
        min: Option<f64>,
        max: Option<f64>,
    },
    Integer {
        min: Option<i64>,
        max: Option<i64>,
    },
    Boolean,
    String {
        max_length: Option<usize>,
    },
    Array {
        element_type: Box<FieldType>,
        max_length: Option<usize>,
    },
    Choice {
        options: Vec<String>,
    },
}

/// Validation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationConstraint {
    /// Minimum value
    Min(f64),
    /// Maximum value
    Max(f64),
    /// Value must be positive
    Positive,
    /// Value must be non-negative
    NonNegative,
    /// Value must be in range
    Range(f64, f64),
    /// String must match regex pattern
    Pattern(String),
    /// Custom validation function name
    Custom(String),
}

/// Plugin lifecycle hooks
pub trait PluginLifecycle {
    /// Called when plugin is loaded
    fn on_load(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is unloaded
    fn on_unload(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is enabled
    fn on_enable(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is disabled
    fn on_disable(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called periodically for maintenance
    fn on_maintenance(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Plugin event system
pub trait PluginEventHandler {
    /// Handle optimization step event
    fn on_step(&mut self, _step: usize, _params: &Array1<f64>, gradients: &Array1<f64>) {}

    /// Handle convergence event
    fn on_convergence(&mut self, _finalparams: &Array1<f64>) {}

    /// Handle error event
    fn on_error(&mut self, error: &OptimError) {}

    /// Handle custom event
    fn on_custom_event(&mut self, _event_name: &str, data: &dyn Any) {}
}

/// Plugin metadata provider
pub trait PluginMetadata {
    /// Get plugin documentation
    fn documentation(&self) -> String {
        String::new()
    }

    /// Get plugin examples
    fn examples(&self) -> Vec<PluginExample> {
        Vec::new()
    }

    /// Get plugin changelog
    fn changelog(&self) -> String {
        String::new()
    }

    /// Get plugin compatibility information
    fn compatibility(&self) -> CompatibilityInfo {
        CompatibilityInfo::default()
    }
}

/// Plugin example
#[derive(Debug, Clone)]
pub struct PluginExample {
    /// Example title
    pub title: String,
    /// Example description
    pub description: String,
    /// Example code
    pub code: String,
    /// Expected output
    pub expected_output: String,
}

/// Compatibility information
#[derive(Debug, Clone, Default)]
pub struct CompatibilityInfo {
    /// Supported Rust versions
    pub rust_versions: Vec<String>,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Known issues
    pub known_issues: Vec<String>,
    /// Breaking changes
    pub breaking_changes: Vec<String>,
}

// Default implementations

impl Default for PluginInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            version: "0.1.0".to_string(),
            author: "Unknown".to_string(),
            description: "No description provided".to_string(),
            homepage: None,
            license: "MIT".to_string(),
            supported_types: vec![DataType::F32, DataType::F64],
            category: PluginCategory::FirstOrder,
            tags: Vec::new(),
            min_sdk_version: "0.1.0".to_string(),
            dependencies: Vec::new(),
        }
    }
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            sparse_gradients: false,
            parameter_groups: false,
            momentum: false,
            adaptive_learning_rate: false,
            weight_decay: false,
            gradient_clipping: false,
            batch_processing: false,
            state_serialization: false,
            thread_safe: false,
            memory_efficient: false,
            gpu_support: false,
            simd_optimized: false,
            custom_loss_functions: false,
            regularization: false,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0,
            momentum: 0.0,
            gradient_clip: None,
            custom_params: HashMap::new(),
        }
    }
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            state_vectors: HashMap::new(),
            step_count: 0,
            custom_state: HashMap::new(),
        }
    }
}

/// Utility functions for plugin development
/// Create a basic plugin info structure
#[allow(dead_code)]
pub fn create_plugin_info(name: &str, version: &str, author: &str) -> PluginInfo {
    PluginInfo {
        name: name.to_string(),
        version: version.to_string(),
        author: author.to_string(),
        ..Default::default()
    }
}

/// Create basic plugin capabilities
#[allow(dead_code)]
pub fn create_basic_capabilities() -> PluginCapabilities {
    PluginCapabilities {
        state_serialization: true,
        thread_safe: true,
        ..Default::default()
    }
}

/// Validate plugin configuration against schema
#[allow(dead_code)]
pub fn validate_config_against_schema(
    config: &OptimizerConfig,
    schema: &ConfigSchema,
) -> Result<()> {
    // Check required fields
    for required_field in &schema.required_fields {
        match required_field.as_str() {
            "learning_rate" => {
                if config.learning_rate <= 0.0 {
                    return Err(OptimError::InvalidConfig(
                        "Learning rate must be positive".to_string(),
                    ));
                }
            }
            "weight_decay" => {
                if config.weight_decay < 0.0 {
                    return Err(OptimError::InvalidConfig(
                        "Weight decay must be non-negative".to_string(),
                    ));
                }
            }
            _ => {
                if !config.custom_params.contains_key(required_field) {
                    return Err(OptimError::InvalidConfig(format!(
                        "Required field '{}' is missing",
                        required_field
                    )));
                }
            }
        }
    }

    // Validate field constraints
    for (field_name, field_schema) in &schema.fields {
        let value = match field_name.as_str() {
            "learning_rate" => Some(ConfigValue::Float(config.learning_rate)),
            "weight_decay" => Some(ConfigValue::Float(config.weight_decay)),
            "momentum" => Some(ConfigValue::Float(config.momentum)),
            _ => config.custom_params.get(field_name).cloned(),
        };

        if let Some(value) = value {
            validate_field_value(&value, field_schema)?;
        } else if field_schema.required {
            return Err(OptimError::InvalidConfig(format!(
                "Required field '{}' is missing",
                field_name
            )));
        }
    }

    Ok(())
}

/// Validate individual field value against schema
#[allow(dead_code)]
fn validate_field_value(value: &ConfigValue, schema: &FieldSchema) -> Result<()> {
    for constraint in &schema.constraints {
        match (value, constraint) {
            (ConfigValue::Float(v), ValidationConstraint::Min(min)) => {
                if v < min {
                    return Err(OptimError::InvalidConfig(format!(
                        "Value {} is below minimum {}",
                        v, min
                    )));
                }
            }
            (ConfigValue::Float(v), ValidationConstraint::Max(max)) => {
                if v > max {
                    return Err(OptimError::InvalidConfig(format!(
                        "Value {} is above maximum {}",
                        v, max
                    )));
                }
            }
            (ConfigValue::Float(v), ValidationConstraint::Positive) => {
                if *v <= 0.0 {
                    return Err(OptimError::InvalidConfig(
                        "Value must be positive".to_string(),
                    ));
                }
            }
            (ConfigValue::Float(v), ValidationConstraint::NonNegative) => {
                if *v < 0.0 {
                    return Err(OptimError::InvalidConfig(
                        "Value must be non-negative".to_string(),
                    ));
                }
            }
            (ConfigValue::Float(v), ValidationConstraint::Range(min, max)) => {
                if v < min || v > max {
                    return Err(OptimError::InvalidConfig(format!(
                        "Value {} is outside range [{}, {}]",
                        v, min, max
                    )));
                }
            }
            _ => {} // Other constraint types can be added as needed
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_info_default() {
        let info = PluginInfo::default();
        assert_eq!(info.name, "Unknown");
        assert_eq!(info.version, "0.1.0");
    }

    #[test]
    fn test_plugin_capabilities_default() {
        let caps = PluginCapabilities::default();
        assert!(!caps.sparse_gradients);
        assert!(!caps.gpu_support);
    }

    #[test]
    fn test_config_validation() {
        let mut schema = ConfigSchema {
            fields: HashMap::new(),
            required_fields: vec!["learning_rate".to_string()],
            version: "1.0".to_string(),
        };

        schema.fields.insert(
            "learning_rate".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: Some(0.0),
                    max: None,
                },
                description: "Learning rate".to_string(),
                default_value: Some(ConfigValue::Float(0.001)),
                constraints: vec![ValidationConstraint::Positive],
                required: true,
            },
        );

        let mut config = OptimizerConfig::default();
        config.learning_rate = 0.001;

        assert!(validate_config_against_schema(&config, &schema).is_ok());

        config.learning_rate = -0.001;
        assert!(validate_config_against_schema(&config, &schema).is_err());
    }
}
