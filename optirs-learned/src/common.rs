//! Common types and configurations for learned optimizers

use num_traits::Float;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Base configuration for learned optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedOptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Meta-learning rate for higher-level optimization
    pub meta_learning_rate: f64,

    /// Batch size for training
    pub batch_size: usize,

    /// Maximum number of optimization steps
    pub max_steps: usize,

    /// Convergence threshold
    pub convergence_threshold: f64,

    /// Whether to use momentum
    pub use_momentum: bool,

    /// Momentum decay factor
    pub momentum_decay: f64,

    /// Weight decay for regularization
    pub weight_decay: f64,

    /// Hidden size for neural networks
    pub hidden_size: usize,

    /// Number of attention heads
    pub attention_heads: usize,

    /// Size of gradient history buffer
    pub gradient_history_size: usize,

    /// Number of input features
    pub input_features: usize,

    /// Number of output features
    pub output_features: usize,

    /// Number of layers in neural networks
    pub num_layers: usize,

    /// Dropout rate for regularization
    pub dropout_rate: f64,

    /// Whether to use attention mechanisms
    pub use_attention: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for LearnedOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            meta_learning_rate: 0.0001,
            batch_size: 32,
            max_steps: 1000,
            convergence_threshold: 1e-6,
            use_momentum: true,
            momentum_decay: 0.9,
            weight_decay: 1e-4,
            hidden_size: 256,
            attention_heads: 8,
            gradient_history_size: 50,
            input_features: 256,
            output_features: 256,
            num_layers: 3,
            dropout_rate: 0.1,
            use_attention: true,
            seed: None,
        }
    }
}

/// Meta-optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaOptimizationStrategy {
    /// First-order approximation (FOMAML)
    FirstOrder,

    /// Full second-order gradients (MAML)
    SecondOrder,

    /// MAML algorithm (alias for SecondOrder)
    MAML,

    /// Reptile algorithm
    Reptile,

    /// Custom gradient-based meta-learning
    Custom {
        inner_steps: usize,
        outer_learning_rate: f64,
    },
}

impl Default for MetaOptimizationStrategy {
    fn default() -> Self {
        MetaOptimizationStrategy::FirstOrder
    }
}

/// Base optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerState<T: Float + Debug + Send + Sync + 'static> {
    /// Current parameters
    pub parameters: Array1<T>,

    /// Current gradients
    pub gradients: Array1<T>,

    /// Momentum buffer
    pub momentum: Option<Array1<T>>,

    /// Hidden states for neural optimizers
    pub hidden_states: HashMap<String, Array1<T>>,

    /// Memory buffers for attention mechanisms
    pub memory_buffers: HashMap<String, Array2<T>>,

    /// Current step number
    pub step: usize,

    /// Step count (alias for step)
    pub step_count: usize,

    /// Current loss value
    pub loss: Option<T>,

    /// Learning rate schedule
    pub learning_rate: T,

    /// Additional state metadata
    pub metadata: StateMetadata,
}

impl<T: Float + Debug + Send + Sync + 'static> OptimizerState<T> {
    pub fn new(num_params: usize) -> Self {
        Self {
            parameters: Array1::zeros(num_params),
            gradients: Array1::zeros(num_params),
            momentum: None,
            hidden_states: HashMap::new(),
            memory_buffers: HashMap::new(),
            step: 0,
            step_count: 0,
            loss: None,
            learning_rate: T::from(0.001).unwrap(),
            metadata: StateMetadata::default(),
        }
    }
}

/// State metadata
#[derive(Debug, Clone)]
pub struct StateMetadata {
    /// Task identifier
    pub task_id: Option<String>,

    /// Optimizer type used
    pub optimizer_type: Option<String>,

    /// Version information
    pub version: String,

    /// Timestamp of creation/update
    pub timestamp: std::time::SystemTime,

    /// Checksum for integrity
    pub checksum: u64,

    /// Compression level used
    pub compression_level: u8,

    /// Additional custom metadata
    pub custom_data: HashMap<String, String>,
}

impl Default for StateMetadata {
    fn default() -> Self {
        Self {
            task_id: None,
            optimizer_type: None,
            version: "1.0".to_string(),
            timestamp: std::time::SystemTime::now(),
            checksum: 0,
            compression_level: 0,
            custom_data: HashMap::new(),
        }
    }
}

/// Neural optimizer type variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralOptimizerType {
    /// Transformer-based optimizer
    Transformer,

    /// LSTM-based optimizer
    LSTM,

    /// Simple MLP optimizer
    MLP,

    /// Convolutional optimizer
    CNN,
}

/// Task context for meta-learning
#[derive(Debug, Clone)]
pub struct TaskContext<T: Float + Debug + Send + Sync + 'static> {
    /// Task identifier
    pub task_id: String,

    /// Initial parameters for the task
    pub initial_parameters: Array1<T>,

    /// Task-specific data
    pub task_data: Array2<T>,

    /// Target values for the task
    pub targets: Array1<T>,

    /// Task difficulty or complexity measure
    pub difficulty: f64,
}

/// Neural optimizer metrics
#[derive(Debug, Clone)]
pub struct NeuralOptimizerMetrics {
    /// Average loss over training
    pub avg_loss: f64,

    /// Convergence rate
    pub convergence_rate: f64,

    /// Number of steps to convergence
    pub steps_to_convergence: Option<usize>,

    /// Final accuracy/performance
    pub final_performance: f64,

    /// Training time in seconds
    pub training_time: f64,
}

/// Task performance metrics
#[derive(Debug, Clone)]
pub struct TaskPerformance {
    /// Task identifier
    pub task_id: String,

    /// Performance score (higher is better)
    pub score: f64,

    /// Whether the task converged
    pub converged: bool,

    /// Number of optimization steps taken
    pub steps_taken: usize,

    /// Final loss value
    pub final_loss: f64,
}
