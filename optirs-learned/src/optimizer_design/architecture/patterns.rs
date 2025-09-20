// Architecture patterns and connection topologies
//
// This module defines common architectural patterns, connection topologies,
// and optimization paths used in neural architecture search.

use std::collections::HashMap;

use super::components::*;

/// Connection topology patterns
#[derive(Debug, Clone)]
pub struct ConnectionTopology {
    /// Topology type
    pub topology_type: TopologyType,

    /// Connection density
    pub density: f64,

    /// Connection pattern
    pub pattern: ConnectionPattern,

    /// Layer-specific connections
    pub layer_connections: HashMap<usize, Vec<usize>>,

    /// Skip connection configuration
    pub skip_configuration: SkipConfiguration,
}

/// Types of connection topologies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopologyType {
    Sequential,
    Parallel,
    Tree,
    Graph,
    Mesh,
    Star,
    Ring,
    Custom,
}

/// Skip connection configuration
#[derive(Debug, Clone)]
pub struct SkipConfiguration {
    /// Skip connection type
    pub skip_type: SkipConnectionType,

    /// Skip intervals (every N layers)
    pub skip_intervals: Vec<usize>,

    /// Adaptive skip connections
    pub adaptive_skips: bool,

    /// Skip connection weights
    pub skip_weights: HashMap<(usize, usize), f64>,
}

/// Residual connection patterns
#[derive(Debug, Clone)]
pub struct ResidualPattern {
    /// Pattern type
    pub pattern_type: ResidualPatternType,

    /// Block size for residual connections
    pub block_size: usize,

    /// Bottleneck configuration
    pub bottleneck_config: Option<BottleneckConfig>,

    /// Normalization placement
    pub norm_placement: NormPlacement,

    /// Activation placement
    pub activation_placement: ActivationPlacement,
}

/// Types of residual patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualPatternType {
    Basic,          // Basic residual block
    Bottleneck,     // Bottleneck residual block
    PreActivation,  // Pre-activation residual
    DenseNet,       // DenseNet-style connections
    Highway,        // Highway networks
    ResNeXt,        // ResNeXt-style blocks
    Custom,
}

/// Bottleneck configuration
#[derive(Debug, Clone)]
pub struct BottleneckConfig {
    /// Reduction ratio
    pub reduction_ratio: f64,

    /// Intermediate dimensions
    pub intermediate_dims: Vec<usize>,

    /// Use 1x1 convolutions
    pub use_pointwise: bool,
}

/// Normalization placement in residual blocks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormPlacement {
    PreActivation,
    PostActivation,
    Both,
    None,
}

/// Activation placement in residual blocks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationPlacement {
    PreNorm,
    PostNorm,
    Both,
    None,
}

/// Attention connectivity patterns
#[derive(Debug, Clone)]
pub struct AttentionConnectivity {
    /// Attention pattern
    pub attention_pattern: AttentionPatternType,

    /// Connectivity matrix
    pub connectivity_matrix: Option<Vec<Vec<bool>>>,

    /// Local attention window size
    pub local_window_size: Option<usize>,

    /// Sparse attention configuration
    pub sparse_config: Option<SparseAttentionPattern>,

    /// Cross-layer attention
    pub cross_layer_attention: bool,
}

/// Types of attention patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionPatternType {
    Full,         // Full attention
    Local,        // Local attention window
    Sparse,       // Sparse attention
    Blocked,      // Block-sparse attention
    Strided,      // Strided attention
    Random,       // Random sparse attention
    Learned,      // Learned attention pattern
    Hierarchical, // Hierarchical attention
}

/// Sparse attention pattern configuration
#[derive(Debug, Clone)]
pub struct SparseAttentionPattern {
    /// Sparsity ratio
    pub sparsity_ratio: f64,

    /// Block size for blocked attention
    pub block_size: Option<usize>,

    /// Stride for strided attention
    pub stride: Option<usize>,

    /// Pattern regularity
    pub pattern_regularity: PatternRegularity,
}

/// Pattern regularity for sparse attention
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternRegularity {
    Regular,      // Regular pattern
    Irregular,    // Irregular pattern
    Adaptive,     // Adaptive based on input
    Learned,      // Learned during training
}

/// Memory access strategies
#[derive(Debug, Clone)]
pub struct MemoryAccessStrategy {
    /// Access pattern type
    pub access_type: MemoryAccessType,

    /// Prefetch strategy
    pub prefetch_strategy: PrefetchStrategy,

    /// Cache optimization
    pub cache_optimization: CacheOptimization,

    /// Memory layout
    pub memory_layout: MemoryLayout,
}

/// Types of memory access patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryAccessType {
    Sequential,
    Random,
    Strided,
    Blocked,
    Hierarchical,
    ContentAddressable,
    Adaptive,
}

/// Prefetch strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchStrategy {
    None,
    Conservative,
    Aggressive,
    Adaptive,
    Learned,
}

/// Cache optimization strategies
#[derive(Debug, Clone)]
pub struct CacheOptimization {
    /// Cache-friendly data layout
    pub data_layout_optimization: bool,

    /// Cache blocking strategy
    pub blocking_strategy: BlockingStrategy,

    /// Cache line utilization
    pub cache_line_optimization: bool,
}

/// Blocking strategies for cache optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockingStrategy {
    None,
    Basic,
    Hierarchical,
    Adaptive,
    Custom,
}

/// Memory layout patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked,
    Tiled,
    Custom,
}

/// Optimization path patterns
#[derive(Debug, Clone)]
pub struct OptimizationPath {
    /// Path type
    pub path_type: OptimizationPathType,

    /// Search trajectory
    pub search_trajectory: SearchTrajectory,

    /// Exploration strategy
    pub exploration_strategy: ExplorationStrategy,

    /// Convergence pattern
    pub convergence_pattern: ConvergencePattern,
}

/// Types of optimization paths
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationPathType {
    Greedy,
    Evolutionary,
    Gradient,
    Bayesian,
    Reinforcement,
    Hybrid,
    Progressive,
}

/// Search trajectory configuration
#[derive(Debug, Clone)]
pub struct SearchTrajectory {
    /// Trajectory smoothness
    pub smoothness: f64,

    /// Exploration vs exploitation balance
    pub exploration_ratio: f64,

    /// Restart strategy
    pub restart_strategy: RestartStrategy,

    /// Path memory
    pub path_memory_size: usize,
}

/// Restart strategies for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartStrategy {
    None,
    Periodic,
    Adaptive,
    Performance,
    Diversity,
}

/// Exploration strategies
#[derive(Debug, Clone)]
pub struct ExplorationStrategy {
    /// Strategy type
    pub strategy_type: ExplorationStrategyType,

    /// Exploration rate schedule
    pub exploration_schedule: ExplorationSchedule,

    /// Novelty detection
    pub novelty_detection: bool,
}

/// Types of exploration strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExplorationStrategyType {
    Random,
    Systematic,
    Curiosity,
    Diversity,
    Uncertainty,
    Hybrid,
}

/// Exploration rate scheduling
#[derive(Debug, Clone)]
pub struct ExplorationSchedule {
    /// Initial exploration rate
    pub initial_rate: f64,

    /// Final exploration rate
    pub final_rate: f64,

    /// Decay type
    pub decay_type: DecayType,

    /// Decay steps
    pub decay_steps: usize,
}

/// Types of decay for exploration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayType {
    Linear,
    Exponential,
    Cosine,
    Polynomial,
    Step,
    Adaptive,
}

/// Convergence patterns
#[derive(Debug, Clone)]
pub struct ConvergencePattern {
    /// Pattern type
    pub pattern_type: ConvergencePatternType,

    /// Early stopping criteria
    pub early_stopping: EarlyStopping,

    /// Convergence detection
    pub convergence_detection: ConvergenceDetection,
}

/// Types of convergence patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergencePatternType {
    Monotonic,
    Oscillatory,
    Plateau,
    Stochastic,
    MultiModal,
    Adaptive,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Patience (iterations without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_delta: f64,

    /// Monitor metric
    pub monitor_metric: String,

    /// Restore best weights
    pub restore_best: bool,
}

/// Convergence detection methods
#[derive(Debug, Clone)]
pub struct ConvergenceDetection {
    /// Detection method
    pub method: ConvergenceDetectionMethod,

    /// Window size for detection
    pub window_size: usize,

    /// Convergence threshold
    pub threshold: f64,
}

/// Methods for detecting convergence
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceDetectionMethod {
    ThresholdBased,
    TrendAnalysis,
    Statistical,
    Plateau,
    Gradient,
    Hybrid,
}

// Implementation methods
impl Default for ConnectionTopology {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Sequential,
            density: 1.0,
            pattern: ConnectionPattern::Sequential,
            layer_connections: HashMap::new(),
            skip_configuration: SkipConfiguration::default(),
        }
    }
}

impl Default for SkipConfiguration {
    fn default() -> Self {
        Self {
            skip_type: SkipConnectionType::None,
            skip_intervals: vec![],
            adaptive_skips: false,
            skip_weights: HashMap::new(),
        }
    }
}

impl Default for ResidualPattern {
    fn default() -> Self {
        Self {
            pattern_type: ResidualPatternType::Basic,
            block_size: 2,
            bottleneck_config: None,
            norm_placement: NormPlacement::PostActivation,
            activation_placement: ActivationPlacement::PostNorm,
        }
    }
}

impl Default for AttentionConnectivity {
    fn default() -> Self {
        Self {
            attention_pattern: AttentionPatternType::Full,
            connectivity_matrix: None,
            local_window_size: None,
            sparse_config: None,
            cross_layer_attention: false,
        }
    }
}

impl Default for MemoryAccessStrategy {
    fn default() -> Self {
        Self {
            access_type: MemoryAccessType::Sequential,
            prefetch_strategy: PrefetchStrategy::Conservative,
            cache_optimization: CacheOptimization {
                data_layout_optimization: true,
                blocking_strategy: BlockingStrategy::Basic,
                cache_line_optimization: true,
            },
            memory_layout: MemoryLayout::RowMajor,
        }
    }
}

impl Default for OptimizationPath {
    fn default() -> Self {
        Self {
            path_type: OptimizationPathType::Greedy,
            search_trajectory: SearchTrajectory {
                smoothness: 0.5,
                exploration_ratio: 0.1,
                restart_strategy: RestartStrategy::None,
                path_memory_size: 100,
            },
            exploration_strategy: ExplorationStrategy {
                strategy_type: ExplorationStrategyType::Random,
                exploration_schedule: ExplorationSchedule {
                    initial_rate: 1.0,
                    final_rate: 0.01,
                    decay_type: DecayType::Exponential,
                    decay_steps: 1000,
                },
                novelty_detection: false,
            },
            convergence_pattern: ConvergencePattern {
                pattern_type: ConvergencePatternType::Monotonic,
                early_stopping: EarlyStopping {
                    patience: 10,
                    min_delta: 0.001,
                    monitor_metric: "loss".to_string(),
                    restore_best: true,
                },
                convergence_detection: ConvergenceDetection {
                    method: ConvergenceDetectionMethod::ThresholdBased,
                    window_size: 10,
                    threshold: 0.001,
                },
            },
        }
    }
}

impl ConnectionTopology {
    /// Create a sequential topology
    pub fn sequential(num_layers: usize) -> Self {
        let mut layer_connections = HashMap::new();
        for i in 0..num_layers - 1 {
            layer_connections.insert(i, vec![i + 1]);
        }

        Self {
            topology_type: TopologyType::Sequential,
            density: 1.0 / (num_layers as f64 - 1.0),
            pattern: ConnectionPattern::Sequential,
            layer_connections,
            skip_configuration: SkipConfiguration::default(),
        }
    }

    /// Create a residual topology with skip connections
    pub fn residual(num_layers: usize, skip_interval: usize) -> Self {
        let mut topology = Self::sequential(num_layers);
        topology.topology_type = TopologyType::Custom;
        topology.pattern = ConnectionPattern::Residual;

        // Add skip connections
        for i in 0..num_layers {
            if i + skip_interval < num_layers {
                topology
                    .layer_connections
                    .entry(i)
                    .or_insert_with(Vec::new)
                    .push(i + skip_interval);
            }
        }

        topology.skip_configuration = SkipConfiguration {
            skip_type: SkipConnectionType::Residual,
            skip_intervals: vec![skip_interval],
            adaptive_skips: false,
            skip_weights: HashMap::new(),
        };

        topology
    }

    /// Calculate connectivity density
    pub fn calculate_density(&self) -> f64 {
        let total_connections: usize = self.layer_connections.values().map(|v| v.len()).sum();
        let max_connections = self.layer_connections.len() * self.layer_connections.len();
        
        if max_connections == 0 {
            0.0
        } else {
            total_connections as f64 / max_connections as f64
        }
    }

    /// Check if topology is valid
    pub fn is_valid(&self) -> bool {
        // Check for cycles (simplified check)
        for (from, connections) in &self.layer_connections {
            for &to in connections {
                if to <= *from {
                    // Backward connection detected - could be a cycle
                    continue; // Allow for some patterns like residual
                }
            }
        }
        true
    }
}

impl ResidualPattern {
    /// Create a basic residual pattern
    pub fn basic(block_size: usize) -> Self {
        Self {
            pattern_type: ResidualPatternType::Basic,
            block_size,
            bottleneck_config: None,
            norm_placement: NormPlacement::PostActivation,
            activation_placement: ActivationPlacement::PostNorm,
        }
    }

    /// Create a bottleneck residual pattern
    pub fn bottleneck(block_size: usize, reduction_ratio: f64) -> Self {
        Self {
            pattern_type: ResidualPatternType::Bottleneck,
            block_size,
            bottleneck_config: Some(BottleneckConfig {
                reduction_ratio,
                intermediate_dims: vec![],
                use_pointwise: true,
            }),
            norm_placement: NormPlacement::PostActivation,
            activation_placement: ActivationPlacement::PostNorm,
        }
    }

    /// Get computational overhead
    pub fn computational_overhead(&self) -> f64 {
        match self.pattern_type {
            ResidualPatternType::Basic => 1.1,
            ResidualPatternType::Bottleneck => 0.8,
            ResidualPatternType::PreActivation => 1.05,
            ResidualPatternType::DenseNet => 1.5,
            ResidualPatternType::Highway => 1.3,
            ResidualPatternType::ResNeXt => 1.2,
            ResidualPatternType::Custom => 1.0,
        }
    }
}

impl AttentionConnectivity {
    /// Create full attention connectivity
    pub fn full_attention(num_heads: usize) -> Self {
        Self {
            attention_pattern: AttentionPatternType::Full,
            connectivity_matrix: None,
            local_window_size: None,
            sparse_config: None,
            cross_layer_attention: false,
        }
    }

    /// Create local attention connectivity
    pub fn local_attention(window_size: usize) -> Self {
        Self {
            attention_pattern: AttentionPatternType::Local,
            connectivity_matrix: None,
            local_window_size: Some(window_size),
            sparse_config: None,
            cross_layer_attention: false,
        }
    }

    /// Create sparse attention connectivity
    pub fn sparse_attention(sparsity_ratio: f64) -> Self {
        Self {
            attention_pattern: AttentionPatternType::Sparse,
            connectivity_matrix: None,
            local_window_size: None,
            sparse_config: Some(SparseAttentionPattern {
                sparsity_ratio,
                block_size: None,
                stride: None,
                pattern_regularity: PatternRegularity::Regular,
            }),
            cross_layer_attention: false,
        }
    }

    /// Calculate attention complexity
    pub fn attention_complexity(&self, sequence_length: usize) -> f64 {
        match self.attention_pattern {
            AttentionPatternType::Full => (sequence_length * sequence_length) as f64,
            AttentionPatternType::Local => {
                let window = self.local_window_size.unwrap_or(8);
                (sequence_length * window) as f64
            }
            AttentionPatternType::Sparse => {
                let sparsity = self
                    .sparse_config
                    .as_ref()
                    .map(|c| c.sparsity_ratio)
                    .unwrap_or(0.1);
                (sequence_length * sequence_length) as f64 * sparsity
            }
            _ => (sequence_length * sequence_length) as f64 * 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_topology() {
        let topology = ConnectionTopology::sequential(5);
        assert_eq!(topology.topology_type, TopologyType::Sequential);
        assert_eq!(topology.layer_connections.len(), 4);
    }

    #[test]
    fn test_residual_topology() {
        let topology = ConnectionTopology::residual(6, 2);
        assert_eq!(topology.pattern, ConnectionPattern::Residual);
        assert_eq!(topology.skip_configuration.skip_intervals, vec![2]);
    }

    #[test]
    fn test_density_calculation() {
        let topology = ConnectionTopology::sequential(4);
        let density = topology.calculate_density();
        assert!(density > 0.0 && density <= 1.0);
    }

    #[test]
    fn test_residual_pattern_overhead() {
        let basic = ResidualPattern::basic(2);
        let bottleneck = ResidualPattern::bottleneck(2, 0.25);
        
        assert!(basic.computational_overhead() > 1.0);
        assert!(bottleneck.computational_overhead() < basic.computational_overhead());
    }

    #[test]
    fn test_attention_complexity() {
        let full = AttentionConnectivity::full_attention(8);
        let local = AttentionConnectivity::local_attention(16);
        let sparse = AttentionConnectivity::sparse_attention(0.1);

        let seq_len = 128;
        assert!(full.attention_complexity(seq_len) > local.attention_complexity(seq_len));
        assert!(sparse.attention_complexity(seq_len) < full.attention_complexity(seq_len));
    }

    #[test]
    fn test_topology_validity() {
        let topology = ConnectionTopology::sequential(3);
        assert!(topology.is_valid());
    }
}