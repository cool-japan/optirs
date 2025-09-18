// Adaptive Compression Strategies and Performance Monitoring
//
// This module handles adaptive compression for TPU event synchronization,
// including algorithm selection, parameter optimization, and performance monitoring.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Adaptive compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompression {
    /// Adaptive controller
    pub controller: AdaptiveController,
    /// Performance monitor
    pub monitor: PerformanceMonitor,
    /// Algorithm selector
    pub selector: AlgorithmSelector,
    /// Parameter optimizer
    pub optimizer: ParameterOptimizer,
}

impl Default for AdaptiveCompression {
    fn default() -> Self {
        Self {
            controller: AdaptiveController::default(),
            monitor: PerformanceMonitor::default(),
            selector: AlgorithmSelector::default(),
            optimizer: ParameterOptimizer::default(),
        }
    }
}

impl AdaptiveCompression {
    /// Performance-optimized adaptive compression
    pub fn performance_optimized() -> Self {
        Self {
            controller: AdaptiveController {
                enabled: true,
                strategy: AdaptiveStrategy::PerformanceBased,
                adaptation_frequency: Duration::from_millis(100),
                sensitivity: 0.8,
                stability_threshold: 0.95,
                rollback_enabled: true,
                learning_rate: 0.1,
                decay_factor: 0.99,
            },
            monitor: PerformanceMonitor::high_frequency(),
            selector: AlgorithmSelector::performance_focused(),
            optimizer: ParameterOptimizer::speed_optimized(),
        }
    }

    /// Compression ratio-optimized adaptive compression
    pub fn ratio_optimized() -> Self {
        Self {
            controller: AdaptiveController {
                enabled: true,
                strategy: AdaptiveStrategy::CompressionRatioBased,
                adaptation_frequency: Duration::from_secs(5),
                sensitivity: 0.6,
                stability_threshold: 0.9,
                rollback_enabled: true,
                learning_rate: 0.05,
                decay_factor: 0.995,
            },
            monitor: PerformanceMonitor::ratio_focused(),
            selector: AlgorithmSelector::ratio_focused(),
            optimizer: ParameterOptimizer::ratio_optimized(),
        }
    }

    /// Balanced adaptive compression
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Latency-optimized adaptive compression
    pub fn latency_optimized() -> Self {
        Self {
            controller: AdaptiveController {
                enabled: true,
                strategy: AdaptiveStrategy::LatencyBased,
                adaptation_frequency: Duration::from_millis(50),
                sensitivity: 0.9,
                stability_threshold: 0.98,
                rollback_enabled: true,
                learning_rate: 0.15,
                decay_factor: 0.98,
            },
            monitor: PerformanceMonitor::low_latency(),
            selector: AlgorithmSelector::latency_focused(),
            optimizer: ParameterOptimizer::latency_optimized(),
        }
    }
}

/// Adaptive compression controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveController {
    /// Enable adaptive compression
    pub enabled: bool,
    /// Adaptation strategy
    pub strategy: AdaptiveStrategy,
    /// Adaptation frequency
    pub adaptation_frequency: Duration,
    /// Sensitivity to performance changes
    pub sensitivity: f64,
    /// Stability threshold
    pub stability_threshold: f64,
    /// Enable rollback on performance degradation
    pub rollback_enabled: bool,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Decay factor for historical data
    pub decay_factor: f64,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: AdaptiveStrategy::HybridOptimization,
            adaptation_frequency: Duration::from_secs(1),
            sensitivity: 0.7,
            stability_threshold: 0.9,
            rollback_enabled: true,
            learning_rate: 0.1,
            decay_factor: 0.99,
        }
    }
}

/// Adaptive compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    /// Performance-based adaptation
    PerformanceBased,
    /// Compression ratio-based adaptation
    CompressionRatioBased,
    /// Latency-based adaptation
    LatencyBased,
    /// Memory usage-based adaptation
    MemoryBased,
    /// Hybrid optimization
    HybridOptimization,
    /// Machine learning-based adaptation
    MachineLearning(MLConfig),
    /// Custom strategy
    Custom(CustomStrategy),
}

/// Machine learning configuration for adaptive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Model type
    pub model_type: MLModelType,
    /// Feature set
    pub features: Vec<String>,
    /// Training parameters
    pub training: MLTraining,
    /// Prediction parameters
    pub prediction: MLPrediction,
}

/// Machine learning model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    LinearRegression,
    DecisionTree,
    RandomForest,
    NeuralNetwork,
    ReinforcementLearning,
}

/// ML training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLTraining {
    /// Training frequency
    pub frequency: Duration,
    /// Training data size
    pub data_size: usize,
    /// Validation split
    pub validation_split: f64,
    /// Early stopping
    pub early_stopping: bool,
}

/// ML prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPrediction {
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Ensemble predictions
    pub ensemble: bool,
    /// Fallback strategy
    pub fallback: String,
}

/// Custom adaptive strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Evaluation function
    pub evaluation_function: String,
}

/// Performance monitor for adaptive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitor {
    /// Monitoring configuration
    pub config: MonitoringConfig,
    /// Metrics collection
    pub metrics: MetricsCollection,
    /// Performance history
    pub history: PerformanceHistory,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetection,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self {
            config: MonitoringConfig::default(),
            metrics: MetricsCollection::default(),
            history: PerformanceHistory::default(),
            anomaly_detection: AnomalyDetection::default(),
        }
    }
}

impl PerformanceMonitor {
    /// High-frequency monitoring configuration
    pub fn high_frequency() -> Self {
        Self {
            config: MonitoringConfig {
                sample_interval: Duration::from_millis(10),
                buffer_size: 10000,
                aggregation_window: Duration::from_secs(1),
                metrics_enabled: true,
                detailed_logging: true,
            },
            metrics: MetricsCollection::comprehensive(),
            history: PerformanceHistory::high_resolution(),
            anomaly_detection: AnomalyDetection::sensitive(),
        }
    }

    /// Ratio-focused monitoring
    pub fn ratio_focused() -> Self {
        Self {
            config: MonitoringConfig::default(),
            metrics: MetricsCollection::ratio_focused(),
            history: PerformanceHistory::default(),
            anomaly_detection: AnomalyDetection::default(),
        }
    }

    /// Low-latency monitoring
    pub fn low_latency() -> Self {
        Self {
            config: MonitoringConfig {
                sample_interval: Duration::from_micros(100),
                buffer_size: 1000,
                aggregation_window: Duration::from_millis(100),
                metrics_enabled: true,
                detailed_logging: false,
            },
            metrics: MetricsCollection::latency_focused(),
            history: PerformanceHistory::minimal(),
            anomaly_detection: AnomalyDetection::fast(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Sample interval
    pub sample_interval: Duration,
    /// Buffer size for samples
    pub buffer_size: usize,
    /// Aggregation window
    pub aggregation_window: Duration,
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Enable detailed logging
    pub detailed_logging: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            buffer_size: 1000,
            aggregation_window: Duration::from_secs(5),
            metrics_enabled: true,
            detailed_logging: false,
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollection {
    /// Enabled metrics
    pub enabled_metrics: Vec<MetricType>,
    /// Metric weights
    pub metric_weights: HashMap<String, f64>,
    /// Aggregation methods
    pub aggregation_methods: HashMap<String, AggregationMethod>,
}

impl Default for MetricsCollection {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                MetricType::CompressionRatio,
                MetricType::CompressionSpeed,
                MetricType::DecompressionSpeed,
                MetricType::MemoryUsage,
                MetricType::CpuUsage,
            ],
            metric_weights: {
                let mut weights = HashMap::new();
                weights.insert("compression_ratio".to_string(), 0.3);
                weights.insert("compression_speed".to_string(), 0.25);
                weights.insert("decompression_speed".to_string(), 0.25);
                weights.insert("memory_usage".to_string(), 0.1);
                weights.insert("cpu_usage".to_string(), 0.1);
                weights
            },
            aggregation_methods: HashMap::new(),
        }
    }
}

impl MetricsCollection {
    /// Comprehensive metrics collection
    pub fn comprehensive() -> Self {
        Self {
            enabled_metrics: vec![
                MetricType::CompressionRatio,
                MetricType::CompressionSpeed,
                MetricType::DecompressionSpeed,
                MetricType::MemoryUsage,
                MetricType::CpuUsage,
                MetricType::Latency,
                MetricType::Throughput,
                MetricType::ErrorRate,
                MetricType::CacheHitRate,
            ],
            metric_weights: {
                let mut weights = HashMap::new();
                weights.insert("compression_ratio".to_string(), 0.2);
                weights.insert("compression_speed".to_string(), 0.15);
                weights.insert("decompression_speed".to_string(), 0.15);
                weights.insert("memory_usage".to_string(), 0.1);
                weights.insert("cpu_usage".to_string(), 0.1);
                weights.insert("latency".to_string(), 0.15);
                weights.insert("throughput".to_string(), 0.1);
                weights.insert("error_rate".to_string(), 0.05);
                weights
            },
            aggregation_methods: HashMap::new(),
        }
    }

    /// Ratio-focused metrics
    pub fn ratio_focused() -> Self {
        Self {
            enabled_metrics: vec![
                MetricType::CompressionRatio,
                MetricType::CompressionSpeed,
                MetricType::MemoryUsage,
            ],
            metric_weights: {
                let mut weights = HashMap::new();
                weights.insert("compression_ratio".to_string(), 0.7);
                weights.insert("compression_speed".to_string(), 0.2);
                weights.insert("memory_usage".to_string(), 0.1);
                weights
            },
            aggregation_methods: HashMap::new(),
        }
    }

    /// Latency-focused metrics
    pub fn latency_focused() -> Self {
        Self {
            enabled_metrics: vec![
                MetricType::Latency,
                MetricType::CompressionSpeed,
                MetricType::DecompressionSpeed,
            ],
            metric_weights: {
                let mut weights = HashMap::new();
                weights.insert("latency".to_string(), 0.6);
                weights.insert("compression_speed".to_string(), 0.2);
                weights.insert("decompression_speed".to_string(), 0.2);
                weights
            },
            aggregation_methods: HashMap::new(),
        }
    }
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    CompressionRatio,
    CompressionSpeed,
    DecompressionSpeed,
    MemoryUsage,
    CpuUsage,
    Latency,
    Throughput,
    ErrorRate,
    CacheHitRate,
    PowerConsumption,
    NetworkUtilization,
}

/// Aggregation methods for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Average,
    WeightedAverage,
    Median,
    Percentile(f64),
    Min,
    Max,
    Sum,
}

/// Performance history management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// History retention duration
    pub retention_duration: Duration,
    /// Maximum history entries
    pub max_entries: usize,
    /// Compression for historical data
    pub historical_compression: bool,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self {
            retention_duration: Duration::from_secs(3600), // 1 hour
            max_entries: 10000,
            historical_compression: true,
            sampling_strategy: SamplingStrategy::Uniform,
        }
    }
}

impl PerformanceHistory {
    /// High-resolution history
    pub fn high_resolution() -> Self {
        Self {
            retention_duration: Duration::from_secs(1800), // 30 minutes
            max_entries: 100000,
            historical_compression: true,
            sampling_strategy: SamplingStrategy::Adaptive,
        }
    }

    /// Minimal history
    pub fn minimal() -> Self {
        Self {
            retention_duration: Duration::from_secs(300), // 5 minutes
            max_entries: 1000,
            historical_compression: false,
            sampling_strategy: SamplingStrategy::LastN(100),
        }
    }
}

/// Sampling strategies for performance history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Uniform,
    Adaptive,
    LastN(usize),
    TimeWindowed(Duration),
    Statistical(StatisticalSampling),
}

/// Statistical sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSampling {
    /// Sampling rate
    pub rate: f64,
    /// Minimum samples
    pub min_samples: usize,
    /// Maximum samples
    pub max_samples: usize,
    /// Variance threshold
    pub variance_threshold: f64,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithm
    pub algorithm: AnomalyAlgorithm,
    /// Sensitivity threshold
    pub sensitivity: f64,
    /// Response actions
    pub response_actions: Vec<AnomalyResponse>,
}

impl Default for AnomalyDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyAlgorithm::StatisticalOutlier,
            sensitivity: 0.95,
            response_actions: vec![
                AnomalyResponse::Log,
                AnomalyResponse::Adjust,
            ],
        }
    }
}

impl AnomalyDetection {
    /// Sensitive anomaly detection
    pub fn sensitive() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyAlgorithm::IsolationForest,
            sensitivity: 0.99,
            response_actions: vec![
                AnomalyResponse::Log,
                AnomalyResponse::Alert,
                AnomalyResponse::Adjust,
            ],
        }
    }

    /// Fast anomaly detection
    pub fn fast() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyAlgorithm::ZScore,
            sensitivity: 0.9,
            response_actions: vec![AnomalyResponse::Adjust],
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    StatisticalOutlier,
    ZScore,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
}

/// Anomaly response actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyResponse {
    Log,
    Alert,
    Adjust,
    Rollback,
    Ignore,
}

/// Algorithm selector for adaptive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSelector {
    /// Selection strategy
    pub strategy: SelectionStrategy,
    /// Selection criteria
    pub criteria: SelectionCriteria,
    /// Selection frequency
    pub frequency: SelectionFrequency,
    /// Candidate algorithms
    pub candidates: Vec<String>,
}

impl Default for AlgorithmSelector {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::Performance,
            criteria: SelectionCriteria::default(),
            frequency: SelectionFrequency::Adaptive,
            candidates: vec![
                "zstd".to_string(),
                "lz4".to_string(),
                "snappy".to_string(),
                "gzip".to_string(),
            ],
        }
    }
}

impl AlgorithmSelector {
    /// Performance-focused selector
    pub fn performance_focused() -> Self {
        Self {
            strategy: SelectionStrategy::Performance,
            criteria: SelectionCriteria::performance_optimized(),
            frequency: SelectionFrequency::High,
            candidates: vec!["lz4".to_string(), "snappy".to_string()],
        }
    }

    /// Ratio-focused selector
    pub fn ratio_focused() -> Self {
        Self {
            strategy: SelectionStrategy::CompressionRatio,
            criteria: SelectionCriteria::ratio_optimized(),
            frequency: SelectionFrequency::Low,
            candidates: vec!["zstd".to_string(), "brotli".to_string()],
        }
    }

    /// Latency-focused selector
    pub fn latency_focused() -> Self {
        Self {
            strategy: SelectionStrategy::Latency,
            criteria: SelectionCriteria::latency_optimized(),
            frequency: SelectionFrequency::VeryHigh,
            candidates: vec!["snappy".to_string(), "lz4".to_string()],
        }
    }
}

/// Algorithm selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    Performance,
    CompressionRatio,
    Latency,
    Balanced,
    MultiObjective,
    Reinforcement,
}

/// Selection criteria configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Optimization objective
    pub objective: OptimizationObjective,
    /// Performance thresholds
    pub thresholds: HashMap<String, f64>,
    /// Weights for different metrics
    pub weights: HashMap<String, f64>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            objective: OptimizationObjective::Balanced,
            thresholds: HashMap::new(),
            weights: {
                let mut weights = HashMap::new();
                weights.insert("compression_ratio".to_string(), 0.3);
                weights.insert("speed".to_string(), 0.4);
                weights.insert("memory".to_string(), 0.2);
                weights.insert("latency".to_string(), 0.1);
                weights
            },
        }
    }
}

impl SelectionCriteria {
    /// Performance-optimized criteria
    pub fn performance_optimized() -> Self {
        Self {
            objective: OptimizationObjective::Speed,
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("min_speed_mbps".to_string(), 100.0);
                thresholds.insert("max_latency_ms".to_string(), 5.0);
                thresholds
            },
            weights: {
                let mut weights = HashMap::new();
                weights.insert("speed".to_string(), 0.6);
                weights.insert("latency".to_string(), 0.3);
                weights.insert("memory".to_string(), 0.1);
                weights
            },
        }
    }

    /// Ratio-optimized criteria
    pub fn ratio_optimized() -> Self {
        Self {
            objective: OptimizationObjective::CompressionRatio,
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("min_compression_ratio".to_string(), 2.0);
                thresholds
            },
            weights: {
                let mut weights = HashMap::new();
                weights.insert("compression_ratio".to_string(), 0.8);
                weights.insert("speed".to_string(), 0.2);
                weights
            },
        }
    }

    /// Latency-optimized criteria
    pub fn latency_optimized() -> Self {
        Self {
            objective: OptimizationObjective::Latency,
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("max_latency_ms".to_string(), 1.0);
                thresholds
            },
            weights: {
                let mut weights = HashMap::new();
                weights.insert("latency".to_string(), 0.8);
                weights.insert("speed".to_string(), 0.2);
                weights
            },
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    Speed,
    CompressionRatio,
    Latency,
    Memory,
    Balanced,
    PowerEfficiency,
}

/// Selection frequency settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionFrequency {
    VeryHigh,   // Every operation
    High,       // Every 10 operations
    Medium,     // Every 100 operations
    Low,        // Every 1000 operations
    VeryLow,    // Every 10000 operations
    Adaptive,   // Based on performance variance
    Time(Duration), // Time-based frequency
}

/// Parameter optimizer for compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterOptimizer {
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Parameter search space
    pub search_space: ParameterSearchSpace,
    /// Optimization constraints
    pub constraints: OptimizationConstraints,
    /// Convergence criteria
    pub convergence: ConvergenceCriteria,
}

impl Default for ParameterOptimizer {
    fn default() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::GradientDescent,
            search_space: ParameterSearchSpace::default(),
            constraints: OptimizationConstraints::default(),
            convergence: ConvergenceCriteria::default(),
        }
    }
}

impl ParameterOptimizer {
    /// Speed-optimized parameter optimizer
    pub fn speed_optimized() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::RandomSearch,
            search_space: ParameterSearchSpace::speed_focused(),
            constraints: OptimizationConstraints::performance_constraints(),
            convergence: ConvergenceCriteria::fast_convergence(),
        }
    }

    /// Ratio-optimized parameter optimizer
    pub fn ratio_optimized() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::BayesianOptimization,
            search_space: ParameterSearchSpace::ratio_focused(),
            constraints: OptimizationConstraints::ratio_constraints(),
            convergence: ConvergenceCriteria::thorough_convergence(),
        }
    }

    /// Latency-optimized parameter optimizer
    pub fn latency_optimized() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::HillClimbing,
            search_space: ParameterSearchSpace::latency_focused(),
            constraints: OptimizationConstraints::latency_constraints(),
            convergence: ConvergenceCriteria::quick_convergence(),
        }
    }
}

/// Optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    GradientDescent,
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    GeneticAlgorithm,
    SimulatedAnnealing,
    ParticleSwarm,
    HillClimbing,
}

/// Parameter search space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSearchSpace {
    /// Parameter ranges
    pub ranges: HashMap<String, ParameterRange>,
    /// Parameter types
    pub types: HashMap<String, ParameterType>,
    /// Parameter dependencies
    pub dependencies: Vec<ParameterDependency>,
}

impl Default for ParameterSearchSpace {
    fn default() -> Self {
        Self {
            ranges: HashMap::new(),
            types: HashMap::new(),
            dependencies: Vec::new(),
        }
    }
}

impl ParameterSearchSpace {
    /// Speed-focused search space
    pub fn speed_focused() -> Self {
        let mut ranges = HashMap::new();
        ranges.insert("compression_level".to_string(), ParameterRange::Integer { min: 1, max: 3 });
        ranges.insert("block_size".to_string(), ParameterRange::Integer { min: 4096, max: 65536 });

        Self {
            ranges,
            types: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    /// Ratio-focused search space
    pub fn ratio_focused() -> Self {
        let mut ranges = HashMap::new();
        ranges.insert("compression_level".to_string(), ParameterRange::Integer { min: 9, max: 22 });
        ranges.insert("window_size".to_string(), ParameterRange::Integer { min: 15, max: 27 });

        Self {
            ranges,
            types: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    /// Latency-focused search space
    pub fn latency_focused() -> Self {
        let mut ranges = HashMap::new();
        ranges.insert("compression_level".to_string(), ParameterRange::Integer { min: 1, max: 1 });
        ranges.insert("buffer_size".to_string(), ParameterRange::Integer { min: 1024, max: 4096 });

        Self {
            ranges,
            types: HashMap::new(),
            dependencies: Vec::new(),
        }
    }
}

/// Parameter range types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    Integer { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Categorical { values: Vec<String> },
    Boolean,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Enum(Vec<String>),
}

/// Parameter dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDependency {
    /// Parameter name
    pub parameter: String,
    /// Dependent parameter
    pub depends_on: String,
    /// Dependency condition
    pub condition: DependencyCondition,
}

/// Dependency conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyCondition {
    Equals(String),
    GreaterThan(f64),
    LessThan(f64),
    InRange { min: f64, max: f64 },
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    /// Maximum optimization time
    pub max_time: Duration,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Performance constraints
    pub performance_constraints: HashMap<String, f64>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            max_time: Duration::from_secs(60),
            max_iterations: 100,
            performance_constraints: HashMap::new(),
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

impl OptimizationConstraints {
    /// Performance constraints
    pub fn performance_constraints() -> Self {
        let mut constraints = HashMap::new();
        constraints.insert("min_speed_mbps".to_string(), 50.0);
        constraints.insert("max_memory_mb".to_string(), 100.0);

        Self {
            max_time: Duration::from_secs(30),
            max_iterations: 50,
            performance_constraints: constraints,
            resource_constraints: ResourceConstraints::performance_focused(),
        }
    }

    /// Ratio constraints
    pub fn ratio_constraints() -> Self {
        let mut constraints = HashMap::new();
        constraints.insert("min_compression_ratio".to_string(), 1.5);

        Self {
            max_time: Duration::from_secs(300),
            max_iterations: 1000,
            performance_constraints: constraints,
            resource_constraints: ResourceConstraints::default(),
        }
    }

    /// Latency constraints
    pub fn latency_constraints() -> Self {
        let mut constraints = HashMap::new();
        constraints.insert("max_latency_ms".to_string(), 1.0);

        Self {
            max_time: Duration::from_secs(10),
            max_iterations: 20,
            performance_constraints: constraints,
            resource_constraints: ResourceConstraints::latency_focused(),
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Maximum CPU usage (percentage)
    pub max_cpu: f64,
    /// Maximum disk usage (bytes)
    pub max_disk: usize,
    /// Maximum network bandwidth (bytes/sec)
    pub max_bandwidth: usize,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory: 100 * 1024 * 1024, // 100MB
            max_cpu: 0.5, // 50%
            max_disk: 1024 * 1024 * 1024, // 1GB
            max_bandwidth: 10 * 1024 * 1024, // 10MB/s
        }
    }
}

impl ResourceConstraints {
    /// Performance-focused resource constraints
    pub fn performance_focused() -> Self {
        Self {
            max_memory: 500 * 1024 * 1024, // 500MB
            max_cpu: 0.8, // 80%
            max_disk: 5 * 1024 * 1024 * 1024, // 5GB
            max_bandwidth: 100 * 1024 * 1024, // 100MB/s
        }
    }

    /// Latency-focused resource constraints
    pub fn latency_focused() -> Self {
        Self {
            max_memory: 50 * 1024 * 1024, // 50MB
            max_cpu: 0.3, // 30%
            max_disk: 100 * 1024 * 1024, // 100MB
            max_bandwidth: 1024 * 1024, // 1MB/s
        }
    }
}

/// Convergence criteria for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Stability window size
    pub stability_window: usize,
    /// Maximum no-improvement iterations
    pub max_no_improvement: usize,
    /// Tolerance for objective function
    pub tolerance: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            min_improvement: 0.01,
            stability_window: 10,
            max_no_improvement: 20,
            tolerance: 1e-6,
        }
    }
}

impl ConvergenceCriteria {
    /// Fast convergence criteria
    pub fn fast_convergence() -> Self {
        Self {
            min_improvement: 0.05,
            stability_window: 3,
            max_no_improvement: 5,
            tolerance: 1e-3,
        }
    }

    /// Thorough convergence criteria
    pub fn thorough_convergence() -> Self {
        Self {
            min_improvement: 0.001,
            stability_window: 50,
            max_no_improvement: 100,
            tolerance: 1e-9,
        }
    }

    /// Quick convergence criteria
    pub fn quick_convergence() -> Self {
        Self {
            min_improvement: 0.1,
            stability_window: 2,
            max_no_improvement: 3,
            tolerance: 1e-2,
        }
    }
}