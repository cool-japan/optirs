// Compression Analytics and Performance Analysis
//
// This module handles analytics for compression performance, metrics collection,
// and reporting for TPU event synchronization systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Compression analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalytics {
    /// Analytics configuration
    pub config: AnalyticsConfig,
    /// Metrics collector
    pub collector: MetricsCollector,
    /// Performance analyzer
    pub analyzer: PerformanceAnalyzer,
    /// Report generator
    pub reporter: ReportGenerator,
    /// Analytics storage
    pub storage: AnalyticsStorage,
}

impl Default for CompressionAnalytics {
    fn default() -> Self {
        Self {
            config: AnalyticsConfig::default(),
            collector: MetricsCollector::default(),
            analyzer: PerformanceAnalyzer::default(),
            reporter: ReportGenerator::default(),
            storage: AnalyticsStorage::default(),
        }
    }
}

impl CompressionAnalytics {
    /// Performance-focused analytics
    pub fn performance_focused() -> Self {
        Self {
            config: AnalyticsConfig::performance_focused(),
            collector: MetricsCollector::performance_focused(),
            analyzer: PerformanceAnalyzer::performance_focused(),
            reporter: ReportGenerator::performance_focused(),
            storage: AnalyticsStorage::high_frequency(),
        }
    }

    /// Ratio-focused analytics
    pub fn ratio_focused() -> Self {
        Self {
            config: AnalyticsConfig::ratio_focused(),
            collector: MetricsCollector::ratio_focused(),
            analyzer: PerformanceAnalyzer::ratio_focused(),
            reporter: ReportGenerator::ratio_focused(),
            storage: AnalyticsStorage::detailed(),
        }
    }

    /// Balanced analytics
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Latency-focused analytics
    pub fn latency_focused() -> Self {
        Self {
            config: AnalyticsConfig::latency_focused(),
            collector: MetricsCollector::latency_focused(),
            analyzer: PerformanceAnalyzer::latency_focused(),
            reporter: ReportGenerator::latency_focused(),
            storage: AnalyticsStorage::real_time(),
        }
    }
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable analytics
    pub enabled: bool,
    /// Analytics level
    pub level: AnalyticsLevel,
    /// Collection interval
    pub collection_interval: Duration,
    /// Analysis interval
    pub analysis_interval: Duration,
    /// Reporting interval
    pub reporting_interval: Duration,
    /// Data retention period
    pub retention_period: Duration,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: AnalyticsLevel::Standard,
            collection_interval: Duration::from_secs(30),
            analysis_interval: Duration::from_secs(300),
            reporting_interval: Duration::from_secs(3600),
            retention_period: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl AnalyticsConfig {
    /// Performance-focused configuration
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            level: AnalyticsLevel::Detailed,
            collection_interval: Duration::from_secs(5),
            analysis_interval: Duration::from_secs(60),
            reporting_interval: Duration::from_secs(300),
            retention_period: Duration::from_secs(43200), // 12 hours
        }
    }

    /// Ratio-focused configuration
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            level: AnalyticsLevel::Comprehensive,
            collection_interval: Duration::from_secs(60),
            analysis_interval: Duration::from_secs(600),
            reporting_interval: Duration::from_secs(3600),
            retention_period: Duration::from_secs(172800), // 48 hours
        }
    }

    /// Latency-focused configuration
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            level: AnalyticsLevel::RealTime,
            collection_interval: Duration::from_secs(1),
            analysis_interval: Duration::from_secs(10),
            reporting_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(7200), // 2 hours
        }
    }
}

/// Analytics levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsLevel {
    None,
    Basic,
    Standard,
    Detailed,
    Comprehensive,
    RealTime,
}

/// Metrics collector for compression analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollector {
    /// Collection configuration
    pub config: CollectionConfig,
    /// Metrics definitions
    pub metrics: CompressionMetrics,
    /// Quality metrics
    pub quality: QualityMetrics,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            config: CollectionConfig::default(),
            metrics: CompressionMetrics::default(),
            quality: QualityMetrics::default(),
            performance: PerformanceMetrics::default(),
        }
    }
}

impl MetricsCollector {
    /// Performance-focused collector
    pub fn performance_focused() -> Self {
        Self {
            config: CollectionConfig::high_frequency(),
            metrics: CompressionMetrics::performance_focused(),
            quality: QualityMetrics::minimal(),
            performance: PerformanceMetrics::comprehensive(),
        }
    }

    /// Ratio-focused collector
    pub fn ratio_focused() -> Self {
        Self {
            config: CollectionConfig::detailed(),
            metrics: CompressionMetrics::ratio_focused(),
            quality: QualityMetrics::comprehensive(),
            performance: PerformanceMetrics::basic(),
        }
    }

    /// Latency-focused collector
    pub fn latency_focused() -> Self {
        Self {
            config: CollectionConfig::real_time(),
            metrics: CompressionMetrics::latency_focused(),
            quality: QualityMetrics::minimal(),
            performance: PerformanceMetrics::latency_focused(),
        }
    }
}

/// Collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Enable collection
    pub enabled: bool,
    /// Collection frequency
    pub frequency: Duration,
    /// Buffer size for collected data
    pub buffer_size: usize,
    /// Sampling strategy
    pub sampling: SamplingStrategy,
    /// Aggregation method
    pub aggregation: AggregationMethod,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(30),
            buffer_size: 1000,
            sampling: SamplingStrategy::Uniform,
            aggregation: AggregationMethod::Average,
        }
    }
}

impl CollectionConfig {
    /// High-frequency collection
    pub fn high_frequency() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(5),
            buffer_size: 10000,
            sampling: SamplingStrategy::Adaptive,
            aggregation: AggregationMethod::WeightedAverage,
        }
    }

    /// Detailed collection
    pub fn detailed() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(60),
            buffer_size: 5000,
            sampling: SamplingStrategy::Stratified,
            aggregation: AggregationMethod::Percentile(95.0),
        }
    }

    /// Real-time collection
    pub fn real_time() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(1),
            buffer_size: 1000,
            sampling: SamplingStrategy::All,
            aggregation: AggregationMethod::MovingAverage,
        }
    }
}

/// Sampling strategies for metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    All,
    Uniform,
    Random,
    Stratified,
    Adaptive,
    Threshold(f64),
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
    MovingAverage,
}

/// Compression metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    /// Enable compression ratio tracking
    pub compression_ratio: bool,
    /// Enable compression speed tracking
    pub compression_speed: bool,
    /// Enable decompression speed tracking
    pub decompression_speed: bool,
    /// Enable throughput tracking
    pub throughput: bool,
    /// Enable latency tracking
    pub latency: bool,
    /// Enable error rate tracking
    pub error_rate: bool,
    /// Enable algorithm usage tracking
    pub algorithm_usage: bool,
    /// Enable resource usage tracking
    pub resource_usage: bool,
}

impl Default for CompressionMetrics {
    fn default() -> Self {
        Self {
            compression_ratio: true,
            compression_speed: true,
            decompression_speed: true,
            throughput: true,
            latency: true,
            error_rate: true,
            algorithm_usage: true,
            resource_usage: false,
        }
    }
}

impl CompressionMetrics {
    /// Performance-focused metrics
    pub fn performance_focused() -> Self {
        Self {
            compression_ratio: false,
            compression_speed: true,
            decompression_speed: true,
            throughput: true,
            latency: true,
            error_rate: true,
            algorithm_usage: true,
            resource_usage: true,
        }
    }

    /// Ratio-focused metrics
    pub fn ratio_focused() -> Self {
        Self {
            compression_ratio: true,
            compression_speed: false,
            decompression_speed: false,
            throughput: false,
            latency: false,
            error_rate: true,
            algorithm_usage: true,
            resource_usage: false,
        }
    }

    /// Latency-focused metrics
    pub fn latency_focused() -> Self {
        Self {
            compression_ratio: false,
            compression_speed: true,
            decompression_speed: true,
            throughput: true,
            latency: true,
            error_rate: true,
            algorithm_usage: false,
            resource_usage: false,
        }
    }
}

/// Quality metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Enable data integrity checking
    pub data_integrity: bool,
    /// Enable compression accuracy tracking
    pub compression_accuracy: bool,
    /// Enable algorithm reliability tracking
    pub algorithm_reliability: bool,
    /// Enable consistency tracking
    pub consistency: bool,
    /// Enable stability tracking
    pub stability: bool,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            data_integrity: true,
            compression_accuracy: true,
            algorithm_reliability: true,
            consistency: false,
            stability: false,
        }
    }
}

impl QualityMetrics {
    /// Minimal quality metrics
    pub fn minimal() -> Self {
        Self {
            data_integrity: true,
            compression_accuracy: false,
            algorithm_reliability: false,
            consistency: false,
            stability: false,
        }
    }

    /// Comprehensive quality metrics
    pub fn comprehensive() -> Self {
        Self {
            data_integrity: true,
            compression_accuracy: true,
            algorithm_reliability: true,
            consistency: true,
            stability: true,
        }
    }
}

/// Performance metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Enable CPU usage tracking
    pub cpu_usage: bool,
    /// Enable memory usage tracking
    pub memory_usage: bool,
    /// Enable disk I/O tracking
    pub disk_io: bool,
    /// Enable network I/O tracking
    pub network_io: bool,
    /// Enable cache performance tracking
    pub cache_performance: bool,
    /// Enable bottleneck detection
    pub bottleneck_detection: bool,
    /// Enable efficiency tracking
    pub efficiency: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: true,
            memory_usage: true,
            disk_io: false,
            network_io: false,
            cache_performance: false,
            bottleneck_detection: false,
            efficiency: false,
        }
    }
}

impl PerformanceMetrics {
    /// Basic performance metrics
    pub fn basic() -> Self {
        Self {
            cpu_usage: true,
            memory_usage: true,
            disk_io: false,
            network_io: false,
            cache_performance: false,
            bottleneck_detection: false,
            efficiency: false,
        }
    }

    /// Comprehensive performance metrics
    pub fn comprehensive() -> Self {
        Self {
            cpu_usage: true,
            memory_usage: true,
            disk_io: true,
            network_io: true,
            cache_performance: true,
            bottleneck_detection: true,
            efficiency: true,
        }
    }

    /// Latency-focused performance metrics
    pub fn latency_focused() -> Self {
        Self {
            cpu_usage: true,
            memory_usage: false,
            disk_io: false,
            network_io: false,
            cache_performance: true,
            bottleneck_detection: true,
            efficiency: false,
        }
    }
}

/// Performance analyzer for compression analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    pub config: AnalysisConfig,
    /// Trend analysis settings
    pub trend_analysis: TrendAnalysis,
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetection,
    /// Correlation analysis settings
    pub correlation_analysis: CorrelationAnalysis,
    /// Prediction settings
    pub prediction: PredictionAnalysis,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self {
            config: AnalysisConfig::default(),
            trend_analysis: TrendAnalysis::default(),
            anomaly_detection: AnomalyDetection::default(),
            correlation_analysis: CorrelationAnalysis::default(),
            prediction: PredictionAnalysis::default(),
        }
    }
}

impl PerformanceAnalyzer {
    /// Performance-focused analyzer
    pub fn performance_focused() -> Self {
        Self {
            config: AnalysisConfig::performance_focused(),
            trend_analysis: TrendAnalysis::performance_focused(),
            anomaly_detection: AnomalyDetection::performance_focused(),
            correlation_analysis: CorrelationAnalysis::performance_focused(),
            prediction: PredictionAnalysis::performance_focused(),
        }
    }

    /// Ratio-focused analyzer
    pub fn ratio_focused() -> Self {
        Self {
            config: AnalysisConfig::ratio_focused(),
            trend_analysis: TrendAnalysis::ratio_focused(),
            anomaly_detection: AnomalyDetection::ratio_focused(),
            correlation_analysis: CorrelationAnalysis::ratio_focused(),
            prediction: PredictionAnalysis::ratio_focused(),
        }
    }

    /// Latency-focused analyzer
    pub fn latency_focused() -> Self {
        Self {
            config: AnalysisConfig::latency_focused(),
            trend_analysis: TrendAnalysis::latency_focused(),
            anomaly_detection: AnomalyDetection::latency_focused(),
            correlation_analysis: CorrelationAnalysis::latency_focused(),
            prediction: PredictionAnalysis::latency_focused(),
        }
    }
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable analysis
    pub enabled: bool,
    /// Analysis algorithms
    pub algorithms: Vec<AnalysisAlgorithm>,
    /// Analysis window size
    pub window_size: Duration,
    /// Analysis frequency
    pub frequency: Duration,
    /// Statistical significance level
    pub significance_level: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnalysisAlgorithm::TrendAnalysis,
                AnalysisAlgorithm::AnomalyDetection,
            ],
            window_size: Duration::from_secs(3600), // 1 hour
            frequency: Duration::from_secs(300), // 5 minutes
            significance_level: 0.05,
        }
    }
}

impl AnalysisConfig {
    /// Performance-focused analysis configuration
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnalysisAlgorithm::TrendAnalysis,
                AnalysisAlgorithm::AnomalyDetection,
                AnalysisAlgorithm::BottleneckDetection,
            ],
            window_size: Duration::from_secs(1800), // 30 minutes
            frequency: Duration::from_secs(60), // 1 minute
            significance_level: 0.01,
        }
    }

    /// Ratio-focused analysis configuration
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnalysisAlgorithm::TrendAnalysis,
                AnalysisAlgorithm::CorrelationAnalysis,
                AnalysisAlgorithm::OptimizationAnalysis,
            ],
            window_size: Duration::from_secs(7200), // 2 hours
            frequency: Duration::from_secs(600), // 10 minutes
            significance_level: 0.01,
        }
    }

    /// Latency-focused analysis configuration
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnalysisAlgorithm::AnomalyDetection,
                AnalysisAlgorithm::BottleneckDetection,
                AnalysisAlgorithm::PredictiveAnalysis,
            ],
            window_size: Duration::from_secs(600), // 10 minutes
            frequency: Duration::from_secs(30), // 30 seconds
            significance_level: 0.001,
        }
    }
}

/// Analysis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisAlgorithm {
    TrendAnalysis,
    AnomalyDetection,
    CorrelationAnalysis,
    BottleneckDetection,
    OptimizationAnalysis,
    PredictiveAnalysis,
    PatternRecognition,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Enable trend analysis
    pub enabled: bool,
    /// Trend detection method
    pub method: TrendDetectionMethod,
    /// Minimum trend duration
    pub min_duration: Duration,
    /// Trend sensitivity
    pub sensitivity: f64,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            enabled: true,
            method: TrendDetectionMethod::LinearRegression,
            min_duration: Duration::from_secs(300), // 5 minutes
            sensitivity: 0.1,
        }
    }
}

impl TrendAnalysis {
    /// Performance-focused trend analysis
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            method: TrendDetectionMethod::MovingAverage,
            min_duration: Duration::from_secs(60), // 1 minute
            sensitivity: 0.05,
        }
    }

    /// Ratio-focused trend analysis
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            method: TrendDetectionMethod::LinearRegression,
            min_duration: Duration::from_secs(1800), // 30 minutes
            sensitivity: 0.02,
        }
    }

    /// Latency-focused trend analysis
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            method: TrendDetectionMethod::ExponentialSmoothing,
            min_duration: Duration::from_secs(30), // 30 seconds
            sensitivity: 0.01,
        }
    }
}

/// Trend detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDetectionMethod {
    LinearRegression,
    MovingAverage,
    ExponentialSmoothing,
    SeasonalDecomposition,
    ChangePointDetection,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithm
    pub algorithm: AnomalyDetectionAlgorithm,
    /// Detection threshold
    pub threshold: f64,
    /// Window size for detection
    pub window_size: usize,
}

impl Default for AnomalyDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyDetectionAlgorithm::ZScore,
            threshold: 2.0,
            window_size: 100,
        }
    }
}

impl AnomalyDetection {
    /// Performance-focused anomaly detection
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyDetectionAlgorithm::IsolationForest,
            threshold: 1.5,
            window_size: 1000,
        }
    }

    /// Ratio-focused anomaly detection
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
            threshold: 3.0,
            window_size: 500,
        }
    }

    /// Latency-focused anomaly detection
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyDetectionAlgorithm::ZScore,
            threshold: 1.0,
            window_size: 50,
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    ZScore,
    ModifiedZScore,
    IQR,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    StatisticalOutlier,
}

/// Correlation analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    /// Enable correlation analysis
    pub enabled: bool,
    /// Correlation method
    pub method: CorrelationMethod,
    /// Minimum correlation threshold
    pub threshold: f64,
    /// Variables to analyze
    pub variables: Vec<String>,
}

impl Default for CorrelationAnalysis {
    fn default() -> Self {
        Self {
            enabled: false,
            method: CorrelationMethod::Pearson,
            threshold: 0.5,
            variables: vec![
                "compression_ratio".to_string(),
                "compression_speed".to_string(),
                "memory_usage".to_string(),
            ],
        }
    }
}

impl CorrelationAnalysis {
    /// Performance-focused correlation analysis
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            method: CorrelationMethod::Spearman,
            threshold: 0.3,
            variables: vec![
                "compression_speed".to_string(),
                "cpu_usage".to_string(),
                "memory_usage".to_string(),
                "throughput".to_string(),
            ],
        }
    }

    /// Ratio-focused correlation analysis
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            method: CorrelationMethod::Pearson,
            threshold: 0.6,
            variables: vec![
                "compression_ratio".to_string(),
                "compression_level".to_string(),
                "algorithm_type".to_string(),
                "data_type".to_string(),
            ],
        }
    }

    /// Latency-focused correlation analysis
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            method: CorrelationMethod::Kendall,
            threshold: 0.4,
            variables: vec![
                "latency".to_string(),
                "compression_speed".to_string(),
                "buffer_size".to_string(),
                "batch_size".to_string(),
            ],
        }
    }
}

/// Correlation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
    MutualInformation,
    DistanceCorrelation,
}

/// Prediction analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAnalysis {
    /// Enable prediction analysis
    pub enabled: bool,
    /// Prediction model
    pub model: PredictionModel,
    /// Prediction horizon
    pub horizon: Duration,
    /// Model update frequency
    pub update_frequency: Duration,
}

impl Default for PredictionAnalysis {
    fn default() -> Self {
        Self {
            enabled: false,
            model: PredictionModel::LinearRegression,
            horizon: Duration::from_secs(3600), // 1 hour
            update_frequency: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl PredictionAnalysis {
    /// Performance-focused prediction analysis
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            model: PredictionModel::ARIMA,
            horizon: Duration::from_secs(1800), // 30 minutes
            update_frequency: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Ratio-focused prediction analysis
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            model: PredictionModel::RandomForest,
            horizon: Duration::from_secs(7200), // 2 hours
            update_frequency: Duration::from_secs(1800), // 30 minutes
        }
    }

    /// Latency-focused prediction analysis
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            model: PredictionModel::LSTM,
            horizon: Duration::from_secs(300), // 5 minutes
            update_frequency: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionModel {
    LinearRegression,
    ARIMA,
    ExponentialSmoothing,
    RandomForest,
    GradientBoosting,
    LSTM,
    Prophet,
}

/// Report generator for compression analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGenerator {
    /// Reporting configuration
    pub config: ReportingConfig,
    /// Report templates
    pub templates: ReportTemplates,
    /// Report formats
    pub formats: ReportFormats,
    /// Report distribution
    pub distribution: ReportDistribution,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self {
            config: ReportingConfig::default(),
            templates: ReportTemplates::default(),
            formats: ReportFormats::default(),
            distribution: ReportDistribution::default(),
        }
    }
}

impl ReportGenerator {
    /// Performance-focused report generator
    pub fn performance_focused() -> Self {
        Self {
            config: ReportingConfig::performance_focused(),
            templates: ReportTemplates::performance_focused(),
            formats: ReportFormats::compact(),
            distribution: ReportDistribution::frequent(),
        }
    }

    /// Ratio-focused report generator
    pub fn ratio_focused() -> Self {
        Self {
            config: ReportingConfig::ratio_focused(),
            templates: ReportTemplates::ratio_focused(),
            formats: ReportFormats::detailed(),
            distribution: ReportDistribution::scheduled(),
        }
    }

    /// Latency-focused report generator
    pub fn latency_focused() -> Self {
        Self {
            config: ReportingConfig::latency_focused(),
            templates: ReportTemplates::latency_focused(),
            formats: ReportFormats::real_time(),
            distribution: ReportDistribution::immediate(),
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable reporting
    pub enabled: bool,
    /// Reporting frequency
    pub frequency: Duration,
    /// Report types to generate
    pub report_types: Vec<ReportType>,
    /// Include visualizations
    pub visualizations: bool,
    /// Include recommendations
    pub recommendations: bool,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(3600), // 1 hour
            report_types: vec![
                ReportType::Summary,
                ReportType::Performance,
            ],
            visualizations: false,
            recommendations: false,
        }
    }
}

impl ReportingConfig {
    /// Performance-focused reporting
    pub fn performance_focused() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(300), // 5 minutes
            report_types: vec![
                ReportType::Performance,
                ReportType::RealTime,
            ],
            visualizations: true,
            recommendations: true,
        }
    }

    /// Ratio-focused reporting
    pub fn ratio_focused() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(3600), // 1 hour
            report_types: vec![
                ReportType::Summary,
                ReportType::Analysis,
                ReportType::Optimization,
            ],
            visualizations: true,
            recommendations: true,
        }
    }

    /// Latency-focused reporting
    pub fn latency_focused() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(60), // 1 minute
            report_types: vec![
                ReportType::RealTime,
                ReportType::Alert,
            ],
            visualizations: false,
            recommendations: true,
        }
    }
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Summary,
    Performance,
    Analysis,
    Optimization,
    RealTime,
    Alert,
    Diagnostic,
}

/// Report templates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplates {
    /// Available templates
    pub templates: HashMap<String, ReportTemplate>,
    /// Default template
    pub default_template: String,
    /// Custom templates enabled
    pub custom_enabled: bool,
}

impl Default for ReportTemplates {
    fn default() -> Self {
        let mut templates = HashMap::new();
        templates.insert("standard".to_string(), ReportTemplate::standard());
        templates.insert("summary".to_string(), ReportTemplate::summary());

        Self {
            templates,
            default_template: "standard".to_string(),
            custom_enabled: false,
        }
    }
}

impl ReportTemplates {
    /// Performance-focused templates
    pub fn performance_focused() -> Self {
        let mut templates = HashMap::new();
        templates.insert("performance".to_string(), ReportTemplate::performance());
        templates.insert("real_time".to_string(), ReportTemplate::real_time());

        Self {
            templates,
            default_template: "performance".to_string(),
            custom_enabled: true,
        }
    }

    /// Ratio-focused templates
    pub fn ratio_focused() -> Self {
        let mut templates = HashMap::new();
        templates.insert("compression_analysis".to_string(), ReportTemplate::compression_analysis());
        templates.insert("optimization".to_string(), ReportTemplate::optimization());

        Self {
            templates,
            default_template: "compression_analysis".to_string(),
            custom_enabled: true,
        }
    }

    /// Latency-focused templates
    pub fn latency_focused() -> Self {
        let mut templates = HashMap::new();
        templates.insert("latency_report".to_string(), ReportTemplate::latency());
        templates.insert("alert".to_string(), ReportTemplate::alert());

        Self {
            templates,
            default_template: "latency_report".to_string(),
            custom_enabled: false,
        }
    }
}

/// Report template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template sections
    pub sections: Vec<ReportSection>,
    /// Template format
    pub format: TemplateFormat,
}

impl ReportTemplate {
    /// Standard report template
    pub fn standard() -> Self {
        Self {
            name: "Standard Report".to_string(),
            sections: vec![
                ReportSection::Overview,
                ReportSection::Metrics,
                ReportSection::Performance,
            ],
            format: TemplateFormat::Text,
        }
    }

    /// Summary report template
    pub fn summary() -> Self {
        Self {
            name: "Summary Report".to_string(),
            sections: vec![
                ReportSection::Overview,
                ReportSection::KeyMetrics,
            ],
            format: TemplateFormat::Text,
        }
    }

    /// Performance report template
    pub fn performance() -> Self {
        Self {
            name: "Performance Report".to_string(),
            sections: vec![
                ReportSection::Performance,
                ReportSection::Bottlenecks,
                ReportSection::Optimization,
            ],
            format: TemplateFormat::JSON,
        }
    }

    /// Real-time report template
    pub fn real_time() -> Self {
        Self {
            name: "Real-time Report".to_string(),
            sections: vec![
                ReportSection::CurrentMetrics,
                ReportSection::Alerts,
            ],
            format: TemplateFormat::JSON,
        }
    }

    /// Compression analysis template
    pub fn compression_analysis() -> Self {
        Self {
            name: "Compression Analysis".to_string(),
            sections: vec![
                ReportSection::CompressionRatio,
                ReportSection::AlgorithmComparison,
                ReportSection::Recommendations,
            ],
            format: TemplateFormat::HTML,
        }
    }

    /// Optimization template
    pub fn optimization() -> Self {
        Self {
            name: "Optimization Report".to_string(),
            sections: vec![
                ReportSection::Optimization,
                ReportSection::Recommendations,
                ReportSection::Projections,
            ],
            format: TemplateFormat::HTML,
        }
    }

    /// Latency report template
    pub fn latency() -> Self {
        Self {
            name: "Latency Report".to_string(),
            sections: vec![
                ReportSection::Latency,
                ReportSection::Trends,
                ReportSection::Alerts,
            ],
            format: TemplateFormat::JSON,
        }
    }

    /// Alert template
    pub fn alert() -> Self {
        Self {
            name: "Alert Report".to_string(),
            sections: vec![
                ReportSection::Alerts,
                ReportSection::ImmediateActions,
            ],
            format: TemplateFormat::Text,
        }
    }
}

/// Report sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportSection {
    Overview,
    Metrics,
    KeyMetrics,
    CurrentMetrics,
    Performance,
    CompressionRatio,
    Latency,
    Bottlenecks,
    Optimization,
    AlgorithmComparison,
    Trends,
    Alerts,
    Recommendations,
    Projections,
    ImmediateActions,
}

/// Template formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateFormat {
    Text,
    JSON,
    HTML,
    Markdown,
    CSV,
}

/// Report formats configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportFormats {
    /// Supported formats
    pub formats: Vec<ReportFormat>,
    /// Default format
    pub default_format: ReportFormat,
    /// Format-specific options
    pub format_options: HashMap<String, FormatOptions>,
}

impl Default for ReportFormats {
    fn default() -> Self {
        Self {
            formats: vec![ReportFormat::Text, ReportFormat::JSON],
            default_format: ReportFormat::Text,
            format_options: HashMap::new(),
        }
    }
}

impl ReportFormats {
    /// Compact report formats
    pub fn compact() -> Self {
        Self {
            formats: vec![ReportFormat::JSON],
            default_format: ReportFormat::JSON,
            format_options: HashMap::new(),
        }
    }

    /// Detailed report formats
    pub fn detailed() -> Self {
        Self {
            formats: vec![
                ReportFormat::HTML,
                ReportFormat::PDF,
                ReportFormat::JSON,
            ],
            default_format: ReportFormat::HTML,
            format_options: HashMap::new(),
        }
    }

    /// Real-time report formats
    pub fn real_time() -> Self {
        Self {
            formats: vec![ReportFormat::JSON, ReportFormat::Text],
            default_format: ReportFormat::JSON,
            format_options: HashMap::new(),
        }
    }
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Text,
    JSON,
    XML,
    HTML,
    PDF,
    CSV,
    Excel,
}

/// Format-specific options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatOptions {
    /// Format-specific settings
    pub settings: HashMap<String, String>,
}

/// Report distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistribution {
    /// Distribution channels
    pub channels: Vec<DistributionChannel>,
    /// Default channel
    pub default_channel: DistributionChannel,
    /// Distribution schedule
    pub schedule: DistributionSchedule,
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            channels: vec![DistributionChannel::Log],
            default_channel: DistributionChannel::Log,
            schedule: DistributionSchedule::OnGeneration,
        }
    }
}

impl ReportDistribution {
    /// Frequent distribution
    pub fn frequent() -> Self {
        Self {
            channels: vec![
                DistributionChannel::Log,
                DistributionChannel::File,
            ],
            default_channel: DistributionChannel::Log,
            schedule: DistributionSchedule::Immediate,
        }
    }

    /// Scheduled distribution
    pub fn scheduled() -> Self {
        Self {
            channels: vec![
                DistributionChannel::Email,
                DistributionChannel::File,
                DistributionChannel::Database,
            ],
            default_channel: DistributionChannel::File,
            schedule: DistributionSchedule::Scheduled(Duration::from_secs(3600)),
        }
    }

    /// Immediate distribution
    pub fn immediate() -> Self {
        Self {
            channels: vec![
                DistributionChannel::Log,
                DistributionChannel::Alert,
            ],
            default_channel: DistributionChannel::Alert,
            schedule: DistributionSchedule::Immediate,
        }
    }
}

/// Distribution channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionChannel {
    Log,
    File,
    Database,
    Email,
    SMS,
    Webhook,
    Alert,
    Dashboard,
}

/// Distribution schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionSchedule {
    OnGeneration,
    Immediate,
    Scheduled(Duration),
    OnDemand,
}

/// Analytics storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsStorage {
    /// Storage backend
    pub backend: StorageBackend,
    /// Storage configuration
    pub config: StorageConfig,
    /// Data retention policy
    pub retention: RetentionPolicy,
    /// Compression settings
    pub compression: StorageCompression,
}

impl Default for AnalyticsStorage {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Memory,
            config: StorageConfig::default(),
            retention: RetentionPolicy::default(),
            compression: StorageCompression::default(),
        }
    }
}

impl AnalyticsStorage {
    /// High-frequency storage
    pub fn high_frequency() -> Self {
        Self {
            backend: StorageBackend::Memory,
            config: StorageConfig::high_performance(),
            retention: RetentionPolicy::short_term(),
            compression: StorageCompression::minimal(),
        }
    }

    /// Detailed storage
    pub fn detailed() -> Self {
        Self {
            backend: StorageBackend::Database,
            config: StorageConfig::reliable(),
            retention: RetentionPolicy::long_term(),
            compression: StorageCompression::efficient(),
        }
    }

    /// Real-time storage
    pub fn real_time() -> Self {
        Self {
            backend: StorageBackend::Memory,
            config: StorageConfig::real_time(),
            retention: RetentionPolicy::minimal(),
            compression: StorageCompression::none(),
        }
    }
}

/// Storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Memory,
    File,
    Database,
    TimeSeries,
    Distributed,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Buffer size
    pub buffer_size: usize,
    /// Batch size for writes
    pub batch_size: usize,
    /// Write frequency
    pub write_frequency: Duration,
    /// Consistency level
    pub consistency: ConsistencyLevel,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            batch_size: 100,
            write_frequency: Duration::from_secs(60),
            consistency: ConsistencyLevel::Eventual,
        }
    }
}

impl StorageConfig {
    /// High-performance storage configuration
    pub fn high_performance() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 1000,
            write_frequency: Duration::from_secs(5),
            consistency: ConsistencyLevel::Eventual,
        }
    }

    /// Reliable storage configuration
    pub fn reliable() -> Self {
        Self {
            buffer_size: 5000,
            batch_size: 50,
            write_frequency: Duration::from_secs(30),
            consistency: ConsistencyLevel::Strong,
        }
    }

    /// Real-time storage configuration
    pub fn real_time() -> Self {
        Self {
            buffer_size: 100,
            batch_size: 1,
            write_frequency: Duration::from_secs(1),
            consistency: ConsistencyLevel::Eventual,
        }
    }
}

/// Consistency levels for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Eventual,
    Strong,
    Sequential,
    Causal,
}

/// Data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Retention duration
    pub duration: Duration,
    /// Archive policy
    pub archive: ArchivePolicy,
    /// Cleanup strategy
    pub cleanup: CleanupStrategy,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(86400), // 24 hours
            archive: ArchivePolicy::None,
            cleanup: CleanupStrategy::Automatic,
        }
    }
}

impl RetentionPolicy {
    /// Short-term retention
    pub fn short_term() -> Self {
        Self {
            duration: Duration::from_secs(3600), // 1 hour
            archive: ArchivePolicy::None,
            cleanup: CleanupStrategy::Aggressive,
        }
    }

    /// Long-term retention
    pub fn long_term() -> Self {
        Self {
            duration: Duration::from_secs(604800), // 7 days
            archive: ArchivePolicy::Compress,
            cleanup: CleanupStrategy::Conservative,
        }
    }

    /// Minimal retention
    pub fn minimal() -> Self {
        Self {
            duration: Duration::from_secs(300), // 5 minutes
            archive: ArchivePolicy::None,
            cleanup: CleanupStrategy::Immediate,
        }
    }
}

/// Archive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivePolicy {
    None,
    Compress,
    Move(String),
    Replicate,
}

/// Cleanup strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupStrategy {
    Automatic,
    Manual,
    Scheduled(Duration),
    Immediate,
    Conservative,
    Aggressive,
}

/// Storage compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCompression {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level
    pub level: u8,
    /// Compression threshold
    pub threshold: usize,
}

impl Default for StorageCompression {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "zstd".to_string(),
            level: 3,
            threshold: 1024,
        }
    }
}

impl StorageCompression {
    /// Minimal compression
    pub fn minimal() -> Self {
        Self {
            enabled: false,
            algorithm: "none".to_string(),
            level: 0,
            threshold: 0,
        }
    }

    /// Efficient compression
    pub fn efficient() -> Self {
        Self {
            enabled: true,
            algorithm: "zstd".to_string(),
            level: 9,
            threshold: 512,
        }
    }

    /// No compression
    pub fn none() -> Self {
        Self {
            enabled: false,
            algorithm: "none".to_string(),
            level: 0,
            threshold: usize::MAX,
        }
    }
}