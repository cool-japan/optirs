// Filter Performance Monitoring and Metrics
//
// This module provides comprehensive performance monitoring, metrics collection,
// and analysis capabilities for event filtering systems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Filter performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterPerformanceMonitoring {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring level
    pub level: MonitoringLevel,
    /// Metrics collection
    pub metrics_collection: MetricsCollection,
    /// Performance analysis
    pub analysis: PerformanceAnalysis,
    /// Alerting configuration
    pub alerting: AlertingConfiguration,
    /// Reporting configuration
    pub reporting: ReportingConfiguration,
}

impl Default for FilterPerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            level: MonitoringLevel::Standard,
            metrics_collection: MetricsCollection::default(),
            analysis: PerformanceAnalysis::default(),
            alerting: AlertingConfiguration::default(),
            reporting: ReportingConfiguration::default(),
        }
    }
}

/// Monitoring levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringLevel {
    /// Minimal monitoring (basic metrics only)
    Minimal,
    /// Standard monitoring (common metrics)
    Standard,
    /// Detailed monitoring (comprehensive metrics)
    Detailed,
    /// Debug monitoring (all metrics with tracing)
    Debug,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollection {
    /// Collection interval
    pub interval: Duration,
    /// Metrics to collect
    pub metrics: Vec<MetricType>,
    /// Sampling configuration
    pub sampling: SamplingConfiguration,
    /// Aggregation settings
    pub aggregation: AggregationSettings,
    /// Storage settings
    pub storage: MetricsStorage,
}

impl Default for MetricsCollection {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            metrics: vec![
                MetricType::ExecutionTime,
                MetricType::Throughput,
                MetricType::ErrorRate,
                MetricType::MemoryUsage,
                MetricType::CacheHitRate,
            ],
            sampling: SamplingConfiguration::default(),
            aggregation: AggregationSettings::default(),
            storage: MetricsStorage::default(),
        }
    }
}

/// Types of metrics to collect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Rule execution time
    ExecutionTime,
    /// Filter throughput (events/sec)
    Throughput,
    /// Error rate
    ErrorRate,
    /// Memory usage
    MemoryUsage,
    /// CPU usage
    CpuUsage,
    /// Cache hit rate
    CacheHitRate,
    /// Rule match rate
    MatchRate,
    /// Queue depth
    QueueDepth,
    /// Latency percentiles
    LatencyPercentiles,
    /// Resource utilization
    ResourceUtilization,
    /// Custom metric
    Custom(String),
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfiguration {
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Sampling rate (0.0 to 1.0)
    pub rate: f64,
    /// Sample size
    pub sample_size: usize,
    /// Adaptive sampling
    pub adaptive: bool,
}

impl Default for SamplingConfiguration {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Random,
            rate: 0.1, // 10% sampling
            sample_size: 1000,
            adaptive: true,
        }
    }
}

/// Sampling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Systematic sampling
    Systematic,
    /// Stratified sampling
    Stratified,
    /// Reservoir sampling
    Reservoir,
    /// Time-based sampling
    TimeBased(Duration),
}

/// Aggregation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSettings {
    /// Aggregation window
    pub window: Duration,
    /// Aggregation functions
    pub functions: Vec<AggregationFunction>,
    /// Time granularity
    pub granularity: TimeGranularity,
    /// Rolling aggregation
    pub rolling: bool,
}

impl Default for AggregationSettings {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(60), // 1 minute
            functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Min,
                AggregationFunction::Max,
                AggregationFunction::Count,
            ],
            granularity: TimeGranularity::Seconds(10),
            rolling: true,
        }
    }
}

/// Aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    /// Sum of values
    Sum,
    /// Mean/average
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of values
    Count,
    /// Standard deviation
    StdDev,
    /// Percentile
    Percentile(f64),
    /// Median
    Median,
    /// Mode
    Mode,
    /// Custom function
    Custom(String),
}

/// Time granularity for aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeGranularity {
    /// Milliseconds
    Milliseconds(u64),
    /// Seconds
    Seconds(u64),
    /// Minutes
    Minutes(u64),
    /// Hours
    Hours(u64),
    /// Days
    Days(u64),
}

/// Metrics storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStorage {
    /// Storage backend
    pub backend: MetricsStorageBackend,
    /// Retention policy
    pub retention: RetentionPolicy,
    /// Compression settings
    pub compression: CompressionSettings,
    /// Partitioning strategy
    pub partitioning: PartitioningStrategy,
}

impl Default for MetricsStorage {
    fn default() -> Self {
        Self {
            backend: MetricsStorageBackend::Memory { capacity: 100000 },
            retention: RetentionPolicy::default(),
            compression: CompressionSettings::default(),
            partitioning: PartitioningStrategy::TimeRange,
        }
    }
}

/// Metrics storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsStorageBackend {
    /// In-memory storage
    Memory { capacity: usize },
    /// File-based storage
    File {
        directory: String,
        format: StorageFormat,
    },
    /// Time-series database
    TimeSeries {
        connection_string: String,
        database: String,
    },
    /// Cloud metrics service
    Cloud {
        provider: CloudMetricsProvider,
        config: HashMap<String, String>,
    },
}

/// Storage formats for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    Json,
    Binary,
    Parquet,
    Csv,
}

/// Cloud metrics providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudMetricsProvider {
    AWS_CloudWatch,
    GCP_Monitoring,
    Azure_Monitor,
    Prometheus,
    InfluxDB,
    Custom(String),
}

/// Retention policy for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Raw data retention
    pub raw_data: Duration,
    /// Aggregated data retention
    pub aggregated_data: Duration,
    /// Archive after
    pub archive_after: Duration,
    /// Delete after
    pub delete_after: Duration,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            raw_data: Duration::from_secs(86400),             // 1 day
            aggregated_data: Duration::from_secs(86400 * 30), // 30 days
            archive_after: Duration::from_secs(86400 * 90),   // 90 days
            delete_after: Duration::from_secs(86400 * 365),   // 1 year
        }
    }
}

/// Compression settings for metrics storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: CompressionLevel,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: CompressionLevel::Default,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}

/// Compression levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionLevel {
    Fastest,
    Default,
    Best,
    Custom(i32),
}

/// Partitioning strategies for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// Time-based partitioning
    TimeRange,
    /// Metric type partitioning
    MetricType,
    /// Hash-based partitioning
    Hash,
    /// No partitioning
    None,
}

/// Performance analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Analysis algorithms
    pub algorithms: Vec<AnalysisAlgorithm>,
    /// Baseline configuration
    pub baseline: BaselineConfiguration,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetection,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
    /// Performance profiling
    pub profiling: PerformanceProfiling,
}

impl Default for PerformanceAnalysis {
    fn default() -> Self {
        Self {
            algorithms: vec![
                AnalysisAlgorithm::StatisticalAnalysis,
                AnalysisAlgorithm::TrendAnalysis,
                AnalysisAlgorithm::AnomalyDetection,
            ],
            baseline: BaselineConfiguration::default(),
            anomaly_detection: AnomalyDetection::default(),
            trend_analysis: TrendAnalysis::default(),
            profiling: PerformanceProfiling::default(),
        }
    }
}

/// Analysis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisAlgorithm {
    /// Statistical analysis
    StatisticalAnalysis,
    /// Trend analysis
    TrendAnalysis,
    /// Anomaly detection
    AnomalyDetection,
    /// Regression analysis
    RegressionAnalysis,
    /// Machine learning analysis
    MachineLearningAnalysis(String),
    /// Custom analysis
    Custom(String),
}

/// Baseline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfiguration {
    /// Enable baseline tracking
    pub enabled: bool,
    /// Baseline window
    pub window: Duration,
    /// Baseline update frequency
    pub update_frequency: Duration,
    /// Baseline calculation method
    pub calculation_method: BaselineCalculation,
}

impl Default for BaselineConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(86400 * 7), // 7 days
            update_frequency: Duration::from_secs(86400), // 1 day
            calculation_method: BaselineCalculation::RollingAverage,
        }
    }
}

/// Baseline calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineCalculation {
    /// Rolling average
    RollingAverage,
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Percentile-based
    Percentile(f64),
    /// Seasonal baseline
    Seasonal,
    /// Custom calculation
    Custom(String),
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithms
    pub algorithms: Vec<AnomalyAlgorithm>,
    /// Sensitivity level
    pub sensitivity: f64,
    /// Detection window
    pub window: Duration,
    /// False positive reduction
    pub false_positive_reduction: FalsePositiveReduction,
}

impl Default for AnomalyDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnomalyAlgorithm::StatisticalOutlier,
                AnomalyAlgorithm::ChangePointDetection,
            ],
            sensitivity: 0.8,
            window: Duration::from_secs(300), // 5 minutes
            false_positive_reduction: FalsePositiveReduction::default(),
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Change point detection
    ChangePointDetection,
    /// Isolation forest
    IsolationForest,
    /// One-class SVM
    OneClassSVM,
    /// DBSCAN clustering
    DBSCAN,
    /// Custom algorithm
    Custom(String),
}

/// False positive reduction techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsePositiveReduction {
    /// Enable noise filtering
    pub noise_filtering: bool,
    /// Confirmation window
    pub confirmation_window: Duration,
    /// Minimum anomaly score
    pub min_anomaly_score: f64,
    /// Context awareness
    pub context_awareness: bool,
}

impl Default for FalsePositiveReduction {
    fn default() -> Self {
        Self {
            noise_filtering: true,
            confirmation_window: Duration::from_secs(60),
            min_anomaly_score: 0.7,
            context_awareness: true,
        }
    }
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Enable trend analysis
    pub enabled: bool,
    /// Analysis window
    pub window: Duration,
    /// Trend algorithms
    pub algorithms: Vec<TrendAlgorithm>,
    /// Seasonality detection
    pub seasonality: SeasonalityDetection,
    /// Forecasting
    pub forecasting: Forecasting,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(86400 * 7), // 7 days
            algorithms: vec![
                TrendAlgorithm::LinearRegression,
                TrendAlgorithm::MovingAverage,
            ],
            seasonality: SeasonalityDetection::default(),
            forecasting: Forecasting::default(),
        }
    }
}

/// Trend analysis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendAlgorithm {
    /// Linear regression
    LinearRegression,
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// Custom algorithm
    Custom(String),
}

/// Seasonality detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityDetection {
    /// Enable seasonality detection
    pub enabled: bool,
    /// Detection algorithms
    pub algorithms: Vec<SeasonalityAlgorithm>,
    /// Seasonal periods to check
    pub periods: Vec<Duration>,
}

impl Default for SeasonalityDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                SeasonalityAlgorithm::Autocorrelation,
                SeasonalityAlgorithm::Fourier,
            ],
            periods: vec![
                Duration::from_secs(3600),      // Hourly
                Duration::from_secs(86400),     // Daily
                Duration::from_secs(86400 * 7), // Weekly
            ],
        }
    }
}

/// Seasonality detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalityAlgorithm {
    /// Autocorrelation analysis
    Autocorrelation,
    /// Fourier transform
    Fourier,
    /// Seasonal decomposition
    SeasonalDecomposition,
    /// Custom algorithm
    Custom(String),
}

/// Forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecasting {
    /// Enable forecasting
    pub enabled: bool,
    /// Forecast horizon
    pub horizon: Duration,
    /// Forecasting models
    pub models: Vec<ForecastingModel>,
    /// Model selection strategy
    pub model_selection: ModelSelection,
}

impl Default for Forecasting {
    fn default() -> Self {
        Self {
            enabled: false,
            horizon: Duration::from_secs(3600), // 1 hour
            models: vec![
                ForecastingModel::LinearRegression,
                ForecastingModel::ExponentialSmoothing,
            ],
            model_selection: ModelSelection::BestFit,
        }
    }
}

/// Forecasting models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastingModel {
    /// Linear regression
    LinearRegression,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// Neural network
    NeuralNetwork,
    /// Custom model
    Custom(String),
}

/// Model selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelection {
    /// Best fit based on historical accuracy
    BestFit,
    /// Ensemble of models
    Ensemble,
    /// Model rotation
    Rotation,
    /// Custom selection
    Custom(String),
}

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfiling {
    /// Enable profiling
    pub enabled: bool,
    /// Profiling mode
    pub mode: ProfilingMode,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Profiling targets
    pub targets: Vec<ProfilingTarget>,
    /// Call stack depth
    pub stack_depth: usize,
}

impl Default for PerformanceProfiling {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ProfilingMode::Sampling,
            sampling_rate: 0.01, // 1%
            targets: vec![
                ProfilingTarget::CPU,
                ProfilingTarget::Memory,
                ProfilingTarget::IO,
            ],
            stack_depth: 10,
        }
    }
}

/// Profiling modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilingMode {
    /// Sampling profiler
    Sampling,
    /// Instrumentation profiler
    Instrumentation,
    /// Hybrid profiling
    Hybrid,
}

/// Profiling targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilingTarget {
    /// CPU profiling
    CPU,
    /// Memory profiling
    Memory,
    /// IO profiling
    IO,
    /// Network profiling
    Network,
    /// Custom profiling
    Custom(String),
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfiguration {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
    /// Alert aggregation
    pub aggregation: AlertAggregation,
    /// Escalation policies
    pub escalation: EscalationPolicies,
}

impl Default for AlertingConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: Vec::new(),
            channels: Vec::new(),
            aggregation: AlertAggregation::default(),
            escalation: EscalationPolicies::default(),
        }
    }
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert frequency limit
    pub frequency_limit: Option<Duration>,
    /// Auto-resolve
    pub auto_resolve: bool,
    /// Notification channels
    pub channels: Vec<String>,
}

/// Alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold condition
    Threshold {
        metric: MetricType,
        operator: ComparisonOperator,
        value: f64,
        window: Duration,
    },
    /// Anomaly condition
    Anomaly {
        metric: MetricType,
        sensitivity: f64,
    },
    /// Trend condition
    Trend {
        metric: MetricType,
        direction: TrendDirection,
        threshold: f64,
    },
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<AlertCondition>,
    },
}

/// Comparison operators for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email {
        addresses: Vec<String>,
        template: Option<String>,
    },
    /// Slack notification
    Slack { webhook: String, channel: String },
    /// SMS notification
    SMS { numbers: Vec<String> },
    /// Webhook notification
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// Custom notification
    Custom {
        handler: String,
        config: HashMap<String, String>,
    },
}

/// Alert aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAggregation {
    /// Enable aggregation
    pub enabled: bool,
    /// Aggregation window
    pub window: Duration,
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Group by fields
    pub group_by: Vec<String>,
}

impl Default for AlertAggregation {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(300), // 5 minutes
            strategy: AggregationStrategy::Count,
            group_by: vec!["rule_name".to_string(), "severity".to_string()],
        }
    }
}

/// Aggregation strategies for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Count similar alerts
    Count,
    /// Suppress duplicates
    Suppress,
    /// Escalate on count
    EscalateOnCount(usize),
    /// Custom strategy
    Custom(String),
}

/// Escalation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicies {
    /// Escalation rules
    pub rules: Vec<EscalationRule>,
    /// Default escalation
    pub default_escalation: Option<EscalationRule>,
}

impl Default for EscalationPolicies {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default_escalation: None,
        }
    }
}

/// Escalation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    /// Rule name
    pub name: String,
    /// Trigger condition
    pub trigger: EscalationTrigger,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
    /// Escalation delay
    pub delay: Duration,
}

/// Escalation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationTrigger {
    /// Time-based escalation
    Time(Duration),
    /// Count-based escalation
    Count(usize),
    /// Severity-based escalation
    Severity(AlertSeverity),
    /// Custom trigger
    Custom(String),
}

/// Escalation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationAction {
    /// Notify additional channels
    NotifyChannels(Vec<String>),
    /// Change alert severity
    ChangeSeverity(AlertSeverity),
    /// Execute custom action
    CustomAction(String),
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfiguration {
    /// Enable reporting
    pub enabled: bool,
    /// Report types
    pub report_types: Vec<ReportType>,
    /// Report schedule
    pub schedule: ReportSchedule,
    /// Report format
    pub format: ReportFormat,
    /// Report distribution
    pub distribution: ReportDistribution,
}

impl Default for ReportingConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            report_types: vec![ReportType::Performance, ReportType::Summary],
            schedule: ReportSchedule::Daily,
            format: ReportFormat::Html,
            distribution: ReportDistribution::default(),
        }
    }
}

/// Types of reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    /// Performance report
    Performance,
    /// Summary report
    Summary,
    /// Detailed analysis report
    DetailedAnalysis,
    /// Trend report
    Trend,
    /// Anomaly report
    Anomaly,
    /// Custom report
    Custom(String),
}

/// Report scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportSchedule {
    /// Hourly reports
    Hourly,
    /// Daily reports
    Daily,
    /// Weekly reports
    Weekly,
    /// Monthly reports
    Monthly,
    /// Custom schedule
    Custom(Duration),
    /// On-demand only
    OnDemand,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// HTML format
    Html,
    /// PDF format
    Pdf,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Custom format
    Custom(String),
}

/// Report distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistribution {
    /// Distribution channels
    pub channels: Vec<DistributionChannel>,
    /// Recipients
    pub recipients: Vec<String>,
    /// Archive location
    pub archive_location: Option<String>,
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            recipients: Vec::new(),
            archive_location: None,
        }
    }
}

/// Distribution channels for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionChannel {
    /// Email distribution
    Email,
    /// File system
    FileSystem(String),
    /// Cloud storage
    CloudStorage {
        provider: CloudMetricsProvider,
        bucket: String,
    },
    /// Custom distribution
    Custom(String),
}
