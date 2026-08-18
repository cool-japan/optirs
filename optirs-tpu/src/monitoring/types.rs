//! Data-only monitoring types: settings/configuration structs, metric
//! collectors, alert/anomaly/analytics records, and their supporting
//! enums.
//!
//! Every field here is `pub` (a plain configuration/data bag); the
//! meaningful logic that reads and mutates [`TopologyPerformanceMonitor`]
//! lives in the sibling [`super::monitor_impl`] module.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use crate::tpu_backend::DeviceId;

/// Type alias for topology metrics collection
pub type TopologyMetrics = HashMap<String, f64>;

/// Main topology performance monitor with comprehensive monitoring capabilities
#[derive(Debug, Default)]
pub struct TopologyPerformanceMonitor {
    /// Performance monitoring configuration
    pub performance_monitoring: PerformanceMonitoringSettings,
    /// Health monitoring configuration
    pub health_monitoring: HealthMonitoringSettings,
    /// Traffic monitoring configuration
    pub traffic_monitoring: TrafficMonitoringSettings,
    /// Alert management system
    pub alert_system: AlertSystem,
    /// Metrics collection engine
    pub metrics_collector: MetricsCollector,
    /// Anomaly detection system
    pub anomaly_detector: AnomalyDetector,
    /// Performance analytics
    pub analytics: PerformanceAnalytics,
    /// Monotonic counter used to mint unique performance-report identifiers.
    ///
    /// Stored as an `AtomicU64` so `generate_report(&self)` can allocate a fresh
    /// id through a shared reference without requiring `&mut self`.
    pub report_counter: AtomicU64,
}

/// Comprehensive monitoring settings for topology
#[derive(Debug, Clone, Default)]
pub struct TopologyMonitoringSettings {
    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoringSettings,
    /// Health monitoring
    pub health_monitoring: HealthMonitoringSettings,
    /// Traffic monitoring
    pub traffic_monitoring: TrafficMonitoringSettings,
    /// Alert settings
    pub alert_settings: AlertSettings,
}

/// Performance monitoring configuration and settings
#[derive(Debug, Clone)]
pub struct PerformanceMonitoringSettings {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Metrics collection
    pub metrics_collection: MetricsCollectionSettings,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

/// Metrics collection configuration and settings
#[derive(Debug, Clone)]
pub struct MetricsCollectionSettings {
    /// Collected metrics
    pub collected_metrics: Vec<MetricType>,
    /// Collection granularity
    pub granularity: CollectionGranularity,
    /// Data retention
    pub retention_period: Duration,
}

/// Types of metrics to collect and monitor
#[derive(Debug, Clone)]
pub enum MetricType {
    /// Latency metrics
    Latency,
    /// Throughput metrics
    Throughput,
    /// Bandwidth utilization
    BandwidthUtilization,
    /// Packet loss rate
    PacketLoss,
    /// Queue occupancy
    QueueOccupancy,
    /// Custom metric
    Custom { metric_name: String },
}

/// Collection granularity levels for metrics
#[derive(Debug, Clone)]
pub enum CollectionGranularity {
    /// Per-device granularity
    PerDevice,
    /// Per-link granularity
    PerLink,
    /// Per-flow granularity
    PerFlow,
    /// Aggregate granularity
    Aggregate,
}

/// Performance thresholds for monitoring and alerting
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Latency thresholds
    pub latency_thresholds: ThresholdLevels,
    /// Throughput thresholds
    pub throughput_thresholds: ThresholdLevels,
    /// Utilization thresholds
    pub utilization_thresholds: ThresholdLevels,
    /// Error rate thresholds
    pub error_thresholds: ThresholdLevels,
}

/// Threshold levels for performance metrics
#[derive(Debug, Clone)]
pub struct ThresholdLevels {
    /// Warning threshold
    pub warning: f64,
    /// Critical threshold
    pub critical: f64,
    /// Emergency threshold
    pub emergency: f64,
}

/// Health monitoring configuration and settings
#[derive(Debug, Clone)]
pub struct HealthMonitoringSettings {
    /// Health check frequency
    pub check_frequency: Duration,
    /// Health indicators
    pub health_indicators: Vec<HealthIndicator>,
    /// Failure detection
    pub failure_detection: FailureDetectionSettings,
}

/// Health indicators to monitor for system health
#[derive(Debug, Clone)]
pub enum HealthIndicator {
    /// Link connectivity
    LinkConnectivity,
    /// Device responsiveness
    DeviceResponsiveness,
    /// Performance degradation
    PerformanceDegradation,
    /// Error rate increase
    ErrorRateIncrease,
    /// Custom health indicator
    Custom { indicator_name: String },
}

/// Failure detection configuration and settings
#[derive(Debug, Clone)]
pub struct FailureDetectionSettings {
    /// Detection algorithm
    pub algorithm: FailureDetectionAlgorithm,
    /// Detection sensitivity
    pub sensitivity: f64,
    /// False positive tolerance
    pub false_positive_tolerance: f64,
}

/// Failure detection algorithms for health monitoring
#[derive(Debug, Clone)]
pub enum FailureDetectionAlgorithm {
    /// Threshold-based detection
    ThresholdBased,
    /// Statistical anomaly detection
    StatisticalAnomaly,
    /// Machine learning based
    MachineLearning { model_path: String },
    /// Consensus-based detection
    ConsensusBased,
}

/// Traffic monitoring configuration and settings
#[derive(Debug, Clone, Default)]
pub struct TrafficMonitoringSettings {
    /// Flow monitoring
    pub flow_monitoring: FlowMonitoringSettings,
    /// Pattern analysis
    pub pattern_analysis: PatternAnalysisSettings,
    /// Anomaly detection
    pub anomaly_detection: AnomalyDetectionSettings,
}

/// Flow monitoring configuration and settings
#[derive(Debug, Clone)]
pub struct FlowMonitoringSettings {
    /// Flow tracking granularity
    pub tracking_granularity: FlowTrackingGranularity,
    /// Flow timeout
    pub flow_timeout: Duration,
    /// Sampling rate
    pub sampling_rate: f64,
}

/// Flow tracking granularity levels
#[derive(Debug, Clone)]
pub enum FlowTrackingGranularity {
    /// Per-packet tracking
    PerPacket,
    /// Per-flow tracking
    PerFlow,
    /// Aggregated tracking
    Aggregated,
    /// Sampled tracking
    Sampled { sampling_ratio: f64 },
}

/// Pattern analysis configuration and settings
#[derive(Debug, Clone)]
pub struct PatternAnalysisSettings {
    /// Analysis window size
    pub window_size: Duration,
    /// Pattern detection algorithms
    pub detection_algorithms: Vec<PatternDetectionAlgorithm>,
    /// Pattern classification
    pub classification: PatternClassification,
}

/// Pattern detection algorithms for traffic analysis
#[derive(Debug, Clone)]
pub enum PatternDetectionAlgorithm {
    /// Frequency analysis
    FrequencyAnalysis,
    /// Time series analysis
    TimeSeriesAnalysis,
    /// Spectral analysis
    SpectralAnalysis,
    /// Custom pattern detection
    Custom { algorithm_name: String },
}

/// Pattern classification configuration
#[derive(Debug, Clone)]
pub struct PatternClassification {
    /// Classification method
    pub method: ClassificationMethod,
    /// Pattern categories
    pub categories: Vec<String>,
    /// Classification confidence threshold
    pub confidence_threshold: f64,
}

/// Classification methods for pattern analysis
#[derive(Debug, Clone)]
pub enum ClassificationMethod {
    /// Rule-based classification
    RuleBased,
    /// Machine learning classification
    MachineLearning { model_path: String },
    /// Statistical classification
    Statistical,
    /// Hybrid classification
    Hybrid,
}

/// Anomaly detection configuration and settings
#[derive(Debug, Clone)]
pub struct AnomalyDetectionSettings {
    /// Detection method
    pub method: AnomalyDetectionMethod,
    /// Detection sensitivity
    pub sensitivity: f64,
    /// Baseline establishment
    pub baseline_establishment: BaselineEstablishment,
}

/// Anomaly detection methods for traffic monitoring
#[derive(Debug, Clone)]
pub enum AnomalyDetectionMethod {
    /// Statistical anomaly detection
    Statistical,
    /// Machine learning based
    MachineLearning { model_path: String },
    /// Clustering-based detection
    ClusteringBased,
    /// Time series anomaly detection
    TimeSeries,
}

/// Baseline establishment for anomaly detection
#[derive(Debug, Clone)]
pub struct BaselineEstablishment {
    /// Baseline learning period
    pub learning_period: Duration,
    /// Baseline update frequency
    pub update_frequency: Duration,
    /// Baseline adaptation rate
    pub adaptation_rate: f64,
}

/// Alert system configuration and settings
#[derive(Debug, Clone)]
pub struct AlertSettings {
    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Alert escalation
    pub escalation: AlertEscalation,
}

/// Alert channels for notifications and communication
#[derive(Debug, Clone)]
pub enum AlertChannel {
    /// Email alerts
    Email { recipients: Vec<String> },
    /// SMS alerts
    SMS { phone_numbers: Vec<String> },
    /// Slack alerts
    Slack { webhook_url: String },
    /// Custom alert channel
    Custom {
        channel_name: String,
        config: HashMap<String, String>,
    },
}

/// Alert thresholds for different alert types
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Performance alert thresholds
    pub performance: PerformanceThresholds,
    /// Health alert thresholds
    pub health: HealthThresholds,
    /// Anomaly alert thresholds
    pub anomaly: AnomalyThresholds,
}

/// Health alert thresholds
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Device failure threshold
    pub device_failure: f64,
    /// Link failure threshold
    pub link_failure: f64,
    /// Degradation threshold
    pub degradation: f64,
}

/// Anomaly alert thresholds
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    /// Anomaly score threshold
    pub score_threshold: f64,
    /// Anomaly frequency threshold
    pub frequency_threshold: f64,
    /// Anomaly severity threshold
    pub severity_threshold: f64,
}

/// Alert escalation configuration
#[derive(Debug, Clone)]
pub struct AlertEscalation {
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    /// Escalation timers
    pub timers: Vec<Duration>,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
}

/// Escalation levels for alert management
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level identifier
    pub level_id: String,
    /// Level priority
    pub priority: EscalationPriority,
    /// Notification targets
    pub targets: Vec<String>,
    /// Required acknowledgment
    pub require_ack: bool,
}

/// Escalation priorities for alert handling
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum EscalationPriority {
    /// Low priority escalation
    Low,
    /// Medium priority escalation
    Medium,
    /// High priority escalation
    High,
    /// Critical priority escalation
    Critical,
}

/// Escalation actions for alert handling
#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Send notification
    SendNotification { channel: AlertChannel },
    /// Execute script
    ExecuteScript { script_path: String },
    /// Trigger automation
    TriggerAutomation { automation_id: String },
    /// Custom escalation action
    Custom {
        action_name: String,
        parameters: HashMap<String, String>,
    },
}

/// Comprehensive metrics for communication patterns
#[derive(Debug, Clone)]
pub struct PatternMetrics {
    /// Performance metrics
    pub performance: PatternPerformanceMetrics,
    /// Resource utilization metrics
    pub utilization: PatternUtilizationMetrics,
    /// Quality metrics
    pub quality: PatternQualityMetrics,
    /// Efficiency metrics
    pub efficiency: PatternEfficiencyMetrics,
}

/// Performance metrics for communication patterns
#[derive(Debug, Clone)]
pub struct PatternPerformanceMetrics {
    /// Throughput (messages/second)
    pub throughput: f64,
    /// Latency (microseconds)
    pub latency: f64,
    /// Bandwidth utilization (Gbps)
    pub bandwidth_utilization: f64,
    /// Message success rate
    pub success_rate: f64,
}

/// Resource utilization metrics for patterns
#[derive(Debug, Clone)]
pub struct PatternUtilizationMetrics {
    /// Memory utilization
    pub memory_utilization: f64,
    /// Compute utilization
    pub compute_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Power utilization
    pub power_utilization: f64,
}

/// Quality metrics for communication patterns
#[derive(Debug, Clone)]
pub struct PatternQualityMetrics {
    /// Reliability score
    pub reliability: f64,
    /// Consistency score
    pub consistency: f64,
    /// Availability score
    pub availability: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Efficiency metrics for communication patterns
#[derive(Debug, Clone)]
pub struct PatternEfficiencyMetrics {
    /// Communication efficiency
    pub communication_efficiency: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
    /// Energy efficiency
    pub energy_efficiency: f64,
    /// Cost efficiency
    pub cost_efficiency: f64,
}

/// Quality metrics for layout solutions
#[derive(Debug, Clone)]
pub struct SolutionQualityMetrics {
    /// Total communication cost
    pub communication_cost: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// Load balance score
    pub load_balance: f64,
    /// Fault tolerance score
    pub fault_tolerance: f64,
}

/// Metrics for optimization iterations
#[derive(Debug, Clone)]
pub struct IterationMetrics {
    /// Time taken for iteration
    pub iteration_time: Duration,
    /// Memory usage during iteration
    pub memory_usage: u64,
    /// Number of evaluations performed
    pub evaluations: usize,
    /// Improvement over previous iteration
    pub improvement: f64,
}

/// Comprehensive metrics for layout optimizer
#[derive(Debug, Clone)]
pub struct LayoutOptimizerMetrics {
    /// Total optimization time
    pub total_time: Duration,
    /// Number of iterations performed
    pub iterations_performed: usize,
    /// Best objective value achieved
    pub best_objective: f64,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics,
    /// Resource utilization metrics
    pub resource_metrics: OptimizerResourceMetrics,
}

/// Convergence metrics for optimization algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Convergence rate
    pub convergence_rate: f64,
    /// Time to convergence
    pub time_to_convergence: Duration,
    /// Final improvement rate
    pub final_improvement_rate: f64,
    /// Objective value progression
    pub objective_progression: Vec<f64>,
}

/// Resource utilization metrics for optimizer performance
#[derive(Debug, Clone)]
pub struct OptimizerResourceMetrics {
    /// Peak memory usage
    pub peak_memory: u64,
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    /// Total energy consumption
    pub energy_consumption: f64,
    /// Resource efficiency score
    pub efficiency_score: f64,
}

/// Quality metrics for clustering algorithms
#[derive(Debug, Clone)]
pub struct ClusteringQualityMetrics {
    /// Silhouette score
    pub silhouette_score: f64,
    /// Davies-Bouldin index
    pub davies_bouldin_index: f64,
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f64,
    /// Inertia (within-cluster sum of squares)
    pub inertia: f64,
}

/// Performance statistics for layout systems
#[derive(Debug, Clone)]
pub struct LayoutPerformanceStatistics {
    /// Communication latency statistics
    pub latency_stats: LatencyStatistics,
    /// Bandwidth utilization statistics
    pub bandwidth_stats: BandwidthStatistics,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics,
    /// Resource utilization statistics
    pub resource_stats: ResourceUtilizationStatistics,
}

/// Latency statistics for performance monitoring
#[derive(Debug, Clone)]
pub struct LatencyStatistics {
    /// Mean latency
    pub mean_latency: f64,
    /// Median latency
    pub median_latency: f64,
    /// 95th percentile latency
    pub p95_latency: f64,
    /// 99th percentile latency
    pub p99_latency: f64,
    /// Maximum latency observed
    pub max_latency: f64,
    /// Latency standard deviation
    pub latency_std_dev: f64,
}

/// Bandwidth utilization statistics
#[derive(Debug, Clone)]
pub struct BandwidthStatistics {
    /// Average bandwidth utilization
    pub avg_utilization: f64,
    /// Peak bandwidth utilization
    pub peak_utilization: f64,
    /// Bandwidth efficiency score
    pub efficiency_score: f64,
    /// Utilization distribution
    pub utilization_distribution: Vec<f64>,
}

/// Throughput statistics for performance analysis
#[derive(Debug, Clone)]
pub struct ThroughputStatistics {
    /// Average throughput
    pub avg_throughput: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Throughput variance
    pub throughput_variance: f64,
    /// Sustained throughput duration
    pub sustained_duration: Duration,
}

/// Resource utilization statistics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationStatistics {
    /// Memory utilization statistics
    pub memory_utilization: UtilizationStats,
    /// CPU utilization statistics
    pub cpu_utilization: UtilizationStats,
    /// Network utilization statistics
    pub network_utilization: UtilizationStats,
    /// Storage utilization statistics
    pub storage_utilization: UtilizationStats,
}

/// General utilization statistics template
#[derive(Debug, Clone)]
pub struct UtilizationStats {
    /// Current utilization percentage
    pub current: f64,
    /// Average utilization percentage
    pub average: f64,
    /// Peak utilization percentage
    pub peak: f64,
    /// Utilization trend
    pub trend: UtilizationTrend,
}

/// Utilization trend indicators
#[derive(Debug, Clone)]
pub enum UtilizationTrend {
    /// Utilization is increasing
    Increasing,
    /// Utilization is decreasing
    Decreasing,
    /// Utilization is stable
    Stable,
    /// Utilization is fluctuating
    Fluctuating,
}

/// Alert system for managing notifications and responses
#[derive(Debug, Clone, Default)]
pub struct AlertSystem {
    /// Alert configuration
    pub config: AlertSettings,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Alert history
    pub alert_history: Vec<AlertRecord>,
    /// Alert processors
    pub processors: Vec<AlertProcessor>,
}

/// Individual alert instance
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert identifier
    pub alert_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert timestamp
    pub timestamp: Instant,
    /// Alert message
    pub message: String,
    /// Associated device ID
    pub device_id: Option<DeviceId>,
    /// Alert status
    pub status: AlertStatus,
}

/// Types of alerts that can be generated
#[derive(Debug, Clone)]
pub enum AlertType {
    /// Performance alert
    Performance,
    /// Health alert
    Health,
    /// Anomaly alert
    Anomaly,
    /// System alert
    System,
    /// Custom alert
    Custom { alert_name: String },
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Alert status tracking
#[derive(Debug, Clone, PartialEq)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert is acknowledged
    Acknowledged,
    /// Alert is resolved
    Resolved,
    /// Alert is escalated
    Escalated,
}

/// Alert record for historical tracking
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Alert instance
    pub alert: Alert,
    /// Actions taken
    pub actions_taken: Vec<String>,
    /// Resolution time
    pub resolution_time: Option<Duration>,
    /// Resolution method
    pub resolution_method: Option<String>,
}

/// Alert processor for handling specific alert types
#[derive(Debug, Clone)]
pub struct AlertProcessor {
    /// Processor identifier
    pub processor_id: String,
    /// Supported alert types
    pub supported_types: Vec<AlertType>,
    /// Processing configuration
    pub config: AlertProcessorConfig,
}

/// Alert processor configuration
#[derive(Debug, Clone)]
pub struct AlertProcessorConfig {
    /// Enable automatic processing
    pub auto_process: bool,
    /// Processing timeout
    pub timeout: Duration,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Retry configuration for alert processing
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Exponential backoff factor
    pub backoff_factor: f64,
}

/// Metrics collection engine
#[derive(Debug, Clone, Default)]
pub struct MetricsCollector {
    /// Collection configuration
    pub config: MetricsCollectionSettings,
    /// Collected metrics
    pub metrics: TopologyMetrics,
    /// Collection history
    pub history: Vec<MetricsSnapshot>,
    /// Active collectors
    pub collectors: Vec<MetricCollector>,
}

/// Individual metric collector
#[derive(Debug, Clone)]
pub struct MetricCollector {
    /// Collector identifier
    pub collector_id: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Collection interval
    pub interval: Duration,
    /// Last collection time
    pub last_collection: Instant,
}

/// Metrics snapshot for historical analysis
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    /// Metrics data
    pub metrics: TopologyMetrics,
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
}

/// Metadata for metrics snapshots
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    /// Collection source
    pub source: String,
    /// Collection method
    pub method: String,
    /// Data quality score
    pub quality_score: f64,
}

/// Anomaly detection system
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Detection configuration
    pub config: AnomalyDetectionSettings,
    /// Detection models
    pub models: Vec<AnomalyDetectionModel>,
    /// Detected anomalies
    pub anomalies: Vec<DetectedAnomaly>,
    /// Detection statistics
    pub statistics: AnomalyDetectionStatistics,
    /// Bounded rolling history of recent values per metric name.
    ///
    /// Used to build an online mean/standard-deviation baseline for z-score
    /// based anomaly detection. Each deque is capped at
    /// [`TopologyPerformanceMonitor::ANOMALY_HISTORY_CAPACITY`] samples.
    pub value_history: HashMap<String, VecDeque<f64>>,
}

/// Anomaly detection model
#[derive(Debug, Clone)]
pub struct AnomalyDetectionModel {
    /// Model identifier
    pub model_id: String,
    /// Model type
    pub model_type: AnomalyDetectionMethod,
    /// Model accuracy
    pub accuracy: f64,
    /// Last training time
    pub last_training: Instant,
    /// Model status
    pub status: ModelStatus,
}

/// Model status for anomaly detection
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// Model is training
    Training,
    /// Model is ready
    Ready,
    /// Model is updating
    Updating,
    /// Model has failed
    Failed,
}

/// Detected anomaly instance
#[derive(Debug, Clone)]
pub struct DetectedAnomaly {
    /// Anomaly identifier
    pub anomaly_id: String,
    /// Detection timestamp
    pub timestamp: Instant,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Anomaly score
    pub score: f64,
    /// Associated metrics
    pub metrics: HashMap<String, f64>,
    /// Anomaly status
    pub status: AnomalyStatus,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone)]
pub enum AnomalyType {
    /// Performance anomaly
    Performance,
    /// Traffic pattern anomaly
    TrafficPattern,
    /// Resource utilization anomaly
    ResourceUtilization,
    /// Communication anomaly
    Communication,
    /// Custom anomaly
    Custom { anomaly_name: String },
}

/// Status of detected anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyStatus {
    /// Anomaly is new
    New,
    /// Anomaly is under investigation
    Investigating,
    /// Anomaly is confirmed
    Confirmed,
    /// Anomaly is false positive
    FalsePositive,
    /// Anomaly is resolved
    Resolved,
}

/// Statistics for anomaly detection system
#[derive(Debug, Clone)]
pub struct AnomalyDetectionStatistics {
    /// Total anomalies detected
    pub total_detected: usize,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Detection accuracy
    pub detection_accuracy: f64,
    /// Average detection time
    pub avg_detection_time: Duration,
}

/// Performance analytics system
#[derive(Debug, Clone)]
pub struct PerformanceAnalytics {
    /// Analytics configuration
    pub config: AnalyticsConfig,
    /// Performance reports
    pub reports: Vec<PerformanceReport>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
    /// Predictive models
    pub predictive_models: Vec<PredictiveModel>,
}

/// Configuration for performance analytics
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Analysis window size
    pub analysis_window: Duration,
    /// Report generation frequency
    pub report_frequency: Duration,
    /// Enable predictive analytics
    pub enable_prediction: bool,
    /// Prediction horizon
    pub prediction_horizon: Duration,
}

/// Performance report generation
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Report identifier
    pub report_id: String,
    /// Report timestamp
    pub timestamp: Instant,
    /// Report period
    pub period: Duration,
    /// Performance summary
    pub summary: PerformanceSummary,
    /// Detailed metrics
    pub detailed_metrics: TopologyMetrics,
    /// Recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Performance summary for reports
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Overall performance score
    pub overall_score: f64,
    /// Key performance indicators
    pub kpis: HashMap<String, f64>,
    /// Performance trends
    pub trends: Vec<PerformanceTrend>,
    /// Critical issues
    pub critical_issues: Vec<String>,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric_name: String,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Degrading trend
    Degrading,
    /// Stable trend
    Stable,
    /// Volatile trend
    Volatile,
}

/// Performance recommendations
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Recommendation identifier
    pub recommendation_id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: f64,
}

/// Types of performance recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Configuration optimization
    ConfigurationOptimization,
    /// Resource allocation
    ResourceAllocation,
    /// Topology adjustment
    TopologyAdjustment,
    /// Performance tuning
    PerformanceTuning,
    /// Custom recommendation
    Custom { recommendation_name: String },
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Trend analysis system
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Analysis configuration
    pub config: TrendAnalysisConfig,
    /// Detected trends
    pub trends: Vec<DetectedTrend>,
    /// Trend predictions
    pub predictions: Vec<TrendPrediction>,
}

/// Configuration for trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysisConfig {
    /// Analysis window size
    pub window_size: Duration,
    /// Minimum trend duration
    pub min_trend_duration: Duration,
    /// Trend detection sensitivity
    pub sensitivity: f64,
}

/// Detected trend information
#[derive(Debug, Clone)]
pub struct DetectedTrend {
    /// Trend identifier
    pub trend_id: String,
    /// Metric name
    pub metric_name: String,
    /// Trend type
    pub trend_type: TrendType,
    /// Start time
    pub start_time: Instant,
    /// Duration
    pub duration: Duration,
    /// Trend strength
    pub strength: f64,
}

/// Types of trends that can be detected
#[derive(Debug, Clone)]
pub enum TrendType {
    /// Linear trend
    Linear { slope: f64 },
    /// Exponential trend
    Exponential { rate: f64 },
    /// Periodic trend
    Periodic { period: Duration, amplitude: f64 },
    /// Step change
    StepChange { change_magnitude: f64 },
}

/// Trend prediction information
#[derive(Debug, Clone)]
pub struct TrendPrediction {
    /// Prediction identifier
    pub prediction_id: String,
    /// Predicted metric
    pub metric_name: String,
    /// Prediction horizon
    pub horizon: Duration,
    /// Predicted values
    pub predicted_values: Vec<f64>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Predictive model for performance forecasting
#[derive(Debug, Clone)]
pub struct PredictiveModel {
    /// Model identifier
    pub model_id: String,
    /// Model type
    pub model_type: PredictiveModelType,
    /// Model accuracy
    pub accuracy: f64,
    /// Training data size
    pub training_data_size: usize,
    /// Last training time
    pub last_training: Instant,
}

/// Types of predictive models
#[derive(Debug, Clone)]
pub enum PredictiveModelType {
    /// Time series forecasting
    TimeSeriesForecasting,
    /// Regression model
    Regression,
    /// Neural network
    NeuralNetwork,
    /// Ensemble model
    Ensemble,
    /// Custom model
    Custom { model_name: String },
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Health indicator checked
    pub indicator: HealthIndicator,
    /// Health status result
    pub status: HealthStatus,
    /// Check timestamp
    pub timestamp: Instant,
    /// Additional details
    pub details: String,
}

/// Health status indicators
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings
    Warning,
    /// System has errors
    Error,
    /// System is critical
    Critical,
    /// Status is unknown
    Unknown,
}
