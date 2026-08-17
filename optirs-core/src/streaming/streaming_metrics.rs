// Comprehensive metrics and monitoring for streaming optimization
//
// This module provides detailed performance metrics, monitoring capabilities,
// and analytics for streaming optimization systems.
//
// Honesty contract for this module (findings M1-M4): every field below is
// either derived from data the caller actually supplied, or it is an
// `Option` that stays `None` until a caller feeds the missing measurement
// through one of the `record_*`/`set_*` hooks. No field is ever populated
// with an invented constant.

use scirs2_core::numeric::Float;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::Result;

mod accumulator;
mod aggregation;
mod alerts;
mod export;

#[cfg(test)]
mod regression_tests;

pub use accumulator::{MetricsAccumulator, ResourceProbe, RobustnessProbe};
pub use aggregation::AggregatedSeries;
pub use alerts::KNOWN_METRIC_PATHS;

/// Seconds since the Unix epoch, saturating instead of panicking for times
/// before the epoch (M4: this used to be `duration_since(..).expect(..)`).
pub(crate) fn unix_timestamp(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Microseconds per second, for converting between the second-resolution
/// `MetricsSnapshot::timestamp` and the microsecond-resolution key
/// `HistoricalMetrics::time_series` is indexed by.
pub(crate) const MICROS_PER_SEC: u64 = 1_000_000;

/// Microseconds since the Unix epoch, saturating at zero for pre-epoch times.
///
/// This is the resolution `HistoricalMetrics` keys its time series by. Keying by
/// whole seconds (as it used to) silently dropped every sample but the last
/// within each second, which is most of them under sub-second streaming rates —
/// exactly the regime the retention and compression logic is there to bound.
/// `u64` microseconds spans ~584 000 years, so it cannot overflow in practice.
pub(crate) fn unix_timestamp_micros(time: SystemTime) -> u64 {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    elapsed
        .as_secs()
        .saturating_mul(MICROS_PER_SEC)
        .saturating_add(u64::from(elapsed.subsec_micros()))
}

/// Elapsed time between two instants, saturating at zero when `later`
/// precedes `earlier` (clocks can and do step backwards).
pub(crate) fn saturating_elapsed(later: SystemTime, earlier: SystemTime) -> Duration {
    later.duration_since(earlier).unwrap_or_default()
}

/// Streaming metrics collector and analyzer
#[derive(Debug)]
pub struct StreamingMetricsCollector<A: Float + Send + Sync> {
    /// Performance metrics
    performance_metrics: PerformanceMetrics<A>,

    /// Resource utilization metrics
    resource_metrics: ResourceMetrics,

    /// Quality metrics
    quality_metrics: QualityMetrics<A>,

    /// Business metrics
    business_metrics: BusinessMetrics<A>,

    /// Historical data storage
    historical_data: HistoricalMetrics<A>,

    /// Real-time dashboards
    dashboards: Vec<Dashboard>,

    /// Alert system
    alert_system: AlertSystem<A>,

    /// Metric aggregation settings
    aggregation_config: AggregationConfig,

    /// Export configuration
    export_config: ExportConfig,

    /// Rolling raw observations the aggregate metrics are derived from
    accumulator: MetricsAccumulator<A>,

    /// Service level objectives, when the operator configured any
    slo: Option<SloTargets>,

    /// Cost model, when the operator configured one
    cost_model: Option<CostModel<A>>,

    /// When the last export ran, used to honour `ExportConfig::frequency`
    last_export: Option<SystemTime>,
}

/// Service level objectives used to compute a real SLO compliance ratio.
#[derive(Debug, Clone, Default)]
pub struct SloTargets {
    /// Maximum acceptable end-to-end processing time per sample
    pub max_processing_time: Option<Duration>,

    /// Maximum acceptable loss value
    pub max_loss: Option<f64>,

    /// Maximum acceptable memory usage in bytes
    pub max_memory_bytes: Option<u64>,
}

impl SloTargets {
    /// Whether this target set constrains anything at all.
    pub fn is_empty(&self) -> bool {
        self.max_processing_time.is_none()
            && self.max_loss.is_none()
            && self.max_memory_bytes.is_none()
    }
}

/// Operator-supplied cost model. Without one, cost metrics stay `None`
/// rather than being invented.
#[derive(Debug, Clone)]
pub struct CostModel<A: Float + Send + Sync> {
    /// Currency cost of one second of compute
    pub compute_cost_per_second: A,

    /// Currency cost of holding one gigabyte for one hour
    pub memory_cost_per_gb_hour: A,

    /// Currency cost of one joule of energy
    pub energy_cost_per_joule: A,

    /// Currency value of reducing the loss by one unit
    pub value_per_loss_unit: A,
}

/// Performance-related metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics<A: Float + Send + Sync> {
    /// Throughput measurements
    pub throughput: ThroughputMetrics,

    /// Latency measurements
    pub latency: LatencyMetrics,

    /// Accuracy and convergence metrics
    pub accuracy: AccuracyMetrics<A>,

    /// Stability metrics
    pub stability: StabilityMetrics<A>,

    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics<A>,
}

/// Throughput measurements
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Samples processed per second
    pub samples_per_second: f64,

    /// Updates per second
    pub updates_per_second: f64,

    /// Gradient computations per second
    pub gradients_per_second: f64,

    /// Peak throughput achieved
    pub peak_throughput: f64,

    /// Minimum throughput observed
    pub min_throughput: f64,

    /// Throughput variance
    pub throughput_variance: f64,

    /// Throughput trend (positive = increasing)
    pub throughput_trend: f64,
}

/// Latency measurements
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// End-to-end latency statistics
    pub end_to_end: LatencyStats,

    /// Gradient computation latency, when the caller reports it
    pub gradient_computation: Option<LatencyStats>,

    /// Update application latency, when the caller reports it
    pub update_application: Option<LatencyStats>,

    /// Communication latency (for distributed), when the caller reports it
    pub communication: Option<LatencyStats>,

    /// Queue waiting time, when the caller reports it
    pub queue_wait_time: Option<LatencyStats>,

    /// Processing jitter (mean absolute successive difference)
    pub jitter: f64,
}

/// Detailed latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Mean latency
    pub mean: Duration,

    /// Median latency
    pub median: Duration,

    /// 95th percentile
    pub p95: Duration,

    /// 99th percentile
    pub p99: Duration,

    /// 99.9th percentile
    pub p999: Duration,

    /// Maximum latency observed
    pub max: Duration,

    /// Minimum latency observed
    pub min: Duration,

    /// Standard deviation
    pub std_dev: Duration,
}

/// Accuracy and convergence metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics<A: Float + Send + Sync> {
    /// Current loss value
    pub current_loss: A,

    /// Loss reduction per second over the retained window
    pub loss_reduction_rate: A,

    /// Convergence rate (negative ordinary-least-squares slope of the loss)
    pub convergence_rate: A,

    /// Prediction accuracy, when the caller reports an `accuracy` custom metric
    pub prediction_accuracy: Option<A>,

    /// Gradient magnitude
    pub gradient_magnitude: A,

    /// Parameter stability derived from the gradient-magnitude spread
    pub parameter_stability: A,

    /// Learning progress relative to the first observed loss
    pub learning_progress: A,
}

/// Model stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics<A: Float + Send + Sync> {
    /// Loss variance
    pub loss_variance: A,

    /// Gradient variance
    pub gradient_variance: A,

    /// Mean gradient magnitude, i.e. how far the parameters move per step
    pub parameter_drift: A,

    /// Fraction of successive loss differences that changed sign
    pub oscillation_score: A,

    /// Empirical fraction of steps where the loss increased
    pub divergence_probability: A,

    /// Confidence in the stability estimate (falls with relative loss spread)
    pub stability_confidence: A,
}

/// Efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics<A: Float + Send + Sync> {
    /// Loss reduction per second of processing time
    pub computational_efficiency: Option<A>,

    /// Mean-to-peak memory usage ratio
    pub memory_efficiency: Option<A>,

    /// Share of processing time not spent communicating; requires the caller
    /// to report communication times
    pub communication_efficiency: Option<A>,

    /// Energy efficiency; requires an external energy meter
    pub energy_efficiency: Option<A>,

    /// Fraction of wall-clock time spent processing
    pub resource_utilization: A,

    /// Business value per unit cost; requires a configured cost model
    pub cost_efficiency: Option<A>,
}

/// Resource utilization metrics.
///
/// Every field except `memory_usage` needs an OS-level probe this collector
/// does not perform itself; feed them with
/// [`StreamingMetricsCollector::record_resource_probe`].
#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: Option<f64>,

    /// Memory usage
    pub memory_usage: MemoryUsage,

    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,

    /// Network bandwidth usage in MB/s
    pub network_bandwidth: Option<f64>,

    /// Disk I/O usage in MB/s
    pub disk_io: Option<f64>,

    /// Thread utilization
    pub thread_utilization: Option<f64>,
}

/// Memory usage breakdown
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Total allocated memory (bytes); needs an allocator probe
    pub total_allocated: Option<u64>,

    /// Currently used memory (bytes)
    pub current_used: u64,

    /// Peak memory usage (bytes)
    pub peak_usage: u64,

    /// Memory fragmentation ratio; needs an allocator probe
    pub fragmentation_ratio: Option<f64>,

    /// Garbage collection overhead. Rust has no garbage collector, so this is
    /// permanently `None` for in-process measurements.
    pub gc_overhead: Option<f64>,

    /// Mean-to-peak usage ratio
    pub efficiency: Option<f64>,
}

/// Quality metrics for streaming optimization
#[derive(Debug, Clone)]
pub struct QualityMetrics<A: Float + Send + Sync> {
    /// Fraction of recent samples carrying finite, non-negative measurements
    pub data_quality: A,

    /// Model quality metrics
    pub model_quality: ModelQuality<A>,

    /// Concept drift metrics
    pub concept_drift: ConceptDriftMetrics<A>,

    /// Anomaly detection metrics
    pub anomaly_detection: AnomalyMetrics<A>,

    /// Robustness metrics
    pub robustness: RobustnessMetrics<A>,
}

/// Model quality assessment
#[derive(Debug, Clone)]
pub struct ModelQuality<A: Float + Send + Sync> {
    /// Relative loss improvement since the first observed sample
    pub training_quality: A,

    /// Generalization ability; requires a `val_loss` custom metric
    pub generalization_score: Option<A>,

    /// Overfitting detection; requires a `val_loss` custom metric
    pub overfitting_score: Option<A>,

    /// Underfitting detection; requires a `val_loss` custom metric
    pub underfitting_score: Option<A>,

    /// Model complexity; requires a `parameter_count` custom metric
    pub complexity_score: Option<A>,
}

/// Concept drift monitoring metrics
#[derive(Debug, Clone)]
pub struct ConceptDriftMetrics<A: Float + Send + Sync> {
    /// Confidence of the most recent reported drift
    pub drift_confidence: Option<A>,

    /// Magnitude of the most recent reported drift
    pub drift_magnitude: Option<A>,

    /// Reported drift events per second over the observed window
    pub drift_frequency: f64,

    /// Adaptation effectiveness reported alongside a drift event
    pub adaptation_effectiveness: Option<A>,

    /// Detection latency of the most recent reported drift
    pub detection_latency: Option<Duration>,
}

/// Anomaly detection metrics
#[derive(Debug, Clone)]
pub struct AnomalyMetrics<A: Float + Send + Sync> {
    /// Robust z-score of the most recent loss against the running statistics
    pub anomaly_score: A,

    /// False positive rate; requires labelled ground truth
    pub false_positive_rate: Option<A>,

    /// False negative rate; requires labelled ground truth
    pub false_negative_rate: Option<A>,

    /// Detection accuracy; requires labelled ground truth
    pub detection_accuracy: Option<A>,

    /// Fraction of the retained window flagged as anomalous
    pub anomaly_frequency: f64,
}

/// Model robustness metrics. These require deliberate perturbation
/// experiments; feed them with
/// [`StreamingMetricsCollector::record_robustness_probe`].
#[derive(Debug, Clone, Default)]
pub struct RobustnessMetrics<A: Float + Send + Sync> {
    /// Noise tolerance
    pub noise_tolerance: Option<A>,

    /// Adversarial robustness
    pub adversarial_robustness: Option<A>,

    /// Input perturbation sensitivity
    pub perturbation_sensitivity: Option<A>,

    /// Recovery capability
    pub recovery_capability: Option<A>,

    /// Fault tolerance
    pub fault_tolerance: Option<A>,
}

/// Business and operational metrics
#[derive(Debug, Clone)]
pub struct BusinessMetrics<A: Float + Send + Sync> {
    /// System availability; requires reported outages
    pub availability: Option<f64>,

    /// Service level objective compliance; requires configured SLO targets
    pub slo_compliance: Option<f64>,

    /// Cost metrics
    pub cost_metrics: CostMetrics<A>,

    /// User satisfaction; requires a `user_satisfaction` custom metric
    pub user_satisfaction: Option<A>,

    /// Business value; requires a configured cost model
    pub business_value: Option<A>,
}

/// Cost-related metrics. All `None` unless a [`CostModel`] is configured.
#[derive(Debug, Clone, Default)]
pub struct CostMetrics<A: Float + Send + Sync> {
    /// Computational cost
    pub computational_cost: Option<A>,

    /// Infrastructure cost
    pub infrastructure_cost: Option<A>,

    /// Energy cost
    pub energy_cost: Option<A>,

    /// Opportunity cost
    pub opportunity_cost: Option<A>,

    /// Total cost of ownership
    pub total_cost: Option<A>,
}

/// Historical metrics storage.
#[derive(Debug)]
pub struct HistoricalMetrics<A: Float + Send + Sync> {
    /// Time-series data storage, keyed by **microseconds** since the Unix epoch
    /// (`MetricsSnapshot::timestamp_micros`).
    ///
    /// This key used to be whole seconds, so two samples taken inside the same
    /// second silently overwrote each other and the retention/compression logic
    /// could only ever retain one sample per second no matter how it was
    /// configured. Microsecond resolution matches what the source
    /// `SystemTime` actually carries.
    pub(crate) time_series: BTreeMap<u64, MetricsSnapshot<A>>,

    /// Retention policy
    pub(crate) retention_policy: RetentionPolicy,

    /// Compression settings
    pub(crate) compression_config: CompressionConfig,
}

/// Point-in-time metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot<A: Float + Send + Sync> {
    /// Whole seconds since the Unix epoch.
    ///
    /// Kept at second resolution because that is the granularity the
    /// aggregation buckets (`Minute`/`Hour`/`Day`) are defined over. Use
    /// [`Self::timestamp_micros`] when full resolution matters.
    pub timestamp: u64,

    /// Microseconds since the Unix epoch: the full resolution of the sample
    /// this snapshot was taken from, and the key it is stored under in
    /// [`HistoricalMetrics::time_series`].
    pub timestamp_micros: u64,

    /// Performance metrics at this time
    pub performance: PerformanceMetrics<A>,

    /// Resource metrics at this time
    pub resource: ResourceMetrics,

    /// Quality metrics at this time
    pub quality: QualityMetrics<A>,

    /// Business metrics at this time
    pub business: BusinessMetrics<A>,
}

/// Aggregation periods for historical data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationPeriod {
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

impl AggregationPeriod {
    /// Length of the period in seconds.
    pub fn seconds(self) -> u64 {
        match self {
            AggregationPeriod::Minute => 60,
            AggregationPeriod::Hour => 3_600,
            AggregationPeriod::Day => 86_400,
            AggregationPeriod::Week => 604_800,
            AggregationPeriod::Month => 2_592_000, // 30 days
        }
    }
}

/// Aggregated metrics over a time period.
///
/// M2: this used to be four whole `MetricsSnapshot`s, which forced the
/// aggregator to invent values for every field it could not reduce (and the
/// aggregator simply returned an empty `Vec` instead). It now carries one
/// [`AggregatedSeries`] per metric path that actually had data in the bucket,
/// so an absent metric is absent rather than reported as zero.
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Aggregation period this bucket belongs to
    pub period: AggregationPeriod,

    /// Time period start (seconds since the Unix epoch)
    pub period_start: u64,

    /// Time period end (seconds since the Unix epoch, exclusive)
    pub period_end: u64,

    /// Number of raw snapshots that went into this bucket
    pub sample_count: usize,

    /// One entry per metric path that had at least one observation
    pub series: BTreeMap<String, AggregatedSeries>,
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Raw data retention (seconds)
    pub raw_data_retention: u64,

    /// Aggregated data retention by period
    pub aggregated_retention: HashMap<AggregationPeriod, u64>,

    /// Automatic cleanup enabled
    pub auto_cleanup: bool,

    /// Maximum storage size (bytes)
    pub max_storage_size: u64,
}

/// Data compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression ratio target (retained fraction of raw points)
    pub target_ratio: f64,

    /// Lossy compression tolerance
    pub lossy_tolerance: f64,
}

/// Compression algorithms.
///
/// The historical store applies *temporal* compression (dropping snapshots
/// that are redundant within `lossy_tolerance`). Byte-level codecs are a
/// property of the export path; requesting one there yields an explicit
/// unsupported-operation error rather than silently writing uncompressed
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Lz4,
    Zstd,
    Custom,
}

/// Real-time dashboard
#[derive(Debug)]
pub struct Dashboard {
    /// Dashboard name
    pub name: String,

    /// Dashboard widgets
    pub widgets: Vec<Widget>,

    /// Update frequency
    pub update_frequency: Duration,

    /// Auto-refresh enabled
    pub auto_refresh: bool,
}

/// Dashboard widget
#[derive(Debug)]
pub struct Widget {
    /// Widget type
    pub widget_type: WidgetType,

    /// Metrics to display
    pub metrics: Vec<String>,

    /// Display configuration
    pub config: WidgetConfig,
}

/// Types of dashboard widgets
#[derive(Debug, Clone)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Table,
    Heatmap,
    Histogram,
    ScatterPlot,
    TextDisplay,
}

/// Widget configuration
#[derive(Debug, Clone)]
pub struct WidgetConfig {
    /// Widget title
    pub title: String,

    /// Time range to display
    pub time_range: Duration,

    /// Refresh rate
    pub refresh_rate: Duration,

    /// Color scheme
    pub color_scheme: String,

    /// Size and position
    pub layout: WidgetLayout,
}

/// Widget layout information
#[derive(Debug, Clone)]
pub struct WidgetLayout {
    /// X position
    pub x: u32,

    /// Y position
    pub y: u32,

    /// Width
    pub width: u32,

    /// Height
    pub height: u32,
}

/// Alert system for monitoring
#[derive(Debug)]
pub struct AlertSystem<A: Float + Send + Sync> {
    /// Alert rules
    pub rules: Vec<AlertRule<A>>,

    /// Active alerts
    pub active_alerts: Vec<Alert<A>>,

    /// Alert history
    pub alert_history: Vec<Alert<A>>,

    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,

    /// Per-rule bookkeeping used by the evaluator
    pub(crate) rule_state: HashMap<String, alerts::RuleState>,

    /// Monotonic counter backing collision-free alert identifiers
    pub(crate) next_alert_id: u64,

    /// Maximum retained resolved alerts
    pub(crate) max_history: usize,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule<A: Float + Send + Sync> {
    /// Rule name
    pub name: String,

    /// Metric to monitor
    pub metric_path: String,

    /// Condition
    pub condition: AlertCondition<A>,

    /// Severity level
    pub severity: AlertSeverity,

    /// Evaluation frequency
    pub evaluation_frequency: Duration,

    /// Notification settings
    pub notifications: Vec<String>,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub enum AlertCondition<A: Float + Send + Sync> {
    /// Threshold crossing
    Threshold {
        operator: ComparisonOperator,
        value: A,
    },

    /// Rate of change
    RateOfChange { threshold: A, time_window: Duration },

    /// Anomaly detection
    Anomaly { sensitivity: A },

    /// Custom condition
    Custom { expression: String },
}

/// Comparison operators for alerts
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Active or historical alert
#[derive(Debug, Clone)]
pub struct Alert<A: Float + Send + Sync> {
    /// Alert ID
    pub id: String,

    /// Rule that triggered the alert
    pub rule_name: String,

    /// Timestamp when alert was triggered
    pub triggered_at: SystemTime,

    /// Timestamp when alert was resolved (if applicable)
    pub resolved_at: Option<SystemTime>,

    /// Current metric value
    pub current_value: A,

    /// Threshold that was breached
    pub threshold: A,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert message
    pub message: String,
}

/// Notification channels
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Email {
        addresses: Vec<String>,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    PagerDuty {
        integration_key: String,
    },
    Custom {
        config: HashMap<String, String>,
    },
}

/// Metrics aggregation configuration
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Default aggregation functions
    pub default_functions: Vec<AggregationFunction>,

    /// Custom aggregations by metric
    pub custom_aggregations: HashMap<String, Vec<AggregationFunction>>,

    /// Aggregation intervals
    pub intervals: Vec<Duration>,

    /// Maximum aggregation window
    pub max_window: Duration,
}

/// Aggregation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFunction {
    Mean,
    Median,
    Min,
    Max,
    Sum,
    Count,
    StdDev,
    Percentile(u8), // e.g., Percentile(95) for P95
}

/// Export configuration for metrics
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Export formats
    pub formats: Vec<ExportFormat>,

    /// Export destinations
    pub destinations: Vec<ExportDestination>,

    /// Export frequency
    pub frequency: Duration,

    /// Batch size for exports
    pub batch_size: usize,
}

/// Export formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
    Prometheus,
    InfluxDB,
    Custom { format: String },
}

/// Export destinations
#[derive(Debug, Clone)]
pub enum ExportDestination {
    File {
        path: String,
    },
    Database {
        connection_string: String,
    },
    S3 {
        bucket: String,
        prefix: String,
    },
    Http {
        endpoint: String,
        headers: HashMap<String, String>,
    },
    Kafka {
        topic: String,
        brokers: Vec<String>,
    },
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> StreamingMetricsCollector<A> {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::with_window(512)
    }

    /// Create a collector retaining `window` raw observations per series.
    pub fn with_window(window: usize) -> Self {
        Self {
            performance_metrics: PerformanceMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            quality_metrics: QualityMetrics::default(),
            business_metrics: BusinessMetrics::default(),
            historical_data: HistoricalMetrics::new(),
            dashboards: Vec::new(),
            alert_system: AlertSystem::new(),
            aggregation_config: AggregationConfig::default(),
            export_config: ExportConfig::default(),
            accumulator: MetricsAccumulator::new(window),
            slo: None,
            cost_model: None,
            last_export: None,
        }
    }

    /// Configure service level objectives so `slo_compliance` becomes a real
    /// measurement instead of `None`.
    pub fn set_slo_targets(&mut self, targets: SloTargets) {
        self.slo = if targets.is_empty() {
            None
        } else {
            Some(targets)
        };
    }

    /// Configure a cost model so the cost metrics become real.
    pub fn set_cost_model(&mut self, model: CostModel<A>) {
        self.cost_model = Some(model);
    }

    /// Feed OS-level resource measurements this collector cannot take itself.
    pub fn record_resource_probe(&mut self, probe: ResourceProbe) {
        self.accumulator.record_resource_probe(probe);
    }

    /// Feed robustness measurements obtained from perturbation experiments.
    pub fn record_robustness_probe(&mut self, probe: RobustnessProbe<A>) {
        self.accumulator.record_robustness_probe(probe);
    }

    /// Report an observed outage so `availability` becomes a real ratio.
    pub fn record_outage(&mut self, downtime: Duration) {
        self.accumulator.record_outage(downtime);
    }

    /// Report a detected concept-drift event.
    pub fn record_drift_event(
        &mut self,
        magnitude: A,
        confidence: A,
        detection_latency: Duration,
        adaptation_effectiveness: Option<A>,
    ) {
        self.accumulator.record_drift_event(
            magnitude,
            confidence,
            detection_latency,
            adaptation_effectiveness,
        );
    }

    /// Report measured energy consumption for the most recent samples.
    pub fn record_energy(&mut self, joules: f64) {
        self.accumulator.record_energy(joules);
    }

    /// Record a new metrics sample
    pub fn record_sample(&mut self, sample: MetricsSample<A>) -> Result<()> {
        self.accumulator.ingest(&sample);

        // Update current metrics from the accumulated observations.
        self.update_performance_metrics(&sample)?;
        self.update_resource_metrics(&sample)?;
        self.update_quality_metrics(&sample)?;
        self.update_business_metrics(&sample)?;

        // Store historical data (M4: saturating, never panicking).
        let timestamp = unix_timestamp(sample.timestamp);
        let timestamp_micros = unix_timestamp_micros(sample.timestamp);

        let snapshot = MetricsSnapshot {
            timestamp,
            timestamp_micros,
            performance: self.performance_metrics.clone(),
            resource: self.resource_metrics.clone(),
            quality: self.quality_metrics.clone(),
            business: self.business_metrics.clone(),
        };

        self.historical_data.store_snapshot(snapshot)?;

        // Check alerts
        let summary = self.current_summary();
        self.alert_system.evaluate_rules(&sample, &summary)?;

        Ok(())
    }

    /// Get current metrics summary
    pub fn get_current_metrics(&self) -> MetricsSummary<A> {
        self.current_summary()
    }

    pub(crate) fn current_summary(&self) -> MetricsSummary<A> {
        MetricsSummary {
            performance: self.performance_metrics.clone(),
            resource: self.resource_metrics.clone(),
            quality: self.quality_metrics.clone(),
            business: self.business_metrics.clone(),
            timestamp: SystemTime::now(),
        }
    }

    /// Get historical metrics for a time range
    pub fn get_historical_metrics(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<MetricsSnapshot<A>>> {
        self.historical_data.get_range(start_time, end_time)
    }

    /// Number of raw snapshots currently retained.
    pub fn retained_snapshot_count(&self) -> usize {
        self.historical_data.time_series.len()
    }

    /// Register an alert rule. Rules whose condition cannot be evaluated by
    /// this crate are rejected here instead of being silently ignored at
    /// evaluation time.
    pub fn add_alert_rule(&mut self, rule: AlertRule<A>) -> Result<()> {
        self.alert_system.add_rule(rule)
    }

    /// Currently firing alerts.
    pub fn active_alerts(&self) -> &[Alert<A>] {
        &self.alert_system.active_alerts
    }

    /// Resolved alerts, most recent last.
    pub fn alert_history(&self) -> &[Alert<A>] {
        &self.alert_system.alert_history
    }

    /// Register a dashboard definition.
    pub fn add_dashboard(&mut self, dashboard: Dashboard) {
        self.dashboards.push(dashboard);
    }

    /// Resolve every metric path referenced by a dashboard's widgets against
    /// the current metrics.
    pub fn render_dashboard(&self, name: &str) -> Option<Vec<(String, Option<f64>)>> {
        let dashboard = self.dashboards.iter().find(|d| d.name == name)?;
        let summary = self.current_summary();
        let mut resolved = Vec::new();
        for widget in &dashboard.widgets {
            for metric in &widget.metrics {
                resolved.push((metric.clone(), alerts::resolve_metric(&summary, metric)));
            }
        }
        Some(resolved)
    }

    /// Access to the aggregation configuration.
    pub fn aggregation_config(&self) -> &AggregationConfig {
        &self.aggregation_config
    }

    /// Replace the aggregation configuration.
    pub fn set_aggregation_config(&mut self, config: AggregationConfig) {
        self.aggregation_config = config;
    }

    /// Replace the export configuration.
    pub fn set_export_config(&mut self, config: ExportConfig) {
        self.export_config = config;
    }

    /// Replace the retention policy governing the raw time series.
    pub fn set_retention_policy(&mut self, policy: RetentionPolicy) {
        self.historical_data.retention_policy = policy;
        self.historical_data.prune();
    }

    /// Replace the temporal-compression configuration.
    pub fn set_compression_config(&mut self, config: CompressionConfig) {
        self.historical_data.compression_config = config;
    }
}

impl<A: Float + Default + Clone + std::fmt::Debug + Send + Sync> Default
    for StreamingMetricsCollector<A>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Individual metrics sample
#[derive(Debug, Clone)]
pub struct MetricsSample<A: Float + Send + Sync> {
    /// Timestamp of the sample
    pub timestamp: SystemTime,

    /// Loss value
    pub loss: A,

    /// Gradient magnitude
    pub gradient_magnitude: A,

    /// Processing time
    pub processing_time: Duration,

    /// Memory usage
    pub memory_usage: u64,

    /// Gradient computation time, when the caller measured it separately
    pub gradient_computation_time: Option<Duration>,

    /// Update application time, when the caller measured it separately
    pub update_application_time: Option<Duration>,

    /// Communication time, when the caller measured it separately
    pub communication_time: Option<Duration>,

    /// Queue waiting time, when the caller measured it separately
    pub queue_wait_time: Option<Duration>,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, A>,
}

impl<A: Float + Send + Sync> MetricsSample<A> {
    /// Build a sample from the measurements every caller has.
    pub fn new(
        timestamp: SystemTime,
        loss: A,
        gradient_magnitude: A,
        processing_time: Duration,
        memory_usage: u64,
    ) -> Self {
        Self {
            timestamp,
            loss,
            gradient_magnitude,
            processing_time,
            memory_usage,
            gradient_computation_time: None,
            update_application_time: None,
            communication_time: None,
            queue_wait_time: None,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Complete metrics summary
#[derive(Debug, Clone)]
pub struct MetricsSummary<A: Float + Send + Sync> {
    /// Performance metrics
    pub performance: PerformanceMetrics<A>,

    /// Resource metrics
    pub resource: ResourceMetrics,

    /// Quality metrics
    pub quality: QualityMetrics<A>,

    /// Business metrics
    pub business: BusinessMetrics<A>,

    /// Summary timestamp
    pub timestamp: SystemTime,
}

// Implement default traits for metrics structs
impl<A: Float + Default + Send + Sync> Default for PerformanceMetrics<A> {
    fn default() -> Self {
        Self {
            throughput: ThroughputMetrics::default(),
            latency: LatencyMetrics::default(),
            accuracy: AccuracyMetrics::default(),
            stability: StabilityMetrics::default(),
            efficiency: EfficiencyMetrics::default(),
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            samples_per_second: 0.0,
            updates_per_second: 0.0,
            gradients_per_second: 0.0,
            peak_throughput: 0.0,
            min_throughput: f64::MAX,
            throughput_variance: 0.0,
            throughput_trend: 0.0,
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            end_to_end: LatencyStats::default(),
            gradient_computation: None,
            update_application: None,
            communication: None,
            queue_wait_time: None,
            jitter: 0.0,
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            mean: Duration::from_micros(0),
            median: Duration::from_micros(0),
            p95: Duration::from_micros(0),
            p99: Duration::from_micros(0),
            p999: Duration::from_micros(0),
            max: Duration::from_micros(0),
            min: Duration::from_micros(u64::MAX),
            std_dev: Duration::from_micros(0),
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for AccuracyMetrics<A> {
    fn default() -> Self {
        Self {
            current_loss: A::default(),
            loss_reduction_rate: A::default(),
            convergence_rate: A::default(),
            prediction_accuracy: None,
            gradient_magnitude: A::default(),
            parameter_stability: A::default(),
            learning_progress: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for StabilityMetrics<A> {
    fn default() -> Self {
        Self {
            loss_variance: A::default(),
            gradient_variance: A::default(),
            parameter_drift: A::default(),
            oscillation_score: A::default(),
            divergence_probability: A::default(),
            stability_confidence: A::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for EfficiencyMetrics<A> {
    fn default() -> Self {
        Self {
            computational_efficiency: None,
            memory_efficiency: None,
            communication_efficiency: None,
            energy_efficiency: None,
            resource_utilization: A::default(),
            cost_efficiency: None,
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for QualityMetrics<A> {
    fn default() -> Self {
        Self {
            data_quality: A::default(),
            model_quality: ModelQuality::default(),
            concept_drift: ConceptDriftMetrics::default(),
            anomaly_detection: AnomalyMetrics::default(),
            robustness: RobustnessMetrics::default(),
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for ModelQuality<A> {
    fn default() -> Self {
        Self {
            training_quality: A::default(),
            generalization_score: None,
            overfitting_score: None,
            underfitting_score: None,
            complexity_score: None,
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for ConceptDriftMetrics<A> {
    fn default() -> Self {
        Self {
            drift_confidence: None,
            drift_magnitude: None,
            drift_frequency: 0.0,
            adaptation_effectiveness: None,
            detection_latency: None,
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for AnomalyMetrics<A> {
    fn default() -> Self {
        Self {
            anomaly_score: A::default(),
            false_positive_rate: None,
            false_negative_rate: None,
            detection_accuracy: None,
            anomaly_frequency: 0.0,
        }
    }
}

impl<A: Float + Default + Send + Sync> Default for BusinessMetrics<A> {
    fn default() -> Self {
        Self {
            availability: None,
            slo_compliance: None,
            cost_metrics: CostMetrics::default(),
            user_satisfaction: None,
            business_value: None,
        }
    }
}

impl<A: Float + Send + Sync> HistoricalMetrics<A> {
    fn new() -> Self {
        Self {
            time_series: BTreeMap::new(),
            retention_policy: RetentionPolicy::default(),
            compression_config: CompressionConfig::default(),
        }
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        let mut aggregated_retention = HashMap::new();
        aggregated_retention.insert(AggregationPeriod::Minute, 3600 * 24); // 1 day
        aggregated_retention.insert(AggregationPeriod::Hour, 3600 * 24 * 7); // 1 week
        aggregated_retention.insert(AggregationPeriod::Day, 3600 * 24 * 30); // 1 month
        aggregated_retention.insert(AggregationPeriod::Week, 3600 * 24 * 365); // 1 year
        aggregated_retention.insert(AggregationPeriod::Month, 3600 * 24 * 365 * 5); // 5 years

        Self {
            raw_data_retention: 3600 * 24, // 1 day
            aggregated_retention,
            auto_cleanup: true,
            max_storage_size: 1024 * 1024 * 1024 * 10, // 10GB
        }
    }
}

impl Default for CompressionConfig {
    /// Temporal compression is **opt-in**.
    ///
    /// The previous default was `enabled: true` with `algorithm: Zstd`, which
    /// was a no-op because no byte codec was ever applied. Now that the
    /// temporal pass is real, leaving it on by default would silently discard
    /// ~70% of the retained history (`target_ratio: 0.3`) — a surprising
    /// default for a metrics store, and a behaviour change relative to what
    /// callers actually observed before. Retention and the storage cap are what
    /// bound the series by default; downsampling is something an operator asks
    /// for through [`StreamingMetricsCollector::set_compression_config`].
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::None,
            target_ratio: 0.3,
            lossy_tolerance: 0.01,
        }
    }
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            default_functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Min,
                AggregationFunction::Max,
                AggregationFunction::StdDev,
            ],
            custom_aggregations: HashMap::new(),
            intervals: vec![
                Duration::from_secs(60),    // 1 minute
                Duration::from_secs(3600),  // 1 hour
                Duration::from_secs(86400), // 1 day
            ],
            max_window: Duration::from_secs(86400 * 30), // 30 days
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            formats: vec![ExportFormat::Json],
            destinations: vec![ExportDestination::File {
                path: std::env::temp_dir()
                    .join("streaming_metrics")
                    .to_string_lossy()
                    .into_owned(),
            }],
            frequency: Duration::from_secs(300), // 5 minutes
            batch_size: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = StreamingMetricsCollector::<f64>::new();
        assert_eq!(
            collector.performance_metrics.throughput.samples_per_second,
            0.0
        );
        assert!(collector.dashboards.is_empty());
    }

    #[test]
    fn test_metrics_sample() {
        let sample = MetricsSample::new(
            SystemTime::now(),
            0.5f64,
            0.1f64,
            Duration::from_millis(10),
            1024,
        );

        assert_eq!(sample.loss, 0.5f64);
        assert_eq!(sample.gradient_magnitude, 0.1f64);
    }

    #[test]
    fn test_latency_stats_default() {
        let stats = LatencyStats::default();
        assert_eq!(stats.mean, Duration::from_micros(0));
        assert_eq!(stats.min, Duration::from_micros(u64::MAX));
    }

    #[test]
    fn test_aggregation_period() {
        let periods = [
            AggregationPeriod::Minute,
            AggregationPeriod::Hour,
            AggregationPeriod::Day,
            AggregationPeriod::Week,
            AggregationPeriod::Month,
        ];

        assert_eq!(periods.len(), 5);
    }

    #[test]
    fn test_alert_severity() {
        let severities = [
            AlertSeverity::Critical,
            AlertSeverity::Warning,
            AlertSeverity::Info,
        ];

        assert_eq!(severities.len(), 3);
    }
}
