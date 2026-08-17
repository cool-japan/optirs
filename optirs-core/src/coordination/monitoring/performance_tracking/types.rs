//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::functions::{MetricFilter, NotificationChannel, PerformanceAnalyzer, StorageBackend};

/// Adaptation constraints
#[derive(Debug, Clone)]
pub struct AdaptationConstraints<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum threshold
    pub min_threshold: Option<T>,
    /// Maximum threshold
    pub max_threshold: Option<T>,
    /// Maximum change rate
    pub max_change_rate: T,
    /// Adaptation window
    pub adaptation_window: Duration,
}
/// Query value types
#[derive(Debug, Clone)]
pub enum QueryValue<T: Float + Debug + Send + Sync + 'static> {
    Number(T),
    String(String),
    Boolean(bool),
    List(Vec<String>),
}
/// Alert aggregation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertAggregationFunction {
    Count,
    Sum,
    Average,
    Maximum,
    Minimum,
    Custom,
}
/// Analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult<T: Float + Debug + Send + Sync + 'static> {
    /// Analysis type
    pub analysis_type: String,
    /// Analysis insights
    pub insights: Vec<AnalysisInsight<T>>,
    /// Analysis confidence
    pub confidence: T,
    /// Analysis recommendations
    pub recommendations: Vec<String>,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}
/// Metric value with context
#[derive(Debug, Clone)]
pub struct MetricValue<T: Float + Debug + Send + Sync + 'static> {
    /// Current value
    pub value: T,
    /// Value type
    pub value_type: MetricType,
    /// Unit of measurement
    pub unit: String,
    /// Value bounds
    pub bounds: Option<MetricBounds<T>>,
    /// Value confidence
    pub confidence: T,
    /// Value tags
    pub tags: Vec<String>,
}
/// Category-specific metrics
#[derive(Debug, Clone)]
pub struct CategoryMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Individual metrics
    pub metrics: HashMap<String, MetricValue<T>>,
    /// Category weight
    pub weight: T,
    /// Category status
    pub status: CategoryStatus,
    /// Category trends
    pub trends: CategoryTrends<T>,
}
/// Overflow handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowHandling {
    /// Drop oldest metrics
    DropOldest,
    /// Drop newest metrics
    DropNewest,
    /// Compress metrics
    Compress,
    /// Flush to storage
    FlushToStorage,
    /// Block collection
    Block,
}
/// Cached aggregation
#[derive(Debug, Clone)]
pub struct CachedAggregation<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregation key
    pub key: String,
    /// Aggregated value
    pub value: T,
    /// Cache timestamp
    pub timestamp: SystemTime,
    /// Time-to-live
    pub ttl: Duration,
    /// Access count
    pub access_count: usize,
}
/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Alert condition
    pub condition: AlertCondition<T>,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert threshold
    pub threshold: AlertThreshold<T>,
    /// Alert frequency limits
    pub frequency_limits: FrequencyLimits,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}
/// Analysis insight
#[derive(Debug, Clone)]
pub struct AnalysisInsight<T: Float + Debug + Send + Sync + 'static> {
    /// Insight type
    pub insight_type: InsightType,
    /// Insight description
    pub description: String,
    /// Insight severity
    pub severity: InsightSeverity,
    /// Supporting evidence
    pub evidence: Vec<Evidence<T>>,
    /// Confidence level
    pub confidence: T,
}
/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Theme settings
    pub theme: String,
    /// Auto-refresh enabled
    pub auto_refresh: bool,
    /// Default time range
    pub default_time_range: Duration,
    /// Custom dashboard parameters
    pub custom_params: HashMap<String, T>,
}
/// Insight severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InsightSeverity {
    Informational,
    Minor,
    Major,
    Critical,
}
/// Comparison operators for alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    InRange,
    OutOfRange,
}
/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Metrics stored per second
    pub storage_rate: T,
    /// Queries per second
    pub query_rate: T,
    /// Average storage latency
    pub average_storage_latency: Duration,
    /// Average query latency
    pub average_query_latency: Duration,
    /// Storage errors
    pub storage_errors: usize,
    /// Query errors
    pub query_errors: usize,
}
/// Window alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowAlignment {
    /// Start-aligned
    Start,
    /// End-aligned
    End,
    /// Center-aligned
    Center,
    /// Calendar-aligned
    Calendar,
}
/// Widget types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetType {
    LineChart,
    BarChart,
    PieChart,
    Gauge,
    Table,
    Metric,
    Alert,
    Custom,
}
/// Bounded buffer of collected metric samples.
///
/// Until 0.3.2 every field here was written once at construction and never
/// read: nothing pushed, nothing drained, and `overflow_handling` selected
/// nothing, so a "buffer" sat between the collector and its consumer doing
/// exactly nothing.
#[derive(Debug)]
pub struct MetricBuffer<T: Float + Debug + Send + Sync + 'static> {
    /// Buffer capacity
    capacity: usize,
    /// Buffered metrics
    metrics: VecDeque<BufferedMetric<T>>,
    /// Buffer strategy
    strategy: BufferStrategy,
    /// Overflow handling
    overflow_handling: OverflowHandling,
}

impl<T: Float + Debug + Send + Sync + 'static> MetricBuffer<T> {
    /// A buffer holding at most `capacity` samples.
    pub fn new(
        capacity: usize,
        strategy: BufferStrategy,
        overflow_handling: OverflowHandling,
    ) -> Self {
        Self {
            capacity: capacity.max(1),
            metrics: VecDeque::new(),
            strategy,
            overflow_handling,
        }
    }

    /// Buffer `metric`, applying [`OverflowHandling`] when full.
    ///
    /// Returns `false` when the sample was dropped rather than stored.
    pub fn push(&mut self, metric: BufferedMetric<T>) -> bool {
        if self.metrics.len() < self.capacity {
            self.metrics.push_back(metric);
            return true;
        }
        match self.overflow_handling {
            OverflowHandling::DropOldest => {
                let _ = self.metrics.pop_front();
                self.metrics.push_back(metric);
                true
            }
            OverflowHandling::DropNewest => false,
            OverflowHandling::Compress | OverflowHandling::FlushToStorage => {
                // Neither compression nor a storage flush is implemented at this
                // layer, and silently losing the sample would be the worst
                // reading; grow instead and report the sample as kept.
                self.metrics.push_back(metric);
                self.capacity = self.metrics.len();
                true
            }
            OverflowHandling::Block => {
                // Blocking has no meaning in a synchronous in-process buffer.
                self.metrics.push_back(metric);
                self.capacity = self.metrics.len();
                true
            }
        }
    }

    /// Remove and return every buffered sample.
    pub fn drain(&mut self) -> Vec<BufferedMetric<T>> {
        self.metrics.drain(..).collect()
    }

    /// Number of buffered samples.
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Whether the buffer holds no samples.
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Maximum number of samples retained.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The configured buffering strategy.
    pub fn strategy(&self) -> BufferStrategy {
        self.strategy
    }

    /// The configured overflow behaviour.
    pub fn overflow_handling(&self) -> OverflowHandling {
        self.overflow_handling
    }
}
/// Alert aggregation settings
#[derive(Debug)]
pub struct AlertAggregation<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregation strategy
    pub strategy: AlertAggregationStrategy,
    /// Aggregation window
    pub window: Duration,
    /// Aggregation rules
    rules: Vec<AggregationRule<T>>,
    /// Aggregated alerts
    aggregated_alerts: HashMap<String, AggregatedAlert<T>>,
}
/// Alert threshold specification
#[derive(Debug, Clone)]
pub struct AlertThreshold<T: Float + Debug + Send + Sync + 'static> {
    /// Warning threshold
    pub warning: Option<T>,
    /// Critical threshold
    pub critical: Option<T>,
    /// Fatal threshold
    pub fatal: Option<T>,
    /// Hysteresis values
    pub hysteresis: Option<HysteresisValues<T>>,
    /// Threshold adaptation
    pub adaptation: Option<ThresholdAdaptation<T>>,
}
/// Trend directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
    Unknown,
}
/// Alert condition specification
#[derive(Debug, Clone)]
pub enum AlertCondition<T: Float + Debug + Send + Sync + 'static> {
    /// Threshold-based condition
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: T,
        duration: Option<Duration>,
    },
    /// Trend-based condition
    Trend {
        metric: String,
        direction: TrendDirection,
        magnitude: T,
        duration: Duration,
    },
    /// Anomaly-based condition
    Anomaly {
        metric: String,
        sensitivity: T,
        method: AnomalyDetectionMethod,
    },
    /// Composite condition
    Composite {
        conditions: Vec<AlertCondition<T>>,
        operator: LogicalOperator,
    },
    /// Custom condition
    Custom {
        expression: String,
        parameters: HashMap<String, T>,
    },
}
/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert<T: Float + Debug + Send + Sync + 'static> {
    /// Alert identifier
    pub alert_id: String,
    /// Alert rule that triggered
    pub rule_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Triggering metric
    pub metric: String,
    /// Metric value that triggered alert
    pub metric_value: T,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert context
    pub context: AlertContext<T>,
    /// Alert status
    pub status: AlertStatus,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}
/// Alert manager for performance alerts
#[derive(Debug)]
pub struct AlertManager<T: Float + Debug + Send + Sync + 'static> {
    /// Alert rules
    alert_rules: HashMap<String, AlertRule<T>>,
    /// Active alerts
    active_alerts: HashMap<String, PerformanceAlert<T>>,
    /// Alert history
    alert_history: VecDeque<AlertHistoryEntry<T>>,
    /// Notification channels
    notification_channels: Vec<Box<dyn NotificationChannel<T>>>,
    /// Alert aggregation
    alert_aggregation: AlertAggregation<T>,
    /// Alert statistics
    stats: AlertStatistics<T>,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> AlertManager<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            alert_rules: HashMap::new(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_channels: Vec::new(),
            alert_aggregation: AlertAggregation {
                strategy: AlertAggregationStrategy::None,
                window: Duration::from_secs(60),
                rules: Vec::new(),
                aggregated_alerts: HashMap::new(),
            },
            stats: AlertStatistics::default(),
        })
    }
    /// Register an alert rule.
    ///
    /// Until 0.3.2 `alert_rules` was an empty map nothing could populate and
    /// `check_alerts` returned `Ok(vec![])` for every input, so an "alert
    /// manager" wired into a monitoring loop reported an all-clear regardless of
    /// what the metrics said.
    pub fn add_rule(&mut self, rule: AlertRule<T>) -> Result<()> {
        if rule.rule_id.is_empty() {
            return Err(OptimError::InvalidParameter(
                "an alert rule must have a non-empty rule_id".to_string(),
            ));
        }
        self.alert_rules.insert(rule.rule_id.clone(), rule);
        Ok(())
    }

    /// Register an aggregation rule.
    ///
    /// Until 0.3.2 `AlertAggregation::rules` and `aggregated_alerts` were
    /// written once in `AlertManager::new` and never read, so `strategy` and
    /// `window` selected nothing and no aggregation happened at any setting.
    pub fn add_aggregation_rule(&mut self, rule: AggregationRule<T>) -> Result<()> {
        if rule.min_count == 0 {
            return Err(OptimError::InvalidParameter(format!(
                "aggregation rule '{}' has min_count 0, which would aggregate an empty group",
                rule.rule_id
            )));
        }
        self.alert_aggregation.rules.push(rule);
        Ok(())
    }

    /// Aggregations produced by the most recent [`Self::aggregate_alerts`],
    /// keyed by rule id.
    pub fn aggregated_alerts(&self) -> &HashMap<String, AggregatedAlert<T>> {
        &self.alert_aggregation.aggregated_alerts
    }

    /// Group the currently active alerts under every registered rule and record
    /// the resulting summaries.
    ///
    /// A rule fires only when its group holds at least `min_count` alerts within
    /// [`AlertAggregation::window`] of the newest one; smaller groups are left
    /// alone. [`AlertAggregationStrategy::None`] disables aggregation entirely,
    /// which is the setting `AlertManager::new` installs.
    pub fn aggregate_alerts(&mut self) -> usize {
        self.alert_aggregation.aggregated_alerts.clear();
        if matches!(
            self.alert_aggregation.strategy,
            AlertAggregationStrategy::None
        ) {
            return 0;
        }

        let window = self.alert_aggregation.window;
        let newest = self
            .active_alerts
            .values()
            .map(|alert| alert.timestamp)
            .max();
        let Some(newest) = newest else { return 0 };

        let mut produced = 0usize;
        for rule in &self.alert_aggregation.rules {
            let mut groups: HashMap<String, Vec<&PerformanceAlert<T>>> = HashMap::new();
            for alert in self.active_alerts.values() {
                let age = newest
                    .duration_since(alert.timestamp)
                    .unwrap_or(Duration::ZERO);
                if age > window {
                    continue;
                }
                let key = match &rule.grouping {
                    AlertGrouping::ByMetric => alert.metric.clone(),
                    AlertGrouping::BySeverity => format!("{:?}", alert.severity),
                    AlertGrouping::BySource => alert.context.source.clone(),
                    AlertGrouping::ByRule => alert.rule_id.clone(),
                    AlertGrouping::Custom(label) => label.clone(),
                };
                groups.entry(key).or_default().push(alert);
            }

            for (key, alerts) in groups {
                if alerts.len() < rule.min_count {
                    continue;
                }
                let mut count_by_severity: HashMap<AlertSeverity, usize> = HashMap::new();
                for alert in &alerts {
                    *count_by_severity.entry(alert.severity).or_insert(0) += 1;
                }
                let values: Vec<f64> = alerts
                    .iter()
                    .filter_map(|alert| alert.metric_value.to_f64())
                    .collect();
                let combined = match rule.function {
                    AlertAggregationFunction::Count => Some(alerts.len() as f64),
                    AlertAggregationFunction::Sum => Some(values.iter().sum()),
                    AlertAggregationFunction::Average if !values.is_empty() => {
                        Some(values.iter().sum::<f64>() / values.len() as f64)
                    }
                    AlertAggregationFunction::Maximum => values.iter().copied().reduce(f64::max),
                    AlertAggregationFunction::Minimum => values.iter().copied().reduce(f64::min),
                    // No custom function is installed, so none is applied; the
                    // group is still summarised by count.
                    AlertAggregationFunction::Average | AlertAggregationFunction::Custom => None,
                };
                let oldest = alerts.iter().map(|alert| alert.timestamp).min();
                let time_span = oldest
                    .and_then(|oldest| newest.duration_since(oldest).ok())
                    .unwrap_or(Duration::ZERO);
                let most_common_source = {
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    for alert in &alerts {
                        *counts.entry(alert.context.source.as_str()).or_insert(0) += 1;
                    }
                    counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(source, _)| source.to_string())
                };

                let aggregated_id = format!("{}::{key}", rule.rule_id);
                self.alert_aggregation.aggregated_alerts.insert(
                    aggregated_id.clone(),
                    AggregatedAlert {
                        aggregated_id,
                        alerts: alerts.iter().map(|alert| alert.alert_id.clone()).collect(),
                        summary: AggregationSummary {
                            total_count: alerts.len(),
                            count_by_severity,
                            average_metric_value: combined.and_then(T::from),
                            time_span,
                            most_common_source,
                        },
                        timestamp: newest,
                        metadata: HashMap::new(),
                    },
                );
                produced += 1;
            }
        }
        produced
    }

    /// Register a notification channel. Every alert that fires is dispatched
    /// to each registered channel.
    pub fn add_notification_channel(&mut self, channel: Box<dyn NotificationChannel<T>>) {
        self.notification_channels.push(channel);
    }

    /// Registered rules, keyed by `rule_id`.
    pub fn rules(&self) -> &HashMap<String, AlertRule<T>> {
        &self.alert_rules
    }

    /// Alerts currently in [`AlertStatus::Active`], keyed by `rule_id`.
    pub fn active_alerts(&self) -> &HashMap<String, PerformanceAlert<T>> {
        &self.active_alerts
    }

    /// Status transitions recorded so far, oldest first.
    pub fn history(&self) -> impl Iterator<Item = &AlertHistoryEntry<T>> {
        self.alert_history.iter()
    }

    /// Aggregation settings in force.
    pub fn aggregation(&self) -> &AlertAggregation<T> {
        &self.alert_aggregation
    }

    /// Cumulative alert statistics.
    pub fn statistics(&self) -> &AlertStatistics<T> {
        &self.stats
    }

    /// Evaluate every registered rule against `metrics` and return the alerts
    /// that fired on this call.
    ///
    /// Only [`AlertCondition::Threshold`] is evaluated: it is decidable from a
    /// single metrics snapshot. `Trend` and `Anomaly` conditions need a metric
    /// *history* that this manager is not given, so a rule carrying one is
    /// reported as an error rather than silently treated as "did not fire" --
    /// an unevaluated condition must never read as an all-clear.
    ///
    /// A rule that fires while its previous alert is still active is not
    /// re-reported (the alert stays in `active_alerts`); a rule whose condition
    /// no longer holds resolves its alert and records the transition in the
    /// history.
    pub fn check_alerts(
        &mut self,
        metrics: &PerformanceMetrics<T>,
    ) -> Result<Vec<PerformanceAlert<T>>> {
        let now = SystemTime::now();
        let mut fired = Vec::new();
        let mut resolved: Vec<String> = Vec::new();

        for (rule_id, rule) in &self.alert_rules {
            let (metric_name, operator, bound) = match &rule.condition {
                AlertCondition::Threshold {
                    metric,
                    operator,
                    value,
                    ..
                } => (metric.clone(), *operator, *value),
                other => {
                    return Err(OptimError::UnsupportedOperation(format!(
                        "alert rule '{rule_id}' uses {other:?}, which needs a metric history this \
                         manager does not hold; only AlertCondition::Threshold can be decided from \
                         a single snapshot"
                    )))
                }
            };

            let Some(observed) = lookup_metric_value(metrics, &metric_name) else {
                continue;
            };
            let triggered = compare_metric(observed, operator, bound);

            if triggered {
                if self.active_alerts.contains_key(rule_id) {
                    continue;
                }
                let alert = PerformanceAlert {
                    alert_id: format!("{rule_id}-{}", self.stats.total_alerts + fired.len() + 1),
                    rule_id: rule_id.clone(),
                    severity: rule.severity,
                    message: format!(
                        "{}: {metric_name} = {observed:?} {operator:?} {bound:?}",
                        rule.name
                    ),
                    metric: metric_name,
                    metric_value: observed,
                    timestamp: now,
                    context: AlertContext {
                        source: "PerformanceTracker".to_string(),
                        environment: HashMap::new(),
                        related_metrics: HashMap::new(),
                        historical_data: Vec::new(),
                        predictions: Vec::new(),
                    },
                    status: AlertStatus::Active,
                    metadata: rule.metadata.clone(),
                };
                fired.push(alert);
            } else if self.active_alerts.contains_key(rule_id) {
                resolved.push(rule_id.clone());
            }
        }

        for rule_id in resolved {
            if let Some(mut alert) = self.active_alerts.remove(&rule_id) {
                let duration = now
                    .duration_since(alert.timestamp)
                    .unwrap_or(Duration::ZERO);
                alert.status = AlertStatus::Resolved;
                self.record_history(
                    alert,
                    AlertStatus::Active,
                    AlertStatus::Resolved,
                    duration,
                    "condition no longer holds",
                );
            }
        }

        for alert in &fired {
            for channel in &mut self.notification_channels {
                // A channel that cannot deliver is a real failure: an alert that
                // fired but was never delivered must not look like a quiet system.
                channel.send_notification(alert)?;
            }
            self.active_alerts
                .insert(alert.rule_id.clone(), alert.clone());
            self.stats.total_alerts += 1;
            *self
                .stats
                .alerts_by_severity
                .entry(alert.severity)
                .or_insert(0) += 1;
            self.record_history(
                alert.clone(),
                AlertStatus::Resolved,
                AlertStatus::Active,
                Duration::ZERO,
                "threshold condition met",
            );
        }

        Ok(fired)
    }

    fn record_history(
        &mut self,
        alert: PerformanceAlert<T>,
        from_status: AlertStatus,
        to_status: AlertStatus,
        duration: Duration,
        reason: &str,
    ) {
        self.alert_history.push_back(AlertHistoryEntry {
            entry_id: format!("{}-{to_status:?}", alert.alert_id),
            alert,
            status_change: AlertStatusChange {
                from_status,
                to_status,
                duration,
            },
            timestamp: SystemTime::now(),
            reason: reason.to_string(),
            user: None,
        });
        while self.alert_history.len() > MAX_ALERT_HISTORY {
            let _ = self.alert_history.pop_front();
        }
    }
}

/// Maximum number of alert status transitions retained by an [`AlertManager`].
pub const MAX_ALERT_HISTORY: usize = 1000;

/// Look up a metric by `category.metric` or by bare metric name across every
/// category, returning the first match.
fn lookup_metric_value<T: Float + Debug + Send + Sync + 'static>(
    metrics: &PerformanceMetrics<T>,
    name: &str,
) -> Option<T> {
    if let Some((category, metric)) = name.split_once('.') {
        if let Some(found) = metrics
            .categories
            .get(category)
            .and_then(|c| c.metrics.get(metric))
        {
            return Some(found.value);
        }
    }
    metrics
        .categories
        .values()
        .find_map(|category| category.metrics.get(name).map(|metric| metric.value))
}

/// Evaluate `observed <op> bound`.
///
/// `InRange` / `OutOfRange` need two bounds; a single-valued threshold cannot
/// express them, so they never fire here.
fn compare_metric<T: Float + Debug + Send + Sync + 'static>(
    observed: T,
    operator: ComparisonOperator,
    bound: T,
) -> bool {
    match operator {
        ComparisonOperator::Equal => observed == bound,
        ComparisonOperator::NotEqual => observed != bound,
        ComparisonOperator::GreaterThan => observed > bound,
        ComparisonOperator::GreaterThanOrEqual => observed >= bound,
        ComparisonOperator::LessThan => observed < bound,
        ComparisonOperator::LessThanOrEqual => observed <= bound,
        ComparisonOperator::InRange | ComparisonOperator::OutOfRange => false,
    }
}
/// Alert history entry
#[derive(Debug, Clone)]
pub struct AlertHistoryEntry<T: Float + Debug + Send + Sync + 'static> {
    /// History entry identifier
    pub entry_id: String,
    /// Associated alert
    pub alert: PerformanceAlert<T>,
    /// Status change
    pub status_change: AlertStatusChange,
    /// Change timestamp
    pub timestamp: SystemTime,
    /// Change reason
    pub reason: String,
    /// User who made the change
    pub user: Option<String>,
}
/// Query aggregation
#[derive(Debug, Clone)]
pub struct QueryAggregation {
    /// Aggregation function
    pub function: String,
    /// Group by fields
    pub group_by: Vec<String>,
    /// Having conditions
    pub having: Vec<String>,
}
/// Query filter
#[derive(Debug, Clone)]
pub struct QueryFilter<T: Float + Debug + Send + Sync + 'static> {
    /// Field to filter on
    pub field: String,
    /// Filter operator
    pub operator: ComparisonOperator,
    /// Filter value
    pub value: QueryValue<T>,
}
/// Metric query specification
#[derive(Debug, Clone)]
pub struct MetricQuery<T: Float + Debug + Send + Sync + 'static> {
    /// Metric names to query
    pub metrics: Vec<String>,
    /// Time range
    pub time_range: Option<TimeRange>,
    /// Filters
    pub filters: Vec<QueryFilter<T>>,
    /// Aggregation
    pub aggregation: Option<QueryAggregation>,
    /// Limit
    pub limit: Option<usize>,
    /// Order by
    pub order_by: Option<QueryOrderBy>,
}
/// Layout types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    Grid,
    Flex,
    Fixed,
    Custom,
}
/// Metrics metadata
#[derive(Debug, Clone)]
pub struct MetricsMetadata {
    /// Source identifier
    pub source: String,
    /// Collection method
    pub collection_method: String,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Data quality score
    pub quality_score: f64,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}
/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexType {
    #[default]
    BTree,
    Hash,
    Bitmap,
    FullText,
    Custom,
}
/// Tracker configuration
#[derive(Debug, Clone)]
pub struct TrackerConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Collection interval
    pub collection_interval: Duration,
    /// Enabled collectors
    pub enabled_collectors: Vec<String>,
    /// Enabled analyzers
    pub enabled_analyzers: Vec<String>,
    /// Storage configuration
    pub storage_config: StorageConfiguration<T>,
    /// Alert configuration
    pub alert_config: AlertConfiguration<T>,
    /// Dashboard configuration
    pub dashboard_config: DashboardConfiguration<T>,
}
/// Types of metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Counter metric (monotonically increasing)
    Counter,
    /// Gauge metric (can increase/decrease)
    Gauge,
    /// Histogram metric
    Histogram,
    /// Summary metric
    Summary,
    /// Rate metric
    Rate,
    /// Ratio metric
    Ratio,
    /// Custom metric type
    Custom,
}
/// Compression settings
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level
    pub level: u8,
    /// Compression threshold
    pub threshold_bytes: usize,
}
/// Collection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionStrategy {
    /// Continuous collection
    Continuous,
    /// Periodic collection
    Periodic,
    /// Event-driven collection
    EventDriven,
    /// Adaptive collection
    Adaptive,
    /// On-demand collection
    OnDemand,
}
/// Window types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Fixed time window
    Fixed,
    /// Sliding time window
    Sliding,
    /// Tumbling window
    Tumbling,
    /// Session window
    Session,
    /// Custom window
    Custom,
}
/// Alert context information
#[derive(Debug, Clone)]
pub struct AlertContext<T: Float + Debug + Send + Sync + 'static> {
    /// Source system/component
    pub source: String,
    /// Environment information
    pub environment: HashMap<String, String>,
    /// Related metrics
    pub related_metrics: HashMap<String, T>,
    /// Historical context
    pub historical_data: Vec<T>,
    /// Prediction context
    pub predictions: Vec<TrendPrediction<T>>,
}
/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfiguration {
    /// Indexed fields
    pub indexed_fields: Vec<String>,
    /// Index type
    pub index_type: IndexType,
    /// Index refresh interval
    pub refresh_interval: Duration,
}
/// Category status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoryStatus {
    Normal,
    Warning,
    Critical,
    Unknown,
}
/// Threshold adaptation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationMethod {
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Percentile-based
    PercentileBased,
    /// Machine learning
    MachineLearning,
    /// Custom method
    Custom,
}
/// Aggregation window specification
#[derive(Debug, Clone)]
pub struct AggregationWindow {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub size: Duration,
    /// Window overlap
    pub overlap: Duration,
    /// Window alignment
    pub alignment: WindowAlignment,
}
/// Aggregation statistics
#[derive(Debug, Clone)]
pub struct AggregationStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total aggregations performed
    pub total_aggregations: usize,
    /// Average aggregation time
    pub average_aggregation_time: Duration,
    /// Aggregation accuracy
    pub aggregation_accuracy: T,
    /// Cache performance
    pub cache_performance: CacheStatistics<T>,
}
/// Metric collector for specific metrics
#[derive(Debug)]
pub struct MetricCollector<T: Float + Debug + Send + Sync + 'static> {
    /// Collector identifier
    pub collector_id: String,
    /// Metrics being collected
    collected_metrics: Vec<String>,
    /// Collection strategy
    strategy: CollectionStrategy,
    /// Collection frequency
    frequency: Duration,
    /// Data buffer
    buffer: MetricBuffer<T>,
    /// Collection filters
    filters: Vec<Box<dyn MetricFilter<T>>>,
    /// Collection statistics
    stats: CollectionStatistics<T>,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> MetricCollector<T> {
    /// A collector for the named metrics.
    pub fn new(
        collector_id: String,
        collected_metrics: Vec<String>,
        strategy: CollectionStrategy,
        frequency: Duration,
        buffer_capacity: usize,
    ) -> Self {
        Self {
            collector_id,
            collected_metrics,
            strategy,
            frequency,
            buffer: MetricBuffer::new(
                buffer_capacity,
                BufferStrategy::FIFO,
                OverflowHandling::DropOldest,
            ),
            filters: Vec::new(),
            stats: CollectionStatistics {
                total_collected: 0,
                collection_rate: T::zero(),
                average_latency: Duration::ZERO,
                collection_errors: 0,
                buffer_utilization: T::zero(),
                quality_metrics: HashMap::new(),
            },
        }
    }

    /// Record one observation for later collection.
    ///
    /// This crate measures nothing itself; the process being optimized feeds its
    /// own numbers in here.
    pub fn record(&mut self, name: &str, value: T) {
        let stored = self.buffer.push(BufferedMetric {
            name: name.to_string(),
            value,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        });
        if !stored {
            self.stats.collection_errors += 1;
        }
    }

    /// Register a filter. Samples it rejects are discarded at collection time.
    pub fn add_filter(&mut self, filter: Box<dyn MetricFilter<T>>) {
        self.filters.push(filter);
    }

    /// The configured collection strategy.
    pub fn strategy(&self) -> CollectionStrategy {
        self.strategy
    }

    /// The configured collection interval.
    pub fn frequency(&self) -> Duration {
        self.frequency
    }

    /// Cumulative collection statistics.
    pub fn statistics(&self) -> &CollectionStatistics<T> {
        &self.stats
    }

    /// Drain the buffer into per-metric categories.
    ///
    /// Each requested metric becomes its own category holding the mean of the
    /// samples recorded for it since the last call, with the sample count in the
    /// metric's tags. A metric with no samples is omitted rather than reported.
    ///
    /// Until 0.3.2 this ignored the buffer entirely and synthesised a
    /// `MetricValue { value: 0.5, confidence: 0.9, unit: "unit" }` for every
    /// configured name, with `Stable` trends and `trend_confidence: 0.8` -- a
    /// stream of invented numbers that the tracker then stored and alerted on.
    pub fn collect(&mut self) -> Result<HashMap<String, CategoryMetrics<T>>> {
        let drained = self.buffer.drain();
        let kept: Vec<BufferedMetric<T>> = drained
            .into_iter()
            .filter(|metric| self.filters.iter().all(|filter| filter.filter(metric)))
            .collect();
        self.stats.total_collected += kept.len();
        let capacity = T::from(self.buffer.capacity()).unwrap_or_else(T::one);
        self.stats.buffer_utilization = if capacity > T::zero() {
            T::from(kept.len()).unwrap_or_else(T::zero) / capacity
        } else {
            T::zero()
        };

        let mut categories = HashMap::new();
        for metric_name in &self.collected_metrics {
            let samples: Vec<T> = kept
                .iter()
                .filter(|metric| &metric.name == metric_name)
                .map(|metric| metric.value)
                .collect();
            if samples.is_empty() {
                continue;
            }
            let count = T::from(samples.len()).unwrap_or_else(T::one);
            let mean = samples.iter().fold(T::zero(), |acc, &v| acc + v) / count;

            let mut metrics = HashMap::new();
            metrics.insert(
                metric_name.clone(),
                MetricValue {
                    value: mean,
                    value_type: MetricType::Gauge,
                    unit: String::new(),
                    bounds: None,
                    // Every sample was measured, so the value is exact for the
                    // window it covers.
                    confidence: T::one(),
                    tags: vec![format!("samples={}", samples.len())],
                },
            );
            categories.insert(
                metric_name.clone(),
                CategoryMetrics {
                    metrics,
                    weight: T::one(),
                    status: CategoryStatus::Normal,
                    trends: CategoryTrends {
                        short_term: trend_of(&samples),
                        long_term: trend_of(&samples),
                        trend_strength: trend_strength(&samples),
                        // No model backs a prediction here, so no confidence is
                        // claimed for one.
                        trend_confidence: T::zero(),
                        predictions: Vec::new(),
                    },
                },
            );
        }
        Ok(categories)
    }
}

/// Direction of a sample series: compares the mean of the second half against
/// the first. Fewer than two samples cannot show a direction.
fn trend_of<T: Float + Debug + Send + Sync + 'static>(samples: &[T]) -> TrendDirection {
    if samples.len() < 2 {
        return TrendDirection::Stable;
    }
    let split = samples.len() / 2;
    let mean = |slice: &[T]| -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().filter_map(|v| v.to_f64()).sum::<f64>() / slice.len() as f64
    };
    let first = mean(&samples[..split]);
    let second = mean(&samples[split..]);
    let scale = first.abs().max(second.abs()).max(f64::EPSILON);
    let delta = (second - first) / scale;
    // "Improving" and "Degrading" name a direction of *value*, not of
    // desirability -- this layer has no notion of which way is good.
    if delta > 0.01 {
        TrendDirection::Improving
    } else if delta < -0.01 {
        TrendDirection::Degrading
    } else {
        TrendDirection::Stable
    }
}

/// Magnitude of the trend `trend_of` reports, in `[0, 1]`.
fn trend_strength<T: Float + Debug + Send + Sync + 'static>(samples: &[T]) -> T {
    if samples.len() < 2 {
        return T::zero();
    }
    let split = samples.len() / 2;
    let mean = |slice: &[T]| -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().filter_map(|v| v.to_f64()).sum::<f64>() / slice.len() as f64
    };
    let first = mean(&samples[..split]);
    let second = mean(&samples[split..]);
    let scale = first.abs().max(second.abs()).max(f64::EPSILON);
    let strength = ((second - first) / scale).abs().min(1.0);
    T::from(strength).unwrap_or_else(T::zero)
}
/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfiguration {
    /// Number of columns
    pub columns: usize,
    /// Number of rows
    pub rows: usize,
    /// Cell padding
    pub padding: usize,
    /// Cell margin
    pub margin: usize,
}
/// Notification settings
#[derive(Debug, Clone)]
pub struct NotificationSettings {
    /// Default channels
    pub default_channels: Vec<String>,
    /// Channel configurations
    pub channel_configs: HashMap<String, HashMap<String, String>>,
    /// Notification templates
    pub templates: HashMap<String, String>,
}
/// Metrics quality indicators
#[derive(Debug, Clone)]
pub struct MetricsQuality<T: Float + Debug + Send + Sync + 'static> {
    /// Completeness score
    pub completeness: T,
    /// Accuracy score
    pub accuracy: T,
    /// Timeliness score
    pub timeliness: T,
    /// Consistency score
    pub consistency: T,
    /// Overall quality score
    pub overall_quality: T,
}
/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}
/// Threshold adaptation settings
#[derive(Debug, Clone)]
pub struct ThresholdAdaptation<T: Float + Debug + Send + Sync + 'static> {
    /// Enable adaptation
    pub enabled: bool,
    /// Adaptation method
    pub method: AdaptationMethod,
    /// Adaptation rate
    pub rate: T,
    /// Adaptation constraints
    pub constraints: AdaptationConstraints<T>,
}
/// Aggregation rule
#[derive(Debug, Clone)]
pub struct AggregationRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule identifier
    pub rule_id: String,
    /// Grouping criteria
    pub grouping: AlertGrouping,
    /// Aggregation function
    pub function: AlertAggregationFunction,
    /// Minimum count for aggregation
    pub min_count: usize,
    /// Custom parameters
    pub parameters: HashMap<String, T>,
}
/// Performance tracker for optimization processes
#[derive(Debug)]
pub struct PerformanceTracker<T: Float + Debug + Send + Sync + 'static> {
    /// Active metric collectors
    metric_collectors: HashMap<String, MetricCollector<T>>,
    /// Alert manager
    alert_manager: AlertManager<T>,
    /// Performance analyzers
    analyzers: Vec<Box<dyn PerformanceAnalyzer<T>>>,
    /// Metric storage
    metric_storage: MetricStorage<T>,
    /// Tracker configuration
    config: TrackerConfiguration<T>,
    /// Tracker statistics
    stats: TrackerStatistics<T>,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> PerformanceTracker<T> {
    /// Create new performance tracker
    pub fn new(config: TrackerConfiguration<T>) -> Result<Self> {
        Ok(Self {
            metric_collectors: HashMap::new(),
            alert_manager: AlertManager::new()?,
            analyzers: Vec::new(),
            metric_storage: MetricStorage::new(config.storage_config.clone())?,
            config,
            stats: TrackerStatistics::default(),
        })
    }
    /// The configuration this tracker runs under.
    pub fn config(&self) -> &TrackerConfiguration<T> {
        &self.config
    }

    /// Mutable access to the alert manager, so rules can be registered.
    pub fn alert_manager_mut(&mut self) -> &mut AlertManager<T> {
        &mut self.alert_manager
    }

    /// Register a performance analyzer, run by [`Self::collect_metrics`].
    ///
    /// `analyzers` was a `Vec` nothing could push to and nothing read, so no
    /// analysis ever ran.
    pub fn add_analyzer(&mut self, analyzer: Box<dyn PerformanceAnalyzer<T>>) {
        self.analyzers.push(analyzer);
    }

    /// Add metric collector
    pub fn add_collector(&mut self, collector: MetricCollector<T>) -> Result<()> {
        self.metric_collectors
            .insert(collector.collector_id.clone(), collector);
        Ok(())
    }
    /// Collect metrics
    pub fn collect_metrics(&mut self) -> Result<PerformanceMetrics<T>> {
        let mut categories = HashMap::new();
        for collector in self.metric_collectors.values_mut() {
            let metrics = collector.collect()?;
            for (category, category_metrics) in metrics {
                categories.insert(category, category_metrics);
            }
        }
        let metrics = PerformanceMetrics {
            categories,
            timestamp: SystemTime::now(),
            // The window a snapshot covers is the collectors' configured
            // cadence, not a literal one second.
            interval: self
                .metric_collectors
                .values()
                .map(|collector| collector.frequency())
                .max()
                .unwrap_or(self.config.collection_interval),
            metadata: MetricsMetadata {
                source: "PerformanceTracker".to_string(),
                collection_method: "automatic".to_string(),
                sampling_rate: 1.0,
                quality_score: 0.95,
                custom: HashMap::new(),
            },
            quality: MetricsQuality {
                completeness: T::from(0.95).unwrap_or_else(|| T::zero()),
                accuracy: T::from(0.9).unwrap_or_else(|| T::zero()),
                timeliness: T::from(0.98).unwrap_or_else(|| T::zero()),
                consistency: T::from(0.92).unwrap_or_else(|| T::zero()),
                overall_quality: T::from(0.94).unwrap_or_else(|| T::zero()),
            },
        };
        self.metric_storage.store(&metrics)?;
        for analyzer in &mut self.analyzers {
            // Analysis results are surfaced through the analyzer itself; a
            // failure here is a real error, not something to swallow.
            let _ = analyzer.analyze(&metrics)?;
        }
        self.stats.total_metrics_collected += 1;
        Ok(metrics)
    }
    /// Check for alerts
    pub fn check_alerts(
        &mut self,
        metrics: &PerformanceMetrics<T>,
    ) -> Result<Vec<PerformanceAlert<T>>> {
        self.alert_manager.check_alerts(metrics)
    }
    /// Get tracker statistics
    pub fn get_statistics(&self) -> &TrackerStatistics<T> {
        &self.stats
    }
}
/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Retention period
    pub retention_period: Duration,
    /// Compression settings
    pub compression: CompressionSettings,
    /// Partitioning strategy
    pub partitioning: PartitioningStrategy,
    /// Index configuration
    pub indexing: IndexConfiguration,
    /// Custom storage parameters
    pub custom_params: HashMap<String, T>,
}
/// Trend prediction
#[derive(Debug, Clone)]
pub struct TrendPrediction<T: Float + Debug + Send + Sync + 'static> {
    /// Prediction horizon
    pub horizon: Duration,
    /// Predicted value
    pub predicted_value: T,
    /// Prediction confidence
    pub confidence: T,
    /// Prediction bounds
    pub bounds: (T, T),
}
/// Supporting evidence for insights
#[derive(Debug, Clone)]
pub struct Evidence<T: Float + Debug + Send + Sync + 'static> {
    /// Evidence type
    pub evidence_type: String,
    /// Evidence data
    pub data: Vec<T>,
    /// Evidence description
    pub description: String,
    /// Evidence weight
    pub weight: T,
}
/// Dashboard widget
#[derive(Debug)]
pub struct DashboardWidget<T: Float + Debug + Send + Sync + 'static> {
    /// Widget identifier
    pub widget_id: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Widget data source
    pub data_source: String,
    /// Widget configuration
    pub configuration: WidgetConfiguration<T>,
    /// Widget layout information
    pub layout: WidgetLayout,
}
/// Alert aggregation settings
#[derive(Debug, Clone)]
pub struct AlertAggregationSettings<T: Float + Debug + Send + Sync + 'static> {
    /// Enable aggregation
    pub enabled: bool,
    /// Aggregation window
    pub window: Duration,
    /// Custom aggregation parameters
    pub custom_params: HashMap<String, T>,
}
/// Tracker statistics
#[derive(Debug, Clone)]
pub struct TrackerStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total metrics collected
    pub total_metrics_collected: usize,
    /// Collection rate (metrics/second)
    pub collection_rate: T,
    /// Total alerts generated
    pub total_alerts_generated: usize,
    /// Alert rate (alerts/hour)
    pub alert_rate: T,
    /// Average processing latency
    pub average_processing_latency: Duration,
    /// System utilization
    pub system_utilization: T,
}
/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEvictionPolicy {
    /// Least recently used
    LRU,
    /// Least frequently used
    LFU,
    /// First-in-first-out
    FIFO,
    /// Time-to-live based
    TTL,
    /// Custom policy
    Custom,
}
/// Metric bounds for validation
#[derive(Debug, Clone)]
pub struct MetricBounds<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum value
    pub min: Option<T>,
    /// Maximum value
    pub max: Option<T>,
    /// Warning thresholds
    pub warning_bounds: Option<(T, T)>,
    /// Critical thresholds
    pub critical_bounds: Option<(T, T)>,
}
/// Buffered metric
#[derive(Debug, Clone)]
pub struct BufferedMetric<T: Float + Debug + Send + Sync + 'static> {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: T,
    /// Collection timestamp
    pub timestamp: SystemTime,
    /// Metric metadata
    pub metadata: HashMap<String, String>,
}
/// Storage backend statistics
#[derive(Debug, Clone)]
pub struct StorageBackendStats {
    /// Total metrics stored
    pub total_metrics: usize,
    /// Storage size (bytes)
    pub storage_size_bytes: usize,
    /// Average query time
    pub average_query_time: Duration,
    /// Storage utilization
    pub utilization_percentage: f64,
}
/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
    Escalated,
}
/// Logical operators for composite conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
    Xor,
}
/// Aggregated alert
#[derive(Debug, Clone)]
pub struct AggregatedAlert<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregated alert identifier
    pub aggregated_id: String,
    /// Individual alerts
    pub alerts: Vec<String>,
    /// Aggregation summary
    pub summary: AggregationSummary<T>,
    /// Aggregation timestamp
    pub timestamp: SystemTime,
    /// Aggregation metadata
    pub metadata: HashMap<String, String>,
}
/// Query ordering
#[derive(Debug, Clone)]
pub struct QueryOrderBy {
    /// Field to order by
    pub field: String,
    /// Order direction
    pub direction: OrderDirection,
}
/// Dashboard layout
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    /// Layout type
    pub layout_type: LayoutType,
    /// Grid configuration
    pub grid: GridConfiguration,
    /// Responsive settings
    pub responsive: bool,
}
/// Widget layout information
#[derive(Debug, Clone)]
pub struct WidgetLayout {
    /// X position
    pub x: usize,
    /// Y position
    pub y: usize,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
}
/// Alert grouping criteria
#[derive(Debug, Clone)]
pub enum AlertGrouping {
    /// Group by metric
    ByMetric,
    /// Group by severity
    BySeverity,
    /// Group by source
    BySource,
    /// Group by rule
    ByRule,
    /// Custom grouping
    Custom(String),
}
/// Category trends
#[derive(Debug, Clone)]
pub struct CategoryTrends<T: Float + Debug + Send + Sync + 'static> {
    /// Short-term trend
    pub short_term: TrendDirection,
    /// Long-term trend
    pub long_term: TrendDirection,
    /// Trend strength
    pub trend_strength: T,
    /// Trend confidence
    pub trend_confidence: T,
    /// Trend predictions
    pub predictions: Vec<TrendPrediction<T>>,
}
/// Metric storage for persistent storage
#[derive(Debug)]
pub struct MetricStorage<T: Float + Debug + Send + Sync + 'static> {
    /// Storage backend
    backend: Box<dyn StorageBackend<T>>,
    /// Storage configuration
    config: StorageConfiguration<T>,
    /// Storage statistics
    stats: StorageStatistics<T>,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> MetricStorage<T> {
    pub fn new(_config: StorageConfiguration<T>) -> Result<Self> {
        Ok(Self {
            backend: Box::new(InMemoryStorageBackend::new()),
            config: _config,
            stats: StorageStatistics::default(),
        })
    }
    /// The storage configuration in force.
    pub fn config(&self) -> &StorageConfiguration<T> {
        &self.config
    }

    /// Cumulative storage statistics.
    pub fn statistics(&self) -> &StorageStatistics<T> {
        &self.stats
    }

    /// Persist a metrics snapshot and record it in the statistics.
    ///
    /// `stats` was constructed and never updated, so `total_stored` stayed at
    /// zero no matter how much had been written.
    pub fn store(&mut self, metrics: &PerformanceMetrics<T>) -> Result<()> {
        self.backend.store(metrics)?;
        // Rate is per snapshot rather than per second here: the storage layer is
        // not given a clock, and inventing an elapsed time would fabricate the
        // denominator.
        let stored: usize = metrics
            .categories
            .values()
            .map(|category| category.metrics.len())
            .sum();
        self.stats.storage_rate = self.stats.storage_rate + T::from(stored).unwrap_or_else(T::zero);
        Ok(())
    }
}
/// Time range specification
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time
    pub start: SystemTime,
    /// End time
    pub end: SystemTime,
}
/// Partitioning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitioningStrategy {
    /// Time-based partitioning
    #[default]
    TimeBased,
    /// Metric-based partitioning
    MetricBased,
    /// Hash-based partitioning
    HashBased,
    /// Custom partitioning
    Custom,
}
/// Aggregation summary
#[derive(Debug, Clone)]
pub struct AggregationSummary<T: Float + Debug + Send + Sync + 'static> {
    /// Total alert count
    pub total_count: usize,
    /// Alert count by severity
    pub count_by_severity: HashMap<AlertSeverity, usize>,
    /// Average metric value
    pub average_metric_value: Option<T>,
    /// Time span
    pub time_span: Duration,
    /// Most common source
    pub most_common_source: Option<String>,
}
/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Default alert rules
    pub default_rules: Vec<AlertRule<T>>,
    /// Notification settings
    pub notification_settings: NotificationSettings,
    /// Alert aggregation settings
    pub aggregation_settings: AlertAggregationSettings<T>,
}
/// Anomaly detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyDetectionMethod {
    StatisticalOutlier,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
    Custom,
}
/// Aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Real-time aggregation
    RealTime,
    /// Batch aggregation
    Batch,
    /// Streaming aggregation
    Streaming,
    /// Hybrid aggregation
    Hybrid,
}
/// Types of insights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightType {
    PerformanceBottleneck,
    ResourceContention,
    EfficiencyOpportunity,
    QualityIssue,
    TrendChange,
    Anomaly,
    Custom,
}
#[derive(Debug)]
pub struct InMemoryStorageBackend<T: Float + Debug + Send + Sync + 'static> {
    pub(super) storage: Vec<(SystemTime, String)>,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> Default for InMemoryStorageBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> InMemoryStorageBackend<T> {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}
/// Burst limits for alert frequency
#[derive(Debug, Clone)]
pub struct BurstLimits {
    /// Maximum burst size
    pub max_burst_size: usize,
    /// Burst window
    pub burst_window: Duration,
    /// Recovery time
    pub recovery_time: Duration,
}
/// Performance metrics container
#[derive(Debug, Clone)]
pub struct PerformanceMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Metrics by category
    pub categories: HashMap<String, CategoryMetrics<T>>,
    /// Timestamp of measurement
    pub timestamp: SystemTime,
    /// Measurement interval
    pub interval: Duration,
    /// Metrics metadata
    pub metadata: MetricsMetadata,
    /// Quality indicators
    pub quality: MetricsQuality<T>,
}
/// Hysteresis values for threshold stability
#[derive(Debug, Clone)]
pub struct HysteresisValues<T: Float + Debug + Send + Sync + 'static> {
    /// Upper hysteresis
    pub upper: T,
    /// Lower hysteresis
    pub lower: T,
}
/// Order directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    Ascending,
    Descending,
}
/// Alert aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertAggregationStrategy {
    /// No aggregation
    None,
    /// Count-based aggregation
    Count,
    /// Time-based aggregation
    TimeBased,
    /// Severity-based aggregation
    SeverityBased,
    /// Custom aggregation
    Custom,
}
/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total alerts generated
    pub total_alerts: usize,
    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    /// Alert rate (alerts per hour)
    pub alert_rate: T,
    /// Average alert resolution time
    pub average_resolution_time: Duration,
    /// False positive rate
    pub false_positive_rate: T,
    /// True positive rate
    pub true_positive_rate: T,
}
/// Frequency limits for alerts
#[derive(Debug, Clone)]
pub struct FrequencyLimits {
    /// Maximum alerts per minute
    pub max_per_minute: Option<usize>,
    /// Maximum alerts per hour
    pub max_per_hour: Option<usize>,
    /// Maximum alerts per day
    pub max_per_day: Option<usize>,
    /// Cooldown period
    pub cooldown_period: Option<Duration>,
    /// Burst limits
    pub burst_limits: Option<BurstLimits>,
}
/// Alert status change information
#[derive(Debug, Clone)]
pub struct AlertStatusChange {
    /// Previous status
    pub from_status: AlertStatus,
    /// New status
    pub to_status: AlertStatus,
    /// Change duration
    pub duration: Duration,
}
/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Total metrics collected
    pub total_collected: usize,
    /// Collection rate (metrics/second)
    pub collection_rate: T,
    /// Average collection latency
    pub average_latency: Duration,
    /// Collection errors
    pub collection_errors: usize,
    /// Buffer utilization
    pub buffer_utilization: T,
    /// Data quality metrics
    pub quality_metrics: HashMap<String, T>,
}
/// Buffer strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferStrategy {
    /// First-in-first-out
    FIFO,
    /// Last-in-first-out
    LIFO,
    /// Priority-based
    Priority,
    /// Custom strategy
    Custom,
}
/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Hit ratio
    pub hit_ratio: T,
    /// Cache utilization
    pub utilization: T,
    /// Average access time
    pub average_access_time: Duration,
}
/// Widget configuration
#[derive(Debug, Clone)]
pub struct WidgetConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Widget title
    pub title: String,
    /// Display options
    pub display_options: HashMap<String, String>,
    /// Refresh interval
    pub refresh_interval: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, T>,
}
