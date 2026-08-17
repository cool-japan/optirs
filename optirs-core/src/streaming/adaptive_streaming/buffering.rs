// Adaptive buffering strategies for streaming optimization
//
// This module provides sophisticated buffer management including adaptive sizing,
// quality-based filtering, priority queuing, and intelligent data retention
// strategies for streaming optimization scenarios.

use super::config::*;
use super::optimizer::{Adaptation, AdaptationPriority, AdaptationType, StreamingDataPoint};
use super::performance::PerformanceTracker;

use crate::utils::{scalar_or, try_scalar_str};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Adaptive buffer for managing streaming data with quality-based retention
pub struct AdaptiveBuffer<A: Float + Send + Sync> {
    /// Buffer configuration
    config: BufferConfig,
    /// Main data buffer with priority queue
    buffer: BinaryHeap<PrioritizedDataPoint<A>>,
    /// Secondary buffer for low-quality data
    secondary_buffer: VecDeque<StreamingDataPoint<A>>,
    /// Buffer quality metrics
    quality_metrics: BufferQualityMetrics<A>,
    /// Buffer sizing strategy
    sizing_strategy: BufferSizingStrategy<A>,
    /// Buffer statistics
    statistics: BufferStatistics<A>,
    /// Last processing timestamp
    last_processing: Instant,
    /// Size change tracking
    size_change_log: VecDeque<SizeChangeEvent>,
    /// Running per-feature statistics backing the relevance score.
    feature_statistics: RunningFeatureStatistics<A>,
}

/// Per-item processing time assumed only until a real latency or throughput
/// measurement exists.
const DEFAULT_EXPECTED_PROCESSING_TIME: Duration = Duration::from_millis(100);

/// Smoothing factor for the processing-latency moving average.
const LATENCY_SMOOTHING: f64 = 0.1;

/// Welford accumulators for the buffer's per-feature distribution.
#[derive(Debug, Clone)]
struct RunningFeatureStatistics<A: Float + Send + Sync> {
    /// Running per-coordinate mean.
    means: Vec<A>,
    /// Running per-coordinate sum of squared deviations.
    m2: Vec<A>,
    /// Number of observations folded in.
    sample_count: usize,
}

impl<A: Float + Send + Sync> Default for RunningFeatureStatistics<A> {
    fn default() -> Self {
        Self {
            means: Vec::new(),
            m2: Vec::new(),
            sample_count: 0,
        }
    }
}

/// Data point with priority information for buffering
#[derive(Debug, Clone)]
pub struct PrioritizedDataPoint<A: Float + Send + Sync> {
    /// The actual data point
    pub data_point: StreamingDataPoint<A>,
    /// Priority score (higher = more important)
    pub priority_score: A,
    /// Buffer insertion timestamp
    pub buffer_timestamp: Instant,
    /// Expected processing time
    pub expected_processing_time: Duration,
    /// Data freshness score
    pub freshness_score: A,
    /// Relevance score for current model
    pub relevance_score: A,
}

/// Buffer quality metrics for adaptive management
#[derive(Debug, Clone)]
pub struct BufferQualityMetrics<A: Float + Send + Sync> {
    /// Average quality score of buffered data
    pub average_quality: A,
    /// Quality variance
    pub quality_variance: A,
    /// Minimum quality in buffer
    pub min_quality: A,
    /// Maximum quality in buffer
    pub max_quality: A,
    /// Data freshness distribution
    pub freshness_distribution: Vec<A>,
    /// Priority distribution
    pub priority_distribution: Vec<A>,
    /// Quality trend over time
    pub quality_trend: QualityTrend<A>,
}

/// Quality trend analysis
#[derive(Debug, Clone)]
pub struct QualityTrend<A: Float + Send + Sync> {
    /// Recent quality changes
    pub recent_changes: VecDeque<A>,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend magnitude
    pub trend_magnitude: A,
    /// Trend confidence
    pub confidence: A,
}

/// Trend direction for quality analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Quality improving
    Improving,
    /// Quality degrading
    Degrading,
    /// Quality stable
    Stable,
    /// Quality oscillating
    Oscillating,
}

/// Buffer sizing strategy implementation
pub struct BufferSizingStrategy<A: Float + Send + Sync> {
    /// The configured sizing strategy, which decides *how* `target_size`
    /// moves (and, for `Fixed`, that it does not move at all).
    strategy_type: BufferSizeStrategy,
    /// Initial size, the base for `Linear` steps.
    initial_size: usize,
    /// Target size
    target_size: usize,
    /// Size adjustment parameters
    adjustment_params: SizeAdjustmentParams<A>,
}

/// Parameters for size adjustment
#[derive(Debug, Clone)]
pub struct SizeAdjustmentParams<A: Float + Send + Sync> {
    /// Growth rate for increasing buffer size
    pub growth_rate: A,
    /// Shrinkage rate for decreasing buffer size
    pub shrinkage_rate: A,
    /// Stability threshold (minimum change for adjustment)
    pub stability_threshold: A,
    /// Performance sensitivity
    pub performance_sensitivity: A,
    /// Quality sensitivity
    pub quality_sensitivity: A,
    /// Memory pressure sensitivity
    pub memory_sensitivity: A,
}

/// Performance feedback for buffer sizing
#[derive(Debug, Clone)]
pub struct SizingPerformanceFeedback<A: Float + Send + Sync> {
    /// Buffer size when feedback was recorded
    pub buffer_size: usize,
    /// Processing latency
    pub processing_latency: Duration,
    /// Throughput (items per second)
    pub throughput: A,
    /// Quality score achieved
    pub quality_score: A,
    /// Memory usage
    pub memory_usage: usize,
    /// Timestamp of feedback
    pub timestamp: Instant,
}

/// Buffer sizing event
#[derive(Debug, Clone)]
pub struct SizingEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Old buffer size
    pub old_size: usize,
    /// New buffer size
    pub new_size: usize,
    /// Reason for size change
    pub reason: SizingReason,
    /// Performance impact
    pub performance_impact: Option<f64>,
}

/// Reasons for buffer size changes
#[derive(Debug, Clone)]
pub enum SizingReason {
    /// Performance optimization
    PerformanceOptimization,
    /// Quality improvement
    QualityImprovement,
    /// Memory pressure
    MemoryPressure,
    /// Latency requirements
    LatencyRequirement,
    /// Throughput optimization
    ThroughputOptimization,
    /// Manual adjustment
    Manual,
    /// Configuration change
    Configuration,
}

/// Data retention strategies
#[derive(Debug, Clone)]
pub enum RetentionStrategy {
    /// First In, First Out
    FIFO,
    /// Last In, First Out
    LIFO,
    /// Least Recently Used
    LRU,
    /// Priority-based retention
    Priority,
    /// Quality-based retention
    Quality,
    /// Age-based retention
    Age,
    /// Hybrid retention combining multiple factors
    Hybrid,
    /// Adaptive retention based on performance
    Adaptive,
}

/// Age-based retention configuration
#[derive(Debug, Clone)]
pub struct AgeBasedRetention {
    /// Maximum age for data retention
    pub max_age: Duration,
    /// Soft age limit (start considering for removal)
    pub soft_age_limit: Duration,
    /// Age weight in retention scoring
    pub age_weight: f64,
    /// Enable adaptive age limits
    pub adaptive_limits: bool,
}

/// Quality-based retention configuration
#[derive(Debug, Clone)]
pub struct QualityBasedRetention<A: Float + Send + Sync> {
    /// Minimum quality threshold
    pub min_quality_threshold: A,
    /// Quality weight in retention scoring
    pub quality_weight: A,
    /// Enable adaptive quality thresholds
    pub adaptive_thresholds: bool,
    /// Quality distribution targets
    pub quality_targets: QualityDistributionTargets<A>,
}

/// Target quality distribution for buffer content
#[derive(Debug, Clone)]
pub struct QualityDistributionTargets<A: Float + Send + Sync> {
    /// Target percentage of high-quality data
    pub high_quality_target: A,
    /// Target percentage of medium-quality data
    pub medium_quality_target: A,
    /// Target percentage of low-quality data
    pub low_quality_target: A,
    /// Quality boundaries
    pub high_quality_threshold: A,
    pub medium_quality_threshold: A,
}

/// Relevance-based retention configuration
#[derive(Debug, Clone)]
pub struct RelevanceBasedRetention<A: Float + Send + Sync> {
    /// Relevance calculation method
    pub relevance_method: RelevanceMethod,
    /// Relevance weight in retention scoring
    pub relevance_weight: A,
    /// Enable temporal relevance decay
    pub temporal_decay: bool,
    /// Relevance decay rate
    pub decay_rate: A,
}

/// Methods for calculating data relevance
#[derive(Debug, Clone)]
pub enum RelevanceMethod {
    /// Distance-based relevance
    Distance,
    /// Similarity-based relevance
    Similarity,
    /// Feature importance-based relevance
    FeatureImportance,
    /// Model uncertainty-based relevance
    Uncertainty,
    /// Diversity-based relevance
    Diversity,
    /// Custom relevance function
    Custom(String),
}

/// Weights for different retention factors
#[derive(Debug, Clone)]
pub struct RetentionWeights<A: Float + Send + Sync> {
    /// Age weight
    pub age_weight: A,
    /// Quality weight
    pub quality_weight: A,
    /// Relevance weight
    pub relevance_weight: A,
    /// Priority weight
    pub priority_weight: A,
    /// Freshness weight
    pub freshness_weight: A,
    /// Diversity weight
    pub diversity_weight: A,
}

/// Retention score for a data point
#[derive(Debug, Clone)]
pub struct RetentionScore<A: Float + Send + Sync> {
    /// Overall retention score
    pub overall_score: A,
    /// Individual component scores
    pub component_scores: HashMap<String, A>,
    /// Retention decision
    pub should_retain: bool,
    /// Confidence in decision
    pub confidence: A,
    /// Scoring timestamp
    pub timestamp: Instant,
}

/// Performance feedback for retention decisions
#[derive(Debug, Clone)]
pub struct RetentionPerformanceFeedback<A: Float + Send + Sync> {
    /// Number of items retained
    pub items_retained: usize,
    /// Number of items discarded
    pub items_discarded: usize,
    /// Quality of retained items
    pub retained_quality: A,
    /// Quality of discarded items
    pub discarded_quality: A,
    /// Performance impact
    pub performance_impact: A,
    /// Feedback timestamp
    pub timestamp: Instant,
}

/// Buffer statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct BufferStatistics<A: Float + Send + Sync> {
    /// Total items processed
    pub total_items_processed: u64,
    /// Total items discarded
    pub total_items_discarded: u64,
    /// Average buffer utilization
    pub avg_buffer_utilization: A,
    /// Peak buffer utilization
    pub peak_buffer_utilization: A,
    /// Average processing latency
    pub avg_processing_latency: Duration,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics<A>,
    /// Quality statistics
    pub quality_stats: QualityStatistics<A>,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
}

/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStatistics<A: Float + Send + Sync> {
    /// Current throughput (items per second)
    pub current_throughput: A,
    /// Average throughput
    pub avg_throughput: A,
    /// Peak throughput
    pub peak_throughput: A,
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Throughput stability
    pub stability: A,
}

/// Quality statistics for buffer content
#[derive(Debug, Clone)]
pub struct QualityStatistics<A: Float + Send + Sync> {
    /// Current average quality
    pub current_avg_quality: A,
    /// Historical average quality
    pub historical_avg_quality: A,
    /// Quality improvement rate
    pub quality_improvement_rate: A,
    /// Quality distribution
    pub quality_distribution: HashMap<String, A>,
    /// Quality prediction
    pub predicted_quality: Option<A>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage in bytes
    pub current_usage_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Average memory usage in bytes
    pub avg_usage_bytes: usize,
    /// Memory efficiency (useful data / total memory)
    pub memory_efficiency: f64,
    /// Memory fragmentation
    pub fragmentation: f64,
}

/// Size change tracking event
#[derive(Debug, Clone)]
pub struct SizeChangeEvent {
    /// Change timestamp
    pub timestamp: Instant,
    /// Size before change
    pub old_size: usize,
    /// Size after change
    pub new_size: usize,
    /// Change magnitude
    pub change_magnitude: i32,
    /// Reason for change
    pub reason: String,
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum + std::fmt::Debug>
    AdaptiveBuffer<A>
{
    /// Creates a new adaptive buffer
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let buffer_config = config.buffer_config.clone();

        let quality_metrics = BufferQualityMetrics {
            average_quality: A::zero(),
            quality_variance: A::zero(),
            min_quality: A::one(),
            max_quality: A::zero(),
            freshness_distribution: Vec::new(),
            priority_distribution: Vec::new(),
            quality_trend: QualityTrend {
                recent_changes: VecDeque::with_capacity(50),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: A::zero(),
                confidence: A::zero(),
            },
        };

        let sizing_strategy = BufferSizingStrategy::new(
            buffer_config.size_strategy.clone(),
            buffer_config.initial_size,
        );

        let statistics = BufferStatistics {
            total_items_processed: 0,
            total_items_discarded: 0,
            avg_buffer_utilization: A::zero(),
            peak_buffer_utilization: A::zero(),
            avg_processing_latency: Duration::ZERO,
            throughput_stats: ThroughputStatistics {
                current_throughput: A::zero(),
                avg_throughput: A::zero(),
                peak_throughput: A::zero(),
                throughput_trend: TrendDirection::Stable,
                stability: A::zero(),
            },
            quality_stats: QualityStatistics {
                current_avg_quality: A::zero(),
                historical_avg_quality: A::zero(),
                quality_improvement_rate: A::zero(),
                quality_distribution: HashMap::new(),
                predicted_quality: None,
            },
            memory_stats: MemoryStatistics {
                current_usage_bytes: 0,
                peak_usage_bytes: 0,
                avg_usage_bytes: 0,
                memory_efficiency: 0.0,
                fragmentation: 0.0,
            },
        };

        Ok(Self {
            config: buffer_config,
            buffer: BinaryHeap::new(),
            secondary_buffer: VecDeque::new(),
            quality_metrics,
            sizing_strategy,
            statistics,
            last_processing: Instant::now(),
            size_change_log: VecDeque::with_capacity(100),
            feature_statistics: RunningFeatureStatistics::default(),
        })
    }

    /// Adds a batch of data points to the buffer
    pub fn add_batch(&mut self, batch: Vec<StreamingDataPoint<A>>) -> Result<(), String> {
        for data_point in batch {
            self.add_single_point(data_point)?;
        }

        // Update quality metrics after batch addition
        self.update_quality_metrics()?;

        // Check if buffer needs resizing
        self.check_buffer_resizing()?;

        // Apply retention policy if buffer is too large
        if self.current_size() > self.sizing_strategy.target_size {
            self.apply_retention_policy()?;
        }

        Ok(())
    }

    /// Adds a single data point to the buffer
    fn add_single_point(&mut self, data_point: StreamingDataPoint<A>) -> Result<(), String> {
        // Calculate priority score for the data point
        let priority_score = self.calculate_priority_score(&data_point)?;

        // Calculate freshness and relevance scores. Relevance is measured
        // against the statistics of the points seen *before* this one, so the
        // point cannot make itself look typical.
        let freshness_score = self.calculate_freshness_score(&data_point);
        let relevance_score = self.calculate_relevance_score(&data_point)?;
        self.update_feature_statistics(&data_point);

        let prioritized_point = PrioritizedDataPoint {
            data_point,
            priority_score,
            buffer_timestamp: Instant::now(),
            // Real estimate from the measured average processing latency,
            // falling back to the observed throughput when no latency has been
            // recorded yet.
            expected_processing_time: self.estimated_processing_time(),
            freshness_score,
            relevance_score,
        };

        // Add to appropriate buffer based on quality
        if priority_score >= try_scalar_str::<A, _>(self.config.quality_threshold)? {
            self.buffer.push(prioritized_point);
        } else {
            // Add to secondary buffer for potential later processing
            self.secondary_buffer
                .push_back(prioritized_point.data_point);
        }

        // Update statistics
        self.statistics.total_items_processed += 1;

        Ok(())
    }

    /// Calculates priority score for a data point
    fn calculate_priority_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        let mut score = data_point.quality_score;

        // Adjust score based on recency
        let age = data_point.timestamp.elapsed().as_secs_f64();
        let recency_bonus = try_scalar_str::<A, _>(1.0 / (1.0 + age / 3600.0))?; // Hour-based decay
        score = score + recency_bonus * try_scalar_str::<A, _>(0.1)?;

        // Adjust score based on feature variance (novelty)
        let novelty_score = self.calculate_novelty_score(data_point)?;
        score = score + novelty_score * try_scalar_str::<A, _>(0.2)?;

        Ok(score)
    }

    /// Calculates novelty score based on feature variance
    fn calculate_novelty_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        // Simple novelty calculation based on distance from recent data
        if self.buffer.is_empty() {
            return try_scalar_str::<A, _>(0.5); // Medium novelty for first data
        }

        // Calculate average distance from recent buffer content
        let recent_points: Vec<_> = self.buffer.iter().take(10).collect();
        if recent_points.is_empty() {
            return try_scalar_str::<A, _>(0.5);
        }

        let mut total_distance = A::zero();
        for recent_point in &recent_points {
            let distance = self.calculate_feature_distance(
                &data_point.features,
                &recent_point.data_point.features,
            )?;
            total_distance = total_distance + distance;
        }

        let avg_distance = total_distance / try_scalar_str::<A, _>(recent_points.len())?;

        // Normalize to 0-1 range
        let normalized_novelty = avg_distance / (avg_distance + A::one());
        Ok(normalized_novelty)
    }

    /// Calculates distance between feature vectors
    fn calculate_feature_distance(
        &self,
        features1: &scirs2_core::ndarray::Array1<A>,
        features2: &scirs2_core::ndarray::Array1<A>,
    ) -> Result<A, String> {
        if features1.len() != features2.len() {
            return Err("Feature vectors have different lengths".to_string());
        }

        let mut distance = A::zero();
        for (f1, f2) in features1.iter().zip(features2.iter()) {
            let diff = *f1 - *f2;
            distance = distance + diff * diff;
        }

        Ok(distance.sqrt())
    }

    /// Calculates freshness score based on data age
    fn calculate_freshness_score(&self, data_point: &StreamingDataPoint<A>) -> A {
        let age_seconds = data_point.timestamp.elapsed().as_secs_f64();
        let max_age = 3600.0; // 1 hour maximum age

        let freshness = (max_age - age_seconds.min(max_age)) / max_age;
        scalar_or(freshness.max(0.0), A::zero())
    }

    /// Calculates how relevant a data point is to what the buffer currently
    /// holds.
    ///
    /// B1: this used to `return Ok(0.7)` for every point, so relevance
    /// contributed a constant to the priority score and therefore had no effect
    /// on ordering whatsoever.
    ///
    /// Relevance is now a real, bounded function of two measurable properties:
    ///
    /// - **Typicality**: the Mahalanobis-style standardised distance of the
    ///   point's features from the buffer's running per-feature mean and
    ///   standard deviation. A point that looks like the recent stream is
    ///   relevant to the model currently being fitted; one many sigmas away is
    ///   less so (novelty is scored separately, by
    ///   `calculate_novelty_score`, and combined with a different weight).
    /// - **Supervision**: a labelled point supports a gradient step while an
    ///   unlabelled one cannot, so it is genuinely more relevant.
    ///
    /// With no history yet there is nothing to be relevant *to*, so the point's
    /// own quality score is used as the only available estimate.
    fn calculate_relevance_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        let supervision_bonus = if data_point.target.is_some() {
            A::from(0.2).ok_or_else(|| "0.2 is not representable".to_string())?
        } else {
            A::zero()
        };

        let statistics = &self.feature_statistics;
        if statistics.sample_count < 2 || statistics.means.is_empty() {
            return Ok((data_point.quality_score + supervision_bonus).min(A::one()));
        }

        let count = A::from(statistics.sample_count)
            .ok_or_else(|| "sample count is not representable".to_string())?;
        let mut squared_z_total = A::zero();
        let mut compared = 0usize;
        for (index, &value) in data_point.features.iter().enumerate() {
            let Some(&mean) = statistics.means.get(index) else {
                continue;
            };
            let Some(&m2) = statistics.m2.get(index) else {
                continue;
            };
            let variance = m2 / count;
            if variance <= A::zero() {
                continue;
            }
            let z = (value - mean) / variance.sqrt();
            squared_z_total = squared_z_total + z * z;
            compared += 1;
        }

        if compared == 0 {
            return Ok((data_point.quality_score + supervision_bonus).min(A::one()));
        }

        let compared_count =
            A::from(compared).ok_or_else(|| "compared count is not representable".to_string())?;
        // Root-mean-square z score across the compared coordinates.
        let rms_z = (squared_z_total / compared_count).sqrt();
        // Map [0, inf) monotonically onto (0, 1]: a point sitting on the mean
        // scores 1, a 1-sigma point 0.5, a 3-sigma point 0.25.
        let typicality = A::one() / (A::one() + rms_z);

        Ok((typicality + supervision_bonus).min(A::one()))
    }

    /// Folds a data point into the running per-feature statistics that back the
    /// relevance score.
    fn update_feature_statistics(&mut self, data_point: &StreamingDataPoint<A>) {
        let statistics = &mut self.feature_statistics;
        if statistics.means.len() < data_point.features.len() {
            statistics
                .means
                .resize(data_point.features.len(), A::zero());
            statistics.m2.resize(data_point.features.len(), A::zero());
        }
        statistics.sample_count = statistics.sample_count.saturating_add(1);
        let Some(count) = A::from(statistics.sample_count) else {
            return;
        };

        // Welford update per coordinate.
        for (index, &value) in data_point.features.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            let mean = statistics.means[index];
            let delta = value - mean;
            let new_mean = mean + delta / count;
            statistics.means[index] = new_mean;
            statistics.m2[index] = statistics.m2[index] + delta * (value - new_mean);
        }
    }

    /// Number of data points folded into the relevance statistics.
    pub fn feature_statistics_sample_count(&self) -> usize {
        self.feature_statistics.sample_count
    }

    /// Per-item processing time expected from the measured average latency.
    ///
    /// Falls back to the observed throughput's reciprocal, and only then to a
    /// documented default when neither has been measured yet.
    fn estimated_processing_time(&self) -> Duration {
        if self.statistics.avg_processing_latency > Duration::ZERO {
            return self.statistics.avg_processing_latency;
        }
        let throughput = self
            .statistics
            .throughput_stats
            .avg_throughput
            .to_f64()
            .unwrap_or(0.0);
        if throughput > 0.0 {
            return Duration::from_secs_f64(1.0 / throughput);
        }
        DEFAULT_EXPECTED_PROCESSING_TIME
    }

    /// Average processing latency measured over recent batches.
    pub fn average_processing_latency(&self) -> Duration {
        self.statistics.avg_processing_latency
    }

    /// Folds a real, measured batch processing duration into the buffer's
    /// latency statistics.
    ///
    /// B2: `statistics.avg_processing_latency` was initialised to
    /// `Duration::ZERO` and never written by anything, so the
    /// `avg_processing_latency > 500ms` branch in
    /// `calculate_optimal_batch_size` was unreachable dead code and the buffer
    /// never shrank its batches under load.
    pub fn record_processing_duration(&mut self, duration: Duration) {
        let previous = self.statistics.avg_processing_latency;
        self.statistics.avg_processing_latency = if previous == Duration::ZERO {
            duration
        } else {
            // Exponential moving average with the same smoothing factor used
            // for throughput, so the two statistics track at the same rate.
            let smoothed = LATENCY_SMOOTHING * duration.as_secs_f64()
                + (1.0 - LATENCY_SMOOTHING) * previous.as_secs_f64();
            Duration::from_secs_f64(smoothed.max(0.0))
        };
    }

    /// Gets a batch of data for processing
    pub fn get_batch_for_processing(&mut self) -> Result<Vec<StreamingDataPoint<A>>, String> {
        let batch_size = self.calculate_optimal_batch_size()?;
        let mut processing_batch = Vec::with_capacity(batch_size);

        // Extract high-priority items from main buffer
        while processing_batch.len() < batch_size && !self.buffer.is_empty() {
            if let Some(prioritized_point) = self.buffer.pop() {
                processing_batch.push(prioritized_point.data_point);
            }
        }

        // Fill remaining space with secondary buffer items if needed
        while processing_batch.len() < batch_size && !self.secondary_buffer.is_empty() {
            if let Some(data_point) = self.secondary_buffer.pop_front() {
                processing_batch.push(data_point);
            }
        }

        // Update last processing time
        self.last_processing = Instant::now();

        // Update throughput statistics
        self.update_throughput_stats(processing_batch.len())?;

        Ok(processing_batch)
    }

    /// Calculates optimal batch size based on current conditions
    fn calculate_optimal_batch_size(&self) -> Result<usize, String> {
        let mut batch_size = self.config.initial_size.min(32); // Default reasonable batch size

        // Adjust based on buffer fullness
        let buffer_utilization =
            self.current_size() as f64 / self.sizing_strategy.target_size as f64;
        if buffer_utilization > 0.8 {
            batch_size = (batch_size as f64 * 1.5) as usize; // Larger batches when buffer is full
        } else if buffer_utilization < 0.3 {
            batch_size = (batch_size as f64 * 0.7) as usize; // Smaller batches when buffer is sparse
        }

        // Adjust based on processing latency
        if self.statistics.avg_processing_latency > Duration::from_millis(500) {
            batch_size = (batch_size as f64 * 0.8) as usize; // Smaller batches for slow processing
        }

        // Ensure minimum and maximum bounds
        Ok(batch_size.max(1).min(self.current_size().min(100)))
    }

    /// Updates quality metrics for the buffer
    fn update_quality_metrics(&mut self) -> Result<(), String> {
        if self.buffer.is_empty() && self.secondary_buffer.is_empty() {
            return Ok(());
        }

        let mut quality_sum = A::zero();
        let mut quality_values = Vec::new();

        // Collect quality scores from main buffer
        for prioritized_point in &self.buffer {
            let quality = prioritized_point.data_point.quality_score;
            quality_sum = quality_sum + quality;
            quality_values.push(quality);
        }

        // Collect quality scores from secondary buffer
        for data_point in &self.secondary_buffer {
            let quality = data_point.quality_score;
            quality_sum = quality_sum + quality;
            quality_values.push(quality);
        }

        if !quality_values.is_empty() {
            let count = try_scalar_str::<A, _>(quality_values.len())?;
            self.quality_metrics.average_quality = quality_sum / count;

            // Update min/max quality
            self.quality_metrics.min_quality =
                quality_values.iter().cloned().fold(A::one(), A::min);
            self.quality_metrics.max_quality =
                quality_values.iter().cloned().fold(A::zero(), A::max);

            // Calculate quality variance
            let mean = self.quality_metrics.average_quality;
            let variance_sum = quality_values
                .iter()
                .map(|&q| (q - mean) * (q - mean))
                .sum::<A>();
            self.quality_metrics.quality_variance = variance_sum / count;

            // Update quality trend
            self.update_quality_trend(self.quality_metrics.average_quality)?;
        }

        Ok(())
    }

    /// Updates quality trend analysis
    fn update_quality_trend(&mut self, current_quality: A) -> Result<(), String> {
        let trend = &mut self.quality_metrics.quality_trend;

        // Add current quality to recent changes
        if trend.recent_changes.len() >= 50 {
            trend.recent_changes.pop_front();
        }
        trend.recent_changes.push_back(current_quality);

        // Analyze trend if we have enough data
        if trend.recent_changes.len() >= 10 {
            let recent: Vec<A> = trend.recent_changes.iter().cloned().collect();
            let half = recent.len() / 2;
            let first_count =
                A::from(half).ok_or_else(|| "half-window size is not representable".to_string())?;
            let second_count = A::from(recent.len() - half)
                .ok_or_else(|| "half-window size is not representable".to_string())?;
            let first_half_avg = recent.iter().take(half).cloned().sum::<A>() / first_count;
            let second_half_avg = recent.iter().skip(half).cloned().sum::<A>() / second_count;

            let change = second_half_avg - first_half_avg;
            let change_threshold =
                A::from(0.05).ok_or_else(|| "0.05 is not representable".to_string())?; // 5% change threshold

            trend.trend_direction = if change > change_threshold {
                TrendDirection::Improving
            } else if change < -change_threshold {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            };

            trend.trend_magnitude = change.abs();

            // B3: confidence used to be the constant 0.8, which made it useless
            // for deciding whether to act on the trend. It is now Welch's
            // two-sample t statistic for "the two halves have different means",
            // mapped monotonically into [0, 1): a large, consistent shift
            // relative to the within-half spread gives high confidence, while a
            // shift that is small compared to the noise gives low confidence.
            let variance_of = |values: &[A], count: A, mean: A| -> A {
                if values.len() < 2 {
                    return A::zero();
                }
                let denominator = count - A::one();
                if denominator <= A::zero() {
                    return A::zero();
                }
                values
                    .iter()
                    .fold(A::zero(), |acc, &v| acc + (v - mean) * (v - mean))
                    / denominator
            };
            let first_slice = &recent[..half];
            let second_slice = &recent[half..];
            let first_variance = variance_of(first_slice, first_count, first_half_avg);
            let second_variance = variance_of(second_slice, second_count, second_half_avg);
            let standard_error =
                (first_variance / first_count + second_variance / second_count).sqrt();

            trend.confidence = if standard_error > A::zero() {
                let t_statistic = (change / standard_error).abs();
                t_statistic / (A::one() + t_statistic)
            } else if change.abs() > A::zero() {
                // Zero within-half variance and a non-zero shift is a perfectly
                // clean step change.
                A::one()
            } else {
                A::zero()
            };
        }

        Ok(())
    }

    /// Checks if buffer needs resizing
    fn check_buffer_resizing(&mut self) -> Result<(), String> {
        if !self.config.enable_adaptive_sizing {
            return Ok(());
        }

        let current_size = self.current_size();
        let target_size = self.sizing_strategy.target_size;
        let utilization = current_size as f64 / target_size as f64;

        // Check if resize is needed
        let should_resize = if utilization > 0.9 {
            // Buffer is nearly full - consider growing
            Some(SizingReason::ThroughputOptimization)
        } else if utilization < 0.3 && target_size > self.config.min_size {
            // Buffer is underutilized - consider shrinking
            Some(SizingReason::MemoryPressure)
        } else {
            None
        };

        if let Some(reason) = should_resize {
            self.resize_buffer(reason)?;
        }

        Ok(())
    }

    /// Resizes the buffer based on current conditions
    fn resize_buffer(&mut self, reason: SizingReason) -> Result<(), String> {
        let old_size = self.sizing_strategy.target_size;

        // `BufferConfig::size_strategy` used to be accepted, stored and never
        // consulted: every strategy resized by the same `adjustment_params`
        // multipliers, so a buffer configured `Fixed` still grew and shrank.
        let growing = matches!(reason, SizingReason::ThroughputOptimization);
        let shrinking = matches!(reason, SizingReason::MemoryPressure);
        if !growing && !shrinking {
            return Ok(()); // no sizing signal
        }

        let new_size = match &self.sizing_strategy.strategy_type {
            // A fixed buffer is fixed.
            BufferSizeStrategy::Fixed => return Ok(()),
            // Additive steps of `growth_rate * initial_size`.
            BufferSizeStrategy::Linear { growth_rate } => {
                let step = ((self.sizing_strategy.initial_size as f64) * growth_rate.abs())
                    .round()
                    .max(1.0) as usize;
                if growing {
                    old_size.saturating_add(step)
                } else {
                    old_size.saturating_sub(step)
                }
            }
            // Multiplicative steps of the configured base.
            BufferSizeStrategy::Exponential { base } => {
                let base = if *base > 1.0 { *base } else { 2.0 };
                if growing {
                    ((old_size as f64) * base) as usize
                } else {
                    ((old_size as f64) / base) as usize
                }
            }
            // Performance- and resource-driven sizing both use the tuned
            // sensitivity parameters; `bound_target_size` then applies the
            // configured memory budget on top.
            BufferSizeStrategy::Adaptive | BufferSizeStrategy::ResourceBased => {
                if growing {
                    let growth_factor = 1.0
                        + self
                            .sizing_strategy
                            .adjustment_params
                            .growth_rate
                            .to_f64()
                            .unwrap_or(0.2);
                    ((old_size as f64) * growth_factor) as usize
                } else {
                    let shrink_factor = 1.0
                        - self
                            .sizing_strategy
                            .adjustment_params
                            .shrinkage_rate
                            .to_f64()
                            .unwrap_or(0.2);
                    ((old_size as f64) * shrink_factor) as usize
                }
            }
        };

        // Apply size bounds, including the configured memory budget (CF1).
        let bounded_size = self.bound_target_size(new_size);

        if bounded_size != old_size {
            self.sizing_strategy.target_size = bounded_size;

            // Log the size change
            let change_event = SizeChangeEvent {
                timestamp: Instant::now(),
                old_size,
                new_size: bounded_size,
                change_magnitude: bounded_size as i32 - old_size as i32,
                reason: format!("{:?}", reason),
            };

            if self.size_change_log.len() >= 100 {
                self.size_change_log.pop_front();
            }
            self.size_change_log.push_back(change_event);
        }

        Ok(())
    }

    /// Applies retention policy to manage buffer size
    fn apply_retention_policy(&mut self) -> Result<(), String> {
        let target_size = self.sizing_strategy.target_size;
        let current_size = self.current_size();

        if current_size <= target_size {
            return Ok(());
        }

        let items_to_remove = current_size - target_size;
        let mut removed_count = 0;

        // Apply retention policy to secondary buffer first
        while removed_count < items_to_remove && !self.secondary_buffer.is_empty() {
            if self.should_remove_from_secondary()? {
                self.secondary_buffer.pop_front();
                removed_count += 1;
                self.statistics.total_items_discarded += 1;
            } else {
                break;
            }
        }

        // If still need to remove items, apply to main buffer
        let mut temp_buffer = Vec::new();
        while let Some(item) = self.buffer.pop() {
            temp_buffer.push(item);
        }

        // Sort by retention score and keep the best items
        temp_buffer.sort_by(|a, b| {
            let score_a = self
                .calculate_retention_score(&a.data_point)
                .unwrap_or(A::zero());
            let score_b = self
                .calculate_retention_score(&b.data_point)
                .unwrap_or(A::zero());
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        // Keep only the target number of items
        let items_to_keep = (temp_buffer.len()).saturating_sub(items_to_remove - removed_count);
        for item in temp_buffer.into_iter().take(items_to_keep) {
            self.buffer.push(item);
        }

        Ok(())
    }

    /// Determines if an item should be removed from secondary buffer
    fn should_remove_from_secondary(&self) -> Result<bool, String> {
        // Simple policy: remove oldest items first
        if let Some(oldest) = self.secondary_buffer.front() {
            let age = oldest.timestamp.elapsed();
            Ok(age > Duration::from_secs(3600)) // Remove items older than 1 hour
        } else {
            Ok(false)
        }
    }

    /// Calculates retention score for a data point
    fn calculate_retention_score(&self, data_point: &StreamingDataPoint<A>) -> Result<A, String> {
        let age_score = self.calculate_age_score(data_point);
        let quality_score = data_point.quality_score;
        let freshness_score = self.calculate_freshness_score(data_point);

        // Weighted combination
        let retention_score = quality_score * try_scalar_str::<A, _>(0.5)?
            + freshness_score * try_scalar_str::<A, _>(0.3)?
            + age_score * try_scalar_str::<A, _>(0.2)?;

        Ok(retention_score)
    }

    /// Calculates age score for retention
    fn calculate_age_score(&self, data_point: &StreamingDataPoint<A>) -> A {
        let age_seconds = data_point.timestamp.elapsed().as_secs_f64();
        let max_age = 7200.0; // 2 hours

        let age_score = (max_age - age_seconds.min(max_age)) / max_age;
        scalar_or(age_score.max(0.0), A::zero())
    }

    /// Updates throughput statistics
    fn update_throughput_stats(&mut self, items_processed: usize) -> Result<(), String> {
        let time_since_last = self.last_processing.elapsed().as_secs_f64();
        if time_since_last > 0.0 {
            let current_throughput = items_processed as f64 / time_since_last;
            let throughput_value = try_scalar_str::<A, _>(current_throughput)?;

            self.statistics.throughput_stats.current_throughput = throughput_value;

            // Update average throughput (simple moving average)
            let alpha = try_scalar_str::<A, _>(0.1)?; // Smoothing factor
            self.statistics.throughput_stats.avg_throughput = alpha * throughput_value
                + (A::one() - alpha) * self.statistics.throughput_stats.avg_throughput;

            // Update peak throughput
            self.statistics.throughput_stats.peak_throughput = self
                .statistics
                .throughput_stats
                .peak_throughput
                .max(throughput_value);
        }

        Ok(())
    }

    /// Gets current buffer size (total items across all buffers)
    pub fn current_size(&self) -> usize {
        self.buffer.len() + self.secondary_buffer.len()
    }

    /// Gets time since last processing
    pub fn time_since_last_processing(&self) -> Duration {
        self.last_processing.elapsed()
    }

    /// Gets current buffer quality metrics
    pub fn get_quality_metrics(&self) -> BufferQualityMetrics<A> {
        self.quality_metrics.clone()
    }

    /// Computes size adaptation based on performance feedback
    pub fn compute_size_adaptation(
        &self,
        performance_tracker: &PerformanceTracker<A>,
    ) -> Result<Option<Adaptation<A>>, String> {
        // Get recent performance data
        let recent_performance = performance_tracker.get_recent_performance(10);
        if recent_performance.is_empty() {
            return Ok(None);
        }

        // Calculate average processing time.
        //
        // B2: this used `p.timestamp.elapsed()` — the *age* of each snapshot,
        // which grows without bound the longer the process runs. Every stream
        // therefore looked slower and slower until the "reduce buffer size"
        // branch latched permanently. `processing_duration` is the measured cost
        // of the step that produced the snapshot, which is what this decision
        // actually needs.
        let avg_processing_time = recent_performance
            .iter()
            .map(|p| p.processing_duration.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / recent_performance.len() as f64;

        // If processing is too slow, suggest reducing buffer size
        if avg_processing_time > 1000.0 {
            // More than 1 second
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::BufferSize,
                magnitude: try_scalar_str::<A, _>(-0.2)?, // Reduce by 20%
                target_component: "adaptive_buffer".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::Normal,
                timestamp: Instant::now(),
            };
            return Ok(Some(adaptation));
        }

        // If processing is very fast and buffer is often empty, suggest increasing size
        let avg_utilization = self.current_size() as f64 / self.sizing_strategy.target_size as f64;
        if avg_processing_time < 100.0 && avg_utilization < 0.3 {
            let adaptation = Adaptation {
                adaptation_type: AdaptationType::BufferSize,
                magnitude: try_scalar_str::<A, _>(0.3)?, // Increase by 30%
                target_component: "adaptive_buffer".to_string(),
                parameters: std::collections::HashMap::new(),
                priority: AdaptationPriority::Low,
                timestamp: Instant::now(),
            };
            return Ok(Some(adaptation));
        }

        Ok(None)
    }

    /// Maximum number of buffered items that fit inside
    /// `BufferConfig::memory_limit_mb` (CF1).
    ///
    /// The per-item footprint is *measured*, not assumed: it is the size of a
    /// `PrioritizedDataPoint<A>` plus the heap held by its feature (and target)
    /// vectors, whose dimensionality is known from the running per-feature
    /// statistics. Returns `None` until at least one data point has been
    /// observed, since there is no measurement to bound against before then.
    ///
    /// `memory_limit_mb` used to be a config field nothing read, so a buffer
    /// configured with a 128 MB budget would still grow to `max_size` items
    /// regardless of how wide each sample was.
    fn memory_bounded_capacity(&self) -> Option<usize> {
        let feature_dim = self.feature_statistics.means.len();
        if feature_dim == 0 {
            return None;
        }
        // `features` plus a possible `target` of the same width.
        let per_item = std::mem::size_of::<PrioritizedDataPoint<A>>()
            + 2 * feature_dim * std::mem::size_of::<A>();
        if per_item == 0 {
            return None;
        }
        let budget_bytes = self.config.memory_limit_mb.saturating_mul(1024 * 1024);
        Some((budget_bytes / per_item).max(1))
    }

    /// Clamps a proposed target size to the configured item bounds *and* the
    /// memory budget.
    fn bound_target_size(&self, proposed: usize) -> usize {
        let bounded = proposed.max(self.config.min_size).min(self.config.max_size);
        match self.memory_bounded_capacity() {
            Some(capacity) => bounded.min(capacity).max(1),
            None => bounded,
        }
    }

    /// Test-only view of [`Self::memory_bounded_capacity`].
    #[cfg(test)]
    pub(crate) fn memory_bounded_capacity_for_test(&self) -> Option<usize> {
        self.memory_bounded_capacity()
    }

    /// Test-only view of [`Self::bound_target_size`].
    #[cfg(test)]
    pub(crate) fn bound_target_size_for_test(&self, proposed: usize) -> usize {
        self.bound_target_size(proposed)
    }

    /// Applies size adaptation to the buffer
    pub fn apply_size_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        if adaptation.adaptation_type == AdaptationType::BufferSize {
            let current_target = self.sizing_strategy.target_size;
            let change_factor = A::one() + adaptation.magnitude;
            let new_target =
                (current_target as f64 * change_factor.to_f64().unwrap_or(1.0)) as usize;

            // Apply bounds, including the configured memory budget (CF1).
            let bounded_target = self.bound_target_size(new_target);

            if bounded_target != current_target {
                self.sizing_strategy.target_size = bounded_target;

                // Log the change
                let change_event = SizeChangeEvent {
                    timestamp: Instant::now(),
                    old_size: current_target,
                    new_size: bounded_target,
                    change_magnitude: bounded_target as i32 - current_target as i32,
                    reason: "adaptation".to_string(),
                };

                if self.size_change_log.len() >= 100 {
                    self.size_change_log.pop_front();
                }
                self.size_change_log.push_back(change_event);
            }
        }

        Ok(())
    }

    /// Gets the last size change amount
    pub fn last_size_change(&self) -> f32 {
        if let Some(last_change) = self.size_change_log.back() {
            last_change.change_magnitude as f32
        } else {
            0.0
        }
    }

    /// Resets the buffer to initial state
    pub fn reset(&mut self) -> Result<(), String> {
        self.buffer.clear();
        self.secondary_buffer.clear();

        self.quality_metrics = BufferQualityMetrics {
            average_quality: A::zero(),
            quality_variance: A::zero(),
            min_quality: A::one(),
            max_quality: A::zero(),
            freshness_distribution: Vec::new(),
            priority_distribution: Vec::new(),
            quality_trend: QualityTrend {
                recent_changes: VecDeque::with_capacity(50),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: A::zero(),
                confidence: A::zero(),
            },
        };

        self.statistics.total_items_processed = 0;
        self.statistics.total_items_discarded = 0;
        self.last_processing = Instant::now();
        self.size_change_log.clear();

        Ok(())
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> BufferDiagnostics {
        BufferDiagnostics {
            current_size: self.current_size(),
            target_size: self.sizing_strategy.target_size,
            utilization: self.current_size() as f64 / self.sizing_strategy.target_size as f64,
            average_quality: self.quality_metrics.average_quality.to_f64().unwrap_or(0.0),
            total_processed: self.statistics.total_items_processed,
            total_discarded: self.statistics.total_items_discarded,
            size_changes: self.size_change_log.len(),
        }
    }
}

// Implement Ord for PrioritizedDataPoint to work with BinaryHeap
impl<A: Float + Send + Sync + Send + Sync> Ord for PrioritizedDataPoint<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score
            .partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl<A: Float + Send + Sync + Send + Sync> PartialOrd for PrioritizedDataPoint<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Float + Send + Sync + Send + Sync> PartialEq for PrioritizedDataPoint<A> {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl<A: Float + Send + Sync + Send + Sync> Eq for PrioritizedDataPoint<A> {}

impl<A: Float + Send + Sync + Send + Sync> BufferSizingStrategy<A> {
    fn new(strategy_type: BufferSizeStrategy, initial_size: usize) -> Self {
        Self {
            strategy_type,
            initial_size,
            target_size: initial_size,
            adjustment_params: SizeAdjustmentParams {
                growth_rate: scalar_or(0.2, A::zero()),
                shrinkage_rate: scalar_or(0.15, A::zero()),
                stability_threshold: scalar_or(0.05, A::zero()),
                performance_sensitivity: scalar_or(0.1, A::zero()),
                quality_sensitivity: scalar_or(0.1, A::zero()),
                memory_sensitivity: scalar_or(0.2, A::zero()),
            },
        }
    }
}

/// Diagnostic information for buffer management
#[derive(Debug, Clone)]
pub struct BufferDiagnostics {
    pub current_size: usize,
    pub target_size: usize,
    pub utilization: f64,
    pub average_quality: f64,
    pub total_processed: u64,
    pub total_discarded: u64,
    pub size_changes: usize,
}

#[cfg(test)]
mod buffering_regression_tests {
    use super::super::performance::{DataStatistics, PerformanceSnapshot};
    use super::super::resource_management::ResourceUsage;
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn point(features: Vec<f64>, target: Option<f64>) -> StreamingDataPoint<f64> {
        StreamingDataPoint {
            features: Array1::from_vec(features),
            target: target.map(|t| Array1::from_vec(vec![t])),
            timestamp: Instant::now(),
            source_id: None,
            quality_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn buffer() -> AdaptiveBuffer<f64> {
        AdaptiveBuffer::new(&StreamingConfig::default()).expect("buffer")
    }

    fn snapshot_with(processing: Duration) -> PerformanceSnapshot<f64> {
        PerformanceSnapshot {
            timestamp: Instant::now(),
            processing_duration: processing,
            loss: 1.0,
            accuracy: None,
            convergence_rate: None,
            gradient_norm: None,
            parameter_update_magnitude: None,
            data_statistics: DataStatistics::default(),
            resource_usage: ResourceUsage::default(),
            custom_metrics: HashMap::new(),
        }
    }

    /// B1: `calculate_relevance_score` returned the constant `0.7` for every
    /// point, so relevance carried zero information. It must now discriminate: a
    /// point sitting on the buffer's running mean is more relevant than one many
    /// standard deviations away.
    #[test]
    fn relevance_score_discriminates_typical_from_atypical_points() {
        let mut buffer = buffer();

        // Establish a tight distribution around 10.0.
        for i in 0..40 {
            let value = 10.0 + 0.1 * ((i % 5) as f64 - 2.0);
            buffer.update_feature_statistics(&point(vec![value], None));
        }
        assert_eq!(buffer.feature_statistics_sample_count(), 40);

        let typical = buffer
            .calculate_relevance_score(&point(vec![10.0], None))
            .expect("typical");
        let atypical = buffer
            .calculate_relevance_score(&point(vec![10_000.0], None))
            .expect("atypical");

        assert!(
            typical > atypical,
            "B1 regression: relevance did not discriminate \
             (typical={typical}, atypical={atypical})"
        );
        assert!(
            (typical - 0.7).abs() > 1e-9 || (atypical - 0.7).abs() > 1e-9,
            "B1 regression: both scores are still the hard-coded 0.7"
        );
        assert!(
            atypical < 0.1,
            "a 100-sigma point should score near zero relevance, got {atypical}"
        );
    }

    /// B1: a labelled point can support a gradient step and an unlabelled one
    /// cannot, so supervision must raise relevance.
    #[test]
    fn relevance_score_rewards_labelled_points() {
        let mut buffer = buffer();
        for i in 0..40 {
            let value = 10.0 + 0.1 * ((i % 5) as f64 - 2.0);
            buffer.update_feature_statistics(&point(vec![value], None));
        }

        // Use an off-centre value so neither score saturates at the 1.0 clamp.
        let unlabelled = buffer
            .calculate_relevance_score(&point(vec![10.6], None))
            .expect("unlabelled");
        let labelled = buffer
            .calculate_relevance_score(&point(vec![10.6], Some(1.0)))
            .expect("labelled");
        assert!(
            labelled > unlabelled,
            "a labelled point must be at least as relevant \
             (labelled={labelled}, unlabelled={unlabelled})"
        );
    }

    /// B2: `compute_size_adaptation` measured "processing time" as
    /// `snapshot.timestamp.elapsed()` — the snapshot's *age*. Age grows without
    /// bound as the process runs, so after a while every stream looked slower
    /// than one second per batch and the "shrink the buffer" branch latched
    /// permanently. Snapshots that took 1ms each must produce no shrink request
    /// no matter how old they are.
    #[test]
    fn size_adaptation_uses_measured_processing_time_not_snapshot_age() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");
        for _ in 0..10 {
            tracker
                .add_performance(snapshot_with(Duration::from_millis(1)))
                .expect("add_performance");
        }
        // Let the snapshots visibly age; under the bug this is what was measured.
        std::thread::sleep(Duration::from_millis(60));

        let mut buffer = buffer();
        // Fill past 30% utilisation so the "grow" branch is not taken either.
        for i in 0..200 {
            buffer
                .add_batch(vec![point(vec![i as f64], None)])
                .expect("add_batch");
        }

        let adaptation = buffer
            .compute_size_adaptation(&tracker)
            .expect("compute_size_adaptation");
        if let Some(adaptation) = adaptation {
            assert!(
                adaptation.magnitude > 0.0,
                "B2 regression: a 1ms-per-batch workload produced a shrink \
                 request (magnitude={}), which can only come from reading \
                 snapshot age as processing time",
                adaptation.magnitude
            );
        }
    }

    /// B2: a genuinely slow workload must still be caught, so the fix does not
    /// simply blind the detector.
    #[test]
    fn size_adaptation_still_shrinks_for_genuinely_slow_processing() {
        let config = StreamingConfig::default();
        let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");
        for _ in 0..10 {
            tracker
                .add_performance(snapshot_with(Duration::from_millis(1500)))
                .expect("add_performance");
        }

        let buffer = buffer();
        let adaptation = buffer
            .compute_size_adaptation(&tracker)
            .expect("compute_size_adaptation")
            .expect("a 1.5s-per-batch workload must request a smaller buffer");
        assert!(
            adaptation.magnitude < 0.0,
            "expected a shrink request, got magnitude {}",
            adaptation.magnitude
        );
    }

    /// B2: `statistics.avg_processing_latency` was initialised to
    /// `Duration::ZERO` and never written by anything, making the
    /// `> 500ms` branch of `calculate_optimal_batch_size` unreachable.
    #[test]
    fn processing_latency_is_recorded_and_shrinks_the_batch_size() {
        let mut buffer = buffer();
        assert_eq!(
            buffer.average_processing_latency(),
            Duration::ZERO,
            "no latency should be claimed before any measurement"
        );

        for i in 0..200 {
            buffer
                .add_batch(vec![point(vec![i as f64], None)])
                .expect("add_batch");
        }
        let fast_batch = buffer
            .calculate_optimal_batch_size()
            .expect("calculate_optimal_batch_size");

        // Report a genuinely slow batch several times so the EMA clears 500ms.
        for _ in 0..40 {
            buffer.record_processing_duration(Duration::from_millis(2000));
        }
        assert!(
            buffer.average_processing_latency() > Duration::from_millis(500),
            "B2 regression: recorded latency did not reach the statistics \
             (got {:?})",
            buffer.average_processing_latency()
        );

        let slow_batch = buffer
            .calculate_optimal_batch_size()
            .expect("calculate_optimal_batch_size");
        assert!(
            slow_batch < fast_batch,
            "B2 regression: the slow-processing branch is still unreachable \
             ({fast_batch} -> {slow_batch})"
        );
    }

    /// B3: `trend.confidence` was the constant `0.8`, so no caller could tell a
    /// clean step change from pure noise. A clean, low-noise step must now score
    /// high confidence and a noisy no-op must score low.
    #[test]
    fn quality_trend_confidence_reflects_the_real_fit() {
        // Clean step: first half at 0.2, second half at 0.9, no within-half noise.
        let mut clean = buffer();
        for i in 0..10 {
            let quality = if i < 5 { 0.2 } else { 0.9 };
            clean
                .update_quality_trend(quality)
                .expect("update_quality_trend");
        }
        let clean_confidence = clean.get_quality_metrics().quality_trend.confidence;

        // Noisy, trendless series alternating around the same mean.
        let mut noisy = buffer();
        for i in 0..10 {
            let quality = if i % 2 == 0 { 0.1 } else { 0.9 };
            noisy
                .update_quality_trend(quality)
                .expect("update_quality_trend");
        }
        let noisy_confidence = noisy.get_quality_metrics().quality_trend.confidence;

        assert!(
            clean_confidence > noisy_confidence,
            "B3 regression: confidence did not distinguish a clean step from \
             noise (clean={clean_confidence}, noisy={noisy_confidence})"
        );
        assert!(
            (clean_confidence - 0.8).abs() > 1e-9 || (noisy_confidence - 0.8).abs() > 1e-9,
            "B3 regression: both confidences are still the hard-coded 0.8"
        );
        assert_eq!(
            clean.get_quality_metrics().quality_trend.trend_direction,
            TrendDirection::Improving
        );
    }
}
