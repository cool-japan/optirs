// Adaptive memory sizing based on usage patterns

use std::collections::VecDeque;
use std::time::Instant;

/// Adaptive memory sizing based on usage patterns
#[derive(Debug, Clone)]
pub struct AdaptiveSizing {
    /// Enable adaptive pool resizing
    pub enable_adaptive_resize: bool,
    /// Allocation history for pattern analysis
    pub allocation_history: VecDeque<AllocationEvent>,
    /// Maximum history size
    pub max_history_size: usize,
    /// Resize threshold (utilization percentage)
    pub resize_threshold: f32,
    /// Minimum pool size (bytes)
    pub min_pool_size: usize,
    /// Pool size growth factor
    pub growth_factor: f32,
    /// Pool size shrink factor
    pub shrink_factor: f32,
    /// Analysis window size (number of allocations)
    pub analysis_window: usize,
}

impl Default for AdaptiveSizing {
    fn default() -> Self {
        Self {
            enable_adaptive_resize: true,
            allocation_history: VecDeque::new(),
            max_history_size: 10000,
            resize_threshold: 0.85,
            min_pool_size: 256 * 1024 * 1024, // 256MB
            growth_factor: 1.5,
            shrink_factor: 0.75,
            analysis_window: 1000,
        }
    }
}

/// Allocation event for pattern analysis
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Size of allocation
    pub size: usize,
    /// Timestamp of allocation
    pub timestamp: Instant,
    /// Whether allocation was satisfied from cache
    pub cache_hit: bool,
    /// Allocation latency (microseconds)
    pub latency_us: u64,
}

impl AdaptiveSizing {
    /// Record allocation event
    pub fn record_allocation(&mut self, size: usize, cache_hit: bool, latency_us: u64) {
        let event = AllocationEvent {
            size,
            timestamp: Instant::now(),
            cache_hit,
            latency_us,
        };

        self.allocation_history.push_back(event);

        // Maintain history size limit
        while self.allocation_history.len() > self.max_history_size {
            self.allocation_history.pop_front();
        }
    }

    /// Analyze allocation patterns and suggest pool size adjustment
    pub fn analyze_and_suggest_resize(&self, current_pool_size: usize, current_utilization: f32) -> Option<usize> {
        if !self.enable_adaptive_resize || self.allocation_history.len() < self.analysis_window {
            return None;
        }

        // Analyze recent allocation patterns
        let recent_events: Vec<_> = self.allocation_history
            .iter()
            .rev()
            .take(self.analysis_window)
            .collect();

        let cache_miss_rate = recent_events.iter()
            .filter(|e| !e.cache_hit)
            .count() as f32 / recent_events.len() as f32;

        let avg_latency = recent_events.iter()
            .map(|e| e.latency_us)
            .sum::<u64>() as f32 / recent_events.len() as f32;

        // Decision criteria
        let should_grow = current_utilization > self.resize_threshold
            || cache_miss_rate > 0.3
            || avg_latency > 1000.0; // High latency indicates pressure

        let should_shrink = current_utilization < 0.5
            && cache_miss_rate < 0.1
            && avg_latency < 100.0
            && current_pool_size > self.min_pool_size * 2;

        if should_grow {
            Some((current_pool_size as f32 * self.growth_factor) as usize)
        } else if should_shrink {
            let new_size = (current_pool_size as f32 * self.shrink_factor) as usize;
            Some(new_size.max(self.min_pool_size))
        } else {
            None
        }
    }

    /// Get allocation pattern insights
    pub fn get_pattern_insights(&self) -> AllocationPatternInsights {
        if self.allocation_history.is_empty() {
            return AllocationPatternInsights::default();
        }

        let total_allocations = self.allocation_history.len();
        let cache_hits = self.allocation_history.iter().filter(|e| e.cache_hit).count();
        let cache_hit_rate = cache_hits as f32 / total_allocations as f32;

        let avg_latency = self.allocation_history.iter()
            .map(|e| e.latency_us)
            .sum::<u64>() as f32 / total_allocations as f32;

        let size_distribution = self.calculate_size_distribution();
        let temporal_pattern = self.analyze_temporal_pattern();

        AllocationPatternInsights {
            total_allocations,
            cache_hit_rate,
            average_latency_us: avg_latency,
            size_distribution,
            temporal_pattern,
        }
    }

    /// Calculate size distribution
    fn calculate_size_distribution(&self) -> SizeDistribution {
        let sizes: Vec<usize> = self.allocation_history.iter().map(|e| e.size).collect();

        if sizes.is_empty() {
            return SizeDistribution::default();
        }

        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort_unstable();

        let min_size = sorted_sizes[0];
        let max_size = sorted_sizes[sorted_sizes.len() - 1];
        let median_size = sorted_sizes[sorted_sizes.len() / 2];
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;

        SizeDistribution {
            min_size,
            max_size,
            median_size,
            average_size: avg_size,
        }
    }

    /// Analyze temporal allocation patterns
    fn analyze_temporal_pattern(&self) -> TemporalPattern {
        if self.allocation_history.len() < 2 {
            return TemporalPattern::default();
        }

        let intervals: Vec<u64> = self.allocation_history
            .windows(2)
            .map(|window| {
                window[1].timestamp.duration_since(window[0].timestamp).as_millis() as u64
            })
            .collect();

        let avg_interval = intervals.iter().sum::<u64>() as f32 / intervals.len() as f32;

        // Calculate burstiness (coefficient of variation)
        let variance = intervals.iter()
            .map(|&x| (x as f32 - avg_interval).powi(2))
            .sum::<f32>() / intervals.len() as f32;
        let std_dev = variance.sqrt();
        let burstiness = if avg_interval > 0.0 { std_dev / avg_interval } else { 0.0 };

        TemporalPattern {
            average_interval_ms: avg_interval,
            burstiness_factor: burstiness,
            is_bursty: burstiness > 1.0, // High coefficient of variation indicates bursty behavior
        }
    }

    /// Clear allocation history
    pub fn clear_history(&mut self) {
        self.allocation_history.clear();
    }

    /// Get recommendation for pool configuration
    pub fn get_pool_recommendations(&self) -> PoolRecommendations {
        let insights = self.get_pattern_insights();

        let recommended_strategy = if insights.temporal_pattern.is_bursty {
            "Consider pre-allocation for bursty workloads"
        } else if insights.cache_hit_rate < 0.5 {
            "Increase pool size or implement better caching"
        } else if insights.average_latency_us > 1000.0 {
            "Optimize allocation strategy for better performance"
        } else {
            "Current configuration appears optimal"
        };

        let recommended_min_size = insights.size_distribution.min_size;
        let recommended_max_size = (insights.size_distribution.max_size as f32 * 1.2) as usize;

        PoolRecommendations {
            strategy_recommendation: recommended_strategy.to_string(),
            recommended_min_block_size: recommended_min_size,
            recommended_max_block_size: recommended_max_size,
            should_enable_preallocation: insights.temporal_pattern.is_bursty,
            confidence_score: self.calculate_confidence_score(&insights),
        }
    }

    /// Calculate confidence in recommendations based on data quality
    fn calculate_confidence_score(&self, insights: &AllocationPatternInsights) -> f32 {
        let sample_size_score = (self.allocation_history.len() as f32 / 1000.0).min(1.0);
        let consistency_score = 1.0 - insights.temporal_pattern.burstiness_factor.min(1.0);

        (sample_size_score + consistency_score) / 2.0
    }
}

/// Allocation pattern insights
#[derive(Debug, Clone, Default)]
pub struct AllocationPatternInsights {
    pub total_allocations: usize,
    pub cache_hit_rate: f32,
    pub average_latency_us: f32,
    pub size_distribution: SizeDistribution,
    pub temporal_pattern: TemporalPattern,
}

/// Size distribution statistics
#[derive(Debug, Clone, Default)]
pub struct SizeDistribution {
    pub min_size: usize,
    pub max_size: usize,
    pub median_size: usize,
    pub average_size: f32,
}

/// Temporal allocation pattern
#[derive(Debug, Clone, Default)]
pub struct TemporalPattern {
    pub average_interval_ms: f32,
    pub burstiness_factor: f32,
    pub is_bursty: bool,
}

/// Pool configuration recommendations
#[derive(Debug, Clone)]
pub struct PoolRecommendations {
    pub strategy_recommendation: String,
    pub recommended_min_block_size: usize,
    pub recommended_max_block_size: usize,
    pub should_enable_preallocation: bool,
    pub confidence_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_sizing_basic() {
        let mut adaptive = AdaptiveSizing::default();

        // Record some allocations
        adaptive.record_allocation(1024, true, 100);
        adaptive.record_allocation(2048, false, 200);

        assert_eq!(adaptive.allocation_history.len(), 2);
    }

    #[test]
    fn test_pattern_insights() {
        let mut adaptive = AdaptiveSizing::default();

        // Record pattern
        for i in 0..10 {
            adaptive.record_allocation(1024 * (i + 1), i % 2 == 0, 100 + i as u64 * 10);
        }

        let insights = adaptive.get_pattern_insights();
        assert_eq!(insights.total_allocations, 10);
        assert!(insights.cache_hit_rate >= 0.0 && insights.cache_hit_rate <= 1.0);
    }

    #[test]
    fn test_resize_suggestion() {
        let mut adaptive = AdaptiveSizing::default();
        adaptive.analysis_window = 5; // Lower threshold for testing

        // Record high-pressure pattern
        for i in 0..10 {
            adaptive.record_allocation(1024 * 1024, false, 2000); // High latency, cache misses
        }

        let suggestion = adaptive.analyze_and_suggest_resize(512 * 1024 * 1024, 0.9);
        assert!(suggestion.is_some());
    }
}