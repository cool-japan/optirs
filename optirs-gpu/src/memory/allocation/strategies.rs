// Allocation strategies for GPU memory management
//
// This module provides various allocation strategies optimized for different
// workload patterns and memory usage scenarios.

#[allow(dead_code)]
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Available allocation strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum AllocationStrategy {
    /// First-fit allocation (fastest)
    FirstFit,
    /// Best-fit allocation (memory efficient)
    BestFit,
    /// Worst-fit allocation (reduces fragmentation)
    WorstFit,
    /// Buddy system allocation (power-of-2 sizes)
    BuddySystem,
    /// Segregated list allocation (size-based pools)
    SegregatedList,
    /// Adaptive strategy based on workload
    #[default]
    Adaptive,
    /// Machine learning based allocation
    MLBased,
    /// Hybrid strategy combining multiple approaches
    Hybrid,
}

/// Memory block representation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub ptr: *mut u8,
    pub size: usize,
    pub is_free: bool,
    pub allocated_at: Option<Instant>,
    pub last_accessed: Option<Instant>,
    pub access_count: u64,
    pub fragmentation_score: f32,
}

impl MemoryBlock {
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            is_free: true,
            allocated_at: None,
            last_accessed: None,
            access_count: 0,
            fragmentation_score: 0.0,
        }
    }

    pub fn mark_used(&mut self) {
        self.is_free = false;
        self.allocated_at = Some(Instant::now());
        self.access_count += 1;
    }

    pub fn mark_free(&mut self) {
        self.is_free = true;
        self.allocated_at = None;
    }

    pub fn update_access(&mut self) {
        self.last_accessed = Some(Instant::now());
        self.access_count += 1;
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
    /// Thread ID that made the allocation
    pub thread_id: Option<u64>,
    /// Kernel context information
    pub kernel_context: Option<String>,
}

impl AllocationEvent {
    pub fn new(size: usize, cache_hit: bool, latency_us: u64) -> Self {
        Self {
            size,
            timestamp: Instant::now(),
            cache_hit,
            latency_us,
            thread_id: None,
            kernel_context: None,
        }
    }
}

/// Allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub fragmentation_events: u64,
    pub total_allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub average_allocation_size: f64,
    pub allocation_latency_ms: f64,
}

impl AllocationStats {
    pub fn record_allocation(&mut self, size: usize, cache_hit: bool, latency_us: u64) {
        self.total_allocations += 1;
        self.total_allocated_bytes += size as u64;

        if self.total_allocated_bytes > self.peak_allocated_bytes {
            self.peak_allocated_bytes = self.total_allocated_bytes;
        }

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        // Update average allocation size
        self.average_allocation_size =
            self.total_allocated_bytes as f64 / self.total_allocations as f64;

        // Update average latency
        self.allocation_latency_ms = (self.allocation_latency_ms
            * (self.total_allocations - 1) as f64
            + latency_us as f64 / 1000.0)
            / self.total_allocations as f64;
    }

    pub fn record_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.total_allocated_bytes = self.total_allocated_bytes.saturating_sub(size as u64);
    }

    pub fn get_cache_hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_allocations as f64
        }
    }

    pub fn get_fragmentation_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.fragmentation_events as f64 / self.total_allocations as f64
        }
    }
}

/// Core allocation strategy implementation
pub struct AllocationStrategyManager {
    strategy: AllocationStrategy,
    free_blocks: HashMap<usize, VecDeque<MemoryBlock>>,
    allocation_history: VecDeque<AllocationEvent>,
    stats: AllocationStats,
    adaptive_config: AdaptiveConfig,
    hybrid_config: HybridConfig,
    ml_config: Option<MLConfig>,
}

/// Configuration for adaptive allocation strategy
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    pub history_window: usize,
    pub small_allocation_threshold: usize,
    pub large_allocation_threshold: usize,
    pub fragmentation_threshold: f32,
    pub enable_pattern_detection: bool,
    pub adaptation_interval: u64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            history_window: 1000,
            small_allocation_threshold: 4096,
            large_allocation_threshold: 1024 * 1024,
            fragmentation_threshold: 0.3,
            enable_pattern_detection: true,
            adaptation_interval: 100,
        }
    }
}

/// Configuration for hybrid allocation strategy
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub primary_strategy: AllocationStrategy,
    pub secondary_strategy: AllocationStrategy,
    pub switch_threshold_fragmentation: f32,
    pub switch_threshold_utilization: f32,
    pub evaluation_window: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            primary_strategy: AllocationStrategy::BestFit,
            secondary_strategy: AllocationStrategy::FirstFit,
            switch_threshold_fragmentation: 0.4,
            switch_threshold_utilization: 0.8,
            evaluation_window: 100,
        }
    }
}

/// Configuration for ML-based allocation strategy
#[derive(Debug, Clone)]
pub struct MLConfig {
    pub model_type: MLModelType,
    pub feature_window: usize,
    pub training_interval: u64,
    pub prediction_confidence_threshold: f32,
    pub fallback_strategy: AllocationStrategy,
}

#[derive(Debug, Clone)]
pub enum MLModelType {
    LinearRegression,
    DecisionTree,
    NeuralNetwork,
    ReinforcementLearning,
}

impl AllocationStrategyManager {
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            strategy,
            free_blocks: HashMap::new(),
            allocation_history: VecDeque::new(),
            stats: AllocationStats::default(),
            adaptive_config: AdaptiveConfig::default(),
            hybrid_config: HybridConfig::default(),
            ml_config: None,
        }
    }

    pub fn with_adaptive_config(mut self, config: AdaptiveConfig) -> Self {
        self.adaptive_config = config;
        self
    }

    pub fn with_hybrid_config(mut self, config: HybridConfig) -> Self {
        self.hybrid_config = config;
        self
    }

    pub fn with_ml_config(mut self, config: MLConfig) -> Self {
        self.ml_config = Some(config);
        self
    }

    /// Find free block using the configured allocation strategy
    pub fn find_free_block(&mut self, size: usize) -> Option<*mut u8> {
        let start_time = Instant::now();

        let result = match self.strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::WorstFit => self.find_worst_fit(size),
            AllocationStrategy::BuddySystem => self.find_buddy_block(size),
            AllocationStrategy::SegregatedList => self.find_segregated_block(size),
            AllocationStrategy::Adaptive => self.find_adaptive_block(size),
            AllocationStrategy::MLBased => self.find_ml_based_block(size),
            AllocationStrategy::Hybrid => self.find_hybrid_block(size),
        };

        let latency_us = start_time.elapsed().as_micros() as u64;
        let cache_hit = result.is_some();

        self.stats.record_allocation(size, cache_hit, latency_us);
        self.allocation_history
            .push_back(AllocationEvent::new(size, cache_hit, latency_us));

        // Maintain history window
        if self.allocation_history.len() > self.adaptive_config.history_window {
            self.allocation_history.pop_front();
        }

        result
    }

    /// First-fit allocation: Find first block that fits
    pub fn find_first_fit(&mut self, size: usize) -> Option<*mut u8> {
        for (&block_size, blocks) in &mut self.free_blocks {
            if block_size >= size && !blocks.is_empty() {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    return Some(block.ptr);
                }
            }
        }
        None
    }

    /// Best-fit allocation: Find smallest block that fits
    pub fn find_best_fit(&mut self, size: usize) -> Option<*mut u8> {
        let mut best_size = None;
        let mut best_fit_size = usize::MAX;

        for (&block_size, blocks) in &self.free_blocks {
            if block_size >= size && block_size < best_fit_size && !blocks.is_empty() {
                best_fit_size = block_size;
                best_size = Some(block_size);
            }
        }

        if let Some(block_size) = best_size {
            if let Some(blocks) = self.free_blocks.get_mut(&block_size) {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    return Some(block.ptr);
                }
            }
        }

        None
    }

    /// Worst-fit allocation: Find largest block that fits (reduces fragmentation)
    pub fn find_worst_fit(&mut self, size: usize) -> Option<*mut u8> {
        let mut worst_size = None;
        let mut worst_fit_size = 0;

        for (&block_size, blocks) in &self.free_blocks {
            if block_size >= size && block_size > worst_fit_size && !blocks.is_empty() {
                worst_fit_size = block_size;
                worst_size = Some(block_size);
            }
        }

        if let Some(block_size) = worst_size {
            if let Some(blocks) = self.free_blocks.get_mut(&block_size) {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    return Some(block.ptr);
                }
            }
        }

        None
    }

    /// Buddy system allocation: Find power-of-2 sized block
    pub fn find_buddy_block(&mut self, size: usize) -> Option<*mut u8> {
        let buddy_size = size.next_power_of_two();

        if let Some(blocks) = self.free_blocks.get_mut(&buddy_size) {
            if let Some(mut block) = blocks.pop_front() {
                block.mark_used();
                return Some(block.ptr);
            }
        }

        None
    }

    /// Segregated list allocation: Different size classes
    pub fn find_segregated_block(&mut self, size: usize) -> Option<*mut u8> {
        let size_class = self.get_size_class(size);

        // Search from the appropriate size class upwards
        let mut search_sizes: Vec<usize> = self
            .free_blocks
            .keys()
            .filter(|&&s| s >= size_class)
            .cloned()
            .collect();
        search_sizes.sort();

        for class_size in search_sizes {
            if let Some(blocks) = self.free_blocks.get_mut(&class_size) {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    return Some(block.ptr);
                }
            }
        }

        None
    }

    /// Adaptive allocation based on allocation patterns and workload analysis
    pub fn find_adaptive_block(&mut self, size: usize) -> Option<*mut u8> {
        // Analyze recent allocation patterns
        let pattern = self.analyze_allocation_patterns();

        // Choose strategy based on pattern analysis
        let chosen_strategy = match pattern {
            AllocationPattern::SmallFrequent => AllocationStrategy::FirstFit,
            AllocationPattern::LargeInfrequent => AllocationStrategy::BestFit,
            AllocationPattern::Mixed => AllocationStrategy::WorstFit,
            AllocationPattern::Sequential => AllocationStrategy::SegregatedList,
            AllocationPattern::Random => AllocationStrategy::BuddySystem,
            AllocationPattern::Unknown => AllocationStrategy::BestFit,
        };

        // Apply the chosen strategy
        match chosen_strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::WorstFit => self.find_worst_fit(size),
            AllocationStrategy::SegregatedList => self.find_segregated_block(size),
            AllocationStrategy::BuddySystem => self.find_buddy_block(size),
            _ => self.find_best_fit(size), // Fallback
        }
    }

    /// ML-based allocation using learned patterns
    pub fn find_ml_based_block(&mut self, size: usize) -> Option<*mut u8> {
        if let Some(ml_config) = self.ml_config.clone() {
            // Extract features for ML prediction
            let features = self.extract_ml_features(size);

            // Make prediction (simplified - would use actual ML model)
            let prediction = self.predict_best_strategy(&features, &ml_config);

            // Apply predicted strategy if confidence is high enough
            if prediction.confidence >= ml_config.prediction_confidence_threshold as f64 {
                match prediction.strategy {
                    AllocationStrategy::FirstFit => self.find_first_fit(size),
                    AllocationStrategy::BestFit => self.find_best_fit(size),
                    AllocationStrategy::WorstFit => self.find_worst_fit(size),
                    AllocationStrategy::BuddySystem => self.find_buddy_block(size),
                    AllocationStrategy::SegregatedList => self.find_segregated_block(size),
                    _ => self.apply_fallback_strategy(size, &ml_config.fallback_strategy),
                }
            } else {
                // Fall back to configured fallback strategy
                self.apply_fallback_strategy(size, &ml_config.fallback_strategy)
            }
        } else {
            // No ML config, fall back to best fit
            self.find_best_fit(size)
        }
    }

    /// Hybrid allocation combining multiple strategies
    pub fn find_hybrid_block(&mut self, size: usize) -> Option<*mut u8> {
        // Evaluate current memory state
        let fragmentation_level = self.calculate_fragmentation_level();
        let utilization_level = self.calculate_utilization_level();

        // Choose primary or secondary strategy based on thresholds
        let chosen_strategy = if fragmentation_level
            > self.hybrid_config.switch_threshold_fragmentation as f64
            || utilization_level > self.hybrid_config.switch_threshold_utilization as f64
        {
            self.hybrid_config.secondary_strategy.clone()
        } else {
            self.hybrid_config.primary_strategy.clone()
        };

        // Apply chosen strategy
        self.apply_fallback_strategy(size, &chosen_strategy)
    }

    fn apply_fallback_strategy(
        &mut self,
        size: usize,
        strategy: &AllocationStrategy,
    ) -> Option<*mut u8> {
        match strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::WorstFit => self.find_worst_fit(size),
            AllocationStrategy::BuddySystem => self.find_buddy_block(size),
            AllocationStrategy::SegregatedList => self.find_segregated_block(size),
            AllocationStrategy::Adaptive => self.find_adaptive_block(size),
            _ => self.find_best_fit(size), // Ultimate fallback
        }
    }

    /// Analyze allocation patterns from history
    pub fn analyze_allocation_patterns(&self) -> AllocationPattern {
        if self.allocation_history.len() < 10 {
            return AllocationPattern::Unknown;
        }

        let recent_history: Vec<&AllocationEvent> =
            self.allocation_history.iter().rev().take(50).collect();

        // Analyze size distribution
        let sizes: Vec<usize> = recent_history.iter().map(|e| e.size).collect();
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        let small_count = sizes
            .iter()
            .filter(|&&s| s < self.adaptive_config.small_allocation_threshold)
            .count();
        let large_count = sizes
            .iter()
            .filter(|&&s| s > self.adaptive_config.large_allocation_threshold)
            .count();

        // Analyze temporal patterns
        let time_diffs: Vec<u128> = recent_history
            .windows(2)
            .map(|w| w[0].timestamp.duration_since(w[1].timestamp).as_millis())
            .collect();
        let avg_interval = if !time_diffs.is_empty() {
            time_diffs.iter().sum::<u128>() / time_diffs.len() as u128
        } else {
            0
        };

        // Pattern classification
        if small_count > recent_history.len() * 8 / 10 && avg_interval < 100 {
            AllocationPattern::SmallFrequent
        } else if large_count > recent_history.len() / 2 {
            AllocationPattern::LargeInfrequent
        } else if self.is_sequential_pattern(&sizes) {
            AllocationPattern::Sequential
        } else if self.is_random_pattern(&sizes) {
            AllocationPattern::Random
        } else {
            AllocationPattern::Mixed
        }
    }

    fn is_sequential_pattern(&self, sizes: &[usize]) -> bool {
        if sizes.len() < 3 {
            return false;
        }

        let mut increasing = 0;
        let mut decreasing = 0;

        for window in sizes.windows(2) {
            if window[1] > window[0] {
                increasing += 1;
            } else if window[1] < window[0] {
                decreasing += 1;
            }
        }

        // Consider sequential if more than 70% follow a trend
        let trend_ratio = (increasing.max(decreasing) as f64) / (sizes.len() - 1) as f64;
        trend_ratio > 0.7
    }

    fn is_random_pattern(&self, sizes: &[usize]) -> bool {
        if sizes.len() < 5 {
            return false;
        }

        // Calculate coefficient of variation
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let variance = sizes
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        // High coefficient of variation indicates randomness
        cv > 0.5
    }

    /// Extract features for ML-based allocation
    fn extract_ml_features(&self, size: usize) -> MLFeatures {
        let recent_history: Vec<&AllocationEvent> = self
            .allocation_history
            .iter()
            .rev()
            .take(
                self.ml_config
                    .as_ref()
                    .map(|c| c.feature_window)
                    .unwrap_or(20),
            )
            .collect();

        let avg_size = if !recent_history.is_empty() {
            recent_history.iter().map(|e| e.size).sum::<usize>() as f64
                / recent_history.len() as f64
        } else {
            0.0
        };

        let avg_latency = if !recent_history.is_empty() {
            recent_history.iter().map(|e| e.latency_us).sum::<u64>() as f64
                / recent_history.len() as f64
        } else {
            0.0
        };

        MLFeatures {
            requested_size: size as f64,
            avg_recent_size: avg_size,
            avg_recent_latency: avg_latency,
            cache_hit_rate: self.stats.get_cache_hit_rate(),
            fragmentation_level: self.calculate_fragmentation_level(),
            utilization_level: self.calculate_utilization_level(),
            allocation_frequency: recent_history.len() as f64,
        }
    }

    /// Predict best allocation strategy using ML
    fn predict_best_strategy(&self, features: &MLFeatures, ml_config: &MLConfig) -> MLPrediction {
        // Simplified ML prediction - in real implementation would use trained model
        let score_first_fit =
            self.score_strategy_for_features(features, &AllocationStrategy::FirstFit);
        let score_best_fit =
            self.score_strategy_for_features(features, &AllocationStrategy::BestFit);
        let score_worst_fit =
            self.score_strategy_for_features(features, &AllocationStrategy::WorstFit);
        let score_buddy =
            self.score_strategy_for_features(features, &AllocationStrategy::BuddySystem);
        let score_segregated =
            self.score_strategy_for_features(features, &AllocationStrategy::SegregatedList);

        let mut best_strategy = AllocationStrategy::BestFit;
        let mut _best_score = score_best_fit;
        let mut confidence = 0.5;

        if score_first_fit > _best_score {
            best_strategy = AllocationStrategy::FirstFit;
            _best_score = score_first_fit;
        }
        if score_worst_fit > _best_score {
            best_strategy = AllocationStrategy::WorstFit;
            _best_score = score_worst_fit;
        }
        if score_buddy > _best_score {
            best_strategy = AllocationStrategy::BuddySystem;
            _best_score = score_buddy;
        }
        if score_segregated > _best_score {
            best_strategy = AllocationStrategy::SegregatedList;
            _best_score = score_segregated;
        }

        // Calculate confidence based on score difference
        let mut scores = [
            score_first_fit,
            score_best_fit,
            score_worst_fit,
            score_buddy,
            score_segregated,
        ];
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if scores.len() >= 2 {
            confidence = (scores[0] - scores[1]).clamp(0.0, 1.0);
        }

        MLPrediction {
            strategy: best_strategy,
            confidence,
            predicted_latency: features.avg_recent_latency,
        }
    }

    fn score_strategy_for_features(
        &self,
        features: &MLFeatures,
        strategy: &AllocationStrategy,
    ) -> f64 {
        // Simplified scoring function - would be learned from data
        match strategy {
            AllocationStrategy::FirstFit => {
                // First fit is good for high frequency, small allocations
                let size_score = if features.requested_size < 4096.0 {
                    0.8
                } else {
                    0.3
                };
                let freq_score = if features.allocation_frequency > 10.0 {
                    0.9
                } else {
                    0.4
                };
                (size_score + freq_score) / 2.0
            }
            AllocationStrategy::BestFit => {
                // Best fit is good for memory efficiency
                let util_score = if features.utilization_level > 0.7 {
                    0.9
                } else {
                    0.6
                };
                let frag_score = if features.fragmentation_level < 0.3 {
                    0.8
                } else {
                    0.4
                };
                (util_score + frag_score) / 2.0
            }
            AllocationStrategy::WorstFit => {
                // Worst fit is good for reducing fragmentation

                if features.fragmentation_level > 0.4 {
                    0.8
                } else {
                    0.3
                }
            }
            AllocationStrategy::BuddySystem => {
                // Buddy system is good for power-of-2 sizes

                if features.requested_size.log2().fract() < 0.1 {
                    0.9
                } else {
                    0.4
                }
            }
            AllocationStrategy::SegregatedList => {
                // Segregated lists are good for diverse sizes with patterns

                if features.cache_hit_rate > 0.6 {
                    0.7
                } else {
                    0.5
                }
            }
            _ => 0.5, // Default score
        }
    }

    fn calculate_fragmentation_level(&self) -> f64 {
        // Simplified fragmentation calculation
        if self.free_blocks.is_empty() {
            return 0.0;
        }

        let total_free_space: usize = self
            .free_blocks
            .iter()
            .map(|(size, blocks)| size * blocks.len())
            .sum();

        let free_block_count: usize = self.free_blocks.values().map(|blocks| blocks.len()).sum();

        if total_free_space == 0 {
            0.0
        } else {
            let avg_block_size = total_free_space as f64 / free_block_count as f64;
            let fragmentation = 1.0 - (avg_block_size / total_free_space as f64);
            fragmentation.clamp(0.0, 1.0)
        }
    }

    fn calculate_utilization_level(&self) -> f64 {
        // Simplified utilization calculation based on allocation stats
        let total_capacity = self.stats.peak_allocated_bytes as f64;
        let current_allocated = self.stats.total_allocated_bytes as f64;

        if total_capacity == 0.0 {
            0.0
        } else {
            (current_allocated / total_capacity).clamp(0.0, 1.0)
        }
    }

    /// Get size class for segregated list allocation
    pub fn get_size_class(&self, size: usize) -> usize {
        match size {
            0..=256 => 256,
            257..=512 => 512,
            513..=1024 => 1024,
            1025..=2048 => 2048,
            2049..=4096 => 4096,
            4097..=8192 => 8192,
            8193..=16384 => 16384,
            16385..=32768 => 32768,
            32769..=65536 => 65536,
            65537..=131072 => 131072,
            131073..=262144 => 262144,
            262145..=524288 => 524288,
            524289..=1048576 => 1048576,
            _ => size.next_power_of_two(),
        }
    }

    /// Add free block to the pool
    pub fn add_free_block(&mut self, block: MemoryBlock) {
        let size_class = self.get_size_class(block.size);
        self.free_blocks
            .entry(size_class)
            .or_default()
            .push_back(block);
    }

    /// Remove free block from the pool
    pub fn remove_free_block(&mut self, size: usize, ptr: *mut u8) -> Option<MemoryBlock> {
        let size_class = self.get_size_class(size);
        if let Some(blocks) = self.free_blocks.get_mut(&size_class) {
            if let Some(pos) = blocks.iter().position(|block| block.ptr == ptr) {
                return blocks.remove(pos);
            }
        }
        None
    }

    /// Get current allocation statistics
    pub fn get_stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// Get current allocation strategy
    pub fn get_strategy(&self) -> &AllocationStrategy {
        &self.strategy
    }

    /// Set new allocation strategy
    pub fn set_strategy(&mut self, strategy: AllocationStrategy) {
        self.strategy = strategy;
    }

    /// Clear allocation history
    pub fn clear_history(&mut self) {
        self.allocation_history.clear();
    }

    /// Get allocation history
    pub fn get_history(&self) -> &VecDeque<AllocationEvent> {
        &self.allocation_history
    }
}

/// Allocation pattern analysis results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocationPattern {
    SmallFrequent,
    LargeInfrequent,
    Mixed,
    Sequential,
    Random,
    Unknown,
}

/// Features for ML-based allocation
#[derive(Debug, Clone)]
pub struct MLFeatures {
    pub requested_size: f64,
    pub avg_recent_size: f64,
    pub avg_recent_latency: f64,
    pub cache_hit_rate: f64,
    pub fragmentation_level: f64,
    pub utilization_level: f64,
    pub allocation_frequency: f64,
}

/// ML prediction result
#[derive(Debug, Clone)]
pub struct MLPrediction {
    pub strategy: AllocationStrategy,
    pub confidence: f64,
    pub predicted_latency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_strategies() {
        let mut manager = AllocationStrategyManager::new(AllocationStrategy::BestFit);

        // Add some free blocks
        for i in 0..5 {
            let block = MemoryBlock::new((i * 1024) as *mut u8, 1024 * (i + 1));
            manager.add_free_block(block);
        }

        // Test best fit allocation
        let ptr = manager.find_free_block(1500);
        assert!(ptr.is_some());

        // Test statistics
        let stats = manager.get_stats();
        assert_eq!(stats.total_allocations, 1);
    }

    #[test]
    fn test_size_classes() {
        let manager = AllocationStrategyManager::new(AllocationStrategy::SegregatedList);

        assert_eq!(manager.get_size_class(100), 256);
        assert_eq!(manager.get_size_class(300), 512);
        assert_eq!(manager.get_size_class(1000), 1024);
        assert_eq!(manager.get_size_class(2000000), 2097152);
    }

    #[test]
    fn test_adaptive_strategy() {
        let mut manager = AllocationStrategyManager::new(AllocationStrategy::Adaptive);

        // Simulate small frequent allocations
        for _ in 0..20 {
            let event = AllocationEvent::new(256, true, 10);
            manager.allocation_history.push_back(event);
        }

        let pattern = manager.analyze_allocation_patterns();
        assert_eq!(pattern, AllocationPattern::SmallFrequent);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let mut manager = AllocationStrategyManager::new(AllocationStrategy::BestFit);

        // Add blocks of different sizes
        manager.add_free_block(MemoryBlock::new(0x1000 as *mut u8, 1024));
        manager.add_free_block(MemoryBlock::new(0x2000 as *mut u8, 2048));
        manager.add_free_block(MemoryBlock::new(0x3000 as *mut u8, 512));

        let fragmentation = manager.calculate_fragmentation_level();
        assert!((0.0..=1.0).contains(&fragmentation));
    }
}
