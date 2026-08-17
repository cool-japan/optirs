//! Evaluation cache for storing and retrieving evaluation results
//!
//! Provides caching mechanisms to avoid redundant evaluations.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::*;
use crate::nas_engine::results::EvaluationResults;

/// Default maximum number of entries retained by an [`EvaluationCache`].
///
/// A NAS run evaluates thousands of architectures; without a bound the cache
/// grows for the lifetime of the search. The default keeps a generous working
/// set while making memory use predictable.
pub const DEFAULT_CACHE_CAPACITY: usize = 4096;

/// Evaluation cache for storing results
///
/// Entries are evicted according to [`CacheEvictionPolicy`] once the cache
/// exceeds its capacity. The default policy is LRU: the entry whose last access
/// is oldest is dropped first.
#[derive(Debug)]
pub struct EvaluationCache<T: Float + Debug + Send + Sync + 'static> {
    /// Cached evaluations
    evaluations: HashMap<String, CachedEvaluation<T>>,

    /// Cache metadata
    metadata: CacheMetadata,

    /// Access patterns
    access_patterns: AccessPatterns,

    /// Maximum number of retained entries.
    capacity: usize,

    /// Eviction policy applied when `capacity` is exceeded.
    eviction_policy: CacheEvictionPolicy,

    /// Monotonic access counter used as a logical clock for recency ordering.
    /// Avoids depending on wall-clock resolution, which is too coarse to order
    /// accesses that happen within the same instant.
    access_clock: u64,
}

/// Cached evaluation result
#[derive(Debug, Clone)]
pub struct CachedEvaluation<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation results
    pub results: EvaluationResults<T>,

    /// Cache timestamp (insertion time)
    pub timestamp: SystemTime,

    /// Number of times this entry has been read out of the cache.
    ///
    /// Starts at zero on insertion and is incremented on every successful
    /// lookup; it therefore reflects real reuse and drives the LFU policy.
    pub access_count: usize,

    /// Logical clock value of the most recent access, used for LRU ordering.
    pub last_access_tick: u64,

    /// Validity flag
    pub is_valid: bool,
}

/// Cache metadata
#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// Total entries
    pub total_entries: usize,

    /// Cache size (bytes)
    pub cache_size_bytes: usize,

    /// Last cleanup time
    pub last_cleanup: SystemTime,

    /// Cache version
    pub version: String,
}

/// Access patterns for cache optimization
#[derive(Debug)]
pub struct AccessPatterns {
    /// Frequency distribution
    frequency_distribution: HashMap<String, usize>,

    /// Temporal patterns
    temporal_patterns: Vec<TemporalPattern>,

    /// Correlation patterns
    correlation_patterns: HashMap<String, Vec<String>>,
}

/// Temporal access pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Time window
    time_window: Duration,

    /// Access frequency
    access_frequency: f64,

    /// Pattern type
    pattern_type: TemporalPatternType,
}

impl<T: Float + Debug + Default + Send + Sync> EvaluationCache<T> {
    pub(crate) fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY, CacheEvictionPolicy::LRU)
    }

    /// Create a cache with an explicit capacity and eviction policy.
    ///
    /// A capacity of zero is promoted to one so the cache always retains the
    /// most recent insertion.
    pub fn with_capacity(capacity: usize, eviction_policy: CacheEvictionPolicy) -> Self {
        Self {
            evaluations: HashMap::new(),
            metadata: CacheMetadata {
                total_entries: 0,
                cache_size_bytes: 0,
                last_cleanup: SystemTime::now(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            access_patterns: AccessPatterns {
                frequency_distribution: HashMap::new(),
                temporal_patterns: Vec::new(),
                correlation_patterns: HashMap::new(),
            },
            capacity: capacity.max(1),
            eviction_policy,
            access_clock: 0,
        }
    }

    /// Look up a cached evaluation, recording the access.
    ///
    /// Takes `&mut self` because a hit updates the entry's access count and
    /// recency stamp, which is what makes the eviction policies meaningful.
    pub(crate) fn get(&mut self, key: &str) -> Option<&CachedEvaluation<T>> {
        self.access_clock = self.access_clock.wrapping_add(1);
        let tick = self.access_clock;

        let entry = self.evaluations.get_mut(key)?;
        if !entry.is_valid {
            return None;
        }
        entry.access_count += 1;
        entry.last_access_tick = tick;

        *self
            .access_patterns
            .frequency_distribution
            .entry(key.to_string())
            .or_insert(0) += 1;

        self.evaluations.get(key)
    }

    pub(crate) fn insert(&mut self, key: String, results: EvaluationResults<T>) {
        self.access_clock = self.access_clock.wrapping_add(1);

        let cached_eval = CachedEvaluation {
            results,
            timestamp: SystemTime::now(),
            access_count: 0,
            last_access_tick: self.access_clock,
            is_valid: true,
        };

        self.evaluations.insert(key, cached_eval);
        self.evict_if_needed();
        self.metadata.total_entries = self.evaluations.len();
    }

    /// Drop entries until the cache is back inside its capacity, honouring the
    /// configured [`CacheEvictionPolicy`].
    fn evict_if_needed(&mut self) {
        while self.evaluations.len() > self.capacity {
            let victim = match self.eviction_policy {
                // Least recently used: smallest access tick.
                CacheEvictionPolicy::LRU => self
                    .evaluations
                    .iter()
                    .min_by_key(|(_, v)| v.last_access_tick)
                    .map(|(k, _)| k.clone()),
                // Least frequently used: fewest reads, ties broken by recency.
                CacheEvictionPolicy::LFU => self
                    .evaluations
                    .iter()
                    .min_by_key(|(_, v)| (v.access_count, v.last_access_tick))
                    .map(|(k, _)| k.clone()),
                // First in, first out: oldest insertion timestamp.
                CacheEvictionPolicy::FIFO => self
                    .evaluations
                    .iter()
                    .min_by_key(|(_, v)| v.timestamp)
                    .map(|(k, _)| k.clone()),
                // Deterministic stand-in for random selection: the smallest key
                // by ordering. Avoids pulling an RNG into the cache while still
                // being independent of access history.
                CacheEvictionPolicy::Random => self.evaluations.keys().min().map(|k| k.clone()),
            };

            match victim {
                Some(key) => {
                    self.evaluations.remove(&key);
                    self.access_patterns.frequency_distribution.remove(&key);
                }
                None => break,
            }
        }

        self.metadata.last_cleanup = SystemTime::now();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheMetadata {
        &self.metadata
    }

    /// Maximum number of entries this cache retains.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total number of cache reads recorded across all live entries.
    pub fn total_accesses(&self) -> usize {
        self.evaluations.values().map(|v| v.access_count).sum()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.evaluations.clear();
        self.access_patterns.frequency_distribution.clear();
        self.metadata.total_entries = 0;
        self.metadata.cache_size_bytes = 0;
        self.metadata.last_cleanup = SystemTime::now();
    }

    /// Check if cache contains a key
    pub fn contains(&self, key: &str) -> bool {
        self.evaluations.contains_key(key)
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn results(score: f64) -> EvaluationResults<f64> {
        EvaluationResults {
            metric_scores: std::collections::HashMap::new(),
            overall_score: score,
            confidence_intervals: std::collections::HashMap::new(),
            evaluation_time: std::time::Duration::from_secs(100),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: std::collections::HashMap::new(),
            training_trajectory: vec![],
        }
    }

    #[test]
    fn test_evaluation_cache() {
        let mut cache = EvaluationCache::<f64>::new();
        assert_eq!(cache.metadata.total_entries, 0);

        cache.insert("test_key".to_string(), results(0.95));
        assert_eq!(cache.metadata.total_entries, 1);
        assert!(cache.contains("test_key"));
    }

    #[test]
    fn test_access_count_is_incremented_on_hits() {
        let mut cache = EvaluationCache::<f64>::new();
        cache.insert("k".to_string(), results(0.5));

        // Freshly inserted entries have never been read.
        assert_eq!(cache.total_accesses(), 0);

        for expected in 1..=3 {
            let hit = cache.get("k").expect("hit");
            assert_eq!(hit.access_count, expected);
        }
        assert_eq!(cache.total_accesses(), 3);
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_lru_eviction_bounds_the_cache() {
        let mut cache = EvaluationCache::<f64>::with_capacity(2, CacheEvictionPolicy::LRU);
        cache.insert("a".to_string(), results(0.1));
        cache.insert("b".to_string(), results(0.2));

        // Touch "a" so "b" becomes the least recently used entry.
        assert!(cache.get("a").is_some());

        cache.insert("c".to_string(), results(0.3));

        assert_eq!(cache.len(), 2, "cache must stay within capacity");
        assert!(cache.contains("a"));
        assert!(cache.contains("c"));
        assert!(!cache.contains("b"), "LRU victim must be evicted");
    }

    #[test]
    fn test_lfu_eviction_keeps_hot_entries() {
        let mut cache = EvaluationCache::<f64>::with_capacity(2, CacheEvictionPolicy::LFU);
        cache.insert("hot".to_string(), results(0.1));
        cache.insert("cold".to_string(), results(0.2));

        for _ in 0..5 {
            assert!(cache.get("hot").is_some());
        }

        cache.insert("new".to_string(), results(0.3));

        assert_eq!(cache.len(), 2);
        assert!(cache.contains("hot"));
        assert!(!cache.contains("cold"));
    }

    #[test]
    fn test_cache_never_exceeds_capacity_under_load() {
        let mut cache = EvaluationCache::<f64>::with_capacity(8, CacheEvictionPolicy::LRU);
        for i in 0..1000 {
            cache.insert(format!("key_{}", i), results(i as f64));
            assert!(cache.len() <= 8);
        }
        assert_eq!(cache.len(), 8);
        assert_eq!(cache.stats().total_entries, 8);
    }
}
