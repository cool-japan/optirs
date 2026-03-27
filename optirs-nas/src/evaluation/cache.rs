//! Evaluation cache for storing and retrieving evaluation results
//!
//! Provides caching mechanisms to avoid redundant evaluations.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

use super::types::*;
use crate::nas_engine::results::EvaluationResults;

/// Evaluation cache for storing results
#[derive(Debug)]
pub struct EvaluationCache<T: Float + Debug + Send + Sync + 'static> {
    /// Cached evaluations
    evaluations: HashMap<String, CachedEvaluation<T>>,

    /// Cache metadata
    metadata: CacheMetadata,

    /// Access patterns
    access_patterns: AccessPatterns,
}

/// Cached evaluation result
#[derive(Debug, Clone)]
pub struct CachedEvaluation<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation results
    pub results: EvaluationResults<T>,

    /// Cache timestamp
    pub timestamp: SystemTime,

    /// Access count
    pub access_count: usize,

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
        Self {
            evaluations: HashMap::new(),
            metadata: CacheMetadata {
                total_entries: 0,
                cache_size_bytes: 0,
                last_cleanup: SystemTime::now(),
                version: "1.0.0".to_string(),
            },
            access_patterns: AccessPatterns {
                frequency_distribution: HashMap::new(),
                temporal_patterns: Vec::new(),
                correlation_patterns: HashMap::new(),
            },
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&CachedEvaluation<T>> {
        self.evaluations.get(key)
    }

    pub(crate) fn insert(&mut self, key: String, results: EvaluationResults<T>) {
        let cached_eval = CachedEvaluation {
            results,
            timestamp: SystemTime::now(),
            access_count: 1,
            is_valid: true,
        };

        self.evaluations.insert(key, cached_eval);
        self.metadata.total_entries += 1;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheMetadata {
        &self.metadata
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.evaluations.clear();
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

    #[test]
    fn test_evaluation_cache() {
        let mut cache = EvaluationCache::<f64>::new();
        assert_eq!(cache.metadata.total_entries, 0);

        let results = EvaluationResults {
            metric_scores: std::collections::HashMap::new(),
            overall_score: 0.95,
            confidence_intervals: std::collections::HashMap::new(),
            evaluation_time: std::time::Duration::from_secs(100),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: std::collections::HashMap::new(),
            training_trajectory: vec![],
        };

        cache.insert("test_key".to_string(), results);
        assert_eq!(cache.metadata.total_entries, 1);
        assert!(cache.contains("test_key"));
    }
}
