// Episodic Memory Implementation
//
// Implements advanced methods for EpisodicMemoryBank and SupportSetManager
// types defined in crate::few_shot.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::few_shot::{EpisodicMemoryBank, MemoryBankStats, SupportSetManager};

// ---------------------------------------------------------------------------
// EpisodicMemoryBank additional impl
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static> EpisodicMemoryBank<T> {
    /// Store an episode keyed by task id with a representation vector and
    /// performance score.
    ///
    /// If the bank is at capacity, the eviction policy is applied first.
    pub fn store_lightweight_episode(
        &mut self,
        task_id: String,
        representation: Array1<T>,
        performance: T,
    ) -> Result<()> {
        use crate::few_shot::{
            AdaptationPerformance, AdaptationResult, AdaptationStep, DifficultyLevel,
            DomainCharacteristics, DomainInfo, DomainType, EpisodeMetadata, ExampleMetadata,
            MemoryEpisode, QueryExample, QuerySet, QuerySetStatistics, ResourceUsage,
            SupportExample, SupportSet, SupportSetStatistics, TaskData, TaskMetadata,
        };
        use std::collections::HashMap;
        use std::time::Duration;

        // Evict if at capacity
        if self.episodes().len() >= self.capacity() {
            self.evict()?;
        }

        let dim = representation.len();

        // Build a minimal TaskData wrapping the representation
        let support_example = SupportExample {
            features: representation.clone(),
            target: performance,
            weight: T::one(),
            context: HashMap::new(),
            metadata: ExampleMetadata {
                source: task_id.clone(),
                quality_score: scirs2_core::numeric::NumCast::from(performance).unwrap_or(0.0),
                created_at: std::time::SystemTime::now(),
            },
        };

        let support_set = SupportSet {
            examples: vec![support_example],
            task_metadata: TaskMetadata {
                task_name: task_id.clone(),
                domain: DomainType::Optimization,
                difficulty: DifficultyLevel::Medium,
                created_at: std::time::SystemTime::now(),
            },
            statistics: SupportSetStatistics {
                mean: representation.clone(),
                variance: Array1::zeros(dim),
                size: 1,
                diversity_score: T::zero(),
            },
            temporal_order: None,
        };

        let query_set = QuerySet {
            examples: Vec::<QueryExample<T>>::new(),
            statistics: QuerySetStatistics {
                mean: Array1::zeros(dim),
                variance: Array1::zeros(dim),
                size: 0,
            },
            eval_metrics: Vec::new(),
        };

        let task_data = TaskData {
            task_id: task_id.clone(),
            support_set,
            query_set,
            task_params: HashMap::new(),
            domain_info: DomainInfo {
                domain_type: DomainType::Optimization,
                characteristics: DomainCharacteristics {
                    input_dim: dim,
                    output_dim: 1,
                    temporal: false,
                    stochasticity: 0.0,
                    noise_level: 0.0,
                    sparsity: 0.0,
                },
                difficulty_level: DifficultyLevel::Medium,
                constraints: Vec::new(),
            },
        };

        let adaptation_result = AdaptationResult {
            adapted_state: crate::OptimizerState {
                parameters: Array1::zeros(1),
                gradients: Array1::zeros(1),
                momentum: None,
                hidden_states: HashMap::new(),
                memory_buffers: HashMap::new(),
                step: 0,
                step_count: 0,
                loss: None,
                learning_rate: scirs2_core::numeric::NumCast::from(0.001)
                    .unwrap_or_else(|| T::one()),
                metadata: crate::StateMetadata {
                    task_id: Some(task_id.clone()),
                    optimizer_type: None,
                    version: "1.0".to_string(),
                    timestamp: std::time::SystemTime::now(),
                    checksum: 0,
                    compression_level: 0,
                    custom_data: HashMap::new(),
                },
            },
            performance: AdaptationPerformance {
                query_performance: performance,
                support_performance: performance,
                adaptation_speed: 1,
                final_loss: T::one() - performance,
                improvement: performance,
                stability: T::one(),
            },
            task_representation: representation,
            adaptation_trajectory: Vec::<AdaptationStep<T>>::new(),
            resource_usage: ResourceUsage {
                total_time: Duration::from_secs(0),
                peak_memory_mb: T::zero(),
                compute_cost: T::zero(),
                energy_consumption: T::zero(),
            },
        };

        let episode = MemoryEpisode {
            episode_id: format!("ep_{}", self.usage_stats().total_episodes),
            task_data,
            adaptation_result,
            timestamp: std::time::SystemTime::now(),
            metadata: EpisodeMetadata {
                difficulty: DifficultyLevel::Medium,
                domain: DomainType::Optimization,
                success_rate: scirs2_core::numeric::NumCast::from(performance).unwrap_or(0.0),
                tags: Vec::new(),
            },
            access_count: 0,
        };

        self.episodes_mut().push_back(episode);
        self.usage_stats_mut().total_episodes += 1;
        let len = self.episodes().len();
        let cap = self.capacity();
        self.usage_stats_mut().memory_utilization = len as f64 / cap as f64;
        Ok(())
    }

    /// Retrieve the k nearest episodes to a query representation vector.
    ///
    /// Returns `Vec<(task_id, similarity)>` sorted by descending similarity
    /// (cosine similarity).
    pub fn retrieve_by_repr(&self, query: &Array1<T>, k: usize) -> Result<Vec<(String, T)>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(usize, T)> = Vec::with_capacity(self.len());
        for (idx, ep) in self.episodes().iter().enumerate() {
            let repr = &ep.adaptation_result.task_representation;
            let sim = cosine_similarity(query, repr);
            scored.push((idx, sim));
        }

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let take = k.min(scored.len());
        let result: Vec<(String, T)> = scored[..take]
            .iter()
            .map(|&(idx, sim)| {
                let ep = &self.episodes()[idx];
                (ep.episode_id.clone(), sim)
            })
            .collect();

        Ok(result)
    }

    /// Evict the least useful episode according to the eviction policy.
    ///
    /// For the Performance policy, removes the episode with the lowest
    /// query performance. For others, falls back to removing the oldest.
    pub fn evict(&mut self) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        match self.eviction_policy() {
            crate::few_shot::EvictionPolicy::Performance => {
                // Find the episode with the worst performance
                let mut worst_idx = 0;
                let mut worst_perf = T::infinity();
                for (i, ep) in self.episodes().iter().enumerate() {
                    let perf = ep.adaptation_result.performance.query_performance;
                    if perf < worst_perf {
                        worst_perf = perf;
                        worst_idx = i;
                    }
                }
                self.episodes_mut().remove(worst_idx);
            }
            crate::few_shot::EvictionPolicy::LRU => {
                // Remove least recently used (lowest access_count among oldest)
                let mut lru_idx = 0;
                let mut min_access = usize::MAX;
                for (i, ep) in self.episodes().iter().enumerate() {
                    if ep.access_count < min_access {
                        min_access = ep.access_count;
                        lru_idx = i;
                    }
                }
                self.episodes_mut().remove(lru_idx);
            }
            crate::few_shot::EvictionPolicy::LFU => {
                // Least frequently used
                let mut lfu_idx = 0;
                let mut min_access = usize::MAX;
                for (i, ep) in self.episodes().iter().enumerate() {
                    if ep.access_count < min_access {
                        min_access = ep.access_count;
                        lfu_idx = i;
                    }
                }
                self.episodes_mut().remove(lfu_idx);
            }
            _ => {
                // Default: remove oldest (front of deque)
                self.episodes_mut().pop_front();
            }
        }

        let len = self.episodes().len();
        let cap = self.capacity();
        self.usage_stats_mut().memory_utilization = len as f64 / cap as f64;
        Ok(())
    }

    /// Get summary statistics about the memory bank.
    pub fn get_stats(&self) -> MemoryBankStats<T> {
        let count = self.len();
        let cap = self.capacity();

        let avg_performance = if count == 0 {
            T::zero()
        } else {
            let mut sum = T::zero();
            for ep in self.episodes() {
                sum = sum + ep.adaptation_result.performance.query_performance;
            }
            let count_t: T = scirs2_core::numeric::NumCast::from(count).unwrap_or_else(|| T::one());
            sum / count_t
        };

        MemoryBankStats {
            count,
            avg_performance,
            capacity_used: if cap > 0 {
                count as f64 / cap as f64
            } else {
                0.0
            },
            total_capacity: cap,
        }
    }

    /// Remove all episodes.
    pub fn clear(&mut self) {
        self.episodes_mut().clear();
        self.usage_stats_mut().memory_utilization = 0.0;
    }

    /// Return the number of stored episodes (alias for len).
    pub fn size(&self) -> usize {
        self.len()
    }
}

// ---------------------------------------------------------------------------
// SupportSetManager additional impl
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Send + Sync + 'static> SupportSetManager<T> {
    /// Select a diverse subset of candidate indices for a support set.
    ///
    /// Uses a greedy farthest-point sampling strategy: the first point is
    /// selected as the one with the largest norm, then each subsequent point
    /// is the one that is most distant from all already-selected points
    /// (measured by squared Euclidean distance).
    pub fn select_support_set(
        &self,
        candidates: &[Array1<T>],
        _labels: &[T],
        budget: usize,
    ) -> Result<Vec<usize>> {
        if candidates.is_empty() {
            return Err(OptimError::InsufficientData(
                "No candidates to select from".to_string(),
            ));
        }
        let n = candidates.len();
        let take = budget.min(n).min(self.max_support_size());

        if take >= n {
            return Ok((0..n).collect());
        }

        // Greedy farthest-point sampling
        let mut selected: Vec<usize> = Vec::with_capacity(take);

        // Pick the candidate with the largest norm as the seed
        let mut best_seed = 0;
        let mut best_norm = T::neg_infinity();
        for (i, c) in candidates.iter().enumerate() {
            let norm = vec_norm_sq(c);
            if norm > best_norm {
                best_norm = norm;
                best_seed = i;
            }
        }
        selected.push(best_seed);

        // Track min distance from each candidate to any selected point
        let mut min_dist: Vec<T> = vec![T::infinity(); n];

        while selected.len() < take {
            // Update min_dist using the last selected point
            let last = selected[selected.len() - 1];
            for i in 0..n {
                let d = squared_euclidean(&candidates[i], &candidates[last]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
            // Zero out already-selected points
            for &s in &selected {
                min_dist[s] = T::neg_infinity();
            }

            // Pick the point with the maximum min_dist
            let mut farthest_idx = 0;
            let mut farthest_dist = T::neg_infinity();
            for (i, &dist) in min_dist.iter().enumerate().take(n) {
                if dist > farthest_dist {
                    farthest_dist = dist;
                    farthest_idx = i;
                }
            }
            selected.push(farthest_idx);
        }

        Ok(selected)
    }

    /// Augment a support set by adding Gaussian noise to each example.
    ///
    /// For each input vector, produces a copy with noise ~ N(0, noise_scale^2)
    /// added to each element (using a simple deterministic hash-based approach
    /// for reproducibility without requiring rand).
    pub fn augment_support_set(
        &self,
        support: &[Array1<T>],
        noise_scale: T,
    ) -> Result<Vec<Array1<T>>> {
        if support.is_empty() {
            return Err(OptimError::InsufficientData(
                "Cannot augment empty support set".to_string(),
            ));
        }

        let mut augmented = Vec::with_capacity(support.len() * 2);

        // Keep originals
        for s in support {
            augmented.push(s.clone());
        }

        // Create augmented copies with deterministic pseudo-noise
        for (ex_idx, s) in support.iter().enumerate() {
            let mut noisy = s.clone();
            for (i, val) in noisy.iter_mut().enumerate() {
                // Simple deterministic hash-based noise for reproducibility
                let seed = (ex_idx * 7919 + i * 104729 + 31) as f64;
                let noise_val = ((seed * 0.6180339887).fract() - 0.5) * 2.0; // in [-1, 1]
                let noise_t: T =
                    scirs2_core::numeric::NumCast::from(noise_val).unwrap_or_else(|| T::zero());
                *val = *val + noise_scale * noise_t;
            }
            augmented.push(noisy);
        }

        Ok(augmented)
    }

    /// Evaluate the quality/diversity of a support set.
    ///
    /// Returns the average pairwise squared Euclidean distance between all
    /// support vectors (higher = more diverse = better quality).
    pub fn evaluate_quality(&self, support: &[Array1<T>]) -> Result<T> {
        if support.len() < 2 {
            return Ok(T::zero());
        }

        let n = support.len();
        let mut total_dist = T::zero();
        let mut pair_count = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                total_dist = total_dist + squared_euclidean(&support[i], &support[j]);
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return Ok(T::zero());
        }

        let pair_t: T = scirs2_core::numeric::NumCast::from(pair_count).unwrap_or_else(|| T::one());
        Ok(total_dist / pair_t)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
fn cosine_similarity<T: Float>(a: &Array1<T>, b: &Array1<T>) -> T {
    let len = a.len().min(b.len());
    let mut dot = T::zero();
    let mut na = T::zero();
    let mut nb = T::zero();
    for i in 0..len {
        dot = dot + a[i] * b[i];
        na = na + a[i] * a[i];
        nb = nb + b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == T::zero() {
        T::zero()
    } else {
        dot / denom
    }
}

/// Squared Euclidean distance.
fn squared_euclidean<T: Float>(a: &Array1<T>, b: &Array1<T>) -> T {
    let len = a.len().min(b.len());
    let mut sum = T::zero();
    for i in 0..len {
        let d = a[i] - b[i];
        sum = sum + d * d;
    }
    sum
}

/// Squared L2 norm.
fn vec_norm_sq<T: Float>(v: &Array1<T>) -> T {
    let mut sum = T::zero();
    for &x in v.iter() {
        sum = sum + x * x;
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_episodic_memory_store_retrieve() {
        let mut bank = EpisodicMemoryBank::<f64>::from_capacity(10)
            .expect("failed to create EpisodicMemoryBank");
        assert_eq!(bank.size(), 0);

        bank.store_lightweight_episode(
            "task_a".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            0.9,
        )
        .expect("store failed");
        bank.store_lightweight_episode(
            "task_b".to_string(),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
            0.7,
        )
        .expect("store failed");
        assert_eq!(bank.size(), 2);

        // Retrieve similar to [1, 0, 0] should return task_a first
        let results = bank
            .retrieve_by_repr(&Array1::from_vec(vec![1.0, 0.1, 0.0]), 2)
            .expect("retrieve failed");
        assert_eq!(results.len(), 2);
        // First result should be ep_0 (task_a), which is closer to the query
        assert_eq!(results[0].0, "ep_0");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_episodic_memory_eviction() {
        let mut bank = EpisodicMemoryBank::<f64>::from_capacity(3)
            .expect("failed to create EpisodicMemoryBank");

        bank.store_lightweight_episode("t1".into(), Array1::from_vec(vec![1.0]), 0.5)
            .expect("store failed");
        bank.store_lightweight_episode("t2".into(), Array1::from_vec(vec![2.0]), 0.9)
            .expect("store failed");
        bank.store_lightweight_episode("t3".into(), Array1::from_vec(vec![3.0]), 0.3)
            .expect("store failed");
        assert_eq!(bank.size(), 3);

        // Storing a 4th should trigger eviction (worst performance = t3 with 0.3)
        bank.store_lightweight_episode("t4".into(), Array1::from_vec(vec![4.0]), 0.8)
            .expect("store failed");
        assert_eq!(bank.size(), 3);

        // The episode with performance 0.3 should have been evicted
        let has_low_perf = bank.episodes().iter().any(|ep| {
            let perf = ep.adaptation_result.performance.query_performance;
            (perf - 0.3).abs() < 1e-12
        });
        assert!(
            !has_low_perf,
            "lowest-performance episode should be evicted"
        );
    }

    #[test]
    fn test_memory_bank_stats() {
        let mut bank = EpisodicMemoryBank::<f64>::from_capacity(10)
            .expect("failed to create EpisodicMemoryBank");

        let stats = bank.get_stats();
        assert_eq!(stats.count, 0);
        assert!((stats.avg_performance - 0.0).abs() < 1e-12);
        assert!((stats.capacity_used - 0.0).abs() < 1e-12);
        assert_eq!(stats.total_capacity, 10);

        bank.store_lightweight_episode("a".into(), Array1::from_vec(vec![1.0]), 0.8)
            .expect("store failed");
        bank.store_lightweight_episode("b".into(), Array1::from_vec(vec![2.0]), 0.6)
            .expect("store failed");

        let stats2 = bank.get_stats();
        assert_eq!(stats2.count, 2);
        assert!((stats2.avg_performance - 0.7).abs() < 1e-12);
        assert!((stats2.capacity_used - 0.2).abs() < 1e-12);

        bank.clear();
        assert_eq!(bank.size(), 0);
    }

    #[test]
    fn test_support_set_selection() {
        let mgr = SupportSetManager::<f64>::from_max_size(10)
            .expect("failed to create SupportSetManager");
        let candidates = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![10.0, 0.0]),
            Array1::from_vec(vec![0.0, 10.0]),
            Array1::from_vec(vec![5.0, 5.0]),
            Array1::from_vec(vec![10.0, 10.0]),
        ];
        let labels = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let selected = mgr
            .select_support_set(&candidates, &labels, 3)
            .expect("select failed");
        assert_eq!(selected.len(), 3);

        // Farthest-point should pick well-separated points
        // The seed is the one with largest norm = [10, 10] at index 4
        assert!(selected.contains(&4));
        // All selected indices must be unique
        let mut unique = selected.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), selected.len());
    }

    #[test]
    fn test_support_set_augmentation() {
        let mgr = SupportSetManager::<f64>::from_max_size(10)
            .expect("failed to create SupportSetManager");
        let support = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
        ];
        let augmented = mgr
            .augment_support_set(&support, 0.1)
            .expect("augment failed");
        // Should have original + noisy copies = 4
        assert_eq!(augmented.len(), 4);
        // First two should be identical to originals
        for i in 0..3 {
            assert!((augmented[0][i] - support[0][i]).abs() < 1e-12);
            assert!((augmented[1][i] - support[1][i]).abs() < 1e-12);
        }
        // Noisy copies should be slightly different
        let mut any_different = false;
        for i in 0..3 {
            if (augmented[2][i] - support[0][i]).abs() > 1e-15 {
                any_different = true;
            }
        }
        assert!(any_different, "augmented copy should differ from original");
    }

    #[test]
    fn test_support_set_quality() {
        let mgr = SupportSetManager::<f64>::from_max_size(10)
            .expect("failed to create SupportSetManager");

        // High diversity
        let diverse = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![100.0, 0.0]),
            Array1::from_vec(vec![0.0, 100.0]),
        ];
        let quality_diverse = mgr.evaluate_quality(&diverse).expect("quality failed");

        // Low diversity
        let clustered = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![0.1, 0.0]),
            Array1::from_vec(vec![0.0, 0.1]),
        ];
        let quality_clustered = mgr.evaluate_quality(&clustered).expect("quality failed");

        assert!(
            quality_diverse > quality_clustered,
            "diverse set should have higher quality than clustered set"
        );

        // Single element should return 0
        let single = vec![Array1::from_vec(vec![1.0, 2.0])];
        let quality_single = mgr.evaluate_quality(&single).expect("quality failed");
        assert!((quality_single - 0.0).abs() < 1e-12);
    }
}
