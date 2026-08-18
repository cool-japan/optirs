// Similarity-driven cross-task transfer for [`LifelongOptimizer`].
//
// The lifelong optimizer's EWC and Reptile paths consolidate knowledge *within*
// a parameter vector, but nothing related one task to another: `task_embeddings`,
// `transfer_weights`, `task_dependencies` and `task_clusters` had no producer at
// all, so a new task always started cold no matter how close it was to something
// already learned. This module supplies the missing half:
//
// * every gradient a task sees is folded into running first- and second-moment
//   statistics for that task,
// * those statistics are turned into a fixed-width task embedding, so tasks of
//   different parameter shapes stay comparable,
// * embeddings are compared by cosine similarity, which fills the task graph's
//   similarity matrix,
// * a new task started through [`LifelongOptimizer::start_task_with_probe`] is
//   warm-started from its nearest neighbour when that similarity clears the
//   transfer threshold, and
// * the similarity matrix is clustered agglomeratively into `task_clusters`.

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{LifelongOptimizer, LifelongStrategy};
use crate::error::{OptimError, Result};
use crate::utils::{scalar_or, try_f64};

/// Width of a task embedding when the strategy does not specify one.
///
/// [`LifelongStrategy::MetaLearning`] carries an explicit `task_embedding_size`
/// and that value is used instead.
pub const DEFAULT_TASK_EMBEDDING_DIM: usize = 64;

/// Upper bound on the embedding width, so a mis-configured
/// `task_embedding_size` cannot ask for an unbounded allocation per task.
pub const MAX_TASK_EMBEDDING_DIM: usize = 4096;

/// Cosine similarity a new task must reach before it is warm-started from an
/// existing one, and the linkage at which two tasks are placed in the same
/// cluster.
pub const DEFAULT_TRANSFER_THRESHOLD: f64 = 0.5;

/// Floor for the whitening denominator, so a coordinate whose gradient is
/// always exactly zero contributes 0 rather than a division by zero.
const WHITENING_EPSILON: f64 = 1e-12;

/// Running gradient statistics for one task.
///
/// Both moments are exact running means over every gradient the task has
/// observed, which is what makes the derived embedding a property of the *task*
/// rather than of the particular step it was sampled at.
#[derive(Debug, Clone, Default)]
pub struct TaskStatistics {
    /// Number of gradients folded in.
    observations: usize,
    /// Running mean of the gradient, `E[g]`.
    mean_gradient: Vec<f64>,
    /// Running mean of the squared gradient, `E[g^2]`.
    mean_squared_gradient: Vec<f64>,
}

impl TaskStatistics {
    /// Fold one gradient into the running statistics.
    ///
    /// A gradient of a different length than the ones seen so far restarts the
    /// statistics: it can only come from a task whose parameter vector changed
    /// shape, and averaging across shapes would produce a meaningless
    /// descriptor.
    pub fn observe(&mut self, gradient: &[f64]) {
        if self.mean_gradient.len() != gradient.len() {
            self.mean_gradient = vec![0.0; gradient.len()];
            self.mean_squared_gradient = vec![0.0; gradient.len()];
            self.observations = 0;
        }

        self.observations += 1;
        let weight = 1.0 / self.observations as f64;
        for ((mean, squared), &value) in self
            .mean_gradient
            .iter_mut()
            .zip(self.mean_squared_gradient.iter_mut())
            .zip(gradient.iter())
        {
            *mean += (value - *mean) * weight;
            *squared += (value * value - *squared) * weight;
        }
    }

    /// Number of gradients folded in.
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Running mean gradient.
    pub fn mean_gradient(&self) -> &[f64] {
        &self.mean_gradient
    }

    /// Running mean squared gradient.
    pub fn mean_squared_gradient(&self) -> &[f64] {
        &self.mean_squared_gradient
    }

    /// Fixed-width task embedding derived from the statistics.
    ///
    /// The descriptor is the *whitened* mean gradient
    /// `E[g_i] / sqrt(E[g_i^2] + eps)`: dividing the first moment by the square
    /// root of the second is the diagonal natural-gradient direction, so a
    /// coordinate that is merely noisy (large `E[g^2]`, small `E[g]`) does not
    /// dominate a coordinate that consistently pushes the same way. That vector
    /// is then reduced to `dim` components by signed feature hashing
    /// (Weinberger et al., "Feature Hashing for Large Scale Multitask
    /// Learning", ICML 2009), which preserves inner products in expectation
    /// while making tasks of different parameter dimensionality comparable, and
    /// finally normalized to unit length so cosine similarity is a plain dot
    /// product.
    ///
    /// A task whose gradients have averaged out to exactly zero has no
    /// direction; its embedding is all zeros and its similarity to everything is
    /// zero, which is the honest answer rather than an arbitrary one.
    pub fn embedding(&self, dim: usize) -> Vec<f64> {
        let dim = dim.clamp(1, MAX_TASK_EMBEDDING_DIM);
        let mut projected = vec![0.0f64; dim];

        for (index, (&mean, &squared)) in self
            .mean_gradient
            .iter()
            .zip(self.mean_squared_gradient.iter())
            .enumerate()
        {
            let whitened = mean / (squared + WHITENING_EPSILON).sqrt();
            if !whitened.is_finite() {
                continue;
            }
            let hash = mix64(index as u64);
            let bucket = (hash % dim as u64) as usize;
            let sign = if (hash >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            projected[bucket] += sign * whitened;
        }

        let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return vec![0.0; dim];
        }
        for value in projected.iter_mut() {
            *value /= norm;
        }
        projected
    }
}

/// SplitMix64 finalizer.
///
/// Used as the hash for feature hashing. It is written out rather than taken
/// from [`std::collections::hash_map::DefaultHasher`] because the standard
/// hasher's output is explicitly not stable across releases, and an embedding
/// that changes meaning between compiler versions would make transfer
/// decisions irreproducible.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Cosine similarity of two task embeddings.
///
/// Embeddings produced by [`TaskStatistics::embedding`] are unit length, so this
/// is their dot product; the explicit normalization keeps the function correct
/// for any caller-supplied vector, and embeddings of different widths (which can
/// only come from a strategy change mid-run) are reported as unrelated rather
/// than compared over a truncated prefix.
pub fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (&a, &b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }

    let denominator = (left_norm * right_norm).sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return 0.0;
    }
    (dot / denominator).clamp(-1.0, 1.0)
}

/// Canonical key for an unordered task pair, so `(a, b)` and `(b, a)` land in
/// the same slot of the similarity matrix.
fn similarity_key(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

/// What [`LifelongOptimizer::start_task_with_probe`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferOutcome {
    /// Task the new task was warm-started from, if any.
    pub source_task: Option<String>,
    /// Similarity to the nearest previously-seen task. Reported even when no
    /// transfer happened, so a caller can see how close the decision was.
    pub similarity: f64,
    /// Weight the source parameters were blended in with: `0` means the task
    /// started cold, `1` means it started exactly at the source's parameters.
    pub transfer_weight: f64,
}

impl<A: Float + ScalarOperand + Debug + std::iter::Sum, D: Dimension + Send + Sync>
    LifelongOptimizer<A, D>
{
    /// Similarity a new task must reach before it is warm-started, and the
    /// linkage at which two tasks are clustered together.
    pub fn transfer_threshold(&self) -> f64 {
        self.transfer_threshold
    }

    /// Set the transfer/clustering threshold.
    ///
    /// A cosine similarity lives in `[-1, 1]`; anything outside that range would
    /// either transfer unconditionally or never, which is a configuration error
    /// rather than a policy.
    pub fn set_transfer_threshold(&mut self, threshold: f64) -> Result<()> {
        if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
            return Err(OptimError::InvalidConfig(format!(
                "transfer threshold {threshold} is not a cosine similarity in [-1, 1]"
            )));
        }
        self.transfer_threshold = threshold;
        self.rebuild_task_clusters();
        Ok(())
    }

    /// Embedding of a task, once it has observed at least one gradient.
    pub fn task_embedding(&self, task_id: &str) -> Option<&[f64]> {
        self.shared_knowledge
            .task_embeddings
            .get(task_id)
            .map(|embedding| embedding.as_slice())
    }

    /// Running gradient statistics of a task.
    pub fn task_statistics(&self, task_id: &str) -> Option<&TaskStatistics> {
        self.shared_knowledge.task_statistics.get(task_id)
    }

    /// Weight `target` was warm-started from `source` with, or `0` if no
    /// transfer took place between them.
    pub fn transfer_weight(&self, source: &str, target: &str) -> f64 {
        self.shared_knowledge
            .transfer_weights
            .get(&(source.to_string(), target.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Tasks `task_id` was warm-started from, most recent last.
    pub fn task_dependencies(&self, task_id: &str) -> &[String] {
        self.task_graph
            .task_dependencies
            .get(task_id)
            .map(|dependencies| dependencies.as_slice())
            .unwrap_or(&[])
    }

    /// Current clustering of the tasks, each cluster sorted by task id and the
    /// clusters themselves sorted by their first member.
    pub fn task_clusters(&self) -> &[Vec<String>] {
        &self.task_graph.task_clusters
    }

    /// Current parameters of a task.
    pub fn task_parameters(&self, task_id: &str) -> Option<&Array<A, D>> {
        self.task_optimizers
            .get(task_id)
            .map(|optimizer| optimizer.parameters())
    }

    /// Mean strength of the warm starts performed so far, or `0` when nothing
    /// has been transferred.
    ///
    /// This is what [`super::LifelongStats::transfer_efficiency`] reports.
    pub fn mean_transfer_weight(&self) -> f64 {
        let weights = &self.shared_knowledge.transfer_weights;
        if weights.is_empty() {
            return 0.0;
        }
        weights.values().sum::<f64>() / weights.len() as f64
    }

    /// Start a new task, warm-starting it from the most similar task seen so
    /// far.
    ///
    /// `probe_gradient` is a gradient of the *new* task evaluated at
    /// `initial_parameters` — one backward pass on its first batch. It is what
    /// makes the decision possible at all: a task that has taken no steps yet
    /// has no statistics of its own, so without a probe the nearest neighbour
    /// could only be guessed at. The probe is also folded into the new task's
    /// statistics, so its embedding exists from the first moment.
    ///
    /// When the nearest neighbour's cosine similarity reaches
    /// [`Self::transfer_threshold`] and its parameters have the same shape, the
    /// new task starts at
    /// `(1 - w) * initial_parameters + w * source_parameters` with
    /// `w = similarity`, an edge is recorded in the task graph, and the weight
    /// is recorded in `transfer_weights`. Otherwise the task starts exactly
    /// where the caller put it.
    ///
    /// Everything [`Self::start_task`] does — consolidating the outgoing task
    /// into the EWC anchor, creating the task optimizer, starting performance
    /// tracking — still happens.
    pub fn start_task_with_probe(
        &mut self,
        task_id: String,
        initial_parameters: Array<A, D>,
        probe_gradient: &Array<A, D>,
    ) -> Result<TransferOutcome> {
        if probe_gradient.raw_dim() != initial_parameters.raw_dim() {
            return Err(OptimError::DimensionMismatch(format!(
                "transfer probe: initial parameters have shape {:?} but the probe \
                 gradient has shape {:?}",
                initial_parameters.raw_dim().slice(),
                probe_gradient.raw_dim().slice()
            )));
        }

        let dim = self.embedding_dim();
        let probe_values = flatten_to_f64(probe_gradient)?;
        let mut probe_statistics = TaskStatistics::default();
        probe_statistics.observe(&probe_values);
        let probe_embedding = probe_statistics.embedding(dim);

        // Score the probe against every task that already has an embedding, and
        // record the scores: they are the new task's row of the similarity
        // matrix, and they are what the clustering below reads.
        let mut best: Option<(String, f64)> = None;
        for (candidate, embedding) in &self.shared_knowledge.task_embeddings {
            if *candidate == task_id {
                continue;
            }
            let similarity = cosine_similarity(&probe_embedding, embedding);
            self.task_graph
                .task_similarities
                .insert(similarity_key(&task_id, candidate), similarity);

            // Ties are broken by task id so the choice does not depend on the
            // iteration order of a hash map.
            let better = match &best {
                None => true,
                Some((best_id, best_similarity)) => {
                    similarity > *best_similarity
                        || (similarity == *best_similarity && candidate < best_id)
                }
            };
            if better {
                best = Some((candidate.clone(), similarity));
            }
        }

        let mut parameters = initial_parameters;
        let mut outcome = TransferOutcome {
            source_task: None,
            similarity: best.as_ref().map(|(_, s)| *s).unwrap_or(0.0),
            transfer_weight: 0.0,
        };

        if let Some((source, similarity)) = best {
            if similarity >= self.transfer_threshold {
                let source_parameters = self.task_parameters(&source).cloned();
                if let Some(source_parameters) = source_parameters {
                    if source_parameters.raw_dim() == parameters.raw_dim() {
                        let weight = similarity.clamp(0.0, 1.0);
                        let blend = scalar_or(weight, A::zero());
                        let keep = A::one() - blend;
                        for (slot, &learned) in parameters.iter_mut().zip(source_parameters.iter())
                        {
                            *slot = *slot * keep + learned * blend;
                        }
                        outcome.source_task = Some(source.clone());
                        outcome.transfer_weight = weight;
                    }
                }
            }
        }

        self.start_task(task_id.clone(), parameters)?;

        if let Some(source) = outcome.source_task.clone() {
            self.shared_knowledge
                .transfer_weights
                .insert((source.clone(), task_id.clone()), outcome.transfer_weight);
            let dependencies = self
                .task_graph
                .task_dependencies
                .entry(task_id.clone())
                .or_default();
            if !dependencies.contains(&source) {
                dependencies.push(source);
            }
        }

        // Seed the new task's own statistics with the probe so it has an
        // embedding before its first real update.
        self.shared_knowledge
            .task_statistics
            .insert(task_id.clone(), probe_statistics);
        self.shared_knowledge
            .task_embeddings
            .insert(task_id, probe_embedding);
        self.rebuild_task_clusters();

        Ok(outcome)
    }

    /// Fold one observed gradient into the current task's statistics and refresh
    /// everything derived from them.
    ///
    /// Called from [`Self::update_current_task`] on every update, which is what
    /// keeps a task's embedding tracking the task rather than freezing at
    /// whatever its first gradient happened to be.
    pub(super) fn record_task_observation(&mut self, gradient: &Array<A, D>) -> Result<()> {
        let Some(task_id) = self.current_task.clone() else {
            return Ok(());
        };

        let dim = self.embedding_dim();
        let values = flatten_to_f64(gradient)?;

        let statistics = self
            .shared_knowledge
            .task_statistics
            .entry(task_id.clone())
            .or_default();
        statistics.observe(&values);
        let embedding = statistics.embedding(dim);

        self.shared_knowledge
            .task_embeddings
            .insert(task_id.clone(), embedding);
        self.refresh_similarities(&task_id);
        self.rebuild_task_clusters();

        Ok(())
    }

    /// Embedding width: the strategy's `task_embedding_size` when it has one,
    /// otherwise [`DEFAULT_TASK_EMBEDDING_DIM`].
    fn embedding_dim(&self) -> usize {
        match self.strategy {
            LifelongStrategy::MetaLearning {
                task_embedding_size,
                ..
            } => task_embedding_size.clamp(1, MAX_TASK_EMBEDDING_DIM),
            _ => DEFAULT_TASK_EMBEDDING_DIM,
        }
    }

    /// Recompute one task's row of the similarity matrix.
    fn refresh_similarities(&mut self, task_id: &str) {
        let Some(embedding) = self.shared_knowledge.task_embeddings.get(task_id).cloned() else {
            return;
        };

        for (other, other_embedding) in &self.shared_knowledge.task_embeddings {
            if other == task_id {
                continue;
            }
            let similarity = cosine_similarity(&embedding, other_embedding);
            self.task_graph
                .task_similarities
                .insert(similarity_key(task_id, other), similarity);
        }
    }

    /// Rebuild `task_clusters` by single-linkage agglomerative clustering of the
    /// similarity matrix, cut at [`Self::transfer_threshold`].
    ///
    /// The cutoff is the same threshold that decides whether transfer happens,
    /// so "these tasks cluster together" and "these tasks transfer to each
    /// other" cannot disagree.
    ///
    /// Single linkage merged repeatedly until the closest pair falls below a
    /// cutoff produces exactly the connected components of the graph whose
    /// edges are the pairs at or above that cutoff, so that is what is computed
    /// here: a union-find pass over the pairs, which is `O(T^2)` rather than the
    /// `O(T^4)` of running the merge loop literally. This function runs after
    /// every observed gradient, so the difference is not academic.
    ///
    /// Task ids are sorted before clustering and roots are always the smallest
    /// member index, so the output does not depend on hash-map iteration order.
    fn rebuild_task_clusters(&mut self) {
        let mut tasks: Vec<String> = self
            .shared_knowledge
            .task_embeddings
            .keys()
            .cloned()
            .collect();
        tasks.sort();

        let count = tasks.len();
        if count == 0 {
            self.task_graph.task_clusters.clear();
            return;
        }

        let threshold = self.transfer_threshold;
        let mut parent: Vec<usize> = (0..count).collect();
        for left in 0..count {
            for right in (left + 1)..count {
                if self.compute_task_similarity(&tasks[left], &tasks[right]) >= threshold {
                    let left_root = find_root(&mut parent, left);
                    let right_root = find_root(&mut parent, right);
                    if left_root != right_root {
                        parent[left_root.max(right_root)] = left_root.min(right_root);
                    }
                }
            }
        }

        let mut buckets: Vec<Vec<String>> = vec![Vec::new(); count];
        for (index, task) in tasks.iter().enumerate() {
            let root = find_root(&mut parent, index);
            buckets[root].push(task.clone());
        }

        let mut named: Vec<Vec<String>> = buckets
            .into_iter()
            .filter(|cluster| !cluster.is_empty())
            .collect();
        named.sort();
        self.task_graph.task_clusters = named;
    }
}

/// Union-find root with path halving.
fn find_root(parent: &mut [usize], node: usize) -> usize {
    let mut current = node;
    while parent[current] != current {
        parent[current] = parent[parent[current]];
        current = parent[current];
    }
    current
}

/// Flatten an array into `f64`, reporting a value the target type cannot
/// represent rather than silently substituting one.
fn flatten_to_f64<A: Float, D: Dimension>(array: &Array<A, D>) -> Result<Vec<f64>> {
    array.iter().map(|&value| try_f64(value)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::online_learning::MemoryUpdateStrategy;
    use scirs2_core::ndarray::{Array1, Ix1};

    /// A quadratic task `f(x) = ||x - target||^2`, whose gradient is
    /// `2 (x - target)`.
    fn quadratic_gradient(parameters: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        parameters
            .iter()
            .zip(target.iter())
            .map(|(&x, &t)| 2.0 * (x - t))
            .collect()
    }

    fn quadratic_loss(parameters: &Array1<f64>, target: &Array1<f64>) -> f64 {
        parameters
            .iter()
            .zip(target.iter())
            .map(|(&x, &t)| (x - t) * (x - t))
            .sum()
    }

    fn optimizer() -> LifelongOptimizer<f64, Ix1> {
        LifelongOptimizer::new(LifelongStrategy::MemoryAugmented {
            memory_size: 128,
            update_strategy: MemoryUpdateStrategy::FIFO,
        })
    }

    /// Train `task_id` on the quadratic with `target` for `steps` updates.
    fn train(
        optimizer: &mut LifelongOptimizer<f64, Ix1>,
        task_id: &str,
        target: &Array1<f64>,
        steps: usize,
    ) {
        for _ in 0..steps {
            let parameters = optimizer
                .task_parameters(task_id)
                .expect("task must exist")
                .clone();
            let gradient = quadratic_gradient(&parameters, target);
            let loss = quadratic_loss(&parameters, target);
            optimizer
                .update_current_task(&gradient, loss)
                .expect("update must succeed");
        }
    }

    /// Two tasks that pull in the same direction must transfer, and the warm
    /// start must show up as a faster initial loss drop than a cold start.
    #[test]
    fn a_similar_task_is_warm_started_and_drops_its_loss_faster() {
        let target_a = Array1::from_vec(vec![3.0, 3.0, 3.0, 3.0]);
        let target_b = Array1::from_vec(vec![3.05, 3.05, 3.05, 3.05]);
        let cold_start = Array1::zeros(4);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 4000);

        let learned_a = opt.task_parameters("a").expect("task a").clone();
        assert!(
            quadratic_loss(&learned_a, &target_a) < 1.0,
            "task a did not learn: {learned_a:?}"
        );

        // The probe is one backward pass of task b at its cold-start point.
        let probe = quadratic_gradient(&cold_start, &target_b);
        let outcome = opt
            .start_task_with_probe("b".to_string(), cold_start.clone(), &probe)
            .expect("start b");

        assert_eq!(
            outcome.source_task.as_deref(),
            Some("a"),
            "a task pulling the same way must transfer (similarity {})",
            outcome.similarity
        );
        assert!(
            outcome.similarity > 0.9,
            "similarity {} is implausibly low for near-identical tasks",
            outcome.similarity
        );
        assert!(outcome.transfer_weight > 0.9);

        let warm_parameters = opt.task_parameters("b").expect("task b").clone();
        let warm_initial_loss = quadratic_loss(&warm_parameters, &target_b);
        let cold_initial_loss = quadratic_loss(&cold_start, &target_b);
        assert!(
            warm_initial_loss < cold_initial_loss * 0.1,
            "warm start did not help: {warm_initial_loss} vs cold {cold_initial_loss}"
        );

        // And after the same number of updates the warm-started task is still
        // ahead of an identically configured cold start.
        train(&mut opt, "b", &target_b, 20);
        let warm_after = quadratic_loss(opt.task_parameters("b").expect("task b"), &target_b);

        let mut cold = optimizer();
        cold.start_task("b".to_string(), cold_start.clone())
            .expect("cold start b");
        train(&mut cold, "b", &target_b, 20);
        let cold_after = quadratic_loss(cold.task_parameters("b").expect("cold task b"), &target_b);

        assert!(
            warm_after < cold_after,
            "warm start ({warm_after}) must beat cold start ({cold_after}) after 20 steps"
        );
    }

    /// A task pulling the opposite way must not be warm-started from the
    /// existing one — transfer has to be similarity-driven, not automatic.
    #[test]
    fn a_dissimilar_task_is_not_warm_started() {
        let target_a = Array1::from_vec(vec![3.0, 3.0, 3.0, 3.0]);
        let target_c = Array1::from_vec(vec![-3.0, -3.0, -3.0, -3.0]);
        let cold_start = Array1::zeros(4);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 500);

        let probe = quadratic_gradient(&cold_start, &target_c);
        let outcome = opt
            .start_task_with_probe("c".to_string(), cold_start.clone(), &probe)
            .expect("start c");

        assert_eq!(
            outcome.source_task, None,
            "an opposing task must not transfer (similarity {})",
            outcome.similarity
        );
        assert!(
            outcome.similarity < 0.0,
            "opposing gradients must score below zero, got {}",
            outcome.similarity
        );
        assert_eq!(outcome.transfer_weight, 0.0);
        assert_eq!(
            opt.task_parameters("c").expect("task c"),
            &cold_start,
            "a task that did not transfer must start exactly where the caller put it"
        );
        assert!(opt.task_dependencies("c").is_empty());
    }

    /// The transfer must be recorded in the task graph and in the shared
    /// knowledge, not just applied and forgotten.
    #[test]
    fn transfer_weights_and_dependencies_are_recorded() {
        let target_a = Array1::from_vec(vec![2.0, -2.0]);
        let target_b = Array1::from_vec(vec![2.1, -2.1]);
        let cold_start = Array1::zeros(2);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 500);

        let probe = quadratic_gradient(&cold_start, &target_b);
        opt.start_task_with_probe("b".to_string(), cold_start, &probe)
            .expect("start b");

        assert_eq!(opt.task_dependencies("b").to_vec(), vec!["a".to_string()]);
        assert!(opt.transfer_weight("a", "b") > 0.9);
        assert_eq!(
            opt.transfer_weight("b", "a"),
            0.0,
            "transfer is directed: b was started from a, not the other way round"
        );
        assert!(
            opt.compute_task_similarity("a", "b") > 0.9,
            "the similarity matrix must be populated"
        );
        assert!(
            opt.get_lifelong_stats().transfer_efficiency > 0.9,
            "transfer efficiency must reflect the transfers that happened"
        );
        assert!(opt.task_embedding("a").is_some());
        assert!(opt.task_embedding("b").is_some());
    }

    /// Similar tasks must land in one cluster and a dissimilar one on its own.
    #[test]
    fn similar_tasks_cluster_together() {
        let cold_start = Array1::zeros(3);
        let target_a = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let target_b = Array1::from_vec(vec![1.2, 1.1, 1.05]);
        let target_c = Array1::from_vec(vec![-1.0, -1.0, -1.0]);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 200);

        opt.start_task("b".to_string(), cold_start.clone())
            .expect("start b");
        train(&mut opt, "b", &target_b, 200);

        opt.start_task("c".to_string(), cold_start)
            .expect("start c");
        train(&mut opt, "c", &target_c, 200);

        let clusters = opt.task_clusters().to_vec();
        assert_eq!(
            clusters,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string()]
            ],
            "clusters: {clusters:?}, sim(a,b) = {}, sim(a,c) = {}",
            opt.compute_task_similarity("a", "b"),
            opt.compute_task_similarity("a", "c")
        );
    }

    /// Raising the threshold above the measured similarity must break the
    /// cluster apart and stop the transfer — the threshold is genuinely read,
    /// not decorative.
    ///
    /// The two tasks agree on five of six coordinates and disagree on the
    /// sixth, which puts their similarity between the default threshold and the
    /// raised one. (Two tasks whose gradient directions are exactly parallel —
    /// `[1, 1]` against `[1.1, 1.1]`, say — have a cosine of exactly 1 and can
    /// never be separated by a threshold, which is the honest behaviour of a
    /// direction-based descriptor, not a defect.)
    #[test]
    fn the_threshold_controls_clustering_and_transfer() {
        let cold_start = Array1::zeros(6);
        let target_a = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let target_b = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0]);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 200);
        opt.start_task("b".to_string(), cold_start.clone())
            .expect("start b");
        train(&mut opt, "b", &target_b, 200);

        let similarity = opt.compute_task_similarity("a", "b");
        assert!(
            (0.5..0.9).contains(&similarity),
            "five agreeing coordinates out of six should score in [0.5, 0.9), got {similarity}"
        );
        assert_eq!(
            opt.task_clusters().len(),
            1,
            "similar tasks share a cluster"
        );

        opt.set_transfer_threshold(0.9).expect("valid threshold");
        assert_eq!(
            opt.task_clusters().len(),
            2,
            "a threshold above the measured similarity must separate the tasks"
        );

        // And no transfer happens at that threshold either: task d also agrees
        // with a on five of six coordinates, which clears the default threshold
        // but not the raised one.
        let target_d = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, -1.0, 1.0]);
        let probe = quadratic_gradient(&cold_start, &target_d);
        let outcome = opt
            .start_task_with_probe("d".to_string(), cold_start, &probe)
            .expect("start d");
        assert!(
            (0.5..0.9).contains(&outcome.similarity),
            "probe similarity {} is outside the band this test needs",
            outcome.similarity
        );
        assert_eq!(
            outcome.source_task, None,
            "similarity {} cleared a threshold of 0.9",
            outcome.similarity
        );

        assert!(opt.set_transfer_threshold(2.0).is_err());
        assert!(opt.set_transfer_threshold(f64::NAN).is_err());
    }

    /// A probe of the wrong shape is a caller error and must be reported.
    #[test]
    fn a_mismatched_probe_is_reported() {
        let mut opt = optimizer();
        let error = opt
            .start_task_with_probe(
                "a".to_string(),
                Array1::zeros(3),
                &Array1::from_vec(vec![1.0, 2.0]),
            )
            .expect_err("a shape mismatch must be reported");
        assert!(
            matches!(error, OptimError::DimensionMismatch(_)),
            "{error:?}"
        );
    }

    /// The very first task has nothing to transfer from, and must say so.
    #[test]
    fn the_first_task_has_nothing_to_transfer_from() {
        let mut opt = optimizer();
        let outcome = opt
            .start_task_with_probe(
                "a".to_string(),
                Array1::zeros(2),
                &Array1::from_vec(vec![1.0, -1.0]),
            )
            .expect("start a");
        assert_eq!(outcome.source_task, None);
        assert_eq!(outcome.similarity, 0.0);
        assert_eq!(opt.mean_transfer_weight(), 0.0);
        assert_eq!(opt.task_clusters().to_vec(), vec![vec!["a".to_string()]]);
    }

    /// Feature hashing must keep tasks of different parameter dimensionality
    /// comparable instead of panicking or reporting a fixed zero.
    #[test]
    fn embeddings_have_a_fixed_width_regardless_of_parameter_count() {
        let mut small = TaskStatistics::default();
        small.observe(&[1.0, -1.0, 1.0]);
        let mut large = TaskStatistics::default();
        large.observe(&vec![0.5; 500]);

        assert_eq!(small.embedding(64).len(), 64);
        assert_eq!(large.embedding(64).len(), 64);
        let similarity = cosine_similarity(&small.embedding(64), &large.embedding(64));
        assert!(
            (-1.0..=1.0).contains(&similarity),
            "similarity {similarity} is not a cosine"
        );
    }

    /// Opposite directions must score -1 and identical ones +1, so the
    /// similarity really is a cosine of the task direction.
    #[test]
    fn the_embedding_captures_gradient_direction() {
        let mut forward = TaskStatistics::default();
        let mut backward = TaskStatistics::default();
        for step in 0..10 {
            let scale = 1.0 + step as f64;
            forward.observe(&[scale, 2.0 * scale, -scale]);
            backward.observe(&[-scale, -2.0 * scale, scale]);
        }

        let same = cosine_similarity(&forward.embedding(32), &forward.embedding(32));
        let opposite = cosine_similarity(&forward.embedding(32), &backward.embedding(32));
        assert!((same - 1.0).abs() < 1e-9, "same direction scored {same}");
        assert!(
            (opposite + 1.0).abs() < 1e-9,
            "opposite direction scored {opposite}"
        );
    }

    /// A task whose gradients average to zero has no direction, and must report
    /// that rather than inventing one.
    #[test]
    fn a_directionless_task_has_a_zero_embedding() {
        let mut statistics = TaskStatistics::default();
        statistics.observe(&[1.0, 1.0]);
        statistics.observe(&[-1.0, -1.0]);
        let embedding = statistics.embedding(8);
        assert!(embedding.iter().all(|&value| value == 0.0));
        assert_eq!(cosine_similarity(&embedding, &embedding), 0.0);
    }

    /// Forgetting must be measured from a re-evaluation of an old task, not
    /// reported as the constant `0.1` this used to return.
    #[test]
    fn forgetting_is_measured_from_re_evaluated_tasks() {
        let cold_start = Array1::zeros(2);
        let target_a = Array1::from_vec(vec![1.0, 1.0]);

        let mut opt = optimizer();
        opt.start_task("a".to_string(), cold_start.clone())
            .expect("start a");
        train(&mut opt, "a", &target_a, 50);

        assert_eq!(
            opt.get_lifelong_stats().catastrophic_forgetting,
            0.0,
            "forgetting is not observable while the task is still current"
        );

        opt.start_task("b".to_string(), cold_start)
            .expect("start b");
        assert_eq!(
            opt.get_lifelong_stats().catastrophic_forgetting,
            0.0,
            "switching away does not by itself demonstrate forgetting"
        );

        // Re-evaluating task a and finding it worse is what forgetting is.
        let reference = opt
            .task_performance
            .get("a")
            .and_then(|history| history.last())
            .copied()
            .expect("task a recorded losses");
        opt.record_task_performance("a", reference + 0.25)
            .expect("recording a re-evaluation must succeed");
        let forgetting = opt.get_lifelong_stats().catastrophic_forgetting;
        assert!(
            (forgetting - 0.25).abs() < 1e-9,
            "forgetting was not measured from the re-evaluation: {forgetting}"
        );

        // Recording against the active task, or an unknown one, is rejected.
        assert!(opt.record_task_performance("b", 1.0).is_err());
        assert!(opt.record_task_performance("nope", 1.0).is_err());
    }

    /// Statistics must be running means over every gradient, not the latest one.
    #[test]
    fn statistics_are_running_means() {
        let mut statistics = TaskStatistics::default();
        statistics.observe(&[2.0, 0.0]);
        statistics.observe(&[4.0, 0.0]);
        assert_eq!(statistics.observations(), 2);
        assert!((statistics.mean_gradient()[0] - 3.0).abs() < 1e-12);
        assert!((statistics.mean_squared_gradient()[0] - 10.0).abs() < 1e-12);
    }
}
