//! Cross-domain knowledge transfer for Neural Architecture Search.
//!
//! This module sits on top of [`ArchitectureKnowledgeGraph`] (the persisted
//! catalogue of previously-evaluated architectures) and provides a higher
//! level API for *transferring* knowledge from a richly-explored source
//! domain into a new, less-explored target domain.
//!
//! The core abstraction is [`CrossDomainTransferEngine`], which combines:
//!
//! 1. **Domain similarity** — how close are two domains in embedding space?
//! 2. **Per-architecture transferability** — which specific source
//!    architectures are good warm-starts for the target query?
//! 3. **Warm-start embeddings** — a single performance-weighted embedding
//!    that summarises the best of the source domain and can be used as the
//!    initial query for the target NAS run.
//! 4. **Transfer subgraphs** — a freshly-assembled
//!    [`ArchitectureKnowledgeGraph`] containing only the recommended source
//!    architectures wired to a synthetic target placeholder via `Transfers`
//!    edges, ready to be persisted alongside the new search.
//!
//! # Examples
//!
//! ```ignore
//! use optirs_nas::architecture_knowledge_graph::{
//!     ArchitectureKnowledgeGraph, RelationType, PerformanceRecord,
//! };
//! use optirs_nas::cross_domain_transfer::{
//!     CrossDomainTransferEngine, DomainSimilarityMetric,
//! };
//!
//! let mut graph = ArchitectureKnowledgeGraph::new();
//! let a = graph.add_architecture("resnet50", vec![1.0, 0.0, 0.0], "vision");
//! let b = graph.add_architecture("efficientnet", vec![0.9, 0.1, 0.0], "vision");
//! graph
//!     .set_performance(
//!         a,
//!         PerformanceRecord {
//!             task_id: "imagenet".into(),
//!             accuracy: 0.78,
//!             latency_ms: 20.0,
//!             memory_mb: 200.0,
//!             training_epochs: 100,
//!         },
//!     )
//!     .unwrap();
//!
//! let engine = CrossDomainTransferEngine::new()
//!     .with_metric(DomainSimilarityMetric::MeanCosine);
//! let recommendations = engine
//!     .recommend_transfers(&graph, "medical-imaging", &[1.0, 0.0, 0.0], 2)
//!     .unwrap();
//! ```
//!
//! # Mathematical notes
//!
//! All similarity metrics return a value in `[0, 1]`. Raw cosine values are
//! clipped to `[0, 1]` (negative correlation is treated as "no transferable
//! similarity"). The Wasserstein metric uses a one-step nearest-neighbour
//! approximation in cosine space, which is a coarse but cheap surrogate for
//! the true 1-Wasserstein distance and is sufficient for ranking.

use crate::architecture_knowledge_graph::{
    ArchitectureKnowledgeGraph, NodeId, PerformanceRecord, RelationType,
};
use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How to score similarity between two *sets* of architecture embeddings.
///
/// All variants return a similarity in `[0, 1]` after clipping. The choice
/// trades off robustness vs. sensitivity:
///
/// - [`MeanCosine`](Self::MeanCosine) averages every pairwise cosine. Stable
///   but can wash out concentrated overlap when one set is large.
/// - [`MaxCosine`](Self::MaxCosine) takes the best pair. Sensitive to a
///   single shared mode, but vulnerable to outliers.
/// - [`Wasserstein`](Self::Wasserstein) approximates a 1-Wasserstein
///   distance via mean nearest-neighbour cosine, giving a balanced score.
/// - [`CentroidDistance`](Self::CentroidDistance) compares the two
///   centroids only. Fast, but ignores intra-domain spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DomainSimilarityMetric {
    /// Mean cosine similarity between every source/target embedding pair.
    #[default]
    MeanCosine,
    /// Maximum cosine similarity over every source/target embedding pair.
    MaxCosine,
    /// Approximate 1-Wasserstein distance via mean nearest-neighbour cosine.
    Wasserstein,
    /// Cosine similarity between source and target centroids.
    CentroidDistance,
}

/// Compact description of a domain used when seeding transfer decisions.
///
/// `DomainProfile` is purely informational — the engine does not introspect
/// the fields beyond serialisation — but callers persist these profiles so
/// that historical transfer decisions can be replayed against new domains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DomainProfile {
    /// Human readable name (used as the `domain` field on knowledge graph
    /// nodes).
    pub domain_name: String,
    /// Task family the domain belongs to (e.g. `"classification"`,
    /// `"regression"`, `"rl"`).
    pub task_type: String,
    /// Input tensor dimensionality.
    pub input_dimensionality: usize,
    /// Output tensor dimensionality.
    pub output_dimensionality: usize,
    /// Typical training set size in samples.
    pub typical_dataset_size: usize,
    /// Metric the domain prioritises (e.g. `"accuracy"`, `"latency_ms"`).
    pub priority_metric: String,
}

impl DomainProfile {
    /// Construct a profile with sensible classification defaults.
    pub fn new(domain_name: impl Into<String>, task_type: impl Into<String>) -> Self {
        Self {
            domain_name: domain_name.into(),
            task_type: task_type.into(),
            input_dimensionality: 0,
            output_dimensionality: 0,
            typical_dataset_size: 0,
            priority_metric: "accuracy".to_string(),
        }
    }
}

/// A single transfer recommendation produced by the engine.
///
/// Each recommendation identifies a candidate source-domain architecture
/// together with the metrics that drove its ranking. The
/// [`recommended_fine_tune_epochs`](Self::recommended_fine_tune_epochs)
/// field provides a heuristic budget — architectures that are more
/// dissimilar to the query are given more epochs to adapt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRecommendation {
    /// Stable node identifier inside the source knowledge graph.
    pub source_node: NodeId,
    /// External / human-readable architecture identifier.
    pub source_arch_id: String,
    /// Cosine similarity between the source embedding and the target query.
    /// Clipped to `[0, 1]`.
    pub similarity: f64,
    /// Weighted combination of similarity, source performance, and graph
    /// distance. Used as the primary ranking key.
    pub transferability_score: f64,
    /// Copy of the source node's performance record, if present.
    pub source_performance: Option<PerformanceRecord>,
    /// Heuristic number of fine-tuning epochs recommended for adapting the
    /// source architecture to the target domain.
    pub recommended_fine_tune_epochs: usize,
}

/// Weights for the three components of the transferability score.
///
/// The components — similarity to the query, source performance, and graph
/// distance to the target domain — are combined linearly. Weights need not
/// sum to one but the engine normalises them internally when ranking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransferabilityWeights {
    /// Weight applied to the cosine similarity between the source embedding
    /// and the target query.
    pub similarity_weight: f64,
    /// Weight applied to the normalised source performance score.
    pub performance_weight: f64,
    /// Weight applied to `1 - graph_distance_to_target`.
    pub graph_distance_weight: f64,
}

impl Default for TransferabilityWeights {
    fn default() -> Self {
        Self {
            similarity_weight: 0.5,
            performance_weight: 0.3,
            graph_distance_weight: 0.2,
        }
    }
}

impl TransferabilityWeights {
    /// Sum of the three weights. Used to renormalise the linear
    /// combination so that the resulting transferability score remains in
    /// `[0, 1]` regardless of caller configuration.
    fn total(&self) -> f64 {
        self.similarity_weight + self.performance_weight + self.graph_distance_weight
    }
}

/// Output of [`CrossDomainTransferEngine::compute_domain_similarity`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSimilarityReport {
    /// Name of the source domain that was probed.
    pub source_domain: String,
    /// Name of the target domain that was probed.
    pub target_domain: String,
    /// Metric used to produce the score.
    pub metric: DomainSimilarityMetric,
    /// Aggregate similarity score in `[0, 1]`.
    pub similarity: f64,
    /// Number of nodes that matched `source_domain` in the graph.
    pub source_node_count: usize,
    /// Number of nodes that matched `target_domain` in the graph.
    pub target_node_count: usize,
}

/// Top-level entry point for cross-domain transfer queries.
///
/// `CrossDomainTransferEngine` is constructed with the builder methods
/// [`with_metric`](Self::with_metric),
/// [`with_thresholds`](Self::with_thresholds),
/// [`with_propagation`](Self::with_propagation), and
/// [`with_weights`](Self::with_weights). It is cheap to clone and carries
/// no per-graph state — the same engine can be reused across many
/// knowledge graphs.
#[derive(Debug, Clone)]
pub struct CrossDomainTransferEngine {
    similarity_metric: DomainSimilarityMetric,
    transferability_weights: TransferabilityWeights,
    min_similarity_threshold: f64,
    knowledge_propagation_decay: f64,
    knowledge_propagation_depth: usize,
}

impl Default for CrossDomainTransferEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossDomainTransferEngine {
    /// Construct an engine with conservative defaults: mean-cosine domain
    /// metric, equal-ish transferability weights (0.5 / 0.3 / 0.2), a
    /// transferability threshold of 0.0 (i.e. nothing filtered out), and a
    /// short three-hop propagation with decay 0.85.
    pub fn new() -> Self {
        Self {
            similarity_metric: DomainSimilarityMetric::MeanCosine,
            transferability_weights: TransferabilityWeights::default(),
            min_similarity_threshold: 0.0,
            knowledge_propagation_decay: 0.85,
            knowledge_propagation_depth: 3,
        }
    }

    /// Replace the domain similarity metric.
    pub fn with_metric(mut self, metric: DomainSimilarityMetric) -> Self {
        self.similarity_metric = metric;
        self
    }

    /// Replace the minimum transferability threshold.
    ///
    /// Recommendations with a `transferability_score` strictly less than
    /// `min_similarity` are dropped. Values outside `[0, 1]` are clamped
    /// at construction time.
    pub fn with_thresholds(mut self, min_similarity: f64) -> Self {
        self.min_similarity_threshold = min_similarity.clamp(-1.0, 1.0);
        self
    }

    /// Configure the knowledge-graph propagation used when scoring graph
    /// distance to the target domain. `decay` must lie strictly inside
    /// `(0, 1)` and `depth` must be at least one — invalid values are
    /// silently snapped to the nearest valid configuration so that the
    /// builder pattern stays infallible.
    pub fn with_propagation(mut self, decay: f64, depth: usize) -> Self {
        let safe_decay = if decay.is_finite() && decay > 0.0 && decay < 1.0 {
            decay
        } else {
            0.85
        };
        self.knowledge_propagation_decay = safe_decay;
        self.knowledge_propagation_depth = depth.max(1);
        self
    }

    /// Replace the transferability-score weights.
    pub fn with_weights(mut self, weights: TransferabilityWeights) -> Self {
        self.transferability_weights = weights;
        self
    }

    /// Current similarity metric.
    pub fn similarity_metric(&self) -> DomainSimilarityMetric {
        self.similarity_metric
    }

    /// Current transferability weights.
    pub fn transferability_weights(&self) -> TransferabilityWeights {
        self.transferability_weights
    }

    /// Current minimum transferability threshold.
    pub fn min_similarity_threshold(&self) -> f64 {
        self.min_similarity_threshold
    }

    /// Compare two domains in the same knowledge graph and produce an
    /// aggregate similarity score under the configured metric.
    ///
    /// The two domains can be identical (a useful sanity check — a domain
    /// compared to itself should yield a similarity near 1.0). At least
    /// one node per domain must exist; otherwise the engine returns an
    /// [`OptimError::InvalidParameter`] explaining which domain was empty.
    pub fn compute_domain_similarity(
        &self,
        graph: &ArchitectureKnowledgeGraph,
        source_domain: &str,
        target_domain: &str,
    ) -> Result<DomainSimilarityReport> {
        let source_embeddings = collect_domain_embeddings(graph, source_domain);
        let target_embeddings = collect_domain_embeddings(graph, target_domain);

        if source_embeddings.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "compute_domain_similarity: source domain '{}' has no nodes",
                source_domain
            )));
        }
        if target_embeddings.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "compute_domain_similarity: target domain '{}' has no nodes",
                target_domain
            )));
        }

        let similarity = aggregate_set_similarity(
            self.similarity_metric,
            &source_embeddings,
            &target_embeddings,
        );

        Ok(DomainSimilarityReport {
            source_domain: source_domain.to_string(),
            target_domain: target_domain.to_string(),
            metric: self.similarity_metric,
            similarity: clip_unit(similarity),
            source_node_count: source_embeddings.len(),
            target_node_count: target_embeddings.len(),
        })
    }

    /// Rank source-domain architectures by their suitability as warm-starts
    /// for a target query.
    ///
    /// `target_query_embedding` is the embedding of the architecture we
    /// want to *find a good initialiser for*. The candidate pool is every
    /// node whose `domain` differs from `target_domain` — i.e. the engine
    /// is biased *against* re-recommending architectures from the target
    /// domain itself. Recommendations are scored as a weighted combination
    /// of:
    ///
    /// 1. **Cosine similarity** of the source embedding to the query
    /// 2. **Normalised performance** (clamped accuracy in `[0, 1]`)
    /// 3. **Graph proximity** to nearest target-domain node, expressed as
    ///    `1 - graph_distance`. When no target-domain nodes exist this
    ///    term defaults to a neutral `0.5`.
    ///
    /// The combined `transferability_score` is then filtered by
    /// [`min_similarity_threshold`](Self::min_similarity_threshold) and the
    /// top-`k` survivors are returned in descending order.
    pub fn recommend_transfers(
        &self,
        graph: &ArchitectureKnowledgeGraph,
        target_domain: &str,
        target_query_embedding: &[f64],
        top_k: usize,
    ) -> Result<Vec<TransferRecommendation>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if target_query_embedding.is_empty() {
            return Err(OptimError::InvalidParameter(
                "recommend_transfers: target_query_embedding must be non-empty".to_string(),
            ));
        }
        if !target_query_embedding.iter().all(|v| v.is_finite()) {
            return Err(OptimError::InvalidParameter(
                "recommend_transfers: target_query_embedding contains non-finite values"
                    .to_string(),
            ));
        }

        // Collect target-domain node ids for the graph-distance term.
        let target_node_ids: Vec<NodeId> = graph
            .nodes()
            .iter()
            .filter(|n| n.domain == target_domain)
            .map(|n| n.id)
            .collect();

        // Pre-compute "closeness to target" for every node via outward
        // propagation from each target node. Closeness is the maximum
        // propagation score received from any target seed.
        let closeness = self.compute_closeness_to_target(graph, &target_node_ids)?;

        let weights = self.transferability_weights;
        let total_weight = weights.total();
        let weight_total = if total_weight > 0.0 {
            total_weight
        } else {
            1.0
        };

        let mut scored: Vec<TransferRecommendation> = Vec::new();
        for node in graph.nodes() {
            if node.domain == target_domain {
                continue;
            }
            let similarity = clip_unit(ArchitectureKnowledgeGraph::cosine_similarity(
                target_query_embedding,
                &node.embedding,
            ));
            let performance = node
                .performance
                .as_ref()
                .map(|p| p.accuracy.clamp(0.0, 1.0))
                .unwrap_or(0.0);
            let graph_proximity = if target_node_ids.is_empty() {
                // No target anchor: use the neutral midpoint so this term
                // neither rewards nor penalises any candidate.
                0.5
            } else {
                clip_unit(closeness.get(&node.id).copied().unwrap_or(0.0))
            };

            let raw_score = weights.similarity_weight * similarity
                + weights.performance_weight * performance
                + weights.graph_distance_weight * graph_proximity;
            let transferability = clip_unit(raw_score / weight_total);

            if transferability < self.min_similarity_threshold {
                continue;
            }

            let epochs_raw = ((1.0 - similarity) * 100.0).round() as i64;
            let recommended_fine_tune_epochs = epochs_raw.clamp(5, 100) as usize;

            scored.push(TransferRecommendation {
                source_node: node.id,
                source_arch_id: node.arch_id.clone(),
                similarity,
                transferability_score: transferability,
                source_performance: node.performance.clone(),
                recommended_fine_tune_epochs,
            });
        }

        scored.sort_by(|a, b| {
            b.transferability_score
                .partial_cmp(&a.transferability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Synthesise a warm-start embedding for the target domain.
    ///
    /// Picks the top-`k` source-domain architectures by accuracy and
    /// returns the performance-weighted average of their embeddings,
    /// normalised to unit length. The resulting vector is intended to be
    /// fed back into [`recommend_transfers`](Self::recommend_transfers) as
    /// the target query embedding.
    pub fn warm_start_embedding(
        &self,
        graph: &ArchitectureKnowledgeGraph,
        target_domain: &str,
        top_k: usize,
    ) -> Result<Vec<f64>> {
        if top_k == 0 {
            return Err(OptimError::InvalidParameter(
                "warm_start_embedding: top_k must be > 0".to_string(),
            ));
        }
        // Gather every non-target node with an accuracy entry.
        let mut scored: Vec<(&[f64], f64)> = graph
            .nodes()
            .iter()
            .filter(|n| n.domain != target_domain)
            .filter_map(|n| {
                n.performance
                    .as_ref()
                    .map(|p| (n.embedding.as_slice(), p.accuracy.clamp(0.0, 1.0)))
            })
            .collect();

        if scored.is_empty() {
            // Fall back to nodes without performance records, treating
            // their weight as a neutral 1.0 so we can still synthesise an
            // embedding when the graph has not been evaluated yet.
            scored = graph
                .nodes()
                .iter()
                .filter(|n| n.domain != target_domain)
                .map(|n| (n.embedding.as_slice(), 1.0))
                .collect();
        }

        if scored.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "warm_start_embedding: no source-domain nodes (target='{}')",
                target_domain
            )));
        }

        // Sort by weight descending and keep the top-k.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Length must match across embeddings; pick the first non-empty
        // embedding length as the canonical width.
        let dim = scored
            .iter()
            .map(|(e, _)| e.len())
            .find(|&n| n > 0)
            .ok_or_else(|| {
                OptimError::InvalidParameter(
                    "warm_start_embedding: all candidate embeddings are empty".to_string(),
                )
            })?;

        let mut accumulator = vec![0.0_f64; dim];
        let mut total_weight = 0.0_f64;
        for (embedding, weight) in scored {
            if embedding.len() != dim || weight <= 0.0 {
                continue;
            }
            if !embedding.iter().all(|v| v.is_finite()) {
                continue;
            }
            for (slot, value) in accumulator.iter_mut().zip(embedding.iter()) {
                *slot += weight * *value;
            }
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return Err(OptimError::InvalidParameter(
                "warm_start_embedding: total weight is zero, no usable embeddings".to_string(),
            ));
        }
        for slot in accumulator.iter_mut() {
            *slot /= total_weight;
        }

        // Normalise to unit length. If the average vector is the zero
        // vector (every candidate cancelled out) fall back to a uniform
        // unit vector along the first axis so the caller still gets a
        // well-defined warm-start.
        let norm = accumulator.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for slot in accumulator.iter_mut() {
                *slot /= norm;
            }
        } else {
            accumulator[0] = 1.0;
        }
        Ok(accumulator)
    }

    /// Build a self-contained subgraph containing the recommended source
    /// architectures plus a synthetic target placeholder.
    ///
    /// The placeholder node's embedding is the warm-start embedding
    /// computed from the recommended set (weighted by transferability),
    /// so the resulting subgraph is a complete artefact that a downstream
    /// NAS run can persist and query independently of the original
    /// knowledge graph.
    pub fn build_transfer_subgraph(
        &self,
        source: &ArchitectureKnowledgeGraph,
        target_domain: &str,
        recommendations: &[TransferRecommendation],
    ) -> Result<ArchitectureKnowledgeGraph> {
        if recommendations.is_empty() {
            return Err(OptimError::InvalidParameter(
                "build_transfer_subgraph: recommendations must be non-empty".to_string(),
            ));
        }

        // Determine the embedding dimensionality and a candidate
        // performance-weighted warm-start. We weight by transferability
        // rather than accuracy so the placeholder reflects the engine's
        // ranking semantics.
        let mut placeholder_embedding: Option<Vec<f64>> = None;
        let mut accumulator: Option<Vec<f64>> = None;
        let mut total_weight = 0.0_f64;

        for rec in recommendations {
            let source_node = source.node(rec.source_node).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "build_transfer_subgraph: source node {} missing from graph",
                    rec.source_node
                ))
            })?;
            let embedding = &source_node.embedding;
            let weight = rec.transferability_score.max(0.0);
            if embedding.is_empty() {
                continue;
            }
            if accumulator.is_none() {
                accumulator = Some(vec![0.0_f64; embedding.len()]);
            }
            if let Some(acc) = accumulator.as_mut() {
                if acc.len() != embedding.len() {
                    return Err(OptimError::ArchitectureError(
                        "build_transfer_subgraph: recommendations have mismatched embedding \
                         dimensionality"
                            .to_string(),
                    ));
                }
                if weight > 0.0 && embedding.iter().all(|v| v.is_finite()) {
                    for (slot, value) in acc.iter_mut().zip(embedding.iter()) {
                        *slot += weight * *value;
                    }
                    total_weight += weight;
                }
            }
        }

        if let Some(mut acc) = accumulator {
            if total_weight > 0.0 {
                for slot in acc.iter_mut() {
                    *slot /= total_weight;
                }
            }
            let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for slot in acc.iter_mut() {
                    *slot /= norm;
                }
            } else {
                acc[0] = 1.0;
            }
            placeholder_embedding = Some(acc);
        }

        let placeholder_embedding = placeholder_embedding.ok_or_else(|| {
            OptimError::ArchitectureError(
                "build_transfer_subgraph: no recommendation produced a usable embedding"
                    .to_string(),
            )
        })?;

        // Now assemble the subgraph. The placeholder is added first so it
        // owns a fixed (and predictable) NodeId of 0, which keeps tests
        // and downstream code simple.
        let mut subgraph = ArchitectureKnowledgeGraph::with_capacity(recommendations.len() + 1);
        let placeholder_id = subgraph.add_architecture(
            format!("{}-warmstart", target_domain),
            placeholder_embedding,
            target_domain.to_string(),
        );

        for rec in recommendations {
            let source_node = source.node(rec.source_node).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "build_transfer_subgraph: source node {} missing from graph",
                    rec.source_node
                ))
            })?;
            let clone_id = subgraph.add_architecture(
                source_node.arch_id.clone(),
                source_node.embedding.clone(),
                source_node.domain.clone(),
            );
            if let Some(perf) = source_node.performance.clone() {
                subgraph.set_performance(clone_id, perf)?;
            }
            // Transferability scores already live in [0, 1]; clamp into
            // the [-1, 1] range expected by Similar edges defensively.
            // `Transfers` edges only require finiteness, but clamping
            // keeps weight semantics consistent across the graph.
            let weight = rec.transferability_score.clamp(0.0, 1.0);
            subgraph.add_edge(clone_id, placeholder_id, RelationType::Transfers, weight)?;
        }

        Ok(subgraph)
    }

    /// Compute a `node_id -> closeness` map describing how reachable each
    /// node is from any target-domain anchor under the configured
    /// random-walk-with-restart settings.
    ///
    /// When there are no target-domain nodes the returned map is empty;
    /// callers should treat that as a signal to use the neutral midpoint
    /// proximity score.
    fn compute_closeness_to_target(
        &self,
        graph: &ArchitectureKnowledgeGraph,
        target_node_ids: &[NodeId],
    ) -> Result<HashMap<NodeId, f64>> {
        let mut closeness: HashMap<NodeId, f64> = HashMap::new();
        for &target in target_node_ids {
            // `propagate_knowledge` only returns Err when the parameters
            // themselves are invalid, which we control. A failure here is
            // therefore a configuration bug that the caller should see.
            let scores = graph.propagate_knowledge(
                target,
                self.knowledge_propagation_decay,
                self.knowledge_propagation_depth,
            )?;
            for (node, score) in scores {
                let entry = closeness.entry(node).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
        }
        // Normalise: any value above 1.0 (the seed weight) gets clipped
        // back so the proximity term stays in `[0, 1]`.
        for value in closeness.values_mut() {
            *value = value.clamp(0.0, 1.0);
        }
        Ok(closeness)
    }
}

/// Collect every embedding belonging to `domain`. Empty embeddings are
/// skipped so downstream metrics never operate on degenerate vectors.
fn collect_domain_embeddings<'a>(
    graph: &'a ArchitectureKnowledgeGraph,
    domain: &str,
) -> Vec<&'a [f64]> {
    graph
        .nodes()
        .iter()
        .filter(|n| n.domain == domain && !n.embedding.is_empty())
        .map(|n| n.embedding.as_slice())
        .collect()
}

/// Clip a similarity into the `[0, 1]` range expected by the public API.
fn clip_unit(value: f64) -> f64 {
    if !value.is_finite() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

/// Aggregate a pairwise cosine grid between two embedding sets according
/// to the chosen domain similarity metric.
fn aggregate_set_similarity(
    metric: DomainSimilarityMetric,
    source: &[&[f64]],
    target: &[&[f64]],
) -> f64 {
    if source.is_empty() || target.is_empty() {
        return 0.0;
    }
    match metric {
        DomainSimilarityMetric::MeanCosine => mean_cosine(source, target),
        DomainSimilarityMetric::MaxCosine => max_cosine(source, target),
        DomainSimilarityMetric::Wasserstein => approx_wasserstein_similarity(source, target),
        DomainSimilarityMetric::CentroidDistance => centroid_similarity(source, target),
    }
}

/// Mean cosine over every pair (`|S| * |T|` cosines).
fn mean_cosine(source: &[&[f64]], target: &[&[f64]]) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for s in source {
        for t in target {
            total += ArchitectureKnowledgeGraph::cosine_similarity(s, t);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Maximum cosine across every pair.
fn max_cosine(source: &[&[f64]], target: &[&[f64]]) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for s in source {
        for t in target {
            let sim = ArchitectureKnowledgeGraph::cosine_similarity(s, t);
            if sim > best {
                best = sim;
            }
        }
    }
    if best.is_finite() {
        best
    } else {
        0.0
    }
}

/// Nearest-neighbour approximation of `1 - W1` in cosine space.
///
/// For each source embedding we take the maximum cosine over all target
/// embeddings — i.e. the cosine to its nearest target — and average those
/// maxima. The complement of that average is the approximate distance;
/// we return `1 - distance == mean(max_cos)`, which lives in `[0, 1]`
/// after clipping.
fn approx_wasserstein_similarity(source: &[&[f64]], target: &[&[f64]]) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for s in source {
        let mut best = f64::NEG_INFINITY;
        for t in target {
            let sim = ArchitectureKnowledgeGraph::cosine_similarity(s, t);
            if sim > best {
                best = sim;
            }
        }
        if best.is_finite() {
            total += best;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Cosine similarity of the two domain centroids.
///
/// The centroids are unweighted means of their member embeddings. If the
/// member counts or embedding widths disagree across sets we still try to
/// compute a sensible centroid — embeddings whose width does not match
/// the first source/target embedding are skipped.
fn centroid_similarity(source: &[&[f64]], target: &[&[f64]]) -> f64 {
    let centroid_s = match compute_centroid(source) {
        Some(c) => c,
        None => return 0.0,
    };
    let centroid_t = match compute_centroid(target) {
        Some(c) => c,
        None => return 0.0,
    };
    ArchitectureKnowledgeGraph::cosine_similarity(&centroid_s, &centroid_t)
}

/// Compute the arithmetic-mean centroid of a non-empty embedding set.
fn compute_centroid(set: &[&[f64]]) -> Option<Vec<f64>> {
    let dim = set.first().map(|e| e.len()).unwrap_or(0);
    if dim == 0 {
        return None;
    }
    let mut centroid = vec![0.0_f64; dim];
    let mut count = 0_usize;
    for embedding in set {
        if embedding.len() != dim {
            continue;
        }
        if !embedding.iter().all(|v| v.is_finite()) {
            continue;
        }
        for (slot, value) in centroid.iter_mut().zip(embedding.iter()) {
            *slot += *value;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let inv = 1.0_f64 / count as f64;
    for slot in centroid.iter_mut() {
        *slot *= inv;
    }
    Some(centroid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(task: &str, accuracy: f64) -> PerformanceRecord {
        PerformanceRecord {
            task_id: task.to_string(),
            accuracy,
            latency_ms: 10.0,
            memory_mb: 64.0,
            training_epochs: 5,
        }
    }

    fn populate_two_domain_graph() -> ArchitectureKnowledgeGraph {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let v1 = graph.add_architecture("resnet", vec![1.0, 0.0, 0.0], "vision");
        let v2 = graph.add_architecture("vgg", vec![0.9, 0.1, 0.0], "vision");
        let v3 = graph.add_architecture("vit", vec![0.8, 0.2, 0.05], "vision");
        let n1 = graph.add_architecture("bert", vec![0.0, 1.0, 0.0], "nlp");
        let n2 = graph.add_architecture("gpt", vec![0.1, 0.9, 0.0], "nlp");
        graph
            .set_performance(v1, make_record("imagenet", 0.78))
            .expect("perf v1");
        graph
            .set_performance(v2, make_record("imagenet", 0.72))
            .expect("perf v2");
        graph
            .set_performance(v3, make_record("imagenet", 0.82))
            .expect("perf v3");
        graph
            .set_performance(n1, make_record("glue", 0.88))
            .expect("perf n1");
        graph
            .set_performance(n2, make_record("glue", 0.85))
            .expect("perf n2");
        graph
    }

    #[test]
    fn test_default_engine_configuration() {
        let engine = CrossDomainTransferEngine::new();
        assert_eq!(
            engine.similarity_metric(),
            DomainSimilarityMetric::MeanCosine
        );
        let weights = engine.transferability_weights();
        assert!((weights.similarity_weight - 0.5).abs() < 1e-12);
        assert!((weights.performance_weight - 0.3).abs() < 1e-12);
        assert!((weights.graph_distance_weight - 0.2).abs() < 1e-12);
        assert!((engine.min_similarity_threshold() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_builder_pattern_chains() {
        let engine = CrossDomainTransferEngine::new()
            .with_metric(DomainSimilarityMetric::MaxCosine)
            .with_thresholds(0.4)
            .with_propagation(0.5, 5)
            .with_weights(TransferabilityWeights {
                similarity_weight: 0.7,
                performance_weight: 0.2,
                graph_distance_weight: 0.1,
            });
        assert_eq!(
            engine.similarity_metric(),
            DomainSimilarityMetric::MaxCosine
        );
        assert!((engine.min_similarity_threshold() - 0.4).abs() < 1e-12);
        let w = engine.transferability_weights();
        assert!((w.similarity_weight - 0.7).abs() < 1e-12);
        assert!((w.performance_weight - 0.2).abs() < 1e-12);
        assert!((w.graph_distance_weight - 0.1).abs() < 1e-12);

        // Invalid propagation parameters fall back to safe defaults instead
        // of failing — the builder pattern stays infallible.
        let safe = CrossDomainTransferEngine::new().with_propagation(f64::NAN, 0);
        assert!(safe.knowledge_propagation_decay > 0.0 && safe.knowledge_propagation_decay < 1.0);
        assert!(safe.knowledge_propagation_depth >= 1);

        // Threshold clamps into [-1, 1].
        let clamped = CrossDomainTransferEngine::new().with_thresholds(5.0);
        assert!(clamped.min_similarity_threshold() <= 1.0);
        let clamped_neg = CrossDomainTransferEngine::new().with_thresholds(-5.0);
        assert!(clamped_neg.min_similarity_threshold() >= -1.0);
    }

    #[test]
    fn test_compute_domain_similarity_mean_cosine_known_value() {
        // Construct two trivial domains with hand-computable cosine grid.
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![1.0, 0.0], "source");
        graph.add_architecture("s2", vec![0.0, 1.0], "source");
        graph.add_architecture("t1", vec![1.0, 0.0], "target");
        graph.add_architecture("t2", vec![0.0, 1.0], "target");

        // Pairwise cosines:
        // s1 vs t1 = 1, s1 vs t2 = 0
        // s2 vs t1 = 0, s2 vs t2 = 1
        // mean = (1 + 0 + 0 + 1) / 4 = 0.5
        let engine =
            CrossDomainTransferEngine::new().with_metric(DomainSimilarityMetric::MeanCosine);
        let report = engine
            .compute_domain_similarity(&graph, "source", "target")
            .expect("similarity");
        assert!(
            (report.similarity - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            report.similarity
        );
        assert_eq!(report.source_node_count, 2);
        assert_eq!(report.target_node_count, 2);
        assert_eq!(report.metric, DomainSimilarityMetric::MeanCosine);
    }

    #[test]
    fn test_compute_domain_similarity_empty_domain_errors() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![1.0, 0.0], "source");
        let engine = CrossDomainTransferEngine::new();

        // Target missing.
        let err = engine.compute_domain_similarity(&graph, "source", "missing");
        assert!(err.is_err());

        // Source missing.
        let err = engine.compute_domain_similarity(&graph, "missing", "source");
        assert!(err.is_err());

        // Both missing.
        let err = engine.compute_domain_similarity(&graph, "x", "y");
        assert!(err.is_err());
    }

    #[test]
    fn test_compute_domain_similarity_self_is_one() {
        // Embeddings that are all parallel to the same direction so every
        // pairwise cosine is exactly 1 regardless of magnitude. This lets
        // us assert ~1.0 across every metric, including the mean-based
        // ones.
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![1.0, 0.0, 0.0], "vision");
        graph.add_architecture("s2", vec![2.0, 0.0, 0.0], "vision");
        graph.add_architecture("s3", vec![0.5, 0.0, 0.0], "vision");

        for metric in [
            DomainSimilarityMetric::MeanCosine,
            DomainSimilarityMetric::MaxCosine,
            DomainSimilarityMetric::Wasserstein,
            DomainSimilarityMetric::CentroidDistance,
        ] {
            let engine = CrossDomainTransferEngine::new().with_metric(metric);
            let report = engine
                .compute_domain_similarity(&graph, "vision", "vision")
                .expect("self similarity");
            assert!(
                (report.similarity - 1.0).abs() < 1e-6,
                "metric {:?} self-similarity should be exactly 1, got {}",
                metric,
                report.similarity
            );
        }
    }

    #[test]
    fn test_recommend_transfers_returns_top_k_sorted_descending() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 3)
            .expect("recommendations");
        assert!(recs.len() <= 3);
        assert!(!recs.is_empty());
        for window in recs.windows(2) {
            assert!(
                window[0].transferability_score >= window[1].transferability_score,
                "transferability not sorted descending"
            );
        }
        // The top recommendation for the [1, 0, 0] query should be one of
        // the vision architectures (closest in embedding space).
        let top_id = &recs[0].source_arch_id;
        assert!(
            top_id == "resnet" || top_id == "vgg" || top_id == "vit",
            "expected a vision arch, got {}",
            top_id
        );
    }

    #[test]
    fn test_recommend_transfers_respects_threshold() {
        let graph = populate_two_domain_graph();
        // Threshold above 1.0 cannot be hit, so every candidate is filtered.
        let engine = CrossDomainTransferEngine::new().with_thresholds(1.0);
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 5)
            .expect("recommendations");
        assert!(
            recs.is_empty(),
            "impossibly high threshold should yield no recommendations, got {} items",
            recs.len()
        );

        // Threshold of 0.0 should accept everything.
        let engine = CrossDomainTransferEngine::new().with_thresholds(0.0);
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 10)
            .expect("recommendations");
        assert_eq!(
            recs.len(),
            5,
            "all five non-target nodes should be returned"
        );
    }

    #[test]
    fn test_recommend_transfers_excludes_target_domain() {
        let mut graph = populate_two_domain_graph();
        // Add a vision node belonging to the target domain.
        graph.add_architecture("medical-net", vec![1.0, 0.0, 0.0], "medical");
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 10)
            .expect("recommendations");
        for rec in &recs {
            assert_ne!(rec.source_arch_id, "medical-net");
        }
    }

    #[test]
    fn test_recommend_transfers_empty_top_k_returns_empty() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 0)
            .expect("zero top_k");
        assert!(recs.is_empty());
    }

    #[test]
    fn test_recommend_transfers_invalid_query_errors() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        // Empty embedding.
        let err = engine.recommend_transfers(&graph, "medical", &[], 3);
        assert!(err.is_err());
        // Non-finite values.
        let err = engine.recommend_transfers(&graph, "medical", &[f64::NAN, 0.0, 0.0], 3);
        assert!(err.is_err());
    }

    #[test]
    fn test_warm_start_embedding_unit_length() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let warm = engine
            .warm_start_embedding(&graph, "medical", 3)
            .expect("warm start");
        let norm = warm.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "warm start embedding norm should be 1, got {}",
            norm
        );
    }

    #[test]
    fn test_warm_start_embedding_with_no_source_nodes_errors() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        // Only target-domain nodes exist.
        graph.add_architecture("only", vec![1.0, 0.0], "target");
        let engine = CrossDomainTransferEngine::new();
        let err = engine.warm_start_embedding(&graph, "target", 3);
        assert!(err.is_err());
    }

    #[test]
    fn test_warm_start_embedding_top_k_zero_errors() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let err = engine.warm_start_embedding(&graph, "medical", 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_build_transfer_subgraph_node_count() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 3)
            .expect("recommendations");
        let subgraph = engine
            .build_transfer_subgraph(&graph, "medical", &recs)
            .expect("subgraph");
        assert_eq!(subgraph.len(), recs.len() + 1);
    }

    #[test]
    fn test_build_transfer_subgraph_transfers_edges_present() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 3)
            .expect("recommendations");
        let subgraph = engine
            .build_transfer_subgraph(&graph, "medical", &recs)
            .expect("subgraph");
        // Placeholder is node 0; every other node should have a Transfers
        // edge pointing to it.
        let placeholder_id: NodeId = 0;
        let placeholder = subgraph.node(placeholder_id).expect("placeholder present");
        assert_eq!(placeholder.domain, "medical");

        let mut transfers_edges = 0_usize;
        for node in subgraph.nodes() {
            if node.id == placeholder_id {
                continue;
            }
            let neighbours = subgraph.neighbors(node.id);
            assert!(
                !neighbours.is_empty(),
                "source clone {} should have at least one outgoing edge",
                node.id
            );
            let to_placeholder: Vec<_> = neighbours
                .iter()
                .filter(|(target, relation, _)| {
                    *target == placeholder_id && *relation == RelationType::Transfers
                })
                .collect();
            assert_eq!(
                to_placeholder.len(),
                1,
                "expected exactly one Transfers edge from {} to placeholder, found {}",
                node.id,
                to_placeholder.len()
            );
            transfers_edges += 1;
        }
        assert_eq!(transfers_edges, recs.len());
    }

    #[test]
    fn test_build_transfer_subgraph_empty_recommendations_errors() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let err = engine.build_transfer_subgraph(&graph, "medical", &[]);
        assert!(err.is_err());
    }

    #[test]
    fn test_recommended_fine_tune_epochs_range() {
        let graph = populate_two_domain_graph();
        let engine = CrossDomainTransferEngine::new();
        let recs = engine
            .recommend_transfers(&graph, "medical", &[1.0, 0.0, 0.0], 10)
            .expect("recommendations");
        assert!(!recs.is_empty());
        for rec in &recs {
            assert!(
                rec.recommended_fine_tune_epochs >= 5,
                "epochs {} should be >= 5",
                rec.recommended_fine_tune_epochs
            );
            assert!(
                rec.recommended_fine_tune_epochs <= 100,
                "epochs {} should be <= 100",
                rec.recommended_fine_tune_epochs
            );
        }
    }

    #[test]
    fn test_max_cosine_metric_picks_best_pair() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![1.0, 0.0], "source");
        graph.add_architecture("s2", vec![0.0, 1.0], "source");
        graph.add_architecture("t1", vec![0.95, 0.05], "target");
        graph.add_architecture("t2", vec![0.0, -1.0], "target");
        let engine =
            CrossDomainTransferEngine::new().with_metric(DomainSimilarityMetric::MaxCosine);
        let report = engine
            .compute_domain_similarity(&graph, "source", "target")
            .expect("max cosine");
        // The best pair is s1 / t1 with cosine ~ 0.998.
        assert!(
            report.similarity > 0.99,
            "expected max cosine close to 1, got {}",
            report.similarity
        );
    }

    #[test]
    fn test_centroid_metric_returns_expected_value() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![2.0, 0.0], "source");
        graph.add_architecture("s2", vec![0.0, 2.0], "source");
        graph.add_architecture("t1", vec![1.0, 0.0], "target");
        graph.add_architecture("t2", vec![0.0, 1.0], "target");
        let engine =
            CrossDomainTransferEngine::new().with_metric(DomainSimilarityMetric::CentroidDistance);
        let report = engine
            .compute_domain_similarity(&graph, "source", "target")
            .expect("centroid");
        // Centroids are (1, 1) and (0.5, 0.5); their cosine is exactly 1.
        assert!(
            (report.similarity - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            report.similarity
        );
    }

    #[test]
    fn test_wasserstein_metric_returns_expected_value() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("s1", vec![1.0, 0.0], "source");
        graph.add_architecture("s2", vec![0.0, 1.0], "source");
        graph.add_architecture("t1", vec![1.0, 0.0], "target");
        graph.add_architecture("t2", vec![0.0, 1.0], "target");
        let engine =
            CrossDomainTransferEngine::new().with_metric(DomainSimilarityMetric::Wasserstein);
        let report = engine
            .compute_domain_similarity(&graph, "source", "target")
            .expect("wasserstein");
        // Each source has a perfect match in the target set, so the mean
        // nearest-neighbour cosine is exactly 1.
        assert!(
            (report.similarity - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            report.similarity
        );
    }

    #[test]
    fn test_transferability_weights_default_sums_to_one() {
        let w = TransferabilityWeights::default();
        assert!((w.total() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_domain_profile_round_trip() {
        let profile = DomainProfile {
            domain_name: "medical".to_string(),
            task_type: "classification".to_string(),
            input_dimensionality: 512,
            output_dimensionality: 10,
            typical_dataset_size: 50_000,
            priority_metric: "accuracy".to_string(),
        };
        let json = serde_json::to_string(&profile).expect("serialise");
        let back: DomainProfile = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(profile, back);
    }

    #[test]
    fn test_warm_start_embedding_uses_performance_weighting() {
        // Add two non-target nodes; only one has high accuracy. The
        // resulting warm start should be aligned with that node's
        // embedding direction.
        let mut graph = ArchitectureKnowledgeGraph::new();
        let a = graph.add_architecture("low", vec![1.0, 0.0], "source-low");
        let b = graph.add_architecture("high", vec![0.0, 1.0], "source-high");
        graph
            .set_performance(a, make_record("task", 0.10))
            .expect("perf a");
        graph
            .set_performance(b, make_record("task", 0.99))
            .expect("perf b");
        let engine = CrossDomainTransferEngine::new();
        let warm = engine
            .warm_start_embedding(&graph, "target", 2)
            .expect("warm start");
        assert!(
            warm[1] > warm[0],
            "high-performing axis should dominate, got {:?}",
            warm
        );
    }

    #[test]
    fn test_recommend_transfers_with_propagation_anchors() {
        // When the target domain has anchor nodes wired to the source via
        // Transfers edges, the graph-distance term should boost the score
        // of architectures connected to the target.
        let mut graph = ArchitectureKnowledgeGraph::new();
        let src_a = graph.add_architecture("connected", vec![0.0, 1.0, 0.0], "source");
        let src_b = graph.add_architecture("isolated", vec![0.0, 1.0, 0.0], "source");
        let tgt = graph.add_architecture("anchor", vec![1.0, 0.0, 0.0], "target");
        graph
            .add_edge(tgt, src_a, RelationType::Transfers, 0.95)
            .expect("transfers edge");
        graph
            .set_performance(src_a, make_record("task", 0.80))
            .expect("perf a");
        graph
            .set_performance(src_b, make_record("task", 0.80))
            .expect("perf b");

        let engine = CrossDomainTransferEngine::new()
            .with_propagation(0.85, 3)
            .with_weights(TransferabilityWeights {
                similarity_weight: 0.0,
                performance_weight: 0.0,
                graph_distance_weight: 1.0,
            });
        let recs = engine
            .recommend_transfers(&graph, "target", &[0.0, 1.0, 0.0], 2)
            .expect("recommendations");
        assert_eq!(recs.len(), 2);
        // The connected source must rank ahead of the isolated one.
        assert_eq!(recs[0].source_arch_id, "connected");

        // Ensure the target anchor itself never surfaces as a recommendation.
        for rec in &recs {
            assert_ne!(rec.source_node, tgt);
        }
    }
}
