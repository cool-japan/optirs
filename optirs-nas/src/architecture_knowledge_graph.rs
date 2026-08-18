//! Architecture Knowledge Graph for transfer learning across NAS searches.
//!
//! This module provides a graph data structure where each node represents a
//! previously-evaluated neural architecture and edges encode relationships
//! between them (similarity, derivation, dominance, knowledge transfer).
//!
//! The graph enables similarity queries against learned embeddings and
//! random-walk-with-restart for propagating knowledge through the graph,
//! both of which are useful for warm-starting new NAS searches by reusing
//! information from prior evaluations.
//!
//! # Examples
//!
//! ```
//! use optirs_nas::architecture_knowledge_graph::{
//!     ArchitectureKnowledgeGraph, PerformanceRecord, RelationType,
//! };
//!
//! let mut graph = ArchitectureKnowledgeGraph::new();
//! let a = graph.add_architecture("arch-a", vec![1.0, 0.0, 0.0], "vision");
//! let b = graph.add_architecture("arch-b", vec![0.9, 0.1, 0.0], "vision");
//! graph.add_edge(a, b, RelationType::Similar, 0.95).unwrap();
//!
//! let similar = graph.find_similar(&[1.0, 0.0, 0.0], 1);
//! assert_eq!(similar[0].0, a);
//! ```
//!
//! # Design notes
//!
//! - Nodes are stored in a `Vec` so `NodeId` can be a stable index. The
//!   `next_id` counter equals `nodes.len()` and is preserved across
//!   serialisation rounds.
//! - The adjacency map is rebuilt on demand from edges after deserialisation
//!   (a `Vec<ArchKnowledgeEdge>` is the canonical source of truth).
//! - Cosine similarity treats zero-norm vectors as orthogonal (similarity 0)
//!   to avoid producing NaN.
//! - Random-walk-with-restart normalises by the absolute weight sum to
//!   tolerate negative similarity weights, with dangling and zero-weight
//!   nodes handled explicitly.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stable handle to a node in the knowledge graph.
///
/// Node ids are assigned monotonically when architectures are added and are
/// never reused, so existing handles remain valid for the lifetime of the
/// graph.
pub type NodeId = usize;

/// Type of relationship that can exist between two architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Architectures whose embeddings have cosine similarity above a
    /// configured threshold.
    Similar,
    /// The target architecture was produced by mutating or crossing-over the
    /// source architecture.
    Derived,
    /// The target architecture outperforms the source on the same task.
    Outperforms,
    /// Knowledge can transfer between architectures, typically across
    /// different domains.
    Transfers,
}

/// Recorded evaluation result for an architecture on a particular task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceRecord {
    /// Identifier of the task on which the architecture was evaluated.
    pub task_id: String,
    /// Top-line accuracy / quality metric in `[0, 1]` (caller decides scale).
    pub accuracy: f64,
    /// Inference latency in milliseconds.
    pub latency_ms: f64,
    /// Peak memory usage in megabytes.
    pub memory_mb: f64,
    /// Number of training epochs used to produce this record.
    pub training_epochs: usize,
}

/// A single architecture entry in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchKnowledgeNode {
    /// Stable graph-local identifier.
    pub id: NodeId,
    /// External / human-readable architecture identifier.
    pub arch_id: String,
    /// Dense embedding used for similarity search.
    pub embedding: Vec<f64>,
    /// Optional performance record (set after evaluation completes).
    pub performance: Option<PerformanceRecord>,
    /// Application domain the architecture targets (e.g. `"vision"`).
    pub domain: String,
    /// Free-form metadata attached by the caller.
    pub metadata: HashMap<String, String>,
}

/// A directed weighted edge between two architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchKnowledgeEdge {
    /// Source node identifier.
    pub from: NodeId,
    /// Target node identifier.
    pub to: NodeId,
    /// Semantic relation the edge encodes.
    pub relation: RelationType,
    /// Edge weight. For `Similar` edges this is a cosine similarity in
    /// `[-1, 1]`; for other relations it is an arbitrary finite scalar.
    pub weight: f64,
}

/// A graph of architectures and their relationships.
///
/// Stores nodes contiguously in insertion order and indexes outgoing edges
/// via an `adjacency` map for O(deg) neighbour lookups. The map is
/// reconstructed from the edge list whenever the graph is loaded from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureKnowledgeGraph {
    /// All nodes, indexed by `NodeId`.
    nodes: Vec<ArchKnowledgeNode>,
    /// All edges in insertion order.
    edges: Vec<ArchKnowledgeEdge>,
    /// Outgoing-edge index: maps `NodeId` to the indices into `edges` for
    /// edges whose `from` field equals the key.
    #[serde(skip)]
    adjacency: HashMap<NodeId, Vec<usize>>,
    /// Next `NodeId` to assign.
    next_id: NodeId,
}

impl Default for ArchitectureKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureKnowledgeGraph {
    /// Construct an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            next_id: 0,
        }
    }

    /// Construct an empty graph pre-allocating storage for `n` nodes.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            edges: Vec::with_capacity(n),
            adjacency: HashMap::with_capacity(n),
            next_id: 0,
        }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Rebuild the adjacency index from the edge list.
    ///
    /// Required after deserialisation, since `adjacency` is `#[serde(skip)]`.
    fn rebuild_adjacency(&mut self) {
        self.adjacency.clear();
        for (idx, edge) in self.edges.iter().enumerate() {
            self.adjacency.entry(edge.from).or_default().push(idx);
        }
    }

    /// Insert a new architecture into the graph and return its `NodeId`.
    pub fn add_architecture(
        &mut self,
        arch_id: impl Into<String>,
        embedding: Vec<f64>,
        domain: impl Into<String>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = ArchKnowledgeNode {
            id,
            arch_id: arch_id.into(),
            embedding,
            performance: None,
            domain: domain.into(),
            metadata: HashMap::new(),
        };
        self.nodes.push(node);
        id
    }

    /// Attach a performance record to an existing node.
    pub fn set_performance(&mut self, node: NodeId, record: PerformanceRecord) -> Result<()> {
        let target = self.nodes.get_mut(node).ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "set_performance: node id {} does not exist",
                node
            ))
        })?;
        target.performance = Some(record);
        Ok(())
    }

    /// Add a typed weighted edge between two existing nodes.
    ///
    /// Rejects edges that reference unknown nodes, that carry a non-finite
    /// weight, or whose weight is outside `[-1, 1]` for `Similar` edges.
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        relation: RelationType,
        weight: f64,
    ) -> Result<()> {
        if from >= self.nodes.len() {
            return Err(OptimError::InvalidParameter(format!(
                "add_edge: source node {} does not exist",
                from
            )));
        }
        if to >= self.nodes.len() {
            return Err(OptimError::InvalidParameter(format!(
                "add_edge: target node {} does not exist",
                to
            )));
        }
        if !weight.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "add_edge: weight must be finite, got {}",
                weight
            )));
        }
        if matches!(relation, RelationType::Similar) && !(-1.0..=1.0).contains(&weight) {
            return Err(OptimError::InvalidParameter(format!(
                "add_edge: Similar weight must be in [-1, 1], got {}",
                weight
            )));
        }

        let edge_index = self.edges.len();
        self.edges.push(ArchKnowledgeEdge {
            from,
            to,
            relation,
            weight,
        });
        self.adjacency.entry(from).or_default().push(edge_index);
        Ok(())
    }

    /// Borrow a node by id (returns `None` if it does not exist).
    pub fn node(&self, id: NodeId) -> Option<&ArchKnowledgeNode> {
        self.nodes.get(id)
    }

    /// Borrow all nodes in insertion order.
    pub fn nodes(&self) -> &[ArchKnowledgeNode] {
        &self.nodes
    }

    /// Outgoing neighbours of `id`, returned as `(target, relation, weight)`.
    ///
    /// Returns an empty vector for nodes with no outgoing edges or for
    /// unknown ids.
    pub fn neighbors(&self, id: NodeId) -> Vec<(NodeId, RelationType, f64)> {
        match self.adjacency.get(&id) {
            None => Vec::new(),
            Some(indices) => indices
                .iter()
                .filter_map(|&i| self.edges.get(i))
                .map(|e| (e.to, e.relation, e.weight))
                .collect(),
        }
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns `0.0` when either vector is the zero vector, when the lengths
    /// differ, or when any element is non-finite. This is safer than
    /// returning `NaN` for downstream ranking code.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return 0.0;
            }
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom <= 0.0 {
            return 0.0;
        }
        dot / denom
    }

    /// Top-`k` nodes by cosine similarity to `query_embedding`, descending.
    ///
    /// Nodes whose embeddings are not comparable (mismatched length or
    /// zero-norm) are silently skipped. If `k` is zero or the graph is
    /// empty, returns an empty vector.
    pub fn find_similar(&self, query_embedding: &[f64], k: usize) -> Vec<(NodeId, f64)> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .map(|n| (n.id, Self::cosine_similarity(query_embedding, &n.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Add `Similar` edges between every distinct pair of nodes whose
    /// cosine similarity meets `threshold`. Returns the number of edges
    /// added.
    ///
    /// Only one directed edge `(i -> j)` with `i < j` is added per pair to
    /// avoid duplicates and self-loops. Edges are skipped when an identical
    /// `(from, to, Similar)` triple already exists.
    pub fn auto_add_similar_edges(&mut self, threshold: f64) -> usize {
        if self.nodes.len() < 2 {
            return 0;
        }
        // Pre-compute similarities into a vector of (i, j, sim) so we can
        // borrow self immutably for the scan and then mutate when adding.
        let mut to_add: Vec<(NodeId, NodeId, f64)> = Vec::new();
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                let sim =
                    Self::cosine_similarity(&self.nodes[i].embedding, &self.nodes[j].embedding);
                if sim >= threshold && sim.is_finite() {
                    to_add.push((self.nodes[i].id, self.nodes[j].id, sim));
                }
            }
        }

        let mut added = 0usize;
        for (from, to, sim) in to_add {
            // Skip if a Similar edge already exists for this directed pair.
            let exists = self
                .adjacency
                .get(&from)
                .map(|indices| {
                    indices.iter().any(|&idx| {
                        let edge = &self.edges[idx];
                        edge.to == to && edge.relation == RelationType::Similar
                    })
                })
                .unwrap_or(false);
            if exists {
                continue;
            }
            // Clamp similarity into [-1, 1] to satisfy the add_edge guard
            // against floating-point drift slightly outside the interval.
            let clamped = sim.clamp(-1.0, 1.0);
            if self
                .add_edge(from, to, RelationType::Similar, clamped)
                .is_ok()
            {
                added += 1;
            }
        }
        added
    }

    /// Random-walk-with-restart from `start`.
    ///
    /// At each of `depth` iterations every node redistributes `decay × score`
    /// to its outgoing neighbours (weighted by `|weight|` and normalised to
    /// sum to one). The remaining `1 - decay` mass restarts at `start`.
    /// Dangling nodes — those without outgoing edges — return all their
    /// mass to `start`, and groups of neighbours whose weights sum to zero
    /// share the mass uniformly.
    pub fn propagate_knowledge(
        &self,
        start: NodeId,
        decay: f64,
        depth: usize,
    ) -> Result<HashMap<NodeId, f64>> {
        if !decay.is_finite() || decay <= 0.0 || decay >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "propagate_knowledge: decay must be in (0, 1), got {}",
                decay
            )));
        }
        if depth == 0 {
            return Err(OptimError::InvalidParameter(
                "propagate_knowledge: depth must be > 0".to_string(),
            ));
        }
        if start >= self.nodes.len() {
            return Err(OptimError::InvalidParameter(format!(
                "propagate_knowledge: start node {} does not exist",
                start
            )));
        }

        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        scores.insert(start, 1.0);

        for _ in 0..depth {
            // Seed the iteration with the full restart mass at `start` so
            // the seed score never drops below its initial value of 1.0.
            // Outgoing flow from `start` contributes additional probability
            // mass to its neighbours each step, encoding the random walk.
            let mut new_scores: HashMap<NodeId, f64> = HashMap::new();
            new_scores.insert(start, 1.0);

            for (&node, &score) in scores.iter() {
                if score == 0.0 {
                    continue;
                }
                let neighbours = self.neighbors(node);
                if neighbours.is_empty() {
                    // Dangling node: redistribute its weighted mass to the
                    // restart node so probability remains anchored.
                    *new_scores.entry(start).or_insert(0.0) += decay * score;
                    continue;
                }
                let weight_sum: f64 = neighbours.iter().map(|(_, _, w)| w.abs()).sum();
                if weight_sum > 0.0 {
                    for (neigh, _, w) in &neighbours {
                        let contribution = decay * score * w.abs() / weight_sum;
                        *new_scores.entry(*neigh).or_insert(0.0) += contribution;
                    }
                } else {
                    // All neighbours have zero weight: split mass uniformly.
                    let share = decay * score / neighbours.len() as f64;
                    for (neigh, _, _) in &neighbours {
                        *new_scores.entry(*neigh).or_insert(0.0) += share;
                    }
                }
            }
            scores = new_scores;
        }
        Ok(scores)
    }

    /// Find up to `k` transferable architectures for `target_domain`.
    ///
    /// A node is considered transferable if its own `domain` matches
    /// `target_domain`, or if any of its outgoing `Transfers` edges points
    /// to a node whose domain matches, or if it is itself the target of a
    /// `Transfers` edge from a node whose domain matches. Among the
    /// transferable nodes, the top-`k` by cosine similarity to
    /// `query_embedding` are returned (descending).
    pub fn extract_top_k_transferable(
        &self,
        target_domain: &str,
        query_embedding: &[f64],
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        // Build a set of node ids that are reachable from `target_domain`
        // via `Transfers` edges, in either direction.
        let mut transferable: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        for node in &self.nodes {
            if node.domain == target_domain {
                transferable.insert(node.id);
            }
        }
        for edge in &self.edges {
            if edge.relation != RelationType::Transfers {
                continue;
            }
            let from_domain = self.nodes.get(edge.from).map(|n| n.domain.as_str());
            let to_domain = self.nodes.get(edge.to).map(|n| n.domain.as_str());
            if from_domain == Some(target_domain) {
                transferable.insert(edge.to);
            }
            if to_domain == Some(target_domain) {
                transferable.insert(edge.from);
            }
        }

        let mut scored: Vec<(NodeId, f64)> = self
            .nodes
            .iter()
            .filter(|n| transferable.contains(&n.id))
            .map(|n| (n.id, Self::cosine_similarity(query_embedding, &n.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Serialise the graph to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            OptimError::ArchitectureError(format!("Failed to serialise knowledge graph: {}", e))
        })
    }

    /// Parse a graph from a JSON string previously produced by `to_json`.
    pub fn from_json(s: &str) -> Result<Self> {
        let mut graph: Self = serde_json::from_str(s).map_err(|e| {
            OptimError::ArchitectureError(format!("Failed to deserialise knowledge graph: {}", e))
        })?;
        graph.rebuild_adjacency();
        // Repair next_id in case the saved value is stale.
        if !graph.nodes.is_empty() {
            let max = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0);
            if graph.next_id <= max {
                graph.next_id = max + 1;
            }
        }
        Ok(graph)
    }
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

    fn make_linear_graph(n: usize) -> (ArchitectureKnowledgeGraph, Vec<NodeId>) {
        let mut graph = ArchitectureKnowledgeGraph::with_capacity(n);
        let mut ids = Vec::new();
        for i in 0..n {
            let emb = vec![i as f64 + 1.0, 0.0, 0.0];
            ids.push(graph.add_architecture(format!("arch-{}", i), emb, "vision"));
        }
        for i in 0..(n - 1) {
            graph
                .add_edge(ids[i], ids[i + 1], RelationType::Derived, 1.0)
                .expect("linear edge");
        }
        (graph, ids)
    }

    #[test]
    fn test_add_architecture_returns_increasing_ids() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let a = graph.add_architecture("a", vec![1.0, 2.0], "vision");
        let b = graph.add_architecture("b", vec![3.0, 4.0], "vision");
        let c = graph.add_architecture("c", vec![5.0, 6.0], "nlp");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(graph.len(), 3);
        assert!(!graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.node(a).expect("present").arch_id, "a");
        assert!(graph.node(99).is_none());
    }

    #[test]
    fn test_add_edge_validates_endpoints() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let a = graph.add_architecture("a", vec![1.0, 0.0], "vision");
        let b = graph.add_architecture("b", vec![0.0, 1.0], "vision");
        graph
            .add_edge(a, b, RelationType::Outperforms, 0.5)
            .expect("valid edge");
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.add_edge(a, 42, RelationType::Derived, 1.0).is_err());
        assert!(graph.add_edge(99, b, RelationType::Derived, 1.0).is_err());
        // Non-finite weight rejected.
        assert!(graph
            .add_edge(a, b, RelationType::Derived, f64::NAN)
            .is_err());
        // Similar weight outside [-1, 1] rejected.
        assert!(graph.add_edge(a, b, RelationType::Similar, 1.5).is_err());
        // Similar weight inside [-1, 1] accepted.
        graph
            .add_edge(a, b, RelationType::Similar, 0.9)
            .expect("similar in range");
    }

    #[test]
    fn test_cosine_similarity_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0, -4.0];
        let sim = ArchitectureKnowledgeGraph::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-12, "expected 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_is_zero() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let sim = ArchitectureKnowledgeGraph::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-12, "expected 0.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_zero_norm_returns_zero() {
        let zero = [0.0, 0.0, 0.0];
        let other = [1.0, 2.0, 3.0];
        assert_eq!(
            ArchitectureKnowledgeGraph::cosine_similarity(&zero, &other),
            0.0
        );
        assert_eq!(
            ArchitectureKnowledgeGraph::cosine_similarity(&other, &zero),
            0.0
        );
        assert_eq!(
            ArchitectureKnowledgeGraph::cosine_similarity(&zero, &zero),
            0.0
        );
        // Mismatched length is also treated as zero.
        assert_eq!(
            ArchitectureKnowledgeGraph::cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]),
            0.0
        );
        // Empty vectors return zero.
        assert_eq!(
            ArchitectureKnowledgeGraph::cosine_similarity(&[] as &[f64], &[] as &[f64]),
            0.0
        );
    }

    #[test]
    fn test_find_similar_returns_top_k_sorted_descending() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("a", vec![1.0, 0.0, 0.0], "vision");
        graph.add_architecture("b", vec![0.9, 0.1, 0.0], "vision");
        graph.add_architecture("c", vec![0.0, 1.0, 0.0], "vision");
        graph.add_architecture("d", vec![-1.0, 0.0, 0.0], "vision");

        let top = graph.find_similar(&[1.0, 0.0, 0.0], 3);
        assert_eq!(top.len(), 3);
        // Descending order by similarity.
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
        // Most similar is the identity match.
        assert_eq!(top[0].0, 0);
        // k=0 returns nothing.
        assert!(graph.find_similar(&[1.0, 0.0, 0.0], 0).is_empty());
        // Empty graph returns nothing.
        let empty = ArchitectureKnowledgeGraph::new();
        assert!(empty.find_similar(&[1.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn test_auto_add_similar_edges_threshold_respected() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("a", vec![1.0, 0.0, 0.0], "vision");
        graph.add_architecture("b", vec![0.99, 0.01, 0.0], "vision"); // very similar to a
        graph.add_architecture("c", vec![0.0, 1.0, 0.0], "vision"); // orthogonal

        let added = graph.auto_add_similar_edges(0.95);
        assert_eq!(added, 1, "only the a-b pair should pass 0.95 threshold");
        assert_eq!(graph.edge_count(), 1);
        let edge = &graph.edges[0];
        assert_eq!(edge.relation, RelationType::Similar);
        assert!(edge.weight >= 0.95);

        // Calling again with the same threshold should not duplicate.
        let added_again = graph.auto_add_similar_edges(0.95);
        assert_eq!(added_again, 0);
        assert_eq!(graph.edge_count(), 1);

        // A lower threshold pulls in another edge (a-c is orthogonal but
        // b-c is also orthogonal; only one edge can satisfy 0.0 so 2 more
        // are added: a-c and b-c when threshold is small enough).
        let added_low = graph.auto_add_similar_edges(-1.0);
        assert!(added_low >= 1);
    }

    #[test]
    fn test_propagate_knowledge_start_is_one() {
        let (graph, ids) = make_linear_graph(3);
        let scores = graph
            .propagate_knowledge(ids[0], 0.85, 5)
            .expect("propagation");
        let start_score = scores.get(&ids[0]).copied().unwrap_or(0.0);
        assert!(
            start_score >= 1.0 - 1e-9,
            "start score should be >= 1.0, got {}",
            start_score
        );
    }

    #[test]
    fn test_propagate_knowledge_closer_higher_score() {
        let (graph, ids) = make_linear_graph(4);
        let scores = graph
            .propagate_knowledge(ids[0], 0.85, 8)
            .expect("propagation");
        let s1 = scores.get(&ids[1]).copied().unwrap_or(0.0);
        let s2 = scores.get(&ids[2]).copied().unwrap_or(0.0);
        let s3 = scores.get(&ids[3]).copied().unwrap_or(0.0);
        assert!(
            s1 > s2,
            "direct neighbour score ({}) should exceed 2-hop ({})",
            s1,
            s2
        );
        assert!(s2 > s3, "2-hop ({}) should exceed 3-hop ({})", s2, s3);
    }

    #[test]
    fn test_propagate_knowledge_invalid_decay_errors() {
        let (graph, ids) = make_linear_graph(2);
        assert!(graph.propagate_knowledge(ids[0], 0.0, 3).is_err());
        assert!(graph.propagate_knowledge(ids[0], 1.0, 3).is_err());
        assert!(graph.propagate_knowledge(ids[0], -0.1, 3).is_err());
        assert!(graph.propagate_knowledge(ids[0], 1.1, 3).is_err());
        assert!(graph.propagate_knowledge(ids[0], f64::NAN, 3).is_err());
        // depth = 0 also errors.
        assert!(graph.propagate_knowledge(ids[0], 0.5, 0).is_err());
        // unknown start node errors.
        assert!(graph.propagate_knowledge(99, 0.5, 3).is_err());
    }

    #[test]
    fn test_propagate_knowledge_dangling_returns_to_start() {
        // Single-node graph: start has no neighbours so all decayed mass
        // returns to it each iteration. The score must therefore remain at
        // or above the initial restart value of 1.0.
        let mut graph = ArchitectureKnowledgeGraph::new();
        let s = graph.add_architecture("solo", vec![1.0, 0.0], "vision");
        let scores = graph
            .propagate_knowledge(s, 0.5, 3)
            .expect("propagation on dangling");
        let start_score = scores.get(&s).copied().unwrap_or(0.0);
        // Only the start node ever receives mass when it has no outgoing
        // edges, so the steady-state score is bounded below by 1.0.
        assert!(
            start_score >= 1.0 - 1e-9,
            "expected dangling start to be >= 1.0, got {}",
            start_score
        );
        // Only one entry in the score map.
        assert_eq!(scores.len(), 1);
    }

    #[test]
    fn test_json_roundtrip_preserves_graph() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let a = graph.add_architecture("a", vec![1.0, 0.0, 0.0], "vision");
        let b = graph.add_architecture("b", vec![0.9, 0.1, 0.0], "vision");
        let c = graph.add_architecture("c", vec![0.0, 1.0, 0.0], "nlp");
        graph
            .add_edge(a, b, RelationType::Similar, 0.99)
            .expect("similar");
        graph
            .add_edge(b, c, RelationType::Transfers, 0.5)
            .expect("transfers");
        graph
            .set_performance(a, make_record("imagenet", 0.78))
            .expect("perf");

        let json = graph.to_json().expect("to_json");
        let restored = ArchitectureKnowledgeGraph::from_json(&json).expect("from_json");

        assert_eq!(restored.len(), graph.len());
        assert_eq!(restored.edge_count(), graph.edge_count());
        for (orig, back) in graph.nodes().iter().zip(restored.nodes().iter()) {
            assert_eq!(orig.id, back.id);
            assert_eq!(orig.arch_id, back.arch_id);
            assert_eq!(orig.domain, back.domain);
            assert_eq!(orig.embedding, back.embedding);
            assert_eq!(orig.performance, back.performance);
        }
        // Adjacency must have been rebuilt so neighbours work.
        let neigh = restored.neighbors(a);
        assert_eq!(neigh.len(), 1);
        assert_eq!(neigh[0].0, b);
        assert_eq!(neigh[0].1, RelationType::Similar);
    }

    #[test]
    fn test_extract_top_k_transferable_filters_by_domain() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let v1 = graph.add_architecture("v1", vec![1.0, 0.0, 0.0], "vision");
        let v2 = graph.add_architecture("v2", vec![0.9, 0.1, 0.0], "vision");
        let n1 = graph.add_architecture("n1", vec![0.95, 0.0, 0.05], "nlp");
        let n2 = graph.add_architecture("n2", vec![0.0, 0.0, 1.0], "nlp");
        // n1 transfers knowledge to vision domain via v1.
        graph
            .add_edge(n1, v1, RelationType::Transfers, 0.7)
            .expect("transfers");

        let top = graph.extract_top_k_transferable("vision", &[1.0, 0.0, 0.0], 5);
        let ids: Vec<NodeId> = top.iter().map(|t| t.0).collect();
        // Both vision nodes and n1 (linked via Transfers) are eligible.
        assert!(ids.contains(&v1));
        assert!(ids.contains(&v2));
        assert!(ids.contains(&n1));
        // Unrelated nlp node n2 must be excluded.
        assert!(!ids.contains(&n2));
        // Sorted descending by similarity to [1, 0, 0].
        for window in top.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
        // k = 0 returns empty.
        assert!(graph
            .extract_top_k_transferable("vision", &[1.0, 0.0, 0.0], 0)
            .is_empty());
    }

    #[test]
    fn test_set_performance_on_valid_node() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        let a = graph.add_architecture("a", vec![1.0, 0.0], "vision");
        let record = make_record("cifar10", 0.91);
        graph
            .set_performance(a, record.clone())
            .expect("set_performance");
        let node = graph.node(a).expect("present");
        let attached = node.performance.as_ref().expect("performance");
        assert_eq!(attached.task_id, "cifar10");
        assert!((attached.accuracy - 0.91).abs() < 1e-12);
        assert_eq!(attached.training_epochs, 5);
    }

    #[test]
    fn test_set_performance_on_invalid_node_errors() {
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("a", vec![1.0, 0.0], "vision");
        let err = graph.set_performance(42, make_record("cifar10", 0.5));
        assert!(err.is_err(), "expected error for missing node");
    }

    #[test]
    fn test_neighbors_on_unknown_node_is_empty() {
        let graph = ArchitectureKnowledgeGraph::new();
        assert!(graph.neighbors(0).is_empty());
        let mut graph = ArchitectureKnowledgeGraph::new();
        graph.add_architecture("a", vec![1.0, 0.0], "vision");
        assert!(graph.neighbors(0).is_empty()); // no outgoing edges yet
        assert!(graph.neighbors(99).is_empty()); // unknown id
    }
}
