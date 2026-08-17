// Deadlock Graph Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ChangeType {
    #[default]
    Added,
    Removed,
    Modified,
}

#[derive(Debug, Clone)]
pub struct DependencyEdge {
    pub from: u64,
    pub to: u64,
    pub source: u64,
    pub target: u64,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub metadata: EdgeMetadata,
    pub timestamp: std::time::Instant,
}

impl Default for DependencyEdge {
    fn default() -> Self {
        Self {
            from: 0,
            to: 0,
            source: 0,
            target: 0,
            edge_type: EdgeType::default(),
            weight: 0.0,
            metadata: EdgeMetadata::default(),
            timestamp: std::time::Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DependencyGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<DependencyEdge>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: DependencyEdge) {
        self.edges.push(edge);
    }

    /// Adjacency list keyed by source node id.
    ///
    /// Edges carry both `from`/`to` and `source`/`target`; `source`/`target`
    /// are the fields populated by [`super::DeadlockDetector::add_dependency`],
    /// with `from`/`to` used as a fallback for edges built directly.
    fn adjacency(&self) -> HashMap<u64, Vec<u64>> {
        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        for edge in &self.edges {
            let (source, target) = if edge.source != 0 || edge.target != 0 {
                (edge.source, edge.target)
            } else {
                (edge.from, edge.to)
            };
            adjacency.entry(source).or_default().push(target);
        }
        adjacency
    }

    /// Every node id referenced by the graph, in a deterministic order.
    fn all_nodes(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.nodes.iter().map(|node| node.id).collect();
        for edge in &self.edges {
            for id in [edge.source, edge.target, edge.from, edge.to] {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Check if the graph has a cycle, using depth-first search.
    ///
    /// A wait-for graph containing a cycle *is* a deadlock, so this returning a
    /// constant `false` (the previous behaviour) meant the detector could never
    /// fire.
    pub fn has_cycle(&self) -> bool {
        self.find_cycle().is_some()
    }

    /// Check for a cycle using the requested detection method.
    ///
    /// All three methods are exact cycle oracles and therefore agree; they
    /// differ only in traversal strategy and in the auxiliary state they build.
    pub fn has_cycle_with(&self, method: &super::algorithms::CycleDetectionMethod) -> bool {
        use super::algorithms::CycleDetectionMethod;
        match method {
            CycleDetectionMethod::DFS => self.find_cycle().is_some(),
            CycleDetectionMethod::BFS => self.has_cycle_kahn(),
            CycleDetectionMethod::Tarjan => self.has_nontrivial_scc(),
        }
    }

    /// Locate a cycle using the requested detection method.
    ///
    /// `BFS` and `Tarjan` decide *whether* a cycle exists; when they report one
    /// the concrete member list is recovered with the depth-first search, so
    /// callers always get the participating nodes.
    pub fn find_cycle_with(
        &self,
        method: &super::algorithms::CycleDetectionMethod,
    ) -> Option<Vec<u64>> {
        if !self.has_cycle_with(method) {
            return None;
        }
        self.find_cycle()
    }

    /// Return the nodes forming a cycle, if one exists.
    ///
    /// Iterative depth-first search with three-colour marking: a node absent
    /// from `visited` is white, a node in `on_stack` is grey, and a node that
    /// has been popped is black. An edge into a grey node is a back edge, which
    /// closes a cycle. The traversal uses an explicit stack so that deep
    /// dependency chains cannot overflow the native stack.
    pub fn find_cycle(&self) -> Option<Vec<u64>> {
        enum Step {
            Enter(u64),
            Leave(u64),
        }

        let adjacency = self.adjacency();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut on_stack: HashSet<u64> = HashSet::new();
        let mut path: Vec<u64> = Vec::new();

        for start in self.all_nodes() {
            if visited.contains(&start) {
                continue;
            }

            let mut stack: Vec<Step> = vec![Step::Enter(start)];

            while let Some(step) = stack.pop() {
                match step {
                    Step::Leave(node) => {
                        on_stack.remove(&node);
                        if path.last() == Some(&node) {
                            path.pop();
                        }
                    }
                    Step::Enter(node) => {
                        if !visited.insert(node) {
                            continue;
                        }
                        on_stack.insert(node);
                        path.push(node);
                        stack.push(Step::Leave(node));

                        for &next in adjacency.get(&node).into_iter().flatten() {
                            if on_stack.contains(&next) {
                                // Back edge: report the cycle from `next` on.
                                let start_index = path.iter().position(|&n| n == next).unwrap_or(0);
                                return Some(path[start_index..].to_vec());
                            }
                            if !visited.contains(&next) {
                                stack.push(Step::Enter(next));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Cycle detection by Kahn's algorithm: a graph is acyclic exactly when a
    /// topological order covers every node.
    fn has_cycle_kahn(&self) -> bool {
        let adjacency = self.adjacency();
        let nodes = self.all_nodes();

        let mut in_degree: HashMap<u64, usize> = nodes.iter().map(|&id| (id, 0usize)).collect();
        for targets in adjacency.values() {
            for target in targets {
                *in_degree.entry(*target).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<u64> = nodes
            .iter()
            .copied()
            .filter(|id| in_degree.get(id) == Some(&0))
            .collect();

        let mut removed = 0usize;
        while let Some(node) = queue.pop_front() {
            removed += 1;
            for &next in adjacency.get(&node).into_iter().flatten() {
                if let Some(degree) = in_degree.get_mut(&next) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }

        removed != in_degree.len()
    }

    /// Cycle detection via strongly connected components (Tarjan).
    ///
    /// A directed graph has a cycle exactly when it has an SCC of more than one
    /// node, or a node with an edge to itself.
    fn has_nontrivial_scc(&self) -> bool {
        let adjacency = self.adjacency();

        // Self-loops are single-node cycles that SCC size alone would miss.
        for (&source, targets) in &adjacency {
            if targets.contains(&source) {
                return true;
            }
        }

        let nodes = self.all_nodes();
        let mut index_counter: usize = 0;
        let mut indices: HashMap<u64, usize> = HashMap::new();
        let mut low_links: HashMap<u64, usize> = HashMap::new();
        let mut on_stack: HashSet<u64> = HashSet::new();
        let mut scc_stack: Vec<u64> = Vec::new();

        // Iterative Tarjan: each frame tracks how many successors it consumed.
        struct Frame {
            node: u64,
            next_child: usize,
        }

        for &start in &nodes {
            if indices.contains_key(&start) {
                continue;
            }

            let mut frames: Vec<Frame> = vec![Frame {
                node: start,
                next_child: 0,
            }];
            indices.insert(start, index_counter);
            low_links.insert(start, index_counter);
            index_counter += 1;
            scc_stack.push(start);
            on_stack.insert(start);

            while let Some(frame) = frames.last_mut() {
                let node = frame.node;
                let children: &[u64] = adjacency.get(&node).map(|v| v.as_slice()).unwrap_or(&[]);

                if frame.next_child < children.len() {
                    let child = children[frame.next_child];
                    frame.next_child += 1;

                    if let std::collections::hash_map::Entry::Vacant(entry) = indices.entry(child) {
                        entry.insert(index_counter);
                        low_links.insert(child, index_counter);
                        index_counter += 1;
                        scc_stack.push(child);
                        on_stack.insert(child);
                        frames.push(Frame {
                            node: child,
                            next_child: 0,
                        });
                    } else if on_stack.contains(&child) {
                        let child_index = indices.get(&child).copied().unwrap_or(0);
                        let current = low_links.get(&node).copied().unwrap_or(0);
                        low_links.insert(node, current.min(child_index));
                    }
                    continue;
                }

                // All successors consumed: close this node out.
                frames.pop();

                if low_links.get(&node) == indices.get(&node) {
                    let mut size = 0usize;
                    while let Some(member) = scc_stack.pop() {
                        on_stack.remove(&member);
                        size += 1;
                        if member == node {
                            break;
                        }
                    }
                    if size > 1 {
                        return true;
                    }
                }

                if let Some(parent) = frames.last().map(|f| f.node) {
                    let child_low = low_links.get(&node).copied().unwrap_or(0);
                    let parent_low = low_links.get(&parent).copied().unwrap_or(0);
                    low_links.insert(parent, parent_low.min(child_low));
                }
            }
        }

        false
    }
}

#[derive(Debug, Clone, Default)]
pub struct EdgeMetadata {
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EdgeType {
    #[default]
    Dependency,
    Resource,
    Communication,
    WaitsFor,
}

#[derive(Debug, Clone, Default)]
pub struct GraphChange {
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Default)]
pub struct GraphHistory {
    pub snapshots: Vec<GraphSnapshot>,
}

#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: u64,
    pub node_type: NodeType,
    pub state: NodeState,
    pub metadata: GraphMetadata,
    pub timestamp: std::time::Instant,
}

impl Default for GraphNode {
    fn default() -> Self {
        Self {
            id: 0,
            node_type: NodeType::default(),
            state: NodeState::default(),
            metadata: GraphMetadata::default(),
            timestamp: std::time::Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GraphOptimizationState {
    pub optimized: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GraphProperties {
    pub is_cyclic: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GraphSnapshot {
    pub timestamp_ms: u64,
    pub graph: DependencyGraph,
}

#[derive(Debug, Clone, Default)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum NodeState {
    #[default]
    Active,
    Waiting,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum NodeType {
    #[default]
    Process,
    Resource,
    Lock,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OptimizationOperation {
    #[default]
    Merge,
    Split,
    Remove,
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationRecord {
    pub operation: OptimizationOperation,
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationStatistics {
    pub operations_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceImpact {
    pub improvement_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::super::algorithms::CycleDetectionMethod;
    use super::*;

    fn node(id: u64) -> GraphNode {
        GraphNode {
            id,
            ..Default::default()
        }
    }

    fn edge(source: u64, target: u64) -> DependencyEdge {
        DependencyEdge {
            from: source,
            to: target,
            source,
            target,
            edge_type: EdgeType::WaitsFor,
            ..Default::default()
        }
    }

    fn graph_from(nodes: &[u64], edges: &[(u64, u64)]) -> DependencyGraph {
        let mut graph = DependencyGraph::new();
        for &id in nodes {
            graph.add_node(node(id));
        }
        for &(source, target) in edges {
            graph.add_edge(edge(source, target));
        }
        graph
    }

    /// F7: a planted wait-for cycle A -> B -> C -> A must be detected.
    #[test]
    fn detects_planted_three_node_cycle() {
        let graph = graph_from(&[1, 2, 3], &[(1, 2), (2, 3), (3, 1)]);

        assert!(graph.has_cycle(), "A->B->C->A is a deadlock");

        let cycle = graph.find_cycle().expect("a cycle must be reported");
        assert_eq!(cycle.len(), 3);
        for id in [1u64, 2, 3] {
            assert!(cycle.contains(&id), "node {id} belongs to the cycle");
        }
    }

    /// An acyclic wait-for graph must not be reported as a deadlock.
    #[test]
    fn acyclic_graph_reports_no_cycle() {
        let graph = graph_from(&[1, 2, 3, 4], &[(1, 2), (2, 3), (1, 4), (4, 3)]);

        assert!(!graph.has_cycle());
        assert!(graph.find_cycle().is_none());
    }

    /// A node waiting on itself is a one-node deadlock.
    #[test]
    fn detects_self_loop() {
        let graph = graph_from(&[7], &[(7, 7)]);

        assert!(graph.has_cycle());
        for method in [
            CycleDetectionMethod::DFS,
            CycleDetectionMethod::BFS,
            CycleDetectionMethod::Tarjan,
        ] {
            assert!(
                graph.has_cycle_with(&method),
                "{method:?} must detect a self-loop"
            );
        }
    }

    /// All three configured methods must agree on every graph.
    #[test]
    fn all_detection_methods_agree() {
        let cases: Vec<(Vec<u64>, Vec<(u64, u64)>, bool)> = vec![
            (vec![1, 2, 3], vec![(1, 2), (2, 3), (3, 1)], true),
            (vec![1, 2, 3], vec![(1, 2), (2, 3)], false),
            (vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4), (4, 2)], true),
            (vec![1], vec![], false),
            (vec![], vec![], false),
            // Two disjoint components, only the second of which has a cycle.
            (
                vec![1, 2, 3, 4, 5],
                vec![(1, 2), (3, 4), (4, 5), (5, 3)],
                true,
            ),
        ];

        for (nodes, edges, expected) in cases {
            let graph = graph_from(&nodes, &edges);
            for method in [
                CycleDetectionMethod::DFS,
                CycleDetectionMethod::BFS,
                CycleDetectionMethod::Tarjan,
            ] {
                assert_eq!(
                    graph.has_cycle_with(&method),
                    expected,
                    "{method:?} disagreed on nodes={nodes:?} edges={edges:?}"
                );
            }
        }
    }

    /// Deep chains must not overflow the stack (the search is iterative).
    #[test]
    fn deep_chain_does_not_overflow() {
        let nodes: Vec<u64> = (0..20_000).collect();
        let edges: Vec<(u64, u64)> = (0..19_999).map(|i| (i, i + 1)).collect();
        let graph = graph_from(&nodes, &edges);

        assert!(!graph.has_cycle());
    }
}
