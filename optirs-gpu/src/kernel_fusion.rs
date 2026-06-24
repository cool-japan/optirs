//! # Elementwise Kernel-Fusion Planner
//!
//! A CPU-side, compiler-style planner that operates over an operation DAG and
//! decides which (primarily elementwise) operations can be fused into a single
//! GPU kernel. No GPU execution happens here: the planner reasons purely about
//! the graph topology, fusion legality, and the memory-bandwidth implications
//! of fusion via a bytes-moved cost model.
//!
//! ## Operation graph
//!
//! The graph is a DAG of [`FusionOp`]s. Each op has a stable `id` (equal to its
//! insertion index), an [`OpKind`], a list of `inputs` (the op ids of its
//! producers), and an output tensor described by `output_shape` and
//! `dtype_bytes`. The byte size of an op's output tensor is
//! `product(output_shape) * dtype_bytes` (an empty shape denotes a scalar, i.e.
//! `product == 1`).
//!
//! An op with **no inputs** is a *source* (leaf): it represents data that is
//! already resident in memory (a parameter, gradient, or the result of an
//! upstream subgraph). [`FusionGraph::validate`] guarantees the graph is
//! acyclic and that every input references a real op.
//!
//! ## Fusion legality
//!
//! Two or more ops may be fused into one kernel iff:
//! 1. every op in the group is *fusible* (an elementwise [`OpKind`], see
//!    [`OpKind::is_fusible`]); barrier ops (`MatMul`, `Reduce`, `Transpose`)
//!    can never join an elementwise group and always break chains;
//! 2. the members form a connected producer -> consumer chain in the DAG (a
//!    fusible edge connects a fusible producer to a fusible consumer);
//! 3. the producer's output shape is broadcast-compatible with the consumer's
//!    output shape (NumPy trailing-dimension rule, optional via
//!    [`FusionPlanner::with_broadcast`]); and
//! 4. fusing must not create a cycle in the *group-contracted* dependency
//!    graph. Greedily merging fusible edges can otherwise sandwich a barrier
//!    group between two halves of an elementwise group, which would require the
//!    fused kernel to run both before and after the barrier. Such merges are
//!    rejected so the inter-group schedule stays a DAG.
//!
//! When an intermediate that is internal to a group is *also* consumed by an op
//! **outside** the group, the intermediate cannot be elided: it is marked as a
//! group output and materialized to memory (never illegally dropped).
//!
//! ## Group formation
//!
//! Groups are formed by traversing ops in topological order and greedily
//! merging fusible producer -> consumer edges, subject to the legality checks
//! above (barriers stay as singletons; cycle-creating merges are skipped). The
//! result is a partition where every op belongs to exactly one group
//! (singletons allowed) and the contracted group graph is acyclic.
//!
//! ## Memory-bandwidth cost model
//!
//! - **Unfused** bytes moved: for every op, read each of its *distinct* input
//!   tensors once and write its output tensor once; summed over all ops.
//! - **Fused** bytes moved: for every group, read the group's *external* input
//!   tensors once (distinct producers outside the group) and write the group's
//!   *external* output tensors (members consumed outside the group or that are
//!   graph terminals). Intermediates that stay inside the group live in
//!   registers and are not counted.
//!
//! Because a singleton group reproduces exactly the unfused contribution of its
//! single op, fusion can only ever remove traffic: `bytes_fused <=
//! bytes_unfused` and `speedup_estimate = bytes_unfused / bytes_fused >= 1.0`
//! (a bandwidth-bound proxy).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::GpuOptimError;

/// The kind of an operation in the fusion graph.
///
/// The first nine variants are *elementwise* and therefore fusible; the final
/// three are *barrier* ops that change the iteration space or reduce/permute
/// data and so cannot participate in an elementwise fusion group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Elementwise addition.
    Add,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise division.
    Div,
    /// Rectified linear unit activation.
    Relu,
    /// Logistic sigmoid activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    /// Scalar scaling (`alpha * x`).
    Scale,
    /// Fused multiply-add (`alpha * x + y`).
    Axpy,
    /// Matrix multiplication (barrier: not fusible into an elementwise group).
    MatMul,
    /// Reduction such as sum/mean/max (barrier).
    Reduce,
    /// Transpose / permutation (barrier).
    Transpose,
}

impl OpKind {
    /// Returns `true` if the op is an elementwise op that may be fused into a
    /// single kernel together with other fusible ops.
    pub fn is_fusible(&self) -> bool {
        matches!(
            self,
            OpKind::Add
                | OpKind::Mul
                | OpKind::Sub
                | OpKind::Div
                | OpKind::Relu
                | OpKind::Sigmoid
                | OpKind::Tanh
                | OpKind::Scale
                | OpKind::Axpy
        )
    }

    /// Returns `true` for barrier ops (`MatMul`, `Reduce`, `Transpose`) that
    /// break fusion chains and always form singleton groups.
    pub fn is_barrier(&self) -> bool {
        !self.is_fusible()
    }
}

/// A single operation in the fusion graph.
///
/// `id` equals the op's insertion index in its [`FusionGraph`]. `inputs` lists
/// the op ids of the producers whose output tensors this op consumes; an empty
/// `inputs` list marks a source (leaf) op.
#[derive(Debug, Clone)]
pub struct FusionOp {
    /// Stable identifier (equal to the insertion index in the graph).
    pub id: usize,
    /// The kind of operation.
    pub kind: OpKind,
    /// Op ids of the producers this op reads from.
    pub inputs: Vec<usize>,
    /// Shape of this op's output tensor (empty == scalar).
    pub output_shape: Vec<usize>,
    /// Size in bytes of a single output element.
    pub dtype_bytes: usize,
}

impl FusionOp {
    /// Number of elements in the output tensor (product of the shape dims).
    ///
    /// Returns an error if the product overflows a `u64`.
    pub fn output_elements(&self) -> Result<u64, GpuOptimError> {
        let mut elements: u64 = 1;
        for &dim in &self.output_shape {
            elements = elements.checked_mul(dim as u64).ok_or_else(|| {
                GpuOptimError::UnsupportedOperation(format!(
                    "op {} output shape {:?} overflows the element counter",
                    self.id, self.output_shape
                ))
            })?;
        }
        Ok(elements)
    }

    /// Size in bytes of the output tensor (`product(shape) * dtype_bytes`).
    ///
    /// Returns an error if the computation overflows a `u64`.
    pub fn output_bytes(&self) -> Result<u64, GpuOptimError> {
        let elements = self.output_elements()?;
        elements
            .checked_mul(self.dtype_bytes as u64)
            .ok_or_else(|| {
                GpuOptimError::UnsupportedOperation(format!(
                    "op {} output ({} elements x {} bytes) overflows the byte counter",
                    self.id, elements, self.dtype_bytes
                ))
            })
    }
}

/// An append-only builder for an operation DAG.
#[derive(Debug, Default, Clone)]
pub struct FusionGraph {
    ops: Vec<FusionOp>,
}

impl FusionGraph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Appends an op and returns its freshly assigned id.
    ///
    /// `inputs` must reference the ids of previously added ops (an empty list
    /// denotes a source/leaf). Validity and acyclicity are checked by
    /// [`FusionGraph::validate`], not here, so a forward reference can be built
    /// and later rejected.
    pub fn add_op(
        &mut self,
        kind: OpKind,
        inputs: Vec<usize>,
        output_shape: Vec<usize>,
        dtype_bytes: usize,
    ) -> usize {
        let id = self.ops.len();
        self.ops.push(FusionOp {
            id,
            kind,
            inputs,
            output_shape,
            dtype_bytes,
        });
        id
    }

    /// Returns the ops in insertion order.
    pub fn ops(&self) -> &[FusionOp] {
        &self.ops
    }

    /// Number of ops in the graph.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if the graph has no ops.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Validates the graph: every input must reference an existing op, no op may
    /// reference itself, `dtype_bytes` must be non-zero, byte counts must not
    /// overflow, and the graph must be acyclic.
    ///
    /// Returns [`GpuOptimError::InvalidState`] for a dangling input, a
    /// self-reference, or a cycle.
    pub fn validate(&self) -> Result<(), GpuOptimError> {
        let n = self.ops.len();
        for op in &self.ops {
            if op.dtype_bytes == 0 {
                return Err(GpuOptimError::InvalidState(format!(
                    "op {} has dtype_bytes == 0",
                    op.id
                )));
            }
            for &producer in &op.inputs {
                if producer >= n {
                    return Err(GpuOptimError::InvalidState(format!(
                        "op {} references non-existent input op {}",
                        op.id, producer
                    )));
                }
                if producer == op.id {
                    return Err(GpuOptimError::InvalidState(format!(
                        "op {} references itself as an input",
                        op.id
                    )));
                }
            }
            // Surface overflow eagerly so the planner never has to.
            op.output_bytes()?;
        }
        self.topological_order()?;
        Ok(())
    }

    /// Computes a topological order via Kahn's algorithm.
    ///
    /// Returns [`GpuOptimError::InvalidState`] if a dangling input is found or
    /// the graph contains a cycle.
    fn topological_order(&self) -> Result<Vec<usize>, GpuOptimError> {
        let n = self.ops.len();
        let mut indegree = vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for consumer in &self.ops {
            let mut seen: HashSet<usize> = HashSet::new();
            for &producer in &consumer.inputs {
                if producer >= n {
                    return Err(GpuOptimError::InvalidState(format!(
                        "op {} references non-existent input op {}",
                        consumer.id, producer
                    )));
                }
                // Collapse duplicate edges so indegree counts distinct producers.
                if !seen.insert(producer) {
                    continue;
                }
                adjacency[producer].push(consumer.id);
                indegree[consumer.id] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &consumer in &adjacency[node] {
                indegree[consumer] -= 1;
                if indegree[consumer] == 0 {
                    queue.push_back(consumer);
                }
            }
        }

        if order.len() != n {
            return Err(GpuOptimError::InvalidState(
                "operation graph contains a cycle".to_string(),
            ));
        }
        Ok(order)
    }
}

/// A group of ops fused into a single kernel.
///
/// `members` is the set of op ids in the group (sorted ascending).
/// `external_inputs` are the producer op ids outside the group that the group
/// reads. `external_outputs` are member ids whose results must be materialized
/// (consumed outside the group, or graph terminals). `internal_intermediates`
/// are members whose results stay in registers and are elided.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Op ids fused into this kernel (sorted ascending).
    pub members: Vec<usize>,
    /// Producer op ids outside the group that the group reads (sorted).
    pub external_inputs: Vec<usize>,
    /// Member op ids whose outputs are written to memory (sorted).
    pub external_outputs: Vec<usize>,
    /// Member op ids whose outputs are fully elided into registers (sorted).
    pub internal_intermediates: Vec<usize>,
    /// Bytes read by this group (external inputs, each counted once).
    pub bytes_read: u64,
    /// Bytes written by this group (external outputs).
    pub bytes_written: u64,
}

impl FusionGroup {
    /// Total fused bytes moved by this group (`bytes_read + bytes_written`).
    pub fn bytes_fused(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }

    /// Returns `true` if the group fuses more than one op.
    pub fn is_fused(&self) -> bool {
        self.members.len() > 1
    }
}

/// The output of [`FusionPlanner::plan`]: the fusion groups plus a
/// memory-bandwidth cost summary.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Fusion groups, ordered by their smallest member id.
    pub groups: Vec<FusionGroup>,
    /// Total bytes moved without fusion.
    pub bytes_unfused: u64,
    /// Total bytes moved with fusion.
    pub bytes_fused: u64,
    /// `bytes_unfused - bytes_fused`.
    pub bytes_saved: u64,
    /// `bytes_unfused / bytes_fused` (1.0 when no traffic).
    pub speedup_estimate: f64,
}

impl FusionPlan {
    /// Number of groups in the plan.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
}

/// Plans elementwise kernel fusion over an operation DAG.
#[derive(Debug, Clone)]
pub struct FusionPlanner {
    allow_broadcast: bool,
}

impl Default for FusionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionPlanner {
    /// Creates a planner that allows NumPy-style broadcast on fusible edges.
    pub fn new() -> Self {
        Self {
            allow_broadcast: true,
        }
    }

    /// Enables or disables broadcast compatibility on fusible edges. When
    /// disabled, a producer -> consumer edge only fuses if the shapes are equal.
    pub fn with_broadcast(mut self, allow_broadcast: bool) -> Self {
        self.allow_broadcast = allow_broadcast;
        self
    }

    /// Returns `true` if `producer` can broadcast into `consumer` for the
    /// purpose of fusing into one elementwise kernel.
    ///
    /// Equal shapes are always compatible. With broadcast enabled, the NumPy
    /// trailing-dimension rule applies: the producer's shape is right-aligned
    /// with the consumer's, must not be longer, and each aligned producer
    /// dimension must equal the consumer's or be `1` (a scalar producer
    /// broadcasts to anything).
    fn shapes_fuse_compatible(&self, producer: &[usize], consumer: &[usize]) -> bool {
        if producer == consumer {
            return true;
        }
        if !self.allow_broadcast {
            return false;
        }
        if producer.len() > consumer.len() {
            return false;
        }
        let offset = consumer.len() - producer.len();
        for (i, &producer_dim) in producer.iter().enumerate() {
            let consumer_dim = consumer[offset + i];
            if producer_dim != consumer_dim && producer_dim != 1 {
                return false;
            }
        }
        true
    }

    /// Plans fusion for `graph` and returns the groups plus the cost summary.
    ///
    /// Returns an error if the graph fails [`FusionGraph::validate`] or if any
    /// tensor byte count overflows.
    pub fn plan(&self, graph: &FusionGraph) -> Result<FusionPlan, GpuOptimError> {
        graph.validate()?;
        let ops = graph.ops();
        let n = ops.len();

        // Pre-compute the byte size of every op's output tensor.
        let mut bytes: Vec<u64> = Vec::with_capacity(n);
        for op in ops {
            bytes.push(op.output_bytes()?);
        }

        let topo = graph.topological_order()?;

        // consumers[p] = distinct ops that read op p's output.
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for consumer in ops {
            let mut seen: HashSet<usize> = HashSet::new();
            for &producer in &consumer.inputs {
                if seen.insert(producer) {
                    consumers[producer].push(consumer.id);
                }
            }
        }

        // Greedy union over legal fusible edges. `group_of[i]` is i's group label.
        let mut group_of: Vec<usize> = (0..n).collect();
        for &consumer_id in &topo {
            let consumer = &ops[consumer_id];
            if !consumer.kind.is_fusible() {
                continue;
            }
            let mut seen: HashSet<usize> = HashSet::new();
            for &producer_id in &consumer.inputs {
                if !seen.insert(producer_id) {
                    continue;
                }
                let producer = &ops[producer_id];
                if !producer.kind.is_fusible() {
                    continue;
                }
                if !self.shapes_fuse_compatible(&producer.output_shape, &consumer.output_shape) {
                    continue;
                }
                let group_producer = group_of[producer_id];
                let group_consumer = group_of[consumer_id];
                if group_producer == group_consumer {
                    continue;
                }
                // Only merge if the contracted group graph stays acyclic.
                if merge_keeps_acyclic(ops, &group_of, group_producer, group_consumer) {
                    for label in group_of.iter_mut() {
                        if *label == group_consumer {
                            *label = group_producer;
                        }
                    }
                }
            }
        }

        // Bucket op ids by group label (members collected in topological order).
        let mut label_to_members: HashMap<usize, Vec<usize>> = HashMap::new();
        for &id in &topo {
            label_to_members.entry(group_of[id]).or_default().push(id);
        }
        let mut raw_groups: Vec<Vec<usize>> = label_to_members.into_values().collect();
        for members in raw_groups.iter_mut() {
            members.sort_unstable();
        }
        raw_groups.sort_by_key(|members| members[0]);

        // Materialize each group's external interface and per-group bytes.
        let mut groups: Vec<FusionGroup> = Vec::with_capacity(raw_groups.len());
        let mut bytes_fused: u64 = 0;
        for members in raw_groups {
            let member_set: HashSet<usize> = members.iter().copied().collect();

            let mut external_inputs: Vec<usize> = Vec::new();
            let mut external_input_seen: HashSet<usize> = HashSet::new();
            for &member in &members {
                let mut seen: HashSet<usize> = HashSet::new();
                for &producer in &ops[member].inputs {
                    if !seen.insert(producer) {
                        continue;
                    }
                    if !member_set.contains(&producer) && external_input_seen.insert(producer) {
                        external_inputs.push(producer);
                    }
                }
            }
            external_inputs.sort_unstable();

            let mut external_outputs: Vec<usize> = Vec::new();
            let mut internal_intermediates: Vec<usize> = Vec::new();
            for &member in &members {
                let consumed_externally = consumers[member].iter().any(|c| !member_set.contains(c));
                let is_terminal = consumers[member].is_empty();
                if consumed_externally || is_terminal {
                    external_outputs.push(member);
                } else {
                    internal_intermediates.push(member);
                }
            }

            let bytes_read: u64 = external_inputs.iter().map(|&p| bytes[p]).sum();
            let bytes_written: u64 = external_outputs.iter().map(|&m| bytes[m]).sum();
            bytes_fused += bytes_read + bytes_written;

            groups.push(FusionGroup {
                members,
                external_inputs,
                external_outputs,
                internal_intermediates,
                bytes_read,
                bytes_written,
            });
        }

        // Unfused traffic: every op reads each distinct input and writes once.
        let mut bytes_unfused: u64 = 0;
        for op in ops {
            let mut seen: HashSet<usize> = HashSet::new();
            let mut read: u64 = 0;
            for &producer in &op.inputs {
                if seen.insert(producer) {
                    read += bytes[producer];
                }
            }
            bytes_unfused += read + bytes[op.id];
        }

        let bytes_saved = bytes_unfused.saturating_sub(bytes_fused);
        let speedup_estimate = if bytes_fused == 0 {
            1.0
        } else {
            bytes_unfused as f64 / bytes_fused as f64
        };

        Ok(FusionPlan {
            groups,
            bytes_unfused,
            bytes_fused,
            bytes_saved,
            speedup_estimate,
        })
    }
}

/// Returns `true` if merging groups `group_a` and `group_b` keeps the
/// group-contracted dependency graph acyclic.
///
/// The two groups are treated as a single contracted node; an edge is added
/// between the contracted labels of every producer -> consumer pair whose
/// endpoints land in different groups. A Kahn pass then checks for a cycle.
fn merge_keeps_acyclic(
    ops: &[FusionOp],
    group_of: &[usize],
    group_a: usize,
    group_b: usize,
) -> bool {
    let label = |op_id: usize| -> usize {
        let group = group_of[op_id];
        if group == group_b {
            group_a
        } else {
            group
        }
    };

    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut nodes: HashSet<usize> = HashSet::new();
    for consumer in ops {
        let consumer_label = label(consumer.id);
        nodes.insert(consumer_label);
        for &producer in &consumer.inputs {
            let producer_label = label(producer);
            nodes.insert(producer_label);
            if producer_label != consumer_label {
                adjacency
                    .entry(producer_label)
                    .or_default()
                    .insert(consumer_label);
            }
        }
    }

    let mut indegree: HashMap<usize, usize> = nodes.iter().map(|&node| (node, 0usize)).collect();
    for targets in adjacency.values() {
        for &target in targets {
            if let Some(degree) = indegree.get_mut(&target) {
                *degree += 1;
            }
        }
    }

    let mut queue: VecDeque<usize> = indegree
        .iter()
        .filter_map(|(&node, &degree)| if degree == 0 { Some(node) } else { None })
        .collect();
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(targets) = adjacency.get(&node) {
            for &target in targets {
                if let Some(degree) = indegree.get_mut(&target) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target);
                    }
                }
            }
        }
    }

    visited == nodes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_group(plan: &FusionPlan, op_id: usize) -> &FusionGroup {
        plan.groups
            .iter()
            .find(|g| g.members.contains(&op_id))
            .expect("every op must belong to exactly one group")
    }

    #[test]
    fn is_fusible_classification() {
        assert!(OpKind::Add.is_fusible());
        assert!(OpKind::Mul.is_fusible());
        assert!(OpKind::Axpy.is_fusible());
        assert!(OpKind::Scale.is_fusible());
        assert!(!OpKind::MatMul.is_fusible());
        assert!(!OpKind::Reduce.is_fusible());
        assert!(!OpKind::Transpose.is_fusible());
        assert!(OpKind::MatMul.is_barrier());
        assert!(!OpKind::Relu.is_barrier());
    }

    #[test]
    fn linear_chain_fuses_into_one_group() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Relu, vec![], vec![256], 4);
        let b = graph.add_op(OpKind::Sigmoid, vec![a], vec![256], 4);
        let c = graph.add_op(OpKind::Tanh, vec![b], vec![256], 4);

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("fusible chain must plan");

        assert_eq!(plan.groups.len(), 1);
        let group = &plan.groups[0];
        assert_eq!(group.members, vec![a, b, c]);
        assert!(group.external_inputs.is_empty());
        assert_eq!(group.external_outputs, vec![c]);
        assert_eq!(group.internal_intermediates, vec![a, b]);

        let tensor = 256u64 * 4;
        assert_eq!(plan.bytes_unfused, 5 * tensor);
        assert_eq!(plan.bytes_fused, tensor);
        assert!(plan.bytes_fused < plan.bytes_unfused);
        assert_eq!(plan.bytes_saved, 4 * tensor);
        assert!((plan.speedup_estimate - 5.0).abs() < 1e-9);
    }

    #[test]
    fn barrier_splits_into_three_groups() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Relu, vec![], vec![128], 4);
        let b = graph.add_op(OpKind::Sigmoid, vec![a], vec![128], 4);
        let c = graph.add_op(OpKind::MatMul, vec![b], vec![128], 4); // barrier
        let d = graph.add_op(OpKind::Relu, vec![c], vec![128], 4);
        let e = graph.add_op(OpKind::Tanh, vec![d], vec![128], 4);

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("graph with a barrier must plan");

        assert_eq!(plan.groups.len(), 3);
        assert_eq!(find_group(&plan, a).members, vec![a, b]);
        assert_eq!(find_group(&plan, c).members, vec![c]);
        assert_eq!(find_group(&plan, d).members, vec![d, e]);
    }

    #[test]
    fn external_consumer_materializes_intermediate() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Relu, vec![], vec![256], 4);
        let b = graph.add_op(OpKind::Sigmoid, vec![a], vec![256], 4);
        let c = graph.add_op(OpKind::Tanh, vec![b], vec![256], 4);
        // External (barrier) consumer of `b` forces `b` to be materialized.
        let d = graph.add_op(OpKind::MatMul, vec![b], vec![256], 4);

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("diamond graph must plan");

        assert_eq!(plan.groups.len(), 2);
        let group = find_group(&plan, b);
        assert_eq!(group.members, vec![a, b, c]);
        assert!(
            group.external_outputs.contains(&b),
            "b is consumed outside the group and must be materialized"
        );
        assert!(!group.internal_intermediates.contains(&b));
        assert!(group.external_outputs.contains(&c));
        assert_eq!(group.internal_intermediates, vec![a]);
        assert_eq!(find_group(&plan, d).members, vec![d]);

        // Hand-computed bytes: tensor = 256 * 4 = 1024.
        let tensor = 256u64 * 4;
        assert_eq!(plan.bytes_unfused, 7 * tensor);
        assert_eq!(plan.bytes_fused, 4 * tensor);
        assert_eq!(plan.bytes_saved, 3 * tensor);
        assert!((plan.speedup_estimate - 1.75).abs() < 1e-9);
    }

    #[test]
    fn shared_external_input_counted_once() {
        let mut graph = FusionGraph::new();
        // Barrier source acts as a shared external input tensor (16 * 4 = 64 bytes).
        let x = graph.add_op(OpKind::MatMul, vec![], vec![16], 4);
        let r = graph.add_op(OpKind::Relu, vec![x], vec![16], 4);
        let s = graph.add_op(OpKind::Add, vec![r, x], vec![16], 4);

        let plan = FusionPlanner::new().plan(&graph).expect("graph must plan");

        assert_eq!(plan.groups.len(), 2);
        let group = find_group(&plan, r);
        assert_eq!(group.members, vec![r, s]);
        // x is read by both r and s but appears exactly once as an external input.
        assert_eq!(group.external_inputs, vec![x]);

        let tensor = 16u64 * 4; // 64
                                // Unfused: x:64, r:64+64, s:(64+64)+64 = 64 + 128 + 192 = 384 = 6 * 64.
        assert_eq!(plan.bytes_unfused, 6 * tensor);
        // Fused: {x} writes 64; {r,s} reads x once (64) + writes s (64) = 128.
        assert_eq!(plan.bytes_fused, 3 * tensor);
        assert_eq!(plan.bytes_saved, 3 * tensor);
        assert!((plan.speedup_estimate - 2.0).abs() < 1e-9);
    }

    #[test]
    fn hand_computed_bytes_exact() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Mul, vec![], vec![10], 4); // 40 bytes
        let b = graph.add_op(OpKind::Add, vec![a], vec![10], 4);

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("two-op chain must plan");

        assert_eq!(plan.groups.len(), 1);
        let group = &plan.groups[0];
        assert!(group.external_inputs.is_empty());
        assert_eq!(group.external_outputs, vec![b]);
        assert_eq!(group.internal_intermediates, vec![a]);
        assert_eq!(group.bytes_read, 0);
        assert_eq!(group.bytes_written, 40);

        // Unfused: a writes 40; b reads 40 + writes 40 => 120.
        assert_eq!(plan.bytes_unfused, 120);
        // Fused: write b only (a stays in registers) => 40.
        assert_eq!(plan.bytes_fused, 40);
        assert_eq!(plan.bytes_saved, 80);
        assert!((plan.speedup_estimate - 3.0).abs() < 1e-9);
    }

    #[test]
    fn fusion_avoids_introducing_cycle() {
        // a -> b -> c(barrier) -> d, plus a -> d. Fusing a with d would sandwich
        // the barrier group c and create a cycle, so d must stay separate.
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Relu, vec![], vec![16], 4);
        let b = graph.add_op(OpKind::Sigmoid, vec![a], vec![16], 4);
        let c = graph.add_op(OpKind::MatMul, vec![b], vec![16], 4); // barrier
        let d = graph.add_op(OpKind::Add, vec![a, c], vec![16], 4);

        let plan = FusionPlanner::new().plan(&graph).expect("graph must plan");

        assert_eq!(plan.groups.len(), 3);
        assert_eq!(find_group(&plan, a).members, vec![a, b]);
        assert_eq!(find_group(&plan, c).members, vec![c]);
        assert_eq!(find_group(&plan, d).members, vec![d]);
        // a feeds b (internal) and d (external) so it is materialized.
        assert!(find_group(&plan, a).external_outputs.contains(&a));
    }

    #[test]
    fn broadcast_compatible_edge_fuses() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Relu, vec![], vec![1], 4); // scalar-ish
        let b = graph.add_op(OpKind::Add, vec![a], vec![32], 4); // broadcasts [1] -> [32]

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("broadcast chain must plan");
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].members, vec![a, b]);

        // With broadcast disabled the mismatched shapes do not fuse.
        let strict = FusionPlanner::new()
            .with_broadcast(false)
            .plan(&graph)
            .expect("strict planner must plan");
        assert_eq!(strict.groups.len(), 2);
    }

    #[test]
    fn bytes_saved_and_speedup_invariants() {
        let mut graph = FusionGraph::new();
        let a = graph.add_op(OpKind::Scale, vec![], vec![64, 64], 4);
        let b = graph.add_op(OpKind::Relu, vec![a], vec![64, 64], 4);
        let c = graph.add_op(OpKind::Sigmoid, vec![b], vec![64, 64], 4);
        let d = graph.add_op(OpKind::Tanh, vec![c], vec![64, 64], 4);

        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("fusible chain must plan");

        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.bytes_saved, plan.bytes_unfused - plan.bytes_fused);
        assert!(plan.speedup_estimate >= 1.0);
        assert!(plan.bytes_fused < plan.bytes_unfused);
    }

    #[test]
    fn cyclic_graph_rejected() {
        let mut graph = FusionGraph::new();
        let _a = graph.add_op(OpKind::Add, vec![1], vec![8], 4); // forward ref to op 1
        let _b = graph.add_op(OpKind::Add, vec![0], vec![8], 4); // back ref to op 0 => cycle

        let error = graph
            .validate()
            .expect_err("a cyclic graph must be rejected");
        assert!(matches!(error, GpuOptimError::InvalidState(_)));
        assert!(FusionPlanner::new().plan(&graph).is_err());
    }

    #[test]
    fn dangling_input_rejected() {
        let mut graph = FusionGraph::new();
        let _a = graph.add_op(OpKind::Relu, vec![99], vec![8], 4); // op 99 does not exist

        let error = graph
            .validate()
            .expect_err("a dangling input must be rejected");
        assert!(matches!(error, GpuOptimError::InvalidState(_)));
    }

    #[test]
    fn zero_dtype_rejected() {
        let mut graph = FusionGraph::new();
        let _a = graph.add_op(OpKind::Relu, vec![], vec![8], 0);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn empty_graph_is_valid() {
        let graph = FusionGraph::new();
        assert!(graph.is_empty());
        let plan = FusionPlanner::new()
            .plan(&graph)
            .expect("empty graph must plan");
        assert_eq!(plan.num_groups(), 0);
        assert_eq!(plan.bytes_unfused, 0);
        assert_eq!(plan.bytes_fused, 0);
        assert_eq!(plan.bytes_saved, 0);
        assert!((plan.speedup_estimate - 1.0).abs() < 1e-9);
    }
}
