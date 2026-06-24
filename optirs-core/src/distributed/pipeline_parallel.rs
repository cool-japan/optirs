// Micro-batch pipeline-parallel schedulers, partitioning and analysis.
//
// This module is a pure-Rust, CPU-only *simulation* of pipeline parallelism. No
// real devices are involved: the "stages" are logical and timing is driven by a
// pluggable per-stage compute cost. The module is therefore a faithful reference
// for reasoning about pipeline efficiency (bubble, utilisation, activation
// memory) that a real multi-device runtime could later be calibrated against.
//
// Two classic schedules are implemented:
//
// * **GPipe** (Huang et al., 2019) splits a minibatch into `M` micro-batches and
//   runs *all* forwards through stages `0..P` and then *all* backwards through
//   stages `P-1..=0`. It is simple and has bubble fraction `(P-1)/(M+P-1)`, but
//   every stage must stash `M` micro-batch activations simultaneously.
//
// * **1F1B / PipeDream-Flush** (Narayanan et al., 2021) reaches the same steady
//   state bubble fraction but, after a `P-1-stage` warm-up of forwards, alternates
//   one forward and one backward. This bounds the in-flight (stashed) activations
//   per stage to the pipeline depth (`min(P, M)`) instead of `M`, dramatically
//   lowering peak activation memory for large `M`.
//
// # Timing model
// Each stage is a single resource that runs its assigned operations sequentially
// in the schedule-specified order. An operation's earliest start is the maximum
// finish time of (a) the previous operation on the same stage (resource edge) and
// (b) its data dependencies: a forward `F(m, s)` needs `F(m, s-1)`; a backward
// `B(m, s)` needs `B(m, s+1)` (the downstream gradient) and `F(m, s)` (the
// stashed activations). The resulting directed acyclic graph is scheduled by a
// topological earliest-finish pass, giving exact start/end times for arbitrary,
// non-uniform per-stage costs.
//
// # Stage partitioning
// [`StagePartitioner`] solves the balanced contiguous partition problem: split
// `L` layers, each with a compute cost, into exactly `P` contiguous stages that
// minimise the maximum per-stage load. This is the classic "split an array into
// `P` parts minimising the largest part sum" problem, solved here exactly with
// dynamic programming and reconstructed split points.

use crate::error::{OptimError, Result};
use std::collections::VecDeque;

/// Which pipeline schedule to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// All-forward-then-all-backward (GPipe). Simple, but stashes `M`
    /// activations per stage.
    GPipe,
    /// One-forward-one-backward steady state (PipeDream-Flush). Same bubble as
    /// GPipe but bounds stashed activations to the pipeline depth.
    OneForwardOneBackward,
}

impl PipelineSchedule {
    /// Human-readable name of the schedule.
    pub fn name(self) -> &'static str {
        match self {
            PipelineSchedule::GPipe => "GPipe",
            PipelineSchedule::OneForwardOneBackward => "1F1B",
        }
    }
}

/// Whether a pipeline operation is a forward or a backward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Forward pass: stashes one micro-batch activation on the stage.
    Forward,
    /// Backward pass: releases one stashed micro-batch activation.
    Backward,
}

/// Per-stage forward/backward compute cost in abstract time units.
///
/// Costs are pluggable: they need not correspond to any particular hardware. The
/// only requirement is that both are finite and strictly positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageCost {
    /// Cost of a single forward pass on this stage.
    pub forward: f64,
    /// Cost of a single backward pass on this stage.
    pub backward: f64,
}

impl StageCost {
    /// Create a stage cost from explicit forward and backward costs.
    pub fn new(forward: f64, backward: f64) -> Self {
        Self { forward, backward }
    }

    /// Create a stage cost with equal forward and backward cost.
    pub fn uniform(value: f64) -> Self {
        Self {
            forward: value,
            backward: value,
        }
    }
}

/// A single scheduled pipeline operation with its computed timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineOp {
    /// Stage (device) index in `0..num_stages`.
    pub stage: usize,
    /// Micro-batch index in `0..num_micro_batches`.
    pub micro_batch: usize,
    /// Forward or backward.
    pub kind: OpKind,
    /// Earliest start time in cost units.
    pub start: f64,
    /// Finish time in cost units (`start + cost`).
    pub end: f64,
}

/// Configuration of a pipeline: number of stages and micro-batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    /// Number of pipeline stages `P` (pipeline depth).
    pub num_stages: usize,
    /// Number of micro-batches `M` the minibatch is split into.
    pub num_micro_batches: usize,
}

impl PipelineConfig {
    /// Create a configuration, validating that both counts are at least one.
    pub fn new(num_stages: usize, num_micro_batches: usize) -> Result<Self> {
        if num_stages == 0 {
            return Err(OptimError::InvalidConfig(
                "num_stages must be at least 1".to_string(),
            ));
        }
        if num_micro_batches == 0 {
            return Err(OptimError::InvalidConfig(
                "num_micro_batches must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            num_stages,
            num_micro_batches,
        })
    }

    /// Analytical GPipe bubble fraction `(P-1)/(M+P-1)`.
    ///
    /// This is the fraction of stage-time wasted to pipeline fill/drain under the
    /// idealised uniform-cost model. The generated schedule's measured idle
    /// fraction equals this value when all stage costs are uniform.
    pub fn analytical_bubble_fraction(&self) -> f64 {
        let p = self.num_stages as f64;
        let m = self.num_micro_batches as f64;
        (p - 1.0) / (m + p - 1.0)
    }

    /// Analytical pipeline utilisation `M/(M+P-1) = 1 - bubble_fraction`.
    pub fn analytical_utilization(&self) -> f64 {
        let p = self.num_stages as f64;
        let m = self.num_micro_batches as f64;
        m / (m + p - 1.0)
    }
}

/// Efficiency metrics derived from a generated schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineMetrics {
    /// Fraction of total stage-time spent idle (the pipeline bubble),
    /// `1 - utilization`. Measured directly from the generated schedule.
    pub bubble_fraction: f64,
    /// Fraction of total stage-time spent doing useful work,
    /// `total_busy / (num_stages * makespan)`.
    pub utilization: f64,
    /// Peak number of in-flight (stashed) micro-batch activations across all
    /// stages. GPipe peaks at `M`; 1F1B peaks at `min(P, M)`.
    pub peak_activation_stash: usize,
    /// Peak stashed activations for each stage individually.
    pub per_stage_peak_stash: Vec<usize>,
    /// Total wall-clock makespan of the schedule in cost units.
    pub makespan: f64,
    /// Throughput in micro-batches per cost unit (`M / makespan`).
    pub throughput: f64,
}

/// A fully scheduled pipeline execution: ordered ops plus efficiency metrics.
#[derive(Debug, Clone)]
pub struct PipelineExecution {
    /// Which schedule produced this execution.
    pub schedule: PipelineSchedule,
    /// The configuration that was scheduled.
    pub config: PipelineConfig,
    /// All `2 * P * M` operations, ordered by start time.
    pub ops: Vec<PipelineOp>,
    /// Computed efficiency metrics.
    pub metrics: PipelineMetrics,
}

impl PipelineExecution {
    /// Borrow the ordered operation list.
    pub fn ops(&self) -> &[PipelineOp] {
        &self.ops
    }

    /// Borrow the computed metrics.
    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    /// Total makespan of the schedule in cost units.
    pub fn makespan(&self) -> f64 {
        self.metrics.makespan
    }

    /// Total busy (non-idle) stage-time, i.e. the sum of all op durations.
    pub fn total_busy_time(&self) -> f64 {
        self.ops.iter().map(|op| op.end - op.start).sum()
    }
}

/// Contiguous range of layers assigned to one pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageRange {
    /// Stage (device) index in `0..num_stages`.
    pub stage: usize,
    /// First layer index assigned to the stage (inclusive).
    pub start_layer: usize,
    /// One past the last layer index assigned to the stage (exclusive).
    pub end_layer: usize,
    /// Sum of the costs of the layers in this stage.
    pub load: f64,
}

impl StageRange {
    /// Number of layers assigned to this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }
}

/// Balanced contiguous stage partitioner.
///
/// Splits `L` layers into exactly `P` contiguous stages minimising the maximum
/// per-stage load, via exact dynamic programming.
#[derive(Debug, Clone, Copy, Default)]
pub struct StagePartitioner;

impl StagePartitioner {
    /// Create a new partitioner.
    pub fn new() -> Self {
        Self
    }

    /// Partition `layer_costs` into exactly `num_stages` contiguous stages so the
    /// maximum per-stage load is minimised.
    ///
    /// Returns one [`StageRange`] per stage in order; the ranges are contiguous,
    /// non-overlapping, non-empty and together cover every layer. The maximum
    /// stage load equals the optimum (see [`StagePartitioner::optimal_max_load`]).
    ///
    /// # Errors
    /// Returns an error when `num_stages == 0`, `layer_costs` is empty,
    /// `num_stages` exceeds the number of layers, or any cost is negative or
    /// non-finite.
    pub fn partition(&self, layer_costs: &[f64], num_stages: usize) -> Result<Vec<StageRange>> {
        let num_layers = layer_costs.len();
        if num_stages == 0 {
            return Err(OptimError::InvalidConfig(
                "num_stages must be at least 1".to_string(),
            ));
        }
        if num_layers == 0 {
            return Err(OptimError::InvalidConfig(
                "layer_costs must not be empty".to_string(),
            ));
        }
        if num_stages > num_layers {
            return Err(OptimError::InvalidConfig(format!(
                "num_stages {num_stages} exceeds number of layers {num_layers}: \
                 cannot form non-empty contiguous stages"
            )));
        }
        for (i, &cost) in layer_costs.iter().enumerate() {
            if !cost.is_finite() || cost < 0.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "layer {i} cost {cost} must be finite and non-negative"
                )));
            }
        }

        // Prefix sums: prefix[i] is the total cost of layers 0..i.
        let mut prefix = vec![0.0f64; num_layers + 1];
        for i in 0..num_layers {
            prefix[i + 1] = prefix[i] + layer_costs[i];
        }

        // dp[p][i] = minimal achievable maximum load when splitting the first `i`
        // layers into exactly `p` contiguous stages. choice[p][i] records the
        // start index of the `p`-th (last) stage for reconstruction.
        let mut dp = vec![vec![f64::INFINITY; num_layers + 1]; num_stages + 1];
        let mut choice = vec![vec![0usize; num_layers + 1]; num_stages + 1];

        // Base case: a single stage covering the first `i` layers.
        dp[1][1..=num_layers].copy_from_slice(&prefix[1..=num_layers]);

        // Fill the table: the last stage covers [j, i); the first j layers are
        // split into p-1 stages. Each stage must be non-empty, so j ranges over
        // (p-1)..i (at least one layer per earlier stage, at least one here).
        for stages in 2..=num_stages {
            for i in stages..=num_layers {
                let mut best = f64::INFINITY;
                let mut best_j = stages - 1;
                for j in (stages - 1)..i {
                    let last_load = prefix[i] - prefix[j];
                    let candidate = dp[stages - 1][j].max(last_load);
                    if candidate < best {
                        best = candidate;
                        best_j = j;
                    }
                }
                dp[stages][i] = best;
                choice[stages][i] = best_j;
            }
        }

        // Reconstruct the stage boundaries from the back.
        let mut ranges: Vec<StageRange> = Vec::with_capacity(num_stages);
        let mut end = num_layers;
        let mut stages = num_stages;
        while stages >= 1 {
            let start = if stages == 1 { 0 } else { choice[stages][end] };
            ranges.push(StageRange {
                stage: stages - 1,
                start_layer: start,
                end_layer: end,
                load: prefix[end] - prefix[start],
            });
            end = start;
            stages -= 1;
        }
        ranges.reverse();
        Ok(ranges)
    }

    /// Minimal achievable maximum per-stage load for the balanced partition.
    ///
    /// Equal to the largest stage load of [`StagePartitioner::partition`].
    pub fn optimal_max_load(&self, layer_costs: &[f64], num_stages: usize) -> Result<f64> {
        let ranges = self.partition(layer_costs, num_stages)?;
        Ok(ranges.iter().map(|range| range.load).fold(0.0f64, f64::max))
    }
}

/// Flat operation index for `(stage, micro_batch, kind)`.
///
/// The encoding is a perfect hash over `0..(2 * P * M)`, avoiding any map.
#[inline]
fn op_index(stage: usize, micro: usize, kind: OpKind, num_micro: usize) -> usize {
    let kind_idx = match kind {
        OpKind::Forward => 0,
        OpKind::Backward => 1,
    };
    (stage * num_micro + micro) * 2 + kind_idx
}

/// Inverse of [`op_index`].
#[inline]
fn decode_index(index: usize, num_micro: usize) -> (usize, usize, OpKind) {
    let kind = if index.is_multiple_of(2) {
        OpKind::Forward
    } else {
        OpKind::Backward
    };
    let rest = index / 2;
    let micro = rest % num_micro;
    let stage = rest / num_micro;
    (stage, micro, kind)
}

/// Per-stage execution order for GPipe: all forwards (`0..M`) then all backwards
/// in reverse micro-batch order (`M-1..=0`), identical on every stage.
fn gpipe_stage_orders(num_stages: usize, num_micro: usize) -> Vec<Vec<(usize, OpKind)>> {
    let mut orders = Vec::with_capacity(num_stages);
    for _ in 0..num_stages {
        let mut order = Vec::with_capacity(2 * num_micro);
        for micro in 0..num_micro {
            order.push((micro, OpKind::Forward));
        }
        for micro in (0..num_micro).rev() {
            order.push((micro, OpKind::Backward));
        }
        orders.push(order);
    }
    orders
}

/// Per-stage execution order for 1F1B / PipeDream-Flush.
///
/// Stage `s` first issues `warmup = min(P-1-s, M)` forwards, then `M - warmup`
/// steady-state (forward, backward) pairs, then drains the remaining backwards.
/// Forwards are issued in order `0..M` and backwards in order `0..M`.
fn one_f_one_b_stage_orders(num_stages: usize, num_micro: usize) -> Vec<Vec<(usize, OpKind)>> {
    let mut orders = Vec::with_capacity(num_stages);
    for stage in 0..num_stages {
        let warmup = (num_stages - 1 - stage).min(num_micro);
        let steady = num_micro - warmup;
        let mut order = Vec::with_capacity(2 * num_micro);

        // Warm-up forwards.
        for micro in 0..warmup {
            order.push((micro, OpKind::Forward));
        }
        // Steady state: one forward then one backward.
        for k in 0..steady {
            order.push((warmup + k, OpKind::Forward));
            order.push((k, OpKind::Backward));
        }
        // Cool-down: remaining backwards.
        for micro in steady..num_micro {
            order.push((micro, OpKind::Backward));
        }
        orders.push(order);
    }
    orders
}

/// Earliest-finish topological scheduling of the dependency DAG.
///
/// Returns the timed ops (ordered by start time) and the makespan.
fn compute_timeline(
    num_stages: usize,
    num_micro: usize,
    stage_orders: &[Vec<(usize, OpKind)>],
    stage_costs: &[StageCost],
) -> Result<(Vec<PipelineOp>, f64)> {
    let num_ops = num_stages * num_micro * 2;
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_ops];

    // Resource edges: consecutive ops on the same stage.
    for (stage, order) in stage_orders.iter().enumerate() {
        for window in order.windows(2) {
            let prev = op_index(stage, window[0].0, window[0].1, num_micro);
            let cur = op_index(stage, window[1].0, window[1].1, num_micro);
            preds[cur].push(prev);
        }
    }

    // Data edges: forward chain, backward chain, and activation dependency.
    for micro in 0..num_micro {
        for stage in 0..num_stages {
            let forward = op_index(stage, micro, OpKind::Forward, num_micro);
            if stage > 0 {
                preds[forward].push(op_index(stage - 1, micro, OpKind::Forward, num_micro));
            }
            let backward = op_index(stage, micro, OpKind::Backward, num_micro);
            if stage + 1 < num_stages {
                preds[backward].push(op_index(stage + 1, micro, OpKind::Backward, num_micro));
            }
            preds[backward].push(forward);
        }
    }

    // Build successor lists and in-degrees for Kahn's algorithm.
    let mut indeg = vec![0usize; num_ops];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); num_ops];
    for (op, plist) in preds.iter().enumerate() {
        indeg[op] = plist.len();
        for &pred in plist {
            succ[pred].push(op);
        }
    }

    let mut start = vec![0.0f64; num_ops];
    let mut end = vec![0.0f64; num_ops];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (op, &deg) in indeg.iter().enumerate() {
        if deg == 0 {
            queue.push_back(op);
        }
    }

    let mut processed = 0usize;
    while let Some(op) = queue.pop_front() {
        // Earliest start is the latest finish among data + resource predecessors.
        let mut earliest = 0.0f64;
        for &pred in &preds[op] {
            if end[pred] > earliest {
                earliest = end[pred];
            }
        }
        let (stage, _micro, kind) = decode_index(op, num_micro);
        let cost = match kind {
            OpKind::Forward => stage_costs[stage].forward,
            OpKind::Backward => stage_costs[stage].backward,
        };
        start[op] = earliest;
        end[op] = earliest + cost;
        processed += 1;

        for &next in &succ[op] {
            indeg[next] -= 1;
            if indeg[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if processed != num_ops {
        return Err(OptimError::InvalidState(
            "pipeline dependency graph is cyclic; schedule is infeasible".to_string(),
        ));
    }

    let mut makespan = 0.0f64;
    let mut ops = Vec::with_capacity(num_ops);
    for op in 0..num_ops {
        let (stage, micro, kind) = decode_index(op, num_micro);
        if end[op] > makespan {
            makespan = end[op];
        }
        ops.push(PipelineOp {
            stage,
            micro_batch: micro,
            kind,
            start: start[op],
            end: end[op],
        });
    }

    ops.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.stage.cmp(&b.stage))
            .then((a.kind as usize).cmp(&(b.kind as usize)))
            .then(a.micro_batch.cmp(&b.micro_batch))
    });

    Ok((ops, makespan))
}

/// Peak stashed (in-flight) activations per stage, read off the op order.
///
/// A forward stashes one activation; the matching backward releases it. The peak
/// of the running count is the stage's activation memory pressure.
fn compute_peak_stash(stage_orders: &[Vec<(usize, OpKind)>]) -> Vec<usize> {
    let mut peaks = Vec::with_capacity(stage_orders.len());
    for order in stage_orders {
        let mut current = 0i64;
        let mut peak = 0i64;
        for &(_, kind) in order {
            match kind {
                OpKind::Forward => {
                    current += 1;
                    if current > peak {
                        peak = current;
                    }
                }
                OpKind::Backward => {
                    current -= 1;
                }
            }
        }
        peaks.push(peak.max(0) as usize);
    }
    peaks
}

/// Driver that turns a [`PipelineConfig`] into timed schedules and metrics.
#[derive(Debug, Clone, Copy)]
pub struct PipelineScheduler {
    config: PipelineConfig,
}

impl PipelineScheduler {
    /// Create a scheduler for the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// The configuration this scheduler drives.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Generate a schedule for `schedule_kind` driven by per-stage `stage_costs`.
    ///
    /// `stage_costs` must contain exactly `num_stages` entries with finite,
    /// strictly positive forward and backward costs.
    ///
    /// # Errors
    /// Returns an error when the number of costs does not match the number of
    /// stages, when any cost is non-finite or non-positive, or (defensively) when
    /// the generated dependency graph is cyclic.
    pub fn schedule(
        &self,
        schedule_kind: PipelineSchedule,
        stage_costs: &[StageCost],
    ) -> Result<PipelineExecution> {
        let num_stages = self.config.num_stages;
        let num_micro = self.config.num_micro_batches;

        if stage_costs.len() != num_stages {
            return Err(OptimError::DimensionMismatch(format!(
                "expected {num_stages} stage costs (one per stage), got {}",
                stage_costs.len()
            )));
        }
        for (stage, cost) in stage_costs.iter().enumerate() {
            if !cost.forward.is_finite() || cost.forward <= 0.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "stage {stage} forward cost {} must be finite and positive",
                    cost.forward
                )));
            }
            if !cost.backward.is_finite() || cost.backward <= 0.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "stage {stage} backward cost {} must be finite and positive",
                    cost.backward
                )));
            }
        }

        let stage_orders = match schedule_kind {
            PipelineSchedule::GPipe => gpipe_stage_orders(num_stages, num_micro),
            PipelineSchedule::OneForwardOneBackward => {
                one_f_one_b_stage_orders(num_stages, num_micro)
            }
        };

        let (ops, makespan) = compute_timeline(num_stages, num_micro, &stage_orders, stage_costs)?;
        let per_stage_peak_stash = compute_peak_stash(&stage_orders);
        let peak_activation_stash = per_stage_peak_stash.iter().copied().max().unwrap_or(0);

        let total_busy: f64 = stage_costs
            .iter()
            .map(|cost| (cost.forward + cost.backward) * num_micro as f64)
            .sum();
        let capacity = num_stages as f64 * makespan;
        let utilization = if capacity > 0.0 {
            (total_busy / capacity).min(1.0)
        } else {
            0.0
        };
        let bubble_fraction = (1.0 - utilization).max(0.0);
        let throughput = if makespan > 0.0 {
            num_micro as f64 / makespan
        } else {
            0.0
        };

        let metrics = PipelineMetrics {
            bubble_fraction,
            utilization,
            peak_activation_stash,
            per_stage_peak_stash,
            makespan,
            throughput,
        };

        Ok(PipelineExecution {
            schedule: schedule_kind,
            config: self.config,
            ops,
            metrics,
        })
    }

    /// Convenience wrapper around [`PipelineScheduler::schedule`] with a single
    /// forward/backward cost applied uniformly to every stage.
    pub fn schedule_uniform(
        &self,
        schedule_kind: PipelineSchedule,
        forward: f64,
        backward: f64,
    ) -> Result<PipelineExecution> {
        let stage_costs = vec![StageCost::new(forward, backward); self.config.num_stages];
        self.schedule(schedule_kind, &stage_costs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Independent brute-force optimum for the balanced contiguous partition,
    /// used to validate the dynamic program.
    fn brute_force_max_load(layer_costs: &[f64], num_stages: usize) -> f64 {
        let num_layers = layer_costs.len();
        let mut prefix = vec![0.0f64; num_layers + 1];
        for i in 0..num_layers {
            prefix[i + 1] = prefix[i] + layer_costs[i];
        }

        fn rec(prefix: &[f64], start: usize, stages: usize, num_layers: usize) -> f64 {
            if stages == 1 {
                return prefix[num_layers] - prefix[start];
            }
            let mut best = f64::INFINITY;
            // Leave at least one layer for each of the remaining stages.
            let last_end = num_layers - (stages - 1);
            for end in (start + 1)..=last_end {
                let first = prefix[end] - prefix[start];
                let rest = rec(prefix, end, stages - 1, num_layers);
                let candidate = first.max(rest);
                if candidate < best {
                    best = candidate;
                }
            }
            best
        }

        rec(&prefix, 0, num_stages, num_layers)
    }

    fn assert_contiguous_cover(ranges: &[StageRange], num_layers: usize, num_stages: usize) {
        assert_eq!(ranges.len(), num_stages, "wrong number of stages");
        assert_eq!(
            ranges[0].start_layer, 0,
            "first stage must start at layer 0"
        );
        assert_eq!(
            ranges[num_stages - 1].end_layer,
            num_layers,
            "last stage must end at the final layer"
        );
        for (i, range) in ranges.iter().enumerate() {
            assert_eq!(range.stage, i, "stage index out of order");
            assert!(range.num_layers() >= 1, "every stage must be non-empty");
            if i + 1 < ranges.len() {
                assert_eq!(
                    range.end_layer,
                    ranges[i + 1].start_layer,
                    "stages must be contiguous"
                );
            }
        }
    }

    #[test]
    fn test_partition_balances_uniform_load() {
        let partitioner = StagePartitioner::new();
        let costs = vec![1.0f64; 8];
        let ranges = partitioner.partition(&costs, 4).unwrap();

        assert_contiguous_cover(&ranges, 8, 4);
        for range in &ranges {
            assert_eq!(range.num_layers(), 2);
            assert_relative_eq!(range.load, 2.0, epsilon = 1e-12);
        }
        let max_load = ranges.iter().map(|r| r.load).fold(0.0, f64::max);
        assert_relative_eq!(max_load, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_partition_matches_brute_force_optimum() {
        let partitioner = StagePartitioner::new();
        let cases: &[(Vec<f64>, usize)] = &[
            (vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0], 3),
            (vec![5.0, 2.0, 4.0, 1.0, 1.0, 9.0, 3.0, 2.0], 4),
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2),
            (vec![10.0, 1.0, 1.0, 1.0, 1.0], 5),
            (vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 3),
        ];
        for (costs, num_stages) in cases {
            let ranges = partitioner.partition(costs, *num_stages).unwrap();
            assert_contiguous_cover(&ranges, costs.len(), *num_stages);

            let dp_max = ranges.iter().map(|r| r.load).fold(0.0, f64::max);
            let optimum = brute_force_max_load(costs, *num_stages);
            assert_relative_eq!(dp_max, optimum, epsilon = 1e-9);

            let reported = partitioner.optimal_max_load(costs, *num_stages).unwrap();
            assert_relative_eq!(reported, optimum, epsilon = 1e-9);

            // The balanced max load must never exceed a naive equal-count split.
            let naive = naive_equal_count_max_load(costs, *num_stages);
            assert!(
                dp_max <= naive + 1e-9,
                "balanced split must beat naive split"
            );
        }
    }

    /// Reference: greedily slice into `num_stages` runs of (almost) equal layer
    /// counts and report the largest run load.
    fn naive_equal_count_max_load(costs: &[f64], num_stages: usize) -> f64 {
        let num_layers = costs.len();
        let base = num_layers / num_stages;
        let rem = num_layers % num_stages;
        let mut idx = 0usize;
        let mut max_load = 0.0f64;
        for stage in 0..num_stages {
            let count = if stage < rem { base + 1 } else { base };
            let load: f64 = costs[idx..idx + count].iter().sum();
            if load > max_load {
                max_load = load;
            }
            idx += count;
        }
        max_load
    }

    #[test]
    fn test_partition_invalid_configs() {
        let partitioner = StagePartitioner::new();
        // num_stages == 0.
        assert!(partitioner.partition(&[1.0, 2.0], 0).is_err());
        // empty layers.
        assert!(partitioner.partition(&[], 1).is_err());
        // num_stages > num_layers.
        assert!(partitioner.partition(&[1.0, 2.0], 3).is_err());
        // negative cost.
        assert!(partitioner.partition(&[1.0, -1.0, 2.0], 2).is_err());
        // non-finite cost.
        assert!(partitioner.partition(&[1.0, f64::NAN], 2).is_err());
        // valid edge: one stage per layer.
        assert!(partitioner.partition(&[1.0, 2.0, 3.0], 3).is_ok());
    }

    #[test]
    fn test_pipeline_config_validation() {
        assert!(PipelineConfig::new(0, 4).is_err());
        assert!(PipelineConfig::new(4, 0).is_err());
        assert!(PipelineConfig::new(1, 1).is_ok());
        assert!(PipelineConfig::new(4, 8).is_ok());
    }

    #[test]
    fn test_gpipe_bubble_fraction_matches_formula() {
        let cases = [(2usize, 2usize), (4, 8), (4, 1), (8, 16), (3, 5), (1, 4)];
        for (p, m) in cases {
            let config = PipelineConfig::new(p, m).unwrap();
            let scheduler = PipelineScheduler::new(config);
            let exec = scheduler
                .schedule_uniform(PipelineSchedule::GPipe, 1.0, 1.0)
                .unwrap();

            let analytical = config.analytical_bubble_fraction();
            assert_relative_eq!(
                analytical,
                (p as f64 - 1.0) / (m as f64 + p as f64 - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(exec.metrics.bubble_fraction, analytical, epsilon = 1e-9);
            assert_relative_eq!(
                exec.metrics.utilization,
                config.analytical_utilization(),
                epsilon = 1e-9
            );
            // bubble + utilization == 1.
            assert_relative_eq!(
                exec.metrics.bubble_fraction + exec.metrics.utilization,
                1.0,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_gpipe_generated_idle_matches_analytical_bubble() {
        let cases = [(2usize, 4usize), (4, 8), (3, 6), (5, 10)];
        for (p, m) in cases {
            let config = PipelineConfig::new(p, m).unwrap();
            let scheduler = PipelineScheduler::new(config);
            let exec = scheduler
                .schedule_uniform(PipelineSchedule::GPipe, 1.0, 1.0)
                .unwrap();

            // Recompute idle fraction directly from the generated ops, fully
            // independent of the metrics struct.
            let busy: f64 = exec.ops.iter().map(|op| op.end - op.start).sum();
            let capacity = p as f64 * exec.metrics.makespan;
            let idle_fraction = 1.0 - busy / capacity;

            assert_relative_eq!(
                idle_fraction,
                config.analytical_bubble_fraction(),
                epsilon = 1e-9
            );
            // Uniform GPipe makespan is exactly 2 * (M + P - 1).
            assert_relative_eq!(
                exec.metrics.makespan,
                2.0 * (m as f64 + p as f64 - 1.0),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_one_f_one_b_lower_activation_stash() {
        let cases = [(4usize, 8usize), (8, 16), (4, 4), (3, 10), (6, 2)];
        for (p, m) in cases {
            let config = PipelineConfig::new(p, m).unwrap();
            let scheduler = PipelineScheduler::new(config);

            let gpipe = scheduler
                .schedule_uniform(PipelineSchedule::GPipe, 1.0, 1.0)
                .unwrap();
            let one_f_one_b = scheduler
                .schedule_uniform(PipelineSchedule::OneForwardOneBackward, 1.0, 1.0)
                .unwrap();

            // GPipe stashes exactly M activations per stage.
            assert_eq!(gpipe.metrics.peak_activation_stash, m);

            // 1F1B caps the stash at min(P, M) and never exceeds GPipe.
            assert_eq!(
                one_f_one_b.metrics.peak_activation_stash,
                p.min(m),
                "1F1B peak stash should equal min(P, M)"
            );
            assert!(
                one_f_one_b.metrics.peak_activation_stash <= gpipe.metrics.peak_activation_stash,
                "1F1B peak must not exceed GPipe peak"
            );
            assert!(
                one_f_one_b.metrics.peak_activation_stash <= p,
                "1F1B peak must not exceed pipeline depth P"
            );
            for &stage_peak in &one_f_one_b.metrics.per_stage_peak_stash {
                assert!(stage_peak <= p, "per-stage 1F1B stash must be <= P");
            }
        }
    }

    #[test]
    fn test_one_f_one_b_strictly_lower_stash_for_large_m() {
        let config = PipelineConfig::new(4, 16).unwrap();
        let scheduler = PipelineScheduler::new(config);
        let gpipe = scheduler
            .schedule_uniform(PipelineSchedule::GPipe, 1.0, 1.0)
            .unwrap();
        let one_f_one_b = scheduler
            .schedule_uniform(PipelineSchedule::OneForwardOneBackward, 1.0, 1.0)
            .unwrap();
        assert_eq!(gpipe.metrics.peak_activation_stash, 16);
        assert_eq!(one_f_one_b.metrics.peak_activation_stash, 4);
        assert!(one_f_one_b.metrics.peak_activation_stash < gpipe.metrics.peak_activation_stash);
    }

    #[test]
    fn test_gpipe_and_one_f_one_b_same_bubble_and_makespan_uniform() {
        // 1F1B trades memory, not throughput: same bubble and makespan as GPipe
        // under uniform costs.
        let cases = [(2usize, 2usize), (4, 8), (3, 7), (5, 5)];
        for (p, m) in cases {
            let config = PipelineConfig::new(p, m).unwrap();
            let scheduler = PipelineScheduler::new(config);
            let gpipe = scheduler
                .schedule_uniform(PipelineSchedule::GPipe, 1.0, 1.0)
                .unwrap();
            let one_f_one_b = scheduler
                .schedule_uniform(PipelineSchedule::OneForwardOneBackward, 1.0, 1.0)
                .unwrap();
            assert_relative_eq!(
                gpipe.metrics.makespan,
                one_f_one_b.metrics.makespan,
                epsilon = 1e-9
            );
            assert_relative_eq!(
                gpipe.metrics.makespan,
                2.0 * (m as f64 + p as f64 - 1.0),
                epsilon = 1e-9
            );
            assert_relative_eq!(
                gpipe.metrics.bubble_fraction,
                one_f_one_b.metrics.bubble_fraction,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_throughput_increases_with_micro_batches() {
        for schedule in [
            PipelineSchedule::GPipe,
            PipelineSchedule::OneForwardOneBackward,
        ] {
            let micro_batches = [1usize, 2, 4, 8, 16];
            let mut previous = 0.0f64;
            for &m in &micro_batches {
                let config = PipelineConfig::new(4, m).unwrap();
                let scheduler = PipelineScheduler::new(config);
                let exec = scheduler.schedule_uniform(schedule, 1.0, 1.0).unwrap();
                assert!(
                    exec.metrics.throughput > previous,
                    "throughput must increase with M for {} (M={m})",
                    schedule.name()
                );
                previous = exec.metrics.throughput;
            }
        }
    }

    #[test]
    fn test_schedule_structure_is_valid() {
        let config = PipelineConfig::new(4, 6).unwrap();
        let scheduler = PipelineScheduler::new(config);
        for schedule in [
            PipelineSchedule::GPipe,
            PipelineSchedule::OneForwardOneBackward,
        ] {
            let exec = scheduler.schedule_uniform(schedule, 1.0, 2.0).unwrap();

            // Exactly 2 * P * M ops.
            assert_eq!(exec.ops.len(), 4 * 6 * 2);

            // Each (stage, micro) has exactly one forward and one backward, and
            // the forward finishes no later than the backward starts.
            for stage in 0..4 {
                for micro in 0..6 {
                    let forward = exec
                        .ops
                        .iter()
                        .find(|op| {
                            op.stage == stage
                                && op.micro_batch == micro
                                && op.kind == OpKind::Forward
                        })
                        .unwrap();
                    let backward = exec
                        .ops
                        .iter()
                        .find(|op| {
                            op.stage == stage
                                && op.micro_batch == micro
                                && op.kind == OpKind::Backward
                        })
                        .unwrap();
                    assert!(forward.end <= backward.start + 1e-9);
                    // Forward cost 1.0, backward cost 2.0.
                    assert_relative_eq!(forward.end - forward.start, 1.0, epsilon = 1e-9);
                    assert_relative_eq!(backward.end - backward.start, 2.0, epsilon = 1e-9);
                }
            }

            // Forward data dependency: F(m, s) finishes no later than F(m, s+1)
            // starts.
            for micro in 0..6 {
                for stage in 0..3 {
                    let here = exec
                        .ops
                        .iter()
                        .find(|op| {
                            op.stage == stage
                                && op.micro_batch == micro
                                && op.kind == OpKind::Forward
                        })
                        .unwrap();
                    let next = exec
                        .ops
                        .iter()
                        .find(|op| {
                            op.stage == stage + 1
                                && op.micro_batch == micro
                                && op.kind == OpKind::Forward
                        })
                        .unwrap();
                    assert!(here.end <= next.start + 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_schedule_invalid_costs() {
        let config = PipelineConfig::new(3, 4).unwrap();
        let scheduler = PipelineScheduler::new(config);

        // Wrong number of stage costs.
        let too_few = vec![StageCost::uniform(1.0); 2];
        assert!(scheduler
            .schedule(PipelineSchedule::GPipe, &too_few)
            .is_err());

        // Non-positive cost.
        let bad = vec![
            StageCost::new(1.0, 1.0),
            StageCost::new(0.0, 1.0),
            StageCost::new(1.0, 1.0),
        ];
        assert!(scheduler.schedule(PipelineSchedule::GPipe, &bad).is_err());

        // Non-finite cost.
        let infinite = vec![
            StageCost::new(1.0, 1.0),
            StageCost::new(1.0, f64::INFINITY),
            StageCost::new(1.0, 1.0),
        ];
        assert!(scheduler
            .schedule(PipelineSchedule::GPipe, &infinite)
            .is_err());
    }

    #[test]
    fn test_non_uniform_costs_bottleneck_dominates_makespan() {
        // A heavy middle stage should dominate the steady-state throughput.
        let config = PipelineConfig::new(3, 8).unwrap();
        let scheduler = PipelineScheduler::new(config);
        let stage_costs = [
            StageCost::new(1.0, 1.0),
            StageCost::new(4.0, 4.0),
            StageCost::new(1.0, 1.0),
        ];
        let exec = scheduler
            .schedule(PipelineSchedule::OneForwardOneBackward, &stage_costs)
            .unwrap();

        // The bottleneck stage performs 8 forwards + 8 backwards at cost 4 each =
        // 64 cost units of unavoidable work, so the makespan is at least that.
        assert!(exec.metrics.makespan >= 64.0 - 1e-9);
        assert!(exec.metrics.utilization > 0.0 && exec.metrics.utilization <= 1.0);
        assert!(exec.metrics.bubble_fraction >= 0.0 && exec.metrics.bubble_fraction < 1.0);
    }
}
