// Bandwidth-optimal ring all-reduce and all-gather collectives.
//
// This module implements the classic segmented ring all-reduce algorithm
// (popularised by Baidu and Horovod) as a pure-Rust, in-process simulation.
// There is no real network: every "worker" lives in the same process and the
// `CollectiveTransport` abstraction stands in for the neighbour send/recv that a
// real fabric (MPI / NCCL / gRPC) would perform. This makes the file a faithful
// CPU reference implementation that a real transport could later be plugged into.
//
// # Algorithm
// For a logical ring of `N` workers, each holding a vector of equal length, the
// reduction proceeds in two bandwidth-optimal phases, each of `N - 1` steps:
//
// 1. **Reduce-scatter.** Each vector is split into `N` contiguous segments. At
//    every step each worker sends one segment to its ring successor and folds the
//    segment it receives from its predecessor into the matching local segment
//    using the reduction operator. After `N - 1` steps worker `r` holds the fully
//    reduced value for exactly one distinct segment (segment `(r + 1) mod N`).
//
// 2. **All-gather.** The fully reduced segments are circulated around the ring so
//    that after a further `N - 1` steps every worker holds every reduced segment,
//    i.e. the complete result.
//
// # Bandwidth optimality
// Each worker sends and receives roughly `2 * (N - 1) / N * size` elements in
// total, independent of `N` for large `N`. This is why the segmented two-phase
// structure is preserved rather than collapsing to a trivial global sum: the
// communication volume per worker does not grow with the ring size.
//
// # Uneven segmentation
// When the vector length `L` is not divisible by `N` the first `L mod N` segments
// receive one extra element, so segment lengths differ by at most one. Segments of
// length zero (when `L < N`) are handled transparently.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Per-rank working state: one inner buffer per ring segment (or per gather slot).
type RankSegments<A> = Vec<Vec<A>>;

/// Reduction operators supported by the ring all-reduce.
///
/// The associative combine used during the reduce-scatter phase is exposed via
/// [`ReduceOp::apply`], and the neutral element of that combine via
/// [`ReduceOp::identity`]. [`ReduceOp::Mean`] reuses the additive combine and is
/// finalised by dividing the accumulated sum by the world size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Elementwise sum across all workers.
    Sum,
    /// Elementwise arithmetic mean across all workers (sum divided by world size).
    Mean,
    /// Elementwise maximum across all workers.
    Max,
    /// Elementwise minimum across all workers.
    Min,
    /// Elementwise product across all workers.
    Product,
}

impl ReduceOp {
    /// Combine two values with this operator.
    ///
    /// `Mean` uses the same additive combine as `Sum`; the division by the world
    /// size is applied once at the end of the reduction (see
    /// [`ReduceOp::needs_mean_finalize`]).
    pub fn apply<A: Float>(self, a: A, b: A) -> A {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }

    /// Neutral element of the combine operator.
    ///
    /// Folding `identity` with any value `v` via [`ReduceOp::apply`] yields `v`.
    pub fn identity<A: Float>(self) -> A {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => A::zero(),
            ReduceOp::Product => A::one(),
            ReduceOp::Max => A::neg_infinity(),
            ReduceOp::Min => A::infinity(),
        }
    }

    /// Whether the accumulated result must be divided by the world size.
    ///
    /// This is true only for [`ReduceOp::Mean`].
    pub fn needs_mean_finalize(self) -> bool {
        matches!(self, ReduceOp::Mean)
    }
}

/// Abstraction over the neighbour send/recv used by a single ring step.
///
/// A ring step is synchronous: every rank `r` posts the segment it wants to send
/// to its successor `(r + 1) mod world_size` and then collects the segment posted
/// to it by its predecessor `(r + world_size - 1) mod world_size`. Implementors
/// are free to back this with real network transfers; [`LocalTransport`] backs it
/// with in-process mailboxes.
///
/// The collective driver always posts *all* sends for a step before collecting
/// *any* receives, which preserves the snapshot semantics of a true simultaneous
/// ring exchange (a worker transmits the value it held at the start of the step).
pub trait CollectiveTransport<A> {
    /// Number of ranks participating in the ring.
    fn world_size(&self) -> usize;

    /// Post `segment` to be delivered from `src_rank` to its ring successor.
    fn send_to_successor(&mut self, src_rank: usize, segment: Vec<A>) -> Result<()>;

    /// Collect the segment delivered to `dst_rank` by its ring predecessor.
    fn recv_from_predecessor(&mut self, dst_rank: usize) -> Result<Vec<A>>;
}

/// In-process implementation of [`CollectiveTransport`] using per-rank mailboxes.
///
/// `mailboxes[r]` holds the single segment that has been delivered to rank `r`
/// and not yet collected. Because each rank in a ring step sends to a distinct
/// successor, at most one message is ever in flight per mailbox at a time.
#[derive(Debug)]
pub struct LocalTransport<A> {
    world_size: usize,
    mailboxes: Vec<Option<Vec<A>>>,
}

impl<A> LocalTransport<A> {
    /// Create a transport for a ring of `world_size` ranks.
    ///
    /// Returns an error when `world_size` is zero.
    pub fn new(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(OptimError::InvalidConfig(
                "world_size must be at least 1".to_string(),
            ));
        }
        let mut mailboxes = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            mailboxes.push(None);
        }
        Ok(Self {
            world_size,
            mailboxes,
        })
    }
}

impl<A> CollectiveTransport<A> for LocalTransport<A> {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send_to_successor(&mut self, src_rank: usize, segment: Vec<A>) -> Result<()> {
        if src_rank >= self.world_size {
            return Err(OptimError::InvalidConfig(format!(
                "src_rank {src_rank} out of range for world_size {}",
                self.world_size
            )));
        }
        let dst = (src_rank + 1) % self.world_size;
        if self.mailboxes[dst].is_some() {
            return Err(OptimError::InvalidState(format!(
                "mailbox for rank {dst} already holds an uncollected message"
            )));
        }
        self.mailboxes[dst] = Some(segment);
        Ok(())
    }

    fn recv_from_predecessor(&mut self, dst_rank: usize) -> Result<Vec<A>> {
        if dst_rank >= self.world_size {
            return Err(OptimError::InvalidConfig(format!(
                "dst_rank {dst_rank} out of range for world_size {}",
                self.world_size
            )));
        }
        self.mailboxes[dst_rank].take().ok_or_else(|| {
            OptimError::InvalidState(format!("no message waiting in mailbox for rank {dst_rank}"))
        })
    }
}

/// Non-negative remainder of `value` modulo `n`, returned as a `usize`.
#[inline]
fn modn(value: isize, n: usize) -> usize {
    let m = n as isize;
    (((value % m) + m) % m) as usize
}

/// Compute the `n + 1` segment boundary offsets for splitting a length-`l` vector
/// into `n` contiguous segments. The first `l % n` segments receive one extra
/// element so that segment lengths differ by at most one.
fn compute_segment_offsets(l: usize, n: usize) -> Vec<usize> {
    let base = l / n;
    let rem = l % n;
    let mut offsets = Vec::with_capacity(n + 1);
    let mut acc = 0usize;
    offsets.push(acc);
    for k in 0..n {
        let len = if k < rem { base + 1 } else { base };
        acc += len;
        offsets.push(acc);
    }
    offsets
}

/// Execute one synchronous ring step.
///
/// At `step`, rank `r` sends its segment indexed `(r - step + send_offset) mod n`
/// to its successor and writes the segment received from its predecessor into the
/// slot indexed `(r - step + recv_offset) mod n`. When `op` is `Some`, the
/// received segment is folded into the destination slot; when `None`, it
/// overwrites the destination slot (used by the all-gather phases).
fn ring_step<A, T>(
    transport: &mut T,
    states: &mut [RankSegments<A>],
    step: usize,
    send_offset: isize,
    recv_offset: isize,
    op: Option<ReduceOp>,
) -> Result<()>
where
    A: Float,
    T: CollectiveTransport<A>,
{
    let n = states.len();
    let step_i = step as isize;

    // Phase 1: post every send. Snapshotting the outgoing segments before any
    // receive mutates `states` preserves simultaneous-exchange semantics.
    for (r, segments) in states.iter().enumerate() {
        let send_chunk = modn(r as isize - step_i + send_offset, n);
        let segment = segments[send_chunk].clone();
        transport.send_to_successor(r, segment)?;
    }

    // Phase 2: collect every receive and combine into the destination slot.
    for (r, segments) in states.iter_mut().enumerate() {
        let recv_chunk = modn(r as isize - step_i + recv_offset, n);
        let incoming = transport.recv_from_predecessor(r)?;
        match op {
            Some(reduce_op) => {
                let slot = &mut segments[recv_chunk];
                if slot.len() != incoming.len() {
                    return Err(OptimError::DimensionMismatch(format!(
                        "segment length mismatch at rank {r}, chunk {recv_chunk}: \
                         local {} vs received {}",
                        slot.len(),
                        incoming.len()
                    )));
                }
                for (dst, src) in slot.iter_mut().zip(incoming.iter()) {
                    *dst = reduce_op.apply(*dst, *src);
                }
            }
            None => {
                segments[recv_chunk] = incoming;
            }
        }
    }

    Ok(())
}

/// Reduce-scatter phase: `n - 1` steps folding received segments with `op`.
///
/// After completion, rank `r` holds the fully reduced segment `(r + 1) mod n`.
fn reduce_scatter<A, T>(
    transport: &mut T,
    states: &mut [RankSegments<A>],
    op: ReduceOp,
) -> Result<()>
where
    A: Float,
    T: CollectiveTransport<A>,
{
    let n = states.len();
    for step in 0..(n - 1) {
        ring_step(transport, states, step, 0, -1, Some(op))?;
    }
    Ok(())
}

/// All-gather phase over the already-reduced segments (the second half of a full
/// all-reduce). Worker `r` starts circulating from segment `(r + 1) mod n`.
fn all_gather_reduced<A, T>(transport: &mut T, states: &mut [RankSegments<A>]) -> Result<()>
where
    A: Float,
    T: CollectiveTransport<A>,
{
    let n = states.len();
    for step in 0..(n - 1) {
        ring_step(transport, states, step, 1, 0, None)?;
    }
    Ok(())
}

/// Standalone ring all-gather over `n` owner slots. Worker `r` starts circulating
/// from its own slot `r`; after `n - 1` steps every worker holds every slot.
fn all_gather_slots<A, T>(transport: &mut T, states: &mut [RankSegments<A>]) -> Result<()>
where
    A: Float,
    T: CollectiveTransport<A>,
{
    let n = states.len();
    for step in 0..(n - 1) {
        ring_step(transport, states, step, 0, -1, None)?;
    }
    Ok(())
}

/// Flatten a rank's per-segment buffers into a single contiguous vector, in
/// segment order.
fn flatten_segments<A: Float>(segments: RankSegments<A>, capacity: usize) -> Array1<A> {
    let mut flat = Vec::with_capacity(capacity);
    for segment in segments {
        flat.extend(segment);
    }
    Array1::from_vec(flat)
}

/// Bandwidth-optimal ring all-reduce / all-gather driver.
///
/// A single [`RingAllReduce`] simulates the whole ring in-process: each public
/// method takes the per-rank inputs, drives all `world_size` ranks through the
/// collective using a [`LocalTransport`], and returns each rank's result (all
/// ranks end up holding identical data).
#[derive(Debug, Clone, Copy)]
pub struct RingAllReduce {
    world_size: usize,
}

impl RingAllReduce {
    /// Create a driver for a ring of `world_size` workers.
    ///
    /// Returns an error when `world_size` is zero.
    pub fn new(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(OptimError::InvalidConfig(
                "world_size must be at least 1".to_string(),
            ));
        }
        Ok(Self { world_size })
    }

    /// Number of workers in the ring.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Validate that `inputs` provides exactly one equal-length, non-empty vector
    /// per rank, returning the common length on success.
    fn validate_equal_length(&self, inputs: &[Array1<impl Float>]) -> Result<usize> {
        if inputs.is_empty() {
            return Err(OptimError::InvalidConfig(
                "inputs must not be empty".to_string(),
            ));
        }
        if inputs.len() != self.world_size {
            return Err(OptimError::DimensionMismatch(format!(
                "expected {} inputs (one per rank), got {}",
                self.world_size,
                inputs.len()
            )));
        }
        let len = inputs[0].len();
        if len == 0 {
            return Err(OptimError::InvalidConfig(
                "input vectors must have non-zero length".to_string(),
            ));
        }
        for (rank, vector) in inputs.iter().enumerate() {
            if vector.len() != len {
                return Err(OptimError::DimensionMismatch(format!(
                    "rank {rank} length {} does not match rank 0 length {len}",
                    vector.len()
                )));
            }
        }
        Ok(len)
    }

    /// Run the full simulated ring all-reduce across every rank at once.
    ///
    /// `inputs[r]` is rank `r`'s vector; all vectors must be non-empty and share
    /// the same length. The returned vector holds each rank's result, which are
    /// all identical and equal to the elementwise reduction under `op`.
    pub fn all_reduce_all<A>(&self, inputs: &[Array1<A>], op: ReduceOp) -> Result<Vec<Array1<A>>>
    where
        A: Float + ScalarOperand + Debug,
    {
        let len = self.validate_equal_length(inputs)?;
        let n = self.world_size;

        // A ring of one worker is the identity reduction (a reduction over a
        // single contributor is that contributor's own value, for every op).
        if n == 1 {
            return Ok(vec![inputs[0].clone()]);
        }

        // Split each rank's vector into `n` contiguous segments.
        let offsets = compute_segment_offsets(len, n);
        let mut states: Vec<RankSegments<A>> = Vec::with_capacity(n);
        for vector in inputs {
            let slice = vector.as_slice().ok_or_else(|| {
                OptimError::InvalidConfig("input vector must be contiguous".to_string())
            })?;
            let segments: RankSegments<A> = (0..n)
                .map(|k| slice[offsets[k]..offsets[k + 1]].to_vec())
                .collect();
            states.push(segments);
        }

        let mut transport = LocalTransport::<A>::new(n)?;
        reduce_scatter(&mut transport, &mut states, op)?;
        all_gather_reduced(&mut transport, &mut states)?;

        // Finalise the mean by dividing the accumulated sum by the world size.
        if op.needs_mean_finalize() {
            let denom = A::from(n).ok_or_else(|| {
                OptimError::InvalidConfig("cannot represent world_size as scalar".to_string())
            })?;
            for segments in states.iter_mut() {
                for segment in segments.iter_mut() {
                    for value in segment.iter_mut() {
                        *value = *value / denom;
                    }
                }
            }
        }

        let results = states
            .into_iter()
            .map(|segments| flatten_segments(segments, len))
            .collect();
        Ok(results)
    }

    /// Run the simulated ring all-gather across every rank at once.
    ///
    /// `inputs[r]` is rank `r`'s contribution; contributions may differ in length
    /// (this is a gather, not a reduction) but each must be non-empty. The result
    /// holds, for every rank, the concatenation `inputs[0] || inputs[1] || ... ||
    /// inputs[N - 1]` in rank order.
    pub fn all_gather<A>(&self, inputs: &[Array1<A>]) -> Result<Vec<Array1<A>>>
    where
        A: Float + ScalarOperand + Debug,
    {
        if inputs.is_empty() {
            return Err(OptimError::InvalidConfig(
                "inputs must not be empty".to_string(),
            ));
        }
        if inputs.len() != self.world_size {
            return Err(OptimError::DimensionMismatch(format!(
                "expected {} inputs (one per rank), got {}",
                self.world_size,
                inputs.len()
            )));
        }
        for (rank, vector) in inputs.iter().enumerate() {
            if vector.is_empty() {
                return Err(OptimError::InvalidConfig(format!(
                    "rank {rank} contribution must have non-zero length"
                )));
            }
        }

        let total: usize = inputs.iter().map(|vector| vector.len()).sum();
        let n = self.world_size;

        // A ring of one worker already holds the entire concatenation.
        if n == 1 {
            return Ok(vec![inputs[0].clone()]);
        }

        // Each rank starts owning only its own slot; the others are placeholders
        // that get filled as the data circulates around the ring.
        let mut states: Vec<RankSegments<A>> = Vec::with_capacity(n);
        for (rank, vector) in inputs.iter().enumerate() {
            let slice = vector.as_slice().ok_or_else(|| {
                OptimError::InvalidConfig("input vector must be contiguous".to_string())
            })?;
            let mut segments: RankSegments<A> = vec![Vec::new(); n];
            segments[rank] = slice.to_vec();
            states.push(segments);
        }

        let mut transport = LocalTransport::<A>::new(n)?;
        all_gather_slots(&mut transport, &mut states)?;

        let results = states
            .into_iter()
            .map(|segments| flatten_segments(segments, total))
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Independent, naive reference reduction used to validate the ring result.
    fn naive_reduce(inputs: &[Array1<f64>], op: ReduceOp) -> Array1<f64> {
        let len = inputs[0].len();
        let mut out = vec![op.identity::<f64>(); len];
        for vector in inputs {
            for (acc, &value) in out.iter_mut().zip(vector.iter()) {
                *acc = op.apply(*acc, value);
            }
        }
        if op.needs_mean_finalize() {
            let denom = inputs.len() as f64;
            for acc in out.iter_mut() {
                *acc /= denom;
            }
        }
        Array1::from_vec(out)
    }

    fn assert_all_ranks_eq(results: &[Array1<f64>], expected: &Array1<f64>) {
        for (rank, result) in results.iter().enumerate() {
            assert_eq!(result.len(), expected.len(), "rank {rank} length mismatch");
            for (index, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "rank {rank} coordinate {index}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn test_new_rejects_zero_world_size() {
        assert!(RingAllReduce::new(0).is_err());
        assert!(RingAllReduce::new(1).is_ok());
        assert!(RingAllReduce::new(8).is_ok());
    }

    #[test]
    fn test_ring_all_reduce_sum_matches_naive() {
        // N = 4, L = 10 -> uneven segmentation [3, 3, 2, 2].
        let n = 4;
        let l = 10;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec((0..l).map(|i| (r * 10 + i) as f64).collect()))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Sum).unwrap();

        // Closed form: sum_r (r*10 + i) = 60 + 4*i.
        let expected = Array1::from_vec((0..l).map(|i| 60.0 + 4.0 * i as f64).collect());
        assert_eq!(results.len(), n);
        assert_all_ranks_eq(&results, &expected);
        assert_all_ranks_eq(&results, &naive_reduce(&inputs, ReduceOp::Sum));
    }

    #[test]
    fn test_ring_all_reduce_mean() {
        // N = 3, L = 9 (evenly divisible).
        let n = 3;
        let l = 9;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec((0..l).map(|i| (r + i) as f64).collect()))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Mean).unwrap();

        // Closed form: mean_r (r + i) = i + 1.
        let expected = Array1::from_vec((0..l).map(|i| (i + 1) as f64).collect());
        assert_all_ranks_eq(&results, &expected);
        assert_all_ranks_eq(&results, &naive_reduce(&inputs, ReduceOp::Mean));
    }

    #[test]
    fn test_ring_all_reduce_max() {
        // N = 4, L = 7 -> uneven segmentation [2, 2, 2, 1].
        let n = 4;
        let l = 7;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec((0..l).map(|i| (i * 4 + r) as f64).collect()))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Max).unwrap();

        // Closed form: max_r (i*4 + r) = i*4 + 3.
        let expected = Array1::from_vec((0..l).map(|i| (i * 4 + 3) as f64).collect());
        assert_all_ranks_eq(&results, &expected);
        assert_all_ranks_eq(&results, &naive_reduce(&inputs, ReduceOp::Max));
    }

    #[test]
    fn test_ring_all_reduce_min() {
        let n = 4;
        let l = 7;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec((0..l).map(|i| (i * 4 + r) as f64).collect()))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Min).unwrap();

        // Closed form: min_r (i*4 + r) = i*4.
        let expected = Array1::from_vec((0..l).map(|i| (i * 4) as f64).collect());
        assert_all_ranks_eq(&results, &expected);
        assert_all_ranks_eq(&results, &naive_reduce(&inputs, ReduceOp::Min));
    }

    #[test]
    fn test_ring_all_reduce_product() {
        // N = 3, L = 5. Per-element product 2 * 3 * 0.5 = 3.
        let n = 3;
        let l = 5;
        let values = [2.0f64, 3.0, 0.5];
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec(vec![values[r]; l]))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Product).unwrap();

        let expected = Array1::from_vec(vec![3.0; l]);
        assert_all_ranks_eq(&results, &expected);
        assert_all_ranks_eq(&results, &naive_reduce(&inputs, ReduceOp::Product));
    }

    #[test]
    fn test_world_size_one_identity() {
        let ring = RingAllReduce::new(1).unwrap();
        let inputs = vec![Array1::from_vec(vec![1.0f64, 2.0, 3.0])];

        let sum = ring.all_reduce_all(&inputs, ReduceOp::Sum).unwrap();
        assert_eq!(sum.len(), 1);
        assert_all_ranks_eq(&sum, &inputs[0]);

        let mean = ring.all_reduce_all(&inputs, ReduceOp::Mean).unwrap();
        assert_all_ranks_eq(&mean, &inputs[0]);

        let gathered = ring.all_gather(&inputs).unwrap();
        assert_all_ranks_eq(&gathered, &inputs[0]);
    }

    #[test]
    fn test_length_not_divisible_by_world_size() {
        // N = 3, L = 7 -> uneven segmentation [3, 2, 2].
        let n = 3;
        let l = 7;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec((0..l).map(|i| (r * 100 + i) as f64).collect()))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Sum).unwrap();

        // Closed form: sum_r (r*100 + i) = 300 + 3*i.
        let expected = Array1::from_vec((0..l).map(|i| 300.0 + 3.0 * i as f64).collect());
        assert_all_ranks_eq(&results, &expected);
    }

    #[test]
    fn test_length_smaller_than_world_size() {
        // L < N forces zero-length segments; the collective must still be correct.
        let n = 4;
        let inputs: Vec<Array1<f64>> = (0..n)
            .map(|r| Array1::from_vec(vec![r as f64, (r + 1) as f64]))
            .collect();

        let ring = RingAllReduce::new(n).unwrap();
        let results = ring.all_reduce_all(&inputs, ReduceOp::Sum).unwrap();

        // sum_r [r, r+1] = [0+1+2+3, 1+2+3+4] = [6, 10].
        let expected = Array1::from_vec(vec![6.0, 10.0]);
        assert_all_ranks_eq(&results, &expected);
    }

    #[test]
    fn test_all_gather_round_trip() {
        // Contributions of different lengths concatenated in rank order.
        let inputs = vec![
            Array1::from_vec(vec![1.0f64, 2.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0]),
            Array1::from_vec(vec![6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0, 10.0]),
        ];

        let ring = RingAllReduce::new(4).unwrap();
        let results = ring.all_gather(&inputs).unwrap();

        let expected = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(results.len(), 4);
        assert_all_ranks_eq(&results, &expected);
    }

    #[test]
    fn test_invalid_inputs_are_rejected() {
        let ring = RingAllReduce::new(3).unwrap();

        // Empty input slice.
        let empty: Vec<Array1<f64>> = vec![];
        assert!(ring.all_reduce_all(&empty, ReduceOp::Sum).is_err());
        assert!(ring.all_gather(&empty).is_err());

        // Wrong number of contributions for the ring size.
        let wrong_count = vec![Array1::from_vec(vec![1.0f64]), Array1::from_vec(vec![2.0])];
        assert!(ring.all_reduce_all(&wrong_count, ReduceOp::Sum).is_err());
        assert!(ring.all_gather(&wrong_count).is_err());

        // Inconsistent lengths for all-reduce.
        let inconsistent = vec![
            Array1::from_vec(vec![1.0f64, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![5.0]),
        ];
        assert!(ring.all_reduce_all(&inconsistent, ReduceOp::Sum).is_err());

        // Zero-length contributions.
        let zero_len = vec![
            Array1::<f64>::from_vec(vec![]),
            Array1::from_vec(vec![]),
            Array1::from_vec(vec![]),
        ];
        assert!(ring.all_reduce_all(&zero_len, ReduceOp::Sum).is_err());
        assert!(ring.all_gather(&zero_len).is_err());
    }

    #[test]
    fn test_compute_segment_offsets() {
        assert_eq!(compute_segment_offsets(7, 3), vec![0, 3, 5, 7]);
        assert_eq!(compute_segment_offsets(10, 4), vec![0, 3, 6, 8, 10]);
        assert_eq!(compute_segment_offsets(6, 3), vec![0, 2, 4, 6]);
        // L < N produces trailing zero-length segments.
        assert_eq!(compute_segment_offsets(2, 4), vec![0, 1, 2, 2, 2]);

        // Adjacent segment lengths differ by at most one.
        let offsets = compute_segment_offsets(10, 4);
        let lengths: Vec<usize> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
        let max_len = *lengths.iter().max().unwrap();
        let min_len = *lengths.iter().min().unwrap();
        assert!(max_len - min_len <= 1);
        assert_eq!(lengths.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_local_transport_mailbox_semantics() {
        let mut transport = LocalTransport::<f64>::new(3).unwrap();
        assert_eq!(transport.world_size(), 3);

        // A send from rank 0 lands in rank 1's mailbox.
        transport.send_to_successor(0, vec![1.0, 2.0]).unwrap();
        // Rank 0's own mailbox is empty.
        assert!(transport.recv_from_predecessor(0).is_err());
        // Rank 1 collects the delivered segment, exactly once.
        assert_eq!(transport.recv_from_predecessor(1).unwrap(), vec![1.0, 2.0]);
        assert!(transport.recv_from_predecessor(1).is_err());

        // Posting twice without an intervening receive is rejected.
        transport.send_to_successor(0, vec![3.0]).unwrap();
        assert!(transport.send_to_successor(0, vec![4.0]).is_err());

        // Out-of-range ranks are rejected.
        assert!(transport.send_to_successor(3, vec![0.0]).is_err());
        assert!(transport.recv_from_predecessor(3).is_err());
    }

    #[test]
    fn test_reduce_op_identity_and_apply() {
        assert_eq!(ReduceOp::Sum.identity::<f64>(), 0.0);
        assert_eq!(ReduceOp::Product.identity::<f64>(), 1.0);
        assert_eq!(ReduceOp::Max.identity::<f64>(), f64::NEG_INFINITY);
        assert_eq!(ReduceOp::Min.identity::<f64>(), f64::INFINITY);

        assert_eq!(ReduceOp::Sum.apply(2.0, 3.0), 5.0);
        assert_eq!(ReduceOp::Product.apply(2.0, 3.0), 6.0);
        assert_eq!(ReduceOp::Max.apply(2.0, 3.0), 3.0);
        assert_eq!(ReduceOp::Min.apply(2.0, 3.0), 2.0);
        // Mean reuses the additive combine; division happens at finalisation.
        assert_eq!(ReduceOp::Mean.apply(2.0, 3.0), 5.0);
        assert!(ReduceOp::Mean.needs_mean_finalize());
        assert!(!ReduceOp::Sum.needs_mean_finalize());
    }
}
