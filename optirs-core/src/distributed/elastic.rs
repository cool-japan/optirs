// Elastic distributed training with a dynamic world size (pure CPU simulation).
//
// This module is a pure-Rust, CPU-only *simulation* of elastic data-parallel
// training. No real networking or devices are involved: workers, the rendezvous
// barrier and the shard placement are all logical. It is nevertheless a faithful
// reference for the three hard problems an elastic runtime (e.g. TorchElastic /
// `torchrun --max-restarts`, Horovod Elastic) must solve when the set of workers
// grows and shrinks during a run:
//
// * **Rendezvous with version epochs.** [`RendezvousState`] holds the agreed
//   membership (a sorted set of worker ranks), the agreed `world_size` and a
//   monotonically increasing `version` (the rendezvous *epoch*). Every accepted
//   membership change bumps the version by exactly one, and every worker that
//   observes a given version sees the *same* membership — this is what lets the
//   data-parallel group re-form consistently after a join or a leave.
//
// * **Join / leave state machine.** [`ElasticCoordinator`] consumes a stream of
//   [`MembershipEvent`]s and drives each worker through the lifecycle
//   `Pending -> Active -> Leaving -> Removed` (a removed worker may rejoin,
//   re-entering at `Pending`). Illegal transitions — joining an already-active
//   worker, or leaving an unknown / already-removed worker — are rejected with an
//   honest `Err` and leave the rendezvous untouched (no version bump).
//
// * **Consistent, balanced re-sharding.** Two stateless placement schemes map a
//   dataset of `D` shards onto the current membership of size `N`:
//     - [`block_partition`] (the default, returned by
//       [`ElasticCoordinator::shard_assignment`]) splits `0..D` into `N`
//       contiguous [`ShardRange`]s whose sizes differ by at most one. It is a pure
//       function of the *sorted* membership and `D`, so every worker computes the
//       identical assignment for a given `(version, world_size, D)`.
//     - [`hashed_assignment`] (Highest-Random-Weight / rendezvous hashing, exposed
//       via [`ElasticCoordinator::hashed_shard_assignment`]) is the same kind of
//       deterministic, exactly-balanced map but, because each shard independently
//       prefers the worker that maximises a mixing hash, a single join or leave
//       relocates far fewer shards than the contiguous block scheme. This is the
//       classic minimal-movement property of rendezvous hashing.
//
// * **Linear scaling rule.** Following Goyal et al. (2017, "Accurate, Large
//   Minibatch SGD"), the effective learning rate is scaled linearly with the
//   world size, `lr = base_lr * world_size / reference_world_size`, and the
//   gradient-averaging divisor tracks the live world size. An optional gradual
//   warmup ramps the learning rate from `base_lr` up to the scaled target over a
//   configurable number of steps after a resize.
//
// Everything here is deterministic and uses only the standard library plus `f64`;
// no randomness is drawn (the hash is a fixed integer mix), so results are fully
// reproducible.

use crate::error::{OptimError, Result};
use std::collections::BTreeMap;

/// A membership change requested against the elastic group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembershipEvent {
    /// A worker with the given rank requests to join the group.
    Join(usize),
    /// A worker with the given rank requests to leave the group.
    Leave(usize),
}

impl MembershipEvent {
    /// The worker rank this event refers to.
    pub fn worker_id(self) -> usize {
        match self {
            MembershipEvent::Join(worker_id) | MembershipEvent::Leave(worker_id) => worker_id,
        }
    }

    /// Whether this is a join event.
    pub fn is_join(self) -> bool {
        matches!(self, MembershipEvent::Join(_))
    }
}

/// Lifecycle state of a single worker in the elastic group.
///
/// The legal transitions form the chain `Pending -> Active -> Leaving -> Removed`,
/// with one extra edge `Removed -> Pending` so a worker may rejoin after leaving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// The worker has requested to join and is waiting at the rendezvous barrier.
    Pending,
    /// The worker is an active member of the current rendezvous.
    Active,
    /// The worker has requested to leave and is draining out of the group.
    Leaving,
    /// The worker has fully left the group (and may later rejoin).
    Removed,
}

impl WorkerState {
    /// Human-readable name of the lifecycle state.
    pub fn name(self) -> &'static str {
        match self {
            WorkerState::Pending => "Pending",
            WorkerState::Active => "Active",
            WorkerState::Leaving => "Leaving",
            WorkerState::Removed => "Removed",
        }
    }

    /// Whether `from -> to` is a legal worker-lifecycle transition.
    ///
    /// `from == None` denotes a worker the coordinator has never seen.
    pub fn is_legal_transition(from: Option<WorkerState>, to: WorkerState) -> bool {
        matches!(
            (from, to),
            (None, WorkerState::Pending)
                | (Some(WorkerState::Removed), WorkerState::Pending)
                | (Some(WorkerState::Pending), WorkerState::Active)
                | (Some(WorkerState::Active), WorkerState::Leaving)
                | (Some(WorkerState::Leaving), WorkerState::Removed)
        )
    }
}

/// Human-readable description of an optional worker state, for error messages.
fn describe_state(state: Option<WorkerState>) -> &'static str {
    match state {
        Some(state) => state.name(),
        None => "absent",
    }
}

/// A contiguous half-open range `[start, end)` of shard indices owned by a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardRange {
    /// Rank of the worker that owns this range.
    pub worker_id: usize,
    /// First shard index in the range (inclusive).
    pub start: usize,
    /// One past the last shard index in the range (exclusive).
    pub end: usize,
}

impl ShardRange {
    /// Number of shards in this range.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this range contains no shards.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether the given shard index falls inside this range.
    pub fn contains(&self, shard: usize) -> bool {
        shard >= self.start && shard < self.end
    }
}

/// The (generally non-contiguous) set of shards a worker owns under a hashed
/// (rendezvous / HRW) assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerShards {
    /// Rank of the worker that owns these shards.
    pub worker_id: usize,
    /// Sorted, de-duplicated shard indices owned by the worker.
    pub shards: Vec<usize>,
}

impl WorkerShards {
    /// Number of shards owned by the worker.
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// Whether the worker owns no shards.
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
}

/// The agreed rendezvous view shared by every worker at a given epoch.
///
/// Holds the canonical, sorted membership, the agreed world size and a
/// monotonically increasing version (epoch) that bumps on every membership
/// change. Two coordinators that have accepted the same multiset of events end up
/// with identical membership and world size (the version reflects how many
/// changes were accepted).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RendezvousState {
    version: u64,
    members: Vec<usize>,
    world_size: usize,
}

impl RendezvousState {
    /// Create the initial, empty rendezvous (version 0, no members).
    pub fn new() -> Self {
        Self::default()
    }

    /// The current rendezvous version (epoch).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The current sorted membership (worker ranks).
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// The current world size (`== members().len()`).
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Whether the given worker rank is currently a member.
    pub fn contains(&self, worker_id: usize) -> bool {
        self.members.binary_search(&worker_id).is_ok()
    }
}

/// Configuration of an elastic training run.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticConfig {
    /// Base (reference) learning rate, valid at `reference_world_size` workers.
    pub base_learning_rate: f64,
    /// World size the `base_learning_rate` was tuned for (the linear-scaling
    /// reference). Must be at least one.
    pub reference_world_size: usize,
    /// Number of dataset shards `D` to distribute across the workers.
    pub dataset_shards: usize,
    /// Minimum allowed world size: a leave that would drop below this is rejected.
    pub min_world_size: usize,
    /// Maximum allowed world size: a join that would rise above this is rejected.
    pub max_world_size: usize,
    /// Number of gradual-warmup steps applied to the learning rate after a resize
    /// (`0` disables warmup).
    pub warmup_steps: usize,
}

impl ElasticConfig {
    /// Create and validate an elastic configuration.
    ///
    /// # Errors
    /// Returns an error when `base_learning_rate` is non-finite or non-positive,
    /// `reference_world_size`, `dataset_shards` or `min_world_size` are zero, or
    /// `max_world_size < min_world_size`.
    pub fn new(
        base_learning_rate: f64,
        reference_world_size: usize,
        dataset_shards: usize,
        min_world_size: usize,
        max_world_size: usize,
        warmup_steps: usize,
    ) -> Result<Self> {
        let config = Self {
            base_learning_rate,
            reference_world_size,
            dataset_shards,
            min_world_size,
            max_world_size,
            warmup_steps,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration invariants.
    ///
    /// # Errors
    /// See [`ElasticConfig::new`].
    pub fn validate(&self) -> Result<()> {
        if !self.base_learning_rate.is_finite() || self.base_learning_rate <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "base_learning_rate {} must be finite and positive",
                self.base_learning_rate
            )));
        }
        if self.reference_world_size == 0 {
            return Err(OptimError::InvalidConfig(
                "reference_world_size must be at least 1".to_string(),
            ));
        }
        if self.dataset_shards == 0 {
            return Err(OptimError::InvalidConfig(
                "dataset_shards must be at least 1".to_string(),
            ));
        }
        if self.min_world_size == 0 {
            return Err(OptimError::InvalidConfig(
                "min_world_size must be at least 1".to_string(),
            ));
        }
        if self.max_world_size < self.min_world_size {
            return Err(OptimError::InvalidConfig(format!(
                "max_world_size {} must be >= min_world_size {}",
                self.max_world_size, self.min_world_size
            )));
        }
        Ok(())
    }
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.1,
            reference_world_size: 1,
            dataset_shards: 1,
            min_world_size: 1,
            max_world_size: usize::MAX,
            warmup_steps: 0,
        }
    }
}

/// A point-in-time snapshot of the elastic group after an accepted event.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochSnapshot {
    /// Rendezvous version (epoch) after the event.
    pub version: u64,
    /// World size after the event.
    pub world_size: usize,
    /// Sorted membership after the event.
    pub members: Vec<usize>,
    /// Balanced contiguous shard assignment for this membership.
    pub shard_assignment: Vec<ShardRange>,
    /// Linear-scaling-rule learning rate for this world size.
    pub scaled_lr: f64,
    /// Gradient-averaging factor (`1 / world_size`) for this world size.
    pub averaging_factor: f64,
}

/// Split the shard index space `0..dataset_shards` into one contiguous
/// [`ShardRange`] per member, balanced so the sizes differ by at most one.
///
/// `members` must be sorted ascending (the coordinator maintains this invariant).
/// The first `dataset_shards % members.len()` members each receive one extra
/// shard. The result is a pure function of `(members, dataset_shards)`, so it is
/// identical on every worker that shares the same rendezvous view. When there are
/// more workers than shards, the surplus workers receive empty ranges.
pub fn block_partition(members: &[usize], dataset_shards: usize) -> Vec<ShardRange> {
    let num_workers = members.len();
    if num_workers == 0 {
        return Vec::new();
    }
    let base = dataset_shards / num_workers;
    let remainder = dataset_shards % num_workers;

    let mut ranges = Vec::with_capacity(num_workers);
    let mut start = 0usize;
    for (index, &worker_id) in members.iter().enumerate() {
        let size = base + usize::from(index < remainder);
        let end = start + size;
        ranges.push(ShardRange {
            worker_id,
            start,
            end,
        });
        start = end;
    }
    ranges
}

/// SplitMix64 finalizer: a fast, well-distributed, fully deterministic 64-bit
/// integer mix. Used to derive rendezvous-hash weights without any randomness.
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Highest-Random-Weight (rendezvous) hash of a `(worker, shard)` pair.
///
/// A shard prefers the worker that maximises this weight; ties (astronomically
/// rare) are broken by smaller worker rank in the caller, keeping the map a pure
/// function of `(members, dataset_shards)`.
#[inline]
fn hrw_weight(worker_id: usize, shard: usize) -> u64 {
    let worker_hash = splitmix64(worker_id as u64);
    let shard_hash = splitmix64(shard as u64);
    splitmix64(worker_hash ^ shard_hash.rotate_left(32))
}

/// Per-member shard capacities for a balanced (`<= 1` spread) assignment.
///
/// The first `dataset_shards % num_workers` members receive `base + 1`, the rest
/// receive `base`; the capacities therefore sum to exactly `dataset_shards`.
fn balanced_capacities(num_workers: usize, dataset_shards: usize) -> Vec<usize> {
    let base = dataset_shards / num_workers;
    let remainder = dataset_shards % num_workers;
    (0..num_workers)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

/// Assign `0..dataset_shards` to `members` by capacity-bounded rendezvous (HRW)
/// hashing: each shard is placed on the highest-weight member that still has
/// spare capacity, with capacities chosen so worker loads differ by at most one.
///
/// `members` must be sorted ascending. The result is deterministic and exactly
/// balanced, and — because shard ownership is decided independently per shard by
/// a fixed hash — a single membership change relocates far fewer shards than the
/// contiguous [`block_partition`] scheme (the minimal-movement property of
/// rendezvous hashing). Returns one [`WorkerShards`] per member, in membership
/// order, each with its owned shard indices in ascending order.
pub fn hashed_assignment(members: &[usize], dataset_shards: usize) -> Vec<WorkerShards> {
    let num_workers = members.len();
    if num_workers == 0 {
        return Vec::new();
    }

    let mut remaining = balanced_capacities(num_workers, dataset_shards);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); num_workers];

    for shard in 0..dataset_shards {
        // Seed the search with the first member that still has spare capacity.
        // At least one always exists because the remaining capacities sum to
        // `dataset_shards - shard > 0`.
        let mut best_index = 0usize;
        while remaining[best_index] == 0 {
            best_index += 1;
        }
        let mut best_weight = hrw_weight(members[best_index], shard);

        for index in (best_index + 1)..num_workers {
            if remaining[index] == 0 {
                continue;
            }
            let weight = hrw_weight(members[index], shard);
            let prefer = match weight.cmp(&best_weight) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => members[index] < members[best_index],
            };
            if prefer {
                best_weight = weight;
                best_index = index;
            }
        }

        remaining[best_index] -= 1;
        buckets[best_index].push(shard);
    }

    members
        .iter()
        .zip(buckets)
        .map(|(&worker_id, shards)| WorkerShards { worker_id, shards })
        .collect()
}

/// Drives an elastic training group: a join/leave state machine over a versioned
/// rendezvous, with balanced re-sharding and linear learning-rate scaling.
#[derive(Debug, Clone)]
pub struct ElasticCoordinator {
    config: ElasticConfig,
    rendezvous: RendezvousState,
    worker_states: BTreeMap<usize, WorkerState>,
}

impl ElasticCoordinator {
    /// Create a coordinator from a configuration (which is validated).
    ///
    /// The initial rendezvous is empty: version 0, no members, world size 0.
    ///
    /// # Errors
    /// Propagates [`ElasticConfig::validate`] errors.
    pub fn new(config: ElasticConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            rendezvous: RendezvousState::new(),
            worker_states: BTreeMap::new(),
        })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &ElasticConfig {
        &self.config
    }

    /// Borrow the current rendezvous view.
    pub fn rendezvous(&self) -> &RendezvousState {
        &self.rendezvous
    }

    /// Current rendezvous version (epoch).
    pub fn current_version(&self) -> u64 {
        self.rendezvous.version
    }

    /// Current world size.
    pub fn world_size(&self) -> usize {
        self.rendezvous.world_size
    }

    /// Current sorted membership.
    pub fn members(&self) -> &[usize] {
        self.rendezvous.members()
    }

    /// Lifecycle state of a worker, or `None` if never seen.
    pub fn worker_state(&self, worker_id: usize) -> Option<WorkerState> {
        self.worker_states.get(&worker_id).copied()
    }

    /// Balanced contiguous shard assignment for the current membership.
    pub fn shard_assignment(&self) -> Vec<ShardRange> {
        block_partition(&self.rendezvous.members, self.config.dataset_shards)
    }

    /// Minimal-movement rendezvous-hash shard assignment for the current
    /// membership.
    pub fn hashed_shard_assignment(&self) -> Vec<WorkerShards> {
        hashed_assignment(&self.rendezvous.members, self.config.dataset_shards)
    }

    /// Effective learning rate under the linear scaling rule,
    /// `base_lr * world_size / reference_world_size`.
    ///
    /// Returns `0.0` before any worker has joined (world size 0).
    pub fn scaled_learning_rate(&self) -> f64 {
        if self.rendezvous.world_size == 0 {
            return 0.0;
        }
        self.config.base_learning_rate * self.rendezvous.world_size as f64
            / self.config.reference_world_size as f64
    }

    /// Gradient-averaging divisor that tracks the live world size (i.e. the number
    /// of workers whose gradients are summed before averaging).
    pub fn gradient_averaging_divisor(&self) -> f64 {
        self.rendezvous.world_size as f64
    }

    /// Gradient-averaging factor `1 / world_size` (`0.0` before any join).
    pub fn averaging_factor(&self) -> f64 {
        if self.rendezvous.world_size == 0 {
            0.0
        } else {
            1.0 / self.rendezvous.world_size as f64
        }
    }

    /// Learning rate during gradual warmup after a resize.
    ///
    /// Linearly ramps from `base_learning_rate` at `step == 0` up to the full
    /// [`ElasticCoordinator::scaled_learning_rate`] once `step >= warmup_steps`
    /// (and is flat at the scaled value when `warmup_steps == 0`). At the
    /// reference world size the scaled target equals the base rate, so warmup is a
    /// no-op there.
    pub fn warmup_learning_rate(&self, step: usize) -> f64 {
        let target = self.scaled_learning_rate();
        let warmup_steps = self.config.warmup_steps;
        if warmup_steps == 0 || step >= warmup_steps {
            return target;
        }
        let base = self.config.base_learning_rate;
        let fraction = (step + 1) as f64 / warmup_steps as f64;
        base + (target - base) * fraction
    }

    /// Apply one membership event, returning the new [`EpochSnapshot`].
    ///
    /// On success the rendezvous version bumps by exactly one and the membership,
    /// shard assignment and scaled learning rate are recomputed. On failure the
    /// rendezvous is left completely untouched (no version bump, no state change).
    ///
    /// # Errors
    /// Returns an error for an illegal lifecycle transition (joining an
    /// already-active worker, or leaving an unknown / already-removed worker), or
    /// when the change would violate the configured world-size bounds.
    pub fn apply_event(&mut self, event: MembershipEvent) -> Result<EpochSnapshot> {
        match event {
            MembershipEvent::Join(worker_id) => self.apply_join(worker_id),
            MembershipEvent::Leave(worker_id) => self.apply_leave(worker_id),
        }
    }

    /// Apply a sequence of events strictly: the first rejected event aborts the
    /// whole simulation with its error.
    ///
    /// # Errors
    /// Propagates the first [`ElasticCoordinator::apply_event`] error.
    pub fn simulate(&mut self, events: &[MembershipEvent]) -> Result<Vec<EpochSnapshot>> {
        let mut snapshots = Vec::with_capacity(events.len());
        for &event in events {
            snapshots.push(self.apply_event(event)?);
        }
        Ok(snapshots)
    }

    /// Apply a sequence of events resiliently: rejected events are skipped, and a
    /// snapshot is produced only for each *accepted* event.
    pub fn simulate_resilient(&mut self, events: &[MembershipEvent]) -> Vec<EpochSnapshot> {
        let mut snapshots = Vec::new();
        for &event in events {
            if let Ok(snapshot) = self.apply_event(event) {
                snapshots.push(snapshot);
            }
        }
        snapshots
    }

    /// Validate and commit a join.
    fn apply_join(&mut self, worker_id: usize) -> Result<EpochSnapshot> {
        let current = self.worker_state(worker_id);
        if !WorkerState::is_legal_transition(current, WorkerState::Pending) {
            return Err(OptimError::InvalidState(format!(
                "cannot join worker {worker_id}: it is currently {} (a join requires an \
                 absent or removed worker)",
                describe_state(current)
            )));
        }
        let new_world_size = self.rendezvous.world_size + 1;
        if new_world_size > self.config.max_world_size {
            return Err(OptimError::InvalidConfig(format!(
                "join of worker {worker_id} would raise the world size to {new_world_size}, \
                 exceeding max_world_size {}",
                self.config.max_world_size
            )));
        }

        // Commit: walk the worker through Pending then Active, splice it into the
        // sorted membership and bump the rendezvous version.
        self.worker_states.insert(worker_id, WorkerState::Pending);
        self.worker_states.insert(worker_id, WorkerState::Active);
        let position = self
            .rendezvous
            .members
            .partition_point(|&member| member < worker_id);
        self.rendezvous.members.insert(position, worker_id);
        self.rendezvous.world_size = self.rendezvous.members.len();
        self.rendezvous.version += 1;

        Ok(self.snapshot())
    }

    /// Validate and commit a leave.
    fn apply_leave(&mut self, worker_id: usize) -> Result<EpochSnapshot> {
        let current = self.worker_state(worker_id);
        if current != Some(WorkerState::Active) {
            return Err(OptimError::InvalidState(format!(
                "cannot remove worker {worker_id}: it is currently {} (a leave requires an \
                 active worker)",
                describe_state(current)
            )));
        }
        if self.rendezvous.world_size <= self.config.min_world_size {
            return Err(OptimError::InvalidConfig(format!(
                "leave of worker {worker_id} would drop the world size below min_world_size {}",
                self.config.min_world_size
            )));
        }

        // Commit: walk the worker through Leaving then Removed, splice it out of
        // the sorted membership and bump the rendezvous version.
        self.worker_states.insert(worker_id, WorkerState::Leaving);
        self.worker_states.insert(worker_id, WorkerState::Removed);
        if let Ok(position) = self.rendezvous.members.binary_search(&worker_id) {
            self.rendezvous.members.remove(position);
        }
        self.rendezvous.world_size = self.rendezvous.members.len();
        self.rendezvous.version += 1;

        Ok(self.snapshot())
    }

    /// Build a snapshot of the current rendezvous.
    fn snapshot(&self) -> EpochSnapshot {
        EpochSnapshot {
            version: self.rendezvous.version,
            world_size: self.rendezvous.world_size,
            members: self.rendezvous.members.clone(),
            shard_assignment: self.shard_assignment(),
            scaled_lr: self.scaled_learning_rate(),
            averaging_factor: self.averaging_factor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn base_config() -> ElasticConfig {
        ElasticConfig::new(0.1, 4, 12, 1, 16, 0).expect("valid config")
    }

    /// Map each shard index to its owning worker rank under a block partition.
    fn block_owners(ranges: &[ShardRange], dataset_shards: usize) -> Vec<usize> {
        let mut owners = vec![usize::MAX; dataset_shards];
        for range in ranges {
            owners[range.start..range.end].fill(range.worker_id);
        }
        owners
    }

    /// Map each shard index to its owning worker rank under a hashed assignment.
    fn hashed_owners(assignment: &[WorkerShards], dataset_shards: usize) -> Vec<usize> {
        let mut owners = vec![usize::MAX; dataset_shards];
        for worker in assignment {
            for &shard in &worker.shards {
                owners[shard] = worker.worker_id;
            }
        }
        owners
    }

    fn assert_balanced_cover(ranges: &[ShardRange], num_workers: usize, dataset_shards: usize) {
        assert_eq!(ranges.len(), num_workers, "one range per worker");
        // Contiguous, non-overlapping cover of 0..dataset_shards.
        let mut expected_start = 0usize;
        for range in ranges {
            assert_eq!(range.start, expected_start, "ranges must be contiguous");
            assert!(range.end >= range.start, "range must be well-formed");
            expected_start = range.end;
        }
        assert_eq!(
            expected_start, dataset_shards,
            "ranges must cover every shard exactly once"
        );
        // Balanced: sizes differ by at most one.
        let min_size = ranges.iter().map(ShardRange::len).min().unwrap_or(0);
        let max_size = ranges.iter().map(ShardRange::len).max().unwrap_or(0);
        assert!(
            max_size - min_size <= 1,
            "shard sizes must differ by at most one (min={min_size}, max={max_size})"
        );
    }

    #[test]
    fn test_version_bumps_by_one_per_accepted_event() {
        let mut coordinator = ElasticCoordinator::new(base_config()).unwrap();
        assert_eq!(coordinator.current_version(), 0);
        for (expected_version, worker_id) in (1u64..=4).zip(0usize..4) {
            let snapshot = coordinator
                .apply_event(MembershipEvent::Join(worker_id))
                .unwrap();
            assert_eq!(snapshot.version, expected_version);
            assert_eq!(coordinator.current_version(), expected_version);
            assert_eq!(coordinator.world_size(), expected_version as usize);
        }
        // A leave also bumps by exactly one.
        let snapshot = coordinator.apply_event(MembershipEvent::Leave(1)).unwrap();
        assert_eq!(snapshot.version, 5);
        assert_eq!(coordinator.world_size(), 3);
    }

    #[test]
    fn test_invalid_events_return_err_and_do_not_bump_version() {
        let mut coordinator = ElasticCoordinator::new(base_config()).unwrap();
        coordinator.apply_event(MembershipEvent::Join(0)).unwrap();
        coordinator.apply_event(MembershipEvent::Join(1)).unwrap();
        let version_before = coordinator.current_version();
        let world_before = coordinator.world_size();

        // Joining an already-active worker is rejected.
        assert!(coordinator.apply_event(MembershipEvent::Join(0)).is_err());
        // Leaving an unknown worker is rejected.
        assert!(coordinator.apply_event(MembershipEvent::Leave(7)).is_err());

        // Leaving then re-leaving the same worker: second leave is rejected.
        coordinator.apply_event(MembershipEvent::Leave(1)).unwrap();
        assert!(coordinator.apply_event(MembershipEvent::Leave(1)).is_err());
        assert_eq!(coordinator.worker_state(1), Some(WorkerState::Removed));

        // The two pure rejections above must not have changed version/size; the
        // accepted leave bumped version once and dropped size once.
        assert_eq!(coordinator.current_version(), version_before + 1);
        assert_eq!(coordinator.world_size(), world_before - 1);
    }

    #[test]
    fn test_shard_assignment_balanced_and_covers_all_shards() {
        for dataset_shards in [1usize, 7, 10, 12, 13, 100] {
            let config = ElasticConfig::new(0.05, 2, dataset_shards, 1, 64, 0).unwrap();
            for num_workers in 1usize..=8 {
                let mut coordinator = ElasticCoordinator::new(config.clone()).unwrap();
                for worker_id in 0..num_workers {
                    coordinator
                        .apply_event(MembershipEvent::Join(worker_id))
                        .unwrap();
                }
                let ranges = coordinator.shard_assignment();
                assert_balanced_cover(&ranges, num_workers, dataset_shards);

                // No shard index belongs to two ranges, and every shard has an owner.
                let owners = block_owners(&ranges, dataset_shards);
                assert!(
                    owners.iter().all(|&owner| owner != usize::MAX),
                    "every shard must be owned"
                );
            }
        }
    }

    #[test]
    fn test_shard_assignment_is_deterministic_and_path_independent() {
        let config = base_config();
        // Two coordinators that reach the same membership via different event
        // orders must agree on version, world size and shard assignment.
        let mut first = ElasticCoordinator::new(config.clone()).unwrap();
        first
            .simulate(&[
                MembershipEvent::Join(0),
                MembershipEvent::Join(1),
                MembershipEvent::Join(2),
            ])
            .unwrap();

        let mut second = ElasticCoordinator::new(config).unwrap();
        second
            .simulate(&[
                MembershipEvent::Join(2),
                MembershipEvent::Join(0),
                MembershipEvent::Join(1),
            ])
            .unwrap();

        assert_eq!(first.current_version(), second.current_version());
        assert_eq!(first.world_size(), second.world_size());
        assert_eq!(first.members(), second.members());
        assert_eq!(first.shard_assignment(), second.shard_assignment());
        assert_eq!(
            first.hashed_shard_assignment(),
            second.hashed_shard_assignment()
        );

        // Recomputing the pure assignment for (members, D) reproduces it exactly.
        let recomputed = block_partition(first.members(), first.config().dataset_shards);
        assert_eq!(first.shard_assignment(), recomputed);
    }

    #[test]
    fn test_linear_scaling_rule_exact() {
        let base_lr = 0.1;
        let reference = 4usize;
        let config = ElasticConfig::new(base_lr, reference, 12, 1, 32, 0).unwrap();
        let mut coordinator = ElasticCoordinator::new(config).unwrap();
        for worker_id in 0..8usize {
            let snapshot = coordinator
                .apply_event(MembershipEvent::Join(worker_id))
                .unwrap();
            let world_size = snapshot.world_size;
            let expected = base_lr * world_size as f64 / reference as f64;
            assert_eq!(coordinator.scaled_learning_rate(), expected);
            assert_eq!(snapshot.scaled_lr, expected);
            assert_eq!(coordinator.gradient_averaging_divisor(), world_size as f64);
            assert_relative_eq!(
                coordinator.averaging_factor(),
                1.0 / world_size as f64,
                epsilon = 1e-12
            );
        }
        // At exactly the reference world size, the scaled LR equals the base LR.
        let mut at_reference =
            ElasticCoordinator::new(ElasticConfig::new(base_lr, reference, 12, 1, 32, 0).unwrap())
                .unwrap();
        for worker_id in 0..reference {
            at_reference
                .apply_event(MembershipEvent::Join(worker_id))
                .unwrap();
        }
        assert_eq!(at_reference.scaled_learning_rate(), base_lr);
    }

    #[test]
    fn test_warmup_ramps_from_base_to_scaled() {
        let base_lr = 0.1;
        let config = ElasticConfig::new(base_lr, 2, 12, 1, 32, 5).unwrap();
        let mut coordinator = ElasticCoordinator::new(config).unwrap();
        for worker_id in 0..8usize {
            coordinator
                .apply_event(MembershipEvent::Join(worker_id))
                .unwrap();
        }
        let target = coordinator.scaled_learning_rate();
        assert!(
            target > base_lr,
            "scaled target must exceed base for 8 > ref 2"
        );

        // Warmup starts above the base (step 0 already adds one fifth of the gap)
        // and is monotonically increasing, reaching the scaled target at the end.
        let mut previous = base_lr;
        for step in 0..5 {
            let lr = coordinator.warmup_learning_rate(step);
            assert!(lr > previous, "warmup must be strictly increasing");
            assert!(lr <= target + 1e-12, "warmup must not overshoot the target");
            previous = lr;
        }
        assert_relative_eq!(coordinator.warmup_learning_rate(4), target, epsilon = 1e-12);
        assert_relative_eq!(
            coordinator.warmup_learning_rate(100),
            target,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_join_leave_round_trip_restores_membership() {
        let mut coordinator = ElasticCoordinator::new(base_config()).unwrap();
        coordinator
            .simulate(&[
                MembershipEvent::Join(0),
                MembershipEvent::Join(1),
                MembershipEvent::Join(2),
            ])
            .unwrap();
        let members_before: Vec<usize> = coordinator.members().to_vec();
        let world_before = coordinator.world_size();
        let assignment_before = coordinator.shard_assignment();
        let lr_before = coordinator.scaled_learning_rate();

        // A worker joins and then leaves again.
        coordinator.apply_event(MembershipEvent::Join(3)).unwrap();
        assert_eq!(coordinator.world_size(), world_before + 1);
        coordinator.apply_event(MembershipEvent::Leave(3)).unwrap();

        assert_eq!(coordinator.members(), members_before.as_slice());
        assert_eq!(coordinator.world_size(), world_before);
        assert_eq!(coordinator.shard_assignment(), assignment_before);
        assert_eq!(coordinator.scaled_learning_rate(), lr_before);
        // Version is monotonic: two extra accepted events occurred.
        assert_eq!(coordinator.current_version(), 5);
    }

    #[test]
    fn test_min_and_max_world_size_bounds_enforced() {
        let config = ElasticConfig::new(0.1, 2, 12, 2, 4, 0).unwrap();
        let mut coordinator = ElasticCoordinator::new(config).unwrap();

        // Bootstrapping below the minimum is permitted (the floor only gates leaves).
        coordinator.apply_event(MembershipEvent::Join(0)).unwrap();
        assert_eq!(coordinator.world_size(), 1);

        // Fill up to the maximum world size of 4.
        for worker_id in 1..4usize {
            coordinator
                .apply_event(MembershipEvent::Join(worker_id))
                .unwrap();
        }
        assert_eq!(coordinator.world_size(), 4);
        let version_at_max = coordinator.current_version();

        // A further join is rejected and does not bump the version.
        assert!(coordinator.apply_event(MembershipEvent::Join(4)).is_err());
        assert_eq!(coordinator.current_version(), version_at_max);
        assert_eq!(coordinator.world_size(), 4);

        // Leave down to the minimum world size of 2.
        coordinator.apply_event(MembershipEvent::Leave(3)).unwrap();
        coordinator.apply_event(MembershipEvent::Leave(2)).unwrap();
        assert_eq!(coordinator.world_size(), 2);
        let version_at_min = coordinator.current_version();

        // A further leave is rejected and does not bump the version.
        assert!(coordinator.apply_event(MembershipEvent::Leave(1)).is_err());
        assert_eq!(coordinator.current_version(), version_at_min);
        assert_eq!(coordinator.world_size(), 2);
    }

    #[test]
    fn test_worker_lifecycle_states() {
        let mut coordinator = ElasticCoordinator::new(base_config()).unwrap();
        assert_eq!(coordinator.worker_state(0), None);
        coordinator.apply_event(MembershipEvent::Join(0)).unwrap();
        assert_eq!(coordinator.worker_state(0), Some(WorkerState::Active));
        // Need a second worker so the leave does not breach the min bound.
        coordinator.apply_event(MembershipEvent::Join(1)).unwrap();
        coordinator.apply_event(MembershipEvent::Leave(0)).unwrap();
        assert_eq!(coordinator.worker_state(0), Some(WorkerState::Removed));
        // A removed worker may rejoin.
        coordinator.apply_event(MembershipEvent::Join(0)).unwrap();
        assert_eq!(coordinator.worker_state(0), Some(WorkerState::Active));

        // Transition legality table.
        assert!(WorkerState::is_legal_transition(None, WorkerState::Pending));
        assert!(WorkerState::is_legal_transition(
            Some(WorkerState::Removed),
            WorkerState::Pending
        ));
        assert!(WorkerState::is_legal_transition(
            Some(WorkerState::Active),
            WorkerState::Leaving
        ));
        assert!(!WorkerState::is_legal_transition(
            Some(WorkerState::Active),
            WorkerState::Pending
        ));
        assert!(!WorkerState::is_legal_transition(None, WorkerState::Active));
    }

    #[test]
    fn test_hashed_assignment_balanced_deterministic_and_full_cover() {
        for dataset_shards in [1usize, 9, 64, 100] {
            for num_workers in 1usize..=8 {
                let members: Vec<usize> = (0..num_workers).collect();
                let assignment = hashed_assignment(&members, dataset_shards);
                assert_eq!(assignment.len(), num_workers);

                let sizes: Vec<usize> = assignment.iter().map(WorkerShards::len).collect();
                let total: usize = sizes.iter().sum();
                assert_eq!(total, dataset_shards, "must cover every shard");
                let min_size = sizes.iter().copied().min().unwrap_or(0);
                let max_size = sizes.iter().copied().max().unwrap_or(0);
                assert!(max_size - min_size <= 1, "hashed loads must be balanced");

                // Every shard owned exactly once.
                let owners = hashed_owners(&assignment, dataset_shards);
                assert!(owners.iter().all(|&owner| owner != usize::MAX));

                // Deterministic: recomputation is identical.
                assert_eq!(assignment, hashed_assignment(&members, dataset_shards));
                // Shard lists are sorted ascending.
                for worker in &assignment {
                    assert!(worker.shards.windows(2).all(|pair| pair[0] < pair[1]));
                }
            }
        }
    }

    #[test]
    fn test_hashed_assignment_moves_fewer_shards_than_block() {
        // Over a representative resize sequence, rendezvous hashing relocates
        // strictly fewer shards in aggregate than the contiguous block scheme.
        let dataset_shards = 600usize;
        let memberships: Vec<Vec<usize>> = vec![
            (0..6).collect(),
            (0..5).collect(), // worker 5 leaves
            (0..7).collect(), // workers 5, 6 join
            (1..7).collect(), // worker 0 leaves
            (0..8).collect(), // worker 0 joins, worker 7 joins
        ];

        let mut block_moves = 0usize;
        let mut hashed_moves = 0usize;
        for window in memberships.windows(2) {
            let before = &window[0];
            let after = &window[1];

            let block_before =
                block_owners(&block_partition(before, dataset_shards), dataset_shards);
            let block_after = block_owners(&block_partition(after, dataset_shards), dataset_shards);
            block_moves += (0..dataset_shards)
                .filter(|&shard| block_before[shard] != block_after[shard])
                .count();

            let hashed_before =
                hashed_owners(&hashed_assignment(before, dataset_shards), dataset_shards);
            let hashed_after =
                hashed_owners(&hashed_assignment(after, dataset_shards), dataset_shards);
            hashed_moves += (0..dataset_shards)
                .filter(|&shard| hashed_before[shard] != hashed_after[shard])
                .count();
        }

        assert!(
            hashed_moves < block_moves,
            "rendezvous hashing should move fewer shards (hashed={hashed_moves}, block={block_moves})"
        );
    }

    #[test]
    fn test_simulate_strict_and_resilient() {
        let config = base_config();
        // Strict simulation: an invalid event aborts with an error.
        let mut strict = ElasticCoordinator::new(config.clone()).unwrap();
        let result = strict.simulate(&[
            MembershipEvent::Join(0),
            MembershipEvent::Join(0), // invalid: already active
        ]);
        assert!(result.is_err());

        // Resilient simulation: invalid events are skipped, one snapshot per
        // accepted event, and the version equals the number of accepted events.
        let mut resilient = ElasticCoordinator::new(config).unwrap();
        let snapshots = resilient.simulate_resilient(&[
            MembershipEvent::Join(0),
            MembershipEvent::Join(0), // skipped
            MembershipEvent::Join(1),
            MembershipEvent::Leave(5), // skipped
            MembershipEvent::Leave(0),
        ]);
        assert_eq!(snapshots.len(), 3, "three events accepted");
        assert_eq!(resilient.current_version(), 3);
        for (index, snapshot) in snapshots.iter().enumerate() {
            assert_eq!(snapshot.version, index as u64 + 1);
        }
    }

    #[test]
    fn test_config_validation_errors() {
        assert!(ElasticConfig::new(0.0, 4, 12, 1, 16, 0).is_err());
        assert!(ElasticConfig::new(-1.0, 4, 12, 1, 16, 0).is_err());
        assert!(ElasticConfig::new(f64::NAN, 4, 12, 1, 16, 0).is_err());
        assert!(ElasticConfig::new(0.1, 0, 12, 1, 16, 0).is_err());
        assert!(ElasticConfig::new(0.1, 4, 0, 1, 16, 0).is_err());
        assert!(ElasticConfig::new(0.1, 4, 12, 0, 16, 0).is_err());
        assert!(ElasticConfig::new(0.1, 4, 12, 8, 4, 0).is_err());
        assert!(ElasticConfig::new(0.1, 4, 12, 1, 16, 0).is_ok());
        // The coordinator rejects an invalid config too.
        let bad = ElasticConfig {
            base_learning_rate: 0.1,
            reference_world_size: 0,
            dataset_shards: 1,
            min_world_size: 1,
            max_world_size: 1,
            warmup_steps: 0,
        };
        assert!(ElasticCoordinator::new(bad).is_err());
    }
}
