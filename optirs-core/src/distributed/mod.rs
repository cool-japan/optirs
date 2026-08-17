// Distributed optimization support
//
// This module provides support for distributed training including parameter averaging,
// gradient compression, and communication optimization for multi-node/multi-GPU training.

pub mod fedprox;
pub use fedprox::*;

pub mod ring_allreduce;
pub use ring_allreduce::{CollectiveTransport, LocalTransport, ReduceOp, RingAllReduce};

pub mod pipeline_parallel;
pub use pipeline_parallel::{
    OpKind, PipelineConfig, PipelineExecution, PipelineMetrics, PipelineOp, PipelineSchedule,
    PipelineScheduler, StageCost, StagePartitioner, StageRange,
};

pub mod elastic;
pub use elastic::{
    ElasticConfig, ElasticCoordinator, EpochSnapshot, MembershipEvent, RendezvousState, ShardRange,
    WorkerShards, WorkerState,
};

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;
use std::fmt::Debug;

/// Parameter averaging strategies for distributed training
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingStrategy {
    /// Simple arithmetic mean
    Arithmetic,
    /// Weighted average based on data sizes
    WeightedByData,
    /// Weighted average based on computation times
    WeightedByTime,
    /// Federated averaging (FedAvg)
    Federated,
    /// Momentum-based averaging
    Momentum {
        /// Momentum factor
        momentum: f64,
    },
    /// Exponentially weighted moving average
    ExponentialMovingAverage {
        /// Decay factor
        decay: f64,
    },
}

/// Distributed parameter averager
#[derive(Debug)]
pub struct ParameterAverager<A: Float, D: Dimension> {
    /// Current averaged parameters
    averaged_params: Vec<Array<A, D>>,
    /// Averaging strategy
    strategy: AveragingStrategy,
    /// Node weights for weighted averaging
    node_weights: HashMap<usize, A>,
    /// Number of participating nodes
    numnodes: usize,
    /// Momentum buffer for momentum-based averaging
    momentum_buffer: Option<Vec<Array<A, D>>>,
    /// Step count for EMA decay adjustment
    step_count: usize,
    /// Whether averager is initialized
    initialized: bool,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    ParameterAverager<A, D>
{
    /// Create a new parameter averager
    pub fn new(strategy: AveragingStrategy, numnodes: usize) -> Self {
        Self {
            averaged_params: Vec::new(),
            strategy,
            node_weights: HashMap::new(),
            numnodes,
            momentum_buffer: None,
            step_count: 0,
            initialized: false,
        }
    }

    /// Initialize averager with parameter shapes
    pub fn initialize(&mut self, params: &[Array<A, D>]) -> Result<()> {
        if self.initialized {
            return Err(OptimError::InvalidConfig(
                "Parameter averager already initialized".to_string(),
            ));
        }

        self.averaged_params = params.to_vec();

        // Initialize momentum buffer if needed
        if matches!(self.strategy, AveragingStrategy::Momentum { .. }) {
            self.momentum_buffer = Some(params.iter().map(|p| Array::zeros(p.raw_dim())).collect());
        }

        // Initialize uniform weights
        let numnodes_a = A::from(self.numnodes).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "node count {} could not be represented in the parameter type",
                self.numnodes
            ))
        })?;
        let uniform_weight = A::one() / numnodes_a;
        for nodeid in 0..self.numnodes {
            self.node_weights.insert(nodeid, uniform_weight);
        }

        self.initialized = true;
        Ok(())
    }

    /// Set weight for a specific node
    pub fn set_node_weight(&mut self, nodeid: usize, weight: A) -> Result<()> {
        if nodeid >= self.numnodes {
            return Err(OptimError::InvalidConfig(format!(
                "Node ID {} exceeds number of nodes {}",
                nodeid, self.numnodes
            )));
        }
        self.node_weights.insert(nodeid, weight);
        Ok(())
    }

    /// Average parameters from multiple nodes
    pub fn average_parameters(
        &mut self,
        nodeparameters: &[(usize, Vec<Array<A, D>>)],
    ) -> Result<()> {
        if !self.initialized {
            if let Some((_, first_params)) = nodeparameters.first() {
                self.initialize(first_params)?;
            } else {
                return Err(OptimError::InvalidConfig(
                    "No _parameters provided for initialization".to_string(),
                ));
            }
        }

        // Validate input
        for (nodeid, params) in nodeparameters {
            if *nodeid >= self.numnodes {
                return Err(OptimError::InvalidConfig(format!(
                    "Node ID {} exceeds number of nodes {}",
                    nodeid, self.numnodes
                )));
            }
            if params.len() != self.averaged_params.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Expected {} parameter arrays, got {}",
                    self.averaged_params.len(),
                    params.len()
                )));
            }
        }

        self.step_count += 1;

        match self.strategy {
            AveragingStrategy::Arithmetic => {
                self.arithmetic_average(nodeparameters)?;
            }
            AveragingStrategy::WeightedByData | AveragingStrategy::WeightedByTime => {
                self.weighted_average(nodeparameters)?;
            }
            AveragingStrategy::Federated => {
                self.federated_average(nodeparameters)?;
            }
            AveragingStrategy::Momentum { momentum } => {
                self.momentum_average(nodeparameters, momentum)?;
            }
            AveragingStrategy::ExponentialMovingAverage { decay } => {
                self.ema_average(nodeparameters, decay)?;
            }
        }

        Ok(())
    }

    /// Simple arithmetic averaging
    fn arithmetic_average(&mut self, nodeparameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        // Reset averaged _parameters
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }

        let numnodes = A::from(nodeparameters.len()).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "node count {} could not be represented in the parameter type",
                nodeparameters.len()
            ))
        })?;

        // Sum all _parameters
        for (_node_id, params) in nodeparameters {
            for (avg_param, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p;
                });
            }
        }

        // Divide by number of nodes
        for param in &mut self.averaged_params {
            param.mapv_inplace(|x| x / numnodes);
        }

        Ok(())
    }

    /// Weighted averaging using node weights
    fn weighted_average(&mut self, nodeparameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        // Reset averaged _parameters
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }

        // Compute total weight
        let total_weight: A = nodeparameters
            .iter()
            .map(|(nodeid, _)| self.node_weights.get(nodeid).copied().unwrap_or(A::zero()))
            .fold(A::zero(), |acc, w| acc + w);

        if total_weight <= A::zero() {
            return Err(OptimError::InvalidConfig(
                "Total node weights must be > 0".to_string(),
            ));
        }

        // Weighted sum
        for (nodeid, params) in nodeparameters {
            let weight = self.node_weights.get(nodeid).copied().unwrap_or(A::zero()) / total_weight;

            for (avg_param, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + weight * p;
                });
            }
        }

        Ok(())
    }

    /// Federated averaging (FedAvg). This delegates to the same weighted-average
    /// machinery as `WeightedByData`: the caller MUST call `set_node_weight` with
    /// each node's local sample-size fraction before invoking `average_parameters`,
    /// otherwise `initialize` seeds uniform weights and this degenerates to plain
    /// `Arithmetic` averaging (not an error, but not FedAvg's defining property
    /// either -- callers wanting genuine FedAvg with only local dataset sizes
    /// available should prefer `distributed::fedprox::FedProxOptimizer`, which
    /// accepts sample counts directly).
    fn federated_average(&mut self, nodeparameters: &[(usize, Vec<Array<A, D>>)]) -> Result<()> {
        self.weighted_average(nodeparameters)
    }

    /// Momentum-based averaging
    fn momentum_average(
        &mut self,
        nodeparameters: &[(usize, Vec<Array<A, D>>)],
        momentum: f64,
    ) -> Result<()> {
        if !(0.0..=1.0).contains(&momentum) {
            return Err(OptimError::InvalidConfig(format!(
                "momentum must be in [0, 1], got {momentum}"
            )));
        }
        let momentum_factor = A::from(momentum).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "momentum {momentum} could not be represented in the parameter type"
            ))
        })?;
        let one_minus_momentum = A::one() - momentum_factor;

        // First compute arithmetic average of incoming _parameters
        let mut current_average: Vec<Array<A, D>> = self
            .averaged_params
            .iter()
            .map(|param| Array::zeros(param.raw_dim()))
            .collect();

        let numnodes = A::from(nodeparameters.len()).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "node count {} could not be represented in the parameter type",
                nodeparameters.len()
            ))
        })?;
        for (_node_id, params) in nodeparameters {
            for (avg_param, param) in current_average.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p / numnodes;
                });
            }
        }

        // Apply momentum update. The momentum buffer is only allocated by
        // `initialize` when the strategy is `Momentum` *at that moment* -- if
        // the caller switched strategies afterward (or never initialized this
        // way), silently discarding the incoming update would corrupt training
        // with no signal, so we fail loudly instead.
        let momentum_buf = self.momentum_buffer.as_mut().ok_or_else(|| {
            OptimError::InvalidState(
                "Momentum averaging selected but the momentum buffer was never initialized; \
                 call initialize() while the strategy is AveragingStrategy::Momentum"
                    .to_string(),
            )
        })?;

        for ((avg_param, current_param), momentum_param) in self
            .averaged_params
            .iter_mut()
            .zip(current_average.iter())
            .zip(momentum_buf.iter_mut())
        {
            // Update momentum buffer first
            Zip::from(&mut *momentum_param)
                .and(current_param)
                .for_each(|mom, &curr| {
                    *mom = momentum_factor * *mom + one_minus_momentum * curr;
                });

            // Copy momentum buffer to averaged params
            avg_param.assign(&*momentum_param);
        }

        Ok(())
    }

    /// Exponential moving average
    fn ema_average(
        &mut self,
        nodeparameters: &[(usize, Vec<Array<A, D>>)],
        decay: f64,
    ) -> Result<()> {
        if !(0.0..=1.0).contains(&decay) {
            return Err(OptimError::InvalidConfig(format!(
                "decay must be in [0, 1], got {decay}"
            )));
        }
        let decay_factor = A::from(decay).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "decay {decay} could not be represented in the parameter type"
            ))
        })?;
        let one_minus_decay = A::one() - decay_factor;

        // First compute arithmetic average of incoming _parameters
        let mut current_average: Vec<Array<A, D>> = self
            .averaged_params
            .iter()
            .map(|param| Array::zeros(param.raw_dim()))
            .collect();

        let numnodes = A::from(nodeparameters.len()).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "node count {} could not be represented in the parameter type",
                nodeparameters.len()
            ))
        })?;
        for (_node_id, params) in nodeparameters {
            for (avg_param, param) in current_average.iter_mut().zip(params.iter()) {
                Zip::from(avg_param).and(param).for_each(|avg, &p| {
                    *avg = *avg + p / numnodes;
                });
            }
        }

        // Apply EMA update
        for (avg_param, current_param) in
            self.averaged_params.iter_mut().zip(current_average.iter())
        {
            Zip::from(avg_param)
                .and(current_param)
                .for_each(|avg, &curr| {
                    *avg = decay_factor * *avg + one_minus_decay * curr;
                });
        }

        Ok(())
    }

    /// Get current averaged parameters
    pub fn get_averaged_parameters(&self) -> &[Array<A, D>] {
        &self.averaged_params
    }

    /// Get cloned averaged parameters
    pub fn get_averaged_parameters_cloned(&self) -> Vec<Array<A, D>> {
        self.averaged_params.clone()
    }

    /// Reset averager state
    pub fn reset(&mut self) {
        self.step_count = 0;
        for param in &mut self.averaged_params {
            param.fill(A::zero());
        }
        if let Some(ref mut momentum_buf) = self.momentum_buffer {
            for buf in momentum_buf {
                buf.fill(A::zero());
            }
        }
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get number of nodes
    pub fn numnodes(&self) -> usize {
        self.numnodes
    }

    /// Get averaging strategy
    pub fn strategy(&self) -> AveragingStrategy {
        self.strategy
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Synchronous parameter server for distributed training
#[derive(Debug)]
pub struct ParameterServer<A: Float, D: Dimension> {
    /// Parameter averager
    averager: ParameterAverager<A, D>,
    /// Current global parameters
    global_parameters: Vec<Array<A, D>>,
    /// Node update counters
    update_counts: HashMap<usize, usize>,
    /// Expected updates per round
    expected_updates_per_round: usize,
    /// Current round number
    current_round: usize,
    /// Synchronization barrier
    pending_updates: HashMap<usize, Vec<Array<A, D>>>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    ParameterServer<A, D>
{
    /// Create a new parameter server
    pub fn new(
        strategy: AveragingStrategy,
        numnodes: usize,
        expected_updates_per_round: usize,
    ) -> Self {
        Self {
            averager: ParameterAverager::new(strategy, numnodes),
            global_parameters: Vec::new(),
            update_counts: HashMap::new(),
            expected_updates_per_round,
            current_round: 0,
            pending_updates: HashMap::new(),
        }
    }

    /// Initialize with global parameters
    pub fn initialize(&mut self, initialparams: &[Array<A, D>]) -> Result<()> {
        if self.expected_updates_per_round == 0
            || self.expected_updates_per_round > self.averager.numnodes()
        {
            return Err(OptimError::InvalidConfig(format!(
                "expected_updates_per_round ({}) must be in [1, numnodes={}]",
                self.expected_updates_per_round,
                self.averager.numnodes()
            )));
        }

        self.averager.initialize(initialparams)?;
        self.global_parameters = initialparams.to_vec();

        // Initialize update counts
        for nodeid in 0..self.averager.numnodes() {
            self.update_counts.insert(nodeid, 0);
        }

        Ok(())
    }

    /// Submit parameter update from a node
    ///
    /// A node that has already submitted for the current (not-yet-aggregated)
    /// round is rejected rather than silently overwritten -- `pending_updates`
    /// is a map keyed by node id, so a resubmission would otherwise vanish
    /// without raising the round's completion count, corrupting the barrier.
    pub fn submit_update(&mut self, nodeid: usize, parameters: Vec<Array<A, D>>) -> Result<bool> {
        if nodeid >= self.averager.numnodes() {
            return Err(OptimError::InvalidConfig(format!(
                "Node ID {} exceeds number of nodes {}",
                nodeid,
                self.averager.numnodes()
            )));
        }

        if self.pending_updates.contains_key(&nodeid) {
            return Err(OptimError::InvalidState(format!(
                "Node {} already submitted an update for the current round (round {}); \
                 call force_aggregation() to close the round before resubmitting",
                nodeid,
                self.current_round + 1
            )));
        }

        // Store the update
        self.pending_updates.insert(nodeid, parameters);
        *self.update_counts.entry(nodeid).or_insert(0) += 1;

        // Check if we have enough updates for this round
        let ready_for_aggregation = self.pending_updates.len() >= self.expected_updates_per_round;

        if ready_for_aggregation {
            self.aggregate_and_update()?;
        }

        Ok(ready_for_aggregation)
    }

    /// Force aggregation with current pending updates
    pub fn force_aggregation(&mut self) -> Result<()> {
        if !self.pending_updates.is_empty() {
            self.aggregate_and_update()?;
        }
        Ok(())
    }

    /// Internal aggregation and update
    fn aggregate_and_update(&mut self) -> Result<()> {
        // Convert pending updates to the format expected by averager
        let node_params: Vec<(usize, Vec<Array<A, D>>)> = self.pending_updates.drain().collect();

        // Perform averaging
        self.averager.average_parameters(&node_params)?;

        // Update global parameters
        self.global_parameters = self.averager.get_averaged_parameters_cloned();

        // Increment round
        self.current_round += 1;

        Ok(())
    }

    /// Get current global parameters
    pub fn get_global_parameters(&self) -> &[Array<A, D>] {
        &self.global_parameters
    }

    /// Get cloned global parameters
    pub fn get_global_parameters_cloned(&self) -> Vec<Array<A, D>> {
        self.global_parameters.clone()
    }

    /// Get current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get update count for a node
    pub fn get_update_count(&self, nodeid: usize) -> usize {
        self.update_counts.get(&nodeid).copied().unwrap_or(0)
    }

    /// Get number of pending updates
    pub fn pending_updates_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Set node weight for weighted averaging
    pub fn set_node_weight(&mut self, nodeid: usize, weight: A) -> Result<()> {
        self.averager.set_node_weight(nodeid, weight)
    }

    /// Reset server state
    pub fn reset(&mut self) {
        self.averager.reset();
        self.update_counts.clear();
        self.pending_updates.clear();
        self.current_round = 0;

        for nodeid in 0..self.averager.numnodes() {
            self.update_counts.insert(nodeid, 0);
        }
    }
}

/// Distributed training coordinator
#[derive(Debug)]
pub struct DistributedCoordinator<A: Float, D: Dimension> {
    /// Parameter server
    parameter_server: ParameterServer<A, D>,
    /// Communication rounds completed
    communication_rounds: usize,
    /// Convergence criteria
    convergence_threshold: A,
    /// Maximum rounds before forced stop
    max_rounds: usize,
    /// Training statistics
    training_stats: TrainingStats<A, D>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    DistributedCoordinator<A, D>
{
    /// Create a new distributed coordinator
    pub fn new(
        strategy: AveragingStrategy,
        numnodes: usize,
        expected_updates_per_round: usize,
        max_rounds: usize,
    ) -> Self {
        Self {
            parameter_server: ParameterServer::new(strategy, numnodes, expected_updates_per_round),
            communication_rounds: 0,
            // 1e-6 is representable by every IEEE-754 float type this crate
            // targets; the fallback only guards a constructor that cannot
            // itself return `Result`.
            convergence_threshold: A::from(1e-6).unwrap_or_else(A::epsilon),
            max_rounds,
            training_stats: TrainingStats::new(),
        }
    }

    /// Initialize coordinator
    pub fn initialize(&mut self, initialparams: &[Array<A, D>]) -> Result<()> {
        self.parameter_server.initialize(initialparams)?;
        self.training_stats
            .record_round(0, A::zero(), initialparams);
        Ok(())
    }

    /// Execute a communication round
    pub fn communication_round(
        &mut self,
        node_updates: Vec<(usize, Vec<Array<A, D>>)>,
    ) -> Result<CommunicationResult<A, D>> {
        let mut aggregated = false;

        // Submit all _updates
        for (nodeid, params) in node_updates {
            aggregated = self.parameter_server.submit_update(nodeid, params)? || aggregated;
        }

        // Force aggregation if not done automatically
        if !aggregated {
            self.parameter_server.force_aggregation()?;
            aggregated = true;
        }

        if aggregated {
            self.communication_rounds += 1;

            // Check convergence
            let currentparams = self.parameter_server.get_global_parameters();
            let convergence_metric = self.compute_convergence_metric(currentparams);

            self.training_stats.record_round(
                self.communication_rounds,
                convergence_metric,
                currentparams,
            );

            let converged = convergence_metric < self.convergence_threshold;
            let max_rounds_reached = self.communication_rounds >= self.max_rounds;

            Ok(CommunicationResult {
                round: self.communication_rounds,
                global_parameters: self.parameter_server.get_global_parameters_cloned(),
                converged,
                should_continue: !converged && !max_rounds_reached,
                convergence_metric,
                stats: self.training_stats.clone(),
            })
        } else {
            Ok(CommunicationResult {
                round: self.communication_rounds,
                global_parameters: self.parameter_server.get_global_parameters_cloned(),
                converged: false,
                should_continue: true,
                convergence_metric: A::infinity(),
                stats: self.training_stats.clone(),
            })
        }
    }

    /// Set convergence threshold
    pub fn set_convergence_threshold(&mut self, threshold: A) {
        self.convergence_threshold = threshold;
    }

    /// Get parameter server reference
    pub fn parameter_server(&self) -> &ParameterServer<A, D> {
        &self.parameter_server
    }

    /// Get mutable parameter server reference
    pub fn parameter_server_mut(&mut self) -> &mut ParameterServer<A, D> {
        &mut self.parameter_server
    }

    /// Compute convergence metric (parameter change magnitude)
    fn compute_convergence_metric(&self, currentparams: &[Array<A, D>]) -> A {
        if let Some(prev_params) = self.training_stats.get_previous_parameters() {
            let mut total_change = A::zero();
            let mut total_norm = A::zero();

            for (curr, prev) in currentparams.iter().zip(prev_params.iter()) {
                for (&c, &p) in curr.iter().zip(prev.iter()) {
                    let diff = c - p;
                    total_change = total_change + diff * diff;
                    total_norm = total_norm + c * c;
                }
            }

            if total_norm > A::zero() {
                (total_change / total_norm).sqrt()
            } else {
                A::zero()
            }
        } else {
            A::infinity()
        }
    }
}

/// Result of a communication round
#[derive(Debug, Clone)]
pub struct CommunicationResult<A: Float, D: Dimension> {
    /// Round number
    pub round: usize,
    /// Updated global parameters
    pub global_parameters: Vec<Array<A, D>>,
    /// Whether training has converged
    pub converged: bool,
    /// Whether training should continue
    pub should_continue: bool,
    /// Convergence metric value
    pub convergence_metric: A,
    /// Training statistics
    pub stats: TrainingStats<A, D>,
}

/// Training statistics for distributed training
#[derive(Debug, Clone)]
pub struct TrainingStats<A: Float, D: Dimension> {
    /// Convergence history
    convergence_history: Vec<A>,
    /// Round timestamps
    round_times: Vec<usize>,
    /// Previous round's parameters, kept so `compute_convergence_metric` can
    /// measure real parameter movement instead of always reporting "no
    /// history".
    previous_parameters: Option<Vec<Array<A, D>>>,
}

impl<A: Float + Send + Sync, D: Dimension> TrainingStats<A, D> {
    /// Create new training stats
    pub fn new() -> Self {
        Self {
            convergence_history: Vec::new(),
            round_times: Vec::new(),
            previous_parameters: None,
        }
    }

    /// Record a training round
    pub fn record_round(
        &mut self,
        round: usize,
        convergence_metric: A,
        parameters: &[Array<A, D>],
    ) {
        self.convergence_history.push(convergence_metric);
        self.round_times.push(round);
        self.previous_parameters = Some(parameters.to_vec());
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[A] {
        &self.convergence_history
    }

    /// Get latest convergence metric
    pub fn latest_convergence(&self) -> Option<A> {
        self.convergence_history.last().copied()
    }

    /// Get number of rounds
    pub fn num_rounds(&self) -> usize {
        self.round_times.len()
    }

    /// Get the parameters recorded by the previous `record_round` call, if any
    fn get_previous_parameters(&self) -> Option<&[Array<A, D>]> {
        self.previous_parameters.as_deref()
    }
}

impl<A: Float + Send + Sync, D: Dimension> Default for TrainingStats<A, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient compression strategies for communication optimization
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Top-K sparsification (keep only top K largest gradients)
    TopK {
        /// Number of top gradients to keep
        k: usize,
    },
    /// Random-K sparsification (keep K random gradients)
    RandomK {
        /// Number of random gradients to keep
        k: usize,
    },
    /// Threshold-based sparsification (keep gradients above threshold)
    Threshold {
        /// Threshold value for gradient magnitude
        threshold: f64,
    },
    /// Quantization to fewer bits
    Quantization {
        /// Number of bits for quantization
        bits: u8,
    },
    /// Error feedback compression (maintain error state)
    ErrorFeedback {
        /// Base compression strategy to apply
        base_strategy: Box<CompressionStrategy>,
        /// Whether to enable error compensation
        error_compensation: bool,
    },
    /// Gradient clipping before compression
    ClippedCompression {
        /// Base compression strategy to apply after clipping
        base_strategy: Box<CompressionStrategy>,
        /// Value to clip gradients to
        clip_value: f64,
    },
}

/// Read a little-endian `f64` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer (e.g. from a
/// tampered or short network payload).
fn read_f64_le(bytes: &[u8]) -> Result<f64> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 8 bytes for an f64".to_string(),
        )
    })?;
    Ok(f64::from_le_bytes(arr))
}

/// Read a little-endian `u32` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer.
fn read_u32_le(bytes: &[u8]) -> Result<u32> {
    let arr: [u8; 4] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 4 bytes for a u32".to_string(),
        )
    })?;
    Ok(u32::from_le_bytes(arr))
}

/// Read a little-endian `u16` out of a byte slice, returning an honest error
/// instead of panicking on a truncated/corrupted buffer.
fn read_u16_le(bytes: &[u8]) -> Result<u16> {
    let arr: [u8; 2] = bytes.try_into().map_err(|_| {
        OptimError::InvalidConfig(
            "corrupted compressed data: expected 2 bytes for a u16".to_string(),
        )
    })?;
    Ok(u16::from_le_bytes(arr))
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub struct CompressedGradient<A: Float> {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata<A>,
    /// Original shape information
    pub shapes: Vec<Vec<usize>>,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata<A: Float> {
    /// Compression strategy used
    pub strategy: CompressionStrategy,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Number of non-zero elements (for sparse methods)
    pub nnz_count: usize,
    /// Quantization scale factors (for quantization methods)
    pub scale_factors: Vec<A>,
    /// Additional strategy-specific data
    pub extra_data: Vec<u8>,
}

/// Gradient compression engine
#[derive(Debug)]
pub struct GradientCompressor<A: Float, D: Dimension> {
    /// Compression strategy
    strategy: CompressionStrategy,
    /// Error feedback state for error compensation
    error_state: Option<Vec<Array<A, D>>>,
    /// Compression statistics
    stats: CompressionStats,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    GradientCompressor<A, D>
{
    /// Create a new gradient compressor
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self {
            strategy,
            error_state: None,
            stats: CompressionStats::new(),
        }
    }

    /// Initialize error state for error feedback compression
    pub fn initialize_error_state(&mut self, gradientshapes: &[Array<A, D>]) {
        self.error_state = Some(
            gradientshapes
                .iter()
                .map(|g| Array::zeros(g.raw_dim()))
                .collect(),
        );
    }

    /// Compress gradients
    pub fn compress(&mut self, gradients: &[Array<A, D>]) -> Result<CompressedGradient<A>> {
        // Lazily initialize the error-feedback residual the first time a caller
        // selects ErrorFeedback with compensation enabled but never called
        // `initialize_error_state` themselves. Previously this silently
        // degraded to plain (uncompensated) compression with no signal.
        let needs_lazy_init = matches!(
            &self.strategy,
            CompressionStrategy::ErrorFeedback {
                error_compensation: true,
                ..
            }
        ) && self.error_state.is_none();
        if needs_lazy_init {
            self.initialize_error_state(gradients);
        }

        // Apply error feedback if enabled: working = gradient + accumulated residual
        let mut working_gradients: Vec<Array<A, D>> =
            if let Some(ref error_state) = self.error_state {
                gradients
                    .iter()
                    .zip(error_state.iter())
                    .map(|(grad, error)| grad + error)
                    .collect()
            } else {
                gradients.to_vec()
            };

        let (compressed_data, metadata) = match &self.strategy {
            CompressionStrategy::None => self.compress_none(&working_gradients)?,
            CompressionStrategy::TopK { k } => self.compress_topk(&working_gradients, *k)?,
            CompressionStrategy::RandomK { k } => self.compress_randomk(&working_gradients, *k)?,
            CompressionStrategy::Threshold { threshold } => self.compress_threshold(
                &working_gradients,
                A::from(*threshold).ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "threshold {threshold} could not be represented in the parameter type"
                    ))
                })?,
            )?,
            CompressionStrategy::Quantization { bits } => {
                self.compress_quantization(&working_gradients, *bits)?
            }
            CompressionStrategy::ErrorFeedback {
                base_strategy,
                error_compensation,
            } => {
                // Recursively apply base strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;

                // Honour the `error_compensation` flag: when disabled, no
                // residual should be tracked or applied on future calls.
                if *error_compensation {
                    // EF-SGD residual: e_new = working - decompress(compress(working)).
                    // Using the raw input (`original`) here instead of `working`
                    // (which already folds in the previous residual `e_old`) would
                    // pin the residual at a constant and coordinates that never
                    // individually clear the Top-K threshold would never be sent.
                    let decompressed = temp_compressor.decompress(&compressed)?;
                    if let Some(ref mut error_state) = self.error_state {
                        for ((working, decompressed), error) in working_gradients
                            .iter()
                            .zip(decompressed.iter())
                            .zip(error_state.iter_mut())
                        {
                            *error = working - decompressed;
                        }
                    }
                }

                (compressed.data, compressed.metadata)
            }
            CompressionStrategy::ClippedCompression {
                base_strategy,
                clip_value,
            } => {
                // Clip gradients first
                let clip_val = A::from(*clip_value).ok_or_else(|| {
                    OptimError::InvalidConfig(format!(
                        "clip value {clip_value} could not be represented in the parameter type"
                    ))
                })?;
                for grad in &mut working_gradients {
                    grad.mapv_inplace(|x| {
                        if x > clip_val {
                            clip_val
                        } else if x < -clip_val {
                            -clip_val
                        } else {
                            x
                        }
                    });
                }

                // Apply base compression strategy
                let mut temp_compressor = GradientCompressor::new((**base_strategy).clone());
                let compressed = temp_compressor.compress(&working_gradients)?;
                (compressed.data, compressed.metadata)
            }
        };

        // Collect shape information
        let shapes = gradients.iter().map(|g| g.shape().to_vec()).collect();

        let result = CompressedGradient {
            data: compressed_data,
            metadata,
            shapes,
        };

        // Update statistics
        let original_size = self.calculate_size(gradients);
        let compressed_size = result.data.len();
        self.stats
            .record_compression(original_size, compressed_size);

        Ok(result)
    }

    /// Decompress gradients
    pub fn decompress(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        match &compressed.metadata.strategy {
            CompressionStrategy::None => self.decompress_none(compressed),
            CompressionStrategy::TopK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::RandomK { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Threshold { .. } => self.decompress_sparse(compressed),
            CompressionStrategy::Quantization { bits } => {
                self.decompress_quantization(compressed, *bits)
            }
            CompressionStrategy::ErrorFeedback { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
            CompressionStrategy::ClippedCompression { base_strategy, .. } => {
                let temp_compressor = GradientCompressor::new((**base_strategy).clone());
                temp_compressor.decompress(compressed)
            }
        }
    }

    /// Compress with no compression (passthrough)
    fn compress_none(
        &self,
        gradients: &[Array<A, D>],
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut data = Vec::new();

        // Simple serialization: store all gradient values sequentially
        for grad in gradients {
            for &val in grad.iter() {
                let bits = val.to_f64().ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "gradient value could not be converted to f64 for serialization"
                            .to_string(),
                    )
                })?;
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::None,
            compression_ratio: 1.0,
            nnz_count: gradients.iter().map(|g| g.len()).sum(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Top-K sparsification
    fn compress_topk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Collect (signed value, index) pairs once -- capturing the signed
            // value up front avoids an O(n) `.nth()` re-lookup per selected
            // element below (previously O(n*k) per gradient).
            let mut value_indices: Vec<(A, usize)> =
                grad.iter().enumerate().map(|(i, &val)| (val, i)).collect();

            // Sort by absolute value (descending). NaN gradients (e.g. from a
            // diverged run) must not panic a comparator run by `sort_by` --
            // treat incomparable pairs as equal rather than unwrapping.
            value_indices.sort_by(|a, b| {
                b.0.abs()
                    .partial_cmp(&a.0.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take top k elements
            let k_local = k.min(value_indices.len());
            for &(val, orig_idx) in value_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, orig_idx as u32));
                values.push(val);
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();

        // Store number of sparse elements
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        // Store indices and values
        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::TopK { k },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using Random-K sparsification
    fn compress_randomk(
        &self,
        gradients: &[Array<A, D>],
        k: usize,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;
        let mut rng = thread_rng();

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            // Random sampling of k indices via a genuine partial Fisher-Yates
            // shuffle. The previous implementation picked a swap index that
            // was a pure function of (grad_idx, i) -- every node selected the
            // identical index set every round (losing Random-K's unbiased-
            // estimator property), and it divided by `grad.len() - i`, which
            // is unreachable-but-fragile when i approaches grad.len().
            let k_local = k.min(grad.len());
            let mut selected_indices: Vec<usize> = (0..grad.len()).collect();
            for i in 0..k_local {
                let remaining = grad.len() - i;
                let swap_idx = i + rng.gen_range(0..remaining);
                selected_indices.swap(i, swap_idx);
            }

            // Flatten once so per-element access below is O(1) instead of the
            // previous O(n) `.nth()` walk (O(n*k) total per gradient).
            let flat: Vec<A> = grad.iter().copied().collect();
            for &idx in selected_indices.iter().take(k_local) {
                indices.push((grad_idx as u32, idx as u32));
                values.push(flat[idx]);
            }
        }

        // Serialize sparse representation (same format as Top-K)
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::RandomK { k },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using threshold-based sparsification
    fn compress_threshold(
        &self,
        gradients: &[Array<A, D>],
        threshold: A,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut total_elements = 0;

        for (grad_idx, grad) in gradients.iter().enumerate() {
            total_elements += grad.len();

            for (elem_idx, &val) in grad.iter().enumerate() {
                if val.abs() > threshold {
                    indices.push((grad_idx as u32, elem_idx as u32));
                    values.push(val);
                }
            }
        }

        // Serialize sparse representation
        let mut data = Vec::new();
        data.extend_from_slice(&(indices.len() as u32).to_le_bytes());

        for ((grad_idx, elem_idx), value) in indices.iter().zip(values.iter()) {
            data.extend_from_slice(&grad_idx.to_le_bytes());
            data.extend_from_slice(&elem_idx.to_le_bytes());
            let bits = value.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "gradient value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&bits.to_le_bytes());
        }

        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Threshold {
                threshold: threshold.to_f64().ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "threshold could not be converted to f64 for metadata".to_string(),
                    )
                })?,
            },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: indices.len(),
            scale_factors: Vec::new(),
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Compress using quantization
    fn compress_quantization(
        &self,
        gradients: &[Array<A, D>],
        bits: u8,
    ) -> Result<(Vec<u8>, CompressionMetadata<A>)> {
        if bits == 0 || bits > 32 {
            return Err(OptimError::InvalidConfig(
                "Quantization bits must be in 1..=32".to_string(),
            ));
        }

        let mut data = Vec::new();
        let mut scale_factors = Vec::new();
        let levels = (1u64 << bits) - 1;
        let levels_a = A::from(levels).ok_or_else(|| {
            OptimError::InvalidConfig(format!(
                "quantization level count {levels} could not be represented in the parameter type"
            ))
        })?;

        for grad in gradients {
            // Reject non-finite gradients up front: NaN/inf would otherwise
            // corrupt the min/max fold below (whose behaviour on NaN is
            // unspecified) and could drive `normalized` negative or NaN,
            // which used to panic in `to_u64().expect(...)`.
            if grad.iter().any(|v| !v.is_finite()) {
                return Err(OptimError::InvalidConfig(
                    "gradient contains non-finite (NaN/inf) values; cannot quantize".to_string(),
                ));
            }

            // Find min and max values for this gradient
            let min_val = grad.iter().fold(A::infinity(), |acc, &x| acc.min(x));
            let max_val = grad.iter().fold(A::neg_infinity(), |acc, &x| acc.max(x));

            let range = max_val - min_val;
            let scale = if range > A::zero() {
                range / levels_a
            } else {
                A::one()
            };

            scale_factors.push(scale);

            // Quantize each value, clamping into [0, levels] so the u64
            // conversion below can never fail (previously an unclamped
            // negative/NaN `normalized` would panic).
            for &val in grad.iter() {
                let normalized = ((val - min_val) / scale)
                    .max(A::zero())
                    .min(levels_a)
                    .round();
                let quantized = normalized.to_u64().unwrap_or(levels).min(levels) as u32;

                // Store quantized value
                match bits {
                    1..=8 => data.push(quantized as u8),
                    9..=16 => data.extend_from_slice(&(quantized as u16).to_le_bytes()),
                    17..=32 => data.extend_from_slice(&quantized.to_le_bytes()),
                    _ => unreachable!(),
                }
            }

            // Store min value AND scale inline for reconstruction. Carrying
            // both in the byte stream (rather than trusting that the
            // separately-returned `scale_factors[grad_idx]` stays aligned by
            // position) means decompression never depends on a parallel
            // array matching this stream's gradient order.
            let min_bits = min_val.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "min value could not be converted to f64 for serialization".to_string(),
                )
            })?;
            let scale_bits = scale.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "scale factor could not be converted to f64 for serialization".to_string(),
                )
            })?;
            data.extend_from_slice(&min_bits.to_le_bytes());
            data.extend_from_slice(&scale_bits.to_le_bytes());
        }

        let total_elements: usize = gradients.iter().map(|g| g.len()).sum();
        let metadata = CompressionMetadata {
            strategy: CompressionStrategy::Quantization { bits },
            compression_ratio: data.len() as f64
                / (total_elements.max(1) * std::mem::size_of::<A>()) as f64,
            nnz_count: total_elements,
            scale_factors,
            extra_data: Vec::new(),
        };

        Ok((data, metadata))
    }

    /// Decompress uncompressed data
    fn decompress_none(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;

        for shape in &compressed.shapes {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            for _ in 0..num_elements {
                if data_offset + 8 > compressed.data.len() {
                    return Err(OptimError::InvalidConfig(
                        "Insufficient data for decompression".to_string(),
                    ));
                }

                let value = read_f64_le(&compressed.data[data_offset..data_offset + 8])?;
                values.push(A::from(value).ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "decompressed value could not be represented in the parameter type"
                            .to_string(),
                    )
                })?);
                data_offset += 8;
            }

            // Create a dynamic array first, then convert to the target dimension type
            let dynamic_array = Array::from_shape_vec(shape.as_slice(), values).map_err(|_| {
                OptimError::InvalidConfig("Invalid shape for reconstruction".to_string())
            })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed".to_string())
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Decompress sparse representation
    fn decompress_sparse(&self, compressed: &CompressedGradient<A>) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();

        // Initialize zero arrays
        for shape in &compressed.shapes {
            let dynamic_array = Array::zeros(shape.as_slice());
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig("Dimension conversion failed for zero array".to_string())
            })?;
            result.push(array);
        }

        // Read number of sparse elements
        if compressed.data.len() < 4 {
            return Err(OptimError::InvalidConfig(
                "Invalid compressed data format".to_string(),
            ));
        }

        let num_elements = read_u32_le(&compressed.data[0..4])? as usize;
        let mut data_offset = 4;

        // Restore sparse elements
        for _ in 0..num_elements {
            if data_offset + 16 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Insufficient data for sparse decompression".to_string(),
                ));
            }

            let grad_idx = read_u32_le(&compressed.data[data_offset..data_offset + 4])? as usize;
            let elem_idx =
                read_u32_le(&compressed.data[data_offset + 4..data_offset + 8])? as usize;
            let value_f64 = read_f64_le(&compressed.data[data_offset + 8..data_offset + 16])?;
            let value = A::from(value_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "decompressed value could not be represented in the parameter type".to_string(),
                )
            })?;

            data_offset += 16;

            if grad_idx >= result.len() {
                return Err(OptimError::InvalidConfig(
                    "Invalid gradient index in compressed data".to_string(),
                ));
            }

            // Write via a flat slice (O(1) indexed access) instead of
            // `.iter_mut().nth(elem_idx)`, which re-walks from the start of
            // the array for every restored element.
            let target = result[grad_idx].as_slice_mut().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "target array is not contiguous; cannot write decompressed element".to_string(),
                )
            })?;
            match target.get_mut(elem_idx) {
                Some(elem) => *elem = value,
                None => {
                    return Err(OptimError::InvalidConfig(
                        "Invalid element index in compressed data".to_string(),
                    ));
                }
            }
        }

        Ok(result)
    }

    /// Decompress quantized data
    fn decompress_quantization(
        &self,
        compressed: &CompressedGradient<A>,
        bits: u8,
    ) -> Result<Vec<Array<A, D>>> {
        let mut result = Vec::new();
        let mut data_offset = 0;

        for shape in compressed.shapes.iter() {
            let num_elements: usize = shape.iter().product();
            let mut values = Vec::with_capacity(num_elements);

            // Read quantized values
            for _ in 0..num_elements {
                let quantized = match bits {
                    1..=8 => {
                        if data_offset >= compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = compressed.data[data_offset] as u32;
                        data_offset += 1;
                        val
                    }
                    9..=16 => {
                        if data_offset + 2 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val =
                            read_u16_le(&compressed.data[data_offset..data_offset + 2])? as u32;
                        data_offset += 2;
                        val
                    }
                    17..=32 => {
                        if data_offset + 4 > compressed.data.len() {
                            return Err(OptimError::InvalidConfig(
                                "Insufficient quantized data".to_string(),
                            ));
                        }
                        let val = read_u32_le(&compressed.data[data_offset..data_offset + 4])?;
                        data_offset += 4;
                        val
                    }
                    _ => {
                        return Err(OptimError::InvalidConfig(
                            "Invalid quantization bits".to_string(),
                        ))
                    }
                };

                values.push(quantized);
            }

            // Read min value and scale, stored inline by `compress_quantization`
            // right after each gradient's quantized block. Reading them from
            // the stream itself (rather than indexing into the separately
            // carried `metadata.scale_factors` by position) means
            // reconstruction never depends on that parallel array staying
            // aligned with this one.
            if data_offset + 16 > compressed.data.len() {
                return Err(OptimError::InvalidConfig(
                    "Missing min value/scale for quantization".to_string(),
                ));
            }
            let min_val_f64 = read_f64_le(&compressed.data[data_offset..data_offset + 8])?;
            let scale_f64 = read_f64_le(&compressed.data[data_offset + 8..data_offset + 16])?;
            let min_val = A::from(min_val_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "min value could not be represented in the parameter type".to_string(),
                )
            })?;
            let scale = A::from(scale_f64).ok_or_else(|| {
                OptimError::InvalidConfig(
                    "scale factor could not be represented in the parameter type".to_string(),
                )
            })?;
            data_offset += 16;

            // Dequantize values
            let dequantized_values: Vec<A> = values
                .into_iter()
                .map(|q| -> Result<A> {
                    let q_a = A::from(q).ok_or_else(|| {
                        OptimError::InvalidConfig(
                            "quantized value could not be represented in the parameter type"
                                .to_string(),
                        )
                    })?;
                    Ok(min_val + q_a * scale)
                })
                .collect::<Result<Vec<A>>>()?;

            let dynamic_array = Array::from_shape_vec(shape.as_slice(), dequantized_values)
                .map_err(|_| {
                    OptimError::InvalidConfig(
                        "Invalid shape for quantized reconstruction".to_string(),
                    )
                })?;
            let array = dynamic_array.into_dimensionality::<D>().map_err(|_| {
                OptimError::InvalidConfig(
                    "Dimension conversion failed for quantized array".to_string(),
                )
            })?;
            result.push(array);
        }

        Ok(result)
    }

    /// Calculate size of gradients in bytes
    fn calculate_size(&self, gradients: &[Array<A, D>]) -> usize {
        gradients
            .iter()
            .map(|g| g.len() * std::mem::size_of::<A>())
            .sum()
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::new();
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Total compressions performed
    pub compressions_count: usize,
    /// Total original bytes
    pub total_original_bytes: usize,
    /// Total compressed bytes
    pub total_compressed_bytes: usize,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Best compression ratio achieved
    pub best_compression_ratio: f64,
    /// Worst compression ratio achieved
    pub worst_compression_ratio: f64,
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self {
            compressions_count: 0,
            total_original_bytes: 0,
            total_compressed_bytes: 0,
            average_compression_ratio: 0.0,
            best_compression_ratio: f64::INFINITY,
            worst_compression_ratio: 0.0,
        }
    }

    /// Record a compression operation
    pub fn record_compression(&mut self, original_bytes: usize, compressedbytes: usize) {
        self.compressions_count += 1;
        self.total_original_bytes += original_bytes;
        self.total_compressed_bytes += compressedbytes;

        let ratio = if original_bytes > 0 {
            compressedbytes as f64 / original_bytes as f64
        } else {
            1.0
        };

        self.best_compression_ratio = self.best_compression_ratio.min(ratio);
        self.worst_compression_ratio = self.worst_compression_ratio.max(ratio);

        self.average_compression_ratio = if self.total_original_bytes > 0 {
            self.total_compressed_bytes as f64 / self.total_original_bytes as f64
        } else {
            0.0
        };
    }

    /// Get overall compression ratio
    pub fn overall_compression_ratio(&self) -> f64 {
        self.average_compression_ratio
    }

    /// Get bandwidth savings (as percentage)
    pub fn bandwidth_savings(&self) -> f64 {
        (1.0 - self.average_compression_ratio) * 100.0
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_arithmetic_averaging() {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Arithmetic, 3);

        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let params2 = vec![Array1::from_vec(vec![3.0, 4.0])];
        let params3 = vec![Array1::from_vec(vec![5.0, 6.0])];

        let nodeparameters = vec![(0, params1), (1, params2), (2, params3)];

        averager
            .average_parameters(&nodeparameters)
            .expect("unwrap failed");

        let result = averager.get_averaged_parameters();
        assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6); // (1+3+5)/3
        assert_relative_eq!(result[0][1], 4.0, epsilon = 1e-6); // (2+4+6)/3
    }

    #[test]
    fn test_weighted_averaging() {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

        // Initialize first to avoid overwriting weights
        let params1 = vec![Array1::from_vec(vec![2.0])];
        let params2 = vec![Array1::from_vec(vec![6.0])];
        let nodeparameters = vec![(0, params1.clone()), (1, params2.clone())];
        averager.initialize(&params1).expect("unwrap failed");

        // Set different weights after initialization
        averager.set_node_weight(0, 0.75).expect("unwrap failed"); // 75% weight
        averager.set_node_weight(1, 0.25).expect("unwrap failed"); // 25% weight

        averager
            .average_parameters(&nodeparameters)
            .expect("unwrap failed");

        let result = averager.get_averaged_parameters();
        // Weighted average: 0.75 * 2.0 + 0.25 * 6.0 = 1.5 + 1.5 = 3.0
        assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_momentum_averaging() {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Momentum { momentum: 0.9 }, 2);

        let params1 = vec![Array1::from_vec(vec![1.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])];

        // First update: average = (1+3)/2 = 2.0, momentum buffer starts at 0, so result = 0.1 * 2.0 = 0.2
        let node_parameters1 = vec![(0, params1.clone()), (1, params2.clone())];
        averager
            .average_parameters(&node_parameters1)
            .expect("unwrap failed");

        let result1 = averager.get_averaged_parameters();
        // First result should be small due to zero initialization
        assert!(result1[0][0] >= 0.0 && result1[0][0] <= 0.5);

        // Several more updates to let momentum build up
        for _ in 0..10 {
            let nodeparameters = vec![(0, params1.clone()), (1, params2.clone())];
            averager
                .average_parameters(&nodeparameters)
                .expect("unwrap failed");
        }

        let final_result = averager.get_averaged_parameters();
        // After many updates, momentum should gradually converge towards the average (2.0)
        // But with momentum=0.9, it builds up slowly, so we use a broader range
        assert!(final_result[0][0] > 0.5 && final_result[0][0] < 2.5);
    }

    #[test]
    fn test_parameter_server() {
        let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);

        let initialparams = vec![Array1::from_vec(vec![0.0, 0.0])];
        server.initialize(&initialparams).expect("unwrap failed");

        // Submit updates from both nodes
        let update1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let update2 = vec![Array1::from_vec(vec![3.0, 4.0])];

        let ready1 = server.submit_update(0, update1).expect("unwrap failed");
        assert!(!ready1); // Not ready yet, waiting for second node

        let ready2 = server.submit_update(1, update2).expect("unwrap failed");
        assert!(ready2); // Ready after both nodes submitted

        let global_params = server.get_global_parameters();
        assert_relative_eq!(global_params[0][0], 2.0, epsilon = 1e-6); // (1+3)/2
        assert_relative_eq!(global_params[0][1], 3.0, epsilon = 1e-6); // (2+4)/2

        assert_eq!(server.current_round(), 1);
    }

    #[test]
    fn test_distributed_coordinator() {
        let mut coordinator = DistributedCoordinator::new(
            AveragingStrategy::Arithmetic,
            2,  // 2 nodes
            2,  // expect 2 updates per round
            10, // max 10 rounds
        );

        let initialparams = vec![Array1::from_vec(vec![0.0])];
        coordinator
            .initialize(&initialparams)
            .expect("unwrap failed");

        // Simulate training rounds
        for round in 1..=3 {
            let update1 = vec![Array1::from_vec(vec![round as f64])];
            let update2 = vec![Array1::from_vec(vec![(round * 2) as f64])];

            let node_updates = vec![(0, update1), (1, update2)];

            let result = coordinator
                .communication_round(node_updates)
                .expect("unwrap failed");

            assert_eq!(result.round, round);
            assert!(result.should_continue);
            assert!(!result.converged); // Unlikely to converge with these updates

            // Check that global parameters are updated
            assert!(result.global_parameters[0][0] > 0.0);
        }
    }

    #[test]
    fn test_averaging_strategies() {
        // Test arithmetic and federated strategies that should produce expected ranges
        let simple_strategies = vec![
            AveragingStrategy::Arithmetic,
            AveragingStrategy::WeightedByData,
            AveragingStrategy::Federated,
        ];

        for strategy in simple_strategies {
            let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
                ParameterAverager::new(strategy, 2);

            let params1 = vec![Array1::from_vec(vec![1.0])];
            let params2 = vec![Array1::from_vec(vec![3.0])];

            let nodeparameters = vec![(0, params1), (1, params2)];

            averager
                .average_parameters(&nodeparameters)
                .expect("unwrap failed");
            let result = averager.get_averaged_parameters();
            assert!(result[0][0] >= 1.0 && result[0][0] <= 3.0);
        }

        // Test momentum and EMA strategies separately (they start from zero state)
        let stateful_strategies = vec![
            AveragingStrategy::Momentum { momentum: 0.9 },
            AveragingStrategy::ExponentialMovingAverage { decay: 0.9 },
        ];

        for strategy in stateful_strategies {
            let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
                ParameterAverager::new(strategy, 2);

            let params1 = vec![Array1::from_vec(vec![1.0])];
            let params2 = vec![Array1::from_vec(vec![3.0])];

            let nodeparameters = vec![(0, params1), (1, params2)];

            averager
                .average_parameters(&nodeparameters)
                .expect("unwrap failed");
            let result = averager.get_averaged_parameters();
            // First result from momentum/EMA will be smaller due to zero initialization
            assert!(result[0][0] >= 0.0 && result[0][0] <= 3.0);
        }
    }

    #[test]
    fn test_node_weight_validation() {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

        // Valid node ID
        assert!(averager.set_node_weight(0, 0.5).is_ok());
        assert!(averager.set_node_weight(1, 0.5).is_ok());

        // Invalid node ID
        assert!(averager.set_node_weight(2, 0.5).is_err());
    }

    #[test]
    fn test_parameter_dimension_validation() {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(AveragingStrategy::Arithmetic, 2);

        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])]; // Wrong dimension

        let nodeparameters = vec![(0, params1), (1, params2)];

        // Should fail due to dimension mismatch - currently panics instead of returning error
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            averager.average_parameters(&nodeparameters)
        }));

        // Either it returns an error or panics due to dimension mismatch
        assert!(result.is_err() || (result.is_ok() && result.expect("unwrap failed").is_err()));
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::new();

        assert_eq!(stats.num_rounds(), 0);
        assert!(stats.latest_convergence().is_none());

        let params = vec![Array1::from_vec(vec![1.0])];
        stats.record_round(1, 0.5, &params);

        assert_eq!(stats.num_rounds(), 1);
        assert_eq!(stats.latest_convergence(), Some(0.5));
        assert_eq!(stats.convergence_history(), &[0.5]);
    }

    #[test]
    fn test_gradient_compression_none() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::None);

        let gradients = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0]),
        ];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        assert_eq!(compressed.metadata.strategy, CompressionStrategy::None);
        assert_eq!(compressed.metadata.compression_ratio, 1.0);

        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
        assert_eq!(decompressed.len(), 2);
        assert_eq!(
            decompressed[0].as_slice().expect("unwrap failed"),
            &[1.0, 2.0, 3.0]
        );
        assert_eq!(
            decompressed[1].as_slice().expect("unwrap failed"),
            &[4.0, 5.0]
        );
    }

    #[test]
    fn test_gradient_compression_topk() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 2 });

        let gradients = vec![Array1::from_vec(vec![0.1, 3.0, 0.2, 4.0, 0.05])];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 2); // Top 2 elements

        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
        assert_eq!(decompressed.len(), 1);

        // Should have only the top 2 elements (4.0 and 3.0), others should be 0
        let result = &decompressed[0];
        assert_eq!(result[1], 3.0); // Original position of 3.0
        assert_eq!(result[3], 4.0); // Original position of 4.0
        assert_eq!(result[0], 0.0); // Should be zeroed
        assert_eq!(result[2], 0.0); // Should be zeroed
        assert_eq!(result[4], 0.0); // Should be zeroed
    }

    #[test]
    fn test_gradient_compression_threshold() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::Threshold { threshold: 1.0 });

        let gradients = vec![Array1::from_vec(vec![0.5, 2.0, 0.8, 3.0, 0.3])];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 2); // Elements > 1.0: 2.0 and 3.0

        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
        let result = &decompressed[0];

        // Only elements > 1.0 should remain
        assert_eq!(result[0], 0.0); // 0.5 < 1.0
        assert_eq!(result[1], 2.0); // 2.0 > 1.0
        assert_eq!(result[2], 0.0); // 0.8 < 1.0
        assert_eq!(result[3], 3.0); // 3.0 > 1.0
        assert_eq!(result[4], 0.0); // 0.3 < 1.0
    }

    #[test]
    fn test_gradient_compression_quantization() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::Quantization { bits: 8 });

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        assert!(compressed.metadata.compression_ratio < 1.0); // Should use less space with 8-bit quantization

        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
        let result = &decompressed[0];

        // Values should be approximately restored (with quantization error)
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
        assert!((result[2] - 3.0).abs() < 0.1);
        assert!((result[3] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_gradient_compression_randomk() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::RandomK { k: 3 });

        // Use a larger array to make compression effective
        let gradients = vec![Array1::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ])];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        // With 3 out of 10 elements, compression should be effective
        assert!(compressed.metadata.compression_ratio < 1.0);
        assert_eq!(compressed.metadata.nnz_count, 3); // Exactly 3 elements should be kept

        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
        let result = &decompressed[0];

        // Exactly 3 elements should be non-zero
        let non_zero_count = result.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 3);
    }

    #[test]
    fn test_gradient_compression_error_feedback() {
        let base_strategy = CompressionStrategy::TopK { k: 2 };
        let strategy = CompressionStrategy::ErrorFeedback {
            base_strategy: Box::new(base_strategy),
            error_compensation: true,
        };

        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

        // Initialize error state
        compressor.initialize_error_state(&gradients);

        // First compression
        let compressed1 = compressor.compress(&gradients).expect("unwrap failed");
        let decompressed1 = compressor.decompress(&compressed1).expect("unwrap failed");

        // Second compression (should include error feedback)
        let compressed2 = compressor.compress(&gradients).expect("unwrap failed");
        let decompressed2 = compressor.decompress(&compressed2).expect("unwrap failed");

        // Both should be valid compressions
        assert_eq!(decompressed1.len(), 1);
        assert_eq!(decompressed2.len(), 1);
    }

    #[test]
    fn test_gradient_compression_clipped() {
        let base_strategy = CompressionStrategy::TopK { k: 3 };
        let strategy = CompressionStrategy::ClippedCompression {
            base_strategy: Box::new(base_strategy),
            clip_value: 2.5,
        };

        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 5.0, -3.0, 2.0])];

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");

        let result = &decompressed[0];

        // Values should be clipped to [-2.5, 2.5] and then top-k applied
        for &val in result.iter() {
            if val != 0.0 {
                // Non-zero values from top-k
                assert!((-2.5..=2.5).contains(&val));
            }
        }
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();

        assert_eq!(stats.compressions_count, 0);
        assert_eq!(stats.overall_compression_ratio(), 0.0);

        // Record some compressions
        stats.record_compression(1000, 500); // 50% compression
        assert_eq!(stats.compressions_count, 1);
        assert_relative_eq!(stats.overall_compression_ratio(), 0.5, epsilon = 1e-6);
        assert_relative_eq!(stats.bandwidth_savings(), 50.0, epsilon = 1e-6);

        stats.record_compression(1000, 250); // 25% compression
        assert_eq!(stats.compressions_count, 2);
        assert_relative_eq!(stats.overall_compression_ratio(), 0.375, epsilon = 1e-6); // (500+250)/(1000+1000)
        assert_relative_eq!(stats.bandwidth_savings(), 62.5, epsilon = 1e-6);

        assert_relative_eq!(stats.best_compression_ratio, 0.25, epsilon = 1e-6);
        assert_relative_eq!(stats.worst_compression_ratio, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_compression_roundtrip() {
        let strategies = vec![
            CompressionStrategy::None,
            CompressionStrategy::TopK { k: 2 },
            CompressionStrategy::RandomK { k: 2 },
            CompressionStrategy::Threshold { threshold: 1.5 },
            CompressionStrategy::Quantization { bits: 4 },
        ];

        let gradients = vec![
            Array1::from_vec(vec![1.0, 2.5, 0.5, 3.0]),
            Array1::from_vec(vec![0.1, 4.0]),
        ];

        for strategy in strategies {
            let mut compressor = GradientCompressor::new(strategy.clone());

            let compressed = compressor.compress(&gradients).expect("unwrap failed");
            let decompressed = compressor.decompress(&compressed).expect("unwrap failed");

            // Should decompress to same number of arrays
            assert_eq!(decompressed.len(), gradients.len());

            // Shapes should match
            for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
                assert_eq!(orig.shape(), decomp.shape());
            }

            // For lossless strategies, values should match exactly
            match strategy {
                CompressionStrategy::None => {
                    for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
                        for (&o, &d) in orig.iter().zip(decomp.iter()) {
                            assert_relative_eq!(o, d, epsilon = 1e-10);
                        }
                    }
                }
                _ => {
                    // For lossy strategies, just check that we get reasonable values
                    for decomp in &decompressed {
                        assert!(decomp.iter().all(|&x| x.is_finite()));
                    }
                }
            }
        }
    }

    #[test]
    fn test_compression_invalid_configs() {
        // Invalid quantization bits
        let strategy = CompressionStrategy::Quantization { bits: 64 };
        let mut compressor = GradientCompressor::new(strategy);

        let gradients = vec![Array1::from_vec(vec![1.0, 2.0])];
        assert!(compressor.compress(&gradients).is_err());

        // Invalid decompression data
        let valid_compressor: GradientCompressor<f64, scirs2_core::ndarray::Ix1> =
            GradientCompressor::new(CompressionStrategy::None);
        let invalid_compressed = CompressedGradient {
            data: vec![1, 2, 3], // Insufficient data
            metadata: CompressionMetadata {
                strategy: CompressionStrategy::None,
                compression_ratio: 1.0,
                nnz_count: 1,
                scale_factors: vec![],
                extra_data: vec![],
            },
            shapes: vec![vec![2]],
        };

        assert!(valid_compressor.decompress(&invalid_compressed).is_err());
    }

    #[test]
    fn test_distributed_with_compression() {
        // Test parameter server with compressed gradients
        let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);
        let initialparams = vec![Array1::from_vec(vec![0.0, 0.0])];
        server.initialize(&initialparams).expect("unwrap failed");

        let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 1 });

        // Create gradients and compress them
        let gradients1 = vec![Array1::from_vec(vec![1.0, 3.0])]; // Top-1 should keep 3.0
        let gradients2 = vec![Array1::from_vec(vec![2.0, 1.0])]; // Top-1 should keep 2.0

        let compressed1 = compressor.compress(&gradients1).expect("unwrap failed");
        let compressed2 = compressor.compress(&gradients2).expect("unwrap failed");

        let decompressed1 = compressor.decompress(&compressed1).expect("unwrap failed");
        let decompressed2 = compressor.decompress(&compressed2).expect("unwrap failed");

        // Submit decompressed gradients to server
        server
            .submit_update(0, decompressed1)
            .expect("unwrap failed");
        server
            .submit_update(1, decompressed2)
            .expect("unwrap failed");

        let global_params = server.get_global_parameters();

        // Should have averaged the compressed gradients
        // Node 0 contributes [0, 3.0], Node 1 contributes [2.0, 0]
        // Average: [1.0, 1.5]
        assert_relative_eq!(global_params[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(global_params[0][1], 1.5, epsilon = 1e-6);
    }
}
