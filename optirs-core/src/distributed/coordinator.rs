use super::averaging::AveragingStrategy;
use super::parameter_server::ParameterServer;
use crate::error::Result;
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

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
