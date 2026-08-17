use super::averaging::{AveragingStrategy, ParameterAverager};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

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
