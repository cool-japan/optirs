use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
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
