//! Message-Passing Graph Neural Network Optimizer
//!
//! This module implements a learned optimizer whose update rule is computed by a
//! message-passing graph neural network (GNN) with a gated recurrent (GRU) node
//! update, in the spirit of Gated Graph Neural Networks (Li et al., 2016) applied
//! to the "learning to optimize" setting (Andrychowicz et al., 2016).
//!
//! # Overview
//!
//! The optimizer treats the flat parameter vector as the nodes of a graph:
//!
//! 1. **Graph construction.** The parameter vector is partitioned into
//!    `num_nodes` contiguous chunks (clamped to the parameter count). Each chunk
//!    becomes a graph node that aggregates gradient statistics over its slice. The
//!    inter-node adjacency is one of [`GraphTopology::Chain`],
//!    [`GraphTopology::FullyConnected`], or [`GraphTopology::KNearest`].
//! 2. **Node features.** For every node, per step, we recompute mean / abs-mean
//!    gradient, gradient variance, the bias-corrected EMA of the gradient
//!    (momentum), the bias-corrected EMA of the squared gradient (RMS), the sign
//!    agreement of the chunk, and a running summary of the node's recent updates.
//! 3. **Message passing.** For `num_message_rounds` rounds, every node maps its
//!    persistent hidden state through a learned linear map plus nonlinearity to
//!    produce a message, aggregates incoming neighbour messages (mean or sum), and
//!    updates its hidden state through a real GRU cell (reset gate, update gate,
//!    candidate, convex blend).
//! 4. **Readout.** Each node's final hidden state is mapped through a learned
//!    linear layer and a logistic to a strictly-positive per-node step scale in
//!    `(0, 1)`. The parameter update applies
//!    `params[chunk] -= base_lr * step_scale * effective_grad`, where
//!    `effective_grad` is the bias-corrected EMA-smoothed gradient.
//!
//! The learned weights (feature encoder, message map, GRU weights
//! `W_z, U_z, W_r, U_r, W_h, U_h`, and readout) are initialised deterministically
//! from a seed via [`scirs2_core::random`] and held fixed during [`AdvancedOptimizer::step`]
//! (this is the genuine forward / inference optimizer; the persistent per-node
//! EMAs and GRU hidden vectors evolve across steps). The architecture is laid out
//! so that meta-training operates purely through the flat weight-vector surface
//! in [`meta_training`]: see [`GnnOptimizer::weight_vector`],
//! [`GnnOptimizer::set_weight_vector`] and [`GnnOptimizer::reset_state`], driven
//! by [`crate::es_meta_training::EsMetaTrainer`].

/// Meta-training of the learned weights by evolution strategies.
///
/// F75: everything below computes a real update from real learned weights, but
/// nothing ever *trained* those weights — they stayed at their seeded draw
/// forever. [`meta_training::GnnMetaTrainer`] is the entry point that changes
/// that, and it operates through the public
/// [`GnnOptimizer::weight_vector`] / [`GnnOptimizer::set_weight_vector`] /
/// [`GnnOptimizer::reset_state`] surface it defines.
pub mod meta_training;

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng};
use std::fmt::Debug;

use crate::domain_optimizers::{clip_grad_norm, l2_norm, AdvancedOptimizer, OptimizerStateInfo};
use crate::error::{OptimError, Result};

/// Number of scalar features extracted per graph node, every step.
///
/// The features are, in order: mean gradient, abs-mean gradient, gradient
/// variance, momentum (bias-corrected gradient EMA), RMS (sqrt of bias-corrected
/// squared-gradient EMA), sign agreement, and the update-history summary.
const NODE_FEATURE_DIM: usize = 7;

/// Connectivity used to build the inter-node (parameter/layer) graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphTopology {
    /// Consecutive nodes are linked: node `n` neighbours `n-1` and `n+1`.
    Chain,
    /// Every node is linked to every other node.
    FullyConnected,
    /// Node `n` is linked to the `k` nearest nodes on each side in index space.
    KNearest {
        /// Number of neighbours on each side (`n-k..n` and `n+1..=n+k`).
        k: usize,
    },
}

/// How incoming neighbour messages are combined at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageAggregation {
    /// Average of incoming messages (degree-normalised).
    Mean,
    /// Plain sum of incoming messages.
    Sum,
}

/// Nonlinearity applied to a node's outgoing message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageActivation {
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    Relu,
}

/// Configuration for [`GnnOptimizer`].
///
/// The numeric hyper-parameters are stored as `f64` and converted to the
/// optimizer's scalar type `T` when the optimizer is constructed.
#[derive(Debug, Clone)]
pub struct GnnOptimizerConfig {
    /// Requested number of graph nodes (clamped to the parameter count at run time).
    pub num_nodes: usize,
    /// Dimension of the GRU hidden state and message space.
    pub hidden_dim: usize,
    /// Number of synchronous message-passing rounds per optimization step.
    pub num_message_rounds: usize,
    /// Inter-node graph connectivity.
    pub topology: GraphTopology,
    /// Neighbour message aggregation rule.
    pub aggregation: MessageAggregation,
    /// Message nonlinearity.
    pub activation: MessageActivation,
    /// Base learning rate applied before the learned per-node step scale.
    pub base_lr: f64,
    /// EMA decay for the gradient (momentum), often called `beta1`.
    pub momentum_decay: f64,
    /// EMA decay for the squared gradient (RMS), often called `beta2`.
    pub rms_decay: f64,
    /// EMA decay for the per-node update-history feature.
    pub history_decay: f64,
    /// Maximum gradient L2 norm; larger gradients are rescaled before use.
    pub max_grad_norm: f64,
    /// Seed for deterministic weight initialisation.
    pub seed: u64,
}

impl Default for GnnOptimizerConfig {
    fn default() -> Self {
        Self {
            num_nodes: 16,
            hidden_dim: 8,
            num_message_rounds: 2,
            topology: GraphTopology::Chain,
            aggregation: MessageAggregation::Mean,
            activation: MessageActivation::Tanh,
            base_lr: 0.01,
            momentum_decay: 0.9,
            rms_decay: 0.999,
            history_decay: 0.9,
            max_grad_norm: 10.0,
            seed: 0x5EED_C0DE,
        }
    }
}

/// Logistic (sigmoid) activation for a scalar.
fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// Convert an `f64` into the optimizer scalar, falling back to zero on overflow.
fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

/// Glorot-uniform initialisation of a `rows x cols` weight matrix.
fn init_matrix<T, R>(rng: &mut Random<R>, rows: usize, cols: usize) -> Array2<T>
where
    T: Float,
    R: Rng,
{
    let limit = (6.0 / (rows as f64 + cols as f64)).sqrt();
    Array2::from_shape_fn((rows, cols), |_| {
        let value: f64 = rng.random_range(-limit..limit);
        cast(value)
    })
}

/// Glorot-uniform initialisation of a length-`len` readout vector.
fn init_vector<T, R>(rng: &mut Random<R>, len: usize) -> Array1<T>
where
    T: Float,
    R: Rng,
{
    let limit = (6.0 / (len as f64 + 1.0)).sqrt();
    Array1::from_shape_fn(len, |_| {
        let value: f64 = rng.random_range(-limit..limit);
        cast(value)
    })
}

/// A genuine Gated Recurrent Unit cell.
///
/// With input `a` and previous hidden `h` the cell computes
/// `z = σ(W_z a + U_z h + b_z)` (update gate),
/// `r = σ(W_r a + U_r h + b_r)` (reset gate),
/// `h̃ = tanh(W_h a + U_h (r ⊙ h) + b_h)` (candidate), and returns the convex blend
/// `(1 - z) ⊙ h + z ⊙ h̃`.
#[derive(Debug, Clone)]
struct GruCell<T: Float + Debug + Send + Sync + 'static> {
    w_z: Array2<T>,
    u_z: Array2<T>,
    b_z: Array1<T>,
    w_r: Array2<T>,
    u_r: Array2<T>,
    b_r: Array1<T>,
    w_h: Array2<T>,
    u_h: Array2<T>,
    b_h: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> GruCell<T> {
    /// Deterministically initialise all gate weights for a hidden size of `hidden`.
    ///
    /// The input and hidden dimensions are both `hidden` (messages live in the
    /// hidden space), so every gate weight is a square `hidden x hidden` matrix.
    fn new<R: Rng>(rng: &mut Random<R>, hidden: usize) -> Self {
        Self {
            w_z: init_matrix(rng, hidden, hidden),
            u_z: init_matrix(rng, hidden, hidden),
            b_z: Array1::zeros(hidden),
            w_r: init_matrix(rng, hidden, hidden),
            u_r: init_matrix(rng, hidden, hidden),
            b_r: Array1::zeros(hidden),
            w_h: init_matrix(rng, hidden, hidden),
            u_h: init_matrix(rng, hidden, hidden),
            b_h: Array1::zeros(hidden),
        }
    }

    /// Apply one GRU update, returning the new hidden state.
    fn step(&self, input: &Array1<T>, hidden: &Array1<T>) -> Array1<T> {
        let update_gate =
            (self.w_z.dot(input) + self.u_z.dot(hidden) + &self.b_z).mapv(|x| sigmoid(x));
        let reset_gate =
            (self.w_r.dot(input) + self.u_r.dot(hidden) + &self.b_r).mapv(|x| sigmoid(x));
        let reset_hidden = &reset_gate * hidden;
        let candidate =
            (self.w_h.dot(input) + self.u_h.dot(&reset_hidden) + &self.b_h).mapv(|x| x.tanh());
        let keep = update_gate.mapv(|z| T::one() - z);
        &keep * hidden + &update_gate * &candidate
    }
}

/// The full set of learned GNN weights, held fixed during inference.
#[derive(Debug, Clone)]
struct GnnWeights<T: Float + Debug + Send + Sync + 'static> {
    /// Node-feature encoder: `hidden x NODE_FEATURE_DIM`.
    w_in: Array2<T>,
    b_in: Array1<T>,
    /// Message map applied to a node's hidden state: `hidden x hidden`.
    w_msg: Array2<T>,
    b_msg: Array1<T>,
    /// GRU node-update cell.
    gru: GruCell<T>,
    /// Readout weight (length `hidden`) and bias producing a scalar logit.
    w_out: Array1<T>,
    b_out: T,
}

impl<T: Float + Debug + Send + Sync + 'static> GnnWeights<T> {
    /// Deterministically initialise every learned weight.
    fn new<R: Rng>(rng: &mut Random<R>, hidden: usize) -> Self {
        Self {
            w_in: init_matrix(rng, hidden, NODE_FEATURE_DIM),
            b_in: Array1::zeros(hidden),
            w_msg: init_matrix(rng, hidden, hidden),
            b_msg: Array1::zeros(hidden),
            gru: GruCell::new(rng, hidden),
            w_out: init_vector(rng, hidden),
            b_out: T::zero(),
        }
    }

    /// Encode a node feature vector into the hidden space (tanh nonlinearity).
    fn encode(&self, feature: &Array1<T>) -> Array1<T> {
        (self.w_in.dot(feature) + &self.b_in).mapv(|x| x.tanh())
    }

    /// Compute a node's outgoing message from its hidden state.
    fn message(&self, hidden: &Array1<T>, activation: MessageActivation) -> Array1<T> {
        let linear = self.w_msg.dot(hidden) + &self.b_msg;
        match activation {
            MessageActivation::Tanh => linear.mapv(|x| x.tanh()),
            MessageActivation::Relu => linear.mapv(|x| if x > T::zero() { x } else { T::zero() }),
        }
    }

    /// Map a final hidden state to a strictly-positive step scale in `(0, 1)`.
    fn readout(&self, hidden: &Array1<T>) -> T {
        sigmoid(self.w_out.dot(hidden) + self.b_out)
    }
}

/// Aggregate the messages of a node's neighbours.
fn aggregate_messages<T>(
    neighbors: &[usize],
    messages: &[Array1<T>],
    aggregation: MessageAggregation,
) -> Array1<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    let dim = messages.first().map_or(0, Array1::len);
    if neighbors.is_empty() {
        return Array1::zeros(dim);
    }
    let sum = neighbors
        .iter()
        .map(|&j| &messages[j])
        .fold(Array1::<T>::zeros(dim), |acc, message| acc + message);
    match aggregation {
        MessageAggregation::Sum => sum,
        MessageAggregation::Mean => {
            let count = cast::<T>(neighbors.len() as f64);
            sum.mapv(|x| x / count)
        }
    }
}

/// Build the adjacency (neighbour index lists) for the chosen topology.
fn build_adjacency(num_nodes: usize, topology: GraphTopology) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); num_nodes];
    match topology {
        GraphTopology::Chain => {
            for (n, neighbors) in adjacency.iter_mut().enumerate() {
                if n > 0 {
                    neighbors.push(n - 1);
                }
                if n + 1 < num_nodes {
                    neighbors.push(n + 1);
                }
            }
        }
        GraphTopology::FullyConnected => {
            for (n, neighbors) in adjacency.iter_mut().enumerate() {
                for j in 0..num_nodes {
                    if j != n {
                        neighbors.push(j);
                    }
                }
            }
        }
        GraphTopology::KNearest { k } => {
            for (n, neighbors) in adjacency.iter_mut().enumerate() {
                for d in 1..=k {
                    if n >= d {
                        neighbors.push(n - d);
                    }
                    if n + d < num_nodes {
                        neighbors.push(n + d);
                    }
                }
            }
        }
    }
    adjacency
}

/// Partition `param_len` indices into `num_nodes` contiguous, near-equal chunks.
fn build_chunks(param_len: usize, num_nodes: usize) -> Vec<(usize, usize)> {
    let base = param_len / num_nodes;
    let remainder = param_len % num_nodes;
    let mut bounds = Vec::with_capacity(num_nodes);
    let mut start = 0;
    for n in 0..num_nodes {
        let size = base + usize::from(n < remainder);
        bounds.push((start, start + size));
        start += size;
    }
    bounds
}

/// A learned optimizer driven by a message-passing GNN with a GRU node update.
///
/// Implements [`AdvancedOptimizer`]. The learned weights are fixed; the per-node
/// EMAs and GRU hidden vectors are persistent state that evolves across steps.
#[derive(Debug, Clone)]
pub struct GnnOptimizer<T: Float + Debug + Send + Sync + 'static> {
    // --- configuration (fixed) ---
    requested_num_nodes: usize,
    hidden_dim: usize,
    num_message_rounds: usize,
    topology: GraphTopology,
    aggregation: MessageAggregation,
    activation: MessageActivation,
    base_lr: T,
    momentum_decay: T,
    rms_decay: T,
    history_decay: T,
    max_grad_norm: T,
    norm_ema_decay: T,
    seed: u64,

    // --- learned weights (fixed during step) ---
    weights: GnnWeights<T>,

    // --- graph layout (depends on the actual parameter length) ---
    param_len: usize,
    num_nodes: usize,
    chunk_bounds: Vec<(usize, usize)>,
    adjacency: Vec<Vec<usize>>,

    // --- persistent optimization state ---
    grad_ema: Array1<T>,
    grad_sq_ema: Array1<T>,
    node_hidden: Vec<Array1<T>>,
    node_history: Array1<T>,

    // --- bookkeeping ---
    step_count: usize,
    current_lr: T,
    grad_norm_ema: T,
}

impl<T: Float + Debug + Send + Sync + 'static> GnnOptimizer<T> {
    /// Construct a new optimizer from a configuration.
    ///
    /// The learned weights are initialised deterministically from `config.seed`.
    /// Graph-dependent state is created lazily on the first [`AdvancedOptimizer::step`]
    /// once the parameter length is known.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidConfig`] if a hyper-parameter is out of range.
    pub fn new(config: GnnOptimizerConfig) -> Result<Self> {
        if config.hidden_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "hidden_dim must be at least 1".to_string(),
            ));
        }
        if config.num_message_rounds == 0 {
            return Err(OptimError::InvalidConfig(
                "num_message_rounds must be at least 1".to_string(),
            ));
        }
        if config.num_nodes == 0 {
            return Err(OptimError::InvalidConfig(
                "num_nodes must be at least 1".to_string(),
            ));
        }
        if !(config.base_lr.is_finite() && config.base_lr > 0.0) {
            return Err(OptimError::InvalidConfig(
                "base_lr must be a positive finite value".to_string(),
            ));
        }
        if config.max_grad_norm <= 0.0 || !config.max_grad_norm.is_finite() {
            return Err(OptimError::InvalidConfig(
                "max_grad_norm must be a positive finite value".to_string(),
            ));
        }
        for (name, value) in [
            ("momentum_decay", config.momentum_decay),
            ("rms_decay", config.rms_decay),
            ("history_decay", config.history_decay),
        ] {
            if !(0.0..1.0).contains(&value) {
                return Err(OptimError::InvalidConfig(format!(
                    "{name} must lie in [0, 1), got {value}"
                )));
            }
        }

        let mut rng = Random::seed(config.seed);
        let weights = GnnWeights::new(&mut rng, config.hidden_dim);

        Ok(Self {
            requested_num_nodes: config.num_nodes,
            hidden_dim: config.hidden_dim,
            num_message_rounds: config.num_message_rounds,
            topology: config.topology,
            aggregation: config.aggregation,
            activation: config.activation,
            base_lr: cast(config.base_lr),
            momentum_decay: cast(config.momentum_decay),
            rms_decay: cast(config.rms_decay),
            history_decay: cast(config.history_decay),
            max_grad_norm: cast(config.max_grad_norm),
            norm_ema_decay: cast(0.99),
            seed: config.seed,
            weights,
            param_len: 0,
            num_nodes: 0,
            chunk_bounds: Vec::new(),
            adjacency: Vec::new(),
            grad_ema: Array1::zeros(0),
            grad_sq_ema: Array1::zeros(0),
            node_hidden: Vec::new(),
            node_history: Array1::zeros(0),
            step_count: 0,
            current_lr: cast(config.base_lr),
            grad_norm_ema: T::zero(),
        })
    }

    /// The actual number of graph nodes (after clamping to the parameter count).
    ///
    /// Returns `0` before the first step, when the parameter length is unknown.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The GRU hidden / message dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// The seed used for deterministic weight initialisation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// (Re)initialise all graph-dependent and persistent state for `param_len`.
    fn initialize_for(&mut self, param_len: usize) {
        let num_nodes = self.requested_num_nodes.min(param_len).max(1);
        self.param_len = param_len;
        self.num_nodes = num_nodes;
        self.chunk_bounds = build_chunks(param_len, num_nodes);
        self.adjacency = build_adjacency(num_nodes, self.topology);
        self.grad_ema = Array1::zeros(param_len);
        self.grad_sq_ema = Array1::zeros(param_len);
        self.node_hidden = (0..num_nodes)
            .map(|_| Array1::zeros(self.hidden_dim))
            .collect();
        self.node_history = Array1::zeros(num_nodes);
        self.step_count = 0;
        self.grad_norm_ema = T::zero();
    }

    /// Build the per-node encoded feature vectors from the current gradient and
    /// the (already updated) bias-corrected EMAs.
    fn encode_nodes(
        &self,
        grad: &Array1<T>,
        m_hat: &Array1<T>,
        v_hat: &Array1<T>,
    ) -> Vec<Array1<T>> {
        let mut encoded = Vec::with_capacity(self.num_nodes);
        for (n, &(start, end)) in self.chunk_bounds.iter().enumerate() {
            let size = cast::<T>((end - start) as f64);
            let chunk = grad.slice(s![start..end]);

            let mean_grad = chunk.sum() / size;
            let abs_mean_grad = chunk.iter().fold(T::zero(), |a, &x| a + x.abs()) / size;
            let variance = chunk.iter().fold(T::zero(), |a, &x| {
                let d = x - mean_grad;
                a + d * d
            }) / size;

            let momentum = m_hat.slice(s![start..end]).sum() / size;
            let rms = v_hat
                .slice(s![start..end])
                .iter()
                .fold(T::zero(), |a, &x| a + x.max(T::zero()).sqrt())
                / size;

            let mean_sign = mean_grad.signum();
            let sign_agreement = chunk
                .iter()
                .fold(T::zero(), |a, &x| a + x.signum() * mean_sign)
                / size;

            let history = self.node_history[n];

            let feature = Array1::from_vec(vec![
                mean_grad,
                abs_mean_grad,
                variance,
                momentum,
                rms,
                sign_agreement,
                history,
            ]);
            encoded.push(self.weights.encode(&feature));
        }
        encoded
    }

    /// Run `rounds` synchronous message-passing rounds, updating `node_hidden`.
    ///
    /// Each round computes every node's message from the hidden snapshot, then
    /// every node aggregates its neighbours' messages, injects its encoded node
    /// feature, and applies the GRU update.
    fn message_passing(
        weights: &GnnWeights<T>,
        adjacency: &[Vec<usize>],
        encoded: &[Array1<T>],
        node_hidden: &mut [Array1<T>],
        rounds: usize,
        aggregation: MessageAggregation,
        activation: MessageActivation,
    ) {
        for _ in 0..rounds {
            let messages: Vec<Array1<T>> = node_hidden
                .iter()
                .map(|hidden| weights.message(hidden, activation))
                .collect();
            for (n, neighbors) in adjacency.iter().enumerate() {
                let aggregated = aggregate_messages(neighbors, &messages, aggregation);
                let input = &aggregated + &encoded[n];
                node_hidden[n] = weights.gru.step(&input, &node_hidden[n]);
            }
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> AdvancedOptimizer<T> for GnnOptimizer<T> {
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {
        if params.len() != gradients.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter length {} != gradient length {}",
                params.len(),
                gradients.len()
            )));
        }
        if params.is_empty() {
            return Err(OptimError::InsufficientData(
                "Empty parameter array".to_string(),
            ));
        }

        if self.param_len != params.len() {
            self.initialize_for(params.len());
        }

        // 1. Clip the gradient and track its EMA norm.
        let grad = clip_grad_norm(gradients, self.max_grad_norm);
        let grad_norm = l2_norm(&grad);
        self.grad_norm_ema =
            self.norm_ema_decay * self.grad_norm_ema + (T::one() - self.norm_ema_decay) * grad_norm;

        // 2. Update the per-element gradient and squared-gradient EMAs.
        let beta1 = self.momentum_decay;
        let beta2 = self.rms_decay;
        self.grad_ema = self.grad_ema.mapv(|m| m * beta1) + &grad.mapv(|g| g * (T::one() - beta1));
        self.grad_sq_ema =
            self.grad_sq_ema.mapv(|v| v * beta2) + &grad.mapv(|g| g * g * (T::one() - beta2));

        // 3. Bias-correct the EMAs (Adam-style), giving the effective direction.
        let step = self.step_count + 1;
        let bias1 = T::one() - beta1.powi(step as i32);
        let bias2 = T::one() - beta2.powi(step as i32);
        let m_hat = self.grad_ema.mapv(|m| m / bias1);
        let v_hat = self.grad_sq_ema.mapv(|v| v / bias2);

        // 4. Encode node features, then run message passing + GRU updates.
        let encoded = self.encode_nodes(&grad, &m_hat, &v_hat);
        Self::message_passing(
            &self.weights,
            &self.adjacency,
            &encoded,
            &mut self.node_hidden,
            self.num_message_rounds,
            self.aggregation,
            self.activation,
        );

        // 5. Readout: per-node strictly-positive step scale in (0, 1).
        let step_scales: Vec<T> = self
            .node_hidden
            .iter()
            .map(|hidden| self.weights.readout(hidden))
            .collect();

        // 6. Broadcast the per-node scale and apply the EMA-smoothed update.
        let mut scale_vector = Array1::zeros(self.param_len);
        for (n, &(start, end)) in self.chunk_bounds.iter().enumerate() {
            scale_vector.slice_mut(s![start..end]).fill(step_scales[n]);
        }
        let base_lr = self.base_lr;
        let update = (&scale_vector * &m_hat).mapv(|u| u * base_lr);
        let new_params = params - &update;

        // 7. Update the per-node update-history summary (EMA of the step scale).
        let history_decay = self.history_decay;
        for (history, &scale) in self.node_history.iter_mut().zip(step_scales.iter()) {
            *history = history_decay * *history + (T::one() - history_decay) * scale;
        }

        // 8. Effective learning rate = base_lr * mean step scale; advance counter.
        let scale_count = cast::<T>(step_scales.len() as f64);
        let mean_scale = step_scales.iter().fold(T::zero(), |a, &s| a + s) / scale_count;
        self.current_lr = base_lr * mean_scale;
        self.step_count += 1;

        Ok(new_params)
    }

    fn get_learning_rate(&self) -> T {
        self.current_lr
    }

    fn set_learning_rate(&mut self, lr: T) {
        self.base_lr = lr;
        self.current_lr = lr;
    }

    fn name(&self) -> &str {
        "GNNOptimizer"
    }

    fn get_state(&self) -> OptimizerStateInfo<T> {
        OptimizerStateInfo {
            step_count: self.step_count,
            current_lr: self.current_lr,
            grad_norm_ema: self.grad_norm_ema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_optimizer(seed: u64) -> GnnOptimizer<f64> {
        GnnOptimizer::new(GnnOptimizerConfig {
            seed,
            ..Default::default()
        })
        .expect("default configuration must be valid")
    }

    fn l2(a: &Array1<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_gnn_preserves_shape() {
        for len in [1usize, 3, 7, 16, 40, 100] {
            let mut opt = make_optimizer(7);
            let params = Array1::from_shape_fn(len, |i| (i as f64) * 0.1 + 0.5);
            let grads = Array1::from_shape_fn(len, |i| (i as f64) * 0.01 + 0.2);
            let out = opt.step(&params, &grads).expect("step should succeed");
            assert_eq!(out.len(), params.len(), "output length must match input");
        }
    }

    #[test]
    fn test_gnn_deterministic_same_seed() {
        let params = Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, 0.5, 0.25, 0.1]);
        let grads = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.05, 0.02, 0.3]);

        let mut a = make_optimizer(123);
        let mut b = make_optimizer(123);
        let out_a = a.step(&params, &grads).expect("a step");
        let out_b = b.step(&params, &grads).expect("b step");

        for (x, y) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (x - y).abs() < 1e-12,
                "same seed must give identical output: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_gnn_different_seed_differs() {
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = Array1::from_vec(vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);

        let mut a = make_optimizer(1);
        let mut b = make_optimizer(2);
        let out_a = a.step(&params, &grads).expect("a step");
        let out_b = b.step(&params, &grads).expect("b step");

        let diff: f64 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            diff > 0.0,
            "different seeds should produce different updates"
        );
    }

    #[test]
    fn test_gnn_convergence_quadratic() {
        // f(x) = ||x||^2, grad = 2x. Repeated steps must strictly reduce ||x||.
        let mut opt = make_optimizer(42);
        let mut x = Array1::from_vec(vec![
            3.0, -3.0, 2.5, -2.5, 4.0, -1.0, 1.5, -2.0, 3.5, -0.5, 2.0, -1.5, 1.0, -3.0, 2.0, -2.5,
        ]);
        let initial = l2(&x);
        let mut prev = initial;

        for iteration in 0..400 {
            let grad = x.mapv(|v| 2.0 * v);
            x = opt.step(&x, &grad).expect("step should succeed");
            let norm = l2(&x);
            assert!(
                norm < prev,
                "norm must strictly decrease at iteration {iteration}: {prev} -> {norm}"
            );
            prev = norm;
        }

        assert!(
            prev < 0.8 * initial,
            "norm should reduce substantially over many iterations: {initial} -> {prev}"
        );
    }

    #[test]
    fn test_gnn_all_topologies_run() {
        let topologies = [
            GraphTopology::Chain,
            GraphTopology::FullyConnected,
            GraphTopology::KNearest { k: 2 },
        ];
        for topology in topologies {
            let cfg = GnnOptimizerConfig {
                topology,
                seed: 9,
                ..Default::default()
            };
            let mut opt = GnnOptimizer::new(cfg).expect("config valid");
            let params = Array1::from_shape_fn(20, |i| 1.0 + 0.1 * (i as f64));
            let grads = Array1::from_shape_fn(20, |i| 0.2 - 0.01 * (i as f64));
            let out = opt.step(&params, &grads).expect("step should succeed");
            assert_eq!(out.len(), 20);
            let moved: f64 = out
                .iter()
                .zip(params.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                moved > 0.0,
                "topology {topology:?} should update parameters"
            );
        }
    }

    #[test]
    fn test_gnn_knearest_topology_converges() {
        let cfg = GnnOptimizerConfig {
            topology: GraphTopology::KNearest { k: 3 },
            aggregation: MessageAggregation::Sum,
            activation: MessageActivation::Relu,
            seed: 77,
            ..Default::default()
        };
        let mut opt = GnnOptimizer::new(cfg).expect("config valid");
        let mut x = Array1::from_shape_fn(24, |i| 2.0 - 0.05 * (i as f64));
        let initial = l2(&x);
        for _ in 0..200 {
            let grad = x.mapv(|v| 2.0 * v);
            x = opt.step(&x, &grad).expect("step should succeed");
        }
        assert!(l2(&x) < initial, "k-nearest variant should reduce the norm");
    }

    #[test]
    fn test_gnn_num_nodes_clamped() {
        let cfg = GnnOptimizerConfig {
            num_nodes: 1000,
            seed: 5,
            ..Default::default()
        };
        let mut opt = GnnOptimizer::new(cfg).expect("config valid");
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let out = opt.step(&params, &grads).expect("step should succeed");
        assert_eq!(out.len(), 4);
        assert_eq!(
            opt.num_nodes(),
            4,
            "num_nodes must clamp to the parameter count"
        );
    }

    #[test]
    fn test_gnn_single_parameter() {
        let mut opt = make_optimizer(13);
        let params = Array1::from_vec(vec![5.0]);
        let grads = Array1::from_vec(vec![2.0 * 5.0]);
        let out = opt.step(&params, &grads).expect("step should succeed");
        assert_eq!(out.len(), 1);
        assert_eq!(opt.num_nodes(), 1);
        assert!(
            out[0].abs() < params[0].abs(),
            "single param should move toward 0"
        );
    }

    #[test]
    fn test_gnn_trait_methods() {
        let mut opt = make_optimizer(11);
        assert_eq!(opt.name(), "GNNOptimizer");

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let grads = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]);
        let _ = opt.step(&params, &grads).expect("step should succeed");

        let state = opt.get_state();
        assert_eq!(state.step_count, 1);
        assert!(state.grad_norm_ema > 0.0);
        assert!(state.current_lr > 0.0);

        opt.set_learning_rate(0.5);
        assert!((opt.get_learning_rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gnn_dimension_mismatch() {
        let mut opt = make_optimizer(3);
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        assert!(opt.step(&params, &grads).is_err());
    }

    #[test]
    fn test_gnn_empty_params_error() {
        let mut opt = make_optimizer(3);
        let params = Array1::<f64>::zeros(0);
        let grads = Array1::<f64>::zeros(0);
        assert!(opt.step(&params, &grads).is_err());
    }

    #[test]
    fn test_gnn_invalid_config() {
        assert!(GnnOptimizer::<f64>::new(GnnOptimizerConfig {
            hidden_dim: 0,
            ..Default::default()
        })
        .is_err());
        assert!(GnnOptimizer::<f64>::new(GnnOptimizerConfig {
            momentum_decay: 1.0,
            ..Default::default()
        })
        .is_err());
        assert!(GnnOptimizer::<f64>::new(GnnOptimizerConfig {
            base_lr: 0.0,
            ..Default::default()
        })
        .is_err());
        assert!(GnnOptimizer::<f64>::new(GnnOptimizerConfig {
            num_message_rounds: 0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_gnn_persistent_hidden_state_evolves() {
        // The GRU hidden vectors are persistent and must change across steps.
        let mut opt = make_optimizer(7);
        let params = Array1::from_shape_fn(10, |i| 0.5 + 0.1 * (i as f64));
        let grads = Array1::from_shape_fn(10, |i| 0.3 - 0.02 * (i as f64));

        let _ = opt.step(&params, &grads).expect("first step");
        let hidden_after_first: Vec<f64> =
            opt.node_hidden.iter().flat_map(|h| h.to_vec()).collect();

        let _ = opt.step(&params, &grads).expect("second step");
        let hidden_after_second: Vec<f64> =
            opt.node_hidden.iter().flat_map(|h| h.to_vec()).collect();

        let changed: f64 = hidden_after_first
            .iter()
            .zip(hidden_after_second.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(changed > 0.0, "GRU hidden state should evolve across steps");
        assert_eq!(opt.get_state().step_count, 2);
    }

    #[test]
    fn test_gnn_step_scales_in_unit_interval() {
        // Every per-node readout must be a strictly-positive scale in (0, 1),
        // which is what guarantees monotone descent on the quadratic.
        let opt = make_optimizer(31);
        let hidden = Array1::from_vec(vec![0.4; opt.hidden_dim()]);
        let scale = opt.weights.readout(&hidden);
        assert!(scale > 0.0 && scale < 1.0, "step scale must lie in (0, 1)");
    }
}
