// Reinforcement learning-based search using policy gradients
//
// An LSTM controller autoregressively emits an architecture: the first decoding
// step chooses a component type, each following step chooses a discretized bin
// for one of that component's hyperparameters. The log-probability of every
// sampled action is recorded, and the controller is trained with REINFORCE
// (score-function policy gradient) using a learned baseline for variance
// reduction, an entropy bonus for exploration, and global-norm gradient
// clipping.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Random, Rng as SCRRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::nas_engine::config::{ComponentType as ConfigComponentType, ParameterRange};
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::{SearchStrategy, SearchStrategyStatistics};

/// Width of the controller's state/input vector.
pub const CONTROLLER_INPUT_SIZE: usize = 64;

/// Width of the controller's logit vector. Each decoding step reads the prefix
/// of this vector matching that step's arity, so one output head serves both
/// the component-type choice and the hyperparameter-bin choices.
pub const CONTROLLER_OUTPUT_SIZE: usize = 64;

/// Number of bins a continuous hyperparameter range is discretized into.
pub const HYPERPARAMETER_BINS: usize = 10;

/// Maximum number of hyperparameters decoded for a single component.
const MAX_HYPERPARAMETERS: usize = 8;

/// Convert an `f64` constant into `T`, falling back to zero.
fn scalar<T: Float>(value: f64) -> T {
    scirs2_core::numeric::NumCast::from(value).unwrap_or_else(T::zero)
}

/// Numerically stable logistic function.
fn sigmoid<T: Float>(x: T) -> T {
    if x >= T::zero() {
        T::one() / (T::one() + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::one() + e)
    }
}

/// Numerically stable softmax over the first `active` entries of `logits`.
fn softmax<T: Float>(logits: &Array1<T>, active: usize) -> Vec<T> {
    let active = active.min(logits.len()).max(1);
    let mut max = T::neg_infinity();
    for i in 0..active {
        if logits[i] > max {
            max = logits[i];
        }
    }
    if !max.is_finite() {
        // Degenerate logits: fall back to the uniform distribution rather than
        // propagating NaN into the sampler.
        let uniform = T::one() / scalar::<T>(active as f64);
        return vec![uniform; active];
    }

    let mut exps = Vec::with_capacity(active);
    let mut sum = T::zero();
    for i in 0..active {
        let e = (logits[i] - max).exp();
        exps.push(e);
        sum = sum + e;
    }
    if !(sum > T::zero()) {
        let uniform = T::one() / scalar::<T>(active as f64);
        return vec![uniform; active];
    }
    for e in exps.iter_mut() {
        *e = *e / sum;
    }
    exps
}

/// Sample an index from a categorical distribution.
fn sample_categorical<T: Float>(
    probs: &[T],
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let draw = scalar::<T>(rng.random::<f64>());
    let mut cumulative = T::zero();
    for (idx, &p) in probs.iter().enumerate() {
        cumulative = cumulative + p;
        if draw <= cumulative {
            return idx;
        }
    }
    probs.len() - 1
}

/// Shannon entropy of a categorical distribution, in nats.
fn entropy<T: Float>(probs: &[T]) -> T {
    let floor = scalar::<T>(1e-12);
    let mut h = T::zero();
    for &p in probs {
        if p > floor {
            h = h - p * p.ln();
        }
    }
    h
}

/// Reinforcement learning-based search using policy gradients
pub struct ReinforcementLearningSearch<T: Float + Debug + Send + Sync + 'static> {
    controller_network: ControllerNetwork<T>,
    experience_buffer: ExperienceBuffer<T>,
    policy_optimizer: PolicyOptimizer<T>,
    baseline_predictor: BaselinePredictor<T>,
    epsilon: f64,
    exploration_decay: f64,
    statistics: SearchStrategyStatistics<T>,
    entropy_bonus: f64,
    /// Persistent RNG. A fresh `Random::default()` per call (the old behaviour)
    /// made every sampled architecture identical within a process.
    rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Trajectories awaiting their reward, paired FIFO with incoming results.
    pending_trajectories: VecDeque<Trajectory<T>>,
    /// Trajectories with a known reward, consumed by `train_controller`.
    ready_trajectories: Vec<(Trajectory<T>, T)>,
    /// Number of completed REINFORCE updates.
    updates_applied: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> Debug for ReinforcementLearningSearch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReinforcementLearningSearch")
            .field("controller_network", &self.controller_network)
            .field("epsilon", &self.epsilon)
            .field("entropy_bonus", &self.entropy_bonus)
            .field("updates_applied", &self.updates_applied)
            .field("pending", &self.pending_trajectories.len())
            .finish()
    }
}

/// One decoding step of a sampled trajectory, with everything needed to
/// recompute its gradient.
#[derive(Debug, Clone)]
struct StepRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Per-layer forward caches for this timestep.
    layers: Vec<LayerCache<T>>,
    /// Hidden state fed to the output projection.
    hidden_top: Array1<T>,
    /// Softmax probabilities over the active logit prefix.
    probs: Vec<T>,
    /// Index sampled at this step.
    action: usize,
    /// `ln p(action)` at sampling time.
    log_prob: T,
}

/// Forward-pass cache for one LSTM layer at one timestep.
#[derive(Debug, Clone)]
struct LayerCache<T: Float + Debug + Send + Sync + 'static> {
    input: Array1<T>,
    h_prev: Array1<T>,
    c_prev: Array1<T>,
    gate_i: Array1<T>,
    gate_f: Array1<T>,
    gate_g: Array1<T>,
    gate_o: Array1<T>,
    tanh_c: Array1<T>,
}

/// A full sampled architecture together with its decoding trace.
#[derive(Debug, Clone)]
struct Trajectory<T: Float + Debug + Send + Sync + 'static> {
    /// Encoded search-space state the episode started from.
    state: Array1<T>,
    steps: Vec<StepRecord<T>>,
    /// Component index chosen at step 0.
    component_action: usize,
}

/// Accumulated gradients with the same shape as [`ControllerNetwork`].
#[derive(Debug, Clone)]
struct ControllerGradients<T: Float + Debug + Send + Sync + 'static> {
    input_weights: Vec<Array2<T>>,
    recurrent_weights: Vec<Array2<T>>,
    biases: Vec<Array1<T>>,
    output_weights: Array2<T>,
    output_bias: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ControllerGradients<T> {
    fn zeros_like(network: &ControllerNetwork<T>) -> Self {
        Self {
            input_weights: network
                .input_weights
                .iter()
                .map(|w| Array2::zeros(w.dim()))
                .collect(),
            recurrent_weights: network
                .recurrent_weights
                .iter()
                .map(|w| Array2::zeros(w.dim()))
                .collect(),
            biases: network
                .biases
                .iter()
                .map(|b| Array1::zeros(b.len()))
                .collect(),
            output_weights: Array2::zeros(network.output_weights.dim()),
            output_bias: Array1::zeros(network.output_bias.len()),
        }
    }

    /// Global L2 norm across every gradient tensor.
    fn global_norm(&self) -> T {
        let mut acc = T::zero();
        for w in &self.input_weights {
            acc = acc + w.iter().fold(T::zero(), |a, &v| a + v * v);
        }
        for w in &self.recurrent_weights {
            acc = acc + w.iter().fold(T::zero(), |a, &v| a + v * v);
        }
        for b in &self.biases {
            acc = acc + b.iter().fold(T::zero(), |a, &v| a + v * v);
        }
        acc = acc
            + self
                .output_weights
                .iter()
                .fold(T::zero(), |a, &v| a + v * v);
        acc = acc + self.output_bias.iter().fold(T::zero(), |a, &v| a + v * v);
        acc.sqrt()
    }

    /// Scale every gradient tensor in place.
    fn scale(&mut self, factor: T) {
        for w in self.input_weights.iter_mut() {
            w.mapv_inplace(|v| v * factor);
        }
        for w in self.recurrent_weights.iter_mut() {
            w.mapv_inplace(|v| v * factor);
        }
        for b in self.biases.iter_mut() {
            b.mapv_inplace(|v| v * factor);
        }
        self.output_weights.mapv_inplace(|v| v * factor);
        self.output_bias.mapv_inplace(|v| v * factor);
    }
}

/// Controller network for RL-based search.
///
/// A stacked LSTM with an explicit input width. Layer 0 consumes a
/// `input_size`-wide vector, every following layer consumes the layer below's
/// `hidden_size`-wide hidden state. Each layer owns an input matrix
/// `(4 * hidden, fan_in)`, a recurrent matrix `(4 * hidden, hidden)` and a
/// `4 * hidden` bias, laid out as `[input, forget, cell, output]` gates.
#[derive(Debug)]
pub struct ControllerNetwork<T: Float + Debug + Send + Sync + 'static> {
    input_weights: Vec<Array2<T>>,
    recurrent_weights: Vec<Array2<T>>,
    biases: Vec<Array1<T>>,
    output_weights: Array2<T>,
    output_bias: Array1<T>,
    hidden_states: Vec<Array1<T>>,
    cell_states: Vec<Array1<T>>,
    numlayers: usize,
    hidden_size: usize,
    input_size: usize,
    output_size: usize,
}

/// Experience buffer for RL training
#[derive(Debug)]
pub struct ExperienceBuffer<T: Float + Debug + Send + Sync + 'static> {
    states: VecDeque<Array1<T>>,
    actions: VecDeque<usize>,
    rewards: VecDeque<T>,
    next_states: VecDeque<Array1<T>>,
    dones: VecDeque<bool>,
    capacity: usize,
}

/// Policy optimizer for the RL controller.
///
/// SGD with heavy-ball momentum and global-norm gradient clipping, applied to
/// every controller tensor.
#[derive(Debug)]
pub struct PolicyOptimizer<T: Float + Debug + Send + Sync + 'static> {
    learning_rate: T,
    momentum: T,
    velocity: Option<ControllerGradients<T>>,
    gradient_clip_norm: T,
}

/// Baseline predictor for variance reduction.
///
/// A linear value head `b(s) = w . s + b0` trained online by
/// [`BaselineOptimizer`] to regress the observed return, which is subtracted
/// from the reward to form the REINFORCE advantage.
#[derive(Debug)]
pub struct BaselinePredictor<T: Float + Debug + Send + Sync + 'static> {
    weights: Array1<T>,
    bias: T,
    optimizer: BaselineOptimizer<T>,
}

/// Baseline optimizer: SGD with momentum over the value head.
#[derive(Debug)]
pub struct BaselineOptimizer<T: Float + Debug + Send + Sync + 'static> {
    learning_rate: T,
    momentum: T,
    weight_velocity: Array1<T>,
    bias_velocity: T,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + 'static>
    ReinforcementLearningSearch<T>
{
    /// Create a controller-based search seeded from OS entropy.
    pub fn new(
        controller_hidden_size: usize,
        controller_num_layers: usize,
        learningrate: f64,
    ) -> Self {
        Self::build(
            controller_hidden_size,
            controller_num_layers,
            learningrate,
            None,
        )
    }

    /// Create a fully reproducible controller-based search.
    pub fn new_with_seed(
        controller_hidden_size: usize,
        controller_num_layers: usize,
        learningrate: f64,
        seed: u64,
    ) -> Self {
        Self::build(
            controller_hidden_size,
            controller_num_layers,
            learningrate,
            Some(seed),
        )
    }

    fn build(
        controller_hidden_size: usize,
        controller_num_layers: usize,
        learningrate: f64,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = Random::seed(seed.unwrap_or_else(scirs2_core::random::random::<u64>));

        let controller_network = ControllerNetwork::new_with_shape(
            CONTROLLER_INPUT_SIZE,
            controller_hidden_size.max(1),
            controller_num_layers.max(1),
            CONTROLLER_OUTPUT_SIZE,
            &mut rng,
        );

        Self {
            controller_network,
            experience_buffer: ExperienceBuffer::new(10000),
            policy_optimizer: PolicyOptimizer::new(scalar(learningrate)),
            baseline_predictor: BaselinePredictor::new(CONTROLLER_INPUT_SIZE),
            epsilon: 0.1,
            exploration_decay: 0.995,
            statistics: SearchStrategyStatistics::default(),
            entropy_bonus: 0.01,
            rng,
            pending_trajectories: VecDeque::new(),
            ready_trajectories: Vec::new(),
            updates_applied: 0,
        }
    }

    /// Number of REINFORCE updates applied so far.
    pub fn updates_applied(&self) -> usize {
        self.updates_applied
    }

    /// Immutable view of the controller network.
    pub fn controller(&self) -> &ControllerNetwork<T> {
        &self.controller_network
    }

    /// Current entropy-bonus coefficient.
    pub fn entropy_bonus(&self) -> f64 {
        self.entropy_bonus
    }

    /// Set the entropy-bonus coefficient.
    pub fn set_entropy_bonus(&mut self, entropy_bonus: f64) {
        self.entropy_bonus = entropy_bonus;
    }

    /// Encode a search space into the controller's fixed-width input vector.
    ///
    /// The descriptor is deterministic and carries real structure: the
    /// configured component-count bounds, the connection budget, a normalized
    /// count of declared component types, and a multi-hot block over the
    /// component types actually present. It used to be all zeros, which gave
    /// the controller no signal whatsoever.
    fn encode_search_space(&self, searchspace: &SearchSpaceConfig) -> Result<Array1<T>> {
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "reinforcement-learning search requires a non-empty \
                 SearchSpaceConfig::components"
                    .to_string(),
            ));
        }

        let mut state: Array1<T> = Array1::zeros(CONTROLLER_INPUT_SIZE);
        state[0] = T::one(); // bias term

        let n = searchspace.components.len() as f64;
        state[1] = scalar(n / (n + 1.0));
        state[2] = scalar(searchspace.min_components as f64 / 16.0);
        state[3] = scalar(searchspace.max_components as f64 / 16.0);
        state[4] = scalar(searchspace.max_connections as f64 / 32.0);

        // Mean number of tunable hyperparameters per component.
        let total_ranges: usize = searchspace
            .components
            .iter()
            .map(|c| c.hyperparameter_ranges.len())
            .sum();
        state[5] = scalar((total_ranges as f64 / n) / 8.0);

        // Multi-hot block over component families, offset past the scalars.
        const BLOCK_OFFSET: usize = 8;
        for component in &searchspace.components {
            let idx = BLOCK_OFFSET + component_family_index(&component.component_type);
            if idx < CONTROLLER_INPUT_SIZE {
                state[idx] = state[idx] + T::one();
            }
        }

        Ok(state)
    }

    /// Autoregressively sample an architecture from the controller.
    ///
    /// Step 0 selects a component type; each following step selects a bin for
    /// one of that component's hyperparameters. The sampled one-hot is fed back
    /// in as the next input, and every action's log-probability is recorded so
    /// `train_controller` can form the REINFORCE gradient.
    fn sample_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
    ) -> Result<(OptimizerArchitecture<T>, Trajectory<T>)> {
        let state = self.encode_search_space(searchspace)?;
        self.controller_network.reset_states();

        let mut steps = Vec::new();

        // ---- Step 0: component type -------------------------------------
        let num_components = searchspace
            .components
            .len()
            .min(self.controller_network.output_size);
        let (component_action, step) = self.decode_step(&state, num_components)?;
        steps.push(step);

        let component_config = match searchspace.components.get(component_action) {
            Some(config) => config.clone(),
            None => {
                return Err(OptimError::SearchSpaceError(
                    "controller sampled a component index outside the search space".to_string(),
                ))
            }
        };

        // ---- Following steps: one per hyperparameter --------------------
        let mut parameter_names: Vec<String> = component_config
            .hyperparameter_ranges
            .keys()
            .cloned()
            .collect();
        // Sorted so the decoding order (and therefore the RNG consumption) is
        // deterministic for a given seed.
        parameter_names.sort();
        parameter_names.truncate(MAX_HYPERPARAMETERS);

        let mut hyperparameters: HashMap<String, T> = HashMap::new();
        let mut next_input = one_hot::<T>(component_action, CONTROLLER_INPUT_SIZE);

        for name in &parameter_names {
            let range = match component_config.hyperparameter_ranges.get(name) {
                Some(range) => range,
                None => continue,
            };

            let arity = bin_count(range).min(self.controller_network.output_size);
            let (bin, step) = self.decode_step(&next_input, arity)?;
            steps.push(step);

            let value = decode_bin(range, bin, arity)?;
            hyperparameters.insert(name.clone(), scalar(value));

            next_input = one_hot::<T>(bin, CONTROLLER_INPUT_SIZE);
        }

        let component_type = map_component_type(&component_config.component_type);

        let architecture = OptimizerArchitecture {
            components: vec![format!("{:?}", component_type)],
            parameters: hyperparameters.clone(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters,
            architecture_id: format!("rl_arch_{:08x}", self.rng.random::<u32>()),
        };

        let trajectory = Trajectory {
            state,
            steps,
            component_action,
        };

        Ok((architecture, trajectory))
    }

    /// Run one controller step and sample from its softmax over `arity` logits.
    fn decode_step(&mut self, input: &Array1<T>, arity: usize) -> Result<(usize, StepRecord<T>)> {
        if arity == 0 {
            return Err(OptimError::SearchSpaceError(
                "cannot sample from a zero-arity decoding step".to_string(),
            ));
        }

        let (logits, layers, hidden_top) = self.controller_network.forward_cached(input)?;
        let probs = softmax(&logits, arity);
        let action = sample_categorical(&probs, &mut self.rng);

        let floor = scalar::<T>(1e-12);
        let log_prob = probs.get(action).copied().unwrap_or(floor).max(floor).ln();

        Ok((
            action,
            StepRecord {
                layers,
                hidden_top,
                probs,
                action,
                log_prob,
            },
        ))
    }

    /// Apply one REINFORCE update from every trajectory whose reward is known.
    ///
    /// The gradient ascended is
    /// `sum_t (R - b(s)) * grad log pi(a_t | s_t) + beta * grad H(pi_t)`,
    /// where `b` is the learned baseline and `beta` the entropy bonus.
    /// Backpropagation through the LSTM stack is truncated to the current
    /// timestep (TBPTT with horizon 1) — the standard approximation for
    /// controller networks of this size, and documented here because it is an
    /// approximation rather than the exact gradient.
    fn train_controller(&mut self) -> Result<()> {
        if self.ready_trajectories.is_empty() {
            return Ok(());
        }

        let batch: Vec<(Trajectory<T>, T)> = std::mem::take(&mut self.ready_trajectories);
        let mut gradients = ControllerGradients::zeros_like(&self.controller_network);
        let beta = scalar::<T>(self.entropy_bonus);
        let batch_size = scalar::<T>(batch.len() as f64);

        for (trajectory, reward) in &batch {
            let baseline = self.baseline_predictor.predict(&trajectory.state);
            let advantage = *reward - baseline;

            for step in &trajectory.steps {
                let active = step.probs.len();
                let h_entropy = entropy(&step.probs);

                // dJ/dlogit for the active prefix.
                let mut d_logits: Array1<T> = Array1::zeros(self.controller_network.output_size);
                for j in 0..active {
                    let p = step.probs[j];
                    let indicator = if j == step.action {
                        T::one()
                    } else {
                        T::zero()
                    };
                    // Score-function term.
                    let policy_term = advantage * (indicator - p);
                    // Entropy term: dH/dz_j = -p_j (ln p_j + H).
                    let log_p = if p > scalar::<T>(1e-12) {
                        p.ln()
                    } else {
                        scalar::<T>(-27.6) // ln(1e-12)
                    };
                    let entropy_term = beta * (-p * (log_p + h_entropy));
                    d_logits[j] = policy_term + entropy_term;
                }

                // Output projection: logits = W_out h + b_out.
                for j in 0..self.controller_network.output_size {
                    let dz = d_logits[j];
                    if dz == T::zero() {
                        continue;
                    }
                    gradients.output_bias[j] = gradients.output_bias[j] + dz;
                    for k in 0..self.controller_network.hidden_size {
                        gradients.output_weights[[j, k]] =
                            gradients.output_weights[[j, k]] + dz * step.hidden_top[k];
                    }
                }

                // dL/dh for the topmost hidden state.
                let mut d_hidden: Array1<T> = Array1::zeros(self.controller_network.hidden_size);
                for k in 0..self.controller_network.hidden_size {
                    let mut acc = T::zero();
                    for j in 0..self.controller_network.output_size {
                        acc = acc + self.controller_network.output_weights[[j, k]] * d_logits[j];
                    }
                    d_hidden[k] = acc;
                }

                // Backward through the LSTM stack, top layer first.
                for layer in (0..self.controller_network.numlayers).rev() {
                    let cache = match step.layers.get(layer) {
                        Some(cache) => cache,
                        None => break,
                    };
                    d_hidden = backward_lstm_cell(
                        &self.controller_network,
                        &mut gradients,
                        layer,
                        cache,
                        &d_hidden,
                    );
                }
            }

            // Train the baseline toward the observed return.
            self.baseline_predictor.update(&trajectory.state, *reward);
        }

        // Average over the batch so the step size does not depend on how many
        // results arrived together.
        if batch_size > T::zero() {
            gradients.scale(T::one() / batch_size);
        }

        self.policy_optimizer
            .apply(&mut self.controller_network, gradients);
        self.updates_applied += 1;

        Ok(())
    }
}

/// Backpropagate `d_hidden` through one LSTM cell, accumulating parameter
/// gradients and returning `dL/dx` for the layer below.
fn backward_lstm_cell<T: Float + Debug + Send + Sync + 'static>(
    network: &ControllerNetwork<T>,
    gradients: &mut ControllerGradients<T>,
    layer: usize,
    cache: &LayerCache<T>,
    d_hidden: &Array1<T>,
) -> Array1<T> {
    let hidden = network.hidden_size;
    let one = T::one();

    // h = o * tanh(c)  =>  do = dh * tanh(c);  dc = dh * o * (1 - tanh(c)^2)
    // c = f * c_prev + i * g
    let mut d_gates: Array1<T> = Array1::zeros(4 * hidden);
    for k in 0..hidden {
        let dh = d_hidden[k];
        let tanh_c = cache.tanh_c[k];
        let o = cache.gate_o[k];
        let i = cache.gate_i[k];
        let f = cache.gate_f[k];
        let g = cache.gate_g[k];

        let d_o = dh * tanh_c;
        let d_c = dh * o * (one - tanh_c * tanh_c);

        let d_i = d_c * g;
        let d_g = d_c * i;
        let d_f = d_c * cache.c_prev[k];

        // Pre-activation gradients (gate order: i, f, g, o).
        d_gates[k] = d_i * i * (one - i);
        d_gates[hidden + k] = d_f * f * (one - f);
        d_gates[2 * hidden + k] = d_g * (one - g * g);
        d_gates[3 * hidden + k] = d_o * o * (one - o);
    }

    let fan_in = cache.input.len();

    for row in 0..(4 * hidden) {
        let dz = d_gates[row];
        if dz == T::zero() {
            continue;
        }
        gradients.biases[layer][row] = gradients.biases[layer][row] + dz;
        for col in 0..fan_in {
            gradients.input_weights[layer][[row, col]] =
                gradients.input_weights[layer][[row, col]] + dz * cache.input[col];
        }
        for col in 0..hidden {
            gradients.recurrent_weights[layer][[row, col]] =
                gradients.recurrent_weights[layer][[row, col]] + dz * cache.h_prev[col];
        }
    }

    // dL/dx = W_x^T d_gates
    let mut d_input: Array1<T> = Array1::zeros(fan_in);
    for col in 0..fan_in {
        let mut acc = T::zero();
        for row in 0..(4 * hidden) {
            acc = acc + network.input_weights[layer][[row, col]] * d_gates[row];
        }
        d_input[col] = acc;
    }
    d_input
}

impl<
        T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + 'static + std::iter::Sum,
    > SearchStrategy<T> for ReinforcementLearningSearch<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        if searchspace.components.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "reinforcement-learning search requires a non-empty \
                 SearchSpaceConfig::components"
                    .to_string(),
            ));
        }
        self.controller_network.reset_states();
        self.pending_trajectories.clear();
        self.ready_trajectories.clear();
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        _history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        let (architecture, trajectory) = self.sample_architecture(searchspace)?;

        // Bound the pending queue: a caller that never reports results must not
        // grow it without limit.
        if self.pending_trajectories.len() >= 4096 {
            self.pending_trajectories.pop_front();
        }
        self.pending_trajectories.push_back(trajectory);

        self.statistics.total_architectures_generated += 1;
        Ok(architecture)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        for result in results {
            let reward = result
                .evaluation_results
                .metric_scores
                .get(&EvaluationMetric::FinalPerformance)
                .cloned()
                .unwrap_or(result.evaluation_results.overall_score);

            // Pair the reward with the trajectory that produced it (FIFO), and
            // record the REAL state/action instead of the zero placeholders the
            // buffer used to receive.
            if let Some(trajectory) = self.pending_trajectories.pop_front() {
                self.experience_buffer.add_experience(
                    trajectory.state.clone(),
                    trajectory.component_action,
                    reward,
                    trajectory.state.clone(),
                    true,
                );
                self.ready_trajectories.push((trajectory, reward));
            }
        }

        // Train as soon as at least one complete trajectory is available. The
        // old threshold of 1000 buffered experiences was never reached in a
        // realistic search budget, so the controller never trained at all.
        if !self.ready_trajectories.is_empty() {
            self.train_controller()?;
        }

        // Update statistics
        if !results.is_empty() {
            let performances: Vec<T> = results
                .iter()
                .filter_map(|r| {
                    r.evaluation_results
                        .metric_scores
                        .get(&EvaluationMetric::FinalPerformance)
                })
                .cloned()
                .collect();

            if !performances.is_empty() {
                self.statistics.best_performance = performances
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .cloned()
                    .unwrap_or(T::zero());

                let sum: T = performances.iter().cloned().sum();
                let count: T = scirs2_core::numeric::NumCast::from(performances.len())
                    .unwrap_or_else(|| T::one());
                self.statistics.average_performance = sum / count;
            }
        }

        // Decay exploration
        self.epsilon *= self.exploration_decay;

        Ok(())
    }

    fn name(&self) -> &str {
        "ReinforcementLearningSearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate = scalar(self.epsilon);
        stats.exploitation_rate = scalar(1.0 - self.epsilon);
        stats
    }
}

// ---------------------------------------------------------------------------
// Controller network
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Clone + 'static + Send + Sync> ControllerNetwork<T> {
    /// Construct a controller with the crate's default input/output widths.
    pub fn new(hidden_size: usize, numlayers: usize) -> Self {
        let mut rng = Random::seed(scirs2_core::random::random::<u64>());
        Self::new_with_shape(
            CONTROLLER_INPUT_SIZE,
            hidden_size.max(1),
            numlayers.max(1),
            CONTROLLER_OUTPUT_SIZE,
            &mut rng,
        )
    }

    /// Construct a controller with explicit dimensions.
    ///
    /// Weights are initialised uniformly in `+/- 1/sqrt(fan_in)`. Zero-filled
    /// weights (the previous behaviour) make every logit identical and the
    /// policy gradient identically zero, so the controller could never learn.
    pub fn new_with_shape(
        input_size: usize,
        hidden_size: usize,
        numlayers: usize,
        output_size: usize,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Self {
        let input_size = input_size.max(1);
        let hidden_size = hidden_size.max(1);
        let numlayers = numlayers.max(1);
        let output_size = output_size.max(1);

        let mut input_weights = Vec::with_capacity(numlayers);
        let mut recurrent_weights = Vec::with_capacity(numlayers);
        let mut biases = Vec::with_capacity(numlayers);
        let mut hidden_states = Vec::with_capacity(numlayers);
        let mut cell_states = Vec::with_capacity(numlayers);

        for layer in 0..numlayers {
            // Layer 0 consumes the encoded state; later layers consume the
            // hidden state of the layer below.
            let fan_in = if layer == 0 { input_size } else { hidden_size };

            input_weights.push(random_matrix(4 * hidden_size, fan_in, rng));
            recurrent_weights.push(random_matrix(4 * hidden_size, hidden_size, rng));

            // Forget-gate bias initialised to 1, the standard trick that keeps
            // gradients flowing early in training.
            let mut bias: Array1<T> = Array1::zeros(4 * hidden_size);
            for k in 0..hidden_size {
                bias[hidden_size + k] = T::one();
            }
            biases.push(bias);

            hidden_states.push(Array1::zeros(hidden_size));
            cell_states.push(Array1::zeros(hidden_size));
        }

        Self {
            input_weights,
            recurrent_weights,
            biases,
            output_weights: random_matrix(output_size, hidden_size, rng),
            output_bias: Array1::zeros(output_size),
            hidden_states,
            cell_states,
            numlayers,
            hidden_size,
            input_size,
            output_size,
        }
    }

    /// Clear the recurrent state, starting a fresh episode.
    pub fn reset_states(&mut self) {
        for i in 0..self.numlayers {
            self.hidden_states[i].fill(T::zero());
            self.cell_states[i].fill(T::zero());
        }
    }

    /// Declared input width.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Declared logit width.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Hidden width of each LSTM layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of stacked LSTM layers.
    pub fn num_layers(&self) -> usize {
        self.numlayers
    }

    /// Run one timestep through the stack and return the output logits.
    pub fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        let (logits, _, _) = self.forward_cached(input)?;
        Ok(logits)
    }

    /// Run one timestep, also returning the per-layer caches needed for
    /// backpropagation.
    fn forward_cached(
        &mut self,
        input: &Array1<T>,
    ) -> Result<(Array1<T>, Vec<LayerCache<T>>, Array1<T>)> {
        // Accept any input length by projecting onto the declared width: too
        // short is zero-padded, too long is truncated. This keeps `forward`
        // total instead of panicking on a dimension mismatch.
        let mut current: Array1<T> = Array1::zeros(self.input_size);
        for i in 0..self.input_size.min(input.len()) {
            current[i] = input[i];
        }

        let hidden = self.hidden_size;
        let mut caches = Vec::with_capacity(self.numlayers);

        for layer in 0..self.numlayers {
            let h_prev = self.hidden_states[layer].clone();
            let c_prev = self.cell_states[layer].clone();

            // gates = W_x x + W_h h_prev + b
            let mut gates = self.input_weights[layer].dot(&current);
            let recurrent = self.recurrent_weights[layer].dot(&h_prev);
            for k in 0..gates.len() {
                gates[k] = gates[k] + recurrent[k] + self.biases[layer][k];
            }

            let mut gate_i: Array1<T> = Array1::zeros(hidden);
            let mut gate_f: Array1<T> = Array1::zeros(hidden);
            let mut gate_g: Array1<T> = Array1::zeros(hidden);
            let mut gate_o: Array1<T> = Array1::zeros(hidden);
            let mut c_new: Array1<T> = Array1::zeros(hidden);
            let mut tanh_c: Array1<T> = Array1::zeros(hidden);
            let mut h_new: Array1<T> = Array1::zeros(hidden);

            for k in 0..hidden {
                let i = sigmoid(gates[k]);
                let f = sigmoid(gates[hidden + k]);
                let g = gates[2 * hidden + k].tanh();
                let o = sigmoid(gates[3 * hidden + k]);

                let c = f * c_prev[k] + i * g;
                let tc = c.tanh();

                gate_i[k] = i;
                gate_f[k] = f;
                gate_g[k] = g;
                gate_o[k] = o;
                c_new[k] = c;
                tanh_c[k] = tc;
                h_new[k] = o * tc;
            }

            caches.push(LayerCache {
                input: current.clone(),
                h_prev,
                c_prev,
                gate_i,
                gate_f,
                gate_g,
                gate_o,
                tanh_c,
            });

            self.cell_states[layer] = c_new;
            self.hidden_states[layer] = h_new.clone();
            current = h_new;
        }

        let hidden_top = current.clone();
        let mut logits = self.output_weights.dot(&hidden_top);
        for j in 0..logits.len() {
            logits[j] = logits[j] + self.output_bias[j];
        }

        Ok((logits, caches, hidden_top))
    }
}

/// Uniform `+/- 1/sqrt(fan_in)` weight matrix.
fn random_matrix<T: Float + Debug + Send + Sync + 'static>(
    rows: usize,
    cols: usize,
    rng: &mut Random<scirs2_core::random::rngs::StdRng>,
) -> Array2<T> {
    let bound = 1.0 / (cols.max(1) as f64).sqrt();
    let mut m: Array2<T> = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let u = rng.random::<f64>() * 2.0 - 1.0;
            m[[r, c]] = scalar(u * bound);
        }
    }
    m
}

/// One-hot vector of the given width, with out-of-range indices wrapped.
fn one_hot<T: Float + Debug + Send + Sync + 'static>(index: usize, width: usize) -> Array1<T> {
    let mut v: Array1<T> = Array1::zeros(width.max(1));
    if width > 0 {
        v[index % width] = T::one();
    }
    v
}

// ---------------------------------------------------------------------------
// Search-space decoding helpers
// ---------------------------------------------------------------------------

/// Number of discrete choices a parameter range is decoded into.
fn bin_count(range: &ParameterRange) -> usize {
    match range {
        ParameterRange::Boolean => 2,
        ParameterRange::Discrete(values) => values.len().max(1),
        ParameterRange::Categorical(values) => values.len().max(1),
        ParameterRange::Integer(min, max) => {
            let span = (max - min).max(0) as usize;
            span.max(1).min(HYPERPARAMETER_BINS)
        }
        ParameterRange::Continuous(_, _) | ParameterRange::LogUniform(_, _) => HYPERPARAMETER_BINS,
    }
}

/// Map a sampled bin index back onto a concrete parameter value.
///
/// Continuous and log-uniform ranges use the bin centre so a bin index maps to
/// a representative value rather than a boundary.
fn decode_bin(range: &ParameterRange, bin: usize, arity: usize) -> Result<f64> {
    let arity = arity.max(1);
    let bin = bin.min(arity - 1);
    let position = (bin as f64 + 0.5) / arity as f64;

    let value = match range {
        ParameterRange::Continuous(min, max) => min + (max - min) * position,
        ParameterRange::LogUniform(min, max) => {
            if *min <= 0.0 || *max <= 0.0 {
                return Err(OptimError::SearchSpaceError(format!(
                    "log-uniform range must be strictly positive, got ({}, {})",
                    min, max
                )));
            }
            (min.ln() + (max.ln() - min.ln()) * position).exp()
        }
        ParameterRange::Integer(min, max) => {
            let span = (max - min).max(0) as f64;
            (*min as f64 + (span * position).floor()).min(*max as f64)
        }
        ParameterRange::Boolean => {
            if bin == 0 {
                0.0
            } else {
                1.0
            }
        }
        ParameterRange::Discrete(values) => {
            if values.is_empty() {
                return Err(OptimError::SearchSpaceError(
                    "discrete parameter range is empty".to_string(),
                ));
            }
            values[bin.min(values.len() - 1)]
        }
        ParameterRange::Categorical(values) => {
            if values.is_empty() {
                return Err(OptimError::SearchSpaceError(
                    "categorical parameter range is empty".to_string(),
                ));
            }
            bin.min(values.len() - 1) as f64
        }
    };

    Ok(value)
}

/// Stable index used by the search-space encoder's multi-hot block.
fn component_family_index(component_type: &ConfigComponentType) -> usize {
    match component_type {
        ConfigComponentType::SGD => 0,
        ConfigComponentType::Adam => 1,
        ConfigComponentType::AdamW => 2,
        ConfigComponentType::RMSprop => 3,
        ConfigComponentType::AdaGrad => 4,
        ConfigComponentType::AdaDelta => 5,
        ConfigComponentType::LBFGS => 6,
        ConfigComponentType::Momentum => 7,
        ConfigComponentType::Nesterov => 8,
        ConfigComponentType::Custom(_) => 9,
        _ => 10,
    }
}

/// Map a configured component type onto the architecture-level vocabulary.
fn map_component_type(component_type: &ConfigComponentType) -> crate::architecture::ComponentType {
    use crate::architecture::ComponentType;
    match component_type {
        ConfigComponentType::SGD => ComponentType::SGD,
        ConfigComponentType::Adam => ComponentType::Adam,
        ConfigComponentType::AdamW => ComponentType::AdamW,
        ConfigComponentType::RMSprop => ComponentType::RMSprop,
        ConfigComponentType::AdaGrad => ComponentType::AdaGrad,
        ConfigComponentType::AdaDelta => ComponentType::AdaDelta,
        ConfigComponentType::LBFGS => ComponentType::LBFGS,
        ConfigComponentType::Momentum => ComponentType::Momentum,
        ConfigComponentType::Nesterov => ComponentType::Nesterov,
        ConfigComponentType::Custom(_) => ComponentType::Custom,
        _ => ComponentType::Adam,
    }
}

// ---------------------------------------------------------------------------
// Experience buffer / optimizers
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Send + Sync> ExperienceBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            states: VecDeque::new(),
            actions: VecDeque::new(),
            rewards: VecDeque::new(),
            next_states: VecDeque::new(),
            dones: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn add_experience(
        &mut self,
        state: Array1<T>,
        action: usize,
        reward: T,
        next_state: Array1<T>,
        done: bool,
    ) {
        self.states.push_back(state);
        self.actions.push_back(action);
        self.rewards.push_back(reward);
        self.next_states.push_back(next_state);
        self.dones.push_back(done);

        // Remove oldest if at capacity
        while self.states.len() > self.capacity {
            self.states.pop_front();
            self.actions.pop_front();
            self.rewards.pop_front();
            self.next_states.pop_front();
            self.dones.pop_front();
        }
    }

    /// Number of stored transitions.
    pub fn size(&self) -> usize {
        self.states.len()
    }

    /// Actions recorded so far, oldest first.
    pub fn actions(&self) -> impl Iterator<Item = &usize> {
        self.actions.iter()
    }

    /// Rewards recorded so far, oldest first.
    pub fn rewards(&self) -> impl Iterator<Item = &T> {
        self.rewards.iter()
    }
}

impl<T: Float + Debug + Default + Send + Sync + 'static> PolicyOptimizer<T> {
    fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            momentum: scalar(0.9),
            velocity: None,
            gradient_clip_norm: scalar(5.0),
        }
    }

    /// Apply an ascent step to the controller.
    ///
    /// The gradient is first clipped to `gradient_clip_norm` by global L2 norm,
    /// then smoothed with heavy-ball momentum, then added to the parameters
    /// (ascent, because the objective is the expected return).
    fn apply(&mut self, network: &mut ControllerNetwork<T>, mut gradients: ControllerGradients<T>) {
        // Global-norm clipping.
        let norm = gradients.global_norm();
        if norm.is_finite() && norm > self.gradient_clip_norm && norm > T::zero() {
            gradients.scale(self.gradient_clip_norm / norm);
        } else if !norm.is_finite() {
            // A non-finite gradient carries no information; skip the step
            // rather than corrupting the weights.
            return;
        }

        let velocity = self
            .velocity
            .get_or_insert_with(|| ControllerGradients::zeros_like(network));

        let lr = self.learning_rate;
        let mu = self.momentum;

        for layer in 0..network.numlayers {
            update_matrix(
                &mut network.input_weights[layer],
                &mut velocity.input_weights[layer],
                &gradients.input_weights[layer],
                mu,
                lr,
            );
            update_matrix(
                &mut network.recurrent_weights[layer],
                &mut velocity.recurrent_weights[layer],
                &gradients.recurrent_weights[layer],
                mu,
                lr,
            );
            update_vector(
                &mut network.biases[layer],
                &mut velocity.biases[layer],
                &gradients.biases[layer],
                mu,
                lr,
            );
        }

        update_matrix(
            &mut network.output_weights,
            &mut velocity.output_weights,
            &gradients.output_weights,
            mu,
            lr,
        );
        update_vector(
            &mut network.output_bias,
            &mut velocity.output_bias,
            &gradients.output_bias,
            mu,
            lr,
        );
    }
}

/// `v = mu v + g; w += lr * v`
fn update_matrix<T: Float + Debug + Send + Sync + 'static>(
    weights: &mut Array2<T>,
    velocity: &mut Array2<T>,
    gradient: &Array2<T>,
    momentum: T,
    learning_rate: T,
) {
    let (rows, cols) = weights.dim();
    for r in 0..rows {
        for c in 0..cols {
            let v = momentum * velocity[[r, c]] + gradient[[r, c]];
            velocity[[r, c]] = v;
            weights[[r, c]] = weights[[r, c]] + learning_rate * v;
        }
    }
}

/// `v = mu v + g; w += lr * v`
fn update_vector<T: Float + Debug + Send + Sync + 'static>(
    weights: &mut Array1<T>,
    velocity: &mut Array1<T>,
    gradient: &Array1<T>,
    momentum: T,
    learning_rate: T,
) {
    for i in 0..weights.len() {
        let v = momentum * velocity[i] + gradient[i];
        velocity[i] = v;
        weights[i] = weights[i] + learning_rate * v;
    }
}

impl<T: Float + Debug + Default + Send + Sync + 'static> BaselinePredictor<T> {
    fn new(state_size: usize) -> Self {
        Self {
            weights: Array1::zeros(state_size.max(1)),
            bias: T::zero(),
            optimizer: BaselineOptimizer::new(scalar(0.01), state_size.max(1)),
        }
    }

    /// Predicted return for a state.
    fn predict(&self, state: &Array1<T>) -> T {
        let n = self.weights.len().min(state.len());
        let mut acc = self.bias;
        for i in 0..n {
            acc = acc + self.weights[i] * state[i];
        }
        acc
    }

    /// One SGD-with-momentum step on the squared prediction error.
    fn update(&mut self, state: &Array1<T>, target: T) {
        let prediction = self.predict(state);
        let error = prediction - target;
        let two = scalar::<T>(2.0);

        let lr = self.optimizer.learning_rate;
        let mu = self.optimizer.momentum;
        let n = self.weights.len().min(state.len());

        for i in 0..n {
            let gradient = two * error * state[i];
            let v = mu * self.optimizer.weight_velocity[i] + gradient;
            self.optimizer.weight_velocity[i] = v;
            self.weights[i] = self.weights[i] - lr * v;
        }

        let bias_gradient = two * error;
        let bv = mu * self.optimizer.bias_velocity + bias_gradient;
        self.optimizer.bias_velocity = bv;
        self.bias = self.bias - lr * bv;
    }

    /// Current value estimate for a state (public view for callers/tests).
    pub fn value(&self, state: &Array1<T>) -> T {
        self.predict(state)
    }
}

impl<T: Float + Debug + Default + Send + Sync + 'static> BaselineOptimizer<T> {
    fn new(learning_rate: T, state_size: usize) -> Self {
        Self {
            learning_rate,
            momentum: scalar(0.9),
            weight_velocity: Array1::zeros(state_size.max(1)),
            bias_velocity: T::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::config::OptimizerComponentConfig;
    use crate::nas_engine::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };

    fn component(kind: ConfigComponentType, with_lr: bool) -> OptimizerComponentConfig {
        let mut hyperparameter_ranges = HashMap::new();
        if with_lr {
            hyperparameter_ranges.insert(
                "learning_rate".to_string(),
                ParameterRange::LogUniform(1e-4, 1e-1),
            );
        }
        OptimizerComponentConfig {
            component_type: kind,
            hyperparameter_ranges,
            complexity_score: 1.0,
            memory_requirement: 1024,
            computational_cost: 1.0,
            compatibility_constraints: Vec::new(),
        }
    }

    fn search_space(kinds: &[ConfigComponentType]) -> SearchSpaceConfig {
        let mut space = SearchSpaceConfig::default();
        space.components = kinds.iter().map(|k| component(k.clone(), true)).collect();
        space.min_components = 1;
        space.max_components = 1;
        space
    }

    fn result_with_reward(reward: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, reward);
        SearchResult {
            architecture: OptimizerArchitecture {
                components: Vec::new(),
                parameters: HashMap::new(),
                connections: Vec::new(),
                metadata: HashMap::new(),
                hyperparameters: HashMap::new(),
                architecture_id: "r".to_string(),
            },
            evaluation_results: EvaluationResults {
                metric_scores,
                overall_score: reward,
                confidence_intervals: HashMap::new(),
                evaluation_time: std::time::Duration::from_secs(0),
                success: true,
                error_message: None,
                cv_results: None,
                benchmark_results: HashMap::new(),
                training_trajectory: Vec::new(),
            },
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

    // -----------------------------------------------------------------
    // F2 — the controller must not panic and must produce real shapes
    // -----------------------------------------------------------------

    #[test]
    fn forward_pass_shapes_never_panic() {
        // Regression test: the old controller ALWAYS panicked because
        // `lstm_weights[0]` was (4h, h) while the input was 64-wide.
        for (hidden, layers) in [(64usize, 1usize), (64, 2), (256, 2), (32, 1)] {
            let mut rng = Random::seed(1);
            let mut network = ControllerNetwork::<f64>::new_with_shape(
                CONTROLLER_INPUT_SIZE,
                hidden,
                layers,
                CONTROLLER_OUTPUT_SIZE,
                &mut rng,
            );
            let input = Array1::<f64>::zeros(CONTROLLER_INPUT_SIZE);

            let logits = network
                .forward(&input)
                .unwrap_or_else(|e| panic!("forward failed for ({hidden},{layers}): {e}"));

            assert_eq!(logits.len(), CONTROLLER_OUTPUT_SIZE);
            assert_eq!(network.hidden_size(), hidden);
            assert_eq!(network.num_layers(), layers);
            assert!(logits.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn lstm_cell_updates_hidden_and_cell_state() {
        let mut rng = Random::seed(2);
        let mut network = ControllerNetwork::<f64>::new_with_shape(64, 16, 2, 32, &mut rng);

        let mut input = Array1::<f64>::zeros(64);
        input[0] = 1.0;

        // A fresh network starts from zeroed states.
        assert!(network
            .cell_states
            .iter()
            .all(|c| c.iter().all(|&v| v == 0.0)));

        network.forward(&input).expect("forward");
        let changed_cells = network
            .cell_states
            .iter()
            .any(|c| c.iter().any(|&v| v != 0.0));
        let changed_hidden = network
            .hidden_states
            .iter()
            .any(|h| h.iter().any(|&v| v != 0.0));
        assert!(
            changed_cells,
            "cell_states must be updated by the LSTM cell"
        );
        assert!(changed_hidden, "hidden_states must be updated");

        // The recurrent state must carry over: the same input twice gives a
        // different output because h_{t-1}/c_{t-1} differ.
        let first = network.forward(&input).expect("forward");
        let second = network.forward(&input).expect("forward");
        assert!(
            first
                .iter()
                .zip(second.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12),
            "the cell must be stateful across timesteps"
        );

        network.reset_states();
        assert!(network
            .cell_states
            .iter()
            .all(|c| c.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn forward_tolerates_mismatched_input_lengths() {
        let mut rng = Random::seed(3);
        let mut network = ControllerNetwork::<f64>::new_with_shape(64, 8, 1, 16, &mut rng);
        assert!(network.forward(&Array1::<f64>::zeros(4)).is_ok());
        assert!(network.forward(&Array1::<f64>::zeros(4096)).is_ok());
    }

    // -----------------------------------------------------------------
    // F7 — decoding really depends on the sampled actions
    // -----------------------------------------------------------------

    #[test]
    fn decoding_produces_varied_components_and_hyperparameters() {
        let space = search_space(&[
            ConfigComponentType::Adam,
            ConfigComponentType::SGD,
            ConfigComponentType::RMSprop,
        ]);
        let mut search = ReinforcementLearningSearch::<f64>::new_with_seed(32, 1, 0.05, 7);
        search.initialize(&space).expect("initialize");

        let mut components = std::collections::HashSet::new();
        let mut learning_rates = std::collections::HashSet::new();
        for _ in 0..60 {
            let arch = search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            assert_eq!(arch.components.len(), 1);
            components.insert(arch.components[0].clone());
            if let Some(lr) = arch.hyperparameters.get("learning_rate") {
                assert!(*lr >= 1e-4 && *lr <= 1e-1);
                learning_rates.insert(format!("{:.9}", lr));
            }
        }

        assert!(
            components.len() > 1,
            "the controller must sample more than one component type"
        );
        assert!(
            learning_rates.len() > 1,
            "hyperparameters must come from sampled bins, not a hardcoded 0.01"
        );
    }

    #[test]
    fn empty_search_space_is_an_error() {
        let mut space = search_space(&[ConfigComponentType::Adam]);
        space.components.clear();
        let mut search = ReinforcementLearningSearch::<f64>::new_with_seed(16, 1, 0.05, 1);
        assert!(matches!(
            search.initialize(&space),
            Err(OptimError::SearchSpaceError(_))
        ));
        assert!(matches!(
            search.generate_architecture(&space, &VecDeque::new()),
            Err(OptimError::SearchSpaceError(_))
        ));
    }

    #[test]
    fn experience_buffer_records_real_states_and_actions() {
        let space = search_space(&[ConfigComponentType::Adam, ConfigComponentType::SGD]);
        let mut search = ReinforcementLearningSearch::<f64>::new_with_seed(16, 1, 0.05, 4);
        search.initialize(&space).expect("initialize");

        for _ in 0..5 {
            search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
        }
        let results: Vec<SearchResult<f64>> =
            (0..5).map(|i| result_with_reward(0.1 * i as f64)).collect();
        search.update_with_results(&results).expect("update");

        assert_eq!(search.experience_buffer.size(), 5);
        // The stored states are the real encoded search space, not zeros.
        assert!(search
            .experience_buffer
            .states
            .iter()
            .all(|s| s.iter().any(|&v| v != 0.0)));
        let rewards: Vec<f64> = search.experience_buffer.rewards().copied().collect();
        assert_eq!(rewards, vec![0.0, 0.1, 0.2, 0.30000000000000004, 0.4]);
    }

    #[test]
    fn bins_cover_the_configured_range() {
        let range = ParameterRange::Continuous(0.0, 1.0);
        let first = decode_bin(&range, 0, HYPERPARAMETER_BINS).expect("bin");
        let last = decode_bin(&range, HYPERPARAMETER_BINS - 1, HYPERPARAMETER_BINS).expect("bin");
        assert!((first - 0.05).abs() < 1e-12);
        assert!((last - 0.95).abs() < 1e-12);

        let boolean = ParameterRange::Boolean;
        assert_eq!(decode_bin(&boolean, 0, 2).expect("bin"), 0.0);
        assert_eq!(decode_bin(&boolean, 1, 2).expect("bin"), 1.0);

        let discrete = ParameterRange::Discrete(vec![2.0, 4.0, 8.0]);
        assert_eq!(decode_bin(&discrete, 2, 3).expect("bin"), 8.0);
        assert!(matches!(
            decode_bin(&ParameterRange::Discrete(Vec::new()), 0, 1),
            Err(OptimError::SearchSpaceError(_))
        ));
    }

    // -----------------------------------------------------------------
    // F7 — REINFORCE actually shifts the policy
    // -----------------------------------------------------------------

    #[test]
    fn reinforce_increases_the_probability_of_the_rewarded_component() {
        // Toy problem: three component types, only `RMSprop` (index 2) is
        // rewarded. The controller must learn to select it more often.
        let space = search_space(&[
            ConfigComponentType::Adam,
            ConfigComponentType::SGD,
            ConfigComponentType::RMSprop,
        ]);
        let target = "RMSprop";

        let mut search = ReinforcementLearningSearch::<f64>::new_with_seed(32, 1, 0.5, 2024);
        search.set_entropy_bonus(0.0005);
        search.initialize(&space).expect("initialize");

        let iterations = 400usize;
        let window = 100usize;
        let mut selections: Vec<bool> = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let arch = search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            let chose_target = arch
                .components
                .first()
                .map(|c| c == target)
                .unwrap_or(false);
            selections.push(chose_target);

            let reward = if chose_target { 1.0 } else { 0.0 };
            search
                .update_with_results(&[result_with_reward(reward)])
                .expect("update");
        }

        let early = selections[..window].iter().filter(|&&c| c).count() as f64 / window as f64;
        let late = selections[iterations - window..]
            .iter()
            .filter(|&&c| c)
            .count() as f64
            / window as f64;

        assert!(
            search.updates_applied() >= iterations,
            "the controller must actually train (got {} updates)",
            search.updates_applied()
        );
        assert!(
            late > early + 0.15,
            "REINFORCE must raise the rewarded component's selection rate \
             (early {:.3} -> late {:.3})",
            early,
            late
        );
    }

    #[test]
    fn controller_weights_change_during_training() {
        let space = search_space(&[ConfigComponentType::Adam, ConfigComponentType::SGD]);
        let mut search = ReinforcementLearningSearch::<f64>::new_with_seed(16, 1, 0.2, 11);
        search.initialize(&space).expect("initialize");

        let before = search.controller().output_weights.clone();

        for _ in 0..25 {
            search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generate");
            search
                .update_with_results(&[result_with_reward(1.0)])
                .expect("update");
        }

        let after = &search.controller().output_weights;
        let moved = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(moved, "policy weights must be updated by REINFORCE");
        assert!(after.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn baseline_predictor_learns_the_mean_return() {
        let mut baseline = BaselinePredictor::<f64>::new(4);
        let mut state = Array1::<f64>::zeros(4);
        state[0] = 1.0; // bias-like feature

        for _ in 0..500 {
            baseline.update(&state, 0.75);
        }
        assert!(
            (baseline.value(&state) - 0.75).abs() < 0.05,
            "baseline should regress toward the observed return, got {}",
            baseline.value(&state)
        );
    }

    #[test]
    fn softmax_and_entropy_are_well_behaved() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.0]);
        let probs = softmax(&logits, 3);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);

        // Uniform distribution has maximum entropy ln(n).
        let uniform = vec![0.25_f64; 4];
        assert!((entropy(&uniform) - 4.0_f64.ln()).abs() < 1e-12);
        // A degenerate distribution has zero entropy.
        assert!(entropy(&[1.0_f64, 0.0, 0.0]).abs() < 1e-12);

        // Non-finite logits fall back to uniform instead of producing NaN.
        let broken = Array1::from_vec(vec![f64::NAN, f64::NAN]);
        let fallback = softmax(&broken, 2);
        assert!(fallback.iter().all(|p| (p - 0.5).abs() < 1e-12));
    }

    #[test]
    fn gradient_clipping_bounds_the_step() {
        let mut rng = Random::seed(99);
        let mut network = ControllerNetwork::<f64>::new_with_shape(64, 8, 1, 16, &mut rng);
        let mut optimizer = PolicyOptimizer::<f64>::new(1.0);

        let mut gradients = ControllerGradients::zeros_like(&network);
        gradients.output_bias.fill(1000.0);
        let norm_before = gradients.global_norm();
        assert!(norm_before > 5.0);

        let before = network.output_bias.clone();
        optimizer.apply(&mut network, gradients);
        let step: f64 = before
            .iter()
            .zip(network.output_bias.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            step <= 5.0 + 1e-9,
            "clipped step norm {} must not exceed the clip threshold",
            step
        );

        // Non-finite gradients are skipped rather than corrupting the weights.
        let mut bad = ControllerGradients::zeros_like(&network);
        bad.output_bias[0] = f64::NAN;
        let snapshot = network.output_bias.clone();
        optimizer.apply(&mut network, bad);
        assert_eq!(network.output_bias, snapshot);
    }
}
