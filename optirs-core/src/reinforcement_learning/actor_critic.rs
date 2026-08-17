// Actor-Critic Optimizers
//
// This module implements various actor-critic algorithms including A2C, A3C,
// SAC (Soft Actor-Critic), and other modern actor-critic methods.

use super::{
    add_named_gradients, clip_named_gradients, scale_named_gradients, ActionDistribution,
    DistributionType, PolicyNetwork, QNetwork, RLOptimizationMetrics, RLOptimizerConfig,
    RLScheduler, TrajectoryBatch, ValueNetwork,
};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

/// Actor-Critic optimization methods
#[derive(Debug, Clone, Copy)]
pub enum ActorCriticMethod {
    /// Advantage Actor-Critic (A2C)
    A2C,

    /// Asynchronous Advantage Actor-Critic (A3C)
    A3C,

    /// Soft Actor-Critic (SAC)
    SAC,

    /// Twin Delayed Deep Deterministic Policy Gradients (TD3)
    TD3,

    /// Deep Deterministic Policy Gradients (DDPG)
    DDPG,

    /// Distributed Distributional Deterministic Policy Gradients (D4PG)
    D4PG,

    /// Maximum a Posteriori Policy Optimisation (MPO)
    MPO,
}

/// Actor-Critic configuration
#[derive(Debug, Clone)]
pub struct ActorCriticConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Base RL configuration
    pub base_config: RLOptimizerConfig<T>,

    /// Actor-Critic method
    pub method: ActorCriticMethod,

    /// SAC-specific configuration
    pub sac_config: SACConfig<T>,

    /// TD3-specific configuration
    pub td3_config: TD3Config<T>,

    /// DDPG-specific configuration
    pub ddpg_config: DDPGConfig<T>,

    /// Use target networks
    pub use_target_networks: bool,

    /// Target network soft update rate (tau)
    pub target_update_rate: T,

    /// Target network hard update frequency
    pub target_hard_update_freq: Option<usize>,

    /// Experience replay buffer size
    pub replay_buffer_size: usize,

    /// Enable prioritized experience replay
    pub prioritized_replay: bool,

    /// Prioritized replay alpha parameter
    pub per_alpha: T,

    /// Prioritized replay beta parameter
    pub per_beta: T,

    /// Number of critic networks (for twin critic methods)
    pub n_critics: usize,
}

/// SAC (Soft Actor-Critic) configuration
#[derive(Debug, Clone)]
pub struct SACConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Temperature parameter for entropy regularization
    pub temperature: T,

    /// Automatic entropy tuning
    pub auto_entropy_tuning: bool,

    /// Target entropy (for automatic tuning)
    pub target_entropy: Option<T>,

    /// Temperature learning rate
    pub temperature_lr: T,

    /// Use repameterization trick
    pub use_reparameterization: bool,

    /// Policy update frequency
    pub policy_update_freq: usize,

    /// Target network update frequency
    pub target_update_freq: usize,
}

/// TD3 (Twin Delayed DDPG) configuration
#[derive(Debug, Clone)]
pub struct TD3Config<T: Float + Debug + Send + Sync + 'static> {
    /// Policy noise for target smoothing
    pub policy_noise: T,

    /// Noise clipping range
    pub noise_clip: T,

    /// Policy update delay
    pub policy_delay: usize,

    /// Exploration noise standard deviation
    pub exploration_noise: T,

    /// Action bounds for clipping
    pub action_bounds: Option<(T, T)>,
}

/// DDPG configuration
#[derive(Debug, Clone)]
pub struct DDPGConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Exploration noise standard deviation
    pub exploration_noise: T,

    /// Ornstein-Uhlenbeck mean-reversion rate θ
    pub ou_noise_theta: T,

    /// Ornstein-Uhlenbeck diffusion scale σ
    pub ou_noise_sigma: T,

    /// Ornstein-Uhlenbeck integration timestep dt.
    ///
    /// The process is `dx = θ(μ − x)dt + σ√dt·dW`; without `dt` the discretization
    /// is only valid at `dt = 1`, which is far too coarse for the θ/σ values the
    /// DDPG paper recommends.
    pub ou_noise_dt: T,

    /// Ornstein-Uhlenbeck long-run mean μ
    pub ou_noise_mu: T,

    /// Action bounds for clipping
    pub action_bounds: Option<(T, T)>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ActorCriticConfig<T> {
    fn default() -> Self {
        Self {
            base_config: RLOptimizerConfig::default(),
            method: ActorCriticMethod::A2C,
            sac_config: SACConfig::default(),
            td3_config: TD3Config::default(),
            ddpg_config: DDPGConfig::default(),
            use_target_networks: false,
            target_update_rate: T::from(0.005).unwrap_or_else(|| T::zero()),
            target_hard_update_freq: None,
            replay_buffer_size: 100000,
            prioritized_replay: false,
            per_alpha: T::from(0.6).unwrap_or_else(|| T::zero()),
            per_beta: T::from(0.4).unwrap_or_else(|| T::zero()),
            n_critics: 1,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for SACConfig<T> {
    fn default() -> Self {
        Self {
            temperature: T::from(0.2).unwrap_or_else(|| T::zero()),
            auto_entropy_tuning: true,
            target_entropy: None,
            temperature_lr: T::from(3e-4).unwrap_or_else(|| T::zero()),
            use_reparameterization: true,
            policy_update_freq: 1,
            target_update_freq: 1,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for TD3Config<T> {
    fn default() -> Self {
        Self {
            policy_noise: T::from(0.2).unwrap_or_else(|| T::zero()),
            noise_clip: T::from(0.5).unwrap_or_else(|| T::zero()),
            policy_delay: 2,
            exploration_noise: T::from(0.1).unwrap_or_else(|| T::zero()),
            action_bounds: Some((
                T::from(-1.0).unwrap_or_else(|| T::zero()),
                T::from(1.0).unwrap_or_else(|| T::zero()),
            )),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for DDPGConfig<T> {
    fn default() -> Self {
        Self {
            exploration_noise: T::from(0.1).unwrap_or_else(|| T::zero()),
            ou_noise_theta: T::from(0.15).unwrap_or_else(|| T::zero()),
            ou_noise_sigma: T::from(0.2).unwrap_or_else(|| T::zero()),
            ou_noise_dt: T::from(0.01).unwrap_or_else(|| T::zero()),
            ou_noise_mu: T::zero(),
            action_bounds: Some((
                T::from(-1.0).unwrap_or_else(|| T::zero()),
                T::from(1.0).unwrap_or_else(|| T::zero()),
            )),
        }
    }
}

/// Actor-Critic optimizer
pub struct ActorCriticOptimizer<
    T: Float + Debug + Send + Sync + 'static,
    P: PolicyNetwork<T>,
    V: ValueNetwork<T>,
> {
    /// Configuration
    config: ActorCriticConfig<T>,

    /// Actor (policy) network
    actor: P,

    /// Critic (value) networks
    critics: Vec<V>,

    /// Target networks (if enabled)
    target_actor: Option<P>,
    target_critics: Option<Vec<V>>,

    /// Temperature parameter for SAC
    temperature: T,

    /// Learning rate schedulers
    actor_scheduler: Option<RLScheduler<T>>,
    critic_scheduler: Option<RLScheduler<T>>,
    temperature_scheduler: Option<RLScheduler<T>>,

    /// Optimization metrics
    metrics: ActorCriticMetrics<T>,

    /// Update counters
    update_count: usize,
    policy_update_count: usize,

    /// Experience replay buffer
    replay_buffer: ExperienceReplayBuffer<T>,

    /// Ornstein-Uhlenbeck noise state (for DDPG)
    ou_noise_state: Option<Array1<T>>,
}

/// Actor-Critic specific metrics
#[derive(Debug, Clone)]
pub struct ActorCriticMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Base RL metrics
    pub base_metrics: RLOptimizationMetrics<T>,

    /// Actor loss
    pub actor_loss: T,

    /// Critic loss(es)
    pub critic_losses: Vec<T>,

    /// Temperature (for SAC)
    pub temperature: Option<T>,

    /// Temperature loss (for SAC)
    pub temperature_loss: Option<T>,

    /// Q-values statistics
    pub q_values_mean: T,
    pub q_values_std: T,

    /// Target Q-values statistics
    pub target_q_mean: T,
    pub target_q_std: T,

    /// Policy entropy
    pub policy_entropy: T,

    /// Critic gradient norms
    pub critic_grad_norms: Vec<T>,

    /// Experience replay metrics
    pub replay_buffer_size: usize,
    pub replay_sampling_time: Option<std::time::Duration>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ActorCriticMetrics<T> {
    fn default() -> Self {
        Self {
            base_metrics: RLOptimizationMetrics::default(),
            actor_loss: T::zero(),
            critic_losses: vec![T::zero()],
            temperature: None,
            temperature_loss: None,
            q_values_mean: T::zero(),
            q_values_std: T::zero(),
            target_q_mean: T::zero(),
            target_q_std: T::zero(),
            policy_entropy: T::zero(),
            critic_grad_norms: vec![T::zero()],
            replay_buffer_size: 0,
            replay_sampling_time: None,
        }
    }
}

/// Experience replay buffer entry
#[derive(Debug, Clone)]
pub struct Experience<T: Float + Debug + Send + Sync + 'static> {
    /// State (observation)
    pub state: Array1<T>,

    /// Action taken
    pub action: Array1<T>,

    /// Reward received
    pub reward: T,

    /// Next state
    pub next_state: Array1<T>,

    /// Done flag
    pub done: bool,

    /// Priority (for prioritized replay)
    pub priority: T,

    /// Additional info
    pub info: HashMap<String, T>,
}

/// A sampled replay mini-batch.
///
/// Carries the buffer indices alongside the experiences so their priorities can be
/// refreshed with the TD errors the update produces, plus the importance-sampling
/// weights that correct the bias introduced by non-uniform sampling.
#[derive(Debug, Clone)]
pub struct ReplaySample<T: Float + Debug + Send + Sync + 'static> {
    /// The sampled transitions.
    pub experiences: Vec<Experience<T>>,

    /// Buffer index of each sampled transition.
    pub indices: Vec<usize>,

    /// Importance-sampling weights `w_i = (1 / (N · P(i)))^β`, normalized by their
    /// maximum so they only ever scale gradients *down*. Uniform sampling yields
    /// all-ones.
    pub weights: Vec<T>,
}

/// Experience replay buffer with genuine prioritized sampling.
///
/// When `prioritized` is enabled, transition `i` is drawn with probability
/// `P(i) = p_iᵅ / Σ_k p_kᵅ` using an O(log N) sum tree, and the resulting bias is
/// corrected with importance weights `(1/(N·P(i)))^β` (Schaul et al., 2016). With
/// `prioritized` disabled the buffer samples uniformly and returns unit weights.
///
/// The previous implementation advertised prioritized replay but sampled uniformly
/// and never touched `alpha`/`beta`, and it panicked on an empty buffer
/// (`gen_range(0..0)`) and on `maxsize == 0` (`% 0`). Both are now impossible:
/// `new` rejects a zero capacity and `sample` returns an error on an empty buffer.
pub struct ExperienceReplayBuffer<T: Float + Debug + Send + Sync + 'static> {
    /// Buffer storage
    buffer: Vec<Experience<T>>,

    /// Maximum buffer size
    maxsize: usize,

    /// Next write position (ring buffer)
    position: usize,

    /// Whether the buffer has wrapped at least once
    is_full: bool,

    /// Prioritization exponent (0 = uniform, 1 = fully prioritized)
    alpha: T,

    /// Importance-sampling correction exponent
    beta: T,

    /// Whether prioritized sampling is active
    prioritized: bool,

    /// Sum tree over `p^α`, laid out as a complete binary tree in
    /// `[1, 2·capacity)` with leaves at `[capacity, capacity + maxsize)`.
    priority_tree: Vec<T>,

    /// Power-of-two leaf capacity of the sum tree.
    capacity: usize,

    /// Largest raw priority observed, used to seed new transitions so every
    /// transition is replayed at least once.
    max_priority: T,
}

impl<T: Float + Debug + Send + Sync + 'static> ExperienceReplayBuffer<T> {
    /// Small constant keeping every priority strictly positive.
    fn priority_epsilon() -> T {
        T::from(1e-6).unwrap_or_else(T::epsilon)
    }

    /// Create a new experience replay buffer.
    ///
    /// Returns an error for `maxsize == 0` (a zero-capacity ring buffer cannot
    /// store anything and its modular arithmetic would divide by zero).
    pub fn new(maxsize: usize, alpha: T, beta: T, prioritized: bool) -> Result<Self> {
        if maxsize == 0 {
            return Err(OptimError::InvalidConfig(
                "replay buffer capacity must be greater than zero".to_string(),
            ));
        }
        let capacity = maxsize.next_power_of_two();
        Ok(Self {
            buffer: Vec::with_capacity(maxsize),
            maxsize,
            position: 0,
            is_full: false,
            alpha,
            beta,
            prioritized,
            priority_tree: vec![T::zero(); 2 * capacity],
            capacity,
            max_priority: T::one(),
        })
    }

    /// Whether prioritized sampling is active.
    pub fn is_prioritized(&self) -> bool {
        self.prioritized
    }

    /// Total prioritized mass `Σ p^α` currently stored.
    pub fn total_priority(&self) -> T {
        self.priority_tree[1]
    }

    /// Write `p^α` into leaf `index` and propagate the change to the root.
    fn set_tree_priority(&mut self, index: usize, priority: T) {
        let clamped = priority.max(Self::priority_epsilon());
        let weighted = clamped.powf(self.alpha);

        let mut node = self.capacity + index;
        self.priority_tree[node] = weighted;
        while node > 1 {
            node /= 2;
            self.priority_tree[node] =
                self.priority_tree[2 * node] + self.priority_tree[2 * node + 1];
        }
    }

    /// Locate the leaf whose cumulative interval contains `value`.
    fn find_leaf(&self, mut value: T) -> usize {
        let mut node = 1usize;
        while node < self.capacity {
            let left = 2 * node;
            if value <= self.priority_tree[left] {
                node = left;
            } else {
                value = value - self.priority_tree[left];
                node = left + 1;
            }
        }
        (node - self.capacity).min(self.len().saturating_sub(1))
    }

    /// Add experience to buffer.
    ///
    /// A transition with a non-positive `priority` is inserted at the maximum
    /// priority seen so far, the standard PER convention that guarantees every new
    /// transition is replayed at least once.
    pub fn add(&mut self, experience: Experience<T>) {
        let priority = if experience.priority > T::zero() {
            experience.priority
        } else {
            self.max_priority
        };
        if priority > self.max_priority {
            self.max_priority = priority;
        }

        let index = self.position;
        if self.buffer.len() < self.maxsize {
            self.buffer.push(experience);
        } else {
            self.buffer[index] = experience;
            self.is_full = true;
        }

        self.set_tree_priority(index, priority);
        self.position = (self.position + 1) % self.maxsize;
    }

    /// Sample a mini-batch.
    ///
    /// Prioritized mode uses stratified sampling over `batchsize` equal segments of
    /// the total priority mass (lower variance than independent draws) and returns
    /// max-normalized importance weights. Returns an error when the buffer is
    /// empty or `batchsize` is zero instead of panicking inside the RNG.
    pub fn sample(&self, batchsize: usize) -> Result<ReplaySample<T>> {
        let available = self.len();
        if available == 0 {
            return Err(OptimError::InvalidState(
                "cannot sample from an empty replay buffer".to_string(),
            ));
        }
        if batchsize == 0 {
            return Err(OptimError::InvalidConfig(
                "replay sample size must be greater than zero".to_string(),
            ));
        }

        let sample_size = batchsize.min(available);
        let mut rng = scirs2_core::random::thread_rng();

        let mut indices = Vec::with_capacity(sample_size);
        let mut weights = Vec::with_capacity(sample_size);

        let total = self.total_priority();
        let use_priorities = self.prioritized && total > T::zero();

        if !use_priorities {
            for _ in 0..sample_size {
                indices.push(rng.gen_range(0..available));
                weights.push(T::one());
            }
        } else {
            let n = T::from(available).unwrap_or_else(T::one);
            let segment = total / T::from(sample_size).unwrap_or_else(T::one);
            let mut max_weight = T::zero();

            for k in 0..sample_size {
                let offset = T::from(k as f64 + rng.random::<f64>()).unwrap_or_else(T::zero);
                let value = (segment * offset).min(total);
                let index = self.find_leaf(value);

                // P(i) = p_iᵅ / Σ p^α, w_i = (1 / (N·P(i)))^β
                let leaf = self.priority_tree[self.capacity + index];
                let probability = if total > T::zero() {
                    (leaf / total).max(T::from(1e-12).unwrap_or_else(T::epsilon))
                } else {
                    T::one() / n
                };
                let weight = (T::one() / (n * probability)).powf(self.beta);
                if weight > max_weight {
                    max_weight = weight;
                }

                indices.push(index);
                weights.push(weight);
            }

            if max_weight > T::zero() {
                for weight in weights.iter_mut() {
                    *weight = *weight / max_weight;
                }
            }
        }

        let experiences = indices.iter().map(|&i| self.buffer[i].clone()).collect();

        Ok(ReplaySample {
            experiences,
            indices,
            weights,
        })
    }

    /// Refresh the priorities of previously sampled transitions from their TD errors.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[T]) -> Result<()> {
        if indices.len() != td_errors.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "priority update needs one TD error per index ({} vs {})",
                indices.len(),
                td_errors.len()
            )));
        }

        let available = self.len();
        for (&index, &error) in indices.iter().zip(td_errors.iter()) {
            if index >= available {
                return Err(OptimError::InvalidParameter(format!(
                    "replay index {index} out of range (buffer holds {available} transitions)"
                )));
            }
            let priority = error.abs() + Self::priority_epsilon();
            if priority > self.max_priority {
                self.max_priority = priority;
            }
            self.buffer[index].priority = priority;
            self.set_tree_priority(index, priority);
        }
        Ok(())
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        if self.is_full {
            self.maxsize
        } else {
            self.buffer.len()
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Draw one standard-normal sample via the Box–Muller transform.
///
/// `scirs2_core::random` exposes uniforms; the RL code needs Gaussians for the
/// reparameterization trick, TD3 target smoothing and the Ornstein-Uhlenbeck
/// process. Uniform noise (as used previously) has the wrong tails and the wrong
/// variance, which silently changes the exploration behaviour of every method.
fn standard_normal_sample() -> f64 {
    let mut rng = scirs2_core::random::thread_rng();
    let u1 = rng.random::<f64>().max(1e-12);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Smallest standard deviation used by the Gaussian helpers.
fn min_sigma<T: Float>() -> T {
    T::from(1e-6).unwrap_or_else(T::epsilon)
}

impl<
        T: Float
            + Debug
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + Send
            + Sync
            + ScalarOperand
            + 'static,
        P: PolicyNetwork<T>,
        V: ValueNetwork<T>,
    > ActorCriticOptimizer<T, P, V>
{
    /// Create a new Actor-Critic optimizer.
    ///
    /// When `config.use_target_networks` is set, the target actor and target
    /// critics are **populated here** by cloning the online networks (previously
    /// they stayed `None` forever, so every "target" computation silently used the
    /// online networks and the soft update had nothing to update). That is why the
    /// networks must be `Clone`.
    pub fn new(config: ActorCriticConfig<T>, actor: P, critics: Vec<V>) -> Result<Self>
    where
        P: Clone,
        V: Clone,
    {
        if critics.is_empty() {
            return Err(OptimError::InvalidConfig(
                "At least one critic required".to_string(),
            ));
        }

        let replay_buffer = ExperienceReplayBuffer::new(
            config.replay_buffer_size,
            config.per_alpha,
            config.per_beta,
            config.prioritized_replay,
        )?;

        let temperature = config.sac_config.temperature;

        let (target_actor, target_critics) = if config.use_target_networks {
            (Some(actor.clone()), Some(critics.clone()))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            actor,
            critics,
            target_actor,
            target_critics,
            temperature,
            actor_scheduler: None,
            critic_scheduler: None,
            temperature_scheduler: None,
            metrics: ActorCriticMetrics::default(),
            update_count: 0,
            policy_update_count: 0,
            replay_buffer,
            ou_noise_state: None,
        })
    }

    /// Current SAC temperature α.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// Immutable access to the replay buffer.
    pub fn replay_buffer(&self) -> &ExperienceReplayBuffer<T> {
        &self.replay_buffer
    }

    /// Mutable access to the replay buffer (e.g. to refresh priorities).
    pub fn replay_buffer_mut(&mut self) -> &mut ExperienceReplayBuffer<T> {
        &mut self.replay_buffer
    }

    /// Actor learning rate (scheduler value when configured).
    fn actor_lr(&self) -> T {
        self.actor_scheduler
            .as_ref()
            .map(|s| s.get_lr())
            .unwrap_or(self.config.base_config.policy_lr)
    }

    /// Critic learning rate (scheduler value when configured).
    fn critic_lr(&self) -> T {
        self.critic_scheduler
            .as_ref()
            .map(|s| s.get_lr())
            .unwrap_or(self.config.base_config.value_lr)
    }

    /// Clip, scale by `-lr` and apply a gradient to the actor. Returns the
    /// pre-clipping gradient norm.
    fn apply_actor_gradient(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<T> {
        let (clipped, norm) =
            clip_named_gradients(gradients, self.config.base_config.max_grad_norm);
        let step = scale_named_gradients(&clipped, -self.actor_lr());
        self.actor.update_parameters(&step)?;
        Ok(norm)
    }

    /// Update using trajectory (on-policy methods)
    pub fn update_from_trajectory(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<ActorCriticMetrics<T>> {
        match self.config.method {
            ActorCriticMethod::A2C => self.update_a2c(trajectory),
            ActorCriticMethod::A3C => self.update_a3c(trajectory),
            other => Err(OptimError::InvalidConfig(format!(
                "{other:?} is an off-policy method: use update_from_replay"
            ))),
        }
    }

    /// A2C update from trajectory.
    ///
    /// Both networks receive real gradient steps: the actor through its score
    /// oracle with `∂L/∂log π_i = −A_i/N`, the critic through its value gradient
    /// with `∂L/∂V_i = 2(V_i − R_i)/N`.
    fn update_a2c(&mut self, trajectory: TrajectoryBatch<T>) -> Result<ActorCriticMetrics<T>> {
        let mut traj = trajectory;
        let batch_len = traj.observations.nrows();
        if batch_len == 0 {
            return Err(OptimError::InvalidConfig(
                "A2C received an empty trajectory".to_string(),
            ));
        }
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;
        let two = T::one() + T::one();

        // Bootstrap on the successor state s_T, not on the last stored state.
        let next_value = match (self.critics.first(), traj.final_observation.as_ref()) {
            (Some(critic), Some(final_obs)) => {
                let mut batch = Array2::zeros((1, final_obs.len()));
                batch.row_mut(0).assign(final_obs);
                critic.evaluate_value(&batch)?[0]
            }
            _ => T::zero(),
        };

        traj.compute_advantages(
            self.config.base_config.discount_factor,
            self.config.base_config.gae_lambda,
            next_value,
        )?;

        // Critic regression against the GAE returns.
        let values = self.critics[0].evaluate_value(&traj.observations)?;
        let mut critic_loss = T::zero();
        let mut residuals = Array1::zeros(batch_len);
        for i in 0..batch_len {
            let err = values[i] - traj.returns[i];
            critic_loss = critic_loss + err * err * inv_n;
            residuals[i] = two * err * inv_n * self.config.base_config.value_loss_coeff;
        }
        let critic_grads = self.critics[0].value_gradient(&traj.observations, &residuals)?;
        let (clipped_critic, critic_norm) =
            clip_named_gradients(&critic_grads, self.config.base_config.max_grad_norm);
        let critic_step = scale_named_gradients(&clipped_critic, -self.critic_lr());
        self.critics[0].update_parameters(&critic_step)?;

        // Actor update.
        let policy_eval = self
            .actor
            .evaluate_actions(&traj.observations, &traj.actions)?;
        let mut actor_loss = T::zero();
        let mut dloss_dlogp = Array1::zeros(batch_len);
        for i in 0..batch_len {
            actor_loss = actor_loss - policy_eval.log_probs[i] * traj.advantages[i] * inv_n;
            dloss_dlogp[i] = -traj.advantages[i] * inv_n;
        }

        let mut actor_grads =
            self.actor
                .log_prob_gradient(&traj.observations, &traj.actions, &dloss_dlogp)?;
        let entropy_coeff = self.config.base_config.entropy_coeff;
        if entropy_coeff != T::zero() {
            let entropy_grad = self.actor.entropy_gradient(&traj.observations)?;
            let negated = scale_named_gradients(&entropy_grad, -entropy_coeff);
            add_named_gradients(&mut actor_grads, negated)?;
        }
        let actor_norm = self.apply_actor_gradient(&actor_grads)?;

        if self.config.use_target_networks {
            self.soft_update_targets()?;
        }

        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = vec![critic_loss];
        self.metrics.critic_grad_norms = vec![critic_norm];
        self.metrics.base_metrics.policy_grad_norm = actor_norm;
        self.metrics.base_metrics.value_grad_norm = critic_norm;
        self.metrics.policy_entropy = policy_eval.entropy.iter().copied().sum::<T>() * inv_n;
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// A3C update: asynchronous A2C. The per-worker math is identical; the
    /// asynchrony is an orchestration concern outside this optimizer.
    fn update_a3c(&mut self, trajectory: TrajectoryBatch<T>) -> Result<ActorCriticMetrics<T>> {
        self.update_a2c(trajectory)
    }

    /// A2C update from experiences (for compatibility)
    fn update_a2c_from_experiences(
        &mut self,
        experiences: &[Experience<T>],
    ) -> Result<ActorCriticMetrics<T>> {
        let trajectory = self.experiences_to_trajectory(experiences)?;
        self.update_a2c(trajectory)
    }

    /// Add experience to replay buffer
    pub fn add_experience(&mut self, experience: Experience<T>) {
        self.replay_buffer.add(experience);
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &ActorCriticMetrics<T> {
        &self.metrics
    }

    /// Advance the Ornstein-Uhlenbeck process one step and return the new state.
    ///
    /// ```text
    /// x ← x + θ (μ − x) dt + σ √dt · N(0, 1)
    /// ```
    ///
    /// The state is initialized lazily (it was previously `None` forever, so the
    /// whole routine was dead code), the noise is a genuine Gaussian rather than a
    /// uniform draw, and the timestep `dt` enters both the drift and — as `√dt` —
    /// the diffusion term, as the Ornstein-Uhlenbeck SDE requires.
    fn update_ou_noise(&mut self, action_dim: usize) -> Result<Array1<T>> {
        if action_dim == 0 {
            return Err(OptimError::InvalidConfig(
                "OU noise requires a positive action dimension".to_string(),
            ));
        }

        let theta = self.config.ddpg_config.ou_noise_theta;
        let sigma = self.config.ddpg_config.ou_noise_sigma;
        let dt = self.config.ddpg_config.ou_noise_dt;
        let mu = self.config.ddpg_config.ou_noise_mu;
        let sqrt_dt = dt.max(T::zero()).sqrt();

        let needs_reset = match self.ou_noise_state {
            Some(ref state) => state.len() != action_dim,
            None => true,
        };
        if needs_reset {
            self.ou_noise_state = Some(Array1::from_elem(action_dim, mu));
        }

        let state = match self.ou_noise_state {
            Some(ref mut state) => state,
            None => {
                return Err(OptimError::InvalidState(
                    "OU noise state failed to initialize".to_string(),
                ))
            }
        };

        for value in state.iter_mut() {
            let noise = T::from(standard_normal_sample()).unwrap_or_else(T::zero);
            let drift = theta * (mu - *value) * dt;
            *value = *value + drift + sigma * sqrt_dt * noise;
        }

        Ok(state.clone())
    }

    /// Reset the Ornstein-Uhlenbeck exploration state (call between episodes).
    pub fn reset_ou_noise(&mut self) {
        self.ou_noise_state = None;
    }

    /// Deterministic actions with Ornstein-Uhlenbeck exploration noise **added**,
    /// clipped to the configured action bounds.
    ///
    /// This is the missing half of DDPG exploration: the OU process used to be
    /// advanced (at best) without its output ever reaching an action.
    pub fn explore_actions(&mut self, states: &Array2<T>) -> Result<Array2<T>> {
        let distribution = self.actor.get_action_distribution(states)?;
        let mut actions = distribution.mean.ok_or_else(|| {
            OptimError::InvalidConfig(
                "OU exploration requires a distribution with a mean action".to_string(),
            )
        })?;

        let action_dim = actions.ncols();
        let bounds = self.config.ddpg_config.action_bounds;

        for i in 0..actions.nrows() {
            let noise = self.update_ou_noise(action_dim)?;
            for j in 0..action_dim {
                let mut value = actions[[i, j]] + noise[j];
                if let Some((low, high)) = bounds {
                    value = value.max(low).min(high);
                }
                actions[[i, j]] = value;
            }
        }

        Ok(actions)
    }

    // Helper methods

    fn extract_states(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        if experiences.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Empty experience batch".to_string(),
            ));
        }

        let batchsize = experiences.len();
        let state_dim = experiences[0].state.len();
        let mut states = Array2::zeros((batchsize, state_dim));

        for (i, exp) in experiences.iter().enumerate() {
            if exp.state.len() != state_dim {
                return Err(OptimError::DimensionMismatch(
                    "inconsistent state dimensions in experience batch".to_string(),
                ));
            }
            states.row_mut(i).assign(&exp.state);
        }

        Ok(states)
    }

    fn extract_actions(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        if experiences.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Empty experience batch".to_string(),
            ));
        }

        let batchsize = experiences.len();
        let action_dim = experiences[0].action.len();
        let mut actions = Array2::zeros((batchsize, action_dim));

        for (i, exp) in experiences.iter().enumerate() {
            if exp.action.len() != action_dim {
                return Err(OptimError::DimensionMismatch(
                    "inconsistent action dimensions in experience batch".to_string(),
                ));
            }
            actions.row_mut(i).assign(&exp.action);
        }

        Ok(actions)
    }

    fn extract_rewards(&self, experiences: &[Experience<T>]) -> Result<Array1<T>> {
        let rewards: Vec<T> = experiences.iter().map(|exp| exp.reward).collect();
        Ok(Array1::from_vec(rewards))
    }

    fn extract_next_states(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        if experiences.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Empty experience batch".to_string(),
            ));
        }

        let batchsize = experiences.len();
        let state_dim = experiences[0].next_state.len();
        let mut next_states = Array2::zeros((batchsize, state_dim));

        for (i, exp) in experiences.iter().enumerate() {
            if exp.next_state.len() != state_dim {
                return Err(OptimError::DimensionMismatch(
                    "inconsistent next-state dimensions in experience batch".to_string(),
                ));
            }
            next_states.row_mut(i).assign(&exp.next_state);
        }

        Ok(next_states)
    }

    fn extract_dones(&self, experiences: &[Experience<T>]) -> Result<Array1<bool>> {
        let dones: Vec<bool> = experiences.iter().map(|exp| exp.done).collect();
        Ok(Array1::from_vec(dones))
    }

    fn compute_critic_loss(&self, q_values: &Array1<T>, targetq: &Array1<T>) -> Result<T> {
        Ok((q_values - targetq)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(T::zero()))
    }

    /// Sample actions from action distribution
    fn sample_actions_from_distribution(
        &self,
        action_dist: &ActionDistribution<T>,
    ) -> Result<Array2<T>> {
        match action_dist.distribution_type {
            DistributionType::Gaussian => {
                if let (Some(ref mean), Some(ref std)) = (&action_dist.mean, &action_dist.std) {
                    let mut actions = mean.clone();

                    // Reparameterization trick: action = mean + std · z, z ~ N(0,1),
                    // with σ clamped away from zero so a collapsed policy cannot
                    // produce NaNs downstream.
                    for ((action, &m), &s) in actions.iter_mut().zip(mean.iter()).zip(std.iter()) {
                        let sigma = s.max(min_sigma::<T>());
                        let noise = T::from(standard_normal_sample()).unwrap_or_else(T::zero);
                        *action = m + sigma * noise;
                    }

                    Ok(actions)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid Gaussian distribution".to_string(),
                    ))
                }
            }
            DistributionType::Categorical => {
                if let Some(ref logits) = action_dist.logits {
                    // Sample from categorical distribution
                    let mut actions = Array2::zeros(logits.dim());

                    for i in 0..logits.nrows() {
                        // Convert logits to probabilities (stabilized softmax)
                        let row = logits.row(i);
                        let max_logit = row.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
                        let exp_logits: Vec<T> =
                            row.iter().map(|&x| (x - max_logit).exp()).collect();
                        let sum_exp: T = exp_logits.iter().cloned().sum();
                        if !(sum_exp > T::zero()) {
                            return Err(OptimError::ComputationError(
                                "categorical logits produced a degenerate distribution".to_string(),
                            ));
                        }

                        // Sample from the categorical distribution via inverse-CDF
                        // (walk the cumulative probabilities until they exceed a
                        // uniform draw) so the policy is genuinely stochastic rather
                        // than a deterministic argmax.
                        let u = T::from(scirs2_core::random::thread_rng().random::<f64>())
                            .unwrap_or_else(|| T::zero());
                        let mut cumulative = T::zero();
                        let mut sampled_idx = exp_logits.len().saturating_sub(1);
                        for (j, &prob) in exp_logits.iter().enumerate() {
                            cumulative = cumulative + prob / sum_exp;
                            if u <= cumulative {
                                sampled_idx = j;
                                break;
                            }
                        }

                        actions[[i, sampled_idx]] = T::one();
                    }

                    Ok(actions)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid categorical distribution".to_string(),
                    ))
                }
            }
            other => Err(OptimError::UnsupportedOperation(format!(
                "sampling from a {other:?} action distribution is not implemented"
            ))),
        }
    }

    /// Compute log probabilities of actions under distribution
    fn compute_log_probabilities(
        &self,
        action_dist: &ActionDistribution<T>,
        actions: &Array2<T>,
    ) -> Result<Array1<T>> {
        match action_dist.distribution_type {
            DistributionType::Gaussian => {
                if let (Some(ref mean), Some(ref std)) = (&action_dist.mean, &action_dist.std) {
                    let mut log_probs = Array1::zeros(actions.nrows());

                    let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
                    // ½·ln(2π) — the normalizing constant of a unit Gaussian. The
                    // previous code computed ln(0.5·2·π) = ln(π), which is a
                    // different number and shifted every log-probability.
                    let half_log_two_pi =
                        T::from(0.5 * (2.0 * std::f64::consts::PI).ln()).unwrap_or_else(T::zero);

                    for i in 0..actions.nrows() {
                        let mut log_prob = T::zero();

                        for j in 0..actions.ncols() {
                            let action = actions[[i, j]];
                            let mu = mean[[i, j]];
                            // σ clamped away from zero: 1/σ and ln σ are otherwise
                            // ±inf for a collapsed policy.
                            let sigma = std[[i, j]].max(min_sigma::<T>());

                            // log N(x; μ, σ) = −½((x−μ)/σ)² − ln σ − ½ ln(2π)
                            let normalized_diff = (action - mu) / sigma;
                            log_prob = log_prob
                                - half * normalized_diff * normalized_diff
                                - sigma.ln()
                                - half_log_two_pi;
                        }

                        log_probs[i] = log_prob;
                    }

                    Ok(log_probs)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid Gaussian distribution".to_string(),
                    ))
                }
            }
            DistributionType::Categorical => {
                if let Some(ref logits) = action_dist.logits {
                    let mut log_probs = Array1::zeros(actions.nrows());

                    for i in 0..actions.nrows() {
                        // Find the action index (one-hot encoded)
                        let mut action_idx = 0;
                        for j in 0..actions.ncols() {
                            if actions[[i, j]] > T::from(0.5).unwrap_or_else(|| T::zero()) {
                                action_idx = j;
                                break;
                            }
                        }

                        // Compute log softmax
                        let row = logits.row(i);
                        let max_logit = row.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
                        let log_sum_exp = (row.iter().map(|&x| (x - max_logit).exp()).sum::<T>())
                            .ln()
                            + max_logit;

                        log_probs[i] = logits[[i, action_idx]] - log_sum_exp;
                    }

                    Ok(log_probs)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid categorical distribution".to_string(),
                    ))
                }
            }
            other => Err(OptimError::UnsupportedOperation(format!(
                "log-probabilities for a {other:?} action distribution are not implemented"
            ))),
        }
    }

    /// Polyak-average the target networks towards the online networks.
    ///
    /// `update_parameters` takes an additive **delta**, so the soft update
    /// `target ← τ·online + (1−τ)·target` is applied as the delta
    /// `τ·(online − target)`. The previous code passed the *absolute* target
    /// parameters into that additive API, which doubled the targets on every call
    /// instead of averaging them.
    pub fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.target_update_rate;

        if let Some(ref mut target_critics) = self.target_critics {
            for (target_critic, online_critic) in target_critics.iter_mut().zip(self.critics.iter())
            {
                let online_params = online_critic.get_parameters();
                let target_params = target_critic.get_parameters();
                let mut deltas: HashMap<String, Array1<T>> =
                    HashMap::with_capacity(target_params.len());

                for (name, online_param) in online_params {
                    if let Some(target_param) = target_params.get(&name) {
                        if target_param.len() != online_param.len() {
                            return Err(OptimError::DimensionMismatch(format!(
                                "target critic parameter '{name}' has length {} but the online \
                                 critic has {}",
                                target_param.len(),
                                online_param.len()
                            )));
                        }
                        let mut delta = Array1::zeros(online_param.len());
                        for i in 0..online_param.len() {
                            delta[i] = tau * (online_param[i] - target_param[i]);
                        }
                        deltas.insert(name, delta);
                    }
                }

                target_critic.update_parameters(&deltas)?;
            }
        }

        if let Some(ref mut target_actor) = self.target_actor {
            let online_params = self.actor.get_parameters();
            let target_params = target_actor.get_parameters();
            let mut deltas: HashMap<String, Array1<T>> =
                HashMap::with_capacity(target_params.len());

            for (name, online_param) in online_params {
                if let Some(target_param) = target_params.get(&name) {
                    if target_param.len() != online_param.len() {
                        return Err(OptimError::DimensionMismatch(format!(
                            "target actor parameter '{name}' has length {} but the online actor \
                             has {}",
                            target_param.len(),
                            online_param.len()
                        )));
                    }
                    let mut delta = Array1::zeros(online_param.len());
                    for i in 0..online_param.len() {
                        delta[i] = tau * (online_param[i] - target_param[i]);
                    }
                    deltas.insert(name, delta);
                }
            }

            target_actor.update_parameters(&deltas)?;
        }

        Ok(())
    }

    /// Copy the online parameters into the targets (`τ = 1`).
    pub fn hard_update_targets(&mut self) -> Result<()> {
        let tau = self.config.target_update_rate;
        self.config.target_update_rate = T::one();
        let result = self.soft_update_targets();
        self.config.target_update_rate = tau;
        result
    }

    fn experiences_to_trajectory(
        &self,
        experiences: &[Experience<T>],
    ) -> Result<TrajectoryBatch<T>> {
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Log-probs/values are unknown for replayed transitions; the on-policy
        // path recomputes both from the current networks.
        let log_probs = Array1::zeros(experiences.len());
        let values = Array1::zeros(experiences.len());

        let trajectory = TrajectoryBatch::new(states, actions, log_probs, rewards, values, dones)?;

        match experiences.last() {
            Some(last) => trajectory.with_final_observation(last.next_state.clone()),
            None => Ok(trajectory),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Off-policy methods — these require an *action-value* critic
// ─────────────────────────────────────────────────────────────────────────────

impl<
        T: Float
            + Debug
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + Send
            + Sync
            + ScalarOperand
            + 'static,
        P: PolicyNetwork<T>,
        V: QNetwork<T>,
    > ActorCriticOptimizer<T, P, V>
{
    /// Update using experience replay.
    ///
    /// Available only for `V: QNetwork` — SAC, TD3 and DDPG are all built on
    /// `Q(s, a)`, and the deterministic policy gradient needs `∇_a Q(s, a)`.
    pub fn update_from_replay(&mut self, batchsize: usize) -> Result<ActorCriticMetrics<T>> {
        if self.replay_buffer.len() < batchsize {
            return Err(OptimError::InvalidConfig(format!(
                "not enough experiences in buffer: have {}, need {batchsize}",
                self.replay_buffer.len()
            )));
        }

        let sample = self.replay_buffer.sample(batchsize)?;

        match self.config.method {
            ActorCriticMethod::SAC => self.update_sac(&sample),
            ActorCriticMethod::TD3 => self.update_td3(&sample),
            ActorCriticMethod::DDPG => self.update_ddpg(&sample),
            other => Err(OptimError::UnsupportedOperation(format!(
                "{other:?} does not support experience-replay updates"
            ))),
        }
    }

    /// The target critics, falling back to the online critics when target
    /// networks are disabled (the standard "no target network" ablation).
    fn effective_target_critics(&self) -> &[V] {
        match self.target_critics {
            Some(ref critics) => critics.as_slice(),
            None => self.critics.as_slice(),
        }
    }

    /// The target actor, falling back to the online actor when target networks
    /// are disabled.
    fn effective_target_actor(&self) -> &P {
        match self.target_actor {
            Some(ref actor) => actor,
            None => &self.actor,
        }
    }

    /// `Q(s, a)` — the action argument genuinely participates.
    fn compute_q_values(
        &self,
        critic: &V,
        states: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<Array1<T>> {
        critic.evaluate_q(states, actions)
    }

    /// Regress the first `n_critics` critics onto `targets`, returning the losses
    /// and the TD errors of the first critic (used to refresh replay priorities).
    fn update_critics(
        &mut self,
        states: &Array2<T>,
        actions: &Array2<T>,
        targets: &Array1<T>,
        weights: &[T],
        n_critics: usize,
    ) -> Result<(Vec<T>, Vec<T>, Array1<T>)> {
        let batch_len = states.nrows();
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;
        let two = T::one() + T::one();
        let lr = self.critic_lr();
        let max_norm = self.config.base_config.max_grad_norm;

        let mut losses = Vec::new();
        let mut norms = Vec::new();
        let mut td_errors = Array1::zeros(batch_len);

        for index in 0..n_critics.min(self.critics.len()) {
            let q_values = self.critics[index].evaluate_q(states, actions)?;

            let mut loss = T::zero();
            let mut residuals = Array1::zeros(batch_len);
            for i in 0..batch_len {
                // Importance-sampling weight from prioritized replay (1 otherwise).
                let weight = weights.get(i).copied().unwrap_or_else(T::one);
                let error = q_values[i] - targets[i];
                loss = loss + weight * error * error * inv_n;
                residuals[i] = two * weight * error * inv_n;
                if index == 0 {
                    td_errors[i] = error;
                }
            }

            let gradients = self.critics[index].q_gradient(states, actions, &residuals)?;
            let (clipped, norm) = clip_named_gradients(&gradients, max_norm);
            let step = scale_named_gradients(&clipped, -lr);
            self.critics[index].update_parameters(&step)?;

            losses.push(loss);
            norms.push(norm);
        }

        Ok((losses, norms, td_errors))
    }

    /// Deterministic policy gradient actor update (DDPG / TD3).
    ///
    /// `L = −(1/N) Σ Q(sᵢ, μ(sᵢ))`, so `∂L/∂μᵢ = −∇_a Q(sᵢ, μ(sᵢ))/N` and the
    /// parameter gradient follows by the chain rule through the actor's
    /// `mean_action_gradient` oracle. This is the DPG theorem, and it is why the
    /// critic must be a `QNetwork`.
    fn update_actor_dpg(&mut self, states: &Array2<T>) -> Result<T> {
        let distribution = self.actor.get_action_distribution(states)?;
        let mean_actions = distribution.mean.ok_or_else(|| {
            OptimError::UnsupportedOperation(
                "the deterministic policy gradient requires an actor with a mean action"
                    .to_string(),
            )
        })?;

        let batch_len = states.nrows();
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;

        let q_values = self.critics[0].evaluate_q(states, &mean_actions)?;
        let actor_loss = -q_values.iter().copied().sum::<T>() * inv_n;

        let dq_da = self.critics[0].action_gradient(states, &mean_actions)?;
        let weights = dq_da.mapv(|g| -g * inv_n);

        let gradients = self.actor.mean_action_gradient(states, &weights)?;
        let norm = self.apply_actor_gradient(&gradients)?;
        self.metrics.base_metrics.policy_grad_norm = norm;

        self.policy_update_count += 1;
        Ok(actor_loss)
    }

    /// Reparameterized SAC actor update.
    ///
    /// `L = (1/N) Σ [α log π(ãᵢ|sᵢ) − Q(sᵢ, ãᵢ)]` with `ã = μ + σ·z`. The
    /// parameters enter twice — directly through `log π` and through the sampled
    /// action — so both terms are assembled:
    ///
    /// ```text
    /// ∇_θ L = ∇_θ [α log π]|_{a fixed}  +  Σ_ij (∂L/∂a_ij) ∂a_ij/∂θ
    /// ∂L/∂a_ij = (α ∂log π/∂a_ij − ∂Q/∂a_ij)/N ,  ∂log π/∂a = −(a−μ)/σ²
    /// ```
    ///
    /// Returns `(actor_loss, mean_entropy_estimate)`.
    fn update_actor_sac(&mut self, states: &Array2<T>) -> Result<(T, T)> {
        let distribution = self.actor.get_action_distribution(states)?;
        let sampled_actions = self.sample_actions_from_distribution(&distribution)?;
        let log_probs = self.compute_log_probabilities(&distribution, &sampled_actions)?;

        let batch_len = states.nrows();
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;
        let alpha = self.temperature;

        // Twin-critic minimum, tracking which critic won so the action gradient
        // comes from the same critic the value did.
        let q_first = self.critics[0].evaluate_q(states, &sampled_actions)?;
        let grad_first = self.critics[0].action_gradient(states, &sampled_actions)?;
        let (q_values, dq_da) = if self.critics.len() >= 2 {
            let q_second = self.critics[1].evaluate_q(states, &sampled_actions)?;
            let grad_second = self.critics[1].action_gradient(states, &sampled_actions)?;
            let mut q = Array1::zeros(batch_len);
            let mut grad = Array2::zeros(grad_first.dim());
            for i in 0..batch_len {
                let use_first = q_first[i] <= q_second[i];
                q[i] = if use_first { q_first[i] } else { q_second[i] };
                for j in 0..grad_first.ncols() {
                    grad[[i, j]] = if use_first {
                        grad_first[[i, j]]
                    } else {
                        grad_second[[i, j]]
                    };
                }
            }
            (q, grad)
        } else {
            (q_first, grad_first)
        };

        let mut actor_loss = T::zero();
        let mut mean_entropy = T::zero();
        for i in 0..batch_len {
            actor_loss = actor_loss + (alpha * log_probs[i] - q_values[i]) * inv_n;
            mean_entropy = mean_entropy - log_probs[i] * inv_n;
        }

        // Direct dependence: ∂/∂θ (α/N) Σ log π(ãᵢ|sᵢ) with ã held fixed.
        let coefficients = Array1::from_elem(batch_len, alpha * inv_n);
        let mut gradients =
            self.actor
                .log_prob_gradient(states, &sampled_actions, &coefficients)?;

        // Path through the sampled action.
        let mean = distribution.mean.as_ref().ok_or_else(|| {
            OptimError::UnsupportedOperation(
                "the SAC actor update requires a Gaussian distribution mean".to_string(),
            )
        })?;
        let std = distribution.std.as_ref().ok_or_else(|| {
            OptimError::UnsupportedOperation(
                "the SAC actor update requires a Gaussian distribution std".to_string(),
            )
        })?;

        let mut action_weights = Array2::zeros(sampled_actions.dim());
        for i in 0..batch_len {
            for j in 0..sampled_actions.ncols() {
                let sigma = std[[i, j]].max(min_sigma::<T>());
                let dlogp_da = -(sampled_actions[[i, j]] - mean[[i, j]]) / (sigma * sigma);
                action_weights[[i, j]] = (alpha * dlogp_da - dq_da[[i, j]]) * inv_n;
            }
        }
        let path_gradients = self.actor.mean_action_gradient(states, &action_weights)?;
        add_named_gradients(&mut gradients, path_gradients)?;

        let norm = self.apply_actor_gradient(&gradients)?;
        self.metrics.base_metrics.policy_grad_norm = norm;

        self.policy_update_count += 1;
        Ok((actor_loss, mean_entropy))
    }

    /// SAC temperature (α) update.
    ///
    /// `J(α) = α·(H − H̄)` where `H` is the current policy entropy and `H̄` the
    /// target, so `dJ/dα = H − H̄` and gradient descent gives
    /// `α ← α − lr·(H − H̄)`. The sign matters: an entropy **above** the target
    /// must *reduce* α (less exploration bonus needed). The previous code used
    /// `H̄ − H`, which increased α exactly when it should have decreased it and
    /// drove the temperature away from its target.
    fn update_temperature_sac(&mut self, current_entropy: T, action_dim: usize) -> Result<T> {
        if !self.config.sac_config.auto_entropy_tuning {
            return Ok(T::zero());
        }

        let target_entropy = match self.config.sac_config.target_entropy {
            Some(value) => value,
            None => -T::from(action_dim).ok_or_else(|| {
                OptimError::ComputationError(
                    "failed to convert action dimension to scalar".to_string(),
                )
            })?,
        };

        let gradient = current_entropy - target_entropy;
        let temperature_loss = self.temperature * gradient;

        let lr = self
            .temperature_scheduler
            .as_ref()
            .map(|s| s.get_lr())
            .unwrap_or(self.config.sac_config.temperature_lr);

        let floor = T::from(1e-6).unwrap_or_else(T::epsilon);
        self.temperature = (self.temperature - lr * gradient).max(floor);

        Ok(temperature_loss)
    }

    /// Soft TD target `r + γ(1−done)·[min_i Q_target_i(s', ã') − α log π(ã'|s')]`
    /// with `ã' ~ π(·|s')` — the entropy-regularized Bellman backup of SAC.
    fn compute_target_q_sac(
        &self,
        next_states: &Array2<T>,
        rewards: &Array1<T>,
        dones: &Array1<bool>,
    ) -> Result<Array1<T>> {
        if self.critics.is_empty() {
            return Ok(rewards.clone());
        }

        let gamma = self.config.base_config.discount_factor;
        let actor = self.effective_target_actor();
        let distribution = actor.get_action_distribution(next_states)?;
        let next_actions = self.sample_actions_from_distribution(&distribution)?;
        let next_log_probs = self.compute_log_probabilities(&distribution, &next_actions)?;

        let critics = self.effective_target_critics();
        let mut q_next = critics[0].evaluate_q(next_states, &next_actions)?;
        if critics.len() >= 2 {
            let q_second = critics[1].evaluate_q(next_states, &next_actions)?;
            for i in 0..q_next.len() {
                q_next[i] = q_next[i].min(q_second[i]);
            }
        }

        let mut targets = rewards.clone();
        for i in 0..targets.len() {
            let not_done = if dones[i] { T::zero() } else { T::one() };
            let soft_value = q_next[i] - self.temperature * next_log_probs[i];
            targets[i] = targets[i] + gamma * not_done * soft_value;
        }
        Ok(targets)
    }

    /// SAC update: twin critics, reparameterized actor, tuned temperature.
    fn update_sac(&mut self, sample: &ReplaySample<T>) -> Result<ActorCriticMetrics<T>> {
        let experiences = sample.experiences.as_slice();
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        let targets = self.compute_target_q_sac(&next_states, &rewards, &dones)?;
        let n_critics = self.critics.len();
        let (critic_losses, critic_norms, td_errors) =
            self.update_critics(&states, &actions, &targets, &sample.weights, n_critics)?;

        let (actor_loss, mean_entropy) = self.update_actor_sac(&states)?;
        let temperature_loss = self.update_temperature_sac(mean_entropy, actions.ncols())?;

        if self.config.use_target_networks
            && self
                .update_count
                .is_multiple_of(self.config.sac_config.target_update_freq.max(1))
        {
            self.soft_update_targets()?;
        }

        if self.replay_buffer.is_prioritized() {
            let errors: Vec<T> = td_errors.iter().copied().collect();
            self.replay_buffer
                .update_priorities(&sample.indices, &errors)?;
        }

        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = critic_losses;
        self.metrics.critic_grad_norms = critic_norms;
        self.metrics.temperature = Some(self.temperature);
        self.metrics.temperature_loss = Some(temperature_loss);
        self.metrics.policy_entropy = mean_entropy;
        self.metrics.target_q_mean = mean_of(&targets);
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// TD3 update: clipped double-Q targets, target-policy smoothing and delayed
    /// actor updates.
    fn update_td3(&mut self, sample: &ReplaySample<T>) -> Result<ActorCriticMetrics<T>> {
        if self.critics.len() < 2 {
            return Err(OptimError::InvalidConfig(
                "TD3 requires two critics (clipped double-Q learning); configure n_critics = 2 \
                 and pass two critics"
                    .to_string(),
            ));
        }

        let experiences = sample.experiences.as_slice();
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Target action with clipped Gaussian smoothing noise.
        let target_distribution = self
            .effective_target_actor()
            .get_action_distribution(&next_states)?;
        let mut target_actions = target_distribution.mean.clone().ok_or_else(|| {
            OptimError::UnsupportedOperation(
                "TD3 requires a deterministic (mean-bearing) target actor".to_string(),
            )
        })?;

        let policy_noise = self.config.td3_config.policy_noise;
        let noise_clip = self.config.td3_config.noise_clip;
        for action in target_actions.iter_mut() {
            let raw = T::from(standard_normal_sample()).unwrap_or_else(T::zero) * policy_noise;
            let clipped = raw.max(-noise_clip).min(noise_clip);
            *action = *action + clipped;
            if let Some((low, high)) = self.config.td3_config.action_bounds {
                *action = action.max(low).min(high);
            }
        }

        // min(Q_target1, Q_target2) at the smoothed target action.
        let target_critics = self.effective_target_critics();
        let target_q1 = target_critics[0].evaluate_q(&next_states, &target_actions)?;
        let target_q2 = target_critics[1].evaluate_q(&next_states, &target_actions)?;

        let gamma = self.config.base_config.discount_factor;
        let mut td_targets = Array1::zeros(rewards.len());
        for i in 0..rewards.len() {
            let min_q = target_q1[i].min(target_q2[i]);
            let not_done = if dones[i] { T::zero() } else { T::one() };
            td_targets[i] = rewards[i] + gamma * not_done * min_q;
        }

        let (critic_losses, critic_norms, td_errors) =
            self.update_critics(&states, &actions, &td_targets, &sample.weights, 2)?;

        // Delayed policy updates.
        let delay = self.config.td3_config.policy_delay.max(1);
        let actor_loss = if self.update_count.is_multiple_of(delay) {
            let loss = self.update_actor_dpg(&states)?;
            if self.config.use_target_networks {
                self.soft_update_targets()?;
            }
            loss
        } else {
            self.metrics.actor_loss
        };

        if self.replay_buffer.is_prioritized() {
            let errors: Vec<T> = td_errors.iter().copied().collect();
            self.replay_buffer
                .update_priorities(&sample.indices, &errors)?;
        }

        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = critic_losses;
        self.metrics.critic_grad_norms = critic_norms;
        self.metrics.target_q_mean = mean_of(&td_targets);
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// DDPG update: single critic, deterministic actor, Polyak targets.
    fn update_ddpg(&mut self, sample: &ReplaySample<T>) -> Result<ActorCriticMetrics<T>> {
        let experiences = sample.experiences.as_slice();
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        let target_distribution = self
            .effective_target_actor()
            .get_action_distribution(&next_states)?;
        let mut target_actions = target_distribution.mean.clone().ok_or_else(|| {
            OptimError::UnsupportedOperation(
                "DDPG requires a deterministic (mean-bearing) target actor".to_string(),
            )
        })?;
        if let Some((low, high)) = self.config.ddpg_config.action_bounds {
            target_actions.mapv_inplace(|a| a.max(low).min(high));
        }

        let target_q =
            self.effective_target_critics()[0].evaluate_q(&next_states, &target_actions)?;

        let gamma = self.config.base_config.discount_factor;
        let mut td_targets = Array1::zeros(rewards.len());
        for i in 0..rewards.len() {
            let not_done = if dones[i] { T::zero() } else { T::one() };
            td_targets[i] = rewards[i] + gamma * not_done * target_q[i];
        }

        let (critic_losses, critic_norms, td_errors) =
            self.update_critics(&states, &actions, &td_targets, &sample.weights, 1)?;

        let actor_loss = self.update_actor_dpg(&states)?;

        if self.config.use_target_networks {
            self.soft_update_targets()?;
        }

        if self.replay_buffer.is_prioritized() {
            let errors: Vec<T> = td_errors.iter().copied().collect();
            self.replay_buffer
                .update_priorities(&sample.indices, &errors)?;
        }

        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = critic_losses;
        self.metrics.critic_grad_norms = critic_norms;
        self.metrics.target_q_mean = mean_of(&td_targets);
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }
}

/// Mean of an array, or zero for an empty one.
fn mean_of<T: Float + Debug + Send + Sync + 'static>(values: &Array1<T>) -> T {
    if values.is_empty() {
        return T::zero();
    }
    let mut total = T::zero();
    for &v in values.iter() {
        total = total + v;
    }
    match T::from(values.len()) {
        Some(count) if count > T::zero() => total / count,
        _ => T::zero(),
    }
}

// Import slice syntax
use scirs2_core::ndarray::s;
// use statrs::statistics::Statistics; // statrs not available

#[cfg(test)]
mod tests {
    use super::{
        ActorCriticConfig, ActorCriticMethod, ActorCriticOptimizer, Experience,
        ExperienceReplayBuffer, SACConfig,
    };
    use crate::error::Result;
    use crate::reinforcement_learning::{
        ActionDistribution, DistributionType, PolicyEvaluation, PolicyNetwork, QNetwork,
        ValueNetwork,
    };
    use scirs2_core::ndarray::{Array1, Array2};
    use std::collections::HashMap;

    // ── Minimal mock networks ─────────────────────────────────────────────

    /// Action-value critic returning a constant `q`, with a real (zero) gradient
    /// API so the update paths can run end to end.
    #[derive(Clone)]
    struct MockQ {
        q: f64,
    }

    impl ValueNetwork<f64> for MockQ {
        fn evaluate_value(&self, _obs: &Array2<f64>) -> Result<Array1<f64>> {
            Err(crate::error::OptimError::UnsupportedOperation(
                "MockQ is an action-value critic".to_string(),
            ))
        }
        fn update_parameters(&mut self, _: &HashMap<String, Array1<f64>>) -> Result<()> {
            Ok(())
        }
        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            HashMap::new()
        }
    }

    impl QNetwork<f64> for MockQ {
        fn evaluate_q(&self, states: &Array2<f64>, _actions: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(states.nrows(), self.q))
        }
        fn q_gradient(
            &self,
            _states: &Array2<f64>,
            _actions: &Array2<f64>,
            _residuals: &Array1<f64>,
        ) -> Result<HashMap<String, Array1<f64>>> {
            Ok(HashMap::new())
        }
        fn action_gradient(
            &self,
            states: &Array2<f64>,
            actions: &Array2<f64>,
        ) -> Result<Array2<f64>> {
            Ok(Array2::zeros((states.nrows(), actions.ncols())))
        }
    }

    #[derive(Clone)]
    struct MockPolicy;

    impl PolicyNetwork<f64> for MockPolicy {
        fn evaluate_actions(
            &self,
            obs: &Array2<f64>,
            _: &Array2<f64>,
        ) -> Result<PolicyEvaluation<f64>> {
            Ok(PolicyEvaluation {
                log_probs: Array1::zeros(obs.nrows()),
                entropy: Array1::zeros(obs.nrows()),
                metrics: HashMap::new(),
            })
        }
        fn get_action_distribution(&self, obs: &Array2<f64>) -> Result<ActionDistribution<f64>> {
            let n = obs.nrows();
            Ok(ActionDistribution {
                mean: Some(Array2::zeros((n, 2))),
                std: Some(Array2::from_elem((n, 2), 1.0_f64)),
                logits: None,
                distribution_type: DistributionType::Gaussian,
            })
        }
        fn update_parameters(&mut self, _: &HashMap<String, Array1<f64>>) -> Result<()> {
            Ok(())
        }
        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            HashMap::new()
        }
    }

    /// Build an optimizer with `temperature = 0` so the SAC soft target reduces to
    /// the plain TD target and the assertions stay deterministic.
    fn opt_with(critics: Vec<MockQ>) -> ActorCriticOptimizer<f64, MockPolicy, MockQ> {
        let n = critics.len();
        let cfg = ActorCriticConfig::<f64> {
            n_critics: n,
            sac_config: SACConfig::<f64> {
                temperature: 0.0,
                ..SACConfig::default()
            },
            ..ActorCriticConfig::default()
        };
        ActorCriticOptimizer::new(cfg, MockPolicy, critics).expect("construction")
    }

    // ── compute_target_q_sac — TD bootstrap r + γ(1−done)·min Q ──────────

    #[test]
    fn test_target_q_sac_twin_critics_take_minimum() {
        // critics return 3 and 5 → twin-critic min is 3, not 5
        let opt = opt_with(vec![MockQ { q: 3.0 }, MockQ { q: 5.0 }]);
        let states = Array2::zeros((3_usize, 4));
        let rewards = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let dones = Array1::from_vec(vec![false, true, false]);

        let targets = opt
            .compute_target_q_sac(&states, &rewards, &dones)
            .expect("compute_target_q_sac");

        let gamma = 0.99_f64; // default discount factor
        let q_min = 3.0_f64;
        assert!((targets[0] - (1.0 + gamma * q_min)).abs() < 1e-9);
        assert!(
            (targets[1] - 2.0).abs() < 1e-9,
            "done target[1]={} should equal reward=2.0",
            targets[1]
        );
        assert!((targets[2] - (3.0 + gamma * q_min)).abs() < 1e-9);
    }

    #[test]
    fn test_target_q_sac_single_critic_bootstrap() {
        let opt = opt_with(vec![MockQ { q: 4.0 }]);
        let states = Array2::zeros((2_usize, 4));
        let rewards = Array1::from_vec(vec![2.0_f64, 3.0]);
        let dones = Array1::from_vec(vec![false, false]);

        let targets = opt
            .compute_target_q_sac(&states, &rewards, &dones)
            .expect("compute_target_q_sac");

        let gamma = 0.99_f64;
        assert!((targets[0] - (2.0 + gamma * 4.0)).abs() < 1e-9);
        assert!((targets[1] - (3.0 + gamma * 4.0)).abs() < 1e-9);
    }

    #[test]
    fn test_target_q_sac_all_done_no_bootstrap() {
        let opt = opt_with(vec![MockQ { q: 99.0 }, MockQ { q: 99.0 }]);
        let states = Array2::zeros((3_usize, 2));
        let rewards = Array1::from_vec(vec![5.0_f64, 6.0, 7.0]);
        let dones = Array1::from_vec(vec![true, true, true]);

        let targets = opt
            .compute_target_q_sac(&states, &rewards, &dones)
            .expect("compute_target_q_sac");

        for i in 0..3 {
            assert!(
                (targets[i] - rewards[i]).abs() < 1e-12,
                "done target[{i}]={} should equal reward={}",
                targets[i],
                rewards[i]
            );
        }
    }

    // ── F48 regression: SAC temperature sign ─────────────────────────────

    #[test]
    fn test_temperature_decreases_when_entropy_exceeds_target() {
        let cfg = ActorCriticConfig::<f64> {
            method: ActorCriticMethod::SAC,
            sac_config: SACConfig::<f64> {
                temperature: 0.2,
                auto_entropy_tuning: true,
                target_entropy: Some(-2.0),
                temperature_lr: 0.1,
                ..SACConfig::default()
            },
            ..ActorCriticConfig::default()
        };
        let mut opt =
            ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 0.0 }]).expect("build");

        let before = opt.temperature();
        // Current entropy (1.0) far exceeds the target (−2.0) ⇒ α must shrink.
        let loss = opt
            .update_temperature_sac(1.0, 2)
            .expect("temperature update");
        let after = opt.temperature();

        assert!(
            after < before,
            "α must decrease when entropy exceeds the target: {before} -> {after}"
        );
        // dJ/dα = H − H̄ = 3 ⇒ α ← 0.2 − 0.1·3 = −0.1, clamped to the floor.
        assert!(after > 0.0, "α must stay positive, got {after}");
        assert!((loss - 0.2 * 3.0).abs() < 1e-12, "J(α) = α(H − H̄)");
    }

    #[test]
    fn test_temperature_increases_when_entropy_below_target() {
        let cfg = ActorCriticConfig::<f64> {
            method: ActorCriticMethod::SAC,
            sac_config: SACConfig::<f64> {
                temperature: 0.2,
                auto_entropy_tuning: true,
                target_entropy: Some(1.0),
                temperature_lr: 0.05,
                ..SACConfig::default()
            },
            ..ActorCriticConfig::default()
        };
        let mut opt =
            ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 0.0 }]).expect("build");

        let before = opt.temperature();
        let _ = opt
            .update_temperature_sac(-1.0, 2)
            .expect("temperature update");
        let after = opt.temperature();

        assert!(
            after > before,
            "α must increase when entropy is below the target: {before} -> {after}"
        );
        assert!((after - (0.2 + 0.05 * 2.0)).abs() < 1e-12);
    }

    // ── Gaussian Box–Muller sampling: moments converge ───────────────────

    #[test]
    fn test_gaussian_sampling_standard_normal_moments() {
        let opt = opt_with(vec![MockQ { q: 0.0 }]);
        let n = 2000_usize;
        let dist = ActionDistribution {
            mean: Some(Array2::zeros((n, 1))),
            std: Some(Array2::from_elem((n, 1), 1.0_f64)),
            logits: None,
            distribution_type: DistributionType::Gaussian,
        };

        let samples = opt
            .sample_actions_from_distribution(&dist)
            .expect("sample_actions");

        let vals: Vec<f64> = samples.iter().copied().collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();

        assert!(mean.abs() < 0.12, "N(0,1) mean={mean} (expected ≈0)");
        assert!((std - 1.0).abs() < 0.12, "N(0,1) std={std} (expected ≈1)");
    }

    #[test]
    fn test_gaussian_log_prob_uses_correct_normalizer() {
        // log N(0; 0, 1) = −½ln(2π) ≈ −0.9189385. The old code used −ln(π).
        let opt = opt_with(vec![MockQ { q: 0.0 }]);
        let dist = ActionDistribution {
            mean: Some(Array2::zeros((1, 1))),
            std: Some(Array2::from_elem((1, 1), 1.0_f64)),
            logits: None,
            distribution_type: DistributionType::Gaussian,
        };
        let actions = Array2::zeros((1_usize, 1));
        let log_probs = opt
            .compute_log_probabilities(&dist, &actions)
            .expect("log probs");

        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (log_probs[0] - expected).abs() < 1e-12,
            "log N(0;0,1) = {} but got {}",
            expected,
            log_probs[0]
        );
    }

    #[test]
    fn test_gaussian_log_prob_survives_zero_sigma() {
        let opt = opt_with(vec![MockQ { q: 0.0 }]);
        let dist = ActionDistribution {
            mean: Some(Array2::zeros((1, 1))),
            std: Some(Array2::zeros((1, 1))),
            logits: None,
            distribution_type: DistributionType::Gaussian,
        };
        let actions = Array2::zeros((1_usize, 1));
        let log_probs = opt
            .compute_log_probabilities(&dist, &actions)
            .expect("log probs");
        assert!(
            log_probs[0].is_finite(),
            "σ = 0 must be clamped, got {}",
            log_probs[0]
        );
    }

    // ── Categorical inverse-CDF sampling ─────────────────────────────────

    #[test]
    fn test_categorical_biased_sampling() {
        let opt = opt_with(vec![MockQ { q: 0.0 }]);
        let n = 200_usize;
        let mut logits = Array2::zeros((n, 3_usize));
        for i in 0..n {
            logits[[i, 0]] = 10.0_f64;
        }
        let dist = ActionDistribution {
            mean: None,
            std: None,
            logits: Some(logits),
            distribution_type: DistributionType::Categorical,
        };

        let samples = opt
            .sample_actions_from_distribution(&dist)
            .expect("sample_actions");

        let class0_count = (0..n).filter(|&i| samples[[i, 0]] > 0.5).count();
        assert!(
            class0_count >= 185,
            "biased categorical: class-0 selected {class0_count}/200 (expected ≥185)"
        );
    }

    #[test]
    fn test_categorical_uniform_covers_all_classes() {
        let opt = opt_with(vec![MockQ { q: 0.0 }]);
        let n = 600_usize;
        let dist = ActionDistribution {
            mean: None,
            std: None,
            logits: Some(Array2::zeros((n, 3_usize))),
            distribution_type: DistributionType::Categorical,
        };

        let samples = opt
            .sample_actions_from_distribution(&dist)
            .expect("sample_actions");

        let mut counts = [0_usize; 3];
        for i in 0..n {
            for j in 0..3 {
                if samples[[i, j]] > 0.5 {
                    counts[j] += 1;
                }
            }
        }
        for (c, &cnt) in counts.iter().enumerate() {
            assert!(
                cnt >= 100,
                "uniform categorical class {c} appeared {cnt}/600 (expected ≥100)"
            );
        }
    }

    // ── F71 regression: prioritized replay ───────────────────────────────

    fn experience(value: f64, priority: f64) -> Experience<f64> {
        Experience {
            state: Array1::from_elem(1, value),
            action: Array1::from_elem(1, 0.0),
            reward: value,
            next_state: Array1::from_elem(1, value),
            done: false,
            priority,
            info: HashMap::new(),
        }
    }

    #[test]
    fn test_replay_buffer_rejects_zero_capacity() {
        assert!(
            ExperienceReplayBuffer::<f64>::new(0, 0.6, 0.4, false).is_err(),
            "a zero-capacity buffer must be rejected, not panic on `% 0`"
        );
    }

    #[test]
    fn test_replay_buffer_empty_sample_errors() {
        let buffer = ExperienceReplayBuffer::<f64>::new(8, 0.6, 0.4, true).expect("buffer");
        assert!(
            buffer.sample(4).is_err(),
            "sampling an empty buffer must error, not panic inside gen_range(0..0)"
        );
    }

    #[test]
    fn test_prioritized_sampling_favours_high_priority() {
        let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 0.4, true).expect("buffer");
        buffer.add(experience(0.0, 1.0));
        buffer.add(experience(1.0, 1.0));
        buffer.add(experience(2.0, 1.0));
        // Index 3 is 100x more likely than any other.
        buffer.add(experience(3.0, 100.0));

        let sample = buffer.sample(200).expect("sample");
        assert_eq!(
            sample.experiences.len(),
            4,
            "batch is capped at buffer size"
        );

        // Draw many batches and count how often index 3 appears.
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..100 {
            let s = buffer.sample(4).expect("sample");
            for &index in &s.indices {
                total += 1;
                if index == 3 {
                    hits += 1;
                }
            }
        }
        // p(3) = 100/103 ≈ 0.97; uniform would give 0.25.
        let ratio = hits as f64 / total as f64;
        assert!(
            ratio > 0.8,
            "high-priority transition drawn {ratio:.2} of the time (uniform would be 0.25)"
        );
    }

    #[test]
    fn test_prioritized_importance_weights_are_normalized() {
        let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 1.0, true).expect("buffer");
        buffer.add(experience(0.0, 1.0));
        buffer.add(experience(1.0, 4.0));

        let sample = buffer.sample(2).expect("sample");
        assert_eq!(sample.weights.len(), sample.indices.len());
        for &w in &sample.weights {
            assert!(w > 0.0 && w <= 1.0 + 1e-12, "IS weight out of range: {w}");
        }
        let max = sample.weights.iter().cloned().fold(0.0_f64, f64::max);
        assert!((max - 1.0).abs() < 1e-9, "weights must be max-normalized");
    }

    #[test]
    fn test_uniform_mode_returns_unit_weights() {
        let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 0.6, 0.4, false).expect("buffer");
        buffer.add(experience(0.0, 1.0));
        buffer.add(experience(1.0, 50.0));

        let sample = buffer.sample(2).expect("sample");
        for &w in &sample.weights {
            assert!(
                (w - 1.0).abs() < 1e-12,
                "uniform sampling needs no correction"
            );
        }
    }

    #[test]
    fn test_update_priorities_changes_sampling_mass() {
        let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 0.4, true).expect("buffer");
        buffer.add(experience(0.0, 1.0));
        buffer.add(experience(1.0, 1.0));

        let before = buffer.total_priority();
        buffer
            .update_priorities(&[0], &[9.0])
            .expect("priority update");
        let after = buffer.total_priority();

        assert!(
            after > before,
            "raising a TD error must raise the total priority mass: {before} -> {after}"
        );
        assert!(buffer.update_priorities(&[7], &[1.0]).is_err());
        assert!(buffer.update_priorities(&[0], &[1.0, 2.0]).is_err());
    }

    // ── F72 regression: Ornstein-Uhlenbeck exploration ───────────────────

    #[test]
    fn test_ou_noise_is_initialized_and_reverts_to_the_mean() {
        let mut opt = opt_with(vec![MockQ { q: 0.0 }]);
        // The state starts uninitialized; one step must create and advance it.
        let first = opt.update_ou_noise(2).expect("ou step");
        assert_eq!(first.len(), 2);
        assert!(
            first.iter().any(|&x| x != 0.0),
            "OU noise must actually move away from zero"
        );

        // With σ = 0 the process is pure mean reversion towards μ = 0.
        opt.config.ddpg_config.ou_noise_sigma = 0.0;
        opt.config.ddpg_config.ou_noise_theta = 0.5;
        opt.config.ddpg_config.ou_noise_dt = 1.0;
        let before: Vec<f64> = opt
            .update_ou_noise(2)
            .expect("ou step")
            .iter()
            .copied()
            .collect();
        let after: Vec<f64> = opt
            .update_ou_noise(2)
            .expect("ou step")
            .iter()
            .copied()
            .collect();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                a.abs() <= b.abs() + 1e-12,
                "mean reversion must shrink |x|: {b} -> {a}"
            );
        }
    }

    #[test]
    fn test_explore_actions_adds_noise_and_respects_bounds() {
        let mut opt = opt_with(vec![MockQ { q: 0.0 }]);
        opt.config.ddpg_config.ou_noise_sigma = 5.0;
        opt.config.ddpg_config.ou_noise_dt = 1.0;
        opt.config.ddpg_config.action_bounds = Some((-1.0, 1.0));

        let states = Array2::zeros((8_usize, 3));
        let actions = opt.explore_actions(&states).expect("explore");

        assert_eq!(actions.dim(), (8, 2));
        assert!(
            actions.iter().any(|&a| a != 0.0),
            "exploration noise must reach the returned actions"
        );
        for &a in actions.iter() {
            assert!((-1.0..=1.0).contains(&a), "action {a} out of bounds");
        }
    }

    // ── F15 regression: target networks exist and soft-update correctly ──

    #[test]
    fn test_target_networks_are_populated_when_enabled() {
        let cfg = ActorCriticConfig::<f64> {
            use_target_networks: true,
            ..ActorCriticConfig::default()
        };
        let opt = ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 1.0 }])
            .expect("construction");
        assert!(opt.target_actor.is_some(), "target actor must be created");
        assert!(
            opt.target_critics.is_some(),
            "target critics must be created"
        );
    }

    #[test]
    fn test_soft_update_moves_target_towards_online() {
        /// Critic whose single parameter is observable, to check Polyak averaging.
        #[derive(Clone)]
        struct ParamCritic {
            w: f64,
        }
        impl ValueNetwork<f64> for ParamCritic {
            fn evaluate_value(&self, obs: &Array2<f64>) -> Result<Array1<f64>> {
                Ok(Array1::from_elem(obs.nrows(), self.w))
            }
            fn update_parameters(&mut self, d: &HashMap<String, Array1<f64>>) -> Result<()> {
                if let Some(delta) = d.get("w") {
                    self.w += delta[0];
                }
                Ok(())
            }
            fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
                let mut m = HashMap::new();
                m.insert("w".to_string(), Array1::from_elem(1, self.w));
                m
            }
        }

        let cfg = ActorCriticConfig::<f64> {
            use_target_networks: true,
            target_update_rate: 0.25,
            ..ActorCriticConfig::default()
        };
        let mut opt = ActorCriticOptimizer::new(cfg, MockPolicy, vec![ParamCritic { w: 4.0 }])
            .expect("construction");

        // Move the online critic, then Polyak-average the target towards it.
        opt.critics[0].w = 8.0;
        opt.soft_update_targets().expect("soft update");

        let target_w = opt
            .target_critics
            .as_ref()
            .expect("targets")
            .first()
            .expect("critic")
            .w;
        // target ← 0.25·8 + 0.75·4 = 5.0 (NOT 4 + 5 = 9, which is what passing the
        // absolute target parameters into the additive API used to produce).
        assert!(
            (target_w - 5.0).abs() < 1e-12,
            "Polyak average should be 5.0, got {target_w}"
        );
    }
}
