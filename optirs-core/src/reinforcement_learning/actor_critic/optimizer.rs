use super::config::{ActorCriticConfig, ActorCriticMethod};
use super::metrics::ActorCriticMetrics;
use super::replay::{Experience, ExperienceReplayBuffer, ReplaySample};
use crate::error::{OptimError, Result};
use crate::reinforcement_learning::{
    add_named_gradients, clip_named_gradients, scale_named_gradients, ActionDistribution,
    DistributionType, PolicyNetwork, QNetwork, RLOptimizationMetrics, RLScheduler, TrajectoryBatch,
    ValueNetwork,
};
use scirs2_core::ndarray::{s, Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

/// Actor-Critic optimizer
pub struct ActorCriticOptimizer<
    T: Float + Debug + Send + Sync + 'static,
    P: PolicyNetwork<T>,
    V: ValueNetwork<T>,
> {
    /// Configuration
    pub(super) config: ActorCriticConfig<T>,

    /// Actor (policy) network
    actor: P,

    /// Critic (value) networks
    pub(super) critics: Vec<V>,

    /// Target networks (if enabled)
    pub(super) target_actor: Option<P>,
    pub(super) target_critics: Option<Vec<V>>,

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
    pub(super) fn update_ou_noise(&mut self, action_dim: usize) -> Result<Array1<T>> {
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
    pub(super) fn sample_actions_from_distribution(
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
    pub(super) fn compute_log_probabilities(
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
    pub(super) fn update_temperature_sac(
        &mut self,
        current_entropy: T,
        action_dim: usize,
    ) -> Result<T> {
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
    pub(super) fn compute_target_q_sac(
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
