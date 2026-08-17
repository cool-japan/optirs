// Policy Gradient Optimizers
//
// This module implements various policy gradient methods including REINFORCE,
// PPO (Proximal Policy Optimization), TRPO (Trust Region Policy Optimization),
// and other modern policy gradient algorithms.

use super::{
    clip_named_gradients, flatten_named, scale_named_gradients, PolicyNetwork,
    RLOptimizationMetrics, RLOptimizerConfig, RLScheduler, ScheduleType, TrajectoryBatch,
    TrustRegionConfig, TrustRegionMethod, TrustRegionOptimizer, ValueNetwork,
};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Policy gradient optimization methods
#[derive(Debug, Clone, Copy)]
pub enum PolicyGradientMethod {
    /// REINFORCE algorithm
    Reinforce,

    /// Actor-Critic
    ActorCritic,

    /// Proximal Policy Optimization (PPO) with clipped surrogate
    PPOClip,

    /// PPO with adaptive KL penalty
    PPOAdaptiveKL,

    /// Trust Region Policy Optimization (TRPO)
    TRPO,

    /// Importance Weighted Actor-Learner Architecture (IMPALA)
    IMPALA,

    /// Asynchronous Advantage Actor-Critic (A3C)
    A3C,
}

/// Policy gradient optimizer configuration
#[derive(Debug, Clone)]
pub struct PolicyGradientConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Base RL configuration
    pub base_config: RLOptimizerConfig<T>,

    /// Policy gradient method
    pub method: PolicyGradientMethod,

    /// PPO-specific parameters
    pub ppo_config: PPOConfig<T>,

    /// TRPO-specific parameters
    pub trpo_config: TRPOConfig<T>,

    /// Learning rate scheduler for policy
    pub policy_scheduler: Option<RLScheduler<T>>,

    /// Learning rate scheduler for value function
    pub value_scheduler: Option<RLScheduler<T>>,

    /// Use baseline (value function) for variance reduction
    pub use_baseline: bool,

    /// Enable importance sampling for off-policy updates
    pub importance_sampling: bool,

    /// Maximum importance sampling ratio
    pub max_is_ratio: T,
}

/// PPO-specific configuration
#[derive(Debug, Clone)]
pub struct PPOConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Clipping parameter
    pub clip_epsilon: T,

    /// Dual clipping (clip both positive and negative advantages)
    pub dual_clip: bool,

    /// Value function clipping
    pub value_clip: bool,

    /// Value clipping range
    pub value_clip_range: T,

    /// Target KL divergence for adaptive methods
    pub target_kl: T,

    /// KL coefficient for adaptive penalty
    pub kl_coeff: T,

    /// KL coefficient adaptation factor
    pub kl_coeff_adapt_factor: T,

    /// Early stopping based on KL divergence
    pub early_stop_on_kl: bool,
}

/// TRPO-specific configuration
#[derive(Debug, Clone)]
pub struct TRPOConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum KL divergence for trust region
    pub max_kl: T,

    /// Backtracking line search parameters
    pub backtrack_factor: T,
    pub max_backtracks: usize,

    /// Conjugate gradient parameters
    pub cg_iters: usize,
    pub cg_damping: T,
    pub cg_tolerance: T,

    /// Use natural gradients
    pub use_natural_gradients: bool,
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::numeric::FromPrimitive> Default
    for PolicyGradientConfig<T>
{
    fn default() -> Self {
        Self {
            base_config: RLOptimizerConfig::default(),
            method: PolicyGradientMethod::PPOClip,
            ppo_config: PPOConfig::default(),
            trpo_config: TRPOConfig::default(),
            policy_scheduler: Some(RLScheduler::new(
                T::from(3e-4).unwrap_or_else(|| T::zero()),
                ScheduleType::Constant,
            )),
            value_scheduler: Some(RLScheduler::new(
                T::from(1e-3).unwrap_or_else(|| T::zero()),
                ScheduleType::Constant,
            )),
            use_baseline: true,
            importance_sampling: false,
            max_is_ratio: T::from(2.0).unwrap_or_else(|| T::zero()),
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::numeric::FromPrimitive> Default
    for PPOConfig<T>
{
    fn default() -> Self {
        Self {
            clip_epsilon: T::from(0.2).unwrap_or_else(|| T::zero()),
            dual_clip: false,
            value_clip: true,
            value_clip_range: T::from(0.2).unwrap_or_else(|| T::zero()),
            target_kl: T::from(0.01).unwrap_or_else(|| T::zero()),
            kl_coeff: T::from(0.2).unwrap_or_else(|| T::zero()),
            kl_coeff_adapt_factor: T::from(1.5).unwrap_or_else(|| T::zero()),
            early_stop_on_kl: true,
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::numeric::FromPrimitive> Default
    for TRPOConfig<T>
{
    fn default() -> Self {
        Self {
            max_kl: T::from(0.01).unwrap_or_else(|| T::zero()),
            backtrack_factor: T::from(0.5).unwrap_or_else(|| T::zero()),
            max_backtracks: 10,
            cg_iters: 10,
            cg_damping: T::from(0.1).unwrap_or_else(|| T::zero()),
            cg_tolerance: T::from(1e-8).unwrap_or_else(|| T::zero()),
            use_natural_gradients: true,
        }
    }
}

/// Policy gradient optimizer
pub struct PolicyGradientOptimizer<
    T: Float + Debug + Send + Sync + 'static,
    P: PolicyNetwork<T>,
    V: ValueNetwork<T>,
> {
    /// Configuration
    config: PolicyGradientConfig<T>,

    /// Policy network
    policy_network: P,

    /// Value network
    value_network: Option<V>,

    /// Learning rate schedulers
    policy_scheduler: Option<RLScheduler<T>>,
    value_scheduler: Option<RLScheduler<T>>,

    /// Optimization statistics
    metrics: RLOptimizationMetrics<T>,

    /// Update counter
    update_count: usize,

    /// KL coefficient for adaptive PPO
    kl_coeff: T,

    /// Trajectory buffer for batch updates
    trajectory_buffer: Vec<TrajectoryBatch<T>>,

    /// Maximum buffer size
    max_buffer_size: usize,
}

impl<
        T: Float
            + Debug
            + Send
            + Sync
            + 'static
            + ScalarOperand
            + std::ops::AddAssign
            + std::iter::Sum
            + scirs2_core::numeric::FromPrimitive,
        P: PolicyNetwork<T>,
        V: ValueNetwork<T>,
    > PolicyGradientOptimizer<T, P, V>
{
    /// Create a new policy gradient optimizer
    pub fn new(
        config: PolicyGradientConfig<T>,
        policy_network: P,
        value_network: Option<V>,
    ) -> Self {
        let kl_coeff = config.ppo_config.kl_coeff;
        let policy_scheduler = config.policy_scheduler.clone();
        let value_scheduler = config.value_scheduler.clone();

        Self {
            config,
            policy_network,
            value_network,
            policy_scheduler,
            value_scheduler,
            metrics: RLOptimizationMetrics::default(),
            update_count: 0,
            kl_coeff,
            trajectory_buffer: Vec::new(),
            max_buffer_size: 1000,
        }
    }

    /// Update policy using trajectory data
    pub fn update(&mut self, trajectory: TrajectoryBatch<T>) -> Result<RLOptimizationMetrics<T>> {
        match self.config.method {
            PolicyGradientMethod::PPOClip => self.update_ppo_clip(trajectory),
            PolicyGradientMethod::PPOAdaptiveKL => self.update_ppo_adaptive_kl(trajectory),
            PolicyGradientMethod::TRPO => self.update_trpo(trajectory),
            PolicyGradientMethod::Reinforce => self.update_reinforce(trajectory),
            PolicyGradientMethod::ActorCritic => self.update_actor_critic(trajectory),
            // A3C is *asynchronous* A2C: multiple workers each compute an A2C
            // gradient against a shared set of parameters. The asynchrony /
            // worker-coordination is an orchestration concern handled outside
            // this optimizer (by whoever drives the per-worker `update` calls);
            // the per-update math performed here is identical to synchronous A2C.
            PolicyGradientMethod::A3C => self.update_actor_critic(trajectory),
            // IMPALA is off-policy: it corrects for the lag between the behavior
            // policy that generated the trajectory and the current learner policy
            // using V-trace truncated importance sampling.
            PolicyGradientMethod::IMPALA => self.update_impala(trajectory),
        }
    }

    /// Bootstrap value `V(s_T)` for the step *after* the final transition.
    ///
    /// The successor state is [`TrajectoryBatch::final_observation`], **not** the
    /// last row of `observations` (which is `s_{T-1}`, the state the final action
    /// was taken from). Bootstrapping on `s_{T-1}` double-counts the last
    /// transition and biases every advantage in the batch.
    ///
    /// Returns zero when there is no value network or no recorded successor state
    /// (the batch simply gets no bootstrap, which is the correct behaviour for a
    /// trajectory that ends on termination).
    fn bootstrap_next_value(&self, trajectory: &TrajectoryBatch<T>) -> Result<T> {
        let (Some(value_net), Some(final_obs)) = (
            self.value_network.as_ref(),
            trajectory.final_observation.as_ref(),
        ) else {
            return Ok(T::zero());
        };

        let mut batch = Array2::zeros((1, final_obs.len()));
        batch.row_mut(0).assign(final_obs);
        Ok(value_net.evaluate_value(&batch)?[0])
    }

    /// Current policy learning rate (scheduler value when configured).
    fn policy_lr(&self) -> T {
        self.policy_scheduler
            .as_ref()
            .map(|s| s.get_lr())
            .unwrap_or(self.config.base_config.policy_lr)
    }

    /// Current value-function learning rate (scheduler value when configured).
    fn value_lr(&self) -> T {
        self.value_scheduler
            .as_ref()
            .map(|s| s.get_lr())
            .unwrap_or(self.config.base_config.value_lr)
    }

    /// Fill `trajectory.advantages` / `trajectory.returns` with normalized GAE
    /// estimates, bootstrapping the final step with the current value network.
    /// This is the exact advantage computation shared by PPO (clipped &
    /// adaptive-KL) and A2C so the on-policy variants stay consistent.
    fn prepare_gae(&self, trajectory: &mut TrajectoryBatch<T>) -> Result<()> {
        let next_value = self.bootstrap_next_value(trajectory)?;
        trajectory.compute_advantages(
            self.config.base_config.discount_factor,
            self.config.base_config.gae_lambda,
            next_value,
        )
    }

    /// PPO with clipped surrogate objective.
    ///
    /// Per mini-batch sample the objective is `min(r·A, clip(r, 1±ε)·A)` with
    /// `r = exp(log π − log π_old)`. Its derivative w.r.t. `log π` is `r·A` when
    /// the unclipped branch wins, and zero when the clipped branch wins *and* the
    /// ratio has left the clip interval (where `clip` is flat) — this is exactly
    /// the "no gradient once you have moved too far" behaviour PPO relies on, and
    /// it is what gets fed to the policy's score oracle.
    fn update_ppo_clip(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let mut total_policy_loss = T::zero();
        let mut total_value_loss = T::zero();
        let mut total_entropy_loss = T::zero();
        let mut clip_fraction = T::zero();
        let mut approx_kl = T::zero();
        // Real count of applied mini-batch updates — early stopping means this is
        // NOT `n_epochs · ceil(N / batch)`.
        let mut n_updates = 0usize;

        // Compute advantages using GAE (with value-network bootstrap).
        self.prepare_gae(&mut trajectory)?;

        let n_epochs = self.config.base_config.n_epochs;
        let mini_batch_size = self.config.base_config.mini_batchsize;
        if mini_batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "mini_batchsize must be greater than zero".to_string(),
            ));
        }

        let clip_eps = self.config.ppo_config.clip_epsilon;
        let target_kl = self.config.ppo_config.target_kl;
        let two = T::one() + T::one();
        let half = T::one() / two;

        'epochs: for _epoch in 0..n_epochs {
            for mini_batch in trajectory.get_mini_batches(mini_batch_size) {
                let batch_len = mini_batch.observations.nrows();
                if batch_len == 0 {
                    continue;
                }
                let count = T::from(batch_len).ok_or_else(|| {
                    OptimError::ComputationError(
                        "failed to convert mini-batch size to scalar".to_string(),
                    )
                })?;
                let inv_n = T::one() / count;

                let policy_eval = self
                    .policy_network
                    .evaluate_actions(&mini_batch.observations, &mini_batch.actions)?;

                let log_ratio = &policy_eval.log_probs - &mini_batch.log_probs;
                let ratio = log_ratio.mapv(|x| x.exp());

                let mut policy_loss = T::zero();
                let mut dloss_dlogp = Array1::zeros(batch_len);
                let mut n_clipped = 0usize;

                for i in 0..batch_len {
                    let r = ratio[i];
                    let advantage = mini_batch.advantages[i];
                    let clipped_r = r.max(T::one() - clip_eps).min(T::one() + clip_eps);

                    let surr1 = r * advantage;
                    let surr2 = clipped_r * advantage;

                    let (objective, dobjective) = if surr1 <= surr2 {
                        // Unclipped branch: d(r·A)/d log π = r·A.
                        (surr1, r * advantage)
                    } else {
                        // Clipped branch: the gradient survives only while the
                        // ratio is strictly inside the clip interval.
                        let inside = r > T::one() - clip_eps && r < T::one() + clip_eps;
                        (surr2, if inside { r * advantage } else { T::zero() })
                    };

                    policy_loss = policy_loss - objective * inv_n;
                    dloss_dlogp[i] = -dobjective * inv_n;

                    if r < T::one() - clip_eps || r > T::one() + clip_eps {
                        n_clipped += 1;
                    }
                }

                let entropy_loss = Self::entropy_loss(&policy_eval);

                // Real gradient steps: policy first (so the value update sees the
                // same parameters PPO's reference implementations do), then value.
                self.apply_policy_gradient_step(
                    &mini_batch.observations,
                    &mini_batch.actions,
                    &dloss_dlogp,
                )?;
                let value_loss = self.update_value_on_batch(
                    &mini_batch.observations,
                    &mini_batch.values,
                    &mini_batch.returns,
                )?;

                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy_loss += entropy_loss;
                n_updates += 1;

                clip_fraction = clip_fraction + T::from(n_clipped).unwrap_or_else(T::zero) * inv_n;

                // Standard second-order KL estimator: ½·E[(log ratio)²].
                let batch_kl = half * log_ratio.mapv(|x| x * x).mean().unwrap_or(T::zero());
                approx_kl += batch_kl;

                // Early stopping compares the *current mini-batch* KL against the
                // threshold (an accumulated sum would trip on batch count alone)
                // and abandons the whole update, not just the inner loop.
                if self.config.ppo_config.early_stop_on_kl && batch_kl > target_kl * two {
                    break 'epochs;
                }
            }
        }

        // Update learning rates
        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }

        self.update_count += 1;

        let divisor = T::from(n_updates.max(1)).unwrap_or_else(T::one);
        self.metrics.policy_loss = total_policy_loss / divisor;
        self.metrics.value_loss = total_value_loss / divisor;
        self.metrics.entropy_loss = total_entropy_loss / divisor;
        self.metrics.total_loss = self.metrics.policy_loss
            + self.config.base_config.value_loss_coeff * self.metrics.value_loss
            + self.config.base_config.entropy_coeff * self.metrics.entropy_loss;
        self.metrics.clip_fraction = Some(clip_fraction / divisor);
        self.metrics.kl_divergence = Some(approx_kl / divisor);

        Ok(self.metrics.clone())
    }

    /// PPO with adaptive KL penalty.
    ///
    /// Mirrors [`Self::update_ppo_clip`]'s mini-batch structure but replaces the
    /// clipped surrogate with a KL-penalty surrogate:
    ///
    /// ```text
    /// policy_loss = -E[ ratio · advantage ] + β · KL(old‖new)
    /// ```
    ///
    /// where `ratio = exp(new_logp − old_logp)` and the per-sample KL of the old
    /// policy relative to the new is estimated as `old_logp − new_logp` (the
    /// standard first-order estimator). After each epoch the penalty coefficient
    /// `β` (stored on `self.kl_coeff`) is adapted against the configured target
    /// KL using the canonical PPO rule:
    ///
    /// * `KL > 1.5 · target_kl`  ⇒ `β ← 2β`   (penalty too weak)
    /// * `KL < target_kl / 1.5`  ⇒ `β ← β / 2` (penalty too strong)
    ///
    /// `β` is clamped to a sane range so it neither vanishes nor explodes.
    fn update_ppo_adaptive_kl(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let mut total_policy_loss = T::zero();
        let mut total_value_loss = T::zero();
        let mut total_entropy_loss = T::zero();
        let mut approx_kl = T::zero();

        // Target KL for the adaptation rule (reuse the PPO config field).
        let target_kl = self.config.ppo_config.target_kl;
        let one_point_five = T::from(1.5).unwrap_or_else(|| T::one());
        let two = T::from(2.0).unwrap_or_else(|| T::one() + T::one());
        // Clamp range for β to keep the penalty well-conditioned across updates.
        let beta_min = T::from(1e-4).unwrap_or_else(|| T::zero());
        let beta_max = T::from(1e4).unwrap_or_else(|| T::one());

        // Compute advantages using GAE (identical to the clipped variant).
        self.prepare_gae(&mut trajectory)?;

        let n_epochs = self.config.base_config.n_epochs;
        let mini_batch_size = self.config.base_config.mini_batchsize;
        if mini_batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "mini_batchsize must be greater than zero".to_string(),
            ));
        }

        // The KL measured on the final epoch drives the β adaptation.
        let mut last_epoch_kl = T::zero();
        let mut n_updates = 0usize;

        'epochs: for _epoch in 0..n_epochs {
            let mini_batches = trajectory.get_mini_batches(mini_batch_size);
            let mut epoch_kl_sum = T::zero();
            let mut epoch_batches = T::zero();

            for mini_batch in mini_batches {
                let batch_len = mini_batch.observations.nrows();
                if batch_len == 0 {
                    continue;
                }
                let batch_count = T::from(batch_len).ok_or_else(|| {
                    OptimError::ComputationError(
                        "failed to convert mini-batch size to scalar".to_string(),
                    )
                })?;
                let inv_n = T::one() / batch_count;

                // Current policy evaluation.
                let policy_eval = self
                    .policy_network
                    .evaluate_actions(&mini_batch.observations, &mini_batch.actions)?;

                // Importance sampling ratio = exp(new_logp − old_logp).
                let log_ratio = &policy_eval.log_probs - &mini_batch.log_probs;
                let ratio = log_ratio.mapv(|x| x.exp());

                // Surrogate (un-clipped): E[ ratio · advantage ].
                let surrogate =
                    (&ratio * &mini_batch.advantages).iter().copied().sum::<T>() / batch_count;

                // Per-sample KL(old‖new) estimate = old_logp − new_logp = −log_ratio.
                // Mean over the mini-batch, guarded to be non-negative.
                let mut kl_sum = T::zero();
                for &lr in log_ratio.iter() {
                    kl_sum = kl_sum - lr;
                }
                let batch_kl = (kl_sum / batch_count).max(T::zero());

                // KL-penalty surrogate policy loss.
                let policy_loss = -surrogate + self.kl_coeff * batch_kl;

                // ∂L/∂ log πᵢ = (−rᵢ·Aᵢ − β)/N: the ratio term differentiates to
                // r·A, the KL estimator (−log ratio) contributes −β.
                let mut dloss_dlogp = Array1::zeros(batch_len);
                for i in 0..batch_len {
                    dloss_dlogp[i] = (-ratio[i] * mini_batch.advantages[i] - self.kl_coeff) * inv_n;
                }

                let entropy_loss = Self::entropy_loss(&policy_eval);

                self.apply_policy_gradient_step(
                    &mini_batch.observations,
                    &mini_batch.actions,
                    &dloss_dlogp,
                )?;
                let value_loss = self.update_value_on_batch(
                    &mini_batch.observations,
                    &mini_batch.values,
                    &mini_batch.returns,
                )?;

                // Accumulate metrics.
                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy_loss += entropy_loss;
                n_updates += 1;

                approx_kl += batch_kl;
                epoch_kl_sum += batch_kl;
                epoch_batches += T::one();

                // Early stopping based on KL divergence: abandon the whole update.
                if self.config.ppo_config.early_stop_on_kl
                    && batch_kl > self.config.ppo_config.target_kl * two
                {
                    last_epoch_kl = epoch_kl_sum / epoch_batches;
                    break 'epochs;
                }
            }

            // Mean KL over the epoch's mini-batches (used to adapt β next).
            if epoch_batches > T::zero() {
                last_epoch_kl = epoch_kl_sum / epoch_batches;
            }
        }

        // Adapt β against the target KL using the canonical PPO rule.
        if last_epoch_kl > one_point_five * target_kl {
            self.kl_coeff = self.kl_coeff * two;
        } else if last_epoch_kl < target_kl / one_point_five {
            self.kl_coeff = self.kl_coeff / two;
        }
        // Clamp β into a sane range.
        if self.kl_coeff < beta_min {
            self.kl_coeff = beta_min;
        } else if self.kl_coeff > beta_max {
            self.kl_coeff = beta_max;
        }

        // Update learning rates.
        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }

        self.update_count += 1;

        // Update metrics (averaged over the mini-batch updates actually applied).
        let divisor = T::from(n_updates.max(1)).unwrap_or_else(T::one);
        self.metrics.policy_loss = total_policy_loss / divisor;
        self.metrics.value_loss = total_value_loss / divisor;
        self.metrics.entropy_loss = total_entropy_loss / divisor;
        self.metrics.total_loss = self.metrics.policy_loss
            + self.config.base_config.value_loss_coeff * self.metrics.value_loss
            + self.config.base_config.entropy_coeff * self.metrics.entropy_loss;
        // No clipping in this variant.
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = Some(approx_kl / divisor);
        // Surface the current penalty coefficient for inspection.
        self.metrics
            .custom_metrics
            .insert("kl_coeff".to_string(), self.kl_coeff);

        Ok(self.metrics.clone())
    }

    /// TRPO update: a genuine trust-region step, not a PPO alias.
    ///
    /// The surrogate gradient `g = ∇_θ E[log π(a|s)·A(s,a)]` and the per-sample
    /// score matrix are produced by the policy's analytic oracle, then handed to
    /// [`TrustRegionOptimizer`] configured from this optimizer's [`TRPOConfig`].
    /// The trust-region optimizer borrows the policy (via the `&mut P` forwarding
    /// impl), solves `F x = g` with damped conjugate gradients, takes the
    /// `β = √(2δ / sᵀFs)` step and backtracks against the **real**
    /// importance-weighted surrogate evaluated on this trajectory.
    ///
    /// The value function is fitted by regression on the GAE returns, as in the
    /// original TRPO.
    fn update_trpo(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        self.prepare_gae(&mut trajectory)?;

        let batch_len = trajectory.observations.nrows();
        if batch_len == 0 {
            return Err(OptimError::InvalidConfig(
                "TRPO received an empty trajectory".to_string(),
            ));
        }
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;

        // Fit the value baseline first (it does not participate in the trust region).
        let value_loss = self.update_value_on_batch(
            &trajectory.observations,
            &trajectory.values,
            &trajectory.returns,
        )?;

        // Ascent direction of the surrogate: ∇_θ (1/N) Σ A_i log π_i.
        let coefficients = trajectory.advantages.mapv(|a| a * inv_n);
        let gradient_map = self.policy_network.log_prob_gradient(
            &trajectory.observations,
            &trajectory.actions,
            &coefficients,
        )?;
        let flat_gradient = flatten_named(&gradient_map);
        let grad_norm = flat_gradient.iter().map(|&g| g * g).sum::<T>().sqrt();

        // Empirical Fisher from per-sample scores.
        let scores = self
            .policy_network
            .score_matrix(&trajectory.observations, &trajectory.actions)?;

        // Log-probabilities of the behaviour policy, for the surrogate ratio.
        let old_log_probs = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?
            .log_probs;

        let trpo = self.config.trpo_config.clone();
        let tr_config = TrustRegionConfig {
            method: TrustRegionMethod::TRPO,
            max_kl: trpo.max_kl,
            cg_iters: trpo.cg_iters,
            cg_damping: trpo.cg_damping,
            cg_tolerance: trpo.cg_tolerance,
            max_backtracks: trpo.max_backtracks,
            backtrack_coeff: trpo.backtrack_factor,
            ..TrustRegionConfig::default()
        };

        // Owned copies so the surrogate closure never borrows `self`.
        let observations = trajectory.observations.clone();
        let actions = trajectory.actions.clone();
        let advantages = trajectory.advantages.clone();

        let tr_metrics = {
            let mut trust_region = TrustRegionOptimizer::new(tr_config, &mut self.policy_network);
            trust_region.set_score_samples(scores);
            trust_region.update_trpo_with_surrogate(&flat_gradient, |policy| {
                let evaluation = policy.evaluate_actions(&observations, &actions)?;
                let mut surrogate = T::zero();
                for i in 0..batch_len {
                    let ratio = (evaluation.log_probs[i] - old_log_probs[i]).exp();
                    surrogate = surrogate + ratio * advantages[i];
                }
                Ok(surrogate * inv_n)
            })?
        };

        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }
        self.update_count += 1;

        self.metrics.policy_loss = tr_metrics.policy_loss;
        self.metrics.value_loss = value_loss;
        self.metrics.entropy_loss = T::zero();
        self.metrics.total_loss =
            self.metrics.policy_loss + self.config.base_config.value_loss_coeff * value_loss;
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = tr_metrics.kl_divergence;
        self.metrics.policy_grad_norm = grad_norm;
        for (name, value) in tr_metrics.custom_metrics {
            self.metrics.custom_metrics.insert(name, value);
        }

        Ok(self.metrics.clone())
    }

    /// REINFORCE (Monte-Carlo policy gradient).
    ///
    /// The advantages/returns the loss needs are **computed here** rather than
    /// read out of a freshly zeroed [`TrajectoryBatch`] (which made the loss
    /// identically zero, and therefore the update a no-op):
    ///
    /// * with a value baseline → GAE advantages (plus a value regression step),
    /// * without one → plain discounted Monte-Carlo returns `G_t`.
    fn update_reinforce(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let batch_len = trajectory.observations.nrows();
        if batch_len == 0 {
            return Err(OptimError::InvalidConfig(
                "REINFORCE received an empty trajectory".to_string(),
            ));
        }
        let count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;

        let use_baseline = self.config.use_baseline && self.value_network.is_some();
        let weights = if use_baseline {
            self.prepare_gae(&mut trajectory)?;
            trajectory.advantages.clone()
        } else {
            let next_value = self.bootstrap_next_value(&trajectory)?;
            trajectory
                .compute_discounted_returns(self.config.base_config.discount_factor, next_value)?;
            trajectory.returns.clone()
        };

        let policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        // L = −(1/N) Σ log π_i · w_i  ⇒  ∂L/∂ log π_i = −w_i / N.
        let mut policy_loss = T::zero();
        let mut dloss_dlogp = Array1::zeros(batch_len);
        for i in 0..batch_len {
            policy_loss = policy_loss - policy_eval.log_probs[i] * weights[i] * inv_n;
            dloss_dlogp[i] = -weights[i] * inv_n;
        }

        let entropy_loss = Self::entropy_loss(&policy_eval);

        self.apply_policy_gradient_step(
            &trajectory.observations,
            &trajectory.actions,
            &dloss_dlogp,
        )?;

        let value_loss = if use_baseline {
            self.update_value_on_batch(
                &trajectory.observations,
                &trajectory.values,
                &trajectory.returns,
            )?
        } else {
            T::zero()
        };

        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }
        self.update_count += 1;

        self.metrics.policy_loss = policy_loss;
        self.metrics.value_loss = value_loss;
        self.metrics.entropy_loss = entropy_loss;
        self.metrics.total_loss = policy_loss
            + self.config.base_config.value_loss_coeff * value_loss
            + self.config.base_config.entropy_coeff * entropy_loss;
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = Some(T::zero());

        Ok(self.metrics.clone())
    }

    /// Synchronous Advantage Actor-Critic (A2C) update.
    ///
    /// A2C is the on-policy special case of the actor-critic family: the
    /// trajectory was generated by the *current* policy, so the importance
    /// ratio is identically 1 and there is neither clipping nor multiple
    /// epochs (contrast with [`Self::update_ppo_clip`]). A single pass over the
    /// batch is performed:
    ///
    /// ```text
    /// policy_loss = -E[ log π(a|s) · A(s,a) ] - entropy_coeff · entropy
    /// value_loss  = value_loss_coeff · MSE(V(s), returns)
    /// ```
    ///
    /// where `A(s,a)` are the GAE advantages (normalized, exactly as PPO
    /// computes them via [`Self::prepare_gae`]) and `returns` are the
    /// (un-normalized) GAE returns. The advantage and value-clipping treatment
    /// mirror [`Self::update_ppo_clip`] so the on-policy variants stay
    /// consistent; the value-clip toggle is honoured identically.
    fn update_actor_critic(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        // GAE advantages + returns (normalized advantages, value bootstrap) —
        // shared with PPO so A2C and PPO agree on the advantage definition.
        self.prepare_gae(&mut trajectory)?;

        // Single on-policy pass: evaluate the current policy on the whole batch.
        let policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        let batch_len = trajectory.observations.nrows();
        if batch_len == 0 {
            return Err(OptimError::InvalidConfig(
                "A2C received an empty trajectory".to_string(),
            ));
        }
        let batch_count = T::from(batch_len).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / batch_count;

        // Policy loss = -E[ log π(a|s) · A(s,a) ]. The importance ratio is 1
        // (on-policy), so this is the plain advantage-weighted log-likelihood, and
        // ∂L/∂ log π_i = −A_i / N.
        let mut policy_loss = T::zero();
        let mut dloss_dlogp = Array1::zeros(batch_len);
        for i in 0..batch_len {
            policy_loss = policy_loss - policy_eval.log_probs[i] * trajectory.advantages[i] * inv_n;
            dloss_dlogp[i] = -trajectory.advantages[i] * inv_n;
        }

        // Entropy loss (negative to encourage exploration), same convention as
        // the PPO paths so `entropy_coeff` behaves identically.
        let entropy_loss = Self::entropy_loss(&policy_eval);

        // Apply the real policy gradient, then fit the value function against the
        // GAE returns (identical clipped / unclipped MSE treatment as PPO).
        self.apply_policy_gradient_step(
            &trajectory.observations,
            &trajectory.actions,
            &dloss_dlogp,
        )?;
        let value_loss = self.update_value_on_batch(
            &trajectory.observations,
            &trajectory.values,
            &trajectory.returns,
        )?;

        // Total loss combines policy, value and entropy contributions exactly
        // as the PPO variants do.
        let total_loss = policy_loss
            + self.config.base_config.value_loss_coeff * value_loss
            + self.config.base_config.entropy_coeff * entropy_loss;

        // Step learning-rate schedulers (parity with PPO).
        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }

        self.update_count += 1;

        // Populate metrics. A2C performs no clipping and (by construction) has a
        // unit importance ratio, so there is no clip fraction and the policy-KL
        // is zero relative to the data-generating policy.
        self.metrics.policy_loss = policy_loss;
        self.metrics.value_loss = value_loss;
        self.metrics.entropy_loss = entropy_loss;
        self.metrics.total_loss = total_loss;
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = Some(T::zero());

        Ok(self.metrics.clone())
    }

    /// IMPALA update via V-trace off-policy correction.
    ///
    /// Unlike A2C/PPO, IMPALA is *off-policy*: the trajectory was produced by a
    /// (possibly lagged) behavior policy `μ` whose log-probabilities are stored
    /// in `trajectory.log_probs`, while gradients are taken w.r.t. the current
    /// learner policy `π` (from [`PolicyNetwork::evaluate_actions`]). The lag is
    /// corrected with truncated importance weights:
    ///
    /// ```text
    /// is_t = exp(log π(a_t|s_t) − log μ(a_t|s_t))
    /// ρ_t  = min(ρ̄, is_t)          c_t = min(c̄, is_t)
    /// δ_t  = ρ_t (r_t + γ V(s_{t+1}) − V(s_t))
    /// v_t  = V(s_t) + δ_t + γ c_t (v_{t+1} − V(s_{t+1}))      (recursion, t = T-1 … 0)
    /// ```
    ///
    /// The V-trace targets `v_t` are the value regression targets, and the
    /// policy-gradient advantage uses the bootstrapped next target:
    /// `A_t = ρ_t (r_t + γ v_{t+1} − V(s_t))`. Following the IMPALA paper the
    /// truncation thresholds satisfy `c̄ ≤ ρ̄`; `ρ̄` is taken from
    /// `config.max_is_ratio` (canonical default 1.0–2.0) and `c̄ = min(1, ρ̄)`.
    ///
    /// `V(s_t)` is evaluated with the *current* learner value network (not the
    /// behavior-time `trajectory.values`), and the final step is bootstrapped
    /// with [`Self::bootstrap_next_value`]. When no value network is configured
    /// V-trace degenerates to truncated-importance-weighted REINFORCE.
    fn update_impala(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let batch_size = trajectory.observations.nrows();
        if batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "IMPALA received an empty trajectory".to_string(),
            ));
        }

        let gamma = self.config.base_config.discount_factor;

        // Truncation thresholds. ρ̄ from config (clamped to ≥ 1 so the
        // correction never *down*-weights an unlagged sample); c̄ ≤ ρ̄.
        let rho_bar = self.config.max_is_ratio.max(T::one());
        let c_bar = rho_bar.min(T::one());

        // Current learner value estimates V(s_t) and the bootstrap V(s_T).
        let values_now = if let Some(ref value_net) = self.value_network {
            value_net.evaluate_value(&trajectory.observations)?
        } else {
            Array1::zeros(batch_size)
        };
        let bootstrap_value = self.bootstrap_next_value(&trajectory)?;

        // Current learner policy log-probs log π(a_t|s_t) and entropy.
        let policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        // Per-step truncated importance weights ρ_t and c_t.
        let mut rho = Array1::zeros(batch_size);
        let mut c_trace = Array1::zeros(batch_size);
        for t in 0..batch_size {
            let is_ratio = (policy_eval.log_probs[t] - trajectory.log_probs[t]).exp();
            rho[t] = is_ratio.min(rho_bar);
            c_trace[t] = is_ratio.min(c_bar);
        }

        // V-trace targets v_t via the backward recursion, plus the per-step
        // policy-gradient advantage A_t = ρ_t (r_t + γ v_{t+1} − V(s_t)).
        let mut vtrace_targets = Array1::zeros(batch_size);
        let mut pg_advantages = Array1::zeros(batch_size);
        // v_{t+1} for the last step is the bootstrap value V(s_T).
        let mut next_vtrace = bootstrap_value;
        for t in (0..batch_size).rev() {
            let is_terminal = trajectory.dones[t];
            let nonterminal = T::from(!is_terminal as u8).unwrap_or_else(T::zero);

            // V(s_{t+1}): bootstrap for the final step, otherwise the learner's
            // value at t+1. Masked to zero on episode termination.
            let next_value = if t == batch_size - 1 {
                bootstrap_value
            } else {
                values_now[t + 1]
            } * nonterminal;

            // δ_t^V = ρ_t (r_t + γ V(s_{t+1}) − V(s_t)).
            let delta = rho[t] * (trajectory.rewards[t] + gamma * next_value - values_now[t]);

            // v_t = V(s_t) + δ_t + γ c_t (v_{t+1} − V(s_{t+1})).
            let masked_next_vtrace = next_vtrace * nonterminal;
            let vtrace =
                values_now[t] + delta + gamma * c_trace[t] * (masked_next_vtrace - next_value);
            vtrace_targets[t] = vtrace;

            // Policy-gradient advantage uses the *bootstrapped* next target.
            pg_advantages[t] =
                rho[t] * (trajectory.rewards[t] + gamma * masked_next_vtrace - values_now[t]);

            next_vtrace = vtrace;
        }

        let batch_count = T::from(batch_size).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / batch_count;

        // Policy loss = -E[ log π(a_t|s_t) · A_t ] (ρ is already folded into A_t),
        // so ∂L/∂ log π_t = −A_t / N.
        let mut policy_loss = T::zero();
        let mut dloss_dlogp = Array1::zeros(batch_size);
        for t in 0..batch_size {
            policy_loss = policy_loss - policy_eval.log_probs[t] * pg_advantages[t] * inv_n;
            dloss_dlogp[t] = -pg_advantages[t] * inv_n;
        }

        // Entropy loss (same convention as the other update rules).
        let entropy_loss = Self::entropy_loss(&policy_eval);

        // Value loss = MSE(V(s_t), v_t) against the V-trace targets, which are
        // treated as constants (standard V-trace: the recursion is not
        // differentiated through).
        let two = T::one() + T::one();
        let mut value_loss = T::zero();
        let mut dloss_dv = Array1::zeros(batch_size);
        if self.value_network.is_some() {
            for t in 0..batch_size {
                let err = values_now[t] - vtrace_targets[t];
                value_loss = value_loss + err * err * inv_n;
                dloss_dv[t] = two * err * inv_n * self.config.base_config.value_loss_coeff;
            }
        }

        // Total loss.
        let total_loss = policy_loss
            + self.config.base_config.value_loss_coeff * value_loss
            + self.config.base_config.entropy_coeff * entropy_loss;

        // Apply real policy & value gradient steps.
        self.apply_policy_gradient_step(
            &trajectory.observations,
            &trajectory.actions,
            &dloss_dlogp,
        )?;
        if self.value_network.is_some() {
            self.apply_value_gradient_step(&trajectory.observations, &dloss_dv)?;
        }

        // Step learning-rate schedulers (parity with the other update rules).
        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }

        self.update_count += 1;

        // Populate metrics. The mean truncated importance weight is surfaced as
        // a custom metric for off-policy diagnostics; approx-KL between μ and π
        // is the mean log-ratio magnitude.
        let mean_rho = rho.iter().copied().sum::<T>() / batch_count;
        let approx_kl = (&policy_eval.log_probs - &trajectory.log_probs)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(T::zero());

        self.metrics.policy_loss = policy_loss;
        self.metrics.value_loss = value_loss;
        self.metrics.entropy_loss = entropy_loss;
        self.metrics.total_loss = total_loss;
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = Some(approx_kl);
        self.metrics
            .custom_metrics
            .insert("mean_rho".to_string(), mean_rho);

        Ok(self.metrics.clone())
    }

    /// Apply one gradient-**descent** step to the policy network.
    ///
    /// `dloss_dlogp[i] = ∂L/∂ log π(aᵢ|sᵢ)` — the only thing that differs between
    /// REINFORCE, A2C, PPO-clip, PPO adaptive-KL and V-trace. The chain rule then
    /// gives the parameter gradient through the policy's analytic score oracle:
    ///
    /// ```text
    /// ∇_θ L = Σᵢ (∂L/∂log πᵢ)·∇_θ log πᵢ  −  entropy_coeff · ∇_θ H̄
    /// ```
    ///
    /// (the entropy term enters with a minus sign because the reported
    /// `entropy_loss` is `−H̄`). The result is globally norm-clipped, recorded in
    /// the metrics, scaled by the current learning rate and **negated** before
    /// being handed to the network — parameters move *down* the loss, and the
    /// learning rate is genuinely applied.
    fn apply_policy_gradient_step(
        &mut self,
        observations: &Array2<T>,
        actions: &Array2<T>,
        dloss_dlogp: &Array1<T>,
    ) -> Result<()> {
        let mut gradients =
            self.policy_network
                .log_prob_gradient(observations, actions, dloss_dlogp)?;

        let entropy_coeff = self.config.base_config.entropy_coeff;
        if entropy_coeff != T::zero() {
            let entropy_grad = self.policy_network.entropy_gradient(observations)?;
            for (name, grad) in entropy_grad {
                match gradients.get_mut(&name) {
                    Some(target) => {
                        if target.len() != grad.len() {
                            return Err(OptimError::DimensionMismatch(format!(
                                "entropy gradient for '{name}' has length {} but the log-prob \
                                 gradient has length {}",
                                grad.len(),
                                target.len()
                            )));
                        }
                        for i in 0..target.len() {
                            target[i] = target[i] - entropy_coeff * grad[i];
                        }
                    }
                    None => {
                        gradients.insert(name, grad.mapv(|g| -entropy_coeff * g));
                    }
                }
            }
        }

        let (clipped, norm) =
            clip_named_gradients(&gradients, self.config.base_config.max_grad_norm);
        self.metrics.policy_grad_norm = norm;

        let step = scale_named_gradients(&clipped, -self.policy_lr());
        self.policy_network.update_parameters(&step)
    }

    /// Apply one gradient-descent step to the value network.
    ///
    /// `dloss_dv[i] = ∂L/∂V(sᵢ)`, already multiplied by `value_loss_coeff`.
    /// A no-op when no value network is configured.
    fn apply_value_gradient_step(
        &mut self,
        observations: &Array2<T>,
        dloss_dv: &Array1<T>,
    ) -> Result<()> {
        let max_norm = self.config.base_config.max_grad_norm;
        let lr = self.value_lr();

        let gradients = match self.value_network {
            Some(ref value_net) => value_net.value_gradient(observations, dloss_dv)?,
            None => return Ok(()),
        };

        let (clipped, norm) = clip_named_gradients(&gradients, max_norm);
        self.metrics.value_grad_norm = norm;

        let step = scale_named_gradients(&clipped, -lr);
        if let Some(ref mut value_net) = self.value_network {
            value_net.update_parameters(&step)?;
        }
        Ok(())
    }

    /// Value loss together with its per-sample derivative `∂L/∂V(sᵢ)`.
    ///
    /// Honours the PPO value-clipping toggle: the pessimistic `max` of the raw and
    /// clipped squared errors is used, and the derivative follows whichever branch
    /// the `max` selected (zero when the clipped branch wins *and* the prediction
    /// has left the clip interval, exactly like the clipped policy objective).
    fn value_loss_and_grad(
        &self,
        predicted: &Array1<T>,
        old_values: &Array1<T>,
        returns: &Array1<T>,
    ) -> Result<(T, Array1<T>)> {
        let n = predicted.len();
        if n == 0 {
            return Ok((T::zero(), Array1::zeros(0)));
        }
        if old_values.len() != n || returns.len() != n {
            return Err(OptimError::DimensionMismatch(
                "value prediction, old value and return batches must have equal length".to_string(),
            ));
        }

        let count = T::from(n).ok_or_else(|| {
            OptimError::ComputationError("failed to convert batch size to scalar".to_string())
        })?;
        let inv_n = T::one() / count;
        let two = T::one() + T::one();

        let mut loss = T::zero();
        let mut grad = Array1::zeros(n);

        if self.config.ppo_config.value_clip {
            let clip_range = self.config.ppo_config.value_clip_range;
            for i in 0..n {
                let diff = predicted[i] - old_values[i];
                let clamped = diff.max(-clip_range).min(clip_range);
                let clipped_pred = old_values[i] + clamped;

                let raw_err = predicted[i] - returns[i];
                let clipped_err = clipped_pred - returns[i];
                let l1 = raw_err * raw_err;
                let l2 = clipped_err * clipped_err;

                if l1 >= l2 {
                    loss = loss + l1 * inv_n;
                    grad[i] = two * raw_err * inv_n;
                } else {
                    loss = loss + l2 * inv_n;
                    // d clipped_pred / d predicted is 1 inside the clip interval, 0 outside.
                    let inside = diff.abs() < clip_range;
                    grad[i] = if inside {
                        two * clipped_err * inv_n
                    } else {
                        T::zero()
                    };
                }
            }
        } else {
            for i in 0..n {
                let err = predicted[i] - returns[i];
                loss = loss + err * err * inv_n;
                grad[i] = two * err * inv_n;
            }
        }

        Ok((loss, grad))
    }

    /// Run the value regression step for a batch, returning the value loss.
    ///
    /// Evaluates the critic, forms the (optionally clipped) squared-error loss and
    /// its derivative, and applies a real gradient step scaled by
    /// `value_loss_coeff`.
    fn update_value_on_batch(
        &mut self,
        observations: &Array2<T>,
        old_values: &Array1<T>,
        returns: &Array1<T>,
    ) -> Result<T> {
        let predicted = match self.value_network {
            Some(ref value_net) => value_net.evaluate_value(observations)?,
            None => return Ok(T::zero()),
        };

        let (loss, grad) = self.value_loss_and_grad(&predicted, old_values, returns)?;
        let coeff = self.config.base_config.value_loss_coeff;
        let scaled = grad.mapv(|g| g * coeff);
        self.apply_value_gradient_step(observations, &scaled)?;
        Ok(loss)
    }

    /// Mean entropy loss (`−H̄`) of a policy evaluation.
    fn entropy_loss(evaluation: &super::PolicyEvaluation<T>) -> T {
        let len = evaluation.entropy.len();
        if len == 0 {
            return T::zero();
        }
        let count = T::from(len).unwrap_or_else(T::one);
        -evaluation.entropy.iter().copied().sum::<T>() / count
    }

    /// Get current optimization metrics
    pub fn get_metrics(&self) -> &RLOptimizationMetrics<T> {
        &self.metrics
    }

    /// Add trajectory to buffer
    pub fn add_trajectory(&mut self, trajectory: TrajectoryBatch<T>) {
        self.trajectory_buffer.push(trajectory);
        if self.trajectory_buffer.len() > self.max_buffer_size {
            self.trajectory_buffer.remove(0);
        }
    }

    /// Update using buffered trajectories
    pub fn update_from_buffer(&mut self) -> Result<RLOptimizationMetrics<T>> {
        if self.trajectory_buffer.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No trajectories in buffer".to_string(),
            ));
        }

        // Combine all trajectories
        let combined = self.combine_trajectories()?;
        self.update(combined)
    }

    /// Combine multiple trajectories into one batch
    fn combine_trajectories(&self) -> Result<TrajectoryBatch<T>> {
        if self.trajectory_buffer.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No trajectories to combine".to_string(),
            ));
        }

        let total_size: usize = self
            .trajectory_buffer
            .iter()
            .map(|t| t.observations.nrows())
            .sum();

        let obs_dim = self.trajectory_buffer[0].observations.ncols();
        let action_dim = self.trajectory_buffer[0].actions.ncols();

        let mut combined_obs = Array2::zeros((total_size, obs_dim));
        let mut combined_actions = Array2::zeros((total_size, action_dim));
        let mut combined_log_probs = Array1::zeros(total_size);
        let mut combined_rewards = Array1::zeros(total_size);
        let mut combined_values = Array1::zeros(total_size);
        let mut combined_dones = Vec::with_capacity(total_size);

        let mut offset = 0;
        for trajectory in &self.trajectory_buffer {
            let size = trajectory.observations.nrows();

            combined_obs
                .slice_mut(s![offset..offset + size, ..])
                .assign(&trajectory.observations);
            combined_actions
                .slice_mut(s![offset..offset + size, ..])
                .assign(&trajectory.actions);
            combined_log_probs
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.log_probs);
            combined_rewards
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.rewards);
            combined_values
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.values);

            // `as_slice` returns None for any non-contiguous (sliced/strided)
            // array, so iterate instead of unwrapping a layout assumption.
            combined_dones.extend(trajectory.dones.iter().copied());

            offset += size;
        }

        let combined_dones_array = Array1::from_vec(combined_dones);

        let combined = TrajectoryBatch::new(
            combined_obs,
            combined_actions,
            combined_log_probs,
            combined_rewards,
            combined_values,
            combined_dones_array,
        )?;

        // The combined batch ends where the last buffered trajectory ended.
        match self
            .trajectory_buffer
            .last()
            .and_then(|t| t.final_observation.clone())
        {
            Some(final_observation) => combined.with_final_observation(final_observation),
            None => Ok(combined),
        }
    }

    /// Clear trajectory buffer
    pub fn clear_buffer(&mut self) {
        self.trajectory_buffer.clear();
    }
}

// Import slice syntax
use scirs2_core::ndarray::s;
// use statrs::statistics::Statistics; // statrs not available

#[cfg(test)]
mod tests {
    use super::super::{
        ActionDistribution, DistributionType, PolicyEvaluation, RLOptimizerConfig, TrajectoryBatch,
        ValueNetwork,
    };
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2, Array1, Array2};
    use std::collections::HashMap;

    /// Mock policy whose log-probability *is* its single parameter `w[0]`.
    ///
    /// That makes it genuinely differentiable — `∂ log π/∂w = 1` — so the real
    /// gradient path can be exercised end to end while the log-ratio against the
    /// trajectory's stored (zero) log-probs stays exactly `w[0]`, letting tests
    /// drive the adaptive-KL coefficient up or down.
    struct MockPolicy {
        params: HashMap<String, Array1<f64>>,
        entropy: f64,
        /// Number of times `update_parameters` has been invoked.
        update_calls: usize,
    }

    impl MockPolicy {
        fn new(log_prob_offset: f64) -> Self {
            let mut params = HashMap::new();
            params.insert("w".to_string(), arr1(&[log_prob_offset]));
            Self {
                params,
                entropy: 0.5,
                update_calls: 0,
            }
        }

        fn offset(&self) -> f64 {
            self.params["w"][0]
        }
    }

    impl PolicyNetwork<f64> for MockPolicy {
        fn evaluate_actions(
            &self,
            observations: &Array2<f64>,
            _actions: &Array2<f64>,
        ) -> Result<PolicyEvaluation<f64>> {
            let n = observations.nrows();
            let log_probs = Array1::from_elem(n, self.offset());
            let entropy = Array1::from_elem(n, self.entropy);
            Ok(PolicyEvaluation {
                log_probs,
                entropy,
                metrics: HashMap::new(),
            })
        }

        fn get_action_distribution(
            &self,
            _observations: &Array2<f64>,
        ) -> Result<ActionDistribution<f64>> {
            Ok(ActionDistribution {
                mean: None,
                std: None,
                logits: None,
                distribution_type: DistributionType::Gaussian,
            })
        }

        fn update_parameters(&mut self, deltas: &HashMap<String, Array1<f64>>) -> Result<()> {
            self.update_calls += 1;
            for (key, delta) in deltas {
                if let Some(p) = self.params.get_mut(key) {
                    if p.len() == delta.len() {
                        *p = &*p + delta;
                    }
                }
            }
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            self.params.clone()
        }

        fn log_prob_gradient(
            &self,
            _observations: &Array2<f64>,
            _actions: &Array2<f64>,
            coefficients: &Array1<f64>,
        ) -> Result<HashMap<String, Array1<f64>>> {
            // log π_i = w[0] ⇒ ∂/∂w Σ c_i log π_i = Σ c_i.
            let mut map = HashMap::new();
            map.insert("w".to_string(), arr1(&[coefficients.iter().sum::<f64>()]));
            Ok(map)
        }

        fn entropy_gradient(
            &self,
            _observations: &Array2<f64>,
        ) -> Result<HashMap<String, Array1<f64>>> {
            // Entropy is a constant here, so its gradient is exactly zero.
            let mut map = HashMap::new();
            map.insert("w".to_string(), arr1(&[0.0]));
            Ok(map)
        }
    }

    /// Bias-only linear value function `V(s) = v[0]`: constant in the state but
    /// genuinely differentiable (`∂V/∂v = 1`), so value updates are real.
    struct MockValue {
        params: HashMap<String, Array1<f64>>,
        /// Number of times `update_parameters` has been invoked.
        update_calls: usize,
    }

    impl MockValue {
        fn new() -> Self {
            Self::with_value(0.0)
        }

        fn with_value(value: f64) -> Self {
            let mut params = HashMap::new();
            params.insert("v".to_string(), arr1(&[value]));
            Self {
                params,
                update_calls: 0,
            }
        }
    }

    impl ValueNetwork<f64> for MockValue {
        fn evaluate_value(&self, observations: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(observations.nrows(), self.params["v"][0]))
        }

        fn update_parameters(&mut self, deltas: &HashMap<String, Array1<f64>>) -> Result<()> {
            self.update_calls += 1;
            for (key, delta) in deltas {
                if let Some(p) = self.params.get_mut(key) {
                    if p.len() == delta.len() {
                        *p = &*p + delta;
                    }
                }
            }
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            self.params.clone()
        }

        fn value_gradient(
            &self,
            _observations: &Array2<f64>,
            residuals: &Array1<f64>,
        ) -> Result<HashMap<String, Array1<f64>>> {
            let mut map = HashMap::new();
            map.insert("v".to_string(), arr1(&[residuals.iter().sum::<f64>()]));
            Ok(map)
        }
    }

    /// Build a tiny 4-step trajectory (2-dim observations, 1-dim actions).
    /// Old log-probs are all zero so the mock's offset directly sets log_ratio.
    fn make_trajectory() -> TrajectoryBatch<f64> {
        let observations = arr2(&[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]);
        let actions = arr2(&[[0.0], [1.0], [0.0], [1.0]]);
        let log_probs = arr1(&[0.0, 0.0, 0.0, 0.0]);
        let rewards = arr1(&[1.0, 0.5, 0.25, 1.0]);
        let values = arr1(&[0.0, 0.0, 0.0, 0.0]);
        let dones = Array1::from_vec(vec![false, false, false, true]);
        TrajectoryBatch::new(observations, actions, log_probs, rewards, values, dones)
            .expect("valid trajectory")
    }

    fn make_optimizer(
        log_prob_offset: f64,
        kl_coeff: f64,
        target_kl: f64,
    ) -> PolicyGradientOptimizer<f64, MockPolicy, MockValue> {
        let ppo_config = PPOConfig::<f64> {
            kl_coeff,
            target_kl,
            // Disable early stopping so every mini-batch contributes to epoch KL.
            early_stop_on_kl: false,
            ..PPOConfig::default()
        };
        let base_config = RLOptimizerConfig::<f64> {
            // One epoch; mini-batch covers the whole 4-step trajectory.
            n_epochs: 1,
            mini_batchsize: 4,
            ..RLOptimizerConfig::default()
        };
        let config = PolicyGradientConfig::<f64> {
            base_config,
            method: PolicyGradientMethod::PPOAdaptiveKL,
            ppo_config,
            ..PolicyGradientConfig::default()
        };
        PolicyGradientOptimizer::new(
            config,
            MockPolicy::new(log_prob_offset),
            Some(MockValue::new()),
        )
    }

    #[test]
    fn test_adaptive_kl_runs_and_reports_finite_kl() {
        // Moderate KL via a small negative offset.
        let mut opt = make_optimizer(-0.01, 0.2, 0.01);
        let traj = make_trajectory();
        let metrics = opt.update(traj).expect("update should succeed");

        let kl = metrics.kl_divergence.expect("kl_divergence must be set");
        assert!(kl.is_finite(), "KL must be finite, got {kl}");
        // Adaptive-KL does not clip, so clip_fraction is None.
        assert!(metrics.clip_fraction.is_none());
        // Losses must all be finite.
        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.value_loss.is_finite());
        assert!(metrics.total_loss.is_finite());
    }

    #[test]
    fn test_adaptive_kl_increases_beta_on_large_kl() {
        // offset = -1.0 ⇒ KL ≈ 1.0, far above 1.5 * target_kl (=0.015) ⇒ β doubles.
        let mut opt = make_optimizer(-1.0, 0.2, 0.01);
        let beta_before = opt.kl_coeff;
        let traj = make_trajectory();
        let _ = opt.update(traj).expect("update should succeed");
        let beta_after = opt.kl_coeff;

        assert!(
            beta_after > beta_before,
            "β should increase on large KL: before={beta_before}, after={beta_after}"
        );
        assert!((beta_after - beta_before * 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_kl_decreases_beta_on_small_kl() {
        // offset = 0.0 ⇒ KL = 0, far below target_kl / 1.5 ⇒ β halves.
        let mut opt = make_optimizer(0.0, 0.2, 0.01);
        let beta_before = opt.kl_coeff;
        let traj = make_trajectory();
        let _ = opt.update(traj).expect("update should succeed");
        let beta_after = opt.kl_coeff;

        assert!(
            beta_after < beta_before,
            "β should decrease on small KL: before={beta_before}, after={beta_after}"
        );
        assert!((beta_after - beta_before / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_kl_beta_persists_across_updates() {
        // Repeated large-KL updates should keep growing β multiplicatively
        // until the clamp ceiling (1e4) is reached.
        let mut opt = make_optimizer(-1.0, 0.2, 0.01);
        let beta0 = opt.kl_coeff;

        let _ = opt.update(make_trajectory()).expect("update 1");
        let beta1 = opt.kl_coeff;
        let _ = opt.update(make_trajectory()).expect("update 2");
        let beta2 = opt.kl_coeff;

        assert!(beta1 > beta0);
        assert!(beta2 > beta1);
        // Two doublings from 0.2 (still well under the clamp ceiling).
        assert!((beta2 - beta0 * 4.0).abs() < 1e-9);
    }

    // ----------------------------------------------------------------------
    // A2C / A3C / IMPALA tests
    // ----------------------------------------------------------------------

    /// Build an optimizer for an arbitrary method with a configurable value
    /// baseline and behavior/learner log-prob offset.
    fn make_method_optimizer(
        method: PolicyGradientMethod,
        log_prob_offset: f64,
        value_baseline: f64,
    ) -> PolicyGradientOptimizer<f64, MockPolicy, MockValue> {
        let base_config = RLOptimizerConfig::<f64> {
            n_epochs: 1,
            mini_batchsize: 4,
            ..RLOptimizerConfig::default()
        };
        let config = PolicyGradientConfig::<f64> {
            base_config,
            method,
            ..PolicyGradientConfig::default()
        };
        PolicyGradientOptimizer::new(
            config,
            MockPolicy::new(log_prob_offset),
            Some(MockValue::with_value(value_baseline)),
        )
    }

    #[test]
    fn test_actor_critic_runs_and_forwards_updates() {
        // A2C with a non-zero value baseline so the value loss is informative.
        let mut opt = make_method_optimizer(PolicyGradientMethod::ActorCritic, -0.2, 0.1);
        let traj = make_trajectory();
        let metrics = opt.update(traj).expect("A2C update should succeed");

        // All reported losses must be finite.
        assert!(
            metrics.policy_loss.is_finite(),
            "policy loss must be finite"
        );
        assert!(metrics.value_loss.is_finite(), "value loss must be finite");
        assert!(
            metrics.entropy_loss.is_finite(),
            "entropy loss must be finite"
        );
        assert!(metrics.total_loss.is_finite(), "total loss must be finite");
        // A2C performs no clipping and is on-policy w.r.t. the data.
        assert!(metrics.clip_fraction.is_none());
        assert_eq!(metrics.kl_divergence, Some(0.0));
        // Value loss must be strictly positive: V=0.1 vs non-trivial returns.
        assert!(metrics.value_loss > 0.0);

        // Gradients must actually have been forwarded to BOTH mock networks.
        assert!(
            opt.policy_network.update_calls > 0,
            "policy network must receive parameter updates"
        );
        let value_net = opt.value_network.as_ref().expect("value net present");
        assert!(
            value_net.update_calls > 0,
            "value network must receive parameter updates"
        );
        // And the parameters must have moved away from their zero init (the
        // value gradient is non-zero because the value loss is non-zero).
        let v = &value_net.get_parameters()["v"];
        assert!(v.iter().any(|&x| x != 0.0), "value params must change");
    }

    #[test]
    fn test_a3c_dispatches_and_equals_a2c() {
        // A3C is asynchronous A2C; a single synchronous update must produce
        // results identical to ActorCritic given the same fresh state + data.
        let mut a2c = make_method_optimizer(PolicyGradientMethod::ActorCritic, -0.2, 0.1);
        let mut a3c = make_method_optimizer(PolicyGradientMethod::A3C, -0.2, 0.1);

        let m_a2c = a2c.update(make_trajectory()).expect("A2C update");
        let m_a3c = a3c.update(make_trajectory()).expect("A3C update");

        // Neither must error, and the per-update math is identical.
        assert_eq!(m_a2c.policy_loss, m_a3c.policy_loss);
        assert_eq!(m_a2c.value_loss, m_a3c.value_loss);
        assert_eq!(m_a2c.entropy_loss, m_a3c.entropy_loss);
        assert_eq!(m_a2c.total_loss, m_a3c.total_loss);
        assert_eq!(m_a2c.kl_divergence, m_a3c.kl_divergence);
        assert_eq!(m_a2c.clip_fraction, m_a3c.clip_fraction);

        // The networks must have been updated identically too.
        assert_eq!(
            a2c.policy_network.get_parameters()["w"],
            a3c.policy_network.get_parameters()["w"]
        );
        // Note: this mock's log-probability is state-independent, so with
        // mean-zero (normalized) advantages its exact policy gradient is zero.
        // End-to-end learning is covered by the linear-policy convergence tests.
    }

    #[test]
    fn test_impala_dispatches_and_forwards_updates() {
        // Off-policy IMPALA: behavior log-prob = 0, learner log-prob = ln(0.5)
        // ⇒ importance ratio 0.5, truncated to ρ = c = 0.5.
        let offset = 0.5_f64.ln();
        let mut opt = make_method_optimizer(PolicyGradientMethod::IMPALA, offset, 0.1);
        let metrics = opt
            .update(make_trajectory())
            .expect("IMPALA update should succeed");

        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.value_loss.is_finite());
        assert!(metrics.total_loss.is_finite());
        assert!(metrics.clip_fraction.is_none());
        // Off-policy diagnostic: mean truncated importance weight ≈ 0.5.
        let mean_rho = metrics
            .custom_metrics
            .get("mean_rho")
            .copied()
            .expect("mean_rho must be surfaced");
        assert!(
            (mean_rho - 0.5).abs() < 1e-9,
            "mean ρ should be 0.5, got {mean_rho}"
        );

        // Both networks must receive parameter updates.
        assert!(opt.policy_network.update_calls > 0);
        assert!(opt.value_network.as_ref().expect("value net").update_calls > 0);
    }

    #[test]
    fn test_impala_vtrace_target_matches_discounted_return() {
        // With learner == behavior policy (offset 0 ⇒ ratio 1 ⇒ ρ = c = 1) and a
        // zero value baseline, the V-trace targets collapse to the plain
        // discounted Monte-Carlo return. For a 2-step trajectory the value loss
        // (= mean(v_t²) since V≡0) is therefore exactly computable.
        let base_config = RLOptimizerConfig::<f64> {
            n_epochs: 1,
            mini_batchsize: 2,
            ..RLOptimizerConfig::default()
        };
        let config = PolicyGradientConfig::<f64> {
            base_config,
            method: PolicyGradientMethod::IMPALA,
            ..PolicyGradientConfig::default()
        };
        let mut opt = PolicyGradientOptimizer::new(
            config,
            MockPolicy::new(0.0), // learner log-prob == behavior log-prob
            Some(MockValue::with_value(0.0)),
        );

        // 2-step trajectory: r = [1.0, 0.5], second step terminal, V ≡ 0.
        let observations = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let actions = arr2(&[[0.0], [1.0]]);
        let log_probs = arr1(&[0.0, 0.0]);
        let rewards = arr1(&[1.0, 0.5]);
        let values = arr1(&[0.0, 0.0]);
        let dones = Array1::from_vec(vec![false, true]);
        let traj = TrajectoryBatch::new(observations, actions, log_probs, rewards, values, dones)
            .expect("valid trajectory");

        let metrics = opt.update(traj).expect("IMPALA update should succeed");

        // Discounted returns (γ = 0.99): v1 = 0.5, v0 = 1 + 0.99·0.5 = 1.495.
        let gamma = 0.99_f64;
        let v1 = 0.5;
        let v0 = 1.0 + gamma * v1;
        let expected_value_loss = (v0 * v0 + v1 * v1) / 2.0;

        assert!(
            (metrics.value_loss - expected_value_loss).abs() < 1e-9,
            "V-trace value loss {} should equal discounted-return MSE {}",
            metrics.value_loss,
            expected_value_loss
        );
        // ρ = 1 everywhere here.
        let mean_rho = metrics
            .custom_metrics
            .get("mean_rho")
            .copied()
            .expect("mean_rho");
        assert!((mean_rho - 1.0).abs() < 1e-9);
    }
}
