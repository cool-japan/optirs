// Policy Gradient Optimizers
//
// This module implements various policy gradient methods including REINFORCE,
// PPO (Proximal Policy Optimization), TRPO (Trust Region Policy Optimization),
// and other modern policy gradient algorithms.

use super::{
    PolicyNetwork, RLOptimizationMetrics, RLOptimizerConfig, RLScheduler, ScheduleType,
    TrajectoryBatch, ValueNetwork,
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

    /// Bootstrap value for the step *after* the final observation in the
    /// trajectory. Uses the (current) value network on the last observation,
    /// or zero when no value network is configured. Shared by every update rule
    /// that needs a GAE / V-trace bootstrap so the logic lives in one place.
    fn bootstrap_next_value(&self, trajectory: &TrajectoryBatch<T>) -> Result<T> {
        if let Some(ref value_net) = self.value_network {
            let last_obs = trajectory.observations.slice(s![-1.., ..]).to_owned();
            let mut last_obs_batch = Array2::zeros((1, last_obs.ncols()));
            last_obs_batch.row_mut(0).assign(&last_obs.row(0));
            Ok(value_net.evaluate_value(&last_obs_batch)?[0])
        } else {
            Ok(T::zero())
        }
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

    /// PPO with clipped surrogate objective
    fn update_ppo_clip(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let mut total_policy_loss = T::zero();
        let mut total_value_loss = T::zero();
        let mut total_entropy_loss = T::zero();
        let mut clip_fraction = T::zero();
        let mut approx_kl = T::zero();

        // Compute advantages using GAE (with value-network bootstrap).
        self.prepare_gae(&mut trajectory)?;

        // Store old policy evaluation
        let _old_policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        let n_epochs = self.config.base_config.n_epochs;
        let mini_batch_size = self.config.base_config.mini_batchsize;

        for _epoch in 0..n_epochs {
            let mini_batches = trajectory.get_mini_batches(mini_batch_size);

            for mini_batch in mini_batches {
                // Current policy evaluation
                let policy_eval = self
                    .policy_network
                    .evaluate_actions(&mini_batch.observations, &mini_batch.actions)?;

                // Compute importance sampling ratio
                let log_ratio = &policy_eval.log_probs - &mini_batch.log_probs;
                let ratio = log_ratio.mapv(|x| x.exp());

                // Compute surrogate loss
                let surr1 = &ratio * &mini_batch.advantages;
                let clipped_ratio = ratio.mapv(|r| {
                    let clip_eps = self.config.ppo_config.clip_epsilon;
                    r.max(T::one() - clip_eps).min(T::one() + clip_eps)
                });
                let surr2 = &clipped_ratio * &mini_batch.advantages;

                // Policy loss (negative because we want to maximize)
                let policy_loss = -surr1
                    .iter()
                    .zip(surr2.iter())
                    .map(|(&s1, &s2)| s1.min(s2))
                    .sum::<T>()
                    / T::from(mini_batch.observations.nrows()).expect("unwrap failed");

                // Entropy loss (negative to encourage exploration)
                let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
                    / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

                // Value function loss
                let value_loss = if let Some(ref value_net) = self.value_network {
                    let predicted_values = value_net.evaluate_value(&mini_batch.observations)?;

                    if self.config.ppo_config.value_clip {
                        // Clipped value loss
                        let value_pred_clipped = &mini_batch.values
                            + (&predicted_values - &mini_batch.values).mapv(|diff| {
                                let clip_range = self.config.ppo_config.value_clip_range;
                                diff.max(-clip_range).min(clip_range)
                            });

                        let value_loss_1 =
                            (&predicted_values - &mini_batch.returns).mapv(|x| x * x);
                        let value_loss_2 =
                            (&value_pred_clipped - &mini_batch.returns).mapv(|x| x * x);

                        value_loss_1
                            .iter()
                            .zip(value_loss_2.iter())
                            .map(|(&v1, &v2)| v1.max(v2))
                            .sum::<T>()
                            / T::from(mini_batch.observations.nrows()).expect("unwrap failed")
                    } else {
                        // Standard MSE loss
                        (&predicted_values - &mini_batch.returns)
                            .mapv(|x| x * x)
                            .mean()
                            .unwrap_or(T::zero())
                    }
                } else {
                    T::zero()
                };

                // Total loss
                let total_loss = policy_loss
                    + self.config.base_config.value_loss_coeff * value_loss
                    + self.config.base_config.entropy_coeff * entropy_loss;

                // Compute gradients and update networks
                self.update_networks_with_loss(total_loss, policy_loss, value_loss)?;

                // Accumulate metrics
                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy_loss += entropy_loss;

                // Compute clip fraction
                let n_clipped = ratio
                    .iter()
                    .filter(|&&r| {
                        let clip_eps = self.config.ppo_config.clip_epsilon;
                        r < T::one() - clip_eps || r > T::one() + clip_eps
                    })
                    .count();
                clip_fraction += T::from(n_clipped).unwrap_or_else(|| T::zero())
                    / T::from(ratio.len()).expect("unwrap failed");

                // Compute approximate KL divergence
                approx_kl += log_ratio.mapv(|x| x * x).mean().unwrap_or(T::zero());

                // Early stopping based on KL divergence
                if self.config.ppo_config.early_stop_on_kl
                    && approx_kl
                        > self.config.ppo_config.target_kl
                            * T::from(2.0).unwrap_or_else(|| T::zero())
                {
                    break;
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

        // Update metrics
        let n_updates =
            T::from(n_epochs * trajectory.observations.nrows().div_ceil(mini_batch_size))
                .expect("unwrap failed");
        self.metrics.policy_loss = total_policy_loss / n_updates;
        self.metrics.value_loss = total_value_loss / n_updates;
        self.metrics.entropy_loss = total_entropy_loss / n_updates;
        self.metrics.total_loss = self.metrics.policy_loss
            + self.config.base_config.value_loss_coeff * self.metrics.value_loss
            + self.config.base_config.entropy_coeff * self.metrics.entropy_loss;
        self.metrics.clip_fraction = Some(clip_fraction / n_updates);
        self.metrics.kl_divergence = Some(approx_kl / n_updates);

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

        // Store old policy evaluation (kept for parity with the clipped path).
        let _old_policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        let n_epochs = self.config.base_config.n_epochs;
        let mini_batch_size = self.config.base_config.mini_batchsize;

        // The KL measured on the final epoch drives the β adaptation.
        let mut last_epoch_kl = T::zero();

        for _epoch in 0..n_epochs {
            let mini_batches = trajectory.get_mini_batches(mini_batch_size);
            let mut epoch_kl_sum = T::zero();
            let mut epoch_batches = T::zero();

            for mini_batch in mini_batches {
                // Current policy evaluation.
                let policy_eval = self
                    .policy_network
                    .evaluate_actions(&mini_batch.observations, &mini_batch.actions)?;

                // Importance sampling ratio = exp(new_logp − old_logp).
                let log_ratio = &policy_eval.log_probs - &mini_batch.log_probs;
                let ratio = log_ratio.mapv(|x| x.exp());

                let batch_count = T::from(mini_batch.observations.nrows()).unwrap_or_else(T::one);

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

                // Entropy loss (negative to encourage exploration).
                let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
                    / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

                // Value function loss (identical to the clipped variant).
                let value_loss = if let Some(ref value_net) = self.value_network {
                    let predicted_values = value_net.evaluate_value(&mini_batch.observations)?;

                    if self.config.ppo_config.value_clip {
                        let value_pred_clipped = &mini_batch.values
                            + (&predicted_values - &mini_batch.values).mapv(|diff| {
                                let clip_range = self.config.ppo_config.value_clip_range;
                                diff.max(-clip_range).min(clip_range)
                            });

                        let value_loss_1 =
                            (&predicted_values - &mini_batch.returns).mapv(|x| x * x);
                        let value_loss_2 =
                            (&value_pred_clipped - &mini_batch.returns).mapv(|x| x * x);

                        value_loss_1
                            .iter()
                            .zip(value_loss_2.iter())
                            .map(|(&v1, &v2)| v1.max(v2))
                            .sum::<T>()
                            / batch_count
                    } else {
                        (&predicted_values - &mini_batch.returns)
                            .mapv(|x| x * x)
                            .mean()
                            .unwrap_or(T::zero())
                    }
                } else {
                    T::zero()
                };

                // Total loss.
                let total_loss = policy_loss
                    + self.config.base_config.value_loss_coeff * value_loss
                    + self.config.base_config.entropy_coeff * entropy_loss;

                // Compute gradients and update networks.
                self.update_networks_with_loss(total_loss, policy_loss, value_loss)?;

                // Accumulate metrics.
                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy_loss += entropy_loss;

                approx_kl += batch_kl;
                epoch_kl_sum += batch_kl;
                epoch_batches += T::one();

                // Early stopping based on KL divergence.
                if self.config.ppo_config.early_stop_on_kl
                    && batch_kl > self.config.ppo_config.target_kl * two
                {
                    break;
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

        // Update metrics.
        let n_updates =
            T::from(n_epochs * trajectory.observations.nrows().div_ceil(mini_batch_size))
                .unwrap_or_else(T::one);
        self.metrics.policy_loss = total_policy_loss / n_updates;
        self.metrics.value_loss = total_value_loss / n_updates;
        self.metrics.entropy_loss = total_entropy_loss / n_updates;
        self.metrics.total_loss = self.metrics.policy_loss
            + self.config.base_config.value_loss_coeff * self.metrics.value_loss
            + self.config.base_config.entropy_coeff * self.metrics.entropy_loss;
        // No clipping in this variant.
        self.metrics.clip_fraction = None;
        self.metrics.kl_divergence = Some(approx_kl / n_updates);
        // Surface the current penalty coefficient for inspection.
        self.metrics
            .custom_metrics
            .insert("kl_coeff".to_string(), self.kl_coeff);

        Ok(self.metrics.clone())
    }

    /// TRPO update with trust region constraint
    fn update_trpo(&mut self, trajectory: TrajectoryBatch<T>) -> Result<RLOptimizationMetrics<T>> {
        // TRPO implementation with conjugate gradient and line search
        // This is a simplified version - full TRPO requires more complex optimization
        self.update_ppo_clip(trajectory) // Simplified for now
    }

    /// REINFORCE algorithm
    fn update_reinforce(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        // Use returns as targets (no baseline)
        let policy_loss = if self.config.use_baseline && self.value_network.is_some() {
            // Actor-critic style with baseline
            -(policy_eval.log_probs * trajectory.advantages)
                .mean()
                .unwrap_or(T::zero())
        } else {
            // Pure REINFORCE
            -(policy_eval.log_probs * trajectory.returns)
                .mean()
                .unwrap_or(T::zero())
        };

        let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
            / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());
        let total_loss = policy_loss + self.config.base_config.entropy_coeff * entropy_loss;

        self.update_networks_with_loss(total_loss, policy_loss, T::zero())?;

        self.metrics.policy_loss = policy_loss;
        self.metrics.entropy_loss = entropy_loss;
        self.metrics.total_loss = total_loss;

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

        let batch_count = T::from(trajectory.observations.nrows()).unwrap_or_else(T::one);

        // Policy loss = -E[ log π(a|s) · A(s,a) ]. The importance ratio is 1
        // (on-policy), so this is the plain advantage-weighted log-likelihood.
        let policy_loss = -(&policy_eval.log_probs * &trajectory.advantages)
            .iter()
            .copied()
            .sum::<T>()
            / batch_count;

        // Entropy loss (negative to encourage exploration), same convention as
        // the PPO paths so `entropy_coeff` behaves identically.
        let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
            / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

        // Value function loss against the GAE returns. Mirrors the clipped /
        // unclipped MSE treatment of `update_ppo_clip`.
        let value_loss = if let Some(ref value_net) = self.value_network {
            let predicted_values = value_net.evaluate_value(&trajectory.observations)?;

            if self.config.ppo_config.value_clip {
                let value_pred_clipped = &trajectory.values
                    + (&predicted_values - &trajectory.values).mapv(|diff| {
                        let clip_range = self.config.ppo_config.value_clip_range;
                        diff.max(-clip_range).min(clip_range)
                    });

                let value_loss_1 = (&predicted_values - &trajectory.returns).mapv(|x| x * x);
                let value_loss_2 = (&value_pred_clipped - &trajectory.returns).mapv(|x| x * x);

                value_loss_1
                    .iter()
                    .zip(value_loss_2.iter())
                    .map(|(&v1, &v2)| v1.max(v2))
                    .sum::<T>()
                    / batch_count
            } else {
                (&predicted_values - &trajectory.returns)
                    .mapv(|x| x * x)
                    .mean()
                    .unwrap_or(T::zero())
            }
        } else {
            T::zero()
        };

        // Total loss combines policy, value and entropy contributions exactly
        // as the PPO variants do.
        let total_loss = policy_loss
            + self.config.base_config.value_loss_coeff * value_loss
            + self.config.base_config.entropy_coeff * entropy_loss;

        // Apply policy & value gradient updates through the networks.
        self.update_networks_with_loss(total_loss, policy_loss, value_loss)?;

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

        let batch_count = T::from(batch_size).unwrap_or_else(T::one);

        // Policy loss = -E[ log π(a_t|s_t) · A_t ] (ρ is already folded into A_t).
        let policy_loss = -(&policy_eval.log_probs * &pg_advantages)
            .iter()
            .copied()
            .sum::<T>()
            / batch_count;

        // Entropy loss (same convention as the other update rules).
        let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
            / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

        // Value loss = MSE(V(s_t), v_t) against the V-trace targets.
        let value_loss = if self.value_network.is_some() {
            (&values_now - &vtrace_targets)
                .mapv(|x| x * x)
                .mean()
                .unwrap_or(T::zero())
        } else {
            T::zero()
        };

        // Total loss.
        let total_loss = policy_loss
            + self.config.base_config.value_loss_coeff * value_loss
            + self.config.base_config.entropy_coeff * entropy_loss;

        // Apply policy & value gradient updates through the networks.
        self.update_networks_with_loss(total_loss, policy_loss, value_loss)?;

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

    /// Update networks with computed losses
    fn update_networks_with_loss(
        &mut self,
        _total_loss: T,
        policy_loss: T,
        value_loss: T,
    ) -> Result<()> {
        // 1. Compute gradients from losses (simplified - would use autodiff in practice)
        let policy_gradients = self.compute_policy_gradients(policy_loss)?;
        let value_gradients = if self.value_network.is_some() {
            Some(self.compute_value_gradients(value_loss)?)
        } else {
            None
        };

        // 2. Apply gradient clipping
        let clipped_policy_grads =
            self.clip_gradients(&policy_gradients, self.config.base_config.max_grad_norm)?;
        let clipped_value_grads = if let Some(val_grads) = value_gradients {
            Some(self.clip_gradients(&val_grads, self.config.base_config.max_grad_norm)?)
        } else {
            None
        };

        // 3. Update network parameters
        self.update_policy_parameters(&clipped_policy_grads)?;
        if let Some(ref val_grads) = clipped_value_grads {
            self.update_value_parameters(val_grads)?;
        }

        // 4. Update gradient norms in metrics
        self.metrics.policy_grad_norm = self.compute_gradient_norm(&clipped_policy_grads);
        if let Some(ref val_grads) = clipped_value_grads {
            self.metrics.value_grad_norm = self.compute_gradient_norm(val_grads);
        }

        Ok(())
    }

    /// Compute policy gradients (simplified)
    fn compute_policy_gradients(&self, loss: T) -> Result<HashMap<String, Array1<T>>> {
        let mut gradients = HashMap::new();

        // Simplified gradient computation - in practice would use autodiff
        let policy_params = self.policy_network.get_parameters();
        for (param_name, param_values) in policy_params {
            let grad = Array1::ones(param_values.len()) * loss
                / T::from(param_values.len()).expect("unwrap failed");
            gradients.insert(param_name, grad);
        }

        Ok(gradients)
    }

    /// Compute value function gradients (simplified)
    fn compute_value_gradients(&self, loss: T) -> Result<HashMap<String, Array1<T>>> {
        let mut gradients = HashMap::new();

        if let Some(ref value_net) = self.value_network {
            let value_params = value_net.get_parameters();
            for (param_name, param_values) in value_params {
                let grad = Array1::ones(param_values.len()) * loss
                    / T::from(param_values.len()).expect("unwrap failed");
                gradients.insert(param_name, grad);
            }
        }

        Ok(gradients)
    }

    /// Apply gradient clipping
    fn clip_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        max_norm: T,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut clipped_gradients = HashMap::new();

        // Compute global gradient _norm
        let mut total_norm = T::zero();
        for grad in gradients.values() {
            total_norm += grad.iter().map(|&g| g * g).sum::<T>();
        }
        total_norm = total_norm.sqrt();

        // Apply clipping if necessary
        let clip_factor = if total_norm > max_norm {
            max_norm / total_norm
        } else {
            T::one()
        };

        for (param_name, grad) in gradients {
            let clipped_grad = grad * clip_factor;
            clipped_gradients.insert(param_name.clone(), clipped_grad);
        }

        Ok(clipped_gradients)
    }

    /// Update policy network parameters
    fn update_policy_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        // Apply gradients to policy network
        self.policy_network.update_parameters(gradients)?;
        Ok(())
    }

    /// Update value network parameters
    fn update_value_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(ref mut value_net) = self.value_network {
            value_net.update_parameters(gradients)?;
        }
        Ok(())
    }

    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Array1<T>>) -> T {
        let mut total_norm = T::zero();
        for grad in gradients.values() {
            total_norm += grad.iter().map(|&g| g * g).sum::<T>();
        }
        total_norm.sqrt()
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

            combined_dones.extend_from_slice(trajectory.dones.as_slice().expect("unwrap failed"));

            offset += size;
        }

        let combined_dones_array = Array1::from_vec(combined_dones);

        TrajectoryBatch::new(
            combined_obs,
            combined_actions,
            combined_log_probs,
            combined_rewards,
            combined_values,
            combined_dones_array,
        )
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

    /// Mock policy whose evaluated log-probs differ from the trajectory's stored
    /// (old) log-probs by a fixed `log_prob_offset`. Because the adaptive-KL
    /// estimator is `KL = mean(old_logp − new_logp) = −offset`, a *negative*
    /// offset yields a large positive KL — letting tests drive β up or down.
    struct MockPolicy {
        params: HashMap<String, Array1<f64>>,
        log_prob_offset: f64,
        entropy: f64,
        /// Number of times `update_parameters` has been invoked. Lets update
        /// tests assert that gradients were actually forwarded to the network.
        update_calls: usize,
    }

    impl MockPolicy {
        fn new(log_prob_offset: f64) -> Self {
            let mut params = HashMap::new();
            params.insert("w".to_string(), arr1(&[0.0, 0.0]));
            Self {
                params,
                log_prob_offset,
                entropy: 0.5,
                update_calls: 0,
            }
        }
    }

    impl PolicyNetwork<f64> for MockPolicy {
        fn evaluate_actions(
            &self,
            observations: &Array2<f64>,
            _actions: &Array2<f64>,
        ) -> Result<PolicyEvaluation<f64>> {
            let n = observations.nrows();
            // New log-probs = a constant (offset). Old log-probs in the
            // trajectory are 0.0, so log_ratio = offset for every sample.
            let log_probs = Array1::from_elem(n, self.log_prob_offset);
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

        fn update_parameters(&mut self, gradients: &HashMap<String, Array1<f64>>) -> Result<()> {
            self.update_calls += 1;
            for (key, grad) in gradients {
                if let Some(p) = self.params.get_mut(key) {
                    if p.len() == grad.len() {
                        *p = &*p + grad;
                    }
                }
            }
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            self.params.clone()
        }
    }

    /// Minimal value network returning a fixed constant; only needs to be a
    /// valid `V`. A non-zero baseline (when requested) exercises the value-loss
    /// and V-trace correction paths with informative numbers.
    struct MockValue {
        params: HashMap<String, Array1<f64>>,
        value: f64,
        /// Number of times `update_parameters` has been invoked.
        update_calls: usize,
    }

    impl MockValue {
        fn new() -> Self {
            Self::with_value(0.0)
        }

        fn with_value(value: f64) -> Self {
            let mut params = HashMap::new();
            params.insert("v".to_string(), arr1(&[0.0, 0.0]));
            Self {
                params,
                value,
                update_calls: 0,
            }
        }
    }

    impl ValueNetwork<f64> for MockValue {
        fn evaluate_value(&self, observations: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(observations.nrows(), self.value))
        }

        fn update_parameters(&mut self, gradients: &HashMap<String, Array1<f64>>) -> Result<()> {
            self.update_calls += 1;
            for (key, grad) in gradients {
                if let Some(p) = self.params.get_mut(key) {
                    if p.len() == grad.len() {
                        *p = &*p + grad;
                    }
                }
            }
            Ok(())
        }

        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            self.params.clone()
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
