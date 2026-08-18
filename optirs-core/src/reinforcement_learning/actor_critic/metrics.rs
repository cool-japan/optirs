use crate::reinforcement_learning::RLOptimizationMetrics;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

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
