use crate::reinforcement_learning::RLOptimizerConfig;
use scirs2_core::numeric::Float;
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
