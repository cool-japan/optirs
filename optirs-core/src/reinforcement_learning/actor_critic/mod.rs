// Actor-Critic Optimizers
//
// This module implements various actor-critic algorithms including A2C, A3C,
// SAC (Soft Actor-Critic), and other modern actor-critic methods.

mod config;
pub use config::{ActorCriticConfig, ActorCriticMethod, DDPGConfig, SACConfig, TD3Config};

mod metrics;
pub use metrics::ActorCriticMetrics;

mod replay;
pub use replay::{Experience, ExperienceReplayBuffer, ReplaySample};

mod optimizer;
pub use optimizer::ActorCriticOptimizer;

#[cfg(test)]
mod tests;
