// Adaptive routing optimization

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Adaptive routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRouting {
    /// Adaptation enabled
    pub enabled: bool,
    /// Learning algorithm
    pub learning_algorithm: LearningAlgorithm,
    /// Adaptation interval
    pub interval: Duration,
    /// Performance feedback
    pub feedback: PerformanceFeedback,
}

impl Default for AdaptiveRouting {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_algorithm: LearningAlgorithm::ReinforcementLearning,
            interval: Duration::from_secs(60),
            feedback: PerformanceFeedback::default(),
        }
    }
}

/// Learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    /// Reinforcement learning
    ReinforcementLearning,
    /// Q-learning
    QLearning,
    /// Neural networks
    NeuralNetworks,
    /// Custom algorithm
    Custom(String),
}

/// Performance feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Feedback sources
    pub sources: Vec<String>,
    /// Feedback weight
    pub weight: f64,
    /// Feedback delay
    pub delay: Duration,
}

impl Default for PerformanceFeedback {
    fn default() -> Self {
        Self {
            sources: vec!["latency".to_string(), "throughput".to_string()],
            weight: 1.0,
            delay: Duration::from_secs(10),
        }
    }
}
