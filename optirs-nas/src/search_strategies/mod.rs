// Search strategies for neural architecture search
//
// Implements various NAS algorithms including DARTS, evolutionary search,
// reinforcement learning-based search, and Bayesian optimization.

pub mod bayesian;
pub mod differentiable;
pub mod evolutionary;
pub mod neural_predictor;
pub mod progressive;
pub mod random;
pub mod rl_search;

use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

#[allow(unused_imports)]
use crate::error::Result;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};

// Re-export all public types
pub use bayesian::{
    AcquisitionFunction, AcquisitionType, BayesianOptimization, GPKernel, GaussianProcess,
    KernelType,
};
pub use differentiable::{
    DARTSConfig, DifferentiableSearch, DiscretizationStrategy, MemoryEfficientDARTS, RobustDARTS,
    TemperatureSchedule, WeightOptimizer,
};
pub use evolutionary::EvolutionarySearch;
pub use neural_predictor::{
    ActivationFunction, ArchitectureEncoder, NeuralPredictorSearch, PredictorLayer,
    PredictorNetwork, SearchOptimizer, SearchOptimizerType,
};
pub use progressive::ProgressiveNAS;
pub use random::RandomSearch;
pub use rl_search::{
    BaselineOptimizer, BaselinePredictor, ControllerNetwork, ExperienceBuffer, PolicyOptimizer,
    ReinforcementLearningSearch,
};

/// Base trait for all search strategies
pub trait SearchStrategy<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Initialize the search strategy
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()>;

    /// Generate a new architecture candidate
    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>>;

    /// Update strategy with evaluation results
    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get current search statistics
    fn get_statistics(&self) -> SearchStrategyStatistics<T>;
}

/// Search strategy statistics
#[derive(Debug, Clone)]
pub struct SearchStrategyStatistics<T: Float + Debug + Send + Sync + 'static> {
    pub total_architectures_generated: usize,
    pub best_performance: T,
    pub average_performance: T,
    pub convergence_rate: T,
    pub exploration_rate: T,
    pub exploitation_rate: T,
}

// Default implementations for statistics
impl<T: Float + Debug + Default + Send + Sync> Default for SearchStrategyStatistics<T> {
    fn default() -> Self {
        Self {
            total_architectures_generated: 0,
            best_performance: T::zero(),
            average_performance: T::zero(),
            convergence_rate: T::zero(),
            exploration_rate: scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
            exploitation_rate: scirs2_core::numeric::NumCast::from(0.5)
                .unwrap_or_else(|| T::zero()),
        }
    }
}

/// Convert ComponentType to u8 for encoding
pub fn component_type_to_u8(componenttype: &crate::architecture::ComponentType) -> u8 {
    use crate::architecture::ComponentType;
    match componenttype {
        ComponentType::SGD => 0,
        ComponentType::Adam => 1,
        ComponentType::AdaGrad => 2,
        ComponentType::RMSprop => 3,
        ComponentType::AdamW => 4,
        ComponentType::LAMB => 5,
        ComponentType::LARS => 6,
        ComponentType::Lion => 7,
        ComponentType::RAdam => 8,
        ComponentType::Lookahead => 9,
        ComponentType::SAM => 10,
        ComponentType::LBFGS => 11,
        ComponentType::SparseAdam => 12,
        ComponentType::GroupedAdam => 13,
        ComponentType::MAML => 14,
        ComponentType::Reptile => 15,
        ComponentType::MetaSGD => 16,
        ComponentType::ConstantLR => 17,
        ComponentType::ExponentialLR => 18,
        ComponentType::StepLR => 19,
        ComponentType::CosineAnnealingLR => 20,
        ComponentType::OneCycleLR => 21,
        ComponentType::CyclicLR => 22,
        ComponentType::L1Regularizer => 23,
        ComponentType::L2Regularizer => 24,
        ComponentType::ElasticNetRegularizer => 25,
        ComponentType::DropoutRegularizer => 26,
        ComponentType::GradientClipping => 27,
        ComponentType::WeightDecay => 28,
        ComponentType::AdaptiveLR => 29,
        ComponentType::AdaptiveMomentum => 30,
        ComponentType::AdaptiveRegularization => 31,
        ComponentType::LSTMOptimizer => 32,
        ComponentType::TransformerOptimizer => 33,
        ComponentType::AttentionOptimizer => 34,
        ComponentType::AdaDelta => 35,
        ComponentType::Momentum => 36,
        ComponentType::Nesterov => 37,
        ComponentType::LRScheduler => 38,
        ComponentType::BatchNorm => 39,
        ComponentType::Dropout => 40,
        ComponentType::Custom => 255,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_search_creation() {
        let search = RandomSearch::<f64>::new(Some(42));
        assert_eq!(search.name(), "RandomSearch");
    }

    #[test]
    fn test_evolutionary_search_creation() {
        let search = EvolutionarySearch::<f64>::new(50, 0.1, 0.8, 3);
        assert_eq!(search.name(), "EvolutionarySearch");
        assert_eq!(search.population_size, 50);
    }

    #[test]
    fn test_rl_search_creation() {
        let search = ReinforcementLearningSearch::<f64>::new(256, 2, 0.001);
        assert_eq!(search.name(), "ReinforcementLearningSearch");
    }
}
