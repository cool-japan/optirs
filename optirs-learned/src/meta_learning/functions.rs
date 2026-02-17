//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(unused_imports)]
use crate::error::Result;
#[allow(dead_code)]
use scirs2_core::ndarray::{Array1, Array2, Dimension};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::types::{
    AdaptationStrategy, AntiForgettingStrategy, AugmentationStrategy, ContinualLearningSettings,
    DistanceMetric, FewShotAlgorithm, FewShotSettings, GradientBalancingMethod,
    InterferenceMitigationStrategy, MAMLConfig, MemoryReplaySettings, MemorySelectionCriteria,
    MetaLearningAlgorithm, MetaLearningConfig, MetaTask, MetaTrainingResult,
    MetricLearningSettings, MultiTaskSettings, QueryEvaluationResult, ReplayStrategy,
    SharedRepresentationStrategy, SimilarityMeasure, TaskAdaptationResult,
    TaskIdentificationMethod, TaskSamplingStrategy, TaskWeightingStrategy,
    TransferLearningSettings, TransferStrategy,
};

/// Meta-learner trait
pub trait MetaLearner<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Perform meta-training step
    fn meta_train_step(
        &mut self,
        task_batch: &[MetaTask<T>],
        meta_parameters: &mut HashMap<String, Array1<T>>,
    ) -> Result<MetaTrainingResult<T>>;
    /// Adapt to new task
    fn adapt_to_task(
        &mut self,
        task: &MetaTask<T>,
        meta_parameters: &HashMap<String, Array1<T>>,
        adaptation_steps: usize,
    ) -> Result<TaskAdaptationResult<T>>;
    /// Evaluate on query set
    fn evaluate_query_set(
        &self,
        task: &MetaTask<T>,
        adapted_parameters: &HashMap<String, Array1<T>>,
    ) -> Result<QueryEvaluationResult<T>>;
    /// Get meta-learner type
    fn get_algorithm(&self) -> MetaLearningAlgorithm;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_meta_learning_config() {
        let config = MetaLearningConfig {
            algorithm: MetaLearningAlgorithm::MAML,
            inner_steps: 5,
            outer_steps: 100,
            meta_learning_rate: 0.001,
            inner_learning_rate: 0.01,
            task_batch_size: 16,
            support_set_size: 10,
            query_set_size: 15,
            second_order: true,
            gradient_clip: 1.0,
            adaptation_strategies: vec![AdaptationStrategy::FullFineTuning],
            transfer_settings: TransferLearningSettings {
                domain_adaptation: true,
                source_domain_weights: vec![1.0],
                strategies: vec![TransferStrategy::FineTuning],
                similarity_measures: vec![SimilarityMeasure::CosineDistance],
                progressive_transfer: false,
            },
            continual_settings: ContinualLearningSettings {
                anti_forgetting_strategies: vec![
                    AntiForgettingStrategy::ElasticWeightConsolidation,
                ],
                memory_replay: MemoryReplaySettings {
                    buffer_size: 1000,
                    replay_strategy: ReplayStrategy::Random,
                    replay_frequency: 10,
                    selection_criteria: MemorySelectionCriteria::Random,
                },
                task_identification: TaskIdentificationMethod::Oracle,
                plasticity_stability_balance: 0.5,
            },
            multitask_settings: MultiTaskSettings {
                task_weighting: TaskWeightingStrategy::Uniform,
                gradient_balancing: GradientBalancingMethod::Uniform,
                interference_mitigation: InterferenceMitigationStrategy::OrthogonalGradients,
                shared_representation: SharedRepresentationStrategy::HardSharing,
            },
            few_shot_settings: FewShotSettings {
                num_shots: 5,
                num_ways: 5,
                algorithm: FewShotAlgorithm::MAML,
                metric_learning: MetricLearningSettings {
                    distance_metric: DistanceMetric::Euclidean,
                    embedding_dim: 64,
                    learned_metric: false,
                },
                augmentation_strategies: vec![AugmentationStrategy::Geometric],
            },
            enable_meta_regularization: true,
            meta_regularization_strength: 0.01,
            task_sampling_strategy: TaskSamplingStrategy::Uniform,
        };
        assert_eq!(config.inner_steps, 5);
        assert_eq!(config.task_batch_size, 16);
        assert!(config.second_order);
        assert!(matches!(config.algorithm, MetaLearningAlgorithm::MAML));
    }
    #[test]
    fn test_maml_config() {
        let config = MAMLConfig {
            second_order: true,
            inner_lr: 0.01f64,
            outer_lr: 0.001f64,
            inner_steps: 5,
            allow_unused: true,
            gradient_clip: Some(1.0),
        };
        assert!(config.second_order);
        assert_eq!(config.inner_steps, 5);
        assert_eq!(config.inner_lr, 0.01);
    }
}
