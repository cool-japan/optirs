// Training components for transformer-based learned optimizers
//
// This module contains the training infrastructure for transformer optimizers,
// including meta-learning, curriculum learning, and evaluation frameworks.

pub mod curriculum;
pub mod evaluation;
pub mod meta_learning;

// Re-export key types for convenience
pub use curriculum::{
    CurriculumLearner, CurriculumParams, CurriculumState, CurriculumStrategy, LearningPhase,
    LearningProgressTracker, TaskDifficultyEstimator, TaskScheduler,
};
pub use evaluation::{
    AggregationMethod, ConvergenceInfo, EfficiencyMetrics, EvaluationResult, EvaluationStrategy,
    RobustnessTestSuite, StatisticalSignificance, TransformerEvaluator,
};
pub use meta_learning::{
    ContinualLearningState, DomainAdapter, DomainInfo, DomainType, FewShotLearner, MetaEventType,
    MetaLearningStrategy, MetaPerformanceMetrics, MetaTrainingEvent, TaskCharacteristics, TaskInfo,
    TransformerMetaLearner,
};
