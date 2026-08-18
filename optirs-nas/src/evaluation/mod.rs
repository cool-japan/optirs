//! Performance evaluation system for neural architecture search
//!
//! Provides comprehensive evaluation metrics, benchmarking suites,
//! and performance prediction capabilities for optimizer architectures.

mod benchmark;
mod cache;
mod evaluator;
mod predictor;
mod resource;
mod statistical;
mod types;

// Re-export main types
pub use benchmark::{
    BenchmarkMetadata, BenchmarkResults, BenchmarkSuite, StandardBenchmark, TestFunction,
    TestResult,
};
pub use cache::{CacheMetadata, CachedEvaluation, EvaluationCache};
pub use evaluator::PerformanceEvaluator;
// The predictor's public surface is the predictor itself plus the three types its
// state is made of. The two dozen names that used to be listed here
// (`FeatureExtractor`, `PredictionCache`, `LearningRateSchedule`, `DataSplits`, ...)
// were structs nobody could construct — all fields private, all constructors private
// — and that no code path read; see `predictor::PerformancePredictor` for the list.
pub use predictor::{
    ModelParameters, ModelTrainingState, PerformancePredictor, PredictorModel,
    PredictorTrainingData,
};
pub use resource::{MonitoringConfig, ResourceLimits, ResourceMonitor, ResourceUsageSnapshot};
pub use statistical::{DescriptiveStats, StatisticalAnalyzer, StatisticalTest};
pub use types::{
    ActivationFunction, AnalysisMethod, BenchmarkType, CacheEvictionPolicy, DifficultyLevel,
    MultipleComparisonCorrection, PerformanceRanking, PredictorModelType, ResourceRequirements,
    ResourceSummary, StatisticalSummary, StatisticalTestType, TestFunctionType,
};
