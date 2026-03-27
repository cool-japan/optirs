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
    BenchmarkMetadata, BenchmarkResults, BenchmarkSuite, CustomBenchmark, CustomBenchmarkConfig,
    CustomEvaluator, ProblemDefinition, StandardBenchmark, TestFunction, TestResult,
};
pub use cache::{CacheMetadata, CachedEvaluation, EvaluationCache};
pub use evaluator::PerformanceEvaluator;
pub use predictor::{
    CacheConfig, CacheStatistics, CalibrationCurve, CalibrationData, DataSplits,
    EarlyStoppingState, FeatureCache, FeatureEngineeringPipeline, FeatureExtractor,
    FeatureInteractions, FeatureScaling, FeatureSelection, LearningRateSchedule, ModelArchitecture,
    ModelParameters, ModelTrainingState, PerformancePredictor, PolynomialFeatures, PredictionCache,
    PredictionResult, PredictorModel, PredictorTrainingData, RegularizationParameters,
    ResourceUsageRecord, TrainingMetadata, UncertaintyEstimator, UncertaintyParameters,
};
pub use resource::{MonitoringConfig, ResourceLimits, ResourceMonitor, ResourceUsageSnapshot};
pub use statistical::{DescriptiveStats, StatisticalAnalyzer, StatisticalTest};
pub use types::{
    ActivationFunction, AnalysisMethod, BenchmarkType, CacheEvictionPolicy, CorrelationStructure,
    DataCharacteristics, DataFormat, DifficultyLevel, DistributionType, EarlyStoppingCriteria,
    EvaluationCriterion, EvaluatorType, FeatureExtractionMethod, FeatureSelectionMethod,
    IOSpecification, MetricType, MultipleComparisonCorrection, NormalizationMethod,
    PerformanceRanking, PredictorModelType, ProblemType, ResourceRequirements, ResourceSummary,
    ScalingMethod, ScheduleType, StatisticalSummary, StatisticalTestType, SuccessMetrics,
    TemporalPatternType, TerminationConditions, TestFunctionType, UncertaintyEstimationMethod,
    ValidationCriteria,
};
