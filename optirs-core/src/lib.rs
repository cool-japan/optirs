//! # OptiRS Core
//!
//! Core optimization algorithms and utilities for the OptiRS library.
//!
//! ## 🚨 CRITICAL ARCHITECTURAL DEPENDENCY
//!
//! **OptiRS-Core MUST use the full SciRS2 ecosystem as its foundation.**
//! This is not optional - OptiRS builds upon SciRS2's scientific computing
//! capabilities and extends them with specialized ML optimization algorithms.
//!
//! ### Required SciRS2 Integration:
//! - `scirs2-core`: Core scientific computing primitives
//! - `scirs2-optimize` & `scirs2-optim`: Base optimization algorithms
//! - `scirs2-linalg`: Linear algebra operations for gradients/parameters
//! - `scirs2-autograd`: Automatic differentiation engine
//! - `scirs2-neural`: Neural network component integration
//! - `scirs2-metrics`: Performance monitoring and benchmarking
//! - `scirs2-stats`: Statistical functions for optimization analysis
//! - Plus all other SciRS2 crates for comprehensive functionality
//!
//! **DO NOT REMOVE OR REPLACE SciRS2 DEPENDENCIES** - OptiRS is designed
//! as an extension of the SciRS2 scientific computing ecosystem.
//!
//! ## Features
//!
//! This crate provides:
//! - Basic optimizers (SGD, Adam, AdamW, RMSprop, etc.)
//! - Learning rate schedulers
//! - Regularization techniques
//! - Parameter groups and state management
//! - Error handling and validation utilities

#![allow(deprecated)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(clippy::for_loops_over_fallibles)]
#![allow(unexpected_cfgs)]
#![allow(unused_attributes)]
#![allow(missing_docs)]

pub mod adaptive_selection;
pub mod benchmarking;
pub mod coordination;
pub mod curriculum_optimization;
pub mod distributed;
pub mod domain_specific;
pub mod error;
pub mod gradient_accumulation;
pub mod gradient_processing;
pub mod hardware_aware;
pub mod memory_efficient;
pub mod metrics;
pub mod neural_integration;
pub mod neuromorphic;
pub mod online_learning;
pub mod optimizer_composition;
pub mod optimizers;
pub mod parameter_groups;
pub mod plugin;
pub mod privacy;
pub mod regularizers;
pub mod research;
pub mod schedulers;
pub mod second_order;
pub mod self_tuning;
pub mod streaming;
pub mod training_stabilization;
pub mod unified_api;
pub mod utils;
pub mod visualization;

// Re-export commonly used types
pub use error::{OptimError, OptimizerError, Result};
pub use optimizers::*;
pub use parameter_groups::*;
pub use regularizers::*;
pub use schedulers::*;
pub use unified_api::{OptimizerConfig, OptimizerFactory, Parameter, UnifiedOptimizer};

// Re-export key functionality
pub use adaptive_selection::{
    AdaptiveOptimizerSelector, OptimizerStatistics, OptimizerType, PerformanceMetrics,
    ProblemCharacteristics, ProblemType, SelectionNetwork, SelectionStrategy,
};
pub use curriculum_optimization::{
    AdaptiveCurriculum, AdversarialAttack, AdversarialConfig, CurriculumManager, CurriculumState,
    CurriculumStrategy, ImportanceWeightingStrategy,
};
pub use distributed::{
    AveragingStrategy, CommunicationResult, CompressedGradient, CompressionStrategy,
    DistributedCoordinator, GradientCompressor, ParameterAverager, ParameterServer,
};
pub use domain_specific::{
    CrossDomainKnowledge, DomainOptimizationConfig, DomainPerformanceMetrics, DomainRecommendation,
    DomainSpecificSelector, DomainStrategy, LearningRateScheduleType, OptimizationContext,
    RecommendationType, RegularizationApproach, ResourceConstraints, TrainingConfiguration,
};
pub use gradient_accumulation::{
    AccumulationMode, GradientAccumulator as GradAccumulator, MicroBatchTrainer,
    VariableAccumulator,
};
pub use gradient_processing::*;
pub use neural_integration::architecture_aware::{
    ArchitectureAwareOptimizer, ArchitectureStrategy,
};
pub use neural_integration::forward_backward::{BackwardHook, ForwardHook, NeuralIntegration};
pub use neural_integration::{
    LayerArchitecture, LayerId, OptimizationConfig, ParamId, ParameterManager, ParameterMetadata,
    ParameterOptimizer, ParameterType,
};
pub use online_learning::{
    ColumnGrowthStrategy, LearningRateAdaptation, LifelongOptimizer, LifelongStats,
    LifelongStrategy, MemoryExample, MemoryUpdateStrategy, MirrorFunction, OnlineLearningStrategy,
    OnlineOptimizer, OnlinePerformanceMetrics, SharedKnowledge, TaskGraph,
};
pub use plugin::core::{
    create_basic_capabilities, create_plugin_info, OptimizerPluginFactory, PluginCategory,
    PluginInfo,
};
pub use plugin::sdk::{BaseOptimizerPlugin, PluginTester};
pub use plugin::{
    OptimizerPlugin, PluginCapabilities, PluginLoader, PluginRegistry, PluginValidationFramework,
};
pub use privacy::{
    AccountingMethod, ClippingStats, DifferentialPrivacyConfig, DifferentiallyPrivateOptimizer,
    MomentsAccountant, NoiseMechanism, PrivacyBudget, PrivacyValidation,
};
pub use second_order::{HessianInfo, Newton, SecondOrderOptimizer, LBFGS as SecondOrderLBFGS};
pub use self_tuning::{
    OptimizerInfo, OptimizerTrait, PerformanceStats, SelfTuningConfig, SelfTuningOptimizer,
    SelfTuningStatistics, TargetMetric,
};
pub use streaming::{
    LearningRateAdaptation as StreamingLearningRateAdaptation, StreamingConfig, StreamingDataPoint,
    StreamingHealthStatus, StreamingMetrics, StreamingOptimizer,
};
pub use training_stabilization::{AveragingMethod, ModelEnsemble, PolyakAverager, WeightAverager};
pub use visualization::{
    ColorScheme, ConvergenceInfo, DataSeries, MemoryStats as VisualizationMemoryStats,
    OptimizationMetric, OptimizationVisualizer, OptimizerComparison, PlotType, VisualizationConfig,
};

#[cfg(feature = "metrics_integration")]
pub use metrics::*;
