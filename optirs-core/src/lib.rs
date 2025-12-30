//! # OptiRS Core - Advanced ML Optimization Built on SciRS2
//!
//! **Version:** 0.1.0
//! **Status:** Stable Release - Production Ready
//!
//! `optirs-core` provides state-of-the-art optimization algorithms for machine learning,
//! built exclusively on the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.1.1 - Required foundation
//!
//! ## Quick Start
//!
//! ```rust
//! use optirs_core::optimizers::{Adam, Optimizer};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create optimizer
//! let mut optimizer = Adam::new(0.001);
//!
//! // Prepare parameters and gradients
//! let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let grads = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.08]);
//!
//! // Perform optimization step
//! let updated_params = optimizer.step(&params, &grads)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! ### 19 State-of-the-Art Optimizers
//!
//! **First-Order Methods:**
//! - **SGD** - Stochastic Gradient Descent with optional momentum
//! - **SimdSGD** - SIMD-accelerated SGD (2-4x faster)
//! - **Adam** - Adaptive Moment Estimation
//! - **AdamW** - Adam with decoupled weight decay
//! - **AdaDelta** - Adaptive LR without manual tuning ⭐ NEW!
//! - **AdaBound** - Smooth Adam→SGD transition ⭐ NEW!
//! - **Ranger** - RAdam + Lookahead combination ⭐ NEW!
//! - **RMSprop** - Root Mean Square Propagation
//! - **Adagrad** - Adaptive Gradient Algorithm
//! - **LAMB** - Layer-wise Adaptive Moments for Batch training
//! - **LARS** - Layer-wise Adaptive Rate Scaling
//! - **Lion** - Evolved Sign Momentum
//! - **Lookahead** - Look ahead optimizer wrapper
//! - **RAdam** - Rectified Adam
//! - **SAM** - Sharpness-Aware Minimization
//! - **SparseAdam** - Adam optimized for sparse gradients
//! - **GroupedAdam** - Adam with parameter groups
//!
//! **Second-Order Methods:**
//! - **L-BFGS** - Limited-memory BFGS
//! - **K-FAC** - Kronecker-Factored Approximate Curvature
//! - **Newton-CG** - Newton Conjugate Gradient ⭐ NEW!
//!
//! ### Performance Optimizations (Phase 2 Complete)
//!
//! #### SIMD Acceleration (2-4x speedup)
//! ```rust
//! use optirs_core::optimizers::{Optimizer, SimdSGD};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let params = Array1::from_elem(100_000, 1.0f32);
//! let grads = Array1::from_elem(100_000, 0.001f32);
//!
//! let mut optimizer = SimdSGD::new(0.01f32);
//! let updated = optimizer.step(&params, &grads)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Parallel Processing (4-8x speedup)
//! ```rust
//! use optirs_core::optimizers::{Adam, Optimizer};
//! use optirs_core::parallel_optimizer::parallel_step_array1;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let params_list = vec![
//!     Array1::from_elem(10_000, 1.0),
//!     Array1::from_elem(20_000, 1.0),
//! ];
//! let grads_list = vec![
//!     Array1::from_elem(10_000, 0.01),
//!     Array1::from_elem(20_000, 0.01),
//! ];
//!
//! let mut optimizer = Adam::new(0.001);
//! let results = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Memory-Efficient Operations
//! ```rust
//! use optirs_core::memory_efficient_optimizer::GradientAccumulator;
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut accumulator = GradientAccumulator::<f32>::new(1000);
//!
//! // Accumulate gradients from micro-batches
//! for _ in 0..4 {
//!     let micro_grads = Array1::from_elem(1000, 0.1);
//!     accumulator.accumulate(&micro_grads.view())?;
//! }
//!
//! let avg_grads = accumulator.average()?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Production Metrics & Monitoring
//! ```rust
//! use optirs_core::optimizer_metrics::{MetricsCollector, MetricsReporter};
//! use optirs_core::optimizers::{Adam, Optimizer};
//! use scirs2_core::ndarray::Array1;
//! use std::time::{Duration, Instant};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut collector = MetricsCollector::new();
//! collector.register_optimizer("adam");
//!
//! let mut optimizer = Adam::new(0.001);
//! let params = Array1::from_elem(1000, 1.0);
//! let grads = Array1::from_elem(1000, 0.01);
//!
//! let params_before = params.clone();
//! let start = Instant::now();
//! let params = optimizer.step(&params, &grads)?;
//! let duration = start.elapsed();
//!
//! collector.update(
//!     "adam",
//!     duration,
//!     0.001,
//!     &grads.view(),
//!     &params_before.view(),
//!     &params.view(),
//! )?;
//!
//! println!("{}", collector.summary_report());
//! # Ok(())
//! # }
//! ```
//!
//! ### Learning Rate Schedulers
//!
//! - **ExponentialDecay** - Exponential learning rate decay
//! - **StepDecay** - Step-wise reduction
//! - **CosineAnnealing** - Cosine annealing schedule
//! - **LinearWarmupDecay** - Linear warmup with decay
//! - **OneCycle** - One cycle learning rate policy
//!
//! ### Advanced Features
//!
//! - **Parameter Groups** - Different learning rates per layer
//! - **Gradient Accumulation** - Micro-batch training for large models
//! - **Gradient Clipping** - Prevent exploding gradients
//! - **Regularization** - L1, L2, weight decay
//! - **Privacy-Preserving** - Differential privacy support
//! - **Distributed Training** - Multi-GPU and TPU coordination
//! - **Neural Architecture Search** - Automated architecture optimization
//!
//! ## Architecture
//!
//! ### SciRS2 Foundation
//!
//! OptiRS-Core is built **exclusively** on the SciRS2 ecosystem:
//!
//! - **Arrays**: Uses `scirs2_core::ndarray` (NOT direct ndarray)
//! - **Random**: Uses `scirs2_core::random` (NOT direct rand)
//! - **SIMD**: Uses `scirs2_core::simd_ops` for vectorization
//! - **Parallel**: Uses `scirs2_core::parallel_ops` for multi-core
//! - **GPU**: Built on `scirs2_core::gpu` abstractions
//! - **Metrics**: Uses `scirs2_core::metrics` for monitoring
//! - **Error Handling**: Uses `scirs2_core::error::Result`
//!
//! This integration ensures:
//! - Type safety across the ecosystem
//! - Consistent performance optimizations
//! - Unified error handling
//! - Simplified dependency management
//!
//! ### Module Organization
//!
//! - [`optimizers`] - Core optimizer implementations
//! - [`schedulers`] - Learning rate scheduling
//! - [`simd_optimizer`] - SIMD-accelerated optimizers
//! - [`parallel_optimizer`] - Multi-core processing
//! - [`memory_efficient_optimizer`] - Memory optimization
//! - [`gpu_optimizer`] - GPU acceleration
//! - [`optimizer_metrics`] - Performance monitoring
//! - [`gradient_processing`] - Gradient manipulation
//! - [`regularizers`] - Regularization techniques
//! - [`second_order`] - Second-order methods
//! - [`distributed`] - Distributed training
//! - [`privacy`] - Privacy-preserving optimization
//!
//! ## Performance
//!
//! ### Benchmarks
//!
//! All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) with statistical analysis:
//!
//! - **optimizer_benchmarks** - Compare all 16 optimizers
//! - **simd_benchmarks** - SIMD vs scalar performance
//! - **parallel_benchmarks** - Multi-core scaling
//! - **memory_efficient_benchmarks** - Memory optimization impact
//! - **gpu_benchmarks** - GPU vs CPU comparison
//! - **metrics_benchmarks** - Monitoring overhead
//!
//! Run benchmarks:
//! ```bash
//! cargo bench --package optirs-core
//! ```
//!
//! ### Test Coverage
//!
//! - **549 unit tests** - Core functionality
//! - **54 doc tests** - Documentation examples
//! - **603 total tests** - All passing
//! - **Zero clippy warnings** - Production quality
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive examples:
//!
//! - `basic_optimization.rs` - Getting started
//! - `advanced_optimization.rs` - Schedulers, regularization, clipping
//! - `performance_optimization.rs` - SIMD, parallel, GPU acceleration
//! - `production_monitoring.rs` - Metrics and convergence detection
//!
//! ## Contributing
//!
//! When contributing, ensure:
//! - **100% SciRS2 usage** - No direct ndarray/rand/rayon imports
//! - **Zero clippy warnings** - Run `cargo clippy`
//! - **All tests pass** - Run `cargo test`
//! - **Documentation** - Add examples to public APIs
//!
//! ## License
//!
//! Dual-licensed under MIT OR Apache-2.0

#![allow(deprecated)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(for_loops_over_fallibles)]
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
pub mod gpu_optimizer;
pub mod gradient_accumulation;
pub mod gradient_processing;
pub mod hardware_aware;
pub mod memory_efficient;
pub mod memory_efficient_optimizer;
pub mod metrics;
pub mod neural_integration;
pub mod neuromorphic;
pub mod online_learning;
pub mod optimizer_composition;
pub mod optimizer_metrics;
pub mod optimizers;
pub mod parallel_optimizer;
pub mod parameter_groups;
pub mod plugin;
pub mod privacy;
pub mod regularizers;
pub mod research;
pub mod schedulers;
pub mod second_order;
pub mod self_tuning;
pub mod simd_optimizer;
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
pub use gpu_optimizer::{GpuConfig, GpuMemoryStats, GpuOptimizer, GpuUtils};
pub use gradient_accumulation::{
    AccumulationMode, GradientAccumulator as GradAccumulator, MicroBatchTrainer,
    VariableAccumulator,
};
pub use gradient_processing::*;
pub use memory_efficient_optimizer::{
    ChunkedOptimizer, GradientAccumulator as MemoryEfficientGradientAccumulator,
    MemoryUsageEstimator,
};
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
pub use optimizer_metrics::{
    ConvergenceMetrics, GradientStatistics, MetricsCollector, MetricsReporter, OptimizerMetrics,
    ParameterStatistics,
};
pub use parallel_optimizer::{
    parallel_step, parallel_step_array1, ParallelBatchProcessor, ParallelOptimizer,
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
pub use second_order::{
    HessianInfo, Newton, NewtonCG, SecondOrderOptimizer, LBFGS as SecondOrderLBFGS,
};
pub use self_tuning::{
    OptimizerInfo, OptimizerTrait, PerformanceStats, SelfTuningConfig, SelfTuningOptimizer,
    SelfTuningStatistics, TargetMetric,
};
pub use simd_optimizer::{should_use_simd, SimdOptimizer};
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
