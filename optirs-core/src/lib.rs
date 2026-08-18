//! # OptiRS Core - Advanced ML Optimization Built on SciRS2
//!
//! **Version:** 0.3.2
//! **Status:** Pre-1.0 (0.3.x) - the public API may still change between 0.x releases
//!
//! `optirs-core` provides state-of-the-art optimization algorithms for machine learning,
//! built exclusively on the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.
//!
//! ## Dependencies
//!
//! - `scirs2-core` 0.6.5, `scirs2-optimize` 0.6.5 - Required foundation
//! - `scirs2-neural`, `scirs2-stats` - Unconditional dependencies, pulled in for specific
//!   modules (e.g. `neuromorphic`, distribution-based regularizers) but not behind a feature
//! - `scirs2-metrics` - Optional, behind the `metrics-integration` feature
//! - `scirs2-datasets` - Optional, behind the `cross-platform-testing` feature
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
//! ### 26 Optimizers
//!
//! 22 types implement the [`optimizers::Optimizer`] trait (`optirs_core::optimizers`, listed
//! below). 4 more live in [`second_order`] (`optirs_core::second_order`) as a separate family
//! not reachable through `Optimizer`: 2 implement [`second_order::SecondOrderOptimizer`]
//! (Newton, and a second, independent L-BFGS implementation re-exported as `SecondOrderLBFGS`
//! to avoid colliding with `optimizers::LBFGS`), and 2 expose their own inherent
//! `step`/`step_with_loss` methods instead of a shared trait (NewtonCG, KFAC).
//!
//! **First-Order Methods (17):**
//! - **SGD** - Stochastic Gradient Descent with optional momentum
//! - **SimdSGD** - SIMD-accelerated SGD
//! - **Adam** - Adaptive Moment Estimation
//! - **AdamW** - Adam with decoupled weight decay
//! - **AdaDelta** - Adaptive LR without manual tuning
//! - **AdaBound** - Smooth Adam→SGD transition
//! - **Ranger** - RAdam + Lookahead combination
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
//! **Quasi-Newton (1):**
//! - **LBFGS** (`optimizers::LBFGS`) - Limited-memory BFGS with two-loop recursion
//!
//! **Meta-Learning Optimizers (4)** - these also implement [`optimizers::Optimizer`], so they
//! drop into the same training loop as any other entry above:
//! - **MAML** - Model-Agnostic Meta-Learning (SecondOrder/FirstOrder/Reptile variants)
//! - **MetaSGD** - Meta-learned per-parameter learning rates
//! - **ReptileOptimizer** - First-order meta-learning
//! - **NtmOptimizer** - Neural Turing Machine-style memory-augmented optimizer
//!
//! **`second_order` module (4)** - not part of `optimizers::Optimizer`:
//! - **Newton** (`second_order::Newton`) - implements `SecondOrderOptimizer`; diagonal Newton
//!   step with curvature flooring
//! - **SecondOrderLBFGS** (`second_order::LBFGS`) - implements `SecondOrderOptimizer`; a
//!   separate, simpler L-BFGS implementation from `optimizers::LBFGS` above
//! - **NewtonCG** - own `step`/`step_with_loss` API; Newton Conjugate Gradient with
//!   trust-region control
//! - **KFAC** - own `step` API; Kronecker-Factored Approximate Curvature
//!
//! ### Performance Optimizations
//!
//! #### SIMD Acceleration
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
//! #### Parallel Processing
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
//! ### Learning Rate Schedulers (`optirs_core::schedulers`)
//!
//! - **ConstantScheduler**, **ExponentialDecay**, **StepDecay**, **LinearDecay** - basic decays
//! - **CosineAnnealing**, **CosineAnnealingWarmRestarts** - cosine schedules
//! - **LinearWarmupDecay**, **OneCycle**, **CyclicLR** - warmup / cyclic policies
//! - **ReduceOnPlateau** - metric-driven LR reduction
//! - **CurriculumScheduler** - staged curriculum transitions
//! - **NoiseInjectionScheduler** - stochastic LR perturbation
//! - **AttentionAwareScheduler** - component-specific LR scaling for Transformer models
//! - **ViTLayerDecay** - per-layer exponential LR decay for Vision Transformers
//! - **CustomScheduler** / **CombinedScheduler** / **SchedulerBuilder** - compose schedules
//!   from closures
//!
//! ### Advanced Features
//!
//! - **Parameter Groups** - Different learning rates per layer
//! - **Gradient Accumulation** - Micro-batch training for large models
//! - **Gradient Clipping** - Prevent exploding gradients
//! - **Regularization** - L1, L2, weight decay
//! - **Privacy-Preserving** - Differential privacy support (Rényi DP accountant, secure
//!   aggregation)
//! - **Distributed Training** - Parameter averaging, ring all-reduce/all-gather, pipeline
//!   parallelism (GPipe/1F1B), elastic (join/leave) training. Device-level GPU/TPU
//!   coordination lives in the separate `optirs-gpu` / `optirs-tpu` crates.
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
//! - **optimizer_benchmarks** - Compare optimizer implementations
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
//! - **2202 tests** - library + integration tests (`cargo nextest run -p optirs-core --all-features`)
//! - **95 doc tests** - Documentation examples
//! - **Zero clippy warnings** - `cargo clippy -p optirs-core --all-features --all-targets`
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive examples:
//!
//! - `sgd_example.rs` - Getting started
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
//! licensed under Apache-2.0

pub mod adaptive_selection;
pub mod benchmarking;
#[cfg(not(target_arch = "wasm32"))]
pub mod coordination;
pub mod curriculum_optimization;
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
#[cfg(not(target_arch = "wasm32"))]
pub mod domain_specific;
pub mod error;
pub mod gpu_optimizer;
pub mod gradient_accumulation;
pub mod gradient_flow;
pub mod gradient_processing;
#[cfg(not(target_arch = "wasm32"))]
pub mod hardware_aware;
pub mod loss_landscape;
#[cfg(not(target_arch = "wasm32"))]
pub mod memory_efficient;
pub mod memory_efficient_optimizer;
pub mod metrics;
pub mod neural_integration;
#[cfg(not(target_arch = "wasm32"))]
pub mod neuromorphic;
pub mod online_learning;
pub mod optimizer_composition;
pub mod optimizer_metrics;
pub mod optimizers;
#[cfg(not(target_arch = "wasm32"))]
pub mod parallel_optimizer;
pub mod parameter_groups;
#[cfg(not(target_arch = "wasm32"))]
pub mod plugin;
#[cfg(not(target_arch = "wasm32"))]
pub mod privacy;
pub mod quantum_inspired;
pub mod regularizers;
pub mod reinforcement_learning;
#[cfg(not(target_arch = "wasm32"))]
pub mod research;
pub mod schedulers;
pub mod second_order;
pub mod self_tuning;
pub mod sensitivity_analysis;
pub mod simd_optimizer;
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
pub use distributed::{
    AveragingStrategy, CommunicationResult, CompressedGradient, CompressionStrategy,
    DistributedCoordinator, GradientCompressor, ParameterAverager, ParameterServer,
};
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
pub use parallel_optimizer::{
    parallel_step, parallel_step_array1, ParallelBatchProcessor, ParallelOptimizer,
};
#[cfg(not(target_arch = "wasm32"))]
pub use plugin::core::{
    create_basic_capabilities, create_plugin_info, OptimizerPluginFactory, PluginCategory,
    PluginInfo,
};
#[cfg(not(target_arch = "wasm32"))]
pub use plugin::sdk::BaseOptimizerPlugin;
#[cfg(not(target_arch = "wasm32"))]
pub use plugin::{
    OptimizerPlugin, PluginCapabilities, PluginLoader, PluginRegistry, PluginValidationFramework,
};
#[cfg(not(target_arch = "wasm32"))]
pub use privacy::{
    AccountingMethod, ClippingStats, DifferentialPrivacyConfig, DifferentiallyPrivateOptimizer,
    MomentsAccountant, NoiseMechanism, PrivacyBudget, PrivacyValidation,
};
pub use quantum_inspired::{
    HybridQuantumClassical, OptimizationPhase, QuantumAnnealing, QuantumOptimizerConfig,
    VariationalQuantumOptimizer,
};
pub use second_order::{
    HessianInfo, Newton, NewtonCG, SecondOrderOptimizer, LBFGS as SecondOrderLBFGS,
};
pub use self_tuning::{
    OptimizerInfo, OptimizerTrait, PerformanceStats, SelfTuningConfig, SelfTuningOptimizer,
    SelfTuningStatistics, TargetMetric,
};
pub use sensitivity_analysis::{
    MorrisAnalyzer, MorrisIndices, OatAnalyzer, OatResult, SensitivityAnalyzer, SensitivityIndices,
    SobolAnalyzer,
};
pub use simd_optimizer::{should_use_simd, SimdOptimizer};
#[cfg(not(target_arch = "wasm32"))]
pub use streaming::{
    LearningRateAdaptation as StreamingLearningRateAdaptation, StreamingConfig, StreamingDataPoint,
    StreamingHealthStatus, StreamingMetrics, StreamingOptimizer,
};
pub use training_stabilization::{AveragingMethod, ModelEnsemble, PolyakAverager, WeightAverager};
pub use visualization::{
    ColorScheme, ConvergenceInfo, DataSeries, MemoryStats as VisualizationMemoryStats,
    OptimizationMetric, OptimizationVisualizer, OptimizerComparison, PlotType, VisualizationConfig,
};

#[cfg(feature = "metrics-integration")]
pub use metrics::*;
