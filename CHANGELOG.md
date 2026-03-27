# Changelog

All notable changes to OptiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-03-27

### Changed

- **Version Synchronization** - All lib.rs doc comments now reflect accurate version 0.3.1
- **Documentation** - Updated version references throughout codebase

### Added

- **optirs-wasm README** - Added comprehensive README.md for WebAssembly bindings crate

### Fixed

- **Doc Comment Versions** - Fixed stale 0.1.0 version strings in all crate documentation

---

## [0.3.0] - 2026-03-17

### Added

#### New Optimizers (optirs-core)
- **ReptileOptimizer** - Meta-learning optimizer with inner loop SGD and parameter interpolation
- **MetaSGD** - Per-parameter learnable learning rates with meta-gradient updates

#### New Regularizers (optirs-core)
- **GroupLasso** - Group-wise L2 norm penalty for structured sparsity
- **StructuredSparsity** - Column, row, and block sparsity patterns

#### Optimizer Composition (optirs-core)
- **WeightedOptimizer** - Weighted average of multiple optimizer outputs

#### Continual Learning (optirs-learned)
- **ElasticWeightConsolidation** - Diagonal Fisher information with parameter anchoring (standard + online EWC)
- **ProgressiveNetworks** - Column networks per task with lateral connections

#### Domain-Specific Optimizers (optirs-learned)
- **CVOptimizer** - Spatial-aware LR, channel normalization, progressive resolution
- **NLPOptimizer** - Layer-wise LR decay, warmup/cosine schedule, gradient accumulation
- **AttentionOptimizer** - Head-wise gradient scaling, attention entropy regularization

#### Meta-Learning (optirs-learned)
- **ReptileLearner** - First-order meta-learning with task averaging
- **MetaSGDLearner** - Meta-learning with per-parameter learnable learning rates
- Fixed MAML `compute_gradients` - Now uses finite-difference approximation instead of returning zeros

#### Neural Architecture Search (optirs-nas)
- **MemoryEfficientDARTS** - Partial channel connections and edge normalization (PC-DARTS)
- **RobustDARTS** - Perturbation regularization, collapse detection, early stopping
- **DARTSConfig** - Builder pattern for DARTS variants with temperature scheduling
- **ProgressiveNAS** - Phase-based architecture search with increasing complexity budgets
- **DomainNASEngine** - Domain-specific NAS with pre-configured search spaces for CV, NLP, TimeSeries
- **ArchitectureEmbedder** - Architecture embedding into vector space with cosine similarity search

#### Distributed Training (optirs-core) — Wave 2
- **FedProxOptimizer** - Federated proximal optimizer with configurable mu coefficient (mu=0 degenerates to FedAvg)

#### Learning Rate Schedulers (optirs-core) — Wave 2
- **ViTLayerDecay** - Vision Transformer layer-wise exponential LR decay with warmup + cosine schedule
- **AttentionAwareScheduler** - Transformer component-specific LR scaling (Attention, FeedForward, Embedding, LayerNorm, Output)

#### Gradient & Loss Analysis (optirs-core) — Wave 2
- **GradientFlowAnalyzer** - Gradient flow recording, vanishing/exploding detection, health reports, SVG visualization
- **LossLandscapeAnalyzer** - 2D loss landscape perturbation analysis, sharpness computation, saddle point detection, SVG contour plots

#### Few-Shot Learning (optirs-learned) — Wave 2
- **PrototypicalNetwork** - Encoding, prototype computation, nearest-prototype classification
- **FastAdaptationEngine** - Multi-step gradient adaptation with strategy selection
- **TaskSimilarityCalculator** - Cosine similarity for task representations
- **EpisodicMemoryBank** - Episode storage/retrieval with eviction policies (Performance, LRU, LFU, Age)
- **SupportSetManager** - Diverse support set selection, noise augmentation, quality evaluation

#### Online Meta-Learning (optirs-learned) — Wave 2
- **OnlineMAML** - Continuous task stream meta-learning with staleness decay and buffer management
- **CrossDomainTransfer** - Cross-domain knowledge transfer with shared representations and transferability matrix

### Refactored

- **search_strategies.rs** (2052 lines) split into modular directory structure:
  - `search_strategies/mod.rs` - Trait definitions and re-exports
  - `search_strategies/random.rs` - RandomSearch
  - `search_strategies/evolutionary.rs` - EvolutionarySearch
  - `search_strategies/rl_search.rs` - ReinforcementLearningSearch
  - `search_strategies/differentiable.rs` - DifferentiableSearch (DARTS) + variants
  - `search_strategies/bayesian.rs` - BayesianOptimization
  - `search_strategies/neural_predictor.rs` - NeuralPredictorSearch
  - `search_strategies/progressive.rs` - ProgressiveNAS

### Changed

#### Workspace & Dependencies
- **Bumped workspace version to 0.3.0** - All crates now at version 0.3.0
- Updated all internal crate dependency versions to 0.3.0
- Updated publish script to version 0.3.0

#### Documentation
- Updated TODO.md version references to 0.3.0
- Updated CHANGELOG with 0.3.0 release notes

---

## [0.2.0] - 2026-02-16

### Changed

#### Licensing
- **License changed to Apache-2.0 only** - OptiRS now uses Apache-2.0 as its sole license, aligning with the broader SciRS2 ecosystem
- Consolidated license files into single `LICENSE` file
- Updated all crate manifests and documentation to reflect Apache-2.0 license

#### SciRS2 Integration
- **Updated to SciRS2 v0.3.0** - Upgraded all SciRS2 dependencies from v0.2.0 to v0.3.0
  - `scirs2-core` 0.3.0 - Core scientific computing primitives
  - `scirs2-optimize` 0.3.0 - Base optimization interfaces
  - `scirs2-neural` 0.3.0 - Neural network support
  - `scirs2-metrics` 0.3.0 - Performance monitoring
  - `scirs2-stats` 0.3.0 - Statistical analysis
  - `scirs2-series` 0.3.0 - Time series support
  - `scirs2-datasets` 0.3.0 - Dataset utilities
  - `scirs2-linalg` 0.3.0 - Linear algebra operations
  - `scirs2-signal` 0.3.0 - Signal processing

#### Documentation
- Moved `CLAUDE.md` from repository to temporary directory (per project policy)
- Updated version references across all documentation to v0.3.0
- Updated `MIGRATION_FROM_SCIRS2.md` to reflect new version
- Updated all subcrate documentation headers

#### Build System
- Updated workspace version to 0.2.0 across all crates
- Refined workspace dependency management for SciRS2 v0.3.0
- Updated `optirs-gpu` dependency configuration

### Removed
- Old dual-license files (replaced by single `LICENSE`)
- `LICENSE-APACHE` file (consolidated into `LICENSE`)
- `CLAUDE.md` from repository (moved to temporary directory per policy)

### Maintenance
- Minor version updates to library headers
- Improved consistency across crate versions
- Streamlined license file structure

### Migration Notes
For users upgrading from 0.1.0 to 0.2.0:
- **License Change**: OptiRS is now Apache-2.0 only. Please review your legal requirements if needed
- **SciRS2 Dependencies**: Update all SciRS2 dependencies to v0.3.0
- **No API Changes**: The public API remains fully compatible with v0.1.0
- **No Breaking Changes**: This release is backward compatible at the API level

### Dependencies
All SciRS2 dependencies updated to v0.3.0:
```toml
scirs2-core = "0.3.0"
scirs2-optimize = "0.3.0"
scirs2-neural = "0.3.0"
scirs2-metrics = "0.3.0"
scirs2-stats = "0.3.0"
scirs2-series = "0.3.0"
scirs2-datasets = "0.3.0"
scirs2-linalg = "0.3.0"
scirs2-signal = "0.3.0"
```

---

## [0.1.0] - 2025-12-30

### 🎉 Initial Release

OptiRS v0.1.0 is the first release of a comprehensive ML optimization library for Rust, built exclusively on the SciRS2 scientific computing ecosystem. This release provides 19 production-ready optimizers, comprehensive documentation, full SciRS2 integration, and zero clippy warnings.

### Added

#### Core Optimizers (19 total)

**First-Order Optimizers (17)**
- **SGD** - Stochastic Gradient Descent with momentum and Nesterov acceleration
- **SimdSGD** - SIMD-accelerated SGD (2-4x faster for large arrays)
- **Adam** - Adaptive Moment Estimation
- **AdamW** - Adam with decoupled weight decay
- **AdaDelta** - Adaptive learning rate without manual tuning
- **AdaBound** - Dynamic bounds smoothly transitioning from Adam to SGD
- **RMSprop** - Root Mean Square Propagation
- **Adagrad** - Adaptive Gradient Algorithm
- **LAMB** - Layer-wise Adaptive Moments for batch training
- **LARS** - Layer-wise Adaptive Rate Scaling
- **Lion** - Evolved Sign Momentum optimizer
- **Lookahead** - k steps forward, 1 step back wrapper
- **RAdam** - Rectified Adam with variance rectification
- **Ranger** - RAdam + Lookahead combination
- **SAM** - Sharpness-Aware Minimization
- **SparseAdam** - Adam optimized for sparse gradients
- **GroupedAdam** - Adam with parameter groups

**Second-Order Optimizers (2)**
- **L-BFGS** - Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- **Newton-CG** - Newton Conjugate Gradient with trust region

#### Learning Rate Schedulers
- **ExponentialDecay** - Exponential learning rate decay
- **StepDecay** - Step-wise reduction
- **CosineAnnealing** - Cosine annealing schedule
- **LinearWarmup** - Linear warmup with decay
- **OneCycle** - One cycle learning rate policy

#### Performance Features
- **SIMD Acceleration** - 2-4x speedup for large parameter arrays
  - Automatic SIMD vectorization for f32/f64
  - Threshold-based activation (16 elements for f32, 8 for f64)

- **Parallel Processing** - 4-8x speedup for parameter groups
  - Multi-core parameter group processing
  - Automatic work distribution across CPU cores

- **Memory-Efficient Operations**
  - Gradient accumulation for micro-batch training
  - Chunked parameter processing for billion-parameter models
  - Memory usage estimation and recommendations

- **GPU Framework** - Multi-backend support foundation
  - CUDA, Metal, OpenCL, WebGPU backends
  - GPU context management and initialization
  - Tensor cores and mixed-precision support

#### Production Tools
- **Metrics & Monitoring**
  - Real-time optimizer performance tracking
  - Gradient statistics (mean, std dev, norm, sparsity)
  - Parameter statistics (update magnitude, relative change)
  - Convergence detection with moving averages
  - Export to JSON and CSV formats

- **Comprehensive Benchmarking**
  - Statistical performance analysis with Criterion.rs
  - Memory profiling and leak detection
  - Cross-platform performance testing
  - Regression detection

#### Module Organization
- `optirs-core` - Core optimization algorithms (Production Ready)
- `optirs-bench` - Benchmarking and performance analysis (Production Ready)
- `optirs-gpu` - GPU acceleration framework (In Development)
- `optirs-tpu` - TPU coordination (Framework Ready)
- `optirs-learned` - Learned optimizers and meta-learning (Research Phase)
- `optirs-nas` - Neural Architecture Search (Research Phase)

### Features

#### SciRS2 Ecosystem Integration
Complete integration with SciRS2 v0.3.0:
- ✅ Arrays: `scirs2_core::ndarray` exclusively (NO direct ndarray)
- ✅ Random: `scirs2_core::random` exclusively (NO direct rand)
- ✅ Numerical: `scirs2_core::numeric` for all numerical traits
- ✅ SIMD: `scirs2_core::simd_ops` for vectorization
- ✅ Parallel: `scirs2_core::parallel_ops` for multi-core
- ✅ GPU: `scirs2_core::gpu` abstractions
- ✅ Metrics: `scirs2_core::metrics` for monitoring

#### Quality Assurance
- **1,134 tests passing** (100% pass rate)
  - 1,061 unit tests
  - 73 doc tests
- **Zero clippy warnings** - Production-ready code quality
- **100% public API documentation**
- **Comprehensive examples** - All features demonstrated

### Performance

#### Benchmarks
- **SGD**: < 10ns per parameter update
- **Adam**: < 50ns per parameter update
- **SIMD variants**: 2-4x faster on large arrays
- **Parallel processing**: 4-8x speedup on multi-core

#### Memory Efficiency
- Optimizer state: < 2x parameter memory
- Zero-copy operations where possible
- Efficient gradient accumulation
- Chunked processing for large models

### Documentation

#### Comprehensive Documentation
- **README.md** - Project overview and quick start
- **USAGE_GUIDE.md** - Comprehensive usage guide (8000+ words)
- **MIGRATION_FROM_SCIRS2.md** - Migration guide for SciRS2 users
- **SCIRS2_INTEGRATION_POLICY.md** - Critical dependency policy
- **API Documentation** - 100% coverage with examples
- **Examples** - 4 comprehensive example files

### Dependencies

#### SciRS2 Ecosystem (v0.3.0)
- `scirs2-core` - Core scientific primitives (REQUIRED)
- `scirs2-optimize` - Base optimization interfaces (REQUIRED)
- `scirs2-neural` - Neural network support
- `scirs2-metrics` - Performance monitoring
- `scirs2-stats` - Statistical analysis
- `scirs2-series` - Time series support
- `scirs2-datasets` - Dataset utilities

#### External Dependencies
- Serialization: `serde`, `serde_json`, `oxicode`
- Testing: `approx`, `criterion`, `tokio-test`
- GPU: `cudarc`, `opencl3`, `wgpu`
- ML: `tokenizers`, `autograd`
- Utilities: `thiserror`, `anyhow`, `chrono`

### Policy Compliance

✅ **COOLJAPAN Policy**
- Using `oxiblas` via scirs2-core (NO openblas)

✅ **Latest Crates Policy**
- All dependencies at latest compatible versions

✅ **Workspace Policy**
- Version managed via `workspace = true`

✅ **No Warnings Policy**
- Zero clippy warnings
- Clean compilation with all features

### Platform Support

| Platform | Core | GPU | TPU | Learned | NAS | Bench |
|----------|------|-----|-----|---------|-----|-------|
| Linux    | ✅   | 🚧  | 🚧  | 🔬      | 🔬  | ✅    |
| macOS    | ✅   | 🚧  | ❌  | 🔬      | 🔬  | ✅    |
| Windows  | ✅   | 🚧  | ❌  | 🔬      | 🔬  | ✅    |

Legend: ✅ Production Ready | 🚧 In Development | 🔬 Research Phase | ❌ Not Supported

### Contributors

- COOLJAPAN OU (Team Kitasan)
- SciRS2 Integration Team
- Rust ML Community

### Acknowledgments

- SciRS2 project for providing robust scientific computing foundation
- Rust ML community for feedback and support
- Open source optimization algorithm researchers

---

[0.2.0]: https://github.com/cool-japan/optirs/releases/tag/v0.2.0
[0.1.0]: https://github.com/cool-japan/optirs/releases/tag/v0.1.0
