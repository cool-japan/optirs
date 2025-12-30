# Changelog

All notable changes to OptiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
Complete integration with SciRS2 v0.1.1:
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

#### SciRS2 Ecosystem (v0.1.1)
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

[0.1.0]: https://github.com/cool-japan/optirs/releases/tag/v0.1.0
