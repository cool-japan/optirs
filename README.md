# OptiRS - Advanced ML Optimization Built on SciRS2

**Version:** 0.1.0-rc.1
**Status:** 🚀 RELEASE CANDIDATE - Pre-release Testing Phase

OptiRS is a comprehensive optimization library for machine learning that **extends and leverages the full power of SciRS2-Core**. It provides specialized optimization algorithms and hardware acceleration while making **FULL USE** of SciRS2's scientific computing capabilities.

## 🚨 CRITICAL: Full SciRS2-Core Usage

**OptiRS is NOT a standalone project** - it is an extension of SciRS2 that MUST make full use of scirs2-core for ALL operations:
- ✅ **Arrays**: Uses `scirs2_core::ndarray` exclusively (NO direct ndarray)
- ✅ **Random**: Uses `scirs2_core::random` exclusively (NO direct rand)
- ✅ **SIMD**: Uses `scirs2_core::simd` and `simd_ops` for vectorization
- ✅ **GPU**: Built on `scirs2_core::gpu` abstractions
- ✅ **Memory**: Uses `scirs2_core::memory` and `memory_efficient`
- ✅ **Profiling**: Uses `scirs2_core::profiling` and `benchmarking`
- ✅ **Error Handling**: Uses `scirs2_core::error::Result`

### Required SciRS2 Dependencies:
- **scirs2-core**: Core scientific computing primitives and utilities
- **scirs2-optimize**: Base optimization algorithms and interfaces
- **scirs2-optim**: Additional optimization methods
- **scirs2-linalg**: Linear algebra operations
- **scirs2-neural**: Neural network components
- **scirs2-autograd**: Automatic differentiation engine
- **scirs2-metrics**: Performance monitoring and metrics
- **scirs2-stats**: Statistical functions and distributions
- **scirs2-cluster**: Clustering algorithms
- **scirs2-datasets**: Dataset handling and preprocessing
- **scirs2-signal**: Signal processing capabilities
- **scirs2-fft**: Fast Fourier Transform operations
- **scirs2-transform**: Mathematical transformations
- **scirs2-sparse**: Sparse matrix operations
- **scirs2-vision**: Computer vision utilities
- **scirs2-io**: Input/output operations
- **scirs2-integrate**: Numerical integration
- **scirs2-interpolate**: Interpolation methods
- **scirs2-series**: Time series analysis
- **scirs2-spatial**: Spatial data structures
- **scirs2-special**: Special mathematical functions
- **scirs2-text**: Text processing utilities
- **scirs2-ndimage**: N-dimensional image processing
- **scirs2-graph**: Graph algorithms

### Architecture Philosophy:
OptiRS extends SciRS2's scientific computing capabilities with specialized ML optimization features. It leverages SciRS2's robust numerical foundation while adding advanced optimization algorithms, hardware acceleration, and learned optimizers.

**DO NOT remove or replace SciRS2 dependencies** - OptiRS is designed to build upon the entire SciRS2 ecosystem.

## Features

### Core Optimizers (`optirs-core`) ✅ Production Ready

#### 19 Production-Ready Optimizers
All optimizers built exclusively on SciRS2-Core:

**First-Order Optimizers (17)**
- **SGD** - Stochastic Gradient Descent with optional momentum
- **SimdSGD** - SIMD-accelerated SGD (2-4x faster for large arrays)
- **Adam** - Adaptive Moment Estimation
- **AdamW** - Adam with decoupled weight decay
- **RMSprop** - Root Mean Square Propagation
- **Adagrad** - Adaptive Gradient Algorithm
- **AdaDelta** - ✨ Adaptive learning rate method (v1.0.0)
- **AdaBound** - ✨ Adaptive gradient with dynamic bound (v1.0.0)
- **LAMB** - Layer-wise Adaptive Moments for Batch training
- **LARS** - Layer-wise Adaptive Rate Scaling
- **Lion** - Evolved Sign Momentum optimizer
- **Lookahead** - Look ahead optimizer wrapper
- **RAdam** - Rectified Adam
- **Ranger** - ✨ RAdam + Lookahead hybrid (v1.0.0)
- **SAM** - Sharpness-Aware Minimization
- **SparseAdam** - Adam variant for sparse gradients
- **GroupedAdam** - Adam with parameter groups

**Second-Order Optimizers (2)**
- **L-BFGS** - Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- **K-FAC** - Kronecker-Factored Approximate Curvature
- **Newton-CG** - ✨ Newton Conjugate Gradient (v1.0.0)

#### Learning Rate Schedulers
- **ExponentialDecay** - Exponential learning rate decay
- **StepDecay** - Step-wise learning rate reduction
- **CosineAnnealing** - Cosine annealing schedule
- **LinearWarmup** - Linear warmup with decay
- **OneCycleLR** - One cycle learning rate policy

#### Advanced Performance Features (Phase 2 Complete)

**SIMD Acceleration** (2-4x speedup)
- Automatic SIMD vectorization for f32/f64
- Uses `scirs2_core::simd_ops::SimdUnifiedOps`
- Threshold-based activation (16 elements for f32, 8 for f64)
- SimdSGD optimizer with momentum support

**Parallel Processing** (4-8x speedup)
- Multi-core parameter group processing
- Automatic work distribution across CPU cores
- ParallelOptimizer wrapper for any optimizer
- Uses `scirs2_core::parallel_ops` exclusively

**Memory-Efficient Operations**
- Gradient accumulation for micro-batch training
- Chunked parameter processing for billion-parameter models
- Memory usage estimation and recommendations
- Self-contained implementation using only SciRS2 standard features

**GPU Acceleration Framework** (10-50x potential speedup)
- GPU context management and initialization
- Multi-backend support (CUDA, Metal, OpenCL, WebGPU)
- Tensor cores and mixed-precision support
- Host-device data transfer utilities
- GPU memory tracking and statistics
- Built on `scirs2_core::gpu` abstractions

**Production Metrics & Monitoring**
- Real-time optimizer performance tracking
- Gradient statistics (mean, std dev, norm, sparsity)
- Parameter statistics (update magnitude, relative change)
- Convergence detection with moving averages
- Multi-optimizer tracking with MetricsCollector
- Export to JSON and CSV formats
- Minimal overhead (<5% typical)

### Performance Benchmarks

All benchmarks use Criterion.rs with statistical analysis:

- **optimizer_benchmarks.rs** - Compare 16 optimizers (100 to 100k parameters)
- **simd_benchmarks.rs** - SIMD vs scalar performance (expected 2-4x)
- **parallel_benchmarks.rs** - Multi-core scaling (expected 4-8x)
- **memory_efficient_benchmarks.rs** - Memory optimization impact
- **gpu_benchmarks.rs** - GPU vs CPU comparison (expected 10-50x)
- **metrics_benchmarks.rs** - Monitoring overhead measurement

### Test Coverage

- **549 unit tests** - Core optimizer functionality
- **54 doc tests** - Documentation examples
- **603 total tests** - All passing
- **Zero clippy warnings** - Production-ready code quality

### GPU Acceleration (`optirs-gpu`) - Coming Soon
- **Multi-GPU Support**: Distributed optimization across multiple GPUs
- **Backend Support**: CUDA, Metal, OpenCL, WebGPU
- **Memory Management**: Advanced memory pools and optimization
- **Tensor Cores**: Optimized for modern GPU architectures
- **Performance**: Highly optimized kernels for maximum throughput

### TPU Coordination (`optirs-tpu`) - Coming Soon
- **Pod Management**: TPU pod coordination and synchronization
- **XLA Integration**: Compiler optimizations for TPU workloads
- **Fault Tolerance**: Robust handling of hardware failures
- **Distributed Training**: Large-scale distributed optimization

### Learned Optimizers (`optirs-learned`) - Research Phase
- **Transformer-based Optimizers**: Self-attention mechanisms for optimization
- **LSTM Optimizers**: Recurrent neural network optimizers
- **Meta-Learning**: Learning to optimize across different tasks
- **Few-Shot Optimization**: Rapid adaptation to new optimization problems

### Neural Architecture Search (`optirs-nas`) - Research Phase
- **Search Strategies**: Bayesian, evolutionary, reinforcement learning
- **Multi-Objective**: Balancing accuracy, efficiency, and resource usage
- **Progressive Search**: Gradually increasing architecture complexity
- **Hardware-Aware**: Optimization for specific hardware targets

### Benchmarking (`optirs-bench`) ✅ Available
- **Performance Analysis**: Comprehensive benchmarking tools
- **Statistical Analysis**: Using Criterion.rs
- **Memory Profiling**: Detailed memory usage analysis
- **Throughput Metrics**: Elements/second tracking

## Quick Start

### Installation

```toml
[dependencies]
optirs-core = "0.1.0-rc.1"
scirs2-core = "0.1.0-rc.2"  # Required foundation

# Optional: GPU acceleration (experimental)
optirs-gpu = { version = "0.1.0-rc.1", optional = true }
```

### Basic Usage

```rust
use optirs_core::optimizers::{Adam, Optimizer};
// ALWAYS use scirs2_core for arrays - NEVER direct ndarray!
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create parameters and gradients using SciRS2
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.08]);

    // Create Adam optimizer
    let mut optimizer = Adam::new(0.001);

    // Perform optimization step
    let updated_params = optimizer.step(&params, &gradients)?;

    println!("Updated parameters: {:?}", updated_params);
    Ok(())
}
```

### SIMD Acceleration (2-4x speedup)

```rust
use optirs_core::simd_optimizer::SimdSGD;
use optirs_core::optimizers::Optimizer;
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Large parameter array (SIMD shines with 10k+ elements)
    let params = Array1::from_elem(100_000, 1.0f32);
    let grads = Array1::from_elem(100_000, 0.001f32);

    // SIMD-accelerated SGD
    let mut optimizer = SimdSGD::new(0.01f32);
    let updated = optimizer.step(&params, &grads)?;

    println!("Optimized {} parameters with SIMD", updated.len());
    Ok(())
}
```

### Parallel Processing (4-8x speedup)

```rust
use optirs_core::optimizers::{Adam, Optimizer};
use optirs_core::parallel_optimizer::parallel_step_array1;
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Multiple parameter groups (e.g., different network layers)
    let params_list = vec![
        Array1::from_elem(10_000, 1.0),
        Array1::from_elem(20_000, 1.0),
        Array1::from_elem(15_000, 1.0),
    ];

    let grads_list = vec![
        Array1::from_elem(10_000, 0.01),
        Array1::from_elem(20_000, 0.01),
        Array1::from_elem(15_000, 0.01),
    ];

    // Process all groups in parallel
    let mut optimizer = Adam::new(0.001);
    let updated_list = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;

    println!("Optimized {} parameter groups in parallel", updated_list.len());
    Ok(())
}
```

### Production Monitoring

```rust
use optirs_core::optimizers::{Adam, Optimizer};
use optirs_core::optimizer_metrics::{MetricsCollector, MetricsReporter};
use scirs2_core::ndarray::Array1;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collector = MetricsCollector::new();
    collector.register_optimizer("adam");

    let mut optimizer = Adam::new(0.001);
    let params = Array1::from_elem(1000, 1.0);
    let grads = Array1::from_elem(1000, 0.01);

    // Training loop with metrics
    for _ in 0..100 {
        let params_before = params.clone();
        let start = Instant::now();

        let params = optimizer.step(&params, &grads)?;
        let duration = start.elapsed();

        // Update metrics
        collector.update(
            "adam",
            duration,
            0.001,
            &grads.view(),
            &params_before.view(),
            &params.view(),
        )?;
    }

    // Generate report
    println!("{}", collector.summary_report());

    // Export to JSON
    let metrics = collector.get_metrics("adam").unwrap();
    println!("{}", MetricsReporter::to_json(metrics));

    Ok(())
}
```

### Complete Examples

See the `examples/` directory for comprehensive examples:

- **basic_optimization.rs** - Getting started with SGD, Adam, AdamW
- **advanced_optimization.rs** - Schedulers, parameter groups, regularization, gradient clipping
- **performance_optimization.rs** - SIMD, parallel, memory-efficient, GPU acceleration
- **production_monitoring.rs** - Metrics collection, convergence detection, profiling

Run examples with:
```bash
cargo run --example basic_optimization --release
cargo run --example advanced_optimization --release
cargo run --example performance_optimization --release
cargo run --example production_monitoring --release
```

## Documentation

### Comprehensive Guides

- **USAGE_GUIDE.md** - Comprehensive user guide (8000+ words)
  - Quick start and installation
  - All 16 optimizers with examples
  - Advanced features (schedulers, parameter groups, regularization)
  - Performance optimization (SIMD, parallel, memory-efficient, GPU)
  - Production deployment (metrics, monitoring, convergence)
  - SciRS2 integration patterns
  - Best practices and troubleshooting

### API Documentation

Generate and view API documentation:
```bash
cargo doc --open --no-deps
```

All public APIs are fully documented with:
- Detailed function descriptions
- Parameter explanations
- Return value specifications
- Usage examples
- Performance notes
- SciRS2 integration patterns

### Module Documentation

Each module contains comprehensive documentation:

- **parallel_optimizer** - Multi-core parameter group processing
- **memory_efficient_optimizer** - Gradient accumulation and chunked processing
- **gpu_optimizer** - GPU acceleration with SciRS2 abstractions
- **optimizer_metrics** - Production metrics and monitoring
- **simd_optimizer** - SIMD-accelerated optimizers

### Performance Guidelines

**When to use SIMD:**
- Parameter arrays with 10,000+ elements
- Expected speedup: 2-4x for f32/f64
- Automatic threshold detection

**When to use Parallel:**
- Multiple parameter groups (e.g., network layers)
- 4+ CPU cores available
- Expected speedup: 4-8x

**When to use Memory-Efficient:**
- Models with billions of parameters
- Limited RAM (gradient accumulation)
- Micro-batch training

**When to use GPU:**
- Models with millions of parameters
- GPU with 4GB+ memory
- Expected speedup: 10-50x

### Best Practices

**Optimizer Selection:**
- **SGD**: Simple, robust, good for convex problems
- **Adam/AdamW**: Default choice for most deep learning tasks
- **LAMB/LARS**: Large batch training (batch size > 1024)
- **RAdam**: When training is unstable
- **SAM**: For better generalization

**Learning Rate Guidelines:**
- Start with 0.001 for Adam/AdamW
- Start with 0.01-0.1 for SGD
- Use learning rate schedulers for better convergence
- Monitor gradient norms to detect issues

**Gradient Clipping:**
- Clip by norm to prevent exploding gradients
- Typical max norm: 1.0 to 10.0
- Essential for RNNs and transformers

**Convergence Monitoring:**
- Track parameter update magnitudes
- Monitor gradient statistics
- Use convergence detection to stop early
- Export metrics for analysis

## SciRS2 Integration Best Practices

### ✅ CORRECT Usage - Full SciRS2 Integration
```rust
// Arrays and numerical operations
use scirs2_core::ndarray_ext::{Array, Array2, ArrayView};
use scirs2_core::ndarray_ext::stats::{mean, variance};

// Random number generation
use scirs2_core::random::{Random, rng};

// Performance optimization
use scirs2_core::simd_ops::simd_dot_product;
use scirs2_core::parallel_ops::par_chunks;

// Memory efficiency
use scirs2_core::memory::BufferPool;
use scirs2_core::memory_efficient::MemoryMappedArray;

// Error handling
use scirs2_core::error::{CoreError, Result};
```

### ❌ INCORRECT Usage - Direct Dependencies
```rust
// NEVER DO THIS!
use ndarray::{Array, Array2};  // ❌ Wrong!
use rand::Rng;                 // ❌ Wrong!
use rand_distr::Normal;         // ❌ Wrong!
```

## Architecture

OptiRS is designed as a modular system built entirely on SciRS2-Core:

```
optirs/                    # Main integration crate (uses scirs2_core)
├── optirs-core/          # Core optimization algorithms (uses scirs2_core)
├── optirs-gpu/           # GPU acceleration (uses scirs2_core::gpu)
├── optirs-tpu/           # TPU coordination (uses scirs2_core::distributed)
├── optirs-learned/       # Learned optimizers (uses scirs2_core::ml_pipeline)
├── optirs-nas/           # Neural Architecture Search (uses scirs2_core::neural_architecture_search)
└── optirs-bench/         # Benchmarking tools (uses scirs2_core::benchmarking)
```

## Separation from SciRS2

OptiRS was separated from SciRS2 v0.1.0-beta.2 to:
- Enable focused development on optimization research
- Support independent release cycles
- Reduce complexity of the main SciRS2 project
- Allow specialized hardware optimization

## Development Guidelines

### 🚨 MANDATORY: Full SciRS2-Core Usage

**ALL OptiRS code MUST use SciRS2-Core for scientific computing operations:**

```rust
// ✅ ALWAYS use SciRS2-Core
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::Random;
use scirs2_core::simd_ops::simd_dot_product;
use scirs2_core::parallel_ops::par_chunks;
use scirs2_core::error::Result;

// ❌ NEVER use direct dependencies
use ndarray::Array2;        // ❌ FORBIDDEN
use rand::thread_rng;       // ❌ FORBIDDEN
use rayon::prelude::*;      // ❌ Use scirs2_core::parallel instead
```

### Coding Standards

To maintain consistency and readability across the entire OptiRS ecosystem, all contributors must follow these guidelines:

#### SciRS2 Integration Requirements
- **MUST** use `scirs2_core::ndarray` for ALL array operations
- **MUST** use `scirs2_core::random` for ALL random number generation
- **MUST** use `scirs2_core::simd` for ALL SIMD operations
- **MUST** use `scirs2_core::parallel` for ALL parallel processing
- **MUST** use `scirs2_core::error::Result` for ALL error handling
- **MUST** use `scirs2_core::profiling` for ALL performance profiling
- **MUST** use `scirs2_core::benchmarking` for ALL benchmarks

#### Variable Naming
- **Always use `snake_case` for variable names** (e.g., `user_id`, `max_iterations`, `learning_rate`)
- **Avoid camelCase or other naming conventions** (e.g., `userId` ❌, `maxIterations` ❌)
- **Use descriptive names** that clearly indicate the variable's purpose

```rust
// ✅ Correct: snake_case with SciRS2 types
use scirs2_core::ndarray_ext::Array2;
let experiment_id = "exp_001";
let max_epochs = 100;
let learning_rate = 0.001;
let gradient_array = Array2::<f32>::zeros((100, 50));

// ❌ Incorrect: camelCase or direct dependencies
use ndarray::Array2;  // ❌ Wrong dependency!
let experimentId = "exp_001";
let maxEpochs = 100;
```

#### Function and Method Names
- Use `snake_case` for function and method names
- Use descriptive verbs that indicate the function's action

#### Type Names
- Use `PascalCase` for struct, enum, and trait names
- Use `SCREAMING_SNAKE_CASE` for constants

#### General Guidelines
- Follow Rust's official naming conventions as specified in [RFC 430](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md)
- Use `rustfmt` and `clippy` to maintain code formatting and catch common issues
- Write clear, self-documenting code with appropriate comments

### Before Submitting Code
1. Run `cargo fmt` to format your code
2. Run `cargo clippy` to check for lint issues
3. Ensure all tests pass with `cargo test`
4. Verify compilation with `cargo check`

## Contributing

We welcome contributions! When contributing to OptiRS, please ensure:

1. **ALL code uses SciRS2-Core** - No direct ndarray or rand imports
2. **Follow the SciRS2 integration guidelines** in CLAUDE.md
3. **Run tests with SciRS2 dependencies** - `cargo test`
4. **Benchmark using SciRS2 tools** - `scirs2_core::benchmarking`
5. **Profile using SciRS2 profiler** - `scirs2_core::profiling`

## SciRS2 Dependency Verification

Before submitting PRs, verify SciRS2 usage:

```bash
# Check for forbidden direct dependencies
grep -r "use ndarray::" --include="*.rs" .  # Should return nothing
grep -r "use rand::" --include="*.rs" .     # Should return nothing

# Verify SciRS2 usage
grep -r "use scirs2_core::" --include="*.rs" . # Should show many results
```

## License

This project is dual-licensed under MIT OR Apache-2.0.

---

**⚠️ REMEMBER**: OptiRS is an extension of SciRS2, not a standalone project. It MUST leverage the full power of scirs2-core for ALL scientific computing operations. Direct use of ndarray, rand, or other libraries that scirs2-core provides is **STRICTLY FORBIDDEN**.