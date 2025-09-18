# OptiRS - Advanced ML Optimization Built on SciRS2

OptiRS is a comprehensive optimization library for machine learning that **extends and leverages the full power of SciRS2-Core**. It provides specialized optimization algorithms and hardware acceleration while making **FULL USE** of SciRS2's scientific computing capabilities.

## 🚨 CRITICAL: Full SciRS2-Core Usage

**OptiRS is NOT a standalone project** - it is an extension of SciRS2 that MUST make full use of scirs2-core for ALL operations:
- ✅ **Arrays**: Uses `scirs2_core::ndarray_ext` exclusively (NO direct ndarray)
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

### Core Optimizers (`optirs-core`)
- **Classic Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad, LAMB, LARS, etc.
- **Learning Rate Schedulers**: Exponential decay, cosine annealing, step decay, etc.
- **Regularization**: L1, L2, dropout, elastic net, spectral normalization
- **Second-Order Methods**: L-BFGS, Newton methods, K-FAC
- **Advanced Features**: Parameter groups, gradient accumulation, privacy-preserving optimization

### GPU Acceleration (`optirs-gpu`)
- **Multi-GPU Support**: Distributed optimization across multiple GPUs
- **Backend Support**: CUDA, Metal, OpenCL, WebGPU
- **Memory Management**: Advanced memory pools and optimization
- **Tensor Cores**: Optimized for modern GPU architectures
- **Performance**: Highly optimized kernels for maximum throughput

### TPU Coordination (`optirs-tpu`)
- **Pod Management**: TPU pod coordination and synchronization
- **XLA Integration**: Compiler optimizations for TPU workloads
- **Fault Tolerance**: Robust handling of hardware failures
- **Distributed Training**: Large-scale distributed optimization

### Learned Optimizers (`optirs-learned`)
- **Transformer-based Optimizers**: Self-attention mechanisms for optimization
- **LSTM Optimizers**: Recurrent neural network optimizers
- **Meta-Learning**: Learning to optimize across different tasks
- **Few-Shot Optimization**: Rapid adaptation to new optimization problems

### Neural Architecture Search (`optirs-nas`)
- **Search Strategies**: Bayesian, evolutionary, reinforcement learning
- **Multi-Objective**: Balancing accuracy, efficiency, and resource usage
- **Progressive Search**: Gradually increasing architecture complexity
- **Hardware-Aware**: Optimization for specific hardware targets

### Benchmarking (`optirs-bench`)
- **Performance Analysis**: Comprehensive benchmarking tools
- **Regression Detection**: Automatic performance regression detection
- **Memory Profiling**: Detailed memory usage analysis
- **Security Auditing**: Vulnerability scanning and analysis

## Quick Start

```toml
[dependencies]
optirs = "0.1.0"
# Or specific modules:
optirs-core = "0.1.0"
optirs-gpu = { version = "0.1.0", features = ["cuda"] }
```

```rust
use optirs::prelude::*;
// ALWAYS use scirs2_core for arrays - NEVER direct ndarray!
use scirs2_core::ndarray_ext::Array2;

// Create an optimizer with SciRS2 integration
let optimizer = Adam::new(0.001)?;

// Use with your gradients (using SciRS2 arrays)
let gradients = Array2::zeros((10, 10));
let updated_params = optimizer.step(&gradients)?;
```

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
- **MUST** use `scirs2_core::ndarray_ext` for ALL array operations
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