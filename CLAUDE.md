# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OptiRS is a comprehensive ML optimization library in Rust, separated from SciRS2 to focus on specialized optimization algorithms and hardware acceleration. It provides advanced optimizers, GPU/TPU acceleration, learned optimizers, and neural architecture search capabilities.

## Architecture

OptiRS is a Rust workspace with modular crates:
- `optirs/` - Main integration crate with feature gates
- `optirs-core/` - Core optimization algorithms (SGD, Adam, schedulers, etc.)
- `optirs-gpu/` - GPU acceleration (CUDA, Metal, OpenCL, WebGPU)
- `optirs-tpu/` - TPU coordination and distributed training
- `optirs-learned/` - Transformer-based and LSTM optimizers with meta-learning
- `optirs-nas/` - Neural Architecture Search
- `optirs-bench/` - Benchmarking and performance analysis

### Core Module Structure (`optirs-core/src/`)
- `optimizers/` - Classic optimization algorithms
- `schedulers/` - Learning rate schedulers
- `regularizers/` - L1, L2, dropout implementations
- `second_order/` - L-BFGS, Newton methods, K-FAC
- `gradient_processing/` - Gradient manipulation utilities
- `neural_integration/` - Neural network integration
- `distributed/` - Distributed optimization support
- `privacy/` - Privacy-preserving optimization
- `unified_api.rs` - Unified optimizer interface

## Build Commands

```bash
# Build the project
cargo build

# Build with all features
cargo build --all-features

# Build specific module
cargo build -p optirs-core

# Build with GPU support
cargo build --features gpu

# Build for release
cargo build --release
```

## Test Commands

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run tests for specific module
cargo test -p optirs-core

# Run a single test
cargo test test_name

# Run tests with output
cargo test -- --nocapture
```

## Development Commands

```bash
# Format code
cargo fmt

# Run clippy linter
cargo clippy

# Check compilation
cargo check

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## Feature Flags

Available features for the main `optirs` crate:
- `core` - Core optimization (default)
- `gpu` - GPU acceleration
- `tpu` - TPU coordination
- `learned` - Learned optimizers
- `nas` - Neural Architecture Search
- `bench` - Benchmarking tools
- `full` - All features

Example: `cargo build --features "gpu,learned"`

## 🚨 CRITICAL: SciRS2 Dependency Policy (MANDATORY)

**OptiRS is a NON-CORE SciRS2 ecosystem crate and MUST follow the SciRS2 Ecosystem Policy v3.0.0**

See detailed policy in: `SCIRS2_INTEGRATION_POLICY.md`

**Important**:
1. From SciRS2 v0.1.0+, the `array!` macro is available directly from `scirs2_core::ndarray`
2. OptiRS does **NOT** require `scirs2-autograd` - OptiRS is an optimization library that receives pre-computed gradients, not an automatic differentiation library

### Prohibited Dependencies (NEVER USE)

**NO DIRECT EXTERNAL DEPENDENCIES allowed in OptiRS:**

```toml
# ❌ FORBIDDEN in ALL OptiRS Cargo.toml files
rand = "*"              # Use scirs2_core::random
rand_distr = "*"        # Use scirs2_core::random
ndarray = "*"           # Use scirs2_core::ndarray
ndarray-rand = "*"      # Use scirs2_core::ndarray
ndarray-stats = "*"     # Use scirs2_core::ndarray
num-traits = "*"        # Use scirs2_core::numeric
num-complex = "*"       # Use scirs2_core::numeric
rayon = "*"             # Use scirs2_core::parallel_ops
wide = "*"              # Use scirs2_core::simd_ops
nalgebra = "*"          # Use scirs2_core::linalg
```

### Required Dependencies (ONLY USE THESE)

```toml
# ✅ REQUIRED - SciRS2 Foundation
scirs2-core = { workspace = true }        # MANDATORY - All scientific computing
scirs2-optimize = { workspace = true }    # REQUIRED - Base optimization interfaces

# Additional SciRS2 crates as needed (based on OptiRS requirements)
scirs2-linalg = { workspace = true }      # Linear algebra operations (if needed)
# Note: scirs2-autograd is NOT required - OptiRS receives gradients, doesn't compute them
scirs2-neural = { workspace = true }      # Neural network support (if NN-specific optimizers)
scirs2-metrics = { workspace = true }     # Performance monitoring
scirs2-stats = { workspace = true }       # Statistical analysis
```

SciRS2 is located at `../scirs/` relative to this project.

### Why This Policy Exists

1. **Architectural Integrity**: OptiRS extends SciRS2, not external libraries
2. **Version Control**: Only scirs2-core manages external dependency versions
3. **Type Safety**: Prevents mixing external types with SciRS2 types
4. **Consistency**: All ecosystem crates use same optimized implementations
5. **Maintainability**: Updates in one place benefit all modules

### FULL USE OF SciRS2-Core

OptiRS must make **FULL USE** of scirs2-core's extensive capabilities:

#### Core Array Operations (replaces ndarray)
```rust
// ✅ CORRECT - Option 1: Use ndarray_ext for NumPy-like extensions
use scirs2_core::ndarray_ext::*;  // Includes fancy indexing, broadcasting, stats + all macros
use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView, ArrayViewMut};
use scirs2_core::ndarray_ext::{Axis, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray_ext::{array, s, azip};  // ALL macros available

// ✅ CORRECT - Option 2: Use ndarray for standard operations
use scirs2_core::ndarray::*;  // Standard ndarray re-exports + all macros
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, ArrayViewMut};
use scirs2_core::ndarray::{Axis, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray::{array, s, azip};  // ALL macros available

// Example usage (works with both modules)
let arr = array![[1.0, 2.0], [3.0, 4.0]];  // array! macro works
let slice = arr.slice(s![.., 0]);          // s! macro works
azip!((a in &mut arr) *a *= 2.0);         // azip! macro works

// Choose ndarray_ext when you need:
// - Fancy indexing (boolean masks, index arrays)
// - Broadcasting helpers
// - Statistical functions
// - NumPy-like manipulation

// ❌ FORBIDDEN - Direct ndarray imports
use ndarray::*;  // NEVER USE - must go through scirs2_core
use ndarray::{Array, Array1, Array2};  // NEVER USE
use ndarray::{array, s};  // NEVER USE
```

#### Random Number Generation (replaces rand/rand_distr)
```rust
// ✅ CORRECT - Use scirs2-core's random module (v0.1.0+)
use scirs2_core::random::*;  // Complete rand + rand_distr functionality
use scirs2_core::random::{thread_rng, Rng};
// ALL distributions available: Beta, Cauchy, ChiSquared, FisherF, LogNormal,
// Normal, StudentT, Weibull, Bernoulli, Binomial, Poisson, etc.
use scirs2_core::random::{Normal, RandBeta, Cauchy, ChiSquared, StudentT};

// Example usage
let mut rng = thread_rng();
let normal = Normal::new(0.0, 1.0)?;
let beta = RandBeta::new(2.0, 5.0)?;  // Note: RandBeta to avoid naming conflict
let sample = normal.sample(&mut rng);

// ❌ FORBIDDEN - Direct rand imports
use rand::*;  // NEVER USE
use rand::thread_rng;  // NEVER USE
use rand_distr::*;  // NEVER USE
use rand_distr::{Normal, Beta};  // NEVER USE
```

#### Numerical Traits (replaces num-traits/num-complex)
```rust
// ✅ CORRECT - Use scirs2-core's numeric module
use scirs2_core::numeric::*;  // num-traits, num-complex, num-integer
use scirs2_core::numeric::{Float, Zero, One, Num};
use scirs2_core::numeric::Complex;

// ❌ FORBIDDEN - Direct num-traits imports
use num_traits::*;  // NEVER USE
use num_traits::Float;  // NEVER USE
use num_complex::Complex;  // NEVER USE
```

#### Performance Optimization Features (replaces rayon/wide)
```rust
// ✅ CORRECT - SIMD acceleration (replaces wide)
use scirs2_core::simd_ops::SimdUnifiedOps;

// Automatic SIMD optimization
let result = f32::simd_add(&a.view(), &b.view());
let dot = f64::simd_dot(&x.view(), &y.view());

// ❌ FORBIDDEN - Direct SIMD libraries
use wide::*;  // NEVER USE

// ✅ CORRECT - Parallel processing (replaces rayon)
use scirs2_core::parallel_ops::*;

// Parallel operations
let results: Vec<f64> = (0..n)
    .into_par_iter()
    .map(|i| compute(i))
    .collect();

// ❌ FORBIDDEN - Direct rayon imports
use rayon::prelude::*;  // NEVER USE

// ✅ CORRECT - GPU acceleration
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision};
```

#### Memory Management & Efficiency
```rust
// Memory-efficient operations
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use scirs2_core::memory_efficient::{ZeroCopyOps, AdaptiveChunking, DiskBackedArray};

// Memory management
use scirs2_core::memory::{BufferPool, GlobalBufferPool, ChunkProcessor};
use scirs2_core::memory::{LeakDetector, MemoryMetricsCollector};
```

#### Advanced Scientific Computing
```rust
// Complex numbers and numeric conversions
use scirs2_core::types::{ComplexOps, ComplexExt, NumericConversion};

// Scientific constants and units
use scirs2_core::constants::{math, physical, prefixes};
use scirs2_core::units::{UnitSystem, UnitRegistry, Dimension, convert};

// Validation and error handling
use scirs2_core::validation::{check_finite, check_in_bounds, ValidationSchema};
use scirs2_core::error::{CoreError, Result};
```

#### Production-Ready Features
```rust
// Performance profiling
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// Metrics and monitoring
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::observability::{audit, tracing};

// Stability and versioning
use scirs2_core::stability::{StabilityLevel, ApiContract, BreakingChange};
use scirs2_core::versioning::{Version, VersionManager, CompatibilityLevel};
```

#### Machine Learning Pipeline Integration
```rust
// ML pipeline support
use scirs2_core::ml_pipeline::{MLPipeline, ModelPredictor, FeatureTransformer};
use scirs2_core::ml_pipeline::{DataBatch, PipelineNode, PipelineMetrics};

// Neural architecture search
use scirs2_core::neural_architecture_search::{NeuralArchitectureSearch, SearchSpace};

// Quantum optimization
use scirs2_core::quantum_optimization::{QuantumOptimizer, QuantumStrategy};
```

#### JIT Compilation & Optimization
```rust
// Just-in-time compilation
use scirs2_core::jit::{JitCompiler, JitBackend, CompiledKernel, OptimizationLevel};
use scirs2_core::advanced_jit_compilation::{AdaptiveCodeGenerator, RuntimeOptimizer};
```

#### Cloud & Distributed Computing
```rust
// Cloud storage integration
use scirs2_core::cloud::{CloudStorageClient, CloudProvider, S3, GCS, Azure};

// Distributed computing
use scirs2_core::distributed::{ClusterManager, JobScheduler, DataParallelism};
use scirs2_core::advanced_distributed_computing::{DistributedOptimizer, AllReduce};
```

#### Advanced Array Protocol
```rust
// Array protocol for interoperability
use scirs2_core::array_protocol::{ArrayProtocol, GPUArray, DistributedArray};
use scirs2_core::array_protocol::{DifferentiableArray, AsyncArray, ZeroCopyArray};
```

### Mandatory Usage Guidelines (SciRS2 Ecosystem Policy v3.0.0)

1. **NEVER** import `ndarray` directly → **ALWAYS** use `scirs2_core::ndarray_ext` (for NumPy-like extensions) OR `scirs2_core::ndarray` (for standard operations) - both include `array!`, `s!`, `azip!` macros
2. **NEVER** import `rand` or `rand_distr` directly → **ALWAYS** use `scirs2_core::random` (all distributions included)
3. **NEVER** import `num-traits` or `num-complex` directly → **ALWAYS** use `scirs2_core::numeric`
4. **NEVER** import `rayon` directly → **ALWAYS** use `scirs2_core::parallel_ops`
5. **NEVER** import `wide` directly → **ALWAYS** use `scirs2_core::simd_ops`
6. **ALWAYS** use `scirs2_core::validation` for parameter validation (check_positive, check_finite, etc.)
7. **ALWAYS** use scirs2-core's GPU abstractions for hardware acceleration
8. **ALWAYS** use scirs2-core's memory management for large data operations
9. **ALWAYS** use scirs2-core's profiling and benchmarking tools
10. **ALWAYS** use scirs2-core's error types and result handling

**Violation of these guidelines is a CRITICAL architectural error and must be fixed immediately.**

**Note on Array Modules**:
- `scirs2_core::ndarray_ext`: Extended NumPy-like functionality (fancy indexing, boolean masking, broadcasting helpers, stats)
- `scirs2_core::ndarray`: Standard ndarray re-exports (basic array operations, views, slicing)
- **Both are valid and encouraged** depending on your needs. Choose based on required functionality.

**Key Updates (v0.1.0+)**:
1. The `array!` macro is now available directly from `scirs2_core::ndarray`
2. OptiRS does **NOT** use `scirs2-autograd` - OptiRS receives gradients, doesn't compute them

## Development Guidelines

1. **Variable Naming**: Always use `snake_case` for variables, functions, and methods
2. **Type Naming**: Use `PascalCase` for structs, enums, traits
3. **Constants**: Use `SCREAMING_SNAKE_CASE`
4. **Workspace Dependencies**: Use `workspace = true` in Cargo.toml
5. **Latest Crates**: Always use the latest version available on crates.io
6. **Use SciRS2**: Replace direct `rand` and `ndarray` usage with `scirs2-core` equivalents

## Testing Strategy

- Unit tests in each module's source files
- Integration tests in `tests/` directories
- Benchmarks in `benches/` directories
- Use `approx` for floating-point comparisons
- Mock GPU/TPU for hardware tests when not available

## Key Implementation Patterns (Updated v0.1.0+)

1. **Error Handling**: Use `scirs2_core::error::CoreError` and `scirs2_core::error::Result`
2. **Parameter Validation**: Use `scirs2_core::validation` functions (check_positive, check_finite, check_shape, etc.)
3. **Array Operations**: Use `scirs2_core::ndarray_ext` for extended NumPy-like features (fancy indexing, broadcasting, stats), OR `scirs2_core::ndarray` for standard ndarray re-exports - both include ALL macros (array!, s!, azip!)
4. **Random Numbers**: Use `scirs2_core::random` exclusively - includes ALL distributions from rand_distr
5. **Numerical Traits**: Use `scirs2_core::numeric` exclusively (Float, Zero, One, Complex, etc.)
6. **Parallelization**: Use `scirs2_core::parallel_ops` exclusively (NOT direct `rayon`)
7. **SIMD Operations**: Use `scirs2_core::simd_ops::SimdUnifiedOps` exclusively (NOT direct `wide`)
8. **Async Operations**: Use `tokio` with `scirs2_core::array_protocol::AsyncArray`
9. **GPU Backends**: Use `scirs2_core::gpu` abstractions
10. **Memory Efficiency**: Use `scirs2_core::memory_efficient` for large data
11. **Profiling**: Use `scirs2_core::profiling` and `benchmarking`
12. **Metrics**: Use `scirs2_core::metrics` for monitoring

**Remember**: ALL external dependencies MUST go through scirs2-core abstractions. Direct imports from `ndarray`, `rand`, `num-traits`, etc. are FORBIDDEN.

**Array Module Choice**:
- Use `scirs2_core::ndarray_ext` when you need NumPy-like extensions (fancy indexing, boolean masking, broadcasting helpers, statistical functions)
- Use `scirs2_core::ndarray` for standard ndarray operations (basic array creation, slicing, views)
- Both modules include all essential macros (`array!`, `s!`, `azip!`)

## OptiRS Module-Specific SciRS2 Usage

### optirs-core
- Use `scirs2_core::ndarray_ext` for array operations with NumPy-like extensions (fancy indexing, stats, broadcasting)
- Use `scirs2_core::ndarray` for standard ndarray operations (both include `array!`, `s!`, `azip!` macros - v0.1.0+)
- Use `scirs2_core::random` for all RNG operations (includes ALL distributions: Normal, Beta, Cauchy, etc.)
- Use `scirs2_core::numeric` for numerical traits (Float, Zero, One, Complex)
- Use `scirs2_core::validation` for parameter validation (check_positive, check_finite, etc.)
- Use `scirs2_core::simd_ops::SimdUnifiedOps` for gradient processing
- Use `scirs2_core::parallel_ops` for parameter groups
- Use `scirs2_core::memory::BufferPool` for memory management
- Use `scirs2_core::metrics` for optimization metrics
- Use `scirs2_core::profiling` for performance analysis

### optirs-gpu
- Use `scirs2_core::gpu` as the foundation for GPU abstractions
- Use `scirs2_core::tensor_cores` for mixed-precision training
- Use `scirs2_core::array_protocol::GPUArray` for GPU array interface
- Use `scirs2_core::memory::TrackedGpuBuffer` for GPU memory tracking

### optirs-tpu
- Use `scirs2_core::distributed` for TPU pod coordination
- Use `scirs2_core::advanced_distributed_computing` for AllReduce operations
- Use `scirs2_core::array_protocol::DistributedArray` for distributed arrays

### optirs-learned
- Use `scirs2_core::ml_pipeline` for meta-learning pipelines
- Use `scirs2_core::neural_architecture_search` as foundation
- Use `scirs2_core::memory_efficient::LazyArray` for gradient history
- Use `scirs2_core::jit` for optimized transformer kernels

### optirs-nas
- Use `scirs2_core::neural_architecture_search::SearchSpace`
- Use `scirs2_core::quantum_optimization` for search strategies
- Use `scirs2_core::parallel::LoadBalancer` for parallel search

### optirs-bench
- Use `scirs2_core::benchmarking` exclusively for all benchmarks
- Use `scirs2_core::profiling::Profiler` for detailed analysis
- Use `scirs2_core::metrics::MetricRegistry` for tracking
- Use `scirs2_core::stability` for regression detection

## Important Files

- `README.md` - Main project documentation
- `SCIRS2_INTEGRATION_POLICY.md` - Critical SciRS2 dependency policy
- `MIGRATION_FROM_SCIRS2.md` - Migration documentation
- `TODO.md` files in each module - Module-specific tasks

## Common Workflows

### Importing Core Types - FULL SciRS2 Usage (CORRECT) - v0.1.0+
```rust
// ✅ Arrays and numerical operations (SciRS2 v0.1.0+)
use scirs2_core::ndarray_ext::*;  // Includes array!, s!, azip! macros - ALL AVAILABLE
use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView, ArrayViewMut};
use scirs2_core::ndarray_ext::{Ix1, Ix2, IxDyn, Axis};

// Example: array! macro works directly
let arr = array![[1.0, 2.0], [3.0, 4.0]];
let slice = arr.slice(s![.., 0]);

// ✅ Random number generation (complete rand + rand_distr - ALL distributions)
use scirs2_core::random::*;  // thread_rng, Rng, ALL distributions
use scirs2_core::random::{thread_rng, Normal, RandBeta, Cauchy, ChiSquared, StudentT};

// Example: all distributions work
let mut rng = thread_rng();
let normal = Normal::new(0.0, 1.0)?;
let beta = RandBeta::new(2.0, 5.0)?;

// ✅ Numerical traits (num-traits, num-complex, num-integer)
use scirs2_core::numeric::*;  // Float, Zero, One, Complex, NumCast, etc.
use scirs2_core::numeric::{Float, Zero, One, Complex};

// ✅ Validation (parameter checking)
use scirs2_core::validation::*;  // check_positive, check_finite, check_shape, etc.

// ✅ Performance features
use scirs2_core::simd_ops::SimdUnifiedOps;  // SIMD operations
use scirs2_core::parallel_ops::*;           // Parallel processing
use scirs2_core::gpu::{GpuContext, GpuBuffer};

// ✅ Memory efficiency
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray};
use scirs2_core::memory::{BufferPool, GlobalBufferPool};

// ✅ Error handling
use scirs2_core::error::{CoreError, Result};

// ✅ Profiling and metrics
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::{Counter, Timer};

// ❌ FORBIDDEN - Never use these (CRITICAL ERROR)
use ndarray::*;        // NEVER - Use scirs2_core::ndarray
use ndarray::{array, s};  // NEVER - All macros in scirs2_core::ndarray
use rand::*;           // NEVER - Use scirs2_core::random
use rand_distr::*;     // NEVER - Use scirs2_core::random
use num_traits::*;     // NEVER - Use scirs2_core::numeric
use num_complex::*;    // NEVER - Use scirs2_core::numeric
use rayon::prelude::*; // NEVER - Use scirs2_core::parallel_ops
```

### Adding a New Optimizer with Full SciRS2 Integration
```rust
// optirs-core/src/optimizers/new_optimizer.rs

// ✅ CORRECT imports
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::parallel_ops::*;
use scirs2_core::memory::BufferPool;
use scirs2_core::metrics::Timer;
use scirs2_core::error::Result;

pub struct NewOptimizer<T: Float> {
    learning_rate: T,
    buffer_pool: BufferPool,
    timer: Timer,
}

impl<T: Float> NewOptimizer<T> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            buffer_pool: BufferPool::new(),
            timer: Timer::new("optimizer_step"),
        }
    }

    pub fn step(&mut self, params: ArrayView2<T>, grads: ArrayView2<T>) -> Result<Array2<T>> {
        self.timer.start();

        // Use scirs2-core's SIMD operations for f32/f64
        let updated = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SIMD-optimized path for f32
            let params_f32 = params.mapv(|x| x.to_f32().unwrap());
            let grads_f32 = grads.mapv(|x| x.to_f32().unwrap());

            let mut result = Array2::zeros(params.dim());
            f32::simd_add(&params_f32.view(), &grads_f32.view());
            result.mapv(|x| T::from(x).unwrap())
        } else {
            // Standard path
            params.to_owned() - &grads * self.learning_rate
        };

        // Use parallel processing for large parameter groups
        if params.len() > 1000 {
            updated.axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(|row| {
                    // Process each row in parallel
                    row.to_owned()
                })
                .collect()
        }

        self.timer.stop();
        Ok(updated)
    }
}

// ❌ FORBIDDEN - Never use these patterns
// use ndarray::{Array2, ArrayView2};  // NEVER
// use rand::thread_rng;                // NEVER
// use num_traits::Float;               // NEVER
// use rayon::prelude::*;               // NEVER
```

### GPU Acceleration with SciRS2
```rust
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision};

async fn gpu_optimize() -> Result<()> {
    // Use scirs2-core's GPU abstractions
    let context = GpuContext::new()?;
    let buffer = GpuBuffer::from_slice(&context, &data)?;

    // Use tensor cores for maximum performance
    let tensor_core = TensorCore::new(&context)?;
    tensor_core.gemm_mixed_precision(&a, &b, &mut c)?;

    Ok(())
}
```

### Benchmarking with SciRS2
```rust
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::MetricRegistry;

fn benchmark_optimizer() -> Result<()> {
    let mut suite = BenchmarkSuite::new("optimizer_bench");
    let profiler = Profiler::new();
    let metrics = MetricRegistry::global();

    suite.add_benchmark("adam_step", |b| {
        profiler.start("adam_optimization");
        // Benchmark code
        profiler.stop("adam_optimization");
    });

    let results = suite.run()?;
    metrics.record_benchmark(results);

    Ok(())
}
```

### Memory-Efficient Large Data Processing
```rust
use scirs2_core::memory_efficient::{MemoryMappedArray, AdaptiveChunking};
use scirs2_core::memory::LeakDetector;

fn process_large_dataset(path: &str) -> Result<()> {
    // Use memory-mapped arrays for huge datasets
    let mmap = MemoryMappedArray::open(path)?;

    // Use adaptive chunking for optimal performance
    let chunking = AdaptiveChunking::new()
        .with_memory_limit(1 << 30)  // 1GB
        .build()?;

    // Monitor for memory leaks
    let leak_detector = LeakDetector::new();

    for chunk in mmap.chunks_adaptive(&chunking) {
        // Process chunk without loading entire dataset
    }

    leak_detector.check()?;
    Ok(())
}
```

## Migration Checklist - SciRS2 Ecosystem Policy Compliance

When reviewing or writing OptiRS code, verify these requirements:

### ✅ Cargo.toml Dependencies
- [ ] NO `ndarray = { ... }` in any Cargo.toml
- [ ] NO `rand = { ... }` in any Cargo.toml
- [ ] NO `rand_distr = { ... }` in any Cargo.toml
- [ ] NO `num-traits = { ... }` in any Cargo.toml
- [ ] NO `num-complex = { ... }` in any Cargo.toml
- [ ] NO `rayon = { ... }` in any Cargo.toml (except as optional in workspace)
- [ ] NO `wide = { ... }` in any Cargo.toml (except as optional in workspace)
- [ ] YES `scirs2-core = { workspace = true }` in ALL crates
- [ ] YES `scirs2-optimize = { workspace = true }` in ALL crates

### ✅ Arrays and Numerical Operations
- [ ] NO direct `use ndarray::{...}`
- [ ] NO direct `Array`, `Array1`, `Array2` from ndarray
- [ ] YES `use scirs2_core::ndarray_ext::*` for NumPy-like extensions (fancy indexing, stats, broadcasting)
- [ ] YES `use scirs2_core::ndarray::*` for standard ndarray operations
- [ ] YES both modules include `array!`, `s!`, `azip!` macros (v0.1.0+)
- [ ] YES `use scirs2_core::ndarray_ext::{Array, Array1, Array2, array, s}` (with extensions)
- [ ] YES `use scirs2_core::ndarray::{Array, Array1, Array2, array, s}` (standard)

### ✅ Random Number Generation
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] NO `rng()` function calls without proper import
- [ ] YES `use scirs2_core::random::*` (v0.1.0+ includes ALL distributions)
- [ ] YES `use scirs2_core::random::{thread_rng, Rng, Normal, RandBeta, Cauchy, ...}`
- [ ] YES use `thread_rng()` instead of `rng()`
- [ ] NOTE: Use `RandBeta` instead of `Beta` to avoid naming conflicts

### ✅ Numerical Traits
- [ ] NO direct `use num_traits::{...}`
- [ ] NO direct `use num_complex::{...}`
- [ ] YES `use scirs2_core::numeric::*`
- [ ] YES `use scirs2_core::numeric::{Float, Zero, One, Complex}`

### ✅ Performance Optimization
- [ ] NO direct `use rayon::prelude::*`
- [ ] NO direct `use wide::{...}`
- [ ] YES use `scirs2_core::simd_ops::SimdUnifiedOps` for SIMD operations
- [ ] YES use `scirs2_core::parallel_ops::*` for parallelization
- [ ] YES use `scirs2_core::gpu` for GPU acceleration
- [ ] YES use `scirs2_core::memory_efficient` for large datasets

### ✅ Production Features
- [ ] YES use `scirs2_core::error::{CoreError, Result}`
- [ ] YES use `scirs2_core::validation::{check_positive, check_finite}`
- [ ] YES use `scirs2_core::profiling` for performance analysis
- [ ] YES use `scirs2_core::metrics` for monitoring
- [ ] YES use `scirs2_core::benchmarking` for benchmarks

### Common Anti-Patterns to Avoid (Updated v0.1.0+)
```rust
// ❌ WRONG - Direct external dependencies (FORBIDDEN - CRITICAL ERROR)
use ndarray::{Array2, array};       // NEVER USE
use ndarray::s;                      // NEVER USE
use rand::thread_rng;                // NEVER USE
use rand_distr::{Normal, Beta};      // NEVER USE
use num_traits::Float;               // NEVER USE
use num_complex::Complex;            // NEVER USE
use rayon::prelude::*;               // NEVER USE

let mut rng = rng();  // WRONG - function not available

// ✅ CORRECT - SciRS2-Core abstractions (REQUIRED - v0.1.0+)
use scirs2_core::ndarray_ext::{Array2, array, s};  // With NumPy-like extensions
// OR
use scirs2_core::ndarray::{Array2, array, s};      // Standard ndarray operations
// Both are CORRECT - choose based on your needs

use scirs2_core::random::{thread_rng, Normal, RandBeta};  // ALL distributions
use scirs2_core::numeric::{Float, Complex};
use scirs2_core::validation::{check_positive, check_finite};
use scirs2_core::parallel_ops::*;

// Correct usage examples
let mut rng = thread_rng();         // CORRECT
let arr = array![[1.0, 2.0]];      // CORRECT - macro works in both modules
let slice = arr.slice(s![.., 0]);  // CORRECT - s! macro works in both modules
let beta = RandBeta::new(2.0, 5.0)?;  // CORRECT - all distributions available
check_positive(x, "parameter")?;    // CORRECT - validation
```

### Build Verification
```bash
# Verify no prohibited dependencies
grep -r "num-traits\|num-complex\|^rand\|^ndarray\|^rayon\|^wide" */Cargo.toml

# Should only show workspace comments, not actual dependencies
# If you see active dependencies, they MUST be removed

# Build to find code issues
cargo build --all-targets --all-features
```

**Remember**: OptiRS is part of the SciRS2 ecosystem and MUST follow the strict dependency abstraction policy. Direct external dependencies are a CRITICAL architectural violation and must be fixed immediately.