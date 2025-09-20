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

## Critical Dependencies

OptiRS **must** use SciRS2 as its foundation (see SCIRS2_INTEGRATION_POLICY.md):
- `scirs2-core` - Core scientific primitives (required) - **replaces direct rand and ndarray usage**
- `scirs2-optimize` - Base optimization interfaces (required)
- Additional SciRS2 crates added based on compilation evidence

SciRS2 is located at `../scirs/` relative to this project.

### FULL USE OF SciRS2-Core

OptiRS must make **FULL USE** of scirs2-core's extensive capabilities:

#### Core Array Operations (replaces ndarray)
```rust
// Use scirs2-core's ndarray extensions
use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray_ext::manipulation;  // flip, roll, tile, repeat
use scirs2_core::ndarray_ext::stats;         // mean, median, variance, correlation
use scirs2_core::ndarray_ext::matrix;        // eye, diag, kron
use scirs2_core::ndarray_ext::indexing;      // NumPy-like boolean masking
```

#### Random Number Generation (replaces rand)
```rust
use scirs2_core::random::{Random, rng, DistributionExt};
use scirs2_core::random::{QuasiMonteCarloSequence, SecureRandom};
use scirs2_core::random::{ImportanceSampling, VarianceReduction};
```

#### Performance Optimization Features
```rust
// SIMD acceleration
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

// Parallel processing
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy, LoadBalancer};
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// GPU acceleration
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel, CudaBackend, MetalBackend};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision, AutoTuning};
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

### Mandatory Usage Guidelines

1. **NEVER** import `ndarray` directly - use `scirs2_core::ndarray_ext`
2. **NEVER** import `rand` directly - use `scirs2_core::random`
3. **ALWAYS** use scirs2-core's SIMD operations for performance-critical code
4. **ALWAYS** use scirs2-core's GPU abstractions for hardware acceleration
5. **ALWAYS** use scirs2-core's memory management for large data operations
6. **ALWAYS** use scirs2-core's profiling and benchmarking tools
7. **ALWAYS** use scirs2-core's error types and result handling

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

## Key Implementation Patterns

1. **Error Handling**: Use `scirs2_core::error::CoreError` and `scirs2_core::Result`
2. **Array Operations**: Use `scirs2_core::ndarray_ext` exclusively
3. **Random Numbers**: Use `scirs2_core::random` exclusively
4. **Parallelization**: Use `scirs2_core::parallel` and `parallel_ops`
5. **Async Operations**: Use `tokio` with `scirs2_core::array_protocol::AsyncArray`
6. **GPU Backends**: Use `scirs2_core::gpu` abstractions
7. **SIMD Optimization**: Use `scirs2_core::simd` and `simd_ops`
8. **Memory Efficiency**: Use `scirs2_core::memory_efficient` for large data
9. **Profiling**: Use `scirs2_core::profiling` and `benchmarking`
10. **Metrics**: Use `scirs2_core::metrics` for monitoring

## OptiRS Module-Specific SciRS2 Usage

### optirs-core
- Use `scirs2_core::ndarray_ext` for all array operations
- Use `scirs2_core::simd_ops` for gradient processing
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

### Importing Core Types - FULL SciRS2 Usage
```rust
// Arrays and numerical operations
use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray_ext::stats::{mean, variance, correlation};
use scirs2_core::ndarray_ext::matrix::{eye, diag, kron};

// Random number generation
use scirs2_core::random::{Random, rng, DistributionExt};

// Performance features
use scirs2_core::simd::SimdArray;
use scirs2_core::parallel_ops::{par_chunks, par_join};
use scirs2_core::gpu::{GpuContext, GpuBuffer};

// Memory efficiency
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray};
use scirs2_core::memory::{BufferPool, GlobalBufferPool};

// Error handling
use scirs2_core::error::{CoreError, Result};

// Profiling and metrics
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::{Counter, Timer};
```

### Adding a New Optimizer with Full SciRS2 Integration
```rust
// optirs-core/src/optimizers/new_optimizer.rs
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::Random;
use scirs2_core::simd_ops::simd_dot_product;
use scirs2_core::parallel_ops::par_chunks;
use scirs2_core::memory::BufferPool;
use scirs2_core::metrics::Timer;
use scirs2_core::error::Result;

pub struct NewOptimizer {
    buffer_pool: BufferPool,
    timer: Timer,
    rng: Random,
}

impl NewOptimizer {
    pub fn step(&mut self, params: ArrayView2<f32>, grads: ArrayView2<f32>) -> Result<Array2<f32>> {
        // Use scirs2-core's SIMD operations
        let momentum = simd_dot_product(&params, &grads)?;

        // Use parallel processing
        par_chunks(&params, |chunk| {
            // Process in parallel
        });

        // Use memory-efficient operations
        let buffer = self.buffer_pool.acquire(params.len())?;

        // Track metrics
        self.timer.record("step_duration");

        Ok(updated_params)
    }
}
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

## Migration Checklist - Ensure Full SciRS2 Usage

When reviewing or writing OptiRS code, verify:

### ✅ Arrays and Numerical Operations
- [ ] NO direct `use ndarray::{...}`
- [ ] NO direct `Array`, `Array1`, `Array2` from ndarray
- [ ] YES `use scirs2_core::ndarray_ext::{Array, Array1, Array2, ...}`
- [ ] YES use scirs2-core's stats, matrix, manipulation modules

### ✅ Random Number Generation
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] YES `use scirs2_core::random::{Random, rng, ...}`
- [ ] YES use scirs2-core's distribution extensions

### ✅ Performance Optimization
- [ ] YES use `scirs2_core::simd` for vectorized operations
- [ ] YES use `scirs2_core::parallel_ops` for parallelization
- [ ] YES use `scirs2_core::gpu` for GPU acceleration
- [ ] YES use `scirs2_core::memory_efficient` for large datasets

### ✅ Production Features
- [ ] YES use `scirs2_core::error::{CoreError, Result}`
- [ ] YES use `scirs2_core::profiling` for performance analysis
- [ ] YES use `scirs2_core::metrics` for monitoring
- [ ] YES use `scirs2_core::benchmarking` for benchmarks

### ✅ Advanced Features
- [ ] YES use `scirs2_core::ml_pipeline` for ML pipelines
- [ ] YES use `scirs2_core::jit` for JIT compilation
- [ ] YES use `scirs2_core::cloud` for cloud storage
- [ ] YES use `scirs2_core::distributed` for distributed computing

### Common Anti-Patterns to Avoid
```rust
// ❌ WRONG - Direct dependencies
use ndarray::{Array2, arr2};
use rand::Rng;
use rand_distr::Normal;

// ✅ CORRECT - Full SciRS2 usage
use scirs2_core::ndarray_ext::{Array2, arr2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::Normal;
```

**Remember**: OptiRS is an extension of SciRS2, not a standalone project. It must leverage the full power of the SciRS2 ecosystem to provide advanced ML optimization capabilities.