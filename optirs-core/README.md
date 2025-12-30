# OptiRS Core

Core optimization algorithms and utilities for the OptiRS machine learning optimization library.

## Overview

OptiRS-Core provides the foundational optimization algorithms and mathematical utilities that power the entire OptiRS ecosystem. This crate integrates deeply with the SciRS2 scientific computing foundation and implements state-of-the-art optimization algorithms with high performance and numerical stability.

## Features

- **19 Production-Ready Optimizers**: SGD, Adam, AdamW, RMSprop, AdaDelta, AdaBound, Ranger, LAMB, LARS, Lion, SAM, RAdam, Lookahead, L-BFGS, Newton-CG, and more
- **100% SciRS2 Integration**: Built exclusively on SciRS2's scientific computing primitives
- **High Performance**: SIMD acceleration, parallel processing, GPU support via scirs2-core
- **Linear Algebra**: High-performance matrix operations via scirs2-linalg
- **Performance Monitoring**: Built-in metrics and benchmarking via scirs2-metrics
- **Serialization**: Complete Serde support for checkpointing and model persistence
- **Memory Efficient**: Gradient accumulation, chunked processing for billion-parameter models

## Optimization Algorithms

### Supported Optimizers

- **SGD (Stochastic Gradient Descent)**: Classic optimizer with momentum and weight decay
- **Adam**: Adaptive moment estimation with bias correction
- **AdamW**: Adam with decoupled weight decay for better generalization
- **RMSprop**: Root Mean Square Propagation for adaptive learning rates

### Advanced Features

- Learning rate scheduling and decay
- Gradient clipping and normalization
- Warm-up and cooldown strategies
- Numerical stability guarantees
- Memory-efficient implementations

## Dependencies

### Required Dependencies (SciRS2 Ecosystem)
- `scirs2-core` 0.1.1: Foundation scientific primitives (REQUIRED)
  - Provides: arrays, random, numeric traits, SIMD, parallel ops, GPU abstractions
- `scirs2-optimize` 0.1.1: Base optimization interfaces (REQUIRED)

### Additional SciRS2 Dependencies
- `scirs2-neural`: Neural network optimization support
- `scirs2-metrics`: Performance monitoring and benchmarks
- `scirs2-stats`: Statistical analysis
- `scirs2-series`: Time series support
- `scirs2-datasets`: Dataset utilities (optional)
- `scirs2-linalg`: Linear algebra operations
- `scirs2-signal`: Signal processing

### External Dependencies
- `serde`, `serde_json`: Serialization
- `thiserror`, `anyhow`: Error handling
- `approx`, `criterion`: Testing and benchmarking

**Note**: OptiRS does **NOT** use `scirs2-autograd`. OptiRS receives pre-computed gradients and does not perform automatic differentiation.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-core = "0.1.0"
scirs2-core = "0.1.1"  # Required foundation
```

### Basic Example

```rust
use optirs_core::optimizers::{Adam, Optimizer};
use scirs2_core::ndarray::Array1;  // ✅ CORRECT - Use scirs2_core

// Create an Adam optimizer
let mut optimizer = Adam::new(0.001)
    .beta1(0.9)
    .beta2(0.999)
    .epsilon(1e-8)
    .build();

// Your parameters and gradients
let mut params = Array1::from(vec![1.0, 2.0, 3.0]);
let grads = Array1::from(vec![0.1, 0.2, 0.3]);

// Update parameters
optimizer.step(&mut params, &grads);
```

### With Learning Rate Scheduling

```rust
use optirs_core::optimizers::Adam;
use optirs_core::schedulers::{ExponentialDecay, LRScheduler};
use scirs2_core::ndarray::Array1;

// Create optimizer with learning rate scheduler
let mut optimizer = Adam::new(0.001);
let mut scheduler = ExponentialDecay::new(0.001, 0.95);

let mut params = Array1::from(vec![1.0, 2.0, 3.0]);
let grads = Array1::from(vec![0.1, 0.2, 0.3]);

// Update with scheduled learning rate
let current_lr = scheduler.step();
optimizer.set_learning_rate(current_lr);
optimizer.step(&mut params, &grads);
```

## Features

### Default Features
- `std`: Standard library support (enabled by default)

### Optional Features
- `cross-platform-testing`: Enable cross-platform compatibility testing (requires scirs2-datasets)

Enable features in your `Cargo.toml`:

```toml
[dependencies]
optirs-core = { version = "0.1.0", features = ["cross-platform-testing"] }
```

**Note**: SIMD and parallel processing are built-in via scirs2-core and automatically enabled when beneficial.

## Architecture

OptiRS-Core is designed with modularity and performance in mind:

```
optirs-core/
├── src/
│   ├── lib.rs              # Public API and re-exports
│   ├── optimizers/         # Optimizer implementations
│   │   ├── mod.rs
│   │   ├── sgd.rs
│   │   ├── adam.rs
│   │   ├── adamw.rs
│   │   └── rmsprop.rs
│   ├── schedulers/         # Learning rate scheduling
│   ├── utils/              # Mathematical utilities
│   └── integration/        # SciRS2 integration layer
```

## Performance

OptiRS-Core is optimized for high-performance machine learning workloads:

- **SIMD Acceleration**: 2-4x speedup via scirs2_core::simd_ops
- **Parallel Processing**: 4-8x speedup via scirs2_core::parallel_ops
- **GPU Support**: Multi-backend acceleration via scirs2_core::gpu
- **Memory Efficient**: Gradient accumulation, chunked processing
- **Vectorized Operations**: Via scirs2_core::ndarray abstractions
- **Zero-Copy Operations**: Where possible for maximum efficiency
- **Numerical Stability**: Validated on standard optimization benchmarks

## Development Guidelines

### Coding Standards

To ensure consistency across the OptiRS-Core codebase, all contributors must follow these guidelines:

#### Variable Naming
- **Always use `snake_case` for variable names** (e.g., `gradient_norm`, `parameter_count`, `learning_rate`)
- **Avoid camelCase or other naming conventions** (e.g., `gradientNorm` ❌, `parameterCount` ❌)
- **Use descriptive names** that clearly indicate the variable's purpose

```rust
// ✅ Correct: snake_case
let gradient_norm = gradients.norm();
let parameter_count = model.parameter_count();
let learning_rate = optimizer.learning_rate();

// ❌ Incorrect: camelCase or other formats
let gradientNorm = gradients.norm();
let parameterCount = model.parameter_count();
let learningrate = optimizer.learning_rate();
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

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS repository for contribution guidelines.

## License

This project is licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.