# OptiRS Core

Core optimization algorithms and utilities for the OptiRS machine learning optimization library.

## Overview

OptiRS-Core provides the foundational optimization algorithms and mathematical utilities that power the entire OptiRS ecosystem. This crate integrates deeply with the SciRS2 scientific computing foundation and implements state-of-the-art optimization algorithms with high performance and numerical stability.

## Features

- **Core Optimizers**: SGD, Adam, AdamW, RMSprop with adaptive learning rates
- **SciRS2 Integration**: Built on top of SciRS2's scientific computing primitives
- **Automatic Differentiation**: Full integration with SciRS2's autograd system
- **Linear Algebra**: High-performance matrix operations via SciRS2-linalg
- **Performance Monitoring**: Built-in metrics and benchmarking via SciRS2-metrics
- **Serialization**: Complete Serde support for checkpointing and model persistence
- **Optional Features**: Parallelization with Rayon, SIMD acceleration

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

### Core Dependencies
- `ndarray`: N-dimensional arrays for tensor operations
- `serde`: Serialization and deserialization
- `thiserror`: Error handling
- `rand`: Random number generation

### SciRS2 Integration
- `scirs2-core`: Foundation scientific primitives
- `scirs2-optimize`: Base optimization interfaces
- `scirs2-linalg`: Matrix operations and linear algebra
- `scirs2-autograd`: Automatic differentiation
- `scirs2-neural`: Neural network optimization support
- `scirs2-metrics`: Performance monitoring and benchmarks

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-core = "0.1.0-beta.2"
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

### With SciRS2 Integration

```rust
use optirs_core::optimizers::Adam;
use scirs2_autograd::Variable;

// Create optimizer with SciRS2 autograd integration
let mut optimizer = Adam::new(0.001).with_autograd().build();

// Use with SciRS2 variables for automatic differentiation
let mut params = Variable::new(params_tensor);
optimizer.step_autograd(&mut params);
```

## Features

### Default Features
- `std`: Standard library support (enabled by default)

### Optional Features
- `parallel`: Enable Rayon-based parallelization
- `simd`: Enable SIMD acceleration with wide vectors

Enable features in your `Cargo.toml`:

```toml
[dependencies]
optirs-core = { version = "0.1.0-beta.1", features = ["parallel", "simd"] }
```

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

- Memory-efficient gradient updates
- Vectorized operations with ndarray
- Optional SIMD acceleration
- Zero-copy operations where possible
- Numerical stability guarantees

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