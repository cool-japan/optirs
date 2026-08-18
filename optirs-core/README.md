# OptiRS Core

Core optimization algorithms and utilities for the OptiRS machine learning optimization library.

## Overview

OptiRS-Core provides the foundational optimization algorithms and mathematical utilities that power the entire OptiRS ecosystem. This crate integrates deeply with the SciRS2 scientific computing foundation and implements state-of-the-art optimization algorithms with high performance and numerical stability.

## Features

- **26 Optimizer Implementations**: 22 implement `optimizers::Optimizer` (first-order methods,
  quasi-Newton L-BFGS, and meta-learning optimizers MAML/Reptile/Meta-SGD/NTM). 4 more live in
  `second_order` as a separate family: Newton and a second, independent L-BFGS implement
  `SecondOrderOptimizer`; Newton-CG and K-FAC expose their own `step` API instead
- **100% SciRS2 Integration**: Built exclusively on SciRS2's scientific computing primitives
- **SIMD & Parallel**: SIMD-accelerated (`SimdSGD`) and multi-core (`parallel_optimizer`) paths
  via scirs2-core, measured by the `simd_benchmarks` / `parallel_benchmarks` Criterion suites
- **Performance Monitoring**: Built-in metrics via `optimizer_metrics`, optional
  `scirs2-metrics` integration behind the `metrics-integration` feature
- **Serialization**: Complete Serde support for checkpointing and model persistence
- **Memory Efficient**: Gradient accumulation, chunked processing for billion-parameter models
- **Federated Optimization**: `FedProxOptimizer` with proximal term (mu=0 degenerates to FedAvg)
- **Vision Transformer Support**: `ViTLayerDecay` scheduler for per-layer exponential LR decay
- **Attention-Aware Scheduling**: `AttentionAwareScheduler` - component-specific LR scaling for
  Transformer models
- **Gradient Flow Analysis**: `GradientFlowAnalyzer` with vanishing/exploding detection and SVG
  visualization
- **Loss Landscape Analysis**: 2D perturbation analysis, sharpness computation, saddle point
  detection
- **2202 Tests Passing**: library + integration tests (`cargo nextest run -p optirs-core
  --all-features`), plus 95 passing doc tests

## Optimization Algorithms

### Optimizers

Commonly used: SGD, Adam, AdamW, RMSprop, Adagrad, AdaDelta, AdaBound, LAMB, LARS, Lion, SAM,
RAdam, Ranger, Lookahead, SparseAdam, GroupedAdam, L-BFGS, Newton, Newton-CG, K-FAC, and the
meta-learning optimizers MAML, Reptile, Meta-SGD and NTM. `second_order` also has a second,
independent L-BFGS implementation (re-exported as `SecondOrderLBFGS`) alongside Newton.

See the crate documentation (`cargo doc -p optirs-core --open`) for the complete, categorized
list of all 26 optimizers.

### Advanced Features

- Learning rate scheduling and decay (`optirs_core::schedulers` - see the crate docs for the
  full list of schedulers)
- Gradient clipping and normalization
- Warm-up and cooldown strategies
- Numerical stability guarantees
- Memory-efficient implementations

## Dependencies

### Required Dependencies (SciRS2 Ecosystem)
- `scirs2-core` 0.6.5: Foundation scientific primitives (REQUIRED)
  - Provides: arrays, random, numeric traits, SIMD, parallel ops, GPU abstractions
- `scirs2-optimize` 0.6.5: Base optimization interfaces (REQUIRED)
- `scirs2-neural`: Required by specific modules (e.g. `neuromorphic`)
- `scirs2-stats`: Required by distribution-based regularizers

### Optional SciRS2 Dependencies
- `scirs2-metrics`: Behind the `metrics-integration` feature
- `scirs2-datasets`: Behind the `cross-platform-testing` feature

### External Dependencies
- `serde`, `serde_json`, `toml`: Serialization / config parsing
- `thiserror`: Error handling
- `chrono`, `sha2`, `oxicode`, `x25519-dalek`, `log`: checkpoint storage, hashing, and
  secure-aggregation support
- `rsa` (optional, behind the `crypto` feature): plugin signature verification
- Dev-only: `approx`, `criterion`, `tempfile` (testing and benchmarking)

**Note**: OptiRS does **NOT** use `scirs2-autograd`. OptiRS receives pre-computed gradients and does not perform automatic differentiation.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
optirs-core = "0.3.2"
scirs2-core = "0.6.5"  # Required foundation
```

### Basic Example

```rust
use optirs_core::optimizers::{Adam, Optimizer};
use scirs2_core::ndarray::Array1;  // ✅ CORRECT - Use scirs2_core

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an Adam optimizer
    let mut optimizer = Adam::new(0.001);

    // Your parameters and gradients
    let params = Array1::from(vec![1.0, 2.0, 3.0]);
    let grads = Array1::from(vec![0.1, 0.2, 0.3]);

    // `step` returns the updated parameters rather than mutating in place
    let params = optimizer.step(&params, &grads)?;
    println!("{params:?}");
    Ok(())
}
```

### With Learning Rate Scheduling

```rust
use optirs_core::optimizers::{Adam, Optimizer};
use optirs_core::schedulers::{ExponentialDecay, LearningRateScheduler};
use scirs2_core::ndarray::{Array1, Ix1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create optimizer with learning rate scheduler
    let mut optimizer = Adam::new(0.001);
    let mut scheduler = ExponentialDecay::new(0.001, 0.95, 1000);

    let mut params = Array1::from(vec![1.0, 2.0, 3.0]);
    let grads = Array1::from(vec![0.1, 0.2, 0.3]);

    for _ in 0..3 {
        // Update the learning rate from the schedule, then take a step.
        // `Optimizer` is generic over the array dimension `D`, so a turbofish is
        // needed on `set_learning_rate` (its own signature doesn't mention `D`).
        let current_lr = scheduler.step();
        Optimizer::<f64, Ix1>::set_learning_rate(&mut optimizer, current_lr);
        params = optimizer.step(&params, &grads)?;
    }
    println!("{params:?}");
    Ok(())
}
```

## Cargo Features

### Default Features
- `std`: Standard library support (enabled by default)

### Optional Features
- `cross-platform-testing`: Cross-platform compatibility testing (pulls in `scirs2-datasets`)
- `metrics-integration`: Re-exports `metrics::*` and pulls in `scirs2-metrics`
- `crypto`: Plugin signature verification (pulls in `rsa`)

Enable features in your `Cargo.toml`:

```toml
[dependencies]
optirs-core = { version = "0.3.2", features = ["cross-platform-testing"] }
```

**Note**: SIMD and parallel processing are built-in via scirs2-core and automatically enabled when beneficial.

## Architecture

OptiRS-Core is designed with modularity and performance in mind. Selected top-level modules
(see `src/lib.rs` for the complete list of 40 public modules):

```
optirs-core/
├── src/
│   ├── lib.rs                     # Public API and re-exports
│   ├── optimizers/                 # Optimizer implementations (sgd.rs, adam.rs, adamw.rs, ...)
│   ├── second_order/               # Newton, Newton-CG, K-FAC
│   ├── schedulers/                 # Learning rate scheduling
│   ├── distributed/                # Ring all-reduce, pipeline parallelism, elastic training
│   ├── privacy/                    # Differential privacy, secure aggregation
│   └── utils/                      # Mathematical utilities
```

## Performance

OptiRS-Core is optimized for high-performance machine learning workloads:

- **SIMD Acceleration**: via `scirs2_core::simd_ops` (`optimizers::SimdSGD`, `simd_optimizer`);
  measured by `benches/simd_benchmarks.rs`, no fixed speedup ratio is asserted in these docs
- **Parallel Processing**: via `scirs2_core::parallel_ops` (`parallel_optimizer`, non-wasm32
  targets); measured by `benches/parallel_benchmarks.rs`
- **GPU Support**: Backed by `scirs2_core::gpu` abstractions (`gpu_optimizer`)
- **Memory Efficient**: Gradient accumulation, chunked processing
- **Vectorized Operations**: Via scirs2_core::ndarray abstractions
- **Numerical Stability**: Validated on standard optimization benchmarks (Rosenbrock, Himmelblau)

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
let learning_rate = optimizer.get_learning_rate();

// ❌ Incorrect: camelCase or other formats
let gradientNorm = gradients.norm();
let parameterCount = model.parameter_count();
let learningrate = optimizer.get_learning_rate();
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

This project is licensed under the Apache License, Version 2.0.