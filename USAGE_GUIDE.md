# OptiRS Usage Guide

**Version:** 0.1.0-beta.2
**Status:** Production-Ready with SciRS2 Integration

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Optimization](#basic-optimization)
3. [Advanced Features](#advanced-features)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)
6. [SciRS2 Integration](#scirs2-integration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
optirs-core = "0.1.0-beta.2"
scirs2-core = "0.1.0-rc.1"  # Required - OptiRS foundation
```

### Your First Optimizer

```rust
use optirs_core::optimizers::{SGD, Optimizer};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create optimizer
    let mut optimizer = SGD::new(0.01);

    // Initial parameters
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    // Gradients (from your model)
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    // Perform optimization step
    let updated_params = optimizer.step(&params, &gradients)?;

    println!("Updated parameters: {:?}", updated_params);
    Ok(())
}
```

---

## Basic Optimization

### Available Optimizers

OptiRS provides 16 state-of-the-art optimizers:

#### First-Order Optimizers
- **SGD** - Stochastic Gradient Descent (with momentum & weight decay)
- **Adam** - Adaptive Moment Estimation
- **AdamW** - Adam with decoupled weight decay
- **RMSprop** - Root Mean Square Propagation
- **Adagrad** - Adaptive Gradient Algorithm

#### Advanced Optimizers
- **LAMB** - Layer-wise Adaptive Moments for Batch training
- **LARS** - Layer-wise Adaptive Rate Scaling
- **Lion** - Evolved Sign Momentum
- **Lookahead** - k steps forward, 1 step back
- **RAdam** - Rectified Adam
- **SAM** - Sharpness Aware Minimization
- **SparseAdam** - Adam for sparse gradients
- **GroupedAdam** - Adam for parameter groups

#### Second-Order Optimizers
- **L-BFGS** - Limited-memory BFGS
- **Newton** - Newton's method
- **K-FAC** - Kronecker-Factored Approximate Curvature

### SGD with Momentum

```rust
use optirs_core::optimizers::SGD;

// SGD with momentum and weight decay
let mut optimizer = SGD::new_with_config(
    0.01,  // learning rate
    0.9,   // momentum
    0.0001 // weight decay
);

// Training loop
for epoch in 0..100 {
    let updated = optimizer.step(&params, &gradients)?;
    params = updated;
}
```

### Adam Optimizer

```rust
use optirs_core::optimizers::Adam;

// Adam with default betas
let mut adam = Adam::new(0.001);

// Adam with custom configuration
let mut adam_custom = Adam::new_with_config(
    0.001,  // learning rate
    0.9,    // beta1 (first moment)
    0.999,  // beta2 (second moment)
    1e-8    // epsilon
);

// Optimization step
let updated = adam.step(&params, &gradients)?;
```

### AdamW (Recommended for Deep Learning)

```rust
use optirs_core::optimizers::AdamW;

// AdamW with weight decay
let mut adamw = AdamW::new_with_config(
    0.001,  // learning rate
    0.9,    // beta1
    0.999,  // beta2
    1e-8,   // epsilon
    0.01    // weight decay
);
```

---

## Advanced Features

### 1. Learning Rate Schedulers

```rust
use optirs_core::schedulers::{
    ExponentialDecay, CosineAnnealing, OneCycle, LearningRateScheduler
};

// Exponential decay
let mut scheduler = ExponentialDecay::new(0.1, 0.95);

for step in 0..1000 {
    let lr = scheduler.step();
    // Update optimizer learning rate
}

// Cosine annealing
let cosine = CosineAnnealing::new(0.1, 0.001, 1000);

// One-cycle policy (for super-convergence)
let one_cycle = OneCycle::new(
    0.001,  // base_lr
    0.1,    // max_lr
    1000,   // total_steps
    0.3     // pct_start
);
```

### 2. Parameter Groups

```rust
use optirs_core::parameter_groups::{ParameterGroup, ParameterGroupConfig};

// Different learning rates for different layers
let backbone_params = Array1::from_vec(vec![...]);
let head_params = Array1::from_vec(vec![...]);

let groups = vec![
    ParameterGroup::new(
        backbone_params,
        ParameterGroupConfig {
            learning_rate: Some(0.0001),  // Lower LR for backbone
            weight_decay: Some(0.01),
            ..Default::default()
        }
    ),
    ParameterGroup::new(
        head_params,
        ParameterGroupConfig {
            learning_rate: Some(0.001),  // Higher LR for head
            weight_decay: Some(0.0),
            ..Default::default()
        }
    ),
];
```

### 3. Regularization

```rust
use optirs_core::regularizers::{L1, L2, Dropout, ElasticNet};
use scirs2_core::ndarray::Array2;

// L2 regularization (weight decay)
let l2 = L2::new(0.01);
let regularized_grads = l2.apply(&params, &gradients)?;

// L1 regularization (sparse parameters)
let l1 = L1::new(0.01);

// Elastic Net (L1 + L2)
let elastic = ElasticNet::new(0.5, 0.01);  // alpha, lambda

// Dropout (for training)
let dropout = Dropout::new(0.5);  // 50% dropout rate
let dropped_params = dropout.apply(&params)?;
```

---

## Performance Optimization

### SIMD Acceleration (2-4x Speedup)

```rust
use optirs_core::simd_optimizer::{SimdOptimizer, should_use_simd};
use optirs_core::optimizers::SGD;

let mut optimizer = SGD::new(0.01);

// Automatic SIMD for f32/f64
if should_use_simd::<f32>(params.len()) {
    // SIMD will be used automatically for large arrays
    let updated = optimizer.step(&params, &gradients)?;
}
```

### Parallel Processing (4-8x Speedup)

```rust
use optirs_core::parallel_optimizer::{ParallelOptimizer, parallel_step_array1};
use optirs_core::optimizers::Adam;

// Optimize multiple parameter groups in parallel
let adam = Adam::new(0.001);
let mut parallel_opt = ParallelOptimizer::new(adam);

let params_groups = vec![params1, params2, params3, params4];
let grads_groups = vec![grads1, grads2, grads3, grads4];

// All groups optimized in parallel across CPU cores
let updated_groups = parallel_opt.step_parallel_groups(
    &params_groups,
    &grads_groups
)?;
```

### Memory-Efficient Optimization

```rust
use optirs_core::memory_efficient_optimizer::{
    GradientAccumulator, ChunkedOptimizer, MemoryUsageEstimator
};

// Gradient accumulation for large batch training
let mut accumulator = GradientAccumulator::<f32>::new(10000);

for micro_batch in micro_batches {
    accumulator.accumulate(&micro_batch.gradients.view())?;
}

let averaged_gradients = accumulator.average()?;

// Chunked optimization for billion-parameter models
let sgd = SGD::new(0.01);
let mut chunked_opt = ChunkedOptimizer::new(sgd, Some(1_000_000));

// Process in 1M parameter chunks
let updated = chunked_opt.step_chunked(&huge_params, &huge_gradients)?;

// Memory estimation
let memory_needed = MemoryUsageEstimizer::adam(
    1_000_000_000,  // 1B parameters
    4               // f32 (4 bytes)
);
println!("Adam needs {} GB", memory_needed / 1_000_000_000);
```

### GPU Acceleration (10-50x Speedup)

```rust
use optirs_core::gpu_optimizer::{GpuOptimizer, GpuConfig};
use optirs_core::optimizers::Adam;

// Create GPU-accelerated optimizer
let adam = Adam::new(0.001);
let config = GpuConfig {
    use_tensor_cores: true,
    use_mixed_precision: true,
    preferred_backend: Some("cuda".to_string()),
    ..Default::default()
};

let mut gpu_opt = GpuOptimizer::new(adam, config)?;

// Optimization runs on GPU automatically
let updated = gpu_opt.step(&params, &gradients)?;

// Check GPU status
println!("GPU available: {}", gpu_opt.is_gpu_available());
println!("Backend: {:?}", gpu_opt.gpu_backend());
```

---

## Production Deployment

### Metrics and Monitoring

```rust
use optirs_core::optimizer_metrics::{
    MetricsCollector, MetricsReporter, OptimizerMetrics
};
use std::time::{Duration, Instant};

// Create metrics collector
let mut collector = MetricsCollector::new();
collector.register_optimizer("adam");

// Training loop with metrics
for step in 0..1000 {
    let start = Instant::now();
    let params_before = params.clone();

    // Optimization step
    let updated = optimizer.step(&params, &gradients)?;

    // Collect metrics
    collector.update(
        "adam",
        start.elapsed(),
        0.001,  // learning rate
        &gradients.view(),
        &params_before.view(),
        &updated.view()
    )?;

    params = updated;
}

// Generate report
let report = collector.summary_report();
println!("{}", report);

// Export to JSON
let metrics = collector.get_metrics("adam").unwrap();
let json = MetricsReporter::to_json(metrics);
std::fs::write("metrics.json", json)?;
```

### Convergence Detection

```rust
use optirs_core::optimizer_metrics::OptimizerMetrics;

let metrics = OptimizerMetrics::new("sgd");

// Training loop
for epoch in 0..1000 {
    metrics.update_step(
        step_duration,
        learning_rate,
        &gradients.view(),
        &params_before.view(),
        &params_after.view()
    );

    // Check convergence
    if metrics.convergence.is_converging {
        println!("Converged at epoch {}", epoch);
        println!("Convergence rate: {:.4}", metrics.convergence.convergence_rate);
        break;
    }
}
```

### Profiling and Benchmarking

```rust
use std::time::Instant;

// Profile optimization step
let start = Instant::now();
let updated = optimizer.step(&params, &gradients)?;
let elapsed = start.elapsed();

println!("Step time: {:?}", elapsed);
println!("Throughput: {:.2} params/sec",
         params.len() as f64 / elapsed.as_secs_f64());

// Memory profiling
let memory_usage = MemoryUsageEstimator::estimate_peak_memory(
    params.len(),
    32,     // batch size
    512,    // sequence length
    4,      // f32
    "adam"
);
println!("Peak memory: {} MB", memory_usage / 1_000_000);
```

---

## SciRS2 Integration

OptiRS is built on **SciRS2-Core** and follows strict integration policies.

### Core Principles

1. **ALL scientific computing through scirs2_core**
2. **NO direct imports** from ndarray, rand, rayon, etc.
3. **Use SciRS2 abstractions** for arrays, random, parallel, SIMD

### Correct Usage Patterns

```rust
// ✅ CORRECT - Use scirs2_core
use scirs2_core::ndarray::{Array1, Array2, array};
use scirs2_core::random::{thread_rng, Normal};
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

// ❌ WRONG - Never use direct imports
// use ndarray::{Array1, Array2};     // FORBIDDEN
// use rand::thread_rng;               // FORBIDDEN
// use rayon::prelude::*;              // FORBIDDEN
```

### Array Operations

```rust
use scirs2_core::ndarray::{Array1, Array2, array, s};

// Create arrays
let arr = array![1.0, 2.0, 3.0, 4.0];
let mat = array![[1.0, 2.0], [3.0, 4.0]];

// Slicing
let slice = arr.slice(s![1..3]);

// Operations
let sum = &arr + &arr;
let product = &arr * 2.0;
```

### Random Number Generation

```rust
use scirs2_core::random::{thread_rng, Normal, Rng};

let mut rng = thread_rng();

// Normal distribution
let normal = Normal::new(0.0, 1.0)?;
let sample = normal.sample(&mut rng);

// Random array
let random_params = Array1::from_vec(
    (0..1000).map(|_| normal.sample(&mut rng)).collect()
);
```

### Parallel Operations

```rust
use scirs2_core::parallel_ops::*;

// Parallel iteration
let results: Vec<f64> = (0..10000)
    .into_par_iter()
    .map(|i| expensive_computation(i))
    .collect();
```

---

## Best Practices

### 1. Choose the Right Optimizer

| Task | Recommended Optimizer | Reason |
|------|----------------------|--------|
| Computer Vision | AdamW | Best for CNNs, good generalization |
| NLP/Transformers | AdamW or LAMB | Handles large batches well |
| Reinforcement Learning | Adam | Stable for non-stationary objectives |
| Fine-tuning | SGD with momentum | Better generalization on small data |
| Sparse Features | SparseAdam | Efficient for sparse gradients |
| Large-scale Training | LAMB or LARS | Layer-wise adaptation for huge batches |

### 2. Learning Rate Guidelines

```rust
// Start with these default values
let lr = match optimizer_type {
    "SGD" => 0.01,
    "Adam" | "AdamW" => 0.001,
    "RMSprop" => 0.001,
    "Adagrad" => 0.01,
    _ => 0.001,
};

// Use learning rate warmup for large models
let warmup_scheduler = LinearWarmupDecay::new(
    0.0,     // start_lr
    0.001,   // peak_lr
    0.0001,  // end_lr
    1000,    // warmup_steps
    10000    // total_steps
);
```

### 3. Gradient Clipping

```rust
use optirs_core::utils::{clip_gradient_norm, clip_gradients};

// Clip by norm (prevents exploding gradients)
let clipped = clip_gradient_norm(&gradients.view(), 1.0)?;

// Clip by value
let clipped = clip_gradients(&gradients.view(), -1.0, 1.0)?;
```

### 4. Parameter Initialization

```rust
use scirs2_core::random::{thread_rng, Normal};

fn xavier_init(size: usize, fan_in: usize) -> Array1<f64> {
    let mut rng = thread_rng();
    let std_dev = (2.0 / fan_in as f64).sqrt();
    let normal = Normal::new(0.0, std_dev).unwrap();

    Array1::from_vec(
        (0..size).map(|_| normal.sample(&mut rng)).collect()
    )
}
```

### 5. Checkpointing

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct CheckpointData {
    params: Vec<f64>,
    optimizer_state: Vec<f64>,
    epoch: usize,
    learning_rate: f64,
}

// Save checkpoint
fn save_checkpoint(
    path: &str,
    params: &Array1<f64>,
    optimizer_state: &Array1<f64>,
    epoch: usize,
    lr: f64
) -> Result<()> {
    let checkpoint = CheckpointData {
        params: params.to_vec(),
        optimizer_state: optimizer_state.to_vec(),
        epoch,
        learning_rate: lr,
    };
    let json = serde_json::to_string(&checkpoint)?;
    std::fs::write(path, json)?;
    Ok(())
}
```

---

## Troubleshooting

### Common Issues

#### 1. Exploding Gradients

**Symptoms:** Loss becomes NaN, parameters explode

**Solutions:**
```rust
// Use gradient clipping
let clipped_grads = clip_gradient_norm(&gradients.view(), 1.0)?;

// Reduce learning rate
let mut optimizer = Adam::new(0.0001);  // Smaller LR

// Use gradient centralization
use optirs_core::utils::gradient_centralization;
let centered = gradient_centralization(&gradients.view())?;
```

#### 2. Slow Convergence

**Symptoms:** Training loss decreases very slowly

**Solutions:**
```rust
// Increase learning rate
let mut optimizer = SGD::new(0.1);

// Use adaptive optimizer
let mut optimizer = Adam::new(0.001);

// Add momentum
let mut optimizer = SGD::new_with_config(0.01, 0.9, 0.0);

// Use learning rate warmup
let warmup = LinearWarmupDecay::new(0.0, 0.01, 0.001, 1000, 10000);
```

#### 3. Memory Issues

**Symptoms:** Out of memory errors

**Solutions:**
```rust
// Use gradient accumulation
let mut accumulator = GradientAccumulator::<f32>::new(param_size);
for micro_batch in micro_batches {
    accumulator.accumulate(&micro_batch.gradients.view())?;
}
let avg_grads = accumulator.average()?;

// Use chunked optimization
let chunked_opt = ChunkedOptimizer::new(optimizer, Some(1_000_000));

// Estimate memory first
let memory = MemoryUsageEstimator::adam(num_params, 4);
if memory > available_memory {
    eprintln!("Warning: Insufficient memory!");
}
```

#### 4. Poor Generalization

**Symptoms:** Training accuracy high, validation accuracy low

**Solutions:**
```rust
// Add weight decay
let adamw = AdamW::new_with_config(0.001, 0.9, 0.999, 1e-8, 0.01);

// Use dropout
let dropout = Dropout::new(0.5);

// Switch to SGD for fine-tuning
let sgd = SGD::new_with_config(0.01, 0.9, 0.0001);
```

---

## Additional Resources

- **API Documentation:** Run `cargo doc --open -p optirs-core`
- **Examples:** See `optirs-core/examples/`
- **Benchmarks:** See `optirs-core/benches/`
- **GitHub:** https://github.com/cool-japan/optirs
- **SciRS2 Docs:** https://github.com/cool-japan/scirs

## Contributing

OptiRS follows strict SciRS2 integration policies. See `SCIRS2_INTEGRATION_POLICY.md` for details.

## License

OptiRS is licensed under the same terms as the SciRS2 ecosystem.

---

**Happy Optimizing! 🚀**
