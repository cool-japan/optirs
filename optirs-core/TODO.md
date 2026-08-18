# OptiRS Core TODO (v0.3.3)

## Module Status: Pre-1.0 (0.3.x)

**Tests**: 2202 tests (library + integration, `cargo nextest run -p optirs-core --all-features`) +
95 doc tests, all passing
**Optimizers**: 26 total - 22 implement the `Optimizer` trait (`optirs_core::optimizers`); 4
more live in `optirs_core::second_order` as a separate family (Newton and a second, independent
L-BFGS implement `SecondOrderOptimizer`; NewtonCG and KFAC expose their own `step` API instead)
**SciRS2 Compliance**: 100% (no direct `ndarray`/`rand`/`rayon` dependency; see `Cargo.toml`)

---

## Vision & Goals
Build a state-of-the-art, production-ready optimization library for Rust that rivals PyTorch and TensorFlow optimizers in performance and exceeds them in memory safety and ergonomics.

---

## Completed: Core Optimizers

### First-Order Optimizers (17 total)
- [x] **SGD** - Stochastic Gradient Descent
  - [x] Basic SGD with learning rate
  - [x] Classical momentum (Polyak)
  - [x] Nesterov accelerated gradient (NAG)
  - [x] Weight decay integration
  - [x] Learning rate scheduling support
  - [x] Gradient centralization option

- [x] **SIMD SGD** - SIMD-accelerated SGD
  - [x] Vectorized update path for large parameter arrays (`benches/simd_benchmarks.rs`
    compares it against scalar SGD; no fixed speedup ratio is asserted)
  - [x] Automatic SIMD threshold detection

- [x] **Adam** - Adaptive Moment Estimation
  - [x] Basic Adam algorithm (beta1=0.9, beta2=0.999, epsilon=1e-8)
  - [x] Bias correction for first and second moments
  - [x] Numerical stability with epsilon clipping
  - [x] Memory-efficient implementation

- [x] **AdamW** - Adam with Decoupled Weight Decay
  - [x] Decouple weight decay from gradient-based updates
  - [x] Performance optimization with vectorized operations
  - [x] Support for different weight decay schedules

- [x] **RMSprop** - Root Mean Square Propagation
  - [x] Basic RMSprop implementation
  - [x] Centered variant with mean centering
  - [x] Momentum integration

- [x] **AdaGrad** - Adaptive Gradient Algorithm
  - [x] Basic AdaGrad with accumulator
  - [x] Diagonal approximation for memory efficiency

- [x] **AdaDelta** - Adaptive learning rate without manual tuning
  - [x] Automatic step size adaptation using RMS
  - [x] 10-step warmup boost for cold-start
  - [x] Full convergence validation (13 tests)

- [x] **AdaBound** - Dynamic bounds converging to SGD
  - [x] Dynamic learning rate bounds
  - [x] Smooth transition from adaptive to SGD
  - [x] AMSBound variant support
  - [x] Final learning rate convergence

- [x] **LAMB** - Layer-wise Adaptive Moments
  - [x] Layer-wise adaptation mechanism
  - [x] Trust ratio computation
  - [x] Large batch optimization (batch size > 16K)
  - [x] Mixed precision support

- [x] **LARS** - Layer-wise Adaptive Rate Scaling
  - [x] Layer-wise learning rate adaptation
  - [x] Trust ratio computation

- [x] **RAdam** - Rectified Adam
  - [x] Variance rectification term
  - [x] Automated warmup scheduling
  - [x] Convergence guarantees

- [x] **Lookahead** - Slow/Fast Weight Updates
  - [x] Dual optimizer state management
  - [x] Interpolation mechanism
  - [x] Compatibility wrapper for any base optimizer

- [x] **Ranger** - RAdam + Lookahead combination
  - [x] RAdam + Lookahead combination
  - [x] Variance rectification + trajectory smoothing
  - [x] Proper slow/fast weight synchronization
  - [x] 8 comprehensive tests

- [x] **Lion** - Evolved Sign Momentum
  - [x] Sign-based updates
  - [x] Memory-efficient (no second moment)

- [x] **SAM** - Sharpness Aware Minimization
  - [x] Sharpness-aware perturbation
  - [x] Better generalization characteristics

- [x] **SparseAdam** - Sparse Gradient Support
  - [x] Efficient sparse tensor handling
  - [x] High-dimensional problem optimization

- [x] **GroupedAdam** - Parameter Group Support
  - [x] Different hyperparameters per group
  - [x] Layer-wise configuration

### Second-Order / Quasi-Newton Methods (5 total)
- [x] **L-BFGS** (`optimizers::LBFGS`, implements `Optimizer`) - Limited-memory BFGS
  - [x] Two-loop recursion algorithm
  - [x] Backtracking Armijo line search (`LBFGS::step_with_loss`) - enforces the sufficient-decrease
    (`c1`) condition only; the stored `c2` curvature parameter is not enforced by the current
    backtracking search (see the doc comment on `LBFGS::c2` in `optimizers/lbfgs.rs`), so this
    is Armijo, not full Wolfe
  - [x] Memory-efficient history management
  - [x] Configurable memory size

- [x] **SecondOrderLBFGS** (`second_order::LBFGS`, re-exported as `SecondOrderLBFGS`,
  implements `SecondOrderOptimizer`) - a separate, simpler L-BFGS implementation from the one
  above; no line search, fixed step scaled by the two-loop recursion direction

- [x] **Newton** (`second_order::Newton`, implements `SecondOrderOptimizer`) - Diagonal Newton
  step
  - [x] Uses `|h_ii|` floored at a configurable minimum curvature as the step denominator, so
    the update stays a descent direction under negative or vanishing curvature

- [x] **Newton-CG** (`second_order::NewtonCG`, own `step`/`step_with_loss` API - does not
  implement `SecondOrderOptimizer`) - Newton Conjugate Gradient
  - [x] Conjugate gradient solver for Newton system
  - [x] O(n) memory using only Hessian-vector products
  - [x] Trust region control
  - [x] Negative curvature detection
  - [x] 7 comprehensive tests

- [x] **K-FAC** (`second_order::KFAC`, own `step` API - does not implement
  `SecondOrderOptimizer`) - Kronecker-Factored Approximate Curvature
  - [x] Kronecker-factored Fisher information approximation, per-layer covariance matrices
  - [x] Self-contained Gauss-Jordan matrix inverse with partial pivoting + Tikhonov damping
    (`second_order/kfac/utils::general_matrix_inverse`)

---

## Completed: SciRS2 Integration

- [x] **Full SciRS2-Core Integration** - 100% complete
- [x] **Array Operations** - All ndarray imports via scirs2_core::ndarray
- [x] **Random Generation** - All rand imports via scirs2_core::random
- [x] **Error Handling** - Integrated scirs2_core::error types
- [x] **Namespace Correction** - Fixed all scirs2_optim references
- [x] **Compilation Fixes** - Resolved all SciRS2 policy violations

---

## Completed: Advanced Features

### Mathematical Utilities
- [x] **Gradient Processing**
  - [x] Gradient clipping (by norm: L2, L-inf, and by value)
  - [x] Gradient normalization (layer-wise and global)
  - [x] Gradient accumulation with overflow prevention
  - [x] Gradient centralization (zero-mean gradients)

- [x] **Numerical Stability**
  - [x] Overflow/underflow prevention
  - [x] NaN/Inf detection
  - [x] Mixed precision training support (FP16/BF16/FP32)

### Learning Rate Scheduling (`optirs_core::schedulers`)
- [x] `ConstantScheduler`, `ExponentialDecay`, `LinearDecay`, `StepDecay`
- [x] `CosineAnnealing`, `CosineAnnealingWarmRestarts`
- [x] `LinearWarmupDecay`, `OneCycle`, `CyclicLR`
- [x] `ReduceOnPlateau`
- [x] `CurriculumScheduler`, `NoiseInjectionScheduler` — `CurriculumScheduler::new` panics on
  an empty stage list; this is the one remaining production `panic!` in this crate (every other
  panic path was converted to a typed error during 0.3.2) and needs a fallible constructor to
  close (see workspace `TODO.md`)
- [x] `AttentionAwareScheduler`, `ViTLayerDecay`
- [x] `CustomScheduler`, `CombinedScheduler`, `SchedulerBuilder` (compose schedules from closures)
- [ ] Multi-step and polynomial decay - **not implemented**; no `MultiStepDecay` or
  `PolynomialDecay` type exists in `schedulers/` (checked off in a previous revision of this
  file without a matching implementation)

### Performance Optimization
- [x] **SIMD Acceleration**
  - [x] SIMD-optimized mathematical operations (`simd_optimizer`, `optimizers::SimdSGD`)
  - [x] Platform-specific optimizations via `scirs2_core::simd_ops`
  - [x] Measured by `benches/simd_benchmarks.rs` (no fixed speedup ratio committed to docs)

- [x] **Parallel Processing**
  - [x] Parallel gradient updates (`parallel_optimizer`, non-wasm32 targets)
  - [x] Thread-safe optimizer state management
  - [x] Measured by `benches/parallel_benchmarks.rs` (no fixed speedup ratio committed to docs)

### Memory Efficiency
- [x] In-place operations with mutation tracking
- [x] Memory pool for temporary calculations
- [x] Gradient accumulation for large models
- [x] Chunked processing for billion-parameter models

---

## Completed: Extended Features (v0.3.1 - v0.3.2)

Originally tracked here as forward-looking work under the heading "Future Work"; every item
below has since shipped. See `CHANGELOG.md` for the full 0.3.2 change list across the workspace.

### Meta-Learning Optimizers
- [x] MAML (Model-Agnostic Meta-Learning) support — SecondOrder/FirstOrder/Reptile variants (`optirs-core/src/optimizers/maml.rs`, 14 tests)
- [x] Reptile optimizer
- [x] Meta-SGD with learnable learning rates
- [x] Neural optimizer implementations — NTM-style memory-augmented optimizer with external memory, content/location/hybrid attention addressing, erase+add writes (`optirs-core/src/optimizers/ntm_optimizer.rs`; 23 tests)

### Additional Regularization
- [x] Spectral normalization
- [x] Weight standardization
- [x] Group Lasso and structured sparsity

### Distributed & Federated Learning
- [x] FedAvg implementation (FedProx with mu=0 degenerates to FedAvg)
- [x] FedProx with proximal term (distributed/fedprox.rs)
- [x] Differential privacy integration — Rényi DP accountant with tight subsampled-Gaussian composition (`optirs-core/src/privacy/renyi_accountant.rs`; 29 tests)
- [x] Secure aggregation protocols — Bonawitz-style pairwise additive masking with quantization, modular arithmetic, dropout reconstruction (`optirs-core/src/privacy/secure_aggregation.rs`; 32 tests)
- [x] Bandwidth-optimal ring all-reduce + all-gather — segmented reduce-scatter / all-gather collective with a `CollectiveTransport` trait and in-memory `LocalTransport`, reduce ops sum/mean/max/min/product (`optirs-core/src/distributed/ring_allreduce.rs`; 14 tests) (2026-06-24)
- [x] Pipeline parallelism (GPipe + 1F1B) — micro-batch schedulers, exact DP minimax stage partitioning, bubble-fraction / utilization / activation-stash / makespan metrics (`optirs-core/src/distributed/pipeline_parallel.rs`; 13 tests) (2026-06-24)
- [x] Elastic training — dynamic world-size join/leave state machine, block + rendezvous (HRW) data re-sharding, linear LR scaling on resize, epoch snapshots (`optirs-core/src/distributed/elastic.rs`; 13 tests) (2026-06-24)

### Developer Experience
- [x] Gradient flow visualization (GradientFlowAnalyzer with SVG output, vanishing/exploding detection)
- [x] Loss landscape visualization (LossLandscapeAnalyzer: 2D perturbation, sharpness, saddle point detection)
- [x] Hyperparameter sensitivity analysis (Sobol, Morris, OAT — `optirs-core/src/sensitivity_analysis/`)

### Quantum-Inspired Optimizers (v0.3.2)
- [x] Quantum annealing with tunneling kernel (`optirs-core/src/quantum_inspired/annealing.rs`)
- [x] Variational quantum optimizer with SPSA gradients (`optirs-core/src/quantum_inspired/vqe.rs`)
- [x] Hybrid quantum-classical two-phase optimizer (`optirs-core/src/quantum_inspired/hybrid.rs`)

### Domain-Specific Optimizers
- [x] Vision-specific (ViTLayerDecay scheduler: per-layer exponential LR decay for Vision Transformers)
- [x] NLP-specific (AttentionAwareScheduler: component-specific LR scaling for Transformer models)
- [x] RL-specific (PPO, TRPO variants) — `reinforcement_learning` module now registered in lib.rs and live: PPO (clip + adaptive-KL), TRPO with real empirical Fisher-vector product + conjugate gradient + line search, A2C/A3C actor-critic, natural policy gradients. (2026-06-14)

### v0.3.2 correctness fixes (2026-06-14)
- [x] **KFAC matrix inversion** (`second_order/kfac/`): `compute_matrix_inverse` previously returned `Array2::eye(n)` (identity, ignoring the input matrix) and the natural-gradient path used a diagonal-only approximation for n>3 — both silently reduced the natural gradient to a plain gradient. Replaced with a real self-contained Gauss-Jordan inversion with partial pivoting + Tikhonov damping for singular factors, shared via `kfac/utils::general_matrix_inverse`. Returns `Err` on truly singular input rather than a silent identity. 12+ new tests (A·inv(A)≈I, known inverses, pivoting, damping, not-identity regression).
- [x] **RL natural-gradient parameter update** (`reinforcement_learning/natural_gradients.rs`): `update_policy_parameters` was a no-op; now maps the flat natural-gradient update onto the policy's named parameters (sorted-key split, dimension validation) and applies it via the policy network. 2 new tests.

---

## Testing Status

### Coverage
- [x] Unit tests across all 26 optimizer implementations
- [x] Convergence tests (Rosenbrock, Himmelblau)
- [x] Numerical stability tests
- [x] Edge case handling tests
- [x] Performance regression tests

### Test Count
```
2202 tests passing (library + integration, cargo nextest run -p optirs-core --all-features)
95 doc tests passing
```
Re-measure with the commands above before quoting a count elsewhere - this crate's test suite
grows quickly and a hardcoded number goes stale fast.

---

## Design Principles Followed

- **Zero-cost abstractions**: No runtime overhead for unused features
- **Memory safety**: Leveraging Rust's ownership system
- **Ergonomics**: PyTorch/TensorFlow-like API
- **Performance**: Optimized for both throughput and latency
- **Correctness**: Extensive testing and validation

---

**Status**: Pre-1.0 (0.3.x) - public API may still change between 0.x releases
**Version**: v0.3.3
