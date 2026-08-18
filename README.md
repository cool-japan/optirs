# OptiRS - Advanced ML Optimization Built on SciRS2

**Version:** 0.3.2
**License:** Apache-2.0

OptiRS is a machine-learning optimization library for Rust, built on the
[SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem. It provides
optimizers, learning-rate schedulers, regularizers and gradient tooling in `optirs-core`,
plus separate crates for GPU acceleration, TPU-style coordination, learned optimizers,
neural architecture search, benchmarking and WebAssembly bindings.

## Built on SciRS2-Core

OptiRS does not depend on `ndarray`, `rand`, `rayon`, `num-traits` or any other library
that `scirs2-core` already abstracts. Every array, RNG, numeric-trait, SIMD and
parallelism operation goes through `scirs2-core`:

| Concern | Path used |
|---|---|
| Arrays | `scirs2_core::ndarray` (and `ndarray_ext`) |
| Random numbers | `scirs2_core::random` |
| Numeric traits | `scirs2_core::numeric` |
| SIMD | `scirs2_core::simd_ops` |
| Parallelism | `scirs2_core::parallel_ops` |
| GPU abstractions | `scirs2_core::gpu` |
| Metrics | `scirs2_core::metrics` |

This is a hard rule, enforced by review — see
[`SCIRS2_INTEGRATION_POLICY.md`](SCIRS2_INTEGRATION_POLICY.md). It applies to scientific
computing only; ordinary infrastructure crates (`serde`, `thiserror`, `tokio`, `toml`,
`sha2`, `oxicode`, the GPU backend crates, `wasm-bindgen`) are used directly.

### SciRS2 dependencies

**Required (all crates):**

- `scirs2-core` 0.6.5 — arrays, random, numeric traits, SIMD, parallel, GPU abstractions
- `scirs2-optimize` 0.6.5 — base optimization interfaces (`optirs-core`)

**Also required by `optirs-core`:**

- `scirs2-neural` 0.6.5 — used by `neuromorphic::spike_based`
- `scirs2-stats` 0.6.5 — distributions and statistical functions

**Optional, feature-gated in `optirs-core`:**

- `scirs2-metrics` 0.6.5 — behind the `metrics-integration` feature
- `scirs2-datasets` 0.6.5 — behind the `cross-platform-testing` feature

**Not used:** `scirs2-autograd` (OptiRS consumes pre-computed gradients rather than
computing them), `scirs2-optim` (superseded by `optirs-core`), and `scirs2-linalg`,
`scirs2-signal`, `scirs2-series`, which were removed in 0.3.2 after an audit found no
call sites.

## Workspace layout

```
optirs/                # facade crate re-exporting the others behind feature gates
├── optirs-core/       # optimizers, schedulers, regularizers, privacy, streaming, plugins
├── optirs-gpu/        # GPU acceleration (wgpu / Metal / OpenCL / CUDA scaffolding)
├── optirs-tpu/        # TPU-style coordination and an XLA-shaped compiler (CPU reference)
├── optirs-learned/    # learned optimizers and meta-learning
├── optirs-nas/        # neural architecture search
├── optirs-bench/      # benchmarking, profiling and regression detection
└── optirs-wasm/       # WebAssembly bindings
```

## Features

### `optirs-core` — stable

**Optimizers.** SGD (with momentum / Nesterov), SimdSGD, Adam, AdamW, AdaDelta, AdaBound,
Adagrad, RMSprop, LAMB, LARS, Lion, Lookahead, RAdam, Ranger, SAM, SparseAdam,
GroupedAdam, MAML, MetaSGD, Reptile and an NTM-style optimizer, all exported from
`optirs_core::optimizers`; L-BFGS, Newton, Newton-CG and K-FAC from
`optirs_core::second_order`; FedProx from `optirs_core::distributed`. See the
`optirs-core` API docs for the authoritative roster.

**Learning-rate schedulers.** `ConstantScheduler`, `ExponentialDecay`, `LinearDecay`,
`StepDecay`, `CosineAnnealing`, `CosineAnnealingWarmRestarts`, `CyclicLR`, `OneCycle`,
`LinearWarmupDecay`, `ReduceOnPlateau`, `NoiseInjectionScheduler`, `CurriculumScheduler`,
`CustomScheduler`/`CombinedScheduler`, plus `ViTLayerDecay` and `AttentionAwareScheduler`
for transformer training.

**Gradient and loss analysis.** `gradient_flow` records gradient propagation through
layers and detects vanishing/exploding gradients; `loss_landscape` performs 2-D
perturbation analysis with sharpness and saddle-point detection. Both can emit SVG.

**Performance paths.**

- *SIMD* — `SimdSGD` and the `simd_optimizer` helpers use
  `scirs2_core::simd_ops::SimdUnifiedOps`, activating above a per-type element threshold.
- *Parallel* — `parallel_optimizer::parallel_step_array1` and the `ParallelOptimizer`
  wrapper distribute parameter groups across cores via `scirs2_core::parallel_ops`.
- *Memory-efficient* — gradient accumulation and chunked parameter processing for models
  that do not fit comfortably in RAM.
- *GPU* — `gpu_optimizer` provides context management and host/device transfer built on
  `scirs2_core::gpu`.

Speedups depend entirely on array size, element type and hardware. Measure them on your
own workload with the benchmarks below rather than trusting a headline number.

**Metrics and monitoring.** `optimizer_metrics::MetricsCollector` tracks per-step
duration, learning rate, gradient statistics (mean, standard deviation, norm, sparsity)
and parameter-update statistics, with convergence detection and JSON/CSV export through
`MetricsReporter`.

**Also in `optirs-core`.** Differentially private and federated optimization (including a
real Bonawitz secure-aggregation protocol), streaming/online optimization with statistical
drift and anomaly detection, checkpointed coordination with a filesystem-backed store, a
plugin system with TOML manifests, hardware-aware tuning, and quantum-inspired and
neuromorphic experiments.

### `optirs-gpu` — partial hardware coverage

Read this section as a status report, not a feature list:

- **Metal** — real compute shaders (MSL pipelines, buffers, dispatch, readback) run Adam,
  AdamW, SGD, RMSprop, Adagrad and LAMB end to end.
- **WebGPU** — WGSL kernels are implemented but currently blocked on an upstream
  `scirs2-core` adapter-probe bug.
- **OpenCL** — context creation only; no kernels shipped.
- **CUDA / ROCm** — no backend (`scirs2-core` 0.6.x dropped its CUDA backend).
- **Tensor cores** — real mixed-precision tiled GEMM on the `wgpu` path.
- **Memory management** — the arena/buddy/slab allocators are CPU-side models of GPU
  memory pools; the vendor memory backends are host-memory API-shape simulations whose
  copy functions move zero bytes, and each file says so at the top.
- **Multi-GPU** — single-device reduction kernels work; true cross-device collectives
  return an explicit `UnsupportedOperation` error rather than a fabricated result.

### `optirs-tpu` — CPU reference implementation

No vendor TPU runtime is linked (it is proprietary and not distributable as pure Rust).
Every path runs and is tested on the CPU executor and returns an explicit error where real
TPU silicon would be required:

- Pod management: device/channel topology, barrier synchronization, load balancing, fault
  detection
- An XLA-shaped compiler: graph builder, dead-code elimination, constant folding,
  common-subexpression elimination, kernel-fusion legality checks, a real allocator, shape
  inference
- Fault tolerance: checkpoints serialized with a SHA-256 integrity hash, verified on
  restore
- Collectives: ring all-reduce, broadcast, reduce-scatter

### `optirs-learned` — research-grade

Transformer- and LSTM-based learned optimizers (with real truncated BPTT meta-training and
seeded, reproducible initialization), meta-learning across tasks, few-shot learning
(prototypical networks, fast adaptation, episodic memory), continual learning (EWC,
progressive networks), online MAML, cross-domain transfer, and CV/NLP/attention
domain-specific optimizers. Implementations are real and tested; APIs may still change.

### `optirs-nas` — research-grade

Random, evolutionary, reinforcement-learning, Bayesian and differentiable (DARTS,
PC-DARTS, RobustDARTS) search strategies; multi-objective optimization with exact
hypervolume, NSGA-II and MOEA/D; progressive search; hardware-aware search; architecture
embedding; and real grid / TPE / surrogate-based hyperparameter search. Implementations are
real and tested; APIs may still change.

### `optirs-bench`

Criterion-based benchmarking, memory profiling and leak-report parsing, regression
detection, cross-platform orchestration (local, Docker and SSH execution, with an explicit
error when the runtime is absent) and a security auditor.

### `optirs-wasm`

`wasm-bindgen` bindings for the core optimizers, plus WebGPU adapter detection. Running
WGSL compute kernels from the WASM bindings is documented as not implemented.

## Quick start

```toml
[dependencies]
optirs-core = "0.3.2"
scirs2-core = "0.6.5"  # required foundation
```

Or through the facade crate, which gates the extension crates behind features:

```toml
[dependencies]
optirs = { version = "0.3.2", features = ["gpu", "bench"] }
```

### Basic usage

```rust
use optirs_core::optimizers::{Adam, Optimizer};
// Always use scirs2_core for arrays - never ndarray directly.
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.08]);

    let mut optimizer = Adam::new(0.001);
    let updated_params = optimizer.step(&params, &gradients)?;

    println!("Updated parameters: {:?}", updated_params);
    Ok(())
}
```

### SIMD-accelerated SGD

```rust
use optirs_core::optimizers::{Optimizer, SimdSGD};
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = Array1::from_elem(100_000, 1.0f32);
    let grads = Array1::from_elem(100_000, 0.001f32);

    let mut optimizer = SimdSGD::new(0.01f32);
    let updated = optimizer.step(&params, &grads)?;

    println!("Optimized {} parameters", updated.len());
    Ok(())
}
```

### Parallel parameter groups

```rust
use optirs_core::optimizers::Adam;
use optirs_core::parallel_optimizer::parallel_step_array1;
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut optimizer = Adam::new(0.001);
    let updated_list = parallel_step_array1(&mut optimizer, &params_list, &grads_list)?;

    println!("Optimized {} parameter groups", updated_list.len());
    Ok(())
}
```

### Metrics collection

```rust
use optirs_core::optimizer_metrics::{MetricsCollector, MetricsReporter};
use optirs_core::optimizers::{Adam, Optimizer};
use scirs2_core::ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collector = MetricsCollector::new();
    collector.register_optimizer("adam");

    let mut optimizer = Adam::new(0.001);
    let mut params = Array1::from_elem(1000, 1.0);
    let grads = Array1::from_elem(1000, 0.01);

    for _ in 0..100 {
        let params_before = params.clone();

        let start = Instant::now();
        params = optimizer.step(&params, &grads)?;
        let duration = start.elapsed();

        collector.update(
            "adam",
            duration,
            0.001,
            &grads.view(),
            &params_before.view(),
            &params.view(),
        )?;
    }

    println!("{}", collector.summary_report());

    if let Some(metrics) = collector.get_metrics("adam") {
        println!("{}", MetricsReporter::to_json(metrics));
    }

    Ok(())
}
```

## Examples

The facade crate carries two end-to-end examples:

```bash
cargo run -p optirs --example basic_optimization --release
cargo run -p optirs --example scirs2_integration_demo --release
```

`optirs-core/examples/` holds a larger set of focused ones — schedulers, gradient
clipping, parameter groups, regularization, L-BFGS, LAMB, Lion, memory-efficient
optimization and production monitoring among them:

```bash
cargo run -p optirs-core --example advanced_optimization --release
cargo run -p optirs-core --example production_monitoring --release
```

(`optirs-core/examples/broken/` is a quarantine directory for examples that have not been
brought up to the current API; Cargo does not build it.)

## Benchmarks

`optirs-core` ships six Criterion benchmark targets:

| Target | Measures |
|---|---|
| `optimizer_benchmarks` | SGD, SGD+momentum, Adam and AdamW across parameter sizes, cold start, and 100-step convergence loops |
| `simd_benchmarks` | SIMD versus scalar optimizer steps |
| `parallel_benchmarks` | Multi-core parameter-group scaling |
| `memory_efficient_benchmarks` | Gradient accumulation and chunked processing |
| `gpu_benchmarks` | GPU versus CPU paths |
| `metrics_benchmarks` | Metrics-collection overhead |

```bash
cargo bench -p optirs-core
```

No benchmark numbers are published here: they are hardware-dependent, and the only honest
figure is the one you measure on your own machine.

## Building and testing

```bash
# Build everything
cargo build --workspace --all-features

# Lint (the workspace is kept at zero warnings)
cargo clippy --workspace --all-features --all-targets

# Test
cargo test --workspace --all-features
# or, faster:
cargo nextest run --workspace --all-features

# Documentation
cargo doc --workspace --all-features --no-deps --open

# Dependency policy (banned crates)
cargo deny check bans
```

### Project status (measured 2026-08-18)

- 907 Rust source files, ~342k lines of code (`tokei`, excluding `target/`)
- More than 4,200 unit and integration tests passing, plus doc tests
- Zero `rustc` and zero `clippy` warnings across the workspace with `--all-features
  --all-targets`
- No source file at or above 2,000 lines
- `cargo deny check bans` passes

Run the commands above to reproduce any of these.

## Documentation

- [`CHANGELOG.md`](CHANGELOG.md) — release notes
- [`SCIRS2_INTEGRATION_POLICY.md`](SCIRS2_INTEGRATION_POLICY.md) — the dependency rules
  above, in full
- [`USAGE_GUIDE.md`](USAGE_GUIDE.md) — extended usage guide
- [`MIGRATION_FROM_SCIRS2.md`](MIGRATION_FROM_SCIRS2.md) — for users coming from
  `scirs2-optim`
- API documentation: `cargo doc --open --no-deps`, or [docs.rs/optirs](https://docs.rs/optirs)

Each crate also carries its own `README.md` and `TODO.md`.

## Platform support

Everything except the GPU backends is portable pure Rust, so `optirs-core`,
`optirs-tpu`, `optirs-learned`, `optirs-nas` and `optirs-bench` behave the same on Linux,
macOS and Windows. The hardware-specific differences are:

| Crate | Linux | macOS | Windows |
|---|---|---|---|
| `optirs-gpu` (Metal) | ❌ | ✅ real compute | ❌ |
| `optirs-gpu` (WebGPU) | 🚧 blocked upstream | 🚧 blocked upstream | 🚧 blocked upstream |
| `optirs-gpu` (OpenCL) | 🚧 context only | 🚧 context only | 🚧 context only |
| `optirs-gpu` (CUDA / ROCm) | ❌ no backend | ❌ no backend | ❌ no backend |

For `wasm32-unknown-unknown`, use `optirs-wasm`; see its own README for what is exposed.

## Contributing

Contributions are welcome. Before submitting:

1. `cargo fmt`
2. `cargo clippy --workspace --all-features --all-targets` — must be warning-free
3. `cargo test --workspace --all-features`
4. `cargo deny check bans`

### Coding standards

- **SciRS2 first.** Use `scirs2_core::ndarray`, `scirs2_core::random`,
  `scirs2_core::numeric`, `scirs2_core::simd_ops` and `scirs2_core::parallel_ops` rather
  than `ndarray`, `rand`, `num-traits`, `wide` or `rayon`. You can check this with:

  ```bash
  grep -rn "^use ndarray::" --include='*.rs' .   # must return nothing
  grep -rn "^use rand::"    --include='*.rs' .   # must return nothing
  ```

- **No panics on recoverable conditions.** No `unwrap()`, `panic!()`, `todo!()` or
  `unimplemented!()` in production code — return an `OptimError` instead. `expect()` is
  admissible only where the signature genuinely cannot return an error (a `Default`
  implementation converting a constant, for example), and its message must name the
  invariant that is being relied on.
- **No fabricated results.** A code path that cannot compute something must return an
  error naming what is missing, never a plausible-looking constant.
- **Naming.** `snake_case` for variables, functions and modules; `PascalCase` for types;
  `SCREAMING_SNAKE_CASE` for constants ([RFC 430](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md)).
- **File size.** Keep source files under 2,000 lines; split into directory modules
  instead.

## Sponsorship

OptiRS is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps keep the COOLJAPAN ecosystem (OxiBLAS, OxiFFT, SciRS2 and friends)
100% pure Rust, and funds long-term support and security updates.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
