# OptiRS TPU

TPU-style coordination, pod management, and an XLA-shaped compilation pipeline for
large-scale distributed optimization in the OptiRS machine learning optimization library.

## Overview

`optirs-tpu` implements the coordination and compilation logic a TPU pod optimizer needs
— device/pod topology, barrier synchronization, fault detection and checkpointing, and an
XLA-style graph compiler (shape inference, constant folding, common-subexpression
elimination, dead-code elimination, kernel fusion, memory planning) — as real, tested, pure
Rust algorithms running on a CPU reference executor.

**There is no Google Cloud / vendor TPU runtime linked into this crate.** That runtime is
proprietary and cannot be shipped as pure Rust, so this crate does not provision TPU pods,
authenticate against GCP, or execute on physical TPU silicon. Where a capability genuinely
needs that runtime (e.g. migrating a running workload between devices), the corresponding
function returns a descriptive `Err` rather than a fabricated success. See the crate-level
doc comment (`cargo doc -p optirs-tpu --open`) for the current per-module status.

## What's real

- **`TPUOptimizer`** wraps any `optirs_core::Optimizer` (and implements that trait
  itself), driving it through a real compile → execute → profile pipeline.
- **XLA-shaped compiler** (`xla` module): computation-graph construction with real
  producer/consumer dependency tracking, dead-code elimination (with a fail-safe against
  deleting a graph whose outputs were never declared), constant folding, common
  sub-expression elimination, kernel-fusion legality checks, a real allocator with
  free/coalescing (not bump-only), and shape inference for reshape, convolution, dot, and
  broadcast.
- **`coordination::PodCoordinator`**: device and pairwise-channel topology, barrier
  synchronization, load balancing, and fault detection over real, observable in-process
  state.
- **`fault_tolerance`**: checkpoints are serialized to disk with a SHA-256 integrity hash
  and verified on restore; rollback and replication reuse that same verified path.
- **`synchronization`**: barriers with a correctly-signaled condvar predicate, plus ring
  all-reduce / broadcast / reduce-scatter collectives.
- **`pod_coordination::TPUPodCoordinator<T>`**: delegates every operation to the real
  `coordination::PodCoordinator` above (translating between their two independently
  designed config schemas); it used to hold only its config with no other methods.
- **`pod_coordination::synchronization::clocks::protocols::NtpSynchronizer`**: real
  RFC 5905 four-timestamp round-trip clock-offset estimation, rejecting a physically
  inconsistent exchange with `Err` rather than fabricating an offset.

## What's not implemented

- Execution on real TPU hardware (needs a vendor runtime this crate does not have).
- Cross-device workload migration (`FaultToleranceManager::migrate_workload` returns `Err`
  by design rather than fabricate a live migration).
- Cloud provisioning, billing/spot-bidding, and multi-region orchestration are out of
  scope for this crate; it coordinates a pod you already have, it does not create one.
- `xla::backend::profiling_integration`'s memory export reflects real activity only once a
  program has actually run: `TPUBackend::run_one_attempt` records every device-memory
  reservation and release it makes into that same profiler
  (`ProfilingIntegration::record_memory_allocation`/`record_memory_release`), including a
  peak-occupancy snapshot taken while the reservation is still held. Compile a program
  without executing it and the memory export is honestly empty — there is nothing to
  report yet — while the counter/trace exports already carry real recorded compile-step
  timings regardless.
- Some deeper `pod_coordination` submodules are still mixed-maturity scaffolding — check
  the module's own doc comments.

## Installation

```toml
[dependencies]
optirs-tpu = "0.3.3"
optirs-core = "0.3.3"
```

## Usage

```rust
use optirs_core::optimizers::SGD;
use optirs_tpu::{TPUConfig, TPUOptimizer, TPUVersion};

fn main() -> Result<(), optirs_tpu::error::OptimError> {
    let base_optimizer = SGD::new(0.01f32);
    let config = TPUConfig {
        tpu_version: TPUVersion::V4,
        num_cores: 8,
        ..Default::default()
    };

    let mut tpu_opt = TPUOptimizer::new(base_optimizer, config)?;

    // params/gradients are ndarray arrays; tpu_step compiles (or reuses a cached
    // compilation of) the update graph and runs it on the CPU reference executor.
    // let updated = tpu_opt.tpu_step(&params, &gradients)?;

    Ok(())
}
```

`TPUOptimizer` also implements `optirs_core::Optimizer`, so it can be used anywhere generic
code expects one:

```rust,ignore
fn run_step<O: optirs_core::Optimizer<f32, scirs2_core::ndarray::Ix1>>(
    optimizer: &mut O,
    params: &Array1<f32>,
    grads: &Array1<f32>,
) -> optirs_core::Result<Array1<f32>> {
    optimizer.step(params, grads)
}
```

### Pod coordination

```rust
use optirs_tpu::coordination::{CoordinationError, PodConfig, PodCoordinator};

fn coordinate_one_step() -> Result<(), CoordinationError> {
    let config = PodConfig {
        num_devices: 8,
        ..Default::default()
    };
    let mut pod = PodCoordinator::new(config)?;
    pod.synchronize_devices("step-0".to_string())?;
    Ok(())
}
```

### Checkpointing

`fault_tolerance::FaultToleranceManager::create_checkpoint` serializes real coordination
state to disk under the configured storage path and records a SHA-256 hash;
`restore_checkpoint` re-verifies that hash before applying anything, so a corrupted or
truncated checkpoint fails loudly instead of silently "succeeding".

## Architecture

Built on [SciRS2](https://github.com/cool-japan/scirs) abstractions:
- **Numeric**: `scirs2_core::ndarray`, `scirs2_core::numeric::Float`
- **Errors**: `scirs2_core::error::CoreError`, re-exported here as `optirs_tpu::error::OptimError`

Module map:
- `coordination` — pod/device topology, barriers, load balancing, fault detection
- `synchronization` — condvar barriers and ring collectives
- `fault_tolerance` — checkpoint/restore, recovery strategies
- `monitoring` — health checks and performance reports over live metric history
- `tpu_backend` — device management and the CPU reference executor
- `xla` — the graph-capture / shape-inference / optimization / scheduling pipeline
- `pod_coordination` — larger-scale pod topology and clock-synchronization scaffolding
  (mixed maturity; consult individual module docs)

## Development Guidelines

- `snake_case` for variables and functions, `PascalCase` for types, `SCREAMING_SNAKE_CASE`
  for constants, per [RFC 430](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md).
- No fabricated success values or hardcoded placeholder outputs. Where a capability
  genuinely requires hardware or a runtime this crate does not have, return a descriptive
  `Err` rather than simulate one.
- Before submitting: `cargo fmt`, `cargo clippy --all-features`, `cargo test --all-features`.

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main OptiRS
repository for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0.
