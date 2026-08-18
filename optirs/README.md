# OptiRS

The main integration crate for the OptiRS ecosystem: a thin facade that re-exports
`optirs-core` and, behind feature gates, the GPU, TPU, learned-optimizer, NAS and
benchmarking crates.

If you only need optimizers, depend on `optirs-core` directly. Use this crate when you
want more than one OptiRS crate under a single version and a single import root.

## Installation

```toml
[dependencies]
optirs = "0.3.3"
```

`optirs-core` is always included. Everything else is optional:

```toml
[dependencies]
optirs = { version = "0.3.3", features = ["gpu", "bench"] }
```

| Feature | Enables | Re-exported as |
|---|---|---|
| `core` (default) | `optirs-core` | `optirs::core` |
| `gpu` | `optirs-gpu` | `optirs::gpu` |
| `tpu` | `optirs-tpu` | `optirs::tpu` |
| `learned` | `optirs-learned` | `optirs::learned` |
| `nas` | `optirs-nas` | `optirs::nas` |
| `bench` | `optirs-bench` | `optirs::bench` |
| `full` | all of the above | — |

## Quick start

```rust
use optirs::prelude::*;
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

### The prelude

`optirs::prelude` covers `optirs-core` only — its optimizers, regularizers and schedulers,
whose names are verified not to collide.

The extension crates are deliberately **not** globbed into the prelude. They are
independently versioned and their public names do collide with `core` and with each other
(both `optirs-core::optimizers` and `optirs-gpu` export a `SparseAdam`; both
`optirs-learned` and `optirs-nas` export their own `OptimError`/`Result`). A glob
re-export of colliding names is unusable through the path that introduced the ambiguity,
so globbing them here would silently break `optirs::prelude::SparseAdam` the moment two
such features were enabled together.

Reach extension types through their own namespace instead:

```rust,ignore
use optirs::gpu::GpuAdam;
use optirs::learned::LSTMOptimizer;
use optirs::nas::ArchitectureSpace;
```

## What each crate gives you

### `optirs::core` — stable

Optimizers (SGD, SimdSGD, Adam, AdamW, AdaDelta, AdaBound, Adagrad, RMSprop, LAMB, LARS,
Lion, Lookahead, RAdam, Ranger, SAM, SparseAdam, GroupedAdam, MAML, MetaSGD, Reptile),
second-order methods (L-BFGS, Newton, Newton-CG, K-FAC), a large family of learning-rate
schedulers, regularizers, gradient-flow and loss-landscape analysis, metrics collection,
SIMD and parallel execution paths, differentially private and federated optimization,
streaming/online optimization with drift and anomaly detection, and a plugin system.

### `optirs::gpu` — partial hardware coverage

- **Metal**: real compute shaders run Adam, AdamW, SGD, RMSprop, Adagrad and LAMB end to
  end.
- **WebGPU**: WGSL kernels are implemented but blocked on an upstream `scirs2-core`
  adapter-probe bug.
- **OpenCL**: context creation only.
- **CUDA / ROCm**: no backend (`scirs2-core` 0.6.x dropped its CUDA backend).
- Vendor memory backends are host-memory API-shape simulations, disclosed as such in each
  file. Cross-device collectives return an explicit `UnsupportedOperation` error rather
  than a fabricated result.

### `optirs::tpu` — CPU reference implementation

No vendor TPU runtime is linked; it is proprietary and not distributable as pure Rust.
Pod topology and barrier synchronization, an XLA-shaped compiler (graph builder, DCE,
constant folding, CSE, fusion legality checks, allocator, shape inference), SHA-256
checkpoint integrity, and ring all-reduce / broadcast / reduce-scatter all run and are
tested on the CPU executor. Paths that would need real TPU silicon return an explicit
error.

### `optirs::learned` — research-grade

Transformer and LSTM learned optimizers (real truncated BPTT meta-training, seeded
reproducible initialization), MAML / Reptile / Meta-SGD, online meta-learning, few-shot
learning, continual learning, and cross-domain transfer.

### `optirs::nas` — research-grade

Random, evolutionary, RL, Bayesian and differentiable (DARTS, PC-DARTS, RobustDARTS)
search; NSGA-II and MOEA/D with exact hypervolume; grid, TPE and surrogate hyperparameter
search; progressive search; hardware cost modelling; architecture embedding.

### `optirs::bench`

Criterion-based benchmarking, memory profiling, regression detection, and cross-platform
orchestration over local, Docker and SSH execution.

## Examples

```bash
cargo run -p optirs --example basic_optimization --release
cargo run -p optirs --example scirs2_integration_demo --release
```

## SciRS2 foundation

OptiRS is built on [SciRS2](https://github.com/cool-japan/scirs) 0.6.5 and does not depend
directly on `ndarray`, `rand`, `rayon` or `num-traits` — all of that goes through
`scirs2-core`. The full rule is in
[`SCIRS2_INTEGRATION_POLICY.md`](https://github.com/cool-japan/optirs/blob/master/SCIRS2_INTEGRATION_POLICY.md).

## Documentation

- API documentation: [docs.rs/optirs](https://docs.rs/optirs)
- [Release notes](https://github.com/cool-japan/optirs/blob/master/CHANGELOG.md)
- [Usage guide](https://github.com/cool-japan/optirs/blob/master/USAGE_GUIDE.md)
- Repository: <https://github.com/cool-japan/optirs>

## License

Apache-2.0.
