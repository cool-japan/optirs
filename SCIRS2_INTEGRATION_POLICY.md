# SciRS2 Integration Policy for OptiRS

**Status:** in force, and matching the code as of OptiRS 0.3.2
**SciRS2 version:** 0.6.5
**Base policy:** [SciRS2 Ecosystem Policy](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md)

## The rule

**OptiRS must not depend directly on any library that `scirs2-core` abstracts.** All
arrays, random number generation, numeric traits, SIMD and parallelism go through
`scirs2-core`. OptiRS extends SciRS2; it does not re-implement or bypass its foundation.

This is scoped to *scientific computing*. It is not a ban on all third-party crates:
OptiRS depends directly on ordinary infrastructure crates (`serde`, `thiserror`, `log`,
`tokio`, `futures`, `async-trait`, `toml`, `chrono`, `uuid`, `sha2`, `oxicode`,
`x25519-dalek`, `clap`, the GPU backend crates, `wasm-bindgen`, …), and that is fine. What
is forbidden is reaching around `scirs2-core` for the numerics.

A second, independent rule applies to all of those direct dependencies: they must be pure
Rust. C, C++ and Fortran dependencies are banned by [`deny.toml`](deny.toml) — see
*Enforcement* below.

### Prohibited direct dependencies

```toml
# ❌ FORBIDDEN in every OptiRS crate manifest
[dependencies]
rand = "*"              # use scirs2_core::random
rand_distr = "*"        # use scirs2_core::random
rand_core = "*"         # use scirs2_core::random
fastrand = "*"          # use scirs2_core::random
ndarray = "*"           # use scirs2_core::ndarray
ndarray-rand = "*"      # use scirs2_core::ndarray
ndarray-stats = "*"     # use scirs2_core::ndarray
num-traits = "*"        # use scirs2_core::numeric
num-complex = "*"       # use scirs2_core::numeric
num-integer = "*"       # use scirs2_core::numeric
nalgebra = "*"          # use scirs2_core::linalg
rayon = "*"             # use scirs2_core::parallel_ops
wide = "*"              # use scirs2_core::simd_ops
```

The workspace `Cargo.toml` records each of these as a deliberate, commented-out removal
rather than a silent omission, so the reason survives.

### Required import patterns

```rust
// ❌ FORBIDDEN - direct external imports
use rand::Rng;
use rand_distr::{Beta, Normal};
use ndarray::{array, s, Array, Array1, Array2};
use num_complex::Complex;
use num_traits::Float;
use rayon::prelude::*;

// ✅ REQUIRED - scirs2-core abstractions
use scirs2_core::random::*;       // rand + rand_distr functionality
use scirs2_core::ndarray::*;      // the ndarray ecosystem
use scirs2_core::ndarray_ext::*;  // plus the array!, s!, azip! macros
use scirs2_core::numeric::*;      // num-traits, num-complex, num-integer
use scirs2_core::simd_ops::*;     // SIMD
use scirs2_core::parallel_ops::*; // parallelism
```

Both `scirs2_core::ndarray` (a direct re-export of the `ndarray` API) and
`scirs2_core::ndarray_ext` (the same, plus the macros and statistics helpers) are correct;
the tree uses the former for types and the latter where the macros are needed.

### Architectural hierarchy

```
OptiRS (ML optimization specialization)
    ↓ uses abstractions from
SciRS2-Core (unified scientific computing layer)
    ↓ manages and abstracts
ndarray, rand, num-traits, …
```

## Dependency mapping

| External crate | SciRS2-Core module | OptiRS usage |
|---|---|---|
| `rand`, `rand_distr`, `rand_core` | `scirs2_core::random` | initialization, stochastic optimization, sampling |
| `ndarray`, `ndarray-rand`, `ndarray-stats` | `scirs2_core::ndarray` / `ndarray_ext` | parameters, gradients, array statistics |
| `num-traits`, `num-complex`, `num-integer` | `scirs2_core::numeric` | `Float`, `Zero`, `One`, `ToPrimitive`, complex numbers |
| `nalgebra` | `scirs2_core::linalg` | linear algebra, if ever needed |
| `rayon` | `scirs2_core::parallel_ops` | multi-core parameter-group processing |
| `wide` and hand-written intrinsics | `scirs2_core::simd_ops` | vectorized optimizer steps |
| GPU vendor SDK wrappers | `scirs2_core::gpu` | device, buffer and kernel abstractions |

### SIMD

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
```

`optirs-core`'s `simd_optimizer` and `SimdSGD` dispatch through this trait, so a single
code path covers every target ISA that `scirs2-core` supports.

### Parallelism

```rust
use scirs2_core::parallel_ops::*;
```

`optirs-core`'s `parallel_optimizer` distributes parameter groups across cores through
these abstractions rather than depending on `rayon` directly.

### GPU

```rust
use scirs2_core::gpu::{GpuBackend, GpuContext};
```

`optirs-gpu` and `optirs-core::gpu_optimizer` build on `scirs2-core`'s GPU layer. Where a
backend does not exist in `scirs2-core` 0.6.x (CUDA, ROCm), OptiRS returns an explicit
error rather than substituting a CPU path and calling it GPU work.

### Error handling

OptiRS defines its own `OptimError` in `optirs_core::error` and converts from
`scirs2_core::error::CoreError` where the two meet. Production code returns errors rather
than unwrapping or panicking; `expect()` survives only where the signature cannot return
one (`Default` implementations converting a constant, and similar), and its message names
the invariant.

## SciRS2 crates used by OptiRS

The philosophy is minimal and evidence-based: a SciRS2 crate is a dependency only when
there is a call site. This list is audited each release; 0.3.2 removed three crates that
had none.

### Required

| Crate | Version | Used for |
|---|---|---|
| `scirs2-core` | 0.6.5 | foundation — arrays, random, numeric traits, SIMD, parallel, GPU |
| `scirs2-optimize` | 0.6.5 | base optimization interfaces (`optirs-core`) |
| `scirs2-neural` | 0.6.5 | `optirs-core::neuromorphic::spike_based` |
| `scirs2-stats` | 0.6.5 | distributions and statistical functions (privacy, streaming drift tests) |

### Optional, feature-gated in `optirs-core`

| Crate | Feature | Used for |
|---|---|---|
| `scirs2-metrics` | `metrics-integration` | metrics interop beyond `scirs2-core` |
| `scirs2-datasets` | `cross-platform-testing` | sample data for the cross-platform test harness |

### Not used

- **`scirs2-autograd`** — OptiRS consumes pre-computed gradients; automatic
  differentiation is out of scope. Users pair OptiRS with an autodiff framework.
- **`scirs2-optim`** — superseded by `optirs-core`.
- **`scirs2-linalg`, `scirs2-signal`, `scirs2-series`** — carried as dependencies until
  0.3.2, when an audit found zero call sites for any of them. Removed.
- **`scirs2-cluster`, `scirs2-fft`, `scirs2-graph`, `scirs2-integrate`,
  `scirs2-interpolate`, `scirs2-io`, `scirs2-ndimage`, `scirs2-sparse`,
  `scirs2-spatial`, `scirs2-special`, `scirs2-text`, `scirs2-transform`,
  `scirs2-vision`** — no optimization use case; never added.

## Adding or removing a SciRS2 dependency

**Adding.** Show the call site first. A pull request adding a SciRS2 crate must name the
OptiRS module that will use it, show the code, and explain why nothing already in the
graph covers it. Then update this document.

**Removing.** Audit every SciRS2 dependency each release: grep for `use scirs2_<name>` and
for the crate path across the workspace. Zero hits means the dependency goes. Record the
removal in `CHANGELOG.md` so downstream users are not surprised by a feature-flag change.

## Best practices

**Import granularity.** Prefer specific imports over glob imports of whole crates.

```rust
// ✅ good
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;

// ❌ avoid
use scirs2_core::*;
```

**Array macros.** The `array!`, `s!` and `azip!` macros come from
`scirs2_core::ndarray_ext`:

```rust
#[cfg(test)]
mod tests {
    use scirs2_core::ndarray_ext::{array, s};

    #[test]
    fn slices_a_literal_array() {
        let data = array![1.0, 2.0, 3.0];
        let head = data.slice(s![..2]);
        assert_eq!(head.len(), 2);
    }
}
```

**Feature gates.** Gate an optional SciRS2 dependency at both the manifest and the use
site, and make the disabled path return an honest error rather than a silent default:

```rust
#[cfg(feature = "metrics-integration")]
use scirs2_metrics::MetricRegistry;
```

**Crate manifests.** Versions are managed at the workspace level; individual crates use
`workspace = true` and never pin their own version.

```toml
# workspace Cargo.toml
[workspace.dependencies]
scirs2-core = "0.6.5"
scirs2-optimize = "0.6.5"

# member crate Cargo.toml
[dependencies]
scirs2-core = { workspace = true }
scirs2-optimize = { workspace = true }

# extra scirs2-core features are requested per crate, e.g. optirs-gpu:
# scirs2-core = { workspace = true, features = ["gpu", "random"] }
```

## Enforcement

### Automated

- **`cargo deny check bans`** against [`deny.toml`](deny.toml), across the Linux, macOS,
  Windows and `wasm32-unknown-unknown` targets. It denies BLAS/LAPACK FFI (`openblas-src`,
  `blas-src`, `lapack-src`, `intel-mkl-src`, `netlib-src`), `bincode`, `z3`, `rusqlite`,
  the compression family, TLS/crypto FFI (`openssl`, `native-tls`, `ring`, `aws-lc-sys`),
  the C regex/tokenizer toolchains, and wildcard version requirements.
- **Zero-warning builds.** `cargo check --workspace --all-features --all-targets` and
  `cargo clippy --workspace --all-features --all-targets` must both report zero warnings,
  and each crate must also be warning-free under its own default feature set. There are no
  blanket `#![allow(...)]` attributes anywhere in the workspace, so nothing is hidden.
- **Doc tests.** Integration examples in doc comments are compiled and run.

### Manual

Every pull request is reviewed for direct scientific-computing dependencies. The quick
check:

```bash
grep -rn "^use ndarray::"    --include='*.rs' .   # must return nothing
grep -rn "^use rand::"       --include='*.rs' .   # must return nothing
grep -rn "^use num_traits::" --include='*.rs' .   # must return nothing
grep -rn "use rayon"         --include='*.rs' .   # must return nothing
grep -rn "use scirs2_core::" --include='*.rs' .   # should return a great many
```

As of 0.3.2 the first four return no hits in any build target and the last returns well
over a thousand. (The single match anywhere in the tree is a `use num_traits::Float` in
`optirs-core/examples/broken/`, a quarantined directory Cargo does not compile.)

### On a violation

1. Replace the direct dependency with the `scirs2-core` abstraction — do not add an
   exception.
2. If `scirs2-core` genuinely lacks the capability, raise it upstream in SciRS2 rather
   than working around it locally.
3. Record the resolution in `CHANGELOG.md` if it changes a public dependency.

## Conclusion

OptiRS builds on SciRS2's numerical foundation deliberately and minimally: everything
scientific goes through `scirs2-core`, every other dependency is justified, pure Rust, and
audited, and every dependency in the graph has a call site.
