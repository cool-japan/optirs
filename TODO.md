# OptiRS TODO

**Version:** 0.3.3
**Last audited:** 2026-08-18

This file tracks what is *open*. What has been completed is recorded in
[`CHANGELOG.md`](CHANGELOG.md); duplicating it here only creates two things to keep in
sync. Each crate has its own `TODO.md` for crate-local work.

## Current state (measured, `--all-features`)

| Signal | Value |
|---|---|
| Tests | more than 4,200 unit/integration tests passing, plus doc tests |
| `cargo check --workspace --all-targets` | 0 warnings |
| `cargo clippy --workspace --all-targets` | 0 warnings |
| Blanket `#![allow(...)]` attributes | none anywhere in the workspace |
| Production `.unwrap()` | none |
| Production blind `expect("unwrap failed")` / `expect("lock poisoned")` | none (the only textual matches are doc comments quoting the pattern they replaced) |
| Source files ≥ 2,000 lines | none |
| `cargo deny check bans` | passes |
| Source size | 907 Rust files, ~342k lines of code (`tokei`) |
| SciRS2 | 0.6.5; no direct `ndarray` / `rand` / `rayon` / `num-traits` imports |

Reproduce with `cargo nextest run --workspace --all-features`,
`cargo clippy --workspace --all-features --all-targets` and `cargo deny check bans`.

## Open work

Everything below is a real gap in the current tree. Each item names the code path so the
claim can be checked. Paths that cannot deliver a result return an explicit error today —
none of them fabricate one.

### Blocked on upstream

- **`optirs-gpu` WebGPU backend.** The WGSL kernels are implemented, but `scirs2-core`
  0.6.5's runtime device probe never enumerates `wgpu` adapters, so
  `GpuContext::new(Wgpu)` fails everywhere. The path goes live when the upstream probe is
  fixed; nothing on the OptiRS side is missing.

### Requires a design decision

- **`self_tuning` hyperparameter proposal.** `SelfTuningOptimizer` records the observation
  half of a hyperparameter search (metric history, best-so-far configuration), but the
  proposal step is not implemented and the `SearchStrategy` variants return
  `UnsupportedOperation` (`optirs-core/src/self_tuning.rs`). Implementing it means
  choosing which hyperparameters are searchable and which strategy family
  (Bayesian / successive halving / grid) is canonical.
- **`AdaptiveTuner` Bayesian and reinforcement-learning strategies.** Grid, Greedy and GA
  search work; the other two return `UnsupportedOperation`
  (`optirs-core/src/hardware_aware/adaptive_tuner.rs`). A Bayesian variant should reuse
  the Gaussian process already in `privacy/private_hyperparameter_optimization`, rather
  than growing a third one.
- **Byte-level compression and Parquet export for streaming metrics.**
  `optirs-core/src/streaming/streaming_metrics/export.rs` refuses Gzip/LZ4/Zstd payload
  compression and Parquet output because `optirs-core` bundles no codec or columnar
  writer. If these are wanted, the pure-Rust route is an `oxiarc-*` dependency; that is a
  workspace-level decision.
- **Nested automatic differentiation in `optirs-learned::higher_order`.**
  `HvpMode::NestedAutodiff` and `MixedPartialMethod::NestedAutodiff` return explicit
  errors. This is not an omission that can be patched: the engine's objective is
  `impl Fn(&Array1<T>) -> T`, monomorphic in `T`, and forward mode needs the objective
  generic over a dual type while reverse mode needs it as tape operations. Real nested AD
  requires changing that signature.
- **Shamir `t`-of-`n` self-mask shares for secure aggregation.** The Bonawitz
  implementation in `optirs-core/src/privacy/federated/` handles dropout by disclosure.
  Adding threshold secret sharing of the self-mask seed would make it robust to a
  dropped-then-recovered client. The Shamir primitive already exists in
  `privacy/secure_multiparty`; wiring it in is a protocol change, not a port.
- **NAS `MultiObjectiveConfig::user_preferences` / `constraint_handling`.** Declared and
  documented but not read by any algorithm (`optirs-nas/src/nas_engine/config.rs`);
  preference articulation is unimplemented for NSGA-II, NSGA-III and MOEA/D.
- **`optirs-nas` benchmark-suite custom evaluator.** The callback shape for a
  user-supplied architecture evaluator has not been designed.
- **A meta-learning API reference.** The workspace-root `docs/meta_learning_api.md` was
  deleted in 0.3.2: it documented an `optirs_core::meta_learning` module that does not
  exist (the real one is `optirs_learned::meta_learning`, with an entirely different API),
  four of its seven documented types existed nowhere in the workspace, and the file had a
  corrupted, half-duplicated section. If a narrative meta-learning guide is wanted, it
  should live inside `optirs-learned/` next to the code it describes, written against
  `MetaLearner`, the MAML / Reptile / Meta-SGD learners and `MetaTask`.

### Requires hardware or an external runtime

- **`optirs-gpu` CUDA and ROCm backends.** `scirs2-core` 0.6.x ships no CUDA backend, so
  there is nothing to build on. OpenCL gets as far as context creation; no kernels ship.
- **`optirs-tpu` vendor runtime.** No TPU runtime is linked — it is proprietary and not
  distributable as pure Rust. The crate is a CPU-reference implementation and says so.
- **`optirs-wasm` WGSL compute kernels.** WebGPU adapter detection and the device
  handshake are real; running per-optimizer compute shaders from the WASM bindings is not
  implemented and would need `optirs-gpu`/`wgpu` in the WASM build.
- **`optirs-gpu` vendor memory backends** (`cuda`/`rocm`/`oneapi`/`metal`) are host-memory
  API-shape simulations whose copy functions move zero bytes. Each file states this at the
  top. They become real only alongside a real device backend.

### Code-quality follow-ups

- **One remaining production `panic!`.** `CurriculumScheduler::new`
  (`optirs-core/src/schedulers/curriculum.rs`) panics when handed an empty stage list.
  Every other panic path in the workspace was converted to a typed error during 0.3.2;
  this one is a constructor precondition and needs a fallible constructor (or a
  non-empty-collection parameter type) to close.
- **About 165 `expect(...)` calls remain in production code**, all in positions whose
  signature cannot return an error — overwhelmingly `Default` implementations converting a
  constant literal such as `0.9` into the generic scalar type. Each carries a message
  naming the invariant rather than the old blanket `"unwrap failed"`. Removing them
  entirely means giving those types fallible constructors instead of `Default`.

## Ideas for 0.4.0

Not committed, not started:

- Transformer-based learned-optimizer improvements (`optirs-learned`)
- Preference-articulated multi-objective NAS (depends on the config decision above)
- A dedicated GPU kernel path once the upstream `wgpu` probe is fixed

## Conventions for this file

- An item is listed only if the gap is verifiable in the current tree.
- Completed work moves to `CHANGELOG.md` and is deleted from here.
- "Not implemented" must correspond to code that returns an error, never to code that
  returns a plausible-looking value.
