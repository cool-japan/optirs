# `optirs` (integration crate) TODO

**Version:** 0.3.2
**Last audited:** 2026-08-18

## What this crate is

A facade. `src/lib.rs` is ~310 lines: a documentation header, feature-gated `pub use`
aliases for the six OptiRS crates, and a `prelude` module that re-exports
`optirs-core`'s optimizers, regularizers and schedulers.

It deliberately holds no logic of its own. There is no unifying trait layer, no shared
tensor abstraction and no builder API at this level — each sub-crate owns its own types,
and this crate only gives them a single version number and one import root. Anything that
looks like cross-crate API design belongs in `optirs-core` or the relevant sub-crate, not
here.

Consequently this crate has no unit tests. Its four doc-tested examples in `src/lib.rs` and
its two runnable examples are its test surface.

## Current state

- Feature gates `core` (default), `gpu`, `tpu`, `learned`, `nas`, `bench`, `full` — all
  wired and building.
- The prelude covers `optirs-core` only. The extension crates are intentionally excluded:
  their public names collide with `core` and with each other (`SparseAdam` exists in both
  `optirs-core` and `optirs-gpu`; `OptimError`/`Result` in both `optirs-learned` and
  `optirs-nas`), and a glob re-export of colliding names is unusable through the path that
  introduced the ambiguity. The reasoning is recorded on the `prelude` module itself so it
  is not "simplified" away later.
- Examples: `examples/basic_optimization.rs`, `examples/scirs2_integration_demo.rs`.
- `cargo check` / `cargo clippy -p optirs --all-features --all-targets`: 0 warnings.
- `cargo doc -p optirs --all-features --no-deps`: 0 warnings.

## Open work

- **More examples.** Only two exist. Worth adding: a GPU example (`--features gpu`), a
  learned-optimizer example, and a NAS example — each gated so the default build does not
  require them. These have to be written against the real sub-crate APIs; the previous
  versions of this file claimed a full "example gallery" that never existed.
- **A feature-combination smoke test.** The prelude-collision reasoning above is enforced
  only by a doc comment. A compile test that enables `gpu` + `learned` + `nas` together
  and resolves `optirs::prelude::SparseAdam` would turn it into something CI can catch.

## Not planned here

Cross-component work (GPU/TPU hybrid execution, learned-NAS co-optimization, joint
architecture-optimizer search, cross-component memory pooling) is sub-crate work, not
facade work. If it lands, it lands in the crate that owns the types; this crate would only
gain a `pub use`. Framework interop (PyTorch/TensorFlow/ONNX tensor compatibility) is
likewise out of scope: OptiRS operates on `scirs2_core::ndarray` arrays and expects the
caller to bridge.
