# OptiRS WASM TODO

**Version:** 0.3.2
**Last audited:** 2026-08-18

This file tracks what is *open* for `optirs-wasm`. What has been completed is recorded in
the workspace [`CHANGELOG.md`](../CHANGELOG.md); duplicating it here only creates two things
to keep in sync. See `README.md` for usage, and the module docs in `src/lib.rs` and
`src/webgpu.rs` for the implementation notes this file is checked against.

## Current state (measured, `--all-features`)

| Signal | Value |
|---|---|
| Tests (host target) | 37 passing -- `cargo nextest run -p optirs-wasm --all-features` (7 unit tests under `src/`, 30 in `tests/wasm_tests.rs`) |
| Tests (real `wasm32`, browser/Node host) | 10 more -- `wasm_bindgen_test`s in `tests/wasm_bindgen_tests.rs` (7) and `tests/wasm_bindgen_webgpu_tests.rs` (3), built and run only via `wasm-pack test`; not counted in the 37 above |
| `cargo clippy --workspace --all-features --all-targets` | 0 warnings |
| Optimizers | 13 (`src/optimizers/`) |
| Schedulers | 14 (`src/schedulers/`) |
| `todo!()` / `unimplemented!()` in `src/` | 0 |
| Production `.unwrap()` in `src/` | 0 |

## Open work

Everything below is a real gap in the current tree. Each item names the code path so the
claim can be checked.

### Requires hardware or an external runtime

- **WGSL compute kernel execution from the WASM bindings is not implemented.**
  `WasmGpuOptimizer` (`src/webgpu.rs`, `webgpu` feature) does real work up to the point of
  acquiring a device: `is_available()` checks `navigator.gpu` honestly through
  `js_sys::Reflect` (never hardcoded), and `initialize()` runs an actual `requestAdapter()` /
  `requestDevice()` handshake, reading the acquired adapter's real
  vendor/architecture/description and returning an explicit `Err` when no WebGPU host is
  present. Running an optimizer's compute *on* that acquired device -- dispatching a WGSL
  shader per optimizer step, the way `optirs-gpu` does for its native backends -- is not
  implemented; the module doc comment at the top of `src/webgpu.rs` says so explicitly.
  Closing this gap needs `optirs-gpu`/`wgpu` reachable from a `wasm32` build, which is a
  separate integration project, not a bug fix.

A full read of `src/` for this audit found no other gap: no `todo!()`/`unimplemented!()`, no
production `.unwrap()`, and no other path that returns a fabricated or hardcoded value.

## Conventions for this file

- An item is listed only if the gap is verifiable in the current tree.
- Completed work moves to `CHANGELOG.md` and is deleted from here.
- "Not implemented" must correspond to code that returns an error, never to code that
  returns a plausible-looking value.
