# OptiRS GPU TODO (v0.3.2)

## Module status

**Tests**: 256 passing (`cargo nextest run --all-features`) + 2 passing doctests (`cargo test --doc --all-features`).
**Compiler warnings**: none — `cargo check` is clean on default features, `--all-features`, `--no-default-features`, and each backend feature individually.
**Clippy**: clean (`cargo clippy --all-features --all-targets`); the 2 `clippy::wrong_self_convention` lints on `to_gpu`/`to_cpu` from prior releases are resolved — see "Known minor items" below.
**Real backends**: Metal (macOS, real MSL compute path), WebGPU (WGSL kernels shipped and compiler-verified, but currently unreachable through `scirs2-core` 0.6.5's adapter probe — see below).
**Not real, and not claimed to be**: CUDA, ROCm, OpenCL compute (see "Out of scope" below).

This file describes what is actually implemented and tested, not an aspirational feature list. See `README.md` for the architecture and usage, and the module docs in `src/lib.rs` for the authoritative backend-support matrix.

---

## Done: real GPU optimizer path

- [x] `optimizers::GpuAdam` / `GpuAdamW` / `GpuSgd` / `GpuRmsprop` / `GpuAdagrad` / `GpuLamb` — each implements `optirs_gpu::GpuOptimizer<f32, D>` and runs a real compute shader through `scirs2_core::gpu`: device buffer upload, compiled-pipeline dispatch, readback. Per-parameter state (`m`/`v`, momentum buffer, ...) stays resident on the device between steps (`optimizers.rs`, 7 unit tests).
- [x] Hand-written WGSL + MSL kernel sources for all six optimizers, working around two upstream `scirs2-core` 0.6.5 bugs: non-deterministic uniform-buffer packing (worked around with storage buffers) and empty `metal_source` on the built-in registry kernels (worked around by shipping and compiling this crate's own MSL) (`shaders/`, 8 unit tests + a dedicated `naga`/`wgpu` and `MTLLibrary` compile-check integration test, `tests/shader_compilation.rs`, 3 tests).
- [x] AdamW's weight decay is genuinely decoupled from the moment estimates (distinct kernel from Adam's coupled L2), verified against a CPU reference.
- [x] LAMB's trust ratio is computed for real from an on-device workgroup norm reduction (two dispatches of one pipeline: phase 0 advances moments + reduces `‖p‖²`/`‖update‖²`, the host finishes the reduction and computes `trust = ‖p‖/‖update‖`, phase 1 applies the trusted step).
- [x] RMSprop's centered variant keeps a dedicated running mean-gradient buffer, so `E[g²] − E[g]²` is a real variance estimate, not a relabeled `E[g²]`.
- [x] SGD honors momentum, dampening *and* Nesterov acceleration from host-set kernel uniforms (previously the shader ignored what the host configured).
- [x] GPU/CPU parity integration tests (`tests/gpu_parity.rs`, 9 tests) dispatch on real hardware and assert agreement with a CPU reference optimizer within `1e-5`, including a 10-step trajectory (proves device-resident state and bias-correction step counting are both real), a host round-trip mid-training (proves `move_to_cpu`/`move_to_gpu` genuinely move data, not just flip a flag), and a dedicated check that the deprecated `to_cpu`/`to_gpu` shims delegate to (rather than reimplement) the same state transition.
- [x] `--no-default-features` and each backend feature individually (`wgpu`, `metal`, `opencl`, `cuda`) build clean.

## Done: single-device multi-GPU collective

- [x] `multi_gpu::MultiGpuSync` compiles and dispatches a real `all_reduce_mean` kernel. Honest about its ceiling: `scirs2-core` 0.6.x exposes one device per context and no cross-device transport, so every strategy (ring/tree/hierarchical/pipeline) is real for `num_gpus == 1` and returns `GpuOptimError::UnsupportedOperation` for `num_gpus > 1` — never a silent no-op, never a kernel dispatch to an unregistered name (17 unit tests).
- [x] `compress_gradients` does real host-side top-*k* largest-magnitude selection (previously returned zeros unconditionally).
- [x] Pipeline-parallel sync submits one real dispatch per chunk via `dispatch_no_wait` + one `gpu_sync` fence for the batch — genuine command-queue overlap — and covers the tail when the tensor length does not divide evenly by the configured depth.
- [x] `synchronize_all` waits on a real device fence (`GpuContext::gpu_sync`), not a wall-clock guess.
- [x] `MultiGpuConfig::validate()` rejects every config field later used as a divisor or index bound (`num_gpus`, `local_group_size`, `pipeline_depth`, `rank`, `compression_ratio`) before it can panic deep inside a sync call.
- [x] Bandwidth accounting clamps sub-microsecond samples instead of producing `inf`/`NaN`.

## Done: CPU-side algorithm modules

Pure-Rust reference logic on the CPU host, no device dependency, no fabricated device behavior:

- [x] Kernel fusion for reduced memory bandwidth (`kernel_fusion.rs` — elementwise op-graph fusion planner: DAG with fusion-legality checks, Kahn topo + union-find grouping, acyclicity guard, bytes-moved fused-vs-unfused cost model; 13 tests)
- [x] Occupancy modeling (`occupancy.rs` — CUDA-style occupancy calculator: registers/shared-mem/warps/blocks-per-SM limits → warps-per-SM and occupancy %, `sm_70`…`sm_90` resource tables, `optimal_block_size()` search; 17 tests)
- [x] Sparse tensor optimizers (`sparse_optimizer.rs` — COO/CSR gradients, lazy sparse Adam with dormancy-decay catch-up, sparse SGD touching only nonzero coordinates, global-step bias correction; 21 tests)
- [x] Quantization-aware training (`quantization.rs` — int8/int4 + fp8 (E4M3/E5M2) fake-quant, per-tensor and per-channel scales, straight-through estimator, unbiased stochastic rounding, FP32-master-weight `QatOptimizer`; 22 tests)
- [x] Mixed precision (`mixed_precision.rs` — full IEEE-754 `binary16` conversion with subnormals, round-half-to-even and NaN-payload preservation exercised over all 65,536 bit patterns, plus a standard dynamic loss scaler; 9 tests)
- [x] Tensor-core-aware planning (`tensor_cores/` — matrix layout optimization, precision selection, adaptive scheduling against caller-supplied (never fabricated) hardware telemetry, 2:4 structured sparsity with NaN-safe pruning; 13 tests). The WMMA GEMM entry points themselves report an honest `UnsupportedOperation`: no backend this crate can reach exposes real NVIDIA tensor cores.
- [x] Vendor memory-API shape (`memory/vendors/` — CUDA/ROCm/oneAPI/Metal pool/stream/statistics types over ordinary system-heap allocations; UB-free allocation path with a real `Layout` reconstructed at free time; explicitly documented as a Pure-Rust API-shape simulation, not real driver access). `execute_operation`/`commit_command_buffer`/`wait_until_completed`/`wait_until_idle` no longer inject a `std::thread::sleep` to imitate device-transfer latency — that was fake timing layered on top of an already-disclosed simulation. These calls, and the `memcpy`/`usm_memcpy`/`blit_copy` entry points that reach them, move no bytes and never did (there is no real device to copy to or from); they now return immediately instead of after a fabricated delay, and the per-vendor module docs say so explicitly so a `memory_transfers`/`blit_commands` counter is never mistaken for "bytes actually moved."
- [x] Real GPU vendor *detection* where it is honestly answerable: a real `Metal` `GpuContext` probe on macOS, real PCI vendor IDs from `/sys/bus/pci/devices` on Linux, an empty list everywhere else — never a fabricated "NVIDIA + AMD + Intel found" on every machine.

---

## Out of scope for autonomous implementation

These require real GPU hardware, vendor collectives, or driver-level APIs this crate cannot reach without FFI (forbidden by default under COOLJAPAN policy). Faking them would mean inventing device behavior, so they report an honest `Err` instead:

### Cross-device multi-GPU (needs a real NCCL/RCCL/MPI-equivalent transport)
- [ ] Actual cross-device gradient all-reduce (today: real for `num_gpus == 1`, honest error above that)
- [ ] Load balancing across heterogeneous physical GPUs
- [ ] Fault tolerance and recovery across devices

### Literal NVIDIA tensor cores (needs `wmma`/PTX, i.e. real CUDA)
- [ ] `tensor_core_gemm` / `fused_adam_tensor_core` / `sparse_tensor_core_gemm` device execution
- [ ] Real compute-capability probing (`TensorCoreOptimizer::compute_capability` is honestly `(0, 0)` — no reachable backend exposes it)

### Real vendor hardware telemetry (needs NVML / `rocm-smi` / IOKit FFI)
- [ ] Live GPU utilization / temperature / power draw (`HardwareUtilizationState` accepts caller-supplied readings; this crate has no sensor of its own)
- [ ] Real CUDA/ROCm/oneAPI device memory allocation (belongs in the separate `oxicuda-*`/`oxirocm-*` crates, feature-gated off by default)

### Upstream `scirs2-core` blockers
- [ ] `GpuContext::new(GpuBackend::Wgpu)` — blocked on `scirs2-core` 0.6.5's runtime device probe never enumerating wgpu adapters (the WGSL kernels are already implemented and compiler-verified; this activates automatically once fixed upstream, no code change needed on this side).

---

## Known minor items

- None currently open. (Previously: `GpuOptimizer::to_gpu`/`to_cpu` tripped `clippy::wrong_self_convention` — the lint expects a consuming `self` for a `to_*` name, but the trait's semantics are "move state to/from the device in place", which needs `&mut self`. Resolved in 0.3.2 by renaming the required trait methods and every optimizer's inherent alias to `move_to_gpu`/`move_to_cpu`; `to_gpu`/`to_cpu` remain as `#[deprecated(since = "0.3.2")]` shims that delegate to the new names. This is source-compatible for *callers* — `optimizer.to_gpu()` still compiles, with a deprecation warning, for every `GpuAdam`/`GpuAdamW`/`GpuSgd`/`GpuRmsprop`/`GpuAdagrad`/`GpuLamb` in this crate. It is a **breaking change for external implementors** of `GpuOptimizer`: `move_to_gpu`/`move_to_cpu` are now the required trait methods, so any out-of-tree `impl GpuOptimizer<..> for MyType { fn to_gpu(..) }` fails to compile and must rename to `move_to_gpu`/`move_to_cpu`. No such implementor exists in this workspace. See the changelog.)
