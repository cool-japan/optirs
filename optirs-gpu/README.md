# OptiRS GPU

GPU acceleration for the OptiRS machine learning optimization library, built entirely on [`scirs2-core::gpu`](https://docs.rs/scirs2-core) — no direct `wgpu`/`metal`/`cudarc` dependency, no FFI by default, Pure Rust.

## What this crate actually is

`optirs-gpu` has two halves, and it is worth being precise about which is which:

1. **A real GPU optimizer path.** [`optimizers`](src/optimizers.rs) runs Adam, AdamW, SGD, RMSprop, Adagrad and LAMB as compute shaders through `scirs2_core::gpu`. Parameters and gradients are uploaded to device buffers, a compiled pipeline is dispatched, and the result is read back; the per-parameter optimizer state (Adam's `m`/`v`, SGD's momentum buffer, ...) stays resident in device memory between steps. The kernels ship in both WGSL and MSL ([`shaders`](src/shaders/mod.rs)) so the same optimizer runs on whichever backend the machine can reach.
2. **A CPU library of GPU-*aware* algorithms.** [`occupancy`](src/occupancy.rs), [`kernel_fusion`](src/kernel_fusion.rs), [`quantization`](src/quantization.rs), [`sparse_optimizer`](src/sparse_optimizer.rs), [`mixed_precision`](src/mixed_precision.rs) and [`memory::allocation`](src/memory/allocation) / [`memory::management`](src/memory/management) are pure-CPU models, planners and numerical routines that reason *about* GPU execution. They have no device dependency and are fully covered by unit tests.

Nothing in this crate fabricates a result: where a capability genuinely is not reachable (cross-device collectives, NVIDIA tensor cores, CUDA/ROCm on `scirs2-core` 0.6.x), the function returns an honest `Err` instead of a plausible-looking number.

## Backend support matrix

| Backend | Status |
|---------|--------|
| Metal (`metal`, automatic on macOS) | Real compute: MSL pipelines, buffers, dispatch, readback |
| WebGPU (`wgpu`, default) | WGSL kernels are implemented, but `scirs2-core` 0.6.5's runtime device probe never enumerates wgpu adapters, so `GpuContext::new(Wgpu)` currently fails everywhere. The path goes live when that probe is fixed upstream |
| OpenCL (`opencl`) | Context creation only — no OpenCL C kernel sources are shipped |
| CUDA (`cuda`) | Not available — `scirs2-core` removed its CUDA backend in 0.6.x; the feature gates reporting code only |
| ROCm | Not available |

`optirs_gpu::optimizers::SUPPORTED_BACKENDS` is the authoritative list — currently `[Wgpu, Metal]` — and `GpuOptimizerConfig` defaults to probing it in order.

### Why WGSL *and* MSL, hand-written?

`scirs2-core` 0.6.5 registers its own `adam_optimizer`/`sgd_optimizer`/etc. kernels, but they are not usable as-is:

- their hyper-parameters live in a `var<uniform>` block that the wgpu backend packs by iterating a `HashMap`, so the byte order is effectively random per process;
- their `metal_source` is empty, so a Metal context resolves them to nothing.

`optirs-gpu` therefore ships its own kernel sources ([`src/shaders/wgsl.rs`](src/shaders/wgsl.rs), [`src/shaders/msl.rs`](src/shaders/msl.rs)) that carry every scalar through a **storage buffer** bound by a fixed, deterministic name (`x, y, a, b, result, output`), and compiles them directly through `GpuCompiler::compile`. See the module docs in `src/shaders/mod.rs` for the full rationale.

## Installation

```toml
[dependencies]
optirs-gpu = "0.3.2"
```

Default features enable `wgpu`. On macOS, `metal` is also always compiled in (see `optirs-gpu/Cargo.toml`), because it is currently the backend that actually reaches the GPU there.

### Feature flags

| Feature | Effect |
|---|---|
| `wgpu` (default) | Forwards to `scirs2-core/wgpu` (Vulkan / Metal / DX12 via wgpu) |
| `metal` | Forwards to `scirs2-core/metal`; enabled automatically on macOS |
| `opencl` | Forwards to `scirs2-core/opencl`; context creation only, no kernels |
| `cuda` | Gates CUDA-specific *reporting* code only — `scirs2-core` has no CUDA compute path in 0.6.x |

`cargo check -p optirs-gpu --no-default-features` also builds cleanly: the `scirs2_core::gpu` types this crate's public API is expressed in are always available, independent of which backend runtimes are pulled in.

## Usage

### GPU-accelerated Adam

```rust
use optirs_gpu::optimizers::{AdamParams, GpuAdam};
use optirs_gpu::GpuOptimizer;
use scirs2_core::ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut optimizer = GpuAdam::new(AdamParams::default())?;
    optimizer.to_gpu()?;

    let mut params = Array1::from_elem(1_024, 1.0f32);
    let grads = Array1::from_elem(1_024, 0.01f32);
    optimizer.step_gpu(&mut params, &grads)?;

    // Bring the moment estimates back to host memory when done.
    optimizer.to_cpu()?;
    Ok(())
}
```

The same pattern applies to `GpuAdamW`, `GpuSgd` (+ `SgdParams`, optional momentum/Nesterov), `GpuRmsprop` (+ `RmspropParams`, optional centering), `GpuAdagrad` (+ `AdagradParams`) and `GpuLamb` (real layer-wise trust ratio, computed from an on-device norm reduction). Every optimizer implements the shared `optirs_gpu::GpuOptimizer<f32, D>` trait; there is no `f64` GPU path.

### Choosing a backend explicitly

```rust
use optirs_gpu::optimizers::{AdamParams, GpuAdam, GpuOptimizerConfig};
use scirs2_core::gpu::GpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = GpuAdam::with_config(
        AdamParams::default(),
        GpuOptimizerConfig::with_backend(GpuBackend::Metal),
    )?;
    Ok(())
}
```

### Single-device gradient synchronization

`multi_gpu::MultiGpuSync` drives a real compiled kernel through one `scirs2_core::gpu::GpuContext`. It is honest about its limits: `scirs2-core` 0.6.x exposes one device per context and no cross-device transport, so every method is real for `num_gpus == 1` (the only case answerable from local data alone) and returns `GpuOptimError::UnsupportedOperation` for `num_gpus > 1` — never a silent no-op.

```rust
use optirs_gpu::multi_gpu::{MultiGpuConfig, MultiGpuSync};
use scirs2_core::gpu::{GpuBackend, GpuContext};
use scirs2_core::ndarray::Array1;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Arc::new(GpuContext::new(GpuBackend::Metal)?);
    let mut sync = MultiGpuSync::<f32>::new(context, MultiGpuConfig::default(), 1_048_576)?;

    let mut grads = Array1::from_elem(4096, 1.0f32);
    sync.sync_gradients(&mut grads)?; // real all-reduce-mean kernel, num_gpus == 1

    // Real host-side top-k gradient compression (largest-magnitude selection).
    let (_values, _indices) = sync.compress_gradients(&grads)?;
    Ok(())
}
```

### CPU-side GPU-aware planning

The rest of the crate is pure CPU and needs no device at all:

```rust
use optirs_gpu::{calculate_occupancy, KernelResourceUsage, SmResourceLimits};

let limits = SmResourceLimits::sm_80();
let usage = KernelResourceUsage::new(/* registers_per_thread */ 32, /* shared_mem_per_block_bytes */ 0, /* threads_per_block */ 256);
let result = calculate_occupancy(&usage, &limits).expect("valid configuration");
println!("occupancy: {:.1}%", result.occupancy * 100.0);
```

## What is not implemented (and not faked)

- **Cross-device collectives.** `multi_gpu` can drive a real reduction kernel on a single device; anything that would require moving data between two physical GPUs returns `GpuOptimError::UnsupportedOperation`.
- **Literal NVIDIA tensor cores / WMMA.** `tensor_cores::TensorCoreOptimizer` provides real CPU-side matrix-layout optimization, precision selection and AMP loss scaling; the device GEMM entry points (`tensor_core_gemm`, `fused_adam_tensor_core`, ...) report an honest error because no backend this crate can reach exposes NVIDIA tensor cores.
- **Real vendor hardware telemetry.** `memory::vendors::*` model the CUDA/ROCm/oneAPI/Metal memory-management *API shape* (pools, streams, statistics) over ordinary system-heap allocations — there is no FFI to a real driver here (Pure Rust, no C/C++ dependency by default). `HardwareUtilizationState::unknown_baseline()` is an honest all-idle default rather than a fabricated reading; callers with a real telemetry source (e.g. `nvidia-smi` polled out of band) can supply it directly.

Real CUDA/ROCm execution belongs in the separate `oxicuda-*`/`oxirocm-*` crates, kept feature-gated off by default per COOLJAPAN policy.

## Architecture

```
optirs_gpu
├── optimizers        Real GPU optimizer steps (Adam/AdamW/SGD/RMSprop/Adagrad/LAMB)
├── shaders           WGSL + MSL kernel sources for the optimizers above
├── multi_gpu         Single-device collective kernel + honest multi-device errors
├── tensor_cores       CPU-side layout/precision planning + mixed-precision (AMP) trainer
├── mixed_precision    IEEE-754 binary16 conversion + dynamic loss scaling
├── occupancy          CUDA-style occupancy calculator (pure CPU model)
├── kernel_fusion       Elementwise op-graph fusion planner (pure CPU model)
├── quantization        QAT: int8/int4/fp8 fake-quant, per-channel scales, STE
├── sparse_optimizer     COO/CSR sparse Adam & SGD (lazy updates, dormancy decay)
├── memory
│   ├── allocation      Arena / buddy / slab allocators (host-memory data structures)
│   ├── management      Defragmentation, eviction, GC, prefetching policies
│   └── vendors         Vendor API-shape simulation (CUDA/ROCm/oneAPI/Metal), no FFI
├── backends            GpuBackend identifier + DeviceCapabilities data type
└── utils               Alignment, block-size and fragmentation helpers
```

Every device access goes through SciRS2:
- **GPU context**: `scirs2_core::gpu::GpuContext`
- **GPU memory**: `scirs2_core::gpu::GpuBuffer`
- **Kernel compilation**: `scirs2_core::gpu::GpuCompiler`

## Testing

```
cargo nextest run -p optirs-gpu --all-features
```

The suite includes real-hardware integration tests (`tests/gpu_parity.rs`, `tests/shader_compilation.rs`) that dispatch every optimizer kernel on a real device and compare against a CPU reference implementation bit-for-bit within tolerance, plus a real `naga`/`wgpu` and `MTLLibrary` compile check for every shipped shader. On a machine with no usable GPU adapter these print a `SKIP:` line and pass trivially rather than failing CI; run with `--no-capture` to see which branch was taken.

## Contributing

OptiRS follows the Cool Japan organization's development standards. See the main [OptiRS repository](https://github.com/cool-japan/optirs) for contribution guidelines.

## License

Licensed under the Apache License, Version 2.0.
