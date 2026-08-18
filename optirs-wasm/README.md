# OptiRS WASM - WebAssembly Bindings for OptiRS

**Version:** 0.3.2
**Status:** Production Ready

High-performance WebAssembly bindings for OptiRS deep learning optimizers and learning rate schedulers. Run state-of-the-art ML optimization algorithms in the browser and Node.js.

## Features

- **13 Optimizers** - SGD, Adam, AdamW, RMSprop, RAdam, LAMB, Lion, LARS, Adagrad, AdaDelta, AdaBound, Ranger, SparseAdam
- **14 Schedulers** - CosineAnnealing, CosineAnnealingWarmRestarts, OneCycle, LinearWarmupDecay, ExponentialDecay, StepDecay, CyclicLR, ReduceOnPlateau, Constant, LinearDecay, ViTLayerDecay, AttentionAware, NoiseInjection, Curriculum
- **TypeScript Support** - Type definitions included (`optirs.d.ts`)
- **Multi-Target** - Works with bundlers, web, and Node.js
- **Zero Dependencies** - Pure Rust compiled to WASM

## Installation

### npm / yarn

```bash
npm install @cooljapan/optirs
# or
yarn add @cooljapan/optirs
```

### Browser (ES Module)

```html
<script type="module">
  import init, { WasmAdam } from '@cooljapan/optirs';
  await init();

  // Classes marked `#[wasm_bindgen(constructor)]` are real JS constructors,
  // used with `new`, not a static `.new(...)` factory.
  const optimizer = new WasmAdam(0.001);
</script>
```

## Quick Start

### Basic Optimization

```typescript
import init, { WasmAdam } from '@cooljapan/optirs';

await init();

// Create optimizer
const adam = new WasmAdam(0.001);

// Parameters and gradients as Float64Array
const params = new Float64Array([1.0, 2.0, 3.0, 4.0]);
const grads = new Float64Array([0.1, 0.2, 0.15, 0.08]);

// Optimization step
const updatedParams = adam.step(params, grads);
console.log(updatedParams); // Updated parameters
```

### With Learning Rate Scheduler

```typescript
import init, { WasmAdamW, WasmCosineAnnealingWarmRestarts } from '@cooljapan/optirs';

await init();

const optimizer = WasmAdamW.new_with_config(0.001, 0.9, 0.999, 1e-8, 0.01);
const scheduler = new WasmCosineAnnealingWarmRestarts(0.001, 0.0, 10, 2.0);

for (let epoch = 0; epoch < 100; epoch++) {
  // Training step...
  const lr = scheduler.step();
  optimizer.learning_rate = lr;
}
```

### Factory Functions (JSON Configuration)

For optimizers/schedulers whose full constructor takes many positional
arguments (or can fail, like `WasmAdaDelta`/`WasmAdaBound`/`WasmRanger`), the
`create_optimizer`/`create_scheduler` factory functions build the real object
from a JSON config, filling in any omitted field with that algorithm's own
default:

```typescript
import init, { create_optimizer, create_scheduler, available_optimizers } from '@cooljapan/optirs';

await init();

console.log(available_optimizers());
// ["adam", "adamw", "sgd", "rmsprop", "lamb", "lion", "radam",
//  "adagrad", "adadelta", "adabound", "ranger", "lars", "sparse_adam"]

// Returns the real WasmAdaBound instance (not a description string) --
// ready to call .step(...) on immediately.
const adabound = create_optimizer(JSON.stringify({
  type: 'adabound',
  lr: 0.001,
  final_lr: 0.1,
}));

const scheduler = create_scheduler(JSON.stringify({
  type: 'cosine_annealing_warm_restarts',
  initial_lr: 0.001,
  t_0: 10,
  t_mult: 2.0,
}));
```

Every optimizer `type` accepts `lr` (except `adadelta`, which has no learning
rate -- see the table above) plus its full-configuration field names 1:1
(`beta1`, `beta2`, `epsilon`, `weight_decay`, `momentum`, `rho`,
`trust_coefficient`, `lookahead_k`, `lookahead_alpha`, `bias_correction`,
`final_lr`, `gamma`, `amsbound`, as applicable to that type). Scheduler field
names are **not** uniform across types -- they mirror each scheduler's own
constructor argument names, which differ (e.g. the base rate is `lr` for
`constant`, `initial_lr` for most others, `base_lr` for `vit_layer_decay`/
`attention_aware`/`noise_injection`, `max_lr` for `one_cycle`). The accepted
fields per scheduler `type`:

| `type` | Accepted fields |
|--------|------------------|
| `cosine_annealing` | `initial_lr`, `min_lr`, `t_max` |
| `cosine_annealing_warm_restarts` | `initial_lr`, `min_lr`, `t_0`, `t_mult` |
| `one_cycle` | `max_lr`, `total_steps`, `pct_start`, `div_factor`, `final_div_factor` |
| `linear_warmup_decay` | `initial_lr`, `warmup_steps`, `total_steps`, `min_lr` |
| `exponential_decay` | `initial_lr`, `decay_rate`, `decay_steps` |
| `step_decay` | `initial_lr`, `step_size`, `gamma` |
| `cyclic_lr` | `base_lr`, `max_lr`, `step_size`, `mode` (`"triangular"` default / `"triangular2"` / `"exp_range"`), `gamma` (only for `exp_range`) |
| `reduce_on_plateau` | `initial_lr`, `factor`, `patience` |
| `constant` | `lr` |
| `linear_decay` | `initial_lr`, `final_lr`, `total_steps` |
| `vit_layer_decay` | `base_lr`, `decay_rate`, `num_layers`, plus `warmup_steps`/`total_steps` to opt into the warmup variant |
| `attention_aware` | `base_lr`, `warmup_steps`, `total_steps` |
| `noise_injection` | `base_lr`, `min_lr`, `distribution` (`"uniform"` default / `"gaussian"` / `"cyclical"` / `"decaying"`), plus that distribution's own fields (`min_noise`/`max_noise`, `mean`/`std_dev`, `amplitude`/`period`, or `initial_scale`/`final_scale`/`decay_steps`) |
| `curriculum` | `stages` (required array, e.g. `[{"learning_rate": 0.01, "duration": 100}]`), `final_lr`, `immediate` (bool, default `false`) |

### Advanced Configuration

```typescript
import init, { WasmOptimizerConfig, WasmAdam } from '@cooljapan/optirs';

await init();

const config = new WasmOptimizerConfig(0.001);
config.beta1 = 0.9;
config.beta2 = 0.999;
config.epsilon = 1e-8;
config.weight_decay = 0.01;

// Build the real optimizer from the individual fields (there is no
// `WasmAdam.from_config`; read the fields you need off `config`).
const optimizer = WasmAdam.new_with_config(
  config.lr, config.beta1, config.beta2, config.epsilon, config.weight_decay,
);
```

## Available Optimizers

All constructors below are real JS constructors (`new ClassName(...)`) unless
marked `static`. Optimizers whose primary constructor can fail (invalid
hyperparameters) return a value that throws a JS exception on error, matching
`Result::Err` on the Rust side.

| Class | Constructor | Description |
|-------|-------------|-------------|
| `WasmSGD` | `new(lr)` / `static new_with_config(lr, momentum, weight_decay)` | Stochastic Gradient Descent |
| `WasmAdam` | `new(lr)` / `static new_with_config(lr, beta1, beta2, epsilon, weight_decay)` | Adaptive Moment Estimation |
| `WasmAdamW` | `new(lr)` / `static new_with_config(lr, beta1, beta2, epsilon, weight_decay)` | Adam with decoupled weight decay |
| `WasmRMSprop` | `new(lr)` / `static new_with_config(lr, rho, epsilon, weight_decay)` | Root Mean Square Propagation |
| `WasmRAdam` | `new(lr)` / `static new_with_config(lr, beta1, beta2, epsilon, weight_decay)` | Rectified Adam |
| `WasmLAMB` | `new(lr)` / `static new_with_config(lr, beta1, beta2, epsilon, weight_decay, bias_correction)` | Layer-wise Adaptive Moments |
| `WasmLion` | `new(lr)` / `static new_with_config(lr, beta1, beta2, weight_decay)` | Evolved Sign Momentum |
| `WasmLARS` | `new(lr)` / `static new_with_config(lr, momentum, weight_decay, trust_coefficient, eps)` | Layer-wise Adaptive Rate Scaling |
| `WasmAdagrad` | `new(lr)` / `static new_with_config(lr, epsilon, weight_decay)` | Adaptive Gradient |
| `WasmSparseAdam` | `new(lr)` / `static new_with_config(lr, beta1, beta2, epsilon, weight_decay)` | Adam with a `step_sparse(params, indices, values, total_dim)` path for sparse gradients |
| `WasmAdaDelta` | `new(rho, epsilon)` *(throws)* | Adaptive Delta -- **no learning rate**; adapts from `rho`/`epsilon` alone |
| `WasmAdaBound` | `new(lr, final_lr, beta1, beta2, epsilon, gamma, weight_decay, amsbound)` *(throws)* | Bounded adaptive learning rates; prefer `create_optimizer({type: "adabound", ...})` for defaults |
| `WasmRanger` | `new(lr, beta1, beta2, epsilon, weight_decay, lookahead_k, lookahead_alpha)` *(throws)* | RAdam + Lookahead; prefer `create_optimizer({type: "ranger", ...})` for defaults |

All optimizers expose `.step(params, gradients)`, most expose `.step_list(params, gradients, dim)` for batches of same-size parameter groups, a `.learning_rate` getter/setter (except `WasmAdaDelta`/`WasmAdaBound`/`WasmRanger`, which have no learning rate), `.reset()` (except `WasmSGD`), and `.name()`.

## Available Schedulers

| Class | Constructor | Description |
|-------|-------------|-------------|
| `WasmConstantScheduler` | `new(lr)` | Constant learning rate |
| `WasmLinearDecay` | `new(initial_lr, final_lr, total_steps)` | Linear interpolation |
| `WasmStepDecay` | `new(initial_lr, step_size, gamma)` | Step-wise decay |
| `WasmExponentialDecay` | `new(initial_lr, decay_rate, decay_steps)` | Exponential decay |
| `WasmCosineAnnealing` | `new(initial_lr, min_lr, t_max)` | Cosine annealing |
| `WasmCosineAnnealingWarmRestarts` | `new(initial_lr, min_lr, t_0, t_mult)` | Cosine with warm restarts; `.cycle()` / `.cycle_length()` (plain methods, not properties) |
| `WasmOneCycle` | `new(max_lr, total_steps, pct_start, div_factor, final_div_factor)` | One-cycle policy |
| `WasmLinearWarmupDecay` | `new(initial_lr, warmup_steps, total_steps, min_lr)` | Linear warmup then linear decay |
| `WasmCyclicLR` | `new(base_lr, max_lr, step_size)` / `static new_triangular2(...)` / `static new_exp_range(base_lr, max_lr, step_size, gamma)` | Cyclic learning rate (triangular by default) |
| `WasmReduceOnPlateau` | `new(initial_lr, factor, patience)` | Reduce on plateau via `.step_with_metric(metric)` |
| `WasmViTLayerDecay` | `new(base_lr, decay_rate, num_layers)` / `static new_with_warmup(base_lr, decay_rate, num_layers, warmup_steps, total_steps)` | Vision Transformer per-layer decay; `.get_layer_learning_rate(i)` / `.get_all_layer_rates()` |
| `WasmAttentionAwareScheduler` | `new(base_lr, warmup_steps, total_steps)` | Transformer component-specific LR via `.get_component_lr(name)` / `.set_component_scale(name, scale)` |
| `WasmNoiseInjectionScheduler` | `static new_uniform/new_gaussian/new_cyclical/new_decaying(...)` (no plain constructor) | Adds noise on top of a constant base LR |
| `WasmCurriculumScheduler` | `new(stages_json, final_lr)` *(throws)* / `static new_immediate(stages_json, final_lr)` *(throws)* | Stage-based curriculum; `stages_json` is `[{"learning_rate": 0.01, "duration": 100}, ...]` |

All schedulers expose `.step()` (advance by one step, returns the new `f64` learning rate), a `.learning_rate` readonly getter, `.reset()`, and `.name()`.

## Metrics Collection

```typescript
import init, { WasmMetricsCollector } from '@cooljapan/optirs';

await init();

const metrics = new WasmMetricsCollector();
metrics.register_optimizer('adam');
metrics.update('adam', 0.001, gradsArray, paramsBeforeArray, paramsAfterArray);

console.log(metrics.summary_report());
```

## WebGPU (Experimental)

`WasmGpuOptimizer` (behind the `webgpu` Cargo feature) performs real WebGPU
capability detection and device acquisition -- `is_available()` checks
`navigator.gpu` honestly (never hardcoded), and `initialize()` runs an actual
`requestAdapter()` / `requestDevice()` handshake, failing with a real error
when no WebGPU host is present. Running optimizer compute kernels *on* the
acquired GPU device (WGSL shaders per optimizer) is not yet implemented.

```typescript
import init, { WasmGpuOptimizer } from '@cooljapan/optirs';

await init();

if (WasmGpuOptimizer.is_available()) {
  const gpu = new WasmGpuOptimizer();
  await gpu.initialize();
  console.log(gpu.device_info());
}
```

## Build from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for bundler (webpack, rollup, etc.)
./build-wasm.sh bundler

# Build for web (ES modules)
./build-wasm.sh web

# Build for Node.js
./build-wasm.sh nodejs

# Build all targets
./build-wasm.sh all
```

## TypeScript Support

Full TypeScript definitions are included. Import types directly:

```typescript
import type { WasmOptimizerConfig } from '@cooljapan/optirs';
```

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+
- Node.js 12+

## Performance

- Compiled with `opt-level = 3`, `codegen-units = 1` (the workspace release profile; `lto` is currently **off**, see `Cargo.toml`'s `[profile.release]`)
- No JavaScript overhead in core computation
- SIMD is not enabled by default; opt in with `RUSTFLAGS="-C target-feature=+simd128"` if your deployment target supports the WASM SIMD proposal

## Testing

```bash
cargo nextest run -p optirs-wasm --all-features
```

37 tests pass on the host target: 7 unit tests under `src/`, plus 30 integration tests in
`tests/wasm_tests.rs`. The `#[wasm_bindgen]`-exported surface itself -- `create_optimizer`/
`create_scheduler` returning a real, usable `JsValue`-wrapped instance, and
`WasmGpuOptimizer`'s WebGPU detection/handshake -- is covered separately by
`wasm_bindgen_test`s in `tests/wasm_bindgen_tests.rs` (7 tests) and
`tests/wasm_bindgen_webgpu_tests.rs` (3 tests), which build and run only on the real
`wasm32` target:

```bash
wasm-pack test --node --features wasm      # or: npm test
wasm-pack test --node --features webgpu    # or: npm run test:webgpu
```

## Links

- [OptiRS Documentation](https://docs.rs/optirs)
- [OptiRS Repository](https://github.com/cool-japan/optirs)
- [npm Package](https://www.npmjs.com/package/@cooljapan/optirs)

## License

Apache-2.0

Copyright (c) 2026 COOLJAPAN OU (Team Kitasan)
