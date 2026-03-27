# OptiRS WASM - WebAssembly Bindings for OptiRS

**Version:** 0.3.1  
**Status:** Production Ready

High-performance WebAssembly bindings for OptiRS deep learning optimizers and learning rate schedulers. Run state-of-the-art ML optimization algorithms in the browser and Node.js.

## Features

- **11 Optimizers** - SGD, Adam, AdamW, RAdam, LAMB, Lion, LARS, Adagrad, AdaDelta, AdaBound, Ranger
- **12+ Schedulers** - CosineAnnealing, WarmRestarts, CyclicLR, ViTLayerDecay, and more
- **TypeScript Support** - Complete type definitions included
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
  import init, { WasmAdam, WasmCosineAnnealing } from '@cooljapan/optirs';
  await init();
  
  const optimizer = WasmAdam.new(0.001);
</script>
```

## Quick Start

### Basic Optimization

```typescript
import init, { WasmAdam, WasmCosineAnnealing } from '@cooljapan/optirs';

await init();

// Create optimizer
const adam = WasmAdam.new(0.001);

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

const optimizer = WasmAdamW.new_with_weight_decay(0.001, 0.01);
const scheduler = WasmCosineAnnealingWarmRestarts.new(0.001, 10, 2);

for (let epoch = 0; epoch < 100; epoch++) {
  // Training step...
  const lr = scheduler.step();
  optimizer.set_learning_rate(lr);
}
```

### Advanced Configuration

```typescript
import init, { WasmOptimizerConfig, WasmAdam } from '@cooljapan/optirs';

await init();

const config = WasmOptimizerConfig.new();
config.learning_rate = 0.001;
config.beta1 = 0.9;
config.beta2 = 0.999;
config.epsilon = 1e-8;
config.weight_decay = 0.01;

const optimizer = WasmAdam.from_config(config);
```

## Available Optimizers

| Optimizer | Constructor | Description |
|-----------|-------------|-------------|
| `WasmSgd` | `new(lr)` | Stochastic Gradient Descent |
| `WasmAdam` | `new(lr)` | Adaptive Moment Estimation |
| `WasmAdamW` | `new_with_weight_decay(lr, wd)` | Adam with decoupled weight decay |
| `WasmRAdam` | `new(lr)` | Rectified Adam |
| `WasmLamb` | `new(lr)` | Layer-wise Adaptive Moments |
| `WasmLion` | `new(lr)` | Evolved Sign Momentum |
| `WasmLars` | `new(lr)` | Layer-wise Adaptive Rate Scaling |
| `WasmAdagrad` | `new(lr)` | Adaptive Gradient |
| `WasmAdaDelta` | `new(rho)` | Adaptive Delta |
| `WasmAdaBound` | `new(lr)` | Bounded adaptive learning rates |
| `WasmRanger` | `new(lr)` | RAdam + Lookahead |

## Available Schedulers

| Scheduler | Constructor | Description |
|-----------|-------------|-------------|
| `WasmConstant` | `new(lr)` | Constant learning rate |
| `WasmLinearDecay` | `new(start, end, steps)` | Linear interpolation |
| `WasmStepDecay` | `new(lr, gamma, step_size)` | Step-wise decay |
| `WasmExponentialDecay` | `new(lr, gamma)` | Exponential decay |
| `WasmCosineAnnealing` | `new(lr, t_max)` | Cosine annealing |
| `WasmCosineAnnealingWarmRestarts` | `new(lr, t_0, t_mult)` | Cosine with warm restarts |
| `WasmCyclicLR` | `new(base, max, step_up)` | Cyclic learning rate |
| `WasmReduceOnPlateau` | `new(lr, factor, patience)` | Reduce on plateau |
| `WasmViTLayerDecay` | `new(base, decay, layers)` | Vision Transformer layer decay |
| `WasmAttentionAwareScheduler` | `new(base)` | Transformer component-specific LR |

## Metrics Collection

```typescript
import init, { WasmMetricsCollector } from '@cooljapan/optirs';

await init();

const metrics = WasmMetricsCollector.new();
metrics.record_learning_rate(0.001);
metrics.record_gradient_norm(0.05);

const stats = metrics.get_stats();
console.log(stats);
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

OptiRS WASM provides near-native performance for optimization:

- Compiled with optimizations (`opt-level = 3`, `lto = true`)
- No JavaScript overhead in core computation
- SIMD support where available

## Links

- [OptiRS Documentation](https://docs.rs/optirs)
- [OptiRS Repository](https://github.com/cool-japan/optirs)
- [npm Package](https://www.npmjs.com/package/@cooljapan/optirs)

## License

Apache-2.0

Copyright (c) 2026 COOLJAPAN OU (Team Kitasan)
