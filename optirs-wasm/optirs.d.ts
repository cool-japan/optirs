/* tslint:disable */
/* eslint-disable */
/**
 * OptiRS WASM - TypeScript Definitions
 * High-performance deep learning optimizers and schedulers
 *
 * Cross-checked against the real output of `wasm-bindgen --typescript`
 * (`./build-wasm.sh web` / `nodejs`, using the same `wasm-bindgen` version
 * pinned in `Cargo.toml`) -- not hand-guessed.
 *
 * Every class below is a real `wasm_bindgen`-exported struct; a plain
 * `new(...)` signature means it is a genuine JS constructor (`new ClassName(...)`),
 * a `static` signature is a plain associated function called on the class
 * itself (`ClassName.method(...)`). A `@throws` JSDoc note corresponds to a
 * Rust `Result::Err` path and surfaces as a thrown JS exception, not a
 * return value (this is documentation only -- TypeScript itself has no
 * "throws" annotation, and `wasm-bindgen` does not encode it either).
 * `free()` / `[Symbol.dispose]()` release the underlying WASM memory; call
 * `free()` (or use a `using` declaration) when you are done with an
 * instance instead of waiting on GC, since the WASM linear memory is not
 * garbage-collected by the JS engine.
 */

/**
 * The WASM module's own startup hook (sets the `console_error_panic_hook`).
 * Marked `#[wasm_bindgen(start)]` on the Rust side, so it runs automatically
 * once when the module is instantiated -- user code normally never needs to
 * call this itself.
 */
export function init(): void;

/**
 * Default export for the `web`/`bundler` targets: the async WASM loader
 * `wasm-bindgen` generates (fetches/instantiates the `.wasm` file). This is
 * what `import init from '@cooljapan/optirs'; await init();` actually binds
 * to -- `init` here is just the local alias chosen at the import site, it is
 * a *different* function from the named `init()` above. The `nodejs` target
 * build has no such default export (the `.wasm` is loaded synchronously by
 * `require`), so this declaration only applies when importing the `web`/
 * `bundler` build.
 */
export default function init(module_or_path?: unknown): Promise<unknown>;

/** Get the version string */
export function version(): string;

/** List available optimizer types accepted by {@link create_optimizer} */
export function available_optimizers(): string[];

/** List available scheduler types accepted by {@link create_scheduler} */
export function available_schedulers(): string[];

/**
 * Build a real optimizer instance from a JSON configuration string and
 * return it directly (not a description of it). `config_json` must include
 * a `"type"` field (see {@link available_optimizers}); any other field
 * missing falls back to that optimizer's own default. Throws on an unknown
 * `type` or invalid parameters. Returns one of the `Wasm*` optimizer
 * classes below, dynamically -- there is no single narrower static type.
 */
export function create_optimizer(config_json: string): any;

/**
 * Build a real scheduler instance from a JSON configuration string and
 * return it directly. `config_json` must include a `"type"` field (see
 * {@link available_schedulers}); any other field missing falls back to a
 * documented default. Throws on an unknown `type` or invalid parameters.
 * Returns one of the `Wasm*` scheduler classes below, dynamically.
 */
export function create_scheduler(config_json: string): any;

// ===== Optimizer Configuration =====

/** General optimizer configuration for WASM */
export class WasmOptimizerConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  lr: number;
  weight_decay: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  momentum: number;
  // Getter and setter have different types (`number | undefined` vs
  // `number | null | undefined`), so `wasm-bindgen` cannot collapse this
  // into a single plain field like the others above.
  get grad_clip(): number | undefined;
  set grad_clip(value: number | null | undefined);
  /** Serialize to JSON string */
  to_json(): string;
  /** Deserialize from JSON string */
  static from_json(s: string): WasmOptimizerConfig;
}

// ===== Optimizers =====

export class WasmSGD {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, momentum: number, weight_decay: number): WasmSGD;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_momentum(): number;
  set_momentum(momentum: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  name(): string;
}

export class WasmAdam {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdam;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmAdamW {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdamW;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmRMSprop {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, rho: number, epsilon: number, weight_decay: number): WasmRMSprop;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_rho(): number;
  set_rho(rho: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmLAMB {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number, bias_correction: boolean): WasmLAMB;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmLion {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, weight_decay: number): WasmLion;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmRAdam {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmRAdam;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

export class WasmAdagrad {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, epsilon: number, weight_decay: number): WasmAdagrad;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

/** No learning rate: adapts purely from `rho` (decay) and `epsilon` (stability). */
export class WasmAdaDelta {
  free(): void;
  [Symbol.dispose](): void;
  /** @throws if `rho`/`epsilon` are out of their valid range */
  constructor(rho: number, epsilon: number);
  static default_config(): WasmAdaDelta;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  step_count(): number;
  reset(): void;
  name(): string;
}

export class WasmAdaBound {
  free(): void;
  [Symbol.dispose](): void;
  /** @throws on invalid hyperparameters */
  constructor(lr: number, final_lr: number, beta1: number, beta2: number, epsilon: number, gamma: number, weight_decay: number, amsbound: boolean);
  static default_config(): WasmAdaBound;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  step_count(): number;
  /** `[lower, upper]` */
  current_bounds(): Float64Array;
  reset(): void;
  name(): string;
}

export class WasmRanger {
  free(): void;
  [Symbol.dispose](): void;
  /** @throws on invalid hyperparameters */
  constructor(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number, lookahead_k: number, lookahead_alpha: number);
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  step_count(): number;
  slow_update_count(): number;
  is_rectified(): boolean;
  reset(): void;
  name(): string;
}

export class WasmLARS {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, momentum: number, weight_decay: number, trust_coefficient: number, eps: number): WasmLARS;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmSparseAdam {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmSparseAdam;
  step(params: Float64Array, gradients: Float64Array): Float64Array;
  step_list(params: Float64Array, gradients: Float64Array, dim: number): Float64Array;
  /** Sparse-gradient step: `indices` holds the non-zero gradient positions. */
  step_sparse(params: Float64Array, indices: Uint32Array, values: Float64Array, total_dim: number): Float64Array;
  learning_rate: number;
  get_beta1(): number;
  set_beta1(beta1: number): void;
  get_beta2(): number;
  set_beta2(beta2: number): void;
  get_epsilon(): number;
  set_epsilon(epsilon: number): void;
  get_weight_decay(): number;
  set_weight_decay(weight_decay: number): void;
  reset(): void;
  name(): string;
}

// ===== Schedulers =====

export class WasmCosineAnnealing {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, min_lr: number, t_max: number);
  static new_with_warm_restart(initial_lr: number, min_lr: number, t_max: number): WasmCosineAnnealing;
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmCosineAnnealingWarmRestarts {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, min_lr: number, t_0: number, t_mult: number);
  step(): number;
  readonly learning_rate: number;
  /** Plain method, not a property -- no `#[wasm_bindgen(getter)]` on the Rust side. */
  cycle(): number;
  /** Plain method, not a property -- no `#[wasm_bindgen(getter)]` on the Rust side. */
  cycle_length(): number;
  reset(): void;
  name(): string;
}

export class WasmOneCycle {
  free(): void;
  [Symbol.dispose](): void;
  constructor(max_lr: number, total_steps: number, pct_start: number, div_factor: number, final_div_factor: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmLinearWarmupDecay {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, warmup_steps: number, total_steps: number, min_lr: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmExponentialDecay {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, decay_rate: number, decay_steps: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmStepDecay {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, step_size: number, gamma: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmCyclicLR {
  free(): void;
  [Symbol.dispose](): void;
  /** Triangular (default) cyclic schedule. */
  constructor(base_lr: number, max_lr: number, step_size: number);
  static new_triangular2(base_lr: number, max_lr: number, step_size: number): WasmCyclicLR;
  static new_exp_range(base_lr: number, max_lr: number, step_size: number, gamma: number): WasmCyclicLR;
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmReduceOnPlateau {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, factor: number, patience: number);
  /** No-op (returns the current LR) without a metric -- see `step_with_metric`. */
  step(): number;
  step_with_metric(metric: number): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmConstantScheduler {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lr: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmLinearDecay {
  free(): void;
  [Symbol.dispose](): void;
  constructor(initial_lr: number, final_lr: number, total_steps: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmViTLayerDecay {
  free(): void;
  [Symbol.dispose](): void;
  constructor(base_lr: number, decay_rate: number, num_layers: number);
  static new_with_warmup(base_lr: number, decay_rate: number, num_layers: number, warmup_steps: number, total_steps: number): WasmViTLayerDecay;
  step(): number;
  readonly learning_rate: number;
  get_layer_learning_rate(layer_idx: number): number;
  get_all_layer_rates(): Float64Array;
  reset(): void;
  name(): string;
}

export class WasmAttentionAwareScheduler {
  free(): void;
  [Symbol.dispose](): void;
  constructor(base_lr: number, warmup_steps: number, total_steps: number);
  step(): number;
  readonly learning_rate: number;
  /** @throws if `component` is not one of "attention"/"feed_forward"/"embedding"/"layer_norm"/"output" */
  get_component_lr(component: string): number;
  /** @throws if `component` is not one of "attention"/"feed_forward"/"embedding"/"layer_norm"/"output" */
  set_component_scale(component: string, scale: number): void;
  reset(): void;
  name(): string;
}

/** No plain constructor: pick a noise distribution via one of the `static new_*` factories. */
export class WasmNoiseInjectionScheduler {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static new_uniform(base_lr: number, min_noise: number, max_noise: number, min_lr: number): WasmNoiseInjectionScheduler;
  static new_gaussian(base_lr: number, mean: number, std_dev: number, min_lr: number): WasmNoiseInjectionScheduler;
  static new_cyclical(base_lr: number, amplitude: number, period: number, min_lr: number): WasmNoiseInjectionScheduler;
  static new_decaying(base_lr: number, initial_scale: number, final_scale: number, decay_steps: number, min_lr: number): WasmNoiseInjectionScheduler;
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmCurriculumScheduler {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * `stages_json`: `[{"learning_rate": 0.01, "duration": 100}, ...]`
   * (smooth transitions between stages).
   * @throws if `stages_json` is malformed or empty
   */
  constructor(stages_json: string, final_lr: number);
  /** Same as the constructor but transitions between stages immediately. */
  static new_immediate(stages_json: string, final_lr: number): WasmCurriculumScheduler;
  step(): number;
  readonly learning_rate: number;
  /** JSON object: `{"learning_rate": ..., "duration": ..., "description"?: ...}` */
  current_stage_info(): string;
  completed(): boolean;
  progress(): number;
  advance_stage(): boolean;
  reset(): void;
  name(): string;
}

// ===== Metrics =====

export class WasmMetricsCollector {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  register_optimizer(name: string): void;
  update(name: string, learning_rate: number, gradients: Float64Array, params_before: Float64Array, params_after: Float64Array): void;
  summary_report(): string;
  summary_json(): string;
  /** @throws if `name` was never registered */
  optimizer_metrics_json(name: string): string;
  optimizer_count(): number;
  clear(): void;
  clear_optimizer(name: string): void;
}

// ===== WebGPU (requires the `webgpu` Cargo feature) =====

/**
 * Real WebGPU capability detection and device acquisition. Running
 * optimizer compute kernels on the acquired device is not yet implemented.
 */
export class WasmGpuOptimizer {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  /** Synchronous check: does `navigator.gpu` exist? Does not guarantee a device can be acquired. */
  static is_available(): boolean;
  /** Runs a real `requestAdapter()`/`requestDevice()` handshake. @throws if WebGPU is unavailable or the browser declines. */
  initialize(): Promise<void>;
  device_info(): string;
  is_initialized(): boolean;
}
