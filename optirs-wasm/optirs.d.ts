/* tslint:disable */
/* eslint-disable */
/**
 * OptiRS WASM - TypeScript Definitions
 * High-performance deep learning optimizers and schedulers
 */

/** Initialize the WASM module */
export function init(): Promise<void>;

/** Get the version string */
export function version(): string;

/** List available optimizer types */
export function available_optimizers(): string[];

/** List available scheduler types */
export function available_schedulers(): string[];

// ===== Optimizer Configuration =====

export class WasmOptimizerConfig {
  constructor(lr: number);
  lr: number;
  weight_decay: number;
  grad_clip: number | undefined;
  beta1: number;
  beta2: number;
  epsilon: number;
  momentum: number;
  to_json(): string;
  static from_json(s: string): WasmOptimizerConfig;
}

// ===== Optimizers =====

export class WasmAdam {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdam;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmAdamW {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdamW;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmSGD {
  constructor(lr: number);
  static new_with_config(lr: number, momentum: number, weight_decay: number): WasmSGD;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  name(): string;
}

export class WasmRMSprop {
  constructor(lr: number);
  static new_with_config(lr: number, rho: number, epsilon: number, weight_decay: number): WasmRMSprop;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  name(): string;
}

export class WasmLAMB {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number, bias_correction: boolean): WasmLAMB;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmLion {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, weight_decay: number): WasmLion;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmRAdam {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmRAdam;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmAdagrad {
  constructor(lr: number);
  static new_with_config(lr: number, epsilon: number, weight_decay: number): WasmAdagrad;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  name(): string;
}

export class WasmAdaDelta {
  constructor();
  static new_with_config(rho: number, epsilon: number): WasmAdaDelta;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_count(): number;
  reset(): void;
  name(): string;
}

export class WasmAdaBound {
  constructor();
  static new_with_config(lr: number, final_lr: number, beta1: number, beta2: number, epsilon: number, gamma: number, weight_decay: number, amsbound: boolean): WasmAdaBound;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_count(): number;
  current_bounds(): [number, number];
  reset(): void;
  name(): string;
}

export class WasmRanger {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number, lookahead_k: number, lookahead_alpha: number): WasmRanger;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_count(): number;
  slow_update_count(): number;
  is_rectified(): boolean;
  reset(): void;
  name(): string;
}

export class WasmLARS {
  constructor(lr: number);
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmSparseAdam {
  constructor(lr: number);
  static new_with_config(lr: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmSparseAdam;
  step(params: Float64Array | number[], gradients: Float64Array | number[]): Float64Array;
  step_list(params: Float64Array | number[], gradients: Float64Array | number[], dim: number): Float64Array;
  learning_rate: number;
  reset(): void;
  name(): string;
}

// ===== Schedulers =====

export class WasmCosineAnnealing {
  constructor(initial_lr: number, min_lr: number, t_max: number);
  static new_with_warm_restart(initial_lr: number, min_lr: number, t_max: number): WasmCosineAnnealing;
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmCosineAnnealingWarmRestarts {
  constructor(initial_lr: number, min_lr: number, t_0: number, t_mult: number);
  step(): number;
  readonly learning_rate: number;
  readonly cycle: number;
  readonly cycle_length: number;
  reset(): void;
  name(): string;
}

export class WasmOneCycle {
  constructor(max_lr: number, total_steps: number, pct_start: number, div_factor: number, final_div_factor: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmLinearWarmupDecay {
  constructor(initial_lr: number, warmup_steps: number, total_steps: number, min_lr: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmExponentialDecay {
  constructor(initial_lr: number, decay_rate: number, decay_steps: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmStepDecay {
  constructor(initial_lr: number, step_size: number, gamma: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmCyclicLR {
  constructor(base_lr: number, max_lr: number, step_size: number);
  static new_triangular2(base_lr: number, max_lr: number, step_size: number): WasmCyclicLR;
  static new_exp_range(base_lr: number, max_lr: number, step_size: number, gamma: number): WasmCyclicLR;
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmReduceOnPlateau {
  constructor(initial_lr: number, factor: number, patience: number);
  step(): number;
  step_with_metric(metric: number): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmConstant {
  constructor(lr: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmLinearDecay {
  constructor(initial_lr: number, final_lr: number, total_steps: number);
  step(): number;
  readonly learning_rate: number;
  reset(): void;
  name(): string;
}

export class WasmViTLayerDecay {
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
  constructor(base_lr: number, warmup_steps: number, total_steps: number);
  step(): number;
  readonly learning_rate: number;
  get_component_lr(component: string): number;
  set_component_scale(component: string, scale: number): void;
  reset(): void;
  name(): string;
}

export class WasmNoiseInjectionScheduler {
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
  static new(stages_json: string, final_lr: number): WasmCurriculumScheduler;
  static new_immediate(stages_json: string, final_lr: number): WasmCurriculumScheduler;
  step(): number;
  readonly learning_rate: number;
  current_stage_info(): string;
  completed(): boolean;
  progress(): number;
  advance_stage(): boolean;
  reset(): void;
  name(): string;
}

// ===== Metrics =====

export class WasmMetricsCollector {
  constructor();
  register_optimizer(name: string): void;
  update(name: string, learning_rate: number, gradients: Float64Array | number[], params_before: Float64Array | number[], params_after: Float64Array | number[]): void;
  summary_report(): string;
  summary_json(): string;
  optimizer_metrics_json(name: string): string;
  optimizer_count(): number;
  clear(): void;
  clear_optimizer(name: string): void;
}
