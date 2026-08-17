//! High-level WASM API with factory functions for creating optimizers and schedulers.
//!
//! [`create_optimizer`] and [`create_scheduler`] build a *real* optimizer/scheduler
//! object from a JSON configuration and hand it back to JavaScript as the actual
//! `Wasm*` instance (via `wasm_bindgen`'s generated `From<T> for JsValue`), so the
//! caller can immediately call `.step(...)` etc. on the returned value. They do not
//! return a stringified description of what *would* have been constructed.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::optimizers::*;
use crate::schedulers::*;

/// Read an f64 field from a JSON config object, falling back to `default`.
#[cfg(feature = "wasm")]
fn f64_field(config: &serde_json::Value, key: &str, default: f64) -> f64 {
    config[key].as_f64().unwrap_or(default)
}

/// Read a usize field from a JSON config object, falling back to `default`.
#[cfg(feature = "wasm")]
fn usize_field(config: &serde_json::Value, key: &str, default: usize) -> usize {
    config[key].as_u64().map(|v| v as usize).unwrap_or(default)
}

/// Read a bool field from a JSON config object, falling back to `default`.
#[cfg(feature = "wasm")]
fn bool_field(config: &serde_json::Value, key: &str, default: bool) -> bool {
    config[key].as_bool().unwrap_or(default)
}

/// Create an optimizer from a JSON configuration string.
///
/// The JSON must include a `"type"` field and optimizer-specific parameters.
/// Any parameter that is omitted falls back to the same default the
/// corresponding `optirs-core` optimizer uses internally. Example:
/// `{"type": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.999}`
///
/// Returns the concrete `Wasm*` optimizer instance (e.g. a `WasmAdam`) as a
/// `JsValue`, ready to be used from JavaScript.
///
/// Supported types: `adam`, `adamw`, `sgd`, `rmsprop`, `lamb`, `lion`, `radam`,
/// `adagrad`, `adadelta`, `adabound`, `ranger`, `lars`, `sparse_adam`.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_optimizer(config_json: &str) -> Result<JsValue, JsError> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?;

    let opt_type = config["type"]
        .as_str()
        .ok_or_else(|| JsError::new("Missing 'type' field in config"))?;
    let lr = f64_field(&config, "lr", 0.001);

    match opt_type {
        "adam" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmAdam::new_with_config(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            )))
        }
        "adamw" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            // AdamW's own default weight decay (0.01) differs from Adam's (0.0).
            let weight_decay = f64_field(&config, "weight_decay", 0.01);
            Ok(JsValue::from(WasmAdamW::new_with_config(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            )))
        }
        "sgd" => {
            let momentum = f64_field(&config, "momentum", 0.0);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmSGD::new_with_config(
                lr,
                momentum,
                weight_decay,
            )))
        }
        "rmsprop" => {
            let rho = f64_field(&config, "rho", 0.9);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmRMSprop::new_with_config(
                lr,
                rho,
                epsilon,
                weight_decay,
            )))
        }
        "lamb" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-6);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            let bias_correction = bool_field(&config, "bias_correction", true);
            Ok(JsValue::from(WasmLAMB::new_with_config(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                bias_correction,
            )))
        }
        "lion" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.99);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmLion::new_with_config(
                lr,
                beta1,
                beta2,
                weight_decay,
            )))
        }
        "radam" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmRAdam::new_with_config(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            )))
        }
        "adagrad" => {
            let epsilon = f64_field(&config, "epsilon", 1e-10);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmAdagrad::new_with_config(
                lr,
                epsilon,
                weight_decay,
            )))
        }
        "adadelta" => {
            // AdaDelta has no learning rate; it adapts from `rho` and `epsilon`.
            let rho = f64_field(&config, "rho", 0.95);
            let epsilon = f64_field(&config, "epsilon", 1e-6);
            let opt = WasmAdaDelta::new(rho, epsilon).map_err(|e| JsError::new(&e))?;
            Ok(JsValue::from(opt))
        }
        "adabound" => {
            let final_lr = f64_field(&config, "final_lr", 0.1);
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let gamma = f64_field(&config, "gamma", 1e-3);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            let amsbound = bool_field(&config, "amsbound", false);
            let opt = WasmAdaBound::new(
                lr,
                final_lr,
                beta1,
                beta2,
                epsilon,
                gamma,
                weight_decay,
                amsbound,
            )
            .map_err(|e| JsError::new(&e))?;
            Ok(JsValue::from(opt))
        }
        "ranger" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            let lookahead_k = usize_field(&config, "lookahead_k", 5);
            let lookahead_alpha = f64_field(&config, "lookahead_alpha", 0.5);
            let opt = WasmRanger::new(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                lookahead_k,
                lookahead_alpha,
            )
            .map_err(|e| JsError::new(&e))?;
            Ok(JsValue::from(opt))
        }
        "lars" => {
            let momentum = f64_field(&config, "momentum", 0.9);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            let trust_coefficient = f64_field(&config, "trust_coefficient", 0.001);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            Ok(JsValue::from(WasmLARS::new_with_config(
                lr,
                momentum,
                weight_decay,
                trust_coefficient,
                epsilon,
            )))
        }
        "sparse_adam" => {
            let beta1 = f64_field(&config, "beta1", 0.9);
            let beta2 = f64_field(&config, "beta2", 0.999);
            let epsilon = f64_field(&config, "epsilon", 1e-8);
            let weight_decay = f64_field(&config, "weight_decay", 0.0);
            Ok(JsValue::from(WasmSparseAdam::new_with_config(
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            )))
        }
        other => Err(JsError::new(&format!(
            "Unknown optimizer type: {}. Supported types: {}",
            other,
            available_optimizers().join(", ")
        ))),
    }
}

/// Create a learning rate scheduler from a JSON configuration string.
///
/// Any parameter that is omitted falls back to a documented, reasonable
/// default for that scheduler. Example:
/// `{"type": "cosine_annealing", "initial_lr": 0.001, "min_lr": 0.0001, "t_max": 100}`
///
/// Returns the concrete `Wasm*` scheduler instance as a `JsValue`.
///
/// Supported types: `cosine_annealing`, `cosine_annealing_warm_restarts`,
/// `one_cycle`, `linear_warmup_decay`, `exponential_decay`, `step_decay`,
/// `cyclic_lr`, `reduce_on_plateau`, `constant`, `linear_decay`,
/// `vit_layer_decay`, `attention_aware`, `noise_injection`, `curriculum`.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_scheduler(config_json: &str) -> Result<JsValue, JsError> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?;

    let sched_type = config["type"]
        .as_str()
        .ok_or_else(|| JsError::new("Missing 'type' field in config"))?;

    match sched_type {
        "cosine_annealing" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let min_lr = f64_field(&config, "min_lr", 0.0001);
            let t_max = usize_field(&config, "t_max", 100);
            Ok(JsValue::from(WasmCosineAnnealing::new(
                initial_lr, min_lr, t_max,
            )))
        }
        "cosine_annealing_warm_restarts" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let min_lr = f64_field(&config, "min_lr", 0.0001);
            let t_0 = usize_field(&config, "t_0", 10);
            let t_mult = f64_field(&config, "t_mult", 2.0);
            Ok(JsValue::from(WasmCosineAnnealingWarmRestarts::new(
                initial_lr, min_lr, t_0, t_mult,
            )))
        }
        "one_cycle" => {
            let max_lr = f64_field(&config, "max_lr", 0.01);
            let total_steps = usize_field(&config, "total_steps", 1000);
            let pct_start = f64_field(&config, "pct_start", 0.3);
            let div_factor = f64_field(&config, "div_factor", 25.0);
            let final_div_factor = f64_field(&config, "final_div_factor", 1e4);
            Ok(JsValue::from(WasmOneCycle::new(
                max_lr,
                total_steps,
                pct_start,
                div_factor,
                final_div_factor,
            )))
        }
        "linear_warmup_decay" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let warmup_steps = usize_field(&config, "warmup_steps", 100);
            let total_steps = usize_field(&config, "total_steps", 1000);
            let min_lr = f64_field(&config, "min_lr", 0.0);
            Ok(JsValue::from(WasmLinearWarmupDecay::new(
                initial_lr,
                warmup_steps,
                total_steps,
                min_lr,
            )))
        }
        "exponential_decay" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let decay_rate = f64_field(&config, "decay_rate", 0.96);
            let decay_steps = usize_field(&config, "decay_steps", 100);
            Ok(JsValue::from(WasmExponentialDecay::new(
                initial_lr,
                decay_rate,
                decay_steps,
            )))
        }
        "step_decay" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let step_size = usize_field(&config, "step_size", 10);
            let gamma = f64_field(&config, "gamma", 0.1);
            Ok(JsValue::from(WasmStepDecay::new(
                initial_lr, step_size, gamma,
            )))
        }
        "cyclic_lr" => {
            let base_lr = f64_field(&config, "base_lr", 0.0001);
            let max_lr = f64_field(&config, "max_lr", 0.001);
            let step_size = usize_field(&config, "step_size", 2000);
            let sched = match config["mode"].as_str().unwrap_or("triangular") {
                "triangular2" => WasmCyclicLR::new_triangular2(base_lr, max_lr, step_size),
                "exp_range" => {
                    let gamma = f64_field(&config, "gamma", 0.999_94);
                    WasmCyclicLR::new_exp_range(base_lr, max_lr, step_size, gamma)
                }
                _ => WasmCyclicLR::new(base_lr, max_lr, step_size),
            };
            Ok(JsValue::from(sched))
        }
        "reduce_on_plateau" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let factor = f64_field(&config, "factor", 0.1);
            let patience = usize_field(&config, "patience", 10);
            Ok(JsValue::from(WasmReduceOnPlateau::new(
                initial_lr, factor, patience,
            )))
        }
        "constant" => {
            let lr = f64_field(&config, "lr", 0.001);
            Ok(JsValue::from(WasmConstantScheduler::new(lr)))
        }
        "linear_decay" => {
            let initial_lr = f64_field(&config, "initial_lr", 0.001);
            let final_lr = f64_field(&config, "final_lr", 0.0001);
            let total_steps = usize_field(&config, "total_steps", 1000);
            Ok(JsValue::from(WasmLinearDecay::new(
                initial_lr,
                final_lr,
                total_steps,
            )))
        }
        "vit_layer_decay" => {
            let base_lr = f64_field(&config, "base_lr", 0.001);
            let decay_rate = f64_field(&config, "decay_rate", 0.75);
            let num_layers = usize_field(&config, "num_layers", 12);
            let sched = if config.get("warmup_steps").is_some() {
                let warmup_steps = usize_field(&config, "warmup_steps", 0);
                let total_steps = usize_field(&config, "total_steps", 1000);
                WasmViTLayerDecay::new_with_warmup(
                    base_lr,
                    decay_rate,
                    num_layers,
                    warmup_steps,
                    total_steps,
                )
            } else {
                WasmViTLayerDecay::new(base_lr, decay_rate, num_layers)
            };
            Ok(JsValue::from(sched))
        }
        "attention_aware" => {
            let base_lr = f64_field(&config, "base_lr", 0.001);
            let warmup_steps = usize_field(&config, "warmup_steps", 100);
            let total_steps = usize_field(&config, "total_steps", 1000);
            Ok(JsValue::from(WasmAttentionAwareScheduler::new(
                base_lr,
                warmup_steps,
                total_steps,
            )))
        }
        "noise_injection" => {
            let base_lr = f64_field(&config, "base_lr", 0.001);
            let min_lr = f64_field(&config, "min_lr", 0.0);
            let sched = match config["distribution"].as_str().unwrap_or("uniform") {
                "gaussian" => {
                    let mean = f64_field(&config, "mean", 0.0);
                    let std_dev = f64_field(&config, "std_dev", 0.01);
                    WasmNoiseInjectionScheduler::new_gaussian(base_lr, mean, std_dev, min_lr)
                }
                "cyclical" => {
                    let amplitude = f64_field(&config, "amplitude", 0.01);
                    let period = usize_field(&config, "period", 100);
                    WasmNoiseInjectionScheduler::new_cyclical(base_lr, amplitude, period, min_lr)
                }
                "decaying" => {
                    let initial_scale = f64_field(&config, "initial_scale", 1.0);
                    let final_scale = f64_field(&config, "final_scale", 0.0);
                    let decay_steps = usize_field(&config, "decay_steps", 1000);
                    WasmNoiseInjectionScheduler::new_decaying(
                        base_lr,
                        initial_scale,
                        final_scale,
                        decay_steps,
                        min_lr,
                    )
                }
                _ => {
                    let min_noise = f64_field(&config, "min_noise", -0.01);
                    let max_noise = f64_field(&config, "max_noise", 0.01);
                    WasmNoiseInjectionScheduler::new_uniform(base_lr, min_noise, max_noise, min_lr)
                }
            };
            Ok(JsValue::from(sched))
        }
        "curriculum" => {
            let stages = &config["stages"];
            if stages.is_null() {
                return Err(JsError::new(
                    "curriculum scheduler requires a 'stages' array field",
                ));
            }
            let stages_json =
                serde_json::to_string(stages).map_err(|e| JsError::new(&e.to_string()))?;
            let final_lr = f64_field(&config, "final_lr", 0.0001);
            let immediate = bool_field(&config, "immediate", false);
            let sched = if immediate {
                WasmCurriculumScheduler::new_immediate(&stages_json, final_lr)
            } else {
                WasmCurriculumScheduler::new(&stages_json, final_lr)
            }
            .map_err(|e| JsError::new(&e))?;
            Ok(JsValue::from(sched))
        }
        other => Err(JsError::new(&format!(
            "Unknown scheduler type: {}. Supported types: {}",
            other,
            available_schedulers().join(", ")
        ))),
    }
}

/// List all available optimizer types
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn available_optimizers() -> Vec<String> {
    vec![
        "adam".to_string(),
        "adamw".to_string(),
        "sgd".to_string(),
        "rmsprop".to_string(),
        "lamb".to_string(),
        "lion".to_string(),
        "radam".to_string(),
        "adagrad".to_string(),
        "adadelta".to_string(),
        "adabound".to_string(),
        "ranger".to_string(),
        "lars".to_string(),
        "sparse_adam".to_string(),
    ]
}

/// List all available scheduler types
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn available_schedulers() -> Vec<String> {
    vec![
        "cosine_annealing".to_string(),
        "cosine_annealing_warm_restarts".to_string(),
        "one_cycle".to_string(),
        "linear_warmup_decay".to_string(),
        "exponential_decay".to_string(),
        "step_decay".to_string(),
        "cyclic_lr".to_string(),
        "reduce_on_plateau".to_string(),
        "constant".to_string(),
        "linear_decay".to_string(),
        "vit_layer_decay".to_string(),
        "attention_aware".to_string(),
        "noise_injection".to_string(),
        "curriculum".to_string(),
    ]
}
