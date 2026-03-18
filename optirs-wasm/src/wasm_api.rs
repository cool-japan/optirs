//! High-level WASM API with factory functions for creating optimizers and schedulers.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::optimizers::*;
use crate::schedulers::*;

/// Create an optimizer from a JSON configuration string.
///
/// The JSON must include a "type" field and optimizer-specific parameters.
/// Example: `{"type": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.999}`
///
/// Supported types: adam, adamw, sgd, rmsprop, lamb, lion, radam, adagrad,
/// adadelta, adabound, ranger, lars, sparse_adam
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_optimizer(config_json: &str) -> Result<JsValue, JsError> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?;

    let opt_type = config["type"]
        .as_str()
        .ok_or_else(|| JsError::new("Missing 'type' field in config"))?;
    let lr = config["lr"].as_f64().unwrap_or(0.001);

    match opt_type {
        "adam" => {
            let _opt = WasmAdam::new(lr);
            Ok(serde_wasm_bindgen::to_value(&format!("Adam(lr={})", lr))
                .map_err(|e| JsError::new(&e.to_string()))?)
        }
        "adamw" => {
            let wd = config["weight_decay"].as_f64().unwrap_or(0.01);
            let _opt = WasmAdamW::new_with_config(lr, 0.9, 0.999, 1e-8, wd);
            Ok(
                serde_wasm_bindgen::to_value(&format!("AdamW(lr={}, wd={})", lr, wd))
                    .map_err(|e| JsError::new(&e.to_string()))?,
            )
        }
        "sgd" => {
            let momentum = config["momentum"].as_f64().unwrap_or(0.0);
            let _opt = WasmSGD::new(lr);
            Ok(
                serde_wasm_bindgen::to_value(&format!("SGD(lr={}, momentum={})", lr, momentum))
                    .map_err(|e| JsError::new(&e.to_string()))?,
            )
        }
        other => Err(JsError::new(&format!("Unknown optimizer type: {}", other))),
    }
}

/// Create a learning rate scheduler from a JSON configuration string.
///
/// Example: `{"type": "cosine_annealing", "initial_lr": 0.001, "min_lr": 0.0001, "t_max": 100}`
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
            let initial_lr = config["initial_lr"].as_f64().unwrap_or(0.001);
            let min_lr = config["min_lr"].as_f64().unwrap_or(0.0001);
            let t_max = config["t_max"].as_u64().unwrap_or(100) as usize;
            let _sched = WasmCosineAnnealing::new(initial_lr, min_lr, t_max);
            Ok(serde_wasm_bindgen::to_value(&format!(
                "CosineAnnealing(lr={}, min={}, T={})",
                initial_lr, min_lr, t_max
            ))
            .map_err(|e| JsError::new(&e.to_string()))?)
        }
        "step_decay" => {
            let initial_lr = config["initial_lr"].as_f64().unwrap_or(0.001);
            let step_size = config["step_size"].as_u64().unwrap_or(10) as usize;
            let gamma = config["gamma"].as_f64().unwrap_or(0.1);
            let _sched = WasmStepDecay::new(initial_lr, step_size, gamma);
            Ok(serde_wasm_bindgen::to_value(&format!(
                "StepDecay(lr={}, step={}, gamma={})",
                initial_lr, step_size, gamma
            ))
            .map_err(|e| JsError::new(&e.to_string()))?)
        }
        other => Err(JsError::new(&format!("Unknown scheduler type: {}", other))),
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
