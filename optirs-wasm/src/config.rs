//! Configuration types for WASM optimizer bindings.

use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// General optimizer configuration for WASM
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmOptimizerConfig {
    lr: f64,
    weight_decay: f64,
    grad_clip: Option<f64>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    momentum: f64,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmOptimizerConfig {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            weight_decay: 0.0,
            grad_clip: None,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.0,
        }
    }

    // Getters
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn lr(&self) -> f64 {
        self.lr
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn grad_clip(&self) -> Option<f64> {
        self.grad_clip
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    // Setters
    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_weight_decay(&mut self, wd: f64) {
        self.weight_decay = wd;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_grad_clip(&mut self, gc: Option<f64>) {
        self.grad_clip = gc;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_beta1(&mut self, b1: f64) {
        self.beta1 = b1;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_beta2(&mut self, b2: f64) {
        self.beta2 = b2;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_epsilon(&mut self, eps: f64) {
        self.epsilon = eps;
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_momentum(&mut self, m: f64) {
        self.momentum = m;
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Deserialize from JSON string
    pub fn from_json(s: &str) -> Result<WasmOptimizerConfig, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}
