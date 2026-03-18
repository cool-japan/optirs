//! WASM wrapper for the LARS optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::{Optimizer, LARS};
use scirs2_core::ndarray::Ix1;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the LARS optimizer.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmLARS {
    inner: LARS<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmLARS {
    /// Create a new LARS optimizer with the given learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(lr: f64) -> Self {
        Self {
            inner: LARS::new(lr),
        }
    }

    /// Create a new LARS optimizer with momentum and weight decay.
    pub fn new_with_config(
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        trust_coefficient: f64,
        eps: f64,
    ) -> Self {
        Self {
            inner: LARS::new(lr)
                .with_momentum(momentum)
                .with_weight_decay(weight_decay)
                .with_trust_coefficient(trust_coefficient)
                .with_eps(eps),
        }
    }

    /// Perform a single optimization step on a 1-D parameter array.
    pub fn step(&mut self, params: &[f64], gradients: &[f64]) -> Result<Vec<f64>, String> {
        let p = slice_to_array1(params);
        let g = slice_to_array1(gradients);
        let result =
            Optimizer::<f64, Ix1>::step(&mut self.inner, &p, &g).map_err(|e| e.to_string())?;
        Ok(array1_to_vec(result))
    }

    /// Perform optimization steps on multiple parameter groups packed into flat arrays.
    pub fn step_list(
        &mut self,
        params: &[f64],
        gradients: &[f64],
        dim: usize,
    ) -> Result<Vec<f64>, String> {
        if params.len() != gradients.len() {
            return Err("Parameters and gradients must have the same length".to_string());
        }
        if !params.len().is_multiple_of(dim) {
            return Err(format!(
                "Array length {} is not divisible by dim {}",
                params.len(),
                dim
            ));
        }
        let mut results = Vec::with_capacity(params.len());
        for (p_chunk, g_chunk) in params.chunks(dim).zip(gradients.chunks(dim)) {
            let p = slice_to_array1(p_chunk);
            let g = slice_to_array1(g_chunk);
            let result =
                Optimizer::<f64, Ix1>::step(&mut self.inner, &p, &g).map_err(|e| e.to_string())?;
            results.extend(array1_to_vec(result));
        }
        Ok(results)
    }

    /// Get the current learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn learning_rate(&self) -> f64 {
        Optimizer::<f64, Ix1>::get_learning_rate(&self.inner)
    }

    /// Set the learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(setter))]
    pub fn set_learning_rate(&mut self, lr: f64) {
        Optimizer::<f64, Ix1>::set_learning_rate(&mut self.inner, lr);
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the optimizer name.
    pub fn name(&self) -> String {
        "LARS".to_string()
    }
}
