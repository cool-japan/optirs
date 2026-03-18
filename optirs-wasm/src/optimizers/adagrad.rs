//! WASM wrapper for the Adagrad optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::{Adagrad, Optimizer};
use scirs2_core::ndarray::Ix1;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the Adagrad optimizer.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmAdagrad {
    inner: Adagrad<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmAdagrad {
    /// Create a new Adagrad optimizer with the given learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(lr: f64) -> Self {
        Self {
            inner: Adagrad::new(lr),
        }
    }

    /// Create a new Adagrad optimizer with full configuration.
    pub fn new_with_config(lr: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            inner: Adagrad::new_with_config(lr, epsilon, weight_decay),
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

    /// Get epsilon parameter.
    pub fn get_epsilon(&self) -> f64 {
        self.inner.get_epsilon()
    }

    /// Set epsilon parameter.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.inner.set_epsilon(epsilon);
    }

    /// Get weight decay parameter.
    pub fn get_weight_decay(&self) -> f64 {
        self.inner.get_weight_decay()
    }

    /// Set weight decay parameter.
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.inner.set_weight_decay(weight_decay);
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the optimizer name.
    pub fn name(&self) -> String {
        "Adagrad".to_string()
    }
}
