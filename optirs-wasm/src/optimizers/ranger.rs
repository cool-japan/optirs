//! WASM wrapper for the Ranger optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::Ranger;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the Ranger optimizer.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmRanger {
    inner: Ranger<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmRanger {
    /// Create a new Ranger optimizer with the given parameters.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
        lookahead_k: usize,
        lookahead_alpha: f64,
    ) -> Result<WasmRanger, String> {
        let inner = Ranger::new(
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            lookahead_k,
            lookahead_alpha,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    /// Perform a single optimization step on a 1-D parameter array.
    pub fn step(&mut self, params: &[f64], gradients: &[f64]) -> Result<Vec<f64>, String> {
        let p = slice_to_array1(params);
        let g = slice_to_array1(gradients);
        let result = self
            .inner
            .step(p.view(), g.view())
            .map_err(|e| e.to_string())?;
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
            let result = self
                .inner
                .step(p.view(), g.view())
                .map_err(|e| e.to_string())?;
            results.extend(array1_to_vec(result));
        }
        Ok(results)
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    /// Get the number of slow weight updates (lookahead updates).
    pub fn slow_update_count(&self) -> usize {
        self.inner.slow_update_count()
    }

    /// Check whether the optimizer is currently using rectified updates.
    pub fn is_rectified(&self) -> bool {
        self.inner.is_rectified()
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the optimizer name.
    pub fn name(&self) -> String {
        "Ranger".to_string()
    }
}
