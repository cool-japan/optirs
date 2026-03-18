//! WASM wrapper for the AdaDelta optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::AdaDelta;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the AdaDelta optimizer.
///
/// AdaDelta does not use a learning rate; it adapts based on a running
/// window of gradient updates using `rho` and `epsilon` parameters.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmAdaDelta {
    inner: AdaDelta<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmAdaDelta {
    /// Create a new AdaDelta optimizer with the given rho and epsilon.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(rho: f64, epsilon: f64) -> Result<WasmAdaDelta, String> {
        let inner = AdaDelta::new(rho, epsilon).map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    /// Create a new AdaDelta optimizer with default parameters.
    pub fn default_config() -> Result<WasmAdaDelta, String> {
        let inner = AdaDelta::default();
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

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the optimizer name.
    pub fn name(&self) -> String {
        "AdaDelta".to_string()
    }
}
