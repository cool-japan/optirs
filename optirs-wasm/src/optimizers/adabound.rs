//! WASM wrapper for the AdaBound optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::AdaBound;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the AdaBound optimizer.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmAdaBound {
    inner: AdaBound<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmAdaBound {
    /// Create a new AdaBound optimizer with the given parameters.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lr: f64,
        final_lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        gamma: f64,
        weight_decay: f64,
        amsbound: bool,
    ) -> Result<WasmAdaBound, String> {
        let inner = AdaBound::new(
            lr,
            final_lr,
            beta1,
            beta2,
            epsilon,
            gamma,
            weight_decay,
            amsbound,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    /// Create a new AdaBound optimizer with default parameters.
    pub fn default_config() -> Self {
        Self {
            inner: AdaBound::default(),
        }
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

    /// Get the current learning rate bounds as [lower, upper].
    pub fn current_bounds(&self) -> Vec<f64> {
        let (lower, upper) = self.inner.current_bounds();
        vec![lower, upper]
    }

    /// Get the optimizer name.
    pub fn name(&self) -> String {
        "AdaBound".to_string()
    }
}
