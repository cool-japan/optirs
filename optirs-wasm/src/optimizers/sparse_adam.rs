//! WASM wrapper for the SparseAdam optimizer.

use crate::types::{array1_to_vec, slice_to_array1};
use optirs_core::optimizers::{Optimizer, SparseAdam};
use scirs2_core::ndarray::Ix1;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the SparseAdam optimizer.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmSparseAdam {
    inner: SparseAdam<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmSparseAdam {
    /// Create a new SparseAdam optimizer with the given learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(lr: f64) -> Self {
        Self {
            inner: SparseAdam::new(lr),
        }
    }

    /// Create a new SparseAdam optimizer with full configuration.
    pub fn new_with_config(
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            inner: SparseAdam::new_with_config(lr, beta1, beta2, epsilon, weight_decay),
        }
    }

    /// Perform a single (dense) optimization step on a 1-D parameter array.
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

    /// Perform a sparse optimization step.
    ///
    /// `indices` contains the indices of non-zero gradient values,
    /// `values` contains the corresponding gradient values, and
    /// `total_dim` is the total dimension of the parameter vector.
    pub fn step_sparse(
        &mut self,
        params: &[f64],
        indices: &[usize],
        values: &[f64],
        total_dim: usize,
    ) -> Result<Vec<f64>, String> {
        use optirs_core::optimizers::SparseGradient;

        let p = slice_to_array1(params);
        let sparse_grad = SparseGradient::new(indices.to_vec(), values.to_vec(), total_dim);
        let result = self
            .inner
            .step_sparse(&p, &sparse_grad)
            .map_err(|e| e.to_string())?;
        Ok(array1_to_vec(result))
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

    /// Get beta1 parameter.
    pub fn get_beta1(&self) -> f64 {
        self.inner.get_beta1()
    }

    /// Set beta1 parameter.
    pub fn set_beta1(&mut self, beta1: f64) {
        self.inner.set_beta1(beta1);
    }

    /// Get beta2 parameter.
    pub fn get_beta2(&self) -> f64 {
        self.inner.get_beta2()
    }

    /// Set beta2 parameter.
    pub fn set_beta2(&mut self, beta2: f64) {
        self.inner.set_beta2(beta2);
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
        "SparseAdam".to_string()
    }
}
