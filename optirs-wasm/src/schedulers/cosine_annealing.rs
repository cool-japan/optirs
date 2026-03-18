//! WASM wrapper for the CosineAnnealing learning rate scheduler.

use optirs_core::schedulers::CosineAnnealing;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the CosineAnnealing scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmCosineAnnealing {
    inner: CosineAnnealing<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmCosineAnnealing {
    /// Create a new cosine annealing scheduler (without warm restarts).
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_max` - Maximum number of iterations in a cycle
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, min_lr: f64, t_max: usize) -> Self {
        Self {
            inner: CosineAnnealing::new(initial_lr, min_lr, t_max, false),
        }
    }

    /// Create a new cosine annealing scheduler with warm restarts enabled.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_max` - Maximum number of iterations in a cycle
    pub fn new_with_warm_restart(initial_lr: f64, min_lr: f64, t_max: usize) -> Self {
        Self {
            inner: CosineAnnealing::new(initial_lr, min_lr, t_max, true),
        }
    }

    /// Advance the scheduler by one step and return the new learning rate.
    pub fn step(&mut self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.step()
    }

    /// Get the current learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn learning_rate(&self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.get_learning_rate()
    }

    /// Reset the scheduler to its initial state.
    pub fn reset(&mut self) {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.reset();
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "CosineAnnealing".to_string()
    }
}
