//! WASM wrapper for the CosineAnnealingWarmRestarts learning rate scheduler.

use optirs_core::schedulers::CosineAnnealingWarmRestarts;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the CosineAnnealingWarmRestarts scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmCosineAnnealingWarmRestarts {
    inner: CosineAnnealingWarmRestarts<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmCosineAnnealingWarmRestarts {
    /// Create a new cosine annealing with warm restarts scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_0` - Initial cycle length
    /// * `t_mult` - Multiplicative factor for cycle length after each restart
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, min_lr: f64, t_0: usize, t_mult: f64) -> Self {
        Self {
            inner: CosineAnnealingWarmRestarts::new(initial_lr, min_lr, t_0, t_mult),
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

    /// Get the current cycle index.
    pub fn cycle(&self) -> usize {
        self.inner.cycle()
    }

    /// Get the current cycle length.
    pub fn cycle_length(&self) -> usize {
        self.inner.cycle_length()
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "CosineAnnealingWarmRestarts".to_string()
    }
}
