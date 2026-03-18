//! WASM wrapper for the ConstantScheduler learning rate scheduler.

use optirs_core::schedulers::ConstantScheduler;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the ConstantScheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmConstantScheduler {
    inner: ConstantScheduler<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmConstantScheduler {
    /// Create a new constant scheduler with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `lr` - The constant learning rate to maintain
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(lr: f64) -> Self {
        Self {
            inner: ConstantScheduler::new(lr),
        }
    }

    /// Advance the scheduler by one step and return the learning rate (always constant).
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

    /// Reset the scheduler (no-op for constant scheduler).
    pub fn reset(&mut self) {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.reset();
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "ConstantScheduler".to_string()
    }
}
