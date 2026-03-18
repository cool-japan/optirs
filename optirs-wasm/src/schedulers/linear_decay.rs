//! WASM wrapper for the LinearDecay learning rate scheduler.

use optirs_core::schedulers::LinearDecay;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the LinearDecay scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmLinearDecay {
    inner: LinearDecay<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmLinearDecay {
    /// Create a new linear decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `final_lr` - Final learning rate
    /// * `total_steps` - Total number of steps over which to decay
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize) -> Self {
        Self {
            inner: LinearDecay::new(initial_lr, final_lr, total_steps),
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
        "LinearDecay".to_string()
    }
}
