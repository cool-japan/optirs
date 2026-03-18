//! WASM wrapper for the ExponentialDecay learning rate scheduler.

use optirs_core::schedulers::ExponentialDecay;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the ExponentialDecay scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmExponentialDecay {
    inner: ExponentialDecay<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmExponentialDecay {
    /// Create a new exponential decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `decay_rate` - Rate at which learning rate decays (e.g., 0.95)
    /// * `decay_steps` - Number of steps after which learning rate is decayed by decay_rate
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, decay_rate: f64, decay_steps: usize) -> Self {
        Self {
            inner: ExponentialDecay::new(initial_lr, decay_rate, decay_steps),
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
        "ExponentialDecay".to_string()
    }
}
