//! WASM wrapper for the StepDecay learning rate scheduler.

use optirs_core::schedulers::StepDecay;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the StepDecay scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmStepDecay {
    inner: StepDecay<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmStepDecay {
    /// Create a new step decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `step_size` - Number of steps between learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            inner: StepDecay::new(initial_lr, gamma, step_size),
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
        "StepDecay".to_string()
    }
}
