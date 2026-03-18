//! WASM wrapper for the OneCycle learning rate scheduler.

use optirs_core::schedulers::OneCycle;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the OneCycle scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmOneCycle {
    inner: OneCycle<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmOneCycle {
    /// Create a new one-cycle scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum learning rate reached after warm-up
    /// * `total_steps` - Total number of training steps
    /// * `pct_start` - Fraction of total steps used for warm-up (typically 0.2-0.3)
    /// * `div_factor` - Determines the initial learning rate via initial_lr = max_lr / div_factor
    /// * `final_div_factor` - Determines the final learning rate via final_lr = initial_lr / final_div_factor
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        let initial_lr = max_lr / div_factor;
        let final_lr = initial_lr / final_div_factor;
        Self {
            inner: OneCycle::new(initial_lr, max_lr, total_steps, pct_start)
                .with_final_lr(final_lr),
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
        "OneCycle".to_string()
    }
}
