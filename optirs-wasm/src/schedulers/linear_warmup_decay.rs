//! WASM wrapper for the LinearWarmupDecay learning rate scheduler.

use optirs_core::schedulers::{DecayStrategy, LinearWarmupDecay};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the LinearWarmupDecay scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmLinearWarmupDecay {
    inner: LinearWarmupDecay<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmLinearWarmupDecay {
    /// Create a new linear warmup with linear decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Peak learning rate (reached after warmup)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total number of decay steps after warmup
    /// * `min_lr` - Minimum learning rate (starting point for warmup and final decay target)
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, warmup_steps: usize, total_steps: usize, min_lr: f64) -> Self {
        Self {
            inner: LinearWarmupDecay::new(
                initial_lr,
                min_lr,
                warmup_steps,
                total_steps,
                DecayStrategy::Linear { final_lr: min_lr },
            ),
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
        "LinearWarmupDecay".to_string()
    }
}
