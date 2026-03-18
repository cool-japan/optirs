//! WASM wrapper for the ReduceOnPlateau learning rate scheduler.

use optirs_core::schedulers::ReduceOnPlateau;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the ReduceOnPlateau scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmReduceOnPlateau {
    inner: ReduceOnPlateau<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmReduceOnPlateau {
    /// Create a new ReduceOnPlateau scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor by which the learning rate will be reduced (e.g., 0.1 means 10x reduction)
    /// * `patience` - Number of epochs with no improvement after which LR will be reduced
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(initial_lr: f64, factor: f64, patience: usize) -> Self {
        Self {
            inner: ReduceOnPlateau::new(initial_lr, factor, patience, 1e-7),
        }
    }

    /// Advance the scheduler by one step (no-op without a metric).
    pub fn step(&mut self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.step()
    }

    /// Update the scheduler with a metric value and return the new learning rate.
    ///
    /// The scheduler reduces the learning rate when the metric stops improving.
    ///
    /// # Arguments
    ///
    /// * `metric` - The validation metric value (e.g., validation loss)
    pub fn step_with_metric(&mut self, metric: f64) -> f64 {
        self.inner.step_with_metric(metric)
    }

    /// Get the current learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn learning_rate(&self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.get_learning_rate()
    }

    /// Reset the scheduler state (stagnation count and best metric).
    pub fn reset(&mut self) {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.reset();
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "ReduceOnPlateau".to_string()
    }
}
