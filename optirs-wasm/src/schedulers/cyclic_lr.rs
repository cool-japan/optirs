//! WASM wrapper for the CyclicLR learning rate scheduler.

use optirs_core::schedulers::{CyclicLR, CyclicMode};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the CyclicLR scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmCyclicLR {
    inner: CyclicLR<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmCyclicLR {
    /// Create a new cyclic learning rate scheduler with Triangular mode.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Minimum learning rate
    /// * `max_lr` - Maximum learning rate
    /// * `step_size` - Number of training iterations per half cycle
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        Self {
            inner: CyclicLR::new(base_lr, max_lr, step_size, CyclicMode::Triangular),
        }
    }

    /// Create a new cyclic scheduler with Triangular2 mode (halved amplitude each cycle).
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Minimum learning rate
    /// * `max_lr` - Maximum learning rate
    /// * `step_size` - Number of training iterations per half cycle
    pub fn new_triangular2(base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        Self {
            inner: CyclicLR::new(base_lr, max_lr, step_size, CyclicMode::Triangular2),
        }
    }

    /// Create a new cyclic scheduler with ExpRange mode (exponential scaling).
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Minimum learning rate
    /// * `max_lr` - Maximum learning rate
    /// * `step_size` - Number of training iterations per half cycle
    /// * `gamma` - Exponential scaling factor
    pub fn new_exp_range(base_lr: f64, max_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            inner: CyclicLR::new(base_lr, max_lr, step_size, CyclicMode::ExpRange(gamma)),
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
        "CyclicLR".to_string()
    }
}
