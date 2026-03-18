//! WASM wrapper for the ViTLayerDecay learning rate scheduler.

use optirs_core::schedulers::ViTLayerDecay;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the ViTLayerDecay scheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmViTLayerDecay {
    inner: ViTLayerDecay<f64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmViTLayerDecay {
    /// Create a new ViT layer decay scheduler without warmup.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate (peak LR)
    /// * `decay_rate` - Per-layer decay rate (typically 0.65-0.85)
    /// * `num_layers` - Number of transformer layers
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(base_lr: f64, decay_rate: f64, num_layers: usize) -> Self {
        Self {
            inner: ViTLayerDecay::new(base_lr, decay_rate, num_layers, 0, 1000),
        }
    }

    /// Create a new ViT layer decay scheduler with warmup and cosine decay.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate (peak LR after warmup)
    /// * `decay_rate` - Per-layer decay rate (typically 0.65-0.85)
    /// * `num_layers` - Number of transformer layers
    /// * `warmup_steps` - Number of linear warmup steps
    /// * `total_steps` - Total number of training steps
    pub fn new_with_warmup(
        base_lr: f64,
        decay_rate: f64,
        num_layers: usize,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        Self {
            inner: ViTLayerDecay::new(base_lr, decay_rate, num_layers, warmup_steps, total_steps),
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

    /// Get the learning rate for a specific layer.
    ///
    /// Layer 0 (earliest/deepest) gets the lowest LR, and the last layer gets
    /// the full base learning rate.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0-based)
    pub fn get_layer_learning_rate(&self, layer_idx: usize) -> f64 {
        self.inner.get_layer_learning_rate(layer_idx)
    }

    /// Get learning rates for all layers as a vector.
    ///
    /// Returns learning rates ordered by layer index (layer 0 first).
    pub fn get_all_layer_rates(&self) -> Vec<f64> {
        self.inner.get_all_layer_rates()
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "ViTLayerDecay".to_string()
    }
}
