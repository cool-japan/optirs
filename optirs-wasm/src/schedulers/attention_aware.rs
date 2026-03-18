//! WASM wrapper for the AttentionAwareScheduler learning rate scheduler.

use optirs_core::schedulers::{AttentionAwareScheduler, TransformerComponentType};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the AttentionAwareScheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmAttentionAwareScheduler {
    inner: AttentionAwareScheduler<f64>,
}

/// Parse a string into a TransformerComponentType.
fn parse_component(component: &str) -> Result<TransformerComponentType, String> {
    match component.to_lowercase().as_str() {
        "attention" => Ok(TransformerComponentType::Attention),
        "feed_forward" | "feedforward" | "ff" => Ok(TransformerComponentType::FeedForward),
        "embedding" | "embed" => Ok(TransformerComponentType::Embedding),
        "layer_norm" | "layernorm" | "ln" => Ok(TransformerComponentType::LayerNorm),
        "output" | "head" => Ok(TransformerComponentType::Output),
        _ => Err(format!(
            "Unknown component type: '{}'. Expected one of: attention, feed_forward, embedding, layer_norm, output",
            component
        )),
    }
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmAttentionAwareScheduler {
    /// Create a new attention-aware scheduler with default component scales.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate
    /// * `warmup_steps` - Number of linear warmup steps
    /// * `total_steps` - Total number of training steps
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            inner: AttentionAwareScheduler::new(base_lr, warmup_steps, total_steps),
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

    /// Get the effective learning rate for a specific Transformer component.
    ///
    /// # Arguments
    ///
    /// * `component` - Component name: "attention", "feed_forward", "embedding", "layer_norm", or "output"
    pub fn get_component_lr(&self, component: &str) -> Result<f64, String> {
        let comp_type = parse_component(component)?;
        Ok(self.inner.get_component_lr(comp_type))
    }

    /// Set the learning rate scale for a specific component type.
    ///
    /// # Arguments
    ///
    /// * `component` - Component name: "attention", "feed_forward", "embedding", "layer_norm", or "output"
    /// * `scale` - The learning rate multiplier (e.g., 0.1 means 10% of base LR)
    pub fn set_component_scale(&mut self, component: &str, scale: f64) -> Result<(), String> {
        let comp_type = parse_component(component)?;
        self.inner.set_component_scale(comp_type, scale);
        Ok(())
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "AttentionAwareScheduler".to_string()
    }
}
