//! WASM wrapper for the NoiseInjectionScheduler learning rate scheduler.

use optirs_core::schedulers::{ConstantScheduler, NoiseDistribution, NoiseInjectionScheduler};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the NoiseInjectionScheduler.
///
/// This wraps a `NoiseInjectionScheduler<f64, ConstantScheduler<f64>>` since
/// generic schedulers cannot be directly exposed to WASM. The base learning rate
/// is provided as a constant, and noise is added on top of it.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmNoiseInjectionScheduler {
    inner: NoiseInjectionScheduler<f64, ConstantScheduler<f64>>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmNoiseInjectionScheduler {
    /// Create a noise injection scheduler with uniform noise.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base constant learning rate
    /// * `min_noise` - Minimum noise value
    /// * `max_noise` - Maximum noise value
    /// * `min_lr` - Minimum allowed learning rate (for stability)
    pub fn new_uniform(base_lr: f64, min_noise: f64, max_noise: f64, min_lr: f64) -> Self {
        let base = ConstantScheduler::new(base_lr);
        Self {
            inner: NoiseInjectionScheduler::new(
                base,
                NoiseDistribution::Uniform {
                    min: min_noise,
                    max: max_noise,
                },
                min_lr,
            ),
        }
    }

    /// Create a noise injection scheduler with Gaussian noise.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base constant learning rate
    /// * `mean` - Mean of the Gaussian noise
    /// * `std_dev` - Standard deviation of the Gaussian noise
    /// * `min_lr` - Minimum allowed learning rate (for stability)
    pub fn new_gaussian(base_lr: f64, mean: f64, std_dev: f64, min_lr: f64) -> Self {
        let base = ConstantScheduler::new(base_lr);
        Self {
            inner: NoiseInjectionScheduler::new(
                base,
                NoiseDistribution::Gaussian { mean, std_dev },
                min_lr,
            ),
        }
    }

    /// Create a noise injection scheduler with cyclical (sinusoidal) noise.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base constant learning rate
    /// * `amplitude` - Maximum amplitude of the sinusoidal oscillation
    /// * `period` - Number of steps to complete one full cycle
    /// * `min_lr` - Minimum allowed learning rate (for stability)
    pub fn new_cyclical(base_lr: f64, amplitude: f64, period: usize, min_lr: f64) -> Self {
        let base = ConstantScheduler::new(base_lr);
        Self {
            inner: NoiseInjectionScheduler::new(
                base,
                NoiseDistribution::Cyclical { amplitude, period },
                min_lr,
            ),
        }
    }

    /// Create a noise injection scheduler with decaying noise.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base constant learning rate
    /// * `initial_scale` - Initial scale factor for noise at step 0
    /// * `final_scale` - Final scale factor for noise after decay_steps
    /// * `decay_steps` - Number of steps over which to decay from initial to final scale
    /// * `min_lr` - Minimum allowed learning rate (for stability)
    pub fn new_decaying(
        base_lr: f64,
        initial_scale: f64,
        final_scale: f64,
        decay_steps: usize,
        min_lr: f64,
    ) -> Self {
        let base = ConstantScheduler::new(base_lr);
        Self {
            inner: NoiseInjectionScheduler::new(
                base,
                NoiseDistribution::Decaying {
                    initial_scale,
                    final_scale,
                    decay_steps,
                },
                min_lr,
            ),
        }
    }

    /// Advance the scheduler by one step and return the new (noisy) learning rate.
    pub fn step(&mut self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.step()
    }

    /// Get the current learning rate (with noise applied).
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
        "NoiseInjectionScheduler".to_string()
    }
}
