// ViT Layer Decay learning rate scheduler
//
// Implements per-layer exponential learning rate decay for Vision Transformer (ViT) models.
// Each layer receives a different learning rate based on its depth, with deeper layers
// (closer to the input) receiving lower learning rates. Combined with linear warmup
// and cosine decay scheduling.
//
// Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
// (Dosovitskiy et al., 2021) and layer-wise learning rate decay techniques from
// BEiT (Bao et al., 2022).

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Helper to convert f64 to generic Float type without using unwrap
fn from_f64<A: Float>(v: f64) -> A {
    A::from(v).unwrap_or_else(|| A::zero())
}

/// Helper to convert usize to generic Float type without using unwrap
fn from_usize<A: Float>(v: usize) -> A {
    A::from(v).unwrap_or_else(|| A::zero())
}

/// Builder for constructing a `ViTLayerDecay` scheduler with a fluent API.
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::ViTLayerDecay;
///
/// let scheduler = ViTLayerDecay::<f64>::builder()
///     .base_lr(0.001)
///     .decay_rate(0.75)
///     .num_layers(12)
///     .warmup_steps(500)
///     .total_steps(10000)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ViTLayerDecayBuilder<A: Float + Debug> {
    base_lr: Option<A>,
    decay_rate: Option<A>,
    num_layers: Option<usize>,
    warmup_steps: Option<usize>,
    total_steps: Option<usize>,
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> ViTLayerDecayBuilder<A> {
    /// Create a new builder with all fields unset
    fn new() -> Self {
        Self {
            base_lr: None,
            decay_rate: None,
            num_layers: None,
            warmup_steps: None,
            total_steps: None,
        }
    }

    /// Set the base learning rate (peak LR after warmup)
    pub fn base_lr(mut self, lr: A) -> Self {
        self.base_lr = Some(lr);
        self
    }

    /// Set the per-layer decay rate
    ///
    /// Each layer's LR is multiplied by `decay_rate^(num_layers - layer_idx - 1)`,
    /// so deeper layers (lower index) get lower learning rates.
    pub fn decay_rate(mut self, rate: A) -> Self {
        self.decay_rate = Some(rate);
        self
    }

    /// Set the number of transformer layers
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = Some(n);
        self
    }

    /// Set the number of warmup steps for linear warmup
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = Some(steps);
        self
    }

    /// Set the total number of training steps (warmup + cosine decay)
    pub fn total_steps(mut self, steps: usize) -> Self {
        self.total_steps = Some(steps);
        self
    }

    /// Build the `ViTLayerDecay` scheduler
    ///
    /// Uses sensible defaults for any unset fields:
    /// - base_lr: 0.001
    /// - decay_rate: 0.75
    /// - num_layers: 12
    /// - warmup_steps: 0
    /// - total_steps: 1000
    pub fn build(self) -> ViTLayerDecay<A> {
        let base_lr = self.base_lr.unwrap_or_else(|| from_f64(0.001));
        let decay_rate = self.decay_rate.unwrap_or_else(|| from_f64(0.75));
        let num_layers = self.num_layers.unwrap_or(12);
        let warmup_steps = self.warmup_steps.unwrap_or(0);
        let total_steps = self.total_steps.unwrap_or(1000);

        ViTLayerDecay::new(base_lr, decay_rate, num_layers, warmup_steps, total_steps)
    }
}

/// ViT Layer Decay learning rate scheduler
///
/// Implements per-layer exponential learning rate decay for Vision Transformer models.
/// The scheduler combines three concepts:
///
/// 1. **Linear warmup**: Learning rate linearly increases from 0 to `base_lr` over
///    `warmup_steps` steps.
/// 2. **Cosine decay**: After warmup, learning rate follows a cosine schedule from
///    `base_lr` down to 0 over the remaining steps.
/// 3. **Layer-wise decay**: Each transformer layer receives a scaled learning rate:
///    `layer_lr = base_lr_at_step * decay_rate^(num_layers - layer_idx - 1)`
///
/// This means the last layer (closest to the output, highest index) gets the full
/// learning rate, while earlier layers get progressively smaller learning rates.
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{ViTLayerDecay, LearningRateScheduler};
///
/// let mut scheduler = ViTLayerDecay::<f64>::builder()
///     .base_lr(0.001)
///     .decay_rate(0.75)
///     .num_layers(12)
///     .warmup_steps(500)
///     .total_steps(10000)
///     .build();
///
/// // Step through warmup
/// for _ in 0..500 {
///     scheduler.step();
/// }
///
/// // After warmup, base LR should be at peak
/// let lr = scheduler.get_learning_rate();
/// assert!((lr - 0.001).abs() < 1e-6);
///
/// // Get per-layer learning rates
/// let layer_rates = scheduler.get_all_layer_rates();
/// assert_eq!(layer_rates.len(), 12);
/// // Last layer gets highest LR, first layer gets lowest
/// assert!(layer_rates[11] > layer_rates[0]);
/// ```
#[derive(Debug, Clone)]
pub struct ViTLayerDecay<A: Float + Debug> {
    /// Base learning rate (peak LR after warmup)
    base_lr: A,
    /// Per-layer decay rate: layer_lr = base_lr * decay_rate^(num_layers - layer_idx - 1)
    decay_rate: A,
    /// Number of transformer layers
    num_layers: usize,
    /// Number of linear warmup steps
    warmup_steps: usize,
    /// Total number of training steps (warmup + cosine decay)
    total_steps: usize,
    /// Current training step
    current_step: usize,
    /// Current base learning rate (after warmup/cosine schedule applied)
    current_lr: A,
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> ViTLayerDecay<A> {
    /// Create a new ViT layer decay scheduler
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate (peak learning rate after warmup)
    /// * `decay_rate` - Per-layer decay rate (typically 0.65-0.85)
    /// * `num_layers` - Number of transformer layers
    /// * `warmup_steps` - Number of linear warmup steps
    /// * `total_steps` - Total number of training steps
    pub fn new(
        base_lr: A,
        decay_rate: A,
        num_layers: usize,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        Self {
            base_lr,
            decay_rate,
            num_layers,
            warmup_steps,
            total_steps,
            current_step: 0,
            current_lr: A::zero(),
        }
    }

    /// Create a builder for fluent construction
    pub fn builder() -> ViTLayerDecayBuilder<A> {
        ViTLayerDecayBuilder::new()
    }

    /// Compute the schedule factor at the current step
    ///
    /// During warmup: linear ramp from 0 to 1
    /// After warmup: cosine decay from 1 to 0
    fn compute_schedule_factor(&self) -> A {
        if self.current_step == 0 {
            return A::zero();
        }

        if self.current_step <= self.warmup_steps {
            // Linear warmup: factor = current_step / warmup_steps
            if self.warmup_steps == 0 {
                return A::one();
            }
            from_usize::<A>(self.current_step) / from_usize::<A>(self.warmup_steps)
        } else {
            // Cosine decay after warmup
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
            if decay_steps == 0 {
                return A::one();
            }

            let steps_since_warmup = self.current_step.saturating_sub(self.warmup_steps);
            // Clamp to avoid going past total_steps
            let progress = if steps_since_warmup >= decay_steps {
                A::one()
            } else {
                from_usize::<A>(steps_since_warmup) / from_usize::<A>(decay_steps)
            };

            let pi = from_f64::<A>(std::f64::consts::PI);
            let half = from_f64::<A>(0.5);

            // cosine decay: 0.5 * (1 + cos(pi * progress))
            half * (A::one() + (pi * progress).cos())
        }
    }

    /// Get the learning rate for a specific layer
    ///
    /// The per-layer learning rate is computed as:
    /// `layer_lr = current_base_lr * decay_rate^(num_layers - layer_idx - 1)`
    ///
    /// Layer 0 (earliest/deepest) gets the lowest LR, and the last layer gets
    /// the full base learning rate.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0-based, 0 = earliest layer)
    ///
    /// # Returns
    ///
    /// The learning rate for the specified layer. Returns zero if layer_idx >= num_layers.
    pub fn get_layer_learning_rate(&self, layer_idx: usize) -> A {
        if layer_idx >= self.num_layers {
            return A::zero();
        }

        let exponent = self.num_layers - layer_idx - 1;
        let decay_factor = self.decay_rate.powi(exponent as i32);
        self.current_lr * decay_factor
    }

    /// Get learning rates for all layers
    ///
    /// Returns a vector of learning rates ordered by layer index.
    /// The first element corresponds to layer 0 (deepest/earliest layer)
    /// and the last element corresponds to the final layer.
    pub fn get_all_layer_rates(&self) -> Vec<A> {
        (0..self.num_layers)
            .map(|i| self.get_layer_learning_rate(i))
            .collect()
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the decay rate
    pub fn decay_rate(&self) -> A {
        self.decay_rate
    }

    /// Get the base learning rate
    pub fn base_lr(&self) -> A {
        self.base_lr
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A> for ViTLayerDecay<A> {
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.current_step += 1;
        let factor = self.compute_schedule_factor();
        self.current_lr = self.base_lr * factor;
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = A::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_vit_layer_decay_basic() {
        let mut scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 100, 1000);

        // Initial LR should be zero (before any steps)
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.0);

        // Take a step
        let lr = scheduler.step();
        assert!(lr > 0.0, "LR should be positive after first step");
        assert!(lr < 0.001, "LR should be less than base_lr during warmup");

        // Run through warmup
        for _ in 1..100 {
            scheduler.step();
        }

        // After warmup, should be at base_lr
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-9);

        // Per-layer rates should differ
        let layer_0_lr = scheduler.get_layer_learning_rate(0);
        let layer_11_lr = scheduler.get_layer_learning_rate(11);
        assert!(
            layer_11_lr > layer_0_lr,
            "Last layer should have higher LR than first"
        );

        // Last layer should get full base LR (decay_rate^0 = 1)
        assert_abs_diff_eq!(layer_11_lr, 0.001, epsilon = 1e-9);
    }

    #[test]
    fn test_warmup_phase() {
        let mut scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 100, 1000);

        let mut prev_lr = 0.0;
        for i in 0..100 {
            let lr = scheduler.step();

            // LR should be monotonically increasing during warmup
            assert!(
                lr > prev_lr,
                "LR should increase during warmup at step {}: {} vs {}",
                i + 1,
                lr,
                prev_lr
            );

            // LR should be proportional to step / warmup_steps
            let expected = 0.001 * (i + 1) as f64 / 100.0;
            assert_abs_diff_eq!(lr, expected, epsilon = 1e-12);

            prev_lr = lr;
        }

        // At end of warmup, should be at base_lr
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-9);
    }

    #[test]
    fn test_cosine_decay_phase() {
        let warmup_steps = 100;
        let total_steps = 1000;
        let mut scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, warmup_steps, total_steps);

        // Complete warmup
        for _ in 0..warmup_steps {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-9);

        // Cosine decay phase
        let mut prev_lr = scheduler.get_learning_rate();
        for _ in 0..100 {
            let lr = scheduler.step();
            // LR should generally decrease during cosine decay (at least initially)
            assert!(
                lr <= prev_lr + 1e-12,
                "LR should decrease during cosine decay"
            );
            prev_lr = lr;
        }

        // At end of all steps, LR should approach zero
        for _ in 0..(total_steps - warmup_steps - 100) {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_layer_rates_ordering() {
        let mut scheduler = ViTLayerDecay::<f64>::new(0.001, 0.75, 12, 100, 1000);

        // Complete warmup
        for _ in 0..100 {
            scheduler.step();
        }

        let rates = scheduler.get_all_layer_rates();
        assert_eq!(rates.len(), 12);

        // Deeper layers (lower index) should have lower LR
        for i in 1..rates.len() {
            assert!(
                rates[i] > rates[i - 1],
                "Layer {} (lr={}) should have higher LR than layer {} (lr={})",
                i,
                rates[i],
                i - 1,
                rates[i - 1]
            );
        }

        // Check specific values: layer_lr = base_lr * decay_rate^(num_layers - layer_idx - 1)
        // Layer 11 (last): base_lr * 0.75^0 = 0.001
        assert_abs_diff_eq!(rates[11], 0.001, epsilon = 1e-9);
        // Layer 0 (first): base_lr * 0.75^11
        let expected_layer_0 = 0.001 * 0.75_f64.powi(11);
        assert_abs_diff_eq!(rates[0], expected_layer_0, epsilon = 1e-9);

        // Out-of-range layer should return zero
        assert_abs_diff_eq!(scheduler.get_layer_learning_rate(12), 0.0);
    }

    #[test]
    fn test_builder_pattern() {
        let scheduler = ViTLayerDecay::<f64>::builder()
            .base_lr(0.001)
            .decay_rate(0.75)
            .num_layers(12)
            .warmup_steps(500)
            .total_steps(10000)
            .build();

        assert_abs_diff_eq!(scheduler.base_lr(), 0.001);
        assert_abs_diff_eq!(scheduler.decay_rate(), 0.75);
        assert_eq!(scheduler.num_layers(), 12);
        assert_eq!(scheduler.current_step(), 0);

        // Builder with defaults
        let default_scheduler = ViTLayerDecay::<f64>::builder().build();
        assert_abs_diff_eq!(default_scheduler.base_lr(), 0.001);
        assert_abs_diff_eq!(default_scheduler.decay_rate(), 0.75);
        assert_eq!(default_scheduler.num_layers(), 12);
    }
}
