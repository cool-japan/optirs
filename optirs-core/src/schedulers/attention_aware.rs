// Attention-Aware learning rate scheduler for Transformer models
//
// Provides component-specific learning rate scaling for different parts of a
// Transformer architecture. Different component types (attention, feed-forward,
// embedding, layer norm, output) receive different learning rate multipliers,
// combined with a warmup + cosine decay schedule.
//
// This is inspired by research showing that different Transformer components
// benefit from different learning rates during training.

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
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

/// Types of components in a Transformer architecture
///
/// Each component type can have a different learning rate multiplier
/// applied to the base learning rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformerComponentType {
    /// Self-attention and cross-attention layers (Q, K, V projections and output)
    Attention,
    /// Feed-forward network layers (typically two linear layers with activation)
    FeedForward,
    /// Token and positional embedding layers
    Embedding,
    /// Layer normalization parameters
    LayerNorm,
    /// Final output/classification head
    Output,
}

/// Builder for constructing an `AttentionAwareScheduler` with a fluent API.
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{AttentionAwareScheduler, TransformerComponentType};
///
/// let scheduler = AttentionAwareScheduler::<f64>::builder()
///     .base_lr(0.001)
///     .warmup_steps(1000)
///     .total_steps(50000)
///     .component_scale(TransformerComponentType::Attention, 1.0)
///     .component_scale(TransformerComponentType::Embedding, 0.05)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct AttentionAwareSchedulerBuilder<A: Float + Debug> {
    base_lr: Option<A>,
    warmup_steps: Option<usize>,
    total_steps: Option<usize>,
    component_scales: HashMap<TransformerComponentType, A>,
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> AttentionAwareSchedulerBuilder<A> {
    /// Create a new builder with default component scales
    fn new() -> Self {
        Self {
            base_lr: None,
            warmup_steps: None,
            total_steps: None,
            component_scales: HashMap::new(),
        }
    }

    /// Set the base learning rate
    pub fn base_lr(mut self, lr: A) -> Self {
        self.base_lr = Some(lr);
        self
    }

    /// Set the number of warmup steps
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = Some(steps);
        self
    }

    /// Set the total number of training steps
    pub fn total_steps(mut self, steps: usize) -> Self {
        self.total_steps = Some(steps);
        self
    }

    /// Set the learning rate scale for a specific component type
    ///
    /// The component's effective learning rate will be:
    /// `base_lr * schedule_factor * component_scale`
    pub fn component_scale(mut self, component: TransformerComponentType, scale: A) -> Self {
        self.component_scales.insert(component, scale);
        self
    }

    /// Build the `AttentionAwareScheduler`
    ///
    /// Uses sensible defaults for any unset fields:
    /// - base_lr: 0.001
    /// - warmup_steps: 0
    /// - total_steps: 1000
    /// - Component scales: Attention=1.0, FeedForward=1.0, Embedding=0.1, LayerNorm=0.01, Output=0.5
    pub fn build(self) -> AttentionAwareScheduler<A> {
        let base_lr = self.base_lr.unwrap_or_else(|| from_f64(0.001));
        let warmup_steps = self.warmup_steps.unwrap_or(0);
        let total_steps = self.total_steps.unwrap_or(1000);

        let mut scheduler = AttentionAwareScheduler::new(base_lr, warmup_steps, total_steps);

        // Apply custom scales (overriding defaults)
        for (component, scale) in self.component_scales {
            scheduler.set_component_scale(component, scale);
        }

        scheduler
    }
}

/// Attention-Aware learning rate scheduler for Transformer models
///
/// This scheduler provides component-specific learning rate scaling for different
/// parts of a Transformer architecture. It combines:
///
/// 1. **Linear warmup**: Learning rate linearly increases from 0 to `base_lr`
///    over `warmup_steps` steps.
/// 2. **Cosine decay**: After warmup, learning rate follows a cosine schedule
///    from `base_lr` down to 0 over the remaining steps.
/// 3. **Component-specific scaling**: Each Transformer component type has a
///    multiplier applied to the scheduled learning rate.
///
/// Default component scales:
/// - Attention: 1.0 (full learning rate)
/// - FeedForward: 1.0 (full learning rate)
/// - Embedding: 0.1 (10% of base - embeddings need careful tuning)
/// - LayerNorm: 0.01 (1% of base - normalization params are sensitive)
/// - Output: 0.5 (50% of base)
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{AttentionAwareScheduler, TransformerComponentType, LearningRateScheduler};
///
/// let mut scheduler = AttentionAwareScheduler::<f64>::new(0.001, 100, 1000);
///
/// // Warmup
/// for _ in 0..100 {
///     scheduler.step();
/// }
///
/// // Get component-specific learning rates
/// let attn_lr = scheduler.get_component_lr(TransformerComponentType::Attention);
/// let embed_lr = scheduler.get_component_lr(TransformerComponentType::Embedding);
/// assert!(attn_lr > embed_lr); // Attention gets higher LR than embedding
/// ```
#[derive(Debug, Clone)]
pub struct AttentionAwareScheduler<A: Float + Debug> {
    /// Base learning rate
    base_lr: A,
    /// Number of linear warmup steps
    warmup_steps: usize,
    /// Total number of training steps
    total_steps: usize,
    /// Current training step
    current_step: usize,
    /// Current base learning rate (after warmup/cosine schedule)
    current_lr: A,
    /// Learning rate multipliers per component type
    component_scales: HashMap<TransformerComponentType, A>,
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> AttentionAwareScheduler<A> {
    /// Create a new attention-aware scheduler with default component scales
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate
    /// * `warmup_steps` - Number of linear warmup steps
    /// * `total_steps` - Total number of training steps
    pub fn new(base_lr: A, warmup_steps: usize, total_steps: usize) -> Self {
        let mut component_scales = HashMap::new();
        component_scales.insert(TransformerComponentType::Attention, from_f64(1.0));
        component_scales.insert(TransformerComponentType::FeedForward, from_f64(1.0));
        component_scales.insert(TransformerComponentType::Embedding, from_f64(0.1));
        component_scales.insert(TransformerComponentType::LayerNorm, from_f64(0.01));
        component_scales.insert(TransformerComponentType::Output, from_f64(0.5));

        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            current_lr: A::zero(),
            component_scales,
        }
    }

    /// Create a builder for fluent construction
    pub fn builder() -> AttentionAwareSchedulerBuilder<A> {
        AttentionAwareSchedulerBuilder::new()
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
            if self.warmup_steps == 0 {
                return A::one();
            }
            from_usize::<A>(self.current_step) / from_usize::<A>(self.warmup_steps)
        } else {
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
            if decay_steps == 0 {
                return A::one();
            }

            let steps_since_warmup = self.current_step.saturating_sub(self.warmup_steps);
            let progress = if steps_since_warmup >= decay_steps {
                A::one()
            } else {
                from_usize::<A>(steps_since_warmup) / from_usize::<A>(decay_steps)
            };

            let pi = from_f64::<A>(std::f64::consts::PI);
            let half = from_f64::<A>(0.5);

            half * (A::one() + (pi * progress).cos())
        }
    }

    /// Get the effective learning rate for a specific Transformer component
    ///
    /// The effective LR is: `current_base_lr * component_scale`
    ///
    /// # Arguments
    ///
    /// * `component_type` - The type of Transformer component
    ///
    /// # Returns
    ///
    /// The effective learning rate for the component. If the component type
    /// has no explicit scale set, returns the base learning rate (scale = 1.0).
    pub fn get_component_lr(&self, component_type: TransformerComponentType) -> A {
        let scale = self
            .component_scales
            .get(&component_type)
            .copied()
            .unwrap_or_else(|| A::one());
        self.current_lr * scale
    }

    /// Set the learning rate scale for a specific component type
    ///
    /// # Arguments
    ///
    /// * `component_type` - The type of Transformer component
    /// * `scale` - The learning rate multiplier (e.g., 0.1 means 10% of base LR)
    pub fn set_component_scale(&mut self, component_type: TransformerComponentType, scale: A) {
        self.component_scales.insert(component_type, scale);
    }

    /// Get the current component scales
    pub fn component_scales(&self) -> &HashMap<TransformerComponentType, A> {
        &self.component_scales
    }

    /// Get the base learning rate
    pub fn base_lr(&self) -> A {
        self.base_lr
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A>
    for AttentionAwareScheduler<A>
{
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
    fn test_attention_aware_basic() {
        let mut scheduler = AttentionAwareScheduler::<f64>::new(0.001, 100, 1000);

        // Initial LR should be zero
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.0);

        // Take a step
        let lr = scheduler.step();
        assert!(lr > 0.0, "LR should be positive after first step");
        assert!(lr < 0.001, "LR should be less than base_lr during warmup");

        // Complete warmup
        for _ in 1..100 {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-9);

        // Take some decay steps
        let lr_after_decay = scheduler.step();
        assert!(
            lr_after_decay < 0.001,
            "LR should decrease after warmup ends"
        );

        // Reset should bring back to zero
        scheduler.reset();
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.0);
        assert_eq!(scheduler.current_step(), 0);
    }

    #[test]
    fn test_component_scales() {
        let mut scheduler = AttentionAwareScheduler::<f64>::new(0.001, 100, 1000);

        // Complete warmup
        for _ in 0..100 {
            scheduler.step();
        }

        let base_lr = scheduler.get_learning_rate();
        assert_abs_diff_eq!(base_lr, 0.001, epsilon = 1e-9);

        // Check default component scales
        let attn_lr = scheduler.get_component_lr(TransformerComponentType::Attention);
        assert_abs_diff_eq!(attn_lr, 0.001, epsilon = 1e-9); // scale = 1.0

        let ff_lr = scheduler.get_component_lr(TransformerComponentType::FeedForward);
        assert_abs_diff_eq!(ff_lr, 0.001, epsilon = 1e-9); // scale = 1.0

        let embed_lr = scheduler.get_component_lr(TransformerComponentType::Embedding);
        assert_abs_diff_eq!(embed_lr, 0.0001, epsilon = 1e-9); // scale = 0.1

        let ln_lr = scheduler.get_component_lr(TransformerComponentType::LayerNorm);
        assert_abs_diff_eq!(ln_lr, 0.00001, epsilon = 1e-9); // scale = 0.01

        let output_lr = scheduler.get_component_lr(TransformerComponentType::Output);
        assert_abs_diff_eq!(output_lr, 0.0005, epsilon = 1e-9); // scale = 0.5

        // Verify ordering: Attention/FF > Output > Embedding > LayerNorm
        assert!(attn_lr > output_lr);
        assert!(output_lr > embed_lr);
        assert!(embed_lr > ln_lr);
    }

    #[test]
    fn test_warmup_cosine_schedule() {
        let warmup_steps = 100;
        let total_steps = 1000;
        let mut scheduler = AttentionAwareScheduler::<f64>::new(0.001, warmup_steps, total_steps);

        // Warmup phase: LR should monotonically increase
        let mut prev_lr = 0.0;
        for i in 0..warmup_steps {
            let lr = scheduler.step();
            assert!(
                lr > prev_lr,
                "LR should increase during warmup at step {}: {} vs {}",
                i + 1,
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-9);

        // Cosine decay phase: LR should monotonically decrease
        prev_lr = scheduler.get_learning_rate();
        for _ in 0..100 {
            let lr = scheduler.step();
            assert!(
                lr <= prev_lr + 1e-12,
                "LR should decrease during cosine decay"
            );
            prev_lr = lr;
        }

        // Run to completion
        for _ in 0..(total_steps - warmup_steps - 100) {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_custom_component_scales() {
        let mut scheduler = AttentionAwareScheduler::<f64>::new(0.001, 100, 1000);

        // Set custom scales
        scheduler.set_component_scale(TransformerComponentType::Attention, 2.0);
        scheduler.set_component_scale(TransformerComponentType::Embedding, 0.05);

        // Complete warmup
        for _ in 0..100 {
            scheduler.step();
        }

        let attn_lr = scheduler.get_component_lr(TransformerComponentType::Attention);
        assert_abs_diff_eq!(attn_lr, 0.002, epsilon = 1e-9); // scale = 2.0

        let embed_lr = scheduler.get_component_lr(TransformerComponentType::Embedding);
        assert_abs_diff_eq!(embed_lr, 0.00005, epsilon = 1e-9); // scale = 0.05

        // Unchanged components should keep defaults
        let ff_lr = scheduler.get_component_lr(TransformerComponentType::FeedForward);
        assert_abs_diff_eq!(ff_lr, 0.001, epsilon = 1e-9); // scale = 1.0
    }

    #[test]
    fn test_builder_pattern() {
        let scheduler = AttentionAwareScheduler::<f64>::builder()
            .base_lr(0.0005)
            .warmup_steps(200)
            .total_steps(5000)
            .component_scale(TransformerComponentType::Attention, 1.5)
            .component_scale(TransformerComponentType::Embedding, 0.05)
            .build();

        assert_abs_diff_eq!(scheduler.base_lr(), 0.0005);
        assert_eq!(scheduler.current_step(), 0);

        // Custom scales should be applied
        let scales = scheduler.component_scales();
        assert_abs_diff_eq!(
            *scales
                .get(&TransformerComponentType::Attention)
                .unwrap_or(&0.0),
            1.5,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            *scales
                .get(&TransformerComponentType::Embedding)
                .unwrap_or(&0.0),
            0.05,
            epsilon = 1e-9
        );

        // Default builder
        let default_scheduler = AttentionAwareScheduler::<f64>::builder().build();
        assert_abs_diff_eq!(default_scheduler.base_lr(), 0.001);
    }
}
