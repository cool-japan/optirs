// ReduceOnPlateau learning rate scheduler

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// ReduceOnPlateau learning rate scheduler
///
/// Reduces the learning rate when a metric has stopped improving.
/// This is typically used with validation metrics such as validation loss or accuracy.
///
/// # Driving the scheduler
///
/// This scheduler needs a metric, so [`LearningRateScheduler::step`] alone cannot make
/// progress - it deliberately leaves the state untouched and returns the current learning
/// rate. Drive it through [`LearningRateScheduler::step_with_metric`] (available on
/// `Box<dyn LearningRateScheduler<A>>` as well) or through the inherent
/// `ReduceOnPlateau::step_with_metric`.
///
/// # Cooldown
///
/// After a reduction the scheduler enters a cooldown of `cooldown` epochs during which the
/// plateau counter does not accumulate, matching PyTorch's `ReduceLROnPlateau`. The default
/// cooldown is `0`; use [`ReduceOnPlateau::with_cooldown`] to configure it.
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{ReduceOnPlateau, LearningRateScheduler};
///
/// // Create a scheduler with initial learning rate 0.1, factor 0.1,
/// // patience 2, and minimum learning rate 1e-6
/// let mut scheduler = ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6);
///
/// // Initial learning rate
/// let initial_lr = scheduler.get_learning_rate();
///
/// // Simulate training with decreasing loss, then plateauing loss
/// let mut val_loss = 1.0;
/// for epoch in 0..6 {
///     // Simulate decreasing loss for first three epochs, then plateau
///     if epoch < 3 {
///         val_loss -= 0.1;
///     }
///
///     // Update learning rate by registering validation loss
///     scheduler.step_with_metric(val_loss);
/// }
///
/// // After 6 epochs with patience=2, learning rate should have decreased
/// let final_lr = scheduler.get_learning_rate();
/// assert!(final_lr < initial_lr);
/// ```
#[derive(Debug, Clone)]
pub struct ReduceOnPlateau<A: Float + Debug> {
    /// Current learning rate
    current_lr: A,
    /// Factor by which the learning rate will be reduced
    factor: A,
    /// Number of epochs with no improvement after which learning rate will be reduced
    patience: usize,
    /// Minimum learning rate
    min_lr: A,
    /// Counter for steps with no improvement
    stagnation_count: usize,
    /// Best metric value seen so far
    best_metric: Option<A>,
    /// Threshold for measuring improvement
    threshold: A,
    /// Mode: 'min' (lower is better) or 'max' (higher is better)
    mode_is_min: bool,
    /// Number of epochs to wait after a reduction before resuming plateau counting
    cooldown: usize,
    /// Remaining cooldown epochs
    cooldown_counter: usize,
}

impl<A: Float + Debug + Send + Sync> ReduceOnPlateau<A> {
    /// Create a new ReduceOnPlateau scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor by which the learning rate will be reduced (e.g., 0.1 means 10x reduction)
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `min_lr` - Minimum learning rate
    pub fn new(initial_lr: A, factor: A, patience: usize, min_lr: A) -> Self {
        Self {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            stagnation_count: 0,
            best_metric: None,
            threshold: A::from(1e-4).unwrap_or_else(A::zero),
            mode_is_min: true,
            cooldown: 0,
            cooldown_counter: 0,
        }
    }

    /// Set the mode to 'min' (lower metric is better)
    pub fn mode_min(&mut self) -> &mut Self {
        self.mode_is_min = true;
        self
    }

    /// Set the mode to 'max' (higher metric is better)
    pub fn mode_max(&mut self) -> &mut Self {
        self.mode_is_min = false;
        self
    }

    /// Set the threshold for considering an improvement
    pub fn set_threshold(&mut self, threshold: A) -> &mut Self {
        self.threshold = threshold;
        self
    }

    /// Builder: number of epochs to wait after a reduction before the plateau counter
    /// starts accumulating again (PyTorch's `cooldown`)
    pub fn with_cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Set the cooldown in place
    pub fn set_cooldown(&mut self, cooldown: usize) -> &mut Self {
        self.cooldown = cooldown;
        self
    }

    /// Configured cooldown length in epochs
    pub fn cooldown(&self) -> usize {
        self.cooldown
    }

    /// Remaining cooldown epochs (0 when not cooling down)
    pub fn cooldown_counter(&self) -> usize {
        self.cooldown_counter
    }

    /// Number of consecutive non-improving metrics observed so far
    pub fn stagnation_count(&self) -> usize {
        self.stagnation_count
    }

    /// Best metric value observed so far
    pub fn best_metric(&self) -> Option<A> {
        self.best_metric
    }

    /// Update the scheduler with a new metric value
    ///
    /// Returns the new learning rate.
    pub fn step_with_metric(&mut self, metric: A) -> A {
        self.update_with_metric(metric)
    }

    /// Shared plateau logic used by both the inherent method and the trait override.
    fn update_with_metric(&mut self, metric: A) -> A {
        let is_improvement = match self.best_metric {
            None => true, // First metric value is always an improvement
            Some(best) => {
                if self.mode_is_min {
                    // Mode is 'min', improvement means metric < best * (1 - threshold)
                    metric < best * (A::one() - self.threshold)
                } else {
                    // Mode is 'max', improvement means metric > best * (1 + threshold)
                    metric > best * (A::one() + self.threshold)
                }
            }
        };

        if is_improvement {
            self.best_metric = Some(metric);
            self.stagnation_count = 0;
        }

        if self.cooldown_counter > 0 {
            // Still cooling down after a reduction: the plateau counter must not
            // accumulate (PyTorch ReduceLROnPlateau semantics).
            self.cooldown_counter -= 1;
            self.stagnation_count = 0;
            return self.current_lr;
        }

        if !is_improvement {
            self.stagnation_count += 1;

            if self.stagnation_count >= self.patience {
                // Reduce learning rate
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                // Reset stagnation count and enter cooldown
                self.stagnation_count = 0;
                self.cooldown_counter = self.cooldown;
            }
        }

        self.current_lr
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A>
    for ReduceOnPlateau<A>
{
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    /// This scheduler is metric driven: a plain `step()` cannot decide whether the metric
    /// plateaued, so it leaves the state untouched and returns the current learning rate.
    /// Use [`LearningRateScheduler::step_with_metric`] to actually drive the schedule.
    fn step(&mut self) -> A {
        self.current_lr
    }

    fn step_with_metric(&mut self, metric: A) -> A {
        self.update_with_metric(metric)
    }

    fn reset(&mut self) {
        // Reset stagnation count and best metric, but keep current lr
        self.stagnation_count = 0;
        self.best_metric = None;
        self.cooldown_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduces_after_patience() {
        let mut scheduler = ReduceOnPlateau::new(0.1f64, 0.5, 2, 1e-6);
        assert!((scheduler.step_with_metric(1.0) - 0.1).abs() < 1e-12);
        assert!((scheduler.step_with_metric(0.9) - 0.1).abs() < 1e-12);
        assert!((scheduler.step_with_metric(0.85) - 0.1).abs() < 1e-12);
        // Plateau
        assert!((scheduler.step_with_metric(0.84) - 0.1).abs() < 1e-12);
        assert!((scheduler.step_with_metric(0.84) - 0.1).abs() < 1e-12);
        assert!((scheduler.step_with_metric(0.84) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_plain_step_is_inert() {
        let mut scheduler = ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6);
        for _ in 0..50 {
            assert!((scheduler.step() - 0.1).abs() < 1e-12);
        }
        assert_eq!(scheduler.stagnation_count(), 0);
        assert!(scheduler.best_metric().is_none());
    }

    #[test]
    fn test_trait_object_drives_plateau() {
        let mut scheduler: Box<dyn LearningRateScheduler<f64>> =
            Box::new(ReduceOnPlateau::new(0.1f64, 0.1, 2, 1e-6));
        scheduler.step_with_metric(1.0);
        for _ in 0..4 {
            scheduler.step_with_metric(1.0);
        }
        assert!(scheduler.get_learning_rate() < 0.1);
    }

    #[test]
    fn test_cooldown_suppresses_consecutive_reductions() {
        // patience 1 => a single non-improving metric triggers a reduction.
        let mut without = ReduceOnPlateau::new(1.0f64, 0.5, 1, 1e-9);
        without.step_with_metric(1.0);
        let mut lrs_without = Vec::new();
        for _ in 0..4 {
            lrs_without.push(without.step_with_metric(1.0));
        }
        // No cooldown: reduce on every non-improving epoch.
        assert!((lrs_without[0] - 0.5).abs() < 1e-12);
        assert!((lrs_without[1] - 0.25).abs() < 1e-12);
        assert!((lrs_without[2] - 0.125).abs() < 1e-12);
        assert!((lrs_without[3] - 0.0625).abs() < 1e-12);

        let mut with = ReduceOnPlateau::new(1.0f64, 0.5, 1, 1e-9).with_cooldown(2);
        with.step_with_metric(1.0);
        let mut lrs_with = Vec::new();
        for _ in 0..4 {
            lrs_with.push(with.step_with_metric(1.0));
        }
        // With cooldown = 2 the two epochs after a reduction are skipped.
        assert!((lrs_with[0] - 0.5).abs() < 1e-12);
        assert!((lrs_with[1] - 0.5).abs() < 1e-12);
        assert!((lrs_with[2] - 0.5).abs() < 1e-12);
        assert!((lrs_with[3] - 0.25).abs() < 1e-12);
        assert_eq!(with.cooldown(), 2);
    }

    #[test]
    fn test_min_lr_is_respected() {
        let mut scheduler = ReduceOnPlateau::new(1.0f64, 0.1, 1, 0.5);
        scheduler.step_with_metric(1.0);
        for _ in 0..10 {
            scheduler.step_with_metric(1.0);
        }
        assert!((scheduler.get_learning_rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_max_mode() {
        let mut scheduler = ReduceOnPlateau::new(0.1f64, 0.5, 2, 1e-6);
        scheduler.mode_max();
        scheduler.step_with_metric(0.5);
        scheduler.step_with_metric(0.9);
        assert!((scheduler.get_learning_rate() - 0.1).abs() < 1e-12);
        scheduler.step_with_metric(0.9);
        scheduler.step_with_metric(0.9);
        assert!((scheduler.get_learning_rate() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_reset_clears_cooldown_and_history() {
        let mut scheduler = ReduceOnPlateau::new(1.0f64, 0.5, 1, 1e-9).with_cooldown(3);
        scheduler.step_with_metric(1.0);
        scheduler.step_with_metric(1.0);
        assert!(scheduler.cooldown_counter() > 0);
        scheduler.reset();
        assert_eq!(scheduler.cooldown_counter(), 0);
        assert_eq!(scheduler.stagnation_count(), 0);
        assert!(scheduler.best_metric().is_none());
    }
}
