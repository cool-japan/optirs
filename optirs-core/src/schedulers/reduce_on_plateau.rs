// ReduceOnPlateau learning rate scheduler

use scirs2_core::ndarray_ext::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// ReduceOnPlateau learning rate scheduler
///
/// Reduces the learning rate when a metric has stopped improving.
/// This is typically used with validation metrics such as validation loss or accuracy.
///
/// # Examples
///
/// ```
/// use scirs2_optim::schedulers::{ReduceOnPlateau, LearningRateScheduler};
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
            threshold: A::from(1e-4).unwrap(),
            mode_is_min: true,
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

    /// Update the scheduler with a new metric value
    ///
    /// Returns the new learning rate
    pub fn step_with_metric(&mut self, metric: A) -> A {
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
        } else {
            self.stagnation_count += 1;

            if self.stagnation_count >= self.patience {
                // Reduce learning rate
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                // Reset stagnation count
                self.stagnation_count = 0;
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

    /// Default step method, does not update learning rate without a metric
    fn step(&mut self) -> A {
        self.current_lr
    }

    fn reset(&mut self) {
        // Reset stagnation count and best metric, but keep current lr
        self.stagnation_count = 0;
        self.best_metric = None;
    }
}
