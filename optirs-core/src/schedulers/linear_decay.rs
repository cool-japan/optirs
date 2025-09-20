// Linear decay learning rate scheduler

use num_traits::Float;
use scirs2_core::ndarray_ext::ScalarOperand;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Linear decay learning rate scheduler
///
/// Linearly decays the learning rate from the initial value to the final value
/// over a specified number of steps.
///
/// Formula:
/// lr = initial_lr - (initial_lr - final_lr) * min(1.0, step / total_steps)
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{LinearDecay, LearningRateScheduler};
///
/// // Create a scheduler with initial learning rate 0.1, final learning rate 0.001,
/// // and total steps 100
/// let mut scheduler = LinearDecay::new(0.1f64, 0.001, 100);
///
/// // Initial learning rate
/// let initial_lr = scheduler.get_learning_rate();
///
/// // Run for a few steps (reduced for test)
/// for _ in 0..10 {
///     // Update learning rate
///     let lr = scheduler.step();
///     // Verify learning rate is decaying
///     assert!(lr <= initial_lr);
/// }
///
/// // Verify learning rate has decayed linearly
/// let final_lr = scheduler.get_learning_rate();
/// assert!(final_lr < 0.1);
/// assert!(final_lr > 0.001);
/// ```
#[derive(Debug, Clone)]
pub struct LinearDecay<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Final learning rate
    final_lr: A,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    step: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float + Debug + Send + Sync> LinearDecay<A> {
    /// Create a new linear decay scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `final_lr` - Final learning rate
    /// * `total_steps` - Total number of steps
    pub fn new(initial_lr: A, final_lr: A, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            step: 0,
            current_lr: initial_lr,
        }
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A> for LinearDecay<A> {
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.step += 1;

        // Calculate learning rate with linear decay
        // lr = initial_lr - (initial_lr - final_lr) * min(1.0, step / total_steps)
        let progress = if self.total_steps > 0 {
            (A::from(self.step).unwrap() / A::from(self.total_steps).unwrap()).min(A::one())
        } else {
            A::one()
        };

        self.current_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress;

        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.initial_lr;
    }
}
