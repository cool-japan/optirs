// Step decay learning rate scheduler

use num_traits::Float;
use scirs2_core::ndarray_ext::ScalarOperand;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Step decay learning rate scheduler
///
/// Reduces the learning rate by a factor at specified step intervals.
/// lr = initial_lr * gamma^(floor(step / step_size))
///
/// # Examples
///
/// ```
/// use scirs2_optim::schedulers::{StepDecay, LearningRateScheduler};
///
/// // Create a scheduler with initial learning rate 0.1, gamma 0.1,
/// // and step size 3 (decay learning rate every 3 steps)
/// let mut scheduler = StepDecay::new(0.1f64, 0.1, 3);
///
/// // Initial learning rate
/// let initial_lr = scheduler.get_learning_rate();
/// assert_eq!(initial_lr, 0.1f64);
///
/// // Train for 4 steps (which should trigger one decay)
/// for _ in 0..4 {
///     // Update learning rate
///     scheduler.step();
/// }
///
/// // After 4 steps with step_size=3, we should have decayed once
/// let final_lr = scheduler.get_learning_rate();
/// assert!(final_lr < initial_lr);
/// assert_eq!(final_lr, initial_lr * 0.1); // Decay by gamma
/// ```
#[derive(Debug, Clone)]
pub struct StepDecay<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Multiplicative factor of learning rate decay
    gamma: A,
    /// Number of steps between learning rate decay
    step_size: usize,
    /// Current step
    step: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float + Debug + Send + Sync> StepDecay<A> {
    /// Create a new step decay scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor of learning rate decay
    /// * `step_size` - Number of steps between learning rate decay
    pub fn new(initial_lr: A, gamma: A, step_size: usize) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
            step: 0,
            current_lr: initial_lr,
        }
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A> for StepDecay<A> {
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.step += 1;

        // Calculate learning rate decay
        // lr = initial_lr * gamma^(floor(step / step_size))
        let exponent = self.step / self.step_size;
        self.current_lr = self.initial_lr * self.gamma.powi(exponent as i32);

        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.initial_lr;
    }
}
