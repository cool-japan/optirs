// Exponential decay learning rate scheduler

use num_traits::Float;
use scirs2_core::ndarray_ext::ScalarOperand;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Exponential decay learning rate scheduler
///
/// Applies exponential decay to the learning rate over time:
/// lr = initial_lr * decay_rate^(step / decay_steps)
///
/// # Examples
///
/// ```
/// use scirs2_optim::schedulers::{ExponentialDecay, LearningRateScheduler};
///
/// // Create a scheduler with initial learning rate 0.1, decay rate 0.95
/// // and decay steps 1000
/// let mut scheduler = ExponentialDecay::new(0.1f64, 0.95, 1000);
///
/// // Initial learning rate
/// let initial_lr = scheduler.get_learning_rate();
///
/// // Run for a few steps (reduced for test)
/// for _ in 0..3 {
///     // Update learning rate
///     let lr = scheduler.step();
///     // Verify learning rate is decaying
///     assert!(lr < initial_lr);
/// }
///
/// // Verify scheduler is working
/// let final_lr = scheduler.get_learning_rate();
/// assert!(final_lr < 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialDecay<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Decay rate
    decay_rate: A,
    /// Number of steps after which the learning rate is decayed by decay_rate
    decay_steps: usize,
    /// Current step
    step: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float + Debug + Send + Sync> ExponentialDecay<A> {
    /// Create a new exponential decay scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `decay_rate` - Rate at which learning rate decays (e.g., 0.95)
    /// * `decay_steps` - Number of steps after which learning rate is decayed by decay_rate
    pub fn new(initial_lr: A, decay_rate: A, decay_steps: usize) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_steps,
            step: 0,
            current_lr: initial_lr,
        }
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A>
    for ExponentialDecay<A>
{
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.step += 1;

        // Calculate learning rate decay
        // lr = initial_lr * decay_rate^(step / decay_steps)
        let power = A::from(self.step).unwrap() / A::from(self.decay_steps).unwrap();
        self.current_lr = self.initial_lr * self.decay_rate.powf(power);

        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.initial_lr;
    }
}
