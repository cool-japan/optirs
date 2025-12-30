// Cosine annealing learning rate scheduler

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Cosine annealing learning rate scheduler
///
/// Implements the cosine annealing learning rate scheduler as proposed in the paper:
/// "SGDR: Stochastic Gradient Descent with Warm Restarts" by Loshchilov & Hutter (2017)
///
/// The learning rate follows a cosine schedule from the initial learning rate to the minimum
/// learning rate over a cycle of T_max steps. Optionally, the scheduler can also implement
/// warm restarts by resetting the learning rate at the end of each cycle.
///
/// Formula:
/// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / T_max))
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{CosineAnnealing, LearningRateScheduler};
/// use optirs_core::optimizers::SGD;
/// use scirs2_core::ndarray::Array1;
///
/// // Create a scheduler with initial learning rate 0.1, minimum learning rate 0.001,
/// // cycle length 100 steps, and with warm restarts enabled
/// let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 100, true);
///
/// // Train for a few steps (reduced for test)
/// for _ in 0..3 {
///     // Update learning rate
///     let lr = scheduler.step();
///     // Just check that the learning rate is being updated
///     assert!(lr < 0.1f64);
/// }
///
/// // Verify the learning rate has been updated
/// let final_lr = scheduler.get_learning_rate();
/// assert!(final_lr < 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealing<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Minimum learning rate
    min_lr: A,
    /// Maximum number of iterations in a cycle
    t_max: usize,
    /// Whether to use warm restarts
    warm_restart: bool,
    /// Current step
    step: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float + Debug + Send + Sync> CosineAnnealing<A> {
    /// Create a new cosine annealing scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_max` - Maximum number of iterations in a cycle
    /// * `warm_restart` - Whether to use warm restarts
    pub fn new(initial_lr: A, min_lr: A, t_max: usize, warm_restart: bool) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
            warm_restart,
            step: 0,
            current_lr: initial_lr,
        }
    }
}

impl<A: Float + Debug + ScalarOperand + Send + Sync> LearningRateScheduler<A>
    for CosineAnnealing<A>
{
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.step += 1;

        // Reset step if warm restart is enabled and we've completed a cycle
        if self.warm_restart && self.step.is_multiple_of(self.t_max) && self.step > 0 {
            self.step = 0;
        }

        // Calculate the current position in the cycle
        let t_cur = if self.t_max > 0 {
            self.step % self.t_max
        } else {
            0
        };

        // Calculate cosine annealing learning rate
        // lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t_cur / T_max))
        let pi = A::from(std::f64::consts::PI).unwrap();
        let cos_term =
            A::one() + (pi * A::from(t_cur).unwrap() / A::from(self.t_max).unwrap()).cos();
        self.current_lr =
            self.min_lr + A::from(0.5).unwrap() * (self.initial_lr - self.min_lr) * cos_term;

        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.initial_lr;
    }
}
