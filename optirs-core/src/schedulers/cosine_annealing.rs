// Cosine annealing learning rate scheduler

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::schedulers::LearningRateScheduler;

/// Convert an `f64` constant into the scheduler's float type.
///
/// The conversion is infallible for every primitive float type; the fallback
/// keeps the code free of `unwrap`/`expect`.
fn from_f64<A: Float>(v: f64) -> A {
    A::from(v).unwrap_or_else(A::zero)
}

/// Convert a `usize` counter into the scheduler's float type.
fn from_usize<A: Float>(v: usize) -> A {
    A::from(v).unwrap_or_else(A::zero)
}

/// Convert a `usize` denominator into the scheduler's float type.
///
/// Falls back to `1` instead of `0` so the value can never introduce a
/// division by zero.
fn denom_from_usize<A: Float>(v: usize) -> A {
    match A::from(v) {
        Some(x) if x != A::zero() => x,
        _ => A::one(),
    }
}

/// Cosine annealing learning rate scheduler
///
/// Implements the cosine annealing learning rate schedule from
/// "SGDR: Stochastic Gradient Descent with Warm Restarts" by Loshchilov & Hutter (2017).
///
/// The learning rate follows a cosine schedule from the initial learning rate down to the
/// minimum learning rate over a cycle of `t_max` steps:
///
/// ```text
/// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t_cur / t_max))
/// ```
///
/// # Warm restarts
///
/// * `warm_restart == false` - the schedule anneals **once**. After `t_max` steps the
///   learning rate reaches `min_lr` and stays there for every subsequent step.
/// * `warm_restart == true` - the schedule restarts at `initial_lr` every `t_max` steps.
///
/// For the full SGDR schedule with a growing cycle length (`T_mult`), use
/// [`crate::schedulers::CosineAnnealingWarmRestarts`] instead.
///
/// # Validation
///
/// [`CosineAnnealing::new`] never fails: `t_max == 0` is clamped to `1` so the schedule
/// can never divide by zero and produce `NaN`. Use [`CosineAnnealing::try_new`] when you
/// would rather reject an invalid configuration up front.
///
/// # Examples
///
/// ```
/// use optirs_core::schedulers::{CosineAnnealing, LearningRateScheduler};
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
///
/// Annealing without restarts settles on `min_lr`:
///
/// ```
/// use optirs_core::schedulers::{CosineAnnealing, LearningRateScheduler};
///
/// let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 10, false);
/// for _ in 0..10 {
///     scheduler.step();
/// }
/// assert!((scheduler.get_learning_rate() - 0.001).abs() < 1e-12);
///
/// // Extra steps keep the learning rate pinned at min_lr
/// scheduler.step();
/// assert!((scheduler.get_learning_rate() - 0.001).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealing<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Minimum learning rate
    min_lr: A,
    /// Maximum number of iterations in a cycle (always >= 1)
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
    /// * `t_max` - Maximum number of iterations in a cycle. A value of `0` is invalid and
    ///   is clamped to `1` so the schedule stays finite; use [`CosineAnnealing::try_new`]
    ///   to reject it instead.
    /// * `warm_restart` - Whether to restart at `initial_lr` at the end of every cycle
    pub fn new(initial_lr: A, min_lr: A, t_max: usize, warm_restart: bool) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max: t_max.max(1),
            warm_restart,
            step: 0,
            current_lr: initial_lr,
        }
    }

    /// Create a new cosine annealing scheduler, validating the configuration
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] when
    /// * `t_max == 0`,
    /// * `initial_lr` or `min_lr` is not finite,
    /// * `min_lr` is negative, or
    /// * `min_lr > initial_lr`.
    pub fn try_new(initial_lr: A, min_lr: A, t_max: usize, warm_restart: bool) -> Result<Self> {
        if t_max == 0 {
            return Err(OptimError::InvalidConfig(
                "CosineAnnealing requires t_max > 0".to_string(),
            ));
        }
        if !initial_lr.is_finite() || !min_lr.is_finite() {
            return Err(OptimError::InvalidConfig(
                "CosineAnnealing requires finite initial_lr and min_lr".to_string(),
            ));
        }
        if min_lr < A::zero() {
            return Err(OptimError::InvalidConfig(
                "CosineAnnealing requires min_lr >= 0".to_string(),
            ));
        }
        if min_lr > initial_lr {
            return Err(OptimError::InvalidConfig(
                "CosineAnnealing requires min_lr <= initial_lr".to_string(),
            ));
        }

        Ok(Self::new(initial_lr, min_lr, t_max, warm_restart))
    }

    /// Cycle length in steps (always >= 1)
    pub fn t_max(&self) -> usize {
        self.t_max
    }

    /// Whether warm restarts are enabled
    pub fn warm_restart(&self) -> bool {
        self.warm_restart
    }

    /// Evaluate the cosine schedule at position `t_cur` inside the cycle
    fn cosine_lr(&self, t_cur: usize) -> A {
        let pi = from_f64::<A>(std::f64::consts::PI);
        let half = from_f64::<A>(0.5);
        let progress = from_usize::<A>(t_cur) / denom_from_usize::<A>(self.t_max);
        let cos_term = A::one() + (pi * progress).cos();
        self.min_lr + half * (self.initial_lr - self.min_lr) * cos_term
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

        let t_cur = if self.warm_restart {
            // Completing a cycle restarts the schedule at `initial_lr`.
            if self.step >= self.t_max {
                self.step = 0;
            }
            self.step
        } else {
            // Without restarts the schedule anneals once and then stays at `min_lr`.
            // Saturating the counter also keeps the state bounded for long runs.
            if self.step > self.t_max {
                self.step = self.t_max;
            }
            self.step
        };

        self.current_lr = self.cosine_lr(t_cur);
        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.initial_lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_restart_anneals_once_and_pins_at_min() {
        let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 10, false);
        let mut previous = scheduler.get_learning_rate();
        assert!((previous - 0.1).abs() < 1e-12);

        for _ in 0..10 {
            let lr = scheduler.step();
            assert!(lr <= previous + 1e-12);
            previous = lr;
        }
        assert!((scheduler.get_learning_rate() - 0.001).abs() < 1e-12);

        for _ in 0..25 {
            assert!((scheduler.step() - 0.001).abs() < 1e-12);
        }
    }

    #[test]
    fn test_warm_restart_restarts() {
        let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 10, true);
        let mut lrs = Vec::new();
        for _ in 0..20 {
            lrs.push(scheduler.step());
        }
        // Step 10 restarts the cycle.
        assert!((lrs[9] - 0.1).abs() < 1e-12);
        assert!(lrs[9] > lrs[8]);
        // And it restarts again at step 20.
        assert!((lrs[19] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_zero_t_max_is_clamped_not_nan() {
        let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 0, false);
        assert_eq!(scheduler.t_max(), 1);
        for _ in 0..5 {
            assert!(scheduler.step().is_finite());
        }
    }

    #[test]
    fn test_try_new_rejects_zero_t_max() {
        assert!(CosineAnnealing::try_new(0.1f64, 0.001, 0, false).is_err());
        assert!(CosineAnnealing::try_new(0.1f64, 0.2, 10, false).is_err());
        assert!(CosineAnnealing::try_new(0.1f64, -1.0, 10, false).is_err());
        assert!(CosineAnnealing::try_new(f64::NAN, 0.0, 10, false).is_err());
        assert!(CosineAnnealing::try_new(0.1f64, 0.001, 10, false).is_ok());
    }

    #[test]
    fn test_reset() {
        let mut scheduler = CosineAnnealing::new(0.1f64, 0.001, 10, false);
        for _ in 0..20 {
            scheduler.step();
        }
        scheduler.reset();
        assert!((scheduler.get_learning_rate() - 0.1).abs() < 1e-12);
        // Re-annealing works after a reset.
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_learning_rate() - 0.001).abs() < 1e-12);
    }
}
