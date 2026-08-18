// Cyclic learning rate scheduler
//
// This module provides cyclic learning rate scheduling, which cycles the learning rate
// between two boundaries with a constant frequency.

use crate::error::{OptimError, Result};
use crate::schedulers::LearningRateScheduler;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt;

/// Convert a `usize` exponent into `i32`, saturating instead of wrapping.
fn exponent_i32(v: usize) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

/// Convert a `usize` counter into the scheduler's float type.
fn from_usize<A: Float>(v: usize) -> A {
    A::from(v).unwrap_or_else(A::zero)
}

/// Convert a `usize` denominator into the scheduler's float type.
///
/// Falls back to `1` so the value can never introduce a division by zero.
fn denom_from_usize<A: Float>(v: usize) -> A {
    match A::from(v) {
        Some(x) if x != A::zero() => x,
        _ => A::one(),
    }
}

/// Cyclic learning rate policy
#[derive(Debug, Clone, Copy)]
pub enum CyclicMode {
    /// Triangular mode: linear scaling between min and max
    Triangular,
    /// Triangular2 mode: linear scaling with halved amplitude each cycle
    Triangular2,
    /// Exponential range: the amplitude is scaled by `gamma^global_step`
    ExpRange(f64),
}

/// Cyclic learning rate scheduler
///
/// This scheduler cycles the learning rate between two boundaries with a constant frequency.
/// It's based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
///
/// # Modes
///
/// * [`CyclicMode::Triangular`] - constant amplitude.
/// * [`CyclicMode::Triangular2`] - the amplitude is halved after every full cycle.
/// * [`CyclicMode::ExpRange`] - the amplitude is multiplied by `gamma^global_step`. The
///   exponent is the **global** iteration counter, so the decay carries across cycles
///   (it used to be reset at every cycle boundary, which made the mode a no-op).
///
/// # Validation
///
/// The constructors never fail: a `step_size` of `0` is clamped to `1` so the schedule can
/// never divide by zero. Use [`CyclicLR::try_new`] to reject an invalid configuration
/// instead.
///
/// # Example
///
/// ```
/// use optirs_core::schedulers::{CyclicLR, CyclicMode, LearningRateScheduler};
///
/// let mut scheduler = CyclicLR::new(0.001, 0.01, 2000, CyclicMode::Triangular);
///
/// // Learning rate cycles between 0.001 and 0.01 over 2000 steps
/// for step in 0..6000 {
///     let lr = scheduler.get_learning_rate();
///     // Use lr for optimization
///     scheduler.step();
/// }
/// ```
pub struct CyclicLR<A: Float> {
    base_lr: A,
    max_lr: A,
    /// Number of iterations per half cycle (always >= 1)
    step_size: usize,
    mode: CyclicMode,
    gamma: A,
    current_step: usize,
    scale_fn: Box<dyn Fn(usize, usize, A, A) -> A + Send + Sync>,
}

impl<A: Float + std::fmt::Debug + Send + Sync> fmt::Debug for CyclicLR<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CyclicLR")
            .field("base_lr", &self.base_lr)
            .field("max_lr", &self.max_lr)
            .field("step_size", &self.step_size)
            .field("mode", &self.mode)
            .field("gamma", &self.gamma)
            .field("current_step", &self.current_step)
            .field("scale_fn", &"<function>")
            .finish()
    }
}

impl<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync> CyclicLR<A> {
    /// Create a new cyclic learning rate scheduler
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Minimum learning rate
    /// * `max_lr` - Maximum learning rate
    /// * `step_size` - Number of training iterations per half cycle. `0` is invalid and is
    ///   clamped to `1`; use [`CyclicLR::try_new`] to reject it instead.
    /// * `mode` - Cycling mode (Triangular, Triangular2, or ExpRange)
    pub fn new(base_lr: A, max_lr: A, step_size: usize, mode: CyclicMode) -> Self {
        let step_size = step_size.max(1);
        let gamma = match mode {
            CyclicMode::ExpRange(g) => A::from(g).unwrap_or_else(A::one),
            _ => A::one(),
        };

        let scale_fn: Box<dyn Fn(usize, usize, A, A) -> A + Send + Sync> = match mode {
            CyclicMode::Triangular => Box::new(|_, _, _, _| A::one()),
            CyclicMode::Triangular2 => Box::new(|current, cycle_half, _, _| {
                let cycle_len = cycle_half.saturating_mul(2).max(1);
                let two = A::from(2).unwrap_or_else(A::one);
                A::one() / two.powi(exponent_i32(current / cycle_len))
            }),
            // The exponent is the GLOBAL step count so the decay accumulates across
            // cycles instead of restarting at every cycle boundary.
            CyclicMode::ExpRange(_) => {
                Box::new(|current, _cycle_half, gamma, _| gamma.powi(exponent_i32(current)))
            }
        };

        Self {
            base_lr,
            max_lr,
            step_size,
            mode,
            gamma,
            current_step: 0,
            scale_fn,
        }
    }

    /// Create a new cyclic learning rate scheduler, validating the configuration
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] when
    /// * `step_size == 0`,
    /// * `base_lr` or `max_lr` is not finite,
    /// * `base_lr < 0` or `max_lr < base_lr`, or
    /// * the mode is [`CyclicMode::ExpRange`] with a gamma outside `(0, 1]`.
    pub fn try_new(base_lr: A, max_lr: A, step_size: usize, mode: CyclicMode) -> Result<Self> {
        if step_size == 0 {
            return Err(OptimError::InvalidConfig(
                "CyclicLR requires step_size > 0".to_string(),
            ));
        }
        if !base_lr.is_finite() || !max_lr.is_finite() {
            return Err(OptimError::InvalidConfig(
                "CyclicLR requires finite base_lr and max_lr".to_string(),
            ));
        }
        if base_lr < A::zero() {
            return Err(OptimError::InvalidConfig(
                "CyclicLR requires base_lr >= 0".to_string(),
            ));
        }
        if max_lr < base_lr {
            return Err(OptimError::InvalidConfig(
                "CyclicLR requires max_lr >= base_lr".to_string(),
            ));
        }
        if let CyclicMode::ExpRange(g) = mode {
            if !g.is_finite() || g <= 0.0 || g > 1.0 {
                return Err(OptimError::InvalidConfig(format!(
                    "CyclicLR ExpRange requires gamma in (0, 1], got {g}"
                )));
            }
        }

        Ok(Self::new(base_lr, max_lr, step_size, mode))
    }

    /// Create a new triangular cyclic scheduler
    pub fn triangular(base_lr: A, max_lr: A, step_size: usize) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::Triangular)
    }

    /// Create a new triangular2 cyclic scheduler
    pub fn triangular2(base_lr: A, max_lr: A, step_size: usize) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::Triangular2)
    }

    /// Create a new exponential range cyclic scheduler
    pub fn exp_range(base_lr: A, max_lr: A, step_size: usize, gamma: f64) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::ExpRange(gamma))
    }

    /// Set custom scale function
    ///
    /// The closure receives `(global_step, step_size, gamma, 1)` and returns the amplitude
    /// scale factor for the current step.
    pub fn with_scale_fn<F>(mut self, scale_fn: F) -> Self
    where
        F: Fn(usize, usize, A, A) -> A + Send + Sync + 'static,
    {
        self.scale_fn = Box::new(scale_fn);
        self
    }

    /// Number of iterations per half cycle (always >= 1)
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    /// Get the current cycle number
    pub fn get_cycle(&self) -> usize {
        self.current_step / self.cycle_len()
    }

    /// Full cycle length in steps (always >= 2)
    fn cycle_len(&self) -> usize {
        self.step_size.saturating_mul(2).max(1)
    }

    /// Get position within current cycle (0.0 to 1.0)
    pub fn get_cycle_position(&self) -> A {
        let cycle_len = self.cycle_len();
        let cycle_position = self.current_step % cycle_len;
        if cycle_position < self.step_size {
            // First half: increasing
            from_usize::<A>(cycle_position) / denom_from_usize::<A>(self.step_size)
        } else {
            // Second half: decreasing
            from_usize::<A>(cycle_len - cycle_position) / denom_from_usize::<A>(self.step_size)
        }
    }
}

impl<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync> LearningRateScheduler<A>
    for CyclicLR<A>
{
    fn get_learning_rate(&self) -> A {
        let position = self.get_cycle_position();
        let scale = (self.scale_fn)(self.current_step, self.step_size, self.gamma, A::one());

        let amplitude = (self.max_lr - self.base_lr) * scale;
        self.base_lr + amplitude * position
    }

    fn step(&mut self) -> A {
        self.current_step = self.current_step.saturating_add(1);
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_triangular_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;

        let mut scheduler = CyclicLR::triangular(base_lr, max_lr, step_size);

        // At start, should be base_lr
        assert_relative_eq!(scheduler.get_learning_rate(), base_lr, epsilon = 1e-6);

        // At half cycle, should be max_lr
        for _ in 0..step_size {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), max_lr, epsilon = 1e-6);

        // At full cycle, should be back to base_lr
        for _ in 0..step_size {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), base_lr, epsilon = 1e-6);
    }

    #[test]
    fn test_triangular2_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;

        let mut scheduler = CyclicLR::triangular2(base_lr, max_lr, step_size);

        // First cycle
        for _ in 0..step_size {
            scheduler.step();
        }
        let first_max = scheduler.get_learning_rate();
        assert_relative_eq!(first_max, max_lr, epsilon = 1e-6);

        // Move to second cycle max
        for _ in 0..(2 * step_size) {
            scheduler.step();
        }
        let second_max = scheduler.get_learning_rate();

        // Second cycle should have half amplitude
        assert_relative_eq!(
            second_max,
            base_lr + (max_lr - base_lr) / 2.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_exp_range_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;
        let gamma = 0.99995;

        let mut scheduler = CyclicLR::exp_range(base_lr, max_lr, step_size, gamma);

        // Test that learning rate decreases exponentially within each cycle
        let lr_start = scheduler.get_learning_rate();

        for _ in 0..10 {
            scheduler.step();
        }

        let lr_10_steps = scheduler.get_learning_rate();

        // Learning rate should increase in first half of cycle
        assert!(lr_10_steps > lr_start);

        // But the increase should be modulated by gamma
        assert!(lr_10_steps < base_lr + (max_lr - base_lr) * 0.1);
    }

    #[test]
    fn test_exp_range_decay_is_global_not_per_cycle() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 5;
        let gamma = 0.9;

        let mut scheduler = CyclicLR::exp_range(base_lr, max_lr, step_size, gamma);

        for _ in 0..step_size {
            scheduler.step();
        }
        let first_peak = scheduler.get_learning_rate();

        for _ in 0..(2 * step_size) {
            scheduler.step();
        }
        let second_peak = scheduler.get_learning_rate();

        assert!(second_peak < first_peak);
        assert_relative_eq!(
            first_peak,
            base_lr + (max_lr - base_lr) * gamma.powi(step_size as i32),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            second_peak,
            base_lr + (max_lr - base_lr) * gamma.powi(3 * step_size as i32),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_zero_step_size_is_clamped() {
        for mut scheduler in [
            CyclicLR::triangular(0.001f64, 0.01, 0),
            CyclicLR::triangular2(0.001f64, 0.01, 0),
            CyclicLR::exp_range(0.001f64, 0.01, 0, 0.99),
        ] {
            assert_eq!(scheduler.step_size(), 1);
            assert!(scheduler.get_learning_rate().is_finite());
            for _ in 0..10 {
                assert!(scheduler.step().is_finite());
            }
        }
    }

    #[test]
    fn test_try_new_validates() {
        assert!(CyclicLR::try_new(0.001f64, 0.01, 0, CyclicMode::Triangular).is_err());
        assert!(CyclicLR::try_new(0.01f64, 0.001, 10, CyclicMode::Triangular).is_err());
        assert!(CyclicLR::try_new(-0.1f64, 0.01, 10, CyclicMode::Triangular).is_err());
        assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::ExpRange(0.0)).is_err());
        assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::ExpRange(1.5)).is_err());
        assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::ExpRange(0.99)).is_ok());
        assert!(CyclicLR::try_new(0.001f64, 0.01, 10, CyclicMode::Triangular).is_ok());
    }

    #[test]
    fn test_cycle_counting() {
        let mut scheduler = CyclicLR::triangular(0.001, 0.01, 100);

        assert_eq!(scheduler.get_cycle(), 0);

        // Complete one cycle
        for _ in 0..200 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_cycle(), 1);

        // Half way through second cycle
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_cycle(), 1);
    }

    #[test]
    fn test_reset() {
        let mut scheduler = CyclicLR::triangular(0.001, 0.01, 100);

        // Move forward
        for _ in 0..50 {
            scheduler.step();
        }

        let lr_before_reset = scheduler.get_learning_rate();
        assert!(lr_before_reset > 0.001);

        // Reset
        scheduler.reset();
        assert_relative_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);
        assert_eq!(scheduler.current_step, 0);
    }
}
