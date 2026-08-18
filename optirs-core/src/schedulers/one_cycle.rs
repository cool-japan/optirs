// One-cycle learning rate policy
//
// This module implements the one-cycle learning rate policy as described by Leslie N. Smith
// in "A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate,
// batch size, momentum, and weight decay"

use crate::error::{OptimError, Result};
use crate::schedulers::LearningRateScheduler;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::{self, Debug};

/// Convert an `f64` constant into the scheduler's float type.
fn from_f64<A: Float>(v: f64) -> A {
    A::from(v).unwrap_or_else(A::zero)
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

/// One-cycle learning rate policy
///
/// The one-cycle policy combines triangular learning rate policy with momentum cycling.
/// It consists of two phases:
/// 1. A warm-up phase where learning rate increases and momentum decreases
/// 2. A cool-down phase where learning rate decreases and momentum increases
///
/// The schedule is *saturating*: once `total_steps` have been taken the learning rate
/// stays at its final value instead of continuing past the end of the cycle (which used
/// to produce negative learning rates).
///
/// # Example
///
/// ```
/// use optirs_core::schedulers::{OneCycle, LearningRateScheduler};
///
/// let mut scheduler = OneCycle::new(
///     0.0001,  // initial learning rate
///     0.001,   // max learning rate
///     1000,    // total steps
///     0.25,    // warm-up percentage
/// );
///
/// // The learning rate will increase from 0.0001 to 0.001 in first 250 steps,
/// // then decrease to a value lower than initial in remaining 750 steps
/// for _ in 0..1000 {
///     let lr = scheduler.get_learning_rate();
///     // Use lr for optimization
///     scheduler.step();
/// }
/// ```
pub struct OneCycle<A: Float> {
    initial_lr: A,
    max_lr: A,
    final_lr: Option<A>,
    /// Total number of steps in the cycle (always >= 1)
    total_steps: usize,
    /// Number of warm-up steps (always < `total_steps`)
    warmup_steps: usize,
    current_step: usize,
    max_momentum: Option<A>,
    min_momentum: Option<A>,
    base_momentum: Option<A>,
    anneal_strategy: AnnealStrategy,
    final_div_factor: A,
}

/// Annealing strategy for the cool-down phase
#[derive(Debug, Clone, Copy)]
pub enum AnnealStrategy {
    /// Linear annealing
    Linear,
    /// Cosine annealing
    Cosine,
}

impl<A: Float + ScalarOperand + std::fmt::Debug + Send + Sync> OneCycle<A> {
    /// Create a new one-cycle scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `max_lr` - Maximum learning rate reached after warm-up
    /// * `total_steps` - Total number of training steps. `0` is invalid and is clamped
    ///   to `1`; use [`OneCycle::try_new`] to reject it instead.
    /// * `warmup_frac` - Fraction of total steps used for warm-up (typically 0.2-0.3).
    ///   Values outside `(0, 1)` (and non-finite values) are clamped so the resulting
    ///   schedule always has at least one cool-down step.
    pub fn new(initial_lr: A, max_lr: A, total_steps: usize, warmup_frac: f64) -> Self {
        let total_steps = total_steps.max(1);
        let frac = if warmup_frac.is_finite() {
            warmup_frac.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // `as usize` saturates at 0 for negative/NaN inputs, which `frac` already excludes.
        let warmup_steps = ((total_steps as f64) * frac) as usize;
        // Always leave at least one cool-down step so `total_steps - warmup_steps > 0`.
        let warmup_steps = warmup_steps.min(total_steps.saturating_sub(1));

        let final_div_factor = from_f64::<A>(10000.0); // Very small final LR

        Self {
            initial_lr,
            max_lr,
            final_lr: None,
            total_steps,
            warmup_steps,
            current_step: 0,
            max_momentum: None,
            min_momentum: None,
            base_momentum: None,
            anneal_strategy: AnnealStrategy::Cosine,
            final_div_factor,
        }
    }

    /// Create a new one-cycle scheduler, validating the configuration
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] when
    /// * `total_steps == 0`,
    /// * `warmup_frac` is not finite or is outside the open interval `(0, 1)`,
    /// * `initial_lr` or `max_lr` is not finite, or
    /// * `initial_lr <= 0` or `max_lr < initial_lr`.
    pub fn try_new(initial_lr: A, max_lr: A, total_steps: usize, warmup_frac: f64) -> Result<Self> {
        if total_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "OneCycle requires total_steps > 0".to_string(),
            ));
        }
        if !warmup_frac.is_finite() || warmup_frac <= 0.0 || warmup_frac >= 1.0 {
            return Err(OptimError::InvalidConfig(format!(
                "OneCycle requires pct_start (warmup_frac) in the open interval (0, 1), got {warmup_frac}"
            )));
        }
        if !initial_lr.is_finite() || !max_lr.is_finite() {
            return Err(OptimError::InvalidConfig(
                "OneCycle requires finite initial_lr and max_lr".to_string(),
            ));
        }
        if initial_lr <= A::zero() {
            return Err(OptimError::InvalidConfig(
                "OneCycle requires initial_lr > 0".to_string(),
            ));
        }
        if max_lr < initial_lr {
            return Err(OptimError::InvalidConfig(
                "OneCycle requires max_lr >= initial_lr".to_string(),
            ));
        }

        Ok(Self::new(initial_lr, max_lr, total_steps, warmup_frac))
    }

    /// Create with specific final learning rate
    pub fn with_final_lr(mut self, final_lr: A) -> Self {
        self.final_lr = Some(final_lr);
        self.final_div_factor = if final_lr == A::zero() {
            from_f64::<A>(10000.0)
        } else {
            self.initial_lr / final_lr
        };
        self
    }

    /// Set momentum cycling parameters
    pub fn with_momentum(mut self, min_momentum: A, max_momentum: A, base_momentum: A) -> Self {
        self.min_momentum = Some(min_momentum);
        self.max_momentum = Some(max_momentum);
        self.base_momentum = Some(base_momentum);
        self
    }

    /// Set annealing strategy for cool-down phase
    pub fn with_anneal_strategy(mut self, strategy: AnnealStrategy) -> Self {
        self.anneal_strategy = strategy;
        self
    }

    /// Total number of steps in the cycle (always >= 1)
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Number of warm-up steps (always < `total_steps`)
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Progress through the warm-up phase, or `None` once warm-up is finished.
    ///
    /// The step counter is clamped to `total_steps` first, so the returned progress is
    /// always in `[0, 1]`.
    fn warmup_progress(&self) -> Option<A> {
        let step = self.current_step.min(self.total_steps);
        if self.warmup_steps == 0 || step >= self.warmup_steps {
            return None;
        }
        Some(from_usize::<A>(step) / denom_from_usize::<A>(self.warmup_steps))
    }

    /// Progress through the cool-down phase, clamped to `[0, 1]`.
    fn cooldown_progress(&self) -> A {
        let step = self.current_step.min(self.total_steps);
        // `warmup_steps < total_steps` is guaranteed by the constructor.
        let remaining_steps = self.total_steps.saturating_sub(self.warmup_steps);
        let cooled = step.saturating_sub(self.warmup_steps).min(remaining_steps);
        from_usize::<A>(cooled) / denom_from_usize::<A>(remaining_steps)
    }

    /// Get current momentum value
    pub fn get_momentum(&self) -> Option<A> {
        match (self.min_momentum, self.max_momentum) {
            (Some(min_mom), Some(max_mom)) => match self.warmup_progress() {
                Some(progress) => {
                    // During warm-up: momentum decreases
                    Some(max_mom - (max_mom - min_mom) * progress)
                }
                None => {
                    // During cool-down: momentum increases
                    let cool_progress = self.cooldown_progress();
                    match self.anneal_strategy {
                        AnnealStrategy::Linear => {
                            Some(min_mom + (max_mom - min_mom) * cool_progress)
                        }
                        AnnealStrategy::Cosine => {
                            let cos_out = ((cool_progress * from_f64::<A>(std::f64::consts::PI))
                                .cos()
                                + A::one())
                                / from_f64::<A>(2.0);
                            Some(min_mom + (max_mom - min_mom) * (A::one() - cos_out))
                        }
                    }
                }
            },
            _ => self.base_momentum,
        }
    }

    /// Get fraction of the cycle that has been completed, clamped to `[0, 1]`
    pub fn get_percentage_complete(&self) -> A {
        let step = self.current_step.min(self.total_steps);
        from_usize::<A>(step) / denom_from_usize::<A>(self.total_steps)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync> LearningRateScheduler<A> for OneCycle<A> {
    fn get_learning_rate(&self) -> A {
        match self.warmup_progress() {
            Some(progress) => {
                // Warm-up phase: increase from initial to max
                self.initial_lr + (self.max_lr - self.initial_lr) * progress
            }
            None => {
                // Cool-down phase: decrease from max to final
                let cool_progress = self.cooldown_progress();
                let final_lr = match self.final_lr {
                    Some(lr) => lr,
                    None => self.initial_lr / self.final_div_factor,
                };

                match self.anneal_strategy {
                    AnnealStrategy::Linear => {
                        self.max_lr - (self.max_lr - final_lr) * cool_progress
                    }
                    AnnealStrategy::Cosine => {
                        let cos_out = ((cool_progress * from_f64::<A>(std::f64::consts::PI)).cos()
                            + A::one())
                            / from_f64::<A>(2.0);
                        final_lr + (self.max_lr - final_lr) * cos_out
                    }
                }
            }
        }
    }

    fn step(&mut self) -> A {
        // Saturate at `total_steps`: the one-cycle policy has a defined end.
        self.current_step = self.current_step.saturating_add(1).min(self.total_steps);
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl<A: Float + Debug + Send + Sync> fmt::Debug for OneCycle<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneCycle")
            .field("initial_lr", &self.initial_lr)
            .field("max_lr", &self.max_lr)
            .field("final_lr", &self.final_lr)
            .field("total_steps", &self.total_steps)
            .field("warmup_steps", &self.warmup_steps)
            .field("current_step", &self.current_step)
            .field("anneal_strategy", &self.anneal_strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_one_cycle_basic() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        // Initial learning rate
        assert_relative_eq!(scheduler.get_learning_rate(), 0.0001, epsilon = 1e-6);

        // At end of warm-up (25% = 25 steps)
        for _ in 0..25 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);

        // Final learning rate should be very small
        for _ in 25..100 {
            scheduler.step();
        }
        assert!(scheduler.get_learning_rate() < 0.0001);
    }

    #[test]
    fn test_one_cycle_momentum() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25).with_momentum(0.85, 0.95, 0.9);

        // Initial momentum (max during warm-up)
        assert_relative_eq!(
            scheduler.get_momentum().unwrap_or(f64::NAN),
            0.95,
            epsilon = 1e-6
        );

        // At end of warm-up (min momentum)
        for _ in 0..25 {
            scheduler.step();
        }
        assert_relative_eq!(
            scheduler.get_momentum().unwrap_or(f64::NAN),
            0.85,
            epsilon = 1e-6
        );

        // Final momentum (back to max)
        for _ in 25..100 {
            scheduler.step();
        }
        let final_momentum = scheduler.get_momentum().unwrap_or(f64::NAN);
        assert!(final_momentum > 0.94); // Should be close to max
    }

    #[test]
    fn test_one_cycle_linear_anneal() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25)
            .with_anneal_strategy(AnnealStrategy::Linear)
            .with_final_lr(0.00001);

        // Move past warm-up
        for _ in 0..25 {
            scheduler.step();
        }

        let lr_at_warmup = scheduler.get_learning_rate();
        assert_relative_eq!(lr_at_warmup, 0.001, epsilon = 1e-6);

        // Check linear decrease
        for _ in 0..37 {
            // Halfway through cool-down
            scheduler.step();
        }

        let lr_halfway = scheduler.get_learning_rate();
        assert!(lr_halfway < 0.001);
        assert!(lr_halfway > 0.00001);

        // Should decrease linearly
        let expected = 0.001 - (0.001 - 0.00001) * 0.5;
        assert_relative_eq!(lr_halfway, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_percentage_complete() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        assert_relative_eq!(scheduler.get_percentage_complete(), 0.0, epsilon = 1e-6);

        for _ in 0..50 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_percentage_complete(), 0.5, epsilon = 1e-6);

        for _ in 50..100 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_percentage_complete(), 1.0, epsilon = 1e-6);

        // Stepping past the end keeps the percentage clamped.
        for _ in 0..50 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_percentage_complete(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        // Advance scheduler
        for _ in 0..50 {
            scheduler.step();
        }

        let lr_mid = scheduler.get_learning_rate();
        assert!(lr_mid != 0.0001);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.current_step, 0);
        assert_relative_eq!(scheduler.get_learning_rate(), 0.0001, epsilon = 1e-6);
    }

    #[test]
    fn test_degenerate_configs_are_clamped() {
        // total_steps == 0 must not divide by zero.
        let mut zero = OneCycle::new(0.0001, 0.001, 0, 0.25);
        assert_eq!(zero.total_steps(), 1);
        for _ in 0..5 {
            assert!(zero.step().is_finite());
        }

        // warmup_frac > 1 must not underflow `total_steps - warmup_steps`.
        let mut over = OneCycle::new(0.0001, 0.001, 100, 1.5);
        assert!(over.warmup_steps() < over.total_steps());
        for _ in 0..200 {
            let lr = over.step();
            assert!(lr.is_finite() && lr >= 0.0);
        }

        // Non-finite fractions fall back to "no warm-up".
        let nan = OneCycle::new(0.0001, 0.001, 100, f64::NAN);
        assert_eq!(nan.warmup_steps(), 0);
    }

    #[test]
    fn test_try_new_validates() {
        assert!(OneCycle::try_new(0.0001f64, 0.001, 0, 0.25).is_err());
        assert!(OneCycle::try_new(0.0001f64, 0.001, 100, 0.0).is_err());
        assert!(OneCycle::try_new(0.0001f64, 0.001, 100, 1.0).is_err());
        assert!(OneCycle::try_new(0.0001f64, 0.001, 100, f64::NAN).is_err());
        assert!(OneCycle::try_new(0.0f64, 0.001, 100, 0.25).is_err());
        assert!(OneCycle::try_new(0.001f64, 0.0001, 100, 0.25).is_err());
        assert!(OneCycle::try_new(0.0001f64, 0.001, 100, 0.25).is_ok());
    }
}
