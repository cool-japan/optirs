// Noise injection scheduler
//
// This module provides a learning rate scheduler that adds noise to the learning rate
// to help escape local minima and improve exploration during training.

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{rngs::StdRng, seeded_rng, thread_rng, CoreRandom};
use std::fmt::Debug;

use super::LearningRateScheduler;

/// Deterministic RNG used by the noise injection scheduler.
type NoiseRng = CoreRandom<StdRng>;

/// Convert an `f64` sample into the scheduler's float type.
fn from_f64<A: Float>(v: f64) -> A {
    <A as NumCast>::from(v).unwrap_or_else(A::zero)
}

/// Convert a `usize` counter into the scheduler's float type.
fn from_usize<A: Float>(v: usize) -> A {
    <A as NumCast>::from(v).unwrap_or_else(A::zero)
}

/// Convert a `usize` denominator into the scheduler's float type.
///
/// Falls back to `1` so the value can never introduce a division by zero.
fn denom_from_usize<A: Float>(v: usize) -> A {
    match <A as NumCast>::from(v) {
        Some(x) if x != A::zero() => x,
        _ => A::one(),
    }
}

/// Draw a standard normal sample using the Box-Muller transform.
///
/// The first uniform is drawn from the half-open range `(0, 1]` (rather than `[0, 1)`)
/// so `ln(u1)` is always finite - `ln(0)` would otherwise yield `-inf` and poison the
/// sample with `NaN`.
fn standard_normal(rng: &mut NoiseRng) -> f64 {
    let u1: f64 = 1.0 - rng.gen_range(0.0f64..1.0f64);
    let u2: f64 = rng.gen_range(0.0f64..1.0f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Noise distribution types for learning rate
#[derive(Debug, Clone, Copy)]
pub enum NoiseDistribution<A: Float> {
    /// Uniform noise in the range [min, max]
    Uniform {
        /// Minimum noise value
        min: A,
        /// Maximum noise value
        max: A,
    },
    /// Gaussian noise with specified mean and standard deviation
    Gaussian {
        /// Mean value of noise distribution
        mean: A,
        /// Standard deviation of noise distribution
        std_dev: A,
    },
    /// Cyclical noise that oscillates according to a sine wave with specified amplitude
    Cyclical {
        /// Maximum amplitude of oscillation
        amplitude: A,
        /// Number of steps to complete one full cycle
        period: usize,
    },
    /// Decaying noise that decreases over time to a minimum value
    Decaying {
        /// Initial scale factor for noise at step 0
        initial_scale: A,
        /// Final scale factor for noise after decay_steps
        final_scale: A,
        /// Number of steps over which to decay from initial to final scale
        decay_steps: usize,
    },
}

/// A learning rate scheduler that injects noise into the base learning rate
///
/// The noise sample is drawn **once per [`LearningRateScheduler::step`]** and cached, so
/// [`LearningRateScheduler::get_learning_rate`] is idempotent between steps: reading the
/// learning rate twice without stepping returns the same value.
///
/// The scheduler owns a seeded, reproducible RNG. Use [`NoiseInjectionScheduler::with_seed`]
/// (or [`NoiseInjectionScheduler::new_seeded`]) to pin the seed; two schedulers built with
/// the same seed produce byte-identical learning rate sequences.
pub struct NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    /// The base scheduler to add noise to
    base_scheduler: S,
    /// The noise distribution
    noise_dist: NoiseDistribution<A>,
    /// Current step number
    step_count: usize,
    /// Seed the RNG was built from (used by `reset`)
    seed: u64,
    /// Deterministic random number generator
    rng: NoiseRng,
    /// Noise sample for the current step (kept so reads are idempotent)
    current_noise: A,
    /// Minimum learning rate to ensure training stability
    min_lr: A,
}

impl<A, S> NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    /// Create a new noise injection scheduler with a randomly chosen seed
    ///
    /// # Arguments
    ///
    /// * `base_scheduler` - The base scheduler to add noise to
    /// * `noise_dist` - The noise distribution to use
    /// * `min_lr` - The minimum learning rate allowed (to ensure stability)
    ///
    /// # Example
    ///
    /// ```
    /// use optirs_core::schedulers::{
    ///     ExponentialDecay, NoiseDistribution, NoiseInjectionScheduler, LearningRateScheduler
    /// };
    ///
    /// // Create a base scheduler
    /// let base_scheduler = ExponentialDecay::new(0.1, 0.9, 10);
    ///
    /// // Create a noise injection scheduler with uniform noise
    /// let mut scheduler = NoiseInjectionScheduler::new(
    ///     base_scheduler,
    ///     NoiseDistribution::Uniform { min: -0.01, max: 0.01 },
    ///     0.001, // Minimum learning rate
    /// );
    ///
    /// // Get the learning rate (will be 0.1 plus some noise)
    /// let lr = scheduler.get_learning_rate();
    /// assert!(lr >= 0.001); // Learning rate should be at least min_lr
    ///
    /// // Reading again without stepping yields exactly the same value
    /// assert_eq!(lr, scheduler.get_learning_rate());
    /// assert!(scheduler.step() >= 0.001);
    /// ```
    pub fn new(base_scheduler: S, noise_dist: NoiseDistribution<A>, min_lr: A) -> Self {
        let seed: u64 = thread_rng().gen_range(0u64..u64::MAX);
        Self::new_seeded(base_scheduler, noise_dist, min_lr, seed)
    }

    /// Create a new noise injection scheduler with an explicit seed
    ///
    /// Two schedulers created with the same seed, the same distribution and the same base
    /// scheduler produce identical learning rate sequences.
    ///
    /// # Example
    ///
    /// ```
    /// use optirs_core::schedulers::{
    ///     ConstantScheduler, NoiseDistribution, NoiseInjectionScheduler, LearningRateScheduler
    /// };
    ///
    /// let dist = NoiseDistribution::Uniform { min: -0.01f64, max: 0.01 };
    /// let mut a = NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 7);
    /// let mut b = NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 7);
    /// for _ in 0..16 {
    ///     assert_eq!(a.step(), b.step());
    /// }
    /// ```
    pub fn new_seeded(
        base_scheduler: S,
        noise_dist: NoiseDistribution<A>,
        min_lr: A,
        seed: u64,
    ) -> Self {
        let mut scheduler = Self {
            base_scheduler,
            noise_dist,
            step_count: 0,
            seed,
            rng: seeded_rng(seed),
            current_noise: A::zero(),
            min_lr,
        };
        scheduler.current_noise = scheduler.sample_noise();
        scheduler
    }

    /// Re-seed the scheduler's RNG
    ///
    /// The cached noise sample is re-derived from the new seed so the very first
    /// [`LearningRateScheduler::get_learning_rate`] is already deterministic.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = seeded_rng(seed);
        self.current_noise = self.sample_noise();
        self
    }

    /// Seed the scheduler's RNG was built from
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Noise sample currently applied to the base learning rate
    pub fn current_noise(&self) -> A {
        self.current_noise
    }

    /// Sample noise from the configured distribution for the current step
    fn sample_noise(&mut self) -> A {
        match self.noise_dist {
            NoiseDistribution::Uniform { min, max } => {
                let min_f64 = min.to_f64().unwrap_or(0.0);
                let max_f64 = max.to_f64().unwrap_or(0.0);
                if !min_f64.is_finite() || !max_f64.is_finite() {
                    return A::zero();
                }
                if min_f64 >= max_f64 {
                    // Degenerate range: `gen_range` would panic on an empty range.
                    return from_f64::<A>(min_f64);
                }
                from_f64::<A>(self.rng.gen_range(min_f64..max_f64))
            }
            NoiseDistribution::Gaussian { mean, std_dev } => {
                let mean_f64 = mean.to_f64().unwrap_or(0.0);
                let std_dev_f64 = std_dev.to_f64().unwrap_or(0.0);
                let z0 = standard_normal(&mut self.rng);
                from_f64::<A>(mean_f64 + std_dev_f64 * z0)
            }
            NoiseDistribution::Cyclical { amplitude, period } => {
                let period_f = denom_from_usize::<A>(period.max(1));
                let step = from_usize::<A>(self.step_count);
                let angle =
                    from_f64::<A>(2.0) * from_f64::<A>(std::f64::consts::PI) * (step / period_f);
                amplitude * angle.sin()
            }
            NoiseDistribution::Decaying {
                initial_scale,
                final_scale,
                decay_steps,
            } => {
                let decay_steps = decay_steps.max(1);
                let decay_steps_a = denom_from_usize::<A>(decay_steps);
                let step = from_usize::<A>(self.step_count.min(decay_steps));
                let scale = initial_scale - (step / decay_steps_a) * (initial_scale - final_scale);

                // Sample from a symmetric uniform distribution and scale by the decaying factor
                scale * from_f64::<A>(self.rng.gen_range(-1.0f64..1.0f64))
            }
        }
    }
}

impl<A, S> LearningRateScheduler<A> for NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    fn get_learning_rate(&self) -> A {
        // The noise sample is fixed for the current step, so repeated reads agree.
        let base_lr = self.base_scheduler.get_learning_rate();
        (base_lr + self.current_noise).max(self.min_lr)
    }

    fn step(&mut self) -> A {
        // Step the base scheduler
        self.base_scheduler.step();

        // Advance and draw exactly one noise sample for the new step
        self.step_count = self.step_count.saturating_add(1);
        self.current_noise = self.sample_noise();

        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.base_scheduler.reset();
        self.step_count = 0;
        self.rng = seeded_rng(self.seed);
        self.current_noise = self.sample_noise();
    }
}

// Only implement Clone for NoiseInjectionScheduler when S is Clone
impl<A, S> Clone for NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            base_scheduler: self.base_scheduler.clone(),
            noise_dist: self.noise_dist,
            step_count: self.step_count,
            seed: self.seed,
            // The underlying RNG is not `Clone`, so the clone restarts the noise stream
            // from the configured seed. The clone is still fully deterministic - it just
            // does not share the original's stream position.
            rng: seeded_rng(self.seed),
            current_noise: self.current_noise,
            min_lr: self.min_lr,
        }
    }
}

impl<A, S> Debug for NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseInjectionScheduler")
            .field("base_scheduler", &self.base_scheduler)
            .field("noise_dist", &self.noise_dist)
            .field("step_count", &self.step_count)
            .field("seed", &self.seed)
            .field("current_noise", &self.current_noise)
            .field("min_lr", &self.min_lr)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedulers::ConstantScheduler;

    #[test]
    fn test_uniform_noise() {
        // Create a constant base scheduler
        let base_scheduler = ConstantScheduler::new(0.1);

        // Create a noise injection scheduler with uniform noise
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Uniform {
                min: -0.02,
                max: 0.02,
            },
            0.001,
        );

        // Get multiple learning rates and check they are within expected range
        let mut rates = Vec::with_capacity(100);
        for _ in 0..100 {
            rates.push(scheduler.step());
        }

        // Check that learning rates are within expected range
        for &rate in &rates {
            assert!((0.08..=0.12).contains(&rate));
        }

        // Check that there is some variation in the learning rates
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let variance = rates.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;

        // Variance should be non-zero if noise is being added
        assert!(variance > 0.0);
    }

    #[test]
    fn test_gaussian_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Gaussian {
                mean: 0.0,
                std_dev: 0.01,
            },
            0.001,
        );

        // Collect samples
        let mut rates = Vec::with_capacity(1000);
        for _ in 0..1000 {
            rates.push(scheduler.step());
        }

        // Every sample must be finite (Box-Muller must never see ln(0)).
        assert!(rates.iter().all(|r| r.is_finite()));

        // Statistical checks (basic, just to ensure it's working)
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;

        // Mean should be close to 0.1 (base learning rate)
        assert!((mean - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_cyclical_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Cyclical {
                amplitude: 0.05,
                period: 10,
            },
            0.001,
        );

        // Step for 20 steps (2 complete cycles)
        let mut rates = Vec::with_capacity(20);
        for _ in 0..20 {
            rates.push(scheduler.step());
        }

        // Check that the pattern repeats
        for i in 0..10 {
            // The rate at i should be similar to the rate at i+10 (one period later)
            // Due to how sinusoidal functions work
            assert!((rates[i] - rates[i + 10]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decaying_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Decaying {
                initial_scale: 0.05,
                final_scale: 0.001,
                decay_steps: 100,
            },
            0.001,
        );

        // Check that noise magnitude decreases over time

        // Calculate variance for early steps
        let mut early_rates = Vec::with_capacity(50);
        for _ in 0..50 {
            early_rates.push(scheduler.step());
        }
        let early_mean = early_rates.iter().sum::<f64>() / early_rates.len() as f64;
        let early_variance = early_rates
            .iter()
            .map(|&r| (r - early_mean).powi(2))
            .sum::<f64>()
            / early_rates.len() as f64;

        // Calculate variance for later steps
        let mut late_rates = Vec::with_capacity(50);
        for _ in 0..50 {
            late_rates.push(scheduler.step());
        }
        let late_mean = late_rates.iter().sum::<f64>() / late_rates.len() as f64;
        let late_variance = late_rates
            .iter()
            .map(|&r| (r - late_mean).powi(2))
            .sum::<f64>()
            / late_rates.len() as f64;

        // The variance should decrease over time
        assert!(early_variance > late_variance);
    }

    #[test]
    fn test_min_lr() {
        let base_scheduler = ConstantScheduler::new(0.01);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Uniform {
                min: -0.1, // This would make the learning rate negative
                max: 0.0,
            },
            0.005, // Minimum learning rate
        );

        // All learning rates should be at least min_lr
        for _ in 0..100 {
            assert!(scheduler.step() >= 0.005);
        }
    }

    #[test]
    fn test_get_learning_rate_is_idempotent() {
        let mut scheduler = NoiseInjectionScheduler::new(
            ConstantScheduler::new(0.1),
            NoiseDistribution::Uniform {
                min: -0.02,
                max: 0.02,
            },
            0.001,
        );

        assert_eq!(scheduler.get_learning_rate(), scheduler.get_learning_rate());
        for _ in 0..32 {
            let stepped = scheduler.step();
            assert_eq!(stepped, scheduler.get_learning_rate());
            assert_eq!(stepped, scheduler.get_learning_rate());
        }
    }

    #[test]
    fn test_same_seed_same_sequence() {
        let dist = NoiseDistribution::Gaussian {
            mean: 0.0f64,
            std_dev: 0.01,
        };
        let mut a =
            NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 12345);
        let mut b =
            NoiseInjectionScheduler::new(ConstantScheduler::new(0.1), dist, 0.001).with_seed(12345);

        assert_eq!(a.get_learning_rate(), b.get_learning_rate());
        let seq_a: Vec<f64> = (0..64).map(|_| a.step()).collect();
        let seq_b: Vec<f64> = (0..64).map(|_| b.step()).collect();
        assert_eq!(seq_a, seq_b);

        // A different seed produces a different stream.
        let mut c =
            NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 999);
        let seq_c: Vec<f64> = (0..64).map(|_| c.step()).collect();
        assert_ne!(seq_a, seq_c);
    }

    #[test]
    fn test_reset_restores_deterministic_stream() {
        let dist = NoiseDistribution::Uniform {
            min: -0.02f64,
            max: 0.02,
        };
        let mut scheduler =
            NoiseInjectionScheduler::new_seeded(ConstantScheduler::new(0.1), dist, 0.001, 77);

        let first: Vec<f64> = (0..16).map(|_| scheduler.step()).collect();
        scheduler.reset();
        let second: Vec<f64> = (0..16).map(|_| scheduler.step()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn test_degenerate_distributions_are_finite() {
        // Zero-width uniform range must not panic.
        let mut uniform = NoiseInjectionScheduler::new_seeded(
            ConstantScheduler::new(0.1f64),
            NoiseDistribution::Uniform { min: 0.0, max: 0.0 },
            0.001,
            1,
        );
        for _ in 0..10 {
            assert!(uniform.step().is_finite());
        }

        // Zero period / zero decay_steps must not divide by zero.
        let mut cyclical = NoiseInjectionScheduler::new_seeded(
            ConstantScheduler::new(0.1f64),
            NoiseDistribution::Cyclical {
                amplitude: 0.01,
                period: 0,
            },
            0.001,
            2,
        );
        for _ in 0..10 {
            assert!(cyclical.step().is_finite());
        }

        let mut decaying = NoiseInjectionScheduler::new_seeded(
            ConstantScheduler::new(0.1f64),
            NoiseDistribution::Decaying {
                initial_scale: 0.05,
                final_scale: 0.001,
                decay_steps: 0,
            },
            0.001,
            3,
        );
        for _ in 0..10 {
            assert!(decaying.step().is_finite());
        }
    }
}
