//! Online drift-triggered hyperparameter adaptation.
//!
//! This module implements a streaming change-point / concept-drift controller
//! that watches the training loss and gradient-norm signals during optimization
//! and adjusts the learning rate (and momentum) online in response to detected
//! drift, divergence, or plateaus.
//!
//! Two classical change-detection statistics are combined inside
//! [`DriftDetector`], one detector per scalar stream:
//!
//! - **EWMA control chart** (Roberts, 1959). The exponentially-weighted moving
//!   average of the signal is tracked together with an EWMA estimate of its
//!   variance, and control limits `μ ± L·σ` flag out-of-limit observations and
//!   their direction.
//! - **Two-sided CUSUM** (Page, 1954). Cumulative sums of the deviation from a
//!   warm-up reference mean `μ0`, with a slack `k` (allowance) and decision
//!   threshold `h`, detect small persistent mean shifts in either direction and
//!   reset after raising an alarm.
//!
//! [`RealtimeAdaptationController`] holds one detector for the loss stream and
//! one for the gradient-norm stream, plus the current learning rate / momentum,
//! a cooldown counter, and a plateau tracker. Its [`observe`] method applies an
//! additive-increase / multiplicative-decrease (AIMD) control policy:
//!
//! - Diverging loss or exploding gradient norm → multiplicatively *decrease* the
//!   learning rate and damp momentum.
//! - A loss plateau (ReduceLROnPlateau style) → *decrease* the learning rate to
//!   refine.
//! - Stable and improving loss with a bounded gradient norm → gently *increase*
//!   the learning rate up to a ceiling.
//!
//! The learning rate is always clamped to `[lr_min, lr_max]` and momentum to
//! `[mom_min, mom_max]`, and a `cooldown_steps` window suppresses a second
//! adjustment immediately after any change.
//!
//! [`observe`]: RealtimeAdaptationController::observe
//!
//! # Examples
//!
//! ```
//! use optirs_learned::realtime_adaptation::{
//!     RealtimeAdaptationConfig, RealtimeAdaptationController,
//! };
//!
//! let config = RealtimeAdaptationConfig::<f64>::default();
//! let mut controller = RealtimeAdaptationController::new(config)
//!     .expect("default config is valid");
//! // Feed (loss, grad_norm) observations from a training loop.
//! let decision = controller.observe(1.0, 0.5);
//! assert!(decision.learning_rate > 0.0);
//! ```

use std::fmt::Debug;

use scirs2_core::numeric::Float;

use crate::domain_optimizers::AdvancedOptimizer;
use crate::error::{OptimError, Result};

/// Drift verdict produced by a [`DriftDetector`] for a single observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSignal {
    /// No drift detected: the stream is in control.
    Stable,
    /// The stream's mean has drifted upward (a positive shift).
    UpwardDrift,
    /// The stream's mean has drifted downward (a negative shift).
    DownwardDrift,
}

/// Configuration for a single-stream [`DriftDetector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftDetectorConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// EWMA smoothing factor `λ ∈ (0, 1]`. Larger values weight recent
    /// observations more heavily.
    pub lambda: T,
    /// Control-limit width `L` (in EWMA standard deviations) for the EWMA chart.
    pub control_limit: T,
    /// CUSUM slack / allowance `k`, expressed in reference standard deviations.
    pub cusum_slack: T,
    /// CUSUM decision threshold `h`, expressed in reference standard deviations.
    pub cusum_threshold: T,
    /// Number of warm-up observations used to estimate the reference mean and
    /// standard deviation before any drift is reported.
    pub warmup_steps: usize,
    /// Floor on the reference / EWMA standard deviation, guarding against a
    /// degenerate zero-variance warm-up window.
    pub min_std: T,
}

impl<T> DriftDetectorConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Validate the configuration, returning an honest error for any field that
    /// would make the detector ill-defined.
    pub fn validate(&self) -> Result<()> {
        if !(self.lambda > T::zero() && self.lambda <= T::one()) {
            return Err(OptimError::InvalidConfig(format!(
                "DriftDetector lambda must lie in (0, 1], got {:?}",
                self.lambda
            )));
        }
        if self.control_limit <= T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "DriftDetector control_limit must be positive, got {:?}",
                self.control_limit
            )));
        }
        if self.cusum_slack <= T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "DriftDetector cusum_slack must be positive, got {:?}",
                self.cusum_slack
            )));
        }
        if self.cusum_threshold <= T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "DriftDetector cusum_threshold must be positive, got {:?}",
                self.cusum_threshold
            )));
        }
        if self.warmup_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "DriftDetector warmup_steps must be at least 1".to_string(),
            ));
        }
        if self.min_std <= T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "DriftDetector min_std must be positive, got {:?}",
                self.min_std
            )));
        }
        Ok(())
    }
}

impl<T> Default for DriftDetectorConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    fn default() -> Self {
        let from = |v: f64, fallback: T| T::from(v).unwrap_or(fallback);
        Self {
            lambda: from(0.2, T::one()),
            control_limit: from(3.0, T::one()),
            cusum_slack: from(0.5, T::one()),
            cusum_threshold: from(5.0, T::one()),
            warmup_steps: 20,
            min_std: from(1e-8, T::epsilon()),
        }
    }
}

/// A single-stream online drift detector combining an EWMA control chart with a
/// two-sided CUSUM statistic.
///
/// Construct one with [`DriftDetector::new`] and feed it scalar observations via
/// [`DriftDetector::update`]. During the first `warmup_steps` observations the
/// detector estimates the reference mean `μ0` and standard deviation `σ0` and
/// reports [`DriftSignal::Stable`]. Thereafter each update returns a
/// [`DriftSignal`] derived from both statistics.
#[derive(Debug, Clone)]
pub struct DriftDetector<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Validated configuration.
    config: DriftDetectorConfig<T>,
    /// Number of observations seen so far.
    count: usize,
    /// Running sum of warm-up observations (for `μ0`).
    warmup_sum: T,
    /// Running sum of squares of warm-up observations (for `σ0`).
    warmup_sum_sq: T,
    /// Reference mean `μ0`, finalised at the end of warm-up.
    reference_mean: T,
    /// Reference standard deviation `σ0`, finalised at the end of warm-up.
    reference_std: T,
    /// Whether warm-up has completed and `μ0` / `σ0` are valid.
    warmed_up: bool,
    /// Current EWMA mean `μ_t`.
    ewma_mean: T,
    /// Current EWMA variance estimate `σ²_t`.
    ewma_var: T,
    /// Positive CUSUM accumulator `S⁺`.
    cusum_pos: T,
    /// Negative CUSUM accumulator `S⁻`.
    cusum_neg: T,
}

impl<T> DriftDetector<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Create a new detector from a configuration.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] if [`DriftDetectorConfig::validate`]
    /// fails.
    pub fn new(config: DriftDetectorConfig<T>) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            count: 0,
            warmup_sum: T::zero(),
            warmup_sum_sq: T::zero(),
            reference_mean: T::zero(),
            reference_std: T::zero(),
            warmed_up: false,
            ewma_mean: T::zero(),
            ewma_var: T::zero(),
            cusum_pos: T::zero(),
            cusum_neg: T::zero(),
        })
    }

    /// Finalise the reference mean and standard deviation from the warm-up
    /// accumulators and seed the EWMA chart at `μ0`.
    fn finalize_warmup(&mut self) {
        let n = T::from(self.count).unwrap_or_else(T::one);
        let mean = self.warmup_sum / n;
        // Population variance over the warm-up window: E[x^2] - E[x]^2, floored
        // at zero before taking the root to avoid spurious NaNs from rounding.
        let mean_sq = self.warmup_sum_sq / n;
        let raw_var = mean_sq - mean * mean;
        let var = if raw_var > T::zero() {
            raw_var
        } else {
            T::zero()
        };
        let std = var.sqrt();
        self.reference_mean = mean;
        self.reference_std = if std > self.config.min_std {
            std
        } else {
            self.config.min_std
        };
        // Seed the EWMA at the reference estimates so the first post-warm-up
        // control limits are meaningful.
        self.ewma_mean = mean;
        self.ewma_var = self.reference_std * self.reference_std;
        self.warmed_up = true;
    }

    /// Feed one observation and obtain the drift verdict.
    pub fn update(&mut self, x: T) -> DriftSignal {
        self.count = self.count.saturating_add(1);

        if !self.warmed_up {
            self.warmup_sum = self.warmup_sum + x;
            self.warmup_sum_sq = self.warmup_sum_sq + x * x;
            if self.count >= self.config.warmup_steps {
                self.finalize_warmup();
            }
            return DriftSignal::Stable;
        }

        let lambda = self.config.lambda;
        let one_minus = T::one() - lambda;

        // EWMA variance uses the *prior* mean for the deviation term (Roberts'
        // recursion), so compute it before updating the mean.
        let prev_mean = self.ewma_mean;
        let deviation = x - prev_mean;
        self.ewma_var = lambda * (deviation * deviation) + one_minus * self.ewma_var;
        self.ewma_mean = lambda * x + one_minus * prev_mean;

        let ewma_std = {
            let s = self.ewma_var.sqrt();
            if s > self.config.min_std {
                s
            } else {
                self.config.min_std
            }
        };
        let limit = self.config.control_limit * ewma_std;
        let upper = self.reference_mean + limit;
        let lower = self.reference_mean - limit;
        let ewma_breach_up = self.ewma_mean > upper;
        let ewma_breach_down = self.ewma_mean < lower;

        // Two-sided CUSUM on the standardised deviation from the reference mean.
        let std = self.reference_std;
        let k = self.config.cusum_slack * std;
        let h = self.config.cusum_threshold * std;
        let centered = x - self.reference_mean;
        // S⁺_t = max(0, S⁺_{t-1} + (x - μ0 - k))
        self.cusum_pos = (self.cusum_pos + (centered - k)).max(T::zero());
        // S⁻_t = max(0, S⁻_{t-1} - (x - μ0 + k))  (accumulates downward shifts)
        self.cusum_neg = (self.cusum_neg - (centered + k)).max(T::zero());

        let cusum_alarm_up = self.cusum_pos > h;
        let cusum_alarm_down = self.cusum_neg > h;

        // Combine the two statistics. A CUSUM alarm is the primary, persistent
        // mean-shift signal; an EWMA breach on the same side reinforces it, and
        // an EWMA breach alone also raises a (more reactive) signal. Resolve the
        // direction by whichever side has the stronger evidence.
        let up = cusum_alarm_up || ewma_breach_up;
        let down = cusum_alarm_down || ewma_breach_down;

        let signal = match (up, down) {
            (true, false) => DriftSignal::UpwardDrift,
            (false, true) => DriftSignal::DownwardDrift,
            (true, true) => {
                // Conflicting one-sided evidence: pick the larger CUSUM
                // excursion, falling back to the EWMA mean's side.
                if self.cusum_pos >= self.cusum_neg {
                    DriftSignal::UpwardDrift
                } else {
                    DriftSignal::DownwardDrift
                }
            }
            (false, false) => DriftSignal::Stable,
        };

        // Reset the alarming accumulator(s) so the detector re-arms after a
        // change point, as prescribed for CUSUM monitoring.
        if cusum_alarm_up {
            self.cusum_pos = T::zero();
        }
        if cusum_alarm_down {
            self.cusum_neg = T::zero();
        }

        signal
    }

    /// Reset the detector to its just-constructed state (warm-up restarts).
    pub fn reset(&mut self) {
        self.count = 0;
        self.warmup_sum = T::zero();
        self.warmup_sum_sq = T::zero();
        self.reference_mean = T::zero();
        self.reference_std = T::zero();
        self.warmed_up = false;
        self.ewma_mean = T::zero();
        self.ewma_var = T::zero();
        self.cusum_pos = T::zero();
        self.cusum_neg = T::zero();
    }

    /// Whether warm-up has completed.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Current EWMA mean `μ_t`.
    pub fn ewma_mean(&self) -> T {
        self.ewma_mean
    }

    /// Current EWMA standard deviation `√σ²_t` (floored at `min_std`).
    pub fn ewma_std(&self) -> T {
        let s = self.ewma_var.sqrt();
        if s > self.config.min_std {
            s
        } else {
            self.config.min_std
        }
    }

    /// Reference mean `μ0` estimated during warm-up.
    pub fn reference_mean(&self) -> T {
        self.reference_mean
    }

    /// Reference standard deviation `σ0` estimated during warm-up.
    pub fn reference_std(&self) -> T {
        self.reference_std
    }

    /// Current positive CUSUM accumulator `S⁺`.
    pub fn cusum_pos(&self) -> T {
        self.cusum_pos
    }

    /// Current negative CUSUM accumulator `S⁻`.
    pub fn cusum_neg(&self) -> T {
        self.cusum_neg
    }

    /// Number of observations seen so far.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Why the controller produced a particular adaptation decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationReason {
    /// No change: still warming up, in cooldown, or no actionable signal.
    NoChange,
    /// Learning rate decreased because the loss is diverging.
    LossDivergence,
    /// Learning rate decreased because the gradient norm is exploding.
    GradientExplosion,
    /// Learning rate decreased because the loss has plateaued.
    Plateau,
    /// Learning rate increased because the loss is stably improving.
    StableImprovement,
}

impl AdaptationReason {
    /// A human-readable description of the reason.
    pub fn as_str(&self) -> &'static str {
        match self {
            AdaptationReason::NoChange => "no_change",
            AdaptationReason::LossDivergence => "loss_divergence",
            AdaptationReason::GradientExplosion => "gradient_explosion",
            AdaptationReason::Plateau => "plateau",
            AdaptationReason::StableImprovement => "stable_improvement",
        }
    }
}

/// The decision returned by [`RealtimeAdaptationController::observe`].
#[derive(Debug, Clone)]
pub struct AdaptationDecision<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Learning rate to use after this observation (already clamped).
    pub learning_rate: T,
    /// Momentum to use after this observation (already clamped).
    pub momentum: T,
    /// Whether any drift was detected on either monitored stream this step.
    pub drift_detected: bool,
    /// The categorical reason for the decision.
    pub reason: AdaptationReason,
}

/// Configuration for the [`RealtimeAdaptationController`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealtimeAdaptationConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Detector configuration for the loss stream.
    pub loss_detector: DriftDetectorConfig<T>,
    /// Detector configuration for the gradient-norm stream.
    pub grad_detector: DriftDetectorConfig<T>,
    /// Initial learning rate.
    pub initial_lr: T,
    /// Lower clamp on the learning rate.
    pub lr_min: T,
    /// Upper clamp on the learning rate.
    pub lr_max: T,
    /// Multiplicative factor `< 1` applied to decrease the learning rate.
    pub lr_decay: T,
    /// Multiplicative factor `> 1` applied to increase the learning rate.
    pub lr_growth: T,
    /// Initial momentum.
    pub initial_momentum: T,
    /// Lower clamp on momentum.
    pub mom_min: T,
    /// Upper clamp on momentum.
    pub mom_max: T,
    /// Multiplicative factor `<= 1` used to damp momentum on divergence.
    pub mom_damp: T,
    /// Number of consecutive low-improvement loss steps that constitute a
    /// plateau.
    pub plateau_patience: usize,
    /// Relative-improvement tolerance below which a loss step counts as "no
    /// progress" for plateau detection.
    pub plateau_tol: T,
    /// Number of steps after any adjustment during which no new adjustment is
    /// made.
    pub cooldown_steps: usize,
    /// Multiplier on the gradient-norm reference mean above which the gradient
    /// is considered "unbounded" (suppressing learning-rate increases).
    pub grad_norm_ceiling_factor: T,
}

impl<T> RealtimeAdaptationConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Validate the controller configuration and both detector configurations.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] describing the first malformed
    /// field encountered.
    pub fn validate(&self) -> Result<()> {
        self.loss_detector.validate()?;
        self.grad_detector.validate()?;

        if self.lr_min <= T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "lr_min must be positive, got {:?}",
                self.lr_min
            )));
        }
        if self.lr_min > self.lr_max {
            return Err(OptimError::InvalidConfig(format!(
                "lr_min ({:?}) must not exceed lr_max ({:?})",
                self.lr_min, self.lr_max
            )));
        }
        if !(self.initial_lr >= self.lr_min && self.initial_lr <= self.lr_max) {
            return Err(OptimError::InvalidConfig(format!(
                "initial_lr ({:?}) must lie within [lr_min, lr_max] = [{:?}, {:?}]",
                self.initial_lr, self.lr_min, self.lr_max
            )));
        }
        if !(self.lr_decay > T::zero() && self.lr_decay < T::one()) {
            return Err(OptimError::InvalidConfig(format!(
                "lr_decay must lie in (0, 1), got {:?}",
                self.lr_decay
            )));
        }
        if self.lr_growth <= T::one() {
            return Err(OptimError::InvalidConfig(format!(
                "lr_growth must be greater than 1, got {:?}",
                self.lr_growth
            )));
        }
        if self.mom_min > self.mom_max {
            return Err(OptimError::InvalidConfig(format!(
                "mom_min ({:?}) must not exceed mom_max ({:?})",
                self.mom_min, self.mom_max
            )));
        }
        if !(self.mom_min >= T::zero() && self.mom_max < T::one()) {
            return Err(OptimError::InvalidConfig(format!(
                "momentum bounds must satisfy 0 <= mom_min and mom_max < 1, got [{:?}, {:?}]",
                self.mom_min, self.mom_max
            )));
        }
        if !(self.initial_momentum >= self.mom_min && self.initial_momentum <= self.mom_max) {
            return Err(OptimError::InvalidConfig(format!(
                "initial_momentum ({:?}) must lie within [mom_min, mom_max] = [{:?}, {:?}]",
                self.initial_momentum, self.mom_min, self.mom_max
            )));
        }
        if !(self.mom_damp > T::zero() && self.mom_damp <= T::one()) {
            return Err(OptimError::InvalidConfig(format!(
                "mom_damp must lie in (0, 1], got {:?}",
                self.mom_damp
            )));
        }
        if self.plateau_patience == 0 {
            return Err(OptimError::InvalidConfig(
                "plateau_patience must be at least 1".to_string(),
            ));
        }
        if self.plateau_tol < T::zero() {
            return Err(OptimError::InvalidConfig(format!(
                "plateau_tol must be non-negative, got {:?}",
                self.plateau_tol
            )));
        }
        if self.grad_norm_ceiling_factor <= T::one() {
            return Err(OptimError::InvalidConfig(format!(
                "grad_norm_ceiling_factor must be greater than 1, got {:?}",
                self.grad_norm_ceiling_factor
            )));
        }
        Ok(())
    }
}

impl<T> Default for RealtimeAdaptationConfig<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    fn default() -> Self {
        let from = |v: f64, fallback: T| T::from(v).unwrap_or(fallback);
        Self {
            loss_detector: DriftDetectorConfig::default(),
            grad_detector: DriftDetectorConfig::default(),
            initial_lr: from(1e-3, T::epsilon()),
            lr_min: from(1e-6, T::epsilon()),
            lr_max: from(1e-1, T::one()),
            lr_decay: from(0.5, T::one()),
            lr_growth: from(1.1, T::one() + T::one()),
            initial_momentum: from(0.9, T::one()),
            mom_min: from(0.5, T::zero()),
            mom_max: from(0.99, T::one()),
            mom_damp: from(0.9, T::one()),
            plateau_patience: 10,
            plateau_tol: from(1e-4, T::epsilon()),
            cooldown_steps: 5,
            grad_norm_ceiling_factor: from(3.0, T::one() + T::one()),
        }
    }
}

/// Online controller that adapts the learning rate and momentum in response to
/// drift detected on the loss and gradient-norm streams.
#[derive(Debug, Clone)]
pub struct RealtimeAdaptationController<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Validated configuration.
    config: RealtimeAdaptationConfig<T>,
    /// Drift detector for the loss stream.
    loss_detector: DriftDetector<T>,
    /// Drift detector for the gradient-norm stream.
    grad_detector: DriftDetector<T>,
    /// Current learning rate.
    learning_rate: T,
    /// Current momentum.
    momentum: T,
    /// Steps remaining in the post-adjustment cooldown window.
    cooldown_remaining: usize,
    /// EWMA of the loss used for plateau detection.
    plateau_loss_ema: T,
    /// Whether `plateau_loss_ema` has been initialised.
    plateau_initialised: bool,
    /// Number of consecutive low-improvement steps seen so far.
    plateau_counter: usize,
    /// Total observations processed.
    step_count: usize,
}

impl<T> RealtimeAdaptationController<T>
where
    T: Float + Debug + Send + Sync + 'static,
{
    /// Build a controller from a configuration.
    ///
    /// # Errors
    ///
    /// Returns [`OptimError::InvalidConfig`] if
    /// [`RealtimeAdaptationConfig::validate`] fails.
    pub fn new(config: RealtimeAdaptationConfig<T>) -> Result<Self> {
        config.validate()?;
        let loss_detector = DriftDetector::new(config.loss_detector)?;
        let grad_detector = DriftDetector::new(config.grad_detector)?;
        let learning_rate = config.initial_lr;
        let momentum = config.initial_momentum;
        Ok(Self {
            config,
            loss_detector,
            grad_detector,
            learning_rate,
            momentum,
            cooldown_remaining: 0,
            plateau_loss_ema: T::zero(),
            plateau_initialised: false,
            plateau_counter: 0,
            step_count: 0,
        })
    }

    /// Current learning rate.
    pub fn learning_rate(&self) -> T {
        self.learning_rate
    }

    /// Current momentum.
    pub fn momentum(&self) -> T {
        self.momentum
    }

    /// Number of observations processed so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Steps remaining in the current cooldown window (zero when not cooling).
    pub fn cooldown_remaining(&self) -> usize {
        self.cooldown_remaining
    }

    /// Immutable access to the loss-stream detector.
    pub fn loss_detector(&self) -> &DriftDetector<T> {
        &self.loss_detector
    }

    /// Immutable access to the gradient-norm-stream detector.
    pub fn grad_detector(&self) -> &DriftDetector<T> {
        &self.grad_detector
    }

    /// Clamp a learning-rate candidate to `[lr_min, lr_max]`.
    fn clamp_lr(&self, lr: T) -> T {
        if lr < self.config.lr_min {
            self.config.lr_min
        } else if lr > self.config.lr_max {
            self.config.lr_max
        } else {
            lr
        }
    }

    /// Clamp a momentum candidate to `[mom_min, mom_max]`.
    fn clamp_mom(&self, mom: T) -> T {
        if mom < self.config.mom_min {
            self.config.mom_min
        } else if mom > self.config.mom_max {
            self.config.mom_max
        } else {
            mom
        }
    }

    /// Update the plateau tracker with the latest loss and return whether the
    /// loss has plateaued (no meaningful relative improvement for
    /// `plateau_patience` consecutive steps).
    fn update_plateau(&mut self, loss: T) -> bool {
        let lambda = self.config.loss_detector.lambda;
        let one_minus = T::one() - lambda;

        if !self.plateau_initialised {
            self.plateau_loss_ema = loss;
            self.plateau_initialised = true;
            self.plateau_counter = 0;
            return false;
        }

        let prev = self.plateau_loss_ema;
        let new_ema = lambda * loss + one_minus * prev;
        // Relative improvement of the EWMA loss. Positive means the loss went
        // down. We normalise by |prev| (with a small floor) so the tolerance is
        // scale-free.
        let denom = {
            let a = prev.abs();
            if a > self.config.loss_detector.min_std {
                a
            } else {
                self.config.loss_detector.min_std
            }
        };
        let rel_improvement = (prev - new_ema) / denom;
        self.plateau_loss_ema = new_ema;

        if rel_improvement < self.config.plateau_tol {
            // Not enough progress (this also covers loss increases).
            self.plateau_counter = self.plateau_counter.saturating_add(1);
        } else {
            self.plateau_counter = 0;
        }

        self.plateau_counter >= self.config.plateau_patience
    }

    /// Whether the gradient norm is currently within its "bounded" envelope,
    /// i.e. its EWMA mean is below `grad_norm_ceiling_factor · μ0`. Before the
    /// gradient detector warms up we optimistically treat the gradient as
    /// bounded so early stable-improvement increases are not blocked.
    fn grad_norm_bounded(&self) -> bool {
        if !self.grad_detector.is_warmed_up() {
            return true;
        }
        let ceiling = self.config.grad_norm_ceiling_factor * self.grad_detector.reference_mean();
        self.grad_detector.ewma_mean() <= ceiling
    }

    /// Process one `(loss, grad_norm)` observation and adapt the hyperparameters.
    ///
    /// The two detectors are always updated (so their internal statistics stay
    /// current even during cooldown), but at most one learning-rate adjustment
    /// is applied per call, and none while a cooldown window is active.
    pub fn observe(&mut self, loss: T, grad_norm: T) -> AdaptationDecision<T> {
        self.step_count = self.step_count.saturating_add(1);

        let loss_signal = self.loss_detector.update(loss);
        let grad_signal = self.grad_detector.update(grad_norm);
        let plateaued = self.update_plateau(loss);

        let drift_detected =
            loss_signal != DriftSignal::Stable || grad_signal != DriftSignal::Stable;

        // Respect the cooldown window: decrement and emit a no-change decision.
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            return AdaptationDecision {
                learning_rate: self.learning_rate,
                momentum: self.momentum,
                drift_detected,
                reason: AdaptationReason::NoChange,
            };
        }

        // Priority order: divergence / explosion (safety) first, then plateau,
        // then opportunistic growth.
        let loss_diverging = loss_signal == DriftSignal::UpwardDrift;
        let grad_exploding = grad_signal == DriftSignal::UpwardDrift;
        let loss_improving = loss_signal == DriftSignal::DownwardDrift;

        let reason = if loss_diverging || grad_exploding {
            // Multiplicative decrease + momentum damping.
            self.learning_rate = self.clamp_lr(self.learning_rate * self.config.lr_decay);
            self.momentum = self.clamp_mom(self.momentum * self.config.mom_damp);
            if loss_diverging {
                AdaptationReason::LossDivergence
            } else {
                AdaptationReason::GradientExplosion
            }
        } else if plateaued {
            // ReduceLROnPlateau-style decrease to refine.
            self.learning_rate = self.clamp_lr(self.learning_rate * self.config.lr_decay);
            // Reset the plateau counter so we wait another full patience window
            // before reducing again.
            self.plateau_counter = 0;
            AdaptationReason::Plateau
        } else if loss_improving && self.grad_norm_bounded() {
            // Additive-increase flavour: gently grow towards the ceiling.
            let grown = self.learning_rate * self.config.lr_growth;
            self.learning_rate = self.clamp_lr(grown);
            AdaptationReason::StableImprovement
        } else {
            AdaptationReason::NoChange
        };

        // Engage cooldown only if we actually adjusted.
        if reason != AdaptationReason::NoChange {
            self.cooldown_remaining = self.config.cooldown_steps;
        }

        AdaptationDecision {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            drift_detected,
            reason,
        }
    }

    /// Push the controller's current learning rate onto an
    /// [`AdvancedOptimizer`], a thin bridge for wiring the controller into a
    /// training loop.
    pub fn apply_to(&self, optimizer: &mut dyn AdvancedOptimizer<T>) {
        optimizer.set_learning_rate(self.learning_rate);
    }

    /// Reset the controller and both detectors to their initial state.
    pub fn reset(&mut self) {
        self.loss_detector.reset();
        self.grad_detector.reset();
        self.learning_rate = self.config.initial_lr;
        self.momentum = self.config.initial_momentum;
        self.cooldown_remaining = 0;
        self.plateau_loss_ema = T::zero();
        self.plateau_initialised = false;
        self.plateau_counter = 0;
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_optimizers::{AdvancedOptimizer, OptimizerStateInfo};
    use scirs2_core::ndarray::Array1;

    fn detector_config() -> DriftDetectorConfig<f64> {
        DriftDetectorConfig {
            lambda: 0.2,
            control_limit: 3.0,
            cusum_slack: 0.5,
            cusum_threshold: 5.0,
            warmup_steps: 20,
            min_std: 1e-8,
        }
    }

    /// A tiny deterministic SGD optimizer used to exercise `apply_to`.
    struct DummyOptimizer {
        lr: f64,
    }

    impl AdvancedOptimizer<f64> for DummyOptimizer {
        fn step(&mut self, params: &Array1<f64>, gradients: &Array1<f64>) -> Result<Array1<f64>> {
            Ok(params - &gradients.mapv(|g| g * self.lr))
        }
        fn get_learning_rate(&self) -> f64 {
            self.lr
        }
        fn set_learning_rate(&mut self, lr: f64) {
            self.lr = lr;
        }
        fn name(&self) -> &str {
            "DummyOptimizer"
        }
        fn get_state(&self) -> OptimizerStateInfo<f64> {
            OptimizerStateInfo {
                step_count: 0,
                current_lr: self.lr,
                grad_norm_ema: 0.0,
            }
        }
    }

    /// Deterministic pseudo-random sequence in [0, 1) (LCG), used so tests stay
    /// reproducible without pulling in an RNG dependency.
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_unit(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Use the top 53 bits for a double in [0, 1).
            ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
        }
        /// Centered noise in [-amp, amp].
        fn next_noise(&mut self, amp: f64) -> f64 {
            (self.next_unit() * 2.0 - 1.0) * amp
        }
    }

    #[test]
    fn test_ewma_mean_converges_to_stream_mean() {
        let mut detector = DriftDetector::new(detector_config()).expect("valid config");
        let target = 5.0;
        let mut rng = Lcg::new(42);
        // Warm-up on the target plus tiny noise, then keep feeding the same.
        for _ in 0..200 {
            let x = target + rng.next_noise(0.01);
            let _ = detector.update(x);
        }
        assert!(detector.is_warmed_up());
        assert!(
            (detector.ewma_mean() - target).abs() < 0.05,
            "EWMA mean {} should be close to {}",
            detector.ewma_mean(),
            target
        );
        assert!(
            (detector.reference_mean() - target).abs() < 0.05,
            "reference mean {} should be close to {}",
            detector.reference_mean(),
            target
        );
    }

    #[test]
    fn test_cusum_stable_on_stationary_stream() {
        let mut detector = DriftDetector::new(detector_config()).expect("valid config");
        let mut rng = Lcg::new(7);
        let mut drift_count = 0usize;
        for _ in 0..500 {
            let x = 2.0 + rng.next_noise(0.05);
            if detector.update(x) != DriftSignal::Stable {
                drift_count += 1;
            }
        }
        assert_eq!(
            drift_count, 0,
            "stationary stream should not raise any drift signal"
        );
    }

    #[test]
    fn test_cusum_detects_upward_shift() {
        let mut detector = DriftDetector::new(detector_config()).expect("valid config");
        let mut rng = Lcg::new(11);
        // Warm-up around 0.
        for _ in 0..30 {
            let _ = detector.update(rng.next_noise(0.05));
        }
        assert!(detector.is_warmed_up());
        // Inject a large positive mean shift; a clear upward drift must fire.
        let mut fired_up = false;
        for _ in 0..50 {
            let x = 3.0 + rng.next_noise(0.05);
            if detector.update(x) == DriftSignal::UpwardDrift {
                fired_up = true;
                break;
            }
        }
        assert!(fired_up, "upward mean shift must raise UpwardDrift");
    }

    #[test]
    fn test_cusum_detects_downward_shift() {
        let mut detector = DriftDetector::new(detector_config()).expect("valid config");
        let mut rng = Lcg::new(13);
        for _ in 0..30 {
            let _ = detector.update(rng.next_noise(0.05));
        }
        let mut fired_down = false;
        for _ in 0..50 {
            let x = -3.0 + rng.next_noise(0.05);
            if detector.update(x) == DriftSignal::DownwardDrift {
                fired_down = true;
                break;
            }
        }
        assert!(fired_down, "downward mean shift must raise DownwardDrift");
    }

    #[test]
    fn test_detector_warmup_reports_stable() {
        let mut detector = DriftDetector::new(detector_config()).expect("valid config");
        // Even with wild values, warm-up never reports drift.
        for i in 0..20 {
            let x = if i % 2 == 0 { 100.0 } else { -100.0 };
            assert_eq!(detector.update(x), DriftSignal::Stable);
        }
        assert!(detector.is_warmed_up());
    }

    #[test]
    fn test_controller_decreases_lr_on_diverging_loss() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let initial_lr = controller.learning_rate();
        let mut rng = Lcg::new(101);
        // Stable warm-up phase for both streams.
        for _ in 0..30 {
            let _ = controller.observe(1.0 + rng.next_noise(0.01), 0.5 + rng.next_noise(0.01));
        }
        // Now drive the loss sharply upward (diverging).
        let mut loss = 1.0;
        for _ in 0..60 {
            loss += 0.5;
            let _ = controller.observe(loss, 0.5 + rng.next_noise(0.01));
        }
        assert!(
            controller.learning_rate() < initial_lr,
            "lr {} should drop below initial {} on diverging loss",
            controller.learning_rate(),
            initial_lr
        );
    }

    #[test]
    fn test_controller_decreases_lr_on_exploding_grad() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let initial_lr = controller.learning_rate();
        let mut rng = Lcg::new(202);
        for _ in 0..30 {
            let _ = controller.observe(1.0 + rng.next_noise(0.01), 0.5 + rng.next_noise(0.01));
        }
        // Explode the gradient norm while keeping loss roughly flat.
        let mut g = 0.5;
        for _ in 0..60 {
            g += 0.3;
            let _ = controller.observe(1.0 + rng.next_noise(0.01), g);
        }
        assert!(
            controller.learning_rate() < initial_lr,
            "lr {} should drop below initial {} on exploding grad norm",
            controller.learning_rate(),
            initial_lr
        );
    }

    #[test]
    fn test_controller_increases_lr_on_improving_loss() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let lr_max = config.lr_max;
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let initial_lr = controller.learning_rate();
        let mut rng = Lcg::new(303);
        // Warm-up around a high loss.
        for _ in 0..30 {
            let _ = controller.observe(10.0 + rng.next_noise(0.01), 0.5 + rng.next_noise(0.01));
        }
        // Steadily decreasing loss with a small, bounded gradient norm.
        let mut loss = 10.0;
        for _ in 0..200 {
            loss -= 0.04;
            let _ = controller.observe(loss, 0.5 + rng.next_noise(0.01));
        }
        assert!(
            controller.learning_rate() > initial_lr,
            "lr {} should rise above initial {} on improving loss",
            controller.learning_rate(),
            initial_lr
        );
        assert!(
            controller.learning_rate() <= lr_max + 1e-15,
            "lr {} must never exceed lr_max {}",
            controller.learning_rate(),
            lr_max
        );
    }

    #[test]
    fn test_controller_plateau_triggers_reduction() {
        // Disable the drift paths' interference by using a flat loss: the
        // plateau tracker (not divergence/improvement) must drive the cut.
        let config = RealtimeAdaptationConfig::<f64> {
            plateau_patience: 8,
            cooldown_steps: 0,
            ..Default::default()
        };
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let initial_lr = controller.learning_rate();
        // Warm up, then a long perfectly-flat loss plateau.
        for _ in 0..25 {
            let _ = controller.observe(3.0, 0.5);
        }
        let lr_before = controller.learning_rate();
        for _ in 0..30 {
            let _ = controller.observe(3.0, 0.5);
        }
        assert!(
            controller.learning_rate() < lr_before,
            "flat loss plateau should reduce lr: before={}, after={}",
            lr_before,
            controller.learning_rate()
        );
        assert!(controller.learning_rate() <= initial_lr);
    }

    #[test]
    fn test_lr_and_momentum_stay_within_clamps() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let lr_min = config.lr_min;
        let lr_max = config.lr_max;
        let mom_min = config.mom_min;
        let mom_max = config.mom_max;
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let mut rng = Lcg::new(404);
        for i in 0..1000 {
            // A messy but deterministic stream mixing trends, spikes, and noise.
            let phase = (i / 50) % 4;
            let base = match phase {
                0 => 5.0 - 0.05 * (i % 50) as f64, // improving
                1 => 1.0 + 0.1 * (i % 50) as f64,  // diverging
                2 => 2.0,                          // plateau
                _ => 3.0 + rng.next_noise(1.0),    // noisy
            };
            let loss = base + rng.next_noise(0.05);
            let grad = (0.5 + rng.next_noise(0.4)).abs() + 0.01;
            let decision = controller.observe(loss, grad);
            assert!(
                decision.learning_rate >= lr_min - 1e-15
                    && decision.learning_rate <= lr_max + 1e-15,
                "lr {} out of clamp [{}, {}] at step {}",
                decision.learning_rate,
                lr_min,
                lr_max,
                i
            );
            assert!(
                decision.momentum >= mom_min - 1e-15 && decision.momentum <= mom_max + 1e-15,
                "momentum {} out of clamp [{}, {}] at step {}",
                decision.momentum,
                mom_min,
                mom_max,
                i
            );
        }
    }

    #[test]
    fn test_cooldown_suppresses_immediate_second_adjustment() {
        let config = RealtimeAdaptationConfig::<f64> {
            cooldown_steps: 5,
            ..Default::default()
        };
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let mut rng = Lcg::new(505);
        for _ in 0..30 {
            let _ = controller.observe(1.0 + rng.next_noise(0.01), 0.5 + rng.next_noise(0.01));
        }
        // Force a divergence-driven adjustment.
        let mut loss = 1.0;
        let mut adjusted_step = None;
        for step in 0..100 {
            loss += 0.5;
            let d = controller.observe(loss, 0.5 + rng.next_noise(0.01));
            if d.reason != AdaptationReason::NoChange {
                adjusted_step = Some(step);
                break;
            }
        }
        let adjusted_step = adjusted_step.expect("an adjustment should have occurred");
        let lr_after_adjust = controller.learning_rate();
        // The immediately following observations (still diverging) must not
        // change the lr again while cooldown is active.
        for _ in 0..5 {
            loss += 0.5;
            let d = controller.observe(loss, 0.5);
            assert_eq!(
                d.reason,
                AdaptationReason::NoChange,
                "cooldown must suppress adjustment right after step {adjusted_step}"
            );
            assert!(
                (d.learning_rate - lr_after_adjust).abs() < 1e-18,
                "lr must be unchanged during cooldown"
            );
        }
    }

    /// Build a default controller config, apply a mutation, and return it.
    /// Taking the base as a function argument keeps this out of the
    /// `field_reassign_with_default` lint's pattern (a `let` binding followed by
    /// field assignments) while still letting each test tweak one field.
    fn mutated_config<F>(mut f: F) -> RealtimeAdaptationConfig<f64>
    where
        F: FnMut(&mut RealtimeAdaptationConfig<f64>),
    {
        let mut config = RealtimeAdaptationConfig::<f64>::default();
        f(&mut config);
        config
    }

    #[test]
    fn test_config_validate_rejects_bad_lambda() {
        let low = mutated_config(|c| c.loss_detector.lambda = 0.0);
        assert!(low.validate().is_err());
        let high = mutated_config(|c| c.loss_detector.lambda = 1.5);
        assert!(high.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_bad_cusum_params() {
        let bad_threshold = mutated_config(|c| c.grad_detector.cusum_threshold = 0.0);
        assert!(bad_threshold.validate().is_err());
        let bad_slack = mutated_config(|c| c.grad_detector.cusum_slack = -1.0);
        assert!(bad_slack.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_lr_bounds() {
        let config = mutated_config(|c| {
            c.lr_min = 1.0;
            c.lr_max = 0.5;
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_momentum_bounds() {
        let config = mutated_config(|c| {
            c.mom_min = 0.95;
            c.mom_max = 0.5;
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_zero_patience() {
        let config = mutated_config(|c| c.plateau_patience = 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_bad_growth_decay() {
        let bad_decay = mutated_config(|c| c.lr_decay = 1.0); // must be < 1
        assert!(bad_decay.validate().is_err());
        let bad_growth = mutated_config(|c| c.lr_growth = 0.9); // must be > 1
        assert!(bad_growth.validate().is_err());
    }

    #[test]
    fn test_default_config_validates_ok() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        assert!(config.validate().is_ok());
        let detector_default = DriftDetectorConfig::<f64>::default();
        assert!(detector_default.validate().is_ok());
    }

    #[test]
    fn test_apply_to_pushes_learning_rate() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let initial_lr = config.initial_lr;
        let controller = RealtimeAdaptationController::new(config).expect("valid config");
        let mut opt = DummyOptimizer { lr: 999.0 };
        controller.apply_to(&mut opt);
        assert!(
            (opt.get_learning_rate() - initial_lr).abs() < 1e-15,
            "apply_to should push the controller lr {} onto the optimizer (got {})",
            initial_lr,
            opt.get_learning_rate()
        );
    }

    #[test]
    fn test_reset_restores_initial_state() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let initial_lr = config.initial_lr;
        let initial_mom = config.initial_momentum;
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let mut rng = Lcg::new(606);
        for _ in 0..50 {
            let _ = controller.observe(1.0 + rng.next_noise(0.5), 0.5 + rng.next_noise(0.5).abs());
        }
        controller.reset();
        assert_eq!(controller.step_count(), 0);
        assert!((controller.learning_rate() - initial_lr).abs() < 1e-15);
        assert!((controller.momentum() - initial_mom).abs() < 1e-15);
        assert!(!controller.loss_detector().is_warmed_up());
        assert!(!controller.grad_detector().is_warmed_up());
    }

    #[test]
    fn test_drift_detected_flag_set_on_shift() {
        let config = RealtimeAdaptationConfig::<f64>::default();
        let mut controller = RealtimeAdaptationController::new(config).expect("valid config");
        let mut rng = Lcg::new(707);
        for _ in 0..30 {
            let _ = controller.observe(1.0 + rng.next_noise(0.01), 0.5 + rng.next_noise(0.01));
        }
        let mut saw_drift = false;
        let mut loss = 1.0;
        for _ in 0..60 {
            loss += 0.5;
            if controller
                .observe(loss, 0.5 + rng.next_noise(0.01))
                .drift_detected
            {
                saw_drift = true;
                break;
            }
        }
        assert!(saw_drift, "drift_detected should be set during divergence");
    }
}
