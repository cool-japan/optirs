// Real model-based drift detectors for `ModelType::{NeuralNetwork, DecisionTree, Ensemble}`.
//
// `ModelType` names four model families. Only `Linear` had an implementation
// (`drift_tests::LinearModelDetector`); the other three were name-only variants
// that `EnhancedDriftDetector` answered with an honest "no detector is
// registered" error. This module supplies the three missing models.
//
// All three follow the same contract as the linear detector, because
// `EnhancedDriftDetector::detect_model_drift` depends on it:
//
// * they are **supervised** — a `StreamingDataPoint` without a target cannot
//   train or score a predictor, so unlabelled points are skipped rather than
//   trained against a fabricated label;
// * they are **prequential** — every labelled point is scored by the current
//   model *before* the model learns from it, which is the standard streaming
//   evaluation and the only ordering under which a rise in error means "the
//   world changed" rather than "the model has not seen this yet";
// * `ModelDriftResult::confidence` is `1 - p` where `p` comes from a real null
//   distribution. `detect_model_drift` recovers the p-value as
//   `1 - confidence`, so anything else there (a vote share, a squashed score)
//   would make the reported significance a fabrication.
//
// Drift rule (shared, see [`PrequentialErrorTracker`]): a fast and a slow
// exponentially-weighted mean of the squared prediction error are tracked
// together. On a stationary stream both converge to the same value, so their
// difference fluctuates around zero; a genuine change makes the fast mean jump
// while the slow one lags. Drift is reported when the fast mean stays above
// `slow * (1 + sensitivity)` for `MIN_DRIFT_RUN` consecutive observations *and*
// a one-sided z test on the gap clears `DRIFT_SIGNIFICANCE` — the persistence
// requirement that DDM and Page-Hinkley use to separate a sustained shift from
// ordinary noise, plus a significance gate scaled by how much this particular
// stream's squared error actually varies.
//
// Two properties make that rule survive contact with a *self-adapting* model,
// which is what all three of these detectors monitor:
//
// * the baseline is **Winsorised** (`BASELINE_CLIP_SIGMAS`). The slow mean and
//   the spread the z test divides by are updated from a clipped deviation, so
//   the reference cannot be redefined by the very change it exists to detect.
//   Without it a single 100x jump multiplies the estimated spread by five
//   orders of magnitude on the first observation of the drift, and the
//   detector's own p-value climbs back through `0.05` while the error is still
//   several times its baseline.
// * once drift is reported the error statistics **restart**, which is DDM's
//   published behaviour and what `DdmTest`/`PageHinkleyTest` already do in this
//   crate. The learned model is kept; only the reference is retired with the
//   concept it described.
//
// References
// - Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia, "A Survey on Concept
//   Drift Adaptation", ACM Computing Surveys 2014 (prequential evaluation,
//   error-rate drift detectors).
// - Breiman, Friedman, Olshen & Stone, "Classification and Regression Trees",
//   1984 (the CART split rule used by the tree detector).
// - Fisher, "Statistical Methods for Research Workers", 1932 (the combination
//   of independent p-values used by the ensemble).

use super::drift_detection::{ModelBasedDetector, ModelDriftResult};
use super::drift_tests::LinearModelDetector;
use super::optimizer::StreamingDataPoint;
use super::statistics as stats;

use crate::utils::try_scalar_str;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Weight of the most recent observation in the *fast* error mean.
const ALPHA_FAST: f64 = 0.1;

/// Weight of the most recent observation in the *slow* (baseline) error mean.
const ALPHA_SLOW: f64 = 0.01;

/// Consecutive observations that must exceed the degradation threshold before
/// drift is reported.
///
/// This is deliberately several times the fast mean's correlation length
/// (`1 / ALPHA_FAST = 10` observations). A single excursion of an
/// exponentially-weighted mean above a threshold is *not* rare — successive
/// values of the mean are ~90% correlated, so an excursion that happens at all
/// typically lasts about one correlation length. Requiring five correlation
/// lengths is what separates "the mean wandered" from "the level moved".
///
/// **This budget is only satisfiable because the baseline is Winsorised.** It is
/// not a property of the fast/slow pair on its own: the models these detectors
/// monitor re-adapt, so the excursion ends when either the model re-learns *or*
/// the baseline climbs to meet the fast mean, whichever happens first. Measured
/// with an unclipped baseline, a complete change of the target relationship
/// produced a run of 41–49 observations before the slow mean overtook the fast
/// one — just short of this constant, so the drift was missed. With the clip at
/// [`BASELINE_CLIP_SIGMAS`] the same shift produces a run of 75+ and the verdict
/// lands at observation 50.
///
/// The two constants therefore have to be tuned together: widening the clip
/// lets the baseline climb faster and shortens the achievable run, and at
/// `BASELINE_CLIP_SIGMAS = 6` the baseline overtakes the fast mean before this
/// budget is met.
const MIN_DRIFT_RUN: usize = 50;

/// Significance the one-sided z test on the fast/slow gap must reach before
/// drift is reported, in addition to the persistence rule above.
///
/// The two gates measure different things and both are needed: the persistence
/// rule answers "has the level moved and stayed moved", the z test answers "is
/// the gap large relative to how much this stream's squared error varies".
/// `EnhancedDriftDetector` separately applies the *configured*
/// `significance_level` to the p-value reported here, so this constant only
/// governs the detector's own verdict.
const DRIFT_SIGNIFICANCE: f64 = 0.05;

/// Observations required before any verdict is issued, so that the slow mean
/// has settled (its time constant is `1 / ALPHA_SLOW = 100` observations) and
/// the model has left its initial learning transient.
const WARMUP_OBSERVATIONS: usize = 200;

/// Standard deviations at which the *baseline* update is Winsorised.
///
/// The baseline exists to say what the error level was **before** the change,
/// so it must not absorb the change itself. Without this the detector blinds
/// itself in a single step: the variance estimator is an EWMA of the squared
/// deviation from the baseline, so one observation `k` standard deviations out
/// multiplies the variance by roughly `ALPHA_SLOW * k^2`. A 100x jump in the
/// squared error therefore inflates the estimated spread by five orders of
/// magnitude *on the first observation of the drift*, which is exactly when the
/// z test needs the pre-drift spread. Measured on the neural detector: the
/// variance went from `1.0e-5` to `70` in one observation, and the p-value —
/// `4e-5` on that first observation — climbed back above `0.05` twenty
/// observations later while the error was still five times its baseline.
///
/// So the deviation that drives the slow mean and its spread is clipped to
/// `+/- BASELINE_CLIP_SIGMAS` standard deviations, while the *fast* mean stays
/// unclipped. That is the whole asymmetry the detector rests on: the fast mean
/// must register the change, the baseline must not.
///
/// Four sigma is a deliberate choice rather than a round number. Winsorising
/// biases the scale estimate downwards (the fixed point of
/// `V = E[min(d^2, c^2 V)]` sits below `E[d^2]`), which makes the z test
/// slightly liberal; a wider clip reduces that bias but also lets the baseline
/// climb faster during a real excursion, since the baseline's per-observation
/// step is `ALPHA_SLOW * c * sqrt(V)` and `V` itself then grows by a factor
/// `1 + ALPHA_SLOW * (c^2 - 1)` per clipped observation. At `c = 4` the scale
/// estimate settles around 85% of the true spread on a chi-square-like error
/// distribution (a ~9% inflation of the z score, which the persistence rule
/// absorbs), while the baseline needs well over [`MIN_DRIFT_RUN`] observations
/// to climb far enough to end a genuine excursion. At `c = 6` the baseline
/// overtakes the fast mean before the persistence rule is satisfied, and the
/// drift is missed.
const BASELINE_CLIP_SIGMAS: f64 = 4.0;

/// Observations required before the Winsorising clip engages.
///
/// The clip radius is derived from the tracker's own spread estimate, so it can
/// only be applied once that estimate describes something. Engaging it earlier
/// would bootstrap the radius off the first one or two observations and could
/// pin the baseline to whatever the stream happened to start at.
const BASELINE_CLIP_WARMUP: usize = 30;

/// Hidden units in the online MLP.
const MLP_HIDDEN_UNITS: usize = 8;

/// Seed for the deterministic weight initialiser. Fixed so that two detectors
/// constructed the same way behave identically — a drift detector whose verdict
/// depends on process-level entropy is not reproducible.
const MLP_INIT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Labelled observations retained by the decision-tree detector.
const TREE_WINDOW_CAPACITY: usize = 256;

/// Observations between two refits of the decision tree.
const TREE_REFIT_INTERVAL: usize = 32;

/// Maximum depth of the fitted tree (root counts as depth 0).
const TREE_MAX_DEPTH: usize = 4;

/// Minimum observations that must remain in each child of a split.
const TREE_MIN_SAMPLES_LEAF: usize = 8;

/// Converts a generic float into the element type, reporting an honest error.
fn from_f64<A: Float>(value: f64) -> Result<A, String> {
    try_scalar_str::<A, _>(value)
}

/// Whether `value` is **not** strictly greater than `bound`.
///
/// A NaN on either side answers `true`: it is not greater, and every caller
/// here wants a NaN to take the conservative branch (no split, no clip radius,
/// no significance) rather than to propagate. Written through `partial_cmp`
/// rather than as `!(value > bound)` so that the incomparable case is visible
/// in the code instead of hiding inside a negated float comparison.
fn is_not_above(value: f64, bound: f64) -> bool {
    !matches!(value.partial_cmp(&bound), Some(std::cmp::Ordering::Greater))
}

/// Extracts a supervised `(features, target)` pair from a data point.
///
/// A point without a target cannot train or score a predictor, so it is
/// skipped rather than trained against an invented label.
fn supervised_pair<A: Float + Send + Sync>(
    data_point: &StreamingDataPoint<A>,
) -> Option<(Vec<f64>, f64)> {
    let target = data_point.target.as_ref()?;
    let target_value = target.iter().next()?.to_f64()?;
    if !target_value.is_finite() {
        return None;
    }
    let features: Vec<f64> = data_point
        .features
        .iter()
        .filter_map(|v| v.to_f64())
        .filter(|v| v.is_finite())
        .collect();
    if features.is_empty() {
        return None;
    }
    Some((features, target_value))
}

// ---------------------------------------------------------------------------
// shared error tracking
// ---------------------------------------------------------------------------

/// Fast/slow exponentially-weighted tracking of a model's squared prediction
/// error, plus the persistence rule that turns it into a drift verdict.
///
/// A single mean compared against a min-tracking baseline (what the linear
/// detector does) drifts downwards on a stationary stream, so ordinary noise
/// eventually clears any relative threshold. Comparing two means of the *same*
/// series removes that bias: both are unbiased estimates of the current error
/// level, so on a stationary stream their difference has mean zero regardless
/// of how long the detector has been running.
#[derive(Debug, Clone)]
pub struct PrequentialErrorTracker {
    /// Fast exponentially-weighted mean of the squared error.
    fast: f64,
    /// Slow exponentially-weighted mean of the squared error (the baseline).
    slow: f64,
    /// Exponentially-weighted variance of the squared error, for the z test.
    variance: f64,
    /// Observations folded in so far.
    updates: usize,
    /// Consecutive observations above the degradation threshold.
    run_length: usize,
    /// Relative rise in mean squared error that counts as degradation.
    threshold: f64,
}

impl PrequentialErrorTracker {
    /// Creates a tracker. `sensitivity` is the relative rise in mean squared
    /// error that counts as degradation (`0.05` = a 5% rise).
    fn new(sensitivity: f64) -> Result<Self, String> {
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "model drift sensitivity must be positive and finite, got {sensitivity}"
            ));
        }
        Ok(Self {
            fast: f64::NAN,
            slow: f64::NAN,
            variance: 0.0,
            updates: 0,
            run_length: 0,
            threshold: sensitivity,
        })
    }

    /// Folds one squared prediction error into both means.
    ///
    /// The fast mean sees the observation as it is; the baseline and its spread
    /// see it Winsorised to [`BASELINE_CLIP_SIGMAS`] standard deviations (see
    /// that constant for why). `slow += ALPHA_SLOW * deviation` is algebraically
    /// the same EWMA as `ALPHA_SLOW * x + (1 - ALPHA_SLOW) * slow`, written in
    /// deviation form so the clip has somewhere to apply.
    fn observe(&mut self, squared_error: f64) {
        if !squared_error.is_finite() {
            return;
        }
        if self.fast.is_finite() && self.slow.is_finite() {
            let deviation = squared_error - self.slow;
            let baseline_step = self.winsorise(deviation);
            self.variance =
                ALPHA_SLOW * baseline_step * baseline_step + (1.0 - ALPHA_SLOW) * self.variance;
            self.fast = ALPHA_FAST * squared_error + (1.0 - ALPHA_FAST) * self.fast;
            self.slow += ALPHA_SLOW * baseline_step;
        } else {
            self.fast = squared_error;
            self.slow = squared_error;
            self.variance = 0.0;
        }
        self.updates += 1;

        if self.fast > self.slow * (1.0 + self.threshold) {
            self.run_length += 1;
        } else {
            self.run_length = 0;
        }
    }

    /// Clips a deviation from the baseline to [`BASELINE_CLIP_SIGMAS`] standard
    /// deviations, once the spread estimate is old enough to define a radius.
    ///
    /// Two special cases:
    ///
    /// * Before [`BASELINE_CLIP_WARMUP`] observations there is no trustworthy
    ///   radius yet, so the deviation passes through unchanged.
    /// * A spread of *exactly* zero — a model whose prequential error has not
    ///   varied at all, which a perfectly-fitting model on a noiseless stream
    ///   really does produce — is a degenerate reference with no scale to clip
    ///   against. An upward deviation is then held out entirely: it is the
    ///   drift, and letting it in would define both the baseline level *and*
    ///   the spread the z test divides by from the very change under test. (One
    ///   such observation is enough to blind the detector: `ALPHA_SLOW * d^2`
    ///   for `d = 152100` is a variance of `2.3e8`, against which the whole
    ///   excursion then looks like ordinary noise.) A *downward* deviation is
    ///   adopted in full: a model doing better than its reference is not drift,
    ///   and adopting it is what re-establishes a usable scale.
    fn winsorise(&self, deviation: f64) -> f64 {
        if self.updates < BASELINE_CLIP_WARMUP {
            return deviation;
        }
        if is_not_above(self.variance, 0.0) {
            return deviation.min(0.0);
        }
        let radius = BASELINE_CLIP_SIGMAS * self.variance.sqrt();
        deviation.clamp(-radius, radius)
    }

    /// Relative rise of the fast mean over the slow one, or `None` before the
    /// first observation.
    fn degradation(&self) -> Option<f64> {
        if !(self.fast.is_finite() && self.slow.is_finite()) {
            return None;
        }
        Some((self.fast - self.slow) / self.slow.abs().max(f64::MIN_POSITIVE))
    }

    /// One-sided p-value for "the recent squared error sits above the
    /// baseline".
    ///
    /// The standard error uses the *effective* sample size of the fast mean,
    /// `(2 - alpha) / alpha`, not the total number of updates: an
    /// exponentially-weighted mean averages over roughly that many recent
    /// observations no matter how long the stream is, so dividing by the full
    /// update count would shrink the standard error without bound and make an
    /// arbitrarily small difference look decisive.
    fn p_value(&self) -> Result<f64, String> {
        let (Some(fast), Some(slow)) = (
            self.fast.is_finite().then_some(self.fast),
            self.slow.is_finite().then_some(self.slow),
        ) else {
            return Ok(1.0);
        };
        let effective_n = (2.0 - ALPHA_FAST) / ALPHA_FAST;
        let standard_error = (self.variance / effective_n).sqrt();
        if is_not_above(standard_error, 0.0) {
            // Degenerate reference: the baseline error had no spread at all.
            // Under that null the error is a point mass, so *any* rise above it
            // has probability zero and any other outcome is the null itself.
            // Returning the mid-value `0.5` here (what `sf(0)` gives) would
            // silently veto every verdict on such a stream, which is the
            // opposite of what a zero-variance reference implies.
            return Ok(if fast > slow { 0.0 } else { 1.0 });
        }
        stats::standard_normal_sf((fast - slow) / standard_error)
    }

    /// Whether enough observations have accumulated for a verdict.
    fn ready(&self) -> bool {
        self.updates >= WARMUP_OBSERVATIONS
    }

    /// Whether a sustained *and* statistically significant degradation is
    /// currently in force.
    fn drift_detected(&self) -> Result<bool, String> {
        if !(self.ready() && self.run_length >= MIN_DRIFT_RUN) {
            return Ok(false);
        }
        Ok(self.p_value()? < DRIFT_SIGNIFICANCE)
    }

    /// Length of the current run of observations above the threshold.
    #[cfg(test)]
    fn run_length(&self) -> usize {
        self.run_length
    }

    /// Observations folded in so far.
    fn updates(&self) -> usize {
        self.updates
    }

    fn reset(&mut self) {
        self.fast = f64::NAN;
        self.slow = f64::NAN;
        self.variance = 0.0;
        self.updates = 0;
        self.run_length = 0;
    }
}

// ---------------------------------------------------------------------------
// neural network
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* generator, used only to break the symmetry of the
/// initial hidden layer.
///
/// An all-zero (or all-equal) initialisation is degenerate for a fully
/// connected layer: every hidden unit receives the same gradient forever and
/// the network can never represent more than one unit's worth of function. A
/// fixed-seed generator breaks that symmetry while keeping the detector
/// reproducible.
#[derive(Debug, Clone)]
struct SymmetryBreaker {
    state: u64,
}

impl SymmetryBreaker {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { MLP_INIT_SEED } else { seed },
        }
    }

    /// Next value, uniform on `[-1, 1)`.
    fn next_signed_unit(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let scrambled = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let unit = (scrambled >> 11) as f64 / ((1u64 << 53) as f64);
        unit * 2.0 - 1.0
    }
}

/// Online multilayer perceptron drift detector.
///
/// One hidden layer of `MLP_HIDDEN_UNITS` `tanh` units over the feature
/// window, a linear output, and a stochastic-gradient step on the squared
/// prediction error taken once per labelled point.
///
/// The step is *normalised* by the instantaneous input energy (`1 + ||x||^2`
/// for the hidden layer, `1 + ||a||^2` for the output layer), exactly as the
/// linear detector's normalised-LMS update is. Streaming features are not
/// standardised and their scale is unknown a priori, so a raw step size that is
/// stable for one stream diverges on another; normalising makes the update
/// scale-invariant and stable for any `0 < learning_rate < 2`.
#[derive(Debug, Clone)]
pub struct NeuralNetworkDriftDetector<A: Float + Send + Sync> {
    /// Hidden-layer weights, `[hidden][input]`.
    hidden_weights: Vec<Vec<f64>>,
    /// Hidden-layer biases.
    hidden_bias: Vec<f64>,
    /// Output-layer weights, one per hidden unit.
    output_weights: Vec<f64>,
    /// Output-layer bias.
    output_bias: f64,
    /// Number of input features the network currently spans.
    input_width: usize,
    /// Step size of the normalised SGD update.
    learning_rate: f64,
    /// L2 weight-decay coefficient.
    l2_lambda: f64,
    /// Deterministic initialiser for newly grown weights.
    initialiser: SymmetryBreaker,
    /// Linearised input sensitivity captured when the baseline was taken.
    baseline_sensitivity: Vec<f64>,
    /// Error tracking and the drift rule.
    tracker: PrequentialErrorTracker,
    _marker: PhantomData<A>,
}

impl<A: Float + Send + Sync> NeuralNetworkDriftDetector<A> {
    /// Creates a detector. `sensitivity` is the relative rise in mean squared
    /// error that counts as drift.
    pub fn new(sensitivity: f64) -> Result<Self, String> {
        Ok(Self {
            hidden_weights: vec![Vec::new(); MLP_HIDDEN_UNITS],
            hidden_bias: vec![0.0; MLP_HIDDEN_UNITS],
            output_weights: Vec::new(),
            output_bias: 0.0,
            input_width: 0,
            // Mid-range normalised step: stable for any input scale.
            learning_rate: 0.5,
            l2_lambda: 1e-5,
            initialiser: SymmetryBreaker::new(MLP_INIT_SEED),
            baseline_sensitivity: Vec::new(),
            tracker: PrequentialErrorTracker::new(sensitivity)?,
            _marker: PhantomData,
        })
    }

    /// Number of input features the network currently spans.
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// Current fast mean of the squared prediction error, or `None` before the
    /// first supervised update.
    pub fn current_error(&self) -> Option<f64> {
        self.tracker.fast.is_finite().then_some(self.tracker.fast)
    }

    /// Grows the network to span `width` inputs, initialising the new columns
    /// with the deterministic symmetry breaker scaled by the Glorot factor.
    fn grow_to(&mut self, width: usize) {
        if width <= self.input_width {
            return;
        }
        if self.output_weights.is_empty() {
            // Glorot uniform for the output layer: fan_in = hidden units,
            // fan_out = 1.
            let scale = (6.0 / (MLP_HIDDEN_UNITS as f64 + 1.0)).sqrt();
            self.output_weights = (0..MLP_HIDDEN_UNITS)
                .map(|_| self.initialiser.next_signed_unit() * scale)
                .collect();
        }
        let scale = (6.0 / (width as f64 + MLP_HIDDEN_UNITS as f64)).sqrt();
        for row in &mut self.hidden_weights {
            while row.len() < width {
                row.push(self.initialiser.next_signed_unit() * scale);
            }
        }
        self.input_width = width;
    }

    /// Hidden activations for one input vector.
    fn activate(&self, features: &[f64]) -> Vec<f64> {
        self.hidden_weights
            .iter()
            .zip(self.hidden_bias.iter())
            .map(|(row, bias)| {
                let mut sum = *bias;
                for (weight, value) in row.iter().zip(features.iter()) {
                    sum += weight * value;
                }
                sum.tanh()
            })
            .collect()
    }

    fn predict_from(&self, activations: &[f64]) -> f64 {
        let mut prediction = self.output_bias;
        for (weight, activation) in self.output_weights.iter().zip(activations.iter()) {
            prediction += weight * activation;
        }
        prediction
    }

    /// Linearised sensitivity of the output to each input feature,
    /// `sum_h |w2_h * w1[h][i]|`. This is a real property of the trained
    /// network (the magnitude of the first-order term at the operating point
    /// `tanh'(0) = 1`), so its change since the baseline is a measured
    /// feature-importance shift rather than an invented score.
    fn input_sensitivity(&self) -> Vec<f64> {
        (0..self.input_width)
            .map(|index| {
                self.hidden_weights
                    .iter()
                    .zip(self.output_weights.iter())
                    .map(|(row, output_weight)| {
                        row.get(index)
                            .map(|weight| (weight * output_weight).abs())
                            .unwrap_or(0.0)
                    })
                    .sum()
            })
            .collect()
    }

    /// Restarts the error statistics after a drift has been reported, leaving
    /// the learned network in place.
    ///
    /// This is DDM's published post-detection restart, and the same restart
    /// `DdmTest` and `PageHinkleyTest` perform in this crate: once the verdict
    /// has been issued, the reference the verdict was measured against belongs
    /// to the old concept and must not be carried into the new one. Without it
    /// a baseline that (correctly) refused to absorb the drift would keep the
    /// detector alarmed for as long as it took the fast mean to decay — several
    /// thousand observations on a stream whose reference error was exactly
    /// zero. The importance baseline is cleared with it, so the next concept is
    /// compared against its own starting point rather than the previous one's.
    ///
    /// The cost is [`WARMUP_OBSERVATIONS`]: the detector issues no further
    /// verdict until the restarted statistics have settled, so a second drift
    /// arriving inside that window is not reported. Both halves of that
    /// constant's justification survive a restart and neither can be shortened
    /// here. The slow mean still needs its two time constants to settle (after
    /// 100 observations 37% of its weight is still the single squared error it
    /// was re-seeded from, after 200 it is 13%), and the model is *not* already
    /// trained: a concept change puts it into a fresh learning transient, whose
    /// legitimately elevated and noisy error is exactly what a shortened
    /// warm-up would start testing against a half-formed baseline.
    fn restart_after_detection(&mut self) {
        self.tracker.reset();
        self.baseline_sensitivity.clear();
    }

    /// Scores the point, folds the observed error into the tracker, then takes
    /// one gradient step (prequential order).
    fn learn_one(&mut self, features: &[f64], target: f64) {
        self.grow_to(features.len());

        let activations = self.activate(features);
        let prediction = self.predict_from(&activations);
        let error = prediction - target;
        self.tracker.observe(error * error);

        if self.tracker.updates() == WARMUP_OBSERVATIONS || self.baseline_sensitivity.is_empty() {
            self.baseline_sensitivity = self.input_sensitivity();
        }

        // Backpropagation. The hidden deltas use the output weights *before*
        // they are updated, which is what the chain rule prescribes.
        let hidden_deltas: Vec<f64> = activations
            .iter()
            .zip(self.output_weights.iter())
            .map(|(activation, output_weight)| {
                error * output_weight * (1.0 - activation * activation)
            })
            .collect();

        let output_energy = 1.0 + activations.iter().map(|a| a * a).sum::<f64>();
        let output_step = self.learning_rate * error / output_energy;
        for (weight, activation) in self.output_weights.iter_mut().zip(activations.iter()) {
            *weight -= output_step * activation + self.l2_lambda * *weight;
        }
        self.output_bias -= output_step;

        let input_energy = 1.0 + features.iter().map(|f| f * f).sum::<f64>();
        let hidden_step = self.learning_rate / input_energy;
        for ((row, bias), delta) in self
            .hidden_weights
            .iter_mut()
            .zip(self.hidden_bias.iter_mut())
            .zip(hidden_deltas.iter())
        {
            for (weight, value) in row.iter_mut().zip(features.iter()) {
                *weight -= hidden_step * delta * value + self.l2_lambda * *weight;
            }
            *bias -= hidden_step * delta;
        }
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> ModelBasedDetector<A>
    for NeuralNetworkDriftDetector<A>
{
    fn update_model(&mut self, data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut trained = 0usize;
        for data_point in data {
            if let Some((features, target)) = supervised_pair(data_point) {
                self.learn_one(&features, target);
                trained += 1;
            }
        }
        if trained == 0 && !data.is_empty() {
            return Err(
                "neural-network drift detector requires labelled data points (target is None)"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn detect_drift(
        &mut self,
        data: &[StreamingDataPoint<A>],
    ) -> Result<ModelDriftResult<A>, String> {
        self.update_model(data)?;

        let degradation = self
            .tracker
            .degradation()
            .ok_or_else(|| "neural-network drift detector has no error estimate yet".to_string())?;
        let p_value = self.tracker.p_value()?;

        let sensitivity = self.input_sensitivity();
        let mut feature_importance_changes = Vec::with_capacity(sensitivity.len());
        for (index, current) in sensitivity.iter().enumerate() {
            let baseline = self.baseline_sensitivity.get(index).copied().unwrap_or(0.0);
            feature_importance_changes.push(from_f64::<A>(current - baseline)?);
        }

        let drift_detected = self.tracker.drift_detected()?;
        let result = ModelDriftResult {
            drift_detected,
            performance_degradation: from_f64(degradation)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            feature_importance_changes,
        };
        if drift_detected {
            self.restart_after_detection();
        }
        Ok(result)
    }

    fn reset_model(&mut self) -> Result<(), String> {
        self.hidden_weights = vec![Vec::new(); MLP_HIDDEN_UNITS];
        self.hidden_bias = vec![0.0; MLP_HIDDEN_UNITS];
        self.output_weights.clear();
        self.output_bias = 0.0;
        self.input_width = 0;
        self.initialiser = SymmetryBreaker::new(MLP_INIT_SEED);
        self.baseline_sensitivity.clear();
        self.tracker.reset();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// decision tree
// ---------------------------------------------------------------------------

/// A node of the fitted regression tree.
#[derive(Debug, Clone)]
enum TreeNode {
    /// Terminal node holding the mean target of the observations that reached
    /// it.
    Leaf { value: f64 },
    /// Internal node testing `features[feature] <= threshold`.
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, features: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                // A row that does not carry this feature cannot be routed by
                // it; it follows the left branch, which is the branch the
                // fitting code puts the low side of the split on.
                let value = features.get(*feature).copied().unwrap_or(f64::NEG_INFINITY);
                if value <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

/// Depth-limited CART regression tree over a sliding window of labelled
/// observations.
///
/// The tree is refit from the window every `TREE_REFIT_INTERVAL`
/// observations rather than grown incrementally — a periodically refit CART,
/// not a Hoeffding tree. Each split is the `(feature, threshold)` pair that
/// maximally *diverges* the two children's squared-error rates from the
/// parent's, i.e. that maximises `SSE(parent) - SSE(left) - SSE(right)`, which
/// is CART's variance-reduction criterion.
#[derive(Debug, Clone)]
pub struct DecisionTreeDriftDetector<A: Float + Send + Sync> {
    /// Sliding window of labelled observations the tree is fit from.
    window: VecDeque<(Vec<f64>, f64)>,
    /// Fitted tree, absent until the window has enough observations.
    tree: Option<TreeNode>,
    /// Total squared-error reduction attributed to each feature by the current
    /// tree — CART's own feature-importance measure.
    importances: Vec<f64>,
    /// Importance snapshot taken when the baseline was established.
    baseline_importances: Vec<f64>,
    /// Observations since the last refit.
    since_refit: usize,
    /// Error tracking and the drift rule.
    tracker: PrequentialErrorTracker,
    _marker: PhantomData<A>,
}

impl<A: Float + Send + Sync> DecisionTreeDriftDetector<A> {
    /// Creates a detector. `sensitivity` is the relative rise in mean squared
    /// error that counts as drift.
    pub fn new(sensitivity: f64) -> Result<Self, String> {
        Ok(Self {
            window: VecDeque::with_capacity(TREE_WINDOW_CAPACITY),
            tree: None,
            importances: Vec::new(),
            baseline_importances: Vec::new(),
            since_refit: 0,
            tracker: PrequentialErrorTracker::new(sensitivity)?,
            _marker: PhantomData,
        })
    }

    /// Whether a tree has been fit yet.
    pub fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }

    /// Per-feature squared-error reduction attributed by the current tree.
    pub fn feature_importances(&self) -> &[f64] {
        &self.importances
    }

    /// Sum and sum-of-squares of the targets of a subset.
    fn subset_moments(&self, indices: &[usize]) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        for index in indices {
            if let Some((_, target)) = self.window.get(*index) {
                sum += *target;
                sum_squares += target * target;
            }
        }
        (sum, sum_squares)
    }

    fn subset_sse(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        let (sum, sum_squares) = self.subset_moments(indices);
        (sum_squares - sum * sum / indices.len() as f64).max(0.0)
    }

    fn feature_value(&self, index: usize, feature: usize) -> Option<f64> {
        self.window
            .get(index)
            .and_then(|(features, _)| features.get(feature).copied())
    }

    /// Best `(feature, threshold, gain)` split of a subset, or `None` when no
    /// admissible split reduces the squared error.
    fn best_split(&self, indices: &[usize], width: usize) -> Option<(usize, f64, f64)> {
        let parent_sse = self.subset_sse(indices);
        if is_not_above(parent_sse, 0.0) {
            return None;
        }
        let (total_sum, total_squares) = self.subset_moments(indices);
        let count = indices.len();

        let mut best: Option<(usize, f64, f64)> = None;
        let mut order: Vec<usize> = Vec::with_capacity(count);
        for feature in 0..width {
            order.clear();
            order.extend_from_slice(indices);
            order.sort_by(|left, right| {
                let a = self.feature_value(*left, feature).unwrap_or(0.0);
                let b = self.feature_value(*right, feature).unwrap_or(0.0);
                a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut left_sum = 0.0;
            let mut left_squares = 0.0;
            for position in 0..count.saturating_sub(1) {
                let Some((_, target)) = self.window.get(order[position]) else {
                    continue;
                };
                left_sum += *target;
                left_squares += target * target;

                let current = self.feature_value(order[position], feature).unwrap_or(0.0);
                let next = self
                    .feature_value(order[position + 1], feature)
                    .unwrap_or(0.0);
                if is_not_above(next, current) {
                    // Identical feature values cannot be separated.
                    continue;
                }
                let left_count = position + 1;
                let right_count = count - left_count;
                if left_count < TREE_MIN_SAMPLES_LEAF || right_count < TREE_MIN_SAMPLES_LEAF {
                    continue;
                }
                let left_sse = (left_squares - left_sum * left_sum / left_count as f64).max(0.0);
                let right_sum = total_sum - left_sum;
                let right_sse = ((total_squares - left_squares)
                    - right_sum * right_sum / right_count as f64)
                    .max(0.0);
                let gain = parent_sse - left_sse - right_sse;
                if gain.is_finite() && gain > 0.0 && best.is_none_or(|(_, _, top)| gain > top) {
                    best = Some((feature, (current + next) / 2.0, gain));
                }
            }
        }
        best
    }

    fn grow(
        &self,
        indices: &[usize],
        width: usize,
        depth: usize,
        importances: &mut [f64],
    ) -> TreeNode {
        let leaf = || {
            let (sum, _) = self.subset_moments(indices);
            TreeNode::Leaf {
                value: if indices.is_empty() {
                    0.0
                } else {
                    sum / indices.len() as f64
                },
            }
        };
        if depth >= TREE_MAX_DEPTH || indices.len() < 2 * TREE_MIN_SAMPLES_LEAF {
            return leaf();
        }
        let Some((feature, threshold, gain)) = self.best_split(indices, width) else {
            return leaf();
        };
        if let Some(slot) = importances.get_mut(feature) {
            *slot += gain;
        }

        let mut left = Vec::new();
        let mut right = Vec::new();
        for index in indices {
            let value = self
                .feature_value(*index, feature)
                .unwrap_or(f64::NEG_INFINITY);
            if value <= threshold {
                left.push(*index);
            } else {
                right.push(*index);
            }
        }
        if left.is_empty() || right.is_empty() {
            return leaf();
        }

        TreeNode::Split {
            feature,
            threshold,
            left: Box::new(self.grow(&left, width, depth + 1, importances)),
            right: Box::new(self.grow(&right, width, depth + 1, importances)),
        }
    }

    /// Refits the tree from the current window.
    ///
    /// Only the features every retained row actually carries are considered: a
    /// row with a shorter feature vector has no value for the wider columns,
    /// and substituting a zero would invent an observation.
    fn refit(&mut self) {
        let width = self
            .window
            .iter()
            .map(|(features, _)| features.len())
            .min()
            .unwrap_or(0);
        if width == 0 || self.window.len() < 2 * TREE_MIN_SAMPLES_LEAF {
            return;
        }
        let indices: Vec<usize> = (0..self.window.len()).collect();
        let mut importances = vec![0.0; width];
        let tree = self.grow(&indices, width, 0, &mut importances);
        self.tree = Some(tree);
        self.importances = importances;
        if self.baseline_importances.is_empty() {
            self.baseline_importances = self.importances.clone();
        }
    }

    /// Restarts the error statistics after a drift has been reported, leaving
    /// the fitted tree and its window in place.
    ///
    /// See [`NeuralNetworkDriftDetector::restart_after_detection`] for why the
    /// reference cannot outlive the concept it was measured on, and for the
    /// [`WARMUP_OBSERVATIONS`] blind window the restart costs. The window is
    /// deliberately *not* cleared: the tree is the model, and discarding the
    /// model on every verdict would make the next verdict meaningless.
    fn restart_after_detection(&mut self) {
        self.tracker.reset();
        self.baseline_importances.clear();
    }

    /// Scores the point with the current tree, folds the error in, then adds
    /// it to the window (prequential order) and refits on schedule.
    fn learn_one(&mut self, features: Vec<f64>, target: f64) {
        if let Some(tree) = &self.tree {
            let error = tree.predict(&features) - target;
            self.tracker.observe(error * error);
            if self.tracker.updates() == WARMUP_OBSERVATIONS
                && !self.importances.is_empty()
                && self.baseline_importances.is_empty()
            {
                self.baseline_importances = self.importances.clone();
            }
        }

        if self.window.len() >= TREE_WINDOW_CAPACITY {
            self.window.pop_front();
        }
        self.window.push_back((features, target));
        self.since_refit += 1;

        let due = self.tree.is_none() || self.since_refit >= TREE_REFIT_INTERVAL;
        if due && self.window.len() >= 2 * TREE_MIN_SAMPLES_LEAF {
            self.since_refit = 0;
            self.refit();
        }
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> ModelBasedDetector<A>
    for DecisionTreeDriftDetector<A>
{
    fn update_model(&mut self, data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        let mut trained = 0usize;
        for data_point in data {
            if let Some((features, target)) = supervised_pair(data_point) {
                self.learn_one(features, target);
                trained += 1;
            }
        }
        if trained == 0 && !data.is_empty() {
            return Err(
                "decision-tree drift detector requires labelled data points (target is None)"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn detect_drift(
        &mut self,
        data: &[StreamingDataPoint<A>],
    ) -> Result<ModelDriftResult<A>, String> {
        self.update_model(data)?;

        let degradation = self.tracker.degradation().ok_or_else(|| {
            "decision-tree drift detector has not scored any point yet (the tree is still \
             being fit from its first window)"
                .to_string()
        })?;
        let p_value = self.tracker.p_value()?;

        let mut feature_importance_changes = Vec::with_capacity(self.importances.len());
        for (index, current) in self.importances.iter().enumerate() {
            let baseline = self.baseline_importances.get(index).copied().unwrap_or(0.0);
            feature_importance_changes.push(from_f64::<A>(current - baseline)?);
        }

        let drift_detected = self.tracker.drift_detected()?;
        let result = ModelDriftResult {
            drift_detected,
            performance_degradation: from_f64(degradation)?,
            confidence: from_f64((1.0 - p_value).clamp(0.0, 1.0))?,
            feature_importance_changes,
        };
        if drift_detected {
            self.restart_after_detection();
        }
        Ok(result)
    }

    fn reset_model(&mut self) -> Result<(), String> {
        self.window.clear();
        self.tree = None;
        self.importances.clear();
        self.baseline_importances.clear();
        self.since_refit = 0;
        self.tracker.reset();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ensemble
// ---------------------------------------------------------------------------

/// Ensemble of model-based drift detectors.
///
/// The ensemble owns **its own** freshly constructed linear, neural-network and
/// decision-tree members rather than borrowing the instances registered in
/// `EnhancedDriftDetector::model_detectors`: those are keyed by `ModelType` in
/// the same map this detector is stored in, so they cannot be aliased, and
/// sharing them would also mean the ensemble's verdict depended on how often
/// the other model types happened to be selected.
///
/// The verdict is a weighted majority of the members' own verdicts. Weights
/// default to uniform, which is exactly a plain majority vote; a caller who has
/// measured the members' relative reliability can supply their own through
/// [`EnsembleDriftDetector::with_weights`].
///
/// The reported significance is Fisher's combination of the members' p-values,
/// `-2 * sum(ln p_i) ~ chi^2(2k)`. Averaging the members' confidences would not
/// be a p-value at all, and `EnhancedDriftDetector` recovers the p-value from
/// `1 - confidence`, so it has to be a real one.
pub struct EnsembleDriftDetector<A: Float + Send + Sync> {
    members: Vec<Box<dyn ModelBasedDetector<A>>>,
    names: Vec<&'static str>,
    weights: Vec<f64>,
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum + 'static> EnsembleDriftDetector<A> {
    /// Creates an ensemble with uniform member weights (a plain majority vote).
    pub fn new(sensitivity: f64) -> Result<Self, String> {
        Self::with_weights(sensitivity, &[1.0, 1.0, 1.0])
    }

    /// Creates an ensemble with explicit member weights, in the order
    /// `[linear, neural_network, decision_tree]`.
    pub fn with_weights(sensitivity: f64, weights: &[f64]) -> Result<Self, String> {
        let members: Vec<Box<dyn ModelBasedDetector<A>>> = vec![
            Box::new(LinearModelDetector::<A>::new(sensitivity)?),
            Box::new(NeuralNetworkDriftDetector::<A>::new(sensitivity)?),
            Box::new(DecisionTreeDriftDetector::<A>::new(sensitivity)?),
        ];
        let names = vec!["linear", "neural_network", "decision_tree"];
        if weights.len() != members.len() {
            return Err(format!(
                "ensemble drift detector has {} members but {} weights were supplied",
                members.len(),
                weights.len()
            ));
        }
        if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
            return Err("ensemble member weights must be finite and non-negative".to_string());
        }
        if weights.iter().sum::<f64>() <= 0.0 {
            return Err("ensemble member weights must not sum to zero".to_string());
        }
        Ok(Self {
            members,
            names,
            weights: weights.to_vec(),
        })
    }

    /// Names of the members, in vote order.
    pub fn member_names(&self) -> &[&'static str] {
        &self.names
    }
}

impl<A: Float + Default + Clone + Send + Sync + std::iter::Sum> ModelBasedDetector<A>
    for EnsembleDriftDetector<A>
{
    fn update_model(&mut self, data: &[StreamingDataPoint<A>]) -> Result<(), String> {
        for member in &mut self.members {
            member.update_model(data)?;
        }
        Ok(())
    }

    fn detect_drift(
        &mut self,
        data: &[StreamingDataPoint<A>],
    ) -> Result<ModelDriftResult<A>, String> {
        let mut results = Vec::with_capacity(self.members.len());
        for (member, name) in self.members.iter_mut().zip(self.names.iter()) {
            let result = member
                .detect_drift(data)
                .map_err(|error| format!("ensemble member `{name}` failed: {error}"))?;
            results.push(result);
        }

        let total_weight: f64 = self.weights.iter().sum();
        let mut votes = 0.0;
        let mut degradation_sum = 0.0;
        let mut log_p_sum = 0.0;
        let mut widest = 0usize;
        for (result, weight) in results.iter().zip(self.weights.iter()) {
            if result.drift_detected {
                votes += *weight;
            }
            degradation_sum += result
                .performance_degradation
                .to_f64()
                .ok_or_else(|| "member degradation is not representable as f64".to_string())?
                * *weight;
            let confidence = result
                .confidence
                .to_f64()
                .ok_or_else(|| "member confidence is not representable as f64".to_string())?;
            // Fisher's method needs a strictly positive p; a member reporting
            // an exact zero is clamped to the smallest positive double rather
            // than making the combined statistic infinite.
            let p = (1.0 - confidence).clamp(f64::MIN_POSITIVE, 1.0);
            log_p_sum += p.ln();
            widest = widest.max(result.feature_importance_changes.len());
        }

        let combined_p = stats::chi_square_sf(-2.0 * log_p_sum, 2.0 * results.len() as f64)?;
        let degradation = degradation_sum / total_weight;

        // Element-wise weighted mean of the members' feature-importance
        // changes. The members disagree about how many features they track, so
        // each index averages only over the members that actually report it.
        let mut feature_importance_changes = Vec::with_capacity(widest);
        for index in 0..widest {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for (result, weight) in results.iter().zip(self.weights.iter()) {
                if let Some(change) = result.feature_importance_changes.get(index) {
                    let value = change
                        .to_f64()
                        .ok_or_else(|| "member importance is not representable".to_string())?;
                    sum += value * *weight;
                    weight_sum += *weight;
                }
            }
            let mean = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
            feature_importance_changes.push(from_f64::<A>(mean)?);
        }

        Ok(ModelDriftResult {
            drift_detected: votes > total_weight / 2.0,
            performance_degradation: from_f64(degradation)?,
            confidence: from_f64((1.0 - combined_p).clamp(0.0, 1.0))?,
            feature_importance_changes,
        })
    }

    fn reset_model(&mut self) -> Result<(), String> {
        for member in &mut self.members {
            member.reset_model()?;
        }
        Ok(())
    }
}

impl<A: Float + Send + Sync> std::fmt::Debug for EnsembleDriftDetector<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EnsembleDriftDetector")
            .field("members", &self.names)
            .field("weights", &self.weights)
            .finish()
    }
}

#[cfg(test)]
#[path = "drift_models_tests.rs"]
mod drift_models_tests;
