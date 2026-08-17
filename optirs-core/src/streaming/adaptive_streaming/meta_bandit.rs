// Contextual-bandit machinery for the streaming meta-learner.
//
// The meta-learner chooses which adaptation to recommend with a linear
// contextual bandit: a fixed table of `(adaptation type, magnitude)` arms, each
// carrying its own online linear reward model over the standardised state
// feature vector. This module holds the arms, the feature standardiser and the
// state/action encoding; `meta_learning.rs` holds the learner that drives them.
//
// Split out of `meta_learning.rs` to keep that file under the 2000-line limit.

use super::meta_learning::{MetaAction, MetaState};
use super::optimizer::AdaptationType;

use scirs2_core::numeric::Float;

/// One arm of the meta-learner's contextual bandit.
///
/// An arm is a concrete `(adaptation type, magnitude)` pair. Its expected
/// reward for a given state is modelled linearly in the state features, and the
/// weights are fitted online by stochastic gradient descent on the squared
/// prediction error against the rewards actually observed.
#[derive(Debug, Clone)]
pub(super) struct BanditArm<A: Float + Send + Sync> {
    /// Human-readable identifier.
    pub(super) label: String,
    /// Adaptation this arm applies.
    pub(super) adaptation_type: AdaptationType,
    /// Magnitude this arm applies, in `f64` so the arm table is representable
    /// independently of the element type.
    pub(super) magnitude_f64: f64,
    /// Linear reward-model weights, grown lazily to the state width.
    pub(super) weights: Vec<A>,
    /// Intercept.
    pub(super) bias: A,
    /// Number of training updates applied to this arm.
    pub(super) pulls: usize,
    /// Sum of the rewards observed for this arm.
    pub(super) reward_total: A,
}

impl<A: Float + Send + Sync> BanditArm<A> {
    pub(super) fn new(label: String, adaptation_type: AdaptationType, magnitude_f64: f64) -> Self {
        Self {
            label,
            adaptation_type,
            magnitude_f64,
            weights: Vec::new(),
            bias: A::zero(),
            pulls: 0,
            reward_total: A::zero(),
        }
    }

    pub(super) fn magnitude(&self) -> Option<A> {
        A::from(self.magnitude_f64)
    }

    /// Predicted reward for a state feature vector.
    pub(super) fn predict(&self, features: &[A]) -> A {
        let mut value = self.bias;
        for (weight, &feature) in self.weights.iter().zip(features.iter()) {
            value = value + *weight * feature;
        }
        value
    }

    /// One normalised-LMS step on the squared reward-prediction error.
    ///
    /// The state feature vector mixes wildly different scales — a loss of `0.01`
    /// alongside a memory figure in the hundreds of megabytes — so an
    /// unnormalised gradient step diverges: with a feature of `512` and a step
    /// size of `0.01`, a single update moves the weight by `5.12 * error`.
    /// Dividing by the instantaneous input energy `1 + ||x||^2` (the `1`
    /// accounting for the bias coordinate) makes the step scale-invariant and
    /// stable for any `0 < learning_rate < 2`, which is what lets the bandit
    /// actually learn from a raw, un-standardised state vector. The ridge
    /// penalty is applied as a separate weight decay so it does not interact
    /// with the normalisation.
    pub(super) fn sgd_step(&mut self, features: &[A], error: A, learning_rate: A, l2_lambda: A) {
        if self.weights.len() < features.len() {
            self.weights.resize(features.len(), A::zero());
        }

        let input_energy = features
            .iter()
            .fold(A::one(), |acc, &feature| acc + feature * feature);
        if input_energy <= A::zero() || !input_energy.is_finite() {
            return;
        }
        let step = learning_rate * error / input_energy;

        for (weight, &feature) in self.weights.iter_mut().zip(features.iter()) {
            *weight = *weight - step * feature - l2_lambda * *weight;
        }
        self.bias = self.bias - step;
    }

    /// Records an observed reward for this arm.
    pub(super) fn observe(&mut self, reward: A) {
        self.pulls += 1;
        self.reward_total = self.reward_total + reward;
    }
}

/// Running per-coordinate mean and variance used to standardise the state
/// feature vector before it reaches the arms' linear reward models.
///
/// A raw meta-state mixes a loss of `0.01` with a memory figure in the hundreds
/// of megabytes. Normalising the *step* (see [`BanditArm::sgd_step`]) keeps that
/// from diverging, but it does not make the small coordinates learnable: the
/// instantaneous input energy is dominated by the large ones, so essentially the
/// whole gradient lands on coordinates that may carry no information at all.
/// Standardising each coordinate to zero mean and unit variance puts them on a
/// common footing, and collapses a *constant* coordinate to exactly zero so it
/// cannot absorb gradient it has no right to.
#[derive(Debug, Clone, Default)]
pub(super) struct FeatureScaler<A: Float + Send + Sync> {
    /// Running per-coordinate mean.
    pub(super) means: Vec<A>,
    /// Running per-coordinate sum of squared deviations.
    pub(super) m2: Vec<A>,
    /// Number of observations folded in.
    pub(super) count: usize,
}

impl<A: Float + Send + Sync> FeatureScaler<A> {
    /// Folds one feature vector into the running statistics (Welford).
    pub(super) fn observe(&mut self, features: &[A]) {
        if self.means.len() < features.len() {
            self.means.resize(features.len(), A::zero());
            self.m2.resize(features.len(), A::zero());
        }
        self.count = self.count.saturating_add(1);
        let Some(count) = A::from(self.count) else {
            return;
        };
        for (index, &value) in features.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let mean = self.means[index];
            let delta = value - mean;
            let new_mean = mean + delta / count;
            self.means[index] = new_mean;
            self.m2[index] = self.m2[index] + delta * (value - new_mean);
        }
    }

    /// Standardises a feature vector against the running statistics.
    ///
    /// With fewer than two observations there is no scale to standardise
    /// against, so the raw vector is returned; the arms' weights are still all
    /// zero at that point, so nothing depends on the choice.
    pub(super) fn standardize(&self, features: &[A]) -> Vec<A> {
        if self.count < 2 {
            return features.to_vec();
        }
        let Some(count) = A::from(self.count) else {
            return features.to_vec();
        };
        features
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let (Some(&mean), Some(&m2)) = (self.means.get(index), self.m2.get(index)) else {
                    return A::zero();
                };
                let variance = m2 / count;
                if variance <= A::zero() || !variance.is_finite() {
                    // Constant (or unusable) coordinate: carries no information.
                    return A::zero();
                }
                let standardized = (value - mean) / variance.sqrt();
                if standardized.is_finite() {
                    standardized
                } else {
                    A::zero()
                }
            })
            .collect()
    }
}

/// Flattens a meta-state into the feature vector the bandit is linear in.
///
/// The layout is fixed (performance metrics, then resource state, then drift
/// indicators, then a normalised adaptation-history length), so a weight index
/// always refers to the same quantity. The raw vector is standardised by
/// [`FeatureScaler`] before it reaches an arm.
pub(super) fn state_features<A: Float + Send + Sync>(state: &MetaState<A>) -> Vec<A> {
    let mut features = Vec::with_capacity(
        state.performance_metrics.len()
            + state.resource_state.len()
            + state.drift_indicators.len()
            + 1,
    );
    features.extend(state.performance_metrics.iter().copied());
    features.extend(state.resource_state.iter().copied());
    features.extend(state.drift_indicators.iter().copied());
    // Log-scaled history length keeps this feature on the same order of
    // magnitude as the others no matter how long the run has been going.
    let history = (state.adaptation_history as f64 + 1.0).ln();
    if let Some(value) = A::from(history) {
        features.push(value);
    }
    features
}

/// Recovers which arm an experience's action corresponds to.
pub(super) fn arm_index_for<A: Float + Send + Sync>(action: &MetaAction<A>) -> Option<usize> {
    let adaptation_type = action.adaptation_types.first()?;
    let magnitude = action.adaptation_magnitudes.first()?.to_f64()?;
    let table = arm_table();
    // Nearest arm by magnitude within the matching adaptation type.
    let mut best: Option<(usize, f64)> = None;
    for (index, (_, arm_type, arm_magnitude)) in table.iter().enumerate() {
        if !same_adaptation_type(arm_type, adaptation_type) {
            continue;
        }
        let distance = (arm_magnitude - magnitude).abs();
        if best.map(|(_, d)| distance < d).unwrap_or(true) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

/// Compares adaptation types without requiring `PartialEq` on the payload.
fn same_adaptation_type(left: &AdaptationType, right: &AdaptationType) -> bool {
    left == right
}

/// The fixed arm table: which adaptations the meta-learner can recommend and at
/// what magnitudes.
pub(super) fn arm_table() -> Vec<(&'static str, AdaptationType, f64)> {
    let mut table = Vec::with_capacity(10);
    for magnitude in [-0.2_f64, -0.1, -0.05, 0.05, 0.1, 0.2] {
        table.push(("learning_rate", AdaptationType::LearningRate, magnitude));
    }
    for magnitude in [-0.3_f64, -0.1, 0.1, 0.3] {
        table.push(("buffer_size", AdaptationType::BufferSize, magnitude));
    }
    table
}
