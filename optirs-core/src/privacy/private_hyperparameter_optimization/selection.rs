//! Differentially private selection of hyperparameter configurations.
//!
//! # The defect this replaces
//!
//! `HyperparameterNoiseMechanism` was stored on `PrivateHPOConfig` and never
//! matched on anywhere in the crate: no exponential mechanism, no
//! report-noisy-max, no noise on the choice at any point.
//! `PrivateResultsAggregator::aggregate_results` sorted the evaluations exactly
//! and returned the exact top five, and `optimize()` tracked the exact argmax.
//!
//! Private hyperparameter optimization is *entirely* about privatising the
//! selection step (Liu & Talwar, STOC 2019; Chaudhuri, Monteleoni & Sarwate,
//! JMLR 2011), so an exact argmax over utilities computed from private data
//! leaks the selection and provides no guarantee for the chosen configuration.
//!
//! # What is implemented
//!
//! * The **exponential mechanism** (McSherry & Talwar, FOCS 2007): index `i` is
//!   returned with probability proportional to
//!   `exp(epsilon * u_i / (2 * Delta_u))`. The weighting itself is delegated to
//!   the crate's audited
//!   [`crate::privacy::noise_mechanisms::ExponentialMechanism`] rather than
//!   reimplemented here.
//! * **Report-noisy-max** with Gumbel noise, which is *equivalent in
//!   distribution* to the exponential mechanism -- asserted by a test in this
//!   module -- and with Laplace noise at scale `2 Delta_u / epsilon`.
//! * A **Gaussian** argmax at scale
//!   `sqrt(2 ln(1.25/delta)) * 2 Delta_u / epsilon`, which requires a delta and
//!   errors when none is configured.
//! * `SparseVector` selection is refused: the sparse vector technique answers a
//!   *stream* of threshold queries and is not a one-shot selection primitive.
//!   Use [`crate::privacy::noise_mechanisms::SparseVectorMechanism`].
//!
//! Every selection reports the epsilon it consumed so the caller can charge it.

use crate::error::{OptimError, Result};
use crate::privacy::noise_mechanisms::ExponentialMechanism as ValueExponentialMechanism;
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::fmt::Debug;

use super::types::{
    os_seeded_hpo_rng, HpoRng, HyperparameterNoiseMechanism, SelectionMechanism,
    SelectionParameters, SensitivityBounds, SummaryStatistics, UtilityFunction,
    UtilityFunctionType,
};

/// Key under which the objective's global sensitivity is looked up in
/// [`SensitivityBounds::global_sensitivity`].
pub const OBJECTIVE_SENSITIVITY_KEY: &str = "objective";

/// Outcome of one private selection.
#[derive(Debug, Clone)]
pub struct SelectionOutcome {
    /// Index of the selected candidate.
    pub index: usize,
    /// Epsilon consumed by this selection.
    pub epsilon_spent: f64,
    /// Delta consumed by this selection (0 for the pure-epsilon mechanisms).
    pub delta_spent: f64,
    /// Name of the mechanism that produced the choice.
    pub mechanism: &'static str,
}

/// Draw a uniform sample strictly inside `(0, 1)`.
///
/// `gen_range(0.0..1.0)` can return exactly `0.0`, and `ln(0)` is `-inf`, which
/// silently poisons every noise sample derived from it.
fn open_unit_sample(rng: &mut HpoRng) -> f64 {
    let raw: f64 = rng.gen_range(0.0..1.0);
    if raw <= 0.0 {
        f64::MIN_POSITIVE
    } else if raw >= 1.0 {
        1.0 - f64::EPSILON
    } else {
        raw
    }
}

/// One Laplace sample with scale `b`.
pub fn laplace_sample(rng: &mut HpoRng, scale: f64) -> Result<f64> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the Laplace scale must be positive and finite, got {scale}"
        )));
    }
    let uniform = open_unit_sample(rng) - 0.5;
    let magnitude = (1.0 - 2.0 * uniform.abs()).max(f64::MIN_POSITIVE);
    Ok(-scale * uniform.signum() * magnitude.ln())
}

/// One standard Gumbel sample.
pub fn gumbel_sample(rng: &mut HpoRng) -> f64 {
    let uniform = open_unit_sample(rng);
    -(-uniform.ln()).ln()
}

/// One Gaussian sample with standard deviation `sigma`, by Box-Muller.
pub fn gaussian_sample(rng: &mut HpoRng, sigma: f64) -> Result<f64> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the Gaussian scale must be positive and finite, got {sigma}"
        )));
    }
    let first = open_unit_sample(rng);
    let second = open_unit_sample(rng);
    Ok(sigma * (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos())
}

/// Analytic Gaussian-mechanism standard deviation for `(epsilon, delta)`.
///
/// `sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon` (Dwork & Roth,
/// Thm. A.1), valid for `epsilon <= 1`.
pub fn gaussian_sigma(sensitivity: f64, epsilon: f64, delta: f64) -> Result<f64> {
    if !sensitivity.is_finite() || sensitivity <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the sensitivity must be positive and finite, got {sensitivity}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "epsilon must be positive and finite, got {epsilon}"
        )));
    }
    if epsilon > 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the classic Gaussian-mechanism bound requires epsilon <= 1, got {epsilon}; use a \
             pure-epsilon selection mechanism instead"
        )));
    }
    if !delta.is_finite() || !(0.0..1.0).contains(&delta) || delta <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the Gaussian mechanism requires a delta in (0, 1), got {delta}"
        )));
    }
    Ok((2.0 * (1.25 / delta).ln()).sqrt() * sensitivity / epsilon)
}

/// Convert `utilities` to `f64`, rejecting anything non-finite.
fn utilities_as_f64<T: Float + Debug + Send + Sync + 'static>(utilities: &[T]) -> Result<Vec<f64>> {
    if utilities.is_empty() {
        return Err(OptimError::InvalidParameter(
            "a private selection needs at least one candidate".to_string(),
        ));
    }
    utilities
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let as_f64 = value.to_f64().ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "utility {index} cannot be represented as f64"
                ))
            })?;
            if !as_f64.is_finite() {
                return Err(OptimError::InvalidParameter(format!(
                    "utility {index} is {as_f64}; a private selection cannot be made over \
                     non-finite utilities"
                )));
            }
            Ok(as_f64)
        })
        .collect()
}

/// The index of the largest value, without noise.
fn exact_argmax(utilities: &[f64]) -> Result<usize> {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (index, value) in utilities.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    if best_value.is_finite() {
        Ok(best)
    } else {
        Err(OptimError::InvalidParameter(
            "no finite utility was supplied".to_string(),
        ))
    }
}

/// Select an index with the exponential mechanism.
///
/// Delegates the weighting to the crate's audited value-selecting
/// [`ValueExponentialMechanism`]: the candidate set is the index range encoded
/// as `f64` (exact for any realistic candidate count) and the quality function
/// is a lookup into `utilities`.
pub fn exponential_mechanism_index(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    seed: Option<u64>,
) -> Result<usize> {
    if utilities.is_empty() {
        return Err(OptimError::InvalidParameter(
            "a private selection needs at least one candidate".to_string(),
        ));
    }
    let table = utilities.to_vec();
    let quality = Box::new(move |candidate: &f64| {
        let index = *candidate as usize;
        table.get(index).copied().unwrap_or(f64::NEG_INFINITY)
    });
    let mut mechanism = match seed {
        Some(seed) => ValueExponentialMechanism::<f64>::new_with_seed(quality, seed),
        None => ValueExponentialMechanism::<f64>::new(quality),
    };
    let candidates: Vec<f64> = (0..utilities.len()).map(|index| index as f64).collect();
    let chosen = mechanism.select_output(&candidates, sensitivity, epsilon)?;
    let index = chosen as usize;
    if index >= utilities.len() {
        return Err(OptimError::InvalidState(format!(
            "the exponential mechanism returned index {index} for {} candidates",
            utilities.len()
        )));
    }
    Ok(index)
}

/// The exponential mechanism's selection probabilities, in closed form.
///
/// `P(i) = exp(epsilon * u_i / (2 Delta)) / sum_j exp(epsilon * u_j / (2 Delta))`,
/// computed with the max subtracted for numerical stability. Used to report the
/// probability the mechanism actually assigned to the configuration it returned,
/// instead of a placeholder confidence.
pub fn exponential_mechanism_probabilities(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
) -> Result<Vec<f64>> {
    if utilities.is_empty() {
        return Err(OptimError::InvalidParameter(
            "selection probabilities need at least one candidate".to_string(),
        ));
    }
    if !sensitivity.is_finite() || sensitivity <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the utility sensitivity must be positive and finite, got {sensitivity}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "epsilon must be positive and finite, got {epsilon}"
        )));
    }
    let max = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(OptimError::InvalidParameter(
            "no finite utility was supplied".to_string(),
        ));
    }
    let weights: Vec<f64> = utilities
        .iter()
        .map(|utility| (epsilon * (utility - max) / (2.0 * sensitivity)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(OptimError::InvalidState(format!(
            "the selection weights sum to {total}"
        )));
    }
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

/// The Laplace scale [`noisy_summary_statistics`] uses for the mean release.
///
/// Exposed so a caller can widen a confidence interval by the noise it added,
/// rather than reporting a sampling-only interval as if the release were exact.
pub fn summary_mean_noise_scale(count: usize, value_range: f64, epsilon: f64) -> Result<f64> {
    if count == 0 {
        return Err(OptimError::InvalidParameter(
            "the observation count must be positive".to_string(),
        ));
    }
    if !value_range.is_finite() || value_range <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the public value range must be positive and finite, got {value_range}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "epsilon must be positive and finite, got {epsilon}"
        )));
    }
    Ok(value_range / (count as f64 * (epsilon / 3.0)))
}

/// Report-noisy-max with Gumbel noise.
///
/// Adding `Gumbel(2 Delta / epsilon)` noise to each utility and reporting the
/// argmax is *exactly* the exponential mechanism (the Gumbel-max trick), so it
/// carries the same `epsilon`-DP guarantee.
pub fn report_noisy_max_gumbel(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    rng: &mut HpoRng,
) -> Result<usize> {
    if !sensitivity.is_finite() || sensitivity <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "report-noisy-max requires a positive finite sensitivity, got {sensitivity}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "report-noisy-max requires a positive finite epsilon, got {epsilon}"
        )));
    }
    let scale = 2.0 * sensitivity / epsilon;
    let noisy: Vec<f64> = utilities
        .iter()
        .map(|utility| utility + scale * gumbel_sample(rng))
        .collect();
    exact_argmax(&noisy)
}

/// Report-noisy-max with Laplace noise at scale `2 Delta / epsilon`.
pub fn report_noisy_max_laplace(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    rng: &mut HpoRng,
) -> Result<usize> {
    if !sensitivity.is_finite() || sensitivity <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "report-noisy-max requires a positive finite sensitivity, got {sensitivity}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "report-noisy-max requires a positive finite epsilon, got {epsilon}"
        )));
    }
    let scale = 2.0 * sensitivity / epsilon;
    let mut noisy = Vec::with_capacity(utilities.len());
    for utility in utilities {
        noisy.push(utility + laplace_sample(rng, scale)?);
    }
    exact_argmax(&noisy)
}

/// Argmax after adding Gaussian noise calibrated for `(epsilon, delta)`.
pub fn report_noisy_max_gaussian(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    delta: f64,
    rng: &mut HpoRng,
) -> Result<usize> {
    let sigma = gaussian_sigma(2.0 * sensitivity, epsilon, delta)?;
    let mut noisy = Vec::with_capacity(utilities.len());
    for utility in utilities {
        noisy.push(utility + gaussian_sample(rng, sigma)?);
    }
    exact_argmax(&noisy)
}

impl<T: Float + Debug + Send + Sync + 'static> UtilityFunction<T> {
    /// Map a raw objective value to a selection utility.
    ///
    /// The utility must be monotone in the objective for the exponential
    /// mechanism to select "good" configurations, and its sensitivity is what
    /// `Delta_u` bounds.
    pub fn evaluate(&self, objective: T) -> Result<T> {
        let value = objective.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter(
                "the objective value cannot be represented as f64".to_string(),
            )
        })?;
        if !value.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "the objective value {value} is not finite"
            )));
        }
        let scale = self
            .parameters()
            .first()
            .and_then(|parameter| parameter.to_f64())
            .unwrap_or(1.0);
        let utility = match self.function_type() {
            UtilityFunctionType::Linear => scale * value,
            UtilityFunctionType::Quadratic => scale * value * value,
            UtilityFunctionType::Exponential => {
                let exponent = scale * value;
                if exponent > 700.0 {
                    return Err(OptimError::InvalidParameter(format!(
                        "the exponential utility overflows for objective {value} at scale {scale}"
                    )));
                }
                exponent.exp()
            }
            UtilityFunctionType::Logarithmic => {
                if value <= 0.0 {
                    return Err(OptimError::InvalidParameter(format!(
                        "the logarithmic utility needs a positive objective, got {value}"
                    )));
                }
                scale * value.ln()
            }
            UtilityFunctionType::Custom => {
                return Err(OptimError::UnsupportedOperation(
                    "UtilityFunctionType::Custom carries no function to evaluate; register a \
                     concrete utility instead"
                        .to_string(),
                ))
            }
        };
        T::from(utility).ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "the computed utility {utility} cannot be represented in the parameter type"
            ))
        })
    }

    /// Scalarise a multi-objective value with the configured weights.
    pub fn evaluate_multi(&self, objectives: &[T]) -> Result<T> {
        let weights = self.multi_objective_weights().ok_or_else(|| {
            OptimError::InvalidState(
                "no multi-objective weights are configured on this utility function".to_string(),
            )
        })?;
        if weights.len() != objectives.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "{} weights were configured for {} objectives",
                weights.len(),
                objectives.len()
            )));
        }
        let mut total = 0.0f64;
        for (weight, objective) in weights.iter().zip(objectives.iter()) {
            let weight = weight.to_f64().unwrap_or(f64::NAN);
            let scalar = self.evaluate(*objective)?.to_f64().unwrap_or(f64::NAN);
            total += weight * scalar;
        }
        if !total.is_finite() {
            return Err(OptimError::InvalidParameter(
                "the scalarised utility is not finite".to_string(),
            ));
        }
        T::from(total).ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "the scalarised utility {total} cannot be represented in the parameter type"
            ))
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> SensitivityBounds<T> {
    /// The declared global sensitivity of the objective.
    ///
    /// Looks for [`OBJECTIVE_SENSITIVITY_KEY`] and otherwise takes the largest
    /// declared global sensitivity. Returns `None` when nothing is declared:
    /// guessing a sensitivity would silently invalidate every epsilon derived
    /// from it, so the caller must refuse instead.
    pub fn objective_sensitivity(&self) -> Option<T> {
        if let Some(declared) = self.global_sensitivity.get(OBJECTIVE_SENSITIVITY_KEY) {
            return Some(*declared);
        }
        self.global_sensitivity
            .values()
            .filter(|value| value.is_finite() && **value > T::zero())
            .fold(None, |accumulated: Option<T>, value| match accumulated {
                Some(current) if current >= *value => Some(current),
                _ => Some(*value),
            })
    }

    /// The smooth-sensitivity beta declared for a parameter, if any.
    pub fn smooth_beta(&self, parameter: &str) -> Option<T> {
        self.smooth_sensitivity
            .get(parameter)
            .map(|params| params.beta)
    }

    /// The local sensitivity interval declared for a parameter, if any.
    pub fn local_bounds(&self, parameter: &str) -> Option<(T, T)> {
        self.local_sensitivity.get(parameter).copied()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> SelectionMechanism<T> {
    /// Create a mechanism with explicit parameters.
    pub fn with_parameters(
        mechanism_type: HyperparameterNoiseMechanism,
        selection_params: SelectionParameters<T>,
        utility_function: UtilityFunction<T>,
    ) -> Result<Self> {
        let mut mechanism = Self::new();
        mechanism.set_mechanism_type(mechanism_type);
        mechanism.set_utility_function(utility_function);
        mechanism.set_selection_parameters(selection_params)?;
        Ok(mechanism)
    }

    /// Replace the RNG with a deterministic one (tests only).
    pub fn seed_for_tests(&mut self, seed: u64) {
        self.set_rng(scirs2_core::random::Random::seed(seed));
        self.set_test_seed(Some(seed));
    }

    /// Reseed from OS entropy.
    pub fn reseed_from_os(&mut self) {
        self.set_rng(os_seeded_hpo_rng());
        self.set_test_seed(None);
    }

    /// Privately select the index of a candidate from its utilities.
    ///
    /// The utilities are consumed as supplied: pass them through
    /// [`UtilityFunction::evaluate`] first if the objective needs mapping.
    pub fn select_index(&mut self, utilities: &[T]) -> Result<SelectionOutcome> {
        let table = utilities_as_f64(utilities)?;
        let epsilon = self.selection_params().epsilon;
        let sensitivity = self
            .selection_params()
            .utility_sensitivity
            .to_f64()
            .ok_or_else(|| {
                OptimError::InvalidParameter(
                    "the utility sensitivity cannot be represented as f64".to_string(),
                )
            })?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "private selection requires a positive finite epsilon, got {epsilon}"
            )));
        }
        if !sensitivity.is_finite() || sensitivity <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "private selection requires a positive finite utility sensitivity, got \
                 {sensitivity}"
            )));
        }

        // A configured threshold discards candidates whose utility is below it
        // *before* the mechanism runs. This is only sound when the threshold is
        // public; it is documented as such on `SelectionParameters::threshold`.
        let (table, index_map) = match self.selection_params().threshold {
            Some(threshold) => {
                let threshold = threshold.to_f64().ok_or_else(|| {
                    OptimError::InvalidParameter(
                        "the selection threshold cannot be represented as f64".to_string(),
                    )
                })?;
                let mut kept = Vec::new();
                let mut map = Vec::new();
                for (index, value) in table.iter().enumerate() {
                    if *value >= threshold {
                        kept.push(*value);
                        map.push(index);
                    }
                }
                if kept.is_empty() {
                    return Err(OptimError::InvalidState(format!(
                        "no candidate reaches the configured selection threshold {threshold}"
                    )));
                }
                (kept, Some(map))
            }
            None => (table, None),
        };

        let mechanism_type = self.mechanism_type();
        let seed = self.test_seed();
        let local_index = match mechanism_type {
            HyperparameterNoiseMechanism::Exponential => {
                let call_seed = seed.map(|seed| seed.wrapping_add(self.selection_count() as u64));
                exponential_mechanism_index(&table, sensitivity, epsilon, call_seed)?
            }
            HyperparameterNoiseMechanism::NoisyMax => {
                report_noisy_max_gumbel(&table, sensitivity, epsilon, self.rng_mut())?
            }
            HyperparameterNoiseMechanism::Laplace => {
                report_noisy_max_laplace(&table, sensitivity, epsilon, self.rng_mut())?
            }
            HyperparameterNoiseMechanism::Gaussian => {
                let delta = self.selection_params().delta.ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "Gaussian selection is an (epsilon, delta) mechanism but no delta is \
                         configured in SelectionParameters"
                            .to_string(),
                    )
                })?;
                report_noisy_max_gaussian(&table, sensitivity, epsilon, delta, self.rng_mut())?
            }
            HyperparameterNoiseMechanism::SparseVector => {
                return Err(OptimError::UnsupportedOperation(
                    "the sparse vector technique answers a stream of threshold queries and is not \
                     a one-shot selection mechanism; use \
                     privacy::noise_mechanisms::SparseVectorMechanism, or select with \
                     HyperparameterNoiseMechanism::Exponential"
                        .to_string(),
                ))
            }
        };

        let index = match index_map {
            Some(map) => map[local_index],
            None => local_index,
        };
        let delta_spent = match mechanism_type {
            HyperparameterNoiseMechanism::Gaussian => self.selection_params().delta.unwrap_or(0.0),
            _ => 0.0,
        };
        self.record_selection(epsilon, delta_spent);
        Ok(SelectionOutcome {
            index,
            epsilon_spent: epsilon,
            delta_spent,
            mechanism: mechanism_name(mechanism_type),
        })
    }
}

/// Human-readable mechanism name.
pub fn mechanism_name(mechanism: HyperparameterNoiseMechanism) -> &'static str {
    match mechanism {
        HyperparameterNoiseMechanism::Exponential => "exponential_mechanism",
        HyperparameterNoiseMechanism::Gaussian => "gaussian_report_noisy_max",
        HyperparameterNoiseMechanism::Laplace => "laplace_report_noisy_max",
        HyperparameterNoiseMechanism::NoisyMax => "gumbel_report_noisy_max",
        HyperparameterNoiseMechanism::SparseVector => "sparse_vector",
    }
}

/// Quantiles released by [`noisy_summary_statistics`], in order.
///
/// The median is the `0.5` entry of this list; it is **not** released a second
/// time, because a second release would be a second query against the same data
/// and would have to be paid for separately.
pub const SUMMARY_QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];

/// A differentially private summary, together with what it actually cost.
///
/// `epsilon_spent` is accumulated as each release is made, rather than asserted
/// in a comment, so a caller charges exactly what the code path consumed.
#[derive(Debug, Clone)]
pub struct NoisySummary<T: Float + Debug + Send + Sync + 'static> {
    /// The released statistics.
    pub statistics: SummaryStatistics<T>,
    /// Total epsilon consumed by every release in this summary.
    pub epsilon_spent: f64,
    /// Laplace scale used for the mean release, so a caller can widen a
    /// confidence interval by the noise that was added.
    pub mean_noise_scale: f64,
}

/// Differentially private summary statistics of the observed objectives.
///
/// # Budget split
///
/// `epsilon` is divided into three equal shares -- mean, standard deviation and
/// quantiles -- and the quantile share is divided again across
/// [`SUMMARY_QUANTILES`]. The releases compose linearly, and the total is
/// returned in [`NoisySummary::epsilon_spent`], which is asserted to equal
/// `epsilon` by a test in this module.
///
/// # Sensitivities
///
/// With `R = value_range` the public a-priori range of one observation and `n`
/// observations, under one substitution:
///
/// * **mean**: `|Delta mean| <= R / n`.
/// * **variance**: `var = (1/n) sum x_i^2 - mean^2`; the first term moves by at
///   most `R^2/n` and `mean^2` by at most `2R(R/n) + (R/n)^2`, so
///   `|Delta var| <= 4 R^2 / n` for `n >= 1`. Since `|sqrt(a) - sqrt(b)| <=
///   sqrt(|a - b|)` for non-negative `a, b`, the standard deviation has
///   `|Delta std| <= 2 R / sqrt(n)`. That (deliberately loose) bound is what
///   calibrates the noise, not the tighter-looking `R / sqrt(n)`.
/// * **quantiles**: released by the exponential mechanism over the order
///   statistics with rank utility, whose sensitivity is exactly 1.
pub fn noisy_summary_statistics<T: Float + Debug + Send + Sync + 'static>(
    values: &[T],
    value_range: f64,
    epsilon: f64,
    rng: &mut HpoRng,
) -> Result<NoisySummary<T>> {
    if values.is_empty() {
        return Err(OptimError::InvalidParameter(
            "summary statistics need at least one observation".to_string(),
        ));
    }
    if !value_range.is_finite() || value_range <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "the public value range must be positive and finite, got {value_range}"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "noisy summary statistics need a positive finite epsilon, got {epsilon}"
        )));
    }

    let observations = utilities_as_f64(values)?;
    let count = observations.len() as f64;
    let per_statistic_epsilon = epsilon / 3.0;
    let mut epsilon_spent = 0.0f64;

    // Mean: sensitivity R / n.
    let mean = observations.iter().sum::<f64>() / count;
    let mean_noise_scale = value_range / (count * per_statistic_epsilon);
    let noisy_mean = mean + laplace_sample(rng, mean_noise_scale)?;
    epsilon_spent += per_statistic_epsilon;

    // Standard deviation: sensitivity 2 R / sqrt(n), derived above.
    let variance = observations
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count;
    let std_scale = 2.0 * value_range / (count.sqrt() * per_statistic_epsilon);
    let noisy_std = (variance.sqrt() + laplace_sample(rng, std_scale)?).max(0.0);
    epsilon_spent += per_statistic_epsilon;

    // Quantiles, including the median, by the exponential mechanism over the
    // order statistics (Smith 2011). The median is taken from this loop and is
    // not released a second time.
    let mut sorted = observations.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let per_quantile_epsilon = per_statistic_epsilon / SUMMARY_QUANTILES.len() as f64;
    let mut noisy_quantiles = Vec::with_capacity(SUMMARY_QUANTILES.len());
    let mut noisy_median = None;
    for quantile in SUMMARY_QUANTILES {
        let index = private_quantile_index(&sorted, quantile, per_quantile_epsilon, rng)?;
        epsilon_spent += per_quantile_epsilon;
        if (quantile - 0.5).abs() < f64::EPSILON {
            noisy_median = Some(sorted[index]);
        }
        let value = T::from(sorted[index]).ok_or_else(|| {
            OptimError::InvalidParameter("a quantile cannot be represented".to_string())
        })?;
        noisy_quantiles.push((quantile, value));
    }
    let noisy_median = noisy_median.ok_or_else(|| {
        OptimError::InvalidState(
            "SUMMARY_QUANTILES must contain 0.5 so the median comes out of the quantile releases"
                .to_string(),
        )
    })?;

    let statistics = SummaryStatistics {
        noisy_mean: T::from(noisy_mean).ok_or_else(|| {
            OptimError::InvalidParameter("the noisy mean cannot be represented".to_string())
        })?,
        noisy_std: T::from(noisy_std).ok_or_else(|| {
            OptimError::InvalidParameter(
                "the noisy standard deviation cannot be represented".to_string(),
            )
        })?,
        noisy_median: T::from(noisy_median).ok_or_else(|| {
            OptimError::InvalidParameter("the noisy median cannot be represented".to_string())
        })?,
        noisy_quantiles,
    };

    Ok(NoisySummary {
        statistics,
        epsilon_spent,
        mean_noise_scale,
    })
}

/// Exponential-mechanism index of a quantile over a sorted sample.
///
/// The utility of order statistic `i` is `-|i - q * (n - 1)|`, whose
/// sensitivity is 1 (replacing one observation shifts every rank by at most
/// one). This is the standard private-quantile construction (Smith 2011).
fn private_quantile_index(
    sorted: &[f64],
    quantile: f64,
    epsilon: f64,
    rng: &mut HpoRng,
) -> Result<usize> {
    if sorted.is_empty() {
        return Err(OptimError::InvalidParameter(
            "a quantile needs at least one observation".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&quantile) {
        return Err(OptimError::InvalidParameter(format!(
            "the quantile must lie in [0, 1], got {quantile}"
        )));
    }
    let target = quantile * (sorted.len() - 1) as f64;
    let utilities: Vec<f64> = (0..sorted.len())
        .map(|index| -((index as f64 - target).abs()))
        .collect();
    report_noisy_max_gumbel(&utilities, 1.0, epsilon, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::os_seeded_hpo_rng;
    use std::collections::HashMap;

    fn seeded(seed: u64) -> HpoRng {
        scirs2_core::random::Random::seed(seed)
    }

    /// Empirical selection frequencies over `trials` draws.
    fn frequencies<F: FnMut() -> usize>(mut draw: F, candidates: usize, trials: usize) -> Vec<f64> {
        let mut counts = vec![0usize; candidates];
        for _ in 0..trials {
            counts[draw()] += 1;
        }
        counts
            .into_iter()
            .map(|count| count as f64 / trials as f64)
            .collect()
    }

    #[test]
    fn the_exponential_mechanism_is_not_an_exact_argmax() {
        // Regression for the core finding: selection used to be the exact
        // argmax, which leaks the choice. With a small epsilon the mechanism
        // must sometimes return a non-optimal candidate.
        let utilities = [0.0, 0.1, 0.2, 1.0];
        let mut non_argmax = 0usize;
        for trial in 0..400u64 {
            let index = match exponential_mechanism_index(&utilities, 1.0, 0.5, Some(trial)) {
                Ok(index) => index,
                Err(err) => panic!("selection failed: {err}"),
            };
            if index != 3 {
                non_argmax += 1;
            }
        }
        assert!(
            non_argmax > 40,
            "only {non_argmax}/400 draws deviated from the argmax; the choice is not private"
        );
    }

    #[test]
    fn a_large_epsilon_concentrates_on_the_argmax() {
        let utilities = [0.0, 0.1, 0.2, 1.0];
        let mut argmax_hits = 0usize;
        for trial in 0..200u64 {
            let index = match exponential_mechanism_index(&utilities, 1.0, 200.0, Some(trial)) {
                Ok(index) => index,
                Err(err) => panic!("selection failed: {err}"),
            };
            if index == 3 {
                argmax_hits += 1;
            }
        }
        assert!(
            argmax_hits >= 195,
            "only {argmax_hits}/200 draws hit the argmax at epsilon = 200"
        );
    }

    #[test]
    fn the_selection_distribution_matches_the_closed_form_weights() {
        // P(i) = exp(eps u_i / 2 Delta) / sum_j exp(eps u_j / 2 Delta).
        let utilities = [0.0f64, 1.0, 2.0];
        let epsilon = 1.0;
        let sensitivity = 1.0;
        let weights: Vec<f64> = utilities
            .iter()
            .map(|utility| (epsilon * utility / (2.0 * sensitivity)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let expected: Vec<f64> = weights.iter().map(|weight| weight / total).collect();

        let trials = 20_000usize;
        let mut trial = 0u64;
        let observed = frequencies(
            || {
                trial += 1;
                match exponential_mechanism_index(&utilities, sensitivity, epsilon, Some(trial)) {
                    Ok(index) => index,
                    Err(err) => panic!("selection failed: {err}"),
                }
            },
            utilities.len(),
            trials,
        );
        for (index, (observed, expected)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (observed - expected).abs() < 0.02,
                "candidate {index}: observed {observed:.4} vs expected {expected:.4}"
            );
        }
    }

    #[test]
    fn gumbel_report_noisy_max_matches_the_exponential_mechanism() {
        // The Gumbel-max trick makes these two mechanisms identical in
        // distribution; agreement is a genuine cross-check of both.
        let utilities = [0.0f64, 0.5, 1.0, 1.5];
        let epsilon = 1.0;
        let sensitivity = 1.0;
        let trials = 20_000usize;

        let mut trial = 0u64;
        let exponential = frequencies(
            || {
                trial += 1;
                match exponential_mechanism_index(&utilities, sensitivity, epsilon, Some(trial)) {
                    Ok(index) => index,
                    Err(err) => panic!("selection failed: {err}"),
                }
            },
            utilities.len(),
            trials,
        );

        let mut rng = seeded(12_345);
        let gumbel = frequencies(
            || match report_noisy_max_gumbel(&utilities, sensitivity, epsilon, &mut rng) {
                Ok(index) => index,
                Err(err) => panic!("selection failed: {err}"),
            },
            utilities.len(),
            trials,
        );

        for (index, (left, right)) in exponential.iter().zip(gumbel.iter()).enumerate() {
            assert!(
                (left - right).abs() < 0.02,
                "candidate {index}: exponential {left:.4} vs gumbel {right:.4}"
            );
        }
    }

    #[test]
    fn laplace_report_noisy_max_favours_but_does_not_guarantee_the_argmax() {
        let utilities = [0.0f64, 0.2, 1.0];
        let mut rng = seeded(7);
        let observed = frequencies(
            || match report_noisy_max_laplace(&utilities, 1.0, 1.0, &mut rng) {
                Ok(index) => index,
                Err(err) => panic!("selection failed: {err}"),
            },
            utilities.len(),
            5_000,
        );
        assert!(observed[2] > observed[0], "the argmax must be favoured");
        assert!(
            observed[0] > 0.02,
            "a weak candidate must still be reachable"
        );
    }

    #[test]
    fn gaussian_selection_needs_a_delta_and_a_small_epsilon() {
        let utilities = [0.0f64, 1.0];
        let mut rng = seeded(3);
        assert!(report_noisy_max_gaussian(&utilities, 1.0, 0.5, 0.0, &mut rng).is_err());
        assert!(report_noisy_max_gaussian(&utilities, 1.0, 2.0, 1e-5, &mut rng).is_err());
        assert!(report_noisy_max_gaussian(&utilities, 1.0, 0.5, 1e-5, &mut rng).is_ok());
    }

    #[test]
    fn the_gaussian_sigma_matches_the_published_closed_form() {
        // sqrt(2 ln(1.25/delta)) * sensitivity / epsilon, cross-checked against
        // an independent evaluation of the same expression in Python:
        //   sqrt(2 ln(1.25/1e-5)) = 4.844805262605389
        //   sqrt(2 ln(1.25/1e-6)) = 5.298802526850474
        for (delta, expected) in [
            (1e-5f64, 4.844_805_262_605_389f64),
            (1e-6, 5.298_802_526_850_474),
        ] {
            let sigma = match gaussian_sigma(1.0, 1.0, delta) {
                Ok(sigma) => sigma,
                Err(err) => panic!("sigma failed: {err}"),
            };
            assert!(
                (sigma - expected).abs() < 1e-12,
                "sigma({delta}) = {sigma}, expected {expected}"
            );
        }
        // The scale is linear in the sensitivity and inverse in epsilon.
        let doubled = match gaussian_sigma(2.0, 1.0, 1e-5) {
            Ok(sigma) => sigma,
            Err(err) => panic!("sigma failed: {err}"),
        };
        assert!((doubled - 2.0 * 4.844_805_262_605_389).abs() < 1e-12);
        let halved_epsilon = match gaussian_sigma(1.0, 0.5, 1e-5) {
            Ok(sigma) => sigma,
            Err(err) => panic!("sigma failed: {err}"),
        };
        assert!((halved_epsilon - 2.0 * 4.844_805_262_605_389).abs() < 1e-12);
    }

    #[test]
    fn degenerate_selection_parameters_are_refused() {
        let utilities = [0.0f64, 1.0];
        assert!(exponential_mechanism_index(&[], 1.0, 1.0, Some(1)).is_err());
        assert!(exponential_mechanism_index(&utilities, 0.0, 1.0, Some(1)).is_err());
        assert!(exponential_mechanism_index(&utilities, 1.0, 0.0, Some(1)).is_err());
        assert!(exponential_mechanism_index(&utilities, -1.0, 1.0, Some(1)).is_err());
        assert!(exponential_mechanism_index(&[f64::NAN, 1.0], 1.0, 1.0, Some(1)).is_err());
    }

    #[test]
    fn the_laplace_sampler_never_produces_an_infinity() {
        let mut rng = os_seeded_hpo_rng();
        for _ in 0..20_000 {
            let sample = match laplace_sample(&mut rng, 1.0) {
                Ok(sample) => sample,
                Err(err) => panic!("sampling failed: {err}"),
            };
            assert!(sample.is_finite(), "sample {sample} is not finite");
        }
        assert!(laplace_sample(&mut rng, 0.0).is_err());
        assert!(laplace_sample(&mut rng, f64::NAN).is_err());
    }

    #[test]
    fn the_laplace_sampler_has_the_right_scale() {
        let mut rng = seeded(99);
        let scale = 2.0;
        let trials = 200_000usize;
        let mut absolute_total = 0.0;
        for _ in 0..trials {
            let sample = match laplace_sample(&mut rng, scale) {
                Ok(sample) => sample,
                Err(err) => panic!("sampling failed: {err}"),
            };
            absolute_total += sample.abs();
        }
        // E|Lap(b)| = b.
        let mean_absolute = absolute_total / trials as f64;
        assert!(
            (mean_absolute - scale).abs() < 0.05,
            "E|X| = {mean_absolute}, expected {scale}"
        );
    }

    #[test]
    fn the_gumbel_sampler_has_the_right_mean() {
        let mut rng = seeded(4_242);
        let trials = 200_000usize;
        let mut total = 0.0;
        for _ in 0..trials {
            total += gumbel_sample(&mut rng);
        }
        // E[Gumbel(0,1)] = Euler-Mascheroni constant.
        let mean = total / trials as f64;
        assert!((mean - 0.577_215_664_9).abs() < 0.02, "mean = {mean}");
    }

    #[test]
    fn the_gaussian_sampler_has_the_right_standard_deviation() {
        let mut rng = seeded(24);
        let sigma = 3.0;
        let trials = 200_000usize;
        let mut total = 0.0;
        let mut total_squared = 0.0;
        for _ in 0..trials {
            let sample = match gaussian_sample(&mut rng, sigma) {
                Ok(sample) => sample,
                Err(err) => panic!("sampling failed: {err}"),
            };
            total += sample;
            total_squared += sample * sample;
        }
        let count = trials as f64;
        let mean = total / count;
        let observed = (total_squared / count - mean * mean).sqrt();
        assert!((mean).abs() < 0.05, "mean = {mean}");
        assert!((observed - sigma).abs() < 0.05, "sigma = {observed}");
    }

    #[test]
    fn utility_functions_map_objectives_monotonically() {
        let linear: UtilityFunction<f64> = UtilityFunction::new();
        match linear.evaluate(2.0) {
            Ok(value) => assert!((value - 2.0).abs() < 1e-12),
            Err(err) => panic!("evaluate failed: {err}"),
        }
        assert!(linear.evaluate(f64::NAN).is_err());
    }

    #[test]
    fn a_custom_utility_function_is_refused_rather_than_faked() {
        let mut utility: UtilityFunction<f64> = UtilityFunction::new();
        utility.set_function_type(UtilityFunctionType::Custom);
        assert!(utility.evaluate(1.0).is_err());
    }

    #[test]
    fn the_logarithmic_utility_refuses_a_non_positive_objective() {
        let mut utility: UtilityFunction<f64> = UtilityFunction::new();
        utility.set_function_type(UtilityFunctionType::Logarithmic);
        assert!(utility.evaluate(0.0).is_err());
        assert!(utility.evaluate(-1.0).is_err());
        assert!(utility.evaluate(std::f64::consts::E).is_ok());
    }

    #[test]
    fn multi_objective_scalarisation_checks_the_weight_count() {
        let mut utility: UtilityFunction<f64> = UtilityFunction::new();
        assert!(utility.evaluate_multi(&[1.0, 2.0]).is_err());
        utility.set_multi_objective_weights(Some(vec![0.5, 0.5]));
        match utility.evaluate_multi(&[1.0, 3.0]) {
            Ok(value) => assert!((value - 2.0).abs() < 1e-12),
            Err(err) => panic!("scalarisation failed: {err}"),
        }
        assert!(utility.evaluate_multi(&[1.0]).is_err());
    }

    #[test]
    fn the_objective_sensitivity_must_be_declared() {
        let empty: SensitivityBounds<f64> = SensitivityBounds {
            global_sensitivity: HashMap::new(),
            local_sensitivity: HashMap::new(),
            smooth_sensitivity: HashMap::new(),
        };
        assert!(
            empty.objective_sensitivity().is_none(),
            "an undeclared sensitivity must not be guessed"
        );

        let mut declared = HashMap::new();
        declared.insert(OBJECTIVE_SENSITIVITY_KEY.to_string(), 0.25f64);
        let bounds: SensitivityBounds<f64> = SensitivityBounds {
            global_sensitivity: declared,
            local_sensitivity: HashMap::new(),
            smooth_sensitivity: HashMap::new(),
        };
        assert_eq!(bounds.objective_sensitivity(), Some(0.25));
    }

    #[test]
    fn the_selection_mechanism_charges_and_reports_its_epsilon() {
        let mut mechanism: SelectionMechanism<f64> = SelectionMechanism::new();
        mechanism.seed_for_tests(11);
        let outcome = match mechanism.select_index(&[0.0, 0.5, 1.0]) {
            Ok(outcome) => outcome,
            Err(err) => panic!("selection failed: {err}"),
        };
        assert!(outcome.index < 3);
        assert_eq!(outcome.epsilon_spent, 1.0);
        assert_eq!(outcome.delta_spent, 0.0);
        assert_eq!(outcome.mechanism, "exponential_mechanism");
        assert_eq!(mechanism.epsilon_spent(), 1.0);

        let _ = mechanism.select_index(&[0.0, 1.0]);
        assert_eq!(
            mechanism.epsilon_spent(),
            2.0,
            "each selection must be charged"
        );
        assert_eq!(mechanism.selection_count(), 2);
    }

    #[test]
    fn a_sparse_vector_selection_is_refused_with_a_pointer_to_the_real_primitive() {
        let mut mechanism: SelectionMechanism<f64> = SelectionMechanism::new();
        mechanism.set_mechanism_type(HyperparameterNoiseMechanism::SparseVector);
        let message = match mechanism.select_index(&[0.0, 1.0]) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("the sparse vector technique is not a selection mechanism"),
        };
        assert!(message.contains("SparseVectorMechanism"), "got: {message}");
    }

    #[test]
    fn every_pure_epsilon_mechanism_selects_a_valid_index() {
        for mechanism_type in [
            HyperparameterNoiseMechanism::Exponential,
            HyperparameterNoiseMechanism::NoisyMax,
            HyperparameterNoiseMechanism::Laplace,
        ] {
            let mut mechanism: SelectionMechanism<f64> = SelectionMechanism::new();
            mechanism.set_mechanism_type(mechanism_type);
            mechanism.seed_for_tests(5);
            let outcome = match mechanism.select_index(&[0.0, 0.5, 1.0]) {
                Ok(outcome) => outcome,
                Err(err) => panic!("{mechanism_type:?} failed: {err}"),
            };
            assert!(outcome.index < 3);
            assert_eq!(outcome.delta_spent, 0.0);
        }
    }

    #[test]
    fn a_threshold_restricts_the_candidate_set_and_remaps_the_index() {
        let mut mechanism: SelectionMechanism<f64> = SelectionMechanism::new();
        mechanism.seed_for_tests(2);
        let mut params = mechanism.selection_params().clone();
        params.threshold = Some(0.9);
        let ok = mechanism.set_selection_parameters(params);
        assert!(ok.is_ok());

        for _ in 0..50 {
            let outcome = match mechanism.select_index(&[0.0, 0.5, 1.0, 0.95]) {
                Ok(outcome) => outcome,
                Err(err) => panic!("selection failed: {err}"),
            };
            assert!(
                outcome.index == 2 || outcome.index == 3,
                "index {} is below the threshold",
                outcome.index
            );
        }

        let mut params = mechanism.selection_params().clone();
        params.threshold = Some(5.0);
        let ok = mechanism.set_selection_parameters(params);
        assert!(ok.is_ok());
        assert!(
            mechanism.select_index(&[0.0, 1.0]).is_err(),
            "an unreachable threshold must be an error"
        );
    }

    #[test]
    fn noisy_summary_statistics_track_the_true_values_and_are_not_exact() {
        let values: Vec<f64> = (0..200).map(|index| index as f64 / 200.0).collect();
        let mut rng = seeded(31);
        let released = match noisy_summary_statistics(&values, 1.0, 4.0, &mut rng) {
            Ok(released) => released,
            Err(err) => panic!("summary failed: {err}"),
        };
        assert!(
            (released.epsilon_spent - 4.0).abs() < 1e-12,
            "the summary spent {} of a 4.0 budget",
            released.epsilon_spent
        );
        let summary = released.statistics;
        let true_mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(
            (summary.noisy_mean - true_mean).abs() < 0.2,
            "noisy mean {} vs true {true_mean}",
            summary.noisy_mean
        );
        assert!(
            summary.noisy_std > 0.0,
            "the standard deviation must not be the hardcoded zero it used to be"
        );
        assert_ne!(
            summary.noisy_median, summary.noisy_mean,
            "the median must not be a copy of the mean"
        );
        assert_eq!(summary.noisy_quantiles.len(), 3);
        assert!(summary
            .noisy_quantiles
            .iter()
            .all(|(_, value)| { (0.0..=1.0).contains(value) }));
    }

    #[test]
    fn noisy_summary_statistics_reject_degenerate_inputs() {
        let mut rng = seeded(1);
        assert!(noisy_summary_statistics::<f64>(&[], 1.0, 1.0, &mut rng).is_err());
        assert!(noisy_summary_statistics(&[1.0f64], 0.0, 1.0, &mut rng).is_err());
        assert!(noisy_summary_statistics(&[1.0f64], 1.0, 0.0, &mut rng).is_err());
    }

    #[test]
    fn the_private_median_concentrates_near_the_true_median() {
        let values: Vec<f64> = (0..101).map(|index| index as f64).collect();
        let mut rng = seeded(77);
        let mut sorted = values.clone();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let mut total_offset = 0.0;
        for _ in 0..500 {
            let index = match private_quantile_index(&sorted, 0.5, 2.0, &mut rng) {
                Ok(index) => index,
                Err(err) => panic!("quantile failed: {err}"),
            };
            total_offset += (index as f64 - 50.0).abs();
        }
        let mean_offset = total_offset / 500.0;
        assert!(
            mean_offset < 5.0,
            "the private median drifted {mean_offset} ranks from the truth"
        );
        assert!(
            mean_offset > 0.0,
            "an exact median would leak the selection"
        );
    }
}
