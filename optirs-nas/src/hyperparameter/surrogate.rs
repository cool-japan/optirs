//! GP-free Bayesian optimization for [`super::OptimizationStrategy::Bayesian`]
//! (F15).
//!
//! `bayesian_optimization` used to be `self.random_search()` behind a
//! `// Simplified Bayesian optimization implementation` comment, and the
//! `SurrogateModel` / `AcquisitionFunction` types it was supposed to drive were
//! never constructed.
//!
//! What is implemented here is a real surrogate-plus-acquisition loop that needs
//! no Gaussian process and no linear algebra:
//!
//! * **Surrogate** — Nadaraya-Watson kernel regression over the observed
//!   `(configuration, score)` pairs in normalized parameter space. The predictive
//!   mean is the kernel-weighted average of the observed scores; the predictive
//!   variance combines the kernel-weighted spread of nearby scores with a
//!   distance-driven term that grows where no observation is close, so the model
//!   is genuinely uncertain in unexplored regions. This is
//!   [`SurrogateModelType::KernelRegression`].
//! * **Acquisition** — Expected Improvement, Probability of Improvement, Upper
//!   Confidence Bound and an entropy-style term, each computed from that
//!   mean/variance pair with an `erf`-based normal CDF/PDF. The acquisition is
//!   maximized by scoring a batch of prior draws (random-search-with-a-model, the
//!   standard cheap inner optimizer), which is why the returned point is *not* a
//!   random draw even though the candidates are.
//!
//! Under two observations no surrogate can be fitted, and this module says so
//! (`Err`) rather than quietly sampling at random: the caller owns the initial
//! design.

use super::support::{
    normalize, sample_parameter, snap_to_discrete, sorted_categorical_names,
    sorted_parameter_names, StrategyRng, SuggestedParameters,
};
use super::{
    AcquisitionFunction, HyperparameterConfiguration, HyperparameterSpace, SurrogateModelType,
};
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Tunables of the surrogate search.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateConfig {
    /// Observations required before the surrogate is fitted.
    pub startup_trials: usize,
    /// Kernel bandwidth in normalized space (a distance of `1.0` spans a whole
    /// parameter axis).
    pub bandwidth: f64,
    /// Candidate points scored per suggestion.
    pub candidates: usize,
    /// Exploration weight of [`AcquisitionFunction::UpperConfidenceBound`].
    pub ucb_beta: f64,
    /// Floor on the predictive standard deviation, so acquisition functions never
    /// divide by zero.
    pub min_sigma: f64,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            startup_trials: 5,
            bandwidth: 0.25,
            candidates: 64,
            ucb_beta: 2.0,
            min_sigma: 1e-6,
        }
    }
}

/// A fitted kernel-regression surrogate over normalized parameter vectors.
#[derive(Debug, Clone)]
pub struct KernelSurrogate {
    /// One normalized vector per observation.
    points: Vec<Vec<f64>>,
    /// Observed score for each point (higher is better).
    scores: Vec<f64>,
    bandwidth: f64,
    /// Variance of all observed scores, used as the prior variance far from data.
    prior_variance: f64,
    min_sigma: f64,
}

impl KernelSurrogate {
    /// Fit to `points` / `scores` (equal length, at least two entries).
    pub fn fit(points: Vec<Vec<f64>>, scores: Vec<f64>, config: &SurrogateConfig) -> Result<Self> {
        if points.len() != scores.len() {
            return Err(OptimError::InvalidParameter(format!(
                "surrogate got {} points for {} scores",
                points.len(),
                scores.len()
            )));
        }
        if points.len() < 2 {
            return Err(OptimError::InvalidConfig(format!(
                "the kernel surrogate needs at least 2 scored observations, got {}",
                points.len()
            )));
        }
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let prior_variance = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n;
        Ok(Self {
            points,
            scores,
            bandwidth: config.bandwidth.max(1e-6),
            prior_variance,
            min_sigma: config.min_sigma.max(0.0),
        })
    }

    /// Number of observations backing the surrogate.
    pub fn observation_count(&self) -> usize {
        self.points.len()
    }

    /// Best observed score.
    pub fn best_score(&self) -> f64 {
        self.scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Squared-exponential kernel weight between two normalized vectors.
    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut squared = 0.0;
        for k in 0..a.len().min(b.len()) {
            let diff = a[k] - b[k];
            squared += diff * diff;
        }
        (-0.5 * squared / (self.bandwidth * self.bandwidth)).exp()
    }

    /// Predictive mean and standard deviation at `x`.
    ///
    /// Far from every observation the kernel weights vanish, the mean falls back
    /// to the global average and the standard deviation rises to the prior — which
    /// is exactly the behaviour an acquisition function needs in order to explore.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let mut weight_sum = 0.0;
        let mut weighted_score = 0.0;
        for (point, score) in self.points.iter().zip(self.scores.iter()) {
            let weight = self.kernel(x, point);
            weight_sum += weight;
            weighted_score += weight * score;
        }

        let global_mean = self.scores.iter().sum::<f64>() / self.scores.len() as f64;
        if weight_sum <= f64::MIN_POSITIVE {
            return (global_mean, self.prior_variance.sqrt().max(self.min_sigma));
        }

        let mean = weighted_score / weight_sum;
        let mut weighted_variance = 0.0;
        for (point, score) in self.points.iter().zip(self.scores.iter()) {
            let weight = self.kernel(x, point);
            let diff = score - mean;
            weighted_variance += weight * diff * diff;
        }
        let local_variance = weighted_variance / weight_sum;

        // `density` in (0, 1]: 1 when a point coincides with an observation, ->0
        // when nothing is nearby. The prior variance is blended in proportionally.
        let density = (weight_sum / self.points.len() as f64).clamp(0.0, 1.0);
        let variance = local_variance * density + self.prior_variance * (1.0 - density);
        (mean, variance.max(0.0).sqrt().max(self.min_sigma))
    }

    /// Value of `acquisition` at `x`, always to be **maximized**.
    pub fn acquisition(
        &self,
        x: &[f64],
        acquisition: AcquisitionFunction,
        config: &SurrogateConfig,
    ) -> f64 {
        let (mean, sigma) = self.predict(x);
        let best = self.best_score();
        match acquisition {
            AcquisitionFunction::ExpectedImprovement => {
                // Standard EI for a maximization objective:
                // (mu - f*) * Phi(z) + sigma * phi(z), z = (mu - f*) / sigma.
                let improvement = mean - best;
                let z = improvement / sigma;
                improvement * normal_cdf(z) + sigma * normal_pdf(z)
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                // A real probability, not the binary step the crate used before.
                normal_cdf((mean - best) / sigma)
            }
            AcquisitionFunction::UpperConfidenceBound => mean + config.ucb_beta * sigma,
            AcquisitionFunction::EntropySearch => {
                // Differential entropy of the predictive Gaussian, which ranks
                // points purely by how little is known about them.
                0.5 * (std::f64::consts::TAU * std::f64::consts::E * sigma * sigma).ln()
            }
        }
    }
}

/// Standard normal CDF via `erf`.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
pub fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / std::f64::consts::TAU.sqrt()
}

/// Abramowitz & Stegun 7.1.26 rational approximation of `erf`
/// (absolute error below 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Normalized vector for a configuration, in the canonical parameter order.
/// Categorical parameters are appended as one normalized index per parameter, so
/// the distance metric covers them too.
fn encode<T: Float>(
    space: &HyperparameterSpace<T>,
    configuration: &HyperparameterConfiguration<T>,
    continuous: &[String],
    categorical: &[String],
) -> Vec<f64> {
    let mut vector = Vec::with_capacity(continuous.len() + categorical.len());
    for name in continuous {
        let Some(range) = space.parameter_ranges().get(name) else {
            continue;
        };
        let raw = configuration
            .parameters
            .get(name)
            .map(|v| v.to_f64().unwrap_or(0.0));
        vector.push(match raw {
            Some(value) => normalize(range, value),
            // A configuration that never set this parameter sits at the axis
            // midpoint rather than at an endpoint it never chose.
            None => 0.5,
        });
    }
    for name in categorical {
        let options = space.categorical_options().get(name);
        let position = configuration
            .categorical_parameters
            .get(name)
            .zip(options)
            .and_then(|(value, options)| options.iter().position(|option| option == value));
        let count = options.map(|options| options.len()).unwrap_or(0);
        vector.push(match (position, count) {
            (Some(index), count) if count > 1 => index as f64 / (count - 1) as f64,
            _ => 0.5,
        });
    }
    vector
}

/// Suggest the next configuration by maximizing `acquisition` over the fitted
/// surrogate.
pub fn suggest<T: Float>(
    space: &HyperparameterSpace<T>,
    scored: &[(&HyperparameterConfiguration<T>, f64)],
    acquisition: AcquisitionFunction,
    config: &SurrogateConfig,
    rng: &mut StrategyRng,
) -> Result<SuggestedParameters<T>> {
    if scored.len() < config.startup_trials.max(2) {
        return Err(OptimError::InvalidConfig(format!(
            "Bayesian optimization needs at least {} scored observations to fit its \
             surrogate; only {} are available (run the random initial design first)",
            config.startup_trials.max(2),
            scored.len()
        )));
    }

    let continuous = sorted_parameter_names(space);
    let categorical_names = sorted_categorical_names(space);
    if continuous.is_empty() && categorical_names.is_empty() {
        return Err(OptimError::InvalidConfig(
            "Bayesian optimization requires at least one parameter; the search space declares none"
                .to_string(),
        ));
    }

    let points: Vec<Vec<f64>> = scored
        .iter()
        .map(|(configuration, _)| encode(space, configuration, &continuous, &categorical_names))
        .collect();
    let scores: Vec<f64> = scored.iter().map(|(_, score)| *score).collect();
    let surrogate = KernelSurrogate::fit(points, scores, config)?;

    let mut best: Option<(f64, SuggestedParameters<T>)> = None;
    for _ in 0..config.candidates.max(1) {
        let mut parameters = HashMap::new();
        for name in &continuous {
            if let Some(range) = space.parameter_ranges().get(name) {
                let raw = snap_to_discrete(range, sample_parameter(range, rng));
                parameters.insert(
                    name.clone(),
                    scirs2_core::numeric::NumCast::from(raw).unwrap_or_else(T::zero),
                );
            }
        }
        let mut categorical = HashMap::new();
        for name in &categorical_names {
            if let Some(options) = space.categorical_options().get(name) {
                if !options.is_empty() {
                    let index = rng.gen_range(0..options.len());
                    categorical.insert(name.clone(), options[index].clone());
                }
            }
        }

        let candidate = HyperparameterConfiguration {
            id: String::new(),
            parameters: parameters.clone(),
            categorical_parameters: categorical.clone(),
            score: None,
            metadata: HashMap::new(),
        };
        let vector = encode(space, &candidate, &continuous, &categorical_names);
        let value = surrogate.acquisition(&vector, acquisition, config);
        if best
            .as_ref()
            .map(|(current, _)| value > *current)
            .unwrap_or(true)
        {
            best = Some((value, (parameters, categorical)));
        }
    }

    best.map(|(_, suggestion)| suggestion).ok_or_else(|| {
        OptimError::OptimizationError(
            "surrogate acquisition maximization produced no candidate".to_string(),
        )
    })
}

/// The surrogate family this module implements. Exposed so the optimizer can
/// record honestly what kind of model backs its `Bayesian` strategy.
pub const IMPLEMENTED_SURROGATE: SurrogateModelType = SurrogateModelType::KernelRegression;

#[cfg(test)]
mod tests {
    use super::super::{DistributionType, ParameterRange};
    use super::*;
    use scirs2_core::random::Random;

    fn unit_space() -> HyperparameterSpace<f64> {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterRange {
                name: "x".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                distribution: DistributionType::Uniform,
                log_scale: false,
                discrete_values: None,
            },
        );
        space
    }

    /// Score peaks sharply at x = 0.75.
    fn observations() -> Vec<HyperparameterConfiguration<f64>> {
        (0..21)
            .map(|i| {
                let x = i as f64 / 20.0;
                let mut parameters = HashMap::new();
                parameters.insert("x".to_string(), x);
                HyperparameterConfiguration {
                    id: format!("o{i}"),
                    parameters,
                    categorical_parameters: HashMap::new(),
                    score: Some(-(x - 0.75) * (x - 0.75)),
                    metadata: HashMap::new(),
                }
            })
            .collect()
    }

    fn scored(
        configs: &[HyperparameterConfiguration<f64>],
    ) -> Vec<(&HyperparameterConfiguration<f64>, f64)> {
        configs
            .iter()
            .map(|c| (c, c.score.unwrap_or(0.0)))
            .collect()
    }

    /// The rational approximation is documented to 1.5e-7 absolute error; these
    /// are the textbook reference values.
    #[test]
    fn erf_matches_known_values() {
        assert!(erf(0.0).abs() < 1.5e-7, "got {}", erf(0.0));
        assert!((erf(1.0) - 0.842_700_79).abs() < 1.5e-7, "got {}", erf(1.0));
        assert!((erf(-1.0) + 0.842_700_79).abs() < 1.5e-7);
        assert!((erf(0.5) - 0.520_499_88).abs() < 1.5e-7, "got {}", erf(0.5));
        assert!((erf(2.0) - 0.995_322_27).abs() < 1.5e-7, "got {}", erf(2.0));
        assert!((erf(3.0) - 0.999_977_91).abs() < 1.5e-7, "got {}", erf(3.0));
    }

    #[test]
    fn normal_cdf_and_pdf_are_standard() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!(
            (normal_cdf(1.96) - 0.975).abs() < 1e-3,
            "{}",
            normal_cdf(1.96)
        );
        assert!((normal_pdf(0.0) - 0.398_942_28).abs() < 1e-7);
    }

    #[test]
    fn the_surrogate_interpolates_observations_and_is_uncertain_elsewhere() {
        let config = SurrogateConfig {
            bandwidth: 0.05,
            ..SurrogateConfig::default()
        };
        let surrogate = KernelSurrogate::fit(
            vec![vec![0.0], vec![0.5], vec![1.0]],
            vec![0.0, 1.0, 0.0],
            &config,
        )
        .expect("fit");

        let (mean_at_peak, sigma_at_peak) = surrogate.predict(&[0.5]);
        assert!(
            (mean_at_peak - 1.0).abs() < 0.05,
            "the surrogate should reproduce the observed peak, got {mean_at_peak}"
        );
        assert!(sigma_at_peak >= 0.0);

        // Halfway between observations, with a narrow kernel, uncertainty rises.
        let (_, sigma_between) = surrogate.predict(&[0.25]);
        assert!(
            sigma_between > sigma_at_peak,
            "uncertainty must grow away from the data: {sigma_between} vs {sigma_at_peak}"
        );
    }

    #[test]
    fn expected_improvement_is_non_negative_and_peaks_where_improvement_is_plausible() {
        let config = SurrogateConfig::default();
        let surrogate = KernelSurrogate::fit(
            vec![vec![0.0], vec![0.5], vec![1.0]],
            vec![0.0, 1.0, 0.0],
            &config,
        )
        .expect("fit");
        for unit in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let ei =
                surrogate.acquisition(&[unit], AcquisitionFunction::ExpectedImprovement, &config);
            assert!(ei >= 0.0, "EI at {unit} was negative: {ei}");
        }
        // PI is a genuine probability in [0, 1] — the crate used to return a
        // binary 0/1 step.
        for unit in [0.0, 0.5, 1.0] {
            let pi = surrogate.acquisition(
                &[unit],
                AcquisitionFunction::ProbabilityOfImprovement,
                &config,
            );
            assert!((0.0..=1.0).contains(&pi), "PI at {unit} was {pi}");
        }
    }

    #[test]
    fn each_acquisition_function_ranks_differently() {
        let config = SurrogateConfig::default();
        let surrogate = KernelSurrogate::fit(
            vec![vec![0.0], vec![0.5], vec![1.0]],
            vec![0.0, 1.0, 0.0],
            &config,
        )
        .expect("fit");
        let probe = [0.2];
        let values: Vec<f64> = [
            AcquisitionFunction::ExpectedImprovement,
            AcquisitionFunction::ProbabilityOfImprovement,
            AcquisitionFunction::UpperConfidenceBound,
            AcquisitionFunction::EntropySearch,
        ]
        .iter()
        .map(|acquisition| surrogate.acquisition(&probe, *acquisition, &config))
        .collect();
        // No two acquisition functions may collapse to the same number: that would
        // mean the enum variant is decorative.
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                assert!(
                    (values[i] - values[j]).abs() > 1e-12,
                    "acquisition {i} and {j} both returned {}",
                    values[i]
                );
            }
        }
    }

    #[test]
    fn bayesian_suggestions_concentrate_near_the_observed_optimum() {
        let configs = observations();
        let scored = scored(&configs);
        let space = unit_space();
        let config = SurrogateConfig::default();
        let mut rng = Random::seed(31_337);

        let mut suggestions = Vec::new();
        for _ in 0..30 {
            let (parameters, _) = suggest(
                &space,
                &scored,
                AcquisitionFunction::UpperConfidenceBound,
                &config,
                &mut rng,
            )
            .expect("suggest");
            suggestions.push(parameters["x"]);
        }
        let mean = suggestions.iter().sum::<f64>() / suggestions.len() as f64;
        assert!(
            mean > 0.6,
            "suggestions centred at {mean}; random search would centre at 0.5"
        );
        for value in &suggestions {
            assert!((0.0..=1.0).contains(value), "out of range: {value}");
        }
    }

    #[test]
    fn bayesian_refuses_to_run_before_the_initial_design_is_done() {
        let configs = observations();
        let short = scored(&configs)[..1].to_vec();
        let mut rng = Random::seed(2);
        let error = suggest(
            &unit_space(),
            &short,
            AcquisitionFunction::ExpectedImprovement,
            &SurrogateConfig::default(),
            &mut rng,
        )
        .expect_err("must not silently fall back to random search");
        assert!(
            format!("{error}").contains("Bayesian optimization needs at least"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn fitting_rejects_mismatched_or_insufficient_data() {
        let config = SurrogateConfig::default();
        assert!(KernelSurrogate::fit(vec![vec![0.0]], vec![0.0, 1.0], &config).is_err());
        assert!(KernelSurrogate::fit(vec![vec![0.0]], vec![0.0], &config).is_err());
        let ok = KernelSurrogate::fit(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0], &config)
            .expect("two observations suffice");
        assert_eq!(ok.observation_count(), 2);
        assert_eq!(ok.best_score(), 1.0);
    }
}
