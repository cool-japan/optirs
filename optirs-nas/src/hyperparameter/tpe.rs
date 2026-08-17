//! Tree-structured Parzen Estimator (TPE) for
//! [`super::OptimizationStrategy::TPE`] (F15).
//!
//! `tpe_search` used to be `self.random_search()` behind a
//! `// Simplified TPE implementation` comment.
//!
//! This is the real algorithm (Bergstra, Bardenet, Bengio & Kegl, *Algorithms for
//! Hyper-Parameter Optimization*, NeurIPS 2011): the observation history is split
//! by score quantile into a "good" set and a "bad" set, **two** adaptive
//! kernel-density estimators `l(x)` (good) and `g(x)` (bad) are fitted per
//! parameter, candidates are drawn from `l(x)`, and the candidate maximizing the
//! density ratio `l(x)/g(x)` — which is monotone in Expected Improvement — is
//! returned.
//!
//! Everything happens in the parameter's own normalized `[0, 1]` axis (log-aware,
//! see [`super::support`]), so a learning rate spanning four decades is modelled on
//! the scale it is actually searched on.

use super::support::{
    denormalize, normalize, sample_parameter, snap_to_discrete, sorted_categorical_names,
    sorted_parameter_names, standard_normal, StrategyRng, SuggestedParameters,
};
use super::{HyperparameterConfiguration, HyperparameterSpace};
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Tunables of the estimator. The defaults follow the published algorithm and the
/// values used by mainstream implementations.
#[derive(Debug, Clone, Copy)]
pub struct TpeConfig {
    /// Observations required before the estimator is used at all. Below this the
    /// caller must draw from the prior (a random *initial design*) — this is part
    /// of the published algorithm, not a fallback: two density estimates cannot be
    /// fitted to an empty history.
    pub startup_trials: usize,
    /// Fraction of observations placed in the "good" set.
    pub gamma: f64,
    /// Hard cap on the size of the good set.
    pub max_good: usize,
    /// Number of candidates drawn from `l(x)` per parameter and ranked by the
    /// density ratio.
    pub candidates: usize,
    /// Weight given to the uniform prior component of each KDE, relative to one
    /// observation. Keeps `g(x)` strictly positive everywhere so the ratio never
    /// divides by zero.
    pub prior_weight: f64,
    /// Lower bound on a kernel bandwidth, as a fraction of the normalized axis.
    pub min_bandwidth: f64,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            startup_trials: 10,
            gamma: 0.25,
            max_good: 25,
            candidates: 24,
            prior_weight: 1.0,
            min_bandwidth: 0.05,
        }
    }
}

impl TpeConfig {
    /// Size of the "good" set for `n` observations: `ceil(gamma * n)`, at least 1
    /// and at most `max_good`, and always leaving at least one observation for the
    /// "bad" set so both estimators are non-degenerate.
    pub fn good_count(&self, n: usize) -> usize {
        if n < 2 {
            return n.min(1);
        }
        let raw = (self.gamma * n as f64).ceil() as usize;
        raw.clamp(1, self.max_good.min(n - 1))
    }
}

/// A one-dimensional adaptive Parzen estimator: a mixture of Gaussians, one per
/// observation, plus a wide uniform-like prior component centred on the axis.
///
/// Bandwidths are *adaptive* in the sense of the paper: each observation's
/// bandwidth is the larger of its distances to the neighbouring observations
/// (after sorting), floored at `min_bandwidth`. Dense regions therefore get
/// narrow kernels and isolated observations get wide ones.
#[derive(Debug, Clone)]
pub struct ParzenEstimator {
    /// Kernel centres on the normalized `[0, 1]` axis.
    centres: Vec<f64>,
    /// Per-centre bandwidths.
    bandwidths: Vec<f64>,
    /// Weight of the prior component relative to a single observation.
    prior_weight: f64,
}

impl ParzenEstimator {
    /// Fit an estimator to `observations` (already normalized to `[0, 1]`).
    pub fn fit(observations: &[f64], config: &TpeConfig) -> Self {
        let mut sorted: Vec<f64> = observations
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut bandwidths = Vec::with_capacity(sorted.len());
        for i in 0..sorted.len() {
            let left = if i == 0 {
                sorted[i] - 0.0
            } else {
                sorted[i] - sorted[i - 1]
            };
            let right = if i + 1 == sorted.len() {
                1.0 - sorted[i]
            } else {
                sorted[i + 1] - sorted[i]
            };
            bandwidths.push(left.max(right).max(config.min_bandwidth));
        }

        Self {
            centres: sorted,
            bandwidths,
            prior_weight: config.prior_weight.max(0.0),
        }
    }

    /// Density at `x`. Always strictly positive thanks to the prior component, so
    /// it is safe as the denominator of the TPE ratio.
    pub fn density(&self, x: f64) -> f64 {
        // Prior: a broad Gaussian centred on the axis midpoint with unit-scale
        // spread, which behaves like a flat prior over [0, 1].
        let mut total = self.prior_weight * gaussian(x, 0.5, 1.0);
        let mut weight = self.prior_weight;
        for (centre, bandwidth) in self.centres.iter().zip(self.bandwidths.iter()) {
            total += gaussian(x, *centre, *bandwidth);
            weight += 1.0;
        }
        if weight <= 0.0 {
            return f64::MIN_POSITIVE;
        }
        (total / weight).max(f64::MIN_POSITIVE)
    }

    /// Draw a sample from the mixture: pick a component by weight, then draw from
    /// that Gaussian, resampling until the draw lands inside `[0, 1]`.
    pub fn sample(&self, rng: &mut StrategyRng) -> f64 {
        let total_weight = self.prior_weight + self.centres.len() as f64;
        if total_weight <= 0.0 {
            return rng.gen_range(0.0..1.0);
        }
        for _ in 0..32 {
            let pick = rng.gen_range(0.0..total_weight);
            let (mean, sigma) = if pick < self.prior_weight {
                (0.5, 1.0)
            } else {
                let index = ((pick - self.prior_weight) as usize).min(self.centres.len() - 1);
                (self.centres[index], self.bandwidths[index])
            };
            let draw = mean + standard_normal(rng) * sigma;
            if (0.0..=1.0).contains(&draw) {
                return draw;
            }
        }
        rng.gen_range(0.0..1.0)
    }

    /// Number of observations backing this estimator.
    pub fn observation_count(&self) -> usize {
        self.centres.len()
    }
}

fn gaussian(x: f64, mean: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1e-12);
    let z = (x - mean) / sigma;
    (-0.5 * z * z).exp() / (sigma * (std::f64::consts::TAU).sqrt())
}

/// Categorical density: smoothed category frequencies.
#[derive(Debug, Clone)]
pub struct CategoricalEstimator {
    weights: Vec<f64>,
    total: f64,
}

impl CategoricalEstimator {
    /// Fit to observed category indices over `num_categories`, with the prior
    /// weight spread uniformly (Laplace-style smoothing) so no category has zero
    /// probability.
    pub fn fit(observed: &[usize], num_categories: usize, config: &TpeConfig) -> Self {
        let smoothing = if num_categories > 0 {
            config.prior_weight.max(0.0) / num_categories as f64
        } else {
            0.0
        };
        let mut weights = vec![smoothing; num_categories];
        for index in observed {
            if *index < num_categories {
                weights[*index] += 1.0;
            }
        }
        let total = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        Self { weights, total }
    }

    /// Probability of `index`.
    pub fn probability(&self, index: usize) -> f64 {
        self.weights
            .get(index)
            .map(|w| (w / self.total).max(f64::MIN_POSITIVE))
            .unwrap_or(f64::MIN_POSITIVE)
    }

    /// Draw a category index proportional to the fitted weights.
    pub fn sample(&self, rng: &mut StrategyRng) -> usize {
        if self.weights.is_empty() {
            return 0;
        }
        let target = rng.gen_range(0.0..self.total);
        let mut cumulative = 0.0;
        for (index, weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                return index;
            }
        }
        self.weights.len() - 1
    }
}

/// Split `scored` observations into the good and bad sets by score, higher score
/// being better (which is the convention
/// [`super::HyperparameterOptimizer::record_evaluation`] uses).
///
/// Returns `(good, bad)` as index slices into `scored`.
pub fn split_by_quantile<T: Float>(
    scored: &[(&HyperparameterConfiguration<T>, f64)],
    config: &TpeConfig,
) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..scored.len()).collect();
    // Descending by score: the best observations come first.
    order.sort_by(|a, b| {
        scored[*b]
            .1
            .partial_cmp(&scored[*a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let good_count = config.good_count(scored.len());
    let good = order[..good_count.min(order.len())].to_vec();
    let bad = order[good_count.min(order.len())..].to_vec();
    (good, bad)
}

/// Suggest the next configuration with the real TPE algorithm.
///
/// Returns [`OptimError::InvalidConfig`] when there are fewer than
/// `config.startup_trials` scored observations: the caller is responsible for the
/// random *initial design* phase, and TPE says so out loud instead of quietly
/// becoming random search.
pub fn suggest<T: Float>(
    space: &HyperparameterSpace<T>,
    scored: &[(&HyperparameterConfiguration<T>, f64)],
    config: &TpeConfig,
    rng: &mut StrategyRng,
) -> Result<SuggestedParameters<T>> {
    if scored.len() < config.startup_trials.max(2) {
        return Err(OptimError::InvalidConfig(format!(
            "TPE needs at least {} scored observations to fit its good/bad density \
             estimators; only {} are available (run the random initial design first)",
            config.startup_trials.max(2),
            scored.len()
        )));
    }

    let (good, bad) = split_by_quantile(scored, config);
    if good.is_empty() || bad.is_empty() {
        return Err(OptimError::InvalidConfig(
            "TPE could not form both a good and a bad observation set; scores may all be equal"
                .to_string(),
        ));
    }

    let mut parameters = HashMap::new();
    for name in sorted_parameter_names(space) {
        let Some(range) = space.parameter_ranges().get(&name) else {
            continue;
        };
        let collect = |indices: &[usize]| -> Vec<f64> {
            indices
                .iter()
                .filter_map(|&i| scored[i].0.parameters.get(&name))
                .map(|v| normalize(range, v.to_f64().unwrap_or(0.0)))
                .collect()
        };
        let good_values = collect(&good);
        let bad_values = collect(&bad);
        if good_values.is_empty() {
            // No observation recorded this parameter; sample it from its prior.
            let raw = snap_to_discrete(range, sample_parameter(range, rng));
            parameters.insert(
                name,
                scirs2_core::numeric::NumCast::from(raw).unwrap_or_else(T::zero),
            );
            continue;
        }

        let good_kde = ParzenEstimator::fit(&good_values, config);
        let bad_kde = ParzenEstimator::fit(&bad_values, config);

        // Draw candidates from l(x) and keep the one maximizing l(x)/g(x).
        let mut best_unit = good_kde.sample(rng);
        let mut best_ratio = f64::NEG_INFINITY;
        for _ in 0..config.candidates.max(1) {
            let unit = good_kde.sample(rng);
            let ratio = good_kde.density(unit).ln() - bad_kde.density(unit).ln();
            if ratio > best_ratio {
                best_ratio = ratio;
                best_unit = unit;
            }
        }
        let raw = snap_to_discrete(range, denormalize(range, best_unit));
        parameters.insert(
            name,
            scirs2_core::numeric::NumCast::from(raw).unwrap_or_else(T::zero),
        );
    }

    let mut categorical = HashMap::new();
    for name in sorted_categorical_names(space) {
        let Some(options) = space.categorical_options().get(&name) else {
            continue;
        };
        if options.is_empty() {
            continue;
        }
        let index_of = |value: &str| options.iter().position(|option| option == value);
        let collect = |indices: &[usize]| -> Vec<usize> {
            indices
                .iter()
                .filter_map(|&i| scored[i].0.categorical_parameters.get(&name))
                .filter_map(|value| index_of(value))
                .collect()
        };
        let good_counts = CategoricalEstimator::fit(&collect(&good), options.len(), config);
        let bad_counts = CategoricalEstimator::fit(&collect(&bad), options.len(), config);

        let mut best_index = good_counts.sample(rng);
        let mut best_ratio = f64::NEG_INFINITY;
        for _ in 0..config.candidates.max(1) {
            let index = good_counts.sample(rng);
            let ratio = good_counts.probability(index).ln() - bad_counts.probability(index).ln();
            if ratio > best_ratio {
                best_ratio = ratio;
                best_index = index;
            }
        }
        categorical.insert(name, options[best_index.min(options.len() - 1)].clone());
    }

    Ok((parameters, categorical))
}

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
        space.add_categorical_parameter(
            "kind".to_string(),
            vec!["good_kind".to_string(), "bad_kind".to_string()],
        );
        space
    }

    /// Observations whose score is high exactly when `x` is near 0.8 and the
    /// category is `good_kind`.
    fn observations() -> Vec<HyperparameterConfiguration<f64>> {
        let mut configs = Vec::new();
        for i in 0..30 {
            let x = i as f64 / 29.0;
            let good_kind = i % 2 == 0;
            let score = -(x - 0.8).abs() + if good_kind { 0.5 } else { 0.0 };
            let mut parameters = HashMap::new();
            parameters.insert("x".to_string(), x);
            let mut categorical = HashMap::new();
            categorical.insert(
                "kind".to_string(),
                if good_kind { "good_kind" } else { "bad_kind" }.to_string(),
            );
            configs.push(HyperparameterConfiguration {
                id: format!("obs_{i}"),
                parameters,
                categorical_parameters: categorical,
                score: Some(score),
                metadata: HashMap::new(),
            });
        }
        configs
    }

    fn scored(
        configs: &[HyperparameterConfiguration<f64>],
    ) -> Vec<(&HyperparameterConfiguration<f64>, f64)> {
        configs
            .iter()
            .map(|c| (c, c.score.unwrap_or(0.0)))
            .collect()
    }

    #[test]
    fn parzen_density_is_higher_near_observations() {
        let config = TpeConfig::default();
        let estimator = ParzenEstimator::fit(&[0.8, 0.82, 0.79], &config);
        let near = estimator.density(0.8);
        let far = estimator.density(0.1);
        assert!(near > far, "near = {near}, far = {far}");
        assert!(
            far > 0.0,
            "the prior must keep the density strictly positive"
        );
    }

    #[test]
    fn parzen_bandwidths_are_adaptive() {
        let config = TpeConfig {
            min_bandwidth: 0.001,
            ..TpeConfig::default()
        };
        // A tight cluster around 0.51 embedded among sparse points. The published
        // rule sets each bandwidth to the larger of the two neighbour gaps, so the
        // interior of the cluster gets a narrow kernel and the sparse tails get
        // wide ones.
        let estimator = ParzenEstimator::fit(&[0.0, 0.40, 0.50, 0.51, 0.52, 1.0], &config);
        assert_eq!(estimator.observation_count(), 6);
        let cluster_interior = estimator.bandwidths[3];
        assert!(
            cluster_interior < estimator.bandwidths[0],
            "cluster bandwidth {cluster_interior} should be below the left tail's {}",
            estimator.bandwidths[0]
        );
        assert!(
            cluster_interior < estimator.bandwidths[5],
            "cluster bandwidth {cluster_interior} should be below the right tail's {}",
            estimator.bandwidths[5]
        );
        // Every bandwidth respects the configured floor.
        for bandwidth in &estimator.bandwidths {
            assert!(*bandwidth >= config.min_bandwidth);
        }
    }

    #[test]
    fn quantile_split_puts_the_best_observations_in_the_good_set() {
        let configs = observations();
        let scored = scored(&configs);
        let config = TpeConfig::default();
        let (good, bad) = split_by_quantile(&scored, &config);

        assert!(!good.is_empty() && !bad.is_empty());
        let worst_good = good
            .iter()
            .map(|&i| scored[i].1)
            .fold(f64::INFINITY, f64::min);
        let best_bad = bad
            .iter()
            .map(|&i| scored[i].1)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            worst_good >= best_bad,
            "the good set must dominate the bad set: {worst_good} vs {best_bad}"
        );
        assert_eq!(good.len(), config.good_count(scored.len()));
    }

    #[test]
    fn tpe_concentrates_on_the_high_scoring_region() {
        let configs = observations();
        let scored = scored(&configs);
        let config = TpeConfig::default();
        let space = unit_space();
        let mut rng = Random::seed(20_240_817);

        let mut suggestions = Vec::new();
        for _ in 0..40 {
            let (parameters, categorical) =
                suggest(&space, &scored, &config, &mut rng).expect("tpe suggestion");
            suggestions.push((parameters["x"], categorical["kind"].clone()));
        }

        let mean_x = suggestions.iter().map(|(x, _)| *x).sum::<f64>() / suggestions.len() as f64;
        // Uniform random search would centre on 0.5; TPE must pull toward 0.8.
        assert!(
            mean_x > 0.6,
            "TPE suggestions centred at {mean_x}, which is indistinguishable from random search"
        );

        let good_fraction = suggestions
            .iter()
            .filter(|(_, kind)| kind == "good_kind")
            .count() as f64
            / suggestions.len() as f64;
        assert!(
            good_fraction > 0.65,
            "TPE picked the rewarding category only {:.0}% of the time",
            good_fraction * 100.0
        );
    }

    #[test]
    fn tpe_refuses_to_run_before_the_initial_design_is_done() {
        let configs = observations();
        let short = scored(&configs)[..3].to_vec();
        let mut rng = Random::seed(1);
        let error = suggest(&unit_space(), &short, &TpeConfig::default(), &mut rng)
            .expect_err("TPE must not silently fall back to random search");
        assert!(
            format!("{error}").contains("TPE needs at least"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn tpe_is_reproducible_for_a_given_seed() {
        let configs = observations();
        let scored = scored(&configs);
        let space = unit_space();
        let config = TpeConfig::default();
        let mut a = Random::seed(4242);
        let mut b = Random::seed(4242);
        let first = suggest(&space, &scored, &config, &mut a).expect("suggest");
        let second = suggest(&space, &scored, &config, &mut b).expect("suggest");
        assert_eq!(first, second);
    }

    #[test]
    fn categorical_estimator_smooths_unseen_categories() {
        let config = TpeConfig::default();
        let estimator = CategoricalEstimator::fit(&[0, 0, 0], 3, &config);
        assert!(estimator.probability(0) > estimator.probability(1));
        assert!(
            estimator.probability(2) > 0.0,
            "an unseen category must keep a non-zero probability"
        );
        assert!(
            estimator.probability(99) > 0.0,
            "out of range must not panic"
        );
    }

    #[test]
    fn tpe_respects_a_log_axis() {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        space.add_parameter(
            "lr".to_string(),
            ParameterRange {
                name: "lr".to_string(),
                min_value: 1e-5,
                max_value: 1e-1,
                distribution: DistributionType::Uniform,
                log_scale: true,
                discrete_values: None,
            },
        );

        // Reward small learning rates.
        let configs: Vec<HyperparameterConfiguration<f64>> = (0..24)
            .map(|i| {
                let lr = 1e-5 * 10f64.powf(4.0 * i as f64 / 23.0);
                let mut parameters = HashMap::new();
                parameters.insert("lr".to_string(), lr);
                HyperparameterConfiguration {
                    id: format!("o{i}"),
                    parameters,
                    categorical_parameters: HashMap::new(),
                    score: Some(-lr.log10()),
                    metadata: HashMap::new(),
                }
            })
            .collect();
        let scored: Vec<(&HyperparameterConfiguration<f64>, f64)> = configs
            .iter()
            .map(|c| (c, c.score.unwrap_or(0.0)))
            .collect();

        let mut rng = Random::seed(77);
        let mut values = Vec::new();
        for _ in 0..30 {
            let (parameters, _) =
                suggest(&space, &scored, &TpeConfig::default(), &mut rng).expect("suggest");
            let lr = parameters["lr"];
            assert!((1e-5..=1e-1).contains(&lr), "lr {lr} escaped the range");
            values.push(lr);
        }
        let median_exponent = {
            let mut exponents: Vec<f64> = values.iter().map(|v| v.log10()).collect();
            exponents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            exponents[exponents.len() / 2]
        };
        // Log-uniform random search would centre at 10^-3.
        assert!(
            median_exponent < -3.3,
            "TPE median exponent {median_exponent} shows no preference for small rates"
        );
    }
}
