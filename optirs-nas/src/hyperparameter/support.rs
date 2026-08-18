//! Shared helpers used by every hyperparameter search strategy: deterministic
//! parameter ordering, log-aware normalization, and single-parameter sampling.
//!
//! Deterministic ordering matters more than it looks: the search space stores its
//! parameters in a `HashMap`, so anything that walks them (grid enumeration, the
//! normalized vector a surrogate model consumes) must impose an explicit order or
//! it produces a different answer on every run.

use super::{DistributionType, HyperparameterSpace, ParameterRange};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;

/// A seeded RNG of the concrete type every strategy in this module uses.
pub type StrategyRng = Random<scirs2_core::random::rngs::StdRng>;

/// What a strategy returns: the continuous parameter assignment plus the
/// categorical one.
pub type SuggestedParameters<T> = (
    std::collections::HashMap<String, T>,
    std::collections::HashMap<String, String>,
);

/// Continuous parameter names in a fixed (lexicographic) order.
pub fn sorted_parameter_names<T: Float>(space: &HyperparameterSpace<T>) -> Vec<String> {
    let mut names: Vec<String> = space.parameter_ranges().keys().cloned().collect();
    names.sort();
    names
}

/// Categorical parameter names in a fixed (lexicographic) order.
pub fn sorted_categorical_names<T: Float>(space: &HyperparameterSpace<T>) -> Vec<String> {
    let mut names: Vec<String> = space.categorical_options().keys().cloned().collect();
    names.sort();
    names
}

/// Inclusive `[min, max]` bounds of `range` as `f64`, ordered so `lower <= upper`
/// even if the range was declared backwards.
pub fn bounds<T: Float>(range: &ParameterRange<T>) -> (f64, f64) {
    let a = range.min_value.to_f64().unwrap_or(0.0);
    let b = range.max_value.to_f64().unwrap_or(0.0);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Whether `range` should be treated on a logarithmic axis. A log axis requires a
/// strictly positive lower bound; a declared `log_scale` with a non-positive bound
/// is silently unusable, so it is reported as linear instead of producing `NaN`.
pub fn is_log_axis<T: Float>(range: &ParameterRange<T>) -> bool {
    let (lower, _) = bounds(range);
    range.log_scale && lower > 0.0
}

/// Map a raw value onto `[0, 1]` along the parameter's own axis (log-aware).
/// Degenerate ranges collapse to `0.0`.
pub fn normalize<T: Float>(range: &ParameterRange<T>, value: f64) -> f64 {
    let (lower, upper) = bounds(range);
    if is_log_axis(range) {
        let log_lower = lower.ln();
        let log_upper = upper.ln();
        if log_upper <= log_lower {
            return 0.0;
        }
        ((value.max(f64::MIN_POSITIVE).ln() - log_lower) / (log_upper - log_lower)).clamp(0.0, 1.0)
    } else {
        if upper <= lower {
            return 0.0;
        }
        ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    }
}

/// Inverse of [`normalize`]: map `unit` in `[0, 1]` back to a raw value.
pub fn denormalize<T: Float>(range: &ParameterRange<T>, unit: f64) -> f64 {
    let (lower, upper) = bounds(range);
    let unit = unit.clamp(0.0, 1.0);
    // Endpoints are returned verbatim: `exp(ln(lower))` drifts below `lower` for
    // typical learning-rate ranges, which would fail range validation.
    if unit <= 0.0 {
        return lower;
    }
    if unit >= 1.0 {
        return upper;
    }
    if is_log_axis(range) {
        let log_lower = lower.ln();
        let log_upper = upper.ln();
        (log_lower + unit * (log_upper - log_lower)).exp()
    } else {
        lower + unit * (upper - lower)
    }
}

/// Snap a raw value onto the nearest entry of `discrete_values` when the range
/// declares one, so a strategy that reasons in continuous space still emits a
/// legal value.
pub fn snap_to_discrete<T: Float>(range: &ParameterRange<T>, value: f64) -> f64 {
    let Some(values) = range.discrete_values.as_ref() else {
        return value;
    };
    let mut best = value;
    let mut best_distance = f64::INFINITY;
    for candidate in values {
        let candidate = candidate.to_f64().unwrap_or(0.0);
        let distance = (candidate - value).abs();
        if distance < best_distance {
            best_distance = distance;
            best = candidate;
        }
    }
    best
}

/// Draw one value for `range` from its declared distribution.
///
/// Every [`DistributionType`] is sampled for real; none of them silently
/// degrades to uniform, which is what the previous `_ => uniform` arm did for
/// four of the five variants. All draws are transformed back into
/// `[min_value, max_value]`, so a configuration is always in-range.
pub fn sample_parameter<T: Float>(range: &ParameterRange<T>, rng: &mut StrategyRng) -> f64 {
    if let Some(values) = range.discrete_values.as_ref() {
        if !values.is_empty() {
            let idx = rng.gen_range(0..values.len());
            return values[idx].to_f64().unwrap_or(0.0);
        }
    }

    let unit = match range.distribution {
        DistributionType::Uniform => rng.gen_range(0.0..1.0),
        DistributionType::Normal => {
            // Truncated standard normal mapped onto the unit interval: mean at the
            // range midpoint, one standard deviation covering a quarter of the
            // range, resampled (not clamped) so the shape is not distorted by a
            // spike at the bounds.
            let mut unit = 0.5 + standard_normal(rng) * 0.25;
            let mut attempts = 0;
            while !(0.0..=1.0).contains(&unit) && attempts < 16 {
                unit = 0.5 + standard_normal(rng) * 0.25;
                attempts += 1;
            }
            unit.clamp(0.0, 1.0)
        }
        DistributionType::LogNormal => {
            // exp(N(0,1)) squashed into the unit interval by its own CDF-like
            // normalization: values concentrate near the lower bound with a long
            // upper tail, which is the point of a log-normal prior.
            let raw = standard_normal(rng).exp();
            (raw / (raw + 1.0)).clamp(0.0, 1.0)
        }
        DistributionType::Beta => {
            // Beta(2,2) via the order statistic of three uniforms: a symmetric,
            // centre-weighted draw on [0,1].
            let mut draws = [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ];
            draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            draws[1]
        }
        DistributionType::Exponential => {
            // Exponential(1) truncated to [0,1) by inverse transform.
            let u = rng.gen_range(0.0..1.0_f64).max(f64::MIN_POSITIVE);
            let raw = -u.ln();
            (raw / (raw + 1.0)).clamp(0.0, 1.0)
        }
    };
    denormalize(range, unit)
}

/// Box-Muller standard normal draw.
pub fn standard_normal(rng: &mut StrategyRng) -> f64 {
    let u1 = rng.gen_range(0.0..1.0_f64).max(f64::MIN_POSITIVE);
    let u2 = rng.gen_range(0.0..1.0_f64);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_range() -> ParameterRange<f64> {
        ParameterRange {
            name: "x".to_string(),
            min_value: 2.0,
            max_value: 6.0,
            distribution: DistributionType::Uniform,
            log_scale: false,
            discrete_values: None,
        }
    }

    fn log_range() -> ParameterRange<f64> {
        ParameterRange {
            name: "lr".to_string(),
            min_value: 1e-4,
            max_value: 1e-1,
            distribution: DistributionType::Uniform,
            log_scale: true,
            discrete_values: None,
        }
    }

    #[test]
    fn normalization_round_trips_on_both_axes() {
        for range in [linear_range(), log_range()] {
            for unit in [0.0, 0.1, 0.5, 0.9, 1.0] {
                let raw = denormalize(&range, unit);
                let back = normalize(&range, raw);
                assert!(
                    (back - unit).abs() < 1e-9,
                    "unit {unit} round-tripped to {back}"
                );
            }
        }
    }

    #[test]
    fn log_axis_is_ignored_for_non_positive_bounds() {
        let range = ParameterRange {
            name: "x".to_string(),
            min_value: -1.0,
            max_value: 1.0,
            distribution: DistributionType::Uniform,
            log_scale: true,
            discrete_values: None,
        };
        assert!(!is_log_axis(&range));
        let value = denormalize(&range, 0.5);
        assert!(value.is_finite(), "got {value}");
        assert!((value - 0.0).abs() < 1e-12);
    }

    #[test]
    fn every_distribution_stays_in_range_and_is_not_uniform_by_default() {
        let mut rng = Random::seed(1234);
        for distribution in [
            DistributionType::Uniform,
            DistributionType::Normal,
            DistributionType::LogNormal,
            DistributionType::Beta,
            DistributionType::Exponential,
        ] {
            let range = ParameterRange {
                name: "x".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                distribution,
                log_scale: false,
                discrete_values: None,
            };
            let samples: Vec<f64> = (0..2000)
                .map(|_| sample_parameter(&range, &mut rng))
                .collect();
            for value in &samples {
                assert!(
                    (0.0..=1.0).contains(value),
                    "{distribution:?} produced {value}"
                );
            }
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            match distribution {
                // The non-uniform shapes must genuinely differ from uniform.
                DistributionType::Exponential => {
                    assert!(mean < 0.45, "Exponential mean {mean} looks uniform")
                }
                DistributionType::Uniform => {
                    assert!((mean - 0.5).abs() < 0.05, "uniform mean {mean}")
                }
                _ => {}
            }
        }
    }

    #[test]
    fn discrete_ranges_only_ever_emit_declared_values() {
        let mut rng = Random::seed(9);
        let range = ParameterRange {
            name: "batch".to_string(),
            min_value: 8.0,
            max_value: 128.0,
            distribution: DistributionType::Uniform,
            log_scale: false,
            discrete_values: Some(vec![8.0, 16.0, 32.0, 64.0, 128.0]),
        };
        for _ in 0..200 {
            let value = sample_parameter(&range, &mut rng);
            assert!(
                [8.0, 16.0, 32.0, 64.0, 128.0].contains(&value),
                "got {value}"
            );
        }
        assert_eq!(snap_to_discrete(&range, 30.0), 32.0);
        assert_eq!(snap_to_discrete(&range, 1000.0), 128.0);
    }

    #[test]
    fn ordering_helpers_are_deterministic() {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        for name in ["zeta", "alpha", "mu"] {
            space.add_parameter(
                name.to_string(),
                ParameterRange {
                    name: name.to_string(),
                    min_value: 0.0,
                    max_value: 1.0,
                    distribution: DistributionType::Uniform,
                    log_scale: false,
                    discrete_values: None,
                },
            );
        }
        space.add_categorical_parameter("opt".to_string(), vec!["adam".to_string()]);
        space.add_categorical_parameter("act".to_string(), vec!["relu".to_string()]);

        assert_eq!(sorted_parameter_names(&space), vec!["alpha", "mu", "zeta"]);
        assert_eq!(sorted_categorical_names(&space), vec!["act", "opt"]);
    }
}
