//! Real grid enumeration for [`super::OptimizationStrategy::Grid`] (F15).
//!
//! `grid_search` used to be `self.random_search()` behind a
//! `// Simplified grid search implementation` comment, so selecting `Grid`
//! silently performed random search: no coverage guarantee, no reproducibility,
//! and repeated points.
//!
//! This module builds an explicit, deterministic axis per parameter and walks the
//! Cartesian product by index (mixed-radix decoding), so the *n*-th call returns
//! the *n*-th grid point and the whole grid is visited exactly once before it
//! wraps.

use super::support::{
    bounds, is_log_axis, sorted_categorical_names, sorted_parameter_names, StrategyRng,
    SuggestedParameters,
};
use super::{HyperparameterSpace, ParameterRange};
use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Default number of points per continuous axis when the parameter does not
/// declare `discrete_values`.
pub const DEFAULT_GRID_RESOLUTION: usize = 5;

/// Candidate values along one continuous axis.
///
/// * `discrete_values` (when present) *is* the axis — the declared set is used
///   verbatim, in declaration order.
/// * Otherwise the axis is `resolution` evenly spaced points spanning
///   `[min_value, max_value]` inclusive, spaced on the log axis when the range
///   declares `log_scale` with a positive lower bound.
/// * A degenerate range (`min == max`, or `resolution <= 1`) yields the single
///   lower bound rather than an empty axis, so the grid never collapses to zero
///   points.
pub fn axis_values<T: Float>(range: &ParameterRange<T>, resolution: usize) -> Vec<f64> {
    if let Some(values) = range.discrete_values.as_ref() {
        if !values.is_empty() {
            return values.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        }
    }

    let (lower, upper) = bounds(range);
    if resolution <= 1 || upper <= lower {
        return vec![lower];
    }

    let steps = resolution - 1;
    // The endpoints are emitted verbatim rather than recomputed: `exp(ln(1e-5))`
    // is `9.999999999999997e-6`, which falls *below* the declared minimum and
    // would make `HyperparameterSpace::validate_configuration` reject the very
    // first grid point.
    (0..resolution)
        .map(|i| {
            if i == 0 {
                lower
            } else if i == steps {
                upper
            } else if is_log_axis(range) {
                let log_lower = lower.ln();
                let log_upper = upper.ln();
                (log_lower + (log_upper - log_lower) * i as f64 / steps as f64).exp()
            } else {
                lower + (upper - lower) * i as f64 / steps as f64
            }
        })
        .collect()
}

/// Total number of distinct grid points, or `None` when the product overflows
/// `usize` (an honest "too large to enumerate" rather than a wrapped count).
pub fn grid_size<T: Float>(space: &HyperparameterSpace<T>, resolution: usize) -> Option<usize> {
    let mut total: usize = 1;
    for name in sorted_parameter_names(space) {
        let Some(range) = space.parameter_ranges().get(&name) else {
            continue;
        };
        total = total.checked_mul(axis_values(range, resolution).len().max(1))?;
    }
    for name in sorted_categorical_names(space) {
        let count = space
            .categorical_options()
            .get(&name)
            .map(|options| options.len())
            .unwrap_or(0)
            .max(1);
        total = total.checked_mul(count)?;
    }
    Some(total)
}

/// Decode grid point `index` into concrete parameter values.
///
/// Axes are ordered lexicographically by name (continuous axes first, then
/// categorical), and the index is decoded mixed-radix with the *last* axis
/// varying fastest. Indices at or beyond [`grid_size`] wrap, so a long run keeps
/// cycling the grid instead of erroring.
pub fn grid_point<T: Float>(
    space: &HyperparameterSpace<T>,
    index: usize,
    resolution: usize,
) -> Result<SuggestedParameters<T>> {
    let total = grid_size(space, resolution).ok_or_else(|| {
        OptimError::InvalidConfig(
            "grid search space is too large to enumerate (point count overflows usize)".to_string(),
        )
    })?;

    let mut parameters = HashMap::new();
    let mut categorical = HashMap::new();

    // Build the axis list once so the radix decoding and the value lookup agree.
    let mut continuous_axes: Vec<(String, Vec<f64>)> = Vec::new();
    for name in sorted_parameter_names(space) {
        if let Some(range) = space.parameter_ranges().get(&name) {
            let mut values = axis_values(range, resolution);
            if values.is_empty() {
                values.push(bounds(range).0);
            }
            continuous_axes.push((name, values));
        }
    }
    let mut categorical_axes: Vec<(String, Vec<String>)> = Vec::new();
    for name in sorted_categorical_names(space) {
        if let Some(options) = space.categorical_options().get(&name) {
            if !options.is_empty() {
                categorical_axes.push((name, options.clone()));
            }
        }
    }

    if continuous_axes.is_empty() && categorical_axes.is_empty() {
        return Err(OptimError::InvalidConfig(
            "grid search requires at least one parameter axis; the search space declares none"
                .to_string(),
        ));
    }
    let mut remaining = index % total.max(1);

    // Last axis varies fastest: divide out the trailing radices first.
    for (name, options) in categorical_axes.iter().rev() {
        let radix = options.len();
        let position = remaining % radix;
        remaining /= radix;
        categorical.insert(name.clone(), options[position].clone());
    }
    for (name, values) in continuous_axes.iter().rev() {
        let radix = values.len();
        let position = remaining % radix;
        remaining /= radix;
        parameters.insert(
            name.clone(),
            scirs2_core::numeric::NumCast::from(values[position]).unwrap_or_else(T::zero),
        );
    }

    Ok((parameters, categorical))
}

/// Grid search needs no randomness; the parameter exists so the strategy
/// dispatch in [`super`] can call every strategy through one signature.
pub fn suggest<T: Float>(
    space: &HyperparameterSpace<T>,
    index: usize,
    resolution: usize,
    _rng: &mut StrategyRng,
) -> Result<SuggestedParameters<T>> {
    grid_point(space, index, resolution)
}

#[cfg(test)]
mod tests {
    use super::super::DistributionType;
    use super::*;
    use scirs2_core::random::Random;

    fn space_with_two_axes() -> HyperparameterSpace<f64> {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        space.add_parameter(
            "alpha".to_string(),
            ParameterRange {
                name: "alpha".to_string(),
                min_value: 0.0,
                max_value: 1.0,
                distribution: DistributionType::Uniform,
                log_scale: false,
                discrete_values: None,
            },
        );
        space.add_categorical_parameter(
            "optimizer".to_string(),
            vec!["adam".to_string(), "sgd".to_string()],
        );
        space
    }

    #[test]
    fn continuous_axis_spans_the_range_inclusively() {
        let range = ParameterRange {
            name: "x".to_string(),
            min_value: 0.0,
            max_value: 1.0,
            distribution: DistributionType::Uniform,
            log_scale: false,
            discrete_values: None,
        };
        let values = axis_values(&range, 5);
        assert_eq!(values, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn log_axis_is_geometrically_spaced() {
        let range = ParameterRange {
            name: "lr".to_string(),
            min_value: 1e-4,
            max_value: 1e-1,
            distribution: DistributionType::Uniform,
            log_scale: true,
            discrete_values: None,
        };
        let values = axis_values(&range, 4);
        assert_eq!(values.len(), 4);
        for expected in [1e-4, 1e-3, 1e-2, 1e-1] {
            assert!(
                values.iter().any(|v| (v - expected).abs() < 1e-12),
                "missing {expected} in {values:?}"
            );
        }
    }

    #[test]
    fn discrete_values_define_the_axis_verbatim() {
        let range = ParameterRange {
            name: "batch".to_string(),
            min_value: 8.0,
            max_value: 128.0,
            distribution: DistributionType::Uniform,
            log_scale: false,
            discrete_values: Some(vec![8.0, 32.0, 128.0]),
        };
        assert_eq!(axis_values(&range, 17), vec![8.0, 32.0, 128.0]);
    }

    #[test]
    fn degenerate_range_yields_exactly_one_point() {
        let range = ParameterRange {
            name: "fixed".to_string(),
            min_value: 3.0,
            max_value: 3.0,
            distribution: DistributionType::Uniform,
            log_scale: false,
            discrete_values: None,
        };
        assert_eq!(axis_values(&range, 5), vec![3.0]);
    }

    #[test]
    fn the_grid_is_enumerated_exhaustively_and_without_repeats() {
        let space = space_with_two_axes();
        let resolution = 3;
        // 3 continuous points x 2 categories.
        assert_eq!(grid_size(&space, resolution), Some(6));

        let mut seen = std::collections::HashSet::new();
        for index in 0..6 {
            let (parameters, categorical) =
                grid_point(&space, index, resolution).expect("grid point");
            let key = (
                parameters["alpha"].to_bits(),
                categorical["optimizer"].clone(),
            );
            assert!(seen.insert(key), "grid point {index} repeated");
        }
        assert_eq!(seen.len(), 6, "the grid must be covered exactly once");

        // A random search over 6 draws would almost never cover all 6 cells; the
        // grid must, and it must then wrap deterministically.
        let (first, first_cat) = grid_point(&space, 0, resolution).expect("point");
        let (wrapped, wrapped_cat) = grid_point(&space, 6, resolution).expect("point");
        assert_eq!(first, wrapped);
        assert_eq!(first_cat, wrapped_cat);
    }

    #[test]
    fn grid_points_are_reproducible_and_rng_independent() {
        let space = space_with_two_axes();
        let mut rng_a = Random::seed(1);
        let mut rng_b = Random::seed(999_999);
        let a = suggest(&space, 4, 3, &mut rng_a).expect("suggest");
        let b = suggest(&space, 4, 3, &mut rng_b).expect("suggest");
        assert_eq!(a, b, "grid search must not depend on the RNG at all");
    }

    #[test]
    fn an_empty_space_is_an_error_not_a_random_draw() {
        let space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        let mut rng = Random::seed(1);
        assert!(suggest(&space, 0, 3, &mut rng).is_err());
    }
}
