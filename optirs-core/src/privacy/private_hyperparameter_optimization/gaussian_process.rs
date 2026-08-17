//! Gaussian-process surrogate and acquisition function for private Bayesian
//! optimization.
//!
//! # The defect this replaces
//!
//! `PrivateBayesianOptimization::suggest_next` returned
//! `ParameterConfiguration { values: HashMap::new(), .. }` -- a configuration
//! with **no parameters at all** -- as soon as the evaluation history was
//! non-empty, and `update` was `Ok(())`, discarding every result. There was no
//! Gaussian process, no kernel and no acquisition function, and the cold-start
//! branch collapsed every non-continuous parameter to the constant `0.5`.
//!
//! # What is implemented
//!
//! * [`ConfigurationEncoding`]: a stable, invertible encoding of a
//!   [`ParameterSpace`] into `[0, 1]^d`, covering all five parameter kinds.
//! * [`GaussianProcessFit`]: exact GP regression with an RBF kernel, solved
//!   through a Cholesky factorisation of `K + sigma_n^2 I` (no iterative
//!   approximation, no external BLAS).
//! * [`ExpectedImprovement`]: the standard EI acquisition function.
//!
//! # Why the surrogate costs no privacy budget
//!
//! The GP is fitted on the objective values that were **already released under
//! differential privacy** by [`super::types::PrivateObjective::evaluate`].
//! Fitting a model to DP outputs is post-processing, which the DP guarantee is
//! closed under, so the surrogate and the acquisition maximisation consume no
//! additional epsilon. The observation noise the GP is told about is derived
//! from the DP noise scale that was actually applied, so the surrogate does not
//! over-trust the releases.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::{HpoRng, ParameterConfiguration, ParameterSpace, ParameterType, ParameterValue};

/// Encoding scheme for one hyperparameter.
#[derive(Debug, Clone, PartialEq)]
enum Scheme {
    /// Min-max scaled real value.
    Numeric {
        /// Lower bound of the declared range.
        min: f64,
        /// Upper bound of the declared range.
        max: f64,
    },
    /// Min-max scaled integer value.
    Integer {
        /// Lower bound of the declared range.
        min: i64,
        /// Upper bound of the declared range.
        max: i64,
    },
    /// Boolean, encoded as 0 or 1.
    Boolean,
    /// Categorical, encoded as the position in the declared order.
    Categorical {
        /// The declared categories, in order.
        values: Vec<String>,
    },
    /// Ordinal, encoded as the position in the declared order.
    Ordinal {
        /// Number of declared levels.
        levels: usize,
    },
}

/// A stable encoding of a parameter space into `[0, 1]^d`.
#[derive(Debug, Clone)]
pub struct ConfigurationEncoding {
    /// Parameter names, sorted so the encoding is stable across processes
    /// (`HashMap` iteration order is not).
    names: Vec<String>,
    /// Per-parameter scheme, parallel to `names`.
    schemes: Vec<Scheme>,
}

impl ConfigurationEncoding {
    /// Derive the encoding from a parameter space.
    pub fn from_space<T: Float + Debug + Send + Sync + 'static>(
        space: &ParameterSpace<T>,
    ) -> Result<Self> {
        if space.parameters.is_empty() {
            return Err(OptimError::InvalidConfig(
                "the parameter space declares no hyperparameter".to_string(),
            ));
        }
        let mut names: Vec<String> = space.parameters.keys().cloned().collect();
        names.sort();

        let mut schemes = Vec::with_capacity(names.len());
        for name in &names {
            let definition = space.parameters.get(name).ok_or_else(|| {
                OptimError::InvalidState(format!("parameter `{name}` vanished from the space"))
            })?;
            let scheme = match &definition.param_type {
                ParameterType::Continuous => {
                    let min = definition
                        .bounds
                        .min
                        .and_then(|value| value.to_f64())
                        .unwrap_or(0.0);
                    let max = definition
                        .bounds
                        .max
                        .and_then(|value| value.to_f64())
                        .unwrap_or(1.0);
                    if !(min.is_finite() && max.is_finite()) || max <= min {
                        return Err(OptimError::InvalidConfig(format!(
                            "continuous parameter `{name}` has an empty or non-finite range \
                             [{min}, {max}]"
                        )));
                    }
                    Scheme::Numeric { min, max }
                }
                ParameterType::Integer => {
                    let min = definition
                        .bounds
                        .min
                        .and_then(|value| value.to_i64())
                        .unwrap_or(0);
                    let max = definition
                        .bounds
                        .max
                        .and_then(|value| value.to_i64())
                        .unwrap_or(100);
                    if max <= min {
                        return Err(OptimError::InvalidConfig(format!(
                            "integer parameter `{name}` has an empty range [{min}, {max}]"
                        )));
                    }
                    Scheme::Integer { min, max }
                }
                ParameterType::Boolean => Scheme::Boolean,
                ParameterType::Categorical(values) => {
                    if values.is_empty() {
                        return Err(OptimError::InvalidConfig(format!(
                            "categorical parameter `{name}` declares no category"
                        )));
                    }
                    Scheme::Categorical {
                        values: values.clone(),
                    }
                }
                ParameterType::Ordinal(levels) => {
                    if levels.is_empty() {
                        return Err(OptimError::InvalidConfig(format!(
                            "ordinal parameter `{name}` declares no level"
                        )));
                    }
                    Scheme::Ordinal {
                        levels: levels.len(),
                    }
                }
            };
            schemes.push(scheme);
        }
        Ok(Self { names, schemes })
    }

    /// Number of encoded dimensions.
    pub fn dimension(&self) -> usize {
        self.names.len()
    }

    /// The parameter names, in encoding order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Encode a configuration into `[0, 1]^d`.
    pub fn encode<T: Float + Debug + Send + Sync + 'static>(
        &self,
        config: &ParameterConfiguration<T>,
    ) -> Result<Vec<f64>> {
        let mut encoded = Vec::with_capacity(self.names.len());
        for (name, scheme) in self.names.iter().zip(self.schemes.iter()) {
            let value = config.values.get(name).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "configuration `{}` does not set parameter `{name}`",
                    config.id
                ))
            })?;
            encoded.push(encode_value(name, scheme, value)?);
        }
        Ok(encoded)
    }

    /// Decode a point of `[0, 1]^d` back into a configuration.
    pub fn decode<T: Float + Debug + Send + Sync + 'static>(
        &self,
        point: &[f64],
        id: impl Into<String>,
    ) -> Result<ParameterConfiguration<T>> {
        if point.len() != self.names.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "the encoding has {} dimensions, {} were supplied",
                self.names.len(),
                point.len()
            )));
        }
        let mut values = HashMap::with_capacity(self.names.len());
        for ((name, scheme), coordinate) in
            self.names.iter().zip(self.schemes.iter()).zip(point.iter())
        {
            values.insert(name.clone(), decode_value(name, scheme, *coordinate)?);
        }
        Ok(ParameterConfiguration {
            values,
            id: id.into(),
            metadata: HashMap::new(),
        })
    }

    /// Draw a uniformly random point of `[0, 1]^d`.
    pub fn sample_point(&self, rng: &mut HpoRng) -> Vec<f64> {
        (0..self.names.len())
            .map(|_| {
                let raw: f64 = rng.gen_range(0.0..1.0);
                raw
            })
            .collect()
    }
}

/// Encode one parameter value.
fn encode_value<T: Float + Debug + Send + Sync + 'static>(
    name: &str,
    scheme: &Scheme,
    value: &ParameterValue<T>,
) -> Result<f64> {
    match (scheme, value) {
        (Scheme::Numeric { min, max }, ParameterValue::Continuous(raw)) => {
            let raw = raw.to_f64().ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "the value of `{name}` cannot be represented as f64"
                ))
            })?;
            Ok(((raw - min) / (max - min)).clamp(0.0, 1.0))
        }
        (Scheme::Integer { min, max }, ParameterValue::Integer(raw)) => {
            Ok(((*raw - *min) as f64 / (*max - *min) as f64).clamp(0.0, 1.0))
        }
        (Scheme::Boolean, ParameterValue::Boolean(raw)) => Ok(f64::from(*raw)),
        (Scheme::Categorical { values }, ParameterValue::Categorical(raw)) => {
            let position = values
                .iter()
                .position(|value| value == raw)
                .ok_or_else(|| {
                    OptimError::InvalidParameter(format!(
                        "`{raw}` is not a declared category of parameter `{name}`"
                    ))
                })?;
            Ok(category_coordinate(position, values.len()))
        }
        (Scheme::Ordinal { levels }, ParameterValue::Ordinal(index)) => {
            if *index >= *levels {
                return Err(OptimError::InvalidParameter(format!(
                    "ordinal index {index} is out of range for parameter `{name}` with {levels} \
                     levels"
                )));
            }
            Ok(category_coordinate(*index, *levels))
        }
        (scheme, value) => Err(OptimError::InvalidParameter(format!(
            "parameter `{name}` is declared as {scheme:?} but was given {value:?}"
        ))),
    }
}

/// Coordinate of a discrete level, at the centre of its bucket.
fn category_coordinate(index: usize, count: usize) -> f64 {
    if count <= 1 {
        0.5
    } else {
        (index as f64 + 0.5) / count as f64
    }
}

/// Bucket index of a coordinate in `[0, 1]`.
fn coordinate_bucket(coordinate: f64, count: usize) -> usize {
    if count <= 1 {
        return 0;
    }
    let scaled = (coordinate.clamp(0.0, 1.0) * count as f64).floor() as usize;
    scaled.min(count - 1)
}

/// Decode one parameter value.
fn decode_value<T: Float + Debug + Send + Sync + 'static>(
    name: &str,
    scheme: &Scheme,
    coordinate: f64,
) -> Result<ParameterValue<T>> {
    if !coordinate.is_finite() {
        return Err(OptimError::InvalidParameter(format!(
            "the encoded coordinate of `{name}` is {coordinate}"
        )));
    }
    let unit = coordinate.clamp(0.0, 1.0);
    match scheme {
        Scheme::Numeric { min, max } => {
            let raw = min + unit * (max - min);
            let value = T::from(raw).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "the decoded value {raw} of `{name}` cannot be represented"
                ))
            })?;
            Ok(ParameterValue::Continuous(value))
        }
        Scheme::Integer { min, max } => {
            let span = (*max - *min) as f64;
            let raw = (*min as f64 + unit * span).round() as i64;
            Ok(ParameterValue::Integer(raw.clamp(*min, *max)))
        }
        Scheme::Boolean => Ok(ParameterValue::Boolean(unit >= 0.5)),
        Scheme::Categorical { values } => {
            let index = coordinate_bucket(unit, values.len());
            Ok(ParameterValue::Categorical(values[index].clone()))
        }
        Scheme::Ordinal { levels } => Ok(ParameterValue::Ordinal(coordinate_bucket(unit, *levels))),
    }
}

/// Encode a single configuration against a space, deriving the encoding.
pub fn encode_configuration<T: Float + Debug + Send + Sync + 'static>(
    space: &ParameterSpace<T>,
    config: &ParameterConfiguration<T>,
) -> Result<Vec<f64>> {
    ConfigurationEncoding::from_space(space)?.encode(config)
}

/// Exact Gaussian-process regression with an RBF kernel.
#[derive(Debug, Clone)]
pub struct GaussianProcessFit {
    /// Training inputs.
    inputs: Vec<Vec<f64>>,
    /// Kernel length scale.
    length_scale: f64,
    /// Kernel signal variance.
    signal_variance: f64,
    /// Observation noise variance.
    noise_variance: f64,
    /// Constant prior mean (the training mean).
    prior_mean: f64,
    /// Lower-triangular Cholesky factor of `K + sigma_n^2 I`.
    cholesky: Vec<Vec<f64>>,
    /// `alpha = (K + sigma_n^2 I)^{-1} (y - prior_mean)`.
    alpha: Vec<f64>,
}

impl GaussianProcessFit {
    /// Fit the GP.
    pub fn fit(
        inputs: &[Vec<f64>],
        targets: &[f64],
        length_scale: f64,
        signal_variance: f64,
        noise_variance: f64,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(OptimError::InvalidParameter(
                "a Gaussian process needs at least one observation".to_string(),
            ));
        }
        if inputs.len() != targets.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "{} inputs and {} targets were supplied",
                inputs.len(),
                targets.len()
            )));
        }
        let dimension = inputs[0].len();
        if dimension == 0 {
            return Err(OptimError::InvalidParameter(
                "the encoded inputs are zero-dimensional".to_string(),
            ));
        }
        for (index, input) in inputs.iter().enumerate() {
            if input.len() != dimension {
                return Err(OptimError::DimensionMismatch(format!(
                    "input {index} has {} dimensions, expected {dimension}",
                    input.len()
                )));
            }
            if !input.iter().all(|value| value.is_finite()) {
                return Err(OptimError::InvalidParameter(format!(
                    "input {index} contains a non-finite coordinate"
                )));
            }
        }
        if !targets.iter().all(|value| value.is_finite()) {
            return Err(OptimError::InvalidParameter(
                "every target must be finite".to_string(),
            ));
        }
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the kernel length scale must be positive and finite, got {length_scale}"
            )));
        }
        if !signal_variance.is_finite() || signal_variance <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the kernel signal variance must be positive and finite, got {signal_variance}"
            )));
        }
        if !noise_variance.is_finite() || noise_variance <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the observation noise variance must be positive and finite, got {noise_variance}"
            )));
        }

        let count = inputs.len();
        let prior_mean = targets.iter().sum::<f64>() / count as f64;
        let mut gram = vec![vec![0.0f64; count]; count];
        for row in 0..count {
            for column in 0..count {
                let mut value =
                    rbf_kernel(&inputs[row], &inputs[column], length_scale, signal_variance);
                if row == column {
                    value += noise_variance;
                }
                gram[row][column] = value;
            }
        }
        let cholesky = cholesky_decompose(&gram)?;
        let centred: Vec<f64> = targets.iter().map(|value| value - prior_mean).collect();
        let alpha = cholesky_solve(&cholesky, &centred)?;

        Ok(Self {
            inputs: inputs.to_vec(),
            length_scale,
            signal_variance,
            noise_variance,
            prior_mean,
            cholesky,
            alpha,
        })
    }

    /// Fit with the median heuristic for the length scale and the target
    /// variance for the signal variance.
    ///
    /// `noise_variance` should be the variance of the differential-privacy noise
    /// that was added to the targets, so the surrogate does not treat a heavily
    /// perturbed release as exact.
    pub fn fit_with_median_heuristic(
        inputs: &[Vec<f64>],
        targets: &[f64],
        noise_variance: f64,
    ) -> Result<Self> {
        let length_scale = median_pairwise_distance(inputs).max(1e-3);
        let count = targets.len().max(1) as f64;
        let mean = targets.iter().sum::<f64>() / count;
        let variance = targets
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        let signal_variance = variance.max(1e-6);
        Self::fit(
            inputs,
            targets,
            length_scale,
            signal_variance,
            noise_variance.max(1e-9),
        )
    }

    /// Number of training observations.
    pub fn observation_count(&self) -> usize {
        self.inputs.len()
    }

    /// The fitted length scale.
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// The fitted signal variance.
    pub fn signal_variance(&self) -> f64 {
        self.signal_variance
    }

    /// The observation noise variance the GP was told about.
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// Posterior mean and (latent) variance at `point`.
    pub fn predict(&self, point: &[f64]) -> Result<(f64, f64)> {
        if point.len() != self.inputs[0].len() {
            return Err(OptimError::DimensionMismatch(format!(
                "the GP was fitted on {}-dimensional inputs, {} were supplied",
                self.inputs[0].len(),
                point.len()
            )));
        }
        let cross: Vec<f64> = self
            .inputs
            .iter()
            .map(|input| rbf_kernel(input, point, self.length_scale, self.signal_variance))
            .collect();
        let mean = self.prior_mean
            + cross
                .iter()
                .zip(self.alpha.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>();
        let solved = forward_substitute(&self.cholesky, &cross)?;
        let explained = solved.iter().map(|value| value * value).sum::<f64>();
        let variance = (self.signal_variance - explained).max(0.0);
        Ok((mean, variance))
    }
}

/// RBF (squared-exponential) kernel.
fn rbf_kernel(left: &[f64], right: &[f64], length_scale: f64, signal_variance: f64) -> f64 {
    let squared: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    signal_variance * (-squared / (2.0 * length_scale * length_scale)).exp()
}

/// Median pairwise Euclidean distance of the inputs.
fn median_pairwise_distance(inputs: &[Vec<f64>]) -> f64 {
    let mut distances = Vec::new();
    for (index, left) in inputs.iter().enumerate() {
        for right in inputs.iter().skip(index + 1) {
            let squared: f64 = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            distances.push(squared.sqrt());
        }
    }
    if distances.is_empty() {
        return 1.0;
    }
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    distances[distances.len() / 2]
}

/// Cholesky factorisation of a symmetric positive-definite matrix.
pub fn cholesky_decompose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let size = matrix.len();
    let mut lower = vec![vec![0.0f64; size]; size];
    for row in 0..size {
        for column in 0..=row {
            let mut sum = matrix[row][column];
            for (left, right) in lower[row][..column]
                .iter()
                .zip(lower[column][..column].iter())
            {
                sum -= left * right;
            }
            if row == column {
                if sum <= 0.0 {
                    return Err(OptimError::ComputationError(format!(
                        "the kernel matrix is not positive definite: pivot {row} came out as {sum}"
                    )));
                }
                lower[row][column] = sum.sqrt();
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
    Ok(lower)
}

/// Solve `L z = b` by forward substitution.
fn forward_substitute(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let size = lower.len();
    if rhs.len() != size {
        return Err(OptimError::DimensionMismatch(format!(
            "the factor is {size}x{size} but the right-hand side has {} entries",
            rhs.len()
        )));
    }
    let mut solution = vec![0.0f64; size];
    for row in 0..size {
        let mut value = rhs[row];
        for column in 0..row {
            value -= lower[row][column] * solution[column];
        }
        let pivot = lower[row][row];
        if pivot.abs() <= f64::MIN_POSITIVE {
            return Err(OptimError::ComputationError(format!(
                "the Cholesky factor has a zero pivot at {row}"
            )));
        }
        solution[row] = value / pivot;
    }
    Ok(solution)
}

/// Solve `L L^T x = b`.
pub fn cholesky_solve(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let intermediate = forward_substitute(lower, rhs)?;
    let size = lower.len();
    let mut solution = vec![0.0f64; size];
    for row in (0..size).rev() {
        let mut value = intermediate[row];
        for column in (row + 1)..size {
            value -= lower[column][row] * solution[column];
        }
        let pivot = lower[row][row];
        if pivot.abs() <= f64::MIN_POSITIVE {
            return Err(OptimError::ComputationError(format!(
                "the Cholesky factor has a zero pivot at {row}"
            )));
        }
        solution[row] = value / pivot;
    }
    Ok(solution)
}

/// Standard normal probability density.
fn standard_normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (std::f64::consts::TAU).sqrt()
}

/// Standard normal cumulative distribution.
///
/// Uses the Abramowitz & Stegun 7.1.26 rational approximation of `erf`, whose
/// absolute error is bounded by `1.5e-7` -- ample for ranking acquisition
/// values, and asserted against published values in the tests.
fn standard_normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + erf(value / std::f64::consts::SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26).
pub fn erf(value: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Expected-improvement acquisition function.
#[derive(Debug, Clone, Copy)]
pub struct ExpectedImprovement {
    /// Exploration margin added to the incumbent.
    pub xi: f64,
}

impl ExpectedImprovement {
    /// EI with the conventional `xi = 0.01` margin.
    pub fn new() -> Self {
        Self { xi: 0.01 }
    }

    /// EI with an explicit margin.
    pub fn with_margin(xi: f64) -> Result<Self> {
        if !xi.is_finite() || xi < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the exploration margin must be non-negative and finite, got {xi}"
            )));
        }
        Ok(Self { xi })
    }

    /// Expected improvement of a posterior `(mean, variance)` over `incumbent`.
    pub fn evaluate(&self, mean: f64, variance: f64, incumbent: f64) -> Result<f64> {
        if !mean.is_finite() || !variance.is_finite() || variance < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the posterior (mean {mean}, variance {variance}) is not usable"
            )));
        }
        if !incumbent.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "the incumbent value {incumbent} is not finite"
            )));
        }
        let improvement = mean - incumbent - self.xi;
        let sigma = variance.sqrt();
        if sigma <= f64::MIN_POSITIVE {
            return Ok(improvement.max(0.0));
        }
        let z = improvement / sigma;
        Ok(improvement * standard_normal_cdf(z) + sigma * standard_normal_pdf(z))
    }
}

impl Default for ExpectedImprovement {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::private_hyperparameter_optimization::types::{
        ParameterBounds, ParameterDefinition,
    };

    fn space() -> ParameterSpace<f64> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "learning_rate".to_string(),
            ParameterDefinition {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous,
                bounds: ParameterBounds {
                    min: Some(0.001),
                    max: Some(0.1),
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        parameters.insert(
            "batch_size".to_string(),
            ParameterDefinition {
                name: "batch_size".to_string(),
                param_type: ParameterType::Integer,
                bounds: ParameterBounds {
                    min: Some(8.0),
                    max: Some(256.0),
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        parameters.insert(
            "use_momentum".to_string(),
            ParameterDefinition {
                name: "use_momentum".to_string(),
                param_type: ParameterType::Boolean,
                bounds: ParameterBounds {
                    min: None,
                    max: None,
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        parameters.insert(
            "optimizer".to_string(),
            ParameterDefinition {
                name: "optimizer".to_string(),
                param_type: ParameterType::Categorical(vec![
                    "sgd".to_string(),
                    "adam".to_string(),
                    "rmsprop".to_string(),
                ]),
                bounds: ParameterBounds {
                    min: None,
                    max: None,
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        parameters.insert(
            "schedule".to_string(),
            ParameterDefinition {
                name: "schedule".to_string(),
                param_type: ParameterType::Ordinal(vec![0.1, 0.5, 1.0, 2.0]),
                bounds: ParameterBounds {
                    min: None,
                    max: None,
                    step: None,
                    valid_values: None,
                },
                prior: None,
                transformation: None,
            },
        );
        ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        }
    }

    #[test]
    fn the_encoding_covers_every_parameter_kind_and_round_trips() {
        let encoding = match ConfigurationEncoding::from_space(&space()) {
            Ok(encoding) => encoding,
            Err(err) => panic!("encoding failed: {err}"),
        };
        assert_eq!(encoding.dimension(), 5);
        assert_eq!(
            encoding.names(),
            [
                "batch_size".to_string(),
                "learning_rate".to_string(),
                "optimizer".to_string(),
                "schedule".to_string(),
                "use_momentum".to_string(),
            ]
        );

        let mut values = HashMap::new();
        values.insert(
            "learning_rate".to_string(),
            ParameterValue::Continuous(0.05),
        );
        values.insert("batch_size".to_string(), ParameterValue::Integer(64));
        values.insert("use_momentum".to_string(), ParameterValue::Boolean(true));
        values.insert(
            "optimizer".to_string(),
            ParameterValue::Categorical("adam".to_string()),
        );
        values.insert("schedule".to_string(), ParameterValue::Ordinal(2));
        let config = ParameterConfiguration {
            values,
            id: "c0".to_string(),
            metadata: HashMap::new(),
        };

        let point = match encoding.encode(&config) {
            Ok(point) => point,
            Err(err) => panic!("encode failed: {err}"),
        };
        assert!(point.iter().all(|value| (0.0..=1.0).contains(value)));

        let decoded: ParameterConfiguration<f64> = match encoding.decode(&point, "c0") {
            Ok(decoded) => decoded,
            Err(err) => panic!("decode failed: {err}"),
        };
        // Every non-continuous parameter must round-trip exactly; the
        // continuous one to within the encoding's precision.
        assert!(matches!(
            decoded.values.get("use_momentum"),
            Some(ParameterValue::Boolean(true))
        ));
        match decoded.values.get("optimizer") {
            Some(ParameterValue::Categorical(name)) => assert_eq!(name, "adam"),
            other => panic!("unexpected optimizer {other:?}"),
        }
        match decoded.values.get("schedule") {
            Some(ParameterValue::Ordinal(index)) => assert_eq!(*index, 2),
            other => panic!("unexpected schedule {other:?}"),
        }
        match decoded.values.get("batch_size") {
            Some(ParameterValue::Integer(size)) => assert_eq!(*size, 64),
            other => panic!("unexpected batch_size {other:?}"),
        }
        match decoded.values.get("learning_rate") {
            Some(ParameterValue::Continuous(rate)) => assert!((rate - 0.05).abs() < 1e-9),
            other => panic!("unexpected learning_rate {other:?}"),
        }
    }

    #[test]
    fn every_categorical_level_is_reachable_by_decoding() {
        let encoding = match ConfigurationEncoding::from_space(&space()) {
            Ok(encoding) => encoding,
            Err(err) => panic!("encoding failed: {err}"),
        };
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=100 {
            let unit = step as f64 / 100.0;
            let point: Vec<f64> = encoding.names().iter().map(|_| unit).collect();
            let decoded: ParameterConfiguration<f64> = match encoding.decode(&point, "c") {
                Ok(decoded) => decoded,
                Err(err) => panic!("decode failed: {err}"),
            };
            if let Some(ParameterValue::Categorical(name)) = decoded.values.get("optimizer") {
                seen.insert(name.clone());
            }
        }
        assert_eq!(
            seen.len(),
            3,
            "every declared category must be reachable, saw {seen:?}"
        );
    }

    #[test]
    fn a_mistyped_parameter_value_is_refused() {
        let encoding = match ConfigurationEncoding::from_space(&space()) {
            Ok(encoding) => encoding,
            Err(err) => panic!("encoding failed: {err}"),
        };
        let mut values = HashMap::new();
        values.insert(
            "learning_rate".to_string(),
            ParameterValue::Continuous(0.05),
        );
        values.insert("batch_size".to_string(), ParameterValue::Continuous(64.0));
        values.insert("use_momentum".to_string(), ParameterValue::Boolean(true));
        values.insert(
            "optimizer".to_string(),
            ParameterValue::Categorical("adam".to_string()),
        );
        values.insert("schedule".to_string(), ParameterValue::Ordinal(2));
        let config = ParameterConfiguration {
            values,
            id: "bad".to_string(),
            metadata: HashMap::new(),
        };
        assert!(encoding.encode(&config).is_err());
    }

    #[test]
    fn an_unknown_category_is_refused() {
        let encoding = match ConfigurationEncoding::from_space(&space()) {
            Ok(encoding) => encoding,
            Err(err) => panic!("encoding failed: {err}"),
        };
        let mut values = HashMap::new();
        values.insert(
            "learning_rate".to_string(),
            ParameterValue::Continuous(0.05),
        );
        values.insert("batch_size".to_string(), ParameterValue::Integer(64));
        values.insert("use_momentum".to_string(), ParameterValue::Boolean(true));
        values.insert(
            "optimizer".to_string(),
            ParameterValue::Categorical("lion".to_string()),
        );
        values.insert("schedule".to_string(), ParameterValue::Ordinal(2));
        let config = ParameterConfiguration {
            values,
            id: "bad".to_string(),
            metadata: HashMap::new(),
        };
        assert!(encoding.encode(&config).is_err());
    }

    #[test]
    fn a_missing_parameter_is_refused() {
        let encoding = match ConfigurationEncoding::from_space(&space()) {
            Ok(encoding) => encoding,
            Err(err) => panic!("encoding failed: {err}"),
        };
        let config: ParameterConfiguration<f64> = ParameterConfiguration {
            values: HashMap::new(),
            id: "empty".to_string(),
            metadata: HashMap::new(),
        };
        // This is exactly the shape `suggest_next` used to return.
        assert!(
            encoding.encode(&config).is_err(),
            "an empty configuration must not encode"
        );
    }

    #[test]
    fn cholesky_reproduces_a_known_factorisation() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        let matrix = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        let lower = match cholesky_decompose(&matrix) {
            Ok(lower) => lower,
            Err(err) => panic!("factorisation failed: {err}"),
        };
        let expected = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]];
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (lower[row][column] - expected[row][column]).abs() < 1e-9,
                    "L[{row}][{column}] = {}",
                    lower[row][column]
                );
            }
        }
    }

    #[test]
    fn cholesky_solve_recovers_the_solution() {
        let matrix = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        let lower = match cholesky_decompose(&matrix) {
            Ok(lower) => lower,
            Err(err) => panic!("factorisation failed: {err}"),
        };
        let expected = [1.0f64, -2.0, 0.5];
        let rhs: Vec<f64> = (0..3)
            .map(|row| {
                (0..3)
                    .map(|column| matrix[row][column] * expected[column])
                    .sum()
            })
            .collect();
        let solved = match cholesky_solve(&lower, &rhs) {
            Ok(solved) => solved,
            Err(err) => panic!("solve failed: {err}"),
        };
        for (index, (got, want)) in solved.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "x[{index}] = {got}");
        }
    }

    #[test]
    fn a_non_positive_definite_matrix_is_refused() {
        let matrix = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert!(cholesky_decompose(&matrix).is_err());
    }

    #[test]
    fn the_gp_interpolates_its_training_data_when_the_noise_is_tiny() {
        let inputs = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
        let targets: Vec<f64> = inputs.iter().map(|point| (point[0] * 3.0).sin()).collect();
        let gp = match GaussianProcessFit::fit(&inputs, &targets, 0.3, 1.0, 1e-10) {
            Ok(gp) => gp,
            Err(err) => panic!("fit failed: {err}"),
        };
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let (mean, variance) = match gp.predict(input) {
                Ok(prediction) => prediction,
                Err(err) => panic!("predict failed: {err}"),
            };
            assert!(
                (mean - target).abs() < 1e-4,
                "GP predicted {mean} at a training point whose target is {target}"
            );
            assert!(
                variance < 1e-4,
                "the variance at a training point must be near zero, got {variance}"
            );
        }
    }

    #[test]
    fn the_gp_variance_grows_away_from_the_data() {
        let inputs = vec![vec![0.0], vec![0.1], vec![0.2]];
        let targets = vec![0.0, 0.1, 0.2];
        let gp = match GaussianProcessFit::fit(&inputs, &targets, 0.1, 1.0, 1e-6) {
            Ok(gp) => gp,
            Err(err) => panic!("fit failed: {err}"),
        };
        let (_, near) = match gp.predict(&[0.15]) {
            Ok(prediction) => prediction,
            Err(err) => panic!("predict failed: {err}"),
        };
        let (_, far) = match gp.predict(&[5.0]) {
            Ok(prediction) => prediction,
            Err(err) => panic!("predict failed: {err}"),
        };
        assert!(
            far > near,
            "variance at a far point ({far}) must exceed a near point ({near})"
        );
        assert!(
            (far - gp.signal_variance()).abs() < 1e-6,
            "far from the data the posterior variance must return to the prior"
        );
    }

    #[test]
    fn the_gp_recovers_a_linear_trend_between_observations() {
        let inputs: Vec<Vec<f64>> = (0..11).map(|index| vec![index as f64 / 10.0]).collect();
        let targets: Vec<f64> = inputs.iter().map(|point| 2.0 * point[0] + 1.0).collect();
        let gp = match GaussianProcessFit::fit_with_median_heuristic(&inputs, &targets, 1e-8) {
            Ok(gp) => gp,
            Err(err) => panic!("fit failed: {err}"),
        };
        let (mean, _) = match gp.predict(&[0.55]) {
            Ok(prediction) => prediction,
            Err(err) => panic!("predict failed: {err}"),
        };
        assert!(
            (mean - 2.1).abs() < 0.05,
            "GP interpolated {mean} where the truth is 2.1"
        );
    }

    #[test]
    fn degenerate_gp_inputs_are_refused() {
        assert!(GaussianProcessFit::fit(&[], &[], 1.0, 1.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![0.0]], &[], 1.0, 1.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![]], &[0.0], 1.0, 1.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![0.0]], &[0.0], 0.0, 1.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![0.0]], &[0.0], 1.0, 0.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![0.0]], &[0.0], 1.0, 1.0, 0.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![f64::NAN]], &[0.0], 1.0, 1.0, 1.0).is_err());
        assert!(GaussianProcessFit::fit(&[vec![0.0]], &[f64::NAN], 1.0, 1.0, 1.0).is_err());
        assert!(
            GaussianProcessFit::fit(&[vec![0.0], vec![0.0, 1.0]], &[0.0, 1.0], 1.0, 1.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn erf_matches_published_values() {
        // Abramowitz & Stegun table values.
        let cases = [
            (0.0f64, 0.0f64),
            (0.5, 0.520_499_877),
            (1.0, 0.842_700_793),
            (1.5, 0.966_105_146),
            (2.0, 0.995_322_265),
            (-1.0, -0.842_700_793),
        ];
        for (input, expected) in cases {
            let got = erf(input);
            assert!(
                (got - expected).abs() < 2e-7,
                "erf({input}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn the_normal_cdf_matches_published_values() {
        let cases = [
            (0.0f64, 0.5f64),
            (1.0, 0.841_344_746),
            (1.645, 0.950_015_192),
            (1.96, 0.975_002_105),
            (-1.96, 0.024_997_895),
        ];
        for (input, expected) in cases {
            let got = standard_normal_cdf(input);
            assert!(
                (got - expected).abs() < 2e-7,
                "Phi({input}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn expected_improvement_prefers_promising_and_uncertain_points() {
        let ei = ExpectedImprovement { xi: 0.0 };
        let incumbent = 1.0;
        let promising = match ei.evaluate(1.5, 0.01, incumbent) {
            Ok(value) => value,
            Err(err) => panic!("EI failed: {err}"),
        };
        let hopeless = match ei.evaluate(0.2, 0.01, incumbent) {
            Ok(value) => value,
            Err(err) => panic!("EI failed: {err}"),
        };
        assert!(promising > hopeless);

        // At equal means, more uncertainty is more attractive.
        let certain = match ei.evaluate(1.0, 1e-8, incumbent) {
            Ok(value) => value,
            Err(err) => panic!("EI failed: {err}"),
        };
        let uncertain = match ei.evaluate(1.0, 1.0, incumbent) {
            Ok(value) => value,
            Err(err) => panic!("EI failed: {err}"),
        };
        assert!(uncertain > certain);
    }

    #[test]
    fn expected_improvement_matches_the_closed_form() {
        // mu = 1.5, sigma = 1, incumbent = 1, xi = 0 => z = 0.5
        // EI = 0.5 * Phi(0.5) + 1 * phi(0.5)
        let ei = ExpectedImprovement { xi: 0.0 };
        let got = match ei.evaluate(1.5, 1.0, 1.0) {
            Ok(value) => value,
            Err(err) => panic!("EI failed: {err}"),
        };
        let expected = 0.5 * 0.691_462_461 + 0.352_065_326;
        assert!((got - expected).abs() < 1e-6, "EI = {got}");
    }

    #[test]
    fn expected_improvement_is_never_negative_and_rejects_bad_input() {
        let ei = ExpectedImprovement::new();
        for (mean, variance, incumbent) in [
            (0.0f64, 0.0f64, 10.0f64),
            (-5.0, 1.0, 10.0),
            (10.0, 0.0, 10.0),
        ] {
            match ei.evaluate(mean, variance, incumbent) {
                Ok(value) => assert!(value >= 0.0, "EI was {value}"),
                Err(err) => panic!("EI failed: {err}"),
            }
        }
        assert!(ei.evaluate(f64::NAN, 1.0, 0.0).is_err());
        assert!(ei.evaluate(0.0, -1.0, 0.0).is_err());
        assert!(ei.evaluate(0.0, 1.0, f64::INFINITY).is_err());
        assert!(ExpectedImprovement::with_margin(-1.0).is_err());
    }
}
