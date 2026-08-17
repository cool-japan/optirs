//! Budget allocation, sensitivity analysis and utility-degradation prediction.

use crate::error::{OptimError, Result};
use crate::privacy::PrivacyBudget;
use scirs2_core::ndarray::{ArrayBase, Data, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

use super::stats;
use super::types::{
    to_f64, to_t, validate_privacy_configuration, PrivacyUtilityAnalyzer, SensitivityResults,
};
use super::types_3::{
    AllocationStrategy, BudgetAllocation, DegradationPrediction, LocalSensitivity, PredictionModel,
    PrivacyConfiguration,
};

/// Relative step used by the finite-difference sensitivity analysis. The step
/// applied to a parameter `x` is `RELATIVE_STEP * max(|x|, 1)`, which never
/// degenerates to zero even when `x` itself is zero (the normal case for
/// `delta` under pure epsilon-DP).
const RELATIVE_STEP: f64 = 0.01;

/// Salt for the replicate-jitter random stream.
const SENSITIVITY_SALT: u64 = 0x5e_1471_7a17;

/// A privacy parameter that the sensitivity analysis differentiates against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SensitivityParameter {
    /// Privacy budget epsilon
    Epsilon,
    /// Gaussian/Laplace noise multiplier
    NoiseMultiplier,
    /// Gradient clipping threshold
    ClippingThreshold,
    /// Failure probability delta
    Delta,
}

impl SensitivityParameter {
    const ALL: [SensitivityParameter; 4] = [
        SensitivityParameter::Epsilon,
        SensitivityParameter::NoiseMultiplier,
        SensitivityParameter::ClippingThreshold,
        SensitivityParameter::Delta,
    ];

    fn name(self) -> &'static str {
        match self {
            SensitivityParameter::Epsilon => "epsilon",
            SensitivityParameter::NoiseMultiplier => "noise_multiplier",
            SensitivityParameter::ClippingThreshold => "clipping_threshold",
            SensitivityParameter::Delta => "delta",
        }
    }

    /// Open interval the parameter must stay inside for the configuration to
    /// remain valid.
    fn domain(self) -> (f64, f64) {
        match self {
            SensitivityParameter::Delta => (0.0, 1.0),
            _ => (0.0, f64::INFINITY),
        }
    }

    /// Whether the domain includes its lower bound.
    fn lower_bound_inclusive(self) -> bool {
        matches!(self, SensitivityParameter::Delta)
    }

    fn value<T: Float + Debug + Send + Sync + 'static>(
        self,
        config: &PrivacyConfiguration<T>,
    ) -> Result<f64> {
        match self {
            SensitivityParameter::Epsilon => to_f64(config.epsilon),
            SensitivityParameter::NoiseMultiplier => to_f64(config.noise_multiplier),
            SensitivityParameter::ClippingThreshold => to_f64(config.clipping_threshold),
            SensitivityParameter::Delta => to_f64(config.delta),
        }
    }

    fn admissible(self, value: f64) -> bool {
        let (lo, hi) = self.domain();
        value.is_finite()
            && value < hi
            && if self.lower_bound_inclusive() {
                value >= lo
            } else {
                value > lo
            }
    }

    fn with_value<T: Float + Debug + Send + Sync + 'static>(
        self,
        config: &PrivacyConfiguration<T>,
        value: f64,
    ) -> Result<PrivacyConfiguration<T>> {
        let mut updated = config.clone();
        let value_t = to_t(value)?;
        match self {
            SensitivityParameter::Epsilon => updated.epsilon = value_t,
            SensitivityParameter::NoiseMultiplier => updated.noise_multiplier = value_t,
            SensitivityParameter::ClippingThreshold => updated.clipping_threshold = value_t,
            SensitivityParameter::Delta => updated.delta = value_t,
        }
        Ok(updated)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Candidate budget allocations, best first.
    ///
    /// Six allocation shapes are evaluated against the caller's utility model
    /// `utility_model(epsilon) -> utility`; the predicted utility of an
    /// allocation is the mean of that model over the per-iteration epsilons.
    /// The analyzer has no built-in utility model, so the caller's model is
    /// what decides the ranking.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for a non-positive budget or
    /// zero iterations, and [`OptimError::ComputationError`] if the model
    /// returns a non-finite value.
    pub fn budget_allocation_candidates(
        &self,
        total_epsilon: f64,
        iterations: usize,
        utility_model: &impl Fn(f64) -> f64,
    ) -> Result<Vec<BudgetAllocation<T>>> {
        if iterations == 0 {
            return Err(OptimError::InvalidParameter(
                "iterations must be > 0".to_string(),
            ));
        }
        if !total_epsilon.is_finite() || total_epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "total epsilon must be finite and > 0, got {total_epsilon}"
            )));
        }
        let n = iterations;
        let strategies: [AllocationStrategy; 6] = [
            AllocationStrategy::Uniform,
            AllocationStrategy::Decreasing,
            AllocationStrategy::Increasing,
            AllocationStrategy::Adaptive,
            AllocationStrategy::ImportanceBased,
            AllocationStrategy::RiskBased,
        ];
        let mut allocations = Vec::with_capacity(strategies.len());
        for strategy in strategies {
            let raw_weights: Vec<f64> = match strategy {
                AllocationStrategy::Uniform => vec![1.0_f64; n],
                AllocationStrategy::Decreasing => {
                    if n == 1 {
                        vec![1.0_f64]
                    } else {
                        (0..n)
                            .map(|i| 2.0 * (1.0 - i as f64 / (n - 1) as f64))
                            .collect()
                    }
                }
                AllocationStrategy::Increasing => {
                    if n == 1 {
                        vec![1.0_f64]
                    } else {
                        (0..n).map(|i| 2.0 * (i as f64 / (n - 1) as f64)).collect()
                    }
                }
                AllocationStrategy::Adaptive => {
                    let r = 0.5_f64.powf(1.0 / n as f64);
                    (0..n).map(|i| (1.0 - r) * r.powi(i as i32)).collect()
                }
                AllocationStrategy::ImportanceBased => (0..n)
                    .map(|i| {
                        let v = (n - i) as f64;
                        v * v
                    })
                    .collect(),
                AllocationStrategy::RiskBased => (0..n).map(|i| (2.0 + i as f64).ln()).collect(),
            };
            let weight_sum: f64 = raw_weights.iter().sum();
            let per_iteration: Vec<f64> = if weight_sum > 1e-12 {
                raw_weights
                    .iter()
                    .map(|w| w / weight_sum * total_epsilon)
                    .collect()
            } else {
                vec![total_epsilon / n as f64; n]
            };
            let mut predicted_sum = 0.0_f64;
            for &eps in &per_iteration {
                let utility = utility_model(eps);
                if !utility.is_finite() {
                    return Err(OptimError::ComputationError(format!(
                        "utility model returned a non-finite value for epsilon={eps}"
                    )));
                }
                predicted_sum += utility;
            }
            let predicted_mean_utility = predicted_sum / n as f64;
            let mean_eps = total_epsilon / n as f64;
            let max_eps = per_iteration
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let allocation_imbalance = if mean_eps > 1e-12 {
                max_eps / mean_eps - 1.0
            } else {
                0.0
            };
            let per_iteration_allocation = per_iteration
                .iter()
                .map(|&v| to_t(v))
                .collect::<Result<Vec<T>>>()?;
            allocations.push(BudgetAllocation {
                total_epsilon,
                per_iteration_allocation,
                allocation_strategy: strategy,
                predicted_mean_utility: to_t(predicted_mean_utility)?,
                allocation_imbalance: to_t(allocation_imbalance)?,
            });
        }
        allocations.sort_by(|a, b| {
            b.predicted_mean_utility
                .partial_cmp(&a.predicted_mean_utility)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(allocations)
    }

    /// Optimize the allocation of a privacy budget over `iterations` steps.
    ///
    /// Returns the allocation with the highest predicted utility under the
    /// caller's `utility_model`, provided it reaches `utility_threshold`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an exhausted or invalid
    /// budget, and [`OptimError::OptimizationError`] when no allocation shape
    /// reaches the requested utility threshold (rather than silently returning
    /// one that does not).
    pub fn optimize_budget_allocation(
        &self,
        total_budget: &PrivacyBudget,
        iterations: usize,
        utility_threshold: T,
        utility_model: &impl Fn(f64) -> f64,
    ) -> Result<BudgetAllocation<T>> {
        if total_budget.epsilon_remaining <= 0.0 {
            return Err(OptimError::InvalidParameter(
                "total_budget.epsilon_remaining must be > 0".to_string(),
            ));
        }
        let mut candidates = self.budget_allocation_candidates(
            total_budget.epsilon_remaining,
            iterations,
            utility_model,
        )?;
        if candidates.is_empty() {
            return Err(OptimError::ComputationError(
                "no budget allocation candidate could be built".to_string(),
            ));
        }
        let best = candidates.remove(0);
        if best.predicted_mean_utility < utility_threshold {
            return Err(OptimError::OptimizationError(format!(
                "no allocation reaches the requested utility threshold {:?}; best predicted utility is {:?}",
                utility_threshold, best.predicted_mean_utility
            )));
        }
        Ok(best)
    }

    /// Sensitivity of the utility oracle to each privacy parameter.
    ///
    /// Derivatives are estimated by finite differences with a step
    /// `h = 0.01 * max(|x|, 1)`, which stays well defined for parameters that
    /// are zero (notably `delta = 0` under pure epsilon-DP). Every perturbed
    /// value is kept inside the parameter's domain, and every quotient is
    /// checked for finiteness, so no NaN can reach the rankings.
    ///
    /// When `AnalysisConfig::monte_carlo_samples > 1`, each derivative is
    /// re-estimated with randomly jittered steps (magnitude in `[0.5h, 1.5h]`,
    /// random direction) and the reported value is the replicate mean, with a
    /// confidence interval derived from the replicate standard error at the
    /// configured confidence level. With a single replicate no interval is
    /// reported, since one estimate carries no uncertainty information.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an invalid base
    /// configuration and [`OptimError::ComputationError`] when the oracle
    /// produces non-finite values.
    pub fn perform_sensitivity_analysis<D: Data<Elem = T>, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(&ArrayBase<D, Dim>, &PrivacyConfiguration<T>) -> Result<T> + Sync,
        base_config: &PrivacyConfiguration<T>,
    ) -> Result<SensitivityResults<T>> {
        validate_privacy_configuration(base_config)?;
        let base_utility_t = model_fn(data, base_config)?;
        let base_utility = to_f64(base_utility_t)?;
        if !base_utility.is_finite() {
            return Err(OptimError::ComputationError(
                "utility oracle returned a non-finite base utility".to_string(),
            ));
        }

        let replicates = self.config.monte_carlo_samples;
        let z = self.z_alpha()?;
        let mut rng = self.rng_for(SENSITIVITY_SALT);

        let mut parameter_sensitivities: HashMap<String, f64> = HashMap::new();
        let mut gradient_magnitudes: HashMap<String, f64> = HashMap::new();
        let mut confidence_intervals: HashMap<String, (f64, f64)> = HashMap::new();
        let mut local_sensitivities: Vec<LocalSensitivity<T>> = Vec::new();

        for parameter in SensitivityParameter::ALL {
            let x = parameter.value(base_config)?;
            let h = RELATIVE_STEP * x.abs().max(1.0);
            let evaluate = |value: f64| -> Result<f64> {
                let config = parameter.with_value(base_config, value)?;
                let utility = to_f64(model_fn(data, &config)?)?;
                if utility.is_finite() {
                    Ok(utility)
                } else {
                    Err(OptimError::ComputationError(format!(
                        "utility oracle returned a non-finite value for {}={value}",
                        parameter.name()
                    )))
                }
            };

            let forward_ok = parameter.admissible(x + h);
            let backward_ok = parameter.admissible(x - h);
            if !forward_ok && !backward_ok {
                return Err(OptimError::InvalidParameter(format!(
                    "parameter {} cannot be perturbed by {h} without leaving its domain",
                    parameter.name()
                )));
            }
            let utility_forward = if forward_ok {
                Some(evaluate(x + h)?)
            } else {
                None
            };
            let utility_backward = if backward_ok {
                Some(evaluate(x - h)?)
            } else {
                None
            };
            let central_derivative = match (utility_forward, utility_backward) {
                (Some(f_plus), Some(f_minus)) => (f_plus - f_minus) / (2.0 * h),
                (Some(f_plus), None) => (f_plus - base_utility) / h,
                (None, Some(f_minus)) => (base_utility - f_minus) / h,
                (None, None) => unreachable_domain(parameter.name())?,
            };
            if !central_derivative.is_finite() {
                return Err(OptimError::ComputationError(format!(
                    "finite difference for {} is not finite",
                    parameter.name()
                )));
            }

            // Second derivative, available only when the parameter can be
            // perturbed in both directions.
            let hessian = match (utility_forward, utility_backward) {
                (Some(f_plus), Some(f_minus)) => {
                    let value = (f_plus - 2.0 * base_utility + f_minus) / (h * h);
                    if value.is_finite() {
                        Some(to_t(value)?)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let (reported_derivative, interval) = if replicates > 1 {
                let mut estimates = Vec::with_capacity(replicates);
                for _ in 0..replicates {
                    let scale: f64 = rng.gen_range(0.5..1.5);
                    let forward: bool = rng.gen_range(0.0..1.0) < 0.5;
                    let mut step = if forward { h * scale } else { -h * scale };
                    if !parameter.admissible(x + step) {
                        step = -step;
                    }
                    if !parameter.admissible(x + step) {
                        continue;
                    }
                    let utility = evaluate(x + step)?;
                    let derivative = (utility - base_utility) / step;
                    if derivative.is_finite() {
                        estimates.push(derivative);
                    }
                }
                if estimates.len() > 1 {
                    let (mean, variance) = stats::mean_var(&estimates);
                    let standard_error = (variance / estimates.len() as f64).sqrt();
                    let margin = z * standard_error;
                    (mean, Some((mean - margin, mean + margin)))
                } else {
                    (central_derivative, None)
                }
            } else {
                (central_derivative, None)
            };

            parameter_sensitivities.insert(parameter.name().to_string(), reported_derivative);
            gradient_magnitudes.insert(parameter.name().to_string(), reported_derivative.abs());
            if let Some((lo, hi)) = interval {
                confidence_intervals.insert(parameter.name().to_string(), (lo, hi));
            }
            local_sensitivities.push(LocalSensitivity {
                parameter: parameter.name().to_string(),
                gradient: to_t(reported_derivative)?,
                hessian,
                step_size: to_t(h)?,
                confidence_interval: match interval {
                    Some((lo, hi)) => Some((to_t(lo)?, to_t(hi)?)),
                    None => None,
                },
            });
        }

        // Mixed second derivative between epsilon and the noise multiplier.
        let mut interaction_effects: HashMap<String, f64> = HashMap::new();
        let eps = SensitivityParameter::Epsilon.value(base_config)?;
        let noise = SensitivityParameter::NoiseMultiplier.value(base_config)?;
        let h_eps = RELATIVE_STEP * eps.abs().max(1.0);
        let h_noise = RELATIVE_STEP * noise.abs().max(1.0);
        if SensitivityParameter::Epsilon.admissible(eps + h_eps)
            && SensitivityParameter::NoiseMultiplier.admissible(noise + h_noise)
        {
            let config_eps = SensitivityParameter::Epsilon.with_value(base_config, eps + h_eps)?;
            let config_noise =
                SensitivityParameter::NoiseMultiplier.with_value(base_config, noise + h_noise)?;
            let config_both =
                SensitivityParameter::NoiseMultiplier.with_value(&config_eps, noise + h_noise)?;
            let f_eps = to_f64(model_fn(data, &config_eps)?)?;
            let f_noise = to_f64(model_fn(data, &config_noise)?)?;
            let f_both = to_f64(model_fn(data, &config_both)?)?;
            let mixed = (f_both - f_eps - f_noise + base_utility) / (h_eps * h_noise);
            if mixed.is_finite() {
                interaction_effects.insert("epsilon_noise_multiplier".to_string(), mixed);
            }
        }

        let mut rankings: Vec<(String, f64)> = gradient_magnitudes
            .iter()
            .map(|(name, magnitude)| (name.clone(), *magnitude))
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let most_sensitive_parameter = rankings
            .first()
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let least_sensitive_parameter = rankings
            .last()
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let max_magnitude = rankings.first().map(|(_, m)| *m).unwrap_or(0.0);
        let min_magnitude = rankings.last().map(|(_, m)| *m).unwrap_or(0.0);
        let sensitivity_rankings = rankings
            .iter()
            .map(|(name, magnitude)| Ok((name.clone(), to_t(*magnitude)?)))
            .collect::<Result<Vec<(String, T)>>>()?;

        Ok(SensitivityResults {
            base_utility: base_utility_t,
            parameter_sensitivities,
            gradient_magnitudes,
            interaction_effects,
            local_sensitivities,
            global_sensitivity_bounds: (to_t(min_magnitude)?, to_t(max_magnitude)?),
            sensitivity_rankings,
            robustness_score: to_t(1.0 / (1.0 + max_magnitude))?,
            most_sensitive_parameter,
            least_sensitive_parameter,
            confidence_intervals,
        })
    }

    /// Predict utility degradation at the requested privacy parameters.
    ///
    /// A polynomial of degree 1-3 (chosen from the number of observations) is
    /// fitted to `historical_data`, given as `(epsilon, relative utility
    /// loss)` pairs. The interval around a prediction is the normal
    /// approximation `z * sigma * sqrt(1 + 1/n)` of a prediction interval,
    /// where `sigma` is the residual standard error of the fit and `z` follows
    /// `AnalysisConfig::confidence_level`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for empty history or too few
    /// distinct abscissae, and propagates regression failures.
    pub fn predict_utility_degradation(
        &self,
        privacy_parameters: &[T],
        historical_data: &[(T, T)],
    ) -> Result<Vec<DegradationPrediction<T>>> {
        if historical_data.is_empty() {
            return Err(OptimError::InvalidParameter(
                "historical_data must not be empty".to_string(),
            ));
        }
        if privacy_parameters.is_empty() {
            return Ok(Vec::new());
        }
        let n = historical_data.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (x, y) in historical_data {
            xs.push(to_f64(*x)?);
            ys.push(to_f64(*y)?);
        }
        let mut distinct = xs.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let degree = if n < 5 {
            1
        } else if n < 15 {
            2
        } else {
            3
        }
        .min(distinct.len().saturating_sub(1));
        if degree == 0 {
            return Err(OptimError::InvalidParameter(
                "historical_data must contain at least two distinct privacy parameters".to_string(),
            ));
        }
        let coefficients = stats::polyfit(&xs, &ys, degree)?;
        let fitted: Vec<f64> = xs
            .iter()
            .map(|&x| stats::polyeval(&coefficients, x))
            .collect();
        let mean_y = ys.iter().sum::<f64>() / n as f64;
        let ss_res: f64 = ys
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| (y - f).powi(2))
            .sum();
        let ss_tot: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();
        let r_squared = if ss_tot > 1e-12 {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let dof = (n as i64 - degree as i64 - 1).max(1) as f64;
        let sigma = (ss_res / dof).sqrt();
        let z = self.z_alpha()?;
        let prediction_model = if degree <= 1 {
            PredictionModel::LinearRegression
        } else {
            PredictionModel::PolynomialRegression
        };

        let mut predictions = Vec::with_capacity(privacy_parameters.len());
        for parameter in privacy_parameters {
            let x = to_f64(*parameter)?;
            let predicted = stats::polyeval(&coefficients, x).clamp(0.0, 1.0);
            let margin = z * sigma * (1.0 + 1.0 / n as f64).sqrt();
            predictions.push(DegradationPrediction {
                privacy_parameter: *parameter,
                predicted_utility_loss: to_t(predicted)?,
                confidence_interval: (
                    to_t((predicted - margin).clamp(0.0, 1.0))?,
                    to_t((predicted + margin).clamp(0.0, 1.0))?,
                ),
                prediction_model: prediction_model.clone(),
                model_accuracy: to_t(r_squared)?,
            });
        }
        Ok(predictions)
    }
}

/// Reached only if the admissibility checks above are inconsistent; reported
/// as an error instead of panicking.
fn unreachable_domain(parameter: &str) -> Result<f64> {
    Err(OptimError::ComputationError(format!(
        "no admissible finite-difference step for parameter {parameter}"
    )))
}
