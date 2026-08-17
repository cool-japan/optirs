//! Robustness evaluation: how much utility the configuration loses under
//! parameter, data, noise, environment and worst-case perturbations.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{ArrayBase, Data, DataOwned, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RandNormal, Random};
use std::fmt::Debug;

use super::types::{
    to_f64, to_t, validate_privacy_configuration, PerturbationAnalysis, PerturbationEffect,
    PerturbationType, PrivacyUtilityAnalyzer,
};
use super::types_3::{
    FailureMode, FailureType, PrivacyConfiguration, RobustnessResults, StabilityAnalysis,
};

/// Relative perturbation magnitudes probed by the robustness evaluation.
const PERTURBATION_LEVELS: [f64; 3] = [0.01, 0.05, 0.20];

/// Number of random data draws per perturbation level.
const DATA_DRAWS: usize = 3;

/// Salt for the data-perturbation random stream.
const ROBUSTNESS_SALT: u64 = 0x_b055_7e55;

/// Utilities observed for one perturbation family at one magnitude.
#[derive(Debug, Clone, Copy)]
struct BranchOutcome {
    /// Worst (lowest) utility observed for this family
    worst_utility: f64,
}

impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Evaluate the robustness of a privacy configuration.
    ///
    /// Five genuinely different perturbation families are probed at each of
    /// the magnitudes `1%`, `5%` and `20%`:
    ///
    /// * **Parameter** - relative perturbation of `epsilon` in both directions.
    /// * **Data** - additive Gaussian perturbation of the *input array*,
    ///   scaled by the magnitude and the sample standard deviation of the
    ///   data. This is what `distributional_robustness` is computed from.
    /// * **Noise** - relative perturbation of the noise multiplier.
    /// * **Environmental** - joint perturbation of clipping threshold and
    ///   learning rate.
    /// * **Adversarial** - the worst point found by searching the perturbation
    ///   ball: every single-coordinate direction, the corner combining each
    ///   coordinate's individually worst direction, and the worst data draw.
    ///   Being a search over a finite subset of the ball, the reported drop is
    ///   a lower bound on the true worst case.
    ///
    /// Degradations are reported relative to `|base utility|`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an invalid configuration
    /// or a zero base utility (relative degradation would be undefined), and
    /// propagates oracle errors.
    pub fn evaluate_robustness<D: Data<Elem = T> + DataOwned, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(&ArrayBase<D, Dim>, &PrivacyConfiguration<T>) -> Result<T> + Sync,
        config: &PrivacyConfiguration<T>,
    ) -> Result<RobustnessResults<T>> {
        validate_privacy_configuration(config)?;
        let base_utility = to_f64(model_fn(data, config)?)?;
        if !base_utility.is_finite() {
            return Err(OptimError::ComputationError(
                "utility oracle returned a non-finite base utility".to_string(),
            ));
        }
        let scale = base_utility.abs();
        if scale < 1e-12 {
            return Err(OptimError::InvalidParameter(
                "relative robustness is undefined for a zero base utility".to_string(),
            ));
        }

        let epsilon = to_f64(config.epsilon)?;
        let noise = to_f64(config.noise_multiplier)?;
        let clip = to_f64(config.clipping_threshold)?;
        let learning_rate = to_f64(config.learning_rate)?;
        let data_scale = sample_spread(data)?;
        let mut rng = self.rng_for(ROBUSTNESS_SALT);

        let mut parameter_branch = Vec::with_capacity(PERTURBATION_LEVELS.len());
        let mut data_branch = Vec::with_capacity(PERTURBATION_LEVELS.len());
        let mut noise_branch = Vec::with_capacity(PERTURBATION_LEVELS.len());
        let mut environment_branch = Vec::with_capacity(PERTURBATION_LEVELS.len());
        let mut adversarial_branch = Vec::with_capacity(PERTURBATION_LEVELS.len());
        let mut all_utilities = vec![base_utility];

        for &level in PERTURBATION_LEVELS.iter() {
            // Parameter family: epsilon moved both ways.
            let mut parameter_worst = f64::INFINITY;
            let mut worst_epsilon = epsilon;
            for signed in [1.0_f64, -1.0] {
                let value = epsilon * (1.0 + signed * level);
                if value <= 0.0 {
                    continue;
                }
                let mut candidate = config.clone();
                candidate.epsilon = to_t(value)?;
                let utility = to_f64(model_fn(data, &candidate)?)?;
                all_utilities.push(utility);
                if utility < parameter_worst {
                    parameter_worst = utility;
                    worst_epsilon = value;
                }
            }

            // Noise family: noise multiplier moved both ways.
            let mut noise_worst = f64::INFINITY;
            let mut worst_noise = noise;
            for signed in [1.0_f64, -1.0] {
                let value = noise * (1.0 + signed * level);
                if value <= 0.0 {
                    continue;
                }
                let mut candidate = config.clone();
                candidate.noise_multiplier = to_t(value)?;
                let utility = to_f64(model_fn(data, &candidate)?)?;
                all_utilities.push(utility);
                if utility < noise_worst {
                    noise_worst = utility;
                    worst_noise = value;
                }
            }

            // Environmental family: clipping threshold and learning rate.
            let mut environment_worst = f64::INFINITY;
            let mut worst_clip = clip;
            let mut worst_lr = learning_rate;
            for signed in [1.0_f64, -1.0] {
                let clip_value = clip * (1.0 + signed * level);
                let lr_value = learning_rate * (1.0 - signed * level);
                if clip_value <= 0.0 || lr_value <= 0.0 {
                    continue;
                }
                let mut candidate = config.clone();
                candidate.clipping_threshold = to_t(clip_value)?;
                candidate.learning_rate = to_t(lr_value)?;
                let utility = to_f64(model_fn(data, &candidate)?)?;
                all_utilities.push(utility);
                if utility < environment_worst {
                    environment_worst = utility;
                    worst_clip = clip_value;
                    worst_lr = lr_value;
                }
            }

            // Data family: the input array itself is perturbed.
            let mut data_worst = f64::INFINITY;
            let mut worst_data: Option<ArrayBase<D, Dim>> = None;
            for _ in 0..DATA_DRAWS {
                let perturbed = perturb_data(data, level * data_scale, &mut rng)?;
                let utility = to_f64(model_fn(&perturbed, config)?)?;
                all_utilities.push(utility);
                if utility < data_worst {
                    data_worst = utility;
                    worst_data = Some(perturbed);
                }
            }

            // Adversarial family: worst single coordinate, worst corner, and
            // the corner combined with the worst data draw.
            let mut adversarial_worst = parameter_worst
                .min(noise_worst)
                .min(environment_worst)
                .min(data_worst);
            let mut corner = config.clone();
            corner.epsilon = to_t(worst_epsilon)?;
            corner.noise_multiplier = to_t(worst_noise)?;
            corner.clipping_threshold = to_t(worst_clip)?;
            corner.learning_rate = to_t(worst_lr)?;
            if validate_privacy_configuration(&corner).is_ok() {
                let utility = to_f64(model_fn(data, &corner)?)?;
                all_utilities.push(utility);
                adversarial_worst = adversarial_worst.min(utility);
                if let Some(perturbed) = worst_data.as_ref() {
                    let utility = to_f64(model_fn(perturbed, &corner)?)?;
                    all_utilities.push(utility);
                    adversarial_worst = adversarial_worst.min(utility);
                }
            }

            parameter_branch.push(BranchOutcome {
                worst_utility: finite_or_base(parameter_worst, base_utility),
            });
            noise_branch.push(BranchOutcome {
                worst_utility: finite_or_base(noise_worst, base_utility),
            });
            environment_branch.push(BranchOutcome {
                worst_utility: finite_or_base(environment_worst, base_utility),
            });
            data_branch.push(BranchOutcome {
                worst_utility: finite_or_base(data_worst, base_utility),
            });
            adversarial_branch.push(BranchOutcome {
                worst_utility: finite_or_base(adversarial_worst, base_utility),
            });
        }

        for utility in &all_utilities {
            if !utility.is_finite() {
                return Err(OptimError::ComputationError(
                    "utility oracle returned a non-finite value during robustness evaluation"
                        .to_string(),
                ));
            }
        }

        let relative_drop = |utility: f64| ((base_utility - utility) / scale).max(0.0);
        let branch_worst_drop = |branch: &[BranchOutcome]| {
            branch
                .iter()
                .map(|outcome| relative_drop(outcome.worst_utility))
                .fold(0.0_f64, f64::max)
        };

        let parameter_drop = branch_worst_drop(&parameter_branch);
        let noise_drop = branch_worst_drop(&noise_branch);
        let environment_drop = branch_worst_drop(&environment_branch);
        let data_drop = branch_worst_drop(&data_branch);
        let adversarial_drop = branch_worst_drop(&adversarial_branch);
        let worst_case_degradation = parameter_drop
            .max(noise_drop)
            .max(environment_drop)
            .max(data_drop)
            .max(adversarial_drop);

        let threshold = self.config.utility_degradation_threshold;
        let largest_level = PERTURBATION_LEVELS[PERTURBATION_LEVELS.len() - 1];

        // Largest probed magnitude whose worst drop stays under the threshold.
        let mut stable_radius: Option<f64> = None;
        for (index, &level) in PERTURBATION_LEVELS.iter().enumerate() {
            let level_drop = relative_drop(parameter_branch[index].worst_utility)
                .max(relative_drop(noise_branch[index].worst_utility))
                .max(relative_drop(environment_branch[index].worst_utility))
                .max(relative_drop(data_branch[index].worst_utility))
                .max(relative_drop(adversarial_branch[index].worst_utility));
            if level_drop <= threshold {
                stable_radius = Some(level);
            } else {
                break;
            }
        }

        // Slope of ln(utility) against the perturbation magnitude, estimated
        // by least squares over the adversarial branch. Only defined when all
        // adversarial utilities are strictly positive.
        let log_utility_sensitivity = if base_utility > 0.0
            && adversarial_branch
                .iter()
                .all(|outcome| outcome.worst_utility > 0.0)
        {
            let xs: Vec<f64> = std::iter::once(0.0)
                .chain(PERTURBATION_LEVELS.iter().copied())
                .collect();
            let ys: Vec<f64> = std::iter::once(base_utility.ln())
                .chain(
                    adversarial_branch
                        .iter()
                        .map(|outcome| outcome.worst_utility.ln()),
                )
                .collect();
            let coefficients = super::stats::polyfit(&xs, &ys, 1)?;
            coefficients.get(1).copied().filter(|v| v.is_finite())
        } else {
            None
        };

        let (mean_utility, variance) = super::stats::mean_var(&all_utilities);
        let coefficient_of_variation = if mean_utility.abs() > 1e-12 {
            Some(variance.sqrt() / mean_utility.abs())
        } else {
            None
        };

        let perturbation_effects = vec![
            perturbation_effect(
                PerturbationType::Parameter,
                largest_level,
                base_utility,
                parameter_branch[parameter_branch.len() - 1].worst_utility,
                scale,
            )?,
            perturbation_effect(
                PerturbationType::Data,
                largest_level,
                base_utility,
                data_branch[data_branch.len() - 1].worst_utility,
                scale,
            )?,
            perturbation_effect(
                PerturbationType::Noise,
                largest_level,
                base_utility,
                noise_branch[noise_branch.len() - 1].worst_utility,
                scale,
            )?,
            perturbation_effect(
                PerturbationType::Adversarial,
                largest_level,
                base_utility,
                adversarial_branch[adversarial_branch.len() - 1].worst_utility,
                scale,
            )?,
            perturbation_effect(
                PerturbationType::Environmental,
                largest_level,
                base_utility,
                environment_branch[environment_branch.len() - 1].worst_utility,
                scale,
            )?,
        ];

        let mut failure_modes: Vec<FailureMode<T>> = Vec::new();
        let first_breach = |branch: &[BranchOutcome]| -> Option<(f64, f64)> {
            for (index, outcome) in branch.iter().enumerate() {
                let drop = relative_drop(outcome.worst_utility);
                if drop > threshold {
                    return Some((PERTURBATION_LEVELS[index], drop));
                }
            }
            None
        };
        if let Some((level, drop)) = first_breach(&parameter_branch) {
            failure_modes.push(FailureMode {
                failure_type: FailureType::ParameterSensitivityFailure,
                observed_relative_drop: to_t(drop)?,
                perturbation_level: to_t(level)?,
                mitigation_strategies: vec![
                    "Choose a configuration in a flatter region of the privacy-utility surface"
                        .to_string(),
                    "Re-run the analysis with a finer parameter grid around this point".to_string(),
                ],
            });
        }
        if let Some((level, drop)) = first_breach(&data_branch) {
            failure_modes.push(FailureMode {
                failure_type: FailureType::DistributionalFailure,
                observed_relative_drop: to_t(drop)?,
                perturbation_level: to_t(level)?,
                mitigation_strategies: vec![
                    "Validate the model against shifted data before deployment".to_string(),
                    "Apply data augmentation or distributionally robust training".to_string(),
                ],
            });
        }
        if let Some((level, drop)) = first_breach(&adversarial_branch) {
            failure_modes.push(FailureMode {
                failure_type: FailureType::AdversarialFailure,
                observed_relative_drop: to_t(drop)?,
                perturbation_level: to_t(level)?,
                mitigation_strategies: vec![
                    "Budget for the worst case rather than the nominal configuration".to_string(),
                    "Increase the noise multiplier to flatten the utility surface".to_string(),
                ],
            });
        }
        if worst_case_degradation > threshold {
            failure_modes.push(FailureMode {
                failure_type: FailureType::UtilityCollapse,
                observed_relative_drop: to_t(worst_case_degradation)?,
                perturbation_level: to_t(largest_level)?,
                mitigation_strategies: vec![
                    "Reduce the per-iteration privacy cost and retrain".to_string(),
                    "Increase the amount of training data".to_string(),
                ],
            });
        }

        Ok(RobustnessResults {
            robustness_score: to_t(1.0 / (1.0 + worst_case_degradation))?,
            worst_case_degradation: to_t(worst_case_degradation)?,
            adversarial_robustness: to_t(1.0 / (1.0 + adversarial_drop))?,
            distributional_robustness: to_t(1.0 / (1.0 + data_drop))?,
            stability_analysis: StabilityAnalysis {
                log_utility_sensitivity: log_utility_sensitivity.map(to_t).transpose()?,
                utility_coefficient_of_variation: coefficient_of_variation.map(to_t).transpose()?,
                perturbation_analysis: PerturbationAnalysis {
                    perturbation_sensitivity: to_t(worst_case_degradation / largest_level)?,
                    stable_perturbation_radius: stable_radius.map(to_t).transpose()?,
                    perturbation_effects,
                },
            },
            failure_modes,
        })
    }
}

/// Effect of one perturbation family at one magnitude.
fn perturbation_effect<T: Float + Debug + Send + Sync + 'static>(
    perturbation_type: PerturbationType,
    magnitude: f64,
    base_utility: f64,
    perturbed_utility: f64,
    scale: f64,
) -> Result<PerturbationEffect<T>> {
    let drop = (base_utility - perturbed_utility).max(0.0);
    Ok(PerturbationEffect {
        perturbation_type,
        perturbation_magnitude: to_t(magnitude)?,
        utility_drop: to_t(drop)?,
        relative_utility_drop: to_t(drop / scale)?,
    })
}

/// Replace a branch outcome that never produced an admissible candidate with
/// the unperturbed utility, so it contributes no phantom degradation.
fn finite_or_base(value: f64, base: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        base
    }
}

/// Sample standard deviation of the data, falling back to its root mean
/// square and finally to `1.0`, so that a constant array still receives a
/// perturbation of a meaningful magnitude.
fn sample_spread<T: Float + Debug + Send + Sync + 'static, D: Data<Elem = T>, Dim: Dimension>(
    data: &ArrayBase<D, Dim>,
) -> Result<f64> {
    let mut values = Vec::with_capacity(data.len());
    for value in data.iter() {
        values.push(to_f64(*value)?);
    }
    if values.is_empty() {
        return Ok(1.0);
    }
    let (_, variance) = super::stats::mean_var(&values);
    let std_dev = variance.sqrt();
    if std_dev > 1e-12 {
        return Ok(std_dev);
    }
    let rms = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
    if rms > 1e-12 {
        Ok(rms)
    } else {
        Ok(1.0)
    }
}

/// Additive Gaussian perturbation of the input data with standard deviation
/// `sigma`, drawn from the analyzer's seeded generator.
fn perturb_data<T: Float + Debug + Send + Sync + 'static, D, Dim: Dimension>(
    data: &ArrayBase<D, Dim>,
    sigma: f64,
    rng: &mut Random<StdRng>,
) -> Result<ArrayBase<D, Dim>>
where
    D: Data<Elem = T> + DataOwned,
{
    let normal = RandNormal::new(0.0_f64, sigma.max(f64::MIN_POSITIVE))
        .map_err(|_| OptimError::ComputationError(format!("invalid perturbation scale {sigma}")))?;
    let mut values = Vec::with_capacity(data.len());
    for value in data.iter() {
        let noise: f64 = rng.sample(normal);
        values.push(to_t::<T>(to_f64(*value)? + noise)?);
    }
    ArrayBase::<D, Dim>::from_shape_vec(data.raw_dim(), values).map_err(|err| {
        OptimError::DimensionMismatch(format!("could not rebuild the perturbed array: {err}"))
    })
}
