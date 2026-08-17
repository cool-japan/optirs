//! Core types and entry point of the privacy-utility tradeoff analyzer.
//!
//! The analyzer explores a space of differential-privacy configurations,
//! evaluates a user supplied utility oracle on each of them, and reports the
//! Pareto frontier together with sensitivity, robustness, budget, risk and
//! statistical-test results.
//!
//! Two rules govern every number produced here:
//!
//! 1. Nothing is reported unless it was computed from the user's oracle or
//!    from a documented closed-form bound. Quantities that cannot be measured
//!    are reported as `None`/empty ("not assessed") rather than as a plausible
//!    looking default.
//! 2. Every randomized step is driven by the seed reported in
//!    [`AnalysisMetadata`], so an analysis can be reproduced exactly.

use crate::error::{OptimError, Result};
use crate::privacy::{DifferentialPrivacyConfig, NoiseMechanism};
use scirs2_core::ndarray::{ArrayBase, Data, DataOwned, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Random};
use std::collections::HashMap;
use std::fmt::Debug;

use super::stats;
use super::types_3::{
    AnalysisConfig, AnalysisMetadata, BudgetEfficiencyMetrics, BudgetRecommendations,
    DegradationPrediction, EpsilonSemantics, LocalSensitivity, OptimizationObjective,
    ParameterRange, ParetoPoint, PrivacyConfiguration, PrivacyRiskAssessment, RobustnessResults,
    SamplingStrategy, StatisticalTestResults,
};

/// Convert an `f64` into the analyzer's float type, failing loudly instead of
/// substituting a plausible looking default.
pub(super) fn to_t<T: Float>(value: f64) -> Result<T> {
    T::from(value).ok_or_else(|| {
        OptimError::ComputationError(format!(
            "value {value} is not representable in the target float type"
        ))
    })
}

/// Convert the analyzer's float type into `f64`.
pub(super) fn to_f64<T: Float + Debug>(value: T) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        OptimError::ComputationError(format!("value {value:?} is not representable as f64"))
    })
}

/// Validate a candidate privacy configuration.
///
/// # Errors
/// Returns [`OptimError::InvalidParameter`] when any field is outside the
/// domain on which the analysis routines are defined. This is what keeps the
/// finite-difference machinery from dividing by zero.
pub(super) fn validate_privacy_configuration<T: Float + Debug + Send + Sync + 'static>(
    config: &PrivacyConfiguration<T>,
) -> Result<()> {
    let epsilon = to_f64(config.epsilon)?;
    let delta = to_f64(config.delta)?;
    let noise = to_f64(config.noise_multiplier)?;
    let clip = to_f64(config.clipping_threshold)?;
    let sampling = to_f64(config.sampling_probability)?;
    let lr = to_f64(config.learning_rate)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires a finite epsilon > 0, got {epsilon}"
        )));
    }
    if !delta.is_finite() || !(0.0..1.0).contains(&delta) {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires delta in [0, 1), got {delta}"
        )));
    }
    if !noise.is_finite() || noise <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires a finite noise_multiplier > 0, got {noise}"
        )));
    }
    if !clip.is_finite() || clip <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires a finite clipping_threshold > 0, got {clip}"
        )));
    }
    if !sampling.is_finite() || sampling <= 0.0 || sampling > 1.0 {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires sampling_probability in (0, 1], got {sampling}"
        )));
    }
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimError::InvalidParameter(format!(
            "privacy configuration requires a finite learning_rate > 0, got {lr}"
        )));
    }
    if config.iterations == 0 {
        return Err(OptimError::InvalidParameter(
            "privacy configuration requires iterations > 0".to_string(),
        ));
    }
    if config.batch_size == 0 {
        return Err(OptimError::InvalidParameter(
            "privacy configuration requires batch_size > 0".to_string(),
        ));
    }
    Ok(())
}

/// Optimal configuration recommendation
#[derive(Debug, Clone)]
pub struct OptimalConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Privacy parameters. Fields the analysis does not explore keep the
    /// library defaults of [`DifferentialPrivacyConfig`].
    pub privacy_config: DifferentialPrivacyConfig,
    /// Utility reported by the oracle for this configuration
    pub expected_utility: T,
    /// Composed privacy guarantee (total epsilon) of this configuration
    pub privacy_guarantee: T,
    /// Objective this configuration optimizes
    pub objective: OptimizationObjective,
    /// `1 - p` of the frontier utility comparison that was actually run, or
    /// `None` when the frontier was too small to run that test. This is never
    /// a placeholder: no test, no score.
    pub confidence_score: Option<T>,
    /// Utility per unit of composed epsilon
    pub tradeoff_ratio: T,
}
/// Computational resources used by an analysis run
#[derive(Debug, Clone)]
pub struct ComputationalResources {
    /// Wall-clock duration of the analysis on the calling thread
    pub wall_time: std::time::Duration,
    /// Logical cores the process may use, as reported by
    /// [`std::thread::available_parallelism`]; `None` when the platform does
    /// not report it. The analysis itself runs on the calling thread.
    pub available_parallelism: Option<usize>,
    /// Peak memory usage. Always `None`: the crate takes no dependency on a
    /// process-introspection library, so this is not measured.
    pub peak_memory_bytes: Option<usize>,
    /// GPU usage. Always `None`: the analyzer runs on the CPU.
    pub gpu_usage: Option<GpuUsage>,
}
/// GPU usage information
#[derive(Debug, Clone)]
pub struct GpuUsage {
    /// GPU time used
    pub gpu_time: std::time::Duration,
    /// GPU memory usage
    pub gpu_memory_usage: usize,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
}
/// Reproducibility information
#[derive(Debug, Clone)]
pub struct ReproducibilityInfo {
    /// Seed that drove every randomized step of the run. Feeding this value
    /// back through `AnalysisConfig::random_seed` reproduces the analysis.
    pub random_seed: u64,
    /// Software versions relevant to the result
    pub software_versions: HashMap<String, String>,
    /// Target triple components the binary was compiled for
    pub hardware_info: String,
}
/// Utility metrics for evaluation
///
/// The analyzer consumes a single scalar utility oracle, so this enum is a
/// label the caller can use to record which metric that oracle implements.
#[derive(Debug, Clone)]
pub enum UtilityMetric {
    /// Model accuracy
    Accuracy,
    /// Model precision
    Precision,
    /// Model recall
    Recall,
    /// F1 score
    F1Score,
    /// Area under ROC curve
    AUROC,
    /// Area under precision-recall curve
    AUPRC,
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// Cross-entropy loss
    CrossEntropy,
    /// Log-likelihood
    LogLikelihood,
    /// Mutual information
    MutualInformation,
    /// Convergence rate
    ConvergenceRate,
    /// Training stability
    TrainingStability,
    /// Generalization gap
    GeneralizationGap,
    /// Custom metric
    Custom(String),
}
/// Perturbation analysis
#[derive(Debug, Clone)]
pub struct PerturbationAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Largest relative utility drop divided by the largest perturbation
    /// magnitude that was probed
    pub perturbation_sensitivity: T,
    /// Largest probed perturbation magnitude at which the relative utility
    /// drop stayed below `AnalysisConfig::utility_degradation_threshold`;
    /// `None` when even the smallest probed magnitude breached it
    pub stable_perturbation_radius: Option<T>,
    /// Per-perturbation-type effects at the largest probed magnitude
    pub perturbation_effects: Vec<PerturbationEffect<T>>,
}
/// Perturbation effect
#[derive(Debug, Clone)]
pub struct PerturbationEffect<T: Float + Debug + Send + Sync + 'static> {
    /// Perturbation type
    pub perturbation_type: PerturbationType,
    /// Perturbation magnitude at which the effect was measured
    pub perturbation_magnitude: T,
    /// Absolute utility drop relative to the unperturbed configuration
    pub utility_drop: T,
    /// Utility drop as a fraction of the unperturbed utility
    pub relative_utility_drop: T,
}
/// Types of perturbations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerturbationType {
    /// Privacy-parameter perturbation (epsilon)
    Parameter,
    /// Perturbation of the input data itself
    Data,
    /// Noise-multiplier perturbation
    Noise,
    /// Worst case found by searching the perturbation ball
    Adversarial,
    /// Optimization-environment perturbation (clipping threshold, learning rate)
    Environmental,
}
/// Multiple comparison correction
#[derive(Debug, Clone)]
pub struct MultipleComparisonCorrection<T: Float + Debug + Send + Sync + 'static> {
    /// Correction method
    pub correction_method: CorrectionMethodApplied,
    /// Adjusted p-values, in the order of the tests they belong to
    pub adjusted_p_values: Vec<T>,
    /// Level at which this procedure *controls* the family-wise error rate
    /// (not an estimate of it); `None` when the procedure does not control it
    pub controlled_family_wise_error_rate: Option<T>,
    /// Level at which this procedure *controls* the false discovery rate
    /// (not an estimate of it)
    pub controlled_false_discovery_rate: Option<T>,
}
/// Multiple comparison correction methods that are actually implemented
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionMethodApplied {
    /// Bonferroni correction
    Bonferroni,
    /// Holm-Bonferroni step-down correction
    HolmBonferroni,
    /// Benjamini-Hochberg step-up correction
    BenjaminiHochberg,
}
/// Privacy-utility analysis results
///
/// Every optional field is `None` exactly when the corresponding analysis
/// could not be run (disabled in the configuration, or not enough Pareto
/// points to support it).
#[derive(Debug, Clone)]
pub struct PrivacyUtilityResults<T: Float + Debug + Send + Sync + 'static> {
    /// Pareto frontier points
    pub pareto_frontier: Vec<ParetoPoint<T>>,
    /// Optimal privacy-utility configurations
    pub optimal_configurations: Vec<OptimalConfiguration<T>>,
    /// Sensitivity analysis results
    pub sensitivity_results: Option<SensitivityResults<T>>,
    /// Robustness evaluation results
    pub robustness_results: Option<RobustnessResults<T>>,
    /// Budget allocation recommendations
    pub budget_recommendations: Option<BudgetRecommendations<T>>,
    /// Utility degradation predictions; empty when the frontier does not
    /// support a regression
    pub degradation_predictions: Vec<DegradationPrediction<T>>,
    /// Privacy risk assessment of the recommended configuration
    pub privacy_risk_assessment: Option<PrivacyRiskAssessment<T>>,
    /// Statistical significance tests over the frontier
    pub statistical_tests: Option<StatisticalTestResults<T>>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}
/// Risk categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskCategory {
    /// Membership inference risk
    MembershipInference,
    /// Attribute inference risk
    AttributeInference,
    /// Model inversion risk
    ModelInversion,
    /// Property inference risk
    PropertyInference,
    /// Reconstruction risk
    Reconstruction,
    /// Re-identification risk
    ReIdentification,
}
/// Sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityResults<T: Float + Debug + Send + Sync + 'static> {
    /// Utility of the unperturbed base configuration
    pub base_utility: T,
    /// Finite-difference derivative of utility with respect to each parameter
    pub parameter_sensitivities: HashMap<String, f64>,
    /// Absolute value of the derivatives above
    pub gradient_magnitudes: HashMap<String, f64>,
    /// Mixed second derivatives between parameter pairs
    pub interaction_effects: HashMap<String, f64>,
    /// Per-parameter local analysis (first and second derivative)
    pub local_sensitivities: Vec<LocalSensitivity<T>>,
    /// Smallest and largest gradient magnitude observed over the analyzed
    /// parameters
    pub global_sensitivity_bounds: (T, T),
    /// Parameters ranked by gradient magnitude, largest first
    pub sensitivity_rankings: Vec<(String, T)>,
    /// `1 / (1 + max gradient magnitude)`: a bounded reparameterization of the
    /// largest sensitivity, not a probability
    pub robustness_score: T,
    /// Most sensitive parameter
    pub most_sensitive_parameter: String,
    /// Least sensitive parameter
    pub least_sensitive_parameter: String,
    /// Confidence intervals of the derivatives, derived from the spread of
    /// `AnalysisConfig::monte_carlo_samples` replicate estimates. Empty when
    /// only one replicate was requested, since a single estimate carries no
    /// uncertainty information.
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}
/// Power analysis
#[derive(Debug, Clone)]
pub struct PowerAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Power of the two-sample test at the observed effect size and sample
    /// size: `Phi(|d| sqrt(n/2) - z_alpha)`
    pub statistical_power: T,
    /// Per-group sample size needed to reach `AnalysisConfig::target_power`
    /// at the observed effect size: `2 ((z_alpha + z_beta) / d)^2`; `None`
    /// when the observed effect is numerically zero
    pub required_sample_size: Option<usize>,
    /// Effect size detectable at the observed per-group sample size with the
    /// configured power: `(z_alpha + z_beta) sqrt(2/n)`
    pub minimum_detectable_effect: T,
    /// `(effect size, power)` pairs at the observed sample size
    pub power_curve: Vec<(T, T)>,
}
/// Risk evolution over time
#[derive(Debug, Clone)]
pub struct RiskEvolution<T: Float + Debug + Send + Sync + 'static> {
    /// Number of composed iterations this point describes
    pub time_point: usize,
    /// Risk score at that point
    pub risk_score: T,
    /// Contributing factors
    pub contributing_factors: Vec<String>,
    /// Risk trend
    pub risk_trend: super::types_3::RiskTrend,
}

/// Comprehensive privacy-utility tradeoff analyzer
pub struct PrivacyUtilityAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration for analysis (validated at construction)
    pub(super) config: AnalysisConfig,
    /// Seed backing every randomized step of the analysis
    pub(super) resolved_seed: u64,
    /// The analyzer is generic over the float type used by the utility oracle
    pub(super) element_type: std::marker::PhantomData<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> PrivacyUtilityAnalyzer<T> {
    /// Create a new privacy-utility analyzer.
    ///
    /// The configuration is validated up front, and the seed that will drive
    /// every randomized step is resolved here: either the caller's
    /// `random_seed` or a single draw from system entropy, which is then
    /// reported in [`AnalysisMetadata`] so the run can be reproduced.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] when the configuration is
    /// invalid (see [`AnalysisConfig::validate`]).
    pub fn new(config: AnalysisConfig) -> Result<Self> {
        config.validate()?;
        let resolved_seed = match config.random_seed {
            Some(seed) => seed,
            None => thread_rng().gen_range(0_u64..u64::MAX),
        };
        Ok(Self {
            config,
            resolved_seed,
            element_type: std::marker::PhantomData,
        })
    }

    /// Configuration this analyzer was built with.
    pub fn config(&self) -> &AnalysisConfig {
        &self.config
    }

    /// Seed driving every randomized step of the analysis.
    pub fn seed(&self) -> u64 {
        self.resolved_seed
    }

    /// Deterministic per-purpose random generator derived from the resolved
    /// seed, so that independent randomized steps neither share nor collide
    /// with each other's streams while the whole run stays reproducible.
    pub(super) fn rng_for(&self, salt: u64) -> Random<StdRng> {
        Random::seed(self.resolved_seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    /// Two-sided critical value implied by `AnalysisConfig::confidence_level`.
    ///
    /// # Errors
    /// Propagates [`OptimError::InvalidParameter`] for a degenerate level.
    pub(super) fn z_alpha(&self) -> Result<f64> {
        stats::z_for_confidence(self.config.confidence_level)
    }

    /// One-sided critical value implied by `AnalysisConfig::target_power`.
    ///
    /// # Errors
    /// Propagates [`OptimError::InvalidParameter`] for a degenerate power.
    pub(super) fn z_beta(&self) -> Result<f64> {
        stats::z_for_power(self.config.target_power)
    }

    /// Significance level `alpha = 1 - confidence_level`.
    pub(super) fn alpha(&self) -> f64 {
        1.0 - self.config.confidence_level
    }

    /// Compose the `(epsilon, delta)` of a configuration into the guarantee of
    /// the whole run, following [`AnalysisConfig::epsilon_semantics`].
    ///
    /// With [`EpsilonSemantics::PerIteration`] this is basic (sequential)
    /// composition: `eps_total = k * eps` and `delta_total = 1 - (1-delta)^k`
    /// for `k` iterations. The exponent is evaluated in floating point, so a
    /// large `k` saturates towards `1` instead of wrapping.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an invalid configuration.
    pub(super) fn composed_budget(&self, config: &PrivacyConfiguration<T>) -> Result<(f64, f64)> {
        validate_privacy_configuration(config)?;
        let epsilon = to_f64(config.epsilon)?;
        let delta = to_f64(config.delta)?;
        match self.config.epsilon_semantics {
            EpsilonSemantics::Total => Ok((epsilon, delta)),
            EpsilonSemantics::PerIteration => {
                let k = config.iterations as f64;
                let eps_total = epsilon * k;
                let delta_total = 1.0 - (1.0 - delta).powf(k);
                Ok((eps_total, delta_total.clamp(0.0, 1.0)))
            }
        }
    }

    /// Perform a comprehensive privacy-utility analysis.
    ///
    /// The frontier is built first; every downstream analysis is then run
    /// against real points of that frontier. Analyses that the frontier cannot
    /// support are reported as `None` rather than filled with defaults.
    ///
    /// # Errors
    /// Propagates oracle errors, and returns [`OptimError::InvalidParameter`]
    /// when the parameter space yields no usable configuration.
    pub fn analyze<D: Data<Elem = T> + DataOwned, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(&ArrayBase<D, Dim>, &PrivacyConfiguration<T>) -> Result<T> + Sync,
    ) -> Result<PrivacyUtilityResults<T>> {
        let start_time = std::time::Instant::now();
        let pareto_frontier = self.generate_pareto_frontier(data, &model_fn)?;

        // Statistical comparison of the strict-privacy half of the frontier
        // against the permissive half. This is the only test the analysis has
        // the data to run, and it is what `confidence_score` is derived from.
        let statistical_tests = self.frontier_statistical_tests(&pareto_frontier)?;
        let confidence_score = match statistical_tests.as_ref() {
            Some(tests) => tests
                .hypothesis_tests
                .first()
                .map(|test| T::one() - test.p_value),
            None => None,
        };

        let best_utility_point = pareto_frontier.iter().try_fold(
            None::<&ParetoPoint<T>>,
            |acc: Option<&ParetoPoint<T>>, point| -> Result<Option<&ParetoPoint<T>>> {
                Ok(match acc {
                    Some(current) if current.utility_value >= point.utility_value => Some(current),
                    _ => Some(point),
                })
            },
        )?;
        let strictest_privacy_point =
            pareto_frontier
                .iter()
                .fold(None::<&ParetoPoint<T>>, |acc, point| match acc {
                    Some(current) if current.privacy_cost <= point.privacy_cost => Some(current),
                    _ => Some(point),
                });

        let mut optimal_configurations = Vec::new();
        if let Some(point) = best_utility_point {
            optimal_configurations.push(self.optimal_configuration_from_point(
                data,
                point,
                OptimizationObjective::MaximizeUtility,
                confidence_score,
            )?);
        }
        if let Some(point) = strictest_privacy_point {
            optimal_configurations.push(self.optimal_configuration_from_point(
                data,
                point,
                OptimizationObjective::MinimizePrivacyLoss,
                confidence_score,
            )?);
        }

        let sensitivity_results = match (
            self.config.enable_sensitivity_analysis,
            pareto_frontier.is_empty(),
        ) {
            (true, false) => {
                let base_config = &pareto_frontier[pareto_frontier.len() / 2].configuration;
                Some(self.perform_sensitivity_analysis(data, &model_fn, base_config)?)
            }
            _ => None,
        };

        let robustness_results =
            match (self.config.enable_robustness_evaluation, best_utility_point) {
                (true, Some(point)) => {
                    Some(self.evaluate_robustness(data, &model_fn, &point.configuration)?)
                }
                _ => None,
            };

        let budget_recommendations = match best_utility_point {
            Some(point) => self.budget_recommendations_for(&pareto_frontier, point)?,
            None => None,
        };

        let degradation_predictions = self.frontier_degradation_predictions(&pareto_frontier)?;

        let privacy_risk_assessment = match best_utility_point {
            Some(point) => Some(self.assess_privacy_risk(&point.configuration)?),
            None => None,
        };

        let metadata = self.build_metadata(start_time)?;

        Ok(PrivacyUtilityResults {
            pareto_frontier,
            optimal_configurations,
            sensitivity_results,
            robustness_results,
            budget_recommendations,
            degradation_predictions,
            privacy_risk_assessment,
            statistical_tests,
            metadata,
        })
    }

    /// Build an [`OptimalConfiguration`] from a frontier point.
    fn optimal_configuration_from_point<D: Data<Elem = T>, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        point: &ParetoPoint<T>,
        objective: OptimizationObjective,
        confidence_score: Option<T>,
    ) -> Result<OptimalConfiguration<T>> {
        let (eps_total, delta_total) = self.composed_budget(&point.configuration)?;
        if eps_total <= 0.0 {
            return Err(OptimError::ComputationError(
                "composed epsilon must be positive".to_string(),
            ));
        }
        let privacy_config = DifferentialPrivacyConfig {
            target_epsilon: eps_total,
            target_delta: delta_total,
            noise_multiplier: to_f64(point.configuration.noise_multiplier)?,
            l2_norm_clip: to_f64(point.configuration.clipping_threshold)?,
            batch_size: point.configuration.batch_size,
            dataset_size: data.len(),
            max_steps: point.configuration.iterations,
            noise_mechanism: point.configuration.noise_mechanism,
            ..DifferentialPrivacyConfig::default()
        };
        Ok(OptimalConfiguration {
            privacy_config,
            expected_utility: point.utility_value,
            privacy_guarantee: to_t(eps_total)?,
            objective,
            confidence_score,
            tradeoff_ratio: point.utility_value / to_t(eps_total)?,
        })
    }

    /// Generate the Pareto frontier of privacy-utility tradeoffs.
    ///
    /// Dominance is evaluated on all three objectives that the analysis
    /// actually varies: composed epsilon and composed delta (to be minimized)
    /// and utility (to be maximized). No weighted scalarization is involved.
    ///
    /// # Errors
    /// Propagates oracle and configuration errors.
    pub fn generate_pareto_frontier<D: Data<Elem = T>, Dim: Dimension>(
        &self,
        data: &ArrayBase<D, Dim>,
        model_fn: impl Fn(&ArrayBase<D, Dim>, &PrivacyConfiguration<T>) -> Result<T> + Sync,
    ) -> Result<Vec<ParetoPoint<T>>> {
        let privacy_configs = self.generate_privacy_configurations()?;
        let mut evaluated: Vec<(ParetoPoint<T>, f64, f64)> =
            Vec::with_capacity(privacy_configs.len());
        for config in privacy_configs {
            let (eps_total, delta_total) = self.composed_budget(&config)?;
            let utility = model_fn(data, &config)?;
            let utility_f64 = to_f64(utility)?;
            if !utility_f64.is_finite() {
                return Err(OptimError::ComputationError(format!(
                    "utility oracle returned a non-finite value ({utility_f64}) for epsilon={eps_total}"
                )));
            }
            let point = ParetoPoint {
                privacy_guarantee: to_t(eps_total)?,
                utility_value: utility,
                configuration: config,
                confidence_interval: None,
                privacy_cost: to_t(eps_total)?,
                dominated: false,
                distance_to_ideal: T::zero(),
            };
            evaluated.push((point, eps_total, delta_total));
        }

        let mut pareto_points = Vec::new();
        for i in 0..evaluated.len() {
            let (point_i, eps_i, delta_i) = (&evaluated[i].0, evaluated[i].1, evaluated[i].2);
            let utility_i = to_f64(point_i.utility_value)?;
            let mut is_dominated = false;
            for (j, eval_j) in evaluated.iter().enumerate() {
                if i == j {
                    continue;
                }
                let (point_j, eps_j, delta_j) = (&eval_j.0, eval_j.1, eval_j.2);
                let utility_j = to_f64(point_j.utility_value)?;
                let no_worse = eps_j <= eps_i && delta_j <= delta_i && utility_j >= utility_i;
                let strictly_better = eps_j < eps_i || delta_j < delta_i || utility_j > utility_i;
                if no_worse && strictly_better {
                    is_dominated = true;
                    break;
                }
            }
            if !is_dominated {
                let mut point = point_i.clone();
                point.dominated = false;
                pareto_points.push(point);
            }
        }

        pareto_points.sort_by(|a, b| {
            a.privacy_cost
                .partial_cmp(&b.privacy_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !pareto_points.is_empty() {
            let min_privacy = pareto_points
                .iter()
                .map(|p| p.privacy_cost)
                .fold(T::infinity(), |a, b| a.min(b));
            let max_utility = pareto_points
                .iter()
                .map(|p| p.utility_value)
                .fold(T::neg_infinity(), |a, b| a.max(b));
            for point in &mut pareto_points {
                let privacy_dist = point.privacy_cost - min_privacy;
                let utility_dist = max_utility - point.utility_value;
                point.distance_to_ideal =
                    (privacy_dist * privacy_dist + utility_dist * utility_dist).sqrt();
            }
        }
        Ok(pareto_points)
    }

    /// Generate the grid of privacy configurations to evaluate.
    ///
    /// Each of the six explored dimensions is sampled across its **full**
    /// configured range; the resulting grid is then strided down to
    /// `pareto_resolution` points. Striding the flat (mixed-radix) combination
    /// index keeps every dimension's full span represented, which a
    /// take-the-first-N truncation would not.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] when no valid configuration
    /// remains after filtering, and propagates range validation errors.
    pub(super) fn generate_privacy_configurations(&self) -> Result<Vec<PrivacyConfiguration<T>>> {
        let params = &self.config.privacy_parameters;
        params.validate()?;
        let resolution = self.config.pareto_resolution;

        // Six explored dimensions; the per-dimension budget is chosen so that
        // the full grid stays on the order of `pareto_resolution` points.
        const DIMENSIONS: usize = 6;
        let budget_per_dimension =
            ((resolution as f64).powf(1.0 / DIMENSIONS as f64).ceil() as usize).max(2);

        let epsilon_values = self.sample_range(&params.epsilon_range, budget_per_dimension, 1)?;
        let delta_values = self.sample_range(&params.delta_range, budget_per_dimension, 2)?;
        let noise_values =
            self.sample_range(&params.noise_multiplier_range, budget_per_dimension, 3)?;
        let clip_values =
            self.sample_range(&params.clipping_threshold_range, budget_per_dimension, 4)?;
        let batch_values = self.sample_range(&params.batch_size_range, budget_per_dimension, 5)?;
        let iteration_values =
            self.sample_range(&params.iterations_range, budget_per_dimension, 6)?;

        // Dimensions that are not part of the grid are held at the midpoint of
        // their configured range instead of at a magic constant.
        let sampling_probability = params
            .sampling_probability_range
            .midpoint()?
            .clamp(1e-12, 1.0);
        let learning_rate = params.learning_rate_range.midpoint()?;

        let mut candidates: Vec<PrivacyConfiguration<T>> = Vec::new();
        for &epsilon in &epsilon_values {
            for &delta in &delta_values {
                for &noise_multiplier in &noise_values {
                    for &clipping_threshold in &clip_values {
                        for &batch_size in &batch_values {
                            for &iterations in &iteration_values {
                                let config = PrivacyConfiguration {
                                    epsilon: to_t(epsilon)?,
                                    delta: to_t(delta)?,
                                    noise_multiplier: to_t(noise_multiplier)?,
                                    clipping_threshold: to_t(clipping_threshold)?,
                                    sampling_probability: to_t(sampling_probability)?,
                                    iterations: iterations.round().max(1.0) as usize,
                                    batch_size: batch_size.round().max(1.0) as usize,
                                    learning_rate: to_t(learning_rate)?,
                                    noise_mechanism: NoiseMechanism::Gaussian,
                                };
                                if validate_privacy_configuration(&config).is_ok() {
                                    candidates.push(config);
                                }
                            }
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            return Err(OptimError::InvalidParameter(
                "the configured privacy parameter space contains no valid configuration"
                    .to_string(),
            ));
        }
        Ok(stride_select(candidates, resolution))
    }

    /// Sample a parameter range at `count` positions spanning its full extent.
    ///
    /// Deterministic strategies place the samples at the two endpoints and
    /// evenly in between (in log space for
    /// [`SamplingStrategy::Logarithmic`]). [`SamplingStrategy::Random`] draws
    /// from the analyzer's seeded generator, so the draw is reproducible.
    /// `count` is capped by the range's own `num_samples`.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for an invalid range and
    /// [`OptimError::UnsupportedOperation`] for a sampling strategy that is
    /// declared but not implemented, rather than silently substituting linear
    /// sampling.
    pub(super) fn sample_range(
        &self,
        range: &ParameterRange,
        count: usize,
        salt: u64,
    ) -> Result<Vec<f64>> {
        range.validate()?;
        let count = count.clamp(1, range.num_samples.max(1));
        match range.sampling_strategy {
            SamplingStrategy::Linear => Ok(even_samples(range.min, range.max, count)),
            SamplingStrategy::Logarithmic => {
                let log_samples = even_samples(range.min.ln(), range.max.ln(), count);
                Ok(log_samples.into_iter().map(f64::exp).collect())
            }
            SamplingStrategy::Random => {
                let mut rng = self.rng_for(salt);
                Ok((0..count)
                    .map(|_| rng.gen_range(range.min..range.max))
                    .collect())
            }
            SamplingStrategy::LatinHypercube
            | SamplingStrategy::Sobol
            | SamplingStrategy::Adaptive => Err(OptimError::UnsupportedOperation(format!(
                "sampling strategy {:?} is not implemented; use Linear, Logarithmic or Random",
                range.sampling_strategy
            ))),
        }
    }

    /// Privacy cost of a configuration: its composed epsilon.
    ///
    /// The delta of a configuration is *not* folded into this scalar; it is a
    /// separate objective of the dominance test in
    /// [`Self::generate_pareto_frontier`].
    ///
    /// # Errors
    /// Propagates configuration validation errors.
    pub fn compute_privacy_cost(&self, config: &PrivacyConfiguration<T>) -> Result<T> {
        let (eps_total, _) = self.composed_budget(config)?;
        to_t(eps_total)
    }

    /// Statistical comparison of the strict-privacy half of the frontier
    /// against the permissive half.
    ///
    /// Returns `None` when either half has fewer than two points, in which
    /// case no test was run at all.
    ///
    /// # Errors
    /// Propagates numerical errors from the test machinery.
    fn frontier_statistical_tests(
        &self,
        frontier: &[ParetoPoint<T>],
    ) -> Result<Option<StatisticalTestResults<T>>> {
        if frontier.len() < 4 {
            return Ok(None);
        }
        let split = frontier.len() / 2;
        let strict: Vec<(T, T)> = frontier[..split]
            .iter()
            .map(|p| (p.privacy_cost, p.utility_value))
            .collect();
        let permissive: Vec<(T, T)> = frontier[split..]
            .iter()
            .map(|p| (p.privacy_cost, p.utility_value))
            .collect();
        if strict.len() < 2 || permissive.len() < 2 {
            return Ok(None);
        }
        Ok(Some(self.perform_statistical_tests(&permissive, &strict)?))
    }

    /// Fit utility degradation across the frontier and predict it at evenly
    /// spaced epsilon values.
    ///
    /// Returns an empty vector when the frontier does not contain enough
    /// distinct epsilon values (or has a non-positive best utility) for a
    /// regression to be meaningful.
    ///
    /// # Errors
    /// Propagates regression errors for a frontier that passed the checks.
    fn frontier_degradation_predictions(
        &self,
        frontier: &[ParetoPoint<T>],
    ) -> Result<Vec<DegradationPrediction<T>>> {
        if frontier.len() < 3 {
            return Ok(Vec::new());
        }
        let mut best_utility = f64::NEG_INFINITY;
        for point in frontier {
            best_utility = best_utility.max(to_f64(point.utility_value)?);
        }
        // `!(best_utility > 0.0)` rejects NaN as well as non-positive values; the
        // negated form is deliberate (a NaN utility must not pass this guard).
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(best_utility > 0.0) {
            return Ok(Vec::new());
        }
        let mut history: Vec<(T, T)> = Vec::with_capacity(frontier.len());
        let mut epsilons: Vec<f64> = Vec::with_capacity(frontier.len());
        for point in frontier {
            let eps = to_f64(point.privacy_cost)?;
            let utility = to_f64(point.utility_value)?;
            let relative_loss = ((best_utility - utility) / best_utility).clamp(0.0, 1.0);
            history.push((to_t(eps)?, to_t(relative_loss)?));
            epsilons.push(eps);
        }
        let mut distinct = epsilons.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if distinct.len() < 3 {
            return Ok(Vec::new());
        }
        let (eps_min, eps_max) = (distinct[0], distinct[distinct.len() - 1]);
        let query_count = 10.min(distinct.len());
        let queries: Vec<T> = even_samples(eps_min, eps_max, query_count)
            .into_iter()
            .map(to_t)
            .collect::<Result<Vec<T>>>()?;
        self.predict_utility_degradation(&queries, &history)
    }

    /// Budget recommendations derived from the frontier.
    ///
    /// The allocation is optimized against a utility model fitted to the
    /// frontier itself, so the recommendation reflects the caller's oracle
    /// rather than a built-in guess.
    ///
    /// # Errors
    /// Propagates fitting and allocation errors.
    fn budget_recommendations_for(
        &self,
        frontier: &[ParetoPoint<T>],
        best_point: &ParetoPoint<T>,
    ) -> Result<Option<BudgetRecommendations<T>>> {
        if frontier.len() < 2 {
            return Ok(None);
        }
        let mut xs = Vec::with_capacity(frontier.len());
        let mut ys = Vec::with_capacity(frontier.len());
        for point in frontier {
            xs.push(to_f64(point.privacy_cost)?);
            ys.push(to_f64(point.utility_value)?);
        }
        let mut distinct = xs.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if distinct.len() < 2 {
            return Ok(None);
        }
        let coefficients = stats::polyfit(&xs, &ys, 1)?;
        let utility_model = move |eps: f64| stats::polyeval(&coefficients, eps);

        let (eps_total, _) = self.composed_budget(&best_point.configuration)?;
        let iterations = best_point.configuration.iterations;
        let mut candidates =
            self.budget_allocation_candidates(eps_total, iterations, &utility_model)?;
        if candidates.is_empty() {
            return Ok(None);
        }
        let optimal_allocation = candidates.remove(0);

        let best_utility = to_f64(best_point.utility_value)?;
        let utility_per_epsilon = to_t(best_utility / eps_total)?;
        let marginal_utility = marginal_utility_at(&xs, &ys, to_f64(best_point.privacy_cost)?)
            .map(to_t)
            .transpose()?;
        let amplification = self
            .subsampling_amplification(&best_point.configuration)?
            .map(to_t)
            .transpose()?;

        Ok(Some(BudgetRecommendations {
            optimal_allocation,
            alternative_allocations: candidates,
            efficiency_metrics: BudgetEfficiencyMetrics {
                utility_per_epsilon,
                marginal_utility,
                subsampling_amplification_factor: amplification,
            },
        }))
    }

    /// Privacy amplification by subsampling for one mechanism invocation.
    ///
    /// For a mechanism that is `eps`-DP on its input and is applied to a
    /// Poisson subsample of rate `q`, the amplified guarantee is
    /// `eps' = ln(1 + q (e^eps - 1))` (the standard subsampling amplification
    /// bound). The reported factor is `eps / eps' >= 1`.
    ///
    /// Returns `None` when `q = 1` (no subsampling, hence no amplification to
    /// report) or when the numbers overflow into a meaningless factor.
    ///
    /// # Errors
    /// Propagates configuration validation errors.
    pub(super) fn subsampling_amplification(
        &self,
        config: &PrivacyConfiguration<T>,
    ) -> Result<Option<f64>> {
        validate_privacy_configuration(config)?;
        let q = to_f64(config.sampling_probability)?;
        if q >= 1.0 {
            return Ok(None);
        }
        let eps_per_invocation = match self.config.epsilon_semantics {
            EpsilonSemantics::Total => to_f64(config.epsilon)? / config.iterations as f64,
            EpsilonSemantics::PerIteration => to_f64(config.epsilon)?,
        };
        if !eps_per_invocation.is_finite() || eps_per_invocation <= 0.0 {
            return Ok(None);
        }
        let amplified = (1.0 + q * eps_per_invocation.exp_m1()).ln();
        if !amplified.is_finite() || amplified <= 0.0 {
            return Ok(None);
        }
        let factor = eps_per_invocation / amplified;
        if factor.is_finite() && factor >= 1.0 {
            Ok(Some(factor))
        } else {
            Ok(None)
        }
    }

    /// Assemble the metadata of an analysis run.
    ///
    /// Everything reported here is measured or derived: the configuration hash
    /// is an FNV-1a digest of the actual configuration, the version comes from
    /// the crate manifest, the parallelism from the platform, and the seed is
    /// the one that actually drove the run.
    ///
    /// # Errors
    /// Currently infallible, but returns `Result` so future measurements can
    /// report failures.
    fn build_metadata(&self, start_time: std::time::Instant) -> Result<AnalysisMetadata> {
        let elapsed = start_time.elapsed();
        let mut software_versions = HashMap::new();
        software_versions.insert(
            env!("CARGO_PKG_NAME").to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        Ok(AnalysisMetadata {
            timestamp: format!("{:?}", std::time::SystemTime::now()),
            analysis_duration: elapsed,
            analysis_version: env!("CARGO_PKG_VERSION").to_string(),
            configuration_hash: format!(
                "{:016x}",
                stats::fnv1a_64(format!("{:?}", self.config).as_bytes())
            ),
            computational_resources: ComputationalResources {
                wall_time: elapsed,
                available_parallelism: std::thread::available_parallelism()
                    .ok()
                    .map(|value| value.get()),
                peak_memory_bytes: None,
                gpu_usage: None,
            },
            reproducibility_info: ReproducibilityInfo {
                random_seed: self.resolved_seed,
                software_versions,
                hardware_info: format!(
                    "{}-{}-{}",
                    std::env::consts::ARCH,
                    std::env::consts::OS,
                    std::env::consts::FAMILY
                ),
            },
        })
    }
}

/// `count` evenly spaced values covering `[start, end]` inclusive.
pub(super) fn even_samples(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0.5 * (start + end)];
    }
    (0..count)
        .map(|i| start + (end - start) * i as f64 / (count - 1) as f64)
        .collect()
}

/// Keep at most `limit` items, evenly spread over the whole input.
///
/// Because the input is the flat mixed-radix enumeration of a parameter grid,
/// striding it evenly preserves coverage of every dimension's full span.
pub(super) fn stride_select<V>(items: Vec<V>, limit: usize) -> Vec<V> {
    let len = items.len();
    if limit == 0 || len <= limit {
        return items;
    }
    let mut keep = vec![false; len];
    if limit == 1 {
        keep[len / 2] = true;
    } else {
        for i in 0..limit {
            let idx = ((i as f64) * (len - 1) as f64 / (limit - 1) as f64).round() as usize;
            keep[idx.min(len - 1)] = true;
        }
    }
    items
        .into_iter()
        .zip(keep)
        .filter_map(|(item, selected)| if selected { Some(item) } else { None })
        .collect()
}

/// Local slope `du/d(epsilon)` at `x0`, estimated from the two neighbouring
/// samples of an unsorted `(x, y)` cloud. Returns `None` when the neighbours
/// coincide in `x`.
fn marginal_utility_at(xs: &[f64], ys: &[f64], x0: f64) -> Option<f64> {
    if xs.len() < 2 || xs.len() != ys.len() {
        return None;
    }
    let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let idx = pairs
        .iter()
        .position(|(x, _)| (*x - x0).abs() < 1e-12)
        .unwrap_or(pairs.len() / 2);
    let lo = idx.saturating_sub(1);
    let hi = (idx + 1).min(pairs.len() - 1);
    if lo == hi {
        return None;
    }
    let dx = pairs[hi].0 - pairs[lo].0;
    if dx.abs() < 1e-12 {
        return None;
    }
    Some((pairs[hi].1 - pairs[lo].1) / dx)
}
