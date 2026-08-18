//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use crate::privacy::NoiseMechanism;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::{
    ComputationalResources, MultipleComparisonCorrection, PerturbationAnalysis, PowerAnalysis,
    ReproducibilityInfo, RiskCategory, RiskEvolution,
};

/// Compliance status
///
/// [`ComplianceStatus::Unknown`] is what
/// [`PrivacyUtilityAnalyzer::assess_privacy_risk`](super::types::PrivacyUtilityAnalyzer::assess_privacy_risk)
/// reports: compliance with a regulation cannot be decided from privacy
/// parameters alone. The remaining variants exist for callers that perform
/// their own compliance review and want to record its outcome alongside the
/// computed bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplianceStatus {
    /// Fully compliant, as determined by an external review
    Compliant,
    /// Partially compliant, as determined by an external review
    PartiallyCompliant,
    /// Non-compliant, as determined by an external review
    NonCompliant,
    /// Not assessed
    Unknown,
}
/// Statistical test results
#[derive(Debug, Clone)]
pub struct StatisticalTestResults<T: Float + Debug + Send + Sync + 'static> {
    /// Hypothesis test results
    pub hypothesis_tests: Vec<HypothesisTestResult<T>>,
    /// Significance levels
    pub significance_levels: Vec<T>,
    /// Effect sizes
    pub effect_sizes: Vec<T>,
    /// Power analysis
    pub power_analysis: PowerAnalysis<T>,
    /// Multiple comparison corrections
    pub multiple_comparison_corrections: Vec<MultipleComparisonCorrection<T>>,
}
/// Local sensitivity of the utility oracle to one parameter
#[derive(Debug, Clone)]
pub struct LocalSensitivity<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter name
    pub parameter: String,
    /// First derivative of utility with respect to the parameter
    pub gradient: T,
    /// Second derivative, available only when the parameter can be perturbed
    /// in both directions without leaving its domain
    pub hessian: Option<T>,
    /// Finite-difference step used for this parameter
    pub step_size: T,
    /// Confidence interval of the gradient, from replicate estimates; `None`
    /// when only one replicate was requested
    pub confidence_interval: Option<(T, T)>,
}
/// Allocation of a privacy budget over the iterations of a training run
#[derive(Debug, Clone)]
pub struct BudgetAllocation<T: Float + Debug + Send + Sync + 'static> {
    /// Total epsilon that was allocated
    pub total_epsilon: f64,
    /// Per-iteration epsilon allocation; sums to `total_epsilon`
    pub per_iteration_allocation: Vec<T>,
    /// Allocation strategy that produced this split
    pub allocation_strategy: AllocationStrategy,
    /// Mean utility this allocation is predicted to reach under the utility
    /// model supplied by the caller
    pub predicted_mean_utility: T,
    /// `max/mean - 1` of the per-iteration allocation: `0` for a uniform
    /// split, larger for more skewed ones
    pub allocation_imbalance: T,
}
/// How the `epsilon`/`delta` of a [`PrivacyConfiguration`] are to be read.
///
/// The module needs one unambiguous interpretation because the same numbers
/// drive the Pareto axis, the privacy cost and the risk bounds. Both readings
/// are supported explicitly rather than being left implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsilonSemantics {
    /// `epsilon`/`delta` already describe the whole training run (the usual
    /// reading of a "target epsilon"): no composition is applied.
    Total,
    /// `epsilon`/`delta` describe a single mechanism invocation and are
    /// composed over `PrivacyConfiguration::iterations` using basic
    /// (sequential) composition: `eps_total = iterations * eps` and
    /// `delta_total = 1 - (1 - delta)^iterations`.
    PerIteration,
}
/// Privacy configuration parameters
///
/// The meaning of `epsilon`/`delta` is fixed by
/// [`AnalysisConfig::epsilon_semantics`]: either the budget of the whole run
/// ([`EpsilonSemantics::Total`], the default) or the budget of one mechanism
/// invocation that is composed over `iterations`
/// ([`EpsilonSemantics::PerIteration`]).
#[derive(Debug, Clone)]
pub struct PrivacyConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Epsilon value
    pub epsilon: T,
    /// Delta value
    pub delta: T,
    /// Noise multiplier
    pub noise_multiplier: T,
    /// Clipping threshold
    pub clipping_threshold: T,
    /// Sampling probability
    pub sampling_probability: T,
    /// Number of iterations
    pub iterations: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Noise mechanism
    pub noise_mechanism: NoiseMechanism,
}
/// Privacy risk assessment
#[derive(Debug, Clone)]
pub struct PrivacyRiskAssessment<T: Float + Debug + Send + Sync + 'static> {
    /// Overall risk score
    pub overall_risk_score: T,
    /// Risk categories
    pub risk_categories: HashMap<RiskCategory, T>,
    /// Risk mitigation recommendations
    pub mitigation_recommendations: Vec<String>,
    /// Compliance status
    pub compliance_status: ComplianceStatus,
    /// Risk evolution over time
    pub risk_evolution: Vec<RiskEvolution<T>>,
}
/// Configuration for privacy-utility analysis
///
/// Every field is read by the analyzer; there are no decorative options.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Privacy parameters to analyze
    pub privacy_parameters: PrivacyParameterSpace,
    /// Number of replicate finite-difference estimates used to derive the
    /// confidence intervals of the sensitivity analysis. Each replicate costs
    /// one model evaluation per parameter, so this directly controls the cost
    /// (and the resolution) of `SensitivityResults::confidence_intervals`.
    /// A value of `1` disables interval estimation (the intervals are then
    /// reported as "not assessed", i.e. `None`).
    pub monte_carlo_samples: usize,
    /// Enable sensitivity analysis
    pub enable_sensitivity_analysis: bool,
    /// Enable robustness evaluation
    pub enable_robustness_evaluation: bool,
    /// Upper bound on the number of privacy configurations evaluated when
    /// building the Pareto frontier. The parameter grid is strided so that the
    /// full span of every parameter range is covered within this budget.
    pub pareto_resolution: usize,
    /// Confidence level used for every interval and critical value produced by
    /// the analysis (e.g. `0.95` gives two-sided `z = 1.96`).
    pub confidence_level: f64,
    /// Target statistical power used by the power analysis (e.g. `0.8`).
    pub target_power: f64,
    /// Relative utility drop (in `(0, 1]`) above which the robustness
    /// evaluation reports a failure mode. This is a utility threshold and is
    /// deliberately distinct from `confidence_level`.
    pub utility_degradation_threshold: f64,
    /// Interpretation of the `epsilon`/`delta` of each candidate configuration.
    pub epsilon_semantics: EpsilonSemantics,
    /// Seed for every randomized step of the analysis (random parameter
    /// sampling, data perturbation, adversarial search directions). When
    /// `None`, the analyzer draws one seed from system entropy at construction
    /// time and reports it in [`AnalysisMetadata`], so an analysis can always
    /// be reproduced by feeding that seed back in.
    pub random_seed: Option<u64>,
}
impl AnalysisConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] if any field is outside its
    /// admissible range.
    pub fn validate(&self) -> Result<()> {
        if self.pareto_resolution == 0 {
            return Err(OptimError::InvalidParameter(
                "pareto_resolution must be > 0".to_string(),
            ));
        }
        if self.monte_carlo_samples == 0 {
            return Err(OptimError::InvalidParameter(
                "monte_carlo_samples must be > 0".to_string(),
            ));
        }
        if !self.confidence_level.is_finite()
            || self.confidence_level <= 0.0
            || self.confidence_level >= 1.0
        {
            return Err(OptimError::InvalidParameter(format!(
                "confidence_level must be in (0, 1), got {}",
                self.confidence_level
            )));
        }
        if !self.target_power.is_finite() || self.target_power <= 0.0 || self.target_power >= 1.0 {
            return Err(OptimError::InvalidParameter(format!(
                "target_power must be in (0, 1), got {}",
                self.target_power
            )));
        }
        if !self.utility_degradation_threshold.is_finite()
            || self.utility_degradation_threshold <= 0.0
            || self.utility_degradation_threshold > 1.0
        {
            return Err(OptimError::InvalidParameter(format!(
                "utility_degradation_threshold must be in (0, 1], got {}",
                self.utility_degradation_threshold
            )));
        }
        self.privacy_parameters.validate()
    }
}
/// Analysis metadata
#[derive(Debug, Clone)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    /// Analysis duration
    pub analysis_duration: std::time::Duration,
    /// Analysis version
    pub analysis_version: String,
    /// Configuration used
    pub configuration_hash: String,
    /// Computational resources used
    pub computational_resources: ComputationalResources,
    /// Reproducibility information
    pub reproducibility_info: ReproducibilityInfo,
}
/// Hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult<T: Float + Debug + Send + Sync + 'static> {
    /// Test name
    pub test_name: String,
    /// Test statistic
    pub test_statistic: T,
    /// P-value
    pub p_value: T,
    /// Significance level
    pub significance_level: T,
    /// Reject null hypothesis
    pub reject_null: bool,
    /// Effect size
    pub effect_size: T,
}
/// A perturbation family whose worst observed utility drop exceeded
/// `AnalysisConfig::utility_degradation_threshold`
#[derive(Debug, Clone)]
pub struct FailureMode<T: Float + Debug + Send + Sync + 'static> {
    /// Which family failed
    pub failure_type: FailureType,
    /// Observed utility drop, relative to the unperturbed utility
    pub observed_relative_drop: T,
    /// Smallest probed perturbation magnitude at which the drop was observed
    pub perturbation_level: T,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}
/// Stability of the utility surface around a configuration
///
/// Every field is measured from the perturbation sweep; nothing here describes
/// the convergence of a training run, which the analyzer never observes.
#[derive(Debug, Clone)]
pub struct StabilityAnalysis<T: Float + Debug + Send + Sync + 'static> {
    /// Least-squares slope of `ln(utility)` against perturbation magnitude
    /// over the adversarial sweep; `None` when some utility was non-positive
    pub log_utility_sensitivity: Option<T>,
    /// Standard deviation over mean of every utility observed during the
    /// sweep; `None` when the mean is numerically zero
    pub utility_coefficient_of_variation: Option<T>,
    /// Perturbation analysis
    pub perturbation_analysis: PerturbationAnalysis<T>,
}
/// Budget allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    /// Uniform allocation
    Uniform,
    /// Decreasing allocation
    Decreasing,
    /// Increasing allocation
    Increasing,
    /// Adaptive allocation
    Adaptive,
    /// Importance-based allocation
    ImportanceBased,
    /// Risk-based allocation
    RiskBased,
}
/// Efficiency of a budget recommendation, measured on the Pareto frontier
#[derive(Debug, Clone)]
pub struct BudgetEfficiencyMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Utility of the recommended configuration divided by its composed
    /// epsilon
    pub utility_per_epsilon: T,
    /// Local slope `d(utility)/d(epsilon)` of the frontier at the recommended
    /// configuration; `None` when the frontier has no distinct neighbours
    pub marginal_utility: Option<T>,
    /// Privacy amplification by subsampling, `eps / ln(1 + q (e^eps - 1))`;
    /// `None` when the configuration draws the full dataset every step
    pub subsampling_amplification_factor: Option<T>,
}
/// Parameter range specification
///
/// Construct with [`ParameterRange::new`], which rejects ranges that cannot be
/// sampled (inverted, non-finite, empty, or non-positive under logarithmic
/// sampling). Ranges built by hand through the public fields are validated
/// again by [`ParameterRange::validate`] before they are sampled.
#[derive(Debug, Clone)]
pub struct ParameterRange {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of samples
    pub num_samples: usize,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
}
impl ParameterRange {
    /// Create a validated parameter range.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] when `min`/`max` are not
    /// finite, when `max <= min`, when `num_samples == 0`, or when
    /// `min <= 0` for [`SamplingStrategy::Logarithmic`] (whose sample
    /// positions are computed in log space).
    pub fn new(
        min: f64,
        max: f64,
        num_samples: usize,
        sampling_strategy: SamplingStrategy,
    ) -> Result<Self> {
        let range = Self {
            min,
            max,
            num_samples,
            sampling_strategy,
        };
        range.validate()?;
        Ok(range)
    }

    /// Validate the range.
    ///
    /// # Errors
    /// See [`ParameterRange::new`].
    pub fn validate(&self) -> Result<()> {
        if !self.min.is_finite() || !self.max.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "parameter range bounds must be finite, got [{}, {}]",
                self.min, self.max
            )));
        }
        if self.max <= self.min {
            return Err(OptimError::InvalidParameter(format!(
                "parameter range requires max > min, got [{}, {}]",
                self.min, self.max
            )));
        }
        if self.num_samples == 0 {
            return Err(OptimError::InvalidParameter(
                "parameter range requires num_samples > 0".to_string(),
            ));
        }
        if matches!(self.sampling_strategy, SamplingStrategy::Logarithmic) && self.min <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "logarithmic parameter range requires min > 0, got {}",
                self.min
            )));
        }
        Ok(())
    }

    /// Representative interior value of the range: the arithmetic midpoint for
    /// linear-style sampling and the geometric midpoint `sqrt(min * max)` for
    /// logarithmic sampling.
    ///
    /// # Errors
    /// See [`ParameterRange::new`].
    pub fn midpoint(&self) -> Result<f64> {
        self.validate()?;
        Ok(match self.sampling_strategy {
            SamplingStrategy::Logarithmic => (self.min * self.max).sqrt(),
            _ => 0.5 * (self.min + self.max),
        })
    }

    /// Build a range whose bounds are known to be valid at compile time.
    ///
    /// Only used for the library-provided defaults, where the literals are
    /// statically known to satisfy [`ParameterRange::validate`]; every value
    /// is validated again before it is sampled.
    pub(super) const fn new_unchecked(
        min: f64,
        max: f64,
        num_samples: usize,
        sampling_strategy: SamplingStrategy,
    ) -> Self {
        Self {
            min,
            max,
            num_samples,
            sampling_strategy,
        }
    }
}
/// Perturbation families that can breach the utility degradation threshold
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureType {
    /// The worst perturbation of any family collapsed the utility
    UtilityCollapse,
    /// Perturbing the input data collapsed the utility
    DistributionalFailure,
    /// The worst case found in the perturbation ball collapsed the utility
    AdversarialFailure,
    /// Perturbing a privacy parameter collapsed the utility
    ParameterSensitivityFailure,
}
/// Robustness evaluation results
///
/// All three robustness scores are `1 / (1 + relative utility drop)`, so `1`
/// means "no drop observed" and smaller values mean larger drops.
#[derive(Debug, Clone)]
pub struct RobustnessResults<T: Float + Debug + Send + Sync + 'static> {
    /// Score over every probed perturbation family
    pub robustness_score: T,
    /// Largest relative utility drop observed over every family
    pub worst_case_degradation: T,
    /// Score restricted to the worst case found in the perturbation ball
    pub adversarial_robustness: T,
    /// Score restricted to perturbations of the input data
    pub distributional_robustness: T,
    /// Stability analysis
    pub stability_analysis: StabilityAnalysis<T>,
    /// Families that breached the configured degradation threshold
    pub failure_modes: Vec<FailureMode<T>>,
}
/// Budget allocation recommendations
#[derive(Debug, Clone)]
pub struct BudgetRecommendations<T: Float + Debug + Send + Sync + 'static> {
    /// Allocation with the highest predicted utility
    pub optimal_allocation: BudgetAllocation<T>,
    /// The remaining allocation shapes, best first
    pub alternative_allocations: Vec<BudgetAllocation<T>>,
    /// Budget efficiency metrics
    pub efficiency_metrics: BudgetEfficiencyMetrics<T>,
}
/// Sampling strategies for parameter space exploration
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Linear sampling
    Linear,
    /// Logarithmic sampling
    Logarithmic,
    /// Random sampling
    Random,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Sobol sequence sampling
    Sobol,
    /// Adaptive sampling based on gradients
    Adaptive,
}
/// Utility degradation prediction
#[derive(Debug, Clone)]
pub struct DegradationPrediction<T: Float + Debug + Send + Sync + 'static> {
    /// Privacy parameter
    pub privacy_parameter: T,
    /// Predicted utility loss
    pub predicted_utility_loss: T,
    /// Confidence interval
    pub confidence_interval: (T, T),
    /// Prediction model
    pub prediction_model: PredictionModel,
    /// Model accuracy
    pub model_accuracy: T,
}
/// Privacy parameter space definition
#[derive(Debug, Clone)]
pub struct PrivacyParameterSpace {
    /// Epsilon values to analyze
    pub epsilon_range: ParameterRange,
    /// Delta values to analyze
    pub delta_range: ParameterRange,
    /// Noise multiplier values
    pub noise_multiplier_range: ParameterRange,
    /// Clipping threshold values
    pub clipping_threshold_range: ParameterRange,
    /// Sampling probability values
    pub sampling_probability_range: ParameterRange,
    /// Number of iterations to analyze
    pub iterations_range: ParameterRange,
    /// Batch size values
    pub batch_size_range: ParameterRange,
    /// Learning rate values
    pub learning_rate_range: ParameterRange,
}
impl PrivacyParameterSpace {
    /// Validate every range in the space.
    ///
    /// # Errors
    /// Returns [`OptimError::InvalidParameter`] for the first invalid range.
    pub fn validate(&self) -> Result<()> {
        let ranges: [(&str, &ParameterRange); 8] = [
            ("epsilon_range", &self.epsilon_range),
            ("delta_range", &self.delta_range),
            ("noise_multiplier_range", &self.noise_multiplier_range),
            ("clipping_threshold_range", &self.clipping_threshold_range),
            (
                "sampling_probability_range",
                &self.sampling_probability_range,
            ),
            ("iterations_range", &self.iterations_range),
            ("batch_size_range", &self.batch_size_range),
            ("learning_rate_range", &self.learning_rate_range),
        ];
        for (name, range) in ranges {
            range
                .validate()
                .map_err(|err| OptimError::InvalidParameter(format!("{name}: {err}")))?;
        }
        Ok(())
    }
}
/// Point on Pareto frontier
#[derive(Debug, Clone)]
pub struct ParetoPoint<T: Float + Debug + Send + Sync + 'static> {
    /// Composed privacy guarantee (total epsilon)
    pub privacy_guarantee: T,
    /// Utility reported by the oracle for this configuration
    pub utility_value: T,
    /// Configuration parameters
    pub configuration: PrivacyConfiguration<T>,
    /// Confidence interval of the utility. Always `None` today: the oracle is
    /// queried once per configuration, and a single evaluation carries no
    /// replicate variance from which an interval could be derived.
    pub confidence_interval: Option<(T, T)>,
    /// Composed epsilon, the scalar the frontier is sorted by
    pub privacy_cost: T,
    /// Whether this point is dominated by others (always `false` for points
    /// that appear on the frontier)
    pub dominated: bool,
    /// Euclidean distance to the ideal (lowest cost, highest utility) corner
    pub distance_to_ideal: T,
}
/// Regression models used by the degradation prediction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionModel {
    /// Least-squares fit of degree 1
    LinearRegression,
    /// Least-squares fit of degree 2 or 3
    PolynomialRegression,
}
/// Risk trends
#[derive(Debug, Clone)]
pub enum RiskTrend {
    /// Risk increasing
    Increasing,
    /// Risk decreasing
    Decreasing,
    /// Risk stable
    Stable,
    /// Risk oscillating
    Oscillating,
}
/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Maximize utility for given privacy budget
    MaximizeUtility,
    /// Minimize privacy loss for given utility threshold
    MinimizePrivacyLoss,
    /// Balance privacy and utility equally
    BalancePrivacyUtility,
    /// Maximize robustness
    MaximizeRobustness,
    /// Minimize worst-case scenario
    MinimizeWorstCase,
    /// Custom objective function
    Custom(String),
}
