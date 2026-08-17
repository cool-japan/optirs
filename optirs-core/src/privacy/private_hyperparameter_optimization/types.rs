//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{OptimError, Result};
use crate::privacy::moment_accountant::MomentsAccountant;
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

use super::functions::{ObjectiveFn, RuleFn, TestFn};
use super::results::SelectionReport;

/// Generator used by the private HPO search strategies and noise mechanisms.
pub type HpoRng = scirs2_core::random::Random<StdRng>;

/// Create a search generator seeded from OS entropy.
///
/// Hyperparameter proposals must not be predictable from a constant seed:
/// a fixed seed makes the whole search trajectory public knowledge, which
/// leaks which configurations were evaluated against the private dataset.
/// The `*_with_seed` constructors exist for reproducible tests only.
pub fn os_seeded_hpo_rng() -> HpoRng {
    scirs2_core::random::SeedableRng::from_rng(&mut scirs2_core::random::thread_rng())
}

/// Seconds since the Unix epoch, or an error if the clock is before it.
///
/// Replaces a `.expect("unwrap failed")` in the evaluation loop: a clock skewed
/// behind the epoch is a configuration problem, not a reason to abort the
/// process.
pub fn unix_timestamp() -> Result<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .map_err(|err| {
            OptimError::InvalidState(format!(
                "the system clock is set before the Unix epoch, so evaluations cannot be \
                 timestamped: {err}"
            ))
        })
}

/// Result validator
pub struct ResultValidator<T: Float + Debug + Send + Sync + 'static> {
    /// Validation rules
    validation_rules: Vec<ValidationRule<T>>,
    /// Statistical tests
    statistical_tests: Vec<StatisticalTest<T>>,
    /// Anomaly detection
    anomaly_detector: AnomalyDetector<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ResultValidator<T> {
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
            statistical_tests: Vec::new(),
            anomaly_detector: AnomalyDetector::new(),
        }
    }
}
/// Sampling strategies for sensitivity estimation
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Sobol sequence sampling
    Sobol,
    /// Halton sequence sampling
    Halton,
    /// Importance sampling
    ImportanceSampling,
}
/// Hyperparameter optimization evaluation
#[derive(Debug, Clone)]
pub struct HPOEvaluation<T: Float + Debug + Send + Sync + 'static> {
    /// Evaluation identifier
    pub id: String,
    /// Parameter configuration
    pub configuration: ParameterConfiguration<T>,
    /// Evaluation result
    pub result: HPOResult<T>,
    /// Privacy budget consumed
    pub privacy_cost: PrivacyBudget,
    /// Evaluation timestamp
    pub timestamp: u64,
    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}
/// Smooth sensitivity parameters
#[derive(Debug, Clone)]
pub struct SmoothSensitivityParams<T: Float + Debug + Send + Sync + 'static> {
    /// Beta parameter for smooth sensitivity
    pub beta: T,
    /// Maximum local sensitivity
    pub max_local_sensitivity: T,
    /// Smoothness parameter
    pub smoothness: T,
}
/// Search algorithms for hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum SearchAlgorithm {
    /// Random search with differential privacy
    RandomSearch,
    /// Grid search with differential privacy
    GridSearch,
    /// Bayesian optimization with differential privacy
    BayesianOptimization,
    /// Genetic algorithm with differential privacy
    GeneticAlgorithm,
    /// Particle swarm optimization with differential privacy
    ParticleSwarm,
    /// Simulated annealing with differential privacy
    SimulatedAnnealing,
    /// Tree-structured Parzen estimator with differential privacy
    TPE,
}
/// Search strategy for hyperparameter optimization
pub struct SearchStrategy<T: Float + Debug + Send + Sync + 'static> {
    /// Search algorithm
    algorithm: SearchAlgorithm,
    /// Algorithm-specific parameters
    algorithm_params: HashMap<String, f64>,
    /// Exploration-exploitation balance
    exploration_factor: f64,
    /// Convergence criteria
    convergence_criteria: ConvergenceCriteria<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> SearchStrategy<T> {
    pub fn new() -> Self {
        Self {
            algorithm: SearchAlgorithm::RandomSearch,
            algorithm_params: HashMap::new(),
            exploration_factor: 0.1,
            convergence_criteria: ConvergenceCriteria {
                max_iterations: 100,
                tolerance: T::from(1e-6).unwrap_or_else(|| T::zero()),
                patience: 10,
                min_change: T::from(1e-4).unwrap_or_else(|| T::zero()),
            },
        }
    }
}
/// Fold assignment strategies
#[derive(Debug, Clone, Copy)]
pub enum FoldStrategy {
    /// Random assignment
    Random,
    /// Stratified assignment
    Stratified,
    /// Time-based assignment (for time series)
    TimeBased,
    /// Group-based assignment
    GroupBased,
}
/// Types of kernel functions
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    /// Radial basis function kernel
    RBF,
    /// Matern kernel
    Matern,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
}
/// Budget allocation strategies for hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum BudgetAllocationStrategy {
    /// Equal allocation across all evaluations
    Equal,
    /// Adaptive allocation based on promising regions
    Adaptive,
    /// Bandit-based allocation
    Bandit,
    /// Hierarchical allocation (coarse to fine)
    Hierarchical,
    /// Budget allocation based on uncertainty
    Uncertainty,
}
/// Sensitivity bounds for hyperparameters
#[derive(Debug, Clone)]
pub struct SensitivityBounds<T: Float + Debug + Send + Sync + 'static> {
    /// Global sensitivity for each hyperparameter
    pub global_sensitivity: HashMap<String, T>,
    /// Local sensitivity bounds
    pub local_sensitivity: HashMap<String, (T, T)>,
    /// Smooth sensitivity parameters
    pub smooth_sensitivity: HashMap<String, SmoothSensitivityParams<T>>,
}
/// Summary statistics with privacy
#[derive(Debug, Clone)]
pub struct SummaryStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Noisy mean of objective values
    pub noisy_mean: T,
    /// Noisy standard deviation
    pub noisy_std: T,
    /// Noisy median
    pub noisy_median: T,
    /// Noisy quantiles
    pub noisy_quantiles: Vec<(f64, T)>,
}
/// Individual parameter definition
#[derive(Debug, Clone)]
pub struct ParameterDefinition<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType<T>,
    /// Parameter bounds
    pub bounds: ParameterBounds<T>,
    /// Prior distribution (for Bayesian optimization)
    pub prior: Option<ParameterPrior<T>>,
    /// Transformation function
    pub transformation: Option<ParameterTransformation>,
}
/// Statistical test conclusions
#[derive(Debug, Clone, Copy)]
pub enum TestConclusion {
    /// Reject null hypothesis
    Reject,
    /// Fail to reject null hypothesis
    FailToReject,
    /// Insufficient evidence
    InsufficientEvidence,
}
/// Selection parameters
///
/// # 0.3.2 change
///
/// `delta` was added so [`HyperparameterNoiseMechanism::Gaussian`] selection can
/// be calibrated instead of refused. Struct-literal construction needs the extra
/// field; [`SelectionParameters::pure_epsilon`] builds the pure-epsilon shape.
#[derive(Debug, Clone)]
pub struct SelectionParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Temperature parameter, dividing the utility before it is exponentiated
    pub temperature: T,
    /// Sensitivity bound for utility (`Delta_u`)
    pub utility_sensitivity: T,
    /// Epsilon charged per selection
    pub epsilon: f64,
    /// Delta charged per selection; required by the Gaussian mechanism only
    pub delta: Option<f64>,
    /// Public utility threshold below which candidates are discarded before the
    /// mechanism runs.
    ///
    /// Only sound when the threshold is public knowledge: filtering on a
    /// data-dependent threshold would leak through the candidate set.
    pub threshold: Option<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> SelectionParameters<T> {
    /// Parameters for a pure-epsilon mechanism.
    pub fn pure_epsilon(epsilon: f64, utility_sensitivity: T) -> Self {
        Self {
            temperature: T::one(),
            utility_sensitivity,
            epsilon,
            delta: None,
            threshold: None,
        }
    }
}
/// Private aggregation of cross-validation results
pub struct PrivateFoldAggregation<T: Float + Debug + Send + Sync + 'static> {
    /// Aggregation method
    aggregation_method: AggregationMethod,
    /// Noise parameters for aggregation
    noise_params: NoiseParameters<T>,
    /// Confidence interval estimation
    confidence_estimation: ConfidenceEstimation<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivateFoldAggregation<T> {
    pub fn new() -> Self {
        Self {
            aggregation_method: AggregationMethod::NoisyMean,
            noise_params: NoiseParameters {
                scale: T::one(),
                sensitivity: T::one(),
                epsilon: 1.0,
                delta: Some(1e-5),
            },
            confidence_estimation: ConfidenceEstimation::new(),
        }
    }
}
/// Acquisition function for Bayesian optimization
pub struct AcquisitionFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function type
    function_type: AcquisitionFunctionType,
    /// Function parameters
    parameters: Vec<T>,
    /// Exploration-exploitation balance
    exploration_weight: T,
}
impl<T: Float + Debug + Send + Sync + 'static> AcquisitionFunction<T> {
    pub fn new() -> Self {
        Self {
            function_type: AcquisitionFunctionType::ExpectedImprovement,
            parameters: Vec::new(),
            exploration_weight: T::from(0.1).unwrap_or_else(|| T::zero()),
        }
    }
}
/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    /// Patience (number of evaluations without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Maximum number of evaluations
    pub max_evaluations: usize,
}
/// Aggregation methods for cross-validation
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Mean aggregation with noise
    NoisyMean,
    /// Median aggregation
    Median,
    /// Trimmed mean
    TrimmedMean,
    /// Weighted average
    WeightedAverage,
    /// Robust aggregation
    Robust,
}
/// Bootstrap parameters for confidence estimation
#[derive(Debug, Clone)]
pub struct BootstrapParams {
    /// Number of bootstrap samples
    pub num_samples: usize,
    /// Bootstrap type
    pub bootstrap_type: BootstrapType,
    /// Bias correction
    pub bias_correction: bool,
}
/// Gaussian process model for Bayesian optimization
pub struct GaussianProcessModel<T: Float + Debug + Send + Sync + 'static> {
    /// Training inputs
    training_inputs: Vec<Vec<T>>,
    /// Training outputs
    training_outputs: Vec<T>,
    /// Kernel function
    kernel: KernelFunction<T>,
    /// Hyperparameters
    hyperparameters: Vec<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> GaussianProcessModel<T> {
    /// An empty (unfitted) model with an RBF kernel.
    pub fn new() -> Self {
        Self {
            training_inputs: Vec::new(),
            training_outputs: Vec::new(),
            kernel: KernelFunction::new(),
            hyperparameters: Vec::new(),
        }
    }

    /// Number of recorded training points.
    pub fn observation_count(&self) -> usize {
        self.training_inputs.len()
    }

    /// The kernel.
    pub fn kernel(&self) -> &KernelFunction<T> {
        &self.kernel
    }

    /// The fitted hyperparameters: `[length_scale, signal_variance,
    /// noise_variance]` once [`GaussianProcessModel::fit`] has run.
    pub fn hyperparameters(&self) -> &[T] {
        &self.hyperparameters
    }

    /// Record training data and fit the exact posterior.
    ///
    /// Delegates the linear algebra to
    /// [`super::gaussian_process::GaussianProcessFit`], which factorises
    /// `K + sigma_n^2 I` by Cholesky. Returns the fitted posterior so a caller
    /// can predict with it.
    pub fn fit(
        &mut self,
        inputs: &[Vec<T>],
        outputs: &[T],
        noise_variance: f64,
    ) -> Result<super::gaussian_process::GaussianProcessFit> {
        if inputs.len() != outputs.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "{} inputs and {} outputs were supplied",
                inputs.len(),
                outputs.len()
            )));
        }
        let encoded: Vec<Vec<f64>> = inputs
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| {
                        value.to_f64().ok_or_else(|| {
                            OptimError::InvalidParameter(
                                "a training input cannot be represented as f64".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<f64>>>()
            })
            .collect::<Result<Vec<Vec<f64>>>>()?;
        let targets: Vec<f64> = outputs
            .iter()
            .map(|value| {
                value.to_f64().ok_or_else(|| {
                    OptimError::InvalidParameter(
                        "a training output cannot be represented as f64".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        let fit = super::gaussian_process::GaussianProcessFit::fit_with_median_heuristic(
            &encoded,
            &targets,
            noise_variance,
        )?;
        self.training_inputs = inputs.to_vec();
        self.training_outputs = outputs.to_vec();
        self.hyperparameters = [
            fit.length_scale(),
            fit.signal_variance(),
            fit.noise_variance(),
        ]
        .iter()
        .filter_map(|value| T::from(*value))
        .collect();
        Ok(fit)
    }
}
/// Noise mechanisms for hyperparameter selection
#[derive(Debug, Clone, Copy)]
pub enum HyperparameterNoiseMechanism {
    /// Exponential mechanism for discrete selection
    Exponential,
    /// Gaussian mechanism for continuous parameters
    Gaussian,
    /// Laplace mechanism
    Laplace,
    /// Report noisy max for selection
    NoisyMax,
    /// Sparse vector technique
    SparseVector,
}
/// Parameter transformations
#[derive(Debug, Clone, Copy)]
pub enum ParameterTransformation {
    /// No transformation
    Identity,
    /// Logarithmic transformation
    Log,
    /// Exponential transformation
    Exp,
    /// Square root transformation
    Sqrt,
    /// Square transformation
    Square,
}
/// Types of bootstrap methods
#[derive(Debug, Clone, Copy)]
pub enum BootstrapType {
    /// Standard bootstrap
    Standard,
    /// Bias-corrected and accelerated (BCa)
    BCa,
    /// Parametric bootstrap
    Parametric,
    /// Block bootstrap (for time series)
    Block,
}
/// Utility function for hyperparameter selection
///
/// [`UtilityFunction::evaluate`] (in [`super::selection`]) maps a raw objective
/// value to the utility the exponential mechanism scores.
pub struct UtilityFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function type
    function_type: UtilityFunctionType,
    /// Function parameters; `parameters[0]` scales the mapping
    parameters: Vec<T>,
    /// Multi-objective weights
    multi_objective_weights: Option<Vec<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static> UtilityFunction<T> {
    /// A linear, unit-scaled utility.
    pub fn new() -> Self {
        Self {
            function_type: UtilityFunctionType::Linear,
            parameters: vec![T::one()],
            multi_objective_weights: None,
        }
    }

    /// The configured function family.
    pub fn function_type(&self) -> UtilityFunctionType {
        self.function_type
    }

    /// Replace the function family.
    pub fn set_function_type(&mut self, function_type: UtilityFunctionType) {
        self.function_type = function_type;
    }

    /// The function parameters.
    pub fn parameters(&self) -> &[T] {
        &self.parameters
    }

    /// Replace the function parameters.
    pub fn set_parameters(&mut self, parameters: Vec<T>) {
        self.parameters = parameters;
    }

    /// The multi-objective weights, if configured.
    pub fn multi_objective_weights(&self) -> Option<&[T]> {
        self.multi_objective_weights.as_deref()
    }

    /// Replace the multi-objective weights.
    pub fn set_multi_objective_weights(&mut self, weights: Option<Vec<T>>) {
        self.multi_objective_weights = weights;
    }
}
/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResults<T: Float + Debug + Send + Sync + 'static> {
    /// Selected model configuration
    pub selectedconfig: ParameterConfiguration<T>,
    /// Selection confidence
    pub selection_confidence: f64,
    /// Alternative configurations
    pub alternatives: Vec<ParameterConfiguration<T>>,
}
/// Convergence criteria for search
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for objective improvement
    pub tolerance: T,
    /// Patience for early stopping
    pub patience: usize,
    /// Minimum change in best value
    pub min_change: T,
}
/// Confidence estimation methods
#[derive(Debug, Clone, Copy)]
pub enum ConfidenceEstimationMethod {
    /// Normal approximation
    Normal,
    /// Bootstrap confidence intervals
    Bootstrap,
    /// Jackknife estimation
    Jackknife,
    /// Bayesian credible intervals
    Bayesian,
}
/// Noise parameters
#[derive(Debug, Clone)]
pub struct NoiseParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Noise scale
    pub scale: T,
    /// Sensitivity bound
    pub sensitivity: T,
    /// Privacy parameters
    pub epsilon: f64,
    /// Delta parameter (for Gaussian mechanism)
    pub delta: Option<f64>,
}
/// Parameter bounds
#[derive(Debug, Clone)]
pub struct ParameterBounds<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum value
    pub min: Option<T>,
    /// Maximum value
    pub max: Option<T>,
    /// Step size (for discrete parameters)
    pub step: Option<T>,
    /// Valid values (for categorical parameters)
    pub valid_values: Option<Vec<String>>,
}
/// Configuration for private hyperparameter optimization
#[derive(Debug, Clone)]
pub struct PrivateHPOConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Base differential privacy configuration
    pub base_privacyconfig: DifferentialPrivacyConfig,
    /// Privacy budget allocation strategy
    pub budget_allocation: BudgetAllocationStrategy,
    /// Hyperparameter search algorithm
    pub search_algorithm: SearchAlgorithm,
    /// Number of hyperparameter configurations to evaluate
    pub num_evaluations: usize,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Early stopping criteria
    pub early_stopping: EarlyStoppingConfig,
    /// Noise mechanism for hyperparameter selection
    pub noise_mechanism: HyperparameterNoiseMechanism,
    /// Sensitivity bounds for hyperparameters
    pub sensitivity_bounds: SensitivityBounds<T>,
    /// Enable private model selection
    pub private_model_selection: bool,
    /// Validation strategy
    pub validation_strategy: ValidationStrategy,
}
/// Objective sensitivity analyzer
pub struct ObjectiveSensitivityAnalyzer<T: Float + Debug + Send + Sync + 'static> {
    /// Sensitivity estimation method
    estimation_method: SensitivityEstimationMethod,
    /// Sensitivity cache
    sensitivity_cache: HashMap<String, T>,
    /// Sample-based sensitivity estimator
    sample_estimator: SampleBasedSensitivityEstimator<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ObjectiveSensitivityAnalyzer<T> {
    pub fn new() -> Self {
        Self {
            estimation_method: SensitivityEstimationMethod::Global,
            sensitivity_cache: HashMap::new(),
            sample_estimator: SampleBasedSensitivityEstimator::new(),
        }
    }
}
/// Parameter value types
#[derive(Debug, Clone)]
pub enum ParameterValue<T: Float + Debug + Send + Sync + 'static> {
    /// Continuous value
    Continuous(T),
    /// Integer value
    Integer(i64),
    /// Categorical value
    Categorical(String),
    /// Boolean value
    Boolean(bool),
    /// Ordinal value
    Ordinal(usize),
}
/// Statistical test for result validation
pub struct StatisticalTest<T: Float + Debug + Send + Sync + 'static> {
    /// Test name
    pub name: String,
    /// Test function
    pub test_fn: TestFn<T>,
    /// Significance level
    pub alpha: f64,
}
/// Types of parameter constraints
#[derive(Debug, Clone)]
pub enum ConstraintType<T: Float + Debug + Send + Sync + 'static> {
    /// Linear constraint: a^T x <= b
    Linear(Vec<T>, T),
    /// Quadratic constraint: x^T A x + b^T x <= c
    Quadratic(Array2<T>, Array1<T>, T),
    /// Custom constraint function
    Custom(String),
}
/// Prior distributions for parameters
#[derive(Debug, Clone)]
pub enum ParameterPrior<T: Float + Debug + Send + Sync + 'static> {
    /// Uniform prior
    Uniform(T, T),
    /// Normal prior
    Normal(T, T),
    /// Log-normal prior
    LogNormal(T, T),
    /// Beta prior
    Beta(T, T),
    /// Gamma prior
    Gamma(T, T),
}

/// Anomaly detection methods
#[derive(Debug, Clone, Copy)]
pub enum AnomalyDetectionMethod {
    /// Z-score based detection
    ZScore,
    /// Interquartile range method
    IQR,
    /// Isolation forest
    IsolationForest,
    /// Local outlier factor
    LocalOutlierFactor,
}
/// Noise mechanism for objective function evaluation.
///
/// # The defect this replaces
///
/// `add_noise` matched only `Gaussian` and fell through to `_ => Ok(value)`, so
/// the Laplace, Exponential, NoisyMax and SparseVector settings added **no noise
/// at all**. The privacy budget argument was named `_privacybudget` and ignored
/// entirely, so the noise scale was the constant `T::one()` regardless of the
/// epsilon the evaluation had been granted.
///
/// Now the scale is derived from the granted epsilon and the declared
/// sensitivity, and the mechanisms that are not scalar-perturbation mechanisms
/// are refused rather than silently skipped.
pub struct ObjectiveNoiseMechanism<T: Float + Debug + Send + Sync + 'static> {
    /// Mechanism type
    mechanism_type: HyperparameterNoiseMechanism,
    /// Noise parameters; `scale` records the last derived scale
    noise_params: NoiseParameters<T>,
    /// Generator, seeded from OS entropy
    rng: HpoRng,
    /// Total epsilon charged for objective perturbation
    epsilon_spent: f64,
    /// Number of noisy releases
    releases: usize,
}
impl<T: Float + Debug + Send + Sync + 'static> ObjectiveNoiseMechanism<T> {
    /// A Laplace mechanism at unit sensitivity, seeded from OS entropy.
    ///
    /// Laplace is the default because the objective release is charged in pure
    /// epsilon, matching the pure-epsilon selection that consumes it.
    pub fn new() -> Self {
        Self {
            mechanism_type: HyperparameterNoiseMechanism::Laplace,
            noise_params: NoiseParameters {
                scale: T::one(),
                sensitivity: T::one(),
                epsilon: 1.0,
                delta: None,
            },
            rng: os_seeded_hpo_rng(),
            epsilon_spent: 0.0,
            releases: 0,
        }
    }

    /// A mechanism with explicit parameters.
    pub fn with_parameters(
        mechanism_type: HyperparameterNoiseMechanism,
        noise_params: NoiseParameters<T>,
    ) -> Result<Self> {
        let sensitivity = noise_params.sensitivity.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter("the sensitivity cannot be represented as f64".to_string())
        })?;
        if !sensitivity.is_finite() || sensitivity <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the objective sensitivity must be positive and finite, got {sensitivity}"
            )));
        }
        Ok(Self {
            mechanism_type,
            noise_params,
            rng: os_seeded_hpo_rng(),
            epsilon_spent: 0.0,
            releases: 0,
        })
    }

    /// Replace the generator with a deterministic one (tests only).
    pub fn seed_for_tests(&mut self, seed: u64) {
        self.rng = scirs2_core::random::Random::seed(seed);
    }

    /// The configured mechanism.
    pub fn mechanism_type(&self) -> HyperparameterNoiseMechanism {
        self.mechanism_type
    }

    /// The noise parameters, including the last derived scale.
    pub fn noise_params(&self) -> &NoiseParameters<T> {
        &self.noise_params
    }

    /// Total epsilon charged for objective perturbation.
    pub fn epsilon_spent(&self) -> f64 {
        self.epsilon_spent
    }

    /// Number of noisy releases produced.
    pub fn releases(&self) -> usize {
        self.releases
    }

    /// Perturb one objective value under the granted budget.
    ///
    /// `privacy_budget.epsilon_consumed` is the grant for this evaluation (see
    /// [`HPOBudgetManager::get_evaluation_budget`]); the noise scale is derived
    /// from it and from the declared sensitivity.
    pub fn add_noise(&mut self, value: f64, privacy_budget: &PrivacyBudget) -> Result<f64> {
        if !value.is_finite() {
            return Err(OptimError::InvalidParameter(format!(
                "the objective value {value} is not finite, so it cannot be released"
            )));
        }
        let epsilon = privacy_budget.epsilon_consumed;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the evaluation was granted epsilon = {epsilon}; a private objective release needs \
                 a positive finite epsilon"
            )));
        }
        let sensitivity = self.noise_params.sensitivity.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter("the sensitivity cannot be represented as f64".to_string())
        })?;
        if !sensitivity.is_finite() || sensitivity <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the objective sensitivity must be positive and finite, got {sensitivity}"
            )));
        }

        let (noise, scale) = match self.mechanism_type {
            HyperparameterNoiseMechanism::Laplace => {
                let scale = sensitivity / epsilon;
                (
                    super::selection::laplace_sample(&mut self.rng, scale)?,
                    scale,
                )
            }
            HyperparameterNoiseMechanism::Gaussian => {
                let delta = self.noise_params.delta.ok_or_else(|| {
                    OptimError::InvalidConfig(
                        "the Gaussian mechanism is an (epsilon, delta) mechanism but no delta is \
                         configured in NoiseParameters"
                            .to_string(),
                    )
                })?;
                let sigma = super::selection::gaussian_sigma(sensitivity, epsilon, delta)?;
                (
                    super::selection::gaussian_sample(&mut self.rng, sigma)?,
                    sigma,
                )
            }
            HyperparameterNoiseMechanism::Exponential
            | HyperparameterNoiseMechanism::NoisyMax
            | HyperparameterNoiseMechanism::SparseVector => {
                return Err(OptimError::UnsupportedOperation(format!(
                    "{:?} selects among candidates and cannot perturb a scalar objective; \
                     configure Laplace or Gaussian for the objective release and use \
                     SelectionMechanism for the choice",
                    self.mechanism_type
                )))
            }
        };

        // The recorded scale is read back by the Bayesian surrogate to derive the
        // observation-noise variance of an already-released objective. Falling
        // back to `T::one()` on a failed conversion would hand the surrogate a
        // noise level the mechanism never used, so the conversion is fallible.
        self.noise_params.scale = T::from(scale).ok_or_else(|| {
            OptimError::InvalidState(format!(
                "the noise scale {scale} actually used for this release cannot be represented in \
                 the optimizer's value type, so it cannot be recorded"
            ))
        })?;
        self.noise_params.epsilon = epsilon;
        self.epsilon_spent += epsilon;
        self.releases += 1;
        Ok(value + noise)
    }
}
/// Private cross-validation evaluator
pub struct PrivateCrossValidation<T: Float + Debug + Send + Sync + 'static> {
    /// Number of folds
    num_folds: usize,
    /// Privacy budget per fold
    fold_budgets: Vec<PrivacyBudget>,
    /// Fold assignment strategy
    fold_strategy: FoldStrategy,
    /// Result aggregation with privacy
    private_aggregation: PrivateFoldAggregation<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivateCrossValidation<T> {
    pub fn new() -> Self {
        Self {
            num_folds: 5,
            fold_budgets: Vec::new(),
            fold_strategy: FoldStrategy::Random,
            private_aggregation: PrivateFoldAggregation::new(),
        }
    }
}
/// Types of hyperparameters
#[derive(Debug, Clone)]
pub enum ParameterType<T: Float + Debug + Send + Sync + 'static> {
    /// Continuous parameter
    Continuous,
    /// Discrete integer parameter
    Integer,
    /// Categorical parameter
    Categorical(Vec<String>),
    /// Boolean parameter
    Boolean,
    /// Ordinal parameter
    Ordinal(Vec<T>),
}
/// Hyperparameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter definitions
    pub parameters: HashMap<String, ParameterDefinition<T>>,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint<T>>,
    /// Default configuration
    pub defaultconfig: Option<ParameterConfiguration<T>>,
}
/// Parameter constraints
#[derive(Debug, Clone)]
pub struct ParameterConstraint<T: Float + Debug + Send + Sync + 'static> {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType<T>,
    /// Constraint violation penalty
    pub penalty: T,
}
/// Aggregated results from private optimization
#[derive(Debug, Clone)]
pub struct AggregatedResults<T: Float + Debug + Send + Sync + 'static> {
    /// Top-k configurations
    pub topconfigurations: Vec<(ParameterConfiguration<T>, T)>,
    /// Confidence intervals for best score
    pub confidence_intervals: Option<(T, T)>,
    /// Privacy-preserving summary statistics
    pub summary_stats: SummaryStatistics<T>,
    /// Model selection results
    pub model_selection: Option<ModelSelectionResults<T>>,
}
/// Types of acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunctionType {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Knowledge Gradient
    KnowledgeGradient,
}
/// Private objective function
pub struct PrivateObjective<T: Float + Debug + Send + Sync + 'static> {
    /// Underlying objective function
    objective_fn: ObjectiveFn<T>,
    /// Noise mechanism for objective evaluation
    noise_mechanism: ObjectiveNoiseMechanism<T>,
    /// Sensitivity analysis
    sensitivity_analyzer: ObjectiveSensitivityAnalyzer<T>,
    /// Cross-validation with privacy
    cv_evaluator: PrivateCrossValidation<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivateObjective<T> {
    /// A private objective with no function set yet.
    ///
    /// The placeholder objective **errors**. It used to be
    /// `Box::new(|_| Ok(0.0))`, so calling `evaluate` before `set_objective`
    /// returned a perfectly plausible score of zero for every configuration.
    pub fn new() -> Result<Self> {
        Self::with_noise_mechanism(ObjectiveNoiseMechanism::new())
    }

    /// A private objective with an explicit noise mechanism.
    pub fn with_noise_mechanism(noise_mechanism: ObjectiveNoiseMechanism<T>) -> Result<Self> {
        Ok(Self {
            objective_fn: Box::new(|_| {
                Err(OptimError::InvalidState(
                    "no objective function has been set on this PrivateObjective; call \
                     set_objective before evaluating"
                        .to_string(),
                ))
            }),
            noise_mechanism,
            sensitivity_analyzer: ObjectiveSensitivityAnalyzer::new(),
            cv_evaluator: PrivateCrossValidation::new(),
        })
    }

    /// Seed the noise mechanism deterministically (tests only).
    pub fn seed_for_tests(&mut self, seed: u64) {
        self.noise_mechanism.seed_for_tests(seed);
    }

    /// Install the objective function.
    pub fn set_objective(&mut self, objective_fn: ObjectiveFn<T>) -> Result<()> {
        self.objective_fn = objective_fn;
        Ok(())
    }

    /// The noise mechanism used for the objective release.
    pub fn noise_mechanism(&self) -> &ObjectiveNoiseMechanism<T> {
        &self.noise_mechanism
    }

    /// The sensitivity analyzer.
    pub fn sensitivity_analyzer(&self) -> &ObjectiveSensitivityAnalyzer<T> {
        &self.sensitivity_analyzer
    }

    /// The cross-validation evaluator.
    pub fn cv_evaluator(&self) -> &PrivateCrossValidation<T> {
        &self.cv_evaluator
    }

    /// Evaluate the objective and release a noisy value.
    pub fn evaluate(
        &mut self,
        config: &ParameterConfiguration<T>,
        privacy_budget: &PrivacyBudget,
    ) -> Result<HPOResult<T>> {
        let started = std::time::Instant::now();
        let objective_value = (self.objective_fn)(config)?;
        let noisy_value = self
            .noise_mechanism
            .add_noise(objective_value, privacy_budget)?;
        let objective = T::from(noisy_value).ok_or_else(|| {
            OptimError::InvalidParameter(format!(
                "the noisy objective value {noisy_value} cannot be represented in the parameter \
                 type"
            ))
        })?;
        let scale = self.noise_mechanism.noise_params().scale;
        Ok(HPOResult {
            objective_value: objective,
            // The released value's uncertainty is dominated by the noise that
            // was added to it; reporting `None` hid that from the caller.
            standard_error: Some(scale),
            cv_scores: None,
            training_time: Some(started.elapsed().as_secs_f64()),
            complexity_metrics: HashMap::new(),
            additional_metrics: HashMap::new(),
            status: EvaluationStatus::Success,
        })
    }
}
/// Kernel functions for Gaussian processes
pub struct KernelFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Kernel type
    kernel_type: KernelType,
    /// Kernel parameters
    parameters: Vec<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> KernelFunction<T> {
    /// An RBF kernel with a unit length scale.
    pub fn new() -> Self {
        Self {
            kernel_type: KernelType::RBF,
            parameters: vec![T::one()],
        }
    }

    /// A kernel of the given family with explicit parameters.
    pub fn with_parameters(kernel_type: KernelType, parameters: Vec<T>) -> Result<Self> {
        if parameters.is_empty() {
            return Err(OptimError::InvalidParameter(
                "a kernel needs at least one parameter".to_string(),
            ));
        }
        Ok(Self {
            kernel_type,
            parameters,
        })
    }

    /// The kernel family.
    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    /// The kernel parameters.
    pub fn parameters(&self) -> &[T] {
        &self.parameters
    }

    /// Evaluate the kernel between two encoded points.
    ///
    /// * `RBF`: `exp(-||x - y||^2 / (2 l^2))`
    /// * `Matern`: Matern-5/2, `(1 + s + s^2/3) exp(-s)` with `s = sqrt(5) r / l`
    /// * `Linear`: `l * <x, y>`
    /// * `Polynomial`: `(<x, y> + c)^d`, with `c = parameters[1]` (default 1) and
    ///   `d = parameters[2]` (default 2)
    pub fn evaluate(&self, left: &[f64], right: &[f64]) -> Result<f64> {
        if left.len() != right.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "the kernel was given {}- and {}-dimensional points",
                left.len(),
                right.len()
            )));
        }
        if left.is_empty() {
            return Err(OptimError::InvalidParameter(
                "the kernel was given zero-dimensional points".to_string(),
            ));
        }
        let scale = self
            .parameters
            .first()
            .and_then(|value| value.to_f64())
            .unwrap_or(1.0);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the kernel scale must be positive and finite, got {scale}"
            )));
        }
        let squared: f64 = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let dot: f64 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();

        Ok(match self.kernel_type {
            KernelType::RBF => (-squared / (2.0 * scale * scale)).exp(),
            KernelType::Matern => {
                let s = 5.0f64.sqrt() * squared.sqrt() / scale;
                (1.0 + s + s * s / 3.0) * (-s).exp()
            }
            KernelType::Linear => scale * dot,
            KernelType::Polynomial => {
                let offset = self
                    .parameters
                    .get(1)
                    .and_then(|value| value.to_f64())
                    .unwrap_or(1.0);
                let degree = self
                    .parameters
                    .get(2)
                    .and_then(|value| value.to_f64())
                    .unwrap_or(2.0);
                if !degree.is_finite() || degree <= 0.0 {
                    return Err(OptimError::InvalidParameter(format!(
                        "the polynomial kernel degree must be positive and finite, got {degree}"
                    )));
                }
                (dot + offset).powf(degree)
            }
        })
    }
}
/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats<T: Float + Debug + Send + Sync + 'static> {
    /// Total number of evaluations
    pub total_evaluations: usize,
    /// Number of successful evaluations
    pub successful_evaluations: usize,
    /// Number of failed evaluations
    pub failed_evaluations: usize,
    /// Average evaluation time
    pub average_evaluation_time: f64,
    /// Total optimization time
    pub total_optimization_time: f64,
    /// Iteration where convergence was detected
    pub convergence_iteration: Option<usize>,
    /// Budget efficiency score: best-score improvement per unit epsilon spent
    pub budget_efficiency: f64,
    /// Phantom data to mark type parameter as intentionally unused
    pub(super) _phantom: std::marker::PhantomData<T>,
}
/// Private Bayesian Optimization implementation
pub struct PrivateBayesianOptimization<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration
    config: PrivateHPOConfig<T>,
    /// Fitted Gaussian-process surrogate, once there is history to fit it to
    gp_model: Option<super::gaussian_process::GaussianProcessFit>,
    /// Acquisition function
    acquisition_fn: AcquisitionFunction<T>,
    /// Evaluation history
    history: Vec<HPOEvaluation<T>>,
    /// Generator used to draw the initial (history-free) configuration
    pub(super) rng: HpoRng,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivateBayesianOptimization<T> {
    /// Create a Bayesian optimizer seeded from OS entropy.
    ///
    /// An earlier revision constructed `Random::seed(42)` *inside*
    /// `suggest_next`, so the very first configuration was a compile-time
    /// constant, identical in every process and on every run.
    pub fn new(config: PrivateHPOConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            gp_model: None,
            acquisition_fn: AcquisitionFunction::new(),
            history: Vec::new(),
            rng: os_seeded_hpo_rng(),
        })
    }

    /// Create a Bayesian optimizer with a caller-supplied seed.
    ///
    /// Reproducible proposals for tests and benchmarks only.
    pub fn new_with_seed(config: PrivateHPOConfig<T>, seed: u64) -> Result<Self> {
        Ok(Self {
            config,
            gp_model: None,
            acquisition_fn: AcquisitionFunction::new(),
            history: Vec::new(),
            rng: scirs2_core::random::Random::seed(seed),
        })
    }

    /// The configuration this optimizer was built with.
    pub fn config(&self) -> &PrivateHPOConfig<T> {
        &self.config
    }

    /// The acquisition function description.
    pub fn acquisition_fn(&self) -> &AcquisitionFunction<T> {
        &self.acquisition_fn
    }

    /// Recorded evaluations.
    pub fn history(&self) -> &[HPOEvaluation<T>] {
        &self.history
    }

    /// Record an evaluation.
    pub(super) fn push_history(&mut self, evaluation: HPOEvaluation<T>) {
        self.history.push(evaluation);
    }

    /// Mutable access to the generator.
    pub(super) fn rng_mut(&mut self) -> &mut HpoRng {
        &mut self.rng
    }

    /// The most recently fitted surrogate, if any.
    pub fn surrogate(&self) -> Option<&super::gaussian_process::GaussianProcessFit> {
        self.gp_model.as_ref()
    }

    /// Whether a surrogate has been fitted.
    pub(super) fn gp_model_is_fitted(&self) -> bool {
        self.gp_model.is_some()
    }

    /// Store the surrogate that produced the latest proposal.
    pub(super) fn record_surrogate(&mut self, fit: super::gaussian_process::GaussianProcessFit) {
        self.gp_model = Some(fit);
    }
}
/// Evaluation status
#[derive(Debug, Clone, Copy)]
pub enum EvaluationStatus {
    /// Evaluation completed successfully
    Success,
    /// Evaluation failed
    Failed,
    /// Evaluation timed out
    Timeout,
    /// Evaluation was cancelled
    Cancelled,
    /// Evaluation is in progress
    InProgress,
}
/// Types of utility functions
#[derive(Debug, Clone, Copy)]
pub enum UtilityFunctionType {
    /// Linear utility
    Linear,
    /// Exponential utility
    Exponential,
    /// Logarithmic utility
    Logarithmic,
    /// Quadratic utility
    Quadratic,
    /// Custom utility function
    Custom,
}
/// Sample-based sensitivity estimator
pub struct SampleBasedSensitivityEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Number of samples for estimation
    num_samples: usize,
    /// Sampling strategy
    sampling_strategy: SamplingStrategy,
    /// Confidence level for bounds
    confidence_level: f64,
    /// Bootstrap estimator
    bootstrap_estimator: BootstrapEstimator<T>,
    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> SampleBasedSensitivityEstimator<T> {
    pub fn new() -> Self {
        Self {
            num_samples: 1000,
            sampling_strategy: SamplingStrategy::Uniform,
            confidence_level: 0.95,
            bootstrap_estimator: BootstrapEstimator::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}
/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Test conclusion
    pub conclusion: TestConclusion,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}
/// Anomaly detector for results
pub struct AnomalyDetector<T: Float + Debug + Send + Sync + 'static> {
    /// Detection threshold
    threshold: T,
    /// Detection method
    detection_method: AnomalyDetectionMethod,
    /// Historical baseline
    baseline: Option<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> AnomalyDetector<T> {
    pub fn new() -> Self {
        Self {
            threshold: T::from(3.0).unwrap_or_else(|| T::zero()),
            detection_method: AnomalyDetectionMethod::ZScore,
            baseline: None,
        }
    }
}
/// Hyperparameter configuration
#[derive(Debug, Clone)]
pub struct ParameterConfiguration<T: Float + Debug + Send + Sync + 'static> {
    /// Parameter values
    pub values: HashMap<String, ParameterValue<T>>,
    /// Configuration identifier
    pub id: String,
    /// Configuration metadata
    pub metadata: HashMap<String, String>,
}
/// Confidence interval estimation
pub struct ConfidenceEstimation<T: Float + Debug + Send + Sync + 'static> {
    /// Confidence level
    confidence_level: f64,
    /// Estimation method
    estimation_method: ConfidenceEstimationMethod,
    /// Bootstrap parameters
    bootstrap_params: Option<BootstrapParams>,
    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> ConfidenceEstimation<T> {
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            estimation_method: ConfidenceEstimationMethod::Normal,
            bootstrap_params: None,
            _phantom: std::marker::PhantomData,
        }
    }
}
/// Bootstrap estimator for sensitivity
pub struct BootstrapEstimator<T: Float + Debug + Send + Sync + 'static> {
    /// Number of bootstrap samples
    num_bootstrap: usize,
    /// Bootstrap confidence interval
    confidence_interval: (f64, f64),
    /// Bias correction
    bias_correction: bool,
    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Float + Debug + Send + Sync + 'static> BootstrapEstimator<T> {
    pub fn new() -> Self {
        Self {
            num_bootstrap: 1000,
            confidence_interval: (0.025, 0.975),
            bias_correction: true,
            _phantom: std::marker::PhantomData,
        }
    }
}
/// Validation strategies for private hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum ValidationStrategy {
    /// Hold-out validation
    HoldOut,
    /// K-fold cross-validation with privacy
    KFoldCV,
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Bootstrap validation
    Bootstrap,
    /// Time series split for temporal data
    TimeSeriesSplit,
}
/// Selection mechanisms for final hyperparameter choice
///
/// The mechanism implementations live in [`super::selection`]. The important
/// property is that [`SelectionMechanism::select_index`] never returns the exact
/// argmax deterministically, and always charges the epsilon it used.
pub struct SelectionMechanism<T: Float + Debug + Send + Sync + 'static> {
    /// Mechanism type
    mechanism_type: HyperparameterNoiseMechanism,
    /// Selection parameters
    selection_params: SelectionParameters<T>,
    /// Utility function for selection
    utility_function: UtilityFunction<T>,
    /// Generator used by the noise-adding mechanisms
    rng: HpoRng,
    /// Seed recorded when the mechanism was seeded for tests
    test_seed: Option<u64>,
    /// Total epsilon charged so far
    epsilon_spent: f64,
    /// Total delta charged so far
    delta_spent: f64,
    /// Number of selections performed
    selection_count: usize,
}
impl<T: Float + Debug + Send + Sync + 'static> SelectionMechanism<T> {
    /// An exponential mechanism at `epsilon = 1`, `Delta_u = 1`, seeded from OS
    /// entropy.
    pub fn new() -> Self {
        Self {
            mechanism_type: HyperparameterNoiseMechanism::Exponential,
            selection_params: SelectionParameters::pure_epsilon(1.0, T::one()),
            utility_function: UtilityFunction::new(),
            rng: os_seeded_hpo_rng(),
            test_seed: None,
            epsilon_spent: 0.0,
            delta_spent: 0.0,
            selection_count: 0,
        }
    }

    /// The configured mechanism.
    pub fn mechanism_type(&self) -> HyperparameterNoiseMechanism {
        self.mechanism_type
    }

    /// Replace the configured mechanism.
    pub fn set_mechanism_type(&mut self, mechanism_type: HyperparameterNoiseMechanism) {
        self.mechanism_type = mechanism_type;
    }

    /// The selection parameters.
    pub fn selection_params(&self) -> &SelectionParameters<T> {
        &self.selection_params
    }

    /// Replace the selection parameters, validating them.
    pub fn set_selection_parameters(&mut self, params: SelectionParameters<T>) -> Result<()> {
        if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the selection epsilon must be positive and finite, got {}",
                params.epsilon
            )));
        }
        let sensitivity = params.utility_sensitivity.to_f64().ok_or_else(|| {
            OptimError::InvalidParameter(
                "the utility sensitivity cannot be represented as f64".to_string(),
            )
        })?;
        if !sensitivity.is_finite() || sensitivity <= 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "the utility sensitivity must be positive and finite, got {sensitivity}"
            )));
        }
        if let Some(delta) = params.delta {
            if !delta.is_finite() || !(0.0..1.0).contains(&delta) || delta <= 0.0 {
                return Err(OptimError::InvalidParameter(format!(
                    "the selection delta must lie in (0, 1), got {delta}"
                )));
            }
        }
        if !params.temperature.is_finite() || params.temperature <= T::zero() {
            return Err(OptimError::InvalidParameter(
                "the selection temperature must be positive and finite".to_string(),
            ));
        }
        self.selection_params = params;
        Ok(())
    }

    /// The utility function.
    pub fn utility_function(&self) -> &UtilityFunction<T> {
        &self.utility_function
    }

    /// Replace the utility function.
    pub fn set_utility_function(&mut self, utility_function: UtilityFunction<T>) {
        self.utility_function = utility_function;
    }

    /// Total epsilon charged for selections.
    pub fn epsilon_spent(&self) -> f64 {
        self.epsilon_spent
    }

    /// Total delta charged for selections.
    pub fn delta_spent(&self) -> f64 {
        self.delta_spent
    }

    /// Number of selections performed.
    pub fn selection_count(&self) -> usize {
        self.selection_count
    }

    /// Mutable access to the generator, for the mechanisms in
    /// [`super::selection`].
    pub(super) fn rng_mut(&mut self) -> &mut HpoRng {
        &mut self.rng
    }

    /// Replace the generator.
    pub(super) fn set_rng(&mut self, rng: HpoRng) {
        self.rng = rng;
    }

    /// The recorded test seed, if any.
    pub(super) fn test_seed(&self) -> Option<u64> {
        self.test_seed
    }

    /// Record the test seed.
    pub(super) fn set_test_seed(&mut self, seed: Option<u64>) {
        self.test_seed = seed;
    }

    /// Charge one selection.
    pub(super) fn record_selection(&mut self, epsilon: f64, delta: f64) {
        self.epsilon_spent += epsilon;
        self.delta_spent += delta;
        self.selection_count += 1;
    }
}
/// Validation rule for results
pub struct ValidationRule<T: Float + Debug + Send + Sync + 'static> {
    /// Rule name
    pub name: String,
    /// Rule function
    pub rule_fn: RuleFn<T>,
    /// Rule weight
    pub weight: f64,
}
/// Private hyperparameter optimization results
///
/// # 0.3.2 change
///
/// `selection` was added. It records whether the returned configuration was
/// chosen by a differentially private mechanism, which mechanism, and what it
/// cost. Before this release the choice was always the exact argmax with no
/// indication that it leaked.
#[derive(Debug, Clone)]
pub struct PrivateHPOResults<T: Float + Debug + Send + Sync + 'static> {
    /// Best configuration found
    pub bestconfiguration: Option<ParameterConfiguration<T>>,
    /// Best objective score
    pub best_score: T,
    /// All evaluations performed
    pub all_evaluations: Vec<HPOEvaluation<T>>,
    /// Final aggregated results
    pub final_results: AggregatedResults<T>,
    /// Total privacy cost
    pub total_privacy_cost: PrivacyBudget,
    /// Optimization statistics
    pub optimization_stats: OptimizationStats<T>,
    /// How the reported configuration was selected
    pub selection: SelectionReport,
}
/// Private Random Search implementation
pub struct PrivateRandomSearch<T: Float + Debug + Send + Sync + 'static> {
    /// Configuration
    config: PrivateHPOConfig<T>,
    /// Random number generator
    pub(super) rng: scirs2_core::random::Random<StdRng>,
    /// Evaluation history
    pub(super) history: Vec<HPOEvaluation<T>>,
}
impl<T: Float + Debug + Send + Sync + 'static> PrivateRandomSearch<T> {
    /// Create a random search seeded from OS entropy.
    ///
    /// The generator must not be seeded from a constant: a fixed seed makes
    /// the sequence of proposed hyperparameter configurations identical in
    /// every process, so an adversary knows exactly which configurations were
    /// evaluated on the private data.
    pub fn new(config: PrivateHPOConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            rng: os_seeded_hpo_rng(),
            history: Vec::new(),
        })
    }

    /// Create a random search with a caller-supplied seed.
    ///
    /// Reproducible search order for tests and benchmarks only; never use
    /// this when the search touches real private data.
    pub fn new_with_seed(config: PrivateHPOConfig<T>, seed: u64) -> Result<Self> {
        Ok(Self {
            config,
            rng: scirs2_core::random::Random::seed(seed),
            history: Vec::new(),
        })
    }
}
/// Hyperparameter optimization result
#[derive(Debug, Clone)]
pub struct HPOResult<T: Float + Debug + Send + Sync + 'static> {
    /// Objective value
    pub objective_value: T,
    /// Standard error (if available)
    pub standard_error: Option<T>,
    /// Cross-validation scores
    pub cv_scores: Option<Vec<T>>,
    /// Training time
    pub training_time: Option<f64>,
    /// Model complexity metrics
    pub complexity_metrics: HashMap<String, T>,
    /// Additional metrics
    pub additional_metrics: HashMap<String, T>,
    /// Result status
    pub status: EvaluationStatus,
}
/// Methods for estimating objective sensitivity
#[derive(Debug, Clone, Copy)]
pub enum SensitivityEstimationMethod {
    /// Global sensitivity (worst-case)
    Global,
    /// Local sensitivity (data-dependent)
    Local,
    /// Smooth sensitivity
    Smooth,
    /// Sample-based estimation
    SampleBased,
    /// Theoretical bounds
    Theoretical,
}
/// Result aggregation strategies
#[derive(Debug, Clone, Copy)]
pub enum ResultAggregationStrategy {
    /// Select best configuration
    SelectBest,
    /// Ensemble of top configurations
    Ensemble,
    /// Weighted combination
    WeightedCombination,
    /// Consensus-based selection
    Consensus,
    /// Multi-objective selection
    MultiObjective,
}
