//! Hyperparameter optimization for Neural Architecture Search.
//!
//! This module provides hyperparameter optimization capabilities for both
//! the NAS process itself and the discovered architectures.
//!
//! Every strategy [`HyperparameterOptimizer`] dispatches to is a real algorithm:
//!
//! | [`OptimizationStrategy`] | implementation |
//! |---|---|
//! | `Random` | prior sampling, honoring each parameter's [`DistributionType`] |
//! | `Grid` | exhaustive mixed-radix enumeration ([`grid`]) |
//! | `TPE` | two adaptive Parzen estimators over a good/bad quantile split ([`tpe`]) |
//! | `Bayesian` | GP-free kernel-regression surrogate + acquisition maximization ([`surrogate`]) |
//! | `Evolutionary` | tournament selection, crossover and bounded mutation ([`evolution`]) |
//!
//! `Grid`, `Bayesian`, `TPE` and `Evolutionary` previously all returned
//! `self.random_search()`, so selecting any of them silently performed random
//! search. The model-based strategies now require a random *initial design* before
//! they can be fitted (which is part of the published algorithms); that phase is
//! reported through [`HyperparameterOptimizer::last_strategy_used`] instead of
//! being hidden, and [`ParticleSwarm`](OptimizationStrategy::ParticleSwarm),
//! `SuccessiveHalving`, `Hyperband` and `BOHB` — which have no implementation
//! here — return an error rather than quietly becoming random search.

pub mod evolution;
pub mod grid;
pub mod support;
pub mod surrogate;
pub mod tpe;

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use support::StrategyRng;

/// Hyperparameter optimizer for NAS and discovered architectures
#[derive(Debug)]
pub struct HyperparameterOptimizer<T: Float> {
    /// Search space definition
    search_space: HyperparameterSpace<T>,

    /// Optimization strategy
    strategy: OptimizationStrategy,

    /// Evaluation history
    evaluation_history: Vec<HyperparameterEvaluation<T>>,

    /// Current best configuration
    best_config: Option<HyperparameterConfiguration<T>>,

    /// Optimization state
    state: OptimizerState<T>,

    /// RNG shared by every stochastic strategy, so a seeded optimizer is
    /// reproducible end to end.
    rng: StrategyRng,

    /// Number of points per continuous grid axis.
    grid_resolution: usize,

    /// TPE tunables.
    tpe_config: tpe::TpeConfig,

    /// Surrogate (Bayesian) tunables.
    surrogate_config: surrogate::SurrogateConfig,

    /// Evolutionary tunables.
    evolution_config: evolution::EvolutionConfig,

    /// Acquisition function used by the Bayesian strategy.
    acquisition: AcquisitionFunction,

    /// Which strategy actually produced the most recent suggestion. Model-based
    /// strategies need a random initial design first; this records honestly which
    /// of the two ran instead of leaving the caller to assume.
    last_strategy_used: OptimizationStrategy,
}

/// Hyperparameter search space definition
#[derive(Debug, Clone)]
pub struct HyperparameterSpace<T: Float> {
    /// Individual hyperparameter ranges
    parameters: HashMap<String, ParameterRange<T>>,

    /// Parameter dependencies
    dependencies: Vec<ParameterDependency>,

    /// Constraints on parameter combinations
    constraints: Vec<HyperparameterConstraint<T>>,

    /// Categorical parameters
    categorical_parameters: HashMap<String, Vec<String>>,
}

/// Range specification for a continuous hyperparameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange<T: Float> {
    /// Parameter name
    pub name: String,

    /// Minimum value
    pub min_value: T,

    /// Maximum value
    pub max_value: T,

    /// Distribution type for sampling
    pub distribution: DistributionType,

    /// Whether to use log scale
    pub log_scale: bool,

    /// Discrete values (if applicable)
    pub discrete_values: Option<Vec<T>>,
}

/// Type of probability distribution for parameter sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionType {
    /// Uniform distribution
    Uniform,
    /// Normal (Gaussian) distribution
    Normal,
    /// Log-normal distribution
    LogNormal,
    /// Beta distribution
    Beta,
    /// Exponential distribution
    Exponential,
}

/// Dependency between parameters
#[derive(Debug, Clone)]
pub struct ParameterDependency {
    /// Dependent parameter name
    pub dependent: String,

    /// Parent parameter name
    pub parent: String,

    /// Dependency type
    pub dependency_type: DependencyType,

    /// Condition for activation
    pub condition: DependencyCondition,
}

/// Type of parameter dependency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyType {
    /// Parameter is only active when condition is met
    Conditional,
    /// Parameter value is derived from parent
    Derived,
    /// Parameter range depends on parent value
    RangeDependent,
}

/// Condition for parameter dependency
#[derive(Debug, Clone)]
pub enum DependencyCondition {
    /// Parent equals specific value
    Equals(String),
    /// Parent greater than value
    GreaterThan(f64),
    /// Parent less than value
    LessThan(f64),
    /// Parent in range
    InRange(f64, f64),
    /// Parent in set of values
    InSet(Vec<String>),
}

/// Constraint on hyperparameter combinations
#[derive(Debug, Clone)]
pub struct HyperparameterConstraint<T: Float> {
    /// Constraint name
    pub name: String,

    /// Parameters involved in constraint
    pub parameters: Vec<String>,

    /// Constraint type
    pub constraint_type: ConstraintType<T>,

    /// Violation penalty
    pub penalty: T,
}

/// Type of hyperparameter constraint
#[derive(Debug, Clone)]
pub enum ConstraintType<T: Float> {
    /// Linear constraint: sum(coeffs * params) <= bound
    Linear { coefficients: Vec<T>, bound: T },
    /// Nonlinear constraint with custom function
    NonLinear { function_name: String },
    /// Mutual exclusion: only one parameter can be active
    MutualExclusion,
    /// Ordering constraint: param1 <= param2 <= ... <= paramN
    Ordering,
}

/// Hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfiguration<T: Float> {
    /// Configuration ID
    pub id: String,

    /// Parameter values
    pub parameters: HashMap<String, T>,

    /// Categorical parameter values
    pub categorical_parameters: HashMap<String, String>,

    /// Configuration score/performance
    pub score: Option<T>,

    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}

/// Evaluation result for a hyperparameter configuration
#[derive(Debug, Clone)]
pub struct HyperparameterEvaluation<T: Float> {
    /// Configuration that was evaluated
    pub configuration: HyperparameterConfiguration<T>,

    /// Performance metrics
    pub metrics: EvaluationMetrics<T>,

    /// Evaluation duration
    pub duration_seconds: f64,

    /// Success/failure status
    pub status: EvaluationStatus,

    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Performance metrics from hyperparameter evaluation
#[derive(Debug, Clone)]
pub struct EvaluationMetrics<T: Float> {
    /// Primary objective value
    pub primary_objective: T,

    /// Secondary objectives
    pub secondary_objectives: HashMap<String, T>,

    /// Validation score
    pub validation_score: Option<T>,

    /// Training time
    pub training_time: f64,

    /// Memory usage
    pub memory_usage: f64,

    /// Convergence information
    pub converged: bool,

    /// Number of iterations to convergence
    pub convergence_iterations: Option<u32>,
}

/// Status of hyperparameter evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationStatus {
    /// Evaluation completed successfully
    Success,
    /// Evaluation failed due to error
    Failed,
    /// Evaluation was terminated early
    Terminated,
    /// Evaluation is still running
    Running,
    /// Evaluation timed out
    Timeout,
}

/// Optimization strategies for hyperparameter search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Random search
    Random,
    /// Grid search
    Grid,
    /// Bayesian optimization
    Bayesian,
    /// Evolutionary algorithms
    Evolutionary,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Tree-structured Parzen Estimator (TPE)
    TPE,
    /// Successive halving
    SuccessiveHalving,
    /// Hyperband
    Hyperband,
    /// BOHB (Bayesian Optimization and HyperBand)
    BOHB,
}

/// Internal state of the hyperparameter optimizer
#[derive(Debug)]
pub struct OptimizerState<T: Float> {
    /// Current iteration/generation
    pub iteration: u32,

    /// Number of evaluations performed
    pub num_evaluations: u32,

    /// Best score found so far
    pub best_score: Option<T>,

    /// Population (for evolutionary strategies)
    pub population: Vec<HyperparameterConfiguration<T>>,

    /// Gaussian process model (for Bayesian optimization)
    pub surrogate_model: Option<SurrogateModel<T>>,

    /// Early stopping information
    pub early_stopping: EarlyStoppingState<T>,
}

/// Surrogate model for Bayesian optimization
#[derive(Debug)]
pub struct SurrogateModel<T: Float> {
    /// Model type
    pub model_type: SurrogateModelType,

    /// Training data points
    pub training_data: Vec<(Vec<T>, T)>,

    /// Model hyperparameters
    pub hyperparameters: HashMap<String, f64>,

    /// Acquisition function
    pub acquisition_function: AcquisitionFunction,
}

/// Type of surrogate model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurrogateModelType {
    /// Nadaraya-Watson kernel regression — the surrogate this crate actually
    /// implements (see [`surrogate::KernelSurrogate`]). It needs no linear algebra
    /// and no kernel-hyperparameter fitting, which is why it is the default.
    KernelRegression,
    /// Gaussian Process. **Not implemented in this module**; kept because it is
    /// part of the public type and callers may record their own model choice.
    /// [`HyperparameterOptimizer`] never fits one.
    GaussianProcess,
    /// Random Forest. Not implemented here; see [`SurrogateModelType::GaussianProcess`].
    RandomForest,
    /// Neural Network. Not implemented here; see [`SurrogateModelType::GaussianProcess`].
    NeuralNetwork,
    /// Polynomial regression. Not implemented here; see
    /// [`SurrogateModelType::GaussianProcess`].
    Polynomial,
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Entropy Search
    EntropySearch,
}

/// Early stopping state
#[derive(Debug)]
pub struct EarlyStoppingState<T: Float> {
    /// Whether early stopping is enabled
    pub enabled: bool,

    /// Patience (iterations without improvement)
    pub patience: u32,

    /// Current patience counter
    pub patience_counter: u32,

    /// Minimum improvement threshold
    pub min_improvement: T,

    /// Best score for early stopping comparison
    pub best_score_for_stopping: Option<T>,
}

impl<T: Float + Default + Clone + Send + Sync> HyperparameterOptimizer<T> {
    /// Create a new hyperparameter optimizer, seeded from OS entropy.
    pub fn new(search_space: HyperparameterSpace<T>, strategy: OptimizationStrategy) -> Self {
        Self::with_seed(search_space, strategy, scirs2_core::random::random::<u64>())
    }

    /// Create a fully reproducible optimizer.
    pub fn with_seed(
        search_space: HyperparameterSpace<T>,
        strategy: OptimizationStrategy,
        seed: u64,
    ) -> Self {
        Self {
            search_space,
            strategy,
            evaluation_history: Vec::new(),
            best_config: None,
            rng: Random::seed(seed),
            grid_resolution: grid::DEFAULT_GRID_RESOLUTION,
            tpe_config: tpe::TpeConfig::default(),
            surrogate_config: surrogate::SurrogateConfig::default(),
            evolution_config: evolution::EvolutionConfig::default(),
            acquisition: AcquisitionFunction::ExpectedImprovement,
            last_strategy_used: strategy,
            state: OptimizerState {
                iteration: 0,
                num_evaluations: 0,
                best_score: None,
                population: Vec::new(),
                surrogate_model: None,
                early_stopping: EarlyStoppingState {
                    enabled: false,
                    patience: 50,
                    patience_counter: 0,
                    min_improvement: T::from(0.001).unwrap_or_else(T::zero),
                    best_score_for_stopping: None,
                },
            },
        }
    }

    /// Generate the next hyperparameter configuration to evaluate.
    ///
    /// Returns an error for a strategy this module does not implement
    /// ([`OptimizationStrategy::ParticleSwarm`], `SuccessiveHalving`, `Hyperband`,
    /// `BOHB`) rather than silently performing random search, and for a search
    /// space that declares no parameters at all.
    ///
    /// The model-based strategies (`Bayesian`, `TPE`, `Evolutionary`) need a
    /// random initial design first. During that phase they sample from the prior
    /// and set [`HyperparameterOptimizer::last_strategy_used`] to
    /// [`OptimizationStrategy::Random`], so the caller can see which algorithm
    /// actually produced the suggestion.
    pub fn suggest_configuration(&mut self) -> Result<HyperparameterConfiguration<T>> {
        if self.search_space.is_empty() {
            return Err(OptimError::InvalidConfig(
                "hyperparameter search requires a non-empty search space".to_string(),
            ));
        }

        let (parameters, categorical_parameters, used) = match self.strategy {
            OptimizationStrategy::Random => {
                let (parameters, categorical) = self.sample_from_prior();
                (parameters, categorical, OptimizationStrategy::Random)
            }
            OptimizationStrategy::Grid => {
                let index = self.state.num_evaluations as usize;
                let (parameters, categorical) = grid::suggest(
                    &self.search_space,
                    index,
                    self.grid_resolution,
                    &mut self.rng,
                )?;
                (parameters, categorical, OptimizationStrategy::Grid)
            }
            OptimizationStrategy::TPE => {
                let scored = Self::scored_observations(&self.evaluation_history);
                match tpe::suggest(&self.search_space, &scored, &self.tpe_config, &mut self.rng) {
                    Ok((parameters, categorical)) => {
                        (parameters, categorical, OptimizationStrategy::TPE)
                    }
                    Err(_) if scored.len() < self.tpe_config.startup_trials.max(2) => {
                        // Documented initial design: TPE cannot fit two density
                        // estimators before it has observations.
                        drop(scored);
                        let (parameters, categorical) = self.sample_from_prior();
                        (parameters, categorical, OptimizationStrategy::Random)
                    }
                    Err(error) => return Err(error),
                }
            }
            OptimizationStrategy::Bayesian => {
                let scored = Self::scored_observations(&self.evaluation_history);
                match surrogate::suggest(
                    &self.search_space,
                    &scored,
                    self.acquisition,
                    &self.surrogate_config,
                    &mut self.rng,
                ) {
                    Ok((parameters, categorical)) => {
                        (parameters, categorical, OptimizationStrategy::Bayesian)
                    }
                    Err(_) if scored.len() < self.surrogate_config.startup_trials.max(2) => {
                        drop(scored);
                        let (parameters, categorical) = self.sample_from_prior();
                        (parameters, categorical, OptimizationStrategy::Random)
                    }
                    Err(error) => return Err(error),
                }
            }
            OptimizationStrategy::Evolutionary => {
                let population: Vec<(&HyperparameterConfiguration<T>, f64)> = self
                    .state
                    .population
                    .iter()
                    .filter_map(|config| config.score.and_then(|s| s.to_f64()).map(|s| (config, s)))
                    .collect();
                if population.is_empty() {
                    drop(population);
                    let (parameters, categorical) = self.sample_from_prior();
                    (parameters, categorical, OptimizationStrategy::Random)
                } else {
                    let (parameters, categorical) = evolution::breed(
                        &self.search_space,
                        &population,
                        &self.evolution_config,
                        &mut self.rng,
                    )?;
                    (parameters, categorical, OptimizationStrategy::Evolutionary)
                }
            }
            unimplemented => {
                return Err(OptimError::NotImplemented(format!(
                    "hyperparameter strategy {:?} is not implemented in optirs-nas; \
                     available strategies are Random, Grid, TPE, Bayesian and Evolutionary",
                    unimplemented
                )))
            }
        };

        self.last_strategy_used = used;
        Ok(HyperparameterConfiguration {
            id: format!("config_{}", self.state.num_evaluations),
            parameters,
            categorical_parameters,
            score: None,
            metadata: HashMap::from([("strategy".to_string(), format!("{:?}", used))]),
        })
    }

    /// Which strategy produced the most recent suggestion. Differs from
    /// [`HyperparameterOptimizer::strategy`] exactly while a model-based strategy
    /// is still running its random initial design.
    pub fn last_strategy_used(&self) -> OptimizationStrategy {
        self.last_strategy_used
    }

    /// The configured strategy.
    pub fn strategy(&self) -> OptimizationStrategy {
        self.strategy
    }

    /// Number of points per continuous grid axis.
    pub fn grid_resolution(&self) -> usize {
        self.grid_resolution
    }

    /// Set the number of points per continuous grid axis.
    pub fn set_grid_resolution(&mut self, resolution: usize) {
        self.grid_resolution = resolution.max(1);
    }

    /// Total number of distinct grid points, or `None` if the product overflows.
    pub fn grid_size(&self) -> Option<usize> {
        grid::grid_size(&self.search_space, self.grid_resolution)
    }

    /// Choose the acquisition function used by the `Bayesian` strategy.
    pub fn set_acquisition_function(&mut self, acquisition: AcquisitionFunction) {
        self.acquisition = acquisition;
    }

    /// Override the TPE tunables.
    pub fn set_tpe_config(&mut self, config: tpe::TpeConfig) {
        self.tpe_config = config;
    }

    /// Override the Bayesian surrogate tunables.
    pub fn set_surrogate_config(&mut self, config: surrogate::SurrogateConfig) {
        self.surrogate_config = config;
    }

    /// Override the evolutionary tunables.
    pub fn set_evolution_config(&mut self, config: evolution::EvolutionConfig) {
        self.evolution_config = config;
    }

    /// The search space this optimizer explores.
    pub fn get_search_space(&self) -> &HyperparameterSpace<T> {
        &self.search_space
    }

    /// The evolutionary population (fittest first), which is now genuinely read by
    /// [`OptimizationStrategy::Evolutionary`] rather than being write-only state.
    pub fn population(&self) -> &[HyperparameterConfiguration<T>] {
        &self.state.population
    }

    /// The fitted surrogate's recorded training data, or `None` before the
    /// Bayesian strategy has run.
    pub fn surrogate_model(&self) -> Option<&SurrogateModel<T>> {
        self.state.surrogate_model.as_ref()
    }

    /// Scored observations from the evaluation history, newest last.
    fn scored_observations(
        history: &[HyperparameterEvaluation<T>],
    ) -> Vec<(&HyperparameterConfiguration<T>, f64)> {
        history
            .iter()
            .filter(|evaluation| matches!(evaluation.status, EvaluationStatus::Success))
            .filter_map(|evaluation| {
                evaluation
                    .configuration
                    .score
                    .and_then(|score| score.to_f64())
                    .map(|score| (&evaluation.configuration, score))
            })
            .collect()
    }

    /// Record evaluation result
    pub fn record_evaluation(&mut self, evaluation: HyperparameterEvaluation<T>) {
        self.state.num_evaluations += 1;

        // Update best configuration
        if let Some(score) = evaluation.configuration.score {
            let is_better = match self.best_config.as_ref().and_then(|best| best.score) {
                Some(best) => score > best,
                None => true,
            };
            if is_better {
                self.best_config = Some(evaluation.configuration.clone());
                self.state.best_score = Some(score);
            }
        }

        // Update early stopping state
        if self.state.early_stopping.enabled {
            self.update_early_stopping(&evaluation);
        }

        // Store evaluation
        self.evaluation_history.push(evaluation);

        // Update strategy-specific state
        self.update_strategy_state();
    }

    /// Check if optimization should stop early
    pub fn should_stop_early(&self) -> bool {
        if !self.state.early_stopping.enabled {
            return false;
        }

        self.state.early_stopping.patience_counter >= self.state.early_stopping.patience
    }

    /// Get the best configuration found so far
    pub fn get_best_configuration(&self) -> Option<&HyperparameterConfiguration<T>> {
        self.best_config.as_ref()
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> OptimizationStatistics<T> {
        let scores: Vec<T> = self
            .evaluation_history
            .iter()
            .filter_map(|eval| eval.configuration.score)
            .collect();

        let mean_score = if scores.is_empty() {
            T::zero()
        } else {
            scores.iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(scores.len()).unwrap_or_else(|| T::one())
        };

        OptimizationStatistics {
            num_evaluations: self.state.num_evaluations,
            best_score: self.state.best_score,
            mean_score: Some(mean_score),
            num_successful_evaluations: self
                .evaluation_history
                .iter()
                .filter(|eval| matches!(eval.status, EvaluationStatus::Success))
                .count() as u32,
            convergence_iteration: self.get_convergence_iteration(),
        }
    }

    // Private methods for different optimization strategies

    /// Draw one value per parameter from its declared prior. Shared by the
    /// `Random` strategy and by the initial-design phase of the model-based
    /// strategies.
    ///
    /// Unlike the previous version this honors every [`DistributionType`] (the old
    /// code had a `_ => uniform` arm covering four of five variants), respects
    /// `discrete_values`, and contains no `expect`: a value that cannot be
    /// converted falls back to the range's own bound rather than aborting.
    fn sample_from_prior(&mut self) -> (HashMap<String, T>, HashMap<String, String>) {
        let mut parameters = HashMap::new();
        let mut categorical_parameters = HashMap::new();

        for name in support::sorted_parameter_names(&self.search_space) {
            let Some(range) = self.search_space.parameters.get(&name).cloned() else {
                continue;
            };
            let raw =
                support::snap_to_discrete(&range, support::sample_parameter(&range, &mut self.rng));
            parameters.insert(
                name,
                scirs2_core::numeric::NumCast::from(raw).unwrap_or(range.min_value),
            );
        }

        for name in support::sorted_categorical_names(&self.search_space) {
            let Some(categories) = self.search_space.categorical_parameters.get(&name).cloned()
            else {
                continue;
            };
            if categories.is_empty() {
                continue;
            }
            let idx = self.rng.gen_range(0..categories.len());
            categorical_parameters.insert(name, categories[idx].clone());
        }

        (parameters, categorical_parameters)
    }

    fn update_early_stopping(&mut self, evaluation: &HyperparameterEvaluation<T>) {
        if let Some(score) = evaluation.configuration.score {
            if let Some(best_score) = self.state.early_stopping.best_score_for_stopping {
                if score > best_score + self.state.early_stopping.min_improvement {
                    self.state.early_stopping.best_score_for_stopping = Some(score);
                    self.state.early_stopping.patience_counter = 0;
                } else {
                    self.state.early_stopping.patience_counter += 1;
                }
            } else {
                self.state.early_stopping.best_score_for_stopping = Some(score);
            }
        }
    }

    /// Refresh the state each strategy reads between suggestions.
    ///
    /// * The evolutionary population becomes a real, bounded, elitist pool
    ///   (previously it was filled once with random configurations and never read).
    /// * The Bayesian surrogate's recorded training data is rebuilt from the scored
    ///   history, so `state.surrogate_model` stops being permanently `None`.
    fn update_strategy_state(&mut self) {
        self.state.iteration += 1;

        match self.strategy {
            OptimizationStrategy::Evolutionary => {
                let scored: Vec<HyperparameterConfiguration<T>> = self
                    .evaluation_history
                    .iter()
                    .filter(|evaluation| {
                        matches!(evaluation.status, EvaluationStatus::Success)
                            && evaluation.configuration.score.is_some()
                    })
                    .map(|evaluation| evaluation.configuration.clone())
                    .collect();
                self.state.population = scored;
                evolution::survivor_selection(&mut self.state.population, &self.evolution_config);
            }
            OptimizationStrategy::Bayesian => {
                let continuous = support::sorted_parameter_names(&self.search_space);
                let training_data: Vec<(Vec<T>, T)> = self
                    .evaluation_history
                    .iter()
                    .filter_map(|evaluation| {
                        let score = evaluation.configuration.score?;
                        let vector = continuous
                            .iter()
                            .map(|name| {
                                evaluation
                                    .configuration
                                    .parameters
                                    .get(name)
                                    .copied()
                                    .unwrap_or_else(T::zero)
                            })
                            .collect();
                        Some((vector, score))
                    })
                    .collect();
                let mut hyperparameters = HashMap::new();
                hyperparameters.insert("bandwidth".to_string(), self.surrogate_config.bandwidth);
                hyperparameters.insert("ucb_beta".to_string(), self.surrogate_config.ucb_beta);
                self.state.surrogate_model = Some(SurrogateModel {
                    model_type: surrogate::IMPLEMENTED_SURROGATE,
                    training_data,
                    hyperparameters,
                    acquisition_function: self.acquisition,
                });
            }
            _ => {}
        }
    }

    fn get_convergence_iteration(&self) -> Option<u32> {
        // Simple convergence detection based on improvement rate
        let window_size = 10;
        if self.evaluation_history.len() < window_size * 2 {
            return None;
        }

        let recent_scores: Vec<T> = self
            .evaluation_history
            .iter()
            .rev()
            .take(window_size)
            .filter_map(|eval| eval.configuration.score)
            .collect();

        let older_scores: Vec<T> = self
            .evaluation_history
            .iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .filter_map(|eval| eval.configuration.score)
            .collect();

        if recent_scores.len() == window_size && older_scores.len() == window_size {
            let recent_mean = recent_scores.iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(window_size).unwrap_or_else(|| T::one());
            let older_mean = older_scores.iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(window_size).unwrap_or_else(|| T::one());

            let improvement = recent_mean - older_mean;
            let threshold = T::from(0.001).unwrap_or_else(T::zero);

            if improvement < threshold {
                return Some(self.state.iteration - window_size as u32);
            }
        }

        None
    }
}

/// Statistics about the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationStatistics<T: Float> {
    /// Total number of evaluations
    pub num_evaluations: u32,

    /// Best score achieved
    pub best_score: Option<T>,

    /// Mean score across all evaluations
    pub mean_score: Option<T>,

    /// Number of successful evaluations
    pub num_successful_evaluations: u32,

    /// Iteration at which convergence was detected
    pub convergence_iteration: Option<u32>,
}

impl<T: Float> HyperparameterSpace<T> {
    /// Create a new hyperparameter search space
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            dependencies: Vec::new(),
            constraints: Vec::new(),
            categorical_parameters: HashMap::new(),
        }
    }

    /// Continuous parameter ranges, keyed by name.
    pub fn parameter_ranges(&self) -> &HashMap<String, ParameterRange<T>> {
        &self.parameters
    }

    /// Categorical parameter option lists, keyed by name.
    pub fn categorical_options(&self) -> &HashMap<String, Vec<String>> {
        &self.categorical_parameters
    }

    /// Declared parameter dependencies.
    pub fn dependencies(&self) -> &[ParameterDependency] {
        &self.dependencies
    }

    /// Declared cross-parameter constraints.
    pub fn constraints(&self) -> &[HyperparameterConstraint<T>] {
        &self.constraints
    }

    /// Whether the space declares no parameters at all.
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty() && self.categorical_parameters.is_empty()
    }

    /// Add a continuous parameter to the search space
    pub fn add_parameter(&mut self, name: String, range: ParameterRange<T>) {
        self.parameters.insert(name, range);
    }

    /// Add a categorical parameter to the search space
    pub fn add_categorical_parameter(&mut self, name: String, categories: Vec<String>) {
        self.categorical_parameters.insert(name, categories);
    }

    /// Add a dependency between parameters
    pub fn add_dependency(&mut self, dependency: ParameterDependency) {
        self.dependencies.push(dependency);
    }

    /// Add a constraint on parameter combinations
    pub fn add_constraint(&mut self, constraint: HyperparameterConstraint<T>) {
        self.constraints.push(constraint);
    }

    /// Validate a configuration against the search space
    pub fn validate_configuration(&self, config: &HyperparameterConfiguration<T>) -> bool {
        // Check parameter ranges
        for (name, value) in &config.parameters {
            if let Some(range) = self.parameters.get(name) {
                if *value < range.min_value || *value > range.max_value {
                    return false;
                }
            }
        }

        // Check categorical parameters
        for (name, value) in &config.categorical_parameters {
            if let Some(categories) = self.categorical_parameters.get(name) {
                if !categories.contains(value) {
                    return false;
                }
            }
        }

        // Check constraints
        for constraint in &self.constraints {
            if !self.check_constraint(constraint, config) {
                return false;
            }
        }

        true
    }

    fn check_constraint(
        &self,
        constraint: &HyperparameterConstraint<T>,
        config: &HyperparameterConfiguration<T>,
    ) -> bool {
        match &constraint.constraint_type {
            ConstraintType::Linear {
                coefficients,
                bound,
            } => {
                let mut sum = T::zero();
                for (i, param_name) in constraint.parameters.iter().enumerate() {
                    if let Some(value) = config.parameters.get(param_name) {
                        sum = sum + coefficients[i] * *value;
                    }
                }
                sum <= *bound
            }
            ConstraintType::MutualExclusion => {
                let active_count = constraint
                    .parameters
                    .iter()
                    .filter(|param| config.parameters.contains_key(*param))
                    .count();
                active_count <= 1
            }
            ConstraintType::Ordering => {
                let mut values = Vec::new();
                for param_name in &constraint.parameters {
                    if let Some(value) = config.parameters.get(param_name) {
                        values.push(*value);
                    }
                }

                for i in 1..values.len() {
                    if values[i - 1] > values[i] {
                        return false;
                    }
                }
                true
            }
            ConstraintType::NonLinear { .. } => {
                // Custom constraint functions would be implemented here
                true
            }
        }
    }
}

impl<T: Float> Default for HyperparameterSpace<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + fmt::Debug> fmt::Display for HyperparameterConfiguration<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Configuration: {}", self.id)?;
        for (name, value) in &self.parameters {
            writeln!(f, "  {}: {:?}", name, value)?;
        }
        for (name, value) in &self.categorical_parameters {
            writeln!(f, "  {}: {}", name, value)?;
        }
        if let Some(score) = self.score {
            writeln!(f, "  Score: {:?}", score)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
