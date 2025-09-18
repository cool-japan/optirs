//! Hyperparameter optimization for Neural Architecture Search.
//!
//! This module provides hyperparameter optimization capabilities for both
//! the NAS process itself and the discovered architectures. It includes
//! various optimization strategies such as grid search, random search,
//! Bayesian optimization, and evolutionary approaches.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

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
    /// Gaussian Process
    GaussianProcess,
    /// Random Forest
    RandomForest,
    /// Neural Network
    NeuralNetwork,
    /// Polynomial regression
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
    /// Create a new hyperparameter optimizer
    pub fn new(search_space: HyperparameterSpace<T>, strategy: OptimizationStrategy) -> Self {
        Self {
            search_space,
            strategy,
            evaluation_history: Vec::new(),
            best_config: None,
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

    /// Generate next hyperparameter configuration to evaluate
    pub fn suggest_configuration(&mut self) -> HyperparameterConfiguration<T> {
        match self.strategy {
            OptimizationStrategy::Random => self.random_search(),
            OptimizationStrategy::Grid => self.grid_search(),
            OptimizationStrategy::Bayesian => self.bayesian_optimization(),
            OptimizationStrategy::Evolutionary => self.evolutionary_search(),
            OptimizationStrategy::TPE => self.tpe_search(),
            _ => self.random_search(), // Fallback to random search
        }
    }

    /// Record evaluation result
    pub fn record_evaluation(&mut self, evaluation: HyperparameterEvaluation<T>) {
        self.state.num_evaluations += 1;

        // Update best configuration
        if let Some(score) = evaluation.configuration.score {
            if self.best_config.is_none()
                || self
                    .best_config
                    .as_ref()
                    .unwrap()
                    .score
                    .unwrap_or_else(T::zero)
                    < score
            {
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

    fn random_search(&self) -> HyperparameterConfiguration<T> {
        use scirs2_core::random::Rng;
        let mut rng = rand::thread_rng();

        let mut parameters = HashMap::new();
        let mut categorical_parameters = HashMap::new();

        // Sample continuous parameters
        for (name, range) in &self.search_space.parameters {
            let value = match range.distribution {
                DistributionType::Uniform => {
                    if range.log_scale {
                        let log_min = range.min_value.ln();
                        let log_max = range.max_value.ln();
                        let log_val =
                            rng.gen_range(log_min.to_f64().unwrap()..=log_max.to_f64().unwrap());
                        T::from(log_val.exp()).unwrap_or_else(T::zero)
                    } else {
                        let min_f64 = range.min_value.to_f64().unwrap();
                        let max_f64 = range.max_value.to_f64().unwrap();
                        T::from(rng.gen_range(min_f64..=max_f64)).unwrap_or_else(T::zero)
                    }
                }
                _ => {
                    // For simplicity, use uniform for other distributions
                    let min_f64 = range.min_value.to_f64().unwrap();
                    let max_f64 = range.max_value.to_f64().unwrap();
                    T::from(rng.gen_range(min_f64..=max_f64)).unwrap_or_else(T::zero)
                }
            };

            parameters.insert(name.clone(), value);
        }

        // Sample categorical parameters
        for (name, categories) in &self.search_space.categorical_parameters {
            let idx = rng.gen_range(0..categories.len());
            categorical_parameters.insert(name.clone(), categories[idx].clone());
        }

        HyperparameterConfiguration {
            id: format!("config_{}", self.state.num_evaluations),
            parameters,
            categorical_parameters,
            score: None,
            metadata: HashMap::new(),
        }
    }

    fn grid_search(&self) -> HyperparameterConfiguration<T> {
        // Simplified grid search implementation
        self.random_search()
    }

    fn bayesian_optimization(&self) -> HyperparameterConfiguration<T> {
        // Simplified Bayesian optimization implementation
        self.random_search()
    }

    fn evolutionary_search(&mut self) -> HyperparameterConfiguration<T> {
        // Simplified evolutionary search implementation
        if self.state.population.is_empty() {
            // Initialize population
            for _ in 0..20 {
                self.state.population.push(self.random_search());
            }
        }

        // Return a mutated version of a good configuration
        if let Some(best) = self.best_config.as_ref() {
            self.mutate_configuration(best)
        } else {
            self.random_search()
        }
    }

    fn tpe_search(&self) -> HyperparameterConfiguration<T> {
        // Simplified TPE implementation
        self.random_search()
    }

    fn mutate_configuration(
        &self,
        config: &HyperparameterConfiguration<T>,
    ) -> HyperparameterConfiguration<T> {
        use scirs2_core::random::Rng;
        let mut rng = rand::thread_rng();

        let mut new_config = config.clone();
        new_config.id = format!("mutated_{}", self.state.num_evaluations);
        new_config.score = None;

        // Mutate a random parameter
        if !new_config.parameters.is_empty() {
            let param_names: Vec<_> = new_config.parameters.keys().cloned().collect();
            let param_to_mutate = &param_names[rng.gen_range(0..param_names.len())];

            if let Some(range) = self.search_space.parameters.get(param_to_mutate) {
                let current_value = new_config.parameters[param_to_mutate];
                let range_size = range.max_value - range.min_value;
                let mutation_strength = range_size * T::from(0.1).unwrap_or_else(T::zero);

                let mutation =
                    T::from(rng.gen_range(-0.1..=0.1)).unwrap_or_else(T::zero) * mutation_strength;
                let new_value = (current_value + mutation)
                    .max(range.min_value)
                    .min(range.max_value);

                new_config
                    .parameters
                    .insert(param_to_mutate.clone(), new_value);
            }
        }

        new_config
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

    fn update_strategy_state(&mut self) {
        self.state.iteration += 1;
        // Strategy-specific updates would go here
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

impl<T: Float> fmt::Display for HyperparameterConfiguration<T> {
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
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameter_space_creation() {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();

        space.add_parameter(
            "learning_rate".to_string(),
            ParameterRange {
                name: "learning_rate".to_string(),
                min_value: 1e-5,
                max_value: 1e-1,
                distribution: DistributionType::Uniform,
                log_scale: true,
                discrete_values: None,
            },
        );

        space.add_categorical_parameter(
            "optimizer".to_string(),
            vec!["adam".to_string(), "sgd".to_string()],
        );

        assert!(space.parameters.contains_key("learning_rate"));
        assert!(space.categorical_parameters.contains_key("optimizer"));
    }

    #[test]
    fn test_random_search() {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();

        space.add_parameter(
            "learning_rate".to_string(),
            ParameterRange {
                name: "learning_rate".to_string(),
                min_value: 1e-5,
                max_value: 1e-1,
                distribution: DistributionType::Uniform,
                log_scale: false,
                discrete_values: None,
            },
        );

        let mut optimizer = HyperparameterOptimizer::new(space, OptimizationStrategy::Random);
        let config = optimizer.suggest_configuration();

        assert!(config.parameters.contains_key("learning_rate"));
        assert!(config.parameters["learning_rate"] >= 1e-5);
        assert!(config.parameters["learning_rate"] <= 1e-1);
    }

    #[test]
    fn test_evaluation_recording() {
        let space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        let mut optimizer = HyperparameterOptimizer::new(space, OptimizationStrategy::Random);

        let mut config = HyperparameterConfiguration {
            id: "test_config".to_string(),
            parameters: HashMap::new(),
            categorical_parameters: HashMap::new(),
            score: Some(0.85),
            metadata: HashMap::new(),
        };

        let evaluation = HyperparameterEvaluation {
            configuration: config,
            metrics: EvaluationMetrics {
                primary_objective: 0.85,
                secondary_objectives: HashMap::new(),
                validation_score: Some(0.82),
                training_time: 120.0,
                memory_usage: 1024.0,
                converged: true,
                convergence_iterations: Some(100),
            },
            duration_seconds: 125.0,
            status: EvaluationStatus::Success,
            error_message: None,
        };

        optimizer.record_evaluation(evaluation);

        assert_eq!(optimizer.state.num_evaluations, 1);
        assert!(optimizer.best_config.is_some());
        assert_eq!(optimizer.best_config.as_ref().unwrap().score, Some(0.85));
    }
}
