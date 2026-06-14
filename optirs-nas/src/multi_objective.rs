use std::fmt::Debug;
// Multi-objective optimization for neural architecture search
//
// Implements various multi-objective optimization algorithms including NSGA-II, NSGA-III,
// MOEA/D, and other state-of-the-art algorithms for finding Pareto-optimal optimizer architectures.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::random::Rng;
use scirs2_core::RngExt;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::{OptimError, Result};
use crate::nas_engine::{
    ConstraintHandlingMethod, DiversityStrategy, MultiObjectiveAlgorithm, MultiObjectiveConfig,
    ObjectiveConfig, ObjectivePriority, ObjectiveType, OptimizationDirection,
    OptimizerArchitecture, SearchResult, UserPreferences,
};
use crate::EvaluationMetric;

/// Base trait for multi-objective optimizers
pub trait MultiObjectiveOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Initialize the optimizer
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()>;

    /// Update Pareto front with new results
    fn update_pareto_front(&mut self, results: &[SearchResult<T>]) -> Result<ParetoFront<T>>;

    /// Get current Pareto front
    fn get_pareto_front(&self) -> &ParetoFront<T>;

    /// Select next candidates for evaluation
    fn select_candidates(
        &mut self,
        population: &[OptimizerArchitecture<T>],
        objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get optimization statistics
    fn get_statistics(&self) -> MultiObjectiveStatistics<T>;
}

/// Pareto front representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFront<T: Float + Debug + Send + Sync + 'static> {
    /// Pareto-optimal solutions
    pub solutions: Vec<ParetoSolution<T>>,

    /// Objective space bounds
    pub objective_bounds: ObjectiveBounds<T>,

    /// Front metrics
    pub metrics: FrontMetrics<T>,

    /// Generation when last updated
    pub generation: usize,

    /// Timestamp of last update
    pub last_updated: std::time::SystemTime,
}

/// Individual Pareto-optimal solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution<T: Float + Debug + Send + Sync + 'static> {
    /// Optimizer architecture
    pub architecture: OptimizerArchitecture<T>,

    /// Objective values
    pub objectives: Vec<T>,

    /// Constraint violations (if any)
    pub constraint_violations: Vec<T>,

    /// Dominance rank
    pub rank: usize,

    /// Crowding distance
    pub crowding_distance: T,

    /// Solution metadata
    pub metadata: SolutionMetadata,
}

/// Solution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionMetadata {
    /// Solution ID
    pub id: String,

    /// Generation when found
    pub generation: usize,

    /// Evaluation count when found
    pub evaluation_count: usize,

    /// Parent solutions (if offspring)
    pub parents: Vec<String>,

    /// Creation method
    pub creation_method: CreationMethod,
}

/// Solution creation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreationMethod {
    RandomGeneration,
    Crossover,
    Mutation,
    LocalSearch,
    Repair,
    Custom,
}

/// Objective space bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveBounds<T: Float + Debug + Send + Sync + 'static> {
    /// Minimum values for each objective
    pub min_values: Vec<T>,

    /// Maximum values for each objective
    pub max_values: Vec<T>,

    /// Ideal point
    pub ideal_point: Vec<T>,

    /// Nadir point
    pub nadir_point: Vec<T>,
}

/// Front quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Hypervolume
    pub hypervolume: T,

    /// Spread (diversity measure)
    pub spread: T,

    /// Spacing (uniformity measure)
    pub spacing: T,

    /// Convergence measure
    pub convergence: T,

    /// Number of non-dominated solutions
    pub num_solutions: usize,

    /// Coverage metrics
    pub coverage: CoverageMetrics<T>,
}

/// Coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Coverage of objective space
    pub objective_space_coverage: T,

    /// Distance from reference front
    pub reference_distance: T,

    /// Epsilon dominance measure
    pub epsilon_dominance: T,
}

/// Multi-objective optimization statistics
#[derive(Debug, Clone)]
pub struct MultiObjectiveStatistics<T: Float + Debug + Send + Sync + 'static> {
    /// Current generation
    pub generation: usize,

    /// Total evaluations
    pub total_evaluations: usize,

    /// Pareto front size
    pub pareto_front_size: usize,

    /// Best hypervolume achieved
    pub best_hypervolume: T,

    /// Convergence history
    pub convergence_history: Vec<T>,

    /// Diversity history
    pub diversity_history: Vec<T>,

    /// Algorithm-specific metrics
    pub algorithm_metrics: HashMap<String, T>,
}

/// NSGA-II implementation
pub struct NSGA2<T: Float + Debug + Send + Sync + 'static> {
    /// Algorithm configuration
    config: MultiObjectiveConfig<T>,

    /// Current population
    population: Vec<Individual<T>>,

    /// Current Pareto front
    pareto_front: ParetoFront<T>,

    /// Generation counter
    generation: usize,

    /// Statistics
    statistics: MultiObjectiveStatistics<T>,

    /// Population size
    population_size: usize,

    /// Crossover probability
    crossover_prob: f64,

    /// Mutation probability
    mutation_prob: f64,

    /// Random number generator
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

/// Individual in the population
#[derive(Debug, Clone)]
pub struct Individual<T: Float + Debug + Send + Sync + 'static> {
    /// Architecture
    pub architecture: OptimizerArchitecture<T>,

    /// Objective values
    pub objectives: Vec<T>,

    /// Constraint violations
    pub constraints: Vec<T>,

    /// Dominance rank
    pub rank: usize,

    /// Crowding distance
    pub crowding_distance: T,

    /// Fitness value (for single-objective algorithms)
    pub fitness: T,

    /// Individual ID
    pub id: String,
}

/// NSGA-III implementation
pub struct NSGA3<T: Float + Debug + Send + Sync + 'static> {
    /// Base NSGA-II functionality
    base: NSGA2<T>,

    /// Reference directions
    reference_directions: Vec<Array1<T>>,

    /// Association count for each reference direction
    association_count: Vec<usize>,

    /// Niche count for each reference direction
    niche_count: Vec<usize>,
}

/// MOEA/D implementation
pub struct MOEADOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Algorithm configuration
    config: MultiObjectiveConfig<T>,

    /// Weight vectors
    weight_vectors: Vec<Array1<T>>,

    /// Current population
    population: Vec<Individual<T>>,

    /// Neighbor indices for each subproblem
    neighbors: Vec<Vec<usize>>,

    /// Current Pareto front
    pareto_front: ParetoFront<T>,

    /// Ideal point
    ideal_point: Vec<T>,

    /// Decomposition method
    decomposition: DecompositionMethod,

    /// Neighborhood size
    neighborhood_size: usize,

    /// Generation counter
    generation: usize,

    /// Statistics
    statistics: MultiObjectiveStatistics<T>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> MOEADOptimizer<T> {
    pub fn new(config: MultiObjectiveConfig<T>) -> Result<Self> {
        let _population_size = 100; // Default population size
        let neighborhood_size = 20; // Default neighborhood size

        Ok(Self {
            config,
            weight_vectors: Vec::new(),
            population: Vec::new(),
            neighbors: Vec::new(),
            pareto_front: ParetoFront::default(),
            ideal_point: Vec::new(),
            decomposition: DecompositionMethod::WeightedSum,
            neighborhood_size,
            generation: 0,
            statistics: MultiObjectiveStatistics {
                generation: 0,
                total_evaluations: 0,
                pareto_front_size: 0,
                best_hypervolume: T::zero(),
                convergence_history: Vec::new(),
                diversity_history: Vec::new(),
                algorithm_metrics: HashMap::new(),
            },
        })
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + PartialOrd
            + std::iter::Sum,
    > MultiObjectiveOptimizer<T> for MOEADOptimizer<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        Ok(())
    }

    fn update_pareto_front(
        &mut self,
        _new_solutions: &[SearchResult<T>],
    ) -> Result<ParetoFront<T>> {
        Ok(self.pareto_front.clone())
    }

    fn get_pareto_front(&self) -> &ParetoFront<T> {
        &self.pareto_front
    }

    fn select_candidates(
        &mut self,
        _population: &[OptimizerArchitecture<T>],
        _objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "MOEA/D"
    }

    fn get_statistics(&self) -> MultiObjectiveStatistics<T> {
        self.statistics.clone()
    }
}

/// Decomposition methods for MOEA/D
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionMethod {
    /// Weighted sum
    WeightedSum,

    /// Tchebycheff
    Tchebycheff,

    /// Penalty-based boundary intersection
    PBI,

    /// Achievement scalarizing function
    ASF,
}

/// Weighted sum approach
pub struct WeightedSum<T: Float + Debug + Send + Sync + 'static> {
    /// Objective weights
    weights: Vec<T>,

    /// Objective configurations (direction, type, ...) used for
    /// scalarization and dominance computations.
    objectives: Vec<ObjectiveConfig<T>>,

    /// Current best solution
    best_solution: Option<Individual<T>>,

    /// Statistics
    statistics: MultiObjectiveStatistics<T>,

    /// Maintained non-dominated set (Pareto front)
    pareto_front: ParetoFront<T>,

    /// Generation counter
    generation: usize,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> WeightedSum<T> {
    pub fn new(objectives: &[ObjectiveConfig<T>]) -> Result<Self> {
        let weights = objectives.iter().map(|obj| obj.weight).collect();
        Ok(Self {
            weights,
            objectives: objectives.to_vec(),
            best_solution: None,
            statistics: MultiObjectiveStatistics::default(),
            pareto_front: ParetoFront::default(),
            generation: 0,
        })
    }

    /// Return the best scalarized solution observed by the most recent
    /// [`select_candidates`](MultiObjectiveOptimizer::select_candidates) call,
    /// if any.
    pub fn best_solution(&self) -> Option<&Individual<T>> {
        self.best_solution.as_ref()
    }

    /// Extract the objective vector for a search result, mapping each
    /// configured [`ObjectiveType`] onto the corresponding
    /// [`EvaluationMetric`]. Mirrors the mapping used by the NSGA-II
    /// implementation so that the two optimizers agree on objective values.
    fn extract_objectives(&self, result: &SearchResult<T>) -> Vec<T> {
        let mut objectives = Vec::with_capacity(self.objectives.len());
        for obj_config in &self.objectives {
            let metric = match obj_config.objective_type {
                ObjectiveType::Accuracy => EvaluationMetric::Accuracy,
                ObjectiveType::Loss => EvaluationMetric::FinalPerformance,
                ObjectiveType::TrainingTime => EvaluationMetric::TrainingTime,
                ObjectiveType::InferenceTime => EvaluationMetric::ComputationTime,
                ObjectiveType::MemoryUsage => EvaluationMetric::MemoryUsage,
                ObjectiveType::EnergyConsumption => EvaluationMetric::ComputationTime,
                ObjectiveType::ModelSize => EvaluationMetric::MemoryUsage,
                ObjectiveType::Performance => EvaluationMetric::FinalPerformance,
                ObjectiveType::Efficiency => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Robustness => EvaluationMetric::Robustness,
                ObjectiveType::Interpretability => EvaluationMetric::FinalPerformance,
                ObjectiveType::Fairness => EvaluationMetric::FinalPerformance,
                ObjectiveType::Privacy => EvaluationMetric::FinalPerformance,
                ObjectiveType::Sustainability => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Cost => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Custom(_) => EvaluationMetric::FinalPerformance,
            };

            let value = result
                .evaluation_results
                .metric_scores
                .get(&metric)
                .cloned()
                .unwrap_or(T::zero());

            objectives.push(value);
        }
        objectives
    }

    /// Determine whether objective vector `a` Pareto-dominates `b`.
    ///
    /// Uses the same convention as NSGA-II's `dominance_relation`: `a`
    /// dominates `b` when `a` improves at least one objective and worsens
    /// none, honoring each objective's optimization direction.
    fn dominates(&self, a: &[T], b: &[T]) -> bool {
        let mut a_strictly_better = false;

        for (k, obj_config) in self.objectives.iter().enumerate() {
            if k >= a.len() || k >= b.len() {
                break;
            }
            let val_a = a[k];
            let val_b = b[k];

            match obj_config.direction {
                OptimizationDirection::Minimize => {
                    if val_a > val_b {
                        return false;
                    } else if val_a < val_b {
                        a_strictly_better = true;
                    }
                }
                OptimizationDirection::Maximize => {
                    if val_a < val_b {
                        return false;
                    } else if val_a > val_b {
                        a_strictly_better = true;
                    }
                }
            }
        }

        a_strictly_better
    }

    /// Compute the weighted-sum scalarization (as a cost to be **minimized**)
    /// for a single objective vector. Maximize objectives are negated so that
    /// lower scalarized values are always preferred.
    fn scalarize(&self, objectives: &[T], normalized_weights: &[T]) -> T {
        let mut score = T::zero();
        for (k, &weight) in normalized_weights.iter().enumerate() {
            if k >= objectives.len() {
                break;
            }
            let directed = match self.objectives.get(k).map(|c| &c.direction) {
                Some(OptimizationDirection::Maximize) => -objectives[k],
                _ => objectives[k],
            };
            score = score + weight * directed;
        }
        score
    }

    /// Return weights normalized to sum to one. Falls back to uniform weights
    /// when the configured weights are degenerate (empty or non-positive sum).
    fn normalized_weights(&self) -> Vec<T> {
        let n = self.weights.len();
        if n == 0 {
            return Vec::new();
        }

        let mut sum = T::zero();
        for &w in &self.weights {
            if w > T::zero() {
                sum = sum + w;
            }
        }

        if sum > T::zero() {
            self.weights
                .iter()
                .map(|&w| if w > T::zero() { w / sum } else { T::zero() })
                .collect()
        } else {
            let uniform = T::one() / T::from(n).unwrap_or_else(T::one);
            vec![uniform; n]
        }
    }

    /// Rebuild the objective-space bounds and basic front metrics for the
    /// maintained Pareto front. Mirrors NSGA-II's bookkeeping so downstream
    /// consumers observe a consistently-populated [`ParetoFront`].
    fn refresh_front_metrics(&mut self) {
        if self.pareto_front.solutions.is_empty() {
            self.pareto_front.objective_bounds = ObjectiveBounds {
                min_values: Vec::new(),
                max_values: Vec::new(),
                ideal_point: Vec::new(),
                nadir_point: Vec::new(),
            };
            self.pareto_front.metrics.num_solutions = 0;
            self.statistics.pareto_front_size = 0;
            return;
        }

        let num_objectives = self.pareto_front.solutions[0].objectives.len();
        let mut min_values = vec![T::infinity(); num_objectives];
        let mut max_values = vec![T::neg_infinity(); num_objectives];

        for solution in &self.pareto_front.solutions {
            for (i, &obj_val) in solution.objectives.iter().enumerate() {
                if i >= num_objectives {
                    break;
                }
                if obj_val < min_values[i] {
                    min_values[i] = obj_val;
                }
                if obj_val > max_values[i] {
                    max_values[i] = obj_val;
                }
            }
        }

        self.pareto_front.objective_bounds = ObjectiveBounds {
            min_values: min_values.clone(),
            max_values: max_values.clone(),
            ideal_point: min_values,
            nadir_point: max_values,
        };

        let count = self.pareto_front.solutions.len();
        self.pareto_front.metrics.num_solutions = count;
        self.statistics.pareto_front_size = count;
    }
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + PartialOrd
            + std::iter::Sum,
    > MultiObjectiveOptimizer<T> for WeightedSum<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        Ok(())
    }

    fn update_pareto_front(&mut self, new_solutions: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        // No new information: return the current front unchanged.
        if new_solutions.is_empty() {
            return Ok(self.pareto_front.clone());
        }

        self.generation += 1;
        self.statistics.total_evaluations += new_solutions.len();

        // Build the candidate pool from the currently-maintained
        // non-dominated solutions plus the freshly-evaluated results.
        // Each candidate is (objective_vector, ParetoSolution).
        let mut candidates: Vec<(Vec<T>, ParetoSolution<T>)> =
            Vec::with_capacity(self.pareto_front.solutions.len() + new_solutions.len());

        for solution in &self.pareto_front.solutions {
            candidates.push((solution.objectives.clone(), solution.clone()));
        }

        for result in new_solutions {
            let objectives = self.extract_objectives(result);
            let solution = ParetoSolution {
                architecture: result.architecture.clone(),
                objectives: objectives.clone(),
                constraint_violations: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                metadata: SolutionMetadata {
                    id: result.architecture.architecture_id.clone(),
                    generation: self.generation,
                    evaluation_count: self.statistics.total_evaluations,
                    parents: Vec::new(),
                    creation_method: CreationMethod::RandomGeneration,
                },
            };
            candidates.push((objectives, solution));
        }

        // Retain only non-dominated candidates. A candidate survives if no
        // other candidate dominates it. Identical objective vectors do not
        // dominate one another, so duplicates are de-duplicated explicitly.
        let mut front: Vec<ParetoSolution<T>> = Vec::new();
        for i in 0..candidates.len() {
            let mut dominated = false;
            for j in 0..candidates.len() {
                if i != j && self.dominates(&candidates[j].0, &candidates[i].0) {
                    dominated = true;
                    break;
                }
            }

            if dominated {
                continue;
            }

            // Skip exact objective-vector duplicates already kept.
            let is_duplicate = front
                .iter()
                .any(|existing| existing.objectives == candidates[i].0);
            if !is_duplicate {
                front.push(candidates[i].1.clone());
            }
        }

        self.pareto_front.solutions = front;
        self.pareto_front.generation = self.generation;
        self.pareto_front.last_updated = std::time::SystemTime::now();
        self.refresh_front_metrics();

        Ok(self.pareto_front.clone())
    }

    fn get_pareto_front(&self) -> &ParetoFront<T> {
        &self.pareto_front
    }

    fn select_candidates(
        &mut self,
        population: &[OptimizerArchitecture<T>],
        objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        if population.is_empty() {
            return Ok(Vec::new());
        }

        let num_objectives = self.objectives.len();
        let normalized_weights = self.normalized_weights();

        // Score every architecture by its weighted-sum scalarization. The
        // provided `objectives` slice is treated as a flattened, row-major
        // matrix of `population.len() x num_objectives` objective values; if
        // it is too short the missing entries are treated as zero so that
        // selection still produces a deterministic ranking.
        let mut scored: Vec<(usize, T)> = Vec::with_capacity(population.len());
        for (idx, _architecture) in population.iter().enumerate() {
            let obj_vector: Vec<T> = if num_objectives > 0 {
                (0..num_objectives)
                    .map(|k| {
                        let flat = idx * num_objectives + k;
                        objectives.get(flat).cloned().unwrap_or(T::zero())
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let score = self.scalarize(&obj_vector, &normalized_weights);
            scored.push((idx, score));
        }

        // Rank by scalarized cost ascending (best first).
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Record the single best-scoring architecture for later reference.
        if let Some(&(best_idx, best_score)) = scored.first() {
            let best_objectives: Vec<T> = if num_objectives > 0 {
                (0..num_objectives)
                    .map(|k| {
                        let flat = best_idx * num_objectives + k;
                        objectives.get(flat).cloned().unwrap_or(T::zero())
                    })
                    .collect()
            } else {
                Vec::new()
            };

            self.best_solution = Some(Individual {
                architecture: population[best_idx].clone(),
                objectives: best_objectives,
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: best_score,
                id: population[best_idx].architecture_id.clone(),
            });
        }

        // Select the better half of the population (at least one).
        let selection_count = (population.len() / 2).max(1).min(population.len());

        let selected = scored
            .iter()
            .take(selection_count)
            .map(|&(idx, _)| population[idx].clone())
            .collect();

        Ok(selected)
    }

    fn name(&self) -> &str {
        "WeightedSum"
    }

    fn get_statistics(&self) -> MultiObjectiveStatistics<T> {
        self.statistics.clone()
    }
}

/// SMS-EMOA implementation
pub struct SmsEmoa<T: Float + Debug + Send + Sync + 'static> {
    /// Base population
    population: Vec<Individual<T>>,

    /// Pareto front
    pareto_front: ParetoFront<T>,

    /// Hypervolume calculator
    hypervolume_calculator: HypervolumeCalculator<T>,

    /// Reference point for hypervolume
    reference_point: Vec<T>,

    /// Generation counter
    generation: usize,

    /// Statistics
    statistics: MultiObjectiveStatistics<T>,
}

/// Hypervolume calculator
#[derive(Debug)]
pub struct HypervolumeCalculator<T: Float + Debug + Send + Sync + 'static> {
    /// Calculation method
    method: HypervolumeMethod,

    /// Reference point
    reference_point: Vec<T>,

    /// Cached hypervolumes
    cache: HashMap<String, T>,
}

/// Hypervolume calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypervolumeMethod {
    /// Walking Fish Group algorithm
    WFG,

    /// Quick hypervolume
    Quick,

    /// Hypervolume by slicing objectives
    HSO,

    /// Monte Carlo estimation
    MonteCarlo,
}

/// Indicator-Based Evolutionary Algorithm (IBEA)
pub struct IBEA<T: Float + Debug + Send + Sync + 'static> {
    /// Population
    population: Vec<Individual<T>>,

    /// Fitness values based on indicators
    indicator_fitness: Vec<T>,

    /// Quality indicator
    quality_indicator: QualityIndicator<T>,

    /// Scaling factor
    scaling_factor: T,

    /// Pareto front
    pareto_front: ParetoFront<T>,

    /// Generation counter
    generation: usize,

    /// Statistics
    statistics: MultiObjectiveStatistics<T>,
}

/// Quality indicators for IBEA
#[derive(Debug)]
pub struct QualityIndicator<T: Float + Debug + Send + Sync + 'static> {
    /// Indicator type
    indicator_type: IndicatorType,

    /// Indicator parameters
    parameters: HashMap<String, T>,
}

/// Types of quality indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorType {
    /// Additive epsilon indicator
    AdditiveEpsilon,

    /// Multiplicative epsilon indicator
    MultiplicativeEpsilon,

    /// Hypervolume contribution
    HypervolumeContribution,

    /// R2 indicator
    R2,
}

/// Preference handling for interactive optimization
pub struct PreferenceHandler<T: Float + Debug + Send + Sync + 'static> {
    /// User preferences
    preferences: UserPreferences<T>,

    /// Preference articulation method
    articulation_method: ArticulationMethod,

    /// Decision maker utilities
    utilities: Vec<T>,

    /// Preference history
    preference_history: Vec<PreferenceSnapshot<T>>,
}

/// Preference articulation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationMethod {
    /// A priori (before optimization)
    APriori,

    /// Interactive (during optimization)
    Interactive,

    /// A posteriori (after optimization)
    APosteriori,

    /// Progressive (evolving preferences)
    Progressive,
}

/// Preference snapshot
#[derive(Debug, Clone)]
pub struct PreferenceSnapshot<T: Float + Debug + Send + Sync + 'static> {
    /// Timestamp
    timestamp: std::time::SystemTime,

    /// Preference values
    preferences: HashMap<String, T>,

    /// Confidence levels
    confidence: HashMap<String, T>,

    /// Context information
    context: String,
}

/// Constraint handler for constrained multi-objective optimization
pub struct ConstraintHandler<T: Float + Debug + Send + Sync + 'static> {
    /// Constraint handling method
    method: ConstraintHandlingMethod,

    /// Constraint functions
    constraints: Vec<ConstraintFunction<T>>,

    /// Constraint tolerance
    tolerance: T,

    /// Penalty parameters
    penalty_parameters: PenaltyParameters<T>,
}

/// Constraint function
#[derive(Debug)]
pub struct ConstraintFunction<T: Float + Debug + Send + Sync + 'static> {
    /// Function type
    function_type: ConstraintType,

    /// Function parameters
    parameters: HashMap<String, T>,

    /// Constraint bound
    bound: T,

    /// Constraint direction (<=, >=, )
    direction: ConstraintDirection,
}

/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Linear constraint
    Linear,

    /// Quadratic constraint
    Quadratic,

    /// Nonlinear constraint
    Nonlinear,

    /// Resource constraint
    Resource,

    /// Performance constraint
    Performance,

    /// Custom constraint
    Custom,
}

/// Constraint directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintDirection {
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equal,
}

/// Penalty parameters for constraint handling
#[derive(Debug, Clone)]
pub struct PenaltyParameters<T: Float + Debug + Send + Sync + 'static> {
    /// Static penalty weight
    static_weight: T,

    /// Dynamic penalty weight
    dynamic_weight: T,

    /// Penalty increase rate
    increase_rate: T,

    /// Penalty function type
    penalty_function: PenaltyFunctionType,
}

/// Penalty function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyFunctionType {
    /// Linear penalty
    Linear,

    /// Quadratic penalty
    Quadratic,

    /// Exponential penalty
    Exponential,

    /// Logarithmic penalty
    Logarithmic,
}

/// Implementation of NSGA-II
impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + PartialOrd
            + std::iter::Sum,
    > NSGA2<T>
{
    /// Create new NSGA-II optimizer
    pub fn new(population_size: usize, crossover_prob: f64, mutation_prob: f64) -> Self {
        Self {
            config: MultiObjectiveConfig::default(),
            population: Vec::new(),
            pareto_front: ParetoFront::new(),
            generation: 0,
            statistics: MultiObjectiveStatistics::default(),
            population_size,
            crossover_prob,
            mutation_prob,
            rng: Random::seed(42),
        }
    }

    /// Initialize population
    fn initialize_population(&mut self) -> Result<()> {
        // Initialize with random architectures
        // This would be replaced with actual architecture generation
        self.population.clear();

        for i in 0..self.population_size {
            let architecture = self.generate_random_architecture()?;
            let individual = Individual {
                architecture,
                objectives: vec![T::zero(); self.config.objectives.len()],
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
                id: format!("ind_{}", i),
            };
            self.population.push(individual);
        }

        Ok(())
    }

    /// Generate random architecture (placeholder)
    fn generate_random_architecture(&self) -> Result<OptimizerArchitecture<T>> {
        // Simplified random architecture generation
        let mut parameters = HashMap::new();
        parameters.insert(
            "learning_rate".to_string(),
            scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero()),
        );
        parameters.insert(
            "beta1".to_string(),
            scirs2_core::numeric::NumCast::from(0.9).unwrap_or_else(|| T::zero()),
        );
        parameters.insert(
            "beta2".to_string(),
            scirs2_core::numeric::NumCast::from(0.999).unwrap_or_else(|| T::zero()),
        );

        Ok(OptimizerArchitecture {
            components: vec!["Adam".to_string()],
            parameters: parameters.clone(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: parameters,
            architecture_id: format!(
                "arch_{}",
                scirs2_core::random::Random::default().random::<u32>()
            ),
        })
    }

    /// Perform non-dominated sorting
    fn non_dominated_sort(&mut self) -> Vec<Vec<usize>> {
        let n = self.population.len();
        let mut fronts = Vec::new();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];

        // First front
        let mut first_front = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dominance = self.dominance_relation(i, j);
                    match dominance {
                        DominanceRelation::Dominates => {
                            dominated_solutions[i].push(j);
                        }
                        DominanceRelation::DominatedBy => {
                            domination_count[i] += 1;
                        }
                        DominanceRelation::NonDominated => {}
                    }
                }
            }

            if domination_count[i] == 0 {
                self.population[i].rank = 0;
                first_front.push(i);
            }
        }

        fronts.push(first_front.clone());

        // Subsequent fronts
        let mut current_front = first_front;
        let mut rank = 0;

        while !current_front.is_empty() {
            let mut next_front = Vec::new();

            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        self.population[j].rank = rank + 1;
                        next_front.push(j);
                    }
                }
            }

            rank += 1;
            current_front = next_front.clone();
            if !next_front.is_empty() {
                fronts.push(next_front);
            }
        }

        fronts
    }

    /// Determine dominance relation between two individuals
    fn dominance_relation(&self, i: usize, j: usize) -> DominanceRelation {
        let ind_i = &self.population[i];
        let ind_j = &self.population[j];

        let mut i_dominates = false;
        let mut j_dominates = false;

        for k in 0..ind_i.objectives.len() {
            let obj_config = &self.config.objectives[k];
            let val_i = ind_i.objectives[k];
            let val_j = ind_j.objectives[k];

            match obj_config.direction {
                OptimizationDirection::Minimize => {
                    if val_i < val_j {
                        i_dominates = true;
                    } else if val_i > val_j {
                        j_dominates = true;
                    }
                }
                OptimizationDirection::Maximize => {
                    if val_i > val_j {
                        i_dominates = true;
                    } else if val_i < val_j {
                        j_dominates = true;
                    }
                }
            }
        }

        if i_dominates && !j_dominates {
            DominanceRelation::Dominates
        } else if j_dominates && !i_dominates {
            DominanceRelation::DominatedBy
        } else {
            DominanceRelation::NonDominated
        }
    }

    /// Calculate crowding distance
    fn calculate_crowding_distance(&mut self, front: &[usize]) {
        let front_size = front.len();

        // Initialize crowding distance
        for &idx in front {
            self.population[idx].crowding_distance = T::zero();
        }

        if front_size <= 2 {
            // Boundary solutions have infinite crowding distance
            for &idx in front {
                self.population[idx].crowding_distance = T::infinity();
            }
            return;
        }

        let num_objectives = self.config.objectives.len();

        for obj_idx in 0..num_objectives {
            // Sort by objective value
            let mut sorted_front = front.to_vec();
            sorted_front.sort_by(|&a, &b| {
                self.population[a].objectives[obj_idx]
                    .partial_cmp(&self.population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinite distance
            self.population[sorted_front[0]].crowding_distance = T::infinity();
            self.population[sorted_front[front_size - 1]].crowding_distance = T::infinity();

            // Calculate objective range
            let obj_min = self.population[sorted_front[0]].objectives[obj_idx];
            let obj_max = self.population[sorted_front[front_size - 1]].objectives[obj_idx];
            let obj_range = obj_max - obj_min;

            if obj_range > T::zero() {
                // Calculate crowding distance for intermediate points
                for i in 1..front_size - 1 {
                    let idx = sorted_front[i];
                    let prev_obj = self.population[sorted_front[i - 1]].objectives[obj_idx];
                    let next_obj = self.population[sorted_front[i + 1]].objectives[obj_idx];

                    let distance = (next_obj - prev_obj) / obj_range;
                    self.population[idx].crowding_distance =
                        self.population[idx].crowding_distance + distance;
                }
            }
        }
    }

    /// Environmental selection (survival selection)
    fn environmental_selection(
        &mut self,
        combined_population: Vec<Individual<T>>,
    ) -> Vec<Individual<T>> {
        self.population = combined_population;

        // Non-dominated sorting
        let fronts = self.non_dominated_sort();

        let mut new_population = Vec::new();
        let mut front_idx = 0;

        // Add complete fronts
        while front_idx < fronts.len() {
            let front = &fronts[front_idx];

            if new_population.len() + front.len() <= self.population_size {
                // Calculate crowding distance for this front
                self.calculate_crowding_distance(front);

                // Add entire front
                for &idx in front {
                    new_population.push(self.population[idx].clone());
                }
                front_idx += 1;
            } else {
                // Partial front selection based on crowding distance
                self.calculate_crowding_distance(front);

                let mut front_individuals: Vec<_> = front
                    .iter()
                    .map(|&idx| (idx, self.population[idx].crowding_distance))
                    .collect();

                // Sort by crowding distance (descending)
                front_individuals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                let remaining_slots = self.population_size - new_population.len();
                for &(idx, _) in front_individuals.iter().take(remaining_slots) {
                    new_population.push(self.population[idx].clone());
                }
                break;
            }
        }

        new_population
    }

    /// Update Pareto front from current population
    fn update_pareto_front_from_population(&mut self) {
        // Find all non-dominated solutions (rank 0)
        let mut pareto_solutions = Vec::new();

        for individual in &self.population {
            if individual.rank == 0 {
                let solution = ParetoSolution {
                    architecture: individual.architecture.clone(),
                    objectives: individual.objectives.clone(),
                    constraint_violations: individual.constraints.clone(),
                    rank: individual.rank,
                    crowding_distance: individual.crowding_distance,
                    metadata: SolutionMetadata {
                        id: individual.id.clone(),
                        generation: self.generation,
                        evaluation_count: self.statistics.total_evaluations,
                        parents: Vec::new(),
                        creation_method: CreationMethod::RandomGeneration,
                    },
                };
                pareto_solutions.push(solution);
            }
        }

        self.pareto_front.solutions = pareto_solutions;
        self.pareto_front.generation = self.generation;
        self.pareto_front.last_updated = std::time::SystemTime::now();

        // Update objective bounds
        self.update_objective_bounds();

        // Calculate front metrics
        self.calculate_front_metrics();
    }

    fn update_objective_bounds(&mut self) {
        if self.pareto_front.solutions.is_empty() {
            return;
        }

        let num_objectives = self.pareto_front.solutions[0].objectives.len();
        let mut min_values = vec![T::infinity(); num_objectives];
        let mut max_values = vec![T::neg_infinity(); num_objectives];

        for solution in &self.pareto_front.solutions {
            for (i, &obj_val) in solution.objectives.iter().enumerate() {
                if obj_val < min_values[i] {
                    min_values[i] = obj_val;
                }
                if obj_val > max_values[i] {
                    max_values[i] = obj_val;
                }
            }
        }

        self.pareto_front.objective_bounds = ObjectiveBounds {
            min_values: min_values.clone(),
            max_values: max_values.clone(),
            ideal_point: min_values,
            nadir_point: max_values,
        };
    }

    fn calculate_front_metrics(&mut self) {
        // Calculate hypervolume (simplified)
        let hypervolume = self.calculate_hypervolume();

        // Calculate spread (simplified)
        let spread = self.calculate_spread();

        // Calculate spacing (simplified)
        let spacing = self.calculate_spacing();

        self.pareto_front.metrics = FrontMetrics {
            hypervolume,
            spread,
            spacing,
            convergence: T::zero(), // Would be calculated based on reference front
            num_solutions: self.pareto_front.solutions.len(),
            coverage: CoverageMetrics {
                objective_space_coverage: scirs2_core::numeric::NumCast::from(0.5)
                    .unwrap_or_else(|| T::zero()),
                reference_distance: T::zero(),
                epsilon_dominance: T::zero(),
            },
        };

        // Update statistics
        self.statistics.pareto_front_size = self.pareto_front.solutions.len();
        self.statistics.best_hypervolume = hypervolume;
        self.statistics.convergence_history.push(T::zero());
        self.statistics.diversity_history.push(spread);
    }

    fn calculate_hypervolume(&self) -> T {
        // Simplified hypervolume calculation
        // In practice, this would use proper hypervolume algorithms
        if self.pareto_front.solutions.is_empty() {
            return T::zero();
        }

        // Use bounding box approach for simplicity
        let bounds = &self.pareto_front.objective_bounds;
        let mut volume = T::one();

        for i in 0..bounds.max_values.len() {
            let range = bounds.max_values[i] - bounds.min_values[i];
            volume = volume
                * range.max(scirs2_core::numeric::NumCast::from(1e-6).unwrap_or_else(|| T::zero()));
        }

        volume * T::from(self.pareto_front.solutions.len() as f64).expect("unwrap failed")
    }

    fn calculate_spread(&self) -> T {
        if self.pareto_front.solutions.len() < 2 {
            return T::zero();
        }

        // Calculate average distance between consecutive solutions
        let mut total_distance = T::zero();
        let num_objectives = self.pareto_front.solutions[0].objectives.len();

        for i in 0..self.pareto_front.solutions.len() - 1 {
            let mut distance = T::zero();
            for j in 0..num_objectives {
                let diff = self.pareto_front.solutions[i + 1].objectives[j]
                    - self.pareto_front.solutions[i].objectives[j];
                distance = distance + diff * diff;
            }
            total_distance = total_distance + distance.sqrt();
        }

        total_distance / T::from(self.pareto_front.solutions.len() - 1).expect("unwrap failed")
    }

    fn calculate_spacing(&self) -> T {
        if self.pareto_front.solutions.len() < 2 {
            return T::zero();
        }

        // Calculate spacing metric (simplified)
        let mut distances = Vec::new();

        for i in 0..self.pareto_front.solutions.len() {
            let mut min_distance = T::infinity();

            for j in 0..self.pareto_front.solutions.len() {
                if i != j {
                    let mut distance = T::zero();
                    for k in 0..self.pareto_front.solutions[i].objectives.len() {
                        let diff = self.pareto_front.solutions[i].objectives[k]
                            - self.pareto_front.solutions[j].objectives[k];
                        distance = distance + diff.abs();
                    }

                    if distance < min_distance {
                        min_distance = distance;
                    }
                }
            }

            distances.push(min_distance);
        }

        // Calculate mean and standard deviation of distances
        let mean: T =
            distances.iter().cloned().sum::<T>() / T::from(distances.len()).expect("unwrap failed");
        let variance: T = distances
            .iter()
            .map(|&d| (d - mean) * (d - mean))
            .sum::<T>()
            / T::from(distances.len()).expect("unwrap failed");

        variance.sqrt()
    }
}

/// Dominance relation between two solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DominanceRelation {
    Dominates,
    DominatedBy,
    NonDominated,
}

impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + PartialOrd
            + std::iter::Sum,
    > MultiObjectiveOptimizer<T> for NSGA2<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        self.config = config.clone();
        self.initialize_population()?;
        Ok(())
    }

    fn update_pareto_front(&mut self, results: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        // Update population with new results
        for (i, result) in results.iter().enumerate() {
            if i < self.population.len() {
                // Extract objective values from evaluation results
                let mut objectives = Vec::new();
                for obj_config in &self.config.objectives {
                    let metric = match obj_config.objective_type {
                        ObjectiveType::Accuracy => EvaluationMetric::Accuracy,
                        ObjectiveType::Loss => EvaluationMetric::FinalPerformance,
                        ObjectiveType::TrainingTime => EvaluationMetric::TrainingTime,
                        ObjectiveType::InferenceTime => EvaluationMetric::ComputationTime,
                        ObjectiveType::MemoryUsage => EvaluationMetric::MemoryUsage,
                        ObjectiveType::EnergyConsumption => EvaluationMetric::ComputationTime,
                        ObjectiveType::ModelSize => EvaluationMetric::MemoryUsage,
                        ObjectiveType::Performance => EvaluationMetric::FinalPerformance,
                        ObjectiveType::Efficiency => EvaluationMetric::ComputationalEfficiency,
                        ObjectiveType::Robustness => EvaluationMetric::Robustness,
                        ObjectiveType::Interpretability => EvaluationMetric::FinalPerformance,
                        ObjectiveType::Fairness => EvaluationMetric::FinalPerformance,
                        ObjectiveType::Privacy => EvaluationMetric::FinalPerformance,
                        ObjectiveType::Sustainability => EvaluationMetric::ComputationalEfficiency,
                        ObjectiveType::Cost => EvaluationMetric::ComputationalEfficiency,
                        ObjectiveType::Custom(_) => EvaluationMetric::FinalPerformance,
                    };

                    let value = result
                        .evaluation_results
                        .metric_scores
                        .get(&metric)
                        .cloned()
                        .unwrap_or(T::zero());

                    objectives.push(value);
                }

                self.population[i].objectives = objectives;
                self.population[i].architecture = result.architecture.clone();
            }
        }

        self.generation += 1;
        self.statistics.total_evaluations += results.len();

        // Update Pareto front
        self.update_pareto_front_from_population();

        Ok(self.pareto_front.clone())
    }

    fn get_pareto_front(&self) -> &ParetoFront<T> {
        &self.pareto_front
    }

    fn select_candidates(
        &mut self,
        _population: &[OptimizerArchitecture<T>],
        _objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        // Generate new candidates through crossover and mutation
        let mut new_population = Vec::new();

        for _ in 0..self.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(2)?;
            let parent2 = self.tournament_selection(2)?;

            // Crossover
            let mut offspring = if self.rng.gen_range(0.0..1.0) < self.crossover_prob {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };

            // Mutation
            if self.rng.gen_range(0.0..1.0) < self.mutation_prob {
                self.mutate(&mut offspring)?;
            }

            new_population.push(offspring.architecture);
        }

        Ok(new_population)
    }

    fn name(&self) -> &str {
        "NSGA-II"
    }

    fn get_statistics(&self) -> MultiObjectiveStatistics<T> {
        self.statistics.clone()
    }
}

// Implementation of helper methods for NSGA2
impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + PartialOrd
            + std::iter::Sum,
    > NSGA2<T>
{
    /// Map a configured [`ObjectiveType`] onto the [`EvaluationMetric`] used to
    /// read its value from an [`EvaluationResults`]. Kept identical to the
    /// mapping used by [`MultiObjectiveOptimizer::update_pareto_front`] so the
    /// helpers below agree with the main optimization path.
    fn objective_vector_for_result(&self, result: &SearchResult<T>) -> Vec<T> {
        let mut objectives = Vec::with_capacity(self.config.objectives.len());
        for obj_config in &self.config.objectives {
            let metric = match obj_config.objective_type {
                ObjectiveType::Accuracy => EvaluationMetric::Accuracy,
                ObjectiveType::Loss => EvaluationMetric::FinalPerformance,
                ObjectiveType::TrainingTime => EvaluationMetric::TrainingTime,
                ObjectiveType::InferenceTime => EvaluationMetric::ComputationTime,
                ObjectiveType::MemoryUsage => EvaluationMetric::MemoryUsage,
                ObjectiveType::EnergyConsumption => EvaluationMetric::ComputationTime,
                ObjectiveType::ModelSize => EvaluationMetric::MemoryUsage,
                ObjectiveType::Performance => EvaluationMetric::FinalPerformance,
                ObjectiveType::Efficiency => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Robustness => EvaluationMetric::Robustness,
                ObjectiveType::Interpretability => EvaluationMetric::FinalPerformance,
                ObjectiveType::Fairness => EvaluationMetric::FinalPerformance,
                ObjectiveType::Privacy => EvaluationMetric::FinalPerformance,
                ObjectiveType::Sustainability => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Cost => EvaluationMetric::ComputationalEfficiency,
                ObjectiveType::Custom(_) => EvaluationMetric::FinalPerformance,
            };

            let value = result
                .evaluation_results
                .metric_scores
                .get(&metric)
                .cloned()
                .unwrap_or(T::zero());

            objectives.push(value);
        }
        objectives
    }

    /// Load `results` into the population (one individual per result), run the
    /// complete NSGA-II non-dominated sort and crowding-distance assignment,
    /// and return the indices (into `results`) of the best `k` solutions
    /// ordered by ascending rank then descending crowding distance.
    ///
    /// This drives the engine-level multi-objective selection by reusing the
    /// real NSGA-II machinery rather than re-deriving dominance externally.
    pub(crate) fn select_by_rank_and_crowding(
        &mut self,
        results: &[SearchResult<T>],
        k: usize,
    ) -> Vec<usize> {
        if results.is_empty() || k == 0 {
            return Vec::new();
        }

        // Rebuild the population so that index `i` corresponds to `results[i]`.
        self.population = results
            .iter()
            .enumerate()
            .map(|(i, result)| Individual {
                architecture: result.architecture.clone(),
                objectives: self.objective_vector_for_result(result),
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
                id: format!("sel_{}", i),
            })
            .collect();

        // Non-dominated sort assigns `rank` to every individual.
        let fronts = self.non_dominated_sort();

        // Crowding distance within each front.
        for front in &fronts {
            self.calculate_crowding_distance(front);
        }

        let mut order: Vec<usize> = (0..self.population.len()).collect();
        order.sort_by(|&a, &b| {
            let rank_a = self.population[a].rank;
            let rank_b = self.population[b].rank;
            rank_a.cmp(&rank_b).then_with(|| {
                self.population[b]
                    .crowding_distance
                    .partial_cmp(&self.population[a].crowding_distance)
                    .unwrap_or(Ordering::Equal)
            })
        });

        order.truncate(k.min(self.population.len()));
        order
    }

    /// Build a Pareto front from `results` using the complete NSGA-II pipeline:
    /// load one individual per result, run the non-dominated sort to assign
    /// dominance ranks, assign crowding distances within each front, and then
    /// collect the rank-0 (non-dominated) solutions together with objective
    /// bounds and front metrics.
    ///
    /// This is the faithful counterpart to the trait-level
    /// [`MultiObjectiveOptimizer::update_pareto_front`], but it performs the
    /// non-dominated sort that the index-mapped trait method omits, so the
    /// returned front contains exactly the non-dominated solutions.
    pub(crate) fn pareto_front_from_results(
        &mut self,
        results: &[SearchResult<T>],
    ) -> ParetoFront<T> {
        self.population = results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                let id = if result.architecture.architecture_id.is_empty() {
                    format!("ind_{}", i)
                } else {
                    result.architecture.architecture_id.clone()
                };
                Individual {
                    architecture: result.architecture.clone(),
                    objectives: self.objective_vector_for_result(result),
                    constraints: Vec::new(),
                    rank: 0,
                    crowding_distance: T::zero(),
                    fitness: T::zero(),
                    id,
                }
            })
            .collect();

        self.generation += 1;
        self.statistics.total_evaluations += results.len();

        // Assign dominance ranks across the whole population.
        let fronts = self.non_dominated_sort();

        // Assign crowding distances within each front.
        for front in &fronts {
            self.calculate_crowding_distance(front);
        }

        // Collect rank-0 solutions and refresh bounds + metrics.
        self.update_pareto_front_from_population();

        self.pareto_front.clone()
    }

    /// Compute the mean pairwise Euclidean distance between the objective
    /// vectors of `results`, providing a real diversity metric for the
    /// population. Returns `0.0` for fewer than two solutions.
    pub(crate) fn mean_objective_distance(&self, results: &[SearchResult<T>]) -> f64 {
        if results.len() < 2 {
            return 0.0;
        }

        let objective_vectors: Vec<Vec<T>> = results
            .iter()
            .map(|r| self.objective_vector_for_result(r))
            .collect();

        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..objective_vectors.len() {
            for j in (i + 1)..objective_vectors.len() {
                let a = &objective_vectors[i];
                let b = &objective_vectors[j];
                let len = a.len().min(b.len());
                let mut sq_sum = T::zero();
                for d in 0..len {
                    let diff = a[d] - b[d];
                    sq_sum = sq_sum + diff * diff;
                }
                total += sq_sum.sqrt().to_f64().unwrap_or(0.0);
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    fn tournament_selection(&mut self, tournamentsize: usize) -> Result<Individual<T>> {
        if self.population.is_empty() {
            return Err(OptimError::InvalidConfig("Empty population".to_string()));
        }

        let mut best_idx = self.rng.gen_range(0..self.population.len());

        for _ in 1..tournamentsize {
            let idx = self.rng.gen_range(0..self.population.len());

            // Compare based on rank and crowding distance
            if self.population[idx].rank < self.population[best_idx].rank
                || (self.population[idx].rank == self.population[best_idx].rank
                    && self.population[idx].crowding_distance
                        > self.population[best_idx].crowding_distance)
            {
                best_idx = idx;
            }
        }

        Ok(self.population[best_idx].clone())
    }

    fn crossover(
        &mut self,
        parent1: &Individual<T>,
        parent2: &Individual<T>,
    ) -> Result<Individual<T>> {
        // Simplified crossover - in practice would be more sophisticated
        let mut offspring = parent1.clone();
        offspring.id = format!("offspring_{}", self.rng.gen_range(0..u32::MAX));

        // Randomly mix hyperparameters
        for (key, value) in &parent1.architecture.parameters {
            if self.rng.gen_range(0.0..1.0) < 0.5 {
                if let Some(parent2_value) = parent2.architecture.parameters.get(key) {
                    offspring
                        .architecture
                        .parameters
                        .insert(key.clone(), *parent2_value);
                } else {
                    offspring
                        .architecture
                        .parameters
                        .insert(key.clone(), *value);
                }
            } else {
                offspring
                    .architecture
                    .parameters
                    .insert(key.clone(), *value);
            }
        }

        Ok(offspring)
    }

    fn mutate(&mut self, individual: &mut Individual<T>) -> Result<()> {
        // Simplified mutation - in practice would be more sophisticated
        for (_key, value) in individual.architecture.parameters.iter_mut() {
            if self.rng.gen_range(0.0..1.0) < 0.1 {
                // 10% mutation rate per parameter
                let noise = T::from(self.rng.gen_range(-0.05..0.05)).expect("unwrap failed"); // ±5% noise
                *value = *value + noise;
            }
        }

        Ok(())
    }
}

// Default implementations
impl<T: Float + Debug + Default + Send + Sync> Default for ParetoFront<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + Send + Sync> ParetoFront<T> {
    fn new() -> Self {
        Self {
            solutions: Vec::new(),
            objective_bounds: ObjectiveBounds {
                min_values: Vec::new(),
                max_values: Vec::new(),
                ideal_point: Vec::new(),
                nadir_point: Vec::new(),
            },
            metrics: FrontMetrics {
                hypervolume: T::zero(),
                spread: T::zero(),
                spacing: T::zero(),
                convergence: T::zero(),
                num_solutions: 0,
                coverage: CoverageMetrics {
                    objective_space_coverage: T::zero(),
                    reference_distance: T::zero(),
                    epsilon_dominance: T::zero(),
                },
            },
            generation: 0,
            last_updated: std::time::SystemTime::now(),
        }
    }
}

impl<T: Float + Debug + Default + Send + Sync> Default for MultiObjectiveStatistics<T> {
    fn default() -> Self {
        Self {
            generation: 0,
            total_evaluations: 0,
            pareto_front_size: 0,
            best_hypervolume: T::zero(),
            convergence_history: Vec::new(),
            diversity_history: Vec::new(),
            algorithm_metrics: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };

    #[test]
    fn test_nsga2_creation() {
        let nsga2 = NSGA2::<f64>::new(50, 0.8, 0.1);
        assert_eq!(nsga2.population_size, 50);
        assert_eq!(nsga2.name(), "NSGA-II");
    }

    #[test]
    fn test_pareto_front_creation() {
        let front = ParetoFront::<f64>::new();
        assert!(front.solutions.is_empty());
        assert_eq!(front.generation, 0);
    }

    #[test]
    fn test_dominance_relation() {
        let mut nsga2 = NSGA2::<f64>::new(2, 0.8, 0.1);

        // Create two test individuals
        let arch = nsga2.generate_random_architecture().expect("unwrap failed");

        let ind1 = Individual {
            architecture: arch.clone(),
            objectives: vec![1.0, 2.0], // Better in first objective
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            id: "ind1".to_string(),
        };

        let ind2 = Individual {
            architecture: arch,
            objectives: vec![2.0, 1.0], // Better in second objective
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            id: "ind2".to_string(),
        };

        nsga2.population = vec![ind1, ind2];
        nsga2.config.objectives = vec![
            ObjectiveConfig {
                name: "obj1".to_string(),
                objective_type: ObjectiveType::Performance,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
            ObjectiveConfig {
                name: "obj2".to_string(),
                objective_type: ObjectiveType::Efficiency,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
        ];

        let relation = nsga2.dominance_relation(0, 1);
        assert_eq!(relation, DominanceRelation::NonDominated);
    }

    // ---- WeightedSum test helpers -------------------------------------

    /// Build a minimal [`OptimizerArchitecture`] tagged with `id`.
    fn make_architecture(id: &str) -> OptimizerArchitecture<f64> {
        OptimizerArchitecture {
            components: vec!["Adam".to_string()],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: id.to_string(),
        }
    }

    /// Build a [`SearchResult`] whose first objective maps to
    /// [`EvaluationMetric::Accuracy`] (= `obj0`) and whose second maps to
    /// [`EvaluationMetric::MemoryUsage`] (= `obj1`).
    fn make_search_result(id: &str, obj0: f64, obj1: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::Accuracy, obj0);
        metric_scores.insert(EvaluationMetric::MemoryUsage, obj1);

        let evaluation_results = EvaluationResults {
            metric_scores,
            overall_score: 0.0,
            confidence_intervals: HashMap::new(),
            evaluation_time: std::time::Duration::from_secs(0),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: HashMap::new(),
            training_trajectory: Vec::new(),
        };

        SearchResult {
            architecture: make_architecture(id),
            evaluation_results,
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

    /// Two minimization objectives backed by distinct metrics.
    fn two_minimize_objectives() -> Vec<ObjectiveConfig<f64>> {
        vec![
            ObjectiveConfig {
                name: "accuracy".to_string(),
                objective_type: ObjectiveType::Accuracy,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
            ObjectiveConfig {
                name: "memory".to_string(),
                objective_type: ObjectiveType::MemoryUsage,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
        ]
    }

    // ---- WeightedSum tests --------------------------------------------

    #[test]
    fn test_weighted_sum_creation() {
        let ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");
        assert_eq!(ws.name(), "WeightedSum");
        assert_eq!(ws.weights.len(), 2);
        assert!(ws.get_pareto_front().solutions.is_empty());
    }

    #[test]
    fn test_weighted_sum_update_pareto_front_keeps_non_dominated() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        // A = (1,2) and B = (2,1) are mutually non-dominated.
        // C = (3,3) is dominated by both A and B.
        let results = vec![
            make_search_result("A", 1.0, 2.0),
            make_search_result("B", 2.0, 1.0),
            make_search_result("C", 3.0, 3.0),
        ];

        let front = ws
            .update_pareto_front(&results)
            .expect("update pareto front");

        // Exactly the two non-dominated solutions must remain.
        assert_eq!(front.solutions.len(), 2);
        assert_eq!(front.metrics.num_solutions, 2);

        let mut ids: Vec<String> = front
            .solutions
            .iter()
            .map(|s| s.metadata.id.clone())
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["A".to_string(), "B".to_string()]);

        // Bounds should bracket the surviving objective values.
        assert_eq!(front.objective_bounds.min_values, vec![1.0, 1.0]);
        assert_eq!(front.objective_bounds.max_values, vec![2.0, 2.0]);

        // The stored front mirrors the returned one.
        assert_eq!(ws.get_pareto_front().solutions.len(), 2);
    }

    #[test]
    fn test_weighted_sum_update_pareto_front_incremental() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        // Seed the front with a single solution.
        ws.update_pareto_front(&[make_search_result("A", 2.0, 2.0)])
            .expect("seed front");
        assert_eq!(ws.get_pareto_front().solutions.len(), 1);

        // A new, strictly-better solution must dominate and replace it.
        let front = ws
            .update_pareto_front(&[make_search_result("B", 1.0, 1.0)])
            .expect("update front");
        assert_eq!(front.solutions.len(), 1);
        assert_eq!(front.solutions[0].metadata.id, "B");

        // Feeding no new solutions returns the front unchanged.
        let unchanged = ws.update_pareto_front(&[]).expect("empty update");
        assert_eq!(unchanged.solutions.len(), 1);
        assert_eq!(unchanged.solutions[0].metadata.id, "B");
    }

    #[test]
    fn test_weighted_sum_select_candidates_ranking() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        let population = vec![
            make_architecture("p0"),
            make_architecture("p1"),
            make_architecture("p2"),
            make_architecture("p3"),
        ];

        // Row-major objective matrix: 4 architectures x 2 objectives.
        // scores (equal weights, minimize): p0=4, p1=1, p2=2.5, p3=3.
        let objectives = vec![
            4.0, 4.0, // p0 -> 4.0
            1.0, 1.0, // p1 -> 1.0 (best)
            2.0, 3.0, // p2 -> 2.5
            3.0, 3.0, // p3 -> 3.0
        ];

        let selected = ws
            .select_candidates(&population, &objectives)
            .expect("select candidates");

        // Better half of 4 -> top 2 by ascending scalarized cost.
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].architecture_id, "p1");
        assert_eq!(selected[1].architecture_id, "p2");

        // The recorded best solution must match the top-ranked architecture.
        let best = ws.best_solution().expect("best solution recorded");
        assert_eq!(best.architecture.architecture_id, "p1");
    }

    #[test]
    fn test_weighted_sum_select_candidates_honors_maximize() {
        // Single maximize objective: larger objective value is preferred,
        // hence selected first despite weighted-sum minimizing internally.
        let objectives_cfg = vec![ObjectiveConfig {
            name: "accuracy".to_string(),
            objective_type: ObjectiveType::Accuracy,
            direction: OptimizationDirection::Maximize,
            weight: 1.0,
            priority: ObjectivePriority::High,
            normalization_bounds: None,
        }];
        let mut ws = WeightedSum::new(&objectives_cfg).expect("construct WeightedSum");

        let population = vec![make_architecture("low"), make_architecture("high")];
        // One objective per architecture.
        let objectives = vec![0.2, 0.9];

        let selected = ws
            .select_candidates(&population, &objectives)
            .expect("select candidates");

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].architecture_id, "high");
    }

    #[test]
    fn test_weighted_sum_select_candidates_empty_population() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");
        let selected = ws.select_candidates(&[], &[]).expect("select candidates");
        assert!(selected.is_empty());
    }
}
