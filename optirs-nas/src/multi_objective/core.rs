//! Shared core types: the MultiObjectiveOptimizer trait and the Pareto-front / statistics data model used by every algorithm in this module.

use crate::error::Result;
use crate::nas_engine::{MultiObjectiveConfig, OptimizerArchitecture, SearchResult};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

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
impl<T: Float + Debug + Default + Send + Sync> ParetoFront<T> {
    pub(super) fn new() -> Self {
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
impl<T: Float + Debug + Default + Send + Sync> Default for ParetoFront<T> {
    fn default() -> Self {
        Self::new()
    }
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
