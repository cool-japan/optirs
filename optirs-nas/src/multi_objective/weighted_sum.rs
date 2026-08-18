//! Weighted-sum scalarization multi-objective optimizer.

use crate::error::{OptimError, Result};
use crate::nas_engine::{
    MultiObjectiveConfig, ObjectiveConfig, ObjectiveType, OptimizationDirection,
    OptimizerArchitecture, SearchResult,
};
use crate::EvaluationMetric;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::fmt::Debug;

use super::core::{
    CreationMethod, Individual, MultiObjectiveOptimizer, MultiObjectiveStatistics, ObjectiveBounds,
    ParetoFront, ParetoSolution, SolutionMetadata,
};

/// Weighted sum approach
pub struct WeightedSum<T: Float + Debug + Send + Sync + 'static> {
    /// Objective weights
    pub(super) weights: Vec<T>,
    /// Objective configurations (direction, type, ...) used for
    /// scalarization and dominance computations.
    pub(super) objectives: Vec<ObjectiveConfig<T>>,
    /// Current best solution
    pub(super) best_solution: Option<Individual<T>>,
    /// Statistics
    pub(super) statistics: MultiObjectiveStatistics<T>,
    /// Maintained non-dominated set (Pareto front)
    pub(super) pareto_front: ParetoFront<T>,
    /// Generation counter
    pub(super) generation: usize,
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
    pub(super) fn extract_objectives(&self, result: &SearchResult<T>) -> Vec<T> {
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
    /// Weighted-sum scalarized **cost** of a single search result, using the
    /// configured (normalized) weights and honoring each objective's direction so
    /// that lower is always better.
    pub fn scalarized_cost_for_result(&self, result: &SearchResult<T>) -> T {
        let objectives = self.extract_objectives(result);
        self.scalarize(&objectives, &self.normalized_weights())
    }
    /// Mean pairwise Euclidean distance between the objective vectors of
    /// `results` — a real diversity measure over the population.
    pub fn mean_objective_distance(&self, results: &[SearchResult<T>]) -> f64 {
        if results.len() < 2 {
            return 0.0;
        }
        let vectors: Vec<Vec<T>> = results
            .iter()
            .map(|result| self.extract_objectives(result))
            .collect();
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let a = &vectors[i];
                let b = &vectors[j];
                let mut sq_sum = T::zero();
                for d in 0..a.len().min(b.len()) {
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
    /// Determine whether objective vector `a` Pareto-dominates `b`.
    ///
    /// Uses the same convention as NSGA-II's `dominance_relation`: `a`
    /// dominates `b` when `a` improves at least one objective and worsens
    /// none, honoring each objective's optimization direction.
    pub(super) fn dominates(&self, a: &[T], b: &[T]) -> bool {
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
    pub(super) fn scalarize(&self, objectives: &[T], normalized_weights: &[T]) -> T {
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
    pub(super) fn normalized_weights(&self) -> Vec<T> {
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
    pub(super) fn refresh_front_metrics(&mut self) {
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
    /// Adopt `config` as the live objective set and reset the per-run state.
    ///
    /// The body used to be a bare `Ok(())`, so a caller that constructed the
    /// scalarizer from one objective list and then handed `initialize` a different
    /// [`MultiObjectiveConfig`] kept scalarizing against the constructor's
    /// objectives — the configuration it was given was silently discarded. Every
    /// sibling optimizer ([`super::NSGA2`], [`super::NSGA3`],
    /// [`super::MOEADOptimizer`]) absorbs the configuration here; this one now does
    /// too, and clears the front/statistics so a re-initialized optimizer never
    /// reports a front that was built under the previous objectives.
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        if config.objectives.is_empty() {
            return Err(OptimError::InvalidConfig(
                "WeightedSum has nothing to scalarize: the supplied \
                 MultiObjectiveConfig declares no objectives"
                    .to_string(),
            ));
        }
        self.weights = config.objectives.iter().map(|obj| obj.weight).collect();
        self.objectives = config.objectives.clone();
        self.best_solution = None;
        self.generation = 0;
        self.statistics = MultiObjectiveStatistics::default();
        self.pareto_front = ParetoFront::default();
        Ok(())
    }
    fn update_pareto_front(&mut self, new_solutions: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        if new_solutions.is_empty() {
            return Ok(self.pareto_front.clone());
        }
        self.generation += 1;
        self.statistics.total_evaluations += new_solutions.len();
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
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
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
