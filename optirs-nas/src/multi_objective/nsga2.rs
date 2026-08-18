//! NSGA-II: the complete, real multi-objective optimizer implementation (non-dominated sorting, crowding distance, tournament selection, crossover/mutation).

use crate::error::{OptimError, Result};
use crate::nas_engine::{
    MultiObjectiveConfig, OptimizationDirection, OptimizerArchitecture, SearchResult,
};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::cmp::Ordering;
use std::fmt::Debug;

use super::core::{
    CreationMethod, Individual, MultiObjectiveOptimizer, MultiObjectiveStatistics, ObjectiveBounds,
    ParetoFront, ParetoSolution, SolutionMetadata,
};
use super::hypervolume::{
    derive_reference_point, hypervolume_minimization, normalize_front_for_minimization,
    normalize_reference_for_minimization,
};
use super::metrics;
use super::operators;

/// Dominance relation between two solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DominanceRelation {
    Dominates,
    DominatedBy,
    NonDominated,
}
/// NSGA-II implementation
pub struct NSGA2<T: Float + Debug + Send + Sync + 'static> {
    /// Algorithm configuration
    pub(super) config: MultiObjectiveConfig<T>,
    /// Current population
    pub(super) population: Vec<Individual<T>>,
    /// Current Pareto front
    pub(super) pareto_front: ParetoFront<T>,
    /// Generation counter
    pub(super) generation: usize,
    /// Statistics
    pub(super) statistics: MultiObjectiveStatistics<T>,
    /// Population size
    pub(super) population_size: usize,
    /// Crossover probability
    pub(super) crossover_prob: f64,
    /// Mutation probability
    pub(super) mutation_prob: f64,
    /// Random number generator
    pub(super) rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Reference point for the hypervolume indicator, in the **raw** objective
    /// space the configured [`ObjectiveType`]s live in (mixed directions are
    /// allowed; the indicator normalizes internally).
    ///
    /// `None` until the first front is measured, at which point one is derived
    /// from that front and **latched**. Holding it fixed is what makes the
    /// hypervolume comparable between generations — a reference that tracks the
    /// current front moves the box being measured. Use
    /// [`NSGA2::set_hypervolume_reference`] to supply your own instead.
    pub(super) hypervolume_reference: Option<Vec<T>>,
    /// Hypervolume recorded at the previous front update, used to compute the
    /// `convergence` metric as a genuine indicator-change rate.
    pub(super) previous_hypervolume: Option<T>,
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
    /// Create a new NSGA-II optimizer seeded from OS entropy.
    ///
    /// The RNG used to be a hard-coded `Random::seed(42)`. That was invisible while
    /// `generate_random_architecture` returned a fixed architecture, but now that
    /// initialization genuinely samples (F18) a fixed seed would make **every**
    /// NSGA-II instance explore the identical initial population. Use
    /// [`NSGA2::with_seed`] when reproducibility is wanted.
    pub fn new(population_size: usize, crossover_prob: f64, mutation_prob: f64) -> Self {
        Self::with_seed(
            population_size,
            crossover_prob,
            mutation_prob,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Create a fully reproducible NSGA-II optimizer.
    pub fn with_seed(
        population_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        seed: u64,
    ) -> Self {
        Self {
            config: MultiObjectiveConfig::default(),
            population: Vec::new(),
            pareto_front: ParetoFront::new(),
            generation: 0,
            statistics: MultiObjectiveStatistics::default(),
            population_size,
            crossover_prob,
            mutation_prob,
            rng: Random::seed(seed),
            hypervolume_reference: None,
            previous_hypervolume: None,
        }
    }

    /// Pin the hypervolume reference point (raw objective space, one entry per
    /// configured objective). Supplying it explicitly is preferable to the
    /// derived default whenever hypervolumes from different runs are compared.
    pub fn set_hypervolume_reference(&mut self, reference: Vec<T>) -> Result<()> {
        if reference.len() != self.config.objectives.len() {
            return Err(OptimError::InvalidConfig(format!(
                "hypervolume reference has {} entries but {} objectives are configured",
                reference.len(),
                self.config.objectives.len()
            )));
        }
        self.hypervolume_reference = Some(reference);
        Ok(())
    }

    /// The reference point currently used by the hypervolume indicator, if one
    /// has been supplied or derived yet.
    pub fn hypervolume_reference(&self) -> Option<&[T]> {
        self.hypervolume_reference.as_deref()
    }

    /// Per-objective optimization directions, in configuration order.
    pub(super) fn objective_directions(&self) -> Vec<OptimizationDirection> {
        self.config
            .objectives
            .iter()
            .map(|objective| objective.direction.clone())
            .collect()
    }

    /// Objective vectors of the current Pareto front, mapped into the
    /// pure-minimization convention every indicator in [`super::metrics`] and
    /// [`super::hypervolume`] expects.
    pub(super) fn front_in_minimization_space(&self) -> Vec<Vec<T>> {
        let directions = self.objective_directions();
        let raw: Vec<Vec<T>> = self
            .pareto_front
            .solutions
            .iter()
            .map(|solution| solution.objectives.clone())
            .collect();
        normalize_front_for_minimization(&raw, &directions)
    }
    /// Initialize the population with genuinely diverse sampled architectures
    /// (F18).
    ///
    /// Individuals start with an **empty** objective vector, which is the honest
    /// representation of "not evaluated yet": the previous `vec![T::zero(); n]`
    /// made every unevaluated individual mutually non-dominated, so the reported
    /// "Pareto front" was the entire population. See [`NSGA2::is_evaluated`].
    pub(super) fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();
        for i in 0..self.population_size {
            let architecture = self.generate_random_architecture()?;
            let individual = Individual {
                id: format!("ind_{}", i),
                architecture,
                objectives: Vec::new(),
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
            };
            self.population.push(individual);
        }
        Ok(())
    }
    /// Sample a random architecture from the optimizer-component vocabulary and
    /// the searchable hyperparameter ranges, using this optimizer's own RNG so a
    /// seeded NSGA-II remains reproducible.
    pub(super) fn generate_random_architecture(&mut self) -> Result<OptimizerArchitecture<T>> {
        Ok(operators::sample_architecture(&mut self.rng, "arch"))
    }
    /// Whether `individual` carries a complete objective vector, i.e. whether it
    /// has actually been evaluated against the configured objectives. Individuals
    /// that have not are excluded from dominance and from the Pareto front rather
    /// than being treated as all-zero (and therefore optimal).
    pub(super) fn is_evaluated(&self, individual: &Individual<T>) -> bool {
        !self.config.objectives.is_empty()
            && individual.objectives.len() == self.config.objectives.len()
    }
    /// Perform non-dominated sorting
    pub(super) fn non_dominated_sort(&mut self) -> Vec<Vec<usize>> {
        let n = self.population.len();
        let mut fronts = Vec::new();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
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
    /// Determine dominance relation between two individuals.
    ///
    /// An individual that has not been evaluated yet (see [`NSGA2::is_evaluated`])
    /// takes part in no dominance relation at all: it neither dominates nor is
    /// dominated, so it cannot distort the ranks of the evaluated individuals and
    /// is filtered out of the front by
    /// [`NSGA2::update_pareto_front_from_population`].
    pub(super) fn dominance_relation(&self, i: usize, j: usize) -> DominanceRelation {
        let ind_i = &self.population[i];
        let ind_j = &self.population[j];
        if !self.is_evaluated(ind_i) || !self.is_evaluated(ind_j) {
            return DominanceRelation::NonDominated;
        }
        let mut i_dominates = false;
        let mut j_dominates = false;
        for k in 0..ind_i.objectives.len().min(self.config.objectives.len()) {
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
    /// Calculate crowding distance over the evaluated members of `front`.
    ///
    /// Unevaluated individuals carry no objective vector, so they are excluded
    /// here as well — indexing their (empty) objectives would otherwise panic.
    pub(super) fn calculate_crowding_distance(&mut self, front: &[usize]) {
        for &idx in front {
            self.population[idx].crowding_distance = T::zero();
        }
        let front: Vec<usize> = front
            .iter()
            .copied()
            .filter(|&idx| self.is_evaluated(&self.population[idx]))
            .collect();
        let front = front.as_slice();
        let front_size = front.len();
        if front_size == 0 {
            return;
        }
        if front_size <= 2 {
            for &idx in front {
                self.population[idx].crowding_distance = T::infinity();
            }
            return;
        }
        let num_objectives = self.config.objectives.len();
        for obj_idx in 0..num_objectives {
            let mut sorted_front = front.to_vec();
            sorted_front.sort_by(|&a, &b| {
                self.population[a].objectives[obj_idx]
                    .partial_cmp(&self.population[b].objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });
            self.population[sorted_front[0]].crowding_distance = T::infinity();
            self.population[sorted_front[front_size - 1]].crowding_distance = T::infinity();
            let obj_min = self.population[sorted_front[0]].objectives[obj_idx];
            let obj_max = self.population[sorted_front[front_size - 1]].objectives[obj_idx];
            let obj_range = obj_max - obj_min;
            if obj_range > T::zero() {
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
    pub(super) fn environmental_selection(
        &mut self,
        combined_population: Vec<Individual<T>>,
    ) -> Vec<Individual<T>> {
        self.population = combined_population;
        let fronts = self.non_dominated_sort();
        let mut new_population = Vec::new();
        let mut front_idx = 0;
        while front_idx < fronts.len() {
            let front = &fronts[front_idx];
            if new_population.len() + front.len() <= self.population_size {
                self.calculate_crowding_distance(front);
                for &idx in front {
                    new_population.push(self.population[idx].clone());
                }
                front_idx += 1;
            } else {
                self.calculate_crowding_distance(front);
                let mut front_individuals: Vec<_> = front
                    .iter()
                    .map(|&idx| (idx, self.population[idx].crowding_distance))
                    .collect();
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
    /// Update Pareto front from current population.
    ///
    /// Only rank-0 individuals that have actually been evaluated are admitted —
    /// see [`NSGA2::is_evaluated`]. Callers must have run
    /// [`NSGA2::non_dominated_sort`] first, otherwise every rank is still the
    /// initial `0` and the "front" degenerates to the whole population.
    pub(super) fn update_pareto_front_from_population(&mut self) {
        let mut pareto_solutions = Vec::new();
        for individual in &self.population {
            if individual.rank == 0 && self.is_evaluated(individual) {
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
                        creation_method: creation_method_from_id(
                            &individual.architecture.architecture_id,
                        ),
                    },
                };
                pareto_solutions.push(solution);
            }
        }
        self.pareto_front.solutions = pareto_solutions;
        self.pareto_front.generation = self.generation;
        self.pareto_front.last_updated = std::time::SystemTime::now();
        self.update_objective_bounds();
        self.calculate_front_metrics();
    }
    pub(super) fn update_objective_bounds(&mut self) {
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
    /// Compute every [`FrontMetrics`] field from the current front. All of them
    /// are real measurements: the hypervolume is the exact indicator against a
    /// latched reference point (F17), `convergence` is the indicator's own
    /// change rate, and the coverage block is computed from the front's geometry
    /// instead of the previously hardcoded `0.5` / `0`.
    pub(super) fn calculate_front_metrics(&mut self) {
        let front = self.front_in_minimization_space();
        let hypervolume = self.calculate_hypervolume();

        let previous = self.previous_hypervolume;
        self.previous_hypervolume = Some(hypervolume);

        let directions = self.objective_directions();
        let reference = self
            .hypervolume_reference
            .as_ref()
            .map(|raw| normalize_reference_for_minimization(raw, &directions))
            .unwrap_or_default();

        // How far the non-dominated front is from covering the whole population
        // the optimizer has in hand; zero when it dominates all of it.
        let population_objectives: Vec<Vec<T>> = normalize_front_for_minimization(
            &self
                .population
                .iter()
                .map(|individual| individual.objectives.clone())
                .collect::<Vec<_>>(),
            &directions,
        );

        self.pareto_front.metrics = metrics::front_metrics_in_minimization_space(
            &front,
            &population_objectives,
            &reference,
            hypervolume,
            previous,
        );
        let convergence = self.pareto_front.metrics.convergence;
        let spread = self.pareto_front.metrics.spread;
        self.statistics.pareto_front_size = self.pareto_front.solutions.len();
        // `best_hypervolume` is the best value ever seen, not merely the latest.
        if hypervolume > self.statistics.best_hypervolume {
            self.statistics.best_hypervolume = hypervolume;
        }
        self.statistics.generation = self.generation;
        self.statistics.convergence_history.push(convergence);
        self.statistics.diversity_history.push(spread);
    }
    /// Exact hypervolume of the current Pareto front (F17).
    ///
    /// The front is mapped into the pure-minimization convention (so `Maximize`
    /// objectives are handled correctly) and measured against the reference
    /// point, which is derived from the first front seen and then held fixed.
    /// This replaces the previous "bounding-box product times solution count"
    /// heuristic, which was neither a hypervolume nor monotone.
    pub(super) fn calculate_hypervolume(&mut self) -> T {
        let front = self.front_in_minimization_space();
        if front.is_empty() {
            return T::zero();
        }
        let directions = self.objective_directions();

        if self.hypervolume_reference.is_none() {
            // Derive in minimization space, then store back in raw space so the
            // public accessor speaks the caller's objective convention.
            let Some(derived) = derive_reference_point(&front) else {
                return T::zero();
            };
            self.hypervolume_reference =
                Some(normalize_reference_for_minimization(&derived, &directions));
        }
        let Some(raw_reference) = self.hypervolume_reference.as_ref() else {
            return T::zero();
        };
        let reference = normalize_reference_for_minimization(raw_reference, &directions);
        hypervolume_minimization(&front, &reference)
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
    > NSGA2<T>
{
    /// Map a configured [`ObjectiveType`] onto the [`EvaluationMetric`] used to
    /// read its value from an [`EvaluationResults`]. Kept identical to the
    /// mapping used by [`MultiObjectiveOptimizer::update_pareto_front`] so the
    /// helpers below agree with the main optimization path.
    pub(super) fn objective_vector_for_result(&self, result: &SearchResult<T>) -> Vec<T> {
        super::core::objective_vector_for_result(&self.config.objectives, result)
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
        let fronts = self.non_dominated_sort();
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
        let fronts = self.non_dominated_sort();
        for front in &fronts {
            self.calculate_crowding_distance(front);
        }
        self.update_pareto_front_from_population();
        self.pareto_front.clone()
    }
    /// Compute the mean pairwise Euclidean distance between the objective
    /// vectors of `results`, providing a real diversity metric for the
    /// population. Returns `0.0` for fewer than two solutions.
    pub(crate) fn mean_objective_distance(&self, results: &[SearchResult<T>]) -> f64 {
        let objective_vectors: Vec<Vec<T>> = results
            .iter()
            .map(|r| self.objective_vector_for_result(r))
            .collect();
        metrics::mean_pairwise_distance(&objective_vectors)
    }
    pub(super) fn tournament_selection(&mut self, tournamentsize: usize) -> Result<Individual<T>> {
        if self.population.is_empty() {
            return Err(OptimError::InvalidConfig("Empty population".to_string()));
        }
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        for _ in 1..tournamentsize {
            let idx = self.rng.gen_range(0..self.population.len());
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
    /// Uniform crossover over the component sequence **and** the numeric
    /// hyperparameters (F18). The offspring's objectives are cleared, because an
    /// offspring has not been evaluated — carrying the parent's objectives over
    /// (as the previous `parent1.clone()` did) would have let an unevaluated
    /// architecture claim its parent's front position.
    pub(super) fn crossover(
        &mut self,
        parent1: &Individual<T>,
        parent2: &Individual<T>,
    ) -> Result<Individual<T>> {
        let architecture = operators::crossover_architectures(
            &mut self.rng,
            &parent1.architecture,
            &parent2.architecture,
        );
        Ok(Individual {
            id: architecture.architecture_id.clone(),
            architecture,
            objectives: Vec::new(),
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: T::zero(),
            fitness: T::zero(),
        })
    }
    /// Mutate an individual's architecture — components as well as
    /// hyperparameters (F18), with the numeric jitter clamped to each
    /// hyperparameter's declared range instead of the previous unbounded
    /// `+/-0.05` additive noise (which could drive a learning rate negative).
    pub(super) fn mutate(&mut self, individual: &mut Individual<T>) -> Result<()> {
        let changed = operators::mutate_architecture(
            &mut self.rng,
            &mut individual.architecture,
            NSGA2_COMPONENT_MUTATION_RATE,
            NSGA2_PARAMETER_MUTATION_RATE,
        );
        if changed {
            // A mutated architecture is a different candidate: its recorded
            // objectives no longer describe it.
            individual.objectives.clear();
            individual.id = individual.architecture.architecture_id.clone();
        }
        Ok(())
    }
}

/// Per-component-position probability of a structural mutation.
pub(super) const NSGA2_COMPONENT_MUTATION_RATE: f64 = 0.2;

/// Per-parameter probability of a numeric mutation.
pub(super) const NSGA2_PARAMETER_MUTATION_RATE: f64 = 0.3;

/// Recover how an architecture was created from the prefix the variation
/// operators stamp onto its identifier. Reporting the actual provenance is more
/// useful than the previous unconditional
/// [`CreationMethod::RandomGeneration`].
fn creation_method_from_id(architecture_id: &str) -> CreationMethod {
    if architecture_id.starts_with("offspring_") {
        CreationMethod::Crossover
    } else if architecture_id.starts_with("mutant_") {
        CreationMethod::Mutation
    } else if architecture_id.starts_with("arch_") {
        CreationMethod::RandomGeneration
    } else {
        CreationMethod::Custom
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
    > MultiObjectiveOptimizer<T> for NSGA2<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        self.config = config.clone();
        self.initialize_population()?;
        Ok(())
    }
    /// Absorb `results` into the population and rebuild the Pareto front (F18).
    ///
    /// Results are matched to population members by `architecture_id`, not by
    /// position: the previous index-based mapping silently pasted the *i*-th
    /// result's objectives onto the *i*-th individual even though evaluation order
    /// need not match population order, so objectives and architectures could be
    /// attributed to the wrong candidate. Results whose architecture is not in the
    /// population are appended (they are genuinely new evaluated candidates).
    ///
    /// The full non-dominated sort and crowding-distance assignment now run here
    /// too. Without them every rank stayed `0` and the returned "Pareto front"
    /// was the entire population.
    fn update_pareto_front(&mut self, results: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        for result in results {
            let objectives = self.objective_vector_for_result(result);
            let architecture_id = result.architecture.architecture_id.as_str();
            let existing = if architecture_id.is_empty() {
                None
            } else {
                self.population
                    .iter()
                    .position(|ind| ind.architecture.architecture_id == architecture_id)
            };
            match existing {
                Some(index) => {
                    self.population[index].objectives = objectives;
                    self.population[index].architecture = result.architecture.clone();
                }
                None => {
                    // Prefer an unevaluated slot so the population does not grow
                    // without bound when architecture ids are not carried through.
                    let vacancy = self
                        .population
                        .iter()
                        .position(|ind| !self.is_evaluated(ind));
                    let id = if architecture_id.is_empty() {
                        format!("ind_{}", self.population.len())
                    } else {
                        architecture_id.to_string()
                    };
                    match vacancy {
                        Some(index) => {
                            self.population[index].objectives = objectives;
                            self.population[index].architecture = result.architecture.clone();
                            self.population[index].id = id;
                        }
                        None => self.population.push(Individual {
                            id,
                            architecture: result.architecture.clone(),
                            objectives,
                            constraints: Vec::new(),
                            rank: 0,
                            crowding_distance: T::zero(),
                            fitness: T::zero(),
                        }),
                    }
                }
            }
        }
        // NSGA-II's survivor selection, which was implemented but never called.
        // Without it the population grew by one individual for every
        // never-before-seen architecture, for the whole run: it stopped being a
        // population of `population_size` at all, and both the memory it occupies
        // and the O(n^2) non-dominated sort below scaled with the number of
        // evaluations instead. `environmental_selection` truncates the combined
        // parent+offspring pool back to `population_size` by rank, breaking ties on
        // crowding distance — the (mu + lambda) step of the published algorithm.
        //
        // Only reachable once every slot holds an evaluated individual: the absorb
        // loop above fills unevaluated vacancies first and only appends when there
        // are none, so truncation never discards a candidate that has not been
        // measured.
        if self.population.len() > self.population_size {
            let combined = std::mem::take(&mut self.population);
            self.population = self.environmental_selection(combined);
        }

        self.generation += 1;
        self.statistics.total_evaluations += results.len();
        let fronts = self.non_dominated_sort();
        for front in &fronts {
            self.calculate_crowding_distance(front);
        }
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
        let mut new_population = Vec::new();
        for _ in 0..self.population_size {
            let parent1 = self.tournament_selection(2)?;
            let parent2 = self.tournament_selection(2)?;
            let mut offspring = if self.rng.gen_range(0.0..1.0) < self.crossover_prob {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };
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
