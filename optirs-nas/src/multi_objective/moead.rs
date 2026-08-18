//! MOEA/D — decomposition-based multi-objective optimizer (Zhang & Li, IEEE TEVC
//! 2007), implemented for real.
//!
//! The algorithm decomposes the multi-objective problem into `N` single-objective
//! subproblems, one per Das-Dennis weight vector, and keeps one current solution
//! per subproblem. Each subproblem is improved only from the solutions of its
//! neighbouring subproblems, which is what makes the population spread evenly
//! along the front without an explicit diversity operator:
//!
//! 1. **Decomposition** — [`super::decomposition::das_dennis_weights`] builds a
//!    uniform simplex lattice of weight vectors; the number of subproblems *is*
//!    the population size.
//! 2. **Neighbourhoods** — `B(i)` is the `neighborhood_size` weight vectors
//!    closest to `w_i` ([`super::decomposition::weight_neighborhoods`]).
//! 3. **Ideal point** — `z*` tracks the best value seen per objective and is what
//!    every scalarization is measured against.
//! 4. **Reproduction** — parents are drawn from `B(i)` (with probability
//!    [`NEIGHBORHOOD_SELECTION_PROB`], otherwise from the whole population) and
//!    recombined by the shared architecture operators.
//! 5. **Scalarized replacement** — an evaluated offspring replaces up to
//!    [`MAX_REPLACEMENTS`] neighbouring subproblem solutions, each only if it is
//!    genuinely better *for that subproblem's* scalarization.
//! 6. **External archive** — the non-dominated solutions found across the whole
//!    run, which is what is reported as the Pareto front.
//!
//! Before this implementation existed, every fallible method here returned
//! `NotImplemented` (and before *that*, `Ok(())` with an empty front, so a search
//! configured for MOEA/D ran to completion and reported nothing). The type
//! scaffolding — including [`DecompositionMethod`] — is unchanged; it is now
//! backed by the algorithm it describes.

use crate::error::{OptimError, Result};
use crate::nas_engine::{MultiObjectiveConfig, OptimizerArchitecture, SearchResult};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use super::core::{
    objective_directions, objective_vector_for_result, CreationMethod, Individual,
    MultiObjectiveOptimizer, MultiObjectiveStatistics, ObjectiveBounds, ParetoFront,
    ParetoSolution, SolutionMetadata,
};
use super::decomposition::{
    das_dennis_weights, divisions_for_at_least, scalarize, update_ideal_point,
    weight_neighborhoods, Scalarization,
};
use super::hypervolume::{
    derive_reference_point, hypervolume_minimization, normalize_front_for_minimization,
    normalize_reference_for_minimization,
};
use super::metrics;
use super::operators;

/// Default number of subproblems requested when the caller does not say. Kept
/// small on purpose: in NAS every subproblem solution costs a full architecture
/// evaluation, so the canonical `N = 100` of the MOEA/D papers would be an
/// unreasonable default here. The actual count is the smallest Das-Dennis lattice
/// that reaches this target.
pub const DEFAULT_SUBPROBLEM_TARGET: usize = 20;

/// Default neighbourhood size `T`.
pub const DEFAULT_NEIGHBORHOOD_SIZE: usize = 5;

/// Probability that both parents are drawn from the subproblem's neighbourhood
/// rather than the whole population (`delta` in MOEA/D-DE).
pub const NEIGHBORHOOD_SELECTION_PROB: f64 = 0.9;

/// Maximum number of neighbouring subproblems a single offspring may replace
/// (`nr` in MOEA/D-DE). Bounding it is what stops one good solution from taking
/// over the whole population.
pub const MAX_REPLACEMENTS: usize = 2;

/// Default bound on the external non-dominated archive.
pub const DEFAULT_ARCHIVE_CAPACITY: usize = 100;

/// Per-component-position probability of a structural mutation.
const MOEAD_COMPONENT_MUTATION_RATE: f64 = 0.2;

/// Per-parameter probability of a numeric mutation.
const MOEAD_PARAMETER_MUTATION_RATE: f64 = 0.3;

/// Probability that reproduction recombines two parents rather than copying one.
const MOEAD_CROSSOVER_PROB: f64 = 0.9;

/// Probability that an offspring is mutated.
const MOEAD_MUTATION_PROB: f64 = 0.3;

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

impl DecompositionMethod {
    /// The scalarization this method is implemented by.
    pub fn scalarization(self) -> Scalarization {
        match self {
            DecompositionMethod::WeightedSum => Scalarization::WeightedSum,
            DecompositionMethod::Tchebycheff => Scalarization::Tchebycheff,
            DecompositionMethod::PBI => Scalarization::PenaltyBoundaryIntersection,
            DecompositionMethod::ASF => Scalarization::AchievementScalarizing,
        }
    }
}

/// MOEA/D implementation
pub struct MOEADOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Algorithm configuration
    pub(super) config: MultiObjectiveConfig<T>,
    /// Weight vectors, one per subproblem (Das-Dennis lattice)
    pub(super) weight_vectors: Vec<Vec<T>>,
    /// Current solution of each subproblem; an individual with an empty objective
    /// vector is a subproblem that has not been solved yet.
    pub(super) population: Vec<Individual<T>>,
    /// Neighbor indices for each subproblem
    pub(super) neighbors: Vec<Vec<usize>>,
    /// Current Pareto front (the external archive, published)
    pub(super) pareto_front: ParetoFront<T>,
    /// Ideal point `z*`, in **minimization** space
    pub(super) ideal_point: Vec<T>,
    /// Decomposition method
    pub(super) decomposition: DecompositionMethod,
    /// Neighborhood size
    pub(super) neighborhood_size: usize,
    /// Generation counter
    pub(super) generation: usize,
    /// Statistics
    pub(super) statistics: MultiObjectiveStatistics<T>,
    /// Number of divisions of the Das-Dennis lattice actually used
    pub(super) divisions: usize,
    /// Requested number of subproblems (the lattice may be slightly larger)
    pub(super) subproblem_target: usize,
    /// External non-dominated archive
    pub(super) archive: Vec<Individual<T>>,
    /// Bound on the archive size
    pub(super) archive_capacity: usize,
    /// Which subproblem each generated architecture was created for, so an
    /// evaluated result is compared against the subproblem it was meant to
    /// improve rather than one picked by position.
    pub(super) subproblem_of: HashMap<String, usize>,
    /// Insertion order of `subproblem_of`, used to bound it.
    pub(super) subproblem_order: VecDeque<String>,
    /// Random number generator
    pub(super) rng: Random<scirs2_core::random::rngs::StdRng>,
    /// Latched hypervolume reference point, in **raw** objective space
    pub(super) hypervolume_reference: Option<Vec<T>>,
    /// Hypervolume at the previous front update, for the `convergence` metric
    pub(super) previous_hypervolume: Option<T>,
    /// Replacements performed during the most recent update
    pub(super) last_replacements: usize,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> MOEADOptimizer<T> {
    /// Construct MOEA/D with the default subproblem target, Tchebycheff
    /// decomposition and an OS-entropy seed.
    ///
    /// The optimizer is fully usable after construction only once
    /// [`MultiObjectiveOptimizer::initialize`] has built the lattice; `new` stores
    /// the configuration and immediately initializes from it when it already
    /// carries objectives, so a caller that never calls `initialize` still gets a
    /// working optimizer rather than a silent empty one.
    pub fn new(config: MultiObjectiveConfig<T>) -> Result<Self> {
        Self::with_settings(
            config,
            DEFAULT_SUBPROBLEM_TARGET,
            DecompositionMethod::Tchebycheff,
            DEFAULT_NEIGHBORHOOD_SIZE,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Construct a fully reproducible MOEA/D.
    pub fn with_seed(config: MultiObjectiveConfig<T>, seed: u64) -> Result<Self> {
        Self::with_settings(
            config,
            DEFAULT_SUBPROBLEM_TARGET,
            DecompositionMethod::Tchebycheff,
            DEFAULT_NEIGHBORHOOD_SIZE,
            seed,
        )
    }

    /// Construct MOEA/D with every algorithm parameter given explicitly.
    pub fn with_settings(
        config: MultiObjectiveConfig<T>,
        subproblem_target: usize,
        decomposition: DecompositionMethod,
        neighborhood_size: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut optimizer = Self {
            config: config.clone(),
            weight_vectors: Vec::new(),
            population: Vec::new(),
            neighbors: Vec::new(),
            pareto_front: ParetoFront::default(),
            ideal_point: Vec::new(),
            decomposition,
            neighborhood_size: neighborhood_size.max(1),
            generation: 0,
            statistics: MultiObjectiveStatistics::default(),
            divisions: 0,
            subproblem_target: subproblem_target.max(1),
            archive: Vec::new(),
            archive_capacity: DEFAULT_ARCHIVE_CAPACITY,
            subproblem_of: HashMap::new(),
            subproblem_order: VecDeque::new(),
            rng: Random::seed(seed),
            hypervolume_reference: None,
            previous_hypervolume: None,
            last_replacements: 0,
        };
        if !config.objectives.is_empty() {
            optimizer.build_decomposition(&config)?;
        }
        Ok(optimizer)
    }

    /// The decomposition method in use.
    pub fn decomposition(&self) -> DecompositionMethod {
        self.decomposition
    }

    /// Switch the scalarization. Existing subproblem solutions are kept: they are
    /// still valid solutions, they are simply compared differently from now on.
    pub fn set_decomposition(&mut self, decomposition: DecompositionMethod) {
        self.decomposition = decomposition;
    }

    /// Number of subproblems (= weight vectors = population size).
    pub fn subproblem_count(&self) -> usize {
        self.weight_vectors.len()
    }

    /// Weight vector of subproblem `index`, if it exists.
    pub fn weight_vector(&self, index: usize) -> Option<&[T]> {
        self.weight_vectors
            .get(index)
            .map(|weight| weight.as_slice())
    }

    /// The neighbourhood `B(index)`.
    pub fn neighborhood(&self, index: usize) -> Option<&[usize]> {
        self.neighbors.get(index).map(|list| list.as_slice())
    }

    /// The current ideal point `z*`, in minimization space.
    pub fn ideal_point(&self) -> &[T] {
        &self.ideal_point
    }

    /// Bound on the external archive.
    pub fn set_archive_capacity(&mut self, capacity: usize) {
        self.archive_capacity = capacity.max(1);
        self.truncate_archive();
    }

    /// Pin the hypervolume reference point (raw objective space, one entry per
    /// configured objective).
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

    /// Build the weight lattice, neighbourhoods and initial (unevaluated)
    /// subproblem solutions from `config`.
    fn build_decomposition(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        let num_objectives = config.objectives.len();
        if num_objectives == 0 {
            return Err(OptimError::InvalidConfig(
                "MOEA/D needs at least one configured objective to decompose".to_string(),
            ));
        }
        let (divisions, lattice_size) =
            divisions_for_at_least(num_objectives, self.subproblem_target).ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "no Das-Dennis lattice with at least {} vectors exists for {} objectives",
                    self.subproblem_target, num_objectives
                ))
            })?;
        self.divisions = divisions;
        self.weight_vectors = das_dennis_weights::<T>(num_objectives, divisions);
        if self.weight_vectors.len() != lattice_size {
            return Err(OptimError::OptimizationError(format!(
                "weight lattice size {} disagrees with the computed {}",
                self.weight_vectors.len(),
                lattice_size
            )));
        }
        self.neighbors = weight_neighborhoods(&self.weight_vectors, self.neighborhood_size);
        self.ideal_point = vec![T::infinity(); num_objectives];
        self.population.clear();
        self.subproblem_of.clear();
        self.subproblem_order.clear();
        for index in 0..self.weight_vectors.len() {
            let architecture = operators::sample_architecture::<T>(&mut self.rng, "arch");
            let id = architecture.architecture_id.clone();
            self.population.push(Individual {
                architecture,
                objectives: Vec::new(),
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
                id: id.clone(),
            });
            self.remember_subproblem(id, index);
        }
        Ok(())
    }

    /// Record that `architecture_id` was generated for subproblem `index`, keeping
    /// the map bounded so a long run cannot grow it without limit.
    fn remember_subproblem(&mut self, architecture_id: String, index: usize) {
        if architecture_id.is_empty() {
            return;
        }
        let capacity = (self.weight_vectors.len().max(1)).saturating_mul(4);
        if self
            .subproblem_of
            .insert(architecture_id.clone(), index)
            .is_none()
        {
            self.subproblem_order.push_back(architecture_id);
        }
        while self.subproblem_order.len() > capacity {
            if let Some(oldest) = self.subproblem_order.pop_front() {
                self.subproblem_of.remove(&oldest);
            }
        }
    }

    /// Objective vector of `result` mapped into the pure-minimization convention.
    fn minimization_objectives(&self, result: &SearchResult<T>) -> Vec<T> {
        let raw = objective_vector_for_result(&self.config.objectives, result);
        self.to_minimization(&raw)
    }

    /// Map a raw objective vector into minimization space.
    fn to_minimization(&self, raw: &[T]) -> Vec<T> {
        let directions = objective_directions(&self.config.objectives);
        normalize_front_for_minimization(&[raw.to_vec()], &directions)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// The ideal point in a form a scalarization can actually be measured
    /// against.
    ///
    /// [`Self::ideal_point`] starts as `+inf` per objective — the standard "nothing
    /// observed yet" marker — but subtracting an infinite reference makes every
    /// scalarized cost infinite and every comparison meaningless, so any
    /// non-finite entry is reported as `0` here. `reference` lets the caller fold
    /// in the set it is about to rank, which is what makes ranking work before any
    /// result has been absorbed.
    fn effective_ideal(&self, reference: &[Vec<T>]) -> Vec<T> {
        let num_objectives = self.config.objectives.len();
        let mut ideal = self.ideal_point.clone();
        if ideal.len() < num_objectives {
            ideal.resize(num_objectives, T::infinity());
        }
        for point in reference {
            update_ideal_point(&mut ideal, point);
        }
        for value in ideal.iter_mut() {
            if !value.is_finite() {
                *value = T::zero();
            }
        }
        ideal
    }

    /// Scalarized cost of a minimization-space objective vector for subproblem
    /// `index`, measured against `ideal`. Lower is better.
    fn subproblem_cost(&self, index: usize, minimized: &[T], ideal: &[T]) -> Option<T> {
        let weight = self.weight_vectors.get(index)?;
        Some(scalarize(
            self.decomposition.scalarization(),
            minimized,
            weight,
            ideal,
        ))
    }

    /// The subproblem an unassigned solution belongs to: the one whose
    /// scalarization it is best for. This is the honest fallback for a result
    /// whose architecture MOEA/D did not generate (an externally supplied
    /// candidate), and it is what makes the algorithm usable inside an engine that
    /// evaluates architectures from several sources.
    fn best_fit_subproblem(&self, minimized: &[T], ideal: &[T]) -> Option<usize> {
        let mut best_index = None;
        let mut best_cost = T::infinity();
        for index in 0..self.weight_vectors.len() {
            let Some(cost) = self.subproblem_cost(index, minimized, ideal) else {
                continue;
            };
            if cost < best_cost {
                best_cost = cost;
                best_index = Some(index);
            }
        }
        best_index
    }

    /// Whether `individual` carries a complete objective vector.
    fn is_evaluated(&self, individual: &Individual<T>) -> bool {
        !self.config.objectives.is_empty()
            && individual.objectives.len() == self.config.objectives.len()
    }

    /// Does `a` dominate `b`? Both vectors are in minimization space.
    fn dominates(a: &[T], b: &[T]) -> bool {
        if a.len() != b.len() || a.is_empty() {
            return false;
        }
        let mut strictly_better = false;
        for (left, right) in a.iter().zip(b.iter()) {
            if *left > *right {
                return false;
            }
            if *left < *right {
                strictly_better = true;
            }
        }
        strictly_better
    }

    /// Offer `candidate` to the external archive, keeping only the non-dominated
    /// set. Returns `true` when the candidate was accepted.
    ///
    /// The decision is taken over the whole archive *before* anything is removed:
    /// a scan that dropped members as it went and bailed out halfway through on
    /// finding a dominating member would silently lose everything it had not yet
    /// looked at.
    fn offer_to_archive(&mut self, candidate: &Individual<T>, minimized: &[T]) -> bool {
        if minimized.is_empty() {
            return false;
        }
        let member_objectives: Vec<Vec<T>> = self
            .archive
            .iter()
            .map(|member| self.to_minimization(&member.objectives))
            .collect();

        let already_covered = member_objectives
            .iter()
            .any(|member| Self::dominates(member, minimized) || member.as_slice() == minimized);
        if already_covered {
            return false;
        }

        let candidate_id = candidate.architecture.architecture_id.as_str();
        let mut index = 0usize;
        self.archive.retain(|member| {
            let dominated = Self::dominates(minimized, &member_objectives[index]);
            // Identity is by architecture id: re-evaluating the same architecture
            // must refresh its entry, not duplicate it.
            let superseded =
                !candidate_id.is_empty() && member.architecture.architecture_id == candidate_id;
            index += 1;
            !dominated && !superseded
        });
        self.archive.push(candidate.clone());
        self.truncate_archive();
        true
    }

    /// Bound the archive by repeatedly removing the member with the smallest
    /// nearest-neighbour distance in normalized objective space (density-based
    /// truncation, as in SPEA2). The removed member is always the one sitting in
    /// the most crowded region, so the extremes of the front survive.
    fn truncate_archive(&mut self) {
        while self.archive.len() > self.archive_capacity {
            let points: Vec<Vec<T>> = self
                .archive
                .iter()
                .map(|member| self.to_minimization(&member.objectives))
                .collect();
            let Some(victim) = most_crowded_index(&points) else {
                break;
            };
            self.archive.remove(victim);
        }
    }

    /// Rebuild [`Self::pareto_front`] from the archive, including real bounds and
    /// front metrics.
    fn publish_archive(&mut self) {
        let solutions: Vec<ParetoSolution<T>> = self
            .archive
            .iter()
            .map(|member| ParetoSolution {
                architecture: member.architecture.clone(),
                objectives: member.objectives.clone(),
                constraint_violations: member.constraints.clone(),
                rank: 0,
                crowding_distance: member.crowding_distance,
                metadata: SolutionMetadata {
                    id: member.id.clone(),
                    generation: self.generation,
                    evaluation_count: self.statistics.total_evaluations,
                    parents: Vec::new(),
                    creation_method: creation_method_from_id(&member.architecture.architecture_id),
                },
            })
            .collect();
        self.pareto_front.solutions = solutions;
        self.pareto_front.generation = self.generation;
        self.pareto_front.last_updated = std::time::SystemTime::now();
        self.update_objective_bounds();
        self.update_front_metrics();
    }

    fn update_objective_bounds(&mut self) {
        let Some(first) = self.pareto_front.solutions.first() else {
            return;
        };
        let num_objectives = first.objectives.len();
        let mut min_values = vec![T::infinity(); num_objectives];
        let mut max_values = vec![T::neg_infinity(); num_objectives];
        for solution in &self.pareto_front.solutions {
            for (index, value) in solution.objectives.iter().enumerate() {
                if index >= num_objectives {
                    break;
                }
                if *value < min_values[index] {
                    min_values[index] = *value;
                }
                if *value > max_values[index] {
                    max_values[index] = *value;
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

    /// The archive's objective vectors in minimization space.
    fn front_in_minimization_space(&self) -> Vec<Vec<T>> {
        let directions = objective_directions(&self.config.objectives);
        let raw: Vec<Vec<T>> = self
            .pareto_front
            .solutions
            .iter()
            .map(|solution| solution.objectives.clone())
            .collect();
        normalize_front_for_minimization(&raw, &directions)
    }

    /// Exact hypervolume of the published front against the latched reference
    /// point, derived from the first measured front when none was supplied.
    fn calculate_hypervolume(&mut self) -> T {
        let front = self.front_in_minimization_space();
        if front.is_empty() {
            return T::zero();
        }
        let directions = objective_directions(&self.config.objectives);
        if self.hypervolume_reference.is_none() {
            let Some(derived) = derive_reference_point(&front) else {
                return T::zero();
            };
            // Derived in minimization space; stored back in raw space so the public
            // setter and getter speak the caller's convention.
            self.hypervolume_reference =
                Some(normalize_reference_for_minimization(&derived, &directions));
        }
        let Some(raw_reference) = self.hypervolume_reference.as_ref() else {
            return T::zero();
        };
        let reference = normalize_reference_for_minimization(raw_reference, &directions);
        hypervolume_minimization(&front, &reference)
    }

    fn update_front_metrics(&mut self) {
        let front = self.front_in_minimization_space();
        let hypervolume = self.calculate_hypervolume();
        let previous = self.previous_hypervolume;
        self.previous_hypervolume = Some(hypervolume);

        let directions = objective_directions(&self.config.objectives);
        let reference = self
            .hypervolume_reference
            .as_ref()
            .map(|raw| normalize_reference_for_minimization(raw, &directions))
            .unwrap_or_default();
        let population: Vec<Vec<T>> = normalize_front_for_minimization(
            &self
                .population
                .iter()
                .map(|individual| individual.objectives.clone())
                .collect::<Vec<_>>(),
            &directions,
        );

        self.pareto_front.metrics = metrics::front_metrics_in_minimization_space(
            &front,
            &population,
            &reference,
            hypervolume,
            previous,
        );
        self.statistics.pareto_front_size = self.pareto_front.solutions.len();
        if hypervolume > self.statistics.best_hypervolume {
            self.statistics.best_hypervolume = hypervolume;
        }
        self.statistics.generation = self.generation;
        self.statistics
            .convergence_history
            .push(self.pareto_front.metrics.convergence);
        self.statistics
            .diversity_history
            .push(self.pareto_front.metrics.spread);
        let solved = self
            .population
            .iter()
            .filter(|individual| self.is_evaluated(individual))
            .count();
        self.statistics.algorithm_metrics.insert(
            "subproblems".to_string(),
            T::from(self.weight_vectors.len()).unwrap_or_else(T::zero),
        );
        self.statistics.algorithm_metrics.insert(
            "solved_subproblems".to_string(),
            T::from(solved).unwrap_or_else(T::zero),
        );
        self.statistics.algorithm_metrics.insert(
            "archive_size".to_string(),
            T::from(self.archive.len()).unwrap_or_else(T::zero),
        );
        self.statistics.algorithm_metrics.insert(
            "replacements".to_string(),
            T::from(self.last_replacements).unwrap_or_else(T::zero),
        );
    }

    /// Number of subproblem solutions replaced by the most recent update.
    pub fn last_replacements(&self) -> usize {
        self.last_replacements
    }

    /// The external non-dominated archive.
    pub fn archive(&self) -> &[Individual<T>] {
        &self.archive
    }

    /// Rank `results` by decomposition and return the indices of the best `k`.
    ///
    /// Each result is assigned to the subproblem it is best for; the best result
    /// per subproblem is taken first (in increasing scalarized cost), then any
    /// remaining results in increasing cost for their own subproblem. Selecting one
    /// per subproblem before doubling up is what preserves the spread MOEA/D exists
    /// to maintain.
    pub(crate) fn select_by_decomposition(
        &self,
        results: &[SearchResult<T>],
        k: usize,
    ) -> Vec<usize> {
        if results.is_empty() || k == 0 || self.weight_vectors.is_empty() {
            return Vec::new();
        }
        let minimized_results: Vec<Vec<T>> = results
            .iter()
            .map(|result| self.minimization_objectives(result))
            .collect();
        // Rank against the best point known *including* the batch being ranked, so
        // a caller that ranks before any result has been absorbed still gets a
        // meaningful ordering rather than a tie between infinities.
        let ideal = self.effective_ideal(&minimized_results);

        let mut scored: Vec<(usize, usize, T)> = Vec::with_capacity(results.len());
        for (index, minimized) in minimized_results.iter().enumerate() {
            let Some(subproblem) = self.best_fit_subproblem(minimized, &ideal) else {
                continue;
            };
            let cost = self
                .subproblem_cost(subproblem, minimized, &ideal)
                .unwrap_or_else(T::infinity);
            scored.push((index, subproblem, cost));
        }
        scored.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut taken_subproblems = std::collections::HashSet::new();
        let mut selected = Vec::with_capacity(k.min(scored.len()));
        for (index, subproblem, _) in &scored {
            if selected.len() >= k {
                break;
            }
            if taken_subproblems.insert(*subproblem) {
                selected.push(*index);
            }
        }
        for (index, _, _) in &scored {
            if selected.len() >= k {
                break;
            }
            if !selected.contains(index) {
                selected.push(*index);
            }
        }
        selected
    }

    /// Mean pairwise Euclidean distance between the objective vectors of
    /// `results` — a real diversity measurement for the engine adapter.
    pub(crate) fn mean_objective_distance(&self, results: &[SearchResult<T>]) -> f64 {
        let vectors: Vec<Vec<T>> = results
            .iter()
            .map(|result| objective_vector_for_result(&self.config.objectives, result))
            .collect();
        metrics::mean_pairwise_distance(&vectors)
    }
}

/// Index of the most crowded point, using SPEA2's truncation rule: compare each
/// point's distances to all others, sorted ascending, **lexicographically**; the
/// point with the smallest such sequence sits in the densest region and is the one
/// to drop.
///
/// Distances are measured in box-normalized objective space so no single
/// objective's scale dominates. `None` for fewer than two points.
///
/// The lexicographic tie-break matters: comparing only the single nearest
/// neighbour leaves every point of an evenly spaced front tied, so the first index
/// wins by accident and repeated truncation eats the front from one end — the
/// extreme solutions, which are exactly the ones a Pareto front must keep, are
/// destroyed first. Under the full rule an interior point always compares smaller
/// than a boundary point (its *second* neighbour is closer), so the extremes
/// survive without needing to be special-cased.
fn most_crowded_index<T: Float>(points: &[Vec<T>]) -> Option<usize> {
    if points.len() < 2 {
        return None;
    }
    let dims = points.iter().map(|point| point.len()).min()?;
    if dims == 0 {
        return None;
    }
    let mut min_values = vec![T::infinity(); dims];
    let mut max_values = vec![T::neg_infinity(); dims];
    for point in points {
        for index in 0..dims {
            if point[index] < min_values[index] {
                min_values[index] = point[index];
            }
            if point[index] > max_values[index] {
                max_values[index] = point[index];
            }
        }
    }
    let spans: Vec<T> = (0..dims)
        .map(|index| {
            let span = max_values[index] - min_values[index];
            if span > T::zero() {
                span
            } else {
                T::one()
            }
        })
        .collect();

    let sorted_distances = |i: usize| -> Vec<T> {
        let point = &points[i];
        let mut distances: Vec<T> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| {
                let mut sum = T::zero();
                for index in 0..dims {
                    let diff = (point[index] - other[index]) / spans[index];
                    sum = sum + diff * diff;
                }
                sum.sqrt()
            })
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distances
    };

    let mut worst_index = 0usize;
    let mut worst_profile = sorted_distances(0);
    for i in 1..points.len() {
        let profile = sorted_distances(i);
        let mut smaller = false;
        for (candidate, incumbent) in profile.iter().zip(worst_profile.iter()) {
            if *candidate < *incumbent {
                smaller = true;
                break;
            }
            if *candidate > *incumbent {
                break;
            }
        }
        if smaller {
            worst_index = i;
            worst_profile = profile;
        }
    }
    Some(worst_index)
}

/// Recover how an architecture was created from the prefix the variation
/// operators stamp onto its identifier.
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
    > MultiObjectiveOptimizer<T> for MOEADOptimizer<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        self.config = config.clone();
        self.generation = 0;
        self.statistics = MultiObjectiveStatistics::default();
        self.archive.clear();
        self.pareto_front = ParetoFront::default();
        self.hypervolume_reference = None;
        self.previous_hypervolume = None;
        self.last_replacements = 0;
        self.build_decomposition(config)
    }

    /// Absorb evaluated `results` through MOEA/D's scalarized replacement, update
    /// the ideal point and the external archive, and publish the archive as the
    /// Pareto front.
    fn update_pareto_front(&mut self, results: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        if self.weight_vectors.is_empty() {
            return Err(OptimError::InvalidConfig(
                "MOEA/D has no weight vectors; call initialize with a configuration \
                 that declares at least one objective first"
                    .to_string(),
            ));
        }
        self.last_replacements = 0;
        for result in results {
            let raw = objective_vector_for_result(&self.config.objectives, result);
            let minimized = self.to_minimization(&raw);
            if minimized.len() != self.config.objectives.len() {
                continue;
            }
            update_ideal_point(&mut self.ideal_point, &minimized);

            let ideal = self.effective_ideal(&[]);
            let architecture_id = result.architecture.architecture_id.clone();
            let owner = self
                .subproblem_of
                .get(&architecture_id)
                .copied()
                .or_else(|| self.best_fit_subproblem(&minimized, &ideal));
            let Some(owner) = owner else {
                continue;
            };
            // The assignment has been consumed.
            self.subproblem_of.remove(&architecture_id);

            let candidate = Individual {
                architecture: result.architecture.clone(),
                objectives: raw,
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
                id: if architecture_id.is_empty() {
                    format!("moead_{}_{}", self.generation, owner)
                } else {
                    architecture_id
                },
            };

            // Initialization: when the arriving result *is* the solution the owning
            // subproblem is already holding and that subproblem has never been
            // evaluated, this is MOEA/D's initial population evaluation — record it
            // there directly. Routing it through the competitive neighbourhood
            // replacement instead would let the shuffle spend the whole
            // `MAX_REPLACEMENTS` budget on neighbours and leave the subproblem the
            // solution was drawn for unsolved.
            let owner_holds_candidate = self.population.get(owner).is_some_and(|incumbent| {
                !candidate.architecture.architecture_id.is_empty()
                    && incumbent.architecture.architecture_id
                        == candidate.architecture.architecture_id
                    && !self.is_evaluated(incumbent)
            });
            if owner_holds_candidate {
                self.population[owner] = candidate.clone();
                self.offer_to_archive(&candidate, &minimized);
                continue;
            }

            // Scalarized replacement over a shuffled neighbourhood, bounded by
            // MAX_REPLACEMENTS so one solution cannot flood the population.
            let mut neighborhood = self
                .neighbors
                .get(owner)
                .cloned()
                .unwrap_or_else(|| vec![owner]);
            for position in (1..neighborhood.len()).rev() {
                let other = self.rng.gen_range(0..=position);
                neighborhood.swap(position, other);
            }
            let mut replacements = 0usize;
            for neighbor in neighborhood {
                if replacements >= MAX_REPLACEMENTS {
                    break;
                }
                let Some(incumbent) = self.population.get(neighbor) else {
                    continue;
                };
                let accept = if self.is_evaluated(incumbent) {
                    let incumbent_min = self.to_minimization(&incumbent.objectives);
                    match (
                        self.subproblem_cost(neighbor, &minimized, &ideal),
                        self.subproblem_cost(neighbor, &incumbent_min, &ideal),
                    ) {
                        (Some(new_cost), Some(old_cost)) => new_cost < old_cost,
                        _ => false,
                    }
                } else {
                    // A subproblem with no solution at all accepts any evaluated
                    // one: MOEA/D assumes an evaluated initial population, and this
                    // is how that population gets filled when solutions arrive from
                    // an external evaluator.
                    true
                };
                if accept {
                    self.population[neighbor] = candidate.clone();
                    replacements += 1;
                }
            }
            self.last_replacements += replacements;
            self.offer_to_archive(&candidate, &minimized);
        }
        self.generation += 1;
        self.statistics.total_evaluations += results.len();
        self.publish_archive();
        Ok(self.pareto_front.clone())
    }

    fn get_pareto_front(&self) -> &ParetoFront<T> {
        &self.pareto_front
    }

    /// Produce one offspring per subproblem by MOEA/D reproduction.
    ///
    /// Parents come from the subproblem's neighbourhood with probability
    /// [`NEIGHBORHOOD_SELECTION_PROB`] and from the whole population otherwise.
    /// Architectures passed in by the caller join the global parent pool, so an
    /// engine that maintains its own candidate population contributes material
    /// instead of being ignored. Every offspring is registered against the
    /// subproblem it was generated for, which is what lets
    /// [`MultiObjectiveOptimizer::update_pareto_front`] compare it against the
    /// right scalarization when its evaluation comes back.
    fn select_candidates(
        &mut self,
        population: &[OptimizerArchitecture<T>],
        _objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        if self.weight_vectors.is_empty() {
            return Err(OptimError::InvalidConfig(
                "MOEA/D has no weight vectors; call initialize with a configuration \
                 that declares at least one objective first"
                    .to_string(),
            ));
        }
        let subproblems = self.weight_vectors.len();
        let mut offspring = Vec::with_capacity(subproblems);
        for index in 0..subproblems {
            let use_neighborhood = self.rng.gen_range(0.0..1.0) < NEIGHBORHOOD_SELECTION_PROB;
            let pool: Vec<usize> = if use_neighborhood {
                self.neighbors
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| vec![index])
            } else {
                (0..self.population.len()).collect()
            };
            if pool.is_empty() {
                continue;
            }
            let first = pool[self.rng.gen_range(0..pool.len())];
            let second = pool[self.rng.gen_range(0..pool.len())];
            let parent1 = self
                .population
                .get(first)
                .map(|individual| individual.architecture.clone());
            // Externally supplied architectures join the pool as second parents.
            let parent2 = if !population.is_empty() && !use_neighborhood {
                Some(population[self.rng.gen_range(0..population.len())].clone())
            } else {
                self.population
                    .get(second)
                    .map(|individual| individual.architecture.clone())
            };
            let (Some(parent1), Some(parent2)) = (parent1, parent2) else {
                continue;
            };

            let mut child = if self.rng.gen_range(0.0..1.0) < MOEAD_CROSSOVER_PROB {
                operators::crossover_architectures(&mut self.rng, &parent1, &parent2)
            } else {
                parent1
            };
            if self.rng.gen_range(0.0..1.0) < MOEAD_MUTATION_PROB {
                operators::mutate_architecture(
                    &mut self.rng,
                    &mut child,
                    MOEAD_COMPONENT_MUTATION_RATE,
                    MOEAD_PARAMETER_MUTATION_RATE,
                );
            }
            self.remember_subproblem(child.architecture_id.clone(), index);
            offspring.push(child);
        }
        Ok(offspring)
    }

    fn name(&self) -> &str {
        "MOEA/D"
    }

    fn get_statistics(&self) -> MultiObjectiveStatistics<T> {
        self.statistics.clone()
    }
}
