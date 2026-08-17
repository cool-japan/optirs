//! NSGA-III — reference-direction based many-objective optimizer (Deb & Jain,
//! IEEE TEVC 2014), implemented for real.
//!
//! NSGA-III keeps NSGA-II's non-dominated sorting but replaces crowding distance,
//! which loses its discriminating power beyond two or three objectives, with
//! **niche-preserving selection against a set of reference directions**:
//!
//! 1. The population and the newly evaluated solutions are sorted into
//!    non-dominated fronts (reused verbatim from [`NSGA2`]).
//! 2. Whole fronts are accepted while they fit. The front that overflows the
//!    population is the *splitting* front.
//! 3. Everything under consideration is normalized adaptively — translated by the
//!    ideal point and scaled by the intercepts of the hyperplane through the
//!    extreme points ([`super::decomposition::normalize_by_hyperplane`]).
//! 4. Every solution is associated with its nearest reference direction by
//!    perpendicular distance.
//! 5. Members of the splitting front are taken one at a time, always from the
//!    reference direction that is currently least represented among the already
//!    selected solutions (the closest member when that direction is still empty).
//!
//! Step 5 is what makes the selection pressure uniform across the whole front
//! instead of concentrating it where solutions happen to be dense.
//!
//! This module used to contain the struct definition and nothing else — no
//! constructor, no `impl`, no algorithm — while the engine advertised
//! `MultiObjectiveAlgorithm::NSGA3` and served it with a placeholder that reported
//! an empty Pareto front.

use crate::error::{OptimError, Result};
use crate::nas_engine::{MultiObjectiveConfig, OptimizerArchitecture, SearchResult};
use scirs2_core::numeric::Float;
use scirs2_core::RngExt;
use std::fmt::Debug;

use super::core::{
    objective_directions, Individual, MultiObjectiveOptimizer, MultiObjectiveStatistics,
    ParetoFront,
};
use super::decomposition::{
    associate_with_references, das_dennis_weights, divisions_for_at_least, normalize_by_hyperplane,
};
use super::hypervolume::normalize_front_for_minimization;
use super::nsga2::NSGA2;
use super::operators;

/// Default number of reference directions requested when the caller does not say.
/// The population is sized to the lattice that supplies them.
pub const DEFAULT_REFERENCE_TARGET: usize = 20;

/// Per-component-position probability of a structural mutation.
const NSGA3_COMPONENT_MUTATION_RATE: f64 = 0.2;

/// Per-parameter probability of a numeric mutation.
const NSGA3_PARAMETER_MUTATION_RATE: f64 = 0.3;

/// NSGA-III implementation
pub struct NSGA3<T: Float + Debug + Send + Sync + 'static> {
    /// Base NSGA-II functionality: population storage, dominance sorting, Pareto
    /// front publication and the variation operators.
    pub(super) base: NSGA2<T>,
    /// Reference directions (a Das-Dennis simplex lattice)
    pub(super) reference_directions: Vec<Vec<T>>,
    /// How many solutions under consideration are associated with each reference
    /// direction, measured during the most recent selection.
    pub(super) association_count: Vec<usize>,
    /// How many *selected* solutions are associated with each reference direction
    /// (`rho` in the paper), measured during the most recent selection.
    pub(super) niche_count: Vec<usize>,
    /// Requested number of reference directions
    pub(super) reference_target: usize,
    /// Divisions of the Das-Dennis lattice actually used
    pub(super) divisions: usize,
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
    > NSGA3<T>
{
    /// Create NSGA-III with a reference-direction target, seeded from OS entropy.
    pub fn new(reference_target: usize, crossover_prob: f64, mutation_prob: f64) -> Self {
        Self::with_seed(
            reference_target,
            crossover_prob,
            mutation_prob,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Create a fully reproducible NSGA-III.
    pub fn with_seed(
        reference_target: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        seed: u64,
    ) -> Self {
        let reference_target = reference_target.max(1);
        Self {
            base: NSGA2::with_seed(reference_target, crossover_prob, mutation_prob, seed),
            reference_directions: Vec::new(),
            association_count: Vec::new(),
            niche_count: Vec::new(),
            reference_target,
            divisions: 0,
        }
    }

    /// The reference directions in use.
    pub fn reference_directions(&self) -> &[Vec<T>] {
        &self.reference_directions
    }

    /// Association counts from the most recent selection: how many solutions under
    /// consideration mapped onto each reference direction.
    pub fn association_count(&self) -> &[usize] {
        &self.association_count
    }

    /// Niche counts (`rho`) from the most recent selection: how many *selected*
    /// solutions mapped onto each reference direction.
    pub fn niche_count(&self) -> &[usize] {
        &self.niche_count
    }

    /// Population size, which NSGA-III ties to the number of reference directions.
    pub fn population_size(&self) -> usize {
        self.base.population_size
    }

    /// Pin the hypervolume reference point (raw objective space).
    pub fn set_hypervolume_reference(&mut self, reference: Vec<T>) -> Result<()> {
        self.base.set_hypervolume_reference(reference)
    }

    /// Build the reference-direction lattice and size the population to it.
    fn build_reference_directions(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        let num_objectives = config.objectives.len();
        if num_objectives == 0 {
            return Err(OptimError::InvalidConfig(
                "NSGA-III needs at least one configured objective to place reference \
                 directions"
                    .to_string(),
            ));
        }
        let (divisions, lattice_size) =
            divisions_for_at_least(num_objectives, self.reference_target).ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "no Das-Dennis lattice with at least {} directions exists for {} \
                     objectives",
                    self.reference_target, num_objectives
                ))
            })?;
        self.divisions = divisions;
        self.reference_directions = das_dennis_weights::<T>(num_objectives, divisions);
        self.association_count = vec![0; self.reference_directions.len()];
        self.niche_count = vec![0; self.reference_directions.len()];
        // NSGA-III's population size is the number of reference directions: one
        // niche per direction is what the selection aims for.
        self.base.population_size = lattice_size;
        Ok(())
    }

    /// Objective vectors of `individuals` in minimization space.
    fn minimization_objectives(&self, individuals: &[Individual<T>]) -> Vec<Vec<T>> {
        let directions = objective_directions(&self.base.config.objectives);
        let raw: Vec<Vec<T>> = individuals
            .iter()
            .map(|individual| individual.objectives.clone())
            .collect();
        normalize_front_for_minimization(&raw, &directions)
    }

    /// Niche-preserving selection (Deb & Jain, Algorithm 1).
    ///
    /// `already_selected` holds the indices accepted from the fronts that fitted
    /// whole; `splitting_front` holds the candidates competing for the remaining
    /// `slots`. Returns the indices taken from the splitting front.
    ///
    /// All indices are into `self.base.population`.
    fn niching_selection(
        &mut self,
        already_selected: &[usize],
        splitting_front: &[usize],
        slots: usize,
    ) -> Vec<usize> {
        if slots == 0 || splitting_front.is_empty() || self.reference_directions.is_empty() {
            return Vec::new();
        }
        // Normalize over everything under consideration, which is what makes the
        // association scale-free.
        let considered: Vec<usize> = already_selected
            .iter()
            .chain(splitting_front.iter())
            .copied()
            .collect();
        let individuals: Vec<Individual<T>> = considered
            .iter()
            .filter_map(|index| self.base.population.get(*index).cloned())
            .collect();
        if individuals.len() != considered.len() {
            // A stale index would silently shift every association; refuse rather
            // than select against a misaligned table.
            return splitting_front.iter().copied().take(slots).collect();
        }
        let points = self.minimization_objectives(&individuals);
        let num_objectives = self.base.config.objectives.len();
        let Some((normalized, _intercepts)) = normalize_by_hyperplane(&points, num_objectives)
        else {
            return splitting_front.iter().copied().take(slots).collect();
        };
        let associations = associate_with_references(&normalized, &self.reference_directions);

        // rho_j: selected solutions per direction; association_count: all of them.
        let mut niche_count = vec![0usize; self.reference_directions.len()];
        let mut association_count = vec![0usize; self.reference_directions.len()];
        for (position, (direction, _)) in associations.iter().enumerate() {
            if *direction < association_count.len() {
                association_count[*direction] += 1;
                if position < already_selected.len() {
                    niche_count[*direction] += 1;
                }
            }
        }

        // Candidates grouped by the direction they are associated with, carrying
        // their perpendicular distance.
        let offset = already_selected.len();
        let mut available: Vec<Vec<(usize, T)>> = vec![Vec::new(); self.reference_directions.len()];
        for (position, (direction, distance)) in associations.iter().enumerate().skip(offset) {
            let candidate = considered[position];
            if *direction < available.len() {
                available[*direction].push((candidate, *distance));
            }
        }

        let mut excluded = vec![false; self.reference_directions.len()];
        let mut chosen = Vec::with_capacity(slots);
        while chosen.len() < slots {
            // Least represented direction that still has candidates.
            let mut best_directions: Vec<usize> = Vec::new();
            let mut best_count = usize::MAX;
            for direction in 0..self.reference_directions.len() {
                if excluded[direction] || available[direction].is_empty() {
                    continue;
                }
                let count = niche_count[direction];
                if count < best_count {
                    best_count = count;
                    best_directions.clear();
                    best_directions.push(direction);
                } else if count == best_count {
                    best_directions.push(direction);
                }
            }
            if best_directions.is_empty() {
                break;
            }
            // Ties between equally represented directions are broken at random,
            // exactly as the published algorithm specifies.
            let picked = best_directions[self.base.rng.gen_range(0..best_directions.len())];

            let position = if niche_count[picked] == 0 {
                // An empty niche takes the candidate closest to its direction.
                let mut best_position = 0usize;
                let mut best_distance = T::infinity();
                for (index, (_, distance)) in available[picked].iter().enumerate() {
                    if *distance < best_distance {
                        best_distance = *distance;
                        best_position = index;
                    }
                }
                best_position
            } else {
                // An occupied niche takes an arbitrary one: the point of the rule is
                // to spread across directions, not to pack one of them tightly.
                self.base.rng.gen_range(0..available[picked].len())
            };
            let (candidate, _) = available[picked].remove(position);
            chosen.push(candidate);
            niche_count[picked] += 1;
            if available[picked].is_empty() {
                excluded[picked] = true;
            }
        }

        self.niche_count = niche_count;
        self.association_count = association_count;
        chosen
    }

    /// NSGA-III environmental selection: keep the best `population_size`
    /// individuals by front rank, resolving the splitting front by niching.
    fn environmental_selection(&mut self) -> Result<()> {
        let target = self.base.population_size.max(1);
        let fronts = self.base.non_dominated_sort();
        let mut selected: Vec<usize> = Vec::with_capacity(target);
        let mut splitting: Vec<usize> = Vec::new();
        for front in &fronts {
            let evaluated: Vec<usize> = front
                .iter()
                .copied()
                .filter(|index| {
                    self.base
                        .population
                        .get(*index)
                        .is_some_and(|individual| self.base.is_evaluated(individual))
                })
                .collect();
            if evaluated.is_empty() {
                continue;
            }
            if selected.len() + evaluated.len() <= target {
                selected.extend(evaluated);
            } else {
                splitting = evaluated;
                break;
            }
        }
        if !splitting.is_empty() && selected.len() < target {
            let slots = target - selected.len();
            let chosen = self.niching_selection(&selected, &splitting, slots);
            selected.extend(chosen);
        } else if splitting.is_empty() {
            // Nothing had to be split, so no niching happened this round; report
            // the associations of what was selected rather than leaving the counts
            // from an earlier generation in place.
            self.refresh_association_counts(&selected);
        }

        // Unevaluated individuals are kept so the optimizer still has architectures
        // to reproduce from, but they never displace an evaluated solution.
        let unevaluated: Vec<usize> = (0..self.base.population.len())
            .filter(|index| {
                self.base
                    .population
                    .get(*index)
                    .is_some_and(|individual| !self.base.is_evaluated(individual))
            })
            .collect();
        let keep_unevaluated = target.saturating_sub(selected.len());

        let mut kept: Vec<Individual<T>> = selected
            .iter()
            .filter_map(|index| self.base.population.get(*index).cloned())
            .collect();
        kept.extend(
            unevaluated
                .iter()
                .take(keep_unevaluated)
                .filter_map(|index| self.base.population.get(*index).cloned()),
        );
        self.base.population = kept;
        Ok(())
    }

    /// Recompute the association/niche tables for a selection that needed no
    /// niching, so the reported counts always describe the current population.
    fn refresh_association_counts(&mut self, selected: &[usize]) {
        let mut niche_count = vec![0usize; self.reference_directions.len()];
        let mut association_count = vec![0usize; self.reference_directions.len()];
        let individuals: Vec<Individual<T>> = selected
            .iter()
            .filter_map(|index| self.base.population.get(*index).cloned())
            .collect();
        if !individuals.is_empty() && !self.reference_directions.is_empty() {
            let points = self.minimization_objectives(&individuals);
            let num_objectives = self.base.config.objectives.len();
            if let Some((normalized, _)) = normalize_by_hyperplane(&points, num_objectives) {
                for (direction, _) in
                    associate_with_references(&normalized, &self.reference_directions)
                {
                    if direction < niche_count.len() {
                        niche_count[direction] += 1;
                        association_count[direction] += 1;
                    }
                }
            }
        }
        self.niche_count = niche_count;
        self.association_count = association_count;
    }

    /// Absorb `results` into the population, matching by architecture id.
    fn absorb_results(&mut self, results: &[SearchResult<T>]) {
        for result in results {
            let objectives = self.base.objective_vector_for_result(result);
            let architecture_id = result.architecture.architecture_id.as_str();
            let existing = if architecture_id.is_empty() {
                None
            } else {
                self.base.population.iter().position(|individual| {
                    individual.architecture.architecture_id == architecture_id
                })
            };
            match existing {
                Some(index) => {
                    self.base.population[index].objectives = objectives;
                    self.base.population[index].architecture = result.architecture.clone();
                }
                None => {
                    let id = if architecture_id.is_empty() {
                        format!("nsga3_{}", self.base.population.len())
                    } else {
                        architecture_id.to_string()
                    };
                    self.base.population.push(Individual {
                        id,
                        architecture: result.architecture.clone(),
                        objectives,
                        constraints: Vec::new(),
                        rank: 0,
                        crowding_distance: T::zero(),
                        fitness: T::zero(),
                    });
                }
            }
        }
    }

    /// Mean pairwise Euclidean distance between the objective vectors of
    /// `results` — the diversity measurement for the engine adapter.
    pub(crate) fn mean_objective_distance(&self, results: &[SearchResult<T>]) -> f64 {
        self.base.mean_objective_distance(results)
    }

    /// Rank `results` and return the indices of the best `k`, by front rank and
    /// then by reference-direction niching — NSGA-III's own selection rather than
    /// NSGA-II's crowding distance.
    pub(crate) fn select_by_rank_and_niching(
        &mut self,
        results: &[SearchResult<T>],
        k: usize,
    ) -> Vec<usize> {
        if results.is_empty() || k == 0 {
            return Vec::new();
        }
        // Selection works on a scratch population built from `results`, so the
        // returned indices are indices into `results`.
        let saved_population = std::mem::take(&mut self.base.population);
        let saved_size = self.base.population_size;
        self.base.population = results
            .iter()
            .enumerate()
            .map(|(index, result)| Individual {
                architecture: result.architecture.clone(),
                objectives: self.base.objective_vector_for_result(result),
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: T::zero(),
                fitness: T::zero(),
                id: format!("sel_{}", index),
            })
            .collect();
        self.base.population_size = k.min(results.len());

        let fronts = self.base.non_dominated_sort();
        let mut selected: Vec<usize> = Vec::with_capacity(self.base.population_size);
        let mut splitting: Vec<usize> = Vec::new();
        for front in &fronts {
            if selected.len() + front.len() <= self.base.population_size {
                selected.extend(front.iter().copied());
            } else {
                splitting = front.clone();
                break;
            }
        }
        if !splitting.is_empty() && selected.len() < self.base.population_size {
            let slots = self.base.population_size - selected.len();
            let chosen = self.niching_selection(&selected, &splitting, slots);
            selected.extend(chosen);
        }

        self.base.population = saved_population;
        self.base.population_size = saved_size;
        selected
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
    > MultiObjectiveOptimizer<T> for NSGA3<T>
{
    fn initialize(&mut self, config: &MultiObjectiveConfig<T>) -> Result<()> {
        self.build_reference_directions(config)?;
        // `build_reference_directions` has already sized the population; NSGA-II's
        // initialize then samples that many architectures.
        self.base.initialize(config)
    }

    /// Absorb `results`, run NSGA-III's environmental selection (dominance ranks
    /// resolved by reference-direction niching) and publish the non-dominated set.
    fn update_pareto_front(&mut self, results: &[SearchResult<T>]) -> Result<ParetoFront<T>> {
        if self.reference_directions.is_empty() {
            return Err(OptimError::InvalidConfig(
                "NSGA-III has no reference directions; call initialize with a \
                 configuration that declares at least one objective first"
                    .to_string(),
            ));
        }
        self.absorb_results(results);
        self.base.generation += 1;
        self.base.statistics.total_evaluations += results.len();
        self.environmental_selection()?;
        // Re-rank what survived so the published front carries correct ranks.
        let fronts = self.base.non_dominated_sort();
        for front in &fronts {
            self.base.calculate_crowding_distance(front);
        }
        self.base.update_pareto_front_from_population();
        let niched = self.niche_count.iter().filter(|count| **count > 0).count();
        self.base.statistics.algorithm_metrics.insert(
            "reference_directions".to_string(),
            T::from(self.reference_directions.len()).unwrap_or_else(T::zero),
        );
        self.base.statistics.algorithm_metrics.insert(
            "occupied_niches".to_string(),
            T::from(niched).unwrap_or_else(T::zero),
        );
        Ok(self.base.get_pareto_front().clone())
    }

    fn get_pareto_front(&self) -> &ParetoFront<T> {
        self.base.get_pareto_front()
    }

    /// Produce one offspring per population slot.
    ///
    /// NSGA-III's published mating selection is **random**, not NSGA-II's crowded
    /// tournament: diversity is enforced by the niching in environmental
    /// selection, so biasing mating by crowding would double-count it.
    fn select_candidates(
        &mut self,
        population: &[OptimizerArchitecture<T>],
        _objectives: &[T],
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        if self.base.population.is_empty() && population.is_empty() {
            return Err(OptimError::InvalidConfig(
                "NSGA-III has no population to reproduce from; call initialize first".to_string(),
            ));
        }
        let pool: Vec<OptimizerArchitecture<T>> = if self.base.population.is_empty() {
            population.to_vec()
        } else {
            self.base
                .population
                .iter()
                .map(|individual| individual.architecture.clone())
                .chain(population.iter().cloned())
                .collect()
        };
        let mut offspring = Vec::with_capacity(self.base.population_size);
        for _ in 0..self.base.population_size {
            let first = pool[self.base.rng.gen_range(0..pool.len())].clone();
            let second = pool[self.base.rng.gen_range(0..pool.len())].clone();
            let mut child = if self.base.rng.gen_range(0.0..1.0) < self.base.crossover_prob {
                operators::crossover_architectures(&mut self.base.rng, &first, &second)
            } else {
                first
            };
            if self.base.rng.gen_range(0.0..1.0) < self.base.mutation_prob {
                operators::mutate_architecture(
                    &mut self.base.rng,
                    &mut child,
                    NSGA3_COMPONENT_MUTATION_RATE,
                    NSGA3_PARAMETER_MUTATION_RATE,
                );
            }
            offspring.push(child);
        }
        Ok(offspring)
    }

    fn name(&self) -> &str {
        "NSGA-III"
    }

    fn get_statistics(&self) -> MultiObjectiveStatistics<T> {
        self.base.get_statistics()
    }
}
