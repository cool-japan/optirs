// Evolutionary search strategy using genetic algorithms

use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::nas_engine::config::ParameterRange;
use crate::nas_engine::{OptimizerArchitecture, SearchResult, SearchSpaceConfig};
use crate::EvaluationMetric;

use super::random::RandomSearch;
use super::{SearchStrategy, SearchStrategyStatistics};

/// How many populations' worth of `(architecture id -> fitness)` entries the
/// search remembers before dropping everything outside the live population.
const FITNESS_MEMO_FACTOR: usize = 64;

/// Evolutionary search strategy using genetic algorithms
pub struct EvolutionarySearch<T: Float + Debug + Send + Sync + 'static> {
    pub(crate) population: Vec<OptimizerArchitecture<T>>,
    pub population_size: usize,
    pub(crate) mutation_rate: f64,
    pub(crate) crossover_rate: f64,
    pub(crate) tournament_size: usize,
    pub(crate) generation_count: usize,
    pub(crate) statistics: SearchStrategyStatistics<T>,
    /// Whether the fittest members of the population are protected from being
    /// overwritten by a new child. See [`EvolutionarySearch::elite_count`].
    pub(crate) elite_preservation: bool,
    /// Fraction of the population protected when `elite_preservation` is set.
    pub(crate) elitism_ratio: f64,
    pub(crate) adaptive_rates: bool,
    /// Best performance observed per architecture id.
    ///
    /// Selection and replacement used to read fitness *positionally* out of the
    /// evaluation history while indexing candidates out of the population:
    /// `fitness_scores[i]` was the i-th most recent result, `population[i]` the
    /// i-th population member, and the two were related only by accident (in the
    /// common case of one-result-per-generation they were exactly reversed). So
    /// "select the fittest parent" and "replace the worst individual" both acted
    /// on architectures they had not measured. Fitness is now keyed by
    /// `architecture_id`, so a member's score always belongs to that member.
    pub(crate) fitness: HashMap<String, T>,
    /// Persistent RNG driving selection, crossover and mutation.
    ///
    /// Every random draw used to come from a freshly constructed
    /// `Random::default()`, and the initial population from a hard-coded
    /// `Random::seed(42)`, so the search was simultaneously unreproducible
    /// (per-draw entropy) and always identical at start-up. One long-lived
    /// generator makes [`EvolutionarySearch::with_seed`] mean what it says.
    pub(crate) rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    EvolutionarySearch<T>
{
    /// Create an evolutionary search seeded from OS entropy.
    pub fn new(
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
    ) -> Self {
        Self::build(
            population_size,
            mutation_rate,
            crossover_rate,
            tournament_size,
            scirs2_core::random::random::<u64>(),
        )
    }

    /// Create a fully reproducible evolutionary search.
    pub fn with_seed(
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
        seed: u64,
    ) -> Self {
        Self::build(
            population_size,
            mutation_rate,
            crossover_rate,
            tournament_size,
            seed,
        )
    }

    fn build(
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        tournament_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            population: Vec::new(),
            population_size,
            mutation_rate,
            crossover_rate,
            // A tournament of size zero would select nothing; one is the floor.
            tournament_size: tournament_size.max(1),
            generation_count: 0,
            statistics: SearchStrategyStatistics::default(),
            elite_preservation: true,
            elitism_ratio: 0.1,
            adaptive_rates: true,
            fitness: HashMap::new(),
            rng: Random::seed(seed),
        }
    }

    /// Enable or disable elite preservation and set the protected fraction.
    ///
    /// `ratio` is clamped to `[0, 1]`; see [`EvolutionarySearch::elite_count`]
    /// for how many members that actually protects.
    pub fn set_elitism(&mut self, preserve: bool, ratio: f64) {
        self.elite_preservation = preserve;
        self.elitism_ratio = if ratio.is_finite() {
            ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    /// Number of population members currently protected from replacement.
    ///
    /// Zero when elitism is off, when the ratio is zero, or when the population
    /// is too small to have both an elite and a replaceable member. Otherwise
    /// `ceil(ratio * population)`, never the entire population — something has to
    /// stay replaceable or the search cannot progress.
    pub fn elite_count(&self) -> usize {
        if !self.elite_preservation || self.population.len() < 2 {
            return 0;
        }
        let ratio = if self.elitism_ratio.is_finite() {
            self.elitism_ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if ratio <= 0.0 {
            return 0;
        }
        let raw = (ratio * self.population.len() as f64).ceil() as usize;
        raw.clamp(1, self.population.len() - 1)
    }

    /// Recorded fitness of a population member, if it has been evaluated.
    pub fn fitness_of(&self, architecture_id: &str) -> Option<T> {
        self.fitness.get(architecture_id).copied()
    }

    /// Population indices ordered best-first.
    ///
    /// Members with no recorded evaluation sort after every evaluated member —
    /// they have not proven themselves, so they are the first candidates for
    /// replacement. Ties break on architecture id so the ordering is
    /// deterministic rather than dependent on storage order.
    pub(crate) fn ranked_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.population.len()).collect();
        indices.sort_by(|&a, &b| {
            let fa = self.fitness_of(&self.population[a].architecture_id);
            let fb = self.fitness_of(&self.population[b].architecture_id);
            match (fa, fb) {
                (Some(x), Some(y)) => y
                    .partial_cmp(&x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.population[a]
                            .architecture_id
                            .cmp(&self.population[b].architecture_id)
                    }),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => self.population[a]
                    .architecture_id
                    .cmp(&self.population[b].architecture_id),
            }
        });
        indices
    }

    fn initialize_population(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        self.population.clear();
        self.fitness.clear();

        // Seed the initial population from the shared random sampler, derived from
        // this search's own generator so `with_seed` reproduces it.
        let seed = self.rng.random::<u64>();
        let mut random_search = RandomSearch::<T>::new(Some(seed));
        random_search.initialize(searchspace)?;

        for _ in 0..self.population_size {
            let architecture =
                random_search.generate_architecture(searchspace, &VecDeque::new())?;
            self.population.push(architecture);
        }

        Ok(())
    }

    /// Tournament selection over the population, using each member's *own*
    /// recorded fitness. An unevaluated member loses every comparison against an
    /// evaluated one; if nothing has been evaluated yet the tournament degenerates
    /// to a uniform draw, which is the honest answer when there is no information.
    fn selection(&mut self) -> Result<usize> {
        if self.population.is_empty() {
            return Err(OptimError::OptimizationError(
                "EvolutionarySearch cannot select a parent from an empty population".to_string(),
            ));
        }
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        let mut best_fitness = self.fitness_of(&self.population[best_idx].architecture_id);

        for _ in 1..self.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            let fitness = self.fitness_of(&self.population[idx].architecture_id);
            let better = match (fitness, best_fitness) {
                (Some(candidate), Some(current)) => candidate > current,
                (Some(_), None) => true,
                _ => false,
            };
            if better {
                best_idx = idx;
                best_fitness = fitness;
            }
        }

        Ok(best_idx)
    }

    /// Index of the population member the next child overwrites.
    ///
    /// An inverse tournament over the members elitism does not protect: sample
    /// `tournament_size` candidates from the replaceable tail of the ranking and
    /// overwrite the least fit of them. With `elite_preservation` off the whole
    /// population is replaceable, so a fit member can be evicted; with it on the
    /// top [`EvolutionarySearch::elite_count`] members never can be.
    ///
    /// # Selection pressure: a deliberate change
    ///
    /// The previous code always replaced the (mis-identified) worst individual.
    /// Deterministic worst-replacement makes `elitism_ratio` vacuous — replacing
    /// the single worst already preserves every fitter member, so no protected
    /// prefix can ever change the outcome. A tournament restores the knob at the
    /// cost of some pressure: with `tournament_size = 3` over ten replaceable
    /// members the true worst is chosen about 27% of the time rather than always,
    /// and a mid-ranked member can be evicted instead.
    ///
    /// That is the trade this implementation takes: `tournament_size` now controls
    /// replacement pressure as well as parent selection (raise it towards the
    /// population size to approach deterministic worst-replacement), and
    /// `elite_preservation` is what guarantees the top of the population survives.
    /// It is the standard steady-state pairing of tournament replacement with
    /// elitism, and it is stated here because it changes search behaviour rather
    /// than only fixing a defect.
    pub(crate) fn replacement_index(&mut self) -> Result<usize> {
        let ranked = self.ranked_indices();
        if ranked.is_empty() {
            return Err(OptimError::OptimizationError(
                "EvolutionarySearch cannot replace a member of an empty population".to_string(),
            ));
        }
        let elites = self.elite_count().min(ranked.len() - 1);
        let replaceable = &ranked[elites..];

        // `ranked` is best-first, so the largest offset into `replaceable` is the
        // least fit candidate; the inverse tournament keeps the largest offset it
        // draws.
        let mut worst_offset = self.rng.gen_range(0..replaceable.len());
        for _ in 1..self.tournament_size {
            let offset = self.rng.gen_range(0..replaceable.len());
            if offset > worst_offset {
                worst_offset = offset;
            }
        }
        Ok(replaceable[worst_offset])
    }

    fn crossover(
        &mut self,
        parent1: &OptimizerArchitecture<T>,
        parent2: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        let mut child_components = Vec::new();
        let max_len = parent1.components.len().max(parent2.components.len());

        for i in 0..max_len {
            let component = if i < parent1.components.len() && i < parent2.components.len() {
                // Crossover between components
                if self.rng.random::<f64>() < 0.5 {
                    parent1.components[i].clone()
                } else {
                    parent2.components[i].clone()
                }
            } else if i < parent1.components.len() {
                parent1.components[i].clone()
            } else {
                parent2.components[i].clone()
            };

            child_components.push(component);
        }

        // Crossover parameters
        let mut child_parameters = HashMap::new();
        for (key, value) in &parent1.parameters {
            if self.rng.random::<f64>() < 0.5 {
                child_parameters.insert(key.clone(), *value);
            } else if let Some(parent2_value) = parent2.parameters.get(key) {
                child_parameters.insert(key.clone(), *parent2_value);
            } else {
                child_parameters.insert(key.clone(), *value);
            }
        }

        Ok(OptimizerArchitecture {
            components: child_components,
            parameters: child_parameters,
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: format!("arch_{}", self.rng.random::<u64>()),
        })
    }

    fn mutate(
        &mut self,
        architecture: &mut OptimizerArchitecture<T>,
        searchspace: &SearchSpaceConfig,
    ) -> Result<()> {
        // Mutate parameters directly
        let mutation_rate = self.mutation_rate;
        for (param_name, current_value) in architecture.parameters.iter_mut() {
            if self.rng.random::<f64>() < mutation_rate {
                // Find the parameter range from search space config
                let param_range = searchspace
                    .components
                    .iter()
                    .flat_map(|c| c.hyperparameter_ranges.iter())
                    .find(|(name, _)| name == &param_name)
                    .map(|(_, range)| range);

                if let Some(param_range) = param_range {
                    match param_range {
                        ParameterRange::Continuous(min, max) => {
                            let noise = self.rng.random::<f64>() * 0.1 - 0.05;
                            let current_f64 = current_value.to_f64().unwrap_or(0.0);
                            let new_val = current_f64 + noise;
                            let clamped = new_val.clamp(*min, *max);
                            *current_value = scirs2_core::numeric::NumCast::from(clamped)
                                .unwrap_or_else(|| T::zero());
                        }
                        ParameterRange::LogUniform(min, max) => {
                            let current_f64 = current_value.to_f64().unwrap_or(0.001);
                            let log_val = current_f64.ln();
                            let noise = self.rng.random::<f64>() * 0.2 - 0.1;
                            let new_log = log_val + noise;
                            let new_val = new_log.exp().clamp(*min, *max);
                            *current_value = scirs2_core::numeric::NumCast::from(new_val)
                                .unwrap_or_else(|| T::zero());
                        }
                        _ => {
                            // For other types, regenerate randomly
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Record one evaluated result against the architecture that produced it,
    /// keeping the best score seen for that id.
    pub(crate) fn record_result(&mut self, result: &SearchResult<T>) {
        let Some(&score) = result
            .evaluation_results
            .metric_scores
            .get(&EvaluationMetric::FinalPerformance)
        else {
            return;
        };
        let id = result.architecture.architecture_id.clone();
        if id.is_empty() {
            return;
        }
        self.fitness
            .entry(id)
            .and_modify(|best| {
                if score > *best {
                    *best = score;
                }
            })
            .or_insert(score);
    }

    /// Absorb every result in `history` that this search has not already recorded.
    fn record_history(&mut self, history: &VecDeque<SearchResult<T>>) {
        for result in history {
            self.record_result(result);
        }
        self.prune_fitness();
    }

    /// Bound the fitness memo.
    ///
    /// The memo keeps a score for every architecture ever evaluated, including
    /// ones long since bred out of the population, because the caller may replay
    /// an old result. Once it grows past [`FITNESS_MEMO_FACTOR`] populations'
    /// worth of entries, everything that is not currently in the population is
    /// dropped: those scores can no longer influence selection or replacement.
    fn prune_fitness(&mut self) {
        let cap = FITNESS_MEMO_FACTOR.saturating_mul(self.population_size.max(1));
        if self.fitness.len() <= cap {
            return;
        }
        let live: std::collections::HashSet<String> = self
            .population
            .iter()
            .map(|member| member.architecture_id.clone())
            .collect();
        self.fitness.retain(|id, _| live.contains(id));
    }

    fn calculate_recent_improvement(&self, performances: &[T]) -> T {
        if performances.len() < 10 {
            return T::zero();
        }

        let recent_avg = performances.iter().rev().take(5).cloned().sum::<T>()
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());
        let earlier_avg = performances
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .cloned()
            .sum::<T>()
            / scirs2_core::numeric::NumCast::from(5.0).unwrap_or_else(|| T::zero());

        recent_avg - earlier_avg
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + std::fmt::Debug + std::iter::Sum>
    SearchStrategy<T> for EvolutionarySearch<T>
{
    fn initialize(&mut self, searchspace: &SearchSpaceConfig) -> Result<()> {
        self.initialize_population(searchspace)?;
        self.generation_count = 0;
        Ok(())
    }

    fn generate_architecture(
        &mut self,
        searchspace: &SearchSpaceConfig,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<OptimizerArchitecture<T>> {
        if self.population.is_empty() {
            self.initialize_population(searchspace)?;
        }
        if self.population.is_empty() {
            return Err(OptimError::SearchSpaceError(
                "EvolutionarySearch produced an empty population; the search space \
                 must allow at least one architecture"
                    .to_string(),
            ));
        }

        // Absorb any fitness the caller has observed but not yet pushed through
        // `update_with_results`, keyed by architecture id.
        self.record_history(history);

        // Breed only once at least one population member has a measured fitness:
        // before that, selection has nothing to select on and the honest answer is
        // to hand back an unevaluated member of the initial population.
        let evaluated = self
            .population
            .iter()
            .any(|member| self.fitness.contains_key(&member.architecture_id));
        if !evaluated {
            let idx = self.rng.gen_range(0..self.population.len());
            self.statistics.total_architectures_generated += 1;
            return Ok(self.population[idx].clone());
        }

        let parent1_idx = self.selection()?;
        let parent2_idx = self.selection()?;
        let parent1 = self.population[parent1_idx].clone();
        let parent2 = self.population[parent2_idx].clone();

        let mut child = if self.rng.random::<f64>() < self.crossover_rate {
            self.crossover(&parent1, &parent2)?
        } else {
            parent1
        };

        self.mutate(&mut child, searchspace)?;

        // Every generated candidate gets its own identity. Without crossover the
        // child is a *mutated copy* of a parent, and it used to keep the parent's
        // `architecture_id`: two different architectures then shared one id, so the
        // parent's measured fitness was silently transferred to the mutant and a
        // later evaluation of either overwrote the other's score.
        child.architecture_id = format!("arch_{}", self.rng.random::<u64>());

        // Replace a member elitism does not protect. The old code replaced
        // `population[worst_idx]` where `worst_idx` indexed the *history*, so the
        // architecture it discarded was not the one whose score it had inspected.
        let victim = self.replacement_index()?;
        self.population[victim] = child.clone();
        self.generation_count += 1;

        self.statistics.total_architectures_generated += 1;
        Ok(child)
    }

    fn update_with_results(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        if !results.is_empty() {
            // Attribution is by architecture id, so a member's score always
            // belongs to that member.
            for result in results {
                self.record_result(result);
            }
            self.prune_fitness();

            let performances: Vec<T> = results
                .iter()
                .filter_map(|r| {
                    r.evaluation_results
                        .metric_scores
                        .get(&EvaluationMetric::FinalPerformance)
                })
                .cloned()
                .collect();

            if !performances.is_empty() {
                self.statistics.best_performance = performances
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .cloned()
                    .unwrap_or(T::zero());

                let sum: T = performances.iter().cloned().sum();
                // `T::from(len)` cannot realistically fail for a float, but a
                // panicking `.expect` here would abort a whole search over a
                // statistics field; report the unaveraged sum's scale instead.
                let count: T = scirs2_core::numeric::NumCast::from(performances.len() as f64)
                    .unwrap_or_else(T::one);
                self.statistics.average_performance = if count > T::zero() {
                    sum / count
                } else {
                    T::zero()
                };

                // Adaptive rate adjustment
                if self.adaptive_rates && self.generation_count > 10 {
                    let recent_improvement = self.calculate_recent_improvement(&performances);
                    if recent_improvement
                        < scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero())
                    {
                        self.mutation_rate = (self.mutation_rate * 1.1).min(0.5);
                    } else {
                        self.mutation_rate = (self.mutation_rate * 0.95).max(0.01);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "EvolutionarySearch"
    }

    fn get_statistics(&self) -> SearchStrategyStatistics<T> {
        let mut stats = self.statistics.clone();
        stats.exploration_rate =
            scirs2_core::numeric::NumCast::from(self.mutation_rate).unwrap_or_else(|| T::zero());
        stats.exploitation_rate = scirs2_core::numeric::NumCast::from(1.0 - self.mutation_rate)
            .unwrap_or_else(|| T::zero());
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::config::{
        ComponentType as ConfigComponentType, OptimizerComponentConfig,
    };
    use crate::nas_engine::results::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };

    fn search_space() -> SearchSpaceConfig {
        SearchSpaceConfig {
            components: vec![
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::Adam,
                    hyperparameter_ranges: {
                        let mut ranges = HashMap::new();
                        ranges.insert(
                            "learning_rate".to_string(),
                            ParameterRange::LogUniform(1e-4, 1e-1),
                        );
                        ranges
                    },
                    complexity_score: 1.0,
                    memory_requirement: 1024,
                    computational_cost: 1.0,
                    compatibility_constraints: Vec::new(),
                },
                OptimizerComponentConfig {
                    component_type: ConfigComponentType::SGD,
                    hyperparameter_ranges: {
                        let mut ranges = HashMap::new();
                        ranges.insert(
                            "momentum".to_string(),
                            ParameterRange::Continuous(0.5, 0.99),
                        );
                        ranges
                    },
                    complexity_score: 0.5,
                    memory_requirement: 512,
                    computational_cost: 0.5,
                    compatibility_constraints: Vec::new(),
                },
            ],
            min_components: 1,
            max_components: 3,
            ..SearchSpaceConfig::default()
        }
    }

    fn result_for(id: &str, score: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, score);
        SearchResult {
            architecture: OptimizerArchitecture {
                architecture_id: id.to_string(),
                components: vec!["Adam".to_string()],
                parameters: HashMap::new(),
                connections: Vec::new(),
                metadata: HashMap::new(),
                hyperparameters: HashMap::new(),
            },
            evaluation_results: EvaluationResults {
                metric_scores,
                overall_score: score,
                confidence_intervals: HashMap::new(),
                evaluation_time: std::time::Duration::from_secs(0),
                success: true,
                error_message: None,
                cv_results: None,
                benchmark_results: HashMap::new(),
                training_trajectory: Vec::new(),
            },
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

    /// Give every population member a distinct fitness, best first in population
    /// order, so the expected ranking is known exactly.
    fn score_population(search: &mut EvolutionarySearch<f64>) {
        let ids: Vec<String> = search
            .population
            .iter()
            .map(|member| member.architecture_id.clone())
            .collect();
        for (rank, id) in ids.iter().enumerate() {
            let result = result_for(id, 1.0 - rank as f64 * 0.01);
            search.record_result(&result);
        }
    }

    fn seeded(population_size: usize, seed: u64) -> EvolutionarySearch<f64> {
        let mut search = EvolutionarySearch::<f64>::with_seed(population_size, 0.2, 0.8, 3, seed);
        search
            .initialize(&search_space())
            .expect("initialize the population");
        search
    }

    #[test]
    fn fitness_is_attributed_by_architecture_id() {
        let mut search = seeded(4, 11);
        let id = search.population[2].architecture_id.clone();

        assert!(search.fitness_of(&id).is_none());
        search.record_result(&result_for(&id, 0.42));
        assert_eq!(search.fitness_of(&id), Some(0.42));

        // Only the best score for an id is kept, and a worse re-evaluation of the
        // same architecture does not overwrite it.
        search.record_result(&result_for(&id, 0.10));
        assert_eq!(search.fitness_of(&id), Some(0.42));
        search.record_result(&result_for(&id, 0.90));
        assert_eq!(search.fitness_of(&id), Some(0.90));

        // A result for an architecture this search never held is recorded under its
        // own id and credited to nobody in the population.
        search.record_result(&result_for("not-in-population", 1.0));
        assert_eq!(search.fitness_of("not-in-population"), Some(1.0));
        for member in &search.population {
            if member.architecture_id != id {
                assert!(search.fitness_of(&member.architecture_id).is_none());
            }
        }
    }

    #[test]
    fn ranking_puts_the_fittest_first_and_the_unevaluated_last() {
        let mut search = seeded(5, 12);
        let ids: Vec<String> = search
            .population
            .iter()
            .map(|m| m.architecture_id.clone())
            .collect();
        // Score only members 1 and 3.
        search.record_result(&result_for(&ids[1], 0.2));
        search.record_result(&result_for(&ids[3], 0.9));

        let ranked = search.ranked_indices();
        assert_eq!(ranked[0], 3, "the best evaluated member must rank first");
        assert_eq!(ranked[1], 1, "the other evaluated member must rank second");
        // The three unevaluated members fill the tail, in a deterministic order.
        let tail: Vec<usize> = ranked[2..].to_vec();
        assert_eq!(tail.len(), 3);
        assert!(tail.iter().all(|&i| i != 1 && i != 3));
        assert_eq!(tail, search.ranked_indices()[2..].to_vec());
    }

    #[test]
    fn elite_count_protects_a_fraction_but_never_the_whole_population() {
        let mut search = seeded(10, 13);

        search.set_elitism(true, 0.1);
        assert_eq!(search.elite_count(), 1);

        search.set_elitism(true, 0.35);
        assert_eq!(search.elite_count(), 4);

        // A ratio of one would leave nothing replaceable, so one slot always stays.
        search.set_elitism(true, 1.0);
        assert_eq!(search.elite_count(), 9);

        search.set_elitism(true, 0.0);
        assert_eq!(search.elite_count(), 0);

        search.set_elitism(false, 0.5);
        assert_eq!(search.elite_count(), 0);

        // Out-of-range and non-finite ratios are clamped, not propagated.
        search.set_elitism(true, f64::NAN);
        assert_eq!(search.elite_count(), 0);
        search.set_elitism(true, 5.0);
        assert_eq!(search.elite_count(), 9);

        // A population too small to have both an elite and a victim protects none.
        let mut tiny = seeded(1, 14);
        tiny.set_elitism(true, 0.5);
        assert_eq!(tiny.elite_count(), 0);
    }

    #[test]
    fn elitism_keeps_the_fittest_members_out_of_the_replacement_pool() {
        let mut search = seeded(10, 15);
        score_population(&mut search);
        search.set_elitism(true, 0.3);

        let protected: Vec<usize> = search.ranked_indices()[..search.elite_count()].to_vec();
        assert_eq!(protected.len(), 3);

        for _ in 0..500 {
            let victim = search.replacement_index().expect("a victim must exist");
            assert!(
                !protected.contains(&victim),
                "elitism must never offer a protected member for replacement"
            );
        }

        // With elitism off every member is replaceable, including the best one.
        // A tournament of one makes the draw uniform, so 500 attempts miss the best
        // member with probability 0.9^500.
        search.set_elitism(false, 0.0);
        search.tournament_size = 1;
        let best = search.ranked_indices()[0];
        let mut saw_best = false;
        for _ in 0..500 {
            if search.replacement_index().expect("a victim must exist") == best {
                saw_best = true;
                break;
            }
        }
        assert!(
            saw_best,
            "without elitism the fittest member must be reachable for replacement"
        );
    }

    #[test]
    fn tournament_size_controls_replacement_pressure() {
        // Documented behaviour: a wide tournament approaches deterministic
        // worst-replacement, a tournament of one is uniform. Both halves are
        // asserted so a future change to the operator cannot pass silently.
        let mut search = seeded(10, 21);
        score_population(&mut search);
        search.set_elitism(false, 0.0);
        let worst = *search
            .ranked_indices()
            .last()
            .expect("a non-empty population has a worst member");

        search.tournament_size = 10;
        let wide = (0..400)
            .filter(|_| search.replacement_index().expect("victim") == worst)
            .count();

        search.tournament_size = 1;
        let uniform = (0..400)
            .filter(|_| search.replacement_index().expect("victim") == worst)
            .count();

        assert!(
            wide > uniform,
            "a wider tournament must concentrate replacement on the worst member: \
             {wide} vs {uniform}"
        );
        assert!(
            wide > 200,
            "a tournament as wide as the population should usually pick the worst: {wide}/400"
        );
        assert!(
            uniform < 120,
            "a tournament of one must be roughly uniform over 10 members: {uniform}/400"
        );
    }

    #[test]
    fn selection_prefers_the_member_with_the_better_recorded_score() {
        let mut search = seeded(2, 16);
        let ids: Vec<String> = search
            .population
            .iter()
            .map(|m| m.architecture_id.clone())
            .collect();
        search.record_result(&result_for(&ids[0], 0.1));
        search.record_result(&result_for(&ids[1], 0.9));

        // A tournament wide enough to see both members must pick the better one
        // essentially always; require a clear majority rather than certainty.
        search.tournament_size = 8;
        let mut picked_better = 0;
        for _ in 0..200 {
            if search.selection().expect("selection must succeed") == 1 {
                picked_better += 1;
            }
        }
        assert!(
            picked_better > 190,
            "tournament selection ignored recorded fitness: {picked_better}/200"
        );
    }

    #[test]
    fn an_evaluated_generation_breeds_and_replaces_by_id() {
        let space = search_space();
        let mut search = seeded(6, 17);
        score_population(&mut search);
        search.set_elitism(true, 0.5);

        let protected: Vec<String> = search.ranked_indices()[..search.elite_count()]
            .iter()
            .map(|&i| search.population[i].architecture_id.clone())
            .collect();
        assert_eq!(protected.len(), 3);

        for _ in 0..30 {
            let child = search
                .generate_architecture(&space, &VecDeque::new())
                .expect("generation must succeed");
            // The child really entered the population.
            assert!(search
                .population
                .iter()
                .any(|member| member.architecture_id == child.architecture_id));
            // and never at the cost of a protected member.
            for id in &protected {
                assert!(
                    search
                        .population
                        .iter()
                        .any(|member| &member.architecture_id == id),
                    "protected architecture {id} was evicted"
                );
            }
        }
        assert_eq!(search.generation_count, 30);
    }

    #[test]
    fn an_unevaluated_population_hands_back_a_member_instead_of_breeding() {
        let space = search_space();
        let mut search = seeded(4, 18);

        let architecture = search
            .generate_architecture(&space, &VecDeque::new())
            .expect("generation must succeed");
        assert!(
            search
                .population
                .iter()
                .any(|member| member.architecture_id == architecture.architecture_id),
            "with no measured fitness the strategy must return an existing member"
        );
        assert_eq!(
            search.generation_count, 0,
            "no generation has completed without a single evaluation"
        );
    }

    #[test]
    fn history_results_are_absorbed_even_without_an_explicit_update() {
        let space = search_space();
        let mut search = seeded(4, 19);
        let id = search.population[0].architecture_id.clone();

        let mut history = VecDeque::new();
        history.push_back(result_for(&id, 0.75));

        let _ = search
            .generate_architecture(&space, &history)
            .expect("generation must succeed");
        assert_eq!(search.fitness_of(&id), Some(0.75));
    }

    #[test]
    fn seeding_is_reproducible_and_unseeded_searches_differ() {
        let space = search_space();
        let mut a = seeded(6, 0xA11CE);
        let mut b = seeded(6, 0xA11CE);
        score_population(&mut a);
        score_population(&mut b);

        for _ in 0..5 {
            let left = a
                .generate_architecture(&space, &VecDeque::new())
                .expect("generation must succeed");
            let right = b
                .generate_architecture(&space, &VecDeque::new())
                .expect("generation must succeed");
            assert_eq!(left.components, right.components);
            assert_eq!(left.architecture_id, right.architecture_id);
        }

        // The initial population used to come from a hard-coded `Random::seed(42)`,
        // so two independent searches started from byte-identical members.
        let first = EvolutionarySearch::<f64>::new(6, 0.2, 0.8, 3);
        let second = EvolutionarySearch::<f64>::new(6, 0.2, 0.8, 3);
        let mut first = first;
        let mut second = second;
        first.initialize(&space).expect("initialize");
        second.initialize(&space).expect("initialize");
        let ids_first: Vec<&String> = first
            .population
            .iter()
            .map(|m| &m.architecture_id)
            .collect();
        let ids_second: Vec<&String> = second
            .population
            .iter()
            .map(|m| &m.architecture_id)
            .collect();
        assert_ne!(ids_first, ids_second);
    }

    #[test]
    fn statistics_survive_an_empty_metric_set() {
        let mut search = seeded(3, 20);
        let mut result = result_for("no-metrics", 0.0);
        result.evaluation_results.metric_scores.clear();
        search
            .update_with_results(std::slice::from_ref(&result))
            .expect("update must succeed");
        assert!(search.fitness_of("no-metrics").is_none());
        assert_eq!(search.get_statistics().average_performance, 0.0);
    }
}
