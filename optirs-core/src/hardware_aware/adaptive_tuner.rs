// Adaptive tuning search for the hardware-aware optimizer.
//
// [`TuningStrategy`] used to be a pure description: the tuner stored a
// performance target and nothing else, so selecting `GridSearch` tuned exactly
// as much as selecting nothing. This module supplies the search loop behind the
// strategy — a real evaluator-driven search over registered parameter ranges
// that records every measurement in `tuning_history` and leaves the best
// candidate in `current_params`.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use scirs2_core::numeric::Float;
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};

use crate::error::{OptimError, Result};
use crate::utils::{scalar_or, total_order, try_f64};

/// Evaluations a single [`AdaptiveTuner::tune`] call performs before giving up.
const DEFAULT_MAX_EVALUATIONS: usize = 256;

/// Measurements kept in [`AdaptiveTuner::tuning_history`].
const MAX_TUNING_HISTORY: usize = 1000;

/// Smallest hill-climbing step, as a fraction of a parameter's range.
///
/// Greedy search halves its step whenever a round finds no improvement; below
/// this fraction the candidates it generates are indistinguishable from the
/// incumbent for any realistic objective, so the search stops.
const MIN_STEP_FRACTION: f64 = 1e-4;

/// Seed the genetic search uses unless the caller sets another one.
///
/// The seed is fixed rather than drawn from the environment so a tuning run is
/// reproducible: two runs of the same evaluator over the same search space must
/// produce the same recommendation, or the tuning history cannot be compared
/// across runs.
pub const DEFAULT_TUNING_SEED: u64 = 0x0071_7250_5f74_756e;

/// Candidates drawn per tournament in the genetic search.
const TOURNAMENT_SIZE: usize = 2;

/// BLX-alpha crossover width: offspring are drawn from the interval spanned by
/// the two parents, widened by this fraction at each end (Eshelman & Schaffer,
/// "Real-Coded Genetic Algorithms and Interval-Schemata", FOGA 1993).
const CROSSOVER_ALPHA: f64 = 0.5;

/// Probability that a given coordinate of an offspring is mutated.
const MUTATION_PROBABILITY: f64 = 0.2;

/// Mutation magnitude, as a fraction of the parameter's range.
const MUTATION_SCALE: f64 = 0.1;

/// Tuning record for adaptive optimization
#[derive(Debug, Clone)]
pub struct TuningRecord<A: Float> {
    /// Tuning parameters used
    pub parameters: HashMap<String, A>,
    /// Performance achieved
    pub performance: A,
    /// Resource consumption
    pub resource_usage: A,
    /// Timestamp
    pub timestamp: u64,
}

/// Tuning strategies
#[derive(Debug, Clone)]
pub enum TuningStrategy {
    /// Grid search over parameter space
    GridSearch {
        /// Grid search resolution
        resolution: usize,
    },
    /// Greedy coordinate-wise hill climbing
    Greedy {
        /// Initial step, as a fraction of each parameter's range
        step_fraction: f64,
        /// Maximum number of sweeps over the parameters
        max_rounds: usize,
    },
    /// Bayesian optimization
    BayesianOptimization {
        /// Number of samples
        num_samples: usize,
    },
    /// Genetic algorithm
    GeneticAlgorithm {
        /// Population size
        population_size: usize,
        /// Number of generations
        generations: usize,
    },
    /// Reinforcement learning based
    ReinforcementLearning {
        /// Exploration rate
        exploration_rate: f64,
    },
}

/// A parameter the tuner is allowed to move, and the range it may move it in.
#[derive(Debug, Clone)]
pub struct TunableParameter<A: Float> {
    /// Name the evaluator will see in the candidate map.
    pub name: String,
    /// Inclusive lower bound.
    pub minimum: A,
    /// Inclusive upper bound.
    pub maximum: A,
}

impl<A: Float> TunableParameter<A> {
    /// Register a parameter over `[minimum, maximum]`.
    ///
    /// A reversed or non-finite range is rejected: every search in this module
    /// interpolates inside the range, so an invalid one would silently produce
    /// candidates the caller never authorised.
    pub fn new(name: impl Into<String>, minimum: A, maximum: A) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(OptimError::InvalidConfig(
                "a tunable parameter needs a non-empty name".to_string(),
            ));
        }
        if !minimum.is_finite() || !maximum.is_finite() {
            return Err(OptimError::InvalidConfig(format!(
                "tunable parameter '{name}' has a non-finite bound"
            )));
        }
        if minimum > maximum {
            return Err(OptimError::InvalidConfig(format!(
                "tunable parameter '{name}' has minimum > maximum"
            )));
        }
        Ok(Self {
            name,
            minimum,
            maximum,
        })
    }

    /// Width of the range.
    pub fn span(&self) -> A {
        self.maximum - self.minimum
    }

    /// Midpoint of the range, used as the starting point when the caller has
    /// not supplied one.
    pub fn midpoint(&self) -> A {
        self.minimum + self.span() / (A::one() + A::one())
    }

    /// Clamp a value into the range.
    pub fn clamp(&self, value: A) -> A {
        if value < self.minimum {
            self.minimum
        } else if value > self.maximum {
            self.maximum
        } else {
            value
        }
    }
}

/// What the evaluator reports back for one candidate parameter set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuningObservation<A: Float> {
    /// Measured performance. **Higher is better** — the tuner maximizes this
    /// and compares it against [`AdaptiveTuner::performance_target`], so a
    /// loss-like metric must be negated by the evaluator.
    pub performance: A,
    /// Measured resource consumption for the same candidate, recorded in the
    /// tuning history so a caller can trade performance against cost after the
    /// fact.
    pub resource_usage: A,
}

/// Result of one [`AdaptiveTuner::tune`] call.
#[derive(Debug, Clone)]
pub struct TuningOutcome<A: Float> {
    /// Best candidate found, also left in [`AdaptiveTuner::current_params`].
    pub best_parameters: HashMap<String, A>,
    /// Performance measured for `best_parameters`.
    pub best_performance: A,
    /// Number of candidates evaluated during this call.
    pub evaluations: usize,
    /// Whether the search stopped because it reached the performance target.
    pub target_reached: bool,
}

/// Adaptive tuner for dynamic optimization
#[derive(Debug)]
pub struct AdaptiveTuner<A: Float> {
    /// Performance target
    performance_target: A,
    /// Search strategy used by [`AdaptiveTuner::tune`]
    strategy: TuningStrategy,
    /// Parameters the search may move, in registration order (which is what
    /// makes every search in this module deterministic)
    parameters: Vec<TunableParameter<A>>,
    /// Best candidate found so far
    current_params: HashMap<String, A>,
    /// Every measurement taken, most recent last
    tuning_history: Vec<TuningRecord<A>>,
    /// Performance of `current_params`, if anything has been measured
    best_performance: Option<A>,
    /// Evaluation budget for a single `tune` call
    max_evaluations: usize,
    /// Lifetime evaluation count, kept separately from `tuning_history` because
    /// the history is bounded
    total_evaluations: usize,
    /// Seed for the stochastic searches, so a tuning run is reproducible
    random_seed: u64,
}

impl<A: Float + Send + Sync> Default for AdaptiveTuner<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Send + Sync> AdaptiveTuner<A> {
    /// Create a new adaptive tuner.
    ///
    /// The default strategy is greedy hill climbing, which needs no search-space
    /// budget beyond the parameters themselves; register those with
    /// [`AdaptiveTuner::add_parameter`] before calling [`AdaptiveTuner::tune`].
    pub fn new() -> Self {
        Self {
            performance_target: crate::utils::scalar_or(100.0, A::zero()),
            strategy: TuningStrategy::Greedy {
                step_fraction: 0.25,
                max_rounds: 8,
            },
            parameters: Vec::new(),
            current_params: HashMap::new(),
            tuning_history: Vec::new(),
            best_performance: None,
            max_evaluations: DEFAULT_MAX_EVALUATIONS,
            total_evaluations: 0,
            random_seed: DEFAULT_TUNING_SEED,
        }
    }

    /// Seed the stochastic searches draw from.
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    /// Set the seed the stochastic searches draw from.
    ///
    /// Two [`AdaptiveTuner::tune`] calls with the same seed, search space and
    /// evaluator produce the same sequence of candidates.
    pub fn set_random_seed(&mut self, seed: u64) {
        self.random_seed = seed;
    }

    /// Performance the search stops at once reached.
    pub fn performance_target(&self) -> A {
        self.performance_target
    }

    /// Set the performance target.
    pub fn set_performance_target(&mut self, target: A) {
        self.performance_target = target;
    }

    /// Search strategy in use.
    pub fn strategy(&self) -> &TuningStrategy {
        &self.strategy
    }

    /// Select the search strategy.
    pub fn set_strategy(&mut self, strategy: TuningStrategy) {
        self.strategy = strategy;
    }

    /// Register a parameter the search may move.
    pub fn add_parameter(&mut self, parameter: TunableParameter<A>) -> Result<()> {
        if self.parameters.iter().any(|p| p.name == parameter.name) {
            return Err(OptimError::InvalidConfig(format!(
                "tunable parameter '{}' is already registered",
                parameter.name
            )));
        }
        self.parameters.push(parameter);
        Ok(())
    }

    /// Parameters the search may move.
    pub fn parameters(&self) -> &[TunableParameter<A>] {
        &self.parameters
    }

    /// Best candidate found so far.
    pub fn current_params(&self) -> &HashMap<String, A> {
        &self.current_params
    }

    /// Performance measured for [`AdaptiveTuner::current_params`].
    pub fn best_performance(&self) -> Option<A> {
        self.best_performance
    }

    /// Every measurement taken so far, oldest first, bounded to the most recent
    /// `MAX_TUNING_HISTORY` entries.
    pub fn tuning_history(&self) -> &[TuningRecord<A>] {
        &self.tuning_history
    }

    /// Total candidates evaluated over the tuner's lifetime.
    pub fn total_evaluations(&self) -> usize {
        self.total_evaluations
    }

    /// Evaluation budget for a single [`AdaptiveTuner::tune`] call.
    pub fn max_evaluations(&self) -> usize {
        self.max_evaluations
    }

    /// Set the evaluation budget for a single [`AdaptiveTuner::tune`] call.
    pub fn set_max_evaluations(&mut self, max_evaluations: usize) {
        self.max_evaluations = max_evaluations.max(1);
    }

    /// Run the search behind the configured [`TuningStrategy`].
    ///
    /// `evaluate` measures one candidate parameter set; it is called at most
    /// [`AdaptiveTuner::max_evaluations`] times. The search maximizes
    /// [`TuningObservation::performance`] and stops early once a candidate
    /// reaches [`AdaptiveTuner::performance_target`].
    ///
    /// The previous best candidate is used as the starting point for greedy
    /// search, but the recorded best *performance* is reset first: the objective
    /// a caller measures now (a different workload, a different platform) is not
    /// comparable with one measured before, and keeping the old number would
    /// silently veto every new measurement.
    pub fn tune<F>(&mut self, mut evaluate: F) -> Result<TuningOutcome<A>>
    where
        F: FnMut(&HashMap<String, A>) -> Result<TuningObservation<A>>,
    {
        if self.parameters.is_empty() {
            return Err(OptimError::InvalidConfig(
                "adaptive tuning needs at least one tunable parameter; register \
                 one with AdaptiveTuner::add_parameter"
                    .to_string(),
            ));
        }

        self.best_performance = None;
        let mut budget = self.max_evaluations;
        let evaluations_before = self.total_evaluations;

        let strategy = self.strategy.clone();
        let target_reached = match strategy {
            TuningStrategy::GridSearch { resolution } => {
                self.tune_grid_search(resolution, &mut budget, &mut evaluate)?
            }
            TuningStrategy::Greedy {
                step_fraction,
                max_rounds,
            } => self.tune_greedy(step_fraction, max_rounds, &mut budget, &mut evaluate)?,
            TuningStrategy::BayesianOptimization { .. } => {
                return Err(OptimError::UnsupportedOperation(
                    "TuningStrategy::BayesianOptimization needs a surrogate model \
                     (a Gaussian process over the tuning history) and an acquisition \
                     optimizer, neither of which optirs-core provides; use GridSearch \
                     or Greedy"
                        .to_string(),
                ));
            }
            TuningStrategy::GeneticAlgorithm {
                population_size,
                generations,
            } => self.tune_genetic(population_size, generations, &mut budget, &mut evaluate)?,
            TuningStrategy::ReinforcementLearning { .. } => {
                return Err(OptimError::UnsupportedOperation(
                    "TuningStrategy::ReinforcementLearning needs an environment model \
                     and a policy to train against it, neither of which optirs-core \
                     provides; use GridSearch or Greedy"
                        .to_string(),
                ));
            }
        };

        match self.best_performance {
            Some(best_performance) => Ok(TuningOutcome {
                best_parameters: self.current_params.clone(),
                best_performance,
                evaluations: self.total_evaluations - evaluations_before,
                target_reached,
            }),
            None => Err(OptimError::InvalidConfig(
                "adaptive tuning evaluated no candidate: the evaluation budget is \
                 exhausted before the first measurement"
                    .to_string(),
            )),
        }
    }

    /// Exhaustive search over an evenly spaced grid.
    ///
    /// Returns whether the target was reached.
    fn tune_grid_search<F>(
        &mut self,
        resolution: usize,
        budget: &mut usize,
        evaluate: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(&HashMap<String, A>) -> Result<TuningObservation<A>>,
    {
        if resolution == 0 {
            return Err(OptimError::InvalidConfig(
                "TuningStrategy::GridSearch needs a resolution of at least 1".to_string(),
            ));
        }

        // Reject an oversized grid instead of silently exploring a truncated
        // prefix of it, which would not be a grid search at all.
        let mut total: usize = 1;
        for _ in 0..self.parameters.len() {
            total = total.checked_mul(resolution).ok_or_else(|| {
                OptimError::InvalidConfig(format!(
                    "grid search over {} parameters at resolution {resolution} overflows",
                    self.parameters.len()
                ))
            })?;
        }
        if total > *budget {
            return Err(OptimError::InvalidConfig(format!(
                "grid search over {} parameters at resolution {resolution} needs \
                 {total} evaluations but the budget is {budget}; lower the \
                 resolution or raise it with set_max_evaluations",
                self.parameters.len()
            )));
        }

        let axes: Vec<Vec<A>> = self
            .parameters
            .iter()
            .map(|parameter| grid_axis(parameter, resolution))
            .collect();

        for index in 0..total {
            let mut candidate = HashMap::with_capacity(self.parameters.len());
            let mut remaining = index;
            for (parameter, axis) in self.parameters.iter().zip(axes.iter()) {
                let position = remaining % axis.len();
                remaining /= axis.len();
                candidate.insert(parameter.name.clone(), axis[position]);
            }

            let performance = self.evaluate_candidate(&candidate, budget, evaluate)?;
            match performance {
                None => return Ok(false),
                Some(performance) if performance >= self.performance_target => return Ok(true),
                Some(_) => {}
            }
        }

        Ok(false)
    }

    /// Coordinate-wise hill climbing with a step that halves on a failed sweep.
    ///
    /// Returns whether the target was reached.
    fn tune_greedy<F>(
        &mut self,
        step_fraction: f64,
        max_rounds: usize,
        budget: &mut usize,
        evaluate: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(&HashMap<String, A>) -> Result<TuningObservation<A>>,
    {
        if !step_fraction.is_finite() || step_fraction <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "TuningStrategy::Greedy needs a positive, finite step_fraction".to_string(),
            ));
        }

        let mut incumbent = self.starting_point();
        match self.evaluate_candidate(&incumbent, budget, evaluate)? {
            None => return Ok(false),
            Some(performance) if performance >= self.performance_target => return Ok(true),
            Some(_) => {}
        }

        // Cloned once so the sweep can call `&mut self` methods while iterating
        // the search space in its (deterministic) registration order.
        let parameters = self.parameters.clone();
        let mut fraction = step_fraction;
        for _ in 0..max_rounds.max(1) {
            let mut improved = false;

            for parameter in &parameters {
                let current = incumbent
                    .get(&parameter.name)
                    .copied()
                    .unwrap_or_else(|| parameter.midpoint());
                let step = parameter.span() * crate::utils::scalar_or(fraction, A::zero());

                for direction in [A::one(), -A::one()] {
                    let proposal = parameter.clamp(current + step * direction);
                    if proposal == current {
                        continue;
                    }

                    let mut candidate = incumbent.clone();
                    candidate.insert(parameter.name.clone(), proposal);

                    let best_before = self.best_performance;
                    let performance = self.evaluate_candidate(&candidate, budget, evaluate)?;
                    match performance {
                        None => return Ok(false),
                        Some(performance) => {
                            // `evaluate_candidate` only replaces the recorded
                            // best on a strict improvement, so comparing against
                            // the pre-evaluation best is what decides whether the
                            // incumbent moves.
                            if best_before.is_none_or(|best| performance > best) {
                                incumbent = candidate;
                                improved = true;
                            }
                            if performance >= self.performance_target {
                                return Ok(true);
                            }
                        }
                    }
                }
            }

            if !improved {
                fraction *= 0.5;
                if fraction < MIN_STEP_FRACTION {
                    break;
                }
            }
        }

        Ok(false)
    }

    /// Elitist real-coded genetic search: tournament selection, BLX-alpha
    /// crossover and bounded uniform mutation, seeded from
    /// [`AdaptiveTuner::random_seed`] so a run is reproducible.
    ///
    /// Unlike greedy search this can leave a local optimum, which is the reason
    /// to reach for it on a multi-modal objective; unlike grid search its cost
    /// does not grow exponentially with the number of parameters.
    ///
    /// Returns whether the target was reached.
    fn tune_genetic<F>(
        &mut self,
        population_size: usize,
        generations: usize,
        budget: &mut usize,
        evaluate: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(&HashMap<String, A>) -> Result<TuningObservation<A>>,
    {
        if population_size < 2 {
            return Err(OptimError::InvalidConfig(
                "TuningStrategy::GeneticAlgorithm needs a population of at least 2 \
                 so selection has something to choose between"
                    .to_string(),
            ));
        }
        if generations == 0 {
            return Err(OptimError::InvalidConfig(
                "TuningStrategy::GeneticAlgorithm needs at least one generation".to_string(),
            ));
        }

        let mut rng = seeded_rng(self.random_seed);
        let parameters = self.parameters.clone();

        // The incumbent seeds individual 0, so a genetic run can only improve on
        // what an earlier search already found.
        let mut population: Vec<Vec<A>> = Vec::with_capacity(population_size);
        let start = self.starting_point();
        population.push(
            parameters
                .iter()
                .map(|parameter| {
                    start
                        .get(&parameter.name)
                        .copied()
                        .unwrap_or_else(|| parameter.midpoint())
                })
                .collect(),
        );
        for _ in 1..population_size {
            population.push(
                parameters
                    .iter()
                    .map(|parameter| {
                        let fraction: f64 = rng.gen_range(0.0..1.0);
                        parameter.clamp(
                            parameter.minimum + parameter.span() * scalar_or(fraction, A::zero()),
                        )
                    })
                    .collect(),
            );
        }

        let mut fitness = Vec::with_capacity(population_size);
        for individual in &population {
            let candidate = as_candidate(&parameters, individual);
            match self.evaluate_candidate(&candidate, budget, evaluate)? {
                None => return Ok(false),
                Some(performance) if performance >= self.performance_target => return Ok(true),
                Some(performance) => fitness.push(performance),
            }
        }

        for _ in 0..generations {
            // Elitism: the best individual survives unchanged, so a generation
            // can never lose ground.
            let elite_index = best_index(&fitness);
            let mut offspring: Vec<Vec<A>> = vec![population[elite_index].clone()];

            while offspring.len() < population_size {
                let first = tournament(&fitness, &mut rng);
                let second = tournament(&fitness, &mut rng);
                let mut child = Vec::with_capacity(parameters.len());

                for (index, parameter) in parameters.iter().enumerate() {
                    let left = population[first][index];
                    let right = population[second][index];
                    let low = if left < right { left } else { right };
                    let high = if left < right { right } else { left };
                    let widening = (high - low) * scalar_or(CROSSOVER_ALPHA, A::zero());
                    let span = (high + widening) - (low - widening);
                    let fraction: f64 = rng.gen_range(0.0..1.0);
                    let mut value = (low - widening) + span * scalar_or(fraction, A::zero());

                    if rng.gen_range(0.0..1.0) < MUTATION_PROBABILITY {
                        let jitter: f64 = rng.gen_range(-MUTATION_SCALE..MUTATION_SCALE);
                        value = value + parameter.span() * scalar_or(jitter, A::zero());
                    }

                    child.push(parameter.clamp(value));
                }

                offspring.push(child);
            }

            let mut offspring_fitness = Vec::with_capacity(offspring.len());
            for individual in &offspring {
                let candidate = as_candidate(&parameters, individual);
                match self.evaluate_candidate(&candidate, budget, evaluate)? {
                    None => return Ok(false),
                    Some(performance) if performance >= self.performance_target => {
                        return Ok(true);
                    }
                    Some(performance) => offspring_fitness.push(performance),
                }
            }

            population = offspring;
            fitness = offspring_fitness;
        }

        Ok(false)
    }

    /// Starting candidate for a local search: the best candidate found so far
    /// when it covers every registered parameter, otherwise the midpoint of each
    /// range.
    fn starting_point(&self) -> HashMap<String, A> {
        let mut start = HashMap::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let value = match self.current_params.get(&parameter.name) {
                Some(&value) => parameter.clamp(value),
                None => parameter.midpoint(),
            };
            start.insert(parameter.name.clone(), value);
        }
        start
    }

    /// Measure one candidate, record it, and keep it if it is the best so far.
    ///
    /// Returns `None` when the evaluation budget is exhausted, which ends the
    /// search without failing it.
    fn evaluate_candidate<F>(
        &mut self,
        candidate: &HashMap<String, A>,
        budget: &mut usize,
        evaluate: &mut F,
    ) -> Result<Option<A>>
    where
        F: FnMut(&HashMap<String, A>) -> Result<TuningObservation<A>>,
    {
        if *budget == 0 {
            return Ok(None);
        }
        *budget -= 1;

        let observation = evaluate(candidate)?;
        if !observation.performance.is_finite() {
            return Err(OptimError::InvalidParameter(
                "the tuning evaluator reported a non-finite performance, which \
                 cannot be ranked"
                    .to_string(),
            ));
        }
        self.total_evaluations += 1;

        self.tuning_history.push(TuningRecord {
            parameters: candidate.clone(),
            performance: observation.performance,
            resource_usage: observation.resource_usage,
            timestamp: unix_timestamp_secs(),
        });
        if self.tuning_history.len() > MAX_TUNING_HISTORY {
            self.tuning_history.remove(0);
        }

        let improved = match self.best_performance {
            Some(best) => observation.performance > best,
            None => true,
        };
        if improved {
            self.best_performance = Some(observation.performance);
            self.current_params = candidate.clone();
        }

        Ok(Some(observation.performance))
    }

    /// Best resource usage recorded among the candidates that met the
    /// performance target, if any.
    ///
    /// Surfaces [`TuningRecord::resource_usage`], which is otherwise only
    /// reachable by walking the history: the cheapest candidate that still hits
    /// the target is usually the one a caller wants to deploy.
    pub fn cheapest_candidate_meeting_target(&self) -> Option<&TuningRecord<A>> {
        self.tuning_history
            .iter()
            .filter(|record| record.performance >= self.performance_target)
            .min_by(|a, b| total_order(&a.resource_usage, &b.resource_usage))
    }
}

/// Pair a genetic individual's coordinates back up with their parameter names.
fn as_candidate<A: Float>(
    parameters: &[TunableParameter<A>],
    individual: &[A],
) -> HashMap<String, A> {
    parameters
        .iter()
        .zip(individual.iter())
        .map(|(parameter, &value)| (parameter.name.clone(), value))
        .collect()
}

/// Index of the fittest individual. Ties go to the earliest index, so the choice
/// does not depend on the sort being stable.
fn best_index<A: Float>(fitness: &[A]) -> usize {
    let mut best = 0;
    for (index, value) in fitness.iter().enumerate() {
        if total_order(value, &fitness[best]) == std::cmp::Ordering::Greater {
            best = index;
        }
    }
    best
}

/// Tournament selection: draw [`TOURNAMENT_SIZE`] individuals and return the
/// fittest one's index.
fn tournament<A: Float>(fitness: &[A], rng: &mut CoreRandom<StdRng>) -> usize {
    let mut best = rng.gen_range(0..fitness.len());
    for _ in 1..TOURNAMENT_SIZE {
        let challenger = rng.gen_range(0..fitness.len());
        if total_order(&fitness[challenger], &fitness[best]) == std::cmp::Ordering::Greater {
            best = challenger;
        }
    }
    best
}

/// Evenly spaced grid values for one parameter.
///
/// A resolution of 1 collapses to the midpoint rather than to a bound, so a
/// single-point grid probes the middle of the authorised range instead of its
/// edge.
fn grid_axis<A: Float>(parameter: &TunableParameter<A>, resolution: usize) -> Vec<A> {
    if resolution <= 1 {
        return vec![parameter.midpoint()];
    }
    let divisor = crate::utils::scalar_or(resolution - 1, A::one());
    (0..resolution)
        .map(|index| {
            let fraction = crate::utils::scalar_or(index, A::zero()) / divisor;
            parameter.clamp(parameter.minimum + parameter.span() * fraction)
        })
        .collect()
}

/// Seconds since the Unix epoch, clamped to the epoch on a mis-set clock rather
/// than panicking a tuning run.
fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Read a tuned parameter back as an `f64`, for callers that need to map it onto
/// an integer configuration field such as a batch size.
pub fn tuned_value_as_f64<A: Float>(params: &HashMap<String, A>, name: &str) -> Option<f64> {
    params.get(name).copied().and_then(|v| try_f64(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unimodal synthetic objective peaking at `x = 3, y = -1`.
    fn synthetic_objective(params: &HashMap<String, f64>) -> Result<TuningObservation<f64>> {
        let x = params.get("x").copied().unwrap_or(0.0);
        let y = params.get("y").copied().unwrap_or(0.0);
        let performance = 10.0 - (x - 3.0).powi(2) - (y + 1.0).powi(2);
        Ok(TuningObservation {
            performance,
            resource_usage: x.abs() + y.abs(),
        })
    }

    fn tuner_with_space() -> AdaptiveTuner<f64> {
        let mut tuner = AdaptiveTuner::new();
        tuner
            .add_parameter(TunableParameter::new("x", -10.0, 10.0).expect("valid range"))
            .expect("register x");
        tuner
            .add_parameter(TunableParameter::new("y", -10.0, 10.0).expect("valid range"))
            .expect("register y");
        tuner
    }

    /// Greedy hill climbing must actually improve the objective and leave the
    /// winner in `current_params`.
    #[test]
    fn greedy_search_improves_a_synthetic_objective() {
        let mut tuner = tuner_with_space();
        tuner.set_performance_target(9.99);
        tuner.set_strategy(TuningStrategy::Greedy {
            step_fraction: 0.25,
            max_rounds: 40,
        });

        let start = synthetic_objective(&tuner.starting_point()).expect("start");
        let outcome = tuner.tune(synthetic_objective).expect("tuning must run");

        assert!(
            outcome.best_performance > start.performance,
            "greedy search did not improve on the starting point ({} -> {})",
            start.performance,
            outcome.best_performance
        );
        assert!(outcome.evaluations > 1, "nothing was searched");
        assert_eq!(tuner.total_evaluations(), outcome.evaluations);
        assert_eq!(
            tuner.tuning_history().len(),
            outcome.evaluations,
            "every measurement must be recorded"
        );
        assert_eq!(
            tuner.current_params(),
            &outcome.best_parameters,
            "current_params must hold the best candidate"
        );

        let x = tuner.current_params().get("x").copied().expect("x tuned");
        let y = tuner.current_params().get("y").copied().expect("y tuned");
        assert!(
            (x - 3.0).abs() < 1.0,
            "x = {x} did not approach the optimum"
        );
        assert!(
            (y + 1.0).abs() < 1.0,
            "y = {y} did not approach the optimum"
        );
    }

    /// Grid search must sweep the whole grid and find its best point.
    #[test]
    fn grid_search_sweeps_the_configured_grid() {
        let mut tuner = tuner_with_space();
        tuner.set_performance_target(1e9); // unreachable: force a full sweep
        tuner.set_strategy(TuningStrategy::GridSearch { resolution: 11 });
        tuner.set_max_evaluations(200);

        let outcome = tuner.tune(synthetic_objective).expect("tuning must run");

        assert_eq!(outcome.evaluations, 121, "11 x 11 grid");
        assert!(!outcome.target_reached);
        // The grid contains x = 2 and x = 4 but not 3; y = -2 and y = 0 but not
        // -1. The best grid point is therefore one step off the true optimum.
        let x = outcome.best_parameters.get("x").copied().expect("x");
        let y = outcome.best_parameters.get("y").copied().expect("y");
        assert!((x - 2.0).abs() < 1e-9 || (x - 4.0).abs() < 1e-9, "x = {x}");
        assert!((y + 2.0).abs() < 1e-9 || y.abs() < 1e-9, "y = {y}");
    }

    /// Reaching the target must stop the search early.
    #[test]
    fn the_search_stops_once_the_target_is_reached() {
        let mut tuner = tuner_with_space();
        // The first grid point is the corner (-10, -10), worth -240; a target of
        // -1000 is therefore met immediately.
        tuner.set_performance_target(-1000.0);
        tuner.set_strategy(TuningStrategy::GridSearch { resolution: 5 });

        let outcome = tuner.tune(synthetic_objective).expect("tuning must run");
        assert!(outcome.target_reached);
        assert_eq!(outcome.evaluations, 1, "the target was met immediately");
    }

    /// A grid that does not fit the evaluation budget must be reported, not
    /// silently truncated to a prefix.
    #[test]
    fn an_oversized_grid_is_reported() {
        let mut tuner = tuner_with_space();
        tuner.set_strategy(TuningStrategy::GridSearch { resolution: 40 });
        tuner.set_max_evaluations(100);

        let error = tuner
            .tune(synthetic_objective)
            .expect_err("1600 evaluations must not fit a budget of 100");
        assert!(matches!(error, OptimError::InvalidConfig(_)), "{error:?}");
    }

    /// Tuning with no registered parameters is a configuration error, not a
    /// silent no-op.
    #[test]
    fn tuning_without_a_search_space_is_reported() {
        let mut tuner: AdaptiveTuner<f64> = AdaptiveTuner::new();
        let error = tuner
            .tune(synthetic_objective)
            .expect_err("an empty search space must be reported");
        assert!(matches!(error, OptimError::InvalidConfig(_)), "{error:?}");
    }

    /// The genetic search must improve the objective and be reproducible.
    #[test]
    fn genetic_search_improves_a_synthetic_objective_reproducibly() {
        let run = || {
            let mut tuner = tuner_with_space();
            tuner.set_performance_target(9.999);
            tuner.set_strategy(TuningStrategy::GeneticAlgorithm {
                population_size: 12,
                generations: 20,
            });
            tuner.set_max_evaluations(1000);
            let outcome = tuner.tune(synthetic_objective).expect("tuning must run");
            (tuner, outcome)
        };

        let (tuner, outcome) = run();
        let start = synthetic_objective(&HashMap::from([
            ("x".to_string(), 0.0),
            ("y".to_string(), 0.0),
        ]))
        .expect("start");
        assert!(
            outcome.best_performance > start.performance,
            "genetic search did not improve on the midpoint ({} -> {})",
            start.performance,
            outcome.best_performance
        );
        for (name, &value) in &outcome.best_parameters {
            assert!(
                (-10.0..=10.0).contains(&value),
                "{name} = {value} left its authorised range"
            );
        }

        let (_, repeat) = run();
        assert_eq!(
            outcome.best_parameters.len(),
            repeat.best_parameters.len(),
            "the same seed must produce the same search"
        );
        assert!(
            (outcome.best_performance - repeat.best_performance).abs() < 1e-12,
            "the same seed produced a different result: {} vs {}",
            outcome.best_performance,
            repeat.best_performance
        );
        assert_eq!(tuner.random_seed(), DEFAULT_TUNING_SEED);
    }

    /// A degenerate population or generation count is a configuration error.
    #[test]
    fn a_degenerate_genetic_configuration_is_reported() {
        for strategy in [
            TuningStrategy::GeneticAlgorithm {
                population_size: 1,
                generations: 5,
            },
            TuningStrategy::GeneticAlgorithm {
                population_size: 10,
                generations: 0,
            },
        ] {
            let mut tuner = tuner_with_space();
            tuner.set_strategy(strategy.clone());
            let error = tuner
                .tune(synthetic_objective)
                .expect_err("a degenerate genetic configuration must be reported");
            assert!(
                matches!(error, OptimError::InvalidConfig(_)),
                "{strategy:?}: {error:?}"
            );
        }
    }

    /// The strategies with no implementation behind them must say so instead of
    /// quietly behaving like a different search.
    #[test]
    fn unimplemented_strategies_report_what_is_missing() {
        for strategy in [
            TuningStrategy::BayesianOptimization { num_samples: 10 },
            TuningStrategy::ReinforcementLearning {
                exploration_rate: 0.1,
            },
        ] {
            let mut tuner = tuner_with_space();
            tuner.set_strategy(strategy.clone());
            let error = tuner
                .tune(synthetic_objective)
                .expect_err("an unimplemented strategy must not fabricate success");
            assert!(
                matches!(error, OptimError::UnsupportedOperation(_)),
                "{strategy:?}: {error:?}"
            );
        }
    }

    /// Every measurement must carry its resource usage, so the cheapest
    /// candidate meeting the target can be recovered.
    #[test]
    fn resource_usage_is_recorded_with_every_measurement() {
        let mut tuner = tuner_with_space();
        tuner.set_strategy(TuningStrategy::GridSearch { resolution: 5 });
        tuner.set_max_evaluations(100);
        // An unreachable target forces the full sweep, so the whole history is
        // available to inspect.
        tuner.set_performance_target(1e9);
        tuner.tune(synthetic_objective).expect("tuning must run");

        assert_eq!(tuner.tuning_history().len(), 25);
        for record in tuner.tuning_history() {
            let expected: f64 = record
                .parameters
                .values()
                .map(|value| value.abs())
                .sum::<f64>();
            assert!(
                (record.resource_usage - expected).abs() < 1e-9,
                "resource usage was not recorded from the evaluator"
            );
            assert!(record.timestamp > 0, "timestamp must be recorded");
        }

        tuner.set_performance_target(5.0);
        let cheapest = tuner
            .cheapest_candidate_meeting_target()
            .expect("some grid point beats 5.0");
        assert!(cheapest.performance >= 5.0);
    }

    /// A non-finite measurement cannot be ranked and must be reported.
    #[test]
    fn a_non_finite_measurement_is_reported() {
        let mut tuner = tuner_with_space();
        let error = tuner
            .tune(|_| {
                Ok(TuningObservation {
                    performance: f64::NAN,
                    resource_usage: 0.0,
                })
            })
            .expect_err("NaN performance must be reported");
        assert!(
            matches!(error, OptimError::InvalidParameter(_)),
            "{error:?}"
        );
    }

    /// An invalid search range must be rejected at registration time.
    #[test]
    fn an_invalid_range_is_rejected() {
        assert!(TunableParameter::new("x", 1.0, 0.0).is_err());
        assert!(TunableParameter::<f64>::new("", 0.0, 1.0).is_err());
        assert!(TunableParameter::new("x", f64::NAN, 1.0).is_err());

        let mut tuner = tuner_with_space();
        assert!(
            tuner
                .add_parameter(TunableParameter::new("x", 0.0, 1.0).expect("valid"))
                .is_err(),
            "a duplicate parameter name must be rejected"
        );
    }
}
