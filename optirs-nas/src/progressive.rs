// Progressive Neural Architecture Search
//
// Phase bookkeeping for a progressive search: which phase the search is in, how
// complexity grows with the phase, and which architectures each phase produced.
//
// This module tracks progression; [`crate::search_strategies::ProgressiveNAS`] is
// the strategy that *generates* architectures phase by phase. The two are
// independent — a caller can drive this tracker from any strategy.
//
// # Attribution
//
// Every recorded evaluation is credited to the architecture that produced it, and
// `best_per_phase` really is the best of each phase. It used to be neither:
//
// * `best_per_phase` was a three-slot FIFO. `record_architecture_evaluation`
//   pushed the id and popped the oldest, so it held the three *most recent*
//   architectures of the phase whatever they scored — a 0.01 result evicted a 0.99
//   one, and `get_best_for_phase` returned the answer to a question nobody asked.
// * `ProgressionRecord::best_performance` was documented as "best performance
//   achieved" and set to the performance of the single architecture being
//   recorded, so the field named the wrong quantity in every record.
// * `analyze_performance_trend` regressed over the last ten performances of the
//   *whole run*. Immediately after a phase transition that window straddles two
//   phases, so an `Adaptive` progression decided whether to advance out of phase
//   *n* using measurements largely taken in phase *n-1*.
// * `update_search_phase` recomputed the phase from scratch for the budget- and
//   time-based strategies, so a caller that passed a smaller `generation` (or a
//   clock that had not yet crossed the next boundary after an adaptive advance)
//   moved the search *backwards* into an earlier phase.
// * `ProgressiveNAS::new` derived `BudgetBased(search_budget / 4)`, which is `0`
//   for any budget under four, and `TimeBased` divided by
//   `duration_per_phase.as_secs()`, which is `0` for any sub-second duration.
//   Both then panicked with "attempt to divide by zero" on the first
//   `update_search_phase` call.
// * Completion was never signalled: the tracker reached `Final` and stayed there
//   silently, with no way for a caller to learn the schedule was exhausted.
//
// Completion is reported through [`ProgressiveNAS::is_search_complete`] rather
// than by returning `Err`, matching `search_strategies::progressive`: a normally
// finished search is not a failure, and recording after completion keeps working.

use std::fmt::Debug;

use super::NASConfig;
use crate::error::{OptimError, Result};
use crate::numeric::{count_as, scalar_as};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::Duration;

/// Number of best-scoring architectures retained per phase.
pub const BEST_PER_PHASE_CAPACITY: usize = 3;

/// Number of recent performances the adaptive trend regression uses.
pub const TREND_WINDOW: usize = 10;

/// Progressive NAS configuration
#[derive(Debug, Clone)]
pub struct ProgressiveNAS<T: Float + Debug + Send + Sync + 'static> {
    /// Current search phase
    pub current_phase: SearchPhase,

    /// Phase progression strategy
    pub progression_strategy: ProgressionStrategy,

    /// Complexity scheduler
    pub complexity_scheduler: ComplexityScheduler<T>,

    /// Architecture progression tracker
    pub architecture_progression: ArchitectureProgression<T>,

    /// Whether the schedule has been exhausted: the search reached
    /// [`SearchPhase::Final`] and a further advance was requested.
    search_complete: bool,
}

/// Search phases in progressive NAS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SearchPhase {
    /// Initial simple architectures
    Initial,

    /// Intermediate complexity
    Intermediate,

    /// Advanced architectures
    Advanced,

    /// Final optimization phase
    Final,
}

impl SearchPhase {
    /// Every phase in schedule order.
    pub const ORDER: [SearchPhase; 4] = [
        SearchPhase::Initial,
        SearchPhase::Intermediate,
        SearchPhase::Advanced,
        SearchPhase::Final,
    ];

    /// Position of this phase in the schedule, `0` for [`SearchPhase::Initial`].
    pub fn index(self) -> usize {
        match self {
            SearchPhase::Initial => 0,
            SearchPhase::Intermediate => 1,
            SearchPhase::Advanced => 2,
            SearchPhase::Final => 3,
        }
    }

    /// Phase at schedule position `index`, saturating at [`SearchPhase::Final`].
    pub fn at(index: usize) -> SearchPhase {
        *Self::ORDER.get(index).unwrap_or(&SearchPhase::Final)
    }

    /// The next phase, or `None` when this is already the last one.
    pub fn next(self) -> Option<SearchPhase> {
        Self::ORDER.get(self.index() + 1).copied()
    }
}

/// Strategy for progressing through phases
#[derive(Debug, Clone)]
pub enum ProgressionStrategy {
    /// Time-based progression
    TimeBased(std::time::Duration),

    /// Performance-based progression
    PerformanceBased(f64),

    /// Budget-based progression
    BudgetBased(usize),

    /// Adaptive progression
    Adaptive,
}

/// Complexity scheduler for progressive search
#[derive(Debug, Clone)]
pub struct ComplexityScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Current complexity level
    pub current_complexity: T,

    /// Maximum complexity
    pub max_complexity: T,

    /// Complexity increase rate
    pub increase_rate: T,

    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
}

/// Scheduling strategies for complexity increase
#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    /// Linear increase
    Linear,

    /// Exponential increase
    Exponential,

    /// Step-wise increase
    StepWise,

    /// Adaptive based on performance
    Adaptive,
}

/// Architecture progression tracker
#[derive(Debug, Clone)]
pub struct ArchitectureProgression<T: Float + Debug + Send + Sync + 'static> {
    /// Progression history
    pub history: Vec<ProgressionRecord<T>>,

    /// The best-scoring architectures of each phase, best first, at most
    /// [`BEST_PER_PHASE_CAPACITY`] per phase.
    ///
    /// Ranked by the performance recorded for the architecture, not by recency.
    /// Use [`ArchitectureProgression::record_evaluation`] to populate it so the
    /// score travels with the id.
    pub best_per_phase: HashMap<SearchPhase, Vec<ScoredArchitecture<T>>>,

    /// Performance trends
    pub performance_trends: Vec<T>,
}

/// An architecture together with the performance it achieved.
#[derive(Debug, Clone)]
pub struct ScoredArchitecture<T: Float + Debug + Send + Sync + 'static> {
    /// Identifier of the architecture that was evaluated.
    pub architecture_id: String,

    /// Performance it achieved.
    pub performance: T,
}

/// Record of progression step
#[derive(Debug, Clone)]
pub struct ProgressionRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Phase when recorded
    pub phase: SearchPhase,

    /// Complexity level
    pub complexity: T,

    /// Performance of the architecture this record is about.
    pub performance: T,

    /// Best performance achieved in `phase` up to and including this record.
    ///
    /// This used to be set to the single architecture's own performance while
    /// being documented as the best achieved, so a poor result overwrote the
    /// phase's high-water mark in every reader's eyes.
    pub best_performance: T,

    /// Number of architectures evaluated
    pub architectures_evaluated: usize,

    /// Timestamp
    pub timestamp: std::time::Instant,
}

impl<T: Float + Debug + Send + Sync + std::iter::Sum> ProgressiveNAS<T> {
    /// Create new progressive NAS.
    ///
    /// The budget-derived phase length is floored at one: `search_budget / 4` is
    /// `0` for any budget under four, and the previous code then divided by it on
    /// the first [`ProgressiveNAS::update_search_phase`] call.
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        let progression_strategy = match config.search_budget {
            budget if budget < 50 => ProgressionStrategy::TimeBased(Duration::from_secs(300)),
            budget if budget < 200 => ProgressionStrategy::BudgetBased((budget / 4).max(1)),
            _ => ProgressionStrategy::Adaptive,
        };

        Ok(Self {
            current_phase: SearchPhase::Initial,
            progression_strategy,
            complexity_scheduler: ComplexityScheduler::new()?,
            architecture_progression: ArchitectureProgression::new(),
            search_complete: false,
        })
    }

    /// Whether the schedule has been exhausted.
    ///
    /// Set once the search is in [`SearchPhase::Final`] and a further advance is
    /// requested. Recording and phase updates keep working afterwards, so a caller
    /// that ignores this still gets valid bookkeeping.
    pub fn is_search_complete(&self) -> bool {
        self.search_complete
    }

    /// Update search phase based on progress.
    ///
    /// Phases never move backwards: the budget- and time-based rules compute the
    /// phase the elapsed budget/time *implies* and take the later of that and the
    /// current phase. Recomputing unconditionally, as this used to, let a caller
    /// passing a smaller `generation` — or a clock that had not yet crossed the
    /// next boundary after an adaptive advance — walk the search back into an
    /// earlier phase and re-lower the complexity budget.
    pub fn update_search_phase(&mut self, generation: usize) -> Result<()> {
        match self.progression_strategy {
            ProgressionStrategy::BudgetBased(budget_per_phase) => {
                if budget_per_phase == 0 {
                    return Err(OptimError::InvalidConfig(
                        "ProgressionStrategy::BudgetBased needs a non-zero budget per \
                         phase; zero would make every generation belong to every phase"
                            .to_string(),
                    ));
                }
                self.advance_to_index(generation / budget_per_phase);
            }
            ProgressionStrategy::Adaptive => {
                // Adaptive progression based on the trend *within the current
                // phase*: a window spanning a transition measures the phase the
                // search has already left.
                if let Some(trend) = self.analyze_performance_trend()? {
                    let epsilon: T = scalar_as(0.01, "adaptive progression threshold")?;
                    if trend < epsilon {
                        self.advance_phase();
                    }
                }
            }
            ProgressionStrategy::TimeBased(duration_per_phase) => {
                let phase_seconds = duration_per_phase.as_secs();
                if phase_seconds == 0 {
                    return Err(OptimError::InvalidConfig(
                        "ProgressionStrategy::TimeBased needs a phase duration of at \
                         least one second; a sub-second duration divided by zero"
                            .to_string(),
                    ));
                }
                let elapsed = self
                    .architecture_progression
                    .history
                    .first()
                    .map(|first| first.timestamp.elapsed())
                    .unwrap_or(Duration::from_secs(0));

                let phases_elapsed = (elapsed.as_secs() / phase_seconds) as usize;
                self.advance_to_index(phases_elapsed);
            }
            ProgressionStrategy::PerformanceBased(threshold) => {
                if let Some(latest_performance) =
                    self.architecture_progression.performance_trends.last()
                {
                    let threshold: T = scalar_as(threshold, "performance progression threshold")?;
                    if *latest_performance > threshold {
                        self.advance_phase();
                    }
                }
            }
        }

        Ok(())
    }

    /// Move to the phase at schedule position `index` if it is later than the
    /// current one; never backwards.
    ///
    /// An `index` past the last phase means the schedule has been consumed, which
    /// latches completion. Completion is never un-latched: a caller that replays
    /// an earlier generation is not un-finishing the search.
    fn advance_to_index(&mut self, index: usize) {
        let phase = SearchPhase::at(index);
        if phase > self.current_phase {
            self.current_phase = phase;
        }
        if index >= SearchPhase::ORDER.len() {
            self.search_complete = true;
        }
    }

    /// Advance to the next phase, or record completion if there is none.
    fn advance_phase(&mut self) {
        match self.current_phase.next() {
            Some(next) => self.current_phase = next,
            None => self.search_complete = true,
        }
    }

    /// Least-squares slope of the last [`TREND_WINDOW`] performances *of the
    /// current phase*, or `None` when the phase has not produced that many yet.
    ///
    /// The window used to be taken from the whole run's performance history, so
    /// right after a transition it mixed two phases and the adaptive rule decided
    /// whether to leave phase *n* from measurements taken in phase *n-1*.
    fn analyze_performance_trend(&self) -> Result<Option<T>> {
        let phase = self.current_phase;
        let phase_performances: Vec<T> = self
            .architecture_progression
            .history
            .iter()
            .filter(|record| record.phase == phase)
            .map(|record| record.performance)
            .collect();

        if phase_performances.len() < TREND_WINDOW {
            return Ok(None);
        }

        let recent_trends = &phase_performances[phase_performances.len() - TREND_WINDOW..];

        // Ordinary least squares over x = 0..n-1.
        let n: T = count_as(recent_trends.len(), "trend window size")?;
        let two: T = scalar_as(2.0, "trend regression constant")?;
        let sum_x = n * (n - T::one()) / two;
        let sum_y = recent_trends.iter().cloned().sum::<T>();
        let mut sum_xy = T::zero();
        let mut sum_x2 = T::zero();
        for (i, &y) in recent_trends.iter().enumerate() {
            let x: T = count_as(i, "trend regression index")?;
            sum_xy = sum_xy + x * y;
            sum_x2 = sum_x2 + x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator == T::zero() {
            // A single-point window has no slope; report "no trend" rather than
            // dividing by zero.
            return Ok(None);
        }
        Ok(Some((n * sum_xy - sum_x * sum_y) / denominator))
    }

    /// Get current search configuration based on phase
    pub fn get_current_search_config(&self) -> SearchPhaseConfig<T> {
        match self.current_phase {
            SearchPhase::Initial => SearchPhaseConfig {
                complexity_limit: scirs2_core::numeric::NumCast::from(0.25)
                    .unwrap_or_else(|| T::zero()),
                max_components: 2,
                max_depth: 3,
                exploration_factor: scirs2_core::numeric::NumCast::from(0.8)
                    .unwrap_or_else(|| T::zero()),
                mutation_rate: scirs2_core::numeric::NumCast::from(0.3)
                    .unwrap_or_else(|| T::zero()),
                population_diversity_weight: scirs2_core::numeric::NumCast::from(0.7)
                    .unwrap_or_else(|| T::zero()),
                conservative_search: true,
            },
            SearchPhase::Intermediate => SearchPhaseConfig {
                complexity_limit: scirs2_core::numeric::NumCast::from(0.5)
                    .unwrap_or_else(|| T::zero()),
                max_components: 4,
                max_depth: 5,
                exploration_factor: scirs2_core::numeric::NumCast::from(0.6)
                    .unwrap_or_else(|| T::zero()),
                mutation_rate: scirs2_core::numeric::NumCast::from(0.2)
                    .unwrap_or_else(|| T::zero()),
                population_diversity_weight: scirs2_core::numeric::NumCast::from(0.5)
                    .unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
            SearchPhase::Advanced => SearchPhaseConfig {
                complexity_limit: scirs2_core::numeric::NumCast::from(0.75)
                    .unwrap_or_else(|| T::zero()),
                max_components: 6,
                max_depth: 7,
                exploration_factor: scirs2_core::numeric::NumCast::from(0.4)
                    .unwrap_or_else(|| T::zero()),
                mutation_rate: scirs2_core::numeric::NumCast::from(0.15)
                    .unwrap_or_else(|| T::zero()),
                population_diversity_weight: scirs2_core::numeric::NumCast::from(0.3)
                    .unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
            SearchPhase::Final => SearchPhaseConfig {
                complexity_limit: T::one(),
                max_components: 8,
                max_depth: 10,
                exploration_factor: scirs2_core::numeric::NumCast::from(0.2)
                    .unwrap_or_else(|| T::zero()),
                mutation_rate: scirs2_core::numeric::NumCast::from(0.1)
                    .unwrap_or_else(|| T::zero()),
                population_diversity_weight: scirs2_core::numeric::NumCast::from(0.2)
                    .unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
        }
    }

    /// Record an architecture evaluation, crediting it to the phase the search is
    /// in and to the architecture that produced it.
    ///
    /// An empty `architecture_id` is rejected: an unidentified result cannot be
    /// credited to anything, and the previous code stored it anyway, so
    /// `get_best_for_phase` handed back an empty string as a "best architecture".
    pub fn record_architecture_evaluation(
        &mut self,
        architecture_id: String,
        performance: T,
        complexity: T,
    ) -> Result<()> {
        if architecture_id.is_empty() {
            return Err(OptimError::InvalidParameter(
                "a progression record needs the id of the architecture it is about; \
                 an empty id cannot be credited to anything"
                    .to_string(),
            ));
        }

        self.architecture_progression.record_evaluation(
            self.current_phase,
            architecture_id,
            performance,
            complexity,
        );

        Ok(())
    }
}

/// Configuration for each search phase
#[derive(Debug, Clone)]
pub struct SearchPhaseConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Maximum complexity allowed in this phase
    pub complexity_limit: T,

    /// Maximum number of components
    pub max_components: usize,

    /// Maximum architecture depth
    pub max_depth: usize,

    /// Exploration factor (0.0 = exploit, 1.0 = explore)
    pub exploration_factor: T,

    /// Mutation rate for genetic algorithms
    pub mutation_rate: T,

    /// Weight for population diversity
    pub population_diversity_weight: T,

    /// Whether to use conservative search strategies
    pub conservative_search: bool,
}

impl<T: Float + Debug + Send + Sync + 'static> ComplexityScheduler<T> {
    /// Create new complexity scheduler
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_complexity: scalar_as(0.1, "initial complexity")?,
            max_complexity: T::one(),
            increase_rate: scalar_as(0.1, "complexity increase rate")?,
            strategy: SchedulingStrategy::Linear,
        })
    }

    /// Fraction of [`ComplexityScheduler::max_complexity`] each phase is allowed.
    fn phase_factor(phase: SearchPhase) -> Result<T> {
        let steps: T = count_as(SearchPhase::ORDER.len(), "phase count")?;
        let position: T = count_as(phase.index() + 1, "phase position")?;
        Ok(position / steps)
    }

    /// Update complexity for `phase` and return the new level.
    ///
    /// Every [`SchedulingStrategy`] is implemented. `StepWise` and `Adaptive` used
    /// to fall into a `_ => {}` arm that silently returned the previous value, so a
    /// caller selecting either got a complexity budget frozen at its initial 0.1
    /// for the whole search while the phases advanced around it. Fallible for the
    /// same reason: keeping the previous level on a conversion failure would
    /// reintroduce exactly that frozen budget, and there is no other value that is
    /// not a guess.
    pub fn update_complexity(&mut self, phase: SearchPhase) -> Result<T> {
        match self.strategy {
            // Complexity is a straight function of the phase: quarter, half,
            // three-quarters, all of it for a four-phase schedule.
            SchedulingStrategy::Linear => {
                self.current_complexity = self.max_complexity * Self::phase_factor(phase)?;
            }
            // Geometric growth per call, capped at the maximum.
            SchedulingStrategy::Exponential => {
                self.current_complexity = self.current_complexity * (T::one() + self.increase_rate);
                if self.current_complexity > self.max_complexity {
                    self.current_complexity = self.max_complexity;
                }
            }
            // One fixed step per phase transition, held flat inside a phase: the
            // level is `initial + increase_rate * phase_index`, capped.
            SchedulingStrategy::StepWise => {
                let steps: T = count_as(phase.index(), "completed phase count")?;
                let base: T = scalar_as(0.1, "step-wise base complexity")?;
                let level = base + self.increase_rate * steps;
                self.current_complexity = if level > self.max_complexity {
                    self.max_complexity
                } else {
                    level
                };
            }
            // Adaptive: grow towards the phase's allowance rather than jumping to
            // it, so a phase transition does not hand the search its full budget in
            // one step. Geometric approach with `increase_rate` as the step size.
            SchedulingStrategy::Adaptive => {
                let target = self.max_complexity * Self::phase_factor(phase)?;
                let gap = target - self.current_complexity;
                self.current_complexity = self.current_complexity + gap * self.increase_rate;
                if self.current_complexity > self.max_complexity {
                    self.current_complexity = self.max_complexity;
                }
            }
        }

        Ok(self.current_complexity)
    }
}

impl<T: Float + Debug + Send + Sync + 'static> Default for ArchitectureProgression<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ArchitectureProgression<T> {
    /// Create new architecture progression tracker
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            best_per_phase: HashMap::new(),
            performance_trends: Vec::new(),
        }
    }

    /// Record one evaluation: append the history entry, update the phase's
    /// high-water mark and its best-scoring set.
    pub fn record_evaluation(
        &mut self,
        phase: SearchPhase,
        architecture_id: String,
        performance: T,
        complexity: T,
    ) {
        let best_in_phase = self.best_in_phase(phase).map_or(performance, |best| {
            if performance > best {
                performance
            } else {
                best
            }
        });

        self.performance_trends.push(performance);
        self.history.push(ProgressionRecord {
            phase,
            complexity,
            performance,
            best_performance: best_in_phase,
            architectures_evaluated: self.history.len() + 1,
            timestamp: std::time::Instant::now(),
        });

        let ranked = self.best_per_phase.entry(phase).or_default();
        ranked.push(ScoredArchitecture {
            architecture_id,
            performance,
        });
        // Best first, ties broken by id so the retained set is deterministic
        // rather than dependent on insertion order.
        ranked.sort_by(|left, right| {
            right
                .performance
                .partial_cmp(&left.performance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.architecture_id.cmp(&right.architecture_id))
        });
        ranked.truncate(BEST_PER_PHASE_CAPACITY);
    }

    /// Record a pre-built progression step.
    ///
    /// Kept for callers that assemble the record themselves; it does *not*
    /// maintain [`ArchitectureProgression::best_per_phase`], because a bare record
    /// carries no architecture id to rank. Prefer
    /// [`ArchitectureProgression::record_evaluation`].
    pub fn record_step(&mut self, record: ProgressionRecord<T>) {
        self.performance_trends.push(record.performance);
        self.history.push(record);
    }

    /// Best performance recorded in `phase`, or `None` if it has none yet.
    pub fn best_in_phase(&self, phase: SearchPhase) -> Option<T> {
        self.best_per_phase
            .get(&phase)
            .and_then(|ranked| ranked.first())
            .map(|entry| entry.performance)
    }

    /// The best-scoring architecture ids of `phase`, best first.
    ///
    /// This used to return the three *most recently recorded* ids of the phase,
    /// whatever they scored, because the underlying vector was a FIFO.
    pub fn get_best_for_phase(&self, phase: SearchPhase) -> Vec<String> {
        self.best_per_phase
            .get(&phase)
            .map(|ranked| {
                ranked
                    .iter()
                    .map(|entry| entry.architecture_id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The best-scoring architectures of `phase` with their scores, best first.
    pub fn best_scored_for_phase(&self, phase: SearchPhase) -> &[ScoredArchitecture<T>] {
        self.best_per_phase
            .get(&phase)
            .map(|ranked| ranked.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_budget(budget: usize) -> NASConfig<f64> {
        NASConfig::<f64> {
            search_budget: budget,
            ..NASConfig::<f64>::default()
        }
    }

    #[test]
    fn test_progressive_nas_creation() {
        let nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&NASConfig::<f64>::default()).expect("construction");
        assert_eq!(nas.current_phase, SearchPhase::Initial);
        assert!(!nas.is_search_complete());
    }

    #[test]
    fn a_tiny_budget_does_not_produce_a_zero_length_phase() {
        // `search_budget / 4` is 0 for any budget under four, and the first
        // `update_search_phase` then divided by it.
        for budget in [0, 1, 2, 3, 50, 51, 199] {
            let mut nas: ProgressiveNAS<f64> =
                ProgressiveNAS::new(&config_with_budget(budget)).expect("construction");
            if let ProgressionStrategy::BudgetBased(per_phase) = nas.progression_strategy {
                assert!(per_phase >= 1, "budget {budget} gave a zero-length phase");
            }
            nas.update_search_phase(7)
                .expect("updating a phase must not divide by zero");
        }
    }

    #[test]
    fn a_sub_second_time_budget_is_an_error_not_a_panic() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");
        nas.progression_strategy =
            ProgressionStrategy::TimeBased(std::time::Duration::from_millis(250));
        nas.record_architecture_evaluation("a".to_string(), 0.5, 0.1)
            .expect("record");

        let err = nas
            .update_search_phase(1)
            .expect_err("a zero-second phase length cannot be honoured");
        assert!(matches!(err, OptimError::InvalidConfig(_)));

        nas.progression_strategy = ProgressionStrategy::BudgetBased(0);
        assert!(matches!(
            nas.update_search_phase(1),
            Err(OptimError::InvalidConfig(_))
        ));
    }

    #[test]
    fn budget_based_phases_advance_and_never_regress() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(100)).expect("construction");
        nas.progression_strategy = ProgressionStrategy::BudgetBased(10);

        nas.update_search_phase(0).expect("update");
        assert_eq!(nas.current_phase, SearchPhase::Initial);
        nas.update_search_phase(10).expect("update");
        assert_eq!(nas.current_phase, SearchPhase::Intermediate);
        nas.update_search_phase(25).expect("update");
        assert_eq!(nas.current_phase, SearchPhase::Advanced);
        nas.update_search_phase(99).expect("update");
        assert_eq!(nas.current_phase, SearchPhase::Final);

        // A smaller generation used to walk the search backwards.
        nas.update_search_phase(0).expect("update");
        assert_eq!(
            nas.current_phase,
            SearchPhase::Final,
            "phases must not regress"
        );
        assert!(
            nas.is_search_complete(),
            "staying in Final after the schedule is exhausted must be reported"
        );
    }

    #[test]
    fn performance_based_progression_advances_once_per_call_and_then_completes() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");
        nas.progression_strategy = ProgressionStrategy::PerformanceBased(0.5);

        nas.record_architecture_evaluation("a".to_string(), 0.9, 0.1)
            .expect("record");
        for expected in [
            SearchPhase::Intermediate,
            SearchPhase::Advanced,
            SearchPhase::Final,
        ] {
            nas.update_search_phase(0).expect("update");
            assert_eq!(nas.current_phase, expected);
            assert!(!nas.is_search_complete());
        }

        nas.update_search_phase(0).expect("update");
        assert_eq!(nas.current_phase, SearchPhase::Final);
        assert!(nas.is_search_complete());
    }

    #[test]
    fn the_adaptive_trend_only_sees_the_current_phase() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");
        nas.progression_strategy = ProgressionStrategy::Adaptive;

        // A steeply improving phase 0: ten records with a strong upward trend.
        for step in 0..TREND_WINDOW {
            nas.record_architecture_evaluation(format!("p0_{step}"), step as f64, 0.1)
                .expect("record");
        }
        nas.update_search_phase(0).expect("update");
        assert_eq!(
            nas.current_phase,
            SearchPhase::Initial,
            "a strongly improving phase must not be abandoned"
        );

        // Move to phase 1 and record a *flat* run there. If the window still
        // straddled phases it would still see phase 0's steep climb and refuse to
        // advance.
        nas.current_phase = SearchPhase::Intermediate;
        for step in 0..TREND_WINDOW {
            nas.record_architecture_evaluation(format!("p1_{step}"), 1.0, 0.1)
                .expect("record");
        }
        nas.update_search_phase(0).expect("update");
        assert_eq!(
            nas.current_phase,
            SearchPhase::Advanced,
            "a flat current phase must trigger an advance"
        );
    }

    #[test]
    fn best_per_phase_keeps_the_best_not_the_most_recent() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");

        for (id, score) in [
            ("great", 0.99),
            ("good", 0.80),
            ("ok", 0.60),
            ("bad", 0.10),
            ("awful", 0.01),
        ] {
            nas.record_architecture_evaluation(id.to_string(), score, 0.1)
                .expect("record");
        }

        let best = nas
            .architecture_progression
            .get_best_for_phase(SearchPhase::Initial);
        assert_eq!(
            best,
            vec!["great".to_string(), "good".to_string(), "ok".to_string()],
            "the FIFO used to return the three most recent ids instead"
        );
        assert_eq!(best.len(), BEST_PER_PHASE_CAPACITY);
        assert_eq!(
            nas.architecture_progression
                .best_in_phase(SearchPhase::Initial),
            Some(0.99)
        );

        // Records are credited to the phase that was current when they arrived.
        nas.current_phase = SearchPhase::Advanced;
        nas.record_architecture_evaluation("late".to_string(), 0.5, 0.2)
            .expect("record");
        assert_eq!(
            nas.architecture_progression
                .get_best_for_phase(SearchPhase::Advanced),
            vec!["late".to_string()]
        );
        assert!(!nas
            .architecture_progression
            .get_best_for_phase(SearchPhase::Initial)
            .contains(&"late".to_string()));
    }

    #[test]
    fn the_record_high_water_mark_is_the_phase_best_not_the_latest_score() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");
        nas.record_architecture_evaluation("first".to_string(), 0.9, 0.1)
            .expect("record");
        nas.record_architecture_evaluation("second".to_string(), 0.2, 0.1)
            .expect("record");

        let history = &nas.architecture_progression.history;
        assert_eq!(history.len(), 2);
        assert!((history[1].performance - 0.2).abs() < 1e-12);
        assert!(
            (history[1].best_performance - 0.9).abs() < 1e-12,
            "a worse result must not lower the phase's high-water mark"
        );
    }

    #[test]
    fn an_unidentified_evaluation_is_rejected() {
        let mut nas: ProgressiveNAS<f64> =
            ProgressiveNAS::new(&config_with_budget(10)).expect("construction");
        let err = nas
            .record_architecture_evaluation(String::new(), 0.5, 0.1)
            .expect_err("an empty architecture id cannot be credited");
        assert!(matches!(err, OptimError::InvalidParameter(_)));
        assert!(nas.architecture_progression.history.is_empty());
    }

    #[test]
    fn test_complexity_scheduler_linear() {
        let mut scheduler = ComplexityScheduler::<f64>::new().expect("construction");

        let initial = scheduler
            .update_complexity(SearchPhase::Initial)
            .expect("phase factors are representable in f64");
        let intermediate = scheduler
            .update_complexity(SearchPhase::Intermediate)
            .expect("phase factors are representable in f64");
        let advanced = scheduler
            .update_complexity(SearchPhase::Advanced)
            .expect("phase factors are representable in f64");
        let final_level = scheduler
            .update_complexity(SearchPhase::Final)
            .expect("phase factors are representable in f64");

        assert!(initial < intermediate);
        assert!(intermediate < advanced);
        assert!(advanced < final_level);
        assert!((final_level - 1.0).abs() < 1e-12);
    }

    #[test]
    fn every_scheduling_strategy_actually_schedules() {
        // `StepWise` and `Adaptive` used to fall into a `_ => {}` arm and return
        // the untouched initial 0.1 forever.
        let mut stepwise = ComplexityScheduler::<f64>::new().expect("construction");
        stepwise.strategy = SchedulingStrategy::StepWise;
        let levels: Vec<f64> = SearchPhase::ORDER
            .iter()
            .map(|&phase| {
                stepwise
                    .update_complexity(phase)
                    .expect("phase factors are representable in f64")
            })
            .collect();
        for window in levels.windows(2) {
            assert!(
                window[1] > window[0],
                "step-wise complexity did not grow: {levels:?}"
            );
        }
        assert!(levels.iter().all(|level| *level <= 1.0));

        let mut adaptive = ComplexityScheduler::<f64>::new().expect("construction");
        adaptive.strategy = SchedulingStrategy::Adaptive;
        let start = adaptive.current_complexity;
        let after = adaptive
            .update_complexity(SearchPhase::Final)
            .expect("phase factors are representable in f64");
        assert!(after > start, "adaptive complexity did not move");
        // It approaches the target rather than jumping to it.
        assert!(after < 1.0);
        for _ in 0..500 {
            adaptive
                .update_complexity(SearchPhase::Final)
                .expect("phase factors are representable in f64");
        }
        assert!((adaptive.current_complexity - 1.0).abs() < 1e-6);

        let mut exponential = ComplexityScheduler::<f64>::new().expect("construction");
        exponential.strategy = SchedulingStrategy::Exponential;
        for _ in 0..500 {
            exponential
                .update_complexity(SearchPhase::Initial)
                .expect("phase factors are representable in f64");
        }
        assert!((exponential.current_complexity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_architecture_progression() {
        let mut progression = ArchitectureProgression::<f64>::new();
        progression.record_evaluation(SearchPhase::Initial, "a".to_string(), 0.8, 0.1);

        assert_eq!(progression.history.len(), 1);
        assert_eq!(progression.performance_trends.len(), 1);
        assert_eq!(progression.history[0].architectures_evaluated, 1);
        assert_eq!(
            progression
                .best_scored_for_phase(SearchPhase::Initial)
                .len(),
            1
        );
        assert!(progression
            .best_scored_for_phase(SearchPhase::Advanced)
            .is_empty());
    }
}
