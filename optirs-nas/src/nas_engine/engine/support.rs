//! Supporting traits (SearchStrategy, MultiObjectiveOptimizer, ArchitectureController) and the real evaluator/predictor/progressive-search implementations they are built from.

use crate::error::Result;
use crate::multi_objective;
use crate::nas_engine::config::*;
use crate::nas_engine::results::*;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Diversity metrics for population
#[derive(Debug, Clone)]
pub struct DiversityMetrics<T: Float + Debug + Send + Sync + 'static> {
    pub crowding_distance: Vec<T>,
    pub entropy: T,
    pub average_distance: T,
    pub min_distance: T,
    pub max_distance: T,
}
/// Performance evaluator for architectures.
///
/// This is the engine-side handle onto the real evaluation subsystem in
/// [`crate::evaluation`]: every call to [`PerformanceEvaluator::evaluate`]
/// instantiates the concrete optimizer described by the candidate architecture
/// and actually runs it on the registered benchmark test functions. Earlier
/// releases carried a stub here that returned a constant score, which made the
/// whole search a no-op; that stub is gone.
#[derive(Debug)]
pub struct PerformanceEvaluator<T: Float + Debug + Send + Sync + 'static> {
    pub(super) config: EvaluationConfig<T>,
    /// Live evaluator performing benchmark execution, caching and statistics.
    pub(super) inner: crate::evaluation::PerformanceEvaluator<T>,
    /// Results returned so far, shared so cheap clones observe the same view.
    pub(super) evaluation_cache: Arc<Mutex<HashMap<String, EvaluationResults<T>>>>,
    pub(super) evaluation_count: usize,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    PerformanceEvaluator<T>
{
    pub fn new(config: EvaluationConfig<T>) -> Result<Self> {
        let evaluation_config = crate::EvaluationConfig::from_engine_config(&config);
        let mut inner = crate::evaluation::PerformanceEvaluator::<T>::new(evaluation_config)?;
        inner.initialize()?;
        Ok(Self {
            config,
            inner,
            evaluation_cache: Arc::new(Mutex::new(HashMap::new())),
            evaluation_count: 0,
        })
    }
    /// Evaluate a candidate architecture by actually running it.
    ///
    /// Delegates to [`crate::evaluation::PerformanceEvaluator::evaluate_architecture`],
    /// which builds the optimizer the architecture describes and minimizes each
    /// registered benchmark function with it. The achieved objectives determine
    /// the returned scores, so two different architectures receive different
    /// scores and an identical architecture reproduces exactly.
    pub fn evaluate(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<EvaluationResults<T>> {
        let results = self.inner.evaluate_architecture(architecture)?;
        self.evaluation_count += 1;
        match self.evaluation_cache.lock() {
            Ok(mut cache) => {
                cache.insert(architecture.architecture_id.clone(), results.clone());
            }
            Err(poisoned) => {
                let mut cache = poisoned.into_inner();
                cache.insert(architecture.architecture_id.clone(), results.clone());
            }
        }
        Ok(results)
    }
    /// Number of architectures evaluated by this evaluator.
    pub fn evaluation_count(&self) -> usize {
        self.evaluation_count
    }
    /// Engine-level evaluation configuration in force.
    pub fn config(&self) -> &EvaluationConfig<T> {
        &self.config
    }
}
/// Performance prediction system.
///
/// Wraps the real learned predictor in [`crate::evaluation::PerformancePredictor`]
/// (a ridge-regularised linear model over a deterministic architecture feature
/// vector, trained online from observed evaluations). Earlier releases returned
/// a constant `0.6` here.
#[derive(Debug)]
pub struct PerformancePredictor<T: Float + Debug + Send + Sync + 'static> {
    pub(super) model_type: PredictorType,
    /// Live predictor performing feature extraction, scoring and online updates.
    pub(super) inner: crate::evaluation::PerformancePredictor<T>,
    pub(super) training_data: Vec<(OptimizerArchitecture<T>, EvaluationResults<T>)>,
    pub(super) prediction_accuracy: T,
    pub(super) confidence_threshold: T,
}
impl<T: Float + Debug + Default + Send + Sync + 'static> PerformancePredictor<T> {
    pub fn new(config: &EvaluationConfig<T>) -> Result<Self> {
        let evaluation_config = crate::EvaluationConfig::from_engine_config(config);
        let inner = crate::evaluation::PerformancePredictor::<T>::new(&evaluation_config)?;
        Ok(Self {
            model_type: PredictorType::LinearRegression,
            inner,
            training_data: Vec::new(),
            prediction_accuracy: T::zero(),
            confidence_threshold: scirs2_core::numeric::NumCast::from(0.7)
                .unwrap_or_else(|| T::zero()),
        })
    }
    /// Predict the performance of an architecture without running it.
    ///
    /// Delegates to the learned model, which extracts a deterministic feature
    /// vector from the architecture and scores it; the returned confidence
    /// interval widens when little training data has been seen.
    pub fn predict(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<EvaluationResults<T>> {
        self.inner.predict_performance(architecture)
    }
    /// Feed observed evaluations back into the model.
    ///
    /// Each result performs one online gradient step on the predictor's
    /// weights, and the running prediction accuracy (1 - mean absolute error
    /// over the retained history) is refreshed.
    pub fn update_training_data(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        let evaluations: Vec<EvaluationResults<T>> = results
            .iter()
            .map(|r| r.evaluation_results.clone())
            .collect();
        self.inner.update_with_results(&evaluations)?;
        for result in results {
            self.training_data.push((
                result.architecture.clone(),
                result.evaluation_results.clone(),
            ));
        }
        const ACCURACY_WINDOW: usize = 64;
        let window: Vec<&(OptimizerArchitecture<T>, EvaluationResults<T>)> = self
            .training_data
            .iter()
            .rev()
            .take(ACCURACY_WINDOW)
            .collect();
        if !window.is_empty() {
            let mut error_sum = T::zero();
            let mut counted = 0usize;
            for (architecture, observed) in &window {
                if let Ok(prediction) = self.inner.predict_performance(architecture) {
                    let diff = prediction.overall_score - observed.overall_score;
                    error_sum = error_sum + diff.abs();
                    counted += 1;
                }
            }
            if counted > 0 {
                let count: T =
                    scirs2_core::numeric::NumCast::from(counted).unwrap_or_else(|| T::one());
                let mean_abs_error = error_sum / count;
                self.prediction_accuracy = (T::one() - mean_abs_error).max(T::zero());
            }
        }
        Ok(())
    }
    /// Current estimated prediction accuracy in `[0, 1]`.
    pub fn prediction_accuracy(&self) -> T {
        self.prediction_accuracy
    }
    /// Minimum accuracy at which the predictor should be trusted in place of a
    /// full evaluation.
    pub fn confidence_threshold(&self) -> T {
        self.confidence_threshold
    }
    /// Kind of model backing this predictor.
    pub fn model_type(&self) -> &PredictorType {
        &self.model_type
    }
    /// Number of `(architecture, evaluation)` pairs observed so far.
    pub fn training_sample_count(&self) -> usize {
        self.training_data.len()
    }
}
/// Types of predictors available
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictorType {
    /// Ridge-regularised linear model over the architecture feature vector.
    /// This is the model actually implemented by
    /// [`crate::evaluation::PerformancePredictor`].
    LinearRegression,
    NeuralNetwork,
    GaussianProcess,
    RandomForest,
    Ensemble,
}
/// Engine-side progressive search: a staged complexity schedule that constrains
/// which candidates may be evaluated in each phase of the run.
///
/// This complements [`crate::search_strategies::ProgressiveNAS`], which *generates*
/// progressively more complex architectures: whatever strategy the engine is
/// configured with, this filter enforces the schedule on the candidates that
/// actually reach the evaluator, so `NASConfig::progressive_search` means
/// something for every strategy.
///
/// It used to mean nothing at all. `new` built an empty `stages` vector and
/// `filter_candidates` was `Ok(candidates)` — the input returned verbatim — while
/// `current_stage` stayed `0` and `stage_history` was never written. Enabling
/// `progressive_search` therefore changed no behaviour whatsoever, even though the
/// engine reports it among the run's key hyperparameters.
#[derive(Debug)]
pub struct ProgressiveNAS<T: Float + Debug + Send + Sync + 'static> {
    pub(super) stages: Vec<ProgressiveStage<T>>,
    pub(super) current_stage: usize,
    pub(super) stage_history: Vec<Vec<SearchResult<T>>>,
    /// Generations allotted to each stage, derived from the search budget.
    pub(super) generations_per_stage: usize,
}
impl<T: Float + Debug + Default + Clone + Send + Sync + 'static> ProgressiveNAS<T> {
    /// Build the stage schedule from the configuration.
    ///
    /// One stage per allowed component count between
    /// `SearchSpaceConfig::min_components` and `max_components`, each stage holding
    /// a narrowed copy of the search space (its own `max_components`) so the stage
    /// configuration describes exactly what that stage may explore. The generation
    /// budget is split evenly across the stages, which is
    /// [`TimeBudgetAllocation::Uniform`] applied to the generation axis.
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        let min_components = config.search_space.min_components.max(1);
        let max_components = config.search_space.max_components.max(min_components);
        let stage_count = max_components - min_components + 1;

        let total_seconds = config
            .resource_constraints
            .time_constraints
            .max_search_time
            .as_secs_f64();
        let per_stage_hours = total_seconds / 3600.0 / stage_count as f64;

        let mut stages = Vec::with_capacity(stage_count);
        for (index, complexity) in (min_components..=max_components).enumerate() {
            let mut stage_search_space = config.search_space.clone();
            stage_search_space.max_components = complexity;
            let mut stage_config = config.clone();
            stage_config.search_space = stage_search_space.clone();
            stages.push(ProgressiveStage {
                name: format!("stage_{index}_upto_{complexity}_components"),
                search_space: stage_search_space,
                duration_hours: scirs2_core::numeric::NumCast::from(per_stage_hours)
                    .unwrap_or_else(T::zero),
                transfer_knowledge: config.enable_transfer_learning,
                stage_config,
            });
        }

        let generations_per_stage = (config.search_budget / stage_count.max(1)).max(1);
        Ok(Self {
            stages,
            current_stage: 0,
            stage_history: vec![Vec::new(); stage_count],
            generations_per_stage,
        })
    }

    /// The stage `generation` falls into, clamped to the last stage once the
    /// schedule is exhausted (the search may legitimately run longer than the
    /// budget the stages were sized from).
    pub fn stage_for_generation(&self, generation: usize) -> usize {
        if self.stages.is_empty() {
            return 0;
        }
        (generation / self.generations_per_stage.max(1)).min(self.stages.len() - 1)
    }

    /// Maximum component count allowed in the stage `generation` falls into.
    pub fn complexity_limit(&self, generation: usize) -> Option<usize> {
        self.stages
            .get(self.stage_for_generation(generation))
            .map(|stage| stage.search_space.max_components)
    }

    /// The stage schedule.
    pub fn stages(&self) -> &[ProgressiveStage<T>] {
        &self.stages
    }

    /// The stage the most recent `filter_candidates` call was in.
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Results recorded for `stage`.
    pub fn stage_results(&self, stage: usize) -> &[SearchResult<T>] {
        self.stage_history
            .get(stage)
            .map(|results| results.as_slice())
            .unwrap_or(&[])
    }

    /// Restrict `candidates` to the complexity the current stage allows.
    ///
    /// Candidates over the limit are held back rather than evaluated early — that
    /// is the whole point of a progressive schedule. If *every* candidate exceeds
    /// the limit the simplest ones are kept anyway: returning an empty list would
    /// stall the search, and a stalled generation is a worse answer than evaluating
    /// the closest candidates available.
    pub fn filter_candidates(
        &mut self,
        candidates: Vec<OptimizerArchitecture<T>>,
        generation: usize,
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        if self.stages.is_empty() || candidates.is_empty() {
            return Ok(candidates);
        }
        let stage = self.stage_for_generation(generation);
        if stage != self.current_stage {
            log::info!(
                "progressive search entering {} at generation {}",
                self.stages
                    .get(stage)
                    .map(|stage| stage.name.as_str())
                    .unwrap_or("<unknown stage>"),
                generation
            );
            self.current_stage = stage;
        }
        let limit = self
            .stages
            .get(stage)
            .map(|stage| stage.search_space.max_components)
            .unwrap_or(usize::MAX);

        let (allowed, deferred): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| candidate.components.len() <= limit);
        if !allowed.is_empty() {
            if !deferred.is_empty() {
                log::debug!(
                    "progressive search deferred {} candidate(s) over the {}-component \
                     limit of stage {}",
                    deferred.len(),
                    limit,
                    stage
                );
            }
            return Ok(allowed);
        }

        // Nothing fitted: keep the simplest candidates so the generation is not
        // empty, and say so.
        let mut fallback = deferred;
        fallback.sort_by_key(|candidate| candidate.components.len());
        let smallest = fallback
            .first()
            .map(|candidate| candidate.components.len())
            .unwrap_or(0);
        fallback.retain(|candidate| candidate.components.len() == smallest);
        log::warn!(
            "progressive search stage {} allows at most {} component(s) but every \
             candidate exceeded it; evaluating the {} simplest ({} components each)",
            stage,
            limit,
            fallback.len(),
            smallest
        );
        Ok(fallback)
    }

    /// Record `results` against the stage they were produced in, so the stage
    /// history describes what each stage actually achieved.
    pub fn record_stage_results(&mut self, generation: usize, results: &[SearchResult<T>]) {
        if results.is_empty() {
            return;
        }
        let stage = self.stage_for_generation(generation);
        if let Some(history) = self.stage_history.get_mut(stage) {
            history.extend(results.iter().cloned());
        }
    }
}
/// Progressive search stage
#[derive(Debug, Clone)]
pub struct ProgressiveStage<T: Float + Debug + Send + Sync + 'static> {
    pub name: String,
    pub search_space: SearchSpaceConfig,
    pub duration_hours: T,
    pub transfer_knowledge: bool,
    pub stage_config: NASConfig<T>,
}
/// Core search strategy trait
pub trait SearchStrategy<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Generate new candidate architectures
    fn generate_candidates(
        &mut self,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<Vec<OptimizerArchitecture<T>>>;

    /// Update strategy based on search results
    fn update_strategy(&mut self, results: &[SearchResult<T>]) -> Result<()>;

    /// Check if strategy has converged
    fn has_converged(&self) -> bool;

    /// Get strategy name
    fn strategy_name(&self) -> &str;
}

/// Multi-objective optimization trait
pub trait MultiObjectiveOptimizer<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Update Pareto front with new results
    fn update_pareto_front(
        &mut self,
        results: &[SearchResult<T>],
    ) -> Result<multi_objective::ParetoFront<T>>;

    /// Select next generation candidates
    fn select_candidates(
        &self,
        candidates: &[SearchResult<T>],
        population_size: usize,
    ) -> Result<Vec<SearchResult<T>>>;

    /// Calculate diversity metrics
    fn calculate_diversity(&self, population: &[SearchResult<T>]) -> f64;
}

/// Architecture controller trait
pub trait ArchitectureController<T: Float + Debug + Send + Sync + 'static>: Send + Sync {
    /// Generate random architecture
    fn generate_random(&mut self) -> Result<OptimizerArchitecture<T>>;

    /// Mutate existing architecture
    fn mutate(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>>;

    /// Crossover two architectures
    fn crossover(
        &mut self,
        parent1: &OptimizerArchitecture<T>,
        parent2: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>>;

    /// Validate architecture
    fn validate(&self, architecture: &OptimizerArchitecture<T>) -> Result<bool>;
}
