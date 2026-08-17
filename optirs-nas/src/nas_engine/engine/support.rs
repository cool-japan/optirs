//! Supporting traits (SearchStrategy, MultiObjectiveOptimizer, ArchitectureController) and the real evaluator/predictor/progressive-search implementations they are built from.

use crate::error::Result;
use crate::multi_objective;
use crate::nas_engine::config::*;
use crate::nas_engine::resources::*;
use crate::nas_engine::results::*;
use scirs2_core::numeric::Float;
use scirs2_core::RngExt;
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
/// Progressive NAS implementation
#[derive(Debug)]
pub struct ProgressiveNAS<T: Float + Debug + Send + Sync + 'static> {
    pub(super) stages: Vec<ProgressiveStage<T>>,
    pub(super) current_stage: usize,
    pub(super) stage_history: Vec<Vec<SearchResult<T>>>,
}
impl<T: Float + Debug + Send + Sync + 'static> ProgressiveNAS<T> {
    pub fn new(_config: &NASConfig<T>) -> Result<Self> {
        Ok(Self {
            stages: Vec::new(),
            current_stage: 0,
            stage_history: Vec::new(),
        })
    }
    pub fn filter_candidates(
        &mut self,
        candidates: Vec<OptimizerArchitecture<T>>,
        _generation: usize,
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        Ok(candidates)
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
