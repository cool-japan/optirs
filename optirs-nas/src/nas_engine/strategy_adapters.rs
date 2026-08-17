// Engine-side adapters that expose the real search strategies from
// [`crate::search_strategies`] through the engine's [`SearchStrategy`] trait.
//
// Historically the engine only ever constructed `RandomStrategy` and
// `EvolutionaryStrategy`; every other `SearchStrategyType` silently fell back to
// one of those two (with an `eprintln!` warning) even though complete,
// separately-tested implementations of Bayesian optimization, an RL controller,
// DARTS, progressive search and a neural predictor already lived in
// `crate::search_strategies`. This module wires those implementations into the
// engine.
//
// Every real strategy in `crate::search_strategies` implements the *inner*
// `SearchStrategy` trait (`initialize` / `generate_architecture` /
// `update_with_results`). [`DelegatingAdapter`] wraps any such strategy and
// forwards the engine trait's batch-oriented calls to it, mirroring the
// hand-written `RandomStrategy` / `EvolutionaryStrategy` adapters in
// `engine.rs`.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;

use scirs2_core::numeric::Float;

use crate::error::{OptimError, Result};
use crate::nas_engine::config::{NASConfig, SearchSpaceConfig};
use crate::nas_engine::engine::SearchStrategy;
use crate::nas_engine::results::{OptimizerArchitecture, SearchResult};
use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

/// Full trait bound shared by the engine and every real inner strategy.
///
/// It is deliberately identical to the bound on the engine's
/// `NeuralArchitectureSearch<T>` impl (which is where the constructors below are
/// called from), so any `T` that can drive the engine can also drive these
/// adapters. `DifferentiableSearch` in particular needs
/// [`scirs2_core::ndarray::ScalarOperand`], and the Bayesian / neural strategies
/// need [`std::iter::Sum`]; requiring the whole set here keeps a single, honest
/// bound instead of five subtly different ones.
pub(crate) trait AdapterFloat:
    Float
    + Debug
    + Default
    + Clone
    + Send
    + Sync
    + 'static
    + std::fmt::Display
    + From<f64>
    + std::iter::Sum
    + for<'a> std::iter::Sum<&'a Self>
    + scirs2_core::ndarray::ScalarOperand
{
}

impl<T> AdapterFloat for T where
    T: Float
        + Debug
        + Default
        + Clone
        + Send
        + Sync
        + 'static
        + std::fmt::Display
        + From<f64>
        + std::iter::Sum
        + for<'a> std::iter::Sum<&'a T>
        + scirs2_core::ndarray::ScalarOperand
{
}

/// Adapter that presents an inner [`crate::search_strategies`] strategy through
/// the engine's [`SearchStrategy`] trait.
///
/// `generate_candidates` calls the inner `generate_architecture` `batch_size`
/// times to produce one generation of candidates; `update_strategy` forwards the
/// evaluated results so the inner strategy can learn (fit the GP, run REINFORCE,
/// update DARTS weights, advance a phase, train the predictor, ...).
pub(crate) struct DelegatingAdapter<T: AdapterFloat, S: InnerSearchStrategy<T>> {
    inner: S,
    search_space: SearchSpaceConfig,
    batch_size: usize,
    name: &'static str,
    _phantom: PhantomData<T>,
}

impl<T: AdapterFloat, S: InnerSearchStrategy<T>> DelegatingAdapter<T, S> {
    fn new(inner: S, config: &NASConfig<T>, name: &'static str) -> Self {
        // Produce one full population per generation; fall back to a small
        // non-zero batch so the search always makes progress.
        let batch_size = if config.population_size > 0 {
            config.population_size
        } else {
            8
        };
        Self {
            inner,
            search_space: config.search_space.clone(),
            batch_size,
            name,
            _phantom: PhantomData,
        }
    }
}

impl<T: AdapterFloat, S: InnerSearchStrategy<T>> SearchStrategy<T> for DelegatingAdapter<T, S> {
    fn generate_candidates(
        &mut self,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        let mut candidates = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            candidates.push(
                self.inner
                    .generate_architecture(&self.search_space, history)?,
            );
        }
        Ok(candidates)
    }

    fn update_strategy(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        self.inner.update_with_results(results)
    }

    fn has_converged(&self) -> bool {
        // Termination is otherwise owned by the engine (search budget, early
        // stopping, resource limits): a controller / GP / DARTS relaxation has no
        // intrinsic fixed point and reports `false` through the default
        // `is_search_complete`.
        //
        // A strategy that *does* have a schedule — `ProgressiveNAS`, whose phases
        // run out — reports it here, so the engine can stop instead of letting the
        // strategy degenerate into sampling at its final complexity level.
        self.inner.is_search_complete()
    }

    fn strategy_name(&self) -> &str {
        self.name
    }
}

/// A strategy that samples concrete hyper-parameters from the search space
/// requires the detailed `components` list to be populated (unlike the random
/// baseline, which can synthesize from the `component_types` vocabulary alone).
/// Reject an empty list with a clear, actionable error rather than panicking
/// deep inside the sampler.
fn require_components<T: AdapterFloat>(config: &NASConfig<T>, strategy: &str) -> Result<()> {
    if config.search_space.components.is_empty() {
        return Err(OptimError::SearchSpaceError(format!(
            "the {strategy} strategy requires a non-empty \
             SearchSpaceConfig::components; populate the component list (not just \
             component_types) before selecting this strategy"
        )));
    }
    Ok(())
}

/// Build a Bayesian-optimization strategy (GP surrogate + acquisition function).
pub(crate) fn bayesian<T: AdapterFloat>(
    config: &NASConfig<T>,
) -> Result<Box<dyn SearchStrategy<T>>> {
    use crate::search_strategies::{AcquisitionType, BayesianOptimization, KernelType};

    require_components(config, "Bayesian optimization")?;

    let mut inner = BayesianOptimization::<T>::new(KernelType::Matern52, AcquisitionType::EI, 0.1);
    inner.initialize(&config.search_space)?;
    Ok(Box::new(DelegatingAdapter::new(
        inner,
        config,
        "BayesianOptimization",
    )))
}

/// Build the reinforcement-learning controller strategy (LSTM + REINFORCE).
pub(crate) fn reinforcement<T: AdapterFloat>(
    config: &NASConfig<T>,
) -> Result<Box<dyn SearchStrategy<T>>> {
    use crate::search_strategies::ReinforcementLearningSearch;

    require_components(config, "reinforcement-learning")?;

    let mut inner = ReinforcementLearningSearch::<T>::new(64, 2, 0.001);
    inner.initialize(&config.search_space)?;
    Ok(Box::new(DelegatingAdapter::new(
        inner,
        config,
        "ReinforcementLearning",
    )))
}

/// Build the differentiable (DARTS-style) strategy.
///
/// The continuous relaxation is sized from the search space: one operation per
/// entry of the component vocabulary and one edge per allowed component slot.
pub(crate) fn differentiable<T: AdapterFloat>(
    config: &NASConfig<T>,
) -> Result<Box<dyn SearchStrategy<T>>> {
    use crate::search_strategies::DifferentiableSearch;

    require_components(config, "differentiable (DARTS)")?;

    let num_operations = config.search_space.component_types.len().max(2);
    let num_edges = config.search_space.max_components.max(1);
    let inner = DifferentiableSearch::<T>::new(num_operations, num_edges, 1.0, true);
    // `DifferentiableSearch::initialize` is a no-op, but call it for symmetry so
    // future search-space validation there is honored.
    let mut inner = inner;
    inner.initialize(&config.search_space)?;
    Ok(Box::new(DelegatingAdapter::new(
        inner,
        config,
        "Differentiable",
    )))
}

/// Build the progressive (staged-complexity) strategy.
pub(crate) fn progressive<T: AdapterFloat>(
    config: &NASConfig<T>,
) -> Result<Box<dyn SearchStrategy<T>>> {
    require_components(config, "progressive")?;

    let max_phases = config.search_space.max_components.clamp(1, 8);
    let phase_budget = config.population_size.max(4);
    let top_k = (config.population_size / 2).max(1);
    let mut inner =
        crate::search_strategies::ProgressiveNAS::<T>::new(max_phases, phase_budget, top_k);
    inner.initialize(&config.search_space)?;
    Ok(Box::new(DelegatingAdapter::new(
        inner,
        config,
        "Progressive",
    )))
}

/// Build the neural-predictor-guided strategy.
///
/// The predictor's input layer must match the architecture encoder's embedding
/// dimension, so both are derived from the same `embedding_dim`.
pub(crate) fn neural_predictor<T: AdapterFloat>(
    config: &NASConfig<T>,
) -> Result<Box<dyn SearchStrategy<T>>> {
    use crate::search_strategies::NeuralPredictorSearch;

    require_components(config, "neural-predictor")?;

    let embedding_dim = 32usize;
    // Input dimension (== embedding_dim) -> hidden -> hidden -> scalar score.
    let predictor_architecture = vec![embedding_dim, 64, 32, 1];
    let mut inner = NeuralPredictorSearch::<T>::new(predictor_architecture, embedding_dim, 0.5);
    inner.initialize(&config.search_space)?;
    Ok(Box::new(DelegatingAdapter::new(
        inner,
        config,
        "NeuralPredictorBased",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::create_minimal_nas_config;

    type Builder = fn(&NASConfig<f64>) -> Result<Box<dyn SearchStrategy<f64>>>;

    fn builders() -> Vec<(&'static str, Builder)> {
        vec![
            ("BayesianOptimization", bayesian::<f64> as Builder),
            ("ReinforcementLearning", reinforcement::<f64> as Builder),
            ("Differentiable", differentiable::<f64> as Builder),
            ("Progressive", progressive::<f64> as Builder),
            ("NeuralPredictorBased", neural_predictor::<f64> as Builder),
        ]
    }

    /// Every adapter must produce a real, non-empty candidate batch. Arity alone
    /// passes `cargo check`; only running them proves the wiring is honest.
    #[test]
    fn every_adapter_generates_non_empty_candidates() {
        let config = create_minimal_nas_config::<f64>();
        let history = VecDeque::new();

        for (name, build) in builders() {
            let mut strategy = build(&config).unwrap_or_else(|e| {
                panic!("{name} adapter must build from a populated config: {e}")
            });
            assert_eq!(strategy.strategy_name(), name);
            assert!(!strategy.has_converged());

            let candidates = strategy
                .generate_candidates(&history)
                .unwrap_or_else(|e| panic!("{name} must generate candidates: {e}"));
            assert_eq!(
                candidates.len(),
                config.population_size,
                "{name} must fill a full population"
            );
            for candidate in &candidates {
                assert!(
                    !candidate.components.is_empty(),
                    "{name} produced an empty architecture"
                );
            }
        }
    }

    /// A strategy that samples concrete hyperparameters cannot run against an
    /// empty component list; it must say so instead of panicking or fabricating.
    #[test]
    fn adapters_reject_an_empty_search_space() {
        let mut config = create_minimal_nas_config::<f64>();
        config.search_space.components.clear();

        for (name, build) in builders() {
            match build(&config) {
                Err(crate::error::OptimError::SearchSpaceError(message)) => {
                    assert!(
                        message.contains("components"),
                        "{name} error must name the missing field, got: {message}"
                    );
                }
                Err(other) => panic!("{name} returned an unexpected error: {other}"),
                Ok(_) => panic!("{name} must reject an empty component list"),
            }
        }
    }

    /// Each adapter must learn from evaluated results without erroring — the
    /// engine calls this once per generation.
    #[test]
    fn every_adapter_accepts_results() {
        let config = create_minimal_nas_config::<f64>();
        let history = VecDeque::new();

        for (name, build) in builders() {
            let mut strategy =
                build(&config).unwrap_or_else(|e| panic!("{name} adapter must build: {e}"));
            let candidates = strategy
                .generate_candidates(&history)
                .unwrap_or_else(|e| panic!("{name} must generate candidates: {e}"));

            let results: Vec<SearchResult<f64>> = candidates
                .into_iter()
                .enumerate()
                .map(|(i, architecture)| scored_result(architecture, 0.1 * i as f64))
                .collect();

            strategy
                .update_strategy(&results)
                .unwrap_or_else(|e| panic!("{name} must accept results: {e}"));
        }
    }

    fn scored_result(
        architecture: OptimizerArchitecture<f64>,
        performance: f64,
    ) -> SearchResult<f64> {
        use crate::nas_engine::results::{
            ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
        };
        use crate::EvaluationMetric;
        use std::collections::HashMap;
        use std::time::Duration;

        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::FinalPerformance, performance);

        SearchResult {
            architecture,
            evaluation_results: EvaluationResults {
                metric_scores,
                overall_score: performance,
                confidence_intervals: HashMap::new(),
                evaluation_time: Duration::from_secs(0),
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
}
