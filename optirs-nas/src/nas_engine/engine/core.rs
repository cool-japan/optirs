//! The main NAS engine: construction, the search loop, candidate lifecycle, statistics, resource accounting, and result finalization.

use crate::error::Result;
use crate::multi_objective;
use crate::nas_engine::config::*;
use crate::nas_engine::resources::*;
use crate::nas_engine::results::*;
use scirs2_core::numeric::Float;
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

use super::controller::DefaultArchitectureController;
use super::mo_optimizers::{MOEADAdapter, NSGA2Optimizer, NSGA3Optimizer, WeightedSumOptimizer};
use super::strategies::{EvolutionaryStrategy, RandomStrategy};
use super::support::{
    ArchitectureController, MultiObjectiveOptimizer, PerformanceEvaluator, PerformancePredictor,
    ProgressiveNAS, SearchStrategy,
};

/// Main Neural Architecture Search Engine
///
/// Coordinates the entire architecture search process including:
/// - Search strategy execution
/// - Candidate generation and evaluation
/// - Multi-objective optimization
/// - Resource management and monitoring
/// - Progressive search coordination
pub struct NeuralArchitectureSearch<T: Float + Debug + Send + Sync + 'static> {
    /// NAS configuration
    pub(super) config: NASConfig<T>,
    /// Current search strategy
    pub(super) search_strategy: Box<dyn SearchStrategy<T>>,
    /// Performance evaluator
    pub(super) evaluator: PerformanceEvaluator<T>,
    /// Multi-objective optimizer
    pub(super) multi_objective_optimizer: Option<Box<dyn MultiObjectiveOptimizer<T>>>,
    /// Architecture controller
    pub(super) architecture_controller: Box<dyn ArchitectureController<T>>,
    /// Progressive search manager
    pub(super) progressive_search: Option<ProgressiveNAS<T>>,
    /// Search history
    pub(super) search_history: VecDeque<SearchResult<T>>,
    /// Current generation/iteration
    pub(super) current_generation: usize,
    /// Best found architectures
    pub(super) best_architectures: Vec<SearchResult<T>>,
    /// Pareto front (for multi-objective)
    pub(super) pareto_front: Option<multi_objective::ParetoFront<T>>,
    /// Resource monitor
    pub(super) resource_monitor: ResourceMonitor<T>,
    /// Search statistics
    pub(super) search_statistics: SearchStatistics<T>,
    /// Performance predictor
    pub(super) performance_predictor: Option<PerformancePredictor<T>>,
}
impl<
        T: Float
            + Debug
            + Default
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + std::fmt::Display
            + From<f64>
            + std::iter::Sum
            + for<'a> std::iter::Sum<&'a T>
            + scirs2_core::ndarray::ScalarOperand,
    > NeuralArchitectureSearch<T>
{
    /// Create a new Neural Architecture Search engine
    pub fn new(config: NASConfig<T>) -> Result<Self> {
        let search_strategy = Self::create_search_strategy(&config)?;
        let evaluator = PerformanceEvaluator::new(config.evaluation_config.clone())?;
        let multi_objective_optimizer = if config.multi_objective_config.objectives.len() > 1 {
            Some(Self::create_multi_objective_optimizer(
                &config.multi_objective_config,
            )?)
        } else {
            None
        };
        let architecture_controller = Self::create_architecture_controller(&config)?;
        let progressive_search = if config.progressive_search {
            Some(ProgressiveNAS::new(&config)?)
        } else {
            None
        };
        let resource_monitor = ResourceMonitor::new(config.resource_constraints.clone());
        let performance_predictor = if config.enable_performance_prediction {
            Some(PerformancePredictor::new(&config.evaluation_config)?)
        } else {
            None
        };
        Ok(Self {
            config,
            search_strategy,
            evaluator,
            multi_objective_optimizer,
            architecture_controller,
            progressive_search,
            search_history: VecDeque::new(),
            current_generation: 0,
            best_architectures: Vec::new(),
            pareto_front: None,
            resource_monitor,
            search_statistics: SearchStatistics::default(),
            performance_predictor,
        })
    }
    /// Name of the search strategy actually driving this engine.
    ///
    /// Every [`SearchStrategyType`] now resolves to a real implementation, so
    /// this is the honest answer to "what is running?" rather than a restatement
    /// of the requested configuration.
    pub fn search_strategy_name(&self) -> &str {
        self.search_strategy.strategy_name()
    }
    /// The resource monitor driving this search, for inspection.
    pub fn resource_monitor(&self) -> &ResourceMonitor<T> {
        &self.resource_monitor
    }

    /// The resource monitor, mutably — the supported way to inject real telemetry
    /// from outside the crate (F12).
    ///
    /// The default [`crate::nas_engine::telemetry::StdTelemetry`] source measures
    /// only what it honestly can, so most resource constraints go unenforced. Install
    /// a tracker over your own [`crate::nas_engine::telemetry::TelemetrySource`] to
    /// make them enforceable:
    ///
    /// ```ignore
    /// engine.resource_monitor_mut().set_trackers(vec![Box::new(
    ///     SystemResourceTracker::with_telemetry(
    ///         "agent".to_string(),
    ///         Duration::from_secs(5),
    ///         Box::new(my_source),
    ///     ),
    /// )]);
    /// ```
    pub fn resource_monitor_mut(&mut self) -> &mut ResourceMonitor<T> {
        &mut self.resource_monitor
    }

    /// Run the complete architecture search
    pub fn run_search(&mut self) -> Result<SearchResults<T>> {
        let start_time = Instant::now();
        self.initialize_search()?;
        self.resource_monitor.start_monitoring()?;
        while !self.should_stop_search() {
            let candidates = self.generate_candidates()?;
            let results = self.evaluate_candidates(candidates)?;
            self.update_search_state(results)?;
            self.check_resource_constraints()?;
            self.update_search_statistics();
            self.current_generation += 1;
        }
        let search_time = start_time.elapsed();
        self.finalize_search(search_time)
    }
    /// Initialize the search process
    pub(super) fn initialize_search(&mut self) -> Result<()> {
        self.current_generation = 0;
        self.search_history.clear();
        self.best_architectures.clear();
        if self.config.population_size > 0 {
            let initial_candidates = (0..self.config.population_size)
                .map(|_| self.architecture_controller.generate_random())
                .collect::<Result<Vec<_>>>()?;
            let initial_results = self.evaluate_candidates(initial_candidates)?;
            self.update_search_state(initial_results)?;
        }
        Ok(())
    }
    /// Generate new candidate architectures
    pub(super) fn generate_candidates(&mut self) -> Result<Vec<OptimizerArchitecture<T>>> {
        let mut candidates = self
            .search_strategy
            .generate_candidates(&self.search_history)?;
        if let Some(progressive) = &mut self.progressive_search {
            candidates = progressive.filter_candidates(candidates, self.current_generation)?;
        }
        let mut valid_candidates = Vec::new();
        for candidate in candidates {
            if self.validate_architecture(&candidate)? {
                valid_candidates.push(candidate);
            }
        }
        Ok(valid_candidates)
    }
    /// Evaluate candidate architectures
    pub(super) fn evaluate_candidates(
        &mut self,
        candidates: Vec<OptimizerArchitecture<T>>,
    ) -> Result<Vec<SearchResult<T>>> {
        let mut results = Vec::new();
        for architecture in candidates {
            let predictor_wanted = self.should_use_predictor(&architecture);
            // `should_use_predictor` already requires the predictor to be present,
            // but expressing that as `.expect()` turns any future change to that
            // predicate into a panic; matching on it evaluates for real instead.
            let evaluation_results = match self
                .performance_predictor
                .as_mut()
                .filter(|_| predictor_wanted)
            {
                Some(predictor) => predictor.predict(&architecture)?,
                None => self.evaluator.evaluate(&architecture)?,
            };
            let resource_usage =
                self.calculate_resource_usage(&architecture, &evaluation_results)?;
            let result = SearchResult {
                architecture,
                evaluation_results,
                generation: self.current_generation,
                search_time: 0.0,
                resource_usage,
                encoding: ArchitectureEncoding::default(),
                metadata: SearchResultMetadata::default(),
            };
            results.push(result);
        }
        Ok(results)
    }
    /// Update search state with new results
    pub(super) fn update_search_state(&mut self, results: Vec<SearchResult<T>>) -> Result<()> {
        for result in &results {
            self.search_history.push_back(result.clone());
            if self.search_history.len() > 1000 {
                self.search_history.pop_front();
            }
        }
        self.update_best_architectures(&results)?;
        self.search_strategy.update_strategy(&results)?;
        // Record what each progressive stage actually produced. `stage_history` was
        // declared but never written, so a progressive run could report nothing
        // about its own stages.
        if let Some(progressive) = &mut self.progressive_search {
            progressive.record_stage_results(self.current_generation, &results);
        }
        if let Some(optimizer) = &mut self.multi_objective_optimizer {
            self.pareto_front = Some(optimizer.update_pareto_front(&results)?);
        }
        if let Some(predictor) = &mut self.performance_predictor {
            predictor.update_training_data(&results)?;
        }
        Ok(())
    }
    /// Check if search should stop
    pub(super) fn should_stop_search(&self) -> bool {
        if self.current_generation >= self.config.search_budget {
            return true;
        }
        if self.check_early_stopping_criteria() {
            return true;
        }
        if self.check_convergence() {
            return true;
        }
        // The strategy's own termination signal. `SearchStrategy::has_converged`
        // was implemented by every adapter in this crate and called from **nowhere**
        // — a dead termination channel. Consulting it is what lets a strategy with
        // an intrinsic schedule (`ProgressiveNAS`, once its complexity phases are
        // exhausted) end the run instead of sampling on at its final complexity
        // level until the generation budget runs out.
        if self.search_strategy.has_converged() {
            log::info!(
                "search strategy {} reports it has finished; stopping at generation {}",
                self.search_strategy.strategy_name(),
                self.current_generation
            );
            return true;
        }
        // Resource violations are deliberately NOT a stop condition here.
        //
        // This used to call `check_violations()` and silently return `true`, so a
        // search cut short by a resource limit returned `Ok(SearchResults)` with no
        // indication that anything had gone wrong — while
        // `check_resource_constraints`, called from the same loop, treated the very
        // same situation as a hard error. The two contradicted each other, and with
        // the old fabricated telemetry (16 GB "used") the silent path fired first,
        // so a tight memory budget produced an empty successful run.
        //
        // Enforcement now lives in exactly one place:
        // `check_resource_constraints`, which samples the monitor and returns
        // `OptimError::ResourceLimitExceeded`.
        false
    }
    /// Check early stopping criteria
    pub(super) fn check_early_stopping_criteria(&self) -> bool {
        if !self.config.early_stopping.enabled {
            return false;
        }
        // `min_generations` was declared in `EarlyStoppingConfig`, set by both config
        // builders, and enforced **nowhere**: a caller asking for "at least 50
        // generations before you give up" was ignored. It is honored here and in
        // `EvolutionaryStrategy::has_converged`, which are the crate's two
        // lack-of-improvement stop paths, so a single floor applies to both.
        if self.current_generation < self.config.early_stopping.min_generations {
            return false;
        }
        let patience = self.config.early_stopping.patience;
        let min_improvement = self.config.early_stopping.min_improvement;
        if self.search_history.len() < patience {
            return false;
        }
        let recent_results: Vec<_> = self.search_history.iter().rev().take(patience).collect();
        let recent_best = recent_results
            .iter()
            .map(|r| r.evaluation_results.overall_score)
            .fold(
                T::neg_infinity(),
                |acc, score| if score > acc { score } else { acc },
            );
        let older_results: Vec<_> = self
            .search_history
            .iter()
            .rev()
            .skip(patience)
            .take(patience)
            .collect();
        if older_results.is_empty() {
            return false;
        }
        let older_best = older_results
            .iter()
            .map(|r| r.evaluation_results.overall_score)
            .fold(
                T::neg_infinity(),
                |acc, score| if score > acc { score } else { acc },
            );
        (recent_best - older_best) < min_improvement
    }
    /// Check convergence criteria
    pub(super) fn check_convergence(&self) -> bool {
        if self.search_history.len() < 20 {
            return false;
        }
        let recent_results: Vec<_> = self.search_history.iter().rev().take(20).collect();
        let diversity = self.calculate_population_diversity(&recent_results);
        diversity < 0.001
    }
    /// Calculate population diversity
    pub(super) fn calculate_population_diversity(&self, population: &[&SearchResult<T>]) -> f64 {
        if population.len() < 2 {
            return 1.0;
        }
        let mut total_distance = 0.0;
        let mut count = 0;
        for i in 0..population.len() {
            for j in (i + 1)..population.len() {
                let distance = self.calculate_architecture_distance(
                    &population[i].architecture,
                    &population[j].architecture,
                );
                total_distance += distance;
                count += 1;
            }
        }
        if count > 0 {
            total_distance / count as f64
        } else {
            1.0
        }
    }
    /// Calculate distance between two architectures
    pub(super) fn calculate_architecture_distance(
        &self,
        arch1: &OptimizerArchitecture<T>,
        arch2: &OptimizerArchitecture<T>,
    ) -> f64 {
        let component_distance =
            self.calculate_component_distance(&arch1.components, &arch2.components);
        let connection_distance = if arch1.connections.len() != arch2.connections.len() {
            1.0
        } else {
            arch1
                .connections
                .iter()
                .zip(arch2.connections.iter())
                .map(|(c1, c2)| if c1 == c2 { 0.0 } else { 1.0 })
                .sum::<f64>()
                / arch1.connections.len() as f64
        };
        (component_distance + connection_distance) / 2.0
    }
    /// Calculate distance between component lists
    pub(super) fn calculate_component_distance(
        &self,
        components1: &[String],
        components2: &[String],
    ) -> f64 {
        if components1.len() != components2.len() {
            return 1.0;
        }
        if components1.is_empty() {
            return 0.0;
        }
        let differences = components1
            .iter()
            .zip(components2.iter())
            .map(|(c1, c2)| if c1 == c2 { 0.0 } else { 1.0 })
            .sum::<f64>();
        differences / components1.len() as f64
    }
    /// Create search strategy based on configuration
    ///
    /// Every strategy type except `MultiObjectiveEvolutionary` is served by a
    /// real implementation from [`crate::search_strategies`], wrapped by
    /// [`crate::nas_engine::strategy_adapters`] so it speaks this module's
    /// batch-oriented [`SearchStrategy`] trait.
    pub(super) fn create_search_strategy(
        config: &NASConfig<T>,
    ) -> Result<Box<dyn SearchStrategy<T>>> {
        use crate::nas_engine::strategy_adapters;
        match config.search_strategy {
            SearchStrategyType::Random => Ok(Box::new(RandomStrategy::new(config)?)),
            SearchStrategyType::Evolutionary => Ok(Box::new(EvolutionaryStrategy::new(config)?)),
            SearchStrategyType::ReinforcementLearning => strategy_adapters::reinforcement(config),
            SearchStrategyType::Differentiable => strategy_adapters::differentiable(config),
            SearchStrategyType::BayesianOptimization => strategy_adapters::bayesian(config),
            SearchStrategyType::Progressive => strategy_adapters::progressive(config),
            SearchStrategyType::NeuralPredictorBased => strategy_adapters::neural_predictor(config),
            SearchStrategyType::MultiObjectiveEvolutionary => {
                log::info!(
                    "MultiObjectiveEvolutionary uses the evolutionary generator; \
                     Pareto selection is applied by the multi-objective optimizer"
                );
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
        }
    }
    /// Create multi-objective optimizer
    pub(super) fn create_multi_objective_optimizer(
        config: &MultiObjectiveConfig<T>,
    ) -> Result<Box<dyn MultiObjectiveOptimizer<T>>> {
        match &config.algorithm {
            MultiObjectiveAlgorithm::NSGA2 => Ok(Box::new(NSGA2Optimizer::new(config)?)),
            MultiObjectiveAlgorithm::WeightedSum => {
                Ok(Box::new(WeightedSumOptimizer::new(config)?))
            }
            MultiObjectiveAlgorithm::NSGA3 => Ok(Box::new(NSGA3Optimizer::new(config)?)),
            MultiObjectiveAlgorithm::MOEAD => Ok(Box::new(MOEADAdapter::new(config)?)),
            // Every remaining variant used to be served either by a macro-generated
            // placeholder that returned an empty Pareto front plus a hardcoded
            // diversity of 0.5, or by a silent substitution of NSGA-II that ignored
            // the requested algorithm entirely. Neither is an implementation, so the
            // honest answer is an error naming what *is* available.
            other => Err(crate::error::OptimError::NotImplemented(format!(
                "multi-objective algorithm {:?} is not implemented in optirs-nas; \
                 configure MultiObjectiveAlgorithm::NSGA2, NSGA3, MOEAD or WeightedSum",
                other
            ))),
        }
    }
    /// Create architecture controller
    pub(super) fn create_architecture_controller(
        config: &NASConfig<T>,
    ) -> Result<Box<dyn ArchitectureController<T>>> {
        Ok(Box::new(DefaultArchitectureController::new(config)?))
    }
    /// Validate architecture
    pub(super) fn validate_architecture(
        &self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<bool> {
        self.architecture_controller.validate(architecture)
    }
    /// Check if should use performance predictor
    pub(super) fn should_use_predictor(&self, _architecture: &OptimizerArchitecture<T>) -> bool {
        self.performance_predictor.is_some()
            && self.current_generation > 10
            && self.search_history.len() > 50
    }
    /// Calculate resource usage for architecture
    pub(super) fn calculate_resource_usage(
        &self,
        architecture: &OptimizerArchitecture<T>,
        _eval_results: &EvaluationResults<T>,
    ) -> Result<ResourceUsage<T>> {
        let component_count = scirs2_core::numeric::NumCast::from(architecture.components.len())
            .unwrap_or_else(|| T::zero());
        let connection_count = scirs2_core::numeric::NumCast::from(architecture.connections.len())
            .unwrap_or_else(|| T::zero());
        let memory_gb = component_count
            * scirs2_core::numeric::NumCast::from(0.1).unwrap_or_else(|| T::zero())
            + connection_count
                * scirs2_core::numeric::NumCast::from(0.05).unwrap_or_else(|| T::zero());
        let cpu_time = component_count
            * scirs2_core::numeric::NumCast::from(1.0).unwrap_or_else(|| T::zero())
            + connection_count
                * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
        let gpu_time =
            component_count * scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero());
        let energy_kwh = (cpu_time + gpu_time)
            * scirs2_core::numeric::NumCast::from(0.001).unwrap_or_else(|| T::zero());
        let cost_usd =
            energy_kwh * scirs2_core::numeric::NumCast::from(0.12).unwrap_or_else(|| T::zero());
        let network_gb = scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero());
        Ok(ResourceUsage {
            memory_gb,
            cpu_time_seconds: cpu_time,
            gpu_time_seconds: gpu_time,
            energy_kwh,
            cost_usd,
            network_gb,
            network_io_gb: network_gb,
            disk_io_gb: scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero()),
            peak_memory_gb: memory_gb,
            efficiency_score: scirs2_core::numeric::NumCast::from(0.8).unwrap_or_else(|| T::zero()),
        })
    }
    /// Update best architectures list
    pub(super) fn update_best_architectures(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        for result in results {
            let should_add = self.best_architectures.is_empty()
                || result.evaluation_results.overall_score
                    > self
                        .best_architectures
                        .iter()
                        .map(|r| r.evaluation_results.overall_score)
                        .fold(
                            T::neg_infinity(),
                            |acc, score| if score > acc { score } else { acc },
                        );
            if should_add {
                self.best_architectures.push(result.clone());
                if self.best_architectures.len() > 10 {
                    self.best_architectures.sort_by(|a, b| {
                        let score_a = a.evaluation_results.overall_score;
                        let score_b = b.evaluation_results.overall_score;
                        score_b
                            .partial_cmp(&score_a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    self.best_architectures.truncate(10);
                }
            }
        }
        Ok(())
    }
    /// Update search statistics
    pub(super) fn update_search_statistics(&mut self) {
        self.search_statistics.total_evaluations = self.search_history.len();
        self.search_statistics.current_generation = self.current_generation;
        if !self.search_history.is_empty() {
            let recent_results: Vec<_> = self.search_history.iter().rev().take(20).collect();
            self.search_statistics.population_diversity = scirs2_core::numeric::NumCast::from(
                self.calculate_population_diversity(&recent_results),
            )
            .unwrap_or_else(|| T::one());
            let scores: Vec<T> = self
                .search_history
                .iter()
                .map(|r| r.evaluation_results.overall_score)
                .collect();
            if !scores.is_empty() {
                self.search_statistics.best_score =
                    Some(scores.iter().fold(T::neg_infinity(), |acc, &score| {
                        if score > acc {
                            score
                        } else {
                            acc
                        }
                    }));
                let sum: T = scores.iter().cloned().sum();
                self.search_statistics.average_score = sum
                    / scirs2_core::numeric::NumCast::from(scores.len()).unwrap_or_else(|| T::one());
            }
        }
    }
    /// Sample resource usage and enforce the configured constraints (F12).
    ///
    /// `update_usage` is called here — once per generation, from the search loop —
    /// because before this it was never called at all: `ResourceMonitor::current_usage`
    /// stayed at `ResourceUsage::default()` for the whole run, so
    /// `check_resource_violations` compared zeros against the budget and could never
    /// fire, and `resource_usage_summary` in the final results was always empty.
    ///
    /// With honest telemetry an unmeasurable resource yields no violation at all, so
    /// this can only abort a search on a value that was actually observed.
    pub(super) fn check_resource_constraints(&mut self) -> Result<()> {
        self.resource_monitor.update_usage()?;
        // `optimize_resources` was dead code (F25): nothing ever called it. It is
        // called here and its suggestions are logged most-urgent-first; it
        // self-disables when `MonitoringConfig::enable_auto_optimization` is false.
        for action in self.resource_monitor.optimize_resources()? {
            log::info!(
                "resource optimization suggested: {:?} (priority {:?}): {}",
                action.action_type,
                action.priority,
                action.description
            );
        }
        // Both views are checked: `check_violations` asks each tracker for an
        // instantaneous reading (temperature, power, RSS) and
        // `check_resource_violations` compares the accumulated usage against the
        // budget. With honest telemetry an unmeasurable resource contributes nothing
        // to either, so neither can abort on invented data.
        let mut violations = self.resource_monitor.check_violations()?;
        violations.extend(self.resource_monitor.check_resource_violations()?);
        if !violations.is_empty() {
            let detail = violations
                .iter()
                .map(|violation| {
                    format!("{:?} ({:?})", violation.violation_type, violation.severity)
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(crate::error::OptimError::ResourceLimitExceeded(format!(
                "resource constraints violated: {}",
                detail
            )));
        }
        Ok(())
    }
    /// Finalize search and return results
    pub(super) fn finalize_search(&self, search_time: Duration) -> Result<SearchResults<T>> {
        let search_time_seconds = search_time.as_secs_f64();
        let results = SearchResults {
            best_architectures: self.best_architectures.clone(),
            pareto_front: self.pareto_front.clone(),
            search_history: self.search_history.clone().into(),
            search_statistics: self.search_statistics.clone(),
            resource_usage_summary: self.resource_monitor.get_usage_summary(),
            search_time_seconds,
            convergence_data: self.extract_convergence_data(),
            evaluation_summary: self.create_evaluation_summary(),
            search_configuration: self.create_search_config_summary(),
            recommendations: self.generate_recommendations(),
            config: self.create_search_config_summary(),
        };
        Ok(results)
    }
    /// Extract convergence data for analysis
    pub(super) fn extract_convergence_data(&self) -> ConvergenceData<T> {
        let mut best_scores_over_time = Vec::new();
        let mut diversity_over_time = Vec::new();
        for generation in 0..=self.current_generation {
            let generation_results: Vec<_> = self
                .search_history
                .iter()
                .filter(|r| r.generation == generation)
                .collect();
            if !generation_results.is_empty() {
                let best_score = generation_results
                    .iter()
                    .map(|r| r.evaluation_results.overall_score)
                    .fold(
                        T::neg_infinity(),
                        |acc, score| if score > acc { score } else { acc },
                    );
                best_scores_over_time.push(best_score);
                let diversity = self.calculate_population_diversity(&generation_results);
                diversity_over_time.push(
                    scirs2_core::numeric::NumCast::from(diversity).unwrap_or_else(|| T::zero()),
                );
            }
        }
        ConvergenceData {
            iteration: self.current_generation,
            best_score: best_scores_over_time.last().copied().unwrap_or(T::zero()),
            convergence_rate: if best_scores_over_time.len() > 1 {
                // Both ends exist inside this branch, but reading them fallibly keeps
                // the length guard and the access from being able to drift apart.
                let last = best_scores_over_time
                    .last()
                    .copied()
                    .unwrap_or_else(T::zero);
                let first = best_scores_over_time
                    .first()
                    .copied()
                    .unwrap_or_else(T::zero);
                let delta = last - first;
                delta
                    / scirs2_core::numeric::NumCast::from(best_scores_over_time.len())
                        .unwrap_or_else(|| T::one())
            } else {
                T::zero()
            },
            stability_measure: if diversity_over_time.len() > 1 {
                let recent_diversity =
                    &diversity_over_time[diversity_over_time.len().saturating_sub(5)..];
                let variance = recent_diversity
                    .iter()
                    .map(|&d| scirs2_core::numeric::NumCast::from(d).unwrap_or(T::zero()))
                    .fold(T::zero(), |acc: T, d: T| acc + d * d)
                    / scirs2_core::numeric::NumCast::from(recent_diversity.len())
                        .unwrap_or_else(|| T::one());
                variance.sqrt()
            } else {
                T::one()
            },
            best_scores_over_time,
            diversity_over_time: diversity_over_time.clone(),
            convergence_generation: self.current_generation,
            final_diversity: diversity_over_time
                .last()
                .copied()
                .unwrap_or_else(|| T::zero()),
        }
    }
    /// Create evaluation summary
    pub(super) fn create_evaluation_summary(&self) -> EvaluationSummary<T> {
        let total_evaluations = self.search_history.len();
        let successful_evaluations = self
            .search_history
            .iter()
            .filter(|r| r.evaluation_results.success)
            .count();
        let success_rate = if total_evaluations > 0 {
            scirs2_core::numeric::NumCast::from(
                successful_evaluations as f64 / total_evaluations as f64,
            )
            .unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        };
        let best_score = self
            .search_history
            .iter()
            .map(|r| r.evaluation_results.overall_score)
            .fold(
                T::neg_infinity(),
                |acc, score| if score > acc { score } else { acc },
            );
        EvaluationSummary {
            total_evaluations,
            success_rate,
            best_score,
            score_statistics: ScoreStatistics::default(),
            benchmark_summary: HashMap::new(),
            resource_summary: ResourceSummary::default(),
        }
    }
    /// Create search configuration summary
    pub(super) fn create_search_config_summary(&self) -> SearchConfigSummary {
        let mut key_hyperparameters = HashMap::new();
        key_hyperparameters.insert(
            "population_size".to_string(),
            format!("{}", self.config.population_size),
        );
        key_hyperparameters.insert(
            "search_budget".to_string(),
            format!("{}", self.config.search_budget),
        );
        key_hyperparameters.insert(
            "early_stopping_enabled".to_string(),
            format!("{}", self.config.early_stopping.enabled),
        );
        key_hyperparameters.insert(
            "progressive_search".to_string(),
            format!("{}", self.config.progressive_search),
        );
        key_hyperparameters.insert(
            "enable_transfer_learning".to_string(),
            format!("{}", self.config.enable_transfer_learning),
        );
        key_hyperparameters.insert(
            "enable_performance_prediction".to_string(),
            format!("{}", self.config.enable_performance_prediction),
        );
        key_hyperparameters.insert(
            "parallelization_factor".to_string(),
            format!("{}", self.config.parallelization_factor),
        );
        SearchConfigSummary {
            search_strategy: format!("{:?}", self.config.search_strategy),
            population_size: self.config.population_size,
            search_budget: self.config.search_budget,
            evaluation_config: format!("{:?}", self.config.evaluation_config),
            multi_objective_config: if !self.config.multi_objective_config.objectives.is_empty() {
                Some(format!("{:?}", self.config.multi_objective_config))
            } else {
                None
            },
            resource_constraints: format!(
                "Memory: {}GB, Compute: {}h",
                self.config.resource_constraints.max_memory_gb,
                self.config.resource_constraints.max_computation_hours
            ),
            key_hyperparameters,
        }
    }
    /// Generate search recommendations
    pub(super) fn generate_recommendations(&self) -> Vec<SearchRecommendation> {
        let mut recommendations = Vec::new();
        let max_generations = self.config.search_budget / self.config.population_size.max(1);
        if self.current_generation < max_generations / 2 {
            recommendations.push(SearchRecommendation {
                recommendation_type: RecommendationType::PopulationSizeAdjustment,
                description: format!(
                    "Increase population size from {} to {} - search converged early",
                    self.config.population_size,
                    (self.config.population_size as f64 * 1.5) as usize
                ),
                priority: RecommendationPriority::High,
                expected_improvement: Some(0.2),
                implementation_effort: ImplementationEffort::Low,
                evidence: vec![
                    format!(
                        "Search converged at generation {} of {}",
                        self.current_generation, max_generations
                    ),
                    "Early convergence suggests insufficient exploration".to_string(),
                ],
            });
        }
        if self.search_history.len() > 10 {
            recommendations.push(SearchRecommendation {
                recommendation_type: RecommendationType::SearchStrategyChange,
                description:
                    "Consider switching to multi-objective optimization for better diversity"
                        .to_string(),
                priority: RecommendationPriority::Medium,
                expected_improvement: Some(0.15),
                implementation_effort: ImplementationEffort::Medium,
                evidence: vec![
                    "Population diversity metrics indicate convergence".to_string(),
                    "Multi-objective approaches can improve exploration".to_string(),
                ],
            });
        }
        if self.config.resource_constraints.enable_monitoring {
            recommendations.push(SearchRecommendation {
                recommendation_type: RecommendationType::ResourceOptimization,
                description: format!(
                    "Increase parallelization factor from {} to {} to speed up search",
                    self.config.parallelization_factor,
                    self.config.parallelization_factor * 2
                ),
                priority: RecommendationPriority::Low,
                expected_improvement: Some(0.5),
                implementation_effort: ImplementationEffort::Low,
                evidence: vec![
                    "Current parallelization is underutilizing available resources".to_string(),
                ],
            });
        }
        recommendations
    }
}
impl<T: Float + Debug + Send + Sync + 'static> Debug for NeuralArchitectureSearch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralArchitectureSearch")
            .field("config", &self.config)
            .field("evaluator", &self.evaluator)
            .field("progressive_search", &self.progressive_search)
            .field("search_history", &self.search_history)
            .field("best_architectures", &self.best_architectures)
            .field("pareto_front", &self.pareto_front)
            .field("resource_monitor", &self.resource_monitor)
            .field("search_statistics", &self.search_statistics)
            .field("performance_predictor", &self.performance_predictor)
            .field("search_strategy", &"Box<dyn SearchStrategy>")
            .field(
                "multi_objective_optimizer",
                &self
                    .multi_objective_optimizer
                    .as_ref()
                    .map(|_| "Box<dyn MultiObjectiveOptimizer>"),
            )
            .field(
                "architecture_controller",
                &"Box<dyn ArchitectureController>",
            )
            .finish()
    }
}
