// Neural Architecture Search Engine
//
// This module implements the core NAS engine that coordinates the entire
// architecture search process, including candidate generation, evaluation,
// and optimization strategy execution.

use super::config::*;
use super::resources::*;
use super::results::*;
use crate::error::Result;
use crate::multi_objective;
use crate::EvaluationMetric;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    config: NASConfig<T>,

    /// Current search strategy
    search_strategy: Box<dyn SearchStrategy<T>>,

    /// Performance evaluator
    evaluator: PerformanceEvaluator<T>,

    /// Multi-objective optimizer
    multi_objective_optimizer: Option<Box<dyn MultiObjectiveOptimizer<T>>>,

    /// Architecture controller
    architecture_controller: Box<dyn ArchitectureController<T>>,

    /// Progressive search manager
    progressive_search: Option<ProgressiveNAS<T>>,

    /// Search history
    search_history: VecDeque<SearchResult<T>>,

    /// Current generation/iteration
    current_generation: usize,

    /// Best found architectures
    best_architectures: Vec<SearchResult<T>>,

    /// Pareto front (for multi-objective)
    pareto_front: Option<multi_objective::ParetoFront<T>>,

    /// Resource monitor
    resource_monitor: ResourceMonitor<T>,

    /// Search statistics
    search_statistics: SearchStatistics<T>,

    /// Performance predictor
    performance_predictor: Option<PerformancePredictor<T>>,
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

/// Performance evaluator for architectures
#[derive(Debug)]
pub struct PerformanceEvaluator<T: Float + Debug + Send + Sync + 'static> {
    config: EvaluationConfig<T>,
    evaluation_cache: Arc<Mutex<HashMap<String, EvaluationResults<T>>>>,
    evaluation_count: usize,
}

/// Progressive NAS implementation
#[derive(Debug)]
pub struct ProgressiveNAS<T: Float + Debug + Send + Sync + 'static> {
    stages: Vec<ProgressiveStage<T>>,
    current_stage: usize,
    stage_history: Vec<Vec<SearchResult<T>>>,
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

/// Performance prediction system
#[derive(Debug)]
pub struct PerformancePredictor<T: Float + Debug + Send + Sync + 'static> {
    model_type: PredictorType,
    training_data: Vec<(OptimizerArchitecture<T>, EvaluationResults<T>)>,
    prediction_accuracy: T,
    confidence_threshold: T,
}

/// Types of predictors available
#[derive(Debug, Clone)]
pub enum PredictorType {
    NeuralNetwork,
    GaussianProcess,
    RandomForest,
    Ensemble,
}

// Note: Using ParetoFront from multi_objective module instead
// #[derive(Debug, Clone)]
// pub struct ParetoFront<T: Float + Debug + Send + Sync + 'static> {
//     pub solutions: Vec<SearchResult<T>>,
//     pub hypervolume: T,
//     pub diversity_metrics: DiversityMetrics<T>,
//     pub generation: usize,
// }

/// Diversity metrics for population
#[derive(Debug, Clone)]
pub struct DiversityMetrics<T: Float + Debug + Send + Sync + 'static> {
    pub crowding_distance: Vec<T>,
    pub entropy: T,
    pub average_distance: T,
    pub min_distance: T,
    pub max_distance: T,
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
        // Initialize search strategy
        let search_strategy = Self::create_search_strategy(&config)?;

        // Initialize evaluator
        let evaluator = PerformanceEvaluator::new(config.evaluation_config.clone())?;

        // Initialize multi-objective optimizer if needed
        let multi_objective_optimizer = if config.multi_objective_config.objectives.len() > 1 {
            Some(Self::create_multi_objective_optimizer(
                &config.multi_objective_config,
            )?)
        } else {
            None
        };

        // Initialize architecture controller
        let architecture_controller = Self::create_architecture_controller(&config)?;

        // Initialize progressive search if enabled
        let progressive_search = if config.progressive_search {
            Some(ProgressiveNAS::new(&config)?)
        } else {
            None
        };

        // Initialize resource monitor
        let resource_monitor = ResourceMonitor::new(config.resource_constraints.clone());

        // Initialize performance predictor if enabled
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

    /// Run the complete architecture search
    pub fn run_search(&mut self) -> Result<SearchResults<T>> {
        let start_time = Instant::now();

        // Initialize search
        self.initialize_search()?;

        // Start resource monitoring
        self.resource_monitor.start_monitoring()?;

        // Main search loop
        while !self.should_stop_search() {
            // Generate candidate architectures
            let candidates = self.generate_candidates()?;

            // Evaluate candidates
            let results = self.evaluate_candidates(candidates)?;

            // Update search state
            self.update_search_state(results)?;

            // Check resource constraints
            self.check_resource_constraints()?;

            // Update statistics
            self.update_search_statistics();

            self.current_generation += 1;
        }

        // Finalize search and return results
        let search_time = start_time.elapsed();
        self.finalize_search(search_time)
    }

    /// Initialize the search process
    fn initialize_search(&mut self) -> Result<()> {
        // Reset counters and history
        self.current_generation = 0;
        self.search_history.clear();
        self.best_architectures.clear();

        // Generate initial population if needed
        if self.config.population_size > 0 {
            let initial_candidates = (0..self.config.population_size)
                .map(|_| self.architecture_controller.generate_random())
                .collect::<Result<Vec<_>>>()?;

            // Evaluate initial population
            let initial_results = self.evaluate_candidates(initial_candidates)?;
            self.update_search_state(initial_results)?;
        }

        Ok(())
    }

    /// Generate new candidate architectures
    fn generate_candidates(&mut self) -> Result<Vec<OptimizerArchitecture<T>>> {
        // Use search strategy to generate candidates
        let mut candidates = self
            .search_strategy
            .generate_candidates(&self.search_history)?;

        // Apply progressive search if enabled
        if let Some(progressive) = &mut self.progressive_search {
            candidates = progressive.filter_candidates(candidates, self.current_generation)?;
        }

        // Validate all candidates
        let mut valid_candidates = Vec::new();
        for candidate in candidates {
            if self.validate_architecture(&candidate)? {
                valid_candidates.push(candidate);
            }
        }

        Ok(valid_candidates)
    }

    /// Evaluate candidate architectures
    fn evaluate_candidates(
        &mut self,
        candidates: Vec<OptimizerArchitecture<T>>,
    ) -> Result<Vec<SearchResult<T>>> {
        let mut results = Vec::new();

        for architecture in candidates {
            // Check if we should use performance predictor
            let evaluation_results = if self.should_use_predictor(&architecture) {
                self.performance_predictor
                    .as_mut()
                    .expect("unwrap failed")
                    .predict(&architecture)?
            } else {
                self.evaluator.evaluate(&architecture)?
            };

            // Calculate resource usage
            let resource_usage =
                self.calculate_resource_usage(&architecture, &evaluation_results)?;

            // Create search result
            let result = SearchResult {
                architecture,
                evaluation_results,
                generation: self.current_generation,
                search_time: 0.0, // Will be updated later
                resource_usage,
                encoding: ArchitectureEncoding::default(),
                metadata: SearchResultMetadata::default(),
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Update search state with new results
    fn update_search_state(&mut self, results: Vec<SearchResult<T>>) -> Result<()> {
        // Add results to history
        for result in &results {
            self.search_history.push_back(result.clone());

            // Maintain history size limit
            if self.search_history.len() > 1000 {
                self.search_history.pop_front();
            }
        }

        // Update best architectures
        self.update_best_architectures(&results)?;

        // Update search strategy
        self.search_strategy.update_strategy(&results)?;

        // Update multi-objective optimizer if enabled
        if let Some(optimizer) = &mut self.multi_objective_optimizer {
            self.pareto_front = Some(optimizer.update_pareto_front(&results)?);
        }

        // Update performance predictor if enabled
        if let Some(predictor) = &mut self.performance_predictor {
            predictor.update_training_data(&results)?;
        }

        Ok(())
    }

    /// Check if search should stop
    fn should_stop_search(&self) -> bool {
        // Check generation limit
        if self.current_generation >= self.config.search_budget {
            return true;
        }

        // Check early stopping criteria
        if self.check_early_stopping_criteria() {
            return true;
        }

        // Check convergence
        if self.check_convergence() {
            return true;
        }

        // Check resource constraints
        if let Ok(violations) = self.resource_monitor.check_violations() {
            if !violations.is_empty() {
                return true;
            }
        }

        false
    }

    /// Check early stopping criteria
    fn check_early_stopping_criteria(&self) -> bool {
        if !self.config.early_stopping.enabled {
            return false;
        }

        let patience = self.config.early_stopping.patience;
        let min_improvement = self.config.early_stopping.min_improvement;

        if self.search_history.len() < patience {
            return false;
        }

        // Get recent best scores
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

        // Check if improvement is below threshold
        (recent_best - older_best) < min_improvement
    }

    /// Check convergence criteria
    fn check_convergence(&self) -> bool {
        if self.search_history.len() < 20 {
            return false;
        }

        // Calculate population diversity
        let recent_results: Vec<_> = self.search_history.iter().rev().take(20).collect();
        let diversity = self.calculate_population_diversity(&recent_results);

        // Consider converged if diversity is very low
        diversity < 0.001
    }

    /// Calculate population diversity
    fn calculate_population_diversity(&self, population: &[&SearchResult<T>]) -> f64 {
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
    fn calculate_architecture_distance(
        &self,
        arch1: &OptimizerArchitecture<T>,
        arch2: &OptimizerArchitecture<T>,
    ) -> f64 {
        // Simple distance metric based on component differences
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
    fn calculate_component_distance(&self, components1: &[String], components2: &[String]) -> f64 {
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
    fn create_search_strategy(config: &NASConfig<T>) -> Result<Box<dyn SearchStrategy<T>>> {
        match config.search_strategy {
            SearchStrategyType::Random => Ok(Box::new(RandomStrategy::new(config)?)),
            SearchStrategyType::Evolutionary => Ok(Box::new(EvolutionaryStrategy::new(config)?)),
            // NOTE (v1.0.0): Additional strategies supported in future versions
            // RL, Differentiable, and Bayesian strategies require more complex infrastructure
            // that is planned for v1.1.0+. For v1.0.0, we provide Random and Evolutionary
            // as proven baseline strategies.
            SearchStrategyType::ReinforcementLearning => {
                // Fallback to Evolutionary for v1.0.0
                eprintln!("Warning: ReinforcementLearning strategy not fully implemented in v1.0.0, using Evolutionary as fallback");
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
            SearchStrategyType::Differentiable => {
                // Fallback to Random for v1.0.0 (Differentiable requires specialized ops)
                eprintln!("Warning: Differentiable strategy not fully implemented in v1.0.0, using Random as fallback");
                Ok(Box::new(RandomStrategy::new(config)?))
            }
            SearchStrategyType::BayesianOptimization => {
                // Fallback to Evolutionary for v1.0.0
                eprintln!("Warning: BayesianOptimization strategy not fully implemented in v1.0.0, using Evolutionary as fallback");
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
            SearchStrategyType::Progressive => {
                // Fallback to Evolutionary for v1.0.0
                eprintln!("Warning: Progressive strategy not fully implemented in v1.0.0, using Evolutionary as fallback");
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
            SearchStrategyType::MultiObjectiveEvolutionary => {
                // Fallback to Evolutionary for v1.0.0
                eprintln!("Warning: MultiObjectiveEvolutionary strategy not fully implemented in v1.0.0, using Evolutionary as fallback");
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
            SearchStrategyType::NeuralPredictorBased => {
                // Fallback to Evolutionary for v1.0.0
                eprintln!("Warning: NeuralPredictorBased strategy not fully implemented in v1.0.0, using Evolutionary as fallback");
                Ok(Box::new(EvolutionaryStrategy::new(config)?))
            }
        }
    }

    /// Create multi-objective optimizer
    fn create_multi_objective_optimizer(
        config: &MultiObjectiveConfig<T>,
    ) -> Result<Box<dyn MultiObjectiveOptimizer<T>>> {
        match config.algorithm {
            MultiObjectiveAlgorithm::NSGA2 => Ok(Box::new(NSGA2Optimizer::new(config)?)),
            MultiObjectiveAlgorithm::NSGA3 => Ok(Box::new(NSGA3Optimizer::new(config)?)),
            MultiObjectiveAlgorithm::MOEAD => Ok(Box::new(MOEADOptimizer::new(config)?)),
            MultiObjectiveAlgorithm::PAES => Ok(Box::new(PAESOptimizer::new(config)?)),
            MultiObjectiveAlgorithm::SPEA2 => Ok(Box::new(SPEA2Optimizer::new(config)?)),
            MultiObjectiveAlgorithm::WeightedSum => Ok(Box::new(NSGA2Optimizer::new(config)?)), // Fallback to NSGA2
            MultiObjectiveAlgorithm::EpsilonConstraint => {
                Ok(Box::new(NSGA2Optimizer::new(config)?))
            } // Fallback to NSGA2
            MultiObjectiveAlgorithm::GoalProgramming => Ok(Box::new(NSGA2Optimizer::new(config)?)), // Fallback to NSGA2
            MultiObjectiveAlgorithm::Custom(_) => Ok(Box::new(NSGA2Optimizer::new(config)?)), // Fallback to NSGA2
        }
    }

    /// Create architecture controller
    fn create_architecture_controller(
        config: &NASConfig<T>,
    ) -> Result<Box<dyn ArchitectureController<T>>> {
        Ok(Box::new(DefaultArchitectureController::new(config)?))
    }

    /// Validate architecture
    fn validate_architecture(&self, architecture: &OptimizerArchitecture<T>) -> Result<bool> {
        // Use architecture controller for validation
        self.architecture_controller.validate(architecture)
    }

    /// Check if should use performance predictor
    fn should_use_predictor(&self, _architecture: &OptimizerArchitecture<T>) -> bool {
        self.performance_predictor.is_some() &&
        self.current_generation > 10 && // Use predictor after some evaluations
        self.search_history.len() > 50 // Need sufficient training data
    }

    /// Calculate resource usage for architecture
    fn calculate_resource_usage(
        &self,
        architecture: &OptimizerArchitecture<T>,
        _eval_results: &EvaluationResults<T>,
    ) -> Result<ResourceUsage<T>> {
        // Estimate resource usage based on architecture complexity
        let component_count = scirs2_core::numeric::NumCast::from(architecture.components.len())
            .unwrap_or_else(|| T::zero());
        let connection_count = scirs2_core::numeric::NumCast::from(architecture.connections.len())
            .unwrap_or_else(|| T::zero());

        // Simple resource estimation model
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
            energy_kwh * scirs2_core::numeric::NumCast::from(0.12).unwrap_or_else(|| T::zero()); // $0.12 per kWh
        let network_gb = scirs2_core::numeric::NumCast::from(0.01).unwrap_or_else(|| T::zero()); // Minimal network usage

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
    fn update_best_architectures(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        for result in results {
            // Add to best architectures if it's good enough
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

                // Keep only top architectures
                if self.best_architectures.len() > 10 {
                    // Sort by performance and keep best
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
    fn update_search_statistics(&mut self) {
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

    /// Check resource constraints
    fn check_resource_constraints(&mut self) -> Result<()> {
        let violations = self.resource_monitor.check_resource_violations()?;
        if !violations.is_empty() {
            return Err(crate::error::OptimError::ResourceLimitExceeded(format!(
                "Resource constraints violated: {} violations detected",
                violations.len()
            )));
        }
        Ok(())
    }

    /// Finalize search and return results
    fn finalize_search(&self, search_time: Duration) -> Result<SearchResults<T>> {
        let search_time_seconds = search_time.as_secs_f64();

        // Create comprehensive search results
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
    fn extract_convergence_data(&self) -> ConvergenceData<T> {
        let mut best_scores_over_time = Vec::new();
        let mut diversity_over_time = Vec::new();

        // Calculate metrics over generations
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
                let delta = *best_scores_over_time.last().expect("unwrap failed")
                    - *best_scores_over_time.first().expect("unwrap failed");
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
    fn create_evaluation_summary(&self) -> EvaluationSummary<T> {
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
    fn create_search_config_summary(&self) -> SearchConfigSummary {
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
    fn generate_recommendations(&self) -> Vec<SearchRecommendation> {
        let mut recommendations = Vec::new();

        // Check if search converged too early
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

        // Check population diversity
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

        // Check resource usage
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

// Strategy implementations.
//
// `RandomStrategy`, `EvolutionaryStrategy`, and `NSGA2Optimizer` are the only
// three types ever constructed by the factory functions above (every other
// `SearchStrategyType` / `MultiObjectiveAlgorithm` variant falls back to one of
// them). They therefore carry real implementations that delegate to the live,
// tested strategies in `crate::search_strategies` and the complete NSGA-II in
// `crate::multi_objective`. The remaining never-instantiated structs keep a
// trivial macro-generated implementation purely so the trait bounds resolve.

/// Random search strategy wrapper.
///
/// Holds a live [`crate::search_strategies::RandomSearch`] together with the
/// search space and batch size so that `generate_candidates` produces a real
/// batch of randomly-sampled valid architectures.
struct RandomStrategy<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum> {
    inner: crate::search_strategies::RandomSearch<T>,
    search_space: SearchSpaceConfig,
    batch_size: usize,
}

/// Evolutionary search strategy wrapper.
///
/// Holds a live [`crate::search_strategies::EvolutionarySearch`] which performs
/// tournament selection, crossover and mutation from the search history (and
/// falls back to random sampling when history is insufficient). Tracks the
/// best score per generation to drive a real convergence criterion.
struct EvolutionaryStrategy<
    T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum,
> {
    inner: crate::search_strategies::EvolutionarySearch<T>,
    search_space: SearchSpaceConfig,
    batch_size: usize,
    /// Best overall score observed in each `update_strategy` call, used to
    /// detect a lack of improvement over the most recent generations.
    best_score_history: Vec<T>,
    /// Number of trailing generations without improvement that signals
    /// convergence.
    convergence_patience: usize,
}
struct BayesianStrategy<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct ReinforcementStrategy<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct DifferentiableStrategy<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct ProgressiveStrategy<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct HybridStrategy<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

// Multi-objective optimizer implementations.
//
// `NSGA2Optimizer` is the only optimizer the factory ever constructs (all
// `MultiObjectiveAlgorithm` variants fall back to it). It delegates to the
// complete NSGA-II implementation in `crate::multi_objective`. The remaining
// optimizer structs keep trivial macro-generated implementations.

/// NSGA-II multi-objective optimizer wrapper that delegates Pareto-front
/// maintenance, candidate selection and diversity measurement to the complete
/// [`crate::multi_objective::NSGA2`] implementation.
struct NSGA2Optimizer<
    T: Float + Debug + Default + Clone + Send + Sync + 'static + PartialOrd + std::iter::Sum,
> {
    /// Live NSGA-II engine holding population, Pareto front and statistics.
    inner: crate::multi_objective::NSGA2<T>,
    /// Configured objectives (direction + metric mapping) required by NSGA-II
    /// to compute dominance.
    config: MultiObjectiveConfig<T>,
}
struct NSGA3Optimizer<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct MOEADOptimizer<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct PAESOptimizer<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}
struct SPEA2Optimizer<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

// Architecture controller implementation (placeholder)
struct DefaultArchitectureController<T: Float + Debug + Send + Sync + 'static> {
    _phantom: std::marker::PhantomData<T>,
}

// Implementations for PerformanceEvaluator
impl<T: Float + Debug + Send + Sync + 'static> PerformanceEvaluator<T> {
    pub fn new(config: EvaluationConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            evaluation_cache: Arc::new(Mutex::new(HashMap::new())),
            evaluation_count: 0,
        })
    }

    pub fn evaluate(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<EvaluationResults<T>> {
        // Implementation would perform actual evaluation
        // For now, return dummy results
        let mut scores = HashMap::new();
        scores.insert(
            EvaluationMetric::FinalPerformance,
            scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
        );

        Ok(EvaluationResults {
            metric_scores: scores,
            overall_score: scirs2_core::numeric::NumCast::from(0.5).unwrap_or_else(|| T::zero()),
            confidence_intervals: HashMap::new(),
            evaluation_time: Duration::from_secs(1),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: HashMap::new(),
            training_trajectory: Vec::new(),
        })
    }
}

// Implementations for ProgressiveNAS
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
        // Progressive filtering logic would go here
        Ok(candidates)
    }
}

// Implementations for PerformancePredictor
impl<T: Float + Debug + Send + Sync + 'static> PerformancePredictor<T> {
    pub fn new(_config: &EvaluationConfig<T>) -> Result<Self> {
        Ok(Self {
            model_type: PredictorType::NeuralNetwork,
            training_data: Vec::new(),
            prediction_accuracy: scirs2_core::numeric::NumCast::from(0.8)
                .unwrap_or_else(|| T::zero()),
            confidence_threshold: scirs2_core::numeric::NumCast::from(0.7)
                .unwrap_or_else(|| T::zero()),
        })
    }

    pub fn predict(
        &mut self,
        _architecture: &OptimizerArchitecture<T>,
    ) -> Result<EvaluationResults<T>> {
        // Prediction logic would go here
        let mut scores = HashMap::new();
        scores.insert(
            EvaluationMetric::FinalPerformance,
            scirs2_core::numeric::NumCast::from(0.6).unwrap_or_else(|| T::zero()),
        );

        Ok(EvaluationResults {
            metric_scores: scores,
            overall_score: scirs2_core::numeric::NumCast::from(0.6).unwrap_or_else(|| T::zero()),
            confidence_intervals: HashMap::new(),
            evaluation_time: Duration::from_millis(10),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: HashMap::new(),
            training_trajectory: Vec::new(),
        })
    }

    pub fn update_training_data(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        for result in results {
            self.training_data.push((
                result.architecture.clone(),
                result.evaluation_results.clone(),
            ));
        }
        Ok(())
    }
}

// Placeholder implementations for strategy traits
macro_rules! impl_search_strategy {
    ($strategy:ident) => {
        impl<T: Float + Debug + Send + Sync + 'static> $strategy<T> {
            pub fn new(_config: &NASConfig<T>) -> Result<Self> {
                Ok(Self {
                    _phantom: std::marker::PhantomData,
                })
            }
        }

        impl<T: Float + Debug + Send + Sync + 'static> SearchStrategy<T> for $strategy<T> {
            fn generate_candidates(
                &mut self,
                _history: &VecDeque<SearchResult<T>>,
            ) -> Result<Vec<OptimizerArchitecture<T>>> {
                Ok(Vec::new())
            }

            fn update_strategy(&mut self, _results: &[SearchResult<T>]) -> Result<()> {
                Ok(())
            }

            fn has_converged(&self) -> bool {
                false
            }

            fn strategy_name(&self) -> &str {
                stringify!($strategy)
            }
        }
    };
}

// NOTE: RandomStrategy and EvolutionaryStrategy have real, hand-written
// implementations below (see `impl ... SearchStrategy for RandomStrategy` and
// `... for EvolutionaryStrategy`). Only the never-instantiated strategies use
// the trivial macro.
impl_search_strategy!(BayesianStrategy);
impl_search_strategy!(ReinforcementStrategy);
impl_search_strategy!(DifferentiableStrategy);
impl_search_strategy!(ProgressiveStrategy);
impl_search_strategy!(HybridStrategy);

// Placeholder implementations for multi-objective optimizers
macro_rules! impl_multi_objective_optimizer {
    ($optimizer:ident) => {
        impl<T: Float + Debug + Send + Sync + 'static> $optimizer<T> {
            pub fn new(_config: &MultiObjectiveConfig<T>) -> Result<Self> {
                Ok(Self {
                    _phantom: std::marker::PhantomData,
                })
            }
        }

        impl<T: Float + Debug + Send + Sync + 'static> MultiObjectiveOptimizer<T>
            for $optimizer<T>
        {
            fn update_pareto_front(
                &mut self,
                _results: &[SearchResult<T>],
            ) -> Result<multi_objective::ParetoFront<T>> {
                use crate::multi_objective::{CoverageMetrics, FrontMetrics, ObjectiveBounds};
                Ok(multi_objective::ParetoFront {
                    solutions: Vec::new(),
                    objective_bounds: ObjectiveBounds {
                        min_values: Vec::new(),
                        max_values: Vec::new(),
                        ideal_point: Vec::new(),
                        nadir_point: Vec::new(),
                    },
                    metrics: FrontMetrics {
                        hypervolume: T::zero(),
                        spread: T::zero(),
                        spacing: T::zero(),
                        convergence: T::zero(),
                        num_solutions: 0,
                        coverage: CoverageMetrics {
                            objective_space_coverage: T::zero(),
                            reference_distance: T::zero(),
                            epsilon_dominance: T::zero(),
                        },
                    },
                    generation: 0,
                    last_updated: std::time::SystemTime::now(),
                })
            }

            fn select_candidates(
                &self,
                candidates: &[SearchResult<T>],
                population_size: usize,
            ) -> Result<Vec<SearchResult<T>>> {
                Ok(candidates.iter().take(population_size).cloned().collect())
            }

            fn calculate_diversity(&self, _population: &[SearchResult<T>]) -> f64 {
                0.5
            }
        }
    };
}

// NOTE: NSGA2Optimizer has a real, hand-written implementation below that
// delegates to `crate::multi_objective::NSGA2`. Only the never-instantiated
// optimizers use the trivial macro.
impl_multi_objective_optimizer!(NSGA3Optimizer);
impl_multi_objective_optimizer!(MOEADOptimizer);
impl_multi_objective_optimizer!(PAESOptimizer);
impl_multi_objective_optimizer!(SPEA2Optimizer);

// ---------------------------------------------------------------------------
// Real RandomStrategy implementation
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    RandomStrategy<T>
{
    /// Construct a random-search strategy from a NAS configuration. A live
    /// [`crate::search_strategies::RandomSearch`] is seeded deterministically
    /// and initialized with the configured search space.
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        let mut inner = crate::search_strategies::RandomSearch::<T>::new(Some(42));
        inner.initialize(&config.search_space)?;

        // Generate the configured population each round; fall back to a small
        // non-zero batch so the search always makes progress.
        let batch_size = if config.population_size > 0 {
            config.population_size
        } else {
            8
        };

        Ok(Self {
            inner,
            search_space: config.search_space.clone(),
            batch_size,
        })
    }

    /// Synthesize a single valid architecture without indexing into the
    /// (possibly empty) component config list. Used when the search space
    /// provides no per-component hyperparameter configs but still declares the
    /// component vocabulary via `component_types`.
    fn synthesize_from_component_types(&self, index: usize) -> OptimizerArchitecture<T> {
        let component_label = self
            .search_space
            .component_types
            .first()
            .map(|ct| format!("{:?}", ct))
            .unwrap_or_else(|| "Adam".to_string());

        OptimizerArchitecture {
            components: vec![component_label],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: format!("random_arch_{}", index),
        }
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum> SearchStrategy<T>
    for RandomStrategy<T>
{
    fn generate_candidates(
        &mut self,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        let mut candidates = Vec::with_capacity(self.batch_size);
        for idx in 0..self.batch_size {
            // The live RandomSearch indexes into `search_space.components`; when
            // that list is empty we synthesize a valid architecture instead of
            // delegating (which would panic on an empty slice).
            if self.search_space.components.is_empty() {
                candidates.push(self.synthesize_from_component_types(idx));
            } else {
                let architecture = self
                    .inner
                    .generate_architecture(&self.search_space, history)?;
                candidates.push(architecture);
            }
        }
        Ok(candidates)
    }

    fn update_strategy(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;
        self.inner.update_with_results(results)
    }

    fn has_converged(&self) -> bool {
        // Random search never converges by construction; it keeps exploring.
        false
    }

    fn strategy_name(&self) -> &str {
        "RandomStrategy"
    }
}

// ---------------------------------------------------------------------------
// Real EvolutionaryStrategy implementation
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum>
    EvolutionaryStrategy<T>
{
    /// Construct an evolutionary strategy from a NAS configuration, wrapping a
    /// live [`crate::search_strategies::EvolutionarySearch`] sized to the
    /// configured population.
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        let population_size = config.population_size.max(2);
        // Reuse the early-stopping patience as the convergence window when
        // available; otherwise fall back to a sensible default.
        let convergence_patience = config.early_stopping.patience.max(5);

        let mut inner = crate::search_strategies::EvolutionarySearch::<T>::new(
            population_size,
            0.1, // mutation rate
            0.8, // crossover rate
            3,   // tournament size
        );
        inner.initialize(&config.search_space)?;

        let batch_size = if config.population_size > 0 {
            config.population_size
        } else {
            8
        };

        Ok(Self {
            inner,
            search_space: config.search_space.clone(),
            batch_size,
            best_score_history: Vec::new(),
            convergence_patience,
        })
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + std::iter::Sum> SearchStrategy<T>
    for EvolutionaryStrategy<T>
{
    fn generate_candidates(
        &mut self,
        history: &VecDeque<SearchResult<T>>,
    ) -> Result<Vec<OptimizerArchitecture<T>>> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        // The live EvolutionarySearch performs tournament selection, crossover
        // and mutation from `history` when enough evaluated results exist, and
        // otherwise returns members of its (randomly-seeded) population. Calling
        // it `batch_size` times yields a generation of offspring.
        let mut candidates = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            let architecture = self
                .inner
                .generate_architecture(&self.search_space, history)?;
            candidates.push(architecture);
        }
        Ok(candidates)
    }

    fn update_strategy(&mut self, results: &[SearchResult<T>]) -> Result<()> {
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        // Record the best overall score of this generation for convergence
        // detection before forwarding the results to the inner strategy (which
        // adapts its mutation rate and tracks performance statistics).
        if !results.is_empty() {
            let best = results
                .iter()
                .map(|r| r.evaluation_results.overall_score)
                .fold(
                    T::neg_infinity(),
                    |acc, score| {
                        if score > acc {
                            score
                        } else {
                            acc
                        }
                    },
                );
            if best > T::neg_infinity() {
                self.best_score_history.push(best);
            }
        }

        self.inner.update_with_results(results)
    }

    fn has_converged(&self) -> bool {
        // Converged when the best score has not improved over the last
        // `convergence_patience` generations. Requires at least one generation
        // beyond the window so a real comparison is possible.
        let window = self.convergence_patience;
        if self.best_score_history.len() <= window {
            return false;
        }

        let len = self.best_score_history.len();
        let recent_best = self.best_score_history[len - window..]
            .iter()
            .fold(T::neg_infinity(), |acc, &s| if s > acc { s } else { acc });

        // Best score achieved strictly before the recent window.
        let prior_best = self.best_score_history[..len - window]
            .iter()
            .fold(T::neg_infinity(), |acc, &s| if s > acc { s } else { acc });

        // No improvement (recent window did not exceed the prior best).
        recent_best <= prior_best
    }

    fn strategy_name(&self) -> &str {
        "EvolutionaryStrategy"
    }
}

// ---------------------------------------------------------------------------
// Real NSGA2Optimizer implementation (delegates to crate::multi_objective)
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + PartialOrd + std::iter::Sum>
    NSGA2Optimizer<T>
{
    /// Construct an NSGA-II optimizer from a multi-objective configuration. The
    /// inner [`crate::multi_objective::NSGA2`] is created with standard
    /// crossover/mutation probabilities; its population is (re)sized per update
    /// so that every individual carries real objective values.
    pub fn new(config: &MultiObjectiveConfig<T>) -> Result<Self> {
        use crate::multi_objective::MultiObjectiveOptimizer as InnerMultiObjective;

        // Population size is set per-update in `update_pareto_front`; seed with
        // a nominal value here. Install the objective configuration so the
        // inner instance can compute objective vectors (used by
        // `calculate_diversity`).
        let mut inner = crate::multi_objective::NSGA2::<T>::new(1, 0.9, 0.1);
        inner.initialize(config)?;
        Ok(Self {
            inner,
            config: config.clone(),
        })
    }
}

impl<T: Float + Debug + Default + Clone + Send + Sync + 'static + PartialOrd + std::iter::Sum>
    MultiObjectiveOptimizer<T> for NSGA2Optimizer<T>
{
    fn update_pareto_front(
        &mut self,
        results: &[SearchResult<T>],
    ) -> Result<multi_objective::ParetoFront<T>> {
        use crate::multi_objective::MultiObjectiveOptimizer as InnerMultiObjective;

        if results.is_empty() {
            return Ok(self.inner.get_pareto_front().clone());
        }

        // Build a fresh NSGA-II instance carrying the objective configuration
        // and delegate to the complete non-dominated-sort pipeline so the
        // returned front contains exactly the non-dominated solutions.
        let mut nsga2 = crate::multi_objective::NSGA2::<T>::new(results.len(), 0.9, 0.1);
        nsga2.initialize(&self.config)?;
        let front = nsga2.pareto_front_from_results(results);
        self.inner = nsga2;
        Ok(front)
    }

    fn select_candidates(
        &self,
        candidates: &[SearchResult<T>],
        population_size: usize,
    ) -> Result<Vec<SearchResult<T>>> {
        use crate::multi_objective::MultiObjectiveOptimizer as InnerMultiObjective;

        if candidates.is_empty() || population_size == 0 {
            return Ok(Vec::new());
        }

        // Delegate ranking to the real NSGA-II non-dominated sort + crowding
        // distance via a dedicated instance so `&self` remains immutable.
        let mut ranker = crate::multi_objective::NSGA2::<T>::new(candidates.len(), 0.9, 0.1);
        ranker.initialize(&self.config)?;
        let selected_indices = ranker.select_by_rank_and_crowding(candidates, population_size);

        Ok(selected_indices
            .into_iter()
            .map(|idx| candidates[idx].clone())
            .collect())
    }

    fn calculate_diversity(&self, population: &[SearchResult<T>]) -> f64 {
        // Mean pairwise distance in objective space, computed by the NSGA-II
        // helper using the same objective mapping as the optimization path.
        self.inner.mean_objective_distance(population)
    }
}

// Implementation for DefaultArchitectureController
impl<T: Float + Debug + Send + Sync + 'static> DefaultArchitectureController<T> {
    pub fn new(_config: &NASConfig<T>) -> Result<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: Float + Debug + Send + Sync + 'static> ArchitectureController<T>
    for DefaultArchitectureController<T>
{
    fn generate_random(&mut self) -> Result<OptimizerArchitecture<T>> {
        Ok(OptimizerArchitecture {
            components: Vec::new(),
            parameters: HashMap::new(),
            connections: Vec::new(),
            hyperparameters: HashMap::new(),
            architecture_id: "random_arch".to_string(),
            metadata: HashMap::new(),
        })
    }

    fn mutate(
        &mut self,
        architecture: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        Ok(architecture.clone())
    }

    fn crossover(
        &mut self,
        parent1: &OptimizerArchitecture<T>,
        _parent2: &OptimizerArchitecture<T>,
    ) -> Result<OptimizerArchitecture<T>> {
        Ok(parent1.clone())
    }

    fn validate(&self, _architecture: &OptimizerArchitecture<T>) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas_engine::config::{
        MultiObjectiveAlgorithm, ObjectiveConfig, ObjectivePriority, ObjectiveType,
        OptimizationDirection,
    };
    use crate::nas_engine::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };
    use crate::EvaluationMetric;

    /// Build a minimal valid architecture tagged with `id`.
    fn make_architecture(id: &str) -> OptimizerArchitecture<f64> {
        OptimizerArchitecture {
            components: vec!["Adam".to_string()],
            parameters: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: id.to_string(),
        }
    }

    /// Build a `SearchResult` whose first objective maps to
    /// [`EvaluationMetric::Accuracy`] and second to
    /// [`EvaluationMetric::MemoryUsage`], with the given `overall_score`.
    fn make_search_result(id: &str, accuracy: f64, memory: f64, overall: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::Accuracy, accuracy);
        metric_scores.insert(EvaluationMetric::MemoryUsage, memory);
        metric_scores.insert(EvaluationMetric::FinalPerformance, overall);

        let evaluation_results = EvaluationResults {
            metric_scores,
            overall_score: overall,
            confidence_intervals: HashMap::new(),
            evaluation_time: Duration::from_secs(0),
            success: true,
            error_message: None,
            cv_results: None,
            benchmark_results: HashMap::new(),
            training_trajectory: Vec::new(),
        };

        SearchResult {
            architecture: make_architecture(id),
            evaluation_results,
            generation: 0,
            search_time: 0.0,
            resource_usage: ResourceUsage::default(),
            encoding: ArchitectureEncoding::default(),
            metadata: SearchResultMetadata::default(),
        }
    }

    /// Two minimization objectives mapped onto distinct evaluation metrics.
    fn two_minimize_objectives() -> Vec<ObjectiveConfig<f64>> {
        vec![
            ObjectiveConfig {
                name: "accuracy".to_string(),
                objective_type: ObjectiveType::Accuracy,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
            ObjectiveConfig {
                name: "memory".to_string(),
                objective_type: ObjectiveType::MemoryUsage,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
        ]
    }

    #[test]
    fn test_random_strategy_generates_non_empty_valid_candidates() {
        let config = crate::nas_engine::create_minimal_nas_config::<f64>();
        let mut strategy = RandomStrategy::<f64>::new(&config).expect("construct RandomStrategy");

        let history: VecDeque<SearchResult<f64>> = VecDeque::new();
        let candidates = strategy
            .generate_candidates(&history)
            .expect("generate candidates");

        // A full population worth of candidates must be produced.
        assert_eq!(candidates.len(), config.population_size);
        assert!(!candidates.is_empty());

        // Each candidate must be a valid architecture: non-empty components and
        // a non-empty identifier.
        for candidate in &candidates {
            assert!(
                !candidate.components.is_empty(),
                "candidate must have at least one component"
            );
            assert!(
                !candidate.architecture_id.is_empty(),
                "candidate must have an identifier"
            );
        }
    }

    #[test]
    fn test_random_strategy_handles_empty_component_configs() {
        // Default search space declares `component_types` but leaves the
        // per-component `components` list empty; the strategy must still
        // synthesize valid candidates without panicking.
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.search_space.components.clear();
        config.population_size = 5;

        let mut strategy = RandomStrategy::<f64>::new(&config).expect("construct RandomStrategy");
        let candidates = strategy
            .generate_candidates(&VecDeque::new())
            .expect("generate candidates");

        assert_eq!(candidates.len(), 5);
        for candidate in &candidates {
            assert!(!candidate.components.is_empty());
        }
    }

    #[test]
    fn test_evolutionary_strategy_with_history_generates_candidates() {
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.population_size = 6;

        let mut strategy =
            EvolutionaryStrategy::<f64>::new(&config).expect("construct EvolutionaryStrategy");

        // Build a non-empty history with enough evaluated results for the
        // evolutionary step (population_size results).
        let mut history: VecDeque<SearchResult<f64>> = VecDeque::new();
        for i in 0..config.population_size {
            history.push_back(make_search_result(
                &format!("hist_{}", i),
                0.1 * i as f64,
                0.2 * i as f64,
                0.5 + 0.05 * i as f64,
            ));
        }

        let candidates = strategy
            .generate_candidates(&history)
            .expect("generate candidates");

        assert_eq!(candidates.len(), config.population_size);
        assert!(!candidates.is_empty());
        for candidate in &candidates {
            assert!(!candidate.components.is_empty());
        }
    }

    #[test]
    fn test_evolutionary_strategy_empty_history_falls_back_to_random() {
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.population_size = 4;

        let mut strategy =
            EvolutionaryStrategy::<f64>::new(&config).expect("construct EvolutionaryStrategy");

        // Empty history: the inner strategy returns members of its randomly
        // seeded population, so candidates are still produced.
        let candidates = strategy
            .generate_candidates(&VecDeque::new())
            .expect("generate candidates");

        assert_eq!(candidates.len(), 4);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_evolutionary_strategy_convergence_criterion() {
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.population_size = 4;
        config.early_stopping.patience = 3;

        let mut strategy =
            EvolutionaryStrategy::<f64>::new(&config).expect("construct EvolutionaryStrategy");

        // Fresh strategy has no history -> not converged.
        assert!(!strategy.has_converged());

        // Feed a stagnating best score (no improvement) over more than the
        // patience window -> converged.
        for _ in 0..(config.early_stopping.patience.max(5) + 2) {
            let results = vec![make_search_result("stag", 1.0, 1.0, 0.5)];
            strategy.update_strategy(&results).expect("update strategy");
        }
        assert!(strategy.has_converged());
    }

    #[test]
    fn test_nsga2_optimizer_update_pareto_front_keeps_non_dominated() {
        let config = MultiObjectiveConfig::<f64> {
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            objectives: two_minimize_objectives(),
            ..Default::default()
        };

        let mut optimizer = NSGA2Optimizer::<f64>::new(&config).expect("construct NSGA2Optimizer");

        // A = (1,2), B = (2,1) are mutually non-dominated.
        // C = (3,3) is dominated by both A and B (minimization).
        let results = vec![
            make_search_result("A", 1.0, 2.0, 0.9),
            make_search_result("B", 2.0, 1.0, 0.8),
            make_search_result("C", 3.0, 3.0, 0.1),
        ];

        let front = optimizer
            .update_pareto_front(&results)
            .expect("update pareto front");

        // Exactly the two non-dominated solutions survive.
        assert_eq!(front.solutions.len(), 2);
        assert_eq!(front.metrics.num_solutions, 2);

        let ids: std::collections::HashSet<String> = front
            .solutions
            .iter()
            .map(|s| s.architecture.architecture_id.clone())
            .collect();
        assert!(ids.contains("A"));
        assert!(ids.contains("B"));
        assert!(!ids.contains("C"));
    }

    #[test]
    fn test_nsga2_optimizer_select_candidates_prefers_non_dominated() {
        let config = MultiObjectiveConfig::<f64> {
            objectives: two_minimize_objectives(),
            ..Default::default()
        };

        let optimizer = NSGA2Optimizer::<f64>::new(&config).expect("construct NSGA2Optimizer");

        let results = vec![
            make_search_result("A", 1.0, 2.0, 0.9), // rank 0
            make_search_result("B", 2.0, 1.0, 0.8), // rank 0
            make_search_result("C", 3.0, 3.0, 0.1), // dominated
        ];

        // Selecting the best 2 must return the two non-dominated solutions.
        let selected = optimizer
            .select_candidates(&results, 2)
            .expect("select candidates");
        assert_eq!(selected.len(), 2);
        let ids: std::collections::HashSet<String> = selected
            .iter()
            .map(|s| s.architecture.architecture_id.clone())
            .collect();
        assert!(ids.contains("A"));
        assert!(ids.contains("B"));
    }

    #[test]
    fn test_nsga2_optimizer_diversity_is_non_constant() {
        let config = MultiObjectiveConfig::<f64> {
            objectives: two_minimize_objectives(),
            ..Default::default()
        };

        let optimizer = NSGA2Optimizer::<f64>::new(&config).expect("construct NSGA2Optimizer");

        // Identical objectives -> zero diversity.
        let identical = vec![
            make_search_result("A", 1.0, 1.0, 0.5),
            make_search_result("B", 1.0, 1.0, 0.5),
        ];
        let div_identical = optimizer.calculate_diversity(&identical);
        assert!(div_identical.abs() < 1e-12);

        // Spread-out objectives -> strictly positive diversity (not the old
        // hard-coded 0.5 constant).
        let spread = vec![
            make_search_result("A", 0.0, 0.0, 0.5),
            make_search_result("B", 3.0, 4.0, 0.5),
        ];
        let div_spread = optimizer.calculate_diversity(&spread);
        // Euclidean distance between (0,0) and (3,4) is exactly 5.
        assert!((div_spread - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_nas_engine_generates_non_empty_candidates_end_to_end() {
        // A full NASEngine built from a minimal config must produce a
        // non-empty, valid candidate batch through its search strategy.
        let config = crate::nas_engine::create_minimal_nas_config::<f64>();
        let mut engine = NeuralArchitectureSearch::<f64>::new(config).expect("construct NASEngine");

        let candidates = engine.generate_candidates().expect("generate candidates");
        assert!(
            !candidates.is_empty(),
            "engine must produce at least one candidate architecture"
        );
        for candidate in &candidates {
            assert!(!candidate.components.is_empty());
        }
    }
}
