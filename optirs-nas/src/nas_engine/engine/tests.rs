//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::nas_engine::config::*;
use crate::nas_engine::resources::*;
use crate::nas_engine::results::*;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use super::*;
// Test-only visibility into the engine's private strategy/optimizer adapters
// (RandomStrategy, EvolutionaryStrategy, NSGA2Optimizer, ...). These are
// `pub(super)` in their defining submodules (visible anywhere under `engine`);
// `engine::mod` does not re-export them (they were never part of the module's
// public API), so the tests import them directly here instead.
// (`super::controller` is not imported: nothing in this module names a type from
// it, and a glob that resolves to nothing is a warning.)
use super::mo_optimizers::*;
use super::strategies::*;

#[cfg(test)]
mod tests_2 {
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

    /// An architecture with `count` components, so a complexity filter has
    /// something to discriminate on.
    fn architecture_with_components(id: &str, count: usize) -> OptimizerArchitecture<f64> {
        OptimizerArchitecture {
            components: (0..count).map(|_| "Adam".to_string()).collect(),
            parameters: HashMap::new(),
            connections: (1..count).map(|i| (i - 1, i)).collect(),
            metadata: HashMap::new(),
            hyperparameters: HashMap::new(),
            architecture_id: id.to_string(),
        }
    }

    #[test]
    fn progressive_search_actually_stages_the_complexity_it_allows() {
        // `NASConfig::progressive_search` used to change nothing: the engine built a
        // `ProgressiveNAS` with an empty stage list whose `filter_candidates`
        // returned its input verbatim, while the engine reported
        // `progressive_search` among the run's key hyperparameters.
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.progressive_search = true;
        config.search_space.min_components = 1;
        config.search_space.max_components = 3;
        config.search_budget = 30;

        let mut progressive =
            ProgressiveNAS::<f64>::new(&config).expect("build the progressive filter");
        assert_eq!(
            progressive.stages().len(),
            3,
            "one stage per allowed component count"
        );
        // Each stage narrows the search space it describes.
        let limits: Vec<usize> = progressive
            .stages()
            .iter()
            .map(|stage| stage.search_space.max_components)
            .collect();
        assert_eq!(limits, vec![1, 2, 3]);
        assert!(progressive.stages().iter().all(|stage| stage
            .stage_config
            .search_space
            .max_components
            == stage.search_space.max_components));

        let candidates = vec![
            architecture_with_components("one", 1),
            architecture_with_components("two", 2),
            architecture_with_components("three", 3),
        ];

        // Generation 0 is in the first stage: only the single-component candidate
        // may be evaluated. The old implementation returned all three.
        let allowed = progressive
            .filter_candidates(candidates.clone(), 0)
            .expect("filter");
        let ids: Vec<&str> = allowed
            .iter()
            .map(|architecture| architecture.architecture_id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["one"],
            "stage 0 must defer the complex candidates"
        );
        assert_eq!(progressive.current_stage(), 0);

        // Late in the run the schedule has widened to the full complexity.
        let allowed = progressive
            .filter_candidates(candidates.clone(), 25)
            .expect("filter");
        assert_eq!(allowed.len(), 3, "the last stage allows everything");
        assert_eq!(progressive.current_stage(), 2);

        // A generation past the budget stays in the last stage rather than
        // indexing out of the schedule.
        assert_eq!(progressive.stage_for_generation(10_000), 2);
        assert_eq!(progressive.complexity_limit(10_000), Some(3));

        // Nothing fitting the stage must not produce an empty generation: the
        // simplest candidates are evaluated and the situation is reported.
        let too_complex = vec![
            architecture_with_components("big", 3),
            architecture_with_components("bigger", 4),
        ];
        let allowed = progressive
            .filter_candidates(too_complex, 0)
            .expect("filter");
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0].architecture_id, "big");
    }

    #[test]
    fn progressive_stage_history_records_what_each_stage_produced() {
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.search_space.min_components = 1;
        config.search_space.max_components = 2;
        config.search_budget = 10;
        let mut progressive =
            ProgressiveNAS::<f64>::new(&config).expect("build the progressive filter");

        // `stage_history` was declared and never written.
        progressive.record_stage_results(0, &[make_search_result("early", 1.0, 1.0, 0.2)]);
        progressive.record_stage_results(9, &[make_search_result("late", 1.0, 1.0, 0.9)]);

        assert_eq!(progressive.stage_results(0).len(), 1);
        assert_eq!(
            progressive.stage_results(0)[0].architecture.architecture_id,
            "early"
        );
        assert_eq!(progressive.stage_results(1).len(), 1);
        assert_eq!(
            progressive.stage_results(1)[0].architecture.architecture_id,
            "late"
        );
        // An empty batch records nothing.
        progressive.record_stage_results(0, &[]);
        assert_eq!(progressive.stage_results(0).len(), 1);
    }

    #[test]
    fn evolutionary_convergence_honors_min_generations() {
        // `EarlyStoppingConfig::min_generations` was declared, set by both config
        // builders, and enforced nowhere. Now that `has_converged` actually stops the
        // engine, a short patience must not end a run before the caller's declared
        // minimum: `patience: 3, min_generations: 50` used to stop at generation ~4.
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.population_size = 4;
        config.early_stopping.patience = 3;
        config.early_stopping.min_generations = 50;

        let mut strategy =
            EvolutionaryStrategy::<f64>::new(&config).expect("construct EvolutionaryStrategy");
        for _ in 0..20 {
            let results = vec![make_search_result("stag", 1.0, 1.0, 0.5)];
            strategy.update_strategy(&results).expect("update strategy");
        }
        assert!(
            !strategy.has_converged(),
            "a stagnating strategy must not stop the search before min_generations"
        );

        // Past the floor, the patience window governs again.
        for _ in 0..35 {
            let results = vec![make_search_result("stag", 1.0, 1.0, 0.5)];
            strategy.update_strategy(&results).expect("update strategy");
        }
        assert!(
            strategy.has_converged(),
            "once min_generations is met, a stagnating strategy must report convergence"
        );
    }

    #[test]
    fn engine_early_stopping_honors_min_generations() {
        // The same floor on the engine's own lack-of-improvement path.
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.early_stopping.enabled = true;
        config.early_stopping.patience = 2;
        config.early_stopping.min_improvement = 10.0;
        config.early_stopping.min_generations = 25;

        let mut engine = NeuralArchitectureSearch::<f64>::new(config).expect("construct engine");
        // A stagnating history that would satisfy patience on its own.
        for _ in 0..8 {
            engine
                .search_history
                .push_back(make_search_result("stag", 1.0, 1.0, 0.5));
        }
        engine.current_generation = 5;
        assert!(
            !engine.check_early_stopping_criteria(),
            "generation 5 is below the configured minimum of 25"
        );

        engine.current_generation = 30;
        assert!(
            engine.check_early_stopping_criteria(),
            "past the minimum, the stagnating history must trigger early stopping"
        );
    }

    #[test]
    fn evolutionary_convergence_honors_the_early_stopping_switch() {
        // `has_converged` is now consulted by `should_stop_search`, so this
        // no-improvement heuristic must respect a caller who turned early stopping
        // off — otherwise disabling early stopping would stop the search anyway.
        let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
        config.population_size = 4;
        config.early_stopping.patience = 3;
        config.early_stopping.enabled = false;

        let mut strategy =
            EvolutionaryStrategy::<f64>::new(&config).expect("construct EvolutionaryStrategy");
        for _ in 0..10 {
            let results = vec![make_search_result("stag", 1.0, 1.0, 0.5)];
            strategy.update_strategy(&results).expect("update strategy");
        }
        assert!(
            !strategy.has_converged(),
            "a stagnating strategy must not report convergence when early stopping \
             is disabled"
        );
    }

    #[test]
    fn a_finished_progressive_strategy_stops_the_engine_loop() {
        // `SearchStrategy::has_converged` was implemented by every adapter and
        // called from nowhere. A progressive search whose complexity schedule was
        // exhausted therefore kept sampling at its final complexity level until the
        // generation budget ran out; the strategy's own "I am done" was ignored.
        use crate::search_strategies::SearchStrategy as InnerSearchStrategy;

        let config = crate::nas_engine::create_minimal_nas_config::<f64>();
        let strategy = crate::nas_engine::strategy_adapters::progressive(&config)
            .expect("build the progressive strategy");
        assert!(
            !strategy.has_converged(),
            "a fresh progressive search has not finished"
        );

        // Drive the inner strategy through its whole schedule directly, so the test
        // does not depend on the engine's evaluator.
        let mut inner = crate::search_strategies::ProgressiveNAS::<f64>::with_seed(2, 1, 1, 12345);
        inner
            .initialize(&config.search_space)
            .expect("initialize the inner strategy");
        let history = std::collections::VecDeque::new();
        for _ in 0..2 {
            let architecture = inner
                .generate_architecture(&config.search_space, &history)
                .expect("generate");
            let mut result = make_search_result("phase", 1.0, 1.0, 0.5);
            result.architecture = architecture;
            inner.update_with_results(&[result]).expect("update");
        }
        assert!(
            inner.is_search_complete(),
            "the schedule must be exhausted after both phases have met their budget"
        );

        // And the engine reads exactly that signal through the adapter.
        let mut finished = crate::nas_engine::strategy_adapters::progressive(&config)
            .expect("build the progressive strategy");
        let mut generated = Vec::new();
        for _ in 0..64 {
            if finished.has_converged() {
                break;
            }
            let candidates = finished
                .generate_candidates(&history)
                .expect("generate candidates");
            let results: Vec<_> = candidates
                .into_iter()
                .map(|architecture| {
                    let mut result = make_search_result("cand", 1.0, 1.0, 0.5);
                    result.architecture = architecture;
                    result
                })
                .collect();
            generated.extend(results.iter().cloned());
            finished.update_strategy(&results).expect("update strategy");
        }
        assert!(
            finished.has_converged(),
            "the progressive adapter never reported completion after {} evaluations",
            generated.len()
        );
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

    /// F10 regression: five of the eight strategy types used to be silently
    /// swapped for `Evolutionary` (or `Random`) with an `eprintln!` warning even
    /// though real implementations existed. The engine must now build and run
    /// the requested strategy.
    #[test]
    fn test_engine_selects_the_requested_search_strategy() {
        let cases = [
            (SearchStrategyType::Random, "RandomStrategy"),
            (SearchStrategyType::Evolutionary, "EvolutionaryStrategy"),
            (
                SearchStrategyType::MultiObjectiveEvolutionary,
                "EvolutionaryStrategy",
            ),
            (
                SearchStrategyType::BayesianOptimization,
                "BayesianOptimization",
            ),
            (
                SearchStrategyType::ReinforcementLearning,
                "ReinforcementLearning",
            ),
            (SearchStrategyType::Differentiable, "Differentiable"),
            (SearchStrategyType::Progressive, "Progressive"),
            (
                SearchStrategyType::NeuralPredictorBased,
                "NeuralPredictorBased",
            ),
        ];

        for (strategy_type, expected_name) in cases {
            let mut config = crate::nas_engine::create_minimal_nas_config::<f64>();
            config.search_strategy = strategy_type;

            let mut engine = NeuralArchitectureSearch::<f64>::new(config)
                .unwrap_or_else(|e| panic!("{expected_name} engine must build: {e}"));
            assert_eq!(
                engine.search_strategy_name(),
                expected_name,
                "engine must not substitute a different strategy"
            );

            let candidates = engine
                .generate_candidates()
                .unwrap_or_else(|e| panic!("{expected_name} engine must generate: {e}"));
            assert!(!candidates.is_empty());
        }
    }
    // ---- F12: the resource monitor is actually driven by the search loop ----

    /// `ResourceMonitor::update_usage` was never called from anywhere, so
    /// `current_usage` stayed at `ResourceUsage::default()` for a whole run and
    /// `resource_usage_summary` in the results was always empty. The search loop now
    /// samples once per generation.
    #[test]
    fn the_search_loop_samples_the_resource_monitor() {
        use crate::nas_engine::create_minimal_nas_config;
        use crate::nas_engine::telemetry::{FixedTelemetry, TelemetrySample};

        let mut config = create_minimal_nas_config::<f64>();
        config.search_budget = 2;
        config.population_size = 2;

        let mut engine = NeuralArchitectureSearch::new(config)
            .expect("engine must build from the minimal config");

        // Inject a source with known, measured values so the assertion is exact.
        let sample = TelemetrySample {
            process_memory_gb: Some(1.75),
            total_memory_gb: Some(8.0),
            available_memory_gb: Some(6.0),
            logical_cpus: Some(4),
            process_count: Some(99),
            ..TelemetrySample::unknown()
        };
        engine.resource_monitor.set_trackers(vec![Box::new(
            SystemResourceTracker::with_telemetry(
                "injected".to_string(),
                Duration::from_secs(1),
                Box::new(FixedTelemetry::new("injected", sample)),
            ),
        )]);

        let results = engine
            .run_search()
            .expect("a two-generation search must complete");

        // The monitor must have been sampled, and with the injected values.
        assert!(
            !engine.resource_monitor.get_usage_history().is_empty(),
            "update_usage must be called from the search loop"
        );
        assert_eq!(engine.resource_monitor.get_current_usage().memory_gb, 1.75);
        assert_eq!(
            engine.resource_monitor.get_usage_history()[0].active_processes,
            Some(99)
        );
        assert_eq!(
            results.resource_usage_summary.total_memory_gb, 1.75,
            "the final summary must reflect the sampled usage, not a default"
        );
        assert!(!results.search_history.is_empty());
    }

    /// The other half of F12: a search must not be aborted by a resource the
    /// telemetry cannot measure, however tight the configured budget is.
    #[test]
    fn a_tight_budget_on_an_unmeasurable_resource_does_not_abort_the_search() {
        use crate::nas_engine::create_minimal_nas_config;
        use crate::nas_engine::telemetry::{FixedTelemetry, TelemetrySample};

        let mut config = create_minimal_nas_config::<f64>();
        config.search_budget = 2;
        config.population_size = 2;
        // Budgets that the old fabricated readings (16 GB used, 250 W, 65 C) would
        // have blown through immediately.
        config.resource_constraints.hardware_resources.max_memory_gb = 0.001;
        config.resource_constraints.max_memory_gb = 0.001;
        config.resource_constraints.max_energy_kwh = 0.0;
        config.resource_constraints.max_cost_usd = 0.0;

        let mut engine = NeuralArchitectureSearch::new(config).expect("engine builds");
        engine.resource_monitor.set_trackers(vec![Box::new(
            SystemResourceTracker::with_telemetry(
                "blind".to_string(),
                Duration::from_secs(1),
                Box::new(FixedTelemetry::new("blind", TelemetrySample::unknown())),
            ),
        )]);

        let results = engine
            .run_search()
            .expect("an unmeasurable resource must never abort the search");
        assert!(!results.search_history.is_empty());
    }

    /// A *measured* overrun must still stop the search, so the guard above has not
    /// disabled enforcement.
    #[test]
    fn a_measured_overrun_still_stops_the_search() {
        use crate::nas_engine::create_minimal_nas_config;
        use crate::nas_engine::telemetry::{FixedTelemetry, TelemetrySample};

        let mut config = create_minimal_nas_config::<f64>();
        config.search_budget = 5;
        config.population_size = 2;
        config.resource_constraints.max_memory_gb = 1.0;
        config.resource_constraints.hardware_resources.max_memory_gb = 1.0;

        let mut engine = NeuralArchitectureSearch::new(config).expect("engine builds");
        engine.resource_monitor.set_trackers(vec![Box::new(
            SystemResourceTracker::with_telemetry(
                "hog".to_string(),
                Duration::from_secs(1),
                Box::new(FixedTelemetry::new(
                    "hog",
                    TelemetrySample {
                        process_memory_gb: Some(64.0),
                        ..TelemetrySample::unknown()
                    },
                )),
            ),
        )]);

        let error = engine
            .run_search()
            .expect_err("a measured 64 GB against a 1 GB budget must stop the search");
        assert!(
            format!("{error}").contains("resource constraints violated"),
            "unexpected error: {error}"
        );
    }

    // ---- the multi-objective algorithm routing ----------------------------

    #[test]
    fn weighted_sum_is_served_by_the_real_weighted_sum_optimizer() {
        let config = MultiObjectiveConfig::<f64> {
            algorithm: MultiObjectiveAlgorithm::WeightedSum,
            objectives: two_minimize_objectives(),
            ..Default::default()
        };
        let mut optimizer = WeightedSumOptimizer::<f64>::new(&config)
            .expect("the weighted-sum adapter must construct");

        let front = optimizer
            .update_pareto_front(&[
                make_search_result("A", 1.0, 2.0, 0.9),
                make_search_result("B", 2.0, 1.0, 0.8),
                make_search_result("C", 3.0, 3.0, 0.1),
            ])
            .expect("update pareto front");
        assert_eq!(front.solutions.len(), 2, "C is dominated by both A and B");

        let selected = optimizer
            .select_candidates(
                &[
                    make_search_result("cheap", 1.0, 1.0, 0.9),
                    make_search_result("costly", 5.0, 5.0, 0.1),
                ],
                1,
            )
            .expect("select candidates");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].architecture.architecture_id, "cheap");

        // Diversity must be measured, not the hardcoded 0.5 the deleted placeholder
        // optimizers returned.
        let diversity = optimizer.calculate_diversity(&[
            make_search_result("a", 0.0, 0.0, 0.0),
            make_search_result("b", 3.0, 4.0, 0.0),
        ]);
        assert!((diversity - 5.0).abs() < 1e-12, "got {diversity}");
        assert_ne!(diversity, 0.5);
    }

    #[test]
    fn nsga3_and_moead_are_served_by_their_real_implementations() {
        // Both used to be routed to a macro-generated placeholder that returned an
        // empty Pareto front and a hardcoded diversity of 0.5, and were then made to
        // report `NotImplemented`. They are now real algorithms.
        for algorithm in [
            MultiObjectiveAlgorithm::NSGA3,
            MultiObjectiveAlgorithm::MOEAD,
        ] {
            let config = MultiObjectiveConfig::<f64> {
                algorithm: algorithm.clone(),
                objectives: two_minimize_objectives(),
                ..Default::default()
            };
            let mut optimizer =
                NeuralArchitectureSearch::<f64>::create_multi_objective_optimizer(&config)
                    .unwrap_or_else(|error| panic!("{algorithm:?} must construct: {error}"));

            let front = optimizer
                .update_pareto_front(&[
                    make_search_result("A", 1.0, 2.0, 0.9),
                    make_search_result("B", 2.0, 1.0, 0.8),
                    make_search_result("C", 3.0, 3.0, 0.1),
                ])
                .unwrap_or_else(|error| panic!("{algorithm:?} update: {error}"));
            assert_eq!(
                front.solutions.len(),
                2,
                "{algorithm:?} must report a real front (C is dominated by both A and B)"
            );
            assert!(
                front.metrics.hypervolume > 0.0,
                "{algorithm:?} reported a zero hypervolume"
            );

            let selected = optimizer
                .select_candidates(
                    &[
                        make_search_result("cheap", 1.0, 1.0, 0.9),
                        make_search_result("costly", 5.0, 5.0, 0.1),
                    ],
                    1,
                )
                .unwrap_or_else(|error| panic!("{algorithm:?} select: {error}"));
            assert_eq!(selected.len(), 1, "{algorithm:?}");
            assert_eq!(
                selected[0].architecture.architecture_id, "cheap",
                "{algorithm:?} ranked the dominated candidate first"
            );

            // Measured diversity, not the placeholder's 0.5.
            let diversity = optimizer.calculate_diversity(&[
                make_search_result("a", 0.0, 0.0, 0.0),
                make_search_result("b", 3.0, 4.0, 0.0),
            ]);
            assert!(
                (diversity - 5.0).abs() < 1e-12,
                "{algorithm:?} diversity = {diversity}"
            );
        }
    }

    #[test]
    fn unimplemented_multi_objective_algorithms_are_rejected() {
        for algorithm in [
            MultiObjectiveAlgorithm::PAES,
            MultiObjectiveAlgorithm::SPEA2,
            MultiObjectiveAlgorithm::EpsilonConstraint,
            MultiObjectiveAlgorithm::GoalProgramming,
        ] {
            let config = MultiObjectiveConfig::<f64> {
                algorithm: algorithm.clone(),
                objectives: two_minimize_objectives(),
                ..Default::default()
            };
            let outcome =
                NeuralArchitectureSearch::<f64>::create_multi_objective_optimizer(&config);
            let error = match outcome {
                Ok(_) => {
                    panic!("{algorithm:?} must not be silently substituted by another optimizer")
                }
                Err(error) => error,
            };
            let message = format!("{error}");
            assert!(
                message.contains("is not implemented"),
                "unexpected error for {algorithm:?}: {message}"
            );
            // The message must name what *is* available.
            assert!(message.contains("NSGA2"), "{message}");
        }
    }
}
