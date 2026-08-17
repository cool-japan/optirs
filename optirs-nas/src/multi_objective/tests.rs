//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::nas_engine::{
    MultiObjectiveConfig, ObjectiveConfig, ObjectivePriority, ObjectiveType, OptimizationDirection,
    OptimizerArchitecture, SearchResult,
};
use crate::EvaluationMetric;
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use crate::nas_engine::{
        ArchitectureEncoding, EvaluationResults, ResourceUsage, SearchResultMetadata,
    };

    #[test]
    fn test_nsga2_creation() {
        let nsga2 = NSGA2::<f64>::new(50, 0.8, 0.1);
        assert_eq!(nsga2.population_size, 50);
        assert_eq!(nsga2.name(), "NSGA-II");
    }

    #[test]
    fn test_pareto_front_creation() {
        let front = ParetoFront::<f64>::new();
        assert!(front.solutions.is_empty());
        assert_eq!(front.generation, 0);
    }

    #[test]
    fn test_dominance_relation() {
        let mut nsga2 = NSGA2::<f64>::new(2, 0.8, 0.1);

        // Create two test individuals
        let arch = nsga2.generate_random_architecture().expect("unwrap failed");

        let ind1 = Individual {
            architecture: arch.clone(),
            objectives: vec![1.0, 2.0], // Better in first objective
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            id: "ind1".to_string(),
        };

        let ind2 = Individual {
            architecture: arch,
            objectives: vec![2.0, 1.0], // Better in second objective
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            id: "ind2".to_string(),
        };

        nsga2.population = vec![ind1, ind2];
        nsga2.config.objectives = vec![
            ObjectiveConfig {
                name: "obj1".to_string(),
                objective_type: ObjectiveType::Performance,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
            ObjectiveConfig {
                name: "obj2".to_string(),
                objective_type: ObjectiveType::Efficiency,
                direction: OptimizationDirection::Minimize,
                weight: 0.5,
                priority: ObjectivePriority::High,
                normalization_bounds: None,
            },
        ];

        let relation = nsga2.dominance_relation(0, 1);
        assert_eq!(relation, DominanceRelation::NonDominated);
    }

    // ---- WeightedSum test helpers -------------------------------------

    /// Build a minimal [`OptimizerArchitecture`] tagged with `id`.
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

    /// Build a [`SearchResult`] whose first objective maps to
    /// [`EvaluationMetric::Accuracy`] (= `obj0`) and whose second maps to
    /// [`EvaluationMetric::MemoryUsage`] (= `obj1`).
    fn make_search_result(id: &str, obj0: f64, obj1: f64) -> SearchResult<f64> {
        let mut metric_scores = HashMap::new();
        metric_scores.insert(EvaluationMetric::Accuracy, obj0);
        metric_scores.insert(EvaluationMetric::MemoryUsage, obj1);

        let evaluation_results = EvaluationResults {
            metric_scores,
            overall_score: 0.0,
            confidence_intervals: HashMap::new(),
            evaluation_time: std::time::Duration::from_secs(0),
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

    /// Two minimization objectives backed by distinct metrics.
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

    // ---- WeightedSum tests --------------------------------------------

    #[test]
    fn test_weighted_sum_creation() {
        let ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");
        assert_eq!(ws.name(), "WeightedSum");
        assert_eq!(ws.weights.len(), 2);
        assert!(ws.get_pareto_front().solutions.is_empty());
    }

    #[test]
    fn test_weighted_sum_update_pareto_front_keeps_non_dominated() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        // A = (1,2) and B = (2,1) are mutually non-dominated.
        // C = (3,3) is dominated by both A and B.
        let results = vec![
            make_search_result("A", 1.0, 2.0),
            make_search_result("B", 2.0, 1.0),
            make_search_result("C", 3.0, 3.0),
        ];

        let front = ws
            .update_pareto_front(&results)
            .expect("update pareto front");

        // Exactly the two non-dominated solutions must remain.
        assert_eq!(front.solutions.len(), 2);
        assert_eq!(front.metrics.num_solutions, 2);

        let mut ids: Vec<String> = front
            .solutions
            .iter()
            .map(|s| s.metadata.id.clone())
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["A".to_string(), "B".to_string()]);

        // Bounds should bracket the surviving objective values.
        assert_eq!(front.objective_bounds.min_values, vec![1.0, 1.0]);
        assert_eq!(front.objective_bounds.max_values, vec![2.0, 2.0]);

        // The stored front mirrors the returned one.
        assert_eq!(ws.get_pareto_front().solutions.len(), 2);
    }

    #[test]
    fn test_weighted_sum_update_pareto_front_incremental() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        // Seed the front with a single solution.
        ws.update_pareto_front(&[make_search_result("A", 2.0, 2.0)])
            .expect("seed front");
        assert_eq!(ws.get_pareto_front().solutions.len(), 1);

        // A new, strictly-better solution must dominate and replace it.
        let front = ws
            .update_pareto_front(&[make_search_result("B", 1.0, 1.0)])
            .expect("update front");
        assert_eq!(front.solutions.len(), 1);
        assert_eq!(front.solutions[0].metadata.id, "B");

        // Feeding no new solutions returns the front unchanged.
        let unchanged = ws.update_pareto_front(&[]).expect("empty update");
        assert_eq!(unchanged.solutions.len(), 1);
        assert_eq!(unchanged.solutions[0].metadata.id, "B");
    }

    #[test]
    fn test_weighted_sum_select_candidates_ranking() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");

        let population = vec![
            make_architecture("p0"),
            make_architecture("p1"),
            make_architecture("p2"),
            make_architecture("p3"),
        ];

        // Row-major objective matrix: 4 architectures x 2 objectives.
        // scores (equal weights, minimize): p0=4, p1=1, p2=2.5, p3=3.
        let objectives = vec![
            4.0, 4.0, // p0 -> 4.0
            1.0, 1.0, // p1 -> 1.0 (best)
            2.0, 3.0, // p2 -> 2.5
            3.0, 3.0, // p3 -> 3.0
        ];

        let selected = ws
            .select_candidates(&population, &objectives)
            .expect("select candidates");

        // Better half of 4 -> top 2 by ascending scalarized cost.
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].architecture_id, "p1");
        assert_eq!(selected[1].architecture_id, "p2");

        // The recorded best solution must match the top-ranked architecture.
        let best = ws.best_solution().expect("best solution recorded");
        assert_eq!(best.architecture.architecture_id, "p1");
    }

    #[test]
    fn test_weighted_sum_select_candidates_honors_maximize() {
        // Single maximize objective: larger objective value is preferred,
        // hence selected first despite weighted-sum minimizing internally.
        let objectives_cfg = vec![ObjectiveConfig {
            name: "accuracy".to_string(),
            objective_type: ObjectiveType::Accuracy,
            direction: OptimizationDirection::Maximize,
            weight: 1.0,
            priority: ObjectivePriority::High,
            normalization_bounds: None,
        }];
        let mut ws = WeightedSum::new(&objectives_cfg).expect("construct WeightedSum");

        let population = vec![make_architecture("low"), make_architecture("high")];
        // One objective per architecture.
        let objectives = vec![0.2, 0.9];

        let selected = ws
            .select_candidates(&population, &objectives)
            .expect("select candidates");

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].architecture_id, "high");
    }

    #[test]
    fn test_weighted_sum_select_candidates_empty_population() {
        let mut ws = WeightedSum::new(&two_minimize_objectives()).expect("construct WeightedSum");
        let selected = ws.select_candidates(&[], &[]).expect("select candidates");
        assert!(selected.is_empty());
    }

    // ---- F18 / F17: the NSGA-II public path -----------------------------

    /// Two-objective minimization config for the NSGA-II public path.
    fn two_objective_config() -> MultiObjectiveConfig<f64> {
        MultiObjectiveConfig {
            objectives: two_minimize_objectives(),
            ..MultiObjectiveConfig::default()
        }
    }

    /// A [`SearchResult`] carrying `id` as its architecture id, so the by-id
    /// matching in `update_pareto_front` has something to match on.
    fn result_with_id(id: &str, obj0: f64, obj1: f64) -> SearchResult<f64> {
        make_search_result(id, obj0, obj1)
    }

    #[test]
    fn test_nsga2_initial_population_is_structurally_diverse() {
        let mut nsga2 = NSGA2::<f64>::with_seed(40, 0.9, 0.1, 0xF18_0001);
        nsga2.initialize(&two_objective_config()).expect("init");

        assert_eq!(nsga2.population.len(), 40);

        // The pre-fix initializer produced the same single-component "Adam"
        // architecture forty times over.
        let signatures: std::collections::HashSet<Vec<String>> = nsga2
            .population
            .iter()
            .map(|ind| ind.architecture.components.clone())
            .collect();
        assert!(
            signatures.len() > 5,
            "expected a diverse initial population, got {} distinct component signatures",
            signatures.len()
        );
        assert!(
            !signatures
                .iter()
                .all(|sig| sig == &vec!["Adam".to_string()]),
            "population collapsed to the old hardcoded Adam architecture"
        );

        // Hyperparameters must vary too, not repeat three fixed constants.
        let rates: std::collections::HashSet<u64> = nsga2
            .population
            .iter()
            .filter_map(|ind| ind.architecture.parameters.get("learning_rate"))
            .map(|v| v.to_bits())
            .collect();
        assert!(
            rates.len() > 5,
            "expected varied learning rates, got {}",
            rates.len()
        );

        // Nothing has been evaluated, so nothing may claim a front position.
        assert!(
            nsga2.population.iter().all(|ind| ind.objectives.is_empty()),
            "a freshly initialized individual must be marked unevaluated"
        );
    }

    #[test]
    fn test_nsga2_pareto_front_is_a_strict_subset_of_the_population() {
        let mut nsga2 = NSGA2::<f64>::new(4, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");

        // A and B are mutually non-dominated; C and D are dominated by both.
        let results = vec![
            result_with_id("A", 1.0, 2.0),
            result_with_id("B", 2.0, 1.0),
            result_with_id("C", 3.0, 3.0),
            result_with_id("D", 4.0, 5.0),
        ];
        let front = nsga2.update_pareto_front(&results).expect("update front");

        // Against the pre-fix code every objective vector was all-zero, so every
        // pair was non-dominated and the "front" was the whole population.
        assert_eq!(
            front.solutions.len(),
            2,
            "front must contain exactly the non-dominated solutions, got {:?}",
            front
                .solutions
                .iter()
                .map(|s| s.metadata.id.clone())
                .collect::<Vec<_>>()
        );
        assert!(front.solutions.len() < nsga2.population.len());

        let mut ids: Vec<String> = front
            .solutions
            .iter()
            .map(|s| s.metadata.id.clone())
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(front.metrics.num_solutions, 2);
    }

    #[test]
    fn test_nsga2_update_pareto_front_matches_results_by_architecture_id() {
        let mut nsga2 = NSGA2::<f64>::new(3, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");

        // Seed three named individuals.
        nsga2
            .update_pareto_front(&[
                result_with_id("alpha", 5.0, 5.0),
                result_with_id("beta", 6.0, 6.0),
                result_with_id("gamma", 7.0, 7.0),
            ])
            .expect("seed");

        // Re-report them in a *different* order with distinct values. Index-based
        // mapping would attribute gamma's objectives to alpha.
        nsga2
            .update_pareto_front(&[
                result_with_id("gamma", 1.0, 9.0),
                result_with_id("alpha", 9.0, 1.0),
            ])
            .expect("reorder");

        let alpha = nsga2
            .population
            .iter()
            .find(|ind| ind.architecture.architecture_id == "alpha")
            .expect("alpha present");
        let gamma = nsga2
            .population
            .iter()
            .find(|ind| ind.architecture.architecture_id == "gamma")
            .expect("gamma present");
        assert_eq!(
            alpha.objectives,
            vec![9.0, 1.0],
            "alpha got the wrong result"
        );
        assert_eq!(
            gamma.objectives,
            vec![1.0, 9.0],
            "gamma got the wrong result"
        );
        assert_eq!(
            nsga2.population.len(),
            3,
            "matching must not duplicate individuals"
        );
    }

    #[test]
    fn test_nsga2_hypervolume_is_the_exact_indicator() {
        let mut nsga2 = NSGA2::<f64>::new(2, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");
        nsga2
            .set_hypervolume_reference(vec![2.0, 2.0])
            .expect("pin reference");

        let front = nsga2
            .update_pareto_front(&[result_with_id("A", 1.0, 0.0), result_with_id("B", 0.0, 1.0)])
            .expect("update front");

        // Exact hypervolume of {(1,0),(0,1)} against (2,2) is 3.
        // The old heuristic returned bounding_box_product * count = 1 * 1 * 2 = 2.
        assert!(
            (front.metrics.hypervolume - 3.0).abs() < 1e-12,
            "expected the exact hypervolume 3.0, got {}",
            front.metrics.hypervolume
        );
        assert_eq!(nsga2.hypervolume_reference(), Some(&[2.0, 2.0][..]));
    }

    #[test]
    fn test_nsga2_hypervolume_handles_a_maximized_objective() {
        // obj0 = accuracy, MAXIMIZED; obj1 = memory, minimized.
        let objectives = vec![
            ObjectiveConfig {
                name: "accuracy".to_string(),
                objective_type: ObjectiveType::Accuracy,
                direction: OptimizationDirection::Maximize,
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
        ];
        let config = MultiObjectiveConfig {
            objectives,
            ..MultiObjectiveConfig::default()
        };

        let mut nsga2 = NSGA2::<f64>::new(2, 0.9, 0.1);
        nsga2.initialize(&config).expect("init");
        // Raw-space reference: accuracy 0.0 is the worst, memory 2.0 is the worst.
        nsga2
            .set_hypervolume_reference(vec![0.0, 2.0])
            .expect("pin reference");

        let front = nsga2
            .update_pareto_front(&[
                result_with_id("high_acc", 0.9, 1.0),
                result_with_id("low_mem", 0.6, 0.5),
            ])
            .expect("update front");

        assert_eq!(front.solutions.len(), 2, "both are non-dominated");
        // In minimization space the front is {(-0.9,1.0), (-0.6,0.5)} against
        // (0.0, 2.0): 0.9*1 + 0.6*1.5 - 0.6*1 = 1.2. Skipping the negation would
        // collapse the indicator to exactly 0.
        assert!(
            (front.metrics.hypervolume - 1.2).abs() < 1e-12,
            "expected 1.2 for the mixed-direction front, got {}",
            front.metrics.hypervolume
        );
        assert!(
            front.metrics.hypervolume > 0.0,
            "a maximized objective must not zero the hypervolume"
        );
    }

    #[test]
    fn test_nsga2_front_metrics_are_computed_not_hardcoded() {
        let mut nsga2 = NSGA2::<f64>::new(3, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");
        nsga2
            .set_hypervolume_reference(vec![2.0, 2.0])
            .expect("pin reference");

        let first = nsga2
            .update_pareto_front(&[
                result_with_id("A", 1.0, 0.5),
                result_with_id("B", 0.5, 1.0),
                result_with_id("C", 1.5, 1.5),
            ])
            .expect("first update");

        // coverage was hardcoded to exactly 0.5 and convergence to exactly 0.
        assert_ne!(
            first.metrics.coverage.objective_space_coverage, 0.5,
            "objective_space_coverage is still the hardcoded 0.5"
        );
        // Front bounding box 0.5x0.5 inside the (2,2) reference box measured from
        // the front ideal (0.5,0.5): 0.25 / (1.5 * 1.5) = 1/9.
        assert!(
            (first.metrics.coverage.objective_space_coverage - 1.0 / 9.0).abs() < 1e-12,
            "got {}",
            first.metrics.coverage.objective_space_coverage
        );
        assert!(
            first.metrics.coverage.reference_distance > 0.0,
            "reference_distance must be measured"
        );
        // C is dominated by both A and B, so the front covers the population.
        assert_eq!(first.metrics.coverage.epsilon_dominance, 0.0);
        assert_eq!(
            first.metrics.convergence, 1.0,
            "the first update cannot claim convergence"
        );

        // A second update that does not improve the front must report a
        // convergence of 0 (the indicator stopped moving).
        let second = nsga2
            .update_pareto_front(&[result_with_id("D", 1.8, 1.8)])
            .expect("second update");
        assert!(
            second.metrics.convergence.abs() < 1e-12,
            "an unchanged hypervolume must read as converged, got {}",
            second.metrics.convergence
        );
        assert!(
            !nsga2.get_statistics().convergence_history.is_empty(),
            "convergence history must record real values"
        );

        // A genuinely better solution must move both the hypervolume and the
        // convergence measure.
        let third = nsga2
            .update_pareto_front(&[result_with_id("E", 0.1, 0.1)])
            .expect("third update");
        assert!(third.metrics.hypervolume > second.metrics.hypervolume);
        assert!(third.metrics.convergence > 0.0);
    }

    #[test]
    fn test_nsga2_select_candidates_varies_the_component_sequence() {
        let mut nsga2 = NSGA2::<f64>::with_seed(24, 0.9, 0.9, 0xF18_0002);
        nsga2.initialize(&two_objective_config()).expect("init");
        nsga2
            .update_pareto_front(&[
                result_with_id("A", 1.0, 2.0),
                result_with_id("B", 2.0, 1.0),
                result_with_id("C", 3.0, 3.0),
            ])
            .expect("evaluate");

        let offspring = nsga2
            .select_candidates(&[], &[])
            .expect("select candidates");
        assert_eq!(offspring.len(), 24);

        // The pre-fix crossover/mutation never touched `components`, so every
        // offspring inherited the same single-element component vector.
        let signatures: std::collections::HashSet<Vec<String>> = offspring
            .iter()
            .map(|arch| arch.components.clone())
            .collect();
        assert!(
            signatures.len() > 1,
            "crossover/mutation must vary the component sequence, got {:?}",
            signatures
        );
        for arch in &offspring {
            assert!(!arch.components.is_empty());
            for (from, to) in &arch.connections {
                assert!(*from < arch.components.len() && *to < arch.components.len());
            }
        }
    }

    #[test]
    fn test_nsga2_unevaluated_individuals_never_enter_the_front() {
        let mut nsga2 = NSGA2::<f64>::new(8, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");

        // Only two of the eight individuals ever get objective values.
        let front = nsga2
            .update_pareto_front(&[result_with_id("A", 1.0, 2.0), result_with_id("B", 2.0, 1.0)])
            .expect("update front");
        assert_eq!(front.solutions.len(), 2);
        assert_eq!(nsga2.population.len(), 8);
        assert_eq!(
            nsga2
                .population
                .iter()
                .filter(|ind| ind.objectives.is_empty())
                .count(),
            6,
            "the remaining individuals must stay marked unevaluated"
        );
    }

    #[test]
    fn test_nsga2_rejects_a_mismatched_hypervolume_reference() {
        let mut nsga2 = NSGA2::<f64>::new(2, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");
        assert!(nsga2.set_hypervolume_reference(vec![1.0]).is_err());
        assert!(nsga2
            .set_hypervolume_reference(vec![1.0, 1.0, 1.0])
            .is_err());
    }

    #[test]
    fn test_nsga2_derives_and_latches_a_reference_point() {
        let mut nsga2 = NSGA2::<f64>::new(2, 0.9, 0.1);
        nsga2.initialize(&two_objective_config()).expect("init");
        assert!(nsga2.hypervolume_reference().is_none());

        let first = nsga2
            .update_pareto_front(&[result_with_id("A", 1.0, 0.0), result_with_id("B", 0.0, 1.0)])
            .expect("first update");
        let derived: Vec<f64> = nsga2
            .hypervolume_reference()
            .expect("a reference must be derived")
            .to_vec();
        assert!(first.metrics.hypervolume > 0.0);

        // Later updates must reuse the same reference, otherwise hypervolumes
        // from different generations are not comparable.
        nsga2
            .update_pareto_front(&[result_with_id("C", 0.2, 0.2)])
            .expect("second update");
        assert_eq!(
            nsga2.hypervolume_reference().map(|r| r.to_vec()),
            Some(derived)
        );
    }
    #[test]
    fn test_nsga2_instances_do_not_share_a_hardcoded_seed() {
        // `NSGA2::new` used to be `Random::seed(42)`, which — now that
        // initialization genuinely samples — would make every instance explore the
        // identical initial population.
        let mut first = NSGA2::<f64>::new(12, 0.9, 0.1);
        first.initialize(&two_objective_config()).expect("init");
        let mut second = NSGA2::<f64>::new(12, 0.9, 0.1);
        second.initialize(&two_objective_config()).expect("init");

        let signature = |nsga2: &NSGA2<f64>| -> Vec<Vec<String>> {
            nsga2
                .population
                .iter()
                .map(|ind| ind.architecture.components.clone())
                .collect()
        };
        assert_ne!(
            signature(&first),
            signature(&second),
            "two unseeded NSGA-II instances must not produce identical populations"
        );

        // An explicit seed must still be fully reproducible.
        let mut a = NSGA2::<f64>::with_seed(12, 0.9, 0.1, 4242);
        a.initialize(&two_objective_config()).expect("init");
        let mut b = NSGA2::<f64>::with_seed(12, 0.9, 0.1, 4242);
        b.initialize(&two_objective_config()).expect("init");
        assert_eq!(signature(&a), signature(&b));
    }

    #[test]
    fn test_moead_reports_that_it_is_not_implemented() {
        use crate::multi_objective::MOEADOptimizer;
        let mut moead = MOEADOptimizer::<f64>::new(two_objective_config())
            .expect("the state container still constructs");

        // Each of these used to return Ok with an empty/unchanged result, so a
        // search configured for MOEA/D ran to completion and reported nothing.
        let init = moead
            .initialize(&two_objective_config())
            .expect_err("initialize must report NotImplemented");
        assert!(format!("{init}").contains("MOEA/D is not implemented"));

        let update = moead
            .update_pareto_front(&[result_with_id("A", 1.0, 2.0)])
            .expect_err("update_pareto_front must report NotImplemented");
        assert!(format!("{update}").contains("MOEA/D is not implemented"));

        let select = moead
            .select_candidates(&[], &[])
            .expect_err("select_candidates must report NotImplemented");
        assert!(format!("{select}").contains("MOEA/D is not implemented"));
    }
}
