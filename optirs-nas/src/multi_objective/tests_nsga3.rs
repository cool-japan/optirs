//! Tests for the reference-direction based optimizer [`super::nsga3::NSGA3`].
//!
//! Kept in its own file (rather than appended to `tests.rs`) so neither module
//! approaches the 2000-line limit.

use crate::nas_engine::{
    ArchitectureEncoding, EvaluationResults, MultiObjectiveAlgorithm, MultiObjectiveConfig,
    ObjectiveConfig, ObjectivePriority, ObjectiveType, OptimizationDirection,
    OptimizerArchitecture, ResourceUsage, SearchResult, SearchResultMetadata,
};
use crate::EvaluationMetric;
use std::collections::{HashMap, HashSet};

use super::core::MultiObjectiveOptimizer;
use super::nsga3::NSGA3;

fn architecture(id: &str) -> OptimizerArchitecture<f64> {
    OptimizerArchitecture {
        components: vec!["Adam".to_string()],
        parameters: HashMap::new(),
        connections: Vec::new(),
        metadata: HashMap::new(),
        hyperparameters: HashMap::new(),
        architecture_id: id.to_string(),
    }
}

fn objective(name: &str, objective_type: ObjectiveType) -> ObjectiveConfig<f64> {
    ObjectiveConfig {
        name: name.to_string(),
        objective_type,
        direction: OptimizationDirection::Minimize,
        weight: 1.0,
        priority: ObjectivePriority::High,
        normalization_bounds: None,
    }
}

/// Two minimized objectives read from `Accuracy` and `MemoryUsage`.
fn two_objective_config() -> MultiObjectiveConfig<f64> {
    MultiObjectiveConfig {
        objectives: vec![
            objective("accuracy", ObjectiveType::Accuracy),
            objective("memory", ObjectiveType::MemoryUsage),
        ],
        algorithm: MultiObjectiveAlgorithm::NSGA3,
        ..MultiObjectiveConfig::default()
    }
}

/// Three minimized objectives — the case NSGA-III actually exists for.
fn three_objective_config() -> MultiObjectiveConfig<f64> {
    MultiObjectiveConfig {
        objectives: vec![
            objective("accuracy", ObjectiveType::Accuracy),
            objective("memory", ObjectiveType::MemoryUsage),
            objective("training_time", ObjectiveType::TrainingTime),
        ],
        algorithm: MultiObjectiveAlgorithm::NSGA3,
        ..MultiObjectiveConfig::default()
    }
}

fn result_from(
    arch: OptimizerArchitecture<f64>,
    values: &[(EvaluationMetric, f64)],
) -> SearchResult<f64> {
    let mut metric_scores = HashMap::new();
    for (metric, value) in values {
        metric_scores.insert(*metric, *value);
    }
    SearchResult {
        architecture: arch,
        evaluation_results: EvaluationResults {
            metric_scores,
            overall_score: 0.0,
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

fn result2(id: &str, obj0: f64, obj1: f64) -> SearchResult<f64> {
    result_from(
        architecture(id),
        &[
            (EvaluationMetric::Accuracy, obj0),
            (EvaluationMetric::MemoryUsage, obj1),
        ],
    )
}

fn result3(id: &str, obj0: f64, obj1: f64, obj2: f64) -> SearchResult<f64> {
    result_from(
        architecture(id),
        &[
            (EvaluationMetric::Accuracy, obj0),
            (EvaluationMetric::MemoryUsage, obj1),
            (EvaluationMetric::TrainingTime, obj2),
        ],
    )
}

fn result_for(arch: OptimizerArchitecture<f64>, obj0: f64, obj1: f64) -> SearchResult<f64> {
    result_from(
        arch,
        &[
            (EvaluationMetric::Accuracy, obj0),
            (EvaluationMetric::MemoryUsage, obj1),
        ],
    )
}

/// The same concave test problem MOEA/D is exercised on: `x` is the log-scaled
/// learning rate in `[0, 1]`, and `(x, 1 - sqrt(x))` is a front on which every `x`
/// is non-dominated.
fn concave_objectives(arch: &OptimizerArchitecture<f64>) -> (f64, f64) {
    let rate = arch
        .parameters
        .get("learning_rate")
        .copied()
        .unwrap_or(1e-3);
    let log_min = 1e-5f64.ln();
    let log_max = 1e-1f64.ln();
    let x = ((rate.max(1e-12).ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    (x, 1.0 - x.sqrt())
}

#[test]
fn nsga3_builds_reference_directions_and_sizes_the_population_to_them() {
    // Before this implementation the struct had no constructor at all, so nothing
    // below was reachable.
    let mut nsga3 = NSGA3::<f64>::with_seed(20, 0.9, 0.1, 1234);
    nsga3.initialize(&three_objective_config()).expect("init");

    let directions = nsga3.reference_directions();
    assert!(
        directions.len() >= 20,
        "expected at least the requested 20 directions, got {}",
        directions.len()
    );
    assert_eq!(nsga3.population_size(), directions.len());
    for direction in directions {
        assert_eq!(direction.len(), 3);
        let total: f64 = direction.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "{direction:?} must sum to 1");
    }
    assert_eq!(nsga3.name(), "NSGA-III");
    // The population is genuinely sampled, not one architecture repeated.
    let signatures: HashSet<Vec<String>> = nsga3
        .base
        .population
        .iter()
        .map(|individual| individual.architecture.components.clone())
        .collect();
    assert!(
        signatures.len() > 3,
        "initial population collapsed to {} distinct signatures",
        signatures.len()
    );
}

#[test]
fn nsga3_publishes_only_the_non_dominated_set() {
    let mut nsga3 = NSGA3::<f64>::with_seed(6, 0.9, 0.1, 99);
    nsga3.initialize(&two_objective_config()).expect("init");

    let front = nsga3
        .update_pareto_front(&[
            result2("edge_a", 0.0, 1.0),
            result2("edge_b", 1.0, 0.0),
            result2("middle", 0.5, 0.5),
            result2("dominated", 2.0, 2.0),
        ])
        .expect("update");

    let ids: HashSet<String> = front
        .solutions
        .iter()
        .map(|solution| solution.architecture.architecture_id.clone())
        .collect();
    assert!(ids.contains("edge_a") && ids.contains("edge_b") && ids.contains("middle"));
    assert!(
        !ids.contains("dominated"),
        "a dominated solution reached the front: {ids:?}"
    );
    // Real measured metrics, not the old hardcoded placeholders.
    assert!(front.metrics.hypervolume > 0.0);
    assert_eq!(front.metrics.num_solutions, front.solutions.len());
    let statistics = nsga3.get_statistics();
    assert!(statistics.algorithm_metrics["reference_directions"] > 0.0);
    assert!(statistics.algorithm_metrics["occupied_niches"] > 0.0);
}

#[test]
fn nsga3_niching_keeps_the_sparse_region_instead_of_the_crowded_one() {
    // A splitting front of nine mutually non-dominated points: eight packed into
    // one corner and one isolated at the other. Selecting four must not take all
    // four from the pack — that is exactly what a "take the first k" or a
    // crowding-blind selection does.
    let mut nsga3 = NSGA3::<f64>::with_seed(12, 0.9, 0.1, 20240818);
    nsga3.initialize(&two_objective_config()).expect("init");

    let mut results = Vec::new();
    for i in 0..8 {
        let x = 0.001 * i as f64;
        results.push(result2(&format!("packed{i}"), x, 1.0 - x));
    }
    results.push(result2("isolated", 1.0, 0.0));

    let selected = nsga3.select_by_rank_and_niching(&results, 4);
    assert_eq!(selected.len(), 4);
    let ids: Vec<&str> = selected
        .iter()
        .map(|index| results[*index].architecture.architecture_id.as_str())
        .collect();
    assert!(
        ids.contains(&"isolated"),
        "niching dropped the only solution in a whole region: {ids:?}"
    );
    // The niche table describes the selection that just happened.
    let occupied = nsga3
        .niche_count()
        .iter()
        .filter(|count| **count > 0)
        .count();
    assert!(
        occupied >= 2,
        "selection concentrated on {occupied} reference direction(s)"
    );
}

#[test]
fn nsga3_environmental_selection_bounds_the_population() {
    let mut nsga3 = NSGA3::<f64>::with_seed(6, 0.9, 0.1, 5);
    nsga3.initialize(&two_objective_config()).expect("init");
    let target = nsga3.population_size();

    // Three times the population, all mutually non-dominated.
    let count = target * 3;
    let results: Vec<SearchResult<f64>> = (0..count)
        .map(|i| {
            let x = i as f64 / (count - 1) as f64;
            result2(&format!("p{i}"), x, 1.0 - x)
        })
        .collect();
    nsga3.update_pareto_front(&results).expect("update");

    assert_eq!(
        nsga3.base.population.len(),
        target,
        "environmental selection must reduce the population back to its target"
    );
    // Everything kept is evaluated: unevaluated individuals must not survive when
    // evaluated ones are competing for the slots.
    assert!(nsga3
        .base
        .population
        .iter()
        .all(|individual| individual.objectives.len() == 2));
    // The extremes of the front survive the reduction.
    let ids: HashSet<String> = nsga3
        .base
        .population
        .iter()
        .map(|individual| individual.architecture.architecture_id.clone())
        .collect();
    assert!(
        ids.contains("p0") && ids.contains(&format!("p{}", count - 1)),
        "the extreme solutions were selected away: {ids:?}"
    );
}

#[test]
fn nsga3_spreads_a_three_objective_front_over_its_reference_directions() {
    let mut nsga3 = NSGA3::<f64>::with_seed(15, 0.9, 0.1, 616);
    nsga3.initialize(&three_objective_config()).expect("init");

    // Points on the unit simplex x + y + z = 1: all mutually non-dominated, and
    // spread over the whole simplex.
    let mut results = Vec::new();
    let mut index = 0usize;
    for i in 0..=4 {
        for j in 0..=(4 - i) {
            let k = 4 - i - j;
            results.push(result3(
                &format!("s{index}"),
                i as f64 / 4.0,
                j as f64 / 4.0,
                k as f64 / 4.0,
            ));
            index += 1;
        }
    }
    assert_eq!(results.len(), 15);
    let front = nsga3.update_pareto_front(&results).expect("update");
    assert_eq!(
        front.solutions.len(),
        15,
        "every point on the simplex is non-dominated"
    );

    // Niching must have spread the population over several directions rather than
    // piling it onto one.
    let occupied = nsga3
        .niche_count()
        .iter()
        .filter(|count| **count > 0)
        .count();
    assert!(
        occupied >= 5,
        "a 15-point simplex front occupied only {occupied} reference directions"
    );
    // Association counts describe the same population.
    let associated: usize = nsga3.association_count().iter().sum();
    assert!(
        associated >= 15,
        "only {associated} solutions were associated"
    );
}

#[test]
fn nsga3_optimizes_a_concave_front_end_to_end() {
    let mut nsga3 = NSGA3::<f64>::with_seed(12, 0.9, 0.3, 0xBEEF);
    nsga3.initialize(&two_objective_config()).expect("init");
    nsga3
        .set_hypervolume_reference(vec![1.5, 1.5])
        .expect("reference matches the objective count");

    let mut hypervolumes = Vec::new();
    for _ in 0..12 {
        let candidates = nsga3.select_candidates(&[], &[]).expect("reproduce");
        assert!(!candidates.is_empty());
        let results: Vec<SearchResult<f64>> = candidates
            .into_iter()
            .map(|arch| {
                let (obj0, obj1) = concave_objectives(&arch);
                result_for(arch, obj0, obj1)
            })
            .collect();
        let front = nsga3.update_pareto_front(&results).expect("update");
        hypervolumes.push(front.metrics.hypervolume);
    }

    let front = nsga3.get_pareto_front();
    assert!(
        front.solutions.len() >= 4,
        "expected a populated front, got {}",
        front.solutions.len()
    );
    let xs: Vec<f64> = front
        .solutions
        .iter()
        .map(|solution| solution.objectives[0])
        .collect();
    let min_x = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_x - min_x > 0.3,
        "front is not spread along the first objective: [{min_x}, {max_x}]"
    );
    for (i, a) in front.solutions.iter().enumerate() {
        for (j, b) in front.solutions.iter().enumerate() {
            if i == j {
                continue;
            }
            let dominated = a.objectives[0] >= b.objectives[0]
                && a.objectives[1] >= b.objectives[1]
                && (a.objectives[0] > b.objectives[0] || a.objectives[1] > b.objectives[1]);
            assert!(
                !dominated,
                "{:?} is dominated by {:?}",
                a.objectives, b.objectives
            );
        }
    }
    let first = hypervolumes.first().copied().unwrap_or(0.0);
    let last = hypervolumes.last().copied().unwrap_or(0.0);
    assert!(
        last >= first,
        "hypervolume regressed over 12 generations: {first} -> {last}"
    );
    assert!(last > 0.0);
}

#[test]
fn nsga3_mating_selection_varies_its_offspring() {
    let mut nsga3 = NSGA3::<f64>::with_seed(10, 0.9, 0.5, 777);
    nsga3.initialize(&two_objective_config()).expect("init");
    let offspring = nsga3.select_candidates(&[], &[]).expect("reproduce");
    assert_eq!(offspring.len(), nsga3.population_size());

    let ids: HashSet<&String> = offspring
        .iter()
        .map(|child| &child.architecture_id)
        .collect();
    assert_eq!(ids.len(), offspring.len(), "offspring ids must be unique");
    let signatures: HashSet<Vec<String>> = offspring
        .iter()
        .map(|child| child.components.clone())
        .collect();
    assert!(
        signatures.len() > 2,
        "mating produced only {} distinct component signatures",
        signatures.len()
    );
    // Externally supplied architectures join the mating pool.
    let external = vec![architecture("external")];
    let with_external = nsga3
        .select_candidates(&external, &[])
        .expect("reproduce with external pool");
    assert_eq!(with_external.len(), nsga3.population_size());
}

#[test]
fn nsga3_reports_an_error_instead_of_an_empty_front_without_objectives() {
    let empty = MultiObjectiveConfig::<f64> {
        objectives: Vec::new(),
        ..MultiObjectiveConfig::default()
    };
    let mut nsga3 = NSGA3::<f64>::with_seed(8, 0.9, 0.1, 3);
    assert!(nsga3.initialize(&empty).is_err());
    assert!(nsga3
        .update_pareto_front(&[result2("a", 1.0, 1.0)])
        .is_err());
    assert!(nsga3.select_candidates(&[], &[]).is_err());
}

#[test]
fn nsga3_seeding_is_reproducible_and_unseeded_runs_differ() {
    let signature = |nsga3: &NSGA3<f64>| -> Vec<String> {
        nsga3
            .base
            .population
            .iter()
            .map(|individual| individual.architecture.components.join("+"))
            .collect()
    };
    let mut a = NSGA3::<f64>::with_seed(10, 0.9, 0.1, 4242);
    a.initialize(&two_objective_config()).expect("init");
    let mut b = NSGA3::<f64>::with_seed(10, 0.9, 0.1, 4242);
    b.initialize(&two_objective_config()).expect("init");
    assert_eq!(signature(&a), signature(&b));

    let mut differ = false;
    for _ in 0..5 {
        let mut x = NSGA3::<f64>::new(10, 0.9, 0.1);
        x.initialize(&two_objective_config()).expect("init");
        let mut y = NSGA3::<f64>::new(10, 0.9, 0.1);
        y.initialize(&two_objective_config()).expect("init");
        if signature(&x) != signature(&y) {
            differ = true;
            break;
        }
    }
    assert!(
        differ,
        "two unseeded NSGA-III instances must not always produce identical populations"
    );
}
