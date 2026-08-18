//! Integration tests for [`super::HyperparameterOptimizer`]'s strategy dispatch
//! (F15). Every test here fails against the pre-fix code, where `Grid`,
//! `Bayesian`, `TPE` and `Evolutionary` all returned `self.random_search()`.

use super::*;

fn learning_rate_range() -> ParameterRange<f64> {
    ParameterRange {
        name: "learning_rate".to_string(),
        min_value: 1e-5,
        max_value: 1e-1,
        distribution: DistributionType::Uniform,
        log_scale: true,
        discrete_values: None,
    }
}

fn unit_range(name: &str) -> ParameterRange<f64> {
    ParameterRange {
        name: name.to_string(),
        min_value: 0.0,
        max_value: 1.0,
        distribution: DistributionType::Uniform,
        log_scale: false,
        discrete_values: None,
    }
}

/// A space with one continuous axis and one two-way categorical axis.
fn small_space() -> HyperparameterSpace<f64> {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter("x".to_string(), unit_range("x"));
    space.add_categorical_parameter(
        "optimizer".to_string(),
        vec!["adam".to_string(), "sgd".to_string()],
    );
    space
}

/// Feed one scored evaluation into the optimizer.
fn record(optimizer: &mut HyperparameterOptimizer<f64>, config: HyperparameterConfiguration<f64>) {
    let score = config.score.unwrap_or(0.0);
    optimizer.record_evaluation(HyperparameterEvaluation {
        configuration: config,
        metrics: EvaluationMetrics {
            primary_objective: score,
            secondary_objectives: HashMap::new(),
            validation_score: Some(score),
            training_time: 1.0,
            memory_usage: 1.0,
            converged: true,
            convergence_iterations: Some(1),
        },
        duration_seconds: 1.0,
        status: EvaluationStatus::Success,
        error_message: None,
    });
}

/// Score peaking at `x = 0.8` and rewarding the `adam` category.
fn score_for(x: f64, optimizer_name: &str) -> f64 {
    -(x - 0.8).abs() + if optimizer_name == "adam" { 0.5 } else { 0.0 }
}

/// Run `n` suggest/record rounds against `score_for`.
fn drive(optimizer: &mut HyperparameterOptimizer<f64>, n: usize) -> Vec<(f64, String)> {
    let mut observed = Vec::new();
    for _ in 0..n {
        let mut config = optimizer
            .suggest_configuration()
            .expect("suggestion must succeed");
        let x = config.parameters.get("x").copied().unwrap_or(0.0);
        let name = config
            .categorical_parameters
            .get("optimizer")
            .cloned()
            .unwrap_or_default();
        config.score = Some(score_for(x, &name));
        observed.push((x, name));
        record(optimizer, config);
    }
    observed
}

// ---- existing behaviour, preserved --------------------------------------

#[test]
fn test_hyperparameter_space_creation() {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter("learning_rate".to_string(), learning_rate_range());
    space.add_categorical_parameter(
        "optimizer".to_string(),
        vec!["adam".to_string(), "sgd".to_string()],
    );

    assert!(space.parameter_ranges().contains_key("learning_rate"));
    assert!(space.categorical_options().contains_key("optimizer"));
    assert!(!space.is_empty());
}

#[test]
fn test_random_search() {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter(
        "learning_rate".to_string(),
        ParameterRange {
            log_scale: false,
            ..learning_rate_range()
        },
    );

    let mut optimizer = HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Random, 11);
    let config = optimizer
        .suggest_configuration()
        .expect("random suggestion");

    assert!(config.parameters.contains_key("learning_rate"));
    assert!(config.parameters["learning_rate"] >= 1e-5);
    assert!(config.parameters["learning_rate"] <= 1e-1);
    assert_eq!(optimizer.last_strategy_used(), OptimizationStrategy::Random);
}

#[test]
fn test_evaluation_recording() {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter("x".to_string(), unit_range("x"));
    let mut optimizer = HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Random, 3);

    let config = HyperparameterConfiguration {
        id: "test_config".to_string(),
        parameters: HashMap::new(),
        categorical_parameters: HashMap::new(),
        score: Some(0.85),
        metadata: HashMap::new(),
    };
    record(&mut optimizer, config);

    assert_eq!(optimizer.get_statistics().num_evaluations, 1);
    assert_eq!(
        optimizer
            .get_best_configuration()
            .and_then(|best| best.score),
        Some(0.85)
    );
}

// ---- F15: grid ----------------------------------------------------------

#[test]
fn grid_strategy_enumerates_the_grid_instead_of_sampling_randomly() {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter("x".to_string(), unit_range("x"));
    space.add_categorical_parameter(
        "optimizer".to_string(),
        vec!["adam".to_string(), "sgd".to_string()],
    );

    let mut optimizer = HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Grid, 7);
    optimizer.set_grid_resolution(3);
    assert_eq!(optimizer.grid_size(), Some(6));

    let mut seen = std::collections::HashSet::new();
    for _ in 0..6 {
        let mut config = optimizer.suggest_configuration().expect("grid suggestion");
        assert_eq!(optimizer.last_strategy_used(), OptimizationStrategy::Grid);
        seen.insert((
            config.parameters["x"].to_bits(),
            config.categorical_parameters["optimizer"].clone(),
        ));
        config.score = Some(0.0);
        record(&mut optimizer, config);
    }
    // Random search would essentially never cover all six cells in six draws.
    assert_eq!(
        seen.len(),
        6,
        "the grid must be covered exactly, got {seen:?}"
    );

    // Continuous values must be exact grid coordinates, not arbitrary draws.
    for (bits, _) in &seen {
        let value = f64::from_bits(*bits);
        assert!(
            [0.0, 0.5, 1.0].iter().any(|g| (g - value).abs() < 1e-12),
            "{value} is not a grid coordinate"
        );
    }
}

#[test]
fn grid_strategy_is_reproducible_and_independent_of_the_seed() {
    let build = |seed: u64| {
        let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
        space.add_parameter("x".to_string(), unit_range("x"));
        let mut optimizer =
            HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Grid, seed);
        optimizer.set_grid_resolution(4);
        (0..4)
            .map(|_| {
                let mut config = optimizer.suggest_configuration().expect("grid");
                let x = config.parameters["x"];
                config.score = Some(0.0);
                record(&mut optimizer, config);
                x
            })
            .collect::<Vec<f64>>()
    };
    assert_eq!(build(1), build(123_456_789));
}

// ---- F15: TPE -----------------------------------------------------------

#[test]
fn tpe_strategy_beats_random_search_on_a_peaked_objective() {
    let mut tpe_optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::TPE, 2024);
    let tpe_observations = drive(&mut tpe_optimizer, 70);

    let mut random_optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::Random, 2024);
    let random_observations = drive(&mut random_optimizer, 70);

    // Compare only the post-initial-design half, where TPE is actually driving.
    let tpe_tail = &tpe_observations[35..];
    let random_tail = &random_observations[35..];
    let tpe_mean_gap =
        tpe_tail.iter().map(|(x, _)| (x - 0.8).abs()).sum::<f64>() / tpe_tail.len() as f64;
    let random_mean_gap = random_tail
        .iter()
        .map(|(x, _)| (x - 0.8).abs())
        .sum::<f64>()
        / random_tail.len() as f64;

    assert!(
        tpe_mean_gap < random_mean_gap * 0.75,
        "TPE mean distance to the optimum {tpe_mean_gap} must clearly beat random search's \
         {random_mean_gap}; the pre-fix implementation *was* random search"
    );

    let tpe_best = tpe_optimizer
        .get_best_configuration()
        .and_then(|c| c.score)
        .unwrap_or(f64::NEG_INFINITY);
    let random_best = random_optimizer
        .get_best_configuration()
        .and_then(|c| c.score)
        .unwrap_or(f64::NEG_INFINITY);
    assert!(
        tpe_best >= random_best,
        "TPE best {tpe_best} fell behind random {random_best}"
    );
    assert_eq!(
        tpe_optimizer.last_strategy_used(),
        OptimizationStrategy::TPE
    );
}

#[test]
fn tpe_reports_the_initial_design_phase_honestly() {
    let mut optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::TPE, 5);
    // First suggestion: no observations yet, so the prior is sampled and the
    // optimizer says so.
    let mut config = optimizer.suggest_configuration().expect("suggestion");
    assert_eq!(
        optimizer.last_strategy_used(),
        OptimizationStrategy::Random,
        "the initial design must be reported as Random, not silently as TPE"
    );
    assert_eq!(
        config.metadata.get("strategy").map(String::as_str),
        Some("Random")
    );
    config.score = Some(0.0);
    record(&mut optimizer, config);

    drive(&mut optimizer, 20);
    assert_eq!(
        optimizer.last_strategy_used(),
        OptimizationStrategy::TPE,
        "once enough observations exist, TPE itself must run"
    );
}

// ---- F15: Bayesian ------------------------------------------------------

#[test]
fn bayesian_strategy_uses_a_real_surrogate() {
    let mut optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::Bayesian, 909);
    optimizer.set_acquisition_function(AcquisitionFunction::UpperConfidenceBound);
    let observations = drive(&mut optimizer, 60);

    assert_eq!(
        optimizer.last_strategy_used(),
        OptimizationStrategy::Bayesian
    );

    // The surrogate must actually be recorded, not left as `None` forever.
    let model = optimizer
        .surrogate_model()
        .expect("the Bayesian strategy must record its surrogate");
    assert_eq!(model.model_type, SurrogateModelType::KernelRegression);
    assert_eq!(
        model.acquisition_function,
        AcquisitionFunction::UpperConfidenceBound
    );
    assert!(!model.training_data.is_empty());

    // Compared against random search on the same seed and budget: the surrogate
    // must genuinely bias the sampling, which the pre-fix `self.random_search()`
    // could not do by construction.
    let mut random_optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::Random, 909);
    let random_observations = drive(&mut random_optimizer, 60);

    let tail = &observations[30..];
    let random_tail = &random_observations[30..];
    let mean_gap = tail.iter().map(|(x, _)| (x - 0.8).abs()).sum::<f64>() / tail.len() as f64;
    let random_gap = random_tail
        .iter()
        .map(|(x, _)| (x - 0.8).abs())
        .sum::<f64>()
        / random_tail.len() as f64;
    assert!(
        mean_gap < random_gap * 0.9,
        "Bayesian mean distance to the optimum {mean_gap} must beat random search's {random_gap}"
    );

    // Mean tail score, not best-of-run: with an explicitly exploratory
    // acquisition (UCB, beta = 2) a single lucky random draw can hold the best
    // score, but the *average* suggestion quality still has to reflect the model.
    let mean_score = tail
        .iter()
        .map(|(x, name)| score_for(*x, name))
        .sum::<f64>()
        / tail.len() as f64;
    let random_mean_score = random_tail
        .iter()
        .map(|(x, name)| score_for(*x, name))
        .sum::<f64>()
        / random_tail.len() as f64;
    assert!(
        mean_score > random_mean_score,
        "Bayesian mean tail score {mean_score} must beat random search's {random_mean_score}"
    );
}

#[test]
fn every_acquisition_function_is_selectable_and_drives_the_search() {
    for acquisition in [
        AcquisitionFunction::ExpectedImprovement,
        AcquisitionFunction::ProbabilityOfImprovement,
        AcquisitionFunction::UpperConfidenceBound,
        AcquisitionFunction::EntropySearch,
    ] {
        let mut optimizer =
            HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::Bayesian, 17);
        optimizer.set_acquisition_function(acquisition);
        drive(&mut optimizer, 25);
        let model = optimizer.surrogate_model().expect("surrogate recorded");
        assert_eq!(model.acquisition_function, acquisition);
        assert_eq!(
            optimizer.last_strategy_used(),
            OptimizationStrategy::Bayesian,
            "{acquisition:?} must not fall back"
        );
    }
}

// ---- F15: Evolutionary --------------------------------------------------

#[test]
fn evolutionary_strategy_reads_its_population() {
    let mut optimizer =
        HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::Evolutionary, 66);
    assert!(optimizer.population().is_empty());

    let observations = drive(&mut optimizer, 60);

    // The population must be a bounded, fitness-sorted pool that the strategy
    // actually selects from (it used to be filled once and never read).
    let population = optimizer.population();
    assert!(!population.is_empty(), "the population must be maintained");
    assert!(population.len() <= 20, "the population must stay bounded");
    let scores: Vec<f64> = population
        .iter()
        .filter_map(|config| config.score)
        .collect();
    for window in scores.windows(2) {
        assert!(
            window[0] >= window[1],
            "the population must be sorted fittest-first: {scores:?}"
        );
    }
    assert_eq!(
        optimizer.last_strategy_used(),
        OptimizationStrategy::Evolutionary
    );

    let tail = &observations[30..];
    let good_fraction =
        tail.iter().filter(|(_, name)| name == "adam").count() as f64 / tail.len() as f64;
    assert!(
        good_fraction > 0.55,
        "selection pressure should favour the rewarding category, got {:.0}%",
        good_fraction * 100.0
    );
}

// ---- F15: unimplemented strategies are errors, not random search --------

#[test]
fn unimplemented_strategies_return_an_error() {
    for strategy in [
        OptimizationStrategy::ParticleSwarm,
        OptimizationStrategy::SuccessiveHalving,
        OptimizationStrategy::Hyperband,
        OptimizationStrategy::BOHB,
    ] {
        let mut optimizer = HyperparameterOptimizer::with_seed(small_space(), strategy, 1);
        let error = optimizer
            .suggest_configuration()
            .expect_err("an unimplemented strategy must not silently do random search");
        let message = format!("{error}");
        assert!(
            message.contains("not implemented"),
            "unexpected error for {strategy:?}: {message}"
        );
    }
}

#[test]
fn an_empty_search_space_is_an_error() {
    let space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    let mut optimizer = HyperparameterOptimizer::with_seed(space, OptimizationStrategy::Random, 1);
    assert!(optimizer.suggest_configuration().is_err());
}

#[test]
fn suggestions_always_satisfy_the_search_space() {
    let mut space: HyperparameterSpace<f64> = HyperparameterSpace::new();
    space.add_parameter("learning_rate".to_string(), learning_rate_range());
    space.add_parameter("dropout".to_string(), unit_range("dropout"));
    space.add_categorical_parameter(
        "optimizer".to_string(),
        vec!["adam".to_string(), "sgd".to_string(), "lion".to_string()],
    );

    for strategy in [
        OptimizationStrategy::Random,
        OptimizationStrategy::Grid,
        OptimizationStrategy::TPE,
        OptimizationStrategy::Bayesian,
        OptimizationStrategy::Evolutionary,
    ] {
        let mut optimizer = HyperparameterOptimizer::with_seed(space.clone(), strategy, 4321);
        for _ in 0..40 {
            let mut config = optimizer
                .suggest_configuration()
                .unwrap_or_else(|e| panic!("{strategy:?} failed: {e}"));
            assert!(
                optimizer.get_search_space().validate_configuration(&config),
                "{strategy:?} produced an invalid configuration: {config:?}"
            );
            config.score = Some(-config.parameters["learning_rate"].log10());
            record(&mut optimizer, config);
        }
    }
}

#[test]
fn seeded_optimizers_are_reproducible_and_unseeded_ones_differ() {
    let run = |seed: u64| {
        let mut optimizer =
            HyperparameterOptimizer::with_seed(small_space(), OptimizationStrategy::TPE, seed);
        drive(&mut optimizer, 30)
    };
    assert_eq!(run(1010), run(1010));
    assert_ne!(run(1010), run(2020));
}
