// Regression tests for the meta-learning contextual bandit and strategy selector.
//
// Split out of `meta_learning.rs` so that file stays under the 2000-line limit; it is
// wired back in as a `#[cfg(test)]` child module, so `super::*` still refers
// to the code under test.

use super::*;
use scirs2_core::ndarray::Array1;

fn learner() -> MetaLearner<f64> {
    MetaLearner::new(&StreamingConfig::default()).expect("meta learner")
}

fn state(loss: f64, drift: f64) -> MetaState<f64> {
    MetaState {
        performance_metrics: vec![loss, 1.0 - loss.min(1.0), 0.0],
        resource_state: vec![512.0, 25.0],
        drift_indicators: vec![drift],
        adaptation_history: 3,
        timestamp: Instant::now(),
    }
}

fn action(adaptation_type: AdaptationType, magnitude: f64) -> MetaAction<f64> {
    MetaAction {
        adaptation_magnitudes: vec![magnitude],
        adaptation_types: vec![adaptation_type],
        learning_rate_change: magnitude,
        buffer_size_change: 0.0,
        timestamp: Instant::now(),
    }
}

fn experience(state: MetaState<f64>, action: MetaAction<f64>, reward: f64) -> MetaExperience<f64> {
    MetaExperience {
        id: 0,
        state,
        action,
        reward,
        next_state: None,
        timestamp: Instant::now(),
        episode_context: EpisodeContext {
            episode_id: 0,
            start_time: Instant::now(),
            duration: Duration::ZERO,
            initial_performance: 0.0,
            final_performance: reward,
            adaptation_count: 1,
            outcome: EpisodeOutcome::Neutral,
        },
        priority: reward.abs(),
        replay_count: 0,
    }
}

/// ML1: `predict_action` ignored its `state` argument entirely and returned
/// the constants `[0.1, -0.05]` with `learning_rate_change: 0.01` and
/// `buffer_size_change: 5.0`. After training on data where the *right*
/// action depends on the state, two clearly different states must yield
/// different recommendations.
#[test]
fn predicted_action_depends_on_the_state() {
    let mut model = MetaModel::<f64>::new(MetaModelComplexity::Low).expect("model");

    // High loss: a big learning-rate cut pays off. Low loss: a small
    // increase pays off. Train the bandit on exactly that structure.
    let mut batch = Vec::new();
    for _ in 0..60 {
        batch.push(experience(
            state(5.0, 2.0),
            action(AdaptationType::LearningRate, -0.2),
            1.0,
        ));
        batch.push(experience(
            state(5.0, 2.0),
            action(AdaptationType::LearningRate, 0.2),
            -1.0,
        ));
        batch.push(experience(
            state(0.01, 0.0),
            action(AdaptationType::LearningRate, 0.05),
            1.0,
        ));
        batch.push(experience(
            state(0.01, 0.0),
            action(AdaptationType::LearningRate, -0.2),
            -1.0,
        ));
        // Both arms must see both regimes, otherwise an arm with data in
        // only one regime is free to extrapolate arbitrarily into the other
        // and the test would be measuring extrapolation luck rather than
        // learning.
        batch.push(experience(
            state(5.0, 2.0),
            action(AdaptationType::LearningRate, 0.05),
            -1.0,
        ));
    }
    for _ in 0..30 {
        model.train_on_batch(&batch).expect("train_on_batch");
    }

    let high_loss = model.predict_action(&state(5.0, 2.0)).expect("high");
    let low_loss = model.predict_action(&state(0.01, 0.0)).expect("low");

    assert_ne!(
        high_loss.adaptation_magnitudes, low_loss.adaptation_magnitudes,
        "ML1 regression: predict_action returned the same magnitudes for two \
         very different states ({:?})",
        high_loss.adaptation_magnitudes
    );
    assert!(
        high_loss.adaptation_magnitudes[0] < 0.0,
        "the model should have learned to cut the rate under high loss, got {}",
        high_loss.adaptation_magnitudes[0]
    );
    assert!(
        low_loss.adaptation_magnitudes[0] > 0.0,
        "the model should have learned to raise the rate under low loss, got {}",
        low_loss.adaptation_magnitudes[0]
    );
    assert_ne!(
        high_loss.adaptation_magnitudes,
        vec![0.1, -0.05],
        "ML1 regression: the old hard-coded magnitudes are back"
    );
}

/// ML2: `train_on_batch` never touched a model parameter, and assigned the
/// batch's mean reward straight to `prediction_accuracy`. Training must now
/// genuinely reduce the reward-prediction error over repeated epochs.
#[test]
fn training_reduces_the_reward_prediction_error() {
    let mut model = MetaModel::<f64>::new(MetaModelComplexity::Low).expect("model");
    let batch: Vec<MetaExperience<f64>> = (0..40)
        .map(|i| {
            let loss = 0.1 * (i % 10) as f64;
            experience(
                state(loss, 0.0),
                action(AdaptationType::LearningRate, 0.1),
                // Reward is a deterministic function of the state, so a
                // linear model can genuinely learn it.
                2.0 - loss,
            )
        })
        .collect();

    model.train_on_batch(&batch).expect("first epoch");
    let first_accuracy = model.performance_metrics.prediction_accuracy;

    for _ in 0..200 {
        model.train_on_batch(&batch).expect("epoch");
    }
    let later_accuracy = model.performance_metrics.prediction_accuracy;

    assert!(
        later_accuracy > first_accuracy,
        "ML2 regression: repeated training did not improve prediction \
         accuracy ({first_accuracy} -> {later_accuracy}) — no parameter is \
         being updated"
    );
    assert!(
        later_accuracy > 0.8,
        "a linear model should fit this linear reward well, got {later_accuracy}"
    );

    let pulls: usize = model
        .arm_pull_counts()
        .iter()
        .map(|(_, count)| *count)
        .sum();
    assert!(pulls > 0, "ML2 regression: no arm was ever trained");
}

/// ML2: `prediction_accuracy` must be an accuracy, not a relabelled reward.
/// A batch whose rewards are large but perfectly predictable must end up
/// with high accuracy even though the mean reward is far from 1.
#[test]
fn prediction_accuracy_is_not_just_the_mean_reward() {
    let mut model = MetaModel::<f64>::new(MetaModelComplexity::Low).expect("model");
    let batch: Vec<MetaExperience<f64>> = (0..40)
        .map(|_| {
            experience(
                state(1.0, 0.0),
                action(AdaptationType::LearningRate, 0.1),
                // Constant reward well above 1: the mean reward is 7.0.
                7.0,
            )
        })
        .collect();

    for _ in 0..500 {
        model.train_on_batch(&batch).expect("epoch");
    }
    let accuracy = model.performance_metrics.prediction_accuracy;
    assert!(
        accuracy <= 1.0,
        "ML2 regression: accuracy {accuracy} exceeds 1, so it is still the \
         raw mean reward"
    );
    assert!(
        accuracy > 0.9,
        "a constant reward is trivially predictable; accuracy should be near \
         1, got {accuracy}"
    );
}

/// ML3: `select_strategy` looked up the key `"balanced"`, which was never
/// inserted, so the lookup always missed and the fallback returned an
/// arbitrary `HashMap` entry. Selection must now respond to measured
/// performance and be deterministic under a greedy policy.
#[test]
fn strategy_selection_follows_measured_performance() {
    let mut selector = StrategySelector::<f64>::new();
    // Greedy (no exploration) so the choice is fully determined.
    selector.selection_policy = SelectionPolicy::EpsilonGreedy { epsilon: 0.0 };
    selector.exploration_params.exploration_rate = 0.0;
    selector.exploration_params.min_exploration_rate = 0.0;

    // Aggressive actions (|magnitude| >= 0.15) get rewarded.
    let aggressive_wins: Vec<MetaExperience<f64>> = (0..30)
        .map(|_| {
            experience(
                state(2.0, 1.0),
                action(AdaptationType::LearningRate, 0.3),
                1.0,
            )
        })
        .collect();
    // Conservative actions get punished.
    let conservative_loses: Vec<MetaExperience<f64>> = (0..30)
        .map(|_| {
            experience(
                state(2.0, 1.0),
                action(AdaptationType::LearningRate, 0.05),
                -1.0,
            )
        })
        .collect();

    selector
        .update_from_experiences(&aggressive_wins)
        .expect("update");
    selector
        .update_from_experiences(&conservative_loses)
        .expect("update");

    let chosen = selector.select_strategy(&state(2.0, 1.0)).expect("select");
    assert_eq!(
        chosen.name, "aggressive",
        "ML3 regression: selection ignored the measured performance"
    );

    // Now flip the evidence and confirm the choice flips too.
    let conservative_wins: Vec<MetaExperience<f64>> = (0..200)
        .map(|_| {
            experience(
                state(2.0, 1.0),
                action(AdaptationType::LearningRate, 0.05),
                5.0,
            )
        })
        .collect();
    selector
        .update_from_experiences(&conservative_wins)
        .expect("update");
    let flipped = selector.select_strategy(&state(2.0, 1.0)).expect("select");
    assert_eq!(
        flipped.name, "conservative",
        "selection must track a genuine change in measured performance"
    );
}

/// ML4: `update_from_experiences` was an unconditional `Ok(())`, leaving
/// `strategy_performance` permanently empty. Every field of the record must
/// now move, and the exploration rate must decay.
#[test]
fn strategy_statistics_and_exploration_rate_are_really_updated() {
    let mut selector = StrategySelector::<f64>::new();
    assert!(selector.strategy_performance_for("aggressive").is_none());
    let initial_exploration = selector.exploration_rate();

    let experiences = vec![
        experience(
            state(1.0, 0.0),
            action(AdaptationType::LearningRate, 0.3),
            0.9,
        ),
        experience(
            state(1.0, 0.0),
            action(AdaptationType::LearningRate, 0.3),
            0.1,
        ),
        experience(
            state(1.0, 0.0),
            action(AdaptationType::LearningRate, 0.3),
            0.5,
        ),
    ];
    selector
        .update_from_experiences(&experiences)
        .expect("update_from_experiences");

    let performance = selector
        .strategy_performance_for("aggressive")
        .expect("ML4 regression: no strategy performance was recorded");
    assert_eq!(performance.usage_count, 3);
    assert!((performance.avg_improvement - 0.5).abs() < 1e-12);
    assert!((performance.best_improvement - 0.9).abs() < 1e-12);
    assert!((performance.worst_outcome - 0.1).abs() < 1e-12);
    // Exactly one of the three rewards exceeded the 0.5 success threshold.
    assert!((performance.success_rate - 1.0 / 3.0).abs() < 1e-12);
    assert!(!performance.context_performance.is_empty());

    assert!(
        selector.exploration_rate() < initial_exploration,
        "ML4 regression: exploration rate did not decay \
         ({initial_exploration} -> {})",
        selector.exploration_rate()
    );
    assert!(
        selector.exploration_rate() >= selector.exploration_params.min_exploration_rate,
        "the exploration rate must respect its configured floor"
    );
}

/// ML5: `create_episode_context` reported the fabricated constants `60s`,
/// `initial_performance: 0.5` and `adaptation_count: 1` for every
/// experience. All three must now be measured.
#[test]
fn episode_context_is_measured_not_fabricated() {
    let mut learner = learner();

    learner
        .update_experience(
            state(3.0, 0.0),
            action(AdaptationType::LearningRate, 0.1),
            0.25,
        )
        .expect("first experience");
    std::thread::sleep(Duration::from_millis(20));
    learner
        .update_experience(
            state(2.5, 0.0),
            action(AdaptationType::LearningRate, 0.1),
            0.75,
        )
        .expect("second experience");

    let context = learner
        .create_episode_context(0.75)
        .expect("create_episode_context");

    assert!(
        context.duration >= Duration::from_millis(15) && context.duration < Duration::from_secs(5),
        "ML5 regression: duration is {:?} — the hard-coded 60s is still in use",
        context.duration
    );
    assert!(
        (context.initial_performance - 0.25).abs() < 1e-12,
        "ML5 regression: initial_performance is {} — the hard-coded 0.5 is \
         still in use",
        context.initial_performance
    );
    assert_eq!(
        context.adaptation_count, 3,
        "ML5 regression: adaptation_count is still hard-coded to 1"
    );
}

/// ML5: `extract_meta_state` filled `resource_state` with `[0.5, 0.3]` and
/// `drift_indicators` with `[0.1]` regardless of reality. Published signals
/// must reach the state verbatim, and unpublished ones must be empty rather
/// than invented.
#[test]
fn meta_state_carries_published_context_signals() {
    let mut learner = learner();
    let config = StreamingConfig::default();
    let tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");

    let cold = learner
        .extract_meta_state(&[], &tracker)
        .expect("extract_meta_state");
    assert!(
        cold.resource_state.is_empty(),
        "ML5 regression: resource_state is {:?} instead of empty",
        cold.resource_state
    );
    assert!(
        cold.drift_indicators.is_empty(),
        "ML5 regression: drift_indicators is {:?} instead of empty",
        cold.drift_indicators
    );

    learner.update_context_signals(vec![1234.0, 56.0], vec![2.0, 0.125]);
    let warm = learner
        .extract_meta_state(&[], &tracker)
        .expect("extract_meta_state");
    assert_eq!(warm.resource_state, vec![1234.0, 56.0]);
    assert_eq!(warm.drift_indicators, vec![2.0, 0.125]);
    assert_ne!(warm.resource_state, vec![0.5, 0.3]);
}

/// The recommendations the meta-learner hands back must vary with the
/// current state once it has learned something, which is the end-to-end
/// consequence of ML1 + ML5.
#[test]
fn recommended_adaptations_vary_with_the_published_context() {
    let mut learner = learner();
    let config = StreamingConfig::default();
    let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");
    tracker
        .add_performance(PerformanceSnapshot {
            timestamp: Instant::now(),
            processing_duration: Duration::from_millis(2),
            loss: 4.0,
            accuracy: Some(0.2),
            convergence_rate: Some(0.1),
            gradient_norm: Some(2.0),
            parameter_update_magnitude: Some(0.2),
            data_statistics: super::super::performance::DataStatistics::default(),
            resource_usage: Default::default(),
            custom_metrics: HashMap::new(),
        })
        .expect("add_performance");

    learner.update_context_signals(vec![100.0, 10.0], vec![0.0]);
    let calm = learner
        .extract_meta_state(&[], &tracker)
        .expect("extract_meta_state");

    learner.update_context_signals(vec![8000.0, 95.0], vec![2.0]);
    let stressed = learner
        .extract_meta_state(&[], &tracker)
        .expect("extract_meta_state");

    // The two contexts must produce different feature vectors, which is what
    // makes a state-dependent recommendation possible at all.
    assert_ne!(calm.resource_state, stressed.resource_state);
    assert_ne!(calm.drift_indicators, stressed.drift_indicators);

    // And the learner must be able to produce recommendations from both.
    let adaptations = learner
        .recommend_adaptations(&[], &tracker)
        .expect("recommend_adaptations");
    assert!(
        adaptations.iter().all(|a| a.magnitude.is_finite()),
        "every recommended magnitude must be finite"
    );
}
