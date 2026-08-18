// Regression tests for the enhanced adaptive learning-rate controller
// (findings E1-E6).

use super::*;
use scirs2_core::ndarray::Array1;

fn config() -> AdaptiveLRConfig<f64> {
    AdaptiveLRConfig {
        base_lr: 0.5,
        min_lr: 1e-6,
        max_lr: 10.0,
        enable_gradient_adaptation: true,
        enable_performance_adaptation: true,
        enable_drift_adaptation: false,
        enable_resource_adaptation: false,
        enable_meta_learning: false,
        history_window_size: 100,
        adaptation_frequency: 1,
        adaptation_sensitivity: 1.0,
        use_ensemble_voting: true,
        step_time_budget: None,
        memory_budget_mb: None,
    }
}

fn controller(config: AdaptiveLRConfig<f64>) -> EnhancedAdaptiveLRController<f64> {
    EnhancedAdaptiveLRController::new(config).expect("a valid config must build a controller")
}

// ---------------------------------------------------------------------------
// E1: `resolve_signals` multiplied the hardcoded 0.001 by the vote instead of
// the live learning rate.
// ---------------------------------------------------------------------------

#[test]
fn adaptation_composes_from_the_live_learning_rate() {
    let mut controller = controller(config());
    let metrics = HashMap::new();

    let mut latest = controller.get_current_lr();
    for step in 0..6usize {
        let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
        // A rising loss: every signal should be asking for a smaller step.
        let loss = 1.0 + 0.1 * step as f64;
        latest = controller
            .update_learning_rate(&gradients, loss, &metrics, step)
            .expect("update must succeed");
    }

    assert!(
        latest > 0.1,
        "E1 regression: the learning rate collapsed to {latest} from a base of 0.5. The old \
         resolver ignored the live rate and returned `0.001 * multiplier`, so every adaptation \
         snapped it to ~0.001"
    );
    assert!(
        latest < 0.5,
        "a monotonically rising loss must reduce the learning rate (got {latest})"
    );
    let previous = controller
        .previous_lr()
        .expect("an adaptation event must be recorded");
    assert_ne!(
        previous,
        controller.get_current_lr(),
        "the recorded `old_lr` must be the rate before the adaptation, not the decision's own \
         output"
    );
}

#[test]
fn no_signals_leaves_the_learning_rate_untouched() {
    let mut controller = controller(AdaptiveLRConfig {
        enable_gradient_adaptation: false,
        enable_performance_adaptation: false,
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3]);
    let updated = controller
        .update_learning_rate(&gradients, 1.0, &HashMap::new(), 0)
        .expect("update must succeed");
    assert_eq!(
        updated, 0.5,
        "E1 regression: with every signal disabled the learning rate must not move; the old \
         resolver returned the hardcoded 0.001"
    );
}

#[test]
fn only_enabled_signals_contribute() {
    let mut controller = controller(AdaptiveLRConfig {
        enable_gradient_adaptation: false,
        enable_performance_adaptation: true,
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3]);
    for step in 0..4usize {
        controller
            .update_learning_rate(&gradients, 1.0 + step as f64, &HashMap::new(), step)
            .expect("update must succeed");
    }
    let decision = controller
        .last_decision()
        .expect("a decision must be recorded");
    assert!(
        decision
            .contributing_signals
            .iter()
            .all(|signal| *signal == AdaptationSignalType::LossProgression),
        "E1/E5 regression: a disabled signal contributed to the decision ({:?})",
        decision.contributing_signals
    );
    assert!(!controller.voting_history().is_empty());
}

#[test]
fn every_conflict_resolution_strategy_is_reachable() {
    let mut strategy = MultiSignalAdaptationStrategy::<f64>::new(&config())
        .expect("strategy construction must succeed");

    let votes = || {
        vec![
            SignalVote {
                signal_type: AdaptationSignalType::GradientMagnitude,
                recommended_lr_change: 0.5,
                confidence: 0.2,
                reasoning: "shrink".to_string(),
                timestamp: Instant::now(),
            },
            SignalVote {
                signal_type: AdaptationSignalType::LossProgression,
                recommended_lr_change: 1.02,
                confidence: 0.9,
                reasoning: "grow".to_string(),
                timestamp: Instant::now(),
            },
        ]
    };

    strategy.conflict_resolution = ConflictResolution::HighestConfidence;
    let highest = strategy
        .resolve_signals(votes(), 1.0, 1.0, true, 0)
        .expect("resolution must succeed");
    assert!(
        (highest.lr_multiplier - 1.02).abs() < 1e-12,
        "HighestConfidence must take the 0.9-confidence vote (got {})",
        highest.lr_multiplier
    );

    strategy.conflict_resolution = ConflictResolution::Conservative;
    let conservative = strategy
        .resolve_signals(votes(), 1.0, 1.0, true, 0)
        .expect("resolution must succeed");
    assert!(
        (conservative.lr_multiplier - 1.02).abs() < 1e-12,
        "Conservative must take the smallest change (got {})",
        conservative.lr_multiplier
    );

    strategy.conflict_resolution = ConflictResolution::MajorityVote { threshold: 0.9 };
    let majority = strategy
        .resolve_signals(votes(), 1.0, 1.0, true, 0)
        .expect("resolution must succeed");
    assert_eq!(
        majority.lr_multiplier, 1.0,
        "no direction reaches a 90% majority here, so nothing may change"
    );

    strategy.conflict_resolution = ConflictResolution::MetaLearned;
    let unmeasured = strategy
        .resolve_signals(votes(), 1.0, 1.0, true, 0)
        .expect("resolution must succeed");
    assert_eq!(
        unmeasured.lr_multiplier, 1.0,
        "without measured reliability the meta-learned resolution must not invent a direction"
    );
    assert!(unmeasured.rationale.contains("no measured reliability"));

    strategy.update_signal_reliability(AdaptationSignalType::GradientMagnitude, 1.0);
    let measured = strategy
        .resolve_signals(votes(), 1.0, 1.0, true, 0)
        .expect("resolution must succeed");
    assert!(
        measured.lr_multiplier < 1.0,
        "with only the shrink signal proven reliable the resolution must shrink (got {})",
        measured.lr_multiplier
    );
}

#[test]
fn adaptation_sensitivity_scales_the_step() {
    let sensitive = MultiSignalAdaptationStrategy::<f64>::new(&config())
        .expect("construction")
        .resolve_signals(
            vec![SignalVote {
                signal_type: AdaptationSignalType::LossProgression,
                recommended_lr_change: 2.0,
                confidence: 1.0,
                reasoning: "grow".to_string(),
                timestamp: Instant::now(),
            }],
            1.0,
            1.0,
            true,
            0,
        )
        .expect("resolution");
    let damped = MultiSignalAdaptationStrategy::<f64>::new(&config())
        .expect("construction")
        .resolve_signals(
            vec![SignalVote {
                signal_type: AdaptationSignalType::LossProgression,
                recommended_lr_change: 2.0,
                confidence: 1.0,
                reasoning: "grow".to_string(),
                timestamp: Instant::now(),
            }],
            1.0,
            0.1,
            true,
            0,
        )
        .expect("resolution");
    assert!(
        (sensitive.lr_multiplier - 2.0).abs() < 1e-12,
        "full sensitivity must apply the whole requested change"
    );
    assert!(
        (damped.lr_multiplier - 1.1).abs() < 1e-12,
        "a sensitivity of 0.1 must apply a tenth of the requested change (got {})",
        damped.lr_multiplier
    );
}

#[test]
fn adaptation_frequency_is_honoured() {
    let mut controller = controller(AdaptiveLRConfig {
        adaptation_frequency: 5,
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3]);

    // Step 0 is an adaptation step; 1..=4 are not.
    controller
        .update_learning_rate(&gradients, 1.0, &HashMap::new(), 0)
        .expect("update");
    let after_first = controller.get_current_lr();
    for step in 1..5usize {
        let value = controller
            .update_learning_rate(&gradients, 1.0 + step as f64, &HashMap::new(), step)
            .expect("update");
        assert_eq!(
            value, after_first,
            "adaptation_frequency = 5 must leave the learning rate alone on step {step}"
        );
    }
    let adapted = controller
        .update_learning_rate(&gradients, 20.0, &HashMap::new(), 5)
        .expect("update");
    assert_ne!(
        adapted, after_first,
        "step 5 is an adaptation step and the loss has risen sharply, so the rate must move"
    );
}

// ---------------------------------------------------------------------------
// E2: `meta_optimize` returned the literal 0.001.
// ---------------------------------------------------------------------------

#[test]
fn meta_optimization_does_not_drag_the_learning_rate_to_a_constant() {
    let mut controller = controller(AdaptiveLRConfig {
        enable_meta_learning: true,
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
    let mut latest = controller.get_current_lr();
    for step in 0..6usize {
        latest = controller
            .update_learning_rate(&gradients, 1.0, &HashMap::new(), step)
            .expect("update");
    }
    assert!(
        latest > 0.05,
        "E2 regression: with meta-learning on the rate fell to {latest} from a base of 0.5; the \
         old `meta_optimize` returned the constant 0.001 and it was blended in at 30% weight"
    );
}

#[test]
fn meta_learning_arms_learn_from_reported_effectiveness() {
    let mut controller = controller(AdaptiveLRConfig {
        enable_meta_learning: true,
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
    assert!(controller.meta_arm_counts().is_empty());

    for step in 0..4usize {
        controller
            .update_learning_rate(&gradients, 1.0, &HashMap::new(), step)
            .expect("update");
        controller.evaluate_adaptation_effectiveness(0.75);
    }

    assert!(
        !controller.meta_arm_counts().is_empty(),
        "E2 regression: `arm_counts` was never written to, so the bandit could not learn"
    );
    assert!(
        controller
            .meta_arm_rewards()
            .values()
            .any(|reward| (*reward - 0.75).abs() < 1e-9),
        "the reported effectiveness must become the played arm's measured reward"
    );
}

#[test]
fn a_registered_source_task_shifts_the_meta_proposal() {
    let mut without = controller(AdaptiveLRConfig {
        enable_meta_learning: true,
        ..config()
    });
    let mut with_transfer = controller(AdaptiveLRConfig {
        enable_meta_learning: true,
        ..config()
    });
    with_transfer.add_source_task(TaskData {
        optimal_lr_sequence: vec![0.05],
    });

    let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
    let plain = without
        .update_learning_rate(&gradients, 1.0, &HashMap::new(), 0)
        .expect("update");
    let transferred = with_transfer
        .update_learning_rate(&gradients, 1.0, &HashMap::new(), 0)
        .expect("update");
    assert_ne!(
        plain, transferred,
        "a registered source task must pull the proposal towards its schedule"
    );
}

// ---------------------------------------------------------------------------
// E3: the drift adapter always reported "No drift detected".
// ---------------------------------------------------------------------------

#[test]
fn the_drift_adapter_reacts_to_a_shifting_gradient_stream() {
    let mut adapter = DriftAwareAdapter::<f64>::new(&config()).expect("construction");
    assert!(
        !adapter.drift_detectors.is_empty(),
        "E3 regression: `drift_detectors` is empty, so no drift can ever be detected"
    );

    let mut reacted = false;
    let mut detected_drift = false;
    for step in 0..200usize {
        let magnitude = if step < 120 { 0.1 } else { 25.0 };
        let gradients = Array1::from_vec(vec![magnitude, -magnitude, magnitude / 2.0]);
        let signal = adapter
            .generate_signal(&gradients, step)
            .expect("signal generation must succeed");
        assert_ne!(
            signal.reasoning, "No drift detected",
            "E3 regression: the adapter still returns its hardcoded reasoning"
        );
        if (signal.recommended_lr_change - 1.0).abs() > 1e-9 {
            reacted = true;
        }
        if adapter.drift_detected() {
            detected_drift = true;
        }
    }
    assert!(
        reacted,
        "E3 regression: a stream whose gradient magnitude jumps 250x never produced anything \
         other than the neutral multiplier 1.0"
    );
    assert!(
        detected_drift,
        "at least one real detector must have reported drift during the shift"
    );
}

#[test]
fn the_drift_adapter_stays_neutral_on_a_stationary_stream() {
    let mut adapter = DriftAwareAdapter::<f64>::new(&config()).expect("construction");
    let mut reactions = 0usize;
    for step in 0..60usize {
        let gradients = Array1::from_vec(vec![0.5f64, -0.5, 0.25]);
        let signal = adapter
            .generate_signal(&gradients, step)
            .expect("signal generation must succeed");
        if (signal.recommended_lr_change - 1.0).abs() > 1e-9 {
            reactions += 1;
        }
    }
    assert!(
        reactions * 2 < 60,
        "a perfectly stationary gradient stream must not be treated as drifting most of the \
         time ({reactions}/60 reactions)"
    );
}

#[test]
fn distribution_tracking_measures_a_real_divergence() {
    let mut tracker = DistributionTracker::<f64>::default();
    let baseline = Array1::from_vec((0..32).map(|i| i as f64 / 32.0).collect::<Vec<f64>>());
    // The first observation has no reference to diverge from.
    assert_eq!(tracker.observe(&baseline), 0.0);
    for _ in 0..5 {
        tracker.observe(&baseline);
    }
    let steady = tracker.observe(&baseline);

    let shifted = Array1::from_vec((0..32).map(|i| 100.0 + i as f64).collect::<Vec<f64>>());
    let shifted_divergence = tracker.observe(&shifted);
    assert!(
        shifted_divergence >= steady,
        "a shifted distribution must not diverge less than a repeated one ({shifted_divergence} \
         vs {steady})"
    );
    assert!(shifted_divergence.is_finite());
}

// ---------------------------------------------------------------------------
// E4: `memory_pressure` was never written, and the never-written 0.0 was read
// as spare capacity so the learning rate was pushed up on every step.
// ---------------------------------------------------------------------------

#[test]
fn the_resource_signal_refuses_to_vote_without_a_measurement() {
    let mut adapter = ResourceAwareAdapter::<f64>::new(&config()).expect("construction");
    assert!(
        adapter.generate_signal(0).is_err(),
        "E4 regression: with nothing measured the adapter used to report `memory_pressure < 0.3` \
         and vote to *increase* the learning rate"
    );
}

#[test]
fn reported_memory_pressure_reduces_the_learning_rate() {
    let mut adapter = ResourceAwareAdapter::<f64>::new(&config()).expect("construction");
    adapter.record_memory_usage(950.0, Some(1000.0));
    let signal = adapter
        .generate_signal(0)
        .expect("a reported measurement must yield a signal");
    assert!(
        signal.recommended_lr_change < 1.0,
        "95% memory pressure must ask for a smaller step (got {})",
        signal.recommended_lr_change
    );
    assert_eq!(adapter.memory_tracker.memory_pressure, Some(0.95));
    assert_eq!(adapter.budget_violations(), 0);

    adapter.record_memory_usage(1500.0, Some(1000.0));
    assert_eq!(
        adapter.budget_violations(),
        1,
        "exceeding the configured memory budget must be counted"
    );
}

#[test]
fn memory_usage_without_a_budget_produces_no_pressure() {
    let mut adapter = ResourceAwareAdapter::<f64>::new(&config()).expect("construction");
    adapter.record_memory_usage(950.0, None);
    assert_eq!(
        adapter.memory_tracker.memory_pressure, None,
        "without a budget there is nothing to be under pressure against"
    );
    assert_eq!(adapter.memory_tracker.current_usage_mb, 950.0);
}

#[test]
fn step_times_are_measured_and_produce_real_time_pressure() {
    let mut controller = controller(AdaptiveLRConfig {
        enable_resource_adaptation: true,
        step_time_budget: Some(Duration::from_nanos(1)),
        ..config()
    });
    let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
    for step in 0..3usize {
        controller
            .update_learning_rate(&gradients, 1.0, &HashMap::new(), step)
            .expect("update");
    }
    let pressure = controller
        .resource_adapter
        .compute_tracker
        .time_pressure
        .expect("a measured step time and a configured budget must give a pressure");
    assert!(
        pressure > 1.0,
        "a one-nanosecond step budget is always exceeded, so pressure must exceed 1 (got \
         {pressure})"
    );
    assert!(controller.resource_adapter.budget_violations() > 0);
    assert!(
        controller
            .resource_adapter
            .compute_tracker
            .average_step_time
            > Duration::ZERO
    );
}

#[test]
fn energy_is_only_reported_when_it_is_measured() {
    let mut controller = controller(config());
    assert!(controller
        .resource_adapter
        .energy_tracker
        .energy_efficiency
        .is_none());
    controller.record_energy_sample(2.0);
    assert_eq!(
        controller.resource_adapter.energy_tracker.energy_efficiency,
        Some(0.5)
    );
    assert_eq!(
        controller.resource_adapter.energy_tracker.cumulative_energy,
        2.0
    );
}

// ---------------------------------------------------------------------------
// E6: the statistics left every map at its `Default` empty value.
// ---------------------------------------------------------------------------

#[test]
fn statistics_report_measured_signal_quality() {
    let mut controller = controller(config());
    let gradients = Array1::from_vec(vec![0.4f64, -0.3, 0.2]);
    for step in 0..5usize {
        controller
            .update_learning_rate(&gradients, 2.0 - 0.1 * step as f64, &HashMap::new(), step)
            .expect("update");
        controller.evaluate_adaptation_effectiveness(0.5);
    }

    let stats = controller.get_adaptation_statistics();
    assert_eq!(stats.total_adaptations, 5);
    assert_eq!(stats.successful_adaptations, 5);
    assert!(
        !stats.signal_reliability_scores.is_empty(),
        "E6 regression: the reliability map was left empty by `..Default::default()`"
    );
    assert!(
        !stats.signal_effectiveness.is_empty(),
        "E6 regression: the effectiveness map was left empty by `..Default::default()`"
    );
    assert!(
        (stats.convergence_speed_improvement - 0.5).abs() < 1e-9,
        "the mean reported effectiveness must be surfaced (got {})",
        stats.convergence_speed_improvement
    );
    assert!(stats.lr_volatility >= 0.0);
}

#[test]
fn gradient_statistics_are_computed_not_left_at_zero() {
    let mut adapter = GradientBasedAdapter::<f64>::new(&config()).expect("construction");
    for step in 0..40usize {
        let magnitude = 1.0 + 0.1 * (step % 5) as f64;
        adapter
            .generate_signal(&Array1::from_vec(vec![magnitude, 0.0]), step)
            .expect("signal generation must succeed");
    }
    assert!(
        adapter.norm_statistics.variance > 0.0,
        "E6 regression: the gradient norm statistics were left at their Default zeros"
    );
    assert_eq!(adapter.norm_statistics.percentiles.len(), 5);
    assert!(adapter.norm_statistics.percentiles[4] >= adapter.norm_statistics.percentiles[0]);
    assert!(!adapter.snr_estimator.snr_history.is_empty());
}
