// Regression tests for the advanced drift-analysis components
// (findings C4, C5, C6, C7, C8).

use super::*;

fn new_analyzer(window: usize) -> DriftPatternAnalyzer<f64> {
    DriftPatternAnalyzer::new(window)
}

fn feed(analyzer: &mut DriftPatternAnalyzer<f64>, values: &[f64]) -> PatternFeatures<f64> {
    let mut last = None;
    for value in values {
        last = Some(analyzer.ingest(*value).expect("ingest must succeed"));
    }
    last.expect("at least one value must be supplied")
}

// ---------------------------------------------------------------------------
// C4: `entropy = variance.ln().abs()` was `+inf`, eight features were
// hardcoded zeros and `fractal_dimension` was the invented constant 1.5.
// ---------------------------------------------------------------------------

#[test]
fn entropy_is_finite_for_a_constant_window() {
    let mut analyzer = new_analyzer(16);
    let features = feed(&mut analyzer, &[3.5f64; 16]);
    let entropy = features
        .entropy
        .expect("a full window must yield an entropy value");
    assert!(
        entropy.is_finite(),
        "C4 regression: a zero-variance window produced entropy {entropy} \
         (`variance.ln().abs()` is +inf for variance 0)"
    );
    assert!(
        entropy.abs() < 1e-12,
        "a constant window carries no information, so entropy must be 0 (got {entropy})"
    );
}

#[test]
fn window_derived_features_are_absent_until_the_window_fills() {
    let mut analyzer = new_analyzer(32);
    let first = analyzer.ingest(1.0).expect("ingest");
    assert!(
        first.entropy.is_none(),
        "C4 regression: a one-sample window reported an entropy value"
    );
    assert!(first.skewness.is_none());
    assert!(
        first.fractal_dimension.is_none(),
        "C4 regression: the fractal dimension was reported from a single sample \
         (it used to be the hardcoded 1.5)"
    );
    assert_eq!(first.variance, 0.0);
}

#[test]
fn fractal_dimension_is_measured_rather_than_constant() {
    let mut smooth = new_analyzer(64);
    let smooth_features = feed(
        &mut smooth,
        &(0..64).map(|i| i as f64).collect::<Vec<f64>>(),
    );
    let mut jagged = new_analyzer(64);
    let jagged_features = feed(
        &mut jagged,
        &(0..64)
            .map(|i| if i % 2 == 0 { 0.0 } else { 10.0 })
            .collect::<Vec<f64>>(),
    );

    let smooth_dimension = smooth_features
        .fractal_dimension
        .expect("a 64-sample window must yield a fractal dimension");
    let jagged_dimension = jagged_features
        .fractal_dimension
        .expect("a 64-sample window must yield a fractal dimension");

    assert!((1.0..=2.0).contains(&smooth_dimension));
    assert!((1.0..=2.0).contains(&jagged_dimension));
    assert!(
        jagged_dimension > smooth_dimension,
        "C4 regression: a jagged series must have a higher fractal dimension than a straight \
         ramp ({jagged_dimension} vs {smooth_dimension}); both used to be the constant 1.5"
    );
}

#[test]
fn trend_features_track_a_real_ramp() {
    let mut analyzer = new_analyzer(32);
    let features = feed(
        &mut analyzer,
        &(0..32).map(|i| 2.0 * i as f64).collect::<Vec<f64>>(),
    );
    let slope = features
        .trend_slope
        .expect("a full window must yield a slope");
    let strength = features
        .trend_strength
        .expect("a full window must yield a trend strength");
    assert!(
        (slope - 2.0).abs() < 1e-9,
        "C4 regression: the slope of a 2x ramp must be 2 (got {slope}); it used to be a \
         hardcoded 0"
    );
    assert!(
        (strength - 1.0).abs() < 1e-9,
        "a perfect line explains all the variance (got {strength})"
    );
}

#[test]
fn non_finite_values_are_rejected() {
    let mut analyzer = new_analyzer(16);
    assert!(
        analyzer.ingest(f64::NAN).is_err(),
        "a non-finite value must be reported instead of poisoning every feature"
    );
}

#[test]
fn empty_windows_are_rejected() {
    let mut analyzer = new_analyzer(16);
    assert!(analyzer.extract_features(&[]).is_err());
}

// ---------------------------------------------------------------------------
// C5: patterns were never learned, similarity was unbounded, strategy
// selection was a constant and outcomes were fabricated.
// ---------------------------------------------------------------------------

#[test]
fn similarity_is_bounded_and_peaks_at_identity() {
    let mut analyzer = new_analyzer(16);
    let close = feed(
        &mut analyzer,
        &(0..16).map(|i| i as f64).collect::<Vec<f64>>(),
    );
    let mut other = new_analyzer(16);
    let far = feed(
        &mut other,
        &(0..16).map(|i| 1000.0 * i as f64).collect::<Vec<f64>>(),
    );

    let identity = analyzer.calculate_similarity(&close, &close);
    let distant = analyzer.calculate_similarity(&close, &far);

    assert!(
        (identity - 1.0).abs() < 1e-9,
        "identical feature sets must be perfectly similar (got {identity})"
    );
    assert!(
        (0.0..=1.0).contains(&distant),
        "C5 regression: similarity left [0, 1] for distant patterns (got {distant}); the old \
         `1 - (|dmean| + |dvar|) / 2` is arbitrarily negative"
    );
    assert!(distant < identity);
}

#[test]
fn strategy_selection_depends_on_the_observed_features() {
    let mut selector = AdaptationStrategySelector::<f64>::new();
    let volatile = PatternFeatures {
        mean: 1.0,
        variance: 25.0, // high dispersion
        skewness: Some(0.0),
        kurtosis: Some(0.0),
        trend_slope: Some(0.0),
        trend_strength: Some(0.0), // no trend
        dominant_frequency: None,
        spectral_entropy: None,
        temporal_locality: None,
        persistence: Some(0.9),
        entropy: Some(0.1),
        fractal_dimension: None,
    };
    let trending = PatternFeatures {
        variance: 0.01, // low dispersion
        trend_strength: Some(0.95),
        ..volatile.clone()
    };
    let impact = DriftImpact {
        performance_degradation: 1.0,
        affected_metrics: Vec::new(),
        estimated_recovery_time: Duration::from_secs(10),
        confidence: 0.5,
        business_impact_score: 1.0,
        urgency_level: UrgencyLevel::Medium,
    };

    let volatile_choice = selector
        .select_strategy(&volatile, &impact, &None)
        .expect("selection must succeed")
        .expect("a strategy must be applicable to a high-variance window");
    let trending_choice = selector
        .select_strategy(&trending, &impact, &None)
        .expect("selection must succeed")
        .expect("a strategy must be applicable to a strongly trending window");

    assert_ne!(
        volatile_choice.id, trending_choice.id,
        "C5 regression: selection returned '{}' for both a high-variance and a strongly \
         trending window; it used to always return the hardcoded 'increase_lr'",
        volatile_choice.id
    );
    assert_ne!(volatile_choice.id, "increase_lr");
}

#[test]
fn applicability_conditions_are_actually_evaluated() {
    let strategies = AdaptationStrategySelector::<f64>::new();
    let decrease = strategies
        .strategies
        .iter()
        .find(|strategy| strategy.id == "decrease_learning_rate")
        .expect("the default strategy set must contain decrease_learning_rate");

    let high_variance = PatternFeatures::<f64> {
        mean: 0.0,
        variance: 10.0,
        skewness: None,
        kurtosis: None,
        trend_slope: None,
        trend_strength: None,
        dominant_frequency: None,
        spectral_entropy: None,
        temporal_locality: None,
        persistence: None,
        entropy: None,
        fractal_dimension: None,
    };
    let low_variance = PatternFeatures::<f64> {
        variance: 0.001,
        ..high_variance.clone()
    };

    assert!(
        (decrease.applicability(&high_variance) - 1.0).abs() < 1e-12,
        "C5 regression: applicability_conditions were never evaluated"
    );
    assert!(decrease.applicability(&low_variance).abs() < 1e-12);
}

#[test]
fn stored_events_carry_no_outcome_until_one_is_reported() {
    let mut database = DriftDatabase::<f64>::new();
    let features = PatternFeatures::<f64> {
        mean: 1.0,
        variance: 2.0,
        skewness: None,
        kurtosis: None,
        trend_slope: Some(0.5),
        trend_strength: Some(0.5),
        dominant_frequency: None,
        spectral_entropy: None,
        temporal_locality: None,
        persistence: None,
        entropy: None,
        fractal_dimension: None,
    };
    let strategy = AdaptationStrategySelector::<f64>::new()
        .strategies
        .first()
        .cloned()
        .expect("default strategies exist");

    database.store_event(&features, &[], &Some(strategy.clone()));
    assert_eq!(database.drift_events.len(), 1);
    assert!(
        database.drift_events[0].outcome.is_none(),
        "C5 regression: `store_event` recorded a fabricated outcome (success: true, \
         improvement 0.1) before anything had been observed"
    );
    assert!(database.pattern_outcomes().is_empty());

    let outcome = AdaptationOutcome {
        success: true,
        performance_improvement: 0.42,
        adaptation_time: Duration::from_secs(3),
        stability_period: Duration::from_secs(120),
        side_effects: Vec::new(),
    };
    let (strategy_id, _) = database
        .complete_pending_event(outcome.clone())
        .expect("the pending event must be completed");
    assert_eq!(strategy_id, strategy.id);
    let recorded = database.drift_events[0]
        .outcome
        .as_ref()
        .expect("the outcome must now be attached");
    assert_eq!(recorded.performance_improvement, 0.42);
    assert_eq!(database.pattern_outcomes()[&strategy.id].len(), 1);
    assert!(!database.find_similar(&features).is_empty());
}

#[test]
fn patterns_are_learned_only_after_a_real_outcome() {
    let config = DriftDetectorConfig {
        threshold: 1e-6,
        warningthreshold: 1e-9,
        window_size: 32,
        min_samples: 4,
        ..DriftDetectorConfig::default()
    };
    let mut detector = AdvancedDriftDetector::<f64>::new(config);

    let mut saw_drift = false;
    for step in 0..200 {
        let value = if step < 100 { 1.0 } else { 50.0 };
        let result = detector
            .detect_drift_advanced(value, &[])
            .expect("detection must succeed");
        if result.status == DriftStatus::Drift {
            saw_drift = true;
            break;
        }
    }
    assert!(saw_drift, "the scenario must produce a drift detection");
    assert!(
        detector.known_patterns().is_empty(),
        "C5 regression: a pattern was learned before any outcome was observed"
    );
    assert!(
        detector
            .stored_events()
            .iter()
            .any(|event| event.outcome.is_none()),
        "the drift event must be stored awaiting its outcome"
    );

    detector
        .record_adaptation_outcome(AdaptationOutcome {
            success: true,
            performance_improvement: 0.3,
            adaptation_time: Duration::from_secs(2),
            stability_period: Duration::from_secs(60),
            side_effects: Vec::new(),
        })
        .expect("recording an outcome must succeed");

    assert!(
        !detector.known_patterns().is_empty(),
        "C5 regression: `known_patterns` stayed empty, so `match_pattern` can never match"
    );
}

#[test]
fn reporting_an_outcome_with_nothing_pending_is_an_error() {
    let mut detector = AdvancedDriftDetector::<f64>::new(DriftDetectorConfig::default());
    assert!(detector
        .record_adaptation_outcome(AdaptationOutcome {
            success: true,
            performance_improvement: 0.1,
            adaptation_time: Duration::from_secs(1),
            stability_period: Duration::from_secs(1),
            side_effects: Vec::new(),
        })
        .is_err());
}

#[test]
fn the_bandit_prefers_the_action_with_the_better_measured_reward() {
    let mut bandit = EpsilonGreedyBandit::<f64>::new(0.0); // pure exploitation
    let actions = vec!["good".to_string(), "bad".to_string()];
    bandit.update("good", 1.0);
    bandit.update("bad", -1.0);
    for _ in 0..10 {
        assert_eq!(bandit.select(&actions).as_deref(), Some("good"));
    }
    assert!(bandit.action_values()["good"] > bandit.action_values()["bad"]);
}

// ---------------------------------------------------------------------------
// C6: the adapted thresholds were computed and then thrown away.
// ---------------------------------------------------------------------------

#[test]
fn adapted_thresholds_reach_the_detectors() {
    let config = DriftDetectorConfig {
        window_size: 32,
        ..DriftDetectorConfig::default()
    };
    let mut detector = AdvancedDriftDetector::<f64>::new(config);
    let before = detector.detector_thresholds();
    assert!(
        !before.is_empty(),
        "C6/C8 regression: `base_detectors` is empty, so nothing votes and nothing can have a \
         threshold applied"
    );

    for step in 0..60 {
        detector
            .detect_drift_advanced(1.0 + 0.01 * step as f64, &[])
            .expect("detection must succeed");
    }
    let after = detector.detector_thresholds();

    assert_eq!(before.len(), after.len());
    assert!(
        before
            .iter()
            .zip(after.iter())
            .any(|((_, old), (_, new))| (old - new).abs() > 1e-12),
        "C6 regression: no detector threshold changed, so the adaptive threshold manager is \
         still write-only bookkeeping"
    );
}

#[test]
fn detection_quality_feedback_shifts_threshold_adaptation() {
    let features = PatternFeatures::<f64> {
        mean: 1.0,
        variance: 1.0,
        skewness: None,
        kurtosis: None,
        trend_slope: Some(0.1),
        trend_strength: Some(0.5),
        dominant_frequency: None,
        spectral_entropy: None,
        temporal_locality: None,
        persistence: None,
        entropy: None,
        fractal_dimension: None,
    };
    let results = vec![("detector".to_string(), DriftStatus::Stable)];

    let mut baseline = AdaptiveThresholdManager::<f64>::new();
    let mut with_false_positives = AdaptiveThresholdManager::<f64>::new();
    with_false_positives.record_feedback(PerformanceFeedback {
        true_positive_rate: 0.1,
        false_positive_rate: 1.0,
        detection_delay: Duration::ZERO,
        adaptation_effectiveness: 0.0,
        timestamp: Instant::now(),
    });

    for _ in 0..20 {
        baseline.update_thresholds(&results, &features);
        with_false_positives.update_thresholds(&results, &features);
    }

    let baseline_threshold = baseline.thresholds()["detector"];
    let biased_threshold = with_false_positives.thresholds()["detector"];
    assert!(
        biased_threshold > baseline_threshold,
        "C6 regression: reported false positives must push thresholds up ({biased_threshold} \
         vs {baseline_threshold}); `performance_feedback` used to be write-only"
    );
}

// ---------------------------------------------------------------------------
// C7: the threshold, impact and pattern logs grew without bound.
// ---------------------------------------------------------------------------

#[test]
fn advanced_analysis_histories_stay_bounded() {
    let config = DriftDetectorConfig {
        window_size: 16,
        threshold: 1e-6,
        warningthreshold: 1e-9,
        min_samples: 4,
        ..DriftDetectorConfig::default()
    };
    let mut detector = AdvancedDriftDetector::<f64>::new(config);
    for step in 0..1200 {
        let value = 1.0 + (step % 17) as f64;
        detector
            .detect_drift_advanced(value, &[])
            .expect("detection must succeed");
    }

    assert!(
        detector.threshold_manager.threshold_history.len() <= 256,
        "C7 regression: the threshold history grew to {} entries",
        detector.threshold_manager.threshold_history.len()
    );
    assert!(
        detector.impact_analyzer.impact_history().len() <= 256,
        "C7 regression: the impact history grew to {} entries",
        detector.impact_analyzer.impact_history().len()
    );
    assert!(
        detector.pattern_analyzer.pattern_buffer.len() <= 256,
        "C7 regression: the pattern buffer grew to {} entries",
        detector.pattern_analyzer.pattern_buffer.len()
    );
    assert!(detector.stored_events().len() <= 1024);
}

// ---------------------------------------------------------------------------
// C8: combining an empty result set divided by zero.
// ---------------------------------------------------------------------------

#[test]
fn combining_no_detector_results_is_stable_not_nan() {
    let detector = AdvancedDriftDetector::<f64>::new(DriftDetectorConfig::default());
    let combined = detector.combine_detection_results(&[], &None);
    assert_eq!(combined.status, DriftStatus::Stable);
    assert!(
        combined.confidence.is_finite(),
        "C8 regression: an empty result set produced a non-finite confidence \
         (0 votes / 0 detectors)"
    );
    assert_eq!(combined.confidence, 0.0);
}

#[test]
fn drift_duration_estimate_never_collapses_to_zero() {
    let detector = AdvancedDriftDetector::<f64>::new(DriftDetectorConfig::default());
    let unmeasured = PatternFeatures::<f64> {
        mean: 0.0,
        variance: 0.0,
        skewness: None,
        kurtosis: None,
        trend_slope: None,
        trend_strength: None,
        dominant_frequency: None,
        spectral_entropy: None,
        temporal_locality: None,
        persistence: None,
        entropy: None,
        fractal_dimension: None,
    };
    let horizon = detector.estimate_drift_duration(&unmeasured);
    assert!(
        horizon > Duration::ZERO,
        "an unmeasured trend must fall back to the base horizon instead of multiplying it by a \
         placeholder zero"
    );
}

#[test]
fn context_classification_uses_every_weighted_feature() {
    let mut detector = ContextAwareDriftDetector::<f64>::new(DriftDetectorConfig::default());
    // A low-importance high value must not by itself select "high_activity".
    detector.update_context(&[
        ContextFeature {
            name: "spike".to_string(),
            value: 1.0,
            importance_weight: 0.01,
            temporal_stability: 0.0,
        },
        ContextFeature {
            name: "baseline".to_string(),
            value: 0.0,
            importance_weight: 1.0,
            temporal_stability: 1.0,
        },
    ]);
    assert_eq!(detector.current_context.as_deref(), Some("low_activity"));

    detector.update_context(&[ContextFeature {
        name: "spike".to_string(),
        value: 1.0,
        importance_weight: 1.0,
        temporal_stability: 0.0,
    }]);
    assert_eq!(detector.current_context.as_deref(), Some("high_activity"));
    assert!(
        !detector.transitions().is_empty(),
        "the observed context transition must be recorded"
    );
}

// ---------------------------------------------------------------------------
// F5: per-context detector banks
// ---------------------------------------------------------------------------

/// A deterministic value in `[-0.5, 0.5)`, so the regimes carry real
/// within-context variation rather than being two constants.
fn context_wobble(step: usize) -> f64 {
    let mut mixed = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    ((mixed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

fn context_feature(value: f64) -> ContextFeature<f64> {
    ContextFeature {
        name: "activity".to_string(),
        value,
        importance_weight: 1.0,
        temporal_stability: 1.0,
    }
}

/// F5: every context gets a detector bank of its own, and those banks do not
/// see each other's observations.
///
/// The stream alternates between two regimes that are each perfectly stationary
/// — around 1 and around 100 — so there is no drift in either context. A single
/// shared bank cannot say that: every switch enters its accumulators as a
/// 99-unit level change. The per-context banks must stay stable on exactly the
/// same values, which is only possible if each one saw only its own context's
/// observations.
#[test]
fn per_context_detector_banks_do_not_cross_contaminate() {
    let config = DriftDetectorConfig::default();
    let mut detector = ContextAwareDriftDetector::<f64>::new(config.clone());
    let mut shared = impls::build_detector_bank::<f64>(&config);

    let mut shared_alarms = 0usize;
    let mut context_alarms = 0usize;

    for step in 0..600usize {
        let is_low = step.is_multiple_of(2);
        let value = if is_low {
            1.0 + 0.05 * context_wobble(step)
        } else {
            100.0 + 0.05 * context_wobble(step)
        };

        detector.update_context(&[context_feature(if is_low { 0.0 } else { 1.0 })]);
        let per_context = detector.observe_in_context(value);
        assert_eq!(
            per_context.len(),
            3,
            "every context bank must run all three configured detectors"
        );
        if impls::combine_bank_status(&per_context) != DriftStatus::Stable {
            context_alarms += 1;
        }

        // The control uses the *same* combination rule, so the only difference
        // between the two counts is whose observations went into which bank.
        let shared_statuses: Vec<DriftStatus> = shared
            .iter_mut()
            .map(|detector| detector.update(value))
            .collect();
        if impls::combine_bank_status(&shared_statuses) != DriftStatus::Stable {
            shared_alarms += 1;
        }
    }

    assert_eq!(
        detector.context_ids(),
        vec!["high_activity", "low_activity"],
        "each classified context must have acquired a bank of its own"
    );
    assert_eq!(detector.context_count(), 2);

    assert_eq!(
        context_alarms, 0,
        "a context whose own observations never change level must never raise \
         drift; {context_alarms} of 600 steps did, so the banks are sharing state"
    );
    assert_eq!(
        detector.context_status("low_activity"),
        Some(DriftStatus::Stable)
    );
    assert_eq!(
        detector.context_status("high_activity"),
        Some(DriftStatus::Stable)
    );
    assert_eq!(
        detector.context_status("never_seen"),
        None,
        "a context that was never observed in has no verdict to report"
    );

    // Control: the very same values through one shared bank do raise alarms.
    // Without this the test above would also pass for a bank that never fires.
    assert!(
        shared_alarms > 100,
        "the control (one shared bank over the interleaved stream) raised only \
         {shared_alarms} alarms out of 600 steps, so the per-context result \
         proves nothing; measured at the time of writing it raises 300 — one \
         for every switch into the high regime"
    );
}

/// F5: the banks are genuinely per context — resetting them clears the recorded
/// verdicts without forgetting which contexts exist.
#[test]
fn per_context_banks_are_resettable_and_keep_their_contexts() {
    let mut detector = ContextAwareDriftDetector::<f64>::new(DriftDetectorConfig::default());
    for step in 0..120usize {
        let is_low = step.is_multiple_of(2);
        detector.update_context(&[context_feature(if is_low { 0.0 } else { 1.0 })]);
        detector.observe_in_context(if is_low { 1.0 } else { 100.0 });
    }
    assert_eq!(detector.context_count(), 2);
    assert!(detector.context_status("low_activity").is_some());

    detector.reset_context_models();
    assert_eq!(
        detector.context_count(),
        2,
        "resetting the banks must not forget the contexts themselves"
    );
    assert_eq!(
        detector.context_status("low_activity"),
        None,
        "a reset bank has no verdict until it observes again"
    );
}

/// F5: an observation that arrives before any context has been classified
/// cannot be attributed to a context, so it must not silently create one.
#[test]
fn an_unclassified_observation_creates_no_context_bank() {
    let mut detector = ContextAwareDriftDetector::<f64>::new(DriftDetectorConfig::default());
    assert!(detector.current_context().is_none());
    assert!(detector.observe_in_context(1.0).is_empty());
    assert_eq!(detector.context_count(), 0);
}

/// F5: the per-context banks are wired into the public detection path — they
/// are fed by `detect_drift_advanced` and their verdicts are reachable — rather
/// than being state that nothing ever runs.
#[test]
fn detect_drift_advanced_runs_the_per_context_banks() {
    let mut detector = AdvancedDriftDetector::<f64>::new(DriftDetectorConfig::default());
    assert_eq!(detector.context_detector().context_count(), 0);

    for step in 0..240usize {
        let is_low = step.is_multiple_of(2);
        let value = if is_low {
            1.0 + 0.05 * context_wobble(step)
        } else {
            100.0 + 0.05 * context_wobble(step)
        };
        detector
            .detect_drift_advanced(value, &[context_feature(if is_low { 0.0 } else { 1.0 })])
            .expect("detection must succeed");
    }

    let context = detector.context_detector();
    assert_eq!(
        context.context_ids(),
        vec!["high_activity", "low_activity"],
        "both contexts must have been observed in through the public path"
    );
    assert_eq!(
        context.context_status("low_activity"),
        Some(DriftStatus::Stable),
        "the low-activity context is stationary in its own right"
    );
    assert_eq!(
        context.context_status("high_activity"),
        Some(DriftStatus::Stable),
        "the high-activity context is stationary in its own right"
    );
    assert!(
        context.current_context().is_some(),
        "the classified context must be reported"
    );
}
