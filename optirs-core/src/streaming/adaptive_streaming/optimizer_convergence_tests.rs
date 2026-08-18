// O5 convergence-rate and adaptive-step regression tests for
// `AdaptiveStreamingOptimizer`.
//
// Extracted verbatim from `optimizer.rs`, which had reached the 2000-line
// ceiling; the module is still declared there as
// `mod o5_convergence_rate_tests`, so the test paths are unchanged.

use super::*;
use scirs2_core::ndarray::Ix1;

fn make_snapshot(loss: f64) -> PerformanceSnapshot<f64> {
    PerformanceSnapshot {
        timestamp: Instant::now(),
        processing_duration: Duration::from_millis(5),
        loss,
        accuracy: None,
        convergence_rate: None,
        gradient_norm: None,
        parameter_update_magnitude: None,
        data_statistics: DataStatistics::default(),
        resource_usage: ResourceUsage::default(),
        custom_metrics: HashMap::new(),
    }
}

fn make_point(features: Vec<f64>) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(features),
        target: None,
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }
}

fn make_optimizer() -> AdaptiveStreamingOptimizer<crate::optimizers::SGD<f64>, f64, Ix1> {
    // Previously `()` was passed as the "base optimizer" -- which compiled
    // only because nothing ever used it. A real optimizer is now required.
    AdaptiveStreamingOptimizer::new(
        crate::optimizers::SGD::new(0.01),
        StreamingConfig::default(),
    )
    .expect("construct")
}

/// O5: `compute_convergence_rate` must report a *positive* rate when
/// loss is genuinely decreasing over the tracked window, and negative
/// when it is increasing. `get_recent_losses` returns most-recent-first,
/// so the previous `recent_losses[0] - recent_losses[len-1]` computed
/// `newest - oldest`, which is negative for a converging (loss
/// decreasing) run — exactly inverted.
#[test]
fn positive_rate_when_loss_is_decreasing() {
    let mut opt = make_optimizer();
    // Oldest -> newest: loss falls from 10.0 to 1.0.
    for loss in [10.0, 8.0, 6.0, 4.0, 2.0, 1.0] {
        opt.performance_tracker
            .add_performance(make_snapshot(loss))
            .expect("add_performance");
    }

    let params = Array::<f64, Ix1>::zeros(3);
    let rate = opt
        .compute_convergence_rate(&params)
        .expect("compute_convergence_rate");

    assert!(
        rate > 0.0,
        "O5 regression: convergence rate was not positive for a \
         genuinely decreasing loss sequence (rate={rate})"
    );
}

/// O5: the mirror case — loss increasing (diverging) must report a
/// negative rate.
#[test]
fn negative_rate_when_loss_is_increasing() {
    let mut opt = make_optimizer();
    for loss in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0] {
        opt.performance_tracker
            .add_performance(make_snapshot(loss))
            .expect("add_performance");
    }

    let params = Array::<f64, Ix1>::zeros(3);
    let rate = opt
        .compute_convergence_rate(&params)
        .expect("compute_convergence_rate");

    assert!(
        rate < 0.0,
        "O5 regression: convergence rate was not negative for a \
         genuinely increasing (diverging) loss sequence (rate={rate})"
    );
}
/// O2: `compute_feature_median` used to `return Ok(features.clone())`, so
/// `adapt_for_anomaly` compared every value against itself, `diff` was
/// always exactly zero, and no value was ever clipped. This asserts the
/// median is a real order statistic over the rolling window and that an
/// extreme value genuinely gets pulled towards it.
#[test]
fn feature_median_is_a_real_order_statistic_over_the_window() {
    let mut opt = make_optimizer();

    // A window whose per-coordinate medians are known exactly:
    // coordinate 0 -> {1,2,3,4,5} median 3; coordinate 1 -> {10,20,30,40,50} median 30.
    for (a, b) in [
        (1.0, 10.0),
        (2.0, 20.0),
        (3.0, 30.0),
        (4.0, 40.0),
        (5.0, 50.0),
    ] {
        opt.remember_features(&Array1::from_vec(vec![a, b]));
    }
    assert_eq!(opt.feature_window_len(), 5);

    let probe = Array1::from_vec(vec![1000.0, -1000.0]);
    let median = opt
        .compute_feature_median(&probe)
        .expect("compute_feature_median");

    assert_eq!(
        median.to_vec(),
        vec![3.0, 30.0],
        "O2 regression: the median echoed its input instead of summarising \
         the rolling window"
    );
    assert_ne!(
        median.to_vec(),
        probe.to_vec(),
        "O2 regression: compute_feature_median returned its argument"
    );
}

/// O2: the consequence — `adapt_for_anomaly` must actually clip an extreme
/// coordinate towards the window median. Against the old code the value came
/// back untouched.
#[test]
fn adapt_for_anomaly_clips_extreme_values_towards_the_median() {
    let mut opt = make_optimizer();
    for value in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9] {
        opt.remember_features(&Array1::from_vec(vec![value]));
    }

    let outlier = make_point(vec![10_000.0]);
    let original = outlier.features[0];
    let adapted = opt.adapt_for_anomaly(outlier).expect("adapt_for_anomaly");

    assert!(
        adapted.features[0] < original,
        "O2 regression: an extreme value was not clipped at all \
         (before={original}, after={})",
        adapted.features[0]
    );
    assert!(
        adapted.features[0] < 1_000.0,
        "the clipped value {} is still nowhere near the window median ~10",
        adapted.features[0]
    );
}

/// O1: the learning-rate controller was a stub whose rate never moved. A
/// real AdaGrad-style controller must shrink the rate as squared gradient
/// norm accumulates, and must report the delta it applied.
#[test]
fn learning_rate_controller_responds_to_real_gradients() {
    let config = StreamingConfig::default();
    let mut controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");

    let initial = controller.current_rate();
    assert!(controller.last_change().is_none());
    assert_eq!(controller.accumulated_squared_gradient_norm(), 0.0);

    let large_gradient = Array1::from_vec(vec![5.0, -5.0, 5.0]);
    let after_first = controller.update_learning_rate(&large_gradient);
    assert!(
        after_first < initial,
        "O1 regression: a large gradient did not shrink the rate \
         ({initial} -> {after_first})"
    );
    assert!(
        controller.accumulated_squared_gradient_norm() > 0.0,
        "O1 regression: the gradient was ignored entirely"
    );
    assert!(
        controller.last_change().is_some(),
        "O1 regression: last_change is still hard-coded to None"
    );

    let after_second = controller.update_learning_rate(&large_gradient);
    assert!(
        after_second < after_first,
        "the accumulator must keep shrinking the rate ({after_first} -> {after_second})"
    );
    assert!(
        after_second >= config.learning_rate_config.min_rate,
        "the rate must respect the configured floor"
    );
}

/// O1: `compute_adaptation` used to ignore its argument and echo the base
/// rate. It must now move the rate in opposite directions for a falling and
/// a rising loss trend.
#[test]
fn learning_rate_adaptation_follows_the_loss_trend() {
    let config = StreamingConfig::default();
    let controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
    let current = controller.current_rate();

    // Oldest -> newest, loss falling: the rate should grow.
    let improving = [10.0, 8.0, 6.0, 4.0, 2.0];
    let grown = controller.compute_adaptation(&improving);
    assert!(
        grown > current,
        "O1 regression: a falling loss did not grow the rate \
         ({current} -> {grown})"
    );

    // Oldest -> newest, loss rising: the rate should shrink.
    let worsening = [2.0, 4.0, 6.0, 8.0, 10.0];
    let shrunk = controller.compute_adaptation(&worsening);
    assert!(
        shrunk < current,
        "O1 regression: a rising loss did not shrink the rate \
         ({current} -> {shrunk})"
    );
    assert!(
        (grown - shrunk).abs() > 1e-12,
        "the two trends must not produce the same rate"
    );

    // Too little data to fit a trend: unchanged, not fabricated.
    assert_eq!(controller.compute_adaptation(&[]), current);
    assert_eq!(controller.compute_adaptation(&[3.0]), current);
}

/// O1: a non-finite or non-positive proposal must be rejected rather than
/// silently destroying the optimizer.
#[test]
fn learning_rate_controller_rejects_invalid_rates() {
    let config = StreamingConfig::default();
    let mut controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
    let before = controller.current_rate();

    controller.apply_adaptation(f64::NAN);
    controller.apply_adaptation(-1.0);
    controller.apply_adaptation(0.0);
    assert_eq!(
        controller.current_rate(),
        before,
        "an invalid proposal must leave the rate untouched"
    );

    controller.apply_adaptation(0.01);
    assert!((controller.current_rate() - 0.01).abs() < 1e-12);
}

/// O3: a custom adaptation with no registered handler must be an honest
/// error rather than a `println!` that makes it look applied.
#[test]
fn custom_adaptation_without_a_handler_is_an_error() {
    let mut opt = make_optimizer();
    let adaptation = Adaptation {
        adaptation_type: AdaptationType::Custom("nonexistent".to_string()),
        magnitude: 0.1,
        target_component: "somewhere".to_string(),
        parameters: HashMap::new(),
        priority: AdaptationPriority::Normal,
        timestamp: Instant::now(),
    };
    let result = opt.apply_adaptations(std::slice::from_ref(&adaptation));
    assert!(
        result.is_err(),
        "an unhandled custom adaptation must not report success"
    );
}

/// A3: the anomaly detector's context signals must be real values published
/// by the optimizer, not the old hard-coded 0.8/0.7, 0.6/0.5, 0.1.
#[test]
fn anomaly_context_signals_come_from_real_state() {
    let mut opt = make_optimizer();
    opt.performance_tracker
        .add_performance(make_snapshot(7.25))
        .expect("add_performance");

    opt.publish_anomaly_context_signals()
        .expect("publish_anomaly_context_signals");

    // Drive one detection so a context is actually built.
    let point = make_point(vec![1.0, 2.0, 3.0]);
    opt.anomaly_detector
        .detect_anomaly(&point)
        .expect("detect_anomaly");

    let context = opt
        .anomaly_detector
        .build_context_for_test(&point)
        .expect("context");
    assert_eq!(
        context.performance_metrics.first().copied(),
        Some(7.25),
        "the published loss must reach the anomaly context verbatim"
    );
    assert!(
        !context.drift_indicators.is_empty(),
        "drift indicators must be published"
    );
    assert_ne!(
        context.performance_metrics,
        vec![0.8, 0.7],
        "A3 regression: the old hard-coded performance placeholders are back"
    );
}
/// `compute_loss` used to score `data_point.features` against the target and
/// ignore `parameters` entirely, so the reported loss never moved when the
/// model improved. It must now be a genuine function of the parameters.
#[test]
fn loss_is_a_function_of_the_parameters() {
    let opt = make_optimizer();
    let batch = vec![StreamingDataPoint {
        features: Array1::from_vec(vec![1.0, 2.0]),
        target: Some(Array1::from_vec(vec![5.0])),
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }];

    // w = (1, 2) gives <w, x> = 1 + 4 = 5, an exact fit.
    let exact = Array::<f64, Ix1>::from_vec(vec![1.0, 2.0]);
    let exact_loss = opt.compute_loss(&batch, &exact).expect("loss");
    assert!(
        exact_loss.abs() < 1e-12,
        "an exact fit must have zero loss, got {exact_loss}"
    );

    // w = (0, 0) predicts 0 against a target of 5, so the loss is 25.
    let wrong = Array::<f64, Ix1>::zeros(2);
    let wrong_loss = opt.compute_loss(&batch, &wrong).expect("loss");
    assert!(
        (wrong_loss - 25.0).abs() < 1e-12,
        "expected loss 25 for a zero model, got {wrong_loss}"
    );
    assert!(
        wrong_loss > exact_loss,
        "the loss must respond to the parameters ({exact_loss} vs {wrong_loss})"
    );
}

/// `compute_accuracy` used to count a point as correct whenever its
/// `quality_score > 0.5` — it measured input data quality, not the model.
/// A high-quality point that the model predicts badly must now count as
/// wrong.
#[test]
fn accuracy_scores_predictions_not_input_quality() {
    let opt = make_optimizer();
    // Perfect input quality, but the target is unreachable for w = 0.
    let batch = vec![StreamingDataPoint {
        features: Array1::from_vec(vec![1.0, 1.0]),
        target: Some(Array1::from_vec(vec![100.0])),
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }];

    let zero_model = Array::<f64, Ix1>::zeros(2);
    let bad = opt.compute_accuracy(&batch, &zero_model).expect("accuracy");
    assert_eq!(
        bad, 0.0,
        "a quality-1.0 point the model gets completely wrong must not count \
         as correct"
    );

    // w = (50, 50) predicts exactly 100.
    let good_model = Array::<f64, Ix1>::from_vec(vec![50.0, 50.0]);
    let good = opt.compute_accuracy(&batch, &good_model).expect("accuracy");
    assert_eq!(good, 1.0, "an exact prediction must count as correct");
}

/// An unlabelled batch cannot be scored, so accuracy must be `0`, not the
/// perfect `1.0` the previous implementation returned.
#[test]
fn unlabelled_batches_do_not_report_perfect_accuracy() {
    let opt = make_optimizer();
    let batch = vec![make_point(vec![1.0, 2.0])];
    let parameters = Array::<f64, Ix1>::zeros(2);
    assert_eq!(
        opt.compute_accuracy(&batch, &parameters).expect("accuracy"),
        0.0
    );
}
