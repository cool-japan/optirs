// Regression tests for the anomaly detector, its context and its response system.
//
// Split out of `anomaly_detection.rs` so that file stays under the 2000-line limit; it is
// wired back in as a `#[cfg(test)]` child module, so `super::*` still refers
// to the code under test.

use super::*;
use crate::streaming::adaptive_streaming::anomaly_statistical::{IQRDetector, ZScoreDetector};
use scirs2_core::ndarray::Array1;

fn make_point(value: f64) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(vec![value]),
        target: None,
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }
}

/// A1: `running_variance` is Welford's M2 accumulator
/// (`sum((x - mean)^2)`), not the variance itself. The z-score must
/// divide by `sqrt(M2 / n)`, not `sqrt(M2)` directly — otherwise the
/// detector's effective sensitivity shrinks by a factor of `sqrt(n)` as
/// more samples accumulate, and it eventually stops firing on real
/// outliers entirely.
#[test]
fn zscore_stays_sensitive_as_sample_count_grows() {
    let mut detector = ZScoreDetector::<f64>::new(3.0).expect("construct");

    // Feed enough stationary, low-variance samples that a naive
    // `sqrt(M2)` denominator (which grows roughly like `sqrt(n)`) would
    // have shrunk the z-score well below the threshold by sample 200,
    // even for a genuine multi-sigma outlier.
    for i in 0..200 {
        let value = 10.0 + 0.01 * ((i % 5) as f64 - 2.0); // tight noise around 10.0
        let point = make_point(value);
        StatisticalAnomalyDetector::update(&mut detector, &point).expect("update");
    }

    // A genuine outlier, many standard deviations away from the mean.
    let outlier = make_point(1000.0);
    let result = StatisticalAnomalyDetector::detect_anomaly(&mut detector, &outlier)
        .expect("detect_anomaly");

    assert!(
        result.is_anomaly,
        "A1 regression: z-score detector failed to flag an extreme \
         outlier after 200 samples (anomaly_score={})",
        result.anomaly_score
    );
}

/// A1: on textbook data with a known population standard deviation, the
/// reported z-score must match the standard `(x - mean) / std`
/// definition (within numerical tolerance for the running/Welford
/// computation), not be off by a `sqrt(n)` factor.
#[test]
fn zscore_matches_closed_form_for_known_distribution() {
    let mut detector = ZScoreDetector::<f64>::new(100.0).expect("construct"); // huge threshold: never flags, just inspect the score

    // Values 0..=20 (21 points): population mean = 10, population std
    // = sqrt(sum((x-10)^2)/21) ≈ 6.06.
    for i in 0..=20 {
        let point = make_point(i as f64);
        StatisticalAnomalyDetector::update(&mut detector, &point).expect("update");
    }

    let probe = make_point(20.0);
    let result =
        StatisticalAnomalyDetector::detect_anomaly(&mut detector, &probe).expect("detect_anomaly");

    // Expected z = (20 - 10) / 6.055... ~= 1.652
    let expected_std = {
        let mean = 10.0_f64;
        let sum_sq: f64 = (0..=20).map(|i| (i as f64 - mean).powi(2)).sum();
        (sum_sq / 21.0).sqrt()
    };
    let expected_z = (20.0 - 10.0) / expected_std;

    assert!(
        (result.anomaly_score - expected_z).abs() < 0.05,
        "A1 regression: z-score {} does not match closed-form {} \
         (likely dividing by sqrt(M2) instead of sqrt(M2/n))",
        result.anomaly_score,
        expected_z
    );
}
fn make_point_with(features: Vec<f64>, quality: f64) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(features),
        target: None,
        timestamp: Instant::now(),
        source_id: None,
        quality_score: quality,
        metadata: HashMap::new(),
    }
}

fn detector_with(strategy: AnomalyResponseStrategy) -> AnomalyDetector<f64> {
    let mut config = StreamingConfig::default();
    config.anomaly_config.response_strategy = strategy;
    AnomalyDetector::new(&config).expect("detector")
}

/// A5: the IQR detector sorted its window with
/// `partial_cmp(..).expect("unwrap failed")`, which panics on the first
/// `NaN`. This drives a `NaN` through the detector and asserts no panic.
#[test]
fn iqr_detector_survives_nan_in_the_window() {
    let mut detector = IQRDetector::<f64>::new(1.5).expect("detector");

    for i in 0..30 {
        let value = 10.0 + (i % 5) as f64;
        StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![value], 1.0))
            .expect("update");
    }
    // Poison the window: this is what used to panic on the next sort.
    StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![f64::NAN], 1.0))
        .expect("update");

    let probe = make_point_with(vec![12.0], 1.0);
    let result = StatisticalAnomalyDetector::detect_anomaly(&mut detector, &probe);
    assert!(
        result.is_ok(),
        "A5 regression: a NaN in the window must not abort detection"
    );
    let result = result.expect("result");
    assert!(
        !result.anomaly_score.is_nan(),
        "the anomaly score must remain finite in the presence of NaN"
    );
}

/// A5: with too few finite observations the detector must report an honest
/// error rather than computing quartiles over `NaN`.
#[test]
fn iqr_detector_errors_when_the_window_is_all_nan() {
    let mut detector = IQRDetector::<f64>::new(1.5).expect("detector");
    for _ in 0..25 {
        StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![f64::NAN], 1.0))
            .expect("update");
    }
    let probe = make_point_with(vec![1.0], 1.0);
    assert!(
        StatisticalAnomalyDetector::detect_anomaly(&mut detector, &probe).is_err(),
        "an all-NaN window has no quartiles; that must be an error"
    );
}

/// A5: `sort_ascending` places `NaN` last, but `+inf` sorts as an ordinary
/// largest value — so a finiteness guard, not just a `is_nan` guard, is needed
/// or the IQR (and therefore the whole decision band) becomes infinite.
#[test]
fn iqr_detector_survives_infinities_in_the_window() {
    let mut detector = IQRDetector::<f64>::new(1.5).expect("detector");
    for i in 0..30 {
        let value = 10.0 + (i % 5) as f64;
        StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![value], 1.0))
            .expect("update");
    }
    StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![f64::INFINITY], 1.0))
        .expect("update");
    StatisticalAnomalyDetector::update(
        &mut detector,
        &make_point_with(vec![f64::NEG_INFINITY], 1.0),
    )
    .expect("update");

    let result = StatisticalAnomalyDetector::detect_anomaly(
        &mut detector,
        &make_point_with(vec![12.0], 1.0),
    )
    .expect("detect_anomaly");
    assert!(
        result.anomaly_score.is_finite(),
        "an infinity in the window must not make the score infinite (got {})",
        result.anomaly_score
    );
    assert!(
        result.confidence.is_finite(),
        "confidence must stay finite, got {}",
        result.confidence
    );
}

/// A flagged verdict must never carry exactly zero confidence: an ensemble
/// averages confidences, and a zero would silently drag the aggregate down.
#[test]
fn a_flagged_verdict_never_reports_zero_confidence() {
    let mut detector = ZScoreDetector::<f64>::new(3.0).expect("detector");
    for i in 0..50 {
        let value = 10.0 + 0.1 * ((i % 5) as f64 - 2.0);
        StatisticalAnomalyDetector::update(&mut detector, &make_point_with(vec![value], 1.0))
            .expect("update");
    }
    // A value sitting almost exactly on the 3-sigma boundary.
    let std_dev = {
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.1 * ((i % 5) as f64 - 2.0))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64).sqrt()
    };
    let marginal = make_point_with(vec![10.0 + 3.0001 * std_dev], 1.0);
    let result =
        StatisticalAnomalyDetector::detect_anomaly(&mut detector, &marginal).expect("detect");
    assert!(result.is_anomaly, "the probe should sit just past 3 sigma");
    assert!(
        result.confidence > 0.0,
        "a flagged verdict must carry non-zero confidence"
    );
}

/// A3: `calculate_recent_statistics` returned a hard-coded two-element
/// summary (means `[0.5, 0.3]`, stds `[0.1, 0.15]`, ...) for every stream.
/// The summary must now be computed from the retained window and must be
/// sized to the real feature width.
#[test]
fn recent_statistics_are_computed_from_the_real_window() {
    let mut detector = detector_with(AnomalyResponseStrategy::Ignore);

    // Three features (not two), with exactly-known statistics.
    for value in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
        let point = make_point_with(vec![value, 100.0, -value], 1.0);
        detector.detect_anomaly(&point).expect("detect_anomaly");
    }
    assert!(detector.recent_window_len() >= 5);

    // Include the probe itself, which the summary also folds in.
    let probe = make_point_with(vec![3.0, 100.0, -3.0], 1.0);
    let context = detector.build_context_for_test(&probe).expect("context");
    let statistics = &context.recent_statistics;

    assert_eq!(
        statistics.means.len(),
        3,
        "A3 regression: the summary is still hard-coded to two features"
    );
    // Coordinate 0 over {1,2,3,4,5} plus the probe 3.0 -> mean 3.0.
    assert!(
        (statistics.means[0] - 3.0).abs() < 1e-12,
        "expected mean 3.0, got {}",
        statistics.means[0]
    );
    // Coordinate 1 is constant at 100: zero variance, zero skew/kurtosis.
    assert!((statistics.means[1] - 100.0).abs() < 1e-12);
    assert!(statistics.std_devs[1].abs() < 1e-12);
    assert!(statistics.skewness[1].abs() < 1e-12);
    assert!(statistics.kurtosis[1].abs() < 1e-12);
    // Coordinate 2 mirrors coordinate 0.
    assert!((statistics.means[2] + 3.0).abs() < 1e-12);
    assert!((statistics.min_values[2] + 5.0).abs() < 1e-12);
    assert!((statistics.max_values[2] + 1.0).abs() < 1e-12);
    assert!((statistics.medians[0] - 3.0).abs() < 1e-12);

    assert_ne!(
        statistics.means,
        vec![0.5, 0.3],
        "A3 regression: the hard-coded means are back"
    );
}

/// A3: with no context signals published, the vectors must be empty
/// ("not reported"), never the old invented placeholders.
#[test]
fn unpublished_context_signals_are_empty_not_invented() {
    let detector = detector_with(AnomalyResponseStrategy::Ignore);
    let probe = make_point_with(vec![1.0], 1.0);
    let context = detector.build_context_for_test(&probe).expect("context");
    assert!(context.performance_metrics.is_empty());
    assert!(context.resource_usage.is_empty());
    assert!(context.drift_indicators.is_empty());
}

/// A4: `trigger_response` was `Ok(())` — it queued nothing, executed
/// nothing and recorded nothing, so `response_actions` was always empty and
/// no state ever changed. Executing a `Quarantine` strategy must genuinely
/// hold the data point.
#[test]
fn quarantine_response_actually_retains_the_data_point() {
    let mut detector = detector_with(AnomalyResponseStrategy::Filter); // -> Quarantine

    // Train the statistical detectors on a tight distribution, then feed a
    // gross outlier so the ensemble fires.
    for i in 0..60 {
        let value = 10.0 + 0.01 * ((i % 5) as f64 - 2.0);
        detector
            .detect_anomaly(&make_point_with(vec![value], 1.0))
            .expect("detect_anomaly");
    }
    let flagged = detector
        .detect_anomaly(&make_point_with(vec![100_000.0], 1.0))
        .expect("detect_anomaly");

    assert!(flagged, "the gross outlier must be flagged");
    assert!(
        detector.quarantined_point_count() > 0,
        "A4 regression: the Quarantine response held nothing"
    );
    assert!(
        detector.response_execution_count() > 0,
        "A4 regression: no response execution was recorded"
    );
    let event = detector
        .get_recent_anomalies(1)
        .first()
        .cloned()
        .cloned()
        .expect("anomaly recorded");
    assert!(
        !event.response_actions.is_empty(),
        "A4 regression: response_actions is still empty"
    );
}

/// A4: `get_success_rate` returned a hard-coded `0.85` from construction.
/// It must be `None` before anything has run, and then the real ratio.
#[test]
fn response_success_rate_is_measured_not_hard_coded() {
    let detector = detector_with(AnomalyResponseStrategy::Adaptive);
    let diagnostics = detector.get_diagnostics();
    assert_eq!(
        diagnostics.response_success_rate, None,
        "A4 regression: a success rate was reported before any response ran"
    );
    assert_eq!(diagnostics.response_executions, 0);

    // A response system whose only configured action has no handler must
    // report a real 0% success rate, not 0.85.
    let mut unhandled = AnomalyResponseSystem::<f64>::new(&AnomalyResponseStrategy::Custom(
        "unregistered".to_string(),
    ))
    .expect("response system");
    // The default mapping for a Custom strategy is `Alert`, which succeeds;
    // override it with an action that genuinely has no handler.
    unhandled.response_strategies.insert(
        AnomalyType::StatisticalOutlier,
        vec![ResponseAction::TriggerRecovery],
    );

    let event = AnomalyEvent {
        id: 1,
        timestamp: Instant::now(),
        anomaly_type: AnomalyType::StatisticalOutlier,
        severity: AnomalySeverity::High,
        confidence: 0.9,
        data_point: make_point_with(vec![1.0], 1.0),
        detector_name: "test".to_string(),
        anomaly_score: 5.0,
        context: AnomalyContext {
            recent_statistics: DataStatistics {
                means: Vec::new(),
                std_devs: Vec::new(),
                min_values: Vec::new(),
                max_values: Vec::new(),
                medians: Vec::new(),
                skewness: Vec::new(),
                kurtosis: Vec::new(),
            },
            performance_metrics: Vec::new(),
            resource_usage: Vec::new(),
            drift_indicators: Vec::new(),
            time_since_last_anomaly: Duration::ZERO,
        },
        response_actions: Vec::new(),
    };
    let executed = unhandled.trigger_response(&event).expect("trigger");
    assert!(
        executed.is_empty(),
        "an action with no handler must not be reported as executed"
    );
    assert_eq!(
        unhandled.get_success_rate(),
        Some(0.0),
        "A4 regression: an all-failed history must report 0.0, not 0.85"
    );
    assert_eq!(unhandled.execution_count(), 1);
}

/// A4: a `ModelAdjustment` response must produce a real threshold change on
/// the statistical detectors, not merely a log line.
#[test]
fn model_adjustment_response_raises_real_thresholds() {
    let mut detector = detector_with(AnomalyResponseStrategy::Adaptive); // Log + ModelAdjustment

    let threshold_before = detector
        .statistical_detectors
        .get("zscore")
        .map(|d| d.get_threshold())
        .expect("zscore detector");

    for i in 0..60 {
        let value = 10.0 + 0.01 * ((i % 5) as f64 - 2.0);
        detector
            .detect_anomaly(&make_point_with(vec![value], 1.0))
            .expect("detect_anomaly");
    }
    let flagged = detector
        .detect_anomaly(&make_point_with(vec![100_000.0], 1.0))
        .expect("detect_anomaly");
    assert!(flagged);

    let threshold_after = detector
        .statistical_detectors
        .get("zscore")
        .map(|d| d.get_threshold())
        .expect("zscore detector");
    assert!(
        threshold_after > threshold_before,
        "A4 regression: ModelAdjustment did not change any threshold \
         ({threshold_before} -> {threshold_after})"
    );
    assert!(
        detector.response_log_entry_count() > 0,
        "A4 regression: the Log response wrote nothing"
    );
}

/// A2/A4: the false-positive rate must be `None` until ground truth is fed
/// back, and then the real `FP / (FP + TN)` ratio — not the target rate the
/// constructor seeded it with.
#[test]
fn false_positive_rate_requires_real_ground_truth() {
    let mut detector = detector_with(AnomalyResponseStrategy::Ignore);
    assert_eq!(
        detector.get_diagnostics().false_positive_rate,
        None,
        "the seeded target rate must not be reported as a measurement"
    );

    // 1 false positive out of 4 genuinely-normal points -> 0.25.
    detector.record_detection_outcome(true, false);
    detector.record_detection_outcome(false, false);
    detector.record_detection_outcome(false, false);
    detector.record_detection_outcome(false, false);
    // Plus a true positive, which does not enter the denominator.
    detector.record_detection_outcome(true, true);

    let rate = detector
        .get_diagnostics()
        .false_positive_rate
        .expect("rate measured");
    assert!(
        (rate - 0.25).abs() < 1e-12,
        "expected FP rate 0.25, got {rate}"
    );
    assert!(
        (rate - 0.05).abs() > 1e-9,
        "the old seeded 0.05 is still being reported"
    );
}
