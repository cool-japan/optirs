// Behavioural regression tests for ensemble voting, in particular the
// measured-performance weighting behind `EnsembleVotingStrategy::Adaptive`.

use super::*;

fn result(is_anomaly: bool, score: f64) -> AnomalyDetectionResult<f64> {
    AnomalyDetectionResult {
        is_anomaly,
        anomaly_score: score,
        confidence: 0.5,
        anomaly_type: is_anomaly.then_some(AnomalyType::StatisticalOutlier),
        severity: AnomalySeverity::Low,
        metadata: HashMap::new(),
    }
}

/// Two members: `reliable` always agrees with the truth, `noisy` always
/// disagrees with it.
fn split_results(truth: bool) -> HashMap<String, AnomalyDetectionResult<f64>> {
    let mut results = HashMap::new();
    results.insert("reliable".to_string(), result(truth, 0.9));
    results.insert("noisy".to_string(), result(!truth, 0.1));
    results
}

/// F4: before any ground truth exists the adaptive strategy must be a working
/// uniform vote, not an error and not an invented ranking.
#[test]
fn adaptive_falls_back_to_uniform_weights_without_ground_truth() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Adaptive).expect("ensemble");
    assert!(
        ensemble.measured_weights().is_none(),
        "no ground truth has been recorded, so there is nothing measured to weight by"
    );

    let combined = ensemble
        .combine_results(split_results(true))
        .expect("adaptive voting must work before any label arrives");
    // One vote each way under uniform weights: the share is exactly one half,
    // which does not clear the strict majority.
    assert!(!combined.is_anomaly);
    assert!(
        (combined.anomaly_score - 0.5).abs() < 1e-12,
        "uniform weights must give the plain mean score, got {}",
        combined.anomaly_score
    );
}

/// F4: once ground truth has arrived the adaptive strategy must follow the
/// member that has been measured to be right, not the plain vote count.
#[test]
fn adaptive_follows_the_measured_member() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Adaptive).expect("ensemble");

    // Alternate the truth so both classes are observed; `reliable` is always
    // right, `noisy` always wrong.
    for round in 0..20 {
        let truth = round % 2 == 0;
        ensemble
            .combine_results(split_results(truth))
            .expect("combine");
        ensemble.record_outcome(truth);
    }

    let reliable = ensemble
        .detector_balanced_accuracy("reliable")
        .expect("reliable has labelled outcomes");
    let noisy = ensemble
        .detector_balanced_accuracy("noisy")
        .expect("noisy has labelled outcomes");
    assert!(
        (reliable - 1.0).abs() < 1e-12,
        "a member that is always right must measure 1.0, got {reliable}"
    );
    assert!(
        noisy.abs() < 1e-12,
        "a member that is always wrong must measure 0.0, got {noisy}"
    );

    let weights = ensemble.measured_weights().expect("measured weights");
    assert!(weights["reliable"] > weights["noisy"]);

    // The tie that uniform weights could not break is now decided by the
    // member with the measured skill.
    let combined = ensemble.combine_results(split_results(true)).expect("vote");
    assert!(
        combined.is_anomaly,
        "the adaptive vote ignored the only member measured to be reliable"
    );
    assert!(
        (combined.anomaly_score - 0.9).abs() < 1e-12,
        "the weighted mean score must follow the reliable member, got {}",
        combined.anomaly_score
    );
}

/// F4: a member with no measured skill at all must not silently dominate; when
/// *no* member has any skill the vote stays uniform rather than erroring.
#[test]
fn adaptive_stays_uniform_when_no_member_has_measured_skill() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Adaptive).expect("ensemble");

    // Both members are always wrong, so both measure a balanced accuracy of 0.
    for round in 0..20 {
        let truth = round % 2 == 0;
        let mut results = HashMap::new();
        results.insert("a".to_string(), result(!truth, 0.9));
        results.insert("b".to_string(), result(!truth, 0.9));
        ensemble.combine_results(results).expect("combine");
        ensemble.record_outcome(truth);
    }
    let weights = ensemble.measured_weights().expect("measured weights");
    assert!(weights.values().all(|w| w.abs() < 1e-12));

    let mut results = HashMap::new();
    results.insert("a".to_string(), result(true, 0.9));
    results.insert("b".to_string(), result(true, 0.9));
    let combined = ensemble
        .combine_results(results)
        .expect("a zero total weight must fall back to a uniform vote, not fail");
    assert!(combined.is_anomaly);
}

/// F4: one detection produces exactly one labelled outcome per member — a
/// second `record_outcome` with no intervening detection must not double-count.
#[test]
fn one_detection_is_scored_once() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Adaptive).expect("ensemble");
    ensemble
        .combine_results(split_results(true))
        .expect("combine");
    ensemble.record_outcome(true);
    ensemble.record_outcome(true);
    ensemble.record_outcome(true);

    let counters = ensemble
        .detector_counters
        .get("reliable")
        .expect("counters exist");
    assert_eq!(counters.labelled(), 1, "the detection was labelled twice");
}

/// The configured-weight strategy must honour the weights it is given — before
/// the setter existed there was no way to configure it at all.
#[test]
fn configured_weights_decide_the_weighted_vote() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Weighted).expect("ensemble");
    // Uniform: one vote each way, no strict majority.
    assert!(
        !ensemble
            .combine_results(split_results(true))
            .expect("combine")
            .is_anomaly
    );

    ensemble.set_detector_weight("reliable", 9.0);
    let combined = ensemble.combine_results(split_results(true)).expect("vote");
    assert!(
        combined.is_anomaly,
        "the configured weight was not consulted"
    );
}

/// F4: `Stacking` genuinely has no implementation behind it, so it must say so
/// precisely instead of impersonating another strategy — and the message must
/// no longer claim `Adaptive` is missing too.
#[test]
fn stacking_is_an_honest_error_naming_what_is_missing() {
    let mut ensemble =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::Stacking).expect("ensemble");
    let error = ensemble
        .combine_results(split_results(true))
        .expect_err("stacking has no meta-learner");
    assert!(error.contains("meta-learner"), "{error}");
    assert!(
        error.contains("Adaptive"),
        "the error should point at the strategy that *is* implemented: {error}"
    );
    assert!(
        !error.contains("`Adaptive` is not implemented"),
        "the message must not claim Adaptive is missing: {error}"
    );
}

/// The remaining strategies still compute the quantity they name.
#[test]
fn score_strategies_compute_what_they_name() {
    let mut results = HashMap::new();
    results.insert("a".to_string(), result(false, 0.10));
    results.insert("b".to_string(), result(false, 0.40));
    results.insert("c".to_string(), result(true, 0.90));

    let mut max_score =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::MaxScore).expect("ensemble");
    assert!(
        (max_score
            .combine_results(results.clone())
            .expect("max")
            .anomaly_score
            - 0.9)
            .abs()
            < 1e-12
    );

    let mut median =
        EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::MedianScore).expect("ensemble");
    assert!(
        (median
            .combine_results(results.clone())
            .expect("median")
            .anomaly_score
            - 0.4)
            .abs()
            < 1e-12
    );

    let mut average = EnsembleAnomalyDetector::<f64>::new(EnsembleVotingStrategy::AverageScore)
        .expect("ensemble");
    let expected = (0.10 + 0.40 + 0.90) / 3.0;
    assert!(
        (average
            .combine_results(results)
            .expect("average")
            .anomaly_score
            - expected)
            .abs()
            < 1e-12
    );
}
