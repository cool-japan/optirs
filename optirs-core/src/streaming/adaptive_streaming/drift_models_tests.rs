// Behavioural regression tests for the model-based drift detectors.
//
// Each detector is exercised on two deterministic streams: a stationary one,
// which must raise no drift at all, and one whose target relationship shifts
// part-way through, which must raise drift. The generators use a fixed linear
// congruential sequence rather than a random source so the verdicts are
// reproducible.

use super::*;
use crate::streaming::adaptive_streaming::optimizer::StreamingDataPoint;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::time::Instant;

/// Deterministic value in `[0, 1)` for index `index`.
///
/// A fixed multiplicative sequence, taken modulo one: reproducible across runs
/// and platforms, and decorrelated enough to act as noise for these tests.
fn pseudo_uniform(index: usize, salt: u64) -> f64 {
    let mixed = (index as u64)
        .wrapping_add(salt)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((mixed >> 11) as f64) / ((1u64 << 53) as f64)
}

/// One labelled point whose target is a fixed function of its two features
/// plus bounded noise.
///
/// `regime` selects the relationship: `0` is the stationary one, `1` is the
/// post-shift one (different coefficients *and* a large offset), which is a
/// genuine concept drift rather than a covariate shift.
fn point(index: usize, regime: usize) -> StreamingDataPoint<f64> {
    let first = pseudo_uniform(index, 0x1234_5678);
    let second = pseudo_uniform(index, 0x9ABC_DEF0);
    let noise = pseudo_uniform(index, 0x0F0F_0F0F) - 0.5;
    let target = if regime == 0 {
        2.0 * first - 1.0 * second + 0.5 + 0.05 * noise
    } else {
        -3.0 * first + 4.0 * second + 12.0 + 0.05 * noise
    };
    StreamingDataPoint {
        features: Array1::from_vec(vec![first, second]),
        target: Some(Array1::from_vec(vec![target])),
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }
}

fn batch(start: usize, count: usize, regime: usize) -> Vec<StreamingDataPoint<f64>> {
    (start..start + count).map(|i| point(i, regime)).collect()
}

/// Feeds `rounds` batches of `size` points from `regime` and reports how many
/// `detect_drift` calls raised drift after the warm-up.
fn run_stream<D: ModelBasedDetector<f64>>(
    detector: &mut D,
    start: usize,
    rounds: usize,
    size: usize,
    regime: usize,
) -> usize {
    let mut fired = 0usize;
    for round in 0..rounds {
        let data = batch(start + round * size, size, regime);
        if let Ok(result) = detector.detect_drift(&data) {
            if result.drift_detected {
                fired += 1;
            }
        }
    }
    fired
}

// ---------------------------------------------------------------------------
// shared tracker
// ---------------------------------------------------------------------------

/// The drift rule must be quiet while the error level is constant, and must
/// fire once the level steps up and stays up for a full run.
#[test]
fn tracker_fires_only_on_a_sustained_step_in_the_error_level() {
    let mut tracker = PrequentialErrorTracker::new(0.2).expect("tracker");

    for _ in 0..600 {
        tracker.observe(1.0);
    }
    assert!(
        !tracker.drift_detected().expect("verdict"),
        "a perfectly constant error level must never look like drift"
    );
    assert_eq!(
        tracker.run_length(),
        0,
        "a constant error level must never start a degradation run"
    );

    // A ten-fold step in the error level.
    for step in 0..MIN_DRIFT_RUN - 1 {
        tracker.observe(10.0);
        assert!(
            !tracker.drift_detected().expect("verdict"),
            "drift was reported after only {} observations above the threshold, \
             but {MIN_DRIFT_RUN} are required",
            step + 1
        );
    }
    tracker.observe(10.0);
    assert!(
        tracker.drift_detected().expect("verdict"),
        "a ten-fold step sustained for {MIN_DRIFT_RUN} observations must be \
         reported as drift"
    );
    assert!(
        tracker.degradation().expect("degradation") > 0.2,
        "the reported degradation must exceed the configured threshold"
    );
}

/// A non-positive or non-finite sensitivity is a configuration error, not
/// something to silently clamp.
#[test]
fn tracker_rejects_an_unusable_sensitivity() {
    assert!(PrequentialErrorTracker::new(0.0).is_err());
    assert!(PrequentialErrorTracker::new(-0.5).is_err());
    assert!(PrequentialErrorTracker::new(f64::NAN).is_err());
}

// ---------------------------------------------------------------------------
// neural network
// ---------------------------------------------------------------------------

/// The hidden layer must not be initialised symmetrically: identical rows
/// receive identical gradients forever, which collapses the MLP to a single
/// unit and makes the "neural network" detector a linear model in disguise.
#[test]
fn neural_network_breaks_hidden_layer_symmetry_deterministically() {
    let mut first = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    let mut second = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    first.update_model(&batch(0, 64, 0)).expect("train");
    second.update_model(&batch(0, 64, 0)).expect("train");

    assert_eq!(first.input_width(), 2);
    let rows = &first.hidden_weights;
    let mut distinct = 0usize;
    for (index, row) in rows.iter().enumerate().skip(1) {
        if (row[0] - rows[index - 1][0]).abs() > 1e-12 {
            distinct += 1;
        }
    }
    assert!(
        distinct >= MLP_HIDDEN_UNITS - 2,
        "the hidden layer collapsed to {} distinct rows out of {MLP_HIDDEN_UNITS}",
        distinct + 1
    );

    // Same construction, same data, same state: the detector is reproducible.
    assert_eq!(
        first.current_error().map(f64::to_bits),
        second.current_error().map(f64::to_bits),
        "two identically constructed detectors diverged on identical input"
    );
}

/// The MLP must actually learn: its prediction error late in a stationary
/// stream has to be far below its error at the start.
#[test]
fn neural_network_learns_the_relationship() {
    let mut detector = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    detector.update_model(&batch(0, 30, 0)).expect("train");
    let early = detector.current_error().expect("early error");
    detector.update_model(&batch(30, 2_000, 0)).expect("train");
    let late = detector.current_error().expect("late error");
    assert!(
        late < early * 0.2,
        "the MLP did not learn: squared error went from {early} to {late}"
    );
}

/// F1: a stationary stream must not raise drift from the neural detector.
#[test]
fn neural_network_stationary_stream_raises_no_drift() {
    let mut detector = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    let fired = run_stream(&mut detector, 0, 20, 60, 0);
    assert_eq!(
        fired, 0,
        "a stationary stream raised {fired} drift verdicts out of 20 batches"
    );
}

/// F1: a shift in the target relationship must raise drift from the neural
/// detector.
#[test]
fn neural_network_target_shift_raises_drift() {
    let mut detector = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    assert_eq!(run_stream(&mut detector, 0, 20, 60, 0), 0);
    let fired = run_stream(&mut detector, 100_000, 4, 60, 1);
    assert!(
        fired > 0,
        "the neural detector missed a complete change of the target relationship"
    );
}

// ---------------------------------------------------------------------------
// decision tree
// ---------------------------------------------------------------------------

/// The tree must actually be fit, and its splits must attribute error
/// reduction to the features that carry the signal.
#[test]
fn decision_tree_fits_and_attributes_importance() {
    let mut detector = DecisionTreeDriftDetector::<f64>::new(0.2).expect("detector");
    detector.update_model(&batch(0, 400, 0)).expect("train");
    assert!(detector.is_fitted(), "no tree was ever fit");

    let importances = detector.feature_importances();
    assert_eq!(importances.len(), 2, "importances: {importances:?}");
    assert!(
        importances.iter().sum::<f64>() > 0.0,
        "the fitted tree reduced no squared error at all: {importances:?}"
    );
    assert!(
        importances[0] > importances[1],
        "the target depends twice as strongly on the first feature, so it must \
         carry more of the error reduction: {importances:?}"
    );
}

/// F1: a stationary stream must not raise drift from the tree detector.
#[test]
fn decision_tree_stationary_stream_raises_no_drift() {
    let mut detector = DecisionTreeDriftDetector::<f64>::new(0.2).expect("detector");
    let fired = run_stream(&mut detector, 0, 20, 60, 0);
    assert_eq!(
        fired, 0,
        "a stationary stream raised {fired} drift verdicts out of 20 batches"
    );
}

/// F1: a shift in the target relationship must raise drift from the tree
/// detector.
#[test]
fn decision_tree_target_shift_raises_drift() {
    let mut detector = DecisionTreeDriftDetector::<f64>::new(0.2).expect("detector");
    assert_eq!(run_stream(&mut detector, 0, 20, 60, 0), 0);
    let fired = run_stream(&mut detector, 100_000, 4, 60, 1);
    assert!(
        fired > 0,
        "the tree detector missed a complete change of the target relationship"
    );
}

// ---------------------------------------------------------------------------
// ensemble
// ---------------------------------------------------------------------------

/// F1: a stationary stream must not raise drift from the ensemble, even though
/// its linear member uses a min-tracking baseline that is noisier on its own —
/// that is exactly what a majority vote is for.
#[test]
fn ensemble_stationary_stream_raises_no_drift() {
    let mut detector = EnsembleDriftDetector::<f64>::new(0.2).expect("detector");
    assert_eq!(detector.member_names().len(), 3);
    let fired = run_stream(&mut detector, 0, 20, 60, 0);
    assert_eq!(
        fired, 0,
        "a stationary stream raised {fired} ensemble drift verdicts out of 20 batches"
    );
}

/// F1: a shift in the target relationship must raise drift from the ensemble.
#[test]
fn ensemble_target_shift_raises_drift() {
    let mut detector = EnsembleDriftDetector::<f64>::new(0.2).expect("detector");
    assert_eq!(run_stream(&mut detector, 0, 20, 60, 0), 0);
    let fired = run_stream(&mut detector, 100_000, 4, 60, 1);
    assert!(
        fired > 0,
        "the ensemble missed a complete change of the target relationship"
    );
}

/// The ensemble's significance is Fisher's combination of its members' own
/// p-values, so it must be a genuine probability and must sharpen when every
/// member agrees.
#[test]
fn ensemble_confidence_is_a_real_combined_p_value() {
    let mut detector = EnsembleDriftDetector::<f64>::new(0.2).expect("detector");
    run_stream(&mut detector, 0, 20, 60, 0);
    let stationary = detector
        .detect_drift(&batch(2_000, 60, 0))
        .expect("stationary verdict");
    assert!(
        (0.0..=1.0).contains(&stationary.confidence),
        "confidence {} is outside [0, 1]",
        stationary.confidence
    );

    let mut shifted = detector
        .detect_drift(&batch(100_000, 60, 1))
        .expect("shift verdict");
    for round in 1..4 {
        shifted = detector
            .detect_drift(&batch(100_000 + round * 60, 60, 1))
            .expect("shift verdict");
    }
    assert!(
        shifted.confidence > stationary.confidence,
        "the combined p-value did not sharpen when every member saw the same \
         shift: {} vs {}",
        shifted.confidence,
        stationary.confidence
    );
}

/// Weighted voting is a real alternative to the uniform majority: a weight
/// vector that is not a valid distribution is an honest error rather than a
/// silently normalised guess.
#[test]
fn ensemble_weights_are_validated_and_honoured() {
    assert!(EnsembleDriftDetector::<f64>::with_weights(0.2, &[1.0, 1.0]).is_err());
    assert!(EnsembleDriftDetector::<f64>::with_weights(0.2, &[1.0, -1.0, 1.0]).is_err());
    assert!(EnsembleDriftDetector::<f64>::with_weights(0.2, &[0.0, 0.0, 0.0]).is_err());

    // A weight vector dominated by one member turns the vote into that
    // member's verdict: with all the weight on the neural network the ensemble
    // must agree with a standalone neural detector fed the same stream.
    let mut weighted =
        EnsembleDriftDetector::<f64>::with_weights(0.2, &[0.0, 1.0, 0.0]).expect("detector");
    let mut solo = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    for round in 0..20 {
        let data = batch(round * 60, 60, 0);
        assert_eq!(
            weighted
                .detect_drift(&data)
                .expect("weighted")
                .drift_detected,
            solo.detect_drift(&data).expect("solo").drift_detected
        );
    }
    for round in 0..4 {
        let data = batch(100_000 + round * 60, 60, 1);
        assert_eq!(
            weighted
                .detect_drift(&data)
                .expect("weighted")
                .drift_detected,
            solo.detect_drift(&data).expect("solo").drift_detected
        );
    }
}

// ---------------------------------------------------------------------------
// contract shared with `EnhancedDriftDetector`
// ---------------------------------------------------------------------------

/// All three detectors are supervised: an unlabelled batch is an honest error,
/// never a silent training pass against an invented label.
#[test]
fn unlabelled_batches_are_an_honest_error() {
    let unlabelled: Vec<StreamingDataPoint<f64>> = (0..32)
        .map(|index| {
            let mut data_point = point(index, 0);
            data_point.target = None;
            data_point
        })
        .collect();

    let mut neural = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    let mut tree = DecisionTreeDriftDetector::<f64>::new(0.2).expect("detector");
    let mut ensemble = EnsembleDriftDetector::<f64>::new(0.2).expect("detector");

    assert!(neural.update_model(&unlabelled).is_err());
    assert!(tree.update_model(&unlabelled).is_err());
    assert!(ensemble.update_model(&unlabelled).is_err());
}

/// `reset_model` must return every detector to its pre-training state, so a
/// reset detector cannot inherit a stale baseline.
#[test]
fn reset_clears_learned_state() {
    let mut neural = NeuralNetworkDriftDetector::<f64>::new(0.2).expect("detector");
    neural.update_model(&batch(0, 200, 0)).expect("train");
    assert!(neural.current_error().is_some());
    neural.reset_model().expect("reset");
    assert!(neural.current_error().is_none());
    assert_eq!(neural.input_width(), 0);

    let mut tree = DecisionTreeDriftDetector::<f64>::new(0.2).expect("detector");
    tree.update_model(&batch(0, 200, 0)).expect("train");
    assert!(tree.is_fitted());
    tree.reset_model().expect("reset");
    assert!(!tree.is_fitted());
    assert!(tree.feature_importances().is_empty());
}
