// Behavioural regression tests for the confusion counters, the score-retention
// buffer and the full-curve ROC area.

use super::*;

/// Scores `values` and immediately labels each one, so every pair lands in the
/// retention buffer.
fn counters_from(pairs: &[(f64, bool)]) -> DetectionCounters {
    let mut counters = DetectionCounters::default();
    for (score, label) in pairs {
        counters.record_prediction(*score > 0.5, *score);
        counters.record_outcome(*score > 0.5, *label);
    }
    counters
}

/// F2: with no labelled scores at all the AUC is an honest error, never a
/// number.
#[test]
fn auc_is_an_honest_error_without_labelled_scores() {
    let counters = DetectionCounters::default();
    let error = counters.auc_roc().expect_err("no scores were retained");
    assert!(
        error.contains("at least two labelled scores of each class"),
        "the error must say what is missing: {error}"
    );
}

/// F2: one labelled score of a class pins a single segment of the curve, which
/// is not a measurement of ranking quality — still an honest error.
#[test]
fn auc_is_an_honest_error_with_a_single_example_of_a_class() {
    let counters = counters_from(&[(0.9, true), (0.1, false), (0.2, false), (0.3, false)]);
    assert_eq!(counters.retained_scores(), 4);
    assert!(
        counters.auc_roc().is_err(),
        "one positive example cannot support an ROC area"
    );
}

/// F2: perfectly separable scores give exactly 1.0; the reversed ranking gives
/// exactly 0.0. The old single-operating-point substitute could produce
/// neither.
#[test]
fn auc_is_exact_for_separable_and_reversed_rankings() {
    let separable = counters_from(&[
        (0.9, true),
        (0.8, true),
        (0.7, true),
        (0.3, false),
        (0.2, false),
        (0.1, false),
    ]);
    let area = separable.auc_roc().expect("separable AUC");
    assert!(
        (area - 1.0).abs() < 1e-12,
        "a perfectly separable ranking must score exactly 1.0, got {area}"
    );

    let reversed = counters_from(&[
        (0.9, false),
        (0.8, false),
        (0.7, false),
        (0.3, true),
        (0.2, true),
        (0.1, true),
    ]);
    let area = reversed.auc_roc().expect("reversed AUC");
    assert!(
        area.abs() < 1e-12,
        "a perfectly reversed ranking must score exactly 0.0, got {area}"
    );
}

/// F2: an interleaved ranking scores exactly one half — the value a coin flip
/// earns, computed rather than assumed.
#[test]
fn auc_is_one_half_for_an_interleaved_ranking() {
    let counters = counters_from(&[
        (0.9, true),
        (0.8, false),
        (0.7, true),
        (0.6, false),
        (0.5, true),
        (0.4, false),
    ]);
    let area = counters.auc_roc().expect("interleaved AUC");
    // Ranking T N T N T N: the concordant-pair count is 6 out of 9.
    assert!(
        (area - 6.0 / 9.0).abs() < 1e-12,
        "interleaved AUC {area} does not match the closed-form 6/9"
    );
}

/// F2: tied scores must be collapsed into a single operating point. Every score
/// identical means the detector ranks nothing, which is exactly 0.5 — stepping
/// through the ties one at a time would instead report whatever order they
/// happened to arrive in.
#[test]
fn tied_scores_are_grouped_into_one_threshold_step() {
    let all_tied = counters_from(&[
        (0.5, true),
        (0.5, false),
        (0.5, true),
        (0.5, false),
        (0.5, true),
        (0.5, false),
    ]);
    let area = all_tied.auc_roc().expect("tied AUC");
    assert!(
        (area - 0.5).abs() < 1e-12,
        "a detector that gives every point the same score must score exactly \
         0.5, got {area}"
    );

    // Order must not matter once ties are grouped.
    let reordered = counters_from(&[
        (0.5, true),
        (0.5, true),
        (0.5, true),
        (0.5, false),
        (0.5, false),
        (0.5, false),
    ]);
    assert!(
        (reordered.auc_roc().expect("tied AUC") - area).abs() < 1e-12,
        "the AUC changed with the arrival order of tied scores"
    );
}

/// F2: the retention buffer is bounded, and the oldest labelled score is
/// evicted first.
#[test]
fn the_score_buffer_is_bounded() {
    let mut counters = DetectionCounters::default();
    for index in 0..(MAX_RETAINED_SCORES + 500) {
        let score = (index % 100) as f64 / 100.0;
        counters.record_prediction(score > 0.5, score);
        counters.record_outcome(score > 0.5, index.is_multiple_of(2));
    }
    assert_eq!(counters.retained_scores(), MAX_RETAINED_SCORES);
    assert!(counters.auc_roc().is_ok());
}

/// A label with no preceding score cannot be placed on the curve, so it moves
/// the confusion matrix without inventing a retained score.
#[test]
fn an_outcome_without_a_score_does_not_enter_the_curve() {
    let mut counters = DetectionCounters::default();
    counters.record_outcome(true, true);
    counters.record_outcome(false, false);
    assert_eq!(counters.labelled(), 2);
    assert_eq!(counters.retained_scores(), 0);

    // One score can only be labelled once.
    counters.record_prediction(true, 0.9);
    counters.record_outcome(true, true);
    counters.record_outcome(true, true);
    assert_eq!(counters.retained_scores(), 1);
}

/// Non-finite scores cannot be placed on a threshold sweep, so they are not
/// retained — but the prediction is still counted.
#[test]
fn non_finite_scores_are_not_retained() {
    let mut counters = DetectionCounters::default();
    counters.record_prediction(true, f64::NAN);
    counters.record_outcome(true, true);
    counters.record_prediction(true, f64::INFINITY);
    counters.record_outcome(true, true);
    assert_eq!(counters.predictions, 2);
    assert_eq!(counters.retained_scores(), 0);
}

/// F2: `to_metrics` keeps reporting the four confusion-matrix statistics even
/// when the ROC area is unavailable — an absent AUC must not destroy the
/// metrics that *are* measured.
#[test]
fn metrics_survive_an_unavailable_auc() {
    let mut counters = DetectionCounters::default();
    counters.record_outcome(true, true);
    counters.record_outcome(false, false);
    let metrics = counters
        .to_metrics::<f64>("test".to_string(), Duration::ZERO, Duration::ZERO)
        .expect("metrics");
    assert!(metrics.auc_roc.is_none(), "no scores were retained");
    assert!((metrics.accuracy - 1.0).abs() < 1e-12);

    let separable = counters_from(&[(0.9, true), (0.8, true), (0.3, false), (0.2, false)]);
    let metrics = separable
        .to_metrics::<f64>("test".to_string(), Duration::ZERO, Duration::ZERO)
        .expect("metrics");
    let area = metrics.auc_roc.expect("a full-curve AUC");
    assert!((area - 1.0).abs() < 1e-12, "auc {area}");
}

/// The adaptive ensemble weights its members by balanced accuracy, so that
/// figure must be measured (and must be `None` before any ground truth).
#[test]
fn balanced_accuracy_is_measured_not_assumed() {
    let mut counters = DetectionCounters::default();
    assert!(counters.balanced_accuracy().is_none());

    // 3 TP, 1 FN => TPR 0.75; 1 FP, 3 TN => TNR 0.75.
    for _ in 0..3 {
        counters.record_outcome(true, true);
    }
    counters.record_outcome(false, true);
    counters.record_outcome(true, false);
    for _ in 0..3 {
        counters.record_outcome(false, false);
    }
    let balanced = counters.balanced_accuracy().expect("balanced accuracy");
    assert!(
        (balanced - 0.75).abs() < 1e-12,
        "balanced accuracy {balanced}"
    );

    // A detector that flags nothing on a rare-anomaly stream has high plain
    // accuracy but a balanced accuracy of one half.
    let mut lazy = DetectionCounters::default();
    for index in 0..100 {
        lazy.record_outcome(false, index.is_multiple_of(100));
    }
    let balanced = lazy.balanced_accuracy().expect("balanced accuracy");
    assert!(
        (balanced - 0.5).abs() < 1e-12,
        "a detector that flags nothing must not look good: {balanced}"
    );
}
