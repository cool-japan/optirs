// Regression tests for the real drift detectors.
//
// Split out of `drift_tests.rs` so that file stays under the 2000-line limit; it is
// wired back in as a `#[cfg(test)]` child module, so `super::*` still refers
// to the code under test.

use super::*;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap as StdHashMap;
use std::time::Instant;

fn point(features: Vec<f64>, target: Option<f64>) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(features),
        target: target.map(|t| Array1::from_vec(vec![t])),
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: StdHashMap::new(),
    }
}

/// Deterministic pseudo-noise so the tests never depend on an RNG.
fn wobble(index: usize) -> f64 {
    // A short irrational-ish cycle keeps the sequence non-repeating over
    // the sample sizes used here without pulling in a random source.
    ((index as f64) * 0.7548776662).fract() - 0.5
}

fn stationary_stream(n: usize) -> Vec<f64> {
    (0..n).map(|i| 10.0 + wobble(i)).collect()
}

fn shifted_stream(n: usize) -> Vec<f64> {
    (0..n).map(|i| 25.0 + wobble(i)).collect()
}

/// D1: `ADWINTest`, `DDMTest` and `PageHinkleyTest` were three copies of the
/// same computation — each took `|mean(reference) - mean(current)|` (Page-
/// Hinkley's first call being the signed version, identical for a positive
/// shift) — so on any input all three reported *numerically identical*
/// `test_statistic` values. This is the property that catches that: fed one
/// identical stream, the three must report genuinely different statistics,
/// because they now measure genuinely different things (a Hoeffding-bounded
/// sub-window mean gap, a binomial sigma count, and a cumulative-sum
/// excursion).
#[test]
fn the_three_mean_diff_clones_now_report_distinct_statistics() {
    let reference = stationary_stream(120);
    let current = shifted_stream(120);

    let mut adwin = AdwinTest::<f64>::new(0.05, 0.05).expect("adwin");
    let mut ddm = DdmTest::<f64>::new(0.05, 0.05).expect("ddm");
    let mut page_hinkley = PageHinkleyTest::<f64>::new(0.05, 0.05).expect("ph");

    let adwin_statistic = adwin
        .test_for_drift(&reference, &current)
        .expect("adwin")
        .test_statistic;
    let ddm_statistic = ddm
        .test_for_drift(&reference, &current)
        .expect("ddm")
        .test_statistic;
    let page_hinkley_statistic = page_hinkley
        .test_for_drift(&reference, &current)
        .expect("ph")
        .test_statistic;

    let trio = [
        ("adwin", adwin_statistic),
        ("ddm", ddm_statistic),
        ("page_hinkley", page_hinkley_statistic),
    ];
    for (i, (name_a, stat_a)) in trio.iter().enumerate() {
        for (name_b, stat_b) in trio.iter().skip(i + 1) {
            assert!(
                (stat_a - stat_b).abs() > 1e-9,
                "D1 regression: {name_a} and {name_b} produced the same \
                 test statistic ({stat_a} vs {stat_b}) — they are still the \
                 same mean-difference computation"
            );
        }
    }

    // The old code also produced exactly the mean difference; none of the
    // three should now equal it.
    let reference_mean: f64 = reference.iter().sum::<f64>() / reference.len() as f64;
    let current_mean: f64 = current.iter().sum::<f64>() / current.len() as f64;
    let mean_difference = (reference_mean - current_mean).abs();
    for (name, statistic) in trio {
        assert!(
            (statistic - mean_difference).abs() > 1e-9,
            "D1 regression: {name} still reports the raw mean difference \
             ({statistic} vs {mean_difference})"
        );
    }
}

/// D1: across all seven registered methods, the statistics must span several
/// genuinely different values rather than collapsing onto one.
#[test]
fn the_seven_statistical_methods_span_distinct_statistics() {
    let reference = stationary_stream(120);
    let current = shifted_stream(120);

    let statistics: Vec<f64> = vec![
        AdwinTest::<f64>::new(0.05, 0.05)
            .expect("adwin")
            .test_for_drift(&reference, &current)
            .expect("adwin")
            .test_statistic,
        DdmTest::<f64>::new(0.05, 0.05)
            .expect("ddm")
            .test_for_drift(&reference, &current)
            .expect("ddm")
            .test_statistic,
        EddmTest::<f64>::new(0.05, 0.05)
            .expect("eddm")
            .test_for_drift(&reference, &current)
            .expect("eddm")
            .test_statistic,
        PageHinkleyTest::<f64>::new(0.05, 0.05)
            .expect("ph")
            .test_for_drift(&reference, &current)
            .expect("ph")
            .test_statistic,
        CusumTest::<f64>::new(0.05, 0.05)
            .expect("cusum")
            .test_for_drift(&reference, &current)
            .expect("cusum")
            .test_statistic,
        KsTest::<f64>::new(0.05, 0.05)
            .expect("ks")
            .test_for_drift(&reference, &current)
            .expect("ks")
            .test_statistic,
        MannWhitneyUTest::<f64>::new(0.05, 0.05)
            .expect("mwu")
            .test_for_drift(&reference, &current)
            .expect("mwu")
            .test_statistic,
    ];

    let mut distinct = 0usize;
    for (i, a) in statistics.iter().enumerate() {
        if statistics.iter().take(i).all(|b| (a - b).abs() > 1e-9) {
            distinct += 1;
        }
    }
    assert!(
        distinct >= 5,
        "D1 regression: only {distinct} distinct statistics across seven \
         methods ({statistics:?}) — the detectors are still largely copies \
         of one another"
    );
}

/// D1: p-values must be real, not the fabricated `0.01 / 0.5` (ADWIN),
/// `0.02 / 0.6` (DDM) and `0.015 / 0.7` (Page-Hinkley) literals the old
/// code returned. A real p-value varies continuously with the data and is
/// monotonically smaller for a larger effect.
#[test]
fn ks_p_value_is_monotone_in_effect_size() {
    let reference = stationary_stream(200);

    let mild: Vec<f64> = (0..200).map(|i| 10.3 + wobble(i)).collect();
    let severe: Vec<f64> = (0..200).map(|i| 40.0 + wobble(i)).collect();

    let mut ks = KsTest::<f64>::new(0.05, 0.05).expect("ks");
    let mild_result = ks.test_for_drift(&reference, &mild).expect("mild");
    let severe_result = ks.test_for_drift(&reference, &severe).expect("severe");

    assert!(
        severe_result.p_value < mild_result.p_value,
        "a larger distribution shift must give a smaller p-value \
         (mild={}, severe={})",
        mild_result.p_value,
        severe_result.p_value
    );
    // The old code could only ever return one of a handful of literals.
    assert!(
        (mild_result.p_value - 0.01).abs() > 1e-12 && (mild_result.p_value - 0.5).abs() > 1e-12,
        "p-value {} looks like a hard-coded literal",
        mild_result.p_value
    );
}

/// A stationary stream must not be flagged, and a genuinely shifted one
/// must be — both for the same detector, with the same configuration.
#[test]
fn ks_separates_stationary_from_shifted_streams() {
    let reference = stationary_stream(200);
    let mut ks = KsTest::<f64>::new(0.01, 0.01).expect("ks");

    let quiet = ks
        .test_for_drift(&reference, &stationary_stream(200))
        .expect("quiet");
    assert!(
        !quiet.drift_detected,
        "KS fired on an identical stationary stream (p={})",
        quiet.p_value
    );

    let loud = ks
        .test_for_drift(&reference, &shifted_stream(200))
        .expect("loud");
    assert!(
        loud.drift_detected,
        "KS failed to flag a 15-sigma mean shift (p={})",
        loud.p_value
    );
}

/// KS is a genuine distribution test: it must catch a variance-only change
/// that leaves the mean exactly where it was. Every mean-difference
/// detector — which is all five of the old ones were — is blind to this.
#[test]
fn ks_detects_a_variance_change_with_an_unchanged_mean() {
    let reference: Vec<f64> = (0..300).map(|i| 10.0 + wobble(i)).collect();
    // Same mean, ten times the spread.
    let current: Vec<f64> = (0..300).map(|i| 10.0 + 10.0 * wobble(i)).collect();

    let reference_mean: f64 = reference.iter().sum::<f64>() / reference.len() as f64;
    let current_mean: f64 = current.iter().sum::<f64>() / current.len() as f64;
    assert!(
        (reference_mean - current_mean).abs() < 0.05,
        "test setup: means must match (got {reference_mean} vs {current_mean})"
    );

    let mut ks = KsTest::<f64>::new(0.05, 0.05).expect("ks");
    let result = ks.test_for_drift(&reference, &current).expect("ks");
    assert!(
        result.drift_detected,
        "KS must catch a pure variance change (D={}, p={})",
        result.test_statistic, result.p_value
    );
}

/// D3: an empty sample must be an honest error, not a silent division by
/// zero producing `NaN`/`inf` (which is what `sum / A::from(0)` did).
#[test]
fn empty_samples_are_rejected_rather_than_producing_nan() {
    let reference = stationary_stream(50);
    let empty: Vec<f64> = Vec::new();

    let mut ks = KsTest::<f64>::new(0.05, 0.05).expect("ks");
    assert!(ks.test_for_drift(&reference, &empty).is_err());
    assert!(ks.test_for_drift(&empty, &reference).is_err());

    let mut adwin = AdwinTest::<f64>::new(0.05, 0.05).expect("adwin");
    assert!(adwin.test_for_drift(&reference, &empty).is_err());

    let mut mwu = MannWhitneyUTest::<f64>::new(0.05, 0.05).expect("mwu");
    assert!(mwu.test_for_drift(&empty, &empty).is_err());

    let mut cusum = CusumTest::<f64>::new(0.05, 0.05).expect("cusum");
    assert!(cusum.test_for_drift(&reference, &empty).is_err());
}

/// ADWIN's `delta` must actually influence the verdict. The old
/// implementation ignored it entirely (it compared a mean difference
/// against `sensitivity` used as a raw magnitude threshold).
#[test]
fn adwin_delta_changes_the_hoeffding_cut() {
    let reference = stationary_stream(10);
    // A modest step, sized so the two confidence levels disagree.
    let mut stream: Vec<f64> = (0..60).map(|i| 10.0 + wobble(i)).collect();
    stream.extend((0..60).map(|i| 10.9 + wobble(i)));

    let mut lenient = AdwinTest::<f64>::new(0.5, 1e-12).expect("lenient");
    let mut strict = AdwinTest::<f64>::new(1e-9, 1e-12).expect("strict");

    let lenient_result = lenient
        .test_for_drift(&reference, &stream)
        .expect("lenient");
    let strict_result = strict.test_for_drift(&reference, &stream).expect("strict");

    // A stricter delta yields a wider cut, hence a larger epsilon.
    let lenient_eps = lenient_result
        .metadata
        .get("epsilon_cut")
        .copied()
        .expect("epsilon_cut recorded");
    let strict_eps = strict_result
        .metadata
        .get("epsilon_cut")
        .copied()
        .expect("epsilon_cut recorded");
    assert!(
        strict_eps > lenient_eps,
        "a smaller delta must widen the Hoeffding cut ({strict_eps} vs {lenient_eps})"
    );
    assert!(
        lenient_result.drift_detected || !strict_result.drift_detected,
        "the strict detector must not fire where the lenient one does not"
    );
}

/// Page-Hinkley must use its own running mean, and must be symmetric:
/// a downward shift is drift too.
#[test]
fn page_hinkley_catches_shifts_in_both_directions() {
    let reference = stationary_stream(10);

    let mut upward = PageHinkleyTest::<f64>::new(0.05, 1e-12).expect("ph");
    upward
        .test_for_drift(&reference, &stationary_stream(200))
        .expect("warmup");
    let up = upward
        .test_for_drift(&reference, &shifted_stream(200))
        .expect("up");
    assert!(
        up.drift_detected,
        "upward shift missed (PH={})",
        up.test_statistic
    );

    let mut downward = PageHinkleyTest::<f64>::new(0.05, 1e-12).expect("ph");
    downward
        .test_for_drift(&reference, &stationary_stream(200))
        .expect("warmup");
    let down = downward
        .test_for_drift(
            &reference,
            &(0..200).map(|i| -25.0 + wobble(i)).collect::<Vec<_>>(),
        )
        .expect("down");
    assert!(
        down.drift_detected,
        "downward shift missed (PH={})",
        down.test_statistic
    );
}

/// Page-Hinkley must not fire on a stationary stream whose level is far
/// from any hard-coded baseline.
#[test]
fn page_hinkley_is_quiet_on_a_stationary_stream_away_from_zero() {
    let reference = stationary_stream(10);
    let mut detector = PageHinkleyTest::<f64>::new(0.05, 1e-12).expect("ph");

    let mut fired = 0usize;
    for _ in 0..20 {
        let batch: Vec<f64> = (0..50).map(|i| 500.0 + wobble(i)).collect();
        if detector
            .test_for_drift(&reference, &batch)
            .expect("ph")
            .drift_detected
        {
            fired += 1;
        }
    }
    assert_eq!(
        fired, 0,
        "Page-Hinkley fired {fired} times on a stationary stream at level 500"
    );
}

/// Wasserstein-1 must equal the true transport cost of a pure shift, not a
/// rescaled mean difference.
#[test]
fn wasserstein_comparator_measures_real_transport_cost() {
    let comparator = WassersteinComparator::<f64>::new(0.5).expect("w1");
    let reference: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let current: Vec<f64> = reference.iter().map(|v| v + 3.0).collect();

    let comparison = comparator
        .compare_distributions(&reference, &current)
        .expect("compare");
    assert!(
        (comparison.distance - 3.0).abs() < 1e-6,
        "W1 of a +3 shift must be 3, got {}",
        comparison.distance
    );
    assert!(comparison.drift_detected);
}

/// The three histogram divergences must be genuinely different measures on
/// the same data (they were all the same mean difference before).
#[test]
fn histogram_divergences_are_distinct_measures() {
    let reference: Vec<f64> = (0..200).map(|i| 10.0 + wobble(i)).collect();
    let current: Vec<f64> = (0..200).map(|i| 12.0 + 2.0 * wobble(i)).collect();

    let kl = HistogramComparator::<f64>::new(HistogramDivergence::KullbackLeibler, 0.1)
        .expect("kl")
        .compare_distributions(&reference, &current)
        .expect("kl compare");
    let js = HistogramComparator::<f64>::new(HistogramDivergence::JensenShannon, 0.1)
        .expect("js")
        .compare_distributions(&reference, &current)
        .expect("js compare");
    let hellinger = HistogramComparator::<f64>::new(HistogramDivergence::Hellinger, 0.1)
        .expect("hellinger")
        .compare_distributions(&reference, &current)
        .expect("hellinger compare");

    assert!((kl.distance - js.distance).abs() > 1e-9);
    assert!((js.distance - hellinger.distance).abs() > 1e-9);
    assert!((kl.distance - hellinger.distance).abs() > 1e-9);

    // JS is bounded by ln 2 and Hellinger by 1 — a rescaled mean
    // difference would not respect either bound.
    assert!(js.distance <= std::f64::consts::LN_2 + 1e-12);
    assert!(hellinger.distance <= 1.0 + 1e-12);
}

/// Identical samples must produce zero divergence and no drift.
#[test]
fn histogram_divergence_is_zero_for_identical_samples() {
    let sample: Vec<f64> = (0..200).map(|i| 5.0 + wobble(i)).collect();
    for divergence in [
        HistogramDivergence::KullbackLeibler,
        HistogramDivergence::JensenShannon,
        HistogramDivergence::Hellinger,
    ] {
        let comparison = HistogramComparator::<f64>::new(divergence, 0.1)
            .expect("comparator")
            .compare_distributions(&sample, &sample)
            .expect("compare");
        // Hellinger takes a square root, which amplifies the residual
        // floating-point error in `1 - sum(sqrt(p*q))`, so the tolerance is
        // set for the worst of the three rather than the best.
        assert!(
            comparison.distance.abs() < 1e-6,
            "{:?} on identical samples gave {}",
            divergence,
            comparison.distance
        );
        assert!(!comparison.drift_detected);
    }
}

/// D2: `LinearModelDetector` was dead code — `model_performance` and
/// `baseline_performance` were both initialised to zero and never written,
/// so `performance_degradation` was permanently `0` and `detect_drift`
/// could never fire. Now the model genuinely learns, and a relationship
/// change genuinely degrades it.
#[test]
fn linear_model_detector_learns_and_reports_real_degradation() {
    let mut detector = LinearModelDetector::<f64>::new(0.5).expect("detector");

    // Train on y = 2x on a bounded feature range.
    let clean: Vec<StreamingDataPoint<f64>> = (0..400)
        .map(|i| {
            let x = ((i % 20) as f64) / 20.0;
            point(vec![x], Some(2.0 * x))
        })
        .collect();
    let baseline_result = detector.detect_drift(&clean).expect("baseline");

    assert!(
        detector.baseline_error().is_some(),
        "D2 regression: baseline error was never established"
    );
    let learned_error = detector.current_error().expect("current error");
    assert!(
        learned_error < 0.05,
        "the model did not actually learn y = 2x (mse={learned_error})"
    );
    assert!(
        !baseline_result.drift_detected,
        "no drift should be reported while the model is healthy"
    );

    // Flip the relationship: y = -5x + 7. The learned model is now wrong.
    let broken: Vec<StreamingDataPoint<f64>> = (0..40)
        .map(|i| {
            let x = ((i % 20) as f64) / 20.0;
            point(vec![x], Some(-5.0 * x + 7.0))
        })
        .collect();
    let drift_result = detector.detect_drift(&broken).expect("drift");

    assert!(
        drift_result.performance_degradation > 0.0,
        "D2 regression: performance_degradation is still {} (was permanently 0)",
        drift_result.performance_degradation
    );
    assert!(
        drift_result.drift_detected,
        "a reversed target relationship must be reported as model drift \
         (degradation={})",
        drift_result.performance_degradation
    );
    assert!(
        !drift_result.feature_importance_changes.is_empty(),
        "feature importance changes must be reported, not an empty Vec"
    );
}

/// An unlabelled batch cannot train a supervised model — that must be an
/// honest error, not a silent no-op that leaves the detector claiming
/// health.
#[test]
fn linear_model_detector_rejects_unlabelled_data() {
    let mut detector = LinearModelDetector::<f64>::new(0.05).expect("detector");
    let unlabelled: Vec<StreamingDataPoint<f64>> =
        (0..10).map(|i| point(vec![i as f64], None)).collect();
    assert!(detector.update_model(&unlabelled).is_err());
    assert!(detector.detect_drift(&unlabelled).is_err());
}
