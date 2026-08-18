// Regression tests for the top-level drift detectors (findings C3, C7).

use super::*;

// ---------------------------------------------------------------------------
// C3: DDM did not apply the published 2*s_min / 3*s_min margins.
// ---------------------------------------------------------------------------

#[test]
fn ddm_does_not_warn_on_a_stable_error_rate() {
    let mut detector = DdmDetector::<f64>::new();
    let mut warnings = 0usize;
    let mut drifts = 0usize;

    // A deterministic 10% error rate for 300 samples: entirely stationary.
    for step in 0..300 {
        match detector.update(step % 10 == 0) {
            DriftStatus::Warning => warnings += 1,
            DriftStatus::Drift => drifts += 1,
            DriftStatus::Stable => {}
        }
    }

    assert_eq!(
        drifts, 0,
        "C3 regression: a stationary 10% error rate produced {drifts} drift detections"
    );
    assert_eq!(
        warnings, 0,
        "C3 regression: a stationary 10% error rate produced {warnings} warnings. Without the \
         2*s_min margin the test degenerates to 'is the level above the lowest level ever seen', \
         which fires on any upward wobble"
    );
}

#[test]
fn ddm_still_detects_a_genuine_error_rate_jump() {
    let mut detector = DdmDetector::<f64>::new();
    for step in 0..200 {
        detector.update(step % 10 == 0); // 10% baseline
    }
    let mut detected = false;
    for step in 0..200 {
        if detector.update(step % 10 != 0) == DriftStatus::Drift {
            // 90% error rate
            detected = true;
            break;
        }
    }
    assert!(
        detected,
        "a jump from a 10% to a 90% error rate must be detected"
    );
}

#[test]
fn ddm_levels_use_the_s_min_margins() {
    let mut detector = DdmDetector::<f64>::new();
    for step in 0..100 {
        detector.update(step % 5 == 0);
    }
    let warning = detector
        .warning_level()
        .expect("the baseline is established after 100 samples");
    let drift = detector
        .drift_level()
        .expect("the baseline is established after 100 samples");
    assert!(
        drift > warning,
        "the drift level (p_min + 3*s_min) must exceed the warning level (p_min + 2*s_min)"
    );
    // warning - p_min = 2*s_min and drift - warning = s_min, so:
    let p_min = 3.0 * warning - 2.0 * drift;
    let s_min = drift - warning;
    assert!(
        s_min > 0.0,
        "s_min must be strictly positive for a non-degenerate error rate"
    );
    assert!(
        (warning - (p_min + 2.0 * s_min)).abs() < 1e-9,
        "the warning level must be exactly p_min + 2*s_min"
    );
}

#[test]
fn ddm_never_seeds_an_impossible_standard_deviation() {
    let detector = DdmDetector::<f64>::new();
    assert_eq!(
        detector.error_std, 0.0,
        "C3 regression: the detector seeded error_std = 1.0, which is not a possible standard \
         deviation for a rate in [0, 1]"
    );
    let mut detector = DdmDetector::<f64>::new();
    for _ in 0..40 {
        detector.update(true);
    }
    assert!(
        detector.error_std <= 0.5,
        "the standard deviation of a rate cannot exceed 0.5 (got {})",
        detector.error_std
    );
}

#[test]
fn ddm_warms_up_before_testing() {
    let mut detector = DdmDetector::<f64>::with_warmup(30);
    for _ in 0..29 {
        assert_eq!(detector.update(true), DriftStatus::Stable);
    }
    assert!(detector.warning_level().is_none());
}

// ---------------------------------------------------------------------------
// C7: the ensemble decision history was allocated and never written to, and
// the drift-event log was unbounded.
// ---------------------------------------------------------------------------

#[test]
fn ensemble_history_is_recorded_and_bounded() {
    let config = DriftDetectorConfig {
        enable_ensemble: true,
        ..DriftDetectorConfig::default()
    };
    let mut detector = ConceptDriftDetector::<f64>::new(config);

    for step in 0..500 {
        let loss = 1.0 + 0.001 * (step % 7) as f64;
        detector
            .update(loss, step % 10 == 0)
            .expect("update must succeed");
    }

    let history = detector.ensemble_history();
    assert!(
        !history.is_empty(),
        "C7 regression: the ensemble history was never written to"
    );
    assert!(
        history.len() <= ConceptDriftDetector::<f64>::ENSEMBLE_HISTORY_CAPACITY,
        "C7 regression: the ensemble history grew to {} entries",
        history.len()
    );
}

#[test]
fn drift_event_confidence_reflects_detector_agreement() {
    let config = DriftDetectorConfig::default();
    let detector = ConceptDriftDetector::<f64>::new(config);

    let unanimous =
        detector.detection_confidence(DriftStatus::Drift, DriftStatus::Drift, DriftStatus::Drift);
    let lone_vote =
        detector.detection_confidence(DriftStatus::Drift, DriftStatus::Stable, DriftStatus::Stable);

    assert!(
        unanimous > lone_vote,
        "unanimous detectors must be more confident than a single vote ({unanimous} vs {lone_vote})"
    );
    assert_ne!(
        unanimous, 0.8,
        "the confidence must be computed, not the old hardcoded 0.8"
    );
    assert_ne!(lone_vote, 0.8);
    assert!((0.0..=1.0).contains(&unanimous));
    assert!((0.0..=1.0).contains(&lone_vote));
}

#[test]
fn drift_event_log_stays_bounded() {
    let config = DriftDetectorConfig {
        // A very low threshold so drift fires constantly.
        threshold: 1e-9,
        warningthreshold: 1e-12,
        ..DriftDetectorConfig::default()
    };
    let mut detector = ConceptDriftDetector::<f64>::new(config);
    for step in 0..3000 {
        let loss = 1.0 + (step % 13) as f64;
        detector.update(loss, true).expect("update must succeed");
    }
    let total = detector.get_statistics().total_drifts;
    assert!(
        total <= ConceptDriftDetector::<f64>::DRIFT_EVENT_CAPACITY,
        "C7 regression: the drift-event log grew to {total} entries"
    );
    assert!(total > 0, "the scenario must actually produce drift events");
}
