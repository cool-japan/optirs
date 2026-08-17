// Regression tests for CF1: config fields that had no reader anywhere in the
// crate, so setting them changed nothing.
//
// Each test names a value the pre-fix code provably could not produce.

use super::anomaly_detection::AnomalyDetector;
use super::buffering::AdaptiveBuffer;
use super::config::{
    AnomalyConfig, BufferConfig, CycleMode, CyclicalRateConfig, DriftConfig,
    ExperienceReplayConfig, LearningRateConfig, MetaLearningConfig, PerformanceConfig,
    ScaleFunction, StreamingConfig,
};
use super::drift_detection::{DriftSeverity, DriftTestResult, EnhancedDriftDetector};
use super::meta_learning::MetaLearner;
use super::optimizer::{AdaptiveLearningRateController, StreamingDataPoint};
use super::performance::{PerformanceSnapshot, PerformanceTracker};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn point(features: Vec<f64>) -> StreamingDataPoint<f64> {
    StreamingDataPoint {
        features: Array1::from_vec(features),
        target: None,
        timestamp: Instant::now(),
        source_id: None,
        quality_score: 1.0,
        metadata: HashMap::new(),
    }
}

fn snapshot(loss: f64) -> PerformanceSnapshot<f64> {
    PerformanceSnapshot {
        timestamp: Instant::now(),
        processing_duration: Duration::from_millis(1),
        loss,
        accuracy: None,
        convergence_rate: None,
        gradient_norm: None,
        parameter_update_magnitude: None,
        data_statistics: Default::default(),
        resource_usage: Default::default(),
        custom_metrics: HashMap::new(),
    }
}

// ---------------------------------------------------------------- BufferConfig

/// `BufferConfig::memory_limit_mb` had no reader, so a tiny memory budget still
/// let the buffer grow all the way to `max_size`.
#[test]
fn buffer_memory_limit_caps_the_target_size() {
    let config = StreamingConfig {
        buffer_config: BufferConfig {
            min_size: 1,
            max_size: 1_000_000,
            // One megabyte of budget against 4096 f64 features per sample: far
            // fewer than a million items fit.
            memory_limit_mb: 1,
            ..BufferConfig::default()
        },
        ..Default::default()
    };

    let mut buffer = AdaptiveBuffer::<f64>::new(&config).expect("buffer");
    // The per-item footprint is measured from real data, so feed one sample.
    buffer
        .add_batch(vec![point(vec![0.5; 4096])])
        .expect("add_batch");

    let capacity = buffer
        .memory_bounded_capacity_for_test()
        .expect("a footprint has been measured");
    assert!(
        capacity < 1_000_000,
        "a 1 MB budget must not admit a million 4096-wide samples, got {capacity}"
    );
    let bounded = buffer.bound_target_size_for_test(1_000_000);
    assert_eq!(
        bounded, capacity,
        "the memory budget must bind before max_size ({bounded} vs {capacity})"
    );
}

// ----------------------------------------------------------------- DriftConfig

/// `DriftConfig::warning_threshold` had no reader at all. A test statistic above
/// it must now escalate the reported severity above `Minor`, even when the
/// p-value is unremarkable.
#[test]
fn drift_warning_threshold_escalates_severity() {
    let config = StreamingConfig {
        drift_config: DriftConfig {
            warning_threshold: 0.5,
            drift_threshold: 1.0,
            ..DriftConfig::default()
        },
        ..Default::default()
    };
    let detector = EnhancedDriftDetector::<f64>::new(&config).expect("detector");

    // A large statistic with a deliberately unremarkable p-value: before the
    // fix this could only ever be classified `Minor`.
    let big = DriftTestResult {
        drift_detected: true,
        p_value: 0.4,
        test_statistic: 2.0,
        confidence: 0.1,
        metadata: HashMap::new(),
    };
    assert_eq!(
        detector.classify_drift_severity_for_test(&big),
        DriftSeverity::Major,
        "a statistic above drift_threshold must report Major"
    );

    let middling = DriftTestResult {
        test_statistic: 0.7,
        ..big.clone()
    };
    assert_eq!(
        detector.classify_drift_severity_for_test(&middling),
        DriftSeverity::Moderate,
        "a statistic between warning_threshold and drift_threshold must report Moderate"
    );

    let small = DriftTestResult {
        test_statistic: 0.1,
        ..big
    };
    assert_eq!(
        detector.classify_drift_severity_for_test(&small),
        DriftSeverity::Minor
    );
}

// ----------------------------------------------------------- PerformanceConfig

/// `PerformanceConfig::enable_tracking` had no reader: turning tracking off did
/// all of the tracking work anyway.
#[test]
fn disabling_performance_tracking_stops_ingestion() {
    let config = StreamingConfig {
        performance_config: PerformanceConfig {
            enable_tracking: false,
            ..PerformanceConfig::default()
        },
        ..Default::default()
    };
    let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");
    tracker.add_performance(snapshot(1.0)).expect("add");
    tracker.add_performance(snapshot(0.5)).expect("add");

    assert!(
        tracker.get_recent_performance(10).is_empty(),
        "enable_tracking = false must not record history"
    );
}

/// `PerformanceConfig::baseline_update_frequency` had no reader, so the baseline
/// was pinned to the very first measurement for the entire run.
#[test]
fn baseline_is_refreshed_on_the_configured_cadence() {
    let config = StreamingConfig {
        performance_config: PerformanceConfig {
            enable_tracking: true,
            baseline_update_frequency: 3,
            ..PerformanceConfig::default()
        },
        ..Default::default()
    };
    let mut tracker = PerformanceTracker::<f64>::new(&config).expect("tracker");

    tracker.add_performance(snapshot(10.0)).expect("add");
    let first = tracker
        .baseline_loss_for_test()
        .expect("baseline set on the first sample");
    assert!((first - 10.0).abs() < 1e-9);

    // Two more samples reach the cadence of 3 and re-base onto the newest loss.
    tracker.add_performance(snapshot(5.0)).expect("add");
    tracker.add_performance(snapshot(1.0)).expect("add");
    let refreshed = tracker.baseline_loss_for_test().expect("baseline");
    assert!(
        (refreshed - 1.0).abs() < 1e-9,
        "baseline was not refreshed after baseline_update_frequency samples: {refreshed}"
    );
}

// ---------------------------------------------------------------- AnomalyConfig

/// `AnomalyConfig::contamination_rate` and `enable_adaptive_threshold` had no
/// readers, so thresholds never tracked the stream's actual score distribution.
#[test]
fn contamination_rate_calibrates_detector_thresholds() {
    let config = StreamingConfig {
        anomaly_config: AnomalyConfig {
            enable_detection: true,
            enable_adaptive_threshold: true,
            contamination_rate: 0.2,
            window_size: 20,
            ..AnomalyConfig::default()
        },
        ..Default::default()
    };
    let mut detector = AnomalyDetector::<f64>::new(&config).expect("detector");

    // Enough points to reach one recalibration window.
    for step in 0..40 {
        let value = (step % 10) as f64;
        detector
            .detect_anomaly(&point(vec![value, value * 0.5]))
            .expect("detect_anomaly");
    }

    let calibrated: Vec<f64> = detector.calibrated_threshold_names_for_test();
    assert!(
        !calibrated.is_empty(),
        "no detector threshold was calibrated from contamination_rate"
    );
}

/// With adaptive thresholding switched off, no calibration may happen at all.
#[test]
fn disabling_adaptive_threshold_freezes_calibration() {
    let config = StreamingConfig {
        anomaly_config: AnomalyConfig {
            enable_detection: true,
            enable_adaptive_threshold: false,
            contamination_rate: 0.2,
            window_size: 5,
            ..AnomalyConfig::default()
        },
        ..Default::default()
    };
    let mut detector = AnomalyDetector::<f64>::new(&config).expect("detector");
    for step in 0..40 {
        let value = (step % 10) as f64;
        detector
            .detect_anomaly(&point(vec![value, value * 0.5]))
            .expect("detect_anomaly");
    }
    assert!(
        detector.calibrated_threshold_names_for_test().is_empty(),
        "enable_adaptive_threshold = false must leave thresholds untouched"
    );
}

// ----------------------------------------------------------- CyclicalRateConfig

/// `enable_cyclical_rates` and every field of `CyclicalRateConfig` had no reader
/// anywhere, so a configured cyclical schedule did nothing.
#[test]
fn cyclical_schedule_sweeps_between_its_bounds() {
    let config = StreamingConfig {
        learning_rate_config: LearningRateConfig {
            min_rate: 0.0,
            max_rate: 1.0,
            enable_cyclical_rates: true,
            cycle_config: CyclicalRateConfig {
                base_rate: 0.1,
                max_rate: 0.5,
                cycle_length: 8,
                cycle_mode: CycleMode::Triangular,
                scale_function: ScaleFunction::Linear,
            },
            ..LearningRateConfig::default()
        },
        ..Default::default()
    };

    let controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
    assert!(controller.is_cyclical());

    // step_size = cycle_length / 2 = 4: the rate peaks at iteration 4 and
    // returns to the base at 0 and 8.
    let rates: Vec<f64> = (0..=8).map(|i| controller.rate_at_for_test(i)).collect();
    assert!(
        (rates[0] - 0.1).abs() < 1e-9,
        "cycle must start at base_rate, got {}",
        rates[0]
    );
    assert!(
        (rates[4] - 0.5).abs() < 1e-9,
        "cycle must peak at the cycle max_rate, got {}",
        rates[4]
    );
    assert!(
        (rates[8] - 0.1).abs() < 1e-9,
        "cycle must return to base_rate, got {}",
        rates[8]
    );
    assert!(
        rates.iter().all(|r| (0.1..=0.5).contains(r)),
        "cyclical rate left its configured bounds: {rates:?}"
    );
}

/// `Triangular2` halves the amplitude each cycle; a schedule that ignored
/// `cycle_mode` would peak at the same value in every cycle.
#[test]
fn triangular2_halves_the_amplitude_each_cycle() {
    let config = StreamingConfig {
        learning_rate_config: LearningRateConfig {
            min_rate: 0.0,
            max_rate: 1.0,
            enable_cyclical_rates: true,
            cycle_config: CyclicalRateConfig {
                base_rate: 0.0,
                max_rate: 1.0,
                cycle_length: 8,
                cycle_mode: CycleMode::Triangular2,
                ..CyclicalRateConfig::default()
            },
            ..LearningRateConfig::default()
        },
        ..Default::default()
    };

    let controller = AdaptiveLearningRateController::<f64>::new(&config).expect("controller");
    let first_peak = controller.rate_at_for_test(4);
    let second_peak = controller.rate_at_for_test(12);
    assert!((first_peak - 1.0).abs() < 1e-9, "first peak {first_peak}");
    assert!(
        (second_peak - 0.5).abs() < 1e-9,
        "Triangular2 must halve the second peak, got {second_peak}"
    );
}

/// A `Custom` cycle mode or scale function has no registered implementation, so
/// it must be refused rather than silently treated as `Triangular`/`Linear`.
#[test]
fn custom_cycle_settings_are_refused() {
    let cyclical = |cycle_config: CyclicalRateConfig| StreamingConfig {
        learning_rate_config: LearningRateConfig {
            enable_cyclical_rates: true,
            cycle_config,
            ..LearningRateConfig::default()
        },
        ..Default::default()
    };

    let custom_scale = cyclical(CyclicalRateConfig {
        scale_function: ScaleFunction::Custom("mystery".to_string()),
        ..CyclicalRateConfig::default()
    });
    assert!(AdaptiveLearningRateController::<f64>::new(&custom_scale).is_err());

    let custom_mode = cyclical(CyclicalRateConfig {
        cycle_mode: CycleMode::Custom("mystery".to_string()),
        ..CyclicalRateConfig::default()
    });
    assert!(AdaptiveLearningRateController::<f64>::new(&custom_mode).is_err());
}

/// `ExponentialRange` needs its decay factor from the scale function; without
/// one there is no honest value to use.
#[test]
fn exponential_range_requires_an_exponential_scale_function() {
    let cyclical = |scale_function: ScaleFunction| StreamingConfig {
        learning_rate_config: LearningRateConfig {
            enable_cyclical_rates: true,
            cycle_config: CyclicalRateConfig {
                cycle_mode: CycleMode::ExponentialRange,
                scale_function,
                ..CyclicalRateConfig::default()
            },
            ..LearningRateConfig::default()
        },
        ..Default::default()
    };

    let without_factor = cyclical(ScaleFunction::Linear);
    assert!(AdaptiveLearningRateController::<f64>::new(&without_factor).is_err());

    let with_factor = cyclical(ScaleFunction::Exponential { factor: 0.9 });
    assert!(AdaptiveLearningRateController::<f64>::new(&with_factor).is_ok());
}

// ------------------------------------------------------- MetaLearningConfig

/// `MetaLearningConfig::experience_buffer_size` had no reader: the buffer was
/// hardcoded to 10 000 entries regardless.
#[test]
fn experience_buffer_size_bounds_the_buffer() {
    let config = StreamingConfig {
        meta_learning_config: MetaLearningConfig {
            experience_buffer_size: 4,
            // Keep the training cadence out of the way of this test.
            update_frequency: 1_000,
            replay_config: ExperienceReplayConfig {
                replay_frequency: 1_000,
                ..ExperienceReplayConfig::default()
            },
            ..MetaLearningConfig::default()
        },
        ..Default::default()
    };

    let mut learner = MetaLearner::<f64>::new(&config).expect("learner");
    for _ in 0..20 {
        learner
            .record_probe_experience_for_test(1.0)
            .expect("record");
    }
    assert_eq!(
        learner.experience_count_for_test(),
        4,
        "experience_buffer_size did not bound the buffer"
    );
}

/// `enable_transfer_learning` had no reader, so transfer learning was neither
/// on nor off. Registering a source must now be refused when it is disabled.
#[test]
fn transfer_learning_flag_gates_source_registration() {
    let disabled = StreamingConfig {
        meta_learning_config: MetaLearningConfig {
            enable_transfer_learning: false,
            ..MetaLearningConfig::default()
        },
        ..Default::default()
    };
    let mut learner = MetaLearner::<f64>::new(&disabled).expect("learner");
    assert!(
        learner
            .register_transfer_source("src".to_string(), Vec::new(), vec![1.0])
            .is_err(),
        "a disabled flag must refuse the registration, not accept and ignore it"
    );
    assert!(learner.transfer_metrics().is_none());

    let enabled = StreamingConfig {
        meta_learning_config: MetaLearningConfig {
            enable_transfer_learning: true,
            ..MetaLearningConfig::default()
        },
        ..Default::default()
    };
    let mut learner = MetaLearner::<f64>::new(&enabled).expect("learner");
    learner
        .register_transfer_source("src".to_string(), Vec::new(), vec![1.0])
        .expect("registration must succeed when enabled");
    assert!(learner.transfer_metrics().is_some());
}
