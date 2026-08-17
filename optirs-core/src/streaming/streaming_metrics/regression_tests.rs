// Regression tests for the streaming metrics collector (findings M1-M4).

use super::*;
use std::time::Duration;

fn at(seconds: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
}

fn sample_at(seconds: u64, loss: f64, gradient: f64, millis: u64) -> MetricsSample<f64> {
    MetricsSample::new(
        at(seconds),
        loss,
        gradient,
        Duration::from_millis(millis),
        1024 * (seconds + 1),
    )
}

fn collector_without_compression() -> StreamingMetricsCollector<f64> {
    let mut collector = StreamingMetricsCollector::<f64>::new();
    collector.set_compression_config(CompressionConfig {
        enabled: false,
        algorithm: CompressionAlgorithm::None,
        target_ratio: 1.0,
        lossy_tolerance: 0.0,
    });
    collector
}

// ---------------------------------------------------------------------------
// M1: the four `update_*_metrics` methods were `Ok(())`, so every metric
// stayed at its `Default` value no matter what was recorded.
// ---------------------------------------------------------------------------

#[test]
fn performance_metrics_are_derived_from_recorded_samples() {
    let mut collector = collector_without_compression();
    for step in 0..5u64 {
        let loss = 1.0 - 0.1 * step as f64;
        collector
            .record_sample(sample_at(step * 2, loss, 0.5 + 0.01 * step as f64, 10))
            .expect("recording a sample must succeed");
    }

    let metrics = collector.get_current_metrics();
    assert!(
        (metrics.performance.accuracy.current_loss - 0.6).abs() < 1e-12,
        "M1 regression: current_loss stayed at its default (got {})",
        metrics.performance.accuracy.current_loss
    );
    assert!(
        metrics.performance.throughput.samples_per_second > 0.0,
        "M1 regression: throughput was never computed"
    );
    assert!(
        (metrics.performance.throughput.samples_per_second - 0.5).abs() < 1e-9,
        "samples every 2s must yield 0.5 samples/s, got {}",
        metrics.performance.throughput.samples_per_second
    );
    assert!(
        metrics.performance.accuracy.convergence_rate > 0.0,
        "M1 regression: a monotonically decreasing loss must give a positive convergence rate \
         (got {})",
        metrics.performance.accuracy.convergence_rate
    );
    assert!(
        metrics.performance.latency.end_to_end.mean >= Duration::from_millis(9),
        "M1 regression: latency statistics were never populated (got {:?})",
        metrics.performance.latency.end_to_end.mean
    );
    assert!(
        metrics.performance.stability.loss_variance > 0.0,
        "M1 regression: loss variance stayed at zero for a varying loss"
    );
    assert_eq!(
        metrics.performance.stability.divergence_probability, 0.0,
        "a monotonically decreasing loss never increases"
    );
    assert!(
        metrics.performance.efficiency.resource_utilization > 0.0,
        "M1 regression: resource utilization was never computed"
    );
    assert!(
        metrics
            .performance
            .efficiency
            .computational_efficiency
            .is_some(),
        "loss reduction per processing second is derivable and must be reported"
    );
    // Never measured, so it must stay absent rather than be invented.
    assert!(metrics.performance.efficiency.energy_efficiency.is_none());
    assert!(metrics.performance.latency.communication.is_none());
}

#[test]
fn optional_sub_latencies_appear_only_once_reported() {
    let mut collector = collector_without_compression();
    let mut sample = sample_at(0, 1.0, 0.5, 10);
    sample.communication_time = Some(Duration::from_millis(3));
    sample.queue_wait_time = Some(Duration::from_millis(1));
    collector.record_sample(sample).expect("record");

    let metrics = collector.get_current_metrics();
    let communication = metrics
        .performance
        .latency
        .communication
        .expect("a reported communication time must be summarised");
    assert_eq!(communication.max, Duration::from_millis(3));
    assert!(metrics.performance.latency.queue_wait_time.is_some());
    assert!(
        metrics.performance.latency.gradient_computation.is_none(),
        "an unreported sub-latency must stay absent"
    );
    assert!(metrics
        .performance
        .efficiency
        .communication_efficiency
        .is_some());
}

#[test]
fn resource_metrics_track_memory_and_stay_honest_about_the_rest() {
    let mut collector = collector_without_compression();
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");
    collector
        .record_sample(sample_at(1, 0.9, 0.5, 5))
        .expect("record");

    let metrics = collector.get_current_metrics();
    assert_eq!(metrics.resource.memory_usage.current_used, 1024 * 2);
    assert_eq!(metrics.resource.memory_usage.peak_usage, 1024 * 2);
    assert!(metrics.resource.memory_usage.efficiency.is_some());
    assert!(
        metrics.resource.cpu_utilization.is_none(),
        "CPU utilization cannot be measured by this collector and must stay None"
    );
    assert!(
        metrics.resource.memory_usage.gc_overhead.is_none(),
        "Rust has no garbage collector, so GC overhead must never be reported"
    );

    collector.record_resource_probe(ResourceProbe {
        cpu_utilization: Some(42.5),
        network_bandwidth_mbps: Some(3.5),
        ..ResourceProbe::default()
    });
    collector
        .record_sample(sample_at(2, 0.8, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert_eq!(metrics.resource.cpu_utilization, Some(42.5));
    assert_eq!(metrics.resource.network_bandwidth, Some(3.5));
}

#[test]
fn quality_metrics_flag_invalid_samples_and_outliers() {
    let mut collector = collector_without_compression();
    for step in 0..10u64 {
        collector
            .record_sample(sample_at(step, 1.0 + 0.001 * step as f64, 0.5, 5))
            .expect("record");
    }
    let metrics = collector.get_current_metrics();
    assert!(
        (metrics.quality.data_quality - 1.0).abs() < 1e-12,
        "ten valid samples must give a data quality of 1.0, got {}",
        metrics.quality.data_quality
    );

    // A far-out loss must produce a large z-score against the running stats.
    collector
        .record_sample(sample_at(10, 50.0, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert!(
        metrics.quality.anomaly_detection.anomaly_score > 3.0,
        "M1 regression: an extreme loss produced an anomaly score of {}",
        metrics.quality.anomaly_detection.anomaly_score
    );
    assert!(metrics.quality.anomaly_detection.anomaly_frequency > 0.0);
    assert!(
        metrics
            .quality
            .anomaly_detection
            .false_positive_rate
            .is_none(),
        "without labels the false-positive rate must stay None"
    );

    // A non-finite measurement must lower the data-quality ratio.
    collector
        .record_sample(sample_at(11, f64::NAN, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert!(
        metrics.quality.data_quality < 1.0,
        "a NaN loss must reduce data quality, got {}",
        metrics.quality.data_quality
    );
}

#[test]
fn drift_reports_become_real_drift_metrics() {
    let mut collector = collector_without_compression();
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert!(metrics.quality.concept_drift.drift_magnitude.is_none());

    collector.record_drift_event(0.42, 0.9, Duration::from_millis(120), Some(0.7));
    collector
        .record_sample(sample_at(10, 0.9, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert_eq!(metrics.quality.concept_drift.drift_magnitude, Some(0.42));
    assert_eq!(metrics.quality.concept_drift.drift_confidence, Some(0.9));
    assert_eq!(
        metrics.quality.concept_drift.detection_latency,
        Some(Duration::from_millis(120))
    );
    assert!(
        (metrics.quality.concept_drift.drift_frequency - 0.1).abs() < 1e-9,
        "one drift over a ten second span is 0.1/s, got {}",
        metrics.quality.concept_drift.drift_frequency
    );
}

#[test]
fn business_metrics_need_configuration_before_they_report_anything() {
    let mut collector = collector_without_compression();
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    assert!(
        metrics.business.slo_compliance.is_none(),
        "without SLO targets there is nothing to comply with"
    );
    assert!(metrics.business.availability.is_none());
    assert!(metrics.business.cost_metrics.total_cost.is_none());

    collector.set_slo_targets(SloTargets {
        max_processing_time: Some(Duration::from_millis(10)),
        ..SloTargets::default()
    });
    collector
        .record_sample(sample_at(1, 0.9, 0.5, 5))
        .expect("record"); // meets the SLO
    collector
        .record_sample(sample_at(2, 0.8, 0.5, 50))
        .expect("record"); // misses it
    let metrics = collector.get_current_metrics();
    assert_eq!(
        metrics.business.slo_compliance,
        Some(0.5),
        "one of two evaluated samples met the SLO"
    );

    collector.set_cost_model(CostModel {
        compute_cost_per_second: 2.0,
        memory_cost_per_gb_hour: 0.0,
        energy_cost_per_joule: 0.0,
        value_per_loss_unit: 10.0,
    });
    collector
        .record_sample(sample_at(3, 0.7, 0.5, 5))
        .expect("record");
    let metrics = collector.get_current_metrics();
    let compute = metrics
        .business
        .cost_metrics
        .computational_cost
        .expect("a configured cost model must yield a real compute cost");
    assert!(compute > 0.0, "compute cost must reflect processing time");
    assert!(metrics.business.business_value.is_some());
}

// ---------------------------------------------------------------------------
// M2: `get_aggregated` returned an empty Vec and `export_metrics` was a no-op.
// ---------------------------------------------------------------------------

#[test]
fn aggregation_buckets_snapshots_and_reduces_them() {
    let mut collector = collector_without_compression();
    // Two minute-buckets: 0..60 and 60..120.
    for seconds in [0u64, 10, 20, 60, 70] {
        let loss = 1.0 + seconds as f64;
        collector
            .record_sample(sample_at(seconds, loss, 0.5, 5))
            .expect("record");
    }

    let buckets = collector
        .get_aggregated_metrics(AggregationPeriod::Minute, at(0), at(120))
        .expect("aggregation must succeed");
    assert_eq!(
        buckets.len(),
        2,
        "M2 regression: aggregation returned {} buckets (the old code always returned none)",
        buckets.len()
    );
    assert_eq!(buckets[0].period_start, 0);
    assert_eq!(buckets[0].sample_count, 3);
    assert_eq!(buckets[1].period_start, 60);
    assert_eq!(buckets[1].sample_count, 2);

    let loss_series = buckets[0]
        .series
        .get("performance.accuracy.current_loss")
        .expect("the loss series must be aggregated");
    assert_eq!(loss_series.count, 3);
    let mean = loss_series.mean.expect("mean is a default function");
    assert!(
        (mean - (1.0 + 11.0 + 21.0) / 3.0).abs() < 1e-9,
        "unexpected bucket mean {mean}"
    );
    assert_eq!(loss_series.min, Some(1.0));
    assert_eq!(loss_series.max, Some(21.0));
    // Metrics that were never measured must be absent, not zero.
    assert!(!buckets[0].series.contains_key("resource.cpu_utilization"));
}

#[test]
fn aggregation_rejects_a_window_beyond_the_configured_maximum() {
    let mut collector = collector_without_compression();
    collector.set_aggregation_config(AggregationConfig {
        max_window: Duration::from_secs(60),
        ..AggregationConfig::default()
    });
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");
    assert!(collector
        .get_aggregated_metrics(AggregationPeriod::Minute, at(0), at(3600))
        .is_err());
}

#[test]
fn export_writes_real_files_for_supported_formats() {
    let directory = std::env::temp_dir().join(format!(
        "optirs_streaming_metrics_export_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    let mut collector = collector_without_compression();
    collector.set_export_config(ExportConfig {
        formats: vec![ExportFormat::Json, ExportFormat::Csv],
        destinations: vec![ExportDestination::File {
            path: directory.join("metrics").to_string_lossy().into_owned(),
        }],
        frequency: Duration::from_secs(0),
        batch_size: 100,
    });
    for seconds in 0..3u64 {
        collector
            .record_sample(sample_at(seconds, 1.0 - 0.1 * seconds as f64, 0.5, 5))
            .expect("record");
    }

    let written = collector.export_metrics_now().expect("export must succeed");
    assert_eq!(
        written.len(),
        2,
        "M2 regression: export wrote {} files (the old code wrote none)",
        written.len()
    );
    for path in &written {
        let content = std::fs::read_to_string(path).expect("exported file must be readable");
        assert!(
            content.contains("performance.accuracy.current_loss")
                || content.contains("performance_accuracy_current_loss"),
            "exported file {path:?} does not mention the loss metric"
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn unsupported_export_targets_are_reported_not_silently_skipped() {
    let mut collector = collector_without_compression();
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");

    collector.set_export_config(ExportConfig {
        formats: vec![ExportFormat::Parquet],
        destinations: vec![ExportDestination::File {
            path: std::env::temp_dir()
                .join("optirs_unused")
                .to_string_lossy()
                .into_owned(),
        }],
        frequency: Duration::from_secs(0),
        batch_size: 10,
    });
    assert!(
        collector.export_metrics_now().is_err(),
        "Parquet export is not implemented and must report that"
    );

    collector.set_export_config(ExportConfig {
        formats: vec![ExportFormat::Json],
        destinations: vec![ExportDestination::Http {
            endpoint: "http://example.invalid".to_string(),
            headers: HashMap::new(),
        }],
        frequency: Duration::from_secs(0),
        batch_size: 10,
    });
    assert!(
        collector.export_metrics_now().is_err(),
        "an HTTP destination cannot be reached from this crate and must report that"
    );
}

#[test]
fn byte_level_compression_request_is_reported_instead_of_ignored() {
    let mut collector = StreamingMetricsCollector::<f64>::new();
    collector.set_compression_config(CompressionConfig {
        enabled: true,
        algorithm: CompressionAlgorithm::Zstd,
        target_ratio: 1.0,
        lossy_tolerance: 0.0,
    });
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");
    assert!(
        collector.export_metrics_now().is_err(),
        "a byte codec this crate cannot provide must be reported, not silently ignored"
    );
}

#[test]
fn export_frequency_throttles_repeat_calls() {
    let directory = std::env::temp_dir().join(format!(
        "optirs_streaming_metrics_throttle_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    let mut collector = collector_without_compression();
    collector.set_export_config(ExportConfig {
        formats: vec![ExportFormat::Json],
        destinations: vec![ExportDestination::File {
            path: directory.join("metrics").to_string_lossy().into_owned(),
        }],
        frequency: Duration::from_secs(3600),
        batch_size: 10,
    });
    collector
        .record_sample(sample_at(0, 1.0, 0.5, 5))
        .expect("record");

    assert_eq!(collector.export_metrics().expect("first export").len(), 1);
    assert!(
        collector
            .export_metrics()
            .expect("second export")
            .is_empty(),
        "a second export inside the configured interval must write nothing"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// ---------------------------------------------------------------------------
// M1: `AlertSystem::evaluate_rules` was `Ok(())`, so rules never fired.
// ---------------------------------------------------------------------------

#[test]
fn threshold_rules_fire_and_then_resolve() {
    let mut collector = collector_without_compression();
    collector
        .add_alert_rule(AlertRule {
            name: "loss_spike".to_string(),
            metric_path: "sample.loss".to_string(),
            condition: AlertCondition::Threshold {
                operator: ComparisonOperator::GreaterThan,
                value: 1.0,
            },
            severity: AlertSeverity::Critical,
            evaluation_frequency: Duration::from_secs(0),
            notifications: Vec::new(),
        })
        .expect("a threshold rule on a known path must be accepted");

    collector
        .record_sample(sample_at(0, 2.5, 0.5, 5))
        .expect("record");
    assert_eq!(
        collector.active_alerts().len(),
        1,
        "M1 regression: a breached threshold raised no alert"
    );
    let alert_id = collector.active_alerts()[0].id.clone();
    assert!(alert_id.starts_with("loss_spike#"));

    // Still breaching: the same alert must be updated, not duplicated.
    collector
        .record_sample(sample_at(1, 3.0, 0.5, 5))
        .expect("record");
    assert_eq!(collector.active_alerts().len(), 1);
    assert_eq!(collector.active_alerts()[0].id, alert_id);

    // Recovered: the alert must be resolved and moved into history.
    collector
        .record_sample(sample_at(2, 0.2, 0.5, 5))
        .expect("record");
    assert!(
        collector.active_alerts().is_empty(),
        "a recovered metric must clear its active alert"
    );
    assert_eq!(collector.alert_history().len(), 1);
    assert!(collector.alert_history()[0].resolved_at.is_some());
}

#[test]
fn alert_ids_are_unique_across_repeated_firings() {
    let mut collector = collector_without_compression();
    collector
        .add_alert_rule(AlertRule {
            name: "loss_spike".to_string(),
            metric_path: "sample.loss".to_string(),
            condition: AlertCondition::Threshold {
                operator: ComparisonOperator::GreaterThan,
                value: 1.0,
            },
            severity: AlertSeverity::Warning,
            evaluation_frequency: Duration::from_secs(0),
            notifications: Vec::new(),
        })
        .expect("rule accepted");

    let mut ids = Vec::new();
    for step in 0..3u64 {
        collector
            .record_sample(sample_at(step * 2, 5.0, 0.5, 5))
            .expect("record");
        ids.push(collector.active_alerts()[0].id.clone());
        collector
            .record_sample(sample_at(step * 2 + 1, 0.1, 0.5, 5))
            .expect("record");
    }
    ids.dedup();
    assert_eq!(ids.len(), 3, "each firing must get a distinct alert id");
}

#[test]
fn unevaluatable_rules_are_rejected_at_registration() {
    let mut collector = collector_without_compression();
    assert!(
        collector
            .add_alert_rule(AlertRule {
                name: "custom".to_string(),
                metric_path: "sample.loss".to_string(),
                condition: AlertCondition::Custom {
                    expression: "loss > 1".to_string()
                },
                severity: AlertSeverity::Info,
                evaluation_frequency: Duration::from_secs(0),
                notifications: Vec::new(),
            })
            .is_err(),
        "a rule this crate cannot evaluate must be rejected rather than silently ignored"
    );

    assert!(
        collector
            .add_alert_rule(AlertRule {
                name: "typo".to_string(),
                metric_path: "performance.does_not_exist".to_string(),
                condition: AlertCondition::Threshold {
                    operator: ComparisonOperator::GreaterThan,
                    value: 1.0,
                },
                severity: AlertSeverity::Info,
                evaluation_frequency: Duration::from_secs(0),
                notifications: Vec::new(),
            })
            .is_err(),
        "an unknown metric path must be rejected"
    );
}

#[test]
fn remote_notification_channels_are_rejected() {
    let mut collector = collector_without_compression();
    assert!(collector
        .add_notification_channel(NotificationChannel::PagerDuty {
            integration_key: "key".to_string(),
        })
        .is_err());

    let path = std::env::temp_dir().join(format!("optirs_alert_sink_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let mut config = HashMap::new();
    config.insert("name".to_string(), "sink".to_string());
    config.insert("path".to_string(), path.to_string_lossy().into_owned());
    collector
        .add_notification_channel(NotificationChannel::Custom { config })
        .expect("a file sink is deliverable and must be accepted");

    collector
        .add_alert_rule(AlertRule {
            name: "loss_spike".to_string(),
            metric_path: "sample.loss".to_string(),
            condition: AlertCondition::Threshold {
                operator: ComparisonOperator::GreaterThan,
                value: 1.0,
            },
            severity: AlertSeverity::Critical,
            evaluation_frequency: Duration::from_secs(0),
            notifications: Vec::new(),
        })
        .expect("rule accepted");
    collector
        .record_sample(sample_at(0, 9.0, 0.5, 5))
        .expect("record");

    let delivered = std::fs::read_to_string(&path).expect("the sink file must have been written");
    assert!(delivered.contains("loss_spike"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn anomaly_rules_use_the_running_distribution() {
    let mut collector = collector_without_compression();
    collector
        .add_alert_rule(AlertRule {
            name: "loss_anomaly".to_string(),
            metric_path: "sample.loss".to_string(),
            condition: AlertCondition::Anomaly { sensitivity: 3.0 },
            severity: AlertSeverity::Warning,
            evaluation_frequency: Duration::from_secs(0),
            notifications: Vec::new(),
        })
        .expect("rule accepted");

    for step in 0..20u64 {
        let loss = 1.0 + 0.001 * (step % 3) as f64;
        collector
            .record_sample(sample_at(step, loss, 0.5, 5))
            .expect("record");
    }
    assert!(
        collector.active_alerts().is_empty(),
        "a stationary stream must not raise an anomaly alert"
    );

    collector
        .record_sample(sample_at(100, 100.0, 0.5, 5))
        .expect("record");
    assert_eq!(
        collector.active_alerts().len(),
        1,
        "a 100x outlier must raise an anomaly alert"
    );
}

// ---------------------------------------------------------------------------
// M3: the raw time series grew without bound.
// ---------------------------------------------------------------------------

#[test]
fn raw_history_is_bounded_by_the_retention_window() {
    let mut collector = collector_without_compression();
    collector.set_retention_policy(RetentionPolicy {
        raw_data_retention: 5,
        auto_cleanup: true,
        ..RetentionPolicy::default()
    });

    for seconds in 0..100u64 {
        collector
            .record_sample(sample_at(seconds, 1.0 + seconds as f64, 0.5, 5))
            .expect("record");
    }
    let retained = collector.retained_snapshot_count();
    assert!(
        retained <= 6,
        "M3 regression: a 5 second retention window kept {retained} snapshots out of 100"
    );
    assert!(retained > 0);
}

#[test]
fn raw_history_is_bounded_by_the_storage_cap_even_without_auto_cleanup() {
    let mut collector = collector_without_compression();
    let per_snapshot = std::mem::size_of::<MetricsSnapshot<f64>>() as u64 + 8;
    collector.set_retention_policy(RetentionPolicy {
        raw_data_retention: u64::MAX,
        auto_cleanup: false,
        max_storage_size: per_snapshot * 4,
        ..RetentionPolicy::default()
    });

    for seconds in 0..50u64 {
        collector
            .record_sample(sample_at(seconds, 1.0 + seconds as f64, 0.5, 5))
            .expect("record");
    }
    assert!(
        collector.retained_snapshot_count() <= 4,
        "M3 regression: the storage cap did not bound the series ({} snapshots)",
        collector.retained_snapshot_count()
    );
}

#[test]
fn temporal_compression_drops_redundant_points_but_keeps_varying_ones() {
    let mut compressing = StreamingMetricsCollector::<f64>::new();
    compressing.set_compression_config(CompressionConfig {
        enabled: true,
        algorithm: CompressionAlgorithm::Custom,
        target_ratio: 0.5,
        lossy_tolerance: 0.05,
    });
    // A perfectly linear loss ramp: interior points are redundant.
    for seconds in 0..20u64 {
        compressing
            .record_sample(sample_at(seconds, 100.0 - seconds as f64, 0.5, 5))
            .expect("record");
    }
    let compressed = compressing.retained_snapshot_count();

    let mut verbatim = collector_without_compression();
    for seconds in 0..20u64 {
        verbatim
            .record_sample(sample_at(seconds, 100.0 - seconds as f64, 0.5, 5))
            .expect("record");
    }
    assert_eq!(verbatim.retained_snapshot_count(), 20);
    assert!(
        compressed < 20,
        "temporal compression must actually reduce the stored point count (kept {compressed})"
    );
    assert!(
        compressed >= 2,
        "compression must never empty the series (kept {compressed})"
    );
}

// ---------------------------------------------------------------------------
// M4: three `duration_since(..).expect(..)` calls could panic.
// ---------------------------------------------------------------------------

#[test]
fn pre_epoch_timestamps_do_not_panic() {
    let mut collector = collector_without_compression();
    let before_epoch = SystemTime::UNIX_EPOCH - Duration::from_secs(120);
    let sample = MetricsSample::new(before_epoch, 1.0f64, 0.5, Duration::from_millis(5), 512);

    collector
        .record_sample(sample)
        .expect("M4 regression: a pre-epoch sample timestamp must not panic");

    let range = collector
        .get_historical_metrics(before_epoch, SystemTime::UNIX_EPOCH)
        .expect("M4 regression: a pre-epoch range query must not panic");
    assert_eq!(range.len(), 1);
}

#[test]
fn inverted_ranges_are_an_error_not_a_panic() {
    let collector = collector_without_compression();
    assert!(collector.get_historical_metrics(at(100), at(10)).is_err());
}

/// The time series used to be keyed by whole seconds, so several samples taken
/// inside the same second silently overwrote one another — under sub-second
/// streaming rates that discarded almost everything and made the retention and
/// compression work below it unobservable. Keying by microseconds keeps every
/// distinct sample.
#[test]
fn sub_second_samples_are_all_retained() {
    let mut collector = collector_without_compression();

    // Five samples inside the same wall-clock second, 100 ms apart.
    for step in 0..5u64 {
        let timestamp =
            SystemTime::UNIX_EPOCH + Duration::from_secs(10) + Duration::from_millis(100 * step);
        let sample = MetricsSample::new(
            timestamp,
            1.0 - 0.1 * step as f64,
            0.5,
            Duration::from_millis(2),
            1024,
        );
        collector.record_sample(sample).expect("record_sample");
    }

    assert_eq!(
        collector.retained_snapshot_count(),
        5,
        "sub-second samples were collapsed onto one key (whole-second key regression)"
    );

    // All five must be retrievable, and all five must report the same *second*
    // while carrying five distinct microsecond timestamps.
    let range = collector
        .get_historical_metrics(at(10), at(11))
        .expect("range query");
    assert_eq!(range.len(), 5);
    assert!(
        range.iter().all(|snapshot| snapshot.timestamp == 10),
        "the second-resolution timestamp must be preserved for bucketing"
    );
    let mut micros: Vec<u64> = range.iter().map(|s| s.timestamp_micros).collect();
    micros.sort_unstable();
    micros.dedup();
    assert_eq!(micros.len(), 5, "microsecond timestamps must be distinct");
    assert_eq!(micros[0], 10 * 1_000_000);
    assert_eq!(micros[4], 10 * 1_000_000 + 400_000);
}

/// Retention is configured in seconds; with a microsecond key the conversion
/// must be applied, otherwise a 3600-second window would be read as 3600
/// microseconds and prune essentially everything.
#[test]
fn retention_window_is_interpreted_in_seconds() {
    let mut collector = collector_without_compression();

    // Samples one second apart across two minutes, well inside the default
    // raw-data retention window.
    for second in 0..120u64 {
        collector
            .record_sample(sample_at(second, 1.0, 0.5, 2))
            .expect("record_sample");
    }

    assert!(
        collector.retained_snapshot_count() > 1,
        "a seconds-vs-micros unit error pruned the whole series: {} snapshots left",
        collector.retained_snapshot_count()
    );
}
