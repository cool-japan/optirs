//! Consolidated unit-test suite for the `monitoring` module tree.

use std::time::Duration;

use super::types::*;

#[test]
fn test_topology_performance_monitor() {
    let mut monitor = TopologyPerformanceMonitor::new();
    let config = TopologyMonitoringSettings::default();

    assert!(monitor.start_monitoring(config).is_ok());
    assert!(monitor.collect_metrics().is_ok());
}

#[test]
fn test_alert_system() {
    let alert_system = AlertSystem::default();
    assert_eq!(alert_system.active_alerts.len(), 0);
    assert_eq!(alert_system.alert_history.len(), 0);
}

#[test]
fn test_metrics_collection() {
    let mut collector = MetricsCollector::default();
    collector.metrics.insert("test_metric".to_string(), 42.0);

    assert_eq!(collector.metrics.get("test_metric"), Some(&42.0));
}

#[test]
fn test_anomaly_detection() {
    let detector = AnomalyDetector::default();
    assert_eq!(detector.anomalies.len(), 0);
    assert_eq!(detector.statistics.total_detected, 0);
}

// --- F12 / F13 regression tests ---

#[test]
fn test_collection_timestamp_is_real_wall_clock() {
    let mut monitor = TopologyPerformanceMonitor::new();
    let metrics = monitor.collect_metrics().expect("collect_metrics failed");
    let timestamp = *metrics
        .get("collection_timestamp")
        .expect("missing collection_timestamp");

    // A real Unix timestamp must be well past the epoch (years, not ~0.0).
    // 1_600_000_000 s corresponds to 2020-09-13, so any honest clock exceeds it.
    assert!(
        timestamp > 1_600_000_000.0,
        "timestamp {timestamp} looks like an Instant delta, not wall-clock time"
    );
}

#[test]
fn test_reports_have_distinct_ids() {
    let monitor = TopologyPerformanceMonitor::new();
    let first = monitor
        .generate_report(Duration::from_secs(60))
        .expect("first report failed");
    let second = monitor
        .generate_report(Duration::from_secs(60))
        .expect("second report failed");

    assert_ne!(
        first.report_id, second.report_id,
        "report ids must be unique, got {} twice",
        first.report_id
    );
    // The monotonic sequence guarantees distinct ids even within one second.
    assert!(first.report_id.starts_with("report_0_"));
    assert!(second.report_id.starts_with("report_1_"));
}

#[test]
fn test_report_overall_score_is_not_hardcoded() {
    let mut monitor = TopologyPerformanceMonitor::new();

    // Inject a degraded latency reading so a real (non-0.85) score results.
    monitor.health_monitoring.health_indicators = vec![HealthIndicator::DeviceResponsiveness];
    monitor
        .metrics_collector
        .metrics
        .insert("latency".to_string(), 60.0); // >= critical (50ms) -> score 0.0

    let report = monitor
        .generate_report(Duration::from_secs(60))
        .expect("report failed");

    // Critical latency -> health score 0.0, so overall must differ from 0.85.
    assert!((report.summary.overall_score - 0.85).abs() > f64::EPSILON);
    assert_eq!(report.summary.overall_score, 0.0);
}

#[test]
fn test_overall_score_reflects_healthy_data() {
    let mut monitor = TopologyPerformanceMonitor::new();
    monitor.health_monitoring.health_indicators = vec![HealthIndicator::DeviceResponsiveness];
    monitor
        .metrics_collector
        .metrics
        .insert("latency".to_string(), 1.0); // well under warning (10ms) -> Healthy

    let report = monitor
        .generate_report(Duration::from_secs(60))
        .expect("report failed");
    assert_eq!(report.summary.overall_score, 1.0);
}

#[test]
fn test_health_check_unknown_without_data() {
    let monitor = TopologyPerformanceMonitor::new();
    let result = monitor
        .perform_health_check(&HealthIndicator::LinkConnectivity)
        .expect("health check failed");
    // No data collected -> must not claim Healthy.
    assert_eq!(result.status, HealthStatus::Unknown);
}

#[test]
fn test_health_check_derives_status_from_data() {
    let mut monitor = TopologyPerformanceMonitor::new();
    // High packet loss (>= 5% critical threshold) must read as Critical.
    monitor
        .metrics_collector
        .metrics
        .insert("packet_loss".to_string(), 0.2);
    let result = monitor
        .perform_health_check(&HealthIndicator::ErrorRateIncrease)
        .expect("health check failed");
    assert_eq!(result.status, HealthStatus::Critical);

    // A near-perfect link connectivity ratio must read as Healthy.
    monitor
        .metrics_collector
        .metrics
        .insert("link_connectivity".to_string(), 0.999);
    let link = monitor
        .perform_health_check(&HealthIndicator::LinkConnectivity)
        .expect("health check failed");
    assert_eq!(link.status, HealthStatus::Healthy);
}

#[test]
fn test_outlier_flagged_but_in_distribution_is_not() {
    let mut monitor = TopologyPerformanceMonitor::new();

    // Seed an in-distribution baseline with genuine variation
    // (mean ~= 100.0, std ~= 0.6) for two independent metric names.
    let baseline = [100.0_f64, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8];
    for &sample in &baseline {
        let metrics: TopologyMetrics = [
            ("stable_metric".to_string(), sample),
            ("outlier_metric".to_string(), sample),
        ]
        .into_iter()
        .collect();
        let found = monitor.detect_anomalies(&metrics).expect("detect failed");
        // While the baseline is being established nothing should be flagged.
        assert!(
            found.is_empty(),
            "baseline value {sample} was wrongly flagged as anomalous"
        );
    }

    // An in-distribution value (z ~= 0.5) must NOT be flagged. Check this on
    // its own metric before any outlier pollutes the baseline.
    let in_dist: TopologyMetrics = [("stable_metric".to_string(), 100.3_f64)]
        .into_iter()
        .collect();
    let normal = monitor.detect_anomalies(&in_dist).expect("detect failed");
    assert!(
        normal.is_empty(),
        "an in-distribution value was incorrectly flagged as anomalous"
    );

    // A clear outlier (z in the thousands) on a separate metric MUST be flagged.
    let spike: TopologyMetrics = [("outlier_metric".to_string(), 5000.0_f64)]
        .into_iter()
        .collect();
    let anomalies = monitor.detect_anomalies(&spike).expect("detect failed");
    assert_eq!(anomalies.len(), 1, "clear outlier was not flagged");
    assert!(anomalies[0].score > 0.5, "outlier score should be high");
    assert!(anomalies[0].score <= 1.0);
}

#[test]
fn test_insufficient_history_is_not_anomalous() {
    let mut monitor = TopologyPerformanceMonitor::new();
    // Fewer than ANOMALY_MIN_SAMPLES observations -> honest "not enough data".
    for value in [10.0_f64, 11.0, 9.0] {
        let metrics: TopologyMetrics = [("m".to_string(), value)].into_iter().collect();
        let found = monitor.detect_anomalies(&metrics).expect("detect failed");
        assert!(found.is_empty());
    }
    // Even a wild value is not flagged yet: too little history to judge.
    let metrics: TopologyMetrics = [("m".to_string(), 1_000_000.0_f64)].into_iter().collect();
    let found = monitor.detect_anomalies(&metrics).expect("detect failed");
    assert!(found.is_empty());
}
