//! `Default` implementations for the monitoring settings/config types
//! declared in [`super::types`].

use std::collections::HashMap;
use std::time::Duration;

use super::types::*;

// Default implementations

impl Default for PerformanceMonitoringSettings {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            metrics_collection: MetricsCollectionSettings::default(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for MetricsCollectionSettings {
    fn default() -> Self {
        Self {
            collected_metrics: vec![
                MetricType::Latency,
                MetricType::Throughput,
                MetricType::BandwidthUtilization,
            ],
            granularity: CollectionGranularity::PerDevice,
            retention_period: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            latency_thresholds: ThresholdLevels {
                warning: 10.0,    // 10ms
                critical: 50.0,   // 50ms
                emergency: 100.0, // 100ms
            },
            throughput_thresholds: ThresholdLevels {
                warning: 80.0,   // 80% of capacity
                critical: 90.0,  // 90% of capacity
                emergency: 95.0, // 95% of capacity
            },
            utilization_thresholds: ThresholdLevels {
                warning: 70.0,   // 70%
                critical: 85.0,  // 85%
                emergency: 95.0, // 95%
            },
            error_thresholds: ThresholdLevels {
                warning: 0.01,   // 1% error rate
                critical: 0.05,  // 5% error rate
                emergency: 0.10, // 10% error rate
            },
        }
    }
}

impl Default for HealthMonitoringSettings {
    fn default() -> Self {
        Self {
            check_frequency: Duration::from_secs(5),
            health_indicators: vec![
                HealthIndicator::LinkConnectivity,
                HealthIndicator::DeviceResponsiveness,
                HealthIndicator::PerformanceDegradation,
            ],
            failure_detection: FailureDetectionSettings {
                algorithm: FailureDetectionAlgorithm::ThresholdBased,
                sensitivity: 0.8,
                false_positive_tolerance: 0.05,
            },
        }
    }
}

impl Default for FlowMonitoringSettings {
    fn default() -> Self {
        Self {
            tracking_granularity: FlowTrackingGranularity::PerFlow,
            flow_timeout: Duration::from_secs(60),
            sampling_rate: 1.0, // 100% sampling
        }
    }
}

impl Default for PatternAnalysisSettings {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(300), // 5 minutes
            detection_algorithms: vec![PatternDetectionAlgorithm::TimeSeriesAnalysis],
            classification: PatternClassification::default(),
        }
    }
}

impl Default for PatternClassification {
    fn default() -> Self {
        Self {
            method: ClassificationMethod::Statistical,
            categories: vec!["normal".to_string(), "anomalous".to_string()],
            confidence_threshold: 0.8,
        }
    }
}

impl Default for AnomalyDetectionSettings {
    fn default() -> Self {
        Self {
            method: AnomalyDetectionMethod::Statistical,
            sensitivity: 0.8,
            baseline_establishment: BaselineEstablishment::default(),
        }
    }
}

impl Default for BaselineEstablishment {
    fn default() -> Self {
        Self {
            learning_period: Duration::from_secs(3600), // 1 hour
            update_frequency: Duration::from_secs(300), // 5 minutes
            adaptation_rate: 0.1,
        }
    }
}

impl Default for AlertSettings {
    fn default() -> Self {
        Self {
            alert_channels: vec![AlertChannel::Email {
                recipients: vec!["admin@example.com".to_string()],
            }],
            alert_thresholds: AlertThresholds::default(),
            escalation: AlertEscalation::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance: PerformanceThresholds::default(),
            health: HealthThresholds {
                device_failure: 0.95, // 95% confidence
                link_failure: 0.90,   // 90% confidence
                degradation: 0.80,    // 80% confidence
            },
            anomaly: AnomalyThresholds {
                score_threshold: 0.8,     // 80% anomaly score
                frequency_threshold: 0.1, // 10% frequency
                severity_threshold: 0.7,  // 70% severity
            },
        }
    }
}

impl Default for AlertEscalation {
    fn default() -> Self {
        Self {
            levels: vec![
                EscalationLevel {
                    level_id: "level1".to_string(),
                    priority: EscalationPriority::Low,
                    targets: vec!["admin@example.com".to_string()],
                    require_ack: false,
                },
                EscalationLevel {
                    level_id: "level2".to_string(),
                    priority: EscalationPriority::High,
                    targets: vec!["manager@example.com".to_string()],
                    require_ack: true,
                },
            ],
            timers: vec![Duration::from_secs(300), Duration::from_secs(900)], // 5min, 15min
            actions: vec![EscalationAction::SendNotification {
                channel: AlertChannel::Email {
                    recipients: vec!["admin@example.com".to_string()],
                },
            }],
        }
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self {
            config: AnomalyDetectionSettings::default(),
            models: Vec::new(),
            anomalies: Vec::new(),
            statistics: AnomalyDetectionStatistics {
                total_detected: 0,
                false_positive_rate: 0.05,
                detection_accuracy: 0.95,
                avg_detection_time: Duration::from_secs(5),
            },
            value_history: HashMap::new(),
        }
    }
}

impl Default for PerformanceAnalytics {
    fn default() -> Self {
        Self {
            config: AnalyticsConfig {
                analysis_window: Duration::from_secs(3600),   // 1 hour
                report_frequency: Duration::from_secs(86400), // 24 hours
                enable_prediction: true,
                prediction_horizon: Duration::from_secs(7200), // 2 hours
            },
            reports: Vec::new(),
            trend_analysis: TrendAnalysis {
                config: TrendAnalysisConfig {
                    window_size: Duration::from_secs(1800),       // 30 minutes
                    min_trend_duration: Duration::from_secs(300), // 5 minutes
                    sensitivity: 0.7,
                },
                trends: Vec::new(),
                predictions: Vec::new(),
            },
            predictive_models: Vec::new(),
        }
    }
}
