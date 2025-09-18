// Monitoring for optimization coordination
//
// This module provides comprehensive monitoring capabilities for optimization
// processes, including performance tracking, convergence detection, and
// anomaly detection for optimization workflows.

#[allow(dead_code)]
pub mod anomaly_detection;
pub mod convergence_detection;
pub mod performance_tracking;

// Re-export key types
pub use performance_tracking::{
    AlertManager, MetricAggregator, MetricCollector, PerformanceAlert, PerformanceMetrics,
    PerformanceTracker,
};

pub use convergence_detection::{
    ConvergenceAnalyzer, ConvergenceCriteria, ConvergenceDetector, ConvergenceIndicator,
    ConvergenceMonitor, ConvergenceResult,
};

pub use anomaly_detection::{
    AnomalyAlert, AnomalyAnalyzer, AnomalyClassifier, AnomalyDetector, AnomalyReporter,
    OutlierDetector,
};
