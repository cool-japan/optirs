// Quality Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClockAccuracyRequirements {
    pub max_drift_ppm: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ClockQualityMonitor {
    pub requirements: ClockAccuracyRequirements,
}

#[derive(Debug, Clone, Default)]
pub struct QualityAssessment {
    pub grade: QualityGrade,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGrade {
    Excellent,
    Good,
    Fair,
    Poor,
}

impl Default for QualityGrade {
    fn default() -> Self {
        Self::Fair
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    Accuracy,
    Stability,
    Reliability,
}

impl Default for QualityMetric {
    fn default() -> Self {
        Self::Accuracy
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMonitoringConfig {
    pub sample_interval_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct QualityRequirements {
    pub min_grade: QualityGrade,
}

#[derive(Debug, Clone, Default)]
pub struct QualitySnapshot {
    pub timestamp_ms: u64,
    pub grade: QualityGrade,
}

#[derive(Debug, Clone, Default)]
pub struct QualityThresholds {
    pub warn_threshold: f64,
    pub error_threshold: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SourceQualityMonitoring {
    pub config: QualityMonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
}

impl Default for TrendDirection {
    fn default() -> Self {
        Self::Stable
    }
}

#[derive(Debug, Clone)]
pub struct QualityMonitorError;

impl std::fmt::Display for QualityMonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Quality monitoring error")
    }
}
