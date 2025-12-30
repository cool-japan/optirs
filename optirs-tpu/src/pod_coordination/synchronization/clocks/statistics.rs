// Statistics Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Statistics {
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct StatisticsCollector {
    pub current: Statistics,
    pub history: Vec<Statistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum StatisticType {
    #[default]
    Mean,
    Median,
    Percentile,
}

#[derive(Debug, Clone, Default)]
pub struct ClockStatistics {
    pub stats: Statistics,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceHistory {
    pub measurements: Vec<PerformanceMeasurement>,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceMeasurement {
    pub value: f64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceReport {
    pub statistics: Statistics,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceTracking {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct QualityReporting {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ReliabilityStatistics {
    pub uptime_percent: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ReportGeneration {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct StatisticsError;

impl std::fmt::Display for StatisticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Statistics error")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TrendDirection {
    Improving,
    #[default]
    Stable,
    Declining,
}

// Add missing types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PerformanceMetric {
    #[default]
    Latency,
    Throughput,
    ErrorRate,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ReportFormat {
    #[default]
    Json,
    Text,
    Html,
}

#[derive(Debug, Clone, Default)]
pub struct StatisticsCollectionConfig {
    pub interval_ms: u64,
}
