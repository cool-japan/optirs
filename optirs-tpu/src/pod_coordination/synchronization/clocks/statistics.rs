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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticType {
    Mean,
    Median,
    Percentile,
}

impl Default for StatisticType {
    fn default() -> Self {
        Self::Mean
    }
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

// Add missing types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Latency,
    Throughput,
    ErrorRate,
}

impl Default for PerformanceMetric {
    fn default() -> Self {
        Self::Latency
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Text,
    Html,
}

impl Default for ReportFormat {
    fn default() -> Self {
        Self::Json
    }
}

#[derive(Debug, Clone, Default)]
pub struct StatisticsCollectionConfig {
    pub interval_ms: u64,
}
