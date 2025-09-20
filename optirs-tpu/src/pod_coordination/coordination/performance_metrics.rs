// Performance Metrics Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceMetricsManager {
    pub items: Vec<PerformanceMetrics>,
}
