// Health Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Failed,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Healthy
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub last_check_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HealthMonitor {
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Clone, Default)]
pub struct AlertConfiguration {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Info
    }
}

#[derive(Debug, Clone, Default)]
pub struct HealthAlert {
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    Connectivity,
    Performance,
    Resource,
}

impl Default for HealthCheckType {
    fn default() -> Self {
        Self::Connectivity
    }
}

#[derive(Debug, Clone, Default)]
pub struct HealthMonitorConfig {
    pub check_interval_ms: u64,
}

#[derive(Debug, Clone)]
pub struct HealthMonitorError;

impl std::fmt::Display for HealthMonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Health monitoring error")
    }
}

#[derive(Debug, Clone, Default)]
pub struct HealthThresholds {
    pub warning: f64,
    pub critical: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RecoveryConfiguration {
    pub auto_recover: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SourceFailoverConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SourceHealthMonitor {
    pub config: HealthMonitorConfig,
}
