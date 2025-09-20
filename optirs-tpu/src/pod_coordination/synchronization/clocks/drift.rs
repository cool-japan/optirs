// Drift Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClockDrift {
    pub drift_ns: i64,
    pub rate_ppm: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DriftCorrector {
    pub current_drift: ClockDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionMethod {
    Linear,
    Exponential,
    Adaptive,
}

impl Default for CorrectionMethod {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftCompensationAlgorithm {
    Linear,
    Quadratic,
    Adaptive,
}

impl Default for DriftCompensationAlgorithm {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Default)]
pub struct DriftCompensationConfig {
    pub algorithm: DriftCompensationAlgorithm,
}

#[derive(Debug, Clone)]
pub struct DriftCompensationError;

impl std::fmt::Display for DriftCompensationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Drift compensation error")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftCompensationStatus {
    Active,
    Inactive,
    Error,
}

impl Default for DriftCompensationStatus {
    fn default() -> Self {
        Self::Inactive
    }
}

#[derive(Debug, Clone, Default)]
pub struct DriftCompensator {
    pub status: DriftCompensationStatus,
}

#[derive(Debug, Clone, Default)]
pub struct DriftMeasurement {
    pub drift_ns: i64,
}

#[derive(Debug, Clone, Default)]
pub struct DriftMeasurementConfig {
    pub sample_interval_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DriftModel {
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct DriftPredictionConfig {
    pub horizon_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DriftPredictionEngine {
    pub config: DriftPredictionConfig,
}
