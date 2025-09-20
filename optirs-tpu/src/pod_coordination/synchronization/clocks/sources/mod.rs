// Clock Sources Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

pub mod authentication;
pub mod management;
pub mod monitoring;
pub mod radio;
pub mod selection;
pub mod time_sources;

pub use authentication::*;
pub use management::*;
pub use monitoring::*;
pub use radio::*;
pub use selection::*;
pub use time_sources::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AtomicClockType {
    Cesium,
    Rubidium,
    Hydrogen,
}

impl Default for AtomicClockType {
    fn default() -> Self {
        Self::Cesium
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClockSource {
    System,
    GPS,
    NTP,
    PTP,
    Atomic,
}

impl Default for ClockSource {
    fn default() -> Self {
        Self::System
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RadioTimeStation {
    WWV,
    WWVB,
    DCF77,
    MSF,
}

impl Default for RadioTimeStation {
    fn default() -> Self {
        Self::WWV
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceSelectionAlgorithm {
    BestQuality,
    RoundRobin,
    Priority,
}

impl Default for SourceSelectionAlgorithm {
    fn default() -> Self {
        Self::BestQuality
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceValidation {
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SystemClockConfig {
    pub use_system_time: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TimeSource {
    pub source_type: ClockSource,
}

#[derive(Debug, Clone, Default)]
pub struct TimeSourceConfig {
    pub source: ClockSource,
}

#[derive(Debug, Clone, Default)]
pub struct TimeSourceManager {
    pub sources: Vec<TimeSource>,
}

#[derive(Debug, Clone)]
pub struct SourceManagementError;

impl std::fmt::Display for SourceManagementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source management error")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationReference {
    GPS,
    Atomic,
    Network,
}

impl Default for CalibrationReference {
    fn default() -> Self {
        Self::GPS
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMethod {
    Linear,
    Polynomial,
    Adaptive,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClockCalibration {
    pub reference: CalibrationReference,
    pub method: CalibrationMethod,
    pub offset_ns: i64,
    pub drift_ppm: f64,
}
