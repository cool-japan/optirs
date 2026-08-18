// Clock Sources Module

use serde::{Deserialize, Serialize};

pub mod selection;

pub use selection::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AtomicClockType {
    #[default]
    Cesium,
    Rubidium,
    Hydrogen,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ClockSource {
    #[default]
    System,
    GPS,
    NTP,
    PTP,
    Atomic,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RadioTimeStation {
    #[default]
    WWV,
    WWVB,
    DCF77,
    MSF,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SourceSelectionAlgorithm {
    #[default]
    BestQuality,
    RoundRobin,
    Priority,
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

    /// Endpoint this source is reached at (an NTP/PTP server host, a GPS
    /// receiver device path, ...).
    ///
    /// `None` for sources that need no address, such as
    /// [`ClockSource::System`] and a locally attached atomic reference. A
    /// network source without an address is not addressable at all, which is
    /// why [`super::utils::create_ntp_sync_manager`] refuses to build one.
    pub address: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CalibrationReference {
    #[default]
    GPS,
    Atomic,
    Network,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CalibrationMethod {
    #[default]
    Linear,
    Polynomial,
    Adaptive,
}

#[derive(Debug, Clone, Default)]
pub struct ClockCalibration {
    pub reference: CalibrationReference,
    pub method: CalibrationMethod,
    pub offset_ns: i64,
    pub drift_ppm: f64,
}
