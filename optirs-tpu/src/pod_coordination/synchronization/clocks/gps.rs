// GPS Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AntennaConfig {
    pub gain_dbi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpsConfig {
    pub receiver_type: GpsReceiverType,
}

#[derive(Debug, Clone)]
pub struct GpsError;

impl std::fmt::Display for GpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPS error")
    }
}

#[derive(Debug, Clone, Default)]
pub struct GpsErrorCorrection {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum GpsReceiverType {
    #[default]
    Standard,
    HighPrecision,
    Military,
}

#[derive(Debug, Clone, Default)]
pub struct GpsSignalProcessing;

#[derive(Debug, Clone, Default)]
pub struct GpsSynchronizationManager {
    pub config: GpsConfig,
}

#[derive(Debug, Clone, Default)]
pub struct GpsTime {
    pub week: u16,
    pub seconds: f64,
}

#[derive(Debug, Clone, Default)]
pub struct IonosphericCorrection {
    pub correction_m: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SatelliteClockCorrection {
    pub correction_ns: i64,
}

#[derive(Debug, Clone, Default)]
pub struct TroposphericCorrection {
    pub correction_m: f64,
}
