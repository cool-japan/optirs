// Clock Protocols Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BerkeleyConfig {
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ClockSyncProtocol {
    #[default]
    NTP,
    PTP,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CristianConfig {
    pub server_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CustomProtocolConfig {
    pub protocol_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NtpConfig {
    pub server: String,
}

#[derive(Debug, Clone, Default)]
pub struct NtpSynchronizer {
    pub config: NtpConfig,
}

#[derive(Debug, Clone)]
pub struct ProtocolError;

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Protocol error")
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProtocolManager {
    pub protocol: ClockSyncProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PtpConfig {
    pub domain: u8,
}

#[derive(Debug, Clone, Default)]
pub struct PtpSynchronizer {
    pub config: PtpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SntpConfig {
    pub server: String,
}

#[derive(Debug, Clone, Default)]
pub struct SntpSynchronizer {
    pub config: SntpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpVersion {
    V1,
    #[default]
    V2,
    V2_1,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpTransport {
    #[default]
    UDP,
    Ethernet,
    Serial,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpProfile {
    #[default]
    Default,
    Telecom,
    Power,
    Industrial,
}
