// Communication Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

pub mod authentication;
pub mod authorization;
pub mod buffer_management;
pub mod buffers;
pub mod certificate_management;
pub mod communication_core;
pub mod compression;
pub mod core;
pub mod encryption;
pub mod identity_management;
pub mod key_management;
pub mod monitoring;
pub mod network_config;
pub mod policy_management;
pub mod protocols;
pub mod qos;
pub mod reliability;
pub mod routing;
pub mod scheduling;
pub mod security;
pub mod security_monitoring;

pub use authentication::*;
pub use authorization::*;
pub use buffer_management::*;
pub use buffers::*;
pub use certificate_management::*;
pub use communication_core::*;
pub use compression::*;
pub use core::*;
pub use encryption::*;
pub use identity_management::*;
pub use key_management::*;
pub use monitoring::*;
pub use network_config::*;
pub use policy_management::*;
pub use protocols::*;
pub use qos::*;
pub use reliability::*;
pub use routing::*;
pub use scheduling::*;
pub use security::*;
pub use security_monitoring::*;

// Define missing types
pub type CommunicationId = u64;

#[derive(Debug, Clone, Default)]
pub struct ActiveCommunication {
    pub id: CommunicationId,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferStatus {
    Empty,
    Partial,
    Full,
}

impl Default for BufferStatus {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationBuffer {
    pub status: BufferStatus,
    pub size: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationManager {
    pub active_comms: Vec<ActiveCommunication>,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationProgress {
    pub bytes_sent: u64,
    pub bytes_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Lz4,
    Zstd,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompressionInfo {
    pub algorithm: CompressionAlgorithm,
    pub ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CommunicationStatistics {
    pub total_bytes: u64,
    pub total_messages: u64,
}
