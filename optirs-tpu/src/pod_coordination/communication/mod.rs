// Communication Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

pub mod buffer_management;
pub mod buffers;
pub mod communication_core;
pub mod compression;
pub mod monitoring;
pub mod network_config;
pub mod protocols;
pub mod qos;
pub mod reliability;
pub mod routing;
pub mod scheduling;
pub mod security;

pub use buffer_management::*;
pub use buffers::*;
pub use communication_core::*;
pub use compression::*;
pub use monitoring::*;
pub use network_config::*;
pub use protocols::*;
pub use qos::*;
pub use reliability::*;
pub use routing::*;
pub use scheduling::*;
pub use security::*;

// Define missing types
pub type CommunicationId = u64;

#[derive(Debug, Clone, Default)]
pub struct ActiveCommunication {
    pub id: CommunicationId,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum BufferStatus {
    #[default]
    Empty,
    Partial,
    Full,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CompressionAlgorithm {
    #[default]
    None,
    Gzip,
    Lz4,
    Zstd,
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
