// Network Configuration Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectionPoolingConfig {
    pub min_connections: usize,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkConfig {
    pub tcp_settings: TcpSettings,
    pub udp_settings: UdpSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkOptimizationConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProtocolSettings;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RdmaSettings;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SocketBufferConfig {
    pub send_buffer_size: usize,
    pub recv_buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TcpSettings {
    pub no_delay: bool,
    pub keepalive_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UdpSettings {
    pub max_packet_size: usize,
}
