// Communication Protocols Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ProtocolError;

pub type ProtocolResult<T> = std::result::Result<T, ProtocolError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolType {
    HTTP,
    GRPC,
    TCP,
    UDP,
}

impl Default for ProtocolType {
    fn default() -> Self {
        Self::GRPC
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProtocolManager {
    pub protocol: ProtocolType,
}
