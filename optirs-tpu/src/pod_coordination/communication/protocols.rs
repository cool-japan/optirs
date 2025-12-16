// Communication Protocols Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ProtocolError;

pub type ProtocolResult<T> = std::result::Result<T, ProtocolError>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ProtocolType {
    HTTP,
    #[default]
    GRPC,
    TCP,
    UDP,
}

#[derive(Debug, Clone, Default)]
pub struct ProtocolManager {
    pub protocol: ProtocolType,
}
