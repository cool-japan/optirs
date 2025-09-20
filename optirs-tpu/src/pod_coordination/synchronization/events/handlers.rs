// Handlers Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct EventHandler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandlerType {
    Sync,
    Async,
    Buffered,
}

impl Default for HandlerType {
    fn default() -> Self {
        Self::Async
    }
}

#[derive(Debug, Clone, Default)]
pub struct HandlerConfig {
    pub handler_type: HandlerType,
}
