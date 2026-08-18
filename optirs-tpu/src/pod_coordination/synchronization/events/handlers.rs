// Handlers Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct EventHandler;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum HandlerType {
    Sync,
    #[default]
    Async,
    Buffered,
}

#[derive(Debug, Clone, Default)]
pub struct HandlerConfig {
    pub handler_type: HandlerType,
}
