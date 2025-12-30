// Security Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Security {
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct SecurityManager {
    pub items: Vec<Security>,
}
