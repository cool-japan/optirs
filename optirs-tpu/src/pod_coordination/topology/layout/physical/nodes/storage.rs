// Storage Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageCapabilities {
    pub capacity_tb: f64,
    pub iops: u64,
}
