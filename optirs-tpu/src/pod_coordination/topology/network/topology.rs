// Topology Module

use crate::pod_coordination::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Topology {
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct TopologyManager {
    pub items: Vec<Topology>,
}
