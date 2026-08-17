// Physical Properties Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodePhysicalProperties {
    pub rack_unit: u32,
    pub weight_kg: f64,
}
