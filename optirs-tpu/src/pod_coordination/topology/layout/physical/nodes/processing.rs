// Processing Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessingCapabilities {
    pub tflops: f64,
    pub cores: u32,
}
