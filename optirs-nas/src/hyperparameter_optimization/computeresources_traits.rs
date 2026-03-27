//! # ComputeResources - Trait Implementations
//!
//! This module contains trait implementations for `ComputeResources`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ComputeResources;

impl Default for ComputeResources {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            gpu_devices: 1,
            memory_per_device_gb: 16.0,
            network_bandwidth_gbps: 10.0,
        }
    }
}

