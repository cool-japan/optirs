//! Device manager: enumerates TPU devices from the backend configuration
//! and tracks per-device health, utilization, and topology state.

use std::collections::HashMap;

use crate::error::Result;

use super::device_defaults::{
    default_device_health, device_compute_capability, device_coordinates, device_memory_capacity,
    device_performance_characteristics,
};
use super::types::{
    CompiledProgram, ComputationId, DeviceHealthStatus, DeviceStatus, DeviceTopology, LoadBalancer,
    TPUBackendConfig, TPUDevice,
};
use super::DeviceId;

/// Device manager for TPU hardware
#[derive(Debug)]
pub struct DeviceManager {
    /// Available TPU devices
    ///
    /// `pub(super)`: read directly by the `tpu_backend` test module, which
    /// lives in a sibling submodule and therefore needs at least
    /// module-subtree visibility rather than file-private access.
    pub(super) devices: Vec<TPUDevice>,

    /// Device assignments
    device_assignments: HashMap<ComputationId, Vec<DeviceId>>,

    /// Device health status
    pub(super) device_health: HashMap<DeviceId, DeviceHealthStatus>,

    /// Device utilization
    pub(super) device_utilization: HashMap<DeviceId, f64>,

    /// Device topology
    topology: DeviceTopology,

    /// Load balancer
    load_balancer: LoadBalancer,
}

impl DeviceManager {
    pub fn new(config: &TPUBackendConfig) -> Result<Self> {
        // Populate a real, honest device set from the configuration. One device
        // record is created per configured TPU core (at least one), with
        // capabilities derived from the configured TPU version and coordinates
        // derived from the configured pod topology.
        let device_count = config.tpu_config.num_cores.max(1);
        let version = config.tpu_config.tpu_version;
        let topology = config.tpu_config.pod_topology;

        let mut devices = Vec::with_capacity(device_count);
        let mut device_health = HashMap::with_capacity(device_count);
        let mut device_utilization = HashMap::with_capacity(device_count);

        for index in 0..device_count {
            let id = DeviceId(index);
            devices.push(TPUDevice {
                id,
                device_type: version,
                memory_capacity: device_memory_capacity(version),
                compute_capability: device_compute_capability(version),
                status: DeviceStatus::Available,
                interconnect_links: Vec::new(),
                coordinates: device_coordinates(topology, index),
                performance_characteristics: device_performance_characteristics(version),
            });
            device_health.insert(id, default_device_health());
            device_utilization.insert(id, 0.0);
        }

        Ok(Self {
            devices,
            device_assignments: HashMap::new(),
            device_health,
            device_utilization,
            topology: DeviceTopology::default(),
            load_balancer: LoadBalancer::default(),
        })
    }

    pub fn get_utilization_stats(&self) -> HashMap<DeviceId, f64> {
        // Return device utilization map directly
        self.device_utilization.clone()
    }
}

impl DeviceManager {
    pub async fn shutdown(&mut self) -> Result<()> {
        // Simple implementation - shutdown all devices
        self.devices.clear();
        self.device_assignments.clear();
        self.device_health.clear();
        self.device_utilization.clear();
        Ok(())
    }

    pub fn select_devices(&self, program: &CompiledProgram) -> Result<Vec<DeviceId>> {
        // Simple implementation - return first available device
        if self.devices.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![self.devices[0].id])
        }
    }
}
