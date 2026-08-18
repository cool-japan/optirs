//! Device manager: enumerates TPU devices from the backend configuration
//! and tracks per-device health, utilization, and topology state.

use std::collections::HashMap;

use scirs2_core::error::ErrorContext;

use crate::error::{OptimError, Result};

use super::device_defaults::{
    default_device_health, device_compute_capability, device_coordinates,
    device_interconnect_bandwidth, device_memory_capacity, device_performance_characteristics,
};
use super::types::{
    CompiledProgram, ComputationId, DeviceHealthStatus, DeviceStatus, DeviceTopology,
    InterconnectLink, InterconnectType, LinkStatus, LoadBalancer, LoadBalancingStrategy,
    TPUBackendConfig, TPUDevice,
};
use super::DeviceId;

/// Minimum health score a device must report before it is considered eligible
/// to run work. Devices below this are excluded from selection rather than
/// silently handed a computation they are known to be degraded for.
const MIN_SELECTABLE_HEALTH_SCORE: f64 = 0.5;

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
    ///
    /// `pub(super)`: asserted on by the `tpu_backend` test module, which checks
    /// that the pod grid produced real neighbour links.
    pub(super) topology: DeviceTopology,

    /// Load balancer
    ///
    /// `pub(super)`: the `tpu_backend` test module drives per-device load
    /// through it to observe the effect on [`Self::select_devices`].
    pub(super) load_balancer: LoadBalancer,
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
        let mut device_loads = HashMap::with_capacity(device_count);

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
            device_loads.insert(id, 0.0);
        }

        // Wire the real fabric: each device gets links to its actual neighbours
        // in the configured pod grid (or its neighbours along the enumeration
        // order for a single-chip topology), and the topology record carries the
        // same adjacency plus the per-link bandwidth. Without this, both
        // `TPUDevice::interconnect_links` and `DeviceTopology` stayed
        // permanently empty and locality-aware selection had nothing to read.
        let adjacency = build_adjacency(&devices);
        let link_bandwidth = device_interconnect_bandwidth(version);
        let link_latency_us = device_performance_characteristics(version).communication_latency_us;

        let mut connections: HashMap<DeviceId, Vec<DeviceId>> =
            HashMap::with_capacity(devices.len());
        let mut bandwidth_matrix: HashMap<(DeviceId, DeviceId), f64> = HashMap::new();

        let coordinates: Vec<Option<(usize, usize)>> =
            devices.iter().map(|device| device.coordinates).collect();

        for (index, neighbours) in adjacency.iter().enumerate() {
            let id = DeviceId(index);
            connections.insert(id, neighbours.clone());
            for neighbour in neighbours {
                bandwidth_matrix.insert((id, *neighbour), link_bandwidth);
                let link_type = link_type_for(
                    coordinates[index],
                    coordinates.get(neighbour.0).copied().flatten(),
                );
                devices[index].interconnect_links.push(InterconnectLink {
                    target_device: *neighbour,
                    bandwidth_gb_s: link_bandwidth,
                    latency_us: link_latency_us,
                    link_type,
                    status: LinkStatus::Active,
                });
            }
        }

        Ok(Self {
            devices,
            device_assignments: HashMap::new(),
            device_health,
            device_utilization,
            topology: DeviceTopology {
                connections,
                bandwidth_matrix,
            },
            load_balancer: LoadBalancer {
                device_loads,
                strategy: config.load_balancing_strategy,
            },
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
        self.load_balancer.device_loads.clear();
        self.topology.connections.clear();
        self.topology.bandwidth_matrix.clear();
        Ok(())
    }

    /// Choose the set of devices that will run `program`.
    ///
    /// The program is not ignored: its
    /// [`ProgramMemoryRequirements::total_memory`] is the constraint the
    /// selection has to satisfy. Devices that are unusable (non-runnable
    /// status, or a health score below this module's private
    /// `MIN_SELECTABLE_HEALTH_SCORE` floor) are
    /// excluded, the remainder are ordered by the configured
    /// [`LoadBalancingStrategy`], and devices are taken from that order until
    /// their combined memory capacity covers the requirement. A program whose
    /// footprint exceeds the whole usable pod is an honest `Err` rather than a
    /// selection that would fault at allocation time.
    ///
    /// [`ProgramMemoryRequirements::total_memory`]: super::types::ProgramMemoryRequirements::total_memory
    pub fn select_devices(&self, program: &CompiledProgram) -> Result<Vec<DeviceId>> {
        if self.devices.is_empty() {
            // Nothing enumerated at all: the caller
            // (`TPUBackend::execute_computation`) turns this into a
            // `DeviceError` with the context it has.
            return Ok(Vec::new());
        }

        let mut candidates: Vec<&TPUDevice> = self
            .devices
            .iter()
            .filter(|device| self.is_runnable(device))
            .collect();

        if candidates.is_empty() {
            return Err(OptimError::DeviceError(ErrorContext::new(format!(
                "all {} TPU device(s) are unavailable or below the minimum health score of {MIN_SELECTABLE_HEALTH_SCORE}",
                self.devices.len()
            ))));
        }

        self.order_candidates(&mut candidates);

        // `total_memory` of zero would otherwise select nothing at all; a
        // program still has to run somewhere, so treat it as needing one byte.
        let required = program.memory_requirements.total_memory.max(1);
        let mut selected = Vec::new();
        let mut covered: usize = 0;

        for device in &candidates {
            selected.push(device.id);
            covered = covered.saturating_add(device.memory_capacity);
            if covered >= required {
                return Ok(selected);
            }
        }

        Err(OptimError::DeviceError(ErrorContext::new(format!(
            "program needs {required} bytes but the {} usable TPU device(s) provide only {covered} bytes in total",
            candidates.len()
        ))))
    }

    /// Record which devices a computation was placed on, so a repeat execution
    /// of the same computation can be recognised as already placed.
    pub fn assign_devices(&mut self, computation_id: ComputationId, devices: Vec<DeviceId>) {
        self.device_assignments.insert(computation_id, devices);
    }

    /// Devices previously assigned to `computation_id`, if any.
    pub fn assigned_devices(&self, computation_id: ComputationId) -> Option<&[DeviceId]> {
        self.device_assignments
            .get(&computation_id)
            .map(Vec::as_slice)
    }

    /// Memory capacity of a device, if it is enumerated.
    pub fn device_capacity(&self, device: DeviceId) -> Option<usize> {
        self.devices
            .iter()
            .find(|candidate| candidate.id == device)
            .map(|candidate| candidate.memory_capacity)
    }

    /// Report observed load (0.0..=1.0) for a device, feeding both the
    /// utilization statistics and the load balancer that orders selection.
    pub fn record_device_load(&mut self, device: DeviceId, load: f64) {
        let clamped = load.clamp(0.0, 1.0);
        self.device_utilization.insert(device, clamped);
        self.load_balancer.device_loads.insert(device, clamped);
    }

    /// Whether a device is in a state that can accept work.
    fn is_runnable(&self, device: &TPUDevice) -> bool {
        if !matches!(
            device.status,
            DeviceStatus::Available | DeviceStatus::Busy | DeviceStatus::Maintenance
        ) {
            return false;
        }
        // A device with no health record has not been observed as degraded, so
        // it stays eligible; a recorded score below the floor excludes it.
        self.device_health
            .get(&device.id)
            .map(|health| health.health_score >= MIN_SELECTABLE_HEALTH_SCORE)
            .unwrap_or(true)
    }

    /// Current load for a device, preferring the load balancer's own record and
    /// falling back to the raw utilization map.
    fn device_load(&self, device: DeviceId) -> f64 {
        self.load_balancer
            .device_loads
            .get(&device)
            .or_else(|| self.device_utilization.get(&device))
            .copied()
            .unwrap_or(0.0)
    }

    /// Order the eligible devices according to the configured strategy. Every
    /// branch is a total order with the device id as the final tiebreak, so
    /// selection stays deterministic for a given state.
    fn order_candidates(&self, candidates: &mut [&TPUDevice]) {
        match self.load_balancer.strategy {
            // Enumeration order: devices are handed out starting from the
            // lowest id, which is exactly round-robin over a fresh pod.
            LoadBalancingStrategy::RoundRobin => candidates.sort_by_key(|device| device.id.0),
            LoadBalancingStrategy::LeastLoaded
            | LoadBalancingStrategy::Adaptive
            | LoadBalancingStrategy::WorkStealing => {
                candidates.sort_by(|a, b| {
                    self.device_load(a.id)
                        .total_cmp(&self.device_load(b.id))
                        .then_with(|| a.id.0.cmp(&b.id.0))
                });
            }
            LoadBalancingStrategy::PowerAware => {
                candidates.sort_by(|a, b| {
                    self.device_power(a.id)
                        .total_cmp(&self.device_power(b.id))
                        .then_with(|| a.id.0.cmp(&b.id.0))
                });
            }
            LoadBalancingStrategy::LocalityAware => {
                // Prefer devices that sit close to the pod origin in the real
                // topology, so a multi-device placement lands on neighbours
                // rather than opposite corners of the grid.
                let anchor = candidates.first().map(|device| device.id);
                candidates.sort_by_key(|device| {
                    (
                        anchor
                            .map(|anchor| self.hop_distance(anchor, device.id))
                            .unwrap_or(0),
                        device.id.0,
                    )
                });
            }
        }
    }

    /// Reported power draw for a device, or zero when unobserved.
    fn device_power(&self, device: DeviceId) -> f64 {
        self.device_health
            .get(&device)
            .map(|health| health.power_consumption)
            .unwrap_or(0.0)
    }

    /// Breadth-first hop count between two devices over the real interconnect
    /// adjacency. `usize::MAX` when they are not connected.
    fn hop_distance(&self, from: DeviceId, to: DeviceId) -> usize {
        if from == to {
            return 0;
        }
        let mut visited: HashMap<DeviceId, usize> = HashMap::new();
        let mut frontier = std::collections::VecDeque::new();
        visited.insert(from, 0);
        frontier.push_back(from);
        while let Some(current) = frontier.pop_front() {
            let depth = visited.get(&current).copied().unwrap_or(0);
            if let Some(neighbours) = self.topology.connections.get(&current) {
                for neighbour in neighbours {
                    if *neighbour == to {
                        return depth + 1;
                    }
                    if !visited.contains_key(neighbour) {
                        visited.insert(*neighbour, depth + 1);
                        frontier.push_back(*neighbour);
                    }
                }
            }
        }
        usize::MAX
    }
}

/// Build the real neighbour lists for a device set.
///
/// Devices carrying pod grid coordinates are connected to their four
/// orthogonal grid neighbours; a single-chip topology (no coordinates) is
/// connected as a chain along the enumeration order so locality still has a
/// defined meaning.
fn build_adjacency(devices: &[TPUDevice]) -> Vec<Vec<DeviceId>> {
    let coordinate_index: HashMap<(usize, usize), DeviceId> = devices
        .iter()
        .filter_map(|device| device.coordinates.map(|coord| (coord, device.id)))
        .collect();

    devices
        .iter()
        .enumerate()
        .map(|(index, device)| match device.coordinates {
            Some((row, col)) => {
                let mut neighbours = Vec::new();
                let candidates = [
                    row.checked_sub(1).map(|r| (r, col)),
                    Some((row + 1, col)),
                    col.checked_sub(1).map(|c| (row, c)),
                    Some((row, col + 1)),
                ];
                for candidate in candidates.into_iter().flatten() {
                    if let Some(neighbour) = coordinate_index.get(&candidate) {
                        neighbours.push(*neighbour);
                    }
                }
                neighbours
            }
            None => {
                let mut neighbours = Vec::new();
                if let Some(previous) = index.checked_sub(1) {
                    neighbours.push(devices[previous].id);
                }
                if index + 1 < devices.len() {
                    neighbours.push(devices[index + 1].id);
                }
                neighbours
            }
        })
        .collect()
}

/// Classify a link from its two endpoints' pod coordinates: neighbours within
/// the same grid row share a tray in this model (intra-chip), a neighbour in an
/// adjacent row crosses trays (inter-chip). Coordinate-less devices belong to a
/// single-chip topology, so their links are intra-chip by construction.
fn link_type_for(from: Option<(usize, usize)>, to: Option<(usize, usize)>) -> InterconnectType {
    match (from, to) {
        (Some((from_row, _)), Some((to_row, _))) if from_row != to_row => {
            InterconnectType::InterChip
        }
        _ => InterconnectType::IntraChip,
    }
}
