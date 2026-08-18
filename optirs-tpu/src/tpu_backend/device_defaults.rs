//! Honest, derived default values for TPU device capabilities and health,
//! keyed off [`TPUVersion`]/[`PodTopology`] rather than hardcoded constants.
//!
//! The two optimization-level-keyed utilization estimates that used to live
//! here (`estimate_compute_utilization`/`estimate_bandwidth_utilization`) are
//! gone: [`super::backend::TPUBackend`] now reports the compute utilization
//! and memory bandwidth that the XLA performance analyzer and memory planner
//! actually derived from the compiled graph, which supersedes a table keyed
//! only off the optimization level.

use std::time::Instant;

use crate::{PodTopology, TPUVersion};

use super::types::{
    ComputeCapability, ComputeHealthStatus, DataType, DeviceHealthStatus,
    DevicePerformanceCharacteristics, MemoryHealthStatus, TPUFeature,
};

/// Approximate per-core high-bandwidth memory capacity (bytes) for a version.
pub(super) fn device_memory_capacity(version: TPUVersion) -> usize {
    let gib = 1024usize * 1024 * 1024;
    match version {
        TPUVersion::V2 => 8 * gib,
        TPUVersion::V3 => 16 * gib,
        TPUVersion::V4 => 32 * gib,
        TPUVersion::V5e => 16 * gib,
        TPUVersion::V5p => 95 * gib,
    }
}

/// Approximate peak compute characteristics for a version.
fn device_memory_bandwidth(version: TPUVersion) -> f64 {
    match version {
        TPUVersion::V2 => 600.0,
        TPUVersion::V3 => 900.0,
        TPUVersion::V4 => 1200.0,
        TPUVersion::V5e => 819.0,
        TPUVersion::V5p => 2765.0,
    }
}

/// Honest default compute capability for a version.
pub(super) fn device_compute_capability(version: TPUVersion) -> ComputeCapability {
    let tera = 1_000_000_000_000u64;
    let peak_flops = match version {
        TPUVersion::V2 => 45 * tera,
        TPUVersion::V3 => 123 * tera,
        TPUVersion::V4 => 275 * tera,
        TPUVersion::V5e => 197 * tera,
        TPUVersion::V5p => 459 * tera,
    };
    ComputeCapability {
        peak_flops,
        matrix_flops: peak_flops,
        memory_bandwidth_gb_s: device_memory_bandwidth(version),
        supported_dtypes: vec![
            DataType::F32,
            DataType::BF16,
            DataType::F16,
            DataType::I32,
            DataType::I8,
        ],
        max_dimensions: 8,
        features: vec![
            TPUFeature::MatrixUnits,
            TPUFeature::VectorUnits,
            TPUFeature::HighBandwidthMemory,
            TPUFeature::MixedPrecision,
        ],
    }
}

/// Approximate peak inter-device (interconnect) bandwidth in GB/s for a
/// version. Distinct from [`device_memory_bandwidth`], which describes the
/// on-package HBM path rather than the device-to-device fabric.
pub(super) fn device_interconnect_bandwidth(version: TPUVersion) -> f64 {
    match version {
        TPUVersion::V2 => 500.0,
        TPUVersion::V3 => 900.0,
        TPUVersion::V4 => 1200.0,
        TPUVersion::V5e => 1600.0,
        TPUVersion::V5p => 4800.0,
    }
}

/// Honest default performance characteristics for a version.
pub(super) fn device_performance_characteristics(
    version: TPUVersion,
) -> DevicePerformanceCharacteristics {
    DevicePerformanceCharacteristics {
        effective_memory_bandwidth: device_memory_bandwidth(version),
        compute_efficiency: 0.85,
        communication_latency_us: 1.0,
        thermal_threshold: 85.0,
    }
}

/// Device grid coordinates derived from the pod topology, or `None` for a
/// single-device topology.
pub(super) fn device_coordinates(topology: PodTopology, index: usize) -> Option<(usize, usize)> {
    let grid = match topology {
        PodTopology::Single => return None,
        PodTopology::Pod2x2 => 2,
        PodTopology::Pod4x4 => 4,
        PodTopology::Pod8x8 => 8,
        PodTopology::Pod16x16 => 16,
        PodTopology::Pod32x32 => 32,
    };
    Some((index / grid, index % grid))
}

/// Healthy default device-health record for a freshly enumerated device.
pub(super) fn default_device_health() -> DeviceHealthStatus {
    DeviceHealthStatus {
        health_score: 1.0,
        temperature: 40.0,
        power_consumption: 0.0,
        memory_health: MemoryHealthStatus {
            error_count: 0,
            bandwidth_efficiency: 1.0,
            fragmentation_ratio: 0.0,
        },
        compute_health: ComputeHealthStatus {
            matrix_unit_efficiency: 1.0,
            vector_unit_efficiency: 1.0,
            scalar_unit_efficiency: 1.0,
            instruction_cache_hit_rate: 1.0,
        },
        last_check: Instant::now(),
    }
}
