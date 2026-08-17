//! Consolidated unit-test suite for the `tpu_backend` module tree.
//!
//! Kept as a single `tests` submodule (rather than split per-subject)
//! per the file-split convention: tests may "move with their subject or
//! into a tests.rs submodule". Several production items are widened to
//! `pub(super)` (see each item's doc comment) purely so this sibling
//! submodule can reach them; none of that is part of the crate's public
//! API.

use std::time::{Duration, Instant};

use super::device_defaults::estimate_compute_utilization;
use super::serialization::{
    decode_ref_tensors, encode_program_binary, encode_ref_tensors, RefTensor,
};
use super::*;
use crate::{TPUVersion, XLAOptimizationLevel};

#[test]
fn test_tpu_backend_creation() {
    let config = TPUBackendConfig::default();
    let backend = TPUBackend::<f32>::new(config);
    assert!(backend.is_ok());
}

#[test]
fn test_tpu_buffer_creation() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let buffer = TPUBuffer::new(data, shape, MemoryLayout::RowMajor);

    assert_eq!(buffer.shape, vec![2, 2]);
    assert_eq!(buffer.data.len(), 4);
}

#[test]
fn test_device_health_status() {
    let health = DeviceHealthStatus {
        health_score: 0.95,
        temperature: 45.0,
        power_consumption: 150.0,
        memory_health: MemoryHealthStatus {
            error_count: 0,
            bandwidth_efficiency: 0.92,
            fragmentation_ratio: 0.05,
        },
        compute_health: ComputeHealthStatus {
            matrix_unit_efficiency: 0.88,
            vector_unit_efficiency: 0.90,
            scalar_unit_efficiency: 0.85,
            instruction_cache_hit_rate: 0.95,
        },
        last_check: Instant::now(),
    };

    assert!(health.health_score > 0.9);
    assert!(health.temperature < 50.0);
}

/// Build a minimal `CompiledProgram` for tests that only need a program
/// value (e.g. `select_devices`, which inspects device state, not the
/// program contents).
fn sample_program() -> CompiledProgram {
    CompiledProgram {
        binary: vec![0u8; 8],
        metadata: ProgramMetadata {
            compiled_at: Instant::now(),
            compiler_version: "test".to_string(),
            optimization_level: XLAOptimizationLevel::Standard,
            target_architecture: TPUVersion::V4,
            program_size: 8,
            output_specs: Vec::new(),
        },
        memory_requirements: ProgramMemoryRequirements {
            code_memory: 8,
            data_memory: 8,
            stack_memory: 8,
            scratch_memory: 8,
            total_memory: 32,
        },
        performance_characteristics: ProgramPerformanceCharacteristics {
            estimated_execution_time: Duration::from_micros(1),
            estimated_flops: 1,
            memory_bandwidth_utilization: 0.5,
            compute_utilization: 0.5,
        },
    }
}

#[tokio::test]
async fn test_execute_computation_returns_real_output() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let input = TPUBuffer::new(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryLayout::RowMajor,
    );
    let outputs = backend
        .execute_computation(ComputationId(1), vec![input])
        .await
        .expect("execution succeeds");

    // Non-empty, correct, shape-preserving identity output.
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![2, 2]);
    assert_eq!(outputs[0].data, vec![1.0f32, 2.0, 3.0, 4.0]);
}

#[tokio::test]
async fn test_execute_computation_multiple_buffers_roundtrip() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f64>::new(config).expect("backend");

    let a = TPUBuffer::new(vec![-1.5f64, 0.0, 2.25], vec![3], MemoryLayout::RowMajor);
    let b = TPUBuffer::new(vec![10.0f64, 20.0], vec![1, 2], MemoryLayout::RowMajor);
    let outputs = backend
        .execute_computation(ComputationId(42), vec![a, b])
        .await
        .expect("execution succeeds");

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].data, vec![-1.5f64, 0.0, 2.25]);
    assert_eq!(outputs[0].shape, vec![3]);
    assert_eq!(outputs[1].data, vec![10.0f64, 20.0]);
    assert_eq!(outputs[1].shape, vec![1, 2]);
}

#[test]
fn test_device_manager_populates_devices() {
    let config = TPUBackendConfig::default();
    let manager = DeviceManager::new(&config).expect("device manager");

    // num_cores defaults to 8, so eight honest device records exist.
    assert_eq!(manager.devices.len(), config.tpu_config.num_cores);
    assert!(!manager.devices.is_empty());
    assert_eq!(manager.device_health.len(), manager.devices.len());
    assert_eq!(manager.device_utilization.len(), manager.devices.len());

    let program = sample_program();
    let selected = manager.select_devices(&program).expect("select devices");
    assert!(!selected.is_empty());
}

#[test]
fn test_device_manager_always_has_at_least_one_device() {
    // `num_cores: 0` must still yield a usable device (the `.max(1)` floor),
    // documenting the precondition that keeps the `DeviceError` guard in
    // `execute_computation` from firing under normal configuration.
    let mut config = TPUBackendConfig::default();
    config.tpu_config.num_cores = 0;
    let manager = DeviceManager::new(&config).expect("device manager");
    assert_eq!(manager.devices.len(), 1);

    let program = sample_program();
    assert!(!manager
        .select_devices(&program)
        .expect("select devices")
        .is_empty());
}

#[test]
fn test_next_task_id_is_monotonic() {
    let config = TPUBackendConfig::default();
    let mut engine = ExecutionEngine::<f32>::new(&config).expect("engine");

    let a = engine.scheduler.next_task_id();
    let b = engine.scheduler.next_task_id();
    let c = engine.scheduler.next_task_id();

    assert_eq!(a, 0);
    assert!(b > a);
    assert!(c > b);
    assert_eq!((a, b, c), (0, 1, 2));
}

#[tokio::test]
async fn test_cache_hit_rate_reflects_lookups() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    // No lookups yet -> 0.0.
    assert_eq!(backend.get_cache_hit_rate(), 0.0);

    let input = TPUBuffer::new(vec![1.0f32], vec![1], MemoryLayout::RowMajor);

    // First execution with a fresh id is a miss (compiles + caches).
    backend
        .execute_computation(
            ComputationId(9),
            vec![TPUBuffer::new(
                input.data.clone(),
                input.shape.clone(),
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("first execution");
    assert_eq!(backend.get_cache_hit_rate(), 0.0);

    // Second execution with the same id is a hit -> hits=1, misses=1 => 0.5.
    backend
        .execute_computation(
            ComputationId(9),
            vec![TPUBuffer::new(
                input.data.clone(),
                input.shape.clone(),
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("second execution");
    assert_eq!(backend.get_cache_hit_rate(), 0.5);

    // The hit must have been served from cache: exactly one cached entry,
    // pinning "hit" to "served from cache" rather than merely "counter++".
    let cached = backend.compilation_cache.read().expect("cache read").len();
    assert_eq!(cached, 1);
}

#[test]
fn test_compute_utilization_is_derived_not_constant() {
    // Distinct optimization levels must produce distinct utilization
    // estimates; this is what proves the value is derived, not a constant.
    let none = estimate_compute_utilization(XLAOptimizationLevel::None);
    let aggressive = estimate_compute_utilization(XLAOptimizationLevel::Aggressive);
    assert!(none < aggressive);
    assert_ne!(none, aggressive);
}

#[test]
fn test_program_binary_is_deterministic_and_distinct() {
    let a = encode_program_binary(
        ComputationId(1),
        TPUVersion::V4,
        XLAOptimizationLevel::Aggressive,
    );
    let a2 = encode_program_binary(
        ComputationId(1),
        TPUVersion::V4,
        XLAOptimizationLevel::Aggressive,
    );
    let b = encode_program_binary(
        ComputationId(2),
        TPUVersion::V4,
        XLAOptimizationLevel::Aggressive,
    );

    assert!(!a.is_empty());
    assert_eq!(a, a2); // deterministic
    assert_ne!(a, b); // different programs differ
}

#[test]
fn test_ref_tensor_codec_roundtrip() {
    let tensors = vec![
        RefTensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        },
        RefTensor {
            shape: vec![3],
            data: vec![-1.0, 0.0, 7.5],
        },
    ];
    let encoded = encode_ref_tensors(&tensors);
    let decoded = decode_ref_tensors(&encoded).expect("decode");
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].shape, vec![2, 2]);
    assert_eq!(decoded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(decoded[1].shape, vec![3]);
    assert_eq!(decoded[1].data, vec![-1.0, 0.0, 7.5]);
}

#[test]
fn test_decode_malformed_payload_is_error_not_panic() {
    // Claims one tensor but truncates before the shape/data: must error.
    let malformed = vec![1u8, 0, 0, 0, 0xff, 0x00];
    assert!(decode_ref_tensors(&malformed).is_err());
    // Empty payload cannot even hold the tensor count.
    assert!(decode_ref_tensors(&[]).is_err());
}
