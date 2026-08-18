//! Consolidated unit-test suite for the `tpu_backend` module tree.
//!
//! Kept as a single `tests` submodule (rather than split per-subject)
//! per the file-split convention: tests may "move with their subject or
//! into a tests.rs submodule". Several production items are widened to
//! `pub(super)` (see each item's doc comment) purely so this sibling
//! submodule can reach them; none of that is part of the crate's public
//! API.

use std::time::{Duration, Instant};

use super::serialization::{
    decode_ref_tensors, encode_ref_tensors, serialize_tpu_buffers, RefTensor,
};
use super::*;
use crate::error::OptimError;
use crate::xla::frontend::{
    ComputationGraphBuilder, ConstantValue, OperationType, TensorShape, XLAComputation,
};
use crate::{TPUVersion, XLAOptimizationLevel};

/// A dense tensor shape over `dims`.
fn shape(dims: &[usize]) -> TensorShape {
    TensorShape {
        dimensions: dims.to_vec(),
        dynamic_dimensions: vec![false; dims.len()],
        element_count: dims.iter().product::<usize>().max(1),
        tuple_shapes: Vec::new(),
    }
}

/// The rank-0 shape used for scalar constants.
fn scalar_shape() -> TensorShape {
    shape(&[])
}

/// Build `param * factor` over a tensor of `dims`: a single-parameter,
/// single-output graph whose result is unmistakably *computed* rather than
/// echoed back, so the tests below actually pin the reference evaluation.
fn scaled_computation<T>(
    builder: &mut ComputationGraphBuilder<T>,
    name: &str,
    dims: &[usize],
    factor: f64,
) -> XLAComputation<T>
where
    T: scirs2_core::numeric::Float + std::fmt::Debug + Default + Clone + Send + Sync + 'static,
{
    let mut computation = builder.create_computation(name);

    let parameter = builder
        .add_operation(
            &mut computation,
            OperationType::Parameter,
            vec![],
            shape(dims),
        )
        .and_then(|op| operand_of(&computation, op))
        .expect("parameter operation must be addable");

    let constant = builder
        .add_operation(
            &mut computation,
            OperationType::Constant(ConstantValue::scalar(factor)),
            vec![],
            scalar_shape(),
        )
        .and_then(|op| operand_of(&computation, op))
        .expect("constant operation must be addable");

    builder
        .add_operation(
            &mut computation,
            OperationType::Multiply,
            vec![parameter, constant],
            shape(dims),
        )
        .expect("multiply operation must be addable");

    builder
        .mark_terminal_operands_as_outputs(&mut computation)
        .expect("the product is the graph's only terminal operand");

    computation
}

/// Build a graph that simply forwards `count` parameters, so a test can pin
/// multi-argument binding and multi-output decoding.
fn forwarding_computation<T>(
    builder: &mut ComputationGraphBuilder<T>,
    name: &str,
    shapes: &[Vec<usize>],
) -> XLAComputation<T>
where
    T: scirs2_core::numeric::Float + std::fmt::Debug + Default + Clone + Send + Sync + 'static,
{
    let mut computation = builder.create_computation(name);
    for dims in shapes {
        builder
            .add_operation(
                &mut computation,
                OperationType::Parameter,
                vec![],
                shape(dims),
            )
            .expect("parameter operation must be addable");
    }
    builder
        .mark_terminal_operands_as_outputs(&mut computation)
        .expect("every parameter is terminal in a forwarding graph");
    computation
}

fn operand_of<T>(
    computation: &XLAComputation<T>,
    op: crate::xla::frontend::OperationId,
) -> crate::error::Result<crate::xla::frontend::OperandId>
where
    T: scirs2_core::numeric::Float + std::fmt::Debug + Send + Sync + 'static,
{
    computation
        .operation_output(op)
        .ok_or_else(|| OptimError::from(format!("operation {op:?} produced no output operand")))
}

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
async fn test_execute_computation_evaluates_the_registered_graph() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let id = backend.register_computation(scaled_computation(
        &mut builder,
        "times_three",
        &[2, 2],
        3.0,
    ));

    let input = TPUBuffer::new(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![2, 2],
        MemoryLayout::RowMajor,
    );
    let outputs = backend
        .execute_computation(id, vec![input])
        .await
        .expect("execution succeeds");

    // The graph is `param * 3`, so an identity round-trip would fail here:
    // this pins that the registered operations were actually evaluated.
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![2, 2]);
    assert_eq!(outputs[0].data, vec![3.0f32, 6.0, 9.0, 12.0]);
}

#[tokio::test]
async fn test_execute_computation_binds_every_declared_parameter() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f64>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f64>::new();
    let id = backend.register_computation(forwarding_computation(
        &mut builder,
        "forward_two",
        &[vec![3], vec![1, 2]],
    ));

    let a = TPUBuffer::new(vec![-1.5f64, 0.0, 2.25], vec![3], MemoryLayout::RowMajor);
    let b = TPUBuffer::new(vec![10.0f64, 20.0], vec![1, 2], MemoryLayout::RowMajor);
    let outputs = backend
        .execute_computation(id, vec![a, b])
        .await
        .expect("execution succeeds");

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].data, vec![-1.5f64, 0.0, 2.25]);
    assert_eq!(outputs[0].shape, vec![3]);
    assert_eq!(outputs[1].data, vec![10.0f64, 20.0]);
    assert_eq!(outputs[1].shape, vec![1, 2]);
}

/// The backend refuses to invent a program for an id it was never given a
/// graph for, rather than synthesizing a descriptor binary as it once did.
#[tokio::test]
async fn executing_an_unregistered_computation_is_an_error() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let input = TPUBuffer::new(vec![1.0f32], vec![1], MemoryLayout::RowMajor);
    assert!(backend
        .execute_computation(ComputationId(1234), vec![input])
        .await
        .is_err());
    assert!(backend.compile(ComputationId(1234)).is_err());
}

/// End-to-end guard on the argument-binding contract: a graph carrying dead
/// operations and a duplicated subexpression goes through the full compile
/// pipeline (DCE and CSE both rewrite the operation list) and must still
/// evaluate correctly afterwards.
///
/// This is what pins `computation.inputs` to the surviving parameters. The two
/// parameters deliberately share a shape, so any rebinding that matched
/// operands by shape -- or any pass that dropped a parameter without the input
/// list following -- would silently swap the arguments and produce `9` instead
/// of `7`.
#[tokio::test]
async fn arguments_still_bind_correctly_after_the_optimization_pipeline() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let mut computation = builder.create_computation("optimized");

    // Two same-shaped parameters: `a` and `b`.
    let a = builder
        .add_operation(
            &mut computation,
            OperationType::Parameter,
            vec![],
            shape(&[1]),
        )
        .and_then(|op| operand_of(&computation, op))
        .expect("parameter a");
    let b = builder
        .add_operation(
            &mut computation,
            OperationType::Parameter,
            vec![],
            shape(&[1]),
        )
        .and_then(|op| operand_of(&computation, op))
        .expect("parameter b");

    // Dead subtree: nothing consumes this, so DCE must remove it.
    builder
        .add_operation(
            &mut computation,
            OperationType::Multiply,
            vec![b, b],
            shape(&[1]),
        )
        .expect("dead operation");

    // `a - b`, computed twice so CSE has a duplicate to merge.
    let difference = builder
        .add_operation(
            &mut computation,
            OperationType::Subtract,
            vec![a, b],
            shape(&[1]),
        )
        .and_then(|op| operand_of(&computation, op))
        .expect("difference");
    builder
        .add_operation(
            &mut computation,
            OperationType::Subtract,
            vec![a, b],
            shape(&[1]),
        )
        .expect("duplicate difference");

    builder
        .set_outputs(&mut computation, &[difference])
        .expect("declare the difference as the output");

    assert_eq!(
        computation.inputs.len(),
        2,
        "both parameters must be declared inputs before optimization"
    );
    let operations_before = computation.operations.len();
    assert_eq!(operations_before, 5);

    // Prove this test is not vacuous: the pipeline really does rewrite this
    // graph, so the assertions below are exercising post-optimization binding
    // rather than an untouched graph.
    let mut pipeline = crate::xla::optimization::OptimizationPipeline::<f32>::new(
        &crate::xla::XLACompilerConfig::default(),
    );
    let optimized = pipeline
        .optimize(computation.clone())
        .expect("the pipeline must accept this graph");
    assert!(
        optimized.operations.len() < operations_before,
        "expected DCE/CSE to remove operations, {} -> {}",
        operations_before,
        optimized.operations.len()
    );
    assert_eq!(
        optimized.inputs.len(),
        2,
        "both parameters must survive optimization and stay declared"
    );

    let id = backend.register_computation(computation);

    // `a - b` is deliberately non-commutative: 10 - 3 = 7, while a swapped
    // binding would give 3 - 10 = -7.
    let outputs = backend
        .execute_computation(
            id,
            vec![
                TPUBuffer::new(vec![10.0f32], vec![1], MemoryLayout::RowMajor),
                TPUBuffer::new(vec![3.0f32], vec![1], MemoryLayout::RowMajor),
            ],
        )
        .await
        .expect("execution succeeds");

    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].data,
        vec![7.0f32],
        "arguments must bind to the parameters they were declared for"
    );
}

/// Supplying the wrong number of arguments is reported, not silently padded.
#[tokio::test]
async fn argument_count_must_match_the_declared_parameters() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let id = backend.register_computation(forwarding_computation(
        &mut builder,
        "forward_two",
        &[vec![2], vec![2]],
    ));

    let only_one = TPUBuffer::new(vec![1.0f32, 2.0], vec![2], MemoryLayout::RowMajor);
    assert!(backend
        .execute_computation(id, vec![only_one])
        .await
        .is_err());
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
    assert_eq!(backend.compilation_cache_statistics().hits, 0);

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let id = backend.register_computation(scaled_computation(&mut builder, "cached", &[1], 2.0));

    // First execution is a miss (compiles + caches).
    backend
        .execute_computation(
            id,
            vec![TPUBuffer::new(
                vec![1.0f32],
                vec![1],
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("first execution");
    assert_eq!(backend.get_cache_hit_rate(), 0.0);
    assert_eq!(backend.compilation_cache_statistics().misses, 1);

    // Second execution of the same graph is a hit -> hits=1, misses=1 => 0.5.
    backend
        .execute_computation(
            id,
            vec![TPUBuffer::new(
                vec![1.0f32],
                vec![1],
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("second execution");

    // "Hit" means the compiler returned a cached binary, not merely that a
    // counter moved: there is exactly one compilation cache in the backend and
    // these are its own statistics.
    let stats = backend.compilation_cache_statistics();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(backend.get_cache_hit_rate(), 0.5);
}

/// The compiler is a deterministic function of the graph: the same graph
/// compiled by two independent backends yields byte-identical binaries, and a
/// different graph yields a different one. (Regression cover for the
/// descriptor-encoding stand-in this replaced, which was deterministic in the
/// *id* and blind to the program.)
#[test]
fn compilation_is_deterministic_in_the_graph_not_the_id() {
    let mut first = TPUBackend::<f32>::new(TPUBackendConfig::default()).expect("backend");
    let mut second = TPUBackend::<f32>::new(TPUBackendConfig::default()).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let same_a = scaled_computation(&mut builder, "same", &[4], 2.0);
    let mut builder_b = ComputationGraphBuilder::<f32>::new();
    let same_b = scaled_computation(&mut builder_b, "same", &[4], 2.0);
    let different = scaled_computation(&mut builder, "different", &[4096], 2.0);

    let id_a = first.register_computation(same_a);
    let id_b = second.register_computation(same_b);
    let id_different = first.register_computation(different);

    let a = first.compile(id_a).expect("compile a");
    let b = second.compile(id_b).expect("compile b");
    let c = first.compile(id_different).expect("compile c");

    assert!(!a.binary.is_empty());
    assert_eq!(a.binary, b.binary, "identical graphs must compile alike");
    assert_ne!(a.binary, c.binary, "different graphs must differ");
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

// `TPUBackend::compile_program` used to encode a fixed-width program
// descriptor and could therefore only ever report `estimated_flops: 0` and
// `estimated_execution_time: ZERO` -- honestly absent, but absent. Now that
// compilation goes through the real XLA pipeline the backend holds an actual
// operation list and tensor shapes, so both figures are measured. This pins
// that they are *derived*: a graph over a thousand times more elements must
// report proportionally more FLOPs and more memory, which no constant can do.
#[test]
fn compiled_program_cost_is_derived_from_the_graph() {
    let mut backend = TPUBackend::<f32>::new(TPUBackendConfig::default()).expect("backend");
    let mut builder = ComputationGraphBuilder::<f32>::new();

    let small = backend.register_computation(scaled_computation(&mut builder, "small", &[4], 2.0));
    let large =
        backend.register_computation(scaled_computation(&mut builder, "large", &[64, 64], 2.0));

    let small = backend.compile(small).expect("compile small");
    let large = backend.compile(large).expect("compile large");

    assert!(
        small.performance_characteristics.estimated_flops > 0,
        "a graph with real arithmetic must report real FLOPs"
    );
    assert!(
        large.performance_characteristics.estimated_flops
            > small.performance_characteristics.estimated_flops,
        "FLOPs ({} vs {}) must scale with the graph's element count",
        large.performance_characteristics.estimated_flops,
        small.performance_characteristics.estimated_flops
    );
    assert!(
        large.memory_requirements.data_memory > small.memory_requirements.data_memory,
        "the planned data footprint must scale with the tensors actually placed"
    );
    assert!(
        small.performance_characteristics.estimated_execution_time > Duration::ZERO,
        "a non-empty program has a non-zero time estimate"
    );
    assert!(small.memory_requirements.code_memory > 0);
    assert_eq!(
        small.memory_requirements.total_memory,
        small.memory_requirements.code_memory + small.memory_requirements.data_memory
    );
    // The metadata really describes the configured target, not a placeholder.
    assert_eq!(small.metadata.target_architecture, TPUVersion::V4);
    assert_eq!(
        small.metadata.optimization_level,
        TPUBackendConfig::default()
            .tpu_config
            .xla_optimization_level
    );
    assert_eq!(small.metadata.output_specs.len(), 1);
}

// ---------------------------------------------------------------------------
// Device selection now reads the program it is given
// ---------------------------------------------------------------------------

/// A program whose footprint exceeds a single device must be spread across
/// several; a footprint that fits stays on one. This is what pins
/// `select_devices` to the program's real memory requirement rather than
/// "always device 0".
#[test]
fn select_devices_scales_with_program_memory_requirements() {
    let config = TPUBackendConfig::default();
    let manager = DeviceManager::new(&config).expect("device manager");
    let per_device = manager
        .device_capacity(DeviceId(0))
        .expect("device 0 is enumerated");

    let mut small = sample_program();
    small.memory_requirements.total_memory = per_device / 4;
    assert_eq!(
        manager.select_devices(&small).expect("small fits").len(),
        1,
        "a program smaller than one device must not be spread"
    );

    let mut large = sample_program();
    large.memory_requirements.total_memory = per_device * 3 + 1;
    let selected = manager.select_devices(&large).expect("large fits the pod");
    assert_eq!(
        selected.len(),
        4,
        "a program needing three-plus devices' capacity must select four"
    );
}

/// A program larger than the whole pod is an honest error, not a selection that
/// would fail later at allocation time.
#[test]
fn select_devices_rejects_a_program_larger_than_the_pod() {
    let config = TPUBackendConfig::default();
    let manager = DeviceManager::new(&config).expect("device manager");

    let mut program = sample_program();
    program.memory_requirements.total_memory = usize::MAX;
    assert!(manager.select_devices(&program).is_err());
}

/// Unhealthy devices drop out of selection entirely.
#[test]
fn select_devices_skips_devices_below_the_health_floor() {
    let mut config = TPUBackendConfig::default();
    config.tpu_config.num_cores = 2;
    let mut manager = DeviceManager::new(&config).expect("device manager");

    if let Some(health) = manager.device_health.get_mut(&DeviceId(0)) {
        health.health_score = 0.1;
    }

    let program = sample_program();
    let selected = manager
        .select_devices(&program)
        .expect("device 1 is healthy");
    assert_eq!(selected, vec![DeviceId(1)]);
}

/// The least-loaded strategy really consults the recorded loads.
#[test]
fn select_devices_prefers_the_least_loaded_device() {
    let mut config = TPUBackendConfig::default();
    config.tpu_config.num_cores = 3;
    config.load_balancing_strategy = LoadBalancingStrategy::LeastLoaded;
    let mut manager = DeviceManager::new(&config).expect("device manager");

    manager.record_device_load(DeviceId(0), 0.9);
    manager.record_device_load(DeviceId(1), 0.8);
    manager.record_device_load(DeviceId(2), 0.1);

    let program = sample_program();
    let selected = manager.select_devices(&program).expect("selection");
    assert_eq!(selected.first().copied(), Some(DeviceId(2)));
}

/// Round-robin ignores load and hands out devices in enumeration order, which
/// is what distinguishes the two strategies.
#[test]
fn round_robin_ignores_recorded_load() {
    let mut config = TPUBackendConfig::default();
    config.tpu_config.num_cores = 3;
    config.load_balancing_strategy = LoadBalancingStrategy::RoundRobin;
    let mut manager = DeviceManager::new(&config).expect("device manager");

    manager.record_device_load(DeviceId(0), 0.99);
    manager.record_device_load(DeviceId(2), 0.0);

    let program = sample_program();
    let selected = manager.select_devices(&program).expect("selection");
    assert_eq!(selected.first().copied(), Some(DeviceId(0)));
}

/// The pod grid produces a real interconnect: neighbours, links and bandwidth,
/// rather than the empty topology the manager used to carry.
#[test]
fn device_manager_builds_a_real_interconnect() {
    let config = TPUBackendConfig::default();
    let manager = DeviceManager::new(&config).expect("device manager");

    assert!(
        !manager.topology.connections.is_empty(),
        "the pod grid must yield adjacency"
    );
    assert!(
        !manager.topology.bandwidth_matrix.is_empty(),
        "every link must carry a bandwidth"
    );
    for device in &manager.devices {
        assert!(
            !device.interconnect_links.is_empty(),
            "device {} has no links",
            device.id.0
        );
        for link in &device.interconnect_links {
            assert!(link.bandwidth_gb_s > 0.0);
        }
    }
}

/// Placements are recorded, so a repeat execution can be recognised.
#[tokio::test]
async fn execute_computation_records_the_device_assignment() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let id = backend.register_computation(scaled_computation(&mut builder, "placed", &[2], 2.0));

    backend
        .execute_computation(
            id,
            vec![TPUBuffer::new(
                vec![1.0f32, 2.0],
                vec![2],
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("execution succeeds");

    let stats = backend.get_performance_statistics();
    assert_eq!(stats.total_executions, 1);
    assert!(stats.average_execution_time > Duration::ZERO);
}

// ---------------------------------------------------------------------------
// Memory manager: real pools, real admission control, real release
// ---------------------------------------------------------------------------

/// Allocation reserves the program's real footprint and release returns it, so
/// the pools do not drift upward across executions.
#[test]
fn allocate_and_release_move_real_bytes() {
    let config = TPUBackendConfig::default();
    let mut manager = TPUMemoryManager::<f32>::new(&config).expect("memory manager");
    assert_eq!(manager.get_utilization_stats(), 0.0);

    let mut program = sample_program();
    program.memory_requirements.total_memory = 4096;

    let allocation = manager
        .allocate_for_computation(&program, &[DeviceId(0), DeviceId(1)])
        .expect("allocation must succeed");
    assert_eq!(allocation.total_allocated, 4096);
    assert_eq!(allocation.device_allocations.len(), 2);
    assert!(manager.get_utilization_stats() > 0.0);
    assert_eq!(manager.usage_statistics().total_allocated, 4096);

    let released = manager.release_allocation(&allocation);
    assert_eq!(released, 4096);
    assert_eq!(manager.get_utilization_stats(), 0.0);
}

/// Release frees exactly the blocks an allocation reserved, not merely blocks
/// of matching size. Two equal-sized allocations are live on one device; after
/// releasing the first, the second must still hold its bytes.
#[test]
fn release_frees_the_blocks_this_allocation_reserved() {
    let config = TPUBackendConfig::default();
    let mut manager = TPUMemoryManager::<f32>::new(&config).expect("memory manager");

    let mut program = sample_program();
    program.memory_requirements.total_memory = 4096;

    let first = manager
        .allocate_for_computation(&program, &[DeviceId(0)])
        .expect("first allocation");
    let second = manager
        .allocate_for_computation(&program, &[DeviceId(0)])
        .expect("second allocation");

    // Distinct blocks, at distinct addresses, on the same device.
    assert_eq!(first.reservations.len(), 1);
    assert_eq!(second.reservations.len(), 1);
    assert_ne!(
        first.reservations[0].address,
        second.reservations[0].address
    );
    assert_ne!(first.reservations[0].handle, second.reservations[0].handle);

    assert_eq!(manager.release_allocation(&first), 4096);
    assert_eq!(
        manager.usage_statistics().total_allocated,
        4096,
        "the surviving allocation must still hold its bytes"
    );

    assert_eq!(manager.release_allocation(&second), 4096);
    assert_eq!(manager.get_utilization_stats(), 0.0);
}

/// A footprint no device can hold is refused, and the refusal is recorded in
/// the success rate rather than silently returning a zero-byte allocation.
#[test]
fn allocation_beyond_capacity_is_refused() {
    let mut config = TPUBackendConfig::default();
    config.tpu_config.num_cores = 1;
    let mut manager = TPUMemoryManager::<f32>::new(&config).expect("memory manager");

    let mut program = sample_program();
    program.memory_requirements.total_memory = usize::MAX / 2;

    assert!(manager
        .allocate_for_computation(&program, &[DeviceId(0)])
        .is_err());
    assert_eq!(manager.usage_statistics().allocation_success_rate, 0.0);
    assert_eq!(manager.get_utilization_stats(), 0.0);
}

/// Allocating without a device is a configuration error, not an empty success.
#[test]
fn allocation_without_devices_is_an_error() {
    let config = TPUBackendConfig::default();
    let mut manager = TPUMemoryManager::<f32>::new(&config).expect("memory manager");
    let program = sample_program();
    assert!(manager.allocate_for_computation(&program, &[]).is_err());
}

/// Repeated executions must not exhaust the pools: the reservation made for one
/// execution is released before the next begins.
#[tokio::test]
async fn repeated_executions_do_not_leak_device_memory() {
    let config = TPUBackendConfig::default();
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let ids: Vec<ComputationId> = (0..8)
        .map(|index| {
            backend.register_computation(scaled_computation(
                &mut builder,
                &format!("leak_{index}"),
                &[3],
                index as f64 + 1.0,
            ))
        })
        .collect();

    for id in ids {
        backend
            .execute_computation(
                id,
                vec![TPUBuffer::new(
                    vec![1.0f32, 2.0, 3.0],
                    vec![3],
                    MemoryLayout::RowMajor,
                )],
            )
            .await
            .expect("execution succeeds");
    }

    let stats = backend.get_performance_statistics();
    assert_eq!(stats.total_executions, 8);
    assert_eq!(
        stats.memory_utilization, 0.0,
        "every reservation must have been released"
    );
    assert_eq!(stats.error_rate, 0.0);
}

// ---------------------------------------------------------------------------
// Memory profiling: the run side really reports into the compile-side profile
// ---------------------------------------------------------------------------

/// Executing a program records its real device-memory reservations and their
/// release into the profiling integration the compiler opened, so an exported
/// memory profile carries actual allocator events instead of empty arrays.
#[tokio::test]
async fn execution_records_real_memory_events_into_the_profile() {
    let config = TPUBackendConfig::default();
    assert!(
        config.enable_performance_monitoring,
        "this test relies on profiling being on by default"
    );
    let mut backend = TPUBackend::<f32>::new(config).expect("backend");

    // Nothing has run yet, so the profile is honestly empty.
    assert_eq!(backend.profiling().memory_profiler().recorded_events(), 0);

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let id = backend.register_computation(scaled_computation(&mut builder, "profiled", &[8], 2.0));

    backend
        .execute_computation(
            id,
            vec![TPUBuffer::new(
                vec![1.0f32; 8],
                vec![8],
                MemoryLayout::RowMajor,
            )],
        )
        .await
        .expect("execution succeeds");

    let profiler = backend.profiling().memory_profiler();
    // One reserve plus one release per device the program was placed on.
    assert!(
        profiler.recorded_events() >= 2,
        "expected reserve and release events, saw {}",
        profiler.recorded_events()
    );

    let stats = profiler
        .tracking_stats(&format!("computation_{}", id.0))
        .expect("the execution must have opened a tracking session");
    assert!(stats.total_allocations > 0);
    assert_eq!(
        stats.total_allocations, stats.total_deallocations,
        "every reservation this execution made must have been released"
    );
    assert_eq!(stats.current_allocations, 0);
    assert_eq!(stats.current_memory_usage, 0);
    assert!(
        stats.peak_memory_usage > 0,
        "the peak must reflect the bytes actually reserved"
    );

    // A snapshot must be taken while the reservations are still live. A
    // snapshot of the post-release state records nothing but zeros, which would
    // make the exported memory profile structurally empty however much real
    // allocation happened.
    assert!(
        profiler
            .usage_snapshots()
            .iter()
            .any(|snapshot| snapshot.total_usage > 0 && !snapshot.regions.is_empty()),
        "expected a snapshot taken at peak occupancy, saw only empty ones"
    );
}

// ---------------------------------------------------------------------------
// Execution budget
// ---------------------------------------------------------------------------

/// A zero budget means "no budget": the check is disabled rather than firing on
/// every task.
#[test]
fn a_zero_budget_disables_the_check() {
    let config = TPUBackendConfig {
        execution_timeout_ms: 0,
        ..Default::default()
    };
    let engine = ExecutionEngine::<f32>::new(&config).expect("engine");
    assert_eq!(engine.execution_timeout(), Duration::ZERO);

    let mut builder = ComputationGraphBuilder::<f32>::new();
    let computation = scaled_computation(&mut builder, "budgetless", &[4], 2.0);

    let buffer = TPUBuffer::new(vec![1.0f32; 4], vec![4], MemoryLayout::RowMajor);
    let input_data = serialize_tpu_buffers(&[buffer]).expect("serialize");
    let task = ComputationTask {
        task_id: TaskId(0),
        computation_id: computation.id,
        input_data,
        expected_outputs: Vec::new(),
    };
    let allocation = MemoryAllocation::default();
    assert!(engine
        .execute_task(task, &computation, &[DeviceId(0)], &allocation)
        .is_ok());
}

/// A budget the payload cannot possibly meet produces a real timeout error
/// rather than a result the caller has already given up on. The payload is
/// deliberately large (two million elements encoded and decoded) so a 1 ms
/// budget is exceeded by orders of magnitude on any machine.
#[test]
fn an_unmeetable_budget_reports_a_timeout() {
    let config = TPUBackendConfig {
        execution_timeout_ms: 1,
        ..Default::default()
    };
    let engine = ExecutionEngine::<f32>::new(&config).expect("engine");

    let elements = 2_000_000usize;
    let mut builder = ComputationGraphBuilder::<f32>::new();
    let computation = scaled_computation(&mut builder, "over_budget", &[elements], 2.0);

    let buffer = TPUBuffer::new(
        vec![1.5f32; elements],
        vec![elements],
        MemoryLayout::RowMajor,
    );
    let input_data = serialize_tpu_buffers(&[buffer]).expect("serialize");
    let task = ComputationTask {
        task_id: TaskId(7),
        computation_id: computation.id,
        input_data,
        expected_outputs: Vec::new(),
    };
    let allocation = MemoryAllocation::default();

    let error = engine
        .execute_task(task, &computation, &[DeviceId(0)], &allocation)
        .expect_err("a 1 ms budget cannot cover a two-million-element payload");
    assert!(
        matches!(TPUErrorHandler::classify(&error), ErrorType::TimeoutError),
        "an over-budget task must be classified as a timeout so the retry policy can fire"
    );
}

/// The configured budget really reaches the engine.
#[test]
fn execution_timeout_comes_from_configuration() {
    let config = TPUBackendConfig {
        execution_timeout_ms: 1234,
        ..Default::default()
    };
    let engine = ExecutionEngine::<f32>::new(&config).expect("engine");
    assert_eq!(engine.execution_timeout(), Duration::from_millis(1234));
}

// ---------------------------------------------------------------------------
// Performance monitor: real sample history
// ---------------------------------------------------------------------------

/// An enabled monitor retains a first sample and keeps the always-on summary
/// in step with it; a disabled monitor keeps the summary but no samples.
#[test]
fn performance_monitor_retains_samples_only_when_enabled() {
    let results = TaskExecutionResult {
        task_id: TaskId(1),
        execution_time: Duration::from_micros(250),
        memory_used: 1024,
        energy_consumed: 1.5,
        output_data: vec![0u8; 512],
    };
    let utilization = ExecutionUtilization {
        device: 0.25,
        memory: 0.5,
    };

    let config = TPUBackendConfig::default();
    let mut enabled = PerformanceMonitor::new(&config);
    assert!(enabled.is_enabled());
    enabled.record_execution(
        ComputationId(3),
        Duration::from_micros(400),
        &results,
        utilization,
    );
    assert_eq!(enabled.total_executions, 1);
    assert_eq!(enabled.average_execution_time, Duration::from_micros(400));
    assert_eq!(enabled.performance_history().len(), 1);
    let sample = &enabled.performance_history()[0];
    assert_eq!(sample.computation, ComputationId(3));
    assert_eq!(sample.execution_time, Duration::from_micros(250));
    assert!(sample.throughput > 0.0);
    assert_eq!(sample.device_utilization, 0.25);
    assert_eq!(sample.memory_utilization, 0.5);

    // The collection interval decimates: a second immediate execution updates
    // the summary but is not retained.
    enabled.record_execution(
        ComputationId(3),
        Duration::from_micros(600),
        &results,
        utilization,
    );
    assert_eq!(enabled.total_executions, 2);
    assert_eq!(enabled.average_execution_time, Duration::from_micros(500));
    assert_eq!(enabled.performance_history().len(), 1);

    enabled.flush_metrics().expect("flush must succeed");
    assert!(enabled.performance_history().is_empty());
    assert_eq!(
        enabled.total_executions, 2,
        "flushing detail must not erase the summary"
    );

    let off_config = TPUBackendConfig {
        enable_performance_monitoring: false,
        ..Default::default()
    };
    let mut disabled = PerformanceMonitor::new(&off_config);
    assert!(!disabled.is_enabled());
    disabled.record_execution(
        ComputationId(3),
        Duration::from_micros(400),
        &results,
        utilization,
    );
    assert_eq!(disabled.total_executions, 1);
    assert!(disabled.performance_history().is_empty());
}

// ---------------------------------------------------------------------------
// Error handler: the recovery policy is real
// ---------------------------------------------------------------------------

/// Retryable classes are retried within the configured budget; classes the
/// policy does not mark retryable are surfaced immediately.
#[test]
fn error_handler_retries_only_the_classes_its_policy_marks_retryable() {
    use scirs2_core::error::ErrorContext;

    let config = TPUBackendConfig {
        max_retry_attempts: 3,
        ..Default::default()
    };
    let mut handler = TPUErrorHandler::new(&config);
    assert_eq!(handler.max_attempts(), 3);

    let timeout = OptimError::TimeoutError(ErrorContext::new("late".to_string()));
    assert!(handler.record_error(&timeout, 0));
    assert!(handler.record_error(&timeout, 1));
    assert!(
        !handler.record_error(&timeout, 2),
        "the attempt budget must be respected"
    );

    // A memory error maps to `Fallback`, never a retry loop.
    let oom = OptimError::MemoryError(ErrorContext::new("full".to_string()));
    assert!(!handler.record_error(&oom, 0));

    let stats = handler.error_statistics();
    assert_eq!(stats.total_errors, 4);
    assert_eq!(stats.errors_by_type.get(&ErrorType::TimeoutError), Some(&3));
    assert_eq!(stats.errors_by_type.get(&ErrorType::MemoryError), Some(&1));

    handler.record_recovery_success();
    assert!(handler.error_statistics().recovery_success_rate > 0.0);
}

/// Recovery switched off means one attempt, whatever the class.
#[test]
fn error_recovery_can_be_switched_off() {
    use scirs2_core::error::ErrorContext;

    let config = TPUBackendConfig {
        enable_error_recovery: false,
        max_retry_attempts: 5,
        ..Default::default()
    };
    let mut handler = TPUErrorHandler::new(&config);
    assert_eq!(handler.max_attempts(), 1);

    let timeout = OptimError::TimeoutError(ErrorContext::new("late".to_string()));
    assert!(!handler.record_error(&timeout, 0));
}
