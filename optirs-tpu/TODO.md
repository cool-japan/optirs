# OptiRS TPU TODO (v0.3.2)

## Module Status: Working CPU-reference implementation (no vendor TPU runtime)

**Tests**: 171 tests passing (0 ignored); `cargo clippy --all-features --tests`
and `cargo doc --all-features` are both clean except one pre-existing,
unrelated test-only style note (no `unwrap`/`panic!` findings)
**Scope**: coordination, checkpointing, and an XLA-shaped compiler pipeline, all
running on a CPU reference executor. No Google Cloud TPU provisioning, no vendor
XLA/TPU runtime linkage (proprietary, not distributable as pure Rust).

---

## Completed: Core TPU infrastructure (real, tested, pure Rust)

### Optimizer integration
- [x] `TPUOptimizer<O, A>` wraps any `optirs_core::Optimizer` and drives a real
      compile → execute → profile pipeline (`tpu_step`)
- [x] `TPUOptimizer` itself implements `optirs_core::Optimizer<A, Ix1>`, so it can
      be used generically wherever an `Optimizer` is expected
- [x] Single-device and data-parallel (pod) execution paths

### XLA-shaped compiler pipeline (`xla`)
- [x] Graph capture with real producer/consumer dependency tracking
      (`add_operation` wires `metadata.producer`/`consumers`; `dependencies` is
      derived from it, not left empty)
- [x] `set_outputs` / `mark_terminal_operands_as_outputs` / `validate_computation`
      (rejects a computation with operations but no declared outputs)
- [x] Dead-code elimination with a fail-safe: refuses to run (rather than delete
      the whole graph) when no outputs are declared
- [x] Constant folding and common-subexpression elimination over a concrete,
      clonable, comparable `ConstantValue` payload
- [x] Kernel fusion with a real legality check (shape-compatible, single
      consumer, not an external output) and multi-output tuple materialization
- [x] Shape inference for reshape (validates element count, resolves at most one
      `-1` dimension), convolution (stride/dilation/padding-aware), dot, and
      broadcast
- [x] Memory allocator with `free`, region coalescing, and per-operand
      `live_range` (not a whole-program constant)
- [x] Cycle detection: iterative DFS plus Kahn's-algorithm and Tarjan-SCC
      cross-checks (`pod_coordination::synchronization::deadlock::graph`)

### Coordination (`coordination`)
- [x] `PodCoordinator`: device/channel topology keyed by `(source, target)`
      pairs, not source-only
- [x] `synchronize_devices`: synchronous barrier accounting over live device
      state, no `sleep`-and-report-success
- [x] Fault detection / performance monitoring register real, queryable
      in-struct state (`is_active`, `started_at`, `monitored_devices`)
- [x] Device capabilities come from `PodConfig::device_capabilities` (defaults
      to a TPU v4-shaped nominal spec), not a hardcoded literal independent of
      configuration

### Fault tolerance (`fault_tolerance`)
- [x] `create_checkpoint` serializes real coordination state to disk with a
      SHA-256 integrity hash; `restore_checkpoint` re-verifies that hash before
      applying anything
- [x] Recovery strategies (`Restart`/`Replicate`/`Rollback`/`Isolate`/`Graceful`)
      do real work through the checkpoint path; `migrate_workload` explicitly
      returns `Err` (it needs a TPU runtime this crate does not have) instead of
      claiming success

### Synchronization (`synchronization`)
- [x] Condvar barriers with a correctly-set predicate (no more indefinite block
      until timeout) and a configurable, honored timeout
- [x] Ring all-reduce / broadcast / reduce-scatter collectives

### Monitoring (`monitoring`)
- [x] Anomaly detection via a real z-score against tracked metric history
- [x] Report IDs are unique (monotonic sequence + timestamp), not a constant
      string; `overall_score` is derived from live health checks

---

## Out of scope for autonomous implementation

Every remaining optirs-tpu item requires Google Cloud TPU APIs, cloud IAM/networking, or live pod hardware. These are intentionally NOT auto-implemented — faking them would invent cloud/hardware behavior. (Note: generic Byzantine-robust aggregation and federated optimization already exist as pure-Rust logic in `optirs-core` — `privacy/federated/byzantine_aggregation.rs` and `distributed/fedprox.rs`; the items below are specifically the Cloud-TPU-pod integrations of those ideas.)

### Google Cloud Integration (Cloud TPU API / billing)
- [ ] Cloud TPU API integration (pod provisioning/deallocation)
- [ ] Preemptible/spot TPU handling
- [ ] Multi-region TPU coordination
- [ ] Cost tracking and optimization
- [ ] Automatic resource scaling against live quota

### Authentication and Security (cloud IAM / networking)
- [ ] Service account authentication
- [ ] OAuth2 flows
- [ ] VPC and firewall configuration
- [ ] Audit logging
- [ ] RBAC integration

### Advanced Scaling (live pod orchestration)
- [ ] Dynamic scaling algorithms against real utilization telemetry
- [ ] Cost-aware scaling decisions
- [ ] Predictive scaling
- [ ] Graceful scaling without interrupting live training

### Hardware-dependent fault tolerance
- [ ] Cross-device workload migration (`migrate_workload`) — needs a TPU
      runtime to quiesce and re-enqueue an executing program
- [ ] Automatic failure detection against real device telemetry (today's
      detector runs on heartbeats the caller supplies)
- [ ] Redundant computation across physically distinct pods

### Research Features (external hardware / cloud)
- [ ] TPU Edge integration
- [ ] Quantum-TPU hybrid optimization
- [ ] Federated learning across physically distributed pods

### Smaller, self-contained gaps — resolved
The six items formerly listed here are now done; kept as a short record
rather than deleted outright, since each was a real, previously-fabricated
behavior:
- [x] `pod_coordination::synchronization::clocks::protocols`: real NTP-style
      round-trip offset estimation (`NtpTimestamps`/`NtpPeer`/
      `NtpSynchronizer::estimate_offset`, RFC 5905's four-timestamp algorithm,
      with a physically-inconsistent exchange rejected via `Err` rather than
      averaged into a bogus offset — never a fake offset).
- [x] `pod_coordination::synchronization::clocks::utils::get_system_uptime`
      (hardcoded `Duration::from_secs(86400)`) renamed to `process_uptime`
      and implemented as a real `Instant`-anchored elapsed-time measurement
      (there is no portable pure-Rust OS process-start-time API).
- [x] `xla::backend::profiling_integration::ProfileExportManager::export_counter_data`
      / `export_trace_data` / `export_memory_data` now serialize the real
      data in the manager/collector/profiler they are given (via new
      `PerformanceCounterManager::record_sample` and
      `TraceCollector::record_event`, wired into real measured timings via
      `XLABackend::compile_and_integrate` -> `ProfilingIntegration::
      record_compile_timings`), rather than always writing a hardcoded empty
      body. Memory export stays honestly empty: nothing in this crate yet
      bridges real TPU memory-allocation events into this profiler.
- [x] `xla_compilation` (the legacy duplicate module) deleted; its one real
      usage, `ComputationId`, now lives in `xla::frontend::graph_capture`
      (the real, tested XLA pipeline) and is re-exported at both
      `xla::ComputationId` and `tpu_backend::ComputationId`.
- [x] The dead `coordination`/`xla`/`profiling` Cargo features (each gated
      zero `cfg(feature = ...)` code) were removed from `Cargo.toml` rather
      than retrofitted, since gating the modules they were named after would
      have meant cfg-ing out functionality nearly everything else in the
      crate depends on unconditionally.
- [x] `pod_coordination::coordination::coordinator::TPUPodCoordinator<T>` now
      delegates every method to the real, tested `coordination::PodCoordinator`
      (translating `PodCoordinationConfig` into `coordination::PodConfig` —
      see `PodCoordinationConfig::to_pod_config`'s doc comment for the
      mapping) instead of holding only its config field. The two dead,
      zero-consumer sibling types that sat in the same file (`Coordinator`,
      `CoordinationContext`) and the unused `performance_metrics` submodule
      were deleted rather than left as further empty shells.

---

## Testing Status

```
171 tests passing, 0 ignored (cargo nextest run -p optirs-tpu --all-features)
```

- [x] XLA compiler pipeline tests (graph capture, optimization passes, shape
      inference, kernel fusion, memory planning, scheduling)
- [x] Coordination tests (device init, barriers, load balancing, channel
      topology)
- [x] Fault-tolerance tests (checkpoint round-trip, integrity-check rejection)
- [x] Synchronization tests (barrier concurrency, collectives)

---

**Status**: Working CPU-reference implementation; hardware/cloud execution is
explicitly out of scope until a vendor runtime is available.
**Version**: v0.3.2
