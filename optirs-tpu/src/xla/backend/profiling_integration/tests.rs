//! Unit-test suite for the `profiling_integration` module tree, split out
//! per the 2000-line file-size convention. A child module sees its parent's
//! private items, so no visibility widening was needed for this split.

use super::*;

#[test]
fn test_profiling_integration_creation() {
    use super::super::{
        super::super::PodTopology, super::TPUConfig, super::TPUVersion, BackendConfig,
    };

    let tpu_config = TPUConfig {
        tpu_version: TPUVersion::V4,
        num_cores: 8,
        enable_xla: true,
        xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
        mixed_precision: true,
        batch_size_per_core: 32,
        enable_pod_coordination: false,
        pod_topology: PodTopology::Pod2x2,
        memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
        gradient_compression: true,
        prefetch_depth: 2,
        experimental_features: false,
    };

    let backend_config = BackendConfig {
        target_tpu: tpu_config,
        enable_optimized_codegen: true,
        enable_profiling: true,
        debug_mode: false,
        verification_mode: false,
        custom_options: std::collections::HashMap::new(),
    };

    let profiling: ProfilingIntegration<f32> = ProfilingIntegration::new(&backend_config);
    assert_eq!(profiling.profiling_stats.samples_collected, 0);
    assert!(profiling.config.enable_perf_counters);
}

#[test]
fn test_counter_manager_creation() {
    let counter_manager = PerformanceCounterManager::new();
    assert!(!counter_manager.available_counters.is_empty());
    assert!(counter_manager
        .available_counters
        .contains_key("matrix_ops"));
    assert!(counter_manager
        .available_counters
        .contains_key("memory_bandwidth"));
}

/// A [`ProfilingConfig`] pointed at a fresh, process- and time-unique
/// directory under [`std::env::temp_dir`], so export tests never collide
/// with each other, with a previous run, or with the crate's own hardcoded
/// `/tmp/scirs_profiles` default.
fn temp_profiling_config() -> ProfilingConfig {
    let dir = std::env::temp_dir().join(format!(
        "optirs_tpu_profiling_test_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system clock is after the epoch")
            .as_nanos()
    ));

    ProfilingConfig {
        enable_perf_counters: true,
        enable_trace_collection: true,
        enable_memory_profiling: true,
        enable_power_profiling: false,
        enable_timeline_profiling: false,
        sampling_rate: 1000,
        max_trace_buffer_mb: 1,
        output_directory: dir.to_string_lossy().to_string(),
        export_format: ExportFormat::JSON,
        detailed_mode: false,
    }
}

// Regression test for the counter-recording fabrication: before
// `record_sample` existed, nothing could ever populate `counter_data`, so
// `TimeSeriesStats` could never reflect anything but its all-zero default.
// This asserts the statistics are real, computed values -- not the default,
// and specifically NOT round/fabricated-looking numbers a stub might return.
#[test]
fn record_sample_computes_real_statistics_not_fabricated_numbers() {
    let mut manager = PerformanceCounterManager::new();
    manager.record_sample("session", "codegen_time_us", CounterValue::Integer(100));
    manager.record_sample("session", "codegen_time_us", CounterValue::Integer(300));

    let data = manager
        .counter_data
        .read()
        .expect("lock is never poisoned in a single-threaded test");
    let series = data
        .get("codegen_time_us")
        .expect("record_sample must have created this counter's time series");

    assert_eq!(series.statistics.sample_count, 2);
    assert_eq!(series.statistics.min, 100.0);
    assert_eq!(series.statistics.max, 300.0);
    assert_eq!(series.statistics.average, 200.0);
    assert!(
        series.statistics.std_dev > 0.0,
        "two distinct samples must have nonzero spread, got std_dev={}",
        series.statistics.std_dev
    );

    let session = manager
        .active_sessions
        .get("session")
        .expect("record_sample must auto-create the session");
    assert_eq!(session.sample_buffer.len(), 2);
}

// Regression test for the trace-recording fabrication: before
// `record_event` existed, `trace_buffer.events` could never be populated.
#[test]
fn record_event_stores_a_real_event_with_a_real_computed_id() {
    let config = temp_profiling_config();
    let mut collector = TraceCollector::new(&config);

    collector.record_event(TraceEvent {
        id: 999, // must be overwritten by the real running counter
        timestamp: Instant::now(),
        event_type: EventType::FunctionCall,
        phase: EventPhase::Complete,
        thread_id: None,
        process_id: None,
        name: "unit_test_event".to_string(),
        category: "test".to_string(),
        duration: Some(Duration::from_millis(7)),
        args: HashMap::new(),
        stack_trace: None,
    });

    let buffer = collector
        .trace_buffer
        .lock()
        .expect("lock is never poisoned in a single-threaded test");
    assert_eq!(buffer.events.len(), 1);
    assert_eq!(
        buffer.events[0].id, 0,
        "the first recorded event's id must come from the real running counter, \
         not the caller-supplied placeholder"
    );
    assert_eq!(buffer.events[0].name, "unit_test_event");
    assert_eq!(buffer.stats.events_written, 1);
}

// Regression test for F8/profiling_integration's core fabrication:
// `export_counter_data` used to ignore its `counter_manager` argument
// (`_`-prefixed) and always write a hardcoded `{"counters": [], ...}` body.
// This asserts the exported file genuinely contains the real recorded name
// and value.
#[test]
fn export_counter_data_writes_the_real_recorded_samples_not_a_hardcoded_empty_body() {
    let config = temp_profiling_config();
    let mut manager = PerformanceCounterManager::new();
    manager.record_sample("session", "codegen_time_us", CounterValue::Integer(12345));

    let mut exporter = ProfileExportManager::new(&config);
    let path = exporter
        .export_counter_data(&manager)
        .expect("exporting real counter data must succeed");
    let contents = std::fs::read_to_string(&path).expect("the exporter must have written a file");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&config.output_directory);

    assert!(
        contents.contains("codegen_time_us"),
        "exported JSON must contain the real counter name, got: {contents}"
    );
    assert!(
        contents.contains("12345"),
        "exported JSON must contain the real recorded value, got: {contents}"
    );
    assert!(
        !contents.contains("\"counters\": []"),
        "must not be the old hardcoded empty body"
    );
}

// Same defect, for `export_trace_data`.
#[test]
fn export_trace_data_writes_the_real_recorded_events_not_a_hardcoded_empty_body() {
    let config = temp_profiling_config();
    let mut collector = TraceCollector::new(&config);
    collector.record_event(TraceEvent {
        id: 0,
        timestamp: Instant::now(),
        event_type: EventType::FunctionCall,
        phase: EventPhase::Complete,
        thread_id: None,
        process_id: None,
        name: "codegen".to_string(),
        category: "compile".to_string(),
        duration: Some(Duration::from_micros(500)),
        args: HashMap::new(),
        stack_trace: None,
    });

    let mut exporter = ProfileExportManager::new(&config);
    let path = exporter
        .export_trace_data(&collector)
        .expect("exporting real trace data must succeed");
    let contents = std::fs::read_to_string(&path).expect("the exporter must have written a file");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&config.output_directory);

    assert!(
        contents.contains("\"codegen\""),
        "exported JSON must contain the real event name, got: {contents}"
    );
    assert!(
        contents.contains("500"),
        "exported JSON must contain the real duration, got: {contents}"
    );
    assert!(
        !contents.contains("\"traceEvents\": []"),
        "must not be the old hardcoded empty body"
    );
}

// The fix must not go too far the other way and fabricate non-empty data:
// with nothing recorded, the export must honestly stay empty.
#[test]
fn export_counter_data_is_honestly_empty_when_nothing_was_recorded() {
    let config = temp_profiling_config();
    let manager = PerformanceCounterManager::new();
    let mut exporter = ProfileExportManager::new(&config);

    let path = exporter
        .export_counter_data(&manager)
        .expect("exporting must succeed even with nothing recorded");
    let contents = std::fs::read_to_string(&path).expect("file must exist");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&config.output_directory);

    assert!(
        contents.contains("\"counters\": []"),
        "genuinely empty state must honestly export as empty, got: {contents}"
    );
}

// End-to-end regression test: `compile_and_integrate`'s real measured
// codegen/runtime-integration durations must actually reach the profiler
// through `record_compile_timings`, not just create empty sessions.
#[test]
fn record_compile_timings_feeds_real_measured_durations_into_the_profiler() {
    use super::super::{
        super::super::PodTopology, super::TPUConfig, super::TPUVersion, BackendConfig,
    };

    let tpu_config = TPUConfig {
        tpu_version: TPUVersion::V4,
        num_cores: 8,
        enable_xla: true,
        xla_optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
        mixed_precision: true,
        batch_size_per_core: 32,
        enable_pod_coordination: false,
        pod_topology: PodTopology::Pod2x2,
        memory_optimization: crate::main_types::TPUMemoryOptimization::Balanced,
        gradient_compression: true,
        prefetch_depth: 2,
        experimental_features: false,
    };
    let backend_config = BackendConfig {
        target_tpu: tpu_config,
        enable_optimized_codegen: true,
        enable_profiling: true,
        debug_mode: false,
        verification_mode: false,
        custom_options: std::collections::HashMap::new(),
    };

    let mut profiling: ProfilingIntegration<f32> = ProfilingIntegration::new(&backend_config);
    assert_eq!(profiling.profiling_stats.samples_collected, 0);

    profiling.record_compile_timings(Duration::from_micros(4200), Duration::from_micros(800));

    assert_eq!(profiling.profiling_stats.samples_collected, 2);
    assert_eq!(profiling.profiling_stats.trace_events, 2);

    let data = profiling
        .counter_manager
        .counter_data
        .read()
        .expect("lock is never poisoned in a single-threaded test");
    let codegen_series = data
        .get("codegen_time_us")
        .expect("record_compile_timings must have recorded a real codegen sample");
    assert_eq!(codegen_series.statistics.sample_count, 1);
    assert_eq!(codegen_series.statistics.average, 4200.0);
}
