// Regression tests for resource management (findings R1, R3-R8, CF1).

use super::*;

fn streaming_config() -> StreamingConfig {
    let mut config = StreamingConfig::default();
    // Sample on every call so the synchronous collection path is exercised.
    config.resource_config.monitoring_frequency = Duration::from_millis(0);
    config
}

fn usage(memory_mb: usize, total_mb: usize, cpu: Option<f64>) -> ResourceUsage {
    ResourceUsage {
        memory_usage_mb: memory_mb,
        total_memory_mb: total_mb,
        cpu_usage_percent: cpu.unwrap_or(0.0),
        cpu_usage_percent_valid: cpu.is_some(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// R1: CPU / network / disk were the literals 50.0 / 1.0 / 5.0.
// ---------------------------------------------------------------------------

#[test]
fn the_probe_measures_memory_and_never_fabricates_a_rate() {
    let mut probe = SystemProbe::new();
    let first = probe.sample();

    assert!(
        first.total_memory_mb > 0,
        "the probe must read a real total memory figure"
    );
    assert_ne!(
        first.cpu_usage_percent, 50.0,
        "R1 regression: CPU usage is the old hardcoded 50.0"
    );
    if !first.cpu_usage_percent_valid {
        assert_eq!(
            first.cpu_usage_percent, 0.0,
            "an unmeasured CPU reading must be an explicit zero paired with a false validity flag"
        );
        assert_eq!(first.cpu_usage(), None);
    }
    assert!(first.active_threads >= 1);

    // A second sample taken after the minimum CPU interval must produce a real
    // CPU measurement rather than repeating a placeholder.
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL + Duration::from_millis(50));
    let second = probe.sample();
    assert!(
        second.cpu_usage_percent_valid,
        "R1 regression: after two refreshes the probe must report a real CPU measurement"
    );
    assert!((0.0..=100.0).contains(&second.cpu_usage_percent));
    // Rates are deltas, so they must now exist (possibly zero traffic, but a
    // measured zero rather than an invented 1.0 / 5.0).
    let network = second
        .network_io_mbps
        .expect("two interface refreshes must yield a rate");
    assert!(network >= 0.0 && network.is_finite());
    assert_ne!(
        second.network_io_mbps,
        Some(1.0),
        "R1 regression: network rate is the old hardcoded 1.0"
    );
    assert_ne!(
        second.disk_io_mbps,
        Some(5.0),
        "R1 regression: disk rate is the old hardcoded 5.0"
    );
}

#[test]
fn update_utilization_collects_a_sample_without_a_monitor_thread() {
    let mut manager = ResourceManager::new(&streaming_config()).expect("manager");
    assert!(!manager.is_monitoring());
    let before = manager.current_usage().expect("usage");
    assert_eq!(
        before.total_memory_mb, 0,
        "the manager starts from the all-zero default"
    );

    manager.update_utilization().expect("update");

    let after = manager.current_usage().expect("usage");
    assert!(
        after.total_memory_mb > 0,
        "update_utilization must collect a real sample when no monitor thread is running \
         (start_monitoring is never called by any caller in this crate)"
    );
    assert_eq!(manager.get_usage_history(10).len(), 1);
}

// ---------------------------------------------------------------------------
// R3: the trend window was built newest-first, inverting every direction.
// ---------------------------------------------------------------------------

#[test]
fn rising_memory_usage_is_reported_as_increasing() {
    let mut predictor = ResourcePredictor::new(true);
    for step in 0..10 {
        predictor
            .update(&usage(100 + step * 50, 32768, Some(10.0)))
            .expect("update");
    }
    assert_eq!(
        predictor.trend_analysis().memory_trend,
        TrendDirection::Increasing,
        "R3 regression: a monotonically rising series was classified as {:?}",
        predictor.trend_analysis().memory_trend
    );
}

#[test]
fn falling_memory_usage_is_reported_as_decreasing() {
    let mut predictor = ResourcePredictor::new(true);
    for step in 0..10 {
        predictor
            .update(&usage(1000 - step * 50, 32768, Some(10.0)))
            .expect("update");
    }
    assert_eq!(
        predictor.trend_analysis().memory_trend,
        TrendDirection::Decreasing,
        "R3 regression: a monotonically falling series was classified as {:?}",
        predictor.trend_analysis().memory_trend
    );
}

#[test]
fn prediction_extrapolates_the_trend_and_tracks_its_own_error() {
    let mut predictor = ResourcePredictor::new(true);
    predictor.prediction_horizon = 1;
    for step in 0..12 {
        predictor
            .update(&usage(100 + step * 10, 32768, Some(20.0)))
            .expect("update");
    }
    let predicted = predictor
        .predict()
        .expect("R7 regression: no prediction was produced at all");
    assert!(
        predicted.memory_usage_mb > 100 + 11 * 10,
        "a rising series must be extrapolated upwards (got {} MB)",
        predicted.memory_usage_mb
    );
    assert!(
        predictor.accuracy().contains_key("memory"),
        "R7 regression: prediction accuracy was never measured"
    );
}

#[test]
fn prediction_is_disabled_when_the_configuration_says_so() {
    let mut predictor = ResourcePredictor::new(false);
    for step in 0..12 {
        predictor
            .update(&usage(100 + step * 10, 32768, Some(20.0)))
            .expect("update");
    }
    assert!(
        predictor.predict().is_none(),
        "CF1: enable_resource_prediction=false must switch prediction off"
    );
    assert!(predictor.usage_patterns.is_empty());
}

// ---------------------------------------------------------------------------
// R4: the monitoring thread had no shutdown path.
// ---------------------------------------------------------------------------

#[test]
fn monitoring_can_be_started_and_stopped() {
    let mut config = streaming_config();
    // A long sampling interval: shutdown must not wait for it.
    config.resource_config.monitoring_frequency = Duration::from_secs(3600);
    let mut manager = ResourceManager::new(&config).expect("manager");

    manager.start_monitoring().expect("start");
    assert!(manager.is_monitoring());

    let started = Instant::now();
    manager.stop_monitoring().expect("stop");
    assert!(!manager.is_monitoring());
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "R4 regression: stopping took {:?}, so shutdown is waiting on the sampling interval \
         instead of a short poll tick",
        started.elapsed()
    );
}

#[test]
fn dropping_the_manager_stops_its_monitor_thread() {
    let mut config = streaming_config();
    config.resource_config.monitoring_frequency = Duration::from_secs(3600);
    let before = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    {
        let mut manager = ResourceManager::new(&config).expect("manager");
        manager.start_monitoring().expect("start");
        // Dropping here must join the thread rather than leaking it.
    }
    // The assertion below is about not hanging: if `Drop` did not signal the
    // thread, the join inside it would block forever and this test would time
    // out instead of completing.
    assert!(before >= 1);
}

// ---------------------------------------------------------------------------
// R5 / R6: active alerts were never cleared and ids collided.
// ---------------------------------------------------------------------------

#[test]
fn alerts_are_raised_once_and_cleared_on_recovery() {
    let mut system = ResourceAlertSystem::new();
    let emergency = system.thresholds.memory_thresholds.emergency;
    let recovery = system.thresholds.memory_thresholds.recovery;
    let total = 1000usize;
    let breaching = ((emergency + 1.0) / 100.0 * total as f64) as usize;
    let recovered = ((recovery - 5.0).max(0.0) / 100.0 * total as f64) as usize;

    system
        .update(&usage(breaching, total, None))
        .expect("update");
    assert_eq!(system.active_alerts.len(), 1);
    let first_id = system.active_alerts[0].id.clone();

    // Still breaching: the same alert is updated, not duplicated.
    system
        .update(&usage(breaching, total, None))
        .expect("update");
    assert_eq!(
        system.active_alerts.len(),
        1,
        "R5 regression: a persistent breach pushed a second alert"
    );
    assert_eq!(system.active_alerts[0].id, first_id);
    assert!(system.active_alerts[0].auto_resolution_attempts >= 1);

    // Recovered: the alert must be cleared and archived.
    system
        .update(&usage(recovered, total, None))
        .expect("update");
    assert!(
        system.active_alerts.is_empty(),
        "R5 regression: a recovered resource kept its alert active forever"
    );
    assert_eq!(system.alert_history.len(), 1);

    // Breaching again must produce a *different* id.
    system
        .update(&usage(breaching, total, None))
        .expect("update");
    assert_eq!(system.active_alerts.len(), 1);
    assert_ne!(
        system.active_alerts[0].id, first_id,
        "R6 regression: a re-raised alert reused the previous id"
    );
}

#[test]
fn alert_ids_are_unique_across_many_firings() {
    let mut system = ResourceAlertSystem::new();
    let total = 1000usize;
    let breaching =
        ((system.thresholds.memory_thresholds.emergency + 1.0) / 100.0 * total as f64) as usize;
    let mut ids = std::collections::HashSet::new();
    for _ in 0..50 {
        let alerts = system
            .check_thresholds(&usage(breaching, total, None))
            .expect("check");
        for alert in alerts {
            assert!(
                ids.insert(alert.id.clone()),
                "R6 regression: duplicate alert id {}",
                alert.id
            );
        }
    }
    assert_eq!(ids.len(), 50);
}

#[test]
fn an_unmeasured_cpu_reading_never_raises_or_clears_a_cpu_alert() {
    let mut system = ResourceAlertSystem::new();
    // 0.0 with a false validity flag must be treated as "unknown", not "idle".
    system.update(&usage(1, 32768, None)).expect("update");
    assert!(system
        .active_alerts
        .iter()
        .all(|alert| alert.resource_type != "cpu"));
    let alerts = system
        .check_thresholds(&usage(1, 32768, None))
        .expect("check");
    assert!(alerts.iter().all(|alert| alert.resource_type != "cpu"));
}

// ---------------------------------------------------------------------------
// R7: budget violations were reported as a hardcoded zero.
// ---------------------------------------------------------------------------

#[test]
fn budget_violations_are_counted_for_real() {
    let mut config = streaming_config();
    config.resource_config.max_memory_mb = 64;
    config.resource_config.budget_constraints.memory_budget_mb = 32;
    let mut manager = ResourceManager::new(&config).expect("manager");

    assert_eq!(manager.get_diagnostics().budget_violations, 0);
    let result = manager.allocate_resources("hog", 4096, 10.0, ResourcePriority::Normal);
    assert!(
        result.is_err(),
        "a 4 GB request against a 64 MB budget must fail"
    );
    assert_eq!(
        manager.get_diagnostics().budget_violations,
        1,
        "R7 regression: the rejected allocation was not counted as a violation"
    );
}

#[test]
fn strict_enforcement_removes_the_budget_head_room() {
    let mut flexible = StreamingConfig::default();
    flexible.resource_config.max_memory_mb = 100;
    flexible
        .resource_config
        .budget_constraints
        .strict_enforcement = false;
    let mut flexible_manager = ResourceManager::new(&flexible).expect("manager");

    let mut strict = StreamingConfig::default();
    strict.resource_config.max_memory_mb = 100;
    strict.resource_config.budget_constraints.strict_enforcement = true;
    let mut strict_manager = ResourceManager::new(&strict).expect("manager");

    // 110 MB is inside the 20% flexibility band but outside a strict budget.
    assert!(flexible_manager
        .allocate_resources("component", 110, 1.0, ResourcePriority::Normal)
        .is_ok());
    assert!(
        strict_manager
            .allocate_resources("component", 110, 1.0, ResourcePriority::Normal)
            .is_err(),
        "CF1: strict_enforcement must remove the budget head-room"
    );
}

#[test]
fn violation_penalty_accumulates_from_the_configured_value() {
    let mut config = streaming_config();
    config.resource_config.max_memory_mb = 1; // guaranteed to be exceeded
    config.resource_config.budget_constraints.violation_penalty = 0.25;
    let mut manager = ResourceManager::new(&config).expect("manager");

    manager.update_utilization().expect("update");
    assert!(
        manager.budget_penalty() >= 0.25,
        "CF1: violation_penalty must feed the accumulated penalty (got {})",
        manager.budget_penalty()
    );
}

#[test]
fn the_optimizer_clamps_and_paces_changes() {
    let mut optimizer = ResourceOptimizer::new(ResourceOptimizationStrategy::Balanced);
    let limit = optimizer
        .constraints
        .change_rate_limits
        .get("memory")
        .copied()
        .expect("balanced strategy configures a memory change-rate limit");

    let clamped = optimizer.clamp_change("memory", -5.0);
    assert!(
        (clamped + limit).abs() < 1e-12,
        "R7 regression: a -500% request was not clamped to the configured limit (got {clamped})"
    );

    assert!(optimizer.accept_change("memory_manager"));
    optimizer.record_applied_change("memory_manager");
    // Immediately reversing direction inside the stable period is exactly the
    // oscillation `prevent_oscillation` exists to stop.
    optimizer.clamp_change("memory", 0.2);
    assert!(
        !optimizer.accept_change("memory_manager"),
        "R7 regression: an immediate sign reversal was accepted"
    );
    assert!(optimizer
        .performance_impact()
        .contains_key("memory_manager"));
}

#[test]
fn a_change_below_the_hysteresis_band_is_rejected() {
    let mut optimizer = ResourceOptimizer::new(ResourceOptimizationStrategy::Balanced);
    optimizer.clamp_change("memory", -0.0001);
    assert!(
        !optimizer.accept_change("memory_manager"),
        "R7 regression: a change far inside the hysteresis band was accepted"
    );
}

// ---------------------------------------------------------------------------
// R8 / CF1: thresholds and budgets now come from the configuration.
// ---------------------------------------------------------------------------

#[test]
fn alert_thresholds_follow_the_configured_limits() {
    let tight = ResourceConfig {
        max_cpu_percent: 40.0,
        cleanup_threshold: 0.5,
        budget_constraints: ResourceBudgetConstraints {
            cpu_budget_percent: 20.0,
            ..ResourceBudgetConstraints::default()
        },
        ..ResourceConfig::default()
    };
    let relaxed = ResourceConfig {
        max_cpu_percent: 95.0,
        cleanup_threshold: 0.95,
        budget_constraints: ResourceBudgetConstraints {
            cpu_budget_percent: 80.0,
            ..ResourceBudgetConstraints::default()
        },
        ..ResourceConfig::default()
    };

    let tight_config = StreamingConfig {
        resource_config: tight,
        ..StreamingConfig::default()
    };
    let relaxed_config = StreamingConfig {
        resource_config: relaxed,
        ..StreamingConfig::default()
    };
    let tight_manager = ResourceManager::new(&tight_config).expect("manager");
    let relaxed_manager = ResourceManager::new(&relaxed_config).expect("manager");

    let tight_memory = tight_manager
        .alert_system
        .thresholds
        .memory_thresholds
        .emergency;
    let relaxed_memory = relaxed_manager
        .alert_system
        .thresholds
        .memory_thresholds
        .emergency;
    assert!(
        tight_memory < relaxed_memory,
        "R8 regression: memory thresholds ignore cleanup_threshold ({tight_memory} vs \
         {relaxed_memory})"
    );

    let tight_cpu = tight_manager.alert_system.thresholds.cpu_thresholds.warning;
    let relaxed_cpu = relaxed_manager
        .alert_system
        .thresholds
        .cpu_thresholds
        .warning;
    assert!(
        tight_cpu < relaxed_cpu,
        "R8 regression: CPU thresholds ignore the configured budget ({tight_cpu} vs {relaxed_cpu})"
    );
}

#[test]
fn budget_constraints_drive_the_derived_budget() {
    let mut config = StreamingConfig::default();
    config.resource_config.max_memory_mb = 8192;
    config.resource_config.max_cpu_percent = 90.0;
    config.resource_config.budget_constraints.memory_budget_mb = 512;
    config.resource_config.budget_constraints.cpu_budget_percent = 25.0;
    config.resource_config.budget_constraints.time_budget = Duration::from_secs(7);

    let manager = ResourceManager::new(&config).expect("manager");
    assert_eq!(
        manager.budget.memory_budget.soft_limit_mb, 512,
        "CF1: memory_budget_mb must bound the soft limit"
    );
    assert!(
        (manager.budget.cpu_budget.target_utilization - 25.0).abs() < 1e-12,
        "CF1: cpu_budget_percent must set the CPU target"
    );
    assert_eq!(
        manager.budget.time_budget.target_batch_processing_time,
        Duration::from_secs(7),
        "CF1: time_budget must set the batch processing target"
    );
    assert_eq!(
        manager.budget.time_budget.operation_timeout,
        Duration::from_secs(7)
    );
}

#[test]
fn a_single_core_machine_does_not_underflow_the_thread_split() {
    // `num_cpus::get() - 2` panicked in debug builds on 1- and 2-core hosts.
    let manager = ResourceManager::new(&StreamingConfig::default()).expect("manager");
    let priority = &manager.budget.cpu_budget.thread_priority;
    assert!(priority.high_priority_threads <= manager.budget.cpu_budget.max_threads);
    assert_eq!(
        priority.high_priority_threads + priority.normal_priority_threads,
        manager.budget.cpu_budget.max_threads
    );
}

#[test]
fn an_unknown_adaptation_target_is_reported() {
    let mut manager = ResourceManager::new(&streaming_config()).expect("manager");
    let adaptation = Adaptation {
        adaptation_type: AdaptationType::ResourceAllocation,
        magnitude: -0.1,
        target_component: "nonexistent_manager".to_string(),
        parameters: std::collections::HashMap::new(),
        priority: AdaptationPriority::High,
        timestamp: Instant::now(),
    };
    assert!(
        manager.apply_allocation_adaptation(&adaptation).is_err(),
        "an adaptation aimed at an unknown component must be reported, not silently dropped"
    );
}

/// The allocation budget is a *per-process* figure, so it must be compared
/// against this process's own footprint. Activating the collection path (see
/// `update_utilization_collects_a_sample_without_a_monitor_thread`) made that
/// comparison live for the first time: with `System::used_memory()` (which is
/// system-wide) on the left-hand side, a single benign call against the default
/// configuration reported ~670% utilization, `has_sufficient_resources ==
/// false` and an immediate budget violation on this machine.
#[test]
fn a_default_configuration_reports_no_budget_violation_after_one_sample() {
    let mut manager = ResourceManager::new(&StreamingConfig::default()).expect("manager");
    manager.update_utilization().expect("update");

    let diagnostics = manager.get_diagnostics();
    assert!(
        diagnostics.system_memory_percent.is_some(),
        "the system-wide ratio must still be available for the alert path"
    );
    let utilization = diagnostics
        .memory_utilization
        .expect("this process's own footprint must be measurable");
    assert!(
        utilization < 100.0,
        "a freshly built manager must not already be over its own allocation budget \
         ({utilization:.1}% of {} MB) — this is what comparing system-wide usage against a \
         per-process budget produced",
        manager.budget.memory_budget.max_allocation_mb
    );
    assert_eq!(
        manager.budget_violations(),
        0,
        "a single benign sample must not count as a budget violation"
    );
    assert_eq!(manager.budget_penalty(), 0.0);
    assert!(
        manager
            .has_sufficient_resources_for_processing()
            .expect("sufficiency check"),
        "an idle process well inside its budget must be reported as having head-room"
    );
    assert!(
        manager
            .compute_allocation_adaptation()
            .expect("adaptation")
            .is_none(),
        "no adaptation is warranted when nothing is under pressure"
    );
}

/// A process genuinely over its allocation budget must still be caught.
#[test]
fn a_process_over_its_allocation_budget_is_still_detected() {
    let mut system = ResourceAlertSystem::new();
    let _ = &mut system; // the alert path is covered separately
    let mut manager = ResourceManager::new(&StreamingConfig::default()).expect("manager");
    let over_budget = ResourceUsage {
        memory_usage_mb: 1000,
        total_memory_mb: 32768,
        process_memory_mb: Some(manager.budget.memory_budget.max_allocation_mb + 1),
        ..Default::default()
    };
    {
        let mut current = lock_recovered(&manager.current_usage);
        *current = over_budget;
    }
    // Suppress the synchronous collection so the injected sample survives.
    manager.last_synchronous_sample = Some(Instant::now());
    manager.config.monitoring_frequency = Duration::from_secs(3600);

    manager.update_utilization().expect("update");
    assert_eq!(
        manager.budget_violations(),
        1,
        "a process over its own allocation budget must be counted"
    );
    assert!(!manager
        .has_sufficient_resources_for_processing()
        .expect("sufficiency check"));
    assert!(manager
        .compute_allocation_adaptation()
        .expect("adaptation")
        .is_some());
}
